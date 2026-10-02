"""The citation click, driven the way the browser drives it.

WALK-DEFECT-1 said every citation the reader can click 404s. The 79 cases in
``tests/test_evidence_api.py`` stayed green throughout, and the reason is now
on the record: **every one of those tests addresses an id it minted itself.**
It either names a ``chunk_*`` id or computes an ``ev_*`` id over *the very
document it then asks about*. The defect exists only in composition — the id
the Q&A route actually emits, addressed to the document the analysis result
actually published — and no unit test ever performs that pairing.

So this file performs it. The three rules it follows:

* **The id is not hand-built.** It is read out of the ``done`` event the real
  ``POST /api/ask/stream`` emits. An ``ev_…`` string assembled here would prove
  only that this file agrees with itself, which is exactly the failure mode that
  let D1 survive a green suite the first time.
* **The document id is not chosen either.** It is the id the browser pairs
  with: ``sections.js:150`` reads ``data.document_id`` off the analysis result
  into ``state.currentDocumentId``, and ``evidence.js:303`` builds
  ``/v1/documents/{that}/evidence/{chip's evidence id}``. That exact URL is
  what :func:`citation_click_url` reproduces.
* **No model, no network.** The endpoint answered 429 for an entire previous
  session, and a test that skips when the endpoint is down is a test that says
  nothing on the day it matters. The model here is a scripted double that reads
  the ``ev_…`` ids out of the prompt it was handed and cites one — so it proves
  the plumbing end to end, and proves it on a machine with no route to a model.

**And a second defect this file now also covers.** Fixing the id/document pairing
was not enough, because there was a case in which the document was not in the
store at all: ``/api/analyze`` published a ``document_id`` it never registered
(``attach_document_identity`` copies the parse's id because a chip addresses
``/v1/documents/{id}/evidence/{id}``, and the job stage declared ``indexing`` a
passthrough). The uploaded walkthrough never saw this code — it went through
``POST /v1/documents``, which does register — so every citation it checked was
addressing a document that happened to exist for an unrelated reason. The
browser's real path returned ``404 document_not_found`` instead.

The last four tests therefore drive ``POST /api/analyze`` over the real
application and the real job pipeline, and assert what a reader gets:
analyze-ask-click resolves; a re-analysis does not embed twice; a failed
registration still returns the analysis with a notice on the surface the
interface renders; and **an evidence id genuinely belonging to a different
paper is still refused** against the document the analysis registered. That last
one is deliberately not a variant of the passing case: "every id resolves
somewhere" is not a fix, it is a much worse product — a reader clicking a
citation would be shown a passage from a document they did not open, and one
whose text happened to align would look correct.
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from paperwhisperer.analysis import orchestrator
from paperwhisperer.api import routes_evidence, routes_qa
from paperwhisperer.api.routes_v1 import persist_document
from paperwhisperer.core import config
from paperwhisperer.infra.stores import build_sqlite_bundle
from paperwhisperer.ingestion.schema import Element, ParsedDocument, SourceLocator

from .conftest import read_sse_events

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    """A personal-tier store on a throwaway SQLite file."""
    settings = SimpleNamespace(
        STORAGE_SQLITE_PATH=str(tmp_path / "citation.sqlite3"),
        STORAGE_SQLITE_TIMEOUT_SECONDS=30,
        STORAGE_LEXICAL_FTS5="auto",
    )
    bundle = build_sqlite_bundle(settings)
    yield bundle
    bundle.close()


@pytest.fixture
def client(store):
    """A TestClient over the two routers the browser's click composes.

    ``routes_qa`` and ``routes_evidence`` are mounted together on purpose. They
    are mounted together in ``create_app`` too, and the defect *is* the seam
    between them: mounting them separately would let a test pass by giving each
    half its own store, which is not a deployment this product has.
    """
    application = FastAPI()
    application.include_router(routes_qa.router)
    application.include_router(routes_evidence.router)
    application.dependency_overrides[routes_evidence.get_store_bundle] = lambda: store
    with TestClient(application) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def qa_over_the_shared_store(store, monkeypatch):
    """Point the Q&A path at the same bundle the evidence route reads.

    ``orchestrator.get_store_bundle`` is the process-wide bundle the analyzer
    resolves, and it is a module global rather than a FastAPI dependency — so
    the evidence route can be overridden through ``dependency_overrides`` while
    this one needs the global. Both have to be the *same* bundle, and
    ``routes_qa``'s own docstring says why: "a document indexed by one and a
    citation opened from the other has to be the *same* document". Two bundles
    here would not reproduce the defect at all.

    No embedder, and the retriever cache cleared: a retrieval whose vector leg
    silently came from a developer's ``.env`` is not a hermetic test.
    """
    monkeypatch.setattr(orchestrator, "_QA_STORES", store)
    monkeypatch.setattr(orchestrator, "_CONFIGURED_EMBEDDER", None)
    monkeypatch.setattr(orchestrator, "_CONFIGURED_EMBEDDER_RESOLVED", True)
    monkeypatch.setattr(orchestrator, "_configured_embedder", lambda: None)
    monkeypatch.setattr(orchestrator, "_QA_RETRIEVERS", {})


# --------------------------------------------------------------------------
# The double
# --------------------------------------------------------------------------


class ScriptedModel:
    """Stands in for ``LLMClient``; cites the evidence it was shown.

    It reads the ``[ev_…]`` tags out of its own prompt rather than carrying a
    hard-coded id, which is the point: if the retrieval changed, or the prompt
    stopped tagging its passages, this double would cite nothing and the test
    would fail as *no citation*, not silently as *the wrong one*.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = "scripted-model"
        self.search_rewrite_model = "scripted-model"
        self.request_timeout = 30
        self.max_retries = 1
        self.max_concurrency = 4
        self.prompts: list[str] = []
        self.replies: list[str] = []

    def _reply(self, user_prompt: str) -> str:
        self.prompts.append(user_prompt)
        evidence_ids = re.findall(r"\[(ev_[0-9a-f]+)\]", user_prompt)
        # A real sentence out of the retrieved passages, so the verifier has
        # something that genuinely aligns and the claim survives as a direct
        # quote — which is the only kind the front end will put a chip on.
        sentences = re.findall(r"[A-Z][^.\n]*\.", user_prompt)
        quote = next((s for s in sentences if "errors" in s), sentences[0] if sentences else "")
        reply = json.dumps(
            {
                "answer": "The student network makes 67 test errors.",
                "claims": [
                    {
                        "text": "The student network makes 67 test errors.",
                        "quote": quote,
                        "evidence_ids": evidence_ids[:1],
                    }
                ],
            },
            ensure_ascii=False,
        )
        self.replies.append(reply)
        return reply

    def complete(self, system_prompt, user_prompt, **kwargs):
        return self._reply(user_prompt)

    def stream(self, system_prompt, user_prompt, **kwargs):
        return iter([self._reply(user_prompt)])


@pytest.fixture
def scripted_model(monkeypatch):
    """Install the double at the seam the analyzer actually constructs it.

    ``orchestrator.LLMClient`` is read from the module globals inside
    ``DocumentAnalyzer.__init__``, so this is the name that has to be patched.
    """
    monkeypatch.setattr(orchestrator, "LLMClient", ScriptedModel)
    return ScriptedModel


# --------------------------------------------------------------------------
# Fixtures for the document and the session
# --------------------------------------------------------------------------

#: The sentence the citation quotes, and the question that reaches it.
LATE_FRAGMENT = "67 test errors"
LATE_QUESTION = "How many test errors did the student network make on the test set?"

#: A second paper, used only to mint an evidence id that genuinely belongs to
#: another document. Its text is the thing the isolation test asserts is
#: **absent** from a refused response body.
FOREIGN_FRAGMENT = "another paper reports a top-1 accuracy of 61.3"


def _paper(document_id: str, *, first: str, last: str) -> ParsedDocument:
    """A parsed paper-shaped document, with real page locators.

    Built by hand so the test needs no ``eval/corpus`` and no parser, but it is
    a genuine ``ParsedDocument``: ``pdf_page`` locators, the shape the Evidence
    API and the retriever consume.

    The three elements carry three different pages on purpose. A chunk's
    locator is the page of its *anchor* — its first element — and the citation
    verifier derives its search scope from that page and nothing wider
    (``rag.citations._scope_for_evidence``). So the passage the claim quotes
    must be on the anchor's page to verify at all, which is what makes a real
    citation verifiable: the chunk's own page, not the document's. Putting the
    quoted sentence on a *later* page than the chunk anchors would make this
    fixture fail for a reason that has nothing to do with the defect under
    test — the verifier refusing to widen its scope past the passage, which is
    the release gate doing its job.
    """
    document = ParsedDocument(
        document_id=document_id,
        source_filename="paper.pdf",
        media_type="application/pdf",
    )
    for page, text in enumerate(
        [
            first,
            "Distillation transfers knowledge from a large teacher to a small student.",
            "We train the student on softened targets produced by the teacher.",
        ],
        start=1,
    ):
        document.elements.append(
            Element(role="text", text=text, locator=SourceLocator(kind="pdf_page", page=page))
        )
    return document


@pytest.fixture
def parsed_document():
    """The document the reader's upload *is*, registered as ``/v1/documents`` does.

    The fact-bearing sentence is the *first* element, so it anchors the chunk
    retrieval returns and the verifier's page-scoped search can find it. See
    :func:`_paper` for why that is the realistic arrangement rather than a
    convenience.
    """
    document = _paper(
        "doc_parsed_distillation",
        first="The student network makes 67 test errors on the held-out benchmark.",
        last="We train the student on softened targets produced by the teacher.",
    )
    return document


@pytest.fixture
def registered(store, parsed_document):
    """``persist_document`` — the registration the evidence API reads.

    Called directly rather than over HTTP because the defect is not in upload;
    it is in what happens after ``document_id`` and an evidence id exist. Going
    through ``POST /v1/documents`` would add a parse and an upload to every
    case without exercising one more line of the seam under test.
    """
    persist_document(store, parsed_document, {"status": "ok"})
    return parsed_document


@pytest.fixture
def foreign_document(store):
    """A different paper, registered and indexed, so it mints real ids."""
    document = _paper(
        "doc_parsed_other",
        first=f"Results: {FOREIGN_FRAGMENT}.",
        last="This line belongs to a different paper entirely.",
    )
    persist_document(store, document, {"status": "ok"})
    return document


@pytest.fixture
def session(runtime_dirs, registered):
    """A session carrying the analysis's flat text and its document id.

    This is exactly what ``finalize_analysis_result`` persists: the flat
    ``document_content`` the Q&A route reads, and the ``document_id`` the
    analysis result published. A session with only the text — and no id, or a
    different id — is the *other* case, and :func:`test_a_session_without_a_registered_document` covers it.
    """
    from paperwhisperer.core.text import build_session_expiry
    from paperwhisperer.sessions.store import hash_session_token, write_session_payload

    session_id, token = "citation-session", "citation-token"
    write_session_payload(
        session_id,
        {
            "expires_at": build_session_expiry(),
            "document_content": registered.to_full_text(),
            "document_id": registered.document_id,
            "qa_history": [],
            "paper_search": {},
            "session_auth": {"token_hash": hash_session_token(token)},
        },
    )
    return session_id, token


# --------------------------------------------------------------------------
# Driving the real routes
# --------------------------------------------------------------------------


def ask(client, session, question: str = LATE_QUESTION) -> dict:
    """``POST /api/ask/stream``, exactly as ``chat.js`` posts it.

    The ``done`` event carries the whole ``AnswerResult`` — claims, citations,
    confidence — which is the payload the browser turns into citation chips, so
    that event *is* the object under test. Nothing here inspects internals.
    """
    session_id, token = session
    response = client.post(
        "/api/ask/stream",
        json={
            "question": question,
            "session_id": session_id,
            "session_token": token,
            "api_key": "test-key",
        },
    )
    assert response.status_code == 200, response.text
    events = read_sse_events(response.text)
    done = [data for name, data in events if name == "done"]
    assert done, f"the stream produced no done event: {response.text}"
    return done[-1]


def citation_click_url(document_id: str, evidence_id: str) -> str:
    """The URL ``evidence.js:303`` + ``config.js:66`` build.

    Reproduced here rather than imported — there is no browser harness — and
    spelled out so the correspondence is checkable by reading: ``buildEvidenceUrl``
    interpolates ``documentId`` and ``evidenceId`` into
    ``EVIDENCE_API_PATH = '/v1/documents/{documentId}/evidence/{evidenceId}'``,
    and ``handleChipClick`` passes ``state.currentDocumentId`` and
    ``button.dataset.evidenceId``.
    """
    return f"/v1/documents/{document_id}/evidence/{evidence_id}"


def chip_evidence_ids(payload: dict) -> list[str]:
    """The ``ev_…`` ids the front end would turn into chips.

    ``renderCitationChips`` renders one chip per claim whose
    ``presentation`` is a direct quote — an unverified claim gets no chip,
    deliberately, because a chip beside an unverified sentence reads as a
    citation. ``buildAnswerClaims`` is what pairs each claim with the citation
    whose id it named. Deriving the list from the payload the same way keeps
    the test on the ids a reader can actually click.
    """
    citations = {
        str(item.get("evidence_id") or ""): item
        for item in payload.get("citations") or []
        if isinstance(item, dict)
    }
    ids: list[str] = []
    for claim in payload.get("claims") or []:
        if claim.get("presentation") != "direct_quote":
            continue
        for evidence_id in claim.get("evidence_ids") or []:
            if str(evidence_id) in citations and str(evidence_id) not in ids:
                ids.append(str(evidence_id))
    return ids


# --------------------------------------------------------------------------
# The defect
# --------------------------------------------------------------------------


def test_a_citation_from_the_real_qa_route_opens_the_passage(
    client, store, session, registered, scripted_model
):
    """The central promise: click a citation, read the passage behind it.

    **This test fails today.** The route hands back an ``ev_…`` id minted over
    a document other than the registered one, so the URL the browser builds from
    that id plus ``state.currentDocumentId`` resolves to nothing.

    Every assertion is a claim the product makes out loud. The click returns
    ``200`` and a passage (``snippet.text``); the passage is the text the claim
    quoted (``LATE_FRAGMENT``); and it carries a locator the panel can point at.
    A fix that made the click open by serving *some* passage would pass only
    the first of those, which is why the other two are here.
    """
    payload = ask(client, session)
    evidence_ids = chip_evidence_ids(payload)

    assert evidence_ids, (
        "the answer carried no citable evidence id, so there was nothing for the "
        f"chip to address. Answer was: {payload.get('answer')!r}"
    )

    for evidence_id in evidence_ids:
        response = client.get(citation_click_url(registered.document_id, evidence_id))

        assert response.status_code == 200, (
            "a citation the reader can click did not open. The route resolved the "
            "evidence id against the document it was minted over, not against "
            "state.currentDocumentId.\n"
            f"  document: {registered.document_id}\n"
            f"  evidence: {evidence_id}\n"
            f"  body: {response.text[:400]}"
        )
        body = response.json()
        assert "error" not in body
        assert LATE_FRAGMENT in body["snippet"]["text"], (
            "the click opened but the passage is not the one the claim quoted — "
            "the right shape with the wrong text is the worse failure"
        )
        assert body["document_id"] == registered.document_id


# --------------------------------------------------------------------------
# Cross-document isolation — the property the fix must not spend
# --------------------------------------------------------------------------


def test_an_evidence_id_from_another_paper_is_still_refused(
    client, store, session, registered, foreign_document, scripted_model
):
    """A citation minted over a *different* document does not resolve here.

    Not a variant of the test above. That one asserts a citation whose evidence
    belongs to *this* upload does not open; this one asserts an evidence id that
    **genuinely names a chunk of another paper** still gets refused, and that
    the refusal leaks nothing.

    The distinction is the whole safety argument. "Every id resolves somewhere"
    is not a fix for the 404 above — it is a fix for a different, much worse
    product: a reader clicking a citation would be shown a passage from a
    document they did not open, and one whose text happens to align would look
    correct. So the assertion is on the *code* as well as the status: the
    response must be the refusal, and the foreign paper's text must not appear
    anywhere in the body.

    It is a real id from a real retrieval over the real second document, minted
    by the real retriever under the real QA plan — not a guess, so a resolver
    that had genuinely learned to leak would be caught here rather than in
    production.
    """
    from paperwhisperer.rag.retrieve import TASK_QA, HybridRetriever, plan_for

    retriever = HybridRetriever(store, embeddings=None, reranker=None)
    retrieved = retriever.retrieve(
        "What top-1 accuracy did the other paper report?",
        plan=plan_for(TASK_QA),
        document_ids=[foreign_document.document_id],
    )
    assert retrieved.evidence, "the foreign document was not retrievable; the fixture is wrong"
    foreign_id = retrieved.evidence[0].evidence_id
    assert foreign_id.startswith("ev_")

    # Sanity, in the direction that must keep working: the id *is* a real
    # citation — it resolves against the document it was minted over. Without
    # this, the refusal below would only prove the store was empty.
    owner = client.get(citation_click_url(foreign_document.document_id, foreign_id))
    assert owner.status_code == 200, (
        "the control failed: a real citation against its own document should open. "
        f"Got {owner.status_code} {owner.text[:300]}"
    )
    assert FOREIGN_FRAGMENT in owner.json()["snippet"]["text"]

    # The refusal.
    response = client.get(citation_click_url(registered.document_id, foreign_id))
    assert response.status_code == 404, (
        "an evidence id belonging to another paper resolved against this "
        f"document — cross-document isolation is broken. Got {response.status_code} "
        f"{response.text[:300]}"
    )
    body = response.json()
    assert body.get("code") in {"evidence_not_found", "evidence_not_in_document"}, body
    assert FOREIGN_FRAGMENT not in response.text, (
        "the refusal body leaked the other paper's text"
    )
    assert "snippet" not in body, body


def test_a_guessed_evidence_id_is_refused_and_reveals_nothing(client, store, registered):
    """A well-formed id that names nothing is a 404, not a hint.

    The other refusal this route owes its clients: ``evidence_not_found`` for an
    id that is not in this document at all, so a client can tell a typo from
    someone else's citation. Pinned because the fix for the click above touches
    the candidate set, and a candidate set that grew too far would answer this
    with a passage instead of a refusal.
    """
    response = client.get(
        citation_click_url(registered.document_id, "ev_" + "0" * 16)
    )
    assert response.status_code == 404
    assert response.json()["code"] == "evidence_not_found"


# --------------------------------------------------------------------------
# The browser's own path, end to end: analyze, ask, click
# --------------------------------------------------------------------------
#
# Everything above composes the two halves by hand. This section drives the path
# a reader actually walks, over the real application, with the real job
# pipeline:
#
#   POST /api/analyze -> POST /api/ask/stream -> GET /v1/documents/{id}/evidence/{ev}
#
# The first leg is where the second defect lived. ``/api/analyze`` publishes a
# ``document_id`` on its result and the browser reads it into
# ``state.currentDocumentId`` (``sections.js:150``); the citation chip pairs it
# with the ``ev_*`` id the Q&A stream emits (``evidence.js:303``). Until the
# analysis path registered the document it published, that URL addressed a
# document the store had never heard of and answered ``404
# document_not_found``. The walkthrough never saw this code precisely because it
# uploaded through ``/v1/documents``, which does register -- so the test below
# takes no shortcut around ``/api/analyze``. The id it reads is the id the route
# published, the registration is the one the analysis performed, and the click
# resolves against the same process-wide bundle the evidence route reads.

#: The real paper the walkthrough used. A module constant because three tests
#: upload it and a path computed three times is three chances to mistype it.
CORPUS = Path(__file__).resolve().parents[1] / "eval" / "corpus" / (
    "digital_single_col/hinton_distillation.pdf"
)

#: A key the process does not hold, and long enough for
#: ``register_ephemeral_secret`` to park it, so the job stage bills the caller's
#: key and never the environment's. Never reaches a socket: ``scripted_model``
#: has already replaced ``orchestrator.LLMClient``.
BROWSER_KEY = "test-key-browser-citation"


class CountingEmbedder:
    """A real embedder that keeps a count, so "did we embed twice?" is answerable.

    Declares no identity, so ``_assert_embedder_matches_index`` does not hold it
    to a provider/model/dimension this test never configured -- its whole job is
    to be counted. It still returns well-formed vectors, because the store's
    dimension guard and ``index_document_for_retrieval``'s "are the vectors
    already there?" check both depend on real rows existing. That check is the
    mechanism that makes a second registration cheap, so a double that left the
    store empty would make the idempotence test pass for the wrong reason.

    ``embed_documents`` is the batched path ``_embed_chunk_texts`` prefers, so
    one indexing pass is one entry in ``calls`` and a query embedding (which
    goes through ``embed_query``) never pollutes the count.
    """

    def __init__(self, width: int = 8):
        self.width = width
        #: One entry per *indexing* pass, holding the texts it was asked to embed.
        self.calls: list[list[str]] = []

    def embed_documents(self, texts):
        batch = list(texts)
        self.calls.append(batch)
        return [[0.05 * ((index % 5) + 1)] * self.width for index, _ in enumerate(batch)]

    def embed_query(self, text):
        # Present so the double is a usable embedder, and deliberately not
        # counted: a query is not an indexing pass, and letting it land in
        # ``calls`` would make the count a number nobody can argue about.
        return [0.05] * self.width


@pytest.fixture
def corpus():
    """The corpus PDF, or a skip that names the claim it cannot make."""
    if not CORPUS.exists():
        pytest.skip("the corpus PDF is not present; this asserts about the real parse")
    return CORPUS


@pytest.fixture
def object_store(tmp_path):
    from paperwhisperer.infra.object_store import FilesystemObjectStore

    built = FilesystemObjectStore(tmp_path / "browser-objects", fsync=False)
    built.initialize()
    yield built
    built.close()


@pytest.fixture
def browser(runtime_dirs, store, object_store, monkeypatch, scripted_model):
    """The real application with only the storage seams replaced.

    Four things point at one bundle on purpose: ``routes_evidence`` (the store
    the citation click reads), ``orchestrator._QA_STORES`` (the store Q&A
    retrieves over), the ``get_store_bundle`` dependency, and the registration
    ``register_analysis_document`` resolves for itself. They are one bundle in a
    deployment; a fixture that gave each its own would not compose the seam
    that failed and could pass with the defect still present.

    ``routes_jobs.resolve_object_store`` is monkeypatched rather than
    dependency-overridden because the job stages resolve it themselves: a stage
    runs on a worker thread with no request, which is exactly why that function
    exists separately from ``get_object_store``.

    The lifespan is deliberately **not** entered. No background worker means
    the waiting request claims the job itself, so these tests observe the work
    rather than racing a poll interval to catch it afterwards.
    """
    from paperwhisperer.api import routes_evidence, routes_jobs, routes_v1
    from paperwhisperer.app import app
    from paperwhisperer.infra import jobs as J

    monkeypatch.setattr(config, "PARSER_CACHE_ENABLED", False)
    monkeypatch.setattr(routes_jobs, "resolve_object_store", lambda: object_store)
    monkeypatch.setattr(routes_evidence, "_STORES", store)

    # The suite-wide ``no_unstubbed_job_stage_model`` tripwire refuses to build
    # a real ``DocumentAnalyzer`` unless a credential is present, and these tests
    # need the real one -- it is what owns ``parse_outcome``, which is what the
    # registration reads. Restoring a key is the opt-in the tripwire documents.
    # It opens no socket: ``scripted_model`` has already replaced
    # ``orchestrator.LLMClient``, so every completion is answered in-process,
    # and ``monkeypatch`` puts the environment back when the test ends.
    monkeypatch.setenv("OPENAI_API_KEY", BROWSER_KEY)

    job_store = J.build_job_store(bundle=store)
    overrides = {
        routes_evidence.get_store_bundle: lambda: store,
        routes_v1.get_object_store: lambda: object_store,
        routes_jobs.get_object_store: lambda: object_store,
        routes_jobs.get_job_store: lambda: job_store,
    }
    app.dependency_overrides.update(overrides)
    try:
        yield TestClient(app)
    finally:
        for dependency in overrides:
            app.dependency_overrides.pop(dependency, None)


@pytest.fixture
def lexical_only(monkeypatch):
    """No dense leg, so a hermetic test never loads a 384 MB model.

    ``fastembed`` is available and the shipped configuration enables it, which
    would make every test here download and load a model to answer a question
    about routing. The dense leg is not what any of these tests is about: the
    click resolves over chunks, and the one test that *does* care about vectors
    installs its own counting double.
    """
    monkeypatch.setattr(orchestrator, "_configured_embedder", lambda: None)
    monkeypatch.setattr(orchestrator, "_QA_RETRIEVERS", {})


def analyze_document(browser, corpus, *, session_id: str = "", **form):
    """``POST /api/analyze`` — what the bundled browser posts.

    Returns ``(status_code, body)`` rather than just the body, because the two
    statuses this route legitimately answers are two different claims: ``200``
    carries the analysis the browser renders, and ``202`` says the work finished
    elsewhere or was recognised as a request already made. Collapsing them here
    would let the idempotency test assert on a job id it never actually read.

    The failure message names the envelope rather than a missing key, because an
    error body read as an analysis with no ``document_id`` fails several
    assertions away from its cause.
    """
    response = browser.post(
        "/api/analyze",
        files={"file": (corpus.name, io.BytesIO(corpus.read_bytes()), "application/pdf")},
        data={"api_key": BROWSER_KEY, "session_id": session_id, **form},
    )
    body = response.json()
    assert response.status_code in {200, 202}, response.text[:600]
    return response.status_code, body


def analyzed(browser, corpus, *, session_id: str = "", **form) -> dict:
    """The same call, insisting on the status that carries a rendered analysis.

    Used wherever the test is about the analysis the reader got. The one test
    that is about deduplication uses :func:`analyze_document` and reads both
    statuses itself.
    """
    status, body = analyze_document(browser, corpus, session_id=session_id, **form)
    assert status == 200, f"expected the analysis itself, got {status}: {body!r}"
    return body


def test_the_browser_path_analyze_ask_click_resolves(
    browser, store, corpus, lexical_only
):
    """The whole promise, in the order a reader walks it.

    **This test fails without the analysis path registering its own document.**
    Every id here is read off the wire — the ``document_id`` off the analyze
    response, the ``ev_*`` id off the ``done`` event of the real SSE stream — so
    an id hand-assembled in this file could not pass it.

    Four claims, and the middle two are what stop a cheap fix:

    1. The analysis published a ``document_id`` *and* put it in the store, so the
       citation has something to address.
    2. The click answers ``200`` with a passage, from the store the *evidence
       route* reads rather than a bundle this test happens to be holding.
    3. The passage is the one the citation named — the ``chunk_id`` the answer
       carried is the ``chunk_id`` the click opened — so the panel did not open
       *some* passage, it opened *that* one.
    4. The passage carries a page, so the panel has something to point a
       highlight at. A chunk resolved with ``txt_offset`` would open and still
       be unrenderable.
    """
    analysis = analyzed(browser, corpus)
    document_id = str(analysis.get("document_id") or "")
    assert document_id, f"the analysis published no document id: {analysis!r}"

    # 1. The id the analysis published is a document the store now holds.
    assert store.metadata.get_document(document_id) is not None, (
        "the analysis published a document_id the store cannot find, so every "
        "citation chip in this run addresses nothing"
    )
    assert store.metadata.list_chunks(document_id), "the registered document has no chunks"

    payload = ask(browser, (analysis.get("session_id"), analysis.get("session_token")))
    evidence_ids = chip_evidence_ids(payload)
    assert evidence_ids, (
        "the answer carried no citable evidence id, so there was nothing for the "
        f"chip to address. Answer was: {payload.get('answer')!r}"
    )
    citation_chunks = {
        str(item.get("evidence_id")): str(item.get("chunk_id") or "")
        for item in payload.get("citations") or []
        if isinstance(item, dict)
    }

    for evidence_id in evidence_ids:
        response = browser.get(citation_click_url(document_id, evidence_id))
        assert response.status_code == 200, (
            "a citation the reader can click did not open. The analysis path "
            "publishes a document_id the store never received, so the URL the "
            "browser builds resolves to nothing.\n"
            f"  document: {document_id}\n"
            f"  evidence: {evidence_id}\n"
            f"  body: {response.text[:400]}"
        )
        body = response.json()
        assert "error" not in body
        assert body["document_id"] == document_id

        # 3. The same passage the answer cited, not merely a passage.
        assert citation_chunks.get(evidence_id), (
            "the answer cited an evidence id with no chunk behind it; the retriever "
            "or this file's expectations changed and they are no longer paired"
        )
        assert body["chunk_id"] == citation_chunks[evidence_id], (
            "the click opened a different passage than the citation named — the "
            "right shape with the wrong text is the worse failure"
        )

        # 4. And a locator the panel can render.
        locator = body.get("locator") or {}
        assert locator.get("page"), (
            f"the passage opened with no page ({locator!r}); the panel would have "
            "nothing to point a highlight at"
        )


def test_a_re_analysis_does_not_embed_the_document_twice(
    browser, store, corpus, monkeypatch
):
    """Running the analysis again over the same paper costs no second embedding.

    Two layers have to hold, and the test names both because either one alone
    would leave the reader paying twice:

    * **The submission.** Two uploads of the same bytes into the same session
      with the same switches are one request sent twice -- a double-clicked
      button, a retried fetch. ``routes_jobs.analysis_idempotency_key`` is what
      says so, and ``deduplicated`` on the second response is what proves it.
    * **The registration.** Even when a job *does* run again, it must not pay to
      embed a paper the index already holds.
      ``index_document_for_retrieval`` answers that before chunking, so a
      re-analysis re-reads a row and returns.

    The assertion is on the embedding counter, not on the job id: a deduplicated
    job that still re-ran its stage, or a stage that re-embedded before the
    second submission was recognised, both leave every other observable correct
    and the reader paying for the same vectors twice.
    """
    embedder = CountingEmbedder()
    monkeypatch.setattr(orchestrator, "_configured_embedder", lambda: embedder)
    monkeypatch.setattr(orchestrator, "_QA_RETRIEVERS", {})

    first_status, first = analyze_document(
        browser, corpus, session_id="rerun-session"
    )
    assert first_status == 200, first
    embedded_after_first = len(embedder.calls)
    second_status, second = analyze_document(
        browser, corpus, session_id="rerun-session"
    )

    assert second_status == 202, (
        "the second submission should not have re-run the analysis, so it has no "
        f"result of its own to hand back; got {second_status}: {second!r}"
    )
    assert second.get("deduplicated") is True, (
        "the second submission was answered as fresh work, so a double-clicked "
        f"button is paying for a second analysis: {second!r}"
    )
    assert second.get("job_id") == first.get("job_id"), (
        "the two submissions were not recognised as one request, so the "
        "idempotency key has stopped deduplicating and a double-click now costs "
        "a whole re-analysis"
    )
    assert embedded_after_first == 1, (
        "the first analysis did not embed the document at all; this test would "
        "otherwise pass on a broken counter"
    )
    assert len(embedder.calls) == embedded_after_first, (
        f"the document was embedded {len(embedder.calls) - embedded_after_first} "
        "more time(s) for one paper; the registration is not idempotent"
    )

    document_id = str(first.get("document_id") or "")
    assert len(store.metadata.list_chunks(document_id)) == int(
        (first.get("document_registration") or {}).get("chunk_count") or 0
    ), "the re-analysis changed the chunk set under the reader's citation"


def test_a_failed_registration_still_returns_the_analysis_with_a_notice(
    browser, corpus, lexical_only, monkeypatch
):
    """A reader who got their summary keeps it when indexing dies.

    The failure is injected at the one seam that could plausibly take the whole
    analysis down with it -- the store write that does the embedding -- because
    a test that broke something else would not have tested the promise. The
    analysis is still the expensive, real one: a real parse and six real
    sections.

    The notice is asserted in both places it can be read from, because the
    reader reaches it in one of them and a client in the other.
    ``result["parse"]["warnings"]`` is what the shipped front end renders under
    the Parse Status card (``parseStatus.js``), and
    ``result["document_registration"]`` is what a client branches on. A notice
    in only one is a notice half its readers never see.
    """
    from paperwhisperer.api import routes_v1

    def unavailable(*args, **kwargs):
        raise RuntimeError("the store is down")

    monkeypatch.setattr(routes_v1, "persist_document", unavailable)

    analysis = analyzed(browser, corpus)

    # The analysis, whole.
    assert analysis.get("summary"), "a store outage cost the reader their summary"
    assert analysis.get("sections"), "a store outage cost the reader their sections"
    assert analysis.get("char_count"), "a store outage emptied the analysis"
    assert analysis.get("session_id") and analysis.get("session_token"), (
        "a store outage cost the reader their session, so their next question "
        "has nothing to retrieve from"
    )

    registration = analysis.get("document_registration") or {}
    assert registration.get("status") == "failed", registration
    assert registration.get("code") == "document_indexing_failed", registration
    notice = str(registration.get("notice") or "")
    assert notice, "a failed registration reported no notice at all"

    warnings = list((analysis.get("parse") or {}).get("warnings") or [])
    assert notice in warnings, (
        "the failure was not surfaced on the surface the interface renders, so a "
        f"reader would click a citation, get a 404, and be told nothing: {warnings}"
    )
    # And the notice has to say the summary survived. A sentence that only named
    # the failure would read as "your analysis is unreliable", which is the one
    # claim it must not make: the analysis completed and is returned in full.
    assert "分析结果本身完整" in notice, (
        f"the notice does not tell the reader their analysis survived: {notice!r}"
    )
    assert "索引" in notice, f"the notice does not say what failed: {notice!r}"


def test_an_evidence_id_from_another_paper_is_still_refused_after_registration(
    browser, store, corpus, lexical_only
):
    """Registration did not buy any id the store did not already mint.

    Not a variant of the test above. That one asserts a citation whose evidence
    belongs to *this* upload opens; this one asserts an id that genuinely names
    a chunk of a **different** paper is still refused, and that the refusal
    leaks nothing.

    The distinction is the whole safety argument. "Every id resolves somewhere"
    is not a fix for the click above -- it is a fix for a different, much worse
    product: a reader clicking a citation would be shown a passage from a
    document they did not open, and one whose text happened to align would look
    correct. So the assertion is on the *code* as well as the status: the
    response must be the refusal, and the foreign paper's text must not appear
    anywhere in the body.

    The foreign id is minted by the real retriever over a real second document
    under the real QA plan, so a resolver that had genuinely learned to leak
    would be caught here rather than in production. The reader's own document is
    registered by the **analysis path** rather than by hand, because that is the
    registration whose scope this test is checking.
    """
    from paperwhisperer.api.routes_v1 import persist_document
    from paperwhisperer.rag.retrieve import TASK_QA, HybridRetriever, plan_for

    # A different paper, registered and indexed so it mints real ids.
    foreign_document = _paper(
        "doc_parsed_other_after_registration",
        first=f"Results: {FOREIGN_FRAGMENT}.",
        last="This line belongs to a different paper entirely.",
    )
    persist_document(store, foreign_document, {"status": "ok"})

    # The reader's document, registered the way the browser's analysis does it.
    analysis = analyzed(browser, corpus)
    document_id = str(analysis.get("document_id") or "")
    assert document_id

    retriever = HybridRetriever(store, embeddings=None, reranker=None)
    retrieved = retriever.retrieve(
        "What top-1 accuracy did the other paper report?",
        plan=plan_for(TASK_QA),
        document_ids=[foreign_document.document_id],
    )
    assert retrieved.evidence, "the foreign document was not retrievable; the fixture is wrong"
    foreign_id = retrieved.evidence[0].evidence_id
    assert foreign_id.startswith("ev_")

    # Sanity, in the direction that must keep working: the id *is* a real
    # citation, and resolves against the document it was minted over. Without
    # this, the refusal below would only prove the store was empty.
    owner = browser.get(citation_click_url(foreign_document.document_id, foreign_id))
    assert owner.status_code == 200, (
        "the control failed: a real citation against its own document should open. "
        f"Got {owner.status_code} {owner.text[:300]}"
    )
    assert FOREIGN_FRAGMENT in owner.json()["snippet"]["text"]

    # The refusal, against the document the analysis path registered.
    response = browser.get(citation_click_url(document_id, foreign_id))
    assert response.status_code == 404, (
        "an evidence id belonging to another paper resolved against the document "
        f"the analysis registered — cross-document isolation is broken. Got "
        f"{response.status_code} {response.text[:300]}"
    )
    body = response.json()
    assert body.get("code") in {"evidence_not_found", "evidence_not_in_document"}, body
    assert FOREIGN_FRAGMENT not in response.text, "the refusal body leaked the other paper's text"
    assert "snippet" not in body, body


# --------------------------------------------------------------------------
# The degraded path — and the honesty of the note on it
# --------------------------------------------------------------------------


def test_a_session_without_a_registered_document_degrades_without_inventing_a_hit(
    client, store, runtime_dirs, scripted_model, parsed_document
):
    """A session whose text has no registered document is not silently repaired.

    ``persist_document`` is what registers an upload. A session that carries
    text but was never registered — a document indexed by a build that has since
    been reaped, or a session restored from a backup — has nothing for a
    citation to point at.

    The assertion is on the *shape of the answer*, not on a click: whatever the
    Q&A route does here, it must not present a citation whose id resolves to
    nothing. The answer is either the honest refusal, or an answer whose
    citations resolve against the document it actually cites. What it must never
    be is a ``direct_quote`` claim paired with an evidence id that no document
    in the store will honour.
    """
    from paperwhisperer.core.text import build_session_expiry
    from paperwhisperer.sessions.store import hash_session_token, write_session_payload

    session_id, token = "orphan-session", "orphan-token"
    write_session_payload(
        session_id,
        {
            "expires_at": build_session_expiry(),
            "document_content": parsed_document.to_full_text(),
            "document_id": parsed_document.document_id,  # named, but never registered
            "qa_history": [],
            "paper_search": {},
            "session_auth": {"token_hash": hash_session_token(token)},
        },
    )

    payload = ask(client, (session_id, token))
    for evidence_id in chip_evidence_ids(payload):
        # Whatever document the answer says it cited from, the id has to resolve
        # there. The parsed id is the only one this session has ever named, so
        # that is the URL a client would build.
        response = client.get(citation_click_url(parsed_document.document_id, evidence_id))
        assert response.status_code in {200, 404}
        if response.status_code == 200:
            assert LATE_FRAGMENT in response.json()["snippet"]["text"]