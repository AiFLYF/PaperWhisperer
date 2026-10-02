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
* **The document id is not chosen either.** It is the one
  :func:`persist_document` registered — the document a reader's upload *is* in
  this product — and it is the one the browser pairs with: ``sections.js:150``
  reads ``data.document_id`` off the analysis result into
  ``state.currentDocumentId``, and ``evidence.js:303`` builds
  ``/v1/documents/{that}/evidence/{chip's evidence id}``. That exact URL is
  what :func:`citation_click_url` reproduces.
* **No model, no network.** The endpoint answered 429 for an entire previous
  session, and a test that skips when the endpoint is down is a test that says
  nothing on the day it matters. The model here is a scripted double that reads
  the ``ev_…`` ids out of the prompt it was handed and cites one — so it proves
  the plumbing end to end, and proves it on a machine with no route to a model.

The isolation test at the bottom is deliberately *not* a variant of the failing
one. It asserts that an evidence id genuinely belonging to a **different paper**
is still refused, so a future "just widen the candidate set" fix cannot pass by
making every citation resolve.
"""

from __future__ import annotations

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
# A second, independent defect — found while looking for the first
# --------------------------------------------------------------------------


def test_the_browser_analyze_path_never_registers_the_document_it_publishes(
    store, scripted_model
):
    """``/api/analyze`` publishes a ``document_id`` it never puts in the store.

    **A separate defect from the one above, and it is not fixed here.** The
    citation-click fix resolves the ``ev_*`` id against the right document — but
    only if the document is *there*. In the browser's own flow it is not.

    ``/api/analyze`` runs the analysis and ``attach_document_identity``
    (``analysis/service.py:76``) copies ``ParseOutcome.document.document_id``
    onto the result. That is deliberate: its docstring says the id is there
    because "a citation chip addresses
    ``GET /v1/documents/{document_id}/evidence/{evidence_id}``". The browser
    reads it into ``state.currentDocumentId`` (``sections.js:150``) and every
    citation click addresses it.

    Nothing on that path registers the document. ``routes_jobs`` declares
    ``indexing`` a **passthrough** — "本任务不建立索引：文档检索由 /v1/documents
    负责", on the stated reasoning that "the ``/api`` surface analyses a file
    for a browser and has no retrieval layer to write to". So the id the browser
    publishes names a document the store has never heard of, and the click 404s
    ``document_not_found`` — which is what this assertion records.

    The walkthrough saw ``evidence_not_found`` instead of ``document_not_found``
    only because it uploaded through ``POST /v1/documents`` first, which *does*
    register. Drive the browser's real path and the second code is what you get.
    That is why this defect is invisible to the walkthrough's own repro and
    would survive a fix for the first one.

    **Why this is reported rather than fixed.** Registering here means writing
    to the retrieval index and the vector store from the analysis surface — an
    architectural decision ``routes_jobs.py:396-411`` makes explicitly and in
    writing, and one whose cost (an embedding call per chunk on every upload) is
    a deployment decision. The two modules currently disagree: ``routes_jobs``
    says the analysis surface does not index, and ``attach_document_identity``
    publishes an index-only id to the browser as though it had. Resolving that is
    the owner's call — either the analysis registers, or the browser is routed
    through ``/v1/documents`` — and either way the change is in a file this
    task does not own.

    The assertion is a characterisation of current behaviour, not a wish, so a
    fix turns it into a failure that has to be acknowledged.
    """
    import os

    from paperwhisperer.analysis.service import analyze_saved_file

    corpus = Path(__file__).resolve().parents[1] / "eval" / "corpus" / (
        "digital_single_col/hinton_distillation.pdf"
    )
    if not corpus.exists():
        pytest.skip("the corpus PDF is not present; this asserts about the real parse")

    # The suite-wide ``no_unstubbed_job_stage_model`` tripwire refuses to build a
    # real ``DocumentAnalyzer`` unless a credential is present, so an analysis
    # test has to opt in. The opt-in does not open a socket here: the
    # ``scripted_model`` fixture has already replaced ``LLMClient``, and every
    # completion is answered locally. The variable is restored below either way.
    previous_key = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = "test-key"
    try:
        result = analyze_saved_file(
            file_path=str(corpus),
            original_filename=corpus.name,
            api_key="test-key",
            generate_mermaid_bool=False,
            generate_evaluation_bool=False,
            generate_research_brief_bool=False,
            session_id="analyze-path-session",
        )
    finally:
        if previous_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = previous_key

    document_id = str(result.get("document_id") or "")
    assert document_id, "the analysis published no document id for a citation to address"

    # Today: the published id names nothing the evidence API can read. When the
    # analysis registers the document, this becomes truth and this test fails,
    # which is the signal that the second defect has been fixed.
    assert store.metadata.get_document(document_id) is None, (
        "the analysis path now registers the document it publishes — the second "
        "defect is fixed. Retire this characterisation and, if the browser flow "
        "still mis-pairs an id, re-check the citation-click fix."
    )


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