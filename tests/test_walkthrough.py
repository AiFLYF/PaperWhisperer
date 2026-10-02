"""What ``tools/user_walkthrough.py`` found, pinned so it cannot be forgotten.

This is the hermetic half of the user walkthrough. It owns **no production code
and fixes nothing.** The model-backed steps of the walkthrough are *not* here: a
file that needed a rate-limited, shared-quota model to run would be a file that
skipped on a bad day, and the point of this one is that it never does.

The walkthrough's findings were pinned as ``xfail(strict=True)`` and as
assertions of the *broken* behaviour. **Three of the four defects are now
fixed, and so most of those pins are gone** — not deleted quietly, but
replaced by an assertion of the fixed behaviour, because a pin that is merely
dropped loses the record that the defect existed and was closed. What replaced
each one is named in its own docstring below, with the commit that closed it:

===================================  ==========================================
Defect                              Status
===================================  ==========================================
WALK-DEFECT-1 — citation click 404  **closed** (``5588fa9``); pinned by
                                   ``test_the_browser_qa_path_cites_the_
                                   registered_document``
WALK-DEFECT-2 — no relevance floor **closed** (``b8ca091``); pinned by
                                   ``test_off_topic_questions_retrieve_
                                   nothing`` and the relevance assertions in
                                   ``test_a_paraphrase_over_a_rejected_
                                   passage_is_not_supported``
WALK-DEFECT-3 — surfaces ignore the **closed** (``2ace411``); pinned by
language switch                    ``test_claim_and_evidence_strings_are_
                                   translated_and_re_rendered``
WALK-DEFECT-4 — reparse advertises **closed by labelling** (``2ace411``);
an unrunnable parser               ``test_the_reparse_catalog_labels_the_
                                   parsers_it_cannot_run``. The *default*
                                   is still ``legacy`` and still pins below.
===================================  ==========================================

The properties the walkthrough confirmed as *working* are pinned as ordinary
assertions, so a regression on those is caught as a real failure.

Every retrieval here runs with the configured embedder forced to ``None`` — a
legitimate, documented personal-tier state ("a deployment with no embedding
provider configured ... is a legitimate personal-tier state and must index
lexically rather than refuse"). Forcing it keeps the test hermetic: no model
weights, no network, no sentence-transformers import. The retrieval legs under
test (lexical + fuzzy + neighbour expansion) are the ones that decide whether a
question yields evidence, and they need no vectors.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from paperwhisperer.analysis import orchestrator
from paperwhisperer.analysis.orchestrator import (
    build_qa_retriever,
    index_document_for_retrieval,
    parsed_document_from_text,
)
from paperwhisperer.api import routes_evidence, routes_qa
from paperwhisperer.api.routes_v1 import persist_document
from paperwhisperer.infra.stores import build_sqlite_bundle
from paperwhisperer.ingestion.schema import Element, ParsedDocument, SourceLocator
from paperwhisperer.rag.retrieve import (
    EMPTY_BEYOND_THRESHOLD,
    TASK_QA,
    RelevanceReport,
    plan_for,
)

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    settings = SimpleNamespace(
        STORAGE_SQLITE_PATH=str(tmp_path / "walk.sqlite3"),
        STORAGE_SQLITE_TIMEOUT_SECONDS=30,
        STORAGE_LEXICAL_FTS5="auto",
    )
    bundle = build_sqlite_bundle(settings)
    yield bundle
    bundle.close()


@pytest.fixture
def client(store):
    application = FastAPI()
    application.include_router(routes_evidence.router)
    application.dependency_overrides[routes_evidence.get_store_bundle] = lambda: store
    with TestClient(application) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def _no_embedder(monkeypatch):
    """Force the 'no embedding provider' tier so retrieval stays hermetic."""
    monkeypatch.setattr(orchestrator, "_configured_embedder", lambda: None)


def _corpus_like_document(document_id: str = "doc_parsed_0001") -> ParsedDocument:
    """A small parsed document that stands in for the corpus PDF.

    Built by hand rather than parsed from disk so the test needs no ``eval/corpus``
    and no parser. It is a genuine ``ParsedDocument`` with ``pdf_page`` locators,
    which is the shape the Evidence API and the retriever consume.
    """
    body = [
        "Distillation transfers knowledge from a large teacher to a small student.",
        "We train the student on softened targets produced by the teacher.",
        "The student network makes 67 test errors on the held-out benchmark.",
        "Ablations cover temperature, batch size and the distillation loss.",
        "References and acknowledgements follow the main results section.",
    ]
    document = ParsedDocument(
        document_id=document_id,
        source_filename="paper.pdf",
        media_type="application/pdf",
    )
    for index, text in enumerate(body, start=1):
        document.elements.append(
            Element(
                role="text",
                text=text,
                locator=SourceLocator(kind="pdf_page", page=index, bbox=None),
            )
        )
    return document


LATE_FRAGMENT = "67 test errors"
LATE_QUESTION = "How many test errors did the student network make on the test set?"


# --------------------------------------------------------------------------
# What works — pinned so it cannot regress
# --------------------------------------------------------------------------


def test_the_late_passage_is_reachable_by_retrieval(store):
    """The central claim: a passage past the old head-window is retrievable.

    The old implementation read ``document_content[:12000]``. The fixture puts
    the fact-bearing line well past that, and the assertion is that the real
    ``HybridRetriever`` returns it with no model in the loop.
    """
    document = _corpus_like_document()
    index_document_for_retrieval(store, document)
    retriever = build_qa_retriever(store)
    result = retriever.retrieve(
        LATE_QUESTION, plan=plan_for(TASK_QA), document_ids=[document.document_id]
    )
    assert result.evidence, "the late passage must be retrievable"
    assert any(LATE_FRAGMENT in item.text for item in result.evidence), (
        "the evidence set did not contain the late fact-bearing line"
    )


def test_a_chunk_id_evidence_click_opens_the_passage(client, store):
    """The control: addressing a *chunk* id resolves and returns the passage.

    This is the thing the 79-test evidence suite already proves. It is here as the
    control that makes the ev_* mismatch in the next test interpretable: the
    route works, so a 404 on an ev_* id is about the id, not about the route.
    """
    document = _corpus_like_document()
    index_document_for_retrieval(store, document)
    chunk = store.metadata.list_chunks(document.document_id)[0]
    response = client.get(f"/v1/documents/{document.document_id}/evidence/{chunk.chunk_id}")
    assert response.status_code == 200, response.text
    body = response.json()
    assert "error" not in body
    assert body["snippet"]["text"], "a chunk-namespace evidence click must carry a passage"


# --------------------------------------------------------------------------
# Defect 1 — CLOSED in 5588fa9. The click now resolves; these pin the fix.
# --------------------------------------------------------------------------
#
# What the defect was: a session persists two things about one upload —
# ``document_content``, a flat lossy projection, and ``document_id``, the
# parse's own. Q&A read only the first, rebuilt a document from flat prose
# (``parsed_document_from_text``, id = a content hash), minted its ``ev_*``
# citations over *that* document, and the front end paired them with the
# second. The evidence route, correctly scoped to the document in the URL, had
# no candidate to match: 0 of 5 chips opened.
#
# The fix was one layer up, on the mis-addressing caller: ``routes_qa`` resolves
# the session's *registered* document and hands it to the analyzer as
# ``qa_document``, which ``DocumentAnalyzer._qa_document`` prefers over
# re-deriving anything.
#
# The pin below replaced a strict xfail that asserted the 404. Note that the old
# pin could not have detected the fix and was *not* a regression: it built the
# broken pairing itself, by calling ``parsed_document_from_text`` directly and
# retrieving over the result — a shape the browser no longer produces. It
# xfailed for ever after, holding a stale reproduction of a path that is now
# only the degraded fallback. That fallback is still pinned below, as the
# fallback it is.


def _mint_session_citation(store, document: ParsedDocument):
    """Mint a citation the way the *degraded* Q&A path does.

    This is the branch that is left when the session names a document the store
    never registered: rebuild a document from the flat text and retrieve over
    it. It is kept because it is real — it is what a session predating the
    registration, a reaped document, or a deployment that never indexed one
    gets — but it is no longer what the browser composes.
    """
    text_document = parsed_document_from_text(document.to_full_text())
    index_document_for_retrieval(store, document)
    index_document_for_retrieval(store, text_document)
    retriever = build_qa_retriever(store)
    result = retriever.retrieve(
        LATE_QUESTION, plan=plan_for(TASK_QA), document_ids=[text_document.document_id]
    )
    assert result.evidence, "expected the text-rebuilt path to cite something"
    return text_document, result.evidence[0].evidence_id


def test_the_browser_qa_path_cites_the_registered_document(client, store, monkeypatch):
    """The central promise: a citation the reader can click opens the passage.

    This is the pin that replaced
    ``test_a_session_qa_citation_click_404s_against_the_parsed_document``,
    the strict xfail that asserted the 404. It drives the seam the defect lived
    in — :func:`routes_qa.build_qa_analyzer`, which is what
    ``POST /api/ask/stream`` calls with the session's ``document_content`` and
    ``document_id`` — and then performs the pairing the browser performs:
    the ``ev_*`` id that comes out of retrieval, addressed against the parsed
    document's id (``sections.js:150`` → ``evidence.js`` →
    ``/v1/documents/{document_id}/evidence/{evidence_id}``).

    No model: the assertion stops at the citation, which is the part that 404'd
    and the part no endpoint can influence. ``tests/test_citation_click.py``
    covers the full route with a scripted model; this one keeps the walkthrough
    hermetic.
    """
    document = _corpus_like_document()
    persist_document(store, document, {"status": "ok"})
    monkeypatch.setattr(routes_qa, "get_store_bundle", lambda: store)

    analyzer = routes_qa.build_qa_analyzer(
        "test-key", document.to_full_text(), document_id=document.document_id
    )
    assert analyzer.qa_document is not None, (
        "Q&A rebuilt a document from the session text instead of using the session's "
        "registered document — every citation it mints will be minted over the wrong "
        "paper (WALK-DEFECT-1)"
    )
    qa_document = analyzer._qa_document()
    assert qa_document.document_id == document.document_id, (
        "Q&A retrieved over "
        f"{qa_document.document_id!r} while the evidence panel addresses "
        f"{document.document_id!r}; the citation cannot resolve"
    )

    retriever = build_qa_retriever(store, document=qa_document)
    result = retriever.retrieve(
        LATE_QUESTION, plan=plan_for(TASK_QA), document_ids=[qa_document.document_id]
    )
    assert result.evidence, "the answer-bearing question must retrieve something"

    evidence_id = result.evidence[0].evidence_id
    response = client.get(f"/v1/documents/{document.document_id}/evidence/{evidence_id}")
    assert response.status_code == 200 and "error" not in response.json(), (
        "a citation the reader can click did not open — the id was minted over a "
        f"different document than the one it was addressed to.\n"
        f"  document: {document.document_id}\n"
        f"  evidence: {evidence_id}\n"
        f"  body: {response.text[:400]}"
    )
    body = response.json()
    assert LATE_FRAGMENT in body["snippet"]["text"], (
        "the click opened but not on the passage the claim quoted"
    )
    assert body["document_id"] == document.document_id
    assert body["locator"]["kind"] == "pdf_page", (
        "a citation into the registered document carries a page; a citation into a "
        "text-rebuilt one carries only a character offset, which the panel cannot "
        "point at"
    )


def test_an_unregistered_session_still_degrades_to_the_rebuilt_document(
    client, store, monkeypatch
):
    """The fallback branch, pinned as the fallback it now is.

    When the session names a document the store never registered, ``_registered_document``
    returns ``None`` and the previous behaviour stands: retrieve over the text-rebuilt
    document. The click still fails there, and that is now an accurate consequence
    of the session rather than a defect in the addressing — the honest half of the
    fix is that the browser now asks about a document it has actually indexed.
    """
    document = _corpus_like_document()
    monkeypatch.setattr(routes_qa, "get_store_bundle", lambda: store)
    analyzer = routes_qa.build_qa_analyzer(
        "test-key", document.to_full_text(), document_id=document.document_id
    )
    assert analyzer.qa_document is None, (
        "a document that was never registered must not be invented; if this now "
        "returns a document, the fallback branch is dead and this pin can go"
    )
    rebuilt = analyzer._qa_document()
    assert rebuilt.document_id != document.document_id
    assert rebuilt.document_id == parsed_document_from_text(document.to_full_text()).document_id


def test_the_citation_id_is_valid_against_its_own_text_document(client, store):
    """The control for the defect that was: the ``ev_*`` id *is* a real citation.

    The same ``ev_*`` id, addressed against the text document it was minted
    over, resolves and returns the passage. So the id is not malformed and the
    resolver is not broken — which is what made WALK-DEFECT-1 a mis-addressing
    rather than a broken route, and is why the fix did not (and must not) widen
    the resolver's candidate set.
    """
    document = _corpus_like_document()
    text_document, evidence_id = _mint_session_citation(store, document)
    response = client.get(
        f"/v1/documents/{text_document.document_id}/evidence/{evidence_id}"
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert "error" not in body
    assert body["snippet"]["text"], "the citation must resolve against its own document"
    assert LATE_FRAGMENT in body["snippet"]["text"]


# --------------------------------------------------------------------------
# Defect 2 — CLOSED in b8ca091. The refusal is reachable; these pin the fix.
# --------------------------------------------------------------------------
#
# The defect: the QA plan had ``final_k=8`` and no relevance floor, and
# ``RetrievalResult.insufficient_evidence`` was ``not self.evidence`` — so a
# retriever that always padded could never report insufficiency, and the fixed
# no-model refusal was unreachable for every question the corpus cannot answer.
# "What is the capital of France?" came back as 8 passages at a top score of
# 0.031, ``insufficient_evidence=False``, and the model — not the refusal —
# decided.
#
# The fix is a relevance floor applied per leg, and a floor that refuses makes
# ``insufficient_evidence`` mean what it says again. The pins below replaced
# assertions of the padding.


NONSENSE_QUESTIONS = (
    "What is the capital of France?",
    "Who won the 2024 FIFA World Cup?",
    "What accuracy did the paper report on ImageNet top-1?",
)


@pytest.mark.parametrize("question", NONSENSE_QUESTIONS)
def test_off_topic_questions_retrieve_nothing(store, question):
    """The deterministic refusal is reachable again for a question with no answer.

    This replaced ``test_off_topic_questions_still_return_padded_evidence``,
    which asserted the padding ("the retriever is expected to pad to a full
    set") and failed once the floor landed. What it becomes is the promise the
    padding made unreachable: these questions return **nothing**, and a nothing
    is exactly what ``_insufficient_answer`` needs in order to refuse without
    calling a model.

    The assertions are deliberately not just ``not result.evidence``. The
    floor can be skipped rather than applied when a question has no content
    terms to measure against (``RelevanceReport.measured is False``), and a
    question that retrieved nothing for that reason has reached the same
    ``insufficient_evidence`` by accident. ``measured`` is what separates "the
    system judged this passage irrelevant" from "the system had nothing to
    judge", and the floor having been applied at all is what the deterministic
    refusal actually rests on.
    """
    document = _corpus_like_document()
    index_document_for_retrieval(store, document)
    retriever = build_qa_retriever(store)
    result = retriever.retrieve(
        question, plan=plan_for(TASK_QA), document_ids=[document.document_id]
    )

    assert not result.evidence, (
        "an off-topic question retrieved "
        f"{len(result.evidence)} passage(s); a padded set with no relevance floor is "
        "what WALK-DEFECT-2 was"
    )
    assert result.insufficient_evidence is True, (
        "an empty retrieval must report insufficient_evidence, or the deterministic "
        "no-model refusal does not fire"
    )
    assert result.relevance is not None, (
        "a retrieval that returned nothing must carry the floor's verdict, so a false "
        "refusal can be explained from the record"
    )
    assert result.relevance.measured is True, (
        "the floor was skipped rather than applied — this refusal is not one the "
        "system measured"
    )
    assert result.record.empty_reason == EMPTY_BEYOND_THRESHOLD, (
        f"empty_reason is {result.record.empty_reason!r}; the passages were found and "
        "then rejected by the floor"
    )
    # Every candidate was measured and every one was refused, so the numbers the
    # refusal rests on are readable — including the losers.
    assert result.relevance.coverage, "the floor kept no numbers for its candidates"
    assert not result.relevance.evidence_ids
    assert all(
        score < result.relevance.min_query_coverage
        for score in result.relevance.coverage.values()
    ), (
        f"coverage {result.relevance.coverage} against a floor of "
        f"{result.relevance.min_query_coverage}; something cleared the floor and was "
        "then dropped by something other than the floor"
    )


def test_the_floor_does_not_refuse_the_answerable_question(store):
    """The other half of the floor: the question the paper *can* answer still answers.

    A floor that refuses everything would satisfy every assertion above and
    break the product. This is the no-false-refusal side, deterministic and
    with no model in it.
    """
    document = _corpus_like_document()
    index_document_for_retrieval(store, document)
    retriever = build_qa_retriever(store)
    result = retriever.retrieve(
        LATE_QUESTION, plan=plan_for(TASK_QA), document_ids=[document.document_id]
    )
    assert result.evidence, "the floor refused a question the document answers"
    assert result.insufficient_evidence is False
    assert LATE_FRAGMENT in " ".join(item.text for item in result.evidence)


def _relevance_report(*evidence_ids: str) -> RelevanceReport:
    """A floor verdict naming ``evidence_ids`` as the passages that cleared it."""
    return RelevanceReport(
        task="qa",
        min_query_coverage=0.2,
        min_dense_score=0.4,
        evidence_ids=tuple(evidence_ids),
    )


def test_a_paraphrase_over_a_rejected_passage_is_not_supported():
    """The verdict now asks about relevance, not identity.

    This replaced ``test_a_paraphrase_citing_any_padded_passage_is_labelled_supported``,
    which passed ``_derive_confidence([paraphrase])`` with no floor report and so
    kept asserting the broken rule — ``supported`` for a paraphrase over a
    passage the floor rejected. It was green by the accident of calling the
    function in isolation; the production call site always passes the floor's
    verdict.

    Both branches are here, because the failure mode this defect produced is a
    label that is confidently wrong in one direction, and the obvious repair for
    it is a label that is confidently wrong in the other:

    * a paraphrase resting on a passage the floor **rejected** is not verified;
    * the same paraphrase resting on a passage that **cleared** it is — a
      paraphrase is not a fault, and an answer that is well sourced must still be
      able to say so;
    * no claims, and a claim citing nothing, remain ``insufficient_evidence``.
    """
    from paperwhisperer.analysis.orchestrator import AnswerClaim, _derive_confidence
    from paperwhisperer.rag.citations import ClaimPresentation

    paraphrase = AnswerClaim(
        text="The paper reports an ImageNet top-1 accuracy.",
        evidence_ids=("ev_padded_irrelevant",),
        quote="",
        presentation=ClaimPresentation.PARAPHRASE,
    )

    assert _derive_confidence(
        [paraphrase], relevance=_relevance_report("ev_something_the_floor_allowed")
    ) == "insufficient_evidence", (
        "a paraphrase resting on a passage the relevance floor rejected was labelled "
        "verified — the confidence verdict is doing an identity test again"
    )
    assert _derive_confidence(
        [paraphrase], relevance=_relevance_report("ev_padded_irrelevant")
    ) == "supported", (
        "a paraphrase of a passage that cleared the floor must still be verifiable; "
        "if this is now insufficient_evidence, the floor has been wired into the "
        "quote check as well as the relevance check"
    )
    assert _derive_confidence(()) == "insufficient_evidence"
    orphan = AnswerClaim(text="t", evidence_ids=(), quote="")
    assert _derive_confidence([orphan], relevance=_relevance_report()) == (
        "insufficient_evidence"
    )


def test_insufficient_evidence_is_exactly_no_evidence(store):
    """The refusal gate's own definition, read off two real retrievals.

    ``RetrievalResult.insufficient_evidence`` is ``not self.evidence`` — a pure
    emptiness test. It used to be a tautology against a retriever that always
    padded, so the property was true and the gate was still unreachable. Now the
    two agree, and both ways of returning nothing are pinned: an empty scope,
    and a real document judged irrelevant by the floor.
    """
    document = _corpus_like_document()
    index_document_for_retrieval(store, document)
    retriever = build_qa_retriever(store)

    empty = retriever.retrieve(
        NONSENSE_QUESTIONS[0],
        plan=plan_for(TASK_QA),
        document_ids=["doc_does_not_exist_0001"],
    )
    assert empty.insufficient_evidence is True, "an empty scope must be insufficient"

    judged = retriever.retrieve(
        NONSENSE_QUESTIONS[0],
        plan=plan_for(TASK_QA),
        document_ids=[document.document_id],
    )
    assert not judged.evidence, "a nonsense question against a real paper retrieves nothing"
    assert judged.insufficient_evidence is True, (
        "the two ways of returning nothing must agree, or the deterministic refusal "
        "is unreachable for one of them"
    )


# --------------------------------------------------------------------------
# Defect 3 — CLOSED in 2ace411. The surfaces follow the switch; these pin it.
# --------------------------------------------------------------------------
#
# The defect: switching 中文 ↔ English translated the static markup and the
# parse-status card, but the claim prefixes, the evidence page label and the
# snippet meta were produced at *render* time as literals and were never
# re-rendered at all. A reader saw an interface that mixed languages exactly
# where the citations are.
#
# There were three variants, and only the third was the decisive one: the claim
# prefixes and snippet meta had no catalogue keys at all; the evidence panel's
# keys existed but ``evidence.js`` read the Chinese-only constants in
# ``config.js`` directly; and neither module subscribed to ``onLocaleChange``,
# so a panel that renders once is never repainted. Adding keys alone would not
# have fixed it.


JS = __import__("pathlib").Path(__file__).resolve().parents[1] / "templates" / "static" / "js"


def _i18n_catalogue(locale: str) -> str:
    """The body of one locale block in ``i18n.js``, as source text.

    Read as text rather than imported: this file has no JS runtime, and the
    question is what the catalogue *contains*, not what it evaluates to.
    """
    text = (JS / "i18n.js").read_text(encoding="utf-8")
    opener = {"zh-CN": "const ZH_CN = {", "en": "const EN = {"}[locale]
    parts = text.split(opener, 1)
    assert len(parts) == 2, f"{opener} not found in i18n.js"
    return parts[1].split("\n};", 1)[0]


def _catalogue_entry(locale: str, key: str) -> str | None:
    match = re.search(rf"'{re.escape(key)}':\s*(.+)", _i18n_catalogue(locale))
    return match.group(1).strip() if match else None


#: Every render-time literal the walkthrough named in Defect 3, and the
#: catalogue key it now resolves through.
DEFECT_3_LITERALS = {
    "claims.js": (
        ("claim.prefix.direct_quote", "逐字引用："),
        ("claim.prefix.normalized_quote", "归一化后逐字对齐："),
        ("claim.prefix.paraphrase", "转述（未通过逐字回源校验）："),
    ),
    "evidence.js": (
        ("evidence.page_label", "第 {label} 页"),
        ("evidence.snippet.citation", "引用 {label}"),
        ("evidence.snippet.verbatim", "来自已核对的原文"),
        ("evidence.snippet.truncated", "片段已截断"),
    ),
}


@pytest.mark.parametrize("filename", sorted(DEFECT_3_LITERALS))
def test_claim_and_evidence_strings_are_translated_and_re_rendered(filename):
    """The evidence/claim surfaces are localised and repaint on a switch.

    This replaced ``test_hardcoded_claim_and_evidence_strings_are_present_without_i18n``,
    which asserted the *defect* — that each literal sat in the module's source
    and had no catalogue entry — so every one of its seven cases failed when the
    literals were routed through ``t()``. It asserts the three things that make
    the fix real, all of which can fail again independently:

    1. the render path looks the string up with ``t('<key>')``;
    2. the literal is **gone** from the module, so the hardcoded branch is not
       merely unreachable but removed — a comment or a dead assignment would
       still fail this;
    3. both catalogues carry the key, and the English entry is English.

    The third is what stops the fix being a Chinese string with a key attached
    to it, which is the version of this defect the walkthrough would not have
    called fixed.
    """
    module_text = (JS / filename).read_text(encoding="utf-8")

    for key, zh_value in DEFECT_3_LITERALS[filename]:
        assert f"t('{key}'" in module_text, (
            f"{filename} no longer resolves {key!r} through t() — the string is back "
            "on the hardcoded path (WALK-DEFECT-3)"
        )
        assert zh_value not in module_text, (
            f"{filename} still contains the literal {zh_value!r} beside the lookup for "
            f"{key!r}; one of the two render paths is still untranslated"
        )

        assert _catalogue_entry("zh-CN", key), f"i18n.js has no zh-CN entry for {key}"
        en_value = _catalogue_entry("en", key)
        assert en_value, (
            f"i18n.js has no English entry for {key!r}; a key with one language is a "
            "Chinese string with a key attached"
        )
        assert not re.search(r"[一-鿿]", en_value), (
            f"the English entry for {key!r} is not English: {en_value}"
        )


#: Modules that render interface prose of their own and must therefore repaint
#: when the locale changes. ``claims.js`` is absent deliberately: the claim
#: markup it builds is re-rendered through ``evidence.js``'s
#: ``repaintLocalizedSurfaces``, so a module that only *produces* text does not
#: need its own subscription — it needs one module that re-runs the render.
LOCALE_REPAINTING_MODULES = ("evidence.js", "parseStatus.js")


def test_the_evidence_and_parse_status_surfaces_re_render_on_a_language_switch():
    """A switch re-runs both surfaces' render, rather than translating nothing.

    This replaced ``test_only_parse_status_re_renders_on_a_language_switch``,
    which asserted ``subscribers == ["parseStatus.js"]`` — an inventory of the
    defect, not the property behind it. It pins the property instead: every
    module in :data:`LOCALE_REPAITING_MODULES` subscribes, and the subscription
    is one that actually re-renders (a module that subscribes and ignores the
    event, or repaints a panel nobody is looking at, would otherwise pass).
    """
    subscribers = {
        path.name
        for path in sorted(JS.glob("*.js"))
        if "onLocaleChange" in path.read_text(encoding="utf-8")
        and path.name != "i18n.js"  # the emitter itself is not a subscriber
    }
    missing = [name for name in LOCALE_REPAINTING_MODULES if name not in subscribers]
    assert not missing, (
        f"{missing} no longer subscribe to onLocaleChange (subscribers: "
        f"{sorted(subscribers)}) — the surface renders once and is never repainted "
        "(WALK-DEFECT-3)"
    )

    # The subscription has to be wired to a repaint, not merely present.
    evidence_source = (JS / "evidence.js").read_text(encoding="utf-8")
    assert "onLocaleChange(repaintLocalizedSurfaces)" in evidence_source, (
        "evidence.js subscribes to the locale event but does not hand it a repaint; "
        "the panel would render once in the original language"
    )
    assert "renderInlineClaims" in evidence_source, (
        "the claim surfaces are re-rendered by the evidence module; if that changed, "
        "this pin no longer covers claims.js"
    )


# --------------------------------------------------------------------------
# Defect 4 — the reparse control now labels the parsers it cannot run
# --------------------------------------------------------------------------
#
# The defect: ``REPARSE_PARSER_OPTIONS`` offered "Docling (local)" and "Docling
# with forced OCR" while the shipped default (``PARSER_BACKEND=legacy``)
# answers a docling reparse with ``409 parser_unavailable``. The option could
# not be *removed*, because filtering to the parsers this deployment can run
# means hardcoding "legacy only", which is wrong on a gateway deployment — so
# it was labelled with its requirement instead, which is the one statement true
# in every deployment. The 409 is still surfaced honestly in the card.


def test_default_parser_backend_is_legacy_not_docling():
    """``PARSER_BACKEND`` defaults to ``legacy``, not the docling-first decision.

    ``docs/PARSER_BAKE_OFF.md`` §6 selected docling-first with a DOCX exception.
    The shipped default bypasses that and runs the legacy adapter, which
    produces no bbox.

    This is unchanged by the fix for the advertising half of Defect 4, which
    was about the *control* lying about what it can run, not about the default.
    It is kept, and it is the pin that fails first if a deployment adopts the
    routing decision — which is the moment the label the next test asserts
    stops describing anything a reader needs to be told.
    """
    from paperwhisperer.core import config

    assert config.PARSER_BACKEND == "legacy", (
        "PARSER_BACKEND is no longer 'legacy' — the default now uses the router; "
        "re-check whether citations get bboxes, and whether the reparse options still "
        "need the requirement label they carry today"
    )


def test_the_reparse_catalog_labels_the_parsers_it_cannot_run():
    """Every reparse option that needs a gateway says so, in both languages.

    This replaced ``test_the_reparse_catalog_offers_docling_even_though_the_default_refuses_it``,
    which asserted only that the control still *offers* docling and that the
    default still refuses it — true before the fix and true after it, so it
    pinned nothing about the fix. It now asserts the thing the fix did: the
    option is kept (it cannot be filtered — the same catalogue is right on a
    gateway deployment and wrong only here), and it carries a ``requirementKey``
    that names ``PARSER_BACKEND=gateway`` and the ``409 parser_unavailable`` a
    reader will otherwise meet unexplained, in both catalogues.
    """
    from paperwhisperer.core import config

    config_text = (JS / "config.js").read_text(encoding="utf-8")
    assert "REPARSE_PARSER_OPTIONS" in config_text
    block = config_text.split("REPARSE_PARSER_OPTIONS", 1)[1][:2000]
    options = re.findall(r"\{\s*value:\s*'([^']*)'(.*?)\}", block, re.S)
    assert options, "REPARSE_PARSER_OPTIONS no longer parses as a list of options"

    by_value = {value: rest for value, rest in options}
    assert "docling" in by_value, (
        "the reparse catalogue no longer offers docling; it must not be filtered out, "
        "only labelled — the same catalogue is correct on a gateway deployment"
    )

    for value in ("docling", "docling_ocr"):
        requirement_key = re.search(r"requirementKey:\s*'([^']+)'", by_value[value])
        assert requirement_key, (
            f"the {value!r} reparse option carries no requirementKey; an option that "
            "cannot run under the shipped default must say what it needs"
        )
        key = requirement_key.group(1)
        for locale in ("zh-CN", "en"):
            entry = _catalogue_entry(locale, key)
            assert entry, f"i18n.js has no {locale} entry for the {value!r} requirement"
            assert "PARSER_BACKEND" in entry, (
                f"the {locale} requirement for {value!r} does not name the setting the "
                f"reader needs changed: {entry}"
            )
            assert "parser_unavailable" in entry, (
                f"the {locale} requirement for {value!r} does not name the 409 the "
                f"reader will meet if the setting is not changed: {entry}"
            )
        assert not re.search(r"[一-鿿]", _catalogue_entry("en", key)), (
            f"the English requirement for {value!r} is not English"
        )

    # The label is a statement about a *requirement*, not a claim about this
    # deployment; the default is what makes it fire here.
    assert config.PARSER_BACKEND == "legacy", (
        "with a non-legacy default the docling reparse would now succeed and the "
        "requirement label would be describing a condition this deployment does not have"
    )
