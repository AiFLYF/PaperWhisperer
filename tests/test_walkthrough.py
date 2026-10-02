"""What ``tools/user_walkthrough.py`` found, pinned so it cannot be forgotten.

This is the hermetic half of the user walkthrough. It owns **no production code
and fixes nothing.** The model-backed steps of the walkthrough are *not* here: a
file that needed a rate-limited, shared-quota model to run would be a file that
skipped on a bad day, and the point of this one is that it never does.

The defects the walkthrough surfaced are pinned as ``xfail(strict=True)``. That
is the convention ``docs/E2E_JOURNEY.md`` already uses for a defect confirmed
today: the assertion records the broken behaviour, and a *fix* turns the xpass
into a failure that forces the marker to be retired. Nothing here is a wish — an
xfail that starts passing is a signal, not a nuisance.

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
from paperwhisperer.api import routes_evidence
from paperwhisperer.infra.stores import build_sqlite_bundle
from paperwhisperer.ingestion.schema import Element, ParsedDocument, SourceLocator
from paperwhisperer.rag.retrieve import TASK_QA, plan_for

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
# Defect 1 — a real citation click 404s (session-QA ev_* vs parsed document id)
# --------------------------------------------------------------------------


def _mint_session_citation(store, document: ParsedDocument):
    """Mint the citation the browser's Q&A actually puts on the wire.

    The browser asks through ``/api/ask/stream``. That route retrieves over a
    document re-derived from the session's *text* (``parsed_document_from_text``),
    which carries a **content-hash document id** distinct from the parsed
    document's id. This reproduces exactly that: index the parsed document (what
    the evidence click addresses) and the text document (what the answer cites),
    retrieve over the text document, and return the first ``ev_*`` id.
    """
    text_document = parsed_document_from_text(document.to_full_text())
    index_document_for_retrieval(store, document)
    index_document_for_retrieval(store, text_document)
    retriever = build_qa_retriever(store)
    result = retriever.retrieve(
        LATE_QUESTION, plan=plan_for(TASK_QA), document_ids=[text_document.document_id]
    )
    assert result.evidence, "expected the session-Q&A path to cite something"
    return text_document, result.evidence[0].evidence_id


@pytest.mark.xfail(
    strict=True,
    reason=(
        "WALK-DEFECT-1: a citation the reader clicks 404s. The browser's Q&A mints "
        "ev_* ids over the session-TEXT document, but the evidence panel addresses the "
        "PARSED document; the two documents never share an id, so the click cannot resolve."
    ),
)
def test_a_session_qa_citation_click_404s_against_the_parsed_document(client, store):
    """A citation emitted by the browser's Q&A does not open — CONFIRMED DEFECT.

    ``state.currentDocumentId`` (evidence.js) is the *analyze* result's document
    id — the parsed one. The citation's ``ev_*`` id is minted over the session-text
    document. ``GET /v1/documents/{parsed_id}/evidence/{ev_id}`` therefore 404s.

    This is pinned as a strict xfail: it asserts the *broken* behaviour. When the
    citation click is fixed this test XPASSes and strict xfail turns that into a
    failure, forcing this marker to be retired deliberately.
    """
    document = _corpus_like_document()
    _, evidence_id = _mint_session_citation(store, document)
    response = client.get(f"/v1/documents/{document.document_id}/evidence/{evidence_id}")
    # Today: 404 evidence_not_found. If a fix lands, this becomes 200 and xfails.
    assert response.status_code == 200 and "error" not in response.json()


def test_the_citation_id_is_valid_against_its_own_text_document(client, store):
    """The control for the defect above: the ev_* id *is* a real citation.

    The same ``ev_*`` id, addressed against the session-text document it was
    minted over, resolves and returns the passage. So the id is not malformed and
    the resolver is not broken — the defect is purely that the browser pairs the
    citation with a *different* document than the one it was minted against.
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
    assert LATE_FRAGMENT in body["snippet"]["text"] or body["snippet"]["text"]


# --------------------------------------------------------------------------
# Defect 2 — the deterministic refusal is unreachable for off-topic questions
# --------------------------------------------------------------------------


NONSENSE_QUESTIONS = (
    "What is the capital of France?",
    "Who won the 2024 FIFA World Cup?",
    "What accuracy did the paper report on ImageNet top-1?",
)


@pytest.mark.parametrize("question", NONSENSE_QUESTIONS)
def test_off_topic_questions_still_return_padded_evidence(store, question):
    """The refusal gate is keyed on "no evidence", and there is always evidence.

    The QA retrieval plan has ``final_k=8`` and **no relevance floor**, and
    ``RetrievalResult.insufficient_evidence`` is exactly ``not self.evidence``.
    The fuzzy + neighbour legs pad every question to a full set, so for a
    question the paper cannot answer the retriever returns near-zero-relevance
    passages, ``insufficient_evidence`` is False, and the model — not the
    deterministic no-model refusal — decides whether to answer or refuse.

    This is pinned as a *characterisation* of the current behaviour rather than a
    wish: it asserts the padding happens. It documents that the central
    "refuses without a model" guarantee is not reached in practice. A future
    relevance floor would make this fail and force it to be re-read.
    """
    document = _corpus_like_document()
    index_document_for_retrieval(store, document)
    retriever = build_qa_retriever(store)
    result = retriever.retrieve(
        question, plan=plan_for(TASK_QA), document_ids=[document.document_id]
    )
    assert result.evidence, "the retriever is expected to pad to a full set"
    assert not result.insufficient_evidence, (
        "if this now returns empty, the deterministic refusal path is reachable — "
        "update this test to assert the refusal instead"
    )
    assert max(item.score for item in result.evidence) < 0.5, (
        "the padded evidence for an off-topic question should be low-relevance; "
        "if it is not, the fixture no longer exercises the padding behaviour"
    )


def test_a_paraphrase_citing_any_padded_passage_is_labelled_supported():
    """The confidence verdict is structural, not a relevance judgement.

    ``_derive_confidence`` calls a claim *verified* when it carries at least one
    retrieved ``evidence_id`` and either presents no quotation (a paraphrase) or
    a quotation that aligned. A paraphrase citing one of the 8 always-returned
    padded passages therefore yields ``supported`` — and ``insufficient_evidence``
    (the honest refusal) is only reached when retrieval returns *nothing*, which
    defect 2 shows it does not.

    This pins the exact chain that makes an unanswerable question able to carry a
    ``supported`` verdict with a citation chip: retrieval pads (no floor) →
    confidence counts a paraphrase against a padded passage as verified. Whether
    the live model actually takes this path needs the endpoint; the structural
    reachability is deterministic and is what this asserts.
    """
    from paperwhisperer.analysis.orchestrator import AnswerClaim, _derive_confidence
    from paperwhisperer.rag.citations import ClaimPresentation

    # A claim that paraphrases a padded, irrelevant passage (no quote).
    paraphrase = AnswerClaim(
        text="The paper reports an ImageNet top-1 accuracy.",
        evidence_ids=("ev_padded_irrelevant",),
        quote="",
        presentation=ClaimPresentation.PARAPHRASE,
    )
    assert _derive_confidence([paraphrase]) == "supported", (
        "a paraphrase citing a padded passage is currently 'supported'; if this "
        "changes, the confidence verdict gained a relevance check"
    )
    # The honest refusal — no claims — is the only route to insufficient_evidence.
    assert _derive_confidence(()) == "insufficient_evidence"
    # A claim that cites nothing at all is never verified.
    orphan = AnswerClaim(text="t", evidence_ids=(), quote="")
    assert _derive_confidence([orphan]) == "insufficient_evidence"


def test_insufficient_evidence_is_exactly_no_evidence(store):
    """The refusal gate's own definition, read off a real retrieval.

    ``RetrievalResult.insufficient_evidence`` is ``not self.evidence`` — a pure
    emptiness test with no relevance floor. Two real retrievals show both
    branches: a scope with no matching document is *insufficient* (the gate can
    fire), but the same document with a nonsense question is *not*. If a future
    change gives the gate a relevance floor, the nonsense branch flips and Defect
    2 is closed.
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

    padded = retriever.retrieve(
        NONSENSE_QUESTIONS[0],
        plan=plan_for(TASK_QA),
        document_ids=[document.document_id],
    )
    assert padded.evidence, "a real scope returns padded evidence"
    assert padded.insufficient_evidence is False, (
        "if this is now insufficient, the deterministic refusal is reachable for "
        "off-topic questions — Defect 2 is closed and should be retired"
    )


# --------------------------------------------------------------------------
# Defect 3 — the evidence/claim surfaces do not follow the language switch
# --------------------------------------------------------------------------

JS = __import__("pathlib").Path(__file__).resolve().parents[1] / "templates" / "static" / "js"

#: Literals that are produced at *render* time in the evidence/claim surfaces and
#: therefore must be routed through ``t()`` to follow a language switch.
HARDCODED_RENDER_STRINGS = {
    "claims.js": ["逐字引用：", "归一化后逐字对齐：", "转述（未通过逐字回源校验）："],
    "evidence.js": ["第 ${label} 页", "来自已核对的原文", "片段已截断"],
}


@pytest.mark.parametrize("filename,literal", [
    (name, literal)
    for name, literals in HARDCODED_RENDER_STRINGS.items()
    for literal in literals
])
def test_hardcoded_claim_and_evidence_strings_are_present_without_i18n(filename, literal):
    """The render-time strings on the evidence surfaces are hardcoded literals.

    They are *expected* to be present and *expected* to bypass the translation
    table today. The test pins the fact so that when someone routes them through
    ``t()`` (the fix), the assertion fails and the walkthrough's Defect 3 is
    updated rather than quietly outliving its fix.
    """
    text = (JS / filename).read_text(encoding="utf-8")
    assert literal in text, f"{filename} no longer contains {literal!r} — re-check Defect 3"

    i18n = (JS / "i18n.js").read_text(encoding="utf-8")
    # A CJK literal is "in i18n" if it appears anywhere in the translation module.
    if re.search(r"[一-鿿]", literal):
        assert literal not in i18n, (
            f"{literal!r} now has an i18n entry — the hardcoded path may be dead code; "
            "re-check whether the render path uses t()"
        )


def test_only_parse_status_re_renders_on_a_language_switch():
    """Only the parse-status card re-renders on ``onLocaleChange``.

    The evidence panel and the claim renderer produce their own text at render
    time but do not subscribe to the locale event, so after a switch the parse
    card is in the new language and the claim/evidence text is not.
    """
    subscribers = [
        path.name
        for path in sorted(JS.glob("*.js"))
        if "onLocaleChange" in path.read_text(encoding="utf-8")
        and path.name != "i18n.js"  # the emitter itself is not a subscriber
    ]
    assert subscribers == ["parseStatus.js"], (
        f"the set of locale-subscribing modules changed to {subscribers} — "
        "re-check Defect 3 (does the evidence/claim surface follow the switch now?)"
    )


# --------------------------------------------------------------------------
# Defect 4 — the shipped default parses with the adapter the routing decision rejected
# --------------------------------------------------------------------------


def test_default_parser_backend_is_legacy_not_docling():
    """``PARSER_BACKEND`` defaults to ``legacy``, not the docling-first decision.

    ``docs/PARSER_BAKE_OFF.md`` §6 selected docling-first with a DOCX exception.
    The shipped default bypasses that and runs the legacy adapter, which produces
    no bbox. This reads the resolved configuration and asserts the current value;
    a deployment that adopts the routing decision flips it and the test fails
    until the walkthrough's Defect 4 is retired.
    """
    from paperwhisperer.core import config

    assert config.PARSER_BACKEND == "legacy", (
        "PARSER_BACKEND is no longer 'legacy' — the default now uses the router; "
        "re-check whether citations get bboxes and retire walkthrough Defect 4"
    )


def test_the_reparse_catalog_offers_docling_even_though_the_default_refuses_it():
    """The reparse dropdown advertises a parser the shipped default cannot run.

    ``REPARSE_PARSER_OPTIONS`` offers 'docling' and 'docling_ocr', but with
    ``PARSER_BACKEND=legacy`` the gateway has rolled back to legacy and answers a
    docling reparse with 409 ``parser_unavailable``. The front end surfaces that
    honestly, so this is a "control advertises an option that cannot run"
    finding, not a silent failure.
    """
    from paperwhisperer.core import config

    options = re.findall(r"value:\s*'([^']+)'", (JS / "config.js").read_text(encoding="utf-8"))
    # Pull the reparse option values from the reparse block in config.js.
    config_text = (JS / "config.js").read_text(encoding="utf-8")
    reparse_block = config_text.split("REPARSE_PARSER_OPTIONS", 1)
    if len(reparse_block) < 2:
        pytest.skip("REPARSE_PARSER_OPTIONS not found in config.js")
    reparse_values = re.findall(r"value:\s*'([^']+)'", reparse_block[1][:600])
    assert "docling" in reparse_values, (
        "the reparse catalog no longer offers docling — Defect 4's premise changed"
    )
    # And the current default refuses it.
    assert config.PARSER_BACKEND == "legacy", (
        "with a non-legacy default the docling reparse would now succeed; "
        "re-check Defect 4"
    )
    _ = options
