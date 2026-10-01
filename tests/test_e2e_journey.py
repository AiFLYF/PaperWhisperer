"""The end-to-end journey, on a real document, through the real surfaces.

Every other test in this repository replaces one collaborator with a fake:
the parser with a fixture, the model with a stub, the store with a temporary
bundle that is trivially correct. That is right for unit tests and wrong for
this file, which exists to ask the one question none of them can: **does the
assembled product work when its parts, each written by a different author, are
finally put together on a real PDF?**

The journey is the product's own pitch: upload a paper, parse it, index it,
analyse it, ask a question whose answer is *late* in the document, get an answer
with claims and citations, click a citation and open the evidence, export it.
Nine steps, all through the real code paths.

**The load-bearing step is step 5.** The project was rebuilt around removing a
hard ``document_content[:12000]`` head excerpt from the Q&A path. Against this
paper, the sentence reporting its result (``67 test errors``) sits at offset
12132 — 132 characters past that window, inside the results section. A
head-excerpt implementation can never answer a question about it. So step 5
asks exactly that question, and this file asserts the answer *deterministically*,
against the retriever, with no model in the loop: the central claim must not be
gated behind a rate-limited endpoint. (The model-backed ``done`` answer is
asserted separately, and skips when the endpoint is unavailable.)

**What is honest about this file.** It is the test, not the repair. Where the
assembled journey breaks, the failing assertions below carry a precise
``reason`` and are marked ``xfail(strict=True)`` so a defect is *recorded* rather
than hidden, and so that fixing it turns the xpass into a failure that forces
the marker to be retired. The central claim is a plain assertion: it passes, and
that is the headline. Every model-backed test skips — never fails — when the
endpoint is down or rate-limited, so an outage cannot masquerade as a broken
product.

Run it with::

    python3 -m pytest tests/test_e2e_journey.py -q -p no:cacheprovider

The printable, runnable narration of the same journey is
``tools/e2e_smoke.py``; this file owns no production code and fixes nothing.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PDF = PROJECT_ROOT / "eval" / "corpus" / "digital_single_col" / "hinton_distillation.pdf"

#: The late evidence and the question that reaches for it. The string is at
#: offset 12132 in the 33880-char parse of this paper; the old head window cut
#: at 12000. This is the sentence the whole retrieval migration exists to make
#: reachable.
LATE_EVIDENCE = "67 test errors"
LATE_QUESTION = "How many test errors did the student network make on the test set?"

#: Corpus guard. A clean clone has no ``eval/corpus``; the journey is meaningless
#: without a real document, so skip with the reason rather than fail.
pytestmark = pytest.mark.skipif(
    not CORPUS_PDF.exists(),
    reason=(
        f"no real corpus document at {CORPUS_PDF}. The journey requires a real PDF; "
        "eval/corpus is gitignored and must be fed per docs/DEVELOPMENT_PLAN.md §6."
    ),
)

#: Credentials are snapshotted at *import* time, before conftest's autouse
#: ``isolated_environment`` fixture strips OPENAI_API_KEY for the session. This
#: is the house pattern (tests/test_llm_integration.py): the value is gone from
#: ``os.environ`` by the time a test body runs, so it has to be read during
#: collection.
_ENV = {name: os.environ.get(name, "") for name in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL")}

#: Markers that mean "the endpoint would not serve us right now". A rate limit
#: is included on purpose: 429 is a fact about a shared quota, and letting it
#: fail a test would report a broken product for a busy server. Mirrors
#: ``paperwhisperer.core.errors.is_retryable_llm_error``.
_UNAVAILABLE = (
    "429", "限流", "rate limit", "too many requests", "timeout", "超时",
    "connection", "连接失败", "无法连接", "502", "503", "504", "网关", "overloaded",
)


def _is_unavailable(text: str) -> bool:
    return any(marker in str(text or "").lower() for marker in _UNAVAILABLE)


def _completion_available() -> tuple[bool, str]:
    """Whether the endpoint can actually serve one chat completion right now.

    ``GET /models`` only proves the socket is open; this endpoint answers that
    while every completion is 429. So the probe makes the cheapest completion
    and reports whether one is truly available — the question the model-backed
    tests need answered before they bill a run.
    """
    import httpx

    base = _ENV["OPENAI_BASE_URL"].rstrip("/")
    if not base:
        return False, "OPENAI_BASE_URL is not set"
    try:
        response = httpx.post(
            f"{base}/chat/completions",
            headers={"Authorization": "Bearer " + _ENV["OPENAI_API_KEY"]},
            json={
                "model": _ENV["OPENAI_MODEL"],
                "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
                "max_tokens": 5,
            },
            timeout=30.0,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    if response.status_code == 200:
        return True, "a real completion returned 200"
    return False, f"completion returned HTTP {response.status_code}: {response.text[:120]}"


@pytest.fixture
def live_model(monkeypatch):
    """Restore the real credentials for one model-backed test, or skip it.

    The autouse ``isolated_environment`` fixture in conftest deletes
    ``OPENAI_API_KEY`` before every test; this puts the import-time snapshot
    back, serializes the shared semaphore so the pipeline is a courteous client
    of a rate-limited endpoint, and pins retries to 1 so an error path costs one
    call rather than three.
    """
    if not _ENV["OPENAI_API_KEY"] or not _ENV["OPENAI_BASE_URL"] or not _ENV["OPENAI_MODEL"]:
        pytest.skip(
            "no live LLM configured (need OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL "
            "in the project .env). The model-backed journey steps are opt-in and skip cleanly."
        )
    available, detail = _completion_available()
    if not available:
        pytest.skip(
            f"LLM endpoint {_ENV['OPENAI_BASE_URL']!r} cannot serve a completion ({detail}). "
            "This is an environment skip, not a product failure."
        )
    monkeypatch.setenv("OPENAI_API_KEY", _ENV["OPENAI_API_KEY"])
    from paperwhisperer.core import config

    monkeypatch.setattr(config, "OPENAI_MAX_RETRIES", 1)
    monkeypatch.setattr(config, "MAX_LLM_CONCURRENCY", 1)
    monkeypatch.setattr(config, "LLM_REQUEST_SEMAPHORE", threading.BoundedSemaphore(1))
    return dict(_ENV)


class Journey:
    """The assembled product, mid-journey, with each step's real result."""

    def __init__(self, client, bundle, document, tmp_root: Path) -> None:
        self.client = client
        self.bundle = bundle
        self.document = document
        self.tmp_root = tmp_root
        self.document_id = str(document["document_id"])
        self.resource = document
        self._retrieval = None

    @property
    def stored(self):
        record = self.bundle.metadata.get_document(self.document_id)
        assert record is not None and record.document is not None, "document not in the store"
        return record

    @property
    def chunks(self):
        return self.bundle.metadata.list_chunks(self.document_id)

    def late_retrieval(self):
        """The Q&A retrieval over the *indexed* document, via the real retriever.

        This is the deterministic heart of step 5 and the central claim of the
        project. It uses the same ``HybridRetriever`` the ``/v1`` Q&A route
        builds, over the same store the upload indexed, so it cannot pass in a
        way the product would not.
        """
        if self._retrieval is None:
            from paperwhisperer.analysis.orchestrator import build_qa_retriever
            from paperwhisperer.rag.retrieve import TASK_QA, plan_for

            retriever = build_qa_retriever(self.bundle, None, document=self.stored.document)
            self._retrieval = retriever.retrieve(
                LATE_QUESTION, plan=plan_for(TASK_QA), document_ids=[self.document_id]
            )
        return self._retrieval


@pytest.fixture(scope="module")
def journey():
    """Upload the corpus PDF once and keep the product mid-journey.

    Session-scoped on purpose: the instruction, and the whole cost argument, is
    to *reuse one parse across every assertion* rather than re-parsing a 9-page
    PDF per test. Every route, store, job worker and Q&A analyzer is wired to a
    single temporary tree through ``dependency_overrides``, so the workers the
    lifespan starts, the routes, and the retriever all see the same store — the
    property a real single-tier deployment has.
    """
    from fastapi.testclient import TestClient

    from paperwhisperer.analysis import orchestrator
    from paperwhisperer.api import routes_evidence, routes_jobs, routes_qa, routes_v1
    from paperwhisperer.api.routes_jobs import build_job_store
    from paperwhisperer.app import create_app
    from paperwhisperer.core import config
    from paperwhisperer.infra.object_store import build_object_store
    from paperwhisperer.infra.stores import build_sqlite_bundle

    with tempfile.TemporaryDirectory(prefix="pw-e2e-") as tmp:
        root = Path(tmp)
        for name in ("uploads", "output", "context"):
            (root / name).mkdir(parents=True, exist_ok=True)
        config.UPLOAD_FOLDER = str(root / "uploads")
        config.OUTPUT_FOLDER = str(root / "output")
        config.CONTEXT_FOLDER = str(root / "context")
        config.STORAGE_SQLITE_PATH = str(root / "context" / "stores" / "e2e.sqlite3")
        config.OBJECT_STORE_ROOT = str(root / "context" / "objects")
        # The page renderer reads originals from its own configured root; point
        # it at the same tree the upload wrote, or it answers
        # ``original_unavailable`` for bytes that are right there.
        config.PAGE_RENDER_ORIGINALS_DIR = str(root / "context" / "objects")

        bundle = build_sqlite_bundle(config)
        objects = build_object_store(config)
        app = create_app()
        job_store = build_job_store(bundle=bundle)
        app.dependency_overrides[routes_evidence.get_store_bundle] = lambda: bundle
        app.dependency_overrides[routes_jobs.get_job_store] = lambda: job_store
        app.dependency_overrides[routes_v1.get_object_store] = lambda: objects
        routes_qa.get_store_bundle = lambda: bundle
        orchestrator.get_store_bundle = lambda: bundle

        with TestClient(app) as client:
            with CORPUS_PDF.open("rb") as handle:
                response = client.post(
                    "/v1/documents",
                    files={"file": (CORPUS_PDF.name, handle, "application/pdf")},
                )
            assert response.status_code == 201, (
                f"uploading a real PDF returned HTTP {response.status_code}: {response.text[:300]}"
            )
            yield Journey(client, bundle, response.json()["document"], root)
        bundle.close()


# ---------------------------------------------------------------------------
# Steps 1-2: upload and parse provenance
# ---------------------------------------------------------------------------


def test_step1_upload_through_the_real_route_creates_a_document(journey):
    """Step 1: a real PDF, uploaded through POST /v1/documents, becomes a document.

    Not a function call — the route, with the real validation, the real object
    store write, the real parse and the real index.
    """
    assert journey.document_id.startswith("doc_"), journey.document_id
    assert journey.resource["page_count"] > 0, journey.resource
    assert journey.resource["char_count"] > 1000, "the parse produced almost no text"
    assert journey.chunks, "the upload indexed no chunks"


def test_step2_parse_records_which_parser_read_it(journey):
    """Step 2a: provenance names a parser and a status.

    A user who uploaded a paper is entitled to know which parser read it and
    whether the parse degraded.
    """
    parse = journey.resource["parse"]
    assert parse["parsed_by"], f"parse names no parser: {parse}"
    assert parse["status"], f"parse has no status: {parse}"
    assert parse["used_cloud"] is False, f"a local corpus PDF was sent to a cloud parser: {parse}"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "DEFECT (owner: parser routing default). PARSER_BACKEND defaults to 'legacy' "
        "(paperwhisperer/core/config.py:575), so the shipped default bypasses the gateway and "
        "runs the legacy adapter — which produces NO bbox and degrades even a clean digital PDF "
        "to status='partial' with the warning 'legacy PDF 解析只能定位到页，无法提供页内坐标（bbox）'. "
        "The bake-off (docs/PARSER_BAKE_OFF.md §6, strategy B) decided docling-first, and docling "
        "2.131.0 is installed and registered, but the default never uses it. Result: the product "
        "ships page-level provenance only, contrary to the Evidence-first promise. This test "
        "asserts a page locator that carries a bbox, which only the gateway/docling route gives."
    ),
)
def test_step2_parse_provides_page_and_bbox_provenance(journey):
    """Step 2b: elements carry a *bbox*, not just a page.

    A locator that stops at the page cannot be visually targeted, and the whole
    Evidence viewer rests on the bbox. Under the current default parser this
    fails, which is the point of recording it.
    """
    document = journey.stored.document
    pdf_elements = [e for e in document.elements if e.locator.kind == "pdf_page"]
    assert pdf_elements, "no pdf_page elements"
    with_bbox = [e for e in pdf_elements if e.locator.bbox]
    assert with_bbox, (
        f"none of the {len(pdf_elements)} pdf_page elements carry a bbox; the default parse is "
        "page-only, so no citation can be visually targeted"
    )


# ---------------------------------------------------------------------------
# Step 3: did vectors actually land?
# ---------------------------------------------------------------------------


def test_step3_vectors_actually_landed_in_the_real_store(journey):
    """Step 3: assert the vectors, not that the call returned.

    Reads every chunk's vector out of the store and checks the count and the
    dimension. A stored index that has no vectors is the state where the dense
    leg abstains on every question.
    """
    chunks = journey.chunks
    assert chunks, "no chunks were indexed"
    vectors = []
    for chunk in chunks:
        try:
            entry = journey.bundle.vectors.get_vector(chunk.chunk_id)
        except Exception:  # noqa: BLE001
            entry = None
        # ``get_vector`` returns a ``VectorEntry`` (the row, carrying the keys a
        # filtered search needs), so the vector itself is ``.vector`` — the
        # dimension assertion below reads the numbers, not the row.
        if entry is not None and entry.vector:
            vectors.append(entry.vector)
    assert vectors, (
        f"0 of {len(chunks)} chunks carry a vector: the upload path indexes lexically-only, "
        "so the dense leg never runs"
    )
    from paperwhisperer.core import config

    assert {len(v) for v in vectors} == {int(config.EMBEDDING_LOCAL_DIMENSION)}, (
        f"vector dimension(s) {sorted({len(v) for v in vectors})} != configured "
        f"{config.EMBEDDING_LOCAL_DIMENSION}"
    )


# ---------------------------------------------------------------------------
# Step 5: the central claim — the late passage is reachable
# ---------------------------------------------------------------------------


def test_step5_the_late_passage_is_reachable_by_retrieval(journey):
    """The central claim, asserted deterministically with no model in the loop.

    The head-excerpt window is gone. The sentence reporting the paper's result
    sits at offset 12132, past the old 12000 cut, and retrieval must surface it.
    If this passes, the fix the project was rebuilt around holds in the
    *assembled* system; if it fails, every unit test can still be green while
    the product cannot answer a question about its own results.
    """
    record = journey.stored
    full_text = record.document.to_full_text()
    assert LATE_EVIDENCE in full_text, "the corpus document no longer contains the evidence"
    offset = full_text.find(LATE_EVIDENCE)
    from paperwhisperer.core import config

    window = config.DOCUMENT_WINDOW_BUDGET["qa"]
    assert offset > window, (
        f"the fixture no longer exercises the late-evidence case: {LATE_EVIDENCE!r} is at "
        f"offset {offset}, inside the {window}-char head window"
    )

    result = journey.late_retrieval()
    assert not result.insufficient_evidence, (
        "retrieval found nothing, but the evidence is in the document — the head excerpt is "
        "gone and something else is dropping it"
    )
    joined = " ".join(item.text for item in result.evidence)
    assert LATE_EVIDENCE in joined, (
        f"retrieval returned {len(result.evidence)} passage(s) but none contains {LATE_EVIDENCE!r}. "
        f"The passage sits at offset {offset}, past the old {window}-char window; a head-excerpt "
        "implementation could not reach it, so retrieval failing here means the central fix does "
        "not hold in the assembled system."
    )


# ---------------------------------------------------------------------------
# Step 7: quote verification and degradation (deterministic)
# ---------------------------------------------------------------------------


def _first_evidence(journey):
    from paperwhisperer.ingestion.schema import Evidence

    chunk = journey.chunks[0]
    element = next((e for e in journey.stored.document.elements if e.text), None)
    return chunk, Evidence(
        evidence_id="ev_journey",
        chunk_id=chunk.chunk_id,
        text=chunk.text,
        locator=element.locator if element else None,
    )


def test_step7_a_real_quote_verifies_and_a_fabrication_degrades(journey):
    """Step 7: the release gate, on the real document.

    A span the paper actually contains aligns and is quotable. A number the
    paper never states does not align and is degraded to a paraphrase carrying
    no verified text — so there is no state in which an unverified string can be
    rendered as though the paper said it.
    """
    from paperwhisperer.analysis.orchestrator import AnswerClaim, _verify_claim

    chunk, evidence = _first_evidence(journey)
    real_span = (chunk.text or "")[:60].strip()
    assert real_span, "the first chunk has no text to quote"

    aligned = _verify_claim(AnswerClaim(text="t", quote=real_span), [evidence], journey.stored.document)
    assert aligned.presentation.value == "direct_quote", (
        f"a real span did not verify: outcome={aligned.outcome!r} message={aligned.message!r}"
    )
    assert aligned.verified_text, "a verified direct quote carries no verified text"

    fabricated = _verify_claim(
        AnswerClaim(text="t", quote="the student made 999999 errors, a number not in the paper"),
        [evidence],
        journey.stored.document,
    )
    assert fabricated.presentation.value == "paraphrase", (
        f"a fabricated quote was NOT degraded: presentation={fabricated.presentation!r} "
        f"verified_text={fabricated.verified_text!r}"
    )
    assert not fabricated.verified_text, "a degraded paraphrase still carries verified text"
    assert not fabricated.is_direct_quote, "a degraded paraphrase is marked quotable"


# ---------------------------------------------------------------------------
# Step 8: citation click -> Evidence API -> page image
# ---------------------------------------------------------------------------


def test_step8_the_page_image_renders_for_a_citation(journey):
    """Step 8b: the renderer works — addressed by the underlying chunk id.

    Proves the Evidence API and the page rasteriser function on the real
    document. It addresses the chunk id, the other of the two namespaces the
    route resolves; the retriever's ``ev_`` id is covered by step 8a.
    """
    chunk, _evidence = _first_evidence(journey)
    response = journey.client.get(
        f"/v1/documents/{journey.document_id}/evidence/{chunk.chunk_id}"
    )
    assert response.status_code == 200, f"evidence did not resolve: {response.text[:200]}"
    payload = response.json()
    page = (payload.get("locator") or {}).get("page")
    assert page, f"evidence has no page to render: {payload.get('locator')}"
    image = journey.client.get(f"/v1/documents/{journey.document_id}/pages/{page}/image")
    assert image.status_code == 200, f"page image did not render: {image.text[:200]}"
    assert image.content[:4] == b"\x89PNG", "the page route did not return a PNG"


def test_step8_a_citation_click_resolves_through_the_evidence_api(journey):
    """Step 8a: a citation carrying its evidence_id opens the passage.

    The citation a reader clicks carries the ``ev_`` id the retriever produced.
    Opening it must resolve to a passage — the identifier namespaces the
    retriever and the API used to disagree on, which made every real click a
    404 while the evidence suite stayed green on ``chunk_*`` ids alone.
    """
    evidence = journey.late_retrieval().evidence[0]
    response = journey.client.get(
        f"/v1/documents/{journey.document_id}/evidence/{evidence.evidence_id}"
    )
    assert response.status_code == 200, (
        f"a citation click carrying evidence_id {evidence.evidence_id} returned "
        f"HTTP {response.status_code} ({response.json().get('code')})"
    )
    assert response.json().get("locator") is not None, "the evidence has no locator"


# ---------------------------------------------------------------------------
# Steps 4, 5-live, 6, 9: model-backed. Skip cleanly when the endpoint is down.
# ---------------------------------------------------------------------------


def test_step4_analysis_reports_section_statuses_and_partial_consistently(journey, live_model):
    """Step 4: analysis through the real pipeline with the real model.

    A full analysis is the most expensive step, so the optional sections are
    turned off to keep the journey re-runnable. What is asserted is the
    bookkeeping: the required sections succeed, and the reported quality
    matches whether any section actually failed (a ``partial`` is reported as
    partial, never as a clean success).
    """
    with CORPUS_PDF.open("rb") as handle:
        response = journey.client.post(
            "/api/analyze",
            files={"file": (CORPUS_PDF.name, handle, "application/pdf")},
            data={
                "api_key": live_model["OPENAI_API_KEY"],
                "generate_mermaid": "false",
                "generate_evaluation": "false",
                "generate_research_brief": "false",
            },
        )
    if response.status_code != 200:
        if _is_unavailable(response.text):
            pytest.skip(f"analysis could not complete; endpoint unavailable: {response.text[:200]}")
        pytest.fail(f"analysis returned HTTP {response.status_code}: {response.text[:300]}")
    result = response.json()
    statuses = {name: (sec or {}).get("status") for name, sec in (result.get("sections") or {}).items()}
    for required in ("summary", "quotes", "mindmap"):
        assert statuses.get(required) == "success", f"{required} did not succeed: {statuses}"
    quality = (result.get("analysis_status") or {}).get("quality")
    expected = "partial" if any(v == "failed" for v in statuses.values()) else "complete"
    assert quality == expected, f"quality={quality} but statuses imply {expected}: {statuses}"
    assert result["parse"]["parsed_by"], "analysis returned no parse provenance"


def test_step5_live_answer_to_the_late_question_carries_claims_and_citations(journey, live_model):
    """Steps 5-live + 6: the generated answer is structured and reaches the late fact.

    This is the model-backed counterpart to the deterministic reachability test:
    the passage is *found* without a model, and here the model is asked a
    question about it and must answer with claims, evidence ids, citations and a
    confidence that matches what the claims actually support.
    """
    def _sse(text: str) -> list[tuple[str, dict]]:
        events: list[tuple[str, dict]] = []
        for block in text.split("\n\n"):
            name, data_lines = None, []
            for line in block.splitlines():
                if line.startswith("event:"):
                    name = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[len("data:"):].strip())
            if name is None:
                continue
            raw = "\n".join(data_lines)
            try:
                events.append((name, json.loads(raw) if raw else {}))
            except json.JSONDecodeError:
                events.append((name, {"_raw": raw}))
        return events

    response = journey.client.post(
        f"/v1/documents/{journey.document_id}/qa",
        json={"question": LATE_QUESTION, "api_key": live_model["OPENAI_API_KEY"]},
    )
    if response.status_code != 200:
        if _is_unavailable(response.text):
            pytest.skip(f"Q&A could not run; endpoint unavailable: {response.text[:200]}")
        pytest.fail(f"Q&A returned HTTP {response.status_code}: {response.text[:300]}")
    events = _sse(response.text)
    done = [payload for name, payload in events if name == "done"]
    if not done:
        detail = json.dumps([p for n, p in events if n == "error"], ensure_ascii=False)[:300]
        if _is_unavailable(detail):
            pytest.skip(f"Q&A stream carried an endpoint error: {detail}")
        pytest.fail(f"no 'done' event in stream {[n for n, _ in events]}: {detail}")
    answer_result = done[0]
    answer = answer_result.get("answer", "")
    assert LATE_EVIDENCE in answer or "67" in answer, (
        f"the late evidence {LATE_EVIDENCE!r} did not reach the answer: {answer[:300]!r}"
    )
    claims = answer_result.get("claims") or []
    assert claims, "the answer carries no claims"
    assert any(claim.get("evidence_ids") for claim in claims), "no claim names an evidence id"
    confidence = answer_result.get("confidence")
    unbacked = [
        claim for claim in claims
        if not claim.get("evidence_ids")
        or (claim.get("quote") and claim.get("presentation") != "direct_quote")
    ]
    if confidence == "supported":
        assert not unbacked, f"confidence 'supported' but {len(unbacked)} claims are unbacked"
    for claim in claims:
        if claim.get("presentation") == "paraphrase":
            assert not claim.get("uses_quotation_marks"), "a paraphrase carries quotation marks"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "DEFECT (partially fixed; owner: analysis route wiring). The transport half of D3 is "
        "CLOSED and verified: the exported report now carries whatever the analysis actually "
        "produced. finalize_analysis_result (analysis/service.py) no longer lets the session drop the "
        "keys /api/export reads -- _session_payload_carrying_provenance places parse/analysis_run/"
        "citations/claims at the session top level (the exact contract routes_documents."
        "_export_result_from_session lifts) and hands document_id/paper_facts to "
        "build_session_payload through its own kwargs, so a run that DOES carry paper_facts/"
        "analysis_run now renders both '生成条件与溯源' and '引用与证据链接' (verified with a stubbed run). "
        "What remains is the recording half: the job's _analyse_stage (api/routes_jobs.py:586) still "
        "calls finalize_analysis_result(run=None, report=None) -- the legacy six-section path -- so the "
        "one analysis route the product exposes produces no paper_facts/analysis_run for the export to "
        "carry. The only honest producer of a run is analysis.service.analyze_with_fact_layer (the "
        "client tracks no token usage), which needs the document indexed inside the job and changes "
        "/api/analyze's output, cost and streaming -- an explicit migration the codebase deliberately "
        "gated ('a one-line change at the call site ... so that the two can be compared on the same "
        "document rather than one replacing the other unmeasured'). That swap is a product decision, "
        "not a composition repair, and cannot be validated without a live model, so it is left for that "
        "decision rather than made here. Until it lands, the real /api/analyze export is still a shell "
        "-- correctly so, because that run recorded no run to report."
    ),
)
def test_step9_export_carries_the_provenance(journey, live_model):
    """Step 9: the export records what produced it, and never quotes a paraphrase.

    Driven through the real ``/api/export/{session}`` for a session a real
    analysis created. The export must carry a provenance section. The session
    no longer drops the provenance (that half is fixed); what is still missing is
    that the one analysis route the product exposes runs the legacy path and
    records no run for the export to carry. See the reason.
    """
    with CORPUS_PDF.open("rb") as handle:
        response = journey.client.post(
            "/api/analyze",
            files={"file": (CORPUS_PDF.name, handle, "application/pdf")},
            data={
                "api_key": live_model["OPENAI_API_KEY"],
                "generate_mermaid": "false",
                "generate_evaluation": "false",
                "generate_research_brief": "false",
            },
        )
    if response.status_code != 200:
        if _is_unavailable(response.text):
            pytest.skip(f"analysis could not complete; endpoint unavailable: {response.text[:200]}")
        pytest.fail(f"analysis returned HTTP {response.status_code}: {response.text[:300]}")
    result = response.json()
    export = journey.client.get(
        f"/api/export/{result['session_id']}",
        params={"session_token": result["session_token"], "format": "markdown"},
    )
    assert export.status_code == 200, f"export returned HTTP {export.status_code}"
    body = export.text
    assert "生成条件与溯源" in body, (
        "the exported markdown has no provenance section; the analysis route records no "
        f"analysis_run and the session drops the fact layer. Export was {len(body)} bytes."
    )
