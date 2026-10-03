#!/usr/bin/env python3
"""Perform the end-to-end research journey and print what actually happened.

This is the runnable half of ``tests/test_e2e_journey.py``. Where the pytest file
asserts, this tool *narrates*: it drives the same steps through the same real
code paths — the real HTTP routes, the real parser, the real store, the real
retriever, the real verifier, the real renderer — and prints, for each, what was
made, what came back, and, when the product breaks, a ``DEFECT`` line naming
what was expected and what arrived.

**The load-bearing split.** The steps that need a model (the analysis, the
generated answer text) degrade to a reported ``SKIP`` whenever the endpoint is
down, rate-limited or slow. The steps that do *not* need a model — parse
provenance, whether vectors landed, whether the late passage is reachable by
retrieval, whether a quote verifies, whether the Evidence API and the page
renderer resolve — always run. That split is deliberate: the central claim of
this project (the head-excerpt window is gone, so a passage at offset ~12132 is
findable) is a property of *retrieval*, and a journey that could only assert it
through a rate-limited model would let an outage hide the one thing the file
exists to check.

Run it with::

    python3 tools/e2e_smoke.py            # full journey
    python3 tools/e2e_smoke.py --no-llm    # only the steps that need no model

Nothing here installs anything or starts a database. Every write goes to a
temporary directory, and the temporary tree is removed on exit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CORPUS_PDF = PROJECT_ROOT / "eval" / "corpus" / "digital_single_col" / "hinton_distillation.pdf"

#: The question whose answer sits late in the paper. The evidence — the string
#: ``67 test errors`` — is at offset 12132 in a 33880-character document, i.e.
#: 132 characters past the old 12000-character head-excerpt window, inside the
#: results section. It is the one question a head-excerpt implementation cannot
#: answer and the reason the retrieval migration exists.
LATE_EVIDENCE = "67 test errors"
LATE_QUESTION = "How many test errors did the student network make on the test set?"

#: Substrings that mean "the endpoint would not serve us right now" rather than
#: "our code is wrong". Mirrors ``paperwhisperer.core.errors``. A rate limit is
#: in this list on purpose: a 429 is a fact about a shared quota, and letting it
#: fail the journey would report a broken product for a busy server.
MODEL_UNAVAILABLE = (
    "429", "限流", "rate limit", "too many requests",
    "timeout", "超时", "connection", "连接失败", "无法连接",
    "502", "503", "504", "网关", "overloaded",
)


def _is_model_unavailable(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(marker in lowered for marker in MODEL_UNAVAILABLE)


def _header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78, flush=True)


class Reporter:
    def __init__(self) -> None:
        self.defects: list[str] = []
        self.skips: list[str] = []
        self.passes: list[str] = []
        self.started = time.time()

    def passed(self, step: str, detail: str = "") -> None:
        line = f"PASS  {step}" + (f" — {detail}" if detail else "")
        self.passes.append(line)
        print("  " + line, flush=True)

    def defect(self, step: str, detail: str) -> None:
        line = f"DEFECT  {step} — {detail}"
        self.defects.append(line)
        print("  !! " + line, flush=True)

    def skipped(self, step: str, reason: str) -> None:
        line = f"SKIP   {step} — {reason}"
        self.skips.append(line)
        print("  -- " + line, flush=True)

    def note(self, text: str) -> None:
        print("      " + text, flush=True)

    def elapsed(self) -> float:
        return time.time() - self.started

    def summary(self) -> None:
        _header("JOURNEY SUMMARY")
        for line in self.passes:
            print("  " + line)
        for line in self.skips:
            print("  " + line)
        print()
        if self.defects:
            print(f"  {len(self.defects)} DEFECT(S):")
            for line in self.defects:
                print("    " + line)
        else:
            print("  No defects among the steps that ran.")
        print(f"\n  wall clock: {self.elapsed():.1f}s")


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def prepare_environment(tmp_root: Path, *, serialize_llm: bool) -> dict:
    """Point every runtime folder and store at a temporary tree.

    The bundle and the object store are built *here* and injected through
    FastAPI's ``dependency_overrides`` so the lifespan's workers, the routes and
    the Q&A analyzer all resolve the same store — the property a real single-tier
    deployment has, and one a test must not break by building three stores.
    """
    from paperwhisperer.core import config
    from paperwhisperer.infra.object_store import build_object_store
    from paperwhisperer.infra.stores import build_sqlite_bundle

    uploads = tmp_root / "uploads"
    output = tmp_root / "output"
    context = tmp_root / "context"
    for folder in (uploads, output, context):
        folder.mkdir(parents=True, exist_ok=True)

    config.UPLOAD_FOLDER = str(uploads)
    config.OUTPUT_FOLDER = str(output)
    config.CONTEXT_FOLDER = str(context)
    config.STORAGE_SQLITE_PATH = str(context / "stores" / "e2e.sqlite3")
    config.OBJECT_STORE_ROOT = str(context / "objects")
    # The page-render route reads originals from its own configured root; point
    # it at the same tree the upload wrote, or the render answers
    # ``original_unavailable`` for bytes that are right there.
    config.PAGE_RENDER_ORIGINALS_DIR = str(context / "objects")

    if serialize_llm:
        # The shared endpoint rate-limits bursts. Serialize the pipeline's model
        # calls so a run of this journey is a courteous client of a shared
        # quota; this is a deployment knob, not a product change.
        config.MAX_LLM_CONCURRENCY = 1
        config.LLM_REQUEST_SEMAPHORE = threading.BoundedSemaphore(1)

    bundle = build_sqlite_bundle(config)
    objects = build_object_store(config)
    return {"bundle": bundle, "objects": objects, "config": config}


def make_client(bundle, objects):
    from fastapi.testclient import TestClient

    from paperwhisperer.analysis import orchestrator
    from paperwhisperer.api import routes_evidence, routes_jobs, routes_qa, routes_v1
    from paperwhisperer.api.routes_jobs import build_job_store
    from paperwhisperer.app import create_app

    app = create_app()
    job_store = build_job_store(bundle=bundle)

    app.dependency_overrides[routes_evidence.get_store_bundle] = lambda: bundle
    app.dependency_overrides[routes_jobs.get_job_store] = lambda: job_store
    app.dependency_overrides[routes_v1.get_object_store] = lambda: objects
    # routes_qa captured ``get_store_bundle`` at import; point it at our bundle
    # so build_qa_analyzer never opens the developer's real sqlite file.
    routes_qa.get_store_bundle = lambda: bundle
    orchestrator.get_store_bundle = lambda: bundle

    return TestClient(app), app


def resolve_api_key(config) -> str:
    return str(config.resolve_api_key("") or os.environ.get("OPENAI_API_KEY", ""))


def chat_completion_available(config) -> tuple[bool, str]:
    """Can the endpoint actually serve a *chat completion* right now?

    The reachability probe in the house pattern is ``GET /models``, which only
    proves the socket is open. That is not enough here: this endpoint serves
    ``/models`` happily while every completion is 429. So this probe makes the
    cheapest possible completion and reports whether one is actually available,
    which is the question the model-backed steps need answered.
    """
    import httpx

    base = str(getattr(config, "OPENAI_BASE_URL", "")).rstrip("/")
    if not base:
        return False, "OPENAI_BASE_URL is not set"
    try:
        response = httpx.post(
            f"{base}/chat/completions",
            headers={"Authorization": "Bearer " + os.environ.get("OPENAI_API_KEY", "")},
            json={
                "model": str(getattr(config, "OPENAI_MODEL", "")),
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


def read_sse(text: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
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
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            payload = {"_raw": raw}
        events.append((name, payload))
    return events


# ---------------------------------------------------------------------------
# The journey
# ---------------------------------------------------------------------------


def run_journey(rep: Reporter, *, use_llm: bool) -> None:
    if not CORPUS_PDF.exists():
        rep.defect("journey-setup", f"corpus PDF not found at {CORPUS_PDF}")
        return

    from paperwhisperer.core import config  # noqa: F401 (loads .env at import)

    with tempfile.TemporaryDirectory(prefix="pw-e2e-") as tmp:
        env = prepare_environment(Path(tmp), serialize_llm=use_llm)
        config = env["config"]
        bundle, objects = env["bundle"], env["objects"]
        client, _app = make_client(bundle, objects)

        api_key = resolve_api_key(config)
        model_up, model_detail = (False, "--no-llm")
        if use_llm:
            model_up, model_detail = chat_completion_available(config)
            rep.note(f"endpoint {config.OPENAI_BASE_URL} model {config.OPENAI_MODEL}: "
                     f"{model_detail}")
            if not model_up:
                rep.skipped("llm", f"endpoint cannot serve a completion ({model_detail}); "
                                   "model-backed steps will skip (environment, not product)")

        document_id = ""
        session_id = ""
        session_token = ""
        answer_result = None

        # -- Step 1: upload through the real HTTP route --------------------
        _header("STEP 1  upload a real PDF through POST /v1/documents")
        with CORPUS_PDF.open("rb") as handle:
            response = client.post(
                "/v1/documents",
                files={"file": ("hinton_distillation.pdf", handle, "application/pdf")},
            )
        rep.note(f"POST /v1/documents -> HTTP {response.status_code}")
        if response.status_code != 201:
            rep.defect("step1-upload", f"expected 201, got {response.status_code}: "
                                    f"{response.text[:300]}")
            return
        created = response.json()
        document_id = created["document"]["document_id"]
        resource = created["document"]
        rep.passed("step1-upload", f"document_id={document_id}")

        # -- Step 2: parse provenance ---------------------------------------
        _header("STEP 2  assert parse provenance (parser, page/bbox coverage)")
        parse = resource.get("parse", {})
        rep.note(json.dumps(
            {k: parse.get(k) for k in
             ("status", "parsed_by", "media_type", "used_cloud", "cloud_attempted",
              "fallback_used")},
            ensure_ascii=False,
        ))
        if not parse.get("parsed_by") or not parse.get("status"):
            rep.defect("step2-provenance", f"parse report incomplete: {parse}")
        else:
            rep.passed("step2-provenance", f"parsed_by={parse.get('parsed_by')} "
                                     f"status={parse.get('status')}")
            if str(parse.get("parsed_by")) == "legacy" and str(config.PARSER_PRIMARY) == "docling":
                rep.note("note: PARSER_PRIMARY=docling but the parse ran the legacy "
                         "adapter (PARSER_BACKEND defaults to 'legacy') — see "
                         "PARSER_BAKE_OFF routing note.")

        pages = client.get(f"/v1/documents/{document_id}/pages")
        if pages.status_code != 200:
            rep.defect("step2-pages", f"GET /pages -> {pages.status_code}")
        else:
            page_body = pages.json()
            rep.note(f"locator_kind={page_body.get('locator_kind')} "
                     f"page_count={page_body.get('page_count')} "
                     f"listed={len(page_body.get('pages') or [])}")
            if not page_body.get("pages"):
                rep.defect("step2-pages", "no page descriptors; provenance has no "
                                      "page/bbox coverage")
            else:
                rep.passed("step2-pages", f"{len(page_body['pages'])} pages, "
                                         f"kind={page_body.get('locator_kind')}")

        # -- Step 3: did vectors actually land? -----------------------------
        _header("STEP 3  assert vectors landed in the real store (dimension + count)")
        record = bundle.metadata.get_document(document_id)
        chunks = bundle.metadata.list_chunks(document_id) if record else []
        vector_count, dims = 0, set()
        for chunk in chunks:
            try:
                vector = bundle.vectors.get_vector(chunk.chunk_id)
            except Exception:
                vector = None
            if vector:
                vector_count += 1
                # get_vector returns a VectorEntry row; the numbers are .vector.
                dims.add(len(vector.vector))
        rep.note(f"chunks={len(chunks)} vectors_present={vector_count} dims={sorted(dims)}")
        if not chunks:
            rep.defect("step3-index", "no chunks were indexed for the uploaded document")
        elif vector_count == 0:
            rep.defect(
                "step3-index",
                "0 vectors landed: the upload path resolved no embedding provider, so the dense "
                "leg never runs (index_document_for_retrieval now falls back to the configured "
                "provider when the caller passes none; a 0 here means this deployment has none)",
            )
        else:
            rep.passed("step3-index", f"{vector_count}/{len(chunks)} vectors, dim={sorted(dims)}")

        # -- Step 4: analysis through the real pipeline ---------------------
        _header("STEP 4  analyse through the real pipeline (POST /api/analyze)")
        if not use_llm or not model_up or not api_key:
            rep.skipped("step4-analysis", "no usable model right now; "
                                      "section statuses not exercised")
        else:
            with CORPUS_PDF.open("rb") as handle:
                analysis_response = client.post(
                    "/api/analyze",
                    files={"file": ("hinton_distillation.pdf", handle, "application/pdf")},
                    data={
                        "api_key": api_key,
                        "generate_mermaid": "false",
                        "generate_evaluation": "false",
                        "generate_research_brief": "false",
                    },
                )
            rep.note(f"POST /api/analyze -> HTTP {analysis_response.status_code}")
            body_text = analysis_response.text
            if analysis_response.status_code != 200:
                if _is_model_unavailable(body_text):
                    rep.skipped("step4-analysis", f"endpoint rate-limited mid-analysis: "
                                         f"{body_text[:120]}")
                else:
                    rep.defect("step4-analysis", f"expected 200, got "
                                           f"{analysis_response.status_code}: {body_text[:300]}")
            else:
                analysis = analysis_response.json()
                session_id = str(analysis.get("session_id") or "")
                session_token = str(analysis.get("session_token") or "")
                statuses = {name: (sec or {}).get("status")
                            for name, sec in (analysis.get("sections") or {}).items()}
                rep.note("section statuses: "
                         + json.dumps(statuses, ensure_ascii=False, sort_keys=True))
                rep.note("analysis_status.quality="
                         f"{(analysis.get('analysis_status') or {}).get('quality')}")
                required_ok = all(
                    (analysis.get("sections") or {}).get(name, {}).get("status") == "success"
                    for name in ("summary", "quotes", "mindmap")
                )
                if required_ok:
                    rep.passed("step4-analysis", "summary/quotes/mindmap all success")
                else:
                    rep.defect("step4-analysis", f"a required section did not succeed: {statuses}")
                quality = (analysis.get("analysis_status") or {}).get("quality")
                failed_any = any(v == "failed" for v in statuses.values())
                expected = "partial" if failed_any else "complete"
                if quality == expected:
                    rep.passed("step4-partial", f"quality={quality} matches statuses")
                else:
                    rep.defect("step4-partial", f"quality={quality} but statuses imply "
                                          f"{expected}: {statuses}")

        # -- Step 5: the late-evidence question -----------------------------
        _header(f"STEP 5  the LATE passage ({LATE_EVIDENCE!r}, offset ~12132) is reachable")
        evidence_pairs = retrieve_for_late_question(bundle, document_id, rep)
        first_pair = evidence_pairs[0] if evidence_pairs else ("", "")
        if use_llm and model_up and api_key:
            qa_response = client.post(
                f"/v1/documents/{document_id}/qa",
                json={"question": LATE_QUESTION, "api_key": api_key},
            )
            rep.note(f"POST /v1/documents/{{id}}/qa -> HTTP {qa_response.status_code}")
            if qa_response.status_code != 200:
                if _is_model_unavailable(qa_response.text):
                    rep.skipped("step5-answer", "endpoint could not answer; "
                                         "retrieval reachability still checked")
                else:
                    rep.defect("step5-answer", f"expected 200, got {qa_response.status_code}: "
                                        f"{qa_response.text[:200]}")
            else:
                events = read_sse(qa_response.text)
                rep.note("SSE events: " + ", ".join(name for name, _ in events))
                done = [payload for name, payload in events if name == "done"]
                if not done:
                    err = [p for name, p in events if name == "error"]
                    detail = (json.dumps(err[0], ensure_ascii=False)[:200]
                                 if err else "[no done event]")
                    if _is_model_unavailable(detail):
                        rep.skipped("step5-answer", f"stream carried an error: {detail}")
                    else:
                        rep.defect("step5-answer",
                                   f"no 'done' event; stream was "
                                   f"{[n for n, _ in events]}: {detail}")
                else:
                    answer_result = done[0]
                    answer = answer_result.get("answer", "")
                    rep.note(f"answer: {answer[:280]!r}")
                    rep.note(f"confidence={answer_result.get('confidence')} "
                             f"evidence_count="
                             f"{(answer_result.get('retrieval') or {}).get('evidence_count')}")
                    if LATE_EVIDENCE in answer or "67" in answer:
                        rep.passed("step5-answer",
                                 "the late fact (67 test errors) is in the answer")
                    else:
                        rep.defect(
                            "step5-answer",
                            f"the late evidence {LATE_EVIDENCE!r} did not reach the answer. "
                            f"answer={answer[:200]}",
                        )
        else:
            rep.skipped("step5-answer", "no usable model right now; "
                                         "retrieval reachability still checked")

        # -- Step 6: answer structure ---------------------------------------
        _header("STEP 6  assert claims + evidence ids + citations + confidence")
        if answer_result is None:
            rep.skipped("step6-structure", "no live answer to inspect")
        else:
            claims = answer_result.get("claims") or []
            citations = answer_result.get("citations") or []
            rep.note(f"claims={len(claims)} citations={len(citations)} "
                         f"confidence={answer_result.get('confidence')}")
            if not claims:
                rep.defect("step6-structure", "answer carries no claims")
            elif not any(c.get("evidence_ids") for c in claims):
                rep.defect("step6-structure", "no claim names an evidence id")
            else:
                rep.passed("step6-structure", f"{len(claims)} claims, {len(citations)} citations")
            confidence = answer_result.get("confidence")
            unverified = [c for c in claims
                          if not c.get("evidence_ids")
                          or (c.get("quote") and c.get("presentation") != "direct_quote")]
            if confidence == "supported" and unverified:
                rep.defect("step6-confidence",
                           f"supported but {len(unverified)} claims are unbacked")
            else:
                rep.passed("step6-confidence", f"confidence={confidence} consistent with claims")

        # -- Step 7: quote verification / degradation ------------------------
        _header("STEP 7  a real quote verifies; a fabricated one degrades (no quote marks)")
        verify_quote_gate(rep, bundle, document_id, answer_result)

        # -- Step 8: citation -> Evidence API -> page image ------------------
        _header("STEP 8  click a citation: Evidence API + page image render")
        evidence_id, chunk_id = first_pair
        if answer_result and answer_result.get("citations"):
            evidence_id = answer_result["citations"][0].get("evidence_id") or evidence_id
            chunk_id = answer_result["citations"][0].get("chunk_id") or chunk_id
        if not evidence_id:
            rep.skipped("step8-evidence", "no citation/evidence id to resolve")
        else:
            # (a) Address the Evidence API with the id the citation/UI actually
            #     carries (the ev_ evidence_id). This is the real click.
            click = client.get(f"/v1/documents/{document_id}/evidence/{evidence_id}")
            rep.note(f"(real click) GET evidence/{evidence_id} -> HTTP {click.status_code}")
            if click.status_code == 200:
                rep.passed("step8-evidence", f"the citation's evidence_id {evidence_id} resolved")
                click_payload = click.json()
            else:
                rep.defect(
                    "step8-evidence",
                    f"a citation click carrying evidence_id {evidence_id} returns HTTP "
                    f"{click.status_code} ({click.json().get('code')}). The retriever and the "
                    "front end both emit ev_* ids, but the Evidence API resolves its path "
                    "parameter as a chunk id (routes_evidence._evidence_payload calls "
                    "get_chunk(evidence_id)). Every evidence test passes because they only "
                    "ever address chunk_* ids, so the click 404s only in composition.",
                )
                click_payload = None

            # (b) Address the same route by the underlying chunk id, to prove the
            #     Evidence API and the renderer themselves work when addressed
            #     correctly — isolating the defect to the id contract above.
            if chunk_id:
                by_chunk = client.get(f"/v1/documents/{document_id}/evidence/{chunk_id}")
                rep.note(f"(by chunk id) GET evidence/{chunk_id} -> HTTP {by_chunk.status_code}")
                if by_chunk.status_code == 200:
                    payload = click_payload or by_chunk.json()
                    locator = payload.get("locator") or {}
                    page = locator.get("page")
                    snippet = payload.get("snippet")
                    rep.note(f"locator={locator} snippet_len="
                                f"{len(snippet) if isinstance(snippet, str) else 'n/a'}")
                    if page:
                        image = client.get(f"/v1/documents/{document_id}/pages/{page}/image")
                        rep.note(f"GET page image -> HTTP {image.status_code} "
                                 f"type={image.headers.get('content-type')} "
                                 f"bytes={len(image.content)}")
                        if image.status_code == 200 and image.content[:4] == b"\x89PNG":
                            rep.passed("step8-page-image", f"page {page} rendered a real PNG "
                                                     f"({len(image.content)} bytes)")
                        else:
                            rep.defect("step8-page-image", f"page image did not render: HTTP "
                                                          f"{image.status_code} {image.text[:200]}")
                    else:
                        rep.skipped("step8-page-image", f"locator has no page "
                                                      f"(kind={locator.get('kind')})")
                else:
                    rep.defect("step8-evidence", "evidence did not resolve even by "
                                                  "chunk id: "
                                                  f"HTTP {by_chunk.status_code} "
                                                  f"{by_chunk.text[:200]}")

        # -- Step 9: export ---------------------------------------------------
        _header("STEP 9  export carries provenance and never quotes a paraphrase")
        if not session_id or not session_token:
            rep.skipped("step9-export", "no analysis session to export (analysis was skipped)")
        else:
            export = client.get(
                f"/api/export/{session_id}",
                params={"session_token": session_token, "format": "markdown"},
            )
            rep.note(f"GET /api/export/{session_id} -> HTTP {export.status_code}")
            if export.status_code != 200:
                rep.defect("step9-export", f"HTTP {export.status_code}: {export.text[:200]}")
            else:
                body = export.text
                has_provenance = "生成条件与溯源" in body
                has_citations = "引用与证据链接" in body
                rep.note(f"export bytes={len(body)} provenance_section={has_provenance} "
                         f"citation_section={has_citations}")
                if has_provenance:
                    rep.passed("step9-export", "export carries a provenance section")
                else:
                    rep.defect(
                        "step9-export",
                        "the exported markdown has no provenance section. "
                        "finalize_analysis_result attaches paper_facts/"
                        "analysis_run to the API result, but sessions.store does "
                        "not persist them, so /api/export (which reads the "
                        "session) loses them",
                    )
        client.close()


def retrieve_for_late_question(bundle, document_id, rep) -> list[str]:
    """Deterministic reachability of the late passage over the real store.

    This is the central claim, and it is a property of retrieval, so it is
    checked against the *indexed* document through the real ``HybridRetriever``
    — the same object the ``/v1`` Q&A route builds — rather than through a
    model. No completion, no rate limit, no flake.
    """
    from paperwhisperer.analysis.orchestrator import build_qa_retriever
    from paperwhisperer.rag.retrieve import TASK_QA, plan_for

    record = bundle.metadata.get_document(document_id)
    if record is None or record.document is None:
        rep.defect("step5-reach", "the uploaded document is not in the store")
        return []
    retriever = build_qa_retriever(bundle, None, document=record.document)
    result = retriever.retrieve(
        LATE_QUESTION, plan=plan_for(TASK_QA), document_ids=[document_id]
    )
    joined = " ".join(item.text for item in result.evidence)
    rep.note(f"retrieved {len(result.evidence)} evidence item(s); "
             f"insufficient={result.insufficient_evidence}")
    if result.insufficient_evidence:
        rep.defect("step5-reach", "retrieval found nothing, but the late "
                                 "evidence is in the document")
        return []
    if LATE_EVIDENCE not in joined:
        rep.defect(
            "step5-reach",
            f"retrieval returned {len(result.evidence)} passage(s) "
            "but none contains "
            f"{LATE_EVIDENCE!r}. "
            "The head-excerpt window is gone, so the passage at offset ~12132 must be findable.",
        )
        return []
    rep.passed("step5-reach", f"the late passage {LATE_EVIDENCE!r} is in the retrieved evidence")
    return [(item.evidence_id, item.chunk_id) for item in result.evidence]


def verify_quote_gate(rep, bundle, document_id, answer_result):
    """Deterministic quote verification: a real span aligns, a fabrication degrades."""
    from paperwhisperer.analysis.orchestrator import AnswerClaim, _verify_claim
    from paperwhisperer.ingestion.schema import Evidence

    record = bundle.metadata.get_document(document_id)
    if record is None or record.document is None:
        rep.skipped("step7-verify", "no stored document to verify against")
        return
    document = record.document
    chunks = bundle.metadata.list_chunks(document_id)
    if not chunks:
        rep.skipped("step7-verify", "no chunks to verify against")
        return
    chunk = chunks[0]
    element = next((e for e in document.elements if e.text), None)
    evidence = Evidence(
        evidence_id="ev_manual",
        chunk_id=chunk.chunk_id,
        text=chunk.text,
        locator=element.locator if element else None,
    )
    real_span = (chunk.text or "")[:60].strip()
    if not real_span:
        rep.skipped("step7-verify", "first chunk has no text to quote")
        return

    aligned = _verify_claim(AnswerClaim(text="t", quote=real_span), [evidence], document)
    if aligned.presentation.value == "direct_quote" and aligned.verified_text:
        rep.passed("step7-verify", "a real span is presented as a verified direct_quote")
    else:
        rep.defect("step7-verify",
                      f"a real span did not verify: outcome={aligned.outcome!r} "
                      f"message={aligned.message!r}")

    fabricated = _verify_claim(
        AnswerClaim(text="t", quote="the student made 999999 errors, a number not in the paper"),
        [evidence], document,
    )
    if fabricated.presentation.value == "paraphrase" and not fabricated.verified_text:
        rep.passed("step7-degrade",
                    "a fabricated quote is degraded to a paraphrase with no "
                    "verified text")
    else:
        rep.defect("step7-degrade",
                  f"a fabricated quote was NOT degraded: "
                  f"{fabricated.presentation!r} "
                  f"verified_text={fabricated.verified_text!r}")
    if answer_result:
        for claim in answer_result.get("claims") or []:
            if claim.get("presentation") == "paraphrase" and claim.get("uses_quotation_marks"):
                rep.defect("step7-degrade", "a live paraphrase claim carries quotation marks")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-llm", action="store_true", help="skip only the model-backed steps")
    args = parser.parse_args()

    rep = Reporter()
    _header("PAPERWHISPERER — END-TO-END JOURNEY")
    rep.note(f"corpus: {CORPUS_PDF}")
    try:
        run_journey(rep, use_llm=not args.no_llm)
    except Exception as exc:  # noqa: BLE001 - a crash is the loudest defect there is
        import traceback

        traceback.print_exc()
        rep.defect("journey-crash", f"{type(exc).__name__}: {exc}")
    rep.summary()
    return 1 if rep.defects else 0


if __name__ == "__main__":
    raise SystemExit(main())
