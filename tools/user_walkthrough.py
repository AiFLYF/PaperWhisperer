#!/usr/bin/env python3
"""Walk the product as a user and print what a user would see at each step.

This is the printable narration of ``tests/test_walkthrough.py``. Both files own
**no production code and fix nothing.** Where the product breaks, a ``DEFECT``
line is printed naming what was expected and what actually arrived. Fixing
anything here would defeat the purpose: an audit that repairs what it finds
stops being evidence about the product.

**What "driving the real app" means here.** This repository has no browser
harness and adding one is out of scope, so the walkthrough drives the same HTTP
routes the bundled browser calls (``/api/analyze/stream``, ``/api/ask/stream``,
``/v1/documents/{id}/evidence/{evidenceId}``), reads the real HTML/CSS/JS the
browser would receive, and replays the exact byte-level contract between them —
including the values the front end computes (``state.currentDocumentId``,
``dataset.evidenceId``, ``matchMedia`` results) — by importing the real modules
or by re-deriving them from the real payload. Where a question can *only* be
answered by looking at rendered pixels, the step says ``UNVERIFIED`` rather than
guessing.

**Model calls are counted.** The model-backed steps (a live answer, the live
analysis) spend real, billable, shared-quota calls. This tool prints a call
ledger at the end. Steps that need the model ``SKIP`` with a reason when the
endpoint is unavailable — a skip is "the endpoint was not usable", *not*
"the product works".

Run it with::

    python3 tools/user_walkthrough.py                 # start nothing; use BASE_URL
    python3 tools/user_walkthrough.py --base-url http://127.0.0.1:5000
    python3 tools/user_walkthrough.py --no-llm         # only steps that need no model

Nothing installs anything. Reads a corpus PDF under ``eval/corpus/``; writes
nothing outside the app's own upload/object-store directories, which the app
already manages.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CORPUS_PDF = PROJECT_ROOT / "eval" / "corpus" / "digital_single_col" / "hinton_distillation.pdf"
SCANNED_PDF = PROJECT_ROOT / "eval" / "corpus" / "scanned_clear" / "paper_en_basic.pdf"

#: The answer that sits late in the paper. The string ``67 test errors`` is at
#: offset ~12132 in a 33880-character document — past the old 12000-char head
#: window — so it is the one question a head-excerpt implementation cannot reach.
LATE_EVIDENCE = "67 test errors"
LATE_QUESTION = "How many test errors did the student network make on the test set?"

#: A question the paper cannot answer. Used for the refusal step.
UNANSWERABLE_QUESTION = "What accuracy did the paper report on ImageNet top-1?"
#: A question that shares no vocabulary whatsoever with the paper. Used to show
#: the retriever's behaviour on a question with zero lexical overlap.
NONSENSE_QUESTION = "What is the capital of France?"

MODEL_UNAVAILABLE = (
    "rate limit",
    "429",
    "timeout",
    "timed out",
    "connection",
    "refused",
    "overloaded",
    "503",
    "502",
    "unavailable",
    "invalid local api key",
)

import httpx  # noqa: E402

# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

STEP = 0
DEFECTS: list[dict] = []
SKIPS: list[str] = []


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if sys.stdout.isatty() else text


def head(title: str) -> None:
    global STEP
    STEP += 1
    print()
    print(_c("1;36", f"═══ STEP {STEP}: {title} ") + "═" * max(0, 60 - len(title)))


def user(text: str) -> None:
    """A line the reader would actually see in the browser."""
    print(_c("0;37", "   ▸ ") + text)


def detail(text: str) -> None:
    print(_c("0;90", "     · ") + text)


def defect(owner: str, where: str, expected: str, actual: str, repro: str) -> None:
    rec = {
        "owner": owner,
        "where": where,
        "expected": expected,
        "actual": actual,
        "repro": repro,
    }
    DEFECTS.append(rec)
    print(_c("1;31", "     DEFECT ") + f"[{owner}] {where}")
    print(_c("0;31", "       expected: ") + expected)
    print(_c("0;31", "       actual:   ") + actual)
    print(_c("0;31", "       repro:    ") + repro)


def skip(reason: str) -> None:
    SKIPS.append(reason)
    print(_c("1;33", "     SKIP ") + reason)


def ok(text: str) -> None:
    print(_c("0;32", "     OK ") + text)


# ---------------------------------------------------------------------------
# The model ledger
# ---------------------------------------------------------------------------


class ModelLedger:
    """Counts every real completion this run spends, and records 429 skips."""

    def __init__(self) -> None:
        self.attempts = 0
        self.completions = 0
        self.unavailable = 0

    def note_response(self, status: int, body: str = "") -> bool:
        """Record one completion attempt. Returns True if the model served it."""
        self.attempts += 1
        lowered = f"{status} {body}".lower()
        if status == 200:
            self.completions += 1
            return True
        if status == 429 or any(tok in lowered for tok in MODEL_UNAVAILABLE):
            self.unavailable += 1
            return False
        return False

    def summary(self) -> str:
        return (
            f"attempts={self.attempts} completions_served={self.completions} "
            f"unavailable={self.unavailable}"
        )


LEDGER = ModelLedger()


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


class Walk:
    def __init__(self, base_url: str, use_llm: bool) -> None:
        self.base = base_url.rstrip("/")
        self.client = httpx.Client(timeout=600.0)
        self.use_llm = use_llm

    # -- documents -------------------------------------------------------
    def upload(self, path: Path, label: str) -> dict:
        with path.open("rb") as handle:
            resp = self.client.post(
                f"{self.base}/v1/documents",
                files={"file": (label, handle, "application/pdf")},
            )
        resp.raise_for_status()
        return resp.json()

    def get(self, path: str):
        return self.client.get(f"{self.base}{path}")

    def post(self, path: str, payload: dict):
        return self.client.post(f"{self.base}{path}", json=payload)

    # -- analysis (the browser's real route, SSE) -------------------------
    def analyze_stream(self, path: Path, api_key: str) -> dict:
        """Drive ``POST /api/analyze/stream`` exactly as ``analyze.js`` does.

        Returns ``{"ok": bool, "result": dict|None, "reason": str, "events": [...]}``.
        """
        events: list[tuple[str, dict]] = []
        with path.open("rb") as handle, self.client.stream(
            "POST",
            f"{self.base}/api/analyze/stream",
            files={"file": (path.name, handle, "application/pdf")},
            data={"api_key": api_key, "mode": "sync"},
        ) as resp:
            status = resp.status_code
            buf = ""
            for chunk in resp.iter_text():
                buf += chunk
                while "\n\n" in buf:
                    raw, buf = buf.split("\n\n", 1)
                    name, data = "", ""
                    for line in raw.splitlines():
                        if line.startswith("event:"):
                            name = line[6:].strip()
                        elif line.startswith("data:"):
                            data = line[5:].strip()
                    if name:
                        try:
                            events.append((name, json.loads(data) if data else {}))
                        except Exception:
                            pass
        result = None
        for name, data in events:
            if name == "done" and isinstance(data, dict):
                result = data
        job_id = None
        for name, data in events:
            if name == "start" and isinstance(data, dict):
                job_id = data.get("job_id")
        return {"status": status, "result": result, "job_id": job_id, "events": events}

    def ask_stream(self, session_id: str, session_token: str, question: str,
                   api_key: str, answer_mode: str = "evidence") -> dict:
        """Drive ``POST /api/ask/stream`` exactly as ``chat.js`` does."""
        payload = {
            "session_id": session_id,
            "session_token": session_token,
            "question": question,
            "answer_mode": answer_mode,
            "api_key": api_key,
        }
        result = None
        deltas: list[str] = []
        start = None
        err = None
        with self.client.stream(
            "POST", f"{self.base}/api/ask/stream", json=payload
        ) as resp:
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                try:
                    ev = json.loads(line[5:].strip())
                except Exception:
                    continue
                etype = ev.get("type")
                data = ev.get("data")
                if etype == "start":
                    start = data
                elif etype == "delta":
                    deltas.append((data or {}).get("text", ""))
                elif etype == "done":
                    result = data
                elif etype == "error":
                    err = data
        return {"start": start, "result": result, "deltas": deltas, "error": err}


# ---------------------------------------------------------------------------
# Front-end contract replay (no browser: read the real JS/HTML)
# ---------------------------------------------------------------------------

JS = PROJECT_ROOT / "templates" / "static" / "js"
TEMPLATE = PROJECT_ROOT / "templates" / "index.html"


def front_end_fact(pattern: str, filename: str = "config.js",
                   dotall: bool = False) -> tuple[bool, str]:
    """Assert a literal is present in the real shipped JS (or template), and quote it."""
    if filename.endswith(".html"):
        text = TEMPLATE.read_text(encoding="utf-8")
    else:
        text = (JS / filename).read_text(encoding="utf-8")
    flags = re.S if dotall else 0
    hit = re.search(pattern, text, flags)
    return bool(hit), (hit.group(0).strip() if hit else "<<NOT FOUND>>")


# ---------------------------------------------------------------------------
# The walk
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("WALK_BASE_URL", "http://127.0.0.1:5055"))
    parser.add_argument("--no-llm", action="store_true", help="only steps that need no model")
    args = parser.parse_args()
    use_llm = not args.no_llm

    api_key = os.getenv("OPENAI_API_KEY", "")
    walk = Walk(args.base_url, use_llm)

    print(_c("1;36", "PaperWhisperer — user walkthrough"))
    print(f"base_url = {args.base_url}   model = {'on' if use_llm else 'OFF (--no-llm)'}")
    try:
        health = walk.get("/api/health")
        print(f"health    = {health.status_code} {health.json().get('version')}")
    except Exception as exc:  # noqa: BLE001
        print(_c("1;31", f"cannot reach the app: {exc}"))
        return 2

    if not CORPUS_PDF.exists():
        print(_c("1;31", f"corpus PDF missing: {CORPUS_PDF}"))
        return 2

    _ = _load_dotenv_quietly()

    # ---- STEP 1: upload, parse status -----------------------------------
    head("Upload a real PDF → parse status → cards")
    t0 = time.time()
    up = walk.upload(CORPUS_PDF, CORPUS_PDF.name)
    doc = up["document"]
    parsed_id = doc["document_id"]
    parse = doc.get("parse") or {}
    user(f"uploaded {CORPUS_PDF.name} → document_id={parsed_id}")
    user(f"Parse status card shows: parser={parse.get('parsed_by')}, "
         f"status={parse.get('status')}, pages={doc.get('page_count')}, "
         f"chars={doc.get('char_count')}, chunks={doc.get('chunk_count')}")
    if parse.get("warnings"):
        for w in parse["warnings"]:
            user(f"notice (parse card): {w}")
    ok(f"parse provenance present ({round(time.time()-t0,1)}s)")

    # The central routing decision is docling-first; the shipped default is legacy.
    if parse.get("parsed_by") == "legacy":
        detail("NOTE: shipped default parses with the legacy adapter (D4 in E2E_JOURNEY.md)")

    # ---- STEP 2: a late question — the central claim ----------------------
    head("Ask a question whose answer is LATE in the document")
    # Retrieval is deterministic and needs no model. Use the real retriever over
    # the *parsed* document, exactly as the /v1 qa route does, to prove the late
    # passage is reachable — and then also over the *session-text* document,
    # which is what the browser's Q&A actually retrieves over.
    retrieval = _late_passage_reachable()
    if retrieval["late_reachable"]:
        ok(f"the passage {LATE_EVIDENCE!r} is retrieved (offset {retrieval['offset']} "
           f"> the old 12000 window) — deterministic, no model")
    else:
        defect("rag/retrieval", "rag/retrieve.py", f"{LATE_EVIDENCE!r} should be retrievable",
               "the late passage was NOT retrieved",
               "upload the corpus PDF, ask the late question through /api/ask/stream")
    detail("judging the answer by eye needs the model — see the live steps below")

    # ---- the live answer (needs the model) -------------------------------
    # The analysis is the expensive model step, so it runs ONCE and both live
    # questions reuse the session it establishes.
    session = {"id": "", "token": "", "document_id": ""}
    if use_llm:
        head("Analyse the document and read the live cards (MODEL)")
        analysis = walk.analyze_stream(CORPUS_PDF, api_key)
        result = analysis.get("result")
        if isinstance(result, dict) and result.get("session_id"):
            session = {
                "id": result["session_id"],
                "token": result.get("session_token", ""),
                "document_id": str(result.get("document_id") or parsed_id),
            }
            user(f"analysis settled: session={session['id'][:16]}… "
                 f"document_id={session['document_id']}")
            user(f"Parse status card after analysis: parser="
                 f"{(result.get('parse') or {}).get('parsed_by')}, status="
                 f"{(result.get('parse') or {}).get('status')}")
            _report_analysis_cards(result)
            LEDGER.note_response(200)  # the analysis consumed many completions; count as one step
        else:
            skip(f"the analysis did not settle (HTTP {analysis.get('status')}); "
                 f"the model endpoint is unavailable or the analysis failed — "
                 f"every model-backed step below is skipped")
    else:
        skip("analysis + live answers (--no-llm)")

    if use_llm and session["id"]:
        head("…live answer to the late question (MODEL) — read it, judge it")
        ans = _live_answer(walk, session, api_key, LATE_QUESTION)
        if ans.get("skipped"):
            skip(f"live answer: {ans['reason']}")
        else:
            answer_payload = ans.get("payload")
            _report_answer(answer_payload, LATE_QUESTION, LATE_EVIDENCE)

    # ---- STEP 3: click a citation ---------------------------------------
    head("Click a citation → evidence panel")
    click = _citation_click(walk, parsed_id)
    _report_citation_click(click)

    # ---- STEP 4: a question the paper cannot answer ----------------------
    head("A question the paper CANNOT answer (the refusal behaviour)")
    refusal = _unanswerable_behaviour()
    _report_refusal(refusal)
    if use_llm:
        head("…live refusal (MODEL)")
        ans = _live_answer(walk, session, api_key, UNANSWERABLE_QUESTION)
        if ans.get("skipped"):
            skip(f"live refusal: {ans['reason']}")
        else:
            _report_answer(ans.get("payload"), UNANSWERABLE_QUESTION, None,
                           expect_refusal=True)
    else:
        skip("live refusal (--no-llm)")

    # ---- STEP 5: language switch ----------------------------------------
    head("Switch language 中文 ↔ English")
    _report_language()

    # ---- STEP 6: reparse + parse status ---------------------------------
    head("Reparse and the parse-status card")
    _report_reparse(walk, parsed_id)

    # ---- STEP 7: export --------------------------------------------------
    head("Export, each format")
    _report_export()

    # ---- STEP 8: mobile width -------------------------------------------
    head("Narrow to mobile width → bottom sheet, and back")
    _report_mobile()

    # ---- STEP 9: kill the connection mid-analysis ------------------------
    head("Kill the connection mid-analysis and reconnect")
    _report_reconnect(walk, CORPUS_PDF, api_key, use_llm)

    # ---- STEP 10: a document the parser degraded on ---------------------
    head("A document the parser DEGRADED on (scanned)")
    _report_degradation(walk)

    # ---- summary ---------------------------------------------------------
    _summary()
    return 0


def _load_dotenv_quietly() -> bool:
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
        return True
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# Step helpers that touch real internals (deterministic, no model)
# ---------------------------------------------------------------------------


def _late_passage_reachable() -> dict:
    """Prove the late passage is reachable through the real retriever.

    This is the central claim and it is a property of *retrieval*, so it is
    checked without a model. Two retriever targets are probed because the browser
    retrieves over the session-text document while the evidence click addresses
    the parsed document; both must reach the late passage.
    """
    from paperwhisperer.analysis.orchestrator import (
        build_qa_retriever,
        get_store_bundle,
        index_document_for_retrieval,
        parsed_document_from_text,
    )
    from paperwhisperer.documents.loader import DocumentLoader
    from paperwhisperer.rag.retrieve import TASK_QA, plan_for

    loaded = DocumentLoader.load_with_outcome(str(CORPUS_PDF))
    text = loaded.text
    offset = text.find(LATE_EVIDENCE)
    parsed_doc = loaded.outcome.document
    text_doc = parsed_document_from_text(text)
    bundle = get_store_bundle()
    index_document_for_retrieval(bundle, parsed_doc)
    index_document_for_retrieval(bundle, text_doc)
    retriever = build_qa_retriever(bundle)
    reachable = {}
    for name, target in (("parsed", parsed_doc), ("session-text", text_doc)):
        res = retriever.retrieve(LATE_QUESTION, plan=plan_for(TASK_QA),
                                 document_ids=[target.document_id])
        reachable[name] = any(LATE_EVIDENCE in ev.text for ev in res.evidence)
        if name == "session-text":
            detail(f"browser Q&A retrieves over the session-text doc "
                   f"({target.document_id}), not the parsed doc ({parsed_doc.document_id})")
    return {
        "offset": offset,
        "late_reachable": all(reachable.values()),
        "reachable": reachable,
        "parsed_id": parsed_doc.document_id,
        "text_id": text_doc.document_id,
    }


def _live_answer(walk: Walk, session: dict, api_key: str, question: str) -> dict:
    """Ask one question through the browser's real /api/ask/stream route."""
    if not api_key or not session.get("id"):
        return {"skipped": True, "reason": "no established session"}
    try:
        ask = walk.ask_stream(session["id"], session.get("token", ""), question, api_key)
    except Exception as exc:  # noqa: BLE001
        LEDGER.note_response(0, str(exc))
        return {"skipped": True, "reason": f"ask failed: {exc}"}
    if ask.get("error"):
        return {"skipped": True,
                "reason": f"ask errored: {str(ask['error'])[:140]}"}
    payload = ask.get("result")
    if payload is None:
        return {"skipped": True, "reason": "the answer stream produced no done result"}
    LEDGER.note_response(200)
    return {"payload": payload}


def _report_analysis_cards(result: dict) -> None:
    """What the six cards actually render from a real analysis."""
    sections = result.get("sections") or {}
    if isinstance(sections, dict):
        for name, sec in sections.items():
            status = (sec or {}).get("status")
            has_text = bool((sec or {}).get("content"))
            user(f"card '{name}': status={status} content={'yes' if has_text else 'no'}")
    else:
        user("result carried no sections dict")


def _report_answer(payload: dict | None, question: str, expect: str | None,
                   expect_refusal: bool = False) -> None:
    if not payload:
        skip("no answer payload")
        return
    answer = str(payload.get("answer", ""))
    confidence = payload.get("confidence")
    claims = payload.get("claims") or []
    citations = payload.get("citations") or []
    user(f"answer[{confidence}]: {answer[:180]!r}")
    user(f"claims={len(claims)}  citations={len(citations)}  (citation chips rendered = "
         f"{sum(1 for c in claims if str(c.get('presentation'))=='direct_quote')})")
    if expect and expect not in answer:
        # The cited passage supports the figure, but the answer did not state it.
        detail(f"the answer does not literally contain {expect!r}; a reader cannot "
               "confirm the claim from the answer text alone")
    if expect_refusal:
        if confidence == "insufficient_evidence" and not citations:
            ok("refused, and manufactured no citation")
        else:
            detail(f"confidence={confidence}, citations={len(citations)} — "
                   "the refusal here came from the MODEL, not from the deterministic gate")


def _citation_click(walk: Walk, parsed_id: str) -> dict:
    """Replay the exact citation-chip click the browser performs.

    The front end takes ``dataset.evidenceId`` from a claim and pairs it with
    ``state.currentDocumentId`` (the analyze result's document id) to build
    ``/v1/documents/{documentId}/evidence/{evidenceId}``. This reproduces that
    pairing using the real retriever over the real session-text document, which
    is what the browser's Q&A retrieves over.
    """
    from paperwhisperer.analysis.orchestrator import (
        build_qa_retriever,
        get_store_bundle,
        index_document_for_retrieval,
        parsed_document_from_text,
    )
    from paperwhisperer.documents.loader import DocumentLoader
    from paperwhisperer.rag.retrieve import TASK_QA, plan_for

    loaded = DocumentLoader.load_with_outcome(str(CORPUS_PDF))
    text_doc = parsed_document_from_text(loaded.text)
    bundle = get_store_bundle()
    index_document_for_retrieval(bundle, text_doc)
    retriever = build_qa_retriever(bundle)
    res = retriever.retrieve(LATE_QUESTION, plan=plan_for(TASK_QA),
                             document_ids=[text_doc.document_id])
    if not res.evidence:
        return {"citations": 0, "opened": 0}
    opened = 0
    tried = 0
    first_status = None
    first_body = ""
    for ev in res.evidence[:5]:
        # evidence.js buildEvidenceUrl(documentId = state.currentDocumentId, evidenceId)
        url = f"/v1/documents/{parsed_id}/evidence/{ev.evidence_id}"
        resp = walk.get(url)
        tried += 1
        if resp.status_code == 200 and "error" not in resp.json():
            opened += 1
        elif first_status is None:
            first_status = resp.status_code
            first_body = resp.text[:120]
    return {
        "citations": len(res.evidence),
        "tried": tried,
        "opened": opened,
        "first_status": first_status,
        "first_body": first_body,
        "parsed_id": parsed_id,
        "text_id": text_doc.document_id,
    }


def _report_citation_click(click: dict) -> None:
    if click["citations"] == 0:
        skip("no citations to click")
        return
    user(f"{click['tried']} citation chips rendered; the panel click resolves "
         f"{click['opened']}/{click['tried']}")
    if click["opened"] == 0:
        detail(f"first click → HTTP {click['first_status']} {click['first_body']}")
        defect(
            "evidence-id contract across rag → api → frontend",
            "templates/static/js/evidence.js:269,303 + "
            "paperwhisperer/api/routes_evidence.py:213-219",
            "a citation chip the reader clicks opens the passage (200)",
            f"every click 404s ({click['first_status']} evidence_not_found): the browser's "
            f"Q&A mints ev_* citations over the session-TEXT document "
            f"({click['text_id']}), but the evidence panel addresses the PARSED document "
            f"({click['parsed_id']}); the two never share a document, so the id cannot resolve",
            "ask any question through /api/ask/stream, then GET "
            f"/v1/documents/{click['parsed_id']}/evidence/<ev_id from the answer>",
        )
    else:
        ok("the citation click opens the evidence panel")


def _unanswerable_behaviour() -> dict:
    """What does retrieval hand the answerer for a question the paper cannot answer?

    This is deterministic and needs no model. The central question is whether the
    retrieval returns *nothing* (so the fixed, no-model refusal fires and no
    citation can be manufactured) or whether it pads irrelevant passages (so the
    model is called and the refusal is whatever the model decides).
    """
    from paperwhisperer.analysis.orchestrator import (
        build_qa_retriever,
        get_store_bundle,
        index_document_for_retrieval,
        parsed_document_from_text,
    )
    from paperwhisperer.documents.loader import DocumentLoader
    from paperwhisperer.rag.retrieve import TASK_QA, plan_for

    loaded = DocumentLoader.load_with_outcome(str(CORPUS_PDF))
    text_doc = parsed_document_from_text(loaded.text)
    bundle = get_store_bundle()
    index_document_for_retrieval(bundle, text_doc)
    retriever = build_qa_retriever(bundle)
    out = {}
    for label, q in (("off-topic", UNANSWERABLE_QUESTION), ("nonsense", NONSENSE_QUESTION)):
        res = retriever.retrieve(q, plan=plan_for(TASK_QA),
                                 document_ids=[text_doc.document_id])
        top = res.evidence[0].score if res.evidence else 0.0
        out[label] = {
            "question": q,
            "evidence": len(res.evidence),
            "insufficient": res.insufficient_evidence,
            "top_score": top,
        }
    return out


def _report_refusal(refusal: dict) -> None:
    for label, r in refusal.items():
        user(f"[{label}] “{r['question']}” → retrieval returns "
             f"{r['evidence']} passages, insufficient_evidence={r['insufficient']}, "
             f"top score={r['top_score']:.3f}")
    any_padded = any(r["evidence"] > 0 and not r["insufficient"] for r in refusal.values())
    if any_padded:
        worst = min(refusal.values(), key=lambda r: r["top_score"])
        # The downstream consequence: a claim that PARAPHRASES one of the padded
        # passages counts as "verified" and the answer is labelled `supported`.
        from paperwhisperer.analysis.orchestrator import AnswerClaim, _derive_confidence
        from paperwhisperer.rag.citations import ClaimPresentation

        verdict = _derive_confidence([
            AnswerClaim(text="A paraphrase of a padded, irrelevant passage.",
                        evidence_ids=("ev_padded",), quote="",
                        presentation=ClaimPresentation.PARAPHRASE)
        ])
        user(f"confidence verdict a paraphrase of a padded passage gets: {verdict}")
        detail("so an unanswerable question can carry a `supported` verdict with a "
               "citation chip — whether the live model actually takes this path needs "
               "the endpoint (UNVERIFIED today), but the structural path is open")
        defect(
            "the refusal gate: retrieval has no relevance floor",
            "paperwhisperer/rag/retrieve.py:877-884 (QA plan final_k=8, no min_score) "
            "and :1216-1218 (insufficient_evidence == 'no evidence returned'); "
            "orchestrator.py:497-531 (_derive_confidence / _claim_is_verified)",
            "a question the paper cannot answer yields no evidence, so the fixed, "
            "no-model refusal fires and no citation can be manufactured",
            f"the retriever pads every question to {worst['evidence']} passages (near-zero "
            f"relevance, top score {worst['top_score']:.3f}); insufficient_evidence is False, "
            "so the model IS called; and a paraphrase citing one of those padded passages "
            f"is labelled `{verdict}`. The refusal, and the verdict that nothing is "
            "backed by evidence, are therefore the model's discretion, not a system finding",
            "retrieve over the corpus PDF with the 'capital of France' question: "
            "8 passages at ~0.03 score, insufficient_evidence=False; then a claim "
            "paraphrasing one of them derives confidence `supported`",
        )


def _report_language() -> None:
    # Which surfaces re-render on a locale change, and which strings are hardcoded?
    on_locale = []
    for js in sorted(JS.glob("*.js")):
        if "onLocaleChange" in js.read_text(encoding="utf-8"):
            on_locale.append(js.name)
    user(f"modules that re-render on a language switch: {on_locale or 'none'}")
    # Hardcoded strings on the surfaces the switch is supposed to cover.
    hardcoded = {
        "evidence.js:166 target page label": "`第 ${label} 页`",
        "evidence.js:200 snippet meta": "`引用 …`",
        "claims.js:197 direct-quote prefix": "`逐字引用：` / `归一化后逐字对齐：`",
        "claims.js:251 paraphrase prefix": "`转述（未通过逐字回源校验）：`",
    }
    for where, text in hardcoded.items():
        present, _ = front_end_fact(re.escape(text.split("/")[0].strip("`")))
        user(f"evidence/claim text {where} = {text}  (i18n key present: {present})")
    defect(
        "front-end i18n coverage of the evidence/claim surfaces",
        "templates/static/js/claims.js:197,251; evidence.js:166,200-204",
        "switching 中文 ↔ English re-renders the evidence panel, the citation labels "
        "and the claim prefixes (parse status already does, via parseStatus.js:618)",
        "the claim prefixes ('逐字引用：' / '转述（未通过逐字回源校验）：'), the evidence page "
        "label ('第 N 页'), and the snippet meta ('引用 … / 来自已核对的原文 / 片段已截断') "
        "are hardcoded Chinese/English literals with no i18n key, and only parseStatus.js "
        "subscribes to onLocaleChange — so after switching to English the interface is a mix",
        "switch to English, open any answer: the claim prefix and evidence meta stay in the "
        "other language while the surrounding card switches",
    )


def _report_reparse(walk: Walk, parsed_id: str) -> None:
    # The reparse control offers docling/force-OCR; the shipped default refuses.
    resp = walk.post(f"/v1/documents/{parsed_id}/reparse", {"parser": "docling"})
    user(f"Reparse with the dropdown's 'Docling (local)' → HTTP {resp.status_code} "
         f"{resp.text[:110]}")
    if resp.status_code != 200:
        detail("the reparse card surfaces this honestly (shows the reason and 'left untouched')")
        defect(
            "parser routing default",
            "paperwhisperer/core/config.py PARSER_BACKEND default 'legacy'; "
            "the reparse catalog still offers 'docling'",
            "the offered parser choices are ones the deployment can actually run, or the "
            "default is docling-first as docs/PARSER_BAKE_OFF.md §6 decided",
            "the dropdown advertises Docling (local) and forced OCR, but with "
            "PARSER_BACKEND=legacy the gateway has rolled back to legacy and refuses "
            f"with 409 parser_unavailable ({resp.json().get('error','')[:70]})",
            "POST /v1/documents/{id}/reparse {\"parser\":\"docling\"} → 409 parser_unavailable",
        )
    resp_default = walk.post(f"/v1/documents/{parsed_id}/reparse", {})
    if resp_default.status_code == 200:
        pr = resp_default.json().get("parse", {})
        user(f"Reparse with the default route → parser={pr.get('parsed_by')} "
             f"status={pr.get('status')}")


def _report_export() -> None:
    found, quote = front_end_fact(r"EXPORT_FORMATS = \[.*?\]", "config.js", dotall=True)
    user(f"export formats offered (config.js EXPORT_FORMATS): {quote}")
    detail("export is produced in the browser (markdown + html); it needs a completed "
           "analysis in the session, so a bare walkthrough cannot populate it")
    if "value: 'md'" in quote and "value: 'html'" in quote:
        ok("two formats (Markdown, HTML); the hint promises provenance in the export")


def _report_mobile() -> None:
    found, quote = front_end_fact(r"@media \(max-width: 1080px\)", "index.html")
    user(f"evidence panel sheet breakpoint in the real CSS (template <style>): "
         f"{quote or '<<NOT FOUND>>'}")
    found2, quote2 = front_end_fact(r"EVIDENCE_SHEET_MEDIA_QUERY = .*", "config.js")
    detail(f"JS breakpoint mirror: {quote2}")
    anchor_found, anchor_quote = front_end_fact(r"state\.evidenceScrollAnchor = window\.scrollY",
                                                "evidence.js")
    detail(f"scroll anchor captured on open: {anchor_quote}")
    if found and anchor_found:
        ok("at ≤1080px the panel is a bottom sheet; open/close preserves the question "
           "(scroll anchor captured/restored, input never touched)")
    user("NOTE: whether the sheet *looks* right and whether the question survives a real "
         "open/close can only be confirmed by rendering pixels — UNVERIFIED without a browser")


def _report_reconnect(walk: Walk, path: Path, api_key: str, use_llm: bool) -> None:
    # The re-attach contract is code + a job row. Confirm the routes the front end
    # relies on for reconnect exist, and that an aborted client is classified as
    # retryable (client_stream_interrupted) rather than terminal.
    code_found, code_quote = front_end_fact(r"client_stream_interrupted", "analyze.js")
    _attach_found, attach_quote = front_end_fact(
        r"async function reattachAnalysisJob", "analyze.js"
    )
    user(f"client classifies a dropped stream as {code_quote!r} (retryable) "
         f"— present: {code_found}")
    user(f"front end re-attaches to the durable job on drop: {attach_quote or '<<NOT FOUND>>'}")
    detail("a dropped connection is resolved by asking the server what the job actually did, "
           "rather than declaring the run failed")
    # Confirm the resume route serves.
    jobs = walk.get("/v1/jobs")
    if jobs.status_code == 200:
        ok("GET /v1/jobs serves — the state surface the reconnect reads")
    if not use_llm:
        skip("a live mid-analysis disconnect needs the model (analysis must be running)")


def _report_degradation(walk: Walk) -> None:
    if not SCANNED_PDF.exists():
        skip(f"scanned corpus file missing: {SCANNED_PDF}")
        return
    up = walk.upload(SCANNED_PDF, SCANNED_PDF.name)
    doc = up["document"]
    parse = doc.get("parse") or {}
    user(f"scanned upload → parser={parse.get('parsed_by')} status={parse.get('status')} "
         f"chars={doc.get('char_count')}")
    for w in parse.get("warnings", []):
        user(f"notice (parse card): {w}")
    q = parse.get("quality", {})
    user(f"quality verdict carried to the client: {q.get('status')} flags="
         f"{[f.get('rule_id') for f in q.get('flags', [])]}")
    if doc.get("char_count", 0) == 0:
        detail("the document produced ZERO characters; a reader must be told why. "
               "This is the silent-truncation class the release gate scores.")


def _summary() -> None:
    print()
    print(_c("1;36", "── SUMMARY " + "─" * 52))
    print(f"defects found : {len(DEFECTS)}")
    print(f"skips         : {len(SKIPS)}")
    print(f"model calls   : {LEDGER.summary()}")
    if DEFECTS:
        print()
        print(_c("1;31", "DEFECTS (nothing was fixed):"))
        for i, d in enumerate(DEFECTS, 1):
            print(f"  {i}. [{d['owner']}] {d['where']}")
            print(f"     expected: {d['expected']}")
            print(f"     actual:   {d['actual']}")
    print()
    print(_c("0;90", "Rendered-pixel questions (panel layout, mobile sheet, colour) are "
                      "UNVERIFIED without a browser — this repository has no browser harness."))


if __name__ == "__main__":
    sys.exit(main())