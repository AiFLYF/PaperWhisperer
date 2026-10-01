# E2E Journey — what the assembled product actually does

This is the record of one run of the full research journey on a real paper,
through the real surfaces: **upload → parse → index → analyse → ask a late
question → get claims/citations → click a citation → export.** It was produced
by running two things this change adds:

- `python3 -m pytest tests/test_e2e_journey.py -q -p no:cacheprovider` — the gate
- `python3 tools/e2e_smoke.py` — the printable narration of the same journey

Both files own **no production code and fix nothing.** Where the journey breaks,
the failing assertion is recorded (and, for defects confirmed today, marked
`xfail(strict=True)` so a fix turns the xpass into a failure that forces the
marker to be retired). This document is the write-up of what actually happened.

---

## Setup used for the recorded run

| | |
|---|---|
| Python | `/home/ubuntu/miniconda3/bin/python3` (conda 3.13), no installs |
| Corpus | `eval/corpus/digital_single_col/hinton_distillation.pdf` (9 pages → 33,880 chars) |
| Parse | `POST /v1/documents` → 201; `parsed_by=legacy`, `status=partial` |
| Index | 16 chunks in a temporary SQLite store (personal tier); PostgreSQL never touched |
| LLM endpoint | `http://127.0.0.1:8084/v1`, model `space-bunny-free` — **HTTP 429 (rate limited) for the whole run** |
| Embedding | `EMBEDDING_LOCAL_ENABLED=1`, MiniLM-Multilingual-L12-v2, 384-dim (configured; see D2) |

The endpoint served `GET /models` 200 the entire time but answered **every**
chat completion with `429 Rate limit exceeded` (shared quota). Per the house
rule (an unavailable endpoint must *skip*, not fail), every model-backed step
is reported as `SKIP`. The deterministic steps — which is where this project's
central claim lives — all ran.

---

## The verdict, in one line

**The central retrieval fix holds in the assembled system** — the passage at
offset **12132** (past the old 12000-char head window) is retrieved
deterministically by the real `HybridRetriever` over the real indexed store,
with no model in the loop. **But the product is not yet "working end to end":**
three composition defects break the journey around that win (a citation click
404s, the dense leg never runs, the export carries no provenance), and the
shipped default parses with the legacy adapter, so no citation can be visually
targeted.

---

## Step-by-step: what each step returned

| # | Step | Result |
|---|------|--------|
| 1 | Upload real PDF via `POST /v1/documents` | **PASS** — `201`, `document_id=doc_…`, 9 pages, 16 chunks |
| 2 | Parse provenance | **PASS (page)** — `parsed_by=legacy`, `status=partial`; 9 page descriptors, `locator_kind=pdf_page`. **bbox: FAILS** — legacy adapter emits no bbox (see D4) |
| 3 | Vectors landed | **FAIL** — `chunks=16 vectors_present=0 dims=[]` (see D2) |
| 4 | Analysis (`POST /api/analyze`) | **SKIP** — endpoint 429 (would be exercised; see Cost) |
| 5 | Late question answer | **PASS (retrieval)** — the real retriever returned 8 evidence items, one containing `67 test errors`. **Live answer: SKIP** (429) |
| 6 | Answer structure (claims/ids/citations/confidence) | **SKIP** — no live answer (429) |
| 7 | Quote verify / fabricate→degrade | **PASS** — a real 60-char span → verified `direct_quote`; a fabricated number → `paraphrase`, no verified text, not quotable |
| 8 | Citation click → Evidence API → page image | **FAIL (click)** — `ev_…` id → `404 evidence_not_found` (D1). **PASS (renderer)** — addressed by the chunk id, page 4 rendered a real 477,986-byte PNG |
| 9 | Export provenance | **SKIP** live (429); a stubbed-model probe of the same route shows the export is an empty shell with no provenance (D3) |

---

## The central claim (step 5) — reported honestly either way

The project was rebuilt around deleting a hard `document_content[:12000]` head
excerpt from the Q&A path. Against this paper the sentence reporting its result
— `67 test errors` — sits at **offset 12132**, 132 characters past that window,
inside the results section. A head-excerpt implementation can never answer a
question about it.

**It passes.** `tests/test_e2e_journey.py::test_step5_the_late_passage_is_reachable_by_retrieval`
asserts, against the *indexed* document through the real `HybridRetriever`
(no model, no stub):

- the evidence is at offset 12132 > the 12000 window (the fixture really is the
  late case), and
- the retriever returns 8 passages, one of which contains `67 test errors`.

So the fix the project was rebuilt around does hold in the assembled system, and
it holds *deterministically* — it does not need a model, so a rate-limited
endpoint can never hide it. This is the headline.

---

## Defects found (ranked by how much they undermine the product's claim)

Each is recorded with owner, location, and the smallest reproduction. **Nothing
here was fixed by this change** — these belong to the owning modules.

### D1 — A real citation click 404s (evidence-id namespace) — *worst*

The promise "click a citation and it opens the passage" is the product's whole
pitch, and it fails. The retriever and the front end both emit **`ev_*`**
evidence ids; the Evidence API resolves its `{evidence_id}` path parameter as a
**`chunk_*` id**. So the id a reader's click carries never resolves.

- **Owner:** the evidence-id contract across `rag` → `api` → frontend.
- **Where:**
  - `paperwhisperer/api/routes_evidence.py:176` — `chunk = stores.metadata.get_chunk(evidence_id)` (path param treated as a chunk id).
  - `paperwhisperer/ingestion/schema.py:1192` — `Evidence.evidence_id` defaults to an `ev_…` id.
  - `templates/static/js/evidence.js:269` — `button.dataset.evidenceId = String(claim?.evidence_id …)`; sent to `EVIDENCE_API_PATH = '/v1/documents/{documentId}/evidence/{evidenceId}'` (`templates/static/js/config.js:66`).
- **Smallest reproduction:** upload the corpus PDF via `POST /v1/documents`, run retrieval for the late question, then `GET /v1/documents/{doc_id}/evidence/{evidence.evidence_id}` → **404 `evidence_not_found`**. The same route with `{evidence.chunk_id}` → **200**.
- **Why the tests are green:** every test in `tests/test_evidence_api.py` addresses **`chunk_*`** ids only (`_get(client, "doc_1", "chunk_pdf")`, …). No test ever passes an `ev_*` id, so the 79-test evidence suite is fully green while the real click 404s. This is a defect that exists **only in composition**.

### D2 — The dense retrieval leg never runs (0 vectors land)

`EMBEDDING_LOCAL_ENABLED=1` is configured and MiniLM is installed, but an
uploaded document is indexed **lexically-only**: `chunks=16`,
`vectors_present=0`. Nothing in the request path ever constructs an embedder.

- **Owner:** analysis/ingestion wiring (whoever owns `persist_document` and the Q&A analyzer construction).
- **Where:**
  - `paperwhisperer/api/routes_v1.py:1483` — `index_document_for_retrieval(stores, document)` (no `embedder`, so vectors are never computed).
  - `paperwhisperer/api/routes_qa.py:117` — `analyzer.store_bundle = get_store_bundle()`; neither this nor the `/v1` Q&A route (`routes_v1.py:2055`) ever sets `analyzer.embedder`.
  - `paperwhisperer/llm/embeddings_local.py:579` — `build_embedder_from_config()` is the one factory that honours `EMBEDDING_LOCAL_ENABLED`, and it is **never called from production** (grep: only its own definition).
- **Smallest reproduction:** upload the corpus PDF, then for each stored chunk call `bundle.vectors.get_vector(chunk.chunk_id)` → all `None`.
- **Why it matters:** the plan's retrieval is "dense + lexical hybrid". Retrieval here is answered from **lexical + fuzzy legs alone**. The measured recall@8 = 1.00 was obtained *in isolation* (with an embedder injected); in the assembled product the dense leg abstains on every question. This is the second canonical "works in isolation, fails in composition."

### D3 — The export carries no provenance (fact layer / `analysis_run` never produced; session drops them)

The analysis route runs the **legacy six-section path**, so its result carries
neither `paper_facts` nor `analysis_run`; and even if it did, the session does
not persist them, while `/api/export` reads the stored session. The exported
report is an empty shell.

- **Owner:** analysis route wiring (`/api/analyze`) **and** `sessions.store` persistence.
- **Where:**
  - `paperwhisperer/api/routes_jobs.py:586` — `_analyse_stage` calls `finalize_analysis_result(..., run=None, report=None)`, i.e. the legacy path. The fact-layer entry point `analysis/service.py:2120 analyze_with_fact_layer` — the only producer of `paper_facts`/`analysis_run` — is **never called by any route** (grep: no callers).
  - `finalize_analysis_result`'s own docstring (`analysis/service.py:2022`) states `sessions.store.build_session_payload` does **not** persist `paper_facts`/`analysis_run`, so `/api/export` would lose them regardless.
- **Smallest reproduction** (model-independent — a stubbed `LLMClient.complete` returning `"CANNED"` is enough, since the model content is irrelevant to which keys exist): `POST /api/analyze` → `200`, result has `paper_facts=False, analysis_run=False`; then `GET /api/export/{session_id}?session_token=…&format=markdown` → a ~203-byte report containing `AI 摘要 / 引用片段 / 思维导图 / 元信息` and **no** provenance section, no citation index, no `analysis_run`.
- **Why it matters:** T1.8 (PaperFacts + cross-card consistency + cost) and T3.5 (traceable export) are unit-tested in isolation but are not wired into the one analysis route the product actually exposes. The exported artefact does not record what produced it.

### D4 — The shipped default parses with the legacy adapter (no bbox; clean PDF → "partial")

The bake-off decided **docling-first** and docling 2.131.0 is installed and
registered (`build_default_parsers()` → `['docling','legacy']`), but
`PARSER_BACKEND` defaults to `"legacy"`, so the shipped default bypasses the
gateway and runs the legacy adapter — which produces **no bbox** and marks even
this perfectly-clean digital PDF as `status=partial` with the warning
`legacy PDF 解析只能定位到页，无法提供页内坐标（bbox）`.

- **Owner:** parser routing default (`core/config.py`).
- **Where:** `paperwhisperer/core/config.py:575` — `PARSER_BACKEND = normalize_parser_backend(os.getenv("PARSER_BACKEND", "legacy"))`. Decision record: `docs/PARSER_BAKE_OFF.md` §6 (strategy B, docling-first).
- **Smallest reproduction:** `POST /v1/documents` with the corpus PDF → `parse.parsed_by == "legacy"`; `GET /v1/documents/{id}/pages` → 9 page descriptors, none with a bbox.
- **Why it matters:** the Evidence-first promise is "every conclusion resolves
  to a page **and** a bbox". With the default parser, citations resolve to a page
  only, and no evidence can be visually targeted. The docling adapter that would
  fix this is present but not in the default path.

---

## What works only in isolation and fails in composition

1. **Dense retrieval (D2).** recall@8 = 1.00 with an embedder injected; 0
   vectors and a lexical-only leg in the assembled product.
2. **Evidence API (D1).** 79 tests green — all using `chunk_*` ids; the real
   `ev_*` click 404s.
3. **Fact layer / `analysis_run` / export provenance (D3).** Unit-tested, but the
   producing entry point has no route caller and the session drops the keys.
4. **Docling parsing (D4).** Works and is the recorded routing decision; the
   default path doesn't use it.

---

## Cost and duration

- **Deterministic journey (no model):** `tools/e2e_smoke.py --no-llm` → **~1.8 s**;
  the pytest file → **~29 s** wall (dominated by one module-scoped upload+parse
  reused by every assertion, plus collection). One parse (<1 s via the legacy
  loader), 16 chunks, one page render (~0.4 s). **The parse is performed once and
  reused across every assertion**, as required.
- **Full journey with the model:** a `/api/analyze` on this paper issues ~12–16
  completions (9 summary chunks + merge + quotes + mindmap; mermaid/evaluation/
  brief are disabled to keep it re-runnable) plus 1 for the late Q&A. Against the
  shared endpoint's throughput that is minutes — and on the recorded run it never
  completed, because the endpoint returned HTTP 429 on every completion for the
  entire window. The journey therefore **skipped** every model-backed step rather
  than failing, and re-run the moment the endpoint is available.

---

## Reproducing this record

```bash
# gate (deterministic + model-backed; model steps skip when the endpoint is 429)
python3 -m pytest tests/test_e2e_journey.py -q -p no:cacheprovider
# narration
python3 tools/e2e_smoke.py            # full journey (model steps skip on 429)
python3 tools/e2e_smoke.py --no-llm   # only the steps that need no model
```

Neither touches PostgreSQL, installs anything, or writes outside a temporary
directory (removed on exit). `tests/test_e2e_journey.py` reads the corpus PDF and
guards it with `skipif`; every model-backed test skips — never fails — when the
endpoint cannot serve a completion.

**A note on the skips.** Steps 4, 6 and 9 (live analysis, live answer structure,
live export) were not exercised in this run because the configured endpoint was
rate-limited throughout. Their product-defect findings (D3 in particular) were
confirmed independently and model-independently with a stubbed `LLMClient`, which
proves the missing keys are a wiring/ persistence fact rather than a model-behaviour
fact. When the endpoint is available, `test_step4`, `test_step5_live` and
`test_step9` will run for real.
