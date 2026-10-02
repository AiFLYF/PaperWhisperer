# User walkthrough — what the product actually does when you use it

This is the record of one run of the product **as a user**, not as a test suite.
Where the previous records (`docs/E2E_JOURNEY.md`, `docs/RELEASE_GATES.md`,
`docs/LLM_TESTS.md`) assert that units behave correctly and report three
composition defects, this document asks the only question those cannot answer:
**when someone uploads a paper and reads the screen, does the screen tell them
the truth?**

It was produced by running two new files this change adds:

- `python3 tools/user_walkthrough.py` — the printable narration of the walk,
  driving the real HTTP routes the bundled browser calls
- `python3 -m pytest tests/test_walkthrough.py` — the hermetic pins for whatever
  the walk found (no live model; it never skips for an endpoint outage)

Both files **own no production code and fix nothing.** Every defect below is
reported with a file:line, an owner, and the smallest reproduction. Fixing
anything while auditing is how an audit stops being evidence about the product.

**The bar applied to every step:** not "was the status code right" but *what
would a reader see, and does it mean anything true*. Where a question can only be
answered by looking at rendered pixels, the step says **UNVERIFIED** rather than
guessing.

---

## How to read this

| | |
|---|---|
| App | `python3 -m uvicorn paperwhisperer.app:app` (recorded run: `127.0.0.1:5055`) |
| Corpus | `eval/corpus/digital_single_col/hinton_distillation.pdf` (9 pages, 33,864 chars) |
| Scanned fixture | `eval/corpus/scanned_clear/paper_en_basic.pdf` |
| LLM endpoint | `http://127.0.0.1:8084/v1`, model `space-bunny-free` — **HTTP 429 (rate limited) for the entire recorded run** |
| Embedding | `EMBEDDING_LOCAL_ENABLED=1`, MiniLM-Multilingual-L12-v2, 384-dim (configured) |

The endpoint answered **every** chat completion with `429 Rate limit exceeded`
(shared quota) for the whole run. Per the house rule an unavailable endpoint
**skips**, it does not fail — but read each skip as *"the endpoint was not
usable"*, **not** *"the product works."* The model-backed steps below are
`SKIP`; every step that does not need a model ran for real, and that is where
this product's central claims live.

**Live model calls spent on this walkthrough: 0 completions served.**
The recorded run attempted the analysis and both live questions; each underlying
completion was refused with 429. A background probe made **60 attempts over
~20 minutes and never once succeeded**, so the endpoint was unusable for the
whole session, not merely unlucky at one moment. Every finding below is
model-independent and reproduces without an endpoint.

---

## The single most misleading thing in the interface

**The confidence verdict beside an answer — `supported` / `partially_supported`
/ `insufficient_evidence` — reads like a system finding about whether the paper
backs the answer. It is not one, and for the question class this product was
rebuilt around it can say `supported` about a paper that does not support it.**

Here is the chain, every link deterministic and reproduced without a model:

1. Retrieval has **no relevance floor** (`rag/retrieve.py:877-884`). The fuzzy
   and neighbour legs pad *every* question to 8 passages. Asked "What is the
   capital of France?", retrieval returns 8 passages at a top score of **0.031** —
   and `insufficient_evidence` is `False`, because it is defined as
   "no evidence returned" (`rag/retrieve.py:1216-1218`).
2. So the deterministic, no-model refusal (`_insufficient_answer`) never fires;
   the **model** is called and handed those 8 irrelevant passages.
3. `_derive_confidence` (`orchestrator.py:497-531`) then labels a claim *verified*
   when it cites **any** retrieved `evidence_id` and either quotes nothing (a
   paraphrase) or quotes something that aligned. A **paraphrase of one of the
   padded, irrelevant passages is labelled `supported`** — reproduced here:
   `_derive_confidence([AnswerClaim(evidence_ids=("ev_padded",), quote="")]) ==
   "supported"`.

So an unanswerable question can reach the screen carrying a `supported` badge and
a citation chip pointing at a page that has nothing to do with the question. The
pixels say *"this is backed by the paper"*; the system never established that, and
could not have — it never looked at relevance.

The honest boundary: **I did not observe the live model taking this path** (the
endpoint refused every call, so I could not read a real answer). What I did
establish, deterministically, is that the path is open and that the verdict
cannot distinguish a supported claim from one resting on a padded passage. The
project's own notes already record that a model handed no real evidence "invents
confidently in the hedging around it." Whether a given draw crosses from "refused
(harmless, but for the wrong reason)" to "answered and labelled supported
(harmful)" is the live behaviour I could not reach.

This is worse than the click-404 (WALK-DEFECT-1), which at least fails loudly.
Here the interface looks *right* — it has a verdict, a colour and a citation — and
is wrong.

---

## The verdict on the journey as a whole

**The retrieval fix still holds and is deterministic — a question whose answer
sits past the old head-window reaches its passage with no model in the loop. But
the product's central UI promise, "click a citation and it opens the passage," is
broken in the composed browser path: every citation the reader can click 404s.**
That is D1 from `E2E_JOURNEY.md`, not fixed — re-introduced in a different form
that the existing 79-test evidence suite cannot see because those tests all
address `chunk_*` ids directly and the *composition* is what fails. Two more
composition defects and one routing-default defect are confirmed live.

---

## Step-by-step: what each step returned

| # | Step | Result |
|---|------|--------|
| 1 | Upload a real PDF → parse status → cards | **PASS** — `201`; status card shows `parser=legacy, status=partial, 9 pages, 33864 chars, 32 chunks`; parse notices rendered. Cards need the model (SKIP). |
| 2 | Late question — the central claim | **PASS (retrieval, deterministic)** — `67 test errors` (offset 12132 > the old 12000 window) is returned by the real `HybridRetriever`, no model. **Live answer: SKIP (429).** |
| 3 | Click a citation → evidence panel | **FAIL (click)** — **0 of 5 chips resolve; each 404s `evidence_not_found`** (WALK-DEFECT-1). |
| 4 | A question the paper cannot answer | **FAIL (gate)** — retrieval pads every off-topic question to 8 near-zero-relevance passages and `insufficient_evidence=False`, so the model, not the deterministic refusal, decides (WALK-DEFECT-2). **Live refusal: SKIP (429).** |
| 5 | Language 中文 ↔ English | **FAIL (evidence/claim surfaces)** — parse status follows the switch; the claim prefixes, evidence page label and snippet meta do not (WALK-DEFECT-3). |
| 6 | Reparse + parse-status card | **PARTIAL** — the advertised "Docling (local)" 409s under the shipped default and the card surfaces it honestly (WALK-DEFECT-4 / D4). Default reparse runs legacy. |
| 7 | Export, each format | **PASS (catalogue)** — Markdown + HTML offered; the hint promises provenance. Cannot populate without a completed analysis (SKIP). |
| 8 | Mobile width → bottom sheet | **PASS (CSS contract)** — `@media (max-width: 1080px)` in the real template; scroll anchor captured/restored; input never touched. Pixel appearance **UNVERIFIED**. |
| 9 | Kill connection mid-analysis → reconnect | **PASS (contract)** — dropped stream classified `client_stream_interrupted` (retryable) and re-attached to the durable job; `GET /v1/jobs` serves. Live disconnect **SKIP** (needs the model). |
| 10 | A document the parser degraded on | **PASS (visible)** — a 0-character scanned parse now reaches the reader as four explicit notices + a `fail` quality verdict. See §10. |

---

## The central claim (step 2) — reported honestly either way

The project was rebuilt around deleting a hard `document_content[:12000]` head
excerpt. Against this paper the sentence reporting its result — `67 test errors`
— sits at offset **12132**, 132 characters past that window. A head-excerpt
implementation can never answer a question about it.

**It passes, deterministically.** With no model and no stub, the real
`HybridRetriever` over the real indexed store returns it, for both the parsed
document and the session-text document. This is pinned in
`tests/test_walkthrough.py::test_the_late_passage_is_reachable_by_retrieval`.

What a **live** answer would add is whether the model, handed that passage,
states the figure and whether the quote is verifier-aligned. That is exactly the
check the endpoint's 429 denied. It is recorded as `SKIP`, and it is the one
judgement-by-eye in this walkthrough that this run could **not** make.

---

## Defects found (ranked by how much they undermine the central claim)

Each is a **new** finding from walking the product as a user, with an owner and
> **Status update — this supersedes the present tense below.** All four findings
> this document reports have since been fixed, and the hermetic pins in
> `tests/test_walkthrough.py` have been rewritten to pin the *fixed* behaviour
> rather than the defect; each replacement names the commit that closed it in its
> own docstring. Read the reproduction lines below as the record of what was true
> on the recorded run, not as a statement about the current tree.
>
> | Finding | Closed in | How |
> |---|---|---|
> | WALK-DEFECT-1 — every clickable citation 404s | `5588fa9` | Q&A now retrieves over the session's **registered** document rather than one rebuilt from flat text, so an `ev_*` id and the id the evidence panel addresses belong to the same paper. |
> | WALK-DEFECT-2 — no relevance floor; the deterministic refusal unreachable | `b8ca091` | A per-leg relevance floor was added. "What is the capital of France?" now retrieves nothing, `insufficient_evidence` is true, and the confidence verdict rejects a claim resting on a passage the floor rejected. |
> | WALK-DEFECT-3 — the evidence/claim surfaces ignore the language switch | `2ace411` | The render-time strings moved into the catalogue in both languages, and both surfaces now subscribe to `onLocaleChange` — a panel that renders once is now repainted. |
> | WALK-DEFECT-4 — the reparse control advertises a parser the default cannot run | `2ace411` | Closed by **labelling**: the options are kept (filtering them means hardcoding "legacy only", which is wrong on a gateway deployment) and each carries a requirement naming `PARSER_BACKEND=gateway` and the `409 parser_unavailable` a reader would otherwise meet unexplained. |
>
> `PARSER_BACKEND` still defaults to `legacy`, so a reader on this deployment still
> meets the 409 — the routing default is unchanged. What changed is that the
> control now says so before they do.

the smallest reproduction. Nothing here was fixed by this change — these belong
to the owning modules.

### WALK-DEFECT-1 — Every citation the reader can click 404s *(worst)*

The promise "click a citation and it opens the passage" is the product's whole
pitch. In the browser it fails for **every** citation. The route itself is fine —
it resolves a `chunk_*` id and even resolves an `ev_*` id — but the two halves of
the click are **addressed to different documents**:

- The browser asks through `/api/ask/stream`, which retrieves over a document
  re-derived from the session's **text** (`parsed_document_from_text`,
  `orchestrator.py:635`), whose `document_id` is a **content hash of that text**
  (a different id from the parsed document).
- The citation's `ev_*` id is therefore minted over the **session-text** document.
- The evidence panel pairs that id with `state.currentDocumentId`, which is the
  **analyze** result's document id — the **parsed** document
  (`sections.js:150`, `evidence.js:303` → `config.js:66`).

The evidence route only consults the chunks of the document named in the URL
(`routes_evidence.py:213-219`). The citation's chunk lives in the *other*
document, so the id never resolves: **404 `evidence_not_found`, 0 of 5 chips open.**

- **Owner:** the evidence-id/document-id contract across `analysis` → `api` →
  frontend. The route is correct; the *caller* pairs an id with the wrong
  document.
- **Where:**
  - `templates/static/js/evidence.js:303` — `openEvidence(state.currentDocumentId, button.dataset.evidenceId, …)`.
  - `templates/static/js/sections.js:150` — `state.currentDocumentId = data.document_id` (the parsed doc).
  - `paperwhisperer/api/routes_qa.py:109-119` (`build_qa_analyzer`) — sets `document_content` (text) but not `qa_document`, so Q&A retrieves over the hashed text document.
  - `paperwhisperer/analysis/orchestrator.py:635-654` — `parsed_document_from_text` mints `doc_<sha256(text)>`, a different id from the parsed doc.
  - `paperwhisperer/api/routes_evidence.py:213-219` — resolves only against the named document's chunks.
- **Smallest reproduction:** upload the corpus PDF, drive the browser's Q&A route
  (`/api/ask/stream`) or reproduce it by minting a citation over
  `parsed_document_from_text(loaded.text)`, then
  `GET /v1/documents/{parsed_doc_id}/evidence/{that ev_id}` → **404**. The *same*
  `ev_id` against the session-text document → **200 with the passage** (this is
  the control that proves the id is real and the route works — pinned in the test
  file).
- **Why the tests are green:** `tests/test_evidence_api.py` (79 cases) addresses
  `chunk_*` ids only, and unit-tests never run the browser's session-QA →
  panel pairing. The defect exists **only in composition**, which is precisely
  what the walkthrough exercises.

### WALK-DEFECT-2 — The deterministic refusal is unreachable; the model decides

The product "was rebuilt around" refusing a question it cannot answer, and the
refusal is deterministic by design: when retrieval returns nothing, a fixed
sentence is returned and **the model is never called** (`_insufficient_answer`,
`orchestrator.py`), which is what makes "no citation can be manufactured"
guaranteed. But that gate is keyed on **`insufficient_evidence == (no evidence
returned)`** (`rag/retrieve.py:1216-1218`), and the QA retrieval plan has
**`final_k=8` with no relevance floor** (`rag/retrieve.py:877-884`). The fuzzy and
neighbour legs pad *every* question to a full set. So for a question the paper
cannot answer, retrieval returns 8 passages at near-zero relevance (~0.03),
`insufficient_evidence=False`, and the model **is** called and decides whether to
refuse.

- **Observed (deterministic, no model):**
  - "What accuracy did the paper report on ImageNet top-1?" → 8 passages, top score 0.032
  - "What is the capital of France?" → 8 passages, top score 0.031
- **Owner:** the refusal gate — retrieval relevance floor (so the fixed refusal is
  actually reachable).
- **Where:** `paperwhisperer/rag/retrieve.py:877-884` (QA plan, no `min_score`);
  `paperwhisperer/rag/retrieve.py:1216-1218` (`insufficient_evidence` = emptiness only).
- **Why it matters:** whether an unanswerable question is refused or answered is
  now entirely up to the model given irrelevant passages. Worse, the confidence
  *verdict* is also structural, not a relevance judgement: `_derive_confidence`
  (`orchestrator.py:497-531`) calls a claim verified when it cites any retrieved
  passage and quotes nothing, so a **paraphrase of one of the 8 padded passages
  is labelled `supported`**. This is the chain behind the "single most misleading
  thing" above. The *guarantee* the interface projects ("nothing here is
  presented as the paper's own claim" / a `supported` badge) is, for this class of
  question, the model's discretion over padded passages, not a system finding
  about the paper.
- **Smallest reproduction:** retrieve over the corpus PDF with the "capital of
  France" question: 8 passages, `insufficient_evidence=False`. Pinned in
  `tests/test_walkthrough.py` (an empty scope still gates, so the mechanism is
  sound — only the floor is missing).

### WALK-DEFECT-3 — The evidence/claim surfaces ignore the language switch

Switching 中文 ↔ English translates the static markup and re-renders the
**parse-status** card (`parseStatus.js:618` subscribes to `onLocaleChange`). It
does **not** re-render the evidence panel or the claim renderer — they produce
their own text at render time and hardcode it. After switching to English, a
reader sees an interface that mixes languages exactly where the citations are.

- **Hardcoded render-time strings with no `t()` key:**
  - `claims.js:197` — `逐字引用：` / `归一化后逐字对齐：` (direct-quote prefix)
  - `claims.js:251` — `转述（未通过逐字回源校验）：` (paraphrase prefix)
  - `evidence.js:166` — `第 ${label} 页` (evidence page label)
  - `evidence.js:200-204` — `引用 …` / `来自已核对的原文` / `片段已截断` (snippet meta)
- **Owner:** front-end i18n coverage of the evidence/claim surfaces.
- **Evidence it made me look:** only `parseStatus.js` subscribes to
  `onLocaleChange`; none of these four literals appear in `i18n.js`.
- **Smallest reproduction:** switch to English, open any answer: the claim prefix
  and evidence meta stay in the previous language while the surrounding card
  switches.

### WALK-DEFECT-4 — The reparse control advertises a parser the default can't run

`docs/PARSER_BAKE_OFF.md` §6 selected **docling-first**. The shipped default is
`PARSER_BACKEND=legacy` (`core/config.py`), which bypasses the gateway and runs
the legacy adapter (no bbox, so citations get a page but never a rectangle). Yet
the reparse dropdown still offers **Docling (local)** and **Docling with forced
OCR**, which under the shipped default answer `409 parser_unavailable` ("the
gateway has rolled back to the legacy adapter"). This is D4 from
`E2E_JOURNEY.md`, still live — but note the front end *does* surface the 409
honestly in the card rather than failing silently, which is an improvement over
a silent no-op.

- **Owner:** parser routing default (`core/config.py`) versus the reparse catalogue.
- **Smallest reproduction:** `POST /v1/documents/{id}/reparse {"parser":"docling"}`
  → `409 parser_unavailable` under the shipped default.

---

## What I could NOT verify without a browser

This repository has **no browser harness**, and adding one is out of scope. The
walkthrough drove the real HTTP surface and read the real HTML/CSS/JS the browser
would receive. The following are therefore **UNVERIFIED** — I am not claiming a
visual improvement or defect I have not seen:

- **Whether the evidence panel *looks* right** when open (layout, the schematic
  page map, the "非原始渲染图" badge, the highlight placement). The data and the
  CSS contract are correct; the pixels are unconfirmed.
- **The mobile bottom sheet's appearance and behaviour** at ≤1080px. The CSS
  breakpoint and the JS `EVIDENCE_SHEET_MEDIA_QUERY` agree (both `1080px`), and
  the scroll anchor is captured on open and restored on close, and the composer
  input is never touched — so "the question you were reading survives" is
  supported by code. Whether it *feels* right in a real viewport is unconfirmed.
- **Whether a language switch actually repaints** the parse card in place
  (asserted from code; not seen).
- **A live mid-analysis disconnect** (step 9) — the reconnect contract is asserted
  from code and the routes exist, but disconnecting a real running analysis needs
  the model.
- **Reading the live answer by eye** (step 2) and **reading a live refusal by eye**
  (step 4) — both need the endpoint.

---

## Cost and duration

- **Deterministic steps (no model):** seconds. One upload (parsed once, reused),
  one scanned upload, one docling-reparse attempt, retrieval-only probes. The
  parse is reused across assertions.
- **Model-backed steps:** the analysis and two live questions were attempted and
  every one was a 429. **0 completions served.** On an available endpoint the
  analysis alone is ~12-16 completions.

---

## Reproducing this record

```bash
# the narration (drives the real routes; model steps skip on 429)
python3 tools/user_walkthrough.py --base-url http://127.0.0.1:5000
python3 tools/user_walkthrough.py --no-llm        # only steps that need no model

# the hermetic pins — never needs an endpoint, never installs anything
python3 -m pytest tests/test_walkthrough.py -q -p no:cacheprovider
```

Neither touches PostgreSQL, installs anything, or writes outside the app's own
managed upload/object-store directories. `tests/test_walkthrough.py` builds its
documents in memory and its store on a throwaway SQLite file; the defects are
pinned as `xfail(strict=True)`, so a fix turns the xpass into a failure that
forces the marker to be retired deliberately.

**A note on the skips.** Steps 1 (cards), 2 (live answer), 4 (live refusal), 7
(populated export) and 9 (live disconnect) needed the endpoint and could not run
to completion. The two defects that concern the central claim — WALK-DEFECT-1 and
WALK-DEFECT-2 — are **model-independent** and were confirmed by driving the real
routes and the real retriever with the endpoint refusing every call. When the
endpoint is available, re-run to add the by-eye judgement on the live answer and
the live refusal to this record.