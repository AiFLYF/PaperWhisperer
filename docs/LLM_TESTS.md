# Live LLM tests — how to run them, and what they are worth

Two files in this repository ask a **real** OpenAI-compatible model real
questions, about a real PDF from `eval/corpus`, through the real product code:
`tests/test_llm_integration.py` and the model-backed steps of
`tests/test_e2e_journey.py`.

This document exists so that a future maintainer knows, before running them,
**which ones they can trust unattended, which ones make real billable calls,
and which failures mean the product is broken rather than the weather.** These
suites were flaky at random for a while; the causes were three different things
and they are now told apart explicitly. Read the classification before you
trust a green run — and read it before you trust a red one, because a red run
from these files is *usually* not a broken product.

---

## TL;DR for a maintainer in a hurry

- These are **opt-in**. Without `OPENAI_BASE_URL` / `OPENAI_API_KEY` /
  `OPENAI_MODEL` in the project `.env`, the model-backed tests **skip** and the
  hermetic suite (`make test`) stays free and green.
- A **red** result from these files means the product is broken — *unless* the
  run also skipped some tests, in which case the endpoint was busy and the
  failures that remain are still the only real signal.
- A **skip** means the endpoint would not serve us (timeout / 429 / connection),
  or the model declined to answer on every allowed draw. A skip is **not**
  evidence the product works; it is evidence the endpoint was not usable. Do
  not read "9 skipped" as "all green".
- They cost real tokens against a **shared** endpoint. Budget for it (§ Cost).

---

## How to run

The full command, against the configured endpoint, with the corpus present:

```bash
python3 -m pytest tests/test_llm_integration.py -q -p no:cacheprovider
python3 -m pytest tests/test_e2e_journey.py      -q -p no:cacheprovider
```

Both need three things in the project `.env` (all three, or they skip):

| Var | Example | Why it is required |
|---|---|---|
| `OPENAI_BASE_URL` | `http://127.0.0.1:8084/v1` | a base URL with no key is an anonymous endpoint |
| `OPENAI_API_KEY` | `…` | without a key nothing is served |
| `OPENAI_MODEL` | `space-bunny-free` | a key with no model falls back to a stranger's default |

They also need a real corpus document at
`eval/corpus/digital_single_col/hinton_distillation.pdf`. A clean clone has no
`eval/corpus` (it is gitignored) and both files skip.

They make real, billable calls and are **never** part of the default `make test`
signal. Do not add them to CI without reading § Which tests are worth CI.

---

## The failure taxonomy (the thing to actually understand)

Every failure in these files is sorted into exactly one of three kinds **before**
anything is asserted. The classifier is `tests/helpers.py::
classify_live_failure`, and it is shared by both files on purpose — see § What
the flakiness actually was.

| Kind | Means | What the test does | Why |
|---|---|---|---|
| **ENDPOINT** | The endpoint would not serve us: timeout, 429, connection failure, gateway 5xx. | **Skip**, naming the endpoint. Never retried. | Says nothing about our prompts, parsing or output handling. The endpoint is shared with other work, so retrying an outage spends *somebody else's* quota to fix *our* test. |
| **MODEL** | The endpoint served us and the model is the difficulty: it reached for a tool this product never offers (a `finish_reason="tool_calls"` envelope with empty content), or drew an answer that didn't carry the fact. | **Retry**, a bounded number of draws (3 for a question, 2 for the whole analysis), printed. Exhausting the budget still fails. | A sampled model is not a function of its prompt. The same question gets a good answer on the next draw. |
| **PRODUCT** | Everything else: a 400 for a request we built, an empty section, a mermaid payload still carrying its code fence, a fabricated answer. | **Fail immediately. Never retried.** | This is the branch that keeps the retries honest. A retry that covered PRODUCT would turn a broken build into a green suite. |

The clause in **bold** is the one that makes the retries safe to have. It is
enforced by tests, not by good intentions — see § The retry cannot hide a
regression.

---

## What each test actually proves (and what it does not)

`tests/test_llm_integration.py`:

| Test | Proves | Does **not** prove | CI? |
|---|---|---|---|
| `test_complete_returns_non_empty_text` | A real completion comes back non-empty, answers, and the client's HTML/empty guards don't misfire. | That the client is fast or cheap. | Local |
| `test_complete_surfaces_a_bad_request_…` | A 400 the product *built* surfaces as a classified error, not silence — the two disagree about where errors appear (SDK raises before the client inspects the status). | That the client doesn't swallow errors *in general* — see the SKIP caveat in its body; the swallowing bug itself is covered offline in `test_llm_client.py`. | Local |
| `test_stream_yields_incremental_text` | The transport parses real SSE (`data:`, keep-alives, `[DONE]`) and the probe buffer flushes a short response instead of dropping it. | — | Local |
| `test_analyze_real_corpus_document` | **The headline.** A real two-column paper → parse → chunk → merge → six sections → bookkeeping, with a model that was not told in advance what to return. Required sections populated, Chinese output, mermaid has no code fence, parse provenance local, `analysis_status` internally consistent. | That every run is flawless — an *optional* section failing is a legal, reported outcome, not a failure. | Local (expensive) |
| `test_qa_answers_a_question_using_evidence…` | A question the paper answers is answered *from the paper* — the 1200-units figure is stated. Checked against the paper, not a fixture, so this assertion had literally never been made before. | — | Local |
| `test_qa_refuses_instead_of_inventing…` | An unanswerable question (ImageNet top-1, which the paper never reports) produces a refusal, **and** the presentation gate holds: quotation marks only ever wrap verifier-aligned text. | That the model "didn't fabricate a figure" by prose-pattern-matching — five attempts at that check are documented in the body and all punished correct answers, so it is deliberately **not** asserted. | Local |
| `test_the_late_evidence_is_now_reachable_by_retrieval` | The passage at offset 12132 — past the old 12000-char head window — is reachable by retrieval. **No model call; deterministic.** | — | **Yes — this one is CI-safe.** |
| `test_stream_answer_question_delivers_a_gradual_answer` | The SSE-driven streaming path really streams and ends with the structured `AnswerResult`, pieces agreeing with the final answer. | — | Local |
| `test_full_run_stayed_within_the_call_budget` | The run stayed within its stated completion ceiling. | — | Local |
| the eight `test_*retry*` / `test_classifier_*` / `test_analysis_*` guards (§ below) | The retry machinery cannot hide a regression. **No endpoint, no corpus, no credentials.** | — | **Yes — always run.** |

`tests/test_e2e_journey.py` (model-backed steps only):

| Step | Proves | CI? |
|---|---|---|
| the deterministic retrieval steps | The assembled product reaches the late evidence with **no model in the loop** — the project's central claim. | **Yes — CI-safe.** |
| `test_step4_…section_statuses…` | Required sections succeed through the real `/api/analyze` and reported quality matches actual statuses. | Local |
| `test_step5_live_…claims_and_citations` | The model-backed answer reaches the late fact and carries claims/evidence ids/confidence. | Local |
| `test_step9_export…` | A recorded, still-open defect — strict `xfail`. | Local |

---

## Which tests are worth CI, and which are for local use

The rule is simple: **a test belongs in unattended CI only if it is
deterministic.** Every test that reaches a real model is, by definition, not.

**Worth running unattended (safe today):**

- The whole hermetic suite (`make test`) — including the eight retry/classifier
  guards from `test_llm_integration.py`, which exercise the harness against
  stubs and make no network call.
- The deterministic retrieval tests in `test_e2e_journey.py` (the central
  claim) — no model, no endpoint.
- `test_the_late_evidence_is_now_reachable_by_retrieval`.

**For local use only (they need a live endpoint and real budget):**

- Everything that makes a completion: the transport tests, the analysis, the Q&A
  tests, the streaming Q&A, the e2e model-backed steps.

If CI ever needs a live smoke signal, the cheapest honest one is
`test_complete_returns_non_empty_text` (a single "ok" completion). Anything
richer will flake for reasons that have nothing to do with the commit.

---

## Cost

Measured off a clean run of `test_llm_integration.py`: **~22 real completions**
and ~6–8 minutes of wall clock against the configured endpoint. The two heaviest
items are the analysis (≈16 completions) and the two Q&A questions.

The file asserts a **ceiling** on that number (`CALL_BUDGET`), and that ceiling
is *computed from the retry policy*, not typed by hand — so raising a retry
budget cannot silently leave the ceiling behind. The realised cost is printed at
session teardown, along with the number of model redraws the run needed. A run
that needed redraws is green but is telling you something; read the teardown
line.

---

## The retry cannot hide a regression

This is the property everything else rests on, so it is checked rather than
asserted. `test_llm_integration.py` contains eight guard tests that need no
endpoint: they drive the real retry helpers with stubs that reproduce each
failure mode and pin the *outcome*.

The two that matter most:

- `test_retry_does_not_hide_a_wrong_answer` — a model that reaches for a tool on
  **every** draw must make the test **fail**, after spending exactly its budget.
  Not pass. Not skip.
- `test_analysis_product_failure_is_not_retried_or_skipped` — an analysis whose
  required section fails for a **product** reason must fail on the first
  attempt, be asked only once, and specifically must **not** come back as a skip.

That second test uses a helper (`_must_fail_hard`) that converts a skip into a
failure, because a plain `pytest.raises(AssertionError)` lets a `Skipped`
exception escape and the guard would then be reported as *skipped* — which, in a
suite full of deliberate environment skips, looks identical to healthy.

**These guards were themselves mutation-tested.** Two realistic bad changes were
applied to the retry and the guards were re-run:

1. *Let the Q&A loop give up and return an empty answer instead of failing when
   its budget is spent* → `test_retry_does_not_hide_a_wrong_answer` **fails**.
2. *Turn a product-flavoured analysis failure into a skip* →
   `test_analysis_product_failure_is_not_retried_or_skipped` **fails** ("it
   produced a SKIP, which would hide a real failure").

So the guards are not decorative: they catch exactly the two ways a retry could
quietly turn a broken build green.

---

## Measured flakiness (before and after)

"Before" is the code at this commit's parent; "after" is this change. Both sets
were run to completion, ten times each.

Ten consecutive runs of `test_llm_integration.py` against
`http://127.0.0.1:8084/v1` / `space-bunny-free`, corpus present.

| | Before (this commit's parent) | After (this change) |
|---|---|---|
| Runs | 10 | 10 |
| **Red runs** | **3** | **0** |
| Green runs | 7 | 10 |
| Failures, all of one kind | `test_analyze_real_corpus_document` — an endpoint timeout made `summary` report `failed`, and the test asserted `success` | none |
| Runs with an endpoint skip | 0 | 1 (run 2: `summary` timed out; skipped as an environment outcome) |
| Worst-case completions in a run | 22 | 21 |

**Read that honestly.** The three "before" failures were *all* the same bug —
an endpoint outage reported as a product defect — and every one of them is now a
skip or a pass. What this change did **not** demonstrate, in these ten runs, is
the model-redraw path firing against the live endpoint: on a quiet endpoint the
live model made **zero** tool-calling redraws across all ten runs (the "4
redraws" the first after-runs printed were an artifact of the stub guard tests
inflating the counter, since fixed with a save/restore fixture — see
`_clean_redraw_count`). So the redraw machinery is proven correct by the
mutation-tested guards, not by these ten runs; the flake these ten runs
actually measured and fixed was the endpoint-timeout misclassification.

Read a skip as "the endpoint/model was not usable this run", not as "the test
passed". A run that skips reports *less* signal, not more — re-run it, or rely
on the deterministic tests for real signal.

---

## What the flakiness actually was (so you don't rediscover it)

Three distinct causes, found by reading the code and confirmed against the live
endpoint. They are listed here because each has a lesson a future maintainer
should not have to relearn the hard way.

1. **An endpoint outage was reported as a product defect.** The two live files
   each carried their *own* copy of "is the endpoint unavailable?", and the two
   copies disagreed. Concretely: one shared-endpoint **timeout** during the
   analysis made a required section report `failed`, and the test asserted
   `status == "success"` — so one busy shared server turned the analysis test
   red while three sibling tests, on the *identical* error, skipped correctly.
   That misclassification alone accounted for every observed failure in the
   baseline runs. **Fixed** by classifying a failed required section's error
   before judging it, and by making one shared vocabulary (in
   `tests/helpers.py`) that both files import.

2. **The Q&A retry loop was dead code.** `_ask_until` handed its inner helper a
   budget of one, so the first tool-calling response ended the helper with an
   `AssertionError` the outer loop did not catch. Three attempts were
   documented; one happened in practice — on exactly the failure mode the loop
   existed to absorb. **Fixed** by collapsing the three overlapping ask-helpers
   into one loop that owns every retry, so there is a single retry policy to
   reason about.

3. **One test asked the same question twice.** The refusal test called the model
   once for the prose and again for the claims. Two independent draws: twice the
   cost, twice the chance the model reached for a tool, and a test whose two
   halves could disagree because they were answered by different samples.
   **Fixed** by reading the prose and the claims off a single answer.

A fourth, smaller one: the evidence Q&A test's loop accepted an answer spelling
"1200" as "一千二", and the very next assertion then rejected that same answer
for lacking "1200" — a guaranteed flake on any draw where the model spelled the
number out. **Fixed** by having one predicate and reusing it.

---

## Product findings (reported, not fixed here)

These files own no production code, so the following are **reported, not
patched**, with locations:

- **The streaming path cannot see a tool call.**
  `paperwhisperer/llm/client.py` — `LLMClient.stream` reads `delta.content` and
  never inspects `finish_reason`. The non-streaming path
  (`_extract_completion_text`) detects the tool-calling envelope and raises a
  clear non-retryable error; the streaming path sees the same envelope as an
  *empty completion* ("AI 服务返回内容为空") with the tool call invisible, and
  retries it. **Impact:** the questions a user asks *because* a document does
  not cover their need fail on the streaming (SSE) route with a misleading
  error. The two paths disagree about the same event. Reported at
  `client.py` `stream()` / `_parse_stream_delta`.

- **A tool-calling refusal is never re-drawn by the client.**
  `LLMClient.complete` treats the tool-calling envelope as final
  (`NonRetryableError`) on the reasoning that "re-sending an identical prompt
  can only produce an identical answer". That is true of a deterministic model
  and false of a sampled one. Measured on the configured endpoint, the model
  emits that envelope on roughly half the no-evidence questions. **Impact:** for
  a real user, the most natural question to ask a paper ("what accuracy did it
  report on X?") intermittently returns an error rather than an answer. The
  fix — a bounded re-draw of a refusal-class response, distinct from a
  transient-outage retry — is a product decision and is *not* made here.
  Reported at `client.py` (the `NonRetryableError` branch in `complete()`) and
  `paperwhisperer/core/errors.py` (the `NonRetryableError` docstring, which
  states the same determinism assumption).

---

## What was deliberately left as-is

- **`test_qa_refuses_instead_of_inventing…` still ends in a conditional
  `xfail`.** `_derive_confidence` reports a *correct* refusal as `supported`,
  because it measures whether a claim is carried by evidence — not whether the
  document answers the question. This is a known design gap, documented in the
  test body, and closing it needs a signal that does not come from the model.
  Left visible (xfail, not hidden) rather than "fixed" by weakening the test.
- **The refusal test still does not assert "did not fabricate a figure" by
  prose-matching.** Five attempts at that check all punished correct answers
  (documented in the body); a sixth would produce a test that is red on correct
  behaviour and green on a well-formatted fabrication.
- **The streaming test still retries on an empty stream.** Because from inside
  the test a tool-calling stream and an empty stream are the same event (see
  the product finding above), retrying is the only honest lever until the
  product distinguishes them.
- **The `CALL_BUDGET` ceiling is the designed worst case, not the typical run.**
  A typical run spends ~22 completions; the ceiling is higher because the retry
  budgets are. This is deliberate — a ceiling below a legitimate unlucky run's
  cost would be a new flake of its own.