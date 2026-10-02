# Refusal draws: the tool-calling rate on no-evidence questions

**Status: leave the product alone.** The premise for changing it — that "the most
natural user question intermittently errors instead of answering" — was measured
and is no longer true. It was true of a *previous* build; it is not true of this
one. This file records the measurement, the decision, and the number at which to
revisit, so the error is not re-litigated by the next person who sees it.

The question this answers: `paperwhisperer/llm/client.py` treats a tool-calling
reply as final (`NonRetryableError`; "re-sending an identical prompt can only
produce an identical answer") and deliberately does not re-draw it. A previous
agent measured that **roughly half** of no-evidence questions against the real
model produced exactly that envelope, and proposed a bounded re-draw. That
measurement predated two changes:

- `PROMPT_STABLE_PREFIX` (`paperwhisperer/llm/prompts.py`) grew the clause
  "你没有任何工具可用：不得请求文件检索、命令执行或任何外部访问。…", telling the
  model it has no tools and must not go looking for other sources;
- `LLMClient._parse_stream_delta` (`client.py`) now inspects `finish_reason`, so
  both transport paths classify the same envelope the same way.

Neither the current rate nor a fresh interval had been counted. That is what
this file is.

---

## 1. The measurement

**Date:** 2026-10-02. **Endpoint:** `http://127.0.0.1:8084/v1`, model
`space-bunny-free` (from the gitignored `.env`).

**Method.** Real completions driven through the product's *own* prompt path
(`build_evidence_answer_prompts`, `qa` task) with two real evidence passages
lifted from the corpus paper (`eval/corpus/.../hinton_distillation.pdf`, §5.5
Results), at the production temperature (`LLM_TEMPERATURE = 0.7`). Four
differently-worded **no-evidence** questions — each asks for a specific figure or
ablation the evidence does not contain — were each drawn 5 times (20 draws), with
a fixed `seed` per draw. The verdict of every draw is the product's own
`llm.output.classify_completion` on the returned envelope, not a hand judgement.

The four questions:

| # | Question | Shape |
|---|---|---|
| 0 | 作者在 ImageNet 上报告的 top-1 准确率具体是多少？ | bare metric, Chinese |
| 1 | 这篇论文用 ImageNet-21k 预训练时用了多少张图片、训练了多久？ | two figures, Chinese |
| 2 | Does this paper report an ablation on the effect of dropout rate on the JFT baseline? | yes/no ablation, English |
| 3 | 论文在 CIFAR-10 上报告的最终错误率是多少？用的是哪种数据增强？ | figure + method, Chinese |

**Result. 0 of 20 draws produced the tool-calling envelope.** Every draw came
back `finish_reason="stop"`, non-empty content, `message.tool_calls` empty — the
`ok` verdict. Point rate **0.000**; **95% Wilson interval [0.000, 0.161]**.

**What the model actually did on those 20 draws** (this matters — the fix is not
"it stopped reaching for tools, now it fabricates instead"): it refused *in
prose*, correctly and with citations. Representative outputs, verbatim:

- "依据当前文档证据，无法回答该问题：文档中没有说明 CIFAR-10 上的最终错误率…"
- "根据当前可回源校验的原文证据，**没有**找到关于 dropout 速率对 JFT 基线影响的消融实验。"

Every one names the document as the reason and cites the evidence it did get. None
invented a figure. So the behaviour we want — "the document does not say" — is
what the model now produces, and the tool-calling failure mode is gone.

**Live calls used: 25 transport attempts → 20 successful completions.** The
other 5 attempts were environment outcomes against the shared endpoint (HTTP 429
rate-limit and slow-model timeouts) and were retried; none of them are a
measurement of the product. Total wall-clock ≈ 9 minutes at the endpoint's
pacing. No other file's live budget was spent.

---

## 2. The decision: leave the product alone

The rate is not "small" — it is **zero** in the sample, with an upper bound of
~16%. The prior "roughly half" figure is refuted; this build does not have that
defect. Therefore:

- **Do not add a bounded re-draw of refusal-class responses.** There is nothing
  to re-draw. An unnecessary retry is a real cost — user-visible latency, an
  extra completion against a shared endpoint's quota — and it would
  *reintroduce* precisely the risk the current `NonRetryableError` branch
  exists to prevent: a retry that covers a failure class can mask a genuine
  regression, and the "the model drew an unlucky sample" story is exactly the one
  that turns a broken build green.
- **Do not change `client.py`.** The `NonRetryableError` treatment of a
  tool-calling envelope remains correct on both paths.
- **Do not change the prompt.** (And `prompts.py` is not owned by this task; the
  clause that fixed the behaviour is already there.)

A measurement that argues *against* acting is a result, not a non-result. The
honest read is: **the prompt fix worked, the transport fix worked, and the
premise is stale.** The correct action is to write the number down and stop.

---

## 3. What would change the decision, and the number to revisit at

**Revisit if the Wilson 95% *lower* bound** on the tool-calling refusal rate
rises above **0.10** (the `REVISIT_LOWER_BOUND` in `tests/test_refusal_draws.py`,
and the number quoted here — they are the same constant, checked by the harness).

Why the *lower* bound and not the raw proportion: a trigger on the point estimate
fires on a lucky small sample. On a 20-draw sample the lower bound crosses 0.10
at about **5 refusals in 20** (5/20 → lower ≈ 0.112); 1–2 stray refusals do not
trip it. The old "roughly half" behaviour (10/20 → lower ≈ 0.30) would trip it
many times over. The threshold is therefore sensitive to a real regression and
insensitive to noise.

Also revisit if, independently of any rate:

- the two transport paths (`complete` vs `stream`) are ever seen disagreeing about
  the same envelope again (the harness's both-paths test would fail), or
- a user reports the "模型尝试调用本产品不提供的工具" error on a no-evidence question.

**How to re-measure.** Run the driver pattern described in
`tests/test_refusal_draws.py`'s module docstring (build the real
`build_evidence_answer_prompts` prompt, draw against the configured endpoint,
classify each envelope with `classify_completion`), append the new envelopes to
`RECORDED_DRAWS`, and recompute with `wilson_interval`. The harness recomputes
the doc's number on every run, so this file and the recorded corpus cannot drift
apart silently.

---

## 4. Was the right fix somewhere else?

Considered and rejected, on the measurement:

- **Tighten the prompt further.** No. The clause that stopped the tool-calling
  is already in `PROMPT_STABLE_PREFIX`; there is nothing left to tighten for
  this failure mode, and speculative prompt edits are the kind of change that
  cannot be falsified by a number.
- **Is the question being asked wrong?** "Answer from this excerpt" does invite a
  model that wants to search. But note the product's *actual* no-evidence path
  (`DocumentAnalyzer._insufficient_answer`, `orchestrator.py`) does **not** call
  the model at all: when retrieval returns nothing it assembles a fixed
  "insufficient" sentence and never prompts. The tool-calling was only ever seen
  on the *harder* case — retrieval returned *some* evidence that did not cover the
  question — and that is exactly the case these 20 draws exercise, at 0/20. So the
  question is not wrong, and the model no longer wants to search it.
- **A bounded re-draw of a refusal** — see §2; not warranted at a measured zero.

---

## 5. Caveats, and what the measurement revealed that was not being looked for

- **20 draws bounds a low rate, not a zero one.** 0/20 rules out "roughly half"
  decisively; it does not rule out a tail of up to ~16%. The four questions are
  all *metric/figure-shaped* ("what top-1 on X?"). A materially different
  question shape is untested, and a different model against the same endpoint
  could behave differently. This is why §3 pins a revisit number rather than
  declaring victory.
- **The endpoint was under contention.** 5 of 25 attempts were 429s or
  slow-model timeouts, retried; the 20 recorded draws are all genuine completed
  generations. Heavy shared-endpoint load is a plausible reason the recorded
  sample is slightly biased toward windows the model served easily — but every
  recorded draw is a real completion with a real verdict, so the 0/20 stands as
  measured.
- **The prior "roughly half" was not fabricated; it was real once.** The
  distinction is that it described a *build* that no longer exists. The clause
  in `PROMPT_STABLE_PREFIX` and the `finish_reason` inspection in
  `_parse_stream_delta` are exactly the two changes that would be expected to
  remove it, and the measurement confirms both landed and took effect. The
  lesson the earlier note missed was recording *which build* a number was
  measured on — that is what this file is for.

## Reproducibility

`tests/test_refusal_draws.py` is a deterministic, non-live harness. It freezes
the 20 recorded envelopes, replays each through the product's own classifier and
both transports, recomputes the rate and its Wilson interval, and pins the
decision rule (including `REVISIT_LOWER_BOUND`). An autouse fixture makes any
socket use fail, so the committed suite never depends on the endpoint being up
and never spends shared quota. The refusal-*classification* path is exercised
with an explicitly **synthetic** tool-call envelope (`SYNTHETIC_TOOL_CALL`),
kept in a separate constant from the measured corpus precisely because the
measured corpus contains no refusal — a unit test of the classifier, not a claim
about how often the model refuses.
