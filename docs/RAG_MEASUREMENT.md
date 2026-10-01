# RAG measurement: what was measured, and how weakly

This is the record for `tests/test_rag_end_to_end.py`. The test asserts
*capabilities*; the *recall numbers* live here, with their sample size and
interval attached, so that deleting an assertion does not delete a measurement.

The short version: **the dense-leg effect is real and is not a regression, but
this sample cannot establish it, and the test no longer pretends that it can.**

## 1. The measured numbers

Corpus: 5 real papers, 87 chunks, 5 documents. Model: multilingual MiniLM at
384 dimensions (`fastembed` local provider). Recall@8, K=8. Reproduced
identically in 12 consecutive runs.

| question set (n=7 each)   | leg mix      | R@1  | R@8  | Wilson 95% CI on R@8 |
|---------------------------|--------------|------|------|----------------------|
| exact-term                | hybrid       | 0.43 | 1.00 | [0.65, 1.00]          |
| exact-term                | lexical only | 0.86 | 1.00 | [0.65, 1.00]          |
| paraphrase                | hybrid       | 0.29 | 0.43 | [0.16, 0.75]          |
| paraphrase                | lexical only | 0.14 | 0.29 | [0.08, 0.64]          |
| paraphrase                | dense only   | 0.29 | 0.71 | [0.36, 0.92]          |

Every interval is wide enough to swallow the effect it is supposed to bracket.

## 2. What the failing assertion actually rested on

The failure was:

```
AssertionError: 加入 dense 通道后，改述类问题的 recall@8 没有提高
（lexical only 0.43 -> hybrid 0.43）
```

That was `assert paraphrase_hybrid[1] > paraphrase_lexical[1]` on 7 questions.
The per-question breakdown of the gap:

| # | question (abridged)                            | hybrid | lexical |
|---|------------------------------------------------|--------|---------|
| 1 | What wraps each sub-layer so gradients flow?   | hit    | hit     |
| 2 | Which skip connection wins at great depth?      | miss   | miss    |
| 3 | Size of the joint EN-FR vocabulary?             | hit    | hit     |
| 4 | Deep vs plain model, single-crop accuracy?      | miss   | miss    |
| 5 | Which training set was not publicly released?  | **hit** | **miss** |
| 6 | Queue backlog over the course of the morning?   | miss   | miss    |
| 7 | Plan for the document parsing queue next quarter? | miss | miss    |

The asserted gap is carried by **one question out of seven**. Six of seven
questions are identical between the two arms. Consequences, all computed
rather than argued:

* **Leave-one-out.** Drop question 5 and the asserted gap is exactly **0.000**
  (0.33 vs 0.33). Drop any other question and it is +0.167. The assertion's
  entire sign rests on a single row.
* **Exact paired test.** McNemar on the paired outcomes: 1 discordant pair in
  favour of hybrid, 0 the other way, **p = 1.000**. The paired design at n=7
  cannot distinguish the two systems at *any* confidence level.
* **Even the "large" effect.** dense-only (0.71) vs lexical-only (0.29) is the
  effect the task brief cites as real. It has 3 discordant pairs, all one way,
  **p = 0.250**. Directionally encouraging; not significant.
* **Quantum.** One question = 0.143 of recall. The asserted gap was 0.143.
  A test asserting a signal exactly equal to its own quantisation step is not
  measuring a margin.

## 3. Why it failed intermittently — it is not the model

The brief anticipated the cause as a shared, non-deterministic endpoint. That
is wrong here, and worth recording because it points at a different fix:

* **The model is deterministic.** 12 consecutive runs of the file produced
  byte-identical recall tables. `embed_query` returns **bit-identical**
  vectors whether called batched or one text at a time (max abs difference
  **0.0**). There is no run-to-run variance to average over.
* **The corpus is the volatile input.** `eval/corpus/` is **gitignored** (48
  PDFs on disk, 0 tracked), and `eval/download_arxiv.sh` fetches from
  `https://arxiv.org/pdf/<id>` with **no version pin** — a fresh fetch can
  return a different arXiv revision. Different bytes → different chunk
  boundaries → a different set of chunks counts as "gold" for a question →
  a question flips. Every paraphrase question here has exactly **one** gold
  chunk, so a single boundary shift is a total loss for that question, with no
  partial credit.

That is the honest explanation for an intermittent `0.43 -> 0.43`: a corpus
refresh moved a boundary. It is not a regression in the dense leg. The effect
was re-measured after the change and is unchanged.

## 4. What the test asserts now

Capabilities — facts that survive a corpus refresh or a model swap:

1. **The dense leg abstains never** on these 14 queries. If it starts
   abstaining or degrading, that is a real fault and a *better* signal than any
   recall delta, because it means the leg never voted and every measured
   difference is the lexical leg's own number.
2. **The dense leg's votes are load-bearing**: turning it on
   (`dense_weight=1.0` vs `0.0`) reorders the fused top-8 on **7/7** paraphrase
   questions. A leg that runs but never changes the answer is decoration.
   This checks the *order*, which is what fusion produces; recall aggregates
   away exactly the signal a capability claim needs.
3. **Exact-term recall@8 must not get worse** — kept, but only because it is a
   non-regression guard, not a quality bar. Both arms sit at the 1.00 ceiling,
   so it can only fire if something is genuinely broken. It is not evidence
   that the dense leg helps.

The recall table is still computed and printed on every run, with an inline
note pointing here. The numbers were not deleted; they were moved to where a
delta on n=7 belongs.

## 5. Sample size for a defensible recall claim

| question                                                      | n needed |
|---------------------------------------------------------------|----------|
| detect hybrid 0.43 vs lexical 0.29 (delta 0.14), 80% power    | ~181 per set |
| detect dense-only 0.71 vs lexical 0.29 (delta 0.42), 80% power | ~19 per set  |
| exact paired test, all discordant pairs one way (p < 0.05)     | ≥6 discordant pairs |
| T4.3 gate: one-sided 95% CI on a number near 0.90, ±10%       | 25 labelled queries |
| T4.3 gate: ±5%                                                  | 98 labelled queries |
| T4.3 gate: ±2%                                                  | 609 labelled queries |

(Power figures are the conservative unpaired two-proportion approximation; a
paired design needs somewhat fewer, so treat them as an upper bound on n.)

**A 7-question sample cannot support a recall claim of any kind.** It cannot
support the 0.14 gap the test was asserting, and it cannot support the 0.42
gap that looked large. For the gate the project actually cares about
(Recall@K ≥ 90%), roughly **100 labelled questions** are needed just to put a
±5% interval on a number, before the categories are split.

## 6. This file is not the release gate

The plan's gate is Recall@K ≥ 90% with Chinese, English, table, formula and
prose counted **separately**. `docs/RELEASE_GATES.md` records
`retrieval_recall_at_k` as **BLOCKED** on the absence of a labelled set
(`eval/labels/retrieval_run.json`). Nothing in this test counts those
categories separately, nothing here reaches 90%, and a 7-question sample cannot
stand in for a gate that is, correctly, still blocked. The gap between this
file and the gate is the labelling work, not more code.

## 7. What else the measurement reveals about the dense leg

* **The tie-band abstention never fires on this corpus.** 0 of 14 queries, and
  it is not close: the measured score spread across the tie window was
  0.10–0.20 against a 0.02 band (`DENSE_TIE_BAND`). The model has a confident
  opinion about every question, including exact-term questions where that
  opinion is wrong often enough to demote passages the lexical leg had ranked
  first — 39 chunks the model preferred each contributed a rank to the fusion,
  with nothing to veto them. A confidence-based abstention catches a model
  with no opinion; this model has a confidently wrong one. `dense_weight` is
  the dial and tuning it is not this file's job.
* **The exact-term rank-1 trade is real in direction.** recall@1 drops
  0.86 → 0.43 when the dense leg is added to exact-term questions, while
  recall@8 stays pinned at 1.00. The direction is established; the magnitude
  is not, at this n.
* **The corpus needs pinning before any recall number is quoted across
  machines.** `eval/corpus/` is gitignored and the arXiv fetch is unpinned, so
  two machines can legitimately measure two different corpora under the same
  test name. Until the manifest's `corpus_version` is recorded alongside a
  recall figure, no recall figure is comparable to any other.
