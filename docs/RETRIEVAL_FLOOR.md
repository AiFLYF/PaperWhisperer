# The retrieval relevance floor

WALK-DEFECT-2, fixed. `docs/USER_WALKTHROUGH.md` §WALK-DEFECT-2 recorded the
finding; this document records the measurement it was fixed with, because a
threshold chosen by taste is a threshold that will be wrong on the first
document that does not resemble the corpus it was chosen against.

**The defect, in four links, all of them deterministic:**

1. Retrieval had no relevance floor. The fuzzy and neighbour legs padded every
   question to `final_k` passages.
2. `RetrievalResult.insufficient_evidence` was `not self.evidence`, so it could
   not be `True` for any question a leg produced candidates for.
3. `_insufficient_answer` — the fixed, model-free refusal, the thing this product
   was rebuilt around — is keyed on that property, so it never ran. **The model
   was called**, and handed eight irrelevant passages.
4. `_derive_confidence` called a claim *verified* when it cited any id the
   retriever had minted. Identity is not relevance, so a paraphrase resting on
   one of the eight padded passages was labelled `supported`.

Measured, against `eval/corpus/digital_single_col/hinton_distillation.pdf`:
"What is the capital of France?" returned **8 passages**,
`insufficient_evidence = False`, and a paraphrase resting on any one of them
derived `supported`.

---

## 1. The measurement

Corpus: three real papers — `hinton_distillation` (two-column),
`vit_an_image_is_worth_16x16`, `attention_is_all_you_need` — plus two real
Chinese documents from `mixed_zh_en`. Dense leg: fastembed
`paraphrase-multilingual-MiniLM-L12-v2`, 384 dimensions, cosine over
L2-normalised vectors. 25 answerable English questions, 10 answerable Chinese,
11 out-of-domain English, 8 out-of-domain Chinese, 15 in-domain-unanswerable.

**"Answerable" means** a literal answer string occurs in the scoped document and
retrieval returns a passage containing it. The string, not a chunk id: a chunk id
goes stale the moment a boundary moves.

### The three classes, and why there are three

The first labelling of "unanswerable" was **wrong**, and finding that out is part
of the result. A first pass put 28 questions in one "unanswerable" bucket. Checking
each one's content words against the scoped document showed that most of them
("What accuracy did the paper report on ImageNet top-1?", "What BLEU score did
the translation model reach?") name things the document genuinely discusses.
A floor that rejected those would be claiming to answer a problem it does not
solve, and the honest number for it is 0.

So the classes are assigned **by construction**, not by whether retrieval
happened to succeed:

| class | definition | n |
|---|---|---|
| **answerable** | the answer is in the document | 25 EN / 10 ZH |
| **out of domain** | *no content term of the question occurs in the document* | 11 EN / 8 ZH |
| **in-domain unanswerable** | the content terms *are* in the document; the answer is not | 15 |

The third class is the open band (§5) and it is carried through every table below
rather than averaged away.

### The distributions

**Dense cosine, per question:**

| class | min | p10 | median | p90 | max |
|---|---|---|---|---|---|
| answerable | **0.334** | 0.381 | 0.579 | 0.726 | 0.758 |
| out of domain | 0.032 | 0.066 | 0.170 | 0.215 | 0.360 |
| in-domain unanswerable | 0.275 | 0.279 | 0.449 | 0.586 | 0.597 |

**Lexical coverage, per question (the store's own score):**

| class | min | median | max |
|---|---|---|---|
| answerable | **0.545** | 0.750 | 0.900 |
| out of domain | 0.143 | 0.500 | **0.667** |

**Content coverage — coverage over the question's *content* terms only:**

| class | min | p25 | median | max |
|---|---|---|---|---|
| answerable (EN+ZH) | **0.333** | 0.600 | 0.750 | 1.000 |
| out of domain (EN+ZH) | 0.000 | 0.000 | 0.000 | **0.000** |
| in-domain unanswerable | 0.167 | 0.333 | 0.600 | 1.000 |

**RRF fused score — `Evidence.score`, the number the walkthrough quoted:**

| class | min | median | max |
|---|---|---|---|
| answerable | 0.030 | 0.032 | 0.033 |
| out of domain | 0.031 | 0.032 | 0.033 |

Chinese answerable content coverage, measured separately because it is a
different tokenizer and deserves to be checked rather than assumed: min **0.333**,
median 0.598, max 0.750. Chinese out-of-domain: max **0.000**, all eight.

---

## 2. Three things the measurement decided before the threshold was chosen

**(a) The floor cannot go on the fused score.** The walkthrough reported a top
score of "0.031" for "What is the capital of France?" as though it were a weak
match. It is not a weak match; **it is the score a ranked first place gets.**
RRF combines *ranks* and never looks at a score, so `Evidence.score` is
0.032 for an answerable question and 0.032 for an unanswerable one. Thresholding
it — the obvious one-line fix — does nothing at all. `FUSED` in the test file
pins this so nobody tries it again.

**(b) The floor cannot go on the lexical leg's own score either, and the reason
is structural.** Coverage is computed over *every* query token, function words
included. "What is the capital of France?" scores **0.667** against a paper on
knowledge distillation, because *what*, *is*, *the* and *of* are all in it. The
answerable *minimum* (0.545) sits below the out-of-domain *maximum* (0.667):
the two classes overlap, so no number on that scale works. Coverage rewards
matching the parts of a question that carry no information.

Restricting the ratio to tokens that carry information removes the overlap
completely. That is the whole innovation, and it is not a new scorer — it is the
existing tokenizer with a measured, auditable function-word list applied
(`QUERY_STOPWORDS`, English only, numerals deliberately kept because "300
classes" puts its content in the numeral). **No Chinese entry is needed**:
`scoring.tokenize` emits CJK runs as character bigrams, so a Chinese question was
already scored on content alone — which is exactly the property the stopword list
exists to produce, arrived at differently.

**(c) The out-of-domain maximum is exactly 0.000, in both languages.** Not 0.02,
not 0.09. Those questions' content words are absent from the document entirely,
so no coverage value above zero is reachable for them. The evidence is a **gap,
not a decision boundary**, and the threshold can be placed anywhere inside it
without touching either class.

---

## 3. Where the floor is, and why that number

`MIN_QUERY_COVERAGE = 0.20` (`rag/retrieve.py`). The measured gap is
`[0.000, 0.333]`; its midpoint is 0.167.

The floor is placed **above** the midpoint, toward the answerable side, and that
is a deliberate asymmetry rather than a rounding: the cost of being wrong is not
symmetric. One false refusal is a reader who stops asking. One padded answer is a
badge that reads "supported" about a claim the system never checked. 0.20 leaves
the Chinese answerable minimum — the tighter of the two languages, at 0.333 —
**0.133 of headroom**, against the 0.033 that the higher-looking 0.30 would have
left it.

`DENSE_MIN_SCORE = 0.40` — chosen by sweeping it, not by splitting two rows

The two cosine distributions **overlap**: the answerable minimum is 0.334 and the
out-of-domain maximum is 0.360. No threshold on them separates the classes, and
the midpoint of two overlapping distributions is arbitrary by construction. So it
was measured as a floor instead — the whole 54-question set, run at each value:

| `dense_min_score` | answerable answered | out-of-domain refused |
|---|---|---|
| 0.20 | **35 / 35** | 15 / 19 |
| 0.25 | 35 / 35 | 17 / 19 |
| 0.36 | 35 / 35 | 17 / 19 |
| **0.38 – 0.45** | **35 / 35** | **18 / 19** |
| 0.50 | 34 / 35 | 18 / 19 |

**The answerable column is flat from 0.20 to 0.45**, and that is structural rather
than lucky: a candidate survives if *any* leg that found it cleared *that* leg's
floor, so raising the dense floor costs an answerable question only when the
dense leg was its sole route to a passage that also fails the content floor.
Answerable questions reach their passage through more than one leg, so this floor
is close to free. Refusals improve until 0.38 and then stop; answers degrade at
0.50. **0.40 is the middle of that measured plateau.**

**A limitation this exposes, with its number.** The dense floor is calibrated
against a corpus whose dominant language matches the query. It is not, and the
abstention is measurable rather than hypothetical: a Chinese question against
these English papers scores cosine **0.340**, where English questions in the same
set score 0.42–0.69. Below 0.40 the dense leg abstains on it. The measured cost
of that abstention is **zero recall** — the paraphrase-set R@8 is 2/7 with the
floor on and 2/7 with it off, and that question's answer was outside the top-8
either way. The leg stops padding a result that could not have answered it.
Recorded here rather than tuned around, because tuning around it means lowering
the floor for everything to accommodate one cross-lingual regime, which is
exactly the corpus-specific overfitting this document exists to prevent.

Both numbers are asserted against the committed distribution in
`tests/test_retrieval_floor.py::TestTheMeasuredDistribution`, including that the
coverage classes still do not overlap and that the dense floor still sits inside
the measured plateau. If a future re-measurement finds they do not, that test
fails and says *"re-derive, do not move"* — which is the right instruction,
because nudging the number until the assertion passes is how a floor becomes a
guess.

---

## 4. Why the floor is per-leg

A single threshold across the legs is wrong in two opposite directions, and
**both were found by this project's own tests failing**, not by argument.

* **Demanding term overlap from the dense leg** would demand two independent
  proofs of the same fact, one of which that leg cannot give by design. "How well
  did the small model do compared to the big one on handwritten digits?" shares
  almost no content term with the passage that answers it, and its cosine is
  0.484. A floor demanding overlap would refuse every paraphrase — the class the
  Q&A plan exists for. Measured: answerable paraphrase content coverage is
  **median 0.600 against 0.750** for the class as a whole, still 3× the floor.

* **Demanding a cosine from the lexical leg** is the same error the other way,
  and worse: they are different quantities in different units, and the raw
  coverage of a lexical-only candidate is exactly the number measured above to be
  uninformative.

So **each leg is judged on its own measure, and a candidate survives if any leg
that returned it cleared that leg's floor**:

| leg | floor, in its own units | why |
|---|---|---|
| dense | `dense_min_score`, cosine | the leg abstains rather than voting if it misses, so a dense candidate's presence in the ranking *is* the finding that it cleared |
| fuzzy | `FUZZY_MIN_SIMILARITY`, enforced inside the leg | a fuzzy hit is by construction a near-miss on a term that is **not present** — `recieve` for `receive`. Asking a fuzzy candidate whether it contains the query's terms asks the one question it was built to answer "no" to. |
| lexical | `min_query_coverage`, over content terms | the lexical leg ranks by term presence; without this it ranks "what/is/the/of" matches above a real answer |

**The first version of this applied the content floor uniformly and it silently
deleted the fuzzy leg's entire reason to exist.** `tests/test_retrieve.py` caught
it — three tests around near-miss recall went empty. That is recorded in the
docstring of `_apply_relevance_floor` because the mistake is a plausible one to
re-make.

`min_lexical_score` is left at `0.0` on every shipped plan, and that is a
measured decision rather than an oversight: §2(b) shows the raw coverage scale
cannot separate the classes, and a reproduction-plan threshold set from a
question set containing no proper-noun queries would be fitting a number to
nothing. The knob is documented, not invented.

---

## 5. The open band, and what was deliberately not shipped

**The band between the floor and a confident answer is not solved, and nothing
here pretends otherwise.**

The in-domain-unanswerable class — the question's content words are all in the
document, the answer is not — is **not separated by any relevance floor on any
signal measured here**:

| | min | median | max |
|---|---|---|---|
| in-domain unanswerable, content coverage | 0.167 | 0.600 | **1.000** |
| in-domain unanswerable, dense cosine | 0.275 | 0.449 | 0.597 |
| *answerable, dense cosine* | *0.334* | *0.579* | *0.758* |

Its maxima reach or exceed the answerable maxima. "How many parameters does the
distilled model have?" against the distillation paper scores content coverage
**1.000** and cosine **0.597**. And that is *correct behaviour by this
measure*: the paper is about that model, every content term of the question
occurs in it, and the passage that scores highest is genuinely the most relevant
one available. What is missing is a specific figure.

This is not a threshold that was set too high. It is a different question —
"is this passage about the question?" versus "does this passage contain the
answer?" — and only the first is answerable from a retrieval score. Measured
effect: **1 of 15** in-domain-unanswerable questions is refused. The other 14
reach the model, and the floor gives the model nothing to refuse on.

**What was deliberately not built.** An intermediate "partially relevant" state
with a score in the band between the floor and confident, or any
calibration of `confidence` against relevance. Either would be a number with no
measured meaning behind it, rendered on screen as though it had one — which is
the defect, in a new costume. The band is documented here instead.

**What *is* honest in the band**, and is shipped: the floor's decision is
propagated rather than discarded. `RelevanceReport` travels on the result and on
the record, so the answer layer can ask whether the passage a claim rests on was
allowed to count as evidence at all. That is a binary judgement, carried, not a
score invented.

---

## 6. What the change cost

A/B over the same 54 questions, the floor run off and on:

| class | n | refused, floor off | refused, floor on | answer found off | answer found on |
|---|---|---|---|---|---|
| answerable (EN) | 25 | 0 | **0** | 25 | **25** |
| answerable (ZH) | 10 | 0 | **0** | 10 | **10** |
| out of domain (EN) | 11 | 0 | **10** | 0 | 0 |
| out of domain (ZH) | 8 | 8 | 8 | 0 | 0 |
| in-domain unanswerable | 15 | 0 | 1 | 0 | 0 |

**Zero false refusals and zero lost answers. 35 of 35 answerable questions, in
both languages, are answered exactly as before.** Out-of-domain refusals go from
8 of 19 to **18 of 19**.

**One out-of-domain question still gets through**, and it is the honest shape of
the residual risk: "How long does it take to drive from Beijing to Shanghai?"
scores content coverage exactly **0.200** against the distillation paper —
clearing the floor — because precisely one of its content terms, *long*, is a
common English word. A single common word in a question is enough to defeat a
content-coverage floor. Closing that needs a stopword list long enough to be
unauditable, or an idf weighting the store does not expose. Recorded rather than
hidden.

---

## 7. Reproducing

```bash
python3 -m pytest tests/test_retrieval_floor.py -q
```

`eval/corpus` is gitignored, so the tests that need the paper skip on a fresh
clone (`eval/download_arxiv.sh` restores it). The arithmetic the floor rests on —
does the measured distribution still separate the classes, is the floor still
inside the gap — is pure and runs everywhere. The measured distributions are
committed as data in the test file, so a change to the threshold, the stopword
list, or the embedding model can be compared against this run rather than
against the next person's recollection of it.
