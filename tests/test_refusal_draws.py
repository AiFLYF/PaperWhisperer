"""Refusal-draw measurement harness — deterministic, non-live.

This file exists so the decision recorded in ``docs/REFUSAL_DRAWS.md`` can be
re-derived **offline**, against no endpoint, with the same numbers every time.
It does two jobs and is deliberately not a third:

1. **Replays a recorded corpus of real response envelopes** (the 20 live draws
   measured against the configured endpoint on 2026-10-02; see the doc) through
   the product's *own* classification — ``llm.output.classify_completion`` and
   the two transports ``LLMClient._extract_completion_text`` /
   ``LLMClient._parse_stream_delta`` — and asserts the verdict each envelope
   earned. The envelopes are frozen so the classification is reproducible
   without the endpoint.

2. **Recomputes the rate and its interval** from that corpus with a Wilson
   score interval, and pins the *decision rule* — including the revisit
   threshold — so the number in the doc is one this file recomputes rather than
   a number somebody typed.

It never opens a socket: an autouse fixture replaces ``socket.socket`` and
``socket.create_connection`` with a raiser, so any accidental live draw fails
here instead of quietly spending a shared endpoint's quota. That property is
why this file is separate from ``tests/test_llm_integration.py``.

**The headline result is a null.** 0 of 20 no-evidence questions produced the
tool-calling envelope (95% Wilson interval on the rate: [0.000, 0.161]). The
prior "roughly half" figure was measured *before* ``PROMPT_STABLE_PREFIX``
grew its "你没有任何工具可用…" clause. This harness freezes the post-change
observations so the next person can see the number rather than re-litigate the
error message. **Because the measured corpus contains no refusal**, the
refusal-classification path is exercised with a small **synthetic** tool-call
envelope (:data:`SYNTHETIC_TOOL_CALL`) — a unit test of the classifier, not a
claim about how often the model refuses. The two must not be conflated, and
the synthetic envelope is named differently and lives in a different constant
for exactly that reason.

**Why a Wilson interval and not a binomial proportion.** The measured rate is a
small-sample proportion (n = 20). A naive ``k/n`` understates uncertainty at this
count, and a normal-approximation interval is badly wrong at k = 0. Wilson is
the standard small-sample interval for a proportion and stays inside [0, 1].
Implemented here rather than pulled from ``scipy`` so the committed suite adds
no scientific dependency — this project installs none.

**A real model is not a function of its prompt.** Each recorded draw is one
sampled outcome; the same question drawn again can give a different verdict.
Nothing here asserts a question *must* produce a given verdict — it asserts
that *these specific recorded draws* earned *these specific verdicts*, and that
the interval around the observed rate is what the doc says it is. A future
draw that disagrees is new evidence and belongs in a new corpus, not in a
weakened assertion.
"""

from __future__ import annotations

import math
import socket

import pytest

from paperwhisperer.core.errors import NonRetryableError
from paperwhisperer.llm import output as model_output
from paperwhisperer.llm.client import LLMClient

# ---------------------------------------------------------------------------
# The recorded corpus (real, post-prompt-change)
# ---------------------------------------------------------------------------
#
# One entry per live draw, captured by the driver described in
# ``docs/REFUSAL_DRAWS.md``. Each is the *envelope* the endpoint returned for a
# no-evidence question (the product's own ``build_evidence_answer_prompts``,
# two real evidence passages from the corpus paper, the production temperature),
# plus the verdict the product's classifier reached. Frozen observations, not
# expectations. ``content`` is the leading fragment of ``message.content``.
#
# Every recorded draw came back ``finish_reason="stop"`` with real content: the
# model refused *in prose* ("文档中没有说明…"), correctly and with citations.
# None reached for a tool.
RECORDED_DRAWS: list[dict] = [
    {"q": 1, "finish": "stop", "tc": False, "content": "{   \"answer\": \"结论：在当前可回源的两条证据片段中，没有关于 ImageNet-21k 预训练所用图片数量或训练时长的任", "verdict": "ok"},
    {"q": 3, "finish": "stop", "tc": False, "content": "{   \"answer\": \"依据当前文档证据，无法回答该问题：文档中没有说明 CIFAR-10 上的最终错误率，也没有说明所用的数据", "verdict": "ok"},
    {"q": 3, "finish": "stop", "tc": False, "content": "{   \"answer\": \"依据不足，无法回答。当前文档中检索到的证据只覆盖5.5 Results 中关于 JFT（规模超过 10 ", "verdict": "ok"},
    {"q": 2, "finish": "stop", "tc": False, "content": "{   \"answer\": \"根据当前可回源校验的原文证据，**没有**找到关于 dropout 速率对 JFT 基线影响的消融实验。", "verdict": "ok"},
    {"q": 3, "finish": "stop", "tc": False, "content": "{   \"answer\": \"依据当前文档证据，无法回答 CIFAR-10 的最终错误率与数据增强方式——这两项内容在文档中没有说明。", "verdict": "ok"},
    {"q": 2, "finish": "stop", "tc": False, "content": "{   \"answer\": \"在你提供的这份文档片段中，没有找到关于 dropout 率（dropout rate）影响的消融实验。", "verdict": "ok"},
    {"q": 3, "finish": "stop", "tc": False, "content": "{   \"answer\": \"依据不足，无法回答。检索到的原文证据只包含论文5.5 Results 一节中关于 JFT 的结果与专家模", "verdict": "ok"},
    {"q": 3, "finish": "stop", "tc": False, "content": "```json {   \"answer\": \"结论：依据当前检索到的文档证据，无法回答 CIFAR-10 的最终错误率与所用数据增强—", "verdict": "ok"},
    {"q": 2, "finish": "stop", "tc": False, "content": "{   \"answer\": \"结论：在我检索到的这段文档内容（5.5 Results 的表 3、表 4 及专家模型训练方式的说明）中，", "verdict": "ok"},
    {"q": 0, "finish": "stop", "tc": False, "content": "{   \"answer\": \"结论：依据当前检索到的证据，无法给出作者在 ImageNet 上报告的 top-1 准确率——文档证据中", "verdict": "ok"},
    {"q": 1, "finish": "stop", "tc": False, "content": "{   \"answer\": \"结论：当前文档中没有关于 ImageNet-21k 预训练所用图片数量或训练时长的任何说明，无法据此回答", "verdict": "ok"},
    {"q": 1, "finish": "stop", "tc": False, "content": "{   \"answer\": \"当前文档中没有关于 ImageNet-21k 预训练的图片数量或训练时长的说明，无法回答。\\n\\n依", "verdict": "ok"},
    {"q": 0, "finish": "stop", "tc": False, "content": "{   \"answer\": \"当前文档内容中没有关于 ImageNet top-1 准确率的任何报告数据，因此无法回答该具体数值。检索", "verdict": "ok"},
    {"q": 0, "finish": "stop", "tc": False, "content": "{   \"answer\": \"当前提供的文档片段中没有关于 ImageNet 的任何 top-1 准确率数据，因此无法回答该问题。\\", "verdict": "ok"},
    {"q": 1, "finish": "stop", "tc": False, "content": "{   \"answer\": \"依据当前检索到的文档证据，无法回答这个问题：证据片段中完全没有出现 ImageNet-21k，也没有给出", "verdict": "ok"},
    {"q": 2, "finish": "stop", "tc": False, "content": "{   \"answer\": \"根据当前检索到的证据，文档中**没有**关于 dropout 率对 JFT baseline 影响的消融", "verdict": "ok"},
    {"q": 1, "finish": "stop", "tc": False, "content": "```json {   \"answer\": \"依据当前检索到的文档片段，无法回答这个问题。片段全部来自论文的 5.5 Results ", "verdict": "ok"},
    {"q": 0, "finish": "stop", "tc": False, "content": "{   \"answer\": \"当前证据中没有关于 ImageNet top-1 准确率的数据，因此无法回答具体数值。检索到的两条片段均", "verdict": "ok"},
    {"q": 0, "finish": "stop", "tc": False, "content": "{\"answer\": \"当前提供的证据中没有说明作者在 ImageNet 上报告的 top-1 准确率。\\n\\n两条证据都只涉及 ", "verdict": "ok"},
    {"q": 2, "finish": "stop", "tc": False, "content": "{   \"answer\": \"在当前文档给出的证据范围内，**没有找到关于 dropout 率对 JFT baseline 影响的消融", "verdict": "ok"},
]

#: A **synthetic** tool-calling envelope. NOT part of the measurement — the
#: measured corpus contains no refusal (see above), so this hand-built envelope
#: is what the refusal-classification and both-transport-agree tests exercise.
#: Kept in its own named constant so nobody can mistake it for an observation.
SYNTHETIC_TOOL_CALL = {"q": -1, "finish": "tool_calls", "tc": True, "content": "",
                       "verdict": "refused"}

#: The questions the corpus was drawn against, in the order the ``q`` indices
#: refer to. Every one is a question the retrieved evidence does not answer.
NO_EVIDENCE_QUESTIONS = [
    "作者在 ImageNet 上报告的 top-1 准确率具体是多少？",
    "这篇论文用 ImageNet-21k 预训练时用了多少张图片、训练了多久？",
    "Does this paper report an ablation on the effect of dropout rate on the JFT baseline?",
    "论文在 CIFAR-10 上报告的最终错误率是多少？用的是哪种数据增强？",
]

#: The number at which the "leave it alone" decision is revisited, stated in
#: ``docs/REFUSAL_DRAWS.md``. Sensitive to the Wilson **lower bound**, not the
#: raw proportion, so the trigger does not fire on a lucky small sample: a run
#: whose point rate is high but whose interval still reaches down near zero is
#: not yet evidence of a regression.
REVISIT_LOWER_BOUND = 0.10


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Prove the harness is offline: any socket use fails the test."""

    def _blocked(*_args, **_kwargs):
        raise AssertionError(
            "the refusal-draw harness must be reproducible offline; "
            "it opened a socket (a live draw belongs in the driver, not here)"
        )

    monkeypatch.setattr(socket, "socket", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)


# ---------------------------------------------------------------------------
# Interval maths
# ---------------------------------------------------------------------------


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. Returns ``(lo, hi)``.

    Chosen over the normal approximation because at n = 20 with k = 0 the
    normal interval is badly wrong (it can even cross below 0), and over
    ``k/n`` because a point estimate is not an interval at all.

    ``total == 0`` yields ``(0.0, 1.0)``: no observation bounds nothing, and
    claiming precision from zero draws is the exact mistake this file exists to
    prevent.
    """
    if total <= 0:
        return (0.0, 1.0)
    p = successes / total
    denom = 1.0 + (z * z) / total
    centre = (p + (z * z) / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + (z * z) / (4 * total * total)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


# ---------------------------------------------------------------------------
# Counting helpers (module-level so every test class shares one definition)
# ---------------------------------------------------------------------------


def recorded_counts() -> tuple[int, int]:
    """``(refusals, total)`` over the recorded (real) corpus.

    Module-level rather than a method on one test class: several classes assert
    on the same two numbers, and a second, separately-typed copy of ``k`` and
    ``n`` is exactly how the doc and the test would drift apart.
    """
    total = len(RECORDED_DRAWS)
    refusals = sum(1 for d in RECORDED_DRAWS if d["verdict"] == "refused")
    return refusals, total


# ---------------------------------------------------------------------------
# Replaying envelopes through the product's own classification
# ---------------------------------------------------------------------------


class _RecordedChoice:
    def __init__(self, draw: dict):
        self.finish_reason = draw["finish"]
        self.message = _RecordedMessage(draw)


class _RecordedMessage:
    def __init__(self, draw: dict):
        self.content = draw["content"]
        self.tool_calls = ["stub-tool-call"] if draw["tc"] else None


class _RecordedResponse:
    def __init__(self, draw: dict):
        self.choices = [_RecordedChoice(draw)]


def classify_recorded(draw: dict):
    """Run one envelope through ``classify_completion``.

    This is the exact call :meth:`LLMClient._extract_completion_text` makes,
    with the same two signals it passes: the ``finish_reason`` and the presence
    of ``message.tool_calls``.
    """
    response = _RecordedResponse(draw)
    message = response.choices[0].message
    called_a_tool = (
        str(response.choices[0].finish_reason or "") == "tool_calls" or bool(message.tool_calls)
    )
    return model_output.classify_completion(
        message.content,
        called_a_tool=called_a_tool,
        finish_reason=response.choices[0].finish_reason,
        label="AI 回复",
    )


class TestRecordedCorpusReplaysToItsRecordedVerdict:
    """Every frozen draw earns, offline, the verdict it earned live.

    This is the reproduction guarantee: the numbers in the doc are recomputable
    from this file with no endpoint, and the *classification* of each envelope
    is re-derived by the product's own code rather than restated as a constant
    the test compares a constant against.
    """

    def test_the_corpus_is_non_trivial(self):
        assert len(RECORDED_DRAWS) >= 20, (
            "the decision needs at least 20 draws to carry an interval; "
            f"the corpus has {len(RECORDED_DRAWS)}"
        )

    def test_every_draw_spans_a_known_question(self):
        assert all(0 <= d["q"] < len(NO_EVIDENCE_QUESTIONS) for d in RECORDED_DRAWS)

    def test_each_recorded_draw_replays_to_its_recorded_verdict(self):
        for index, draw in enumerate(RECORDED_DRAWS):
            verdict = classify_recorded(draw)
            assert verdict.status.value == draw["verdict"], (
                f"draw #{index} (question {draw['q']}, finish={draw['finish']!r}) "
                f"replayed to {verdict.status.value!r} but was recorded as "
                f"{draw['verdict']!r}"
            )

    def test_no_recorded_draw_reached_for_a_tool(self):
        """The measured fact: 0/20 produced the tool-calling envelope."""
        for draw in RECORDED_DRAWS:
            assert draw["finish"] == "stop" and not draw["tc"]
            assert classify_recorded(draw).status is model_output.OutputStatus.OK

    def test_the_synthetic_tool_call_replays_to_refused(self):
        """The classifier still recognises a refusal (unit test, not a claim)."""
        assert (
            classify_recorded(SYNTHETIC_TOOL_CALL).status
            is model_output.OutputStatus.REFUSED
        )


class TestBothTransportsAgreeOnARefusal:
    """The two call shapes must reach the same verdict on the same envelope.

    The streaming path was, for a while, blind to ``finish_reason``: it read
    only ``delta.content``, so a tool-calling reply arrived there as an *empty
    completion* and got retried, while the non-streaming path called it a
    refusal and refused it. Same event, two verdicts. These assertions run the
    synthetic refusal through both transports so the disagreement cannot return.
    """

    def test_non_streaming_transport_raises_non_retryable_on_a_refusal(self):
        with pytest.raises(NonRetryableError):
            LLMClient._extract_completion_text(_RecordedResponse(SYNTHETIC_TOOL_CALL))

    def test_streaming_transport_raises_non_retryable_on_the_same_envelope(self):
        # Shaped from the synthetic refusal's own signals: a delta carrying
        # tool_calls and a finish_reason of "tool_calls".
        payload = {"choices": [{"delta": {"tool_calls": [{"id": "1"}]},
                                "finish_reason": "tool_calls"}]}
        with pytest.raises(NonRetryableError):
            LLMClient._parse_stream_delta(payload)

    def test_an_ordinary_recorded_reply_is_not_mistaken_for_a_refusal(self):
        ok = RECORDED_DRAWS[0]
        assert classify_recorded(ok).status is model_output.OutputStatus.OK


class TestRateAndInterval:
    """The rate the doc quotes, recomputed from the corpus.

    These are the assertions that make ``docs/REFUSAL_DRAWS.md`` a *record*
    rather than a claim: if somebody edits the doc's number without editing the
    corpus, or vice versa, one of these fails.
    """

    def test_point_rate_matches_the_recorded_draws(self):
        refusals, total = recorded_counts()
        # 0 refusals in 20 draws: see docs/REFUSAL_DRAWS.md.
        assert refusals == 0, f"corpus now has {refusals} refusals; doc says 0"
        assert total == 20, f"corpus now has {total} draws; doc says 20"

    def test_wilson_interval_is_a_valid_interval(self):
        refusals, total = recorded_counts()
        low, high = wilson_interval(refusals, total)
        point = refusals / total
        assert 0.0 <= low <= point <= high <= 1.0

    def test_zero_draws_gives_an_unbounded_interval(self):
        """The guard against claiming precision from no evidence."""
        assert wilson_interval(0, 0) == (0.0, 1.0)

    def test_the_interval_matches_the_doc(self):
        # doc: 0/20 = 0.000, Wilson 95% [0.000, 0.161]
        refusals, total = recorded_counts()
        low, high = wilson_interval(refusals, total)
        assert low == pytest.approx(0.000, abs=1e-6)
        assert high == pytest.approx(0.161, abs=0.002), f"upper bound drifted to {high:.3f}"

    def test_per_question_refusal_counts_are_recorded(self):
        """No question produced a refusal, so per-question counts are all zero."""
        per_q = {q: 0 for q in range(len(NO_EVIDENCE_QUESTIONS))}
        for draw in RECORDED_DRAWS:
            if draw["verdict"] == "refused":
                per_q[draw["q"]] += 1
        assert per_q == {0: 0, 1: 0, 2: 0, 3: 0}


class TestDecisionRule:
    """The revisit threshold, pinned against the interval.

    The decision was to leave the product alone at the measured rate. That is
    only defensible if the rule that *would* overturn it is written down and is
    sensitive to the right quantity: the Wilson **lower bound**, so the trigger
    cannot fire on a lucky small sample.
    """

    def test_measured_lower_bound_is_below_the_revisit_threshold(self):
        refusals, total = recorded_counts()
        low, _ = wilson_interval(refusals, total)
        assert low < REVISIT_LOWER_BOUND, (
            f"measured Wilson lower bound {low:.3f} has reached the revisit "
            f"threshold {REVISIT_LOWER_BOUND}; the decision in "
            "docs/REFUSAL_DRAWS.md must be revisited"
        )

    def test_the_threshold_is_not_trivially_met(self):
        """A threshold above 1.0 or at/below 0 would make the guard vacuous."""
        assert 0.0 < REVISIT_LOWER_BOUND < 1.0

    def test_a_would_regressed_corpus_would_trip_the_guard(self):
        """The guard actually fires on a plausible regression.

        A corpus matching the old "roughly half" behaviour (10 refusals in 20)
        has a Wilson lower bound well above the threshold, so the decision rule
        is not vacuously satisfied.
        """
        low, _ = wilson_interval(10, 20)
        assert low >= REVISIT_LOWER_BOUND