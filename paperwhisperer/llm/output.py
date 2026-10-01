"""One parsing layer for model output, and the taxonomy of how it can fail.

Every structured call in this codebase parses its own answer: the answer
payload in :mod:`paperwhisperer.analysis.orchestrator`, the eight fact fields
in :mod:`paperwhisperer.analysis.paper_facts`, the search rewrite in
:mod:`paperwhisperer.analysis.sections`, and the Mermaid preamble in
``DocumentAnalyzer._normalize_mermaid_source``. Each of those is a private pile
of string handling, exercised by whatever the model happens to return on a bad
day, and each of them fails differently.

**The failure this module exists to prevent.** A model that wraps its JSON in a
fence, uses a full-width colon, or explains itself before answering will fail a
naive parser and produce an *empty* section. The user sees a card with nothing
on it and no way to tell the difference between "the paper does not say" and
"we could not read what the model said". The second is a defect and the first
is an answer, and only one of the two is honest.

So the rule this module holds to is one sentence: **an unparseable answer is a
reported degradation, never a silent empty result and never an exception that
surfaces as a 500.** Every entry point is total. It returns an
:class:`OutputResult` for any input at all -- ``None``, prose, a ten-megabyte
string of braces, a document nested two thousand deep -- and a failure names
what was expected together with a bounded, redacted excerpt of what arrived.

**Why recoveries are reported rather than silently applied.** A value pulled out
of a mangled string is a value whose provenance is now unclear. ``{"a": 1}``
inside a fence is not in doubt. ``{"a"：1}`` recovered by rewriting a full-width
colon is in doubt, because the bytes the model sent are not the bytes that were
parsed, and something downstream may depend on the difference.
:class:`OutputResult` therefore carries a status in which
:attr:`OutputStatus.OK` and :attr:`OutputStatus.RECOVERED` are *different
verdicts*, the tuple of :class:`Recovery` steps that fired, and
:meth:`OutputResult.user_message` in Chinese for a person to read.
:mod:`paperwhisperer.analysis.paper_facts` already has this shape in
:class:`~paperwhisperer.analysis.paper_facts.ExtractionReport`; this is the
same discipline one layer lower, and it is the reason a run record can answer
"was this answer what the model said?".

**The taxonomy is the deliverable.** A caller must be able to tell "the model
said nothing useful" (:attr:`OutputStatus.EMPTY`) from "the model said
something we could not read" (:attr:`OutputStatus.NOT_JSON`), because those
want different words on screen, and from "the model declined"
(:attr:`OutputStatus.REFUSED`), which is a different event and is not
retryable. See :func:`classify_completion` and the note at
:attr:`OutputStatus.REFUSED`.

**What a recovery is allowed to touch.** Punctuation is rewritten *outside
string literals only*. ``{"why": "他说：“这是关键”"}`` must survive this module
byte for byte: those curly quotes are the paper's characters, not the model's
syntax. The scan is string-aware for exactly this reason, and a quote counts as
a delimiter only in value position -- after a ``:``, a ``[``, a ``,`` or a
``{`` -- so a preamble that quotes a paper title cannot be mistaken for a
string opening and swallow the payload that follows it.

**Bounds are part of the contract, not an optimisation.** ``json.loads`` on
input nobody bounded is a denial-of-service vector: a caller will happily build
a structure for every byte of a reply. :data:`MAX_INPUT_CHARS` refuses before
any scanning at all, :data:`MAX_JSON_DEPTH` refuses during the single bounded
pass, and that pass is O(n) with no regular expression anywhere on the input --
:mod:`re` appears only over the first few lines of a reply, looking for a code
fence, and over the excerpt that is about to be written to a log.

**Determinism is a requirement, not a nicety.** No clock, no randomness, no
network, no environment. The same bytes produce the same result, which is what
makes any of it testable at all.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from paperwhisperer.core.text import redact_secrets

logger = logging.getLogger(__name__)

__all__ = [
    "EXPECT_ANY",
    "EXPECT_ARRAY",
    "EXPECT_OBJECT",
    "EXCERPT_CHARS",
    "MAX_CANDIDATE_SPANS",
    "MAX_INPUT_CHARS",
    "MAX_JSON_DEPTH",
    "OutputResult",
    "OutputStatus",
    "Recovery",
    "bounded_excerpt",
    "classify_completion",
    "parse_json",
    "parse_json_array",
    "parse_json_object",
]


# --------------------------------------------------------------------------
# Bounds
# --------------------------------------------------------------------------
#
# Every one of these is a *rejection*, not a truncation. Silently cutting a
# reply at a character boundary produces JSON that parses into a value missing
# its tail, which is the exact defect this module is for: a section that
# renders as present-but-empty with no signal that anything was lost.

#: Largest reply this module will look at. A structured answer for the biggest
#: thing the product asks a model for -- eight fact fields with quotations -- is
#: a few kilobytes, so this is orders of magnitude of headroom, and the check
#: is ``len()``: it costs nothing and it refuses before any scanning.
#:
#: A refusal threshold, not a budget. A reply this size from a completion
#: endpoint is a malfunction, and a malfunction is a thing an operator wants
#: reported rather than half-parsed.
MAX_INPUT_CHARS = 262_144

#: Deepest structure accepted, measured during the scan rather than left to
#: ``json.loads``, which recurses and raises ``RecursionError`` on a payload
#: deep enough to matter -- an exception, which is exactly what this module
#: promises not to produce. Anything this product asks for nests three or four
#: deep; the ceiling exists so that ``[`` repeated a quarter of a million times
#: is a reported refusal.
MAX_JSON_DEPTH = 32

#: How many top-level values the scan will consider as a source of the payload.
#: Bounded because the reply is the untrusted input: a model that prefixes a
#: thousand of them should cost a fixed scan, not a proportional one. Eight is
#: comfortably past "a preamble, the payload, and an explanation".
MAX_CANDIDATE_SPANS = 8

#: Lines of preamble searched for an opening code fence. A model that explains
#: itself before a fence does so in a sentence or two; scanning a quarter of a
#: megabyte for one is work with no possible payoff.
FENCE_SEARCH_LINES = 8

#: Length of the excerpt attached to a failure. Long enough to identify a
#: failure from a log line, short enough that attaching one to every degraded
#: run cannot itself become the thing that fills a disk.
EXCERPT_CHARS = 200


# --------------------------------------------------------------------------
# The taxonomy
# --------------------------------------------------------------------------


class OutputStatus(str, Enum):
    """The ways model output can be unusable, as a closed set.

    Closed on purpose. ``if status == ...`` chains spread across a codebase
    over time, and an open set means each branch is decided per call site, per
    author, and inconsistently. A new way for a reply to be unusable is a new
    member here with a Chinese message attached, once.
    """

    #: Parsed as written. No recovery fired and the shape is what was asked for.
    OK = "ok"

    #: Parsed, but only after one or more repairs to the bytes. The value is
    #: usable and its provenance is not: what came out is not what was sent.
    RECOVERED = "recovered"

    #: Something arrived and it could not be read as JSON at all. Distinct from
    #: :attr:`EMPTY` -- the model *said something*, we could not read it.
    NOT_JSON = "not_json"

    #: Valid JSON of the wrong kind: an array where an object was asked for, or
    #: a scalar where a container was. The value is attached, because a caller
    #: may well know what to do with it; the status says the contract broke.
    WRONG_SHAPE = "wrong_shape"

    #: Nothing usable arrived: empty, whitespace, or a fence around nothing.
    #: This is "the model produced no output", not "we failed to read it".
    EMPTY = "empty"

    #: Beyond a declared bound, refused before parsing. :attr:`OutputResult.detail`
    #: names which bound, because "too big" and "too deep" are different bugs.
    OVERSIZED = "oversized"

    #: The model declined, or answered by calling a tool this product does not
    #: offer. Measured against this product: a reasoning model asked a question
    #: the document cannot answer replies with HTTP 200,
    #: ``finish_reason="tool_calls"`` and an empty ``content``, having decided
    #: mid-thought to go looking for the document with a grep tool it was never
    #: given.
    #: Not retryable -- an identical prompt produces an identical tool call --
    #: and not a parse failure either, which is why it is its own status rather
    #: than a flavour of :attr:`EMPTY`.
    REFUSED = "refused"


class Recovery(str, Enum):
    """The repairs applied to a reply, each one visible in the result.

    Every member means the bytes that were parsed differ from the bytes that
    arrived. The tuple is therefore a list of things to be suspicious of, and a
    caller whose rendered section looks wrong can check here first.
    """

    #: A ```-fenced block was unwrapped.
    FENCE = "fence"

    #: Prose before the payload was dropped. The payload is found by scanning
    #: for the first balanced value rather than by assuming the reply starts
    #: with one.
    PREAMBLE = "preamble"

    #: Prose after the payload was dropped.
    TRAILING = "trailing"

    #: Full-width punctuation outside string literals was folded to ASCII --
    #: ``：``, ``，``, ``｛``, ``｝``, ``＂`` and their neighbours.
    FULLWIDTH = "fullwidth"

    #: A smart quote in value position was read as a string delimiter. Curly
    #: quotes *inside* a string are content and are never touched.
    SMART_QUOTES = "smart_quotes"

    #: A single-quoted pseudo-object was requoted as a JSON string.
    SINGLE_QUOTES = "single_quotes"

    #: An unterminated string or an unclosed bracket was closed. The usual
    #: cause is the token budget running out mid-answer, so this is the
    #: recovery most likely to have invented an empty tail -- and the reason it
    #: is reported at all.
    CLOSED = "closed"


#: What a caller asks for. ``EXPECT_ANY`` accepts any JSON value, which is what
#: a transport-level frame wants; the other two are contracts, and breaking one
#: is :attr:`OutputStatus.WRONG_SHAPE` rather than a failure.
EXPECT_ANY = "any"
EXPECT_OBJECT = "object"
EXPECT_ARRAY = "array"

_EXPECTED_LABELS = {
    EXPECT_ANY: "JSON 值",
    EXPECT_OBJECT: "JSON 对象",
    EXPECT_ARRAY: "JSON 数组",
}

_KIND_LABELS = {EXPECT_OBJECT: "对象", EXPECT_ARRAY: "数组", "scalar": "标量"}

_STATUS_MESSAGES = {
    OutputStatus.NOT_JSON: "模型返回的内容不是约定的 JSON 格式，无法解析。",
    OutputStatus.EMPTY: "模型没有返回任何可用内容。",
}

_RECOVERY_LABELS = {
    Recovery.FENCE: "代码块围栏",
    Recovery.PREAMBLE: "前置说明文字",
    Recovery.TRAILING: "尾部多余文字",
    Recovery.FULLWIDTH: "全角标点",
    Recovery.SMART_QUOTES: "中文引号",
    Recovery.SINGLE_QUOTES: "单引号",
    Recovery.CLOSED: "未闭合的括号或字符串",
}

#: The refusal :mod:`paperwhisperer.llm.client` already raises for a tool call.
#: Defined here so that the transport and this module cannot disagree about
#: what the user is told: the layer that classifies the event owns its message.
#: The wording is unchanged from the transport's own, so no user-visible string
#: moves when a call site is migrated onto the layer.
TOOL_CALL_MESSAGE = (
    "模型尝试调用本产品不提供的工具（未提供内容回复）。"
    "请重试，或在提示中明确要求直接作答、不要使用工具。"
)


@dataclass(frozen=True, slots=True)
class OutputResult:
    """One parsed reply, and everything a caller needs to judge it.

    ``value`` is the parsed JSON, or ``None`` for anything that is not
    :attr:`OutputStatus.OK` or :attr:`OutputStatus.RECOVERED`. ``None`` rather
    : than a default in the failure cases, deliberately: a caller that ignores
    :attr:`status` gets an unmistakable ``None`` instead of an empty container
    that would render as a present-but-empty section, which is the defect.

    ``recoveries`` carries the weight. A result whose recoveries are empty is a
    value the model actually wrote; one whose are not is a value this module
    assembled, and :meth:`user_message` says so in words a reader will see.
    """

    status: OutputStatus
    value: Any = None
    kind: str = ""
    expected: str = EXPECT_ANY
    recoveries: tuple[Recovery, ...] = ()
    excerpt: str = ""
    detail: str = ""
    label: str = ""

    @property
    def ok(self) -> bool:
        """Whether a value came back at all.

        ``True`` for :attr:`OutputStatus.RECOVERED` too: the caller asked whether
        it has a value, and it does. Whether to *trust* it is ``status is
        OutputStatus.OK``, which is a different question with a different
        answer.
        """
        return self.status in (OutputStatus.OK, OutputStatus.RECOVERED)

    @property
    def degraded(self) -> bool:
        """Whether anything at all about this reply was repaired or refused."""
        return self.status is not OutputStatus.OK

    @property
    def is_recovered(self) -> bool:
        """Whether the value came from bytes this module rewrote."""
        return self.status is OutputStatus.RECOVERED

    def degradations(self) -> tuple[str, ...]:
        """Chinese, user-visible, one per thing that went wrong.

        The same shape as
        :meth:`paperwhisperer.analysis.paper_facts.ExtractionReport.degradations`,
        so a caller already reporting degradations for a run has one vocabulary
        to report them in rather than two.
        """
        if self.status is OutputStatus.OK:
            return ()
        if not self.ok:
            return (self.user_message(),)
        applied = "、".join(
            _RECOVERY_LABELS.get(item, item.value) for item in self.recoveries
        )
        return (
            self.user_message(),
            f"解析时已自动修复：{applied}。修复后的结果可能与模型原文不一致。",
        )

    def user_message(self) -> str:
        """One Chinese sentence for a person, or ``""`` when nothing is wrong.

        Empty on :attr:`OutputStatus.OK`, rather than a success line: a
        degradation list containing "nothing went wrong" is a line a reader
        learns to skip, and the absence is the signal.
        """
        if self.status is OutputStatus.OK:
            return ""
        if self.status is OutputStatus.REFUSED:
            return TOOL_CALL_MESSAGE
        message = ""
        if self.status is OutputStatus.RECOVERED:
            # Unreachable through ``parse_json``, which never reports RECOVERED
            # with an empty recovery tuple -- but a hand-built result must not be
            # able to produce a *silent* recovery, because "recovered" with
            # nothing to report is the one degraded state a reader cannot act on.
            applied = "、".join(
                _RECOVERY_LABELS.get(item, item.value) for item in self.recoveries
            )
            message = f"模型返回的内容经过了修正才可解析（{applied or '未知修正'}）。"
        elif self.status is OutputStatus.OVERSIZED:
            if self.detail.startswith("depth"):
                # The ceiling is carried in ``detail`` rather than read from the
                # module global, so a caller that lowered the bound for one call
                # is told the bound that actually applied. A message quoting
                # MAX_JSON_DEPTH after the caller asked for depth 5 would send
                # an operator to raise a limit that was never the problem.
                ceiling = self.detail.split("=", 1)[-1] or str(MAX_JSON_DEPTH)
                message = f"模型返回的 JSON 嵌套层级超过 {ceiling} 层，已放弃解析。"
            else:
                message = (
                    f"模型返回的内容超过 {MAX_INPUT_CHARS} 字符上限，"
                    "已放弃解析以避免占用过多资源。"
                )
        elif self.status is OutputStatus.WRONG_SHAPE:
            wanted = _EXPECTED_LABELS.get(self.expected, self.expected)
            found = _KIND_LABELS.get(self.kind, "标量")
            message = f"模型返回的是 JSON {found}，约定要求的是 {wanted}。"
        else:
            message = _STATUS_MESSAGES.get(self.status, "")
        return f"{message}（{self.label}）" if self.label else message

    def to_dict(self) -> dict[str, Any]:
        """A flat, serialisable record for a run record or a log line.

        ``value`` is present on success and absent otherwise, which makes the
        dict safe to attach to an analysis run: a failed parse cannot smuggle a
        partial structure into whatever persists it.
        """
        payload: dict[str, Any] = {
            "status": self.status.value,
            "kind": self.kind,
            "expected": self.expected,
            "recoveries": [item.value for item in self.recoveries],
            "detail": self.detail,
            "label": self.label,
            "degradations": list(self.degradations()),
        }
        if self.ok:
            payload["value"] = self.value
        return payload

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"OutputResult(status={self.status.value}, kind={self.kind!r}, "
            f"recoveries={[item.value for item in self.recoveries]!r}, "
            f"detail={self.detail!r})"
        )


# --------------------------------------------------------------------------
# The scan
# --------------------------------------------------------------------------
#
# One pass, string-aware, linear, with no regular expression on the input. It
# produces the normalised text and the candidate spans together because doing
# them separately would mean two passes over an untrusted reply and two chances
# to disagree about where a string begins.

_FENCE_OPEN = re.compile(r"^```[A-Za-z0-9_+.-]*[ \t]*$")

#: Punctuation that is syntax only when it is outside a string, folded to the
#: ASCII the JSON grammar actually specifies. ``＂`` is here because a
#: full-width quote *is* a delimiter, unlike most of its neighbours in the CJK
#: block; ``、`` is here for the same reason as ``，``, since outside a string
#: there is nowhere else for either to mean anything.
_STRUCTURAL_FOLD = {
    "＂": '"',
    "｛": "{",
    "｝": "}",
    "［": "[",
    "］": "]",
    "：": ":",
    "，": ",",
    "、": ",",
}

#: Which character closes a string opened by each possible opener.
#:
#: *The opener decides*, and that is the whole design. A string opened by ``"``
#: is closed by ``"``, so curly quotes inside it belong to the payload and are
#: preserved verbatim -- which is what makes ``{"why": "他说“这是关键”"}`` come
#: through unchanged. A string opened by ``“`` is closed by ``”``, which is how
#: a model that used curly quotes as delimiters gets read at all.
_QUOTE_TERMINATORS = {'"': '"', "＂": "＂", "“": "”", "”": "“", "'": "'"}

#: Characters after which a quote is in value position, and therefore a
#: delimiter rather than prose. Without this, a preamble quoting a paper title
#: opens a string that swallows the payload.
_VALUE_POSITION = frozenset(":[,{ ")

_OPENERS = "{["
_CLOSERS = "}]"


@dataclass(frozen=True, slots=True)
class _Span:
    """One top-level ``{...}`` or ``[...]`` located during the scan.

    ``stack`` holds the openers still unclosed when the input ran out, in
    order, so a truncated value can be closed without re-reading the text.
    """

    start: int
    end: int
    stack: tuple[str, ...]
    closed: bool


@dataclass(frozen=True, slots=True)
class _Scan:
    text: str
    spans: tuple[_Span, ...]
    too_deep: bool
    recoveries: tuple[Recovery, ...]


def _scan(text: str, max_depth: int = MAX_JSON_DEPTH) -> _Scan:
    """Normalise structural punctuation and locate payloads, in one pass.

    Two jobs, because they are the same job: the pass has to know where every
    string literal begins in order to fold punctuation outside them, and a
    scanner that knows that is already tracking the bracket depth that finds the
    payload spans. Splitting them would mean walking the reply twice.
    """
    out: list[str] = []
    spans: list[_Span] = []
    recoveries: list[Recovery] = []
    stack: list[str] = []
    #: The last non-whitespace character emitted, which is what "value
    #: position" is decided against. Tracked rather than searched backwards
    #: through ``out``: a backward search makes a reply of repeated smart
    #: quotes quadratic, which is a denial-of-service vector wearing a
    #: recovery's clothes.
    last_significant = ""
    emitted = 0
    index = 0
    length = len(text)

    def note(recovery: Recovery) -> None:
        if recovery not in recoveries:
            recoveries.append(recovery)

    while index < length:
        char = _STRUCTURAL_FOLD.get(text[index], text[index])
        if char != text[index]:
            note(Recovery.FULLWIDTH)

        if char == '"' or char in "“”":
            if char != '"' and last_significant not in _VALUE_POSITION:
                out.append(char)
                emitted += 1
                last_significant = char
                index += 1
                continue
            if char != '"':
                note(Recovery.SMART_QUOTES)
            end, unterminated = _scan_string(text, index, _QUOTE_TERMINATORS[char])
            body = text[index + 1 : end]
            # The curly-quoted branches requote into a real JSON string, so a
            # stray straight quote inside one has to be escaped; the straight
            # branch is passed through verbatim, escapes and all, because
            # rewriting them would corrupt the payload.
            payload = body if char == '"' else body.replace('"', '\\"')
            out.append('"' + payload + '"')
            # ``end - index`` counts the opener and the body; the closing quote
            # just appended is the ``+ 1``. Getting this wrong shifts every
            # later offset by one, which shows up as a span that ends one
            # character short and a payload that will not parse.
            emitted += end - index + 1
            index = end + 1
            if unterminated:
                note(Recovery.CLOSED)
            last_significant = '"'
            continue

        if char == "'" and last_significant in _VALUE_POSITION:
            note(Recovery.SINGLE_QUOTES)
            end, _unterminated = _scan_string(text, index, "'")
            out.append('"' + text[index + 1 : end].replace('"', '\\"') + '"')
            emitted += end - index + 1
            index = end + 1
            last_significant = '"'
            continue

        if char in _OPENERS:
            if not stack:
                spans.append(_Span(emitted, -1, (char,), False))
            stack.append(char)
            if len(stack) > max_depth:
                return _Scan(text="".join(out), spans=(), too_deep=True, recoveries=())
            out.append(char)
            emitted += 1
            last_significant = char
            index += 1
            continue

        if char in _CLOSERS:
            if stack:
                stack.pop()
                if not stack and spans and not spans[-1].closed:
                    spans[-1] = _Span(spans[-1].start, emitted + 1, (), True)
            out.append(char)
            emitted += 1
            last_significant = char
            index += 1
            continue

        out.append(char)
        emitted += 1
        if not char.isspace():
            last_significant = char
        index += 1

    if stack:
        # Ran out mid-value. Close the last span with whatever is still open so
        # the closing characters can be appended without a second walk.
        spans[-1] = _Span(spans[-1].start, emitted, tuple(stack), False)
        note(Recovery.CLOSED)
    return _Scan(
        text="".join(out),
        spans=tuple(spans),
        too_deep=False,
        recoveries=tuple(recoveries),
    )


def _scan_string(text: str, start: int, terminator: str) -> tuple[int, bool]:
    """Return ``(end, unterminated)`` for the string opening at ``start``.

    ``end`` points at the terminator, or at ``len(text)`` when the string never
    closed. A backslash escapes whatever follows it, terminator included, which
    is what keeps ``"a\\": 1`` from ending early.
    """
    index = start + 1
    length = len(text)
    while index < length:
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == terminator:
            return index, False
        index += 1
    return length, True


def _closers(stack: Sequence[str]) -> str:
    """The closing characters a still-open stack of openers needs."""
    return "".join("}" if opener == "{" else "]" for opener in reversed(stack))


def _strip_fence(text: str) -> tuple[str, tuple[Recovery, ...]]:
    """Unwrap a ```-fenced reply, reporting whether there was one.

    Only the first :data:`FENCE_SEARCH_LINES` lines are examined for the
    opening fence, and only the tail for the closing one. A fence is a
    convention on the first line or two of a reply; the input is not ours and
    the scan is already linear once.
    """
    lines = text.split("\n")
    head = 0
    while head < len(lines) and not lines[head].strip():
        head += 1
    if head >= len(lines) or not _FENCE_OPEN.match(lines[head]):
        return text, ()
    body = lines[head + 1 :]
    tail = len(body)
    while tail > 0 and not body[tail - 1].strip():
        tail -= 1
    if tail > 0 and body[tail - 1].strip() in ("```", "~~~"):
        body = body[: tail - 1]
    return "\n".join(body), (Recovery.FENCE,)


# --------------------------------------------------------------------------
# The entry points
# --------------------------------------------------------------------------


def bounded_excerpt(text: Any, limit: int = EXCERPT_CHARS) -> str:
    """A short, single-line, redacted rendering of what actually arrived.

    Redacted because it goes into log lines and into analysis run records, and
    a model that echoes a prompt back can echo a credential with it.
    :func:`paperwhisperer.core.text.redact_secrets` is the one implementation of
    that in this project, so an excerpt cannot fall out of step with the log
    filter scrubbing the same strings on the way out.
    """
    flat = re.sub(r"\s+", " ", str(text or ""))[: limit * 8]
    flat = redact_secrets(flat).strip()
    if len(flat) <= limit:
        return flat
    return flat[:limit] + "…"


def _shape_of(value: Any) -> str:
    if isinstance(value, dict):
        return EXPECT_OBJECT
    if isinstance(value, list):
        return EXPECT_ARRAY
    return "scalar"


def _result(
    status: OutputStatus,
    raw: Any,
    expected: str,
    label: str,
    detail: str,
    recoveries: tuple[Recovery, ...] = (),
    value: Any = None,
    kind: str = "",
) -> OutputResult:
    return OutputResult(
        status=status,
        value=value,
        kind=kind,
        expected=expected,
        # Deduped, order-preserving. The same repair can be reached by two
        # routes -- the scan notes ``closed`` while scanning, and the span
        # branch notes it again when it appends the closers -- and a list that
        # says "未闭合的括号或字符串" twice tells a reader nothing a single
        # mention did not.
        recoveries=tuple(dict.fromkeys(recoveries)),
        excerpt=bounded_excerpt(raw),
        detail=detail,
        label=label,
    )


def parse_json(
    text: Any,
    *,
    expected: str = EXPECT_ANY,
    label: str = "",
    recover: bool = True,
    max_chars: int = MAX_INPUT_CHARS,
    max_depth: int = MAX_JSON_DEPTH,
) -> OutputResult:
    """Read one model reply as JSON, and say what it took to read it.

    Total by construction: for any input whatsoever -- ``None``, prose, a
    ten-megabyte string of braces, a document nested past the depth ceiling --
    this returns an :class:`OutputResult` and raises nothing. A caller therefore
    cannot turn a bad model reply into a 500 by forgetting to handle it, and a
    caller that *does* handle it has a named outcome to branch on rather than an
    ``except json.JSONDecodeError``.

    ``expected`` is a contract, not a filter. Asking for :data:`EXPECT_OBJECT`
    and getting an array yields :attr:`OutputStatus.WRONG_SHAPE` with the array
    attached -- not an exception, and not a silently empty mapping, which is the
    shape a naive ``parsed if isinstance(parsed, dict) else {}`` produces and the
    reason an unreadable answer and a short answer look the same.

    ``recover=False`` turns every repair off, for input that did not come from a
    model: a proxy's SSE frame is not prose-plus-JSON, and repairing it would
    turn a corrupt transport frame into a plausible one.
    """
    raw = "" if text is None else str(text)
    wanted = str(expected or EXPECT_ANY)
    name = str(label or "")

    if not raw.strip():
        return _result(OutputStatus.EMPTY, raw, wanted, name, "blank")
    if len(raw) > max_chars:
        return _result(OutputStatus.OVERSIZED, raw, wanted, name, "chars")

    if not recover:
        return _parse_attempts([(raw.strip(), ())], raw, wanted, name)

    body, fence = _strip_fence(raw)
    scan = _scan(body, max_depth)
    base = fence + tuple(scan.recoveries)

    attempts: list[tuple[str, tuple[Recovery, ...]]] = [(raw.strip(), ())]
    if body.strip() and body.strip() != raw.strip():
        attempts.append((body.strip(), fence))
    if scan.too_deep:
        # A reply this deep is refused rather than repaired: the depth ceiling
        # exists to stop unbounded recursion, and closing a thousand brackets
        # would produce a value we have no business claiming to have read.
        return _result(
            OutputStatus.OVERSIZED, raw, wanted, name, f"depth={max_depth}", base
        )
    attempts.append((scan.text.strip(), base))
    for span in scan.spans[:MAX_CANDIDATE_SPANS]:
        recoveries = list(base)
        if span.start > 0:
            recoveries.append(Recovery.PREAMBLE)
        if span.closed:
            if span.end < len(scan.text):
                recoveries.append(Recovery.TRAILING)
            attempts.append((scan.text[span.start : span.end], tuple(recoveries)))
            continue
        # A value the token budget cut off mid-structure. The closers the scan
        # recorded are appended so that the reading is offered at all -- and the
        # CLOSED recovery rides along with it, because a value closed by this
        # module rather than by the model is the one in this set whose
        # provenance is least trustworthy.
        recoveries.append(Recovery.CLOSED)
        attempts.append(
            (scan.text[span.start : span.end] + _closers(span.stack), tuple(recoveries))
        )

    return _parse_attempts(attempts, raw, wanted, name, base)


def _parse_attempts(
    attempts: Sequence[tuple[str, tuple[Recovery, ...]]],
    raw: str,
    wanted: str,
    name: str,
    attempted: tuple[Recovery, ...] = (),
) -> OutputResult:
    """Take the first reading that parses, honouring the shape contract.

    The whole reply is attempted before any substring of it, and the outermost
    span before a nested one. That ordering is the design rather than an
    implementation detail: a preamble quoting a JSON-looking fragment must not
    be able to outrank the payload that follows it, and "the first balanced
    value that fits" is the reading that gets that right far more often than
    "the first ``{`` in the text".

    ``attempted`` is the set of repairs the scan made on the way to failing.
    It is reported on a :attr:`OutputStatus.NOT_JSON` result because it is the
    only evidence a reader gets about *why* the bytes were not readable: a
    failure with an empty recovery list reads as "the model sent prose", and a
    failure that says it applied a fence unwrap and a full-width fold reads as
    "the model sent JSON we could not repair". Those are different bugs and
    they send an operator to different places.
    """
    mismatched: OutputResult | None = None
    for candidate, recoveries in attempts:
        if not candidate.strip():
            continue
        try:
            value = json.loads(candidate)
        except (ValueError, RecursionError):
            continue
        kind = _shape_of(value)
        if wanted != EXPECT_ANY and kind != wanted:
            # Keep looking: a later reading may satisfy the contract, and
            # reporting a mismatch from the first reading would report a
            # problem the reply does not actually have.
            if mismatched is None:
                mismatched = _result(
                    OutputStatus.WRONG_SHAPE,
                    raw,
                    wanted,
                    name,
                    "shape",
                    recoveries,
                    value,
                    kind,
                )
            continue
        recovered = bool(recoveries)
        return _result(
            OutputStatus.RECOVERED if recovered else OutputStatus.OK,
            raw,
            wanted,
            name,
            "recovered" if recovered else "",
            recoveries,
            value,
            kind,
        )
    if mismatched is not None:
        return mismatched
    return _result(
        OutputStatus.NOT_JSON, raw, wanted, name, "unparseable", attempted
    )


def parse_json_object(text: Any, *, label: str = "", **kwargs: Any) -> OutputResult:
    """:func:`parse_json` for the common contract: one JSON object."""
    return parse_json(text, expected=EXPECT_OBJECT, label=label, **kwargs)


def parse_json_array(text: Any, *, label: str = "", **kwargs: Any) -> OutputResult:
    """:func:`parse_json` for the common contract: one JSON array."""
    return parse_json(text, expected=EXPECT_ARRAY, label=label, **kwargs)


def classify_completion(
    text: Any,
    *,
    called_a_tool: bool = False,
    finish_reason: str = "",
    max_chars: int = MAX_INPUT_CHARS,
    label: str = "",
) -> OutputResult:
    """Classify a completion whose content is already in hand.

    Not a parser: this never looks for JSON. It exists so the transport and this
    layer cannot disagree about the two failures that have to be told apart
    before anything is parsed, and it is why the tool-call case is
    :attr:`OutputStatus.REFUSED` here instead of a private branch in
    :mod:`paperwhisperer.llm.client`.

    The distinction is the point, and it is worth one more paragraph because it
    used to cost a 500. A reasoning model asked a question the document cannot
    answer replies with HTTP 200, ``finish_reason="tool_calls"`` and an empty
    ``content``: it has decided mid-thought to go looking for the document with
    a tool this product never offered. Reported as "the service returned empty
    content" that gets retried three times -- and an identical prompt produces
    an identical tool call -- so the user gets a 500 for asking the single most
    likely question they could ask. So the transport classifies it as
    :attr:`OutputStatus.REFUSED`, does not retry it, and says
    :data:`TOOL_CALL_MESSAGE`.

    ``finish_reason == "length"`` with no content is :attr:`OutputStatus.EMPTY`
    with :attr:`~OutputResult.detail` ``"token_budget"``, and that distinction is
    the second half of the same lesson: it is not the model returning nothing,
    it is the model spending the whole budget on reasoning and having none left
    to answer in. A reasoning-capable model against a small ``max_tokens`` fails
    exactly that way, and reporting it as "returned empty content" sends an
    operator to debug the wrong thing entirely.
    """
    name = str(label or "")
    raw = "" if text is None else str(text)
    blank = not raw.strip()
    if called_a_tool and blank:
        return _result(OutputStatus.REFUSED, raw, EXPECT_ANY, name, "tool_call")
    if blank:
        detail = "token_budget" if str(finish_reason or "").strip() == "length" else "blank"
        return _result(OutputStatus.EMPTY, raw, EXPECT_ANY, name, detail)
    if len(raw) > max_chars:
        return _result(OutputStatus.OVERSIZED, raw, EXPECT_ANY, name, "chars")
    return _result(OutputStatus.OK, raw, EXPECT_ANY, name, "", (), raw, "scalar")
