"""Parse quality gate.

A parser that returns "some text" can still have failed badly: a scan that was
never OCR'd yields a page of empty strings, a broken CMap yields U+FFFD noise,
a two-column layout parsed as one column reads perfectly and is still wrong.
None of those failures raise. They surface much later as a citation that
cannot be verified, which is the most expensive place to discover them.

So every parse is inspected here, and the inspection is deliberately
**repetitive, per-rule and non-collapsing**: the report is a list of flags
carrying the numbers that produced them, never a single score. A score would
be the more comfortable artifact — one number to sort on, one threshold to
tune — but it destroys the only thing the reader needs, which is the argument
"this document has 12 blank pages out of 20, so the layout was mis-detected"
versus an unfalsifiable ``quality_score = 0.61``. Severity is per rule and
configurable, because a blank page in a slide deck means something different
than a blank page in a scanned thesis.

The gate never rewrites a document. It reports, and the orchestrator decides
whether to retry with OCR or fall back to another backend.

Two properties the rest of the pipeline depends on:

1. **No score, ever.** ``ParseQualityReport`` has no numeric verdict field and
   neither does any flag. ``summary()`` counts flags; it does not weight them.
2. **A rule that cannot judge stays silent.** A ``.txt`` upload has no pages
   to be missing, so ``page_coverage`` and ``blank_page_ratio`` abstain rather
   than inventing a failure. A flag therefore always means a real observation,
   and "one flag" is a meaningful unit.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any

from paperwhisperer.core import config
from paperwhisperer.ingestion.schema import Element, ParsedDocument, SourceLocator

#: Mirrors ``schema._PDF_PAGE``. Duplicated rather than imported because the
#: schema treats its format constants as private; the value is a wire format
#: both modules must agree on.
_PDF_PAGE = "pdf_page"

#: Whitespace that a text layer is allowed to contain. Everything else in the
#: C0 range is a decoding failure, and ``\x00`` is the classic CMap miss.
_ALLOWED_CONTROL = frozenset("\t\n\r\f\v")

#: Digits are replaced before furniture is compared so that "Page 1" and
#: "Page 2" are recognised as the same running footer. Those repeated numbers
#: are precisely the noise that floods an embedding corpus, yet a literal
#: string comparison never notices them.
_FURNITURE_DIGITS = re.compile(r"\d+")


class QualitySeverity(str, Enum):
    """How badly one rule was violated.

    ``WARN`` means the parse is usable but degraded — a reader can still work
    and should be told. ``FAIL`` means a downstream promise is broken: no
    verifiable citations, or content that is simply missing.
    """

    WARN = "warn"
    FAIL = "fail"


class QualityStatus(str, Enum):
    """Aggregate verdict: the most severe severity present, or ``PASS``."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


#: Compared as floats that come from division, so a ratio that should equal a
#: threshold exactly must not trip the check through representation noise.
_RATIO_EPSILON = 1e-9


def _is_garbled(char: str) -> bool:
    """Whether a single character is a decoding failure rather than content.

    Covers the four ways a text layer goes wrong that are worth catching:
    the explicit replacement character, C0 controls left over from a bad
    encoding, private-use code points (a font subset mapped to PUA instead of
    to Unicode) and unassigned code points.
    """
    if char in _ALLOWED_CONTROL:
        return False
    if char == "\ufffd":
        return True
    return unicodedata.category(char) in {"Cc", "Co", "Cs", "Cn"}


def _ratio(part: int, whole: int) -> float:
    """A share rounded for logging, defined as 0.0 for an empty denominator."""
    if whole <= 0:
        return 0.0
    return round(part / whole, 4)


def _page_of(element: Element) -> int | None:
    """The 1-based page an element came from, if it is a paginated PDF one."""
    locator = element.locator
    if locator.kind == _PDF_PAGE and locator.page is not None:
        return locator.page
    return None


def _normalize_furniture(text: str) -> str:
    """Collapse whitespace and mask digits so running headers compare equal."""
    return _FURNITURE_DIGITS.sub("#", " ".join(text.split())).lower()


def _coerce_like(default: Any, configured: Any) -> Any:
    """Return ``configured`` shaped like ``default``, or ``default`` itself.

    Thresholds arrive from environment strings. A ratio of ``"0.2"`` has to
    become a real number for the comparison to mean anything, but a value that
    cannot be read as one must not take the gate down at import time — the
    default is the safe bar, and the flag still reports the observed number.
    """
    if configured is None:
        return default
    if isinstance(default, bool) or not isinstance(default, (int, float)):
        return default
    try:
        return type(default)(configured)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------
# Thresholds
# --------------------------------------------------------------------------


@dataclass(slots=True)
class QualityThresholds:
    """Every bar the gate can trip, and the severity each rule reports it at.

    All eight rules read from a single instance so that a corpus can be
    re-judged under a different policy without touching rule code, and so that
    a report can carry the bars it was judged against. Field names carry their
    unit (``min_chars_per_page``) because these numbers are read by humans
    tuning a corpus and ``0.2`` means nothing without knowing what it measures.

    The severity overrides live here rather than in a second policy object
    because there is only one thing to re-judge. A rule's own ``severity``
    stays a class attribute — the rule's opinion about itself — and this
    mapping is the deployment's opinion about which of those findings are
    worth blocking ingestion on. Empty overrides mean every rule reports
    exactly what it always did.
    """

    #: Share of declared pages that produced at least one body element. Below
    #: this the parser gave up partway through a long document, which is the
    #: most common way a "successful" parse silently truncates a paper.
    min_page_coverage_ratio: float = 0.9
    #: Share of seen pages that produced no body text at all. Scans without
    #: OCR, and two-column layouts collapsed into one column, both show up here.
    max_blank_page_ratio: float = 0.2
    #: Share of characters that are decoding failures. A single U+FFFD in a
    #: title is noise; a twentieth of the document is a broken text layer.
    max_garbled_char_ratio: float = 0.01
    #: Share of lines short enough to count as fragments. Justified tables,
    #: formulae and narrow columns all produce these, which is why the rule
    #: only ever warns.
    max_fragmented_line_ratio: float = 0.3
    #: Line length at or below which a line is a fragment.
    max_fragment_line_chars: int = 3
    #: Below this many lines a ratio is anecdote, not evidence, so the rule
    #: abstains rather than warning about a three-line abstract.
    min_lines_for_ratio: int = 10
    #: How many times one running header/footer may repeat before it is
    #: reported. Two is normal for a two-page document; twenty is a corpus bug.
    min_furniture_repeat_count: int = 3
    #: Average body characters per page. A dense paper page runs well past
    #: this; a page under it usually means the text layer was never read.
    min_chars_per_page: int = 200
    #: The same bar for unpaginated formats, which have no page to divide by.
    #: A 5-character upload is not a degraded paper, it has no citable content.
    min_document_chars: int = 200
    #: Share of tables that are outlines with nothing in them.
    max_empty_table_ratio: float = 0.5
    #: Share of body elements a citation can be resolved back to. The P0 exit
    #: criterion is 1.0, so anything less is a broken promise to the reader.
    min_resolvable_locator_ratio: float = 1.0
    #: Per-rule severity overrides, ``rule_id -> "warn" | "fail"``. T0.6 asks
    #: for WARN/FAIL to be a policy choice rather than a constant: a blank page
    #: in a slide deck is a different finding from a blank page in a scanned
    #: thesis, and which of them should stop ingestion depends on the
    #: deployment, not on the check. An entry only re-levels a rule; it can
    #: never make a rule fire that would not otherwise have fired, because
    #: "this finding is only a warning" and "this finding is not a finding" are
    #: different claims and the operator is only entitled to make the first.
    severity_overrides: dict[str, str] = field(default_factory=dict)

    def severity_for(self, rule_id: str, default: QualitySeverity) -> QualitySeverity:
        """The severity ``rule_id`` reports at, or ``default`` if not overridden.

        An unreadable override falls back to ``default`` rather than raising,
        for the same reason ``_coerce_like`` does: a gate that refuses to judge
        a document because a settings file says ``block`` instead of ``fail``
        is a worse failure than one that judges it at the severity the rule
        author chose.
        """
        configured = self.severity_overrides.get(rule_id)
        if configured is None:
            return default
        try:
            return QualitySeverity(str(configured).strip().lower())
        except ValueError:
            return default

    def to_dict(self) -> dict[str, Any]:
        """Every bar and every severity override, with the mapping copied out.

        Copied rather than shared, so a stored report's thresholds cannot be
        edited through the live dataclass the report was judged with.
        """
        payload: dict[str, Any] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            payload[item.name] = dict(value) if isinstance(value, dict) else value
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> QualityThresholds:
        """Build thresholds from a mapping, rejecting unknown keys.

        A misspelled threshold name is silently ignored by ``setattr``, which
        would leave a corpus silently judged against a default the operator
        believes they changed. Failing loudly is the whole point of a gate.
        The same reasoning applies to a severity policy: a rule id that does
        not exist, or a severity that is neither warn nor fail, is a decision
        nobody made, so it is reported here rather than sitting in the config
        looking like one.
        """
        known = {item.name for item in fields(cls)}
        unknown = [key for key in payload if key not in known]
        if unknown:
            listed = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown quality threshold key(s): {listed}")
        overrides = _severity_overrides_from(payload.get("severity_overrides"), strict=True)
        return cls(
            severity_overrides=overrides,
            **{key: payload[key] for key in payload if key != "severity_overrides"},
        )


def _registered_rule_ids() -> frozenset[str]:
    """Ids of the registered rules, for validating a severity policy.

    ``RULES`` is defined further down this module; reading it here is safe
    because this is a call-time lookup and nothing resolves it at import time.
    """
    return frozenset(rule.rule_id for rule in RULES)


def _severity_overrides_from(raw: Any, *, strict: bool) -> dict[str, str]:
    """Parse a per-rule severity policy into canonical ``rule_id -> "warn"`` form.

    Accepts a mapping or the ``"rule=severity, rule=severity"`` string an
    environment variable carries. Two postures for the same input, because
    the two callers mean different things by it: a policy written in code is
    an assertion about how this deployment gates parses, so a misspelled rule
    id or an invented severity is raised; a policy read from an operator's
    settings file is filtered instead, because refusing to judge any document
    over a leftover entry in that file trades a cosmetic problem for a dead
    ingestion path.
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        pairs = [(str(key), value) for key, value in raw.items()]
    else:
        pairs = []
        for piece in re.split(r"[,;\s]+", str(raw).strip()):
            if not piece:
                continue
            name, separator, severity = piece.partition("=")
            if not separator:
                if strict:
                    raise ValueError(
                        f"Quality severity policy entry is not rule_id=severity: {piece!r}"
                    )
                continue
            pairs.append((name, severity))

    known = _registered_rule_ids()
    resolved: dict[str, str] = {}
    for name, severity in pairs:
        rule_id = str(name).strip()
        if rule_id not in known:
            if strict and rule_id:
                raise ValueError(f"Unknown quality rule in severity policy: {rule_id!r}")
            continue
        try:
            resolved[rule_id] = QualitySeverity(str(severity).strip().lower()).value
        except ValueError:
            if strict:
                listed = ", ".join(item.value for item in QualitySeverity)
                raise ValueError(
                    f"Unknown severity {severity!r} for rule {rule_id!r} (expected {listed})"
                ) from None
    return resolved


def thresholds_from_config() -> QualityThresholds:
    """Read the bars from ``core.config``, falling back to the defaults.

    ``getattr`` on purpose: this module and the ``PARSER_*`` config block land
    in the same release but not necessarily the same commit, and a quality
    gate that refuses to import is worse than one running on defaults. The
    same reasoning covers ``PARSER_QUALITY_SEVERITY_OVERRIDES`` and any
    individual ``PARSER_QUALITY_<THRESHOLD>``: absent, malformed or
    unreadable, it leaves the built-in default in force.
    """
    defaults = QualityThresholds()
    resolved: dict[str, Any] = {}
    for item in fields(QualityThresholds):
        default_value = getattr(defaults, item.name)
        # The severity policy is a mapping, not a scalar, and has its own
        # reader; `_coerce_like` has nothing to say about it.
        if isinstance(default_value, dict):
            continue
        configured = getattr(config, f"PARSER_QUALITY_{item.name.upper()}", None)
        resolved[item.name] = _coerce_like(default_value, configured)
    resolved["severity_overrides"] = _severity_overrides_from(
        getattr(config, "PARSER_QUALITY_SEVERITY_OVERRIDES", None),
        strict=False,
    )
    return QualityThresholds(**resolved)


# --------------------------------------------------------------------------
# Flags
# --------------------------------------------------------------------------


@dataclass(slots=True)
class QualityFlag:
    """One observation, with the numbers that justify it.

    ``details`` is not decoration: when a user asks why a document was
    rejected, this dict is the answer. It holds the observed value beside the
    threshold it was compared against, so the reading never has to be re-run
    to be understood.
    """

    rule_id: str
    severity: QualitySeverity
    message: str
    locator: SourceLocator | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "rule_id": self.rule_id,
            "severity": self.severity.value,
            "message": self.message,
        }
        if self.locator is not None:
            payload["locator"] = self.locator.to_dict()
        if self.details:
            payload["details"] = dict(self.details)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> QualityFlag:
        locator_payload = payload.get("locator")
        return cls(
            rule_id=str(payload.get("rule_id") or ""),
            severity=QualitySeverity(str(payload.get("severity") or QualitySeverity.WARN)),
            message=str(payload.get("message") or ""),
            locator=SourceLocator.from_dict(locator_payload) if locator_payload else None,
            details=dict(payload.get("details") or {}),
        )


# --------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------


class QualityRule:
    """One independently registrable check.

    Rules are objects rather than functions so that ``rule_id`` and severity
    travel with the logic, and so the eval runner can address a single rule by
    name instead of re-running the whole gate.
    """

    #: Stable identifier used in flags, logs and eval reports.
    rule_id: str = ""
    #: Severity this rule reports at when nothing re-levels it. Every rule owns
    #: its own, because a fragmented line and a missing page are not the same
    #: kind of problem. Deployment policy may override it per run; see
    #: ``QualityThresholds.severity_overrides`` and ``effective_severity``.
    severity: QualitySeverity = QualitySeverity.WARN

    def __init__(self, thresholds: QualityThresholds | None = None) -> None:
        self.thresholds = thresholds or QualityThresholds()

    def evaluate(self, document: ParsedDocument) -> list[QualityFlag]:
        """Flags for this rule only — zero when it passes or cannot judge."""
        raise NotImplementedError

    def with_thresholds(self, thresholds: QualityThresholds) -> QualityRule:
        """A copy of this rule bound to different bars, and to their policy.

        Copies rather than mutation, because ``RULES`` holds one instance per
        rule for the whole process: a per-parse policy applied by mutating the
        shared instance would leak one document's policy into the next one.
        """
        return type(self)(thresholds)

    @property
    def effective_severity(self) -> QualitySeverity:
        """The severity this rule reports at, after the configured policy.

        ``severity`` remains the class attribute so that the rule's own view of
        itself is still readable off ``RULES`` and ``get_rule``; the policy in
        ``thresholds`` only decides what a report says about it. A rule the
        policy does not name emits exactly what it always did.
        """
        return self.thresholds.severity_for(self.rule_id, self.severity)

    def _flag(
        self,
        message: str,
        details: dict[str, Any],
        locator: SourceLocator | None = None,
    ) -> QualityFlag:
        return QualityFlag(
            rule_id=self.rule_id,
            severity=self.effective_severity,
            message=message,
            locator=locator,
            details=details,
        )


class PageCoverageRule(QualityRule):
    """Did the parser reach every page the file claims to have?

    A parser that hits an exception on page 3 of 20 and returns what it has is
    reporting success. The missing tail looks like a short paper, and the
    analysis then confidently summarises half a document.
    """

    rule_id = "page_coverage"
    severity = QualitySeverity.FAIL

    def evaluate(self, document: ParsedDocument) -> list[QualityFlag]:
        page_count = document.page_count
        # Unpaginated formats have no page to be missing; abstaining is the
        # difference between a real finding and a fabricated one.
        if page_count <= 0:
            return []
        covered = {_page_of(element) for element in document.body_elements}
        covered.discard(None)
        ratio = _ratio(len(covered), page_count)
        if ratio + _RATIO_EPSILON >= self.thresholds.min_page_coverage_ratio:
            return []
        missing = [page for page in range(1, page_count + 1) if page not in covered]
        return [self._flag(
            message=(
                f"仅解析出 {len(covered)}/{page_count} 页的正文，"
                f"缺失页码：{_sample_pages(missing)}"
            ),
            details={
                "page_count": page_count,
                "covered_pages": len(covered),
                "coverage_ratio": ratio,
                "min_page_coverage_ratio": self.thresholds.min_page_coverage_ratio,
                "missing_pages": missing[:20],
                "missing_page_count": len(missing),
            },
        )]


class BlankPageRatioRule(QualityRule):
    """How many pages came back without any body text.

    A page with text is not necessarily parsed correctly — a two-column layout
    read as one column still produces text. A page with *no* text is
    unambiguous: either the scan was never OCR'd, or the column logic dropped
    everything on that page.
    """

    rule_id = "blank_page_ratio"
    severity = QualitySeverity.FAIL

    def evaluate(self, document: ParsedDocument) -> list[QualityFlag]:
        body_pages: dict[int, int] = {}
        seen_pages: set[int] = set()
        for element in document.elements:
            page = _page_of(element)
            if page is None:
                continue
            seen_pages.add(page)
            if not element.is_furniture:
                body_pages[page] = body_pages.get(page, 0) + len(element.text.strip())
        if not seen_pages:
            return []
        blank = sorted(page for page in seen_pages if body_pages.get(page, 0) == 0)
        ratio = _ratio(len(blank), len(seen_pages))
        if ratio + _RATIO_EPSILON <= self.thresholds.max_blank_page_ratio:
            return []
        return [self._flag(
            message=(
                f"{len(blank)}/{len(seen_pages)} 页没有任何正文，"
                f"空页码：{_sample_pages(blank)}；常见原因是扫描件未 OCR"
            ),
            details={
                "seen_pages": len(seen_pages),
                "blank_pages": len(blank),
                "blank_page_ratio": ratio,
                "max_blank_page_ratio": self.thresholds.max_blank_page_ratio,
                "blank_page_numbers": blank[:20],
            },
        )]


class GarbledCharactersRule(QualityRule):
    """How much of the text is decoding failure rather than content.

    A missing CMap or a wrong encoding guess produces characters that look
    like text to a length check and like noise to a reader. Every downstream
    consumer — embeddings, citations, the reader's own eyes — is wrong in a way
    no exception would have reported.
    """

    rule_id = "garbled_characters"
    severity = QualitySeverity.FAIL

    def evaluate(self, document: ParsedDocument) -> list[QualityFlag]:
        body = document.body_elements
        total = sum(len(element.text) for element in body)
        if total == 0:
            return []
        bad = 0
        samples: Counter[str] = Counter()
        for element in body:
            for char in element.text:
                if _is_garbled(char):
                    bad += 1
                    if len(samples) < 8:
                        samples[f"U+{ord(char):04X}"] += 1
        ratio = _ratio(bad, total)
        if ratio + _RATIO_EPSILON <= self.thresholds.max_garbled_char_ratio:
            return []
        return [self._flag(
            message=(
                f"{bad}/{total} 个字符为乱码或控制字符（占比 {ratio}），"
                f"典型字符：{', '.join(sorted(samples))}"
            ),
            details={
                "garbled_chars": bad,
                "total_chars": total,
                "garbled_char_ratio": ratio,
                "max_garbled_char_ratio": self.thresholds.max_garbled_char_ratio,
                "sample_codepoints": dict(sorted(samples.items())),
            },
        )]


class FragmentedLinesRule(QualityRule):
    """Whether the text has been shattered into one-to-three-character lines.

    This is what a PDF grid extractor does when it treats every ruled cell as
    a text block: the character count looks fine, so density checks pass, while
    every sentence is now a dozen separate lines. It warns and never fails,
    because narrow table columns and formulae legitimately look the same.
    """

    rule_id = "fragmented_lines"
    severity = QualitySeverity.WARN

    def evaluate(self, document: ParsedDocument) -> list[QualityFlag]:
        lines: list[str] = []
        for element in document.body_elements:
            lines.extend(line for line in element.text.splitlines() if line.strip())
        if len(lines) < self.thresholds.min_lines_for_ratio:
            return []
        limit = self.thresholds.max_fragment_line_chars
        short = [line for line in lines if len(line.strip()) <= limit]
        ratio = _ratio(len(short), len(lines))
        if ratio + _RATIO_EPSILON <= self.thresholds.max_fragmented_line_ratio:
            return []
        return [self._flag(
            message=(
                f"{len(short)}/{len(lines)} 行不超过 {limit} 个字符，"
                f"疑似表格/版面被逐格切碎（占比 {ratio}）"
            ),
            details={
                "total_lines": len(lines),
                "fragmented_lines": len(short),
                "fragmented_line_ratio": ratio,
                "max_fragmented_line_ratio": self.thresholds.max_fragmented_line_ratio,
                "max_fragment_line_chars": limit,
                "samples": [line.strip() for line in short[:5]],
            },
        )]


class RepeatedFurnitureRule(QualityRule):
    """Whether one running header or footer is being repeated as body text.

    A journal name repeated on all 20 pages is not 20 facts, it is one fact
    repeated 20 times, and it is a real share of the embedding corpus. A
    literal comparison misses the commonest form — page numbers that differ
    every page — so digits are masked before counting.
    """

    rule_id = "repeated_furniture"
    severity = QualitySeverity.WARN

    def evaluate(self, document: ParsedDocument) -> list[QualityFlag]:
        furniture = [element for element in document.elements if element.is_furniture]
        if len(furniture) < self.thresholds.min_furniture_repeat_count:
            return []
        counts = Counter(
            normalized
            for normalized in (_normalize_furniture(el.text) for el in furniture)
            if normalized
        )
        if not counts:
            return []
        repeated, count = counts.most_common(1)[0]
        if count < self.thresholds.min_furniture_repeat_count:
            return []
        return [self._flag(
            message=(
                f"同一条页眉/页脚重复出现 {count} 次（占页眉页脚 {count}/{len(furniture)}），"
                f"内容：{repeated[:40]}"
            ),
            details={
                "furniture_elements": len(furniture),
                "repeat_count": count,
                "min_furniture_repeat_count": self.thresholds.min_furniture_repeat_count,
                "furniture_repeat_ratio": _ratio(count, len(furniture)),
                "normalized_text": repeated,
            },
        )]


class TextDensityRule(QualityRule):
    """Whether the pages actually contain text.

    This is the rule that catches a scan which was never OCR'd: the parse
    succeeds, every element is a valid element, and the document is empty. The
    unpaginated branch exists for the same reason in ``.txt`` and ``.docx``,
    where there is no page to average over.
    """

    rule_id = "text_density"
    severity = QualitySeverity.FAIL

    def evaluate(self, document: ParsedDocument) -> list[QualityFlag]:
        chars = document.char_count
        page_count = document.page_count
        if page_count > 0:
            per_page = round(chars / page_count, 2)
            if per_page >= self.thresholds.min_chars_per_page:
                return []
            return [self._flag(
                message=(
                    f"平均每页仅 {per_page} 个正文字符，"
                    f"低于 {self.thresholds.min_chars_per_page}；页面可能未被解析"
                ),
                details={
                    "char_count": chars,
                    "page_count": page_count,
                    "chars_per_page": per_page,
                    "min_chars_per_page": self.thresholds.min_chars_per_page,
                },
            )]
        # No pages to average over, so judge the document as a whole. An
        # empty parse is a failure here rather than a silent pass.
        if chars >= self.thresholds.min_document_chars:
            return []
        return [self._flag(
            message=(
                f"全文仅解析出 {chars} 个正文字符，"
                f"低于 {self.thresholds.min_document_chars}；没有可引用的内容"
            ),
            details={
                "char_count": chars,
                "page_count": page_count,
                "min_document_chars": self.thresholds.min_document_chars,
            },
        )]


class EmptyTableShellsRule(QualityRule):
    """Whether tables were detected that contain nothing.

    Ruled lines are easy to detect and easy to over-detect; a table outline
    with blank cells is what comes back when a figure, a signature block or a
    chart frame was mistaken for a table. It is only a warning because a
    document can legitimately contain one stray shell.
    """

    rule_id = "empty_table_shells"
    severity = QualitySeverity.WARN

    def evaluate(self, document: ParsedDocument) -> list[QualityFlag]:
        tables = [element for element in document.body_elements if element.table is not None]
        if not tables:
            return []
        shells = [element for element in tables if element.table.is_empty_shell]
        ratio = _ratio(len(shells), len(tables))
        if ratio + _RATIO_EPSILON < self.thresholds.max_empty_table_ratio:
            return []
        first = shells[0]
        return [self._flag(
            message=(
                f"{len(shells)}/{len(tables)} 个表格是空壳（只有线框没有内容），"
                f"可能是图片或图表被误判为表格"
            ),
            details={
                "table_count": len(tables),
                "empty_shell_count": len(shells),
                "empty_table_ratio": ratio,
                "max_empty_table_ratio": self.thresholds.max_empty_table_ratio,
                "table_element_ids": [element.element_id for element in shells[:10]],
            },
            locator=first.locator,
        )]


class LocatorCoverageRule(QualityRule):
    """Whether citations can be resolved back to a location.

    This is the promise the whole evidence design rests on. A parser that
    degrades to plain text keeps every other check green while silently
    deleting the ability to point at a page, so the loss has to be caught
    here or not at all.
    """

    rule_id = "locator_coverage"
    severity = QualitySeverity.FAIL

    def evaluate(self, document: ParsedDocument) -> list[QualityFlag]:
        body = document.body_elements
        # An empty document has no coverage to measure; text_density is the
        # rule that reports it.
        if not body:
            return []
        located = [element for element in body if element.locator.is_resolvable]
        ratio = _ratio(len(located), len(body))
        if ratio + _RATIO_EPSILON >= self.thresholds.min_resolvable_locator_ratio:
            return []
        unlocated = [element for element in body if not element.locator.is_resolvable]
        return [self._flag(
            message=(
                f"{len(unlocated)}/{len(body)} 个正文元素没有可解析的来源定位，"
                f"引用将无法回跳到原文"
            ),
            details={
                "body_elements": len(body),
                "resolvable_elements": len(located),
                "locator_coverage": ratio,
                "min_resolvable_locator_ratio": self.thresholds.min_resolvable_locator_ratio,
                "unresolvable_kinds": sorted({
                    element.locator.kind for element in unlocated
                }),
            },
            locator=unlocated[0].locator,
        )]


def _sample_pages(pages: list[int]) -> str:
    """Render a page list for a message without letting it become a log line."""
    if not pages:
        return "无"
    head = ", ".join(str(page) for page in pages[:10])
    if len(pages) > 10:
        return f"{head} 等 {len(pages)} 页"
    return head


#: The registered rules, in report order. This tuple is the single source of
#: truth for which checks exist; ``get_rule`` and ``evaluate_quality`` both
#: read it, and a new rule is added by implementing the class and appending it
#: here. Bound to the built-in defaults, so the registry stays stable while
#: ``evaluate_quality`` resolves config-provided bars per call.
RULES: tuple[QualityRule, ...] = (
    PageCoverageRule(),
    BlankPageRatioRule(),
    GarbledCharactersRule(),
    FragmentedLinesRule(),
    RepeatedFurnitureRule(),
    TextDensityRule(),
    EmptyTableShellsRule(),
    LocatorCoverageRule(),
)


def get_rule(rule_id: str) -> QualityRule:
    """Look up a registered rule by id.

    Raises ``KeyError`` on an unknown id rather than returning ``None``. A
    silently missing rule reads as "this check passed", which is the one
    outcome a quality gate must never produce by accident.
    """
    for rule in RULES:
        if rule.rule_id == rule_id:
            return rule
    raise KeyError(f"Unknown quality rule: {rule_id!r}")


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


@dataclass(slots=True)
class ParseQualityReport:
    """Every flag from one parse, and the verdict they add up to.

    Note what is absent: there is no score, no weighted total and no pass
    percentage. The status is the maximum severity present, and the flags
    remain individually addressable, because "which check failed and on what
    numbers" is the only question this report exists to answer.
    """

    document_id: str = ""
    parser_name: str = ""
    status: QualityStatus = QualityStatus.PASS
    flags: list[QualityFlag] = field(default_factory=list)
    #: Rule ids actually run, so a report that abstained is distinguishable
    #: from one whose rules all passed.
    evaluated_rules: tuple[str, ...] = ()
    #: The bars and severities this report was judged against, kept so a
    #: stored verdict can be re-read months later without guessing which
    #: policy produced it.
    thresholds: QualityThresholds = field(default_factory=QualityThresholds)

    @classmethod
    def from_flags(
        cls,
        document: ParsedDocument,
        flags: list[QualityFlag],
        rules: tuple[QualityRule, ...],
        thresholds: QualityThresholds,
    ) -> ParseQualityReport:
        return cls(
            document_id=document.document_id,
            parser_name=document.parser_name,
            status=status_of(flags),
            flags=list(flags),
            evaluated_rules=tuple(rule.rule_id for rule in rules),
            thresholds=thresholds,
        )

    @property
    def failed_rules(self) -> tuple[str, ...]:
        return tuple(flag.rule_id for flag in self.flags if flag.severity is QualitySeverity.FAIL)

    @property
    def warned_rules(self) -> tuple[str, ...]:
        return tuple(flag.rule_id for flag in self.flags if flag.severity is QualitySeverity.WARN)

    def summary(self) -> dict[str, Any]:
        """A log-friendly digest: counts per severity plus the rules involved.

        Deliberately a count and a list. Anything that collapsed these flags
        into one comparable number would be exactly the black box this gate
        exists to avoid.
        """
        counts = {severity.value: 0 for severity in QualitySeverity}
        for flag in self.flags:
            counts[flag.severity.value] += 1
        return {
            "document_id": self.document_id,
            "parser_name": self.parser_name,
            "status": self.status.value,
            "flag_total": len(self.flags),
            "counts_by_severity": counts,
            "failed_rules": list(self.failed_rules),
            "warned_rules": list(self.warned_rules),
            "evaluated_rules": list(self.evaluated_rules),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "parser_name": self.parser_name,
            "status": self.status.value,
            "flags": [flag.to_dict() for flag in self.flags],
            "evaluated_rules": list(self.evaluated_rules),
            "thresholds": self.thresholds.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ParseQualityReport:
        status = QualityStatus(str(payload.get("status") or QualityStatus.PASS))
        thresholds_payload = payload.get("thresholds")
        return cls(
            document_id=str(payload.get("document_id") or ""),
            parser_name=str(payload.get("parser_name") or ""),
            status=status,
            flags=[QualityFlag.from_dict(item) for item in payload.get("flags") or []],
            evaluated_rules=tuple(str(item) for item in payload.get("evaluated_rules") or ()),
            thresholds=(
                QualityThresholds.from_dict(thresholds_payload)
                if isinstance(thresholds_payload, dict)
                else QualityThresholds()
            ),
        )


def status_of(flags: list[QualityFlag]) -> QualityStatus:
    """The worst severity in ``flags``, or ``PASS`` when there are none."""
    if any(flag.severity is QualitySeverity.FAIL for flag in flags):
        return QualityStatus.FAIL
    if any(flag.severity is QualitySeverity.WARN for flag in flags):
        return QualityStatus.WARN
    return QualityStatus.PASS


def evaluate_quality(
    document: ParsedDocument,
    thresholds: QualityThresholds | None = None,
    rules: tuple[QualityRule, ...] | None = None,
) -> ParseQualityReport:
    """Run the quality gate over one parsed document.

    ``thresholds`` defaults to the bars in ``core.config`` so that a change to
    corpus policy takes effect without a reimport. ``rules`` narrows the run
    for the eval runner, which needs a per-rule view to compare backends.
    """
    resolved = thresholds if thresholds is not None else thresholds_from_config()
    active = tuple(rules) if rules is not None else RULES
    bound = tuple(rule.with_thresholds(resolved) for rule in active)
    flags: list[QualityFlag] = []
    for rule in bound:
        flags.extend(rule.evaluate(document))
    return ParseQualityReport.from_flags(document, flags, bound, resolved)
