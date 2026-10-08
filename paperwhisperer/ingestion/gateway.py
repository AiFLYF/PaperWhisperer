"""Parser gateway: the one door every uploaded document goes through.

Before this module each caller reached for ``DocumentLoader.load`` and got a
string. That made three things impossible, and all three are things this
project is built on: a citation cannot name a page, two parsers cannot be
compared, and a parser cannot be swapped without touching business code. The
gateway fixes the shape of the problem rather than the shape of any one
backend — an adapter returns a :class:`~paperwhisperer.ingestion.schema.ParsedDocument`,
and everything downstream reads only that.

Three decisions are worth arguing for, because the obvious alternative is
worse in each case.

**Nothing propagates out of :meth:`ParserRouter.parse`.** Callers get a
``ParseResult`` with ``status="failed"`` and a Chinese ``error_message``, never
an exception. A parser backend fails for reasons that are not bugs (a
dependency that is not installed, a cloud quota, a password-protected file), and
each of those should degrade to a fallback rather than to a 500. Exceptions
that *are* bugs still surface — as a failed result carrying the original text
plus a logged traceback.

**Adapter output is validated where it enters, and a violation is a
diagnosis rather than an excuse.** The no-raise guarantee above is about not
crashing the request; it is not a licence to accept a structurally broken
document. A parser that emits an element whose locator claims a page it does
not have, or a table that arrived with no grid, produces a document that looks
perfectly fine and cites the wrong places — the one failure mode this project
exists to rule out. So :meth:`ParserRouter._normalize` runs the returned
document through
:class:`~paperwhisperer.ingestion.contracts.ParsedDocumentContract` before
accepting it, and a refusal becomes a failed result whose Chinese message names
the offending field and says why. The router still raises nothing: a contract
violation is a *worse parse*, not a worse request, so the chain continues to
the next candidate exactly as it would have for a parser that raised.

The cost is one Pydantic pass per usable document — order 60 ms for 5 000
elements, on a path that already spent hundreds of milliseconds inside a
converter. That is the right trade at a boundary, and it is only affordable
because the dataclasses stay unvalidated on the hot path. The gate is
unconditional on purpose: a check each caller has to remember to ask for is a
check nobody asks for, and the whole reason the contracts module exists is that
113 passing unit tests over a disconnected module proved nothing.

**One producer defect is repaired rather than refused, and the repair is
loud.** ``adapters/mineru_cloud.py`` builds ``SourceLocator(kind="pdf_page")``
even when MinerU reported no ``page_idx``, so ``page`` is ``None`` — a locator
that names a format it does not have, which the contract correctly refuses.
Rather than weaken the contract to match a bug, or throw away an otherwise-good
cloud parse, :meth:`ParserRouter._repair_known_locator_defects` rewrites exactly
that one shape to ``kind="unknown"`` — the spelling the contract's own error
message recommends — and records a warning on the result plus an entry in
``diagnostics``. It is deliberately narrow. A page number that is *wrong* is
unrecoverable and has to fail; a page number that is *absent* has exactly one
honest value, so the repair is a downgrade rather than a fabrication. The
producer still wants fixing; see that method for the one-line change.

**The timeout is cooperative, and this is stated rather than hidden.** There is
no async timeout primitive in the standard library, so the router runs each
parser on a worker thread and gives up on ``future.result(timeout=...)``. The
thread is *not* killed: ``future.cancel()`` only works while a call is still
queued. A wedged parser therefore keeps burning CPU after the router has moved
on, and Python's interpreter shutdown waits for it. That is a real limitation,
and the reason the timeout exists at all is that a hung parser must not hold an
HTTP request open. A future that needs a hard kill has to run the backend in a
subprocess, which is what the docling-serve and cloud adapters buy.

**``PARSER_BACKEND`` is read per call, not at construction.** The flag exists
so a bad new backend can be rolled back by editing one environment variable
and restarting nothing. A router that captured the flag in ``__init__`` would
still require a restart to honour it, and a test could not exercise both
branches without rebuilding the router.

**The quality gate is attached, not enforced.** Every usable result leaves the
router carrying a ``ParseQualityReport`` (T0.6), because the router is the one
place every parse passes through and a gate nobody calls is not a gate. What it
does *not* do is act on the verdict: rejecting a bad document means deciding
between a local OCR retry and a paid cloud call, which is a policy decision
that belongs to the orchestrator (T0.7) and needs the consent state the router
has no business reading.

Per-call attempt traces (which parsers were tried, in what order, how long each
took, why one was skipped) live on the returned
:attr:`ParseResult.diagnostics` rather than in cumulative counters on the
router. A router serves many documents, so a cumulative counter cannot attribute
elapsed time to any one of them — and attribution is the whole reason the
timeline is worth collecting. The router is assumed to be per-request, so a
cumulative counter would also grow without bound.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import importlib.util
import json
import logging
import mimetypes
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from paperwhisperer.core import config

from .schema import ParsedDocument, SourceLocator

if TYPE_CHECKING:  # pragma: no cover - imported lazily at call time, see _attach_quality
    from pydantic import ValidationError

    from .quality import ParseQualityReport

logger = logging.getLogger(__name__)

#: Every terminal state a parse can end in. ``partial`` is a real outcome and
#: not a softer spelling of failure: the document is usable, but something was
#: dropped and the caller is expected to say so in the UI.
PARSE_STATUSES = ("success", "partial", "failed")

PARSE_SUCCESS = "success"
PARSE_PARTIAL = "partial"
PARSE_FAILED = "failed"

#: Statuses an adapter may use that mean one of ours, for adapters written
#: against a different vocabulary. It lives beside the three constants that
#: define the legal vocabulary because that is what it is about: the gap
#: between what an adapter says and what this module will accept.
#:
#: **Do not delete this as dead code.** ``ok`` is the docling adapter's native
#: spelling of success — its own result type is built on it, and it builds a
#: real :class:`ParseResult` from it. Delete the entry and every *successful*
#: docling parse is downgraded to a failure carrying "解析器返回了未知的解析状态。":
#: a total, silent breakage of the docling path that no docling-side test can
#: see, because the adapter is behaving exactly as written.
#:
#: It stays one entry wide, and the alias is exact: ``OK`` and ``okay`` are
#: failures. Everything not in this table is an unknown status and is
#: downgraded, because a status this module has never heard of means the
#: adapter and the gateway disagree about the contract, and guessing at the
#: intent is how a broken parse turns into a silently empty document.
STATUS_ALIASES = {"ok": PARSE_SUCCESS}

#: The parser that must remain working for the whole of Phase 0. The
#: ``PARSER_BACKEND=legacy`` rollback path selects on this name, so renaming
#: the legacy adapter silently disables the rollback — the name is a contract.
LEGACY_PARSER_NAME = "legacy"

#: Extension to media type for the formats the gateway is required to cover.
#: The values are the IANA registrations rather than the shorter aliases
#: browsers accept, because adapters compare them against what an upload
#: reports and ``application/x-pdf`` would then miss.
MEDIA_TYPE_BY_EXTENSION = {
    ".txt": "text/plain",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}

#: Default ceiling on a single parser call. Mirrors ``PARSER_TIMEOUT_SECONDS``
#: but is read through a function so an import-time snapshot cannot go stale
#: when the environment changes under a long-lived process.
DEFAULT_TIMEOUT_SECONDS = 120


class ParserError(RuntimeError):
    """A parser could not produce a document.

    Adapters raise this instead of leaking a library exception, so the router
    has one thing to catch. It subclasses ``RuntimeError`` because a parse
    failure is an operational fault, not a caller mistake — a caller passing a
    bad path gets a ``ValueError`` from the adapter instead.
    """


class ParserTimeout(ParserError):
    """A parser exceeded ``ParseOptions.timeout_seconds``.

    Distinct from :class:`ParserError` because the recovery differs: a timeout
    is usually a large or pathological document, and the orchestrator's answer
    is a different backend or a longer budget, not a retry of the same call.
    """


# --------------------------------------------------------------------------
# Options
# --------------------------------------------------------------------------


def default_timeout_seconds() -> int:
    """Read the configured per-call budget, tolerating an older config build."""
    value = getattr(config, "PARSER_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)
    try:
        return int(value)
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT_SECONDS


@dataclass
class ParseOptions:
    """Knobs one parse call may turn, and the identity they add up to.

    These are per-call rather than global because the interesting combinations
    are exactly the ones an orchestrator produces: retry the same file with
    ``force_ocr=True`` and the result must not be served from a cache entry
    built without OCR. That is why the timeout participates in
    :meth:`compute_options_hash` too — a parse that gave up early and one that
    ran to completion may hold different text, so they are different cache
    entries even though the file and flags match.
    """

    force_ocr: bool = False
    timeout_seconds: int = field(default_factory=default_timeout_seconds)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.extra, dict):
            raise ValueError("ParseOptions.extra must be a dict")

    def compute_options_hash(self) -> str:
        """A short stable digest of everything that can change the output.

        Sorted keys and compact separators make this independent of dict
        ordering, so two calls that mean the same thing produce the same
        digest. 16 hex characters is plenty: this is a cache key component,
        not a security boundary, and a collision would need two option sets
        that are already nearly identical.
        """
        payload = {
            "extra": self.extra,
            "force_ocr": bool(self.force_ocr),
            "timeout_seconds": self.timeout_seconds,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------
# Parser contract
# --------------------------------------------------------------------------


@runtime_checkable
class DocumentParser(Protocol):
    """What the router requires of a backend.

    A ``Protocol`` rather than an abstract base class on purpose: adapters live
    in separate optional dependency groups (docling pulls in torch, the cloud
    adapter pulls in nothing), and a nominal base class would force every one
    of them to import a module whose own imports may be missing.
    """

    name: str
    version: str

    def supports(self, media_type: str) -> bool:
        """Whether this parser can attempt ``media_type`` at all.

        Cheapest possible check, consulted before the parser is queued onto a
        worker. It must not touch the file: the router calls it once per
        candidate per document, and opening the file N times to answer "is this
        my format" would be absurd.
        """
        ...

    def parse(self, file_path: str, options: ParseOptions) -> ParseResult:
        """Parse ``file_path`` into a canonical document.

        Adapters raise :class:`ParserError` (or anything the router can map)
        for failure; they do not return a failed :class:`ParseResult`
        themselves, because "failed" and "I raised" must not be two spellings
        of the same event.
        """
        ...


class ParserExecutor(Protocol):
    """The slice of ``concurrent.futures.Executor`` the router actually uses.

    Narrowed to two methods so a test can substitute a deterministic fake and
    assert the timeout mapping without a real thread or a real sleep.
    """

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        ...

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        ...


# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------


@dataclass(slots=True)
class ParseResult:
    """The outcome of one parse attempt, successful or not.

    Provenance fields are stamped by the router rather than trusted from the
    adapter. An adapter that forgot to set ``parser_version`` would otherwise
    produce a document that looks cached under the wrong key, and the cache is
    keyed on exactly this triple (see T0.7) — so a missing stamp is a
    correctness bug that is cheapest to fix at the one place that sees every
    parser. For the same reason ``contract_repairs`` records what validation
    had to fix before the document could be accepted, one entry per repaired
    locator: a separate field rather than another ``warnings`` string because
    it is structured evidence, where the warnings are prose for a human. An
    empty list is the overwhelmingly common case, and it means the document was
    well-formed exactly as the adapter produced it.
    """

    status: str = PARSE_SUCCESS
    document: ParsedDocument | None = None
    quality: ParseQualityReport | None = None
    raw_artifact_uri: str = ""
    warnings: list[str] = field(default_factory=list)
    duration_ms: int = 0
    parser_name: str = ""
    #: Why a cloud parse ended the way it did: ``success``, ``partial``,
    #: ``quota_exhausted``, ``timed_out``, ``blocked`` or ``failed``.
    #:
    #: A top-level field rather than a diagnostics entry because a caller has to
    #: *branch* on it. The interface needs to say "top up your quota" for one
    #: outcome, "try again later" for another and "this will never work" for a
    #: third; when all three arrive as ``status="failed"`` plus prose, the only
    #: thing a client can do is show the prose, and the user cannot act on it.
    #: Three different situations that look identical are the same defect as a
    #: missing field, not a present one.
    cloud_outcome: str = ""
    parser_version: str = ""
    options_hash: str = ""
    error_message: str = ""
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    contract_repairs: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Aliases are resolved here rather than at the router's boundary
        # because the router never sees the pre-normalised value: the status
        # has already been rewritten by the time a result is returned to it.
        self.status = STATUS_ALIASES.get(self.status, self.status)
        if self.status not in PARSE_STATUSES:
            self.status = PARSE_FAILED
            self.error_message = self.error_message or "解析器返回了未知的解析状态。"

    @property
    def is_usable(self) -> bool:
        """Whether downstream may treat this as a document.

        ``partial`` counts. The caller is expected to surface the warnings, but
        refusing a partial parse outright would throw away a scanned-but-mostly-
        readable PDF that a human could still work with.
        """
        return self.status in {PARSE_SUCCESS, PARSE_PARTIAL}

    @classmethod
    def from_document(
        cls,
        document: ParsedDocument,
        *,
        status: str = PARSE_SUCCESS,
        warnings: list[str] | None = None,
        duration_ms: int = 0,
        quality: ParseQualityReport | None = None,
        raw_artifact_uri: str = "",
    ) -> ParseResult:
        """Wrap a parsed document, taking provenance from the document itself.

        The document already records which parser produced it, so reading it
        from there rather than asking the caller to repeat it removes a way for
        the two to disagree.
        """
        return cls(
            status=status,
            document=document,
            quality=quality,
            raw_artifact_uri=raw_artifact_uri,
            warnings=list(warnings or []),
            duration_ms=duration_ms,
            parser_name=document.parser_name,
            parser_version=document.parser_version,
            options_hash=document.options_hash,
        )

    @classmethod
    def failure(
        cls,
        error_message: str,
        *,
        parser_name: str = "",
        parser_version: str = "",
        options_hash: str = "",
        duration_ms: int = 0,
        warnings: list[str] | None = None,
        diagnostics: list[dict[str, Any]] | None = None,
        cloud_outcome: str = "",
    ) -> ParseResult:
        """Build the canonical failure value.

        Every error path funnels here, so a failure always looks the same to
        the caller: no document, an empty quality report, a message, and the
        provenance of the parser that failed. ``diagnostics`` is accepted
        because the router has already recorded the attempts that led here by
        the time it decides the chain is exhausted — dropping them would lose
        exactly the evidence needed to explain the failure.
        """
        return cls(
            status=PARSE_FAILED,
            cloud_outcome=cloud_outcome,
            document=None,
            quality=None,
            raw_artifact_uri="",
            warnings=list(warnings or []),
            duration_ms=duration_ms,
            parser_name=parser_name,
            parser_version=parser_version,
            options_hash=options_hash,
            error_message=error_message,
            diagnostics=list(diagnostics or []),
        )

    def to_dict(self) -> dict[str, Any]:
        """The wire form: plain JSON-native data, no live objects.

        Written by hand, like every ``to_dict`` in
        :mod:`~paperwhisperer.ingestion.schema` and
        :mod:`~paperwhisperer.ingestion.quality`, and for the same reason.
        The result of a parse crosses a process boundary when
        :func:`~paperwhisperer.ingestion.limits.run_guarded` is enabled, and
        that channel is JSON. A generic ``dataclasses.asdict`` would work for
        the scalars but would recurse into ``document`` and ``quality`` as
        objects, so the nested types are asked for their own form — which is
        also what keeps this a description of *this* schema rather than a
        reflection of whatever the fields happen to be this week.
        """
        return {
            "status": self.status,
            "document": self.document.to_dict() if self.document is not None else None,
            "quality": self.quality.to_dict() if self.quality is not None else None,
            "raw_artifact_uri": self.raw_artifact_uri,
            "warnings": list(self.warnings),
            "duration_ms": self.duration_ms,
            "parser_name": self.parser_name,
            "cloud_outcome": self.cloud_outcome,
            "parser_version": self.parser_version,
            "options_hash": self.options_hash,
            "error_message": self.error_message,
            "diagnostics": [dict(item) for item in self.diagnostics],
            "contract_repairs": [dict(item) for item in self.contract_repairs],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ParseResult:
        """Rebuild from :meth:`to_dict`.

        Every field takes a default when the key is absent or null, because the
        counterpart of this function may be an older build: a field added since
        the payload was written has to be optional, or a stored result becomes
        unreadable the next time this dataclass grows. The nested payloads are
        handed to their own ``from_dict`` rather than reconstructed inline, so
        the schema-version gate in
        :meth:`~paperwhisperer.ingestion.schema.ParsedDocument.from_dict` still
        decides what counts as a readable document.

        :class:`~paperwhisperer.ingestion.quality.ParseQualityReport` is
        imported here rather than at module scope for the same reason the module
        does it in :func:`_attach_quality` — importing it eagerly would make
        ``quality`` a hard dependency of everything that imports the gateway.
        """
        from .quality import ParseQualityReport  # noqa: PLC0415 - see docstring

        document_payload = payload.get("document")
        quality_payload = payload.get("quality")
        return cls(
            status=str(payload.get("status") or PARSE_FAILED),
            document=(
                ParsedDocument.from_dict(document_payload)
                if isinstance(document_payload, dict)
                else None
            ),
            quality=(
                ParseQualityReport.from_dict(quality_payload)
                if isinstance(quality_payload, dict)
                else None
            ),
            raw_artifact_uri=str(payload.get("raw_artifact_uri") or ""),
            warnings=[str(item) for item in payload.get("warnings") or []],
            duration_ms=int(payload.get("duration_ms") or 0),
            parser_name=str(payload.get("parser_name") or ""),
            cloud_outcome=str(payload.get("cloud_outcome") or ""),
            parser_version=str(payload.get("parser_version") or ""),
            options_hash=str(payload.get("options_hash") or ""),
            error_message=str(payload.get("error_message") or ""),
            diagnostics=[dict(item) for item in payload.get("diagnostics") or []],
            contract_repairs=[dict(item) for item in payload.get("contract_repairs") or []],
        )


# --------------------------------------------------------------------------
# Error mapping
# --------------------------------------------------------------------------


def describe_timeout(parser_name: str, timeout_seconds: float) -> str:
    return (
        f"解析器 {parser_name} 解析超时（超过 {timeout_seconds:g} 秒）。"
        "可尝试换用更轻量的解析器、提高 PARSER_TIMEOUT_SECONDS，或对扫描件启用 OCR。"
    )


def describe_parser_error(
    exc: BaseException,
    *,
    parser_name: str,
    timeout_seconds: float,
) -> str:
    """Translate anything a backend can throw into one user-facing sentence.

    The ordering encodes which diagnosis is more useful, not which type is
    more specific. A ``ValueError`` from the legacy loader is already a
    finished Chinese sentence written for a human ("PDF 读取失败: ..."), so it
    passes through untouched; re-wrapping it would only bury the cause. A bare
    ``Exception`` is the opposite case and gets the parser's name attached,
    because an unlabelled stack trace in a log is the only clue anyone has.
    """
    if isinstance(exc, (ParserTimeout, concurrent.futures.TimeoutError)):
        return describe_timeout(parser_name, timeout_seconds)
    if isinstance(exc, ParserError):
        detail = str(exc).strip()
        return detail or f"解析器 {parser_name} 报告解析失败，但未提供具体原因。"
    if isinstance(exc, FileNotFoundError):
        return f"找不到待解析的文件: {exc.filename or exc}。"
    if isinstance(exc, PermissionError):
        return "没有读取该文件的权限，文件可能受密码保护或被其他程序占用。"
    if isinstance(exc, IsADirectoryError):
        return "解析目标是一个目录，而不是文件。"
    if isinstance(exc, MemoryError):
        return "解析文档时内存不足，请尝试拆分文件或关闭其他占用内存的程序后重试。"
    if isinstance(exc, ValueError):
        detail = str(exc).strip()
        return detail or f"解析器 {parser_name} 认为该文件无法解析。"
    if isinstance(exc, OSError):
        return f"读取文件时发生系统错误: {exc}。"
    detail = str(exc).strip()
    suffix = f": {detail}" if detail else "，且未提供具体原因。"
    return f"解析器 {parser_name} 解析文档时发生未预期的错误{suffix}"


def media_type_for_path(file_path: str) -> str:
    """Best-effort media type for a path.

    Extensions first because they are what the upload gate already trusted;
    ``mimetypes`` is the fallback for names the gateway has no table entry for.
    """
    extension = os.path.splitext(str(file_path))[1].lower()
    known = MEDIA_TYPE_BY_EXTENSION.get(extension)
    if known:
        return known
    guessed, _ = mimetypes.guess_type(str(file_path))
    return guessed or "application/octet-stream"


# --------------------------------------------------------------------------
# Adapter-output validation
# --------------------------------------------------------------------------

#: How many distinct violations one failure message will spell out. One is a
#: bug report; five hundred is a wall of text a user cannot act on. The count
#: that follows them is what tells them it is five hundred.
_MAX_REPORTED_VIOLATIONS = 3

#: The locator kinds that promise a position, mapped to the field carrying it.
#: Mirrors ``_REQUIRED_FIELD_BY_KIND`` in :mod:`contracts`; written out here
#: because this module decides what to *repair* rather than what to accept, and
#: the two rules are deliberately not the same thing.
_POSITION_FIELD_BY_KIND = {
    "pdf_page": "page",
    "slide": "slide",
    "sheet_range": "sheet",
    "docx_element": "paragraph",
    "txt_offset": "offset_start",
}


def _describe_violation(error: Mapping[str, Any]) -> str:
    """Render one Pydantic error as ``field path: reason``.

    The field path is the part that matters to whoever has to fix this. A
    Pydantic error carries it as a tuple — ``("elements", 3, "locator")`` —
    which is precise and unreadable, so it is rendered the way the schema is
    navigated in Python: ``elements[3].locator``. A bare ``locator`` in a
    thousand-element document is not a diagnosis; an index is.
    """
    parts: list[str] = []
    for step in error.get("loc") or ():
        if isinstance(step, int):
            parts.append(f"[{step}]")
        elif parts:
            parts.append(f".{step}")
        else:
            parts.append(str(step))
    path = "".join(parts) or "(document)"
    return f"{path}: {error.get('msg', 'invalid value')}"


def describe_contract_violations(
    parser_name: str,
    exc: ValidationError,
) -> str:
    """Turn a ``ValidationError`` into the one Chinese sentence a user reads.

    The message names the parser, the offending field, and what was wrong with
    it, because the three answers to "why did my upload fail" are "which
    backend", "which part" and "how", and a bare "解析失败" gives none of them.
    Naming the parser is also what makes the message actionable: a reader who
    sees the docling backend named knows the failure is not about their file.
    """
    errors = exc.errors()
    shown = [_describe_violation(error) for error in errors[:_MAX_REPORTED_VIOLATIONS]]
    remainder = len(errors) - len(shown)
    detail = "；".join(shown)
    if remainder > 0:
        detail += f"；另有 {remainder} 处同类问题"
    return (
        f"解析器 {parser_name} 返回的文档结构不符合规范，已拒绝该结果：{detail}。"
        "请更换解析后端重试，或检查该文件是否已损坏。"
    )


def describe_unexpected_validation_error(parser_name: str, exc: BaseException) -> str:
    """The message for a *validator* that broke, as opposed to a bad document.

    Distinct wording on purpose. A ``ValidationError`` means the adapter
    produced something wrong and the refusal is the correct outcome. Anything
    else means the check itself failed, which is a bug in this codebase and
    must not be reported to the user as though their file were at fault — the
    traceback in the log is what gets it fixed.
    """
    return (
        f"解析器 {parser_name} 返回的文档无法完成结构校验（校验器内部错误："
        f"{type(exc).__name__}: {exc}），已拒绝该结果。"
    )


def _iter_locators(document: ParsedDocument):
    """Yield ``(where, get_locator, set_locator)`` for every locator in a document.

    A getter/setter pair rather than the locator itself, because the repair
    *replaces* a locator rather than mutating it: the contract forbids an
    ``unknown`` locator from carrying a ``bbox``, and a box with no page cannot
    be rendered anyway — there is nothing to anchor it to. Keeping one would
    trade a visible absence for an invisible one.

    Units are walked alongside elements because ``build_units`` copies the
    first member's locator into the unit, so a document that has been through
    it carries the same defect twice. Fixing only the elements would leave a
    ``pdf_page`` with no page sitting in ``units`` where the contract would
    still find it.
    """
    for index, element in enumerate(document.elements):
        yield (
            f"elements[{index}]",
            lambda e=element: e.locator,
            lambda new, e=element: setattr(e, "locator", new),
        )
    for index, unit in enumerate(document.units):
        yield (
            f"units[{index}]",
            lambda u=unit: u.locator,
            lambda new, u=unit: setattr(u, "locator", new),
        )


def _repair_known_locator_defects(
    document: ParsedDocument,
) -> list[dict[str, Any]]:
    """Downgrade a locator that names a position it does not have.

    One defect is repaired, and it is a producer bug:
    ``adapters/mineru_cloud.py`` constructs
    ``SourceLocator(kind="pdf_page", page=page, ...)`` where ``page`` is
    ``None`` whenever MinerU's ``content_list`` entry carries no ``page_idx``.
    A ``pdf_page`` with no page is a locator that claims to point at a page and
    points nowhere, so :class:`SourceLocatorContract` refuses it — correctly,
    and the gateway must not weaken that to accommodate a bug.

    But refusing it also means every MinerU document containing one unpaged
    element is thrown away entirely, which is a worse outcome than the defect:
    the rest of that document parsed fine, its tables have grids, and its
    locators resolve. So the locator is rewritten to ``kind="unknown"``, which
    is the honest spelling and the one the contract's own error message names.
    Nothing is fabricated — ``unknown`` is what ``is_resolvable`` already
    reported for this locator, so no consumer's behaviour changes, and the
    affected quote stops being presented as a page citation.

    **Every repair is recorded**, on ``result.warnings`` and in
    ``diagnostics``, because a silent repair is indistinguishable from a
    correct parse. A reader who sees a citation with no page can then connect
    it to a warning saying the backend lost the page index, instead of
    assuming the paper has no page numbers.

    **The producer should still be fixed.** The one-line change in
    ``normalize_mineru_payload`` is to spell the honest kind at the source::

        kind="pdf_page" if page else "unknown",

    until then this method is the compatibility path, and it should stop firing
    the day that lands.

    What is *not* repaired, deliberately: a page number that is present but
    wrong. There is no honest value to substitute for it, and a document whose
    pages are misattributed must fail rather than be quietly downgraded.
    """
    repairs: list[dict[str, Any]] = []
    for where, get_locator, set_locator in _iter_locators(document):
        locator = get_locator()
        field = _POSITION_FIELD_BY_KIND.get(locator.kind)
        if field is None:
            continue
        value = getattr(locator, field, None)
        if value is not None and not (isinstance(value, str) and not value.strip()):
            continue
        original_kind = locator.kind
        dropped: list[str] = []
        if locator.bbox is not None:
            dropped.append("bbox")
        if locator.cell_range is not None:
            dropped.append("cell_range")
        # The heading path survives: it is not positional, it still resolves,
        # and the contract permits it on an ``unknown`` locator. It is also the
        # only thing left that tells a reader which section a quote came from.
        set_locator(SourceLocator(kind="unknown", heading_path=tuple(locator.heading_path)))
        repairs.append({
            "where": where,
            "from_kind": original_kind,
            "field": field,
            "dropped": dropped,
        })
    return repairs


def compute_file_sha256(file_path: str, chunk_size: int = 1 << 20) -> str:
    """Content hash of a file, streamed.

    The parse cache is keyed on this (T0.7). It is computed here rather than in
    the orchestrator so that every caller hashing a file for a cache key uses
    one implementation, and so the streaming choice is made once: a 16 MB
    upload read whole is fine, but the same code will later be asked about
    files that are not.
    """
    digest = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------


def _thread_executor() -> ParserExecutor:
    """One worker per parse attempt.

    Single worker on purpose: the router already sequences attempts, so a pool
    would only add threads that sit idle while a parser is being timed out.
    """
    return concurrent.futures.ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="pw-parser",
    )


class ParserRouter:
    """Chooses a parser, times it, maps its failures, and records what happened.

    The routing policy is intentionally dumb. It walks a fixed order and takes
    the first result that is usable, which means it never spends a paid cloud
    call to upgrade a document it already has. "Compare the candidates and take
    the best whole document" is a policy with cost and privacy consequences, so
    it belongs to the orchestrator (T0.7), which can see the quality reports
    and the user's cloud-consent state. The router's job is to make every
    attempt uniform enough for that comparison to be meaningful.
    """

    def __init__(
        self,
        parsers: list[DocumentParser] | None = None,
        *,
        options: ParseOptions | None = None,
        executor_factory: Callable[[], ParserExecutor] = _thread_executor,
        clock: Callable[[], float] = time.monotonic,
        legacy_parser_name: str = LEGACY_PARSER_NAME,
    ) -> None:
        """
        :param parsers: candidates in *registration* order. Registration order
            is only a tiebreak; the effective order comes from configuration
            (see :meth:`_ordered_candidates`).
        :param options: default options for calls that do not pass their own.
        :param executor_factory: how a parse attempt gets a worker. Injectable
            so tests can drive the timeout path deterministically instead of
            racing a real thread against a real clock.
        :param clock: monotonic clock for durations, injectable for the same
            reason.
        :param legacy_parser_name: which parser the rollback flag selects.
        """
        self.parsers: list[DocumentParser] = list(
            parsers if parsers is not None else build_default_parsers()
        )
        self.options = options or ParseOptions()
        self.executor_factory = executor_factory
        self.clock = clock
        self.legacy_parser_name = legacy_parser_name

    @property
    def parser_names(self) -> tuple[str, ...]:
        """Registered parser names, in registration order."""
        return tuple(getattr(parser, "name", "") for parser in self.parsers)

    # -- ordering ----------------------------------------------------------

    def _configured_order(self) -> tuple[str, ...]:
        """Parser names in the order the environment asks for.

        Read fresh on every call: a long-lived process must see an edited
        ``.env``-derived value without a restart, and the alternative is a
        router whose behaviour silently disagrees with its own configuration.
        """
        names: list[str] = []
        primary = str(getattr(config, "PARSER_PRIMARY", "") or "").strip().lower()
        if primary:
            names.append(primary)
        for name in getattr(config, "PARSER_FALLBACK_CHAIN", ()) or ():
            cleaned = str(name).strip().lower()
            if cleaned and cleaned not in names:
                names.append(cleaned)
        return tuple(names)

    def _ordered_candidates(self, media_type: str = "") -> list[DocumentParser]:
        """Apply the ``PARSER_BACKEND`` feature flag to the registry.

        ``legacy`` collapses the registry to the legacy adapter — the rollback
        is meant to be total, so a half-rolled-back state where the new backend
        still gets a turn is not a state worth supporting. ``gateway`` follows
        ``PARSER_PRIMARY`` then ``PARSER_FALLBACK_CHAIN``, and appends anything
        registered but unmentioned at the end: a parser injected by a caller or
        a test should still be reachable, just never ahead of the configured
        preference.

        ``media_type`` promotes one parser for one format. It exists because
        the measured routing decision cannot be expressed as a single order:
        docling returns a DOCX *successfully* with zero resolvable locators
        (``docs/PARSER_BAKE_OFF.md`` §6), so a fallback that triggers on
        failure never fires and every DOCX citation is unresolvable while
        looking perfectly healthy. Ordering alone cannot fix that, because
        nothing failed.
        """
        backend = _current_backend()
        if backend == "legacy":
            return [
                parser
                for parser in self.parsers
                if getattr(parser, "name", "") == self.legacy_parser_name
            ]
        ordered: list[DocumentParser] = []
        for name in self._configured_order():
            for parser in self.parsers:
                if getattr(parser, "name", "") == name and parser not in ordered:
                    ordered.append(parser)
        for parser in self.parsers:
            if parser not in ordered:
                ordered.append(parser)

        # The per-format fork. Applied LAST so it can only ever promote a
        # parser the operator has already made available -- it cannot introduce
        # one the configuration excludes, so it never becomes a way to smuggle
        # an uninstalled parser into the chain.
        # Read defensively: this is a mapping, but a test may substitute a
        # string for it, and a routing decision that raises on a malformed
        # setting would turn a configuration typo into a total parse outage.
        by_media = getattr(config, "PARSER_PRIMARY_BY_MEDIA_TYPE", {}) or {}
        preferred = ""
        if isinstance(by_media, dict):
            preferred = str(
                by_media.get(str(media_type or "").strip().lower(), "")
            ).strip().lower()
        if preferred:
            for index, parser in enumerate(ordered):
                if getattr(parser, "name", "") == preferred and index:
                    ordered.insert(0, ordered.pop(index))
                    break
        return ordered

    # -- execution ---------------------------------------------------------

    def parse(
        self,
        file_path: str,
        media_type: str | None = None,
        *,
        options: ParseOptions | None = None,
    ) -> ParseResult:
        """Parse ``file_path``, degrading through the chain rather than raising.

        Returns the first usable result. A ``partial`` result ends the walk
        too: continuing past it would mean paying a second backend — possibly a
        cloud one — to improve a document the caller can already use, and
        deciding that trade-off is the orchestrator's job, not the router's.

        When every candidate fails, the *last* failure is returned. The last one
        is kept because it is the most expensive attempt, and its message is
        the one most likely to describe the actual blocker; earlier candidates
        are still visible in ``diagnostics``.
        """
        effective = options or self.options
        options_hash = effective.compute_options_hash()
        resolved_media_type = media_type or media_type_for_path(file_path)
        diagnostics: list[dict[str, Any]] = []

        candidates = self._ordered_candidates(resolved_media_type)
        if not candidates:
            return ParseResult.failure(
                f"当前没有可用的文档解析器（media_type={resolved_media_type}），"
                "请检查 PARSER_BACKEND / PARSER_PRIMARY 配置。",
                options_hash=options_hash,
                diagnostics=diagnostics,
            )

        last_failure: ParseResult | None = None
        for parser in candidates:
            name = str(getattr(parser, "name", "") or "unknown")
            version = str(getattr(parser, "version", "") or "")

            if not self._supports(parser, resolved_media_type):
                diagnostics.append({
                    "parser": name,
                    "version": version,
                    "options_hash": options_hash,
                    "attempted": False,
                    "skipped_reason": "unsupported_media_type",
                    "media_type": resolved_media_type,
                })
                continue

            result, elapsed_ms = self._run_attempt(parser, file_path, effective)
            attempt = {
                "parser": name,
                "version": version,
                "options_hash": options_hash,
                "attempted": True,
                "skipped_reason": "",
                "status": result.status,
                "duration_ms": elapsed_ms,
                "error": result.error_message,
            }
            # Repairs are recorded per attempt rather than only on the winning
            # result, so a chain where an early backend was repaired and a later
            # one succeeded still shows what the first one got wrong. That is
            # the difference between "the MinerU backend lost page indices" and
            # a mystery.
            if result.contract_repairs:
                attempt["contract_repairs"] = list(result.contract_repairs)
            diagnostics.append(attempt)

            if result.is_usable:
                self._attach_quality(result)
                if result.status == PARSE_PARTIAL:
                    result.warnings.append(
                        f"解析器 {name} 返回了不完整结果，已降级使用；请留意该文档的引用定位精度。"
                    )
                result.diagnostics = diagnostics
                return result
            last_failure = result

        if last_failure is None:
            # Every candidate was skipped, which means none of them claims this
            # media type. That is a coverage gap, not a parse failure, and the
            # message says so — the fix is a different adapter.
            return ParseResult.failure(
                f"没有解析器支持该文件类型（media_type={resolved_media_type}）。",
                options_hash=options_hash,
                diagnostics=diagnostics,
            )
        last_failure.diagnostics = diagnostics
        return last_failure

    def _supports(self, parser: DocumentParser, media_type: str) -> bool:
        """Ask a parser whether it handles this type, treating a bad answer as no.

        ``supports`` is a claim about capability, so a parser that raises while
        making it has not made it. Swallowing the error keeps a broken optional
        dependency from taking down the whole chain.
        """
        try:
            return bool(parser.supports(media_type))
        except Exception as exc:  # noqa: BLE001 - see docstring
            logger.warning("parser %r failed its supports() check: %s", parser, exc)
            return False

    def _run_attempt(
        self,
        parser: DocumentParser,
        file_path: str,
        options: ParseOptions,
    ) -> tuple[ParseResult, int]:
        """Run one parser under a timeout and normalise whatever comes back.

        The executor is created and shut down per attempt rather than shared.
        A shared pool would have to be shut down by the caller, and a caller
        that forgets leaks a thread per document; a per-attempt executor is
        slightly more expensive and cannot leak.
        """
        name = str(getattr(parser, "name", "") or "unknown")
        version = str(getattr(parser, "version", "") or "")
        options_hash = options.compute_options_hash()
        # ``None`` means wait forever, matching the socket convention. A zero
        # or negative budget is almost always an unset config value rather than
        # a request to give up instantly.
        budget = options.timeout_seconds if options.timeout_seconds > 0 else None

        started = self.clock()
        executor = self.executor_factory()
        try:
            future = executor.submit(parser.parse, file_path, options)
        except Exception as exc:  # noqa: BLE001 - executor refused the work
            executor.shutdown(wait=False, cancel_futures=True)
            elapsed_ms = self._elapsed_ms(started)
            logger.warning("parser %r could not be scheduled: %s", name, exc)
            return (
                ParseResult.failure(
                    describe_parser_error(
                        exc,
                        parser_name=name,
                        timeout_seconds=budget or 0,
                    ),
                    parser_name=name,
                    parser_version=version,
                    options_hash=options_hash,
                    duration_ms=elapsed_ms,
                ),
                elapsed_ms,
            )

        try:
            raw = future.result(timeout=budget)
        except concurrent.futures.TimeoutError:
            future.cancel()
            elapsed_ms = self._elapsed_ms(started)
            logger.warning("parser %r timed out after %ss", name, budget)
            return (
                ParseResult.failure(
                    describe_timeout(name, budget or 0),
                    parser_name=name,
                    parser_version=version,
                    options_hash=options_hash,
                    duration_ms=elapsed_ms,
                ),
                elapsed_ms,
            )
        except Exception as exc:  # noqa: BLE001 - the whole point of this layer
            elapsed_ms = self._elapsed_ms(started)
            message = describe_parser_error(
                exc,
                parser_name=name,
                timeout_seconds=budget or 0,
            )
            # Logged at warning with the original exception intact: the mapping
            # below is written for the user, and without the traceback a bug in
            # an adapter would be indistinguishable from a bad file.
            logger.warning("parser %r failed: %s", name, exc, exc_info=True)
            return (
                ParseResult.failure(
                    message,
                    parser_name=name,
                    parser_version=version,
                    options_hash=options_hash,
                    duration_ms=elapsed_ms,
                ),
                elapsed_ms,
            )
        finally:
            # ``wait=False`` because a timed-out call is still running and
            # joining it would defeat the timeout entirely.
            executor.shutdown(wait=False, cancel_futures=True)

        elapsed_ms = self._elapsed_ms(started)
        return self._normalize(raw, parser, options_hash, elapsed_ms), elapsed_ms

    def _normalize(
        self,
        raw: Any,
        parser: DocumentParser,
        options_hash: str,
        elapsed_ms: int,
    ) -> ParseResult:
        """Stamp provenance onto a parser's result, or reject it.

        A parser returning something other than a ``ParseResult`` is a bug in
        that parser, and it is reported as a failure rather than coerced: a
        duck-typed value that happened to have the right attributes would let a
        malformed document through the rest of the pipeline.
        """
        name = str(getattr(parser, "name", "") or "unknown")
        version = str(getattr(parser, "version", "") or "")
        if not isinstance(raw, ParseResult):
            return ParseResult.failure(
                f"解析器 {name} 返回了非 ParseResult 的结果（{type(raw).__name__}），"
                "网关无法处理。",
                parser_name=name,
                parser_version=version,
                options_hash=options_hash,
                duration_ms=elapsed_ms,
            )

        raw.parser_name = raw.parser_name or name
        raw.parser_version = raw.parser_version or version
        raw.options_hash = raw.options_hash or options_hash
        raw.duration_ms = raw.duration_ms or elapsed_ms
        if raw.is_usable and raw.document is None:
            # A usable result with nothing to show for it is the one combination
            # that would silently produce an empty analysis, and it is checked
            # through ``is_usable`` rather than against ``PARSE_SUCCESS`` alone:
            # a "partial" that dropped the entire document is the same empty
            # result with a friendlier label, and the caller must not have to
            # check both the status and the document to notice.
            return ParseResult.failure(
                f"解析器 {name} 报告解析成功但没有返回任何内容。",
                parser_name=name,
                parser_version=version,
                options_hash=options_hash,
                duration_ms=elapsed_ms,
                warnings=raw.warnings,
            )
        if raw.is_usable:
            return self._validate_document(raw, name, version, options_hash, elapsed_ms)
        return raw

    def _validate_document(
        self,
        raw: ParseResult,
        parser_name: str,
        parser_version: str,
        options_hash: str,
        elapsed_ms: int,
    ) -> ParseResult:
        """Check an adapter's document against the contracts, or repair and pass.

        This is the boundary the contracts module was written for: adapter
        output entering the gateway. Everything downstream reads the
        :class:`ParsedDocument`, and a document whose locators are subtly wrong
        produces citations that look correct and are not, which no later stage
        can detect because every later stage trusts the locator.

        The order is repair-then-validate and it matters. The repair is narrow
        enough to be an honest downgrade rather than a guess, so it runs first
        and its outcome is recorded; the validation that follows is strict and
        has no repair path at all, so anything the repair did not address — a
        table with no grid, a locator carrying another kind's field, a
        ``pdf_page`` pointing at page 0 — is refused outright.

        **Validation runs on ``document.to_dict()`` rather than through
        ``from_dataclass``,** and the reason is the quality of the resulting
        diagnosis. ``from_dataclass`` constructs each child contract in Python,
        so a ``ValidationError`` raised inside a locator propagates with a
        location of ``("page",)`` — the field, with no indication of *which* of
        five thousand elements carried it. Validating the serialized payload
        lets Pydantic walk the tree itself and report ``("elements", 4187,
        "locator")``, which is the difference between a diagnosis and a shrug.
        It is also marginally faster, and it has the side benefit of checking
        the exact bytes the parse cache is about to be handed.

        Both failure paths return a result rather than raising. That is the
        router's standing guarantee, and a contract violation does not earn an
        exception: it is a worse parse, so the chain moves on to the next
        candidate exactly as it would for a parser that raised, and the user
        gets the reason the last backend gave them.
        """
        document = raw.document
        if document is None:  # pragma: no cover - the caller guarantees this
            # Unreachable: ``_normalize`` returns a failure for a usable result
            # with no document before ever calling in here. Kept as a real
            # check rather than an ``assert`` because assertions vanish under
            # ``python -O``, and the next line would then be an
            # ``AttributeError`` escaping a module whose whole promise is that
            # it does not raise.
            return raw

        # Imported here rather than at module scope, and the reason is not the
        # usual optional-dependency one. Importing ``paperwhisperer.ingestion.schema``
        # runs this package's ``__init__``, which imports the gateway — so a
        # module-level ``from pydantic import ...`` or ``from .contracts import ...``
        # would pull Pydantic into every process that merely touches the
        # canonical dataclasses, and the contract that the schema half of this
        # split is importable without Pydantic would fail. The wiring is still
        # unconditional: this is a function-local import, not a guarded one, so
        # there is no path where validation is skipped.
        from pydantic import ValidationError  # noqa: PLC0415 - see above

        from .contracts import ParsedDocumentContract  # noqa: PLC0415 - see above

        repairs = _repair_known_locator_defects(document)
        for repair in repairs:
            message = (
                f"解析器 {parser_name} 有 {repair['where']} 的定位缺少 "
                f"{repair['field']}，已降级为不可定位（kind='unknown'）；"
                "该处的引用将不再指向具体位置。"
            )
            if repair["dropped"]:
                message += f" 同时丢弃了无法定位的字段：{', '.join(repair['dropped'])}。"
            raw.warnings.append(message)
            logger.warning("gateway repaired %s: %s", repair["where"], repair)

        try:
            ParsedDocumentContract.model_validate(document.to_dict())
        except ValidationError as exc:
            logger.warning(
                "parser %r produced a document that failed contract validation: %s",
                parser_name,
                exc,
            )
            return ParseResult.failure(
                describe_contract_violations(parser_name, exc),
                parser_name=parser_name,
                parser_version=parser_version,
                options_hash=options_hash,
                duration_ms=elapsed_ms,
                # The adapter's own warnings survive the refusal: a backend that
                # said "some pages failed" and then returned a malformed
                # document has still told us something true, and dropping it
                # would make the diagnosis strictly worse.
                warnings=raw.warnings,
            )
        except Exception as exc:  # noqa: BLE001 - the router never raises
            # A ``ValidationError`` means the document is wrong. Anything else
            # means the *checker* is wrong, which is a bug here and not
            # something to blame on the user's file — so the wording says so,
            # and the traceback goes to the log where it can be fixed.
            logger.exception("contract validation itself failed for %r", parser_name)
            return ParseResult.failure(
                describe_unexpected_validation_error(parser_name, exc),
                parser_name=parser_name,
                parser_version=parser_version,
                options_hash=options_hash,
                duration_ms=elapsed_ms,
                warnings=raw.warnings,
            )

        raw.contract_repairs = repairs
        return raw

    def _attach_quality(self, result: ParseResult) -> None:
        """Run the quality gate (T0.6) over a usable result and record the verdict.

        The gateway is the only place every parse passes through, so this is
        where the report gets attached; a caller that forgot to ask for it
        still gets one, and no two callers disagree about whether a gate ran.

        **The verdict is attached, not enforced.** A failing report does not
        change ``status``, because "this document is bad enough to reject"
        is a policy with cost and privacy consequences — the answer might be
        re-parsing with OCR locally, or paying a cloud backend, or telling
        the user the scan came out badly — and all three belong to the
        orchestrator (T0.7), which can see the consent state. A gateway that
        silently turned a readable scan into a failure would be throwing away
        exactly the document the fallback chain exists to rescue.

        Both the import and the call are guarded. The quality module is a
        dependency of the *gate*, not of the parse: a document that parsed
        cleanly must still parse cleanly on a build where the gate could not
        import, and a quality module that raises is a bug in the gate, not a
        reason to lose the document it was about to judge.
        """
        if result.document is None or result.quality is not None:
            return
        try:
            from .quality import evaluate_quality  # noqa: PLC0415 - see docstring

            report = evaluate_quality(result.document)
        except Exception:  # noqa: BLE001 - see docstring
            logger.warning(
                "quality gate could not evaluate the parse from %r",
                result.parser_name,
                exc_info=True,
            )
            return
        result.quality = report

    def _elapsed_ms(self, started: float) -> int:
        return max(0, int(round((self.clock() - started) * 1000)))


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


def _current_backend() -> str:
    """The active ``PARSER_BACKEND`` value, normalized and read live."""
    raw = getattr(config, "PARSER_BACKEND", "legacy")
    normalizer = getattr(config, "normalize_parser_backend", None)
    if callable(normalizer):
        return str(normalizer(raw))
    text = str(raw or "").strip().lower()
    return text if text in {"legacy", "gateway"} else "legacy"


def _module_available(dotted_name: str) -> bool:
    """Whether ``dotted_name`` could be imported, without importing it.

    ``find_spec`` rather than a real import because the answer is needed for
    docling, whose import pulls in torch: a registry that had to import the
    dependency to decide whether the dependency exists would make "is docling
    installed?" the most expensive question in the process.
    """
    try:
        return importlib.util.find_spec(dotted_name) is not None
    except (ImportError, ValueError):  # pragma: no cover - broken parent package
        return False


def _maybe_docling_parser() -> DocumentParser | None:
    """The docling adapter, or ``None`` when this build cannot run it.

    The two execution modes are gated differently, and the difference is not
    cosmetic. ``inprocess`` imports docling in this process, so without the
    package installed the adapter is dead weight and would only add a skipped
    attempt to every document's diagnostics. ``serve`` talks to a local
    docling-serve over loopback and never imports docling here, so a build
    without the package is still a build that can use it.

    Construction is guarded as well as the import: the adapter resolves its
    configuration in ``__init__``, and a build where that configuration is
    half-landed must still be able to parse a PDF with the legacy adapter.
    """
    try:
        from .adapters.docling_adapter import DoclingParser  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001 - optional dependency
        logger.info("docling adapter is not importable: %s", exc)
        return None
    try:
        parser = DoclingParser()
    except Exception as exc:  # noqa: BLE001 - optional dependency
        logger.warning("docling adapter could not be constructed: %s", exc)
        return None
    if str(getattr(parser, "mode", "")).strip().lower() != "serve" and not _module_available(
        "docling"
    ):
        logger.info("docling is not installed; the in-process adapter stays out of the registry.")
        return None
    return parser


def _maybe_mineru_parser() -> DocumentParser | None:
    """The MinerU cloud adapter, or ``None`` when cloud parsing is not on.

    Registration is gated on the same two facts the adapter itself refuses on
    — a token and ``DOCUMENT_PRIVACY_MODE=allow_cloud`` — so that a default
    deployment does not carry a parser that can only ever fail. Per-call
    consent is deliberately *not* checked here: consent is a property of one
    upload, and a registry that refused to exist would take the parser away
    from the requests that were allowed to use it.
    """
    if not _cloud_parsing_allowed():
        return None
    try:
        from .adapters.mineru_cloud import MineruCloudParser  # noqa: PLC0415

        return MineruCloudParser()
    except Exception as exc:  # noqa: BLE001 - optional dependency
        logger.warning("MinerU cloud adapter is unavailable: %s", exc)
        return None


def _cloud_parsing_allowed() -> bool:
    """Whether a cloud backend could run at all, judged before it is registered."""
    mode = str(getattr(config, "DOCUMENT_PRIVACY_MODE", "local_only") or "local_only")
    token = str(getattr(config, "MINERU_API_TOKEN", "") or "").strip()
    return mode.strip().lower() == "allow_cloud" and bool(token)


def build_default_parsers() -> list[DocumentParser]:
    """The parsers this build ships with, most preferred first.

    The legacy adapter is imported inside the function rather than at module
    scope: it reaches back into this module for the ``DocumentParser`` and
    ``ParseResult`` types, and a top-level import on both sides is a cycle.
    The cost is one dict lookup per router construction, which happens once
    per request at most.

    The order is preference, not correctness: the router re-orders by
    ``PARSER_PRIMARY`` anyway. It is last because it is the one parser that
    must work for the whole of Phase 0 — anything registered ahead of it can
    fail, and it is the reason a document still parses after they have.
    Optional adapters are appended ahead of it only when they can actually
    run here, so the default build of a machine without docling installed and
    without a cloud token registers exactly one parser, exactly as before.
    """
    from .adapters.legacy_loader import LegacyDocumentParser  # noqa: PLC0415

    parsers: list[DocumentParser] = []
    for candidate in (_maybe_docling_parser(), _maybe_mineru_parser()):
        if candidate is not None:
            parsers.append(candidate)
    parsers.append(LegacyDocumentParser())
    return parsers


def build_default_router(**kwargs: Any) -> ParserRouter:
    """A router over the shipped parsers.

    Exists so callers do not have to know what the registry happens to hold
    today — an optional adapter that is missing on this machine, or a cloud
    token that has not been configured, changes the registry and not the call
    sites.
    """
    return ParserRouter(build_default_parsers(), **kwargs)
