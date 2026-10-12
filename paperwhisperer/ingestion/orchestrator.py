"""Parse orchestration: route → local repair (force OCR) → cloud fallback.

The gateway (T0.3) answers "how do I run one parser and survive its failure".
This module answers the next question: *given that something went wrong, what
should I do about it, and am I allowed to?* The three decisions below are the
reason this is a separate layer rather than more logic in
:meth:`~paperwhisperer.ingestion.gateway.ParserRouter.parse`.

**Escalate in cost order, and stop as soon as the document is good enough.**
The chain is primary route → local repair (force OCR) → cloud fallback. It is
strictly ordered by cost: the local parsers are free, the cloud one is billed
per document and uploads the user's file to a third party. The orchestrator
therefore spends nothing it does not have to, and it re-checks quality after
*every* stage — including the stage that came back from a cache — because a
cached bad parse is still a bad parse, and the repair that fixes it is cached
under its own key.

**One document, one parser, one locator namespace. Never a page-level best
of.** When two results are compared here, the comparison picks a single whole
``ParsedDocument`` and discards the other. This is not a simplification, it is
the project's central invariant. The whole point of ``schema.py`` is that a
citation resolves to a page and a bounding box in the file the reader has in
front of them. A document stitched from page 1 of Docling and pages 2-9 of
MinerU has no single reading order (two parsers disagree about where a
two-column page splits into one flow), no single locator namespace (a
``pdf_page`` box from one adapter and a ``pdf_page`` box from another are not
comparable — different page geometry, different coordinate origin), and no
single provenance for any given sentence. A citation built on that document
would be unresolvable exactly where the user clicks it, which is the failure
mode the entire Evidence-first design exists to prevent. A slightly worse
document from one parser is recoverable; a chimera of two is not. The quality
report is what makes "slightly worse" measurable: it grades whole documents,
so the comparison is meaningful even though the documents are not merged.

**The privacy gate is consulted before any network call, and once is enough.**
``decide_cloud_fallback`` runs before the cloud parser is constructed, let
alone invoked, so a ``local_only`` deployment cannot reach the network even by
accident or by a future refactor that forgets a guard further down. The cloud
adapter re-checks the same gate on its own side; the duplicate is deliberate.
One gate in two places means a mistake has to be made twice to leak a
document, and the orchestrator's own check is the one that produces the
user-facing reason recorded in the attempt trail.

**The parse cache is what stops paying twice.** It is keyed on
``(sha256, parser_name, parser_version, options_hash)`` and it is consulted
*before* the parser is called, not after — a cache that only recorded what had
already been paid for would still pay again. ``options_hash`` is part of the
key precisely so that a force-OCR repair can never be served from the
non-OCR entry it was meant to replace, and ``parser_version`` is part of it
so that upgrading Docling does not serve the old layout indefinitely.

Everything external is injected: the router, the cloud parser, the cache, the
privacy-gate evaluator, the file hasher and the sleeper. A test can drive the
whole chain with no network, no docling, and no sleeping.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import re
import tempfile
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from paperwhisperer.core import config
from paperwhisperer.core.formats import canonical_media_type

from .gateway import (
    PARSE_FAILED,
    PARSE_PARTIAL,
    PARSE_SUCCESS,
    ParseOptions,
    ParseResult,
    ParserRouter,
    build_default_router,
    compute_file_sha256,
    media_type_for_path,
)
from .quality import (
    ParseQualityReport,
    QualityStatus,
    QualityThresholds,
    evaluate_quality,
)
from .schema import Element, ParsedDocument, SourceLocator

logger = logging.getLogger(__name__)

#: The three stages of the chain, in the only order they are ever allowed to
#: run. Exposed as a tuple so a caller iterating ``outcome.attempts`` can rely
#: on the sequence meaning "cheapest first".
STAGE_PRIMARY = "primary"
STAGE_FORCE_OCR = "force_ocr"
STAGE_CLOUD = "cloud"
STAGES = (STAGE_PRIMARY, STAGE_FORCE_OCR, STAGE_CLOUD)

#: ``ParseAttempt.status`` for a stage that never ran. Not one of the gateway's
#: three ``ParseResult`` statuses, and deliberately so: a skipped stage is
#: neither a success (nothing was parsed) nor a failure (nothing was wrong with
#: the document). Collapsing it into "success" would make a refused cloud
#: fallback look like a successful parse of nothing.
ATTEMPT_SKIPPED = "skipped"

#: Media types for which force-OCR retry is worth attempting. Only formats that
#: can physically carry a raster page qualify: asking a DOCX parser to OCR is
#: not a repair, it is a second full parse that is guaranteed to return the
#: same text it already returned.
OCR_CANDIDATE_MEDIA_TYPES = frozenset({
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "image/tiff",
    "image/bmp",
})

#: Bumped when the on-disk cache layout changes. An entry whose version does
#: not match is treated as a miss, so an old cache is never misread as a new
#: one — a stale hit here would mean serving a document the current parser
#: would never produce.
CACHE_SCHEMA_VERSION = "parse-cache/1"

#: Default cap on the in-process mirror of the cache. The directory is the
#: durable store; this dict only exists so a repeated parse inside one process
#: does not re-read a JSON file. Bounded because the process is long-lived and
#: unbounded growth here is a slow leak, not a feature.
DEFAULT_MEMORY_CACHE_ENTRIES = 64

#: Severity ordering for whole-document comparison. Lower wins. Quality is
#: consulted before anything numeric because a FAIL means a promise is broken
#: (unresolvable citations, missing pages) and no amount of extra text repairs
#: that.
_STATUS_RANK = {
    QualityStatus.PASS: 0,
    QualityStatus.WARN: 1,
    QualityStatus.FAIL: 2,
}

_SAFE_SEGMENT = re.compile(r"[^A-Za-z0-9._-]+")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_segment(value: str) -> str:
    """Make a cache-key component safe to put in a filename.

    Parser names and versions are code constants today, but the cache key also
    carries caller-supplied option hashes, and a slash in a filename would turn
    a write into a failure in a directory that may not exist. Collapsing to a
    safe alphabet keeps the worst case cosmetic.
    """
    cleaned = _SAFE_SEGMENT.sub("_", str(value or "")).strip("_")
    return cleaned or "unknown"


# --------------------------------------------------------------------------
# Cache key and entry
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ParseCacheKey:
    """The identity of a parse: same four values, same result, by construction.

    ``sha256`` is the document's bytes rather than its path, so re-uploading the
    same file under a different name is a hit — which is the common case when a
    user retries a failed upload. ``options_hash`` covers every knob that can
    change the output, including the timeout, so an attempt that gave up early
    is not confused with one that ran to completion.
    """

    sha256: str
    parser_name: str
    parser_version: str
    options_hash: str

    @property
    def identity(self) -> str:
        return "|".join((self.sha256, self.parser_name, self.parser_version, self.options_hash))

    def as_dict(self) -> dict[str, str]:
        return {
            "sha256": self.sha256,
            "parser_name": self.parser_name,
            "parser_version": self.parser_version,
            "options_hash": self.options_hash,
        }

    def filename(self) -> str:
        """A collision-free, filesystem-safe name for this key.

        A digest of the full identity is appended rather than relying on the
        readable parts being unique, because sanitising a parser name is lossy
        by design and two different keys must never share a file.
        """
        digest = hashlib.sha256(self.identity.encode("utf-8")).hexdigest()[:12]
        return (
            f"{_safe_segment(self.parser_name)}"
            f"_{_safe_segment(self.parser_version)}"
            f"_{_safe_segment(self.options_hash)}_{digest}.json"
        )


@dataclass(slots=True)
class CacheEntry:
    """One stored whole-document parse.

    Only usable results are stored. Caching a failure would mean a transient
    backend outage gets pinned to the document forever, and the repair that
    would have fixed it would never run.
    """

    document: ParsedDocument
    quality: ParseQualityReport | None = None
    status: str = PARSE_SUCCESS
    warnings: list[str] = field(default_factory=list)
    raw_artifact_uri: str = ""
    created_at: str = ""


class ParseCache:
    """Whole-document parse cache, on disk and optionally mirrored in memory.

    Write-then-read is not the pattern here: :meth:`get` is called *before* the
    parser runs, which is the only arrangement that satisfies "a cache hit must
    not pay twice" for a billed cloud parse.

    Durability is best-effort by design. A cache that cannot be written (a
    read-only deployment directory, a full disk) degrades to "no cache"; it
    never fails a parse. A parse the user has already paid for should still be
    returned to them.
    """

    def __init__(
        self,
        directory: str | os.PathLike[str] | None = None,
        *,
        enabled: bool = True,
        max_memory_entries: int = DEFAULT_MEMORY_CACHE_ENTRIES,
    ) -> None:
        self.directory = str(directory) if directory else ""
        self.enabled = bool(enabled)
        self.max_memory_entries = max(1, int(max_memory_entries))
        self._memory: OrderedDict[str, CacheEntry] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.writes = 0

    # -- keys --------------------------------------------------------------

    def key_for(
        self,
        sha256: str,
        parser_name: str,
        parser_version: str,
        options_hash: str,
    ) -> ParseCacheKey:
        return ParseCacheKey(
            sha256=str(sha256 or ""),
            parser_name=str(parser_name or ""),
            parser_version=str(parser_version or ""),
            options_hash=str(options_hash or ""),
        )

    def _path_for(self, key: ParseCacheKey) -> str:
        # Sharded on the first two hex characters of the content hash. A
        # directory of tens of thousands of JSON files is unpleasant to list
        # and unpleasant to delete; 256 shards is not.
        shard = (key.sha256 or "0")[:2] or "00"
        return os.path.join(self.directory, shard, key.filename())

    # -- reads and writes --------------------------------------------------

    def get(self, key: ParseCacheKey) -> CacheEntry | None:
        """Return the stored entry for ``key``, or ``None`` on any doubt.

        Every failure mode here is a miss, never an exception: a corrupt file,
        an entry written by an older layout, a document the schema has since
        renumbered. A miss costs one parse; a false hit costs a wrong document.
        """
        if not self.enabled:
            return None
        cached = self._memory.get(key.identity)
        if cached is not None:
            self._memory.move_to_end(key.identity)
            self.hits += 1
            return cached
        entry = self._read_disk(key)
        if entry is None:
            self.misses += 1
            return None
        self._remember(key.identity, entry)
        self.hits += 1
        return entry

    def put(self, key: ParseCacheKey, entry: CacheEntry) -> None:
        """Store ``entry``. Never raises."""
        if not self.enabled or entry.document is None:
            return
        entry.created_at = entry.created_at or _now_iso()
        self._remember(key.identity, entry)
        self._write_disk(key, entry)
        self.writes += 1

    def clear(self) -> None:
        """Drop the in-memory mirror. Does not touch the directory."""
        self._memory.clear()

    def stats(self) -> dict[str, int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "memory_entries": len(self._memory),
        }

    # -- internals ---------------------------------------------------------

    def _remember(self, identity: str, entry: CacheEntry) -> None:
        self._memory[identity] = entry
        self._memory.move_to_end(identity)
        while len(self._memory) > self.max_memory_entries:
            self._memory.popitem(last=False)

    def _read_disk(self, key: ParseCacheKey) -> CacheEntry | None:
        if not self.directory:
            return None
        path = self._path_for(key)
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            # A half-written or truncated file is a miss with a note, not a
            # crash: writes are atomic, so this is a real disk event, and the
            # only useful response is to re-parse and overwrite.
            logger.warning("parse cache entry unreadable (%s): %s", path, exc)
            return None
        if not isinstance(payload, dict) or payload.get("schema_version") != CACHE_SCHEMA_VERSION:
            return None
        try:
            document = ParsedDocument.from_dict(payload.get("document") or {})
            quality_payload = payload.get("quality")
            quality = (
                ParseQualityReport.from_dict(quality_payload)
                if isinstance(quality_payload, dict)
                else None
            )
        except Exception as exc:  # noqa: BLE001 - a stale entry must not break a parse
            logger.warning("parse cache entry could not be decoded (%s): %s", path, exc)
            return None
        return CacheEntry(
            document=document,
            quality=quality,
            status=str(payload.get("status") or PARSE_SUCCESS),
            warnings=[str(item) for item in payload.get("warnings") or []],
            raw_artifact_uri=str(payload.get("raw_artifact_uri") or ""),
            created_at=str(payload.get("created_at") or ""),
        )

    def _write_disk(self, key: ParseCacheKey, entry: CacheEntry) -> None:
        if not self.directory:
            return
        path = self._path_for(key)
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "key": key.as_dict(),
            "status": entry.status,
            "document": entry.document.to_dict(),
            "quality": entry.quality.to_dict() if entry.quality is not None else None,
            "warnings": list(entry.warnings),
            "raw_artifact_uri": entry.raw_artifact_uri,
            "created_at": entry.created_at or _now_iso(),
        }
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # Atomic replace: a reader in another process (or the same one,
            # after a crash) must see either the old entry or the new one,
            # never a truncated file. os.replace is atomic on POSIX and on
            # Windows for same-volume files. mkstemp + fdopen rather than
            # NamedTemporaryFile because the file has to outlive the ``with``
            # block long enough to be renamed.
            descriptor, temp_path = tempfile.mkstemp(
                dir=os.path.dirname(path), prefix=".tmp-", suffix=".json"
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, ensure_ascii=False)
                os.replace(temp_path, path)
            except BaseException:
                _silent_unlink(temp_path)
                raise
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("parse cache entry not written (%s): %s", path, exc)


def _silent_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:  # pragma: no cover - best effort cleanup
        pass


def build_default_parse_cache() -> ParseCache:
    """The cache this deployment's configuration asks for.

    ``PARSER_CACHE_ENABLED`` off means a fresh object with no directory, so
    every lookup is a miss. The directory prefers ``PARSER_CACHE_DIR`` and
    falls back to a ``parse_cache`` folder under the context directory, which
    is the folder the rest of the ingestion artefacts already live in.
    """
    enabled = bool(getattr(config, "PARSER_CACHE_ENABLED", True))
    directory = str(getattr(config, "PARSER_CACHE_DIR", "") or "").strip()
    if not directory:
        context_folder = str(getattr(config, "CONTEXT_FOLDER", "context") or "context")
        directory = os.path.join(context_folder, "parse_cache")
    return ParseCache(directory, enabled=enabled)


# --------------------------------------------------------------------------
# Privacy policy
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CloudFallbackDecision:
    """Whether this call may send the document off the machine, and why.

    ``reason_code`` is the machine-readable half and it distinguishes the two
    refusals a user has to be told apart: the *operator* forbade it
    (``privacy_mode_local_only``) or the *user* has not consented yet
    (``consent_not_granted``). Collapsing them into one "cloud parsing is off"
    message sends people to change the setting that was never the problem.

    The two fields are separate so that the attempt trail can be asserted on
    in tests, which is what the local-only release gate does.
    """

    allowed: bool
    reason_code: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {"allowed": self.allowed, "reason_code": self.reason_code, "message": self.message}


def evaluate_cloud_privacy_gate(settings: Any | None = None) -> CloudFallbackDecision:
    """Read the cloud adapter's own privacy gate and normalise it.

    The adapter owns the two-factor policy (deployment mode × user consent);
    re-deriving it here would give two places to change when the policy grows a
    third factor, and they would disagree. What this function adds is the
    import-time tolerance: the cloud adapter is an optional surface, and an
    orchestrator that cannot even import it must still be able to run the
    local chain.
    """
    try:
        from .adapters.mineru_cloud import evaluate_privacy_gate
    except Exception as exc:  # noqa: BLE001 - optional dependency surface
        logger.warning("cloud privacy gate unavailable: %s", exc)
        return CloudFallbackDecision(
            allowed=False,
            reason_code="cloud_adapter_unavailable",
            message=(
                "云端回退不可用：云端解析适配器未能加载，已跳过云端解析。"
                "如需云端回退，请检查 MinerU 相关依赖是否安装。"
            ),
        )
    gate = evaluate_privacy_gate(settings)
    return CloudFallbackDecision(
        allowed=bool(getattr(gate, "allowed", False)),
        reason_code=str(getattr(gate, "reason_code", "") or "unknown"),
        message=str(getattr(gate, "message", "") or "云端解析未启用。"),
    )


# --------------------------------------------------------------------------
# Attempt and outcome
# --------------------------------------------------------------------------


@dataclass(slots=True)
class ParseAttempt:
    """One stage of the chain, and what became of it.

    Every stage is recorded, including the ones that never ran. "Cloud fallback
    was refused because privacy mode is local_only" is a fact the user needs and
    an absent entry cannot express — an absence is indistinguishable from a bug
    in the orchestration.
    """

    stage: str
    parser_name: str = ""
    parser_version: str = ""
    options_hash: str = ""
    status: str = PARSE_FAILED
    attempted: bool = False
    skipped_reason: str = ""
    quality_status: str = ""
    duration_ms: int = 0
    error_message: str = ""
    cache_hit: bool = False
    #: Whether this stage was allowed to reach the network. Only the cloud
    #: stage can set it, and it is the field a zero-egress test asserts on.
    network: bool = False
    #: The policy verdict, when this stage was skipped by the privacy gate.
    decision: CloudFallbackDecision | None = None
    #: Per-parser detail forwarded from the gateway, so the trail shows which
    #: candidates the router walked and why one was skipped.
    diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "parser_name": self.parser_name,
            "parser_version": self.parser_version,
            "options_hash": self.options_hash,
            "status": self.status,
            "attempted": self.attempted,
            "skipped_reason": self.skipped_reason,
            "quality_status": self.quality_status,
            "duration_ms": self.duration_ms,
            "error_message": self.error_message,
            "cache_hit": self.cache_hit,
            "network": self.network,
            "decision": self.decision.to_dict() if self.decision is not None else None,
            "diagnostics": [dict(item) for item in self.diagnostics],
        }

    def describe(self) -> str:
        """A short Chinese clause naming this attempt and its outcome."""
        who = self.parser_name or "未知解析器"
        if not self.attempted:
            return f"{self.stage}：跳过（{self.skipped_reason or '未执行'}）"
        if self.status in {PARSE_SUCCESS, PARSE_PARTIAL}:
            verdict = f"成功（质量 {self.quality_status or '未评估'}）"
            if self.cache_hit:
                verdict += "，命中缓存"
            return f"{self.stage}：{who} {verdict}"
        return f"{self.stage}：{who} 失败（{self.error_message or '未知原因'}）"


@dataclass(slots=True)
class ParseOutcome:
    """Everything a caller needs to know about how one file was parsed.

    The last three fields exist for the P0 exit criterion "用户能明确知道文档被
    谁解析、是否经过云端". Answering that from a bare ``ParsedDocument`` is not
    possible — a document carries its parser name but not whether another
    parser was tried, whether the cloud was offered and refused, or whether the
    answer came off disk. So the outcome answers all four explicitly.
    """

    status: str
    document: ParsedDocument | None = None
    quality: ParseQualityReport | None = None
    attempts: list[ParseAttempt] = field(default_factory=list)
    #: The winning document came from the cache; no parser was invoked for it.
    cache_hit: bool = False
    #: Escalation actually happened: a stage beyond the primary produced the
    #: winning document, or at least ran.
    fallback_used: bool = False
    #: The cloud parser was invoked. Distinct from :attr:`used_cloud`, which
    #: means the cloud is *the* answer: a cloud attempt that lost to a local
    #: parse is money spent, and the UI should be able to say so.
    cloud_attempted: bool = False
    #: The winning parser's cloud outcome, or "" when the winner was local.
    #: See :attr:`gateway.ParseResult.cloud_outcome` for why it is not a
    #: diagnostics entry.
    cloud_outcome: str = ""
    #: The winning document was produced by the cloud parser, i.e. this
    #: document left the machine.
    used_cloud: bool = False
    warnings: list[str] = field(default_factory=list)
    error_message: str = ""
    sha256: str = ""
    media_type: str = ""
    raw_artifact_uri: str = ""
    #: The refusal that stopped the cloud stage, when it was stopped.
    cloud_refusal: CloudFallbackDecision | None = None

    @property
    def is_usable(self) -> bool:
        return self.status in {PARSE_SUCCESS, PARSE_PARTIAL}

    @property
    def parsed_by(self) -> str:
        """Which parser produced the winning document."""
        return str(getattr(self.document, "parser_name", "") or "")

    @property
    def touched_network(self) -> bool:
        """Whether any attempt was allowed to reach the network.

        Distinct from :attr:`used_cloud`: a cloud attempt that was made and then
        discarded still touched the network and still cost money. The
        local-only gate asserts on this, and it is the honest reading of "zero
        egress".
        """
        return any(attempt.network and attempt.attempted for attempt in self.attempts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "document": self.document.to_dict() if self.document is not None else None,
            "quality": self.quality.to_dict() if self.quality is not None else None,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "cache_hit": self.cache_hit,
            "fallback_used": self.fallback_used,
            "cloud_attempted": self.cloud_attempted,
            "used_cloud": self.used_cloud,
            "touched_network": self.touched_network,
            "warnings": list(self.warnings),
            "error_message": self.error_message,
            "sha256": self.sha256,
            "media_type": self.media_type,
            "parsed_by": self.parsed_by,
            "cloud_refusal": self.cloud_refusal.to_dict() if self.cloud_refusal else None,
        }

    def summary_line(self) -> str:
        """One Chinese sentence naming the parser and the cloud question.

        This is the line the UI shows next to a document. It is written to be
        unambiguous on its own: a user must never have to guess whether their
        file was uploaded somewhere.
        """
        if not self.is_usable:
            return self.error_message or "文档解析失败。"
        who = self.parsed_by or "未知解析器"
        where = "已上传至云端解析服务" if self.used_cloud else "未上传至任何云端服务"
        extra = "（结果来自缓存，本次未重新解析）" if self.cache_hit else ""
        return f"本文档由 {who} 解析，{where}{extra}。"


# --------------------------------------------------------------------------
# Cache-aware parser proxy
# --------------------------------------------------------------------------


class _CallLog:
    """Per-call scratch: the content hash and which lookups were hits.

    The file hash is computed once by the orchestrator and shared by every
    parser attempt for that call. Hashing the same 16 MB upload once per
    candidate would turn a cheap key into the most expensive part of a parse.
    """

    def __init__(self, sha256: str) -> None:
        self.sha256 = sha256
        self._lookups: dict[tuple[str, str], bool] = {}

    def note(self, parser_name: str, options_hash: str, *, hit: bool) -> None:
        self._lookups[(str(parser_name or ""), str(options_hash or ""))] = bool(hit)

    def was_cached(self, parser_name: str, options_hash: str) -> bool:
        return bool(self._lookups.get((str(parser_name or ""), str(options_hash or ""))))


class _CachingParser:
    """A parser wrapped so the cache is consulted *before* it runs.

    Implemented as a proxy rather than as a branch inside
    :class:`ParseCache` so that every caller of the gateway gets the cache for
    free: the router is handed proxies instead of parsers and cannot tell.

    A cache miss costs one hash lookup and one JSON read. That is a real cost
    on a large corpus of small files, and it is still the right trade: the
    alternative is paying a cloud provider per document to save a local file
    read.
    """

    def __init__(self, inner: Any, cache: ParseCache, log: _CallLog) -> None:
        self._inner = inner
        self._cache = cache
        self._log = log
        self.name = str(getattr(inner, "name", "") or "")
        self.version = str(getattr(inner, "version", "") or "")

    def __getattr__(self, item: str) -> Any:
        # Anything this proxy does not implement is the inner parser's business
        # (an adapter-specific knob, say), so it is forwarded rather than
        # hidden. ``_inner`` is set before anything can reach this path, so
        # there is no recursion to guard.
        if item in {"_inner", "_cache", "_log"}:
            raise AttributeError(item)
        return getattr(self._inner, item)

    def supports(self, media_type: str) -> bool:
        return bool(self._inner.supports(media_type))

    def parse(self, file_path: str, options: Any) -> ParseResult:
        options_hash = _options_hash(options)
        digest = self._log.sha256
        if not self._cache.enabled or not digest:
            # No hash means no key. Parsing anyway is correct; skipping the
            # cache is a small price for not guessing.
            return self._inner.parse(file_path, options)

        key = self._cache.key_for(digest, self.name, self.version, options_hash)
        entry = self._cache.get(key)
        if entry is not None:
            self._log.note(self.name, options_hash, hit=True)
            logger.info("parse cache hit for %s (%s)", self.name, key.filename())
            return _result_from_entry(entry, self.name, self.version, options_hash)

        result = self._inner.parse(file_path, options)
        self._log.note(self.name, options_hash, hit=False)
        if isinstance(result, ParseResult) and result.is_usable and result.document is not None:
            self._cache.put(
                key,
                CacheEntry(
                    document=result.document,
                    quality=result.quality,
                    status=result.status,
                    warnings=list(result.warnings),
                    raw_artifact_uri=result.raw_artifact_uri,
                ),
            )
        return result


def _result_from_entry(
    entry: CacheEntry,
    parser_name: str,
    parser_version: str,
    options_hash: str,
) -> ParseResult:
    """Rebuild a ``ParseResult`` from a cache entry.

    Provenance is restamped from the key rather than copied out of the stored
    document, so a result can never claim a provenance its cache key does not
    support. The document is restored with the stored provenance intact, which
    is what keeps "which parser produced this" answerable after a restart.
    """
    status = entry.status if entry.status in {PARSE_SUCCESS, PARSE_PARTIAL} else PARSE_SUCCESS
    return ParseResult(
        status=status,
        document=entry.document,
        quality=entry.quality,
        raw_artifact_uri=entry.raw_artifact_uri,
        warnings=list(entry.warnings),
        duration_ms=0,
        parser_name=parser_name,
        parser_version=parser_version,
        options_hash=options_hash,
    )


def _options_hash(options: Any) -> str:
    """The option identity, tolerating a parser called with something else."""
    hasher = getattr(options, "compute_options_hash", None)
    if callable(hasher):
        try:
            return str(hasher())
        except Exception as exc:  # noqa: BLE001 - a broken hasher must not parse-block
            logger.warning("options hash failed, cache key degraded: %s", exc)
            return ""
    return ""


# --------------------------------------------------------------------------
# Cloud bridge
# --------------------------------------------------------------------------


@dataclass(slots=True)
class CloudParseOptions:
    """The options object the cloud adapter reads duck-typed attributes from.

    The adapter takes ``source_filename`` / ``media_type`` / ``trigger_reason`` /
    ``operator`` off whatever it is handed rather than taking a ParseOptions, so
    it can be driven from the gateway's option type or from this. Building a
    dedicated object rather than overloading ``ParseOptions.extra`` keeps the
    two option shapes from drifting into each other.
    """

    source_filename: str = ""
    media_type: str = ""
    trigger_reason: str = "fallback"
    operator: str | None = None
    force_ocr: bool = False
    timeout_seconds: int = 900

    def compute_options_hash(self) -> str:
        """Stable identity, including the file name and media type.

        The name is in the key on purpose: the same bytes uploaded as
        ``paper.pdf`` and ``paper_final.pdf`` are one parse to the cache but two
        different audit records, and the stored document records the name it was
        parsed under.
        """
        payload = {
            "source_filename": self.source_filename,
            "media_type": self.media_type,
            "trigger_reason": self.trigger_reason,
            "force_ocr": bool(self.force_ocr),
            "timeout_seconds": int(self.timeout_seconds),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _record_field(record: Any, name: str, default: str = "") -> str:
    """Read one field off an audit record, dataclass or mapping."""
    if isinstance(record, dict):
        return str(record.get(name, default) or default)
    return str(getattr(record, name, default) or default)


class _CloudParserBridge:
    """Turn the cloud parser's own return shape into a ``ParseResult``.

    The cloud adapter returns ``(document | None, audit_record)`` and does not
    raise: privacy interception, missing token, quota exhaustion and network
    failure all come back as ``(None, record)``. That is a good contract for an
    adapter and a bad one for the router, which only understands
    ``ParseResult``. The mapping lives here rather than in the adapter so the
    adapter keeps its own, single-purpose shape, and so this module reads
    defensively: a future adapter that *does* return a ``ParseResult``, or a
    bare document, is accepted without a change here.
    """

    def __init__(
        self,
        inner: Any,
        *,
        source_filename: str = "",
        media_type: str = "",
        trigger_reason: str = "fallback",
        operator: str | None = None,
    ) -> None:
        self._inner = inner
        self.source_filename = source_filename
        self.media_type = media_type
        self.trigger_reason = trigger_reason
        self.operator = operator
        self.name = str(getattr(inner, "name", "") or "cloud")
        self.version = str(getattr(inner, "version", "") or "")

    def supports(self, media_type: str) -> bool:
        supports = getattr(self._inner, "supports", None)
        if not callable(supports):
            # A cloud parser that cannot answer is assumed to handle the media
            # type the orchestrator already routed to it. The gate above has
            # already decided the policy question; re-asking about file types
            # would only add a second way to silently skip the cloud.
            return True
        try:
            return bool(supports(media_type))
        except Exception as exc:  # noqa: BLE001 - see the gateway's supports() rule
            logger.warning("cloud parser supports() failed: %s", exc)
            return False

    def parse(self, file_path: str, options: Any) -> ParseResult:
        try:
            raw = self._inner.parse(file_path, options)
        except Exception as exc:  # noqa: BLE001 - the adapter's contract, enforced here
            logger.warning("cloud parser raised: %s", exc, exc_info=True)
            return ParseResult.failure(
                f"云端解析失败：{exc}",
                parser_name=self.name,
                parser_version=self.version,
                options_hash=_options_hash(options),
            )
        return self._normalize(raw, options)

    def _normalize(self, raw: Any, options: Any) -> ParseResult:
        options_hash = _options_hash(options)
        if isinstance(raw, ParseResult):
            return self._stamp(raw)
        if isinstance(raw, ParsedDocument):
            return self._stamp(ParseResult.from_document(raw, options_hash=options_hash))
        if isinstance(raw, tuple) and raw:
            document = raw[0]
            record = raw[1] if len(raw) > 1 else None
            if isinstance(document, ParsedDocument):
                message = _record_field(record, "message")
                outcome = _record_field(record, "outcome")
                return self._stamp(
                    ParseResult(
                        status=PARSE_PARTIAL if outcome == "partial" else PARSE_SUCCESS,
                        document=document,
                        quality=None,
                        warnings=[message] if message else [],
                        options_hash=options_hash,
                        cloud_outcome=str(outcome or ""),
                        diagnostics=[{"cloud_outcome": outcome}],
                    )
                )
            if document is None:
                outcome = _record_field(record, "outcome")
                message = _record_field(record, "message") or "云端解析未返回任何内容。"
                return ParseResult.failure(
                    message,
                    parser_name=self.name,
                    parser_version=self.version,
                    options_hash=options_hash,
                    cloud_outcome=str(outcome or ""),
                    diagnostics=[{"cloud_outcome": outcome}],
                )
        return ParseResult.failure(
            f"云端解析器返回了无法识别的结果类型（{type(raw).__name__}）。",
            parser_name=self.name,
            parser_version=self.version,
            options_hash=options_hash,
        )

    def _stamp(self, result: ParseResult) -> ParseResult:
        """Put the cloud parser's identity on the result *and* the document.

        The document carries the identity, so ``ParseResult.from_document`` can
        only copy a blank one. A cloud document whose ``parser_name`` is empty
        would make "who parsed this, and did it leave the machine?" unanswerable
        from the stored document alone — which is the single question the P0
        exit criterion asks. Stamped here rather than in the adapter because
        the adapter is duck-typed and has no ParseResult to stamp.
        """
        result.parser_name = result.parser_name or self.name
        result.parser_version = result.parser_version or self.version
        if result.document is not None:
            if not result.document.parser_name:
                result.document.parser_name = result.parser_name
            if not result.document.parser_version:
                result.document.parser_version = result.parser_version
        return result


def _build_default_cloud_parser(
    source_filename: str,
    media_type: str,
    trigger_reason: str,
    operator: str | None,
    sleeper: Callable[[float], None] | None,
) -> Any | None:
    """Construct the shipped cloud parser, or ``None`` if it is unavailable.

    Imported lazily so that a deployment which never enables cloud parsing does
    not pay for the adapter's module import, and so a broken optional
    dependency cannot take down the local chain.
    """
    try:
        from .adapters.mineru_cloud import MineruCloudParser
    except Exception as exc:  # noqa: BLE001 - optional dependency surface
        logger.warning("MinerU cloud adapter unavailable: %s", exc)
        return None
    return MineruCloudParser(operator=operator, sleeper=sleeper)


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------


def _quality_rank(quality: ParseQualityReport | None) -> int:
    if quality is None:
        # An unevaluated report is not a pass. Ranking it as PASS would let a
        # parser that never ran the gate beat one that did and failed.
        return _STATUS_RANK[QualityStatus.WARN]
    # ``.value`` first: a ``str``-mixin enum stringifies to "QualityStatus.PASS"
    # on Python 3.11+, and feeding that back into the enum raises.
    raw = getattr(quality.status, "value", quality.status)
    try:
        return _STATUS_RANK[QualityStatus(str(raw))]
    except ValueError:
        logger.warning("unknown quality status %r; ranked as a warning", raw)
        return _STATUS_RANK[QualityStatus.WARN]


def _document_rank(result: ParseResult) -> tuple[int, int, float, int]:
    """Sort key for comparing two *whole* documents. Lower is better.

    Four components, in priority order:

    1. usable before failed — nothing else matters if there is no document;
    2. quality severity — a FAIL means citations will not resolve, which no
       amount of extra recovered text repairs;
    3. locator coverage — a document that cannot be pointed at is a document
       this project cannot cite from;
    4. body size — last, and only as a tiebreak among documents that already
       pass the promises above.

    The comparison is deliberately total and deliberately shallow. It compares
    two whole documents and returns a number; it never opens two documents and
    writes a third. See the module docstring for why the second thing is
    forbidden.
    """
    document = result.document
    usable_rank = 0 if result.is_usable else 1
    quality_rank = _quality_rank(result.quality)
    if document is None or not result.is_usable:
        return (usable_rank, quality_rank, 0.0, 0)
    coverage = round(float(document.locator_coverage), 4)
    return (usable_rank, quality_rank, -coverage, -int(document.char_count))


def _better(candidate: ParseResult, incumbent: ParseResult) -> bool:
    """Whether ``candidate`` strictly beats ``incumbent``.

    Strict, so a tie keeps the earlier attempt. The primary route runs first and
    is free; making it lose a tie to a paid cloud parse would mean paying for
    an equal answer.
    """
    return _document_rank(candidate) < _document_rank(incumbent)


# --------------------------------------------------------------------------
# Locator vocabulary: the file's own units, not the parse's
# --------------------------------------------------------------------------
#
# Why this section exists
# ~~~~~~~~~~~~~~~~~~~~~~~
# The P0 exit criterion asks for a resolvable ``SourceLocator`` on every
# key body element, *in the vocabulary of the file*: a page for a PDF, a slide
# for a deck, a heading/paragraph for a Word file. The canonical schema
# (``LOCATOR_KINDS``) already has a kind per format, and
# :mod:`adapters.legacy_loader` builds the right one. But docling — and
# MinerU, which reaches its PDF pipeline the same way — normalise *every* input
# into one page-oriented model before the adapter sees it, so a deck comes back
# as pages and a Word file comes back as pages. Their adapters therefore emit
# ``pdf_page`` for all of them, which is honest at their layer and insufficient
# at ours.
#
# The fix belongs here because this is the only layer that holds the original
# file's identity *and* the parse result at the same time. The adapter is
# handed a media type and a docling payload and knows nothing else; the
# orchestrator has the real path, the real media type, and the winning
# document, and can therefore say "this page-ified result came from a .pptx".
#
# The hard rule this section obeys
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# **A locator must never resolve to a place the reader will be sent to the
# WRONG content.** That makes "always emit the native kind" the wrong instinct.
# Only one of the three office formats can be translated faithfully from a
# page-ified parse, and the other two are recorded as a *limitation* rather
# than papered over. See :data:`_NATIVE_LOCATORS` for the per-format verdict and
# the reasoning behind each; the short version is that only a slide is a page
# with a different name, while a paragraph index and a sheet name are values
# that exist in the original file and that a page-ified parse simply never
# reports. Minting them from a page number would produce a citation that looks
# exact and lands one section away, which in an Evidence-first product is worse
# than an honest page number and a sentence saying why.
#
# What is left open, and belongs to T3.3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The honest page number survives here, but *resolving* it — rendering a Word
# file and jumping to the right page, or opening a deck at the right slide —
# still requires re-opening the original file, which is the evidence viewer's
# job (T3.3), not the parse orchestrator's. This section therefore closes the
# vocabulary half of the criterion for decks and records the other two formats
# as explicitly unavailable; it does not close the resolution half for any
# non-PDF format.
#
# Two placement constraints, both deliberate:
#
# * **After the quality gate, not before it.** ``quality._page_of`` reads a page
#   only off a ``pdf_page`` locator, so a retargeted deck would score zero
#   page coverage against a non-zero ``page_count`` and FAIL — turning a
#   perfectly parsed presentation into a cloud escalation. The gate is
#   therefore run against the parse as the backend reported it, and the
#   retargeted document is what the user is finally handed.
# * **After the winner is chosen.** The retarget is a property of the *file*,
#   but a warning about it is only true of the document the user actually got.


def _normalized_media_type(media_type: str) -> str:
    """Strip parameters and case so ``.PPTX`` and ``; charset=`` cannot miss."""
    return str(media_type or "").split(";")[0].strip().lower()


@dataclass(frozen=True, slots=True)
class _NativeLocator:
    """What the file's own unit is, and whether a page-ified parse can say so.

    ``page_maps_one_to_one`` is the whole question. When it holds, the parser's
    page *n* is the file's slide *n* and the locator can be rewritten with
    complete confidence. When it does not, the parser's page is an artefact of a
    rendering step and carries no information about which paragraph, heading or
    worksheet the text came from — so ``reason`` explains that in the words a
    user reads, and the page locator is kept.
    """

    kind: str
    page_maps_one_to_one: bool
    reason: str = ""


#: Why a page-ified Word file keeps its page locators. One constant rather than
#: two copies, because the strict and the legacy media type are the same
#: promise to a reader and must not drift into saying different things.
_WORD_PAGE_LIMITATION = (
    "该 Word 文档先被渲染成分页再解析，解析结果没有报告这些段落、"
    "标题在原文件中的真实序号，因此引用只能定位到渲染页，"
    "无法安全地落到具体的标题或段落上。"
)

#: The same for a workbook, where the missing value is a name rather than an
#: index and therefore cannot be reconstructed at all.
_SHEET_PAGE_LIMITATION = (
    "该表格文件先被渲染成分页再解析，解析结果没有报告工作表名称与单元格范围，"
    "因此引用只能定位到渲染页，无法定位到具体的工作表。"
)

#: Office media types whose canonical locator kind is not a page, mapped to
#: whether this layer can honestly produce it. Keys are normalized media types;
#: both the strict OOXML types and the legacy Microsoft ones are listed because
#: a ``.ppt`` and a ``.pptx`` are the same promise to a reader and must not
#: differ only by spelling.
#:
#: The three verdicts, and why they differ:
#:
#: ``slide`` — **translatable.** A converter renders one slide onto one page and
#: nothing else, in both of docling's shapes for a deck (the dedicated backend
#: and the render-then-parse path), so page 3 of a converted deck *is* slide 3.
#: The only assumption worth defending is that no slide was dropped or inserted
#: between pages, which is why the page sequence is checked for contiguity
#: before any locator is rewritten.
#:
#: ``docx_element`` — **not translatable.** The field is a paragraph index into
#: the original ``word/document.xml`` body. A page-ified parse enumerates what
#: survived rendering: empty paragraphs, text boxes and repeated headers are
#: dropped, and figure captions are folded into their figure. Its k-th element
#: is therefore not the file's k-th paragraph, and the divergence grows the
#: longer the document is. The heading path *is* faithful and is preserved on
#: the element, but it is not a resolvable locator on its own —
#: ``SourceLocator.is_resolvable`` for ``docx_element`` requires a paragraph
#: index — so the locator stays a page and the limitation is stated out loud.
#:
#: ``sheet_range`` — **not translatable, and further away than the others.** The
#: field is a sheet *name*, a string the original workbook contains and a
#: rendered page cannot. There is no arithmetic from a page number to a sheet
#: name, so this is not a matter of an index that needs aligning; the value does
#: not exist anywhere in the parse.

_NATIVE_LOCATORS: dict[str, _NativeLocator] = {
    "application/vnd.openxmlformats-officedocument.presentationml.presentation":
    _NativeLocator("slide", True),
    "application/vnd.ms-powerpoint": _NativeLocator("slide", True),
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
    _NativeLocator("docx_element", False, _WORD_PAGE_LIMITATION),
    "application/msword": _NativeLocator("docx_element", False, _WORD_PAGE_LIMITATION),
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
    _NativeLocator("sheet_range", False, _SHEET_PAGE_LIMITATION),
    "application/vnd.ms-excel":
    _NativeLocator("sheet_range", False, _SHEET_PAGE_LIMITATION),
}

#: Appended to every degradation warning so the reader knows this is a known
#: limit of the parse route and not a document that failed to parse. T3.3 is
#: where the original file is re-opened and the finer unit becomes reachable.
_RETARGET_UNAVAILABLE_TAIL = (
    "定位到原文件中的页/幻灯片属于证据展示层（T3.3）的职责，"
    "届时会重新打开原文件补齐这一级定位。"
)


def _retarget_locators(document: ParsedDocument, media_type: str) -> ParsedDocument | None:
    """Rewrite page locators into the file format's own vocabulary.

    Returns a new document when anything changed, and ``None`` when the parse
    already speaks the file's language. Nothing is mutated in place: the
    document handed in may be the live parse cache entry, and rewriting it
    there would make the next call see an already-converted document while the
    on-disk copy still held page locators.

    :param document: the winning document, exactly as the backend produced it.
    :param media_type: the *original* file's media type, which is the only
        thing that decides which vocabulary the file should speak.
    """
    native = _NATIVE_LOCATORS.get(_normalized_media_type(media_type or document.media_type))
    if native is None:
        return None

    paged = [
        element
        for element in document.elements
        if element.locator.kind == "pdf_page" and element.locator.page is not None
    ]
    if not paged:
        # Either the adapter already speaks the file's vocabulary (the legacy
        # loader does) or the parse lost its page numbers entirely, in which
        # case there is nothing to translate *from* and the quality gate has
        # already reported it. Neither is this section's business.
        return None

    if native.page_maps_one_to_one:
        retargeted, reason = _retarget_to_slides(document, paged)
        if retargeted is not None:
            return retargeted
        return _with_warning(document, reason)

    return _with_warning(document, f"{native.reason}{_RETARGET_UNAVAILABLE_TAIL}")


def _retarget_to_slides(
    document: ParsedDocument,
    paged: list[Element],
) -> tuple[ParsedDocument | None, str]:
    """Convert a page-ified deck into ``slide`` locators, or decline to.

    Declining is the interesting branch. A deck whose page numbers are not
    ``1..n`` contiguously has lost a slide somewhere — or gained a cover, or a
    converter decided two slides share a page — and every page number after the
    break would then name a *different* slide than the one it was read from.
    Rewriting would produce locators that look perfect and point one or two
    slides off, so the pages are kept and the reason is reported instead.

    The ``bbox`` is dropped on the way. It is real geometry, but it is geometry
    in the *rendered page's* coordinate space, and the original ``.pptx`` has no
    such space — a shape's position in a deck is a placeholder-relative offset,
    not a point on a 72 dpi page. Keeping it would invite an evidence viewer to
    draw a highlight against the original file in coordinates that do not apply
    to it, which is the precise failure this whole section exists to prevent.
    The geometry is not lost: the raw docling artifact carries it, and T3.3 can
    reconcile it against a render it controls.
    """
    pages = sorted({element.locator.page for element in paged})
    if pages != list(range(1, len(pages) + 1)):
        missing = [page for page in range(1, max(pages) + 1) if page not in pages]
        return None, (
            f"该演示文稿解析出的页码序列不连续（{len(pages)} 个页码，最大页码 {max(pages)}，"
            f"缺失 {len(missing)} 页），无法确认渲染页与幻灯片一一对应，"
            f"因此保留页码定位而不改写为幻灯片编号。{_RETARGET_UNAVAILABLE_TAIL}"
        )
    return _slide_document(document), ""


def _slide_document(document: ParsedDocument) -> ParsedDocument:
    """Rebuild ``document`` with every page locator restated as a slide."""
    elements = [
        _with_locator(
            element,
            SourceLocator(kind="slide", slide=element.locator.page),
            label=f"slide {element.locator.page}",
        )
        for element in document.elements
    ]
    retargeted = _copy_document(document, elements)
    # Units are what a citation resolves to, and they were derived from the
    # page locators. Leaving them behind would mean a deck whose units say
    # "p. 3" while its elements say "slide 3" — two vocabularies in one
    # document, which is the chimera problem in miniature.
    retargeted.build_units()
    _attach_unit_ids(retargeted)
    return retargeted


def _with_locator(element: Element, locator: SourceLocator, *, label: str) -> Element:
    """A copy of ``element`` carrying ``locator`` and a reader-facing label.

    Every field is carried over explicitly rather than through ``replace`` or
    ``copy``: a locator rewrite must not be the place where a table grid, a
    LaTeX body or a heading context quietly disappears, and an explicit copy is
    what makes the next schema field a compile-time question rather than a
    silent data loss.
    """
    return Element(
        role=element.role,
        text=element.text,
        locator=locator,
        unit_id=element.unit_id,
        element_id=element.element_id,
        order=element.order,
        heading_path=element.heading_path,
        level=element.level,
        table=element.table,
        latex=element.latex,
        page_number_label=element.page_number_label or label,
    )


def _copy_document(
    document: ParsedDocument,
    elements: list[Element],
    *,
    warnings: list[str] | None = None,
) -> ParsedDocument:
    """A new document carrying ``elements``, with every other field kept."""
    return ParsedDocument(
        schema_version=document.schema_version,
        document_id=document.document_id,
        source_filename=document.source_filename,
        media_type=document.media_type,
        sha256=document.sha256,
        page_count=document.page_count,
        elements=elements,
        units=document.units,
        parser_name=document.parser_name,
        parser_version=document.parser_version,
        options_hash=document.options_hash,
        created_at=document.created_at,
        warnings=list(document.warnings) if warnings is None else warnings,
    )


def _with_warning(document: ParsedDocument, warning: str) -> ParsedDocument:
    """A copy of ``document`` whose warnings name a limitation of the parse."""
    if warning in document.warnings:
        return document
    return _copy_document(document, list(document.elements), warnings=[*document.warnings, warning])


def _attach_unit_ids(document: ParsedDocument) -> None:
    """Re-point every element at the unit that now contains it.

    ``build_units`` derives units from locators and does not write back, so
    without this the rebuilt units would carry fresh ids that nothing points at
    and the elements would keep ids for units that no longer exist.
    """
    unit_id_by_locator = {
        (unit.locator.kind, unit.locator.slide, unit.locator.page, unit.locator.paragraph):
        unit.unit_id
        for unit in document.units
    }
    for element in document.elements:
        locator = element.locator
        element.unit_id = unit_id_by_locator.get(
            (locator.kind, locator.slide, locator.page, locator.paragraph), ""
        )


# --------------------------------------------------------------------------
# The entry point
# --------------------------------------------------------------------------


def default_options() -> ParseOptions:
    """The options a call starts from, read live from configuration.

    Read per call rather than snapshotted, for the same reason the router reads
    ``PARSER_BACKEND`` per call: an operator who fixes a bad default in the
    environment should not have to restart the process to get it.
    """
    force_ocr = bool(getattr(config, "PARSER_FORCE_OCR", False))
    return ParseOptions(force_ocr=force_ocr)


def _with_force_ocr(options: ParseOptions) -> ParseOptions:
    """A copy of ``options`` with OCR forced on.

    A new object rather than a mutation: the original may be the caller's, may
    already be cached under, and mutating a shared options object is how a
    later, unrelated call ends up paying for an OCR parse it did not ask for.
    """
    return ParseOptions(
        force_ocr=True,
        timeout_seconds=options.timeout_seconds,
        extra=dict(options.extra),
    )


def _caching_router(router: ParserRouter, cache: ParseCache, log: _CallLog) -> ParserRouter:
    """A view of ``router`` whose parsers all consult the cache first.

    A shallow copy with the parser list replaced, rather than a new router, so
    the caller's router keeps its configuration: timeout executor, clock,
    rollback flag and parser order all come along untouched, and the caller's
    own ``parsers`` attribute is not mutated. Rebuilding from keyword arguments
    would have to guess at every constructor parameter the gateway grows next.
    """
    parsers = list(getattr(router, "parsers", []) or [])
    if not cache.enabled or not parsers:
        return router
    wrapped = [_CachingParser(parser, cache, log) for parser in parsers]
    view = copy.copy(router)
    view.parsers = wrapped
    return view


def _sha256_or_blank(file_path: str, hasher: Callable[[str], str]) -> str:
    try:
        return str(hasher(str(file_path)) or "")
    except OSError as exc:
        # The router will produce the real user-facing error for an unreadable
        # file. Here we only need to know that the cache cannot be keyed.
        logger.warning("cannot hash %s for the parse cache: %s", file_path, exc)
        return ""


def _is_usable(result: Any) -> bool:
    """Whether ``result`` is a ``ParseResult`` that yielded a document."""
    return (
        isinstance(result, ParseResult)
        and result.is_usable
        and result.document is not None
    )


def _is_clean(result: Any) -> bool:
    """Whether the quality gate passed outright, with nothing to repair.

    ``None`` is not clean. An unevaluated result has not been shown to be
    good, and treating "not yet judged" as "fine" would end the chain on the
    primary parse for every document whose adapter skipped the gate.
    """
    return (
        isinstance(result, ParseResult)
        and result.is_usable
        and result.quality is not None
        and result.quality.status is QualityStatus.PASS
    )


def _should_retry_with_ocr(
    result: ParseResult,
    media_type: str,
    options: ParseOptions,
    *,
    enabled: bool | None,
) -> tuple[bool, str]:
    """Whether a force-OCR retry is worth a second full parse.

    Four refusals, each with its own reason recorded on the attempt so the trail
    explains itself:

    * disabled by configuration or by the caller;
    * OCR already on — re-running the identical call is not a repair, it is the
      same parse twice, and if the second one is a cloud parser it is billed
      twice;
    * a media type that cannot carry a raster page, where OCR has nothing to
      find;
    * a clean document — the chain exists for documents that need rescuing, and
      burning time on a paper that parsed perfectly is how a 900-page thesis
      makes a user wait four minutes for nothing.
    """
    if enabled is False:
        return False, "ocr_retry_disabled"
    if options.force_ocr:
        return False, "ocr_already_enabled"
    if media_type not in OCR_CANDIDATE_MEDIA_TYPES:
        return False, "media_type_not_ocr_candidate"
    if _is_clean(result):
        return False, "quality_passed"
    return True, ""


def _cloud_outcome_of(attempts) -> str:
    """The cloud outcome behind a result, read off the attempt that produced it.

    The adapter records the machine-readable outcome and a message saying the
    same thing in prose. A caller must be able to branch on the former: "top up
    your quota" and "try again later" are different instructions for the same
    ``status``, and only one of them is actionable.
    """
    for attempt in reversed(list(attempts)):
        if attempt.status and attempt.status != PARSE_SUCCESS:
            for entry in attempt.diagnostics or ():
                value = str(entry.get("cloud_outcome") or "").strip()
                if value:
                    return value
    return ""


def _failure_message(attempts: list[ParseAttempt]) -> str:
    """The Chinese sentence a user reads when nothing parsed.

    Built from the attempts rather than written once, because the useful thing
    about a parse failure is *what was tried* — a single canned sentence
    ("parsing failed") tells a user with a password-protected PDF nothing they
    can act on.

    Each failed stage is expanded into one clause per candidate parser from the
    gateway's own diagnostics, not just the parser whose failure the router
    happened to return. The router keeps only the last failure (the most
    expensive attempt, by its reasoning), and a user whose file defeated two
    local parsers should be told about both.
    """
    if not attempts:
        return "文档解析失败：没有任何解析器被调用，请检查解析器配置。"
    details = "；".join(_describe_attempt(attempt) for attempt in attempts)
    return f"文档解析失败，所有解析器均未产出可用结果（{details}）。请更换文件或联系管理员。"


def _describe_attempt(attempt: ParseAttempt) -> str:
    if not attempt.attempted:
        return attempt.describe()
    others = [
        f"{item.get('parser')} 失败（{item.get('error') or '未知原因'}）"
        for item in attempt.diagnostics
        if item.get("attempted") and str(item.get("parser") or "") != attempt.parser_name
    ]
    if not others:
        return attempt.describe()
    head = f"{attempt.parser_name} 失败（{attempt.error_message}）"
    return f"{attempt.stage}：" + "；".join([head, *others])


def parse_document(
    file_path: str,
    *,
    router: ParserRouter | None = None,
    media_type: str | None = None,
    options: ParseOptions | None = None,
    quality_thresholds: QualityThresholds | None = None,
    cache: ParseCache | None = None,
    cloud_parser: Any | None = None,
    allow_cloud_fallback: bool = True,
    force_ocr_retry: bool | None = None,
    file_hasher: Callable[[str], str] = compute_file_sha256,
    sleeper: Callable[[float], None] | None = None,
    operator: str | None = None,
    privacy_gate: Callable[[], CloudFallbackDecision] | None = None,
) -> ParseOutcome:
    """Parse one file, escalating only as far as it has to.

    The business-layer entry point for ingestion. Callers get a
    :class:`ParseOutcome`, never an exception: a document that cannot be parsed
    is a normal operational outcome with a Chinese message and a full attempt
    trail, and the API layer turns it into a 4xx/5xx with something the user can
    read.

    The sequence is: primary route through the gateway, then — only if the
    document is unusable or its quality report FAILs — a force-OCR retry, then
    — only if the privacy policy allows it — a cloud fallback. Every stage
    produces a *whole* document; :func:`_better` picks one of them and the rest
    are discarded, never merged.

    :param router: the gateway to route through. Built from the shipped parsers
        when omitted.
    :param options: starting options; the force-OCR stage derives its own from
        these rather than mutating them.
    :param cache: parse cache. Built from configuration when omitted; pass
        ``ParseCache(enabled=False)`` to prove the uncached path.
    :param cloud_parser: anything satisfying the ``DocumentParser`` protocol
        plus the cloud adapter's ``(document, record)`` return shape. Built from
        the shipped adapter when omitted and the policy allows it.
    :param allow_cloud_fallback: a coarse operator switch evaluated *before* the
        privacy gate, so a deployment can forbid the cloud stage without
        touching the mode setting the gate reads.
    :param privacy_gate: injectable replacement for the adapter's gate, for
        tests that need a specific refusal.
    """
    path = str(file_path)
    # The declared type is normalised through the shared alias table before it
    # is used to route, and the extension answers when the table has no opinion
    # ("" means exactly that -- an untaught type, or none declared at all).
    #
    # This is where the two halves of an upload are reconciled. The edge accepts
    # a declared ``text/markdown`` (and ``application/x-pdf``, and
    # ``text/x-python``) as a legitimate spelling and validates it; adapters then
    # compare media types literally, so a declared alias that reached them
    # unchanged matched nothing and the document failed with no parser even
    # named in the attempt trail. Routing on the canonical form keeps the "an
    # alias is accepted" promise the edge already makes.
    #
    # A type whose meaning depends on the filename -- ``application/zip`` is the
    # real one, legitimate for both ``.docx`` and ``.pptx`` -- is deliberately
    # absent from the table, so it lands on the extension instead of on a guess.
    resolved_media_type = canonical_media_type(media_type) or str(media_type_for_path(path) or "")
    base_options = options if options is not None else default_options()
    resolved_cache = cache if cache is not None else build_default_parse_cache()
    active_router = router if router is not None else build_default_router()
    log = _CallLog(_sha256_or_blank(path, file_hasher))
    attempts: list[ParseAttempt] = []
    results: list[tuple[ParseAttempt, ParseResult]] = []

    def fail(message: str, *, status: str = PARSE_FAILED) -> ParseOutcome:
        return ParseOutcome(
            status=status,
            document=None,
            quality=None,
            attempts=attempts,
            sha256=log.sha256,
            media_type=resolved_media_type,
            error_message=message,
        )

    if not os.path.isfile(path):
        return fail(f"找不到待解析的文件：{path}。")

    view = _caching_router(active_router, resolved_cache, log)

    # -- stage 1: the primary route ---------------------------------------
    primary_options_hash = base_options.compute_options_hash()
    primary = view.parse(path, resolved_media_type, options=base_options)
    primary_attempt = _attempt_from_result(
        STAGE_PRIMARY, primary, primary_options_hash, log, network=False
    )
    attempts.append(primary_attempt)
    if _is_usable(primary):
        _ensure_quality(primary, quality_thresholds)
        primary_attempt.quality_status = _quality_label(primary)
        results.append((primary_attempt, primary))

    # -- stage 2: local repair, force OCR ----------------------------------
    retry, reason = _should_retry_with_ocr(
        primary, resolved_media_type, base_options, enabled=force_ocr_retry
    )
    if retry:
        ocr_options = _with_force_ocr(base_options)
        ocr_hash = ocr_options.compute_options_hash()
        repaired = view.parse(path, resolved_media_type, options=ocr_options)
        attempt = _attempt_from_result(
            STAGE_FORCE_OCR, repaired, ocr_hash, log, network=False
        )
        attempts.append(attempt)
        if _is_usable(repaired):
            _ensure_quality(repaired, quality_thresholds)
            attempt.quality_status = _quality_label(repaired)
            results.append((attempt, repaired))
    else:
        # Recorded even when the stage never runs, and recorded even when the
        # caller switched it off. "The repair was considered and declined" is
        # a fact about this document; an absent entry cannot say so.
        attempts.append(
            ParseAttempt(
                stage=STAGE_FORCE_OCR,
                skipped_reason=reason,
                status=ATTEMPT_SKIPPED,
            )
        )

    # -- stage 3: cloud fallback ------------------------------------------
    needs_cloud = _needs_cloud(resolved_media_type, [result for _, result in results])
    if not allow_cloud_fallback:
        attempts.append(
            ParseAttempt(
                stage=STAGE_CLOUD,
                skipped_reason="cloud_fallback_disabled",
                status=ATTEMPT_SKIPPED,
            )
        )
    elif not needs_cloud:
        attempts.append(
            ParseAttempt(
                stage=STAGE_CLOUD,
                skipped_reason="quality_sufficient",
                status=ATTEMPT_SKIPPED,
            )
        )
    else:
        _run_cloud_stage(
            path=path,
            media_type=resolved_media_type,
            cache=resolved_cache,
            log=log,
            cloud_parser=cloud_parser,
            sleeper=sleeper,
            operator=operator,
            quality_thresholds=quality_thresholds,
            privacy_gate=privacy_gate,
            attempts=attempts,
            results=results,
        )

    # -- pick one whole document ------------------------------------------
    return _finish(log, resolved_media_type, attempts, results)


def _needs_cloud(
    media_type: str,
    results: list[ParseResult],
) -> bool:
    """Whether the cloud stage is worth considering at all.

    The cloud is a paid, privacy-incurring last resort, so it is considered
    only when the local chain produced nothing usable, or produced something the
    quality gate rejected. A local ``PASS`` ends the chain, and so does a
    non-paginated format: sending a TXT file to a cloud OCR service costs money
    to solve a problem it cannot have.
    """
    if not results:
        return True
    if media_type not in OCR_CANDIDATE_MEDIA_TYPES:
        return False
    fail_rank = _STATUS_RANK[QualityStatus.FAIL]
    return any(_quality_rank(result.quality) >= fail_rank for result in results)


def _run_cloud_stage(
    *,
    path: str,
    media_type: str,
    cache: ParseCache,
    log: _CallLog,
    cloud_parser: Any | None,
    sleeper: Callable[[float], None] | None,
    operator: str | None,
    quality_thresholds: QualityThresholds | None,
    privacy_gate: Callable[[], CloudFallbackDecision] | None,
    attempts: list[ParseAttempt],
    results: list[tuple[ParseAttempt, ParseResult]],
) -> None:
    """Consult the privacy gate, then — only then — reach for the cloud.

    The order in this function is the release gate. The decision is made
    before the parser is even constructed, so a ``local_only`` deployment has
    no code path that can open a socket, whatever the cloud adapter does once
    handed a file.
    """
    decision = _decide(privacy_gate)
    if not decision.allowed:
        logger.info("cloud fallback refused (%s) for %s", decision.reason_code, path)
        attempts.append(
            ParseAttempt(
                stage=STAGE_CLOUD,
                skipped_reason=decision.reason_code,
                status=ATTEMPT_SKIPPED,
                error_message=decision.message,
                decision=decision,
            )
        )
        return

    parser = cloud_parser or _build_default_cloud_parser(
        os.path.basename(path), media_type, "fallback", operator, sleeper
    )
    if parser is None:
        unavailable = CloudFallbackDecision(
            allowed=False,
            reason_code="cloud_adapter_unavailable",
            message="云端解析适配器不可用，已跳过云端回退。",
        )
        attempts.append(
            ParseAttempt(
                stage=STAGE_CLOUD,
                skipped_reason=unavailable.reason_code,
                status=ATTEMPT_SKIPPED,
                error_message=unavailable.message,
                decision=unavailable,
            )
        )
        return

    bridge = _CloudParserBridge(
        parser,
        source_filename=os.path.basename(path),
        media_type=media_type,
        trigger_reason="fallback",
        operator=operator,
    )
    cloud_options = CloudParseOptions(
        source_filename=bridge.source_filename,
        media_type=media_type,
        trigger_reason="fallback",
        operator=operator,
    )
    options_hash = cloud_options.compute_options_hash()

    if cache.enabled and log.sha256:
        bridge = _CachingParser(bridge, cache, log)  # type: ignore[assignment]

    result = bridge.parse(path, cloud_options)
    if not isinstance(result, ParseResult):  # pragma: no cover - defensive
        result = ParseResult.failure("云端解析器返回了无法识别的结果。")
    attempt = _attempt_from_result(STAGE_CLOUD, result, options_hash, log, network=True)
    attempts.append(attempt)
    if _is_usable(result):
        _ensure_quality(result, quality_thresholds)
        attempt.quality_status = _quality_label(result)
        results.append((attempt, result))


def _decide(privacy_gate: Callable[[], CloudFallbackDecision] | None) -> CloudFallbackDecision:
    if privacy_gate is not None:
        try:
            return privacy_gate()
        except Exception as exc:  # noqa: BLE001 - a broken gate must refuse, not allow
            logger.warning("privacy gate raised, treating as refusal: %s", exc)
            return CloudFallbackDecision(
                allowed=False,
                reason_code="privacy_gate_error",
                message=f"隐私门控校验失败，已按拒绝处理：{exc}",
            )
    return evaluate_cloud_privacy_gate()


def _ensure_quality(result: ParseResult, thresholds: QualityThresholds | None) -> None:
    """Give ``result`` a quality report if nobody has.

    The adapter may have computed one; the cloud bridge certainly has not. A
    missing report is not a pass — it means the stage was never judged, and
    judging it here is what lets stage 2 and stage 3 be compared at all.
    """
    if result.document is None or result.quality is not None:
        return
    result.quality = evaluate_quality(result.document, thresholds)


def _quality_label(result: ParseResult) -> str:
    quality = result.quality
    if quality is None:
        return ""
    return str(getattr(quality.status, "value", quality.status) or "")


def _attempt_from_result(
    stage: str,
    result: Any,
    options_hash: str,
    log: _CallLog,
    *,
    network: bool,
) -> ParseAttempt:
    """Describe one stage from the ``ParseResult`` it produced."""
    if not isinstance(result, ParseResult):  # pragma: no cover - the gateway never does this
        return ParseAttempt(
            stage=stage,
            attempted=True,
            network=network,
            status=PARSE_FAILED,
            error_message=f"{stage} 阶段返回了无法识别的结果类型。",
        )
    parser_name = str(result.parser_name or "")
    return ParseAttempt(
        stage=stage,
        parser_name=parser_name,
        parser_version=str(result.parser_version or ""),
        options_hash=options_hash or str(result.options_hash or ""),
        status=result.status,
        attempted=True,
        quality_status=_quality_label(result),
        duration_ms=int(result.duration_ms or 0),
        error_message=str(result.error_message or ""),
        cache_hit=log.was_cached(parser_name, options_hash or str(result.options_hash or "")),
        network=network,
        diagnostics=[dict(item) for item in result.diagnostics or []],
    )


def _finish(
    log: _CallLog,
    media_type: str,
    attempts: list[ParseAttempt],
    results: list[tuple[ParseAttempt, ParseResult]],
) -> ParseOutcome:
    """Pick the winning whole document and assemble the outcome."""
    cloud_attempt = next((attempt for attempt in attempts if attempt.stage == STAGE_CLOUD), None)
    # A cloud attempt that was *made* touched the network and cost money, even
    # if the local chain beat it. Only ``attempted`` and ``network`` together
    # mean a real egress; a stage refused by the gate sets neither.
    cloud_attempted = bool(cloud_attempt and cloud_attempt.attempted and cloud_attempt.network)
    refusal = cloud_attempt.decision if cloud_attempt is not None else None

    winner: ParseResult | None = None
    winner_attempt: ParseAttempt | None = None
    for attempt, result in results:
        if winner is None or _better(result, winner):
            winner = result
            winner_attempt = attempt

    if winner is None or winner.document is None:
        return ParseOutcome(
            status=PARSE_FAILED,
            document=None,
            quality=None,
            attempts=attempts,
            sha256=log.sha256,
            media_type=media_type,
            cloud_attempted=cloud_attempted,
            cloud_outcome=_cloud_outcome_of(attempts),
            cloud_refusal=refusal,
            error_message=_failure_message(attempts),
        )

    document = winner.document
    if not document.sha256:
        # Fill in the content hash the cache was keyed on, so a document that
        # reached storage can be matched back to the bytes it came from.
        document.sha256 = log.sha256
    if not document.media_type:
        document.media_type = media_type
    if winner_attempt is not None:
        # The winning parser's identity is taken from the attempt, which the
        # gateway stamps from the parser it actually ran. The document's own
        # field is a courtesy to downstream readers and is left alone when the
        # adapter already filled it in.
        if not document.parser_name:
            document.parser_name = winner_attempt.parser_name
        if not document.parser_version:
            document.parser_version = winner_attempt.parser_version
        if not document.options_hash:
            document.options_hash = winner_attempt.options_hash

    used_cloud = winner_attempt is not None and winner_attempt.stage == STAGE_CLOUD
    fallback_used = winner_attempt is not None and winner_attempt.stage != STAGE_PRIMARY

    # Last, and only on the document the reader is actually being handed. A
    # retargeted deck must never reach the quality gate, whose page-coverage
    # rule reads a page only off a ``pdf_page`` locator; the gate has already
    # been run above, against the parse as the backend reported it.
    retargeted = _retarget_locators(document, media_type)
    if retargeted is not None:
        document = retargeted

    # De-duplicate while preserving order. The result's warnings and the
    # document's own warnings overlap by construction -- a document accumulates
    # the same capability notice the adapter emitted for it -- and a warning
    # list that repeats itself is worse than one that is short: a reader
    # learns to skim it, and the one warning they needed was the second copy.
    # It also quietly broke T4.3's silent-truncation gate, which could not
    # tell a real problem from a capability notice appearing twice.
    warnings = list(
        dict.fromkeys([*winner.warnings, *document.warnings])
    )
    if winner.quality is not None and winner.quality.status is not QualityStatus.PASS:
        # A degraded parse is surfaced, never swallowed. The release gate says
        # silent truncation on the eval corpus must be zero, so a document the
        # gate disliked travels with the rules that disliked it.
        blamed = winner.quality.failed_rules or winner.quality.warned_rules
        warnings.append(
            f"解析质量为 {winner.quality.status.value}，"
            f"问题规则：{'、'.join(blamed) or '未知'}。"
            "引用定位可能不完整，请以原文为准核对。"
        )

    return ParseOutcome(
        status=winner.status,
        document=document,
        quality=winner.quality,
        attempts=attempts,
        cache_hit=bool(winner_attempt and winner_attempt.cache_hit),
        fallback_used=fallback_used,
        cloud_attempted=cloud_attempted,
        cloud_outcome=_cloud_outcome_of(attempts),
        used_cloud=used_cloud,
        warnings=warnings,
        sha256=log.sha256,
        media_type=media_type,
        raw_artifact_uri=winner.raw_artifact_uri,
        cloud_refusal=refusal,
    )


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "OCR_CANDIDATE_MEDIA_TYPES",
    "STAGE_CLOUD",
    "STAGE_FORCE_OCR",
    "STAGE_PRIMARY",
    "STAGES",
    "CacheEntry",
    "CloudFallbackDecision",
    "CloudParseOptions",
    "ParseAttempt",
    "ParseCache",
    "ParseCacheKey",
    "ParseOutcome",
    "build_default_parse_cache",
    "default_options",
    "evaluate_cloud_privacy_gate",
    "parse_document",
]
