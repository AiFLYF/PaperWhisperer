"""Resource limits and container defence for the parser path (T4.2).

The plan asks for four limits on a parser — CPU, memory, duration, pages — plus
zip-bomb and nested-archive defence. Two of the four already exist and one of
them does not work; the other two need a mechanism this project cannot get for
free. This module is the mechanism, and the long docstrings below are the
argument for it.

What already exists, and what actually runs
-------------------------------------------

*Duration* is enforced by :class:`~paperwhisperer.ingestion.gateway.ParseOptions`
through a per-attempt ``future.result(timeout=...)``. It is **cooperative**: the
worker thread is not killed, so a wedged parser keeps burning CPU after the
router has moved on. The gateway says so itself and is right to.

*Pages* looked enforced and is not. ``MAX_DOCUMENT_PAGES`` is defined in
``core/config.py``, reported by ``/api/health``, described in ``.env.example``
as *"NOT YET ENFORCED -- the parser gateway does not refuse an oversized
document yet"*, and referenced nowhere else. :func:`admit_document` is what
makes that sentence false: it counts the pages of a PDF from its own bytes,
before any parser is constructed, and refuses the upload. Counting before
parsing is the only version of this check worth having — a limit discovered
after a 4 000-page scan has been converted is a report, not a limit.

Why a subprocess, and why it is off by default
-----------------------------------------------

The personal tier runs the API, the job worker and the parser in **one
process**. That single fact decides this design, because the only way to bound
CPU and memory *hard* in CPython is ``resource.setrlimit``, and rlimits are a
process-wide, irreversible-downward property:

* ``RLIMIT_AS`` exceeded raises :class:`MemoryError` **in whatever code
  happened to allocate**, which in a single-process deployment is the web
  worker serving the next request. The parser did not die; the server did.
* A rlimit can only be lowered further afterwards. There is no "restore", so a
  transient spike would permanently cap the whole application.

So an in-process rlimit is not a limit, it is an outage with a configuration
file attached. This is the constraint the task states, and it is why the
mechanism here is layered instead of singular:

**Layer 0 — admission control. Always on, in-process, cannot fail the server.**
:func:`admit_document` checks input bytes, page count and container shape
before a parser is built. These are the limits that catch the realistic
attack (a huge document) and they cost microseconds.

**Layer 1 — cooperative observation. Always on, in-process, detects only.**
:class:`ResourceGuard` samples the process's own RSS and CPU time on a timer
and records a verdict. It deliberately **cannot stop the parser** — see
:func:`run_guarded` for why a thread cannot — so what it buys is a *result*
that is refused when the parse went over budget, rather than a good-looking
document derived from a run that was already pathological. Detecting is not
enforcing, and this module never claims otherwise.

**Layer 2 — hard isolation. Opt-in, and the only real enforcement.**
:func:`run_guarded` forks, installs ``RLIMIT_AS`` / ``RLIMIT_CPU`` /
``RLIMIT_NPROC`` / ``RLIMIT_FSIZE`` **in the child only**, and ``SIGKILL``s it
when the wall clock runs out. The parent is untouched, which is the whole
point. It is off by default (``PARSER_ISOLATION_ENABLED``) for three honest
reasons, and the decision to turn it on belongs to the deployment:

1. ``fork()`` from a process with live threads is only async-signal-safe in the
   child. CPython 3.12+ warns about it for exactly this reason. The child here
   calls a parser, which is not async-signal-safe, so a deadlock in the child
   is possible in principle. The *parse* is still bounded, and a hung child is
   ``SIGKILL``-able, so the failure mode is "this parse failed", not "the
   server is stuck" — but it is not free.
2. The result crosses a pipe by pickle. :class:`~paperwhisperer.ingestion.schema.ParsedDocument`
   and :class:`~paperwhisperer.ingestion.gateway.ParseResult` both survive that;
   an adapter holding a live library handle does not, and would have to return
   only its schema.
3. A forked child inherits the parent's copy-on-write pages, so ``RLIMIT_AS``
   measures *virtual* address space, which torch inflates well beyond its
   resident footprint. The default ceiling is therefore generous enough for a
   torch process to import and convert a normal paper, and tight enough that a
   40 GB allocation fails.

The recommendation this module encodes: **personal tier runs layers 0 and 1 and
leaves layer 2 off**; a deployment that converts many large PDFs, or a hosted
tier, turns it on. A limit that is off by default is only honest if the
default is the safe one, and here it is.

There is no escape hatch, and that is deliberate
-----------------------------------------------

No environment variable disables any of this. There is no ``PARSER_LIMITS_OFF``,
no ``enforce=False``, and a limit of ``0`` means "use the default", not
"unlimited" — see :func:`_clamp_limit`. A security limit that can be turned off
by a value that looks like a limit is a limit that will be turned off by a
value that looks like a limit.

Where archives actually enter
----------------------------

``config.ALLOWED_EXTENSIONS`` is ``{.txt, .pdf, .docx, .pptx}``. ``.zip`` is not
in it, so an upload *named* ``.zip`` is refused by
:func:`~paperwhisperer.core.text.build_safe_upload_filename` before any
decompression logic runs. Asked whether that is a correct defence, the honest
answer has two halves:

* **For a file named ``.zip`` it is a correct defence**, and the MinerU guard at
  ``validate_zip_bomb`` is genuinely off the live path — there is no zip-ingest
  feature to guard.
* **It is not the defence that matters, and it is accidental.** It is a list of
  formats the product *parses*, not a decompression policy, and the two entries
  next to the hole are themselves zip archives: **``.docx`` and ``.pptx`` are
  ZIP containers**, they are allowed, and
  :func:`~paperwhisperer.documents.validation.validate_document_file_signature`
  *requires* them to start with ``PK\\x03\\x04``. A 3.7 MB ``.docx`` that
  inflates to 4 GB therefore passes the extension gate, the magic-byte gate and
  the 16 MB upload ceiling, and lands in ``python-docx``, which opens it as a
  zip.

So :func:`check_container` puts the archive defence on the path that is live,
keyed on **magic bytes rather than extension**, and adds the two dimensions the
existing guard does not have:

* **compression ratio**, which is what actually separates "a Word file with
  200 pages of XML" (typically 30-80x) from "a bomb" (1000x+). The existing
  three dimensions are all *absolute* and tuned for one producer's output, so a
  bomb split across four 60 MB members passes all of them: 6 entries < 512,
  each entry < 64 MB, total 240 MB < 256 MB. Verified, not hypothesised.
* **nesting depth**, bounded at every level against a **shared** budget rather
  than a per-level one, because a per-level budget is exactly what lets three
  levels each stay under while the total does not.

The three existing dimensions are **reused, not reimplemented**
--------------------------------------------------------------

:func:`check_container` calls ``adapters.mineru_cloud.validate_zip_bomb`` for the
entry count, per-entry and total-declared ceilings rather than copying its
constants, because a second copy of "64 MB per entry" is a second value that
will drift. What this module adds on top is stated in the two bullets above and
in :func:`check_container`'s own docstring.

Secrets
-------

:func:`redact_secrets` and :class:`SecretRedactingFilter` live here because
this is the security module of the ingestion package and they are what the
three known credential-leak paths need — see the test file for the exact
call sites, which are in modules this task does not own.
"""

from __future__ import annotations

import io
import logging
import os
import pickle
import re
import select
import signal
import sys
import time
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import Any

from paperwhisperer.core import config

# layer -- the LLM client, the API routes, this module -- scrubs with the
# same code. Re-exported here because this is where it lived, and callers
# that already import it from here keep working.
# The redaction implementation lives in ``core/text.py`` so that every
#: dependencies, in the order they are tried. Pattern-matching cannot be
#: complete — a bare opaque token has no shape — so :func:`redact_secrets`
#: takes the *values* the process is actually holding as well and replaces
#: those literally. The patterns are for the case where the value is not
#: available to the caller, and the literal pass is the one that is certain.
# The redaction implementation moved to ``core/text.py`` so that every
# layer -- the LLM client, the API routes, this module -- scrubs with the
# same code. Re-exported here because this is where it lived, and callers
# that already import it from here keep working.
#: Private helpers, re-exported so this module's own tests and any
#: diagnostic that reached for them keep working.
from paperwhisperer.core.text import (  # noqa: F401 - re-exported  # noqa: F401 - re-exported
    _SECRET_PATTERNS,
    REDACTED,
    SecretRedactingFilter,
    _live_secret_values,
    redact_secrets,
)

logger = logging.getLogger(__name__)

#: Default per-parse memory ceiling for the optional subprocess sandbox, in
#: bytes. 2 GiB is roughly what a torch-backed docling process needs to import
#: its models and convert a 200-page paper; the point of the ceiling is to stop
#: a runaway allocation, not to ration a working one.
DEFAULT_MAX_MEMORY_BYTES = 2 * 1024 * 1024 * 1024

#: Default CPU ceiling for the subprocess sandbox, in seconds of CPU time
#: (not wall clock). Slightly above :data:`DEFAULT_TIMEOUT_SECONDS` so the
#: wall clock is normally what fires, and the CPU limit is the backstop for a
#: parse that is spinning without blocking.
DEFAULT_MAX_CPU_SECONDS = 180

#: Default wall-clock budget when nothing is configured. Mirrors the gateway's
#: own default so the two layers do not disagree about what "a parse takes".
DEFAULT_TIMEOUT_SECONDS = 120

#: Default page ceiling. Matches ``MAX_DOCUMENT_PAGES``' own default; read from
#: config at call time rather than imported, because ``core.config`` and the
#: ingestion package land in the same release but not the same commit.
DEFAULT_MAX_PAGES = 400

#: Hard floor and ceiling for any configured limit. A configured value outside
#: this range is clamped rather than honoured, and the clamp is logged — the
#: same posture :func:`paperwhisperer.core.config.parse_int_env` takes, for the
#: same reason: a typo in ``.env`` must not silently become "no limit".
MIN_LIMIT = 1
MAX_LIMIT_CEILING = 64 * 1024 * 1024 * 1024

#: Compression ratio above which an archive is treated as a bomb regardless of
#: its absolute size. 200 is chosen from the shape of real office documents: a
#: text-heavy DOCX of 200 pages of XML compresses 30-80x, a PDF with embedded
#: images is under 5x, and the classic bombs are 1000x+. The gap is wide, which
#: is what makes a threshold like this usable at all.
DEFAULT_MAX_COMPRESSION_RATIO = 200

#: How many levels of nesting are walked. 1 means "the container plus whatever
#: is inside it"; a member that is *itself* an archive at depth 1 is inspected
#: once and anything inside that is refused. Real documents do not nest, so any
#: value above 0 buys nothing but attack surface.
DEFAULT_MAX_ARCHIVE_DEPTH = 1

#: Ceiling on the total *declared* uncompressed size across **all** nesting
#: levels combined. The existing per-producer ceiling is 256 MB; this one is the
#: same number, deliberately, so enabling this module does not change what a
#: legitimate MinerU result is allowed to be.
DEFAULT_MAX_ARCHIVE_TOTAL_BYTES = 256 * 1024 * 1024

#: Ceiling on any single member's declared uncompressed size, and the same
#: reasoning as above: identical to the MinerU per-entry cap on purpose.
DEFAULT_MAX_ARCHIVE_ENTRY_BYTES = 64 * 1024 * 1024

#: Ceiling on entry count, again identical to the MinerU cap.
DEFAULT_MAX_ARCHIVE_ENTRIES = 512

#: Bytes read from a member to decide whether it is itself an archive. Four is
#: enough for every magic number below and small enough that the decision
#: cannot itself be a decompression.
ARCHIVE_MAGIC_SNIFF_BYTES = 8

#: Leading bytes that mean "this is an archive". Checked against the *content*
#: of each member, not its name, because ``payload.txt`` inside a zip is a
#: perfectly good place to hide a zip and a name is attacker-controlled.
ARCHIVE_MAGIC_PREFIXES: tuple[bytes, ...] = (
    b"PK\x03\x04",  # zip / docx / pptx / xlsx / odt / jar
    b"PK\x05\x06",  # empty zip
    b"PK\x07\x08",  # spanned zip
    b"\x1f\x8b",  # gzip
    b"BZh",  # bzip2
    b"\xfd7zXZ\x00",  # xz
    b"\x04\x22\x4d\x18",  # lz4
    b"\x28\xb5\x2f\xfd",  # zstd
    b"7z\xbc\xaf\x27\x1c",  # 7z
    b"Rar!\x1a\x07",  # rar
)

#: Offsets used to read resident memory. ``/proc/self/statm`` field 2 is the
#: resident set in pages; this is the page size to multiply it by.
_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


class ParserLimitExceeded(RuntimeError):
    """A parser input or a parse result exceeded a configured limit.

    Carries a machine-readable ``code`` for the same reason
    :class:`~paperwhisperer.ingestion.adapters.mineru_cloud.MineruError` does:
    the orchestrator needs to distinguish "the file is too big" from "the
    backend is broken" to decide whether a retry with different options could
    ever help. ``str(exc)`` is a finished Chinese sentence for the user, which
    is what the gateway surfaces.
    """

    def __init__(
        self, message: str, code: str = "parser_limit_exceeded", *, limit: str = ""
    ) -> None:
        super().__init__(message)
        self.code = code
        self.limit = limit


# --------------------------------------------------------------------------
# Limits
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ParserLimits:
    """Every ceiling one parse runs under, resolved once from configuration.

    Frozen because it is a policy snapshot: a limit that could be edited halfway
    through a parse would mean the verdict at the end was judged against
    different bars than the admission check at the start, which is precisely
    the "which rule fired" question :mod:`paperwhisperer.ingestion.quality`
    refuses to answer with a single number.

    Every field is read through ``getattr(config, ...)`` with a default rather
    than imported, for the same reason :func:`thresholds_from_config` in the
    quality gate uses: this module and the config keys that feed it must be
    able to disagree in order without an ``ImportError`` at module scope.
    """

    max_pages: int = DEFAULT_MAX_PAGES
    max_input_bytes: int = 16 * 1024 * 1024
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    max_memory_bytes: int = DEFAULT_MAX_MEMORY_BYTES
    max_cpu_seconds: int = DEFAULT_MAX_CPU_SECONDS
    max_processes: int = 64
    max_output_bytes: int = 64 * 1024 * 1024
    max_archive_entries: int = DEFAULT_MAX_ARCHIVE_ENTRIES
    max_archive_entry_bytes: int = DEFAULT_MAX_ARCHIVE_ENTRY_BYTES
    max_archive_total_bytes: int = DEFAULT_MAX_ARCHIVE_TOTAL_BYTES
    max_compression_ratio: int = DEFAULT_MAX_COMPRESSION_RATIO
    max_archive_depth: int = DEFAULT_MAX_ARCHIVE_DEPTH
    isolation_enabled: bool = False
    #: How often :class:`ResourceGuard` samples. 0.1 s is frequent enough that
    #: a 10-second runaway is sampled ~100 times and rare enough that the
    #: sampler thread is not itself measurable next to a converter.
    sample_interval_seconds: float = 0.1

    def describe(self) -> dict[str, Any]:
        """The ceilings, for a health endpoint or a diagnostics block.

        Only numbers and booleans. A limit set is not a secret and an operator
        needs to see it, but nothing in this method reads a credential — see
        :func:`describe_parser_limits` for the credential-adjacent one.
        """
        return {
            "max_pages": self.max_pages,
            "max_input_bytes": self.max_input_bytes,
            "timeout_seconds": self.timeout_seconds,
            "max_memory_bytes": self.max_memory_bytes,
            "max_cpu_seconds": self.max_cpu_seconds,
            "max_output_bytes": self.max_output_bytes,
            "max_archive_entries": self.max_archive_entries,
            "max_archive_entry_bytes": self.max_archive_entry_bytes,
            "max_archive_total_bytes": self.max_archive_total_bytes,
            "max_compression_ratio": self.max_compression_ratio,
            "max_archive_depth": self.max_archive_depth,
            "isolation_enabled": self.isolation_enabled,
        }


def _clamp_limit(name: str, raw: Any, default: int, *, maximum: int = MAX_LIMIT_CEILING) -> int:
    """Resolve one configured ceiling, refusing to resolve to "unlimited".

    The escape-hatch question is decided here and nowhere else. Three ways a
    limit is conventionally weakened, and what this does with each:

    * ``0`` or a negative number — the usual "0 means no limit" convention, and
      also what a ``MAX_DOCUMENT_PAGES=`` line with a typo produces. Treated as
      *unset* and the default applies. A limit that can be deleted by typing
      zero is a limit that will be.
    * A value beyond :data:`MAX_LIMIT_CEILING` — an operator who genuinely
      wants 100 GiB is welcome to have it, but not by accident and not past
      64 GiB, which no document in this product's corpus is within an order of
      magnitude of.
    * A non-integer — falls back to the default with a log line, matching
      :func:`~paperwhisperer.core.config.parse_int_env`.

    The clamp is **logged** in every case that changes the value. A limit that
    quietly reverts is indistinguishable from a limit that was never read.
    """
    try:
        value = int(raw)
    except (TypeError, ValueError):
        if str(raw or "").strip():
            logger.warning("%s=%r is not an integer; using the default %d.", name, raw, default)
        return default
    if value <= 0:
        logger.warning(
            "%s=%r is not a positive limit; treating it as unset and using the default %d. "
            "There is no way to disable a parser limit from the environment.",
            name, value, default,
        )
        return default
    if value > maximum:
        logger.warning(
            "%s=%d exceeds the hard ceiling %d; clamping.", name, value, maximum
        )
        return maximum
    return value


def limits_from_config() -> ParserLimits:
    """Read every ceiling from ``core.config``, clamping anything unreasonable.

    Read at call time rather than at import so an operator editing ``.env`` and
    restarting nothing sees the change on the next parse — the same reasoning
    as ``PARSER_BACKEND`` being read per call in the gateway.
    """
    defaults = ParserLimits()
    return ParserLimits(
        max_pages=_clamp_limit("MAX_DOCUMENT_PAGES", getattr(config, "MAX_DOCUMENT_PAGES", None),
                               defaults.max_pages, maximum=5000),
        max_input_bytes=_clamp_limit("MAX_UPLOAD_BYTES", getattr(config, "MAX_UPLOAD_BYTES", None),
                                     defaults.max_input_bytes),
        timeout_seconds=_clamp_limit("PARSER_TIMEOUT_SECONDS",
                                     getattr(config, "PARSER_TIMEOUT_SECONDS", None),
                                     defaults.timeout_seconds, maximum=1800),
        max_memory_bytes=_clamp_limit("PARSER_MAX_MEMORY_MB",
                                      _bytes_from_mb(getattr(config, "PARSER_MAX_MEMORY_MB", None)),
                                      defaults.max_memory_bytes),
        max_cpu_seconds=_clamp_limit("PARSER_MAX_CPU_SECONDS",
                                     getattr(config, "PARSER_MAX_CPU_SECONDS", None),
                                     defaults.max_cpu_seconds, maximum=3600),
        max_processes=_clamp_limit("PARSER_MAX_PROCESSES",
                                   getattr(config, "PARSER_MAX_PROCESSES", None),
                                   defaults.max_processes, maximum=4096),
        max_output_bytes=_clamp_limit("PARSER_MAX_OUTPUT_MB",
                                      _bytes_from_mb(getattr(config, "PARSER_MAX_OUTPUT_MB", None)),
                                      defaults.max_output_bytes),
        max_archive_entries=_clamp_limit("PARSER_MAX_ARCHIVE_ENTRIES",
                                         getattr(config, "PARSER_MAX_ARCHIVE_ENTRIES", None),
                                         defaults.max_archive_entries),
        max_archive_entry_bytes=_clamp_limit(
            "PARSER_MAX_ARCHIVE_ENTRY_MB",
            _bytes_from_mb(getattr(config, "PARSER_MAX_ARCHIVE_ENTRY_MB", None)),
            defaults.max_archive_entry_bytes),
        max_archive_total_bytes=_clamp_limit(
            "PARSER_MAX_ARCHIVE_TOTAL_MB",
            _bytes_from_mb(getattr(config, "PARSER_MAX_ARCHIVE_TOTAL_MB", None)),
            defaults.max_archive_total_bytes),
        max_compression_ratio=_clamp_limit("PARSER_MAX_COMPRESSION_RATIO",
                                           getattr(config, "PARSER_MAX_COMPRESSION_RATIO", None),
                                           defaults.max_compression_ratio),
        max_archive_depth=_clamp_limit("PARSER_MAX_ARCHIVE_DEPTH",
                                       getattr(config, "PARSER_MAX_ARCHIVE_DEPTH", None),
                                       defaults.max_archive_depth, maximum=8),
        isolation_enabled=bool(getattr(config, "PARSER_ISOLATION_ENABLED", False)),
    )


def _bytes_from_mb(raw: Any) -> Any:
    """Convert a megabyte-denominated setting to bytes, tolerating absence.

    A sandbox ceiling is naturally written in gigabytes by the person setting
    it and in bytes by the code that compares it. The unit lives in the
    environment variable's *name* (``PARSER_MAX_MEMORY_MB``) so there is exactly
    one place to change it. ``None`` passes through so that "not configured" is
    still distinguishable from "configured as 0", which :func:`_clamp_limit`
    treats differently on purpose.
    """
    if raw is None or raw == "":
        return None
    try:
        return int(raw) * 1024 * 1024
    except (TypeError, ValueError):
        return raw  # let _clamp_limit report it as a non-integer


def describe_parser_limits(limits: ParserLimits | None = None) -> dict[str, Any]:
    """Limit settings for a health or diagnostics block, credentials excluded.

    Exists beside :func:`paperwhisperer.core.config.describe_parser_settings`
    and repeats none of its keys, so that adding a limit here cannot collide
    with a config export that is already consumed by a template. It reads no
    credential at all — in particular it does **not** report whether a token is
    configured, which :func:`describe_parser_settings` already does as a
    boolean and which this deliberately does not duplicate.
    """
    return (limits or limits_from_config()).describe()


# --------------------------------------------------------------------------
# Redaction
# --------------------------------------------------------------------------


def install_secret_filter(logger_: logging.Logger | None = None, *,
                          extra_values: Iterable[str] = ()) -> SecretRedactingFilter:
    """Attach a :class:`SecretRedactingFilter` to a logger and return it.

    Idempotent by marker attribute: calling it twice does not redact twice,
    which matters because the natural places to call it are ``app`` startup and
    a test fixture, and double redaction would replace ``[REDACTED]`` with
    ``[REDACTED]`` and then try to redact a string that is no longer a secret.
    """
    target = logger_ if logger_ is not None else logging.getLogger()
    # Reuse whatever is already attached, to the logger or to any of its
    # handlers. Returning a second object would break the idempotence this
    # function promises and that its test pins: two filters means two passes
    # over every record, and a caller that then removes "the" filter removes
    # one of them and believes it removed both.
    installed = next(
        (item for item in target.filters if isinstance(item, SecretRedactingFilter)),
        None,
    ) or next(
        (
            item
            for handler in target.handlers
            for item in handler.filters
            if isinstance(item, SecretRedactingFilter)
        ),
        None,
    )
    if installed is None:
        installed = SecretRedactingFilter(extra_values=extra_values)

    if not any(isinstance(item, SecretRedactingFilter) for item in target.filters):
        target.addFilter(installed)

    # Also on the handlers, and this is the part that actually does the work.
    # A filter on a *parent* logger sees only the records emitted to that
    # logger directly. A record from `paperwhisperer.llm.client` travels up
    # through the root logger's HANDLERS, not through its filters -- so a
    # filter attached to the root logger alone scrubs almost nothing while
    # reading exactly as though the process were protected. Handlers are where
    # every propagated record passes, so that is where the filter has to be.
    for handler in target.handlers:
        if not any(isinstance(item, SecretRedactingFilter) for item in handler.filters):
            handler.addFilter(installed)
    return installed


# --------------------------------------------------------------------------
# Page counting
# --------------------------------------------------------------------------


def _pdf_pages_via_pypdf(path: str) -> int | None:
    try:
        from pypdf import PdfReader  # noqa: PLC0415 - optional, and a heavy import
    except ImportError:  # pragma: no cover - pypdf is a declared dependency
        return None
    try:
        return len(PdfReader(path).pages)
    except Exception:
        # A malformed or encrypted PDF is a parse failure, not a page-count
        # failure. Returning None lets admission pass the file through to a
        # parser that will produce a proper Chinese error message; refusing it
        # here would replace a good diagnosis with a bad one.
        return None


#: Count of ``/Type /Page`` occurrences, used only when pypdf cannot open the
#: file. It over-counts slightly (a ``/Type /Pages`` node is not a page) and so
#: is deliberately given a small tolerance in :func:`count_pdf_pages`.
_PAGE_OBJECT_RE = re.compile(rb"/Type\s*/Page[^s]")


def count_pdf_pages(path: str) -> int | None:
    """How many pages a PDF declares, or ``None`` when that cannot be read.

    ``None`` means **unknown**, and admission treats unknown as allowed. That
    asymmetry is deliberate and is the only safe one: a limit that refuses
    everything it cannot measure refuses every encrypted PDF, every
    password-protected scan and every PDF whose xref a non-``pypdf`` build
    cannot follow — which is a product that does not work, in exchange for
    protecting against an attack that requires the file to be a *readable*
    many-thousand-page PDF, which the timeout layer already bounds.

    The byte-level fallback exists because a truncated or linearised PDF is
    exactly the case where ``pypdf`` gives up, and "I could not count the
    pages" is a worse answer than a count that is 1% high.
    """
    counted = _pdf_pages_via_pypdf(path)
    if counted is not None:
        return counted
    try:
        with open(path, "rb") as handle:
            blob = handle.read()
    except OSError:
        return None
    if b"%PDF-" not in blob[:1024]:
        return None
    # A document declaring more than a million page objects is not a document;
    # counting them all is the attack. The cap keeps the fallback linear in
    # something bounded.
    found = min(len(_PAGE_OBJECT_RE.findall(blob)), 1_000_000)
    # **Zero is not a page count, it is the absence of one.** Returning it would
    # tell admission this document is empty rather than unreadable, and the two
    # deserve different answers: an empty document is refused by the quality
    # gate, an unreadable one is the parser's problem to diagnose. This branch
    # is only reached when pypdf has already given up, so "no page objects
    # found" means the scan found nothing, not that the document has none.
    return found or None


# --------------------------------------------------------------------------
# Container (archive) admission
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ContainerReport:
    """What :func:`check_container` concluded about one file.

    ``depth`` is the deepest nesting level actually walked and is reported even
    on success, because "this DOCX contained no nested archive" is a fact a
    security review wants and a bare ``None`` return does not record.
    """

    is_container: bool
    entries: int = 0
    declared_uncompressed_bytes: int = 0
    compressed_bytes: int = 0
    depth: int = 0
    nested_members: tuple[str, ...] = ()

    @property
    def compression_ratio(self) -> int:
        """Declared-uncompressed over on-disk, floored at 1.

        Floored rather than divided: an archive that *expands* less than it
        occupies on disk has a ratio below 1, and comparing that against a
        ceiling is still a pass, but reporting ``0`` would read as "no data".
        """
        if self.compressed_bytes <= 0:
            return 1
        return max(1, self.declared_uncompressed_bytes // self.compressed_bytes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_container": self.is_container,
            "entries": self.entries,
            "declared_uncompressed_bytes": self.declared_uncompressed_bytes,
            "compressed_bytes": self.compressed_bytes,
            "compression_ratio": self.compression_ratio,
            "depth": self.depth,
            "nested_members": list(self.nested_members),
        }


def sniff_archive(raw: bytes) -> bool:
    """Whether ``raw`` starts with the magic of a known archive container."""
    head = bytes(raw or b"")[:ARCHIVE_MAGIC_SNIFF_BYTES]
    return any(head.startswith(prefix) for prefix in ARCHIVE_MAGIC_PREFIXES)


def is_safe_member_name(name: str, *, limits: ParserLimits | None = None) -> bool:
    """Whether a member name is safe to write out of an archive.

    A **superset** of ``adapters.mineru_cloud._is_safe_member_name`` rather than
    a copy of it, and deliberately so: that function is documented as a
    defence-in-depth check for a path that never writes to disk, and this one
    guards a path that inspects a container a user uploaded. The extra
    rejections are the ones a name can carry that a plain ``..`` check misses:

    * a **NUL byte**, which truncates the name in every C API a later
      extraction would use, so ``"ok.txt\\x00../../x"`` passes a string
      comparison and lands somewhere else entirely;
    * a **symlink entry**, recognised from ``ZipInfo.external_attr``'s high
      16 bits. The existing check judges the *name* and so cannot see it: a
      member called ``word/link`` with the symlink bit set is a name-safe path
      and a write-outside-the-tree payload. Nothing in this repository extracts
      an uploaded archive today, so this is latent — which is exactly why it
      belongs in the function whose docstring says "if you ever write these out,
      do not introduce a bug", rather than being rediscovered at the moment
      someone does.
    * an **empty or dot-only name**, which resolves to a directory.

    The ``..`` and absolute-path rules themselves are the existing module's and
    are duplicated here on purpose: this function must be correct when called on
    its own, and importing a private helper across a package boundary is how a
    guard quietly stops being called.
    """
    normalized = str(name or "").replace("\\", "/")
    if not normalized or normalized in {".", "./", "/"}:
        return False
    if "\x00" in normalized:
        return False
    if normalized.startswith("/"):
        return False
    # Windows drive-relative: ``C:foo`` and ``C:/foo`` both resolve against a
    # drive, not the extraction root.
    first = normalized.split("/", 1)[0]
    if len(first) >= 2 and first[1] == ":" and first[0].isalpha():
        return False
    return ".." not in normalized.split("/")


def _member_is_symlink(entry: zipfile.ZipInfo) -> bool:
    """Whether a zip entry carries the Unix symlink bit in ``external_attr``."""
    # The high 16 bits hold st_mode when the archive was written by a Unix
    # tool; 0o120000 is S_IFLNK. Windows-written archives leave them zero, so
    # this is False for them, which is the correct answer.
    return (entry.external_attr >> 16) & 0o170000 == 0o120000


def _check_absolute_dimensions(archive: zipfile.ZipFile, limits: ParserLimits) -> None:
    """Entry count, per-entry declared size, total declared size.

    One implementation, called from two places: directly when the MinerU
    adapter is not importable, and to apply an operator's *stricter*
    configured numbers on top of MinerU's looser ones.
    """
    entries = [entry for entry in archive.infolist() if not entry.is_dir()]
    if len(archive.infolist()) > limits.max_archive_entries:
        raise ParserLimitExceeded(
            f"压缩包条目过多（{len(archive.infolist())} > {limits.max_archive_entries}），"
            "疑似恶意压缩包。",
            code="archive_bomb",
            limit="max_archive_entries",
        )
    total = 0
    for entry in entries:
        if entry.file_size > limits.max_archive_entry_bytes:
            raise ParserLimitExceeded(
                f"压缩包中 {entry.filename} 声明解压后超过 "
                f"{limits.max_archive_entry_bytes // (1024 * 1024)}MB，疑似 zip 炸弹。",
                code="archive_bomb",
                limit="max_archive_entry_bytes",
            )
        total += entry.file_size
        if total > limits.max_archive_total_bytes:
            raise ParserLimitExceeded(
                "压缩包声明的总解压大小超过 "
                f"{limits.max_archive_total_bytes // (1024 * 1024)}MB，疑似 zip 炸弹。",
                code="archive_bomb",
                limit="max_archive_total_bytes",
            )


def _reject_reusing_mineru_dimensions(archive: zipfile.ZipFile, limits: ParserLimits) -> None:
    """Apply the three MinerU ceilings by *calling* them, not copying them.

    ``validate_zip_bomb`` is the tested implementation of "entry count, per
    entry, total declared", and duplicating its three constants here would
    create a second set of numbers that drifts. It speaks only the module-level
    ``MAX_ZIP_*`` values, so the stricter of "what MinerU allows" and "what this
    deployment configured" wins: :func:`_check_absolute_dimensions` runs first
    when the configured numbers are tighter, and MinerU's own check always runs
    afterwards so a relaxed local configuration cannot *widen* what the
    well-tested guard accepts.

    The import is local and guarded because ``mineru_cloud`` is an adapter whose
    optional-dependency story is not this module's to assume. On a build where it
    cannot import, :func:`_check_absolute_dimensions` is the whole check — and
    its defaults are documented in the module docstring as deliberately identical
    to the MinerU constants, so the two paths agree by construction rather than
    by coincidence.
    """
    try:
        from .adapters import mineru_cloud as mineru  # noqa: PLC0415 - see above
    except Exception:
        _check_absolute_dimensions(archive, limits)
        return

    if (
        limits.max_archive_entries < mineru.MAX_ZIP_ENTRIES
        or limits.max_archive_entry_bytes < mineru.MAX_ZIP_ENTRY_UNCOMPRESSED_BYTES
        or limits.max_archive_total_bytes < mineru.MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES
    ):
        _check_absolute_dimensions(archive, limits)
    try:
        mineru.validate_zip_bomb(archive)
    except Exception as exc:  # noqa: BLE001 - MineruError, translated not re-typed
        # The Chinese sentence carries over unchanged, which is the point: the
        # user sees the same diagnosis whichever layer produced it.
        raise ParserLimitExceeded(
            str(exc), code="archive_bomb", limit="mineru_zip_bomb"
        ) from None


def _walk_archive(
    archive: zipfile.ZipFile,
    *,
    limits: ParserLimits,
    budget: list[int],
    depth: int,
    nested_names: list[str],
) -> None:
    """Account one archive level against the **shared** budget, recursively.

    ``budget`` is a one-element list holding the remaining total-uncompressed
    allowance for *all* levels combined. A per-level budget is the specific bug
    this signature exists to prevent: three levels of 200 MB each is 600 MB of
    allocation from a 90 KB file, and each level on its own is comfortably
    inside a limit that says "256 MB".

    Depth is checked *before* descending, so a container nested past the ceiling
    is refused without its contents ever being opened.
    """
    entries = [entry for entry in archive.infolist() if not entry.is_dir()]
    budget[0] -= sum(entry.file_size for entry in entries)
    if budget[0] < 0:
        raise ParserLimitExceeded(
            "压缩包在嵌套展开后声明的总解压大小超过 "
            f"{limits.max_archive_total_bytes // (1024 * 1024)}MB，疑似嵌套 zip 炸弹。",
            code="archive_nested_bomb",
            limit="max_archive_total_bytes",
        )
    # The ceiling is tested *per member, here*, rather than as an early return
    # at the top of the level. An early return would stop this level from
    # inspecting its own members, which is exactly the level where a bomb
    # hides: with `max_archive_depth=1`, a zip holding a zip holding a bomb
    # would have the outer two levels opened and the third never looked at.
    for entry in entries:
        if not is_safe_member_name(entry.filename):
            raise ParserLimitExceeded(
                f"压缩包成员名不安全（{entry.filename!r}），已拒绝该文件。",
                code="archive_traversal",
                limit="member_name",
            )
        if _member_is_symlink(entry):
            raise ParserLimitExceeded(
                f"压缩包成员 {entry.filename!r} 是一个符号链接，已拒绝该文件。",
                code="archive_symlink",
                limit="member_symlink",
            )
        if entry.file_size > limits.max_archive_entry_bytes:
            raise ParserLimitExceeded(
                f"压缩包中 {entry.filename} 声明解压后超过 "
                f"{limits.max_archive_entry_bytes // (1024 * 1024)}MB，疑似 zip 炸弹。",
                code="archive_bomb", limit="max_archive_entry_bytes",
            )
        try:
            with archive.open(entry) as handle:
                head = handle.read(ARCHIVE_MAGIC_SNIFF_BYTES)
        except Exception:
            # An unreadable member is not evidence of a bomb. The declared size
            # has already been charged to the budget above, which is the part
            # that matters for resource safety.
            continue
        if not sniff_archive(head):
            continue
        nested_names.append(entry.filename)
        if depth + 1 > limits.max_archive_depth:
            raise ParserLimitExceeded(
                f"压缩包内含嵌套压缩包 {entry.filename!r}，嵌套层级超过 "
                f"{limits.max_archive_depth} 层，已拒绝该文件。",
                code="archive_nested_too_deep",
                limit="max_archive_depth",
            )
        if len(nested_names) > limits.max_archive_entries:
            raise ParserLimitExceeded(
                f"压缩包内含超过 {limits.max_archive_entries} 个嵌套压缩包，已拒绝该文件。",
                code="archive_nested_bomb", limit="max_archive_entries",
            )
        try:
            with archive.open(entry) as handle:
                payload = handle.read(min(entry.file_size, limits.max_archive_entry_bytes) + 1)
        except Exception:
            continue
        if len(payload) > limits.max_archive_entry_bytes:
            raise ParserLimitExceeded(
                f"压缩包内嵌压缩包 {entry.filename!r} 实际解压大小超过上限，疑似 zip 炸弹。",
                code="archive_bomb", limit="max_archive_entry_bytes",
            )
        try:
            with zipfile.ZipFile(io.BytesIO(payload)) as inner:
                _walk_archive(
                    inner, limits=limits, budget=budget, depth=depth + 1,
                    nested_names=nested_names,
                )
        except zipfile.BadZipFile:
            # Sniffed as an archive by magic but not parseable as one. That is a
            # truncated or non-zip container with a colliding magic number (an
            # OOXML file is a zip, so this is not hypothetical for a nested
            # member). Refusing would reject legitimate documents; the size has
            # been charged to the budget, so the resource case is covered.
            continue


def check_container(
    file_path: str | None = None,
    *,
    raw: bytes | None = None,
    limits: ParserLimits | None = None,
) -> ContainerReport:
    """Validate a file that is (or contains) an archive container.

    Keyed on **magic bytes, not extension**, which is the whole point of this
    function existing: ``.docx`` and ``.pptx`` are zip containers that
    :data:`config.ALLOWED_EXTENSIONS` admits, that
    :func:`~paperwhisperer.documents.validation.validate_document_file_signature`
    *requires* to begin with ``PK\\x03\\x04``, and that no archive check
    anywhere in this repository currently looks at. A 3.7 MB ``.docx`` that
    inflates to 4 GB passes the extension gate, the magic-byte gate and a 16 MB
    upload ceiling, and this is the function that stops it.

    A file that is not a container returns ``is_container=False`` and no
    exception — "this is not a zip" is a fact, not a refusal. Only a container
    that violates a ceiling raises.

    The four checks, in the order they run, and what each is for:

    1. **Declared total and ratio, shared across nesting levels.** The ratio is
       what actually discriminates: a 200-page Word document of XML
       compresses 30-80x, the bombs are 1000x+. The three absolute ceilings
       that already exist are all satisfiable by a bomb — four 60 MB members
       are 6 entries, 60 MB each and 240 MB total, inside every one of them,
       and inflate 1000x from a 230 KB upload.
    2. **Traversal and symlink member names**, at every level.
    3. **Nesting depth**, refused before the inner container is opened rather
       than after, and bounded against the same shared budget.
    4. **Per-entry declared size**, which the existing guard also has and which
       is retained so this function is safe to call on its own.

    ``raw`` is accepted so a caller holding bytes (a downloaded response, an
    in-memory upload) can check without writing them to disk first.
    """
    resolved = limits or limits_from_config()
    if raw is None:
        if not file_path or not os.path.isfile(file_path):
            return ContainerReport(is_container=False)
        try:
            with open(file_path, "rb") as handle:
                raw = handle.read(ARCHIVE_MAGIC_SNIFF_BYTES)
        except OSError:
            return ContainerReport(is_container=False)
    if not sniff_archive(raw):
        return ContainerReport(is_container=False)

    blob = (
        _read_capped_file(file_path, resolved.max_input_bytes)
        if file_path
        else bytes(raw or b"")
    )
    try:
        archive = zipfile.ZipFile(io.BytesIO(blob))
    except zipfile.BadZipFile:
        # Sniffed as a container by magic but not one. A truncated DOCX is a
        # parse failure with an excellent existing message; inventing an archive
        # error here would replace a good diagnosis with a confusing one.
        return ContainerReport(is_container=True, compressed_bytes=len(blob))

    nested: list[str] = []
    with archive:
        _reject_reusing_mineru_dimensions(archive, resolved)
        entries = [entry for entry in archive.infolist() if not entry.is_dir()]
        declared = sum(entry.file_size for entry in entries)
        compressed = os.path.getsize(file_path) if file_path else len(blob)
        report = ContainerReport(
            is_container=True,
            entries=len(entries),
            declared_uncompressed_bytes=declared,
            compressed_bytes=max(1, compressed),
        )
        if report.compression_ratio > resolved.max_compression_ratio:
            raise ParserLimitExceeded(
                f"压缩包解压比达到 {report.compression_ratio}:1，超过上限 "
                f"{resolved.max_compression_ratio}:1，疑似 zip 炸弹。",
                code="archive_bomb", limit="max_compression_ratio",
            )
        _walk_archive(
            archive, limits=resolved, budget=[resolved.max_archive_total_bytes],
            depth=0, nested_names=nested,
        )
    return ContainerReport(
        is_container=True,
        entries=report.entries,
        declared_uncompressed_bytes=report.declared_uncompressed_bytes,
        compressed_bytes=report.compressed_bytes,
        depth=min(resolved.max_archive_depth, 1 + (len(nested) > 0)),
        nested_members=tuple(nested),
    )


def _read_capped_file(file_path: str, max_bytes: int) -> bytes:
    """Read at most ``max_bytes`` for inspection, refusing a larger file.

    The ceiling is the *upload* ceiling rather than an archive-specific one,
    so this function can never be the reason a legal upload is rejected for
    being large: the size check has already happened by the time any container
    is inspected.
    """
    with open(file_path, "rb") as handle:
        blob = handle.read(max_bytes + 1)
    if len(blob) > max_bytes:
        raise ParserLimitExceeded(
            f"文件超过 {max_bytes // (1024 * 1024)}MB 上限，已拒绝。",
            code="input_too_large", limit="max_input_bytes",
        )
    return blob


# --------------------------------------------------------------------------
# Admission
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AdmissionVerdict:
    """Whether a file may be handed to a parser, and what was learned.

    A verdict rather than a bare bool so that a caller can report *why*
    something was allowed as well as why it was refused — "the page count is
    unknown, so the ceiling could not be applied" is a fact an operator needs
    and a bare ``True`` throws away.
    """

    allowed: bool
    code: str = "admitted"
    message: str = ""
    limit: str = ""
    observed: Any = None
    allowed_value: Any = None
    page_count: int | None = None
    container: ContainerReport | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "code": self.code,
            "message": self.message,
            "limit": self.limit,
            "observed": self.observed,
            "allowed_value": self.allowed_value,
            "page_count": self.page_count,
            "container": self.container.to_dict() if self.container else None,
        }


def admit_document(
    file_path: str,
    *,
    limits: ParserLimits | None = None,
    media_type: str = "",
) -> AdmissionVerdict:
    """Refuse a document that is too big before any parser is constructed.

    This is where ``MAX_DOCUMENT_PAGES`` stops being a documented intention.
    The order is cheapest-first and the order is the design: a stat, then a
    container check that reads at most the upload ceiling, then a page count.
    None of it constructs a converter, imports torch, or writes a cache entry,
    so a refused 4 000-page document costs a few milliseconds instead of the
    minutes it would cost to convert and then discover it was too big.

    The three refusals, and why each is worth making at all:

    * **bytes** — already enforced at ingress by
      :func:`~paperwhisperer.core.text.save_upload_file`, repeated here because
      this function is reachable from the remote-download path and from any
      caller that did not go through the route, and a limit that only exists on
      one of the two ingresses is the failure mode the test suite for
      ``tests/test_ingestion_limits.py`` was written to catch.
    * **container** — the archive defence, for the reason spelled out in
      :func:`check_container`.
    * **pages** — the ceiling the config has always claimed. Counted from the
      file's own bytes, so a document that lies about its length in its
      ``/Count`` does not get a free pass; ``pypdf``'s page tree is derived
      from the objects actually present.
    """
    resolved = limits or limits_from_config()

    try:
        size = os.path.getsize(file_path)
    except OSError as exc:
        return AdmissionVerdict(
            allowed=False, code="unreadable", limit="readable",
            message=f"无法读取待解析文件：{exc}。",
        )
    if size > resolved.max_input_bytes:
        return AdmissionVerdict(
            allowed=False, code="input_too_large", limit="max_input_bytes",
            observed=size, allowed_value=resolved.max_input_bytes,
            message=(
                f"文件大小 {size // (1024 * 1024)}MB 超过上限 "
                f"{resolved.max_input_bytes // (1024 * 1024)}MB，已拒绝。"
                "请拆分文档后重试。"
            ),
        )

    try:
        container = check_container(file_path, limits=resolved)
    except ParserLimitExceeded as exc:
        return AdmissionVerdict(
            allowed=False, code=exc.code, limit=exc.limit, message=str(exc),
        )

    pages: int | None = None
    suffix = os.path.splitext(file_path)[1].lower()
    if suffix == ".pdf" or (media_type or "").split(";")[0].strip() == "application/pdf":
        pages = count_pdf_pages(file_path)
    if pages is not None and pages > resolved.max_pages:
        return AdmissionVerdict(
            allowed=False, code="too_many_pages", limit="max_pages",
            observed=pages, allowed_value=resolved.max_pages, page_count=pages,
            container=container,
            message=(
                f"文档共 {pages} 页，超过单文档页数上限 {resolved.max_pages} 页，已拒绝解析。"
                "请拆分文档后重试，或调高 MAX_DOCUMENT_PAGES。"
            ),
        )

    return AdmissionVerdict(allowed=True, page_count=pages, container=container)


def enforce_output_limits(
    document: Any,
    *,
    limits: ParserLimits | None = None,
) -> None:
    """Refuse a parse whose *result* is larger than a document may be.

    The admission check bounds the input; this bounds the output, and the two
    are not the same number. A 40 MB PDF of nothing but vector art can expand
    into a parsed document with millions of elements, and the thing that
    actually exhausts memory in this product is the canonical
    :class:`~paperwhisperer.ingestion.schema.ParsedDocument` — every element is
    held in memory, then serialised into the raw artifact, then written into the
    parse cache. A limit on the result catches that at the only point where the
    size is knowable without having already paid for it.

    Two dimensions, because they fail differently: element count bounds the
    per-element overhead (a ``ParsedDocument`` with two million one-character
    elements costs orders of magnitude more than its character count suggests),
    and serialised size bounds the artefact and cache-write paths. Neither is a
    substitute for the other, which is why both are checked and the message
    names the one that fired.
    """
    resolved = limits or limits_from_config()
    elements = list(getattr(document, "elements", ()) or ())
    if len(elements) > resolved.max_archive_entries * 64:
        raise ParserLimitExceeded(
            f"解析结果包含 {len(elements)} 个元素，超过上限，文档可能异常。",
            code="output_too_large", limit="max_elements",
        )
    char_count = int(getattr(document, "char_count", 0) or 0)
    if char_count > resolved.max_output_bytes:
        raise ParserLimitExceeded(
            f"解析结果正文 {char_count} 个字符，超过 "
            f"{resolved.max_output_bytes // (1024 * 1024)}MB 上限。",
            code="output_too_large", limit="max_output_bytes",
        )


# --------------------------------------------------------------------------
# Duration
# --------------------------------------------------------------------------


class Deadline:
    """A monotonic wall-clock budget a parser can consult cooperatively.

    Exists because :meth:`~paperwhisperer.ingestion.gateway.ParserRouter.parse`
    can only *stop waiting* — the thread it abandoned is still running, and that
    is stated in the gateway's own docstring. A parser that has agreed to check
    :meth:`check` between pages can actually stop, and the page loop is where a
    long parse spends essentially all of its time.

    ``None`` from :meth:`remaining` means expired rather than "unbounded",
    because a caller that treats ``None`` as infinite will call
    :meth:`check` in a loop that never exits.
    """

    __slots__ = ("_started", "_budget", "_clock")

    def __init__(self, budget_seconds: float, *, clock: Callable[[], float] | None = None) -> None:
        self._started = (clock or time.monotonic)()
        self._budget = float(budget_seconds)
        self._clock = clock or time.monotonic

    @property
    def budget_seconds(self) -> float:
        return self._budget

    def elapsed(self) -> float:
        return self._clock() - self._started

    def remaining(self) -> float | None:
        left = self._budget - self.elapsed()
        return left if left > 0 else None

    def expired(self) -> bool:
        return self.remaining() is None

    def check(self, what: str = "解析") -> None:
        """Raise :class:`ParserLimitExceeded` if the budget is gone.

        The message names ``what`` because the caller knows which page, stage or
        file it was in and the gateway does not — "解析超时" alone sends an
        operator to the wrong log line.
        """
        if self.expired():
            raise ParserLimitExceeded(
                f"{what}超过 {self._budget:g} 秒的时限，已中止。",
                code="parser_timeout", limit="timeout_seconds",
            )


# --------------------------------------------------------------------------
# Observation (layer 1)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResourceUsage:
    """A sample of this process's footprint."""

    rss_bytes: int = 0
    peak_rss_bytes: int = 0
    cpu_seconds: float = 0.0
    #: CPU seconds since the guard started, high-water mark over the samples
    #: taken. Separate from ``cpu_seconds`` because the sampler's own bookkeeping
    #: runs in the parent while ``cpu_seconds`` is read from the process: they
    #: are the same number measured at different times, and conflating them makes
    #: the ceiling fire on a different parse than the one it was measured on.
    peak_cpu_seconds: float = 0.0
    wall_seconds: float = 0.0
    available: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "rss_bytes": self.rss_bytes,
            "peak_rss_bytes": self.peak_rss_bytes,
            "cpu_seconds": round(self.cpu_seconds, 3),
            "peak_cpu_seconds": round(self.peak_cpu_seconds, 3),
            "wall_seconds": round(self.wall_seconds, 3),
            "available": self.available,
        }


def _read_rss_bytes() -> int:
    """Resident set size of this process, or 0 when it cannot be read.

    ``/proc/self/statm`` is read because it is a single short line and does not
    allocate; ``/proc/self/status`` is a hundred lines and parsing it per sample
    is a cost the sampler pays every tick. ``resource.getrusage`` is the
    portable fallback and gives *peak* rather than current, which is enough for
    a ceiling check.
    """
    try:
        with open("/proc/self/statm", "rb", buffering=0) as handle:
            fields = handle.read().split()
        if len(fields) >= 2:
            return int(fields[1]) * _PAGE_SIZE
    except (OSError, ValueError, IndexError):
        pass
    try:
        import resource as _resource  # noqa: PLC0415 - POSIX only

        peak = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
        # Linux reports kilobytes; the BSDs report bytes. The module is
        # POSIX-only, so this is a two-way choice and not three.
        return int(peak) * (1024 if sys.platform.startswith("linux") else 1)
    except Exception:
        return 0


def _read_cpu_seconds() -> float:
    try:
        import resource as _resource  # noqa: PLC0415 - POSIX only

        usage = _resource.getrusage(_resource.RUSAGE_SELF)
        return float(usage.ru_utime + usage.ru_stime)
    except Exception:
        return 0.0


def sample_usage() -> ResourceUsage:
    """One sample of this process's memory and CPU."""
    rss = _read_rss_bytes()
    try:
        import resource as _resource  # noqa: PLC0415 - POSIX only

        peak = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
        peak_bytes = int(peak) * (1024 if sys.platform.startswith("linux") else 1)
    except Exception:
        peak_bytes = rss
    return ResourceUsage(
        rss_bytes=rss,
        peak_rss_bytes=peak_bytes,
        cpu_seconds=_read_cpu_seconds(),
        wall_seconds=time.monotonic(),
        available=rss > 0,
    )


class ResourceGuard:
    """Samples the process footprint while a parse runs and records the peak.

    **This cannot stop anything, and the docstring says so first.** A thread in
    this process cannot interrupt a thread that is inside ``libtorch``, inside
    a C ``malloc``, or inside a Python loop that never checks a flag; the only
    way to interrupt it is a signal, and the only signal that reliably stops an
    arbitrary thread is one that stops the whole process. So a guard that
    pretended to enforce a memory ceiling would be worse than no guard: it
    would produce a log line saying the limit was enforced when it was only
    noticed.

    What it does instead is turn an unbounded run into a **refusable result**.
    :meth:`verdict` is consulted after the parse, and a run that exceeded its
    memory or CPU budget produces a failed :class:`ParserLimitExceeded` instead
    of a good-looking document. That is weaker than not running it, and it is
    also the only thing achievable without a subprocess — which is exactly why
    :func:`run_guarded` exists and is the answer to "where is the enforcement".

    The sampler thread is a daemon, so a guard that is never closed cannot keep
    the interpreter alive, and it holds no lock the parse path could be waiting
    on.
    """

    def __init__(
        self,
        limits: ParserLimits | None = None,
        *,
        parser_name: str = "",
        enabled: bool = True,
    ) -> None:
        self.limits = limits or limits_from_config()
        self.parser_name = parser_name
        self.enabled = enabled
        self.samples: int = 0
        self.peak_rss_bytes = 0
        self.peak_cpu_seconds = 0.0
        self._start_usage: ResourceUsage | None = None
        self._stop = None  # type: ignore[assignment]
        self._thread: Any = None

    def __enter__(self) -> ResourceGuard:
        return self.start()

    def __exit__(self, *exc_info: Any) -> bool:
        self.stop()
        return False

    def start(self) -> ResourceGuard:
        import threading  # noqa: PLC0415 - only needed when a guard is used

        self._start_usage = sample_usage()
        self.peak_rss_bytes = self._start_usage.rss_bytes
        self.peak_cpu_seconds = self._start_usage.cpu_seconds
        if not self.enabled or self.limits.sample_interval_seconds <= 0:
            return self
        self._stop = threading.Event()

        def loop() -> None:
            while not self._stop.is_set():  # type: ignore[union-attr]
                if self._stop.wait(self.limits.sample_interval_seconds):  # type: ignore[union-attr]
                    return
                current = sample_usage()
                if not current.available:
                    continue
                self.samples += 1
                self.peak_rss_bytes = max(self.peak_rss_bytes, current.rss_bytes)
                self.peak_cpu_seconds = max(
                    self.peak_cpu_seconds, current.cpu_seconds - self._start_usage.cpu_seconds
                )

        self._thread = threading.Thread(
            target=loop, name="pw-resource-guard", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._stop is not None:
            self._stop.set()
            self._stop = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def usage(self) -> ResourceUsage:
        """The peak observed, with wall and CPU measured against the start."""
        current = sample_usage()
        if self._start_usage is None:
            return current
        return ResourceUsage(
            rss_bytes=current.rss_bytes,
            peak_rss_bytes=self.peak_rss_bytes or current.rss_bytes,
            cpu_seconds=max(
                0.0, current.cpu_seconds - self._start_usage.cpu_seconds
            ),
            peak_cpu_seconds=self.peak_cpu_seconds,
            wall_seconds=max(0.0, current.wall_seconds - self._start_usage.wall_seconds),
            available=current.available,
        )

    def verdict(self) -> None:
        """Raise if the observed peak exceeded a configured ceiling.

        Compared against **peak**, not current: a parser that allocated 8 GB and
        freed it has already done the damage, and a check on the current
        footprint would report success. Compared against the process total
        rather than a delta, because a guard cannot attribute a shared
        interpreter's memory to one parse, and pretending otherwise would
        produce a limit that fires on a big document because something else
        happened to be resident.
        """
        observed = self.usage()
        if not observed.available:
            return
        if observed.peak_rss_bytes > self.limits.max_memory_bytes:
            raise ParserLimitExceeded(
                f"解析器 {self.parser_name or '未知'} 解析期间常驻内存达到 "
                f"{observed.peak_rss_bytes // (1024 * 1024)}MB，超过 "
                f"{self.limits.max_memory_bytes // (1024 * 1024)}MB 上限，已拒绝该结果。",
                code="memory_limit_exceeded", limit="max_memory_bytes",
            )
        if observed.peak_cpu_seconds > self.limits.max_cpu_seconds:
            raise ParserLimitExceeded(
                f"解析器 {self.parser_name or '未知'} 解析期间消耗 CPU "
                f"{observed.peak_cpu_seconds:.1f} 秒，超过 "
                f"{self.limits.max_cpu_seconds} 秒上限，已拒绝该结果。",
                code="cpu_limit_exceeded", limit="max_cpu_seconds",
            )


# --------------------------------------------------------------------------
# Hard isolation (layer 2)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SandboxOutcome:
    """What a guarded run produced, including *how* it ended.

    ``killed_by`` is the field that makes a sandbox incident diagnosable. A
    parse that died because the kernel sent ``SIGKILL`` for ``RLIMIT_CPU`` looks
    from the outside identical to one that was OOM-killed by the parent for
    exceeding the wall clock, and the two have completely different causes.
    """

    completed: bool
    value: Any = None
    error: BaseException | None = None
    killed_by: str = ""
    wall_seconds: float = 0.0
    child_pid: int = 0
    exit_status: int = 0

    def unwrap(self) -> Any:
        """The return value, or a raised error, or a limit refusal.

        A :class:`MemoryError` is translated rather than propagated, and the
        translation is the useful part. A ``MemoryError`` raised inside a child
        that has a known ``RLIMIT_AS`` is overwhelmingly the rlimit firing, and
        propagating it raw would give the caller an exception type that means
        "this machine is out of memory" — sending an operator to look at the
        box instead of at ``PARSER_MAX_MEMORY_MB``. The raw ``MemoryError`` is
        kept as the cause so the original traceback is not lost.

        The refusal for every other ending is synthesised here rather than in
        the child because the child cannot report *why* it died — it is dead,
        or it is a string on a pipe. Deciding the diagnosis in the parent, where
        the ``waitpid`` status and the wall clock are both in scope, is the only
        place the information exists.
        """
        if self.completed:
            return self.value
        if isinstance(self.error, MemoryError):
            raise ParserLimitExceeded(
                "解析在受限子进程中被系统终止（内存上限），已放弃该次解析。",
                code="memory_limit_exceeded",
                limit="max_memory_bytes",
            ) from self.error
        if self.error is not None:
            raise self.error
        if self.killed_by in {"memory", "cpu"}:
            limit = "max_memory_bytes" if self.killed_by == "memory" else "max_cpu_seconds"
            code = "memory_limit_exceeded" if self.killed_by == "memory" else "cpu_limit_exceeded"
            raise ParserLimitExceeded(
                "解析在受限子进程中被系统终止"
                + ("（内存上限）" if self.killed_by == "memory" else "（CPU 上限）")
                + "，已放弃该次解析。",
                code=code, limit=limit,
            )
        if self.killed_by == "wall_clock":
            raise ParserLimitExceeded(
                "解析超过时限，已终止受限子进程。",
                code="parser_timeout", limit="timeout_seconds",
            )
        raise ParserLimitExceeded(
            "解析在受限子进程中异常终止，已放弃该次解析。",
            code="sandbox_failed", limit="sandbox",
        )


def _apply_child_rlimits(limits: ParserLimits) -> None:
    """Install the ceilings **in this process, which must be the child**.

    The precondition is checked rather than assumed. ``setrlimit`` is a
    process-wide, one-way property: an ``RLIMIT_AS`` installed into the parent
    would make the *next request the web worker serves* raise
    :class:`MemoryError`, and nothing in this process can undo it. The guard
    below is therefore not a nicety — it is the assertion that makes calling
    this function outside :func:`run_guarded` a loud error rather than an
    outage.
    """
    if not _in_sandbox_child():
        raise ParserLimitExceeded(
            "内部错误：资源上限只能在受限子进程中安装，"
            "不能在 Web 服务进程内调用（否则一次解析超限会拖垮整个服务）。",
            code="rlimit_in_serving_process", limit="sandbox",
        )
    import resource as _resource  # noqa: PLC0415 - POSIX only

    # AS before the others: it is the one that turns a runaway allocation into a
    # MemoryError instead of an OOM kill of the whole box.
    _resource.setrlimit(_resource.RLIMIT_AS, (limits.max_memory_bytes, limits.max_memory_bytes))
    _resource.setrlimit(_resource.RLIMIT_CPU, (limits.max_cpu_seconds, limits.max_cpu_seconds))
    _resource.setrlimit(
        _resource.RLIMIT_NPROC, (limits.max_processes, limits.max_processes)
    )
    _resource.setrlimit(
        _resource.RLIMIT_FSIZE,
        (limits.max_output_bytes, limits.max_output_bytes),
    )


#: The pid of the process that entered :func:`run_guarded`. A pid rather than a
#: flag because a flag survives into a further fork, and a grandchild of a
#: sandbox is not the thing whose rlimits were configured.
_SANDBOX_PARENT_PID = 0


def _in_sandbox_child() -> bool:
    return bool(_SANDBOX_PARENT_PID) and os.getpid() != _SANDBOX_PARENT_PID


def sandbox_available() -> bool:
    """Whether this platform can install the rlimits :func:`run_guarded` needs.

    ``False`` on Windows, where :mod:`resource` exists but ``RLIMIT_AS`` and
    ``RLIMIT_CPU`` are not, and on any platform where :func:`os.fork` is absent.
    Callers should treat ``False`` as "layer 2 is not available here" rather
    than as an error, because layers 0 and 1 remain correct without it.
    """
    if not hasattr(os, "fork"):
        return False
    try:
        import resource as _resource  # noqa: PLC0415
    except ImportError:  # pragma: no cover - Windows
        return False
    return hasattr(_resource, "RLIMIT_AS") and hasattr(_resource, "RLIMIT_CPU")


def run_guarded(
    fn: Callable[..., Any],
    /,
    *args: Any,
    limits: ParserLimits | None = None,
    timeout_seconds: float | None = None,
    **kwargs: Any,
) -> Any:
    """Run ``fn`` in a forked child under real rlimits, and return its result.
    This is the only enforcement in this module that can *stop* anything, and
    the reason it is opt-in is in the module docstring. Read that before
    enabling it.

    The mechanics, in the order they matter:

    * **The child gets the ceilings; the parent never does.** ``fork()`` copies
      the process, ``setrlimit`` is called on the copy, and the parent's limits
      are untouched. This is the entire reason a hard memory ceiling is
      survivable in a single-process deployment.
    * **The result crosses a pipe, not shared memory.** Pickle is the only
      channel, so a child that wedges holding a lock cannot corrupt the
      parent, and the only thing that can hang the parent is a pipe the child
      never fills — which the wall clock handles.
    * **The wall clock is enforced by the parent**, with ``SIGKILL`` rather than
      ``SIGTERM``, because a parser blocked in an uninterruptible C call ignores
      ``SIGTERM`` and a half-dead child holding a torch model is worse than a
      dead one.
    * **The diagnosis is made in the parent.** ``RLIMIT_CPU`` delivers
      ``SIGXCPU``, and if the process ignores that too, ``SIGKILL``. The child's
      own ``MemoryError`` arrives as a string on the pipe, since the child
      cannot report its own death.

    The child never returns normally: it ``os._exit``s so that no ``atexit``
    handler inherited from the parent — a database flush, a logging shutdown, a
    temp-file cleanup — runs in a process that is holding half the parent's
    state.
    """
    if not sandbox_available():  # pragma: no cover - Windows CI
        raise ParserLimitExceeded(
            "当前平台不支持子进程资源限制，无法在隔离环境中解析该文档。",
            code="sandbox_unavailable", limit="sandbox",
        )
    resolved = limits or limits_from_config()
    budget = float(timeout_seconds if timeout_seconds is not None else resolved.timeout_seconds)
    if budget <= 0:
        raise ParserLimitExceeded(
            "解析时限必须为正数。", code="parser_timeout", limit="timeout_seconds",
        )

    global _SANDBOX_PARENT_PID  # noqa: PLW0603 - one process-wide marker, see above

    read_fd, write_fd = os.pipe()
    started = time.monotonic()
    pid = os.fork()
    if pid == 0:  # pragma: no cover - the child never returns; see _child_main
        os.close(read_fd)
        _SANDBOX_PARENT_PID = os.getppid()
        try:
            _apply_child_rlimits(resolved)
        except BaseException as exc:  # noqa: BLE001 - the child reports and dies
            _write_frame(write_fd, ("error", exc))
            os._exit(1)
        _run_child_body(fn, args, kwargs, write_fd)
        os._exit(0)

    os.close(write_fd)
    frame, timed_out = _read_frame(read_fd, budget)
    elapsed = time.monotonic() - started
    if timed_out:
        _kill(pid)
    _, status = os.waitpid(pid, 0)
    os.close(read_fd)

    outcome = SandboxOutcome(
        completed=frame is not None and frame[0] == "ok",
        value=frame[1] if frame is not None and frame[0] == "ok" else None,
        error=frame[1] if frame is not None and frame[0] == "error" else None,
        killed_by=_classify_exit(frame, status, timed_out=timed_out),
        wall_seconds=elapsed,
        child_pid=pid,
        exit_status=status,
    )
    if frame is None and not timed_out and status == 0:
        # The child exited cleanly without writing a frame, which happens when
        # the rlimit killed it between the two statements. Report it as what it
        # is rather than as a generic failure.
        outcome = replace(outcome, killed_by="memory")
    return outcome.unwrap()


def _run_child_body(  # pragma: no cover - runs in the forked child
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    write_fd: int,
) -> None:
    try:
        result = fn(*args, **kwargs)
    except MemoryError as exc:
        _write_frame(write_fd, ("error", exc))
    except BaseException as exc:  # noqa: BLE001 - the parent decides what it means
        _write_frame(write_fd, ("error", exc))
    else:
        try:
            _write_frame(write_fd, ("ok", result))
        except BaseException:  # noqa: BLE001 - an unpicklable result is a failure
            _write_frame(write_fd, ("error", ParserLimitExceeded(
                "解析结果无法跨进程传递，该解析器不适合在隔离模式下运行。",
                code="sandbox_unpicklable", limit="sandbox",
            )))
    finally:
        os.close(write_fd)


def _write_frame(fd: int, frame: tuple[str, Any]) -> None:  # pragma: no cover - child side
    try:
        payload = pickle.dumps(frame, protocol=pickle.HIGHEST_PROTOCOL)
    except BaseException:  # noqa: BLE001
        payload = pickle.dumps(
            ("error", ParserLimitExceeded(
                "解析结果无法序列化。", code="sandbox_unpicklable", limit="sandbox"
            )),
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    # One write, so two frames can never interleave. A short write is retried
    # because a pipe write of a few hundred KB is routinely partial.
    written = 0
    while written < len(payload):
        try:
            written += os.write(fd, payload[written:])
        except BrokenPipeError:
            return


def _read_frame(fd: int, budget: float) -> tuple[tuple[str, Any] | None, bool]:
    """Read one frame from the pipe. Returns ``(frame, timed_out)``.

    The two-outcome return exists because "the deadline expired" and "the child
    is gone and wrote nothing" are different failures with different fixes, and
    collapsing them into one ``None`` makes a ``RLIMIT_CPU`` kill report itself
    as a timeout — sending an operator to raise ``PARSER_TIMEOUT_SECONDS`` when
    the real answer is that the parse burned its CPU budget. EOF on the read
    end is the signal that distinguishes them: the child died, closing its copy
    of the write end, so ``select`` reports readable and ``os.read`` returns
    zero bytes.

    ``select`` rather than a blocking read with a timer, because there is no
    portable way to put a timeout on a blocking read and a read that blocks
    forever is precisely the failure this function exists to prevent.
    """
    deadline = time.monotonic() + budget
    chunks: list[bytes] = []
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None, True
        try:
            readable, _, _ = select.select([fd], [], [], remaining)
        except (OSError, ValueError):  # pragma: no cover - fd closed under us
            return None, False
        if not readable:
            return None, True
        try:
            chunk = os.read(fd, 1 << 16)
        except OSError:
            return None, False
        if not chunk:
            # EOF: the child closed the pipe. It exited without a frame.
            break
        chunks.append(chunk)
    if not chunks:
        return None, False
    try:
        return pickle.loads(b"".join(chunks)), False  # noqa: S301 - our own child's frame
    except Exception:  # noqa: BLE001 - a truncated frame is a failed sandbox
        return None, False


def _kill(pid: int) -> None:
    for sig in (signal.SIGKILL,):
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            return


def _classify_exit(
    frame: tuple[str, Any] | None,
    status: int,
    *,
    timed_out: bool,
) -> str:
    """Name the reason a sandboxed run ended, or ``""`` if it succeeded.

    ``os.waitpid`` status encodes the signal in the low bits when
    ``os.WIFSIGNALED`` is true. ``RLIMIT_CPU`` sends ``SIGXCPU`` (24 on Linux);
    the kernel escalates to ``SIGKILL`` (9) if the process ignores it, so
    ``SIGKILL`` is treated as "CPU" only when the wall clock did not fire —
    which is the only way to tell the two apart, since both end in signal 9.
    """
    if frame is not None:
        return ""
    if timed_out:
        return "wall_clock"
    if os.WIFSIGNALED(status):
        signum = os.WTERMSIG(status)
        if signum in (signal.SIGXCPU, signal.SIGKILL):
            return "cpu"
        return f"signal_{signum}"
    if os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0:
        return "memory"
    return f"exit_{os.WEXITSTATUS(status) if os.WIFEXITED(status) else status}"


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def limits_report(limits: ParserLimits | None = None) -> dict[str, Any]:
    """A diagnostics block combining the ceilings, the platform and the guards.

    ``sandbox_available`` is reported next to ``isolation_enabled`` because the
    combination is what an operator needs: isolation *requested* on a platform
    that cannot provide it is a silent no-op, and reporting only the setting
    would make it look enforced.
    """
    resolved = limits or limits_from_config()
    return {
        "limits": resolved.describe(),
        "sandbox_available": sandbox_available(),
        "isolation_effective": resolved.isolation_enabled and sandbox_available(),
        "layer_1_detection_only": True,
        "secrets_in_this_report": False,
    }
