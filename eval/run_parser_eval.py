"""Parser bake-off runner: one corpus, every parser, one comparable report.

Task T0.9 asks for the same corpus to be pushed through legacy / Docling /
MinerU cloud and for the resulting numbers to choose the default routing. The
acceptance criterion is not "a report exists" but "the report is
reproducible", which is the constraint everything below is arranged around.

Why the runner is shaped this way
---------------------------------

1. **The report names the corpus it ran against.** A bake-off number without
   the corpus it came from cannot be compared to the next bake-off number, so
   every report carries a ``corpus_version``. The stamped value is the
   manifest's *own* ``corpus_version`` field — the one
   ``eval/build_manifest.py`` reproduces byte for byte on an unchanged corpus,
   and the one any future CI check or colleague will read. A manifest without
   that field produces a report that says so out loud, rather than quietly
   filling in a locally-computed number under a name that promises provenance.
   The runner's own digest over the entries it loaded travels beside it in a
   separately-named field, as a cross-check on the loader's view of the file
   set — never as a second identity for the same thing.

2. **Parsers are called directly, not through ``ParserRouter``.** The router's
   job is to stop at the first usable result — precisely the behaviour a bake-
   off has to suppress, because a bake-off needs every backend to see every
   document. Routing *policy* is what this run produces evidence for; the
   runner must not import it as an input.

3. **An absent dependency is a result, not an omission.** If docling is not
   installed, or no MinerU token is configured, the report contains a row for
   every corpus document with status ``unavailable`` and a reason code, and a
   dedicated "parsers that could not run" section. A blank column in a
   comparison table is unreadable — a reader cannot tell "docling found
   nothing" from "docling was never asked" — and a report that quietly
   compares two parsers on Monday and three on Tuesday invites a routing
   decision drawn from a table that changed shape.

4. **Nothing expensive happens by default.** ``--parsers`` defaults to
   ``legacy`` alone, the MinerU slot additionally requires a token, a
   ``privacy_mode=allow_cloud`` policy and an explicit ``--allow-cloud``, and
   the availability check for docling uses ``importlib.util.find_spec`` rather
   than ``import docling`` so that merely *asking* whether docling is present
   cannot drag torch into the process.

Everything the runner measures is derived from the canonical
:class:`~paperwhisperer.ingestion.schema.ParsedDocument` — the same value the
product will consume. That is deliberate: a bake-off that scored parser-native
output would be comparing two different definitions of "correct", and the
number that ships is the number measured here.

What this module does *not* do: it does not pick a default parser. The
routing decision is written by a human into ``docs/PARSER_BAKE_OFF.md`` after
looking at a real run; see the file for the template and the gaps.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ``eval/`` is a sibling of the package, not a subpackage of it, and it has no
# ``__init__.py``. Bootstrapping the repo root here is what lets both
# ``python eval/run_parser_eval.py`` and ``python -m eval.run_parser_eval``
# work; without it the first form cannot import ``paperwhisperer`` at all.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml  # noqa: E402

from paperwhisperer.core import config  # noqa: E402
from paperwhisperer.ingestion.gateway import (  # noqa: E402
    ParseOptions,
    media_type_for_path,
)
from paperwhisperer.ingestion.quality import evaluate_quality  # noqa: E402
from paperwhisperer.ingestion.schema import ParsedDocument, SourceLocator  # noqa: E402

#: Bumped when the set of measured metrics or the report layout changes, so a
#: stored report can be read against the runner that produced it.
RUNNER_VERSION = "1.0"

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = EVAL_DIR / "manifest.yaml"
DEFAULT_REPORT_DIR = EVAL_DIR / "reports"

#: The manifest schema family this loader understands. The trailing version is
#: deliberately not pinned to "0.1": a corpus that grew a field is still
#: readable, whereas a differently-shaped file is not.
MANIFEST_SCHEMA_PREFIX = "paperwhisperer-eval-manifest/"

#: Prefix the manifest's own ``corpus_version`` carries
#: (``sha256:<64 hex>``), as written by ``eval/build_manifest.py``. A value
#: that does not match is a manifest the runner cannot cite, so it is an
#: error rather than something to pass through into a report.
CORPUS_VERSION_PREFIX = "sha256:"

#: Category name given to the manifest's top-level ``adversarial`` list. They
#: are loaded but excluded from a batch run; see :func:`load_manifest`.
ADVERSARIAL_CATEGORY = "adversarial"

#: Canonical parser names, in the order the report lists them. ``docling`` and
#: ``docling_serve`` are the same adapter in two execution modes and are
#: reported separately because they are different deployments with different
#: latency and different failure modes.
PARSER_LEGACY = "legacy"
PARSER_DOCLING = "docling"
PARSER_DOCLING_SERVE = "docling_serve"
PARSER_MINERU = "mineru_cloud"
KNOWN_PARSERS = (PARSER_LEGACY, PARSER_DOCLING, PARSER_DOCLING_SERVE, PARSER_MINERU)

#: Per-document outcomes. ``success`` and ``partial`` are usable documents;
#: the rest are, in order, "this parser does not handle this format", "this
#: parser is not installed / not permitted" and "this parser ran and failed".
#: They are kept apart because only one of the four justifies a fallback.
STATUS_UNAVAILABLE = "unavailable"
STATUS_UNSUPPORTED = "unsupported"
STATUS_OK = "success"
STATUS_DEGRADED = "partial"
STATUS_BROKEN = "failed"
ALL_STATUSES = (
    STATUS_OK,
    STATUS_DEGRADED,
    STATUS_BROKEN,
    STATUS_UNSUPPORTED,
    STATUS_UNAVAILABLE,
)
#: Statuses that mean the parser was actually asked to parse the file.
ATTEMPTED_STATUSES = (STATUS_OK, STATUS_DEGRADED, STATUS_BROKEN)

#: Ratio metrics compared by ``--fail-on-regression``. Duration is excluded on
#: purpose: it is the one number whose movement is dominated by machine load
#: rather than by parser behaviour, so gating on it would produce CI noise
#: rather than signal.
REGRESSION_METRICS = (
    "mean_page_coverage",
    "mean_reading_order",
    "mean_locator_coverage",
    "mean_bbox_locator_ratio",
    "table_2d_ratio",
    "formula_latex_ratio",
    "failure_rate",
)

#: Decimal places every ratio is rounded to before it reaches a report.
#: Full float repr is noise in a diff; these are ratios, not measurements.
RATIO_PLACES = 4


class ManifestError(ValueError):
    """``manifest.yaml`` is not something this loader can trust.

    Separate from ``ValueError`` so the CLI can report a bad corpus as a
    configuration problem rather than a crash, and so a test can assert the
    failure mode without matching on message text.
    """


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """One corpus document, as registered in ``manifest.yaml``.

    Only the fields the runner needs are promoted to attributes; the rest of
    the entry (preview text, per-format counts, the Chinese note) stays
    reachable through :attr:`extra` so a future metric can use them without
    re-parsing the YAML.
    """

    category: str
    file: str
    sha256: str
    source: str = ""
    bytes: int = 0
    pages: int = 0
    text_layer: bool = False
    chars_total: int = 0
    preview: str = ""
    batch_excluded: bool = False
    extra: dict[str, Any] = field(default_factory=dict)
    #: Corpus root this entry resolves against, absolute. Filled in by
    #: :func:`load_manifest`; overridable per run so a caller can point a
    #: real manifest at a relocated corpus.
    root: str = ""

    @property
    def path(self) -> Path:
        """Absolute path of the file this entry describes."""
        if not self.root:
            raise ValueError(
                f"条目 {self.file} 没有 corpus root；请通过 load_manifest() 加载条目。"
            )
        return Path(self.root) / self.file

    @property
    def is_scanned(self) -> bool:
        """Whether the manifest says this file has no usable text layer."""
        return self.source == "synthetic" and not self.text_layer

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "file": self.file,
            "sha256": self.sha256,
            "source": self.source,
            "bytes": self.bytes,
            "pages": self.pages,
            "text_layer": self.text_layer,
            "chars_total": self.chars_total,
            "preview": self.preview,
            "batch_excluded": self.batch_excluded,
        }


@dataclass(frozen=True, slots=True)
class Manifest:
    """The loaded corpus registry.

    Carries **two** version strings, and they are not the same number:

    ``version``
        The manifest's own ``corpus_version`` field — the authoritative
        identity, computed by ``eval/build_manifest.py`` over category names,
        quotas and every file's content digest. It is the one a report stamps
        and the one any other tooling (a future CI check, a colleague
        re-running the bake-off) will read. Empty only for a hand-written
        manifest that predates the field.

    ``computed_version``
        This runner's own digest over the same ``(file, sha256)`` pairs. It
        exists as a **cross-check**, not as an alternative identity: the two
        are computed over deliberately different inputs, so they will not
        match, and presenting a second hash under the same name would be the
        fastest way to make a reproducibility claim that cannot be verified.
        What it can verify is coverage — if this runner had silently dropped
        entries while loading, the two inputs would diverge in a way a human
        can see in the report.
    """

    schema: str
    created: str
    corpus_root: Path
    entries: tuple[ManifestEntry, ...]
    summary: dict[str, Any] = field(default_factory=dict)
    source_path: Path = DEFAULT_MANIFEST
    version: str = ""
    computed_version: str = ""

    @property
    def regular_entries(self) -> tuple[ManifestEntry, ...]:
        """Entries that take part in a batch run.

        ``adversarial/`` files are excluded because they are supposed to
        fail — folded into a failure rate they would make every parser look
        equally broken. They are not deleted either: a report that cannot
        name which files it skipped cannot be audited, and ``--include-excluded``
        is there to run them deliberately.
        """
        return tuple(entry for entry in self.entries if not entry.batch_excluded)

    def select(
        self,
        *,
        categories: Sequence[str] = (),
        limit: int | None = None,
        include_excluded: bool = False,
    ) -> tuple[ManifestEntry, ...]:
        """Filter to a runnable subset, in a deterministic order.

        Sorting by ``(category, file)`` before applying ``limit`` matters:
        ``--limit 5`` has to mean the same five documents on every machine,
        otherwise a smoke run and a full run are not comparable and the
        smoke run's numbers quietly become the baseline.
        """
        pool = self.entries if include_excluded else self.regular_entries
        if categories:
            wanted = {str(name).strip() for name in categories if str(name).strip()}
            pool = tuple(entry for entry in pool if entry.category in wanted)
        ordered = tuple(sorted(pool, key=lambda entry: (entry.category, entry.file)))
        if limit is not None and limit >= 0:
            ordered = ordered[:limit]
        return ordered


def _as_int(value: Any, default: int = 0) -> int:
    """Coerce a manifest scalar to ``int`` without raising on ``None``/``"-"``."""
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _entry_from_mapping(
    payload: Any,
    *,
    category: str,
    where: str,
    root: str = "",
) -> ManifestEntry:
    """Build one :class:`ManifestEntry`, or explain precisely what is wrong.

    The error names the category and index it came from. A loader that says
    "entry 3 is invalid" makes rebuilding a 600-line manifest a guessing game.
    """
    if not isinstance(payload, dict):
        raise ManifestError(f"{where} 不是映射（得到 {type(payload).__name__}）。")
    file_name = str(payload.get("file") or "").strip()
    if not file_name:
        raise ManifestError(f"{where} 缺少必填字段 file。")
    sha256 = str(payload.get("sha256") or "").strip()
    if not sha256:
        # The corpus version hash is built from this field, so a missing one
        # would make the reproducibility stamp a lie rather than merely thin.
        raise ManifestError(f"{where}（{file_name}）缺少必填字段 sha256。")
    known = {
        "file", "sha256", "source", "bytes", "pages", "text_layer",
        "chars_total", "preview", "batch_excluded",
    }
    return ManifestEntry(
        category=category,
        file=file_name,
        sha256=sha256,
        source=str(payload.get("source") or ""),
        bytes=_as_int(payload.get("bytes")),
        pages=_as_int(payload.get("pages")),
        text_layer=_as_bool(payload.get("text_layer")),
        chars_total=_as_int(payload.get("chars_total")),
        preview=str(payload.get("preview") or ""),
        batch_excluded=_as_bool(payload.get("batch_excluded")),
        extra={key: value for key, value in payload.items() if key not in known},
        root=root,
    )


def load_manifest(
    path: str | os.PathLike[str] | None = None,
    *,
    corpus_root: str | os.PathLike[str] | None = None,
) -> Manifest:
    """Read ``manifest.yaml`` into a :class:`Manifest`.

    :raises ManifestError: on a missing file, a non-mapping document, a schema
        this loader does not understand, a missing ``categories`` block, or an
        entry without ``file``/``sha256``. Every one of those is a corpus a
        human has to fix, so failing loudly beats reporting a partial bake-off
        that looks complete.
    """
    manifest_path = Path(path) if path is not None else DEFAULT_MANIFEST
    if not manifest_path.is_file():
        raise ManifestError(f"找不到语料清单：{manifest_path}")
    try:
        payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ManifestError(f"manifest.yaml 不是合法 YAML：{exc}") from exc

    if not isinstance(payload, dict):
        raise ManifestError(f"manifest.yaml 顶层应为映射（得到 {type(payload).__name__}）。")

    schema = str(payload.get("schema") or "").strip()
    if not schema:
        raise ManifestError("manifest.yaml 缺少 schema 字段。")
    if not schema.startswith(MANIFEST_SCHEMA_PREFIX):
        raise ManifestError(
            f"不支持的 manifest schema：{schema!r}（期望 {MANIFEST_SCHEMA_PREFIX}*）"
        )

    categories = payload.get("categories")
    if not isinstance(categories, dict) or not categories:
        raise ManifestError("manifest.yaml 缺少非空的 categories 映射。")

    declared_root = payload.get("corpus_root")
    root_text = str(corpus_root) if corpus_root is not None else str(declared_root or "corpus")
    root_path = Path(root_text)
    if not root_path.is_absolute():
        # Relative to the manifest, not the cwd: the manifest is the thing
        # that says where the corpus is, and a report run from another
        # directory must still find the same files.
        root_path = (manifest_path.parent / root_path).resolve()
    resolved_root = str(root_path)

    entries: list[ManifestEntry] = []
    for category_name, category_payload in categories.items():
        where = f"categories.{category_name}"
        if not isinstance(category_payload, dict):
            raise ManifestError(f"{where} 不是映射。")
        category_entries = category_payload.get("entries")
        if category_entries is None:
            continue
        if not isinstance(category_entries, list):
            raise ManifestError(f"{where}.entries 应为列表。")
        for index, raw_entry in enumerate(category_entries):
            entries.append(_entry_from_mapping(
                raw_entry,
                category=str(category_name),
                where=f"{where}.entries[{index}]",
                root=resolved_root,
            ))

    # ``adversarial`` is a top-level list rather than a category, because the
    # plan treats it as a separate bucket: the files are supposed to fail, so
    # folding them into a category average would read as every parser being
    # broken. They are still loaded — a bake-off that cannot point at them
    # cannot verify the error paths — and marked excluded, so
    # ``--include-excluded`` is the single switch that opts into running them.
    adversarial = payload.get("adversarial")
    if isinstance(adversarial, list):
        for index, raw_entry in enumerate(adversarial):
            entry = _entry_from_mapping(
                raw_entry,
                category=ADVERSARIAL_CATEGORY,
                where=f"adversarial[{index}]",
                root=resolved_root,
            )
            entries.append(replace(entry, batch_excluded=True))

    # ``corpus_version`` is the manifest's own authoritative identity, written
    # by ``eval/build_manifest.py``. It is read from the top level on purpose:
    # a consumer should not have to reconstruct the manifest generator's idea
    # of what "version" means before it can check one. The ``or`` fallback
    # covers a hand-written manifest from before the field existed, and is
    # left empty rather than filled with this runner's own hash, because a
    # report that cannot tell where its version string came from is not
    # reproducible — it is merely labelled.
    declared_version = str(payload.get("corpus_version") or "").strip()
    if declared_version and not declared_version.startswith(CORPUS_VERSION_PREFIX):
        raise ManifestError(
            f"manifest.yaml 的 corpus_version 格式应为 {CORPUS_VERSION_PREFIX}<64 hex>，"
            f"实际为 {declared_version!r}。"
        )
    return Manifest(
        schema=schema,
        created=str(payload.get("created") or ""),
        corpus_root=root_path,
        entries=tuple(entries),
        summary=dict(payload.get("summary") or {}),
        source_path=manifest_path,
        version=declared_version,
        computed_version=compute_entry_digest(entries),
    )


def compute_entry_digest(entries: Iterable[ManifestEntry]) -> str:
    """This runner's own digest over the entries it loaded: ``file`` + ``sha256``.

    **This is not the corpus version.** The authoritative identity is the
    manifest's own ``corpus_version`` field, which ``eval/build_manifest.py``
    computes over a deliberately different input — category names and quotas
    as well as file digests, in a different serialisation. The two therefore
    produce different strings on the same corpus, and that is fine; what makes
    them worth carrying together is that they are computed from *the same
    loaded entries*. If this digest ever stops tracking a change that moved
    the manifest's, the loader dropped something, and a report showing both
    lets a reviewer notice that without re-deriving either by hand.

    Keeping it at all, rather than deleting it and relying on the manifest, is
    about the cases where the manifest cannot help: a report generated from a
    hand-written manifest with no ``corpus_version`` still gets a stamp, and a
    corpus rebuilt without ``build_manifest.py`` still has one.

    The two properties that matter are *stability* and *sensitivity*:

    * Stability comes from sorting before hashing, so reordering the manifest
      does not move the digest and mark every later report incomparable.
    * Sensitivity comes from hashing the *content digest* the manifest already
      stores rather than re-reading 79 MB of PDFs — which would be slow, and
      would disagree with the manifest the instant a file was edited in place.

    Both are covered by ``tests/test_eval_runner.py``.
    """
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: (item.file, item.sha256)):
        digest.update(entry.file.encode("utf-8"))
        digest.update(b"\0")
        digest.update(entry.sha256.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

#: The primary "where am I in this file" coordinate, per locator kind. This is
#: the format-aware part of the reading-order check: a two-column PDF is hard
#: because column order is *not* derivable from y alone, but page order always
#: is, and page order is the axis a missing page betrays.
_ORDER_COORDINATE: dict[str, Callable[[SourceLocator], int | None]] = {
    "pdf_page": lambda locator: locator.page,
    "slide": lambda locator: locator.slide,
    "docx_element": lambda locator: locator.paragraph,
    "txt_offset": lambda locator: locator.offset_start,
}


def _order_coordinates(elements: Sequence[Any]) -> tuple[list[int], str]:
    """The reading-order coordinate sequence and which coordinate it came from.

    Tries ``pdf_page`` first, then the other paginated formats, then character
    offsets, and finally gives up. The basis is returned because a score
    computed on offsets and a score computed on pages are not the same
    measurement, and a table that mixes them without saying so is the kind of
    number that survives review and fails in production.
    """
    for kind, extract in _ORDER_COORDINATE.items():
        values = [
            value
            for value in (extract(element.locator) for element in elements)
            if value is not None
        ]
        if len(values) >= 2:
            return values, kind
    return [], "none"


def reading_order_report(document: ParsedDocument) -> tuple[float | None, str, int]:
    """Measure whether reading order is monotonic in the primary coordinate.

    Returns ``(score, basis, inversions)`` where the score is the share of
    adjacent element pairs whose coordinate never goes backwards. It is
    deliberately *not* a claim about two-column correctness — that would need
    a ground-truth column annotation this corpus does not carry — but it does
    catch the failure the plan actually worries about: a parser that interleaves
    pages, which reads as a document that has been scrambled.

    ``None`` means "not measurable here" (fewer than two comparable elements),
    and is kept distinct from ``1.0`` so a one-page document cannot be read as
    evidence that reading order works.
    """
    elements = document.iter_reading_order()
    values, basis = _order_coordinates(elements)
    if basis == "none" or len(values) < 2:
        return None, basis, 0
    inversions = sum(1 for index in range(len(values) - 1) if values[index + 1] < values[index])
    score = 1.0 - inversions / (len(values) - 1)
    return round(score, RATIO_PLACES), basis, inversions


def page_coverage_report(document: ParsedDocument) -> tuple[float | None, int, int]:
    """``(ratio, pages_covered, pages_reported)`` over body elements.

    ``None`` rather than ``0.0`` for a document with no page count: DOCX and
    PPTX have no pages to miss, and scoring them 0% would drag every aggregate
    down with a number that does not mean anything. The quality gate makes the
    same abstention for the same reason (``PageCoverageRule``).
    """
    reported = _as_int(document.page_count)
    if reported <= 0:
        return None, 0, 0
    covered = {
        element.locator.page
        for element in document.body_elements
        if element.locator.kind == "pdf_page" and element.locator.page is not None
    }
    return round(min(1.0, len(covered) / reported), RATIO_PLACES), len(covered), reported


def locator_report(document: ParsedDocument) -> tuple[float, float]:
    """``(coverage, bbox_ratio)`` for element locators.

    Coverage reuses ``ParsedDocument.locator_coverage`` — the share of body
    elements a citation could point at — because the P0 exit criterion is
    written against that exact number and a bake-off measuring a private
    reimplementation of it could disagree with the release gate.

    ``bbox_ratio`` is the share of *those* locators that also carry a bounding
    box. The plan's exit criteria ask for page **and** bbox on PDFs; a parser
    that resolves to the right page but no rectangle satisfies half of a
    citation and cannot draw a highlight, so the two are counted separately.
    """
    body = document.body_elements
    located = [element for element in body if element.locator.is_resolvable]
    if not located:
        return 0.0, 0.0
    with_bbox = sum(1 for element in located if element.locator.bbox is not None)
    return (
        round(document.locator_coverage, RATIO_PLACES),
        round(with_bbox / len(located), RATIO_PLACES),
    )


def table_report(document: ParsedDocument) -> tuple[int, int, int]:
    """``(total, two_dimensional, flattened)`` counts over body elements.

    *Two-dimensional* means the element still carries a :class:`Table` with at
    least two rows and two columns and at least one non-blank cell. A table
    that arrived as a run of tab-separated prose is counted as ``flattened``
    instead: cross-column retrieval is impossible on it, and the difference
    between "no tables found" and "tables found but destroyed" is the whole
    reason the ``table_dense`` category exists.
    """
    total = 0
    structured = 0
    flattened = 0
    for element in document.body_elements:
        if element.role != "table" and element.table is None:
            continue
        total += 1
        table = element.table
        if table is None or table.is_empty_shell or table.row_count < 2 or table.column_count < 2:
            flattened += 1
        else:
            structured += 1
    return total, structured, flattened


def formula_report(document: ParsedDocument) -> tuple[int, int]:
    """``(total, with_latex)`` counts over body elements.

    A formula survives when it arrives as LaTeX in ``Element.latex``, which is
    what the chunker and the formula search path downstream expect. A formula
    recovered as a picture or as flattened unicode has not survived, however
    readable it looks in a rendered view.
    """
    total = 0
    with_latex = 0
    for element in document.body_elements:
        if element.role != "formula" and not element.latex:
            continue
        total += 1
        if str(element.latex).strip():
            with_latex += 1
    return total, with_latex


@dataclass(slots=True)
class DocumentMetrics:
    """Everything measured for one (document, parser) pair.

    A single flat record, because the CSV is the artifact people diff and
    re-sort. Numbers that could not be measured are ``None`` rather than zero:
    "no formula was found" and "formula preservation was not applicable" are
    different findings and the report must not merge them.
    """

    document: str
    category: str
    parser: str
    status: str
    duration_ms: int = 0
    reason_code: str = ""
    error: str = ""
    parser_version: str = ""
    options_hash: str = ""
    elements: int = 0
    chars: int = 0
    chars_ratio_vs_manifest: float | None = None
    pages_expected: int = 0
    pages_reported: int = 0
    pages_covered: int = 0
    page_coverage: float | None = None
    reading_order: float | None = None
    reading_order_basis: str = "none"
    reading_order_inversions: int = 0
    locator_coverage: float = 0.0
    bbox_locator_ratio: float = 0.0
    tables: int = 0
    tables_2d: int = 0
    tables_flattened: int = 0
    formulas: int = 0
    formulas_with_latex: int = 0
    quality_status: str = ""
    quality_flags: str = ""

    @property
    def is_attempted(self) -> bool:
        return self.status in ATTEMPTED_STATUSES

    @property
    def is_usable(self) -> bool:
        return self.status in (STATUS_OK, STATUS_DEGRADED)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_csv_row(self) -> list[Any]:
        """A fixed-order row matching :data:`CSV_COLUMNS`.

        Written out by hand rather than via ``asdict`` because the CSV column
        order is part of the report's contract: reordering columns turns every
        ``diff`` between two runs into noise.
        """
        payload = self.to_dict()
        return [payload[column] for column in CSV_COLUMNS]


CSV_COLUMNS: tuple[str, ...] = tuple(DocumentMetrics.__slots__)

STATUS_LABELS = {
    STATUS_OK: "成功",
    STATUS_DEGRADED: "降级可用",
    STATUS_BROKEN: "失败",
    STATUS_UNSUPPORTED: "不支持该格式",
    STATUS_UNAVAILABLE: "无法运行",
}


def _identity(
    parser: str,
    entry: ManifestEntry,
    *,
    status: str,
    reason_code: str = "",
    error: str = "",
) -> DocumentMetrics:
    """The skeleton row every outcome starts from."""
    return DocumentMetrics(
        document=entry.file,
        category=entry.category,
        parser=parser,
        status=status,
        reason_code=reason_code,
        error=error,
        pages_expected=entry.pages,
    )


def metrics_for_document(
    parser: str,
    entry: ManifestEntry,
    document: ParsedDocument,
    *,
    status: str = STATUS_OK,
    duration_ms: int = 0,
    reason_code: str = "",
    error: str = "",
) -> DocumentMetrics:
    """Derive the full metric row from a parsed document.

    The quality gate is run here rather than trusted from the parser because
    its verdict is what the orchestrator will act on, and a bake-off that
    reported adapter-optimistic quality numbers would be measuring something
    the product never sees.
    """
    row = _identity(
        parser, entry,
        status=status, reason_code=reason_code, error=error,
    )
    row.duration_ms = int(duration_ms)
    row.parser_version = document.parser_version
    row.options_hash = document.options_hash
    row.elements = len(document.body_elements)
    row.chars = document.char_count
    if entry.chars_total > 0:
        # A recall *proxy*, not a score. The denominator is what pypdf's text
        # extraction returned when the manifest was built, and pypdf
        # under-counts exactly the documents whose fonts lack a usable
        # ToUnicode CMap — which is the set this column would be most needed
        # for. A ratio well above 1.0 therefore usually indicts the reference,
        # not the parser, and the report says so next to the number.
        row.chars_ratio_vs_manifest = round(document.char_count / entry.chars_total, RATIO_PLACES)
    row.page_coverage, row.pages_covered, row.pages_reported = page_coverage_report(document)
    row.reading_order, row.reading_order_basis, row.reading_order_inversions = (
        reading_order_report(document)
    )
    row.locator_coverage, row.bbox_locator_ratio = locator_report(document)
    row.tables, row.tables_2d, row.tables_flattened = table_report(document)
    row.formulas, row.formulas_with_latex = formula_report(document)

    report = evaluate_quality(document)
    row.quality_status = getattr(report.status, "value", str(report.status))
    row.quality_flags = ";".join(flag.rule_id for flag in report.flags)
    return row


def metrics_for_failure(
    parser: str,
    entry: ManifestEntry,
    *,
    error: str,
    duration_ms: int = 0,
    status: str = STATUS_BROKEN,
    reason_code: str = "",
) -> DocumentMetrics:
    """A row for a parse that produced no usable document."""
    row = _identity(parser, entry, status=status, reason_code=reason_code, error=error)
    row.duration_ms = int(duration_ms)
    return row


# --------------------------------------------------------------------------
# Parser slots
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ParserSlot:
    """One candidate backend and whether this run can actually use it.

    A slot that cannot run still carries a ``reason_code``. That code is what
    the report prints, what the JSON stores and what a reader needs in order
    to tell "we did not have the dependency" apart from "the dependency was
    there and produced nothing" — the two require completely different
    follow-up.

    ``parser`` is the adapter object behind the slot. It is kept beside
    ``run`` because ``run`` is called with a stable three-argument signature
    the runner controls, while ``parser.supports()`` is the adapter's own
    answer about which formats it handles.
    """

    name: str
    available: bool
    reason: str = ""
    reason_code: str = ""
    version: str = ""
    #: ``(entry, path, options) -> raw parse outcome``. Injected in tests, so
    #: the whole runner can be exercised without a parser installed.
    run: Callable[[ManifestEntry, Path, Any], Any] | None = None
    parser: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "available": self.available,
            "reason": self.reason,
            "reason_code": self.reason_code,
            "version": self.version,
        }


#: Reason codes, kept as constants so the report and the tests cannot drift
#: apart by retyping a string.
REASON_OK = ""
REASON_NOT_REQUESTED = "not_requested"
REASON_DEPENDENCY_MISSING = "dependency_missing"
REASON_MODE_NOT_CONFIGURED = "mode_not_configured"
REASON_NO_TOKEN = "no_api_token"
REASON_PRIVACY_MODE = "privacy_mode_local_only"
REASON_CONSENT_MISSING = "cloud_consent_not_granted"
REASON_CLOUD_NOT_ALLOWED = "cloud_not_allowed_by_cli"
REASON_UNSUPPORTED_MEDIA_TYPE = "unsupported_media_type"
REASON_PARSE_ERROR = "parse_error"


def _module_available(module_name: str) -> bool:
    """Whether ``module_name`` is importable, **without importing it**.

    ``importlib.util.find_spec`` reads the finder, not the module, so asking
    about docling costs a directory scan. ``import docling`` would cost
    several hundred megabytes of torch — and the runner must be able to
    produce a report *about* a machine where docling is missing.
    """
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        # A namespace package or a broken .pth can make find_spec raise
        # without the module being present; absence is the useful answer.
        return False


def _installed_version(distribution: str) -> str:
    """Best-effort version of a distribution, read from metadata.

    Metadata lives on disk as a text file, so this reports a version without
    executing the package. ``"unknown"`` rather than an exception, because a
    version is diagnostic detail, never a precondition for a report.
    """
    try:
        from importlib.metadata import version as dist_version

        return dist_version(distribution)
    except Exception:  # noqa: BLE001 - see docstring
        return "unknown"


def _as_slot_run(call: Callable[..., Any]) -> Callable[[ManifestEntry, Path, Any], Any]:
    """Wrap an adapter's ``parse(path, options)`` in the runner's call shape.

    Every adapter in the tree takes ``(file_path, options)``; the runner wants
    ``(entry, path, options)`` so a fake in a test can key off the manifest
    entry (to return a different document for a different category) without a
    side channel. Binding here rather than at each call site keeps the four
    adapters free of any knowledge that a bake-off exists.
    """

    def _run(entry: ManifestEntry, path: Path, options: Any) -> Any:
        return call(str(path), options)

    return _run


def _legacy_slot() -> ParserSlot:
    """The legacy adapter, which is always present: it has no optional dep."""
    from paperwhisperer.ingestion.adapters.legacy_loader import LegacyDocumentParser

    parser = LegacyDocumentParser()
    return ParserSlot(
        name=PARSER_LEGACY,
        available=True,
        reason="",
        reason_code=REASON_OK,
        version=str(getattr(parser, "version", "") or ""),
        run=_as_slot_run(parser.parse),
        parser=parser,
    )


def _docling_slot(mode: str) -> ParserSlot:
    """A docling slot for ``mode`` in ``{"inprocess", "serve"}``.

    The in-process mode additionally requires the package; the serve mode
    requires only a loopback URL, because a missing service fails per document
    with a connection error that is a real result, whereas a missing package
    cannot be discovered that way. docling-serve is *not* gated on the privacy
    setting or on cloud consent: it is a local container, and refusing to use
    a local service in a local-only deployment would be the wrong default.
    """
    from paperwhisperer.ingestion.adapters.docling_adapter import DoclingParser

    name = PARSER_DOCLING if mode == "inprocess" else PARSER_DOCLING_SERVE
    if mode == "inprocess" and not _module_available("docling"):
        return ParserSlot(
            name=name,
            available=False,
            reason=(
                "未安装 docling（进程内模式需要 docling 包及其模型权重）。"
                "安装：pip install 'paperwhisperer[parse]'，或改用 --parsers docling_serve。"
            ),
            reason_code=REASON_DEPENDENCY_MISSING,
            version="unknown",
        )
    parser = DoclingParser(mode=mode)
    return ParserSlot(
        name=name,
        available=True,
        reason="",
        reason_code=REASON_OK,
        version=str(getattr(parser, "version", "") or "unknown"),
        run=_as_slot_run(parser.parse),
        parser=parser,
    )


def _mineru_slot(allow_cloud: bool) -> ParserSlot:
    """MinerU cloud, behind four independent gates.

    The gates are stacked rather than merged because they are answered by
    different people: the token by whoever owns the account, the privacy mode
    by the deployment, the consent flag by the end user, and ``--allow-cloud``
    by whoever typed the command. Reporting which one is missing turns a
    "why is there no cloud column" question into a one-line answer.
    """
    from paperwhisperer.ingestion.adapters.mineru_cloud import PARSER_VERSION, MineruCloudParser

    checks: tuple[tuple[bool, str, str], ...] = (
        (bool(str(getattr(config, "MINERU_API_TOKEN", "") or "").strip()),
         "未配置 MINERU_API_TOKEN。", REASON_NO_TOKEN),
        (str(getattr(config, "DOCUMENT_PRIVACY_MODE", "local_only")) == "allow_cloud",
         "DOCUMENT_PRIVACY_MODE=local_only，禁止云端解析。", REASON_PRIVACY_MODE),
        (bool(getattr(config, "DOCUMENT_CLOUD_CONSENT_GRANTED", False)),
         "DOCUMENT_CLOUD_CONSENT_GRANTED 未开启。", REASON_CONSENT_MISSING),
        (bool(allow_cloud),
         "未显式传入 --allow-cloud；评测默认不做付费云端调用。", REASON_CLOUD_NOT_ALLOWED),
    )
    for passed, reason, code in checks:
        if not passed:
            return ParserSlot(
                name=PARSER_MINERU,
                available=False,
                reason=reason,
                reason_code=code,
                version=str(PARSER_VERSION),
            )
    mineru = MineruCloudParser()
    return ParserSlot(
        name=PARSER_MINERU,
        available=True,
        reason="",
        reason_code=REASON_OK,
        version=str(PARSER_VERSION),
        run=_as_slot_run(mineru.parse),
        parser=mineru,
    )


def build_parser_slots(
    parsers: Sequence[str] = (PARSER_LEGACY,),
    *,
    allow_cloud: bool = False,
    docling_mode: str = "",
) -> list[ParserSlot]:
    """Resolve the requested parser names into slots, in a stable order.

    Requested-but-unresolvable names become unavailable slots rather than an
    error, for the same reason absent dependencies do: a report that says
    "docling_serve is not a thing here" is more useful than a traceback. An
    entirely *unrecognised* name is different — it is a typo, and a typo that
    silently evaluates to "no parser" would produce an empty comparison that
    still looks like a report.
    """
    requested: list[str] = []
    for name in parsers:
        cleaned = str(name).strip().lower()
        if not cleaned:
            continue
        if cleaned not in KNOWN_PARSERS:
            raise ValueError(
                f"未知解析器 {cleaned!r}；可选：{', '.join(KNOWN_PARSERS)}"
            )
        if cleaned not in requested:
            requested.append(cleaned)

    resolved_mode = str(docling_mode or getattr(config, "PARSER_DOCLING_MODE", "inprocess"))
    if resolved_mode not in {"inprocess", "serve"}:
        resolved_mode = "inprocess"

    slots: list[ParserSlot] = []
    for name in KNOWN_PARSERS:
        if name not in requested:
            continue
        try:
            if name == PARSER_LEGACY:
                slots.append(_legacy_slot())
            elif name == PARSER_DOCLING:
                slots.append(_docling_slot("inprocess"))
            elif name == PARSER_DOCLING_SERVE:
                slots.append(_docling_slot("serve"))
            else:
                slots.append(_mineru_slot(allow_cloud))
        except Exception as exc:  # noqa: BLE001 - a slot must never abort a run
            slots.append(ParserSlot(
                name=name,
                available=False,
                reason=f"构造解析器时失败：{exc}",
                reason_code=REASON_DEPENDENCY_MISSING,
                version="unknown",
            ))
    return slots


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------


@dataclass(slots=True)
class EvalParseOptions:
    """Options handed to each adapter for one document.

    The adapters read what they need with ``getattr`` and none of them import
    the gateway, so rather than teach three adapters about three extra fields
    this is a thin proxy: unknown attributes fall through to a real
    ``ParseOptions``, which keeps ``compute_options_hash()`` — the cache key
    component T0.7 depends on — the gateway's own implementation instead of a
    second copy of it.
    """

    parse_options: ParseOptions
    media_type: str = ""
    source_filename: str = ""
    trigger_reason: str = "manual"
    operator: str | None = None

    def __getattr__(self, item: str) -> Any:
        # Only reached for names that are not slots on this class.
        return getattr(self.parse_options, item)


def _coerce_outcome(raw: Any, parser_name: str) -> tuple[str, ParsedDocument | None, str, int, str]:
    """Normalize whatever an adapter returned into one common shape.

    Returns ``(status, document, error, duration_ms, version)``.

    Three return conventions exist today and the runner reads all of them:
    a gateway ``ParseResult``; the docling adapter's duck-typed mirror, which
    uses ``"ok"`` where the gateway uses ``"success"``; and MinerU's
    ``(document, audit_record)`` tuple, where every failure mode is a
    ``None`` document plus a message on the record. Coercing here rather than
    in each adapter keeps a bake-off from silently scoring a parser as
    successful-with-no-content because its status string was spelled
    differently.
    """
    if raw is None:
        return STATUS_BROKEN, None, f"解析器 {parser_name} 没有返回结果。", 0, ""

    if isinstance(raw, tuple):
        document = raw[0] if raw else None
        record = raw[1] if len(raw) > 1 else None
        message = str(getattr(record, "message", "") or "")
        if isinstance(document, ParsedDocument):
            return (
                STATUS_OK, document, message,
                0, str(getattr(document, "parser_version", "") or ""),
            )
        return STATUS_BROKEN, None, message or "云端解析未返回文档。", 0, ""

    status_text = str(getattr(raw, "status", "") or "").strip().lower()
    document = getattr(raw, "document", None)
    error = str(getattr(raw, "error_message", "") or "")
    duration_ms = _as_int(getattr(raw, "duration_ms", 0))
    version = str(getattr(raw, "parser_version", "") or "")

    if status_text in {"ok", "success"}:
        status = STATUS_OK
    elif status_text in {"partial", "degraded"}:
        status = STATUS_DEGRADED
    else:
        status = STATUS_BROKEN
    if status == STATUS_OK and not isinstance(document, ParsedDocument):
        return (
            STATUS_BROKEN, None,
            error or f"解析器 {parser_name} 报告成功但没有文档。",
            duration_ms, version,
        )
    resolved = document if isinstance(document, ParsedDocument) else None
    return status, resolved, error, duration_ms, version


def _supports(parser_object: Any, media_type: str) -> bool:
    """Whether an adapter claims this media type, treating a raising call as no.

    ``supports`` is a claim about capability (the gateway reasons the same way,
    in ``ParserRouter._supports``); a parser that cannot answer has not made
    the claim, and a broken optional dependency must not take the run down.
    """
    if parser_object is None:
        return True
    try:
        return bool(parser_object.supports(media_type))
    except Exception:  # noqa: BLE001 - see docstring
        return False


def _slot_parser_object(slot: ParserSlot) -> Any:
    """The adapter instance behind a slot, when the slot has one."""
    return getattr(slot, "parser", None)


def run_document(
    slot: ParserSlot,
    entry: ManifestEntry,
    *,
    parse_options: ParseOptions | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> DocumentMetrics:
    """Run one (document, parser) pair and return its metric row.

    Timing is done here rather than taken from the adapter, because adapters
    report what they spent internally and the report needs what the bake-off
    spent — including the adapter's own file reads and setup. An adapter
    duration is used only when it exceeds the measured one, which happens when
    a parser times out internally and the outer clock saw a clean return.
    """
    if not slot.available or slot.run is None:
        return metrics_for_failure(
            slot.name, entry,
            status=STATUS_UNAVAILABLE,
            reason_code=slot.reason_code or REASON_NOT_REQUESTED,
            error=slot.reason or "解析器不可用。",
        )

    path = entry.path
    media_type = media_type_for_path(str(path))
    if not _supports(_slot_parser_object(slot), media_type):
        return metrics_for_failure(
            slot.name, entry,
            status=STATUS_UNSUPPORTED,
            reason_code=REASON_UNSUPPORTED_MEDIA_TYPE,
            error=f"解析器 {slot.name} 不支持 {media_type}。",
        )

    options = EvalParseOptions(
        parse_options=parse_options or ParseOptions(),
        media_type=media_type,
        source_filename=os.path.basename(entry.file),
        trigger_reason="manual",
    )

    started = clock()
    try:
        raw = slot.run(entry, path, options)
    except Exception as exc:  # noqa: BLE001 - a bake-off records failures
        elapsed = int(round((clock() - started) * 1000))
        return metrics_for_failure(
            slot.name, entry,
            status=STATUS_BROKEN,
            reason_code=REASON_PARSE_ERROR,
            error=f"{type(exc).__name__}: {exc}",
            duration_ms=elapsed,
        )
    elapsed = int(round((clock() - started) * 1000))

    status, document, error, duration_ms, _version = _coerce_outcome(raw, slot.name)
    if document is None:
        return metrics_for_failure(
            slot.name, entry,
            status=status if status != STATUS_OK else STATUS_BROKEN,
            reason_code=REASON_PARSE_ERROR,
            error=error or "解析器未产出文档。",
            duration_ms=duration_ms or elapsed,
        )
    return metrics_for_document(
        slot.name, entry, document,
        status=status,
        duration_ms=duration_ms or elapsed,
        error=error,
    )


def entry_path(entry: ManifestEntry) -> Path:
    """Absolute path of a corpus document.

    The corpus root travels on the entry rather than in a module global so a
    test can load a manifest from ``tmp_path`` and have its entries resolve
    there without leaking that root into the next test.
    """
    return entry.path


def run_bakeoff(
    entries: Sequence[ManifestEntry],
    slots: Sequence[ParserSlot],
    *,
    corpus_root: str | os.PathLike[str] | None = None,
    parse_options: ParseOptions | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> list[DocumentMetrics]:
    """Cross the corpus with the parser slots and return every metric row.

    The matrix is always complete: a document that no available parser claims,
    and a parser that was not available at all, both produce rows. A short row
    set is the failure mode this design exists to prevent, because a table
    that shrinks when a dependency disappears is impossible to review.

    ``corpus_root`` overrides the root baked into each entry, which is what
    lets a caller point a real manifest at a relocated corpus without editing
    it.
    """
    if corpus_root is not None:
        resolved = str(Path(corpus_root))
        entries = [
            replace(entry, root=resolved) if entry.root != resolved else entry
            for entry in entries
        ]
    rows: list[DocumentMetrics] = []
    for entry in entries:
        for slot in slots:
            rows.append(run_document(slot, entry, parse_options=parse_options, clock=clock))
    return rows


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def _mean(values: Iterable[float]) -> float | None:
    numbers = [value for value in values if value is not None]
    if not numbers:
        return None
    return round(statistics.fmean(numbers), RATIO_PLACES)


def aggregate_rows(rows: Sequence[DocumentMetrics]) -> dict[str, Any]:
    """Collapse metric rows into the per-parser summary the report leads with.

    Ratio-of-ratios is used for the table and formula columns on purpose:
    ``tables_2d`` summed over the corpus and divided by ``tables`` summed over
    the corpus, not the mean of per-document ratios. A document with one table
    and a document with forty should not weigh the same in a decision about
    how often tables survive.
    """
    attempted = [row for row in rows if row.is_attempted]
    usable = [row for row in rows if row.is_usable]
    skipped = [row for row in rows if row.status == STATUS_UNSUPPORTED]
    unavailable = [row for row in rows if row.status == STATUS_UNAVAILABLE]

    tables_total = sum(row.tables for row in usable)
    tables_2d = sum(row.tables_2d for row in usable)
    formulas_total = sum(row.formulas for row in usable)
    formulas_latex = sum(row.formulas_with_latex for row in usable)

    return {
        "documents": len({row.document for row in rows}),
        "attempted": len(attempted),
        "usable": len(usable),
        "failed": sum(1 for row in rows if row.status == STATUS_BROKEN),
        "skipped_unsupported": len(skipped),
        "unavailable": len(unavailable),
        "failure_rate": _ratio(
            sum(1 for row in rows if row.status == STATUS_BROKEN), len(attempted)
        ),
        "mean_page_coverage": _mean(row.page_coverage for row in usable),
        "mean_reading_order": _mean(row.reading_order for row in usable),
        "mean_locator_coverage": _mean(row.locator_coverage for row in usable),
        "mean_bbox_locator_ratio": _mean(row.bbox_locator_ratio for row in usable),
        "table_2d_ratio": _ratio(tables_2d, tables_total),
        "formula_latex_ratio": _ratio(formulas_latex, formulas_total),
        "mean_chars_ratio_vs_manifest": _mean(row.chars_ratio_vs_manifest for row in usable),
        "median_duration_ms": _median_ms(usable),
        "mean_duration_ms": (
            int(round(statistics.fmean([row.duration_ms for row in usable])))
            if usable else None
        ),
        "quality_fail": sum(1 for row in usable if row.quality_status == "fail"),
        "quality_warn": sum(1 for row in usable if row.quality_status == "warn"),
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, RATIO_PLACES)


def _median_ms(rows: Sequence[DocumentMetrics]) -> int | None:
    durations = [row.duration_ms for row in rows]
    if not durations:
        return None
    return int(round(statistics.median(durations)))


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


@dataclass(slots=True)
class EvalReport:
    """The complete, self-describing output of one bake-off run.

    Two version fields, and the distinction is the point:

    ``corpus_version``
        The manifest's authoritative ``corpus_version``, and the value a
        report cites as *the corpus it ran against*. It is the one
        ``eval/build_manifest.py`` reproduces byte for byte on an unchanged
        corpus, and the one any future CI check or colleague will read to
        decide whether two reports are comparable.

    ``runner_entry_digest``
        The runner's independent digest over the entries it loaded, carried
        alongside so the loader's view and the manifest's view can be compared
        at a glance. It is a cross-check, not a second identity, and it is
        never substituted for ``corpus_version``.
    """

    corpus_version: str
    manifest_schema: str
    manifest_created: str
    generated_at: str
    runner_entry_digest: str = ""
    runner_version: str = RUNNER_VERSION
    environment: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)
    slots: tuple[ParserSlot, ...] = ()
    rows: tuple[DocumentMetrics, ...] = ()

    def parsers(self) -> list[str]:
        return [slot.name for slot in self.slots]

    def rows_for(self, parser: str) -> list[DocumentMetrics]:
        return [row for row in self.rows if row.parser == parser]

    def summary(self) -> dict[str, dict[str, Any]]:
        """Per-parser aggregates, keyed by parser name."""
        return {parser: aggregate_rows(self.rows_for(parser)) for parser in self.parsers()}

    def category_summary(self) -> dict[str, dict[str, Any]]:
        """Per-(parser, category) aggregates, keyed ``parser / category``."""
        out: dict[str, dict[str, Any]] = {}
        for parser in self.parsers():
            for row in self.rows_for(parser):
                out.setdefault(parser, {}).setdefault(row.category, [])
        buckets: dict[str, dict[str, Any]] = {}
        for parser, per_category in out.items():
            for category in per_category:
                key = f"{parser} / {category}"
                buckets[key] = aggregate_rows([
                    row for row in self.rows
                    if row.parser == parser and row.category == category
                ])
        return buckets

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "paperwhisperer-parser-eval-report/1",
            "runner_version": self.runner_version,
            "generated_at": self.generated_at,
            # The stamped, authoritative identity. Key name kept plain because
            # it is the one a consumer should reach for without reading docs.
            "corpus_version": self.corpus_version,
            # The runner's own digest over the entries it loaded. Present so a
            # reviewer can see the loader and the manifest agreeing on which
            # files were in play; explicitly *not* an alternative identity.
            "runner_entry_digest": self.runner_entry_digest,
            "manifest": {
                "schema": self.manifest_schema,
                "created": self.manifest_created,
            },
            "environment": self.environment,
            "options": self.options,
            "parsers": [slot.to_dict() for slot in self.slots],
            "summary": self.summary(),
            "category_summary": self.category_summary(),
            "rows": [row.to_dict() for row in self.rows],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=False) + "\n"

    def to_csv(self) -> str:
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(CSV_COLUMNS)
        for row in self.rows:
            writer.writerow(["" if value is None else value for value in row.to_csv_row()])
        return buffer.getvalue()

    def write(self, out_dir: str | os.PathLike[str]) -> list[Path]:
        """Write the markdown, CSV and JSON artifacts, returning their paths.

        Fixed filenames on purpose: the point of a bake-off is comparing two
        runs, and a timestamped directory per run is one more thing to
        correlate by hand. ``generated_at`` inside each file carries the time.
        """
        directory = Path(out_dir)
        directory.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for name, content in (
            ("parser_eval.md", self.to_markdown()),
            ("parser_eval.csv", self.to_csv()),
            ("parser_eval.json", self.to_json()),
        ):
            target = directory / name
            target.write_text(content, encoding="utf-8")
            written.append(target)
        return written

    # -- markdown ----------------------------------------------------------

    def to_markdown(self) -> str:
        """Render the human-facing comparison report.

        Bilingual column headings on purpose: the plan, the corpus notes and
        the config comments are all Chinese, and a table whose only labels are
        English gets misread by exactly the reviewers who own the routing
        decision.
        """
        lines: list[str] = []
        lines.append("# Parser bake-off report / 解析器对比报告")
        lines.append("")
        # Two version lines, deliberately not collapsed into one. The first is
        # the corpus identity a report is *about*; the second is the runner's
        # own digest over the entries it loaded, which exists so a reader can
        # see that the loader and the manifest agree on the file set.
        lines.append(
            f"- Corpus version (manifest `corpus_version`, authoritative): "
            f"`{self.corpus_version or 'n/a'}`"
        )
        lines.append(
            f"- Runner entry digest (local cross-check over loaded "
            f"(file, sha256) pairs, **not** a corpus identity): "
            f"`{self.runner_entry_digest or 'n/a'}`"
        )
        if not self.corpus_version:
            lines.append(
                "> **警告**：manifest 没有 `corpus_version` 字段，本报告无法声明"
                "可复现的语料版本。请用 `build_manifest.py` 重新生成清单。"
            )
        lines.append(
            f"- Manifest: `{self.manifest_schema}`, "
            f"created `{self.manifest_created or 'n/a'}`"
        )
        lines.append(f"- Generated: {self.generated_at}")
        lines.append(f"- Runner version: {self.runner_version}")
        lines.append(f"- Environment: {self.environment.get('python', '?')} on "
                     f"{self.environment.get('platform', '?')}")
        lines.append("")

        options = self.options
        lines.append("## Run options / 运行参数")
        lines.append("")
        lines.append("| Option | Value |")
        lines.append("|---|---|")
        for key in sorted(options):
            lines.append(f"| `{key}` | {options[key]} |")
        lines.append("")

        lines.append(self._availability_markdown())
        lines.append(self._summary_markdown())
        lines.append(self._category_markdown())
        lines.append(self._detail_markdown())
        lines.append(
            "## How to read these numbers / 指标解读\n\n"
            "- `page_coverage`: 出现正文的页数 / 解析器自报的页数；无分页格式记为 n/a。\n"
            "- `reading_order`: 正文在主坐标（页/幻灯片/段落/字符偏移）上单调的比例；"
            "n/a 表示可比元素不足两个。\n"
            "- `tables 2d/total`: 保留行列结构的表格数 / 表格总数；差额为被拍平成文本的表格。\n"
            "- `formulas latex/total`: 带 LaTeX 的公式数 / 公式总数。\n"
            "- `locator`: 可解析 SourceLocator 的正文元素比例（对应 P0 退出标准 100%）。\n"
            "- `bbox`: 上述 locator 中带 bbox 的比例（PDF 引用高亮的前置条件）。\n"
            "- `chars/manifest`: 解析出的正文字符数 / manifest 记录的文本层字符数。"
            "后者来自 pypdf 文本层，是参考值而非 ground truth。\n"
            "- `ms`: 外层墙钟耗时，含适配器自身的文件读取与初始化。\n"
        )
        return "\n".join(lines) + "\n"

    def _availability_markdown(self) -> str:
        lines = ["## Parsers that could not run / 无法运行的解析器", ""]
        unavailable = [slot for slot in self.slots if not slot.available]
        if not unavailable:
            lines.append(
                "All requested parsers ran. 本次请求的解析器全部可运行。"
            )
            return "\n".join(lines) + "\n"
        lines.append("| Parser | Reason code | Detail /")
        lines.append("|---|---|---|")
        for slot in unavailable:
            detail = (slot.reason or "").replace("|", "\\|").replace("\n", " ")
            lines.append(f"| `{slot.name}` | `{slot.reason_code}` | {detail} |")
        return "\n".join(lines) + "\n"

    def _summary_markdown(self) -> str:
        lines = ["## Parser summary / 解析器汇总", ""]
        lines.append(
            "| Parser | Version | Usable / attempted | Failed | Failure rate | "
            "Page coverage | Reading order | Locator | bbox | Tables 2D | Formulas LaTeX | "
            "Median ms | Quality FAIL |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for parser in self.parsers():
            slot = next(item for item in self.slots if item.name == parser)
            stats = self.summary()[parser]
            lines.append(
                "| `{parser}` | {version} | {usable}/{attempted} | {failed} | {failure} | "
                "{page} | {order} | {locator} | {bbox} | {tables} | {formulas} | {ms} | {qfail} |"
                .format(
                    parser=parser,
                    version=slot.version or "n/a",
                    usable=stats["usable"],
                    attempted=stats["attempted"],
                    failed=stats["failed"],
                    failure=_fmt(stats["failure_rate"]),
                    page=_fmt(stats["mean_page_coverage"]),
                    order=_fmt(stats["mean_reading_order"]),
                    locator=_fmt(stats["mean_locator_coverage"]),
                    bbox=_fmt(stats["mean_bbox_locator_ratio"]),
                    tables=_fmt(stats["table_2d_ratio"]),
                    formulas=_fmt(stats["formula_latex_ratio"]),
                    ms=_fmt(stats["median_duration_ms"]),
                    qfail=stats["quality_fail"],
                )
            )
        return "\n".join(lines) + "\n"

    def _category_markdown(self) -> str:
        buckets = self.category_summary()
        if not buckets:
            return ""
        lines = ["## Per-category / 分类别对比", ""]
        lines.append(
            "| Parser / category | Usable | Page coverage | Reading order | Locator | "
            "Tables 2D | Formulas LaTeX | Median ms |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for key in sorted(buckets):
            stats = buckets[key]
            lines.append(
                "| `{key}` | {usable}/{attempted} | {page} | {order} | {locator} | "
                "{tables} | {formulas} | {ms} |".format(
                    key=key,
                    usable=stats["usable"],
                    attempted=stats["attempted"],
                    page=_fmt(stats["mean_page_coverage"]),
                    order=_fmt(stats["mean_reading_order"]),
                    locator=_fmt(stats["mean_locator_coverage"]),
                    tables=_fmt(stats["table_2d_ratio"]),
                    formulas=_fmt(stats["formula_latex_ratio"]),
                    ms=_fmt(stats["median_duration_ms"]),
                )
            )
        return "\n".join(lines) + "\n"

    def _detail_markdown(self) -> str:
        lines = ["## Per document / 逐文档明细", ""]
        for parser in self.parsers():
            rows = self.rows_for(parser)
            lines.append(f"### `{parser}`")
            lines.append("")
            if not rows:
                lines.append("No rows. 该解析器没有任何结果行。")
                lines.append("")
                continue
            lines.append(
                "| Document | Category | Status | Page cov | Order | Tables 2D | Formulas LaTeX | "
                "Locator | bbox | Chars/manifest | ms | Note |"
            )
            lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
            for row in sorted(rows, key=lambda item: (item.category, item.document)):
                note = row.error or row.quality_flags
                lines.append(
                    "| `{doc}` | {cat} | {status} | {page} | {order} | {tables} | {formulas} | "
                    "{locator} | {bbox} | {chars} | {ms} | {note} |".format(
                        doc=row.document,
                        cat=row.category,
                        status=f"{row.status}" if not STATUS_LABELS.get(row.status)
                        else f"{row.status} ({STATUS_LABELS[row.status]})",
                        page=_fmt(row.page_coverage),
                        order=_fmt(row.reading_order),
                        tables=f"{row.tables_2d}/{row.tables}",
                        formulas=f"{row.formulas_with_latex}/{row.formulas}",
                        locator=_fmt(row.locator_coverage),
                        bbox=_fmt(row.bbox_locator_ratio),
                        chars=_fmt(row.chars_ratio_vs_manifest),
                        ms=row.duration_ms,
                        note=note.replace("|", "\\|").replace("\n", " ")[:120],
                    )
                )
            lines.append("")
        return "\n".join(lines)


def _fmt(value: float | int | None) -> str:
    """Render a nullable number for a markdown cell."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def describe_environment() -> dict[str, Any]:
    """Environment facts that change parse results, and nothing sensitive.

    ``mineru_configured`` is a boolean, never the token: this dictionary ends
    up in a committed report, and a report is a document with a much wider
    readership than ``.env``. It mirrors ``core.config.describe()`` for the
    parser-relevant subset.
    """
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "parser_backend": str(getattr(config, "PARSER_BACKEND", "")),
        "parser_primary": str(getattr(config, "PARSER_PRIMARY", "")),
        "parser_fallback_chain": ",".join(getattr(config, "PARSER_FALLBACK_CHAIN", ()) or ()),
        "parser_timeout_seconds": getattr(config, "PARSER_TIMEOUT_SECONDS", None),
        "docling_mode": str(getattr(config, "PARSER_DOCLING_MODE", "")),
        "docling_version": _installed_version("docling"),
        "privacy_mode": str(getattr(config, "DOCUMENT_PRIVACY_MODE", "")),
        "cloud_consent_granted": bool(getattr(config, "DOCUMENT_CLOUD_CONSENT_GRANTED", False)),
        "mineru_configured": bool(str(getattr(config, "MINERU_API_TOKEN", "") or "").strip()),
    }


def build_report(
    manifest: Manifest,
    slots: Sequence[ParserSlot],
    rows: Sequence[DocumentMetrics],
    *,
    options: dict[str, Any] | None = None,
    generated_at: str = "",
) -> EvalReport:
    """Assemble a report from a manifest, the slots and the metric rows.

    Kept as a separate function from :func:`run_bakeoff` so a test — or a
    future comparison mode — can assemble a report from injected rows with no
    parsing at all. That separation is the whole reason the runner is
    testable without docling, without a token and without a network.

    **The stamped ``corpus_version`` is the manifest's own value, never this
    runner's digest.** The fallback that would have been convenient —
    ``manifest.version or compute_entry_digest(...)`` — is exactly the thing
    to avoid: it puts a locally-computed number into the field a reader
    trusts, under a name that promises it came from the manifest. So an absent
    ``corpus_version`` propagates as an empty string, the report prints a
    warning saying the run is not reproducible, and the local digest stays in
    its own clearly-labelled field where it can be read as the cross-check it
    actually is. The manifest's value is authoritative because it is the one
    that survives regeneration by ``build_manifest.py`` and the one any other
    tooling will read.
    """
    return EvalReport(
        corpus_version=manifest.version,
        runner_entry_digest=(
            manifest.computed_version or compute_entry_digest(manifest.entries)
        ),
        manifest_schema=manifest.schema,
        manifest_created=manifest.created,
        generated_at=generated_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        environment=describe_environment(),
        options=dict(options or {}),
        slots=tuple(slots),
        rows=tuple(rows),
    )


# --------------------------------------------------------------------------
# Regression gate
# --------------------------------------------------------------------------


def compare_to_baseline(
    report: EvalReport,
    baseline: dict[str, Any],
    *,
    tolerance: float = 0.0,
) -> list[str]:
    """Return human-readable regressions against a stored report's aggregates.

    The comparison is per parser and per corpus version: a baseline recorded
    on a different corpus is not evidence of a regression, it is evidence that
    the corpus changed. Reporting it as a finding and failing the run is
    deliberate — the alternative, a silent pass, prints "no regressions"
    over a comparison that never happened, which is the precise failure this
    gate exists to prevent. Rebuild the baseline on the new corpus instead.

    ``tolerance`` is subtracted before the comparison, so a small drop is
    tolerated and anything beyond it fails. It defaults to zero: a bake-off
    gate that quietly tolerates a drop is not a gate.
    """
    findings: list[str] = []
    baseline_version = str(baseline.get("corpus_version") or "")
    if baseline_version and baseline_version != report.corpus_version:
        findings.append(
            f"语料版本不同（baseline {baseline_version[:12]}… "
            f"vs 本次 {report.corpus_version[:12]}…），不比较指标；"
            "请在相同语料上重跑 baseline。"
        )
        return findings

    current = report.summary()
    baseline_summary = baseline.get("summary") or {}
    for parser, stats in current.items():
        # A parser that was not evaluated cannot have regressed, and must not
        # be reported as missing from the baseline either: an absent docling
        # installation produces an ``unavailable`` column that is correct, not
        # a baseline that failed to predict the future.
        if stats["attempted"] == 0:
            continue
        previous = baseline_summary.get(parser)
        if not isinstance(previous, dict):
            findings.append(f"{parser}: baseline 中没有该解析器，无法比较。")
            continue
        for metric in REGRESSION_METRICS:
            new_value = stats.get(metric)
            old_value = previous.get(metric)
            if new_value is None or old_value is None:
                continue
            try:
                delta = float(new_value) - float(old_value)
            except (TypeError, ValueError):
                continue
            if delta < -abs(tolerance):
                findings.append(
                    f"{parser}.{metric} 回归：{old_value:.4f} → {new_value:.4f} "
                    f"(下降 {abs(delta):.4f}，容差 {tolerance})"
                )
    return findings


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_parser_eval.py",
        description=(
            "在同一语料上批量对比各解析器并生成可复现报告。"
            "默认只跑 legacy，不做任何云端/付费调用。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python -m eval.run_parser_eval --parsers legacy --limit 3 --out /tmp/eval\n"
            "  python -m eval.run_parser_eval --parsers legacy,docling --categories table_dense\n"
            "  python -m eval.run_parser_eval --parsers legacy,mineru_cloud --allow-cloud\n"
        ),
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="manifest.yaml 路径")
    parser.add_argument("--corpus-root", default=None, help="覆盖 manifest 中的 corpus_root")
    parser.add_argument(
        "--parsers",
        default=PARSER_LEGACY,
        help=f"逗号分隔的解析器列表，可选：{','.join(KNOWN_PARSERS)}（默认 {PARSER_LEGACY}）",
    )
    parser.add_argument("--categories", default="", help="逗号分隔的类别过滤，留空表示全部")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="按 (类别, 文件) 排序后取前 N 份；用于冒烟运行，不是随机抽样",
    )
    parser.add_argument(
        "--include-excluded", action="store_true",
        help="包含 adversarial/ 等 batch_excluded 文件（它们设计上就会失败）",
    )
    parser.add_argument("--out", default=str(DEFAULT_REPORT_DIR), help="报告输出目录")
    parser.add_argument(
        "--allow-cloud", action="store_true",
        help="允许调用 MinerU 云端（仍需 token + privacy_mode=allow_cloud + 显式同意）",
    )
    parser.add_argument(
        "--force-ocr", action="store_true", help="以 force_ocr=True 调用各解析器"
    )
    parser.add_argument(
        "--list-parsers", action="store_true",
        help="只打印各解析器的可用性与原因，不解析任何文件",
    )
    parser.add_argument(
        "--fail-on-regression", default="", metavar="BASELINE_JSON",
        help="与既有的 parser_eval.json 比较，出现指标下降时以非零码退出",
    )
    parser.add_argument(
        "--regression-tolerance", type=float, default=0.0,
        help="允许的指标下降幅度（默认 0，即不允许任何下降）",
    )
    parser.add_argument("--quiet", action="store_true", help="只输出关键行")
    return parser


def _print_slot_status(slots: Sequence[ParserSlot]) -> None:
    for slot in slots:
        state = "可运行" if slot.available else "无法运行"
        detail = slot.reason or ""
        line = f"  {slot.name:<14} {state:<8} version={slot.version or 'n/a'}"
        if detail:
            line += f"  ({slot.reason_code}: {detail})"
        print(line)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    :return: ``0`` on success, ``1`` when the corpus manifest is unusable or
        ``--fail-on-regression`` found one, ``2`` for a bad command line.

    The default invocation parses nothing beyond the legacy adapter and makes
    no network call at all — not even an availability probe, since every
    probe in this module reads the filesystem or package metadata rather than
    a socket.
    """
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)

    requested = [name for name in str(args.parsers).split(",") if name.strip()]
    try:
        slots = build_parser_slots(requested, allow_cloud=bool(args.allow_cloud))
    except ValueError as exc:
        print(f"error: {exc}")
        return 2

    if args.list_parsers:
        print("解析器可用性 / parser availability:")
        _print_slot_status(slots)
        return 0

    try:
        manifest = load_manifest(args.manifest, corpus_root=args.corpus_root)
    except ManifestError as exc:
        print(f"error: {exc}")
        return 1

    categories = [name for name in str(args.categories).split(",") if name.strip()]
    entries = manifest.select(
        categories=categories,
        limit=args.limit,
        include_excluded=bool(args.include_excluded),
    )
    if not entries:
        print("error: 过滤后没有任何文档；检查 --categories / --limit。")
        return 1

    if not args.quiet:
        print(f"语料版本 corpus_version: {manifest.version or '(manifest 未提供)'}")
        print(f"条目摘要 runner_entry_digest: {manifest.computed_version}")
        print(f"语料清单 manifest: {manifest.source_path} ({manifest.schema}, created "
              f"{manifest.created or 'n/a'})")
        print(f"文档数 documents: {len(entries)} / {len(manifest.regular_entries)}")
        print("解析器可用性 / parser availability:")
        _print_slot_status(slots)

    if not manifest.version:
        # Not fatal: the run is still useful, it just cannot claim to be
        # reproducible. The same warning is printed into the report itself.
        print(
            "warning: manifest 没有 corpus_version 字段，本报告无法声明可复现的语料版本；"
            "请运行 build_manifest.py 重新生成清单。",
            file=sys.stderr,
        )

    parse_options = ParseOptions(force_ocr=bool(args.force_ocr))
    rows = run_bakeoff(entries, slots, parse_options=parse_options)
    report = build_report(manifest, slots, rows, options={
        "manifest": str(manifest.source_path),
        "parsers": ",".join(requested),
        "categories": ",".join(categories) or "(all)",
        "limit": args.limit if args.limit is not None else "(all)",
        "include_excluded": bool(args.include_excluded),
        "allow_cloud": bool(args.allow_cloud),
        "force_ocr": bool(args.force_ocr),
        "documents": len(entries),
    })

    written = report.write(args.out)
    if not args.quiet:
        print("已写入 / written:")
        for path in written:
            print(f"  {path}")

    unusable = [slot for slot in slots if not slot.available]
    if unusable:
        print("注意：以下解析器未能运行，已作为结果记录在报告中：")
        _print_slot_status(unusable)

    exit_code = 0
    if args.fail_on_regression:
        baseline_path = Path(args.fail_on_regression)
        try:
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"error: 无法读取 baseline {baseline_path}: {exc}")
            return 1
        findings = compare_to_baseline(
            report, baseline, tolerance=float(args.regression_tolerance)
        )
        if findings:
            print("回归检测发现以下问题 / regressions found:")
            for finding in findings:
                print(f"  - {finding}")
            exit_code = 1
        elif not args.quiet:
            print("回归检测通过 / no regressions vs baseline.")
    return exit_code


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
