"""The bake-off runner: manifest loading, corpus versioning, report assembly.

The property under test throughout is the one the acceptance criteria name:
**a report is only worth reading if it is reproducible and if it cannot
pretend a parser that never ran was tested.** Every test here runs with no
parser installed, no MinerU token, no network and no real file parsing —
documents are hand-built ``ParsedDocument`` values, exactly as the ingestion
tests build them, and parsers are fakes injected as ``ParserSlot`` objects.

That constraint is not incidental. This repository cannot install docling and
has no cloud token, so a test suite that needed either would be a suite that
never ran in CI. It also means the runner's own logic — hashing, coercion,
aggregation, gating, rendering — is exercised end to end without a single
byte of a PDF being opened, which is the only way the parts that *are* ours
get covered at all.

Two behaviours are pinned deliberately, because both are easy to regress and
neither is obvious from reading the report:

* **Unavailability is a row, not a gap.** A parser that cannot run produces
  one metric row per corpus document, carrying a reason code. If a test can
  only see "the docling column disappeared", nobody notices the day the
  reason code stops being written.
* **The cloud is off unless the command line says otherwise.** Even with a
  token, an ``allow_cloud`` privacy mode and consent granted, a run without
  ``--allow-cloud`` must not call MinerU. Config permission is a deployment
  setting; who spends money is a decision of whoever typed the command.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paperwhisperer.core import config  # noqa: E402
from paperwhisperer.ingestion.schema import (  # noqa: E402
    BoundingBox,
    Element,
    ParsedDocument,
    SourceLocator,
    Table,
)

MODULE_PATH = PROJECT_ROOT / "eval" / "run_parser_eval.py"


@pytest.fixture(scope="module")
def runner():
    """Import ``eval/run_parser_eval.py`` as a module.

    ``eval/`` is a script directory rather than a package (no
    ``__init__.py``, and adding one is not this task's business), so the
    module is loaded by path. Loading it under a fixed name — and caching it
    on the fixture — keeps ``isinstance`` checks meaningful across tests and
    keeps the module out of the rest of the suite's import namespace.
    """
    if "run_parser_eval" in sys.modules:
        return sys.modules["run_parser_eval"]
    spec = importlib.util.spec_from_file_location("run_parser_eval", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_parser_eval"] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# Fixtures / doubles
# --------------------------------------------------------------------------


def _write_manifest(tmp_path: Path, payload: str) -> Path:
    """Write a manifest body to ``tmp_path`` and return its path."""
    target = tmp_path / "manifest.yaml"
    target.write_text(payload, encoding="utf-8")
    return target


VALID_MANIFEST = """
schema: paperwhisperer-eval-manifest/0.2
created: '2026-09-29'
corpus_root: corpus
corpus_version: sha256:1111111111111111111111111111111111111111111111111111111111111111
categories:
  digital_single_col:
    quota: 2
    entries:
      - file: digital_single_col/a.pdf
        source: arxiv:1706.03762
        bytes: 100
        sha256: aaaa1111
        pages: 3
        text_layer: true
        chars_total: 1000
        preview: 'alpha'
      - file: digital_single_col/b.pdf
        source: arxiv:1406.2661
        bytes: 200
        sha256: bbbb2222
        pages: 5
        text_layer: true
        chars_total: 2000
  table_dense:
    quota: 1
    entries:
      - file: table_dense/c.pdf
        source: synthetic
        bytes: 300
        sha256: cccc3333
        pages: 2
        text_layer: true
        chars_total: 500
adversarial:
  - file: adversarial/broken.pdf
    bytes: 10
    sha256: dddd4444
    batch_excluded: true
summary:
  total_documents: 3
"""


def _entry(root, category: str, file: str, sha256: str = "hash", **kwargs):
    """A ``ManifestEntry`` that resolves against ``root``.

    Reaches for the already-imported module rather than taking it as an
    argument so call sites stay short; the module fixture guarantees the
    import has happened.
    """
    return sys.modules["run_parser_eval"].ManifestEntry(
        category=category,
        file=file,
        sha256=sha256,
        root=str(root),
        **kwargs,
    )


def _page_element(page: int, text: str = "body text", *, bbox: bool = False, order: int = 0):
    return Element(
        role="paragraph",
        text=text,
        order=order,
        locator=SourceLocator(
            kind="pdf_page",
            page=page,
            bbox=BoundingBox(x0=0.0, y0=0.0, x1=10.0, y1=10.0) if bbox else None,
        ),
    )


def _rich_document(*, pages: int = 3, with_table: bool = True, with_formula: bool = True):
    """A document that should score well on every metric the runner computes.

    Written to be obviously correct rather than realistic: a bake-off test
    whose fixture is subtly broken produces a test that passes for the wrong
    reason.
    """
    elements = [_page_element(page, f"page {page} prose", bbox=True) for page in range(1, pages + 1)]
    if with_table:
        elements.append(Element(
            role="table",
            order=pages,
            table=Table(rows=[["h1", "h2"], ["a", "b"], ["c", "d"]], header_rows=1),
            locator=SourceLocator(
                kind="pdf_page", page=pages,
                bbox=BoundingBox(x0=0.0, y0=0.0, x1=10.0, y1=10.0),
            ),
        ))
    if with_formula:
        elements.append(Element(
            role="formula",
            order=pages + 1,
            latex="\\int_0^1 x dx",
            locator=SourceLocator(
                kind="pdf_page", page=pages,
                bbox=BoundingBox(x0=0.0, y0=0.0, x1=10.0, y1=10.0),
            ),
        ))
    return ParsedDocument(
        document_id="doc-1",
        source_filename="a.pdf",
        media_type="application/pdf",
        page_count=pages,
        elements=elements,
        parser_name="fake",
        parser_version="9.9",
        options_hash="opts",
    )


class _FakeParserResult:
    """Stand-in for a gateway ``ParseResult`` with no import of gateway needed."""

    def __init__(self, status="success", document=None, duration_ms=12, error_message=""):
        self.status = status
        self.document = document
        self.duration_ms = duration_ms
        self.error_message = error_message
        self.parser_version = "9.9"
        self.options_hash = "opts"


def _partial_document(reported: int = 5, present: int = 3):
    """A document that stopped partway — the silent-truncation shape.

    The parse "succeeds", every element it produced is locatable, and the tail
    of the file simply is not there. It is the failure the plan's release gate
    ("silent truncation on eval corpus: 0") exists for, and the fixture the
    regression tests compare against.
    """
    return ParsedDocument(
        document_id="doc-partial",
        page_count=reported,
        elements=[
            _page_element(page, f"page {page}", bbox=True, order=index)
            for index, page in enumerate(range(1, present + 1))
        ],
        parser_name="fake",
        parser_version="9.9",
    )


def _fake_slot(name="fake", *, available=True, document=None, result=None, calls=None):
    """A ``ParserSlot`` whose ``run`` records its calls and returns a document."""
    runner = sys.modules["run_parser_eval"]

    def _run(entry, path, options):
        if calls is not None:
            calls.append((entry.file, str(path), options))
        if result is not None:
            return result
        return _FakeParserResult(document=document or _rich_document())

    return runner.ParserSlot(
        name=name,
        available=available,
        version="9.9",
        run=_run if available else None,
    )


# --------------------------------------------------------------------------
# Manifest loading
# --------------------------------------------------------------------------


def test_load_manifest_reads_entries_and_resolves_corpus_root(runner, tmp_path):
    manifest = runner.load_manifest(_write_manifest(tmp_path, VALID_MANIFEST))

    assert manifest.schema == "paperwhisperer-eval-manifest/0.2"
    assert manifest.created == "2026-09-29"
    assert manifest.corpus_root == (tmp_path / "corpus").resolve()
    assert [entry.file for entry in manifest.regular_entries] == [
        "digital_single_col/a.pdf",
        "digital_single_col/b.pdf",
        "table_dense/c.pdf",
    ]

    first = manifest.entries[0]
    assert first.sha256 == "aaaa1111"
    assert first.pages == 3
    assert first.text_layer is True
    assert first.chars_total == 1000
    # A relative corpus_root resolves against the manifest, not the cwd, so a
    # report generated from another directory still finds the same files.
    assert first.path == (tmp_path / "corpus" / "digital_single_col" / "a.pdf")


def test_load_manifest_drops_adversarial_from_the_batch_pool(runner, tmp_path):
    """``adversarial/`` files stay visible but out of the batch run.

    They are *designed* to fail (truncated, encrypted, a zip bomb), so
    folding them into a failure rate would make every parser look equally
    broken. They are not dropped either — a report that cannot name the files
    it skipped cannot be audited, and the error paths still need verifying.
    """
    manifest = runner.load_manifest(_write_manifest(tmp_path, VALID_MANIFEST))

    assert len(manifest.entries) == 4
    assert len(manifest.regular_entries) == 3

    adversarial = [
        entry for entry in manifest.entries
        if entry.category == runner.ADVERSARIAL_CATEGORY
    ]
    assert [entry.file for entry in adversarial] == ["adversarial/broken.pdf"]
    assert all(entry.batch_excluded for entry in adversarial)
    # The flag is forced on, whether or not the YAML said so: a file in the
    # adversarial bucket must not sneak into a failure rate by omission.
    assert not any(
        entry.file.startswith("adversarial/") for entry in manifest.regular_entries
    )
    # ...and the opt-in is a single flag, which is what the bake-off's
    # error-path section in docs/PARSER_BAKE_OFF.md tells people to run.
    assert {entry.file for entry in manifest.select(include_excluded=True)} >= {
        "adversarial/broken.pdf",
    }


def test_load_manifest_records_batch_excluded_entries_when_present(runner, tmp_path):
    payload = """
schema: paperwhisperer-eval-manifest/0.2
corpus_root: corpus
categories:
  adversarial_included:
    quota: 1
    entries:
      - file: adversarial/zip_bomb.zip
        sha256: eeee5555
        batch_excluded: true
  docx:
    quota: 1
    entries:
      - file: docx/a.docx
        sha256: ffff6666
        pages: 1
"""
    manifest = runner.load_manifest(_write_manifest(tmp_path, payload))

    excluded = [entry for entry in manifest.entries if entry.batch_excluded]
    assert [entry.file for entry in excluded] == ["adversarial/zip_bomb.zip"]
    assert [entry.file for entry in manifest.regular_entries] == ["docx/a.docx"]
    # Opting in puts it back in the batch pool without editing the manifest.
    assert [entry.file for entry in manifest.select(include_excluded=True)] == [
        "adversarial/zip_bomb.zip", "docx/a.docx",
    ]


def test_manifest_select_filters_and_limits_deterministically(runner, tmp_path):
    manifest = runner.load_manifest(_write_manifest(tmp_path, VALID_MANIFEST))

    assert [entry.file for entry in manifest.select(categories=["table_dense"])] == [
        "table_dense/c.pdf",
    ]
    # Sorted by (category, file) before slicing, so --limit 2 is the same two
    # documents on every machine and a smoke run stays comparable to a full one.
    assert [entry.file for entry in manifest.select(limit=2)] == [
        "digital_single_col/a.pdf",
        "digital_single_col/b.pdf",
    ]
    assert manifest.select(categories=["nope"]) == ()


@pytest.mark.parametrize(
    ("body", "fragment"),
    [
        ("- one\n- two\n", "顶层应为映射"),
        ("created: '2026-09-29'\n", "缺少 schema"),
        ("schema: something-else/1.0\ncategories: {}\n", "不支持的 manifest schema"),
        ("schema: paperwhisperer-eval-manifest/0.2\n", "缺少非空的 categories"),
        ("schema: paperwhisperer-eval-manifest/0.2\ncategories: 7\n", "缺少非空的 categories"),
        (
            "schema: paperwhisperer-eval-manifest/0.2\n"
            "categories:\n  docx:\n    entries: not-a-list\n",
            "entries 应为列表",
        ),
        (
            "schema: paperwhisperer-eval-manifest/0.2\n"
            "categories:\n  docx:\n    entries:\n      - just-a-string\n",
            "不是映射",
        ),
        (
            "schema: paperwhisperer-eval-manifest/0.2\n"
            "categories:\n  docx:\n    entries:\n      - source: arxiv:1\n",
            "缺少必填字段 file",
        ),
        (
            "schema: paperwhisperer-eval-manifest/0.2\n"
            "categories:\n  docx:\n    entries:\n      - file: a.pdf\n",
            "缺少必填字段 sha256",
        ),
    ],
)
def test_load_manifest_rejects_malformed_input(runner, tmp_path, body, fragment):
    """Every malformed shape fails loudly and says where.

    A manifest that loads with silently dropped entries produces a bake-off
    that looks complete and is missing a category, which is the exact failure
    a reproducibility criterion exists to prevent.
    """
    with pytest.raises(runner.ManifestError) as excinfo:
        runner.load_manifest(_write_manifest(tmp_path, body))
    assert fragment in str(excinfo.value)


def test_load_manifest_reports_a_missing_file(runner, tmp_path):
    with pytest.raises(runner.ManifestError) as excinfo:
        runner.load_manifest(tmp_path / "nope.yaml")
    assert "找不到语料清单" in str(excinfo.value)


# --------------------------------------------------------------------------
# Corpus version
# --------------------------------------------------------------------------
#
# There are two version strings in play and the tests below keep them apart,
# because conflating them is the failure this whole section exists to prevent:
#
#   manifest.version          the manifest's own `corpus_version` field —
#                             authoritative, stamped into every report
#   manifest.computed_version the runner's own digest over the entries it
#                             loaded — a cross-check on the loader, never an
#                             alternative identity
#
# They are computed over deliberately different inputs, so on any real corpus
# they differ. That is expected, and a test asserts it, so nobody later
# "fixes" the mismatch by making one shadow the other.


def test_entry_digest_is_stable_under_reordering(runner, tmp_path):
    """Same file set, different manifest order, same digest.

    ``eval/build_manifest.py`` rewrites the manifest from scratch, so entry
    order is a property of the generator rather than of the corpus. A digest
    that moved on a re-sort would mark every subsequent report as
    incomparable to the previous one.
    """
    forward = [
        _entry("/c", "cat_a", "a.pdf", "aaaa"),
        _entry("/c", "cat_b", "b.pdf", "bbbb"),
    ]
    reversed_entries = list(reversed(forward))

    assert runner.compute_entry_digest(forward) == runner.compute_entry_digest(reversed_entries)
    # ...and a different manifest root must not change it either: the digest is
    # over content, not over where the files happen to live on this machine.
    moved = [_entry("/somewhere/else", "cat_a", "a.pdf", "aaaa"),
             _entry("/somewhere/else", "cat_b", "b.pdf", "bbbb")]
    assert runner.compute_entry_digest(forward) == runner.compute_entry_digest(moved)


def test_entry_digest_changes_when_one_byte_changes(runner, tmp_path):
    """A single changed digest moves the value.

    This is what makes the stamp a reproducibility guarantee rather than a
    label: replace one corpus file and the two runs name different file sets,
    so their reports cannot be compared.
    """
    base = [
        _entry("/c", "cat_a", "a.pdf", "aaaa"),
        _entry("/c", "cat_b", "b.pdf", "bbbb"),
    ]
    changed = [
        _entry("/c", "cat_a", "a.pdf", "aaaa"),
        _entry("/c", "cat_b", "b.pdf", "bbbc"),
    ]

    assert runner.compute_entry_digest(base) != runner.compute_entry_digest(changed)


def test_entry_digest_changes_when_a_file_is_added_or_removed(runner, tmp_path):
    base = [_entry("/c", "cat_a", "a.pdf", "aaaa")]
    extended = [*base, _entry("/c", "cat_a", "b.pdf", "bbbb")]

    assert runner.compute_entry_digest(base) != runner.compute_entry_digest(extended)


def test_manifest_version_is_read_from_the_corpus_version_field(runner, tmp_path):
    """The stamped version is the manifest's, read verbatim from the file.

    Reading it from anywhere else — reconstructing it, or substituting the
    runner's own digest — puts a number into a field whose whole purpose is
    to say where the corpus identity came from.
    """
    manifest = runner.load_manifest(_write_manifest(tmp_path, VALID_MANIFEST))

    assert manifest.version == (
        "sha256:1111111111111111111111111111111111111111111111111111111111111111"
    )
    assert manifest.computed_version == runner.compute_entry_digest(manifest.entries)
    # Two different numbers under two different names. If this ever starts
    # failing because someone made them equal, read the two docstrings again
    # before "fixing" it.
    assert manifest.version != manifest.computed_version


def test_manifest_rejects_a_malformed_corpus_version(runner, tmp_path):
    """A version string in the wrong shape cannot be cited as an identity."""
    payload = VALID_MANIFEST.replace("sha256:1111", "1.2.3")
    with pytest.raises(runner.ManifestError) as excinfo:
        runner.load_manifest(_write_manifest(tmp_path, payload))
    assert "corpus_version 格式" in str(excinfo.value)


def test_manifest_without_a_corpus_version_is_empty_not_guessed(runner, tmp_path):
    """A missing ``corpus_version`` stays empty; nothing fills it in.

    The runner still has a digest, and using it as the stamp would produce a
    report that looks reproducible while being stamped with a number no other
    tool can reproduce. The report says "n/a" and warns instead.
    """
    payload = "\n".join(
        line for line in VALID_MANIFEST.splitlines()
        if not line.startswith("corpus_version:")
    )
    manifest = runner.load_manifest(_write_manifest(tmp_path, payload))

    assert manifest.version == ""
    assert manifest.computed_version


def test_adversarial_files_join_the_entry_digest(runner, tmp_path):
    """The error-path files are part of the corpus a report ran against.

    They are not part of a *batch* run, but a report whose error-path numbers
    changed should not carry the same digest as one from before the change.
    """
    full = runner.load_manifest(_write_manifest(tmp_path, VALID_MANIFEST))
    assert len(full.computed_version) == 64

    # Dropping the adversarial list changes the digest, because those entries
    # are part of the loaded set even though they are excluded from the run.
    trimmed = _write_manifest(tmp_path, VALID_MANIFEST.split("adversarial:")[0])
    assert runner.load_manifest(trimmed).computed_version != full.computed_version


def test_shipped_manifest_still_loads(runner):
    """The real ``eval/manifest.yaml`` must stay readable.

    A loader that only ever sees a hand-written fixture will happily accept a
    schema the real corpus no longer matches. This test reads the file that
    actually ships, and asserts the two things a bake-off report depends on:
    every category from the plan's section 6 is present, and every entry
    carries the digest the corpus version is computed from. A corpus change
    that drops a category or forgets a sha256 breaks here rather than
    producing a report that is quietly one category short.
    """
    manifest = runner.load_manifest(MODULE_PATH.parent / "manifest.yaml")

    assert manifest.schema.startswith(runner.MANIFEST_SCHEMA_PREFIX)
    assert {entry.category for entry in manifest.regular_entries} == {
        "digital_single_col", "digital_two_col", "digital_complex_layout",
        "formula_dense", "table_dense", "scanned_clear", "scanned_lowq",
        "hybrid_scan_text", "mixed_zh_en", "long_paper", "docx", "pptx",
    }
    assert all(entry.sha256 for entry in manifest.entries)
    assert manifest.computed_version == runner.compute_entry_digest(manifest.entries)
    # The declared summary and the entries have to agree, or the report's
    # document count contradicts the manifest it claims to have read.
    assert manifest.summary.get("total_documents") == len(manifest.regular_entries)


def test_report_against_the_real_corpus_carries_a_corpus_version(runner, tmp_path):
    """The reproducibility stamp must survive on the manifest that ships.

    This is the test that fails if the loader ever goes back to reading a key
    that does not exist (``version``) instead of the manifest's real
    ``corpus_version``: the stamp would silently come out empty, every report
    would still render, and the acceptance criterion "报告可复现（corpus 版本固定）"
    would be satisfied only on paper. A hand-written fixture cannot catch that
    — it has whatever keys the test author put in it — so the check is made
    against ``eval/manifest.yaml`` as committed.

    The rows are injected, so no corpus file is opened and no parser runs.
    """
    manifest = runner.load_manifest(MODULE_PATH.parent / "manifest.yaml")
    slots = [_fake_slot("legacy")]
    rows = runner.run_bakeoff(manifest.select(limit=3), slots)
    report = runner.build_report(manifest, slots, rows)

    assert manifest.version, "真实 manifest 必须带 corpus_version"
    assert report.corpus_version == manifest.version
    assert report.corpus_version.startswith(runner.CORPUS_VERSION_PREFIX)

    # Both values are present, in the artifacts, under distinct names.
    markdown = report.to_markdown()
    assert report.corpus_version in markdown
    assert report.runner_entry_digest in markdown
    assert "authoritative" in markdown
    assert "not** a corpus identity" in markdown

    payload = json.loads(report.to_json())
    assert payload["corpus_version"] == manifest.version
    assert payload["runner_entry_digest"] == manifest.computed_version
    assert payload["corpus_version"] != payload["runner_entry_digest"]


def test_report_warns_when_the_corpus_version_is_missing(runner, tmp_path):
    """An unstamped report says so in the report, not just on the terminal."""
    payload = "\n".join(
        line for line in VALID_MANIFEST.splitlines()
        if not line.startswith("corpus_version:")
    )
    manifest = runner.load_manifest(_write_manifest(tmp_path, payload))
    slots = [_fake_slot("legacy")]
    report = runner.build_report(
        manifest, slots, runner.run_bakeoff(manifest.regular_entries, slots)
    )

    assert report.corpus_version == ""
    markdown = report.to_markdown()
    assert "无法声明可复现的语料版本" in markdown
    assert "`n/a`" in markdown


def test_shipped_manifest_adversarial_is_excluded_by_default(runner):
    """The real corpus keeps its error-path files out of the batch run.

    The check is on the *directory* component: ``gan_generative_adversarial.pdf``
    is a Generative Adversarial Nets paper and belongs in the batch.
    """
    manifest = runner.load_manifest(MODULE_PATH.parent / "manifest.yaml")

    assert all(
        Path(entry.file).parts[0] != "adversarial" for entry in manifest.regular_entries
    )
    assert [
        entry.file for entry in manifest.entries
        if entry.category == runner.ADVERSARIAL_CATEGORY
    ] == [
        "adversarial/corrupted_truncated.pdf",
        "adversarial/encrypted_aes128.pdf",
        "adversarial/fake_extension_actually_text.pdf",
        "adversarial/zip_bomb_nested.zip",
    ]


# --------------------------------------------------------------------------
# Metric derivation
# --------------------------------------------------------------------------


def test_metrics_measure_a_good_document(runner, tmp_path):
    entry = _entry(tmp_path, "table_dense", "c.pdf", "cccc", pages=3, chars_total=1000)
    document = _rich_document(pages=3)

    row = runner.metrics_for_document("fake", entry, document, duration_ms=42)

    assert row.status == runner.STATUS_OK
    assert row.duration_ms == 42
    assert row.page_coverage == 1.0
    assert row.pages_covered == 3
    assert row.reading_order == 1.0
    assert row.reading_order_basis == "pdf_page"
    assert row.tables == 1
    assert row.tables_2d == 1
    assert row.tables_flattened == 0
    assert row.formulas == 1
    assert row.formulas_with_latex == 1
    assert row.locator_coverage == 1.0
    assert row.bbox_locator_ratio == 1.0
    assert row.chars_ratio_vs_manifest is not None


def test_page_coverage_is_none_for_unpaginated_formats(runner, tmp_path):
    """A DOCX has no pages to miss, so coverage is abstained, not zero.

    Scoring it 0% would drag every cross-format average down with a number
    that means nothing, and would make a parser that handles DOCX perfectly
    look like the worst performer in the table.
    """
    entry = _entry(tmp_path, "docx", "a.docx", "aaaa")
    document = ParsedDocument(
        page_count=0,
        elements=[
            Element(role="paragraph", text="hello", locator=SourceLocator(kind="docx_element", paragraph=1)),
            Element(role="paragraph", text="world", locator=SourceLocator(kind="docx_element", paragraph=2)),
        ],
        parser_name="fake",
    )

    row = runner.metrics_for_document("fake", entry, document)

    assert row.page_coverage is None
    assert row.reading_order_basis == "docx_element"
    assert row.reading_order == 1.0


def test_reading_order_scores_inversions_down(runner, tmp_path):
    """Page order 1, 3, 2 is one inversion out of two transitions."""
    entry = _entry(tmp_path, "digital_single_col", "a.pdf", "aaaa", pages=3)
    scrambled = ParsedDocument(
        page_count=3,
        elements=[
            _page_element(1, order=0),
            _page_element(3, order=1),
            _page_element(2, order=2),
        ],
        parser_name="fake",
    )

    row = runner.metrics_for_document("fake", entry, scrambled)

    assert row.reading_order == 0.5
    assert row.reading_order_inversions == 1
    # Page *coverage* is unaffected: all three pages produced text even though
    # two of them arrived in the wrong place.
    assert row.page_coverage == 1.0


def test_flattened_tables_are_counted_separately(runner, tmp_path):
    """A table that arrived as prose is not a table that was found.

    Cross-column retrieval is impossible on a flattened grid, and the
    difference between "no tables here" and "tables destroyed" is the reason
    the ``table_dense`` category exists at all.
    """
    entry = _entry(tmp_path, "table_dense", "c.pdf", "cccc", pages=1)
    document = ParsedDocument(
        page_count=1,
        elements=[
            _page_element(1, order=0),
            Element(role="table", text="a b c", order=1,
                    locator=SourceLocator(kind="pdf_page", page=1)),
        ],
        parser_name="fake",
    )

    row = runner.metrics_for_document("fake", entry, document)

    assert row.tables == 1
    assert row.tables_2d == 0
    assert row.tables_flattened == 1


def test_locator_metrics_separate_page_from_bbox(runner, tmp_path):
    """Resolving to a page is not the same as resolving to a rectangle.

    The P0 exit criteria ask for page *and* bbox on PDFs, so a parser that
    gets the page right and the box wrong satisfies half a citation and
    cannot draw a highlight. The two have to be countable separately.
    """
    entry = _entry(tmp_path, "digital_single_col", "a.pdf", "aaaa", pages=2)
    document = ParsedDocument(
        page_count=2,
        elements=[
            _page_element(1, bbox=True, order=0),
            _page_element(2, bbox=False, order=1),
        ],
        parser_name="fake",
    )

    row = runner.metrics_for_document("fake", entry, document)

    assert row.locator_coverage == 1.0
    assert row.bbox_locator_ratio == 0.5


def test_unresolvable_locators_lower_coverage(runner, tmp_path):
    entry = _entry(tmp_path, "docx", "a.docx", "aaaa")
    document = ParsedDocument(
        page_count=0,
        elements=[
            Element(role="paragraph", text="located",
                    locator=SourceLocator(kind="docx_element", paragraph=1)),
            Element(role="paragraph", text="lost", locator=SourceLocator(kind="unknown")),
        ],
        parser_name="fake",
    )

    row = runner.metrics_for_document("fake", entry, document)

    assert row.locator_coverage == 0.5


# --------------------------------------------------------------------------
# Outcome coercion
# --------------------------------------------------------------------------


def test_coerce_accepts_the_doclings_ok_status(runner, tmp_path):
    """The docling adapter spells success ``"ok"``; the gateway spells it ``"success"``.

    A bake-off that treated the other spelling as a failure would report
    docling as 100% broken on a machine where it worked perfectly, and that
    is precisely the kind of number that gets baked into a routing decision.
    """
    document = _rich_document()
    status, coerced, error, duration, _version = runner._coerce_outcome(
        _FakeParserResult(status="ok", document=document, duration_ms=7), "docling"
    )

    assert status == runner.STATUS_OK
    assert coerced is document
    assert duration == 7
    assert error == ""


def test_coerce_reads_the_mineru_tuple_convention(runner, tmp_path):
    """MinerU returns ``(document, audit_record)`` and never raises."""

    class _Record:
        message = "quota exhausted"

    document = _rich_document()
    status, coerced, error, _ms, _v = runner._coerce_outcome(
        (document, _Record()), "mineru_cloud"
    )
    assert status == runner.STATUS_OK
    assert coerced is document
    assert error == "quota exhausted"

    status, coerced, error, _ms, _v = runner._coerce_outcome((None, _Record()), "mineru_cloud")
    assert status == runner.STATUS_BROKEN
    assert coerced is None
    assert error == "quota exhausted"


def test_coerce_treats_success_without_a_document_as_failure(runner, tmp_path):
    status, document, error, _ms, _v = runner._coerce_outcome(
        _FakeParserResult(status="success", document=None), "legacy"
    )
    assert status == runner.STATUS_BROKEN
    assert document is None
    assert "没有文档" in error


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------


def test_run_bakeoff_produces_a_row_per_document_per_parser(runner, tmp_path):
    calls: list[tuple[str, str, object]] = []
    slots = [_fake_slot("alpha", calls=calls), _fake_slot("beta", calls=calls)]
    entries = [
        _entry(tmp_path, "cat_a", "a.pdf", "aaaa", pages=1),
        _entry(tmp_path, "cat_b", "b.pdf", "bbbb", pages=1),
    ]

    rows = runner.run_bakeoff(entries, slots)

    assert len(rows) == 4
    assert {row.parser for row in rows} == {"alpha", "beta"}
    assert {row.document for row in rows} == {"a.pdf", "b.pdf"}
    assert len(calls) == 4
    # The adapter-facing options proxy must still answer the gateway's own
    # cache-key method, or the bake-off would not be measuring the same parse
    # identity the orchestrator caches on.
    assert calls[0][2].compute_options_hash()


def test_run_bakeoff_records_an_exception_as_a_failure_row(runner, tmp_path):
    """A parser that raises is a recorded result, not an aborted run.

    Half the corpus failing is a finding about the parser; the run stopping
    is a finding about the runner, and only the first one is useful.
    """
    def _boom(entry, path, options):
        raise RuntimeError("torch exploded")

    slot = runner.ParserSlot(name="docling", available=True, version="1.0", run=_boom)
    rows = runner.run_bakeoff([_entry(tmp_path, "cat_a", "a.pdf", "aaaa", pages=1)], [slot])

    assert len(rows) == 1
    assert rows[0].status == runner.STATUS_BROKEN
    assert rows[0].reason_code == runner.REASON_PARSE_ERROR
    assert "torch exploded" in rows[0].error


def test_unavailable_parser_yields_a_row_per_document(runner, tmp_path):
    """The core anti-omission property.

    A parser that could not run is represented in the data as a full column
    of ``unavailable`` rows carrying a reason code — not as a missing column.
    A reader must be able to tell "docling found nothing" from "docling was
    never asked", and only the row makes the difference visible.
    """
    calls: list = []
    unavailable = runner.ParserSlot(
        name="docling",
        available=False,
        reason="未安装 docling。",
        reason_code=runner.REASON_DEPENDENCY_MISSING,
        version="unknown",
        run=None,
    )
    available = _fake_slot("legacy", calls=calls)
    entries = [
        _entry(tmp_path, "cat_a", "a.pdf", "aaaa", pages=1),
        _entry(tmp_path, "cat_b", "b.pdf", "bbbb", pages=1),
    ]

    rows = runner.run_bakeoff(entries, [available, unavailable])
    docling_rows = [row for row in rows if row.parser == "docling"]

    assert len(docling_rows) == len(entries)
    assert all(row.status == runner.STATUS_UNAVAILABLE for row in docling_rows)
    assert all(row.reason_code == runner.REASON_DEPENDENCY_MISSING for row in docling_rows)
    # The unavailable parser contributes no duration and no numbers, so it can
    # never be mistaken for a parser that ran and found nothing.
    assert all(row.duration_ms == 0 and row.chars == 0 for row in docling_rows)


def test_unsupported_media_type_is_distinct_from_failure(runner, tmp_path):
    """A parser declining a format is not a parser that broke on it.

    They need different follow-up — "add an adapter" versus "fix the
    backend" — so they are different statuses rather than one bucket.
    """

    class _PickyParser:
        def supports(self, media_type):
            return media_type == "application/pdf"

    slot = runner.ParserSlot(
        name="mineru_cloud",
        available=True,
        version="1.0",
        run=lambda entry, path, options: _FakeParserResult(),
        parser=_PickyParser(),
    )
    rows = runner.run_bakeoff([_entry(tmp_path, "docx", "a.docx", "aaaa")], [slot])

    assert rows[0].status == runner.STATUS_UNSUPPORTED
    assert rows[0].reason_code == runner.REASON_UNSUPPORTED_MEDIA_TYPE
    assert not rows[0].is_attempted


# --------------------------------------------------------------------------
# Aggregation and report
# --------------------------------------------------------------------------


def _aggregate_for(runner, rows, parser: str):
    """Aggregate one parser's rows.

    ``aggregate_rows`` deliberately takes a flat list and does no grouping: a
    bake-off's summary is per parser, and a helper that silently mixed them
    would average across backends.
    """
    return runner.aggregate_rows([row for row in rows if row.parser == parser])


def test_aggregate_excludes_unavailable_from_the_failure_rate(runner, tmp_path):
    """A parser that never ran cannot have a failure rate.

    Counting 0/0 as 0% would read as "this parser never fails", which is the
    exact opposite of the truth, and would let an unevaluated parser win a
    comparison it was not in.
    """
    good = _fake_slot("legacy")
    missing = runner.ParserSlot(
        name="docling", available=False, reason="no docling",
        reason_code=runner.REASON_DEPENDENCY_MISSING,
    )
    entries = [
        _entry(tmp_path, "cat_a", "a.pdf", "aaaa", pages=1),
        _entry(tmp_path, "cat_b", "b.pdf", "bbbb", pages=1),
    ]

    rows = runner.run_bakeoff(entries, [good, missing])

    legacy = _aggregate_for(runner, rows, "legacy")
    assert legacy["failure_rate"] == 0.0
    assert legacy["usable"] == 2

    docling = _aggregate_for(runner, rows, "docling")
    assert docling["unavailable"] == 2
    assert docling["attempted"] == 0
    assert docling["failure_rate"] is None
    assert docling["mean_page_coverage"] is None


def test_aggregate_failure_rate_counts_only_attempted(runner, tmp_path):
    def _boom(entry, path, options):
        raise RuntimeError("nope")

    slots = [
        _fake_slot("half"),
        runner.ParserSlot(name="broken", available=True, run=_boom),
    ]
    entries = [
        _entry(tmp_path, "cat_a", "a.pdf", "aaaa", pages=1),
        _entry(tmp_path, "cat_b", "b.pdf", "bbbb", pages=1),
    ]

    rows = runner.run_bakeoff(entries, slots)

    assert _aggregate_for(runner, rows, "half")["failure_rate"] == 0.0
    broken = _aggregate_for(runner, rows, "broken")
    assert broken["failure_rate"] == 1.0
    assert broken["usable"] == 0
    assert broken["median_duration_ms"] is None


def test_aggregate_uses_ratio_of_totals_for_tables(runner, tmp_path):
    """A document with one table must not outweigh one with forty.

    Averaging per-document ratios would let a run of single-table documents
    decide a number about how often table structure survives.
    """
    many_tables = ParsedDocument(
        page_count=1,
        elements=[
            _page_element(1, order=0),
            *[
                Element(
                    role="table", order=index,
                    table=Table(rows=[["a", "b"], ["c", "d"]]),
                    locator=SourceLocator(kind="pdf_page", page=1),
                )
                for index in range(1, 41)
            ],
        ],
        parser_name="fake",
    )
    one_flattened = ParsedDocument(
        page_count=1,
        elements=[
            _page_element(1, order=0),
            Element(role="table", text="a b c", order=1,
                    locator=SourceLocator(kind="pdf_page", page=1)),
        ],
        parser_name="fake",
    )

    def _run(document):
        rows = runner.metrics_for_document(
            "p", _entry(tmp_path, "t", "t.pdf", "aaaa", pages=1), document
        )
        return runner.aggregate_rows([rows])

    # 40 of 41 tables keep their grid.
    assert _run(many_tables)["table_2d_ratio"] == 1.0
    assert _run(one_flattened)["table_2d_ratio"] == 0.0


def _assemble_report(runner, tmp_path, *, document=None):
    """A small but complete report: one parser, three documents, one missing."""
    manifest = runner.load_manifest(_write_manifest(tmp_path, VALID_MANIFEST))
    slots = [
        _fake_slot("legacy", document=document),
        runner.ParserSlot(
            name="docling",
            available=False,
            reason="未安装 docling。",
            reason_code=runner.REASON_DEPENDENCY_MISSING,
            version="unknown",
        ),
    ]
    rows = runner.run_bakeoff(manifest.regular_entries, slots)
    report = runner.build_report(
        manifest, slots, rows,
        options={"parsers": "legacy,docling"},
        generated_at="2026-01-01T00:00:00+00:00",
    )
    return manifest, slots, rows, report


def test_report_is_stamped_with_the_corpus_version(runner, tmp_path):
    """The reproducibility criterion, in the artifact itself.

    A report that does not name the corpus cannot be compared with the next
    one, and a bake-off feeding a routing decision has to be checkable
    months later against the bytes it actually ran on.
    """
    _manifest, _slots, _rows, report = _assemble_report(runner, tmp_path)

    assert report.corpus_version
    markdown = report.to_markdown()
    assert report.corpus_version in markdown
    assert report.corpus_version in json.loads(report.to_json())["corpus_version"]


def test_report_markdown_records_unavailable_parsers(runner, tmp_path):
    _manifest, _slots, _rows, report = _assemble_report(runner, tmp_path)
    markdown = report.to_markdown()

    assert "无法运行的解析器" in markdown
    assert "dependency_missing" in markdown
    assert "未安装 docling。" in markdown
    # And the parser still has its own per-document section, showing the
    # reason on every row rather than only once in a header.
    assert "`docling`" in markdown
    assert markdown.count("unavailable") >= 3


def test_report_markdown_and_json_summarise_parsers(runner, tmp_path):
    _manifest, _slots, _rows, report = _assemble_report(runner, tmp_path)

    markdown = report.to_markdown()
    assert "解析器汇总" in markdown
    assert "分类别对比" in markdown
    assert "逐文档明细" in markdown

    payload = json.loads(report.to_json())
    assert payload["corpus_version"] == report.corpus_version
    assert set(payload["summary"]) == {"legacy", "docling"}
    assert payload["summary"]["legacy"]["usable"] == 3
    docling = next(item for item in payload["parsers"] if item["name"] == "docling")
    assert docling["available"] is False
    assert docling["reason_code"] == runner.REASON_DEPENDENCY_MISSING


def test_report_csv_has_a_stable_header_and_one_row_per_pair(runner, tmp_path):
    _manifest, _slots, rows, report = _assemble_report(runner, tmp_path)
    lines = report.to_csv().splitlines()

    assert lines[0].split(",") == list(runner.CSV_COLUMNS)
    assert len(lines) - 1 == len(rows) == 6
    # A regenerated report must diff cleanly against the previous one, so an
    # empty ratio is written as an empty cell rather than the word "None".
    assert "None" not in lines[1]


def test_report_write_emits_three_artifacts(runner, tmp_path):
    _manifest, _slots, _rows, report = _assemble_report(runner, tmp_path)
    out_dir = tmp_path / "out"

    written = report.write(out_dir)

    assert [path.name for path in written] == [
        "parser_eval.md", "parser_eval.csv", "parser_eval.json",
    ]
    assert all(path.is_file() for path in written)
    # Fixed filenames, not timestamped: comparing two runs is the point.
    assert sorted(p.name for p in out_dir.iterdir()) == [
        "parser_eval.csv", "parser_eval.json", "parser_eval.md",
    ]


# --------------------------------------------------------------------------
# Regression gate
# --------------------------------------------------------------------------


def test_baseline_comparison_skips_a_different_corpus(runner, tmp_path):
    """A corpus swap is not a regression, and must not be reported as one.

    Comparing aggregates across two corpora would surface every metric that
    the corpus happened to emphasize, and send someone hunting for a parser
    bug that does not exist.
    """
    _manifest, _slots, _rows, report = _assemble_report(runner, tmp_path)
    baseline = {
        "corpus_version": "0" * 64,
        "summary": {"legacy": {"mean_locator_coverage": 1.0, "failure_rate": 0.0}},
    }

    findings = runner.compare_to_baseline(report, baseline)

    assert len(findings) == 1
    assert "语料版本不同" in findings[0]


def test_baseline_comparison_reports_a_real_drop(runner, tmp_path):
    """A truncated parse against a baseline that saw every page is a drop."""
    _manifest, _slots, _rows, report = _assemble_report(
        runner, tmp_path, document=_partial_document()
    )
    assert report.summary()["legacy"]["mean_page_coverage"] == 0.6
    baseline = {
        "corpus_version": report.corpus_version,
        "summary": {"legacy": {"mean_page_coverage": 1.0, "failure_rate": 0.0}},
    }

    findings = runner.compare_to_baseline(report, baseline)

    assert any("mean_page_coverage" in finding for finding in findings)
    # The unavailable parser is neither a regression nor a missing baseline:
    # it was not evaluated, so there is nothing to compare.
    assert not any("docling" in finding for finding in findings)


def test_baseline_tolerance_can_absorb_a_small_drop(runner, tmp_path):
    """The default tolerance is zero; a gate that tolerates a drop is not a gate."""
    _manifest, _slots, _rows, report = _assemble_report(
        runner, tmp_path, document=_partial_document()
    )
    baseline = {
        "corpus_version": report.corpus_version,
        "summary": {"legacy": {"mean_page_coverage": 0.99}},
    }

    assert runner.compare_to_baseline(report, baseline, tolerance=0.5) == []
    assert runner.compare_to_baseline(report, baseline, tolerance=0.0)


# --------------------------------------------------------------------------
# Parser availability and the cloud gate
# --------------------------------------------------------------------------


def test_default_cli_requests_legacy_only(runner):
    """The safe default is the only parser with no optional dependency."""
    args = runner.build_arg_parser().parse_args([])

    assert args.parsers == "legacy"
    assert args.allow_cloud is False
    assert args.fail_on_regression == ""


def test_mineru_slot_is_blocked_by_the_privacy_mode(runner, monkeypatch):
    monkeypatch.setattr(config, "MINERU_API_TOKEN", "a-token")
    monkeypatch.setattr(config, "DOCUMENT_PRIVACY_MODE", "local_only")
    monkeypatch.setattr(config, "DOCUMENT_CLOUD_CONSENT_GRANTED", True)

    slot = runner._mineru_slot(allow_cloud=True)

    assert slot.available is False
    assert slot.reason_code == runner.REASON_PRIVACY_MODE
    # A blocked slot has no callable at all: even a mistaken caller cannot
    # reach the network through it.
    assert slot.run is None


def test_mineru_slot_is_blocked_without_a_token(runner, monkeypatch):
    monkeypatch.setattr(config, "MINERU_API_TOKEN", "   ")
    monkeypatch.setattr(config, "DOCUMENT_PRIVACY_MODE", "allow_cloud")
    monkeypatch.setattr(config, "DOCUMENT_CLOUD_CONSENT_GRANTED", True)

    slot = runner._mineru_slot(allow_cloud=True)

    assert slot.available is False
    assert slot.reason_code == runner.REASON_NO_TOKEN


def test_mineru_slot_is_available_when_everything_is_configured(runner, monkeypatch):
    monkeypatch.setattr(config, "MINERU_API_TOKEN", "a-token")
    monkeypatch.setattr(config, "DOCUMENT_PRIVACY_MODE", "allow_cloud")
    monkeypatch.setattr(config, "DOCUMENT_CLOUD_CONSENT_GRANTED", True)

    slot = runner._mineru_slot(allow_cloud=True)

    assert slot.available is True
    assert slot.reason_code == runner.REASON_OK
    assert slot.run is not None
    # Constructing the slot must not call anything; only a parse would.
    assert slot.version


def test_cli_never_reaches_the_cloud_without_allow_cloud(runner, tmp_path, monkeypatch):
    """The default path is offline even when the environment says otherwise.

    Every *deployment-level* cloud gate is opened here — token,
    ``allow_cloud`` privacy mode and explicit user consent — so the only
    thing left that can stop the call is the ``--allow-cloud`` flag. Both the
    adapter's own ``parse`` and ``urllib.request.urlopen`` are replaced with
    functions that raise, so "no cloud call" is verified rather than assumed;
    and the resulting report has to carry a full ``unavailable`` column with
    the reason code that says which gate closed.
    """
    monkeypatch.setattr(config, "MINERU_API_TOKEN", "a-token")
    monkeypatch.setattr(config, "DOCUMENT_PRIVACY_MODE", "allow_cloud")
    monkeypatch.setattr(config, "DOCUMENT_CLOUD_CONSENT_GRANTED", True)

    def _explode(*args, **kwargs):
        raise AssertionError("评测默认路径不得发起任何云端调用")

    from paperwhisperer.ingestion.adapters.mineru_cloud import MineruCloudParser

    monkeypatch.setattr(MineruCloudParser, "parse", _explode)
    monkeypatch.setattr("urllib.request.urlopen", _explode)

    calls: list = []
    monkeypatch.setattr(runner, "_legacy_slot", lambda: _fake_slot("legacy", calls=calls))

    out_dir = tmp_path / "out"
    exit_code = runner.main([
        "--manifest", str(_write_manifest(tmp_path, VALID_MANIFEST)),
        "--parsers", "legacy,mineru_cloud",
        "--out", str(out_dir),
    ])

    assert exit_code == 0
    assert calls, "legacy 应当仍然被实际调用"

    payload = json.loads((out_dir / "parser_eval.json").read_text(encoding="utf-8"))
    mineru = next(item for item in payload["parsers"] if item["name"] == "mineru_cloud")
    assert mineru["available"] is False
    assert mineru["reason_code"] == runner.REASON_CLOUD_NOT_ALLOWED

    mineru_rows = [row for row in payload["rows"] if row["parser"] == "mineru_cloud"]
    assert len(mineru_rows) == 3
    assert all(row["status"] == runner.STATUS_UNAVAILABLE for row in mineru_rows)
    assert all(row["duration_ms"] == 0 for row in mineru_rows)


def test_cli_writes_a_report_even_when_a_parser_is_missing(runner, tmp_path, monkeypatch):
    """A real unavailable slot survives all the way into the written report.

    ``build_parser_slots`` is not stubbed here, so the mineru gate is
    evaluated for real (no token in the test environment) and the reason code
    in the report is the one the environment actually produced.
    """
    monkeypatch.setattr(config, "MINERU_API_TOKEN", "")

    out_dir = tmp_path / "out"
    exit_code = runner.main([
        "--manifest", str(_write_manifest(tmp_path, VALID_MANIFEST)),
        "--parsers", "legacy,mineru_cloud",
        "--out", str(out_dir),
    ])

    assert exit_code == 0
    markdown = (out_dir / "parser_eval.md").read_text(encoding="utf-8")
    assert runner.REASON_NO_TOKEN in markdown
    assert "无法运行的解析器" in markdown


def test_cli_rejects_an_unknown_parser_name(runner, capsys):
    exit_code = runner.main(["--parsers", "not-a-parser"])

    assert exit_code == 2
    assert "未知解析器" in capsys.readouterr().out


def test_cli_reports_a_malformed_manifest(runner, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner, "_legacy_slot", lambda: _fake_slot("legacy"))
    target = _write_manifest(tmp_path, "schema: wrong/1.0\n")

    exit_code = runner.main(["--manifest", str(target), "--out", str(tmp_path / "out")])

    assert exit_code == 1
    assert "error" in capsys.readouterr().out
    assert not (tmp_path / "out").exists(), "语料不可用时不应写出报告"


def test_cli_fail_on_regression_exits_nonzero(runner, tmp_path, monkeypatch):
    """The gate has to actually fail the command, not just print a note."""
    monkeypatch.setattr(
        runner, "_legacy_slot",
        lambda: _fake_slot("legacy", document=_partial_document()),
    )
    out_dir = tmp_path / "out"
    argv = [
        "--manifest", str(_write_manifest(tmp_path, VALID_MANIFEST)),
        "--out", str(out_dir),
    ]
    assert runner.main(argv) == 0

    # The first run is its own baseline: identical numbers, nothing to find.
    own_output = out_dir / "parser_eval.json"
    assert runner.main([*argv, "--fail-on-regression", str(own_output)]) == 0

    # A baseline that saw every page, against a parser that now loses two of
    # five, is a regression the command must surface with a non-zero exit.
    payload = json.loads(own_output.read_text(encoding="utf-8"))
    assert payload["summary"]["legacy"]["mean_page_coverage"] == 0.6
    payload["summary"]["legacy"]["mean_page_coverage"] = 1.0
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(payload), encoding="utf-8")

    assert runner.main([*argv, "--fail-on-regression", str(baseline_path)]) == 1


def test_cli_fail_on_regression_refuses_to_compare_across_corpora(runner, tmp_path, monkeypatch):
    """A baseline from another corpus makes the gate fail loudly, not silently.

    Passing would be worse than failing: the whole point of the flag is to
    stop a corpus swap from being mistaken for a parser result, and a
    "no regressions" line printed over a comparison that never happened is
    exactly the kind of false assurance the gate exists to prevent.
    """
    monkeypatch.setattr(runner, "_legacy_slot", lambda: _fake_slot("legacy"))
    out_dir = tmp_path / "out"
    argv = ["--manifest", str(_write_manifest(tmp_path, VALID_MANIFEST)), "--out", str(out_dir)]
    assert runner.main(argv) == 0

    payload = json.loads((out_dir / "parser_eval.json").read_text(encoding="utf-8"))
    payload["corpus_version"] = "0" * 64
    payload["summary"]["legacy"]["mean_page_coverage"] = 0.0
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(payload), encoding="utf-8")

    assert runner.main([*argv, "--fail-on-regression", str(baseline_path)]) == 1
