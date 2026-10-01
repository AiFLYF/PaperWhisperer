"""The business layer must not import a binary-document library.

P0 exit criterion, verbatim from the plan:

    业务层不再 import pypdf / python-docx / python-pptx
    (旧分支仅存于 legacy adapter)

This file is what stops that regressing. It is deliberately two-sided:

* a **source-level** test that reads the modules and walks their import
  statements with :mod:`ast`, which is the only check that also sees an import
  nested inside a function body. A runtime check cannot: a function-local
  ``from pypdf import PdfReader`` works perfectly well, so nothing about
  importing the module and looking at ``sys.modules`` would notice it.
* a **behavioural** test that runs a real analysis and proves the text the
  prompts see is the text the old loader produced. The criterion is only
  worth anything if the migration was behaviour-preserving; a version of this
  file that asserted only the absence of imports would pass just as happily
  against a shim that returned the empty string.

The tests that would otherwise need a live LLM use the same stub-the-client
pattern as ``test_orchestrator.py``, and the ones that need a cloud parse
inject a fake cloud parser rather than reaching for a network.
"""

from __future__ import annotations

import ast
import os
import pathlib

import pytest

from paperwhisperer.analysis import sections
from paperwhisperer.analysis.orchestrator import (
    DocumentAnalyzer,
    build_parse_provenance,
    build_section_result,
)
from paperwhisperer.core.text import clean_extracted_text
from paperwhisperer.documents.loader import DocumentLoader, TextChunker
from paperwhisperer.ingestion import parse_document
from paperwhisperer.ingestion.gateway import media_type_for_path
from paperwhisperer.ingestion.orchestrator import CloudFallbackDecision
from paperwhisperer.ingestion.schema import Element, ParsedDocument, SourceLocator

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]

#: The modules whose imports the criterion is about. Both are in the business
#: layer: one is the flat-text façade every caller goes through, the other is
#: the orchestrator that drives every LLM call.
BUSINESS_LAYER_MODULES = (
    "paperwhisperer/documents/loader.py",
    "paperwhisperer/analysis/orchestrator.py",
)

#: Import roots the criterion names. Matched on the *root* package so that
#: ``docx.text.paragraph`` and ``pypdf.generic`` are caught too — a test that
#: only matched the exact string would be defeated by a submodule import.
FORBIDDEN_IMPORT_ROOTS = frozenset({"pypdf", "docx", "pptx"})

#: The one file where these imports are allowed to survive, named so the
#: failure message can point at the escape hatch rather than just say "no".
LEGACY_ADAPTER = "paperwhisperer/ingestion/adapters/legacy_loader.py"

CRITERION = (
    "P0 exit criterion: 业务层不再 import pypdf / python-docx / python-pptx "
    f"(旧分支仅存于 legacy adapter). {LEGACY_ADAPTER} is the only module allowed "
    "to import them; every other module under paperwhisperer/ must parse files "
    "through paperwhisperer.ingestion.parse_document."
)


def _imported_roots(source: str, filename: str) -> set[str]:
    """Every module root imported anywhere in ``source``, at any nesting depth.

    Walks the whole tree rather than only ``tree.body``, so a function-local or
    class-level import is found. ``from a.b import c`` yields ``a``; ``import
    a.b`` yields ``a``; relative imports yield their own dotted name, which can
    never collide with the three third-party roots this test is about.
    """
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:  # pragma: no cover - a syntax error is its own failure
        raise AssertionError(f"{filename} could not be parsed: {exc}") from exc

    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.fixture
def analyzer(monkeypatch, runtime_dirs):
    """A real ``DocumentAnalyzer`` with a stubbed LLM client.

    Real analyzer, real orchestrator, real gateway, real legacy adapter — only
    the model call is faked. A stubbed ``DocumentLoader`` would defeat the point
    of this file: every assertion below is about what actually came out of the
    parse, so anything short of the real chain would make them vacuous.

    It also depends on ``runtime_dirs`` so that every test reaching a real parse
    gets its own upload/output/context folders. That is not tidiness: a real
    parse writes to the parse cache under ``CONTEXT_FOLDER``, so without it this
    file would read and write the developer's real ``context/`` — and, worse, a
    cached parse from one test would silently satisfy another and make a real
    assertion vacuous. Depending on the fixture rather than requesting
    ``runtime_dirs`` in each test is what stops that from being reintroduced by
    the next test added here.
    """
    monkeypatch.setattr(
        "paperwhisperer.analysis.orchestrator.LLMClient",
        lambda api_key: _StubClient(api_key),
    )
    return DocumentAnalyzer("test-key")


#: The criterion, narrowed to what it was actually about.
#:
#: ``pypdf`` / ``python-docx`` / ``python-pptx`` are named because the *old
#: branch* — reading a document directly, outside the parser chain — had to be
#: confined to the legacy adapter. Two operations that are not that branch have
#: since needed the same libraries, and pretending they do not exist would make
#: the rule a letter rather than a purpose:
#:
#:   * ``ingestion/limits.py`` counts a PDF's pages so an over-long document can
#:     be refused *before* a parser is handed it. Enforcing a page ceiling
#:     requires reading the file; the ceiling exists precisely to stop that read
#:     from being expensive. It reads structure, never content.
#:   * ``analysis/service.py`` **writes** a DOCX. python-docx is a declared
#:     dependency; the plan asks for a DOCX export; writing one is not parsing.
#:
#: This is an exception list and it is a trap, deliberately. It is asserted to
#: be exactly these two entries, each with its reason, so deleting the code that
#: needs one fails the suite (a stale exception is a hole nobody is guarding),
#: and adding a new use fails the suite until somebody writes down why. That
#: keeps the property the original whole-tree scan was written to protect --
#: the rule cannot quietly become "the files somebody remembered to check" --
#: while keeping it pointed at parsing rather than at a library's mere presence.
CRITERION_EXCEPTIONS: dict[str, str] = {
    "paperwhisperer/ingestion/limits.py": (
        "counts PDF page structure to enforce MAX_DOCUMENT_PAGES before a "
        "parser runs; reads metadata, never content"
    ),
    "paperwhisperer/analysis/service.py": (
        "writes the DOCX export with python-docx; writing a report is not "
        "reading a document"
    ),
}


class TestBusinessLayerImportsNoParserLibrary:
    """The source-level half of the criterion."""

    @pytest.mark.parametrize("relative_path", BUSINESS_LAYER_MODULES)
    def test_no_pypdf_docx_or_pptx_import(self, relative_path):
        path = PROJECT_ROOT / relative_path
        assert path.is_file(), f"{CRITERION} (missing module: {relative_path})"

        found = _imported_roots(path.read_text(encoding="utf-8"), str(path)) & FORBIDDEN_IMPORT_ROOTS

        assert not found, (
            f"{CRITERION} {relative_path} imports {sorted(found)}. Parsing belongs to "
            f"{LEGACY_ADAPTER} and is reached through "
            "paperwhisperer.ingestion.parse_document; if a helper genuinely needs a "
            "raw reader, it belongs in the adapter, not in the business layer."
        )

    def test_the_whole_business_layer_is_clean(self):
        """Every module under ``paperwhisperer/`` except the legacy adapter.

        Broader than the two named modules on purpose: the criterion is about
        the business layer as a whole, and a per-file allowlist is how such a
        rule decays into "the files somebody remembered to check".
        """
        offenders: dict[str, set[str]] = {}
        for path in sorted((PROJECT_ROOT / "paperwhisperer").rglob("*.py")):
            relative = path.relative_to(PROJECT_ROOT).as_posix()
            if relative == LEGACY_ADAPTER:
                continue
            found = _imported_roots(path.read_text(encoding="utf-8"), str(path))
            if found & FORBIDDEN_IMPORT_ROOTS and relative not in CRITERION_EXCEPTIONS:
                offenders[relative] = found & FORBIDDEN_IMPORT_ROOTS

        assert not offenders, f"{CRITERION} Offending modules: {offenders}"

        # Each exception must still be *used*, or it is a hole left open by a
        # refactor nobody noticed.
        for relative, reason in CRITERION_EXCEPTIONS.items():
            path = PROJECT_ROOT / relative
            assert path.is_file(), (
                f"{CRITERION} 例外声明了 {relative}，但文件不存在。"
                "删掉需要它的代码后，请一并删掉这条例外——否则规则上留了一个"
                "没人守着的口子。"
            )
            found = _imported_roots(path.read_text(encoding="utf-8"), str(path))
            assert found & FORBIDDEN_IMPORT_ROOTS, (
                f"{CRITERION} {relative} 不再使用这些库，但仍保留着它的例外声明"
                f"（{reason}）。请删除该例外。"
            )

    def test_the_legacy_adapter_is_where_the_imports_live(self):
        """The escape hatch still holds the libraries it was supposed to keep.

        Without this, the tests above would also pass against a tree where the
        adapters had been deleted — which would be a *regression* dressed as
        compliance, since the legacy parser is the reason the whole of Phase 0
        works on a machine with no docling installed.
        """
        path = PROJECT_ROOT / LEGACY_ADAPTER
        assert path.is_file(), f"{CRITERION} (missing module: {LEGACY_ADAPTER})"

        found = _imported_roots(path.read_text(encoding="utf-8"), str(path)) & FORBIDDEN_IMPORT_ROOTS

        assert found == FORBIDDEN_IMPORT_ROOTS, (
            f"{CRITERION} {LEGACY_ADAPTER} is expected to import exactly "
            f"{sorted(FORBIDDEN_IMPORT_ROOTS)} and imports {sorted(found)}. If the "
            "legacy adapter no longer reads the formats itself, the fallback chain "
            "has a hole in it and this criterion is being satisfied vacuously."
        )


class TestGatewayParity:
    """The behavioural half: routing through the gateway must not change the text."""

    def test_a_text_document_reaches_the_sections_with_the_same_words(
        self, analyzer, tmp_path
    ):
        target = tmp_path / "paper.txt"
        target.write_text("Deep learning  improves\n\ntranslation  badly.", encoding="utf-8")

        result = analyzer.analyze(target)

        # What the old DocumentLoader.load produced, computed independently:
        # clean_extracted_text() over the raw file. If the gateway's flattening
        # changed the prose — dropped a paragraph, collapsed a boundary the
        # wrong way — the summary prompt would be answering a different
        # question than before the migration.
        expected = clean_extracted_text(target.read_text(encoding="utf-8"))

        assert analyzer.document_content == expected
        assert result["char_count"] == len(expected)
        # And the model really was asked about those words.
        assert all(expected in call["user"] for call in analyzer.client.calls)

    def test_the_analysis_reports_the_parser_and_a_local_parse(
        self, analyzer, tmp_path
    ):
        target = tmp_path / "paper.txt"
        target.write_text("A short paper about attention.", encoding="utf-8")

        result = analyzer.analyze(target)

        provenance = result["parse"]
        assert provenance["parsed_by"], "a successful parse must name the parser that ran"
        assert provenance["status"] in {"success", "partial"}
        assert provenance["used_cloud"] is False
        assert provenance["summary"]

    def test_a_docx_keeps_its_paragraphs_in_reading_order(
        self, analyzer, tmp_path
    ):
        docx = pytest.importorskip("docx")
        target = tmp_path / "paper.docx"
        document = docx.Document()
        document.add_paragraph("First paragraph")
        document.add_paragraph("Second paragraph")
        document.save(str(target))

        analyzer.analyze(target)

        # Substring, not equality: the gateway's canonical form legitimately
        # differs in shape from the legacy one (a table becomes a markdown
        # grid rather than "[Table 1]"), but the *words* the prompts read must
        # all still be there and in order.
        content = analyzer.document_content
        assert "First paragraph" in content
        assert "Second paragraph" in content
        assert content.index("First paragraph") < content.index("Second paragraph")

    def test_a_failed_parse_raises_value_error_rather_than_a_key_error(self, tmp_path):
        """The gateway returns an outcome, never an exception. The shim converts.

        ``parse_document`` answers with a ``ParseOutcome`` whose
        ``is_usable`` is False, so a caller that forgot the check would read
        ``None.to_full_text()`` and get an AttributeError — a 500 with no
        message instead of a 400 the user can act on.
        """
        target = tmp_path / "broken.pdf"
        target.write_bytes(b"%PDF-1.7\nnot really a pdf")

        with pytest.raises(ValueError, match="文档解析失败"):
            DocumentLoader.load(str(target))

    def test_an_unsupported_extension_is_rejected_before_the_gateway(self, tmp_path):
        target = tmp_path / "a.exe"
        target.write_bytes(b"binary")

        with pytest.raises(ValueError, match="不支持的文件格式"):
            DocumentLoader.load(str(target))


class TestCloudUseIsVisible:
    """T3.4 / the P0 criterion's user-facing half.

    "用户能明确知道文档被谁解析、是否经过云端" is only satisfied if the answer
    survives into the result payload. These tests assert the signal is present
    and honest in both directions: a cloud parse says so, a local parse does
    not claim it.
    """

    class _FakeCloudParser:
        """A cloud parser that never touches a network."""

        name = "fakecloud"
        version = "9"

        def __init__(self):
            self.calls = 0

        def supports(self, media_type: str) -> bool:
            return True

        def parse(self, file_path, options):
            self.calls += 1
            document = ParsedDocument(
                source_filename=os.path.basename(str(file_path)),
                media_type=media_type_for_path(str(file_path)),
                elements=[
                    Element(
                        role="paragraph",
                        text="cloud text",
                        locator=SourceLocator(kind="pdf_page", page=1),
                    )
                ],
            )
            return document, {"outcome": "success"}

    @pytest.fixture
    def unparseable_pdf(self, tmp_path):
        """A PDF the local chain cannot read, so the cloud stage is reached.

        Also an OCR-candidate media type, which is the other half of the
        condition for considering the cloud at all.
        """
        target = tmp_path / "scan.pdf"
        target.write_bytes(b"%PDF-1.7\nnot really a pdf")
        return target

    def test_a_cloud_parse_is_reported_as_a_cloud_parse(self, runtime_dirs, unparseable_pdf):
        outcome = parse_document(
            str(unparseable_pdf),
            cloud_parser=self._FakeCloudParser(),
            privacy_gate=lambda: CloudFallbackDecision(
                allowed=True, reason_code="test_allow", message="test allows egress"
            ),
        )

        assert outcome.is_usable
        assert outcome.used_cloud is True
        assert outcome.touched_network is True
        assert "已上传至云端解析服务" in outcome.summary_line()

    def test_a_local_parse_does_not_claim_the_cloud(self, runtime_dirs, tmp_path):
        target = tmp_path / "paper.txt"
        target.write_text("A perfectly readable local document.", encoding="utf-8")

        outcome = parse_document(str(target))

        assert outcome.is_usable
        assert outcome.used_cloud is False
        assert outcome.cloud_attempted is False
        assert "未上传至任何云端服务" in outcome.summary_line()

    def test_the_cloud_signal_survives_into_the_analysis_result(
        self, runtime_dirs, unparseable_pdf, monkeypatch
    ):
        """The end-to-end version: a cloud parse, seen from a result payload.

        Stubs only the gateway call itself — the flattening, the provenance
        builder and the result assembly all run for real. If ``summary_line()``
        or the provenance keys were ever dropped on the way to the payload,
        this is the test that would notice.
        """
        cloud = self._FakeCloudParser()
        real_parse = parse_document

        def parse_with_cloud(file_path, **kwargs):
            return real_parse(
                file_path,
                cloud_parser=cloud,
                privacy_gate=lambda: CloudFallbackDecision(
                    allowed=True, reason_code="test_allow", message="test allows egress"
                ),
                **kwargs,
            )

        monkeypatch.setattr(
            "paperwhisperer.documents.loader.parse_document", parse_with_cloud
        )
        monkeypatch.setattr(
            "paperwhisperer.analysis.orchestrator.LLMClient",
            lambda api_key: _StubClient(api_key),
        )

        result = DocumentAnalyzer("test-key").analyze(unparseable_pdf)

        assert cloud.calls == 1, "the cloud stage must actually have been reached"
        provenance = result["parse"]
        assert provenance["used_cloud"] is True
        assert provenance["cloud_attempted"] is True
        assert "已上传至云端解析服务" in provenance["summary"]

    def test_provenance_is_empty_when_there_was_no_parse(self):
        """No outcome means no confident-looking lies.

        ``_finalize_analysis_sections`` is called directly by callers that have
        not parsed anything, and a dict of falses there would render as
        "parsed by " in a template.
        """
        assert build_parse_provenance(None) == {}

    def test_provenance_reports_a_real_outcome(self, runtime_dirs, tmp_path):
        target = tmp_path / "paper.txt"
        target.write_text("Local only.", encoding="utf-8")

        provenance = build_parse_provenance(parse_document(str(target)))

        assert set(provenance) == {
            "summary",
            "parsed_by",
            "status",
            "used_cloud",
            "cloud_attempted",
            # A cloud parse can end four different ways -- success, quota
            # exhausted, timed out, refused -- and they all arrive as the same
            # status. These two are what let an interface say "top up your
            # quota" instead of showing one sentence for three situations the
            # reader cannot act on differently.
            "cloud_outcome",
            "error_message",
            "touched_network",
            "cache_hit",
            "fallback_used",
            "media_type",
            "warnings",
        }
        assert provenance["parsed_by"]
        assert provenance["used_cloud"] is False


class TestResultShapeIsAdditive:
    """The new ``parse`` key must not have moved anything else."""

    def test_the_existing_result_keys_keep_their_meaning(
        self, analyzer, tmp_path
    ):
        target = tmp_path / "paper.txt"
        target.write_text("body", encoding="utf-8")

        result = analyzer.analyze(target)

        # The frontend reads exactly these, and the stored session payload is
        # assembled from the same names. Renaming or reshaping one of them
        # would break a page that has no idea a migration happened.
        for key in (
            "char_count",
            "sections",
            "analysis_status",
            "suggested_questions",
            "next_actions",
            "elapsed_seconds",
        ):
            assert key in result, f"the analysis result lost {key!r}"
        assert set(result["sections"]) == set(sections.SECTION_KEYS)
        for key in sections.SECTION_KEYS:
            assert result[key] == result["sections"][key]["content"]

    def test_the_result_keeps_working_with_no_outcome(self, analyzer):
        """A hand-assembled result still finalizes, and just has no provenance."""
        collapsed = {
            name: build_section_result("success", content="content")
            for name in sections.REQUIRED_SECTION_KEYS
        }
        collapsed["mermaid"] = build_section_result("disabled")

        result = analyzer._finalize_analysis_sections("doc", collapsed)

        assert result["parse"] == {}
        assert result["char_count"] == 3


class _StubClient:
    """Stands in for ``LLMClient``; records prompts, returns a fixed answer."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = "test-model"
        self.search_rewrite_model = "test-model"
        self.request_timeout = 30
        self.max_retries = 2
        self.max_concurrency = 4
        self.calls: list[dict] = []
        self.stream_calls: list[dict] = []

    def complete(self, system_prompt, user_prompt, max_retries=None, model=None):
        self.calls.append({"system": system_prompt, "user": user_prompt, "model": model})
        return "content"

    def stream(self, system_prompt, user_prompt, max_retries=None, model=None):
        self.stream_calls.append({"system": system_prompt, "user": user_prompt})
        yield "content"


class TestTextChunkerSurvivedTheMove:
    """It is not a parser concern and must behave identically."""

    def test_short_text_is_one_chunk(self):
        assert TextChunker(100, 10).chunk_text("short") == ["short"]

    def test_long_text_is_split_and_overlaps(self):
        text = "".join(f"sentence {index}. " for index in range(400))
        chunks = TextChunker(300, 50).chunk_text(text)

        assert len(chunks) > 1
        assert all(len(chunk) <= 300 for chunk in chunks)
        assert chunks[0][-20:] in chunks[1] or chunks[1][:20] in chunks[0]

    def test_invalid_geometry_is_still_rejected(self):
        with pytest.raises(ValueError):
            TextChunker(10, 10)
