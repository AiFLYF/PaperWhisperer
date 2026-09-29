"""Upload persistence, signature enforcement and document loading."""

from __future__ import annotations

import asyncio
import zipfile

import pytest

from paperwhisperer.core import config
from paperwhisperer.documents.loader import DocumentLoader, TextChunker
from paperwhisperer.documents.uploads import save_upload_file
from paperwhisperer.documents.validation import validate_saved_file_signature

from .helpers import FakeUploadFile

PDF_BODY = b"%PDF-1.7\nlocal upload"
DOCX_MAGIC = b"PK\x03\x04"


class TestSaveUploadFile:
    def test_writes_the_body_and_reports_its_size(self, tmp_path):
        destination = tmp_path / "paper.pdf"

        total = asyncio.run(
            save_upload_file(FakeUploadFile(PDF_BODY), str(destination), config.MAX_CONTENT_LENGTH)
        )

        assert total == len(PDF_BODY)
        assert destination.read_bytes() == PDF_BODY

    def test_streams_large_bodies_in_chunks(self, tmp_path):
        destination = tmp_path / "big.txt"
        body = ("word " * 100_000).encode("utf-8")
        upload = FakeUploadFile(body)

        total = asyncio.run(save_upload_file(upload, str(destination), config.MAX_CONTENT_LENGTH))

        assert total == len(body)
        assert destination.stat().st_size == len(body)

    def test_rejects_an_empty_upload(self, tmp_path):
        destination = tmp_path / "empty.txt"

        with pytest.raises(ValueError, match="empty"):
            asyncio.run(save_upload_file(FakeUploadFile(b""), str(destination), config.MAX_CONTENT_LENGTH))

        assert not destination.exists()

    def test_rejects_an_oversized_upload_and_removes_the_partial_file(self, tmp_path):
        destination = tmp_path / "huge.txt"
        body = b"x" * 5000

        with pytest.raises(ValueError, match="too large"):
            asyncio.run(save_upload_file(FakeUploadFile(body), str(destination), max_bytes=100))

        assert not destination.exists()

    def test_rejects_html_disguised_as_pdf_and_cleans_up(self, tmp_path):
        destination = tmp_path / "paper.pdf"

        with pytest.raises(ValueError, match="HTML page"):
            asyncio.run(
                save_upload_file(
                    FakeUploadFile(b"<html></html>"), str(destination), config.MAX_CONTENT_LENGTH
                )
            )

        assert not destination.exists()

    @pytest.mark.parametrize(
        ("name", "body"),
        [
            ("a.txt", b"plain text"),
            ("a.pdf", PDF_BODY),
            ("a.docx", DOCX_MAGIC + b"rest"),
            ("a.pptx", DOCX_MAGIC + b"rest"),
        ],
    )
    def test_accepts_every_supported_signature(self, tmp_path, name, body):
        destination = tmp_path / name
        asyncio.run(save_upload_file(FakeUploadFile(body), str(destination), config.MAX_CONTENT_LENGTH))
        assert destination.read_bytes() == body


class TestValidateSavedFileSignature:
    def test_accepts_a_valid_pdf(self, tmp_path):
        target = tmp_path / "ok.pdf"
        target.write_bytes(PDF_BODY)
        validate_saved_file_signature(str(target))

    def test_rejects_html_written_to_a_pdf_name(self, tmp_path):
        target = tmp_path / "bad.pdf"
        target.write_bytes(b"<!DOCTYPE html><html><body>hi</body></html>")
        with pytest.raises(ValueError, match="HTML page"):
            validate_saved_file_signature(str(target))


class TestDocumentLoader:
    def test_loads_plain_text(self, tmp_path):
        target = tmp_path / "a.txt"
        target.write_text("Hello  \n\n\n\n  world ", encoding="utf-8")
        assert DocumentLoader.load(str(target)) == "Hello\n\nworld"

    def test_rejects_an_unsupported_extension(self, tmp_path):
        target = tmp_path / "a.exe"
        target.write_bytes(b"binary")
        with pytest.raises(ValueError, match="不支持的文件格式"):
            DocumentLoader.load(str(target))

    def test_wraps_a_broken_pdf_in_a_value_error(self, tmp_path):
        target = tmp_path / "broken.pdf"
        target.write_bytes(b"%PDF-1.7\nnot really a pdf")
        with pytest.raises(ValueError, match="PDF 读取失败"):
            DocumentLoader.load(str(target))

    def test_loads_a_real_docx(self, tmp_path):
        docx = pytest.importorskip("docx")
        target = tmp_path / "a.docx"
        document = docx.Document()
        document.add_paragraph("First paragraph")
        document.add_paragraph("Second paragraph")
        document.save(str(target))

        text = DocumentLoader.load(str(target))

        assert "First paragraph" in text
        assert "Second paragraph" in text

    def test_loads_a_real_pptx(self, tmp_path):
        pptx = pytest.importorskip("pptx")
        target = tmp_path / "a.pptx"
        presentation = pptx.Presentation()
        slide = presentation.slides.add_slide(presentation.slide_layouts[5])
        slide.shapes.title.text = "Slide title"
        presentation.save(str(target))

        text = DocumentLoader.load(str(target))

        assert "Slide title" in text
        assert "[Slide 1]" in text

    def test_a_docx_with_a_table_keeps_the_rows(self, tmp_path):
        docx = pytest.importorskip("docx")
        target = tmp_path / "table.docx"
        document = docx.Document()
        table = document.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "Metric"
        table.cell(0, 1).text = "Value"
        table.cell(1, 0).text = "Accuracy"
        table.cell(1, 1).text = "0.91"
        document.save(str(target))

        text = DocumentLoader.load(str(target))

        assert "[Table 1]" in text
        assert "Accuracy | 0.91" in text

    def test_a_zip_without_the_docx_extension_is_not_loaded(self, tmp_path):
        target = tmp_path / "archive.zip"
        with zipfile.ZipFile(target, "w") as archive:
            archive.writestr("a.txt", "x")

        with pytest.raises(ValueError, match="不支持的文件格式"):
            DocumentLoader.load(str(target))


class TestTextChunker:
    def test_short_text_is_one_chunk(self):
        assert TextChunker(100, 10).chunk_text("short") == ["short"]

    @pytest.mark.parametrize(
        ("chunk_size", "overlap"), [(0, 0), (10, -1), (10, 10), (10, 20)]
    )
    def test_rejects_invalid_geometry(self, chunk_size, overlap):
        with pytest.raises(ValueError):
            TextChunker(chunk_size, overlap)

    def test_splits_long_text_and_covers_all_of_it(self):
        text = "".join(f"sentence {index}. " for index in range(400))
        chunks = TextChunker(300, 50).chunk_text(text)

        assert len(chunks) > 1
        assert all(chunk.strip() for chunk in chunks)
        assert all(len(chunk) <= 300 for chunk in chunks)
        # Overlap means consecutive chunks share content.
        assert chunks[0][-20:] in chunks[1] or chunks[1][:20] in chunks[0]

    def test_never_loops_forever_on_pathological_input(self):
        # A run of characters with no sentence separator forces the fallback
        # branch where ``end`` is left at start + chunk_size.
        chunks = TextChunker(50, 10).chunk_text("x" * 500)
        assert len(chunks) < 100

    def test_respects_a_sentence_boundary_when_one_is_available(self):
        text = ("A" * 40 + ". ") * 20
        chunks = TextChunker(100, 10).chunk_text(text)
        assert len(chunks) > 1
        assert all(len(chunk) <= 100 for chunk in chunks)
