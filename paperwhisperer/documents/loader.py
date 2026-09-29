"""Document text extraction and chunking."""

from __future__ import annotations

import os

from docx import Document
from pptx import Presentation
from pypdf import PdfReader

from paperwhisperer.core import config
from paperwhisperer.core.text import clean_extracted_text


class DocumentLoader:
    """Extract plain text from the four supported document formats."""

    @staticmethod
    def load_txt(file_path: str) -> str:
        with open(file_path, encoding="utf-8", errors="ignore") as handle:
            return handle.read()

    @staticmethod
    def load_pdf(file_path: str) -> str:
        try:
            reader = PdfReader(file_path)
            text = "\n\n".join([page.extract_text() or "" for page in reader.pages])
            return text.strip()
        except Exception as exc:
            raise ValueError(f"PDF 读取失败: {exc}") from exc

    @staticmethod
    def load_docx(file_path: str) -> str:
        try:
            document = Document(file_path)
            parts = []

            for paragraph in document.paragraphs:
                text = paragraph.text.strip()
                if text:
                    parts.append(text)

            for table_index, table in enumerate(document.tables, start=1):
                rows_text = []
                for row in table.rows:
                    cells = [cell.text.strip() for cell in row.cells]
                    rows_text.append(" | ".join(cells))
                if rows_text:
                    parts.append(f"[Table {table_index}]\n" + "\n".join(rows_text))

            return "\n\n".join(parts).strip()
        except Exception as exc:
            raise ValueError(f"DOCX 读取失败: {exc}") from exc

    @staticmethod
    def load_pptx(file_path: str) -> str:
        try:
            presentation = Presentation(file_path)
            slides_text = []
            for slide_index, slide in enumerate(presentation.slides, start=1):
                shape_texts = []
                for shape in slide.shapes:
                    text = getattr(shape, "text", "")
                    if text and text.strip():
                        shape_texts.append(text.strip())

                if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
                    notes = slide.notes_slide.notes_text_frame.text.strip()
                    if notes:
                        shape_texts.append(f"[Notes] {notes}")

                if shape_texts:
                    slides_text.append(f"[Slide {slide_index}]\n" + "\n".join(shape_texts))
            return "\n\n".join(slides_text).strip()
        except Exception as exc:
            raise ValueError(f"PPTX 读取失败: {exc}") from exc

    @staticmethod
    def load(file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        loaders = {
            ".txt": DocumentLoader.load_txt,
            ".pdf": DocumentLoader.load_pdf,
            ".docx": DocumentLoader.load_docx,
            ".pptx": DocumentLoader.load_pptx,
        }
        loader = loaders.get(ext)
        if not loader:
            raise ValueError(
                f"不支持的文件格式: {ext} (支持: {config.SUPPORTED_FILE_TYPES_TEXT})"
            )
        return clean_extracted_text(loader(file_path))


class TextChunker:
    """Split long text into overlapping chunks, preferring sentence boundaries."""

    _SENTENCE_SEPARATORS = ("。", ". ", "\n")

    def __init__(self, chunk_size: int = 4000, overlap: int = 200):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if overlap < 0:
            raise ValueError("overlap must be >= 0")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]

        chunks: list[str] = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]

            if end < text_length:
                best_pos = -1
                for separator in self._SENTENCE_SEPARATORS:
                    pos = chunk.rfind(separator)
                    if pos > best_pos:
                        best_pos = pos

                if best_pos > self.chunk_size // 2:
                    chunk = chunk[:best_pos + 1]
                    end = start + best_pos + 1

            if chunk.strip():
                chunks.append(chunk.strip())

            # `end` only ever moves forward from start + 1, so the next start
            # is always >= 0; the max() keeps that invariant explicit.
            start = max(0, end - self.overlap)

        return chunks
