"""The legacy ``DocumentLoader`` reshaped into a canonical parser.

This adapter is the bottom of the fallback chain and the escape hatch for the
whole of Phase 0: when ``PARSER_BACKEND=legacy`` the router collapses to this
one parser, so the application keeps working with no new dependency installed.
It earns that position by not trying to be better than what it replaces. It
reads the same bytes with the same libraries and produces the same prose; all
it adds is structure — locators, roles, and real tables — so downstream code
can be written once against the canonical schema instead of once per backend.

What it deliberately does **not** do:

* **Re-parse anything.** Text comes from ``DocumentLoader``'s own methods.
  A second extraction path would be a second set of bugs, and the loader is
  locked down by its own tests.
* **Invent geometry.** pypdf's text layer has no coordinates, so PDF elements
  carry ``locator.page`` and ``bbox=None``. A fabricated rectangle would be
  worse than none: the evidence viewer would highlight the wrong region with
  total confidence, which is the failure mode the schema's ``unknown`` kind
  exists to avoid.
* **Flatten tables.** The legacy loader joins cells with ``" | "``, which
  destroys exactly the column structure that cross-column retrieval needs.
  Tables become a :class:`~paperwhisperer.ingestion.schema.Table` grid.

Two structural details are inherited from the legacy text and upgraded rather
than re-derived: the ``[Slide N]`` prefix the loader writes into prose becomes
a real ``heading`` element, and DOCX headings become a maintained
``heading_path`` so a paragraph can say which section it came from.
"""

from __future__ import annotations

import os
import re

from docx import Document
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph as DocxParagraph
from pptx import Presentation
from pypdf import PdfReader

from paperwhisperer.core import config
from paperwhisperer.core.text import clean_extracted_text, now_iso
from paperwhisperer.documents.loader import DocumentLoader
from paperwhisperer.ingestion.gateway import (
    ParseOptions,
    ParseResult,
    compute_file_sha256,
    media_type_for_path,
)
from paperwhisperer.ingestion.schema import (
    Element,
    ParsedDocument,
    SourceLocator,
    Table,
)

#: Characters per plain-text element. Plain text has no paragraph structure
#: worth trusting — blank-line runs are an artefact of whatever wrote the file
#: — so the span is a fixed budget chosen to sit under a typical embedding
#: window. The locator is the character range, which means a citation can
#: still be pointed at an exact offset.
TXT_ELEMENT_CHARS = 1200

#: DOCX style names that mean "this is a heading", mapped to level. Matched
#: case-insensitively because Word localises style names ("标题 1"), and the
#: English fallback covers documents created by python-docx.
_HEADING_STYLE_RE = re.compile(r"^(?:heading|标题)\s*(\d+)$", re.IGNORECASE)

#: A shape whose placeholder type marks it as the slide's own title. Used to
#: separate a real slide title from the first line of body text.
_TITLE_PLACEHOLDER_TYPES = (13, 1)  # CENTER_TITLE, TITLE


def _docx_heading_level(paragraph: DocxParagraph) -> int | None:
    """Heading level for a DOCX paragraph, or ``None`` if it is body text."""
    try:
        style_name = str(paragraph.style.name or "")
    except (AttributeError, KeyError):
        # A style reference can dangle when a document was produced by a tool
        # that wrote style ids it never defined. That is a body paragraph, not
        # a crash.
        return None
    match = _HEADING_STYLE_RE.match(style_name.strip())
    return int(match.group(1)) if match else None


def _iter_docx_body(document) -> list:
    """Walk paragraphs and tables in true body order.

    ``document.paragraphs`` and ``document.tables`` are two separate lists, and
    concatenating them would put every table after every paragraph. Reading
    order is the one thing a reader notices immediately when it is wrong, so
    the body element is walked directly instead.
    """
    body = document.element.body
    blocks = []
    for child in body.iterchildren():
        tag = str(getattr(child, "tag", "") or "")
        if tag.endswith("}p"):
            blocks.append(DocxParagraph(child, document))
        elif tag.endswith("}tbl"):
            blocks.append(DocxTable(child, document))
    return blocks


def _docx_table_to_grid(table: DocxTable) -> Table:
    """Convert a DOCX table to a rectangular grid of stripped cell text.

    Ragged rows are padded to the widest row rather than left ragged: a grid
    whose rows differ in length cannot be rendered as a table at all, and
    ``Table.column_count`` already defines the width as the maximum.
    """
    raw_rows = [
        [cell.text.strip() for cell in row.cells]
        for row in table.rows
    ]
    if not raw_rows:
        return Table()
    width = max(len(row) for row in raw_rows)
    padded = [row + [""] * (width - len(row)) for row in raw_rows]
    # The first row is treated as a header because that is how a table in a
    # paper is written far more often than not; ``header_rows=0`` would render
    # the header as data and duplicate it in the markdown projection.
    return Table(rows=padded, header_rows=1)


class LegacyDocumentParser:
    """Adapter over :class:`DocumentLoader`, emitting canonical documents."""

    name = "legacy"
    version = "1.0"

    def supports(self, media_type: str) -> bool:
        """True for the four formats the legacy loader can read.

        The media type is compared against the same table the router derives
        extensions from, so an upload and a path agree on what a ``.docx`` is.
        """
        return media_type in _SUPPORTED_MEDIA_TYPES

    def parse(self, file_path: str, options: ParseOptions) -> ParseResult:
        """Parse ``file_path`` into a :class:`ParsedDocument`.

        Raises ``ValueError`` for an unsupported extension, matching the legacy
        loader exactly — including the message — so the two paths fail
        identically and callers that matched on that text keep working.
        """
        extension = os.path.splitext(file_path)[1].lower()
        if extension not in _EXTENSION_BY_FORMAT:
            raise ValueError(
                f"不支持的文件格式: {extension} (支持: {config.SUPPORTED_FILE_TYPES_TEXT})"
            )

        builder = _EXTENSION_BY_FORMAT[extension]
        document = builder(self, file_path)
        document.parser_name = self.name
        document.parser_version = self.version
        document.options_hash = options.compute_options_hash()
        document.build_units()
        self._attach_unit_ids(document)
        return ParseResult.from_document(
            document,
            status="partial" if document.warnings else "success",
            warnings=list(document.warnings),
            raw_artifact_uri="",
        )

    # -- internals ---------------------------------------------------------

    def _attach_unit_ids(self, document: ParsedDocument) -> None:
        """Point every element at the unit that contains it.

        ``build_units`` derives units from locators and does not write back,
        so without this step ``Element.unit_id`` would be empty and the
        evidence API would have to re-derive the grouping on every request.
        """
        unit_id_by_locator = {
            (unit.locator.kind, unit.locator.page, unit.locator.slide,
             unit.locator.paragraph, unit.locator.offset_start): unit.unit_id
            for unit in document.units
        }
        for element in document.elements:
            locator = element.locator
            key = (
                locator.kind,
                locator.page,
                locator.slide,
                locator.paragraph,
                locator.offset_start,
            )
            element.unit_id = unit_id_by_locator.get(key, "")

    def _new_document(self, file_path: str, media_type: str) -> ParsedDocument:
        return ParsedDocument(
            source_filename=os.path.basename(file_path),
            media_type=media_type,
            sha256=compute_file_sha256(file_path),
            created_at=now_iso(),
        )

    def _parse_txt(self, file_path: str) -> ParsedDocument:
        """Slice plain text into offset-addressable paragraph elements.

        The text is cleaned first so the offsets refer to the text the rest of
        the pipeline sees, not to the raw file. Without that, every offset
        would be wrong by however much whitespace cleaning removed — and an
        offset that points somewhere else is worse than no offset.
        """
        media_type = media_type_for_path(file_path)
        text = clean_extracted_text(DocumentLoader.load_txt(file_path))
        document = self._new_document(file_path, media_type)

        for start, end in _txt_spans(text):
            chunk = text[start:end].strip()
            if not chunk:
                continue
            document.elements.append(
                Element(
                    role="paragraph",
                    text=chunk,
                    locator=SourceLocator(kind="txt_offset", offset_start=start,
                                          offset_end=end),
                )
            )
        if not document.elements:
            document.warnings.append("TXT 文档没有解析出任何正文内容。")
        return document

    def _parse_pdf(self, file_path: str) -> ParsedDocument:
        """Emit one element per non-empty PDF page.

        Known degradation: pypdf's text layer carries no geometry, so elements
        get ``pdf_page`` locators with ``bbox=None``. A page resolves and a
        reader lands on the right page; they just do not get a highlight. The
        alternative — estimating boxes from text offsets — would produce
        plausible-looking wrong coordinates, and the docling adapter exists to
        supply real ones.
        """
        media_type = media_type_for_path(file_path)
        document = self._new_document(file_path, media_type)
        try:
            reader = PdfReader(file_path)
            pages = list(reader.pages)
        except Exception as exc:
            raise ValueError(f"PDF 读取失败: {exc}") from exc

        document.page_count = len(pages)
        for page_index, page in enumerate(pages, start=1):
            try:
                raw = page.extract_text() or ""
            except Exception:  # noqa: BLE001 - one bad page must not lose the file
                document.warnings.append(f"PDF 第 {page_index} 页文本提取失败，已跳过。")
                continue
            body = clean_extracted_text(raw)
            if not body:
                document.warnings.append(f"PDF 第 {page_index} 页没有可提取的文本。")
                continue
            document.elements.append(
                Element(
                    role="paragraph",
                    text=body,
                    locator=SourceLocator(kind="pdf_page", page=page_index),
                    page_number_label=f"p. {page_index}",
                )
            )
        if document.elements and not document.page_count:
            document.page_count = max(
                element.locator.page or 0 for element in document.elements
            )
        if any(element.locator.bbox is None for element in document.elements):
            document.warnings.append(
                "legacy PDF 解析只能定位到页，无法提供页内坐标（bbox）。"
            )
        if not document.elements:
            document.warnings.append(
                "PDF 没有解析出任何文本，可能是扫描件，需要启用 OCR。"
            )
        return document

    def _parse_docx(self, file_path: str) -> ParsedDocument:
        """Emit paragraphs, headings and real tables in body order."""
        media_type = media_type_for_path(file_path)
        try:
            source = Document(file_path)
        except Exception as exc:
            raise ValueError(f"DOCX 读取失败: {exc}") from exc

        document = self._new_document(file_path, media_type)
        heading_stack: list[tuple[int, str]] = []
        paragraph_index = 0

        for block in _iter_docx_body(source):
            if isinstance(block, DocxTable):
                grid = _docx_table_to_grid(block)
                if not grid.rows:
                    continue
                caption = f"Table {len([e for e in document.elements if e.role == 'table']) + 1}"
                document.elements.append(
                    Element(
                        role="table",
                        text=grid.to_markdown(),
                        table=grid,
                        locator=SourceLocator(
                            kind="docx_element",
                            paragraph=paragraph_index,
                            heading_path=tuple(title for _, title in heading_stack),
                        ),
                        heading_path=tuple(title for _, title in heading_stack),
                        page_number_label=caption,
                    )
                )
                paragraph_index += 1
                continue

            body = block.text.strip()
            if not body:
                paragraph_index += 1
                continue

            level = _docx_heading_level(block)
            if level is not None:
                # A heading replaces every same-or-higher level already on the
                # stack, which is what makes the path a path rather than a log.
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, body))
                document.elements.append(
                    Element(
                        role="heading",
                        text=body,
                        level=level,
                        locator=SourceLocator(
                            kind="docx_element",
                            paragraph=paragraph_index,
                            heading_path=tuple(title for _, title in heading_stack),
                        ),
                        heading_path=tuple(title for _, title in heading_stack),
                    )
                )
            else:
                document.elements.append(
                    Element(
                        role="paragraph",
                        text=body,
                        locator=SourceLocator(
                            kind="docx_element",
                            paragraph=paragraph_index,
                            heading_path=tuple(title for _, title in heading_stack),
                        ),
                        heading_path=tuple(title for _, title in heading_stack),
                    )
                )
            paragraph_index += 1

        if not document.elements:
            document.warnings.append("DOCX 文档没有解析出任何正文内容。")
        return document

    def _parse_pptx(self, file_path: str) -> ParsedDocument:
        """Emit per-slide elements, promoting the slide title to a heading.

        The legacy loader writes ``[Slide N]`` into the body text, which makes
        the slide boundary indistinguishable from a sentence the author wrote.
        Here the title placeholder becomes a ``heading`` element and the slide
        number lives in the locator, so prose and navigation stop competing.
        """
        media_type = media_type_for_path(file_path)
        try:
            presentation = Presentation(file_path)
        except Exception as exc:
            raise ValueError(f"PPTX 读取失败: {exc}") from exc

        document = self._new_document(file_path, media_type)
        slides = list(presentation.slides)
        document.page_count = len(slides)

        for slide_index, slide in enumerate(slides, start=1):
            title_shape = _slide_title_shape(slide)
            for element_index, shape in enumerate(slide.shapes):
                # Identity comparison, not object comparison: python-pptx
                # builds a fresh shape proxy on every traversal, so the object
                # ``slide.shapes.title`` returned and the one ``slide.shapes``
                # yields are never the same instance even when they describe
                # the same shape. ``is`` was always False here, which let the
                # title fall through to the paragraph branch below and left
                # every slide with an empty heading list.
                if title_shape is not None and shape.shape_id == title_shape.shape_id:
                    title = str(getattr(shape, "text", "") or "").strip()
                    if title:
                        document.elements.append(
                            Element(
                                role="heading",
                                text=title,
                                level=1,
                                locator=SourceLocator(
                                    kind="slide",
                                    slide=slide_index,
                                    element=element_index,
                                ),
                                page_number_label=f"slide {slide_index}",
                            )
                        )
                    continue

                text = str(getattr(shape, "text", "") or "").strip()
                if not text:
                    continue
                document.elements.append(
                    Element(
                        role="paragraph",
                        text=clean_extracted_text(text),
                        locator=SourceLocator(
                            kind="slide",
                            slide=slide_index,
                            element=element_index,
                        ),
                        page_number_label=f"slide {slide_index}",
                    )
                )

            notes = _slide_notes(slide)
            if notes:
                document.elements.append(
                    Element(
                        role="paragraph",
                        text=notes,
                        locator=SourceLocator(kind="slide", slide=slide_index),
                        page_number_label=f"slide {slide_index}",
                    )
                )

        if not document.elements:
            document.warnings.append("PPTX 演示文稿没有解析出任何文本内容。")
        return document


def _txt_spans(text: str, size: int = TXT_ELEMENT_CHARS) -> list[tuple[int, int]]:
    """Split ``text`` into ``(start, end)`` spans of at most ``size`` chars.

    Cuts prefer a paragraph or line break near the end of the window so an
    element does not usually end mid-sentence, but the span boundaries stay
    exact: the offsets are the citation handle, and adjusting them to make the
    text prettier would break that.
    """
    if not text:
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + size, length)
        if end < length:
            window = text[start:end]
            for separator in ("\n\n", "\n", "。", ". "):
                cut = window.rfind(separator)
                if cut > size // 2:
                    end = start + cut + len(separator)
                    break
        spans.append((start, end))
        start = end
    return spans


def _slide_title_shape(slide):
    """The shape acting as the slide's title, or ``None``.

    ``slide.shapes.title`` is the cheap path and is tried first; the
    placeholder-type scan is the fallback for templates whose title shape
    python-pptx does not recognise as a title placeholder.
    """
    title = getattr(slide.shapes, "title", None)
    if title is not None:
        return title
    for shape in slide.shapes:
        if getattr(shape, "is_placeholder", False) and (
            getattr(shape.placeholder_format, "type", None) in _TITLE_PLACEHOLDER_TYPES
        ):
            return shape
    return None


def _slide_notes(slide) -> str:
    """Speaker notes for a slide, or an empty string.

    The legacy loader prefixes these with ``[Notes]`` inside the body text.
    Here they are a separate element: notes are not slide content, and mixing
    them into the prose means a quote can be attributed to a slide the author
    never put it on.
    """
    if not getattr(slide, "has_notes_slide", False):
        return ""
    notes_frame = getattr(slide.notes_slide, "notes_text_frame", None)
    if notes_frame is None:
        return ""
    return clean_extracted_text(str(getattr(notes_frame, "text", "") or "")).strip()


#: Media types the legacy loader can read, resolved through the gateway's own
#: extension table so the adapter and the router cannot disagree.
_SUPPORTED_MEDIA_TYPES = frozenset(
    media_type_for_path(f"sample{extension}")
    for extension in (".txt", ".pdf", ".docx", ".pptx")
)

#: Extension to the builder that handles it. A dict rather than a chain of
#: ``if`` statements so adding a format is one line, and so the support check
#: above and the dispatch below are driven by the same keys.
_EXTENSION_BY_FORMAT = {
    ".txt": LegacyDocumentParser._parse_txt,
    ".pdf": LegacyDocumentParser._parse_pdf,
    ".docx": LegacyDocumentParser._parse_docx,
    ".pptx": LegacyDocumentParser._parse_pptx,
}
