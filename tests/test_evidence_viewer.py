"""The T3.3 evidence viewer: what it draws, and what it refuses to draw.

Two halves, and they are deliberately different kinds of test.

**The API half is behavioural.** A ``TestClient`` over the registered
``routes_evidence`` router — the same mount ``create_app`` performs — is asked
for one evidence per format, and the response is read as a reader's browser
would read it. What is asserted is the *pairing* the viewer depends on: a
highlight exists in ``view.target`` if and only if the resolver produced a
``xywh=`` fragment, and the four numbers in one are the four numbers in the
other. A viewer that drew from a second source would pass every test that only
looked at "is there a highlight", and fail the only one that matters.

**The JS half is structural, and it is the half the release gate actually
lives in.** There is no browser in this repository and adding one is out of
scope, so what can be asserted about ``evidenceViewer.js`` is not "the highlight
looks right" but "there is no code path in this module that could produce a
wrong one". The properties are reachability, mirroring the pattern
``tests/test_frontend_evidence.py`` established for the quotation-mark gate:

* the target rectangle is read from ``view.target`` and from nowhere else, and
  ``xywh=`` never appears in the client at all — the client cannot reconstruct a
  box from the fragment, so a fragment and a view can never disagree because the
  client never looks at the fragment;
* ``isDrawableBox`` refuses non-finite coordinates and non-positive extents, so
  a future server build that starts sending a ``NaN`` produces *no highlight*
  rather than a ``NaN%``-positioned div that renders as nothing;
* the y-axis flip happens in exactly one function, shared by the blocks and the
  highlight, so they cannot be flipped in opposite directions;
* the absence of a highlight is a named, visible state with a reason code, never
  an empty container — an empty page map reads as a load failure.

What none of this can prove is that the drawing looks right. That remains a
manual check, and the last test in the file says so rather than pretending
otherwise.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from paperwhisperer.api import routes_evidence
from paperwhisperer.infra.stores import build_sqlite_bundle
from paperwhisperer.infra.stores.interfaces import DocumentRecord
from paperwhisperer.ingestion.schema import (
    BoundingBox,
    Chunk,
    Element,
    ParsedDocument,
    SourceLocator,
    Table,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
JS_DIR = PROJECT_ROOT / "templates" / "static" / "js"
CONFIG_JS = PROJECT_ROOT / "templates" / "static" / "js" / "config.js"
RESOLVER = (
    PROJECT_ROOT / "paperwhisperer" / "rag" / "evidence.py"
).read_text(encoding="utf-8")
INDEX_HTML = (PROJECT_ROOT / "templates" / "index.html").read_text(encoding="utf-8")


def _read(name: str) -> str:
    return (JS_DIR / name).read_text(encoding="utf-8")


VIEWER = _read("evidenceViewer.js")
EVIDENCE = _read("evidence.js")
CONFIG = CONFIG_JS.read_text(encoding="utf-8")
MAIN = _read("main.js")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def strip_js_comments(source: str) -> str:
    """Remove ``//`` and ``/* */`` comments, leaving string literals intact.

    Needed so a comment *discussing* a guarantee cannot satisfy a structural
    assertion about it. The scanner tracks string state, so a ``//`` inside a
    string is not mistaken for a comment.
    """
    out: list[str] = []
    index = 0
    length = len(source)
    while index < length:
        char = source[index]
        nxt = source[index + 1] if index + 1 < length else ""
        if char == "/" and nxt == "/":
            while index < length and source[index] != "\n":
                index += 1
        elif char == "/" and nxt == "*":
            index += 2
            while index < length and not (
                source[index] == "*" and source[index + 1 : index + 2] == "/"
            ):
                index += 1
            index += 2
        elif char in "\"'`":
            quote = char
            out.append(char)
            index += 1
            while index < length:
                out.append(source[index])
                if source[index] == "\\":
                    index += 1
                    if index < length:
                        out.append(source[index])
                        index += 1
                    continue
                if source[index] == quote:
                    index += 1
                    break
                index += 1
        else:
            out.append(char)
            index += 1
    return "".join(out)


def function_body(source: str, name: str) -> str:
    """Return the body of ``function name(...)`` by brace matching."""
    match = re.search(rf"function\s+{re.escape(name)}\s*\(", source)
    assert match, f"function {name}() is missing"

    index = source.index("{", match.end())
    depth = 0
    start = index
    while index < len(source):
        char = source[index]
        if char in "\"'`":
            quote = char
            index += 1
            while index < len(source):
                if source[index] == "\\":
                    index += 2
                    continue
                if source[index] == quote:
                    break
                index += 1
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
        index += 1
    raise AssertionError(f"unbalanced braces in {name}()")


def parse_xywh(fragment: str) -> tuple[float, float, float, float]:
    """Read the four numbers out of a ``xywh=`` page fragment."""
    match = re.match(r"^xywh=([-\d.]+),([-\d.]+),([-\d.]+),([-\d.]+)$", fragment or "")
    assert match, f"{fragment!r} is not a pdf.js xywh fragment"
    return tuple(float(group) for group in match.groups())  # type: ignore[return-value]


# --------------------------------------------------------------------------
# Fixtures — a real store, a client over the real router
# --------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    settings = SimpleNamespace(
        STORAGE_SQLITE_PATH=str(tmp_path / "viewer.sqlite3"),
        STORAGE_SQLITE_TIMEOUT_SECONDS=30,
        STORAGE_LEXICAL_FTS5="auto",
    )
    bundle = build_sqlite_bundle(settings)
    yield bundle
    bundle.close()


@pytest.fixture
def client(store):
    application = FastAPI()
    application.include_router(routes_evidence.router)
    application.dependency_overrides[routes_evidence.get_store_bundle] = lambda: store
    with TestClient(application) as test_client:
        yield test_client


def _page_element(element_id: str, page: int, box, text: str, role: str = "text") -> Element:
    return Element(
        element_id=element_id,
        role=role,
        text=text,
        locator=SourceLocator(kind="pdf_page", page=page, bbox=box),
    )


def _seed_pdf(store, document_id: str = "doc_pdf") -> ParsedDocument:
    """A page with two placeable blocks, one mirrored block, and a table."""
    elements = [
        _page_element(
            f"{document_id}_top",
            7,
            BoundingBox(61.5, 700.0, 361.5, 740.0),
            "Introduction: the task we address.",
        ),
        _page_element(
            f"{document_id}_bottom",
            7,
            BoundingBox(61.5, 120.0, 361.5, 180.0),
            "Conclusion: it works.",
        ),
        # A TOPLEFT-converted box: y0 < 0 is what BoundingBox flags as
        # mirrored, and a real adapter produces exactly this shape.
        _page_element(
            f"{document_id}_mirrored",
            7,
            BoundingBox(61.5, -180.0, 361.5, -120.0),
            "Footer line the converter mirrored.",
        ),
        Element(
            element_id=f"{document_id}_table",
            role="table",
            table=Table(
                rows=[["Model", "BLEU"], ["Ours", "31.4"]],
                header_rows=1,
                caption="Main results",
            ),
            locator=SourceLocator(
                kind="pdf_page", page=7, bbox=BoundingBox(61.5, 300.0, 361.5, 400.0)
            ),
        ),
    ]
    document = ParsedDocument(
        document_id=document_id,
        source_filename="paper.pdf",
        media_type="application/pdf",
        page_count=12,
        elements=elements,
    )
    store.metadata.upsert_document(
        DocumentRecord(
            document_id=document_id,
            document=document,
            source_filename="paper.pdf",
            media_type="application/pdf",
            page_count=12,
            index_version="idx-7",
        )
    )
    return document


def _seed_pptx(store, document_id: str = "doc_pptx") -> ParsedDocument:
    elements = [
        Element(
            element_id=f"{document_id}_title",
            role="heading",
            text="Results",
            locator=SourceLocator(kind="slide", slide=3, element=0),
        ),
        Element(
            element_id=f"{document_id}_body",
            text="BLEU 31.4 on the dev set",
            locator=SourceLocator(kind="slide", slide=3, element=1),
        ),
        Element(
            element_id=f"{document_id}_notes",
            text="A note that is on a different slide",
            locator=SourceLocator(kind="slide", slide=4, element=0),
        ),
    ]
    document = ParsedDocument(
        document_id=document_id, source_filename="deck.pptx", media_type="application/vnd.ms-powerpoint"
    )
    document.elements = elements
    store.metadata.upsert_document(
        DocumentRecord(
            document_id=document_id,
            document=document,
            source_filename="deck.pptx",
            media_type="application/vnd.ms-powerpoint",
        )
    )
    return document


def _seed_docx(store, document_id: str = "doc_docx") -> ParsedDocument:
    elements = [
        Element(
            element_id=f"{document_id}_p12",
            text="We fine-tune on the training split.",
            locator=SourceLocator(kind="docx_element", paragraph=12),
        ),
        Element(
            element_id=f"{document_id}_p13",
            text="Section 2.1 Related work",
            role="heading",
            heading_path=("Introduction", "Related work"),
            locator=SourceLocator(
                kind="docx_element", paragraph=13, heading_path=("Introduction", "Related work")
            ),
        ),
        Element(
            element_id=f"{document_id}_p14",
            text="Prior work does not model length.",
            locator=SourceLocator(kind="docx_element", paragraph=14),
        ),
        Element(
            element_id=f"{document_id}_p99",
            text="A paragraph from a different document section.",
            locator=SourceLocator(kind="docx_element", paragraph=99),
        ),
    ]
    document = ParsedDocument(
        document_id=document_id, source_filename="paper.docx",
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    document.elements = elements
    store.metadata.upsert_document(
        DocumentRecord(
            document_id=document_id,
            document=document,
            source_filename="paper.docx",
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    )
    return document


def _seed_chunk(
    store,
    document_id: str,
    chunk_id: str,
    locator: SourceLocator,
    *,
    element_ids: tuple[str, ...] = (),
    text: str = "The cited passage, verbatim.",
    index_version: str = "idx-7",
) -> Chunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        text=text,
        element_ids=element_ids,
        locator=locator,
        index_version=index_version,
    )
    store.metadata.put_chunk(document_id, chunk)
    return chunk


def _resolve(client, document_id: str, evidence_id: str) -> dict:
    response = client.get(f"/v1/documents/{document_id}/evidence/{evidence_id}")
    assert response.status_code == 200, response.text
    return response.json()


# --------------------------------------------------------------------------
# PDF — the page, and a highlight that is either right or absent
# --------------------------------------------------------------------------


def test_pdf_lays_out_the_page_from_the_parser_own_coordinates(client, store):
    """The page map is the parse's geometry, in one frame, and says so."""
    _seed_pdf(store)
    # A box between the two prose blocks rather than on either of them, so the
    # background map is intact and the highlight has to prove it is separate
    # from the blocks rather than coinciding with one.
    bbox = BoundingBox(61.5, 640.0, 361.5, 660.0)
    _seed_chunk(
        store,
        "doc_pdf",
        "chunk_pdf",
        SourceLocator(kind="pdf_page", page=7, bbox=bbox),
        element_ids=(),
    )

    view = _resolve(client, "doc_pdf", "chunk_pdf")["view"]

    assert view["kind"] == "page_map"
    assert view["page"] == 7
    assert view["coordinate_space"] == "pdf_user_space_points"
    # The claim this drawing makes is that it is not a picture of the page, and
    # it is stated in the payload so a client can render the statement rather
    # than the assumption.
    assert view["is_raster"] is False
    assert "不是原始 PDF 的渲染图" in view["note"]

    placed = {block["element_id"] for block in view["blocks"]}
    assert {"doc_pdf_top", "doc_pdf_bottom", "doc_pdf_table"} <= placed
    assert view["omitted"]["relative_bbox"] == 1, (
        "the mirrored block must be omitted, not placed at a mirrored position"
    )
    assert view["frame"]["width"] > 0 and view["frame"]["height"] > 0


def test_the_highlight_is_the_same_rectangle_as_the_pdfjs_fragment(client, store):
    """One rectangle, two representations. They must be the same numbers.

    This is the load-bearing assertion of the whole viewer. The client draws
    from ``view.target``; a reader debugging a highlight compares it against the
    ``xywh=`` fragment. If the two disagreed by a rounding step the client would
    be drawing a box the resolver never promised, and nothing downstream would
    know.
    """
    _seed_pdf(store)
    bbox = BoundingBox(61.5, 142.25, 361.5, 190.75)
    _seed_chunk(store, "doc_pdf", "chunk_pdf", SourceLocator(kind="pdf_page", page=7, bbox=bbox))

    payload = _resolve(client, "doc_pdf", "chunk_pdf")
    fragment = parse_xywh(payload["visual_targeting"]["highlight"])
    target = payload["view"]["target"]

    assert (target["x"], target["y"], target["width"], target["height"]) == fragment
    # And both are the stored box, not a re-derivation of it.
    assert (target["x"], target["y"]) == (bbox.x0, bbox.y0)
    assert (target["width"], target["height"]) == (bbox.width, bbox.height)


def test_a_highlight_exists_in_the_view_exactly_when_the_resolver_produced_one(client, store):
    """The invariant the client is allowed to rely on, across every case.

    Parameterised over the whole family rather than spot-checked, because the
    failure this guards is not a wrong number — it is a *highlight that appears
    where the API said it could not*, or a missing one where the API promised
    it. Both look identical to a reader.
    """
    _seed_pdf(store)
    cases = {
        "absolute": BoundingBox(61.5, 142.25, 361.5, 190.75),
        "mirrored": BoundingBox(61.5, -142.25, 361.5, -90.75),
        "reversed": BoundingBox(361.5, 142.25, 61.5, 190.75),
    }
    for name, box in cases.items():
        _seed_chunk(
            store, "doc_pdf", f"chunk_{name}", SourceLocator(kind="pdf_page", page=7, bbox=box)
        )
    # A locator with no box at all: the page is still known, the box is not.
    _seed_chunk(store, "doc_pdf", "chunk_nobox", SourceLocator(kind="pdf_page", page=7))

    for name in (*cases, "nobox"):
        payload = _resolve(client, "doc_pdf", f"chunk_{name}")
        has_highlight = payload["visual_targeting"]["highlight"] is not None
        has_target = payload["view"]["target"] is not None
        assert has_highlight is has_target, (
            f"{name}: resolver produced a highlight={has_highlight} but the view "
            f"produced a target={has_target}"
        )


def test_a_mirrored_box_yields_the_page_and_no_highlight(client, store):
    """The exact release-gate case: right page, no box, and it says which.

    The view still renders the page. That is the point — requirement 2 asks for
    "the passage without a highlight", and a blank panel would be the failure
    instead. The reason code travels with it so the panel can say *why* the box
    is missing rather than showing an empty rectangle.
    """
    _seed_pdf(store)
    _seed_chunk(
        store,
        "doc_pdf",
        "chunk_mirror",
        SourceLocator(kind="pdf_page", page=7, bbox=BoundingBox(61.5, -142.25, 361.5, -90.75)),
    )

    payload = _resolve(client, "doc_pdf", "chunk_mirror")
    targeting, view = payload["visual_targeting"], payload["view"]

    assert targeting["available"] is False
    assert targeting["reason_code"] == "relative_bbox"
    assert view["target"] is None
    assert view["target_rejected"]["reason_code"] == "relative_bbox"
    assert view["target_rejected"]["message"], "a missing highlight must carry a reason in words"
    # The page survives the missing box.
    assert view["page"] == 7
    assert view["blocks"], "the page layout must still render without a highlight"
    assert view["frame"] is not None


def test_a_reversed_box_is_refused_rather_than_drawn_mirrored(client, store):
    """A corner order that is not PDF's would draw a rectangle anchored wrong.

    ``BoundingBox.width`` takes an absolute value, so a swapped pair still has a
    positive extent and passes every check that looks at size. Nothing marks it
    relative, so without this guard it would be drawn mirrored in one axis —
    a plausible-looking highlight over the wrong lines, which is the defect the
    whole design is about.
    """
    _seed_pdf(store)
    _seed_chunk(
        store,
        "doc_pdf",
        "chunk_reversed",
        SourceLocator(kind="pdf_page", page=7, bbox=BoundingBox(361.5, 142.25, 61.5, 190.75)),
    )

    payload = _resolve(client, "doc_pdf", "chunk_reversed")

    assert payload["visual_targeting"]["highlight"] is None
    assert payload["visual_targeting"]["reason_code"] == "malformed_bbox"
    assert payload["view"]["target"] is None
    assert payload["view"]["target_rejected"]["reason_code"] == "malformed_bbox"


def test_a_page_without_a_box_still_names_the_page(client, store):
    """A missing box is an absence; a relative box is a poisoned value.

    Both report ``available: False`` and neither gets a highlight, but the first
    is an honest gap in what the parser recorded and the second is a value that
    would land in the wrong place. Collapsing them would teach a reader the
    wrong thing about their own document.
    """
    _seed_pdf(store)
    _seed_chunk(store, "doc_pdf", "chunk_nobox", SourceLocator(kind="pdf_page", page=7))

    payload = _resolve(client, "doc_pdf", "chunk_nobox")

    assert payload["visual_targeting"]["reason_code"] == "missing_bbox"
    assert payload["view"]["target"] is None
    assert payload["view"]["page"] == 7
    assert payload["view"]["blocks"]


def test_the_cited_element_is_not_drawn_twice(client, store):
    """The highlight is the cited rectangle; the blocks are everything else.

    If the cited element's own block stayed in ``blocks`` it would be drawn
    under the highlight, and the two would be indistinguishable — a reader
    could not tell which rectangle is the answer to "where does this point".
    """
    _seed_pdf(store)
    _seed_chunk(
        store,
        "doc_pdf",
        "chunk_pdf",
        SourceLocator(kind="pdf_page", page=7, bbox=BoundingBox(61.5, 700.0, 361.5, 740.0)),
        element_ids=("doc_pdf_top",),
    )

    view = _resolve(client, "doc_pdf", "chunk_pdf")["view"]

    assert view["target"]["element_id"] == "doc_pdf_top"
    rectangles = [
        (block["x"], block["y"], block["width"], block["height"]) for block in view["blocks"]
    ]
    target = (view["target"]["x"], view["target"]["y"], view["target"]["width"], view["target"]["height"])
    assert target not in rectangles, "the cited rectangle is drawn twice"


def test_a_page_with_too_many_blocks_says_it_dropped_them(client, store):
    """Silent truncation is a release gate at zero, so it must be loud.

    A layout that quietly showed 40 of 400 blocks teaches the reader that the
    page has 40 blocks. The count that was dropped is in the payload for the
    same reason the view's ``omitted`` counts are: the reader has to be able to
    tell "this page is sparse" from "this panel is small".
    """
    from paperwhisperer.rag import evidence as resolver

    _seed_pdf(store)
    document = store.metadata.get_document("doc_pdf").document
    original = document.elements
    document.elements = list(original) + [
        _page_element(
            f"doc_pdf_extra_{index}",
            7,
            BoundingBox(400.0 + index, 500.0, 420.0 + index, 520.0),
            f"Filler element {index}.",
        )
        for index in range(resolver.VIEW_MAX_BLOCKS + 20)
    ]
    store.metadata.upsert_document(
        DocumentRecord(
            document_id="doc_pdf",
            document=document,
            source_filename="paper.pdf",
            media_type="application/pdf",
            index_version="idx-7",
        )
    )
    _seed_chunk(
        store,
        "doc_pdf",
        "chunk_pdf",
        SourceLocator(kind="pdf_page", page=7, bbox=BoundingBox(61.5, 700.0, 361.5, 740.0)),
    )

    view = _resolve(client, "doc_pdf", "chunk_pdf")["view"]

    assert view["blocks_truncated"] is True
    assert view["blocks_shown"] == resolver.VIEW_MAX_BLOCKS
    assert view["blocks_total"] > resolver.VIEW_MAX_BLOCKS
    assert view["blocks_truncated"] == (view["blocks_total"] > view["blocks_shown"])


# --------------------------------------------------------------------------
# PPTX — the slide
# --------------------------------------------------------------------------


def test_pptx_jumps_to_the_slide_and_marks_the_cited_element(client, store):
    """A slide is a slide number, so the viewer gives the slide's own text.

    There are no pixel coordinates for a slide in this schema, and inventing
    some would be worse than having none — so "jump to slide 3" means: here is
    slide 3's text, and this is the element in it the citation came from.
    """
    _seed_pptx(store)
    _seed_chunk(
        store,
        "doc_pptx",
        "chunk_slide",
        SourceLocator(kind="slide", slide=3),
        element_ids=("doc_pptx_body",),
    )

    payload = _resolve(client, "doc_pptx", "chunk_slide")
    view = payload["view"]

    assert payload["visual_targeting"]["capability"] == "slide_jump"
    assert view["kind"] == "slide_view"
    assert view["slide"] == 3
    # Only the cited slide's elements, in the slide's own order.
    assert [block["element_id"] for block in view["blocks"]] == [
        "doc_pptx_title",
        "doc_pptx_body",
    ]
    assert view["target"]["element_id"] == "doc_pptx_body"
    assert view["target"]["is_target"] is True
    assert view["target_rejected"] is None
    # Nothing is positioned, because there is nothing to position.
    assert not any(key in view for key in ("frame", "x", "y", "width", "height"))


def test_pptx_without_a_matching_element_reports_it_rather_than_guessing(client, store):
    """A slide whose elements are not the cited ones gets no target.

    Marking the slide's first element because the citation names the slide
    would be a highlight on the wrong lines in a different costume: the reader
    is told a specific element was cited and it was not.
    """
    _seed_pptx(store)
    _seed_chunk(
        store,
        "doc_pptx",
        "chunk_slide",
        SourceLocator(kind="slide", slide=3),
        element_ids=("doc_pptx_missing",),
    )

    view = _resolve(client, "doc_pptx", "chunk_slide")["view"]

    assert view["target"] is None
    assert view["target_rejected"]["reason_code"] == "target_not_in_region"
    assert view["target_rejected"]["message"]
    assert view["blocks"], "the slide still renders; only the marker is missing"


# --------------------------------------------------------------------------
# DOCX — the element
# --------------------------------------------------------------------------


def test_docx_locates_the_paragraph_with_its_context(client, store):
    """DOCX has no page grid, so context is what makes a paragraph findable."""
    _seed_docx(store)
    _seed_chunk(
        store,
        "doc_docx",
        "chunk_para",
        SourceLocator(
            kind="docx_element", paragraph=13, heading_path=("Introduction", "Related work")
        ),
        element_ids=("doc_docx_p13",),
    )

    payload = _resolve(client, "doc_docx", "chunk_para")
    view = payload["view"]

    assert payload["visual_targeting"]["capability"] == "element_jump"
    assert view["kind"] == "element_view"
    assert view["paragraph"] == 13
    assert view["heading_path"] == ["Introduction", "Related work"]
    assert view["target"]["element_id"] == "doc_docx_p13"
    # The neighbours come with it, because a paragraph index in a document with
    # no page numbers on screen is otherwise unfollowable.
    assert {block["paragraph"] for block in view["context"]} >= {12, 13, 14}
    assert [block["is_target"] for block in view["context"]].count(True) == 1
    assert view["target_rejected"] is None


def test_docx_falls_back_to_the_paragraph_index_when_the_chunk_names_no_element(client, store):
    """Content lookup, not a coordinate lookup — which is why it is allowed.

    The module's rule is that a *bounding box* is never borrowed from another
    element, because a borrowed box is a plausible wrong highlight. A DOCX
    locator carries no rectangle for that to happen to, and the alternative is
    refusing to show a paragraph the citation genuinely names. The paragraph
    number is matched exactly, never "nearest".
    """
    _seed_docx(store)
    _seed_chunk(
        store,
        "doc_docx",
        "chunk_para",
        SourceLocator(kind="docx_element", paragraph=14),
        element_ids=(),
    )

    view = _resolve(client, "doc_docx", "chunk_para")["view"]

    assert view["target"]["element_id"] == "doc_docx_p14"
    assert view["target"]["paragraph"] == 14


def test_a_format_with_nothing_to_lay_out_returns_no_view_at_all(client, store):
    """``None``, not an empty view.

    A spreadsheet range and a character offset have no layout, and the other
    two locator kinds in the schema are the only ones this task covers. A client
    handed an empty view would have to invent one; a client handed ``None``
    shows the passage and the state banner, which is the honest answer.
    """
    _seed_docx(store)
    _seed_chunk(store, "doc_docx", "chunk_txt", SourceLocator(kind="txt_offset", offset_start=0, offset_end=5))
    _seed_chunk(store, "doc_docx", "chunk_sheet", SourceLocator(kind="sheet_range", sheet="S1"))
    _seed_chunk(store, "doc_docx", "chunk_unknown", SourceLocator(kind="unknown"))

    for chunk_id in ("chunk_txt", "chunk_sheet", "chunk_unknown"):
        payload = _resolve(client, "doc_docx", chunk_id)
        assert payload["view"] is None, chunk_id
        # ...and the answer is still delivered, not lost with the view.
        assert payload["snippet"]["text"], chunk_id


# --------------------------------------------------------------------------
# The page-image route
# --------------------------------------------------------------------------


def test_the_page_image_route_answers_501_rather_than_a_blank_200(client, store):
    """A definite "no" beats three wrong answers.

    404 says the page is not in the document, 500 says the server broke, and an
    empty 200 says "here is your page" while handing over nothing. A client that
    cannot tell those apart renders a white rectangle with a highlight on it,
    which reads as a page that failed to draw its text.
    """
    _seed_pdf(store)

    response = client.get("/v1/documents/doc_pdf/pages/7/image")

    assert response.status_code == 501
    body = response.json()
    assert body["code"] == routes_evidence.PAGE_RENDER_CODE
    assert "渲染" in body["error"] or "渲染" in str(body)


def test_the_page_image_route_still_reports_a_missing_document_as_404(client, store):
    """501 is a statement about this build, and must not mask a 404."""
    response = client.get("/v1/documents/doc_nope/pages/1/image")

    assert response.status_code == 404
    assert response.json()["code"] == "document_not_found"


def test_the_page_image_route_rejects_a_page_below_one(client, store):
    """A negative page is a client bug, not a missing renderer."""
    response = client.get("/v1/documents/doc_pdf/pages/0/image")

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_page"


def test_the_page_image_route_rejects_a_malformed_document_id(client, store):
    response = client.get("/v1/documents/..%2Fetc/pages/1/image")

    assert response.status_code in (400, 404, 422)
    assert routes_evidence.PAGE_RENDER_CODE not in response.text


# --------------------------------------------------------------------------
# The client: structural guarantees
# --------------------------------------------------------------------------


def test_the_viewer_module_exists_and_adds_no_dependency():
    assert VIEWER, "evidenceViewer.js is missing"
    for framework in ("react", "vue", "angular", "svelte", "jquery", "preact", "solid-js"):
        assert framework not in VIEWER.lower()
    for specifier in re.findall(r"from\s+'([^']+)'", VIEWER):
        assert specifier.startswith("./"), f"the viewer imports {specifier}"
        assert (JS_DIR / specifier[2:]).is_file()
    assert "export " in VIEWER


def test_the_viewer_reads_the_target_rectangle_from_the_view_and_nowhere_else():
    """One source, one highlight. The client never parses the fragment.

    ``xywh=`` appearing in the client would mean a second representation of the
    same rectangle, and two representations of a coordinate are two chances to
    draw a highlight the resolver never promised. Its absence is the invariant.
    """
    body = strip_js_comments(VIEWER)
    assert "xywh" not in body, (
        "the client must not reconstruct a box from the pdf.js fragment; the "
        "view carries the same numbers and there must be exactly one of them"
    )
    assert body.count("view.target") >= 1
    # The only place a target is *placed* is inside the page-map renderer. The
    # slide and paragraph views read `view.target` as a presence flag and
    # nothing more — they have no coordinates to place, because their formats
    # have none.
    assert "view.target" in function_body(body, "renderPageMap")
    for name in ("renderSlideView", "renderElementView", "buildHighlightStatus", "placeBox"):
        renderer = function_body(body, name)
        assert "placeBox" not in renderer, f"{name} must not position anything"
    for name in ("renderSlideView", "renderElementView"):
        renderer = function_body(body, name)
        for coordinate in (".x", ".y", ".width", ".height"):
            assert coordinate not in renderer, f"{name} reads a coordinate off the target"
    # The page map is the one place the four numbers are read.
    page_map = function_body(body, "renderPageMap")
    assert "isDrawableBox(target)" in page_map


def test_the_client_re_validates_the_four_numbers_before_drawing():
    """Defence in depth: a client that trusts the server is one build behind.

    ``isDrawableBox`` is the second of two gates on the same failure. The
    server refuses a relative or reversed box; this refuses a non-finite
    coordinate, a zero extent and a negative one, so a future build that starts
    emitting a ``NaN`` produces no highlight rather than a ``NaN%``-positioned
    div that renders as nothing at all.
    """
    body = function_body(strip_js_comments(VIEWER), "isDrawableBox")

    assert "Number.isFinite" in body
    assert "typeof" in body, "a non-number must be refused, not coerced"
    assert "box.width > 0" in body
    assert "box.height > 0" in body
    assert "return false" in body


def test_the_y_axis_flip_happens_once_and_is_shared_by_the_blocks_and_the_highlight():
    """PDF y is up, CSS y is down. Getting this backwards mirrors the page.

    One function does the conversion and both the page blocks and the highlight
    go through it, so the two cannot be flipped in opposite directions — which
    would be a correct-looking highlight over a mirrored layout, i.e. wrong
    twice and detectably wrong to nobody.
    """
    body = function_body(strip_js_comments(VIEWER), "placeBox")

    assert "frame.y + frame.height - (box.y + box.height)" in body, (
        "the top offset must be measured down from the frame's top edge"
    )
    assert "placeBox(frame, block)" in VIEWER
    assert "placeBox(frame, target)" in VIEWER


def test_the_absent_highlight_is_a_named_state_and_never_an_empty_container():
    """The one thing a reader must be able to tell: "no box" is not "no page".

    ``data-highlight`` has exactly two legal values and the absence branch
    writes a reason code next to the sentence. A view that rendered nothing at
    all would be indistinguishable from a page that failed to load.
    """
    body = strip_js_comments(VIEWER)

    assert "'drawn'" in body and "'absent'" in body
    assert "data-highlight" in body or "dataset.highlight" in body
    status = function_body(body, "buildHighlightStatus")
    assert "target_rejected" in status
    assert "reasonCode" in status or "reason_code" in status
    # The reason code vocabulary is read, not hard-coded twice.
    assert "NO_TARGET_MESSAGES" in body
    assert "VIEW_TARGET_MISSING_CODE" in body


def test_the_view_renders_in_both_surviving_panel_states():
    """A page map with no highlight still says which page the citation is on.

    So the view is rendered before the availability branch is consulted, not
    inside it. Gating it on ``targeting.available`` would hide the located
    region in exactly the case — the relative box — where the reader is most
    entitled to see what *is* known.
    """
    body = function_body(strip_js_comments(EVIDENCE), "renderEvidencePayload")

    assert "renderEvidenceView" in body
    call = body.index("renderEvidenceView")
    # Locate the availability gate by the identifier, not by an exact statement
    # form. It previously matched on the literal ``if (!targeting.available)``
    # and the current code writes ``if (targeting.available === false)``. Both
    # gate the same way, so pinning the wording meant a faithful reconstruction
    # of this file was reported as a regression -- which is the failure mode
    # this suite has already produced once.
    branch = body.find("targeting.available")
    assert branch != -1, "the availability branch is gone from renderEvidencePayload"
    assert call < branch, "the view must render before the availability branch"


def test_the_viewer_container_is_declared_in_the_served_template():
    assert 'id="evidenceViewer"' in INDEX_HTML
    # ...and it lives inside the passage region, which is the only region shown
    # in both surviving states.
    content_start = INDEX_HTML.index('id="evidenceContent"')
    viewer_at = INDEX_HTML.index('id="evidenceViewer"')
    content_end = INDEX_HTML.index('id="evidenceWarnings"')
    assert content_start < viewer_at < content_end


def test_the_client_does_not_attempt_a_page_image_and_does_not_draw_one():
    """The renderer is absent, so nothing may pretend to be one.

    The path is written down in ``config.js`` and nowhere else in the front
    end: a module that issued the request would add one round trip per citation
    click to learn something the evidence response already carries in
    ``page_image.renderer``. An ``<img>`` would be worse — it is the exact
    "blank page that looks like a load failure" the requirement forbids.
    """
    others = {
        path.name: strip_js_comments(path.read_text(encoding="utf-8"))
        for path in sorted(JS_DIR.glob("*.js"))
        if path.name != "config.js"
    }
    for name, source in others.items():
        assert "EVIDENCE_PAGE_IMAGE_PATH" not in source, name
    body = strip_js_comments(VIEWER)
    assert "new Image(" not in body
    assert "createElement('img')" not in body
    assert ".src =" not in body
    # The absence is stated on the drawing, in the DOM, every render.
    assert "data-raster" in VIEWER or "dataset.raster" in VIEWER
    assert "VIEW_NOT_A_RASTER_BADGE" in VIEWER


def test_the_viewer_never_reads_a_bounding_box_off_a_text_fragment_or_a_neighbour():
    """Structural, not textual: the guarantee is that no other read exists.

    ``view.blocks`` is the page around the citation and must not become a source
    of coordinates. The only reads of a target's numbers are ``view.target`` and
    the four properties :func:`isDrawableBox` checks.
    """
    body = strip_js_comments(VIEWER)
    for name in ("renderPageMap", "renderSlideView", "renderElementView"):
        renderer = function_body(body, name)
        for forbidden in ("view.blocks.find", "blocks.find(", ".sort(", "querySelector"):
            assert forbidden not in renderer, f"{name} contains {forbidden}"


def test_the_degraded_reason_vocabulary_matches_the_resolver_exactly():
    """Two copies of a reason vocabulary, one contract, checked by a test.

    The resolver names the codes and ``config.js`` renders them. There is no
    shared module across the Python/JavaScript boundary, so this is the only
    thing that notices when a new ``REASON_*`` code lands on one side only —
    and a code the client does not know falls back to a generic sentence, which
    is a reader being told the wrong reason for a highlight that is missing.
    """
    server_codes = {
        re.search(rf'{name}\s*=\s*"([^"]+)"', RESOLVER).group(1)
        for name in re.findall(r"^(REASON_[A-Z_]+)\s*=\s*\"", RESOLVER, re.MULTILINE)
    }
    block = re.search(
        r"REASON_MESSAGES: dict\[str, str\] = \{(.*?)\n\}", RESOLVER, re.DOTALL
    )
    assert block, "REASON_MESSAGES is gone; this test needs updating deliberately"
    named = set(re.findall(r"^\s{4}(REASON_[A-Z_]+):\s*[\"(\n]", block.group(1), re.MULTILINE))
    # Every code the resolver can emit has a sentence; a code without one would
    # make `REASON_MESSAGES[reason_code]` a KeyError on the live path.
    declared = set(re.findall(r"^(REASON_[A-Z_]+)\s*=\s*\"", RESOLVER, re.MULTILINE))
    assert named == declared, f"no sentence for {sorted(declared - named)}"
    config_block = re.search(
        r"export const NO_TARGET_MESSAGES = \{(.*?)\n\};", CONFIG, re.DOTALL
    )
    assert config_block
    assert set(re.findall(r"^\s{4}([a-z_]+):", config_block.group(1), re.MULTILINE)) == server_codes


def test_the_citation_deep_link_an_export_writes_is_readable_by_the_client():
    """A link in an exported file has to resolve, or it is decoration.

    The server writes ``#evidence=<id>&document=<id>``; the client reads those
    two parameter names out of ``EVIDENCE_DEEP_LINK_PARAMS``. The names are the
    shared contract, and this is what holds both ends of it.
    """
    params = re.search(
        r"export const EVIDENCE_DEEP_LINK_PARAMS = \{(.*?)\};", CONFIG, re.DOTALL
    )
    assert params
    names = dict(re.findall(r"(\w+):\s*'([^']+)'", params.group(1)))
    assert names == {"evidence": "evidence", "document": "document"}

    body = strip_js_comments(EVIDENCE)
    read = function_body(body, "readDeepLinkTarget")
    assert "URLSearchParams" in read
    for key, value in names.items():
        assert f"EVIDENCE_DEEP_LINK_PARAMS.{key}" in read, key
        assert value in read, value
    # And it is actually wired to the two things that can navigate the client.
    assert "hashchange" in body
    assert "openDeepLinkedEvidence()" in function_body(body, "initializeEvidencePanel")


def test_the_viewer_is_reachable_from_the_entry_point():
    assert "renderEvidenceView" in EVIDENCE
    assert "from './evidenceViewer.js'" in EVIDENCE
    assert "initializeEvidencePanel" in MAIN
    assert INDEX_HTML.count('<script type="module"') == 1


def test_the_page_map_is_never_clamped_in_height():
    """A clamped map would silently rescale the coordinates and move the highlight.

    Every block is positioned as a percentage of this box, so a ``max-height``
    here would compress the frame while the percentages still claim to be of
    the whole — the highlight would end up over the wrong text while looking
    exactly as correct as before. That is the same class of defect as a
    relative box, reached by a different road, and it is a one-line CSS change
    away, which is precisely why it gets a test.
    """
    block = re.search(r"\.evidence-page-map \{(.*?)\}", INDEX_HTML, re.DOTALL)
    assert block, ".evidence-page-map is not styled in the template"
    rules = block.group(1)
    properties = [
        declaration.split(":")[0].strip()
        for declaration in rules.split(";")
        if ":" in declaration
    ]
    for forbidden in ("height", "max-height", "transform", "zoom"):
        assert forbidden not in properties, (
            f".evidence-page-map declares {forbidden}; the frame's own aspect "
            "ratio is the only thing that may set its size"
        )
    # ``min-height`` is allowed and ``max-height`` is not, and the difference is
    # the point: a floor makes a very flat page visible, while a ceiling would
    # clip the drawing and leave a block that says it is at 90% off the bottom
    # of a box that ends at 40%.
    assert "min-height" in properties
    # The aspect ratio comes from the frame, set per render in JS.
    assert "aspectRatio" in VIEWER
    # ...and the panel body is what scrolls, in both layouts.
    assert "overflow-y: auto" in INDEX_HTML
    assert "overscroll-behavior: contain" in INDEX_HTML


def test_what_these_tests_cannot_prove_is_stated_rather_than_implied():
    """A structural test proves there is no wrong-highlight *path*.

    It cannot prove the drawing looks right on a real page, at a real zoom, in a
    real browser, on a two-column paper whose blocks overlap. That remains a
    manual check, and pretending otherwise would be the same class of error as
    shipping a highlight nobody looked at: a green suite that certifies a
    property it does not test.
    """
    assert "isDrawableBox" in VIEWER  # the machine-checkable half exists
    assert "aspectRatio" in VIEWER, (
        "the frame's own aspect is what keeps a correct rectangle from being "
        "stretched into a wrong one; its absence would be untestable by hand too"
    )
