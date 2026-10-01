"""T3.3's real page image: a raster that exists, and a highlight that lands on it.

The single thing this file exists to establish is **where the highlight goes**.
Everything else — that the PNG is valid, that the status codes differ, that the
cache is bounded — is comparatively easy to get right, and all of it is worth
almost nothing if a reader clicks a citation and the rectangle lands on the
wrong lines while looking completely plausible. That is the one failure this
project's evidence design is built to make impossible, and "the image renders
and is the right size" is not evidence against it.

So the placement tests do not assert the formula. They assert **ink**. A
synthetic PDF with solid black rectangles at coordinates the test chose is
rendered through the real route, the returned PNG is decoded, and the rectangle
the metadata says to draw is checked against where the black pixels actually
are. A transform that is off by the page height, off by the scale factor,
mirrored, or transposed moves the box off solid black and onto white, and the
test fails. There is no way to pass these by restating the arithmetic.

Three properties are pinned separately, because they fail differently:

* **The flip.** ``BoundingBox`` is PDF user space — origin bottom-left, ``y``
  up — and an image is origin top-left, ``y`` down. Getting this backwards
  produces a highlight on the right page at the wrong height, which is the
  single most likely bug in this whole feature and the one a size assertion
  would never catch.
* **The scale.** Not the requested scale: the *achieved* one. pdfium's bitmap
  is not exactly ``round(page_pt * scale)`` (measured: a 595.276pt page at
  scale 1.0 renders 596px), so a client that multiplies by what it asked for
  is wrong by a fraction of a percent — too small to see, and wrong.
* **The rotation.** pdfium reports the *rotated* viewport from ``get_size()``
  while every coordinate in this project stays in unrotated user space. A
  ``/Rotate 90`` page is therefore a trap for anyone who uses ``get_size()``
  directly, and the route neutralises it. ``test_the_highlight_lands_on_the_ink``
  runs at all four rotation values to prove the client's formula has no special
  case.

Corpus reads are guarded with ``skipif`` and a reason: ``eval/corpus`` is not
committed in a fresh clone, and a test that fails on a missing fixture teaches
a contributor that the suite is broken rather than that the corpus is.
"""

from __future__ import annotations

import hashlib
import io
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from paperwhisperer.api import routes_evidence
from paperwhisperer.core import config
from paperwhisperer.infra.object_store import LAYER_ORIGINAL, FilesystemObjectStore
from paperwhisperer.infra.stores import build_sqlite_bundle
from paperwhisperer.infra.stores.interfaces import DocumentRecord
from paperwhisperer.ingestion.schema import (
    BoundingBox,
    Chunk,
    Element,
    ParsedDocument,
    SourceLocator,
)
from paperwhisperer.rag import evidence as evidence_module

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS = PROJECT_ROOT / "eval" / "corpus"
SINGLE_COL_PDF = CORPUS / "digital_single_col" / "hinton_distillation.pdf"

#: The rasteriser is an *extra*. Every test that needs it to actually render
#: skips without it, loudly and with a reason, so a personal-tier checkout
#: running ``make check`` reports "this extra is not installed" rather than a
#: wall of failures that look like a broken route.
HAS_RENDERER = evidence_module.page_renderer_name() != "none"
RENDERER_REASON = (
    "the optional 'render' extra is not installed (pypdfium2/Pillow); "
    "install it with: pip install -e \".[render]\""
)
requires_renderer = pytest.mark.skipif(not HAS_RENDERER, reason=RENDERER_REASON)

#: Reading pixels back out of the PNG needs an image library. Both are declared
#: parts of the ``render`` extra, so their absence is the same condition as the
#: rasteriser's; a separate marker is kept so the reason names what is missing.
try:
    import numpy as _np
    from PIL import Image as _Image

    HAS_PIXELS = HAS_RENDERER
except ImportError:  # pragma: no cover - the extra is absent
    _np = None
    _Image = None
    HAS_PIXELS = False

PIXEL_REASON = f"reading rendered pixels back requires numpy and Pillow: {RENDERER_REASON}"
requires_pixels = pytest.mark.skipif(not HAS_PIXELS, reason=PIXEL_REASON)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def objects_root(tmp_path) -> Path:
    """The object-store root the route will look in, redirected at ``tmp_path``.

    A fixture rather than a module constant because it has to be per-test: a
    shared root would let one test's originals satisfy another's lookup, and a
    page image for the wrong document is the exact failure this file is about.
    """
    return tmp_path / "objects"


@pytest.fixture
def store(tmp_path):
    """A personal-tier store on a throwaway SQLite file."""
    settings = SimpleNamespace(
        STORAGE_SQLITE_PATH=str(tmp_path / "page-render.sqlite3"),
        STORAGE_SQLITE_TIMEOUT_SECONDS=30,
        STORAGE_LEXICAL_FTS5="auto",
    )
    bundle = build_sqlite_bundle(settings)
    yield bundle
    bundle.close()


@pytest.fixture
def client(store, objects_root, monkeypatch):
    """A TestClient over an app that includes the evidence router.

    The object-store root is redirected at ``tmp_path`` so the test controls
    exactly which original bytes exist, and the render cache is emptied on the
    way out so nothing leaks between tests — a cached page from one test would
    satisfy a later test's assertion without rendering anything at all.
    """
    monkeypatch.setattr(config, "PAGE_RENDER_ORIGINALS_DIR", str(objects_root))
    application = FastAPI()
    application.include_router(routes_evidence.router)
    application.dependency_overrides[routes_evidence.get_store_bundle] = lambda: store
    routes_evidence._render_cache().clear()
    with TestClient(application) as test_client:
        yield test_client
    routes_evidence._render_cache().clear()


# --------------------------------------------------------------------------
# Fixture documents
# --------------------------------------------------------------------------


def make_pdf_bytes(
    rects, *, rotate: int = 0, width: float = 300, height: float = 400
) -> bytes:
    """A one-page PDF with solid black rectangles at exact user-space positions.

    Written byte by byte rather than with a PDF library for one reason: the
    point is to know precisely where the ink is, to the point, with no
    library's idea of a page box or a content stream getting in the way. Each
    rect is ``(x0, y0, x1, y1)`` in unrotated PDF user space, and pdfium will
    paint exactly that rectangle and nothing else.

    ``width`` and ``height`` accept fractions because that is not pedantry: a
    page whose box divides exactly into the requested scale cannot tell a
    client that computed the scale from the page size whether it is right, and
    a fractional page box (595.276pt, the A4 width most of this corpus uses) is
    the ordinary case rather than the exotic one.

    ``rotate`` writes a ``/Rotate`` entry, which is the whole reason this helper
    takes the parameter: ``/Rotate`` is invisible in every other way a test
    could set it up, and it is the one page property that silently invalidates
    the geometry contract.
    """
    content = " ".join(f"{x0} {y0} {x1 - x0} {y1 - y0} re" for x0, y0, x1, y1 in rects)
    stream = f"1 0 0 rg {content} f".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] "
            f"/Rotate {rotate} /Resources << >> /Contents 4 0 R >>"
        ).encode(),
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.7\n")
    offsets = []
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{index} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_at}\n%%EOF\n"
    ).encode()
    return bytes(out)


#: Two solid rectangles, chosen to pin both axes and both halves of the page in
#: one render. ``LOW_RIGHT`` is on the right of the page and near its *bottom*
#: in user space; ``HIGH_LEFT`` is on the left and near its *top*. A transform
#: that forgets to flip ``y`` swaps the two, a transform that transposes swaps
#: them differently, and a transform that gets the scale wrong puts both on
#: white. The page is 300x400 so "the right half" and "the bottom half" are not
#: the same predicate.
LOW_RIGHT = (200.0, 40.0, 260.0, 90.0)
HIGH_LEFT = (40.0, 300.0, 100.0, 360.0)
SYNTHETIC_RECTS = (LOW_RIGHT, HIGH_LEFT)
SYNTHETIC_PAGE_PT = (300.0, 400.0)


def store_original(
    store,
    objects_root: Path,
    payload: bytes,
    document_id: str,
    *,
    media_type: str = "application/pdf",
) -> str:
    """File ``payload`` as this document's original, and register the document.

    Going through the real object store rather than stubbing the route's
    resolver is the point of the fixture: it exercises the content-addressed
    lookup the route actually performs, so a change to how originals are filed
    (T2.2 finishing the job, say) is caught here instead of in production.

    ``put_file`` rather than ``put`` because the corpus fixture is a real file
    and reading it through the same API the ingestion path would is the most
    faithful thing a test can do.
    """
    objects = FilesystemObjectStore(objects_root)
    objects.initialize()
    staging = objects_root.parent / f"{document_id}.pdf"
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.write_bytes(payload)
    objects.put_file(LAYER_ORIGINAL, staging, suffix=".pdf")

    digest = hashlib.sha256(payload).hexdigest()
    store.metadata.upsert_document(
        DocumentRecord(
            document_id=document_id,
            document=ParsedDocument(
                document_id=document_id,
                source_filename=f"{document_id}.pdf",
                media_type=media_type,
                sha256=digest,
                page_count=1,
            ),
            source_filename=f"{document_id}.pdf",
            media_type=media_type,
            sha256=digest,
            page_count=1,
        )
    )
    return digest


def seed_synthetic(store, objects_root, *, rotate: int = 0, document_id: str = "doc_synth"):
    """Register a one-page document whose single page carries ``SYNTHETIC_RECTS``."""
    payload = make_pdf_bytes(SYNTHETIC_RECTS, rotate=rotate)
    store_original(store, objects_root, payload, document_id)
    return payload


def seed_corpus_pdf(store, objects_root, document_id: str = "doc_corpus") -> str:
    """Register a real paper from the eval corpus, originals and all."""
    if not SINGLE_COL_PDF.is_file():
        pytest.skip(f"the eval corpus is not present: {SINGLE_COL_PDF}")
    return store_original(store, objects_root, SINGLE_COL_PDF.read_bytes(), document_id)


def seed_chunk(store, document_id: str, chunk_id: str, locator: SourceLocator) -> None:
    """Store one retrieval unit — the thing an evidence id addresses."""
    store.metadata.put_chunk(
        document_id,
        Chunk(chunk_id=chunk_id, text="A line.", locator=locator, index_version=""),
    )


# --------------------------------------------------------------------------
# Reading the answers back
# --------------------------------------------------------------------------


def png_size(data: bytes) -> tuple[int, int]:
    """``(width, height)`` from a PNG's IHDR.

    Parsed by hand rather than through an image library so that "it is a PNG of
    the expected size" is a statement about the bytes on the wire, made by code
    that does not share a decoder with the encoder. A test that used the same
    library to write and to check would agree with a broken encoder.
    """
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG signature"
    assert data[12:16] == b"IHDR", "first chunk is not IHDR"
    width, height = struct.unpack(">II", data[16:24])
    return int(width), int(height)


def png_ink_bbox(data: bytes) -> tuple[int, int, int, int]:
    """The bounding box of non-white pixels, as ``(x0, y0, x1, y1)`` half-open.

    Measured on the decoded image, not on the encoded bytes, because that is
    what a browser puts on screen. "Where is the ink" is the ground truth this
    whole file is built on.
    """
    image = _Image.open(io.BytesIO(data)).convert("L")
    array = _np.asarray(image)
    ys, xs = _np.nonzero(array < 128)
    assert xs.size, "the rendered page has no ink at all"
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def place_box(box: BoundingBox, geometry: dict) -> tuple[int, int, int, int]:
    """Where a client must draw ``box`` on this image, using *only* the geometry.

    This is the contract, transcribed from
    :func:`paperwhisperer.rag.evidence.page_image_geometry`'s docstring and from
    nothing else — no import of the server's own transform, no shared helper. A
    test that called the server's function to check the server's function would
    pass if that function were wrong in a self-consistent way, which is the
    failure mode a geometry bug actually has.
    """
    scale_x = geometry["scale_x"]
    scale_y = geometry["scale_y"]
    height_pt = geometry["page_height_pt"]
    left = box.x0 * scale_x
    right = box.x1 * scale_x
    top = (height_pt - box.y1) * scale_y
    bottom = (height_pt - box.y0) * scale_y
    return (int(round(left)), int(round(top)), int(round(right)), int(round(bottom)))


#: How far the ink is allowed to sit from the placed box, in pixels.
#:
#: One pixel, and for a specific reason rather than out of caution. pdfium
#: rasterises with edge coverage, so the pixel row just outside a filled
#: rectangle picks up partial coverage and reads as ink. The ink bounding box
#: is therefore up to one pixel *larger* than the exact geometric rectangle —
#: measured at exactly one pixel on a 595.276pt page at scale 1.0. Anything the
#: transform gets wrong here is wrong by tens or hundreds of pixels (a
#: forgotten ``y``-flip is off by the page height, 841px), so a one-pixel slack
#: costs no discriminating power at all.
#:
#: The alternative — asserting exact equality — would be asserting something
#: false, and a test that is red for a reason nobody can explain is a test
#: people learn to ignore.
EDGE_COVERAGE_TOLERANCE_PX = 1


def assert_placement_on_ink(placed, ink, *, what: str = "") -> None:
    """The placed box coincides with the ink, to within edge coverage."""
    for index, (got, want) in enumerate(zip(placed, ink, strict=True)):
        assert abs(got - want) <= EDGE_COVERAGE_TOLERANCE_PX, (
            f"{what}placement {placed} does not coincide with the ink {ink} "
            f"(axis {index} is off by {abs(got - want)}px, tolerance is "
            f"{EDGE_COVERAGE_TOLERANCE_PX}px)"
        )


def fetch_geometry(client, document_id: str, page: int = 1, **params) -> tuple[dict, bytes]:
    """Fetch the geometry and the image for one page, asserting they agree.

    The two are fetched in the two forms a client has: ``format=json`` for a
    client that needs the viewport before the bytes, and the PNG plus its
    headers for a client that already has an ``<img>``. They come from one code
    path and one cache entry, and this asserts the JSON and the headers cannot
    drift — a client that sized a container from one and drew a box against the
    other would be off by exactly the discrepancy, and the discrepancy would be
    invisible.
    """
    url = f"/v1/documents/{document_id}/pages/{page}/image"
    meta = client.get(url, params={**params, "format": "json"})
    assert meta.status_code == 200, meta.text
    geometry = meta.json()

    image = client.get(url, params=params)
    assert image.status_code == 200, image.text

    for field, header in routes_evidence.GEOMETRY_HEADERS.items():
        assert float(image.headers[header]) == pytest.approx(
            float(geometry[field]), abs=1e-6
        ), f"header {header} disagrees with the json body for {field}"

    assert png_size(image.content) == (
        geometry["image_width_px"],
        geometry["image_height_px"],
    )
    return geometry, image.content


# --------------------------------------------------------------------------
# The render itself
# --------------------------------------------------------------------------


@requires_renderer
def test_a_real_corpus_page_renders_as_a_png_of_the_stated_size(client, store, objects_root):
    """The whole point of the extra: a real paper produces real pixels.

    Size, not just "it returned 200": a route that answered 200 with a blank
    image would pass a status assertion, and a blank image is precisely what
    this feature was rebuilt to stop being.
    """
    seed_corpus_pdf(store, objects_root)

    response = client.get("/v1/documents/doc_corpus/pages/1/image")

    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "image/png"
    # US Letter at scale 2: 612x792 points, so 1224x1584 device pixels. The
    # page box is read from the response rather than hard-coded, so the test
    # says something about the relationship instead of about my memory of it.
    assert float(response.headers["X-Page-Width-Pt"]) == 612.0
    assert float(response.headers["X-Page-Height-Pt"]) == 792.0
    assert png_size(response.content) == (1224, 1584)
    assert response.headers["X-Page-Source-Rotation"] == "0"


@requires_renderer
def test_the_scale_the_client_asked_for_is_the_scale_it_gets(client, store, objects_root):
    """A client that asks for 3x must be able to trust the geometry it is given.

    The failure this guards is not "the image is the wrong size" but "the image
    is the right size for a scale the client does not know", which makes every
    highlight three times too small and looks like a highlight on the wrong
    sentence rather than like a scaling bug.
    """
    seed_corpus_pdf(store, objects_root)

    geometry, image = fetch_geometry(client, "doc_corpus", page=1, scale=3)

    assert geometry["scale_requested"] == 3.0
    assert (geometry["image_width_px"], geometry["image_height_px"]) == (1836, 2376)
    assert png_size(image) == (1836, 2376)
    assert geometry["scale_x"] == pytest.approx(3.0, abs=1e-6)


@requires_renderer
def test_an_oversized_scale_request_is_clamped_rather_than_honoured(client, store, objects_root):
    """A render allocates an image proportional to a number off the wire.

    64x on a Letter page is 39 megapixels and about 120MB of raw pixels. The
    ceiling is what stops a mistaken or hostile query from turning a citation
    click into an out-of-memory kill, and clamping rather than 400 is what keeps
    a cosmetic mistake from becoming a citation that will not open.
    """
    seed_corpus_pdf(store, objects_root)

    geometry, _ = fetch_geometry(client, "doc_corpus", page=1, scale=64)

    assert geometry["scale_requested"] == 4.0
    assert geometry["image_width_px"] == 2448


# --------------------------------------------------------------------------
# The geometry contract, proved against ink
# --------------------------------------------------------------------------


@requires_pixels
@pytest.mark.parametrize("rotate", [0, 90, 180, 270])
def test_the_highlight_lands_on_the_ink(client, store, objects_root, rotate):
    """The decisive test: a box placed from the metadata alone covers the ink.

    Each case renders a page carrying **one** rectangle, so "the ink" is
    unambiguous and the expected placement is exactly the box. A page with two
    rectangles would not do: its ink bounding box is the union of the two, so
    any box landing on *either* would pass, and a transform that transposed the
    axes would sail through.

    Run at all four ``/Rotate`` values because that is where the contract is
    quietly load-bearing. pdfium's ``get_size()`` reports the *rotated*
    viewport, so a route that used it directly would report 400x300 for a
    ``/Rotate 90`` page while every ``BoundingBox`` in this project stays in
    unrotated user space — and the client's simple, correct, two-operation
    formula would then place the highlight on a page 90° from the text it
    names, often outside the image entirely.

    Because the route neutralises ``/Rotate`` before rendering, the ink lands in
    the same place for all four and the client needs no special case. That is
    asserted by *ink position*, not by comparing the four geometries to each
    other, so a route that rotated both the image and the coordinates
    consistently would also pass — which is correct, because that variant is
    also safe.
    """
    for index, rect in enumerate(SYNTHETIC_RECTS):
        document_id = f"doc_ink{index}"
        store_original(store, objects_root, make_pdf_bytes((rect,), rotate=rotate), document_id)

        geometry, image = fetch_geometry(client, document_id)

        # Rendered unrotated, so every rotation reports the same viewport: the
        # unrotated one, which is the space the bbox is in.
        assert geometry["rotation"] == 0
        assert geometry["source_rotation"] == rotate % 360
        assert geometry["page_width_pt"] == SYNTHETIC_PAGE_PT[0]
        assert geometry["page_height_pt"] == SYNTHETIC_PAGE_PT[1]

        box = BoundingBox(x0=rect[0], y0=rect[1], x1=rect[2], y1=rect[3])
        placed = place_box(box, geometry)
        ink = png_ink_bbox(image)
        assert_placement_on_ink(placed, ink, what=f"rotate={rotate} rect={rect}: ")


@requires_pixels
def test_the_highlight_lands_in_the_right_half_of_the_page(client, store, objects_root):
    """Placement is correct *within* the page, not merely on it.

    The weaker assertion — that the rectangle lands somewhere on the image — is
    satisfied by a transform that is 90° out, because a page is roughly
    square-ish and a transposed box still lands on paper. This pins the two
    halves independently, because "right" and "bottom" are different predicates
    and a ``y``-flip bug that gets ``x`` right is exactly the bug that looks
    plausible.
    """
    seed_synthetic(store, objects_root)
    geometry, _ = fetch_geometry(client, "doc_synth")

    width = geometry["image_width_px"]
    height = geometry["image_height_px"]

    def place(rect):
        return place_box(
            BoundingBox(x0=rect[0], y0=rect[1], x1=rect[2], y1=rect[3]), geometry
        )

    low_right = place(LOW_RIGHT)
    high_left = place(HIGH_LEFT)

    for box in (low_right, high_left):
        left, top, right, bottom = box
        assert 0 <= left < right <= width, f"{box} is not inside the {width}px-wide image"
        assert 0 <= top < bottom <= height, f"{box} is not inside the {height}px-tall image"

    # LOW_RIGHT is on the right of the page, and near its bottom in user space.
    assert low_right[0] > width / 2
    assert low_right[2] > width / 2
    assert (low_right[1] + low_right[3]) / 2 > height / 2
    # HIGH_LEFT is on the left, and near its top.
    assert high_left[2] < width / 2
    assert (high_left[1] + high_left[3]) / 2 < height / 2


@requires_pixels
def test_a_known_box_lands_on_the_text_it_names(client, store, objects_root):
    """Against real prose, not just synthetic blocks.

    The synthetic rectangles prove the transform. This proves the thing a
    reader cares about: a box over an actual line of a real paper lands on that
    line. The box is taken from pdfium's own text extraction, which is
    independent of everything this repository does with coordinates — the parse,
    the schema, the resolver — and therefore a real check that "the box the
    evidence names" and "the pixels under it" agree.
    """
    seed_corpus_pdf(store, objects_root)
    geometry, image = fetch_geometry(client, "doc_corpus", page=1)

    import pypdfium2 as pdfium

    page = pdfium.PdfDocument(SINGLE_COL_PDF.read_bytes())[0]
    textpage = page.get_textpage()
    textpage.count_rects()
    assert textpage.count_rects() > 4, "the fixture page has no text to aim at"

    # The title line: wide, near the top of the page, and unambiguous.
    rect = textpage.get_rect(1)
    box = BoundingBox(x0=rect[0], y0=rect[1], x1=rect[2], y1=rect[3])
    left, top, right, bottom = place_box(box, geometry)

    width = geometry["image_width_px"]
    height = geometry["image_height_px"]
    assert 0 <= left < right <= width
    assert 0 <= top < bottom <= height
    # The title is in the top third of a Letter page in user space, and a
    # y-flip that was forgotten would put it in the bottom third.
    assert (top + bottom) / 2 < height / 3

    # And it lands on ink: the text line the box names, not blank paper. The
    # topmost ink on the page is the arXiv stamp in the left margin, at
    # roughly the same height as the title, so a box that is on the title is
    # within a line's height of the topmost ink.
    ink = png_ink_bbox(image)
    assert ink[0] <= left and ink[2] >= right, (
        f"the placed box {(left, top, right, bottom)} does not overlap the page's ink {ink}"
    )
    assert abs(ink[1] - top) < 60, (
        f"the placed box {(left, top, right, bottom)} sits {top - ink[1]}px from the "
        f"topmost ink {ink}"
    )


@requires_pixels
@pytest.mark.parametrize("scale", ["1.0", "1.5", "2.0"])
def test_the_placement_factor_is_the_achieved_scale_not_the_requested_one(
    client, store, objects_root, scale
):
    """The reason ``scale_x`` exists at all, pinned where it actually bites.

    595.276 x 841.89 is the A4 page box most of the eval corpus uses, and it
    does not divide evenly into any convenient scale. At scale 1.0 pdfium
    produces a 596px-wide bitmap for a 595.276pt page, so the true ratio is
    1.0012 — 0.12% larger than the 1.0 a client would multiply by.

    The rectangle is at the far corner of the page on purpose. A box at
    ``x=50`` is off by 0.06px under this error and rounds to the same integer;
    one at ``x=500`` is off by 0.6px and does not. This is why "it looked fine"
    is not a defence against a geometry bug: the error is largest exactly where
    the text is, at the right margin.

    What this test cannot do is make the wrong scale fail by *ink
    displacement*, because the error is sub-pixel by construction — the largest
    it gets on any page narrower than a metre is about 0.7px. So the assertion
    that actually bites is the numeric one: the reported placement factor must
    be the achieved ratio and must *not* be the requested scale. A descriptor
    that reported ``scale_requested`` as the placement factor fails that, and
    the ink check is there to confirm the factor is the right one.
    """
    rect = (500.0, 700.0, 590.0, 800.0)
    store_original(
        store,
        objects_root,
        make_pdf_bytes((rect,), width=595.276, height=841.89),
        "doc_awkward",
    )

    geometry, image = fetch_geometry(client, "doc_awkward", scale=scale)
    requested = float(scale)
    ink = png_ink_bbox(image)

    # The reported factor is the ratio of the bytes actually produced.
    assert geometry["scale_x"] == pytest.approx(
        geometry["image_width_px"] / geometry["page_width_pt"], abs=1e-6
    )
    assert geometry["scale_y"] == pytest.approx(
        geometry["image_height_px"] / geometry["page_height_pt"], abs=1e-6
    )
    # The fixture's whole purpose: the two numbers must differ here, or the
    # test would prove nothing about the difference. This assertion is the one
    # that fails if the placement factor is ever the requested scale.
    assert geometry["scale_x"] != pytest.approx(requested, abs=1e-6) or (
        geometry["scale_y"] != pytest.approx(requested, abs=1e-6)
    ), (
        "this page box divides evenly into the tested scale, so the requested "
        "and achieved scales agree and the test would be vacuous"
    )

    box = BoundingBox(x0=rect[0], y0=rect[1], x1=rect[2], y1=rect[3])
    assert_placement_on_ink(place_box(box, geometry), ink)

    # And, stated as the thing it protects against: how far a client that
    # multiplied by the scale it asked for would put the right-hand edge of
    # this box. It is a fraction of a pixel, which is precisely why it is
    # dangerous — small enough that nobody reports a highlight as being "a bit
    # off" and large enough that the edge of the rectangle never quite reaches
    # the text it covers. The rect is at the far margin so this is close to the
    # largest displacement the page allows, and it varies with the scale
    # because a different scale lands the bitmap on a different rounding error
    # (0.6px at scale 1.0, 0.08px at 1.5, 0.35px at 2.0).
    displacement = box.x1 * abs(geometry["scale_x"] - requested)
    assert displacement > 0.05, (
        f"the requested and achieved scales differ by only {displacement}px at the "
        "right margin, so this fixture can no longer distinguish them"
    )


@requires_renderer
def test_geometry_only_answers_without_sending_bytes_to_the_client(client, store, objects_root):
    """A client needs the viewport *before* the bytes to size a container.

    Both forms come from one cache entry and one code path, so a client cannot
    end up with a scale from the JSON and a viewport from the image. The ETag
    is the same on both, which is what lets a client revalidate cheaply instead
    of re-fetching on every citation click.
    """
    seed_corpus_pdf(store, objects_root)
    url = "/v1/documents/doc_corpus/pages/1/image"

    meta = client.get(url, params={"format": "json"})
    image = client.get(url)

    assert meta.json()["image_width_px"] == png_size(image.content)[0]
    assert meta.headers["ETag"] == image.headers["ETag"]
    assert meta.json()["page_count"] == 9
    # A browser cache may hold this, but only privately: the bytes are a user's
    # document. "no-store" would be the wrong call here (the render is a pure
    # function of content-addressed bytes, so two readers of one document
    # legitimately get the same image) and a public cache would be a leak.
    assert "private" in image.headers["cache-control"]


# --------------------------------------------------------------------------
# The safety rails
# --------------------------------------------------------------------------


@requires_renderer
def test_a_relative_box_is_still_refused_even_though_the_page_now_renders(
    client, store, objects_root
):
    """The image must not become a way around ``to_query()``'s refusal.

    This is the requirement the whole feature could most easily have broken, and
    it is worth being precise about why rendering does not help. A relative box
    is a *mirrored* box: it has the right size, the right ordering, the right
    page, and it points at a region mirrored about the page's top edge. The only
    thing that distinguishes it is where it came from, and that knowledge is
    gone by the time a client has a page number. Showing the real page makes the
    mistake *more* legible, not less — the reader sees a confident highlight on
    confidently wrong lines, with the actual passage somewhere else on the page
    they are looking at.

    So the verdict must be identical with and without a rasteriser, and the
    payload must not smuggle a drawable box out through the image descriptor.
    """
    seed_synthetic(store, objects_root)
    # y0 < 0 is the shape a TOPLEFT-converting adapter produces, and the shape
    # BoundingBox itself flags as relative.
    mirrored = BoundingBox(x0=61.5, y0=-180.0, x1=361.5, y1=-120.0)
    assert mirrored.relative, "the fixture box is not the mirrored shape it claims to be"
    seed_chunk(
        store, "doc_synth", "chunk_mirrored", SourceLocator(kind="pdf_page", page=1, bbox=mirrored)
    )

    response = client.get("/v1/documents/doc_synth/evidence/chunk_mirrored")

    assert response.status_code == 200
    targeting = response.json()["visual_targeting"]
    # The verdict is unchanged, and that is the assertion: a rasteriser exists
    # and the reader still gets no box.
    assert targeting["available"] is False
    assert targeting["reason_code"] == "relative_bbox"
    assert targeting["highlight"] is None
    # And the page image descriptor does not become a back door: it advertises
    # the image, never a box, so a client reading only this block still has
    # nothing to draw.
    page_image = targeting["page_image"]
    assert page_image["renderer"] == "pypdfium2"
    assert page_image["url"] is not None
    assert "box" not in page_image and "highlight" not in page_image


@requires_renderer
def test_the_page_image_route_refuses_to_be_given_a_box(client, store, objects_root):
    """There is deliberately no ``?bbox=`` parameter, and that is a refusal.

    It is worth pinning because it is the obvious next feature and the obvious
    wrong turn. A bbox parameter here would be a second, weaker copy of
    ``BoundingBox.to_query()``: it would accept a rectangle this module has
    already decided is undrawable, and it would do so behind a URL that looks
    like a *rendering* rather than a *verdict*. Rendering the box server-side
    would also quietly make the server the place highlights are decided, which
    is the one place this project's design says they must not be decided.

    So a bbox in the query string is ignored, and the bytes are byte-identical
    to a request without one.
    """
    seed_synthetic(store, objects_root)
    url = "/v1/documents/doc_synth/pages/1/image"

    plain = client.get(url)
    with_box = client.get(url, params={"bbox": "10,10,50,50"})
    with_xywh = client.get(url, params={"xywh": "10,10,50,50"})

    assert plain.status_code == 200
    assert with_box.content == plain.content
    assert with_xywh.content == plain.content
    # And the response carries no rectangle for a client to have smuggled one
    # in through: the only coordinates in it describe the page, not a region.
    body = client.get(url, params={"format": "json"}).json()
    assert not any(
        "bbox" in key or key in ("x", "y", "width", "height", "box") for key in body
    )


# --------------------------------------------------------------------------
# The failure modes, and the codes that keep them apart
# --------------------------------------------------------------------------


@requires_renderer
def test_a_page_past_the_end_of_the_document_is_a_clear_error_not_a_blank_image(
    client, store, objects_root
):
    """Out of range is a fact about the document, and the body says how long it is.

    The alternative — letting pdfium's ``__getitem__`` raise, or returning a
    white page — is how a reader ends up looking at a blank rectangle and
    concluding the citation points at an empty part of the paper. The page count
    in the message is the one thing that lets a client recover rather than just
    report.
    """
    seed_corpus_pdf(store, objects_root)  # 9 pages

    response = client.get("/v1/documents/doc_corpus/pages/10/image")

    assert response.status_code == 404
    body = response.json()
    assert body["code"] == "page_out_of_range"
    assert "9" in body["error"]


@requires_renderer
def test_a_non_pdf_document_is_refused_by_its_bytes_not_its_filename(client, store, objects_root):
    """A ``.docx`` named ``.pdf`` must not reach the rasteriser.

    The check is on the content, because the content is the only thing that
    cannot be forged by a rename. Handing pdfium a ZIP would produce either an
    exception or, worse, a page of a document the user never opened — and a
    highlight placed on it would be exactly as wrong as any other wrong
    highlight, and harder to explain.
    """
    objects = FilesystemObjectStore(objects_root)
    objects.initialize()
    # A ZIP header is what a .docx/.pptx actually starts with.
    payload = b"PK\x03\x04" + b"\x00" * 512
    objects.put(LAYER_ORIGINAL, payload, content_type="application/pdf", suffix=".pdf")
    digest = hashlib.sha256(payload).hexdigest()
    store.metadata.upsert_document(
        DocumentRecord(
            document_id="doc_notpdf",
            document=ParsedDocument(
                document_id="doc_notpdf",
                source_filename="looks-like-a-paper.pdf",
                media_type="application/pdf",
                sha256=digest,
            ),
            source_filename="looks-like-a-paper.pdf",
            # Declared as a PDF on purpose: the media type is the claim, the
            # bytes are the evidence, and the route must believe the evidence.
            media_type="application/pdf",
            sha256=digest,
        )
    )

    response = client.get("/v1/documents/doc_notpdf/pages/1/image")

    assert response.status_code == 415
    assert response.json()["code"] == "page_render_unsupported_format"


@requires_renderer
def test_a_corrupt_pdf_is_a_file_problem_not_a_server_bug(client, store, objects_root):
    """A PDF pdfium cannot open is 422, and the server does not say 500.

    A 500 tells the operator something broke on the server; what actually
    happened is that a file arrived truncated or encrypted. The client's advice
    is different too — ask for a different file — so the two are worth
    distinguishing, and a bare 500 would also bury the real reason in a log line
    nobody opens.
    """
    payload = b"%PDF-1.7\n" + b"this is not a body\n" * 20
    store_original(store, objects_root, payload, "doc_broken")

    response = client.get("/v1/documents/doc_broken/pages/1/image")

    assert response.status_code == 422
    assert response.json()["code"] == "page_render_failed"


@requires_renderer
def test_a_document_whose_original_is_gone_is_reported_not_rendered(client, store, objects_root):
    """No retained bytes is a named outcome, and definitely not a white page.

    This is the state most documents on a fresh checkout are actually in: T2.2's
    ``original`` layer is not yet written by anything, and the upload route
    deletes the file once analysis is done. Reporting it under its own code keeps
    "this deployment keeps no uploads" distinguishable from "this deployment has
    no renderer", which a client needs because the two have completely different
    fixes — and both are kept away from a 200, because a 200 with a blank page
    is the failure this feature exists to eliminate.
    """
    store.metadata.upsert_document(
        DocumentRecord(
            document_id="doc_no_original",
            document=ParsedDocument(
                document_id="doc_no_original",
                source_filename="paper.pdf",
                media_type="application/pdf",
                sha256="a" * 64,
            ),
            source_filename="paper.pdf",
            media_type="application/pdf",
            sha256="a" * 64,
        )
    )

    response = client.get("/v1/documents/doc_no_original/pages/1/image")

    assert response.status_code == 501
    body = response.json()
    # The code is the one a client already branches on — it is unchanged, so
    # no front-end change is needed for this branch — and the narrower fact
    # rides alongside it.
    assert body["code"] == routes_evidence.PAGE_RENDER_CODE
    assert body["reason"] == routes_evidence.ORIGINAL_UNAVAILABLE_REASON
    assert response.headers.get("content-type", "").startswith("application/json")


@requires_renderer
def test_a_missing_document_is_still_a_404_and_never_a_capability_answer(
    client, store, objects_root
):
    """The ordering that keeps a typo from looking like a dependency problem.

    501 is a statement about this build. Answering it for a document that is
    also missing would hide a 404 behind a capability answer and send a client
    looking for something to install instead of for the typo it made.
    """
    seed_corpus_pdf(store, objects_root)

    response = client.get("/v1/documents/doc_nope/pages/1/image")

    assert response.status_code == 404
    assert response.json()["code"] == "document_not_found"


def test_the_501_path_still_answers_when_the_renderer_is_absent(
    client, store, objects_root, monkeypatch
):
    """With the extra deliberately missing, the old definite "no" comes back.

    Simulated by putting ``None`` in ``sys.modules``, which is how Python spells
    "this package is not installed" and which makes both ``import`` and
    ``importlib.util.find_spec`` fail — so the probe and the route agree, which
    is the property under test. Using the same string the probe looks up matters:
    a test that blocked some *other* name would prove nothing.

    The payload has to say so too. A client that reads ``renderer`` to decide
    whether to ask must not be told a renderer exists on a deployment that
    cannot render, or it shows an empty panel and calls it a loading failure.
    """
    seed_synthetic(store, objects_root)
    seed_chunk(
        store,
        "doc_synth",
        "chunk_ok",
        SourceLocator(kind="pdf_page", page=1, bbox=BoundingBox(200.0, 40.0, 260.0, 90.0)),
    )

    for module in evidence_module.PAGE_RENDERER_MODULES:
        monkeypatch.setitem(sys.modules, module, None)

    assert evidence_module.page_renderer_name() == "none"

    image = client.get("/v1/documents/doc_synth/pages/1/image")
    assert image.status_code == 501
    assert image.json()["code"] == routes_evidence.PAGE_RENDER_CODE

    evidence = client.get("/v1/documents/doc_synth/evidence/chunk_ok")
    assert evidence.status_code == 200
    payload = evidence.json()
    # The verdict about the box is untouched by the renderer's absence: the
    # whole point of the ``view`` block is that a deployment with no rasteriser
    # still locates the passage.
    assert payload["visual_targeting"]["available"] is True
    assert payload["visual_targeting"]["highlight"] == "xywh=200,40,60,50"
    page_image = payload["visual_targeting"]["page_image"]
    assert page_image["renderer"] == "none"
    assert page_image["url"] is None
    assert page_image["scale"] is None
    assert page_image["rendered"] is False
    # The schematic still arrives, and still says what it is.
    assert payload["view"] is not None
    assert payload["view"]["is_raster"] is False
    # And the badge is not gone: a client must still be able to tell the reader
    # that what it is about to show is not a picture of the page.
    assert "不是原始" in payload["view"]["note"]


def test_the_capability_check_comes_before_the_storage_check(
    client, store, objects_root, monkeypatch
):
    """"No renderer" must win over "no original bytes", for a document with neither.

    Both are 501, so the *status* cannot tell a client which situation it is
    in — only the code can, and the code is decided by which check runs first.
    A build without the rasteriser should say so, because that is the one fact
    a reader can act on and the one that explains every request failing; a
    deployment that has the renderer but keeps no uploads should say that
    instead, because that is a different fact with a different fix.

    The ordering is asserted by constructing the case where the two disagree:
    no renderer installed *and* a document whose bytes cannot be found. If the
    storage check ran first, the answer would be ``original_unavailable`` and
    the client would go looking for retained uploads on a machine that cannot
    render at all.

    This is also what makes the early capability check worth having beyond
    saving work. Without it the route would read and hash every original before
    discovering it cannot render anything, and a deployment without the extra
    would pay the full cost of the feature it does not have.
    """
    store.metadata.upsert_document(
        DocumentRecord(
            document_id="doc_nothing",
            document=ParsedDocument(
                document_id="doc_nothing", source_filename="paper.pdf",
                media_type="application/pdf", sha256="b" * 64,
            ),
            source_filename="paper.pdf",
            media_type="application/pdf",
            sha256="b" * 64,
        )
    )

    for module in evidence_module.PAGE_RENDERER_MODULES:
        monkeypatch.setitem(sys.modules, module, None)

        # The same document answers differently once the renderer is back, which is
    # what proves the two codes are telling apart two situations rather than
    # one code answering everything.
    without = client.get("/v1/documents/doc_nothing/pages/1/image")
    assert without.status_code == 501
    assert without.json()["code"] == routes_evidence.PAGE_RENDER_CODE
    # No ``reason``: nothing was looked up, so there is no narrower fact to
    # report. This is what distinguishes the early capability answer from the
    # storage one — a route that reached the storage check anyway would answer
    # the same status and the same code, and only this field says which.
    assert "reason" not in without.json()

    monkeypatch.undo()
    assert evidence_module.page_renderer_name() == "pypdfium2"

    with_renderer = client.get("/v1/documents/doc_nothing/pages/1/image")
    assert with_renderer.status_code == 501
    assert with_renderer.json()["code"] == routes_evidence.PAGE_RENDER_CODE
    assert (
        with_renderer.json()["reason"] == routes_evidence.ORIGINAL_UNAVAILABLE_REASON
    )


# --------------------------------------------------------------------------
# The cache
# --------------------------------------------------------------------------


@requires_renderer
def test_the_cache_is_keyed_on_content_so_a_re_parse_cannot_stale_it(
    client, store, objects_root
):
    """The key is the original's digest, not the document id.

    A re-parse keeps the document id and changes everything derived from it —
    the parse, the chunks, the index version, the boxes. Keying the render cache
    on the id would serve a pre-re-parse raster to a post-re-parse reader, which
    is the same defect as a wrong highlight arrived at by a different road. It
    cannot happen here: a render is a function of the original bytes and the
    page number, so a re-parse leaves the key untouched and the answer
    legitimately unchanged.
    """
    seed_corpus_pdf(store, objects_root)
    url = "/v1/documents/doc_corpus/pages/1/image"

    first = client.get(url)
    assert first.status_code == 200
    assert len(routes_evidence._render_cache()) == 1

    # Re-parse: same document id, a new record, the same original bytes. This is
    # what T3.4 does, and it must not produce a second render or a different
    # one. Round-tripped through ``from_dict`` so this goes through the same
    # rehydration a real store would.
    record = store.metadata.get_document("doc_corpus")
    store.metadata.upsert_document(
        DocumentRecord.from_dict(
            {
                **record.to_dict(),
                "version": record.version + 1,
                "index_version": "a-different-index-version",
            }
        )
    )
    after_reparse = client.get(url)
    assert after_reparse.content == first.content
    assert len(routes_evidence._render_cache()) == 1

    # A different page, and a different scale, are different renders and get
    # their own entries — so the key is not so coarse that it answers a
    # different question with the right bytes.
    client.get(url.replace("/1/image", "/2/image"))
    assert len(routes_evidence._render_cache()) == 2
    client.get(url, params={"scale": 3})
    assert len(routes_evidence._render_cache()) == 3


@requires_renderer
def test_a_document_id_whose_bytes_change_gets_a_new_render(client, store, objects_root):
    """The digest in the key is what makes a stale render impossible.

    A re-parse of the *same* file cannot go wrong here, because a re-parse does
    not change the original — so the test that covers that is above. This covers
    the other direction: the same ``document_id`` now naming *different bytes*.
    Nothing in the current pipeline does that, which is exactly why it needs a
    test rather than a comment: the moment something does (a re-upload, a
    migration that re-files an upload, a restore that puts different content
    under an id a store already knew), a cache keyed on the id would serve the
    previous document's page under the new document's evidence, and the
    highlight would sit on a page of a different paper. That is a data leak
    with a plausible-looking UI, which is the worst combination available.
    """
    first = make_pdf_bytes(((200.0, 40.0, 260.0, 90.0),))
    store_original(store, objects_root, first, "doc_swapped")
    first_render = client.get("/v1/documents/doc_swapped/pages/1/image")
    assert first_render.status_code == 200

    # Same id, different bytes, and the record now points at the new digest.
    second = make_pdf_bytes(((20.0, 20.0, 90.0, 70.0),))
    digest = store_original(store, objects_root, second, "doc_swapped")
    record = store.metadata.get_document("doc_swapped")
    assert record.sha256 == digest
    assert len(routes_evidence._render_cache()) == 1

    second_render = client.get("/v1/documents/doc_swapped/pages/1/image")

    assert second_render.status_code == 200
    assert second_render.content != first_render.content, (
        "the same document id served a render of the previous document's bytes"
    )
    assert second_render.headers["ETag"] != first_render.headers["ETag"]
    assert len(routes_evidence._render_cache()) == 2


@requires_renderer
def test_two_documents_with_the_same_page_do_not_share_a_render(client, store, objects_root):
    """Content addressing is what stops one paper's page from being another's.

    Identical page numbers, identical scales, and a cache that would happily
    answer both from one entry if the key were ``(page, scale)``. This is the
    concrete form of "a shared cache answering a later reader with an earlier
    reader's page", and it is why the key carries a digest.
    """
    seed_synthetic(store, objects_root, document_id="doc_a")
    store_original(
        store, objects_root, make_pdf_bytes(((20.0, 20.0, 90.0, 70.0),)), "doc_b"
    )

    a = client.get("/v1/documents/doc_a/pages/1/image")
    b = client.get("/v1/documents/doc_b/pages/1/image")

    assert a.status_code == b.status_code == 200
    assert a.content != b.content
    assert a.headers["ETag"] != b.headers["ETag"]
    assert len(routes_evidence._render_cache()) == 2


@requires_renderer
def test_the_cache_is_bounded_and_evicts_rather_than_growing(
    client, store, objects_root, monkeypatch
):
    """An unbounded render cache in a long-lived process is a memory leak.

    A page render is a few hundred kilobytes of PNG, and a personal-tier
    process is up for weeks. Both bounds are enforced — an entry count and a
    byte budget — because a bound on only one of them is not a bound on memory:
    4096 small entries or 8 large ones would each sail past the other limit.
    """
    seed_corpus_pdf(store, objects_root)
    monkeypatch.setattr(config, "PAGE_RENDER_CACHE_MAX_ENTRIES", 3)
    monkeypatch.setattr(config, "PAGE_RENDER_CACHE_MAX_BYTES", 8 * 1024 * 1024)
    routes_evidence._render_cache().clear()

    for page in range(1, 7):
        response = client.get(f"/v1/documents/doc_corpus/pages/{page}/image")
        assert response.status_code == 200, page

    cache = routes_evidence._render_cache()
    assert cache.max_entries == 3
    assert len(cache) == 3
    # Re-requesting the newest page must not have to re-render, which is the
    # whole reason the newest entry is the one that survived eviction.
    again = client.get("/v1/documents/doc_corpus/pages/6/image")
    assert again.status_code == 200
    assert len(cache) == 3


@requires_renderer
def test_the_cache_byte_budget_evicts_too(client, store, objects_root, monkeypatch):
    """The byte budget is a real bound, not a second copy of the entry count.

    Set to a size that fits fewer pages than the entry count allows, so the
    entry limit is never the binding one and this test would pass even if the
    byte accounting were deleted.
    """
    seed_corpus_pdf(store, objects_root)
    one_page = client.get("/v1/documents/doc_corpus/pages/1/image")
    page_bytes = len(one_page.content)
    monkeypatch.setattr(config, "PAGE_RENDER_CACHE_MAX_ENTRIES", 100)
    monkeypatch.setattr(config, "PAGE_RENDER_CACHE_MAX_BYTES", page_bytes * 2)
    routes_evidence._render_cache().clear()

    for page in range(1, 6):
        assert client.get(f"/v1/documents/doc_corpus/pages/{page}/image").status_code == 200

    cache = routes_evidence._render_cache()
    assert len(cache) <= 2, f"the byte budget did not bound the cache: {len(cache)} entries"


@requires_renderer
def test_a_cache_of_zero_entries_renders_correctly_and_caches_nothing(
    client, store, objects_root, monkeypatch
):
    """Turning the cache off must turn off only the cache.

    Worth pinning because "no cache" is the configuration an operator reaches for
    when the cache is suspected of being the problem, and a route that treated
    it as "no rendering" would turn a memory question into a dead feature.
    """
    seed_corpus_pdf(store, objects_root)
    monkeypatch.setattr(config, "PAGE_RENDER_CACHE_MAX_ENTRIES", 0)
    monkeypatch.setattr(config, "PAGE_RENDER_CACHE_MAX_BYTES", 0)
    routes_evidence._render_cache().clear()

    first = client.get("/v1/documents/doc_corpus/pages/1/image")
    second = client.get("/v1/documents/doc_corpus/pages/1/image")

    assert first.status_code == second.status_code == 200
    assert first.content == second.content
    assert len(routes_evidence._render_cache()) == 0


# --------------------------------------------------------------------------
# Where the original comes from
# --------------------------------------------------------------------------


@requires_renderer
def test_the_original_is_found_by_content_address(client, store, objects_root):
    """The digest on the record is the lookup key, and nothing else is needed.

    No object key, no path, no filename: the ``original`` layer is
    content-addressed, so a digest resolves to a file arithmetically. That is
    what makes "find this document's bytes" a lookup rather than a directory
    walk, and it is why the route does not need T2.2 to be finished before it
    works.

    The fixture's record carries *only* a ``sha256`` — no ``original_key``, no
    ``original_path`` — so the digest really is doing all the work. That the
    route also has to try the ``.pdf`` spelling is a real wart in
    :mod:`paperwhisperer.infra.object_store`: ``put_file`` files an upload under
    its extension, but ``find_by_digest`` only probes the layer's *declared*
    suffix, and ``original`` declares none. Reported separately; the route works
    around it rather than depending on it being fixed.
    """
    seed_corpus_pdf(store, objects_root)

    response = client.get("/v1/documents/doc_corpus/pages/1/image")

    assert response.status_code == 200
    record = store.metadata.get_document("doc_corpus")
    assert record.metadata == {}, "the fixture record has no key and no path for the route to use"
    digest = record.sha256
    # And the bytes are where a content-addressed store says they are.
    assert (objects_root / "original" / digest[:2] / f"{digest}.pdf").is_file()


@requires_renderer
def test_a_path_on_the_record_is_a_fallback_not_the_primary(
    client, store, objects_root, monkeypatch, tmp_path
):
    """A deployment that keeps uploads in a plain directory still renders.

    The object store is the designed home and the digest lookup is preferred,
    but the ``original`` layer is not yet written by anything in this
    repository, so a record carrying an explicit path is the difference between
    the feature working and reporting ``original_unavailable`` on a deployment
    that plainly has the bytes on disk.
    """
    payload = make_pdf_bytes(SYNTHETIC_RECTS)
    loose = tmp_path / "loose"
    loose.mkdir()
    loose_pdf = loose / "paper.pdf"
    loose_pdf.write_bytes(payload)

    # Point the object store at a root with no ``original`` directory, so only
    # the path fallback can succeed.
    monkeypatch.setattr(config, "PAGE_RENDER_ORIGINALS_DIR", str(tmp_path / "no-objects-here"))

    store.metadata.upsert_document(
        DocumentRecord(
            document_id="doc_by_path",
            document=ParsedDocument(
                document_id="doc_by_path",
                source_filename="paper.pdf",
                media_type="application/pdf",
            ),
            source_filename="paper.pdf",
            media_type="application/pdf",
            metadata={"original_path": str(loose_pdf)},
        )
    )

    response = client.get("/v1/documents/doc_by_path/pages/1/image")

    assert response.status_code == 200, response.text
    assert png_size(response.content)[0] == 600


# --------------------------------------------------------------------------
# The descriptor the client reads
# --------------------------------------------------------------------------


@requires_renderer
def test_the_descriptor_keeps_its_shape_and_fills_in_the_url(client, store, objects_root):
    """The shape a client is being written against must not move.

    ``kind``, the field naming the page, ``document_id``, ``rendered``,
    ``renderer`` and ``note`` all stay, and ``rendered`` stays **false** even
    now that a rasteriser exists — because the bytes of an image are not in a
    JSON payload, and a client that read ``rendered: true`` as "the image is
    here" would show an empty panel. What changed is that ``renderer`` now names
    a real backend and ``url`` says where the image is, which is what lets the
    front end stop asking whether to bother.
    """
    seed_synthetic(store, objects_root)
    seed_chunk(
        store,
        "doc_synth",
        "chunk_ok",
        SourceLocator(kind="pdf_page", page=1, bbox=BoundingBox(200.0, 40.0, 260.0, 90.0)),
    )

    payload = client.get("/v1/documents/doc_synth/evidence/chunk_ok").json()
    page_image = payload["visual_targeting"]["page_image"]

    for key in ("kind", "page", "document_id", "rendered", "renderer", "note"):
        assert key in page_image, key
    assert page_image["kind"] == "pdf_page"
    assert page_image["page"] == 1
    assert page_image["rendered"] is False
    assert page_image["renderer"] == "pypdfium2"
    assert page_image["url"] == "/v1/documents/doc_synth/pages/1/image"
    assert page_image["scale"] == 2.0
    # The transform, named. A client that has to guess which way is up is a
    # client that will eventually guess wrong.
    assert page_image["coordinate_space"] == "pdf_user_space_points"
    assert page_image["origin"] == "top-left"
    assert page_image["y_axis"] == "down"
    assert page_image["note"] and "渲染" in page_image["note"]


@requires_renderer
def test_a_slide_still_has_no_raster(client, store, objects_root):
    """A ``.pptx`` is not a PDF, and the descriptor must not pretend otherwise.

    The renderer reads PDFs. A slide descriptor that claimed ``renderer:
    "pypdfium2"`` and handed out a ``/pages/{n}/image`` URL would send the front
    end to a route that answers 415 for a document it has already told the user
    it can show.
    """
    document = ParsedDocument(
        document_id="doc_deck",
        source_filename="talk.pptx",
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        page_count=12,
        elements=[
            Element(
                element_id="doc_deck_s1",
                role="text",
                text="Slide one.",
                locator=SourceLocator(kind="slide", slide=1),
            )
        ],
    )
    store.metadata.upsert_document(
        DocumentRecord(
            document_id="doc_deck",
            document=document,
            source_filename="talk.pptx",
            media_type=document.media_type,
        )
    )
    seed_chunk(store, "doc_deck", "chunk_slide", SourceLocator(kind="slide", slide=1))

    payload = client.get("/v1/documents/doc_deck/evidence/chunk_slide").json()
    page_image = payload["visual_targeting"]["page_image"]

    assert page_image["kind"] == "slide"
    assert page_image["slide"] == 1
    assert page_image["url"] is None
    assert page_image["scale"] is None
    # The transform is null rather than merely absent, so a client cannot read
    # a populated "which way is up" triple and apply a y-flip to a paragraph.
    assert page_image["coordinate_space"] is None
    assert page_image["origin"] is None
    assert page_image["y_axis"] is None
