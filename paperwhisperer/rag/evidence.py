"""Turn a stored locator into a view a reader can actually look at.

This module is task T1.5's whole reason to exist, and it exists apart from
the HTTP route for two reasons that both point the same way. A test wants to
assert what a citation resolves to without standing up a server, and T3.3
(the full evidence viewer) wants to reuse the exact same resolution to build a
richer client-side panel. If the logic lived in the route, both would have to
re-derive it and the two would drift — which is how a citation comes to point
at one place in the API and another in the browser. So the route is a thin
lookup and everything with a decision in it is here.

The contract this module implements is narrow and absolute. A citation is a
promise that clicking it shows the passage that backs the claim, and that
promise has exactly two acceptable answers: it resolves to the right thing,
or it says plainly that it cannot. There is no third option. A 500, a blank
panel and a highlight drawn over the wrong lines are the *same* defect — the
user believes something they should not — so every branch below is written to
make the third outcome unreachable rather than merely unlikely.

That constraint shows up concretely in three places:

**A relative box is refused, not drawn.** :meth:`BoundingBox.to_query` raises
for one, and this module never catches that into a fragment. A mirrored box
has the right size, the right ordering and the right page, so nothing about
it *looks* wrong — the only thing that distinguishes it is where it came
from, and that knowledge is gone by now. A pdf_page whose box is relative
therefore reports that it has no highlight capability and still names the
page, which is the most a reader can honestly be given.

**The bbox comes from the evidence's own locator and nowhere else.** Not from
the nearest element, not from a spatial search of the document, not from a
sibling chunk that happens to share a page. The evidence names a rectangle;
that rectangle is what gets drawn. A lookup that went looking for a "good
enough" box is exactly the lookup that eventually returns another element's
box, and the highlight would then be wrong while remaining entirely
plausible.

**A version mismatch is surfaced, never absorbed.** An evidence carries the
``index_version`` it was built under, and the document record carries the one
it is being read as. When they differ, the geometry the locator names was
computed against a different parse and the numbers may no longer mean what
they meant. The response says so in a dedicated field rather than quietly
serving the older version, because a stale highlight is a correct-looking
wrong answer and a labelled one is a warning a user can act on.

The snippet is likewise not a rendering. It is a contiguous slice of the
chunk's ``verbatim_text`` — the body without the generated heading prefix,
which is the only text :mod:`paperwhisperer.rag.citations` will verify a
direct quote against — reported with the exact ``[start, end)`` offsets that
produced it. Nothing is re-cleaned, re-wrapped or whitespace-collapsed on the
way out, because every one of those transformations shifts the characters a
citation was verified against, and a snippet that reads nicely while pointing
at the wrong characters is a worse bug than an ugly one. A client that needs
to reconcile against a quote can slice the stored chunk itself and get the
same string back.

**T3.3's addition: the ``view`` block.** The locator decides *where* a
citation points; the view decides *how the reader gets there*, and the two
problems are genuinely different. The reason the block exists separately rather
than as fields on the resolution is worth stating here: ``visual_targeting`` is
a verdict ("can this be drawn, and if not why not") and the view is the drawing.
Merging them would mean the verdict about a missing highlight also carried the
coordinates of the surrounding page, and a client could not tell which fields
it was allowed to draw from.

**The page image, and why the view is still a schematic.** With the
``render`` extra installed the server can rasterise a real page
(:mod:`paperwhisperer.api.routes_evidence` renders it;
:func:`page_image_geometry` states the transform a client needs to place a box
on it). The ``view`` block deliberately did **not** become the raster. It is a
machine-readable companion that says where every element on the page is, in
coordinates, and it stays useful in three situations the image is not: a client
drawing a highlight before the image finishes loading, a reader who wants to
know *which* element was cited rather than merely where it sits, and any
build where the optional extra is absent. Collapsing it into the image would
have traded a text-assertable contract for a picture, and the failure this
whole module exists to prevent is a highlight that looks right and points at
the wrong lines — which is exactly what you get when the only record of where
something is, is a rectangle drawn by code nobody here can assert on.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from html import escape
from typing import Any

from paperwhisperer.infra.stores.interfaces import DocumentRecord
from paperwhisperer.ingestion.schema import (
    BoundingBox,
    Chunk,
    Element,
    Evidence,
    ParsedDocument,
    SourceLocator,
    Table,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# The vocabulary of "this cannot be pointed at"
# --------------------------------------------------------------------------

#: Why a locator produced no visual target. These are machine-readable because
#: a UI branches on them — the degraded state is different for "the parser
#: lost the page" than for "this box would be mirrored" — and they are stable
#: strings because a client is allowed to key off them.
REASON_UNKNOWN_LOCATOR = "unknown_locator"
REASON_RELATIVE_BBOX = "relative_bbox"
REASON_MISSING_BBOX = "missing_bbox"
REASON_MISSING_PAGE = "missing_page"
REASON_INCOMPLETE_LOCATOR = "incomplete_locator"
#: Added by T3.3. A rectangle whose far edge sits *before* its near edge is not
#: a rectangle in the coordinate system this class documents -- it is a
#: converter that emitted the corners the wrong way round. :meth:`width` and
#: :meth:`height` take absolute values, so the box still has a positive extent
#: and every "is this usable" check that looks only at size passes; what it
#: would draw is a rectangle anchored at the far edge, mirrored horizontally or
#: vertically. That is the same class of defect as a relative box -- a
#: plausible-looking highlight over the wrong lines -- so it is refused here
#: rather than at the moment somebody notices.
REASON_MALFORMED_BBOX = "malformed_bbox"

#: The human face of each reason, in the same Chinese the rest of the API
#: speaks. Kept next to the codes so a code and its sentence cannot drift
#: apart, and written as sentences rather than labels because this is the text
#: a user reads when a citation cannot be opened.
REASON_MESSAGES: dict[str, str] = {
    REASON_UNKNOWN_LOCATOR: "该证据的定位信息为 unknown，没有可用的视觉定位能力。",
    REASON_RELATIVE_BBOX: (
        "该证据的边界框是相对坐标，映射到页面后会高亮到错误的区域，因此不提供高亮框。"
    ),
    REASON_MISSING_BBOX: "该证据只有页码、没有边界框，无法定位到页面上的具体位置。",
    REASON_MISSING_PAGE: "该证据没有页码，无法定位到页面。",
    REASON_MALFORMED_BBOX: (
        "该证据的边界框坐标顺序有误（远边在近边之前），"
        "映射到页面后会镜像到错误的方向，因此不提供高亮框。"
    ),
    REASON_INCOMPLETE_LOCATOR: "该证据的定位信息不完整，没有可用的视觉定位能力。",
}

#: What the reader *can* do, when the locator does resolve. Named per kind so
#: a client can tell "highlight this box on page 7" from "jump to slide 3" —
#: the two need different UI, and collapsing them into one boolean is how a
#: slide number ends up being drawn as a rectangle.
CAP_PDF_PAGE_HIGHLIGHT = "pdf_page_highlight"
CAP_SLIDE_JUMP = "slide_jump"
CAP_SHEET_CELLS = "sheet_cells"
CAP_ELEMENT_JUMP = "element_jump"
CAP_OFFSET_JUMP = "offset_jump"

# --------------------------------------------------------------------------
# The view vocabulary (T3.3)
# --------------------------------------------------------------------------

#: The three views T3.3 has to deliver, named per format so a client branches
#: on the shape it is about to lay out rather than on the file extension it
#: thinks it is looking at. There is deliberately no "generic" member: an
#: unknown kind returns ``None`` (no view at all) rather than a view that
#: guesses, for the same reason :func:`resolve_locator` ends in a definite
#: refusal instead of an empty success.
VIEW_PAGE_MAP = "page_map"
VIEW_SLIDE = "slide_view"
VIEW_ELEMENT = "element_view"

#: The view could not name the cited block inside the region it laid out.
#: This is *not* one of the ``REASON_*`` codes above and is never written into
#: ``visual_targeting``: it says the region resolved fine and the target is
#: missing from it, which is a different statement from "this locator cannot
#: be pointed at". Overloading the reason vocabulary would make a client that
#: keys on ``reason_code`` render a locator failure for what is really a
#: content-lookup miss.
VIEW_TARGET_MISSING = "target_not_in_region"

VIEW_TARGET_MISSING_MESSAGE = (
    "该证据的定位指向了这一页/这一段，但文档的解析结果里没有对应的元素，"
    "因此无法在视图中标出它的位置。"
)

#: Hard caps on the view. Every one of them **states** that it bit, because
#: "silent truncation on the eval corpus: 0" is a release gate and a view that
#: quietly dropped 300 blocks to fit a payload would be the same defect wearing
#: a smaller costume. ``blocks_truncated`` / ``preview_truncated`` travel with
#: the payload for exactly that reason.
VIEW_MAX_BLOCKS = 400
VIEW_BLOCK_PREVIEW_CHARS = 240
VIEW_TARGET_TEXT_CHARS = 4000
VIEW_CONTEXT_TEXT_CHARS = 300
VIEW_ELEMENT_CONTEXT_BEFORE = 2
VIEW_ELEMENT_CONTEXT_AFTER = 2

#: What a rebuilt page map is and is not. The text is user-facing, so it is a
#: sentence a reader reads: the boxes are the *parse's* coordinates, laid out
#: in the same PDF user space the box came from, which makes the highlight's
#: position relative to the surrounding blocks correct — and which is not the
#: same thing as a picture of the page. Saying so is the difference between a
#: schematic and a rendering, and a schematic that pretends to be a rendering
#: is a viewer that teaches a reader to trust pixels it does not have.
VIEW_PAGE_MAP_NOTE = (
    "这是按解析得到的元素坐标重建的页内布局，不是原始 PDF 的渲染图："
    "方块与高亮框使用同一套 PDF 用户空间坐标，因此彼此的相对位置与页面一致。"
)
VIEW_SLIDE_NOTE = (
    "幻灯片没有像素坐标，因此这里列出该 slide 上的全部文本元素，并标出被引用的那个。"
)
VIEW_ELEMENT_NOTE = (
    "DOCX 没有页码坐标，因此按段落序号给出目标段落及其前后文，"
    "标题路径是读者在文档里找到它的依据。"
)

#: The snippet is a window on text the chunk already bounds, so the cap is a
#: runaway guard rather than a design target. It is high enough that a normal
#: chunk never reaches it, which matters: every character of a chunk is text a
#: citation may have been verified against, and a cap low enough to bite would
#: be a cap quietly truncating evidence.
SNIPPET_MAX_CHARS = 8000

#: ``R2C3`` / ``R2C3:R4C5`` — the notation :mod:`paperwhisperer.rag.citations`
#: uses when it reports which table cell a quote landed in. Parsed here so the
#: cells it names can be marked in the rendered HTML.
_CELL_ADDRESS = re.compile(r"^R(\d+)C(\d+)$")


# --------------------------------------------------------------------------
# Values
# --------------------------------------------------------------------------


@dataclass(slots=True)
class LocatorResolution:
    """The answer to "where does this locator point, and can we show it?".

    ``available`` is deliberately narrow: it means *the cited passage itself*
    can be pointed at — for a PDF that is a page **and** an absolute box, for
    a slide or sheet or paragraph or character range that one concrete
    position. It is not "we found something nearby". A ``pdf_page`` with a
    page but no box reports ``available=False`` with ``missing_bbox`` while
    still handing back the page, because the page is a true fact about where
    the evidence came from and dropping it would be a small lie of omission;
    what it does not do is let a client draw a highlight it cannot justify.

    The asymmetry with a *relative* box is the point. A missing box is an
    absence — there is simply nothing to draw, and saying so is honest. A
    relative box is a poisoned value: it carries a real rectangle that would
    land in the wrong place while looking perfect. Both report
    ``available=False``; only the second one is a trap, and the reason code is
    what tells a client (and a human reading the panel) which happened.
    """

    kind: str
    available: bool
    capability: str = ""
    reason_code: str = ""
    reason: str = ""
    target: dict[str, Any] | None = None
    page_image: dict[str, Any] | None = None
    highlight: str = ""
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        """The wire shape, and the shape T3.3 will read.

        ``highlight`` is present-and-null rather than absent when there is
        none, so a client reading this as data can tell "there is no box"
        from "this build does not know about boxes". The distinction is only
        theoretical today, and it is exactly the kind of theoretical
        distinction that is expensive to add once a client has shipped a
        default for the missing-key case.
        """
        return {
            "available": self.available,
            "capability": self.capability or None,
            "highlight": self.highlight or None,
            "reason_code": self.reason_code or None,
            "reason": self.reason or None,
            "target": self.target,
            "page_image": self.page_image,
        }


@dataclass(slots=True)
class ResolvedEvidence:
    """A citation, resolved: the passage, where it is, and what can be drawn.

    ``warnings`` carries the things a reader must be told before trusting the
    panel — a version mismatch above all. They are warnings and not errors
    because the passage is still real even when its coordinates are stale, and
    an API that 409s on a stale index would hide evidence rather than
    qualifying it, which is the other way a user ends up believing something
    they should not.
    """

    evidence_id: str
    document_id: str
    chunk_id: str
    citation_label: str
    locator: dict[str, Any]
    snippet: dict[str, Any]
    #: Always present. This block is where the "or explicitly marked" half of
    #: the release gate lives, and it is named so that a client looking for
    #: "can I point at this?" finds one boolean rather than inferring it from
    #: the presence or absence of a highlight fragment.
    visual_targeting: dict[str, Any] = field(default_factory=dict)
    element_ids: list[str] = field(default_factory=list)
    table: dict[str, Any] | None = None
    document_version: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    #: T3.3. The laid-out region a client draws, or ``None`` for a format with
    #: nothing to lay out. It is a *sibling* of ``visual_targeting`` and not
    #: part of it: the targeting block is the verdict and this is the drawing,
    #: and the difference matters because a page map can be perfectly useful
    #: while carrying no highlight at all — that is the relative-box case, and
    #: it is the whole point of requirement 2.
    view: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "citation_label": self.citation_label,
            "locator": self.locator,
            "snippet": self.snippet,
            "visual_targeting": self.visual_targeting,
            "view": self.view,
            "element_ids": list(self.element_ids),
            "table": self.table,
            "document_version": self.document_version,
            "warnings": list(self.warnings),
        }


# --------------------------------------------------------------------------
# Locator resolution — total over every kind
# --------------------------------------------------------------------------


def page_renderer_name() -> str:
    """The name of the page rasteriser this process can actually use, or ``"none"``.

    Deliberately a *name*, and deliberately probed with
    :func:`importlib.util.find_spec` rather than by importing. Importing
    ``pypdfium2`` at module scope to answer a capability question would make
    this module — which every citation click and every retrieval leg pulls in —
    depend on an optional rendering extra, and the personal tier must run
    without it. ``find_spec`` asks the import machinery whether a package is
    *installed* without executing it, so a personal install without the extra
    pays a few microseconds of path search and nothing else.

    Every module the render path needs is probed, not just the rasteriser.
    ``pypdfium2`` hands back a raw BGR buffer and the PNG is encoded through
    Pillow, so an environment that somehow has the rasteriser and not the
    encoder cannot render either — and reporting a renderer that will then
    raise :class:`ModuleNotFoundError` on the first citation click is precisely
    the "payload says yes, route says 501" split this function exists to
    prevent.

    The function is the single place that answers "can this build render a
    page?". :mod:`paperwhisperer.api.routes_evidence` asks it before doing any
    work, and :func:`_page_image` puts the answer in the payload, so the
    capability the client reads and the capability the route acts on cannot be
    two opinions. That matters more than usual here: a payload that says a
    renderer exists while the route answers 501 produces a client that shows an
    empty panel and calls it a loading failure.
    """
    for module in PAGE_RENDERER_MODULES:
        try:
            if importlib.util.find_spec(module) is None:
                return "none"
        except (ImportError, ValueError):  # pragma: no cover - broken install
            return "none"
    return PAGE_RENDERER_MODULE


#: The name reported to clients when a render is possible. It is the rasteriser
#: rather than the encoder because the rasteriser is the part a reader of a
#: dependency list will recognise, and because it is the part whose absence is
#: the interesting fact.
PAGE_RENDERER_MODULE = "pypdfium2"

#: Everything the render path imports. See :func:`page_renderer_name` for why
#: this is a list and not one name. Kept as data so the 501 path, the payload
#: and the test that simulates absence all reference the same strings — a test
#: that blocked ``sys.modules["pypdfium2"]`` while the probe looked up some
#: other name would prove nothing at all.
PAGE_RENDERER_MODULES: tuple[str, ...] = ("pypdfium2", "PIL")

#: What ``view`` shows when a real rasteriser *is* available: the reader gets the
#: page itself, and the honest remaining caveat is not "this is not a picture"
#: but "this picture is of the page as stored, so a page carrying ``/Rotate`` is
#: shown in its stored orientation".
VIEW_PAGE_RASTER_NOTE = (
    "这是原始 PDF 页面的真实渲染图：高亮框使用与解析结果相同的 PDF 用户空间坐标"
    "（原点左下、y 向上），因此可以直接叠加在页图上定位。"
)

#: What ``view`` says when there is still no rasteriser. Kept verbatim from the
#: build that had none, so a deployment that upgrades the extra does not have to
#: recognise a second "no" wording.
VIEW_PAGE_SCHEMATIC_ONLY_NOTE = (
    "这是按解析得到的元素坐标重建的页内布局，不是原始 PDF 的渲染图："
    "方块与高亮框使用同一套 PDF 用户空间坐标，因此彼此的相对位置与页面一致。"
)


def _page_image(
    kind: str, field_name: str, value: Any, document_id: str
) -> dict[str, Any] | None:
    """A descriptor for a page/slide image, never the image itself.

    The shape is unchanged from the build that had no renderer, because a client
    is being written against it right now: ``kind``, the field naming the page
    or slide, ``document_id``, ``rendered``, ``renderer``, ``note``. What
    changed is what two of those fields *mean*, which is the whole point of
    T3.3's remaining criterion:

    * ``renderer`` is now :func:`page_renderer_name` — the real backend's name
      when one is importable, and the literal ``"none"`` when it is not. A
      client reads this to decide whether it is worth asking for the image at
      all, so it has to be the truth about the server and not the truth about
      the build that first wrote this key.
    * ``rendered`` stays **false**, and this is worth being explicit about
      because it reads like an oversight. The key means "the bytes of a rendered
      image are in this payload", and they are not and never will be: this is a
      JSON document and the image is a few hundred kilobytes of PNG. The
      descriptor's job is to say *where the image lives and whether it can be
      produced*, and the client fetches ``url`` to get it. A client that
      flipped its logic to "rendered is true, so do not look for a url" would
      be correct about the payload and wrong about the world.

    The new fields exist because the client cannot place a highlight from
    ``kind`` and a page number alone, and the two numbers it needs — the scale
    and the viewport — are properties of the render, not of the parse:

    ``url``
        The image, with the scale baked into the query string. Baked rather
        than defaulted so the response's geometry describes the bytes the
        client actually received; a client that asked for scale 3 and drew with
        the default's geometry would land the highlight three times too small,
        which is the exact failure mode this descriptor exists to prevent.
    ``scale``
        The device pixels per PDF point the render is asked for. The client must
        not compute it: the server clamps it, and the clamped value is the one
        in force.
    ``coordinate_space`` / ``origin`` / ``y_axis``
        The transform, named. See :func:`page_image_geometry` for the numbers
        and the arithmetic; these three say which way is up, which is the part
        a client gets wrong most often.

    Slides keep the old answer even when a rasteriser is present, and that is
    not an oversight either. This renderer reads PDFs; a ``.pptx`` has no
    raster path through it, and claiming otherwise would be the same
    "the key means the image is here" confusion in the other direction.
    """
    if value is None:
        return None

    renderer = page_renderer_name()
    can_render = renderer != "none" and kind == "pdf_page"
    return {
        "kind": kind,
        field_name: value,
        "document_id": document_id,
        "rendered": False,
        "renderer": renderer,
        "url": (
            f"/v1/documents/{document_id}/pages/{int(value)}/image"
            if can_render
            else None
        ),
        "scale": default_page_render_scale() if can_render else None,
        # The transform, and it is null rather than merely absent whenever
        # there is no raster. A client reading a populated
        # ``coordinate_space``/``origin``/``y_axis`` triple has been told which
        # way is up, and a slide has no up — handing it a PDF convention it
        # cannot use is how a client ends up applying a y-flip to a paragraph
        # index. Null is a value the client has to handle; a plausible-looking
        # one it will not.
        "coordinate_space": "pdf_user_space_points" if can_render else None,
        "origin": "top-left" if can_render else None,
        "y_axis": "down" if can_render else None,
        "note": VIEW_PAGE_RASTER_NOTE if can_render else PAGE_RENDER_UNAVAILABLE_NOTE,
    }


#: Said to the reader when this build genuinely cannot rasterise. The route's
#: 501 body and this note are the same sentence on purpose: a client that shows
#: the payload's note above a schematic and gets a different sentence from the
#: route when it does eventually ask is two explanations for one fact, and the
#: reader is the one who has to reconcile them.
PAGE_RENDER_UNAVAILABLE_NOTE = (
    "本版本不提供页图/幻灯片光栅图：运行环境没有安装可选的 PDF 渲染依赖"
    f"（{PAGE_RENDERER_MODULE}）。面板改用解析得到的元素坐标重建页内布局来定位，"
    "并在视图上标明它不是原始渲染图。"
)

#: Device pixels per PDF point asked for when nothing else is specified.
#:
#: 2.0 is 144 DPI, which is the point at which body text in a two-column paper
#: is legible in a panel about 600 CSS pixels wide without the reader zooming,
#: and it is also the number the ``pypdfium2`` documentation itself uses as the
#: example, so an operator comparing against upstream docs is comparing like
#: with like. It is a fallback, not the source of truth: the deployment's
#: ``PAGE_RENDER_SCALE`` wins, and the route echoes the value it actually used
#: back to the client, so an operator who raises the scale to serve a retina
#: panel does not also have to edit this file for the descriptor to stop lying.
PAGE_RENDER_SCALE = 2.0


def default_page_render_scale() -> float:
    """The scale this deployment renders at when nothing else is specified."""
    return resolve_page_render_scale(None)


def resolve_page_render_scale(raw: Any) -> float:
    """The scale to render at, given whatever a client asked for.

    One function, because three places need this answer and three copies of a
    clamp is three chances for the descriptor to offer a client a scale the
    route then refuses — the "payload says 3x is fine, the route renders 2x"
    split, which is a highlight drawn a third too small and looks like nothing
    in particular.

    **Clamped, not rejected.** A client that asks for ``3.7`` because it
    computed a device-pixel-ratio it likes has asked a reasonable question and
    deserves a sensible answer, not a 400 it will surface to the user as "the
    server rejected my page". What is *not* allowed is the unbounded request,
    and that is what the ceiling is for: a render allocates an image
    proportional to a number that arrived on the wire, so an A0 poster at
    scale 64 is roughly 256 megapixels and about a gigabyte of raw pixels
    before any compression. The ceiling is high enough that a retina panel
    asking for 3x gets it, and low enough that the worst case is a large but
    survivable image rather than a killed process.

    **A value that is not a number falls back rather than erroring.** The
    parameter is a hint about display density, and the default is always a
    renderable answer, so refusing a bad hint would turn a cosmetic mistake
    into a citation that will not open. ``NaN`` and the infinities get the same
    treatment for the same reason, and each is rejected explicitly: they are
    the values that survive a comparison and then poison every subsequent
    arithmetic operation on the number.

    Settings are read with ``getattr`` for the same reason
    :func:`paperwhisperer.analysis.orchestrator._embedding_index_version` reads
    its settings that way: the configuration block is owned by another task, and
    a module that raised :class:`AttributeError` because a setting had not
    landed yet would take every citation click down with it.
    """
    from paperwhisperer.core import config

    default = _bounded(getattr(config, "PAGE_RENDER_SCALE", ""), PAGE_RENDER_SCALE)
    ceiling = _bounded(
        getattr(config, "PAGE_RENDER_MAX_SCALE", ""), PAGE_RENDER_SCALE_CEILING
    )
    # A ceiling configured *below* the floor would make every value clamp to
    # the ceiling and every clamp pointless, so the floor wins and the ceiling
    # is raised to meet it. An operator who sets 0.1 has not disabled page
    # images — they have configured a range that cannot be satisfied, and
    # honouring it literally would answer every request with an unreadable
    # render rather than with a usable one.
    ceiling = max(ceiling, PAGE_RENDER_SCALE_MIN)
    if raw is None or not str(raw).strip():
        return min(max(default, PAGE_RENDER_SCALE_MIN), ceiling)
    try:
        requested = float(str(raw).strip())
    except (TypeError, ValueError):
        return min(max(default, PAGE_RENDER_SCALE_MIN), ceiling)
    if requested != requested or requested in (float("inf"), float("-inf")):
        return min(max(default, PAGE_RENDER_SCALE_MIN), ceiling)
    return min(max(requested, PAGE_RENDER_SCALE_MIN), ceiling)


def _bounded(value: Any, fallback: float) -> float:
    """``value`` as a finite float, or ``fallback``.

    Separate from :func:`resolve_page_render_scale` so the *configuration*
    reads and the *client request* read are each guarded once. A ``.env`` with
    ``PAGE_RENDER_SCALE=laptop`` must not take a citation click down, and that
    is the same class of refusal as a client sending ``?scale=laptop``.
    """
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return fallback
    if resolved != resolved or resolved in (float("inf"), float("-inf")):
        return fallback
    return resolved


#: Device pixels per PDF point asked for when nothing else is specified, and
#: when configuration does not say.
#:
#: 2.0 is 144 DPI, which is the point at which body text in a two-column paper
#: is legible in a panel about 600 CSS pixels wide without the reader zooming,
#: and it is also the number the ``pypdfium2`` documentation itself uses as the
#: example, so an operator comparing against upstream docs is comparing like
#: with like. It lives here as well as in configuration because this module has
#: to keep importing on a checkout where the setting has not been read yet.
PAGE_RENDER_SCALE = 2.0

#: The floor. Below half a device pixel per point a page is smaller than its own
#: text and a highlight on it is a smear, so a smaller request is answered with
#: the smallest legible render rather than honoured.
PAGE_RENDER_SCALE_MIN = 0.5

#: The ceiling used when configuration does not name one. Same reasoning as
#: :data:`paperwhisperer.core.config.PAGE_RENDER_MAX_SCALE`, restated because
#: the descriptor and the route have to agree and a client reading the
#: descriptor must not be offered something the route would refuse.
PAGE_RENDER_SCALE_CEILING = 4.0


def page_image_geometry(
    *,
    page_width_pt: float,
    page_height_pt: float,
    image_width_px: int,
    image_height_px: int,
    rotation: int,
    scale: float,
) -> dict[str, Any]:
    """The numbers a client needs to place a box, and nothing else.

    **The contract, exactly.** The ``bbox`` in a payload is in PDF user space:
    points, origin bottom-left, ``y`` increasing upward. The image this function
    describes is a raster: pixels, origin top-left, ``y`` increasing downward.
    Those two facts are the whole of the mismatch, and both differences are
    accounted for here — the flip in ``y``, and the scale from points to device
    pixels — so a client that reads these numbers and applies the two operations
    below places a rectangle correctly without knowing anything else about PDF.

    For the overwhelmingly common case, ``rotation == 0``::

        left   = x0 * scale
        top    = (page_height_pt - y1) * scale
        width  = (x1 - x0) * scale
        height = (y1 - y0) * scale

    and the box is done. The ``rotation != 0`` branch is not decoration: a
    ``/Rotate 90`` page is a landscape scan of a portrait original, and pdfium
    reports its *rotated* size from ``get_size()`` while every coordinate in
    this project — every ``BoundingBox``, every element rectangle, every
    ``xywh=`` fragment — is in the *unrotated* space. A client that applied the
    formula above to such a page would put the highlight on the right lines of
    the wrong page: rotated, and often outside the image entirely.

    **Why the route renders unrotated instead of making clients handle this.**
    :meth:`pypdfium2.PdfPage.set_rotation` can neutralise a page's ``/Rotate``
    before rendering, which makes the ``rotation == 0`` formula correct for
    every document and reduces the client contract to two operations that never
    have a special case. The cost is that a rotated page is *displayed* in its
    stored orientation rather than the orientation a reader would turn their
    head for — which is why ``rotated`` and the true rotation are reported
    rather than hidden: a sideways page is visible and explainable, a
    highlight on the wrong lines is neither. This is the same trade the rest of
    the module makes, in the same direction: prefer a visible, honest
    imperfection to an invisible, plausible-looking wrong answer.

    ``scale_x`` and ``scale_y`` are reported separately rather than one
    ``scale`` because pdfium's bitmap dimensions are not exactly
    ``round(page_pt * scale)``: measured across the eval corpus, a 595.276pt
    page at scale 1.0 renders 596px wide, and a 595pt page at 1.5 renders 893
    where the exact product is 892.5. A client multiplying by the requested
    scale would therefore be off by up to a pixel at the edge, and "up to a
    pixel" is fine for a box and not fine for a test. Using the achieved ratio
    makes the placement exact against the bytes that were actually produced,
    which is the only definition of correct that a client can act on.

    **The ratios are computed from the *rounded* page dimensions, on purpose.**
    An earlier version derived them from pdfium's full-precision ``get_size()``
    and then published the page box rounded to two decimals, and the two did
    not agree: for the 595.276pt page above, the reported ``scale_x`` was
    1.001216 while ``image_width_px / page_width_pt`` read off the same payload
    was 1.001210. The difference is fourteen thousandths of a pixel — invisible,
    and exactly the kind of thing that is nobody's fault and everybody's
    problem. A client that used the reported ratio and a client that recomputed
    it from the two published numbers would place the same box in two different
    places, and a payload that can be read two ways is a payload whose geometry
    nobody can check. So the published numbers are the ones everything is
    derived from, and the relationship ``scale_x == image_width_px /
    page_width_pt`` holds on the wire as well as in memory.
    """
    page_width = _round(float(page_width_pt))
    page_height = _round(float(page_height_pt))
    scale_x = image_width_px / page_width if page_width else 0.0
    scale_y = image_height_px / page_height if page_height else 0.0
    return {
        "page_width_pt": page_width,
        "page_height_pt": page_height,
        "image_width_px": int(image_width_px),
        "image_height_px": int(image_height_px),
        "scale_x": round(scale_x, 6),
        "scale_y": round(scale_y, 6),
        "scale_requested": round(float(scale), 6),
        # Always 0: the route renders unrotated so this formula never has a
        # special case. Reported because a client that hard-codes nothing and
        # reads nothing should still be able to see that the value it relies on
        # is genuinely the value in force.
        "rotation": 0,
        "source_rotation": int(rotation) % 360,
        "coordinate_space": "pdf_user_space_points",
        "origin": "top-left",
        "y_axis": "down",
        "flip_y": True,
    }


def _unavailable(kind: str, reason_code: str, **parts: Any) -> LocatorResolution:
    """A resolution that names what is missing, with the honest parts kept."""
    return LocatorResolution(
        kind=kind,
        available=False,
        reason_code=reason_code,
        reason=REASON_MESSAGES[reason_code],
        label=parts.pop("label", ""),
        **parts,
    )


def _box_is_ordered(bbox: BoundingBox | None) -> bool:
    """Whether a rectangle's corners are in the order the schema documents.

    ``BoundingBox.width`` and ``height`` are ``abs`` differences, so a box
    whose corners arrived swapped still reports a positive size and passes
    every check that looks at extent. It does not pass *this* one, and that is
    the point: ``x0, y0`` is the near corner in PDF user space (y upward from
    the bottom-left), and a rectangle anchored anywhere else draws mirrored in
    one axis. Off-page (``relative``) is a separate and earlier rejection — this
    is about ordering, not about provenance.
    """
    if bbox is None or bbox.relative:
        return False
    values = (bbox.x0, bbox.y0, bbox.x1, bbox.y1)
    if not all(isinstance(value, (int, float)) for value in values):
        return False
    if any(value != value or value in (float("inf"), float("-inf")) for value in values):
        # NaN compares unequal to itself, so `value != value` is the NaN test.
        # Infinities are caught explicitly because they are the other value
        # that survives arithmetic and produces a rectangle of infinite size.
        return False
    return bbox.x1 >= bbox.x0 and bbox.y1 >= bbox.y0


def _resolve_pdf_page(
    locator: SourceLocator, document_id: str
) -> LocatorResolution:
    """A page, a box, or an explicit statement of which one is missing.

    The page is recovered independently of the box, because a page is a
    complete fact about where an element came from and a reader who cannot
    get a highlight can still be shown the right page. The box is only
    offered when it is absolute, and the offer is the ``to_query()`` call:
    its refusal for a relative box is the guard, and it is allowed to
    propagate as a last line of defence even though the flag is checked
    first, because a guard that can be bypassed by a future edit to this
    function is not a guard.
    """
    label = Evidence(locator=locator).citation_label
    page_image = _page_image("pdf_page", "page", locator.page, document_id)

    if locator.page is None:
        # A box without a page names a region of nothing. Refusing here
        # rather than defaulting to page 1 is deliberate: page 1 is a real
        # page of a real document, and pointing a reader at it because a
        # number was missing is precisely the wrong-highlight failure.
        return _unavailable("pdf_page", REASON_MISSING_PAGE, label=label)

    target: dict[str, Any] = {"kind": "pdf_page", "page": locator.page, "page_label": label}

    if locator.bbox is None:
        return _unavailable(
            "pdf_page", REASON_MISSING_BBOX, target=target, page_image=page_image, label=label
        )
    if locator.bbox.relative:
        return _unavailable(
            "pdf_page", REASON_RELATIVE_BBOX, target=target, page_image=page_image, label=label
        )
    if not _box_is_ordered(locator.bbox):
        return _unavailable(
            "pdf_page", REASON_MALFORMED_BBOX, target=target, page_image=page_image, label=label
        )

    return LocatorResolution(
        kind="pdf_page",
        available=True,
        capability=CAP_PDF_PAGE_HIGHLIGHT,
        target=target,
        page_image=page_image,
        highlight=locator.bbox.to_query(),
        label=label,
    )


def _resolve_slide(locator: SourceLocator, document_id: str) -> LocatorResolution:
    """A slide number is a complete target: there is nothing finer to have."""
    label = Evidence(locator=locator).citation_label
    page_image = _page_image("slide", "slide", locator.slide, document_id)
    if locator.slide is None:
        return _unavailable("slide", REASON_INCOMPLETE_LOCATOR, label=label)
    return LocatorResolution(
        kind="slide",
        available=True,
        capability=CAP_SLIDE_JUMP,
        target={"kind": "slide", "slide": locator.slide, "page_label": label},
        page_image=page_image,
        label=label,
    )


def _resolve_sheet_range(locator: SourceLocator) -> LocatorResolution:
    """A sheet is the target; a cell range narrows it when the parser knew one."""
    label = Evidence(locator=locator).citation_label
    if not locator.sheet:
        return _unavailable("sheet_range", REASON_INCOMPLETE_LOCATOR, label=label)
    return LocatorResolution(
        kind="sheet_range",
        available=True,
        capability=CAP_SHEET_CELLS,
        target={
            "kind": "sheet_range",
            "sheet": locator.sheet,
            "cell_range": locator.cell_range or None,
            "page_label": label,
        },
        label=label,
    )


def _resolve_docx_element(locator: SourceLocator) -> LocatorResolution:
    """A paragraph index, with the heading path the reader recognises it by.

    DOCX has no page grid in this schema, so there is no image descriptor to
    offer — the target is the paragraph and the heading path is what makes it
    findable in a document that has no page numbers on screen.
    """
    label = Evidence(locator=locator).citation_label
    if locator.paragraph is None:
        return _unavailable("docx_element", REASON_INCOMPLETE_LOCATOR, label=label)
    return LocatorResolution(
        kind="docx_element",
        available=True,
        capability=CAP_ELEMENT_JUMP,
        target={
            "kind": "docx_element",
            "paragraph": locator.paragraph,
            "element": locator.element,
            "heading_path": list(locator.heading_path),
            "page_label": label,
        },
        label=label,
    )


def _resolve_txt_offset(locator: SourceLocator) -> LocatorResolution:
    """A character range into the document's full text.

    The offsets are reported exactly as stored and never nudged. They index
    into the text the plain-text adapters produced, and any adjustment here
    — a "helpful" ±1 for an off-by-one — would land the reader one character
    away from the sentence, which for a quote is the difference between the
    paper's words and ours.
    """
    label = Evidence(locator=locator).citation_label
    if locator.offset_start is None:
        return _unavailable("txt_offset", REASON_INCOMPLETE_LOCATOR, label=label)
    return LocatorResolution(
        kind="txt_offset",
        available=True,
        capability=CAP_OFFSET_JUMP,
        target={
            "kind": "txt_offset",
            "offset_start": locator.offset_start,
            "offset_end": locator.offset_end,
            "page_label": label,
        },
        label=label,
    )


def resolve_locator(locator: SourceLocator, *, document_id: str = "") -> LocatorResolution:
    """Resolve one locator to a target — total over every kind in the schema.

    Written as an explicit chain rather than a table of handlers because the
    count is small, fixed by the schema, and the *absence* of a branch would
    be the bug: an unhandled kind has to fall through to a definite "no
    visual targeting" answer, never to an empty resolution a client would
    read as success. The final return is that answer, and it is reached for
    any kind this build does not know — which is why it does not raise.
    """
    if locator.kind == "pdf_page":
        return _resolve_pdf_page(locator, document_id)
    if locator.kind == "slide":
        return _resolve_slide(locator, document_id)
    if locator.kind == "sheet_range":
        return _resolve_sheet_range(locator)
    if locator.kind == "docx_element":
        return _resolve_docx_element(locator)
    if locator.kind == "txt_offset":
        return _resolve_txt_offset(locator)
    return _unavailable(locator.kind, REASON_UNKNOWN_LOCATOR)


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------


def _cell_positions(cell_range: str | None) -> list[tuple[int, int]]:
    """Parse ``R2C3`` / ``R2C3:R4C5`` into 1-based ``(row, column)`` pairs.

    Reuses the notation the citation verifier already emits in
    :attr:`~paperwhisperer.rag.citations.CitationVerification.matched_cell`, so
    a cell address that came out of quote verification can be handed
    straight to this view and land on the same cell. An unparseable range
    yields nothing rather than a guess: marking cell (0, 0) because the
    string looked cell-ish is how a highlight lands on a header.
    """
    text = str(cell_range or "").strip()
    if not text:
        return []
    start_text, _, end_text = text.partition(":")
    start_match = _CELL_ADDRESS.match(start_text.strip())
    if start_match is None:
        return []
    first = (int(start_match.group(1)), int(start_match.group(2)))
    if not end_text.strip():
        return [first]
    end_match = _CELL_ADDRESS.match(end_text.strip())
    if end_match is None:
        return [first]
    last = (int(end_match.group(1)), int(end_match.group(2)))
    if last < first:
        # A reversed range would otherwise enumerate an empty set and look
        # like a range that simply covered nothing, which reads as "no cell
        # was cited" rather than "this range is nonsense".
        return []
    return [
        (row, column)
        for row in range(first[0], last[0] + 1)
        for column in range(first[1], last[1] + 1)
    ]


def render_table_html(table: Table, cited: list[tuple[int, int]] | None = None) -> str:
    """Render a table as HTML whose cells a client can address and highlight.

    The header/body split mirrors :meth:`Table.to_html` exactly rather than
    re-deriving it, so this richer render cannot disagree with the canonical
    one about which row is a header — the same reasoning
    :func:`paperwhisperer.rag.citations._table_cell_spans` gives for walking
    the markdown instead of reimplementing it. A second opinion about the
    layout is a second opinion waiting to drift.

    What this adds over :meth:`Table.to_html` is ``data-cell`` on every cell
    and ``data-cited`` on the ones a locator named. Without the first, a
    verifier's ``R2C3`` is a string with nothing to attach to and the viewer
    can only highlight a cell by counting rows; with it, addressing a cell is
    a selector. Cells are addressed by position rather than by a stable id
    because the grid is the only thing that exists — an id minted per render
    would not survive a re-parse, and a highlight bound to a stale id is
    another way to point at nothing.
    """
    marked = set(cited or ())
    parts = ["<table>"]
    if not table.rows:
        parts.append("</table>")
        return "".join(parts)

    head_rows = table.rows[: table.header_rows] or table.rows[:1]
    body_rows = table.rows[table.header_rows :] if table.header_rows else table.rows[1:]

    def _cells(row: list[str], row_index: int, tag: str) -> str:
        out = []
        for column_index, cell in enumerate(row):
            address = (row_index + 1, column_index + 1)
            attributes = f' data-cell="R{address[0]}C{address[1]}"'
            if address in marked:
                attributes += ' data-cited="true"'
            out.append(f"<{tag}{attributes}>{escape(str(cell))}</{tag}>")
        return "".join(out)

    if head_rows:
        parts.append("<thead>")
        for row_index, row in enumerate(head_rows):
            parts.append(f"<tr>{_cells(row, row_index, 'th')}</tr>")
        parts.append("</thead>")
    parts.append("<tbody>")
    body_offset = len(head_rows)
    for row_index, row in enumerate(body_rows):
        parts.append(f"<tr>{_cells(row, row_index + body_offset, 'td')}</tr>")
    parts.extend(["</tbody>", "</table>"])
    return "".join(parts)


def _table_payload(element: Element, locator: SourceLocator) -> dict[str, Any] | None:
    """Describe a table element as data, or ``None`` when it is not a table."""
    table = element.table
    if table is None:
        return None
    cited = _cell_positions(locator.cell_range)
    single = cited[0] if len(cited) == 1 else None
    return {
        "html": render_table_html(table, cited),
        "caption": table.caption,
        "row_count": table.row_count,
        "column_count": table.column_count,
        "header_rows": table.header_rows,
        "element_id": element.element_id,
        # A single cited cell gets the verifier's own notation, so a client
        # holding ``matched_cell`` can match it without a second convention.
        "cited_cell": f"R{single[0]}C{single[1]}" if single else "",
        "cited_cells": [f"R{row}C{column}" for row, column in cited],
        "is_empty_shell": table.is_empty_shell,
    }


# --------------------------------------------------------------------------
# Snippet
# --------------------------------------------------------------------------


def _build_snippet(verbatim_text: str) -> dict[str, Any]:
    """A contiguous window of the stored chunk text, with its exact offsets.

    No cleaning, no re-wrapping, no ellipsis inserted mid-word. The offsets
    are the point: ``verbatim_text[start:end]`` is the snippet, byte for
    byte, so a client holding a character range from quote verification can
    slice the same string and know it is looking at the same characters. Any
    tidying done here would shift those characters, and a quote that verifies
    against one text and is displayed as another is a citation that reports
    itself as verified and is not.

    The cap exists only so a pathological chunk cannot return a multi-megabyte
    body. It sits far above a token-budgeted chunk, so it does not bite in
    practice; when it does, the truncation is a head slice, which keeps the
    offsets valid, and it says so rather than silently shortening.
    """
    text = str(verbatim_text or "")
    end = min(len(text), SNIPPET_MAX_CHARS)
    return {
        "text": text[:end],
        "start": 0,
        "end": end,
        "source": "chunk_verbatim_text",
        "source_id": "",
        "length": len(text),
        "truncated": end < len(text),
    }


# --------------------------------------------------------------------------
# Version provenance
# --------------------------------------------------------------------------


def _version_check(
    chunk: Chunk, record: DocumentRecord | None
) -> tuple[dict[str, Any], str | None]:
    """Compare the evidence's index version against the document's.

    The chunk remembers the index it was built under; the record remembers
    the one the document is being served as. They differ when the document
    was re-parsed or re-chunked after this evidence was written, and then the
    page numbers and boxes in the locator were computed against a document
    that no longer exists in that form. The evidence is still real — the text
    was in the paper — but its coordinates are not to be trusted silently, so
    a mismatch becomes a warning the caller sees rather than a page it draws.

    An evidence carrying no version at all is reported as *unverifiable* rather
    than as matching: an empty string equals an empty string, and calling that
    agreement would let a chunk of unknown provenance pass as confirmed.
    """
    evidence_version = str(chunk.index_version or "")
    document_version = str(record.index_version or "") if record is not None else ""

    payload: dict[str, Any] = {
        "document_version": int(record.version) if record is not None else None,
        "document_index_version": document_version or None,
        "evidence_index_version": evidence_version or None,
        "provenance": "unverifiable",
        "stale": False,
        "message": "",
    }

    if not evidence_version or not document_version:
        payload["message"] = (
            "该证据缺少索引版本信息，无法确认它与当前文档版本一致。"
            if not evidence_version
            else "当前文档没有索引版本信息，无法确认该证据的版本来源。"
        )
        return payload, payload["message"]

    if evidence_version != document_version:
        payload["provenance"] = "mismatched"
        payload["stale"] = True
        payload["message"] = (
            f"该证据基于索引版本 {evidence_version} 生成，"
            f"当前文档的索引版本是 {document_version}；页码与高亮框可能已失效。"
        )
        return payload, payload["message"]

    payload["provenance"] = "matched"
    return payload, None


# --------------------------------------------------------------------------
# The view (T3.3) — putting the reader in the region
# --------------------------------------------------------------------------
#
# Everything below answers one question: *given that the locator resolves, what
# does the client draw?* It is not a renderer, and it is not a second opinion
# about where the citation points. Rasterising the page is the route's job
# (:mod:`paperwhisperer.api.routes_evidence`), and :func:`_page_image` is the
# descriptor that tells a client the raster exists and where to fetch it.
#
# What lives here instead is the parse's own geometry, and it lives here rather
# than being replaced by the raster because it answers a question the raster
# cannot. Every element on a page carries the same PDF user-space rectangle the
# cited box is expressed in, so laying those rectangles out inside one frame
# reproduces the page's *layout* with correct relative positions, and it does so
# in numbers this repository can assert on. No second opinion about where the
# passage is is involved: nothing here searches for a box, infers one, or
# repairs one. The cited rectangle is either the one the locator named or the
# view has no highlight.
#
# The two are complements, not substitutes. The raster is what a reader
# recognises as a page; this is what a client needs to know *which* element was
# cited, what the surrounding elements are called, and what to draw while the
# image is still in flight.


def _round(value: float) -> float:
    """Two decimals, matching how a coordinate reaches the wire.

    The same rounding the ``xywh=`` fragment uses, so a client that lays the
    block out from ``view`` and one that reads ``highlight`` are working from
    the same numbers rather than from two opinions that agree to within a
    rounding error.
    """
    return round(float(value), 2)


def _clip(text: str, limit: int) -> tuple[str, bool]:
    """Clip text to ``limit`` characters, reporting whether it bit.

    The flag is not optional. A view that quietly shortened a paragraph to fit
    a payload would be a silent truncation, which the release gate counts at
    zero, so every clipped string in the view says so at the level of the
    individual block as well as the view.
    """
    raw = str(text or "")
    if len(raw) <= limit:
        return raw, False
    return raw[:limit], True


def _box_payload(bbox: BoundingBox | None) -> dict[str, Any] | None:
    """The four numbers a client positions a rectangle by, or ``None``.

    ``None`` for anything :func:`_box_is_ordered` refuses, which is the single
    place the drawability rule lives: the resolution and the view cannot
    disagree about whether a box may be drawn, because they ask the same
    function. The numbers are exactly the ones ``to_query()`` emits — ``x0``,
    ``y0`` and the two absolute extents — so a client reading either source
    places the rectangle identically.
    """
    if not _box_is_ordered(bbox):
        return None
    assert bbox is not None
    return {
        "x": _round(bbox.x0),
        "y": _round(bbox.y0),
        "width": _round(bbox.width),
        "height": _round(bbox.height),
    }


def _missing_target(resolution: LocatorResolution) -> dict[str, Any]:
    """Why the view has no highlight, in the resolution's own vocabulary.

    Prefers the resolution's reason code, because that is the one the client is
    already rendering above the passage, and a view that explained the absence
    differently from the state banner would be two sentences about one fact.
    Falls back to the view-local "the region has no such block" when the region
    itself resolved and the target simply is not in it.
    """
    if resolution.reason_code:
        return {"reason_code": resolution.reason_code, "message": resolution.reason}
    return {"reason_code": VIEW_TARGET_MISSING, "message": VIEW_TARGET_MISSING_MESSAGE}


def _page_map_view(
    document: ParsedDocument, locator: SourceLocator, resolution: LocatorResolution
) -> dict[str, Any]:
    """One PDF page, laid out from the coordinates the parse recorded.

    The frame is the union of every rectangle on the page, not the paper's
    dimensions — the schema does not record a page size, and inventing one
    (a US Letter default, say) would put every block in a plausible wrong
    place. A union frame is a single uniform rescale of the true page, so
    positions *relative to each other* are exactly the page's, which is the
    only relation a highlight can be wrong about.

    Blocks whose box is relative are **omitted, not placed**. A mirrored block
    inside the page map would drag the frame to a region of the page where
    nothing is, and the honest blocks would then be positioned against a frame
    that is itself wrong. The count and the reason are reported instead, which
    is the release gate's "or explicitly marked".
    """
    page = locator.page
    target_box = _box_payload(locator.bbox)

    blocks: list[dict[str, Any]] = []
    omitted_relative = 0
    omitted_unpositioned = 0
    total = 0

    for element in document.elements:
        if element.locator.kind != "pdf_page" or element.locator.page != page:
            continue
        total += 1
        box = _box_payload(element.locator.bbox)
        if box is None:
            if element.locator.bbox is not None and element.locator.bbox.relative:
                omitted_relative += 1
            else:
                omitted_unpositioned += 1
            continue
        if target_box is not None and box == target_box:
            # The cited rectangle is drawn once, from ``target``. Leaving its
            # element in the background map would put two identical rectangles
            # on the page and leave the reader no way to tell which one is the
            # answer to "where does this citation point".
            continue
        preview, preview_truncated = _clip(element.text, VIEW_BLOCK_PREVIEW_CHARS)
        blocks.append(
            {
                "element_id": element.element_id,
                "role": element.role,
                "heading_path": list(element.heading_path),
                "text_preview": preview,
                "preview_truncated": preview_truncated,
                **box,
            }
        )

    blocks.sort(key=lambda item: (item["x"], -item["y"]))

    target: dict[str, Any] | None = None
    if target_box is not None:
        target = {
            "element_id": "",
            "text": "",
            "text_truncated": False,
            **target_box,
        }
        for element in document.elements:
            if (
                element.locator.kind == "pdf_page"
                and element.locator.page == page
                and _box_payload(element.locator.bbox) == target_box
            ):
                target["element_id"] = element.element_id
                text, truncated = _clip(element.text, VIEW_TARGET_TEXT_CHARS)
                target["text"] = text
                target["text_truncated"] = truncated
                break

    truncated_blocks = max(0, len(blocks) - VIEW_MAX_BLOCKS)
    blocks = blocks[:VIEW_MAX_BLOCKS]

    frame: dict[str, Any] | None = None
    rectangles = [(item["x"], item["y"], item["width"], item["height"]) for item in blocks]
    if target is not None:
        rectangles.append((target["x"], target["y"], target["width"], target["height"]))
    if rectangles:
        left = min(item[0] for item in rectangles)
        bottom = min(item[1] for item in rectangles)
        frame = {
            "x": _round(left),
            "y": _round(bottom),
            "width": _round(max(item[0] + item[2] for item in rectangles) - left),
            "height": _round(max(item[1] + item[3] for item in rectangles) - bottom),
        }

    return {
        "kind": VIEW_PAGE_MAP,
        "label": resolution.label,
        "page": page,
        "coordinate_space": "pdf_user_space_points",
        "is_raster": False,
        "note": VIEW_PAGE_MAP_NOTE,
        "frame": frame,
        "blocks": blocks,
        "target": target,
        "target_rejected": None if target is not None else _missing_target(resolution),
        "blocks_total": total,
        "blocks_shown": len(blocks),
        "blocks_truncated": truncated_blocks > 0,
        "omitted": {
            "relative_bbox": omitted_relative,
            "unpositioned": omitted_unpositioned,
        },
    }


def _slide_view(
    document: ParsedDocument,
    chunk: Chunk,
    locator: SourceLocator,
    resolution: LocatorResolution,
) -> dict[str, Any]:
    """One slide, as the ordered list of its text elements.

    A slide has no pixel coordinates in this schema — the locator names a slide
    number and, sometimes, an element index — so there is nothing to position
    and a fabricated position would be worse than none. What a reader can
    actually be given is the slide's own text with the cited element marked,
    which is what "jump to slide" means when there is no slide deck on screen.

    The cited element is found by intersecting the chunk's own ``element_ids``
    with the slide's elements. That set is what the chunk was built from, so
    an element reached through it is by construction part of the cited
    passage; searching the whole document for "some element on slide 3" would
    be a second opinion about which element was meant.
    """
    slide = locator.slide
    members = [
        element
        for element in document.elements
        if element.locator.kind == "slide" and element.locator.slide == slide
    ]
    members.sort(key=lambda element: (element.locator.element or 0, element.order))

    cited = {
        element_id
        for element_id in chunk.element_ids
        if any(element.element_id == element_id for element in members)
    }

    blocks: list[dict[str, Any]] = []
    target: dict[str, Any] | None = None
    for index, element in enumerate(members):
        is_target = element.element_id in cited
        text, truncated = _clip(
            element.text, VIEW_TARGET_TEXT_CHARS if is_target else VIEW_CONTEXT_TEXT_CHARS
        )
        block = {
            "index": index,
            "element_id": element.element_id,
            "role": element.role,
            "element": element.locator.element,
            "text": text,
            "text_truncated": truncated,
            "is_target": is_target,
        }
        blocks.append(block)
        if is_target and target is None:
            target = block

    return {
        "kind": VIEW_SLIDE,
        "label": resolution.label,
        "slide": slide,
        "note": VIEW_SLIDE_NOTE,
        "blocks": blocks,
        "target": target,
        "target_rejected": None if target is not None else _missing_target(resolution),
    }


def _element_view(
    document: ParsedDocument,
    chunk: Chunk,
    locator: SourceLocator,
    resolution: LocatorResolution,
) -> dict[str, Any]:
    """One DOCX paragraph, with the paragraphs around it for orientation.

    DOCX has no page grid in this schema, so "locate the element" cannot mean
    "draw a box". It means: name the heading path the paragraph sits under and
    show where it falls among its neighbours, because that pair is what a
    reader uses to find a paragraph in a document that shows no page numbers.

    The target is looked up first among the chunk's own elements and only then
    among the document's — the second lookup is safe *here* and would not be
    for a box, because a DOCX locator carries no rectangle for a wrong element
    to masquerade as. It finds the paragraph with this index, not a similar
    one.
    """
    paragraph = locator.paragraph
    members = [
        element
        for element in document.elements
        if element.locator.kind == "docx_element" and element.locator.paragraph is not None
    ]
    members.sort(key=lambda element: (element.locator.paragraph or 0, element.order))

    def _paragraph_of(element: Element) -> int:
        return int(element.locator.paragraph or 0)

    position: int | None = None
    for index, element in enumerate(members):
        if element.element_id in chunk.element_ids and _paragraph_of(element) == paragraph:
            position = index
            break
    if position is None:
        for index, element in enumerate(members):
            if _paragraph_of(element) == paragraph:
                position = index
                break

    first = max(0, (position or 0) - VIEW_ELEMENT_CONTEXT_BEFORE)
    last = position + VIEW_ELEMENT_CONTEXT_AFTER if position is not None else -1

    context: list[dict[str, Any]] = []
    target: dict[str, Any] | None = None
    for index, element in enumerate(members):
        if position is None or not (first <= index <= last):
            continue
        is_target = index == position
        text, truncated = _clip(
            element.text, VIEW_TARGET_TEXT_CHARS if is_target else VIEW_CONTEXT_TEXT_CHARS
        )
        block = {
            "paragraph": _paragraph_of(element),
            "element_id": element.element_id,
            "role": element.role,
            "heading_path": list(element.heading_path),
            "text": text,
            "text_truncated": truncated,
            "is_target": is_target,
        }
        context.append(block)
        if is_target:
            target = block

    return {
        "kind": VIEW_ELEMENT,
        "label": resolution.label,
        "paragraph": paragraph,
        "heading_path": list(locator.heading_path),
        "note": VIEW_ELEMENT_NOTE,
        "context": context,
        "target": target,
        "target_rejected": None if target is not None else _missing_target(resolution),
    }


def build_view(
    *,
    document: ParsedDocument,
    chunk: Chunk,
    locator: SourceLocator,
    resolution: LocatorResolution,
) -> dict[str, Any] | None:
    """The laid-out region for one resolved evidence, or ``None``.

    ``None`` for the formats with nothing to lay out — a spreadsheet range and
    a character offset into plain text — and for any locator kind this build
    does not know. Both are the same answer, and it is the safe one: a client
    that receives no view shows the passage and the state banner and nothing
    else, whereas a client handed a view it cannot draw would have to invent
    one.

    The view is built from the *resolution*, not from the locator alone, so
    that a highlight can only appear where ``to_query()`` already agreed to
    produce one. There is no path through this function that draws a rectangle
    the resolver refused, and that is the property the whole viewer rests on.
    """
    if locator.kind == "pdf_page":
        return _page_map_view(document, locator, resolution)
    if locator.kind == "slide":
        return _slide_view(document, chunk, locator, resolution)
    if locator.kind == "docx_element":
        return _element_view(document, chunk, locator, resolution)
    return None


# --------------------------------------------------------------------------
# The evidence-id namespace
# --------------------------------------------------------------------------
#
# There are two ways to name the same passage, and this module is where they are
# reconciled. The *chunk* id addresses the stored retrieval unit; the *evidence*
# id is what :mod:`paperwhisperer.rag.retrieve` mints around it and what the
# retriever, the Q&A claims and the front end's citation chips all put on the
# wire. A reader clicks an evidence id, so an API that resolves only chunk ids
# fails the product's central promise for every id the product actually emits --
# which is exactly the defect this section exists to close.
#
# **The evidence id is a content address over (chunk, plan), not a stored row.**
# ``rag.retrieve`` derives it as ``"ev_" + sha256(f"{chunk_id}|{plan_signature}")[:16]``
# and does not persist the mapping anywhere, so there is no lookup that would
# answer "which chunk is this?". The honest way to resolve one is therefore to
# *re-derive* it rather than to look it up: recompute the address for every
# candidate (chunk, plan) pair the document could have produced and match. This
# is a first-principles resolution, not a lookup table, and that is why it
# survives a process restart, a different store tier, and a reader who never ran
# the retrieval that minted the id.
#
# **Why re-derivation is safe and not a guess.** A match is not "the closest
# thing": it is equality against a value the retriever itself computed, over a
# candidate set drawn from the document's *own* chunks. The id keeps 64 bits of a
# SHA-256 digest, so the chance that a wrong candidate collides is about
# ``len(chunks) * len(plans) / 2**64`` -- for a 2,000-chunk paper against five
# registered plans that is roughly one in 10^15. When it matches, it matched.
#
# **Scope is the cross-document guarantee.** :func:`find_chunk_for_evidence_id`
# only ever looks at the chunks it is handed, and the route hands it the chunks of
# the document named in the URL. So an evidence id minted for *another* document
# cannot resolve here even if the caller guessed it correctly: the candidates
# simply do not contain its chunk. That is the reason this is a re-derivation
# scoped to one document rather than a corpus-wide evidence index -- there is no
# global index to leak from, and the answer for a foreign id is the same
# not-found a typo gets.
#
# **Why the plan signature is a candidate and not a lookup.** The same passage
# retrieved under two strategies gets two evidence ids, so the id alone does not
# name a plan. The candidates are therefore two sets unioned together, and the
# second one exists because the first was not enough.
#
# The first is the registered plans (:data:`paperwhisperer.rag.retrieve.PLANS`),
# which is what the Q&A surface -- the one that mints the ids behind the
# citation chips -- retrieves under. That set alone is a *silent* ceiling: a plan
# built with :meth:`~paperwhisperer.rag.retrieve.RetrievalPlan.with_overrides` has
# a signature no registry enumerates, and
# :func:`~paperwhisperer.analysis.paper_facts.make_evidence_provider` builds one
# that way for every fact field it extracts. Those ids are exactly what an
# analysis card would put on a citation chip, so a registry-only resolver 404s
# every one of them -- and passes every test that retrieves under the Q&A plan,
# because those are the ids the Q&A plan mints. That is D1 recurring through the
# second door, and the population it hides is the one the registry-only suite
# cannot see.
#
# The second set is what *this document's own evidence* was minted under, which
# the retriever records as it mints
# (:func:`~paperwhisperer.rag.retrieve.record_plan_signature`, read back by
# :func:`document_plan_signatures`). It is recorded rather than recomputed because
# the ids are stored -- `paper_facts` writes them into `PaperFacts.evidence_ids`,
# and sessions carry them -- so a citation outlives the process that minted it.
# Learning signatures in-process would work for a citation minted this
# request-cycle and fail for every stored one, which is the same defect with a
# narrower trigger.
#
# Widening the candidate *set* does not weaken the cross-document guarantee above,
# and the distinction is the one to hold onto: a signature recorded against some
# other paper contributes a hash to try, never a chunk whose text could be read.
# The candidates that can *match* are still this document's chunks and no others.
# A caller that knows the set for some other reason can still pass
# ``plan_signatures`` and override both halves.


#: The prefix :func:`paperwhisperer.rag.retrieve._evidence_id` puts on an
#: evidence id. The schema's own default factory (``_new_id("ev")``) mints the
#: same shape, so the prefix alone does not prove a *retriever* id -- which is
#: fine, because an id this resolver cannot re-derive simply will not match and
#: the caller gets a plain not-found.
EVIDENCE_ID_PREFIX = "ev_"

#: How much of the SHA-256 digest the retriever keeps. Mirrors
#: :func:`paperwhisperer.rag.retrieve._evidence_id`; see :data:`EVIDENCE_ID_PREFIX`.
EVIDENCE_ID_HEX_CHARS = 16

#: The exact shape of a retriever-minted evidence id. Used only to *skip* the
#: re-derivation for ids that cannot possibly match -- a malformed id is
#: rejected earlier, at the route, and an id of the wrong length or alphabet is
#: simply not one of ours. It never causes a miss that a real id would survive.
_EVIDENCE_ID_SHAPE = re.compile(rf"^{EVIDENCE_ID_PREFIX}[0-9a-f]{{{EVIDENCE_ID_HEX_CHARS}}}$")


def evidence_id_for(chunk_id: str, plan_signature: str) -> str:
    """The id the retriever mints for one ``(chunk, plan)`` pair.

    **This mirrors :func:`paperwhisperer.rag.retrieve._evidence_id`, deliberately.**
    The derivation is a content address, so a resolver that did not compute the
    identical value would resolve nothing, and one that did compute it twice
    could drift the moment the minting side changed. The mirror is pinned to the
    minting side by ``tests/test_evidence_api.py``, which asserts the two agree
    for every registered plan; that test is what makes this a mirror rather than
    a second definition, and it fails loudly the day the minting changes.

    The alternative -- reaching into ``rag.retrieve`` for its private minter --
    would couple the resolution of every citation click to the import graph of
    the retrieval engine (and through it to the store-bundle machinery). This
    module is pulled in by every citation click and every retrieval leg and is
    documented as staying free of that weight, so it re-states the four lines of
    arithmetic and pins them, rather than importing the machinery.
    """
    payload = f"{chunk_id}|{plan_signature}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return EVIDENCE_ID_PREFIX + digest[:EVIDENCE_ID_HEX_CHARS]


#: The key :func:`paperwhisperer.rag.retrieve.record_plan_signature` writes the
#: signatures under. Named as a literal rather than imported, for the same
#: reason :data:`EVIDENCE_ID_PREFIX` is: this module stays free of the retrieval
#: engine's import graph, and ``tests/test_evidence_api.py`` pins the two halves
#: together so a rename cannot pass unnoticed.
PLAN_SIGNATURES_METADATA_KEY = "retrieval_plan_signatures"


def document_plan_signatures(record: DocumentRecord | None) -> tuple[str, ...]:
    """Plan signatures this document's own evidence has been minted under.

    The second half of the candidate set, and the half that matters. The
    registered plans (:func:`known_plan_signatures`) cover the Q&A surface, which
    is where the citation chips come from -- but a plan built with
    :meth:`~paperwhisperer.rag.retrieve.RetrievalPlan.with_overrides` has a
    signature no registry enumerates, and the fact layer builds one that way for
    every field it extracts. An analysis card's citation is minted under exactly
    such a plan, so enumerating only the registry is what makes *that* click
    404 while every Q&A test stays green.

    The retriever writes this as it mints, which is what makes the answer
    survive a process restart: the ids outlive the process, and so does the
    record of how they were addressed. A resolver that learned signatures
    in-process instead would work for a citation minted this request-cycle and
    fail for every stored one -- D1 again, through the second door.
    """
    if record is None:
        return ()
    metadata = getattr(record, "metadata", None)
    if not isinstance(metadata, dict):
        return ()
    raw = metadata.get(PLAN_SIGNATURES_METADATA_KEY)
    if not isinstance(raw, (list, tuple, set)):
        return ()
    return tuple(sorted({str(item) for item in raw if str(item or "").strip()}))


def known_plan_signatures() -> tuple[str, ...]:
    """Signatures of every retrieval plan this deployment has registered.

    Imported lazily because :mod:`paperwhisperer.rag.retrieve` reaches the
    store bundle; see :func:`evidence_id_for` for why this module does not import
    it at module scope. A deployment that registers its own plan (see
    :func:`~paperwhisperer.rag.retrieve.register_plan`) is covered here for free,
    because the registry *is* the enumeration.
    """
    try:
        from paperwhisperer.rag.retrieve import PLANS
    except Exception:  # noqa: BLE001 - a missing retrieval engine is "no plans"
        logger.warning("无法导入检索计划注册表，evidence id 解析将只尝试已传入的签名。")
        return ()
    return tuple(sorted({plan.signature() for plan in PLANS.values()}))


def find_chunk_for_evidence_id(
    evidence_id: str,
    chunks: Iterable[Chunk],
    *,
    plan_signatures: Sequence[str] | None = None,
    document_record: DocumentRecord | None = None,
) -> Chunk | None:
    """The chunk this evidence id addresses, or ``None``.

    A pure function of ``(evidence_id, chunks)`` -- no store, no side effects --
    which is the whole reason it lives here rather than in the route: a test can
    assert that a real retriever-minted id resolves without standing up a server,
    and the route stays a lookup.

    ``chunks`` must be the chunks of the document the caller asked about. That is
    the cross-document guarantee in one sentence: a foreign evidence id resolves
    to ``None`` here because its chunk is not among the candidates, so this
    function -- and therefore the route -- never even considers another
    document's text. See the note above on why that is deliberate. That guarantee
    holds *however* wide the signature set becomes, which is what licensed
    widening it below: a signature recorded against some other paper contributes
    a hash to try, never a chunk whose text could be read.

    **The candidate set is the union of two things, and both are necessary.**
    :func:`known_plan_signatures` is the registered plans -- what the Q&A surface
    retrieves under. :func:`document_plan_signatures` is what *this* document's
    evidence was actually minted under, and it is the only way to reach a plan
    built with ``with_overrides``. Passing ``plan_signatures`` explicitly
    replaces both, for a caller that already knows the set and wants neither the
    union nor a registry import.

    Cost is ``len(chunks) * len(signatures)`` short SHA-256 computations, which
    for a full-length paper is a few thousand hashes -- sub-millisecond, and
    bounded by the document the caller already named rather than by the corpus.
    """
    if not _EVIDENCE_ID_SHAPE.match(str(evidence_id or "")):
        return None
    if plan_signatures is not None:
        signatures = tuple(str(item) for item in plan_signatures)
    else:
        signatures = tuple(
            sorted(set(known_plan_signatures()) | set(document_plan_signatures(document_record)))
        )
    if not signatures:
        return None
    for chunk in chunks:
        chunk_id = str(getattr(chunk, "chunk_id", "") or "")
        if not chunk_id:
            continue
        for signature in signatures:
            if evidence_id_for(chunk_id, signature) == evidence_id:
                return chunk
    return None


# --------------------------------------------------------------------------
# The entry point
# --------------------------------------------------------------------------


def resolve_evidence(
    *,
    evidence_id: str,
    document: ParsedDocument,
    chunk: Chunk,
    document_record: DocumentRecord | None = None,
) -> ResolvedEvidence:
    """Resolve one stored evidence into the view a citation click opens.

    An evidence, in this build, *is* a stored retrieval unit: a chunk carries
    the text a claim was grounded in and the locator that text came from, and
    those two together are everything a reader needs. T1.4's retriever mints
    :class:`~paperwhisperer.ingestion.schema.Evidence` objects around exactly
    this pair, so the resolver is already the function that has to be right
    for the retriever's output to be renderable — only the lookup that found
    the chunk would change when the retriever persists evidence under its own
    identifiers, and it changes in the route, not here.

    The table lookup walks the chunk's own ``element_ids`` and nothing else.
    That set is what the chunk was built from, so a table found through it is
    by construction part of the cited passage; widening the search to "some
    table in this document" would let a citation about a paragraph render
    the nearest table instead, which is a wrong answer wearing a right one.

    ``view`` is built last and from the resolution rather than re-deriving
    anything: the block a client draws is a rendering of the decision this
    function's callee already made, and a second derivation would be a second
    decision.
    """
    locator = chunk.locator
    resolution = resolve_locator(locator, document_id=document.document_id)

    snippet = _build_snippet(chunk.verbatim_text)
    snippet["source_id"] = chunk.chunk_id

    elements_by_id = {element.element_id: element for element in document.elements}
    table: dict[str, Any] | None = None
    for element_id in chunk.element_ids:
        element = elements_by_id.get(element_id)
        if element is not None and element.table is not None:
            table = _table_payload(element, locator)
            break

    version, warning = _version_check(chunk, document_record)
    warnings = [warning] if warning else []

    return ResolvedEvidence(
        evidence_id=evidence_id,
        document_id=document.document_id,
        chunk_id=chunk.chunk_id,
        citation_label=resolution.label,
        locator=locator.to_dict(),
        snippet=snippet,
        visual_targeting=resolution.to_dict(),
        element_ids=list(chunk.element_ids),
        table=table,
        document_version=version,
        warnings=warnings,
        view=build_view(
            document=document, chunk=chunk, locator=locator, resolution=resolution
        ),
    )
