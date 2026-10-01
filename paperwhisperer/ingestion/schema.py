"""Canonical parsed-document schema.

Everything a parser adapter produces is normalized into the types defined
here, so the rest of the application never learns which backend ran. Two
properties matter more than anything else in this module:

1. **A locator is a value, not a pointer.** ``SourceLocator`` round-trips
   through JSON without loss, because a citation written to a session today
   must still resolve when the document is re-read next month. ``to_dict`` /
   ``from_dict`` are therefore exact inverses and are tested as such.
2. **Absence is explicit.** A parsed element that could not be located says
   so via ``locator.kind == "unknown"`` rather than carrying a fabricated
   page number. Downstream, an unlocatable quote is a different product
   decision than a wrong page number.

The planning document asks for Pydantic here, and the first answer to that was
that the project carries no Pydantic dependency. That answer was wrong —
FastAPI has pulled in Pydantic 2.x since long before this module existed — and
the decision has since been revisited. What survived the revisit is a split:
**dataclasses stay, but only because they are the hot path.**

A parsed document holds thousands of elements, and an adapter builds them in a
tight loop. A Pydantic model on that path would run field validation on every
element, for a document the adapter just constructed in memory and no user has
seen yet. Validating data the process itself produced, moments ago, to catch
mistakes the type system could have caught, is cost without a boundary. So the
canonical types here are stdlib dataclasses, and the ``__post_init__``
normalization they do is deliberately small and total: an unknown role becomes
``text`` and a malformed number becomes ``None``, which is the correct response
to a value a parser emitted a moment ago.

Pydantic is not absent, it is *relocated*. Untrusted data reaches this project
at a small number of well-defined edges — a parser backend's output, a JSON
file written by an older build, an API request or response body — and there a
malformed value is not the process's own sloppiness, it is someone else's.
:mod:`paperwhisperer.ingestion.contracts` holds the Pydantic models that guard
those edges, including the one that matters most here: a locator that is
subtly wrong produces a citation that confidently points at the wrong place,
which is worse than a parse failure the user can see and retry.

Locator shapes, one per format:

===============  ==========================================================
``pdf_page``     ``page`` plus a ``bbox`` when the backend reports geometry
``slide``        ``slide`` and optional ``element`` for a shape index
``sheet_range``  ``sheet`` plus a cell range
``docx_element`` a paragraph/table index plus the heading path
``txt_offset``   a character span, since plain text has no page concept
===============  ==========================================================

A ``bbox`` additionally records whether it is *absolutely* positioned.
Rectangles are the one value here that a converter can get quietly, silently
wrong, and ``BoundingBox.relative`` is what keeps that visible downstream
instead of leaving it in a log line.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from paperwhisperer.ingestion.latex import (
    canonical_latex,
    formulas_match,
    join_latex,
    latex_defects,
)

#: Bumped whenever the shape of a serialized document changes in a way an
#: older reader could not understand. Readers should reject a major mismatch
#: rather than silently misinterpreting locators.
SCHEMA_VERSION = "1.0"

#: The only shape a version string may take: ``MAJOR.MINOR``.
#:
#: A major comparison alone cannot establish that. ``"1"`` splits to ``["1"]``,
#: whose first element equals the supported major, so a string with no minor
#: part passes a check that is supposed to be asking "is this the same
#: generation of the format?". The answer for an unparseable version cannot be
#: "yes, probably": at that point nothing is known about the document's shape,
#: and the honest response is to refuse it. The pattern lives here rather than
#: in :mod:`contracts` because the dataclass reader has to enforce it too — the
#: two are supposed to agree about what a version is, and a second copy of the
#: rule is a second opinion waiting to drift.
SCHEMA_VERSION_PATTERN = re.compile(r"^\d+\.\d+$")

#: Every ``SourceLocator.kind`` this module can emit.
LOCATOR_KINDS = (
    "pdf_page",
    "slide",
    "sheet_range",
    "docx_element",
    "txt_offset",
    "unknown",
)

#: Every ``Element.role`` this module can emit. Roles drive chunking and
#: rendering, so adapters must map onto these rather than invent new ones;
#: an unknown role is downgraded to ``text`` by ``Element.__post_init__``.
ELEMENT_ROLES = (
    "title",
    "heading",
    "paragraph",
    "list_item",
    "caption",
    "table",
    "formula",
    "figure",
    "footnote",
    "header",
    "footer",
    "code",
    "text",
)

#: Extraction status recorded per fact, so a partially filled fact is visible
#: instead of indistinguishable from a complete one.
EXTRACTION_STATUSES = (
    "extracted",
    "not_found",
    "conflicting",
    "needs_review",
)

_PDF_PAGE = "pdf_page"
_SLIDE = "slide"
_SHEET_RANGE = "sheet_range"
_DOCX_ELEMENT = "docx_element"
_TXT_OFFSET = "txt_offset"
_UNKNOWN = "unknown"


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _prune(mapping: dict[str, Any]) -> dict[str, Any]:
    """Drop keys whose value is ``None`` so serialized output stays small."""
    return {key: value for key, value in mapping.items() if value is not None}


# --------------------------------------------------------------------------
# Locators
# --------------------------------------------------------------------------


@dataclass(slots=True)
class BoundingBox:
    """A rectangle in PDF user-space points, origin bottom-left.

    ``relative`` records that the rectangle is *not* absolutely positioned.
    It is set when a converter reported a ``TOPLEFT`` (image-convention) box
    and the page height needed to mirror it into PDF space was unavailable, so
    the adapter had to negate ``y`` instead. The result has the right size and
    the right within-page vertical ordering, and it points at a region that is
    mirrored about the page's top edge — the highlight lands in the wrong place.

    **Why the flag exists rather than a log line.** A relative box is
    indistinguishable from an absolute one by inspecting the rectangle: it is
    finite, ordered, and positive-extent, so every existing check passes. The
    only thing that distinguishes it is where it came from, and that knowledge
    is available exactly once, in the adapter, at the moment of conversion.
    A warning on the document says "some boxes are approximate" to a reader
    three stages later; a flag on the box says "this one is approximate" to
    the code that is about to draw it. This project is Evidence-first: a bbox
    that is silently wrong is a citation that highlights the wrong region, and
    the difference between "approximate" and "exact" is invisible in the
    serialized rectangle.

    It defaults to ``False`` — the overwhelming majority of boxes are
    absolutely positioned, and every payload serialized before this field
    existed describes boxes that were. The flag is pruned at its default in
    :meth:`to_dict` for the same reason ``Formula.display`` is: a document
    written by this build has to be byte-identical to one written by the build
    before it, so an older reader is unaffected by a key it has never heard of.
    """

    x0: float
    y0: float
    x1: float
    y1: float
    relative: bool = False

    def __post_init__(self) -> None:
        # A producer that knows it mirrored a box should say so; one that
        # forgets is caught here. PDF user space measures ``y`` upward from the
        # bottom-left of the page, so a rectangle whose *lower* edge falls
        # below the page origin cannot be an absolute position — it is either a
        # mirrored box (a ``TOPLEFT`` rectangle whose y was negated because the
        # page height was unavailable) or a corrupt one, and in neither case may
        # it be rendered as a highlight. Deriving the flag from the coordinates
        # is therefore not a guess about intent: it is the one reading of an
        # off-page rectangle that is consistent with the coordinate system this
        # class documents.
        #
        # The *lower* edge is the test rather than the upper one, because the
        # upper edge can legitimately land on zero — a ``TOPLEFT`` box flush
        # with the top of the page negates to ``y1 == 0`` — and flagging by
        # ``y1`` would let exactly the case this exists for slip through.
        if self.y0 < 0:
            self.relative = True

    @property
    def width(self) -> float:
        return abs(self.x1 - self.x0)

    @property
    def height(self) -> float:
        return abs(self.y1 - self.y0)

    def to_dict(self) -> dict[str, Any]:
        return _prune({
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
            "relative": True if self.relative else None,
        })

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BoundingBox:
        return cls(
            x0=float(payload["x0"]),
            y0=float(payload["y0"]),
            x1=float(payload["x1"]),
            y1=float(payload["y1"]),
            relative=bool(payload.get("relative") or False),
        )

    def to_query(self) -> str:
        """Render a ``pdf.js`` page fragment for the evidence viewer.

        Numbers are normalized so the same rectangle always produces
        byte-identical output. That matters because the fragment ends up in a
        cache key: ``10`` and ``10.0`` would otherwise evict each other.

        **Refuses a relative box.** This method's entire output is consumed
        by something that draws a highlight at those coordinates, and a
        mirrored rectangle draws that highlight over the wrong lines while
        looking entirely correct. Refusing is the only response that cannot
        produce a wrong citation; a caller that genuinely wants to draw an
        approximate region has to say so by handling the error rather than by
        getting a fragment that quietly means something else.
        """
        if self.relative:
            raise ValueError(
                "a relative bounding box has no absolute position and cannot be "
                "rendered as a page fragment; the highlight would be mirrored "
                "onto the wrong region of the page"
            )
        return (
            f"xywh={_number(self.x0)},{_number(self.y0)}"
            f",{_number(self.width)},{_number(self.height)}"
        )


@dataclass(slots=True)
class SourceLocator:
    """Where an element came from, in the vocabulary of its file format.

    Only the fields meaningful for ``kind`` are populated; the rest stay
    ``None``. ``unknown`` is a first-class kind rather than an error, because
    a text layer that loses page boundaries is a degraded parse the user
    should see flagged, not a failure.
    """

    kind: str = _UNKNOWN
    page: int | None = None
    bbox: BoundingBox | None = None
    slide: int | None = None
    element: int | None = None
    sheet: str | None = None
    cell_range: str | None = None
    paragraph: int | None = None
    heading_path: tuple[str, ...] = ()
    offset_start: int | None = None
    offset_end: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in LOCATOR_KINDS:
            self.kind = _UNKNOWN
        if isinstance(self.heading_path, list):
            self.heading_path = tuple(str(item) for item in self.heading_path)

    @property
    def is_resolvable(self) -> bool:
        """Whether this locator can be turned back into a location a reader sees.

        A ``pdf_page`` without a bbox still resolves to a page; a bare
        ``unknown`` does not resolve at all. The evidence API branches on
        this instead of re-deriving the rule.
        """
        if self.kind == _PDF_PAGE:
            return self.page is not None
        if self.kind == _SLIDE:
            return self.slide is not None
        if self.kind == _SHEET_RANGE:
            return bool(self.sheet)
        if self.kind == _DOCX_ELEMENT:
            return self.paragraph is not None
        if self.kind == _TXT_OFFSET:
            return self.offset_start is not None
        return False

    @property
    def sort_key(self) -> tuple:
        """Reading order, per format.

        Formats that have no spatial order (plain text) fall back to the
        character offset, so a single sort works for every backend.
        """
        if self.kind == _PDF_PAGE:
            return (0, self.page or 0, 0, 0, self.offset_start or 0)
        if self.kind == _SLIDE:
            return (1, self.slide or 0, 0, 0, self.element or 0)
        if self.kind == _DOCX_ELEMENT:
            return (2, self.paragraph or 0, 0, 0, 0)
        if self.kind == _SHEET_RANGE:
            return (3, 0, 0, 0, 0)
        if self.kind == _TXT_OFFSET:
            return (4, 0, 0, 0, self.offset_start or 0)
        return (5, 0, 0, 0, 0)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"kind": self.kind}
        if self.page is not None:
            payload["page"] = self.page
        if self.bbox is not None:
            payload["bbox"] = self.bbox.to_dict()
        if self.slide is not None:
            payload["slide"] = self.slide
        if self.element is not None:
            payload["element"] = self.element
        if self.sheet is not None:
            payload["sheet"] = self.sheet
        if self.cell_range is not None:
            payload["cell_range"] = self.cell_range
        if self.paragraph is not None:
            payload["paragraph"] = self.paragraph
        if self.heading_path:
            payload["heading_path"] = list(self.heading_path)
        if self.offset_start is not None:
            payload["offset_start"] = self.offset_start
        if self.offset_end is not None:
            payload["offset_end"] = self.offset_end
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SourceLocator:
        bbox_payload = payload.get("bbox")
        return cls(
            kind=str(payload.get("kind") or _UNKNOWN),
            page=_as_int(payload.get("page")),
            bbox=BoundingBox.from_dict(bbox_payload) if bbox_payload else None,
            slide=_as_int(payload.get("slide")),
            element=_as_int(payload.get("element")),
            sheet=_as_optional_str(payload.get("sheet")),
            cell_range=_as_optional_str(payload.get("cell_range")),
            paragraph=_as_int(payload.get("paragraph")),
            heading_path=tuple(payload.get("heading_path") or ()),
            offset_start=_as_int(payload.get("offset_start")),
            offset_end=_as_int(payload.get("offset_end")),
        )


def _unit_group_key(members: list[Element]) -> tuple:
    """Sort key for a page/slide group: the locator order of its first member."""
    return members[0].locator.sort_key


def _number(value: float) -> str:
    """Format a coordinate for a URL fragment without a trailing ``.0``.

    Page fragments land in cache keys, so ``10`` and ``10.0`` have to render
    identically or the same rectangle evicts itself.
    """
    rounded = round(float(value), 2)
    return str(int(rounded)) if rounded == int(rounded) else str(rounded)


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


# --------------------------------------------------------------------------
# Rich payloads
# --------------------------------------------------------------------------


#: Where a formula's ``display`` bit came from.
#:
#: This exists because ``display`` alone is a claim the schema cannot support.
#: A reader of a serialized document sees ``display: true`` and has no way to
#: know whether the backend said so or the value was assumed. Those are very
#: different facts: one is an observation about the page, the other is this
#: code's guess. And a table of "which formulas are display" built out of
#: assumptions is worse than no table at all, because it looks like evidence.
FORMULA_DISPLAY_SOURCES = (
    #: The backend distinguished it. MinerU's ``content_list`` types an entry
    #: ``equation`` or ``equation_inline``, which is an observation about the
    #: page and not an inference from it.
    "backend",
    #: Inferred from geometry: the formula occupies a line of its own and is
    #: centred, or it carries an equation number. A real signal, and still an
    #: inference — a wide inline formula at the end of a line looks the same.
    "layout",
    #: The formula says so — a ``\begin{equation}`` wrapper, or ``$$``/``$``
    #: delimiters the producer kept. Strongest of the three after ``backend``,
    #: because the formula itself carries the claim rather than the code
    #: inferring it.
    "latex",
    #: Nobody knows. The bit is present because the field is not optional, and
    #: its value is a default chosen for rendering safety, not a finding.
    "unknown",
)

#: The value ``display`` takes when the source is ``unknown``.
#:
#: ``True``, and the choice is argued rather than convenient. An inline formula
#: misrendered as a display one produces a ``$$…$$`` block in the middle of a
#: sentence — ugly, and visible. A display formula misrendered as inline gets
#: demoted to prose by the chunker and merged into the paragraph around it,
#: which is the flattening this module exists to prevent and which no reader
#: can see. So the error that is invisible when it happens is the one the
#: default refuses to make. A caller that genuinely cannot know should say so
#: with ``display_source`` rather than by choosing the other bit.
DEFAULT_DISPLAY_WHEN_UNKNOWN = True

#: How much of the source formula's *structure* survived parsing.
#:
#: This is not a quality score and it is not about the model. It is about the
#: parser, and it exists because of one measured fact: docling 2.131.0 on
#: ``eval/corpus/digital_single_col/hinton_distillation.pdf`` returns the
#: formula ``qi = exp(zi/T) ∑ j exp(zj/T) (1)`` — the subscripts are gone.
#: There is no ``_`` in the string at all, because a PDF text layer has no
#: notion of a subscript: the glyph ``i`` is simply rendered smaller and lower
#: than ``q``. A model reading the same page and writing ``q_i`` has produced
#: the *correct* formula, and comparing the two byte-wise reports the model as
#: having invented it.
#:
#: That is a false accusation produced by a comparison that could not have
#: succeeded, and it is the same failure
#: :class:`~paperwhisperer.ingestion.latex.LatexVerdict` exists to prevent for
#: an inline-image fraction. So the structure loss is recorded, and the formula
#: index downgrades a ``DIFFERENT`` verdict to "could not be compared" when the
#: *source* was flattened. Without it, a parser that cannot read subscripts
#: would make every formula in the paper look fabricated.
LATEX_SOURCES = (
    #: The source carried grouping or a script, so its structure is intact.
    "structured",
    #: No ``{...}`` group and no ``_``/``^`` anywhere: the signature of a
    #: text-layer transcription, where subscripts and superscripts were
    #: flattened into adjacent letters.
    "flat",
    #: Not determined. The default, and a real answer rather than a claim.
    "unknown",
)


def infer_latex_source(latex: str) -> str:
    """Whether a formula's LaTeX kept the structure the page had.

    Conservative in the direction that matters: it says ``flat`` only when
    there is *no* evidence of structure at all, so a TeX-source formula is
    never mislabelled as a transcription. A formula genuinely written without
    braces and without a script (``a+b``) is labelled ``flat``, which costs
    nothing — such a formula is spelled identically on both sides of any
    comparison, so the flag never decides anything about it.
    """
    text = str(latex or "")
    if not text:
        return "unknown"
    if "{" in text or "}" in text or "_" in text or "^" in text:
        return "structured"
    return "flat"


def infer_display(latex: str, *, centred: bool = False, numbered: bool = False) -> str:
    """Which ``FORMULA_DISPLAY_SOURCES`` a producer's flags justify.

    Takes the two layout facts an adapter can actually observe on a PDF and
    answers with the *strongest source that is warranted*, not with a boolean.
    The distinction matters at the only place it is visible: a report that says
    "4 display formulas" is making a claim about the document, and this
    function is what decides whether that claim was observed or guessed.

    ``latex`` wins when it carries its own delimiters, because a formula
    wrapped in ``\\begin{equation}`` is a block by construction and no amount of
    layout evidence should be able to contradict it. Centring and an equation
    number are the two layout signals that survive PDF extraction — an inline
    formula has neither — so either one justifies ``layout``. Neither does, and
    the answer is ``unknown``, which is a real answer rather than a failure.
    """
    text = str(latex or "")
    if "\\begin{" in text and any(
        f"\\end{{{name}}}" in text for name in _DISPLAY_ENVIRONMENTS
    ):
        return "latex"
    if "$$" in text or "\\[" in text or "\\]" in text:
        return "latex"
    if centred or numbered:
        return "layout"
    return "unknown"


#: LaTeX environments that are a display equation by construction. A formula
#: inside one of these is a block in the source no matter what the layout said.
_DISPLAY_ENVIRONMENTS = ("equation", "align", "gather", "multline", "eqnarray", "flalign")


@dataclass(slots=True)
class Formula:
    """A formula kept as LaTeX plus the two bits LaTeX cannot express.

    A bare ``Element.latex`` string loses *display* versus *inline*, and
    MinerU's ``content_list`` distinguishes the two (``equation`` and
    ``equation_inline``) before normalization collapses both onto the single
    role ``formula``. That collapse is invisible in the output but not in the
    document: an inline formula belongs inside the sentence around it, while a
    display formula is its own block and gets its own chunk. Naming the type is
    what makes the distinction a field something can eventually be written to,
    instead of an argument about which string format was used.

    ``display_source`` is the second bit, and it is the one that makes the
    first one trustworthy. ``display`` is a boolean with a default, so a
    document in which nothing set it looks exactly like a document in which a
    backend observed every formula to be a block. Those are different documents
    and only one of them is a finding, so the provenance travels with the flag.

    ``latex`` is validated to be non-blank. A formula that survived parsing as
    an empty string is not a formula, and the quality gate reports that case by
    counting role ``formula`` with no LaTeX — so the count is only meaningful if
    an empty formula cannot quietly masquerade as one.
    """

    latex: str
    display: bool = DEFAULT_DISPLAY_WHEN_UNKNOWN
    display_source: str = "unknown"

    def __post_init__(self) -> None:
        self.latex = str(self.latex).strip()
        if not self.latex:
            raise ValueError("A formula must carry non-empty LaTeX")
        if self.display_source not in FORMULA_DISPLAY_SOURCES:
            self.display_source = "unknown"

    @property
    def latex_source(self) -> str:
        """How much of the page's structure this formula's LaTeX kept.

        Derived from the LaTeX rather than stored, and that is a considered
        choice rather than a shortcut. A stored field would have to be written
        by every producer, and the ones that would write it are the ones whose
        output proves it cannot be right: an adapter that transliterates
        Unicode maths has, by that act, already lost the structure, and
        recording that it lost it is not information it possesses. The
        signature is in the string, and the string is what every reader sees.

        It also means the answer cannot go stale. A stored value describes the
        LaTeX as it was when someone decided; this describes the LaTeX that is
        actually in the element.
        """
        return infer_latex_source(self.latex)

    @property
    def is_structure_flattened(self) -> bool:
        """Whether the parser is known to have lost sub/superscripts here.

        Read by the formula index before it reports a formula as invented.
        A source flattened this way cannot express ``q_i``, so a model that
        writes ``q_i`` has not diverged from the paper -- it has *recovered*
        structure the parser dropped, and calling that a fabrication would be
        the gate accusing the model of the parser's own failure.
        """
        return self.latex_source == "flat"

    @property
    def defects(self) -> tuple[str, ...]:
        """What is structurally wrong with this formula's LaTeX.

        A *report*, never a filter. A formula with an unbalanced brace still
        renders, because the alternative — dropping it — produces a document
        that is silently missing an equation the paper has, which is a
        corruption no reader can see. What the defects buy is that the renderer
        can say "this one line may not typeset" instead of letting the reader
        assume it did.

        Total by construction: the question is asked of a string a model wrote,
        and the answer to a malformed question is a malformed string rather
        than an exception in a render path.
        """
        return latex_defects(self.latex)

    @property
    def is_well_formed(self) -> bool:
        return not self.defects

    @property
    def canonical_latex(self) -> str:
        """The normalised form two formulas are compared in.

        Exposed because the comparison and the gate are the same question, and
        a caller that wanted to canonicalise by reaching into
        :mod:`paperwhisperer.ingestion.latex` would be depending on a private
        path through the schema to answer something the schema owns.
        """
        return canonical_latex(self.latex).text

    def matches(self, other: object) -> bool:
        """Whether this formula is the same expression as ``other``.

        Accepts a :class:`Formula` or a LaTeX string, because the caller
        usually has the other one as a raw model output and converting it
        first is a step they will forget. An unestablished match is ``False``
        — see :attr:`~paperwhisperer.ingestion.latex.LatexComparison.matches`
        for why that direction is the safe one.
        """
        other_latex = other.latex if isinstance(other, Formula) else str(other or "")
        return formulas_match(self.latex, other_latex)

    @property
    def rendered(self) -> str:
        """LaTeX wrapped in the delimiters the prose projection needs.

        Display and inline are delimited differently because a ``$$…$$`` block
        dropped into the middle of a sentence is what a reader sees as a
        rendering bug, and the alternative — no delimiters — is what makes a
        formula indistinguishable from the prose it is embedded in.
        """
        if self.display:
            return f"$${self.latex}$$"
        return f"${self.latex}$"

    def to_dict(self) -> dict[str, Any]:
        # Both keys are pruned at their defaults, so a formula whose display
        # bit was never set serializes to the same ``{"latex": ...}`` payload
        # a pre-``display`` build produced and an older reader is unaffected.
        #
        # ``display_source`` is pruned at ``"unknown"`` for the same reason
        # ``display`` is pruned at ``True``: an absent key means "nobody said",
        # which is exactly what ``unknown`` means, and writing the word out on
        # every formula would buy nothing.
        return _prune({
            "latex": self.latex,
            "display": None if self.display else False,
            "display_source": None if self.display_source == "unknown" else self.display_source,
        })

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Formula:
        raw_display = payload.get("display")
        return cls(
            latex=str(payload.get("latex") or ""),
            display=True if raw_display is None else bool(raw_display),
            display_source=str(payload.get("display_source") or "unknown"),
        )


#: The route that renders one page of a stored document as a PNG. Read from
#: :mod:`paperwhisperer.api.routes_evidence`; duplicated as a constant here
#: because :mod:`paperwhisperer.analysis.service` is the other half of this
#: contract and a test pins both against each other, the same way
#: ``EVIDENCE_API_PATH_TEMPLATE`` is pinned against ``config.js``.
PAGE_IMAGE_PATH_TEMPLATE = "/v1/documents/{document_id}/pages/{page}/image"

#: URI schemes a figure's image reference may use.
#:
#: A scheme, and nothing else is accepted, because the failure being designed
#: against is a *dangling* reference: a relative path written next to a
#: caption, resolvable from one working directory and not from another, so the
#: reader either sees nothing or sees whatever file happens to have that name.
#: A figure that says it has no image is a smaller lie than a figure whose
#: image silently 404s — and much smaller than one that resolves to a
#: *different* image, which is the case a bare filename eventually reaches.
FIGURE_IMAGE_SCHEMES = ("http", "https", "file", "artifact")


@dataclass(slots=True)
class Figure:
    """A figure: its caption, and where the image itself lives.

    The caption is the retrievable content. Docling folds a picture's caption
    into the figure rather than emitting it separately, so the two must be one
    element or the same sentence is counted twice against the same region of
    the page.

    **A figure's image is the page region, not a crop.** The schema keeps
    image *bytes* out — they live in the raw artifact, not in the canonical
    document — and the pipeline has no stage that extracts one. So
    ``image_uri`` is *not* how a figure points at its picture. It is reserved
    for a genuine, resolvable reference, and this build produces none: no
    adapter crops, so every figure arrives with it empty and says so.

    What a figure *can* honestly point at is the region of the page it
    occupies, and that is what :meth:`page_image_url` builds from the
    element's own locator. The distinction is the point of this class. A
    figure is either a caption with no image at all — a figure a reader can
    quote, cite and locate, and which says plainly that the picture itself was
    not extracted — or it has a real reference. It is never a caption with a
    path to nowhere, which is the shape a bare ``Element.text`` string invites
    and the reason this is a type.
    """

    caption: str = ""
    image_uri: str = ""

    def __post_init__(self) -> None:
        self.caption = str(self.caption).strip()
        self.image_uri = str(self.image_uri).strip()
        if self.image_uri and not self.has_resolvable_image:
            # Refused rather than carried. A reference this class cannot prove
            # resolves is a reference that will eventually be rendered next to
            # the wrong paragraph, and clearing it is recoverable (the caption
            # and the locator survive) while rendering it is not.
            self.image_uri = ""

    @property
    def has_image(self) -> bool:
        """Whether this figure carries a real image reference.

        Not the same as ``bool(image_uri)``, and the difference is the
        validation above: a caller can construct a ``Figure`` with a
        relative path, and this property is what tells it the value was
        refused.
        """
        return bool(self.image_uri)

    @property
    def has_resolvable_image(self) -> bool:
        """Whether ``image_uri`` names something a reader could open.

        A scheme is the test. ``artifact://`` is the scheme a stored blob gets
        from :mod:`paperwhisperer.infra.object_store`; ``file://`` and the two
        HTTP schemes are what a deployment or an export produces. Anything else
        is a path with no anchor.
        """
        if not self.image_uri:
            return False
        scheme, separator, _ = self.image_uri.partition("://")
        return bool(separator) and scheme.lower() in FIGURE_IMAGE_SCHEMES

    def page_image_url(self, locator: SourceLocator, document_id: str = "") -> str:
        """The rendered page this figure sits on, or ``""`` if it has none.

        The honest image reference for a figure in this build, and it is built
        from the element's locator rather than stored on the figure — because
        the locator is the thing that is already verified to resolve, and a
        second copy of "which page is this on" would be a second opinion free
        to disagree with the first.

        Empty when the locator does not name a page. A relative URL is returned
        rather than an absolute one, for the reason the evidence deep link is:
        an export that hard-codes a host is a link that works for exactly one
        person on exactly one machine.
        """
        if locator.kind != _PDF_PAGE or locator.page is None or not document_id:
            return ""
        from urllib.parse import quote as _urlquote

        return PAGE_IMAGE_PATH_TEMPLATE.format(
            document_id=_urlquote(str(document_id), safe=""),
            page=locator.page,
        )

    def to_dict(self) -> dict[str, Any]:
        # ``image_uri`` is pruned when empty, matching how ``latex`` and
        # ``page_number_label`` are pruned on ``Element``: no adapter emits
        # one, so writing a permanent empty key into every figure's payload is
        # bytes bought for nothing. An absent key is also the honest signal —
        # it is the same absence ``Figure.has_image`` reports.
        return _prune({"caption": self.caption, "image_uri": self.image_uri or None})

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Figure:
        return cls(
            caption=str(payload.get("caption") or ""),
            image_uri=str(payload.get("image_uri") or ""),
        )


# --------------------------------------------------------------------------
# Content units and elements
# --------------------------------------------------------------------------


@dataclass(slots=True)
class SourceUnit:
    """The smallest region a citation can point at — one page, slide or section.

    Elements carry their own locators; the unit is what an evidence request
    resolves to, because a reader wants "page 7", not "paragraph 43".
    """

    unit_id: str = field(default_factory=lambda: _new_id("unit"))
    locator: SourceLocator = field(default_factory=SourceLocator)
    label: str = ""
    text: str = ""
    order: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "locator": self.locator.to_dict(),
            "label": self.label,
            "text": self.text,
            "order": self.order,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SourceUnit:
        return cls(
            unit_id=str(payload.get("unit_id") or _new_id("unit")),
            locator=SourceLocator.from_dict(payload.get("locator") or {}),
            label=str(payload.get("label") or ""),
            text=str(payload.get("text") or ""),
            order=int(payload.get("order") or 0),
        )


@dataclass(slots=True)
class Table:
    """A table kept as a two-dimensional grid, never flattened to text.

    Flattening is what makes cross-column retrieval impossible later, so the
    grid survives the whole pipeline. ``html`` is derived on demand rather
    than stored, because a stored copy would drift from ``rows``.
    """

    rows: list[list[str]] = field(default_factory=list)
    header_rows: int = 0
    caption: str = ""

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def column_count(self) -> int:
        return max((len(row) for row in self.rows), default=0)

    @property
    def is_empty_shell(self) -> bool:
        """A grid whose cells are all blank — a table outline with no content.

        The quality gate flags these because they usually mean the parser
        found ruled lines and nothing else.
        """
        if not self.rows:
            return True
        return not any(cell.strip() for row in self.rows for cell in row)

    def to_markdown(self) -> str:
        """Render as a GitHub-flavoured markdown table."""
        if not self.rows:
            return ""
        width = self.column_count
        padded = [row + [""] * (width - len(row)) for row in self.rows]
        head = padded[0] if padded else []
        divider = ["---"] * width
        body = padded[1:]
        lines = [
            "| " + " | ".join(cell.replace("|", "\\|") for cell in head) + " |",
            "| " + " | ".join(divider) + " |",
        ]
        lines.extend(
            "| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |" for row in body
        )
        return "\n".join(lines)

    def to_html(self) -> str:
        """Render as escaped HTML for the evidence viewer."""
        from html import escape

        if not self.rows:
            return "<table></table>"

        head_rows = self.rows[: self.header_rows] or self.rows[:1]
        body_rows = self.rows[self.header_rows :] if self.header_rows else self.rows[1:]
        parts = ["<table>"]
        if head_rows:
            parts.append("<thead>")
            for row in head_rows:
                cells = "".join(f"<th>{escape(str(cell))}</th>" for cell in row)
                parts.append(f"<tr>{cells}</tr>")
            parts.append("</thead>")
        parts.append("<tbody>")
        for row in body_rows:
            cells = "".join(f"<td>{escape(str(cell))}</td>" for cell in row)
            parts.append(f"<tr>{cells}</tr>")
        parts.extend(["</tbody>", "</table>"])
        return "".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": [list(row) for row in self.rows],
            "header_rows": self.header_rows,
            "caption": self.caption,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Table:
        return cls(
            rows=[list(row) for row in payload.get("rows") or []],
            header_rows=int(payload.get("header_rows") or 0),
            caption=str(payload.get("caption") or ""),
        )


@dataclass(slots=True)
class Element:
    """One addressable piece of a parsed document.

    ``text`` is the plain rendering used for embedding and for the markdown
    projection. Rich payloads (a table grid, a LaTeX formula, a figure caption)
    live in ``table`` / ``formula`` / ``figure`` and are projected into ``text``
    at construction so the two can never disagree.

    The rich payloads are reachable two ways, and that is deliberate rather
    than redundant. Adapters build elements from flat arguments — a docling
    ``texts[i]`` entry becomes ``Element(role="formula", latex=...)`` — and
    changing that would mean rewriting every adapter's call site for a field
    they have no use for. So ``formula`` and ``figure`` are *derived and
    verified* in ``__post_init__``: whichever side the caller populated, both
    ends agree by the time the element exists. Storing both without
    synchronizing them would create a second source of truth for the same
    content, which is the failure mode this module is built to rule out, so the
    synchronization is unconditional and happens in exactly one place.
    """

    role: str = "text"
    text: str = ""
    locator: SourceLocator = field(default_factory=SourceLocator)
    unit_id: str = ""
    element_id: str = field(default_factory=lambda: _new_id("el"))
    order: int = 0
    heading_path: tuple[str, ...] = ()
    level: int | None = None
    table: Table | None = None
    latex: str = ""
    figure: Figure | None = None
    formula: Formula | None = None
    page_number_label: str = ""

    def __post_init__(self) -> None:
        if self.role not in ELEMENT_ROLES:
            self.role = "text"
        if isinstance(self.heading_path, list):
            self.heading_path = tuple(str(item) for item in self.heading_path)
        self._sync_formula()
        self._sync_figure()
        if self.table is not None and not self.text:
            self.text = self.table.to_markdown()
        if self.latex and not self.text:
            self.text = self.formula.rendered if self.formula else f"$${self.latex}$$"

    def _sync_formula(self) -> None:
        """Reconcile ``latex`` and ``formula`` into one consistent state.

        Two cases, one direction each. An adapter that only has LaTeX gets a
        ``Formula`` built from it; a caller that has a structured formula gets
        its LaTeX projected back onto the flat field, because every existing
        consumer — chunking, the eval runner, the prose projection — reads
        ``latex``. A structured formula always wins, because it carries
        information the flat field cannot hold and the flat field carries
        nothing the structured one lacks.

        The derived case runs :func:`infer_display` rather than assuming
        ``True``, because a formula whose LaTeX wraps itself in
        ``\\begin{equation}`` *is* a block and one that does not is not
        necessarily an inline formula — but it is certainly not an observed
        one either. Either way ``display_source`` records which of the three
        happened, which is the only way a reader can tell the difference
        between "the backend said so" and "we inferred it".
        """
        if self.formula is not None:
            self.latex = self.formula.latex
            return
        if self.latex.strip():
            source = infer_display(self.latex)
            self.formula = Formula(
                latex=self.latex,
                display=DEFAULT_DISPLAY_WHEN_UNKNOWN if source == "unknown" else True,
                display_source=source,
            )
            self.latex = self.formula.latex

    def _sync_figure(self) -> None:
        """Reconcile a figure's caption across ``text`` and ``figure``.

        A figure's caption is its ``text`` — that is what makes it retrievable
        — so a ``Figure`` with neither a caption nor an image reference says
        nothing that ``role="figure"`` does not already say, and deriving one
        would add an empty ``"figure": {}`` to every captionless figure's
        payload. Those are the figures the adapters emit on purpose: an
        addressable region with no text is still worth a citation, and the
        element is kept either way.

        So a ``Figure`` is derived only when there is something to record. The
        structured direction fills ``text`` from the caption, because every
        existing consumer — the prose projection, the search path — reads
        ``text`` and a figure whose caption lives only in the typed field is
        invisible to all of them.
        """
        if self.role != "figure":
            return
        if self.figure is None:
            if self.text:
                self.figure = Figure(caption=self.text)
            return
        caption = self.figure.caption or self.text
        if not self.text:
            self.text = caption
        self.figure = Figure(caption=caption, image_uri=self.figure.image_uri)

    @property
    def is_structural(self) -> bool:
        """Whether this element carries layout meaning rather than prose.

        Chunking keeps these whole; prose is merged up to a token budget.
        """
        return self.role in {"table", "formula", "figure"}

    @property
    def is_furniture(self) -> bool:
        """Running headers and footers, excluded from body text by default."""
        return self.role in {"header", "footer"}

    def to_dict(self) -> dict[str, Any]:
        return _prune({
            "element_id": self.element_id,
            "role": self.role,
            "text": self.text,
            "locator": self.locator.to_dict(),
            "unit_id": self.unit_id,
            "order": self.order,
            "heading_path": list(self.heading_path) or None,
            "level": self.level,
            "table": self.table.to_dict() if self.table else None,
            "latex": self.latex or None,
            "figure": self.figure.to_dict() if self.figure else None,
            "formula": self.formula.to_dict() if self.formula else None,
            "page_number_label": self.page_number_label or None,
        })

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Element:
        table_payload = payload.get("table")
        figure_payload = payload.get("figure")
        formula_payload = payload.get("formula")
        return cls(
            role=str(payload.get("role") or "text"),
            text=str(payload.get("text") or ""),
            locator=SourceLocator.from_dict(payload.get("locator") or {}),
            unit_id=str(payload.get("unit_id") or ""),
            element_id=str(payload.get("element_id") or _new_id("el")),
            order=int(payload.get("order") or 0),
            heading_path=tuple(payload.get("heading_path") or ()),
            level=_as_int(payload.get("level")),
            table=Table.from_dict(table_payload) if table_payload else None,
            latex=str(payload.get("latex") or ""),
            figure=Figure.from_dict(figure_payload) if figure_payload else None,
            formula=Formula.from_dict(formula_payload) if formula_payload else None,
            page_number_label=str(payload.get("page_number_label") or ""),
        )


# --------------------------------------------------------------------------
# Derived layer
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Chunk:
    """A retrieval unit: one or more elements with their heading context.

    Produced by ``rag.chunking`` in P1. Defined here because a chunk has to
    point back at elements through the same locator vocabulary, and keeping
    it beside ``Element`` makes that coupling visible.
    """

    chunk_id: str = field(default_factory=lambda: _new_id("chunk"))
    text: str = ""
    element_ids: tuple[str, ...] = ()
    locator: SourceLocator = field(default_factory=SourceLocator)
    heading_path: tuple[str, ...] = ()
    order: int = 0
    token_estimate: int = 0
    index_version: str = ""

    # The three fields below are all about the gap between what a chunk shows
    # and what the document actually said. They exist because a consumer that
    # cannot tell those apart will cite something that is not in the paper.

    #: Text prepended to ``text`` for retrieval context — the heading line and
    #: any lead-in sentence. **It is not part of the verbatim body.** A
    #: citation verifier that cannot separate the two will "find" a quote that
    #: exists only in text this product generated, and report a fabricated
    #: citation as verified. `rag.chunking` documents its own layout; this
    #: field is the machine-readable form of that contract, so the separation
    #: survives serialization instead of depending on a reader parsing the
    #: string.
    context_prefix: str = ""

    #: When one element is too large for the token budget it is emitted as
    #: several chunks. All of them carry the same ``locator`` and
    #: ``element_ids``, so without these two fields a consumer cannot address
    #: "the second of the three pieces" and every piece looks like the same
    #: source location. 0 / 0 means the chunk was not split.
    part_index: int = 0
    part_count: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.element_ids, list):
            self.element_ids = tuple(str(item) for item in self.element_ids)
        if isinstance(self.heading_path, list):
            self.heading_path = tuple(str(item) for item in self.heading_path)

    @property
    def verbatim_text(self) -> str:
        """The chunk's body without the generated context prefix.

        This is the only text a direct quote may be checked against. It equals
        ``text`` whenever no prefix was prepended, so the common case costs
        nothing.

        The blank line separating prefix from body is a join artefact, not
        content, so it is dropped too — otherwise the body's offsets would not
        line up with the document the quote came from, and every consumer that
        sliced ``verbatim_text`` by a range computed elsewhere would be off by
        the separator width.
        """
        if not self.context_prefix or not self.text.startswith(self.context_prefix):
            return self.text
        return self.text[len(self.context_prefix):].lstrip("\n")

    def to_dict(self) -> dict[str, Any]:
        return _prune({
            "chunk_id": self.chunk_id,
            "text": self.text,
            "element_ids": list(self.element_ids),
            "locator": self.locator.to_dict(),
            "heading_path": list(self.heading_path) or None,
            "order": self.order,
            "token_estimate": self.token_estimate,
            "index_version": self.index_version or None,
            "context_prefix": self.context_prefix or None,
            "part_index": self.part_index or None,
            "part_count": self.part_count or None,
        })

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Chunk:
        return cls(
            chunk_id=str(payload.get("chunk_id") or _new_id("chunk")),
            text=str(payload.get("text") or ""),
            element_ids=tuple(payload.get("element_ids") or ()),
            locator=SourceLocator.from_dict(payload.get("locator") or {}),
            heading_path=tuple(payload.get("heading_path") or ()),
            order=int(payload.get("order") or 0),
            token_estimate=int(payload.get("token_estimate") or 0),
            index_version=str(payload.get("index_version") or ""),
            context_prefix=str(payload.get("context_prefix") or ""),
            part_index=int(payload.get("part_index") or 0),
            part_count=int(payload.get("part_count") or 0),
        )


@dataclass(slots=True)
class Evidence:
    """A retrieved passage, already tied to where it came from.

    ``is_direct_quote`` is set only after the quote has been checked against
    the source text, never from the model's own claim to be quoting. That
    distinction is the entire point of the citation verifier, so the flag
    lives on the value that verification writes to.
    """

    evidence_id: str = field(default_factory=lambda: _new_id("ev"))
    chunk_id: str = ""
    text: str = ""
    locator: SourceLocator = field(default_factory=SourceLocator)
    unit_id: str = ""
    heading_path: tuple[str, ...] = ()
    score: float = 0.0
    is_direct_quote: bool = False
    source_parser: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.heading_path, list):
            self.heading_path = tuple(str(item) for item in self.heading_path)

    @property
    def citation_label(self) -> str:
        """A short human label such as ``p. 7`` or ``slide 3``."""
        locator = self.locator
        if locator.kind == _PDF_PAGE and locator.page is not None:
            return f"p. {locator.page}"
        if locator.kind == _SLIDE and locator.slide is not None:
            return f"slide {locator.slide}"
        if locator.kind == _SHEET_RANGE and locator.sheet:
            return f"sheet {locator.sheet}"
        if locator.kind == _DOCX_ELEMENT and locator.paragraph is not None:
            return f"paragraph {locator.paragraph}"
        if self.heading_path:
            return " › ".join(self.heading_path[-2:])
        return "document"

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "locator": self.locator.to_dict(),
            "unit_id": self.unit_id,
            "heading_path": list(self.heading_path),
            "score": self.score,
            "is_direct_quote": self.is_direct_quote,
            "source_parser": self.source_parser,
            "citation_label": self.citation_label,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Evidence:
        return cls(
            evidence_id=str(payload.get("evidence_id") or _new_id("ev")),
            chunk_id=str(payload.get("chunk_id") or ""),
            text=str(payload.get("text") or ""),
            locator=SourceLocator.from_dict(payload.get("locator") or {}),
            unit_id=str(payload.get("unit_id") or ""),
            heading_path=tuple(payload.get("heading_path") or ()),
            score=float(payload.get("score") or 0.0),
            is_direct_quote=bool(payload.get("is_direct_quote")),
            source_parser=str(payload.get("source_parser") or ""),
        )


@dataclass(slots=True)
class Fact:
    """One extracted claim, with where it came from and how sure we are.

    ``extraction_status`` is what lets the UI say "not stated in this paper"
    instead of rendering an empty section that reads like an oversight.
    """

    field_name: str = ""
    value: str = ""
    evidence_ids: tuple[str, ...] = ()
    extraction_status: str = "not_found"
    conflicting_values: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.evidence_ids, list):
            self.evidence_ids = tuple(str(item) for item in self.evidence_ids)
        if isinstance(self.conflicting_values, list):
            self.conflicting_values = tuple(str(item) for item in self.conflicting_values)
        if self.extraction_status not in EXTRACTION_STATUSES:
            self.extraction_status = "not_found"
        if self.value and self.extraction_status == "not_found":
            self.extraction_status = "extracted"

    @property
    def is_grounded(self) -> bool:
        return bool(self.value) and bool(self.evidence_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_name": self.field_name,
            "value": self.value,
            "evidence_ids": list(self.evidence_ids),
            "extraction_status": self.extraction_status,
            "conflicting_values": list(self.conflicting_values),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Fact:
        return cls(
            field_name=str(payload.get("field_name") or ""),
            value=str(payload.get("value") or ""),
            evidence_ids=tuple(payload.get("evidence_ids") or ()),
            extraction_status=str(payload.get("extraction_status") or "not_found"),
            conflicting_values=tuple(payload.get("conflicting_values") or ()),
        )


@dataclass(slots=True)
class PaperFacts:
    """The single structured extraction every analysis card renders from.

    One extraction, many cards: Overview, Brief and Mindmap all read these
    fields, so a conclusion cannot contradict itself between two cards.
    """

    research_question: Fact = field(default_factory=lambda: Fact(field_name="research_question"))
    contributions: list[Fact] = field(default_factory=list)
    method: Fact = field(default_factory=lambda: Fact(field_name="method"))
    datasets: list[Fact] = field(default_factory=list)
    metrics: list[Fact] = field(default_factory=list)
    main_results: list[Fact] = field(default_factory=list)
    limitations: list[Fact] = field(default_factory=list)
    reproducibility: Fact = field(default_factory=lambda: Fact(field_name="reproducibility"))
    extraction_model: str = ""
    prompt_version: str = ""
    evidence_version: str = ""

    def all_facts(self) -> list[Fact]:
        every = [self.research_question, self.method, self.reproducibility]
        every.extend(self.contributions)
        every.extend(self.datasets)
        every.extend(self.metrics)
        every.extend(self.main_results)
        every.extend(self.limitations)
        return every

    @property
    def coverage(self) -> float:
        """Share of facts that carry both a value and at least one citation."""
        facts = self.all_facts()
        if not facts:
            return 0.0
        return sum(1 for fact in facts if fact.is_grounded) / len(facts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "research_question": self.research_question.to_dict(),
            "contributions": [fact.to_dict() for fact in self.contributions],
            "method": self.method.to_dict(),
            "datasets": [fact.to_dict() for fact in self.datasets],
            "metrics": [fact.to_dict() for fact in self.metrics],
            "main_results": [fact.to_dict() for fact in self.main_results],
            "limitations": [fact.to_dict() for fact in self.limitations],
            "reproducibility": self.reproducibility.to_dict(),
            "extraction_model": self.extraction_model,
            "prompt_version": self.prompt_version,
            "evidence_version": self.evidence_version,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PaperFacts:
        def one(key: str) -> Fact:
            raw = payload.get(key)
            return Fact.from_dict(raw) if isinstance(raw, dict) else Fact(field_name=key)

        def many(key: str) -> list[Fact]:
            raw_items = payload.get(key) or []
            return [Fact.from_dict(item) for item in raw_items if isinstance(item, dict)]

        return cls(
            research_question=one("research_question"),
            contributions=many("contributions"),
            method=one("method"),
            datasets=many("datasets"),
            metrics=many("metrics"),
            main_results=many("main_results"),
            limitations=many("limitations"),
            reproducibility=one("reproducibility"),
            extraction_model=str(payload.get("extraction_model") or ""),
            prompt_version=str(payload.get("prompt_version") or ""),
            evidence_version=str(payload.get("evidence_version") or ""),
        )


# --------------------------------------------------------------------------
# The document
# --------------------------------------------------------------------------


@dataclass(slots=True)
class ParsedDocument:
    """The canonical result of parsing one file.

    Adapters produce this; the router, quality gate and every downstream
    stage read only this. ``to_markdown`` and ``to_full_text`` are the two
    projections the existing analysis pipeline needs, and they deliberately
    differ: the pipeline's prompts expect clean prose, so ``to_full_text``
    skips furniture and does not collapse table structure into prose.
    """

    schema_version: str = SCHEMA_VERSION
    document_id: str = field(default_factory=lambda: _new_id("doc"))
    source_filename: str = ""
    media_type: str = ""
    sha256: str = ""
    page_count: int = 0
    elements: list[Element] = field(default_factory=list)
    units: list[SourceUnit] = field(default_factory=list)
    parser_name: str = ""
    parser_version: str = ""
    options_hash: str = ""
    created_at: str = ""
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.schema_version:
            self.schema_version = SCHEMA_VERSION
        self.rejoin_split_formulas()
        # Reading order is fixed here, once, so no consumer has to remember
        # that locators sort per format.
        self.elements.sort(key=lambda element: (element.order, element.locator.sort_key))
        for index, element in enumerate(self.elements):
            element.order = index

    def rejoin_split_formulas(self) -> int:
        """Merge a formula the parser cut across two elements. Returns the count.

        A display equation laid out over two lines arrives as two formula
        elements, and each half on its own is not LaTeX: ``\\begin{align} x`` has
        no ``\\end``, and ``y \\end{align}`` opens nothing. Both render as an
        error, so an export of such a document shows the reader two broken
        lines where the paper has one equation — a corruption the reader cannot
        see the origin of.

        The merge is **not** done by deleting the second element. The second
        element's id is kept on the merged formula, so a citation that pointed
        at the second half still resolves, and a warning naming both ids goes on
        the document. Dropping the tail instead would leave a formula that looks
        complete and is not, which is the worse of the two failures: a reader
        who trusts it is worse off than one who sees it flagged.

        Structural only — :func:`~paperwhisperer.ingestion.latex.join_latex`
        rejoins a half that ends inside an unclosed group and refuses
        everything else. Two genuinely consecutive equations stay two
        equations, which is why this cannot quietly fuse a document's
        mathematics into one unreadable run.

        Called from ``__post_init__`` rather than offered as an opt-in, because
        the failure it prevents is invisible in the output and an optional
        repair is a repair nobody runs. It is a no-op on every document whose
        formulas are whole, which is nearly all of them.
        """
        if not self.elements:
            return 0
        merged: list[Element] = []
        rejoin_count = 0
        for element in self.elements:
            previous = merged[-1] if merged else None
            if (
                previous is not None
                and previous.role == "formula"
                and element.role == "formula"
                and previous.latex
                and element.latex
            ):
                joined = join_latex(previous.latex, element.latex)
                if joined is not None:
                    previous.formula = Formula(
                        latex=joined,
                        display=previous.formula.display if previous.formula else True,
                        display_source=(
                            previous.formula.display_source if previous.formula else "unknown"
                        ),
                    )
                    previous.latex = joined
                    if not previous.text:
                        previous.text = previous.formula.rendered
                    rejoin_count += 1
                    self.warnings.append(
                        f"公式被解析器切成了两段，已在第 {element.locator.page or '?'} 页"
                        f"重新合并（{previous.element_id} + {element.element_id}）。"
                    )
                    continue
            merged.append(element)
        if rejoin_count:
            self.elements = merged
        return rejoin_count

    @property
    def body_elements(self) -> list[Element]:
        """Prose and structural content, with running furniture removed."""
        return [element for element in self.elements if not element.is_furniture]

    @property
    def char_count(self) -> int:
        return sum(len(element.text) for element in self.body_elements)

    @property
    def locator_coverage(self) -> float:
        """Share of body elements that can be pointed at.

        P0 exits at 100% for documents that parsed successfully; anything
        below that is a visible degradation, not a silent one.
        """
        body = self.body_elements
        if not body:
            return 0.0
        located = sum(1 for element in body if element.locator.is_resolvable)
        return located / len(body)

    def iter_reading_order(self) -> list[Element]:
        return sorted(self.body_elements,
                      key=lambda element: (element.order, element.locator.sort_key))

    def to_full_text(self) -> str:
        """Plain prose for the analysis prompts, matching the legacy loader.

        Tables become their markdown grid and formulas their LaTeX so the
        prompt sees the real content rather than an empty gap.
        """
        blocks: list[str] = []
        for element in self.iter_reading_order():
            text = element.text.strip()
            if not text:
                continue
            if element.role in {"title", "heading"} and element.level in (None, 1):
                blocks.append(f"# {text}")
            elif element.role == "heading":
                marks = "#" * min(6, max(2, element.level or 2))
                blocks.append(f"{marks} {text}")
            else:
                blocks.append(text)
        return "\n\n".join(blocks).strip()

    def to_markdown(self) -> str:
        """Full markdown including tables, for export."""
        return self.to_full_text()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "document_id": self.document_id,
            "source_filename": self.source_filename,
            "media_type": self.media_type,
            "sha256": self.sha256,
            "page_count": self.page_count,
            "elements": [element.to_dict() for element in self.elements],
            "units": [unit.to_dict() for unit in self.units],
            "parser_name": self.parser_name,
            "parser_version": self.parser_version,
            "options_hash": self.options_hash,
            "created_at": self.created_at,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ParsedDocument:
        version = str(payload.get("schema_version") or SCHEMA_VERSION)
        # The shape is checked before the major is compared, and the order
        # matters. ``"1"`` splits to ``["1"]``, whose first element equals the
        # supported major, so a major-only comparison accepts a document whose
        # shape cannot be established at all — the compatibility gate passing
        # for a string it never actually understood. A version this build
        # cannot parse is a document whose locators may mean anything, and
        # nothing downstream recomputes them, so it is refused. This is the
        # same rule :class:`~paperwhisperer.ingestion.contracts.ParsedDocumentContract`
        # enforces; the two agreeing is the point.
        if not SCHEMA_VERSION_PATTERN.match(version):
            raise ValueError(
                f"Malformed parsed-document schema version: {version!r} "
                f"(expected MAJOR.MINOR, this build reads {SCHEMA_VERSION})"
            )
        if version.split(".")[0] != SCHEMA_VERSION.split(".")[0]:
            raise ValueError(
                f"Unsupported parsed-document schema version: {version} "
                f"(this build reads {SCHEMA_VERSION})"
            )
        return cls(
            schema_version=version,
            document_id=str(payload.get("document_id") or _new_id("doc")),
            source_filename=str(payload.get("source_filename") or ""),
            media_type=str(payload.get("media_type") or ""),
            sha256=str(payload.get("sha256") or ""),
            page_count=int(payload.get("page_count") or 0),
            elements=[Element.from_dict(item) for item in payload.get("elements") or []],
            units=[SourceUnit.from_dict(item) for item in payload.get("units") or []],
            parser_name=str(payload.get("parser_name") or ""),
            parser_version=str(payload.get("parser_version") or ""),
            options_hash=str(payload.get("options_hash") or ""),
            created_at=str(payload.get("created_at") or ""),
            warnings=[str(item) for item in payload.get("warnings") or []],
        )

    def build_units(self) -> list[SourceUnit]:
        """Derive one unit per page/slide/section from the element locators.

        Adapters that can supply units directly should; this exists so a
        weaker adapter still yields evidence that resolves to a page.
        """
        grouped: dict[tuple, list[Element]] = {}
        for element in self.body_elements:
            locator = element.locator
            if locator.kind == _PDF_PAGE:
                key = (_PDF_PAGE, locator.page)
            elif locator.kind == _SLIDE:
                key = (_SLIDE, locator.slide)
            elif locator.kind == _DOCX_ELEMENT:
                key = (_DOCX_ELEMENT, locator.paragraph)
            elif locator.kind == _TXT_OFFSET:
                key = (_TXT_OFFSET, locator.offset_start)
            else:
                continue
            grouped.setdefault(key, []).append(element)

        units: list[SourceUnit] = []
        for order, members in enumerate(sorted(grouped.values(), key=_unit_group_key)):
            first = members[0]
            label = first.page_number_label or Evidence(locator=first.locator).citation_label
            units.append(
                SourceUnit(
                    locator=first.locator,
                    label=label,
                    text="\n\n".join(
                        member.text.strip() for member in members if member.text.strip()
                    ),
                    order=order,
                )
            )
        self.units = units
        return units


def is_supported_schema_version(version: str) -> bool:
    """Whether a serialized document can be read by this build.

    Both halves of the answer have to hold. The major must match, and the
    string must be a version at all: ``"1"`` happens to satisfy the major
    comparison by accident, because ``"1".split(".")[0]`` is ``"1"``, which is
    precisely the failure that made a version check pass for a string it never
    understood. The predicate answers "can this be read", and something whose
    shape is unknown cannot be read.
    """
    text = str(version)
    if not SCHEMA_VERSION_PATTERN.match(text):
        return False
    return text.split(".")[0] == SCHEMA_VERSION.split(".")[0]
