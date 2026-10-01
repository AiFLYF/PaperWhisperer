"""Pydantic contracts for the untrusted edges of the ingestion pipeline.

:mod:`paperwhisperer.ingestion.schema` is the canonical representation and it
stays a set of stdlib dataclasses, because a parser adapter builds thousands of
elements per document and validating data the process itself just produced is
cost without a boundary. This module is the other half of that decision: the
places where data arrives from *outside* the process, where a malformed value is
somebody else's mistake and the only question is whether to absorb it or refuse
it. Here the answer is always refuse.

The distinction that makes this worth a module rather than a pile of validators
is **who is allowed to be wrong, silently**. Everything the schema does is
lossy-tolerant by design: an unknown role degrades to ``text``, an unparseable
page number becomes ``None``, an unrecognised locator kind becomes ``unknown``.
Those are correct answers when the input is a docling JSON blob from this
process's own adapter, minutes ago, and re-parsing is cheap. They are the wrong
answer at a trust boundary, where the same silent coercion turns into a
citation that points confidently at a location the document never had. A
``pdf_page`` locator whose page number came back as ``"seven"`` degrading to
``page=None`` is not a visible degradation — it is a quote attributed to a page
it was never on, and the user has no way to notice.

**The boundaries these contracts guard**

``ParsedDocumentContract``
    The parse cache on disk and any JSON document read back from storage
    (``ParseCache._read_disk``). Written by a previous process, possibly a
    previous build, possibly a partially-flushed file. This is the boundary
    where silent coercion is most dangerous and least recoverable: the result
    is a document that looks fine and cites the wrong pages, and nothing
    recomputes it.

``ElementContract`` and everything it nests
    Adapter output, on the way into the gateway. The gateway deliberately
    propagates no exceptions — a parser failing is a degraded parse, not a
    500 — so an adapter that emits a structurally broken element is exactly the
    case where validation has to happen before the router, not inside it.

``EvidenceContract``
    The evidence API response body. This is the one that goes straight to a
    reader's screen as a clickable citation, and it is assembled from stored
    retrieval results rather than from a live parse, so its locators have no
    later stage that could catch them.

``PaperFactsContract`` and ``FactContract``
    Model-generated structured output. The extraction comes from an LLM, which
    is the least trustworthy producer in the whole pipeline, and the whole point
    of ``extraction_status`` is to let the UI say "not stated" — a status the
    model can freely get wrong is a status the UI cannot rely on.

``UploadRequestContract``
    The user-supplied file metadata at the API edge. A filename extension and a
    declared media type are the only things a user controls before parsing, and
    they are what the gateway routes on.

``ChunkContract``
    Persisted retrieval units read back from the index. Same class of risk as
    the document cache: a chunk whose locator is subtly wrong sends a reader to
    the wrong page for a passage that is otherwise perfectly retrieved.

**What "strict" means here, and where it stops**

``extra="forbid"`` everywhere. A field this build does not recognise in a
document written by another build is a version mismatch that the major-version
check cannot see, and defaulting it away is how a locator written by a newer
build loses its discriminating field and starts meaning something else. The
cost is that a forward-compatible minor addition has to be added to the
contracts in the same commit that adds it to the schema, which is the correct
pressure: the contract is a promise about what this build understands.

Locator validation is the strictest part, and specifically checks the fields
whose corruption is invisible:

* a ``pdf_page`` must carry a page number, and an ``offset_start`` must not
  exceed ``offset_end`` — a reversed span resolves to a real-looking region
  that highlights the wrong text rather than failing;
* a bbox must be finite, ordered, and non-degenerate, because ``width`` and
  ``height`` take an absolute value, so a rectangle built from swapped
  coordinates is indistinguishable from a correct one downstream;
* a bbox's ``relative`` provenance is carried through rather than dropped: a
  mirrored rectangle is *correctly flagged* as approximate, and the flag is
  the only thing that lets the evidence viewer refuse to draw it;
* a locator's *other* fields are rejected when its ``kind`` does not use them.
  A ``pdf_page`` carrying a ``slide`` number means the adapter mixed two
  vocabularies; every consumer reads the field its ``kind`` implies and would
  never look at the stray one.

The obvious alternative — reusing the dataclasses' own tolerant
``from_dict`` and adding a ``validate=True`` flag — is rejected because it
makes the strictness a property each call site opts into, and a call site that
forgets is indistinguishable from one that chose tolerance. Importing this
module at all is the opt-in.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .schema import (
    DEFAULT_DISPLAY_WHEN_UNKNOWN,
    ELEMENT_ROLES,
    EXTRACTION_STATUSES,
    FIGURE_IMAGE_SCHEMES,
    FORMULA_DISPLAY_SOURCES,
    LOCATOR_KINDS,
    SCHEMA_VERSION,
    SCHEMA_VERSION_PATTERN,
    BoundingBox,
    Chunk,
    Element,
    Evidence,
    Fact,
    Figure,
    Formula,
    PaperFacts,
    ParsedDocument,
    SourceLocator,
    SourceUnit,
    Table,
    is_supported_schema_version,
)

#: ``MAJOR.MINOR``, and nothing else. A version string that is not this shape
#: cannot be compared against ``SCHEMA_VERSION``, so accepting one would mean
#: guessing what the writer meant. Imported from :mod:`schema` rather than
#: recompiled: the dataclass reader enforces the same rule (and once did not,
#: which is how the two drifted), and a second copy of the pattern is a second
#: opinion waiting to diverge again.
_SCHEMA_VERSION_PATTERN = SCHEMA_VERSION_PATTERN

#: A ``pdf_page`` number. Zero is not a page, and neither is anything
#: negative — both are what an off-by-one in a converter produces, and both
#: resolve to a real-looking citation pointing one before the start.
_PAGE_MIN = 1

#: Which positional field each locator kind is required to carry. ``unknown``
#: is absent from the table on purpose: it is the honest spelling of "this
#: element could not be located", and requiring a position for it would force
#: adapters to fabricate one — the exact failure the kind exists to prevent.
_REQUIRED_FIELD_BY_KIND = {
    "pdf_page": "page",
    "slide": "slide",
    "sheet_range": "sheet",
    "docx_element": "paragraph",
    "txt_offset": "offset_start",
}

#: The positional fields belong to exactly one kind. A locator carrying any
#: other kind's field has been assembled from two vocabularies, and the stray
#: field is invisible to every consumer that dispatches on ``kind``.
_FIELDS_BY_KIND = {
    "pdf_page": {"page", "bbox"},
    "slide": {"slide", "element"},
    "sheet_range": {"sheet", "cell_range"},
    "docx_element": {"paragraph", "heading_path"},
    "txt_offset": {"offset_start", "offset_end"},
    "unknown": set(),
}

_ALL_POSITIONAL_FIELDS = set().union(*_FIELDS_BY_KIND.values()) - {"heading_path"}

#: Extension → IANA media type, the same set the gateway routes on. Copied
#: rather than imported from ``gateway`` so this module stays free of the
#: parser registry and the config: a contract that pulls in the registry to
#: check a string cannot be imported by the thing deciding whether to invoke
#: the registry. Kept as a module constant rather than a class attribute
#: because Pydantic reads an underscore-prefixed class attribute as a private
#: attribute, not a table.
_UPLOAD_MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".txt": "text/plain",
}

#: Every media type a client may legitimately send, mapped to the canonical
#: registration. ``application/x-pdf`` is here because browsers send it and an
#: adapter comparing against the IANA spelling will not match it, which is how a
#: PDF silently takes the plain-text path.
_UPLOAD_MEDIA_TYPE_ALIASES = {
    "application/pdf": "application/pdf",
    "application/x-pdf": "application/pdf",
    "text/plain": "text/plain",
    "text/x-python": "text/plain",
    "text/markdown": "text/plain",
    _UPLOAD_MEDIA_TYPES[".docx"]: _UPLOAD_MEDIA_TYPES[".docx"],
    _UPLOAD_MEDIA_TYPES[".pptx"]: _UPLOAD_MEDIA_TYPES[".pptx"],
}


class _Contract(BaseModel):
    """Shared configuration for every contract in this module.

    ``frozen`` because a contract is a validated snapshot taken at a boundary;
    a caller that wants to adjust a validated value should re-validate, not
    mutate, or the guarantee becomes a claim about a state nobody checked.
    ``extra="forbid"`` for the forward-compatibility reason in the module
    docstring.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------
# Locators
# --------------------------------------------------------------------------


class BoundingBoxContract(_Contract):
    """A rectangle that is safe to hand to ``pdf.js``.

    The dataclass takes an unordered rectangle on purpose: adapters normalize
    with ``min``/``max`` and every consumer computes ``width`` as an absolute
    value, so a mirrored box from a buggy converter still yields a positive
    width. At a boundary that tolerance has to go. The mirrored box survives
    every local check and is then used to draw a highlight, and the highlight is
    wrong in a way nothing downstream can report. So the contract requires
    ``x0 <= x1`` and ``y0 <= y1`` and refuses the rectangle.

    ``relative`` is carried, not checked. It is provenance, not a validity
    claim: a relative rectangle is *correctly flagged* as approximate, and the
    contract's job is to make sure the flag survives the trip so that
    :meth:`~paperwhisperer.ingestion.schema.BoundingBox.to_query` can still
    refuse to render it. What must not happen is a relative box arriving at a
    consumer that believes it is absolute, and that is exactly what dropping
    the field here would cause — so ``from_dataclass``/``to_dataclass`` are
    required to round-trip it, and the equality-based round-trip tests in
    ``tests/test_ingestion_contracts.py`` hold it to that.
    """

    x0: float
    y0: float
    x1: float
    y1: float
    relative: bool = False

    @field_validator("x0", "y0", "x1", "y1")
    @classmethod
    def _must_be_finite(cls, value: float) -> float:
        # NaN survives every comparison in Python, including the ordering check
        # below, so it has to be excluded explicitly or a NaN coordinate
        # produces a bbox whose width is NaN and whose highlight covers nothing.
        if not math.isfinite(value):
            raise ValueError(f"bounding-box coordinate must be finite, got {value!r}")
        return value

    @model_validator(mode="after")
    def _must_be_ordered(self) -> BoundingBoxContract:
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError(
                f"bounding box is not ordered: ({self.x0}, {self.y0}) to "
                f"({self.x1}, {self.y1}); expected x0 <= x1 and y0 <= y1"
            )
        if self.x1 == self.x0 and self.y1 == self.y0:
            # A zero-area rectangle is not a highlight, it is a stray mark on
            # the page. It is the signature of a converter that emitted a
            # default box when it found no geometry.
            raise ValueError("bounding box is degenerate: zero width and height")
        return self

    def to_dataclass(self) -> BoundingBox:
        return BoundingBox(
            x0=self.x0,
            y0=self.y0,
            x1=self.x1,
            y1=self.y1,
            relative=self.relative,
        )

    @classmethod
    def from_dataclass(cls, box: BoundingBox) -> BoundingBoxContract:
        return cls(
            x0=box.x0,
            y0=box.y0,
            x1=box.x1,
            y1=box.y1,
            relative=bool(box.relative),
        )


class SourceLocatorContract(_Contract):
    """Where a citation points, validated as the thing a reader will click.

    This is the contract that matters most in the module. Everything else
    degrades visibly; a locator that is subtly wrong produces a citation that
    looks completely correct and is not.
    """

    kind: str
    page: int | None = None
    bbox: BoundingBoxContract | None = None
    slide: int | None = None
    element: int | None = None
    sheet: str | None = None
    cell_range: str | None = None
    paragraph: int | None = None
    heading_path: tuple[str, ...] = ()
    offset_start: int | None = None
    offset_end: int | None = None

    @field_validator("kind")
    @classmethod
    def _must_be_a_known_kind(cls, value: str) -> str:
        # The dataclass downgrades an unknown kind to ``unknown``. Here the
        # whole point is that the loss is visible to the writer, and the only
        # way it is visible is if the validation fails instead.
        if value not in LOCATOR_KINDS:
            raise ValueError(
                f"unknown locator kind {value!r}; expected one of {', '.join(LOCATOR_KINDS)}"
            )
        return value

    @field_validator("page", "slide")
    @classmethod
    def _must_be_positive(cls, value: int | None) -> int | None:
        # ``page`` and ``slide`` are numbers a reader sees: "p. 7", "slide 3".
        # Zero is not either of them — it is what an off-by-one in a converter
        # produces, and it resolves to a real-looking citation pointing one
        # before the start. ``element`` is deliberately not in this group; see
        # :meth:`_must_not_be_negative`.
        if value is not None and value < _PAGE_MIN:
            raise ValueError(f"index must be >= {_PAGE_MIN}, got {value}")
        return value

    @field_validator("paragraph", "offset_start", "offset_end", "element")
    @classmethod
    def _must_not_be_negative(cls, value: int | None) -> int | None:
        # ``element`` is a *shape index* — the position of a shape within a
        # slide — not a number anyone reads. The schema documents it as one and
        # every adapter builds it with ``enumerate``, so the first shape on a
        # slide is index 0. Requiring >= 1 here would refuse the first shape of
        # every real presentation, which is a contract inventing a bug the
        # producer does not have. The two index families differ in what they
        # mean, and the validator is where that difference is recorded:
        # ``page``/``slide`` are 1-based because a reader counts pages from one,
        # ``paragraph``/``element``/``offset_*`` are 0-based because they are
        # array positions.
        if value is not None and value < 0:
            raise ValueError(f"index must be >= 0, got {value}")
        return value

    @field_validator("heading_path", mode="before")
    @classmethod
    def _accepts_a_json_list(cls, value: Any) -> Any:
        # A heading path arrives as a JSON array. Rejecting the list outright
        # would push the tuple conversion onto every caller, for a shape the
        # serialized form has always used.
        return tuple(value) if isinstance(value, list) else value

    @field_validator("sheet")
    @classmethod
    def _sheet_must_be_named(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("sheet name must not be blank")
        return value

    @model_validator(mode="after")
    def _fields_must_match_the_kind(self) -> SourceLocatorContract:
        required = _REQUIRED_FIELD_BY_KIND.get(self.kind)
        if required is not None:
            value = getattr(self, required)
            if value is None or (isinstance(value, str) and not value.strip()):
                raise ValueError(
                    f"a {self.kind!r} locator must carry {required!r}; "
                    f"use kind='unknown' when the position is genuinely lost"
                )

        permitted = _FIELDS_BY_KIND.get(self.kind, set())
        for name in sorted(_ALL_POSITIONAL_FIELDS - permitted):
            if getattr(self, name) is not None:
                raise ValueError(
                    f"field {name!r} does not belong to a {self.kind!r} locator"
                )

        if self.offset_start is not None and self.offset_end is not None and (
            # A reversed span still resolves to a region, and that region is
            # somewhere a reader did not highlight from.
            self.offset_end < self.offset_start
        ):
            raise ValueError(
                f"offset_end ({self.offset_end}) precedes offset_start "
                f"({self.offset_start})"
            )
        return self

    def to_dataclass(self) -> SourceLocator:
        return SourceLocator(
            kind=self.kind,
            page=self.page,
            bbox=self.bbox.to_dataclass() if self.bbox else None,
            slide=self.slide,
            element=self.element,
            sheet=self.sheet,
            cell_range=self.cell_range,
            paragraph=self.paragraph,
            heading_path=self.heading_path,
            offset_start=self.offset_start,
            offset_end=self.offset_end,
        )

    @classmethod
    def from_dataclass(cls, locator: SourceLocator) -> SourceLocatorContract:
        return cls(
            kind=locator.kind,
            page=locator.page,
            bbox=BoundingBoxContract.from_dataclass(locator.bbox) if locator.bbox else None,
            slide=locator.slide,
            element=locator.element,
            sheet=locator.sheet,
            cell_range=locator.cell_range,
            paragraph=locator.paragraph,
            heading_path=tuple(locator.heading_path),
            offset_start=locator.offset_start,
            offset_end=locator.offset_end,
        )


# --------------------------------------------------------------------------
# Rich payloads
# --------------------------------------------------------------------------


class FormulaContract(_Contract):
    """LaTeX plus the display/inline distinction a bare string cannot hold.

    ``display_source`` is validated against the same enumeration the
    dataclass uses, and the two agreeing is the point: this is the edge where
    a value written by an older build, by another process or by a hand-edited
    file arrives, and a formula whose provenance reads ``"guessed"`` when the
    producer actually observed it is exactly the kind of quiet wrongness this
    project is arranging itself against. A document that claims
    ``display_source: "backend"`` when nothing set it is a document whose
    "these are display equations" table is a work of fiction.
    """

    latex: str
    display: bool = DEFAULT_DISPLAY_WHEN_UNKNOWN
    display_source: str = "unknown"

    @field_validator("latex")
    @classmethod
    def _must_be_non_blank(cls, value: str) -> str:
        if not value.strip():
            # An empty formula counts as a *survived* formula in the eval
            # runner's `formula_latex_ratio`, so accepting one here makes that
            # metric quietly wrong.
            raise ValueError("a formula must carry non-empty LaTeX")
        return value.strip()

    @field_validator("display_source")
    @classmethod
    def _source_must_be_known(cls, value: str) -> str:
        if value not in FORMULA_DISPLAY_SOURCES:
            raise ValueError(
                f"unknown formula display_source {value!r}; "
                f"expected one of {', '.join(FORMULA_DISPLAY_SOURCES)}"
            )
        return value

    def to_dataclass(self) -> Formula:
        return Formula(latex=self.latex, display=self.display, display_source=self.display_source)

    @classmethod
    def from_dataclass(cls, formula: Formula) -> FormulaContract:
        return cls(
            latex=formula.latex, display=formula.display, display_source=formula.display_source
        )


class FigureContract(_Contract):
    """A caption and an optional image reference.

    A blank caption is accepted. The adapters emit a figure with no caption on
    purpose — the addressable region is still worth a citation — and refusing it
    here would mean refusing the parse, which is a worse outcome than a figure
    the search path finds no text in.

    **A non-resolvable ``image_uri`` is refused rather than cleared.** The
    dataclass clears it, because there the value came from this process moments
    ago and the cheapest correct response is to not carry it. Here the value
    came from outside — a JSON file, another process — and silently dropping it
    would turn "your figure references an image that does not resolve" into a
    parse that reports no problem at all. The two layers answer differently on
    purpose, and the caller can tell which one it was handed.
    """

    caption: str = ""
    image_uri: str = ""

    @field_validator("image_uri")
    @classmethod
    def _image_must_be_resolvable(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        probe = Figure(caption="", image_uri=text)
        if not probe.has_resolvable_image:
            raise ValueError(
                f"image_uri {text!r} has no resolvable scheme; expected one of "
                f"{', '.join(scheme + '://' for scheme in FIGURE_IMAGE_SCHEMES)}. "
                "A relative path renders next to the wrong figure once the "
                "working directory changes."
            )
        return text

    def to_dataclass(self) -> Figure:
        return Figure(caption=self.caption, image_uri=self.image_uri)

    @classmethod
    def from_dataclass(cls, figure: Figure) -> FigureContract:
        return cls(caption=figure.caption, image_uri=figure.image_uri)


class TableContract(_Contract):
    """A two-dimensional grid.

    ``header_rows`` is bounded by the row count because ``to_html`` and
    ``to_markdown`` slice on it: a header count larger than the grid renders the
    whole table into a ``<thead>``, which reads as a table where every row is a
    column label.
    """

    rows: list[list[str]] = Field(default_factory=list)
    header_rows: int = 0
    caption: str = ""

    @model_validator(mode="after")
    def _header_rows_must_fit(self) -> TableContract:
        if self.header_rows < 0:
            raise ValueError(f"header_rows must be >= 0, got {self.header_rows}")
        if self.header_rows > len(self.rows):
            raise ValueError(
                f"header_rows ({self.header_rows}) exceeds the row count "
                f"({len(self.rows)})"
            )
        return self

    def to_dataclass(self) -> Table:
        return Table(
            rows=[list(row) for row in self.rows],
            header_rows=self.header_rows,
            caption=self.caption,
        )

    @classmethod
    def from_dataclass(cls, table: Table) -> TableContract:
        return cls(
            rows=[list(row) for row in table.rows],
            header_rows=table.header_rows,
            caption=table.caption,
        )


# --------------------------------------------------------------------------
# Elements and the document
# --------------------------------------------------------------------------


class SourceUnitContract(_Contract):
    """A citation-sized region: one page, slide or section."""

    unit_id: str
    locator: SourceLocatorContract
    label: str = ""
    text: str = ""
    order: int = 0

    @field_validator("unit_id")
    @classmethod
    def _must_be_present(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("unit_id must not be blank; it is what evidence points at")
        return value

    @field_validator("order")
    @classmethod
    def _must_not_be_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError(f"order must be >= 0, got {value}")
        return value

    def to_dataclass(self) -> SourceUnit:
        return SourceUnit(
            unit_id=self.unit_id,
            locator=self.locator.to_dataclass(),
            label=self.label,
            text=self.text,
            order=self.order,
        )

    @classmethod
    def from_dataclass(cls, unit: SourceUnit) -> SourceUnitContract:
        return cls(
            unit_id=unit.unit_id,
            locator=SourceLocatorContract.from_dataclass(unit.locator),
            label=unit.label,
            text=unit.text,
            order=unit.order,
        )


class ElementContract(_Contract):
    """One addressable element, with its role and locator pinned down.

    Two rules go beyond "is this shaped right". A ``formula`` role must carry
    LaTeX, because the eval runner counts role-``formula`` elements as formulas
    regardless and a role with nothing behind it inflates the denominator of the
    metric that decides whether the parser is fit. A ``table`` role must carry a
    grid, for the same reason from the other direction: a table that arrived as
    tab-separated prose is exactly what ``table_dense`` exists to catch, and it
    can only be caught if the flattening is visible as a missing grid rather
    than as an empty one.
    """

    role: str
    text: str = ""
    locator: SourceLocatorContract
    unit_id: str = ""
    element_id: str
    order: int = 0
    heading_path: tuple[str, ...] = ()
    level: int | None = None
    table: TableContract | None = None
    latex: str = ""
    figure: FigureContract | None = None
    formula: FormulaContract | None = None
    page_number_label: str = ""

    @field_validator("role")
    @classmethod
    def _must_be_a_known_role(cls, value: str) -> str:
        if value not in ELEMENT_ROLES:
            raise ValueError(
                f"unknown element role {value!r}; expected one of {', '.join(ELEMENT_ROLES)}"
            )
        return value

    @field_validator("element_id")
    @classmethod
    def _must_be_present(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("element_id must not be blank; it is what a chunk points back at")
        return value

    @field_validator("order")
    @classmethod
    def _must_not_be_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError(f"order must be >= 0, got {value}")
        return value

    @field_validator("level")
    @classmethod
    def _level_must_be_positive(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError(f"heading level must be >= 1, got {value}")
        return value

    @field_validator("heading_path", mode="before")
    @classmethod
    def _accepts_a_json_list(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _role_must_match_its_payload(self) -> ElementContract:
        if self.role == "formula" and not self.latex:
            raise ValueError("a 'formula' element must carry LaTeX")
        if self.role == "table" and self.table is None:
            raise ValueError("a 'table' element must carry a grid")
        if self.latex and self.role != "formula":
            raise ValueError(
                f"element role {self.role!r} does not use LaTeX; "
                "a formula that arrives under another role is invisible to "
                "the formula-counting path"
            )
        if self.formula is not None and self.latex and self.formula.latex != self.latex:
            raise ValueError(
                f"latex ({self.latex!r}) and formula.latex "
                f"({self.formula.latex!r}) disagree"
            )
        if self.formula is not None and self.role != "formula":
            raise ValueError(f"element role {self.role!r} does not carry a formula")
        if self.figure is not None and self.role != "figure":
            raise ValueError(f"element role {self.role!r} does not carry a figure")
        return self

    def to_dataclass(self) -> Element:
        return Element(
            role=self.role,
            text=self.text,
            locator=self.locator.to_dataclass(),
            unit_id=self.unit_id,
            element_id=self.element_id,
            order=self.order,
            heading_path=self.heading_path,
            level=self.level,
            table=self.table.to_dataclass() if self.table else None,
            latex=self.latex,
            figure=self.figure.to_dataclass() if self.figure else None,
            formula=self.formula.to_dataclass() if self.formula else None,
            page_number_label=self.page_number_label,
        )

    @classmethod
    def from_dataclass(cls, element: Element) -> ElementContract:
        return cls(
            role=element.role,
            text=element.text,
            locator=SourceLocatorContract.from_dataclass(element.locator),
            unit_id=element.unit_id,
            element_id=element.element_id,
            order=element.order,
            heading_path=tuple(element.heading_path),
            level=element.level,
            table=TableContract.from_dataclass(element.table) if element.table else None,
            latex=element.latex,
            figure=FigureContract.from_dataclass(element.figure) if element.figure else None,
            formula=FormulaContract.from_dataclass(element.formula) if element.formula else None,
            page_number_label=element.page_number_label,
        )


class ParsedDocumentContract(_Contract):
    """A whole parsed document, validated before it is trusted.

    The major-version check is not duplicated from ``ParsedDocument.from_dict``
    so much as *agreed with* it. It used to be weaker there: ``from_dict``
    compared ``"1.7"`` against ``"1.0"`` after a ``split(".")``, so a version
    string like ``"1"`` — or one that is not ``MAJOR.MINOR`` at all — fell
    through to a comparison against ``["1"]`` and was accepted by a gate whose
    whole job is to establish that the writer and the reader agree on the
    document's shape. The dataclass now requires the same ``MAJOR.MINOR`` form
    these contracts always required, and both sides take the pattern from
    ``SCHEMA_VERSION_PATTERN`` in :mod:`schema`, because a second copy of the
    rule is how the two drifted apart in the first place.
    """

    schema_version: str
    document_id: str
    source_filename: str = ""
    media_type: str = ""
    sha256: str = ""
    page_count: int = 0
    elements: list[ElementContract] = Field(default_factory=list)
    units: list[SourceUnitContract] = Field(default_factory=list)
    parser_name: str = ""
    parser_version: str = ""
    options_hash: str = ""
    created_at: str = ""
    warnings: list[str] = Field(default_factory=list)

    @field_validator("schema_version")
    @classmethod
    def _must_be_a_known_version(cls, value: str) -> str:
        if not _SCHEMA_VERSION_PATTERN.match(value):
            raise ValueError(
                f"malformed schema_version {value!r}; expected MAJOR.MINOR"
            )
        if not is_supported_schema_version(value):
            raise ValueError(
                f"unsupported schema version {value!r} "
                f"(this build reads {SCHEMA_VERSION})"
            )
        return value

    @field_validator("document_id")
    @classmethod
    def _must_be_present(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("document_id must not be blank; it is the storage key")
        return value

    @field_validator("page_count")
    @classmethod
    def _must_not_be_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError(f"page_count must be >= 0, got {value}")
        return value

    def to_dataclass(self) -> ParsedDocument:
        return ParsedDocument(
            schema_version=self.schema_version,
            document_id=self.document_id,
            source_filename=self.source_filename,
            media_type=self.media_type,
            sha256=self.sha256,
            page_count=self.page_count,
            elements=[element.to_dataclass() for element in self.elements],
            units=[unit.to_dataclass() for unit in self.units],
            parser_name=self.parser_name,
            parser_version=self.parser_version,
            options_hash=self.options_hash,
            created_at=self.created_at,
            warnings=list(self.warnings),
        )

    @classmethod
    def from_dataclass(cls, document: ParsedDocument) -> ParsedDocumentContract:
        return cls(
            schema_version=document.schema_version,
            document_id=document.document_id,
            source_filename=document.source_filename,
            media_type=document.media_type,
            sha256=document.sha256,
            page_count=document.page_count,
            elements=[ElementContract.from_dataclass(item) for item in document.elements],
            units=[SourceUnitContract.from_dataclass(item) for item in document.units],
            parser_name=document.parser_name,
            parser_version=document.parser_version,
            options_hash=document.options_hash,
            created_at=document.created_at,
            warnings=list(document.warnings),
        )


class ChunkContract(_Contract):
    """A persisted retrieval unit.

    Validated for the same reason as the document: a chunk is read back from
    the index days after it was written, and its locator is what turns a
    retrieved passage into a clickable citation.
    """

    chunk_id: str
    text: str = ""
    element_ids: tuple[str, ...] = ()
    locator: SourceLocatorContract
    heading_path: tuple[str, ...] = ()
    order: int = 0
    token_estimate: int = 0
    index_version: str = ""
    # Generated context, split off from the verbatim body on purpose: a
    # citation check must run against `verbatim_text`, not `text`.
    context_prefix: str = ""
    part_index: int = 0
    part_count: int = 0

    @field_validator("chunk_id")
    @classmethod
    def _must_be_present(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("chunk_id must not be blank; it is what evidence points at")
        return value

    @field_validator("order", "token_estimate", "part_index", "part_count")
    @classmethod
    def _must_not_be_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError(f"must be >= 0, got {value}")
        return value

    @field_validator("element_ids", "heading_path", mode="before")
    @classmethod
    def _accepts_a_json_list(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _part_numbering_is_coherent(self) -> ChunkContract:
        # A half-written part index is worse than none: it reads as
        # "chunk 2 of 2" while addressing the wrong piece, and every consumer
        # would carry the error into a citation. 0/0 means unsplit and is the
        # only inconsistent-but-valid form.
        if (self.part_index == 0) != (self.part_count == 0):
            raise ValueError(
                "part_index and part_count must both be set (a 1-based "
                f"part_index with part_count) or both left at 0; got "
                f"part_index={self.part_index}, part_count={self.part_count}"
            )
        if self.part_count and self.part_index >= self.part_count:
            raise ValueError(
                f"part_index {self.part_index} is out of range for part_count "
                f"{self.part_count}"
            )
        return self

    @model_validator(mode="after")
    def _element_ids_must_be_named(self) -> ChunkContract:
        # An empty element_ids is not a smaller chunk, it is a passage that
        # cannot be traced back to the document it claims to quote.
        if not self.element_ids:
            raise ValueError("a chunk must name at least one element it came from")
        if any(not str(item).strip() for item in self.element_ids):
            raise ValueError("element_ids must not contain a blank id")
        return self

    def to_dataclass(self) -> Chunk:
        return Chunk(
            chunk_id=self.chunk_id,
            text=self.text,
            element_ids=self.element_ids,
            locator=self.locator.to_dataclass(),
            heading_path=self.heading_path,
            order=self.order,
            token_estimate=self.token_estimate,
            index_version=self.index_version,
            context_prefix=self.context_prefix,
            part_index=self.part_index,
            part_count=self.part_count,
        )

    @classmethod
    def from_dataclass(cls, chunk: Chunk) -> ChunkContract:
        return cls(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            element_ids=tuple(chunk.element_ids),
            locator=SourceLocatorContract.from_dataclass(chunk.locator),
            heading_path=tuple(chunk.heading_path),
            order=chunk.order,
            token_estimate=chunk.token_estimate,
            index_version=chunk.index_version,
            context_prefix=chunk.context_prefix,
            part_index=chunk.part_index,
            part_count=chunk.part_count,
        )


# --------------------------------------------------------------------------
# Evidence and the fact layer
# --------------------------------------------------------------------------


class EvidenceContract(_Contract):
    """A passage already tied to where it came from.

    ``score`` is bounded to ``[0, 1]`` because it is a normalized retrieval
    score that the UI renders as a confidence bar. An out-of-range score
    arrives from a reranker or a fusion step that changed its scale, and a
    progress bar rendered from it is a bar wider than its own track.

    ``is_direct_quote`` is *not* validated against the text. Whether a quote
    aligns with the source is a job for the citation verifier (T1.6), which
    compares against the document; the only thing a schema can check is that
    the flag is a boolean at all, and pretending otherwise here would duplicate
    a check that cannot be done without the source.
    """

    evidence_id: str
    chunk_id: str = ""
    text: str = ""
    locator: SourceLocatorContract
    unit_id: str = ""
    heading_path: tuple[str, ...] = ()
    score: float = 0.0
    is_direct_quote: bool = False
    source_parser: str = ""

    @field_validator("evidence_id")
    @classmethod
    def _must_be_present(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("evidence_id must not be blank; the API addresses evidence by it")
        return value

    @field_validator("score")
    @classmethod
    def _must_be_normalized(cls, value: float) -> float:
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"score must be a finite value in [0, 1], got {value!r}")
        return value

    @field_validator("heading_path", mode="before")
    @classmethod
    def _accepts_a_json_list(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    def to_dataclass(self) -> Evidence:
        return Evidence(
            evidence_id=self.evidence_id,
            chunk_id=self.chunk_id,
            text=self.text,
            locator=self.locator.to_dataclass(),
            unit_id=self.unit_id,
            heading_path=self.heading_path,
            score=self.score,
            is_direct_quote=self.is_direct_quote,
            source_parser=self.source_parser,
        )

    @classmethod
    def from_dataclass(cls, evidence: Evidence) -> EvidenceContract:
        return cls(
            evidence_id=evidence.evidence_id,
            chunk_id=evidence.chunk_id,
            text=evidence.text,
            locator=SourceLocatorContract.from_dataclass(evidence.locator),
            unit_id=evidence.unit_id,
            heading_path=tuple(evidence.heading_path),
            score=evidence.score,
            is_direct_quote=evidence.is_direct_quote,
            source_parser=evidence.source_parser,
        )


class FactContract(_Contract):
    """One extracted claim, with where it came from and how sure we are.

    The invariant that earns this contract its place: a fact carrying a value
    must be marked ``extracted`` and must cite at least one evidence. The UI
    distinguishes "the paper says X" from "not stated in this paper" by exactly
    this pair, and an LLM that returns a value with ``extraction_status`` left
    at the ``not_found`` default — or with no citation — produces a card that
    reads like a finding and is not traceable to anything. Refusing it here
    means the extraction has to either ground the claim or drop it, which is
    the behaviour the plan's "insufficient evidence" exit requires.
    """

    field_name: str
    value: str = ""
    evidence_ids: tuple[str, ...] = ()
    extraction_status: str = "not_found"
    conflicting_values: tuple[str, ...] = ()

    @field_validator("field_name")
    @classmethod
    def _must_be_named(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("field_name must not be blank; it is the card's label")
        return value

    @field_validator("extraction_status")
    @classmethod
    def _must_be_a_known_status(cls, value: str) -> str:
        # The dataclass downgrades an unknown status to ``not_found``, which is
        # the safe direction for a status that governs a UI claim: the field
        # says "not stated" rather than inventing a result. But it is the wrong
        # place to absorb it, because the downgrade is invisible to whoever fed
        # the status in, and an extraction pipeline that is silently mislabelling
        # every status it does not recognize is a bug nobody sees.
        if value not in EXTRACTION_STATUSES:
            raise ValueError(
                f"unknown extraction_status {value!r}; expected one of "
                f"{', '.join(EXTRACTION_STATUSES)}"
            )
        return value

    @field_validator("evidence_ids", "conflicting_values", mode="before")
    @classmethod
    def _accepts_a_json_list(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _a_value_must_be_grounded(self) -> FactContract:
        if any(not str(item).strip() for item in self.evidence_ids):
            raise ValueError("evidence_ids must not contain a blank id")
        if self.value and not self.evidence_ids:
            raise ValueError(
                f"fact {self.field_name!r} has a value but no evidence_ids; "
                "an ungrounded claim must be reported with an explicit status "
                "instead"
            )
        if self.value and self.extraction_status == "not_found":
            raise ValueError(
                f"fact {self.field_name!r} has a value but is marked not_found; "
                "the two disagree about whether the paper said anything"
            )
        return self

    def to_dataclass(self) -> Fact:
        return Fact(
            field_name=self.field_name,
            value=self.value,
            evidence_ids=self.evidence_ids,
            extraction_status=self.extraction_status,
            conflicting_values=self.conflicting_values,
        )

    @classmethod
    def from_dataclass(cls, fact: Fact) -> FactContract:
        return cls(
            field_name=fact.field_name,
            value=fact.value,
            evidence_ids=tuple(fact.evidence_ids),
            extraction_status=fact.extraction_status,
            conflicting_values=tuple(fact.conflicting_values),
        )


class PaperFactsContract(_Contract):
    """The single structured extraction every analysis card renders from.

    Not validating the facts individually, only the assembly: this is the
    boundary where a whole extraction arrives from a model as one JSON object,
    and the failure that matters is a *missing* block rather than a malformed
    one. ``Fact.from_dict`` already substitutes a ``not_found`` fact for an
    absent block, and that substitution is the right behaviour — "the paper does
    not say" is a real answer and must render as a card, not as a 500.
    """

    research_question: FactContract
    contributions: list[FactContract] = Field(default_factory=list)
    method: FactContract
    datasets: list[FactContract] = Field(default_factory=list)
    metrics: list[FactContract] = Field(default_factory=list)
    main_results: list[FactContract] = Field(default_factory=list)
    limitations: list[FactContract] = Field(default_factory=list)
    reproducibility: FactContract
    extraction_model: str = ""
    prompt_version: str = ""
    evidence_version: str = ""

    def to_dataclass(self) -> PaperFacts:
        return PaperFacts(
            research_question=self.research_question.to_dataclass(),
            contributions=[fact.to_dataclass() for fact in self.contributions],
            method=self.method.to_dataclass(),
            datasets=[fact.to_dataclass() for fact in self.datasets],
            metrics=[fact.to_dataclass() for fact in self.metrics],
            main_results=[fact.to_dataclass() for fact in self.main_results],
            limitations=[fact.to_dataclass() for fact in self.limitations],
            reproducibility=self.reproducibility.to_dataclass(),
            extraction_model=self.extraction_model,
            prompt_version=self.prompt_version,
            evidence_version=self.evidence_version,
        )

    @classmethod
    def from_dataclass(cls, facts: PaperFacts) -> PaperFactsContract:
        return cls(
            research_question=FactContract.from_dataclass(facts.research_question),
            contributions=[FactContract.from_dataclass(item) for item in facts.contributions],
            method=FactContract.from_dataclass(facts.method),
            datasets=[FactContract.from_dataclass(item) for item in facts.datasets],
            metrics=[FactContract.from_dataclass(item) for item in facts.metrics],
            main_results=[FactContract.from_dataclass(item) for item in facts.main_results],
            limitations=[FactContract.from_dataclass(item) for item in facts.limitations],
            reproducibility=FactContract.from_dataclass(facts.reproducibility),
            extraction_model=facts.extraction_model,
            prompt_version=facts.prompt_version,
            evidence_version=facts.evidence_version,
        )


# --------------------------------------------------------------------------
# The API edge
# --------------------------------------------------------------------------


class UploadRequestContract(_Contract):
    """What a user's upload claims about itself.

    The only two things a client controls before a byte is parsed are the
    filename and the declared content type, and the gateway routes on them. The
    extension is checked against the supported set and the media type against
    the IANA registration rather than the short browser alias, because
    ``application/x-pdf`` is a spelling a browser will send and an adapter
    comparing against it will not match — which is how a PDF silently takes the
    plain-text path. That is a boundary decision, so it is made once, here,
    rather than in each route.

    The empty filename is refused rather than defaulted. ``UploadFile`` with an
    empty filename is a file input that was submitted before anything was
    chosen, and the two produce a ``filename`` that is an empty string rather
    than ``None``; treating that as "no file" is the check the upload route
    already makes, and duplicating it here means a caller that validates
    through the contract cannot get it wrong by forgetting.
    """

    filename: str
    media_type: str = ""

    @field_validator("filename")
    @classmethod
    def _must_name_a_supported_file(cls, value: str) -> str:
        name = value.strip()
        if not name:
            raise ValueError("filename must not be empty; no file was selected")
        if "/" in name or "\\" in name:
            # A path in a filename is either a client sending a full path or a
            # traversal attempt. Either way the name is not a filename, and the
            # storage layer should never be handed one it did not construct.
            raise ValueError(f"filename must be a bare name, got {value!r}")
        _, separator, extension = name.rpartition(".")
        lowered = f".{extension.lower()}" if separator else ""
        if lowered not in _UPLOAD_MEDIA_TYPES:
            supported = ", ".join(sorted(_UPLOAD_MEDIA_TYPES))
            raise ValueError(
                f"unsupported file type {lowered or name!r}; expected one of {supported}"
            )
        return name

    @field_validator("media_type")
    @classmethod
    def _must_be_a_known_type(cls, value: str) -> str:
        declared = value.strip().lower()
        if not declared:
            # A client that declares nothing has not disagreed with the
            # extension, it has said nothing, and the extension is enough.
            return ""
        if declared not in _UPLOAD_MEDIA_TYPE_ALIASES:
            raise ValueError(f"unsupported media type {value!r}")
        return _UPLOAD_MEDIA_TYPE_ALIASES[declared]

    def media_type_for_extension(self) -> str:
        """The canonical media type implied by the filename.

        Separate from validation because the extension and the declared type
        are two independent claims and the caller has to decide what to do when
        they disagree; the contract normalizes both and leaves that policy
        outside, where the router can see the whole request.
        """
        _, separator, extension = self.filename.rpartition(".")
        if not separator:
            return ""
        return _UPLOAD_MEDIA_TYPES.get(f".{extension.lower()}", "")
