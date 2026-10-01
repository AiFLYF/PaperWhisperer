r"""Docling adapter: ``DoclingDocument`` dict to canonical :class:`ParsedDocument`.

Three decisions shape this module, and all three come from the same place —
docling is a heavy optional dependency that this repository does not install.

1. **docling is imported lazily, inside :meth:`DoclingParser.parse`.** Docling
   drags in the whole torch chain (several GB) plus model weights. Importing it
   at module scope would make ``import paperwhisperer.ingestion.adapters`` cost
   gigabytes and would break every deployment that only uses the legacy
   parser. The adapter therefore treats "docling missing" as an ordinary parse
   *failure* with a Chinese, actionable message rather than an ``ImportError``
   traceback: the gateway reports failures to users, and a stack trace is not a
   user-facing message.
2. **Normalization is a pure function over a plain dict.** The input is what
   ``DoclingDocument.export_to_dict()`` returns, never a live docling object.
   That keeps the interesting half of this adapter — coordinate conversion,
   table reconstruction, reading order — testable on machines where docling
   cannot be installed, which is every machine in CI today. "Pure" is taken
   literally: the payload is never written back to, because the *same* object
   is what :func:`write_raw_artifact` persists, and one adapter-private key
   leaking into that file would quietly turn "the exact bytes the backend
   produced" into "the exact bytes the backend produced, plus our bookkeeping".
3. **The gateway protocol is duck-typed, not imported.** ``gateway.py`` defines
   ``ParseResult``/``ParserError`` and is being built alongside this file. The
   adapter defines a local :class:`DoclingParseResult` carrying the same field
   names, and :func:`_build_result` upgrades it to the gateway's real class the
   moment that class exists. A hard import here would be a circular import
   (gateway dispatches to adapters) and would also make this file unimportable
   while the two are being written in parallel.

What docling 2.131.0 actually emits
-----------------------------------
The second decision above — normalization is a pure function over a plain dict
— has a cost that is only payable if the dict is the *real* one. This adapter
was written against hand-built fixtures, and the fixtures disagreed with
docling on five points. Every one of them degraded silently, because each
produced a well-formed element with less information attached rather than an
error. All five are now checked against dumps from ``eval/corpus``
(``schema_name: DoclingDocument``, ``version: 1.10.0``):

``prov`` carries ``page_no``, not ``page``
    ``ProvenanceItem`` spells the field ``page_no``. Reading ``page`` found
    nothing on every item of every document, so all 96 elements of a 9-page
    paper came back with ``SourceLocator(kind="unknown")`` — page coverage 0.0,
    bbox 0/96, no highlight anywhere. :func:`_locator_for` now reads ``page_no``
    and records *why* it failed when it does.

``texts[i]`` carries ``label``, not ``text_label``
    ``TextItem`` has no ``text_label`` at all. Reading it made every label
    resolve to the ``"text"`` default, so 20 ``section_header``\\ s, 6
    ``formula``\\ s, 5 ``caption``\\ s and 10 furniture items all became plain
    text: no headings, no LaTeX, no page headers, and nothing for the
    ``repeated_furniture`` quality rule to find.

Table cell offsets are **exclusive**
    ``end_row_offset_idx = start_row_offset_idx + row_span`` — docling's own
    ``TableData.grid`` property iterates ``range(start, end)``. Treating the
    end as inclusive grew every table by one row and one column: a 4x3 table
    was rebuilt as 5x4.

Formulas live in ``orig``, not ``text``
    Docling labels a formula and leaves ``text`` empty; the content is in the
    untreated ``orig``. Reading ``text`` alone found no formula content at all.
    What arrives there is Unicode maths (``∑``, ``∂``, ``λ``), not LaTeX, so
    :func:`_to_latex` transliterates it; see "Formulas" below.

``pages`` is a dict keyed by the *string* page number
    ``{"1": {"size": {...}, "page_no": 1}, ...}``. Handled: a payload whose
    ``pages`` is absent or a list degrades to positional sorting and says so.

The lesson generalises beyond the five: **a normalization path that drops
information cannot report that it dropped it**, so every stage that can lose
data records a named reason on ``document.warnings``. :data:`LocatorIssue`
carries those reasons, and the aggregate counts are appended to the document.

Coordinate systems
------------------
Our :class:`BoundingBox` is PDF user-space points with the origin at the
bottom-left, because that is what ``pdf.js`` expects for a highlight fragment.
Docling reports ``l``/``t``/``r``/``b`` plus an explicit ``coord_origin`` which
may be ``BOTTOMLEFT`` (PDF convention, y grows upwards) or ``TOPLEFT`` (image
convention, y grows downwards). A ``TOPLEFT`` box cannot simply have ``t`` and
``b`` swapped: the flip has to be mirrored against the page height, otherwise
the highlight lands mirrored around y=0. When the page height is unknown the
best available answer is a *relative* flip (negate y), which preserves
top-to-bottom ordering within a page — which is what reading order needs — at
the cost of absolute coordinates. That case is reported in
``document.warnings`` instead of being silently absorbed.

Locator kinds across input formats
----------------------------------
The canonical schema has a locator kind per format (``slide`` for a deck,
``sheet_range`` for a workbook, ``docx_element`` for a Word file), and this
adapter emits ``pdf_page`` for all of them. That is deliberate for now and
worth stating plainly: docling normalizes *every* input into one page-oriented
document model before the adapter sees it, so a page and a rectangle is
genuinely all the provenance that exists for a converted DOCX. Minting a
``paragraph`` index from a page number, or a sheet name docling never reported,
would produce a locator that looks precise and resolves to the wrong place —
worse than an honest page. Turning docling's page into a slide is a sound
mapping and is left to the router (T0.7) / evidence viewer (T3.3), which
together know the original file type; the adapter only sees a media type.

Furniture: in, and marked
------------------------
Docling tags running headers and footers twice over, and both signals are
used. ``label`` says *what the region is* (``page_header`` / ``page_footer``)
and ``content_layer`` says *which plane it belongs to* (``furniture``, plus
``background`` and ``invisible``). Either alone is insufficient: a running
header docling did not classify is still furniture, and an item labelled
``page_header`` in the body layer is still not prose.

**Furniture enters the document as ``header``/``footer`` elements and is not
excluded.** The reason is that exclusion is unrecoverable in both directions.
``ParsedDocument.body_elements`` already filters furniture out of every prose
projection, chunk and embedding, so admitting it costs nothing downstream —
while a quality gate cannot act on what it never receives. ``RepeatedFurnitureRule``
exists precisely to notice a journal name repeated on every page, and on the
9-page distillation paper docling reports 9 ``page_footer`` items all reading
``"N"``: the rule is written to mask digits and count them, and it can only do
that if the elements are there. Dropping them at the adapter would make the
rule permanently vacuous while still reporting "0 furniture elements found" —
the same silent-degradation failure this module exists to prevent. So they are
kept, tagged, and available to be counted; the *reason* for the tag is recorded
so a reader can tell an unclassified body-layer header from a classified one.

Formulas
--------
Docling's ``formula`` label arrives with ``text`` empty and the content in
``orig``, and what is in ``orig`` is Unicode maths, not LaTeX — a real PDF text
layer has no notion of a command. :func:`_to_latex` transliterates the Unicode
operators and Greek letters to their LaTeX commands using the same table
``ingestion/latex.py`` already maintains for exactly this corpus (its
``unicode_maths`` tolerance documents that a real text layer emits ``α``
rather than ``\\alpha``), and leaves a string that is already LaTeX alone.

What this does *not* do is turn the formula into a typeset one: there are no
fractions, no ``\left``/``\right`` sizing, and no distinction between display
and inline beyond the trailing equation number, which is kept. Real LaTeX
requires docling's own ``do_formula_enrichment`` pipeline stage, which is off
by default, downloads a separate model, and roughly doubles parse time; that
is an operator's cost decision and not a normalization bug, so this adapter
does not turn it on. A formula is never dropped for having empty ``text``.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from paperwhisperer.core import config
from paperwhisperer.core.text import (
    atomic_write_json,
    compute_retry_after_delay,
    get_retry_delay_seconds,
    now_iso,
)
from paperwhisperer.ingestion.schema import (
    BoundingBox,
    Element,
    ParsedDocument,
    SourceLocator,
    Table,
)

logger = logging.getLogger(__name__)

#: Parser name recorded on every document this adapter produces. The gateway
#: routes and the eval runner key on this string, so it is part of the contract.
PARSER_NAME = "docling"

#: Formats docling can convert. Kept broad on purpose: the gateway decides
#: which parser to try, the adapter only reports what it could handle.
SUPPORTED_MEDIA_TYPES = frozenset(
    {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
        "application/vnd.ms-powerpoint",
        "application/msword",
        "text/html",
        "text/markdown",
        "text/csv",
        "text/plain",
        "image/png",
        "image/jpeg",
        "image/tiff",
        "image/bmp",
        "image/webp",
    }
)

#: Shown when docling is not installed. Written for a user staring at an error
#: toast, not for a log file, so it names the exact extra to install.
MISSING_DOCLING_MESSAGE = (
    "未安装 docling 解析后端，无法解析该文档。"
    "请执行 pip install 'paperwhisperer[parse]' 后重试，"
    "或改用 docling-serve 模式（PARSER_DOCLING_MODE=serve）。"
)

#: docling ``label`` (the field is spelled ``label`` on every ``DocItem``;
#: there is no ``text_label`` in any docling release this adapter has met) to
#: our ``Element.role``. Labels outside this table are downgraded to ``text``
#: rather than rejected: a newer docling adding a label must not turn a whole
#: page into a parse failure, and ``Element`` would downgrade it anyway — doing
#: it here keeps the intent explicit.
TEXT_LABEL_ROLES: dict[str, str] = {
    "title": "title",
    "section_header": "heading",
    "formula": "formula",
    "caption": "caption",
    "footnote": "footnote",
    "page_header": "header",
    "page_footer": "footer",
    "list_item": "list_item",
    "code": "code",
    "checkbox_selected": "list_item",
    "checkbox_unselected": "list_item",
    "form": "text",
    "unknown": "text",
    # Labels present in ``DocItemLabel`` but rare or absent in the eval
    # corpus. Named explicitly so that adding one is a deliberate line rather
    # than an accident of the default, and so a future docling release that
    # starts emitting them has already been thought about.
    "chart": "figure",
    "document_index": "text",
    "key_value_region": "text",
    "grading_scale": "text",
    "handwritten_text": "text",
    "field_region": "text",
    "field_heading": "heading",
    "field_item": "list_item",
    "field_key": "text",
    "field_value": "text",
    "field_hint": "text",
    "reference": "footnote",
    "empty_value": "text",
    "marker": "text",
    "paragraph": "text",
}

#: ``content_layer`` values that are *not* body prose. ``furniture`` is the
#: one docling actually emits for PDFs; ``background`` and ``invisible`` are
#: declared by ``ContentLayer`` and mean the same thing. ``notes`` is a
#: separate plane of marginal content (callout boxes in the synthetic
#: corpus) that is *not* furniture, so it stays in the body.
FURNITURE_LAYERS = frozenset({"furniture", "background", "invisible"})

#: Provenance page-number field names, in priority order. ``page_no`` is what
#: ``ProvenanceItem`` calls it and what every dump in ``eval/corpus`` carries;
#: ``page`` is read second so a hand-built or older payload still resolves
#: rather than silently degrading to ``kind="unknown"``.
PROV_PAGE_KEYS = ("page_no", "page")

#: Why an element ended up without a usable locator, as a named reason rather
#: than a dropped value. The whole point is that a drop is visible: these are
#: counted and reported on ``document.warnings``. ``""`` means no issue.
LocatorIssue = str

#: The item carried no ``prov`` at all. Observed on 46 real items, all in
#: DOCX payloads — a format docling paginates inconsistently — so the
#: elements are kept and simply cannot be pointed at.
ISSUE_NO_PROV: LocatorIssue = "no_prov"

#: ``prov`` was present but carried no usable page number, or a number below
#: 1. A different failure from a missing ``prov`` and worth counting
#: separately: it usually means a docling version renamed the field again.
ISSUE_NO_PAGE: LocatorIssue = "no_page"

#: The page was found but its geometry could not be read, so the locator
#: resolves to a page with no rectangle. A highlight with no box still
#: resolves to a page in the viewer, which is why this is not fatal.
ISSUE_NO_BBOX: LocatorIssue = "no_bbox"

#: A ``TOPLEFT`` box with no page height to mirror against, so the box is
#: relative: right within the page, but not in absolute PDF user space.
ISSUE_RELATIVE_BBOX: LocatorIssue = "relative_bbox"

#: An item docling labelled but gave no readable content to — in practice a
#: ``formula`` whose ``text`` and ``orig`` are both empty. It is dropped rather
#: than emitted as a blank ``$$``, and counted here so that "the parser found
#: no formulas" is never what a quality report actually means.
ISSUE_NO_CONTENT: LocatorIssue = "no_content"

#: ``coord_origin`` values that mean "y grows downwards from the page top".
_TOPLEFT_ORIGINS = frozenset({"TOPLEFT", "TOP-LEFT", "TOP_LEFT"})

#: Retries for a docling-serve round trip. Parsing is a pure function of the
#: uploaded bytes, so replaying a POST is safe here in a way it would not be for
#: a mutating API call.
SERVE_MAX_ATTEMPTS = 3

#: Backoff ceiling for docling-serve retries, in seconds.
SERVE_RETRY_MAX_DELAY = 6


@dataclass
class DoclingParseResult:
    """Local mirror of ``gateway.ParseResult``.

    Field names and meanings match the gateway contract exactly so the gateway
    can accept this object by duck typing while both files are developed in
    parallel. :func:`_build_result` swaps in the real class when it exists; see
    the module docstring for why this indirection is necessary.
    """

    status: str = "ok"
    document: ParsedDocument | None = None
    quality: Any = None
    raw_artifact_uri: str = ""
    warnings: list[str] = field(default_factory=list)
    duration_ms: int = 0
    parser_name: str = PARSER_NAME
    parser_version: str = ""
    options_hash: str = ""
    error_message: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok" and self.document is not None

    @property
    def is_usable(self) -> bool:
        """Mirror of ``gateway.ParseResult.is_usable``, with the same name.

        Once the gateway is importable :func:`_build_result` returns *its* class
        instead, and that class has ``is_usable`` but not ``ok``. Exposing both
        here means a caller can write ``result.is_usable`` and be correct
        whichever class it ended up with — the failure mode this avoids is an
        ``AttributeError`` on ``.ok`` that only appears once the gateway lands.
        """
        return self.ok


# --------------------------------------------------------------------------
# Small defensive readers
# --------------------------------------------------------------------------


def _as_int(value: Any) -> int | None:
    """Best-effort int, or ``None``.

    Docling's JSON carries many optional numeric fields, and a version bump can
    turn any of them into a string or a null. A missing index must degrade one
    cell, never abort the document, so every read here is total.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ref_of(item: Any) -> str:
    """Extract a ``self_ref``-style pointer from a child entry."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return str(item.get("$ref") or item.get("self_ref") or "")
    return ""


def _first_prov(item: dict[str, Any]) -> dict[str, Any]:
    """The first provenance entry of a docling item, or an empty dict."""
    prov = item.get("prov")
    if isinstance(prov, list):
        for entry in prov:
            if isinstance(entry, dict):
                return entry
        return {}
    return prov if isinstance(prov, dict) else {}


def _prov_page(prov: dict[str, Any]) -> int | None:
    """The 1-based page number in a provenance entry, or ``None``.

    ``ProvenanceItem`` spells this ``page_no``. Reading ``page`` — which is what
    this used to do, against a fixture rather than against docling — returns
    ``None`` for every item of every real document, which is exactly how the
    adapter managed to lose 100% of its locators without a single warning:
    an element with no page is still a perfectly well-formed element. Both
    spellings are read, ``page_no`` first, so that a payload written by a
    hand-built fixture or by an older release still resolves.
    """
    for key in PROV_PAGE_KEYS:
        page = _as_int(prov.get(key))
        if page is not None and page >= 1:
            return page
    return None


#: Unicode maths to LaTeX, borrowed from :mod:`paperwhisperer.ingestion.latex`
#: rather than restated here. That module's ``unicode_maths`` tolerance exists
#: because real text layers in ``eval/corpus`` emit ``α`` and ``≤`` instead of
#: ``\alpha`` and ``\le``; this is the same observation seen from the other
#: side, and a second copy of a 130-entry table would be free to drift from it.
#:
#: The import is defensive on purpose. The name is private, so a rename in
#: that module must degrade this adapter to "transliterate nothing" rather than
#: make every document unparseable; the consequence of that is a formula
#: element whose ``latex`` is still the raw Unicode maths, which is valid
#: LaTeX and simply renders as symbols.
try:  # pragma: no cover - exercised by the import itself
    from paperwhisperer.ingestion.latex import _GLYPH_NAMES as _GLYPH_TO_LATEX
    from paperwhisperer.ingestion.latex import _UNICODE_MATHS as _MATH_TO_LATEX
except Exception:  # noqa: BLE001 - a missing table is a degraded mode, not a failure
    _MATH_TO_LATEX: dict[str, str] = {}
    _GLYPH_TO_LATEX: dict[str, str] = {}

#: A glyph name as a PDF text layer emits it when the font has no ToUnicode
#: entry. Observed verbatim in ``eval/corpus/formula_dense/green_tao_primes``.
_GLYPH_NAME_PATTERN = re.compile(r"/[A-Za-z]{3,}")


#: A LaTeX control word. In TeX a backslash binds every following letter, so
#: ``\sumj`` is the single undefined control sequence ``\sumj`` — not ``\sum``
#: applied to ``j``, and not a formula that fails to compile. A transliteration
#: that emits it has moved the defect from "the symbol is Unicode" to "the
#: string is not LaTeX at all", which is worse: the eval counts a non-empty
#: ``latex`` as preserved.
#:
#: Deliberately matched *without* a trailing lookahead. An earlier version used
#: ``(\\[A-Za-z]+)(?=[A-Za-z])``, which is a trap: the lookahead forces the
#: greedy ``+`` to give back letters until one is followed by a letter, so it
#: matched the **shortest** viable prefix and turned ``\frac`` into ``\fra c``,
#: ``\lambda`` into ``\lambd a`` and ``\sum`` into ``\su m`` — corrupting every
#: command in the formula while still looking like a fix. The run is taken in
#: full here and the decision to break it is made afterwards, in
#: :func:`_break_run_on_control_words`.
_CONTROL_WORD = re.compile(r"\\[A-Za-z]+")


def _substitute_unicode_maths(source: str) -> str:
    """Map each Unicode maths character to its LaTeX command, safely.

    The lookahead is the whole point. ``∑`` becomes ``\\sum`` and the operand
    that follows it is a plain letter, so a naive per-character join emits
    ``\\sumj`` — and by the time any post-pass could look at the string, that
    is one control word ``\\sumj`` with nothing after it, indistinguishable
    from a legitimate (if unknown) command. TeX, meanwhile, reads ``\\sumj`` as
    the single undefined control sequence ``\\sumj``, so the result is a string
    that looks like a formula and is not one. Deciding at the moment of
    substitution is the only point where the boundary is still knowable.

    A space is the separator because TeX discards spaces in math mode, making
    ``\\sum j`` the same expression to a typesetter while being a legal program.
    """
    pieces: list[str] = []
    for position, char in enumerate(source):
        replacement = _MATH_TO_LATEX.get(char)
        if replacement is None:
            pieces.append(char)
            continue
        pieces.append(replacement)
        if source[position + 1 : position + 2].isalpha():
            pieces.append(" ")
    return "".join(pieces)


def _break_run_on_control_words(text: str) -> str:
    """Insert a space after any control word that runs into a following letter.

    A second pass over the *result* of substitution, for the cases the
    character loop cannot see: a glyph name resolved from a font that had no
    ToUnicode entry, and LaTeX that arrived already run together. It repairs
    ``\\frac ab``'s cousin ``\\fracab`` and leaves the legitimate
    ``\\frac ab`` alone, and it cannot repair a run-on that
    :func:`_substitute_unicode_maths` already merged — which is why that
    function does its own lookahead rather than relying on this one.
    """
    if "\\" not in text:
        return text

    def _separate(match: re.Match[str]) -> str:
        following = text[match.end() : match.end() + 1]
        return f"{match.group(0)} " if following.isalpha() else match.group(0)

    return _CONTROL_WORD.sub(_separate, text)


def _to_latex(text: str) -> str:
    r"""Transliterate one formula's text into LaTeX.

    Docling gives a formula's content as Unicode maths, so ``∑`` becomes
    ``\sum``, ``∂`` becomes ``\partial`` and ``λ`` becomes ``\lambda``. String
    that is already LaTeX is left alone in the sense that it is not *harmed*:
    the substitutions are per-character lookups, so a formula containing no
    Unicode maths comes back identical, and one containing some comes back with
    only those characters changed.

    The non-obvious half is :func:`_substitute_unicode_maths`, which decides
    the control-word boundary as it substitutes rather than afterwards. See its
    docstring for why a post-pass cannot do the job.

    This is a transliteration, not a typesetting pass: the result carries no
    ``\limits``, no sizing and no grouping, so ``∑`` over a subscript renders
    beside its operand rather than above it. It is enough for what this field
    is for — a formula that is retrievable, that the chunker can embed, and
    that a comparison downstream can canonicalise back — and the module
    docstring says plainly that it is not the whole story. Real LaTeX needs
    docling's ``do_formula_enrichment`` stage, which is an operator's cost
    decision rather than a normalization fix.
    """
    source = str(text or "").strip()
    if not source:
        return ""
    translated = _substitute_unicode_maths(source)
    translated = _GLYPH_NAME_PATTERN.sub(
        lambda match: _GLYPH_TO_LATEX.get(match.group(0), match.group(0)),
        translated,
    )
    return _break_run_on_control_words(translated).strip()


def _item_text(item: dict[str, Any]) -> str:
    """The best available text for a docling item.

    ``TextItem.text`` is the sanitized rendering and ``orig`` the untreated
    one. Docling leaves ``text`` **empty on a formula** and puts the content in
    ``orig``, so reading ``text`` alone is how "6 formulas, 0 with LaTeX"
    happened: there was no formula *text*, and the element was dropped before
    the LaTeX question was ever asked. Falling back to ``orig`` is safe for
    every other label, where the two agree except for ligature and Unicode
    normalization.
    """
    text = str(item.get("text") or "").strip()
    if text:
        return text
    return str(item.get("orig") or "").strip()


def _item_label(item: dict[str, Any]) -> str:
    """The lowercased docling label for an item.

    ``label`` is the field. ``text_label`` is kept as a second spelling purely
    so a payload built to the adapter's old (incorrect) assumption still maps
    instead of collapsing every element to ``text``; it does not exist in any
    docling release and is read second for that reason.
    """
    for key in ("label", "text_label"):
        value = str(item.get(key) or "").strip().lower()
        if value:
            return value
    return ""


def _item_role(item: dict[str, Any]) -> str:
    """Map a docling item's label and layer onto an ``Element.role``.

    Furniture is decided by *either* signal, and the combination matters.
    ``page_header``/``page_footer`` say what the region is; ``content_layer``
    says which plane it is on. Docling classifies a running header with the
    former and tags an unclassified one with the latter, and a rule that
    honoured only one of them would misclassify exactly half the furniture.
    """
    label = _item_label(item)
    layer = str(item.get("content_layer") or "").strip().lower()
    if label == "page_footer":
        return "footer"
    if label == "page_header":
        return "header"
    if layer in FURNITURE_LAYERS:
        # Furniture plane wins over an unclassified label. A footer docling
        # failed to label is far more often a running page number than a
        # journal name, but that guess is only load-bearing because
        # ``is_furniture`` asks *whether* the role is furniture, not which
        # furniture it is — so the worst case is a mislabelled element that is
        # still correctly excluded from the body and still counted by the
        # ``repeated_furniture`` rule.
        return "header"
    return TEXT_LABEL_ROLES.get(label, "text")


def _list_marker_text(item: dict[str, Any], text: str) -> str:
    """Re-attach a list item's ``marker`` to its text.

    Docling lifts the bullet or number out of the text and into ``marker``:
    across the eval corpus every one of 2363 marked ``list_item`` entries has
    ``text`` *starting after* the marker while ``orig`` still contains it. So
    reading ``text`` alone silently deletes the numbering of every enumerated
    reference list — the references of the distillation paper arrive as nine
    unlabelled sentences. The marker is prefixed only when it is not already
    there, so a docling release that stops stripping it changes nothing.
    """
    marker = str(item.get("marker") or "").strip()
    if not marker or not text or text.startswith(marker):
        return text
    return f"{marker} {text}"



def to_bounding_box(
    docling_bbox: dict[str, Any],
    *,
    page_height: float | None = None,
) -> BoundingBox | None:
    """Convert one docling bbox into our PDF-space :class:`BoundingBox`.

    Docling boxes are ``{l, t, r, b, coord_origin}`` where the meaning of
    ``t``/``b`` depends on ``coord_origin``:

    * ``BOTTOMLEFT`` — the PDF convention, and the one our schema already uses.
      ``b`` is the lower edge and ``t`` the upper edge, so the mapping is a
      direct rename.
    * ``TOPLEFT`` — the image convention, where both ``t`` and ``b`` are
      distances *downwards* from the page top. Converting requires the page
      height: ``y0 = height - b`` and ``y1 = height - t``. Swapping ``t``/``b``
      without the mirror is the classic bug here — it yields a box of the right
      size in the wrong half of the page.

    When the page height is unavailable the mirror cannot be computed, so the
    box is flipped relatively (``y`` negated). That is wrong in absolute terms
    but preserves the within-page vertical ordering that reading order depends
    on; callers signal the degradation through ``document.warnings``.

    Returns ``None`` for anything that is not a usable rectangle — a missing
    key, a non-numeric coordinate, a non-dict input. A fabricated box would be
    worse than no box, because the evidence viewer would draw a highlight over
    unrelated text.
    """
    if not isinstance(docling_bbox, dict):
        return None
    values: dict[str, float] = {}
    for key in ("l", "t", "r", "b"):
        number = _as_float(docling_bbox.get(key))
        if number is None:
            return None
        values[key] = number

    origin = str(docling_bbox.get("coord_origin") or "BOTTOMLEFT").strip().upper()
    if origin in _TOPLEFT_ORIGINS:
        if page_height is not None and page_height > 0:
            y0 = page_height - values["b"]
            y1 = page_height - values["t"]
        else:
            y0 = -values["b"]
            y1 = -values["t"]
    else:
        y0 = values["b"]
        y1 = values["t"]

    # ``l``/``r`` are normalized rather than trusted: a mirrored box from a
    # buggy converter should still produce a positive-width rectangle, since
    # every consumer computes width with ``abs``.
    #
    # ``relative`` is set here at the point of the fork, not left to
    # ``BoundingBox.__post_init__``'s geometric derivation (which infers it
    # from a negative y0). The derivation is a safety net for boxes built
    # elsewhere; only this branch knows *why* the box cannot be absolute. A
    # TOPLEFT box with no page height is mirrored about y=0 and lands in a
    # coordinate space the page does not share, so any consumer that trusts it
    # as absolute — notably ``to_query()``, whose whole output is a highlight
    # fragment — will draw over the wrong lines while looking correct.
    return BoundingBox(
        x0=min(values["l"], values["r"]),
        y0=min(y0, y1),
        x1=max(values["l"], values["r"]),
        y1=max(y0, y1),
        relative=origin in _TOPLEFT_ORIGINS and not (page_height is not None and page_height > 0),
    )


# --------------------------------------------------------------------------
# Table reconstruction
# --------------------------------------------------------------------------


def _cell_text(cell: dict[str, Any]) -> str:
    """Cell text, preferring ``text`` and falling back to ``orig``.

    Numeric cells occasionally arrive as numbers rather than strings, which is
    why this coerces instead of assuming ``str``.
    """
    for key in ("text", "orig"):
        value = cell.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def build_table_from_docling(payload: dict[str, Any]) -> Table:
    """Rebuild a two-dimensional grid from a docling ``tables[i]`` entry.

    Docling describes a table as a flat list of cells, each carrying its
    position (``start_row_offset_idx`` … ``end_col_offset_idx``) and its span.
    Four details make this fiddly:

    * **The end offsets are exclusive.** ``end_row_offset_idx`` is
      ``start_row_offset_idx + row_span``, not the index of the last row the
      cell covers — every docling backend writes it that way and docling's own
      ``TableData.grid`` property iterates ``range(start, end)``. Reading them
      as inclusive is a one-row-and-one-column error on *every* table, which is
      precisely what a 4x3 table on the distillation paper was rebuilt as:
      5x4, with a trailing phantom row and column of blanks. The span is
      therefore the floor for the end and a reported offset can only push it
      further, never pull it back, so a payload that means something else by
      the end index degrades to "at least the span" rather than to a
      misaligned grid.
    * **Merged cells occupy several grid positions.** A cell with
      ``row_span=2`` is one logical value covering two rows, so its text is
      written into every position it covers. Writing it once would shift the
      whole grid; writing it into the span is what makes a merged header read
      as a merged header. Positions are filled first-write-wins so a merged
      cell is not overwritten by a later zero-width continuation entry.
    * **Offsets are authoritative but not always present.** Older payloads
      carry only ``row_span``/``col_span``, so a running cursor supplies the
      missing start. The span supplies the missing end in every case, which is
      why a payload with no offsets at all still lands on the right grid as
      long as it declares its width.
    * **Cells carry their own ``bbox``** in a per-cell coordinate frame that
      is *not* the page frame (``coord_origin`` is ``TOPLEFT`` for a PDF
      whose own ``prov`` is ``BOTTOMLEFT``). It is deliberately ignored: the
      table's ``prov`` bbox is the addressable region, and a per-cell box in a
      different frame would be a highlight in the wrong place.

    The cursor is the one place where a heuristic has to admit ignorance. It
    walks the cell list in row-major order — which is the order docling emits
    them in — and needs a table width to know when a row is full. Two cases:

    * ``num_cols`` is declared: the cursor wraps at it, and the result is exact.
    * ``num_cols`` is absent: the cursor keeps advancing *along* the row and
      never wraps, so the cells land in one row. That is deliberately the
      least committal answer — the payload really does not say how wide the
      table is, and guessing a width by counting cells would invent rows and
      columns that the converter never reported. The alternative failure mode
      (a diagonal staircase, one cell per row and column) looks like data.

    ``header_rows`` is inferred by docling's own ``column_header`` flag: the
    header block ends at the deepest row that any column-header cell reaches.
    That is more robust than assuming row 0, which breaks for tables with a
    title row above the header. Note that the count is unchanged by the
    exclusive convention: the inclusive last row plus one *is* the exclusive
    end.
    """
    data = payload.get("data")
    if not isinstance(data, dict):
        data = {}
    raw_cells = data.get("table_cells")
    cells = [cell for cell in raw_cells if isinstance(cell, dict)] if isinstance(
        raw_cells, list
    ) else []
    declared_columns = _as_int(data.get("num_cols")) or 0

    placed: list[tuple[int, int, int, int, str, bool]] = []
    cursor_row = 0
    cursor_col = 0
    header_rows = 0

    for cell in cells:
        span_rows = max(1, _as_int(cell.get("row_span")) or 1)
        span_cols = max(1, _as_int(cell.get("col_span")) or 1)

        start_row = _as_int(cell.get("start_row_offset_idx"))
        start_col = _as_int(cell.get("start_col_offset_idx"))
        if start_row is None:
            start_row = cursor_row
        if start_col is None:
            start_col = cursor_col

        # Exclusive end, from the span as a floor and the reported offset as a
        # ceiling-only override. ``end_row`` below is the *inclusive* last row,
        # kept as such because the grid-filling loop is naturally written
        # against an inclusive range.
        end_row_exclusive = start_row + span_rows
        reported_end_row = _as_int(cell.get("end_row_offset_idx"))
        if reported_end_row is not None and reported_end_row > end_row_exclusive:
            end_row_exclusive = reported_end_row
        end_col_exclusive = start_col + span_cols
        reported_end_col = _as_int(cell.get("end_col_offset_idx"))
        if reported_end_col is not None and reported_end_col > end_col_exclusive:
            end_col_exclusive = reported_end_col

        end_row = end_row_exclusive - 1
        end_col = end_col_exclusive - 1
        if end_row < start_row or end_col < start_col:
            continue
        # The cursor always re-syncs to the cell it just placed, so a payload
        # that mixes offset and offset-less cells stays in order either way.
        if declared_columns and end_col_exclusive >= declared_columns:
            cursor_row, cursor_col = end_row_exclusive, 0
        else:
            cursor_row, cursor_col = end_row, end_col + 1

        is_column_header = bool(cell.get("column_header"))
        if is_column_header:
            header_rows = max(header_rows, end_row + 1)
        placed.append((start_row, end_row, start_col, end_col, _cell_text(cell), is_column_header))

    row_count = max(
        _as_int(data.get("num_rows")) or 0,
        max((entry[1] + 1 for entry in placed), default=0),
    )
    column_count = max(
        declared_columns,
        max((entry[3] + 1 for entry in placed), default=0),
    )

    grid = [[""] * column_count for _ in range(row_count)]
    for start_row, end_row, start_col, end_col, text, _is_header in placed:
        for row_index in range(start_row, min(end_row, row_count - 1) + 1):
            row = grid[row_index]
            for col_index in range(start_col, min(end_col, column_count - 1) + 1):
                if not row[col_index]:
                    row[col_index] = text

    header_rows = min(header_rows, row_count)
    return Table(rows=grid, header_rows=header_rows)


# --------------------------------------------------------------------------
# Element construction
# --------------------------------------------------------------------------


def _page_height_for(pages: dict[str, Any], page: int | None) -> float | None:
    """The height of one page, from docling's ``pages`` map.

    ``pages`` is a **dict keyed by the string page number**, not the list a
    list-shaped helper would want::

        {"1": {"size": {"width": 612.0, "height": 792.0}, "page_no": 1}, ...}

    Both spellings of the key are tried so a payload keyed by ``int`` also
    resolves, and ``None`` is returned for anything else — which is what makes
    a ``TOPLEFT`` box relative, and is reported rather than hidden.
    """
    if page is None:
        return None
    entry = pages.get(str(page)) or pages.get(page)
    if not isinstance(entry, dict):
        return None
    size = entry.get("size")
    if not isinstance(size, dict):
        return None
    return _as_float(size.get("height"))


def _locator_for(item: dict[str, Any], pages: dict[str, Any]) -> tuple[SourceLocator, LocatorIssue]:
    """Build a ``pdf_page`` locator for a docling item.

    Returns the locator plus a :data:`LocatorIssue` naming why it is
    incomplete, or ``""`` when it is whole. The reason matters more than the
    locator: a dropped locator is indistinguishable from a parser that
    produced none, and that indistinguishability is how 96 of 96 elements once
    arrived with ``kind="unknown"`` while the parse still reported success.
    Naming the cause is what makes the failure visible to a quality gate and
    to whoever reads the warnings.
    """
    prov = _first_prov(item)
    if not prov:
        return SourceLocator(kind="unknown"), ISSUE_NO_PROV
    page = _prov_page(prov)
    if page is None:
        return SourceLocator(kind="unknown"), ISSUE_NO_PAGE

    page_height = _page_height_for(pages, page)
    raw_bbox = prov.get("bbox")
    bbox = to_bounding_box(raw_bbox, page_height=page_height)
    issue: LocatorIssue = ""
    if bbox is None:
        issue = ISSUE_NO_BBOX
    elif isinstance(raw_bbox, dict):
        origin = str(raw_bbox.get("coord_origin") or "").strip().upper()
        if origin in _TOPLEFT_ORIGINS and not (page_height and page_height > 0):
            issue = ISSUE_RELATIVE_BBOX
    return SourceLocator(kind="pdf_page", page=page, bbox=bbox), issue


def _text_element(
    item: dict[str, Any],
    pages: dict[str, Any],
) -> tuple[Element | None, LocatorIssue]:
    """Turn a docling ``texts[i]`` entry into an :class:`Element`.

    Three things are read that a naive reader misses, each of which silently
    removes content on a real payload:

    * ``label``, not ``text_label`` — see the module docstring. Getting this
      wrong cost every heading, every caption and every formula.
    * ``orig``, when ``text`` is empty — docling labels a formula and leaves
      ``text`` blank, so this is the only place a formula's content exists.
    * ``marker``, for ``list_item`` — docling lifts the enumeration out of the
      text, so a references list arrives as unlabelled sentences.

    ``level`` is carried through from docling's heading depth. It is set only
    when docling reported one; ``None`` otherwise, and consumers still must not
    depend on it, because a docling release that stops reporting depth would
    make it vanish without any other trace.
    """
    text = _item_text(item)
    role = _item_role(item)
    if not text:
        # A formula with no content at all is still not a formula. Anything
        # else empty is dropped, as before: it carries nothing to retrieve.
        return None, ""
    if role == "list_item":
        text = _list_marker_text(item, text)

    locator, issue = _locator_for(item, pages)
    # A formula's LaTeX goes in ``latex`` and ``text`` is left empty, so
    # ``Element.__post_init__`` renders it as ``$$...$$`` for the prose
    # projection while the structured field stays available to the chunker.
    if role == "formula":
        latex = _to_latex(text)
        if not latex:
            # The formula had a label and a bbox but nothing readable. Emitting
            # it as an empty formula would put a blank ``$$`` in the prose and
            # a `latex`-less formula in the corpus, which the eval counts as a
            # formula that did not survive. Dropping it says nothing, so it is
            # reported instead.
            return None, ISSUE_NO_CONTENT
        return Element(role=role, text="", locator=locator, latex=latex), issue
    level = _as_int(item.get("level")) if role == "heading" else None
    return Element(role=role, text=text, locator=locator, level=level), issue


def _caption_refs(payload: dict[str, Any]) -> list[str]:
    """Refs of the caption items attached to a picture or table."""
    captions = payload.get("captions")
    if not isinstance(captions, list):
        return []
    return [ref for ref in (_ref_of(entry) for entry in captions) if ref]


def _caption_text(payload: dict[str, Any], index: dict[str, dict[str, Any]]) -> str:
    """Collect a picture's / table's caption text.

    Docling has shipped two caption encodings: a reference to a ``texts`` item
    (the current form, where the caption is also a body child in its own right)
    and an inline ``{"text": ...}`` payload. Both are read, because a caption
    that silently disappears is a figure with no retrievable description.
    """
    captions = payload.get("captions")
    if not isinstance(captions, list):
        return ""
    parts: list[str] = []
    for entry in captions:
        if isinstance(entry, dict) and entry.get("text"):
            text = str(entry["text"]).strip()
            if text:
                parts.append(text)
            continue
        target = index.get(_ref_of(entry))
        if not target:
            continue
        text = str(target.get("text") or "").strip()
        if text:
            parts.append(text)
    return " ".join(parts)


def _table_element(
    payload: dict[str, Any],
    pages: dict[str, Any],
    index: dict[str, dict[str, Any]],
) -> tuple[Element | None, LocatorIssue]:
    table = build_table_from_docling(payload)
    caption = _caption_text(payload, index)
    if table.is_empty_shell and not caption:
        # An empty grid with no caption carries no retrievable content; the
        # quality gate reports ruled-lines-but-no-text separately, so dropping
        # it here does not hide a problem.
        return None, ""
    locator, issue = _locator_for(payload, pages)
    # ``Table.caption`` is set from the docling caption rather than left empty
    # even though the caption is *also* a ``texts`` entry: the caption text
    # item is folded into this table rather than standing as its own element,
    # so without this line the caption is not in the document at all. That is
    # content loss, not a rendering choice — "Table 1: frame classification
    # accuracy and WER…" is the only sentence that says what the grid is.
    table.caption = caption
    # ``text`` stays empty on purpose: ``Element.__post_init__`` fills it from
    # ``Table.to_markdown()``, which guarantees the prose projection and the
    # grid can never drift apart.
    return Element(role="table", text="", locator=locator, table=table), issue


def _picture_element(
    payload: dict[str, Any],
    pages: dict[str, Any],
    index: dict[str, dict[str, Any]],
) -> tuple[Element, LocatorIssue]:
    """A picture becomes a ``figure`` element whose text is its caption.

    The image bytes are not part of the canonical schema, so a figure with no
    caption has no text. It is still emitted — an addressable region on the
    page is worth more to a citation than nothing — and downstream projections
    skip the empty text.

    ``pictures[i].children`` also holds the text *inside* the figure (axis
    labels, in-plot annotations) alongside the caption. Those are ordinary
    ``texts`` entries that the body tree does not list, so the reading-order
    walk reaches them through here; without it they fall out of the ordered
    pass entirely and are appended at the end of the document, positionally.
    """
    locator, issue = _locator_for(payload, pages)
    return (
        Element(role="figure", text=_caption_text(payload, index), locator=locator),
        issue,
    )


def _opaque_item_element(
    item: dict[str, Any],
    pages: dict[str, Any],
) -> tuple[Element | None, LocatorIssue]:
    """Best-effort element for a ``key_value_items`` / ``form_items`` entry.

    **Unproven against real docling**: all 53 corpus dumps have both lists
    empty, so there is no real payload to read the shape from. The function
    exists because the alternative — not indexing those lists at all — means a
    document that *does* contain form fields has its content dropped with no
    trace, which is the failure mode this module is written against. It takes
    the same text path as any other item and, if the entry carries a ``key``
    and a ``value`` that are themselves refs into ``texts``, concatenates what
    they point at; when it cannot produce text it returns ``None`` and the
    caller counts it into a warning rather than emitting an empty element.
    """
    text = _item_text(item)
    if not text:
        # A key/value entry carries its halves as nested text items, not as
        # fields on itself. Read them as dicts (the real shape) and as plain
        # strings (a payload that inlined them), and join whatever came out.
        parts: list[str] = []
        for half in (item.get("key"), item.get("value")):
            if isinstance(half, dict):
                rendered = str(half.get("text") or half.get("orig") or "").strip()
            else:
                rendered = str(half or "").strip()
            if rendered:
                parts.append(rendered)
        text = " ".join(parts)
    if not text:
        return None, ISSUE_NO_CONTENT
    locator, issue = _locator_for(item, pages)
    return Element(role="text", text=text, locator=locator), issue


# --------------------------------------------------------------------------
# Reading order
# --------------------------------------------------------------------------


def _collect_ordered_refs(
    docling_dict: dict[str, Any],
    index: dict[str, dict[str, Any]],
    kinds: dict[str, str],
) -> tuple[list[str], bool]:
    """Walk the body tree to recover docling's reading order.

    Docling builds a tree: ``body.children`` points at groups (columns, lists,
    sections) and each group points at its own children in reading order. That
    ordering is the whole point of using docling over a text-layer extractor —
    it is what keeps a two-column page from interleaving line by line. So the
    tree is walked depth-first and every content ref is emitted in the order it
    is met, exactly once (a caption reachable both from a picture and from the
    body would otherwise be duplicated).

    Two properties of the real tree are load-bearing and were not visible in
    the fixture this was written against:

    * **A group's children are not in the body.** On the distillation paper
      ``body.children`` lists 94 refs of which one is ``#/groups/0``, and the
      nine reference-list items live only under that group. A walk that stops
      at the body loses them; they then reappear as positionally-sorted
      leftovers, i.e. a references section at the end of the paper.
    * **Non-group items have children too.** ``pictures[i].children`` and
      ``tables[i].children`` list the text inside the figure and the caption.
      Those ``texts`` entries are *not* body children, so they are equally
      invisible to a body-only walk. They are followed here, in place, which
      is also the order a reader would expect: a figure's own labels arrive
      with the figure, not after the whole document.

    Returns the ordered refs and whether the walk was usable. When it is not —
    no body, no groups, or nothing resolvable — the caller falls back to
    positional sorting and records a warning. A silent fallback would look
    identical to a correct parse while quietly scrambling every multi-column
    document, so this is never silent.
    """
    body = docling_dict.get("body")
    if not isinstance(body, dict):
        return [], False
    roots = body.get("children")
    if not isinstance(roots, list) or not roots:
        return [], False

    ordered: list[str] = []
    seen: set[str] = set()

    def walk(ref: str, depth: int) -> None:
        # Depth is bounded so a malformed payload with a self-referential group
        # cannot hang the parse.
        if depth > 32 or not ref or ref in seen:
            return
        seen.add(ref)
        item = index.get(ref)
        if item is None:
            return
        children = item.get("children")
        child_refs = children if isinstance(children, list) else []
        if kinds.get(ref) == "group":
            for child in child_refs:
                walk(_ref_of(child), depth + 1)
            return
        if child_refs:
            # The item itself is content *and* a container. Emit it first, then
            # its children, so a picture is followed by the text inside it
            # rather than the text replacing it.
            ordered.append(ref)
            for child in child_refs:
                walk(_ref_of(child), depth + 1)
            return
        ordered.append(ref)

    for root in roots:
        walk(_ref_of(root), 0)

    return (ordered, True) if ordered else ([], False)


#: Top-level lists that hold content, and the kind recorded for each. The
#: singular is derived by dropping the ``_items``/``es`` suffix, matching how
#: ``texts`` becomes ``text``.
_ITEM_KINDS: tuple[tuple[str, str], ...] = (
    ("texts", "text"),
    ("tables", "table"),
    ("pictures", "picture"),
    ("groups", "group"),
    ("key_value_items", "key_value"),
    ("form_items", "form"),
)


def _index_docling_items(
    docling_dict: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Build a ``self_ref`` index over every item kind we care about.

    Returns the index plus a parallel ``self_ref -> kind`` map. The kinds live
    beside the items rather than inside them for one reason: the caller's
    payload must come out of :func:`normalize_docling_document` unchanged.
    Annotating the dicts in place would be invisible here but visible in the
    raw artifact written moments later from the same object, and an artifact
    that claims to be docling's output should be exactly that.

    Items without a ``self_ref`` are skipped rather than given a synthetic one:
    nothing can point at them, so a made-up key could only ever collide.

    Three top-level keys are handled that a ``texts``/``tables``/``pictures``
    walk would miss, and each was a place where content could vanish without
    trace:

    ``furniture``
        A **dict, not a list** — one group node with a ``self_ref`` and a
        ``children`` list (empty for all 53 corpus dumps, because docling also
        lists the furniture items in ``texts`` with
        ``content_layer="furniture"``). Older payloads shipped it as a list of
        groups, so both shapes are accepted. It is indexed as a group so that
        anything it does contain is reachable and cannot be silently ignored.
    ``key_value_items`` / ``form_items``
        Content the body walk never visits, because they hang off the body
        rather than appearing in ``body.children``. Empty across the eval
        corpus, so their exact item shape is unverified — they are indexed so
        that their *presence* is countable, and anything that turns out to
        carry no readable text is reported rather than dropped quietly.
    """
    index: dict[str, dict[str, Any]] = {}
    kinds: dict[str, str] = {}

    def register(item: Any, kind: str) -> None:
        if not isinstance(item, dict):
            return
        ref = str(item.get("self_ref") or "")
        if not ref:
            return
        index.setdefault(ref, item)
        kinds.setdefault(ref, kind)

    for key, kind in _ITEM_KINDS:
        items = docling_dict.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            register(item, kind)

    furniture = docling_dict.get("furniture")
    if isinstance(furniture, dict):
        register(furniture, "group")
    elif isinstance(furniture, list):
        for item in furniture:
            register(item, "group")
    return index, kinds


def _positional_key(element: Element) -> tuple:
    """Fallback reading-order key: page, then top-to-bottom, then left.

    Our boxes are bottom-left PDF space, so the *top* edge of a box is the
    larger ``y``; sorting ascending on ``-y1`` walks down the page. Elements
    without geometry sort last within their page rather than first, because an
    unlocated paragraph is more likely to be a stray caption than the opening
    line of the article.
    """
    locator = element.locator
    bbox = locator.bbox
    if bbox is None:
        return (locator.page or 0, 1, 0.0, 0.0)
    return (locator.page or 0, 0, -bbox.y1, bbox.x0)


# --------------------------------------------------------------------------
# Normalization entry point
# --------------------------------------------------------------------------


def normalize_docling_document(
    docling_dict: dict[str, Any],
    *,
    source_filename: str,
    media_type: str,
    force_ocr: bool = False,
    options_hash: str = "",
    sha256: str = "",
    parser_version: str = "",
) -> ParsedDocument:
    """Turn a ``DoclingDocument.export_to_dict()`` payload into a document.

    This is the adapter's real work and its only interesting entry point. It is
    a pure function of ``docling_dict`` plus its keyword arguments: no docling
    import, no filesystem, no network. That is deliberate — it means the parts
    most likely to be wrong (coordinate conversion, merged-cell expansion,
    multi-column ordering) are covered by tests that run anywhere, and that a
    docling version bump shows up as a failing assertion rather than a wrong
    highlight nobody notices.

    ``force_ocr`` is recorded rather than acted on: it was already applied by
    the converter, and keeping it in the signature means the same payload can
    be re-normalized under a different options hash when the gateway retries
    with OCR forced.

    **Nothing is dropped quietly.** Every element that loses a locator records
    a :data:`LocatorIssue`, every dropped formula records
    :data:`ISSUE_NO_CONTENT`, and a ``pages`` map that is missing or of the
    wrong type says so. The counts become entries in ``document.warnings``.
    That is the contract that was missing when this function silently returned
    a document with 96 elements and not one of them addressable.
    """
    if not isinstance(docling_dict, dict):
        raise ValueError(
            "docling 返回的文档结构不是对象，无法归一化；"
            f"实际类型为 {type(docling_dict).__name__}。"
        )

    warnings: list[str] = []
    pages = docling_dict.get("pages")
    if not isinstance(pages, dict):
        # Docling's ``pages`` is a dict keyed by the string page number. An
        # absent or list-shaped one is not fatal — page numbers still come from
        # ``prov`` — but it removes the page heights that ``TOPLEFT`` boxes are
        # mirrored against, so it is worth saying out loud.
        if pages is not None:
            warnings.append(
                "docling 的 pages 字段不是以页码为键的字典，页面尺寸不可用；"
                f"实际类型为 {type(pages).__name__}。"
            )
        pages = {}

    index, kinds = _index_docling_items(docling_dict)
    ordered_refs, order_ok = _collect_ordered_refs(docling_dict, index, kinds)
    if not order_ok:
        warnings.append(
            "docling 未提供可解析的 groups 阅读顺序（body 缺失或为空），"
            "已降级为按 (页码, 纵坐标) 排序；多栏文档的段落顺序可能被打乱。"
        )
        ordered_refs = []

    # A caption is collected up front rather than as the picture or table that
    # owns it is reached, because the body tree may list the caption text item
    # *before* its owner — the usual encoding has it as the owner's own child,
    # but a payload that also lists it in the body would otherwise print the
    # figure description twice: once folded into the figure and once on its
    # own. Folding is decided by ownership, not by position, so the result does
    # not depend on which of the two encodings a given document used.
    folded_captions: set[str] = set()
    for ref, kind in kinds.items():
        if kind in {"picture", "table"}:
            folded_captions.update(_caption_refs(index[ref]))

    elements: list[Element] = []
    issues: dict[str, int] = {}
    opaque_seen = 0
    opaque_dropped = 0
    emitted: set[str] = set()

    def note(issue: LocatorIssue) -> None:
        if issue:
            issues[issue] = issues.get(issue, 0) + 1

    def build(ref: str) -> Element | None:
        """Build the element for ``ref``, or ``None`` if it carries nothing."""
        nonlocal opaque_seen, opaque_dropped
        item = index.get(ref)
        if item is None:
            return None
        kind = kinds.get(ref)
        if kind == "text":
            if ref in folded_captions:
                return None
            element, issue = _text_element(item, pages)
        elif kind == "table":
            element, issue = _table_element(item, pages, index)
        elif kind == "picture":
            element, issue = _picture_element(item, pages, index)
        elif kind in {"key_value", "form"}:
            element, issue = _opaque_item_element(item, pages)
        else:
            return None
        note(issue)
        if element is None:
            if kind in {"key_value", "form"}:
                opaque_dropped += 1
            return None
        if kind in {"key_value", "form"}:
            opaque_seen += 1
        return element

    for ref in ordered_refs:
        emitted.add(ref)
        element = build(ref)
        if element is not None:
            elements.append(element)

    # Anything the body tree never mentioned still has to be kept: docling
    # emits some regions (notably page furniture in older payloads) outside the
    # body group, and dropping them would quietly lose text. Caption text
    # reaches this list too, and is dropped by ``build`` for the same reason it
    # is dropped from the ordered pass.
    leftovers = [
        ref for ref, kind in kinds.items() if ref not in emitted and kind != "group"
    ]
    leftover_elements = [element for element in map(build, leftovers) if element is not None]
    # Leftovers have no tree order to inherit, so they are placed
    # positionally — but only among themselves, appended after the ordered
    # content, because interleaving them by position would reorder the body.
    leftover_elements.sort(key=_positional_key)
    elements.extend(leftover_elements)

    if issues.get(ISSUE_NO_PROV):
        count = issues[ISSUE_NO_PROV]
        warnings.append(
            f"有 {count} 个 docling 元素没有 prov 溯源信息，无法定位到页码；"
            "这些元素的内容已保留但不可高亮引用。"
        )
    if issues.get(ISSUE_NO_PAGE):
        count = issues[ISSUE_NO_PAGE]
        warnings.append(
            f"有 {count} 个 docling 元素的 prov 缺少可用的页码字段"
            f"（已尝试 {', '.join(PROV_PAGE_KEYS)}），无法定位到页码。"
            "这通常意味着 docling 版本再次修改了溯源结构。"
        )
    if issues.get(ISSUE_NO_BBOX):
        count = issues[ISSUE_NO_BBOX]
        warnings.append(
            f"有 {count} 个 docling 元素有页码但没有可用的 bbox，"
            "引用将只能定位到页，无法定位到具体区域。"
        )
    if issues.get(ISSUE_RELATIVE_BBOX):
        count = issues[ISSUE_RELATIVE_BBOX]
        warnings.append(
            f"有 {count} 个元素使用 TOPLEFT 坐标系但缺少页面高度，"
            "其 bbox 仅保留页内相对位置，绝对坐标可能偏移。"
        )
    if issues.get(ISSUE_NO_CONTENT):
        count = issues[ISSUE_NO_CONTENT]
        warnings.append(
            f"有 {count} 个被 docling 标记为公式的元素没有可读内容（text 与 orig 均为空），"
            "已丢弃而未作为空公式写入。"
        )
    if opaque_dropped:
        warnings.append(
            f"docling 报告了 {opaque_seen + opaque_dropped} 个 key_value/form 区域，"
            f"其中 {opaque_dropped} 个的内容无法表示为文本元素，已丢弃；"
            "原始 JSON 已完整保留在 raw artifact 中。"
        )

    page_count = len(pages)
    highest_page = max(
        (
            element.locator.page
            for element in elements
            if element.locator.kind == "pdf_page" and element.locator.page
        ),
        default=0,
    )
    # ``pages`` can be absent for formats docling treats as a single sheet, and
    # its keys are page *numbers* rather than a count when it is present — a
    # map with a hole in it is shorter than the document is long, so the
    # highest page actually seen is used as a floor either way.
    page_count = max(page_count, highest_page)

    if not elements:
        warnings.append("docling 未能从该文件提取到任何元素（可能是空白页或纯图片文档）。")

    # Heading context is tracked while walking in final order, so every element
    # knows the section path it appeared under. Docling does not report heading
    # depth, so ``level`` stays ``None`` and consumers must not depend on it.
    heading_path: list[str] = []
    for order, element in enumerate(elements):
        if element.role == "heading" and element.text:
            heading_path.append(element.text)
        element.heading_path = tuple(heading_path)
        element.order = order

    document = ParsedDocument(
        source_filename=source_filename,
        media_type=media_type,
        sha256=sha256,
        page_count=page_count,
        elements=elements,
        parser_name=PARSER_NAME,
        parser_version=parser_version,
        options_hash=options_hash,
        created_at=now_iso(),
        warnings=warnings,
    )
    # Units are one per page for a PDF, which is exactly what a reader asks for
    # when they click a citation. Derived here rather than supplied by the
    # adapter because docling has no unit concept of its own.
    document.build_units()
    return document


# --------------------------------------------------------------------------
# docling-serve transport
# --------------------------------------------------------------------------


def _encode_multipart(
    fields: dict[str, str],
    *,
    file_field: str,
    filename: str,
    content: bytes,
    content_type: str,
) -> tuple[bytes, str]:
    """Build a ``multipart/form-data`` body with the stdlib only.

    ``requests`` is not a dependency of this project, and hand-rolling the
    envelope is a few lines. The boundary is random per call so a document
    cannot inject one.
    """
    boundary = f"----PaperWhisperer{uuid.uuid4().hex}"
    marker = f"--{boundary}\r\n".encode("ascii")
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.append(marker)
        chunks.append(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        )
    chunks.append(marker)
    chunks.append(
        (
            f'Content-Disposition: form-data; name="{file_field}"; '
            f'filename="{filename}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode()
    )
    chunks.append(content)
    chunks.append(f"\r\n--{boundary}--\r\n".encode("ascii"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


#: Host names accepted as "this machine" without going through
#: :func:`ipaddress`. ``localhost`` is the documented name for the loopback
#: interface; the other two address the local machine without being loopback
#: per se — ``0.0.0.0`` is what a server binds when it listens on every
#: interface, and connecting to it lands on the local host.
_LOCAL_HOST_NAMES = frozenset({"localhost", "0.0.0.0", "::"})


def _is_local_host(host: str) -> bool:
    """Whether ``host`` names this machine.

    Accepts the whole loopback range rather than the two literals that happen
    to be common: ``127.0.0.2`` and ``::1`` both resolve to the loopback
    interface, and refusing them would be a refusal of a *local* service based
    on which of its addresses the operator happened to type. Anything that is
    not a name we recognise and not an address at all is not local, which is
    the answer that keeps the promise this method makes.
    """
    candidate = str(host or "").strip().lower().strip("[]")
    if not candidate:
        return False
    if candidate in _LOCAL_HOST_NAMES:
        return True
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


def _serve_endpoint(base_url: str) -> str:
    """The docling-serve conversion endpoint.

    The path is ``/v1/convert/file``, which is what docling-serve has exposed
    since 0.1 and still exposes at 1.35.0 — verified against a live server's
    ``/openapi.json``. There has never been a ``/v1/parse/file``: the route is
    named after what the service does (convert one file into N output
    formats), and the parallel ``/v1/chunk/*`` tree uses ``chunk``, not
    ``parse``. Posting there returns 404, and because a 404 is a non-retryable
    4xx the failure surfaced as ``docling 解析失败：HTTP Error 404`` — a parse
    error attributed to the *document*, when nothing was ever sent to a parser.
    """
    return f"{str(base_url).rstrip('/')}/v1/convert/file"


#: The keys that make a dict a serialized ``DoclingDocument``. A docling-serve
#: response is *not* itself one of these even though it contains one: the
#: document sits under ``document.json_content``, alongside the markdown and
#: HTML renderings of the same parse.
_DOCLING_DOCUMENT_KEYS = ("texts", "tables", "pictures", "pages", "body")


def _looks_like_docling_document(candidate: Any) -> bool:
    return isinstance(candidate, dict) and any(
        key in candidate for key in _DOCLING_DOCUMENT_KEYS
    )


def _extract_serve_document(payload: Any) -> dict[str, Any]:
    """Pull the DoclingDocument dict out of a docling-serve response.

    docling-serve wraps the document in a per-format envelope::

        {"status": "success", "document": {"filename": ..., "md_content": ...,
                                           "json_content": {...DoclingDocument...}},
         "errors": [], "processing_time": 17.8, "timings": {...}}

    so ``payload["document"]`` is a dict but is **not** the document, and
    returning it — which is what "is ``document`` a dict" does — hands
    normalization an object with no ``texts``, no ``pages`` and no ``body``.
    That does not raise; it silently yields a document with zero elements and
    full page coverage of nothing, which is the worst failure shape available:
    the parse reports success on an empty page.

    The real document is ``document.json_content``, which is sometimes a dict
    and sometimes a JSON *string* depending on version and on whether the
    response was rendered by the synchronous or the async path. Both are
    handled. Bare-document responses (no envelope) are still accepted.
    """
    if not isinstance(payload, dict):
        raise ValueError("docling-serve 返回的响应中找不到 DoclingDocument 结构。")

    envelope = payload.get("document")
    if isinstance(envelope, str):
        # A bare response whose only content is the document, as a JSON string.
        try:
            envelope = json.loads(envelope)
        except (TypeError, ValueError):
            envelope = None
    if _looks_like_docling_document(envelope):
        return envelope

    if isinstance(envelope, dict):
        for key in ("json_content", "document"):
            nested = envelope.get(key)
            if isinstance(nested, str):
                try:
                    nested = json.loads(nested)
                except (TypeError, ValueError):
                    continue
            if _looks_like_docling_document(nested):
                return nested

    if _looks_like_docling_document(payload):
        return payload

    raise ValueError("docling-serve 返回的响应中找不到 DoclingDocument 结构。")


def post_to_serve(
    base_url: str,
    file_path: str | Path,
    *,
    media_type: str,
    force_ocr: bool = False,
    timeout: float = 120.0,
    opener: Any = None,
) -> dict[str, Any]:
    """POST a file to docling-serve and return the DoclingDocument dict.

    Retry policy, and why it is what it is:

    * **429 is retried** using ``Retry-After`` when the server sends it, via the
      same :func:`compute_retry_after_delay` the metadata fetchers use — a
      shared backoff means a rate limit behaves the same everywhere in the app
      instead of having a second, subtly different curve.
    * **Transport errors and timeouts are retried** with exponential backoff.
      A docling-serve container that is still loading its models answers with a
      connection error for the first few seconds, and that is a startup
      transient, not a bad document.
    * **Other 4xx responses are not retried.** 400/413/415 mean the request
      itself is wrong; replaying it just burns the caller's timeout budget
      before producing the same error.

    Replaying the POST is safe because parsing is a pure function of the uploaded
    bytes — docling-serve keeps no per-request state we depend on.

    ``opener`` is resolved at call time (rather than bound as a default
    argument) so tests can monkeypatch ``urllib.request.urlopen``.
    """
    open_url = opener or urllib.request.urlopen
    path = Path(file_path)
    content = path.read_bytes()
    body, content_type = _encode_multipart(
        {"force_ocr": "true" if force_ocr else "false", "to_formats": "json"},
        file_field="files",
        filename=path.name or "document",
        content=content,
        content_type=media_type or "application/octet-stream",
    )
    url = _serve_endpoint(base_url)
    last_error: Exception | None = None

    for attempt in range(max(1, SERVE_MAX_ATTEMPTS)):
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": content_type,
                "Accept": "application/json",
                "Content-Length": str(len(body)),
            },
        )
        try:
            response = open_url(request, timeout=timeout)
            try:
                raw = response.read()
            finally:
                close = getattr(response, "close", None)
                if callable(close):
                    close()
            return _extract_serve_document(json.loads(raw.decode("utf-8", errors="ignore")))
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code == 404:
                # A 404 here is a configuration fault, not a document fault: the
                # service answered, so the URL is not merely unreachable, and the
                # file never reached a parser. Saying so is the difference
                # between an operator fixing a setting and an operator hunting
                # for a corrupt upload.
                raise ValueError(
                    f"docling-serve 在 {_serve_endpoint(base_url)} 上返回 404。"
                    "服务本身可达，但这个地址不存在——通常是 PARSER_DOCLING_SERVE_URL "
                    "指向了根路径之外的位置（例如填了带 /v1/convert 的完整端点），"
                    "或所接的并非 docling-serve。请只配置到主机与端口。"
                ) from exc
            if exc.code == 429 and attempt < SERVE_MAX_ATTEMPTS - 1:
                time.sleep(compute_retry_after_delay(exc, attempt, SERVE_RETRY_MAX_DELAY))
                continue
            if 400 <= exc.code < 500 and exc.code != 429:
                raise
            if attempt < SERVE_MAX_ATTEMPTS - 1:
                time.sleep(get_retry_delay_seconds(attempt, max_delay=SERVE_RETRY_MAX_DELAY))
                continue
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt < SERVE_MAX_ATTEMPTS - 1:
                time.sleep(get_retry_delay_seconds(attempt, max_delay=SERVE_RETRY_MAX_DELAY))
                continue
            raise

    raise RuntimeError(f"docling-serve 请求失败：{last_error}")


# --------------------------------------------------------------------------
# Raw artifact persistence
# --------------------------------------------------------------------------


def write_raw_artifact(
    docling_dict: dict[str, Any],
    *,
    sha256: str = "",
    options_hash: str = "",
    artifact_dir: str | None = None,
) -> str:
    """Persist the raw docling JSON and return its path.

    The raw payload is kept because it is the only way to reproduce a parse
    after the fact: when a highlight lands in the wrong place three weeks later,
    the fix has to be testable against the exact bytes the backend produced, and
    re-running docling may give a different answer on a different model
    version. The filename is derived from the content hash and options hash
    rather than a UUID, so re-parsing the same file with the same options
    overwrites rather than accumulating near-duplicates.

    Failures are logged and swallowed: a missing artifact degrades
    reproducibility, but failing the user's parse over it would not be a
    sensible trade.
    """
    base_dir = artifact_dir or getattr(config, "PARSER_ARTIFACT_DIR", "context/parser_artifacts")
    stem = (sha256[:16] or uuid.uuid4().hex[:16]) + (f"-{options_hash[:8]}" if options_hash else "")
    target = Path(base_dir) / f"{stem}.docling.json"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(str(target), docling_dict)
    except OSError:
        logger.warning("写入 docling raw artifact 失败：%s", target, exc_info=True)
        return ""
    return str(target)


# --------------------------------------------------------------------------
# Result plumbing
# --------------------------------------------------------------------------


def _build_result(**kwargs: Any) -> Any:
    """Return a ``gateway.ParseResult`` when it exists, else the local mirror.

    The gateway dispatches to adapters, so importing it at module scope would
    be circular. Resolving the class here — at call time, inside a ``try`` —
    means the adapter works standalone today and upgrades to the real contract
    the instant the gateway lands, with no edit to this file.
    """
    try:
        from paperwhisperer.ingestion.gateway import ParseResult  # noqa: PLC0415
    except Exception:  # pragma: no cover - gateway 尚未落地时的正常路径
        return DoclingParseResult(**kwargs)
    try:
        return ParseResult(**kwargs)
    except TypeError:
        # A gateway that has since renamed a field must not take the parse down
        # with it; the local mirror still carries every value.
        logger.warning("gateway.ParseResult 签名不匹配，回退到本地结果对象", exc_info=True)
        return DoclingParseResult(**kwargs)


def _detect_docling_version() -> str:
    """Read the installed docling version without importing it.

    ``importlib.metadata`` reads distribution metadata from disk, so asking for
    the version does not pull torch into the process. A build without docling
    reports ``"unknown"`` rather than failing — the version is diagnostic, not
    a precondition.
    """
    module = sys.modules.get("docling")
    version = getattr(module, "__version__", "") if module is not None else ""
    if version:
        return str(version)
    try:
        from importlib.metadata import PackageNotFoundError
        from importlib.metadata import version as dist_version

        try:
            return dist_version("docling")
        except PackageNotFoundError:
            return "unknown"
    except Exception:
        return "unknown"


def _sha256_of(file_path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_options_hash(
    *,
    force_ocr: bool,
    mode: str,
    timeout_seconds: float,
) -> str:
    """Hash the parse-affecting options.

    The parse cache is keyed on ``(sha256, parser, version, options_hash)``, so
    this must cover exactly the inputs that change the output. Timeout is
    included because a timed-out parse produces a different artifact than a
    completed one, and reusing that cache entry would hide the failure.
    """
    payload = json.dumps(
        {"force_ocr": bool(force_ocr), "mode": str(mode), "timeout": float(timeout_seconds)},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _resolve_converter_class(docling_module: Any) -> Any:
    """Find ``DocumentConverter`` in docling, which has moved.

    On docling 2.131.0 (verified against a real install) the class is NOT a
    top-level attribute: ``docling.DocumentConverter`` does not exist, and the
    import is ``from docling.document_converter import DocumentConverter``.
    Reading only the top-level attribute -- which is what this used to do --
    raised on every real parse, and the failure was invisible until now because
    docling was never installed on the machine the tests ran on: the gateway
    degraded to the legacy parser and returned ``partial``, which is correct
    behaviour for a missing optional dependency and completely wrong for a
    present one that is being looked up in the wrong place.

    The submodule is tried first because it is the documented path for 2.x;
    the top-level attribute is kept because earlier releases exported it there.
    """
    import importlib  # noqa: PLC0415 - only needed on this branch

    try:
        module = importlib.import_module("docling.document_converter")
    except Exception:  # noqa: BLE001 - a missing submodule is just a miss
        module = None
    for candidate in (module, docling_module):
        if candidate is None:
            continue
        converter = getattr(candidate, "DocumentConverter", None)
        if converter is not None:
            return converter
    return None


#: ``ConversionStatus`` values that mean docling produced no document. Compared
#: as text rather than by importing the enum, so this function stays usable on
#: a machine where docling is not installed — which is every machine in CI.
_FAILED_STATUSES = frozenset({"failure", "failed", "skipped", "error"})


def _conversion_failed(result: Any) -> bool:
    """Whether a docling ``ConversionResult`` means the parse did not happen.

    ``pending`` and ``started`` are not read: a synchronous ``convert`` cannot
    return either, and treating an unknown status as a failure would turn a
    docling release that renames the enum into a total outage. An absent status
    is likewise not a failure, because a stubbed converter in a test returns an
    object with no ``status`` at all and a real document.
    """
    status = str(getattr(result, "status", "") or "").strip().lower()
    # ``ConversionStatus`` is a str enum, so ``str()`` may render it as
    # ``conversionstatus.failure``; the last segment is the value.
    return status.rsplit(".", 1)[-1] in _FAILED_STATUSES


class _OptionAPIError(RuntimeError):
    """The installed docling's converter-option API is not one this adapter knows.

    Kept distinct from every other failure here because it has a different
    meaning and a different fix. "Docling will not accept this option" says the
    *document* cannot be OCR'd by this build; "this adapter cannot build the
    option" says the *adapter* is out of step with the installed docling and
    should be updated. Collapsing the two into one catch-all is how the forced
    OCR path ended up claiming to degrade gracefully while in fact failing the
    whole parse.
    """


def _find_docling_attr(*module_and_name: tuple[str, str]) -> Any:
    """First importable attribute among ``(module, attribute)`` pairs, else raise.

    docling has moved these between releases, and guessing wrong is a
    compatibility problem this adapter reports rather than hides.
    """
    import importlib  # noqa: PLC0415 - only needed on this branch

    tried: list[str] = []
    for module_name, attribute in module_and_name:
        tried.append(f"{module_name}.{attribute}")
        try:
            module = importlib.import_module(module_name)
        except Exception:  # noqa: BLE001 - a missing module is just a miss
            continue
        found = getattr(module, attribute, None)
        if found is not None:
            return found
    raise _OptionAPIError(
        "以下 docling 接口均不可用：" + "、".join(tried) + "；当前 docling 版本的"
        "转换选项结构与本适配器已知的不一致。"
    )


def _build_ocr_format_option() -> tuple[Any, Any]:
    """Build the ``InputFormat.PDF`` option that turns OCR on.

    **The subtlety that made this fail.** ``DocumentConverter(format_options=…)``
    takes a mapping of ``InputFormat -> FormatOption``, *not* of
    ``InputFormat -> PipelineOptions``. ``PdfFormatOption`` is the wrapper that
    carries a pipeline class, a document backend and the
    ``PdfPipelineOptions``; handing docling a bare ``PdfPipelineOptions`` looks
    like it works — the constructor is a plain dataclass assignment and raises
    nothing — and then fails at pipeline-initialization time inside
    ``convert()`` with ``AttributeError: 'PdfPipelineOptions' object has no
    attribute 'pipeline_options'`` (``backend`` on the ``InputFormat.IMAGE``
    branch, ``pipeline_cls``/``pipeline_options`` in ``_get_pipeline``; the
    exact name depends on which code path touches it first, the cause is the
    same). Because the failure arrives *after* this function returns, a
    ``try``/``except`` around the construction can never catch it: the
    whole parse fails instead of degrading, which is the opposite of what the
    old warning claimed was happening.

    ``PdfFormatOption`` lives in ``docling.document_converter`` in 2.131.0; the
    ``datamodel`` spelling is tried as well because that is where its sibling
    option classes live, and both misses are reported as one
    :class:`_OptionAPIError` rather than a generic failure.
    """
    input_format = _find_docling_attr(("docling.datamodel.base_models", "InputFormat"))
    pdf_pipeline_options = _find_docling_attr(
        ("docling.datamodel.pipeline_options", "PdfPipelineOptions")
    )
    pdf_format_option = _find_docling_attr(
        ("docling.document_converter", "PdfFormatOption"),
        ("docling.datamodel.pipeline_options", "PdfFormatOption"),
    )

    pipeline_options = pdf_pipeline_options()
    # Assigned rather than passed to the constructor because the field is
    # pydantic-validated on construction and this is the spelling every docling
    # 2.x backend has agreed on.
    pipeline_options.do_ocr = True
    pdf_format = getattr(input_format, "PDF", None)
    if pdf_format is None:
        raise _OptionAPIError("docling 的 InputFormat 中没有 PDF，无法按格式设置 OCR 选项。")
    return pdf_format_option(pipeline_options=pipeline_options), pdf_format


def _ocr_is_active(converter: Any) -> bool | None:
    """Whether the built converter really has OCR on. ``None`` if unknowable.

    Construction succeeding is not evidence that the option took: docling
    resolves per-format options lazily, and a converter can be built and
    reported as configured while the pipeline it will actually run was chosen
    by a different default. Reading the resolved option back is the only check
    that answers the question, and it is cheap because it touches no model.
    """
    resolved = getattr(converter, "format_to_options", None)
    if not isinstance(resolved, dict):
        return None
    for option in resolved.values():
        pipeline_options = getattr(option, "pipeline_options", None)
        do_ocr = getattr(pipeline_options, "do_ocr", None)
        if do_ocr is not None:
            return bool(do_ocr)
    return None


def _build_converter(docling_module: Any, *, force_ocr: bool) -> tuple[Any, list[str]]:
    """Create a ``DocumentConverter``, honouring ``force_ocr`` when possible.

    Docling exposes OCR as a per-format option rather than a converter-wide
    flag. Three outcomes, each with its own message, because they have
    different fixes:

    * **Applied.** The option was built, the converter accepted it, and reading
      it back says ``do_ocr`` is on. No warning.
    * **This adapter cannot build the option** (:class:`_OptionAPIError`) —
      a compatibility problem. The plain converter is used and the warning
      names the adapter/docling version gap, so the fix is "update the
      adapter", not "the document is unparseable".
    * **Built but not in effect** — docling accepted an option and did not
      apply it, or applied it and the read-back disagrees. The warning says
      OCR may be missing, which is a document-quality problem.

    Note for whoever tunes this next: on docling 2.131.0 ``PdfPipelineOptions``
    already defaults to ``do_ocr=True``, so a *plain* converter OCRs anyway and
    ``force_ocr`` is close to a no-op there. It is still set explicitly, because
    that default is not a contract and other releases differ — but it means
    ``force_ocr=False`` is not a promise that OCR is off, and should not be
    documented as one.
    """
    warnings: list[str] = []
    converter_cls = _resolve_converter_class(docling_module)
    if converter_cls is None:
        raise RuntimeError(
            "docling 包中找不到 DocumentConverter（已尝试 docling.document_converter "
            "与 docling 顶层），当前 docling 版本可能不兼容。"
        )
    if not force_ocr:
        return converter_cls(), warnings

    try:
        option, pdf_format = _build_ocr_format_option()
    except _OptionAPIError as exc:
        logger.warning("force_ocr 无法通过 docling 格式选项应用", exc_info=True)
        warnings.append(
            f"force_ocr 未生效，已按普通模式解析：{exc}"
            "这是适配器与所安装 docling 版本的接口差异，不是文档本身的问题。"
        )
        return converter_cls(), warnings

    try:
        converter = converter_cls(format_options={pdf_format: option})
    except Exception:  # noqa: BLE001 - any construction failure degrades
        logger.warning("docling 拒绝了 force_ocr 格式选项", exc_info=True)
        warnings.append(
            "force_ocr 未生效：当前 docling 版本拒绝了该格式选项，已按普通模式解析。"
        )
        return converter_cls(), warnings

    active = _ocr_is_active(converter)
    if active is False:
        warnings.append(
            "force_ocr 已提交给 docling，但回读后 do_ocr 仍为关闭；"
            "该文档可能未经 OCR 解析。"
        )
    elif active is None:
        warnings.append(
            "force_ocr 已提交给 docling，但无法回读解析器选项以确认其生效；"
            "该文档可能未经 OCR 解析。"
        )
    return converter, warnings


# --------------------------------------------------------------------------
# The adapter
# --------------------------------------------------------------------------


class DoclingParser:
    """Adapter implementing the gateway's ``DocumentParser`` protocol.

    The gateway accepts this object structurally: it reads ``name``,
    ``version``, ``supports()`` and ``parse()``. Nothing here imports the
    gateway module, so the two can be developed, tested and deployed
    independently.

    Two execution modes, selected by ``PARSER_DOCLING_MODE``:

    ``inprocess`` (default)
        Import docling and convert in this process. Best quality and no
        service to operate, at the cost of a multi-hundred-megabyte dependency
        and a slow first run while models download.

    ``serve``
        POST the file to a docling-serve instance. Keeps torch out of the app
        process and moves the memory spike to a container that can be
        restarted independently.

    On privacy: ``serve`` talks to ``PARSER_DOCLING_SERVE_URL``, which
    defaults to ``http://localhost:5001``. ``DOCUMENT_PRIVACY_MODE=local_only``
    forbids *cloud* egress, and a loopback call to a container on the same
    machine is not that — the document never leaves the host either way. The
    mode is therefore **not** gated on the privacy setting or on cloud consent.
    It would be wrong to gate it, because doing so would make a local-only
    deployment unable to use a local service; and it would be worse not to
    notice if an operator pointed ``PARSER_DOCLING_SERVE_URL`` at a remote
    host, so :meth:`resolve_serve_url` refuses a non-loopback host outright and says
    why. Cloud parsers with a real consent check are a separate adapter.
    """

    name = PARSER_NAME

    #: Media types this adapter claims. Exposed as a class attribute so the
    #: router can filter without instantiating anything.
    supported_media_types = SUPPORTED_MEDIA_TYPES

    def __init__(self, mode: str | None = None, serve_url: str | None = None):
        self.mode = mode or getattr(config, "PARSER_DOCLING_MODE", "inprocess")
        self._serve_url_override = serve_url

    @property
    def version(self) -> str:
        return _detect_docling_version()

    def supports(self, media_type: str) -> bool:
        """Whether this parser can be asked to handle ``media_type``."""
        normalized = str(media_type or "").split(";")[0].strip().lower()
        return normalized in self.supported_media_types

    def resolve_serve_url(self) -> str:
        """Resolve and validate the docling-serve base URL.

        The host is checked with :mod:`ipaddress` rather than a hand-rolled
        string match, because the string that reaches us is a URL, not a bare
        host: an IPv6 literal arrives as ``[::1]:5001``, and splitting on the
        first colon turns that into the string ``"["`` — which is not a loopback
        address, so a perfectly local docling-serve bound to IPv6 would be
        refused as if it were a remote one. ``urlsplit`` also unwraps the
        brackets, so the address test below sees a plain address.

        ``0.0.0.0`` is accepted alongside the loopback range because connecting
        to it reaches the local host on every platform this app runs on, and an
        operator who wrote it meant "the service next door".
        """
        base = self._serve_url_override or getattr(
            config, "PARSER_DOCLING_SERVE_URL", ""
        )
        base = str(base or "").strip() or "http://localhost:5001"
        parts = urlsplit(base)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise ValueError(
                f"docling-serve 地址 {base} 不是合法的 http(s) URL，"
                "无法解析出主机名；请配置形如 http://localhost:5001 的地址。"
            )
        if not _is_local_host(parts.hostname):
            # A remote docling-serve would be an unconsented cloud egress, which
            # is exactly what local_only exists to prevent. Refusing here is
            # cheaper than discovering it in a privacy audit.
            raise ValueError(
                f"docling-serve 地址 {base} 不是本机地址。远程 docling 服务属于出网解析，"
                "本适配器只允许 loopback；如需云端解析请使用专门的云 parser 并走隐私策略检查。"
            )
        return base

    def _fetch_inprocess(self, file_path: str, *, force_ocr: bool) -> tuple[dict, list[str]]:
        try:
            import docling  # noqa: PLC0415
            from docling.document_converter import DocumentConverter  # noqa: F401,PLC0415
        except ImportError as exc:
            raise RuntimeError(MISSING_DOCLING_MESSAGE) from exc

        converter, warnings = _build_converter(docling, force_ocr=force_ocr)
        result = converter.convert(file_path)
        # ``convert`` does not raise on a document it could not handle: it
        # returns a result whose ``status`` is ``failure`` and whose ``document``
        # is an *empty* ``DoclingDocument``, not ``None``. Normalizing that would
        # produce a parse reported as successful with zero elements and a
        # "no elements extracted" warning, which is the same silent degradation
        # this module exists to prevent, one level up. A ``partial_success``
        # carries real content, so it is kept along with its errors.
        if _conversion_failed(result):
            errors = "; ".join(str(item) for item in (getattr(result, "errors", None) or []))
            raise RuntimeError(
                "docling 转换该文件失败"
                + (f"：{errors}" if errors else "（docling 未给出具体错误信息）。")
            )
        document = getattr(result, "document", None)
        if document is None:
            raise RuntimeError("docling 转换结果中没有 document 字段。")
        return document.export_to_dict(), warnings

    def parse(self, file_path: str, options: Any = None) -> Any:
        """Parse ``file_path`` and return a gateway-compatible result object.

        Never raises for an expected failure: a missing backend, an unreachable
        docling-serve or a corrupt file all come back as a result with
        ``status != "ok"`` and a Chinese ``error_message``. The gateway turns
        that into a fallback decision or a user-facing message, and an
        exception here would bypass both.
        """
        started = time.monotonic()
        force_ocr = bool(
            getattr(options, "force_ocr", None)
            if options is not None
            else getattr(config, "PARSER_FORCE_OCR", False)
        )
        timeout = float(
            getattr(options, "timeout_seconds", None)
            or getattr(config, "PARSER_TIMEOUT_SECONDS", 120)
        )
        media_type = getattr(options, "media_type", "") if options is not None else ""
        if not media_type:
            # Guarded even though the helper is total: ``parse`` promises never
            # to raise for an expected failure, and this is the one call on
            # that path that inspects an unvalidated caller-supplied string.
            try:
                media_type = _guess_media_type(file_path)
            except Exception:  # noqa: BLE001 - a bad name is not a parse failure
                media_type = "application/octet-stream"
        options_hash = str(getattr(options, "options_hash", "") or "") or compute_options_hash(
            force_ocr=force_ocr, mode=self.mode, timeout_seconds=timeout
        )
        parser_version = self.version

        def elapsed_ms() -> int:
            return int((time.monotonic() - started) * 1000)

        def failure(message: str) -> Any:
            return _build_result(
                status="failed",
                document=None,
                quality=None,
                raw_artifact_uri="",
                warnings=[],
                duration_ms=elapsed_ms(),
                parser_name=PARSER_NAME,
                parser_version=parser_version,
                options_hash=options_hash,
                error_message=message,
            )

        if not Path(file_path).exists():
            return failure(f"待解析文件不存在：{file_path}")

        try:
            digest = _sha256_of(file_path)
        except OSError as exc:
            return failure(f"读取待解析文件失败：{exc}")

        try:
            if self.mode == "serve":
                payload = post_to_serve(
                    self.resolve_serve_url(),
                    file_path,
                    media_type=media_type,
                    force_ocr=force_ocr,
                    timeout=timeout,
                )
                warnings: list[str] = []
            else:
                payload, warnings = self._fetch_inprocess(file_path, force_ocr=force_ocr)
        except ValueError as exc:
            # Configuration problems (a remote serve URL) are not parse
            # failures the gateway should retry differently — but they must not
            # be exceptions either.
            return failure(str(exc))
        except Exception as exc:
            logger.warning("docling 解析失败：%s", file_path, exc_info=True)
            return failure(f"docling 解析失败：{exc}")

        try:
            document = normalize_docling_document(
                payload,
                source_filename=Path(file_path).name,
                media_type=media_type,
                force_ocr=force_ocr,
                options_hash=options_hash,
                sha256=digest,
                parser_version=parser_version,
            )
        except Exception as exc:
            logger.warning("docling 结果归一化失败：%s", file_path, exc_info=True)
            return failure(f"docling 解析结果归一化失败：{exc}")

        artifact_uri = write_raw_artifact(
            payload, sha256=digest, options_hash=options_hash
        )
        warnings.extend(document.warnings)

        return _build_result(
            status="ok",
            document=document,
            quality=None,
            raw_artifact_uri=artifact_uri,
            warnings=warnings,
            duration_ms=elapsed_ms(),
            parser_name=PARSER_NAME,
            parser_version=parser_version,
            options_hash=options_hash,
            error_message="",
        )


def _guess_media_type(file_path: str | Path) -> str:
    """Best-effort media type from the file extension.

    Only the suffix is consulted: the gateway has already validated the
    signature by the time an adapter runs, so this only needs to pick the right
    parser and content type for the upload envelope.

    The suffix table is *derived* from :data:`SUPPORTED_MEDIA_TYPES` rather than
    written out, and it maps an extension to the media type — the previous
    version called ``.items()`` on a ``frozenset``, which raises
    ``AttributeError`` for every extension ``mimetypes`` does not know. Since
    this is called before the parse is guarded, that ``AttributeError`` escaped
    :meth:`DoclingParser.parse` entirely, on a document whose type the caller
    had every reason to be unsure about.
    """
    import mimetypes  # noqa: PLC0415

    guessed, _encoding = mimetypes.guess_type(str(file_path))
    if guessed:
        return guessed
    suffix = Path(file_path).suffix.lower().lstrip(".")
    by_suffix = {
        media_type.rsplit("/", 1)[-1]: media_type
        for media_type in SUPPORTED_MEDIA_TYPES
        if "/" in media_type
    }
    return by_suffix.get(suffix, "application/octet-stream")
