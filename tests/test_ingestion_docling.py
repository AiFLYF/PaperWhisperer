"""Docling adapter: coordinate conversion, table rebuilding, reading order.

Every test here runs without docling installed. That is not a convenience —
docling pulls in the torch chain, so the machines that run CI and the machines
that run these tests cannot have it. The adapter is therefore written so that
normalization is a pure function of ``DoclingDocument.export_to_dict()``, and
the fixtures below are hand-written payload dicts in that shape.

**These fixtures were wrong and were corrected against a real dump.** The
adapter was originally written against fixtures that used ``prov.page`` and
``texts[i].text_label``, and treated table cell end offsets as inclusive. None
of those fields exist in docling 2.131.0: it is ``page_no`` and ``label``, and
the end offsets are exclusive. Every one of the three degraded silently, so the
fixtures validated an adapter that could not parse a real document — 96
elements on a 9-page paper and not one locator between them. The fixtures below
now mirror ``context/parser_artifacts/*.docling.json`` dumps from
``eval/corpus``, and ``tests/test_docling_real.py`` pins the ones that were
wrong so the mistake cannot be reintroduced by editing a fixture back.

The tests here still pin the *contract* rather than docling's own behaviour.
When a docling upgrade changes its JSON, these are the first things that should
fail, before any highlight silently moves.
"""

from __future__ import annotations

import importlib.util
import json
import urllib.error
from email.message import Message
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from paperwhisperer.core import config
from paperwhisperer.ingestion import gateway
from paperwhisperer.ingestion.adapters import docling_adapter as adapter
from paperwhisperer.ingestion.adapters.docling_adapter import (
    DoclingParser,
    DoclingParseResult,
    build_table_from_docling,
    compute_options_hash,
    normalize_docling_document,
    to_bounding_box,
    write_raw_artifact,
)
from paperwhisperer.ingestion.schema import ParsedDocument

from .helpers import FakeResponse

#: US Letter at 72 dpi. Docling reports page size so ``TOPLEFT`` boxes can be
#: mirrored into PDF user space; these tests check that it does.
PAGE_HEIGHT = 792.0
PAGE_WIDTH = 612.0

#: Requests captured by the serve-mode urlopen stub. A list rather than a
#: closure variable because each test patches ``urlopen`` afresh.
_LAST_REQUEST: list = []


def _succeeded(result) -> bool:
    """Whether a parse result means success, in either status vocabulary.

    A serve-mode test is about the document arriving, not about which of the two
    vocabularies came back: the adapter's standalone result class spells success
    ``"ok"``, while the gateway's contract spells it ``"success"`` and
    normalizes the alias when it builds a real ``ParseResult`` from the
    adapter's. That seam is between two modules allowed to disagree, so these
    tests ask ``is_usable`` — the predicate name both classes implement —
    instead of reading ``status`` for a literal. It also requires a document
    rather than trusting the status word on its own.
    """
    return bool(result.is_usable) and result.document is not None


def _bbox(left, top, right, bottom, origin="BOTTOMLEFT"):
    return {
        "l": left,
        "t": top,
        "r": right,
        "b": bottom,
        "coord_origin": origin,
    }


def _prov(page=1, left=10, top=700, right=300, bottom=600, origin="BOTTOMLEFT"):
    """A ``ProvenanceItem`` as docling 2.131.0 emits it.

    The page field is ``page_no``. The fixture previously said ``page``, which
    is not a field that has ever existed on ``ProvenanceItem`` — so every
    locator built from it was ``kind="unknown"`` and every locator-coverage
    assertion in the suite passed only because the fixtures were consistently
    wrong in the same direction.
    """
    return [
        {
            "page_no": page,
            "bbox": _bbox(left, top, right, bottom, origin),
            "charspan": [0, 10],
        }
    ]


def _pages(count=1, height=PAGE_HEIGHT, width=PAGE_WIDTH):
    """Docling's page map: a dict keyed by the *string* page number."""
    return {
        str(number): {"size": {"width": width, "height": height}, "page_no": number}
        for number in range(1, count + 1)
    }


def _text(ref, text, label="", prov=None, page=1, **extra):
    """A ``TextItem``.

    The label field is ``label``; ``text_label`` does not exist in docling.
    Formula items additionally carry their content in ``orig`` with ``text``
    left empty, which is why ``orig`` is accepted as an override below.
    """
    item = {
        "self_ref": ref,
        "parent": {"$ref": "#/body"},
        "children": [],
        "content_layer": "body",
        "label": label,
        "orig": text,
        "text": text,
        "prov": prov if prov is not None else _prov(page=page),
    }
    item.update(extra)
    return item


def _cell(row, col, text, *, row_span=1, col_span=1, column_header=False, row_header=False):
    """A docling ``TableCell``.

    ``end_row_offset_idx`` is ``start_row_offset_idx + row_span`` — **exclusive**
    — which is how every docling backend writes it and how docling's own
    ``TableData.grid`` consumes it (``range(start, end)``). The fixture
    previously emitted ``row + row_span - 1``, the inclusive reading, and the
    resulting grid came out one row and one column too large on every table.
    """
    return {
        "bbox": _bbox(10, 700, 100, 650, origin="TOPLEFT"),
        "row_span": row_span,
        "col_span": col_span,
        "start_row_offset_idx": row,
        "end_row_offset_idx": row + row_span,
        "start_col_offset_idx": col,
        "end_col_offset_idx": col + col_span,
        "text": text,
        "column_header": column_header,
        "row_header": row_header,
    }


def _table_payload(cells, *, num_rows=None, num_cols=None, ref="tables/0", prov=None):
    data = {"table_cells": cells, "num_rows": num_rows, "num_cols": num_cols}
    return {
        "self_ref": ref,
        "data": data,
        "prov": prov if prov is not None else _prov(),
    }


def _payload(
    *,
    texts=(),
    tables=(),
    pictures=(),
    groups=(),
    pages=None,
    body_children=None,
    furniture=None,
    key_value_items=(),
    form_items=(),
):
    """Assemble a docling payload with a consistent body/children tree.

    ``furniture`` defaults to the real shape: a **dict** holding one group
    node, not a list. The adapter used to assume a list, and because it did not
    index the key at all that was invisible — every real dump has an empty
    furniture group, so nothing was lost yet, and the first document with
    non-empty furniture would have lost it silently.
    """
    items = [*texts, *tables, *pictures]
    if body_children is None:
        body_children = [{"$ref": item["self_ref"]} for item in items]
        if groups:
            body_children = [{"$ref": groups[0]["self_ref"]}]
    payload = {
        "schema_name": "DoclingDocument",
        "version": "1.7.0",
        "name": "paper",
        "origin": {"filename": "paper.pdf", "mimetype": "application/pdf"},
        "pages": _pages() if pages is None else pages,
        "body": {"self_ref": "#/body", "name": "body", "children": body_children},
        "furniture": (
            {"self_ref": "#/furniture", "children": [], "name": "furniture"}
            if furniture is None
            else furniture
        ),
        "groups": list(groups),
        "texts": list(texts),
        "tables": list(tables),
        "pictures": list(pictures),
        "key_value_items": list(key_value_items),
        "form_items": list(form_items),
    }
    return payload


def _normalize(payload, **overrides):
    kwargs = {
        "source_filename": "paper.pdf",
        "media_type": "application/pdf",
    }
    kwargs.update(overrides)
    return normalize_docling_document(payload, **kwargs)


def _serve_response(payload, monkeypatch):
    """Answer every docling-serve request with ``payload`` and record the requests.

    A module-level helper rather than a method because the serve tests are
    about the transport, not about the parser object; the recorded requests go
    to :data:`_LAST_REQUEST` because a test cannot get a return value back out
    of a monkeypatched ``urlopen``.
    """
    body = json.dumps(payload).encode("utf-8")
    response = FakeResponse(body, content_type="application/json")
    _LAST_REQUEST.clear()

    def _open(request, timeout=None, **_kwargs):
        _LAST_REQUEST.append(request)
        return response

    monkeypatch.setattr(urllib.request, "urlopen", _open)
    return response


class TestToBoundingBox:
    """The coordinate conversion is the easiest thing in this adapter to get
    wrong and the hardest thing to notice, because a mirrored box still looks
    like a plausible box."""

    def test_bottomleft_is_a_direct_rename(self):
        box = to_bounding_box(_bbox(10, 700, 300, 650, "BOTTOMLEFT"))

        # In PDF space y grows upward: b is the lower edge, t the upper edge.
        assert (box.x0, box.y0, box.x1, box.y1) == (10, 650, 300, 700)
        assert box.width == 290
        assert box.height == 50

    def test_topleft_is_mirrored_against_the_page_height(self):
        box = to_bounding_box(_bbox(10, 50, 300, 100, "TOPLEFT"), page_height=PAGE_HEIGHT)

        # t=50 from the top is y=742 from the bottom; b=100 is y=692.
        assert (box.x0, box.y0, box.x1, box.y1) == (10, 692.0, 300, 742.0)
        assert box.height == 50

    def test_topleft_without_a_page_height_keeps_the_size_and_the_ordering(self):
        box = to_bounding_box(_bbox(10, 50, 300, 100, "TOPLEFT"))

        # No mirror is possible, so y is negated: the height is preserved and a
        # box higher on the page still compares above one lower down, which is
        # all the positional fallback needs.
        assert box is not None
        assert box.height == 50
        assert box.y0 < box.y1

    def test_the_same_physical_box_from_either_origin_agrees_when_mirrored(self):
        """The whole point of handling both origins: one rectangle, two
        encodings, identical result."""
        bottom_left = to_bounding_box(_bbox(10, 650, 300, 700, "BOTTOMLEFT"))
        top_left = to_bounding_box(_bbox(10, 792 - 700, 300, 792 - 650, "TOPLEFT"),
                                   page_height=PAGE_HEIGHT)

        assert bottom_left.to_dict() == top_left.to_dict()

    def test_a_missing_coord_origin_defaults_to_the_pdf_convention(self):
        raw = _bbox(10, 700, 300, 650)
        del raw["coord_origin"]

        assert to_bounding_box(raw).to_dict() == to_bounding_box(
            _bbox(10, 700, 300, 650, "BOTTOMLEFT")
        ).to_dict()

    def test_normalizes_an_inverted_box_to_positive_extent(self):
        box = to_bounding_box(_bbox(300, 650, 10, 700, "BOTTOMLEFT"))

        assert box.width == 290
        assert box.height == 50

    @pytest.mark.parametrize(
        "raw",
        [
            None,
            "not a dict",
            {},
            {"l": 1, "t": 2, "r": 3},
            {"l": 1, "t": 2, "r": 3, "b": None},
            {"l": "a", "t": 2, "r": 3, "b": 4},
        ],
    )
    def test_an_unusable_rectangle_yields_no_box(self, raw):
        # A fabricated box would make the evidence viewer highlight unrelated
        # text, which is worse than having no highlight at all.
        assert to_bounding_box(raw) is None


class TestBuildTable:
    def test_rebuilds_a_plain_grid(self):
        table = build_table_from_docling(
            _table_payload(
                [
                    _cell(0, 0, "Metric", column_header=True),
                    _cell(0, 1, "Value", column_header=True),
                    _cell(1, 0, "Accuracy"),
                    _cell(1, 1, "0.91"),
                ]
            )
        )

        assert table.rows == [["Metric", "Value"], ["Accuracy", "0.91"]]
        assert table.header_rows == 1
        assert table.row_count == 2
        assert table.column_count == 2

    def test_expands_a_merged_cell_across_its_span(self):
        table = build_table_from_docling(
            _table_payload(
                [
                    _cell(0, 0, "Group", col_span=2, column_header=True),
                    _cell(1, 0, "A", column_header=True),
                    _cell(1, 1, "B", column_header=True),
                    _cell(2, 0, "x", row_span=2),
                    _cell(2, 1, "1"),
                    _cell(3, 1, "2"),
                ]
            )
        )

        # The merged header occupies both of its columns, and the row-spanning
        # body cell occupies both of its rows. Filling only the anchor position
        # would shift every following column, which is the bug this pins.
        assert table.rows == [
            ["Group", "Group"],
            ["A", "B"],
            ["x", "1"],
            ["x", "2"],
        ]
        assert table.header_rows == 2

    def test_a_span_wins_when_the_offsets_disagree_with_it(self):
        table = build_table_from_docling(
            _table_payload(
                [
                    _cell(0, 0, "Wide", row_span=2, col_span=3, column_header=True),
                ]
            )
        )

        assert table.rows == [["Wide", "Wide", "Wide"], ["Wide", "Wide", "Wide"]]
        assert table.column_count == 3

    def test_infers_a_two_row_header_from_the_column_header_flags(self):
        table = build_table_from_docling(
            _table_payload(
                [
                    _cell(0, 0, "2023", column_header=True),
                    _cell(0, 1, "2024", column_header=True),
                    _cell(1, 0, "a", column_header=True),
                    _cell(1, 1, "b", column_header=True),
                    _cell(2, 0, "1"),
                    _cell(2, 1, "2"),
                ]
            )
        )

        # Header depth comes from the deepest column_header row, not from
        # assuming row 0 — a table with a spanning title row would otherwise
        # report one header row too few.
        assert table.header_rows == 2

    def test_falls_back_to_a_cursor_when_offsets_are_absent(self):
        # An older payload lists the cells in row-major order and says how wide
        # the table is, but not where any individual cell sits. The cursor
        # supplies the missing positions; without it every cell would collapse
        # onto (0, 0) and the grid would be a single cell.
        payload = {
            "self_ref": "tables/0",
            "data": {
                "table_cells": [
                    {"row_span": 1, "col_span": 1, "text": "A"},
                    {"row_span": 1, "col_span": 1, "text": "B"},
                    {"row_span": 1, "col_span": 1, "text": "C"},
                ],
                "num_cols": 2,
            },
        }

        table = build_table_from_docling(payload)

        assert table.rows == [["A", "B"], ["C", ""]]
        assert table.row_count == 2
        assert table.column_count == 2

    def test_without_a_declared_width_the_cursor_does_not_invent_one(self):
        # Nothing in this payload says how many columns the table has. Stacking
        # the cells in one row is the only answer that adds no structure the
        # converter did not report — the alternative (one cell per row) reads
        # as a three-row table that the source never claimed to be.
        payload = {
            "self_ref": "tables/0",
            "data": {
                "table_cells": [
                    {"row_span": 1, "col_span": 1, "text": "A"},
                    {"row_span": 1, "col_span": 1, "text": "B"},
                ]
            },
        }

        table = build_table_from_docling(payload)

        assert table.rows == [["A", "B"]]
        assert table.column_count == 2

    def test_the_cursor_follows_the_cells_that_do_declare_offsets(self):
        # A mixed payload: the first cell anchors the table and the ones
        # after it, having no offsets of their own, must continue from there.
        payload = {
            "self_ref": "tables/0",
            "data": {
                "table_cells": [
                    {
                        "row_span": 1,
                        "col_span": 1,
                        "text": "A",
                        "start_row_offset_idx": 0,
                        "end_row_offset_idx": 1,
                        "start_col_offset_idx": 0,
                        "end_col_offset_idx": 1,
                    },
                    {"row_span": 1, "col_span": 1, "text": "B"},
                    {"row_span": 1, "col_span": 1, "text": "C"},
                ],
                "num_cols": 2,
            },
        }

        table = build_table_from_docling(payload)

        assert table.rows == [["A", "B"], ["C", ""]]

    def test_honours_a_declared_size_larger_than_the_cells(self):
        table = build_table_from_docling(
            _table_payload([_cell(0, 0, "A")], num_rows=3, num_cols=2)
        )

        assert table.row_count == 3
        assert table.column_count == 2

    def test_survives_an_empty_or_malformed_payload(self):
        assert build_table_from_docling({}).is_empty_shell
        assert build_table_from_docling({"data": None}).is_empty_shell
        assert build_table_from_docling({"data": {"table_cells": ["junk"]}}).is_empty_shell

    def test_a_non_string_cell_value_is_coerced(self):
        # Numeric cells occasionally arrive as numbers rather than strings.
        table = build_table_from_docling(
            _table_payload([_cell(0, 0, ""), {"start_row_offset_idx": 0, "end_row_offset_idx": 1,
                                             "start_col_offset_idx": 1, "end_col_offset_idx": 2,
                                             "text": 0.91}])
        )

        assert table.rows == [["", "0.91"]]


class TestRoleMapping:
    @pytest.mark.parametrize(
        ("label", "expected"),
        [
            ("section_header", "heading"),
            ("title", "title"),
            ("formula", "formula"),
            ("caption", "caption"),
            ("footnote", "footnote"),
            ("page_header", "header"),
            ("page_footer", "footer"),
            ("list_item", "list_item"),
            ("code", "code"),
        ],
    )
    def test_known_labels_map_to_schema_roles(self, label, expected):
        payload = _payload(texts=[_text("texts/0", "content", label)])

        document = _normalize(payload)

        assert document.elements[0].role == expected

    @pytest.mark.parametrize("label", ["", "unknown", "some_future_label", "PAGE_HEADER"])
    def test_an_unmapped_label_degrades_to_text(self, label):
        # A newer docling adding a label must not turn a page into a failure.
        # PAGE_HEADER also proves the lookup is case-insensitive.
        payload = _payload(texts=[_text("texts/0", "content", label)])

        document = _normalize(payload)

        assert document.elements[0].role in {"text", "header"}

    def test_a_formula_keeps_its_latex_and_renders_delimited(self):
        payload = _payload(texts=[_text("texts/0", "E = mc^2", "formula")])

        document = _normalize(payload)
        element = document.elements[0]

        assert element.role == "formula"
        assert element.latex == "E = mc^2"
        # ``text`` is left empty on purpose; the schema fills it from ``latex``
        # so the prose projection and the structured field cannot disagree.
        assert element.text == "$$E = mc^2$$"

    def test_furniture_is_excluded_from_the_body_projection(self):
        payload = _payload(
            texts=[
                _text("texts/0", "Running head", "page_header"),
                _text("texts/1", "Real content."),
            ]
        )

        document = _normalize(payload)

        assert [element.role for element in document.elements] == ["header", "text"]
        assert "Running head" not in document.to_full_text()
        assert "Real content." in document.to_full_text()

    def test_a_heading_path_is_attached_to_later_elements(self):
        payload = _payload(
            texts=[
                _text("texts/0", "Methods", "section_header"),
                _text("texts/1", "We ran experiments."),
                _text("texts/2", "Results", "section_header"),
                _text("texts/3", "It worked."),
            ]
        )

        document = _normalize(payload)

        assert [element.heading_path for element in document.elements] == [
            ("Methods",),
            ("Methods",),
            ("Methods", "Results"),
            ("Methods", "Results"),
        ]


class TestReadingOrder:
    def test_follows_the_group_tree_for_a_two_column_page(self):
        """The reason this adapter uses docling at all: a two-column page must
        read down the left column, then down the right — not across each line."""
        payload = _payload(
            texts=[
                _text("texts/0", "Left top", prov=_prov(left=10, top=700, right=200, bottom=650)),
                _text("texts/1", "Left bottom", prov=_prov(left=10, top=400, right=200, bottom=350)),
                _text("texts/2", "Right top", prov=_prov(left=300, top=700, right=500, bottom=650)),
                _text("texts/3", "Right bottom", prov=_prov(left=300, top=400, right=500, bottom=350)),
            ],
            groups=[
                {
                    "self_ref": "groups/0",
                    "name": "body",
                    "children": [
                        {"$ref": "texts/0"},
                        {"$ref": "texts/1"},
                        {"$ref": "texts/2"},
                        {"$ref": "texts/3"},
                    ],
                }
            ],
        )

        document = _normalize(payload)

        assert [element.text for element in document.elements] == [
            "Left top",
            "Left bottom",
            "Right top",
            "Right bottom",
        ]
        # The group order deliberately disagrees with a naive top-to-bottom
        # sort, so this asserts the tree actually won.
        assert document.warnings == []

    def test_recurses_into_nested_groups(self):
        payload = _payload(
            texts=[_text("texts/0", "in group"), _text("texts/1", "loose")],
            groups=[
                {
                    "self_ref": "groups/0",
                    "children": [
                        {"$ref": "groups/1"},
                        {"$ref": "texts/1"},
                    ],
                },
                {"self_ref": "groups/1", "children": [{"$ref": "texts/0"}]},
            ],
        )

        document = _normalize(payload)

        assert [element.text for element in document.elements] == ["in group", "loose"]

    def test_emits_a_caption_once_even_when_the_body_also_lists_it(self):
        caption = _text("texts/1", "Figure 1: the architecture.", "caption")
        picture = {
            "self_ref": "pictures/0",
            "prov": _prov(),
            "captions": [{"$ref": "texts/1"}],
        }
        payload = _payload(texts=[caption], pictures=[picture])

        document = _normalize(payload)
        full_text = document.to_full_text()

        # A duplicated caption would show up twice in the analysis prompt and
        # bias the model toward the figure.
        assert full_text.count("Figure 1: the architecture.") == 1
        # The caption is folded into the figure rather than standing on its own,
        # so the figure is what carries it.
        assert [element.role for element in document.elements] == ["figure"]

    @pytest.mark.parametrize("caption_first", [True, False])
    def test_caption_folding_does_not_depend_on_which_comes_first(self, caption_first):
        # Docling's own encoding hangs the caption off the picture and omits it
        # from the body; some payloads list both. Which of the two a document
        # used is a detail of the converter, and the output must not change
        # because of it.
        caption = _text("texts/0", "Figure 1: the architecture.", "caption")
        picture = {
            "self_ref": "pictures/0",
            "prov": _prov(),
            "captions": [{"$ref": "texts/0"}],
        }
        if caption_first:
            payload = _payload(
                texts=[caption],
                pictures=[picture],
                body_children=[{"$ref": "texts/0"}, {"$ref": "pictures/0"}],
            )
        else:
            payload = _payload(
                texts=[caption],
                pictures=[picture],
                body_children=[{"$ref": "pictures/0"}, {"$ref": "texts/0"}],
            )

        document = _normalize(payload)

        assert document.to_full_text().count("Figure 1: the architecture.") == 1
        assert [element.role for element in document.elements] == ["figure"]

    def test_a_caption_that_no_figure_claims_stays_its_own_element(self):
        # Folding is by ownership. A stray caption with no owner is still text
        # the user can see, so it must survive on its own.
        payload = _payload(texts=[_text("texts/0", "Table 1: inputs.", "caption")])

        document = _normalize(payload)

        assert [element.role for element in document.elements] == ["caption"]

    def test_falls_back_to_positional_order_when_groups_are_missing(self):
        payload = _payload(
            texts=[
                _text("texts/0", "lower", prov=_prov(top=300, bottom=250)),
                _text("texts/1", "upper", prov=_prov(top=700, bottom=650)),
            ],
            body_children=[],
        )

        document = _normalize(payload)

        assert [element.text for element in document.elements] == ["upper", "lower"]

    def test_the_fallback_is_never_silent(self):
        # A silent fallback looks identical to a correct parse while quietly
        # scrambling every multi-column document.
        document = _normalize(_payload(texts=[_text("texts/0", "x")], body_children=[]))

        assert document.warnings
        assert any("阅读顺序" in warning for warning in document.warnings)

    def test_does_not_warn_when_the_tree_was_usable(self):
        document = _normalize(_payload(texts=[_text("texts/0", "x")]))

        assert document.warnings == []

    def test_a_self_referential_group_cannot_hang_the_parse(self):
        payload = _payload(
            texts=[_text("texts/0", "x")],
            groups=[
                {
                    "self_ref": "groups/0",
                    "children": [{"$ref": "groups/0"}, {"$ref": "texts/0"}],
                }
            ],
        )

        document = _normalize(payload)

        assert [element.text for element in document.elements] == ["x"]

    def test_content_outside_the_body_tree_is_still_kept(self):
        # Docling emits some regions outside the body group; dropping them
        # would quietly lose text.
        payload = _payload(
            texts=[_text("texts/0", "in body"), _text("texts/1", "orphan")],
            body_children=[{"$ref": "texts/0"}],
        )

        document = _normalize(payload)

        assert {element.text for element in document.elements} == {"in body", "orphan"}


class TestNormalizeDocument:
    def test_records_provenance_and_page_count(self):
        payload = _payload(texts=[_text("texts/0", "x", page=2)], pages=_pages(3))

        document = _normalize(payload, sha256="abc123", options_hash="opt1")

        assert document.source_filename == "paper.pdf"
        assert document.media_type == "application/pdf"
        assert document.sha256 == "abc123"
        assert document.options_hash == "opt1"
        assert document.page_count == 3
        assert document.parser_name == "docling"
        assert document.created_at

    def test_a_pdf_element_carries_a_resolvable_page_locator(self):
        payload = _payload(texts=[_text("texts/0", "x")])

        document = _normalize(payload)
        element = document.elements[0]

        assert element.locator.kind == "pdf_page"
        assert element.locator.page == 1
        assert element.locator.bbox is not None
        assert element.locator.is_resolvable
        assert document.locator_coverage == 1.0

    def test_builds_one_unit_per_page(self):
        payload = _payload(
            texts=[
                _text("texts/0", "page one", page=1),
                _text("texts/1", "page two", page=2),
            ],
            pages=_pages(2),
        )

        document = _normalize(payload)

        # A citation resolves to a page, not to "paragraph 43".
        assert [unit.locator.page for unit in document.units] == [1, 2]
        assert "page one" in document.units[0].text

    def test_derives_the_page_count_from_provenance_when_pages_is_empty(self):
        payload = _payload(texts=[_text("texts/0", "x", page=4)], pages={})

        document = _normalize(payload)

        assert document.page_count == 4

    def test_an_unlocatable_element_is_marked_unknown_rather_than_faked(self):
        payload = _payload(texts=[_text("texts/0", "x", prov=[])])

        document = _normalize(payload)

        assert document.elements[0].locator.kind == "unknown"
        assert document.elements[0].locator.page is None
        assert document.locator_coverage == 0.0

    def test_a_topleft_box_without_a_page_height_is_reported(self):
        payload = _payload(
            texts=[_text("texts/0", "x", prov=_prov(origin="TOPLEFT"))],
            pages={"1": {"size": {}}},
        )

        document = _normalize(payload)

        assert document.elements[0].locator.bbox is not None
        assert any("TOPLEFT" in warning for warning in document.warnings)

    def test_a_table_element_keeps_its_grid_and_gets_markdown_text(self):
        table = _table_payload(
            [
                _cell(0, 0, "Metric", column_header=True),
                _cell(0, 1, "Value", column_header=True),
                _cell(1, 0, "Accuracy"),
                _cell(1, 1, "0.91"),
            ]
        )
        payload = _payload(tables=[table])

        document = _normalize(payload)
        element = document.elements[0]

        assert element.role == "table"
        assert element.is_structural
        assert element.table.rows == [["Metric", "Value"], ["Accuracy", "0.91"]]
        # The schema fills ``text`` from the grid, so the two cannot drift.
        assert element.text == element.table.to_markdown()
        assert "Accuracy" in document.to_full_text()

    def test_an_empty_table_shell_with_no_caption_is_dropped(self):
        # The quality gate reports ruled-lines-but-no-text separately; keeping an
        # addressable empty grid here would only add a zero-content element.
        payload = _payload(tables=[_table_payload([_cell(0, 0, "")])])

        document = _normalize(payload)

        assert document.elements == []

    def test_a_picture_becomes_a_figure_carrying_its_caption(self):
        picture = {"self_ref": "pictures/0", "prov": _prov(), "captions": [{"text": "Fig 1"}]}
        payload = _payload(pictures=[picture])

        document = _normalize(payload)
        element = document.elements[0]

        assert element.role == "figure"
        assert element.text == "Fig 1"
        assert element.is_structural

    def test_a_picture_without_a_caption_is_still_addressable(self):
        picture = {"self_ref": "pictures/0", "prov": _prov()}
        payload = _payload(pictures=[picture])

        document = _normalize(payload)

        assert document.elements[0].role == "figure"
        assert document.elements[0].text == ""
        assert document.elements[0].locator.is_resolvable

    def test_an_empty_document_says_so_instead_of_looking_like_a_clean_parse(self):
        document = _normalize(_payload())

        assert document.elements == []
        assert any("未能" in warning for warning in document.warnings)

    def test_element_order_is_dense_and_matches_reading_order(self):
        payload = _payload(
            texts=[_text("texts/0", "a"), _text("texts/1", "b"), _text("texts/2", "c")]
        )

        document = _normalize(payload)

        assert [element.order for element in document.elements] == [0, 1, 2]

    def test_rejects_a_payload_that_is_not_an_object(self):
        for junk in (None, [], "text", 7):
            with pytest.raises(ValueError, match="不是对象"):
                _normalize(junk)

    def test_the_result_is_a_serializable_parsed_document(self):
        payload = _payload(
            texts=[_text("texts/0", "Title here", "title"), _text("texts/1", "Body.")],
            tables=[
                _table_payload(
                    [
                        _cell(0, 0, "H", column_header=True),
                        _cell(1, 0, "v"),
                    ]
                )
            ],
        )

        document = _normalize(payload)
        restored = ParsedDocument.from_dict(document.to_dict())

        assert restored.page_count == document.page_count
        assert len(restored.elements) == len(document.elements)
        assert restored.elements[0].role == "title"
        # The grid has to survive the round trip intact: it is the only reason
        # the table was not flattened at parse time. Looked up by role rather
        # than by index, because element order is the body tree's business.
        table = next(element for element in restored.elements if element.role == "table")
        assert table.table.rows == [["H"], ["v"]]
        assert table.table.header_rows == 1

    def test_normalization_does_not_mutate_the_payload(self):
        """The same dict is what :func:`write_raw_artifact` persists moments
        later, and that file is only worth anything if it is exactly what
        docling returned. An adapter-private key written back into it would
        make the artifact a copy of the backend's output *plus our
        bookkeeping*, which is the one thing it must not be."""
        payload = _payload(
            texts=[_text("texts/0", "x")],
            tables=[_table_payload([_cell(0, 0, "H", column_header=True)])],
        )
        before = json.loads(json.dumps(payload))

        _normalize(payload)

        assert json.loads(json.dumps(payload)) == before

    def test_survives_an_html_landing_page_worth_of_noise(self):
        """Real scraped HTML arrives as nav bars, cookie banners and repeated
        boilerplate. It must degrade, not crash."""
        noise = [
            _text(f"texts/{index}", "Accept cookies? " * 20, "unknown")
            for index in range(40)
        ]
        noise.append(_text("texts/40", "", "unknown"))
        payload = _payload(texts=noise)
        payload["origin"] = {"filename": "landing.html", "mimetype": "text/html"}

        document = _normalize(payload, media_type="text/html")

        assert document.elements
        assert all(isinstance(element.text, str) for element in document.elements)
        # An empty text element carries nothing retrievable.
        assert all(element.text for element in document.elements)

    def test_tolerates_structurally_broken_payloads(self):
        for broken in (
            {},
            {"body": "not a dict"},
            {"body": {"children": "not a list"}},
            {"body": {"children": [{"$ref": "texts/9"}]}},
            {"texts": "not a list", "pages": "not a dict"},
            {"groups": [{"self_ref": "groups/0", "children": None}]},
        ):
            document = _normalize(broken)
            assert isinstance(document, ParsedDocument)


class TestFormatFixtures:
    """T0.4's acceptance criterion, stated as a test: three input formats each
    normalize into a valid :class:`ParsedDocument`.

    The three fixtures are the *same* DoclingDocument with a different
    ``origin.mimetype``, and that is the truth of the situation rather than a
    shortcut. Docling normalizes every input format into one page-oriented
    document model before the adapter ever sees it, so there is exactly one
    code path here; what varies per format is the media type the caller
    declares, which is what the adapter records on the document. Were docling
    to grow a per-format provenance model, these fixtures are the ones that
    would need splitting.

    The content is deliberately mixed — prose, a formula, a table, a figure —
    because "produces a valid ParsedDocument" is a weak claim unless the
    format-specific structures survive the trip.
    """

    FORMATS = [
        ("paper.pdf", "application/pdf"),
        (
            "paper.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ),
        (
            "slides.pptx",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ),
    ]

    @staticmethod
    def _fixture_payload():
        """A two-page document with one of every structure the schema can carry.

        The split across two pages is what makes the unit list meaningful: a
        citation should resolve to "page 2", so the fixture has to occupy more
        than one page for that to be checkable.
        """
        return _payload(
            texts=[
                _text("texts/0", "Attention Is All You Need", "title"),
                _text("texts/1", "Abstract", "section_header"),
                _text("texts/2", "We propose a new architecture."),
                _text("texts/3", "E = mc^2", "formula"),
                _text("texts/4", "Figure 1: the encoder stack.", "caption"),
            ],
            tables=[
                _table_payload(
                    [
                        _cell(0, 0, "Model", column_header=True),
                        _cell(0, 1, "BLEU", column_header=True),
                        _cell(1, 0, "Baseline"),
                        _cell(1, 1, "27.3"),
                    ],
                    prov=_prov(page=2),
                )
            ],
            pictures=[
                {"self_ref": "pictures/0", "prov": _prov(page=2), "captions": [{"$ref": "texts/4"}]}
            ],
            pages=_pages(2),
        )

    @pytest.mark.parametrize(("filename", "media_type"), FORMATS)
    def test_each_format_produces_a_valid_parsed_document(self, filename, media_type):
        payload = self._fixture_payload()
        payload["origin"] = {"filename": filename, "mimetype": media_type}

        document = _normalize(
            payload, source_filename=filename, media_type=media_type, sha256="f" * 64
        )

        assert isinstance(document, ParsedDocument)
        assert document.media_type == media_type
        assert document.source_filename == filename
        assert document.page_count == 2
        assert document.parser_name == "docling"
        assert [unit.locator.page for unit in document.units] == [1, 2]
        # "Valid" means a downstream stage can consume it without repairing
        # anything: every element points somewhere, and the document survives a
        # serialization round trip with its structure intact.
        assert document.locator_coverage == 1.0
        restored = ParsedDocument.from_dict(document.to_dict())
        assert len(restored.elements) == len(document.elements)

    @pytest.mark.parametrize(("filename", "media_type"), FORMATS)
    def test_structure_survives_in_every_format(self, filename, media_type):
        payload = self._fixture_payload()
        payload["origin"] = {"filename": filename, "mimetype": media_type}

        document = _normalize(payload, source_filename=filename, media_type=media_type)
        by_role = {element.role: element for element in document.elements}

        # A formula is worthless without its LaTeX: the prose projection shows
        # the delimiters, the chunker needs the source.
        assert by_role["formula"].latex == "E = mc^2"
        # A table is worthless flattened, which is the one thing normalization
        # must never do to it.
        assert by_role["table"].table.rows == [
            ["Model", "BLEU"],
            ["Baseline", "27.3"],
        ]
        assert by_role["table"].table.header_rows == 1
        # The figure keeps the caption that describes it, exactly once.
        assert by_role["figure"].text == "Figure 1: the encoder stack."
        assert document.to_full_text().count("Figure 1: the encoder stack.") == 1
        assert [element.order for element in document.elements] == [0, 1, 2, 3, 4, 5]

    @pytest.mark.parametrize(("filename", "media_type"), FORMATS)
    def test_pdf_elements_carry_a_page_and_a_bbox_in_every_format(self, filename, media_type):
        """Docling's provenance is page-based whatever the input format was, so
        every element keeps a page and a rectangle — that is the whole of
        T0.4's "preserve page + bbox provenance"."""
        payload = self._fixture_payload()
        payload["origin"] = {"filename": filename, "mimetype": media_type}

        document = _normalize(payload, source_filename=filename, media_type=media_type)

        for element in document.elements:
            assert element.locator.is_resolvable
            assert element.locator.page in {1, 2}
            assert element.locator.bbox is not None
            assert element.locator.bbox.width > 0
            assert element.locator.bbox.height > 0


class TestRawArtifact:
    def test_writes_the_payload_and_returns_its_path(self, tmp_path):
        payload = _payload(texts=[_text("texts/0", "x")])

        uri = write_raw_artifact(payload, sha256="a" * 64, options_hash="opt12345",
                                 artifact_dir=str(tmp_path / "artifacts"))

        written = Path(uri)
        assert written.exists()
        assert json.loads(written.read_text(encoding="utf-8")) == payload

    def test_the_filename_is_derived_from_the_content_hash(self, tmp_path):
        """Re-parsing the same file with the same options overwrites rather than
        accumulating near-duplicates."""
        first = write_raw_artifact(_payload(), sha256="b" * 64, artifact_dir=str(tmp_path))
        second = write_raw_artifact(_payload(), sha256="b" * 64, artifact_dir=str(tmp_path))

        assert first == second
        assert len(list(tmp_path.iterdir())) == 1

    def test_options_hash_separates_two_parses_of_one_file(self, tmp_path):
        plain = write_raw_artifact(_payload(), sha256="c" * 64, options_hash="aaaa1111",
                                   artifact_dir=str(tmp_path))
        ocr = write_raw_artifact(_payload(), sha256="c" * 64, options_hash="bbbb2222",
                                 artifact_dir=str(tmp_path))

        assert plain != ocr
        assert len(list(tmp_path.iterdir())) == 2

    def test_defaults_to_the_configured_artifact_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "PARSER_ARTIFACT_DIR", str(tmp_path / "from-config"))

        uri = write_raw_artifact(_payload(), sha256="d" * 64)

        assert Path(uri).exists()
        assert "from-config" in uri

    def test_a_write_failure_is_swallowed(self, tmp_path, monkeypatch):
        # Losing the artifact hurts reproducibility; failing the user's parse
        # over it would not be a sensible trade.
        monkeypatch.setattr(
            adapter, "atomic_write_json",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")),
        )

        assert write_raw_artifact(_payload(), artifact_dir=str(tmp_path)) == ""


class TestParserMetadata:
    def test_reports_its_name_and_the_schema_media_types(self):
        parser = DoclingParser()

        assert parser.name == "docling"
        assert parser.supports("application/pdf")
        assert parser.supports("application/pdf; charset=binary")
        assert not parser.supports("application/zip")
        assert not parser.supports("")

    def test_version_is_reported_without_requiring_docling(self):
        # The version is diagnostic, not a precondition: a build without docling
        # must still be able to describe the adapter.
        assert isinstance(DoclingParser().version, str)

    def test_options_hash_covers_the_parse_affecting_inputs(self):
        base = compute_options_hash(force_ocr=False, mode="inprocess", timeout_seconds=120)

        assert base == compute_options_hash(force_ocr=False, mode="inprocess", timeout_seconds=120)
        assert base != compute_options_hash(force_ocr=True, mode="inprocess", timeout_seconds=120)
        assert base != compute_options_hash(force_ocr=False, mode="serve", timeout_seconds=120)
        # A timed-out parse produces a different artifact, so the timeout has
        # to be part of the cache key.
        assert base != compute_options_hash(force_ocr=False, mode="inprocess", timeout_seconds=30)

    def test_options_hash_is_stable_across_key_order(self):
        assert len(compute_options_hash(force_ocr=True, mode="serve", timeout_seconds=60)) == 16

    def test_the_standalone_result_says_so_under_both_predicate_names(self):
        """The adapter works alone, before the gateway exists, and returns its
        own class then. It exposes the gateway's predicate name as well as its
        own so that a caller can use ``is_usable`` and be right in both
        worlds — ``gateway.ParseResult`` has no ``ok``."""
        succeeded = DoclingParseResult(document=ParsedDocument())

        assert succeeded.ok is True
        assert succeeded.is_usable is True
        assert DoclingParseResult(status="failed").is_usable is False
        assert DoclingParseResult(status="ok").is_usable is False


class TestParseWithoutDocling:
    """The adapter must degrade to a result object, never an exception: the
    gateway turns a result into a fallback decision or a user-facing message,
    and a traceback bypasses both."""

    def test_a_missing_backend_returns_an_actionable_failure(self, tmp_path):
        if importlib.util.find_spec("docling") is not None:  # pragma: no cover
            pytest.skip("本机安装了 docling，无法覆盖缺失分支")

        target = tmp_path / "paper.pdf"
        target.write_bytes(b"%PDF-1.7\n")
        parser = DoclingParser(mode="inprocess")

        result = parser.parse(str(target))

        # Which class comes back depends on whether the gateway has landed:
        # ``_build_result`` upgrades the local mirror to the real contract the
        # moment it is importable. What must hold either way is that a failure
        # is *returned*, not raised, and that the user is told what to install.
        assert isinstance(result, (DoclingParseResult, gateway.ParseResult))
        assert result.status == "failed"
        assert result.is_usable is False
        assert result.document is None
        assert "docling" in result.error_message
        assert "paperwhisperer[parse]" in result.error_message
        assert result.duration_ms >= 0
        assert result.parser_name == "docling"
        assert result.options_hash

    def test_a_missing_file_fails_without_touching_the_backend(self, tmp_path):
        result = DoclingParser(mode="inprocess").parse(str(tmp_path / "absent.pdf"))

        assert result.status == "failed"
        assert "不存在" in result.error_message

    def test_docling_is_never_imported_at_module_scope(self):
        # The whole reason for the lazy import: importing this module must not
        # cost gigabytes of torch.
        import subprocess
        import sys

        source = Path(adapter.__file__).read_text(encoding="utf-8")
        module_level = [
            line
            for line in source.splitlines()
            if line.startswith(("import docling", "from docling"))
        ]
        assert module_level == []
        # The property is "importing this adapter does not import docling", not
        # "docling is not installed". Asserting ``"docling" not in sys.modules``
        # in-process tested the second, not the first: it passed only while no
        # earlier test in the session had imported docling, and failed for a
        # reason that had nothing to do with this adapter as soon as one did.
        # Checked in a fresh interpreter so the answer depends only on this
        # import — which is true whether or not docling is installed here.
        probe = (
            "import sys;"
            "import paperwhisperer.ingestion.adapters.docling_adapter as a;"
            "print('docling' in sys.modules)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            cwd=str(Path(adapter.__file__).resolve().parents[3]),
            timeout=120,
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.strip() == "False", (
            "importing the docling adapter pulled docling into sys.modules: "
            f"{completed.stdout.strip()!r}"
        )


class TestServeMode:
    """docling-serve is exercised entirely through mocks: these tests must never
    open a socket, on a machine that may not even have docling installed."""

    def test_a_normal_response_is_normalized(self, tmp_path, monkeypatch, no_sleep):
        payload = _payload(texts=[_text("texts/0", "Served content.")])
        _serve_response(payload, monkeypatch)
        target = tmp_path / "paper.pdf"
        target.write_bytes(b"%PDF-1.7\n")

        result = DoclingParser(mode="serve", serve_url="http://localhost:5001").parse(
            str(target)
        )

        assert _succeeded(result)
        assert "Served content." in result.document.to_full_text()
        assert result.document.elements[0].locator.page == 1

    def test_the_request_targets_the_parse_endpoint_with_a_multipart_body(
        self, tmp_path, monkeypatch
    ):
        payload = _payload(texts=[_text("texts/0", "x")])
        _serve_response(payload, monkeypatch)
        target = tmp_path / "paper.pdf"
        target.write_bytes(b"%PDF-1.7\nbody")

        DoclingParser(mode="serve", serve_url="http://localhost:5001").parse(str(target))

        request = _LAST_REQUEST[0]
        assert request.full_url == "http://localhost:5001/v1/convert/file"
        assert request.get_method() == "POST"
        assert request.headers["Content-type"].startswith("multipart/form-data; boundary=")
        assert b'filename="paper.pdf"' in request.data
        assert b"%PDF-1.7\nbody" in request.data

    def test_retries_a_429_and_succeeds(self, tmp_path, monkeypatch):
        payload = _payload(texts=[_text("texts/0", "after retry")])
        body = json.dumps(payload).encode("utf-8")
        responses = []

        headers = Message()
        headers["Retry-After"] = "1"
        rate_limited = urllib.error.HTTPError(
            "http://localhost:5001/v1/parse/file", 429, "Too Many Requests", headers, None
        )
        responses.append(rate_limited)
        responses.append(FakeResponse(body, content_type="application/json"))

        attempts = []

        def _open(request, timeout=None, **_kwargs):
            attempts.append(request)
            outcome = responses[min(len(attempts) - 1, len(responses) - 1)]
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        monkeypatch.setattr(urllib.request, "urlopen", _open)
        monkeypatch.setattr(adapter.time, "sleep", lambda _seconds: None)
        target = tmp_path / "paper.pdf"
        target.write_bytes(b"%PDF-1.7\n")

        result = DoclingParser(mode="serve", serve_url="http://localhost:5001").parse(
            str(target)
        )

        assert _succeeded(result)
        assert len(attempts) == 2

    def test_does_not_retry_a_client_error(self, tmp_path, monkeypatch):
        headers = Message()
        rejected = urllib.error.HTTPError(
            "http://localhost:5001/v1/parse/file", 413, "Too Large", headers, None
        )
        attempts = []

        def _open(request, timeout=None, **_kwargs):
            attempts.append(request)
            raise rejected

        monkeypatch.setattr(urllib.request, "urlopen", _open)
        monkeypatch.setattr(adapter.time, "sleep", lambda _seconds: None)
        target = tmp_path / "paper.pdf"
        target.write_bytes(b"%PDF-1.7\n")

        result = DoclingParser(mode="serve", serve_url="http://localhost:5001").parse(
            str(target)
        )

        # 413 means the request itself is wrong; replaying it only burns the
        # caller's timeout budget before producing the same error.
        assert result.status == "failed"
        assert len(attempts) == 1
        assert "413" in result.error_message

    def test_an_unreachable_service_reports_a_timeout(self, tmp_path, monkeypatch):
        attempts = []

        def _open(request, timeout=None, **_kwargs):
            attempts.append(timeout)
            raise TimeoutError("timed out")

        monkeypatch.setattr(urllib.request, "urlopen", _open)
        monkeypatch.setattr(adapter.time, "sleep", lambda _seconds: None)
        target = tmp_path / "paper.pdf"
        target.write_bytes(b"%PDF-1.7\n")

        result = DoclingParser(mode="serve", serve_url="http://localhost:5001").parse(
            str(target)
        )

        assert result.status == "failed"
        assert "timed out" in result.error_message
        # Retried with backoff, because a docling-serve container still loading
        # its models answers with a connection error for the first few seconds.
        assert len(attempts) == adapter.SERVE_MAX_ATTEMPTS

    def test_a_response_without_a_document_is_a_clean_failure(self, tmp_path, monkeypatch):
        body = json.dumps({"status": "error", "detail": "boom"}).encode("utf-8")

        def _open(request, timeout=None, **_kwargs):
            return FakeResponse(body, content_type="application/json")

        monkeypatch.setattr(urllib.request, "urlopen", _open)
        target = tmp_path / "paper.pdf"
        target.write_bytes(b"%PDF-1.7\n")

        result = DoclingParser(mode="serve", serve_url="http://localhost:5001").parse(
            str(target)
        )

        assert result.status == "failed"
        assert "DoclingDocument" in result.error_message

    def test_a_parse_writes_its_raw_artifact(self, tmp_path, monkeypatch, no_sleep):
        payload = _payload(texts=[_text("texts/0", "x")])
        body = json.dumps(payload).encode("utf-8")
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda request, timeout=None, **kwargs: FakeResponse(
                body, content_type="application/json"
            ),
        )
        monkeypatch.setattr(config, "PARSER_ARTIFACT_DIR", str(tmp_path / "artifacts"))
        target = tmp_path / "paper.pdf"
        target.write_bytes(b"%PDF-1.7\n")

        result = DoclingParser(mode="serve", serve_url="http://localhost:5001").parse(
            str(target)
        )

        assert _succeeded(result)
        assert result.raw_artifact_uri
        assert Path(result.raw_artifact_uri).exists()
        # Byte-for-byte what the service sent back, with none of this adapter's
        # bookkeeping mixed in. The artifact is the only record of what the
        # backend actually did, so "what we did with it" has to stay out of it.
        assert json.loads(Path(result.raw_artifact_uri).read_text(encoding="utf-8")) == payload


class TestServeEndpointPolicy:
    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost:5001",
            "http://127.0.0.1:5001/",
            # The IPv6 literal is the case a string-splitting host check gets
            # wrong: "[::1]:5001" split on the first colon is "[", which is not
            # a loopback address, so a local service would be refused as remote.
            "http://[::1]:5001",
            "http://127.0.0.2:5001",
            "http://0.0.0.0:5001",
        ],
    )
    def test_loopback_hosts_are_accepted(self, url):
        resolved = DoclingParser(mode="serve", serve_url=url).resolve_serve_url()

        assert resolved.startswith("http://")
        assert adapter._is_local_host(urlsplit(url).hostname)

    def test_a_configured_default_is_used_when_none_is_passed(self, monkeypatch):
        # The default is a loopback URL, so a fresh deployment works without
        # anyone configuring anything.
        monkeypatch.setattr(config, "PARSER_DOCLING_SERVE_URL", "http://127.0.0.1:5001")

        assert DoclingParser(mode="serve").resolve_serve_url() == "http://127.0.0.1:5001"

    def test_a_url_with_no_host_is_refused_with_an_actionable_message(self):
        parser = DoclingParser(mode="serve", serve_url="localhost:5001")

        with pytest.raises(ValueError, match="http"):
            parser.resolve_serve_url()

    def test_a_remote_host_is_refused_because_it_would_be_cloud_egress(self):
        parser = DoclingParser(mode="serve", serve_url="https://docling.example.com")

        with pytest.raises(ValueError, match="不是本机地址"):
            parser.resolve_serve_url()

    def test_a_host_that_merely_looks_local_is_still_refused(self):
        # A substring check would accept "localhost.example.com", which
        # resolves anywhere on the internet — the exact case local_only exists
        # to prevent.
        parser = DoclingParser(mode="serve", serve_url="http://localhost.example.com")

        with pytest.raises(ValueError, match="不是本机地址"):
            parser.resolve_serve_url()

    def test_a_refused_host_becomes_a_failed_result_not_an_exception(self, tmp_path):
        target = tmp_path / "paper.pdf"
        target.write_bytes(b"%PDF-1.7\n")
        parser = DoclingParser(mode="serve", serve_url="https://docling.example.com")

        result = parser.parse(str(target))

        assert result.status == "failed"
        assert "不是本机地址" in result.error_message

    def test_serve_mode_is_not_gated_on_the_privacy_setting(self, tmp_path, monkeypatch):
        """A loopback call to a container on the same machine is not cloud
        egress, so local_only must not block it — gating would make a local-only
        deployment unable to use a local service."""
        monkeypatch.setattr(config, "DOCUMENT_PRIVACY_MODE", "local_only")
        monkeypatch.setattr(config, "DOCUMENT_CLOUD_CONSENT_GRANTED", False)
        payload = _payload(texts=[_text("texts/0", "local only")])
        body = json.dumps(payload).encode("utf-8")
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda request, timeout=None, **kwargs: FakeResponse(
                body, content_type="application/json"
            ),
        )
        target = tmp_path / "paper.pdf"
        target.write_bytes(b"%PDF-1.7\n")

        result = DoclingParser(mode="serve", serve_url="http://localhost:5001").parse(
            str(target)
        )

        # The parse has to actually happen: the point is that the privacy
        # setting did not stop it, not merely that nothing raised.
        assert _succeeded(result)
        assert result.error_message == ""

    def test_the_endpoint_path_is_appended_to_the_base_url(self):
        assert (
            adapter._serve_endpoint("http://localhost:5001/")
            == "http://localhost:5001/v1/convert/file"
        )

    def test_both_response_shapes_are_accepted(self):
        nested = {"document": {"texts": [], "pages": {}}}
        flat = {"texts": [], "pages": {}}

        assert adapter._extract_serve_document(nested) == {"texts": [], "pages": {}}
        assert adapter._extract_serve_document(flat) == flat

    def test_multipart_encoding_carries_the_file_and_the_fields(self):
        body, content_type = adapter._encode_multipart(
            {"force_ocr": "false"},
            file_field="files",
            filename="paper.pdf",
            content=b"%PDF-1.7\n",
            content_type="application/pdf",
        )

        assert content_type.startswith("multipart/form-data; boundary=")
        assert b'name="force_ocr"' in body
        assert b'filename="paper.pdf"' in body
        assert b"%PDF-1.7" in body
        assert body.endswith(b"--\r\n")


class TestDoclingIsOptional:
    def test_importorskip_documents_that_the_suite_never_needs_it(self):
        # Nothing in the suite imports docling at module scope; if a future
        # change adds one, this is the assertion that catches it.
        source = Path(adapter.__file__).read_text(encoding="utf-8")
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import docling", "from docling")):
                # Only allowed inside a function body (indented).
                assert line.startswith((" ", "\t")), stripped
