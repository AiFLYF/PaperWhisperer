"""Canonical schema: locator fidelity, projection and version gating.

The schema is the contract every parser adapter writes against, so the tests
here are mostly round-trip assertions. A locator that loses its page number
in serialization produces a citation that silently points at the wrong place
— the exact failure this project is being rebuilt to eliminate — and such a
bug is invisible in normal use. Hence the emphasis on exact inverses rather
than on field presence.
"""

from __future__ import annotations

import pytest

from paperwhisperer.ingestion.schema import (
    SCHEMA_VERSION,
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


def paragraph(text: str, page: int | None = 1, order: int = 0) -> Element:
    locator = SourceLocator(kind="pdf_page", page=page)
    return Element(role="paragraph", text=text, locator=locator, order=order)


class TestSourceLocatorRoundTrip:
    @pytest.mark.parametrize(
        "locator",
        [
            SourceLocator(kind="pdf_page", page=7, bbox=BoundingBox(10, 20, 110, 60)),
            SourceLocator(kind="slide", slide=3, element=2),
            SourceLocator(kind="sheet_range", sheet="Sheet1", cell_range="A1:D9"),
            SourceLocator(kind="docx_element", paragraph=43, heading_path=("Intro", "Method")),
            SourceLocator(kind="txt_offset", offset_start=100, offset_end=400),
            SourceLocator(),
        ],
    )
    def test_serialization_is_lossless(self, locator):
        assert SourceLocator.from_dict(locator.to_dict()) == locator

    def test_only_the_fields_meaningful_for_the_kind_are_serialized(self):
        payload = SourceLocator(kind="slide", slide=1).to_dict()

        assert payload == {"kind": "slide", "slide": 1}
        assert "page" not in payload
        assert "bbox" not in payload

    def test_an_unrecognised_kind_degrades_to_unknown(self):
        assert SourceLocator(kind="hologram").kind == "unknown"

    def test_malformed_numbers_do_not_break_deserialization(self):
        locator = SourceLocator.from_dict({"kind": "pdf_page", "page": "not-a-number"})

        assert locator.page is None
        assert locator.is_resolvable is False

    def test_a_pdf_page_without_a_bbox_still_resolves(self):
        assert SourceLocator(kind="pdf_page", page=1).is_resolvable is True

    def test_a_page_without_a_number_does_not_resolve(self):
        assert SourceLocator(kind="pdf_page").is_resolvable is False

    def test_bbox_renders_as_a_pdf_js_fragment(self):
        assert BoundingBox(10, 20, 110, 60).to_query() == "xywh=10,20,100,40"

    def test_a_fractional_coordinate_keeps_its_decimal_places(self):
        assert BoundingBox(10.5, 20, 110, 60).to_query() == "xywh=10.5,20,99.5,40"

    def test_the_same_rectangle_always_renders_identically(self):
        # Page fragments become cache keys, so integer and float spellings of
        # the same coordinate must not produce two different keys.
        assert BoundingBox(10, 20, 110, 60).to_query() == BoundingBox(10.0, 20.0, 110.0, 60.0).to_query()

    def test_bbox_measurements_are_absolute(self):
        box = BoundingBox(110, 60, 10, 20)

        assert (box.width, box.height) == (100.0, 40.0)

    def test_heading_path_is_normalized_to_a_tuple(self):
        locator = SourceLocator(kind="docx_element", paragraph=1, heading_path=["A", "B"])

        assert locator.heading_path == ("A", "B")


class TestReadingOrder:
    def test_each_format_sorts_within_its_own_scheme(self):
        pdf = SourceLocator(kind="pdf_page", page=2)
        slide = SourceLocator(kind="slide", slide=99)
        unknown = SourceLocator()

        assert pdf.sort_key < slide.sort_key < unknown.sort_key

    def test_pages_sort_numerically_rather_than_lexically(self):
        keys = [SourceLocator(kind="pdf_page", page=page).sort_key for page in (2, 10)]

        assert keys[0] < keys[1]

    def test_plain_text_falls_back_to_its_character_offset(self):
        early = SourceLocator(kind="txt_offset", offset_start=0)
        late = SourceLocator(kind="txt_offset", offset_start=900)

        assert early.sort_key < late.sort_key


class TestTable:
    def test_a_grid_of_blanks_is_flagged_as_an_empty_shell(self):
        assert Table(rows=[["", "  "], ["", ""]]).is_empty_shell is True

    def test_a_grid_with_content_is_not_a_shell(self):
        assert Table(rows=[["Metric", ""], ["", "0.91"]]).is_empty_shell is False

    def test_a_table_with_no_rows_is_a_shell(self):
        assert Table().is_empty_shell is True

    def test_markdown_carries_the_header_separator(self):
        markdown = Table(rows=[["Metric", "Value"], ["Acc", "0.91"]], header_rows=1).to_markdown()

        assert markdown.splitlines() == ["| Metric | Value |", "| --- | --- |", "| Acc | 0.91 |"]

    def test_pipes_inside_cells_do_not_break_the_markdown_grid(self):
        markdown = Table(rows=[["a|b", "c"]]).to_markdown()

        assert markdown.splitlines()[0] == "| a\\|b | c |"

    def test_ragged_rows_are_padded_so_columns_stay_aligned(self):
        table = Table(rows=[["a", "b", "c"], ["d"]])

        assert table.column_count == 3
        assert table.to_markdown().splitlines()[2] == "| d |  |  |"

    def test_html_escapes_cell_content(self):
        html = Table(rows=[["<script>"]], header_rows=1).to_html()

        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_html_wraps_header_rows_in_a_thead(self):
        html = Table(rows=[["h"], ["b"]], header_rows=1).to_html()

        assert "<thead><tr><th>h</th></tr></thead>" in html
        assert "<tbody><tr><td>b</td></tr></tbody>" in html

    def test_with_no_header_rows_the_first_row_still_becomes_a_header(self):
        html = Table(rows=[["a"], ["b"]]).to_html()

        assert "<th>a</th>" in html
        assert "<td>b</td>" in html

    def test_round_trip_preserves_the_grid(self):
        table = Table(rows=[["a", "b"], ["c", "d"]], header_rows=1, caption="T")

        assert Table.from_dict(table.to_dict()) == table


class TestElement:
    def test_an_unknown_role_degrades_to_plain_text(self):
        assert Element(role="interpretive_dance").role == "text"

    def test_a_table_element_derives_its_text_from_the_grid(self):
        element = Element(role="table", table=Table(rows=[["Acc", "0.91"]]))

        assert "0.91" in element.text

    def test_supplied_text_wins_over_the_derived_projection(self):
        table = Table(rows=[["0.91"]])
        element = Element(role="table", text="Accuracy on the test set", table=table)

        assert element.text == "Accuracy on the test set"

    def test_a_formula_element_renders_its_latex(self):
        assert Element(role="formula", latex="E=mc^2").text == "$$E=mc^2$$"

    def test_tables_formulas_and_figures_count_as_structural(self):
        roles = {element.role for element in
                 (Element(role="table"), Element(role="formula"), Element(role="figure"))}

        assert all(Element(role=role).is_structural for role in roles)

    def test_headers_and_footers_count_as_furniture(self):
        assert Element(role="header").is_furniture is True
        assert Element(role="footer").is_furniture is True
        assert Element(role="paragraph").is_furniture is False

    def test_round_trip_preserves_the_locator_and_grid(self):
        element = Element(
            role="table",
            text="Acc | 0.91",
            locator=SourceLocator(kind="pdf_page", page=4, bbox=BoundingBox(1, 2, 3, 4)),
            table=Table(rows=[["Acc", "0.91"]], header_rows=1),
            heading_path=("Results",),
        )

        assert Element.from_dict(element.to_dict()) == element


class TestParsedDocument:
    def test_the_adapter_supplied_order_outranks_the_page_number(self):
        # Reading order is the adapter's call, not the schema's: only a parser
        # that saw the page geometry knows that column two of a two-column PDF
        # continues the paragraph at the top of column one. Sorting by page
        # alone would interleave the columns, so ``order`` wins and the locator
        # is only a tiebreak.
        document = ParsedDocument(elements=[paragraph("column two top", page=1, order=0),
                                            paragraph("column one top", page=1, order=1)])

        assert [element.text for element in document.elements] == ["column two top",
                                                                   "column one top"]

    def test_elements_sharing_an_order_fall_back_to_the_locator(self):
        document = ParsedDocument(elements=[paragraph("page two", page=2, order=0),
                                            paragraph("page one", page=1, order=0)])

        assert [element.text for element in document.elements] == ["page one", "page two"]

    def test_order_is_renumbered_to_match_the_sorted_sequence(self):
        document = ParsedDocument(elements=[paragraph("b", page=2, order=0),
                                            paragraph("a", page=1, order=1)])

        assert [element.order for element in document.elements] == [0, 1]

    def test_running_furniture_is_excluded_from_the_body(self):
        document = ParsedDocument(elements=[
            paragraph("real content", page=1, order=0),
            Element(role="header", text="Journal of Things", order=1),
            Element(role="footer", text="12", order=2),
        ])

        assert [element.text for element in document.body_elements] == ["real content"]

    def test_locator_coverage_is_the_share_of_body_elements_that_resolve(self):
        document = ParsedDocument(elements=[
            paragraph("located", page=1, order=0),
            Element(role="paragraph", text="lost", locator=SourceLocator(), order=1),
        ])

        assert document.locator_coverage == 0.5

    def test_an_empty_document_reports_zero_coverage_rather_than_dividing_by_zero(self):
        assert ParsedDocument().locator_coverage == 0.0

    def test_titles_become_a_top_level_heading_in_the_projection(self):
        document = ParsedDocument(elements=[
            Element(role="title", text="Attention Is All You Need", order=0),
            paragraph("Body text.", order=1),
        ])

        assert document.to_full_text().splitlines()[0] == "# Attention Is All You Need"

    def test_a_section_heading_deepens_with_its_level(self):
        document = ParsedDocument(elements=[Element(role="heading", text="Method", level=2)])

        assert document.to_full_text() == "## Method"

    def test_a_table_appears_in_the_projection_as_a_grid_not_flattened_prose(self):
        document = ParsedDocument(elements=[
            Element(role="table", table=Table(rows=[["Metric", "Value"], ["Acc", "0.91"]],
                                              header_rows=1)),
        ])

        assert "| Acc | 0.91 |" in document.to_full_text()

    def test_blank_elements_are_dropped_from_the_projection(self):
        document = ParsedDocument(elements=[paragraph("   ", page=1)])

        assert document.to_full_text() == ""

    def test_build_units_groups_body_elements_by_page(self):
        document = ParsedDocument(elements=[
            paragraph("page one", page=1, order=0),
            paragraph("page two", page=2, order=1),
            paragraph("page two again", page=2, order=2),
        ])

        units = document.build_units()

        assert [unit.locator.page for unit in units] == [1, 2]
        assert "page two again" in units[1].text

    def test_build_units_skips_elements_with_no_resolvable_locator(self):
        document = ParsedDocument(elements=[Element(role="paragraph", text="orphan",
                                                    locator=SourceLocator())])

        assert document.build_units() == []

    def test_build_units_ignores_furniture(self):
        document = ParsedDocument(elements=[Element(role="footer", text="12",
                                                    locator=SourceLocator(kind="pdf_page", page=9))])

        assert document.build_units() == []

    def test_a_unit_is_labelled_with_a_human_readable_page_number(self):
        document = ParsedDocument(elements=[paragraph("body", page=7)])

        assert document.build_units()[0].label == "p. 7"

    def test_round_trip_is_exact(self):
        document = ParsedDocument(
            source_filename="paper.pdf",
            media_type="application/pdf",
            sha256="abc123",
            page_count=2,
            elements=[paragraph("first", page=1, order=0),
                      Element(role="table", table=Table(rows=[["a", "b"]]), order=1)],
            parser_name="docling",
            parser_version="2.0",
            options_hash="h1",
        )

        assert ParsedDocument.from_dict(document.to_dict()).to_dict() == document.to_dict()

    def test_a_future_major_version_is_refused_rather_than_misread(self):
        with pytest.raises(ValueError, match="schema version"):
            ParsedDocument.from_dict({"schema_version": "9.0", "elements": []})

    def test_a_minor_version_bump_still_reads(self):
        assert is_supported_schema_version("1.7") is True
        assert is_supported_schema_version("2.0") is False

    def test_the_serialized_form_records_the_schema_version(self):
        assert ParsedDocument().to_dict()["schema_version"] == SCHEMA_VERSION

    def test_units_round_trip_too(self):
        document = ParsedDocument(elements=[paragraph("body", page=1)])
        document.build_units()

        assert ParsedDocument.from_dict(document.to_dict()).units == document.units


class TestFormula:
    """LaTeX plus the one distinction a bare string cannot hold.

    The plan names ``Formula`` as a canonical type, and the reason it is more
    than a wrapper around ``Element.latex`` is that MinerU's ``content_list``
    distinguishes ``equation`` from ``equation_inline`` and normalization
    collapses both onto one role. That collapse is invisible in the output and
    still present in the document, because an inline formula belongs inside the
    sentence around it while a display formula is a block of its own.
    """

    def test_a_display_formula_renders_as_a_block(self):
        assert Formula(latex="E = mc^2").rendered == "$$E = mc^2$$"

    def test_an_inline_formula_renders_with_single_delimiters(self):
        # A $$…$$ block dropped into the middle of a sentence is what a reader
        # sees as a rendering bug, and no delimiters makes a formula
        # indistinguishable from the prose it is embedded in.
        assert Formula(latex="x_i", display=False).rendered == "$x_i$"

    def test_blank_latex_is_refused(self):
        # A formula that survived parsing as an empty string is not a formula,
        # and the quality gate counts role-formula elements against their LaTeX
        # count — so an empty one has to be impossible to construct.
        with pytest.raises(ValueError, match="non-empty LaTeX"):
            Formula(latex="   ")

    def test_surrounding_whitespace_is_stripped(self):
        assert Formula(latex="  E = mc^2  ").latex == "E = mc^2"

    def test_round_trip_is_exact(self):
        for display in (True, False):
            formula = Formula(latex="\\int_0^1 x dx", display=display)

            assert Formula.from_dict(formula.to_dict()) == formula

    def test_a_formula_serialized_without_the_display_flag_defaults_to_block(self):
        # The flag is pruned when it is the default, so the payload stays small
        # and an older reader is unaffected by the new key.
        assert Formula(latex="x").to_dict() == {"latex": "x"}
        assert Formula.from_dict({"latex": "x"}).display is True

    def test_an_inline_formula_writes_the_flag_explicitly(self):
        # The one case that is not the default has to say so, or a reader would
        # silently read an inline formula as a display one.
        assert Formula(latex="x_i", display=False).to_dict() == {
            "latex": "x_i", "display": False}

    def test_an_element_built_from_latex_gains_a_formula(self):
        # The adapter-facing direction: a docling texts[i] entry becomes
        # Element(role="formula", latex=...), and the structured type is
        # derived so the two can never disagree.
        element = Element(role="formula", latex="E = mc^2")

        assert element.formula == Formula(latex="E = mc^2")
        assert element.text == "$$E = mc^2$$"

    def test_a_structured_formula_projects_onto_the_flat_field(self):
        # The other direction, for the consumers that read ``latex``: chunking,
        # the eval runner, the prose projection.
        element = Element(role="formula", formula=Formula(latex="x_i", display=False))

        assert element.latex == "x_i"
        assert element.text == "$x_i$"

    def test_supplied_text_still_wins_over_the_formula_projection(self):
        element = Element(role="formula", text="Equation 3", latex="E = mc^2")

        assert element.text == "Equation 3"

    def test_a_formula_element_round_trips_with_its_latex(self):
        element = Element(role="formula", latex="E = mc^2",
                          locator=SourceLocator(kind="pdf_page", page=3))

        assert Element.from_dict(element.to_dict()) == element

    def test_a_blank_latex_element_is_never_produced(self):
        # Whitespace-only LaTeX is not a formula, so it does not become one
        # either. This is the difference from the dataclass contract, which
        # raises.
        element = Element(role="formula", latex="   ")

        assert element.formula is None
        assert element.latex == "   "


class TestFigure:
    def test_a_caption_typed_by_an_adapter_becomes_a_figure(self):
        # The adapter-facing direction: docling folds a picture's caption into
        # the figure rather than emitting it separately, so the caption reaches
        # us as ``text`` under role ``figure``.
        element = Element(role="figure", text="Figure 1: the encoder stack.")

        assert element.figure == Figure(caption="Figure 1: the encoder stack.")

    def test_a_figure_with_no_caption_is_still_a_figure(self):
        # An addressable region on the page is worth more to a citation than
        # nothing, so the adapters emit it and the schema must keep it. The
        # element is findable even though it has nothing to retrieve.
        element = Element(role="figure", text="")

        assert element.role == "figure"
        assert element.text == ""

    def test_a_captionless_figure_gets_no_empty_payload(self):
        # A ``Figure`` with neither a caption nor an image reference says
        # nothing that ``role="figure"`` does not already say. Deriving one
        # would put an empty ``"figure": {}`` into the payload of every
        # captionless figure, which is the common case for a scanned page.
        element = Element(role="figure", text="")

        assert element.figure is None
        assert "figure" not in element.to_dict()

    def test_an_image_reference_is_carried_without_touching_the_caption(self):
        element = Element(role="figure", text="Figure 1",
                          figure=Figure(image_uri="artifact://fig1.png"))

        assert element.figure.image_uri == "artifact://fig1.png"
        # The caption still comes from ``text``, because that is the field the
        # prose projection and the search path read.
        assert element.figure.caption == "Figure 1"

    def test_a_figure_is_not_built_for_a_non_figure_role(self):
        # Otherwise every paragraph with text would acquire a figure payload
        # and the serialized document would roughly double in size.
        element = Element(role="paragraph", text="plain prose")

        assert element.figure is None

    def test_round_trip_is_exact(self):
        for figure in (Figure(), Figure(caption="Figure 1"),
                       Figure(caption="Figure 1", image_uri="artifact://fig1.png")):
            assert Figure.from_dict(figure.to_dict()) == figure

    def test_a_figure_element_round_trips_with_its_caption(self):
        element = Element(role="figure", text="Figure 1: the encoder stack.",
                          locator=SourceLocator(kind="pdf_page", page=4))

        assert Element.from_dict(element.to_dict()) == element

    def test_a_figure_with_an_image_reference_round_trips(self):
        element = Element(role="figure", text="Figure 1",
                          figure=Figure(caption="Figure 1", image_uri="artifact://f.png"),
                          locator=SourceLocator(kind="pdf_page", page=4))

        assert Element.from_dict(element.to_dict()) == element


class TestSourceUnit:
    def test_round_trip_is_exact(self):
        unit = SourceUnit(locator=SourceLocator(kind="pdf_page", page=3), label="p. 3",
                          text="content", order=2)

        assert SourceUnit.from_dict(unit.to_dict()) == unit


class TestChunk:
    def test_list_inputs_are_normalized_to_tuples(self):
        chunk = Chunk(element_ids=["a", "b"], heading_path=["Intro"])

        assert chunk.element_ids == ("a", "b")
        assert chunk.heading_path == ("Intro",)

    def test_round_trip_is_exact(self):
        chunk = Chunk(text="body", element_ids=("e1",),
                      locator=SourceLocator(kind="pdf_page", page=2), token_estimate=42)

        assert Chunk.from_dict(chunk.to_dict()) == chunk


class TestEvidence:
    @pytest.mark.parametrize(
        ("locator", "label"),
        [
            (SourceLocator(kind="pdf_page", page=7), "p. 7"),
            (SourceLocator(kind="slide", slide=3), "slide 3"),
            (SourceLocator(kind="sheet_range", sheet="Sheet1"), "sheet Sheet1"),
            (SourceLocator(kind="docx_element", paragraph=12), "paragraph 12"),
            (SourceLocator(kind="txt_offset", offset_start=0), "document"),
        ],
    )
    def test_each_format_produces_its_own_citation_label(self, locator, label):
        assert Evidence(locator=locator).citation_label == label

    def test_a_heading_path_is_used_when_there_is_no_positional_locator(self):
        evidence = Evidence(locator=SourceLocator(), heading_path=("Method", "Details"))

        assert evidence.citation_label == "Method › Details"

    def test_a_direct_quote_flag_defaults_to_false(self):
        # The flag is written by the verifier, never taken from the model's
        # own claim to be quoting, so an unverified evidence is never a quote.
        assert Evidence(text="...").is_direct_quote is False

    def test_the_derived_label_survives_serialization(self):
        payload = Evidence(locator=SourceLocator(kind="pdf_page", page=2)).to_dict()

        assert payload["citation_label"] == "p. 2"
        assert Evidence.from_dict(payload).citation_label == "p. 2"


class TestPaperFacts:
    def test_a_fact_with_a_value_but_no_citation_is_not_grounded(self):
        assert Fact(field_name="c", value="we invented this").is_grounded is False

    def test_a_fact_with_both_is_grounded(self):
        assert Fact(field_name="c", value="claimed", evidence_ids=("e1",)).is_grounded is True

    def test_stating_a_value_implies_the_extracted_status(self):
        assert Fact(field_name="c", value="claimed").extraction_status == "extracted"

    def test_an_unrecognised_status_degrades_to_not_found(self):
        assert Fact(field_name="c", extraction_status="vibes").extraction_status == "not_found"

    def test_coverage_counts_every_field_including_singletons(self):
        facts = PaperFacts(
            research_question=Fact(field_name="research_question", value="Q",
                                   evidence_ids=("e1",)),
            contributions=[Fact(field_name="c1", value="C1", evidence_ids=("e2",))],
        )

        # research_question + method + reproducibility + the one contribution.
        assert facts.coverage == 0.5

    def test_no_facts_means_no_coverage_rather_than_an_error(self):
        assert PaperFacts().coverage == 0.0

    def test_round_trip_is_exact(self):
        facts = PaperFacts(
            research_question=Fact(field_name="research_question", value="Q",
                                   evidence_ids=("e1",)),
            contributions=[Fact(field_name="c1", value="C1", evidence_ids=("e2",),
                                extraction_status="conflicting",
                                conflicting_values=("something else",))],
            main_results=[Fact(field_name="r1", value="99%", evidence_ids=("e3",))],
            extraction_model="gpt-4o-mini",
        )

        assert PaperFacts.from_dict(facts.to_dict()).to_dict() == facts.to_dict()

    def test_absent_fact_blocks_still_deserialize_into_not_found(self):
        facts = PaperFacts.from_dict({"contributions": [{"field_name": "c1", "value": "C"}]})

        assert facts.method.extraction_status == "not_found"
        assert facts.contributions[0].value == "C"
