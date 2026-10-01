"""Boundary contracts: what gets refused, and what does not.

The dataclasses in ``schema`` degrade rather than raise, and that is correct for
them — an adapter that emits a role nobody defined should not lose the
paragraph. At a boundary the same coercion is the bug: a locator whose page
number parsed to ``None`` becomes a citation that looks fine and points nowhere,
and no stage downstream ever notices.

So these tests are mostly negative. Each one names a shape the dataclass accepts
and the contract refuses, and the interesting part is the *comment* on why the
refusal is right — an unexplained strictness test is indistinguishable from a
brittle one, and the next person to loosen it needs to know what it was
protecting.

The other half is round-trip losslessness. A contract that validates and then
loses a field on the way back to a dataclass is worse than no contract, because
the field it dropped was fine and now it cannot be stored at all.
"""

from __future__ import annotations

import math
import sys
import time

import pytest
from pydantic import ValidationError

from paperwhisperer.ingestion.contracts import (
    BoundingBoxContract,
    ChunkContract,
    ElementContract,
    EvidenceContract,
    FactContract,
    FigureContract,
    FormulaContract,
    PaperFactsContract,
    ParsedDocumentContract,
    SourceLocatorContract,
    SourceUnitContract,
    TableContract,
    UploadRequestContract,
)
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
)


def paragraph(text: str = "body", page: int | None = 1, order: int = 0) -> Element:
    return Element(
        role="paragraph",
        text=text,
        locator=SourceLocator(kind="pdf_page", page=page),
        order=order,
    )


def located(**kwargs) -> SourceLocatorContract:
    return SourceLocatorContract(**kwargs)


# --------------------------------------------------------------------------
# The property the whole split rests on
# --------------------------------------------------------------------------


class TestInternalPathCarriesNoValidation:
    """The dataclasses must stay constructible without Pydantic in play.

    The owner-level decision is that dataclasses stay for the hot internal path
    and Pydantic guards the edges. That decision is only worth anything if the
    internal path is genuinely free of it, so it is asserted rather than
    assumed: if a future import ever pulls ``contracts`` into the parse path,
    every element in every document starts paying for validation nobody asked
    for, and the only symptom is latency.
    """

    def test_importing_the_schema_alone_does_not_import_pydantic(self):
        # Checked in a subprocess because by the time this test runs the test
        # module itself has already imported both. Asserting on ``sys.modules``
        # in-process would be asserting on the test file's imports, which is
        # the opposite of the claim being made.
        import subprocess

        script = (
            "import sys; import paperwhisperer.ingestion.schema as s; "
            "elements = [s.Element(role='paragraph', text=str(i)) for i in range(5000)]; "
            "assert 'pydantic' not in sys.modules; "
            "print(len(elements))"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
        )

        assert result.stdout.strip() == "5000"

    def test_building_elements_does_not_invoke_a_contract_validator(self, monkeypatch):
        from paperwhisperer.ingestion import contracts

        calls: list[object] = []
        original = contracts.SourceLocatorContract.model_validate

        def spy(*args, **kwargs):
            calls.append(args)
            return original(*args, **kwargs)

        monkeypatch.setattr(contracts.SourceLocatorContract, "model_validate", spy)

        document = ParsedDocument(elements=[paragraph(f"p{i}", page=1, order=i)
                                            for i in range(2000)])

        assert len(document.elements) == 2000
        assert calls == []

    def test_a_hot_loop_over_elements_is_not_pydantic_bound(self):
        # A wall-clock bound would be flaky in CI, so this asserts the
        # structural property instead: the per-element cost of the dataclass
        # path is flat with respect to whether contracts are imported, because
        # nothing in it consults them. The 20k element loop is here to make the
        # "thousands of elements" claim concrete rather than theoretical.
        start = time.perf_counter()
        document = ParsedDocument(
            elements=[paragraph(f"paragraph {i}", order=i) for i in range(20_000)]
        )
        elapsed = time.perf_counter() - start

        assert len(document.elements) == 20_000
        # Generous, and deliberately so: the assertion is that a quarter of a
        # second is not spent on per-element validation, not that the machine
        # is fast.
        assert elapsed < 1.0

    def test_a_dataclass_element_carries_no_pydantic_state(self):
        element = paragraph("plain prose")

        assert not hasattr(element, "model_dump")
        assert not hasattr(element, "__pydantic_fields_set__")
        assert type(element).__mro__[1].__name__ == "object"


# --------------------------------------------------------------------------
# Bounding boxes
# --------------------------------------------------------------------------


class TestBoundingBoxContract:
    def test_a_normal_rectangle_is_accepted(self):
        box = BoundingBoxContract(x0=10, y0=20, x1=110, y1=60)

        assert box.to_dataclass() == BoundingBox(10.0, 20.0, 110.0, 60.0)

    def test_a_mirrored_rectangle_is_refused(self):
        # The dataclass accepts this on purpose: width is ``abs``, so a
        # mirrored box still yields a positive width. At a boundary that is
        # exactly the problem — it survives every local check and is then used
        # to draw a highlight in the wrong place.
        with pytest.raises(ValidationError, match="not ordered"):
            BoundingBoxContract(x0=110, y0=60, x1=10, y1=20)

    def test_a_degenerate_rectangle_is_refused(self):
        # Zero width and zero height is not a highlight, it is a stray mark. It
        # is the signature of a converter that emitted a default box because it
        # found no geometry.
        with pytest.raises(ValidationError, match="degenerate"):
            BoundingBoxContract(x0=5, y0=5, x1=5, y1=5)

    def test_a_one_dimensional_rectangle_is_still_a_rectangle(self):
        # A single line of text legitimately has zero height in one axis. The
        # degenerate rule is deliberately two-dimensional for that reason.
        assert BoundingBoxContract(x0=0, y0=4, x1=100, y1=4).to_dataclass().height == 0.0

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_a_non_finite_coordinate_is_refused(self, bad):
        # NaN survives every comparison in Python, so the ordering check alone
        # would accept it and produce a bbox whose width is NaN and whose
        # highlight covers nothing.
        with pytest.raises(ValidationError, match="finite"):
            BoundingBoxContract(x0=bad, y0=0, x1=10, y1=10)

    def test_an_unknown_field_is_refused(self):
        with pytest.raises(ValidationError):
            BoundingBoxContract(x0=0, y0=0, x1=1, y1=1, page=3)

    def test_it_round_trips_losslessly(self):
        box = BoundingBox(10.5, 20.25, 110.5, 60.75)

        assert BoundingBoxContract.from_dataclass(box).to_dataclass() == box


# --------------------------------------------------------------------------
# Locators: the strictest contract in the module
# --------------------------------------------------------------------------


class TestSourceLocatorContractRejections:
    def test_an_unrecognised_kind_is_refused_rather_than_downgraded(self):
        # ``SourceLocator(kind="hologram")`` becomes ``unknown``. Silently, here,
        # that is a document whose locators all claim to be unlocatable, and the
        # only symptom is that every citation in it shows no page.
        with pytest.raises(ValidationError, match="unknown locator kind"):
            located(kind="hologram")

    def test_a_pdf_page_without_a_page_number_is_refused(self):
        # The dataclass keeps this and reports ``is_resolvable is False``. At a
        # boundary, a pdf_page that resolves to nothing is a citation the reader
        # will click and find empty; ``kind="unknown"`` is the honest spelling.
        with pytest.raises(ValidationError, match="must carry 'page'"):
            located(kind="pdf_page")

    def test_page_zero_is_refused(self):
        # Zero is what an off-by-one in a converter produces, and it resolves
        # to a real-looking citation pointing one before the start.
        with pytest.raises(ValidationError, match="index must be >= 1"):
            located(kind="pdf_page", page=0)

    def test_a_negative_page_is_refused(self):
        with pytest.raises(ValidationError, match="index must be >= 1"):
            located(kind="pdf_page", page=-3)

    def test_a_locator_carrying_another_kinds_field_is_refused(self):
        # This is the corruption that survives every consumer. A pdf_page
        # carrying a slide number is read by dispatching on ``kind``, so the
        # slide is never looked at — and the page it does carry may be the
        # wrong one precisely because two vocabularies got mixed.
        with pytest.raises(ValidationError, match="does not belong"):
            located(kind="pdf_page", page=2, slide=3)

    def test_a_txt_offset_carrying_a_page_is_refused(self):
        with pytest.raises(ValidationError, match="does not belong"):
            located(kind="txt_offset", offset_start=0, offset_end=10, page=4)

    def test_a_reversed_offset_span_is_refused(self):
        # A reversed span still resolves to a region, and that region is
        # somewhere the reader did not highlight from.
        with pytest.raises(ValidationError, match="precedes offset_start"):
            located(kind="txt_offset", offset_start=400, offset_end=100)

    def test_a_blank_sheet_name_is_refused(self):
        # A sheet that is named but empty is a locator that resolves to a
        # citation reading "sheet ", which looks like a real answer.
        with pytest.raises(ValidationError, match="sheet name must not be blank"):
            located(kind="sheet_range", sheet="   ")

    def test_an_unknown_field_is_refused(self):
        with pytest.raises(ValidationError):
            located(kind="pdf_page", page=1, zoom=2)

    def test_a_docx_element_without_a_paragraph_is_refused(self):
        with pytest.raises(ValidationError, match="must carry 'paragraph'"):
            located(kind="docx_element")

    def test_an_unknown_locator_carries_nothing_and_is_accepted(self):
        # Requiring a position for ``unknown`` would force adapters to fabricate
        # one, which is the exact failure the kind exists to prevent.
        contract = located(kind="unknown")

        assert contract.to_dataclass() == SourceLocator(kind="unknown")


class TestSourceLocatorContractAcceptance:
    @pytest.mark.parametrize(
        ("locator"),
        [
            SourceLocator(kind="pdf_page", page=7, bbox=BoundingBox(10, 20, 110, 60)),
            SourceLocator(kind="slide", slide=3, element=2),
            SourceLocator(kind="sheet_range", sheet="Sheet1", cell_range="A1:D9"),
            SourceLocator(kind="docx_element", paragraph=43, heading_path=("Intro",)),
            SourceLocator(kind="txt_offset", offset_start=100, offset_end=400),
            SourceLocator(),
        ],
    )
    def test_every_valid_locator_shape_round_trips(self, locator):
        assert SourceLocatorContract.from_dataclass(locator).to_dataclass() == locator

    def test_a_json_heading_path_array_is_accepted(self):
        # The serialized form has always been a JSON array, and rejecting the
        # array would push the tuple conversion onto every caller.
        contract = located(kind="docx_element", paragraph=1, heading_path=["A", "B"])

        assert contract.heading_path == ("A", "B")

    def test_a_pdf_page_without_a_bbox_is_still_valid(self):
        assert located(kind="pdf_page", page=1).bbox is None

    def test_a_contract_is_frozen(self):
        # A validated snapshot that can be mutated after validation is a claim
        # about a state nobody checked.
        contract = located(kind="pdf_page", page=1)

        with pytest.raises(ValidationError):
            contract.page = 99


# --------------------------------------------------------------------------
# Elements
# --------------------------------------------------------------------------


class TestElementContractRejections:
    def test_an_unknown_role_is_refused_rather_than_downgraded(self):
        # ``Element(role="interpretive_dance")`` becomes ``text``, and the
        # element is still there. Here the loss is different: a role the build
        # does not know means the producer and this build disagree about the
        # vocabulary, and every downstream branch on role will be wrong.
        with pytest.raises(ValidationError, match="unknown element role"):
            ElementContract(
                role="interpretive_dance",
                text="x",
                element_id="el_1",
                locator=located(kind="pdf_page", page=1),
            )

    def test_a_blank_role_is_refused(self):
        with pytest.raises(ValidationError, match="unknown element role"):
            ElementContract(
                role="",
                text="x",
                element_id="el_1",
                locator=located(kind="pdf_page", page=1),
            )

    def test_a_formula_role_without_latex_is_refused(self):
        # The eval runner counts role-``formula`` elements as formulas whether
        # or not they carry LaTeX, so an empty one inflates the denominator of
        # the metric that decides whether a parser is fit for use.
        with pytest.raises(ValidationError, match="must carry LaTeX"):
            ElementContract(
                role="formula",
                element_id="el_1",
                locator=located(kind="pdf_page", page=1),
            )

    def test_a_table_role_without_a_grid_is_refused(self):
        # A table that arrived as tab-separated prose is exactly what the
        # ``table_dense`` corpus category exists to catch, and it can only be
        # caught if the flattening shows up as a *missing* grid rather than as
        # an empty one.
        with pytest.raises(ValidationError, match="must carry a grid"):
            ElementContract(
                role="table",
                text="a\tb",
                element_id="el_1",
                locator=located(kind="pdf_page", page=1),
            )

    def test_latex_under_a_non_formula_role_is_refused(self):
        # Silently carried LaTeX under a paragraph role is invisible to the
        # formula-counting path, so the document both contains a formula and
        # reports no formulas.
        with pytest.raises(ValidationError, match="does not use LaTeX"):
            ElementContract(
                role="paragraph",
                text="x",
                element_id="el_1",
                locator=located(kind="pdf_page", page=1),
                latex="E=mc^2",
            )

    def test_latex_and_formula_that_disagree_are_refused(self):
        with pytest.raises(ValidationError, match="disagree"):
            ElementContract(
                role="formula",
                text="$$E=mc^2$$",
                element_id="el_1",
                locator=located(kind="pdf_page", page=1),
                latex="E=mc^2",
                formula=FormulaContract(latex="E=mc^3"),
            )

    def test_a_figure_under_a_non_figure_role_is_refused(self):
        with pytest.raises(ValidationError, match="does not carry a figure"):
            ElementContract(
                role="paragraph",
                text="x",
                element_id="el_1",
                locator=located(kind="pdf_page", page=1),
                figure=FigureContract(caption="Figure 1"),
            )

    def test_a_blank_element_id_is_refused(self):
        with pytest.raises(ValidationError, match="element_id"):
            ElementContract(
                role="paragraph",
                text="x",
                element_id="",
                locator=located(kind="pdf_page", page=1),
            )

    def test_a_negative_order_is_refused(self):
        with pytest.raises(ValidationError, match="order must be >= 0"):
            ElementContract(
                role="paragraph",
                text="x",
                element_id="el_1",
                order=-1,
                locator=located(kind="pdf_page", page=1),
            )

    def test_a_zero_heading_level_is_refused(self):
        with pytest.raises(ValidationError, match="heading level"):
            ElementContract(
                role="heading",
                text="x",
                element_id="el_1",
                level=0,
                locator=located(kind="pdf_page", page=1),
            )

    def test_a_header_count_larger_than_the_grid_is_refused(self):
        # ``to_html`` slices on header_rows, so an oversized count renders the
        # whole table into a ``<thead>`` — a table where every row is a label.
        with pytest.raises(ValidationError, match="exceeds the row count"):
            TableContract(rows=[["a"]], header_rows=3)

    def test_a_negative_header_count_is_refused(self):
        with pytest.raises(ValidationError, match="header_rows must be >= 0"):
            TableContract(rows=[["a"]], header_rows=-1)


class TestElementContractRoundTrip:
    @pytest.mark.parametrize(
        ("element"),
        [
            paragraph("plain prose", page=1),
            Element(role="title", text="Attention Is All You Need", order=0),
            Element(role="heading", text="Method", level=2, order=1),
            Element(role="formula", latex="E = mc^2",
                    locator=SourceLocator(kind="pdf_page", page=3)),
            Element(role="figure", text="Figure 1: the encoder stack.",
                    locator=SourceLocator(kind="pdf_page", page=4)),
            Element(role="figure", text="",
                    locator=SourceLocator(kind="pdf_page", page=4)),
            Element(role="table", table=Table(rows=[["Acc", "0.91"]], header_rows=1,
                                               caption="Results"),
                    locator=SourceLocator(kind="pdf_page", page=5,
                                          bbox=BoundingBox(1, 2, 30, 40))),
            Element(role="header", text="Journal of Things",
                    locator=SourceLocator(kind="pdf_page", page=9)),
            Element(role="text", text="", locator=SourceLocator()),
            Element(role="paragraph", text="x", heading_path=("A", "B"), level=3,
                    page_number_label="p. 1", unit_id="unit_1"),
        ],
    )
    def test_every_element_shape_survives_the_round_trip(self, element):
        assert ElementContract.from_dataclass(element).to_dataclass() == element

    def test_a_whole_document_survives_the_round_trip(self):
        document = ParsedDocument(
            source_filename="paper.pdf",
            media_type="application/pdf",
            sha256="a" * 64,
            page_count=3,
            elements=[
                paragraph("first", page=1, order=0),
                Element(role="table", table=Table(rows=[["a", "b"]]),
                        locator=SourceLocator(kind="pdf_page", page=2), order=1),
                Element(role="formula", latex="E = mc^2",
                        locator=SourceLocator(kind="pdf_page", page=2), order=2),
            ],
            parser_name="docling",
            parser_version="2.0",
            options_hash="h1",
            warnings=["degraded bbox"],
        )

        assert ParsedDocumentContract.from_dataclass(document).to_dataclass() == document

    def test_units_survive_the_round_trip(self):
        unit = SourceUnit(locator=SourceLocator(kind="pdf_page", page=3), label="p. 3",
                          text="content", order=2)

        assert SourceUnitContract.from_dataclass(unit).to_dataclass() == unit

    def test_a_blank_unit_id_is_refused(self):
        with pytest.raises(ValidationError, match="unit_id"):
            SourceUnitContract(unit_id="", locator=located(kind="pdf_page", page=1))


# --------------------------------------------------------------------------
# The document contract
# --------------------------------------------------------------------------


class TestParsedDocumentContractVersionGating:
    def test_the_current_version_is_accepted(self):
        contract = ParsedDocumentContract(
            schema_version=SCHEMA_VERSION, document_id="doc_1")

        assert contract.schema_version == SCHEMA_VERSION

    def test_a_future_major_version_is_refused(self):
        with pytest.raises(ValidationError, match="unsupported schema version"):
            ParsedDocumentContract(schema_version="9.0", document_id="doc_1")

    @pytest.mark.parametrize("bad", ["1", "", "one.zero", "1.0.0", "v1.0", " 1.0"])
    def test_a_malformed_version_is_refused(self, bad):
        # ``ParsedDocument.from_dict`` accepts "1": it splits on "." and
        # compares the first piece, so a version that is not MAJOR.MINOR falls
        # through to a comparison that happens to pass. At a boundary an
        # unparseable version is not a minor addition, it is a document whose
        # shape cannot be established.
        with pytest.raises(ValidationError, match="malformed schema_version"):
            ParsedDocumentContract(schema_version=bad, document_id="doc_1")

    def test_a_minor_bump_still_reads(self):
        assert ParsedDocumentContract(
            schema_version="1.7", document_id="doc_1").schema_version == "1.7"

    def test_a_blank_document_id_is_refused(self):
        with pytest.raises(ValidationError, match="document_id"):
            ParsedDocumentContract(schema_version=SCHEMA_VERSION, document_id="  ")

    def test_a_negative_page_count_is_refused(self):
        with pytest.raises(ValidationError, match="page_count"):
            ParsedDocumentContract(
                schema_version=SCHEMA_VERSION, document_id="doc_1", page_count=-2)

    def test_an_unknown_top_level_field_is_refused(self):
        # A field this build does not recognise in a document written by another
        # build is a version mismatch the major-version check cannot see.
        # Defaulting it away is how a locator written by a newer build loses its
        # discriminating field and starts meaning something else.
        with pytest.raises(ValidationError):
            ParsedDocumentContract(
                schema_version=SCHEMA_VERSION, document_id="doc_1", ocr_used=True)


# --------------------------------------------------------------------------
# Chunks
# --------------------------------------------------------------------------


class TestChunkContract:
    def test_a_chunk_that_names_no_element_is_refused(self):
        # Not a smaller chunk: a passage that cannot be traced back to the
        # document it claims to quote.
        with pytest.raises(ValidationError, match="at least one element"):
            ChunkContract(chunk_id="c1", locator=located(kind="pdf_page", page=1))

    def test_a_blank_element_id_is_refused(self):
        with pytest.raises(ValidationError, match="blank id"):
            ChunkContract(
                chunk_id="c1",
                element_ids=("  ",),
                locator=located(kind="pdf_page", page=1),
            )

    def test_a_negative_token_estimate_is_refused(self):
        with pytest.raises(ValidationError, match="must be >= 0"):
            ChunkContract(
                chunk_id="c1", element_ids=("e1",), token_estimate=-1,
                locator=located(kind="pdf_page", page=1))

    def test_it_round_trips_losslessly(self):
        chunk = Chunk(
            text="body",
            element_ids=("e1", "e2"),
            locator=SourceLocator(kind="pdf_page", page=2, bbox=BoundingBox(1, 2, 3, 4)),
            heading_path=("Intro",),
            order=1,
            token_estimate=42,
            index_version="openai:text-embedding-3-small:1536",
        )

        assert ChunkContract.from_dataclass(chunk).to_dataclass() == chunk

    def test_a_json_element_id_array_is_accepted(self):
        contract = ChunkContract(
            chunk_id="c1", element_ids=["e1", "e2"],
            locator=located(kind="pdf_page", page=1))

        assert contract.element_ids == ("e1", "e2")


# --------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------


class TestEvidenceContract:
    @pytest.mark.parametrize("bad", [-0.1, 1.1, math.nan, math.inf])
    def test_a_score_outside_the_unit_interval_is_refused(self, bad):
        # The score is rendered as a confidence bar. A reranker that changed its
        # scale produces a bar wider than its own track, which reads as
        # certainty the retrieval does not have.
        with pytest.raises(ValidationError, match=r"\[0, 1\]"):
            EvidenceContract(evidence_id="e1", locator=located(kind="pdf_page", page=1),
                             score=bad)

    def test_the_score_bounds_are_inclusive(self):
        assert EvidenceContract(evidence_id="e1", locator=located(kind="pdf_page", page=1),
                                score=0.0).score == 0.0
        assert EvidenceContract(evidence_id="e1", locator=located(kind="pdf_page", page=1),
                                score=1.0).score == 1.0

    def test_a_blank_evidence_id_is_refused(self):
        # The API addresses evidence by its id, so a blank one is a route that
        # resolves to nothing.
        with pytest.raises(ValidationError, match="evidence_id"):
            EvidenceContract(evidence_id="", locator=located(kind="pdf_page", page=1))

    def test_it_round_trips_losslessly(self):
        evidence = Evidence(
            chunk_id="c1",
            text="A retrieved passage.",
            locator=SourceLocator(kind="pdf_page", page=7, bbox=BoundingBox(1, 2, 3, 4)),
            unit_id="unit_1",
            heading_path=("Method", "Details"),
            score=0.87,
            is_direct_quote=True,
            source_parser="docling",
        )

        assert EvidenceContract.from_dataclass(evidence).to_dataclass() == evidence

    def test_an_unlocatable_evidence_is_accepted(self):
        # An evidence that cannot be pointed at is a degraded retrieval result,
        # not a broken one. The citation API's job is to say so explicitly
        # (release gate: "resolvable, or explicitly marked as having no visual
        # location"), which requires such an evidence to exist at all.
        contract = EvidenceContract(
            evidence_id="e1", text="passage", locator=located(kind="unknown"))

        assert contract.to_dataclass().locator.is_resolvable is False


# --------------------------------------------------------------------------
# Facts
# --------------------------------------------------------------------------


class TestFactContract:
    def test_a_value_without_evidence_is_refused(self):
        # The UI distinguishes "the paper says X" from "not stated in this
        # paper" by exactly this pair. A value with no citation renders like a
        # finding and traces to nothing.
        with pytest.raises(ValidationError, match="no evidence_ids"):
            FactContract(field_name="contributions", value="We invented this",
                         extraction_status="extracted")

    def test_a_value_marked_not_found_is_refused(self):
        # The two fields disagree about whether the paper said anything, and one
        # of them is going to be believed.
        with pytest.raises(ValidationError, match="marked not_found"):
            FactContract(field_name="contributions", value="C", evidence_ids=("e1",),
                         extraction_status="not_found")

    def test_an_unrecognised_status_is_refused(self):
        # The dataclass downgrades it to ``not_found``, which is the safe
        # direction, but the downgrade is invisible to whoever supplied it — so
        # an extraction pipeline mislabelling every status it does not recognize
        # is a bug nobody sees.
        with pytest.raises(ValidationError, match="unknown extraction_status"):
            FactContract(field_name="c", extraction_status="vibes")

    def test_a_blank_field_name_is_refused(self):
        with pytest.raises(ValidationError, match="field_name"):
            FactContract(field_name="  ")

    def test_a_blank_evidence_id_is_refused(self):
        with pytest.raises(ValidationError, match="blank id"):
            FactContract(field_name="c", value="C", evidence_ids=("",))

    def test_a_not_found_fact_with_no_value_is_accepted(self):
        # "The paper does not say" is a real answer and has to be storable, or
        # the UI cannot render it as anything but a missing section.
        fact = FactContract(field_name="c", extraction_status="not_found")

        assert fact.value == ""

    def test_a_grounded_fact_round_trips(self):
        fact = Fact(field_name="c1", value="We scale to 8 GPUs.", evidence_ids=("e1",),
                    extraction_status="extracted", conflicting_values=("2 GPUs",))

        assert FactContract.from_dataclass(fact).to_dataclass() == fact

    def test_a_conflicting_fact_with_several_alternatives_is_accepted(self):
        fact = Fact(field_name="r1", value="99%", evidence_ids=("e1", "e2"),
                    extraction_status="conflicting",
                    conflicting_values=("94%", "97%", "99%"))

        assert FactContract.from_dataclass(fact).conflicting_values == ("94%", "97%", "99%")

    def test_whole_paper_facts_survive_the_round_trip(self):
        facts = PaperFacts(
            research_question=Fact(field_name="research_question", value="Q",
                                   evidence_ids=("e1",)),
            contributions=[Fact(field_name="c1", value="C1", evidence_ids=("e2",))],
            main_results=[Fact(field_name="r1", value="99%", evidence_ids=("e3",))],
            method=Fact(field_name="method", value="LoRA", evidence_ids=("e4",)),
            extraction_model="gpt-4o-mini",
            prompt_version="p1",
            evidence_version="e1",
        )

        assert PaperFactsContract.from_dataclass(facts).to_dataclass() == facts

    def test_the_unfilled_singleton_blocks_are_accepted(self):
        # ``PaperFacts.from_dict`` substitutes a not_found fact for an absent
        # block, and that substitution is correct: a card that says nothing was
        # found must render, not 500.
        contract = PaperFactsContract(
            research_question=FactContract(field_name="research_question"),
            method=FactContract(field_name="method"),
            reproducibility=FactContract(field_name="reproducibility"),
        )

        assert contract.to_dataclass().coverage == 0.0


# --------------------------------------------------------------------------
# The upload edge
# --------------------------------------------------------------------------


class TestUploadRequestContract:
    @pytest.mark.parametrize(
        ("filename", "media_type"),
        [
            ("paper.pdf", "application/pdf"),
            ("notes.docx",
             "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
            ("deck.pptx",
             "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
            ("readme.txt", "text/plain"),
            ("READER.PDF", "application/pdf"),
        ],
    )
    def test_a_supported_upload_is_accepted(self, filename, media_type):
        contract = UploadRequestContract(filename=filename, media_type=media_type)

        assert contract.filename == filename

    def test_the_browser_pdf_alias_is_normalized(self):
        # Browsers send ``application/x-pdf``. An adapter comparing against the
        # IANA spelling will not match it, and that is how a PDF silently takes
        # the plain-text path.
        contract = UploadRequestContract(filename="p.pdf", media_type="application/x-pdf")

        assert contract.media_type == "application/pdf"

    def test_an_unsupported_extension_is_refused(self):
        with pytest.raises(ValidationError, match="unsupported file type"):
            UploadRequestContract(filename="payload.exe")

    def test_a_file_with_no_extension_is_refused(self):
        with pytest.raises(ValidationError, match="unsupported file type"):
            UploadRequestContract(filename="paper")

    def test_an_empty_filename_is_refused(self):
        # A file input submitted before anything was chosen sends "", not None.
        with pytest.raises(ValidationError, match="must not be empty"):
            UploadRequestContract(filename="")

    def test_a_filename_carrying_a_path_is_refused(self):
        # Either a client sending a full path or a traversal attempt. The
        # storage layer should never be handed a name it did not construct.
        with pytest.raises(ValidationError, match="bare name"):
            UploadRequestContract(filename="../../etc/passwd.pdf")

    def test_a_windows_path_in_a_filename_is_refused(self):
        with pytest.raises(ValidationError, match="bare name"):
            UploadRequestContract(filename=r"C:\Windows\paper.pdf")

    def test_an_undeclared_media_type_is_fine(self):
        # A client that declares nothing has not disagreed with the extension,
        # it has said nothing.
        assert UploadRequestContract(filename="p.pdf").media_type == ""

    def test_an_unrecognized_media_type_is_refused(self):
        with pytest.raises(ValidationError, match="unsupported media type"):
            UploadRequestContract(filename="p.pdf", media_type="image/png")

    def test_the_extension_and_the_declared_type_are_reported_separately(self):
        # They are two independent claims; deciding which wins is router policy,
        # so the contract normalizes both and refuses to collapse them.
        contract = UploadRequestContract(filename="p.docx", media_type="text/plain")

        assert contract.media_type == "text/plain"
        assert contract.media_type_for_extension().endswith("wordprocessingml.document")


# --------------------------------------------------------------------------
# Rich payloads on their own
# --------------------------------------------------------------------------


class TestRichPayloadContracts:
    def test_a_blank_formula_is_refused(self):
        with pytest.raises(ValidationError, match="non-empty LaTeX"):
            FormulaContract(latex="   ")

    def test_a_formula_round_trips(self):
        formula = Formula(latex="E = mc^2", display=False)

        assert FormulaContract.from_dataclass(formula).to_dataclass() == formula

    def test_a_figure_with_no_caption_is_accepted(self):
        # The adapters emit one on purpose — an addressable region is still worth
        # a citation — and refusing it would mean refusing the parse, which is
        # worse than a figure the search path finds no text in.
        figure = FigureContract()

        assert figure.to_dataclass() == Figure()

    def test_a_figure_round_trips(self):
        figure = Figure(caption="Figure 1: the encoder stack.",
                        image_uri="artifact://fig1.png")

        assert FigureContract.from_dataclass(figure).to_dataclass() == figure

    def test_a_table_round_trips(self):
        table = Table(rows=[["a", "b"], ["c"]], header_rows=1, caption="T")

        assert TableContract.from_dataclass(table).to_dataclass() == table

    def test_a_ragged_table_is_accepted(self):
        # Ragged rows are padded on projection, so refusing them here would
        # reject a parse that renders correctly.
        table = TableContract(rows=[["a", "b", "c"], ["d"]], header_rows=1)

        assert table.to_dataclass().column_count == 3
