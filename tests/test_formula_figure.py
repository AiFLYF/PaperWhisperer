"""T0.2: a formula is the same formula everywhere, and a reader can tell whose it is.

The plan names ``Formula`` and ``Figure`` among the required canonical types and
ships a ``formula_dense`` corpus category whose stated focus is "公式保留：
LaTeX/公式是否被破坏或丢失" — formula preservation. Five real documents exist for
it, and they are used here rather than a fixture of invented strings, because
the failures this suite is about are failures of *real* text layers: a Type 1
font that hands back U+FB01 for "fi", a TeX-produced page whose text layer
emits ``α`` rather than ``\\alpha``, an encoding with no ToUnicode entry that
leaks ``/greaterorequalslant`` into running prose.

Three properties are under test, in ascending order of how quietly they fail.

**Display and inline are different things.** A ``$$…$$`` block dropped into the
middle of a sentence is what a reader calls a rendering bug; a display
equation demoted to prose is a flattening the reader cannot see. Both come
from the same boolean, so the boolean carries its own provenance and the tests
assert the provenance, not just the value — a table that says "all 40 formulas
are display" for a document whose parser never distinguished one of them is a
lie that looks like data.

**A formula is compared, not echoed.** The gate between "this formula is the
paper's" and "this formula is the model's" cannot be a byte comparison: the
parser transliterated Unicode maths out of a text layer and the model wrote
LaTeX from reading the same line, and the two spellings differ in ways no
reader can see. So :mod:`paperwhisperer.ingestion.latex` states which
differences are noise, and these tests check both halves of that statement —
the variations a real PDF produces are equal, and formulas that differ are
different. A tolerance nobody can trace to an observation is a tolerance that
eventually folds two different expressions together, and the second half of
this file is what stops that.

**Nothing is dropped silently.** An unbalanced brace, an unclosed environment,
a formula the parser cut in half, a ``$`` the model never closed, a caption
with no picture: each has a defined outcome, and none of them is a formula
vanishing from an export. A dropped formula is a corruption the user cannot
see, which is the one failure this whole task is arranged against.

The corpus reads are guarded by ``skipif`` with a stated reason rather than
skipped silently, so a run without ``eval/corpus`` says why it did not check
the real thing instead of quietly reporting a pass.
"""

from __future__ import annotations

import ast
import io
import pathlib
import re
import zipfile

import pytest

from paperwhisperer.analysis import service
from paperwhisperer.analysis.service import (
    EXPORT_FORMULA_PRESENTATION_LABELS,
    build_citation_index,
    build_formula_display_summary,
    build_formula_index,
    render_docx_report,
    render_formula_index_markdown,
    render_html_report,
    render_markdown_report,
)
from paperwhisperer.ingestion.contracts import FigureContract, FormulaContract
from paperwhisperer.ingestion.latex import (
    LATEX_EQUIVALENCE_RULES,
    LatexVerdict,
    canonical_latex,
    compare_latex,
    find_inline_math,
    join_latex,
    latex_defects,
)
from paperwhisperer.ingestion.schema import (
    DEFAULT_DISPLAY_WHEN_UNKNOWN,
    FORMULA_DISPLAY_SOURCES,
    PAGE_IMAGE_PATH_TEMPLATE,
    Element,
    Figure,
    Formula,
    ParsedDocument,
    SourceLocator,
)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
CORPUS = PROJECT_ROOT / "eval" / "corpus" / "formula_dense"

#: Every document in the ``formula_dense`` category, from ``eval/manifest.yaml``.
#: Named explicitly rather than globbed so that a corpus which has been
#: re-downloaded or partially removed fails by *name* -- "green_tao_primes is
#: gone" is a diagnosable fact and "the formula_dense tests did not run" is
#: not.
FORMULA_DENSE_DOCUMENTS = (
    "dbrane_theory_review.pdf",
    "green_tao_primes.pdf",
    "maldacena_large_n.pdf",
    "maldacena_toroidal_compactification.pdf",
    "witten_ads_holography.pdf",
)

corpus_missing = pytest.mark.skipif(
    not CORPUS.is_dir() or not any((CORPUS / name).is_file() for name in FORMULA_DENSE_DOCUMENTS),
    reason=(
        "eval/corpus/formula_dense is not present in this checkout. The five "
        "documents are the real input this task is about -- a fixture of "
        "invented strings would pass while every real text layer failed."
    ),
)


def docling_export_available() -> bool:
    """Whether the real docling backend can be imported at all."""
    try:
        import docling  # noqa: F401
    except Exception:  # noqa: BLE001 - any import failure means "not available"
        return False
    return True


def formula_text(pdf_name: str, pages: int = 3) -> str:
    """The first ``pages`` pages of a corpus PDF as one text-layer string."""
    import pypdf

    reader = pypdf.PdfReader(str(CORPUS / pdf_name))
    return "".join(reader.pages[index].extract_text() for index in range(min(pages, len(reader.pages))))


# --------------------------------------------------------------------------
# display: who sets it, and what the fallback honestly says
# --------------------------------------------------------------------------


class TestDisplayProvenance:
    """A boolean with a default cannot be a claim on its own.

    ``display`` alone makes "nobody looked" indistinguishable from "the backend
    observed every formula to be a block", and those are different documents.
    The second field is what tells them apart, so it is the second field the
    tests pin.
    """

    def test_a_formula_nobody_annotated_does_not_claim_to_be_observed(self):
        formula = Formula(latex="E = mc^2")

        assert formula.display_source == "unknown"
        assert formula.display is DEFAULT_DISPLAY_WHEN_UNKNOWN

    def test_the_unknown_default_prefers_the_visible_error_to_the_invisible_one(self):
        """Documented, not incidental: the two failure modes are not symmetric.

        An inline formula rendered as a display block is a ``$$…$$`` in the
        middle of a sentence -- ugly, and the reader can see it. A display
        formula rendered inline is demoted to prose by the chunker and merged
        into the paragraph around it, which is invisible. The default refuses
        the error nobody can see.
        """
        assert DEFAULT_DISPLAY_WHEN_UNKNOWN is True

    @pytest.mark.parametrize("source", FORMULA_DISPLAY_SOURCES)
    def test_every_declared_source_is_accepted_and_kept(self, source):
        formula = Formula(latex="x", display=False, display_source=source)

        assert formula.display_source == source

    def test_an_invented_source_degrades_to_unknown_rather_than_being_carried(self):
        # A display table built out of this value would otherwise report a
        # provenance that nothing in the system can produce.
        formula = Formula(latex="x", display_source="the backend felt sure")

        assert formula.display_source == "unknown"

    def test_the_backend_source_survives_serialization_while_the_default_does_not(self):
        # Pruned at the default so a document written by this build is
        # byte-identical to one written by the build before the field existed.
        assert Formula(latex="x").to_dict() == {"latex": "x"}
        assert Formula(latex="x", display_source="backend").to_dict() == {
            "latex": "x",
            "display_source": "backend",
        }


class TestDisplayInference:
    def test_a_tex_environment_says_so_in_its_own_right(self):
        element = Element(role="formula", latex=r"\begin{equation}E = mc^2\end{equation}")

        assert element.formula.display_source == "latex"
        assert element.formula.display is True

    def test_kept_display_delimiters_are_an_assertion_too(self):
        element = Element(role="formula", latex="$$E = mc^2$$")

        assert element.formula.display_source == "latex"

    def test_a_bare_formula_is_not_assumed_to_be_observed(self):
        element = Element(role="formula", latex="E = mc^2")

        assert element.formula.display_source == "unknown"

    def test_the_two_directions_round_trip_distinctly(self):
        display = Element(role="formula", latex="x", formula=Formula(latex="x", display=True))
        inline = Element(
            role="formula", latex="x", formula=Formula(latex="x", display=False)
        )

        assert display.text == "$$x$$"
        assert inline.text == "$x$"
        assert display.to_dict()["formula"] != inline.to_dict()["formula"]
        assert Element.from_dict(inline.to_dict()).formula.display is False
        assert Element.from_dict(display.to_dict()).formula.display is True

    def test_an_element_built_from_flat_latex_still_ends_up_with_both_ends_in_agreement(self):
        # The adapter-facing direction: docling builds
        # ``Element(role="formula", latex=...)`` and the structured type is
        # derived, so the two can never disagree.
        element = Element(role="formula", latex=r"q_i = \exp(z_i/T)")

        assert element.formula is not None
        assert element.latex == element.formula.latex
        assert element.text == element.formula.rendered


# --------------------------------------------------------------------------
# The normalisation contract
# --------------------------------------------------------------------------


class TestNormalisationContract:
    def test_the_rules_are_data_and_each_one_is_named(self):
        # A tolerance nobody can attribute to an observation is a tolerance
        # that will eventually fold two different formulas together.
        rule_ids = [rule.rule_id for rule in LATEX_EQUIVALENCE_RULES]

        assert len(rule_ids) == len(set(rule_ids))
        for rule in LATEX_EQUIVALENCE_RULES:
            assert rule.description.strip(), rule.rule_id

    @pytest.mark.parametrize(
        ("left", "right", "why"),
        [
            (
                r"\dfrac{a}{b}",
                r"\frac{a}{b}",
                "a display fraction reproduced inline comes back as \\dfrac",
            ),
            (r"\tfrac12", r"\frac{1}{2}", "a text layer drops the braces on short arguments"),
            ("$$x$$", "x", "the carrier's delimiters are not the expression"),
            (r"\left(\frac{a}{b}\right)", r"(\frac{a}{b})", "\\left/\\right choose a size only"),
            ("x^{2}", "x^2", "a model braces a one-character exponent, a text layer does not"),
            (r"\mathrm{ln} x", "ln x", "upright-ness is invisible in a text layer"),
            (r"\alpha' = \mathrm{fixed} . (2.1)", r"\alpha' = \mathrm{fixed}",
             "a trailing equation number is page furniture"),
            (r"\sqrt2", r"\sqrt{2}", "the radical's argument may or may not be braced"),
        ],
    )
    def test_a_real_variation_of_the_same_formula_compares_equal(self, left, right, why):
        comparison = compare_latex(left, right)

        assert comparison.matches, f"{why}: {left!r} vs {right!r} -> {comparison.reason}"

    @pytest.mark.parametrize(
        ("left", "right", "why"),
        [
            ("E = mc^2", "E = mc^3", "different exponents"),
            (r"\frac{a}{b}", r"\frac{b}{a}", "the fraction is inverted"),
            (r"\sum_{i} x_i", r"\prod_{i} x_i", "different operators"),
            (r"x_{i}", r"x_{j}", "different indices"),
            (r"\alpha", r"\beta", "different Greek letters"),
            (r"\mathbf{v}", "v", "weight is visible to a reader, so it is not noise"),
            (r"\mathbb{R}", r"R", "a blackboard bold R is not an R"),
        ],
    )
    def test_a_genuinely_different_formula_compares_different(self, left, right, why):
        # The other half of the contract. A tolerance wide enough to make the
        # cases above pass would pass every one of these, and the gate would
        # then call a fabricated formula the paper's.
        comparison = compare_latex(left, right)

        assert comparison.verdict is LatexVerdict.DIFFERENT, why

    def test_an_empty_side_is_uncomparable_and_never_different(self):
        """A formula the parser recovered as an image has no LaTeX to compare.

        Reporting that as ``DIFFERENT`` would accuse the model of inventing a
        formula the parser simply could not read -- a false accusation produced
        by a comparison that never happened.
        """
        comparison = compare_latex("", r"\frac{a}{b}")

        assert comparison.verdict is LatexVerdict.UNCOMPARABLE
        assert comparison.matches is False

    def test_an_unestablished_match_is_never_presented_as_a_match(self):
        assert compare_latex("", "x").matches is False
        assert compare_latex("x", "x").matches is True

    def test_canonicalisation_is_idempotent(self):
        # The gate calls it once per side; a normaliser that is not idempotent
        # would make the answer depend on how many times a caller normalised.
        once = canonical_latex(r"\dfrac{α}{2}").text

        assert canonical_latex(once).text == once

    def test_a_formula_answers_the_comparison_through_its_own_api(self):
        formula = Formula(latex=r"q_i = \exp(z_i/T)")

        assert formula.matches(r"q_{i} = \exp\left(z_{i}/T\right)")
        assert not formula.matches(r"q_i = \exp(z_j/T)")

    def test_unicode_maths_and_leaked_glyph_names_both_land_on_latex(self):
        # Both are observed in the corpus rather than invented for the test:
        # the first in every formula_dense document, the second specifically in
        # green_tao_primes, which emits '/greaterorequalslant' in prose.
        assert canonical_latex("α ≤ β").text == r"\alpha\le\beta"
        assert canonical_latex("E(f(x)|x ∈ ZN) /greaterorequalslantδ").text == (
            r"E(f(x)|x\inZN)\ge\delta"
        )


@corpus_missing
class TestAgainstRealCorpus:
    """What a real text layer actually produces, read from the documents.

    Everything else in this file is a string somebody typed. These are strings a
    TeX-produced PDF emitted, which is the only place the normalisation contract
    can be falsified.
    """

    def test_every_formula_dense_document_extracts_text(self):
        for name in FORMULA_DENSE_DOCUMENTS:
            text = formula_text(name, pages=2)
            assert text.strip(), f"{name} produced no text layer"

    def test_the_corpus_exercises_the_ligature_tolerance(self):
        # The tolerance exists because a Type 1 font maps "fi" to one glyph and
        # the extractor hands back U+FB01. If that stopped being true the rule
        # would be dead weight; if it is not being handled, every formula in
        # the document is mangled.
        text = formula_text("maldacena_large_n.pdf", pages=8)

        assert "ﬁ" in text, "no U+FB01 in the corpus page this tolerance was written for"
        assert canonical_latex("ﬁxed").text == "fixed"

    def test_the_corpus_exercises_the_unicode_maths_tolerance(self):
        text = formula_text("maldacena_large_n.pdf", pages=8)

        assert any(char in text for char in "αβγδ≤≡→√")

    def test_the_glyph_name_tolerance_is_needed_by_a_real_document(self):
        # This is the observation the rule is named after. If green_tao_primes
        # no longer leaks glyph names, the rule can go; while it does, every
        # formula containing one is unrenderable LaTeX without it.
        text = formula_text("green_tao_primes.pdf", pages=4)

        assert "/greaterorequalslant" in text
        assert canonical_latex("/greaterorequalslant").text == r"\ge"

    def test_a_line_of_real_math_normalises_without_raising(self):
        # Every formula_dense document is a document whose formulas arrive
        # mangled. The normaliser is asked about all of them and must answer
        # for every one rather than failing on the first oddity.
        for name in FORMULA_DENSE_DOCUMENTS:
            text = formula_text(name, pages=3)
            for line in text.splitlines():
                stripped = line.strip()
                if len(stripped) < 4:
                    continue
                result = canonical_latex(stripped)
                assert isinstance(result.text, str)
                assert isinstance(latex_defects(stripped), tuple)

    def test_equation_numbers_are_recognised_where_the_corpus_puts_them(self):
        # A numbered display equation is the same expression with a number
        # beside it, and the number changes page to page. Treating it as part
        # of the formula means no formula ever matches the paper.
        text = formula_text("witten_ads_holography.pdf", pages=3)
        numbered = [
            line.strip()
            for line in text.splitlines()
            if re.search(r"\(\d+(?:\.\d+)*\)\s*$", line.strip())
        ]

        assert numbered, "no numbered equation found in witten_ads_holography"
        for line in numbered:
            body = re.sub(r"\s*\(\d+(?:\.\d+)*\)\s*$", "", line)
            assert canonical_latex(line).text == canonical_latex(body).text, line


# --------------------------------------------------------------------------
# Robustness: five malformed cases, five defined outcomes
# --------------------------------------------------------------------------


class TestMalformedFormula:
    def test_unbalanced_braces_are_reported_and_the_formula_still_exists(self):
        formula = Formula(latex=r"\frac{a}{b")

        assert "unbalanced_braces" in formula.defects
        assert formula.is_well_formed is False
        # It still renders. Dropping it would put a document that is silently
        # missing an equation the paper has.
        assert formula.rendered

    def test_an_unclosed_environment_is_reported(self):
        formula = Formula(latex=r"\begin{align} a &= b \\\\ c &= d")

        assert "unclosed_environment" in formula.defects

    def test_a_closed_environment_is_not_reported(self):
        formula = Formula(latex=r"\begin{align} a &= b\end{align}")

        assert formula.is_well_formed is True

    def test_a_dangling_command_is_reported(self):
        assert "dangling_command" in latex_defects("\\frac \\")

    def test_a_formula_split_across_two_elements_is_rejoined_not_dropped(self):
        """A display equation over two lines arrives as two broken halves.

        Each half is not LaTeX, so an export would show two errors where the
        paper has one equation. The merge keeps the second half's element id
        in the warning, because deleting the tail instead would leave a formula
        that looks complete and is not.
        """
        document = ParsedDocument(elements=[
            Element(role="formula", latex=r"\begin{align} a &= b",
                    locator=SourceLocator(kind="pdf_page", page=3), order=0),
            Element(role="formula", latex=r"\\ c &= d \end{align}",
                    locator=SourceLocator(kind="pdf_page", page=3), order=1),
        ])

        formulas = [element for element in document.elements if element.role == "formula"]
        assert len(formulas) == 1
        assert formulas[0].latex == r"\begin{align} a &= b \\ c &= d \end{align}"
        assert document.warnings
        # The tail's id is named, so a citation that pointed at the second half
        # can still be traced to the merged one.
        assert formulas[0].element_id in document.warnings[0]

    def test_two_consecutive_equations_are_not_fused(self):
        document = ParsedDocument(elements=[
            Element(role="formula", latex="a = b", locator=SourceLocator(kind="pdf_page", page=1)),
            Element(role="formula", latex="c = d", locator=SourceLocator(kind="pdf_page", page=1)),
        ])

        assert len([element for element in document.elements if element.role == "formula"]) == 2

    def test_join_refuses_to_guess(self):
        assert join_latex("a + b", "c + d") is None
        assert join_latex(r"\frac{a}{", "b}") == r"\frac{a}{ b}"
        assert join_latex("", "x") is None

    def test_latex_defects_never_raises_on_nonsense(self):
        for value in ("", "   ", "$$$", "\\", "}{", "\\begin{", "（", "\\frac{" * 50):
            assert isinstance(latex_defects(value), tuple)


class TestMalformedDelimiters:
    def test_an_unclosed_dollar_is_kept_and_flagged_not_discarded(self):
        # The reader still has to see what the model wrote; a span that is
        # removed because it is ambiguous is a formula that vanished.
        spans = find_inline_math("the result is $x + y and then some prose")

        assert len(spans) == 1
        assert spans[0].latex == "x + y and then some prose"
        assert spans[0].defect == "unbalanced_delimiters"

    def test_a_currency_amount_does_not_swallow_the_formulas_after_it(self):
        """The report's own cost figures are the reason this rule exists.

        Reading ``$5`` as an opening delimiter makes every ``$…$`` after it in
        the same paragraph part of one span, so a cost and a display equation
        reach the comparator as a single string and neither can match anything.
        The exporter writes dollar amounts, so this is not hypothetical.
        """
        text = "本次运行花了 $0.004213，核心公式是 $$E = mc^2$$。"
        spans = find_inline_math(text)

        assert [span.latex for span in spans] == ["E = mc^2"]
        assert spans[0].display is True

    def test_a_longer_dollar_run_is_flagged_rather_than_trimmed_to_length(self):
        # Choosing which of the three dollars to keep is a guess, and a guess
        # that silently changes a formula is the failure this module prevents.
        spans = find_inline_math("a $$x$$ b $$$y$$ c")

        assert any(span.defect == "ambiguous_delimiters" for span in spans)

    def test_a_mismatched_pair_does_not_pair_across_into_the_next_formula(self):
        """Resynchronisation, asserted on its failure mode.

        Stepping over both tokens on a mismatch would consume the closer as an
        opener, and it would then pair with the next real formula -- reporting
        a span running from inside one formula to inside the next.
        """
        spans = find_inline_math("$x$$y$$")

        assert [span.latex for span in spans] == ["x", "y"]

    def test_an_escaped_dollar_is_a_literal(self):
        assert [span.latex for span in find_inline_math("a literal \\$5 and a real $x$")] == ["x"]

    def test_no_maths_in_prose_yields_no_spans(self):
        assert find_inline_math("a paragraph with no mathematics in it") == ()


class TestFigureWithAndWithoutAnImage:
    def test_a_figure_with_a_real_reference_keeps_it(self):
        figure = Figure(caption="Figure 1", image_uri="artifact://sha256/ab/fig1.png")

        assert figure.has_image is True
        assert figure.has_resolvable_image is True

    def test_a_caption_with_no_image_says_so(self):
        figure = Figure(caption="Figure 1: the encoder stack.")

        assert figure.has_image is False
        assert figure.has_resolvable_image is False

    def test_a_dangling_relative_path_is_refused_rather_than_carried(self):
        """A figure with a path that resolves from nowhere is worse than one
        that admits it has no picture.

        A bare filename eventually resolves to a *different* image once the
        working directory changes, which is the case a caption with a dangling
        URI invites and the reason this is a type rather than a string.
        """
        figure = Figure(caption="Figure 1", image_uri="images/fig1.png")

        assert figure.image_uri == ""
        assert figure.has_image is False

    def test_the_caption_survives_the_refusal(self):
        figure = Figure(caption="Figure 1: the encoder stack.",
                        image_uri="images/fig1.png")

        assert figure.caption == "Figure 1: the encoder stack."

    def test_a_figure_round_trips_with_and_without_an_image(self):
        for figure in (Figure(), Figure(caption="Figure 1"),
                       Figure(caption="Figure 1", image_uri="artifact://a/b.png")):
            assert Figure.from_dict(figure.to_dict()) == figure

    def test_an_empty_image_uri_is_pruned_from_the_payload(self):
        # A permanent empty key in every figure's payload is bytes bought for
        # nothing, and its absence is also the honest signal.
        assert "image_uri" not in Figure(caption="Figure 1").to_dict()

    def test_a_figure_points_at_its_page_rather_than_at_a_crop(self):
        """What the pipeline can honestly produce, encoded.

        There is no stage that crops a picture, so the honest reference is the
        region of the page the figure occupies -- which the evidence API
        already serves. Inventing a crop path would be a figure that points
        somewhere real only by accident.
        """
        figure = Figure(caption="Figure 1: the encoder stack.")
        locator = SourceLocator(kind="pdf_page", page=7)

        assert figure.page_image_url(locator, "doc_1") == (
            "/v1/documents/doc_1/pages/7/image"
        )
        assert figure.page_image_url(SourceLocator(kind="unknown"), "doc_1") == ""
        assert figure.page_image_url(locator, "") == ""

    def test_the_page_image_path_is_the_one_the_evidence_route_serves(self):
        # A link an export advertises and a path the server serves are one
        # contract, and only a test comparing the two notices when they part.
        routes = (PROJECT_ROOT / "paperwhisperer" / "api" / "routes_evidence.py").read_text(
            encoding="utf-8"
        )
        template = PAGE_IMAGE_PATH_TEMPLATE.replace("{document_id}", "{document_id}")

        assert template.strip("/") in routes or "/pages/{page}/image" in routes

    def test_a_figure_element_keeps_its_caption_and_locator(self):
        element = Element(role="figure", text="Figure 1: the encoder stack.",
                          locator=SourceLocator(kind="pdf_page", page=4))

        assert element.figure is not None
        assert element.figure.caption == "Figure 1: the encoder stack."
        assert Element.from_dict(element.to_dict()) == element


class TestContractsGuardTheBoundary:
    def test_the_contract_refuses_a_provenance_nothing_can_produce(self):
        with pytest.raises(ValueError, match="display_source"):
            FormulaContract(latex="x", display_source="certainly")

    def test_the_contract_agrees_with_the_dataclass_about_the_provenance(self):
        for source in FORMULA_DISPLAY_SOURCES:
            formula = Formula(latex="x", display=False, display_source=source)

            assert FormulaContract.from_dataclass(formula).to_dataclass() == formula

    def test_the_contract_refuses_a_dangling_image_uri(self):
        # The dataclass clears it because the value came from this process
        # moments ago; the contract refuses it because the value came from
        # outside and dropping it silently would report no problem at all.
        with pytest.raises(ValueError, match="resolvable scheme"):
            FigureContract(caption="Figure 1", image_uri="images/fig1.png")

    def test_the_contract_accepts_a_real_image_uri(self):
        assert FigureContract(caption="Figure 1", image_uri="artifact://a/b.png").image_uri

    def test_a_formula_role_with_no_latex_is_refused(self):
        # Otherwise the structured formula is invisible to the chunker, the
        # eval runner and the prose projection: present in the payload, absent
        # from the document.
        from pydantic import ValidationError

        from paperwhisperer.ingestion.contracts import ElementContract

        with pytest.raises(ValidationError, match="LaTeX"):
            ElementContract(
                role="formula",
                element_id="el_1",
                locator=source_locator_contract(),
            )


def source_locator_contract():
    from paperwhisperer.ingestion.contracts import SourceLocatorContract

    return SourceLocatorContract(kind="pdf_page", page=1)


# --------------------------------------------------------------------------
# The gate: a verified formula is distinguishable from an invented one
# --------------------------------------------------------------------------


PAPER_LATEX = r"q_i = \exp(z_i/T)"

#: A formula the model invents and the paper never had. A module constant
#: rather than a literal inside the assertions because an f-string may not
#: contain a backslash on the Python 3.10 this project targets.
INVENTED_LATEX = r"\frac{a}{b}"


def document_with_formulas(*pairs, role="formula"):
    """A parsed document carrying one element per ``(latex, display)`` pair."""
    return ParsedDocument(elements=[
        Element(
            role=role,
            text="" if role == "formula" else latex,
            latex=latex if role == "formula" else "",
            locator=SourceLocator(kind="pdf_page", page=index + 1),
            order=index,
            **({} if role != "formula" else {
                "formula": Formula(latex=latex, display=display,
                                   display_source="backend" if display else "backend")
            }),
        )
        for index, (latex, display) in enumerate(pairs)
    ])


def analysis_result(summary: str, document_id: str = "doc_1") -> dict:
    return {
        "document_id": document_id,
        "summary": summary,
        "quotes": "",
        "mindmap": "",
        "paper_facts": {},
        "fact_extraction": {},
    }


class TestTheFormulaGate:
    def test_a_formula_the_model_reproduced_in_another_convention_verifies(self):
        # This is the case a byte-for-byte gate would fail on every single
        # time, and a gate that always fails is a gate readers learn to ignore.
        document = document_with_formulas((PAPER_LATEX, True))
        result = analysis_result(r"核心结论是 $q_{i} = \exp\left(z_{i}/T\right)$。")

        entries = build_formula_index(result, document=document)
        verified = [entry for entry in entries if entry["presentation"] == "paper_formula"]

        assert verified, "a faithful restatement in another convention did not verify"
        assert verified[0]["status"] == "verified"
        assert verified[0]["verdict"] in {"identical", "equivalent"}

    def test_a_formula_the_model_invented_does_not_verify(self):
        document = document_with_formulas((PAPER_LATEX, True))
        result = analysis_result(r"另外我们还有 $$\frac{a}{b}$$ 这样的结论。")

        entries = build_formula_index(result, document=document)
        invented = [entry for entry in entries if entry["presentation"] == "model_formula"]

        assert invented, "an invented formula was presented as the paper's"
        assert invented[0]["status"] == "unverified"
        assert invented[0]["notes"]

    def test_a_verified_entry_carries_the_papers_latex_not_the_models(self):
        # The paper's characters are the ones worth showing, exactly as the
        # citation index swaps a verified quote in for a fact's own wording.
        document = document_with_formulas((PAPER_LATEX, True))
        result = analysis_result(r"见 $q_{i} = \exp\left(z_{i}/T\right)$")

        entry = next(
            item for item in build_formula_index(result, document=document)
            if item["presentation"] == "paper_formula"
        )

        assert entry["latex"] == PAPER_LATEX

    def test_no_document_means_nothing_verifies(self):
        # Absence of a document is not a reason to be optimistic. A claim that
        # was never checked is unverified, and the gate asks whether it may be
        # *rendered* as checked.
        result = analysis_result(r"见 $q_i = \exp(z_i/T)$")

        entries = build_formula_index(result, document=None)

        assert entries
        assert all(entry["presentation"] == "model_formula" for entry in entries)

    def test_a_paper_formula_the_model_never_used_is_still_exported(self):
        # The section is an inventory of the paper's mathematics, not a log of
        # what the model repeated. The equation the model ignored is the one a
        # reader cannot check for themselves.
        document = document_with_formulas((PAPER_LATEX, True))
        result = analysis_result("一个没有任何公式的摘要。")

        entries = build_formula_index(result, document=document)

        assert [entry["status"] for entry in entries] == ["source_only"]
        assert entries[0]["presentation"] == "paper_formula"
        assert entries[0]["mentioned"] is False

    def test_a_malformed_model_formula_is_exported_with_its_defect(self):
        document = document_with_formulas((PAPER_LATEX, True))
        result = analysis_result(r"我们认为 $\frac{a}{b 结论成立。")

        entries = build_formula_index(result, document=document)
        invented = [entry for entry in entries if entry["presentation"] == "model_formula"]

        assert invented, "a malformed model formula was dropped instead of flagged"
        assert invented[0]["defects"]

    def test_the_display_summary_reports_the_guess_as_a_guess(self):
        """The honest table. A block/inline table with no third row would
        report "all display" for a document nobody distinguished."""
        document = ParsedDocument(elements=[
            Element(role="formula", latex="a = b", locator=SourceLocator(kind="pdf_page", page=1)),
            Element(role="formula", latex="c = d", locator=SourceLocator(kind="pdf_page", page=1)),
        ])
        result = analysis_result("no maths here")

        summary = build_formula_display_summary(build_formula_index(result, document=document))

        assert summary["unknown"] == 2
        assert summary["display"] == 0
        assert summary["inline"] == 0


# --------------------------------------------------------------------------
# Rendering: one LaTeX string, three formats, one gate
# --------------------------------------------------------------------------


def render_all(result: dict) -> tuple[str, str, str]:
    """Markdown, HTML and the DOCX document XML, from the same result."""
    markdown = render_markdown_report(result, "paper.pdf", "0.9.0")
    html = render_html_report(result, "paper.pdf", "0.9.0")
    payload = render_docx_report(result, "paper.pdf", "0.9.0")
    document_xml = zipfile.ZipFile(io.BytesIO(payload)).read("word/document.xml").decode("utf-8")
    return markdown, html, document_xml


def result_with_one_verified_and_one_invented():
    document = document_with_formulas((PAPER_LATEX, True))
    result = analysis_result(r"论文给出 $q_{i} = \exp\left(z_{i}/T\right)$，而 $$\frac{a}{b}$$ 是我们的补充。")
    service.attach_formula_index(result, document=document)
    return result


class TestRendering:
    def test_the_same_latex_string_appears_in_all_three_formats(self):
        """A reader who copies from the Markdown and compiles it must get the
        expression the paper had, so the three exports cannot disagree about
        what the formula was."""
        markdown, html, document_xml = render_all(result_with_one_verified_and_one_invented())

        # The DOCX escapes a backslash as itself in XML text; the LaTeX itself
        # carries no markup, so a substring search is the right assertion.
        assert PAPER_LATEX in markdown
        assert PAPER_LATEX in html
        assert PAPER_LATEX in document_xml
        assert INVENTED_LATEX in markdown
        assert INVENTED_LATEX in html
        assert INVENTED_LATEX in document_xml

    def test_a_verified_formula_is_typeset_and_an_invented_one_is_not(self):
        markdown, html, document_xml = render_all(result_with_one_verified_and_one_invented())

        # Markdown: the delimiters wrap the paper's formula and never the
        # model's. The model's own sentence is rendered as inline code, so the
        # invented formula cannot be typeset by the summary either.
        assert f"$${PAPER_LATEX}$$" in markdown
        assert f"$${INVENTED_LATEX}$$" not in markdown
        assert f"`{INVENTED_LATEX}`" in markdown
        # HTML: the element type differs, and the attribute survives a
        # stylesheet that never loads.
        assert 'data-presentation="paper_formula"' in html
        assert 'data-presentation="model_formula"' in html
        assert "<code class=\"pw-math-unverified\">" in html
        assert "pw-math-unverified" in html
        # DOCX: the Quote style is on the verified branch and not the other.
        assert document_xml.count("IntenseQuote") == 1, (
            "the Quote style reached a formula that was not verified, or failed "
            "to reach one that was"
        )

    def test_the_gate_is_structural_and_not_a_convention(self):
        """The same assertion ``tests/test_export.py`` makes for quotation marks.

        Read out of the source rather than out of the output: a test asserting
        "the output contains no ``$$``" would pass for a file that happened to
        have no verified formula in it, and would keep passing if the marker
        moved to the wrong branch of the wrong function.
        """
        source = (PROJECT_ROOT / "paperwhisperer" / "analysis" / "service.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)

        for writer in ("_formula_body_markdown", "_formula_body_html", "_formula_body_docx"):
            body = _function_body(source, tree, writer)
            assert '"$$"' in body or "_FORMULA_DISPLAY_DELIMITER" in body, writer
            assert "_FORMULA_INLINE_DELIMITER" in body, writer

        # Every call site of a delimiter-bearing writer sits on the arm of an
        # ``if entry["presentation"] == "paper_formula"`` that the condition
        # selects, so an unverified formula has no path to one. Read off the
        # tree rather than out of the text: a regex over the body would also
        # match the branch that is *not* taken.
        for renderer, writer in (
            ("render_formula_index_markdown", "_formula_body_markdown"),
            ("render_formula_index_html", "_formula_body_html"),
            ("render_formula_index_docx", "_formula_body_docx"),
        ):
            guarded, unguarded = _branch_containing_call(source, tree, renderer, writer)
            assert guarded, f"{renderer} never calls {writer} on the paper_formula branch"
            assert not unguarded, (
                f"{renderer} calls {writer} outside the paper_formula branch, so "
                "an unverified formula could acquire a maths delimiter"
            )

    def test_an_unverified_formula_never_reaches_a_maths_delimiter(self):
        document = document_with_formulas((PAPER_LATEX, True))
        result = analysis_result(r"只有模型写的 $$\frac{a}{b}$$，论文里没有。")
        service.attach_formula_index(result, document=document)
        markdown, html, _ = render_all(result)

        # The document's own formula still typesets -- it is a paper formula.
        # What must not happen is the *invented* one acquiring delimiters, in
        # the formula section or in the model's own prose.
        assert f"$${PAPER_LATEX}$$" in markdown
        assert f"$${INVENTED_LATEX}$$" not in markdown
        assert f"`{INVENTED_LATEX}`" in markdown, (
            "the invented formula was dropped from the report instead of shown "
            "as code; a formula that silently disappears is a corruption"
        )
        assert 'data-presentation="paper_formula"' in html
        assert 'data-presentation="model_formula"' in html

    def test_the_section_is_absent_rather_than_empty_without_formulas(self):
        markdown, html, document_xml = render_all(analysis_result("没有任何公式的一篇总结。"))

        assert "## 公式与回源校验" not in markdown
        assert "公式与回源校验" not in html
        assert "公式与回源校验" not in document_xml

    def test_a_result_without_a_formula_index_is_still_exportable(self):
        # The legacy six-section path predates the key entirely, and a renderer
        # that raised on its absence would make a document analysed last month
        # un-exportable today.
        result = {"document_id": "doc_1", "summary": "s", "quotes": "q", "mindmap": "m"}

        assert render_markdown_report(result, "p.pdf", "0.9.0")

    def test_the_export_carries_the_display_inventory_with_its_guess_row(self):
        markdown, _, _ = render_all(result_with_one_verified_and_one_invented())

        assert "来源未区分" in markdown
        assert "论文原式" in markdown

    def test_labels_cover_both_presentations_and_neither_is_empty(self):
        for presentation in ("paper_formula", "model_formula"):
            label = EXPORT_FORMULA_PRESENTATION_LABELS[presentation]
            assert label.strip()
            assert "未" in label or "已" in label

    def test_the_malformed_one_is_still_in_the_file_with_its_reason(self):
        document = document_with_formulas((PAPER_LATEX, True))
        result = analysis_result(r"我们认为 $\frac{a}{b 结论成立。")
        service.attach_formula_index(result, document=document)
        markdown, _, _ = render_all(result)

        assert r"\frac{a}{b" in markdown
        assert "花括号没有配对" in markdown

    def test_a_figure_is_rendered_from_the_document_not_the_model(self):
        document = ParsedDocument(elements=[
            Element(role="figure", text="Figure 1: the encoder stack.",
                    locator=SourceLocator(kind="pdf_page", page=4)),
        ])
        result = analysis_result("摘要里提到了 Figure 1。")
        service.attach_formula_index(result, document=document)

        entries = build_formula_index(result, document=document)

        # A figure is not a formula, and the formula section must not claim one.
        assert entries == []

    def test_the_rendered_markdown_section_is_a_function_of_the_index(self):
        result = result_with_one_verified_and_one_invented()
        entries = build_formula_index(result)

        assert render_formula_index_markdown(entries)
        assert render_formula_index_markdown([]) == ""


def _function_body(source: str, tree: ast.Module, name: str) -> str:
    """The source of ``def name(...)``, signature through end.

    Extracted with :mod:`ast` rather than by counting braces, for the reason
    ``tests/test_export.py`` gives: an f-string delimiter is a quote to a
    brace-matcher and a token to a parser, and a scanner that mis-pairs the two
    returns the middle of a *different* function -- so a structural assertion
    over such a slice is not flaky, it is an assertion about code that is not
    there.
    """
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            lines = source.splitlines(keepends=True)
            return "".join(lines[node.lineno - 1 : node.end_lineno])
    raise AssertionError(f"function {name}() is missing")


# --------------------------------------------------------------------------
# The citation index is unaffected
# --------------------------------------------------------------------------


def test_the_quotation_mark_gate_still_holds_alongside_the_formula_one():
    """Two gates, one report, neither able to borrow the other's mark.

    A formula rendered as a verified quote would type a paper's mathematics as
    a quotation, and a quotation rendered as a verified formula would claim a
    sentence is an equation. They are separate branches in separate functions
    and this asserts the two symbols do not appear in each other's.
    """
    result = result_with_one_verified_and_one_invented()
    entries = build_citation_index(result)
    markdown = render_markdown_report(result, "paper.pdf", "0.9.0")

    # build_citation_index saw no facts and so produced nothing; the two
    # indexes are independent by construction.
    assert entries == []
    assert "## 引用与证据链接" not in markdown
    assert "公式与回源校验" in markdown


def _branch_containing_call(
    source: str, tree: ast.Module, function_name: str, call_name: str
) -> tuple[bool, bool]:
    """Whether ``call_name`` sits in the taken arm of the gate, and elsewhere.

    Returns ``(on_paper_formula_branch, on_any_other_branch)``. Walking the
    tree rather than scanning the text is the whole point: a textual search for
    the call finds it wherever it is, including on the arm a ``model_formula``
    takes, and the assertion that matters is about *which arm* it is in.

    The gate is identified structurally — a comparison of ``entry["presentation"]``
    against ``"paper_formula"`` — rather than by a comment saying so, so
    rewriting the comment cannot make this test pass.
    """
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
    )
    taken = False
    elsewhere = False
    for node in ast.walk(function):
        if not isinstance(node, ast.If):
            continue
        is_gate = (
            isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Subscript)
            and "presentation" in ast.dump(node.test.left)
            and any(
                isinstance(part, ast.Constant) and part.value == "paper_formula"
                for part in node.test.comparators
            )
        )
        on_taken_arm = is_gate and any(
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Name)
            and inner.func.id == call_name
            for inner in ast.walk(node.body[0])
        )
        on_else_arm = is_gate and any(
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Name)
            and inner.func.id == call_name
            for arm in node.orelse
            for inner in ast.walk(arm)
        )
        taken = taken or on_taken_arm
        elsewhere = elsewhere or on_else_arm
    # Anything called outside every gated if-statement counts as unguarded.
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == call_name
        ):
            inside_an_if = any(
                isinstance(inner, ast.If) and node in list(ast.walk(inner)) for inner in ast.walk(function)
            )
            if not inside_an_if:
                elsewhere = True
    return taken, elsewhere


# --------------------------------------------------------------------------
# The parser's own limitation is not the model's fault
# --------------------------------------------------------------------------


#: The exact string docling 2.131.0 returns for a formula whose page shows
#: ``q_i = \exp(z_i/T) \sum_j \exp(z_j/T)``. Measured on
#: ``eval/corpus/digital_single_col/hinton_distillation.pdf``: six formulas
#: survive, and this is one of them. There is no ``_`` in it anywhere, because
#: a PDF text layer has no notion of a subscript -- the glyph arrives smaller
#: and lower than the ``q``, and by the time it is text the structure is gone.
DOCLING_TRANSLITERATED = "qi = exp(zi/T) \\sum j exp(zj/T) (1)"


class TestFlattenedSourceStructure:
    """A parser that cannot read subscripts must not convict the model of lying.

    This is the measured failure, not a hypothetical one. The formula above is
    what the real backend produces; the model writing ``q_i`` has *recovered*
    the structure the parser dropped. Reporting that as an invented formula
    would make every formula in every docling-parsed paper look fabricated, and
    the release gate would be the thing that taught readers to distrust it.
    """

    def test_a_transliterated_formula_is_recognised_as_flattened(self):
        formula = Element(role="formula", latex=DOCLING_TRANSLITERATED).formula

        assert formula.latex_source == "flat"
        assert formula.is_structure_flattened is True

    def test_a_formula_with_a_script_is_not(self):
        formula = Element(role="formula", latex=r"q_i = \exp(z_i/T)").formula

        assert formula.latex_source == "structured"
        assert formula.is_structure_flattened is False

    def test_the_answer_follows_the_latex_rather_than_a_stored_claim(self):
        """Derived, not stored, so it cannot go stale.

        A stored value describes the LaTeX as it was when somebody decided to
        write it; this describes the LaTeX actually in the element. The
        producers that would be expected to write the field are precisely the
        ones whose output proves they cannot know it.
        """
        formula = Formula(latex="a = b")

        assert formula.latex_source == "flat"
        assert formula.is_structure_flattened is True

    def test_a_correct_restatement_is_not_reported_as_invented(self):
        document = ParsedDocument(elements=[
            Element(role="formula", latex=DOCLING_TRANSLITERATED,
                    locator=SourceLocator(kind="pdf_page", page=2)),
        ])
        result = analysis_result(r"论文给出 $q_i = \exp(z_i/T) \sum_j \exp(z_j/T)$。")

        entries = build_formula_index(result, document=document)
        model_entries = [e for e in entries if e["presentation"] == "model_formula"]

        assert model_entries, "the model's formula was dropped rather than reported"
        assert model_entries[0]["status"] == "uncomparable", (
            "a model formula that merely restored structure the parser dropped "
            "was reported as invented -- the gate blaming the model for the "
            "parser's own limitation"
        )
        assert model_entries[0]["status"] != "unverified"

    def test_the_note_names_the_real_cause_rather_than_guessing_at_an_image(self):
        # The reader's next action differs between the two cases: for a picture
        # the formula is simply unavailable, whereas here the page shows it and
        # only this build's parser lost it, so a reader holding the PDF can
        # check it and this export cannot.
        document = ParsedDocument(elements=[
            Element(role="formula", latex=DOCLING_TRANSLITERATED,
                    locator=SourceLocator(kind="pdf_page", page=2)),
        ])
        result = analysis_result(r"$q_i = \exp(z_i/T)$")

        entry = build_formula_index(result, document=document)[0]

        assert "下标" in "".join(entry["notes"])
        assert "图片" not in "".join(entry["notes"])

    def test_a_structured_source_still_calls_an_invented_formula_invented(self):
        # The downgrade is conditional. With an intact source there is nothing
        # to excuse, and widening it to cover every mismatch would defeat the
        # gate it exists to keep.
        document = document_with_formulas((r"q_i = \exp(z_i/T)", True))
        result = analysis_result(r"我们另有 $$\frac{a}{b}$$ 的结论。")

        invented = [
            e for e in build_formula_index(result, document=document)
            if e["presentation"] == "model_formula"
        ]

        assert invented[0]["status"] == "unverified"

    def test_the_downgrade_does_not_make_a_matching_formula_unverified(self):
        document = ParsedDocument(elements=[
            Element(role="formula", latex=DOCLING_TRANSLITERATED,
                    locator=SourceLocator(kind="pdf_page", page=2)),
        ])
        result = analysis_result(r"完全一致的 $qi = exp(zi/T) \sum j exp(zj/T)$。")

        entries = build_formula_index(result, document=document)
        verified = [e for e in entries if e["status"] == "verified"]

        assert verified, "an exact restatement failed to verify"
        assert verified[0]["verdict"] in {"identical", "equivalent"}

    def test_a_figure_image_is_still_not_a_formula(self):
        # The other uncomparable case must not be confused with this one: an
        # image has no LaTeX at all, which is a different absence from a
        # transcription that lost its scripts.
        assert compare_latex("", r"\frac{a}{b}").verdict is LatexVerdict.UNCOMPARABLE
        assert Element(role="formula", latex=DOCLING_TRANSLITERATED).formula.is_well_formed is True
