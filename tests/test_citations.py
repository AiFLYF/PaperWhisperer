"""Citation verifier: alignment, scoping, and the degradation contract.

The module under test exists to make one promise — *a direct quote is the
paper's own words* — and to keep it when verification fails. The tests are
therefore organised around what a caller can and cannot do with a result,
rather than around the functions, because the failure this project is
redesigning to eliminate is not a wrong verdict but a right verdict nobody
was forced to obey: a template that prints a claim in quotation marks has
checked nothing at all.

What is pinned here, and why it is easy to regress:

* **Two modes, and the difference is on the value.** ``EXACT`` means the
  quote is a byte-for-byte substring of the cited element; ``NORMALIZED``
  means it matched after a specific, named set of transformations. The
  transformations are the dangerous part — each one is a chance for a
  citation to point at the wrong thing — so each has a test of its own,
  including the ones that must *not* fire (a hyphen in a list item is not a
  split word; a long-s ligature is not a ligature).
* **Scoping is the whole game.** A quote that exists somewhere else in the
  document is not evidence that it exists in the cited passage, so
  verification never widens past the scope it was given, not even when the
  whole document is right there.
* **Degradation cannot be skipped by accident.** A failed result is
  structurally incapable of being presented as a quote: the presentation is
  derived, the verified text raises, the rendered paraphrase has its
  quotation marks stripped and carries no locator, and a direct-quote render
  without a span cannot be constructed at all.
* **The release gate is executable.** The last class runs the verifier over
  real corpus documents, self-quoting every page, and asserts 100% exact
  alignment. That is the section 7 gate — "direct quote exact/normalized
  alignment 100%" — written as a check rather than as a claim in a report.

Everything here is deterministic: no clock, no network, no model. A test that
could pass on Tuesday and fail on Wednesday would be testing the wrong thing.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from paperwhisperer.ingestion.schema import (
    Element,
    Evidence,
    ParsedDocument,
    SourceLocator,
    Table,
)
from paperwhisperer.rag.citations import (
    NORMALIZATION_RULES,
    AlignmentMode,
    AlignmentOutcome,
    CitationVerification,
    ClaimPresentation,
    MatchedSpan,
    RenderedClaim,
    UnverifiedQuoteError,
    normalization_rule_ids,
    normalize_for_alignment,
    verify_evidence_quote,
    verify_quote,
    verify_quote_in_scope,
)

_CORPUS_ROOT = Path(__file__).resolve().parents[1] / "eval" / "corpus"

#: The corpus is gitignored, so these tests are only meaningful where it has
#: been materialised. Skipping rather than omitting them keeps the release
#: gate visible in a CI report instead of quietly unverified.
requires_corpus = pytest.mark.skipif(
    not _CORPUS_ROOT.is_dir(),
    reason=(
        f"eval corpus is not present at {_CORPUS_ROOT}; it is gitignored, so the "
        "release-gate check only runs where the corpus has been materialised"
    ),
)


# --------------------------------------------------------------------------
# Fixtures as plain builders — the states being tested are schema states that
# no parser has to produce on demand.
# --------------------------------------------------------------------------


def _page_text(
    text: str,
    *,
    page: int = 3,
    role: str = "paragraph",
    order: int = 0,
) -> Element:
    """One element on one page, with a locator a citation can point at."""
    return Element(
        role=role,
        text=text,
        order=order,
        locator=SourceLocator(kind="pdf_page", page=page, element=1),
    )


def _two_page_document() -> ParsedDocument:
    return ParsedDocument(
        source_filename="two-page.pdf",
        media_type="application/pdf",
        page_count=2,
        parser_name="test-parser",
        elements=[
            _page_text("Attention is all you need.", page=3, order=0),
            _page_text("The dominant sequence models are recurrent.", page=7, order=1),
        ],
    )


def _table_element() -> Element:
    return Element(
        role="table",
        locator=SourceLocator(kind="pdf_page", page=4, element=2),
        table=Table(
            rows=[
                ["Model", "Params", "Top-1"],
                ["ResNet-50", "25.6M", "76.1"],
                ["ViT-B/16", "86M", "77.9"],
            ],
            header_rows=1,
            caption="Image classification results",
        ),
    )


# --------------------------------------------------------------------------
# Exact alignment
# --------------------------------------------------------------------------


class TestExactAlignment:
    def test_a_verbatim_quote_aligns_and_says_so(self):
        element = _page_text("Attention is all you need for sequence transduction.")

        result = verify_quote("all you need for sequence", element)

        assert result.aligned is True
        assert result.mode is AlignmentMode.EXACT
        assert result.verified_text == "all you need for sequence"
        assert element.text[result.span.start : result.span.end] == "all you need for sequence"

    def test_the_reported_span_is_character_offsets_into_the_cited_element(self):
        element = _page_text("Attention is all you need for sequence transduction.")

        result = verify_quote("sequence transduction", element)

        assert result.span == MatchedSpan(start=30, end=51)
        assert result.element_id == element.element_id
        assert result.locator == element.locator
        assert result.locator.page == 3

    def test_a_quote_may_be_the_whole_element(self):
        element = _page_text("Attention is all you need.")

        result = verify_quote(element.text, element)

        assert result.aligned is True
        assert result.span == MatchedSpan(start=0, end=len(element.text))
        assert result.verified_text == element.text

    def test_a_quote_that_is_a_substring_of_a_longer_passage_is_still_exact(self):
        # The substring case is the one that matters: refusing short quotes
        # would push honest citations into paraphrase for no reason, and
        # accepting only whole-element quotes would make a claim's length
        # decide whether it is true.
        element = _page_text(
            "The dominant sequence transduction models are based on complex "
            "recurrent or convolutional neural networks."
        )

        result = verify_quote("complex recurrent or convolutional", element)

        assert result.aligned is True
        assert result.mode is AlignmentMode.EXACT
        assert result.span == MatchedSpan(start=55, end=89)

    def test_a_quote_the_element_never_said_does_not_align(self):
        element = _page_text("Attention is all you need for sequence transduction.")

        result = verify_quote("Attention is all you might need", element)

        assert result.aligned is False
        assert result.outcome is AlignmentOutcome.NOT_FOUND
        assert result.span is None

    def test_alignment_survives_a_round_trip_through_the_stored_form(self):
        element = _page_text("Attention is all you need for sequence transduction.")

        result = verify_quote("all you need", element)
        restored = CitationVerification.from_dict(result.to_dict())

        assert restored.to_dict() == result.to_dict()
        assert restored.verified_text == result.verified_text
        assert restored.mode is AlignmentMode.EXACT


# --------------------------------------------------------------------------
# Normalisation, one rule at a time
# --------------------------------------------------------------------------


class TestNormalizationRules:
    def test_every_declared_rule_is_documented_and_reachable(self):
        # A rule with a description and no implementation is documentation
        # that lies, and a rule with an implementation and no description is a
        # transformation nobody can review. Reachability is proved by the rest
        # of this class: every rule below is reported by a real alignment, and
        # a rule nothing reports is a rule that is not doing anything.
        assert normalization_rule_ids() == tuple(
            rule.rule_id for rule in NORMALIZATION_RULES
        )
        assert all(rule.description.strip() for rule in NORMALIZATION_RULES)
        assert len(set(normalization_rule_ids())) == len(NORMALIZATION_RULES)

    def test_whitespace_collapse_aligns_a_quote_that_retyped_the_line_wraps(self):
        element = _page_text("The dominant sequence\n  transduction models are recurrent.")

        result = verify_quote("The dominant sequence transduction models", element)

        assert result.mode is AlignmentMode.NORMALIZED
        assert result.applied_rules == ("whitespace_collapse",)
        # The span stops at the last character the quote covered, so the span
        # carries the wrapped form of exactly the quoted words.
        assert result.verified_text == "The dominant sequence\n  transduction models"

    def test_whitespace_collapse_also_covers_a_quote_the_caller_did_not_trim(self):
        # Models trim; humans paste. Trailing and leading space is invisible,
        # so a quote that carries it is quoting the same characters.
        element = _page_text("Attention is all you need.")

        result = verify_quote("  all you need  ", element)

        assert result.aligned is True
        assert result.applied_rules == ("whitespace_collapse",)

    def test_line_break_hyphen_joins_a_word_the_layout_split(self):
        element = _page_text("A trans-\n   formation model is deterministic.")

        result = verify_quote("transformation model", element)

        assert result.mode is AlignmentMode.NORMALIZED
        assert result.applied_rules == ("line_break_hyphen",)
        # The span covers the hyphen and the break that normalisation removed,
        # because the reader has to see the whole word highlighted.
        assert result.verified_text == "trans-\n   formation model"

    def test_line_break_hyphen_does_not_join_a_list_item(self):
        # A hyphen between a bullet and a word is not a split word. Joining it
        # would invent a word that is not on the page and let a quote verify
        # against text nobody can see.
        element = _page_text("Method - second item\nthird item", role="list_item")

        result = verify_quote("Methodsecond item", element)

        assert result.aligned is False

    def test_the_line_wrap_hyphen_limitation_is_measured_not_assumed(self):
        # The documented cost of the de-hyphenation rule. The source holds one
        # word the layout split; the reader holds either half of the two
        # possible readings of it, and only one of them is a substring of the
        # de-hyphenated source.
        split_word = _page_text("a trans-\n   formation model")
        compound = _page_text("the English-\nto-German task")

        # The reading the rule was written for: the hyphen existed only to end
        # the line, and the word is whole again without it.
        assert verify_quote("transformation model", split_word).applied_rules == (
            "line_break_hyphen",
        )
        # The retyping that fails: a reader who copies the line break as a
        # space. The obvious fix is to delete a space after a hyphen, and that
        # is refused — a space is visible, and a rule that deletes visible
        # characters to make a citation succeed is a rule that can point at
        # the wrong thing ("state of the-art" folding onto
        # "state-of-the-art"). The quote degrades instead, which is a claim
        # the reader can see rather than a citation they cannot check.
        assert verify_quote("trans- formation model", split_word).aligned is False
        # The reading the rule does not serve: a compound whose hyphen is real,
        # so joining the halves removes a character the page shows.
        assert verify_quote("the English-to-German task", compound).aligned is False
        for quote in ("trans- formation model", "the English-to-German task"):
            assert verify_quote(quote, split_word).render().uses_quotation_marks is False
        assert verify_quote(
            "the English-to-German task", compound
        ).render().uses_quotation_marks is False

    def test_soft_hyphen_is_dropped(self):
        element = _page_text("An inter\u00adnational standard body met.")

        result = verify_quote("international standard", element)

        assert result.applied_rules == ("soft_hyphen",)
        assert result.verified_text == "inter\u00adnational standard"

    def test_fullwidth_ascii_folds_to_the_characters_it_names(self):
        # A CJK PDF text layer emits full-width forms for keys the reader never
        # pressed; the sentence is the ASCII one.
        element = _page_text("本文使用 １２３ 个数据集进行评估。")

        result = verify_quote("本文使用 123 个数据集", element)

        assert result.applied_rules == ("fullwidth_ascii",)
        assert result.mode is AlignmentMode.NORMALIZED

    def test_ideographic_space_is_whitespace(self):
        element = _page_text("\u672c\u6587\u3000\u63d0\u51fa\u3000\u4e00\u79cd\u65b9\u6cd5\u3002")

        result = verify_quote("本文 提出 一种方法", element)

        assert result.aligned is True
        assert result.applied_rules == ("whitespace_collapse",)

    def test_quote_marks_fold_typographic_variants(self):
        element = _page_text("The author writes “attention is all you need”.")

        result = verify_quote('The author writes "attention is all you need"', element)

        assert result.applied_rules == ("quote_marks",)
        assert result.aligned is True

    def test_quote_marks_keep_single_and_double_forms_apart(self):
        # 'x' and ‘x’ are not the same quotation, and a quote that swapped its
        # nesting has changed what it claims the author was quoting.
        element = _page_text("The term ‘attention’ is used throughout.")

        result = verify_quote("The term \"attention\" is used throughout.", element)

        assert result.aligned is False

    def test_ligature_expands_so_a_typed_word_matches_a_typographic_one(self):
        # The real corpus emits these: word2vec's title on page 1 of the PDF
        # reads "Efﬁcient" in the text layer and "Efficient" to every reader.
        element = _page_text("Ef\ufb01cient Estimation of Word Representations.")

        result = verify_quote("Efficient Estimation", element)

        assert result.applied_rules == ("ligature",)
        assert result.verified_text == "Ef\ufb01cient Estimation"

    def test_the_long_s_ligature_is_not_folded(self):
        # U+FB05 is a letter change, not a typographic shortcut. Folding it
        # would make "past" and "paſt" verify against each other.
        element = _page_text("the pa\ufb05t tense")

        assert verify_quote("past tense", element).aligned is False
        assert verify_quote("pa\ufb05t tense", element).mode is AlignmentMode.EXACT

    def test_several_rules_can_fire_on_one_quote(self):
        element = _page_text("\u672c\u6587\u4f7f\u7528 \uff21t\u00adtention\n  \u6a21\u578b\u3002")

        result = verify_quote("本文使用 Atten­tion 模型", element)

        assert result.aligned is True
        assert result.mode is AlignmentMode.NORMALIZED
        assert set(result.applied_rules) == {
            "fullwidth_ascii",
            "soft_hyphen",
            "whitespace_collapse",
        }

    def test_case_is_never_folded(self):
        # The obvious "just make it case-insensitive" is refused: a capital
        # letter is visible, it is how a paper introduces a term, and folding
        # it would let a quote misreport the source's own emphasis.
        element = _page_text("The Dominant Sequence models are recurrent.")

        assert verify_quote("the dominant sequence models", element).aligned is False

    def test_punctuation_is_never_stripped(self):
        element = _page_text("Results: 76.1% top-1 accuracy.")

        assert verify_quote("Results 76.1 top-1 accuracy", element).aligned is False

    def test_no_rule_is_reported_when_none_fired(self):
        # A verdict that names rules it did not use is worse than one that
        # names none: it teaches a reader to distrust the field.
        element = _page_text("Attention is all you need.")

        result = verify_quote("Attention is all you need", element)

        assert result.mode is AlignmentMode.EXACT
        assert result.applied_rules == ()


class TestNormalizeForAlignment:
    def test_it_keeps_a_map_back_to_the_original(self):
        normalized = normalize_for_alignment("a  b")

        assert normalized.text == "a b"
        # The second space is gone from the comparison but still in the source,
        # and the offsets are the only thing that can prove where the match
        # will be reported.
        assert normalized.offsets == (0, 1, 3)
        assert "".join("a  b"[index] for index in normalized.offsets) == "a b"

    def test_a_deleted_character_leaves_no_entry_behind_but_still_reports(self):
        # Trailing whitespace is removed by deleting it, so no character
        # carries the rule's provenance — the rule still has to be reported,
        # or a quote aligned only because both sides were trimmed would claim
        # it used no normalisation.
        normalized = normalize_for_alignment("a ")

        assert normalized.text == "a"
        assert normalized.applied_rules == ("whitespace_collapse",)

    def test_it_is_total(self):
        for text in ("", "   ", "\n\t", "", "a" * 5000):
            normalized = normalize_for_alignment(text)
            assert isinstance(normalized.text, str)

    def test_a_span_outside_the_normalised_text_is_refused(self):
        normalized = normalize_for_alignment("abc")

        with pytest.raises(ValueError):
            normalized.map_span(0, 99)


# --------------------------------------------------------------------------
# Scoping
# --------------------------------------------------------------------------


class TestScopeEnforcement:
    def test_a_quote_from_another_element_does_not_align(self):
        cited = _page_text("Attention is all you need.", order=0)
        other = _page_text("The dominant sequence models are recurrent.", order=1)

        result = verify_quote(other.text, cited)

        assert result.aligned is False
        assert result.outcome is AlignmentOutcome.NOT_FOUND

    def test_a_quote_that_exists_elsewhere_in_the_document_does_not_rescue_this_citation(self):
        # The failure this module exists to prevent: a fluent sentence the
        # model attributed to page 3 that turns out to be on page 7. Searching
        # the document would turn a wrong citation into a wrong-looking right
        # one, and nothing downstream could tell the difference.
        cited = _page_text("Attention is all you need.", page=3, order=0)
        elsewhere = _page_text("Attention is all you need, and then some.", page=7, order=1)
        document = ParsedDocument(page_count=7, elements=[cited, elsewhere])

        assert verify_quote(elsewhere.text, cited).aligned is False
        assert verify_quote(elsewhere.text, document.elements[0]).aligned is False
        assert verify_quote_in_scope(elsewhere.text, [cited]).aligned is False

    def test_a_scope_of_several_elements_may_be_widened_on_purpose(self):
        first = _page_text("Attention is all you need.", page=3, order=0)
        second = _page_text("The dominant sequence models are recurrent.", page=3, order=1)

        result = verify_quote_in_scope("dominant sequence models", [first, second])

        assert result.aligned is True
        assert result.element_id == second.element_id
        assert result.locator == second.locator

    def test_the_element_that_matched_is_reported_not_the_one_that_was_asked_for(self):
        first = _page_text("Attention is all you need.", page=3, order=0)
        second = _page_text("The dominant sequence models are recurrent.", page=3, order=1)

        result = verify_quote_in_scope("Attention is all you need", [first, second])

        assert result.element_id == first.element_id
        assert result.locator.page == 3

    def test_a_blank_quote_is_refused_before_anything_is_searched(self):
        # An empty string is a substring of everything, so a verifier that
        # tried it would report success for every element in the document.
        element = _page_text("Attention is all you need.")

        for quote in ("", "   ", "\n\t"):
            result = verify_quote(quote, element)
            assert result.outcome is AlignmentOutcome.EMPTY_QUOTE
            assert result.aligned is False

    def test_an_empty_scope_is_no_scope(self):
        result = verify_quote_in_scope("Attention is all you need", [])

        assert result.outcome is AlignmentOutcome.NO_SCOPE
        assert result.aligned is False

    def test_an_element_with_no_text_is_no_scope(self):
        result = verify_quote("Attention is all you need", _page_text("   "))

        assert result.outcome is AlignmentOutcome.NO_SCOPE

    def test_scope_order_does_not_depend_on_the_order_the_caller_passed(self):
        first = _page_text("Attention is all you need.", page=3, order=0)
        second = _page_text("The dominant sequence models are recurrent.", page=3, order=1)

        forward = verify_quote_in_scope("is all you need", [first, second])
        backward = verify_quote_in_scope("is all you need", [second, first])

        assert forward.element_id == backward.element_id == first.element_id
        assert forward.span == backward.span


# --------------------------------------------------------------------------
# Repetition
# --------------------------------------------------------------------------


class TestRepeatedOccurrences:
    def test_the_first_occurrence_wins_and_the_count_is_reported(self):
        element = _page_text("we study X. we study X again, and again we study X.")

        result = verify_quote("we study X", element)

        assert result.aligned is True
        assert result.span == MatchedSpan(start=0, end=10)
        assert result.occurrence_count == 3

    def test_overlapping_occurrences_are_counted_as_occurrences(self):
        # "aa" occurs four times in "aaaaa" by the only definition a reader
        # would use. ``str.count`` says two, because it advances past a whole
        # match; a citation whose count cannot be checked is not a count.
        element = _page_text("aaaaa")

        assert verify_quote("aa", element).occurrence_count == 4

    def test_repetition_is_not_ambiguity(self):
        # Every occurrence is a real occurrence, and the one being cited is a
        # real one. Refusing here would be refusing a true statement, and the
        # quote does exist in the cited element.
        element = _page_text("we study X. we study X again.")

        result = verify_quote("we study X", element)

        assert result.outcome is AlignmentOutcome.ALIGNED
        assert result.quote_marks_allowed is True

    def test_a_repeated_quote_across_a_scope_resolves_to_the_first_in_reading_order(self):
        first = _page_text("we study X. we study X.", page=3, order=0)
        second = _page_text("we study X on another page too.", page=9, order=1)

        result = verify_quote_in_scope("we study X", [second, first])

        assert result.element_id == first.element_id
        assert result.locator.page == 3
        assert result.occurrence_count == 3
        assert [item["locator"]["page"] for item in result.occurrence_locators] == [9]

    def test_the_occurrences_disclosed_are_capped(self):
        elements = [
            _page_text(f"we study X on page {page}.", page=page, order=index)
            for index, page in enumerate((1, 2, 3, 4, 5, 6, 7, 8))
        ]

        result = verify_quote_in_scope("we study X", elements)

        assert result.occurrence_count == 8
        assert len(result.occurrence_locators) == 4


# --------------------------------------------------------------------------
# Element boundaries
# --------------------------------------------------------------------------


class TestElementBoundary:
    def test_a_quote_running_across_two_elements_is_refused_with_its_own_reason(self):
        first = _page_text("Attention is all you need.", page=3, order=0)
        second = _page_text("The dominant sequence models are recurrent.", page=3, order=1)

        result = verify_quote_in_scope("you need. The dominant sequence", [first, second])

        assert result.aligned is False
        assert result.outcome is AlignmentOutcome.SPANS_ELEMENTS
        assert set(result.spanned_element_ids) == {first.element_id, second.element_id}

    def test_refusing_a_boundary_quote_is_not_the_same_as_finding_nothing(self):
        # The two failures mean different things to whoever has to fix
        # something: one is a hallucination to regenerate, the other is a
        # chunk boundary the model cannot see. Collapsing them would lose the
        # only diagnostic the caller has.
        first = _page_text("Attention is all you need.", page=3, order=0)
        second = _page_text("The dominant sequence models are recurrent.", page=3, order=1)

        spanning = verify_quote_in_scope("you need. The dominant sequence", [first, second])
        absent = verify_quote_in_scope("The transformer remembers nothing", [first, second])

        assert spanning.outcome is AlignmentOutcome.SPANS_ELEMENTS
        assert absent.outcome is AlignmentOutcome.NOT_FOUND
        assert spanning.message != absent.message

    def test_a_quote_inside_the_second_element_still_aligns_to_it(self):
        first = _page_text("Attention is all you need.", page=3, order=0)
        second = _page_text("The dominant sequence models are recurrent.", page=3, order=1)

        result = verify_quote_in_scope("dominant sequence", [first, second])

        assert result.aligned is True
        assert result.element_id == second.element_id

    def test_a_cross_element_quote_within_one_element_is_just_a_quote(self):
        # The same characters, in one element, are ordinary text. A verifier
        # that reported SPANS_ELEMENTS here would be describing the scope
        # rather than the string.
        element = _page_text("Attention is all you need. The model is recurrent.")

        assert verify_quote("you need. The model", element).mode is AlignmentMode.EXACT


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------


class TestTableCells:
    def test_a_cell_value_aligns_and_names_its_cell(self):
        result = verify_quote("76.1", _table_element())

        assert result.aligned is True
        assert result.matched_cell == "R2C3"
        assert result.matched_cell_range == "R2C3"

    def test_a_quote_running_across_cells_reports_the_range_it_covers(self):
        # A row is genuine text of the table as it is rendered, so refusing it
        # would be wrong; claiming it came from one cell would also be wrong.
        result = verify_quote("ResNet-50 | 25.6M | 76.1", _table_element())

        assert result.aligned is True
        assert result.matched_cell == ""
        assert result.matched_cell_range == "R2C1:R2C3"

    def test_a_cell_belonging_to_another_table_does_not_align(self):
        other = Element(role="table", table=Table(rows=[["BLEU", "20.1"]]))

        assert verify_quote("76.1", other).aligned is False

    def test_a_value_from_another_table_is_not_borrowed(self):
        # A number repeated across a results table is the classic way a
        # citation ends up one row off from the row it claims. The two tables
        # share a header and a column, and nothing but the numbers.
        elsewhere = Element(
            role="table",
            table=Table(rows=[["Model", "Top-1"], ["ViT-L/16", "79.9"]]),
        )

        assert verify_quote("76.1", elsewhere).aligned is False
        assert verify_quote("76.1", _table_element()).aligned is True

    def test_a_match_that_spills_out_of_its_cell_names_no_cell(self):
        # "accuracy |" is one cell's tail plus the separator that renders after
        # it. The table does not contain it as one cell, and naming that cell
        # would be a claim about a value the source never wrote as one cell.
        table = Element(role="table", table=Table(rows=[["Top-1 accuracy", "76.1"]]))

        result = verify_quote("accuracy |", table)

        assert result.aligned is True
        assert result.matched_cell == ""
        assert result.matched_cell_range == "R1C1"

    def test_a_cell_containing_a_pipe_is_quoted_as_it_is_rendered(self):
        # A pipe inside a cell is escaped in the markdown, so the rendered text
        # is not the cell's own text. The verbatim cell value therefore does
        # not verify, and the caller's only honest options are to quote the
        # rendered form or to present a paraphrase. Both are better than
        # un-escaping the text to make a quote succeed.
        table = Element(role="table", table=Table(rows=[["A|B", "c"]]))

        assert verify_quote("A|B", table).aligned is False
        assert verify_quote("A\\|B", table).mode is AlignmentMode.EXACT
        assert verify_quote("A\\|B", table).matched_cell == "R1C1"


# --------------------------------------------------------------------------
# Formulas
# --------------------------------------------------------------------------


class TestFormulaLatex:
    def test_the_latex_body_is_a_verbatim_substring_of_the_rendered_formula(self):
        element = Element(role="formula", latex="E = mc^2", locator=SourceLocator(kind="pdf_page", page=5))

        result = verify_quote("E = mc^2", element)

        assert element.text == "$$E = mc^2$$"
        assert result.mode is AlignmentMode.EXACT
        assert result.verified_text == "E = mc^2"

    def test_the_delimited_form_quotes_too(self):
        element = Element(role="formula", latex="E = mc^2")

        assert verify_quote("$$E = mc^2$$", element).mode is AlignmentMode.EXACT

    def test_tex_spacing_is_not_interpreted(self):
        # \quad and a space are different formulas, and a gate that says they
        # are the same has stopped comparing characters and started guessing.
        element = Element(role="formula", latex="x = 1 \\quad y = 2")

        assert verify_quote("x = 1 \\quad y = 2", element).mode is AlignmentMode.EXACT
        assert verify_quote("x = 1 \\quad y=2", element).aligned is False

    def test_tex_grouping_is_not_interpreted(self):
        element = Element(role="formula", latex="a_{i} + b_{i}")

        assert verify_quote("a_{i} + b_{i}", element).mode is AlignmentMode.EXACT
        assert verify_quote("a_i + b_i", element).aligned is False
        assert verify_quote("ai + bi", element).aligned is False

    def test_a_formula_does_not_become_prose(self):
        # \frac{a}{b} rendered as "frac a b" is what a reader sees on no page.
        element = Element(role="formula", latex="\\frac{a}{b}")

        assert verify_quote("frac a b", element).aligned is False

    def test_a_formula_quoted_against_prose_does_not_align(self):
        prose = _page_text("We assume the model follows Equation 3 below.", page=5)
        formula = Element(role="formula", latex="E = mc^2", locator=SourceLocator(kind="pdf_page", page=5))

        assert verify_quote("E = mc^2", prose).aligned is False
        assert verify_quote("We assume the model", formula).aligned is False


# --------------------------------------------------------------------------
# The degradation contract
# --------------------------------------------------------------------------


class TestDegradationContract:
    def test_every_outcome_has_a_declared_presentation(self):
        # The presentation lookup has no fallback, so an outcome nobody
        # classified fails loudly instead of rendering as a paraphrase by
        # accident. This walks the whole enum so adding an outcome without
        # deciding how it may be shown fails here.
        for outcome in AlignmentOutcome:
            result = CitationVerification(quote="q", outcome=outcome)
            if outcome is AlignmentOutcome.ALIGNED:
                assert result.presentation is ClaimPresentation.DIRECT_QUOTE
            else:
                assert result.presentation is ClaimPresentation.PARAPHRASE
                assert result.message

    def test_a_failed_verification_cannot_be_presented_as_a_quote(self):
        result = verify_quote("Attention is all you might need", _page_text("Attention is all you need."))

        assert result.aligned is False
        assert result.presentation is ClaimPresentation.PARAPHRASE
        assert result.quote_marks_allowed is False
        assert result.is_direct_quote is False

    def test_the_verified_text_of_a_failed_check_raises_instead_of_returning_the_quote(self):
        result = verify_quote("a sentence nobody said", _page_text("Attention is all you need."))

        with pytest.raises(UnverifiedQuoteError):
            _ = result.verified_text

    def test_a_degraded_render_drops_the_quotation_marks_the_model_wrote(self):
        # The model wrapping its own wording in quotation marks is the most
        # likely way an unverified claim reaches a reader wearing a quote, so
        # the render strips them rather than trusting the caller to notice.
        result = verify_quote('"Attention is all you might need"', _page_text("Attention is all you need."))

        rendered = result.render()

        assert rendered.presentation is ClaimPresentation.PARAPHRASE
        assert rendered.uses_quotation_marks is False
        assert rendered.text == "Attention is all you might need"
        assert '"' not in rendered.text

    def test_a_degraded_render_carries_no_locator_to_attach_a_citation_chip_to(self):
        element = _page_text("Attention is all you need.", page=3)
        result = verify_quote("a sentence nobody said", element)

        rendered = result.render()

        assert rendered.locator is None
        assert result.locator == element.locator
        assert rendered.to_dict()["locator"] is None

    def test_a_degraded_render_explains_itself_in_the_users_language(self):
        result = verify_quote("a sentence nobody said", _page_text("Attention is all you need."))

        message = result.render().message

        assert message
        assert any("一" <= char <= "鿿" for char in message)

    def test_a_verified_render_carries_the_sources_own_characters(self):
        # The quote is aligned, so what the reader sees inside the quotation
        # marks is the paper's text rather than the model's, even when the two
        # differ in ways normalisation forgives.
        element = _page_text("Attention is all you\nneed.")
        result = verify_quote("Attention is all you need", element)

        rendered = result.render()

        assert rendered.uses_quotation_marks is True
        assert rendered.quote_text == "Attention is all you\nneed"
        assert rendered.span == result.span

    def test_a_direct_quote_render_without_a_span_cannot_be_built(self):
        # The guard is a constructor, not a convention: the state that must not
        # be rendered in quotation marks is a state that cannot be built.
        with pytest.raises(ValueError):
            RenderedClaim(
                text="something the model said",
                presentation=ClaimPresentation.DIRECT_QUOTE,
                outcome=AlignmentOutcome.NOT_FOUND,
            )

    def test_a_paraphrase_cannot_be_built_carrying_a_locator(self):
        with pytest.raises(ValueError):
            RenderedClaim(
                text="something the model said",
                presentation=ClaimPresentation.PARAPHRASE,
                outcome=AlignmentOutcome.NOT_FOUND,
                locator=SourceLocator(kind="pdf_page", page=3),
            )

    def test_the_quote_marks_flag_is_derived_and_cannot_disagree_with_the_verdict(self):
        result = verify_quote("a sentence nobody said", _page_text("Attention is all you need."))
        payload = result.render().to_dict()

        assert payload["uses_quotation_marks"] is False
        assert payload["presentation"] == "paraphrase"
        assert payload["uses_quotation_marks"] is (payload["outcome"] == "aligned")

    def test_evidence_is_only_marked_direct_after_a_pass(self):
        element = _page_text("Attention is all you need.")
        evidence = Evidence(evidence_id="ev_1", text="Attention is all you need.", locator=element.locator)

        marked = verify_quote("Attention is all you need", element).mark_evidence(evidence)
        unmarked = verify_quote("a sentence nobody said", element).mark_evidence(evidence)

        assert marked.is_direct_quote is True
        assert marked.text == "Attention is all you need"
        # A copy, so a later re-check cannot leave the flag standing.
        assert evidence.is_direct_quote is False
        assert unmarked.is_direct_quote is False
        assert unmarked.text == "Attention is all you need."

    def test_the_result_carries_a_verdict_and_reasons_but_no_score(self):
        # The one shape this module is not allowed to take. A similarity number
        # invites a threshold, and a threshold is a place where judgement
        # re-enters a pipeline that promised to remove it.
        payload = verify_quote("Attention is all you need", _page_text("Attention is all you need.")).to_dict()

        assert not any(
            key in payload
            for key in ("score", "confidence", "similarity", "ratio", "grade")
        )
        assert payload["outcome"] == "aligned"
        assert isinstance(payload["message"], str) and payload["message"]


# --------------------------------------------------------------------------
# Evidence-scoped verification, the Q&A entry point
# --------------------------------------------------------------------------


class TestEvidenceScoping:
    def test_a_quote_verifies_against_the_passage_the_evidence_came_from(self):
        document = _two_page_document()
        first = document.elements[0]
        evidence = Evidence(evidence_id="ev_1", text=first.text, locator=first.locator)

        result = verify_evidence_quote("Attention is all you need", evidence, document)

        assert result.aligned is True
        assert result.evidence_id == "ev_1"
        assert result.locator.page == 3

    def test_a_quote_from_a_different_page_does_not_borrow_the_evidence(self):
        document = _two_page_document()
        evidence = Evidence(
            evidence_id="ev_1",
            text=document.elements[0].text,
            locator=document.elements[0].locator,
        )

        result = verify_evidence_quote("The dominant sequence models are recurrent.", evidence, document)

        assert result.outcome is AlignmentOutcome.NOT_FOUND

    def test_an_evidence_without_a_locator_is_scoped_by_what_its_passage_contains(self):
        # A chunk built from elements that inherited ``unknown`` is common
        # enough to need a path, and containment cannot widen the scope by
        # accident: an element that is not in the passage does not appear
        # inside the passage.
        document = _two_page_document()
        evidence = Evidence(
            evidence_id="ev_2",
            text="Attention is all you need.",
            locator=SourceLocator(kind="unknown"),
        )

        assert verify_evidence_quote("Attention is all you need", evidence, document).aligned is True
        assert verify_evidence_quote("dominant sequence", evidence, document).aligned is False

    def test_an_evidence_that_matches_no_passage_is_no_scope(self):
        # "Somewhere in this document" is the scope this module exists to
        # refuse, so a citation that cannot say which passage it came from has
        # nothing to verify against.
        document = _two_page_document()
        evidence = Evidence(evidence_id="ev_3", text="a passage from another paper",
                            locator=SourceLocator(kind="unknown"))

        result = verify_evidence_quote("Attention is all you need", evidence, document)

        assert result.outcome is AlignmentOutcome.NO_SCOPE
        assert result.aligned is False

    def test_a_furniture_only_passage_cannot_verify_a_quote(self):
        document = ParsedDocument(
            page_count=1,
            parser_name="test-parser",
            elements=[_page_text("Attention is all you need.", role="footer", page=3)],
        )
        evidence = Evidence(evidence_id="ev_4", text="", locator=SourceLocator(kind="unknown"))

        assert verify_evidence_quote("Attention is all you need", evidence, document).aligned is False


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


class TestDeterminism:
    def test_the_same_inputs_produce_the_same_verdict_every_time(self):
        element = _page_text("The dominant sequence\n models are recurrent.")

        first = verify_quote("dominant sequence models", element)
        second = verify_quote("dominant sequence models", element)

        assert first.to_dict() == second.to_dict()
        assert first.span == second.span
        assert first.applied_rules == second.applied_rules

    def test_verification_reads_nothing_but_its_arguments(self):
        # No clock, no randomness, no network, no model: an element built here
        # verifies identically in a second process on another machine, which is
        # what "100% deterministic" has to mean for a release gate.
        element = _page_text("Attention is all you need.")
        payload = verify_quote("all you need", element).to_dict()
        payload.pop("element_id")
        payload.pop("locator")

        again = verify_quote("all you need", element).to_dict()
        again.pop("element_id")
        again.pop("locator")

        assert payload == again


# --------------------------------------------------------------------------
# The release gate, over the real corpus
# --------------------------------------------------------------------------

#: A spread of the categories that break different things: a plain single
#: column paper, a two-column one, a table-dense one, a formula-dense one and
#: a Chinese-English mixed one. Five files is a sample, and the report says so;
#: it is here to make the gate executable on every change, not to stand in for
#: the full 50-file run T4.3 promotes to CI.
_GATE_FILES = (
    "digital_single_col/attention_is_all_you_need.pdf",
    "digital_two_col/batchnorm.pdf",
    "table_dense/glue_benchmark.pdf",
    "formula_dense/maldacena_large_n.pdf",
    "mixed_zh_en/bilingual_paper_zh_en.pdf",
)

#: Pages per file. Four is enough to cover title, abstract, body and one
#: table or formula on every document while keeping the check to a few
#: seconds; the full corpus run belongs in the eval runner that T4.3 promotes
#: to CI, and this file's job is to make the gate fail loudly on every change.
_GATE_PAGES = 4

#: Word windows drawn per page, sampled with a fixed stride. Every window
#: would be more thorough and much slower, and the stride is fixed so the
#: sample is the same on every run — a gate whose sample changes with a timing
#: budget is a gate that stops being comparable.
_GATE_STRIDE = 5
_GATE_WINDOW = 14


def _gate_pages(relative_path: str) -> list[tuple[int, str]]:
    """The extracted text of the first pages of one corpus file.

    Read with pypdf directly rather than through a parser adapter: the gate is
    a property of the verifier, and making it depend on a backend would test
    the backend instead. A missing file is a failure, not a skip — the corpus
    is a fixed set and a file that cannot be opened is a finding about it.
    """
    from pypdf import PdfReader

    path = _CORPUS_ROOT / relative_path
    assert path.is_file(), f"corpus file missing: {relative_path}"
    reader = PdfReader(str(path))
    return [
        (number, page.extract_text() or "")
        for number, page in enumerate(reader.pages[:_GATE_PAGES], start=1)
    ]


def _verbatim_windows(text: str, width: int = _GATE_WINDOW) -> list[str]:
    """Word windows sliced out of ``text`` without retyping a single character.

    The point of the gate is that a quote the source genuinely contains is
    recognised. Rebuilding a window with ``" ".join(words)`` would normalise
    whitespace on the way in, so it would be testing the normalisation rules
    against themselves; slicing the source's own characters keeps the quote a
    genuine substring and leaves exactly one question open, which is whether
    the verifier finds it.
    """
    words = [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]
    windows = [
        text[start:end]
        for (start, _), (_other, end) in zip(words, words[width - 1 :], strict=False)
    ]
    return windows[::_GATE_STRIDE]


def _retyped_windows(text: str, width: int = _GATE_WINDOW) -> list[str]:
    """Word windows with the line wrapping removed, as a reader would retype them.

    This is what a model reading a page actually produces: the reader sees one
    line where the text layer has four, so such a quote is found by alignment
    rather than by byte comparison, and the gate has to accept that as long as
    the only rule that fired is the one that explains it.
    """
    words = text.split()
    windows = [
        " ".join(words[start : start + width])
        for start in range(0, max(1, len(words) - width), width)
    ]
    return windows[::_GATE_STRIDE]


#: A window that retypes a word the layout split with a hyphen. The source
#: holds "transduc-\n tion" and the reader holds "transduc- tion"; the
#: de-hyphenation rule joins the source's two halves into "transduction",
#: which is not the same string as the window, so these do not align. They
#: are the one documented reason a retyped quote can fail, and the gate below
#: asserts that no *other* reason exists rather than pretending they align.
_LINE_WRAP_HYPHEN = re.compile(r"\w- ")


@requires_corpus
class TestCorpusExactAlignment:
    """The release gate: "direct quote exact/normalized alignment 100%"."""

    def test_every_gate_file_is_present(self):
        # A corpus that lost a file would otherwise turn the gate below into a
        # smaller claim that still reports 100%.
        missing = [name for name in _GATE_FILES if not (_CORPUS_ROOT / name).is_file()]

        assert not missing, f"corpus files named by the citation gate are missing: {missing}"

    def test_a_quote_taken_from_the_page_always_aligns_exactly(self):
        checked = 0
        failures: list[str] = []

        for relative_path in _GATE_FILES:
            for page_number, text in _gate_pages(relative_path):
                element = _page_text(text, page=page_number)
                for quote in _verbatim_windows(text):
                    checked += 1
                    result = verify_quote(quote, element)
                    if not (result.aligned and result.mode is AlignmentMode.EXACT):
                        failures.append(
                            f"{relative_path} p.{page_number}: {result.outcome.value} {quote[:60]!r}"
                        )

        assert checked > 500, f"the gate only checked {checked} quotes; it is not a real sample"
        assert failures == [], failures[:5]

    def test_a_quote_retyped_across_line_wraps_aligns_or_degrades_and_never_mislies(self):
        # The other half of the gate: a reader — or a model reading the page
        # rather than the text layer — retypes a line-wrapped quote without its
        # line breaks, so it has to be found by alignment rather than by byte
        # comparison. Every such window has to end in one of exactly two
        # states: aligned, and aligned by a rule that explains it; or not
        # aligned, and then carrying a render that cannot be put inside
        # quotation marks. The set of failures is asserted to be *exactly* the
        # windows that retyped a line-wrap hyphen, so a new failure mode shows
        # up as a failing count rather than as a statistic nobody reads.
        aligned = 0
        degraded: list[str] = []
        unexplained: list[str] = []

        for relative_path in _GATE_FILES:
            for page_number, text in _gate_pages(relative_path):
                element = _page_text(text, page=page_number)
                for quote in _retyped_windows(text):
                    result = verify_quote(quote, element)
                    if result.aligned:
                        aligned += 1
                        if result.mode is AlignmentMode.NORMALIZED:
                            # A retyped window that needed normalising needed
                            # it for the line wrapping. Other rules can appear
                            # beside it — a window carrying a curly quote has
                            # that rule fire on *both* sides of the
                            # comparison, which is what a symmetric rule is
                            # for — so the assertion is that the reason is
                            # present, not that the list is short.
                            assert "whitespace_collapse" in result.applied_rules, (
                                relative_path,
                                quote[:60],
                                result.applied_rules,
                            )
                        continue
                    rendered = result.render()
                    assert rendered.uses_quotation_marks is False
                    if not _LINE_WRAP_HYPHEN.search(quote):
                        unexplained.append(
                            f"{relative_path} p.{page_number}: {result.outcome.value} {quote[:60]!r}"
                        )
                    degraded.append(
                        f"{relative_path} p.{page_number}: {result.outcome.value} {quote[:60]!r}"
                    )

        # The sample is asserted as a count so that a corpus change which
        # empties a page cannot turn this gate into a pass over nothing. 100 of
        # 136 windows is what the current corpus yields; the interesting number
        # is not this one but ``unexplained``, which must stay empty.
        assert aligned + len(degraded) > 100, "the retyped sample shrank"
        assert aligned > 80, f"only {aligned} retyped quotes aligned"
        assert unexplained == [], unexplained[:5]

    def test_a_quote_from_one_page_never_verifies_against_another(self):
        pages = _gate_pages(_GATE_FILES[0])
        borrowed = pages[0][1].split()
        if len(borrowed) < 20:
            pytest.skip("first corpus page has too little text to borrow from")
        quote = " ".join(borrowed[4:18])
        target_page, target_text = pages[-1]

        result = verify_quote(quote, _page_text(target_text, page=target_page))

        assert result.aligned is False
