"""Parse quality gate: eight rules, flag-level reporting and status aggregation.

The gate is judged here as a reporting artifact rather than a classifier, so
the tests read the way the product does: build a document that fails in one
specific way, ask which flags came back, and check that each flag carries the
numbers behind it. Most documents are hand-built ``ParsedDocument`` values because
most of the failures being tested — a blank page, a missing CMap, a truncated parse —
is a schema-level state that no real parser needs to produce. The T0.6
acceptance case is different: it asks what happens to *known bad corpus
files*, so the last two classes rebuild those documents from the properties
``eval/manifest.yaml`` already recorded about them (page count, text-layer
presence, extractable characters) rather than inventing properties. The PDFs
themselves are never opened, and those tests skip loudly when the corpus is
absent.

Three behaviours are pinned deliberately, because all are easy to regress and
neither is obvious from the code:

* **WARN is not FAIL.** A document with only soft findings must report ``WARN``
  so the orchestrator can proceed with a visible warning. Collapsing the two
  would either block good parses or hide broken ones.
* **Severity is policy, and the policy starts as the rule's own view.** Every
  rule ships a severity and configuration may re-level it at runtime; with no
  policy configured each rule reports exactly what it always did.
* **There is no score.** A single number is the one shape this module is not
  allowed to take, and the absence is asserted directly.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from paperwhisperer.core import config
from paperwhisperer.ingestion.quality import (
    RULES,
    ParseQualityReport,
    QualityFlag,
    QualityRule,
    QualitySeverity,
    QualityStatus,
    QualityThresholds,
    evaluate_quality,
    get_rule,
    thresholds_from_config,
)
from paperwhisperer.ingestion.schema import Element, ParsedDocument, SourceLocator, Table

ALL_RULE_IDS = (
    "page_coverage",
    "blank_page_ratio",
    "garbled_characters",
    "fragmented_lines",
    "repeated_furniture",
    "text_density",
    "empty_table_shells",
    "locator_coverage",
)


def _prose(line_count: int = 12, line_chars: int = 57) -> str:
    """Body text with healthy line lengths, for documents meant to pass."""
    line = "lorem ipsum dolor sit amet consectetur adipiscing elit sed "[:line_chars]
    return "\n".join(line for _ in range(line_count))


def _page(page: int, text: str = "", role: str = "paragraph") -> Element:
    return Element(
        role=role,
        text=text,
        locator=SourceLocator(kind="pdf_page", page=page),
    )


def _furniture(page: int, text: str, role: str = "footer") -> Element:
    return Element(role=role, text=text, locator=SourceLocator(kind="pdf_page", page=page))


def _short_lines(count: int = 20) -> str:
    """One- and two-character lines, the signature of a shattered grid."""
    return "\n".join("ab" for _ in range(count))


def _healthy_document() -> ParsedDocument:
    """A four-page parse that trips nothing."""
    return ParsedDocument(
        source_filename="clean.pdf",
        media_type="application/pdf",
        page_count=4,
        parser_name="test-parser",
        elements=[_page(page, _prose()) for page in range(1, 5)],
    )


def _degraded_document() -> ParsedDocument:
    """A parse that trips all eight rules at once."""
    garbled = "".join("\ufffd" for _ in range(200))
    elements = [
        _page(1, _prose()),
        Element(role="paragraph", text=garbled, locator=SourceLocator(kind="pdf_page", page=1)),
        Element(
            role="paragraph",
            text=_short_lines(),
            locator=SourceLocator(kind="pdf_page", page=1),
        ),
        Element(role="table", table=Table(rows=[["", ""], ["", ""]]),
                locator=SourceLocator(kind="pdf_page", page=1)),
        Element(role="paragraph", text=_prose(), locator=SourceLocator()),
    ]
    # Footers on every page register the pages that produced no body text, and
    # their page numbers all normalize to the same string.
    elements.extend(_furniture(page, f"{page}") for page in range(1, 21))
    return ParsedDocument(
        source_filename="broken.pdf",
        media_type="application/pdf",
        page_count=20,
        parser_name="test-parser",
        elements=elements,
    )


def _flags_of(rule_id: str, document: ParsedDocument) -> list[QualityFlag]:
    """Run exactly one rule, the way the eval runner addresses them."""
    return get_rule(rule_id).evaluate(document)


class TestPageCoverageRule:
    def test_reports_pages_the_parser_never_reached(self):
        document = ParsedDocument(page_count=20, elements=[_page(1, _prose())])

        flags = _flags_of("page_coverage", document)

        assert [flag.rule_id for flag in flags] == ["page_coverage"]
        assert flags[0].details["covered_pages"] == 1
        assert flags[0].details["missing_page_count"] == 19

    def test_stays_silent_when_every_page_produced_body_text(self):
        document = ParsedDocument(
            page_count=4,
            elements=[_page(page, _prose()) for page in range(1, 5)],
        )

        assert _flags_of("page_coverage", document) == []

    def test_stays_silent_for_an_unpaginated_document(self):
        # A .txt upload has no page to be missing, so the rule must abstain
        # rather than invent a failure.
        document = ParsedDocument(elements=[Element(role="paragraph", text=_prose())])

        assert _flags_of("page_coverage", document) == []

    def test_a_loose_threshold_admits_a_truncated_parse(self):
        document = ParsedDocument(page_count=10, elements=[_page(1, _prose())])
        lenient = QualityThresholds(min_page_coverage_ratio=0.1)

        assert get_rule("page_coverage").with_thresholds(lenient).evaluate(document) == []


class TestBlankPageRatioRule:
    def test_reports_pages_that_came_back_without_body_text(self):
        document = ParsedDocument(
            page_count=3,
            elements=[
                _page(1, _prose()),
                _furniture(2, "2"),
                _furniture(3, "3"),
            ],
        )

        flags = _flags_of("blank_page_ratio", document)

        assert flags[0].severity is QualitySeverity.FAIL
        assert flags[0].details["blank_pages"] == 2
        assert flags[0].details["blank_page_numbers"] == [2, 3]

    def test_ignores_pages_that_carry_only_a_header(self):
        # A page whose only content is running furniture is still a page that
        # yielded nothing to analyze.
        document = ParsedDocument(
            page_count=3,
            elements=[
                _page(1, _prose()),
                Element(role="header", text="Journal of Tests",
                        locator=SourceLocator(kind="pdf_page", page=2)),
                _furniture(3, "3"),
            ],
        )

        flags = _flags_of("blank_page_ratio", document)

        assert flags[0].details["blank_page_numbers"] == [2, 3]

    def test_stays_silent_when_no_page_is_blank(self):
        document = ParsedDocument(
            page_count=2,
            elements=[_page(1, _prose()), _page(2, _prose())],
        )

        assert _flags_of("blank_page_ratio", document) == []


class TestGarbledCharactersRule:
    def test_reports_text_that_is_mostly_decoding_failure(self):
        document = ParsedDocument(elements=[_page(1, "\ufffd" * 50 + "real words here")])

        flags = _flags_of("garbled_characters", document)

        assert flags[0].details["garbled_chars"] == 50
        assert "U+FFFD" in flags[0].details["sample_codepoints"]

    def test_reports_null_bytes_left_by_a_missing_cmap(self):
        document = ParsedDocument(elements=[_page(1, "\x00" * 40 + "text")])

        flags = _flags_of("garbled_characters", document)

        assert flags[0].details["garbled_chars"] == 40

    def test_allows_the_whitespace_controls_a_text_layer_may_contain(self):
        document = ParsedDocument(elements=[_page(1, "line one\nline two\tindented\r\n")])

        assert _flags_of("garbled_characters", document) == []

    def test_stays_silent_on_clean_text(self):
        document = ParsedDocument(elements=[_page(1, _prose())])

        assert _flags_of("garbled_characters", document) == []


class TestFragmentedLinesRule:
    def test_reports_text_shattered_into_tiny_lines(self):
        document = ParsedDocument(elements=[_page(1, _short_lines())])

        flags = _flags_of("fragmented_lines", document)

        assert flags[0].severity is QualitySeverity.WARN
        assert flags[0].details["fragmented_lines"] == 20
        assert flags[0].details["max_fragment_line_chars"] == 3

    def test_stays_silent_below_the_minimum_sample_size(self):
        # Three short lines are an abstract, not evidence of a broken grid.
        document = ParsedDocument(elements=[_page(1, "ab\ncd\nef")])

        assert _flags_of("fragmented_lines", document) == []

    def test_stays_silent_on_well_formed_lines(self):
        document = ParsedDocument(elements=[_page(1, _prose(line_count=20))])

        assert _flags_of("fragmented_lines", document) == []


class TestRepeatedFurnitureRule:
    def test_reports_one_running_header_repeated_on_every_page(self):
        document = ParsedDocument(
            page_count=3,
            elements=[
                _page(page, _prose()) for page in range(1, 4)
            ] + [
                _furniture(page, "Journal of Tests", role="header")
                for page in range(1, 4)
            ],
        )

        flags = _flags_of("repeated_furniture", document)

        assert flags[0].details["repeat_count"] == 3
        assert flags[0].details["normalized_text"] == "journal of tests"

    def test_page_numbers_are_recognised_as_one_repeated_footer(self):
        # The literal strings differ every page, which is exactly why digits
        # are masked before counting.
        document = ParsedDocument(
            page_count=4,
            elements=[_furniture(page, f"{page} of 4") for page in range(1, 5)],
        )

        flags = _flags_of("repeated_furniture", document)

        assert flags[0].details["repeat_count"] == 4

    def test_stays_silent_below_the_repeat_threshold(self):
        document = ParsedDocument(
            page_count=2,
            elements=[_furniture(1, "Journal of Tests"), _furniture(2, "Other Journal")],
        )

        assert _flags_of("repeated_furniture", document) == []

    def test_stays_silent_when_there_is_no_furniture(self):
        document = ParsedDocument(page_count=3, elements=[_page(1, _prose())])

        assert _flags_of("repeated_furniture", document) == []


class TestTextDensityRule:
    def test_reports_pages_the_parser_never_read(self):
        document = ParsedDocument(page_count=20, elements=[_page(1, _prose())])

        flags = _flags_of("text_density", document)

        assert flags[0].severity is QualitySeverity.FAIL
        assert flags[0].details["chars_per_page"] < flags[0].details["min_chars_per_page"]

    def test_reports_an_unpaginated_document_with_no_content(self):
        document = ParsedDocument(elements=[Element(role="paragraph", text="hi")])

        flags = _flags_of("text_density", document)

        assert flags[0].details["min_document_chars"] == 200

    def test_stays_silent_on_dense_pages(self):
        document = ParsedDocument(
            page_count=4,
            elements=[_page(page, _prose()) for page in range(1, 5)],
        )

        assert _flags_of("text_density", document) == []

    def test_furniture_does_not_count_towards_density(self):
        # Headers are excluded from body text by the schema, so padding a
        # document with them cannot disguise an empty parse.
        document = ParsedDocument(
            page_count=10,
            elements=[_page(1, _prose())]
            + [_furniture(page, "x" * 500) for page in range(2, 11)],
        )

        flags = _flags_of("text_density", document)

        assert flags[0].details["char_count"] == len(_prose())


class TestEmptyTableShellsRule:
    def test_reports_a_table_that_is_only_an_outline(self):
        document = ParsedDocument(
            elements=[
                Element(
                    role="table",
                    table=Table(rows=[["", ""], ["", ""]]),
                    locator=SourceLocator(kind="pdf_page", page=4),
                )
            ],
        )

        flags = _flags_of("empty_table_shells", document)

        assert flags[0].severity is QualitySeverity.WARN
        assert flags[0].details["empty_shell_count"] == 1
        assert flags[0].locator is not None
        assert flags[0].locator.page == 4

    def test_stays_silent_when_a_table_has_content(self):
        document = ParsedDocument(
            elements=[
                Element(role="table", table=Table(rows=[["Metric", "Value"], ["Acc", "0.91"]]))
            ],
        )

        assert _flags_of("empty_table_shells", document) == []

    def test_stays_silent_when_the_document_has_no_tables(self):
        document = ParsedDocument(elements=[_page(1, _prose())])

        assert _flags_of("empty_table_shells", document) == []

    def test_tolerates_one_shell_among_several_real_tables(self):
        document = ParsedDocument(
            elements=[
                Element(role="table", table=Table(rows=[["A", "1"]])),
                Element(role="table", table=Table(rows=[["B", "2"]])),
                Element(role="table", table=Table(rows=[["", ""]])),
            ],
        )

        assert _flags_of("empty_table_shells", document) == []


class TestLocatorCoverageRule:
    def test_reports_elements_a_citation_cannot_point_at(self):
        document = ParsedDocument(
            elements=[
                _page(1, _prose()),
                Element(role="paragraph", text=_prose(), locator=SourceLocator()),
            ],
        )

        flags = _flags_of("locator_coverage", document)

        assert flags[0].severity is QualitySeverity.FAIL
        assert flags[0].details["resolvable_elements"] == 1
        assert flags[0].details["body_elements"] == 2
        assert flags[0].details["unresolvable_kinds"] == ["unknown"]

    def test_stays_silent_when_every_element_resolves(self):
        document = ParsedDocument(elements=[_page(1, _prose())])

        assert _flags_of("locator_coverage", document) == []

    def test_a_tolerated_ratio_is_configurable(self):
        document = ParsedDocument(
            elements=[
                _page(1, _prose()),
                Element(role="paragraph", text=_prose(), locator=SourceLocator()),
            ],
        )
        lenient = QualityThresholds(min_resolvable_locator_ratio=0.5)

        assert get_rule("locator_coverage").with_thresholds(lenient).evaluate(document) == []


class TestQualityThresholds:
    def test_overrides_a_known_key(self):
        assert QualityThresholds.from_dict({"min_chars_per_page": 100}).min_chars_per_page == 100

    def test_rejects_a_misspelled_key_instead_of_ignoring_it(self):
        # A silently dropped threshold would leave a corpus judged against a
        # default the operator believes they changed.
        with pytest.raises(ValueError, match="Unknown quality threshold key"):
            QualityThresholds.from_dict({"typo_key": 1})

    def test_round_trips_through_a_dict(self):
        thresholds = QualityThresholds(max_blank_page_ratio=0.4, min_document_chars=50)

        assert QualityThresholds.from_dict(thresholds.to_dict()) == thresholds

    def test_reads_configured_bars_and_falls_back_for_the_rest(self, monkeypatch):
        monkeypatch.setattr(config, "PARSER_QUALITY_MIN_CHARS_PER_PAGE", 1000, raising=False)

        thresholds = thresholds_from_config()

        assert thresholds.min_chars_per_page == 1000
        assert thresholds.max_blank_page_ratio == QualityThresholds().max_blank_page_ratio

    def test_ignores_a_configured_value_that_is_not_a_number(self, monkeypatch):
        monkeypatch.setattr(config, "PARSER_QUALITY_MIN_CHARS_PER_PAGE", "lots", raising=False)

        assert thresholds_from_config().min_chars_per_page == 200


class TestRuleRegistry:
    def test_registers_exactly_the_eight_planned_rules(self):
        assert tuple(rule.rule_id for rule in RULES) == ALL_RULE_IDS

    def test_every_rule_declares_its_own_severity(self):
        for rule in RULES:
            assert isinstance(rule.severity, QualitySeverity)

    @pytest.mark.parametrize("rule_id", ALL_RULE_IDS)
    def test_get_rule_resolves_every_registered_id(self, rule_id):
        assert get_rule(rule_id).rule_id == rule_id

    def test_get_rule_raises_on_an_unknown_id(self):
        # Returning None would read downstream as "this check passed".
        with pytest.raises(KeyError, match="Unknown quality rule"):
            get_rule("nope")

    def test_with_thresholds_returns_a_bound_copy(self):
        rule = get_rule("text_density")

        rebound = rule.with_thresholds(QualityThresholds(min_chars_per_page=5))

        assert rebound.thresholds.min_chars_per_page == 5
        assert rule.thresholds.min_chars_per_page == 200

    def test_the_base_class_refuses_to_evaluate(self):
        with pytest.raises(NotImplementedError):
            QualityRule().evaluate(_healthy_document())


class TestEvaluateQuality:
    def test_a_healthy_document_passes_with_no_flags(self):
        report = evaluate_quality(_healthy_document())

        assert report.status is QualityStatus.PASS
        assert report.flags == []

    def test_a_fully_degraded_document_fails_and_names_every_rule(self):
        report = evaluate_quality(_degraded_document())

        assert report.status is QualityStatus.FAIL
        assert {flag.rule_id for flag in report.flags} == set(ALL_RULE_IDS)

    def test_soft_findings_alone_report_warn_not_fail(self):
        # The orchestrator must be able to proceed on a degraded parse, so
        # warnings are not allowed to escalate into a block.
        document = ParsedDocument(
            page_count=3,
            parser_name="test-parser",
            elements=[_page(page, _prose()) for page in range(1, 4)]
            + [
                _furniture(page, "Journal of Tests", role="header")
                for page in range(1, 4)
            ]
            + [
                Element(
                    role="paragraph",
                    text=_short_lines(),
                    locator=SourceLocator(kind="pdf_page", page=1),
                ),
                Element(
                    role="table",
                    table=Table(rows=[["", ""]]),
                    locator=SourceLocator(kind="pdf_page", page=1),
                ),
            ],
        )

        report = evaluate_quality(document)

        assert report.status is QualityStatus.WARN
        assert report.failed_rules == ()
        assert set(report.warned_rules) == {
            "fragmented_lines",
            "repeated_furniture",
            "empty_table_shells",
        }

    def test_an_empty_document_fails_on_density(self):
        report = evaluate_quality(ParsedDocument())

        assert report.status is QualityStatus.FAIL
        assert [flag.rule_id for flag in report.flags] == ["text_density"]

    def test_records_which_parser_and_document_were_judged(self):
        report = evaluate_quality(_degraded_document())

        assert report.parser_name == "test-parser"
        assert report.document_id.startswith("doc_")
        assert report.evaluated_rules == ALL_RULE_IDS

    def test_can_run_a_single_rule_for_the_eval_runner(self):
        document = _degraded_document()

        report = evaluate_quality(document, rules=(get_rule("text_density"),))

        assert [flag.rule_id for flag in report.flags] == ["text_density"]
        assert report.evaluated_rules == ("text_density",)

    def test_honours_caller_supplied_thresholds(self):
        document = ParsedDocument(page_count=20, elements=[_page(1, _prose())])
        lenient = QualityThresholds(min_page_coverage_ratio=0.01, min_chars_per_page=1)

        report = evaluate_quality(document, thresholds=lenient)

        assert report.thresholds == lenient
        assert "page_coverage" not in {flag.rule_id for flag in report.flags}


class TestReportCarriesNoScore:
    """The design constraint, asserted rather than documented."""

    def test_the_report_has_no_score_attribute(self):
        report = evaluate_quality(_degraded_document())

        assert not hasattr(report, "score")
        assert not hasattr(report, "quality_score")

    def test_no_serialized_field_mentions_a_score(self):
        report = evaluate_quality(_degraded_document())

        assert "score" not in report.to_dict()
        assert "score" not in report.summary()

    def test_the_status_is_the_worst_severity_and_nothing_more(self):
        flags = [QualityFlag(rule_id="a", severity=QualitySeverity.WARN, message="soft")]

        from paperwhisperer.ingestion.quality import status_of

        assert status_of([]) is QualityStatus.PASS
        assert status_of(flags) is QualityStatus.WARN
        assert status_of(
            flags + [QualityFlag(rule_id="b", severity=QualitySeverity.FAIL, message="hard")]
        ) is QualityStatus.FAIL

    def test_every_flag_explains_itself_with_observed_numbers(self):
        report = evaluate_quality(_degraded_document())

        for flag in report.flags:
            assert flag.message
            assert flag.details
            assert any(
                key.startswith(("min_", "max_"))
                for key in flag.details
            ), f"{flag.rule_id} 没有记录判定阈值"


class TestReportSerialization:
    def test_round_trips_through_a_dict(self):
        report = evaluate_quality(_degraded_document())

        restored = ParseQualityReport.from_dict(report.to_dict())

        assert restored.to_dict() == report.to_dict()
        assert restored.status is report.status
        assert [flag.rule_id for flag in restored.flags] == [f.rule_id for f in report.flags]

    def test_round_trips_a_report_that_raised_no_flags(self):
        report = evaluate_quality(_healthy_document())

        assert ParseQualityReport.from_dict(report.to_dict()).flags == []

    def test_summary_counts_by_severity_and_names_the_rules(self):
        report = evaluate_quality(_degraded_document())

        summary = report.summary()

        assert summary["status"] == "fail"
        assert summary["flag_total"] == len(report.flags)
        assert sum(summary["counts_by_severity"].values()) == summary["flag_total"]
        assert "text_density" in summary["failed_rules"]

    def test_keeps_the_thresholds_the_verdict_was_judged_against(self):
        thresholds = QualityThresholds(min_chars_per_page=1)
        report = evaluate_quality(_healthy_document(), thresholds=thresholds)

        restored = ParseQualityReport.from_dict(report.to_dict())

        assert restored.thresholds == thresholds

    def test_a_stored_flag_keeps_its_locator(self):
        report = evaluate_quality(_degraded_document())
        payload = report.to_dict()

        restored = ParseQualityReport.from_dict(payload)
        located = [flag for flag in restored.flags if flag.locator is not None]

        assert located
        assert located[0].locator.kind in {
            "pdf_page",
            "slide",
            "sheet_range",
            "docx_element",
            "txt_offset",
            "unknown",
        }


# --------------------------------------------------------------------------
# Runtime-configurable severity
# --------------------------------------------------------------------------

#: The severity every rule ships with. Written out as a table rather than left
#: to the class bodies so that flipping a default, or adding a ninth rule, is a
#: change somebody has to argue for here instead of one that happens quietly.
DEFAULT_SEVERITIES = {
    "page_coverage": QualitySeverity.FAIL,
    "blank_page_ratio": QualitySeverity.FAIL,
    "garbled_characters": QualitySeverity.FAIL,
    "fragmented_lines": QualitySeverity.WARN,
    "repeated_furniture": QualitySeverity.WARN,
    "text_density": QualitySeverity.FAIL,
    "empty_table_shells": QualitySeverity.WARN,
    "locator_coverage": QualitySeverity.FAIL,
}


def _empty_document() -> ParsedDocument:
    """A parse that produced nothing: one finding, and it is unambiguous."""
    return ParsedDocument()


def _running_header_document() -> ParsedDocument:
    """Three dense pages under one repeated header: exactly one soft finding."""
    return ParsedDocument(
        page_count=3,
        parser_name="test-parser",
        elements=[_page(page, _prose()) for page in range(1, 4)]
        + [_furniture(page, "Journal of Tests", role="header") for page in range(1, 4)],
    )


def _two_blank_pages() -> ParsedDocument:
    """Three pages of which only the first produced anything to analyse."""
    return ParsedDocument(
        page_count=3,
        elements=[_page(1, _prose()), _furniture(2, "2"), _furniture(3, "3")],
    )


class TestRuntimeConfigurableSeverity:
    """T0.6: WARN or FAIL is a per-rule decision a deployment can change."""

    def test_the_shipped_severity_of_every_rule_is_unchanged(self):
        assert {rule.rule_id: rule.severity for rule in RULES} == DEFAULT_SEVERITIES

    def test_no_policy_configured_means_every_rule_reports_its_own_severity(self):
        # The compatibility guarantee: an empty policy is the pre-configurable
        # behaviour, so nothing an existing caller does can change.
        assert {rule.rule_id: rule.effective_severity for rule in RULES} == DEFAULT_SEVERITIES

    def test_a_policy_demotes_a_failure_to_a_warning(self):
        policy = QualityThresholds(severity_overrides={"text_density": "warn"})

        report = evaluate_quality(_empty_document(), thresholds=policy)

        assert report.status is QualityStatus.WARN
        assert [flag.severity for flag in report.flags] == [QualitySeverity.WARN]
        assert report.failed_rules == ()

    def test_a_policy_escalates_a_warning_to_a_failure(self):
        policy = QualityThresholds(severity_overrides={"repeated_furniture": "fail"})

        report = evaluate_quality(_running_header_document(), thresholds=policy)

        assert report.status is QualityStatus.FAIL
        assert report.failed_rules == ("repeated_furniture",)

    def test_a_policy_relevels_a_rule_without_silencing_it(self):
        # "This is only a warning" and "this is not a finding" are different
        # claims, and configuration is only entitled to make the first.
        policy = QualityThresholds(severity_overrides={"text_density": "warn"})

        report = evaluate_quality(_empty_document(), thresholds=policy)

        assert [flag.rule_id for flag in report.flags] == ["text_density"]

    def test_a_policy_leaves_the_rules_it_does_not_name_alone(self):
        policy = QualityThresholds(severity_overrides={"text_density": "warn"})

        report = evaluate_quality(_two_blank_pages(), thresholds=policy)

        assert set(report.failed_rules) == {"page_coverage", "blank_page_ratio"}

    def test_a_policy_does_not_leak_into_the_next_parse(self):
        # RULES holds one instance per rule for the life of the process, so a
        # per-parse policy has to be applied to a copy. Mutating the shared
        # instance would let one document's policy judge the next one.
        evaluate_quality(
            _empty_document(),
            thresholds=QualityThresholds(severity_overrides={"text_density": "warn"}),
        )

        assert {rule.rule_id: rule.severity for rule in RULES} == DEFAULT_SEVERITIES
        assert get_rule("text_density").thresholds.severity_overrides == {}
        assert evaluate_quality(_empty_document()).status is QualityStatus.FAIL

    def test_an_unreadable_severity_falls_back_to_the_rule_default(self):
        # Same posture as an unreadable numeric bar: the bar the rule author
        # chose stands, and the flag still explains itself.
        rule = get_rule("text_density").with_thresholds(
            QualityThresholds(severity_overrides={"text_density": "block"})
        )

        assert rule.effective_severity is QualitySeverity.FAIL

    def test_a_policy_written_in_code_rejects_a_misspelled_rule_id(self):
        # An override nobody can read is a decision nobody made, and it would
        # otherwise sit in a report looking like one.
        with pytest.raises(ValueError, match="Unknown quality rule in severity policy"):
            QualityThresholds.from_dict({"severity_overrides": {"text_densiy": "warn"}})

    def test_a_policy_written_in_code_rejects_an_invented_severity(self):
        with pytest.raises(ValueError, match="Unknown severity 'block'"):
            QualityThresholds.from_dict({"severity_overrides": {"text_density": "block"}})

    def test_a_policy_written_in_code_rejects_a_malformed_entry(self):
        with pytest.raises(ValueError, match="not rule_id=severity"):
            QualityThresholds.from_dict({"severity_overrides": "text_density"})

    def test_a_policy_accepts_the_comma_separated_form_a_config_value_uses(self):
        policy = QualityThresholds.from_dict({
            "severity_overrides": "text_density=warn, repeated_furniture=fail",
        })

        demoted = policy.severity_for("text_density", QualitySeverity.FAIL)
        escalated = policy.severity_for("repeated_furniture", QualitySeverity.WARN)

        assert demoted is QualitySeverity.WARN
        assert escalated is QualitySeverity.FAIL

    def test_a_rule_the_policy_does_not_name_keeps_the_default_it_was_given(self):
        unconfigured = QualityThresholds()
        resolved = unconfigured.severity_for("text_density", QualitySeverity.FAIL)

        assert resolved is QualitySeverity.FAIL

    def test_the_policy_is_kept_in_a_stored_report(self):
        # A verdict read back months later has to be readable as a verdict
        # under the policy that produced it, not just as a list of flags.
        policy = QualityThresholds(severity_overrides={"text_density": "warn"})

        report = evaluate_quality(_empty_document(), thresholds=policy)
        restored = ParseQualityReport.from_dict(report.to_dict())

        assert restored.thresholds.severity_overrides == {"text_density": "warn"}
        assert restored.flags[0].severity is QualitySeverity.WARN

    def test_serializing_thresholds_does_not_share_the_policy_mapping(self):
        policy = QualityThresholds(severity_overrides={"text_density": "warn"})

        payload = policy.to_dict()
        payload["severity_overrides"]["garbled_characters"] = "warn"

        assert policy.severity_overrides == {"text_density": "warn"}


class TestConfigDrivenPolicy:
    """The same knobs, arriving through ``core.config`` instead of a caller.

    ``quality.py`` and the ``PARSER_QUALITY_*`` block of ``core.config`` are
    allowed to land in different commits, so the whole point of the ``getattr``
    dance is that this module runs correctly while the constants are absent —
    which is a real state, not a hypothetical one.
    """

    _CONFIG_KEYS = (
        "PARSER_QUALITY_MIN_CHARS_PER_PAGE",
        "PARSER_QUALITY_MAX_BLANK_PAGE_RATIO",
        "PARSER_QUALITY_MIN_DOCUMENT_CHARS",
        "PARSER_QUALITY_SEVERITY_OVERRIDES",
    )

    def test_runs_on_built_in_defaults_while_the_config_constants_are_absent(self, monkeypatch):
        for name in self._CONFIG_KEYS:
            monkeypatch.delattr(config, name, raising=False)

        assert thresholds_from_config() == QualityThresholds()
        report = evaluate_quality(_degraded_document())
        assert report.status is QualityStatus.FAIL
        assert report.failed_rules

    def test_a_configured_threshold_reaches_the_verdict(self, monkeypatch):
        # Thresholds arrive from the environment as strings, so this also
        # proves the string is coerced rather than compared as text.
        assert set(evaluate_quality(_two_blank_pages()).failed_rules) == {
            "page_coverage",
            "blank_page_ratio",
        }
        monkeypatch.setattr(config, "PARSER_QUALITY_MAX_BLANK_PAGE_RATIO", "1.0", raising=False)

        report = evaluate_quality(_two_blank_pages())

        # The pages are still unanalysable, they are simply no longer over the
        # blank-page bar — a threshold change must move exactly that one line.
        assert report.failed_rules == ("page_coverage",)

    def test_a_configured_document_length_bar_reaches_the_verdict(self, monkeypatch):
        document = ParsedDocument(elements=[_page(1, "hi")])
        assert evaluate_quality(document).failed_rules == ("text_density",)

        monkeypatch.setattr(config, "PARSER_QUALITY_MIN_DOCUMENT_CHARS", 1, raising=False)

        assert evaluate_quality(document).flags == []

    def test_a_configured_severity_policy_reaches_the_verdict(self, monkeypatch):
        monkeypatch.setattr(
            config, "PARSER_QUALITY_SEVERITY_OVERRIDES", "text_density=warn", raising=False
        )

        report = evaluate_quality(_empty_document())

        assert report.status is QualityStatus.WARN
        assert report.thresholds.severity_overrides == {"text_density": "warn"}

    def test_config_is_read_on_every_call_rather_than_cached_at_import(self, monkeypatch):
        # A corpus re-judged under a new policy must not need a reimport.
        monkeypatch.setattr(config, "PARSER_QUALITY_MIN_DOCUMENT_CHARS", 1, raising=False)
        assert thresholds_from_config().min_document_chars == 1

        monkeypatch.setattr(config, "PARSER_QUALITY_MIN_DOCUMENT_CHARS", 999, raising=False)
        assert thresholds_from_config().min_document_chars == 999

    def test_an_oversized_operator_policy_is_filtered_rather_than_fatal(self, monkeypatch):
        # A settings file left over from a build that had a rule this one does
        # not, or a value nobody ever validated, must not stop the gate from
        # judging anything at all. The entries that survive are the ones that
        # named a real rule and a real severity.
        monkeypatch.setattr(
            config,
            "PARSER_QUALITY_SEVERITY_OVERRIDES",
            "text_density=warn, retired_rule=fail, garbled_characters=maybe, nonsense",
            raising=False,
        )

        assert thresholds_from_config().severity_overrides == {"text_density": "warn"}
        assert evaluate_quality(_empty_document()).status is QualityStatus.WARN


# --------------------------------------------------------------------------
# The known-bad corpus: T0.6's acceptance case
# --------------------------------------------------------------------------

_CORPUS_ROOT = Path(__file__).resolve().parents[1] / "eval" / "corpus"
_MANIFEST_PATH = _CORPUS_ROOT.parent / "manifest.yaml"

#: The corpus is gitignored, so these tests are only meaningful where it has
#: been materialised. Skipping rather than omitting them keeps the acceptance
#: criterion visible in a report from CI instead of quietly unverified.
requires_corpus = pytest.mark.skipif(
    not _CORPUS_ROOT.is_dir() or not _MANIFEST_PATH.is_file(),
    reason=(
        f"eval corpus is not present at {_CORPUS_ROOT}; it is gitignored, so this "
        "check only runs where the corpus has been materialised"
    ),
)

_MANIFEST_ENTRY = re.compile(r"^(?P<indent>\s*)-?\s*file:\s*(?P<path>\S+)\s*$")
_MANIFEST_FIELD = re.compile(r"^(?P<indent>\s+)(?P<key>[A-Za-z0-9_]+):(?:\s+(?P<value>.*?))?\s*$")


def _manifest_scalar(raw: str) -> object:
    """Read one manifest scalar: the file only ever uses bools, numbers, text."""
    text = raw.strip()
    if len(text) > 1 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    if text in ("true", "false"):
        return text == "true"
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            continue
    return text


def _manifest_entries() -> dict[str, dict[str, object]]:
    """Every corpus file's recorded properties, keyed by its manifest path.

    Deliberately not PyYAML. The manifest belongs to the eval harness rather
    than to the package, PyYAML is not one of this project's dependencies, and
    the slice needed here is regular enough to read with a regex: a
    ``- file: path`` line followed by deeper-indented ``key: value`` lines, in
    both the nested per-category form and the flat adversarial list. Anything
    that does not fit either shape ends the current record, so a manifest whose
    format changes shows up as a failing count in ``TestCorpusManifest`` rather
    than as corpus tests that quietly stop finding anything.
    """
    entries: dict[str, dict[str, object]] = {}
    current: dict[str, object] | None = None
    entry_indent = -1
    for line in _MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = _MANIFEST_ENTRY.match(line)
        if entry:
            # The record keeps its own path so that a failure message can name
            # the file without the caller having to carry the key alongside it.
            current = {"file": entry["path"]}
            entry_indent = len(entry["indent"])
            entries[entry["path"]] = current
            continue
        field = _MANIFEST_FIELD.match(line)
        if current is not None and field and len(field["indent"]) > entry_indent:
            current[field["key"]] = _manifest_scalar(field["value"] or "")
            continue
        current = None
    return entries


def _parse_of(entry: dict[str, object], *, elements_per_page: bool) -> ParsedDocument:
    """The document a backend would hand back for a real corpus file.

    Built from properties the manifest already recorded about the file — its
    page count, whether it has a text layer at all, how many characters a plain
    text extraction found — and never by opening the file. Parsing these PDFs
    would need pypdf or docling and would turn a unit test into a performance
    test, while what the gate has to decide is a function of exactly those
    recorded numbers, which are the numbers a real parse of that file reports.

    ``elements_per_page`` picks between the two shapes a parser can return for
    a document with no text layer: one empty element per page it walked, or no
    elements at all. They are different failures, and which rule catches each
    is the part worth pinning.
    """
    pages = int(entry.get("pages") or 0)
    return ParsedDocument(
        source_filename=str(entry["file"]).rsplit("/", 1)[-1],
        media_type="application/pdf",
        page_count=pages,
        parser_name="docling",
        elements=[_page(page) for page in range(1, pages + 1)] if elements_per_page else [],
    )


def _unread_scans() -> dict[str, dict[str, object]]:
    """Every scanned corpus file the manifest says has no text layer."""
    scans = {
        path: entry
        for path, entry in _manifest_entries().items()
        if entry.get("text_layer") is False
    }
    assert scans, "manifest.yaml records no scan without a text layer"
    return scans


@requires_corpus
class TestCorpusManifest:
    """The reader above, checked before anything is built from it.

    A change to the manifest's shape should fail here, loudly, rather than
    turning the corpus assertions below into a vacuous pass over nothing.
    """

    def test_reads_every_recorded_corpus_file(self):
        entries = _manifest_entries()

        # 52 documents plus 4 adversarial samples, per the manifest summary.
        assert len(entries) >= 56
        assert any(path.startswith("scanned_clear/") for path in entries)
        assert any(path.startswith("scanned_lowq/") for path in entries)
        assert "adversarial/fake_extension_actually_text.pdf" in entries

    def test_reads_the_properties_the_verdicts_are_built_from(self):
        entry = _manifest_entries()["scanned_clear/paper_en_basic.pdf"]

        assert entry["pages"] == 1
        assert entry["text_layer"] is False
        assert entry["chars_total"] == 0


@requires_corpus
class TestKnownBadScannedCorpus:
    """Scans and low-quality scans, at the level the gate is supposed to use.

    The failure these files share is not garbled characters or a broken page
    order — it is that there is no text layer, so the parse succeeds and the
    document is empty. That is the case a score would average away and a
    per-rule report cannot.
    """

    def test_every_scan_without_a_text_layer_fails_the_density_bar(self):
        for path, entry in _unread_scans().items():
            # The gate is only being asked about the parse this file produces,
            # so the manifest has to still say the parse would be empty.
            assert entry.get("chars_total") == 0, path

            report = evaluate_quality(_parse_of(entry, elements_per_page=True))

            assert report.status is QualityStatus.FAIL, path
            assert "text_density" in report.failed_rules, path

    def test_the_failure_names_only_the_rules_that_can_judge_an_empty_parse(self):
        # A scan that was never OCR'd is empty, not garbled and not
        # unlocatable, so a report that claimed all eight findings would be
        # inventing observations rather than recording them.
        entry = next(iter(_unread_scans().values()))

        report = evaluate_quality(_parse_of(entry, elements_per_page=True))

        assert set(report.failed_rules) == {"blank_page_ratio", "text_density"}
        assert report.warned_rules == ()

    def test_a_parse_that_returned_no_elements_reports_the_pages_missing(self):
        # The other shape of the same scan, and the contrast with the test
        # above: a page the parser walked and found empty is blank, a page it
        # never emitted an element for is missing. Only the second is a
        # coverage failure, and the difference is the only thing the two
        # documents have between them.
        entry = next(iter(_unread_scans().values()))

        report = evaluate_quality(_parse_of(entry, elements_per_page=False))

        assert set(report.failed_rules) == {"page_coverage", "text_density"}

    def test_low_quality_scans_are_judged_the_same_as_clear_ones(self):
        # The manifest records the same absent text layer for both categories,
        # and guessing which scan OCR would read better is the bake-off's job,
        # not the gate's. If a later OCR path ever rescues one of these, the
        # verdict should change because the parse did, not because the file
        # was known to be difficult.
        scans = _unread_scans()
        clear = [p for p in scans if p.startswith("scanned_clear/")]
        lowq = [p for p in scans if p.startswith("scanned_lowq/")]
        assert len(clear) == 5
        assert len(lowq) == 3

        for path in clear + lowq:
            report = evaluate_quality(_parse_of(scans[path], elements_per_page=True))

            assert set(report.failed_rules) == {"blank_page_ratio", "text_density"}, path


@requires_corpus
class TestAdversarialCorpus:
    """Files that should have raised, checked for not passing quietly."""

    @pytest.mark.parametrize("name", [
        "corrupted_truncated.pdf",
        "encrypted_aes128.pdf",
        "fake_extension_actually_text.pdf",
    ])
    def test_a_file_that_never_parses_still_fails_instead_of_passing(self, name):
        path = f"adversarial/{name}"
        entry = _manifest_entries().get(path)
        assert entry is not None, f"{path} is not recorded in manifest.yaml"
        assert (_CORPUS_ROOT / path).is_file()

        # The manifest records no page count and no character count for these,
        # because no text extraction ever completed on them. Reporting "this
        # file is encrypted" is ParseResult.status in the gateway, which is why
        # the manifest excludes them from the batch; what the gate owns is the
        # backstop — a document that reaches it with nothing in it is a
        # failure, never a silent pass, whatever the parser felt about it.
        report = evaluate_quality(_parse_of(entry, elements_per_page=False))

        assert report.status is QualityStatus.FAIL
        assert report.failed_rules == ("text_density",)

    def test_a_fake_extension_parsed_as_text_is_caught_by_the_length_bar(self):
        # 161 bytes of plain text wearing a .pdf extension: the shape a
        # content-sniffing parser produces when it trusts the file rather than
        # the extension. It is not empty, so only the length bar can see it.
        entry = _manifest_entries()["adversarial/fake_extension_actually_text.pdf"]
        document = ParsedDocument(
            source_filename="fake_extension_actually_text.pdf",
            media_type="application/pdf",
            parser_name="docling",
            elements=[_page(1, "not a pdf at all, just text with a pdf name")],
        )

        report = evaluate_quality(document)

        assert int(entry["bytes"]) < QualityThresholds().min_document_chars
        assert report.failed_rules == ("text_density",)

