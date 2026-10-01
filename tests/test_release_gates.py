"""Release-gate tests: every gate has a threshold, and the threshold decides.

What this file is for
---------------------
A gate is only worth having if three things hold, and each has a test here:

1. **It has a real threshold.** Not "someone reads a report and agrees". Every
   gate carries a number from the plan, and :func:`TestGateDefinitions` pins
   each one against that number so a later edit cannot quietly relax it.
2. **The threshold decides.** The pass/fail/skip verdict is a pure function of
   a measurement, :func:`eval.gates.decide_status`, and
   :func:`TestDecisionArithmetic` feeds it hand-built measurements rather than
   whatever a runner happened to produce. A gate that returns ``PASS`` from
   inside its runner is a gate whose threshold has never been shown to bind.
3. **It can actually fail.** Every executable gate is run against a
   *deliberately broken* fixture and asserted to go red. The corpora the gates
   walk are real and currently green, which means a green run demonstrates
   nothing unless the red path is exercised too.

The distinction the whole module rests on -- ``SKIP`` is not ``PASS`` -- is
tested from both sides: a gate whose fixtures are absent must skip, and a
gate that skipped must not be counted among the passes anywhere in the
summary.

**Nothing here touches the network, a database, or ``eval/corpus/``.** The
corpus gates are driven through :meth:`eval.gates.GateContext.with_documents`
and the two subprocess gates through ``GateContext.test_runner``, so a broken
fixture can be constructed in a few lines instead of shipped as a corrupt PDF,
and running this file cannot recurse into pytest.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.gates import (  # noqa: E402  # noqa: E402
    ALL_GATES,
    ALL_STATUSES,
    BLOCKED_GATES,
    DIRECT_QUOTE_FIXTURES,
    EGRESS_NODE_IDS,
    GATE_DIRECT_QUOTE,
    GATE_EGRESS,
    GATE_LOCATOR,
    GATE_PRECISION,
    GATE_RECALL,
    GATE_RECOVERY,
    GATE_TRUNCATION,
    GATES,
    GATES_BY_KEY,
    GATES_VERSION,
    KNOWN_DEFECTS,
    PG_DSN_ENV,
    RECOVERY_POSTGRES_NODES,
    RECOVERY_SQLITE_NODES,
    STATUS_BLOCKED,
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_SKIP,
    THRESHOLD_EXACT,
    THRESHOLD_PRECISION,
    THRESHOLD_RECALL,
    THRESHOLD_ZERO,
    ClaimLabel,
    GateContext,
    GateMeasurement,
    GateResult,
    GateRun,
    GateSpec,
    Measurement,
    MeasurementSkip,
    QueryLabel,
    ReleaseNotReady,
    assert_release_ready,
    corpus_present,
    decide_status,
    defect_explains,
    load_labels,
    render_text,
    run_gate,
    run_gates,
    score_precision,
    score_recall_at_k,
)
from paperwhisperer.ingestion.schema import Element, ParsedDocument, SourceLocator  # noqa: E402
from paperwhisperer.rag.citations import (  # noqa: E402
    AlignmentMode,
    AlignmentOutcome,
    verify_quote,
)

#: The plan's section 7 table, transcribed. Tests assert the registry against
#: this rather than against itself, so a gate cannot be renamed away from the
#: plan without a test noticing.
PLAN_REQUIREMENTS = {
    GATE_DIRECT_QUOTE: "100%；对不齐降级 paraphrase，不带引号",
    GATE_LOCATOR: "100%；或明确标记无视觉定位能力",
    GATE_EGRESS: "0 次（集成测试验证）",
    GATE_TRUNCATION: "0；截断/低质量必须是可见警告或失败态",
    GATE_PRECISION: "≥95%（beta 目标，按标注集校准）",
    GATE_RECALL: "≥90% 起步，中英、表格/公式/正文分开统计",
    GATE_RECOVERY: "worker/API 重启后任务可恢复或明确进入可重试终态",
}

PLAN_THRESHOLDS = {
    GATE_DIRECT_QUOTE: THRESHOLD_EXACT,
    GATE_LOCATOR: THRESHOLD_EXACT,
    GATE_EGRESS: THRESHOLD_ZERO,
    GATE_TRUNCATION: THRESHOLD_ZERO,
    GATE_PRECISION: THRESHOLD_PRECISION,
    GATE_RECALL: THRESHOLD_RECALL,
    GATE_RECOVERY: THRESHOLD_EXACT,
}

#: Text long enough that ``text_density`` (200 chars/page) does not fire, so a
#: fixture meant to exercise one rule does not accidentally trip another.
_DENSE = "解析质量门禁用于判断一次解析是否发生了静默截断。" * 12


def _document(
    *,
    filename: str = "sample.pdf",
    page_count: int = 1,
    covered_pages: tuple[int, ...] = (1,),
    warnings: tuple[str, ...] = (),
    kind: str = "pdf_page",
    with_bbox: bool = False,
) -> ParsedDocument:
    """A minimal canonical document with a controllable page footprint.

    Built directly on the real schema rather than on a fixture file so a test
    can express "nine declared pages, one page of text, no warning" -- which is
    the silent truncation this gate exists to catch -- without shipping a
    deliberately damaged PDF.
    """
    elements = [
        Element(
            role="paragraph",
            text=_DENSE,
            locator=SourceLocator(kind=kind, page=page),
            order=index,
        )
        for index, page in enumerate(covered_pages)
    ]
    if with_bbox:
        from paperwhisperer.ingestion.schema import BoundingBox  # noqa: PLC0415

        for element in elements:
            element.locator.bbox = BoundingBox(x0=0.1, y0=0.1, x1=0.6, y1=0.3)
    return ParsedDocument(
        source_filename=filename,
        media_type="application/pdf",
        page_count=page_count,
        elements=elements,
        warnings=list(warnings),
    )


def _spec_with(measure) -> GateSpec:
    """A copy of a real gate with its runner swapped.

    A copy rather than a mutation: ``GATES_BY_KEY`` is module state shared by
    every test in the session, and a test that left a broken runner installed
    would silently disarm the truncation gate for whatever ran next. Swapping
    the runner on a copy is the same experiment with none of that blast
    radius.
    """
    return replace(GATES_BY_KEY[GATE_TRUNCATION], measure=measure)


# --------------------------------------------------------------------------
# Definitions
# --------------------------------------------------------------------------


class TestGateDefinitions:
    """The registry is the plan's table, entry for entry."""

    def test_every_planned_gate_is_registered(self):
        assert set(GATES_BY_KEY) == set(ALL_GATES)
        assert set(ALL_GATES) == set(PLAN_REQUIREMENTS)

    def test_no_gate_is_declared_twice(self):
        keys = [spec.key for spec in GATES]
        assert len(keys) == len(set(keys))

    def test_each_gate_quotes_the_plan_verbatim(self):
        for key, expected in PLAN_REQUIREMENTS.items():
            assert GATES_BY_KEY[key].requirement == expected

    def test_each_gate_carries_the_plans_threshold(self):
        for key, expected in PLAN_THRESHOLDS.items():
            spec = GATES_BY_KEY[key]
            assert spec.threshold == expected
            assert spec.threshold_text

    def test_a_rate_gate_names_the_direction_it_is_written_in(self):
        # A count gate and a rate gate are different comparisons. A registry
        # that conflated them would let "violations == 0" drift into
        # "violations >= 0.05", which is always true and measures nothing.
        for spec in GATES:
            if spec.kind == "count":
                assert spec.threshold == 0, spec.key
            else:
                assert 0 < spec.threshold <= 1, spec.key

    def test_every_gate_states_what_it_needs(self):
        for spec in GATES:
            assert spec.needs.strip(), f"{spec.key} 没有说明它需要什么"

    def test_the_report_stamps_a_version(self):
        assert GATES_VERSION
        assert GATES[0].to_dict()["threshold"] == THRESHOLD_EXACT


# --------------------------------------------------------------------------
# The threshold decides
# --------------------------------------------------------------------------


class TestDecisionArithmetic:
    """``decide_status`` is the whole verdict, tested on hand-built numbers."""

    def test_a_full_rate_meets_a_100_percent_threshold(self):
        spec = GATES_BY_KEY[GATE_DIRECT_QUOTE]
        status = decide_status(spec, Measurement(10, 10, "at_least"))
        assert status == STATUS_PASS

    def test_one_missed_quote_breaks_a_100_percent_gate(self):
        # 9999/10000 is not 100%. This is the assertion that makes the gate a
        # gate: the plan wrote 100%, and a threshold that tolerated a miss would
        # be a different, weaker claim wearing the same name.
        spec = GATES_BY_KEY[GATE_DIRECT_QUOTE]
        status = decide_status(spec, Measurement(9999, 10000, "at_least"))
        assert status == STATUS_FAIL

    def test_near_miss_does_not_pass_a_100_percent_gate(self):
        spec = GATES_BY_KEY[GATE_LOCATOR]
        assert decide_status(spec, Measurement(0.9999, 1, "at_least")) == STATUS_FAIL

    def test_a_rate_gate_uses_the_plan_threshold_not_a_rounder_one(self):
        spec = GATES_BY_KEY[GATE_PRECISION]
        assert decide_status(spec, Measurement(95, 100, "at_least")) == STATUS_PASS
        assert decide_status(spec, Measurement(94, 100, "at_least")) == STATUS_FAIL

    def test_a_zero_tolerance_gate_fails_on_a_single_violation(self):
        # The numerator of an ``at_most`` measurement *is* the violation count,
        # so one silent truncation out of 52 documents is a failure and not a
        # rate of 0.98 that happens to be under some bar.
        spec = GATES_BY_KEY[GATE_TRUNCATION]
        assert decide_status(spec, Measurement(0, 52, "at_most")) == STATUS_PASS
        assert decide_status(spec, Measurement(1, 52, "at_most")) == STATUS_FAIL
        assert decide_status(spec, Measurement(40, 52, "at_most")) == STATUS_FAIL

    def test_zero_samples_is_never_a_pass(self):
        # The single most important line in this file's module: a threshold met
        # zero times is not a threshold met. Without this, a gate whose corpus
        # failed to materialise would report 100% and the build would go green.
        for spec in GATES:
            if spec.blocked:
                continue
            empty = Measurement(0, 0, "at_least" if spec.kind == "rate" else "at_most")
            assert decide_status(spec, empty) == STATUS_SKIP, spec.key

    def test_a_measurement_with_no_rate_is_a_skip_not_a_zero(self):
        spec = GATES_BY_KEY[GATE_RECALL]
        assert decide_status(spec, Measurement(0, 0, "at_least")) == STATUS_SKIP

    def test_a_none_measurement_is_a_skip(self):
        for spec in GATES:
            assert decide_status(spec, None) == STATUS_SKIP, spec.key

    def test_thresholds_are_not_relaxed_to_make_a_run_green(self):
        # Every threshold is at or tighter than the plan's own wording.
        assert GATES_BY_KEY[GATE_DIRECT_QUOTE].threshold == 1.0
        assert GATES_BY_KEY[GATE_LOCATOR].threshold == 1.0
        assert GATES_BY_KEY[GATE_PRECISION].threshold == 0.95
        assert GATES_BY_KEY[GATE_RECALL].threshold == 0.90


# --------------------------------------------------------------------------
# SKIP is not PASS
# --------------------------------------------------------------------------


class TestSkipIsNotPass:
    """The distinction the module exists to preserve."""

    def test_a_gate_with_no_corpus_skips_and_names_why(self, tmp_path):
        context = GateContext(
            corpus_root=tmp_path / "corpus_that_does_not_exist",
        )
        reason = corpus_present(context)
        assert reason, "缺少语料时必须给出原因"
        result = run_gate(GATES_BY_KEY[GATE_TRUNCATION], context)
        assert result.status == STATUS_SKIP
        assert result.observed == "未测量"
        assert "未测量" in result.observed

    def test_a_skip_is_not_counted_as_a_pass(self, tmp_path):
        context = GateContext(
            corpus_root=tmp_path / "absent",
            run_tests=False,
        )
        run = run_gates(context)
        assert run.passed == ()
        assert run.skipped
        for result in run.skipped:
            assert result not in run.passed
            assert result.status != STATUS_PASS

    def test_skips_do_not_fail_the_build(self, tmp_path):
        context = GateContext(corpus_root=tmp_path / "absent", run_tests=False)
        run = run_gates(context)
        assert run.failed == ()
        assert run.exit_code == 0

    def test_blocked_gates_stay_blocked_even_when_fixtures_exist(self, tmp_path):
        # The corpus is present and everything else is available, and the two
        # label gates are still BLOCKED. Blocked is defined by the absence of
        # ground truth, not by the absence of a fixture, so a green environment
        # must not turn them green.
        context = GateContext(run_tests=False)
        for key in BLOCKED_GATES:
            result = run_gate(GATES_BY_KEY[key], context)
            assert result.status == STATUS_BLOCKED
            assert result.observed == "未测量"
            assert "标注" in result.detail

    def test_the_four_statuses_are_distinguishable(self):
        assert len(set(ALL_STATUSES)) == 4
        run = GateRun(results=())
        for status in ALL_STATUSES:
            assert len(run.by_status(status)) == 0
        assert STATUS_PASS != STATUS_FAIL != STATUS_SKIP != STATUS_BLOCKED

    def test_summary_counts_the_three_outcomes_separately(self, tmp_path):
        context = GateContext(corpus_root=tmp_path / "absent", run_tests=False)
        run = run_gates(context)
        payload = run.to_dict()
        counts = payload["counts"]
        assert set(counts) == set(ALL_STATUSES)
        assert counts[STATUS_PASS] == 0
        assert counts[STATUS_BLOCKED] == len(BLOCKED_GATES)
        assert counts[STATUS_SKIP] >= 1
        # A skipped gate must be absent from the pass list in the artifact too,
        # not merely labelled differently in prose.
        passed_keys = {
            item["gate"] for item in payload["results"] if item["status"] == STATUS_PASS
        }
        skipped_keys = {
            item["gate"] for item in payload["results"] if item["status"] == STATUS_SKIP
        }
        assert not (passed_keys & skipped_keys)

    def test_the_rendered_summary_never_calls_a_skip_a_pass(self, tmp_path):
        context = GateContext(corpus_root=tmp_path / "absent", run_tests=False)
        text = render_text(run_gates(context))
        assert "SKIP" in text
        assert "BLOCK" in text
        for line in text.splitlines():
            if "SKIP" in line:
                assert "PASS" not in line

    def test_a_failure_is_the_first_gate_row_not_the_last(self):
        """The reader scanning top-down must meet the failure first.

        The sort used to be keyed on ``ALL_STATUSES``, whose order happens to
        start with ``pass`` -- so five green rows were printed above the one
        red row, and the two blocked ones below that. The module defines
        ``STATUS_ORDER`` for exactly this ("what failed, then what did not run")
        and then sorted by the other tuple. A CI log is read top-down; the
        failure cannot be the last thing in the table.
        """
        results = (
            GateResult(key="a_pass", title="p", status=STATUS_PASS, requirement="",
                       threshold="", observed="1.0"),
            GateResult(key="z_fail", title="f", status=STATUS_FAIL, requirement="",
                       threshold="", observed="0.0", reason="坏了"),
            GateResult(key="m_skip", title="s", status=STATUS_SKIP, requirement="",
                       threshold="", observed="未测量"),
            GateResult(key="b_block", title="b", status=STATUS_BLOCKED, requirement="",
                       threshold="", observed="未测量"),
        )
        text = render_text(GateRun(results=results))
        # Read the table rows back out, so this asserts the rendered order
        # rather than the order the fixtures happened to be written in.
        rows = [
            key
            for line in text.splitlines()
            if line.strip().startswith("| `")
            for key in ("a_pass", "z_fail", "m_skip", "b_block")
            if f"`{key}`" in line
        ]
        assert rows == ["z_fail", "b_block", "m_skip", "a_pass"], rows


class TestGateEvidenceStillExists:
    """The two proxy gates name other people's tests; those names must hold.

    ``local_only_cloud_egress`` and ``task_recovery`` do not measure anything
    themselves -- they run a fixed list of node ids and count what passed. That
    is the right design (a second implementation of "zero egress" would be a
    third thing to keep in sync), but it moves the risk somewhere the gate
    cannot see: if a node id is *renamed*, the gate skips and says so; if one is
    *deleted*, the denominator quietly shrinks and the gate passes over a
    smaller population while still reporting full coverage.

    A rename is already handled. This is the other half, and it costs a file
    read rather than a test run: every node id must still name something real in
    the test file it points at.
    """

    @pytest.mark.parametrize(
        "node_ids",
        [EGRESS_NODE_IDS, RECOVERY_SQLITE_NODES, RECOVERY_POSTGRES_NODES],
        ids=["egress", "recovery-sqlite", "recovery-postgres"],
    )
    def test_every_node_id_names_a_test_that_still_exists(self, node_ids):
        sources: dict[str, str] = {}
        for node_id in node_ids:
            path, _, selector = node_id.partition("::")
            if not selector:
                pytest.fail(f"node id 没有选择器：{node_id}")
            if path not in sources:
                sources[path] = (PROJECT_ROOT / path).read_text(encoding="utf-8")
            body = sources[path]
            for part in selector.split("::"):
                # A class selector or a function selector; both appear verbatim
                # in the file as a definition or a decorator.
                assert f"class {part}" in body or f"def {part}" in body, (
                    f"{node_id} 指向的 {part!r} 在 {path} 里不存在——"
                    "门禁会在更小的集合上通过而不自知。"
                )


# --------------------------------------------------------------------------
# Broken fixtures fail
# --------------------------------------------------------------------------


class TestBrokenFixtureFails:
    """Every executable gate goes red on a fixture built to break it."""

    def test_locator_gate_fails_when_an_element_cannot_be_pointed_at(self):
        broken = ParsedDocument(
            source_filename="broken.pdf",
            page_count=1,
            elements=[_document().elements[0]],
            warnings=["legacy PDF 解析只能定位到页，无法提供页内坐标（bbox）。"],
        )
        broken.elements[0].locator = SourceLocator(kind="unknown")
        result = run_gate(
            GATES_BY_KEY[GATE_LOCATOR], GateContext().with_documents([broken])
        )
        assert result.status == STATUS_FAIL
        assert "低于阈值" in result.reason

    def test_locator_gate_fails_on_an_undeclared_missing_bbox(self):
        # A PDF element the plan expects to carry a rectangle, with none and no
        # warning: the reader clicks the citation and finds nothing, and nothing
        # on screen said that would happen.
        undeclared = _document(warnings=())
        result = run_gate(
            GATES_BY_KEY[GATE_LOCATOR], GateContext().with_documents([undeclared])
        )
        assert result.status == STATUS_FAIL
        assert "bbox" in result.detail or "bbox" in "".join(result.evidence)

    def test_locator_gate_passes_when_the_loss_is_declared(self):
        declared = _document(warnings=(
            "legacy PDF 解析只能定位到页，无法提供页内坐标（bbox）。",
        ))
        result = run_gate(
            GATES_BY_KEY[GATE_LOCATOR], GateContext().with_documents([declared])
        )
        assert result.status == STATUS_PASS

    def test_locator_gate_passes_when_the_bbox_is_present(self):
        result = run_gate(
            GATES_BY_KEY[GATE_LOCATOR], GateContext().with_documents([_document(with_bbox=True)])
        )
        assert result.status == STATUS_PASS

    def test_a_document_with_no_elements_cannot_resolve_a_citation(self):
        """A document that produced no citable content counts as unresolvable.

        Mixed in with a healthy document so the gate has samples at all: on its
        own a corpus of empty documents reports "no samples" and skips, which is
        a different (also correct) answer.
        """
        empty = _document(filename="scan.pdf", covered_pages=())
        assert empty.body_elements == []
        result = run_gate(
            GATES_BY_KEY[GATE_LOCATOR],
            GateContext().with_documents([_document(with_bbox=True), empty]),
        )
        # Element coverage is 1.0 -- every element that exists resolves -- and
        # the gate is still red. That gap between the two numbers is the bug.
        assert result.status == STATUS_FAIL
        assert "1/2" in result.evidence[0] or "1/2" in "".join(result.evidence)
        assert "零正文元素" in "".join(result.evidence)

    def test_one_empty_document_fails_a_corpus_whose_elements_all_resolve(self):
        """The pooled element ratio alone would be 1.0000 here. That is the point."""
        good = _document(filename="good.pdf", with_bbox=True)
        empty = _document(filename="scan.pdf", covered_pages=())
        result = run_gate(
            GATES_BY_KEY[GATE_LOCATOR], GateContext().with_documents([good, empty])
        )
        # 1 of 2 documents resolves and 0 undeclared bbox gaps, so the combined
        # figure is 0.75 against a 100% threshold.
        assert result.status == STATUS_FAIL
        assert "0.75" in result.observed

    def test_the_locator_gate_reports_the_parser_it_measured(self):
        """A number from the fallback parser must not read as the product's.

        The corpus gates default to ``legacy`` because it needs no optional
        dependency and runs in ~80s, but ``PARSER_PRIMARY`` defaults to
        ``docling`` and the bake-off chose it. So the parser is named in the
        evidence; otherwise "100% locator coverage" is a claim about a parser
        the product does not primarily use.
        """
        context = GateContext(parsers=("legacy",)).with_documents(
            [_document(with_bbox=True)]
        )
        result = run_gate(GATES_BY_KEY[GATE_LOCATOR], context)
        assert result.status == STATUS_PASS
        assert any("legacy" in line for line in result.evidence)

    def test_truncation_gate_fails_on_a_silent_truncation(self):
        # Nine declared pages, one page of text, no warning and no flag saying
        # so. This is the exact shape the plan calls "静默截断".
        silent = _document(page_count=10, covered_pages=(1,), warnings=())
        result = run_gate(
            GATES_BY_KEY[GATE_TRUNCATION], GateContext().with_documents([silent])
        )
        assert result.status == STATUS_FAIL
        assert "静默截断" in "".join(result.evidence) or result.notes

    def test_truncation_gate_passes_when_the_same_parse_is_warned(self):
        visible = _document(
            page_count=10,
            covered_pages=(1,),
            warnings=("解析质量为 fail：page_coverage",),
        )
        result = run_gate(
            GATES_BY_KEY[GATE_TRUNCATION], GateContext().with_documents([visible])
        )
        assert result.status == STATUS_PASS

    def test_a_capability_notice_is_not_a_truncation_warning(self):
        """The bbox declaration must not launder a silent truncation.

        "Did anyone find out" was implemented as ``bool(document.warnings)``.
        On the real corpus that is not a test of anything: 37 of 52 documents
        carry exactly one warning, the legacy adapter's bbox notice, emitted on
        every PDF whether or not anything went wrong. So a PDF that lost nine
        of its ten pages scored as *visible* on the strength of a sentence about
        highlighting, and the gate passed. A capability notice is true of every
        document the parser handles; it cannot be evidence that anything was
        reported about this one.
        """
        silent = _document(
            page_count=10,
            covered_pages=(1,),
            warnings=("legacy PDF 解析只能定位到页，无法提供页内坐标（bbox）。",),
        )
        result = run_gate(
            GATES_BY_KEY[GATE_TRUNCATION], GateContext().with_documents([silent])
        )
        assert result.status == STATUS_FAIL, result.observed
        assert "能力声明" in "".join(result.notes)

    def test_a_real_warning_beside_a_capability_notice_still_counts_as_visible(self):
        """The notices are subtracted, not replaced.

        A document carrying both the bbox declaration and a real
        "page 3 has no extractable text" is visible, and it is the second
        warning that makes it so. Were this to regress, the fix for the test
        above would have over-corrected into a gate that ignores warnings.
        """
        both = _document(
            page_count=10,
            covered_pages=(1,),
            warnings=(
                "legacy PDF 解析只能定位到页，无法提供页内坐标（bbox）。",
                "PDF 第 3 页没有可提取的文本。",
            ),
        )
        result = run_gate(
            GATES_BY_KEY[GATE_TRUNCATION], GateContext().with_documents([both])
        )
        assert result.status == STATUS_PASS

    def test_truncation_gate_passes_on_a_healthy_document(self):
        result = run_gate(
            GATES_BY_KEY[GATE_TRUNCATION], GateContext().with_documents([_document()])
        )
        assert result.status == STATUS_PASS

    def test_a_recorded_defect_suppresses_only_its_own_rule(self):
        # A deck trips page_coverage and text_density purely because the
        # quality rules divide by slide count; both are recorded defects. The
        # suppression must be narrow -- it is not a switch that turns the gate
        # off.
        deck = _document(filename="deck.pptx", page_count=6, covered_pages=(1, 2, 3, 4, 5, 6),
                         kind="slide", warnings=())
        for element in deck.elements:
            element.locator = SourceLocator(kind="slide", slide=element.order + 1)
        result = run_gate(
            GATES_BY_KEY[GATE_TRUNCATION], GateContext().with_documents([deck])
        )
        assert result.status == STATUS_PASS
        assert any("已登记缺陷" in note for note in result.notes)

    def test_a_rule_that_is_not_a_known_defect_is_never_excused(self):
        # The suppression is keyed on a specific rule_id. Any other failing
        # rule -- the ones that actually catch truncated PDFs and scans -- is
        # never discounted, so a deck that is genuinely broken still fails.
        deck = _document(filename="deck.pptx", page_count=6, kind="slide", warnings=())
        for element in deck.elements:
            element.locator = SourceLocator(kind="slide", slide=element.order + 1)
        for rule_id in ("page_coverage", "text_density"):
            assert defect_explains(deck, rule_id) is not None, rule_id
        for rule_id in ("garbled_characters", "blank_page_ratio", "locator_coverage"):
            assert defect_explains(deck, rule_id) is None, rule_id

    def test_the_suppression_is_strict_and_names_where_to_fix(self):
        # If quality.py's _page_of ever learns to read slide locators, these
        # entries stop applying and the gate goes red again. That is the
        # contract: a recorded defect cannot quietly outlive its cause.
        for defect in KNOWN_DEFECTS:
            assert defect.defect_id
            assert defect.location.startswith("paperwhisperer/")
            assert defect.summary
        assert {defect.defect_id for defect in KNOWN_DEFECTS} == {
            "page-coverage-ignores-non-pdf-locators",
            "text-density-threshold-is-paper-shaped",
        }

    def test_a_pdf_is_never_excused_by_the_non_pdf_defects(self):
        # The predicates are about format, not about whether the gate would
        # prefer the document to fail.
        pdf = _document(filename="paper.pdf", page_count=6, covered_pages=(1,),
                        warnings=())
        for defect in KNOWN_DEFECTS:
            assert defect.applies(pdf) is False, defect.defect_id

    def test_egress_gate_fails_when_the_egress_tests_fail(self):
        context = GateContext(test_runner=lambda node_ids: (0, 4, "4 failed"))
        result = run_gate(GATES_BY_KEY[GATE_EGRESS], context)
        assert result.status == STATUS_FAIL

    def test_egress_gate_skips_when_the_tests_did_not_run(self):
        # The delegated node ids collecting nothing is not a pass. If the test
        # that proves zero egress is renamed, this gate must say "not run".
        context = GateContext(test_runner=lambda node_ids: (0, 0, "no tests ran"))
        result = run_gate(GATES_BY_KEY[GATE_EGRESS], context)
        assert result.status == STATUS_SKIP
        assert "未执行" in result.reason

    def test_recovery_gate_fails_when_a_recovery_test_fails(self):
        context = GateContext(
            recovery_tiers=("personal",),
            test_runner=lambda node_ids: (1, 2, "2 failed"),
        )
        result = run_gate(GATES_BY_KEY[GATE_RECOVERY], context)
        assert result.status == STATUS_FAIL

    def test_recovery_gate_reports_an_absent_postgres_as_not_run(self):
        # A DSN-less environment must not silently look like the hosted tier
        # passed.
        context = GateContext(
            recovery_tiers=("personal", "hosted"),
            test_runner=lambda node_ids: (3, 0, "3 passed"),
        )
        result = run_gate(GATES_BY_KEY[GATE_RECOVERY], context)
        assert result.status == STATUS_PASS
        assert any(PG_DSN_ENV in note for note in result.notes)

    def test_a_broken_runner_fails_the_gate_rather_than_passing_it(self):
        def exploding(context: GateContext) -> GateMeasurement:
            raise RuntimeError("runner 崩了")

        result = run_gate(_spec_with(exploding), GateContext().with_documents([]))
        assert result.status == STATUS_FAIL
        assert "runner" in result.reason

    def test_a_runner_that_returns_nothing_fails_rather_than_passing(self):
        def empty_runner(context: GateContext) -> GateMeasurement:
            return GateMeasurement()

        result = run_gate(_spec_with(empty_runner), GateContext().with_documents([]))
        assert result.status == STATUS_FAIL
        assert "未产出测量值" in result.reason or "无结果" in result.observed

    def test_a_runner_that_asks_to_skip_is_a_skip_not_a_fail(self):
        def skipping(context: GateContext) -> GateMeasurement:
            raise MeasurementSkip("fixture 缺失", "说明")

        result = run_gate(_spec_with(skipping), GateContext())
        assert result.status == STATUS_SKIP
        assert result.reason == "fixture 缺失"


# --------------------------------------------------------------------------
# The direct-quote gate's negative control is not vacuous
# --------------------------------------------------------------------------


class TestDirectQuoteControlIsNotVacuous:
    """A gate that only checked "real quotes align" would prove very little."""

    def test_genuine_windows_align_exactly(self):
        text = _DENSE
        element = Element(
            role="paragraph",
            text=text,
            locator=SourceLocator(kind="pdf_page", page=1),
        )
        result = verify_quote(text[:40], element)
        assert result.aligned
        assert result.mode is AlignmentMode.EXACT

    def test_the_fabrication_is_a_near_miss_not_an_unrelated_string(self):
        # If the fabricated quote bore no resemblance to the source, any
        # substring search would reject it and the negative control would
        # prove nothing about a sloppy verifier.
        from eval.gates import _mutate  # noqa: PLC0415

        source = "accuracy improved from 72.4 to 79.1 on the benchmark set"
        mutated = _mutate(source)
        assert mutated
        assert mutated != source
        overlap = len(set(source.split()) & set(mutated.split()))
        assert overlap >= len(set(source.split())) - 1

    def test_a_near_miss_is_rejected_and_degrades_without_quote_marks(self):
        from eval.gates import _mutate  # noqa: PLC0415

        source = "accuracy improved from 72.4 to 79.1 on the benchmark set"
        element = Element(
            role="paragraph",
            text=source,
            locator=SourceLocator(kind="pdf_page", page=1),
        )
        result = verify_quote(_mutate(source), element)
        assert not result.aligned
        assert result.outcome in {
            AlignmentOutcome.NOT_FOUND,
            AlignmentOutcome.SPANS_ELEMENTS,
        }
        assert result.render().uses_quotation_marks is False

    def test_the_gate_names_the_fixtures_it_needs(self):
        assert len(DIRECT_QUOTE_FIXTURES) == 5
        for relative in DIRECT_QUOTE_FIXTURES:
            assert relative.endswith(".pdf")

    def test_the_direct_quote_gate_skips_without_its_fixtures(self, tmp_path):
        context = GateContext(corpus_root=tmp_path / "absent")
        result = run_gate(GATES_BY_KEY[GATE_DIRECT_QUOTE], context)
        assert result.status == STATUS_SKIP


# --------------------------------------------------------------------------
# The label harness
# --------------------------------------------------------------------------


class TestRecallScorer:
    """Recall@K and the two-axis breakdown the plan requires."""

    def _labels(self) -> list[QueryLabel]:
        return [
            QueryLabel("q1", "a.pdf", "多头注意力", "zh", "prose", ("el_zh_1",)),
            QueryLabel("q2", "b.pdf", "multi head attention", "en", "prose", ("el_en_1",)),
            QueryLabel("q3", "c.pdf", "GLUE results table", "en", "table", ("el_tbl_1",)),
            QueryLabel("q4", "d.pdf", "the horizon radius", "en", "formula", ("el_f_1",)),
        ]

    def test_a_hit_at_rank_one_counts(self):
        labels = self._labels()
        measurement, _ = score_recall_at_k(labels, {"q1": ["el_zh_1"]}, k=5)
        assert measurement.rate == pytest.approx(0.25)

    def test_a_hit_beyond_k_does_not_count(self):
        labels = self._labels()
        retrieved = {"q1": ["x"] * 5 + ["el_zh_1"]}
        measurement, _ = score_recall_at_k(labels, retrieved, k=5)
        assert measurement.numerator == 0

    def test_any_one_relevant_element_is_enough(self):
        labels = self._labels()
        measurement, _ = score_recall_at_k(labels, {"q1": ["other", "el_zh_1"]}, k=5)
        assert measurement.numerator == 1

    def test_the_threshold_is_90_percent_not_100(self):
        labels = self._labels()
        retrieved = {label.query_id: list(label.relevant_element_ids) for label in labels[:3]}
        measurement, _ = score_recall_at_k(labels, retrieved, k=5)
        assert measurement.rate == pytest.approx(0.75)
        spec = GATES_BY_KEY[GATE_RECALL]
        assert decide_status(spec, measurement) == STATUS_FAIL

    def test_full_recall_passes(self):
        labels = self._labels()
        retrieved = {label.query_id: list(label.relevant_element_ids) for label in labels}
        measurement, _ = score_recall_at_k(labels, retrieved, k=5)
        assert decide_status(GATES_BY_KEY[GATE_RECALL], measurement) == STATUS_PASS

    def test_chinese_and_english_are_broken_out_separately(self):
        labels = self._labels()
        retrieved = {label.query_id: list(label.relevant_element_ids) for label in labels}
        _, breakdown = score_recall_at_k(labels, retrieved, k=5)
        assert "lang:zh" in breakdown
        assert "lang:en" in breakdown
        assert breakdown["lang:zh"]["total"] == 1
        assert breakdown["lang:en"]["total"] == 3

    def test_table_formula_and_prose_are_broken_out_separately(self):
        labels = self._labels()
        retrieved = {label.query_id: list(label.relevant_element_ids) for label in labels}
        _, breakdown = score_recall_at_k(labels, retrieved, k=5)
        assert breakdown["kind:prose"]["total"] == 2
        assert breakdown["kind:table"]["total"] == 1
        assert breakdown["kind:formula"]["total"] == 1

    def test_the_two_axes_are_independent_not_merged(self):
        # "中英、表格/公式/正文分开统计" is two axes. Collapsing them into one
        # bucket would let a Chinese-prose query and an English-table query
        # average into a single number and hide whichever one is failing.
        labels = self._labels()
        _, breakdown = score_recall_at_k(labels, {}, k=5)
        assert len(breakdown) == 5

    def test_a_miss_in_one_language_shows_up_in_that_language(self):
        labels = self._labels()
        retrieved = {
            "q1": [],  # the only Chinese query misses
            "q2": ["el_en_1"],
            "q3": ["el_tbl_1"],
            "q4": ["el_f_1"],
        }
        _, breakdown = score_recall_at_k(labels, retrieved, k=5)
        assert breakdown["lang:zh"]["recall"] == pytest.approx(0.0)
        assert breakdown["lang:en"]["recall"] == pytest.approx(1.0)

    def test_empty_labels_give_no_rate_rather_than_a_perfect_one(self):
        measurement, breakdown = score_recall_at_k([], {}, k=5)
        assert measurement.rate is None
        assert breakdown == {}
        assert decide_status(GATES_BY_KEY[GATE_RECALL], measurement) == STATUS_SKIP


class TestPrecisionScorer:
    """Citation precision, from human judgements only."""

    def _claims(self, supported: tuple[bool, ...]) -> list[ClaimLabel]:
        return [
            ClaimLabel(f"c{index}", "q1", f"quote {index}", f"el_{index}", value)
            for index, value in enumerate(supported)
        ]

    def test_full_support_meets_the_95_percent_bar(self):
        measurement = score_precision(self._claims((True,) * 20))
        assert measurement.rate == pytest.approx(1.0)
        assert decide_status(GATES_BY_KEY[GATE_PRECISION], measurement) == STATUS_PASS

    def test_one_unsupported_claim_in_a_hundred_still_passes(self):
        measurement = score_precision(self._claims((False,) + (True,) * 99))
        assert measurement.rate == pytest.approx(0.99)
        assert decide_status(GATES_BY_KEY[GATE_PRECISION], measurement) == STATUS_PASS

    def test_ninety_five_percent_exactly_passes(self):
        measurement = score_precision(self._claims((False,) * 5 + (True,) * 95))
        assert measurement.rate == pytest.approx(0.95)
        assert decide_status(GATES_BY_KEY[GATE_PRECISION], measurement) == STATUS_PASS

    def test_just_under_95_percent_fails(self):
        measurement = score_precision(self._claims((False,) * 6 + (True,) * 94))
        assert decide_status(GATES_BY_KEY[GATE_PRECISION], measurement) == STATUS_FAIL

    def test_no_claims_is_a_skip_not_a_pass(self):
        measurement = score_precision([])
        assert decide_status(GATES_BY_KEY[GATE_PRECISION], measurement) == STATUS_SKIP


# --------------------------------------------------------------------------
# Label loading
# --------------------------------------------------------------------------


GOOD_LABELS = """\
schema: paperwhisperer-eval-labels/0.1
corpus_version: sha256:abc
queries:
- id: q1
  document: digital_single_col/attention.pdf
  query: 多头注意力如何计算
  language: zh
  kind: prose
  relevant_element_ids: [el_1, el_2]
claims:
- id: c1
  query_id: q1
  quote: "some quote"
  element_id: el_1
  supported: true
"""


class TestLabelLoading:
    """A label set that loads with rows dropped would report a false number."""

    def test_a_missing_label_file_raises_rather_than_scoring(self, tmp_path):
        with pytest.raises(FileNotFoundError) as excinfo:
            load_labels(tmp_path / "absent.yaml")
        assert "标注集不存在" in str(excinfo.value)

    def test_a_well_formed_label_file_loads(self, tmp_path):
        path = tmp_path / "labels.yaml"
        path.write_text(GOOD_LABELS, encoding="utf-8")
        labels = load_labels(path)
        assert len(labels.queries) == 1
        assert len(labels.claims) == 1
        assert labels.queries[0].language == "zh"
        assert labels.queries[0].relevant_element_ids == ("el_1", "el_2")

    def test_an_unknown_schema_is_refused(self, tmp_path):
        path = tmp_path / "labels.yaml"
        path.write_text(GOOD_LABELS.replace("paperwhisperer-eval-labels/", "other/"), "utf-8")
        with pytest.raises(ValueError, match="schema"):
            load_labels(path)

    def test_a_query_without_ground_truth_is_refused(self, tmp_path):
        # A query with no relevant element would inflate the denominator's
        # share of unreachable queries rather than being honestly unanswerable.
        text = GOOD_LABELS.replace("relevant_element_ids: [el_1, el_2]", "relevant_element_ids: []")
        path = tmp_path / "labels.yaml"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match="relevant_element_ids"):
            load_labels(path)

    def test_an_unclassifiable_language_is_refused(self, tmp_path):
        path = tmp_path / "labels.yaml"
        path.write_text(GOOD_LABELS.replace("language: zh", "language: french"), "utf-8")
        with pytest.raises(ValueError, match="language"):
            load_labels(path)

    def test_an_unclassifiable_kind_is_refused(self, tmp_path):
        path = tmp_path / "labels.yaml"
        path.write_text(GOOD_LABELS.replace("kind: prose", "kind: diagram"), "utf-8")
        with pytest.raises(ValueError, match="kind"):
            load_labels(path)

    def test_a_claim_without_a_human_verdict_is_refused(self, tmp_path):
        # Precision must be a person's judgement. A claim with no verdict is the
        # system grading itself, which is exactly what must not happen.
        text = GOOD_LABELS.replace("  supported: true\n", "")
        path = tmp_path / "labels.yaml"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match="supported"):
            load_labels(path)

    def test_a_query_without_an_id_is_refused(self, tmp_path):
        text = GOOD_LABELS.replace("- id: q1", "- id: ''")
        path = tmp_path / "labels.yaml"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match="id"):
            load_labels(path)


# --------------------------------------------------------------------------
# Release mode
# --------------------------------------------------------------------------


class TestReleaseMode:
    """CI rules and release rules are different questions about the same data."""

    def _result(self, key: str, status: str) -> GateResult:
        return GateResult(
            key=key,
            title=key,
            status=status,
            requirement="",
            threshold="",
            observed="",
        )

    def test_an_all_pass_run_is_ready(self):
        run = GateRun(results=tuple(
            self._result(key, STATUS_PASS) for key in ALL_GATES
        ))
        assert_release_ready(run)

    def test_a_failed_gate_blocks_the_release(self):
        run = GateRun(results=(
            self._result(GATE_DIRECT_QUOTE, STATUS_FAIL),
        ))
        with pytest.raises(ReleaseNotReady):
            assert_release_ready(run)

    def test_a_skipped_gate_blocks_the_release(self):
        run = GateRun(results=(self._result(GATE_DIRECT_QUOTE, STATUS_SKIP),))
        with pytest.raises(ReleaseNotReady, match=GATE_DIRECT_QUOTE):
            assert_release_ready(run)

    def test_a_blocked_gate_blocks_the_release(self):
        # This is what makes the current state honest: today, two of seven
        # gates are blocked, so --release must refuse.
        run = GateRun(results=tuple(
            self._result(key, STATUS_PASS)
            for key in ALL_GATES if key not in BLOCKED_GATES
        ) + tuple(self._result(key, STATUS_BLOCKED) for key in BLOCKED_GATES))
        with pytest.raises(ReleaseNotReady) as excinfo:
            assert_release_ready(run)
        message = str(excinfo.value)
        assert GATE_PRECISION in message
        assert GATE_RECALL in message

    def test_the_build_rule_does_not_block_on_a_skip(self):
        # The asymmetry the whole design rests on: CI must not go red because
        # the corpus is gitignored, but a release must not ship green either.
        run = GateRun(results=(self._result(GATE_DIRECT_QUOTE, STATUS_SKIP),))
        assert run.exit_code == 0
        with pytest.raises(ReleaseNotReady):
            assert_release_ready(run)


# --------------------------------------------------------------------------
# Corpus accounting
# --------------------------------------------------------------------------


class TestCorpusAccounting:
    """Corpus size is reported, never gated -- it is the owner's data task."""

    def test_the_current_corpus_is_below_the_published_target(self):
        from eval.gates import CORPUS_TARGET_MAX, corpus_stats  # noqa: PLC0415

        stats = corpus_stats(GateContext())
        if not stats.get("available"):
            pytest.skip("语料不在本机，无法统计")
        assert stats["target_max"] == CORPUS_TARGET_MAX
        assert stats["at_target"] is False
        assert stats["present"] < stats["target_min"]

    def test_the_manifest_version_is_carried_into_the_report(self):
        from eval.gates import corpus_stats  # noqa: PLC0415

        stats = corpus_stats(GateContext())
        if not stats.get("available"):
            pytest.skip("语料不在本机，无法统计")
        assert stats["corpus_version"].startswith("sha256:")

    def test_a_shortfall_does_not_fail_a_gate(self, tmp_path):
        # A gate that failed because 100 PDFs have not been downloaded would be
        # measuring a backlog, not the product.
        context = GateContext(corpus_root=tmp_path / "absent", run_tests=False)
        run = run_gates(context)
        assert run.failed == ()
