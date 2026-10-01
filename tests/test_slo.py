"""SLO 的算术与它的证据。

T4.6 的验收标准是「**SLO 有数据支撑且可告警**」。这句话拆成两半，各有
一种失败方式，而本文件对两半都设了闸：

* **有数据支撑** —— 失败方式是"目标写得很漂亮，但背后没有一次测量"。
  本文件里最重的一个测试 :func:`test_every_measured_row_recomputes_from_its_artifact`
  从仓内那份测量产物**独立重算**每一个标为 ``measured`` 的数字。表里写 379，
  产物里重算出来就必须也是 379。这让"实测"从一个形容词变成一个可证伪的断言。
* **可告警** —— 失败方式是"规则写了，但没有一次真实读数能让它响"。所以积压
  与限速的判定都对着**真的** T4.1 指标对象跑，而不是对着手写的字典。

**为什么"实测"要在测试里被重算，而不只是被读到。** 因为一个数字只要在代码里
出现一次，半年后就没有人能说清它是量出来的还是定出来的。把重算写进测试，
"这个数字还站得住吗"就从一个人的记忆变成一次 ``pytest``。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from paperwhisperer.core import slo
from paperwhisperer.core.slo import (
    MEASURE_REF,
    MIN_MEASURED_SAMPLES,
    ErrorBudget,
    LatencyTarget,
    ProviderRateLimitRule,
    SLOSample,
    alerts_from_registry,
    evaluate_backlog,
    evaluate_provider_rate_limits,
    histogram_quantile,
    nearest_rank_percentile,
    rate_limit_signals_from,
    slo_report,
    target_for_parse,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------
# 分位数：对着手算值
# --------------------------------------------------------------------------


class TestNearestRankPercentile:
    """nearest-rank 的定义是"排序后第 ceil(q*n) 个"，没有插值系数。"""

    def test_it_takes_the_ceil_rank_not_an_interpolation(self):
        # 1..10，q=0.95 -> ceil(9.5)=10 -> 第 10 个 = 10。
        assert nearest_rank_percentile(range(1, 11), 0.95) == 10
        # n=20, q=0.95 -> ceil(19)=19 -> 第 19 个 = 19（**不是** 19.05）。
        assert nearest_rank_percentile(range(1, 21), 0.95) == 19

    def test_every_returned_value_is_an_observed_value(self):
        """这个函数存在的理由：P95 一定能在原始样本里指到某一条。"""
        values = [17, 3, 900, 42, 8, 61, 12, 5, 990, 33]
        result = nearest_rank_percentile(values, 0.95)
        assert result in values

    def test_input_order_does_not_matter(self):
        assert nearest_rank_percentile([9, 1, 5, 3, 7], 0.5) == nearest_rank_percentile(
            [1, 3, 5, 7, 9], 0.5
        )

    def test_a_single_sample_is_its_own_p95(self):
        assert nearest_rank_percentile([42], 0.95) == 42

    def test_no_samples_is_zero_rather_than_an_exception(self):
        """告警路径上抛异常会把一次"没数据"变成一次 500。"""
        assert nearest_rank_percentile([], 0.95) == 0.0

    @pytest.mark.parametrize(
        ("quantile", "expected"),
        [(0.0, 1), (0.5, 5), (1.0, 10)],
    )
    def test_the_two_ends_and_the_middle(self, quantile, expected):
        assert nearest_rank_percentile(range(1, 11), quantile) == expected

    def test_an_out_of_range_quantile_is_clamped_not_rejected(self):
        """一个算 SLA 的函数不该因为调用方传了 1.5 就崩掉。"""
        assert nearest_rank_percentile(range(1, 11), 1.5) == 10
        assert nearest_rank_percentile(range(1, 11), -3) == 1


class TestHistogramQuantile:
    """Prometheus 口径的**估计**。它与精确分位数的差就是它的误差上限。"""

    def test_it_interpolates_inside_the_target_bucket(self):
        # 累积计数：桶 (1,10] 装了全部 20 个样本。q=0.5 -> rank=10 -> 桶的正中。
        assert histogram_quantile([1, 10, 100], [0, 20, 20], 0.5) == pytest.approx(5.5)

    def test_counts_are_cumulative_not_per_bucket(self):
        """Prometheus 口径：``counts`` 是累积值，末位才是总量。

        这个区别是本函数最容易用错的地方——传入每桶计数会让
        ``cumulative[-1]`` 取到最后一个桶的增量，于是总量算错、结果全错。
        """
        assert histogram_quantile([1, 10, 100], [0, 20, 20], 0.5) != histogram_quantile(
            [1, 10, 100], [0, 20, 0], 0.5
        )

    def test_below_the_first_bucket_returns_the_first_upper_edge(self):
        assert histogram_quantile([5, 10], [3, 3], 0.5) == 5

    def test_landing_in_the_inf_bucket_returns_the_last_finite_edge(self):
        """朝 ``inf`` 插值会让告警渲染成 "inf"，并让任何乘法变成 nan。"""
        result = histogram_quantile([1, 10, float("inf")], [0, 5, 10], 0.95)
        assert math.isfinite(result)
        assert result == 10

    def test_empty_or_mismatched_input_is_zero(self):
        assert histogram_quantile([], [], 0.95) == 0.0
        assert histogram_quantile([1, 2], [1], 0.95) == 0.0
        assert histogram_quantile([1, 2], [0, 0], 0.95) == 0.0

    def test_it_interpolates_at_a_fraction_that_is_not_a_half(self):
        # 桶 (10,100] 装 10 个（累积 20 -> 30）。q=0.95 -> rank=28.5 -> 落在该桶。
        # 10 + (100-10) * (28.5-20)/10 = 86.5。
        assert histogram_quantile([1, 10, 100, float("inf")], [0, 20, 30, 30], 0.95) == (
            pytest.approx(86.5)
        )


# --------------------------------------------------------------------------
# 目标表：每一个"实测"都必须能被重算
# --------------------------------------------------------------------------


def _bucket_of(pages: int) -> str:
    """测试自己的分桶实现。刻意不复用 slo.classify_pages。

    如果这里 import 生产的分桶函数，那么"分桶写错了"这件事在测试与被测代码里
    会同时发生、同时为真，于是测试永远绿。分桶边界是本表最容易被误抄的东西
    （1-10 / 11-30 / 31-80 / 81-200 / 201+），所以这里手写一遍。
    """
    count = max(1, int(pages or 0))
    for name, low, high in (
        ("s", 1, 10),
        ("m", 11, 30),
        ("l", 31, 80),
        ("xl", 81, 200),
        ("xxl", 201, None),
    ):
        if count >= low and (high is None or count <= high):
            return name
    return "xxl"


@pytest.fixture(scope="module")
def measurement_artifact() -> dict:
    path = PROJECT_ROOT / MEASURE_REF
    if not path.exists():
        pytest.fail(
            f"实测目标引用的测量产物不存在：{MEASURE_REF}。"
            "这正是本文件要防的那件事——表里标了「实测」，而支撑它的报告"
            "没有被提交进仓。请把 eval/ 一并纳入版本管理"
            "（当前它整体处于未跟踪状态），否则「实测」二字无法复核。"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _legacy_durations_by_bucket(artifact: dict) -> dict[str, list[float]]:
    """从产物里取出 legacy 一行，按页数分桶。"""
    buckets: dict[str, list[float]] = {}
    for row in artifact.get("rows", []):
        if row.get("parser") != "legacy":
            continue
        name = str(row.get("document") or "")
        if not name.lower().endswith(".pdf"):
            continue
        buckets.setdefault(_bucket_of(row.get("pages_expected")), []).append(
            float(row.get("duration_ms") or 0.0)
        )
    return buckets


class TestLatencyTargets:
    def test_every_measured_row_recomputes_from_its_artifact(self, measurement_artifact):
        """本文件最重要的一条：表里的"实测"必须能从产物里逐字重算出来。"""
        buckets = _legacy_durations_by_bucket(measurement_artifact)
        checked = 0
        for target in slo.LATENCY_TARGETS:
            if not target.is_measured:
                continue
            assert target.evidence_ref, f"{target.bucket} 标为实测却没有 evidence_ref"
            values = buckets.get(target.bucket, [])
            assert values, f"产物里找不到 {target.bucket} 桶的 legacy 样本"
            assert len(values) == target.samples, (
                f"{target.bucket} 桶表里写 n={target.samples}，"
                f"产物里数出来是 {len(values)}"
            )
            recomputed = nearest_rank_percentile(values, 0.95)
            assert recomputed == float(target.p95_ms), (
                f"{target.media_format}/{target.bucket}/{target.parser}："
                f"表里写 P95={target.p95_ms}ms，从 {MEASURE_REF} 重算是 "
                f"{recomputed}ms。两者必须一致——不一致意味着要么表过时了，"
                f"要么有人改过语料而没重跑。"
            )
            checked += 1
        assert checked >= 3, "实测行少得可疑，确认重算逻辑本身还活着"

    def test_a_measured_row_without_an_artifact_is_rejected_at_construction(self):
        """标"实测"却不指产物，构造时就要炸——不能等到有人去信它。"""
        with pytest.raises(ValueError, match="evidence_ref"):
            LatencyTarget(
                stage="parse",
                media_format="pdf",
                bucket="s",
                parser="legacy",
                p95_ms=100,
                evidence_kind=slo.EVIDENCE_MEASURED,
                evidence="在某台机器上量的",
                samples=20,
                evidence_ref="",
            )

    def test_too_few_samples_cannot_be_called_measured(self):
        """样本量不足时 P95 恒等于最大值，那不是分位数。"""
        with pytest.raises(ValueError, match="样本"):
            LatencyTarget(
                stage="parse",
                media_format="pdf",
                bucket="xxl",
                parser="legacy",
                p95_ms=8000,
                evidence_kind=slo.EVIDENCE_MEASURED,
                evidence="n=1",
                samples=1,
                evidence_ref=MEASURE_REF,
            )

    def test_an_evidence_kind_that_is_neither_measured_nor_target_is_rejected(self):
        with pytest.raises(ValueError, match="evidence_kind"):
            LatencyTarget(
                stage="parse",
                media_format="pdf",
                bucket="s",
                parser="legacy",
                p95_ms=100,
                evidence_kind="大概",
                evidence="感觉",
            )

    def test_every_row_carries_a_usable_explanation(self):
        for target in slo.LATENCY_TARGETS:
            assert target.evidence.strip(), f"{target} 没有说明依据"
            assert target.p95_ms > 0

    def test_the_table_is_honest_about_how_much_is_measured(self):
        """把"实测占比"钉住，是为了让新增的行**必须**说明自己属于哪一类。

        刻意不写成"实测行数 == 3"：那是把一个会随语料增长的数字写死。
        钉住的是**每行都有标签**这件事本身——多写一行目标没问题，
        多写一行没有依据的行才有问题。
        """
        for target in slo.LATENCY_TARGETS:
            assert target.evidence_kind in slo.EVIDENCE_KINDS

    def test_unmeasured_buckets_carry_their_measured_value_for_reference(self):
        """样本量不足的那几行，实测值仍然写在 evidence 里，不能丢失。"""
        by_key = {
            (t.media_format, t.bucket, t.parser): t
            for t in slo.LATENCY_TARGETS
            if t.stage == "parse"
        }
        xl = by_key[("pdf", "xl", "legacy")]
        assert not xl.is_measured
        assert "6575" in xl.evidence
        assert "n=2" in xl.evidence

    def test_lookup_by_format_pages_and_parser(self):
        assert target_for_parse("pdf", 5, "legacy").p95_ms == 379
        assert target_for_parse("pdf", 20, "legacy").p95_ms == 3699
        assert target_for_parse("pdf", 50, "legacy").p95_ms == 23621
        # 未知 parser 没有目标——返回 None 而不是回落到别的桶。
        assert target_for_parse("pdf", 5, "某个不存在的解析器") is None

    def test_the_measured_sample_floor_matches_the_documented_constant(self):
        assert MIN_MEASURED_SAMPLES == 5
        for target in slo.LATENCY_TARGETS:
            if target.is_measured:
                assert target.samples >= MIN_MEASURED_SAMPLES


class TestPageBuckets:
    @pytest.mark.parametrize(
        ("pages", "expected"),
        [
            (0, "s"),  # 尚未解析完的登记行
            (-5, "s"),
            (1, "s"),
            (10, "s"),
            (11, "m"),
            (30, "m"),
            (31, "l"),
            (80, "l"),
            (81, "xl"),
            (200, "xl"),
            (201, "xxl"),
            (10_000, "xxl"),
        ],
    )
    def test_boundaries(self, pages, expected):
        assert slo.classify_pages(pages).name == expected

    def test_the_buckets_are_contiguous_and_cover_everything(self):
        """不能有页数落在两个桶之间，也不可能没有桶收留。"""
        for pages in range(1, 400):
            owners = [b for b in slo.PAGE_BUCKETS if b.contains(pages)]
            assert len(owners) == 1, f"{pages} 页匹配了 {len(owners)} 个桶"


# --------------------------------------------------------------------------
# 报告：达标率与"缺目标"要分开算
# --------------------------------------------------------------------------


class TestSLOReport:
    def test_compliance_is_the_fraction_within_target(self):
        report = slo_report(
            [
                SLOSample("parse", 10, "pdf", 5, "legacy"),
                SLOSample("parse", 20, "pdf", 5, "legacy"),
                SLOSample("parse", 9_000, "pdf", 5, "legacy"),
                SLOSample("parse", 10, "pdf", 5, "legacy"),
            ]
        )
        assert report.checked == 4
        assert report.within_target == 3
        assert report.compliance == pytest.approx(0.75)
        assert not report.ok
        assert len(report.breaches) == 1

    def test_no_observations_is_perfect_compliance_not_total_failure(self):
        """0.0 会让"还没有任何观测"看起来像"全部不达标"——告警方向正好相反。"""
        report = slo_report([])
        assert report.compliance == 1.0
        assert report.ok
        assert report.checked == 0

    def test_samples_without_a_target_are_counted_separately(self):
        """把"我们没定过这个组合"算成"不达标"，看报告的人会去修那个红。"""
        report = slo_report(
            [
                SLOSample("parse", 10, "pdf", 5, "legacy"),
                SLOSample("parse", 10, "pdf", 5, "没配过的解析器"),
            ]
        )
        assert report.checked == 1
        assert report.missing_target == 1
        assert report.compliance == 1.0
        assert report.ok

    def test_a_breach_carries_the_evidence_kind_of_its_target(self):
        """告警里必须能看出这条判的是实测还是目标。"""
        report = slo_report([SLOSample("parse", 10_000, "pdf", 5, "docling")])
        assert report.breaches[0]["evidence_kind"] == slo.EVIDENCE_TARGET
        report = slo_report([SLOSample("parse", 10_000, "pdf", 5, "legacy")])
        assert report.breaches[0]["evidence_kind"] == slo.EVIDENCE_MEASURED


class TestRetrieveTargetSelection:
    """检索按语料规模选档。选错档是一个**必然误报**的规则。"""

    def test_the_bucket_follows_the_corpus_size(self):
        for chunks, bucket in ((500, "corpus-1k"), (2_000, "corpus-3k"), (5_000, "corpus-7k")):
            sample = SLOSample("retrieve", 1.0, parser="personal-sqlite", corpus_chunks=chunks)
            assert sample.target().bucket == bucket

    def test_an_unstated_corpus_size_does_not_silently_use_the_smallest_bucket(self):
        """这是本文件防的那个具体 bug：拿 1k 的标准去判 7k 的查询。"""
        sample = SLOSample("retrieve", 5.0, parser="personal-sqlite")
        assert sample.target() is None
        report = slo_report([sample])
        assert report.checked == 0
        assert report.missing_target == 1

    def test_a_corpus_beyond_the_largest_bucket_reports_a_missing_target(self):
        sample = SLOSample("retrieve", 5.0, parser="personal-sqlite", corpus_chunks=50_000)
        assert sample.target() is None

    def test_a_slow_query_against_a_large_corpus_uses_that_corpus_budget(self):
        small = SLOSample("retrieve", 150.0, parser="personal-sqlite", corpus_chunks=500)
        large = SLOSample("retrieve", 150.0, parser="personal-sqlite", corpus_chunks=5_000)
        assert small.within_target() is False  # 1k 档只有 80ms
        assert large.within_target() is True  # 7k 档有 180ms


# --------------------------------------------------------------------------
# 错误预算
# --------------------------------------------------------------------------


class TestErrorBudget:
    def test_allowed_bad_events_floor_rather_than_round(self):
        """允许 0.5 个坏事件的预算不是一个可以发放的预算。"""
        budget = ErrorBudget(objective=0.995, good=100, total=100)
        assert budget.allowed_bad == 0

    def test_the_three_numbers_come_from_one_pair_and_cannot_disagree(self):
        budget = ErrorBudget(objective=0.99, good=990, total=1000)
        assert budget.bad == 10
        assert budget.allowed_bad == 10
        assert budget.remaining == 0
        assert budget.exhausted
        assert budget.consumed == 1.0

    def test_overspending_goes_negative_rather_than_clamping(self):
        """负数就是已经超支；钳成 0 会让"刚好用完"和"用了两倍"看起来一样。"""
        budget = ErrorBudget(objective=0.99, good=980, total=1000)
        assert budget.remaining == -10
        assert budget.exhausted
        assert budget.consumed == 1.0

    def test_burn_rate_is_the_multiple_of_expected_consumption(self):
        # 目标 99%：1000 次里预期坏 10 次。实际坏 20 -> 2 倍速。
        assert ErrorBudget(objective=0.99, good=980, total=1000).burn_rate == pytest.approx(2.0)
        assert ErrorBudget(objective=0.99, good=995, total=1000).burn_rate == pytest.approx(0.5)

    def test_burn_rate_is_window_independent(self):
        """这才是小窗口上唯一可用的数：同样的比例，不同的窗口，同样读。"""
        small = ErrorBudget(objective=0.99, good=8, total=10)
        large = ErrorBudget(objective=0.99, good=800, total=1000)
        assert small.burn_rate == pytest.approx(large.burn_rate)

    def test_no_events_means_no_burn(self):
        """把"没有观测"报成"烧穿预算"是这类预算最常见的误报。"""
        budget = ErrorBudget(objective=0.99, good=0, total=0)
        assert budget.burn_rate == 0.0
        assert not budget.exhausted
        assert budget.consumed == 0.0

    def test_a_perfect_objective_makes_any_failure_infinite_burn(self):
        budget = ErrorBudget(objective=1.0, good=99, total=100)
        assert budget.burn_rate == float("inf")
        assert budget.to_dict()["burn_rate"] == "inf"

    def test_good_cannot_exceed_total(self):
        budget = ErrorBudget(objective=0.99, good=500, total=100)
        assert budget.good == 100
        assert budget.bad == 0

    def test_recording_accumulates_and_only_bad_events_count(self):
        budget = ErrorBudget(objective=0.99)
        budget.record(ok=True, count=90)
        budget.record(ok=False, count=10)
        assert budget.total == 100
        assert budget.good == 90
        budget.record(ok=True, count=-5)  # 负数不倒退
        assert budget.total == 100
        budget.reset()
        assert budget.total == 0


# --------------------------------------------------------------------------
# 告警：积压
# --------------------------------------------------------------------------


class TestBacklogAlerts:
    RULES = (slo.BacklogRule("queue_backlog", "queued", 20, 300.0),)

    def test_depth_alone_is_not_enough(self):
        """刚提交 20 个任务然后开始消费的队列不是积压——批量上传是正常用法。"""
        assert evaluate_backlog({"queued": 100}, rules=self.RULES) == ()

    def test_duration_alone_is_not_enough(self):
        """队列空着待一小时也不构成积压。"""
        assert evaluate_backlog({"queued": 0}, rules=self.RULES, above_since={"queued": 0.0}, now=9_999) == ()

    def test_both_together_fire_once(self):
        alerts = evaluate_backlog(
            {"queued": 25}, rules=self.RULES, above_since={"queued": 0.0}, now=300.0
        )
        assert len(alerts) == 1
        assert alerts[0].evidence["depth"] == 25
        assert alerts[0].evidence["source_metric"] == "jobs.queue_depth"
        assert alerts[0].severity == slo.SEVERITY_WARNING

    def test_the_clock_starts_on_the_first_breach_not_on_the_first_call(self):
        marks: dict[str, float] = {}
        # 第一次看到超阈值：只记时间，不报。
        assert evaluate_backlog({"queued": 99}, rules=self.RULES, above_since=marks, now=100.0) == ()
        assert marks == {"queued": 100.0}
        # 299 秒后仍不够持续时间。
        assert evaluate_backlog({"queued": 99}, rules=self.RULES, above_since=marks, now=399.0) == ()
        # 300 秒后报。
        assert len(evaluate_backlog({"queued": 99}, rules=self.RULES, above_since=marks, now=400.0)) == 1

    def test_dropping_back_below_clears_the_clock(self):
        """回到阈值以下就该重新计时，否则"抖动"会被记成"持续"。"""
        marks: dict[str, float] = {}
        evaluate_backlog({"queued": 99}, rules=self.RULES, above_since=marks, now=0.0)
        evaluate_backlog({"queued": 0}, rules=self.RULES, above_since=marks, now=10.0)
        assert marks == {}
        assert evaluate_backlog({"queued": 99}, rules=self.RULES, above_since=marks, now=11.0) == ()

    def test_five_times_the_threshold_is_critical(self):
        severe = slo.BacklogRule("queue_backlog", "queued", 20, 300.0)
        alerts = evaluate_backlog(
            {"queued": 100}, rules=(severe,), above_since={"queued": 0.0}, now=300.0
        )
        assert alerts[0].severity == slo.SEVERITY_CRITICAL

    def test_the_shipped_defaults_cover_every_state_a_user_can_wait_in(self):
        assert {"queued", "parsing", "analyzing"} <= {rule.state for rule in slo.BACKLOG_RULES}

    def test_a_message_with_no_evidence_is_normalized_to_an_empty_dict(self):
        alert = slo.Alert("x", "info", "消息")
        assert alert.to_dict()["evidence"] == {}


# --------------------------------------------------------------------------
# 告警：provider 限速——对着真的 T4.1 指标读
# --------------------------------------------------------------------------


def fresh_metrics():
    """T4.1 真正的 MetricsRegistry，不是手搓的替身。

    刻意用真的 counter：``slo`` 依赖 ``Counter.get(**labels)`` 的子集匹配与
    ``.total``，一个手写的字典替身会在"读得出计数"的同时把"读不出"也变成能读，
    于是限速告警的判定逻辑永远得不到真正的检验。
    """
    from paperwhisperer.core.observability import MetricsRegistry

    return MetricsRegistry()


class TestProviderRateLimitAlerts:
    def test_a_lone_429_is_not_an_alert(self):
        """单次限速是重试退避本来就要处理的事，报了只会训练人忽略告警。"""
        metrics = fresh_metrics()
        metrics.llm_failures.inc(stage="overview", outcome="rate_limited")
        metrics.llm_failures.inc(stage="overview", outcome="ok")
        assert evaluate_provider_rate_limits({"llm": metrics.llm_failures}) == ()

    def test_a_systematic_ratio_fires_and_says_which_source(self):
        metrics = fresh_metrics()
        for _ in range(4):
            metrics.llm_failures.inc(stage="overview", outcome="rate_limited")
        for _ in range(16):
            metrics.llm_failures.inc(stage="overview", outcome="ok")
        alerts = evaluate_provider_rate_limits({"llm": metrics.llm_failures})
        assert len(alerts) == 1
        assert alerts[0].evidence["limited"] == 4
        assert alerts[0].evidence["total"] == 20
        assert alerts[0].evidence["ratio"] == pytest.approx(0.2)
        assert alerts[0].evidence["source"] == "llm"
        assert "rate_limited" in alerts[0].evidence["outcomes"]

    def test_both_the_absolute_count_and_the_ratio_must_hold(self):
        """绝对数排除了样本太小，比例排除了"一次就报"。两个都要。"""
        rule = ProviderRateLimitRule("x", ("rate_limited",), 3, 0.2)
        metrics = fresh_metrics()
        # 1 次限速 / 1 次成功 = 100%，但绝对数不够。
        metrics.llm_failures.inc(stage="qa", outcome="rate_limited")
        metrics.llm_failures.inc(stage="qa", outcome="ok")
        assert evaluate_provider_rate_limits({"llm": metrics.llm_failures}, rules=(rule,)) == ()
        # 补到 3 次，绝对数够了，比例 3/23 = 13% 不够。
        for _ in range(2):
            metrics.llm_failures.inc(stage="qa", outcome="rate_limited")
        for _ in range(20):
            metrics.llm_failures.inc(stage="qa", outcome="ok")
        assert evaluate_provider_rate_limits({"llm": metrics.llm_failures}, rules=(rule,)) == ()

    def test_the_ratio_denominator_is_every_call_not_just_the_failures(self):
        """分母是全量：一次限速 + 三次其它失败，占比是 1/4 而不是 1/1。"""
        rule = ProviderRateLimitRule("x", ("rate_limited",), 1, 0.2)
        metrics = fresh_metrics()
        metrics.llm_failures.inc(stage="qa", outcome="rate_limited")
        for _ in range(3):
            metrics.llm_failures.inc(stage="qa", outcome="server_error")
        signals = rate_limit_signals_from(metrics.llm_failures, rule)
        assert signals["limited"] == 1
        assert signals["total"] == 4
        assert signals["ratio"] == pytest.approx(0.25)

    def test_the_parsing_counter_is_read_through_its_own_label(self):
        """回归测试：``parsing.attempts_total`` 把状态记在 ``status=`` 上。

        读成 ``outcome=`` 时子集匹配不到任何 series，于是读到 0 —— 那个 0
        完全正常，因此 MinerU 的 429 会**永远不告警**且没有任何报错。
        """
        metrics = fresh_metrics()
        for _ in range(5):
            metrics.parsing_attempts.inc(
                stage="parse", parser="mineru_cloud", status="rate_limited"
            )
        assert rate_limit_signals_from(
            metrics.parsing_attempts,
            ProviderRateLimitRule("x", ("rate_limited",), 1, 0.2),
            label="status",
        )["limited"] == 5
        # 走 evaluate 的路径（按来源名自动选标签）必须同样能读到。
        alerts = evaluate_provider_rate_limits({"parsing": metrics.parsing_attempts})
        assert len(alerts) == 1
        assert alerts[0].evidence["source"] == "parsing"

    def test_reading_a_registry_never_crashes_when_a_metric_is_absent(self):
        """一个"还没限速过"的健康部署不该在 health 端点 500。

        registry 里的 counter 可能为 None（压根没发生过），而 None 上的
        属性读取抛的是 AttributeError。
        """
        assert alerts_from_registry(object()) == ()
        assert alerts_from_registry(fresh_metrics()) == ()

    def test_the_source_name_reaches_the_message(self):
        metrics = fresh_metrics()
        for _ in range(5):
            metrics.parsing_attempts.inc(parser="mineru_cloud", status="rate_limited")
        alerts = alerts_from_registry(metrics)
        assert len(alerts) == 1
        assert alerts[0].evidence["source"] == "parsing"
        assert "mineru" not in alerts[0].message  # 消息讲来源，不讲具体 parser
        # 面向用户的消息是中文的，不该把英文指标键原样抛给人。
        assert "解析服务" in alerts[0].message
        assert "'parsing'" not in alerts[0].message
        # 原始键仍在 evidence 里，方便对照指标名。
        assert alerts[0].evidence["source"] == "parsing"


# --------------------------------------------------------------------------
# 渲染
# --------------------------------------------------------------------------


class TestTargetTable:
    def test_the_table_distinguishes_measured_from_aspirational(self):
        """渲染出来的表格必须让人一眼看出哪几行有数据支撑。"""
        table = slo.format_target_table()
        assert "**实测**" in table
        assert "| --- | --- | --- | --- | --- | --- | --- |" in table
        # 每一行都要么是"实测"要么是"目标"，不能有第三种写法。
        body = [line for line in table.splitlines()[2:] if line.strip()]
        assert body
        for line in body:
            assert "**实测**" in line or "| 目标 |" in line

    def test_the_rendered_row_count_matches_the_table(self):
        assert len(slo.format_target_table().splitlines()) == len(slo.LATENCY_TARGETS) + 2

    def test_to_dict_carries_the_evidence_kind_to_the_client(self):
        payload = target_for_parse("pdf", 5, "legacy").to_dict()
        assert payload["evidence_kind"] == slo.EVIDENCE_MEASURED
        assert payload["samples"] == 22
        assert payload["pages"] == "1–10 页"
        assert payload["format"] == "pdf"
