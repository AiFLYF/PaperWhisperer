"""SLO：按页数/格式的 P95 时延目标、预算核算、以及**建立在 T4.1 指标之上**的告警信号。

T4.6 的计划原文是「按页数/格式建立 P95 时延目标；queue backlog 与 provider
限速告警」，验收标准是「**SLO 有数据支撑且可告警**」。这句话里有三件事，
本文件各给一件，且顺序不能颠倒：

1. **目标**（按格式 × 页数分桶的 P95）
2. **证据**（这些数字是量出来的还是定出来的）
3. **告警**（队列积压与 provider 限速）

**第一件与第二件必须写在同一个数据结构里，而不是一段说明文字。** 一个写在
文档里的"实测 P95"和一行写在代码里的阈值，半年后没有人能对得上号。所以
:data:`LATENCY_TARGETS` 的每一行都带一个 :data:`EVIDENCE_MEASURED` 或
:data:`EVIDENCE_TARGET` 标记，以及一句说明它是**怎么**得到的。查询
:func:`targets_for` 的结果自带这个标记，因此"哪些数字有数据支撑"是可以被
代码检查的，而不是靠读文档。

**没有测量支撑的目标在本文件里被明确标为 target，而且给出理由。** 这是本
任务最容易被做假的地方：写一串好看的 P95 数字很容易，而它们会在有人真的
依赖它时变成一个谎言。本文件的规矩是——

* :data:`EVIDENCE_MEASURED` 的数字**必须**带 :attr:`LatencyTarget.evidence`，
  内容是"在什么语料上、用什么 parser、跑了几次、哪台机器"。
* :data:`EVIDENCE_TARGET` 的数字**必须**在 ``evidence`` 里说明为什么此刻
  还没有测量（"docling 首次运行要下载模型，本环境无网络"），而不是假装它是
  保守估计。

**本仓库里哪些是实测的。** 见 ``docs/SLO.md`` 的测量一节。简版：``legacy``
（pypdf）parser 在 48 份语料 PDF 上的分桶 P95 是**实测**；**生产默认路由
（docling）与 MinerU 云解析没有实测**，因为 docling 首次运行要下载模型、
云解析要花钱，两者在本次环境里都做不到。因此本文件对这两者给的是
**目标**，并标明它们需要在 T0.9 bake-off 之后用同一套测量替换。

**告警信号不建第二套指标系统。** 队列积压读的是 T4.1 已经写好的
``jobs.queue_depth`` gauge，provider 限速读的是 ``llm.failures_total`` 与
``llm.retries_total``。本模块**不声明任何新指标**，它只对 T4.1 的
:class:`~paperwhisperer.core.observability.MetricsRegistry` 做读取与判定。
这样做的理由很实际：两套指标里只有一套会被人盯着看，而"待命告警"的那个
人一定会先看旧的。
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "ALERT_BACKLOG",
    "ALERT_BUDGET_EXHAUSTED",
    "ALERT_PROVIDER_RATE_LIMIT",
    "ALERT_SLO_BREACH",
    "BACKLOG_RULES",
    "BURN_RATE_ALERT",
    "EVIDENCE_KINDS",
    "EVIDENCE_MEASURED",
    "EVIDENCE_TARGET",
    "FORMAT_DOCX",
    "FORMAT_PDF",
    "FORMAT_PPTX",
    "FORMAT_TXT",
    "LATENCY_TARGETS",
    "MEASURE_REF",
    "MEASUREMENT_ENVIRONMENT",
    "MIN_MEASURED_SAMPLES",
    "PAGE_BUCKETS",
    "PROVIDER_RATE_LIMIT_RULES",
    "SEVERITY_CRITICAL",
    "SEVERITY_INFO",
    "SEVERITY_WARNING",
    "Alert",
    "BacklogRule",
    "ErrorBudget",
    "LatencyTarget",
    "PageBucket",
    "ProviderRateLimitRule",
    "SLOReport",
    "SLOSample",
    "alerts_from_registry",
    "classify_pages",
    "evaluate_backlog",
    "evaluate_provider_rate_limits",
    "format_target_table",
    "histogram_quantile",
    "latency_targets_for",
    "nearest_rank_percentile",
    "rate_limit_signals_from",
    "slo_report",
    "target_for_parse",
]


# --------------------------------------------------------------------------
# 证据标记
# --------------------------------------------------------------------------

#: 这个数字是在真实语料上量出来的。
EVIDENCE_MEASURED = "measured"
#: 这个数字是目标，**还没有**测量支撑。
EVIDENCE_TARGET = "target"

#: 只有这两种。声明成常量而不是一个裸字符串，是为了让"实测与目标要可区分"
#: 这件事在类型上就成立——``evidence_kind`` 的取值空间是这两个值，
#: :func:`latency_targets_for` 可以对它做穷尽判断。
EVIDENCE_KINDS = (EVIDENCE_MEASURED, EVIDENCE_TARGET)

#: 一个 ``measured`` 目标至少要多少个样本。
#:
#: **5 是从 nearest-rank 的定义里长出来的，不是拍脑袋定的。** 取
#: ``ceil(0.95 * n)``：n=5 时是第 5 个（也就是最大值），n=6 时才是第 6 个……
#: 换句话说 nearest-rank 在 n < 20 时**总是返回最大值**，而"最大值"是一个
#: 观测值，不是分位数。所以 5 是"勉强还能称作分位数"的下限，低于它的桶
#: 只能标 target——那个 :data:`EVIDENCE_TARGET` 标签在这种情况下不是谦虚，
#: 是准确。
MIN_MEASURED_SAMPLES = 5


# --------------------------------------------------------------------------
# 格式与页数分桶
# --------------------------------------------------------------------------

FORMAT_PDF = "pdf"
FORMAT_DOCX = "docx"
FORMAT_PPTX = "pptx"
FORMAT_TXT = "txt"

#: 测量与分桶所用的机器，写进每一行 ``evidence``。
#:
#: 一次延迟数字在**另一台机器**上就是另一个数字，而一个没写清楚机器的数字
#: 等于没写。把机器与运行环境作为数据的一部分（而不是文档里的一句话），
#: 是为了让"这个数字是在哪里来的"这个问题有**一个**答案。
MEASUREMENT_ENVIRONMENT = (
    "Linux 6.8 x86_64 / 72 vCPU / 251 GiB RAM / CPython 3.13 / "
    "pypdf 纯文本层提取，单进程，无 GPU"
)


@dataclass(frozen=True, slots=True)
class PageBucket:
    """一个页数区间。闭区间 ``[min_pages, max_pages]``，``max_pages=None`` 表示无上界。"""

    name: str
    min_pages: int
    max_pages: int | None

    def contains(self, pages: int) -> bool:
        count = max(1, int(pages or 0))
        if count < self.min_pages:
            return False
        return self.max_pages is None or count <= self.max_pages

    def describe(self) -> str:
        upper = "∞" if self.max_pages is None else str(self.max_pages)
        return f"{self.min_pages}–{upper} 页"


#: 页数分桶。边界来自实测语料的实际分布（见 ``docs/SLO.md``）：1–10 页覆盖
#: 语料里 22/45 份文档，11–30 页覆盖另外 15 份，而 31 页以上只剩 8 份——
#: 再细分下去每一桶的样本量都不足以谈 P95（5 个样本的 P95 就是最大值，
#: 那不是分位数，是一个观测值）。**分桶边界由样本量决定，而不是由直觉**。
PAGE_BUCKETS: tuple[PageBucket, ...] = (
    PageBucket("s", 1, 10),
    PageBucket("m", 11, 30),
    PageBucket("l", 31, 80),
    PageBucket("xl", 81, 200),
    PageBucket("xxl", 201, None),
)


def classify_pages(pages: int) -> PageBucket:
    """页数落在哪个桶。0/负数按 1 页处理。

    按 1 页而不是报错，是因为一个 ``page_count=0`` 的文档（尚未解析完成的
    登记行）**确实要参与配额与 SLO 归类**；把它当成错误会让一个刚上传的
    文档无法被任何 SLO 规则归类，而它的解析耗时会落在"1–10 页"桶里，
    那正是它在解析完成后的归属。
    """
    count = max(1, int(pages or 0))
    for bucket in PAGE_BUCKETS:
        if bucket.contains(count):
            return bucket
    return PAGE_BUCKETS[-1]


# --------------------------------------------------------------------------
# 时延目标
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LatencyTarget:
    """一条时延目标：谁的、哪个桶、多少、**以及它凭什么存在**。

    ``parser`` 单独成字段而不是塞进 ``stage``，是因为"解析耗时"这一个指标
    在不同 parser 之间**差一个数量级**：本仓库实测到的 legacy（pypdf）解析
    与 docling 的差距远大于桶与桶之间的差距。把它并进 stage 会让
    "按格式建立 P95 目标"变成"按 parser 建立 P95 目标"，而计划里没提过
    parser——但没提不等于可以假装它不存在。
    """

    stage: str
    media_format: str
    bucket: str
    parser: str
    p95_ms: int
    evidence_kind: str
    evidence: str
    samples: int = 0
    #: 标为实测的行，**必须**指向仓内一份可复核的产物（仓根相对路径）。
    #: 这不是装饰：一句“在 45 份语料上量的”无法被复核，而一个不存在的路径
    #: 更直接地说明那次测量没有被留下来。``tests/test_slo.py`` 断言这个文件
    #: 真的存在，于是“标了实测却没留下产物”会**让测试失败**，而不是留在
    #: 文档里等人相信。
    evidence_ref: str = ""

    def __post_init__(self) -> None:
        if self.evidence_kind not in EVIDENCE_KINDS:
            raise ValueError(
                f"evidence_kind={self.evidence_kind!r} 非法"
                f"（可用值：{', '.join(EVIDENCE_KINDS)}）。"
                "每个时延目标都必须声明它是有实测支撑还是目标。"
            )
        if not str(self.evidence or "").strip():
            raise ValueError(
                f"目标 {self.stage}/{self.media_format}/{self.bucket} 缺少 evidence 说明。"
                "没有说明的数字无法被复核，等于没有依据。"
            )
        if self.evidence_kind == EVIDENCE_MEASURED:
            if int(self.samples) < MIN_MEASURED_SAMPLES:
                raise ValueError(
                    f"目标 {self.stage}/{self.media_format}/{self.bucket} 标为实测，"
                    f"但只有 {self.samples} 个样本（少于 {MIN_MEASURED_SAMPLES}）。"
                    "样本量过小时 P95 不是分位数，而是最大值那一个观测值——"
                    "把它标成实测，等于让一条观测冒充一个统计量。"
                )
            if not str(self.evidence_ref or "").strip():
                raise ValueError(
                    f"目标 {self.stage}/{self.media_format}/{self.bucket} 标为实测，"
                    "但没有 evidence_ref 指向仓内的测量产物。"
                    "实测必须可复核：把跑出这批数字的报告提交进仓，并在这里写出它的路径。"
                )

    @property
    def is_measured(self) -> bool:
        return self.evidence_kind == EVIDENCE_MEASURED

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "format": self.media_format,
            "bucket": self.bucket,
            "pages": _bucket_describe(self.bucket),
            "parser": self.parser,
            "p95_ms": int(self.p95_ms),
            "evidence_kind": self.evidence_kind,
            "evidence": self.evidence,
            "samples": int(self.samples),
        }


def _bucket_describe(name: str) -> str:
    for bucket in PAGE_BUCKETS:
        if bucket.name == name:
            return bucket.describe()
    return name


def _measured(
    stage: str,
    media_format: str,
    bucket: str,
    parser: str,
    p95_ms: int,
    evidence: str,
    samples: int,
    evidence_ref: str,
) -> LatencyTarget:
    return LatencyTarget(
        stage=stage,
        media_format=media_format,
        bucket=bucket,
        parser=parser,
        p95_ms=p95_ms,
        evidence_kind=EVIDENCE_MEASURED,
        evidence=evidence,
        samples=samples,
        evidence_ref=evidence_ref,
    )


def _target(
    stage: str,
    media_format: str,
    bucket: str,
    parser: str,
    p95_ms: int,
    evidence: str,
) -> LatencyTarget:
    return LatencyTarget(
        stage=stage,
        media_format=media_format,
        bucket=bucket,
        parser=parser,
        p95_ms=p95_ms,
        evidence_kind=EVIDENCE_TARGET,
        evidence=evidence,
        samples=0,
    )


#: 测量产物的路径，仓根相对。所有 ``measured`` 行都指回它。
#:
#: **一份产物，而不是一个描述。** 由 ``eval/run_parser_eval.py`` 产出，可用
#: ``python3 eval/run_parser_eval.py --parsers legacy`` 重跑。``tests/test_slo.py``
#: 会断言这个文件存在、且每一行 ``measured`` 目标都能从它重算出来——所以这批
#: 数字不是"某次有人跑过"，而是"任何人重跑都能得到同一个数"。
MEASURE_REF = "eval/reports/run1-local/parser_eval.json"

#: 那份产物里 legacy 一行的可复算摘要，写进 ``evidence``，省得读者自己去翻。
_MEASURE_METHOD = (
    "LegacyDocumentParser（pypdf 文本层）在 eval/corpus 的成功样本上跑 "
    "eval/run_parser_eval.py 一次，每份文档一个样本；分位数用 "
    f"nearest_rank_percentile。产物：{MEASURE_REF}（corpus_version sha256:0dfe3197…）。"
    f"环境：{MEASUREMENT_ENVIRONMENT}"
)

_NO_DOCLING_EVIDENCE = (
    "目标，暂无实测：生产默认路由是 docling，其首次运行需要下载模型权重，"
    "本次开发环境无网络，测不了；本行取值按 legacy 实测值加一个量级余量给出，"
    "必须在 T0.9 bake-off（届时有模型与语料）之后用同一测量脚本替换为 measured。"
)

_NO_MINERU_EVIDENCE = (
    "目标，暂无实测：MinerU 云解析是有付费出网调用，本次开发环境无网络无凭据，"
    "任何数字都只能是目标；实测必须在持有 MINERU_API_TOKEN 的环境里跑，"
    "并与本地解析做同语料对照。"
)

_NO_LLM_EVIDENCE = (
    "目标，暂无实测：LLM 调用的耗时取决于部署方选择的模型与供应商，"
    "本项目无法在离线环境测得；上界取自 T4.1 已写死的默认阈值"
    "OBSERVABILITY_SLOW_LLM_MS=30000ms 与网关 900s 的解析上限之间的量级。"
)

_NO_RETRIEVAL_SCALE_EVIDENCE = (
    "目标，暂无实测：个人档位是全表扫描，耗时随语料线性增长，"
    "因此不存在与语料无关的 P95；本行的分档须由部署方在真实语料上重测。"
)

#: 语料里**样本量不足**因而不能自称实测的那些行。理由是同一个，差别只在数量。
_TOO_FEW_SAMPLES = (
    "目标，暂无合格实测：语料在这一桶里的样本数 n={n}，"
    f"低于 nearest-rank 分位数所需的 {MIN_MEASURED_SAMPLES} 个样本——"
    "n 这么小时 P95 恒等于最大值，那是一个观测值而不是分位数。"
    "实测值 {p95}ms 记录在此仅作参考；补语料后用同一脚本重跑即可转为 measured。"
)

#: 曾经被标成实测、但仓内没有留下任何可复核产物的那几行。
_NO_RETRIEVAL_ARTIFACT_EVIDENCE = (
    "目标，暂无实测：检索时延曾在本地跑过并得到过一组数字，但**那次测量的产物"
    "没有提交进仓**，因此它无法被复核，本行按目标对待。补齐办法是提交一个"
    "检索基准脚本及其报告（与本表同格式），然后把 evidence_ref 指过去。"
    "在补齐之前，个人档位的检索延迟没有任何有据可查的承诺。"
)

#: 全部时延目标。**每一行都必须自己说清凭什么存在**（见
#: :class:`LatencyTarget.__post_init__` 的硬校验）。
#:
#: **本表里有实测支撑的只有 3 行**，其余全部是目标。这个比例不好看，但它就是
#: 当前真实的证据量：本次开发环境无网络、docling 要下模型、MinerU 要花钱，
#: 真正跑过并留下产物的是 legacy（pypdf）在 45 份 PDF 上的解析。把它写成
#: "大部分都有数据支撑"才是那个谎言。
LATENCY_TARGETS: tuple[LatencyTarget, ...] = (
    # -- 解析：legacy（pypdf）—— 实测 ---------------------------------------
    # 这三行的数字就是从 MEASURE_REF 里 nearest-rank 重算出来的，可逐字复现。
    _measured("parse", FORMAT_PDF, "s", "legacy", 379, _MEASURE_METHOD, 22, MEASURE_REF),
    _measured("parse", FORMAT_PDF, "m", "legacy", 3_699, _MEASURE_METHOD, 15, MEASURE_REF),
    _measured("parse", FORMAT_PDF, "l", "legacy", 23_621, _MEASURE_METHOD, 5, MEASURE_REF),
    # -- 解析：legacy —— 样本量不足，改标目标 --------------------------------
    _target("parse", FORMAT_PDF, "xl", "legacy", 6_575,
            _TOO_FEW_SAMPLES.format(n=2, p95=6575)),
    _target("parse", FORMAT_PDF, "xxl", "legacy", 8_206,
            _TOO_FEW_SAMPLES.format(n=1, p95=8206)),
    _target("parse", FORMAT_DOCX, "s", "legacy", 50,
            _TOO_FEW_SAMPLES.format(n=4, p95=16)),
    _target("parse", FORMAT_PPTX, "s", "legacy", 50,
            _TOO_FEW_SAMPLES.format(n=3, p95=9)),
    # -- 解析：docling（生产默认路由）—— 目标 --------------------------------
    _target("parse", FORMAT_PDF, "s", "docling", 2_000, _NO_DOCLING_EVIDENCE),
    _target("parse", FORMAT_PDF, "m", "docling", 8_000, _NO_DOCLING_EVIDENCE),
    _target("parse", FORMAT_PDF, "l", "docling", 20_000, _NO_DOCLING_EVIDENCE),
    _target("parse", FORMAT_PDF, "xl", "docling", 40_000, _NO_DOCLING_EVIDENCE),
    _target("parse", FORMAT_PDF, "xxl", "docling", 90_000, _NO_DOCLING_EVIDENCE),
    _target("parse", FORMAT_DOCX, "s", "docling", 4_000, _NO_DOCLING_EVIDENCE),
    _target("parse", FORMAT_PPTX, "s", "docling", 4_000, _NO_DOCLING_EVIDENCE),
    # -- 解析：MinerU 云端 —— 目标 -------------------------------------------
    _target("parse", FORMAT_PDF, "s", "mineru_cloud", 60_000, _NO_MINERU_EVIDENCE),
    _target("parse", FORMAT_PDF, "m", "mineru_cloud", 90_000, _NO_MINERU_EVIDENCE),
    _target("parse", FORMAT_PDF, "l", "mineru_cloud", 180_000, _NO_MINERU_EVIDENCE),
    _target("parse", FORMAT_PDF, "xl", "mineru_cloud", 300_000, _NO_MINERU_EVIDENCE),
    _target("parse", FORMAT_PDF, "xxl", "mineru_cloud", 600_000, _NO_MINERU_EVIDENCE),
    # -- 检索：按语料规模分档 —— 目标（曾测过，但产物没有提交进仓）---------
    _target("retrieve", "any", "corpus-1k", "personal-sqlite", 80,
            _NO_RETRIEVAL_ARTIFACT_EVIDENCE),
    _target("retrieve", "any", "corpus-3k", "personal-sqlite", 110,
            _NO_RETRIEVAL_ARTIFACT_EVIDENCE),
    _target("retrieve", "any", "corpus-7k", "personal-sqlite", 180,
            _NO_RETRIEVAL_ARTIFACT_EVIDENCE),
    _target("retrieve", "any", "hosted-pgvector", "hosted-postgres", 200,
            _NO_RETRIEVAL_SCALE_EVIDENCE),
    # -- LLM —— 目标 ---------------------------------------------------------
    _target("llm", "any", "any", "any", 30_000, _NO_LLM_EVIDENCE),
)


def latency_targets_for(
    stage: str, media_format: str = "", parser: str = ""
) -> tuple[LatencyTarget, ...]:
    """取某一阶段（可再按格式/parser 过滤）的全部目标。"""
    return tuple(
        target
        for target in LATENCY_TARGETS
        if target.stage == stage
        and (not media_format or target.media_format == media_format)
        and (not parser or target.parser == parser)
    )


def target_for_parse(
    media_format: str, pages: int, parser: str
) -> LatencyTarget | None:
    """一个具体文档的解析时延目标。找不到返回 ``None``。

    返回 ``None`` 而不是回落到某一档，是因为"没有目标"和"目标很宽"在
    告警里是两件事：前者应该让这一条**不参与判定**（而不是用别的桶的宽目标
    把它判成通过），后者是一次真实的达标。
    """
    bucket = classify_pages(pages)
    for target in LATENCY_TARGETS:
        if (
            target.stage == "parse"
            and target.parser == parser
            and target.media_format == media_format
            and target.bucket == bucket.name
        ):
            return target
    return None


def format_target_table() -> str:
    """把目标表渲染成一段可粘贴的 Markdown。给 ``docs/SLO.md`` 用。"""
    lines = [
        "| 阶段 | 格式 | 页数 | parser | P95 目标 | 依据 | 样本 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for target in LATENCY_TARGETS:
        kind = "**实测**" if target.is_measured else "目标"
        lines.append(
            f"| {target.stage} | {target.media_format} | {_bucket_describe(target.bucket)} "
            f"| {target.parser} | {target.p95_ms} ms | {kind} | {target.samples or '—'} |"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 分位数算术
# --------------------------------------------------------------------------


def nearest_rank_percentile(values: Sequence[float], quantile: float = 0.95) -> float:
    """精确分位数：排序后取第 ``ceil(q * n)`` 个（nearest-rank）。

    **为什么是 nearest-rank 而不是线性插值。** 因为它对**任何一个**已观测
    值都返回**一个**已观测值——一个 90 个样本的 P95 就是"第 86 个观测值"，
    可以回去查是哪一份文件。线性插值会返回一个从未发生过的数（夹在两个观测
    之间），它在报表上更好看，在出事时无法回答"到底是哪一次慢"。

    它也是本项目 :func:`nearest_rank_percentile` 与测试里手算值能逐字对上的
    原因：没有插值系数，只有一次 ``ceil``。

    ``quantile`` 越界时钳到 ``[0, 1]`` 而不是抛异常——一个算 SLA 的函数
    不应该因为调用方传了 1.5 而让一次告警检查崩掉。
    """
    numbers = sorted(float(value) for value in values)
    if not numbers:
        return 0.0
    q = min(1.0, max(0.0, float(quantile)))
    index = max(0, min(len(numbers) - 1, math.ceil(q * len(numbers)) - 1))
    return numbers[index]


def histogram_quantile(
    buckets: Sequence[float],
    counts: Sequence[int],
    quantile: float = 0.95,
) -> float:
    """从**累积桶计数**里估分位数——即 Prometheus ``histogram_quantile`` 的算法。

    为什么需要它：T4.1 的
    :class:`~paperwhisperer.core.observability.Histogram` 存的是桶计数而不是
    原始样本（有界内存的取舍写在那个模块里），所以一个**运行时**的 P95 只能
    从桶里估。而 :func:`nearest_rank_percentile` 需要原始样本，只在离线分析
    （跑语料、算报告）时可用。两个函数都在，是因为它们回答的是两个问题：
    "过去 90 次里第 86 个是多少"（精确，需要样本）与
    "从现在起挂着的直方图能告诉你第 95 百分位大约是多少"（估计，不需要）。

    **桶内线性插值**，与 Prometheus 一致：

    * 落在第一个桶之下 → 返回第一个桶的上界。
    * 落在最后一个桶（``+Inf``）里 → 返回最后一个**有限**桶的上界。
    * 落在桶 ``i`` 内 → ``edge[i-1] + (edge[i] - edge[i-1]) * frac``。

    这是一个**估计**，其误差上限是目标桶的宽度。这是本文件唯一一个会
    低估最大值的地方，而它被写进函数名与文档，而不是被藏起来。
    """
    edges = [float(edge) for edge in buckets]
    cumulative = [int(count) for count in counts]
    if not edges or len(edges) != len(cumulative) or not any(cumulative):
        return 0.0
    total = cumulative[-1]
    if total <= 0:
        return 0.0
    q = min(1.0, max(0.0, float(quantile)))
    rank = q * total

    previous_count = 0
    for index, edge in enumerate(edges):
        current = cumulative[index]
        if current >= rank:
            if index == 0:
                return edge
            previous_edge = edges[index - 1]
            span = current - previous_count
            if span <= 0:
                return edge
            fraction = (rank - previous_count) / span
            upper = edge
            if not math.isfinite(upper):
                # 落进了 +Inf 桶。**不能朝无穷插值**：`previous_edge +
                # (inf - previous_edge) * frac` 的结果是 inf，它会让告警渲染成
                # "inf"，并让任何乘以它的算术变成 nan。Prometheus 自己在这一步
                # 返回最后一个**有限**桶的上界，本实现原先在文档里承诺了这一点
                # 却没做到——末尾那句 `return edges[-1]` 是死代码，因为
                # `rank <= cumulative[-1]` 恒成立，循环必然在内部返回。
                upper = previous_edge
            return previous_edge + (upper - previous_edge) * fraction
        previous_count = current
    # 只有在 counts 末位小于 rank 时才会走到这里（正常数据不会发生）。
    return edges[-1]


# --------------------------------------------------------------------------
# 样本与报告
# --------------------------------------------------------------------------


#: 检索各档名义上的 chunk 数上界。选档靠它，而不是靠"取第一条"——
#: 见 :meth:`SLOSample.target`。
_RETRIEVE_CEILINGS: dict[str, int] = {
    "corpus-1k": 1_000,
    "corpus-3k": 3_000,
    "corpus-7k": 7_000,
}


def _retrieve_target(parser: str, corpus_chunks: int) -> LatencyTarget | None:
    """按语料规模挑检索目标。**挑不出来就返回 ``None``。**

    返回 ``None`` 的两种情况都值得被当成"缺一个目标"而不是"用最宽的那档"：

    * 调用方没有告诉这次查询面对多大的语料（``corpus_chunks <= 0``）；
    * 语料比最大的那一档还大（超过 7000 chunk），表里没有对应的行。

    两种情况都**不猜**。个人档位是全表扫描，检索时延随语料线性增长，所以
    "拿 1k 档的目标去判一个 7k 语料的查询"是一个必然误报的规则——它会在
    语料变大之后把每一次正常查询都报成超标。宁可让它进 ``missing_target``。
    """
    count = int(corpus_chunks or 0)
    if count <= 0:
        return None
    for target in latency_targets_for("retrieve", "any", parser):
        ceiling = _RETRIEVE_CEILINGS.get(target.bucket)
        if ceiling is not None and count <= ceiling:
            return target
    return None


@dataclass(frozen=True, slots=True)
class SLOSample:
    """一次观测：哪个阶段、多久、以及它该跟谁比。

    ``corpus_chunks`` 只对 ``stage="retrieve"`` 有意义，而它**必须由调用方
    如实填写**。不填的后果不是"用小语料的标准去判"，而是一次 ``None``：
    这一次观测不参与判定，计入报告的 ``missing_target``。
    """

    stage: str
    duration_ms: float
    media_format: str = ""
    pages: int = 0
    parser: str = ""
    ok: bool = True
    corpus_chunks: int = 0

    def target(self) -> LatencyTarget | None:
        if self.stage == "parse":
            return target_for_parse(self.media_format, self.pages, self.parser)
        if self.stage == "retrieve":
            return _retrieve_target(self.parser, self.corpus_chunks)
        return latency_targets_for(self.stage)[:1] or None

    def within_target(self) -> bool:
        target = self.target()
        if target is None:
            return True
        return float(self.duration_ms) <= float(target.p95_ms)


@dataclass(frozen=True, slots=True)
class SLOReport:
    """一次 SLO 评估的结果。给 health 端点与告警评估用。"""

    generated_at: str
    checked: int
    within_target: int
    breaches: tuple[dict[str, Any], ...]
    missing_target: int

    @property
    def compliance(self) -> float:
        """达标率。没有样本时是 ``1.0`` 而不是 ``0.0``。

        ``0.0`` 会让"还没有任何观测"看起来像"全部不达标"，而这两件事在
        告警上是完全相反的：前者不需要任何人做任何事。
        """
        return 1.0 if self.checked <= 0 else self.within_target / self.checked

    @property
    def ok(self) -> bool:
        return not self.breaches

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "checked": self.checked,
            "within_target": self.within_target,
            "compliance": round(self.compliance, 4),
            "missing_target": self.missing_target,
            "breaches": list(self.breaches),
            "ok": self.ok,
        }


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slo_report(
    samples: Iterable[SLOSample],
    *,
    generated_at: str = "",
) -> SLOReport:
    """拿一批观测去比 :data:`LATENCY_TARGETS`，产出一份报告。

    **没有目标的样本不计入 ``checked``**，只计入 ``missing_target``。理由：
    把"我们没有为这个组合定过目标"算成"不达标"，会让一份报告在没有目标的
    地方报红，而看报告的人会去修那个红，而真正要修的是"缺一个目标"这件事。
    这两件事都值得做，但不该混在同一个数字里。
    """
    checked = 0
    within = 0
    missing = 0
    breaches: list[dict[str, Any]] = []
    for sample in samples:
        target = sample.target()
        if target is None:
            missing += 1
            continue
        checked += 1
        if sample.within_target():
            within += 1
        else:
            breaches.append(
                {
                    "stage": target.stage,
                    "format": target.media_format,
                    "bucket": target.bucket,
                    "parser": target.parser,
                    "duration_ms": round(float(sample.duration_ms), 1),
                    "p95_ms": int(target.p95_ms),
                    "evidence_kind": target.evidence_kind,
                }
            )
    return SLOReport(
        generated_at=generated_at or _now_iso(),
        checked=checked,
        within_target=within,
        breaches=tuple(breaches),
        missing_target=missing,
    )


# --------------------------------------------------------------------------
# 预算核算
# --------------------------------------------------------------------------


@dataclass(slots=True)
class ErrorBudget:
    """可用性预算：允许坏事件的数量、已用掉多少、还剩多少。

    三个数都由**同一对** ``good``/``total`` 推导，所以它们不可能互相矛盾——
    这是本类不持有任何其它状态的原因。允许的坏事件数是
    ``floor(total * (1 - objective))``，用 ``floor`` 而不是四舍五入，因为
    ``total=100``、目标 99.5% 时允许 0.5 个坏事件——**允许半个事件的预算
    不是一个可以发放的预算**，向下取整让预算非负且可解释。

    ``burn_rate`` 是"消耗速度相对预期消耗速度的倍数"。它才是告警真正该看的
    数：达标率 99.0% 在一个 1000 次请求的窗口里可能完全健康，也可能在
    一个 10 次请求的窗口里已经烧穿了预算。只看达标率的告警在小窗口上会
    疯狂误报，而 :data:`BURN_RATE_ALERT` 把它换算成了与窗口无关的量。
    """

    objective: float = 0.99
    good: int = 0
    total: int = 0
    window_seconds: int = 86400

    def __post_init__(self) -> None:
        self.objective = min(1.0, max(0.0, float(self.objective)))
        self.good = max(0, int(self.good))
        self.total = max(0, int(self.total))
        if self.good > self.total:
            self.good = self.total

    @property
    def allowed_bad(self) -> int:
        """这个窗口允许的坏事件数（向下取整）。"""
        return int(math.floor(self.total * (1.0 - self.objective)))

    @property
    def bad(self) -> int:
        return self.total - self.good

    @property
    def remaining(self) -> int:
        """还剩多少次坏事件的余量。**可以为负**——负数就是已经超支。"""
        return self.allowed_bad - self.bad

    @property
    def consumed(self) -> float:
        """预算已用比例 0.0–1.0+。``allowed_bad == 0`` 时看 :attr:`burn_rate`。"""
        allowed = self.allowed_bad
        return 0.0 if allowed <= 0 else min(1.0, self.bad / allowed)

    @property
    def exhausted(self) -> bool:
        return self.remaining <= 0 and self.total > 0

    @property
    def burn_rate(self) -> float:
        """消耗速度相对预期的倍数。

        ``total=0`` 时是 ``0.0``：没有任何事件就没有消耗，而"没有观测"
        不该被报成"烧穿预算"。

        分母 ``total * (1 - objective)`` 就是"这个窗口本该出现的坏事件数"；
        分子是实际出现的。分子除以分母 = 相对消耗速度。
        """
        if self.total <= 0:
            return 0.0
        expected_bad = self.total * (1.0 - self.objective)
        if expected_bad <= 0:
            # 目标是 100%：任何一次坏事件都是无限倍速烧穿。
            return 0.0 if self.bad == 0 else float("inf")
        return self.bad / expected_bad

    def record(self, ok: bool, count: int = 1) -> None:
        """记入一批事件。``ok=False`` 计入坏事件。"""
        amount = max(0, int(count))
        if amount == 0:
            return
        self.total += amount
        if ok:
            self.good += amount

    def reset(self) -> None:
        self.good = 0
        self.total = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "good": self.good,
            "total": self.total,
            "bad": self.bad,
            "allowed_bad": self.allowed_bad,
            "remaining": self.remaining,
            "consumed": round(self.consumed, 4),
            "burn_rate": (
                self.burn_rate if math.isfinite(self.burn_rate) else "inf"
            ),
            "exhausted": self.exhausted,
            "window_seconds": int(self.window_seconds),
        }


#: 触发告警的消耗速度倍数。取 2.0 的理由写在这里：**1.0 恰好烧完整个窗口**
#: 是一个事后才知道的事实，而按 SRE 惯例，把告警放在"半个窗口就会烧完"的
#: 位置（burn rate 2 = 相对于 30 天预算，这是 Google SRE 的 "fast burn"
#: 档位）留出的正是人工介入需要的时间。
BURN_RATE_ALERT = 2.0


# --------------------------------------------------------------------------
# 告警：queue backlog 与 provider 限速
# --------------------------------------------------------------------------
#
# 两个信号，**都读 T4.1 已经声明的指标**，本模块不新增任何 metric。
# 这是刻意的：两套指标里只有一套会被人盯着看，而两套告警规则里必然有一套
# 永远没人调。第三个理由是本任务明确要求的——"在 T4.1 的指标之上构建"。

ALERT_BACKLOG = "queue_backlog"
ALERT_PROVIDER_RATE_LIMIT = "provider_rate_limit"
ALERT_SLO_BREACH = "slo_breach"
ALERT_BUDGET_EXHAUSTED = "error_budget_exhausted"

SEVERITY_INFO = "info"
SEVERITY_WARNING = "warning"
SEVERITY_CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class Alert:
    """一条待发的告警。**自带证据**——没有 ``evidence`` 的告警没人敢处理。"""

    name: str
    severity: str
    message: str
    evidence: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not self.evidence:
            object.__setattr__(self, "evidence", {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "severity": self.severity,
            "message": self.message,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True, slots=True)
class BacklogRule:
    """队列积压的判定规则。

    两个条件而不是一个，理由是它们回答不同的问题：

    * ``threshold`` —— **绝对深度**。队列里有 200 个任务时，机器就已经在
      积压了，不管它排多久。
    * ``sustained_seconds`` —— **持续时间**。一个刚提交 5 个任务然后开始
      消费的队列不是积压。一条只看深度的规则会在每次批量上传时误报，而批量
      上传是这个产品的正常用法。

    两条都满足才报。少一条都会在正常使用中产生噪音，而一个每天响十次的
    告警等于没有告警。
    """

    name: str
    state: str
    threshold: int
    sustained_seconds: float

    def describe(self) -> str:
        return (
            f"状态 {self.state!r} 连续 {self.sustained_seconds:.0f}s 超过 "
            f"{self.threshold} 个任务"
        )


#: 默认规则。``queued`` 是用户唯一能看见的积压（他提交了但还没开始），
#: 所以它是主规则；``parsing`` 卡住则是另一个问题（解析本身慢或挂），
#: 阈值更高因为一个 worker 同时只跑一个任务，parsing 深度天然很小。
BACKLOG_RULES: tuple[BacklogRule, ...] = (
    BacklogRule(ALERT_BACKLOG, "queued", 20, 300.0),
    BacklogRule(ALERT_BACKLOG, "parsing", 8, 600.0),
    BacklogRule(ALERT_BACKLOG, "analyzing", 8, 600.0),
)


def evaluate_backlog(
    counts_by_state: Mapping[str, int],
    *,
    rules: Sequence[BacklogRule] = BACKLOG_RULES,
    above_since: Mapping[str, float] | None = None,
    now: float = 0.0,
    rule_suffix: str = "",
) -> tuple[Alert, ...]:
    """拿 T4.1 的 ``jobs.queue_depth`` gauge 判断是否积压。

    ``counts_by_state`` 就是
    :meth:`~paperwhisperer.observability.Observability.record_job_queue` 写进
    gauge 的那个映射（或者
    :meth:`~paperwhisperer.infra.jobs.JobStore.count_by_state` 的直接返回值）——
    两种来源形状相同，所以判据可以在两种上跑。

    ``above_since`` 是"该状态从什么时候起超过阈值"，由调用方用同一个
    :class:`BacklogRule` 的 ``state`` 作为键维护。**本模块不自己计时**，
    因为它是一个纯函数：一个会自己 sleep 的告警函数没法被测试，而且它在
    一次检查里做的等待会拖慢 health 端点。计时是调用方（一个后台 tick）
    的事，而那正是它该在的地方。

    **传入的 ``dict`` 会被就地更新。** 这不是可有可无的便利：首次越过阈值时
    本函数只记下时刻、不发告警，如果它把这份时刻写进一份副本，调用方就永远
    学不到"从什么时候开始"——而下一次检查需要这个值才能判断"是否已持续够
    久"。一个把状态藏在局部变量里的持续时间判定，等于永远不判定。返回值里
    不带 ``marks`` 就是因为它已经在调用方手里了。
    """
    alerts: list[Alert] = []
    marks: dict[str, float] = (
        above_since if isinstance(above_since, dict) else dict(above_since or {})
    )
    for rule in rules:
        depth = int(counts_by_state.get(rule.state, 0) or 0)
        if depth <= rule.threshold:
            marks.pop(rule.state, None)
            continue
        since = marks.get(rule.state)
        if since is None:
            marks[rule.state] = now
            continue
        held = max(0.0, float(now) - float(since))
        if held < rule.sustained_seconds:
            continue
        severity = (
            SEVERITY_CRITICAL if depth >= rule.threshold * 5 else SEVERITY_WARNING
        )
        alerts.append(
            Alert(
                name=f"{rule.name}{rule_suffix}",
                severity=severity,
                message=(
                    f"任务队列积压：状态 {rule.state} 已有 {depth} 个任务，"
                    f"持续 {held:.0f} 秒（阈值：{rule.threshold} 个 / "
                    f"{rule.sustained_seconds:.0f} 秒）。"
                    "新提交的任务将长时间不会被处理。"
                ),
                evidence={
                    "state": rule.state,
                    "depth": depth,
                    "threshold": rule.threshold,
                    "sustained_seconds": round(held, 1),
                    "source_metric": "jobs.queue_depth",
                },
            )
        )
    return tuple(alerts)


@dataclass(frozen=True, slots=True)
class ProviderRateLimitRule:
    """provider 限速的判定规则。

    ``outcomes`` 列出**算作限速**的 T4.1 标签取值。默认只认两种，因为这是
    T4.1 实际会写出来的两个：

    * ``llm.failures_total{outcome="rate_limited"}`` —— 模型侧明确降速。
    * ``parsing.attempts_total{status="rate_limited"}`` —— MinerU 等外部
      解析服务的 429（``RETRYABLE_STATUS_CODES`` 含 429，见
      ``ingestion/adapters/mineru_cloud.py``）。

    阈值同时看**绝对次数**与**占比**：一次 429 不值得叫人，一分钟内二十次
    也不值得（用户等一下就好），而"占比超过 20% 且绝对数超过 3"是一个
    真的需要人看的组合——前者说明是系统性问题，后者排除了样本太小带来的
    误报。两个条件都满足才报。
    """

    name: str
    outcomes: tuple[str, ...]
    threshold: int
    min_ratio: float
    #: 限速记在哪个标签上。**这个字段不能省**：T4.1 的两个 counter 标签名不同
    #: （``llm.failures_total`` 用 ``outcome=``，``parsing.attempts_total`` 用
    #: ``status=``），而读错标签名**不会报错**——它只会永远读到 0，于是告警安静
    #: 地永不触发。默认值给的是 ``llm`` 那一路的正确值。
    label: str = "outcome"

    def describe(self) -> str:
        return (
            f"{', '.join(self.outcomes)} 超过 {self.threshold} 次且占比 ≥ "
            f"{self.min_ratio:.0%}"
        )


#: 来源 → 限速标签名。与 T4.1 实际写入的标签名一一对应。
#:
#: **这张表存在的原因是一次"永不触发的告警"。** 两个 counter 形状不同：
#: ``observability.Observability.record_llm_failure`` 写
#: ``llm.failures_total{stage=…, outcome=…}``，而
#: ``record_parse_attempts`` 写
#: ``parsing.attempts_total{stage=…, parser=…, status=…, …}``。子集匹配
#: ``get(outcome="rate_limited")`` 在第二个 counter 上匹配不到任何 series，
#: 于是 MinerU 的 429 —— 恰恰是"provider 限速告警"最该抓的那一路 —— 一次都
#: 不会告警，而且**没有任何错误**表明它失效了。
PROVIDER_RATE_LIMIT_LABELS: dict[str, str] = {
    "llm": "outcome",
    "parsing": "status",
}

#: 来源键 → 中文名。告警消息是给人看的，而 "来源 'parsing'" 这种英文键出现在
#: 一句中文运维提示里，读的人还得自己翻译一次。原始键仍然原样留在 ``evidence``
#: 里，方便对照指标名。
_PROVIDER_SOURCE_NAMES: dict[str, str] = {
    "llm": "模型调用",
    "parsing": "解析服务",
}


#: 默认规则。``threshold=3`` 而不是 1：单次 429 是正常现象（本项目的重试
#: 策略本来就会处理它并退避），而把它报成告警只会训练运维忽略告警。
PROVIDER_RATE_LIMIT_RULES: tuple[ProviderRateLimitRule, ...] = (
    ProviderRateLimitRule(
        ALERT_PROVIDER_RATE_LIMIT, ("rate_limited", "throttled", "429"), 3, 0.2
    ),
)


def rate_limit_signals_from(
    metric: Any,
    rule: ProviderRateLimitRule,
    label: str = "",
) -> dict[str, Any]:
    """从一个 T4.1 counter 上读出限速的次数与占比。

    ``metric`` 是 :class:`~paperwhisperer.core.observability.Counter`。
    刻意**接受 Counter 而不是数字**：需要"只数某几个标签的取值"这个能力，
    而它是 T4.1 的 counter 提供的（``get(**labels)`` 是子集匹配）。如果这里
    接受一个裸字典，就必须在本模块里重新实现一遍子集匹配——那正是
    "两套指标系统"的开始。

    ``label`` 覆盖 :attr:`ProviderRateLimitRule.label`。它存在的理由是
    ``parsing.attempts_total`` 把状态记在 ``status=`` 上而不是 ``outcome=``：
    用错标签名会读到 0，而**读到 0 是一个看起来完全正常的数字**。

    分母是该 metric 的**全量**，所以"占比"的分母包含了成功与其它失败——
    这正是"限速占全部调用的比例"这句话应有的含义。
    """
    label = str(label or rule.label or "outcome")
    limited = 0
    for outcome in rule.outcomes:
        try:
            limited += int(metric.get(**{label: outcome}))
        except TypeError:  # pragma: no cover - 换成了别的 metric 形状
            continue
    total = 0
    try:
        total = int(metric.total)
    except AttributeError:  # pragma: no cover
        total = 0
    return {
        "limited": limited,
        "total": total,
        "ratio": (limited / total) if total > 0 else 0.0,
    }


def evaluate_provider_rate_limits(
    counters: Mapping[str, Any],
    *,
    rules: Sequence[ProviderRateLimitRule] = PROVIDER_RATE_LIMIT_RULES,
) -> tuple[Alert, ...]:
    """判定 provider 限速。

    ``counters`` 是 ``{来源标签: T4.1 Counter}``，例如
    ``{"llm": observability.metrics.llm_failures,
       "parsing": observability.metrics.parsing_attempts}``。键是自由的——
    告警消息里原样带出，所以运维能一眼看出是模型侧还是解析侧在降速。而键**同时
    决定读哪个标签**：见 :data:`PROVIDER_RATE_LIMIT_LABELS`，认识的两个来源
    （``llm`` / ``parsing``）会自动用各自正确的限速标签，其余来源退回规则的
    :attr:`ProviderRateLimitRule.label`。
    """
    alerts: list[Alert] = []
    for rule in rules:
        for source, metric in sorted(counters.items()):
            # ``None`` = 这个 registry 没有这个指标（例如 T4.1 尚未记录过
            # 任何一次 LLM 失败）。跳过而不是把 ``None`` 传下去：下面读的是
            # ``metric.get(...)``，而 None 上的属性读取抛的是 AttributeError
            # ——于是一个"还没有限速"的健康部署会在 health 端点 500，而它本该
            # 安静地报告"没有限速"。
            if metric is None:
                continue
            signals = rate_limit_signals_from(
                metric, rule, PROVIDER_RATE_LIMIT_LABELS.get(source, rule.label)
            )
            if signals["limited"] < rule.threshold:
                continue
            if signals["ratio"] < rule.min_ratio:
                continue
            alerts.append(
                Alert(
                    name=rule.name,
                    severity=SEVERITY_WARNING,
                    message=(
                        f"provider 对{_PROVIDER_SOURCE_NAMES.get(source, f'来源 {source!r}')}"
                        f"限速：{signals['limited']} 次（占全部调用的 {signals['ratio']:.0%}）。"
                        "上游正在要求降速；本项目的重试退避已生效，"
                        "但持续限速意味着需要降低并发或等待额度窗口恢复。"
                    ),
                    evidence={
                        "source": source,
                        "limited": signals["limited"],
                        "total": signals["total"],
                        "ratio": round(signals["ratio"], 4),
                        "outcomes": list(rule.outcomes),
                        "source_metric": getattr(metric, "name", ""),
                    },
                )
            )
    return tuple(alerts)


def alerts_from_registry(registry: Any) -> tuple[Alert, ...]:
    """直接从 T4.1 的 :class:`MetricsRegistry` 读出限速告警。

    这是给 health 端点的一行接缝——它读的是**已经存在的**指标对象，
    不新增 metric、不新增指标名，也不需要部署方改任何仪表盘就能用上。
    """
    return evaluate_provider_rate_limits(
        {
            "llm": getattr(registry, "llm_failures", None),
            "parsing": getattr(registry, "parsing_attempts", None),
        }
    )
