"""Per-section prompt definitions for document analysis.

Keeping the task text out of the orchestration code makes each analysis
contract reviewable in isolation, and lets a section be added by defining one
entry here.

There are two generations of prompt in this file and the difference between
them is the point of task T1.8. The older ones (``summary_chunk``,
``summary_merge``, ``quotes``, ``mindmap``, ``mermaid``, ``evaluation``,
``research_brief``) each ask a model to *read* a document and report what it
found. Run together they read the same paper seven times, which is why the
overview and the brief could state different contributions and why the run
cost fifteen calls.

The fact-layer prompts at the bottom of this file invert that. They never ask
the model to read the document: they ask it to *structure* a set of retrieved
passages into eight named fields, and every card afterwards is rendered from
that one result by
:mod:`paperwhisperer.analysis.paper_facts` without a model in the loop. The
two generations coexist deliberately -- the orchestrator still drives the old
path -- but the acceptance criterion for the task is a call-count and
consistency comparison between them, and deleting the old prompts before that
comparison has been run would make the number unreproducible.

The older builders are therefore kept verbatim. Their wording is part of the
baseline being measured.
"""

from __future__ import annotations

import logging

from paperwhisperer.core.text import compact_text
from paperwhisperer.llm import output
from paperwhisperer.llm.prompts import (
    ANSWER_MODES,
    build_stable_system_prompt,
    build_task_user_prompt,
)

logger = logging.getLogger(__name__)

SECTION_KEYS = (
    "summary",
    "quotes",
    "mindmap",
    "mermaid",
    "evaluation",
    "research_brief",
)

#: Sections that must succeed for an analysis to be considered usable.
REQUIRED_SECTION_KEYS = ("summary", "quotes", "mindmap")

SECTION_LABELS = {
    "summary": "概览",
    "quotes": "引用片段",
    "mindmap": "文本结构",
    "mermaid": "视觉图谱",
    "evaluation": "批判评价",
    "research_brief": "深度简报",
}

MERMAID_DIAGRAM_PREFIXES = (
    "graph ", "mindmap", "flowchart ", "pie", "sequenceDiagram", "stateDiagram", "classDiagram",
)

# --------------------------------------------------------------------------
# The fact layer (T1.8)
# --------------------------------------------------------------------------
#
# Everything below exists to serve one extraction. The old path asked five
# different sections to read the same document and report what they thought,
# which is why the overview and the brief could disagree about what the paper
# contributes and why it cost fifteen calls. The fact layer inverts that: one
# structured read, and every card a *formatting* of the result.
#
# The templates are declared as module-level constants rather than assembled
# inside a function for a reason that is load-bearing rather than cosmetic:
# ``paperwhisperer.analysis.paper_facts.prompt_version`` hashes the text below
# to produce the ``prompt_version`` stamped on every ``PaperFacts`` and every
# analysis run. A template buried in a function body could not be hashed, and a
# prompt version that is hand-maintained is a prompt version that stops
# changing on the day the prompt changes -- which is the one day it has to.

#: The eight fields of the fact layer, in the order the extraction asks for
#: them. The order is also the order every card renders in, so a fact never
#: moves position between two cards.
FACT_FIELD_ORDER = (
    "research_question",
    "contributions",
    "method",
    "datasets",
    "metrics",
    "main_results",
    "limitations",
    "reproducibility",
)

#: Chinese label per field. These are the strings a reader sees, so they live
#: beside the fields rather than being written out again by each renderer.
FACT_FIELD_LABELS = {
    "research_question": "研究问题",
    "contributions": "核心贡献",
    "method": "方法",
    "datasets": "数据集",
    "metrics": "评估指标",
    "main_results": "主要结果",
    "limitations": "局限性",
    "reproducibility": "可复现性",
}

#: Fields that hold a list of facts rather than one. A list field the paper
#: does not discuss is still *recorded* -- as a single ``not_found`` fact
#: inside the list -- so "the paper lists no datasets" and "we failed to
#: extract datasets" cannot be confused by a reader.
MULTI_FACT_FIELDS = frozenset(
    {"contributions", "datasets", "metrics", "main_results", "limitations"}
)

#: The system-prompt task key for the fact extraction.
#:
#: ``llm/prompts.py`` does not register this key, so
#: :func:`build_stable_system_prompt` falls back to its generic contract. That
#: fallback is correct -- it is the same contract every unregistered key gets,
#: and the stable prefix still carries the Evidence-first rules -- but it is a
#: gap worth closing: the task contract belongs in ``PROMPT_TASK_CONTRACTS``
#: next to the other six, and that file is not owned by this task.
FACT_TASK_KEY = "paper_facts"

#: The temperature the extraction asks for.
#:
#: Low rather than zero, on purpose. At exactly zero several providers
#: degenerate into returning a fixed prior for a structured request and refuse
#: the call, and the determinism this task needs does not come from the
#: temperature at all -- it comes from grounding, which is decided in
#: :mod:`paperwhisperer.analysis.paper_facts` in code *after* the call. A low
#: sampler plus a program that drops what it cannot cite is a different thing
#: from a low sampler and a trust-the-model pipeline, and the second is not
#: this design.
FACT_EXTRACTION_TEMPERATURE = 0.0

FACT_EXTRACTION_TASK = (
    "一次性阅读下面提供的全部证据片段，抽取这篇论文的八项事实。"
    "证据片段是本产品的检索结果，每条都有唯一 evidence_id。"
)

FACT_EXTRACTION_CONSTRAINTS = (
    "只依据给出的证据片段作答，不要使用文档之外的知识。\n"
    "每个字段都必须出现在输出中：论文没有说明的字段，value 写空字符串、"
    "status 写 \"not_found\"，不要省略字段，也不要猜测。\n"
    "每条事实的 evidence_ids 只能填写给定片段中真实存在的 evidence_id；"
    "一个 id 都不能确定时，status 写 \"not_found\"。\n"
    "quote 必须是给定片段中逐字出现的连续原文（用于回源校验），"
    "无法逐字给出时写空字符串，不要改写、不要拼接。\n"
    "value 用中文简述，控制在 60 字以内；数据集名、指标名、方法名保留原文写法。\n"
    "同一个字段有互相矛盾的说法时，status 写 \"conflicting\"，"
    "把各个说法放进 conflicting_values，value 写最主要的一种。\n"
    "只输出 JSON，不要 Markdown 代码围栏，不要解释。"
)

FACT_EXTRACTION_OUTPUT_FORMAT = """{
  "research_question": {"value": "", "evidence_ids": ["ev_..."], "status": "extracted", "quote": ""},
  "contributions": [{"value": "", "evidence_ids": ["ev_..."], "status": "extracted", "quote": ""}],
  "method": {"value": "", "evidence_ids": ["ev_..."], "status": "extracted", "quote": ""},
  "datasets": [{"value": "", "evidence_ids": ["ev_..."], "status": "extracted", "quote": ""}],
  "metrics": [{"value": "", "evidence_ids": ["ev_..."], "status": "extracted", "quote": ""}],
  "main_results": [{"value": "", "evidence_ids": ["ev_..."], "status": "extracted", "quote": ""}],
  "limitations": [{"value": "", "evidence_ids": ["ev_..."], "status": "extracted", "quote": ""}],
  "reproducibility": {"value": "", "evidence_ids": ["ev_..."], "status": "extracted", "quote": ""}
}"""

#: The part of the extraction prompt that does not vary between documents.
#: Hashed into ``prompt_version``; see the note above for why it is a constant.
FACT_EXTRACTION_TEMPLATE = "\n\n".join([
    FACT_EXTRACTION_TASK,
    FACT_EXTRACTION_CONSTRAINTS,
    FACT_EXTRACTION_OUTPUT_FORMAT,
])

#: The appraisal prompt for the evaluation card. It receives the *fact layer*
#: and a targeted retrieval pass, not the document: the critical reading is the
#: one thing a structured extraction genuinely cannot supply, and it is the
#: one thing the old path got by re-reading fifteen thousand characters.
FACT_APPRAISAL_TASK = (
    "基于下面这份结构化事实和补充证据，对这篇论文做批判性评价。"
    "事实层已经写明贡献、方法、数据、结果与局限，你只补充事实层没有覆盖的判断。"
)

FACT_APPRAISAL_CONSTRAINTS = (
    "贡献、方法、数据集、指标、主要结果、局限性必须沿用 fact_layer 给出的表述，"
    "不要改写、不要补充、不要提出相反的说法——那会造成同一结论在不同卡片里不一致。\n"
    "只写 fact_layer 没有的判断：威胁效度、基线公平性、消融充分性、外部效度、"
    "以及值得借鉴与值得质疑之处。\n"
    "依据不足的判断要明说“证据不足”，不要补全。\n"
    "只输出 JSON，不要 Markdown 代码围栏。"
)

FACT_APPRAISAL_OUTPUT_FORMAT = """{
  "strengths": ["..."],
  "weaknesses": ["..."],
  "threats_to_validity": ["..."],
  "worth_learning": ["..."],
  "open_questions": ["..."]
}"""

FACT_APPRAISAL_TEMPLATE = "\n\n".join([
    FACT_APPRAISAL_TASK,
    FACT_APPRAISAL_CONSTRAINTS,
    FACT_APPRAISAL_OUTPUT_FORMAT,
])


def document_excerpt(content, limit: int) -> str:
    return (content or "")[:limit]


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------

def build_summary_chunk_prompt(chunk: str):
    return (
        build_stable_system_prompt("summary_chunk"),
        build_task_user_prompt(
            task="从文献片段中提取 3-5 个核心观点，并找出 2-3 个最值得引用的片段。",
            constraints="每个核心观点必须是一句话；引用片段尽量保留原文措辞；不要补充文档外背景。",
            output_format="""## 核心观点
1. [观点1]
2. [观点2]
3. [观点3]

## 引用片段
- "[引用1]"
- "[引用2]""",
            input_blocks=[("document_excerpt", chunk)],
        ),
    )


def build_summary_merge_prompt(combined: str):
    return (
        build_stable_system_prompt("summary_merge"),
        build_task_user_prompt(
            task="把长文献不同片段的摘要整合成一份完整、连贯、去重后的论文概要。",
            constraints="合并重复观点；保留关键方法、贡献、实验结论和引用片段；不要添加片段中没有的信息。",
            output_format="""## 核心观点
1. [整合后的观点1]
2. [整合后的观点2]

## 引用片段
- "[整合后的引用1]"
- "[整合后的引用2]""",
            input_blocks=[("chunk_summaries", combined)],
        ),
    )


# --------------------------------------------------------------------------
# Other sections
# --------------------------------------------------------------------------

def build_quotes_prompt(content):
    return (
        build_stable_system_prompt("quotes"),
        build_task_user_prompt(
            task="从文献中提取 3-5 个最值得引用的原句、定义、结论或核心观点。",
            constraints="优先选择能支撑论文主张的方法、发现或结论；尽量保留原文措辞；不要把普通摘要改写成引用。",
            output_format="""## 引用片段
1. "[原句1]"
2. "[原句2]"
3. "[原句3]""",
            input_blocks=[("document_excerpt", document_excerpt(content, 15000))],
        ),
    )


def build_mindmap_prompt(content):
    return (
        build_stable_system_prompt("mindmap"),
        build_task_user_prompt(
            task="为文献生成文本格式的研究结构图，帮助用户快速定位论文逻辑。",
            constraints="覆盖研究问题、核心方法、实验或论证、关键结论、局限性；层级控制在 3-4 层；节点短句化。",
            output_format="""## 思维导图
论文主题
├── 研究问题
├── 方法框架
│   ├── [关键模块]
│   └── [关键模块]
├── 证据与实验
└── 结论与局限""",
            input_blocks=[("document_excerpt", document_excerpt(content, 10000))],
        ),
    )


def build_mermaid_prompt(content):
    return (
        build_stable_system_prompt("mermaid"),
        build_task_user_prompt(
            task="生成可渲染的 Mermaid 论文结构图代码。",
            constraints=(
                "必须以 graph TD 或 graph LR 开头；节点 ID 只能包含字母、数字和下划线；"
                "节点文本使用方括号；不超过 20 个节点；节点文本不要包含复杂 LaTeX 公式；"
                "不要输出 Markdown 代码围栏。"
            ),
            output_format="""graph TD
    A[论文标题]
    A --> B[研究问题]
    A --> C[方法]
    A --> D[实验]
    A --> E[结论]""",
            input_blocks=[("document_excerpt", document_excerpt(content, 4000))],
        ),
    )


# --------------------------------------------------------------------------
# The fact layer: one extraction, then renderings
# --------------------------------------------------------------------------

#: How much evidence text the extraction prompt carries. A cap, and a real one:
#: the extraction is a single call, so its window is the whole fact layer's
#: budget and an uncapped pool would make a long paper fail for a reason that
#: has nothing to do with the paper. Whatever the cap drops is recorded as a
#: degradation by the caller rather than silently truncated -- the plan's
#: "silent truncation must be zero" gate.
FACT_EVIDENCE_BUDGET_CHARS = 24000

#: How much of the *digest* (the rendered fact layer) the appraisal prompt
#: carries. Small on purpose: the appraisal is one call whose entire job is to
#: disagree with the facts in the places a structured extraction cannot.
FACT_DIGEST_BUDGET_CHARS = 4000

#: The same for the quote-selection fallback. See
#: :func:`paperwhisperer.analysis.paper_facts.build_quote_card` for why this
#: call exists at all and why it usually does not fire.
FACT_QUOTE_EVIDENCE_BUDGET_CHARS = 8000


def format_evidence_blocks(evidence, budget: int = 0) -> tuple[str, int]:
    """Render retrieved passages as the numbered blocks a prompt cites.

    The ``[ev_xxxx]`` prefix is the whole contract: it is the id the model is
    told it may cite, and it is the id that
    :mod:`paperwhisperer.analysis.paper_facts` resolves back to a real
    :class:`~paperwhisperer.ingestion.schema.Evidence` before the fact is kept.
    Anything the model cites that is not one of these strings is dropped on
    purpose -- see that module's docstring for why an unresolvable citation is
    worse than none.

    Returns the text and how many passages were left out, because a truncation
    that the caller cannot see is the failure the plan's release gate names.
    ``budget`` of 0 means "no cap", which is what the tests use so that
    truncation is a property of one caller rather than of the formatter.

    Per-passage truncation marks itself with an ellipsis. A passage cut
    mid-word with no marker would let a model quote across the seam and
    produce a quote that cannot be found in the source, which the verifier
    would (correctly) reject for reasons the model could not have known. A
    whole passage is dropped rather than clipped to a few characters: a
    passage with no room left cannot support a quote, and an empty block with
    a real ``evidence_id`` in it would invite exactly that.
    """
    blocks: list[str] = []
    dropped = 0
    used = 0
    for item in evidence or ():
        evidence_id = str(getattr(item, "evidence_id", "") or "")
        if not evidence_id:
            continue
        text = str(getattr(item, "text", "") or "").strip()
        if not text:
            continue
        label = str(getattr(item, "citation_label", "") or "document")
        heading = " › ".join(str(part) for part in (getattr(item, "heading_path", ()) or ()))
        header = f"[{evidence_id}] {label}"
        if heading:
            header = f"{header} | {heading}"
        block = f"{header}\n{text}"
        if budget and used and used + len(block) > budget:
            dropped += 1
            continue
        used += len(block)
        blocks.append(block)
    return "\n\n".join(blocks), dropped


def build_fact_extraction_prompt(evidence, budget: int = FACT_EVIDENCE_BUDGET_CHARS):
    """The one structured call. Its output is the fact layer.

    Deliberately not a summary prompt with a JSON coat: the eight fields are
    named in the task, the output format and the constraints, and the
    constraints carry the two rules the rest of the pipeline depends on -- a
    field the paper does not state is written as ``not_found`` rather than
    omitted, and every ``evidence_id`` is one the prompt actually printed.
    """
    return (
        build_stable_system_prompt(FACT_TASK_KEY),
        build_task_user_prompt(
            task=FACT_EXTRACTION_TASK,
            constraints=FACT_EXTRACTION_CONSTRAINTS,
            output_format=FACT_EXTRACTION_OUTPUT_FORMAT,
            input_blocks=[("evidence_blocks", format_evidence_blocks(evidence, budget)[0])],
        ),
    )


def build_fact_appraisal_prompt(fact_digest, evidence, budget: int = FACT_DIGEST_BUDGET_CHARS):
    """Evaluation = fact layer + targeted retrieval. One call, narrow inputs.

    The digest rather than the document is the point. A model asked to
    summarize a paper will restate the contributions in its own words, and two
    cards asked the same question independently produce two different
    wordings -- which is the inconsistency this whole module exists to remove.
    Told to reuse the given wording and told only to add what the facts do not
    cover, the appraisal has one job and cannot create a second opinion about
    the paper's claims.
    """
    return (
        build_stable_system_prompt("evaluation"),
        build_task_user_prompt(
            task=FACT_APPRAISAL_TASK,
            constraints=FACT_APPRAISAL_CONSTRAINTS,
            output_format=FACT_APPRAISAL_OUTPUT_FORMAT,
            input_blocks=[
                ("fact_layer", document_excerpt(fact_digest, budget)),
                (
                    "critical_appraisal_evidence",
                    format_evidence_blocks(evidence, budget)[0],
                ),
            ],
        ),
    )


def build_fact_quote_prompt(evidence, budget: int = FACT_QUOTE_EVIDENCE_BUDGET_CHARS):
    """The quote-selection fallback. Normally not called at all.

    The quotes card is a rendering of the quotes the extraction already
    verified against the source, which is strictly better than the old free
    prompt: those quotes are known to be verbatim. This builder exists for the
    case where the extraction came back with no verified quote at all, so the
    required section does not simply go empty. The caller only reaches it when
    that has happened, and records that it did.
    """
    return (
        build_stable_system_prompt("quotes"),
        build_task_user_prompt(
            task=(
                "从下面这些已检索的证据片段里挑出 3-5 个最值得引用的原句，"
                "覆盖方法、主要结果和局限性。"
            ),
            constraints=(
                "只从给定片段中逐字摘录，不要改写、不要拼接、不要补充片段外的内容；"
                "每条引用后面用 [ev_xxxx] 标注它来自哪一条证据。"
            ),
            output_format="""1. "[逐字原文]" [ev_xxxx]
2. "[逐字原文]" [ev_xxxx]""",
            input_blocks=[("evidence_blocks", format_evidence_blocks(evidence, budget)[0])],
        ),
    )


def build_mermaid_deepening_prompt(section_tree_text, budget: int = FACT_DIGEST_BUDGET_CHARS):
    """The optional second Mermaid layer, over an already-correct first layer.

    Only the levels *below* the section tree are the model's to draw. The
    section level itself is generated deterministically from the document's own
    headings (see
    :func:`paperwhisperer.analysis.paper_facts.build_mermaid_sections`), and
    this prompt never sees it as something to produce -- it is handed the tree
    as an input to refine, which is the only arrangement in which a model
    cannot quietly redraw the paper's structure into the shape it expected.
    """
    return (
        build_stable_system_prompt("mermaid"),
        build_task_user_prompt(
            task=(
                "下面是论文真实的章节结构。请在保持这些章节节点与层级完全不变的前提下，"
                "补充每个章节内部的关键子节点。"
            ),
            constraints=(
                "不得新增、删除、改名或移动任何章节节点；"
                "只能为已有章节添加子节点，每章最多 3 个；"
                "节点 ID 只能包含字母、数字和下划线；不要输出 Markdown 代码围栏。"
            ),
            output_format="""graph TD
    A[论文标题]
    A --> B[章节]
    B --> B1[子节点]""",
            input_blocks=[("section_tree", document_excerpt(section_tree_text, budget))],
        ),
    )


def build_evaluation_prompt(content):
    return (
        build_stable_system_prompt("evaluation"),
        build_task_user_prompt(
            task="对文献做总结性评价，兼顾审稿视角、读者复盘和后续研究启发。",
            constraints="贡献和局限必须能从文档内容推出；历史地位不确定时说明不确定；避免泛泛而谈。",
            output_format="""## 论文评价

### 主要贡献
[评价内容]

### 历史地位
[评价内容]

### 主要优点
- 优点1
- 优点2

### 局限性
- 局限性1
- 局限性2

### 值得学习的地方
- 学习点1
- 学习点2""",
            input_blocks=[("document_excerpt", document_excerpt(content, 15000))],
        ),
    )


def build_research_brief_prompt(content):
    return (
        build_stable_system_prompt("research_brief"),
        build_task_user_prompt(
            task="生成一份可直接用于组会、文献综述和后续检索决策的深度阅读简报。",
            constraints=(
                "所有结论必须能从文档内容推出；证据不足时标注不确定；"
                "推荐检索词用英文短语；避免泛泛背景介绍。"
            ),
            output_format="""## 深度阅读简报

### 一句话定位
[这篇论文解决什么问题、适合放在哪条研究脉络]

### 核心贡献与适用场景
- 贡献1：对应证据或章节线索
- 贡献2：对应证据或章节线索

### 证据-结论链
| 结论 | 文档依据 | 可信度 |
| --- | --- | --- |
| [结论] | [依据] | 高/中/低 |

### 复现检查清单
- 数据与输入要求
- 方法/模型关键变量
- 实验或评估指标
- 潜在失败点

### 局限与后续问题
- 局限1
- 后续问题1

### 推荐检索关键词
- keyword phrase 1
- keyword phrase 2
- keyword phrase 3""",
            input_blocks=[("document_excerpt", document_excerpt(content, 18000))],
        ),
    )


# --------------------------------------------------------------------------
# Q&A
# --------------------------------------------------------------------------

def build_answer_prompt(question: str, document_window: str, history_block: str, answer_mode: str):
    """Assemble the follow-up question prompt within explicit context budgets."""
    input_blocks = [("document_excerpt", document_window)]
    if history_block:
        input_blocks.append(("recent_qa_history", history_block))
    input_blocks.append(("user_question", question))

    constraints = (
        "优先依据 document_excerpt；history 只用于理解追问上下文；"
        "如果文档没有答案，明确说明缺少依据；回答要简洁但保留关键证据。"
        f"\n{ANSWER_MODES[answer_mode]}"
    )
    return (
        build_stable_system_prompt("qa"),
        build_task_user_prompt(
            task="回答用户关于当前文档的追问。",
            constraints=constraints,
            output_format="先给直接答案；必要时用要点列出依据、公式或不确定处。",
            input_blocks=input_blocks,
        ),
    )


def build_search_rewrite_prompt(query: str, context_excerpt: str):
    return (
        build_stable_system_prompt("search_rewrite"),
        build_task_user_prompt(
            task=(
                "Rewrite the paper search request into a concise English query "
                "for Semantic Scholar and arXiv."
            ),
            constraints="""- rewritten_query must be concise English.
- Preserve specific-paper intent; if a nickname or shorthand points to a known paper, prefer the canonical title.
- Do not broaden a specific-paper query into a vague family query.
- Only expand when the intent is ambiguous.
- topics must be short.
- why must be Chinese.
- Return JSON only, without markdown fences.""",
            output_format="""{
  "original_query": "original user query",
  "rewritten_query": "better english academic query",
  "topics": ["topic 1", "topic 2", "topic 3"],
  "why": "brief reason in Chinese"
}""",
            input_blocks=[
                ("user_query", query),
                ("optional_context", context_excerpt),
            ],
        ),
    )


def normalize_search_rewrite(raw_response: str, original_query: str, model: str) -> dict:
    """The model's rewrite, or a ``ValueError`` naming why there isn't one.

    Search rewriting is one of the few structured calls in this codebase with no
    degraded rendering to fall back on -- there is no "here is the query
    unrewritten" card, so a failure here is a failure. What the shared layer
    buys is the *diagnosis*: the old ``except json.JSONDecodeError`` reported
    the decoder's own text, which says nothing about whether the model sent
    prose, sent an array, or sent JSON wrapped in a fence. The taxonomy names
    which of those it was, and a fenced or full-width reply is now recovered
    rather than thrown away -- the model fences most of its answers here.

    The English messages are kept because this path is operator-facing in
    logs, and the Chinese degradation line is appended so a support ticket gets
    something readable.
    """
    parsed = output.parse_json_object(raw_response, label="搜索改写")
    if parsed.status is output.OutputStatus.WRONG_SHAPE:
        # Kept as its own sentence, and not folded into the generic failure
        # below: "returned an array where an object was asked for" is a
        # different operator problem from "returned prose we could not read",
        # and this is the one message here that already said which it was.
        raise ValueError(
            "Search query rewriting returned an unexpected payload. "
            f"{parsed.user_message()}"
        )
    if not parsed.ok:
        raise ValueError(
            f"Search query rewriting failed: {parsed.user_message()} "
            f"(status={parsed.status.value}, detail={parsed.detail or 'none'})"
        ) from None
    rewrite_meta = parsed.value
    if not isinstance(rewrite_meta, dict):  # pragma: no cover - layer guarantees this
        raise ValueError("Search query rewriting returned an unexpected payload.")

    rewritten_query = compact_text(rewrite_meta.get("rewritten_query") or "", limit=240)
    if not rewritten_query:
        raise ValueError("Search query rewriting returned an empty rewritten query.")
    if parsed.recoveries:
        logger.warning(
            "搜索改写回复经过修复后才可解析（%s），结果可能与模型原文不一致。",
            "、".join(item.value for item in parsed.recoveries),
        )

    return {
        "original_query": original_query,
        "rewritten_query": rewritten_query,
        "topics": rewrite_meta.get("topics") or [],
        "reason": str(rewrite_meta.get("why") or "").strip(),
        "model": model,
    }
