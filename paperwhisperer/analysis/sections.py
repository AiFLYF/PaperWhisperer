"""Per-section prompt definitions for document analysis.

Keeping the task text out of the orchestration code makes each analysis
contract reviewable in isolation, and lets a section be added by defining one
entry here.
"""

from __future__ import annotations

import json

from paperwhisperer.core.text import compact_text, extract_json_object
from paperwhisperer.llm.prompts import (
    ANSWER_MODES,
    build_stable_system_prompt,
    build_task_user_prompt,
)

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
    try:
        rewrite_meta = json.loads(extract_json_object(raw_response))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Search query rewriting failed: {exc}") from exc
    if not isinstance(rewrite_meta, dict):
        raise ValueError("Search query rewriting returned an unexpected payload.")

    rewritten_query = compact_text(rewrite_meta.get("rewritten_query") or "", limit=240)
    if not rewritten_query:
        raise ValueError("Search query rewriting returned an empty rewritten query.")

    return {
        "original_query": original_query,
        "rewritten_query": rewritten_query,
        "topics": rewrite_meta.get("topics") or [],
        "reason": str(rewrite_meta.get("why") or "").strip(),
        "model": model,
    }
