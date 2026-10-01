"""Prompt construction.

Prompts are built from a stable system prefix (identity + format rules +
quality rules + a per-task contract) and a structured user message whose parts
are wrapped in XML-ish tags. The fixed system prefix keeps the response format
consistent across models, while the tagged user blocks keep variable content
from colliding with the instructions.
"""

from __future__ import annotations

import re

PROMPT_STABLE_PREFIX = """你是 PaperWhisperer 的学术研究助手。目标是帮助研究者快速理解论文、保留证据、识别结构与局限。
始终优先依据用户提供的文档内容，不编造文献细节；文档没有依据时明确说明。
输出默认使用中文；只有检索词、JSON 字段值或代码任务明确要求时才使用英文。
你没有任何工具可用：不得请求文件检索、命令执行或任何外部访问。你看到的全部依据就是本消息中的文档内容。
当文档内容明显被截断、不足以回答问题时，直接说明依据不足并指出缺失的部分，不要试图自行寻找其他来源。"""

PROMPT_FORMULA_RULES = """公式与格式规则：
- 行内公式必须使用 $...$，不要使用 \\( ... \\)。
- 块级公式必须使用 $$...$$，不要使用 \\[ ... \\]。
- 公式内部的下划线和星号不要做 Markdown 转义。
- 不要把公式放进普通代码块。"""

PROMPT_QUALITY_RULES = """回答质量规则：
- 先给结论，再给依据或层级展开。
- 区分文档事实、合理推断和不确定内容。
- 避免空泛套话，优先输出可直接用于阅读、复盘或追问的内容。
- 保持结构清晰，标题层级不要过深。"""

PROMPT_TASK_CONTRACTS = {
    "summary_chunk": "任务契约：从文献片段提取核心观点和高价值引用，保留关键术语、方法、数据集、结论和公式。",
    "summary_merge": "任务契约：整合多个片段摘要，去重、合并同义观点，并形成一份连贯的整篇论文概要。",
    "paper_facts": (
        "任务契约：一次性抽取结构化事实，字段为 research_question / contributions / method / datasets / metrics / main_results / limitations / reproducibility。\n"
        "每条事实必须给出支持它的证据编号（形如 [ev_xxxx]），并附一段论文原句作为 quote。\n"
        "论文没有说明的字段必须留空并在 extraction_status 标注 not_found，不得根据经验或常识补全。\n"
        "只允许引用本消息中给出编号的证据；不得引用不存在的编号。\n"
        "不要总结、不要评价、不要添加论文中不存在的结论或数字。"
    ),
    "quotes": "任务契约：提取最值得引用的原句或接近原文的关键表述，不要改写成泛泛总结。",
    "mindmap": "任务契约：识别论文的研究问题、方法、实验、结论与局限，输出文本层级结构。",
    "mermaid": "任务契约：输出可渲染的 Mermaid 结构图代码，只输出 Mermaid，不解释。",
    "evaluation": "任务契约：以审稿和读者复盘视角评价论文贡献、优点、局限、历史地位与学习价值。",
    "research_brief": "任务契约：生成面向研究决策的深度阅读简报，连接贡献、证据、复现风险和后续检索方向。",
    "qa": "任务契约：基于当前文档和最近问答历史回答用户追问；文档优先级高于历史。",
    "search_rewrite": (
        "Task contract: rewrite paper-search intent into concise English retrieval "
        "queries and return JSON only."
    ),
}

ANSWER_MODES = {
    "evidence": "证据模式：先给直接结论，再列出文档依据、可推断内容和不确定处。",
    "explain": "讲解模式：用教学方式分步骤解释概念、方法和公式，必要时补充类比。",
    "critique": "评审模式：从贡献、假设、局限、威胁效度和可改进点进行批判性分析。",
    "reproduce": "复现模式：输出复现步骤、关键变量、数据/实验依赖、风险点和检查清单。",
}

DEFAULT_ANSWER_MODE = "evidence"

SELF_CHECK_BLOCK = (
    "提交前确认：没有编造文档外事实；公式格式符合要求；输出结构与任务契约一致。"
)


def normalize_answer_mode(value) -> str:
    """Map any input to a known answer mode, defaulting to ``evidence``."""
    mode = str(value or "").strip().lower()
    return mode if mode in ANSWER_MODES else DEFAULT_ANSWER_MODE


def build_stable_system_prompt(task_key: str) -> str:
    task_contract = PROMPT_TASK_CONTRACTS.get(
        task_key, "任务契约：完成用户指定的学术阅读任务。"
    )
    return "\n\n".join(
        [PROMPT_STABLE_PREFIX, PROMPT_FORMULA_RULES, PROMPT_QUALITY_RULES, task_contract]
    )


#: A sequence that a model would read as an opening or closing tag.
#:
#: The block delimiters are XML-ish, so untrusted content that contains one can
#: close its own block and have everything after it read as a *sibling of the
#: real instruction blocks*. That is not a hypothetical: a document containing
#: the literal text ``</document_excerpt>`` does exactly this, and the result
#: is that the document gets to write the task. Sanitising the tag name (which
#: this function already does) protects the frame but not the content.
_TAG_LIKE = re.compile(r"<(?=[A-Za-z_][A-Za-z0-9_]*\s*/?>)")


def _neutralize_tags(text: str) -> str:
    """Break tag-like sequences in untrusted content without mangling prose.

    Only a ``<`` that actually begins a tag is touched, so LaTeX (``a<b``),
    comparisons (``x <= 3``) and prose are untouched. The bracket is replaced
    with a fullwidth one rather than deleted: the sentence still reads
    correctly to a human and to the model, and the sequence can no longer
    close a block. Deleting would silently alter the document's meaning, and
    escaping to ``&lt;`` would leave the content looking like markup it is not.
    """
    return _TAG_LIKE.sub("＜", text)


def build_prompt_block(tag: str, content) -> str:
    """Wrap content in a named block, neutralising any block syntax inside it.

    The document is data. It goes in the prompt, and the prompt's block
    structure is what keeps the system instructions and the system's data
    distinguishable -- so content that can reconstruct that structure must not
    be able to.
    """
    safe_tag = re.sub(r"[^A-Za-z0-9_]", "_", str(tag or "input")).strip("_") or "input"
    body = _neutralize_tags(str(content or "").strip())
    return f"<{safe_tag}>\n{body}\n</{safe_tag}>"


def build_task_user_prompt(
    task: str,
    input_blocks,
    constraints: str = "",
    output_format: str = "",
) -> str:
    """Assemble the structured user prompt: task, constraints, format, inputs, self-check."""
    parts = [build_prompt_block("task", task)]
    if constraints:
        parts.append(build_prompt_block("constraints", constraints))
    if output_format:
        parts.append(build_prompt_block("output_format", output_format))
    for tag, content in input_blocks:
        parts.append(build_prompt_block(tag, content))
    parts.append(build_prompt_block("self_check", SELF_CHECK_BLOCK))
    return "\n\n".join(parts)
