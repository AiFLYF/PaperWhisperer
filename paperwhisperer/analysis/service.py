"""Analysis service: run an analysis and persist its session.

This is the layer the API routes call. It owns the side effects of an
analysis — writing the Markdown report, minting the session token and storing
the session — so route handlers stay thin.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

from paperwhisperer.analysis.orchestrator import DocumentAnalyzer
from paperwhisperer.core import config
from paperwhisperer.core.text import build_session_id, now_iso
from paperwhisperer.sessions.store import (
    build_document_excerpt,
    build_session_payload,
    generate_session_token,
    get_session_document_content,
    write_session_payload,
)

logger = logging.getLogger(__name__)


def render_markdown_report(result: dict, original_filename: str, version: str) -> str:
    """Render the analysis as a standalone Markdown report."""
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed = result.get("elapsed_seconds", "N/A")

    parts = [
        "# PaperWhisperer 分析报告",
        "",
        f"> 生成时间: {generated_at}",
        f"> 源文件: {original_filename}",
        f"> 耗时: {elapsed}s",
        "",
        "---",
        "",
        "## AI 摘要",
        "",
        result.get("summary", ""),
        "",
        "---",
        "",
        "## 引用片段",
        "",
        result.get("quotes", ""),
        "",
        "---",
        "",
        "## 思维导图",
        "",
        result.get("mindmap", ""),
        "",
        "---",
        "",
    ]
    if result.get("evaluation"):
        parts.extend([f"## 论文评价\n\n{result['evaluation']}\n\n---\n"])
    if result.get("research_brief"):
        parts.extend([f"## 深度阅读简报\n\n{result['research_brief']}\n\n---\n"])
    parts.extend([f"## 元信息\n\n- 版本: {version}\n- 字符数: {result.get('char_count', 0)}\n"])
    return "".join(parts)


def finalize_analysis_result(
    result: dict,
    analyzer: DocumentAnalyzer,
    original_filename: str,
    generate_evaluation_bool: bool,
    session_id,
    generate_research_brief_bool: bool = True,
) -> dict:
    """Write the report, mint a session token and persist the session.

    The raw token is placed on ``result`` exactly here; only its hash is
    stored server-side.
    """
    safe_session_id = build_session_id(session_id)
    session_token = generate_session_token()
    base_name = os.path.splitext(original_filename)[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        config.OUTPUT_FOLDER, f"{base_name}_analysis_{timestamp}.md"
    )

    markdown = render_markdown_report(result, original_filename, analyzer.version)
    with open(output_file, "w", encoding="utf-8") as handle:
        handle.write(markdown)

    result["output_file"] = output_file
    result["session_id"] = safe_session_id
    result["session_token"] = session_token
    result["source_filename"] = original_filename

    write_session_payload(
        safe_session_id,
        build_session_payload(
            session_id=safe_session_id,
            source_filename=original_filename,
            document_content=analyzer.document_content,
            analysis=result,
            session_token=session_token,
        ),
    )
    return result


def analyze_saved_file(
    file_path,
    original_filename,
    api_key,
    generate_mermaid_bool,
    generate_evaluation_bool,
    session_id,
    generate_research_brief_bool: bool = True,
) -> dict:
    """Analyze a file on disk and return the finalized, session-backed result."""
    resolved_api_key = config.resolve_api_key(api_key)
    if not resolved_api_key:
        raise ValueError("API key is required. Provide api_key or set OPENAI_API_KEY.")

    analyzer = DocumentAnalyzer(resolved_api_key)
    result = analyzer.analyze(
        file_path,
        generate_mermaid_bool,
        generate_evaluation_bool,
        generate_research_brief_bool,
    )
    return finalize_analysis_result(
        result=result,
        analyzer=analyzer,
        original_filename=original_filename,
        generate_evaluation_bool=generate_evaluation_bool,
        session_id=session_id,
        generate_research_brief_bool=generate_research_brief_bool,
    )


def record_qa_turn(session_payload: dict, question: str, answer: str, answer_mode: str) -> None:
    """Append a Q&A turn and refresh the cached document excerpt."""
    qa_history = session_payload.get("qa_history", [])
    qa_history.append(
        {
            "question": question,
            "answer": answer,
            "answer_mode": answer_mode,
            "timestamp": now_iso(),
        }
    )
    session_payload["qa_history"] = qa_history
    session_payload["document_excerpt"] = build_document_excerpt(
        get_session_document_content(session_payload)
    )
