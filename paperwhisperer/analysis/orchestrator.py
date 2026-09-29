"""Document analysis orchestration.

Runs the analysis sections concurrently under a shared LLM concurrency cap and
turns each one into a self-describing result. A section that fails is reported
in place rather than aborting the run; only a total failure of the required
sections is treated as a failed analysis.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import re
import time

from paperwhisperer.analysis import sections
from paperwhisperer.core import config
from paperwhisperer.core.errors import is_retryable_llm_error
from paperwhisperer.core.text import compact_text, trim_text_for_log
from paperwhisperer.documents.loader import DocumentLoader, TextChunker
from paperwhisperer.llm.client import LLMClient
from paperwhisperer.llm.prompts import normalize_answer_mode

logger = logging.getLogger(__name__)

#: Section name -> whether it is optional (can be switched off by the caller).
OPTIONAL_SECTIONS = ("mermaid", "evaluation", "research_brief")


def build_section_result(
    status: str,
    content: str = "",
    error: str = "",
    retryable: bool = False,
) -> dict:
    return {
        "status": status,
        "content": content or "",
        "error": error or "",
        "retryable": bool(retryable),
    }


def build_analysis_metadata(sections_map: dict) -> dict:
    """Derive follow-up suggestions from which sections actually succeeded."""
    section_statuses = {
        name: str((section or {}).get("status") or "empty")
        for name, section in (sections_map or {}).items()
    }
    completed = [
        sections.SECTION_LABELS.get(name, name)
        for name, status in section_statuses.items()
        if status == "success"
    ]
    failed = [
        sections.SECTION_LABELS.get(name, name)
        for name, status in section_statuses.items()
        if status == "failed"
    ]
    disabled = [
        sections.SECTION_LABELS.get(name, name)
        for name, status in section_statuses.items()
        if status == "disabled"
    ]

    suggested_questions = [
        "这篇论文要解决的核心问题是什么？",
        "它的主要方法和创新点分别是什么？",
        "实验或论证最支持哪些结论？",
        "这篇论文有哪些局限性和后续研究方向？",
    ]
    if section_statuses.get("quotes") == "success":
        suggested_questions.append("哪些原文片段最适合在综述或笔记中引用？")
    if section_statuses.get("evaluation") == "success":
        suggested_questions.append("如果我要复现或扩展这篇论文，应该优先关注什么？")
    if section_statuses.get("research_brief") == "success":
        suggested_questions.append("这篇论文最适合放进哪条研究脉络或综述段落？")

    next_actions = [
        {"label": "追问方法细节", "prompt": "请解释这篇论文的方法流程，并指出每一步解决了什么问题。"},
        {"label": "整理局限性", "prompt": "请基于文档总结这篇论文的局限性，并给出可能的改进方向。"},
        {"label": "生成阅读路线", "prompt": "请把这篇论文拆成适合精读的阅读路线和检查清单。"},
    ]
    if section_statuses.get("research_brief") == "success":
        next_actions.extend([
            {"label": "复现路线", "prompt": "请基于深度简报生成一份复现路线，包括数据、实验变量、依赖和风险点。"},
            {"label": "找后续工作", "prompt": "请提炼 5 个英文检索关键词，用于寻找这篇论文的后续工作或相邻研究。"},
        ])
    if section_statuses.get("evaluation") == "disabled":
        next_actions.append({"label": "手动评价", "prompt": "请基于当前文档补充一份批判性评价。"})
    if failed:
        next_actions.append(
            {"label": "补全失败部分", "prompt": f"请重新生成以下分析部分：{', '.join(failed)}。"}
        )

    return {
        "suggested_questions": suggested_questions[:6],
        "next_actions": next_actions[:5],
        "analysis_status": {
            "quality": "partial" if failed else "complete",
            "completed_sections": completed,
            "failed_sections": failed,
            "disabled_sections": disabled,
            "section_statuses": section_statuses,
        },
    }


class DocumentAnalyzer:
    """Owns one analysis run: document text, section prompts and LLM dispatch."""

    def __init__(self, api_key: str):
        self.version = config.APP_VERSION
        self.client = LLMClient(api_key)
        self.chunker = TextChunker(4000, 200)
        self.summary_chunk_workers = min(3, self.client.max_concurrency)
        self.analysis_workers = min(5, self.client.max_concurrency)
        self.document_content = ""

    # -- section implementations ------------------------------------------

    def _generate_summary_chunk(self, chunk: str) -> str:
        system_prompt, user_prompt = sections.build_summary_chunk_prompt(chunk)
        return self.client.complete(system_prompt, user_prompt)

    def _merge_summaries(self, summaries: list[str]) -> str | None:
        if not summaries:
            return None
        if len(summaries) == 1:
            return summaries[0]

        combined = "\n\n--- 章节 ---\n\n".join(summaries)
        system_prompt, user_prompt = sections.build_summary_merge_prompt(combined)
        return self.client.complete(system_prompt, user_prompt)

    def generate_summary(self, content: str) -> str:
        chunks = self.chunker.chunk_text(content)
        if len(chunks) == 1:
            return self._generate_summary_chunk(content)

        worker_count = self._get_worker_count(len(chunks), self.summary_chunk_workers)
        if worker_count == 1:
            chunk_summaries = [
                summary for summary in map(self._generate_summary_chunk, chunks) if summary
            ]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
                chunk_summaries = list(
                    filter(None, executor.map(self._generate_summary_chunk, chunks))
                )

        if len(chunk_summaries) > 1:
            return self._merge_summaries(chunk_summaries)
        return chunk_summaries[0] if chunk_summaries else "无法生成摘要"

    def extract_quotes(self, content: str) -> str:
        system_prompt, user_prompt = sections.build_quotes_prompt(content)
        return self.client.complete(system_prompt, user_prompt)

    def generate_mindmap(self, content: str) -> str:
        system_prompt, user_prompt = sections.build_mindmap_prompt(content)
        return self.client.complete(system_prompt, user_prompt)

    def generate_mermaid_mindmap(self, content: str) -> str | None:
        system_prompt, user_prompt = sections.build_mermaid_prompt(content)
        result = self.client.complete(system_prompt, user_prompt)
        if not result:
            return None
        return self._normalize_mermaid_source(result)

    @staticmethod
    def _normalize_mermaid_source(result: str) -> str:
        """Strip preamble so the payload starts at the diagram declaration.

        Models often emit a sentence of explanation before the diagram; the
        frontend renders the string verbatim, so that preamble would break the
        parse. Fenced fallbacks are handled for the rare case where no known
        diagram keyword appears.
        """
        lines = result.strip().split("\n")
        start_index = -1
        for index, line in enumerate(lines):
            if any(line.strip().startswith(prefix) for prefix in sections.MERMAID_DIAGRAM_PREFIXES):
                start_index = index
                break

        if start_index != -1:
            result = "\n".join(lines[start_index:]).strip()
        else:
            match = re.search(
                r"```(?:mermaid)?\s*\n(.*?)\n```", result, re.DOTALL | re.IGNORECASE
            )
            result = match.group(1).strip() if match else "graph TD\n" + result

        if not any(result.startswith(prefix) for prefix in sections.MERMAID_DIAGRAM_PREFIXES):
            result = "graph TD\n" + result
        return DocumentAnalyzer._strip_mermaid_fence(result)

    @staticmethod
    def _strip_mermaid_fence(source: str) -> str:
        """Drop the code fence that follows the diagram body.

        Slicing from the diagram keyword keeps everything after it, including
        the closing ``` of the block the model wrapped the diagram in. Left in
        place that fence reaches the renderer as diagram source and the parse
        fails, so it has to come off here rather than at render time.
        """
        lines = source.split("\n")
        end = len(lines)
        while end > 1 and lines[end - 1].strip() in {"```", "~~~"}:
            end -= 1
        body = "\n".join(lines[:end]).rstrip()
        # A model can also close the fence without a newline before it.
        return re.sub(r"(?:\s*(?:```|~~~))\s*$", "", body)

    def generate_evaluation(self, content: str) -> str:
        system_prompt, user_prompt = sections.build_evaluation_prompt(content)
        return self.client.complete(system_prompt, user_prompt)

    def generate_research_brief(self, content: str) -> str:
        system_prompt, user_prompt = sections.build_research_brief_prompt(content)
        return self.client.complete(system_prompt, user_prompt)

    # -- Q&A ---------------------------------------------------------------

    @staticmethod
    def _build_history_block(history) -> str:
        history = history or []
        history_sections = []
        used_history_chars = 0

        for turn in reversed(history):
            question_text = trim_text_for_log(turn.get("question", ""), limit=400)
            answer_text = trim_text_for_log(turn.get("answer", ""), limit=800)
            if not question_text and not answer_text:
                continue

            section = f"Q: {question_text}\nA: {answer_text}"
            if len(section) > config.QA_TURN_BUDGET:
                section = section[: config.QA_TURN_BUDGET].rstrip() + "\n...[truncated]"

            if used_history_chars + len(section) > config.QA_HISTORY_BUDGET:
                break

            history_sections.append(section)
            used_history_chars += len(section)

        history_sections.reverse()
        return "\n\n---\n\n".join(history_sections)

    def _build_answer_prompts(self, question, history=None, answer_mode="evidence"):
        if not self.document_content:
            raise ValueError("没有文档内容，请先上传文档进行分析。")

        history_block = self._build_history_block(history)
        document_window = self.document_content[: config.DOCUMENT_WINDOW_BUDGET["qa"]]
        return sections.build_answer_prompt(
            question=question,
            document_window=document_window,
            history_block=history_block,
            answer_mode=normalize_answer_mode(answer_mode),
        )

    def answer_question(self, question, history=None, answer_mode="evidence") -> str:
        system_prompt, user_prompt = self._build_answer_prompts(
            question, history=history, answer_mode=answer_mode
        )
        return self.client.complete(system_prompt, user_prompt)

    def stream_answer_question(self, question, history=None, answer_mode="evidence"):
        system_prompt, user_prompt = self._build_answer_prompts(
            question, history=history, answer_mode=answer_mode
        )
        yield from self.client.stream(system_prompt, user_prompt)

    # -- search rewrite / recommendation ----------------------------------

    def rewrite_search_query(self, query, context_text: str = "") -> dict:
        clean_query = compact_text(query, limit=240)
        if not clean_query:
            raise ValueError("Please enter a search query.")

        context_excerpt = sections.document_excerpt(
            context_text or "", limit=config.SEARCH_REWRITE_CONTEXT_LIMIT
        )
        system_prompt, user_prompt = sections.build_search_rewrite_prompt(
            clean_query, context_excerpt
        )
        raw_response = self.client.complete(
            system_prompt, user_prompt, model=self.client.search_rewrite_model
        )
        return sections.normalize_search_rewrite(
            raw_response, clean_query, self.client.search_rewrite_model
        )

    def recommend_papers(self, content, limit=None, search_fn=None) -> dict:
        excerpt = sections.document_excerpt(content, limit=12000)
        if not excerpt:
            raise ValueError("Current session does not contain document content.")

        resolved_limit = config.clamp_int_value(
            limit, config.RECOMMENDATION_RESULT_LIMIT, min_value=1,
            max_value=config.RECOMMENDATION_RESULT_LIMIT,
        )
        rewrite_meta = self.rewrite_search_query(
            query="Find closely related follow-up papers for this paper.",
            context_text=excerpt,
        )
        search_result = (search_fn or _default_search)(
            rewrite_meta.get("rewritten_query", ""), resolved_limit
        )
        return {
            "original_query": rewrite_meta.get("original_query", ""),
            "query": rewrite_meta.get("rewritten_query", ""),
            "topics": rewrite_meta.get("topics") or [],
            "reason": rewrite_meta.get("reason", ""),
            "rewrite_model": rewrite_meta.get("model", ""),
            "items": search_result.get("items", []),
            "errors": search_result.get("errors", []),
        }

    # -- orchestration -----------------------------------------------------

    def _get_worker_count(self, task_count: int, configured_workers: int) -> int:
        return max(1, min(task_count, configured_workers))

    def _resolve_section_future(self, future, enabled: bool = True) -> dict:
        if not enabled:
            return build_section_result("disabled")
        try:
            content_value = future.result()
            if not content_value:
                return build_section_result("empty")
            return build_section_result("success", content=content_value)
        except Exception as exc:
            return build_section_result(
                "failed",
                error=str(exc),
                retryable=is_retryable_llm_error(str(exc)),
            )

    def _plan_tasks(self, generate_mermaid, generate_evaluation, generate_research_brief):
        """Return (callable, section_name, enabled) for every section to run."""
        plan = [
            (self.generate_summary, "summary", True),
            (self.extract_quotes, "quotes", True),
            (self.generate_mindmap, "mindmap", True),
        ]
        if generate_mermaid:
            plan.append((self.generate_mermaid_mindmap, "mermaid", True))
        if generate_evaluation:
            plan.append((self.generate_evaluation, "evaluation", True))
        if generate_research_brief:
            plan.append((self.generate_research_brief, "research_brief", True))
        return plan

    @staticmethod
    def _initial_sections(generate_mermaid, generate_evaluation, generate_research_brief) -> dict:
        def status(enabled):
            return build_section_result("pending") if enabled else build_section_result("disabled")

        return {
            "summary": status(True),
            "quotes": status(True),
            "mindmap": status(True),
            "mermaid": status(generate_mermaid),
            "evaluation": status(generate_evaluation),
            "research_brief": status(generate_research_brief),
        }

    def _finalize_analysis_sections(self, content, sections_map) -> dict:
        result = {"char_count": len(content), "sections": sections_map}
        for key in sections.SECTION_KEYS:
            result[key] = (sections_map.get(key) or {}).get("content", "")

        result.update(build_analysis_metadata(sections_map))

        required = [sections_map.get(key) or {} for key in sections.REQUIRED_SECTION_KEYS]
        if required and all(section.get("status") == "failed" for section in required):
            raise RuntimeError(required[0].get("error") or "核心分析项全部失败")
        return result

    def analyze(
        self,
        file_path,
        generate_mermaid: bool = True,
        generate_evaluation: bool = True,
        generate_research_brief: bool = True,
    ) -> dict:
        """Run the full analysis and return once every section has settled."""
        content = DocumentLoader.load(file_path)
        self.document_content = content

        plan = self._plan_tasks(generate_mermaid, generate_evaluation, generate_research_brief)
        worker_count = self._get_worker_count(len(plan), self.analysis_workers)
        started_at = time.time()

        # Seed with the disabled entries so the map covers every SECTION_KEYS.
        # Building it purely from ``plan`` would leave the switched-off sections
        # missing, and the flattening step in _finalize_analysis_sections indexes
        # all of them.
        section_results = self._initial_sections(
            generate_mermaid, generate_evaluation, generate_research_brief
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(func, content): (name, enabled)
                for func, name, enabled in plan
            }
            for future in concurrent.futures.as_completed(futures):
                name, enabled = futures[future]
                section_results[name] = self._resolve_section_future(future, enabled=enabled)

        result = self._finalize_analysis_sections(content, section_results)
        result["elapsed_seconds"] = round(time.time() - started_at, 1)

        logger.info(
            "Analysis completed in %.1fs for %s (%d chars)",
            result["elapsed_seconds"],
            os.path.basename(file_path),
            len(content),
        )
        return result

    def analyze_stream(
        self,
        file_path,
        generate_mermaid: bool = True,
        generate_evaluation: bool = True,
        generate_research_brief: bool = True,
    ):
        """Yield a ``section`` event per completion, then a final ``done`` event.

        Consumers must run this generator off the event loop: each step is a
        blocking LLM call.
        """
        content = DocumentLoader.load(file_path)
        self.document_content = content

        plan = self._plan_tasks(generate_mermaid, generate_evaluation, generate_research_brief)
        section_results = self._initial_sections(
            generate_mermaid, generate_evaluation, generate_research_brief
        )
        worker_count = self._get_worker_count(len(plan), self.analysis_workers)
        started_at = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(func, content): (name, enabled)
                for func, name, enabled in plan
            }
            for future in concurrent.futures.as_completed(futures):
                name, enabled = futures[future]
                section_result = self._resolve_section_future(future, enabled=enabled)
                section_results[name] = section_result
                yield {"type": "section", "name": name, "section": section_result}

        result = self._finalize_analysis_sections(content, section_results)
        result["elapsed_seconds"] = round(time.time() - started_at, 1)
        logger.info(
            "Streaming analysis completed in %.1fs for %s (%d chars)",
            result["elapsed_seconds"],
            os.path.basename(file_path),
            len(content),
        )
        yield {"type": "done", "result": result}


def _default_search(query, limit):
    """Late import to avoid a cycle between analysis and search packages."""
    from paperwhisperer.search.papers import search_papers

    return search_papers(query, limit)
