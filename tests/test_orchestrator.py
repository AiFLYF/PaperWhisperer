"""Analysis orchestration: section planning, failure isolation and metadata.

The orchestrator's contract is that one broken section must not take down the
run. ``summary``, ``quotes`` and ``mindmap`` are required; the rest are
optional and can be switched off. These tests drive the real planning and
result-assembly code with a stubbed LLM client so the failure paths are
exercised without a network call.
"""

from __future__ import annotations

import json

import pytest

from paperwhisperer.analysis import sections
from paperwhisperer.analysis.orchestrator import (
    DocumentAnalyzer,
    build_analysis_metadata,
    build_section_result,
)


def make_analyzer(monkeypatch, complete=None, stream=None, concurrency=4):
    """Build a ``DocumentAnalyzer`` without touching a real LLM client."""
    monkeypatch.setattr(
        "paperwhisperer.analysis.orchestrator.LLMClient",
        lambda api_key: StubClient(
            api_key, complete=complete, stream=stream, concurrency=concurrency
        ),
    )
    analyzer = DocumentAnalyzer("test-key")
    analyzer.client.calls = []
    analyzer.client.stream_calls = []
    return analyzer


class StubClient:
    """Records prompts and returns canned completions."""

    def __init__(self, api_key, complete=None, stream=None, concurrency=4):
        self.api_key = api_key
        self.model = "test-model"
        self.search_rewrite_model = "test-model"
        self.request_timeout = 30
        self.max_retries = 2
        self.max_concurrency = concurrency
        self._complete = complete or (lambda *_: "content")
        self._stream = stream or (lambda *_: iter(["content"]))
        self.calls = []
        self.stream_calls = []

    def complete(self, system_prompt, user_prompt, max_retries=None, model=None):
        self.calls.append({"system": system_prompt, "user": user_prompt, "model": model})
        return self._complete(system_prompt, user_prompt)

    def stream(self, system_prompt, user_prompt, max_retries=None, model=None):
        self.stream_calls.append({"system": system_prompt, "user": user_prompt})
        return self._stream(system_prompt, user_prompt)


def rewrite_payload(query="transformer attention", topics=None):
    return json.dumps(
        {"rewritten_query": query, "topics": topics or ["attention", "nlp"], "why": "closer work"}
    )


class TestSectionResultShape:
    def test_every_field_is_always_present(self):
        result = build_section_result("success")
        assert result == {"status": "success", "content": "", "error": "", "retryable": False}

    def test_none_is_coerced_to_an_empty_string(self):
        result = build_section_result("failed", content=None, error=None, retryable=1)
        assert result["content"] == ""
        assert result["error"] == ""
        assert result["retryable"] is True

    @pytest.mark.parametrize("truthy", [1, "yes", object()])
    def test_retryable_is_strictly_boolean(self, truthy):
        assert build_section_result("failed", retryable=truthy)["retryable"] is True


class TestAnalysisMetadata:
    def test_a_fully_successful_run_reports_complete(self):
        statuses = {name: {"status": "success"} for name in sections.SECTION_KEYS}
        meta = build_analysis_metadata(statuses)
        assert meta["analysis_status"]["quality"] == "complete"
        assert meta["analysis_status"]["failed_sections"] == []
        assert set(meta["analysis_status"]["completed_sections"]) == set(sections.SECTION_LABELS.values())

    def test_a_failure_degrades_quality_to_partial(self):
        meta = build_analysis_metadata(
            {"summary": {"status": "success"}, "quotes": {"status": "failed", "error": "boom"}}
        )
        assert meta["analysis_status"]["quality"] == "partial"
        assert meta["analysis_status"]["failed_sections"] == [sections.SECTION_LABELS["quotes"]]

    def test_section_names_are_reported_as_labels(self):
        meta = build_analysis_metadata({"research_brief": {"status": "success"}})
        assert meta["analysis_status"]["completed_sections"] == ["深度简报"]

    def test_an_unknown_section_name_falls_back_to_itself(self):
        meta = build_analysis_metadata({"mystery": {"status": "success"}})
        assert meta["analysis_status"]["completed_sections"] == ["mystery"]

    @pytest.mark.parametrize("container", [None, {}])
    def test_empty_input_yields_the_base_question_set(self, container):
        meta = build_analysis_metadata(container)
        assert len(meta["suggested_questions"]) == 4
        assert len(meta["next_actions"]) == 3
        assert meta["analysis_status"]["quality"] == "complete"

    def test_disabled_is_tracked_separately_from_failed(self):
        meta = build_analysis_metadata({"evaluation": {"status": "disabled"}})
        status = meta["analysis_status"]
        assert status["disabled_sections"] == ["批判评价"]
        assert status["failed_sections"] == []
        assert status["quality"] == "complete"

    def test_a_successful_quotes_section_adds_a_citation_question(self):
        without = build_analysis_metadata({"quotes": {"status": "failed"}})
        with_quotes = build_analysis_metadata({"quotes": {"status": "success"}})
        assert len(with_quotes["suggested_questions"]) == len(without["suggested_questions"]) + 1
        assert "引用" in with_quotes["suggested_questions"][-1]

    def test_a_successful_brief_adds_a_reproduction_action(self):
        actions = build_analysis_metadata({"research_brief": {"status": "success"}})["next_actions"]
        assert any(action["label"] == "复现路线" for action in actions)

    def test_a_disabled_evaluation_offers_a_manual_action(self):
        actions = build_analysis_metadata({"evaluation": {"status": "disabled"}})["next_actions"]
        assert any(action["label"] == "手动评价" for action in actions)

    def test_a_failure_offers_a_regeneration_action_naming_the_section(self):
        actions = build_analysis_metadata({"mindmap": {"status": "failed"}})["next_actions"]
        retry = next(action for action in actions if action["label"] == "补全失败部分")
        assert sections.SECTION_LABELS["mindmap"] in retry["prompt"]

    def test_a_malformed_status_defaults_to_empty_rather_than_success(self):
        meta = build_analysis_metadata({"summary": None, "quotes": {"status": ""}})
        assert meta["analysis_status"]["section_statuses"] == {
            "summary": "empty",
            "quotes": "empty",
        }
        assert meta["analysis_status"]["quality"] == "complete"

    def test_suggestions_are_capped_at_six(self):
        everything = {name: {"status": "success"} for name in sections.SECTION_KEYS}
        assert len(build_analysis_metadata(everything)["suggested_questions"]) == 6

    def test_actions_are_capped_at_five(self):
        everything = {name: {"status": "success"} for name in sections.SECTION_KEYS}
        everything["evaluation"] = {"status": "disabled"}
        assert len(build_analysis_metadata(everything)["next_actions"]) == 5

    def test_every_action_carries_a_label_and_a_prompt(self):
        everything = {name: {"status": "success"} for name in sections.SECTION_KEYS}
        for action in build_analysis_metadata(everything)["next_actions"]:
            assert action["label"] and action["prompt"]


class TestWorkerCount:
    @pytest.mark.parametrize(
        ("task_count", "configured", "expected"),
        [(0, 5, 1), (1, 5, 1), (3, 5, 3), (6, 5, 5), (2, 0, 1), (2, -4, 1)],
    )
    def test_never_below_one_and_never_above_the_cap(self, task_count, configured, expected):
        analyzer = DocumentAnalyzer.__new__(DocumentAnalyzer)
        assert analyzer._get_worker_count(task_count, configured) == expected


class TestPlanAndInitialSections:
    @pytest.fixture
    def analyzer(self, monkeypatch):
        return make_analyzer(monkeypatch)

    def test_required_sections_always_run(self, analyzer):
        plan = analyzer._plan_tasks(False, False, False)
        assert [name for _, name, _ in plan] == list(sections.REQUIRED_SECTION_KEYS)

    def test_optional_sections_are_appended_in_a_stable_order(self, analyzer):
        plan = analyzer._plan_tasks(True, True, True)
        assert [name for _, name, _ in plan] == list(sections.SECTION_KEYS)

    def test_switches_are_independent(self, analyzer):
        plan = analyzer._plan_tasks(True, False, False)
        assert [name for _, name, _ in plan] == ["summary", "quotes", "mindmap", "mermaid"]

    def test_the_initial_snapshot_marks_pending_and_disabled(self, analyzer):
        initial = analyzer._initial_sections(True, False, True)
        assert initial["summary"]["status"] == "pending"
        assert initial["evaluation"]["status"] == "disabled"
        assert set(initial) == set(sections.SECTION_KEYS)

    def test_every_planned_callable_takes_the_document_content(self, analyzer):
        for func, _, _ in analyzer._plan_tasks(True, True, True):
            assert func.__code__.co_varnames[1] in {"content", "self"}


class TestResolveSectionFuture:
    @pytest.fixture
    def analyzer(self, monkeypatch):
        return make_analyzer(monkeypatch)

    class Done:
        @staticmethod
        def result():
            return "text"

    class Blank:
        @staticmethod
        def result():
            return ""

    class Broken:
        @staticmethod
        def result():
            raise RuntimeError("上游限流，请稍后重试")

    def test_a_disabled_section_is_never_awaited(self, analyzer):
        class Exploding:
            def result(self):
                raise AssertionError("a disabled section must not call the model")

        assert analyzer._resolve_section_future(Exploding(), enabled=False)["status"] == "disabled"

    def test_content_becomes_a_success(self, analyzer):
        assert analyzer._resolve_section_future(self.Done()) == {
            "status": "success", "content": "text", "error": "", "retryable": False,
        }

    def test_empty_output_is_distinguished_from_failure(self, analyzer):
        assert analyzer._resolve_section_future(self.Blank())["status"] == "empty"

    def test_an_exception_is_captured_in_place(self, analyzer):
        result = analyzer._resolve_section_future(self.Broken())
        assert result["status"] == "failed"
        assert result["error"] == "上游限流，请稍后重试"
        assert result["retryable"] is True

    def test_a_non_retryable_failure_is_flagged(self, analyzer):
        class Fatal:
            @staticmethod
            def result():
                raise ValueError("AI 服务未返回可识别的 HTTP 状态码。")

        assert analyzer._resolve_section_future(Fatal())["retryable"] is False


class TestFinalizeAnalysisSections:
    @pytest.fixture
    def analyzer(self, monkeypatch):
        return make_analyzer(monkeypatch)

    @staticmethod
    def sections_map(**overrides):
        base = {name: build_section_result("success", content=f"{name}-body") for name in sections.SECTION_KEYS}
        base.update(overrides)
        return base

    def test_section_content_is_also_flattened_to_the_top_level(self, analyzer):
        result = analyzer._finalize_analysis_sections("doc", self.sections_map())
        for name in sections.SECTION_KEYS:
            assert result[name] == f"{name}-body"

    def test_char_count_reflects_the_document(self, analyzer):
        result = analyzer._finalize_analysis_sections("12345", self.sections_map())
        assert result["char_count"] == 5

    def test_metadata_is_merged_into_the_result(self, analyzer):
        result = analyzer._finalize_analysis_sections("doc", self.sections_map())
        assert result["analysis_status"]["quality"] == "complete"
        assert result["suggested_questions"]

    def test_one_failed_required_section_does_not_abort_the_run(self, analyzer):
        result = analyzer._finalize_analysis_sections(
            "doc", self.sections_map(quotes=build_section_result("failed", error="boom"))
        )
        assert result["quotes"] == ""
        assert result["analysis_status"]["quality"] == "partial"

    def test_every_required_section_failing_raises_its_error(self, analyzer):
        collapsed = self.sections_map(**{
            name: build_section_result("failed", error="全部超时")
            for name in sections.REQUIRED_SECTION_KEYS
        })
        with pytest.raises(RuntimeError, match="全部超时"):
            analyzer._finalize_analysis_sections("doc", collapsed)

    def test_a_total_failure_without_a_message_still_raises(self, analyzer):
        collapsed = self.sections_map(**{
            name: build_section_result("failed") for name in sections.REQUIRED_SECTION_KEYS
        })
        with pytest.raises(RuntimeError, match="核心分析项全部失败"):
            analyzer._finalize_analysis_sections("doc", collapsed)

    def test_a_disabled_required_section_is_not_a_total_failure(self, analyzer):
        merged = self.sections_map(summary=build_section_result("disabled"))
        assert analyzer._finalize_analysis_sections("doc", merged)["summary"] == ""


class TestAnalyze:
    @pytest.fixture
    def document(self, tmp_path, monkeypatch):
        path = tmp_path / "paper.txt"
        path.write_text("paper body", encoding="utf-8")
        monkeypatch.setattr(
            "paperwhisperer.analysis.orchestrator.DocumentLoader.load",
            staticmethod(lambda file_path: "paper body"),
        )
        return path

    @staticmethod
    def fail_only(analyzer, *method_names, message="该部分生成失败"):
        """Make specific section methods raise, leaving the rest working.

        Dispatching on the analyzer's own methods keeps this independent of
        prompt wording, which overlaps between sections.
        """
        for name in method_names:
            def boom(content, _message=message):
                raise RuntimeError(_message)

            setattr(analyzer, name, boom)
        return analyzer

    def test_a_clean_run_reports_every_section(self, monkeypatch, document):
        analyzer = make_analyzer(monkeypatch)
        result = analyzer.analyze(document)
        assert result["analysis_status"]["quality"] == "complete"
        assert result["elapsed_seconds"] >= 0
        assert result["char_count"] == len("paper body")
        assert result["summary"] == "content"

    def test_the_document_is_retained_for_follow_up_questions(self, monkeypatch, document):
        analyzer = make_analyzer(monkeypatch)
        analyzer.analyze(document)
        assert analyzer.document_content == "paper body"

    @pytest.mark.parametrize(
        ("switches", "expected_disabled"),
        [
            ({"generate_mermaid": False}, ["视觉图谱"]),
            ({"generate_evaluation": False}, ["批判评价"]),
            ({"generate_research_brief": False}, ["深度简报"]),
            (
                {"generate_mermaid": False, "generate_evaluation": False, "generate_research_brief": False},
                ["视觉图谱", "批判评价", "深度简报"],
            ),
        ],
    )
    def test_switched_off_sections_are_reported_as_disabled(
        self, monkeypatch, document, switches, expected_disabled
    ):
        analyzer = make_analyzer(monkeypatch)
        result = analyzer.analyze(document, **switches)
        assert result["analysis_status"]["disabled_sections"] == expected_disabled
        assert result["analysis_status"]["failed_sections"] == []
        # Every section key must still be present, just empty.
        assert set(result["sections"]) == set(sections.SECTION_KEYS)
        for name in expected_disabled:
            key = next(k for k, v in sections.SECTION_LABELS.items() if v == name)
            assert result[key] == ""

    def test_a_switched_off_section_is_never_called(self, monkeypatch, document):
        analyzer = make_analyzer(monkeypatch)
        analyzer.generate_evaluation = lambda content: (_ for _ in ()).throw(
            AssertionError("a disabled section must not call the model")
        )
        result = analyzer.analyze(document, generate_evaluation=False)
        assert result["analysis_status"]["disabled_sections"] == ["批判评价"]

    def test_one_broken_section_does_not_stop_the_others(self, monkeypatch, document):
        analyzer = self.fail_only(make_analyzer(monkeypatch), "extract_quotes")
        result = analyzer.analyze(document)
        assert result["analysis_status"]["failed_sections"] == ["引用片段"]
        assert result["summary"] == "content"

    def test_total_failure_surfaces_the_underlying_message(self, monkeypatch, document):
        analyzer = self.fail_only(
            make_analyzer(monkeypatch),
            "generate_summary", "extract_quotes", "generate_mindmap",
            message="模型不可用",
        )
        with pytest.raises(RuntimeError, match="模型不可用"):
            analyzer.analyze(document)

    def test_mermaid_output_is_normalized_before_it_is_reported(self, monkeypatch, document):
        analyzer = make_analyzer(monkeypatch)
        analyzer.generate_mermaid_mindmap = lambda content: DocumentAnalyzer._normalize_mermaid_source(
            "好的，这是图谱：\n```mermaid\ngraph TD\nA-->B\n```"
        )
        result = analyzer.analyze(document, generate_evaluation=False, generate_research_brief=False)
        assert result["mermaid"] == "graph TD\nA-->B"

    def test_concurrency_is_capped_by_the_client_limit(self, monkeypatch, document):
        analyzer = make_analyzer(monkeypatch, concurrency=2)
        analyzer.analyze(document)
        assert analyzer.analysis_workers == 2


class TestAnalyzeStream:
    @pytest.fixture
    def document(self, tmp_path, monkeypatch):
        path = tmp_path / "paper.txt"
        path.write_text("paper body", encoding="utf-8")
        monkeypatch.setattr(
            "paperwhisperer.analysis.orchestrator.DocumentLoader.load",
            staticmethod(lambda file_path: "paper body"),
        )
        return path

    def test_the_stream_ends_with_exactly_one_done_event(self, monkeypatch, document):
        analyzer = make_analyzer(monkeypatch)
        events = list(analyzer.analyze_stream(document))
        assert events[-1]["type"] == "done"
        assert sum(1 for event in events if event["type"] == "done") == 1

    def test_one_section_event_per_settled_section(self, monkeypatch, document):
        analyzer = make_analyzer(monkeypatch)
        events = list(analyzer.analyze_stream(document))
        names = [event["name"] for event in events if event["type"] == "section"]
        assert sorted(names) == sorted(sections.SECTION_KEYS)

    def test_switched_off_sections_never_produce_an_event(self, monkeypatch, document):
        analyzer = make_analyzer(monkeypatch)
        events = list(
            analyzer.analyze_stream(document, generate_mermaid=False, generate_evaluation=False)
        )
        names = {event["name"] for event in events if event["type"] == "section"}
        assert "mermaid" not in names
        assert "evaluation" not in names

    def test_the_final_payload_matches_the_batch_shape(self, monkeypatch, document):
        analyzer = make_analyzer(monkeypatch)
        streamed = list(analyzer.analyze_stream(document))[-1]["result"]
        batch = make_analyzer(monkeypatch).analyze(document)
        assert set(streamed) == set(batch)
        assert streamed["sections"].keys() == batch["sections"].keys()

    def test_a_failed_section_is_streamed_as_a_failure_not_raised(self, monkeypatch, document):
        analyzer = make_analyzer(monkeypatch)
        analyzer.extract_quotes = lambda content: (_ for _ in ()).throw(
            RuntimeError("引用段生成失败")
        )
        events = list(analyzer.analyze_stream(document))
        failed = [e for e in events if e["type"] == "section" and e["section"]["status"] == "failed"]
        assert [event["name"] for event in failed] == ["quotes"]
        assert failed[0]["section"]["error"] == "引用段生成失败"
        assert events[-1]["type"] == "done"

    def test_a_section_event_carries_the_whole_result_object(self, monkeypatch, document):
        analyzer = make_analyzer(monkeypatch)
        for event in analyzer.analyze_stream(document):
            if event["type"] == "section":
                assert set(event["section"]) == {"status", "content", "error", "retryable"}


class TestMermaidNormalization:
    normalize = staticmethod(DocumentAnalyzer._normalize_mermaid_source)

    @pytest.mark.parametrize(
        "prefix",
        ["graph TD", "mindmap", "flowchart LR", "pie", "sequenceDiagram", "stateDiagram", "classDiagram"],
    )
    def test_a_recognised_diagram_is_kept_verbatim(self, prefix):
        payload = f"{prefix}\nA-->B"
        assert self.normalize(payload) == payload

    def test_leading_prose_is_dropped(self):
        assert self.normalize("下面是你的图谱：\ngraph TD\nA-->B") == "graph TD\nA-->B"

    def test_a_fenced_block_without_a_keyword_is_unwrapped(self):
        assert self.normalize("```\nA-->B\n```") == "graph TD\nA-->B"

    def test_a_bare_payload_gets_a_safe_default_container(self):
        assert self.normalize("A-->B") == "graph TD\nA-->B"

    def test_prose_only_output_is_wrapped_rather_than_rendered_raw(self):
        result = self.normalize("我无法生成图谱。")
        assert result.startswith("graph TD")

    def test_a_blank_payload_is_wrapped_in_a_valid_container(self):
        result = self.normalize("   \n  ")
        assert result == "graph TD"

    @pytest.mark.parametrize(
        "fenced",
        [
            "好的，这是图谱：\n```mermaid\ngraph TD\nA-->B\n```",
            "```\ngraph LR\nA-->B\n```",
            "graph TD\nA-->B\n```",
            "flowchart TD\nA-->B```",
            "```mermaid\nmindmap\nroot\n~~~",
        ],
    )
    def test_a_closing_fence_never_reaches_the_renderer(self, fenced):
        result = self.normalize(fenced)
        assert "```" not in result
        assert "~~~" not in result
        assert result.startswith(("graph ", "flowchart ", "mindmap"))

    def test_an_unfenced_diagram_is_untouched(self):
        assert self.normalize("graph TD\nA-->B\nC-->D") == "graph TD\nA-->B\nC-->D"

    def test_normalization_is_idempotent(self):
        once = self.normalize("说明：\ngraph TD\nA-->B")
        assert self.normalize(once) == once

    def test_an_empty_model_response_becomes_none(self, monkeypatch):
        analyzer = make_analyzer(monkeypatch, complete=lambda *_: "")
        assert analyzer.generate_mermaid_mindmap("doc") is None


class TestHistoryBlock:
    build = staticmethod(DocumentAnalyzer._build_history_block)

    def test_no_history_produces_an_empty_block(self):
        assert self.build([]) == ""
        assert self.build(None) == ""

    def test_turns_are_rendered_newest_last(self):
        block = self.build([
            {"question": "第一问", "answer": "第一答"},
            {"question": "第二问", "answer": "第二答"},
        ])
        assert block.index("第一问") < block.index("第二问")

    def test_the_most_recent_turn_wins_the_budget(self):
        history = [
            {"question": f"q{index}", "answer": "a" * 400} for index in range(20)
        ]
        block = self.build(history)
        assert "q19" in block
        assert "q0" not in block

    def test_an_oversized_turn_is_truncated_with_a_marker(self, monkeypatch):
        monkeypatch.setattr("paperwhisperer.core.config.QA_TURN_BUDGET", 50)
        block = self.build([{"question": "q" * 400, "answer": "a" * 400}])
        assert block.endswith("...[truncated]")
        assert len(block) < 200

    def test_empty_turns_are_dropped(self):
        assert self.build([{"question": "", "answer": ""}, {"question": "有效", "answer": ""}]) == "Q: 有效\nA: "

    def test_a_missing_turn_is_skipped_rather_than_crashing(self):
        assert self.build([{"question": "只有问题", "answer": "有答案"}]) == "Q: 只有问题\nA: 有答案"


class TestAnswerPrompts:
    @pytest.fixture
    def analyzer(self, monkeypatch):
        instance = make_analyzer(monkeypatch)
        instance.document_content = "document body"
        return instance

    def test_answering_without_a_document_is_refused(self, monkeypatch):
        instance = make_analyzer(monkeypatch)
        instance.document_content = ""
        with pytest.raises(ValueError, match="没有文档内容"):
            instance._build_answer_prompts("问题")

    def test_the_answer_mode_is_normalised_before_it_reaches_the_prompt(self, analyzer, monkeypatch):
        captured = {}

        def fake_build(**kwargs):
            captured.update(kwargs)
            return ("system", "user")

        monkeypatch.setattr(sections, "build_answer_prompt", fake_build)
        analyzer._build_answer_prompts("问题", answer_mode="nonsense-mode")
        assert captured["answer_mode"] in {"evidence", "plain", "deep"}

    def test_the_document_window_is_truncated_to_the_qa_budget(self, analyzer, monkeypatch):
        from paperwhisperer.core import config

        analyzer.document_content = "x" * (config.DOCUMENT_WINDOW_BUDGET["qa"] + 500)
        captured = {}
        monkeypatch.setattr(
            sections, "build_answer_prompt",
            lambda **kwargs: captured.update(kwargs) or ("system", "user"),
        )
        analyzer._build_answer_prompts("问题")
        assert len(captured["document_window"]) == config.DOCUMENT_WINDOW_BUDGET["qa"]

    def test_answering_reaches_the_client(self, analyzer):
        assert analyzer.answer_question("问题") == "content"
        assert analyzer.client.calls

    def test_streaming_answer_reaches_the_client(self, analyzer):
        assert list(analyzer.stream_answer_question("问题")) == ["content"]
        assert analyzer.client.stream_calls


class TestSearchRewriteAndRecommend:
    @pytest.fixture
    def analyzer(self, monkeypatch):
        return make_analyzer(monkeypatch, complete=lambda *_: rewrite_payload())

    def test_a_blank_query_is_refused(self, analyzer):
        with pytest.raises(ValueError, match="search query"):
            analyzer.rewrite_search_query("   ")

    def test_the_original_query_is_preserved(self, analyzer):
        meta = analyzer.rewrite_search_query("原始问题")
        assert meta["original_query"] == "原始问题"
        assert meta["rewritten_query"] == "transformer attention"

    def test_the_rewrite_uses_the_cheap_model(self, analyzer):
        analyzer.rewrite_search_query("原始问题")
        assert analyzer.client.calls[-1]["model"] == "test-model"

    def test_an_unparsable_rewrite_is_refused(self, monkeypatch):
        instance = make_analyzer(monkeypatch, complete=lambda *_: "not json at all")
        with pytest.raises(ValueError, match="rewriting failed"):
            instance.rewrite_search_query("原始问题")

    def test_a_json_array_instead_of_an_object_is_refused(self, monkeypatch):
        instance = make_analyzer(monkeypatch, complete=lambda *_: "[1, 2, 3]")
        with pytest.raises(ValueError, match="unexpected payload"):
            instance.rewrite_search_query("原始问题")

    def test_an_empty_rewritten_query_is_refused(self, monkeypatch):
        instance = make_analyzer(
            monkeypatch, complete=lambda *_: json.dumps({"rewritten_query": "  "})
        )
        with pytest.raises(ValueError, match="empty rewritten query"):
            instance.rewrite_search_query("原始问题")

    def test_recommendation_requires_document_content(self, analyzer):
        with pytest.raises(ValueError, match="document content"):
            analyzer.recommend_papers("")

    def test_the_recommendation_search_receives_the_rewritten_query(self, analyzer):
        seen = {}

        def fake_search(query, limit):
            seen.update(query=query, limit=limit)
            return {"items": [{"title": "t"}], "errors": ["arxiv down"]}

        result = analyzer.recommend_papers("paper body", limit=3, search_fn=fake_search)
        assert seen == {"query": "transformer attention", "limit": 3}
        assert result["items"] == [{"title": "t"}]
        assert result["errors"] == ["arxiv down"]

    def test_the_limit_is_clamped_into_range(self, analyzer):
        seen = {}
        analyzer.recommend_papers(
            "paper body", limit=10_000, search_fn=lambda q, limit: seen.update(limit=limit) or {"items": []}
        )
        assert seen["limit"] <= 20

    def test_a_missing_limit_falls_back_to_the_configured_default(self, analyzer):
        seen = {}
        analyzer.recommend_papers(
            "paper body", limit=None, search_fn=lambda q, limit: seen.update(limit=limit) or {"items": []}
        )
        assert seen["limit"] >= 1

    def test_topics_default_to_an_empty_list(self, monkeypatch):
        instance = make_analyzer(
            monkeypatch, complete=lambda *_: json.dumps({"rewritten_query": "q"})
        )
        assert instance.rewrite_search_query("原始问题")["topics"] == []
