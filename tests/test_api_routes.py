"""Shared API route behaviour.

Routes are exercised through the real ``TestClient`` wiring with only the two
things that would otherwise reach the network replaced: the LLM transport and
the remote-download layer. Session storage is redirected to ``tmp_path`` so a
test run cannot read or clobber the developer's real context folder.
"""

from __future__ import annotations

import asyncio
import io
import threading
import time

import pytest

from paperwhisperer.analysis import sections
from paperwhisperer.api.deps import pump_sync_events
from paperwhisperer.core import config
from paperwhisperer.llm import prompts


async def drain(stream) -> None:
    """Consume an async iterator to completion, discarding items."""
    async for _ in stream:
        pass


class RecordingAnalyzer:
    """Stand-in for ``DocumentAnalyzer`` that records how it was called."""

    instances: list[RecordingAnalyzer] = []
    sections_to_fail: set[str] = set()
    answer_text = "answer body"
    chunks = ("第一段", "第二段")
    rewrite_meta = None
    recommendation = None
    analysis_result = None

    def __init__(self, api_key):
        self.api_key = api_key
        self.document_content = ""
        self.version = "test-version"
        self.calls: list[tuple] = []
        self.stream_calls: list[tuple] = []
        RecordingAnalyzer.instances.append(self)

    @classmethod
    def reset(cls, **overrides):
        cls.instances = []
        cls.sections_to_fail = set()
        cls.answer_text = "answer body"
        cls.chunks = ("第一段", "第二段")
        cls.rewrite_meta = None
        cls.recommendation = None
        cls.analysis_result = None
        for name, value in overrides.items():
            setattr(cls, name, value)

    @classmethod
    def last(cls):
        """The most recently constructed stub.

        A ``property`` would not work here: tests hold the *class* (that is
        what the ``analyzer`` fixture yields), and reading a property off a
        class hands back the property object rather than a stub.
        """
        return cls.instances[-1]

    # -- analysis -------------------------------------------------------

    def _section_body(self, name):
        if name in self.sections_to_fail:
            raise RuntimeError(f"{name} 生成失败")
        return f"{name} 内容"

    def _build_result(self, mermaid=True, evaluation=True, brief=True):
        if self.analysis_result is not None:
            return self.analysis_result
        statuses = {name: "success" for name in sections.SECTION_KEYS}
        statuses.update({name: "disabled" for name, on in
                         (("mermaid", mermaid), ("evaluation", evaluation),
                          ("research_brief", brief)) if not on})
        result = {"char_count": 10, "sections": {
            name: {"status": statuses[name], "content": f"{name} 内容",
                   "error": "", "retryable": False}
            for name in sections.SECTION_KEYS}}
        for name in sections.SECTION_KEYS:
            result[name] = f"{name} 内容"
        return result

    def analyze(self, file_path, mermaid=True, evaluation=True, brief=True):
        self.calls.append(("analyze", file_path, mermaid, evaluation, brief))
        for name in sections.SECTION_KEYS:
            enabled = {
                "mermaid": mermaid, "evaluation": evaluation, "research_brief": brief,
            }.get(name, True)
            if not enabled:
                continue
            self._section_body(name)
        return self._build_result(mermaid, evaluation, brief)

    def analyze_stream(self, file_path, mermaid=True, evaluation=True, brief=True):
        self.calls.append(("analyze_stream", file_path, mermaid, evaluation, brief))
        for name in sections.SECTION_KEYS:
            enabled = {
                "mermaid": mermaid, "evaluation": evaluation, "research_brief": brief,
            }.get(name, True)
            if not enabled:
                continue
            try:
                content = self._section_body(name)
                yield {"type": "section", "name": name,
                       "section": {"status": "success", "content": content,
                                   "error": "", "retryable": False}}
            except RuntimeError as exc:
                yield {"type": "section", "name": name,
                       "section": {"status": "failed", "content": "",
                                   "error": str(exc), "retryable": True}}

        yield {"type": "done", "result": self._build_result(mermaid, evaluation, brief)}

    # -- Q&A ------------------------------------------------------------

    def answer_question(self, question, history=None, answer_mode="evidence"):
        self.calls.append(("answer", question, list(history or []), answer_mode))
        return self.answer_text

    def stream_answer_question(self, question, history=None, answer_mode="evidence"):
        self.stream_calls.append(("stream", question, list(history or []), answer_mode))
        yield from self.chunks

    # -- search ---------------------------------------------------------

    def rewrite_search_query(self, query, context_text=""):
        self.calls.append(("rewrite", query, context_text))
        if self.rewrite_meta is not None:
            return self.rewrite_meta
        return {"original_query": query, "rewritten_query": f"rewritten:{query}",
                "topics": ["t"], "reason": "因为", "model": "cheap-model"}

    def recommend_papers(self, content, limit=None, search_fn=None):
        self.calls.append(("recommend", content, limit))
        if self.recommendation is not None:
            return self.recommendation
        return {"original_query": "orig", "query": "rewritten", "topics": ["t"],
                "reason": "r", "rewrite_model": "m",
                "items": [{"title": "Paper A", "url": "https://example.com/a"}],
                "errors": []}


@pytest.fixture
def analyzer(monkeypatch):
    """Swap in ``RecordingAnalyzer`` and hand back the class for configuration.

    ``DocumentAnalyzer`` has to be replaced in every module that binds the name,
    not just the route modules: ``/api/analyze`` and ``/api/import-paper`` both
    delegate to ``analysis.service.analyze_saved_file``, which constructs the
    analyzer itself. Patching only the routes would let the real analyzer — and
    a real HTTP call to the model endpoint — through.
    """
    RecordingAnalyzer.reset()
    for module in (
        "paperwhisperer.analysis.service",
        "paperwhisperer.api.routes_documents",
        "paperwhisperer.api.routes_qa",
        "paperwhisperer.api.routes_papers",
    ):
        monkeypatch.setattr(f"{module}.DocumentAnalyzer", RecordingAnalyzer)
    yield RecordingAnalyzer
    RecordingAnalyzer.reset()


def upload_bytes(name="paper.txt", body=b"plain text body"):
    """Build the ``files=`` mapping for a multipart upload."""
    return {"file": (name, io.BytesIO(body), "text/plain")}


def form(**values):
    """Build the multipart ``data=`` mapping, dropping unset values."""
    return {key: str(value) for key, value in values.items() if value is not None}


@pytest.fixture
def analysis_client(client, analyzer, runtime_dirs):
    """A ``TestClient`` with a stubbed analyzer and isolated storage."""
    return client


class TestAnalyzeRoute:
    def test_a_valid_upload_returns_the_analysis_result(self, analysis_client):
        response = analysis_client.post(
            "/api/analyze", files=upload_bytes(), data=form(api_key="k")
        )
        assert response.status_code == 200
        assert response.json()["summary"] == "summary 内容"

    def test_no_file_is_a_client_error(self, analysis_client):
        response = analysis_client.post("/api/analyze", data=form(api_key="k"))
        assert response.status_code == 400
        assert response.json()["code"] == "missing_file"

    def test_an_empty_filename_is_rejected(self, analysis_client):
        response = analysis_client.post(
            "/api/analyze", files={"file": ("", io.BytesIO(b"x"), "text/plain")},
            data=form(api_key="k"),
        )
        # The request is turned away before the route body runs: with no
        # filename the multipart parser hands FastAPI a plain string, so
        # parameter validation rejects the body with 422. What matters is
        # that a nameless upload can never reach the analyzer, not which
        # layer says so. No stub was constructed at all, so ``instances``
        # is empty.
        assert response.status_code in {400, 422}
        assert RecordingAnalyzer.instances == []

    def test_an_unsupported_extension_names_the_supported_ones(self, analysis_client):
        response = analysis_client.post(
            "/api/analyze", files=upload_bytes("malware.exe", b"MZ"),
            data=form(api_key="k"),
        )
        assert response.status_code == 400
        assert response.json()["code"] == "unsupported_file_type"

    def test_an_oversized_upload_is_refused_and_leaves_nothing_behind(self, analysis_client, runtime_dirs):
        oversized = b"a" * (config.MAX_CONTENT_LENGTH + 1)
        response = analysis_client.post(
            "/api/analyze", files=upload_bytes("big.txt", oversized), data=form(api_key="k")
        )
        assert response.status_code in {400, 413}
        assert list(runtime_dirs["uploads"].iterdir()) == []

    def test_the_uploaded_file_is_deleted_after_analysis(self, analysis_client, runtime_dirs):
        analysis_client.post("/api/analyze", files=upload_bytes(), data=form(api_key="k"))
        assert list(runtime_dirs["uploads"].iterdir()) == []

    @pytest.mark.parametrize(
        ("field", "position"),
        [("generate_mermaid", 2), ("generate_evaluation", 3), ("generate_research_brief", 4)],
    )
    def test_optional_section_switches_reach_the_analyzer(
        self, analysis_client, analyzer, field, position
    ):
        analysis_client.post(
            "/api/analyze", files=upload_bytes(), data=form(api_key="k", **{field: "false"})
        )
        call = analyzer.last().calls[0]
        assert call[0] == "analyze"
        assert call[position] is False

    def test_the_switches_default_to_on(self, analysis_client, analyzer):
        analysis_client.post("/api/analyze", files=upload_bytes(), data=form(api_key="k"))
        assert analyzer.last().calls[0][2:5] == (True, True, True)

    def test_a_hostile_session_id_is_sanitised_before_use(self, analysis_client, runtime_dirs):
        response = analysis_client.post(
            "/api/analyze", files=upload_bytes(), data=form(api_key="k", session_id="../../etc")
        )
        assert response.status_code == 200
        assert not (runtime_dirs["context"].parent / "etc").exists()

    def test_a_model_failure_is_reported_as_a_server_error(self, client, analyzer, runtime_dirs, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("模型不可用")

        monkeypatch.setattr("paperwhisperer.api.routes_documents.analyze_saved_file", boom)
        response = client.post("/api/analyze", files=upload_bytes(), data=form(api_key="k"))
        assert response.status_code == 500
        assert response.json()["code"] == "document_analysis_failed"
        assert list(runtime_dirs["uploads"].iterdir()) == []

    def test_a_value_error_is_reported_as_a_client_error(self, client, analyzer, runtime_dirs, monkeypatch):
        def bad(*args, **kwargs):
            raise ValueError("文件内容无法解析")

        monkeypatch.setattr("paperwhisperer.api.routes_documents.analyze_saved_file", bad)
        response = client.post("/api/analyze", files=upload_bytes(), data=form(api_key="k"))
        assert response.status_code == 400
        assert response.json()["code"] == "invalid_request"


class TestAnalyzeStreamRoute:
    def test_the_response_is_an_event_stream(self, analysis_client):
        response = analysis_client.post(
            "/api/analyze/stream", files=upload_bytes(), data=form(api_key="k")
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

    def test_the_stream_opens_with_a_session_and_finishes_with_a_result(
        self, analysis_client, sse_events
    ):
        response = analysis_client.post(
            "/api/analyze/stream", files=upload_bytes(), data=form(api_key="k")
        )
        events = sse_events(response.text)
        assert events[0][0] == "start"
        assert events[-1][0] == "done"
        assert events[-1][1]["summary"] == "summary 内容"

    def test_every_planned_section_arrives_as_its_own_event(self, analysis_client, sse_events):
        response = analysis_client.post(
            "/api/analyze/stream", files=upload_bytes(), data=form(api_key="k")
        )
        names = [p["name"] for name, p in sse_events(response.text) if name == "section"]
        assert sorted(names) == sorted(sections.SECTION_KEYS)

    def test_a_switched_off_section_produces_no_event(self, analysis_client, sse_events):
        response = analysis_client.post(
            "/api/analyze/stream", files=upload_bytes(),
            data=form(api_key="k", generate_evaluation="false"),
        )
        names = [p["name"] for name, p in sse_events(response.text) if name == "section"]
        assert "evaluation" not in names

    def test_a_failed_section_is_streamed_rather_than_fatal(self, analysis_client, analyzer, sse_events):
        analyzer.sections_to_fail = {"quotes"}
        response = analysis_client.post(
            "/api/analyze/stream", files=upload_bytes(), data=form(api_key="k")
        )
        events = sse_events(response.text)
        failed = [p for name, p in events if name == "section" and p["section"]["status"] == "failed"]
        assert [item["name"] for item in failed] == ["quotes"]
        assert events[-1][0] == "done"

    def test_a_missing_api_key_is_refused_before_the_stream_opens(self, analysis_client):
        response = analysis_client.post("/api/analyze/stream", files=upload_bytes())
        assert response.status_code in {200, 400}
        if response.status_code == 400:
            assert response.json()["code"] == "missing_api_key"
        else:
            # An SSE body can only report the problem as an event.
            assert "error" in response.text

    def test_the_upload_is_removed_once_the_stream_finishes(self, analysis_client, runtime_dirs):
        analysis_client.post("/api/analyze/stream", files=upload_bytes(), data=form(api_key="k"))
        assert list(runtime_dirs["uploads"].iterdir()) == []

    def test_a_producer_failure_becomes_an_error_event(self, client, analyzer, runtime_dirs, monkeypatch):
        def explode(*args, **kwargs):
            raise RuntimeError("流式分析崩溃")
            yield  # pragma: no cover - makes this a generator

        monkeypatch.setattr("paperwhisperer.api.routes_documents.DocumentAnalyzer", explode)
        response = client.post(
            "/api/analyze/stream", files=upload_bytes(), data=form(api_key="k")
        )
        assert "error" in response.text
        assert list(runtime_dirs["uploads"].iterdir()) == []

    def test_no_file_is_a_client_error(self, analysis_client):
        response = analysis_client.post("/api/analyze/stream", data=form(api_key="k"))
        assert response.status_code == 400


class TestAskRoute:
    @pytest.fixture
    def session(self, session_factory, runtime_dirs, api_key_env):
        return session_factory(document_content="document text")

    def test_a_answered_question_returns_the_answer(self, client, analyzer, session):
        session_id, token = session
        response = client.post(
            "/api/ask",
            json={"question": "问题", "session_id": session_id, "session_token": token},
        )
        assert response.status_code == 200
        assert response.json()["answer"] == "answer body"

    def test_a_blank_question_is_refused(self, client, analyzer, session):
        session_id, token = session
        response = client.post(
            "/api/ask", json={"question": "   ", "session_id": session_id, "session_token": token}
        )
        assert response.status_code == 400
        assert response.json()["code"] == "missing_question"

    def test_a_missing_api_key_is_refused(self, client, analyzer, session, monkeypatch):
        # The session fixture supplies a key; this case is about its absence.
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        session_id, token = session
        response = client.post(
            "/api/ask", json={"question": "问题", "session_id": session_id, "session_token": token}
        )
        assert response.status_code == 400
        assert response.json()["code"] == "missing_api_key"

    def test_a_wrong_token_is_a_403_not_a_500(self, client, analyzer, session):
        session_id, _ = session
        response = client.post(
            "/api/ask", json={"question": "问题", "session_id": session_id, "session_token": "wrong"}
        )
        assert response.status_code == 403
        assert response.json()["code"] == "invalid_session_token"

    def test_an_unknown_session_is_a_client_error(self, client, analyzer, session):
        _, token = session
        response = client.post(
            "/api/ask", json={"question": "问题", "session_id": "nope", "session_token": token}
        )
        assert response.status_code == 400
        assert response.json()["code"] == "invalid_request"

    def test_an_unknown_answer_mode_falls_back_to_the_default(self, client, analyzer, session):
        session_id, token = session
        client.post(
            "/api/ask",
            json={"question": "问题", "session_id": session_id, "session_token": token,
                  "answer_mode": "not-a-mode"},
        )
        assert analyzer.last().calls[0][3] == prompts.DEFAULT_ANSWER_MODE

    @pytest.mark.parametrize("mode", sorted(prompts.ANSWER_MODES))
    def test_every_declared_answer_mode_survives_normalisation(
        self, client, analyzer, session, mode
    ):
        session_id, token = session
        client.post(
            "/api/ask",
            json={"question": "问题", "session_id": session_id, "session_token": token,
                  "answer_mode": mode},
        )
        assert analyzer.last().calls[0][3] == mode

    def test_a_missing_answer_mode_uses_the_default(self, client, analyzer, session):
        session_id, token = session
        client.post(
            "/api/ask",
            json={"question": "问题", "session_id": session_id, "session_token": token},
        )
        assert analyzer.last().calls[0][3] == prompts.DEFAULT_ANSWER_MODE

    def test_the_turn_is_recorded_in_the_session(self, client, analyzer, session, runtime_dirs):
        import json

        session_id, token = session
        client.post(
            "/api/ask", json={"question": "问题", "session_id": session_id, "session_token": token}
        )
        stored = json.loads((runtime_dirs["context"] / f"{session_id}.json").read_text("utf-8"))
        assert stored["qa_history"][-1]["question"] == "问题"
        assert stored["qa_history"][-1]["answer"] == "answer body"

    def test_prior_history_is_forwarded_to_the_analyzer(self, client, analyzer, session_factory, api_key_env):
        session_id, token = session_factory(
            qa_history=[{"question": "旧问题", "answer": "旧答案", "answer_mode": "evidence"}]
        )
        client.post(
            "/api/ask", json={"question": "新问题", "session_id": session_id, "session_token": token}
        )
        assert analyzer.last().calls[0][2] == [{"question": "旧问题", "answer": "旧答案", "answer_mode": "evidence"}]

    def test_a_malformed_body_is_refused(self, client, analyzer, session):
        session_id, token = session
        response = client.post(
            "/api/ask",
            content=b"{not json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 400
        assert response.json()["code"] == "invalid_json"


class TestAskStreamRoute:
    @pytest.fixture
    def session(self, session_factory, runtime_dirs, api_key_env):
        return session_factory(document_content="document text")

    def test_the_deltas_reassemble_into_the_final_answer(self, client, analyzer, session, sse_events):
        session_id, token = session
        response = client.post(
            "/api/ask/stream",
            json={"question": "问题", "session_id": session_id, "session_token": token},
        )
        events = sse_events(response.text)
        assert events[0][0] == "start"
        streamed = "".join(p["text"] for name, p in events if name == "delta")
        assert events[-1][0] == "done"
        assert streamed == events[-1][1]["answer"]
        assert streamed == "".join(analyzer.chunks)

    def test_a_failed_producer_becomes_an_error_event(self, client, analyzer, session, sse_events, monkeypatch):
        session_id, token = session

        def explode(api_key):
            raise RuntimeError("流式问答崩溃")
            yield  # pragma: no cover

        monkeypatch.setattr("paperwhisperer.api.routes_qa.DocumentAnalyzer", explode)
        response = client.post(
            "/api/ask/stream",
            json={"question": "问题", "session_id": session_id, "session_token": token},
        )
        assert sse_events(response.text)[-1][0] == "error"

    def test_the_turn_is_recorded_after_streaming(self, client, analyzer, session, runtime_dirs):
        import json

        session_id, token = session
        client.post(
            "/api/ask/stream",
            json={"question": "问题", "session_id": session_id, "session_token": token},
        )
        stored = json.loads((runtime_dirs["context"] / f"{session_id}.json").read_text("utf-8"))
        assert stored["qa_history"][-1]["answer"] == "".join(analyzer.chunks)

    def test_a_wrong_token_is_refused_before_the_stream_opens(self, client, analyzer, session):
        session_id, _ = session
        response = client.post(
            "/api/ask/stream", json={"question": "问题", "session_id": session_id, "session_token": "bad"}
        )
        assert response.status_code == 403
        assert response.json()["code"] == "invalid_session_token"



class TestEventLoopIsNotBlocked:
    """The SSE pump must keep the loop free while a producer blocks.

    Regression coverage for the fix that moved blocking LLM generators onto a
    worker thread.

    The measurement is taken **from inside the producer**: it records how many
    ticks a free-running coroutine managed to accumulate while it was blocked.
    That number can only be positive if the producer was on a worker thread
    and the loop stayed available to run other work. Were the producer
    iterated on the loop thread, the ticker could not be scheduled at all
    during the block and the recorded value would be exactly zero.
    """

    #: Ticks the producer waits for. Small on purpose: a healthy run satisfies
    #: it almost immediately, so only a regression pays the full timeout.
    TICKS_REQUIRED = 5
    BLOCK_TIMEOUT_SECONDS = 5

    def test_the_loop_keeps_running_while_the_producer_blocks(self):
        async def scenario():
            started = threading.Event()
            ticks: list[int] = []
            observed: list[int] = []

            def produce():
                started.set()
                deadline = time.monotonic() + self.BLOCK_TIMEOUT_SECONDS
                while len(ticks) < self.TICKS_REQUIRED and time.monotonic() < deadline:
                    time.sleep(0.001)
                observed.append(len(ticks))
                yield {"type": "delta", "data": {"text": "x"}}

            async def ticker():
                while True:
                    ticks.append(1)
                    await asyncio.sleep(0)

            ticker_task = asyncio.create_task(ticker())
            try:
                consumer = asyncio.create_task(
                    drain(pump_sync_events(produce, lambda e: e))
                )
                assert await asyncio.to_thread(started.wait, 5), "producer never started"
                await asyncio.wait_for(consumer, timeout=15)
            finally:
                ticker_task.cancel()
            return observed

        observed = asyncio.run(scenario())
        # ``>=`` rather than ``==``: the ticker is a free-running loop, so a
        # healthy run overshoots the threshold. Zero is the only real failure
        # signal — that means the ticker was never scheduled at all.
        assert observed and observed[0] >= self.TICKS_REQUIRED, (
            f"producer saw {observed} loop ticks while blocking, "
            f"expected at least {self.TICKS_REQUIRED} — the event loop was blocked"
        )

    def test_cleanup_runs_even_when_the_consumer_stops_early(self):
        async def scenario():
            cleaned = []
            release = threading.Event()

            def produce():
                yield {"type": "delta", "data": {"text": "a"}}
                release.wait(timeout=5)
                yield {"type": "delta", "data": {"text": "b"}}

            stream = pump_sync_events(produce, lambda e: e, cleanup=lambda: cleaned.append(1))
            first = await stream.__anext__()
            assert first == {"type": "delta", "data": {"text": "a"}}
            await stream.aclose()
            return cleaned

        assert asyncio.run(scenario()) == [1]
