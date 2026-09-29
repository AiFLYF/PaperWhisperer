"""LLM transport: retry policy, stream framing and response validation.

The behaviour that matters most here is the asymmetry between the two call
shapes. A non-streaming completion can be replayed safely. A stream cannot:
once deltas have been yielded the client has already rendered them, so a retry
would duplicate visible text. ``STREAM_RETRY_PROBE_CHUNKS`` exists to widen the
safe-retry window by one chunk, and these tests pin that boundary down.
"""

from __future__ import annotations

import json

import pytest

from paperwhisperer.core.errors import (
    describe_llm_status_code,
    html_response_error,
    is_retryable_llm_error,
    looks_like_html_response,
)
from paperwhisperer.llm.client import STREAM_RETRY_PROBE_CHUNKS, LLMClient

from .helpers import FakeOpenAI, FakeStreamingResponse


def sse_line(text: str) -> bytes:
    """One ``data:`` line carrying a single content delta."""
    return b"data: " + json.dumps({"choices": [{"delta": {"content": text}}]}).encode("utf-8")


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeCompletion:
    def __init__(self, content):
        self.choices = [FakeChoice(content)] if content is not None else []


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeRawResponse:
    """Mimics ``with_raw_response.create``: exposes status, then ``parse()``."""

    def __init__(self, content, status_code=200):
        self._completion = FakeCompletion(content)
        self.status_code = status_code

    def parse(self):
        return self._completion


def make_client(raw_behaviours=None, stream_behaviours=None, **overrides):
    """Build an ``LLMClient`` with a scripted transport.

    The real constructor would instantiate an ``openai.OpenAI`` client, which
    is irrelevant to transport behaviour and needs a real key, so the
    transport is swapped in afterwards.
    """
    instance = LLMClient.__new__(LLMClient)
    instance.api_key = "test-key"
    instance.base_url = "https://api.example.invalid/v1"
    instance.model = "test-model"
    instance.search_rewrite_model = "test-model"
    instance.request_timeout = 60
    instance.max_retries = 3
    instance.max_concurrency = 5
    for name, value in overrides.items():
        setattr(instance, name, value)
    instance.client = FakeOpenAI(
        raw_behaviours=raw_behaviours, stream_behaviours=stream_behaviours
    )
    return instance


class TestErrorClassification:
    @pytest.mark.parametrize(
        ("status", "fragment"),
        [
            (401, "认证失败"),
            (404, "地址或模型不存在"),
            (429, "限流"),
            (500, "内部错误"),
            (503, "暂时不可用"),
        ],
    )
    def test_known_status_codes_get_a_specific_message(self, status, fragment):
        assert fragment in describe_llm_status_code(status)

    def test_an_unknown_status_code_still_reports_the_number(self):
        assert "418" in describe_llm_status_code(418)

    @pytest.mark.parametrize(
        "payload",
        [
            "<!DOCTYPE html><html></html>",
            "<html><body>hi</body></html>",
            "  <head><title>Login</title></head>",
            "<body>error</body>",
            '<meta charset="utf-8">',
        ],
    )
    def test_detects_an_html_body_where_a_model_reply_was_expected(self, payload):
        assert looks_like_html_response(payload) is True

    @pytest.mark.parametrize("payload", ["plain answer", "", None, '{"a": 1}'])
    def test_normal_text_is_not_mistaken_for_html(self, payload):
        assert looks_like_html_response(payload) is False

    def test_the_html_error_explains_the_likely_cause(self):
        message = html_response_error()
        assert "API Key" in message
        assert "OPENAI_BASE_URL" in message

    @pytest.mark.parametrize(
        "message", ["请求超时", "timeout", "连接失败", "connection", "429", "限流", "502", "网关"]
    )
    def test_transient_failures_are_marked_retryable(self, message):
        assert is_retryable_llm_error(message) is True

    @pytest.mark.parametrize("message", ["401 认证失败", "invalid request", ""])
    def test_permanent_failures_are_not_retryable(self, message):
        assert is_retryable_llm_error(message) is False


class TestRequireClient:
    def test_complete_without_a_key_fails_fast(self):
        with pytest.raises(ValueError, match="API key is required"):
            LLMClient("").complete("system", "user")

    def test_stream_without_a_key_fails_fast(self):
        with pytest.raises(ValueError, match="API key is required"):
            list(LLMClient("").stream("system", "user"))


class TestComplete:
    def test_returns_the_message_content(self):
        instance = make_client(raw_behaviours=[FakeRawResponse("the answer")])
        assert instance.complete("s", "u") == "the answer"

    def test_retries_a_transient_failure_then_succeeds(self, no_sleep):
        instance = make_client(raw_behaviours=
            [ConnectionError("reset"), FakeRawResponse("recovered")],
        )
        assert instance.complete("s", "u") == "recovered"
        assert instance.client.raw_calls == 2

    def test_raises_after_exhausting_retries(self, no_sleep):
        instance = make_client(raw_behaviours=[ConnectionError("down")])
        with pytest.raises(RuntimeError, match="down"):
            instance.complete("s", "u", max_retries=2)

    def test_a_non_2xx_status_is_refused(self, no_sleep):
        instance = make_client(raw_behaviours=[FakeRawResponse("x", status_code=503)])
        with pytest.raises(RuntimeError, match="暂时不可用"):
            instance.complete("s", "u", max_retries=1)

    def test_a_missing_status_code_is_refused(self, no_sleep):
        # Bypass ``FakeRawResponse.__init__``: it assigns an instance
        # ``status_code`` that would shadow any class-level override.
        class NoStatus:
            status_code = None

            def parse(self):
                return FakeCompletion("x")

        instance = make_client(raw_behaviours=[NoStatus()])
        with pytest.raises(RuntimeError, match="状态码"):
            instance.complete("s", "u", max_retries=1)

    def test_an_html_body_is_reported_as_such(self, no_sleep):
        instance = make_client(raw_behaviours=
            [FakeRawResponse("<html><body>login</body></html>")],
        )
        with pytest.raises(RuntimeError, match="网页内容"):
            instance.complete("s", "u", max_retries=1)

    def test_an_empty_completion_is_refused(self, no_sleep):
        instance = make_client(raw_behaviours=[FakeRawResponse("")])
        with pytest.raises(RuntimeError, match="内容为空"):
            instance.complete("s", "u", max_retries=1)

    def test_a_string_instead_of_a_response_object_is_refused(self, no_sleep):
        class StringResponse:
            status_code = 200

            def parse(self):
                return "just a string"

        instance = make_client(raw_behaviours=[StringResponse()])
        with pytest.raises(RuntimeError, match="字符串"):
            instance.complete("s", "u", max_retries=1)

    def test_choices_are_required(self, no_sleep):
        class NoChoices:
            status_code = 200

            def parse(self):
                return object()

        instance = make_client(raw_behaviours=[NoChoices()])
        with pytest.raises(RuntimeError, match="choices"):
            instance.complete("s", "u", max_retries=1)

    def test_structured_content_parts_are_flattened(self):
        class Part:
            def __init__(self, type_, text):
                self.type = type_
                self.text = text

        class Message:
            content = [Part("text", "line one"), Part("image", None), Part("text", "line two")]

        class Choice:
            message = Message()

        class StructuredResponse:
            status_code = 200

            def parse(self):
                return type("Completion", (), {"choices": [Choice()]})()

        instance = make_client(raw_behaviours=[StructuredResponse()])
        assert instance.complete("s", "u") == "line one\nline two"


class TestStream:
    def test_yields_each_delta(self):
        response = FakeStreamingResponse([sse_line("Hello "), sse_line("world")])
        instance = make_client(stream_behaviours=[response])

        assert "".join(instance.stream("s", "u")) == "Hello world"

    def test_ignores_non_data_lines_and_the_done_sentinel(self):
        response = FakeStreamingResponse(
            [b": keep-alive", b"event: ping", b"data: [DONE]", sse_line("only text"), b""]
        )
        instance = make_client(stream_behaviours=[response])

        assert "".join(instance.stream("s", "u")) == "only text"

    def test_skips_malformed_json_lines(self):
        response = FakeStreamingResponse(
            [b"data: {broken", sse_line("good"), b"data: {\"choices\": []}"]
        )
        instance = make_client(stream_behaviours=[response])

        assert "".join(instance.stream("s", "u")) == "good"

    def test_accepts_string_lines(self):
        response = FakeStreamingResponse([sse_line("text").decode("utf-8")])
        instance = make_client(stream_behaviours=[response])

        assert "".join(instance.stream("s", "u")) == "text"


class TestStreamRetryPolicy:
    """The core guarantee: a stream is replayed only while nothing is visible.

    The probe buffer withholds exactly ``STREAM_RETRY_PROBE_CHUNKS`` (one)
    chunk, which makes the boundary sharp:

    * a failure at or before the first chunk yields nothing to the caller, so
      replaying the request is safe;
    * a failure once a second chunk arrives means text has already been
      yielded, so replaying would duplicate it and is refused.
    """

    @staticmethod
    def exploding_stream(chunks_before_failure, exc):
        """A stream that emits ``chunks_before_failure`` then dies."""

        class Exploding(FakeStreamingResponse):
            def iter_lines(self):
                for text in chunks_before_failure:
                    yield sse_line(text)
                raise exc

        return Exploding([])

    def test_probe_window_is_exactly_one_chunk(self):
        assert STREAM_RETRY_PROBE_CHUNKS == 1

    def test_a_failure_before_any_chunk_is_replayed(self, no_sleep):
        instance = make_client(
            stream_behaviours=[
                self.exploding_stream([], ConnectionError("reset")),
                FakeStreamingResponse([sse_line("full answer")]),
            ]
        )

        assert "".join(instance.stream("s", "u")) == "full answer"
        assert instance.client.stream_calls == 2

    def test_a_failure_after_the_buffered_chunk_is_still_replayed(self, no_sleep):
        """The single buffered chunk was never yielded, so the replay is safe."""
        instance = make_client(
            stream_behaviours=[
                self.exploding_stream(["withheld"], ConnectionError("reset")),
                FakeStreamingResponse([sse_line("clean replay")]),
            ]
        )

        collected = list(instance.stream("s", "u"))

        assert collected == ["clean replay"]
        assert "withheld" not in collected
        assert instance.client.stream_calls == 2

    def test_a_failure_after_visible_output_is_never_replayed(self, no_sleep):
        instance = make_client(
            stream_behaviours=[
                self.exploding_stream(["visible ", "text"], ConnectionError("lost mid-stream")),
                FakeStreamingResponse([sse_line("DUPLICATE")]),
            ]
        )

        collected = []
        with pytest.raises(RuntimeError, match="lost mid-stream"):
            for chunk in instance.stream("s", "u", max_retries=3):
                collected.append(chunk)

        # Exactly one copy of what was emitted, and no second attempt.
        assert "".join(collected) == "visible text"
        assert "DUPLICATE" not in collected
        assert instance.client.stream_calls == 1

    def test_a_long_emitted_prefix_is_never_duplicated(self, no_sleep):
        instance = make_client(
            stream_behaviours=[
                self.exploding_stream(["a", "b", "c", "d"], ConnectionError("gone")),
                FakeStreamingResponse([sse_line("a"), sse_line("b"), sse_line("c"), sse_line("d")]),
            ]
        )

        collected = []
        with pytest.raises(RuntimeError, match="gone"):
            collected.extend(instance.stream("s", "u", max_retries=3))

        assert collected == ["a", "b", "c", "d"]
        assert instance.client.stream_calls == 1

    def test_a_single_chunk_stream_is_delivered_not_replayed(self, no_sleep):
        """A one-chunk stream ends inside the probe window and must flush it."""
        instance = make_client(
            stream_behaviours=[
                FakeStreamingResponse([sse_line("only chunk")]),
                FakeStreamingResponse([sse_line("REPLAY")]),
            ]
        )

        assert "".join(instance.stream("s", "u")) == "only chunk"
        assert instance.client.stream_calls == 1

    def test_raises_after_exhausting_retries_before_any_output(self, no_sleep):
        instance = make_client(stream_behaviours=[ConnectionError("down")])

        with pytest.raises(RuntimeError, match="down"):
            list(instance.stream("s", "u", max_retries=2))

        assert instance.client.stream_calls == 2

    def test_a_non_2xx_stream_status_is_refused(self, no_sleep):
        instance = make_client(stream_behaviours=[FakeStreamingResponse([], status_code=500)])

        with pytest.raises(RuntimeError, match="内部错误"):
            list(instance.stream("s", "u", max_retries=1))

    def test_a_stream_with_no_text_is_refused(self, no_sleep):
        instance = make_client(stream_behaviours=[FakeStreamingResponse([b"data: [DONE]"])])

        with pytest.raises(RuntimeError, match="内容为空"):
            list(instance.stream("s", "u", max_retries=1))

    def test_an_html_body_in_a_stream_is_refused_before_emitting(self, no_sleep):
        """A gateway login page must not reach the user as the answer."""
        instance = make_client(
            stream_behaviours=[
                FakeStreamingResponse([sse_line("<!DOCTYPE html><html><body>login</body></html>")])
            ]
        )

        collected = []
        with pytest.raises(RuntimeError, match="网页内容"):
            for chunk in instance.stream("s", "u", max_retries=1):
                collected.append(chunk)

        assert collected == []

class TestClientConfiguration:
    def test_a_request_scoped_key_wins_over_the_environment(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "server-key")
        assert LLMClient("request-key").api_key == "request-key"

    def test_falls_back_to_the_environment(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "server-key")
        assert LLMClient("").api_key == "server-key"

    def test_no_key_anywhere_leaves_the_transport_unbound(self):
        assert LLMClient("").client is None

    def test_a_rewrite_model_override_is_honoured(self, monkeypatch):
        from paperwhisperer.core import config

        monkeypatch.setattr(config, "PAPER_SEARCH_REWRITE_MODEL", "rewrite-model")
        assert LLMClient("k").search_rewrite_model == "rewrite-model"

    def test_the_rewrite_model_falls_back_to_the_main_model(self, monkeypatch):
        from paperwhisperer.core import config

        monkeypatch.setattr(config, "PAPER_SEARCH_REWRITE_MODEL", "")
        assert LLMClient("k").search_rewrite_model == config.OPENAI_MODEL

    def test_worker_counts_never_exceed_the_configured_concurrency(self, monkeypatch):
        from paperwhisperer.core import config

        monkeypatch.setattr(config, "MAX_LLM_CONCURRENCY", 2)
        instance = LLMClient("k")
        assert instance.max_concurrency == 2

    def test_stream_delta_parsing_ignores_empty_choices(self):
        assert LLMClient._parse_stream_delta({}) == ""
        assert LLMClient._parse_stream_delta({"choices": []}) == ""
        assert LLMClient._parse_stream_delta({"choices": [{"delta": {}}]}) == ""
        assert LLMClient._parse_stream_delta({"choices": [{"delta": {"content": "x"}}]}) == "x"
