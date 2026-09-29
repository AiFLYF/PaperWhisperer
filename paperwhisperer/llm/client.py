"""OpenAI-compatible LLM transport.

Two call shapes are provided: :meth:`LLMClient.complete` for one-shot
requests and :meth:`LLMClient.stream` for incremental token delivery.

Retry policy differs deliberately between them. A non-streaming call can be
retried freely because nothing has been handed to the caller yet. A streaming
call cannot: once deltas have been yielded the consumer has already rendered
them, so replaying the request would duplicate output. After the first token is
emitted, a failure is surfaced as-is.
"""

from __future__ import annotations

import json
import logging
import time

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from paperwhisperer.core import config
from paperwhisperer.core.errors import (
    describe_llm_status_code,
    html_response_error,
    looks_like_html_response,
)
from paperwhisperer.core.text import (
    extract_message_text,
    get_retry_delay_seconds,
)

logger = logging.getLogger(__name__)

#: Streaming chunks buffered before the first yield. Anything earlier can be
#: discarded and replayed; anything later is already visible to the user.
STREAM_RETRY_PROBE_CHUNKS = 1


class LLMClient:
    """Thin wrapper over the OpenAI SDK with retries, limits and diagnostics."""

    def __init__(self, api_key: str):
        self.api_key = config.resolve_api_key(api_key)
        self.base_url = config.OPENAI_BASE_URL
        self.model = config.OPENAI_MODEL
        self.search_rewrite_model = config.PAPER_SEARCH_REWRITE_MODEL or self.model
        self.request_timeout = config.OPENAI_REQUEST_TIMEOUT_SECONDS
        self.max_retries = config.OPENAI_MAX_RETRIES
        self.max_concurrency = config.MAX_LLM_CONCURRENCY
        self.client = (
            OpenAI(api_key=self.api_key, base_url=self.base_url)
            if self.api_key
            else None
        )

    # -- internals ---------------------------------------------------------

    def _require_client(self) -> None:
        if not self.client:
            raise ValueError(
                "API key is required. Provide it in request body or set OPENAI_API_KEY."
            )

    def _build_messages(self, system_prompt: str, user_prompt: str) -> list[dict]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _classify_error(self, exc: Exception) -> str:
        if isinstance(exc, APIStatusError):
            return describe_llm_status_code(exc.status_code)
        if isinstance(exc, APITimeoutError):
            return "AI 服务请求超时，请稍后重试。"
        if isinstance(exc, APIConnectionError):
            return "AI 服务连接失败，请检查网络、API 地址或供应商服务状态。"
        return str(exc)

    @staticmethod
    def _extract_completion_text(response) -> str:
        if isinstance(response, str):
            if looks_like_html_response(response):
                raise ValueError(html_response_error())
            raise ValueError(
                "AI 服务返回了字符串而不是标准响应对象，请检查供应商接口兼容性。"
            )

        choices = getattr(response, "choices", None)
        if not choices:
            if looks_like_html_response(response):
                raise ValueError(html_response_error())
            raise ValueError("AI 服务返回成功，但响应中缺少 choices 字段。")

        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None) if message else None
        text_content = extract_message_text(content)
        if looks_like_html_response(text_content):
            raise ValueError(html_response_error())
        if not text_content:
            raise ValueError("AI 服务返回内容为空")
        return text_content

    @staticmethod
    def _parse_stream_delta(payload: dict) -> str:
        choices = payload.get("choices") or []
        if not choices:
            return ""
        delta = choices[0].get("delta") or {}
        return delta.get("content") or ""

    # -- public API --------------------------------------------------------

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_retries: int | None = None,
        model: str | None = None,
    ) -> str:
        """Run a non-streaming completion, retrying transient failures."""
        self._require_client()
        retries = self.max_retries if max_retries is None else max_retries
        messages = self._build_messages(system_prompt, user_prompt)
        last_message = ""

        for attempt in range(retries):
            try:
                with config.LLM_REQUEST_SEMAPHORE:
                    raw_response = self.client.chat.completions.with_raw_response.create(
                        model=(model or self.model),
                        messages=messages,
                        temperature=config.LLM_TEMPERATURE,
                        max_tokens=config.LLM_MAX_TOKENS,
                        timeout=self.request_timeout,
                    )

                status_code = getattr(raw_response, "status_code", None)
                response = raw_response.parse()

                if status_code is None:
                    raise ValueError("AI 服务未返回可识别的 HTTP 状态码。")
                if not (200 <= status_code < 300):
                    raise ValueError(describe_llm_status_code(status_code))

                return self._extract_completion_text(response)
            except Exception as exc:
                last_message = self._classify_error(exc)

            if attempt < retries - 1:
                time.sleep(get_retry_delay_seconds(attempt, max_delay=8))
            else:
                logger.error(last_message)
                raise RuntimeError(last_message)

        raise RuntimeError(last_message or "AI 服务请求失败。")

    def stream(
        self,
        system_prompt: str,
        user_prompt: str,
        max_retries: int | None = None,
        model: str | None = None,
    ):
        """Yield incremental text chunks.

        Retries are only safe before the first chunk reaches the caller. Once
        output has been emitted, any failure is raised immediately instead of
        replaying the request and duplicating the client's text.
        """
        self._require_client()
        retries = self.max_retries if max_retries is None else max_retries
        messages = self._build_messages(system_prompt, user_prompt)

        for attempt in range(retries):
            emitted_any = False
            probe_buffer: list[str] = []
            message = ""

            try:
                # Kept as two nested ``with`` statements rather than a combined
                # one: the semaphore has to be acquired *before* the request is
                # built, and the second block is long enough that merging them
                # would bury that ordering.
                with config.LLM_REQUEST_SEMAPHORE:  # noqa: SIM117
                    with self.client.chat.completions.with_streaming_response.create(
                        model=(model or self.model),
                        messages=messages,
                        temperature=config.LLM_TEMPERATURE,
                        max_tokens=config.LLM_MAX_TOKENS,
                        timeout=self.request_timeout,
                        stream=True,
                    ) as response:
                        status_code = getattr(response, "status_code", None)
                        if status_code is None:
                            raise ValueError("AI 服务未返回可识别的 HTTP 状态码。")
                        if not (200 <= status_code < 300):
                            raise ValueError(describe_llm_status_code(status_code))

                        saw_text = False
                        for line in response.iter_lines():
                            if not line:
                                continue
                            decoded = line.decode("utf-8") if isinstance(line, bytes) else str(line)
                            stripped = decoded.strip()
                            if not stripped.startswith("data:"):
                                continue
                            data = stripped[5:].strip()
                            if not data or data == "[DONE]":
                                continue
                            try:
                                payload = json.loads(data)
                            except json.JSONDecodeError:
                                continue

                            text = self._parse_stream_delta(payload)
                            if not text:
                                continue
                            # A proxy login page or API gateway error arrives as
                            # HTML deltas. Catching it before the probe buffer
                            # means nothing has been emitted yet, so the caller
                            # gets a diagnostic instead of raw markup.
                            if looks_like_html_response(text):
                                raise ValueError(html_response_error())
                            saw_text = True

                            if not emitted_any and len(probe_buffer) < STREAM_RETRY_PROBE_CHUNKS:
                                # Hold the first chunk back: if the stream dies
                                # right after it, we can still replay cleanly.
                                probe_buffer.append(text)
                                continue

                            if not emitted_any:
                                for buffered in probe_buffer:
                                    yield buffered
                                probe_buffer.clear()
                                emitted_any = True
                            yield str(text)

                        if not saw_text:
                            raise ValueError("AI 服务返回内容为空")

                        if not emitted_any:
                            # Stream ended while still holding the probe buffer.
                            for buffered in probe_buffer:
                                yield buffered
                            probe_buffer.clear()
                        return
            except Exception as exc:
                if emitted_any:
                    # Text already reached the client; replaying would duplicate it.
                    logger.error("Streaming response failed after output was emitted: %s", exc)
                    raise RuntimeError(self._classify_error(exc)) from exc
                message = self._classify_error(exc)
                probe_buffer.clear()

            if attempt < retries - 1:
                time.sleep(get_retry_delay_seconds(attempt, max_delay=8))
            else:
                logger.error(message)
                raise RuntimeError(message)

        raise RuntimeError("AI 服务请求失败。")
