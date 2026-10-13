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
from collections.abc import Callable
from dataclasses import dataclass

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from paperwhisperer.core import config
from paperwhisperer.core.errors import (
    NonRetryableError,
    describe_llm_status_code,
    html_response_error,
    looks_like_html_response,
)
from paperwhisperer.core.text import (
    extract_message_text,
    get_retry_delay_seconds,
    redact_secrets,
)
from paperwhisperer.ingestion.limits import SecretRedactingFilter, install_secret_filter
from paperwhisperer.llm import output as model_output

logger = logging.getLogger(__name__)

#: Streaming chunks buffered before the first yield. Anything earlier can be
#: discarded and replayed; anything later is already visible to the user.
STREAM_RETRY_PROBE_CHUNKS = 1

#: Largest SSE frame the client will parse. A content delta is a fragment of a
#: sentence; a frame anywhere near this size is a gateway returning an error
#: page or a proxy misbehaving, and parsing it costs memory the request has no
#: reason to spend. Read by :meth:`LLMClient._parse_stream_frame`.
STREAM_FRAME_MAX_CHARS = 65_536


@dataclass(frozen=True, slots=True)
class LLMUsage:
    """Everything a cost ledger needs to know about one completion.

    The counterpart of :class:`paperwhisperer.llm.embeddings.EmbeddingUsage`,
    and shaped like it on purpose: one immutable object handed to the caller, a
    ``cost`` that returns ``None`` when the number cannot be known, and an
    ``as_analysis_run_columns`` for the row ``analysis_runs`` will eventually
    hold. Two cost types in one codebase that disagree about what ``None`` means
    are worse than one cost type -- a dashboard would average them into a
    plausible lie.

    Until this existed, ``complete()`` returned a ``str`` and threw
    ``response.usage`` away, so nothing in the product could answer "what did
    this analysis cost" or "which section is expensive" at all.
    """

    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    #: Served from the provider's cache. Reported *inside* ``prompt_tokens`` by
    #: the providers that report it, so it is not added to it: a cached prompt
    #: is billed at a discount rather than for free, and summing the two would
    #: double-count the same tokens.
    cached_prompt_tokens: int = 0
    #: Reasoning tokens. Also inside ``completion_tokens`` where reported
    #: separately, and the reason a run that returned three words can still be
    #: the expensive one.
    reasoning_tokens: int = 0
    #: False when the provider reported no usage block. Many
    #: OpenAI-compatible gateways send none, and an absent block is not a
    #: zero-cost run -- it is an unknown one.
    tokens_reported: bool = True
    price_per_mtok: float | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost(self) -> float | None:
        """Cost in the account's currency, or ``None`` if it cannot be known.

        ``None`` for "unpriced" and ``None`` for "not reported" are the same
        answer for the same reason: neither is a number, and reporting ``0.0``
        for either would let a cost view claim a free run that was never
        measured.
        """
        if self.price_per_mtok is None or not self.tokens_reported:
            return None
        return self.total_tokens * self.price_per_mtok / 1_000_000

    def merge(self, other: LLMUsage) -> LLMUsage:
        """Accumulate another completion into this one (counters add up)."""
        if not isinstance(other, LLMUsage):
            return self
        return LLMUsage(
            model=self.model or other.model,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cached_prompt_tokens=self.cached_prompt_tokens + other.cached_prompt_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            tokens_reported=self.tokens_reported and other.tokens_reported,
            price_per_mtok=(
                other.price_per_mtok if other.price_per_mtok is not None else self.price_per_mtok
            ),
        )

    def as_analysis_run_columns(self) -> dict:
        """Columns ``analysis_runs`` should carry for an LLM run.

        The same contract ``EmbeddingUsage.as_analysis_run_columns`` offers, so
        one row can hold both sides of a run's cost. ``llm_tokens_reported`` is
        a column rather than something inferred: a row with zero tokens and
        ``tokens_reported=False`` means "not measured", and reading it as
        "nothing was spent" is exactly the mistake this object exists to
        prevent.
        """
        return {
            "llm_model": self.model,
            "llm_prompt_tokens": self.prompt_tokens,
            "llm_completion_tokens": self.completion_tokens,
            "llm_cached_prompt_tokens": self.cached_prompt_tokens,
            "llm_reasoning_tokens": self.reasoning_tokens,
            "llm_tokens_reported": self.tokens_reported,
            "llm_cost": self.cost,
        }


def _usage_from_response(response, model: str, price_per_mtok: float | None) -> LLMUsage:
    """Read a completion's ``usage`` block, tolerating a provider that omits it.

    Every field is read through ``getattr`` because the OpenAI-compatible field
    is a convention and not a contract: gateways routinely drop the ``*_details``
    sub-objects, some report ``total_tokens`` only, and some report nothing. A
    ledger that raises on the first of those collects nothing at all.

    ``price_per_mtok`` is passed in rather than read from config so a caller
    that knows better (a per-model price table) can supply it, and so the tests
    do not have to mutate a module global to price a run.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return LLMUsage(model=model, tokens_reported=False, price_per_mtok=price_per_mtok)

    def _int(name: str) -> int:
        value = getattr(usage, name, None)
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    cached = _int_from(prompt_details, "cached_tokens") if prompt_details is not None else 0
    reasoning = (
        _int_from(completion_details, "reasoning_tokens")
        if completion_details is not None else 0
    )
    return LLMUsage(
        model=str(getattr(usage, "model", None) or model),
        prompt_tokens=_int("prompt_tokens"),
        completion_tokens=_int("completion_tokens"),
        cached_prompt_tokens=cached,
        reasoning_tokens=reasoning,
        tokens_reported=True,
        price_per_mtok=price_per_mtok,
    )


def _int_from(holder, name: str) -> int:
    value = getattr(holder, name, None)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0



def _called_a_tool(response, message) -> bool:
    """Whether the model answered by calling a tool instead of replying.

    Checks both the documented ``finish_reason`` and the message's own
    ``tool_calls``, because providers disagree about which they populate and
    an empty ``content`` with no visible signal is the case this exists for.
    """
    finish_reason = getattr(getattr(response, "choices", [None])[0], "finish_reason", None)
    if isinstance(finish_reason, str) and finish_reason == "tool_calls":
        return True
    return bool(getattr(message, "tool_calls", None))


class LLMClient:
    """Thin wrapper over the OpenAI SDK with retries, limits and diagnostics."""

    #: The redaction filter is installed once per process, on first client
    #: construction, rather than at import: a library module that mutates the
    #: root logger as a side effect of being imported is a trap, and this
    #: module is imported by tests that must not have their log configuration
    #: changed underneath them.
    def __init__(self, api_key: str):
        LLMClient._ensure_secret_filter()
        self.api_key = config.resolve_api_key(api_key)
        self.base_url = config.OPENAI_BASE_URL
        self.model = config.OPENAI_MODEL
        self.search_rewrite_model = config.PAPER_SEARCH_REWRITE_MODEL or self.model
        self.request_timeout = config.OPENAI_REQUEST_TIMEOUT_SECONDS
        self.max_retries = config.OPENAI_MAX_RETRIES
        self.max_concurrency = config.MAX_LLM_CONCURRENCY
        # Read at construction for the same reason the embedding gateway reads
        # its price there: the usage object is built per request, and a config
        # module global re-read per request is a value that can change between
        # two halves of one analysis.
        self.price_per_mtok = config.OPENAI_PRICE_PER_MTOK
        self.client = (
            OpenAI(api_key=self.api_key, base_url=self.base_url)
            if self.api_key
            else None
        )

    # -- internals ---------------------------------------------------------

    @classmethod
    def _ensure_secret_filter(cls) -> None:
        """Put a credential filter on the root logger, once.

        This is the defence the redaction tests are really about. Redacting
        inside ``_classify_error`` only covers the strings this module chooses
        to emit; a provider SDK, a third-party library, or a future call site
        can log a credential through any logger in the process, and "where a
        secret gets logged is not knowable in advance". A filter on the root
        logger sees every record regardless of which logger made it.

        Without it the function exists and is never called, which is the worst
        outcome: the code reads as protected and is not.
        """
        # Ask the logger whether the filter is attached rather than keeping a
        # flag. A flag is wrong here: anything that reconfigures logging -- a
        # test fixture that restores handlers, a host application that resets
        # the root logger at startup -- removes the filter while the flag
        # still says "installed", and the protection is silently gone. The
        # check is a list scan on a construction path, which is nothing.
        root = logging.getLogger()
        if any(isinstance(item, SecretRedactingFilter) for item in root.filters):
            return
        try:
            install_secret_filter()
        except Exception:  # pragma: no cover - never block construction
            logger.warning("凭据脱敏日志过滤器安装失败，日志可能包含敏感值。", exc_info=True)

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
        # The fallback, and the only place an unvetted provider string enters
        # this module. Whatever comes back from here is logged verbatim *and*
        # raised to the HTTP layer, so an SDK that quotes the Authorization
        # header in its error body would put the key in both the log and the
        # response. The recognised paths above cannot: they map a status code
        # to a sentence this module wrote.
        #
        # Redacted rather than dropped, because the diagnostic is worth keeping
        # and the credential is not — and `redact_secrets` is the one
        # implementation in the codebase, so this cannot fall out of step with
        # the logging filter that scrubs the same strings on the way out.
        return redact_secrets(str(exc)) or exc.__class__.__name__

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

        # The verdict itself belongs to ``llm.output`` so that the failure
        # taxonomy has one owner. What stays here is the decision about what to
        # *do* with a verdict, which is transport policy: ``REFUSED`` is not
        # retried, because re-sending an identical prompt can only produce an
        # identical refusal.
        #
        # A reasoning model asked a question the document cannot answer will
        # sometimes reach for a tool — grep, glob, bash — that this product
        # never offers and never executes, and reply with HTTP 200,
        # finish_reason="tool_calls" and an empty content.
        #
        # This used to be reported as "AI 服务返回内容为空" and then retried,
        # which turned the *most likely* question a user asks (one the
        # document does not answer) into a 500 after three identical resends.
        # It is not retried any more: re-sending the same prompt can only
        # produce the same tool call.
        #
        # The model's ``reasoning_content`` does contain a usable refusal here
        # — it usually reasons straight to "the document does not say" — and it
        # is tempting to return it. It is deliberately NOT used. That field is
        # scratchpad: in the measured case it ran "... I should say the
        # document excerpt doesn't contain that info. I could check if there's
        # a file in the working directory. I have tools. Let me try to see."
        # Handing a user the model's private reasoning, mid-thought, about
        # tools it does not have, is a worse defect than an honest error.
        verdict = model_output.classify_completion(
            text_content,
            called_a_tool=_called_a_tool(response, message),
            finish_reason=getattr(choices[0], "finish_reason", None),
            label="AI 回复",
        )
        if verdict.status is model_output.OutputStatus.REFUSED:
            raise NonRetryableError(model_output.TOOL_CALL_MESSAGE)

        if looks_like_html_response(text_content):
            raise ValueError(html_response_error())
        if not text_content:
            if verdict.detail == "token_budget":
                raise NonRetryableError(
                    "模型在输出正文前就用满了 token 上限（推理占用了全部预算）。"
                    f"当前上限 {config.LLM_MAX_TOKENS}，请调高 OPENAI_LLM_MAX_TOKENS 后重试。"
                )
            raise ValueError("AI 服务返回内容为空")
        return text_content

    @staticmethod
    def _parse_stream_delta(payload: dict) -> str:
        choices = payload.get("choices") or []
        if not choices:
            return ""
        choice = choices[0]
        # A stream carries the same envelope the non-streaming path already
        # inspects, and it was not inspecting it: a tool-calling reply arrived
        # as a well-formed chunk with empty content, which the caller could
        # only read as "the model returned nothing" — so the one case with a
        # precise diagnosis was the one case the two paths disagreed about.
        # Sides other than ``delta`` are intentionally not touched: a
        # transcript, not a control channel.
        if str(choice.get("finish_reason") or "") == "tool_calls" or choice.get(
            "delta", {}
        ).get("tool_calls"):
            raise NonRetryableError(
                "模型尝试调用本产品不提供的工具，且未给出可用回复。"
                "请重试，或在提示中明确要求直接作答、不要使用工具。"
            )
        delta = choice.get("delta") or {}
        return delta.get("content") or ""

    @staticmethod
    def _parse_stream_frame(data: str) -> dict | None:
        """One SSE frame, or ``None`` for a frame we will not act on.

        Routed through ``llm.output`` with ``recover=False`` on purpose. This
        is the one place in the product that is *not* parsing model prose: the
        frame comes from a proxy or a gateway, and every repair the model-output
        path performs is wrong here -- turning a corrupt transport frame into a
        plausible one is how a truncated stream becomes a sentence that stops
        mid-word and is rendered as though the model chose to stop there.

        What the layer does add is the bound. A frame is skipped for being
        unreadable exactly as before, and now also for being too large to be
        worth a parse.
        """
        frame = model_output.parse_json(
            data, label="流式响应帧", recover=False, max_chars=STREAM_FRAME_MAX_CHARS
        )
        return frame.value if frame.ok and isinstance(frame.value, dict) else None

    # -- public API --------------------------------------------------------

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_retries: int | None = None,
        model: str | None = None,
        temperature: float | None = None,
        usage_sink: Callable[[LLMUsage], None] | None = None,
    ) -> str:
        """Run a non-streaming completion, retrying transient failures.

        ``temperature`` defaults to ``config.LLM_TEMPERATURE``, exactly as
        before. It is a parameter because not every task wants the same one:
        the structured fact extraction in ``analysis/paper_facts.py`` needs a
        low temperature, and until this existed the only route to one was
        mutating a module global around the call — which two concurrent
        requests cannot both do. Callers that omit it are unaffected.

        ``usage_sink`` receives the completion's :class:`LLMUsage` on success,
        and is a callback rather than a second return value because the return
        is a ``str`` at thirteen call sites and in a dozen test assertions —
        changing it would be a migration, not a feature. A caller that wants
        the cost passes one in; a caller that does not is unaffected. It is
        called only for a completion that actually produced text, so a retried
        request reports the attempt that succeeded and not the ones that
        burned tokens before it.
        """
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
                        temperature=(
                            config.LLM_TEMPERATURE if temperature is None else temperature
                        ),
                        max_tokens=config.LLM_MAX_TOKENS,
                        timeout=self.request_timeout,
                    )

                status_code = getattr(raw_response, "status_code", None)
                response = raw_response.parse()

                if status_code is None:
                    raise ValueError("AI 服务未返回可识别的 HTTP 状态码。")
                if not (200 <= status_code < 300):
                    raise ValueError(describe_llm_status_code(status_code))

                text = self._extract_completion_text(response)
                if usage_sink is not None:
                    usage_sink(_usage_from_response(
                        response, model or self.model, self.price_per_mtok
                    ))
                return text
            except NonRetryableError as exc:
                # A final answer, not a blip. Re-sending an identical prompt
                # can only produce an identical final answer, so retrying here
                # costs the user seconds and changes nothing.
                logger.error(str(exc))
                raise
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
                                payload = self._parse_stream_frame(data)
                            except json.JSONDecodeError:
                                continue
                            if payload is None:
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
                    # The exception object, not the classified message: this
                    # logged straight to `str(exc)`, which is the unredacted
                    # path the classifier exists to avoid. Classify first, so
                    # the same redaction applies.
                    logger.error(
                        "Streaming response failed after output was emitted: %s",
                        self._classify_error(exc),
                    )
                    raise RuntimeError(self._classify_error(exc)) from exc
                message = self._classify_error(exc)
                probe_buffer.clear()

            if attempt < retries - 1:
                time.sleep(get_retry_delay_seconds(attempt, max_delay=8))
            else:
                logger.error(message)
                raise RuntimeError(message)

        raise RuntimeError("AI 服务请求失败。")
