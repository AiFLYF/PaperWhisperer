"""Structured error responses.

Every JSON API failure carries ``error``, ``code`` and ``timestamp`` so the
frontend and scripts can branch on a stable machine-readable code.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def build_error_payload(message: str, code: str = "bad_request") -> dict:
    """Build the error body every route returns.

    The message is redacted here rather than at each call site. Callers pass
    ``str(exc)`` — 21 places across three route modules were audited doing it,
    plus an f-string variant a pattern-based audit would have missed — and a
    provider SDK that quotes an Authorization header in its error body turns
    any of them into a credential in a browser. Scrubbing at the boundary
    closes those 21 and every future one; scrubbing at the call site closes
    the 21 that exist today and leaves the next one open.
    """
    from paperwhisperer.core.text import redact_secrets

    return {"error": redact_secrets(message), "code": code, "timestamp": now_iso()}


def build_error_response(message: str, status_code: int = 400,
        code: str = "bad_request") -> JSONResponse:
    return JSONResponse(content=build_error_payload(message, code=code), status_code=status_code)


def build_sse_event(event_name: str, payload: dict | None = None) -> str:
    data = json.dumps(payload or {}, ensure_ascii=False)
    return f"event: {event_name}\ndata: {data}\n\n"


def build_sse_headers() -> dict:
    return dict(SSE_HEADERS)


def describe_remote_http_error(status_code: int) -> str:
    if status_code == 404:
        return "The paper file could not be found at the remote source."
    if status_code == 403:
        return "The remote source denied access to the paper file."
    if status_code == 429:
        return "The remote source rate limited the paper download. Please retry in a moment."
    return f"Remote paper download failed with HTTP {status_code}."


def describe_remote_url_error(reason) -> str:
    import ssl

    if isinstance(reason,
            ssl.SSLCertVerificationError) or "CERTIFICATE_VERIFY_FAILED" in str(reason):
        return "SSL certificate verification failed while downloading the paper file."
    return f"Paper download failed: {reason}"


def describe_llm_status_code(status_code: int) -> str:
    status_messages = {
        400: "AI 服务请求格式错误，请检查模型配置、参数或接口兼容性。",
        401: "AI 服务认证失败，请检查 API Key 是否正确或已过期。",
        403: "AI 服务拒绝访问，当前 API Key 可能无权使用该模型或接口。",
        404: "AI 服务地址或模型不存在，请检查 OPENAI_BASE_URL 和模型名称。",
        408: "AI 服务请求超时，请稍后重试。",
        409: "AI 服务请求冲突，请稍后重试。",
        415: "AI 服务不支持当前请求媒体类型，请检查供应商兼容性。",
        422: "AI 服务无法处理当前请求，请检查输入内容或参数。",
        429: "AI 服务触发限流，请稍后重试或降低并发。",
        500: "AI 服务提供商内部错误，请稍后重试。",
        502: "AI 服务网关异常，请稍后重试。",
        503: "AI 服务暂时不可用，请稍后重试。",
        504: "AI 服务网关超时，请稍后重试。",
    }
    return status_messages.get(status_code, f"AI 服务请求失败，状态码: {status_code}。")


def looks_like_html_response(value) -> bool:
    """Detect an HTML page returned where a model response was expected.

    This is the classic symptom of a missing API key or a ``base_url`` that
    points at a website rather than an API endpoint.
    """
    text = str(value or "").lstrip().lower()
    html_markers = ("<!doctype html", "<html", "<head", "<body", "<meta ")
    return any(text.startswith(marker) for marker in html_markers)


def html_response_error() -> str:
    return (
        "AI 服务返回了网页内容而不是模型结果。"
        "通常是 API Key 缺失、无效，或 OPENAI_BASE_URL 指向了网页地址。"
    )


def is_retryable_llm_error(message: str) -> bool:
    normalized = str(message or "").lower()
    retry_markers = (
        "超时", "timeout", "连接失败", "connection",
        "429", "限流", "502", "503", "504", "网关",
    )
    return any(marker in normalized for marker in retry_markers)


class NonRetryableError(ValueError):
    """An LLM response that is final, not transient.

    The client retries almost everything, which is right for a timeout or a
    503 and wrong for a response the model has already committed to. Retrying
    one of those does not produce a different answer — it produces the same
    answer three times, slowly, and then a 500 for the user.

    A worked case: a reasoning model that answers "this document does not
    say" by calling a tool the product does not offer, with an empty
    ``content``. Re-sending the identical prompt produces an identical empty
    response, so the retry loop can only burn time.
    """
