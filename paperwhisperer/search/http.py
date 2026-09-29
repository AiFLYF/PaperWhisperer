"""HTTP helpers for outbound metadata requests."""

from __future__ import annotations

import json
import logging
import ssl
import time
import urllib.error
import urllib.request

from paperwhisperer.core.text import compute_retry_after_delay, get_retry_delay_seconds

logger = logging.getLogger(__name__)

try:
    import certifi
except ImportError:  # pragma: no cover - certifi ships with requirements
    certifi = None


def build_ssl_context() -> ssl.SSLContext:
    """Prefer certifi's CA bundle so TLS works on hosts without system certs."""
    if certifi:
        return ssl.create_default_context(cafile=certifi.where())
    return ssl.create_default_context()


def _open_with_retry(request, timeout, retries, ssl_context, retry_max_delay):
    """Perform a GET, retrying on 429 and transient network errors."""
    last_error = None
    for attempt in range(max(1, retries)):
        try:
            return urllib.request.urlopen(request, timeout=timeout, context=ssl_context)
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code == 429 and attempt < retries - 1:
                time.sleep(compute_retry_after_delay(exc, attempt, retry_max_delay))
                continue
            raise
        except Exception as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(get_retry_delay_seconds(attempt, max_delay=retry_max_delay))
                continue
            raise
    raise last_error


def http_get_json(url, timeout=20, headers=None, retries=1, ssl_context=None, retry_max_delay=6):
    request = urllib.request.Request(url, headers=headers or {})
    with _open_with_retry(request, timeout, retries, ssl_context, retry_max_delay) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return json.loads(response.read().decode(charset, errors="ignore"))


def http_get_text(url, timeout=20, headers=None, retries=1, ssl_context=None, retry_max_delay=6):
    request = urllib.request.Request(url, headers=headers or {})
    with _open_with_retry(request, timeout, retries, ssl_context, retry_max_delay) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="ignore")


def describe_source_error(source_name: str, exc: Exception) -> str:
    """Turn a provider exception into a short, user-facing reason string."""
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code == 429:
            return f"{source_name}: rate limit reached, please retry in a moment"
        return f"{source_name}: HTTP {exc.code}"
    if isinstance(exc, ssl.SSLCertVerificationError):
        return f"{source_name}: SSL certificate verification failed"
    if isinstance(exc, urllib.error.URLError):
        reason = getattr(exc, "reason", exc)
        if isinstance(reason,
                ssl.SSLCertVerificationError) or "CERTIFICATE_VERIFY_FAILED" in str(reason):
            return f"{source_name}: SSL certificate verification failed"
        if isinstance(reason, TimeoutError) or "timed out" in str(reason).lower():
            return f"{source_name}: request timed out"
        return f"{source_name}: {reason}"
    if isinstance(exc, TimeoutError):
        return f"{source_name}: request timed out"
    return f"{source_name}: {exc}"
