"""HTTP helpers for outbound metadata requests."""

from __future__ import annotations

import json
import logging
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from paperwhisperer.core import config
from paperwhisperer.core.text import compute_retry_after_delay, get_retry_delay_seconds

logger = logging.getLogger(__name__)

try:
    import certifi
except ImportError:  # pragma: no cover - certifi ships with requirements
    certifi = None


# --------------------------------------------------------------------------
# Proactive per-provider rate limiting
#
# The retry path above only reacts once a provider has already said "429". Some
# providers state their rule as a hard rate they refuse bursts against — arXiv
# asks for no more than one request every three seconds on a single connection,
# summed across all clients — so reacting to a refusal is both late and, for
# those providers, rude. This spaces requests out *before* they are sent.
#
# The gate is a per-host reservation, not a lock held across the request: a
# caller reserves its slot under a short lock, then sleeps outside it until the
# slot arrives, so requests overlap once they are spaced. Under concurrency
# every caller still gets a distinct slot, which is the property a plain "sleep
# interval then request" lacks.
#
# The interval is read live from config on each call, so an operator (or a test)
# can retune a provider without re-importing the process, and setting an
# interval to 0 disables the proactive spacing for that host while leaving the
# 429 backoff above in force.
# --------------------------------------------------------------------------

_HOST_INTERVAL_LOCK = threading.Lock()
_NEXT_SLOT: dict[str, float] = {}


def _host_interval_seconds(host: str) -> float:
    """The configured minimum seconds between requests to ``host``.

    Read live rather than snapshotted so config changes (and test overrides)
    take effect immediately. An unrecognised host returns 0: this module must
    not guess a rate for a service whose terms were never read, and the 429
    backoff above still applies to it.
    """
    host = (host or "").lower()
    if host in {"export.arxiv.org", "arxiv.org"}:
        return config.ARXIV_MIN_INTERVAL_SECONDS
    if host == "api.semanticscholar.org":
        return config.SEMANTIC_SCHOLAR_MIN_INTERVAL_SECONDS
    if host == "api.crossref.org":
        return config.CROSSREF_MIN_INTERVAL_SECONDS
    if host == "api.openalex.org":
        return config.OPENALEX_MIN_INTERVAL_SECONDS
    return 0.0


def reset_rate_limits() -> None:
    """Forget every reserved slot. Tests use this to avoid cross-test spacing."""
    with _HOST_INTERVAL_LOCK:
        _NEXT_SLOT.clear()


def _throttle(url: str) -> None:
    """Block until this provider's configured gap has elapsed since its last call."""
    host = urllib.parse.urlparse(url).hostname or ""
    interval = _host_interval_seconds(host)
    if interval <= 0:
        return
    now = time.monotonic()
    with _HOST_INTERVAL_LOCK:
        start = max(now, _NEXT_SLOT.get(host, 0.0))
        _NEXT_SLOT[host] = start + interval
    delay = start - now
    if delay > 0:
        time.sleep(delay)


def build_ssl_context() -> ssl.SSLContext:
    """Prefer certifi's CA bundle so TLS works on hosts without system certs."""
    if certifi:
        return ssl.create_default_context(cafile=certifi.where())
    return ssl.create_default_context()


def _open_with_retry(request, timeout, retries, ssl_context, retry_max_delay):
    """Perform a GET, retrying on 429 and transient network errors.

    Each attempt — including each retry — passes the provider's proactive rate
    gate first, because a retry is another request and is subject to the same
    published rate as the first one.
    """
    last_error = None
    for attempt in range(max(1, retries)):
        try:
            _throttle(request.full_url)
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
