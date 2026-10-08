"""SSRF protection for remote document fetching.

The import flow lets a user point the server at an arbitrary URL. Without
guards that becomes a request-forgery primitive against internal services, so
every hostname is resolved and each resulting IP is checked before we connect.

**What this defends against, and what it does not.** Requiring *every* address a
hostname resolves to be public refuses a name that mixes a public record with a
private one — multi-record rebinding, where a single lookup returns both.

It does not, on its own, close time-gap rebinding, where two successive lookups
disagree. :mod:`paperwhisperer.remote.download` keeps the time between its check
and its connection as short as the protocol allows, but a client that resolves
the name again when it dials still gets a second, unchecked answer.

**Why failures are cached and successes are not.** A memoized ``True`` is the
wrong thing to keep: it tells a later caller that an address it has not looked
at is one that was checked, which is precisely the assumption that fails under
rebinding, and a TTL of minutes makes that window large. Caching a *failure* has
the opposite shape — it can only refuse a request, never admit one — so that is
what survives here.
"""

from __future__ import annotations

import ipaddress
import socket
import threading
import time
import urllib.parse

from paperwhisperer.core import config

#: hostname -> ``(expires_at, verdict)``. Only ``False`` verdicts are stored;
#: see the module docstring for why a ``True`` is not worth keeping.
_PUBLIC_HOSTNAME_CACHE: dict[str, tuple[float, bool]] = {}
_PUBLIC_HOSTNAME_CACHE_LOCK = threading.Lock()


def is_public_ip_address(value) -> bool:
    """True only for globally routable unicast addresses."""
    try:
        ip = ipaddress.ip_address(str(value or "").strip())
    except ValueError:
        return False
    return ip.is_global and not (
        ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_private
        or ip.is_reserved
        or ip.is_unspecified
    )


def is_ip_literal(value) -> bool:
    try:
        ipaddress.ip_address(value)
        return True
    except ValueError:
        return False


def _get_cached_hostname_result(normalized: str):
    now = time.time()
    with _PUBLIC_HOSTNAME_CACHE_LOCK:
        cached = _PUBLIC_HOSTNAME_CACHE.get(normalized)
        if cached and cached[0] > now:
            return cached[1]
        if cached:
            _PUBLIC_HOSTNAME_CACHE.pop(normalized, None)
    return None


def _set_cached_hostname_result(normalized: str, result: bool) -> None:
    """Memoize a hostname verdict — but only a *negative* one.

    Dropping the positive case is deliberate, not an oversight. A cached ``True``
    would be reused by a caller that then resolves the name again and connects to
    whatever that second lookup returns, so the cache would be asserting the
    safety of an address nobody checked. A cached ``False`` has no such reach: it
    only ever turns a request away.
    """
    if result:
        return
    with _PUBLIC_HOSTNAME_CACHE_LOCK:
        if len(_PUBLIC_HOSTNAME_CACHE) >= config.PUBLIC_HOSTNAME_CACHE_MAX_SIZE:
            oldest_key = min(
                _PUBLIC_HOSTNAME_CACHE, key=lambda key: _PUBLIC_HOSTNAME_CACHE[key][0]
            )
            _PUBLIC_HOSTNAME_CACHE.pop(oldest_key, None)
        _PUBLIC_HOSTNAME_CACHE[normalized] = (
            time.time() + config.PUBLIC_HOSTNAME_CACHE_TTL_SECONDS,
            False,
        )


def clear_hostname_cache() -> None:
    """Drop memoized DNS verdicts. Used by tests to avoid cross-test bleed."""
    with _PUBLIC_HOSTNAME_CACHE_LOCK:
        _PUBLIC_HOSTNAME_CACHE.clear()


def resolve_public_hostname(hostname: str) -> bool:
    """Resolve ``hostname`` and require *every* resolved IP to be public.

    Refuses a name that resolves to a private, loopback, link-local or otherwise
    non-public address, and refuses one that resolves to nothing at all.

    Requiring all addresses (rather than any) blocks a DNS setup that mixes a
    public record with a private one. The one thing this does not do is bind the
    caller to these answers: see the module docstring.
    """
    normalized = (hostname or "").strip().strip(".").lower()
    if not normalized:
        return False
    if normalized in {"localhost", "localhost.localdomain"} or normalized.endswith(".localhost"):
        return False

    if is_public_ip_address(normalized):
        return True
    if is_ip_literal(normalized):
        # A literal that is not public (private range, loopback, reserved).
        return False

    cached_result = _get_cached_hostname_result(normalized)
    if cached_result is not None:
        return cached_result

    try:
        infos = socket.getaddrinfo(normalized, None, type=socket.SOCK_STREAM)
    except socket.gaierror:
        _set_cached_hostname_result(normalized, False)
        return False

    resolved_ips = {info[4][0] for info in infos if info and info[4]}
    result = bool(resolved_ips) and all(is_public_ip_address(ip) for ip in resolved_ips)
    _set_cached_hostname_result(normalized, result)
    return result


def is_public_http_url(raw_url) -> bool:
    """Validate scheme and that the host resolves exclusively to public IPs."""
    parsed = urllib.parse.urlparse(str(raw_url or "").strip())
    try:
        # Accessing ``.port`` is the validation: urlparse defers parsing, so a
        # malformed or out-of-range port only raises here.
        parsed.port  # noqa: B018 - the attribute access is the point
    except ValueError:
        return False

    if parsed.scheme not in {"http", "https"}:
        return False
    return resolve_public_hostname(parsed.hostname or "")
