"""SSRF protection for remote document fetching.

The import flow lets a user point the server at an arbitrary URL. Without
guards that becomes a request-forgery primitive against internal services, so
every hostname is resolved and each resulting IP is checked before we connect.

**What this defends against, and what it does not.** Requiring *every* address a
hostname resolves to be public refuses a name that mixes a public record with a
private one — multi-record rebinding, where a single lookup returns both.

A caller that only asks "is this hostname safe?" still cannot close time-gap
rebinding, where two successive lookups disagree. :func:`resolve_public_ips` is
the function that does: it resolves once and hands back *the addresses*, so the
caller dials an address somebody looked at instead of resolving the name again
when it connects. :mod:`paperwhisperer.remote.download` pins every hop that way.

**Why failures are cached and successes are not.** A memoized ``True`` is the
wrong thing to keep: it tells a later caller that an address it has not looked
at is one that was checked, which is precisely the assumption that fails under
rebinding, and a TTL of minutes makes that window large. Caching a *failure* has
the opposite shape — it can only refuse a request, never admit one — so that is
what survives here. The pinned resolver below is unaffected by that rule either
way: it never reuses a verdict, it publishes an address the caller dials
directly.
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


def normalize_hostname(hostname: str) -> str:
    """Strip the decorations a hostname picks up in a URL, or ``""`` if unusable."""
    return (hostname or "").strip().strip(".").lower()


def _is_refused_by_name(normalized: str) -> bool:
    """Names that must be refused without consulting DNS at all.

    ``localhost`` and its relatives are refused up front because they are
    resolved by every machine's own resolver, not by a record an attacker
    controls — a DNS answer here carries no authority to vouch for anything.
    """
    return normalized in {"localhost", "localhost.localdomain"} or normalized.endswith(
        ".localhost"
    )


def _resolve_addresses(normalized: str) -> list:
    """Every address ``normalized`` resolves to, in the order the resolver gave."""
    infos = socket.getaddrinfo(normalized, None, type=socket.SOCK_STREAM)
    addresses = []
    for info in infos:
        address = info[4][0] if info and info[4] else None
        if address and address not in addresses:
            addresses.append(address)
    return addresses


def _public_addresses(normalized: str) -> list:
    """The addresses for ``normalized``, or ``[]`` when any of them is not public.

    This is the one place the "every resolved address must be public" rule
    lives, so the yes/no predicate and the pinned resolver cannot drift apart.
    An empty list covers every refusal at once — unusable name, loopback name,
    non-public literal, unresolvable name, mixed public/private answer — because
    a caller has no correct way to act on a partial answer.
    """
    if not normalized or _is_refused_by_name(normalized):
        return []
    if is_public_ip_address(normalized):
        return [normalized]
    if is_ip_literal(normalized):
        # A literal that is not public (private range, loopback, reserved).
        return []

    try:
        addresses = _resolve_addresses(normalized)
    except socket.gaierror:
        return []
    if not addresses or not all(is_public_ip_address(ip) for ip in addresses):
        return []
    return addresses


def resolve_public_ips(hostname: str) -> list:
    """Resolve ``hostname`` once and return the addresses that are safe to dial.

    This is the function that closes time-gap rebinding. ``resolve_public_hostname``
    answers yes/no and leaves the caller free to resolve the name a second time
    when it dials — and a name with a one-second TTL can answer "public" the
    first time and ``169.254.169.254`` the second. Here there is no second lookup
    to disagree: the caller dials one of the addresses returned below, and every
    one of them was part of the single resolution that was just vetted.

    Returns an empty list whenever any resolved address is not public, so a
    caller reads emptiness as "do not connect" and never has to decide what a
    partial answer means. An IP literal is returned as-is when public and
    refused when not; a literal has no second lookup to rebind, so pinning it
    is a formality rather than a defence.

    Deliberately does *not* consult the verdict cache. The cache exists to
    avoid repeating refusals, and this function's whole job is to produce a
    fresh, address-bearing answer rather than a remembered yes.
    """
    normalized = normalize_hostname(hostname)
    if not normalized or _is_refused_by_name(normalized) or is_ip_literal(normalized):
        # Refused on the name alone, so there is no lookup worth remembering.
        return _public_addresses(normalized)

    addresses = _public_addresses(normalized)
    if not addresses:
        # A failure is safe to memoize: it can only ever turn a request away.
        _set_cached_hostname_result(normalized, False)
    return addresses


def resolve_public_hostname(hostname: str) -> bool:
    """Resolve ``hostname`` and require *every* resolved IP to be public.

    Refuses a name that resolves to a private, loopback, link-local or otherwise
    non-public address, and refuses one that resolves to nothing at all.

    Requiring all addresses (rather than any) blocks a DNS setup that mixes a
    public record with a private one. The one thing this does not do is bind the
    caller to these answers: a caller that resolves the name again at connect
    time gets a second, unchecked answer. Use :func:`resolve_public_ips` and dial
    what it returns when that binding is required.
    """
    normalized = normalize_hostname(hostname)
    if not normalized:
        return False
    if _is_refused_by_name(normalized) or is_ip_literal(normalized):
        # Refused on the name alone; no lookup happened, so nothing to memoize.
        return bool(_public_addresses(normalized))

    cached_result = _get_cached_hostname_result(normalized)
    if cached_result is not None:
        return cached_result

    addresses = _public_addresses(normalized)
    if not addresses:
        _set_cached_hostname_result(normalized, False)
    return bool(addresses)


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
