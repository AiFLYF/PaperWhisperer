"""Remote document download with SSRF and content-type defenses.

Downloaded bytes are validated at three levels — URL reachability, declared
content type, and magic bytes — because any single one can be spoofed by a
misconfigured or hostile origin.

Redirects are validated **per hop, before the request is sent**. urllib follows
a 30x by building the next request itself and opening it immediately, so a plain
"validate, fetch, then re-check ``geturl()``" loop is not an SSRF gate — by the
time the re-check runs, the request to the redirect target has already gone out,
and all it prevents is reading the body back. :class:`_ManualRedirectRequest`
makes the 30x surface as an ``HTTPError`` instead, and :func:`_open_following_redirects`
re-validates the target *before* choosing to go there, so a redirect to
``169.254.169.254`` is refused rather than requested.

**And every hop dials the address that was checked.** Validating a hostname and
then handing the hostname to a client that resolves it again is two lookups, and
they need not agree: a record on a one-second TTL can answer with a public
address for the check and ``169.254.169.254`` for the dial. So the address is
resolved here, vetted, and pinned — :class:`_PinnedHTTPSConnection` below opens
its socket to the vetted IP while still presenting the hostname for SNI and
certificate validation. The connection is what dials, so that is where the pin
has to live.
"""

from __future__ import annotations

import errno
import http.client
import logging
import os
import re
import socket
import urllib.error
import urllib.parse
import urllib.request

from paperwhisperer.core import config
from paperwhisperer.core.text import (
    build_safe_upload_filename,
    build_unique_storage_path,
    close_response_safely,
    remove_file_safely,
    secure_filename,
)
from paperwhisperer.documents.validation import (
    looks_like_direct_file_url,
    normalize_content_type,
    read_remote_file_sample,
    validate_remote_content_length,
    validate_remote_content_type,
)
from paperwhisperer.remote.ssrf import (
    is_ip_literal,
    is_public_http_url,
    resolve_public_ips,
)
from paperwhisperer.search.http import build_ssl_context

logger = logging.getLogger(__name__)

#: How many redirects one download may walk. urllib's own default is 10; this
#: bounds how many hops get re-validated rather than how far apart two documents
#: can be, and stops a redirect cycle from spinning.
MAX_REMOTE_REDIRECTS = 5

#: The 3xx codes urllib treats as redirects. Anything else (a 404, a 500) is a
#: real response and propagates to the caller untouched.
_REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}


class _ManualRedirectRequest(urllib.request.Request):
    """A request urllib will not follow by itself.

    ``urllib`` decides whether to follow a redirect inside
    ``HTTPRedirectHandler.http_error_302``, which loops over ``redirect_dict``
    and raises ``HTTPError`` once that mapping is as long as ``max_redirections``
    allows. Seeding ``redirect_dict`` to that limit therefore converts every 30x
    into an ``HTTPError`` that still carries the original response headers —
    ``Location`` included — without any of the request being re-sent.

    That is exactly the seam this module needs: the redirect arrives as inert
    data the caller can vet, rather than as a request urllib has already made on
    its behalf.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.redirect_dict = {
            str(index): 0
            for index in range(urllib.request.HTTPRedirectHandler.max_redirections)
        }


def _dial_addresses(hostname: str) -> list:
    """The addresses to dial for ``hostname``, resolved once, right here.

    An IP literal is returned as itself: there is no second lookup for a literal
    to disagree with, so re-resolving it would add nothing. The per-hop
    :func:`is_public_http_url` gate is what refuses a private literal, and it
    runs before any connection is built.

    A hostname is resolved here rather than by the connection, so the address
    that gets vetted is the exact address that gets dialled.
    """
    if is_ip_literal(hostname):
        return [hostname]
    return resolve_public_ips(hostname)


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """An HTTP connection that dials a checked address, not a fresh lookup.

    ``host`` keeps naming the origin — that is what the ``Host`` header is built
    from, and what a virtual-hosted origin expects to see. Only the socket is
    pointed elsewhere, at an address resolved and vetted microseconds earlier by
    :func:`_dial_addresses`. The two are separated here rather than by rewriting
    the URL, so the effective URL the caller reports stays in hostname form.
    """

    def _dial_checked(self):
        """Open the socket to a vetted address, or raise the last failure.

        Resolution happens inside ``connect`` so that the address vetted and the
        address dialled cannot come apart: there is no code between them, and no
        second lookup to rebind in between. A hostname can resolve to several
        public addresses and the first can be down while the second answers, so
        each is tried in the order the resolver gave — which is what
        ``socket.create_connection`` does with the addresses of a name, except
        these ones were checked.
        """
        addresses = _dial_addresses(self.host)
        if not addresses:
            raise OSError(errno.ENETUNREACH, "refusing to dial a non-public address")

        last_error = None
        for address in addresses:
            try:
                self.sock = self._create_connection(
                    (address, self.port), self.timeout, self.source_address
                )
            except OSError as exc:
                # A vetted address can still be unreachable; only give up once
                # every vetted address has failed.
                last_error = exc
                continue
            try:
                self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError as exc:
                if exc.errno != errno.ENOPROTOOPT:
                    raise
            return
        raise last_error

    def connect(self):
        self._dial_checked()
        if self._tunnel_host:
            self._tunnel()


class _PinnedHTTPSConnection(_PinnedHTTPConnection, http.client.HTTPSConnection):
    """HTTPS on the same dial, with SNI and certificate checks left intact.

    This is the part that is easy to get subtly wrong. TLS has to name the
    *hostname*: the SNI extension tells the server which certificate to present,
    and that certificate is then validated against the name presented. Handing
    the dialled IP to ``wrap_socket`` instead would negotiate fine against a
    server that ignores SNI and then fail certificate validation against an
    address no certificate can carry — or, far worse, succeed against one that
    did. So the socket is dialled at the vetted address while
    ``server_hostname`` is the hostname, which is the whole point of pinning.

    The MRO puts :class:`_PinnedHTTPConnection` first, so ``connect`` below runs
    the dial itself and then the TLS half explicitly — the same two steps as
    ``HTTPSConnection.connect``, minus the assumption that the name it dials is
    the name it validates.
    """

    def connect(self):
        self._dial_checked()
        self.sock = self._context.wrap_socket(
            self.sock, server_hostname=self._tunnel_host or self.host
        )


class _PinnedHTTPHandler(urllib.request.HTTPHandler):
    """An ``HTTPHandler`` whose connections dial a vetted address."""

    def http_open(self, req):
        return self.do_open(_PinnedHTTPConnection, req)


class _PinnedHTTPSHandler(urllib.request.HTTPSHandler):
    """An ``HTTPSHandler`` whose connections dial a vetted address."""

    def https_open(self, req):
        return self.do_open(_PinnedHTTPSConnection, req, context=self._context)


def _build_pinned_opener(ssl_context) -> urllib.request.OpenerDirector:
    """An opener whose HTTP and HTTPS connections dial vetted addresses.

    Built per download rather than installed globally: ``install_opener`` would
    mutate process-wide state, and every other module in the process shares it.

    ``ProxyHandler({})`` disables proxy support for this opener on purpose. A
    proxy resolves the hostname itself, on its own clock, so a request routed
    through one is not pinned by anything here — keeping proxies would make the
    pin a silent no-op in any proxied deployment rather than an honest absence
    of it.
    """
    return urllib.request.build_opener(
        _PinnedHTTPHandler(),
        _PinnedHTTPSHandler(context=ssl_context),
        urllib.request.ProxyHandler({}),
    )


def _urlopen_pinned(request, opener, timeout):
    """The one place a hop leaves the process, through the pinning opener.

    Every hop goes through here so there is a single seam where the connection
    is established. Two things follow: the addresses the connection dials are the
    ones :func:`_dial_addresses` vetted, and a test can replace the whole
    transport at one address.
    """
    return opener.open(request, timeout=timeout)


def _redirect_location(exc):
    """Read the ``Location`` off a refused redirect, or ``None``."""
    headers = getattr(exc, "headers", None)
    if not headers:
        return None
    location = headers.get("Location") or headers.get("Uri")
    return str(location).strip() if location else None


def _open_following_redirects(source_url, ssl_context, timeout):
    """Open ``source_url``, validating each redirect target before requesting it.

    Returns ``(response, final_url)``, where ``final_url`` is the effective URL
    the response came from, so the caller can name the document after it.

    Each hop is guarded twice over, by two mechanisms that fail independently:
    the target is vetted here *before* it becomes a request, and the connection
    that eventually opens it dials an address it vetted itself. The first stops
    a redirect to ``169.254.169.254`` from being requested at all; the second
    stops a name that answers one way for the check and another way for the dial
    from being connected to unchecked.
    """
    opener = _build_pinned_opener(ssl_context)
    current_url = source_url
    for _hop in range(MAX_REMOTE_REDIRECTS + 1):
        request = _ManualRedirectRequest(
            current_url, headers={"User-Agent": config.APP_USER_AGENT}
        )
        try:
            response = _urlopen_pinned(request, opener, timeout)
        except urllib.error.HTTPError as exc:
            location = _redirect_location(exc)
            if exc.code not in _REDIRECT_STATUS_CODES or not location:
                raise

            # The check happens here, before the next request exists. Refusing
            # here is what keeps a redirect to a private address from ever being
            # requested; raising after the fact would only stop the body read.
            next_url = urllib.parse.urljoin(current_url, location)
            if not is_public_http_url(next_url):
                close_response_safely(exc, "rejected remote redirect")
                raise ValueError("The paper link redirected to a non-public URL.") from None

            close_response_safely(exc, "remote redirect response")
            current_url = next_url
            continue

        # urllib only hands back a response for the URL it was given now, so
        # `geturl()` is this hop's own URL and no post-hoc redirect re-check is
        # needed: the loop above validated the target before requesting it. It
        # stays in hostname form because the pin lives in the connection rather
        # than in the URL, so the caller names the file after the origin and not
        # after an address.
        return response, (response.geturl() or current_url)

    raise ValueError("The paper link redirected too many times.")


def guess_extension_from_content_type(content_type) -> str:
    mapping = {
        "application/pdf": ".pdf",
        "text/plain": ".txt",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    }
    return mapping.get(normalize_content_type(content_type), "")


def extract_filename_from_content_disposition(content_disposition) -> str:
    value = str(content_disposition or "")
    match = re.search(r"filename\*=UTF-8''([^;]+)", value, flags=re.IGNORECASE)
    if match:
        return urllib.parse.unquote(match.group(1)).strip('" ')
    match = re.search(r'filename="?([^";]+)"?', value, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def build_import_filename(title, source_url, content_disposition, content_type) -> str:
    """Derive a safe local filename from the response metadata.

    Preference order: server-suggested name, then the URL path, then a slug of
    the title, with the extension inferred from the URL or content type.
    """
    disposition_name = extract_filename_from_content_disposition(content_disposition)
    url_name = os.path.basename(urllib.parse.urlparse(str(source_url or "")).path)
    title_slug = secure_filename(title or "")
    candidate_name = disposition_name or url_name or title_slug or "imported_paper"
    candidate_root, candidate_ext = os.path.splitext(candidate_name)

    inferred_ext = (
        candidate_ext.lower() if candidate_ext.lower() in config.ALLOWED_EXTENSIONS else ""
    )
    if not inferred_ext:
        inferred_ext = guess_extension_from_content_type(content_type)
    if not inferred_ext:
        inferred_ext = ".pdf"

    safe_root = secure_filename(candidate_root) or title_slug or "imported_paper"
    return build_safe_upload_filename(f"{safe_root}{inferred_ext}")


def iter_downloadable_paper_urls(pdf_url, url):
    """Yield candidate URLs; a landing page must look like a direct file link."""
    seen = set()
    for candidate, require_direct_file in ((pdf_url, False), (url, True)):
        normalized_candidate = str(candidate or "").strip()
        if not normalized_candidate or normalized_candidate in seen:
            continue
        seen.add(normalized_candidate)
        yield normalized_candidate, require_direct_file


def stream_remote_paper(title, pdf_url, url):
    """Open the first usable candidate URL and return its response plus metadata.

    The caller owns the returned response and must close it.
    """
    candidate_urls = list(iter_downloadable_paper_urls(pdf_url, url))
    if not candidate_urls:
        raise ValueError("No downloadable paper link found for this result.")

    ssl_context = build_ssl_context()
    last_error = None

    for source_url, require_direct_file in candidate_urls:
        if not is_public_http_url(source_url):
            last_error = ValueError("Only public http/https paper URLs are allowed.")
            continue
        if require_direct_file and not looks_like_direct_file_url(source_url):
            last_error = ValueError(
                "This result does not provide a direct downloadable file. "
                "Please open it manually and upload the paper file."
            )
            continue

        response = None
        try:
            # Each redirect target is re-validated inside this call, before it is
            # requested — not after, when the damage would already be done.
            response, final_url = _open_following_redirects(
                source_url, ssl_context, config.REMOTE_IMPORT_TIMEOUT_SECONDS
            )

            content_type = normalize_content_type(response.headers.get("Content-Type"))
            # Same reasoning as the upload routes: the byte cap is read live
            # under its own name, so shrinking MAX_UPLOAD_BYTES also shrinks
            # what a remote origin is allowed to push at us.
            validate_remote_content_length(response.headers, config.MAX_UPLOAD_BYTES)
            file_name = build_import_filename(
                title=title,
                source_url=final_url,
                content_disposition=response.headers.get("Content-Disposition"),
                content_type=content_type,
            )
            if not file_name.lower().endswith(tuple(config.SUPPORTED_EXTENSIONS)):
                raise ValueError(
                    "Unsupported remote file type. Please use one of: "
                    f"{config.SUPPORTED_FILE_TYPES_TEXT}"
                )
            validate_remote_content_type(file_name, content_type)
            initial_chunk = read_remote_file_sample(response, file_name)

            return response, file_name, content_type, initial_chunk
        except Exception as exc:
            close_response_safely(response, "failed remote paper response")
            last_error = exc

    if last_error:
        raise last_error
    raise ValueError("Paper import failed.")


def iter_remote_file_chunks(response, max_bytes, initial_chunk=b""):
    """Stream a response in bounded chunks, enforcing the size cap as we go."""
    total_bytes = 0
    try:
        if initial_chunk:
            total_bytes += len(initial_chunk)
            if total_bytes > max_bytes:
                raise ValueError(
                    f"Remote file is too large. Limit: {max_bytes // (1024 * 1024)} MB"
                )
            yield initial_chunk

        while True:
            chunk = response.read(1024 * 64)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > max_bytes:
                raise ValueError(
                    f"Remote file is too large. Limit: {max_bytes // (1024 * 1024)} MB"
                )
            yield chunk
        if total_bytes <= 0:
            raise ValueError("Downloaded file is empty.")
    finally:
        close_response_safely(response, "remote file stream response")


def download_remote_paper(title, pdf_url, url):
    """Download a remote paper to a temp file and return ``(path, filename)``."""
    response = None
    temp_path = None
    try:
        response, file_name, _content_type, initial_chunk = stream_remote_paper(
            title=title, pdf_url=pdf_url, url=url
        )
        temp_path = build_unique_storage_path(config.UPLOAD_FOLDER, file_name)
        with open(temp_path, "wb") as handle:
            for chunk in iter_remote_file_chunks(
                response, config.MAX_UPLOAD_BYTES, initial_chunk
            ):
                handle.write(chunk)
        return temp_path, file_name
    except Exception:
        remove_file_safely(temp_path, "failed remote download file")
        raise
    finally:
        close_response_safely(response, "remote paper download response")


def is_remote_import_error(exc: Exception) -> bool:
    return isinstance(exc, (urllib.error.HTTPError, urllib.error.URLError))
