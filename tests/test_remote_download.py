"""SSRF gate and remote-download validation.

The import flow fetches a user-supplied URL, which is a request-forgery
primitive unless every hop is checked. These tests pin down the gate itself
and the three independent content checks (declared length, declared type,
magic bytes) that run after a connection succeeds.
"""

from __future__ import annotations

import http.server
import os
import socket
import ssl
import threading
import urllib.error
import urllib.request
from email.message import Message

import pytest

from paperwhisperer.core import config
from paperwhisperer.documents.validation import (
    looks_like_direct_file_url,
    normalize_content_type,
    parse_content_length,
    validate_document_file_signature,
    validate_remote_content_length,
    validate_remote_content_type,
)
from paperwhisperer.remote import download, ssrf

from .helpers import FakeResponse


def _addrinfo(*ips):
    return [(None, None, None, None, (ip, 443)) for ip in ips]


def redirect_error(location: str) -> urllib.error.HTTPError:
    """Build the 30x urllib surfaces once redirect following is switched off.

    ``urllib`` turns a redirect it refuses to follow into an ``HTTPError`` that
    still carries the original response headers, so ``Location`` is readable —
    which is what lets the download loop vet the target before going there.
    """
    headers = Message()
    headers["Location"] = location
    return urllib.error.HTTPError(
        "https://example.com/paper.pdf", 302, "Found", headers, None
    )


class TestIsPublicIpAddress:
    @pytest.mark.parametrize(
        "value",
        ["93.184.216.34", "8.8.8.8", "2606:2800:220:1:248:1893:25c8:1946"],
    )
    def test_accepts_globally_routable_addresses(self, value):
        assert ssrf.is_public_ip_address(value) is True

    @pytest.mark.parametrize(
        "value",
        [
            "127.0.0.1",
            "10.0.0.5",
            "192.168.1.1",
            "172.16.0.1",
            "169.254.169.254",  # cloud metadata endpoint
            "0.0.0.0",
            "224.0.0.1",
            "::1",
            "fe80::1",
            "not-an-ip",
            "",
            None,
        ],
    )
    def test_rejects_private_loopback_and_malformed(self, value):
        assert ssrf.is_public_ip_address(value) is False


class TestIsIpLiteral:
    def test_detects_literals(self):
        assert ssrf.is_ip_literal("127.0.0.1") is True
        assert ssrf.is_ip_literal("::1") is True

    def test_hostnames_are_not_literals(self):
        assert ssrf.is_ip_literal("example.com") is False
        assert ssrf.is_ip_literal("") is False


class TestResolvePublicHostname:
    def test_accepts_a_hostname_resolving_only_to_public_ips(self, monkeypatch):
        monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: _addrinfo("93.184.216.34"))
        assert ssrf.resolve_public_hostname("example.com") is True

    def test_rejects_a_hostname_with_any_private_address(self, monkeypatch):
        # Requiring *all* addresses blocks DNS rebinding that mixes a public
        # record with a private one.
        monkeypatch.setattr(
            socket, "getaddrinfo", lambda *a, **k: _addrinfo("93.184.216.34", "127.0.0.1")
        )
        assert ssrf.resolve_public_hostname("rebind.example") is False

    def test_rejects_when_dns_fails(self, monkeypatch):
        def boom(*_args, **_kwargs):
            raise socket.gaierror("nope")

        monkeypatch.setattr(socket, "getaddrinfo", boom)
        assert ssrf.resolve_public_hostname("unresolvable.example") is False

    def test_rejects_empty_hostname(self):
        assert ssrf.resolve_public_hostname("") is False
        assert ssrf.resolve_public_hostname("   ") is False

    @pytest.mark.parametrize(
        "hostname", ["localhost", "LOCALHOST", "localhost.localdomain", "svc.localhost"]
    )
    def test_rejects_loopback_names_without_dns(self, hostname, monkeypatch):
        def fail(*_args, **_kwargs):
            raise AssertionError("loopback names must not reach DNS")

        monkeypatch.setattr(socket, "getaddrinfo", fail)
        assert ssrf.resolve_public_hostname(hostname) is False

    def test_trailing_dot_and_case_are_normalized(self, monkeypatch):
        calls = []

        def record(hostname, *args, **kwargs):
            calls.append(hostname)
            return _addrinfo("93.184.216.34")

        monkeypatch.setattr(socket, "getaddrinfo", record)
        assert ssrf.resolve_public_hostname("Example.COM.") is True
        assert calls == ["example.com"]


class TestHostnameCache:
    def test_a_positive_verdict_is_not_cached(self, monkeypatch):
        """A ``True`` must not be reused; a ``False`` is fine to remember.

        This inverts an older test that asserted the opposite — that a public
        hostname resolved once and was served from cache afterwards. That
        behaviour is the widening: a memoized ``True`` tells a later caller that
        an address it has not looked at is one that was already checked, when the
        caller is about to resolve the name *again* to dial it. Under rebinding
        those are two different answers, and only the first was vetted. So the
        second lookup has to actually happen.
        """
        calls = []

        def record(hostname, *args, **kwargs):
            calls.append(hostname)
            return _addrinfo("93.184.216.34")

        monkeypatch.setattr(socket, "getaddrinfo", record)

        assert ssrf.resolve_public_hostname("cached.example") is True
        assert ssrf.resolve_public_hostname("cached.example") is True
        assert calls == ["cached.example", "cached.example"]
        assert ssrf._PUBLIC_HOSTNAME_CACHE == {}

    def test_private_ip_literals_are_never_cached(self):
        assert ssrf.resolve_public_hostname("127.0.0.1") is False
        assert ssrf.resolve_public_hostname("10.0.0.1") is False
        assert ssrf._PUBLIC_HOSTNAME_CACHE == {}

    def test_failed_dns_result_is_cached(self, monkeypatch):
        calls = []

        def boom(hostname, *args, **kwargs):
            calls.append(hostname)
            raise socket.gaierror("nope")

        monkeypatch.setattr(socket, "getaddrinfo", boom)

        assert ssrf.resolve_public_hostname("bad.example") is False
        assert ssrf.resolve_public_hostname("bad.example") is False
        assert calls == ["bad.example"]

    def test_expired_entries_are_recomputed(self, monkeypatch):
        import time

        calls = []

        def record(hostname, *args, **kwargs):
            calls.append(hostname)
            return _addrinfo("93.184.216.34")

        monkeypatch.setattr(socket, "getaddrinfo", record)
        ssrf.resolve_public_hostname("stale.example")

        with ssrf._PUBLIC_HOSTNAME_CACHE_LOCK:
            for key in list(ssrf._PUBLIC_HOSTNAME_CACHE):
                expiry, verdict = ssrf._PUBLIC_HOSTNAME_CACHE[key]
                ssrf._PUBLIC_HOSTNAME_CACHE[key] = (time.time() - 1, verdict)

        assert ssrf.resolve_public_hostname("stale.example") is True
        assert calls == ["stale.example", "stale.example"]

    def test_cache_is_bounded(self, monkeypatch):
        monkeypatch.setattr(config, "PUBLIC_HOSTNAME_CACHE_MAX_SIZE", 3)
        monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: _addrinfo("93.184.216.34"))

        for index in range(6):
            ssrf.resolve_public_hostname(f"host{index}.example")

        assert len(ssrf._PUBLIC_HOSTNAME_CACHE) <= 3


class TestIsPublicHttpUrl:
    def test_rejects_non_http_schemes(self):
        for url in (
            "file:///etc/passwd",
            "ftp://example.com/paper.pdf",
            "gopher://example.com",
            "javascript:alert(1)",
        ):
            assert ssrf.is_public_http_url(url) is False

    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com:99999/paper.pdf",
            "https://example.com:notaport/paper.pdf",
        ],
    )
    def test_rejects_malformed_ports(self, url):
        assert ssrf.is_public_http_url(url) is False

    def test_rejects_private_literals(self):
        assert ssrf.is_public_http_url("http://127.0.0.1/paper.pdf") is False
        assert ssrf.is_public_http_url("http://[::1]/paper.pdf") is False

    def test_rejects_empty_input(self):
        assert ssrf.is_public_http_url("") is False
        assert ssrf.is_public_http_url(None) is False


class TestContentTypeHelpers:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("application/PDF; charset=utf-8", "application/pdf"),
            ("  TEXT/Plain  ", "text/plain"),
            (None, ""),
        ],
    )
    def test_normalize_content_type(self, raw, expected):
        assert normalize_content_type(raw) == expected

    def test_parse_content_length(self):
        class Headers(dict):
            pass

        assert parse_content_length(Headers({"Content-Length": "42"})) == 42
        assert parse_content_length(Headers({"Content-Length": "abc"})) is None
        assert parse_content_length(Headers({"Content-Length": "-5"})) is None
        assert parse_content_length(Headers({})) is None


class TestValidateRemoteContentLength:
    def test_accepts_a_size_under_the_cap(self):
        validate_remote_content_length({"Content-Length": "100"}, max_bytes=1000)

    def test_ignores_a_missing_header(self):
        validate_remote_content_length({}, max_bytes=1000)

    def test_rejects_zero_length(self):
        with pytest.raises(ValueError, match="empty"):
            validate_remote_content_length({"Content-Length": "0"}, max_bytes=1000)

    def test_rejects_oversized_declaration(self):
        with pytest.raises(ValueError, match="too large"):
            validate_remote_content_length({"Content-Length": "1001"}, max_bytes=1000)


class TestValidateRemoteContentType:
    def test_accepts_a_matching_type(self):
        validate_remote_content_type("a.pdf", "application/pdf")
        validate_remote_content_type("a.txt", "text/plain; charset=utf-8")

    def test_accepts_generic_binary_for_any_extension(self):
        validate_remote_content_type("a.pdf", "application/octet-stream")

    def test_allows_any_text_subtype_for_txt(self):
        validate_remote_content_type("a.txt", "text/markdown")

    def test_ignores_an_absent_type(self):
        validate_remote_content_type("a.pdf", None)
        validate_remote_content_type("a.pdf", "")

    @pytest.mark.parametrize(
        "content_type", ["text/html", "application/json", "application/xhtml+xml", "text/xml"]
    )
    def test_rejects_html_and_xml(self, content_type):
        with pytest.raises(ValueError, match="non-document"):
            validate_remote_content_type("a.pdf", content_type)

    def test_rejects_a_mismatched_binary_type(self):
        with pytest.raises(ValueError, match="not compatible"):
            validate_remote_content_type("a.pdf", "image/png")


class TestValidateDocumentFileSignature:
    def test_accepts_real_magic_bytes(self):
        validate_document_file_signature("a.pdf", b"%PDF-1.7\nbody", "test file")
        validate_document_file_signature("a.docx", b"PK\x03\x04body", "test file")
        validate_document_file_signature("a.pptx", b"PK\x03\x04body", "test file")
        validate_document_file_signature("a.txt", b"plain text", "test file")

    def test_rejects_empty_payload(self):
        with pytest.raises(ValueError, match="empty"):
            validate_document_file_signature("a.pdf", b"", "test file")

    def test_rejects_html_disguised_as_pdf(self):
        with pytest.raises(ValueError, match="HTML page"):
            validate_document_file_signature("a.pdf", b"<!doctype html><html></html>", "test file")

    def test_rejects_html_later_in_the_sample(self):
        with pytest.raises(ValueError, match="HTML page"):
            validate_document_file_signature(
                "a.pdf", b"\xef\xbb\xbf" + b"x" * 10 + b"<html>", "test file"
            )

    def test_rejects_a_non_pdf_claiming_pdf(self):
        with pytest.raises(ValueError, match="valid PDF"):
            validate_document_file_signature("a.pdf", b"just text", "test file")

    def test_rejects_a_non_zip_claiming_docx(self):
        with pytest.raises(ValueError, match="valid DOCX"):
            validate_document_file_signature("a.docx", b"%PDF-1.7", "test file")

    def test_rejects_binary_content_claiming_txt(self):
        with pytest.raises(ValueError, match="binary"):
            validate_document_file_signature("a.txt", b"abc\x00def", "test file")


class TestLooksLikeDirectFileUrl:
    @pytest.mark.parametrize("ext", [".pdf", ".txt", ".docx", ".pptx"])
    def test_accepts_supported_extensions(self, ext):
        assert looks_like_direct_file_url(f"https://example.com/a{ext}") is True

    def test_rejects_a_landing_page(self):
        assert looks_like_direct_file_url("https://example.com/abs/1234") is False


class TestBuildImportFilename:
    def test_prefers_the_content_disposition_name(self):
        name = download.build_import_filename(
            title="Ignored",
            source_url="https://example.com/other.pdf",
            content_disposition='attachment; filename="Real Name.pdf"',
            content_type="application/pdf",
        )
        assert name == "Real_Name.pdf"

    def test_decodes_rfc5987_disposition(self):
        name = download.build_import_filename(
            title="Ignored",
            source_url="https://example.com/other.pdf",
            content_disposition="attachment; filename*=UTF-8''%E8%AE%BA%E6%96%87.pdf",
            content_type="application/pdf",
        )
        assert name.endswith(".pdf")
        assert "论文" not in name

    def test_falls_back_to_the_url_path(self):
        name = download.build_import_filename(
            title="Some Paper",
            source_url="https://example.com/download/paper-name.PDF",
            content_disposition=None,
            content_type="application/pdf",
        )
        # The extension is normalized to lower case so downstream allowlist
        # checks stay case-insensitive.
        assert name == "paper-name.pdf"

    def test_falls_back_to_a_title_slug(self):
        name = download.build_import_filename(
            title="My Great Paper",
            source_url="https://example.com/",
            content_disposition=None,
            content_type="text/plain",
        )
        assert name == "My_Great_Paper.txt"

    def test_infers_the_extension_from_the_content_type(self):
        name = download.build_import_filename(
            title="Doc",
            source_url="https://example.com/abs/1",
            content_disposition=None,
            content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        assert name.endswith(".docx")

    def test_defaults_to_pdf_when_nothing_hints(self):
        name = download.build_import_filename(
            title="Doc", source_url="https://example.com/abs/1", content_disposition=None,
            content_type="application/octet-stream",
        )
        assert name.endswith(".pdf")

    def test_result_is_always_an_allowed_extension(self):
        for disposition in (None, 'attachment; filename="x.exe"', "filename*=UTF-8''x.html"):
            name = download.build_import_filename(
                title="T",
                source_url="https://example.com/a.pdf",
                content_disposition=disposition,
                content_type="application/pdf",
            )
            assert os.path.splitext(name)[1].lower() in config.ALLOWED_EXTENSIONS


class TestIterDownloadablePaperUrls:
    def test_yields_pdf_url_first_without_requiring_a_direct_file(self):
        urls = list(
            download.iter_downloadable_paper_urls("https://example.com/a.pdf", "https://example.com/b")
        )
        assert urls == [
            ("https://example.com/a.pdf", False),
            ("https://example.com/b", True),
        ]

    def test_deduplicates_and_skips_blanks(self):
        urls = list(
            download.iter_downloadable_paper_urls("https://example.com/a.pdf", "https://example.com/a.pdf")
        )
        assert urls == [("https://example.com/a.pdf", False)]

    def test_no_candidates_at_all(self):
        assert list(download.iter_downloadable_paper_urls("", "")) == []


class TestStreamRemotePaper:
    def test_returns_metadata_and_the_initial_chunk(self, monkeypatch, public_example_urls, patch_remote_response):
        body = b"%PDF-1.7\n" + b"x" * 5000
        patch_remote_response(FakeResponse(body))

        response, file_name, content_type, initial_chunk = download.stream_remote_paper(
            title="Valid PDF", pdf_url="https://example.com/paper.pdf", url=""
        )

        assert file_name.endswith(".pdf")
        assert content_type == "application/pdf"
        assert initial_chunk == body[:4096]
        assert b"".join(
            download.iter_remote_file_chunks(response, config.MAX_CONTENT_LENGTH, initial_chunk)
        ) == body

    def test_sends_the_app_user_agent(self, monkeypatch, public_example_urls):
        captured = {}

        def fake_urlopen(request, *args, **kwargs):
            captured["user_agent"] = request.get_header("User-agent")
            return FakeResponse(b"%PDF-1.7\nbody")

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

        download.stream_remote_paper(
            title="Valid PDF", pdf_url="https://example.com/paper.pdf", url=""
        )

        assert captured["user_agent"] == config.APP_USER_AGENT

    def test_refuses_a_redirect_to_a_private_host_before_requesting_it(self, monkeypatch):
        """A 30x to a private address must die at the redirect check itself.

        The previous version of this test let urllib follow the redirect and then
        asserted the post-hoc ``geturl()`` re-check caught it. That is the
        vulnerable shape: by the time anything in the module looked, the request
        to ``169.254.169.254`` had already been sent, so the re-check stopped the
        body from being read but not the internal GET from happening.

        So the fake here raises the 30x the way urllib surfaces it once redirect
        following is switched off, and the assertion is on the *number of
        requests*: exactly one means the target was never contacted.
        """
        monkeypatch.setattr(
            download, "is_public_http_url", lambda url: url.startswith("https://example.com")
        )
        seen = []

        def redirect_to_metadata(*_args, **_kwargs):
            seen.append("requested")
            raise redirect_error("http://169.254.169.254/latest/meta-data")

        monkeypatch.setattr("urllib.request.urlopen", redirect_to_metadata)

        with pytest.raises(ValueError, match="redirected to a non-public URL"):
            download.stream_remote_paper(
                title="T", pdf_url="https://example.com/paper.pdf", url=""
            )

        # The load-bearing assertion: the internal endpoint was never dialed.
        assert seen == ["requested"]

    def test_falls_through_to_the_landing_page_url(self, monkeypatch, public_example_urls, patch_remote_response):
        seen = []

        def fake_urlopen(request, *args, **kwargs):
            seen.append(request.full_url)
            if len(seen) == 1:
                return FakeResponse(b"<html>not a file</html>", content_type="text/html")
            return FakeResponse(b"%PDF-1.7\nsecond try")

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

        _response, file_name, _content_type, _chunk = download.stream_remote_paper(
            title="T", pdf_url="https://example.com/a.pdf", url="https://example.com/b.pdf"
        )

        assert len(seen) == 2
        assert file_name.endswith(".pdf")

    def test_rejects_a_landing_page_url_without_a_document_extension(self, monkeypatch, public_example_urls):
        def fail(*_args, **_kwargs):
            raise AssertionError("must not open a non-document landing page")

        monkeypatch.setattr("urllib.request.urlopen", fail)

        with pytest.raises(ValueError, match="direct downloadable file"):
            download.stream_remote_paper(
                title="T", pdf_url="", url="https://example.com/abs/1234"
            )

    def test_raises_when_no_candidate_url_is_supplied(self):
        with pytest.raises(ValueError, match="No downloadable paper link"):
            download.stream_remote_paper(title="T", pdf_url="", url="")

    def test_surfaces_an_http_error_from_the_origin(self, monkeypatch, public_example_urls):
        def raise_http(*_args, **_kwargs):
            raise urllib.error.HTTPError("https://example.com/a.pdf", 404, "Not Found", None, None)

        monkeypatch.setattr("urllib.request.urlopen", raise_http)

        with pytest.raises(urllib.error.HTTPError):
            download.stream_remote_paper(
                title="T", pdf_url="https://example.com/a.pdf", url=""
            )


class TestRedirectsAreValidatedPerHop:
    """The redirect gate, driven against the *real* urllib machinery.

    The mocked tests elsewhere in this file can only prove that the code calls
    ``urlopen`` and inspects what comes back. These two prove the property that
    actually matters: a 30x is inert data the module can vet, rather than a
    request urllib has already sent on its own. If ``redirect_dict`` ever stops
    suppressing urllib's automatic following, the test below fails — and without
    it, the private-redirect guard would pass its unit tests while the SSRF it
    exists to stop went out over the wire anyway.
    """

    @staticmethod
    def _serve(handler_cls):
        server = http.server.HTTPServer(("127.0.0.1", 0), handler_cls)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server, f"http://127.0.0.1:{server.server_address[1]}"

    def test_urllib_does_not_follow_a_redirect_on_its_own(self):
        """The seam the whole guard rests on, asserted directly.

        Without this, the private-redirect test could keep passing against a
        mock while the real client quietly followed the redirect anyway.
        """

        class Redirector(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(302)
                self.send_header("Location", "/target")
                self.end_headers()

            def log_message(self, *_args):
                pass

        server, base = self._serve(Redirector)
        try:
            with pytest.raises(urllib.error.HTTPError) as excinfo:
                urllib.request.urlopen(
                    download._ManualRedirectRequest(base + "/start"), timeout=5
                )
            # Surfaced as an error carrying Location — not silently followed.
            assert excinfo.value.code == 302
            assert excinfo.value.headers.get("Location") == "/target"
        finally:
            server.shutdown()
            server.server_close()

    def test_a_redirect_to_a_private_address_is_never_requested(self, monkeypatch):
        class Redirector(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(302)
                self.send_header("Location", "http://169.254.169.254/latest/meta-data")
                self.end_headers()

            def log_message(self, *_args):
                pass

        server, base = self._serve(Redirector)
        try:
            monkeypatch.setattr(
                download, "is_public_http_url", lambda url: str(url) == base + "/start"
            )
            with pytest.raises(ValueError, match="redirected to a non-public URL"):
                download._open_following_redirects(
                    base + "/start", ssl.create_default_context(), 5
                )
        finally:
            server.shutdown()
            server.server_close()

    def test_a_permitted_redirect_is_still_followed(self, monkeypatch):
        """The guard must not have become a redirect blocker.

        A module that simply refused every 30x would pass the test above. This
        one asserts a public target is actually followed and read.
        """
        body = b"%PDF-1.7\nredirected body"
        seen = []

        class Final(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                seen.append("final")
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args):
                pass

        final_server, final_base = self._serve(Final)
        target = final_base + "/final.pdf"

        class Redirector(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                seen.append("start")
                self.send_response(302)
                self.send_header("Location", target)
                self.end_headers()

            def log_message(self, *_args):
                pass

        start_server, start_base = self._serve(Redirector)
        try:
            monkeypatch.setattr(
                download,
                "is_public_http_url",
                lambda url: str(url) in (start_base + "/start", target),
            )
            response, final_url = download._open_following_redirects(
                start_base + "/start", ssl.create_default_context(), 5
            )
            try:
                assert response.read() == body
            finally:
                response.close()
            assert final_url == target
            assert seen == ["start", "final"]
        finally:
            start_server.shutdown()
            start_server.server_close()
            final_server.shutdown()
            final_server.server_close()

    def test_a_redirect_cycle_terminates(self, monkeypatch):
        class Looper(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(302)
                self.send_header("Location", "/loop")
                self.end_headers()

            def log_message(self, *_args):
                pass

        server, base = self._serve(Looper)
        try:
            monkeypatch.setattr(
                download, "is_public_http_url", lambda url: str(url).startswith(base)
            )
            with pytest.raises(ValueError, match="redirected too many times"):
                download._open_following_redirects(
                    base + "/loop", ssl.create_default_context(), 5
                )
        finally:
            server.shutdown()
            server.server_close()


class TestIterRemoteFileChunks:
    def test_closes_the_response_when_the_stream_ends(self, monkeypatch, public_example_urls, patch_remote_response):
        response = FakeResponse(b"%PDF-1.7\nbody")
        patch_remote_response(response)
        opened, _n, _c, chunk = download.stream_remote_paper(
            title="T", pdf_url="https://example.com/a.pdf", url=""
        )

        list(download.iter_remote_file_chunks(opened, config.MAX_CONTENT_LENGTH, chunk))
        assert opened.closed is True

    def test_closes_the_response_when_the_cap_is_exceeded(self, monkeypatch, public_example_urls, patch_remote_response):
        body = b"%PDF-1.7\n" + b"x" * 5000
        patch_remote_response(FakeResponse(body))
        opened, _n, _c, chunk = download.stream_remote_paper(
            title="T", pdf_url="https://example.com/a.pdf", url=""
        )

        with pytest.raises(ValueError, match="too large"):
            list(download.iter_remote_file_chunks(opened, max_bytes=100, initial_chunk=chunk))

        assert opened.closed is True

    def test_a_zero_byte_response_is_rejected_while_sampling(
        self, public_example_urls, patch_remote_response
    ):
        # An empty body never reaches the chunk loop: the magic-byte sample
        # reads 0 bytes and the signature check refuses it first.
        patch_remote_response(FakeResponse(b"", content_length=0))

        with pytest.raises(ValueError, match="empty"):
            download.stream_remote_paper(
                title="T", pdf_url="https://example.com/a.pdf", url=""
            )


class TestDownloadRemotePaper:
    def test_writes_the_full_body_to_disk(self, runtime_dirs, monkeypatch, public_example_urls, patch_remote_response):
        body = b"%PDF-1.7\n" + b"x" * 5000
        patch_remote_response(FakeResponse(body))

        temp_path, file_name = download.download_remote_paper(
            title="Valid PDF", pdf_url="https://example.com/paper.pdf", url=""
        )
        try:
            assert file_name.endswith(".pdf")
            with open(temp_path, "rb") as handle:
                assert handle.read() == body
        finally:
            from paperwhisperer.core.text import remove_file_safely

            remove_file_safely(temp_path, "test imported paper")

    def test_leaves_no_partial_file_when_the_download_fails(
        self, runtime_dirs, monkeypatch, public_example_urls, patch_remote_response
    ):
        body = b"%PDF-1.7\n" + b"x" * 5000
        patch_remote_response(FakeResponse(body))
        # Shrink the cap below the body size so the chunk loop aborts mid-write.
        # MAX_UPLOAD_BYTES is the setting; MAX_CONTENT_LENGTH is now only an
        # alias of it, so patching the alias would no longer bound anything.
        monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 100)

        before = set(os.listdir(runtime_dirs["uploads"]))

        with pytest.raises(ValueError, match="too large"):
            download.download_remote_paper(
                title="Valid PDF", pdf_url="https://example.com/paper.pdf", url=""
            )

        assert set(os.listdir(runtime_dirs["uploads"])) == before


class TestIsRemoteImportError:
    def test_classifies_urllib_errors(self):
        assert download.is_remote_import_error(urllib.error.HTTPError("u", 500, "e", None, None)) is True
        assert download.is_remote_import_error(urllib.error.URLError("boom")) is True
        assert download.is_remote_import_error(ValueError("nope")) is False
