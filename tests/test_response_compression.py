"""Response compression, and the one response shape it must not touch.

The middleware exists for one measured reason: the page's own assets are 408 KB
uncompressed and 124 KB gzipped. It exists *safely* for a second reason, and
that is what these tests pin -- a partial response carries
``Content-Range: bytes 0-5000/6793``, a claim about raw file offsets. Compress
the body and the claim stays while the bytes stop matching it, so a resumed
download or a seeking media element gets a corrupt file. The guard is on the
request (a client that sends ``Range`` gets bytes verbatim) because that is
where the answer is already known.

The SSE case is pinned too. Starlette excludes ``text/event-stream`` from
compression on its own, but every streaming route in this app is SSE, so a
future change of middleware or of Starlette's exclusion list would break the
one property those routes exist for. A test that says so turns that into a red
build instead of a stream that quietly arrives in one lump.
"""

from __future__ import annotations

import gzip

from paperwhisperer.app import RangeSafeGZipMiddleware
from paperwhisperer.core import config

STATIC_JS = config.STATIC_DIR / "js" / "main.js"


def _raw_wire_bytes(client, url: str, **headers) -> tuple[bytes, object]:
    """The bytes as sent, before httpx decodes any content-encoding.

    ``response.content`` cannot answer "was this compressed": httpx decodes it
    and strips ``Content-Encoding``, so the header a plain request sees is the
    same either way. ``iter_raw`` is the only place the wire bytes are visible.
    """
    with client.stream("GET", url, headers=headers or {"Accept-Encoding": "gzip"}) as response:
        raw = b"".join(response.iter_raw())
        return raw, response


class TestCompressionIsOn:
    def test_a_static_asset_is_served_gzipped(self, client):
        body = STATIC_JS.read_bytes()
        raw, response = _raw_wire_bytes(client, "/static/js/main.js")

        assert response.status_code == 200
        assert raw[:2] == b"\x1f\x8b"  # gzip magic
        assert gzip.decompress(raw) == body
        assert len(raw) < len(body), "compressed body was not smaller"

    def test_the_browser_is_told_the_response_varies_by_encoding(self, client):
        # Without Vary, a cache can hand the gzipped body to a client that never
        # asked for one -- the classic way a compression change breaks only the
        # users behind a CDN.
        response = client.get("/static/js/main.js", headers={"Accept-Encoding": "gzip"})
        assert "accept-encoding" in response.headers.get("vary", "").lower()

    def test_a_client_that_asks_for_nothing_gets_nothing_compressed(self, client):
        body = STATIC_JS.read_bytes()
        raw, _ = _raw_wire_bytes(client, "/static/js/main.js", **{"Accept-Encoding": "identity"})

        assert raw[:2] != b"\x1f\x8b"
        assert raw == body


class TestARangeRequestIsLeftAlone:
    """The defect these exist for, measured before the guard landed:

        HTTP/1.1 206 Partial Content
        content-range: bytes 0-5000/6793
        content-encoding: gzip
    """

    def test_a_large_range_is_not_compressed(self, client):
        body = STATIC_JS.read_bytes()
        response = client.get(
            "/static/js/main.js",
            headers={"Range": "bytes=0-5000", "Accept-Encoding": "gzip"},
        )

        assert response.status_code == 206
        assert response.headers["content-range"] == f"bytes 0-5000/{len(body)}"
        # Starlette's gzip path deletes Content-Length and adds Content-Encoding.
        # So an intact Content-Length is the observable proof that the body went
        # out verbatim -- and it is exactly what was missing in the response
        # quoted above.
        assert response.headers["content-length"] == "5001"
        assert response.content == body[:5001]

    def test_a_small_range_is_not_compressed_either(self, client):
        # Not because of the guard -- 100 bytes slips under minimum_size -- but
        # the outcome a client sees must not depend on which of the two reasons
        # spared it, so both are pinned.
        body = STATIC_JS.read_bytes()
        response = client.get(
            "/static/js/main.js",
            headers={"Range": "bytes=0-99", "Accept-Encoding": "gzip"},
        )

        assert response.status_code == 206
        assert response.headers["content-length"] == "100"
        assert response.content == body[:100]

    def test_a_suffixed_range_is_not_compressed(self, client):
        # `bytes=-2000` is the other spelling a resuming client sends, and it is
        # a different branch of StaticFiles' range handling.
        body = STATIC_JS.read_bytes()
        response = client.get(
            "/static/js/main.js",
            headers={"Range": "bytes=-2000", "Accept-Encoding": "gzip"},
        )

        assert response.status_code == 206
        assert response.headers["content-length"] == "2000"
        assert response.content == body[-2000:]


class TestTheEventStreamIsNeverCompressed:
    """A property this app depends on, pinned against the middleware itself.

    Exercised with a stand-in ASGI app rather than a real streaming route,
    because the point is the middleware's behaviour on a content type, not any
    one route's payload -- and a route here would need a stubbed model to say
    the same thing.
    """

    @staticmethod
    def _app_returning(content_type: str, body: bytes):
        async def app(scope, receive, send):
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", content_type.encode())],
            })
            await send({"type": "http.response.body", "body": body})

        return app

    def test_an_event_stream_passes_through_unencoded(self):
        import asyncio

        middleware = RangeSafeGZipMiddleware(
            self._app_returning("text/event-stream; charset=utf-8", b"data: x\n\n" * 100)
        )
        sent: list[dict] = []

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            sent.append(message)

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/api/analyze/stream",
            "headers": [(b"accept-encoding", b"gzip")],
        }
        asyncio.run(middleware(scope, receive, send))

        start = next(m for m in sent if m["type"] == "http.response.start")
        header_names = {name.decode().lower() for name, _ in start["headers"]}
        assert "content-encoding" not in header_names
        payload = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
        assert payload == b"data: x\n\n" * 100
