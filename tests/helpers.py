"""Test doubles shared across the suite.

Kept in a real module (rather than in ``conftest.py``) so test files can import
them directly instead of relying on fixture injection for plain data classes.
"""

from __future__ import annotations

import json
from email.message import Message


class FakeResponse:
    """Minimal stand-in for the object ``urllib.request.urlopen`` returns.

    Implements only what ``remote.download`` actually touches: incremental
    ``read``, ``geturl`` for redirect checks, ``headers`` and ``close``.
    """

    def __init__(
        self,
        body: bytes,
        content_type: str | None = "application/pdf",
        content_length: int | None = None,
        url: str = "https://example.com/paper.pdf",
        headers: dict | None = None,
    ):
        self.body = body
        self.offset = 0
        self.headers = Message()
        for key, value in (headers or {}).items():
            self.headers[key] = value
        if content_type is not None:
            self.headers["Content-Type"] = content_type
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)
        self.url = url
        self.closed = False

    def geturl(self) -> str:
        return self.url

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = len(self.body) - self.offset
        start = self.offset
        end = min(len(self.body), start + size)
        self.offset = end
        return self.body[start:end]

    def close(self) -> None:
        self.closed = True


class FakeUploadFile:
    """Async ``read()``-only stand-in for ``fastapi.UploadFile``."""

    def __init__(self, body: bytes):
        self.body = body
        self.offset = 0
        self.closed = False

    async def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = len(self.body) - self.offset
        start = self.offset
        end = min(len(self.body), start + size)
        self.offset = end
        return self.body[start:end]

    async def close(self) -> None:
        self.closed = True


class FakeStreamingResponse:
    """Context manager yielding SSE-shaped byte lines to ``LLMClient.stream``."""

    def __init__(self, lines, status_code: int = 200):
        self._lines = list(lines)
        self.status_code = status_code

    def __enter__(self):
        return self

    def __exit__(self, *_exc_info):
        return False

    def iter_lines(self):
        return iter(self._lines)


def sse_data(text: str) -> bytes:
    """Build one ``data:`` line the way the OpenAI streaming API does."""
    return b"data: " + json.dumps({"choices": [{"delta": {"content": text}}]}).encode("utf-8")


class _ScriptedEndpoint:
    """Replays a fixed list of behaviours, one per call, repeating the last.

    An entry that is an exception instance is raised instead of returned, so a
    test can express "fail twice, then succeed" as a plain list.
    """

    def __init__(self, behaviours):
        self.behaviours = list(behaviours)
        self.calls = 0

    def create(self, **_kwargs):
        self.calls += 1
        behaviour = self.behaviours[min(self.calls - 1, len(self.behaviours) - 1)]
        if isinstance(behaviour, BaseException):
            raise behaviour
        return behaviour


class FakeOpenAI:
    """Stand-in for the ``openai.OpenAI`` client.

    Mirrors the attribute path the transport actually walks:
    ``client.chat.completions.with_raw_response.create`` and
    ``client.chat.completions.with_streaming_response.create``.
    """

    def __init__(self, raw_behaviours=None, stream_behaviours=None):
        completions = type("_Completions", (), {})()
        completions.with_raw_response = _ScriptedEndpoint(raw_behaviours or [])
        completions.with_streaming_response = _ScriptedEndpoint(stream_behaviours or [])
        chat = type("_Chat", (), {})()
        chat.completions = completions
        self.chat = chat

    @property
    def raw_calls(self) -> int:
        return self.chat.completions.with_raw_response.calls

    @property
    def stream_calls(self) -> int:
        return self.chat.completions.with_streaming_response.calls
