"""Test doubles shared across the suite, and the live-model failure taxonomy.

Kept in a real module (rather than in ``conftest.py``) so test files can import
them directly instead of relying on fixture injection for plain data classes.

The second half of this module -- :class:`LiveFailureKind` and
:func:`classify_live_failure` -- is the *shared* taxonomy for tests that talk to
a real LLM endpoint. It lives here rather than in one of the two live test files
because both need it, and a fix applied to only one of them guarantees the two
drift apart. That is not hypothetical: the two files shipped with different
ideas of what "the endpoint is unavailable" meant, so one outage skipped three
tests in one file and hard-failed a fourth in the other. One list, one
classifier, imported by both.

**The three kinds, and what each is allowed to do.**

``ENDPOINT``
    The endpoint would not serve us: a timeout, a 429, a connection failure, a
    gateway 5xx. Says nothing about whether our prompts, our parsing or our
    handling of the model's output are correct. *Skips*, with a message naming
    the endpoint. Retrying is not on the table -- the endpoint is shared, so
    hammering it turns our inconvenience into somebody else's.

``MODEL``
    The endpoint served us and the model is the difficulty: it answered a
    no-evidence question by reaching for a tool this product does not offer, or
    produced nothing usable. *Retried, a bounded number of times*, because a
    sampled model is not a function of its prompt -- the same question gets a
    good answer on the next draw. Exhausting the attempts is still a failure.

``PRODUCT``
    Everything else: a 400 for a request we built, an empty section, a mermaid
    payload still carrying its code fence, a fabricated answer. *Fails
    immediately, and is never retried.* That last clause is the one that
    matters. A retry which covered ``PRODUCT`` would convert a regression into a
    pass, which is worse than a flaky test, because it removes the signal
    permanently rather than intermittently.
"""

from __future__ import annotations

import json
from email.message import Message
from enum import Enum


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


# ---------------------------------------------------------------------------
# The live-model failure taxonomy
# ---------------------------------------------------------------------------


class LiveFailureKind(str, Enum):
    """Whose fault a failed call against a real endpoint is.

    A ``str`` enum so a kind can go straight into a skip reason without anyone
    having to remember to call ``.value``.
    """

    ENDPOINT = "endpoint"
    MODEL = "model"
    PRODUCT = "product"


#: The endpoint would not serve us. Substrings of the message, lowercased, so a
#: Chinese message matches on its Chinese markers and an English one on its own.
#:
#: Mirrors ``paperwhisperer.core.errors.is_retryable_llm_error`` (the product's
#: own view of the same boundary) and adds the strings only a client can
#: produce: the SDK's connection-failure text, and the English phrasings the
#: gateway emits when a shared quota is exhausted. Both live test files used to
#: carry their own partial copy of this list; measured against a live endpoint,
#: one run failed a test on a timeout that the other three tests had skipped
#: correctly, so the divergence had a price.
UNAVAILABLE_MARKERS = (
    # product classifications
    "超时",
    "timeout",
    "连接失败",
    "connection",
    "429",
    "限流",
    "502",
    "503",
    "504",
    "网关",
    # SDK / gateway English phrasings for a shared, exhausted quota
    "rate limit",
    "too many requests",
    "overloaded",
    "service unavailable",
    "bad gateway",
)

#: The model reached for a tool this product does not offer, and so returned no
#: content. Matched on a stable prefix of ``llm.output.TOOL_CALL_MESSAGE`` so the
#: wording can change without breaking the classification; the full message is
#: the product's to word.
MODEL_TOOL_CALL_MARKERS = (
    "模型尝试调用本产品不提供的工具",
    "tool_calls",
)

#: The model served us an envelope with nothing in it. Checked after the tool-call
#: markers, and *only* for the streaming shape: in ``client.py`` a tool-calling
#: stream is indistinguishable from an empty one, because ``_parse_stream_delta``
#: reads ``delta.content`` and a tool call has none. On the non-streaming path an
#: empty completion is a real product question, so it is not claimed here.
MODEL_EMPTY_MARKERS = ("AI 服务返回内容为空",)


def _lower(markers: tuple[str, ...]) -> tuple[str, ...]:
    """Markers lowercased, so a mixed-case constant still matches lowered text.

    The constants are written the way the product spells them ("AI 服务返回内容
    为空", "Rate limit"), because that is how a reader greps for them in
    ``client.py``. Comparing them against a lowercased message without this
    would silently never match anything with a capital in it -- which is exactly
    the sort of bug that makes a classifier quietly classify everything as
    PRODUCT and reports a shared endpoint's outage as a broken product.
    """
    return tuple(marker.lower() for marker in markers)


def classify_live_failure(message: str, *, streaming: bool = False) -> LiveFailureKind:
    """Whose fault is this failure: the endpoint's, the model's, or the product's.

    ``streaming`` widens the ``MODEL`` verdict to cover "the model returned an
    empty body" as well as "the model asked for a tool". The streaming path
    cannot tell those two apart -- a ``finish_reason="tool_calls"`` frame yields
    no ``delta.content`` -- so a model that reaches for a tool mid-stream
    surfaces as an empty completion. Classifying that as ``PRODUCT`` would
    report a broken product for a model's choice, which is the specific mistake
    this taxonomy exists to stop.

    Order matters and is not arbitrary. ``ENDPOINT`` is checked first because a
    timeout and a model quirk are indistinguishable once both have been flattened
    to a string, and the endpoint is the cheaper mistake to make: misfiling a
    model quirk as an outage costs a skip, misfiling an outage as a product bug
    costs the credibility of every failure this suite has ever reported.
    """
    text = str(message or "").lower()
    if any(marker in text for marker in _lower(UNAVAILABLE_MARKERS)):
        return LiveFailureKind.ENDPOINT
    if any(marker in text for marker in _lower(MODEL_TOOL_CALL_MARKERS)):
        return LiveFailureKind.MODEL
    if streaming and any(marker in text for marker in _lower(MODEL_EMPTY_MARKERS)):
        return LiveFailureKind.MODEL
    return LiveFailureKind.PRODUCT

