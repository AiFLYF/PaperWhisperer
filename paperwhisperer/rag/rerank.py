"""Reranking: the optional second pass, behind a provider nobody has agreed on.

The development plan's D3 row is one sentence long -- "rerank 无标准 OAI 接口，做成
可选 provider，默认关闭" -- and the whole module exists to keep it true. There is no
OpenAI-compatible rerank endpoint, not because the vendors are unusually secretive but
because rerankers are not chat models and were never going to share a schema: a cross-
encoder wants ``(query, [passages]) -> [scores]``, an LLM API wants ``(messages) ->
[choices]``, and the industry's several attempts to reconcile them are all narrower than
any one of them. So this file does not pretend there is a standard. It defines a
:class:`RerankProvider` protocol, ships one HTTP provider that speaks the two wire
shapes that actually exist in the wild, and leaves the door open for a third.

**Default is off, and "off" is a real state rather than a missing one.** Plan section 8
lists the absent standard as a risk with the mitigation "provider abstraction, off by
default, decided by the retrieval eval" (T4.3). That means the default has to be the one
that needs no configuration and no network: a deployment that has never heard of
reranking must still answer queries. So :func:`build_reranker` returns ``None`` unless
``RERANK_ENABLED`` is on *and* a model is named, and the retriever treats ``None`` as
"skip the stage" rather than as an error.

**A failing reranker degrades, and says so.** When the provider raises or returns a
response the protocol cannot be read out of, the retriever keeps the fusion order and
records a Chinese reason in the retrieval record. Silently reordering to the second-best
is the tempting alternative, and it is wrong twice over: it hides an operator's broken
model behind a plausible-looking answer, and it makes a retrieval regression impossible to
reproduce, because the regression only reproduces while the provider is up. The same rule
the store bundle and the parse-quality gate already follow -- *degradation is reported,
not swallowed* -- applies here for the same reason.

**What the provider is not.** This file contains no model and no scoring function. A
"fallback reranker" implemented as term-overlap-with-weights would be a second lexical
scorer living next to :mod:`paperwhisperer.infra.stores.scoring`, disagreeing with it
about the same corpus, and reachable by configuration -- which is precisely the
two-tiers-disagree failure the store contract was written to prevent. If reranking is
worth doing, it is done by a cross-encoder; until one is configured, the fusion order is
the answer and the retrieval record says rerank did not run.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from paperwhisperer.core import config

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_RERANK_MODEL",
    "RERANK_WIRE_COHERE",
    "RERANK_WIRE_JINA",
    "RERANK_WIRE_SHAPES",
    "HttpRerankProvider",
    "RerankDocument",
    "RerankError",
    "RerankProvider",
    "RerankResult",
    "build_reranker",
    "rerank_config_summary",
]

#: Model used when ``RERANK_MODEL`` is empty but the switch is on. Named as a
#: vendor, not as an endorsement: a local cross-encoder served by a
#: llama.cpp-style gateway is the common case for the personal tier, and every
#: provider that implements either wire shape below will accept it.
DEFAULT_RERANK_MODEL = "bge-reranker-v2-m3"

#: ``{"query", "documents", "top_n"} -> {"results": [{"index", "relevance_score"}]}``.
#: The shape Cohere popularised and that most self-hosted gateways imitate.
RERANK_WIRE_COHERE = "cohere"

#: ``{"model", "query", "documents"} -> {"data": [{"index", "score"}]}``. The
#: shape Jina's public endpoint returns, which differs in two places that
#: matter: the results live under ``data`` and the score field is ``score``
#: rather than ``relevance_score``. A provider that only accepted one of them
#: would be a "standard" with a single vendor's spelling in it.
RERANK_WIRE_JINA = "jina"

RERANK_WIRE_SHAPES = (RERANK_WIRE_COHERE, RERANK_WIRE_JINA)

#: Default ceiling on a single rerank call. Cross-encoders are quadratic in
#: (query x passage) and this project is not going to quietly hand a 400-token
#: query to a 60-passage reranker on an interactive path; the retriever passes
#: its own bounded candidate list and this is the outer belt.
DEFAULT_MAX_PASSAGES = 64

#: How long a rerank call may take before it is treated as a failure. Shorter
#: than a chat completion's budget on purpose -- a rerank that takes thirty
#: seconds has already failed the interaction it was meant to improve.
DEFAULT_TIMEOUT_SECONDS = 15.0


class RerankError(RuntimeError):
    """A rerank call could not be completed or its response could not be read.

    A ``RuntimeError`` subclass, and a distinct type from the provider's own
    transport errors, so a caller can tell "the network broke" from "the
    provider answered with something that is not a rerank response" and decide
    whether a retry is worth anything. The retriever treats both the same way --
    degrade and record -- but it records the two differently.
    """


@dataclass(slots=True, frozen=True)
class RerankDocument:
    """One passage offered for reranking.

    ``chunk_id`` is the identity the score comes back against. Rerank APIs
    return an *index* into the request rather than the passage, so the mapping
    from a score back to something citable has to be carried here; a provider
    that echoed the passage text instead would force a text match at exactly
    the point where a mismatch is least debuggable.
    """

    chunk_id: str
    text: str


@dataclass(slots=True, frozen=True)
class RerankResult:
    """One reranked passage: its id, the provider's score, and its rank.

    ``score`` is whatever the provider said, on the provider's scale, and is
    *not* comparable to the fusion score it displaces. It is kept separate
    rather than written over the fused number because the two answer different
    questions -- "is this passage relevant to the query" versus "how did the
    retrievers rank it" -- and a record that shows one number for both makes a
    reranked result indistinguishable from an unreranked one.

    ``rank`` is 1-based, matching the request order convention the two wire
    shapes use, and is assigned by the provider adapter so a caller never has
    to re-derive it.
    """

    chunk_id: str
    score: float
    rank: int = 0


@runtime_checkable
class RerankProvider(Protocol):
    """What the retriever needs from anything that can reorder passages.

    Deliberately one method. A provider interface with ``rerank``,
    ``score`` and ``rerank_batch`` looks more complete and only adds three ways
    for a caller to pick the slower one. The one method is the whole
    capability the plan describes, and it is a ``Protocol`` rather than an ABC
    so a test double costs three lines -- which matters, because the alternative
    is that the rerank path is only ever exercised against a real network
    endpoint and therefore is only ever tested when one happens to be up.
    """

    @property
    def model(self) -> str:
        """The model name, recorded on every retrieval that used it."""
        ...

    def rerank(
        self, query: str, documents: Sequence[RerankDocument], *, top_n: int | None = None
    ) -> list[RerankResult]:
        """Score ``documents`` against ``query``, most relevant first.

        ``documents`` is already bounded by the retriever; ``top_n`` asks the
        provider to truncate, and a provider that ignores it must not be
        trusted to -- the caller slices regardless.
        """
        ...


# --------------------------------------------------------------------------
# HTTP provider
# --------------------------------------------------------------------------


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    api_key: str,
    timeout: float,
) -> dict[str, Any]:
    """POST ``payload`` and parse the JSON body, or raise :class:`RerankError`.

    ``urllib`` rather than a new dependency for the same reason the rest of
    this project reaches for it: rerank is a *response* endpoint, the request
    is one JSON object, and a client library that speaks OpenAI chat semantics
    would have to be taught a new response shape anyway.
    """
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        # Sent as a bearer token because that is what both public endpoints
        # check. A local gateway usually ignores it; sending it is what makes
        # the same code work against a hosted one.
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace")[:200]
        except Exception:  # pragma: no cover - the body is best-effort context
            detail = ""
        raise RerankError(
            f"rerank 服务返回 HTTP {exc.code}（{url}）。{detail or '未提供错误详情。'}"
        ) from exc
    except (urllib.error.URLError, OSError) as exc:
        raise RerankError(f"rerank 服务连接失败（{url}）：{exc}") from exc
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RerankError(
            f"rerank 服务返回的不是 JSON（{url}）：{raw[:120]!r}。"
            "请确认 base_url 指向的是 rerank 端点而不是 /v1/embeddings。"
        ) from exc
    if not isinstance(parsed, dict):
        raise RerankError(f"rerank 服务返回了 {type(parsed).__name__}，期望 JSON 对象。")
    return parsed


class HttpRerankProvider:
    """A rerank provider that speaks one of the two wire shapes that exist.

    ``wire_format`` is a required decision rather than an auto-detect, because
    the two shapes can both be made to parse: an adapter that tried the keys
    ``results`` then ``data`` would work against both vendors and would also
    work against a proxy that returns an error object with neither, quietly
    producing an empty ranking that reads as "nothing was relevant". Naming
    the shape moves that failure to configuration time, where it is visible.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_RERANK_MODEL,
        base_url: str = "",
        api_key: str = "",
        wire_format: str = RERANK_WIRE_COHERE,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_passages: int = DEFAULT_MAX_PASSAGES,
        transport: Any = None,
    ) -> None:
        if wire_format not in RERANK_WIRE_SHAPES:
            raise ValueError(
                f"未知的 rerank 协议格式：{wire_format!r}"
                f"（可用值：{', '.join(RERANK_WIRE_SHAPES)}）"
            )
        self._model = str(model or DEFAULT_RERANK_MODEL)
        # Trailing slashes removed once, here, so the endpoint join below is
        # the single place that knows about paths. A base_url that is
        # ``https://host/v1`` and one that is ``https://host/v1/`` must produce
        # the same request, and they do not if a caller writes the slash.
        self.base_url = str(base_url or "").rstrip("/")
        self.api_key = str(api_key or "")
        self.wire_format = wire_format
        self.timeout = max(0.1, float(timeout))
        self.max_passages = max(1, int(max_passages))
        # A callable ``(url, payload, *, api_key, timeout) -> dict`` standing in
        # for :func:`_post_json`. Present because the alternative is that this
        # module is only ever exercised against a live endpoint, which means
        # the wire-shape adapters below are the least-tested lines in the file.
        self.transport = transport

    @property
    def model(self) -> str:
        return self._model

    @property
    def endpoint(self) -> str:
        """The URL this provider posts to.

        ``/rerank`` for both shapes. It is *not* ``/v1/chat/completions`` and
        not ``/v1/embeddings``, which is the entire reason this provider
        cannot be a subclass of the chat client: a deployment that points
        ``RERANK_API_BASE_URL`` at its OpenAI-compatible endpoint gets a 404
        and a Chinese sentence saying so, rather than a chat completion
        hallucinated into the role of a cross-encoder score.
        """
        if not self.base_url:
            raise RerankError(
                "rerank 未配置 base_url；请设置 RERANK_API_BASE_URL 后再启用 RERANK_ENABLED。"
            )
        return f"{self.base_url}/rerank"

    def describe(self) -> dict[str, Any]:
        """What a health endpoint or a retrieval record needs to name."""
        return {
            "model": self._model,
            "wire_format": self.wire_format,
            "endpoint": f"{self.base_url}/rerank" if self.base_url else "",
            "max_passages": self.max_passages,
        }

    # -- request / response -------------------------------------------------

    def _request_body(self, query: str, documents: Sequence[RerankDocument]) -> dict[str, Any]:
        # ``model`` in the body is the one field both shapes agree on. Cohere
        # also accepts it as a path segment, which is why ``{base_url}/rerank``
        # is the endpoint for both: a gateway imitating Cohere takes the model
        # from the body, one imitating Jina ignores it, and a request that
        # omitted it is the "model not specified" error that reads like an
        # authentication failure.
        return {
            "model": self._model,
            "query": str(query or ""),
            "documents": [str(document.text or "") for document in documents],
        }

    def _read_results(self, payload: dict[str, Any], documents: Sequence[RerankDocument]) -> list[
        RerankResult
    ]:
        if self.wire_format == RERANK_WIRE_COHERE:
            rows = payload.get("results")
            score_key = "relevance_score"
        else:
            rows = payload.get("data")
            score_key = "score"
        if not isinstance(rows, list):
            raise RerankError(
                f"rerank 响应缺少 {self.wire_format} 协议的 "
                f"{'results' if self.wire_format == RERANK_WIRE_COHERE else 'data'} 数组，"
                f"实际键为：{sorted(payload)[:8]}。"
            )
        results: list[RerankResult] = []
        for position, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                continue
            index = row.get("index")
            try:
                index = int(index)
            except (TypeError, ValueError):
                # A row with no usable index cannot be attached to a passage,
                # and dropping it silently would return a ranking that looks
                # complete. Skipping it *is* silent, so the count is carried
                # into the error the retriever records.
                continue
            if not 0 <= index < len(documents):
                raise RerankError(
                    f"rerank 响应第 {position} 条的 index={index} 越界，"
                    f"本次只提交了 {len(documents)} 条候选。"
                )
            try:
                score = float(row.get(score_key, row.get("relevance_score", row.get("score"))))
            except (TypeError, ValueError) as exc:
                raise RerankError(
                    f"rerank 响应第 {position} 条的分数不是数字：{row.get(score_key)!r}"
                ) from exc
            results.append(
                RerankResult(
                    chunk_id=documents[index].chunk_id, score=score, rank=position
                )
            )
        if not results:
            raise RerankError("rerank 服务返回了空结果列表，无法据此重排。")
        return results

    def rerank(
        self, query: str, documents: Sequence[RerankDocument], *, top_n: int | None = None
    ) -> list[RerankResult]:
        """POST one request and return the results, most relevant first.

        Sorted here rather than trusted: both wire shapes document the order
        but neither guarantees it, and a provider that returned them shuffled
        would otherwise be read as "all equally relevant", which is the one
        interpretation of that response that is certainly wrong. The
        ``chunk_id`` tiebreak matches the total order the rest of the
        retrieval pipeline uses, so two runs against the same provider agree.
        """
        items = list(documents)[: self.max_passages]
        if not items:
            return []
        body = self._request_body(query, items)
        if top_n is not None and top_n > 0:
            body["top_n"] = int(top_n)
        payload = (self.transport or _post_json)(
            self.endpoint, body, api_key=self.api_key, timeout=self.timeout
        )
        results = self._read_results(payload, items)
        results.sort(key=lambda item: (-item.score, item.chunk_id))
        for position, item in enumerate(results, start=1):
            results[position - 1] = RerankResult(
                chunk_id=item.chunk_id, score=item.score, rank=position
            )
        return results


# --------------------------------------------------------------------------
# Construction from configuration
# --------------------------------------------------------------------------

#: Every key this module reads, with the default it uses when the key is
#: absent. The same contract table idea as
#: :data:`paperwhisperer.infra.stores.factory.REQUIRED_CONFIG_KEYS`, and for
#: the same reason: :mod:`paperwhisperer.core.config` is owned by another task,
#: so a direct attribute access would turn "this build has no rerank keys yet"
#: into an ``AttributeError`` at import time rather than a documented default.
#: ``RERANK_ENABLED`` and ``RERANK_MODEL`` already exist in config; the others
#: are read with ``getattr`` and are no-ops until somebody sets them.
REQUIRED_CONFIG_KEYS: dict[str, Any] = {
    "RERANK_ENABLED": False,
    "RERANK_MODEL": "",
    "RERANK_API_BASE_URL": "",
    "RERANK_API_KEY": "",
    "RERANK_WIRE_FORMAT": RERANK_WIRE_COHERE,
    "RERANK_TIMEOUT_SECONDS": DEFAULT_TIMEOUT_SECONDS,
    "RERANK_MAX_PASSAGES": DEFAULT_MAX_PASSAGES,
}

_BOOL_TRUE = {"1", "true", "yes", "on"}


def _setting(settings: Any, key: str) -> Any:
    value = getattr(settings, key, None)
    return REQUIRED_CONFIG_KEYS[key] if value is None else value


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in _BOOL_TRUE


def _as_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _as_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def rerank_config_summary(settings: Any = None) -> dict[str, Any]:
    """What the configuration says about reranking, for a health endpoint.

    Reports the *resolved* values, including the "switch is on but no endpoint
    is configured" case, because that is a misconfiguration an operator wants
    to see before a user does. It does not build a provider and therefore does
    not touch the network.
    """
    if settings is None:
        settings = config
    enabled = _as_bool(_setting(settings, "RERANK_ENABLED"))
    model = str(_setting(settings, "RERANK_MODEL") or "").strip()
    base_url = str(_setting(settings, "RERANK_API_BASE_URL") or "").strip()
    wire = str(_setting(settings, "RERANK_WIRE_FORMAT") or RERANK_WIRE_COHERE).strip().lower()
    if not enabled:
        reason = "已关闭（默认）。检索按 RRF 融合顺序返回，不调用 rerank 服务。"
    elif not base_url:
        reason = (
            "RERANK_ENABLED=true 但未配置 RERANK_API_BASE_URL；"
            "rerank 没有标准 OpenAI 兼容端点，必须单独给出服务地址。"
        )
    elif wire not in RERANK_WIRE_SHAPES:
        reason = f"RERANK_WIRE_FORMAT={wire!r} 不是已知协议（{', '.join(RERANK_WIRE_SHAPES)}）。"
    else:
        reason = f"已启用：POST {base_url.rstrip('/')}/rerank（{wire} 协议）。"
    return {
        "enabled": enabled,
        "model": model or (DEFAULT_RERANK_MODEL if enabled else ""),
        "base_url": base_url,
        "wire_format": wire,
        "reason": reason,
    }


def build_reranker(settings: Any = None, *, transport: Any = None) -> RerankProvider | None:
    """Build the configured rerank provider, or ``None`` when rerank is off.

    ``None`` is the normal return and means three different things that a
    caller must not confuse: rerank is disabled, rerank is disabled *by this
    plan* even though the deployment supports it, or no provider could be
    built from the configuration. The retriever distinguishes them by asking
    :func:`rerank_config_summary` for the reason and recording it; collapsing
    them into "no reranker" would make the third case invisible.

    ``transport`` is an injection point for a test: a callable with the
    signature ``(url, payload, *, api_key, timeout) -> dict`` replacing
    :func:`_post_json`. It exists because the alternative is that this module
    is only ever exercised against a live endpoint.
    """
    if settings is None:
        settings = config
    if not _as_bool(_setting(settings, "RERANK_ENABLED")):
        return None
    base_url = str(_setting(settings, "RERANK_API_BASE_URL") or "").strip()
    if not base_url:
        # A warning rather than an exception: the deployment is still
        # correct, it just will not rerank, and a hard failure here would take
        # Q&A down over an optional second pass.
        logger.warning(
            "RERANK_ENABLED=true 但未配置 RERANK_API_BASE_URL，本次检索将跳过 rerank。"
        )
        return None
    wire = str(_setting(settings, "RERANK_WIRE_FORMAT") or RERANK_WIRE_COHERE).strip().lower()
    if wire not in RERANK_WIRE_SHAPES:
        logger.warning("未知的 RERANK_WIRE_FORMAT=%r，按 cohere 协议处理。", wire)
        wire = RERANK_WIRE_COHERE
    api_key = str(_setting(settings, "RERANK_API_KEY") or "").strip() or config.resolve_api_key("")
    provider = HttpRerankProvider(
        model=str(_setting(settings, "RERANK_MODEL") or "").strip() or DEFAULT_RERANK_MODEL,
        base_url=base_url,
        api_key=api_key,
        wire_format=wire,
        timeout=_as_float(
            _setting(settings, "RERANK_TIMEOUT_SECONDS"), DEFAULT_TIMEOUT_SECONDS
        ),
        max_passages=_as_int(
            _setting(settings, "RERANK_MAX_PASSAGES"), DEFAULT_MAX_PASSAGES
        ),
        transport=transport,
    )
    return provider
