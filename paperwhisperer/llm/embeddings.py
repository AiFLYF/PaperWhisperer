"""OpenAI-compatible embedding transport (T1.3).

The sibling of :mod:`paperwhisperer.llm.client`: same transport, same error
classification, same retry vocabulary -- pointed at ``/embeddings`` instead of
``/chat/completions``. Every provider decision (which base_url, which model,
how wide a vector is) is read from :mod:`paperwhisperer.core.config`, so
switching vendor is an environment edit and not a code edit.

Three decisions are worth arguing for, because the obvious alternative is worse
in each case.

**Dimension is part of a vector's identity, not a property of the call.** A
1536-float vector and a 768-float vector are not the same kind of object, and a
cosine similarity between them is not a small wrong number -- it is a
meaningless one, produced without complaint. Two mechanisms enforce that here.
:class:`EmbeddingGateway` refuses any response whose vector length disagrees
with the configured dimension (:class:`EmbeddingDimensionMismatchError`), and
:attr:`EmbeddingGateway.index_version` encodes provider + model + dimension, so
an index built by one setting is recognisably not the index another setting
queries. :meth:`EmbeddingGateway.assert_same_space` turns that recognition into
a refusal at the point where the two would be mixed. This is the section 8 risk
"each vendor ships its own dimension and model" written as a code requirement:
rebuild the index, never blend the two.

**The cache key is ``(provider, model, content_hash)``, not ``(model,
content_hash)``.** The narrower key is what the plan asks for and it is a strict
subset of this one. The two extra coordinates exist because model names are not
globally unique: ``bge-m3`` served by a local gateway and by a cloud vendor may
have different dimensions, and a cache keyed only on the name would hand one
provider's vectors to the other. Hashing the *actual* text (never a chunk id, a
document id or a position) is what makes a re-embedded passage free, which is
the whole point: the same sentence appearing in the body and in the appendix
must cost one call, not two.

**Retries are classified, not blanket.** A 400 means the request is malformed
and replaying it identically will fail identically while spending the caller's
quota and hiding the bug behind a timeout-shaped error; 401/403/404 mean the
configuration is wrong, which is exactly the thing an operator needs to see
verbatim. Only 408/409/425/429/5xx, connection failures and timeouts are
replayed. Note this is deliberately *not* the message-sniffing
:func:`paperwhisperer.core.errors.is_retryable_llm_error` that the older
call sites use: that helper looks for substrings such as "网关" and misses
HTTP 500 ("AI 服务提供商内部错误"), so routing retries through it would
classify the most common server-side blip as permanent.
:class:`paperwhisperer.llm.client.LLMClient` itself retries everything, which is
tolerable for an interactive completion and wrong for a batch of a thousand
chunks.

Cost is measured, not stored. The plan's ``analysis_runs`` table does not exist
until T2.2, so this module deliberately does **not** invent a half-table and
does **not** open a connection. :meth:`EmbeddingUsage.as_analysis_run_columns`
returns exactly the columns T2.2 should add to that table, and the docstring
there names the write site. A caller that has no table yet can still log the
usage object or assert on it in tests, which is why the numbers are carried on
the result rather than pushed somewhere.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from paperwhisperer.core import config
from paperwhisperer.core.errors import describe_llm_status_code
from paperwhisperer.core.identity import provider_identity as _provider_identity
from paperwhisperer.core.text import get_retry_delay_seconds

logger = logging.getLogger(__name__)

#: Status codes worth replaying: transient server-side or rate-limit conditions.
#: Everything else -- 400/401/403/404/413/415/422 -- is a request or
#: configuration fault, and replaying it burns quota to produce the same error.
RETRYABLE_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

#: Fallbacks for the knobs config does not define yet. Read through
#: :func:`_config_int` / :func:`_config_float`, so adding the real keys to
#: config.py changes behaviour without touching this file.
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_CONCURRENCY = 4
DEFAULT_MAX_RETRIES = 3
DEFAULT_REQUEST_TIMEOUT_SECONDS = 60
DEFAULT_MAX_CACHE_ENTRIES = 4096


def _config_int(name: str, default: int, minimum: int, maximum: int) -> int:
    """Read an int setting that config may not define yet.

    ``getattr`` rather than a direct attribute access: T0.8 owns config.py and
    these keys are not in it yet, so a hard reference would turn a missing knob
    into an ImportError at startup instead of a documented default.
    """
    try:
        return max(minimum, min(maximum, int(getattr(config, name, default))))
    except (TypeError, ValueError):
        return default


def _config_float(name: str) -> float | None:
    """Read an optional price; ``None`` means "no price configured, no cost"."""
    raw = getattr(config, name, None)
    if raw is None or raw == "":
        return None
    try:
        price = float(raw)
    except (TypeError, ValueError):
        logger.warning("%s=%r is not a number; embedding cost will stay unreported.", name, raw)
        return None
    return price if price >= 0 else None


def content_hash(text: str) -> str:
    """SHA-256 over the exact text that will be sent to the provider.

    Hashing the payload rather than an identifier is deliberate: a chunk that
    moved position, or the same sentence quoted twice in a paper, is the same
    work for the provider and must be the same cache entry.
    """
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def provider_identity(base_url: str) -> str:
    """A short, stable name for the vendor behind ``base_url``.

    Kept importable from here because this is where callers knew to look for it,
    but the implementation moved to :mod:`paperwhisperer.core.identity`: the
    name is now stamped into ``index_version`` by ``core/config.py`` as well, and
    config sits at the bottom of the tree -- it cannot import this module. Two
    copies of the rule is how the index and the gateway came to disagree in the
    first place, so there is exactly one, one layer lower.
    """
    return _provider_identity(base_url)


def _slug(value: str) -> str:
    """Reduce a model or provider name to characters safe in a stored key."""
    return re.sub(r"[^a-z0-9]+", "-", str(value or "").strip().lower()).strip("-") or "unset"


class EmbeddingDimensionMismatchError(ValueError):
    """A vector's length disagrees with the configured dimension.

    A :class:`ValueError` subclass on purpose: this is a programming or
    configuration fault rather than a transport failure, so the retry loop
    treats it as permanent and surfaces it immediately instead of replaying the
    batch three times.
    """


class EmbeddingCache:
    """Bounded LRU keyed on ``(provider, model, content_hash)``.

    Bounded because the alternative is a leak with a paper-sized corpus: a
    single 400-page document chunks into thousands of distinct 512-token
    vectors, and a full personal library into hundreds of thousands. Eviction
    is least-recently-used, so the passages a retrieval pass keeps hitting stay
    resident while one-off indexing work ages out.
    """

    def __init__(self, max_entries: int = DEFAULT_MAX_CACHE_ENTRIES):
        self.max_entries = max(1, int(max_entries))
        self._entries: OrderedDict[tuple[str, str, str], tuple[float, ...]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: tuple[str, str, str]) -> tuple[float, ...] | None:
        with self._lock:
            vector = self._entries.get(key)
            if vector is None:
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return vector

    def put(self, key: tuple[str, str, str], vector) -> None:
        stored = tuple(float(value) for value in vector)
        with self._lock:
            self._entries[key] = stored
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self.hits = 0
            self.misses = 0

    def stats(self) -> dict:
        with self._lock:
            return {
                "entries": len(self._entries),
                "max_entries": self.max_entries,
                "hits": self.hits,
                "misses": self.misses,
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __contains__(self, key: tuple[str, str, str]) -> bool:
        with self._lock:
            return key in self._entries


@dataclass(frozen=True)
class EmbeddingUsage:
    """Everything a cost ledger needs to know about one embedding operation.

    Immutable so it can be handed to a caller, logged, or stored without the
    gateway's ongoing accounting mutating it afterwards.
    """

    provider: str
    model: str
    dimension: int
    index_version: str
    requests: int = 0
    texts: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    #: Embeddings have no completion side, so this is structurally 0. It is kept
    #: as a column rather than omitted so the cost schema matches the LLM one.
    output_tokens: int = 0
    #: False when the provider reported no ``usage`` block. Cost is then
    #: unknown rather than zero, and ``cost`` returns ``None``.
    tokens_reported: bool = True
    price_per_mtok: float | None = None

    @property
    def cost(self) -> float | None:
        """Cost in the account's currency, or ``None`` if it cannot be known.

        A cache hit has no tokens and so costs exactly ``0.0`` -- a real number,
        not ``None``. ``None`` is reserved for "we could not tell", which is a
        materially different statement.
        """
        if self.price_per_mtok is None or not self.tokens_reported:
            return None
        return self.input_tokens * self.price_per_mtok / 1_000_000

    def merge(self, other: EmbeddingUsage) -> EmbeddingUsage:
        """Accumulate another operation into this one (counters add up)."""
        if not isinstance(other, EmbeddingUsage):
            return self
        return EmbeddingUsage(
            provider=self.provider or other.provider,
            model=self.model or other.model,
            dimension=self.dimension or other.dimension,
            index_version=self.index_version or other.index_version,
            requests=self.requests + other.requests,
            texts=self.texts + other.texts,
            cache_hits=self.cache_hits + other.cache_hits,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            tokens_reported=self.tokens_reported and other.tokens_reported,
            price_per_mtok=(
                other.price_per_mtok if other.price_per_mtok is not None else self.price_per_mtok
            ),
        )

    def as_analysis_run_columns(self) -> dict:
        """Columns T2.2 must persist on ``analysis_runs`` for an embedding run.

        The table does not exist yet. When it does, one row per analysis run
        gets these columns, written by the analysis DAG (T1.8) from the usage
        this gateway reports:

        ==================================  =========================================
        column                              source
        ==================================  =========================================
        ``embedding_provider``              :attr:`provider`
        ``embedding_model``                 :attr:`model`
        ``embedding_dimension``             :attr:`dimension`
        ``embedding_index_version``         :attr:`index_version`
        ``embedding_requests``              :attr:`requests`
        ``embedding_texts``                 :attr:`texts`
        ``embedding_cache_hits``            :attr:`cache_hits`
        ``embedding_input_tokens``          :attr:`input_tokens`
        ``embedding_output_tokens``         :attr:`output_tokens`
        ``embedding_cost``                  :attr:`cost`
        ==================================  =========================================

        These are additive columns on ``analysis_runs`` (the same rows T1.8
        already records ``model``/``prompt_version``/``evidence_version``/``cost``
        for), not a new table. The ``cache_hits`` column matters for reading the
        others: a run that spent nothing because everything was cached and a run
        that spent nothing because the index was empty are indistinguishable in
        ``embedding_cost`` alone.
        """
        return {
            "embedding_provider": self.provider,
            "embedding_model": self.model,
            "embedding_dimension": self.dimension,
            "embedding_index_version": self.index_version,
            "embedding_requests": self.requests,
            "embedding_texts": self.texts,
            "embedding_cache_hits": self.cache_hits,
            "embedding_input_tokens": self.input_tokens,
            "embedding_output_tokens": self.output_tokens,
            "embedding_cost": self.cost,
        }


@dataclass(frozen=True)
class EmbeddingBatch:
    """Vectors plus the cost of producing them, aligned to the input order."""

    vectors: tuple[tuple[float, ...], ...]
    usage: EmbeddingUsage

    @property
    def cached(self) -> int:
        """How many texts were served from cache with zero provider calls."""
        return self.usage.cache_hits

    def __len__(self) -> int:
        return len(self.vectors)

    def __iter__(self):
        return iter(self.vectors)

    def __getitem__(self, index):
        return self.vectors[index]


class EmbeddingGateway:
    """Provider-agnostic embedding interface.

    Business code calls :meth:`embed_documents` / :meth:`embed_query` and never
    learns which vendor answered. It is constructed from config by default;
    the keyword arguments exist so a test (or a caller managing its own HTTP
    client) can supply a transport without monkeypatching the SDK.
    """

    def __init__(
        self,
        api_key: str = "",
        *,
        base_url: str | None = None,
        model: str | None = None,
        dimension: int | None = None,
        batch_size: int | None = None,
        max_concurrency: int | None = None,
        max_retries: int | None = None,
        request_timeout: int | None = None,
        cache: EmbeddingCache | None = None,
        price_per_mtok: float | None = None,
        client=None,
    ):
        self.api_key = config.resolve_api_key(api_key)
        self.base_url = str(base_url or getattr(config, "EMBEDDING_API_BASE_URL", "")
                            or config.OPENAI_BASE_URL).rstrip("/")
        self.model = str(
            model or getattr(config, "EMBEDDING_MODEL", "") or "text-embedding-3-small"
        )
        self.dimension = int(dimension or getattr(config, "EMBEDDING_DIMENSION", 1536) or 1536)
        # An operator who named their space wins; otherwise the endpoint names
        # it. Reading the configured name rather than only deriving one from the
        # URL is what lets this agree with the ``index_version`` the chunker
        # stamps: the two are compared before every index write and a
        # disagreement refuses the write, so a name only one side knows is worse
        # than no name at all. That is not hypothetical -- with a placeholder
        # default the two could never agree and every upload failed.
        self.provider = (
            str(getattr(config, "EMBEDDING_PROVIDER", "") or "").strip().lower()
            or provider_identity(self.base_url)
        )
        self.batch_size = int(batch_size or _config_int(
            "EMBEDDING_BATCH_SIZE", DEFAULT_BATCH_SIZE, 1, 2048))
        self.max_concurrency = int(max_concurrency or _config_int(
            "EMBEDDING_MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY, 1, 64))
        self.max_retries = int(max_retries or _config_int(
            "EMBEDDING_MAX_RETRIES", DEFAULT_MAX_RETRIES, 1, 10))
        self.request_timeout = int(request_timeout or _config_int(
            "EMBEDDING_REQUEST_TIMEOUT_SECONDS", DEFAULT_REQUEST_TIMEOUT_SECONDS, 1, 600))
        self.price_per_mtok = (
            price_per_mtok if price_per_mtok is not None
            else _config_float("EMBEDDING_PRICE_PER_MTOK")
        )
        self.cache = cache if cache is not None else EmbeddingCache(
            max_entries=_config_int(
                "EMBEDDING_MAX_CACHE_ENTRIES", DEFAULT_MAX_CACHE_ENTRIES, 1, 1_000_000)
        )
        # Caps in-flight provider requests across threads. Requests are issued
        # synchronously, so this bounds concurrency rather than creating it: an
        # embedding batch must not be reordered to save a few hundred ms, and a
        # naive ThreadPoolExecutor fan-out would also bypass the provider's own
        # rate limits.
        self._semaphore = threading.BoundedSemaphore(self.max_concurrency)
        self._usage_lock = threading.Lock()
        self.usage = self._empty_usage()
        self.client = client if client is not None else (
            OpenAI(api_key=self.api_key, base_url=self.base_url) if self.api_key else None
        )

    # -- identity ----------------------------------------------------------

    @property
    def index_version(self) -> str:
        """Provider + model + dimension, as the section 8 risk requires.

        Stored alongside every vector so a later deployment with a different
        vendor, model or width can be recognised as incompatible and rebuild
        the index instead of blending incompatible spaces.
        """
        return f"emb-{_slug(self.provider)}-{_slug(self.model)}-d{self.dimension}"

    def assert_same_space(self, other_version: str) -> None:
        """Refuse to mix this gateway with a space built by another setting.

        Call this at the seam where vectors from an existing index meet a live
        query. The two halves may differ in provider or model, not only in
        dimension, and every one of those differences makes the comparison
        meaningless rather than merely inaccurate.
        """
        if str(other_version or "") != self.index_version:
            raise EmbeddingDimensionMismatchError(
                f"向量空间不匹配：当前配置为 {self.index_version}，"
                f"索引记录的版本为 {other_version or '(空)'}。"
                "换模型或换维度后必须重建索引，不能混用旧向量。"
            )

    # -- internals ---------------------------------------------------------

    def _empty_usage(self) -> EmbeddingUsage:
        return EmbeddingUsage(
            provider=self.provider,
            model=self.model,
            dimension=self.dimension,
            index_version=self.index_version,
            price_per_mtok=self.price_per_mtok,
        )

    def _require_client(self) -> None:
        if not self.client:
            raise ValueError(
                "Embedding 服务需要 API Key，请在请求中提供或设置 OPENAI_API_KEY。"
            )

    def cache_key(self, text: str) -> tuple[str, str, str]:
        """``(provider, model, content_hash)`` -- see the module docstring."""
        return (self.provider, self.model, content_hash(text))

    def _classify_error(self, exc: Exception) -> str:
        if isinstance(exc, APIStatusError):
            return describe_llm_status_code(exc.status_code)
        if isinstance(exc, APITimeoutError):
            return "Embedding 服务请求超时，请稍后重试。"
        if isinstance(exc, APIConnectionError):
            return "Embedding 服务连接失败，请检查网络、API 地址或供应商服务状态。"
        return str(exc) or "Embedding 服务请求失败。"

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        """Only transient conditions are replayed; faults are surfaced at once."""
        if isinstance(exc, EmbeddingDimensionMismatchError):
            return False
        if isinstance(exc, APIStatusError):
            return exc.status_code in RETRYABLE_STATUS_CODES
        # Only transport faults are replayed. A ``ValueError`` reaching here is
        # this module's own response validation, and a malformed response will
        # be malformed again.
        return isinstance(exc, (APITimeoutError, APIConnectionError, OSError))

    def _check_dimension(self, vector) -> None:
        if len(vector) != self.dimension:
            raise EmbeddingDimensionMismatchError(
                f"向量维度与配置不符：{self.model}（{self.provider}）返回 {len(vector)} 维，"
                f"而配置 EMBEDDING_DIMENSION={self.dimension}。"
                "请修正维度配置并重建索引，不要混用不同维度的向量。"
            )

    def _extract_vectors(self, response, expected: int) -> list[list[float]]:
        data = getattr(response, "data", None) or []
        if len(data) != expected:
            raise ValueError(
                f"Embedding 服务返回了 {len(data)} 条结果，请求了 {expected} 条，"
                "请检查供应商接口兼容性。"
            )
        # Providers are expected to echo an ``index`` per item; sort on it so a
        # response that arrives out of order still maps back to its input.
        ordered = sorted(data, key=lambda item: getattr(item, "index", 0))
        vectors = []
        for item in ordered:
            raw = getattr(item, "embedding", None)
            if raw is None:
                raise ValueError("Embedding 服务返回的条目缺少 embedding 字段。")
            vector = [float(value) for value in raw]
            self._check_dimension(vector)
            vectors.append(vector)
        return vectors

    @staticmethod
    def _extract_tokens(response) -> int | None:
        """Prompt tokens as reported by the provider, or ``None`` if it said nothing."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        for name in ("prompt_tokens", "total_tokens"):
            value = getattr(usage, name, None)
            if value is None and isinstance(usage, dict):
                value = usage.get(name)
            if value is not None:
                return int(value)
        return None

    def _batches(self, texts: list[str]) -> list[list[str]]:
        size = max(1, self.batch_size)
        return [texts[start:start + size] for start in range(0, len(texts), size)]

    def _create_embeddings(self, inputs: list[str]):
        """One provider request, retried only while the failure is transient."""
        retries = self.max_retries
        last_message = ""

        for attempt in range(retries):
            try:
                with self._semaphore:
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=list(inputs),
                        timeout=self.request_timeout,
                    )
                return response, self._extract_vectors(response, len(inputs))
            except EmbeddingDimensionMismatchError as exc:
                # Kept as its own type rather than folded into the RuntimeError
                # below: a wrong dimension is a rebuild-the-index instruction,
                # and a caller that catches the generic failure would only see
                # a message string.
                logger.error("Embedding 维度不符（不可重试）: %s", exc)
                raise
            except Exception as exc:
                last_message = self._classify_error(exc)
                cause = exc
                if not self._is_retryable(exc):
                    logger.error("Embedding 请求失败（不可重试）: %s", last_message)
                    raise RuntimeError(last_message) from exc

            if attempt < retries - 1:
                time.sleep(get_retry_delay_seconds(attempt, max_delay=8))
            else:
                logger.error(last_message)
                # ``exc`` is unbound out here -- Python unbinds the target at
                # the end of an ``except`` block -- so the cause is carried
                # across in a local instead of read off the exception.
                raise RuntimeError(last_message) from cause

        raise RuntimeError(last_message or "Embedding 服务请求失败。")

    def _record_usage(self, usage: EmbeddingUsage) -> EmbeddingUsage:
        with self._usage_lock:
            self.usage = self.usage.merge(usage)
        return usage

    def reset_usage(self) -> EmbeddingUsage:
        """Zero the running total and return what it was (T1.8 run boundaries)."""
        with self._usage_lock:
            previous, self.usage = self.usage, self._empty_usage()
        return previous

    # -- public API --------------------------------------------------------

    def embed_documents(self, texts) -> EmbeddingBatch:
        """Embed many passages, honouring the batch limit and the cache.

        Cached texts never reach the provider; duplicates inside one call are
        collapsed before the request, so a paper that quotes itself costs one
        call. Results keep the input order regardless of how the work was
        batched or served from cache.
        """
        self._require_client()
        items = [str(text or "") for text in texts]
        for text in items:
            if not text.strip():
                raise ValueError("不能对空文本生成向量，请检查传入内容。")
        if not items:
            return EmbeddingBatch((), self._record_usage(self._empty_usage()))

        keys = [self.cache_key(text) for text in items]
        vectors: list[list[float] | None] = [None] * len(items)
        cache_hits = 0
        pending: dict[tuple[str, str, str], str] = {}
        for index, (text, key) in enumerate(zip(items, keys, strict=True)):
            cached = self.cache.get(key)
            if cached is not None:
                vectors[index] = list(cached)
                cache_hits += 1
                continue
            pending.setdefault(key, text)

        pending_keys = list(pending.keys())
        pending_texts = [pending[key] for key in pending_keys]
        produced: dict[tuple[str, str, str], list[float]] = {}
        requests = 0
        tokens = 0
        tokens_reported = True

        offset = 0
        for batch_texts in self._batches(pending_texts):
            batch_keys = pending_keys[offset:offset + len(batch_texts)]
            offset += len(batch_texts)
            response, batch_vectors = self._create_embeddings(batch_texts)
            requests += 1
            reported = self._extract_tokens(response)
            if reported is None:
                tokens_reported = False
            else:
                tokens += reported
            for key, vector in zip(batch_keys, batch_vectors, strict=True):
                produced[key] = vector
                self.cache.put(key, vector)

        for index, key in enumerate(keys):
            if vectors[index] is None:
                vectors[index] = produced[key]

        usage = EmbeddingUsage(
            provider=self.provider,
            model=self.model,
            dimension=self.dimension,
            index_version=self.index_version,
            requests=requests,
            texts=len(items),
            cache_hits=cache_hits,
            input_tokens=tokens,
            tokens_reported=tokens_reported,
            price_per_mtok=self.price_per_mtok,
        )
        return EmbeddingBatch(
            vectors=tuple(tuple(vector) for vector in vectors),
            usage=self._record_usage(usage),
        )

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query.

        Same path as :meth:`embed_documents` -- a query that has already been
        embedded is served from cache exactly like a chunk, and a caller cannot
        accidentally pick up a different dimension by using the other method.
        """
        return list(self.embed_documents([text])[0])
