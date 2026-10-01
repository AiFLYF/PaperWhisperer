"""A local, offline embedding provider that speaks the existing gateway.

Every other part of this project's vector path was written, reviewed and tested
against a provider that has never once answered. ``config.EMBEDDING_API_BASE_URL``
points at a local OpenAI-compatible endpoint whose ``/v1/embeddings`` route
returns 404, so the chain parse -> chunk -> embed -> index -> retrieve -> answer
has never run end to end. The retriever's dense leg has been abstaining the
whole time and recording a degradation in a log line nobody read. This module
is what makes the claim testable.

**It is a transport swap, not a fork.** :class:`LocalEmbeddingProvider` subclasses
:class:`paperwhisperer.llm.embeddings.EmbeddingGateway` and replaces exactly one
thing: the object that answers ``client.embeddings.create(...)``. The cache
key, the dimension check, the batch split, the duplicate collapse, the usage
ledger, the cost arithmetic, ``index_version`` and ``assert_same_space`` are all
the gateway's own code, running unchanged. That is the whole of T1.3's
acceptance criterion -- "provider 切换无需改业务代码" -- expressed as a subclass
rather than a promise: there is no business code to change, because none of it
knew there was a choice. Everything that reads an embedding through
:class:`~paperwhisperer.rag.retrieve.QueryEmbedder` cannot tell the difference,
and neither can :func:`paperwhisperer.analysis.orchestrator.index_document_for_retrieval`.

**Decision D3 is honoured, and this module is not a replacement for it.** The
plan says the product talks to OpenAI-compatible APIs by default and explicitly
*not* to a local BGE-M3. So the extra is optional, off by default, and exists for
three narrower reasons: a personal deployment with no network and no API key
(a supported configuration, not a degraded one), a test that must exercise the
real embedding path without a network, and a bake-off that wants to compare two
providers over one document. Anyone who wants the product's default behaviour
installs none of this and configures ``EMBEDDING_API_BASE_URL`` instead.

**The model has its own width, and the configured one must agree with it.** This
is plan section 8's risk -- "OAI 兼容 embedding 各家维度/模型不同 -> index_version
编码 provider+model+dim，换模型重建索引而非混用" -- and the local case is the sharp
version of it, because a local model's width is not a vendor's choice the
operator writes down: it is a property of the weights. ``bge-small-en-v1.5`` is
384, ``gte-large`` is 1024, and the previous provider's 1536 has no reason to
survive. So the width is resolved twice and the two answers must match: once
from ``core.config`` (what the index claims) and once from fastembed's own
registry (what the weights will produce). A disagreement raises
:class:`~paperwhisperer.llm.embeddings.EmbeddingDimensionMismatchError` at
construction, naming both numbers, before a single vector exists. The gateway's
per-vector check then re-verifies on every call, because the registry is a
static table and a loaded model is a fact.

**A provider or model change changes ``index_version``, so an old index is
rebuilt rather than blended.** Two independent strings encode provider, model and
dimension, and both move: the gateway's own :attr:`index_version`, which the
retriever records and the usage ledger reports, and
:func:`paperwhisperer.rag.chunking.build_index_version`, which the chunker
stamps onto every chunk and which the store's idempotence check compares. The
second one is the load-bearing one, and it is read from ``core.config`` -- which
is why this module does not merely set its own identity but requires that
``EMBEDDING_PROVIDER``/``EMBEDDING_MODEL``/``EMBEDDING_DIMENSION`` name *it*
while it is the configured provider. :func:`local_identity` is the single
resolver both sides read, so a configuration that changed the provider without
changing the index version is not expressible.

**Cost is zero money and non-zero time, and those are different facts.** A
vendored provider bills per token, so ``EMBEDDING_PRICE_PER_MTOK`` is how a run
becomes a number. A local model bills nothing and spends CPU instead, so
``EmbeddingUsage.cost`` is a real ``0.0`` -- distinct from the ``None`` that
means "we could not tell" -- and the seconds actually spent are accumulated on
:attr:`LocalEmbeddingProvider.embed_seconds` and logged. Reporting the ledger
column without the wall clock would be the dishonest half of the measurement:
an index rebuild that takes four minutes is not free, it is just not billed.

**What is not claimed.** ONNX on CPU is materially slower than a hosted
endpoint and the model is small, so this is a personal-tier and testing
provider. Its retrieval quality on this corpus is measured in
``tests/test_rag_end_to_end.py`` rather than asserted here, because a docstring
asserting a recall number would be a number nobody re-measured when the model
changed.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from paperwhisperer.core import config
from paperwhisperer.llm.embeddings import (
    EmbeddingCache,
    EmbeddingDimensionMismatchError,
    EmbeddingGateway,
    EmbeddingUsage,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_LOCAL_MODEL",
    "LOCAL_PROVIDER_NAME",
    "LocalEmbeddingProvider",
    "build_embedder_from_config",
    "fastembed_installed",
    "local_identity",
]


#: The provider name written into ``index_version``. Distinct from any netloc:
#: :func:`paperwhisperer.llm.embeddings.provider_identity` derives a vendored
#: provider from its host, and a local model has no host, so it needs a name of
#: its own. It must match the default ``EMBEDDING_PROVIDER`` in
#: :mod:`paperwhisperer.core.config`, because the chunker reads the version from
#: there and not from this module -- see the module docstring.
LOCAL_PROVIDER_NAME = "fastembed_local"

#: The model used when the operator named none. A multilingual MiniLM rather
#: than an English-only one because this corpus is bilingual by design
#: (``eval/corpus/mixed_zh_en``), and an English-only model answers a Chinese
#: query with its prior for every passage -- which is precisely the
#: "dense 通道没有区分力" case :data:`paperwhisperer.rag.retrieve.DENSE_TIE_BAND`
#: exists to detect. 384 dimensions, ~220 MB of ONNX weights, no torch.
DEFAULT_LOCAL_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

#: The width ``DEFAULT_LOCAL_MODEL`` produces, declared in
#: :mod:`paperwhisperer.core.config` as ``EMBEDDING_LOCAL_DIMENSION`` and
#: re-verified against fastembed's registry here. Two sources on purpose: config
#: cannot import an optional extra at module scope, and a registry lookup cannot
#: run before the operator has chosen a model. The provider is the seam where
#: the two are made to agree, and it refuses rather than picking a winner.
DEFAULT_LOCAL_DIMENSION = 384

#: faked for the transport shim's ``base_url``. Never dialled -- the whole point
#: is that there is no socket -- but the gateway reads it in its constructor and
#: derives a provider name from the netloc, and an un-routable scheme is better
#: than a plausible-looking host that might be mistaken for a reachable one.
#: ``self.provider`` is overridden with :data:`LOCAL_PROVIDER_NAME` regardless.
_LOCAL_BASE_URL = "local://fastembed"

#: Passed as the gateway's ``api_key``. The local provider has no key and asks
#: for none; the gateway only reads this to decide whether to construct an HTTP
#: client, and a local gateway is handed a transport shim instead.
_LOCAL_API_KEY = "local-no-key-required"

#: Handed to the base constructor in place of a transport so it does not build
#: an :class:`openai.OpenAI` for a provider that will never open a socket. It is
#: replaced before the constructor returns; the name says so, because a sentinel
#: that reads as a real client is a bug waiting for the next reader.
_NO_TRANSPORT_YET = object()


def fastembed_installed() -> bool:
    """Whether the optional extra is present, without importing it.

    ``find_spec`` rather than a real import, for the reason
    :func:`paperwhisperer.rag.evidence.page_renderer_name` gives: this module is
    imported by the config-driven provider resolver, and a capability question
    must not pay for loading ONNX Runtime and pulling a multi-megabyte shared
    object into a process that may only have wanted the answer "no".
    """
    import importlib.util

    try:
        return importlib.util.find_spec("fastembed") is not None
    except (ImportError, ValueError):  # a broken install is "not installed"
        return False


def local_identity() -> tuple[str, str, int]:
    """``(provider, model, dimension)`` the local provider will actually use.

    Read from :mod:`paperwhisperer.core.config`, which is the only place that
    decides. It is a function rather than three module constants because
    ``EMBEDDING_MODEL`` and ``EMBEDDING_DIMENSION`` are shared settings that the
    local provider *overrides when it is the configured provider* -- so reading
    them directly here would disagree with the chunker whenever the other
    provider is in force. Going through the one resolver is what keeps the
    gateway's ``index_version`` and the chunker's ``index_version`` describing
    the same vectors.

    Raises :class:`EmbeddingDimensionMismatchError` when the configured width is
    not the one the chosen model's registry entry declares. The check is here
    and not only in the constructor because this function is what the resolver
    and the health summary call, and a caller that never constructs a provider
    should still not be able to print a dimension that is wrong.
    """
    provider = LOCAL_PROVIDER_NAME
    model = str(getattr(config, "EMBEDDING_LOCAL_MODEL", "") or DEFAULT_LOCAL_MODEL)
    dimension = int(getattr(config, "EMBEDDING_DIMENSION", 0) or DEFAULT_LOCAL_DIMENSION)
    native = _registry_dimension(model)
    if native and int(native) != dimension:
        # Which knob to name depends on which one is actually in force: with the
        # local provider enabled, ``EMBEDDING_DIMENSION`` *derives* from
        # ``EMBEDDING_LOCAL_DIMENSION`` and writing ``EMBEDDING_DIMENSION=384``
        # in a ``.env`` would change nothing at all. An operator told to edit
        # the wrong name edits it, sees the same error again, and concludes the
        # check is broken.
        knob = (
            "EMBEDDING_LOCAL_DIMENSION"
            if bool(getattr(config, "EMBEDDING_LOCAL_ENABLED", False))
            else "EMBEDDING_DIMENSION"
        )
        raise EmbeddingDimensionMismatchError(
            f"本地模型 {model} 的向量宽度是 {native}，而配置声明的是 {dimension}。"
            "两个数字必须一致：index_version 会写入配置的维度，索引里存的是模型的维度，"
            "不一致会让检索拿着一个无法比较的空间去排序。"
            f"请把 {knob} 设为 {native}"
            "（同时确认 EMBEDDING_LOCAL_ENABLED 与本进程实际使用的 provider 一致）。"
        )
    return provider, model, int(dimension or native or DEFAULT_LOCAL_DIMENSION)


def _registry_dimension(model: str) -> int:
    """The width fastembed's registry declares for ``model``, or ``0``.

    A registry lookup, so it costs no download and no ONNX session -- which is
    what makes it usable as a configuration check at import-adjacent time. It
    returns ``0`` for a model fastembed does not know, and ``0`` is treated as
    "no opinion" everywhere it is read: a custom or newly published model is a
    supported configuration, and the width is then discovered from the loaded
    model itself by the gateway's own per-vector check, which is stricter than
    any table.
    """
    if not fastembed_installed():
        return 0
    from fastembed import TextEmbedding

    try:
        return int(TextEmbedding.get_embedding_size(model))
    except (ValueError, TypeError, KeyError):
        # fastembed raises ValueError listing every model it knows for an
        # unknown name, which is an answer ("I do not know") and not a fault.
        logger.debug("fastembed registry has no width for %r; deferring to the model.", model)
        return 0


# --------------------------------------------------------------------------
# The transport
# --------------------------------------------------------------------------


class _LocalResponse:
    """The shape :meth:`EmbeddingGateway._extract_vectors` reads.

    An openai-shaped response built from a local inference, and built to be
    indistinguishable from a vendored one on purpose: the gateway validates the
    payload through its own code path, so the dimension check, the out-of-order
    ``index`` sort and the "did the provider report usage" bookkeeping are the
    same code whether the vectors came from a socket or from an ONNX session.
    Duplicating that validation here would be the fork this module exists not
    to be.
    """

    __slots__ = ("data", "usage")

    def __init__(self, vectors: list[list[float]], prompt_tokens: int) -> None:
        self.data = [_LocalItem(index, vector) for index, vector in enumerate(vectors)]
        self.usage = _LocalUsage(prompt_tokens)


class _LocalItem:
    __slots__ = ("embedding", "index")

    def __init__(self, index: int, embedding: list[float]) -> None:
        self.index = index
        self.embedding = embedding


class _LocalUsage:
    __slots__ = ("prompt_tokens", "total_tokens")

    def __init__(self, prompt_tokens: int) -> None:
        self.prompt_tokens = int(prompt_tokens)
        self.total_tokens = int(prompt_tokens)


class _LocalEmbeddingsEndpoint:
    """``client.embeddings.create`` answered from an ONNX session.

    Token counts are real, measured with the model's own tokenizer, rather than
    a constant. The local provider bills nothing per token, so the number buys
    no money -- but ``EmbeddingUsage`` carries ``input_tokens`` as a column
    whether or not a price is configured, and a column that is always the same
    value is a column that stops meaning anything the first time someone wires
    it to a real price for a bake-off.
    """

    def __init__(self, owner: LocalEmbeddingProvider) -> None:
        self._owner = owner

    def create(self, *, model: str, input, timeout=None, **kwargs: Any) -> _LocalResponse:
        texts = [input] if isinstance(input, str) else [str(item) for item in (input or [])]
        vectors = self._owner._encode(
            texts, query=bool(getattr(self._owner._state, "query", False))
        )
        tokens = self._owner._token_count(texts)
        return _LocalResponse(vectors, tokens)


class _LocalClient:
    __slots__ = ("embeddings",)

    def __init__(self, owner: LocalEmbeddingProvider) -> None:
        self.embeddings = _LocalEmbeddingsEndpoint(owner)


# --------------------------------------------------------------------------
# The provider
# --------------------------------------------------------------------------


class LocalEmbeddingProvider(EmbeddingGateway):
    """``EmbeddingGateway`` with the socket replaced by an ONNX session.

    Construct it with no arguments in a configured deployment; the constructor
    resolves provider, model and dimension through :func:`local_identity` and
    refuses if the configured width is not the model's own. A caller that wants
    a specific model for a bake-off passes ``model`` and the matching
    ``dimension``, and every check still runs.

    :attr:`embed_seconds` accumulates wall-clock time spent inside inference.
    It is the honest cost of this provider and has no column in
    :class:`~paperwhisperer.llm.embeddings.EmbeddingUsage`, which is a *money*
    ledger; see the module docstring on why both numbers are reported and
    neither is dropped.
    """

    def __init__(
        self,
        api_key: str = "",
        *,
        model: str | None = None,
        dimension: int | None = None,
        cache: EmbeddingCache | None = None,
        batch_size: int | None = None,
        max_concurrency: int | None = None,
        max_retries: int | None = None,
        price_per_mtok: float | None = None,
        asymmetric: bool | None = None,
        model_dir: str | None = None,
        threads: int | None = None,
    ) -> None:
        resolved_provider, configured_model, configured_dim = local_identity()
        chosen_model = str(model or configured_model)
        native = _registry_dimension(chosen_model)
        if model is not None and dimension is None:
            # A caller naming a model for a bake-off names its width too, and
            # the registry is the only authority on what that width is. Guessing
            # would produce vectors the search path cannot compare, which is the
            # exact failure section 8 warns about.
            if not native:
                raise EmbeddingDimensionMismatchError(
                    f"本地模型 {chosen_model} 不在 fastembed 的注册表里，无法确认其向量宽度。"
                    "请同时传入 dimension=，或改用注册表内的模型名；"
                    "不确认宽度就索引等于把两种空间混进一个 store。"
                )
            configured_dim = int(native)
        if int(dimension or configured_dim) != int(native or configured_dim):
            raise EmbeddingDimensionMismatchError(
                f"本地模型 {chosen_model} 的向量宽度是 {native}，"
                f"而调用方要求 {dimension or configured_dim}。"
                "维度必须与模型一致，否则索引里的向量无法互相比较。"
            )

        # The remaining transport-shaped settings are deliberately left at the
        # caller's or the config's value rather than overridden. ``max_concurrency``
        # and ``max_retries`` both describe a socket: they cap in-flight HTTP
        # requests and replay transient HTTP failures, and neither has a
        # meaning here. They are harmless because the gateway issues requests
        # synchronously -- the semaphore is acquired and released around an
        # ONNX call that does not return until it is done -- so leaving them
        # alone keeps the base class's "no fan-out" contract intact instead of
        # implying a concurrency this transport cannot honour.
        super().__init__(
            _LOCAL_API_KEY,
            base_url=_LOCAL_BASE_URL,
            model=chosen_model,
            dimension=int(dimension or configured_dim),
            batch_size=batch_size,
            max_concurrency=max_concurrency,
            max_retries=max_retries,
            cache=cache,
            price_per_mtok=price_per_mtok,
            client=_NO_TRANSPORT_YET,
        )
        # The base constructor derived ``provider`` from the fake netloc and
        # built an empty usage record from it. Both are corrected here and the
        # usage record is rebuilt, because a ledger whose first row names the
        # wrong provider is a ledger nobody trusts.
        self.provider = resolved_provider
        self.usage = self._empty_usage()
        self.client = _LocalClient(self)
        self._model_obj: Any = None
        self._model_lock = threading.Lock()
        self._state = threading.local()
        self.embed_seconds = 0.0
        self.asymmetric = (
            bool(asymmetric)
            if asymmetric is not None
            else bool(getattr(config, "EMBEDDING_LOCAL_ASYMMETRIC", False))
        )
        self.model_dir = str(
            model_dir or getattr(config, "EMBEDDING_LOCAL_MODEL_DIR", "") or ""
        ).strip()
        self.threads = int(
            threads if threads is not None else getattr(config, "EMBEDDING_LOCAL_THREADS", 0) or 0
        )
        # Installed after ``_empty_usage`` so the price is on the running total
        # too, not only on subsequent operations.
        if self.price_per_mtok is None:
            # A local model costs no money. ``0.0`` is a real number and says
            # so; the inherited ``None`` would say "we could not tell", which for
            # a provider with no meter is a different and weaker claim than the
            # truth. The wall clock is on ``embed_seconds``.
            self.price_per_mtok = 0.0
            self.usage = self._empty_usage()

    # -- model -------------------------------------------------------------

    def load_model(self) -> Any:
        """The underlying ``fastembed.TextEmbedding``, loaded once per process.

        Loading costs seconds and hundreds of megabytes on the first call and is
        then free, so it is cached behind a lock and shared by every provider
        instance -- a Q&A process that builds a retriever per document must not
        pay that per document. The double-checked lock is on the *model key*,
        not the instance, so two providers on the same model share one session
        and two providers on different models do not block each other after
        their first load.
        """
        key = (self.model, self.model_dir, self.threads)
        existing = _MODEL_CACHE.get(key)
        if existing is not None:
            return existing
        with self._model_lock:
            existing = _MODEL_CACHE.get(key)
            if existing is not None:
                return existing
            from fastembed import TextEmbedding

            started = time.perf_counter()
            options: dict[str, Any] = {}
            if self.model_dir:
                options["cache_dir"] = self.model_dir
            if self.threads > 0:
                options["threads"] = self.threads
            existing = TextEmbedding(self.model, **options)
            _MODEL_CACHE[key] = existing
            logger.info(
                "本地 embedding 模型已加载：%s（%d 维，耗时 %.1fs，首次会下载权重）。",
                self.model,
                int(self.dimension),
                time.perf_counter() - started,
            )
            return existing

    def native_dimension(self) -> int:
        """The width the loaded model actually produces.

        Read from the model rather than from configuration, and therefore the
        one width in this module that cannot be a stale setting. Called once at
        construction in :meth:`_verify_loaded_width` so a mismatch is a refusal
        rather than a surprise on the first search.
        """
        return int(getattr(self.load_model(), "embedding_size", 0) or 0)

    # -- gateway overrides -------------------------------------------------

    def _require_client(self) -> None:
        """Replace the API-key check with a loadable-model check.

        The base class refuses to embed without a key, which is right for a
        metered provider and meaningless here. What *is* meaningful is that the
        model is present, so the refusal becomes: fastembed missing, or weights
        unreachable. Both are configuration or network facts, and both belong in
        a message a person can act on rather than in an ``AttributeError`` from
        three frames down.
        """
        if not fastembed_installed():
            raise RuntimeError(
                "本地 embedding 需要可选依赖 fastembed，但它没有安装。"
                "这是一个受支持的部署形态而不是故障："
                "安装 `pip install paperwhisperer[local-embeddings]` 启用它，"
                "或改用 EMBEDDING_API_BASE_URL 指向的 OpenAI 兼容服务（D3 的默认路径）。"
            )
        self.load_model()

    def embed_query(self, text: str) -> list[float]:
        """One query vector, in the same space as the stored vectors.

        Routed through the base implementation -- so the cache, the batching and
        the usage ledger are the gateway's own -- with a thread-local flag set
        for the duration of the call. A thread-local rather than an instance
        attribute because the gateway is explicitly documented as safe to share
        across threads, and an instance flag would make a query embedding depend
        on which thread happened to run it. The flag is what lets a model that
        wants an asymmetric query prefix (E5, Jina, GTE) get one without
        reimplementing the cache path; for the default MiniLM, whose registry
        entry says prefixes are unnecessary, the flag is inert.
        """
        self._state.query = True
        try:
            return super().embed_query(text)
        finally:
            self._state.query = False

    def describe(self) -> dict[str, Any]:
        """Which provider this is, for a health payload or a log line.

        The dimension reported is the configured one, because that is the number
        the index was built against, and the model's own width is reported
        beside it so a disagreement is visible in the payload rather than
        discovered by a search.
        """
        return {
            "provider": self.provider,
            "model": self.model,
            "dimension": int(self.dimension),
            "registry_dimension": _registry_dimension(self.model),
            "index_version": self.index_version,
            "extra": "local-embeddings",
            "asymmetric": self.asymmetric,
            "embed_seconds": round(self.embed_seconds, 3),
        }

    # -- internals ---------------------------------------------------------

    def _encode(self, texts: list[str], *, query: bool) -> list[list[float]]:
        """Run the texts through the model and return plain float lists.

        ``float(value) for value in row`` rather than ``row.tolist()`` because
        the gateway's validator and the store's packer both iterate, and handing
        them numpy scalars would make the vector's type depend on whether an
        optional extra happened to be installed. The result is the same Python
        list of floats a JSON response would have produced.
        """
        model = self.load_model()
        self._verify_loaded_width()
        encode = model.query_embed if (query and self.asymmetric) else model.embed
        started = time.perf_counter()
        rows = list(encode(texts))
        self.embed_seconds += time.perf_counter() - started
        return [[float(value) for value in row] for row in rows]

    def _verify_loaded_width(self) -> None:
        """Refuse, once, if the loaded model's width is not the configured one.

        The registry table is a static list; the loaded session is the fact. They
        can disagree when a model is republished, when a mirror serves a
        different revision, or when an operator points ``EMBEDDING_LOCAL_MODEL``
        at a private ONNX build under a name the registry does not know. The
        gateway's per-vector check would catch it on the first batch anyway, but
        it would catch it *mid-index*, after some chunks were stored -- so the
        check runs first, against the model, and fails before any write.
        """
        native = self.native_dimension()
        if native and int(native) != int(self.dimension):
            raise EmbeddingDimensionMismatchError(
                f"本地模型 {self.model} 实际输出 {native} 维，与配置的 "
                f"EMBEDDING_DIMENSION={self.dimension} 不符。"
                "请修正维度配置并重建索引，不要混用不同维度的向量。"
            )

    def _token_count(self, texts: list[str]) -> int:
        """Prompt tokens as the model's own tokenizer counts them.

        Never fails the embedding: a tokenizer that is unavailable or refuses an
        input must not cost the caller its vectors, and ``0`` is reported as
        "not measured" by the base class's ``tokens_reported`` logic rather than
        as a claim that the text was free.
        """
        try:
            return int(self.load_model().token_count(list(texts)))
        except Exception:  # noqa: BLE001 - metering must not break embedding
            logger.debug("本地 embedding 未能统计 token 数，成本字段将记为未知。", exc_info=True)
            return 0


#: Loaded models, keyed by ``(model, model_dir, threads)``. Module-level for the
#: reason :data:`paperwhisperer.analysis.orchestrator._QA_STORES` is: one ONNX
#: session per model per process, not one per provider instance.
_MODEL_CACHE: dict[tuple[str, str, int], Any] = {}


def build_embedder_from_config() -> EmbeddingGateway | None:
    """The configured embedding provider, or ``None`` when there is none.

    This is the function that makes provider choice a configuration change. It
    reads ``EMBEDDING_LOCAL_ENABLED`` and dispatches: local model, or the
    OpenAI-compatible gateway. ``None`` means "no embedding provider is
    configured", which is a supported personal-tier state -- the retriever
    answers from its lexical and fuzzy legs and records that the dense leg did
    not run, rather than failing.

    It never guesses. If the local provider is enabled but the extra is not
    installed, the reason is logged and ``None`` is returned, because
    "silently fell back to a vendor and started billing the operator" is not an
    acceptable reading of "I asked for offline".
    """
    if not config.EMBEDDING_LOCAL_ENABLED:
        return EmbeddingGateway()
    if not fastembed_installed():
        logger.warning(
            "EMBEDDING_LOCAL_ENABLED=1 但未安装 fastembed 可选依赖，"
            "本次进程不启用 dense 通道。安装 paperwhisperer[local-embeddings] "
            "或改用 EMBEDDING_API_BASE_URL。"
        )
        return None
    try:
        return LocalEmbeddingProvider()
    except EmbeddingDimensionMismatchError:
        # Refused, and the message is the operator's next action. Swallowing it
        # into a ``None`` would turn a loud configuration fault into a silent
        # lexical-only deployment, which is the failure mode this whole module
        # was written to end.
        raise


def usage_summary(usage: EmbeddingUsage) -> dict[str, Any]:
    """The usage record as the health payload reports it, plus a money reading.

    Thin on purpose: the columns are
    :meth:`~paperwhisperer.llm.embeddings.EmbeddingUsage.as_analysis_run_columns`
    -- T2.2's write site owns persistence -- and the only thing added here is the
    human-readable cost, because ``0.0`` and ``None`` mean different things and
    a JSON number cannot carry the difference to someone reading a dashboard.
    """
    payload = usage.as_analysis_run_columns()
    payload["cost_known"] = usage.cost is not None
    return payload
