"""Embedding gateway: provider-agnosticism, cache, batching, retry, cost.

The tests use a scripted fake transport (no network, no API key, no real
provider). The behaviours that must not regress are the ones the plan calls
out by name: a provider swap is a config edit, a cache hit is a *zero-call*
hit, and a dimension mismatch is refused loudly rather than silently blended.
"""

from __future__ import annotations

import httpx
import pytest
from openai import APIConnectionError, APIStatusError

from paperwhisperer.core import config
from paperwhisperer.llm.embeddings import (
    RETRYABLE_STATUS_CODES,
    EmbeddingCache,
    EmbeddingDimensionMismatchError,
    EmbeddingGateway,
    content_hash,
)

# --------------------------------------------------------------------------
# Scripted fake transport
# --------------------------------------------------------------------------

_REQUEST = httpx.Request("POST", "https://provider.invalid/v1/embeddings")


def vector_for(text: str, dim: int) -> list[float]:
    """A deterministic, text-dependent vector. Different text -> different vector."""
    seed = int(content_hash(text)[:8], 16)
    return [float((seed + position) % 97) for position in range(dim)]


class FakeEmbeddingItem:
    def __init__(self, index: int, embedding):
        self.index = index
        self.embedding = embedding


class FakeUsage:
    def __init__(self, prompt_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.total_tokens = prompt_tokens


class FakeEmbeddingResponse:
    def __init__(self, vectors, prompt_tokens: int | None = None):
        self.data = [FakeEmbeddingItem(i, vector) for i, vector in enumerate(vectors)]
        self.usage = FakeUsage(prompt_tokens) if prompt_tokens is not None else None


class FakeEmbeddingsEndpoint:
    """Replays one behaviour per call, repeating the last.

    An exception entry is raised; a callable entry is invoked with the request's
    ``input`` list and must return a response.
    """

    def __init__(self, behaviours):
        self.behaviours = list(behaviours)
        self.calls = 0
        self.batches: list[list[str]] = []
        self.requests: list[dict] = []

    def create(self, **kwargs):
        self.calls += 1
        inputs = list(kwargs.get("input") or [])
        self.batches.append(inputs)
        self.requests.append(dict(kwargs))
        behaviour = self.behaviours[min(self.calls - 1, len(self.behaviours) - 1)]
        if isinstance(behaviour, BaseException):
            raise behaviour
        if callable(behaviour):
            return behaviour(inputs)
        return behaviour


class FakeEmbeddingClient:
    def __init__(self, behaviours, name="fake"):
        self.name = name
        self.embeddings = FakeEmbeddingsEndpoint(behaviours)


def make_success(dim, prompt_tokens_per_text=4, report_tokens=True):
    """A behaviour callable producing ``dim``-wide vectors for whatever it is sent."""

    def _behaviour(inputs):
        tokens = prompt_tokens_per_text * len(inputs) if report_tokens else None
        return FakeEmbeddingResponse([vector_for(text, dim) for text in inputs], tokens)

    return _behaviour


def make_gateway(dim=4, behaviours=None, client=None, **kwargs):
    """Build a gateway wired to a fake transport, with a private cache."""
    client = client or FakeEmbeddingClient(behaviours or [make_success(dim)])
    kwargs.setdefault("cache", EmbeddingCache(max_entries=64))
    return EmbeddingGateway("test-key", dimension=dim, client=client, **kwargs)


def business_embed_passages(gateway, passages):
    """Business code: knows the interface, never the provider or the transport."""
    return [list(vector) for vector in gateway.embed_documents(passages)]


def status_error(code: int, message: str = "provider error") -> APIStatusError:
    return APIStatusError(
        message, response=httpx.Response(code, request=_REQUEST), body=None
    )


# --------------------------------------------------------------------------
# 1. Provider swap with zero business-code change
# --------------------------------------------------------------------------


class TestProviderSwap:
    def test_the_same_business_call_works_against_two_providers(self, monkeypatch):
        """Switching provider is an environment edit, not a code edit.

        The identical ``business_embed_passages`` function is the only caller in
        this test; it never learns a base_url. Two gateways are built purely by
        changing config, and both satisfy it.
        """
        passages = ["attention is all you need", "向量检索需要重建索引"]

        # Provider A
        monkeypatch.setattr(config, "EMBEDDING_API_BASE_URL", "https://alpha.invalid/v1")
        monkeypatch.setattr(config, "EMBEDDING_MODEL", "alpha-embed")
        monkeypatch.setattr(config, "EMBEDDING_DIMENSION", 4)
        client_a = FakeEmbeddingClient([make_success(4)], name="alpha")
        gateway_a = make_gateway(dim=4, client=client_a)

        # Provider B -- only config changed
        monkeypatch.setattr(config, "EMBEDDING_API_BASE_URL", "https://beta.invalid/v1")
        monkeypatch.setattr(config, "EMBEDDING_MODEL", "beta-embed")
        monkeypatch.setattr(config, "EMBEDDING_DIMENSION", 4)
        client_b = FakeEmbeddingClient([make_success(4)], name="beta")
        gateway_b = make_gateway(dim=4, client=client_b)

        vectors_a = business_embed_passages(gateway_a, passages)
        vectors_b = business_embed_passages(gateway_b, passages)

        assert [len(vector) for vector in vectors_a] == [4, 4]
        assert [len(vector) for vector in vectors_b] == [4, 4]
        # Each provider actually received the call, addressed with its own model.
        assert client_a.embeddings.calls == 1
        assert client_b.embeddings.calls == 1
        assert client_a.embeddings.requests[0]["model"] == "alpha-embed"
        assert client_b.embeddings.requests[0]["model"] == "beta-embed"

    def test_the_two_providers_produce_distinct_index_versions(self, monkeypatch):
        monkeypatch.setattr(config, "EMBEDDING_API_BASE_URL", "https://alpha.invalid/v1")
        monkeypatch.setattr(config, "EMBEDDING_MODEL", "m")
        monkeypatch.setattr(config, "EMBEDDING_DIMENSION", 4)
        alpha = make_gateway(dim=4)
        monkeypatch.setattr(config, "EMBEDDING_API_BASE_URL", "https://beta.invalid/v1")
        beta = make_gateway(dim=4)

        assert alpha.provider == "alpha.invalid"
        assert beta.provider == "beta.invalid"
        assert alpha.index_version != beta.index_version


# --------------------------------------------------------------------------
# 2. Cache: keyed on (provider, model, content_hash); hit == zero calls
# --------------------------------------------------------------------------


class TestEmbeddingCache:
    def test_a_repeat_passage_makes_zero_provider_calls(self):
        client = FakeEmbeddingClient([make_success(4)])
        gateway = make_gateway(dim=4, client=client)
        passages = ["unique passage one", "unique passage two"]

        first = gateway.embed_documents(passages)
        assert client.embeddings.calls == 1
        assert first.usage.requests == 1

        second = gateway.embed_documents(passages)
        # The acceptance criterion, asserted on the call count, not the value.
        assert client.embeddings.calls == 1
        assert second.usage.requests == 0
        assert second.usage.cache_hits == 2
        assert [list(vector) for vector in second] == [list(vector) for vector in first]

    def test_different_content_gets_a_different_cache_entry(self):
        client = FakeEmbeddingClient([make_success(4)])
        gateway = make_gateway(dim=4, client=client)

        gateway.embed_documents(["passage alpha"])
        gateway.embed_documents(["passage beta"])

        assert client.embeddings.calls == 2  # different text -> not a hit

    def test_identical_content_different_position_is_a_hit(self):
        """The hash is over the text, not the index, so a re-quoted passage is free."""
        client = FakeEmbeddingClient([make_success(4)])
        gateway = make_gateway(dim=4, client=client)

        gateway.embed_documents(["alpha", "beta"])
        client.embeddings.calls = 0
        gateway.embed_documents(["beta"])
        assert client.embeddings.calls == 0

    def test_duplicates_within_one_call_are_deduplicated(self):
        client = FakeEmbeddingClient([make_success(4)])
        gateway = make_gateway(dim=4, client=client)

        batch = gateway.embed_documents(["same", "same", "same"])

        # One request carrying one text, but three returned vectors.
        assert client.embeddings.calls == 1
        assert client.embeddings.batches[0] == ["same"]
        assert len(batch) == 3

    def test_an_injected_cache_is_consulted_for_a_zero_call_hit(self):
        cache = EmbeddingCache(max_entries=8)
        gateway = make_gateway(dim=4, cache=cache, client=FakeEmbeddingClient([make_success(4)]))
        cache.put(gateway.cache_key("pre-seeded"), vector_for("pre-seeded", 4))

        vectors = business_embed_passages(gateway, ["pre-seeded"])

        assert gateway.client.embeddings.calls == 0
        assert vectors == [vector_for("pre-seeded", 4)]

    def test_the_cache_is_bounded_and_evicts_least_recently_used(self):
        cache = EmbeddingCache(max_entries=2)
        key_a = ("p", "m", "a")
        key_b = ("p", "m", "b")
        key_c = ("p", "m", "c")

        cache.put(key_a, [1.0])
        cache.put(key_b, [2.0])
        # Touch A so B becomes the least-recently-used entry.
        assert cache.get(key_a) == (1.0,)
        cache.put(key_c, [3.0])

        assert len(cache) == 2
        assert key_a in cache
        assert key_c in cache
        assert key_b not in cache  # evicted

    def test_clear_empties_the_cache(self):
        cache = EmbeddingCache(max_entries=2)
        cache.put(("p", "m", "a"), [1.0])
        cache.clear()
        assert len(cache) == 0

    def test_stats_report_hits_and_misses(self):
        cache = EmbeddingCache(max_entries=4)
        cache.put(("p", "m", "a"), [1.0])
        cache.get(("p", "m", "a"))
        cache.get(("p", "m", "zzz"))
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["entries"] == 1


# --------------------------------------------------------------------------
# 3. Dimension identity: refuse to mix, rebuild instead
# --------------------------------------------------------------------------


class TestDimensionIdentity:
    def test_a_wrong_width_response_is_refused_loudly(self):
        # Config says 4, provider returns 8 -- refuse, do not pad or truncate.
        gateway = make_gateway(dim=4, behaviours=[make_success(8)])

        with pytest.raises(EmbeddingDimensionMismatchError) as excinfo:
            gateway.embed_documents(["a passage"])
        assert "维度" in str(excinfo.value)

    def test_a_dimension_mismatch_is_not_retried(self):
        client = FakeEmbeddingClient([make_success(8)])
        gateway = make_gateway(dim=4, client=client)

        with pytest.raises(EmbeddingDimensionMismatchError):
            gateway.embed_documents(["a passage"])
        assert client.embeddings.calls == 1  # permanent fault, not replayed

    def test_matching_spaces_pass_the_assertion(self):
        gateway = make_gateway(dim=4)
        gateway.assert_same_space(gateway.index_version)  # must not raise

    def test_mixing_spaces_is_refused(self):
        built = make_gateway(dim=4)
        other = make_gateway(dim=8)
        assert built.index_version != other.index_version

        with pytest.raises(EmbeddingDimensionMismatchError) as excinfo:
            built.assert_same_space(other.index_version)
        assert "重建索引" in str(excinfo.value)

    def test_an_empty_space_label_is_refused(self):
        gateway = make_gateway(dim=4)
        with pytest.raises(EmbeddingDimensionMismatchError):
            gateway.assert_same_space("")


# --------------------------------------------------------------------------
# 4. Batching honours the configured batch size
# --------------------------------------------------------------------------


class TestBatching:
    def test_an_oversized_batch_is_split(self):
        client = FakeEmbeddingClient([make_success(4)])
        gateway = make_gateway(dim=4, client=client, batch_size=2)
        passages = [f"passage {i}" for i in range(5)]

        batch = gateway.embed_documents(passages)

        assert client.embeddings.calls == 3
        assert [len(chunk) for chunk in client.embeddings.batches] == [2, 2, 1]
        assert len(batch) == 5

    def test_an_exactly_sized_batch_is_one_request(self):
        client = FakeEmbeddingClient([make_success(4)])
        gateway = make_gateway(dim=4, client=client, batch_size=3)

        gateway.embed_documents(["a", "b", "c"])
        assert client.embeddings.calls == 1

    def test_the_result_keeps_input_order_across_batches(self):
        client = FakeEmbeddingClient([make_success(4)])
        gateway = make_gateway(dim=4, client=client, batch_size=2)
        passages = [f"passage {i}" for i in range(5)]

        vectors = business_embed_passages(gateway, passages)

        assert vectors == [vector_for(text, 4) for text in passages]

    def test_an_empty_request_short_circuits(self):
        client = FakeEmbeddingClient([make_success(4)])
        gateway = make_gateway(dim=4, client=client)
        batch = gateway.embed_documents([])
        assert client.embeddings.calls == 0
        assert len(batch) == 0

    def test_an_empty_passage_is_refused(self):
        gateway = make_gateway(dim=4)
        with pytest.raises(ValueError, match="空文本"):
            gateway.embed_documents(["   "])


# --------------------------------------------------------------------------
# 5. Retry policy
# --------------------------------------------------------------------------


class TestRetryPolicy:
    def test_a_transient_connection_failure_is_retried_then_succeeds(self, no_sleep):
        client = FakeEmbeddingClient(
            [APIConnectionError(request=_REQUEST), make_success(4)]
        )
        gateway = make_gateway(dim=4, client=client)

        batch = gateway.embed_documents(["a passage"])

        assert client.embeddings.calls == 2
        assert list(batch[0]) == vector_for("a passage", 4)

    def test_a_5xx_is_retried(self, no_sleep):
        assert 500 in RETRYABLE_STATUS_CODES
        client = FakeEmbeddingClient([status_error(500), make_success(4)])
        gateway = make_gateway(dim=4, client=client)

        batch = gateway.embed_documents(["a passage"])

        assert client.embeddings.calls == 2
        assert len(batch[0]) == 4

    def test_a_non_retryable_400_is_not_retried_and_surfaces_chinese(self, no_sleep):
        client = FakeEmbeddingClient([status_error(400), make_success(4)])
        gateway = make_gateway(dim=4, client=client, max_retries=3)

        with pytest.raises(RuntimeError) as excinfo:
            gateway.embed_documents(["a passage"])

        # Exactly one attempt: a 400 replayed is a 400, plus wasted quota.
        assert client.embeddings.calls == 1
        message = str(excinfo.value)
        assert "格式错误" in message or "400" in message
        assert any("一" <= ch <= "鿿" for ch in message)  # Chinese message

    def test_a_401_is_not_retried(self, no_sleep):
        client = FakeEmbeddingClient([status_error(401), make_success(4)])
        gateway = make_gateway(dim=4, client=client)

        with pytest.raises(RuntimeError) as excinfo:
            gateway.embed_documents(["a passage"])
        assert client.embeddings.calls == 1
        assert "认证" in str(excinfo.value)

    def test_retries_are_bounded_then_the_error_surfaces(self, no_sleep):
        client = FakeEmbeddingClient([APIConnectionError(request=_REQUEST)])
        gateway = make_gateway(dim=4, client=client, max_retries=2)

        with pytest.raises(RuntimeError):
            gateway.embed_documents(["a passage"])
        assert client.embeddings.calls == 2


# --------------------------------------------------------------------------
# 6. Cost accounting
# --------------------------------------------------------------------------


class TestCostAccounting:
    def test_a_provider_call_records_tokens_and_a_computed_cost(self):
        client = FakeEmbeddingClient([make_success(4, prompt_tokens_per_text=10)])
        gateway = make_gateway(dim=4, client=client, price_per_mtok=0.5)

        batch = gateway.embed_documents(["one", "two"])

        assert batch.usage.requests == 1
        assert batch.usage.input_tokens == 20
        assert batch.usage.cost == pytest.approx(20 * 0.5 / 1_000_000)
        assert batch.usage.cost > 0

    def test_a_cache_hit_records_zero_cost(self):
        client = FakeEmbeddingClient([make_success(4, prompt_tokens_per_text=10)])
        gateway = make_gateway(dim=4, client=client, price_per_mtok=0.5)

        gateway.embed_documents(["one"])
        hit = gateway.embed_documents(["one"])

        assert hit.usage.requests == 0
        assert hit.usage.input_tokens == 0
        # A hit genuinely costs zero -- a real 0.0, not None ("unknown").
        assert hit.usage.cost == 0.0

    def test_cost_is_none_when_no_price_is_configured(self):
        client = FakeEmbeddingClient([make_success(4)])
        gateway = make_gateway(dim=4, client=client, price_per_mtok=None)
        batch = gateway.embed_documents(["one"])
        assert batch.usage.cost is None

    def test_cost_is_none_when_the_provider_reports_no_usage(self):
        client = FakeEmbeddingClient([make_success(4, report_tokens=False)])
        gateway = make_gateway(dim=4, client=client, price_per_mtok=0.5)
        batch = gateway.embed_documents(["one"])
        assert batch.usage.tokens_reported is False
        assert batch.usage.cost is None

    def test_columns_match_the_planned_analysis_run_schema(self):
        client = FakeEmbeddingClient([make_success(4, prompt_tokens_per_text=10)])
        gateway = make_gateway(dim=4, client=client, price_per_mtok=0.5)
        batch = gateway.embed_documents(["one", "two"])

        columns = batch.usage.as_analysis_run_columns()
        assert columns["embedding_provider"] == gateway.provider
        assert columns["embedding_model"] == gateway.model
        assert columns["embedding_dimension"] == 4
        assert columns["embedding_index_version"] == gateway.index_version
        assert columns["embedding_requests"] == 1
        assert columns["embedding_texts"] == 2
        assert columns["embedding_cache_hits"] == 0
        assert columns["embedding_input_tokens"] == 20
        assert columns["embedding_cost"] == pytest.approx(1e-5)

    def test_the_gateway_accumulates_and_resets_a_running_total(self):
        client = FakeEmbeddingClient([make_success(4, prompt_tokens_per_text=10)])
        gateway = make_gateway(dim=4, client=client, price_per_mtok=0.5)

        gateway.embed_documents(["a"])
        gateway.embed_documents(["b"])
        assert gateway.usage.requests == 2
        assert gateway.usage.input_tokens == 20

        previous = gateway.reset_usage()
        assert previous.requests == 2
        assert gateway.usage.requests == 0

    def test_embed_query_shares_the_document_path(self):
        client = FakeEmbeddingClient([make_success(4)])
        gateway = make_gateway(dim=4, client=client)
        vector = gateway.embed_query("what is attention")
        assert vector == vector_for("what is attention", 4)
        assert gateway.embed_query("what is attention") == vector
        assert client.embeddings.calls == 1  # second query served from cache


# --------------------------------------------------------------------------
# 7. Transport guards
# --------------------------------------------------------------------------


class TestTransportGuards:
    def test_a_missing_api_key_leaves_the_transport_unbound(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        gateway = EmbeddingGateway("")
        assert gateway.client is None
        with pytest.raises(ValueError, match="API Key"):
            gateway.embed_documents(["a passage"])

    def test_a_wrong_item_count_is_refused(self):
        # Response returns one vector where two were asked for.
        def _short(inputs):
            return FakeEmbeddingResponse([vector_for(inputs[0], 4)], prompt_tokens=4)

        client = FakeEmbeddingClient([_short])
        gateway = make_gateway(dim=4, client=client)
        with pytest.raises(RuntimeError, match="条结果"):
            gateway.embed_documents(["a", "b"])

    def test_a_missing_embedding_field_is_refused(self):
        class BadItem:
            index = 0
            embedding = None

        class BadResponse:
            data = [BadItem()]

        client = FakeEmbeddingClient([BadResponse()])
        gateway = make_gateway(dim=4, client=client)
        with pytest.raises(RuntimeError, match="embedding"):
            gateway.embed_documents(["a"])
