"""What every tier's vector store must do, asserted once.

The contract is *cosine similarity over unit-length vectors*, expressed in
those terms rather than as a distance, so that a caller comparing a score
from the personal tier with one from the Hosted tier is comparing the same
quantity. Every expected score below is a real number the implementation has
to produce, which is the reason scoring lives in
:mod:`~paperwhisperer.infra.stores.scoring` rather than in either backend.

The failure paths get as much attention as the happy path. A vector store
that silently accepts a wrong-width embedding is a store whose search results
are confidently about a different space, and nothing in a retrieval test
would notice.

Scores are compared with a relative tolerance rather than exactly, because
the two tiers do not compute in the same precision: this tier's path is
float64 arithmetic on float32-stored values, and pgvector's distance is
float4. Anything tighter than a few parts in 10^7 would be testing the
floating-point implementation rather than the store.
"""

from __future__ import annotations

import math
import tempfile
import threading
from pathlib import Path

import pytest

from paperwhisperer.infra.stores import VectorEntry

from .test_stores_tiers import (
    TIER_HOSTED,
    TIER_PERSONAL,
    open_harness,
    requires_postgres,
)

#: Three orthonormal axes, plus the diagonal. Distinct enough that a wrong
#: ranking fails, and exactly representable in float32 so the expected
#: scores are not themselves an approximation of an approximation.
ORTHOGONAL = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
DIAGONAL = tuple(1 / math.sqrt(3) for _ in range(3))


def seed(store, vectors, *, document_id="doc-vec", start=0) -> int:
    return store.upsert_many(
        [
            VectorEntry(chunk_id=f"c{start + index}", vector=vector, document_id=document_id)
            for index, vector in enumerate(vectors)
        ]
    )


@pytest.fixture
def harness(request, tmp_path):
    """The store bundle for whichever tier the test class names.

    The class attribute is the whole mechanism: ``tier = "sqlite"`` on one
    subclass and ``tier = "hosted"`` on the other, so the two differ by one
    line and cannot drift in their assertions.
    """
    yield from open_harness(request.node.cls.tier, tmp_path)


class StoreContract:
    """Behaviour that must be identical on every tier.

    Named so that pytest's ``Test*`` collection rule leaves it alone. Only
    the two subclasses at the bottom of this file run it.
    """

    # -- storage -----------------------------------------------------------

    def test_the_index_starts_empty(self, harness):
        assert harness.vectors.count_vectors() == 0
        assert harness.vectors.dimension is None

    def test_a_vector_survives_the_round_trip(self, harness):
        harness.vectors.upsert_vector("c1", (3.0, 4.0, 0.0), document_id="doc-1", metadata={"h": "A"})

        stored = harness.vectors.get_vector("c1")
        assert stored is not None
        assert stored.document_id == "doc-1"
        assert stored.metadata == {"h": "A"}
        # Stored unit-length, so the caller can rely on the invariant and the
        # similarity search can be a dot product.
        assert stored.vector == pytest.approx((0.6, 0.8, 0.0), rel=1e-6)
        assert harness.vectors.dimension == 3

    def test_a_missing_vector_is_none(self, harness):
        assert harness.vectors.get_vector("never-written") is None

    def test_writing_the_same_chunk_twice_replaces_it(self, harness):
        harness.vectors.upsert_vector("c1", (1.0, 0.0, 0.0))
        harness.vectors.upsert_vector("c1", (0.0, 1.0, 0.0))
        assert harness.vectors.count_vectors() == 1
        assert harness.vectors.get_vector("c1").vector == pytest.approx((0.0, 1.0, 0.0), rel=1e-6)

    def test_upsert_many_reports_what_it_wrote(self, harness):
        assert seed(harness.vectors, ORTHOGONAL) == 3
        assert harness.vectors.count_vectors() == 3

    def test_an_empty_batch_writes_nothing(self, harness):
        assert harness.vectors.upsert_many([]) == 0
        assert harness.vectors.count_vectors() == 0

    # -- search ------------------------------------------------------------

    def test_a_vector_is_its_own_nearest_neighbour(self, harness):
        seed(harness.vectors, ORTHOGONAL)
        for index, vector in enumerate(ORTHOGONAL):
            hits = harness.vectors.search(vector, top_k=3)
            assert hits[0].chunk_id == f"c{index}"
            assert hits[0].score == pytest.approx(1.0, rel=1e-6)

    def test_results_are_ordered_by_similarity(self, harness):
        seed(harness.vectors, [*ORTHOGONAL, DIAGONAL])
        hits = harness.vectors.search((1.0, 0.1, 0.0), top_k=4)
        assert [hit.chunk_id for hit in hits] == ["c0", "c3", "c1", "c2"]
        scores = [hit.score for hit in hits]
        assert scores == sorted(scores, reverse=True)
        # Orthogonal to the query, so zero by definition.
        assert hits[-1].score == pytest.approx(0.0, abs=1e-6)

    def test_equal_scores_break_ties_on_the_chunk_id(self, harness):
        """``top_k`` has to be a function of the data, not of iteration order.

        Three identical vectors with ids in a deliberately unhelpful order.
        Without the tiebreak the personal tier and the Hosted tier would each
        keep a different one at the boundary, and which one they kept would
        change between runs.
        """
        for chunk_id in ("chunk-c", "chunk-a", "chunk-b"):
            harness.vectors.upsert_vector(chunk_id, (1.0, 0.0, 0.0))
        hits = harness.vectors.search((1.0, 0.0, 0.0), top_k=3)
        assert [hit.chunk_id for hit in hits] == ["chunk-a", "chunk-b", "chunk-c"]

    def test_top_k_truncates(self, harness):
        seed(harness.vectors, ORTHOGONAL)
        assert len(harness.vectors.search((1.0, 1.0, 0.0), top_k=2)) == 2
        assert harness.vectors.search((1.0, 1.0, 0.0), top_k=0) == []

    def test_a_document_filter_narrows_the_result(self, harness):
        seed(harness.vectors, ORTHOGONAL, document_id="doc-a", start=0)
        seed(harness.vectors, ORTHOGONAL, document_id="doc-b", start=10)

        assert len(harness.vectors.search((1.0, 0.0, 0.0), top_k=10)) == 6
        filtered = harness.vectors.search((1.0, 0.0, 0.0), top_k=10, document_ids=["doc-b"])
        assert [hit.chunk_id for hit in filtered] == ["c10", "c11", "c12"]
        assert {hit.document_id for hit in filtered} == {"doc-b"}
        assert harness.vectors.search((1.0, 0.0, 0.0), document_ids=[]) == []

    def test_a_score_floor_filters_weak_matches(self, harness):
        seed(harness.vectors, [*ORTHOGONAL, DIAGONAL])
        hits = harness.vectors.search((1.0, 0.0, 0.0), top_k=4, min_score=0.5)
        assert [hit.chunk_id for hit in hits] == ["c0", "c3"]
        assert harness.vectors.search((1.0, 0.0, 0.0), min_score=1.5) == []

    def test_scores_stay_inside_the_cosine_range(self, harness):
        seed(harness.vectors, ORTHOGONAL)
        hits = harness.vectors.search((-1.0, 0.0, 0.0), top_k=3)
        assert all(-1.0 <= hit.score <= 1.0 for hit in hits)
        # The antiparallel vector really does score -1.0. It is the *last*
        # result, not the first: the contract is a ranking by similarity, and
        # a negative similarity is a worse match than an orthogonal one.
        worst = [hit for hit in hits if hit.chunk_id == "c0"][0]
        assert worst.score == pytest.approx(-1.0, rel=1e-6)
        assert hits[-1].chunk_id == "c0"

    # -- deletion ----------------------------------------------------------

    def test_deleting_reports_whether_it_was_there(self, harness):
        seed(harness.vectors, ORTHOGONAL)
        assert harness.vectors.delete_vector("c1") is True
        assert harness.vectors.delete_vector("c1") is False
        assert harness.vectors.count_vectors() == 2

    def test_deleting_a_document_removes_only_its_vectors(self, harness):
        seed(harness.vectors, ORTHOGONAL, document_id="doc-a", start=0)
        seed(harness.vectors, ORTHOGONAL, document_id="doc-b", start=10)
        assert harness.vectors.delete_document("doc-a") == 3
        assert harness.vectors.count_vectors() == 3
        assert harness.vectors.get_vector("c0") is None
        assert harness.vectors.get_vector("c10") is not None

    def test_a_deleted_vector_stops_being_returned(self, harness):
        seed(harness.vectors, ORTHOGONAL)
        harness.vectors.delete_document("doc-vec")
        assert harness.vectors.search((1.0, 0.0, 0.0), top_k=3) == []

    # -- refusals ----------------------------------------------------------

    def test_a_zero_vector_is_refused(self, harness):
        # Cosine similarity against a zero vector is undefined, not zero. A
        # store that answered 0.0 would be reporting a number it did not
        # measure, and a min_score filter could not tell the difference.
        with pytest.raises(ValueError, match="零向量"):
            harness.vectors.upsert_vector("c-zero", (0.0, 0.0, 0.0))

    def test_an_empty_vector_is_refused(self, harness):
        with pytest.raises(ValueError, match="维度"):
            harness.vectors.upsert_vector("c-empty", ())

    def test_a_vector_of_the_wrong_width_is_refused(self, harness):
        seed(harness.vectors, ORTHOGONAL)
        with pytest.raises(ValueError, match="维度"):
            harness.vectors.upsert_vector("c-wide", (1.0, 0.0, 0.0, 0.0))

    def test_an_empty_chunk_id_is_refused(self, harness):
        # An id of "  " would become a real row that no document owns and no
        # query can name, and nothing would ever report it.
        with pytest.raises(ValueError, match="chunk_id"):
            harness.vectors.upsert_vector("  ", (1.0, 0.0, 0.0))
        assert harness.vectors.count_vectors() == 0

    def test_a_query_of_the_wrong_width_is_refused(self, harness):
        # Better a loud error than a search that quietly compares a truncated
        # query against full-width vectors and returns confident nonsense.
        seed(harness.vectors, ORTHOGONAL)
        with pytest.raises(ValueError, match="维度"):
            harness.vectors.search((1.0, 0.0), top_k=3)

    def test_a_query_against_an_empty_index_is_empty_not_an_error(self, harness):
        # No established dimension yet, so there is nothing to disagree with.
        assert harness.vectors.search((1.0, 0.0, 0.0), top_k=3) == []

    # -- concurrency -------------------------------------------------------

    def test_concurrent_writes_all_land(self, harness):
        workers = 8
        per_worker = 5
        errors: list[BaseException] = []
        barrier = threading.Barrier(workers)

        def _write(worker: int) -> None:
            try:
                barrier.wait(timeout=10)
                for index in range(per_worker):
                    harness.vectors.upsert_vector(
                        f"w{worker}-{index}",
                        (1.0, float(index + 1), 0.0),
                        document_id=f"doc-{worker}",
                    )
            except BaseException as exc:  # noqa: BLE001 - reported below
                errors.append(exc)

        threads = [threading.Thread(target=_write, args=(worker,)) for worker in range(workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)

        assert errors == []
        assert harness.vectors.count_vectors() == workers * per_worker
        assert len(harness.vectors.search((1.0, 1.0, 0.0), top_k=100)) == workers * per_worker

    def test_initialize_is_idempotent(self, harness):
        seed(harness.vectors, ORTHOGONAL)
        harness.vectors.initialize()
        harness.vectors.initialize()
        assert harness.vectors.count_vectors() == 3


class TestVectorSqlite(StoreContract):
    tier = TIER_PERSONAL


@requires_postgres
class TestVectorPostgres(StoreContract):
    tier = TIER_HOSTED


class TestVectorPersonalTierOnly:
    """What the personal tier can do that the contract does not promise.

    Each test is named for the tier. None of this is a contract test, and
    putting it in the shared suite would be the drift this arrangement
    exists to prevent: the Hosted tier has no
    ``pytest.importorskip("numpy")``, and a shared assertion that needed it
    would fail there.
    """

    def test_the_numpy_path_and_the_fallback_agree_exactly(self, monkeypatch):
        """Two code paths, one answer.

        numpy is a soft dependency, so a build without it has to still work.
        "Still works" is a claim about behaviour, not availability, so it is
        checked by comparing the rankings and the scores the two paths
        produce for the same index.
        """
        from paperwhisperer.infra.stores import sqlite_backend

        from .test_stores_tiers import open_bundle

        vectors = [ORTHOGONAL[0], ORTHOGONAL[1], DIAGONAL, (0.6, 0.8, 0.0)]

        def _run() -> list:
            with tempfile.TemporaryDirectory() as directory:
                harness = open_bundle(TIER_PERSONAL, Path(directory))
                try:
                    seed(harness.vectors, vectors)
                    return harness.vectors.search((1.0, 0.2, 0.0), top_k=4)
                finally:
                    harness.close()

        with_numpy = _run()
        monkeypatch.setattr(sqlite_backend, "_numpy", None)
        without_numpy = _run()

        assert [hit.chunk_id for hit in with_numpy] == [hit.chunk_id for hit in without_numpy]
        for fast, slow in zip(with_numpy, without_numpy, strict=True):
            assert fast.score == pytest.approx(slow.score, rel=1e-9)

    def test_the_personal_tier_needs_no_driver_or_service(self, tmp_path):
        """The acceptance criterion, stated as an assertion.

        The Hosted tier raises on construction without a driver; the personal
        tier must construct, initialize, write, read and close with nothing
        else in the process.
        """
        from paperwhisperer.infra.stores import build_stores

        from .test_stores_tiers import _Settings, sample_record

        bundle = build_stores(_Settings(STORAGE_SQLITE_PATH=str(tmp_path / "solo.db")))
        try:
            bundle.metadata.upsert_document(sample_record("doc-solo"))
            assert bundle.metadata.get_document("doc-solo") is not None
        finally:
            bundle.close()
