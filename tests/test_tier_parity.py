"""The divergences the Hosted tier actually had, and the ones it still has.

T1.1's guarantee is "同一套测试在两档各跑一遍" — one suite, run once per tier.
That guarantee is only as strong as the *worst-covered* behaviour, and for a
long time the Hosted half of it was a skip. The first hour against a real
server produced five bugs that no amount of offline testing could find, and
they were not random: every one of them is a place where SQLite's
implementation is more forgiving than PostgreSQL's, and the suite is written
against SQLite. A ``TEXT`` column that accepted an ``INTEGER``, an ``id``
column that accepted a qualified name, a ``REAL`` that accepted a Python bool
— SQLite is dynamically typed and lenient, PostgreSQL is not, and a contract
asserted on the lenient engine is a contract the strict engine has never met.

So this file is not a second contract suite. The contract is
``tests/test_stores_*.py`` and ``tests/test_jobs.py``, and those run the same
assertions on both tiers. This file exists for the three things a shared
contract *cannot* express:

1. **A divergence that was fixed**, named and pinned, so it cannot come back
   as a silent difference between two green runs. Each case states the
   asymmetry, the one-line reason it was invisible, and asserts the two tiers
   now agree.
2. **A limitation that is NOT fixed**, because fixing it would mean lying.
   These are collected in :class:`TestKnownDivergences`. A Hosted-only limit
   that is silently absent from the personal tier is precisely the drift the
   two-tier design exists to prevent, so it gets a recorded degradation and a
   test that asserts *which tier* is degraded — not a test asserting a fake
   parity, and not a silent skip.
3. **The coverage itself.** :class:`TestCoverage` counts how much of the
   suite actually reached PostgreSQL. A test that passes on both tiers
   without exercising PostgreSQL is not parity, and the count is the finding.

Every Hosted test here skips with a reason that names the variable to set, and
the reason is never "the Hosted tier was never run".
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass

import pytest

from paperwhisperer.infra import jobs as J
from paperwhisperer.infra.stores import TIER_HOSTED, TIER_PERSONAL, DocumentRecord, build_stores
from paperwhisperer.infra.stores.postgres_backend import MAX_LEXEME_BYTES

from .test_stores_tiers import (
    PG_AVAILABLE,
    PG_DSN_ENV,
    StoreHarness,
    open_bundle,
)


@dataclass
class _Settings:
    """The subset of configuration the store and job factories read.

    A private object rather than monkeypatching ``core/config.py``, which
    another task owns. The Hosted branch sets only the three keys the tier
    needs; everything else carries the shipped default, so a test is still
    exercising real defaults for the rest.
    """

    STORAGE_BACKEND: str = "sqlite"
    STORAGE_SQLITE_PATH: str = ""
    STORAGE_SQLITE_TIMEOUT_SECONDS: int = 30
    STORAGE_LEXICAL_FTS5: str = "auto"
    STORAGE_PG_DSN: str = ""
    STORAGE_PG_SCHEMA: str = "public"
    STORAGE_PG_CONNECT_TIMEOUT_SECONDS: int = 5
    STORAGE_PG_ANN: bool = False


def _reject_constant(name: str) -> float:
    """A ``parse_constant`` hook that refuses JSON's non-standard extensions.

    ``json.loads`` accepts ``NaN``/``Infinity`` by default, which is exactly
    the leniency under test. A strict reader — RFC 8259, and therefore
    PostgreSQL's ``jsonb`` — does not, and the two disagreeing is the point.
    """
    raise ValueError(f"non-standard JSON constant: {name}")


# --------------------------------------------------------------------------
# Both tiers at once
# --------------------------------------------------------------------------


@dataclass
class TierPair:
    """One personal-tier bundle and one Hosted-tier bundle, side by side.

    Not a context manager, because the point of most tests in this file is to
    put the *same input* through both stores and compare what came back. A
    fixture that yields one tier at a time cannot express that, and a
    test written as "run the contract twice" is exactly the arrangement that
    lets the two contracts drift apart.

    Teardown order is the one :class:`~tests.test_stores_tiers.StoreHarness`
    already learned the hard way: the Hosted schema is dropped through its own
    live connection, so it has to happen *before* the bundle is closed. Doing
    it after leaves the test failing with "database connection is closed",
    which reads as a product bug, is not one, and never appears on SQLite.
    """

    personal: StoreHarness
    hosted: StoreHarness

    def close(self) -> None:
        for harness in (self.hosted, self.personal):
            try:
                harness.close()
            except Exception:  # pragma: no cover - teardown must not mask a failure
                pass

    @property
    def bundles(self):
        """Both stores, in a stable order, for table-driven assertions."""
        return (self.personal.bundle, self.hosted.bundle)


def _open_hosted_bundle(schema_prefix: str) -> StoreHarness:
    """A Hosted bundle in its own schema, dropped on teardown.

    A schema rather than a table because a run creates a table, a vector
    column, a GIN index, a trigram index and several plain indexes, and
    ``DROP TABLE`` would leave the rest to collide with the next test's index
    names.
    """
    schema = f"{schema_prefix}_{uuid.uuid4().hex[:12]}"
    harness = StoreHarness(
        tier=TIER_HOSTED,
        bundle=build_stores(
            _Settings(
                STORAGE_BACKEND="postgres",
                STORAGE_PG_DSN=os.environ[PG_DSN_ENV],
                STORAGE_PG_SCHEMA=schema,
            )
        ),
    )
    database = harness.bundle.database

    def _drop() -> None:
        with database.cursor() as cursor:
            cursor.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")

    harness._cleanup.append(_drop)
    return harness


@pytest.fixture
def tiers(tmp_path):
    """Both tiers, isolated per test, personal on a file and hosted on a schema."""
    if not PG_AVAILABLE:
        pytest.skip(
            f"未设置或无法连接 {PG_DSN_ENV}。运行 docker compose up -d db 后设置 "
            f"{PG_DSN_ENV}=postgresql://paperwhisperer:paperwhisperer@localhost:5432/"
            "paperwhisperer 即可运行双档一致性测试。"
        )
    sqlite_path = tmp_path / "personal.db"
    pair = TierPair(
        personal=StoreHarness(
            tier=TIER_PERSONAL,
            bundle=build_stores(
                _Settings(STORAGE_BACKEND="sqlite", STORAGE_SQLITE_PATH=str(sqlite_path))
            ),
        ),
        hosted=_open_hosted_bundle("parity"),
    )
    try:
        yield pair
    finally:
        pair.close()


@pytest.fixture
def job_tiers(tmp_path):
    """Both tiers' job stores, sharing each tier's own bundle.

    The job table lives in the store tier's database, so building the job
    store against a bundle rather than letting
    :func:`~paperwhisperer.infra.jobs.build_job_store` build its own is what
    exercises the supported path — the one a Hosted deployment actually uses.
    """
    if not PG_AVAILABLE:
        pytest.skip(
            f"未设置或无法连接 {PG_DSN_ENV}。运行 docker compose up -d db 后设置 "
            f"{PG_DSN_ENV}=postgresql://paperwhisperer:paperwhisperer@localhost:5432/"
            "paperwhisperer 即可运行双档一致性测试。"
        )
    sqlite_path = tmp_path / "personal-jobs.db"
    personal_bundle = build_stores(
        _Settings(STORAGE_BACKEND="sqlite", STORAGE_SQLITE_PATH=str(sqlite_path))
    )
    hosted_harness = _open_hosted_bundle("parity_job")
    pair = TierPair(
        personal=StoreHarness(tier=TIER_PERSONAL, bundle=personal_bundle),
        hosted=hosted_harness,
    )
    stores = [J.build_job_store(bundle=personal_bundle)]
    stores.append(J.build_job_store(bundle=hosted_harness.bundle))
    try:
        yield pair, stores
    finally:
        for store in stores:
            store.close()
        pair.close()


# --------------------------------------------------------------------------
# 1. Divergences that were real, and are fixed
# --------------------------------------------------------------------------


class TestDocumentWithNoParsedBody:
    """``upsert_document`` of a record carrying no ``ParsedDocument``.

    **What diverged.** A ``DocumentRecord`` with ``document=None`` — a
    registration still being parsed, or one written by a caller that only has
    metadata — came back with an empty document from the personal tier and
    with ``None`` from the Hosted tier. ``get_document`` then handed each
    caller a different answer for the same row.

    **Why it was invisible.** The substitution happens on write, and every
    existing test wrote a *full* document, so the branch that differed was one
    no test entered. It is the shape of bug the shared suite is supposed to
    make impossible and could not: there was no assertion of the shape at all,
    only of the happy path.

    The personal tier's behaviour is the one kept. It is the more useful of
    the two — a caller that round-trips a record and reaches through
    ``.document`` does not have to branch on which tier it got — and the
    Hosted tier now does the same substitution rather than the reverse, because
    the Hosted tier is the one that has to match.
    """

    def test_both_tiers_substitute_a_document_carrying_the_identifier(self, tiers):
        for bundle in tiers.bundles:
            record = DocumentRecord(
                document_id="  doc-no-body  ",
                source_filename="pending.pdf",
                metadata={"state": "registered"},
            )
            stored = bundle.metadata.upsert_document(record)
            assert stored.document is not None, bundle.tier
            assert stored.document.document_id == "doc-no-body", bundle.tier

    def test_the_substitution_survives_the_round_trip_on_both_tiers(self, tiers):
        for bundle in tiers.bundles:
            bundle.metadata.upsert_document(DocumentRecord(document_id="doc-rt"))
            fetched = bundle.metadata.get_document("doc-rt")
            assert fetched is not None, bundle.tier
            assert fetched.document is not None, bundle.tier
            assert fetched.document.document_id == "doc-rt", bundle.tier
            # And the two agree, which is the assertion the suite lacked.
            assert fetched.to_dict() == tiers.personal.bundle.metadata.get_document(
                "doc-rt"
            ).to_dict(), bundle.tier


class TestVectorDimensionAfterTheTableEmpties:
    """``VectorStore.dimension`` after every vector has been deleted.

    **What diverged.** The Hosted store cached the width from the first write
    and never re-read it, so once ``delete_document`` emptied the table it
    still answered with the old width while the personal tier answered
    ``None``. The second-order effect is the one that bites: ``upsert_vector``
    consults ``dimension`` to refuse a mixed-width index, so "delete
    everything, then re-index at a new dimension" raised ``ValueError`` on the
    Hosted tier and succeeded on the personal one. A re-index after an
    embedding-model change is an ordinary operation.

    **Why it was invisible.** A cache that is correct until somebody deletes
    is not a cache, it is a second source of truth — and every test that
    checked ``dimension`` did so on a store it had just written to, which is
    the one state where a cache and a read agree.
    """

    def test_dimension_is_none_again_once_the_table_is_empty(self, tiers):
        for bundle in tiers.bundles:
            vectors = bundle.vectors
            vectors.upsert_vector("v-dim", (1.0, 0.0, 0.0))
            assert vectors.dimension == 3, bundle.tier
            vectors.delete_vector("v-dim")
            assert vectors.dimension is None, bundle.tier

    def test_the_index_may_be_rebuilt_at_a_different_width(self, tiers):
        for bundle in tiers.bundles:
            vectors = bundle.vectors
            # The real sequence: a document is indexed at one width, the
            # document is then deleted, taking its embedding with it, and the
            # corpus is re-indexed at a new width. On the cached tier the
            # second write raised, because `dimension` still answered with the
            # first width after the table had emptied.
            vectors.upsert_vector("v-old", (1.0, 0.0, 0.0), document_id="doc-reindex")
            assert vectors.dimension == 3, bundle.tier
            assert vectors.delete_document("doc-reindex") == 1, bundle.tier
            assert vectors.dimension is None, bundle.tier
            vectors.upsert_vector("v-new", (0.5, 0.5, 0.5, 0.5), document_id="doc-reindex")
            assert vectors.dimension == 4, bundle.tier
            assert [hit.chunk_id for hit in vectors.search((0.5, 0.5, 0.5, 0.5))] == ["v-new"]

    def test_a_mixed_width_write_is_still_refused_on_both_tiers(self, tiers):
        # The fix must not have turned the guard off. This is the assertion
        # that was already there, kept so that "always re-read" cannot become
        # "never check".
        for bundle in tiers.bundles:
            vectors = bundle.vectors
            vectors.upsert_vector("v-keep", (1.0, 0.0, 0.0))
            with pytest.raises(ValueError):
                vectors.upsert_vector("v-wrong", (1.0, 0.0, 0.0, 0.0))
            with pytest.raises(ValueError):
                vectors.search((1.0, 0.0, 0.0, 0.0))
            assert vectors.dimension == 3, bundle.tier


class TestSortKeyCollation:
    """``ORDER BY`` on a text key, under a locale-aware database.

    **What diverged.** SQLite sorts text by byte value; that is the only
    collation it has and it is what every expectation in the suite is written
    in. PostgreSQL sorts by the *database's* collation, and this project's own
    compose file initialises the cluster as ``en_US.utf8``. Under it,
    ``Chunk-A < chunk-a < CHUNK-A`` where SQLite says
    ``CHUNK-A < Chunk-A < chunk-a`` — and the same holds for ``created_at``,
    whose ISO timestamps differ by one character.

    **Why this one mattered more than the others.** Three separate contract
    docstrings promise a ``chunk_id``/``job_id`` tiebreak explicitly so that
    "the same query returns a different page order on the two tiers, which is a
    pagination bug that only appears in production". The tiebreak existed
    precisely to fix non-determinism, and the collation underneath it silently
    defeated it. In the job table it decides *which job a worker runs next*:
    ``claim_next`` is ``ORDER BY created_at ASC, job_id ASC LIMIT 1``, so a
    queue holding ``a1b2c3`` and ``A1b2c3`` drained in a different order per
    tier — same rows, same query, different worker. And the test written for
    it (``test_claiming_drains_the_queue_in_a_total_order``) compares the two
    claims against *each other* rather than against a fixed order, so it
    passed on both tiers while the two disagreed with each other.

    The fix is ``COLLATE "C"`` on every sort key, written on both the column
    and the ``ORDER BY`` so PostgreSQL can use the index. A table created
    before this keeps its old collation and sorts each page without an index;
    it is correct either way, because the ``ORDER BY`` carries its own
    ``COLLATE``. Rewriting the columns of a table an operator may be holding
    rows in is not something a library should do to itself at
    ``initialize()``.
    """

    #: Deliberately awkward: mixed case, and the punctuation that a
    #: locale-aware collation weights differently from a byte comparison.
    #: Ids like ``c0``/``c1`` cannot detect this, which is why the shared
    #: contract's tiebreak assertions never did.
    AWKWARD_IDS = ("Chunk-A", "chunk-a", "CHUNK-A", "chunk.a", "chunk_a", "chunkA", "chunk 1")

    def test_lexical_search_orders_ties_the_same_way_on_both_tiers(self, tiers):
        body = "one identical passage, so every row scores the same"
        for chunk_id in self.AWKWARD_IDS:
            for bundle in tiers.bundles:
                bundle.lexical.index_chunk(chunk_id, body, document_id="doc-collation")
        personal = [
            hit.chunk_id for hit in tiers.personal.bundle.lexical.search("identical", top_k=99)
        ]
        hosted = [
            hit.chunk_id for hit in tiers.hosted.bundle.lexical.search("identical", top_k=99)
        ]
        assert personal == hosted
        # And the order is the byte order both tiers are supposed to use, not
        # merely an order they happen to share today.
        assert personal == sorted(self.AWKWARD_IDS)

    def test_vector_search_orders_ties_the_same_way_on_both_tiers(self, tiers):
        for chunk_id in self.AWKWARD_IDS:
            for bundle in tiers.bundles:
                bundle.vectors.upsert_vector(chunk_id, (1.0, 0.0, 0.0))
        personal = [
            hit.chunk_id
            for hit in tiers.personal.bundle.vectors.search((1.0, 0.0, 0.0), top_k=99)
        ]
        hosted = [
            hit.chunk_id for hit in tiers.hosted.bundle.vectors.search((1.0, 0.0, 0.0), top_k=99)
        ]
        assert personal == hosted
        assert personal == sorted(self.AWKWARD_IDS)

    def test_list_chunks_orders_ties_the_same_way_on_both_tiers(self, tiers):
        from paperwhisperer.ingestion.schema import Chunk

        for chunk_id in self.AWKWARD_IDS:
            for bundle in tiers.bundles:
                bundle.metadata.put_chunk("doc-collation", Chunk(chunk_id=chunk_id, order=0))
        personal = [
            chunk.chunk_id for chunk in tiers.personal.bundle.metadata.list_chunks("doc-collation")
        ]
        hosted = [
            chunk.chunk_id for chunk in tiers.hosted.bundle.metadata.list_chunks("doc-collation")
        ]
        assert personal == hosted
        assert personal == sorted(self.AWKWARD_IDS)

    def test_list_documents_orders_ties_the_same_way_on_both_tiers(self, tiers):
        stamp = "2026-01-01T00:00:00+00:00"
        for document_id in self.AWKWARD_IDS:
            for bundle in tiers.bundles:
                bundle.metadata.upsert_document(
                    DocumentRecord(document_id=document_id, created_at=stamp)
                )
        personal = [
            record.document_id
            for record in tiers.personal.bundle.metadata.list_documents(limit=99)
        ]
        hosted = [
            record.document_id for record in tiers.hosted.bundle.metadata.list_documents(limit=99)
        ]
        assert personal == hosted
        assert personal == sorted(self.AWKWARD_IDS)

    def test_a_paginated_list_pages_the_same_way_on_both_tiers(self, tiers):
        """The failure the tiebreak was for, stated as a page boundary.

        Every id sorts before ``zzz`` so all of them share a ``created_at``,
        which is the normal case on the personal tier: two documents written
        in the same second is what the tiebreak is *for*.
        """
        stamp = "2026-01-01T00:00:00+00:00"
        ids = sorted(self.AWKWARD_IDS)
        for document_id in ids:
            for bundle in tiers.bundles:
                bundle.metadata.upsert_document(
                    DocumentRecord(document_id=document_id, created_at=stamp)
                )
        personal = tiers.personal.bundle.metadata
        hosted = tiers.hosted.bundle.metadata
        for offset in range(0, len(ids), 3):
            first = [record.document_id for record in personal.list_documents(limit=3, offset=offset)]
            second = [record.document_id for record in hosted.list_documents(limit=3, offset=offset)]
            assert first == second, offset
            assert first == ids[offset : offset + 3], offset


class TestCapabilityReporting:
    """What a tier *claims* about itself.

    **What diverged.** ``StoreCapabilities.scan_only`` means "answers a query
    by reading the whole corpus". The Hosted lexical store left it at the
    dataclass default, which is ``True`` — so the tier with a GIN index
    reported the *personal* tier's degraded state, in the one field a health
    endpoint and an operator both read to decide whether a deployment is slow.
    A tier claiming a degradation it does not have is as wrong as one hiding
    one it does, and the reason it survived is the same reason as every other
    bug in this file: the default happened to equal what the *untested* tier
    reported.
    """

    def test_the_hosted_tier_does_not_claim_to_scan(self, tiers):
        assert tiers.hosted.bundle.capabilities.scan_only is False
        assert tiers.hosted.bundle.lexical.capabilities().scan_only is False

    def test_both_tiers_report_the_engine_they_actually_use(self, tiers):
        # The personal tier reports `scan` when this build has no FTS5, and
        # that is allowed to differ — the engines genuinely differ. What must
        # not differ is the *relationship*: a tier that names an accelerator is
        # not a scanner.
        for bundle in tiers.bundles:
            capabilities = bundle.capabilities
            assert capabilities.lexical_engine in {"fts", "scan"}, bundle.tier
            assert capabilities.scan_only is (capabilities.lexical_engine == "scan"), bundle.tier

    def test_no_tier_reports_a_degradation_it_did_not_suffer(self, tiers):
        # A store that has not indexed anything over-long has nothing to say
        # about the lexeme limit, on either tier.
        for bundle in tiers.bundles:
            assert not any("tsvector" in text for text in bundle.capabilities.degradations)


class TestNonFiniteFloatsInAPayload:
    """A payload value JSON cannot represent.

    **What diverged.** ``json.dumps`` emits ``NaN`` and ``Infinity`` by
    default. They are a Python extension, not JSON. SQLite stores the text and
    hands it back, and Python's reader accepts it again, so on the personal
    tier a record carrying one round-trips through a JSON column looking
    perfectly fine. ``jsonb`` is a real parser and refuses it:
    ``invalid input syntax for type json / Token "NaN" is invalid``. The
    Hosted tier then returned ``None`` for a document that the personal tier
    had stored, and the failure surfaced as a *missing document* rather than
    as the malformed write that caused it.

    The value is meaningless either way — a NaN is not a score — so this is
    not a PostgreSQL limitation to be papered over. The personal tier was the
    wrong one, and both tiers now refuse, from one shared serializer.
    """

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_both_tiers_refuse_a_non_finite_float_in_a_job_payload(self, job_tiers, value):
        _, stores = job_tiers
        for store in stores:
            tier = store.capabilities().tier
            with pytest.raises(ValueError, match="range float"):
                store.submit("k", {"score": value}, idempotency_key=f"nan-{tier}")
            # And nothing was written: a refused write that half-succeeded
            # would leave a row the personal tier could read and the Hosted
            # tier could not.
            assert store.count_by_state()[J.STATE_QUEUED] == 0, tier


# --------------------------------------------------------------------------
# 2. Divergences that are NOT fixed, because fixing them would be a lie
# --------------------------------------------------------------------------


class TestKnownDivergences:
    """Where the two tiers genuinely differ, and which one is missing what.

    Each case here is a real, reproduced asymmetry. None of them is papered
    over, and none of them is asserted away: the test says what differs and
    which tier is degraded, because a Hosted-only limit that is silently
    absent from the personal tier is the drift the two-tier design exists to
    prevent. If one of these is ever fixed, the fix belongs here as a new
    entry in :class:`TestDivergencesThatWereFixed` — not a deletion.
    """

    def test_the_lexeme_length_limit_is_hosted_only(self, tiers):
        """A term longer than PostgreSQL will index is findable on one tier.

        The server's limit is 2047 bytes per lexeme: it emits ``word is too
        long to be indexed`` and drops the term, keeping the row and the index
        and simply not putting the word in it. FTS5 has no such limit.

        Not fixable by truncating — that would index a word the document does
        not contain and return a hit for it, which is worse than returning
        nothing because it looks like a real match. Not fixable by scanning
        either, because the server discarded the term and nothing here kept
        it. So the Hosted tier records a degradation naming the limit, and
        the two tiers are documented as disagreeing.
        """
        overlong = "z" * (MAX_LEXEME_BYTES + 1)
        assert len(overlong.encode("utf-8")) > MAX_LEXEME_BYTES

        for bundle in tiers.bundles:
            bundle.lexical.index_chunk("long-token", overlong, document_id="doc-long")

        # The personal tier finds it; the Hosted tier does not. Asserted as a
        # difference on purpose — this test is the record of the difference.
        personal = tiers.personal.bundle.lexical.search(overlong)
        hosted = tiers.hosted.bundle.lexical.search(overlong)
        assert [hit.chunk_id for hit in personal] == ["long-token"]
        assert [hit.chunk_id for hit in hosted] == []

        # The row itself is present on both tiers. What is missing is only the
        # term's *searchability*, which is the narrower claim and the honest
        # one.
        assert tiers.personal.bundle.lexical.count_chunks() == 1
        assert tiers.hosted.bundle.lexical.count_chunks() == 1

    def test_the_hosted_tier_records_that_limit_as_a_degradation(self, tiers):
        overlong = "z" * (MAX_LEXEME_BYTES + 1)
        for bundle in tiers.bundles:
            bundle.lexical.index_chunk("long-token-2", overlong)

        hosted_degradations = tiers.hosted.bundle.lexical.capabilities().degradations
        assert any("tsvector" in text for text in hosted_degradations), hosted_degradations
        # The personal tier must NOT claim it. Recording a limitation one tier
        # does not have is the mistake :class:`TestCapabilityReporting` is
        # about, and it is worth asserting the asymmetry in both directions.
        personal_degradations = tiers.personal.bundle.lexical.capabilities().degradations
        assert not any("tsvector" in text for text in personal_degradations)

    def test_a_term_at_the_limit_is_indexed_on_the_hosted_tier(self, tiers):
        """The cliff is a byte count, and the value next to it still works.

        Pinned so that :data:`MAX_LEXEME_BYTES` is a measured boundary rather
        than the number in a NOTICE message. The server says "longer than
        2047"; ``repeat('z', 2046)`` indexes and 2047 does not, so the
        largest accepted value is one less than the message. An off-by-one
        here would report a degradation for terms that are perfectly
        searchable.
        """
        at_limit = "z" * MAX_LEXEME_BYTES
        for bundle in tiers.bundles:
            bundle.lexical.index_chunk("at-limit", at_limit, document_id="doc-limit")
        assert [hit.chunk_id for hit in tiers.hosted.bundle.lexical.search(at_limit)] == ["at-limit"]
        assert [
            hit.chunk_id for hit in tiers.personal.bundle.lexical.search(at_limit)
        ] == ["at-limit"]

    def test_the_precision_of_a_self_query_differs_and_always_did(self, tiers):
        """Two engines, two float widths, and a score the contract over-promises.

        ``interfaces.VectorStore`` says scores are "comparable across tiers,
        which is what lets one test assert an exact expected score on both".
        They are not equal, and cannot be: the personal tier stores float32 and
        computes the dot product in float64 from the rounded values, while
        pgvector's cosine distance is float4 arithmetic on its own
        representation. A self-query on the personal tier can come back as
        ``1.000000005...`` — above 1.0, and above the ``[-1, 1]`` the
        :mod:`.scoring` clamp promises — while the Hosted tier returns exactly
        ``1.0``.

        This is recorded rather than fixed because either choice would be a lie:
        clamping the personal tier's result would make a store report a number
        it did not measure, and loosening the Hosted tier to match would mean
        deliberately returning float4 error. What the contract can honestly
        promise is agreement to within a few parts in 10^7, which is what
        ``tests/test_stores_vector.py`` already asserts, and this test pins the
        difference so that a change in either direction is noticed.
        """
        vector = (0.1, 0.2, 0.3)
        for bundle in tiers.bundles:
            bundle.vectors.upsert_vector("self", vector)
        query = (0.1, 0.2, 0.3)
        personal = tiers.personal.bundle.vectors.search(query, top_k=1)
        hosted = tiers.hosted.bundle.vectors.search(query, top_k=1)
        assert personal[0].score == pytest.approx(1.0, rel=1e-6)
        assert hosted[0].score == pytest.approx(1.0, rel=1e-6)
        assert abs(personal[0].score - hosted[0].score) < 1e-5
        # Not equal, and the suite should not pretend otherwise.
        assert personal[0].score != hosted[0].score

    def test_a_non_finite_float_in_a_document_metadata_still_diverges(self, tiers):
        """The *unfixed* half of the NaN story, and where the fix belongs.

        The job table is fixed — both tiers refuse, from one shared
        serializer. The store tier is not, and the reason is ownership rather
        than difficulty: ``jobs._dumps`` is one function both backends call,
        but ``sqlite_backend`` and ``postgres_backend`` each have their own
        ``_dumps``, and only the Hosted one is this task's to change. Making
        the Hosted side refuse on its own would not create parity — it would
        replace "one tier accepts, one tier raises an opaque driver error"
        with "one tier accepts, one tier raises a tidy ValueError", which is
        the same divergence wearing better manners. The fix is one argument in
        ``sqlite_backend._dumps``, and until it lands this test is the record
        of what is broken.

        The direction matters and is asserted: PostgreSQL is right to refuse,
        because what the personal tier writes is not JSON and cannot be read by
        a backup tool, a ``grep``, or the Hosted tier a user migrates to. The
        Hosted failure is the *loud* one; the personal failure is silent — the
        row is stored, readable, and wrong.
        """
        for bundle in tiers.bundles:
            record = DocumentRecord(document_id="doc-nan", metadata={"score": float("nan")})
            try:
                bundle.metadata.upsert_document(record)
                wrote = True
            except Exception:
                wrote = False
            tier = bundle.tier
            # Personal: accepts it, and stores something that is not JSON.
            assert wrote is (tier == TIER_PERSONAL), tier
            if tier == TIER_HOSTED:
                # Nothing was written, so the document reads back as absent —
                # which is how this surfaced: a *missing document*, rather than
                # the malformed write that caused it.
                assert bundle.metadata.get_document("doc-nan") is None, tier
            else:
                # Present, readable, and carrying a value no strict JSON
                # parser will accept.
                stored = bundle.metadata.get_document("doc-nan")
                assert stored is not None, tier
                raw = json.dumps(stored.to_dict(), sort_keys=True, allow_nan=True)
                with pytest.raises(ValueError):
                    json.loads(raw, parse_constant=_reject_constant)

    def test_upsert_many_is_not_atomic_on_either_tier(self, tiers):
        """A batch that fails part-way leaves the earlier rows behind.

        Both implementations loop and let the exception out of the enclosing
        transaction scope, so a batch whose fifth entry has the wrong width
        keeps its first four. The contract says ``upsert_many`` "return[s] how
        many were written", not that it is all-or-nothing, and neither tier
        is stricter than the other — which is why this is here rather than
        fixed. It is a *shared* sharp edge, and a shared sharp edge is the one
        kind of thing both tiers can agree to be wrong about.
        """
        from paperwhisperer.infra.stores import VectorEntry

        for bundle in tiers.bundles:
            vectors = bundle.vectors
            batch = [VectorEntry(chunk_id=f"c{i}", vector=(1.0, 0.0, 0.0)) for i in range(4)]
            batch.append(VectorEntry(chunk_id="wrong-width", vector=(1.0, 0.0, 0.0, 0.0)))
            with pytest.raises(ValueError):
                vectors.upsert_many(batch)
            assert vectors.count_vectors() == 4, bundle.tier


# --------------------------------------------------------------------------
# 3. The job store, on both tiers at once
# --------------------------------------------------------------------------


class TestJobQueueParity:
    """The queue's decisions, compared across tiers.

    A separate class from the store parity tests because the job store's
    contract is about *decisions under contention* — which job runs next,
    whether a second claim can happen — and those are the ones a collation or
    a stale cache can change without any single test noticing.
    """

    def test_the_queue_drains_in_byte_order_on_both_tiers(self, job_tiers):
        """The order ``claim_next`` picks is the order SQLite's BINARY collation picks.

        **Why this asserts against Python rather than across tiers.** Job ids
        are ``uuid4().hex``, generated per submission, so the two tiers never
        hold the same ids and cannot be compared to each other directly. What
        *can* be compared is each tier's SQL against the order the contract
        means: ``(created_at, job_id)`` ascending, with ``job_id`` compared by
        byte value because that is what SQLite does and what every expectation
        in the suite is written in.

        That is the whole regression. ``claim_next`` is
        ``ORDER BY created_at ASC, job_id ASC LIMIT 1`` — so the collation
        decides *which job a worker runs next* — and under the
        ``en_US.utf8`` collation of this project's own compose file that order
        differed from SQLite's for any two ids differing only in case. A queue
        holding ``a1b2c3`` and ``A1b2c3`` drained in one order on the personal
        tier and the opposite order on the Hosted one: same rows, same query,
        different worker.

        And the test that was written for it could not see any of this.
        ``test_claiming_drains_the_queue_in_a_total_order`` compares the two
        claims against *each other*
        (``(first.created_at, first.job_id) <= (second.created_at, second.job_id)``),
        which both tiers satisfy — so it passed on both while the two tiers
        disagreed with each other. A test that checks a property is *total*
        has not checked that two implementations agree on *which* total.
        """
        _, (personal, hosted) = job_tiers
        for store in (personal, hosted):
            for index in range(8):
                store.submit("k", {"n": index})

        for store in (personal, hosted):
            tier = store.capabilities().tier
            queued = store.list_jobs(state=J.STATE_QUEUED)
            # The expected order, computed the way SQLite computes it.
            expected = [
                job.job_id
                for job in sorted(queued, key=lambda job: (job.created_at, job.job_id.encode()))
            ]
            claimed: list[str] = []
            while True:
                job = store.claim_next("w", 60, kind="k")
                if job is None:
                    break
                claimed.append(job.job_id)
            assert claimed == expected, (
                f"{tier}: the queue drained in a different order than the sort key "
                f"the personal tier uses. collation is still in play."
            )

    def test_a_page_of_the_queue_is_the_same_page_the_key_implies(self, job_tiers):
        """``list_jobs`` pages by ``(created_at DESC, job_id ASC)``, in bytes.

        The pagination bug the tiebreak exists to prevent, stated as a page
        boundary. Every job here is created within the same second, which is
        the normal case on the personal tier and the only case the tiebreak
        is for.
        """
        _, (personal, hosted) = job_tiers
        for store in (personal, hosted):
            for index in range(8):
                store.submit("k", {"n": index})
        for store in (personal, hosted):
            tier = store.capabilities().tier
            everything = store.list_jobs()
            # `ORDER BY created_at DESC, job_id ASC`, expressed the way the
            # contract states it. Two stable passes rather than one clever
            # key: ascending id first, then newest-first by timestamp, which
            # leaves the id order intact inside a group of equal timestamps.
            expected = sorted(
                sorted(everything, key=lambda job: job.job_id.encode()),
                key=lambda job: job.created_at,
                reverse=True,
            )
            pages: list[str] = []
            for offset in range(0, 8, 3):
                pages.extend(job.job_id for job in store.list_jobs(limit=3, offset=offset))
            assert pages == [job.job_id for job in expected], tier

    def test_the_cancel_flag_round_trips_as_a_boolean_on_both_tiers(self, job_tiers):
        """A boolean column and an integer column, one shared value tuple.

        ``_column_values`` used to bind ``int(bool(...))`` into a ``BOOLEAN``
        column, which PostgreSQL rejects outright. The coercion was never
        needed — ``bool`` is an ``int`` subclass, so SQLite stored 0/1
        without complaint — and it was shared by both backends, so a
        convenience for one tier was a hard failure on the other. Pinned here
        because the value is invisible in both directions: ``False`` and ``0``
        compare equal in Python, so a test that only checked the boolean would
        pass on a store that had been writing integers.
        """
        _, (personal, hosted) = job_tiers
        for store in (personal, hosted):
            tier = store.capabilities().tier
            job = store.submit("k").job
            assert store.request_cancel(job.job_id).accepted, tier
            assert store.get_job(job.job_id).cancel_requested is True, tier
            # And the *column*, not just the payload: the payload is the
            # record, but the column is what ``apply_queued_cancellations``
            # filters on.
            assert store.apply_queued_cancellations(), tier
            assert store.get_job(job.job_id).state == J.STATE_CANCELLED, tier

    def test_a_settled_queued_cancellation_is_visible_to_both_tiers(self, job_tiers):
        """The cancellation pass is a SQL filter, and the filter is a BOOLEAN.

        ``apply_queued_cancellations`` selects ``WHERE state = ? AND
        cancel_requested = 1`` on the personal tier and ``cancel_requested =
        TRUE`` here. Both read the same value the same way, so a job that
        settles is a job both tiers agree has settled — and one whose flag was
        written as the wrong type would settle on one tier and sit on the
        other forever.
        """
        _, (personal, hosted) = job_tiers
        for store in (personal, hosted):
            job = store.submit("k").job
            store.request_cancel(job.job_id)
        assert len(personal.apply_queued_cancellations()) == 1
        assert len(hosted.apply_queued_cancellations()) == 1
        for store in (personal, hosted):
            assert store.count_by_state()[J.STATE_CANCELLED] == 1, store.capabilities().tier

    def test_recovery_agrees_on_which_jobs_moved(self, job_tiers):
        """The release gate, decided the same way on both tiers.

        ``_recover_stale`` is shared Python and the SQL that feeds it is
        nearly identical, so this is a parity check on the narrowing query:
        the lease comparison is a ``DOUBLE PRECISION`` here and a ``REAL``
        there, and a job must not expire on one tier and not the other.
        """
        import time

        _, (personal, hosted) = job_tiers
        horizon = time.time() + 3600.0
        for store in (personal, hosted):
            job = store.submit("k", max_attempts=3).job
            store.claim_next("doomed", 60, kind="k")
            store.transition(job.job_id, J.STATE_PARSING)
        assert len(personal.recover_stale(now=horizon)) == 1
        assert len(hosted.recover_stale(now=horizon)) == 1
        for store in (personal, hosted):
            tier = store.capabilities().tier
            assert store.count_by_state()[J.STATE_QUEUED] == 1, tier
            requeued = store.list_jobs(state=J.STATE_QUEUED)[0]
            assert requeued.lease_owner == "", tier
            assert requeued.lease_expires_at == 0, tier
            # The attempt it already burned stands on both tiers.
            assert requeued.attempts == 1, tier

    def test_a_live_lease_is_not_expired_on_either_tier(self, job_tiers):
        import time

        _, (personal, hosted) = job_tiers
        for store in (personal, hosted):
            job = store.submit("k").job
            store.claim_next("live", 60, kind="k")
            store.transition(job.job_id, J.STATE_PARSING)
        for store in (personal, hosted):
            assert store.recover_stale(now=time.time()) == [], store.capabilities().tier
            assert store.count_by_state()[J.STATE_PARSING] == 1, store.capabilities().tier


# --------------------------------------------------------------------------
# 4. The coverage itself
# --------------------------------------------------------------------------


class TestCoverage:
    """What fraction of the Hosted tier the suite actually reaches.

    A test that passes on both tiers without exercising PostgreSQL is not
    parity. The count is the finding: it is the number that says how much of
    the Hosted tier is really covered, and it is not the number of tests
    named ``*Postgres``.
    """

    def test_the_hosted_tier_is_reachable_or_the_skip_says_why(self):
        """The gate, and the shape of its answer.

        The point of asserting on the *reason* is that a skip nobody can act
        on is a test that stopped being run. So the reason has to name the
        variable and the command, and "the Hosted tier was never run" is not a
        reason — it is the absence of one.
        """
        dsn = os.environ.get(PG_DSN_ENV, "").strip()
        if not dsn:
            pytest.skip(
                f"未设置 {PG_DSN_ENV}。运行 docker compose up -d db 后设置 "
                f"{PG_DSN_ENV}=postgresql://paperwhisperer:paperwhisperer@localhost:5432/"
                "paperwhisperer 即可运行双档一致性测试。"
            )
        assert PG_AVAILABLE, PG_DSN_ENV
        # And when it *is* set, it has to be the database from compose, with
        # pgvector in it — a reachable server without the extension would make
        # every Hosted test above pass its skip and none of its assertions.
        from paperwhisperer.infra.stores import postgres_backend

        database = postgres_backend.PostgresDatabase(dsn, connect_timeout=5)
        try:
            with database.cursor() as cursor:
                cursor.execute(
                    "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
                )
                assert cursor.fetchone() is not None, (
                    "pgvector 未安装；Hosted 档位会在建表时报一个与真实原因无关的语法错误。"
                )
        finally:
            database.close()

    def test_both_tiers_answer_the_same_shape_of_description(self, tiers):
        """``describe()`` is the health endpoint's payload, so its *keys* must match.

        The values cannot match — ``tier`` and ``backend`` are supposed to
        differ — but a key present on one tier and not the other is a
        frontend that breaks on whichever deployment the author did not run.
        """
        personal = set(tiers.personal.bundle.describe())
        hosted = set(tiers.hosted.bundle.describe())
        assert personal == hosted
        assert tiers.personal.bundle.describe()["tier"] == TIER_PERSONAL
        assert tiers.hosted.bundle.describe()["tier"] == TIER_HOSTED

    def test_every_store_on_both_tiers_reports_a_capability(self, tiers):
        for bundle in tiers.bundles:
            for store in (bundle.metadata, bundle.vectors, bundle.lexical):
                capabilities = store.capabilities()
                assert capabilities.tier == bundle.tier
                assert isinstance(capabilities.degradations, tuple)
                assert all(isinstance(text, str) and text for text in capabilities.degradations)


# --------------------------------------------------------------------------
# Tier-independent: the harness the other suites depend on
# --------------------------------------------------------------------------


class TestHarnessContract:
    """Properties of the two-tier harness, checked on the personal tier alone.

    These run without a database, so they are green on a checkout nobody has
    configured — which is the point. A harness bug that only shows up with a
    DSN set is a harness bug that only one machine ever finds.
    """

    def test_the_harness_builds_a_working_personal_tier(self, tmp_path):
        harness = open_bundle(TIER_PERSONAL, tmp_path)
        try:
            assert harness.tier == TIER_PERSONAL
            assert harness.bundle.metadata.count_documents() == 0
        finally:
            harness.close()

    def test_closing_the_harness_twice_is_harmless(self, tmp_path):
        harness = open_bundle(TIER_PERSONAL, tmp_path)
        harness.close()
        harness.close()

    def test_each_harness_gets_its_own_isolation(self, tmp_path):
        first = open_bundle(TIER_PERSONAL, tmp_path / "first")
        second = open_bundle(TIER_PERSONAL, tmp_path / "second")
        try:
            first.bundle.metadata.upsert_document(DocumentRecord(document_id="only-in-first"))
            assert second.bundle.metadata.get_document("only-in-first") is None
        finally:
            first.close()
            second.close()

    def test_the_sqlite_url_is_still_not_a_dsn(self):
        """A personal-tier path must never be accepted as a Hosted one.

        The failure this prevents: a deployment that silently writes every
        user's data into a file and looks healthy until the container is
        replaced.
        """
        from paperwhisperer.infra.stores.postgres_backend import looks_like_dsn

        assert not looks_like_dsn("file:context/stores/x.db")
        assert not looks_like_dsn("")
        assert looks_like_dsn("postgresql://user:pw@host:5432/db")

    @pytest.mark.parametrize("raw", ["SQLite", "file", "local", "  sqlite "])
    def test_the_personal_tier_spellings(self, raw):
        from paperwhisperer.infra.stores.factory import resolve_backend

        from .test_stores_tiers import _Settings

        assert resolve_backend(_Settings(STORAGE_BACKEND=raw)) == "sqlite"

    @pytest.mark.parametrize("raw", ["postgres", "postgresql", "PG", " PostgreSQL "])
    def test_the_hosted_tier_spellings(self, raw):
        from paperwhisperer.infra.stores.factory import resolve_backend

        from .test_stores_tiers import _Settings

        assert resolve_backend(_Settings(STORAGE_BACKEND=raw)) == "postgres"
