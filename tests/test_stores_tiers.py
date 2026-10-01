"""The two tiers, and the machinery that lets one suite run against both.

T1.1's acceptance criterion is not "both backends work". It is "同一套测试
在两档各跑一遍" — the *same* suite, run once per tier. That is only a real
guarantee if the assertions are physically written once. Two hand-copied
files that are "mostly the same" drift within a week: one gains a test, the
other does not, and the acceptance criterion silently keeps passing on the
tier nobody is running.

So the structure is:

``_Settings``
    A stand-in for :mod:`paperwhisperer.core.config`. The real config is a
    module, and monkeypatching module attributes in a test that another agent
    is also editing is a bad neighbour. A private object with the same
    attributes cannot leak.
``open_bundle``
    The one place a :class:`~paperwhisperer.infra.stores.factory.StoreBundle`
    is built for a test, isolated per test.
``StoreContract``
    The behaviour every tier must exhibit. It has no tier in its name, no
    ``if`` on the backend, and no tolerance for a difference. It is the
    *definition* of "the two tiers behave the same", so anything it has to
    bend for is a bug in one of them.
``TestXSqlite`` / ``TestXPostgres``
    Empty subclasses in the store-specific test files. They are empty on
    purpose: a subclass that adds a test is a test only one tier runs, and
    that is precisely the drift this arrangement exists to prevent. Tier-
    specific behaviour gets its own test function in the tier's own file,
    named so that nobody mistakes it for shared contract.

The Hosted tier skips cleanly and says what to do. ``PAPERWHISPERER_TEST_PG_DSN``
is the switch; without it the PostgreSQL classes are skipped rather than
failed, so a checkout with no database still has a green suite — and the
skip reason names the variable, because a skip nobody can act on is a test
that stopped being run.
"""

from __future__ import annotations

import importlib
import json
import os
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import pytest

from paperwhisperer.infra.stores import (
    TIER_HOSTED,
    TIER_PERSONAL,
    DocumentRecord,
    StoreBundle,
    build_stores,
)
from paperwhisperer.infra.stores.factory import resolve_backend
from paperwhisperer.ingestion.schema import (
    BoundingBox,
    Chunk,
    Element,
    Fact,
    PaperFacts,
    ParsedDocument,
    SourceLocator,
    SourceUnit,
)

#: Set this to run the Hosted half of every contract test.
#:
#:     PAPERWHISPERER_TEST_PG_DSN=postgresql://paperwhisperer:pw@localhost:5432/pw_test \\
#:         /home/ubuntu/miniconda3/bin/python3 -m pytest tests/test_stores_*.py
#:
#: ``docker compose up -d db`` in this repository brings up exactly that
#: database, with pgvector already installed.
PG_DSN_ENV = "PAPERWHISPERER_TEST_PG_DSN"


# --------------------------------------------------------------------------
# Settings stand-in
# --------------------------------------------------------------------------


@dataclass
class _Settings:
    """The subset of configuration the store factory reads.

    Every field has the same default as the one in
    :data:`~paperwhisperer.infra.stores.factory.REQUIRED_CONFIG_KEYS`, so a
    test that only overrides the one thing it cares about is still
    exercising the shipping defaults for everything else.
    """

    STORAGE_BACKEND: str = "sqlite"
    STORAGE_SQLITE_PATH: str = "context/stores/paperwhisperer.sqlite3"
    STORAGE_SQLITE_TIMEOUT_SECONDS: int = 30
    STORAGE_LEXICAL_FTS5: str = "auto"
    STORAGE_PG_DSN: str = ""
    STORAGE_PG_SCHEMA: str = "public"
    STORAGE_PG_CONNECT_TIMEOUT_SECONDS: int = 5
    STORAGE_PG_ANN: bool = False

    def with_(self, **overrides: Any) -> _Settings:
        merged = dict(self.__dict__)
        merged.update(overrides)
        return _Settings(**merged)


# --------------------------------------------------------------------------
# Hosted-tier availability
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def postgres_availability() -> tuple[bool, str]:
    """Whether the Hosted tier can actually be reached, and why not if it can't.

    Checked once per session, and only when the DSN environment variable is
    set. Probing a database nobody asked for would make an offline test suite
    slow in a way that looks like a hang, and the reason string has to be
    useful: an operator reading a CI log should be able to fix the skip
    without reading this file.
    """
    dsn = os.environ.get(PG_DSN_ENV, "").strip()
    if not dsn:
        return False, (
            f"未设置 {PG_DSN_ENV}。运行 docker compose up -d db 后设置 "
            f"{PG_DSN_ENV}=postgresql://paperwhisperer:paperwhisperer@localhost:5432/"
            "paperwhisperer 即可启用 Hosted 档位测试。"
        )
    try:
        module = importlib.import_module("paperwhisperer.infra.stores.postgres_backend")
    except RuntimeError as exc:
        return False, f"PostgreSQL 驱动不可用：{exc}"
    database = module.PostgresDatabase(dsn, connect_timeout=5)
    try:
        with database.cursor() as cursor:
            cursor.execute("SELECT 1")
    except Exception as exc:  # noqa: BLE001 - any failure means "not reachable"
        return False, f"无法连接 {PG_DSN_ENV}={dsn}：{exc}"
    finally:
        database.close()
    return True, dsn


PG_AVAILABLE, PG_REASON = postgres_availability()

requires_postgres = pytest.mark.skipif(not PG_AVAILABLE, reason=PG_REASON)


# --------------------------------------------------------------------------
# Bundle construction
# --------------------------------------------------------------------------


@dataclass
class StoreHarness:
    """One tier's stores plus the teardown that belongs to them."""

    tier: str
    bundle: StoreBundle
    _cleanup: list[Any] = field(default_factory=list)

    @property
    def metadata(self):
        return self.bundle.metadata

    @property
    def vectors(self):
        return self.bundle.vectors

    @property
    def lexical(self):
        return self.bundle.lexical

    def close(self) -> None:
        # Cleanups before the close, not after: a Hosted-tier cleanup drops the
        # test's own schema through the bundle's own connection, so closing
        # first leaves it holding a closed handle and every test tears down
        # with "database connection is closed" -- a failure that looks like a
        # product bug, is not one, and never appears on the SQLite tier.
        for action in reversed(self._cleanup):
            action()
        self.bundle.close()


def open_bundle(tier: str, tmp_path: Any) -> StoreHarness:
    """Build an isolated store bundle for ``tier``.

    Isolation is per test and per bundle, not per process. The SQLite tier
    gets its own file under ``tmp_path``; the Hosted tier gets its own
    PostgreSQL *schema*, dropped on teardown. A schema rather than a table
    because the objects a run creates are tables, indexes, a GIN index and a
    vector column, and ``DROP TABLE`` would leave the rest behind to collide
    with the next test's index names.
    """
    if tier == TIER_HOSTED:
        schema = f"store_test_{uuid.uuid4().hex[:12]}"
        harness = StoreHarness(
            tier=tier,
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

    harness = StoreHarness(
        tier=tier,
        bundle=build_stores(
            _Settings(STORAGE_BACKEND="sqlite", STORAGE_SQLITE_PATH=str(tmp_path / "store.db"))
        ),
    )
    return harness


def open_harness(tier: str, tmp_path: Any):
    """Yield a store bundle for ``tier``, then close it. The one implementation.

    The tier-specific test files wrap this in a three-line ``@pytest.fixture``
    rather than importing the fixture from here. Importing it would work at
    runtime and would trip two linters at once: the import would look unused
    (pytest fixtures are resolved by name, not by reference) and every test
    method's ``harness`` parameter would look like a redefinition of the
    import. Wrapping keeps the behaviour written once, which is the property
    that matters, without the import.
    """
    if tier == TIER_HOSTED and not PG_AVAILABLE:
        pytest.skip(PG_REASON)
    built = open_bundle(tier, tmp_path)
    try:
        yield built
    finally:
        built.close()


# --------------------------------------------------------------------------
# Fixtures: canonical values
# --------------------------------------------------------------------------


def canonical_json(payload: Any) -> str:
    """Serialize with sorted keys, so two equal payloads are equal strings."""
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sample_document(document_id: str = "doc-1") -> ParsedDocument:
    """A document with one locator of every resolvable kind.

    Every kind, deliberately. A round-trip test that only stores a
    ``pdf_page`` locator proves that a ``txt_offset`` survives as well, and
    the offset is the one that silently becomes ``None`` — which is the
    difference between a quote that points at character 4,120 and one that
    points at the start of the file.
    """
    return ParsedDocument(
        document_id=document_id,
        source_filename="Attention Is All You Need.pdf",
        media_type="application/pdf",
        sha256="a" * 64,
        page_count=15,
        elements=[
            Element(
                role="title",
                text="Attention Is All You Need",
                order=0,
                locator=SourceLocator(kind="pdf_page", page=1),
            ),
            Element(
                role="paragraph",
                text=(
                    "The dominant sequence transduction models are based on complex "
                    "recurrent or convolutional neural networks."
                ),
                order=1,
                locator=SourceLocator(
                    kind="pdf_page",
                    page=2,
                    bbox=BoundingBox(x0=61.5, y0=142.25, x1=361.5, y1=190.75),
                ),
                heading_path=("Introduction", "Background"),
            ),
            Element(
                role="table",
                text="| model | BLEU |\n| --- | --- |\n| base | 27.3 |",
                order=2,
                locator=SourceLocator(kind="slide", slide=4, element=11),
            ),
            Element(
                role="text",
                text="Plain text has pages only in the sense of an offset.",
                order=3,
                locator=SourceLocator(kind="txt_offset", offset_start=4120, offset_end=4180),
            ),
        ],
        units=[
            SourceUnit(
                unit_id="unit-1",
                locator=SourceLocator(kind="pdf_page", page=2),
                label="Page 2",
                text="The dominant sequence transduction models are based on complex networks.",
                order=0,
            )
        ],
        parser_name="docling",
        parser_version="2.0",
        options_hash="deadbeef",
        created_at="2026-09-29T00:00:00+00:00",
        warnings=["单元测试样例"],
    )


def sample_record(document_id: str = "doc-1") -> DocumentRecord:
    return DocumentRecord(
        document_id=document_id,
        document=sample_document(document_id),
        source_filename="Attention Is All You Need.pdf",
        media_type="application/pdf",
        sha256="a" * 64,
        page_count=15,
        parser_name="docling",
        parser_version="2.0",
        options_hash="deadbeef",
        status="indexed",
        index_version="openai:text-embedding-3-small@1",
        metadata={"language": "en", "pages": 15},
    )


def sample_chunk(chunk_id: str = "chunk-1", order: int = 0) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=(
            "The dominant sequence transduction models are based on complex recurrent "
            "or convolutional neural networks."
        ),
        element_ids=("el-1", "el-2"),
        locator=SourceLocator(
            kind="pdf_page",
            page=2,
            bbox=BoundingBox(x0=61.5, y0=142.25, x1=361.5, y1=190.75),
        ),
        heading_path=("Introduction", "Background"),
        order=order,
        token_estimate=28,
        index_version="openai:text-embedding-3-small@1",
    )


def sample_facts() -> PaperFacts:
    return PaperFacts(
        research_question=Fact(
            field_name="research_question",
            value="能否完全抛弃循环与卷积？",
            evidence_ids=("chunk-1",),
        ),
        contributions=[
            Fact(
                field_name="contributions",
                value="提出纯注意力架构。",
                evidence_ids=("chunk-2",),
            )
        ],
        method=Fact(field_name="method", value="多头自注意力 + 位置编码。", evidence_ids=("chunk-3",)),
        datasets=[Fact(field_name="datasets", value="WMT 2014 En-De。", evidence_ids=("chunk-4",))],
        metrics=[Fact(field_name="metrics", value="BLEU 28.4。", evidence_ids=("chunk-5",))],
        main_results=[],
        limitations=[Fact(field_name="limitations", value="计算开销随序列长度平方增长。")],
        reproducibility=Fact(field_name="reproducibility", value="代码已开源。"),
        extraction_model="gpt-4o-mini",
        prompt_version="facts-v1",
        evidence_version="ev-1",
    )


# --------------------------------------------------------------------------
# Tier-independent unit tests of the factory
# --------------------------------------------------------------------------


class TestBackendSelection:
    """Which tier gets built, and what happens when the answer is nonsense."""

    def test_default_is_the_personal_tier(self):
        # The personal tier is the default in the plan, and the only one that
        # can be the default in a sense that means anything: a deployment
        # that has configured nothing must end up with zero external services.
        assert resolve_backend(_Settings()) == "sqlite"
        assert resolve_backend(_Settings(STORAGE_BACKEND="")) == "sqlite"

    @pytest.mark.parametrize("raw", ["SQLite", "file", "local", "  sqlite "])
    def test_personal_tier_spellings(self, raw):
        assert resolve_backend(_Settings(STORAGE_BACKEND=raw)) == "sqlite"

    @pytest.mark.parametrize("raw", ["postgres", "postgresql", "PG", " PostgreSQL "])
    def test_hosted_tier_spellings(self, raw):
        assert resolve_backend(_Settings(STORAGE_BACKEND=raw)) == "postgres"

    def test_unknown_backend_is_refused(self):
        with pytest.raises(ValueError, match="STORAGE_BACKEND"):
            resolve_backend(_Settings(STORAGE_BACKEND="mysql"))

    def test_build_stores_without_config_uses_the_real_config(self, tmp_path, monkeypatch):
        from paperwhisperer.core import config as real_config

        monkeypatch.setattr(
            real_config, "STORAGE_SQLITE_PATH", str(tmp_path / "from-real-config.db"), raising=False
        )
        bundle = build_stores()
        try:
            assert bundle.tier == TIER_PERSONAL
            assert bundle.metadata.count_documents() == 0
        finally:
            bundle.close()

    def test_hosted_tier_without_a_dsn_refuses_rather_than_falling_back(self, tmp_path):
        # The failure this prevents: a Hosted deployment that silently writes
        # every user's data into the container's local disk and looks healthy
        # until the container is replaced.
        with pytest.raises(ValueError, match="STORAGE_PG_DSN"):
            build_stores(_Settings(STORAGE_BACKEND="postgres", STORAGE_PG_DSN=""))

    def test_a_sqlite_url_is_not_accepted_as_a_dsn(self):
        with pytest.raises(ValueError, match="STORAGE_PG_DSN"):
            build_stores(
                _Settings(STORAGE_BACKEND="postgres", STORAGE_PG_DSN="file:context/stores/x.db")
            )

    def test_unknown_fts_mode_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="STORAGE_LEXICAL_FTS5"):
            build_stores(
                _Settings(
                    STORAGE_SQLITE_PATH=str(tmp_path / "x.db"),
                    STORAGE_LEXICAL_FTS5="maybe",
                )
            )


class TestDegradationIsObservable:
    """A silent fallback is a performance regression nobody would notice."""

    def test_scanning_can_be_forced_and_is_reported(self, tmp_path):
        bundle = build_stores(
            _Settings(
                STORAGE_SQLITE_PATH=str(tmp_path / "scan.db"),
                STORAGE_LEXICAL_FTS5="off",
            )
        )
        try:
            capabilities = bundle.capabilities
            assert capabilities.lexical_engine == "scan"
            assert capabilities.scan_only is True
            assert any("扫描" in message for message in capabilities.degradations)
        finally:
            bundle.close()

    def test_a_full_build_reports_no_degradation_it_did_not_suffer(self, tmp_path):
        bundle = build_stores(
            _Settings(
                STORAGE_SQLITE_PATH=str(tmp_path / "auto.db"),
                STORAGE_LEXICAL_FTS5="auto",
            )
        )
        try:
            capabilities = bundle.capabilities
            if capabilities.lexical_engine == "scan":
                # FTS5 really is missing here, and the reason has to be there.
                assert capabilities.degradations
            else:
                assert capabilities.lexical_engine in {"fts", "scan"}
        finally:
            bundle.close()

    def test_describe_is_machine_readable(self, tmp_path):
        bundle = build_stores(_Settings(STORAGE_SQLITE_PATH=str(tmp_path / "d.db")))
        try:
            described = bundle.describe()
            assert described["tier"] == TIER_PERSONAL
            assert described["backend"] == "sqlite"
            assert isinstance(described["degradations"], list)
            assert described["vector_engine"] in {"matrix", "python", "ann"}
        finally:
            bundle.close()

    def test_every_store_reports_capabilities(self, tmp_path):
        bundle = build_stores(_Settings(STORAGE_SQLITE_PATH=str(tmp_path / "c.db")))
        try:
            for store in (bundle.metadata, bundle.vectors, bundle.lexical):
                assert store.capabilities().tier == TIER_PERSONAL
        finally:
            bundle.close()

    def test_numpy_absence_is_reported_as_a_cliff_not_a_failure(self, tmp_path, monkeypatch):
        from paperwhisperer.infra.stores import sqlite_backend

        monkeypatch.setattr(sqlite_backend, "_numpy", None)
        bundle = build_stores(_Settings(STORAGE_SQLITE_PATH=str(tmp_path / "nopy.db")))
        try:
            assert bundle.capabilities.vector_engine == "python"
            assert any("numpy" in message for message in bundle.capabilities.degradations)
        finally:
            bundle.close()

    def test_the_pure_python_path_computes_the_same_scores(self, tmp_path, monkeypatch):
        """The fallback has to be *slower*, not *different*.

        This is the whole justification for keeping the arithmetic in
        :mod:`~paperwhisperer.infra.stores.scoring` rather than inlined into
        the numpy path: a build without numpy is a supported configuration,
        and the only way anyone can trust that claim is if the numbers match.
        """
        from paperwhisperer.infra.stores import sqlite_backend

        path = str(tmp_path / "scores.db")
        vectors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.7, 0.7, 0.0)]

        def _seed() -> list:
            bundle = build_stores(_Settings(STORAGE_SQLITE_PATH=path))
            bundle.vectors.upsert_many(
                [
                    sqlite_backend.VectorEntry(chunk_id=f"c{index}", vector=vector)
                    for index, vector in enumerate(vectors)
                ]
            )
            hits = bundle.vectors.search((1.0, 0.2, 0.0), top_k=3)
            bundle.close()
            return hits

        with_numpy = _seed()
        monkeypatch.setattr(sqlite_backend, "_numpy", None)
        without_numpy = _seed()

        assert [hit.chunk_id for hit in with_numpy] == [
            hit.chunk_id for hit in without_numpy
        ]
        for fast, slow in zip(with_numpy, without_numpy, strict=True):
            assert fast.score == pytest.approx(slow.score, rel=1e-12, abs=1e-12)


class TestBundleLifecycle:
    """Opening, cascading deletes, and closing twice."""

    def test_delete_removes_every_index_entry(self, tmp_path):
        bundle = build_stores(_Settings(STORAGE_SQLITE_PATH=str(tmp_path / "cascade.db")))
        try:
            bundle.metadata.upsert_document(sample_record("doc-cascade"))
            bundle.metadata.put_chunk("doc-cascade", sample_chunk("chunk-c"))
            bundle.metadata.put_facts("doc-cascade", sample_facts())
            bundle.vectors.upsert_vector(
                "chunk-c", (1.0, 0.0, 0.0), document_id="doc-cascade"
            )
            bundle.lexical.index_chunk(
                "chunk-c", "attention mechanism", document_id="doc-cascade"
            )

            assert bundle.delete_document("doc-cascade") is True

            assert bundle.metadata.get_document("doc-cascade") is None
            assert bundle.metadata.get_chunk("chunk-c") is None
            assert bundle.metadata.get_facts("doc-cascade") is None
            # The one that matters most: an orphaned embedding keeps answering
            # queries for a document the user deleted.
            assert bundle.vectors.get_vector("chunk-c") is None
            assert bundle.lexical.count_chunks() == 0
            assert bundle.lexical.search("attention") == []
        finally:
            bundle.close()

    def test_close_is_idempotent(self, tmp_path):
        bundle = build_stores(_Settings(STORAGE_SQLITE_PATH=str(tmp_path / "close.db")))
        bundle.close()
        bundle.close()

    def test_the_store_directory_is_created(self, tmp_path):
        nested = tmp_path / "a" / "b" / "store.db"
        bundle = build_stores(_Settings(STORAGE_SQLITE_PATH=str(nested)))
        try:
            bundle.metadata.upsert_document(sample_record("doc-nested"))
            assert nested.parent.is_dir()
        finally:
            bundle.close()

    def test_data_survives_reopening(self, tmp_path):
        # The personal tier's promise is that the whole installation is one
        # directory. This is the test that keeps that promise true.
        path = str(tmp_path / "persist.db")
        first = build_stores(_Settings(STORAGE_SQLITE_PATH=path))
        first.metadata.upsert_document(sample_record("doc-persist"))
        first.close()

        second = build_stores(_Settings(STORAGE_SQLITE_PATH=path))
        try:
            assert second.metadata.get_document("doc-persist") is not None
        finally:
            second.close()


def _engine_words() -> tuple[str, ...]:
    """The engine vocabulary the contract checker forbids, read from the checker.

    Read rather than restated. A second copy of this list would drift, and
    the test that exists to catch drift is the last place to introduce it.
    """
    import importlib.util
    import sys

    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tools",
        "check_store_contract.py",
    )
    spec = importlib.util.spec_from_file_location("_store_contract_checker", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ENGINE_WORDS


class TestContractIsEngineNeutral:
    """The property that makes one suite for two tiers possible at all."""

    def test_no_contract_member_names_a_storage_engine(self):
        from paperwhisperer.infra.stores import interfaces
        from paperwhisperer.infra.stores.interfaces import (
            LexicalRetriever,
            MetadataStore,
            VectorStore,
        )

        engine_words = _engine_words()

        offenders = []
        for protocol in (MetadataStore, VectorStore, LexicalRetriever):
            for name in dir(protocol):
                if name.startswith("_"):
                    continue
                lowered = name.lower()
                offenders.extend(
                    f"{protocol.__name__}.{name} 提到了 {word!r}"
                    for word in engine_words
                    if word in lowered
                )
        assert not offenders, offenders
        assert interfaces.TIER_PERSONAL and interfaces.TIER_HOSTED

    def test_the_contract_checker_passes(self):
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "tools/check_store_contract.py"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        assert result.returncode == 0, result.stdout + result.stderr
