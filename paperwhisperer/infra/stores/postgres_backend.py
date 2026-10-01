"""The Hosted tier: PostgreSQL, pgvector, full-text search.

Same three interfaces as the personal tier, a real server underneath. What
changes is only what the server is better at, and this module is careful not
to let it change *anything else* — see :mod:`.scoring` for why the score a
caller sees is computed in Python on both tiers.

Three decisions here are worth stating up front, because each of them is the
place where this tier could quietly become a different product:

**Both driver generations are supported.** ``psycopg`` (3) and ``psycopg2``
are in wide use and there is no good reason for a storage layer to insist on
one. The only common API that matters is used throughout: ``execute`` with
``%s`` placeholders, tuple results, explicit commit. Everything else is
deliberately avoided, because it is exactly where the two differ.

**Schema DDL is not parameterized, so the schema name is validated.** Every
identifier in this module is interpolated, which is the one way SQL
injection gets in. The schema name is therefore checked against a strict
pattern and the table names are constants, rather than pretending a bind
parameter could carry an identifier.

**Vector search is exact by default, and that is a choice, not a
limitation.** pgvector's approximate indexes would be faster on a large
corpus, and they would also return slightly different neighbours from one run
to the next, which would mean a top-10 that is not a function of the data.
For a product whose whole claim is that a citation is checkable, a
deterministic result list is worth more than a faster one at this size. The
approximate index is available through ``STORAGE_PG_ANN=true`` and is created
lazily, because its dimension is only known once the first embedding has been
written.

The full-text index is built over a *pre-tokenized* column, not over raw
document text, for the same reason the personal tier does it that way: the
database's own tokenizer has no way to know how this project segments Chinese,
and a query that works on the personal tier must not quietly fail here.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from paperwhisperer.ingestion.schema import Chunk, PaperFacts, ParsedDocument

from . import scoring
from .interfaces import (
    LEXICAL_ENGINE_FTS,
    TIER_HOSTED,
    VECTOR_ENGINE_ANN,
    VECTOR_ENGINE_MATRIX,
    DocumentRecord,
    LexicalHit,
    StoreCapabilities,
    VectorEntry,
    VectorHit,
)

logger = logging.getLogger(__name__)

#: A schema name is interpolated into DDL, so it is checked rather than
#: trusted. Letters, digits and underscores only, and it cannot start with a
#: digit — the exact grammar of an unquoted SQL identifier.
SCHEMA_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

#: The longest lexeme PostgreSQL will put in a ``tsvector``. Beyond it the
#: server emits ``word is too long to be indexed`` and drops the term, so this
#: number is the boundary of a real difference between the tiers and not an
#: internal detail: SQLite's FTS5 has no length limit at all, so a chunk
#: carrying a longer term is findable on the personal tier and silent on the
#: Hosted one. Measured rather than quoted — the server's own message says
#: "longer than 2047" and ``repeat('z', 2046)`` indexes while 2047 does not,
#: so the largest accepted value is one less than the number in the message.
MAX_LEXEME_BYTES = 2046

#: How every column and every ``ORDER BY`` in this module spells a text sort
#: key. This is a correctness requirement disguised as a type, so it is a
#: constant rather than a literal repeated in nine places.
#:
#: **SQLite sorts text by byte value.** That is what ``BINARY`` collation is,
#: it is the only collation SQLite has for a column, and it is the order every
#: expectation in the shared suite is written in. **PostgreSQL sorts text by
#: the database's collation**, which is ``en_US.utf8`` for the image this
#: project's compose file runs and ``C`` for many others — so the same
#: ``ORDER BY chunk_id`` returns a different list depending on how somebody
#: initialised their cluster, and case- and punctuation-insensitive relative
#: to SQLite. Concretely: ``Chunk-A < CHUNK-A < chunk-a`` on the personal tier
#: and ``Chunk-A < chunk-a < CHUNK-A`` on the Hosted one, for the same three
#: rows.
#:
#: That breaks a promise the contract makes in three separate places —
#: ``list_documents``, ``list_chunks`` and ``VectorStore.search`` all document
#: a ``chunk_id`` tiebreak specifically so that "the same query returns a
#: different page order on the two tiers, which is a pagination bug that only
#: appears in production". The tiebreak existed to *fix* non-determinism and
#: was quietly defeated by the collation underneath it. It also affects
#: ``created_at``, whose ISO timestamps differ from SQLite's order by exactly
#: one character: ``+00:00`` sorts before ``+0000`` under ``C`` and after it
#: under ``en_US.utf8``.
#:
#: Written on both the column and the sort so that the two are guaranteed to
#: agree, which is what lets PostgreSQL use the index instead of sorting every
#: page. A database created before this was written keeps its old collation and
#: therefore sorts those pages without an index; it is *correct* either way,
#: because the ``ORDER BY`` carries its own ``COLLATE``. Rewriting the columns
#: of a table an operator may be holding rows in is not something a library
#: should do to itself at ``initialize()``, and this table is small enough that
#: the sort is not worth the risk.
SORT_KEY = 'TEXT COLLATE "C"'



def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_driver() -> Any:
    """Return a usable DB-API driver module, preferring psycopg 3.

    Raising at construction rather than at first query: a Hosted deployment
    that cannot reach its database should fail at startup, where an operator
    is looking, rather than on the first user request.
    """
    try:
        import psycopg  # noqa: PLC0415 - an optional dependency probed at runtime

        return psycopg
    except ImportError:
        pass
    try:
        import psycopg2  # noqa: PLC0415

        return psycopg2
    except ImportError as exc:
        raise RuntimeError(
            "Hosted 档位需要 PostgreSQL 驱动："
            "请安装 psycopg[binary]（推荐）或 psycopg2-binary，"
            "或改用 STORAGE_BACKEND=sqlite 的个人档位。"
        ) from exc


class PostgresDatabase:
    """Per-thread connections to one PostgreSQL database.

    Mirrors :class:`~.sqlite_backend.SqliteDatabase` deliberately: a
    connection per thread, closed together, so the two tiers fail the same
    way. With no server to start, "no external service" is a property of the
    personal tier's constructor, not of its code path.
    """

    def __init__(
        self,
        dsn: str,
        *,
        schema: str = "public",
        connect_timeout: int = 5,
    ) -> None:
        if not SCHEMA_NAME_RE.match(str(schema or "")):
            raise ValueError(
                f"非法的 PostgreSQL schema 名：{schema!r}"
                "（只允许字母、数字和下划线，且不以数字开头）"
            )
        self.dsn = str(dsn or "")
        self.schema = str(schema)
        self.connect_timeout = max(1, int(connect_timeout))
        self.driver = load_driver()
        self._local = threading.local()
        self._connections: list[Any] = []
        self._lock = threading.RLock()
        self._closed = False

    def _open(self) -> Any:
        return self.driver.connect(self.dsn, connect_timeout=self.connect_timeout)

    def connection(self) -> Any:
        if self._closed:
            raise RuntimeError("数据库连接已关闭。")
        existing = getattr(self._local, "connection", None)
        if existing is None:
            existing = self._open()
            self._local.connection = existing
            with self._lock:
                self._connections.append(existing)
        return existing

    @contextmanager
    def cursor(self):
        """A cursor inside an explicit transaction, always committed or rolled back.

        The commit is written out rather than left to a context manager
        because ``with connection:`` means *commit* on psycopg2 and
        *commit-or-rollback-then-close* on psycopg3, and this module supports
        both.
        """
        connection = self.connection()
        cursor = connection.cursor()
        try:
            yield cursor
        except BaseException:
            try:
                connection.rollback()
            except Exception:  # pragma: no cover - the original error is the useful one
                logger.warning("回滚事务失败。", exc_info=True)
            raise
        else:
            connection.commit()
        finally:
            cursor.close()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            candidates = list(self._connections)
            self._connections.clear()
        for connection in candidates:
            try:
                connection.close()
            except Exception:  # pragma: no cover - close must not raise
                pass

    def qualified(self, table: str) -> str:
        return f"{self.schema}.{table}"


def _dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _vector_literal(values: Sequence[float]) -> str:
    """Render a vector as the text form pgvector parses.

    A string literal rather than a driver-specific adaptation: both drivers
    would need different placeholder syntax for a vector value, and one text
    form works on both. The cast in the SQL supplies the type.
    """
    return "[" + ",".join(repr(float(value)) for value in values) + "]"


def _safe_ident(name: str) -> str:
    return name if SCHEMA_NAME_RE.match(name) else "store"


class _PostgresStoreBase:
    """Shared plumbing: the database handle, the logger, and ``close``."""

    tier = TIER_HOSTED

    def __init__(self, database: PostgresDatabase) -> None:
        self.database = database

    @property
    def table(self) -> str:
        return self.database.qualified(self.TABLE)

    def close(self) -> None:
        """No-op per store; the owning :class:`PostgresDatabase` closes."""


class PostgresMetadataStore(_PostgresStoreBase):
    """Documents, chunks and the fact layer.

    The canonical payload is stored as JSONB for the same reason the personal
    tier stores JSON: the reader must be the same code the writer used, so
    that a document written by a build that has since changed its schema is
    rehydrated by the schema module's own compatibility rules instead of
    being reinterpreted by whatever columns happen to exist.
    """

    TABLE = "store_documents"
    CHUNKS_TABLE = "store_chunks"
    FACTS_TABLE = "store_paper_facts"

    def __init__(self, database: PostgresDatabase) -> None:
        super().__init__(database)
        self._capabilities = StoreCapabilities(tier=self.tier)

    def initialize(self) -> None:
        with self.database.cursor() as cursor:
            cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {self.database.schema}")
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    document_id {SORT_KEY} PRIMARY KEY,
                    created_at {SORT_KEY} NOT NULL,
                    payload JSONB NOT NULL
                )
            """)
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_documents_order
                ON {self.table} (created_at DESC, document_id ASC)
            """)
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.database.qualified(self.CHUNKS_TABLE)} (
                    chunk_id {SORT_KEY} PRIMARY KEY,
                    document_id {SORT_KEY} NOT NULL,
                    chunk_order INTEGER NOT NULL,
                    payload JSONB NOT NULL
                )
            """)
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_chunks_document
                ON {self.database.qualified(self.CHUNKS_TABLE)}
                (document_id, chunk_order, chunk_id)
            """)
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.database.qualified(self.FACTS_TABLE)} (
                    document_id {SORT_KEY} PRIMARY KEY,
                    payload JSONB NOT NULL
                )
            """)

    def capabilities(self) -> StoreCapabilities:
        return self._capabilities

    # -- documents ---------------------------------------------------------

    def upsert_document(self, record: DocumentRecord) -> DocumentRecord:
        identifier = str(record.document_id or "").strip()
        if not identifier:
            raise ValueError("document_id 不能为空。")
        with self.database.cursor() as cursor:
            # A transaction-level advisory lock keyed on the document id.
            # ``FOR UPDATE`` locks a row that exists; two processes inserting
            # the *same* new document both see no row, both read version 0 and
            # both write version 1 — the lost update the optimistic lock in the
            # contract is supposed to prevent. The lock is what makes the two
            # tiers agree, and the personal tier gets the same guarantee for
            # free from its exclusive write lock.
            cursor.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s)::bigint)", (identifier,)
            )
            cursor.execute(
                f"SELECT payload FROM {self.table} WHERE document_id = %s FOR UPDATE",
                (identifier,),
            )
            row = cursor.fetchone()
            previous_version = 0
            if row is not None:
                payload = row[0]
                if isinstance(payload, (str, bytes)):
                    payload = json.loads(payload)
                previous_version = int((payload or {}).get("version") or 0)
            stored = DocumentRecord.from_dict(record.to_dict())
            stored.version = previous_version + 1
            stored.created_at = record.created_at or _now()
            stored.updated_at = _now()
            # A record may legitimately arrive with no parsed document — a
            # registration that is still being parsed, say — and the personal
            # tier does not leave that as a bare ``None``. It substitutes an
            # empty document carrying the identifier, so a caller that
            # round-trips a record and reaches through ``.document`` never has
            # to branch on which tier it got. This tier stored the ``None``,
            # and the two tiers disagreed about the same value in both
            # directions: the row this returned claimed to have no document
            # while the row the personal tier returned for the identical
            # document had one. Offline testing could not find it, because the
            # substitution is invisible until you read the record back on both
            # tiers and compare.
            if stored.document is None:
                stored.document = ParsedDocument(document_id=identifier)
                stored.document.document_id = identifier
            cursor.execute(
                f"""
                INSERT INTO {self.table}(document_id, created_at, payload)
                VALUES (%s, %s, %s::jsonb)
                ON CONFLICT (document_id) DO UPDATE
                SET created_at = EXCLUDED.created_at, payload = EXCLUDED.payload
                """,
                (identifier, stored.created_at, _dumps(stored.to_dict())),
            )
        return stored

    def get_document(self, document_id: str) -> DocumentRecord | None:
        with self.database.cursor() as cursor:
            cursor.execute(
                f"SELECT payload FROM {self.table} WHERE document_id = %s", (str(document_id),)
            )
            row = cursor.fetchone()
        if row is None:
            return None
        payload = row[0]
        if isinstance(payload, (str, bytes)):
            payload = json.loads(payload)
        return DocumentRecord.from_dict(payload)

    def list_documents(self, limit: int = 50, offset: int = 0) -> list[DocumentRecord]:
        with self.database.cursor() as cursor:
            cursor.execute(
                f"SELECT payload FROM {self.table} "
                "ORDER BY created_at DESC, document_id COLLATE \"C\" ASC LIMIT %s OFFSET %s",
                (max(0, int(limit)), max(0, int(offset))),
            )
            rows = cursor.fetchall()
        records = []
        for row in rows:
            payload = row[0]
            if isinstance(payload, (str, bytes)):
                payload = json.loads(payload)
            records.append(DocumentRecord.from_dict(payload))
        return records

    def delete_document(self, document_id: str) -> bool:
        chunks = self.database.qualified(self.CHUNKS_TABLE)
        facts = self.database.qualified(self.FACTS_TABLE)
        with self.database.cursor() as cursor:
            cursor.execute(f"DELETE FROM {self.table} WHERE document_id = %s", (str(document_id),))
            removed = cursor.rowcount
            cursor.execute(f"DELETE FROM {chunks} WHERE document_id = %s", (str(document_id),))
            cursor.execute(f"DELETE FROM {facts} WHERE document_id = %s", (str(document_id),))
        return bool(removed)

    def count_documents(self) -> int:
        with self.database.cursor() as cursor:
            cursor.execute(f"SELECT COUNT(*) FROM {self.table}")
            return int(cursor.fetchone()[0])

    # -- chunks ------------------------------------------------------------

    def put_chunk(self, document_id: str, chunk: Chunk) -> Chunk:
        stored = Chunk.from_dict(chunk.to_dict())
        if not stored.chunk_id:
            raise ValueError("chunk_id 不能为空。")
        chunks = self.database.qualified(self.CHUNKS_TABLE)
        with self.database.cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO {chunks}(chunk_id, document_id, chunk_order, payload)
                VALUES (%s, %s, %s, %s::jsonb)
                ON CONFLICT (chunk_id) DO UPDATE
                SET document_id = EXCLUDED.document_id,
                    chunk_order = EXCLUDED.chunk_order,
                    payload = EXCLUDED.payload
                """,
                (stored.chunk_id, str(document_id), int(stored.order), _dumps(stored.to_dict())),
            )
        return stored

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        chunks = self.database.qualified(self.CHUNKS_TABLE)
        with self.database.cursor() as cursor:
            cursor.execute(
                f"SELECT payload FROM {chunks} WHERE chunk_id = %s", (str(chunk_id),)
            )
            row = cursor.fetchone()
        if row is None:
            return None
        payload = row[0]
        if isinstance(payload, (str, bytes)):
            payload = json.loads(payload)
        return Chunk.from_dict(payload)

    def list_chunks(self, document_id: str) -> list[Chunk]:
        chunks = self.database.qualified(self.CHUNKS_TABLE)
        with self.database.cursor() as cursor:
            cursor.execute(
                f"SELECT payload FROM {chunks} WHERE document_id = %s "
                "ORDER BY chunk_order ASC, chunk_id COLLATE \"C\" ASC",
                (str(document_id),),
            )
            rows = cursor.fetchall()
        result = []
        for row in rows:
            payload = row[0]
            if isinstance(payload, (str, bytes)):
                payload = json.loads(payload)
            result.append(Chunk.from_dict(payload))
        return result

    def delete_chunks(self, document_id: str) -> int:
        chunks = self.database.qualified(self.CHUNKS_TABLE)
        with self.database.cursor() as cursor:
            cursor.execute(f"DELETE FROM {chunks} WHERE document_id = %s", (str(document_id),))
            return int(cursor.rowcount or 0)

    # -- facts -------------------------------------------------------------

    def put_facts(self, document_id: str, facts: PaperFacts) -> PaperFacts:
        stored = PaperFacts.from_dict(facts.to_dict())
        facts_table = self.database.qualified(self.FACTS_TABLE)
        with self.database.cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO {facts_table}(document_id, payload) VALUES (%s, %s::jsonb)
                ON CONFLICT (document_id) DO UPDATE SET payload = EXCLUDED.payload
                """,
                (str(document_id), _dumps(stored.to_dict())),
            )
        return stored

    def get_facts(self, document_id: str) -> PaperFacts | None:
        facts_table = self.database.qualified(self.FACTS_TABLE)
        with self.database.cursor() as cursor:
            cursor.execute(
                f"SELECT payload FROM {facts_table} WHERE document_id = %s", (str(document_id),)
            )
            row = cursor.fetchone()
        if row is None:
            return None
        payload = row[0]
        if isinstance(payload, (str, bytes)):
            payload = json.loads(payload)
        return PaperFacts.from_dict(payload)


class PostgresVectorStore(_PostgresStoreBase):
    """Embeddings in a pgvector column, searched by cosine distance.

    The column is declared as an unconstrained ``vector`` rather than
    ``vector(768)`` because the width is discovered from the first embedding
    and not from configuration — a deployment that hard-codes it would have to
    be re-deployed when the embedding model changes, and the dimension check
    in :meth:`upsert_vector` already refuses a mixed-width index. The cost is
    that the server cannot reject a malformed row on its own, which is why
    the check is in the contract rather than implied by the schema.
    """

    TABLE = "store_vectors"

    def __init__(self, database: PostgresDatabase, *, ann: bool = False) -> None:
        super().__init__(database)
        self.ann = bool(ann)
        self._dimension: int | None = None
        self._index_dimension: int | None = None
        self._degradations: list[str] = []
        if not self.ann:
            self._degradations.append(
                "向量检索当前为精确全表扫描（STORAGE_PG_ANN=false）："
                "结果确定且可复现，代价是语料变大后查询耗时线性增长。"
            )

    def initialize(self) -> None:
        with self.database.cursor() as cursor:
            cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {self.database.schema}")
            # Without pgvector there is no vector type at all, and the failure
            # would otherwise be a syntax error on the next statement — which
            # says nothing about the actual cause. This says it directly.
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    chunk_id {SORT_KEY} PRIMARY KEY,
                    document_id {SORT_KEY} NOT NULL DEFAULT '',
                    dim INTEGER NOT NULL,
                    embedding VECTOR NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb
                )
            """)
            cursor.execute(
                f"CREATE INDEX IF NOT EXISTS idx_vectors_document "
                f"ON {self.table} (document_id)"
            )
        for message in self._degradations:
            logger.warning(message)

    def capabilities(self) -> StoreCapabilities:
        return StoreCapabilities(
            tier=self.tier,
            vector_engine=VECTOR_ENGINE_ANN if self.ann else VECTOR_ENGINE_MATRIX,
            degradations=tuple(self._degradations),
        )

    @property
    def dimension(self) -> int | None:
        """The width of the space, read from the table rather than remembered.

        Deliberately re-read every time even though the value cannot change
        without a conflicting write. The personal tier has no cache here
        because it has no reason to build one, and the difference that made:
        this tier kept ``_dimension`` from the first write, so after
        ``delete_document`` emptied the table the property still answered with
        the old width while the personal tier answered ``None`` — and then
        ``upsert_vector`` *refused* a vector of any other width, turning an
        ordinary "delete everything and re-index at a new dimension" into a
        ValueError on one tier and a success on the other. A cache that is
        only correct until somebody deletes is not a cache, it is a second
        source of truth, and this module's own rule is that the row is the
        record. The cost is one indexed ``LIMIT 1`` per call, against a
        corpus small enough that the Hosted tier exists for a reason other
        than this query's latency.
        """
        with self.database.cursor() as cursor:
            cursor.execute(f"SELECT dim FROM {self.table} LIMIT 1")
            row = cursor.fetchone()
        return int(row[0]) if row is not None else None

    def upsert_vector(
        self,
        chunk_id: str,
        vector: Sequence[float],
        *,
        document_id: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        identifier = str(chunk_id).strip()
        if not identifier:
            raise ValueError("chunk_id 不能为空。")
        values = scoring.l2_normalize(vector)
        self._assert_dimension(len(values))
        with self.database.cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO {self.table}(chunk_id, document_id, dim, embedding, metadata)
                VALUES (%s, %s, %s, %s::vector, %s::jsonb)
                ON CONFLICT (chunk_id) DO UPDATE
                SET document_id = EXCLUDED.document_id,
                    dim = EXCLUDED.dim,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata
                """,
                (
                    identifier,
                    str(document_id),
                    len(values),
                    _vector_literal(values),
                    _dumps(dict(metadata or {})),
                ),
            )
        self._dimension = len(values)
        if self.ann:
            self._ensure_ann_index(len(values))

    def upsert_many(self, entries: Iterable[VectorEntry]) -> int:
        materialized = list(entries)
        with self.database.cursor():
            for entry in materialized:
                self.upsert_vector(
                    entry.chunk_id,
                    entry.vector,
                    document_id=entry.document_id,
                    metadata=entry.metadata,
                )
        return len(materialized)

    def get_vector(self, chunk_id: str) -> VectorEntry | None:
        with self.database.cursor() as cursor:
            cursor.execute(
                f"SELECT chunk_id, document_id, dim, embedding::text, metadata FROM {self.table} "
                "WHERE chunk_id = %s",
                (str(chunk_id),),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        metadata = row[4]
        if isinstance(metadata, (str, bytes)):
            metadata = json.loads(metadata)
        return VectorEntry(
            chunk_id=row[0],
            vector=_parse_vector_literal(row[3], int(row[2])),
            document_id=row[1],
            metadata=metadata or {},
        )

    def delete_vector(self, chunk_id: str) -> bool:
        with self.database.cursor() as cursor:
            cursor.execute(f"DELETE FROM {self.table} WHERE chunk_id = %s", (str(chunk_id),))
            return bool(cursor.rowcount)

    def delete_document(self, document_id: str) -> int:
        with self.database.cursor() as cursor:
            cursor.execute(f"DELETE FROM {self.table} WHERE document_id = %s", (str(document_id),))
            return int(cursor.rowcount or 0)

    def search(
        self,
        vector: Sequence[float],
        top_k: int = 10,
        *,
        document_ids: Sequence[str] | None = None,
        min_score: float | None = None,
    ) -> list[VectorHit]:
        query = scoring.l2_normalize(vector)
        limit = max(0, int(top_k))
        if limit == 0:
            return []
        established = self.dimension
        if established is not None and len(query) != established:
            raise ValueError(
                f"查询向量维度 {len(query)} 与索引维度 {established} 不一致，"
                "请确认 embedding 模型未更换。"
            )
        sql = (
            f"SELECT chunk_id, document_id, (embedding <=> %s::vector) AS distance "
            f"FROM {self.table}"
        )
        params: list[Any] = [_vector_literal(query)]
        if document_ids is not None:
            keys = [str(item) for item in document_ids]
            if not keys:
                return []
            sql += " WHERE document_id = ANY(%s)"
            params.append(keys)
        # Over-fetch, then apply the shared total order. Ordering by distance
        # in SQL would leave ties in the server's order, and the chunk_id
        # tiebreak is what makes top_k a function of the data.
        # ``COLLATE "C"`` on the tiebreak for the reason in :data:`SORT_KEY`:
        # without it this page boundary is computed under the database's
        # collation while the personal tier computes it in byte order, and the
        # over-fetch is silently choosing the wrong top_k.
        sql += ' ORDER BY distance ASC, chunk_id COLLATE "C" ASC LIMIT %s'
        params.append(max(limit * 4, limit))
        with self.database.cursor() as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
        hits = []
        for chunk_id, document_id, distance in rows:
            score = max(-1.0, min(1.0, 1.0 - float(distance)))
            if min_score is not None and score < float(min_score):
                continue
            hits.append(VectorHit(chunk_id=chunk_id, score=score, document_id=document_id))
        hits.sort(key=lambda hit: scoring.sort_key(hit.score, hit.chunk_id))
        return hits[:limit]

    def count_vectors(self) -> int:
        with self.database.cursor() as cursor:
            cursor.execute(f"SELECT COUNT(*) FROM {self.table}")
            return int(cursor.fetchone()[0])

    # -- internals ---------------------------------------------------------

    def _assert_dimension(self, width: int) -> None:
        existing = self.dimension
        if existing is not None and existing != width:
            raise ValueError(
                f"向量维度 {width} 与索引已建立的维度 {existing} 不一致。"
                "更换 embedding 模型后必须重建索引，而不是写入维度不同的向量。"
            )

    def _ensure_ann_index(self, width: int) -> None:
        """Create the approximate index once the width is known.

        Deferred because pgvector requires the dimension in the index
        definition, and the dimension is not known until the first embedding
        lands. HNSW is tried first and IVFFlat second: HNSW has no training
        step, which matters because a freshly started database has no rows to
        train on, and a table with an untrained IVFFlat index returns nothing
        at all rather than something slow.
        """
        if self._index_dimension == width:
            return
        index_name = _safe_ident(f"idx_vectors_hnsw_{width}")
        fallback_name = _safe_ident(f"idx_vectors_ivfflat_{width}")
        attempts = (
            (f"CREATE INDEX IF NOT EXISTS {index_name} ON {self.table} "
             f"USING hnsw (embedding vector_cosine_ops)", None),
            (
                f"CREATE INDEX IF NOT EXISTS {fallback_name} ON {self.table} "
                f"USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)",
                "pgvector 的 HNSW 索引不可用",
            ),
        )
        for statement, failure in attempts:
            try:
                with self.database.cursor() as cursor:
                    cursor.execute(statement)
                self._index_dimension = width
                return
            except Exception as exc:  # noqa: BLE001 - any driver error means "try the next one"
                if failure is None:
                    continue
                message = f"{failure}，已退化为精确全表扫描：{exc}"
                if message not in self._degradations:
                    self._degradations.append(message)
                logger.warning(message)
        self._index_dimension = width


def _parse_vector_literal(text: str, width: int) -> tuple[float, ...]:
    """Parse pgvector's ``[1,2,3]`` text form back into floats."""
    body = str(text or "").strip().strip("[]")
    if not body:
        return ()
    values = [float(part) for part in body.split(",")]
    if len(values) != width:
        raise ValueError(f"向量宽度不一致：声明 {width}，实际 {len(values)}")
    return tuple(values)


class PostgresLexicalStore(_PostgresStoreBase):
    """Full-text search over a column this project tokenized itself.

    The search column is a ``tsvector`` built from *this project's* tokens by
    :mod:`.scoring`, not from raw text by PostgreSQL's parser. That is the
    single most important line in this class: the server's default parser
    treats a run of Chinese as one word, so an index built on raw text would
    recall nothing for every Chinese query this product is built to serve —
    and it would do so only on the hosted tier, which is the one users are
    less likely to be running.
    """

    TABLE = "store_lexical"

    def __init__(self, database: PostgresDatabase) -> None:
        super().__init__(database)
        self._degradations: list[str] = []

    def initialize(self) -> None:
        with self.database.cursor() as cursor:
            cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {self.database.schema}")
            cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    chunk_id {SORT_KEY} PRIMARY KEY,
                    document_id {SORT_KEY} NOT NULL DEFAULT '',
                    text TEXT NOT NULL,
                    normalized TEXT NOT NULL,
                    tokens TEXT NOT NULL,
                    search tsvector NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb
                )
            """)
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_lexical_search ON {self.table} "
                           "USING GIN (search)")
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_lexical_document ON {self.table} "
                           "(document_id)")
            # The trigram index is provisioned but not queried by this
            # module. The plan's tier table names ``pg_trgm`` as part of the
            # Hosted tier, and fuzzy recall is T1.4's job rather than the
            # storage layer's: a partial-matching path that only one tier can
            # answer is exactly the drift T1.1 exists to prevent. Creating
            # the extension now means T1.4 does not have to arrange a
            # migration on a live database to add it.
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_lexical_trgm ON {self.table} "
                           f"USING GIN (tokens gin_trgm_ops)")

    def capabilities(self) -> StoreCapabilities:
        return StoreCapabilities(
            tier=self.tier,
            vector_engine=VECTOR_ENGINE_MATRIX,
            lexical_engine=LEXICAL_ENGINE_FTS,
            # Spelled out rather than left to the dataclass default, because
            # the default is True and True is a claim: "this store answers a
            # query by reading the whole corpus". This tier does not — it
            # narrows through a GIN index. Leaving it unset made the Hosted
            # tier report ``scan_only=True``, i.e. the personal tier's degraded
            # state, in the one field a health endpoint and an operator both
            # read to decide whether a deployment is slow. A tier claiming a
            # degradation it does not have is as wrong as one hiding one.
            scan_only=False,
            degradations=tuple(self._degradations),
        )

    def _note_overlong_lexeme(self, tokens: str) -> None:
        """Record that a token this store indexed is too long to be searchable.

        A genuine PostgreSQL limitation, recorded rather than worked around.
        ``to_tsvector`` refuses to index a lexeme longer than 2047 bytes and
        says so with a NOTICE that no caller in this process ever sees; the
        row is written, the GIN index is maintained, and the term is simply
        absent from it. SQLite's FTS5 has no such limit, so the same chunk is
        findable on the personal tier and not on this one — the two tiers
        answering the same contract differently, which is the failure this
        module exists to prevent.

        It cannot be fixed by shortening the term: truncating it would index a
        word the document does not contain and return a hit for it, which is
        worse than returning nothing because it looks like a real match. Nor by
        scanning instead, because the term is not stored anywhere a scan could
        look — the server discarded it. So the honest answer is the one
        :mod:`.interfaces` asks for: say which tier is degraded, in a field a
        human reads. What is *not* done is pretending the tiers agree; the
        parity test asserts that they do not, and names which one is missing
        what.

        The limit is per *lexeme*, not per document, so the measurement is the
        longest single token. Measuring the joined string instead would fire
        on essentially every real chunk — a paragraph is thousands of bytes of
        perfectly indexable words — and the degradation would be reported for
        corpora that lose nothing at all. The tokens are space-joined by the
        caller and contain no spaces themselves, because
        :func:`.scoring.tokenize` emits one term per whitespace run and
        :func:`.scoring.normalize` has already replaced every space inside a
        term.
        """
        if not any(len(token.encode("utf-8")) > MAX_LEXEME_BYTES for token in tokens.split()):
            return
        message = (
            f"词法索引存在超过 {MAX_LEXEME_BYTES} 字节的词条，"
            "PostgreSQL 会拒绝将其写入 tsvector，因此该词条在 Hosted 档位不可检索"
            "（个人档位的 FTS5 无此限制）。"
            "这不是可配置项：截断会索引出文档中不存在的词，"
            "而服务器并未保留原词，扫描路径也找不回来。"
        )
        if message not in self._degradations:
            self._degradations.append(message)
            logger.warning(message)

    def index_chunk(
        self,
        chunk_id: str,
        text: str,
        *,
        document_id: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        # strip() matters and this tier did not have it: a whitespace-only
        # chunk_id is a blank identifier, and the SQLite tier refuses one.
        # Without the strip the Hosted tier stored a row keyed on "  " and
        # returned a hit for it, while the personal tier raised -- the two
        # tiers disagreeing about a contract, which is the single thing the
        # shared contract suite exists to make impossible.
        identifier = str(chunk_id).strip()
        if not identifier:
            raise ValueError("chunk_id 不能为空。")
        body = str(text or "")
        tokens = " ".join(scoring.tokenize(body))
        # Checked here, at the one moment the offending term is in hand. The
        # NOTICE the server would otherwise emit goes to the database's log,
        # not to the process that is about to serve a query which will come
        # back empty, so the only place this can be surfaced is a capability
        # the caller of this method may never look at again.
        self._note_overlong_lexeme(tokens)
        with self.database.cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO {self.table}
                    (chunk_id, document_id, text, normalized, tokens, search, metadata)
                VALUES (%s, %s, %s, %s, %s, to_tsvector('simple', %s), %s::jsonb)
                ON CONFLICT (chunk_id) DO UPDATE
                SET document_id = EXCLUDED.document_id,
                    text = EXCLUDED.text,
                    normalized = EXCLUDED.normalized,
                    tokens = EXCLUDED.tokens,
                    search = EXCLUDED.search,
                    metadata = EXCLUDED.metadata
                """,
                (
                    identifier,
                    str(document_id),
                    body,
                    scoring.normalize(body),
                    tokens,
                    tokens,
                    _dumps(dict(metadata or {})),
                ),
            )

    def delete_chunk(self, chunk_id: str) -> bool:
        with self.database.cursor() as cursor:
            cursor.execute(f"DELETE FROM {self.table} WHERE chunk_id = %s", (str(chunk_id),))
            return bool(cursor.rowcount)

    def delete_document(self, document_id: str) -> int:
        with self.database.cursor() as cursor:
            cursor.execute(f"DELETE FROM {self.table} WHERE document_id = %s", (str(document_id),))
            return int(cursor.rowcount or 0)

    def search(
        self,
        query: str,
        top_k: int = 10,
        *,
        document_ids: Sequence[str] | None = None,
    ) -> list[LexicalHit]:
        prepared = scoring.LexicalQuery.prepare(query)
        limit = max(0, int(top_k))
        if limit == 0 or prepared.is_empty:
            return []
        sql = (
            "SELECT chunk_id, document_id, text, normalized, tokens "
            f"FROM {self.table} WHERE search @@ to_tsquery('simple', %s)"
        )
        params: list[Any] = [_tsquery(prepared)]
        if document_ids is not None:
            keys = [str(item) for item in document_ids]
            if not keys:
                return []
            sql += " AND document_id = ANY(%s)"
            params.append(keys)
        sql += ' ORDER BY chunk_id COLLATE "C" ASC'
        with self.database.cursor() as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
        hits = []
        for chunk_id, document_id, text, normalized, tokens in rows:
            score = scoring.score_lexical(prepared, normalized, tokens.split())
            if score > 0.0:
                hits.append(
                    LexicalHit(
                        chunk_id=chunk_id,
                        score=score,
                        document_id=document_id,
                        snippet=scoring.make_snippet(text, prepared),
                    )
                )
        hits.sort(key=lambda hit: scoring.sort_key(hit.score, hit.chunk_id))
        return hits[:limit]

    def count_chunks(self) -> int:
        with self.database.cursor() as cursor:
            cursor.execute(f"SELECT COUNT(*) FROM {self.table}")
            return int(cursor.fetchone()[0])


def _tsquery(prepared: scoring.LexicalQuery) -> str:
    """Render a prepared query as an OR of its distinct terms.

    Each lexeme is double-quoted, so an operator character inside a term
    cannot be read as tsquery syntax — a query of ``"c++"`` must not parse as
    the ``&`` operator applied to nonsense.

    Plain terms, not ``:*`` prefixes. A prefix query would put the same rows
    in the candidate set, and the shared scorer would then drop every one of
    them at zero, because the document's term is ``transformer`` and the
    query's is ``transfor``. So the prefix form bought nothing and cost the
    GIN index: PostgreSQL cannot use it for an ordinary term match, and this
    column is the one thing on the Hosted tier that is indexed for the sole
    purpose of answering a query faster.
    """
    return " | ".join(f'"{token}"' for token in sorted(prepared.unique_tokens))


#: Environment variables checked, in order, for a Hosted-tier DSN. The
#: project-specific name wins over the two generic ones so a deployment can
#: point the store somewhere else without redefining ``DATABASE_URL`` for
#: everything else in the stack.
DEFAULT_DSN_ENV_VARS = ("PAPERWHISPERER_PG_DSN", "DATABASE_URL", "POSTGRES_DSN")


def looks_like_dsn(value: str) -> bool:
    """Whether ``value`` could be a PostgreSQL DSN at all.

    A ``file:`` URL is the interesting rejection: SQLite accepts one and
    PostgreSQL does not, so a DSN copied out of a personal-tier config would
    otherwise fail at connect time with a message about an unknown option
    rather than about the wrong tier.
    """
    text = str(value or "").strip()
    return bool(text) and not text.startswith("file:")


def default_sqlite_path() -> str:
    """Where the personal tier's database lives when nothing says otherwise.

    ``context/`` is the folder the project already treats as its mutable
    state, so the store joins the sessions that are there rather than
    inventing a directory a backup script would not know to copy.
    """
    return os.path.join("context", "stores", "paperwhisperer.sqlite3")
