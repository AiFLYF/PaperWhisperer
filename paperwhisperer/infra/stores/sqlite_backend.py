"""The personal tier: one SQLite file, no server, no Docker.

This is the default tier and the one that has to work on a laptop with
nothing installed but the project's own dependencies, so the two rules that
shaped this module are *no external service* and *no silent degradation*.

**Everything is in one file.** Documents, chunks, the fact layer, the vectors
and the lexical index all live in the same SQLite database, opened once
per thread. Splitting them into several files would buy nothing at this scale
and would make "copy the data directory to back up the installation" — the
operation the plan asks T2.6 to document — a multi-file affair.

**Each row is stored as the JSON its canonical type produces.** The obvious
alternative is a column per field, and it is wrong here for a specific
reason: :class:`~paperwhisperer.ingestion.schema.ParsedDocument` is a nested
structure that the schema module explicitly promises to be able to evolve. A
column-per-field table turns every schema change into a migration, and a
migration is code that has to be right for rows written by a build nobody is
running any more. Storing the canonical payload whole means the reader is the
same code the writer used, so a document written by an older build is read by
that build's own round-trip rules and rejected loudly if it is incompatible.
A few columns are *also* written — the ones ordering and filtering need — and
they are not a second source of truth, because they are copied from the
payload on write and never read as anything but a sort key.

**WAL, and why it is not optional.** The default rollback journal makes
every writer block every reader for the duration of its transaction. The
indexing path writes vectors in batches while a request may be reading
chunks, and under the default journal those two serialise behind each other
with a busy error at the end. WAL lets them overlap, and ``busy_timeout``
turns the remaining contention into a wait instead of an exception.

**numpy is used if it is there, and its absence is announced.** The pure
Python path computes exactly the same scores, because the arithmetic is
spelled out in :mod:`.scoring` and numpy is only ever asked for a matrix
product. What changes is the constant factor, which is roughly two orders of
magnitude — so a build without numpy is a supported configuration with a
documented cliff, not a broken one, and
:attr:`~.interfaces.StoreCapabilities.vector_engine` says which one you got.

The full-text accelerator is probed rather than assumed. SQLite's FTS5 module
is a compile-time option, and the stock wheels on most platforms have it
while a hand-built libsqlite3 often does not. The probe happens once, at
construction; the result decides the candidate strategy and is recorded on
:class:`~.interfaces.StoreCapabilities` — see :mod:`.interfaces` for why a
silent fallback is not acceptable.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from paperwhisperer.ingestion.schema import Chunk, PaperFacts, ParsedDocument

from . import scoring
from .interfaces import (
    LEXICAL_ENGINE_FTS,
    LEXICAL_ENGINE_SCAN,
    TIER_PERSONAL,
    VECTOR_ENGINE_MATRIX,
    VECTOR_ENGINE_PURE_PYTHON,
    DocumentRecord,
    LexicalHit,
    StoreCapabilities,
    VectorEntry,
    VectorHit,
)

logger = logging.getLogger(__name__)

try:  # pragma: no cover - the branch taken depends on the installed environment
    import numpy as _numpy
except ImportError:  # pragma: no cover
    _numpy = None

#: How long a blocked statement waits for the write lock before giving up.
#: Long enough to ride out a batch index write, short enough that a test
#: suite never appears to hang: a genuine deadlock is a bug this does not
#: paper over.
BUSY_TIMEOUT_MS = 30_000

#: Vector full-text modes. ``auto`` probes, ``on`` requires the accelerator
#: and fails loudly if it is missing, ``off`` uses the scan path even when the
#: accelerator would work. ``off`` exists so the degraded path can be
#: exercised on a machine that *does* have the accelerator — a degradation
#: branch that can only be tested on a machine without the dependency is a
#: degradation branch nobody runs.
LEXICAL_MODE_AUTO = "auto"
LEXICAL_MODE_REQUIRED = "on"
LEXICAL_MODE_SCAN = "off"
LEXICAL_MODES = (LEXICAL_MODE_AUTO, LEXICAL_MODE_REQUIRED, LEXICAL_MODE_SCAN)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _dumps(payload: Any) -> str:
    """Serialize with sorted keys and no ASCII escaping.

    Sorted keys make a stored payload byte-comparable, which is what lets a
    test assert that a document came back *exactly* as it went in rather than
    merely equivalently. ``ensure_ascii=False`` keeps CJK text readable in the
    file; SQLite stores text as UTF-8 and this is not a size problem at
    personal scale.
    """
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


class SqliteDatabase:
    """Per-thread connections to one SQLite file, with WAL enabled.

    Connections are per thread because SQLite serializes writers at the file
    level anyway, and one shared connection would have to be wrapped in a
    lock to be safe — giving up exactly the reader/writer overlap WAL exists
    to provide. A connection is created on first use by each thread and closed
    together by :meth:`close`.

    An in-memory database is the exception: ``:memory:`` gives every new
    connection a *different, empty* database, which would make a
    multi-threaded test see an empty store and read as a lost-write bug. So
    for that one path a single connection is shared behind a lock.
    """

    def __init__(self, path: str | os.PathLike[str], *, timeout: float = 30.0) -> None:
        self.path = str(path)
        self.timeout = float(timeout)
        self._local = threading.local()
        self._connections: list[sqlite3.Connection] = []
        # The thread that opened each connection, keyed by ``id``. ``close()``
        # needs it to tell a handle nobody can be using (its owner already
        # finished) from one a live thread is mid-``execute()`` on, and that
        # difference is the whole of a safe teardown. See ``close()``.
        self._owners: dict[int, threading.Thread] = {}
        self._lock = threading.RLock()
        self._closed = False
        self._shared: sqlite3.Connection | None = None
        self._is_memory = self.path == ":memory:" or self.path.startswith("file::memory:")
        # Keeps the single in-memory connection alive past ``close()`` so
        # refcounting cannot free it under a thread still executing on it. At
        # most one per database, so this is a bounded, deliberate retention
        # reclaimed by process exit rather than an unbounded leak.
        self._retired: list[sqlite3.Connection] = []

    def _configure(self, connection: sqlite3.Connection) -> sqlite3.Connection:
        connection.execute("PRAGMA journal_mode=WAL")
        # NORMAL rather than FULL: a crash can lose the last transactions, and
        # in exchange a page write is not an fsync. Everything in this file is
        # re-derivable — a lost index batch means re-indexing, not losing a
        # user's upload — and FULL would cost an fsync per chunk.
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        return connection

    def _open(self) -> sqlite3.Connection:
        if not self._is_memory:
            parent = os.path.dirname(os.path.abspath(self.path))
            if parent:
                os.makedirs(parent, exist_ok=True)
        connection = sqlite3.connect(
            self.path,
            timeout=self.timeout,
            # Autocommit: an explicit BEGIN is issued only where a group of
            # writes has to be atomic, so a single-row write costs one statement.
            isolation_level=None,
            check_same_thread=False,
        )
        return self._configure(connection)

    def connection(self) -> sqlite3.Connection:
        """The calling thread's connection, opened on first use."""
        if self._closed:
            raise RuntimeError("数据库连接已关闭。")
        if self._is_memory:
            with self._lock:
                if self._shared is None:
                    self._shared = self._open()
                return self._shared
        existing = getattr(self._local, "connection", None)
        if existing is None:
            existing = self._open()
            self._local.connection = existing
            with self._lock:
                self._connections.append(existing)
                self._owners[id(existing)] = threading.current_thread()
        return existing

    def close(self) -> None:
        """Close every connection this object opened. Idempotent, never raises.

        **A connection is only ever closed by the thread that is using it, or
        after that thread has finished.** This is the difference between an
        orderly teardown and a SIGSEGV.

        ``check_same_thread=False`` means any thread may be inside
        ``connection.execute()`` on a handle this object handed out, and
        SQLite's C layer frees that handle the instant ``.close()`` is called.
        Calling it from a second thread while the first is mid-``execute()`` is a
        use-after-free, and CPython does not raise — the interpreter dies. That
        is not hypothetical here: a :class:`~paperwhisperer.infra.jobs.JobWorker`
        runs on its own thread, and a bounded ``stop()`` join returns while a
        long parse is still running, so the worker can still be executing on its
        connection when the owning bundle is torn down.

        So each connection remembers which thread opened it, and ``close()``
        splits the two cases:

        * **owner finished, or is this thread** — nobody can be using it, so it
          is closed here. This is the ordinary path and the only one that
          releases a handle eagerly.
        * **owner still running** — the reference is dropped but ``.close()``
          is *not* called. The connection stays alive behind the owning thread's
          ``threading.local``, and is reclaimed when that thread finishes and
          its thread-local is collected. Closing it here is the crash.

        The in-memory shared connection has no single owner — every thread uses
        the same handle — so no ownership test can prove it is idle. It is moved
        to ``_retired`` and left to process exit instead, which is bounded (one
        per database) and always safe.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            candidates = list(self._connections)
            self._connections.clear()
            shared, self._shared = self._shared, None
            current = threading.current_thread()
            closing: list[sqlite3.Connection] = []
            for connection in candidates:
                owner = self._owners.pop(id(connection), None)
                if owner is None or owner is current or not owner.is_alive():
                    closing.append(connection)
                else:
                    logger.debug(
                        "SQLite 连接仍属于运行中的线程 %s，交由该线程回收而非在此关闭。",
                        owner.name,
                    )
            if shared is not None:
                self._retired.append(shared)
        for connection in closing:
            try:
                connection.close()
            except sqlite3.Error:  # pragma: no cover - close must not raise
                pass


class _SqliteStoreBase:
    """Shared plumbing: the database handle, the logger, and ``close``."""

    tier = TIER_PERSONAL

    def __init__(self, database: SqliteDatabase) -> None:
        self.database = database

    @property
    def connection(self) -> sqlite3.Connection:
        return self.database.connection()

    def close(self) -> None:
        """No-op per store; the owning :class:`SqliteDatabase` closes.

        The protocol requires an idempotent, non-raising close on every
        store, and having three of them each close the same file would make
        the order they are called in significant.
        """


_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS documents (
        document_id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        payload TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_documents_order ON documents(created_at, document_id)",
    """
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id TEXT PRIMARY KEY,
        document_id TEXT NOT NULL,
        chunk_order INTEGER NOT NULL,
        payload TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id, chunk_order, chunk_id)",
    """
    CREATE TABLE IF NOT EXISTS paper_facts (
        document_id TEXT PRIMARY KEY,
        payload TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS vectors (
        chunk_id TEXT PRIMARY KEY,
        document_id TEXT NOT NULL DEFAULT '',
        dim INTEGER NOT NULL,
        vector BLOB NOT NULL,
        metadata TEXT NOT NULL DEFAULT '{}'
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_vectors_document ON vectors(document_id)",
    """
    CREATE TABLE IF NOT EXISTS lexical_chunks (
        chunk_id TEXT PRIMARY KEY,
        document_id TEXT NOT NULL DEFAULT '',
        text TEXT NOT NULL,
        normalized TEXT NOT NULL,
        tokens TEXT NOT NULL,
        metadata TEXT NOT NULL DEFAULT '{}'
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_lexical_document ON lexical_chunks(document_id)",
)


class SqliteMetadataStore(_SqliteStoreBase):
    """Documents, chunks and the fact layer, in the one SQLite file."""

    def __init__(self, database: SqliteDatabase) -> None:
        super().__init__(database)
        # A metadata store has no opinion about how the other two work, so it
        # reports the tier and nothing else. The composite picture is
        # assembled by the factory, which has all three in hand.
        self._capabilities = StoreCapabilities(tier=self.tier)

    def initialize(self) -> None:
        connection = self.connection
        for statement in _SCHEMA_STATEMENTS:
            connection.execute(statement)

    def capabilities(self) -> StoreCapabilities:
        return self._capabilities

    # -- documents ---------------------------------------------------------

    def upsert_document(self, record: DocumentRecord) -> DocumentRecord:
        connection = self.connection
        identifier = str(record.document_id or "").strip()
        if not identifier:
            raise ValueError("document_id 不能为空。")
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = connection.execute(
                "SELECT payload FROM documents WHERE document_id = ?", (identifier,)
            ).fetchone()
            previous_version = 0
            if row is not None:
                previous_version = int(json.loads(row[0]).get("version") or 0)
            stored = DocumentRecord.from_dict(record.to_dict())
            stored.version = previous_version + 1
            stored.created_at = record.created_at or _now()
            stored.updated_at = _now()
            if stored.document is None:
                stored.document = ParsedDocument(document_id=identifier)
                stored.document.document_id = identifier
            connection.execute(
                "INSERT INTO documents(document_id, created_at, payload) VALUES (?, ?, ?) "
                "ON CONFLICT(document_id) DO UPDATE SET created_at = excluded.created_at, "
                "payload = excluded.payload",
                (identifier, stored.created_at, _dumps(stored.to_dict())),
            )
        finally:
            connection.execute("COMMIT")
        return stored

    def get_document(self, document_id: str) -> DocumentRecord | None:
        row = self.connection.execute(
            "SELECT payload FROM documents WHERE document_id = ?", (str(document_id),)
        ).fetchone()
        return DocumentRecord.from_dict(json.loads(row[0])) if row is not None else None

    def list_documents(self, limit: int = 50, offset: int = 0) -> list[DocumentRecord]:
        # Newest first, but ordered *by* the same key it is sorted on. The
        # document_id tiebreak matters: two documents written in the same
        # second are ordinary under a concurrent index, and without it the two
        # tiers can page a result set in different orders.
        rows = self.connection.execute(
            "SELECT payload FROM documents ORDER BY created_at DESC, document_id ASC "
            "LIMIT ? OFFSET ?",
            (max(0, int(limit)), max(0, int(offset))),
        ).fetchall()
        return [DocumentRecord.from_dict(json.loads(row[0])) for row in rows]

    def delete_document(self, document_id: str) -> bool:
        """Delete the document, its chunks and its facts, in one transaction.

        Only the tables this store owns. The vectors and the lexical index
        belong to their own stores, and reaching across into them from here
        would make each store responsible for another's schema — so a caller
        deleting a document for good wants
        :meth:`~.factory.StoreBundle.delete_document`, which does all three
        in the right order and is the one call the API layer should make.
        """
        connection = self.connection
        identifier = str(document_id)
        connection.execute("BEGIN IMMEDIATE")
        try:
            existing = connection.execute(
                "SELECT 1 FROM documents WHERE document_id = ?", (identifier,)
            ).fetchone()
            connection.execute("DELETE FROM documents WHERE document_id = ?", (identifier,))
            connection.execute("DELETE FROM chunks WHERE document_id = ?", (identifier,))
            connection.execute("DELETE FROM paper_facts WHERE document_id = ?", (identifier,))
        finally:
            connection.execute("COMMIT")
        return existing is not None

    def count_documents(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0])

    # -- chunks ------------------------------------------------------------

    def put_chunk(self, document_id: str, chunk: Chunk) -> Chunk:
        stored = Chunk.from_dict(chunk.to_dict())
        if not stored.chunk_id:
            raise ValueError("chunk_id 不能为空。")
        self.connection.execute(
            "INSERT INTO chunks(chunk_id, document_id, chunk_order, payload) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(chunk_id) DO UPDATE SET document_id = excluded.document_id, "
            "chunk_order = excluded.chunk_order, payload = excluded.payload",
            (stored.chunk_id, str(document_id), int(stored.order), _dumps(stored.to_dict())),
        )
        return stored

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        row = self.connection.execute(
            "SELECT payload FROM chunks WHERE chunk_id = ?", (str(chunk_id),)
        ).fetchone()
        return Chunk.from_dict(json.loads(row[0])) if row is not None else None

    def list_chunks(self, document_id: str) -> list[Chunk]:
        rows = self.connection.execute(
            "SELECT payload FROM chunks WHERE document_id = ? "
            "ORDER BY chunk_order ASC, chunk_id ASC",
            (str(document_id),),
        ).fetchall()
        return [Chunk.from_dict(json.loads(row[0])) for row in rows]

    def delete_chunks(self, document_id: str) -> int:
        cursor = self.connection.execute(
            "DELETE FROM chunks WHERE document_id = ?", (str(document_id),)
        )
        return int(cursor.rowcount or 0)

    # -- facts -------------------------------------------------------------

    def put_facts(self, document_id: str, facts: PaperFacts) -> PaperFacts:
        stored = PaperFacts.from_dict(facts.to_dict())
        self.connection.execute(
            "INSERT INTO paper_facts(document_id, payload) VALUES (?, ?) "
            "ON CONFLICT(document_id) DO UPDATE SET payload = excluded.payload",
            (str(document_id), _dumps(stored.to_dict())),
        )
        return stored

    def get_facts(self, document_id: str) -> PaperFacts | None:
        row = self.connection.execute(
            "SELECT payload FROM paper_facts WHERE document_id = ?", (str(document_id),)
        ).fetchone()
        return PaperFacts.from_dict(json.loads(row[0])) if row is not None else None


class SqliteVectorStore(_SqliteStoreBase):
    """Embeddings as float32 BLOBs, searched by brute force.

    There is no index here and that is the honest engineering answer for a
    single user: a few thousand chunks times a few hundred dimensions is a
    few million multiply-adds, which numpy does in single-digit milliseconds,
    and an approximate index would have to be built, tuned, invalidated on
    every re-index, and would then return slightly different results on
    different days. Exact and fast enough beats approximate and fast.
    """

    def __init__(self, database: SqliteDatabase) -> None:
        super().__init__(database)
        self._degradations: list[str] = []
        engine = VECTOR_ENGINE_MATRIX if _numpy is not None else VECTOR_ENGINE_PURE_PYTHON
        if _numpy is None:
            self._degradations.append(
                "向量检索退化为纯 Python 暴力比对：未安装 numpy，"
                "单次查询耗时约为矩阵运算的百倍量级。"
            )
        # No lexical_engine here: this store has no opinion about how text is
        # searched, and a default would read as a measurement. The factory
        # reads each field from the store that can actually answer for it.
        self._capabilities = StoreCapabilities(
            tier=self.tier,
            vector_engine=engine,
            degradations=tuple(self._degradations),
        )

    def initialize(self) -> None:
        for statement in _SCHEMA_STATEMENTS:
            self.connection.execute(statement)
        if self._degradations:
            for message in self._degradations:
                logger.warning(message)

    def capabilities(self) -> StoreCapabilities:
        return self._capabilities

    @property
    def dimension(self) -> int | None:
        row = self.connection.execute("SELECT dim FROM vectors LIMIT 1").fetchone()
        return int(row[0]) if row is not None else None

    def upsert_vector(
        self,
        chunk_id: str,
        vector: Sequence[float],
        *,
        document_id: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        # Stripped and checked, for the same reason the lexical store does it:
        # an id of "  " would otherwise become a real row that no document
        # owns and no query can name, and nothing would ever report it.
        identifier = str(chunk_id).strip()
        if not identifier:
            raise ValueError("chunk_id 不能为空。")
        values = scoring.l2_normalize(vector)
        self._assert_dimension(len(values))
        self.connection.execute(
            "INSERT INTO vectors(chunk_id, document_id, dim, vector, metadata) "
            "VALUES (?, ?, ?, ?, ?) ON CONFLICT(chunk_id) DO UPDATE SET "
            "document_id = excluded.document_id, dim = excluded.dim, "
            "vector = excluded.vector, metadata = excluded.metadata",
            (
                identifier,
                str(document_id),
                len(values),
                scoring.pack_f32(values),
                _dumps(dict(metadata or {})),
            ),
        )

    def upsert_many(self, entries: Iterable[VectorEntry]) -> int:
        """Write a batch in one transaction.

        Autocommit would fsync once per row. The batch exists because the
        indexing path produces one per chunk and a paper has thousands.
        """
        materialized = list(entries)
        connection = self.connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            for entry in materialized:
                self.upsert_vector(
                    entry.chunk_id,
                    entry.vector,
                    document_id=entry.document_id,
                    metadata=entry.metadata,
                )
        finally:
            connection.execute("COMMIT")
        return len(materialized)

    def get_vector(self, chunk_id: str) -> VectorEntry | None:
        row = self.connection.execute(
            "SELECT chunk_id, document_id, vector, metadata FROM vectors WHERE chunk_id = ?",
            (str(chunk_id),),
        ).fetchone()
        if row is None:
            return None
        return VectorEntry(
            chunk_id=row[0],
            vector=scoring.unpack_f32(row[2]),
            document_id=row[1],
            metadata=json.loads(row[3]) if row[3] else {},
        )

    def delete_vector(self, chunk_id: str) -> bool:
        cursor = self.connection.execute(
            "DELETE FROM vectors WHERE chunk_id = ?", (str(chunk_id),)
        )
        return bool(cursor.rowcount)

    def delete_document(self, document_id: str) -> int:
        cursor = self.connection.execute(
            "DELETE FROM vectors WHERE document_id = ?", (str(document_id),)
        )
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
        rows = self._candidates(document_ids)
        if not rows:
            return []
        scores = self._scores(query, [scoring.unpack_f32(row[2]) for row in rows])
        hits = [
            VectorHit(chunk_id=row[0], score=float(score), document_id=row[1])
            for row, score in zip(rows, scores, strict=True)
            if min_score is None or score >= float(min_score)
        ]
        return _rank_hits(hits, limit)

    def count_vectors(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM vectors").fetchone()[0])

    # -- internals ---------------------------------------------------------

    def _assert_dimension(self, width: int) -> None:
        existing = self.dimension
        if existing is not None and existing != width:
            raise ValueError(
                f"向量维度 {width} 与索引已建立的维度 {existing} 不一致。"
                "更换 embedding 模型后必须重建索引，而不是写入维度不同的向量。"
            )

    def _candidates(self, document_ids: Sequence[str] | None) -> list[tuple]:
        sql = "SELECT chunk_id, document_id, vector FROM vectors"
        params: list[Any] = []
        if document_ids is not None:
            keys = [str(item) for item in document_ids]
            if not keys:
                return []
            sql += " WHERE document_id IN (" + ", ".join("?" * len(keys)) + ")"
            params.extend(keys)
        return self.connection.execute(sql, params).fetchall()

    @staticmethod
    def _scores(query: tuple[float, ...], vectors: list[tuple[float, ...]]) -> list[float]:
        """Cosine similarity of the query against every candidate.

        The inputs are already unit length, so this is a dot product, and the
        only difference between the two paths is who performs it. numpy is
        asked for a float64 matrix product rather than a float32 one: the
        stored vectors are float32 because that is what was written, but the
        *arithmetic* is done in double precision in both paths so that a score
        is a function of the data rather than of which machine ran the query.
        """
        if _numpy is not None:
            matrix = _numpy.asarray(vectors, dtype=_numpy.float64)
            column = _numpy.asarray(query, dtype=_numpy.float64)
            return [float(value) for value in matrix @ column]
        return [scoring.cosine_similarity(query, vector) for vector in vectors]


def _rank_hits(hits: list[VectorHit], limit: int) -> list[VectorHit]:
    hits.sort(key=lambda hit: scoring.sort_key(hit.score, hit.chunk_id))
    return hits[:limit]


class SqliteLexicalStore(_SqliteStoreBase):
    """Normalized substring matching, accelerated by a full-text index if there is one.

    The two strategies are not interchangeable, and the difference is reported
    rather than papered over:

    ``scan``
        Correct, at a cost linear in the corpus. Matches whole tokens only,
        exactly as the accelerated path does, so the two strategies are
        interchangeable from a caller's point of view.

    ``inverted``
        Faster. It narrows with the query's terms before scoring, which is
        the whole of the speed-up.

    Both find whole tokens and nothing else. A term inside a word — ``"raph"``
    inside ``"graph"`` — matches neither, and neither does a bare prefix,
    ``"transfor"``: the narrowed rows come back and the shared scorer scores
    them at zero, because the document's term is ``transformer`` and the
    query's is ``transfor``.

    That is a real limit, and a decision rather than an omission. Making
    prefix or fuzzy recall work means changing what counts as a match, and if
    only one tier could do it the two would answer differently for the same
    corpus — which is the one thing T1.1 exists to prevent. Recall that
    widens beyond exact terms is a retrieval *strategy*, and it belongs in
    ``rag/retrieve.py`` where it can be switched on for both tiers at once
    (T1.4). Until then the contract is exact terms on both, and the engines
    differ only in how fast they get there.
    """

    def __init__(
        self,
        database: SqliteDatabase,
        *,
        mode: str = LEXICAL_MODE_AUTO,
    ) -> None:
        super().__init__(database)
        self.mode = mode if mode in LEXICAL_MODES else LEXICAL_MODE_AUTO
        self._fts_enabled = False
        # Whether the accelerator table *exists*, which is a different question
        # from whether this build is *using* it. A database that was written
        # by a run with the accelerator on, and is now opened with it off,
        # still holds an index full of entries — and if a later run turns it
        # back on, that index has to have been maintained in between or it
        # silently stops finding documents that are right there. So the two
        # questions are tracked separately and the writes follow the table.
        self._fts_present = False
        self._fts_reason = "尚未初始化。"
        self._degradations: list[str] = []

    def initialize(self) -> None:
        for statement in _SCHEMA_STATEMENTS:
            self.connection.execute(statement)
        self._fts_enabled = self._probe_fts()
        self._fts_present = self._fts_table_exists()
        if not self._fts_enabled and self._degradations:
            for message in self._degradations:
                logger.warning(message)

    def capabilities(self) -> StoreCapabilities:
        return StoreCapabilities(
            tier=self.tier,
            vector_engine=(
                VECTOR_ENGINE_MATRIX if _numpy is not None else VECTOR_ENGINE_PURE_PYTHON
            ),
            lexical_engine=LEXICAL_ENGINE_FTS if self._fts_enabled else LEXICAL_ENGINE_SCAN,
            scan_only=not self._fts_enabled,
            degradations=tuple(self._degradations),
        )

    def index_chunk(
        self,
        chunk_id: str,
        text: str,
        *,
        document_id: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        # Stripped, not merely truth-tested: an id of "  " is a real value to
        # a naive caller and an unreachable one to everyone, and it would
        # silently become a row that no document owns and no query names.
        identifier = str(chunk_id).strip()
        if not identifier:
            raise ValueError("chunk_id 不能为空。")
        body = str(text or "")
        normalized = scoring.normalize(body)
        tokens = " ".join(scoring.tokenize(body))
        connection = self.connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(
                "INSERT INTO lexical_chunks"
                "(chunk_id, document_id, text, normalized, tokens, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(chunk_id) DO UPDATE SET "
                "document_id = excluded.document_id, text = excluded.text, "
                "normalized = excluded.normalized, tokens = excluded.tokens, "
                "metadata = excluded.metadata",
                (
                    identifier,
                    str(document_id),
                    body,
                    normalized,
                    tokens,
                    _dumps(dict(metadata or {})),
                ),
            )
            if self._fts_present:
                connection.execute("DELETE FROM lexical_fts WHERE chunk_id = ?", (identifier,))
                connection.execute(
                    "INSERT INTO lexical_fts(chunk_id, tokens) VALUES (?, ?)",
                    (identifier, tokens),
                )
        finally:
            connection.execute("COMMIT")

    def delete_chunk(self, chunk_id: str) -> bool:
        connection = self.connection
        identifier = str(chunk_id)
        connection.execute("BEGIN IMMEDIATE")
        try:
            existing = connection.execute(
                "SELECT 1 FROM lexical_chunks WHERE chunk_id = ?", (identifier,)
            ).fetchone()
            connection.execute("DELETE FROM lexical_chunks WHERE chunk_id = ?", (identifier,))
            if self._fts_present:
                connection.execute("DELETE FROM lexical_fts WHERE chunk_id = ?", (identifier,))
        finally:
            connection.execute("COMMIT")
        return existing is not None

    def delete_document(self, document_id: str) -> int:
        connection = self.connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            # The index is cleared first, because the id list it is keyed on
            # lives in the table being emptied. The other order deletes the
            # map and then looks for the entries with it.
            if self._fts_present:
                self._sync_fts_deletions(str(document_id))
            cursor = connection.execute(
                "DELETE FROM lexical_chunks WHERE document_id = ?", (str(document_id),)
            )
        finally:
            connection.execute("COMMIT")
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
        hits: list[LexicalHit] = []
        for chunk_id, document_id, text, normalized, tokens in self._candidates(
            prepared, document_ids
        ):
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
        return int(self.connection.execute("SELECT COUNT(*) FROM lexical_chunks").fetchone()[0])

    # -- internals ---------------------------------------------------------

    def _probe_fts(self) -> bool:
        """Decide whether the full-text accelerator is available, and say why not.

        The probe is a real ``CREATE`` against the database rather than a
        version string check, because "the module is compiled in" and "this
        build can actually create the table" are not the same claim, and the
        second is the one that matters.
        """
        if self.mode == LEXICAL_MODE_SCAN:
            self._fts_reason = "已通过 STORAGE_LEXICAL_FTS5=off 显式关闭全文索引。"
            # Recorded even though the operator asked for it. "Degradations"
            # means "differences from the fastest available path", and someone
            # reading a health endpoint has no way to tell an operator's
            # choice from a missing feature unless both are written down.
            self._degradations.append(
                "词法检索使用全表子串扫描："
                f"{self._fts_reason} 查询仍然正确，耗时随语料线性增长。"
            )
            return False
        try:
            self.connection.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS lexical_fts "
                "USING fts5(chunk_id UNINDEXED, tokens, tokenize='unicode61')"
            )
        except sqlite3.OperationalError as exc:
            self._fts_reason = f"当前 SQLite 构建不支持 FTS5：{exc}"
            if self.mode == LEXICAL_MODE_REQUIRED:
                raise RuntimeError(
                    "STORAGE_LEXICAL_FTS5=on 要求全文索引，但当前 SQLite 构建不支持 FTS5："
                    f"{exc}"
                ) from exc
            self._degradations.append(
                "词法检索退化为全表子串扫描："
                f"{self._fts_reason} 查询仍然正确，耗时随语料线性增长。"
            )
            logger.warning("词法检索退化为全表子串扫描：%s", self._fts_reason)
            return False
        self._fts_reason = "已启用 FTS5 全文索引。"
        return True

    def _fts_table_exists(self) -> bool:
        """Whether an accelerator table is present, however it came to be."""
        row = self.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'lexical_fts'"
        ).fetchone()
        return row is not None

    def _sync_fts_deletions(self, document_id: str) -> None:
        """Drop a document's rows from the full-text index.

        The accelerator is kept in step by hand rather than by trigger. A
        trigger would be less code here and would then also have to be
        correct for the write paths, the delete path and the cascade order —
        in a second engine, with its own transaction behaviour, that nobody
        can see. Two explicit statements are easier to be sure of.
        """
        self.connection.execute(
            "DELETE FROM lexical_fts WHERE chunk_id IN "
            "(SELECT chunk_id FROM lexical_chunks WHERE document_id = ?)",
            (document_id,),
        )

    def _candidates(
        self, prepared: scoring.LexicalQuery, document_ids: Sequence[str] | None
    ) -> list[tuple]:
        """Narrow the corpus to the rows worth scoring.

        When the accelerator is present the narrowing is a prefix query over
        terms this module tokenized itself. The stored ``tokens`` column is
        the space-joined output of :func:`.scoring.tokenize`, which is what
        makes that safe: the index is a plain inverted index over exactly the
        terms the scorer knows about, so it cannot disagree with the scorer
        about what a term is. Handing the index the raw document text instead
        would hand it the engine's own tokenizer — which treats a Chinese
        phrase as one term, and would then quietly fail to recall every
        CJK query.
        """
        keys: list[str] | None = None
        if document_ids is not None:
            keys = [str(item) for item in document_ids]
            if not keys:
                return []
        if self._fts_enabled:
            match = " OR ".join(f'"{token}"*' for token in sorted(prepared.unique_tokens))
            sql = (
                "SELECT c.chunk_id, c.document_id, c.text, c.normalized, c.tokens "
                "FROM lexical_fts f JOIN lexical_chunks c ON c.chunk_id = f.chunk_id "
                "WHERE lexical_fts MATCH ?"
            )
            params: list[Any] = [match]
            if keys is not None:
                sql += " AND c.document_id IN (" + ", ".join("?" * len(keys)) + ")"
                params.extend(keys)
            # No LIMIT. The candidate set is deliberately complete, because a
            # truncated one would change which rows make it into the top_k,
            # and a result set that shifts because a query was 3% more
            # selective is a result set nobody can test.
            return self.connection.execute(sql, params).fetchall()
        sql = "SELECT chunk_id, document_id, text, normalized, tokens FROM lexical_chunks"
        params: list[Any] = []
        if keys is not None:
            sql += " WHERE document_id IN (" + ", ".join("?" * len(keys)) + ")"
            params.extend(keys)
        # No substring pre-filter. An earlier version narrowed the scan to rows
        # containing a query term *anywhere*, including inside a word, which
        # looked like a free recall win and was not one: the shared scorer
        # scores a row by whole tokens, so every such row came back at 0.0 and
        # was dropped. The filter cost a full scan and changed no result. The
        # guaranteed contract is whole-token matching on every tier, for the
        # reason set out on the class.
        return self.connection.execute(sql, params).fetchall()
