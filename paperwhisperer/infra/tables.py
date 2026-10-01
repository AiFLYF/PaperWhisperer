"""The non-store tables of T2.2, and the optimistic lock that protects them.

T2.2 names seven tables. Two of them are already somebody else's:
``jobs`` is owned by :mod:`paperwhisperer.infra.jobs` and ``documents`` /
``chunks`` / ``paper_facts`` by
:mod:`~paperwhisperer.infra.stores.sqlite_backend`. Redefining any of them
would be the second source of truth the store tier's own docstrings argue
against — a job row written by a handler and a job row written by a DDL
migration would disagree about what a column means, and the disagreement would
only surface as a job that never reaches a terminal state. So this module owns
exactly the five the plan lists that nothing else does, and
:data:`EXTERNAL_TABLES` names the rest with their owner so that a reader does
not have to grep to find out whether ``documents`` is here.

**The optimistic lock is the deliverable, so it is a required argument and not
a flag.** Every table carries a ``version`` column, and every update method takes
``expected_version`` as a *keyword-only argument with no default*. That signature
is the enforcement. The alternative — ``update(row, expected_version=None)``,
where ``None`` means "just write" — puts the losing path one typo away, and the
losing path is silent: two requests both read version 3, both write, one wins,
and the UI shows the loser's data until the next read. There is no error to
notice. A required argument means the second writer has to *say* which version it
read, and the compare-and-set in the ``WHERE`` clause then rejects it with a
:class:`VersionConflict` it cannot ignore.

The lock is a real ``UPDATE ... WHERE key = ? AND version = ?``, not a
read-then-write in Python, so it holds across processes rather than only across
threads. :meth:`TableStore.update_conversation` and its siblings bump the
version in the same statement that writes the data, which means a row can never
be observed with new data and an old version.

**Why every table here is also on the ``document_id``.** The point of storing
runs, messages and versions is being able to say "everything this document
produced", and a schema where that question needs four tables and a ``UNION``
is a schema where it gets asked with a join nobody writes. The indexes on
``document_id`` are declared in :data:`TABLES` rather than added by hand, so
they are created by the same idempotent ``initialize()`` that creates the tables
— an index is part of a migration or it is a performance regression somebody
discovers in production.

**Idempotence is the acceptance criterion, so it is structural rather than
bookkeeping.** :meth:`TableStore.initialize` issues ``CREATE TABLE IF NOT
EXISTS`` and then, for every declared column the live table does not have, an
``ALTER TABLE ... ADD COLUMN``. There is no migration ledger, no version
counter, and no "have we run this" question — running ``initialize()`` twice,
or on a database an older build created, converges on the same schema and does
not touch a single row. A migration that needs to know whether it already ran is
a migration nobody can safely retry, and this one has to be retried by
construction: ``build_table_store`` calls it, the tests call it, and the app
will call it on every start.

**The columns are declared once, as data, and the DDL is generated from them.**
A hand-written ``CREATE TABLE`` next to a hand-written row dataclass is two
spellings of one schema, and they drift the first time somebody adds a column to
one of them. Here a :class:`Column` list *is* the schema: it generates the DDL
for both engines, it validates every write against the real column set (an
unknown key raises :class:`UnknownColumn` instead of being silently dropped),
and it is what the tests compare against the DDL and against
``AnalysisRun.as_analysis_run_columns()``. The dialect is a small set of
differences — placeholder syntax, four type names, the default literal, and how
a row is read — rather than a second copy of every statement.

**No row type is declared per table.** That is a deliberate omission and the
opposite of this repository's usual style. The store tier can afford
:class:`~paperwhisperer.infra.stores.interfaces.DocumentRecord` because it
rehydrates a Pydantic object with its own compatibility rules. These tables hold
opaque JSON: a session payload, a parse trail, a paper record. A dataclass per
table would be a fifth place to add a column, it would re-type values that were
deliberately stored as JSON, and it would make "what is in this row" a question
answered by a Python file rather than by the DDL. Rows come back as
``dict[str, Any]`` with JSON columns decoded, and the schema is
:data:`TABLES`.

**What these tables deliberately do not do.** They do not cascade into the
object store, and they do not garbage-collect. :meth:`TableStore.
delete_document` reports which object keys the caller should remove and leaves
the bytes alone, because deciding what is still referenced is a question about
this table set — and answering it halfway, from one of two places, is how an
orphan embedding ends up answering queries for a document the user deleted.
T2.3 wires the two together once the job pipeline writes both.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

from .stores.factory import REQUIRED_CONFIG_KEYS, StoreBundle, build_stores, resolve_backend
from .stores.interfaces import BACKEND_POSTGRES, TIER_HOSTED, TIER_PERSONAL
from .stores.postgres_backend import PostgresDatabase
from .stores.sqlite_backend import SqliteDatabase

logger = logging.getLogger(__name__)

Row = dict[str, Any]

#: Bumped when a column's meaning changes in a way a reader must know about.
#: Not a migration counter: :meth:`TableStore.initialize` needs no such thing,
#: and a counter is one more thing that can disagree with the database.
SCHEMA_VERSION = 1


def _now_iso() -> str:
    """UTC, second resolution — the convention the store and job rows write."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    """An opaque row id. Prefixed so a log line names the kind of row.

    Random rather than sequential on purpose: a document id that encodes a
    counter is a document id that tells one user how many documents another
    user has, and these rows are visible in URLs and log files.
    """
    return f"{prefix}-{uuid.uuid4().hex[:20]}"


# --------------------------------------------------------------------------
# Column and table declarations
# --------------------------------------------------------------------------


class Kind:
    """The five column types, named for what they mean rather than how they are stored."""

    TEXT = "TEXT"
    INT = "INT"
    REAL = "REAL"
    JSON = "JSON"
    BOOL = "BOOL"


@dataclass(frozen=True, slots=True)
class Column:
    """One column, declared once and rendered into both engines' DDL.

    ``default`` is the Python value a writer gets when it omits the column, and
    it is also what ``ALTER TABLE ADD COLUMN`` writes to existing rows. Both
    facts together are why every column here has a default: an additive
    migration on a column without one is refused by SQLite, and a migration that
    works on one engine and not the other is exactly the drift the single
    declaration exists to prevent.
    """

    name: str
    kind: str
    default: Any = ""
    nullable: bool = False
    check: str = ""


def TEXT(name: str, default: str = "", *, nullable: bool = False, check: str = "") -> Column:
    return Column(name, Kind.TEXT, default, nullable, check)


def INT(name: str, default: int = 0, *, nullable: bool = False, check: str = "") -> Column:
    return Column(name, Kind.INT, default, nullable, check)


def REAL(name: str, default: float = 0.0, *, nullable: bool = False) -> Column:
    return Column(name, Kind.REAL, default, nullable)


def JSON(name: str, default: Any = None, *, nullable: bool = False) -> Column:
    """A JSON column. ``default=None`` means ``{}``; pass ``[]`` for an array.

    JSON rather than separate columns for anything the product grows: a session
    payload, a parse trail, a paper record. Their shapes are owned by the code
    that writes them and are expected to change, and a column per nested field
    would mean a migration per upstream schema change in
    :mod:`paperwhisperer.search.normalize`.
    """
    return Column(name, Kind.JSON, {} if default is None else default, nullable)


def BOOL(name: str, default: bool = False) -> Column:
    return Column(name, Kind.BOOL, default)


@dataclass(frozen=True, slots=True)
class Index:
    """One index. Named from its table so SQLite's file-global index namespace
    cannot collide with the store tier's ``idx_documents_order``."""

    name: str
    columns: tuple[str, ...]
    unique: bool = False

    @property
    def sql_name(self) -> str:
        return f"idx_{self.name}"


@dataclass(frozen=True, slots=True)
class Table:
    """A table: its columns, its key, and the indexes that make it usable."""

    name: str
    title: str
    purpose: str
    primary_key: str
    columns: tuple[Column, ...]
    indexes: tuple[Index, ...] = ()

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)

    def column(self, name: str) -> Column | None:
        for candidate in self.columns:
            if candidate.name == name:
                return candidate
        return None


#: The five tables this module owns. The plan's list also names ``documents``,
#: ``jobs`` and ``chunks``; see :data:`EXTERNAL_TABLES`.
TABLES: tuple[Table, ...] = (
    Table(
        name="document_versions",
        title="文档版本",
        purpose="一次解析产出的一个版本；重新解析新增一行并把 is_current 移走，"
        "用户的原始上传（object store 的 original 层）永不被改写。",
        primary_key="version_id",
        columns=(
            TEXT("version_id"),
            TEXT("document_id"),
            INT("version_no", 1),
            TEXT("source_filename"),
            TEXT("media_type"),
            TEXT("sha256"),
            INT("page_count"),
            TEXT("original_key"),
            TEXT("parsed_key"),
            TEXT("canonical_key"),
            TEXT("parser_name"),
            TEXT("parser_version"),
            TEXT("options_hash"),
            TEXT("parse_status"),
            TEXT("parse_quality"),
            BOOL("used_cloud"),
            BOOL("cloud_attempted"),
            BOOL("fallback_used"),
            TEXT("index_version"),
            BOOL("is_current", True),
            JSON("warnings", []),
            JSON("parse_meta", {}),
            TEXT("created_at"),
            TEXT("updated_at"),
            INT("version", 1),
        ),
        indexes=(
            Index("document_versions_document", ("document_id", "version_no")),
            Index("document_versions_current", ("document_id", "is_current")),
            Index("document_versions_sha", ("sha256",)),
        ),
    ),
    Table(
        name="conversations",
        title="会话",
        purpose="一个会话（一次上传+分析）的持久化行；payload 保存旧 session JSON 的"
        "逐字副本，是无损迁移与可回滚的依据。",
        primary_key="conversation_id",
        columns=(
            TEXT("conversation_id"),
            TEXT("session_id"),
            TEXT("document_id"),
            TEXT("title"),
            TEXT("status", "active", check="status IN ('active', 'archived')"),
            TEXT("token_hash"),
            JSON("payload", {}),
            TEXT("payload_sha256"),
            TEXT("source_path"),
            TEXT("source_sha256"),
            INT("source_mtime_ns"),
            BOOL("source_migrated"),
            TEXT("migrated_at"),
            TEXT("expires_at"),
            TEXT("created_at"),
            TEXT("updated_at"),
            INT("version", 1),
        ),
        indexes=(
            Index("conversations_session", ("session_id",)),
            Index("conversations_document", ("document_id",)),
            Index("conversations_order", ("created_at", "conversation_id")),
        ),
    ),
    Table(
        name="messages",
        title="消息",
        purpose="一问一答与分析产物，按 conversation_id + sequence 排序；"
        "citations/evidence_ids 保存可回溯的证据引用。",
        primary_key="message_id",
        columns=(
            TEXT("message_id"),
            TEXT("conversation_id"),
            INT("sequence"),
            TEXT("role", "user", check="role IN ('user', 'assistant', 'system')"),
            TEXT("kind", "qa", check="kind IN ('qa', 'analysis', 'note')"),
            TEXT("content"),
            TEXT("answer_mode"),
            TEXT("confidence"),
            JSON("citations", []),
            JSON("evidence_ids", []),
            TEXT("created_at"),
            TEXT("updated_at"),
            INT("version", 1),
        ),
        indexes=(
            Index("messages_conversation", ("conversation_id", "sequence", "message_id")),
        ),
    ),
    Table(
        name="reading_queue",
        title="阅读队列",
        purpose="用户稍后要读的论文列表；position 是用户拖动的顺序，"
        "replace_reading_queue 在一个事务里整体替换，避免读-改-写丢更新。",
        primary_key="queue_item_id",
        columns=(
            TEXT("queue_item_id"),
            TEXT("conversation_id"),
            INT("position"),
            TEXT(
                "status",
                "pending",
                check="status IN ('pending', 'reading', 'done', 'dropped')",
            ),
            TEXT("title"),
            TEXT("doi"),
            TEXT("arxiv_id"),
            TEXT("paper_id"),
            TEXT("url"),
            TEXT("pdf_url"),
            INT("year"),
            TEXT("venue"),
            JSON("authors", []),
            TEXT("abstract"),
            JSON("payload", {}),
            TEXT("added_at"),
            TEXT("updated_at"),
            INT("version", 1),
        ),
        indexes=(
            Index("reading_queue_conversation", ("conversation_id", "position", "queue_item_id")),
            Index("reading_queue_doi", ("doi",)),
        ),
    ),
    Table(
        name="analysis_runs",
        title="分析运行",
        purpose="T1.8 每次分析一行：model/prompt_version/evidence_version 三个坐标，"
        "LLM 与 embedding 的成本，以及旧路径的对照列（legacy_*）。",
        primary_key="run_id",
        columns=(
            TEXT("run_id"),
            TEXT("document_id"),
            TEXT("session_id"),
            TEXT("created_at"),
            TEXT("updated_at"),
            TEXT("model"),
            TEXT("prompt_version"),
            TEXT("evidence_version"),
            INT("llm_calls"),
            INT("llm_input_tokens"),
            INT("llm_output_tokens"),
            REAL("llm_cost", 0.0, nullable=True),
            INT("fact_count"),
            INT("grounded_fact_count"),
            REAL("fact_coverage"),
            JSON("not_stated_fields", []),
            JSON("degradations", []),
            JSON("retrieval_record_ids", []),
            INT("legacy_llm_calls"),
            INT("legacy_input_tokens"),
            JSON("calls", []),
            TEXT("embedding_provider"),
            TEXT("embedding_model"),
            INT("embedding_dimension"),
            TEXT("embedding_index_version"),
            INT("embedding_requests"),
            INT("embedding_texts"),
            INT("embedding_cache_hits"),
            INT("embedding_input_tokens"),
            INT("embedding_output_tokens"),
            REAL("embedding_cost", 0.0, nullable=True),
            INT("version", 1),
        ),
        indexes=(
            Index("analysis_runs_document", ("document_id", "created_at", "run_id")),
            Index("analysis_runs_evidence", ("evidence_version",)),
        ),
    ),
)

#: Name → table, for the callers that take a table name.
TABLE_BY_NAME: dict[str, Table] = {table.name: table for table in TABLES}

#: The plan's other three tables, and who owns each. Present so that a reader
#: asking "where is ``documents``" gets an answer here instead of a grep, and so
#: a test can assert this module's DDL never creates one of them.
EXTERNAL_TABLES: dict[str, str] = {
    "documents": "paperwhisperer.infra.stores（MetadataStore.upsert_document）",
    "chunks": "paperwhisperer.infra.stores（MetadataStore.put_chunk）",
    "paper_facts": "paperwhisperer.infra.stores（MetadataStore.put_facts）",
    "jobs": "paperwhisperer.infra.jobs（_SQLITE_SCHEMA / _POSTGRES_SCHEMA）",
}

#: Columns T1.8 requires on ``analysis_runs``, read from
#: :meth:`paperwhisperer.analysis.paper_facts.AnalysisRun.as_analysis_run_columns`
#: and :meth:`paperwhisperer.llm.embeddings.EmbeddingUsage.as_analysis_run_columns`.
#: Duplicated here on purpose: :meth:`TableStore.record_analysis_run` refuses a
#: run missing any of them, so the requirement survives a refactor of the
#: analysis module. ``tests/test_asset_model.py`` asserts this set is still a
#: subset of the live DDL *and* still equals what those two methods return, so a
#: column added upstream without a home here fails the suite rather than
#: disappearing.
REQUIRED_ANALYSIS_RUN_COLUMNS: tuple[str, ...] = (
    # AnalysisRun.as_analysis_run_columns()
    "run_id",
    "document_id",
    "created_at",
    "model",
    "prompt_version",
    "evidence_version",
    "llm_calls",
    "llm_input_tokens",
    "llm_output_tokens",
    "llm_cost",
    "fact_count",
    "grounded_fact_count",
    "fact_coverage",
    "not_stated_fields",
    "degradations",
    "retrieval_record_ids",
    "legacy_llm_calls",
    "legacy_input_tokens",
    # EmbeddingUsage.as_analysis_run_columns()
    "embedding_provider",
    "embedding_model",
    "embedding_dimension",
    "embedding_index_version",
    "embedding_requests",
    "embedding_texts",
    "embedding_cache_hits",
    "embedding_input_tokens",
    "embedding_output_tokens",
    "embedding_cost",
)

#: Columns this module adds to ``analysis_runs`` beyond the two lists above.
#: Named explicitly so the report and the schema cannot disagree about whether
#: they are load-bearing. ``calls`` is the per-call cost trail from
#: ``AnalysisRun.to_dict()`` — it is in ``to_dict`` but not in
#: ``as_analysis_run_columns``, and it is the only place a run's per-stage
#: breakdown survives, so a cost dashboard can show which stage spent the money.
#: The rest are the optimistic-lock and bookkeeping columns every table carries.
EXTRA_ANALYSIS_RUN_COLUMNS: tuple[str, ...] = ("session_id", "updated_at", "calls", "version")


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class TableError(Exception):
    """Base class for every refusal this module makes."""


class UnknownColumn(TableError):
    """A write named a column the table does not have."""


class MissingColumn(TableError):
    """A write omitted a column with no default, or one T1.8 must supply."""


class UnknownRow(TableError):
    """The row a caller named is not there."""


class DuplicateRow(TableError):
    """A row with that key already exists.

    Separate from the driver's integrity error because the two mean different
    things to a caller: "this key is taken, re-read and decide" is an ordinary
    outcome of two workers racing, and it deserves a name of its own rather than
    arriving as a class name from psycopg.
    """


class VersionConflict(TableError):
    """A compare-and-set failed: somebody else wrote this row first.

    The single most important exception in this module. It is raised instead of
    a silent overwrite, and it carries both versions so a caller can decide
    between re-reading (the other writer's data was fine) and reporting a
    conflict (the caller was editing something the user changed underneath).
    """

    def __init__(self, table: str, key: str, expected: int, actual: int) -> None:
        self.table = table
        self.key = key
        self.expected = int(expected)
        self.actual = int(actual)
        super().__init__(
            f"乐观锁冲突：{table} 的 {key!r} 期望版本 {self.expected}，"
            f"当前版本 {self.actual}。请重新读取后再写入。"
        )


class ConstraintViolation(TableError):
    """A CHECK or NOT NULL constraint refused the row.

    Translated from whichever integrity error the driver raised, because the
    driver's class name is not a contract: ``sqlite3.IntegrityError`` and
    psycopg's ``CheckViolation`` mean the same thing here, and a caller that had
    to know which engine it was on would be a caller whose code differs by tier.
    The write is rolled back by the transaction either way — this exception is
    about *what the failure is called*, not about what was left behind.
    """

    def __init__(self, table: str, reason: str) -> None:
        self.table = table
        self.reason = str(reason)
        super().__init__(f"写入 {table} 违反约束：{self.reason}")


def _is_constraint_failure(exc: BaseException) -> bool:
    """Whether an exception is the engine refusing a row's *values*.

    Matched on the class name rather than on driver imports, because the driver
    is optional: the Hosted half of this module is skipped on a checkout with no
    PostgreSQL driver, and an ``except`` clause that named ``psycopg2.errors``
    would make the personal tier fail to import.
    """
    name = type(exc).__name__
    return name in {
        "IntegrityError",
        "CheckViolation",
        "NotNullViolation",
        "IntegrityConstraintViolation",
        "ForeignKeyViolation",
    }


# --------------------------------------------------------------------------
# Dialects
# --------------------------------------------------------------------------


_SQLITE_TYPES = {
    Kind.TEXT: "TEXT",
    Kind.INT: "INTEGER",
    Kind.REAL: "REAL",
    # SQLite has no JSON type and JSON1 is not guaranteed to exist. TEXT holding
    # canonical JSON is the portable choice; the reader parses it in Python, so
    # neither tier depends on an extension being compiled in.
    Kind.JSON: "TEXT",
    Kind.BOOL: "INTEGER",
}

_POSTGRES_TYPES = {
    Kind.TEXT: "TEXT",
    Kind.INT: "BIGINT",
    Kind.REAL: "DOUBLE PRECISION",
    Kind.JSON: "JSONB",
    Kind.BOOL: "BOOLEAN",
}


def _sql_literal(value: Any, kind: str, *, boolean_as_int: bool, jsonb: bool) -> str:
    """Render a Python default as a SQL literal.

    Hand-written rather than delegated to the driver, because a default has to
    be part of the ``CREATE TABLE`` *text* for both engines and for a test that
    compares the DDL; a parameterised default is not a thing either engine
    accepts.

    The three cases are genuinely different and conflating them is a silent
    corruption: a **string** default is SQL-quoted (``DEFAULT ''``), not
    JSON-quoted — ``json.dumps("")`` is ``'""'`` and the row would come back
    holding two quote characters; a **JSON column's** default is JSON text plus
    a cast, because its value really is a document; and a number or a boolean is
    a bare literal. Getting the string case wrong is invisible in every test that
    passes the column explicitly, and wrong on every row that does not.
    """
    if isinstance(value, bool):
        if boolean_as_int:
            return "1" if value else "0"
        return "TRUE" if value else "FALSE"
    if kind == Kind.JSON:
        text = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return "'" + text.replace("'", "''") + "'" + ("::jsonb" if jsonb else "")
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


class _Dialect:
    """The four differences between the two engines, and nothing else."""

    backend = ""
    tier = ""
    placeholder = "?"
    types: dict[str, str] = {}
    boolean_as_int = True
    #: Whether a JSON default needs an explicit ``::jsonb`` cast to be a JSON
    #: value rather than the string that looks like one.
    jsonb_cast = False
    #: Whether the engine takes an advisory lock for a row being inserted.
    advisory_locks = False

    def qualify(self, name: str) -> str:
        return name

    def scalar(self, value: Any) -> Any:
        """Encode a value used in a predicate, where no column type is known.

        A ``WHERE is_current = ?`` has a column behind it but the call site does
        not name one, so booleans are converted here rather than passed through.
        Python's ``bool`` is an ``int`` and SQLite would accept it, which makes
        the question "does it work" the wrong test — the right test is whether
        both tiers store the same value for the same predicate, and only the
        conversion guarantees that.
        """
        if isinstance(value, bool) and self.boolean_as_int:
            return int(value)
        return value

    def render_column(self, column: Column) -> str:
        parts = [column.name, self.types[column.kind]]
        if not column.nullable:
            parts.append("NOT NULL")
        if column.default is not None:
            literal = _sql_literal(
                column.default,
                column.kind,
                boolean_as_int=self.boolean_as_int,
                jsonb=self.jsonb_cast,
            )
            parts.append(f"DEFAULT {literal}")
        if column.check:
            parts.append(f"CHECK ({column.check})")
        return " ".join(parts)

    def ddl(self, table: Table) -> tuple[str, ...]:
        lines = [f"{self.render_column(column)}" for column in table.columns]
        lines.append(f"PRIMARY KEY ({table.primary_key})")
        body = ",\n    ".join(lines)
        statements = [
            f"CREATE TABLE IF NOT EXISTS {self.qualify(table.name)} (\n    {body}\n)"
        ]
        for index in table.indexes:
            unique = "UNIQUE " if index.unique else ""
            statements.append(
                f"CREATE {unique}INDEX IF NOT EXISTS {self.qualify(index.sql_name)} "
                f"ON {self.qualify(table.name)} ({', '.join(index.columns)})"
            )
        return tuple(statements)

    def existing_columns(self, cursor: Any, table: Table) -> set[str]:
        raise NotImplementedError

    def encode(self, column: Column, value: Any) -> Any:
        if value is None:
            return None
        if column.kind == Kind.JSON:
            return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        if column.kind == Kind.BOOL:
            return int(bool(value)) if self.boolean_as_int else bool(value)
        if column.kind == Kind.INT:
            return int(value)
        if column.kind == Kind.REAL:
            return float(value)
        return str(value)

    def decode(self, column: Column, value: Any) -> Any:
        if value is None:
            return None
        if column.kind == Kind.JSON:
            # PostgreSQL hands back a parsed object; SQLite hands back the text.
            # Both are correct and neither is repaired, because repairing it here
            # would hide a driver that changed shape.
            if isinstance(value, (str, bytes, bytearray)):
                try:
                    return json.loads(value)
                except (TypeError, ValueError):
                    return None
            return value
        if column.kind == Kind.BOOL:
            return bool(value)
        if column.kind == Kind.INT:
            return int(value)
        if column.kind == Kind.REAL:
            return float(value)
        return str(value)


class _SqliteDialect(_Dialect):
    backend = "sqlite"
    tier = TIER_PERSONAL
    placeholder = "?"
    types = _SQLITE_TYPES
    boolean_as_int = True
    advisory_locks = False

    def existing_columns(self, cursor: Any, table: Table) -> set[str]:
        cursor.execute(f"PRAGMA table_info({self.qualify(table.name)})")
        return {str(row[1]) for row in cursor.fetchall()}


class _PostgresDialect(_Dialect):
    backend = "postgres"
    tier = TIER_HOSTED
    placeholder = "%s"
    types = _POSTGRES_TYPES
    boolean_as_int = False
    jsonb_cast = True
    advisory_locks = True

    def __init__(self, database: PostgresDatabase) -> None:
        self.database = database

    def qualify(self, name: str) -> str:
        # Schema-qualified, not search_path: a Hosted deployment whose search_path
        # is wrong would otherwise create the tables in whatever schema comes
        # first and leave the configured one empty, which reads as "the tables
        # do not exist" from the application's point of view.
        return self.database.qualified(name)

    def existing_columns(self, cursor: Any, table: Table) -> set[str]:
        cursor.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s",
            (self.database.schema, table.name),
        )
        return {str(row[0]) for row in cursor.fetchall()}


# --------------------------------------------------------------------------
# The contract
# --------------------------------------------------------------------------


@runtime_checkable
class TableStore(Protocol):
    """Durable rows for versions, conversations, messages, queues and runs.

    Two conventions run through every method and neither is optional.

    **The version argument is required.** Every ``update_*`` takes
    ``expected_version`` as a keyword-only argument with no default, and raises
    :class:`VersionConflict` when the row is no longer at that version. A caller
    that has not read the row cannot write it, which is the whole point.

    **Writes return the stored row.** Not the argument, not a re-read of the
    caller's own object: the row as the database holds it, with the version the
    database assigned. A caller that trusts its own copy is holding a row that
    does not exist, and that is the same class of silent failure the version
    argument exists to prevent.
    """

    def initialize(self) -> None:
        """Create or migrate every table this module owns. Idempotent.

        Additive: a missing column on an existing table is added with its
        default, and no existing row is read, rewritten or deleted. Called by
        :func:`build_table_store` and again by any test that wants a clean
        store.
        """
        ...

    def add_document_version(self, values: Mapping[str, Any]) -> Row:
        """Append a parse version and make it the current one, in one transaction.

        ``version_no`` is assigned here rather than by the caller, and every
        other row for the document has ``is_current`` cleared in the same
        transaction. A caller that could set ``is_current`` itself would be able
        to produce a document with two current versions or none, and the
        "which parse produced this answer" question would have two answers.
        """
        ...

    def get_document_version(self, version_id: str) -> Row | None: ...

    def get_current_document_version(self, document_id: str) -> Row | None: ...

    def list_document_versions(self, document_id: str) -> list[Row]: ...

    def update_document_version(
        self, version_id: str, changes: Mapping[str, Any], *, expected_version: int
    ) -> Row:
        """Patch a version row under the optimistic lock."""
        ...

    def add_conversation(self, values: Mapping[str, Any]) -> Row: ...

    def get_conversation(self, conversation_id: str) -> Row | None: ...

    def get_conversation_by_session(self, session_id: str) -> Row | None: ...

    def list_conversations(self, limit: int = 50, offset: int = 0) -> list[Row]: ...

    def update_conversation(
        self, conversation_id: str, changes: Mapping[str, Any], *, expected_version: int
    ) -> Row:
        """Patch a conversation under the optimistic lock.

        The row a session read-modify-write lands on, so this is the method the
        lost-update test drives.
        """
        ...

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and its messages and queue items. Idempotent."""
        ...

    def append_message(self, values: Mapping[str, Any]) -> Row:
        """Append one message, assigning its sequence within the conversation."""
        ...

    def get_message(self, message_id: str) -> Row | None: ...

    def list_messages(self, conversation_id: str) -> list[Row]:
        """A conversation's messages in order."""
        ...

    def update_message(
        self, message_id: str, changes: Mapping[str, Any], *, expected_version: int
    ) -> Row: ...

    def replace_messages(
        self, conversation_id: str, items: Sequence[Mapping[str, Any]]
    ) -> int:
        """Replace a conversation's whole message list in one transaction.

        The same shape and the same reason as
        :meth:`replace_reading_queue`: a transcript is a list, and rebuilding it
        from its source is only honest if the rebuild is atomic. A
        delete-then-insert pair in a route handler is two commits, and a reader
        between them sees a conversation with no messages — which for an
        evidence-first product looks exactly like a model that answered nothing.

        Used by the session migration, where the source is the session file and
        the projection must match it exactly rather than approximately. Once
        T2.4 owns message writes, the caller of this method changes; the method
        does not.
        """
        ...

    def add_queue_item(self, values: Mapping[str, Any]) -> Row: ...

    def get_queue_item(self, queue_item_id: str) -> Row | None: ...

    def list_queue_items(self, conversation_id: str) -> list[Row]: ...

    def update_queue_item(
        self, queue_item_id: str, changes: Mapping[str, Any], *, expected_version: int
    ) -> Row: ...

    def remove_queue_item(self, queue_item_id: str) -> bool: ...

    def replace_reading_queue(
        self, conversation_id: str, items: Sequence[Mapping[str, Any]]
    ) -> int:
        """Replace a conversation's whole queue in one transaction, returning the count.

        Exists because the queue is a *list*, and every write to a list is a
        read-modify-write: the client sends the order it wants, and the server
        has to make the new order the only order. Delete-then-insert inside one
        transaction is the whole mechanism, and it is atomic on both engines —
        which is the property a blind "delete, then insert" in a route handler
        does not have.
        """
        ...

    def record_analysis_run(
        self, columns: Mapping[str, Any], *, embedding: Mapping[str, Any] | None = None
    ) -> Row:
        """Persist one T1.8 run, requiring every column it names.

        Takes the dicts verbatim as
        :meth:`~paperwhisperer.analysis.paper_facts.AnalysisRun.as_analysis_run_columns`
        and :meth:`~paperwhisperer.llm.embeddings.EmbeddingUsage.as_analysis_run_columns`
        produce them. Every key in :data:`REQUIRED_ANALYSIS_RUN_COLUMNS` must be
        present, and a key the table does not have raises :class:`UnknownColumn`
        rather than being ignored — so a column added to the analysis module
        without a home here fails loudly at the write, which is the only place it
        can be caught.
        """
        ...

    def get_analysis_run(self, run_id: str) -> Row | None: ...

    def list_analysis_runs(self, document_id: str, limit: int = 50) -> list[Row]: ...

    def latest_analysis_run(self, document_id: str) -> Row | None: ...

    def cost_summary(self, document_id: str) -> Row:
        """Totals across a document's runs, with the T1.8 comparison in them.

        ``llm_cost`` totals to ``None`` — not ``0.0`` — when any contributing run
        could not report its cost, because a partial sum of a partly-unknown
        total is a number that looks like a total. ``legacy_*`` are summed the
        same way, so T1.8's "cost went down, here is the data" is answerable
        from stored rows rather than from a measurement somebody repeated once.
        """
        ...

    def object_keys_for_document(self, document_id: str) -> list[Row]:
        """Every object key this document's rows reference, with its layer.

        The join between these tables and the object store, and it is a *report*
        rather than a delete on purpose: the tables do not own the bytes, and a
        store that removed them would be making a reachability decision from one
        of the two places that know the answer. T2.3 calls this and then removes.
        """
        ...

    def delete_document(self, document_id: str) -> Row:
        """Delete every row belonging to a document. Returns what went by table."""
        ...

    def describe(self) -> Row:
        """Row counts per table, for a health endpoint."""
        ...

    def close(self) -> None:
        """Release what this store owns. Idempotent, and never raises."""
        ...


# --------------------------------------------------------------------------
# Implementation
# --------------------------------------------------------------------------


class _TableStoreBase:
    """The engine-neutral half: SQL construction, the lock, and every method.

    Both tiers run this code. The only thing that differs is
    :attr:`dialect` — placeholder syntax, type names, the default literal, how a
    row is fetched, and whether an insert takes an advisory lock — which is
    exactly the list of places where the two engines genuinely differ, so a
    behavioural difference between the tiers has to be written down as a
    dialect difference rather than discovered as a bug report from the Hosted
    deployment.
    """

    def __init__(
        self, database: Any, dialect: _Dialect, *, owned_bundle: StoreBundle | None = None
    ):
        self.database = database
        self.dialect = dialect
        self._owned_bundle = owned_bundle
        self._closed = False

    # -- capabilities ------------------------------------------------------

    @property
    def tier(self) -> str:
        return self.dialect.tier

    @property
    def backend(self) -> str:
        return self.dialect.backend

    def close(self) -> None:
        """Close the bundle this store built, if it built one. Never raises.

        A store handed an existing bundle closes nothing: the caller that opened
        the database decides when it goes away, and a store that closed it would
        take down the other stores sharing the same file.
        """
        if self._closed:
            return
        self._closed = True
        if self._owned_bundle is not None:
            try:
                self._owned_bundle.close()
            except Exception:  # pragma: no cover - close must not raise
                logger.warning("关闭表存储失败。", exc_info=True)

    # -- transactions ------------------------------------------------------

    @contextmanager
    def transaction(self) -> Iterator[Any]:
        """A cursor inside one write transaction, always committed or rolled back.

        Reentrant, and that is a requirement rather than a convenience: a
        transaction that the caller opened must be joined by the methods it
        calls, or :meth:`add_document_version`'s "clear ``is_current`` then
        insert" would commit in two pieces and a reader could see a document
        with no current version. The depth counter is per thread, because the
        two tiers' connections are per thread and a shared counter would let one
        request's transaction swallow another's rollback.
        """
        state = self._thread_state()
        if state["depth"]:
            cursor = self._connection().cursor()
            try:
                yield cursor
            finally:
                cursor.close()
            return
        connection = self._connection()
        sqlite = self.dialect.backend == "sqlite"
        if sqlite:
            # IMMEDIATE, not DEFERRED: the whole point of the CAS is to take the
            # write lock before reading the version it compares against, so two
            # writers cannot both read version 3 and then both write. A DEFERRED
            # transaction takes the lock at the first write instead, which leaves
            # exactly the race the lock exists to close.
            connection.execute("BEGIN IMMEDIATE")
        cursor = connection.cursor()
        state["depth"] = 1
        try:
            yield cursor
        except BaseException:
            state["depth"] = 0
            if sqlite:
                connection.execute("ROLLBACK")
            else:
                connection.rollback()
            raise
        else:
            state["depth"] = 0
            if sqlite:
                connection.execute("COMMIT")
            else:
                connection.commit()
        finally:
            cursor.close()

    @contextmanager
    def read(self) -> Iterator[Any]:
        """A cursor for a read, with no explicit write transaction.

        Reads deliberately do not go through :meth:`transaction`. On SQLite a
        write transaction is an exclusive lock on the file, so a
        ``get_document`` would block every other reader in the process for the
        duration of a read that changes nothing. On PostgreSQL a transaction
        that is opened and never committed holds a snapshot, and a connection
        that accumulates snapshots is a connection nobody can vacuum. So: SQLite
        reads run in autocommit and PostgreSQL reads use the database's own
        cursor context, which commits them immediately.
        """
        if self.dialect.backend == "sqlite":
            cursor = self._connection().cursor()
            try:
                yield cursor
            finally:
                cursor.close()
            return
        with self.database.cursor() as cursor:
            yield cursor

    def _thread_state(self) -> dict[str, int]:
        if not hasattr(self, "_local"):
            self._local = threading.local()
        state = getattr(self._local, "state", None)
        if state is None:
            state = {"depth": 0}
            self._local.state = state
        return state

    def _connection(self) -> Any:
        return self.database.connection()

    def _lock_key(self, cursor: Any, value: str) -> None:
        """Serialize two writers inserting the same key. No-op on SQLite.

        ``INSERT`` on a primary key that already exists is safe from corruption
        but not from *interpretation*: without this, the second writer's
        existence check happens before the first writer's insert commits, both
        see no row, and the loser gets a driver integrity error instead of
        :class:`DuplicateRow`. SQLite gets the same guarantee from its
        ``BEGIN IMMEDIATE``, which is why this is a no-op there rather than a
        second mechanism.
        """
        if not self.dialect.advisory_locks:
            return
        cursor.execute("SELECT pg_advisory_xact_lock(hashtext(%s)::bigint)", (str(value),))

    # -- schema ------------------------------------------------------------

    def initialize(self) -> None:
        """Create or migrate every table. Idempotent, and additive.

        ``CREATE TABLE IF NOT EXISTS`` for the table, then an ``ALTER TABLE ADD
        COLUMN`` for each declared column the live table lacks. The second step
        is what makes this a migration and not just a bootstrap: a database
        created by an earlier build of this project gains the new columns and
        keeps every row. No ledger, no "has this run" flag, no down-migration —
        a schema that only moves forward is a schema that can be applied from
        any version to any later one, which is the property that makes "run it
        twice to be sure" a safe thing to do.
        """
        for table in TABLES:
            self._ensure_table(table)
        for table in TABLES:
            self._ensure_indexes(table)

    def _ensure_table(self, table: Table) -> None:
        with self.transaction() as cursor:
            cursor.execute(self.dialect.ddl(table)[0])
            live = self.dialect.existing_columns(cursor, table)
            for column in table.columns:
                if column.name in live or column.name == table.primary_key:
                    # A primary key cannot be added by ALTER TABLE on either
                    # engine. Skipping it is safe because every table is created
                    # with it, so a live table that lacks it is not a table this
                    # module made.
                    continue
                if column.nullable and column.default is None:
                    raise TableError(
                        f"无法为 {table.name} 添加可空且无默认值的列 {column.name}；"
                        "请给它一个默认值。"
                    )
                rendered = self.dialect.render_column(column)
                if self.dialect.backend == "sqlite":
                    statement = (
                        f"ALTER TABLE {self.dialect.qualify(table.name)} "
                        f"ADD COLUMN {rendered}"
                    )
                else:
                    statement = (
                        f"ALTER TABLE {self.dialect.qualify(table.name)} "
                        f"ADD COLUMN IF NOT EXISTS {rendered}"
                    )
                logger.info("为已有表添加列：%s.%s", table.name, column.name)
                cursor.execute(statement)

    def _ensure_indexes(self, table: Table) -> None:
        with self.transaction() as cursor:
            for index in table.indexes:
                unique = "UNIQUE " if index.unique else ""
                cursor.execute(
                    f"CREATE {unique}INDEX IF NOT EXISTS "
                    f"{self.dialect.qualify(index.sql_name)} "
                    f"ON {self.dialect.qualify(table.name)} ({', '.join(index.columns)})"
                )

    # -- generic row operations -------------------------------------------

    def _table(self, name: str) -> Table:
        table = TABLE_BY_NAME.get(str(name))
        if table is None:
            raise TableError(
                f"未知的表：{name!r}（可用值：{', '.join(TABLE_BY_NAME)}）"
            )
        return table

    def _resolve_values(
        self,
        table: Table,
        values: Mapping[str, Any],
        *,
        require: Sequence[str] = (),
        fill: bool = True,
    ) -> dict[str, Any]:
        """Validate a write's columns, and fill defaults for the ones it omits.

        An unknown key raises rather than being dropped. The tempting
        alternative — ignore keys the table does not have — is how a renamed
        column survives as a default value forever, with the writer believing it
        was stored.

        ``fill`` is off for updates, and it must be. Filling on an update would
        write every column the caller did not mention *back to its default*, so
        patching a conversation's title would erase its payload, its token hash
        and its migration bookkeeping in the same statement — and it would do so
        while reporting success.
        """
        resolved: dict[str, Any] = {}
        for key, value in values.items():
            column = table.column(str(key))
            if column is None:
                raise UnknownColumn(
                    f"{table.name} 没有列 {key!r}"
                    f"（可用列：{', '.join(table.column_names)}）"
                )
            resolved[column.name] = value
        for name in require:
            if name not in resolved:
                raise MissingColumn(
                    f"写入 {table.name} 缺少必需列 {name!r}"
                    "（该列没有默认值，缺失说明调用方漏传了）"
                )
        if not fill:
            return resolved
        for column in table.columns:
            if column.name not in resolved:
                if column.default is None and not column.nullable:
                    raise MissingColumn(f"写入 {table.name} 缺少必需列 {column.name!r}")
                resolved[column.name] = column.default
        return resolved

    def _insert(self, table_name: str, values: Mapping[str, Any]) -> Row:
        """Insert one row, refusing a duplicate key, and return the stored row."""
        table = self._table(table_name)
        resolved = self._resolve_values(table, values, require=(table.primary_key,))
        key_value = resolved[table.primary_key]
        with self.transaction() as cursor:
            self._lock_key(cursor, key_value)
            cursor.execute(
                f"SELECT 1 FROM {self.dialect.qualify(table.name)} "
                f"WHERE {table.primary_key} = {self.dialect.placeholder}",
                (self.dialect.scalar(key_value),),
            )
            if cursor.fetchone() is not None:
                raise DuplicateRow(
                    f"{table.name} 中已存在 {table.primary_key}={key_value!r} 的行。"
                )
            self._insert_within(cursor, table.name, resolved)
        stored = self._get(table_name, table.primary_key, key_value)
        if stored is None:  # pragma: no cover - the row was just inserted
            raise TableError(f"{table.name} 插入后读不回来：{key_value!r}")
        return stored

    def _get(self, table_name: str, key_column: str, key_value: Any) -> Row | None:
        table = self._table(table_name)
        columns = ", ".join(table.column_names)
        with self.read() as cursor:
            cursor.execute(
                f"SELECT {columns} FROM {self.dialect.qualify(table.name)} "
                f"WHERE {key_column} = {self.dialect.placeholder}",
                (self.dialect.scalar(key_value),),
            )
            row = cursor.fetchone()
        return self._row(table, row)

    def _find(
        self,
        table_name: str,
        where: str,
        params: Sequence[Any],
        order: str,
        limit: int = 0,
        offset: int = 0,
    ) -> list[Row]:
        table = self._table(table_name)
        columns = ", ".join(table.column_names)
        statement = f"SELECT {columns} FROM {self.dialect.qualify(table.name)}"
        if where:
            statement += f" WHERE {where}"
        if order:
            statement += f" ORDER BY {order}"
        if limit or offset:
            # Both, in SQL, and not as a Python slice: ``LIMIT`` without
            # ``OFFSET`` returns the first window of rows, so a paged caller
            # asking for page 2 with limit 50 would be handed page 1 again. The
            # two arguments are interpolated because they are ints this method
            # just coerced, not strings from a caller.
            statement += f" LIMIT {max(0, int(limit))} OFFSET {max(0, int(offset))}"
        with self.read() as cursor:
            cursor.execute(statement, tuple(self.dialect.scalar(item) for item in params))
            rows = cursor.fetchall()
        return [row for row in (self._row(table, item) for item in rows) if row is not None]

    def _row(self, table: Table, raw: Any) -> Row | None:
        if raw is None:
            return None
        values = list(raw)
        decoded: Row = {}
        for column, value in zip(table.columns, values, strict=True):
            decoded[column.name] = self.dialect.decode(column, value)
        return decoded

    def _update(
        self,
        table_name: str,
        key_column: str,
        key_value: Any,
        changes: Mapping[str, Any],
        *,
        expected_version: int,
    ) -> Row:
        """The compare-and-set. One statement, no read-modify-write in Python.

        ``version = version + 1`` is in the same ``UPDATE`` as the data, so a
        reader can never see new data at an old version, and the ``WHERE`` on
        ``version`` is what makes the second of two writers fail. The version is
        then read back and returned, because the row a caller holds after a
        successful write must be the row the database has.
        """
        table = self._table(table_name)
        if "version" not in table.column_names:
            raise TableError(f"{table.name} 没有乐观锁列 version。")
        # fill=False: an update writes the columns it names and nothing else.
        resolved = self._resolve_values(table, changes, fill=False)
        for guarded in ("version", table.primary_key):
            if guarded in resolved:
                # ``version`` is the lock and the key is the row's identity;
                # neither is something a caller may set. Accepting them quietly
                # would be a way to defeat the compare-and-set with an argument
                # that looks like a field update.
                raise UnknownColumn(
                    f"{table.name} 的 {guarded!r} 不能由调用方写入"
                    "（version 由乐观锁自增，主键由 store 分配）。"
                )
        if "updated_at" in table.column_names and "updated_at" not in resolved:
            resolved["updated_at"] = _now_iso()
        assignments = [f"{name} = {self.dialect.placeholder}" for name in resolved]
        assignments.append("version = version + 1")
        mark = self.dialect.placeholder
        where = f"{key_column} = {mark} AND version = {mark}"
        with self.transaction() as cursor:
            if self.dialect.advisory_locks:
                self._lock_key(cursor, key_value)
            cursor.execute(
                f"UPDATE {self.dialect.qualify(table.name)} SET {', '.join(assignments)} "
                f"WHERE {where}",
                (
                    *(
                        self.dialect.encode(table.column(name) or Column(name, Kind.TEXT), value)
                        for name, value in resolved.items()
                    ),
                    self.dialect.scalar(key_value),
                    int(expected_version),
                ),
            )
            affected = cursor.rowcount
            if affected == 0:
                self._raise_write_failure(cursor, table, key_column, key_value, expected_version)
        stored = self._get(table_name, key_column, key_value)
        if stored is None:  # pragma: no cover - the row exists, we just updated it
            raise TableError(f"{table.name} 更新后读不回来：{key_value!r}")
        return stored

    def _raise_write_failure(
        self, cursor: Any, table: Table, key_column: str, key_value: Any, expected_version: int
    ) -> None:
        """Explain a failed compare-and-set, and never guess.

        Two causes, and they mean opposite things to the caller: the row is gone
        (somebody deleted it between the read and the write) or the row moved on
        (somebody else wrote it). Reporting a version conflict for a deleted
        row would send a caller into a retry loop against a row that will never
        come back, so the two are distinguished here rather than collapsed.
        """
        cursor.execute(
            f"SELECT version FROM {self.dialect.qualify(table.name)} "
            f"WHERE {key_column} = {self.dialect.placeholder}",
            (self.dialect.scalar(key_value),),
        )
        row = cursor.fetchone()
        if row is None:
            raise UnknownRow(f"{table.name} 中不存在 {key_column}={key_value!r} 的行，无法更新。")
        raise VersionConflict(table.name, str(key_value), expected_version, int(row[0]))

    def _delete_where(self, table_name: str, where: str, params: Sequence[Any]) -> int:
        table = self._table(table_name)
        with self.transaction() as cursor:
            cursor.execute(
                f"DELETE FROM {self.dialect.qualify(table.name)} WHERE {where}",
                tuple(self.dialect.scalar(item) for item in params),
            )
            return int(cursor.rowcount or 0)

    def _count(self, table_name: str, where: str = "", params: Sequence[Any] = ()) -> int:
        table = self._table(table_name)
        statement = f"SELECT COUNT(*) FROM {self.dialect.qualify(table.name)}"
        if where:
            statement += f" WHERE {where}"
        with self.read() as cursor:
            cursor.execute(statement, tuple(self.dialect.scalar(item) for item in params))
            row = cursor.fetchone()
        return int(row[0]) if row else 0

    # -- document versions -------------------------------------------------

    def add_document_version(self, values: Mapping[str, Any]) -> Row:
        """Append a version and make it current, atomically.

        ``version_no`` is the highest existing plus one and the previous
        ``is_current`` rows are cleared, both inside this transaction. A
        document therefore never has two current versions, and never has none
        — the two states a reader cannot interpret.
        """
        table = self._table("document_versions")
        document_id = str(values.get("document_id") or "").strip()
        if not document_id:
            raise MissingColumn("写入 document_versions 缺少必需列 'document_id'。")
        is_current = table.column("is_current") or BOOL("is_current")
        placeholder = self.dialect.placeholder
        with self.transaction() as cursor:
            if self.dialect.advisory_locks:
                self._lock_key(cursor, document_id)
            cursor.execute(
                f"SELECT COALESCE(MAX(version_no), 0) FROM {self.dialect.qualify(table.name)} "
                f"WHERE document_id = {placeholder}",
                (self.dialect.scalar(document_id),),
            )
            row = cursor.fetchone()
            next_no = int(row[0]) + 1 if row else 1
            cursor.execute(
                f"UPDATE {self.dialect.qualify(table.name)} "
                f"SET is_current = {placeholder}, updated_at = {placeholder} "
                f"WHERE document_id = {placeholder} AND is_current = {placeholder}",
                (
                    self.dialect.encode(is_current, False),
                    _now_iso(),
                    self.dialect.scalar(document_id),
                    self.dialect.encode(is_current, True),
                ),
            )
        payload = dict(values)
        payload["document_id"] = document_id
        payload.setdefault("version_id", new_id("ver"))
        payload.setdefault("created_at", _now_iso())
        payload.setdefault("updated_at", payload.get("created_at") or _now_iso())
        if payload.get("version_no") in (None, 0, ""):
            payload["version_no"] = next_no
        payload["is_current"] = True
        return self._insert("document_versions", payload)

    def get_document_version(self, version_id: str) -> Row | None:
        return self._get("document_versions", "version_id", str(version_id))

    def get_current_document_version(self, document_id: str) -> Row | None:
        found = self._find(
            "document_versions",
            "document_id = ? AND is_current = ?",
            (str(document_id), True),
            "version_no DESC, version_id ASC",
            limit=1,
        )
        return found[0] if found else None

    def list_document_versions(self, document_id: str) -> list[Row]:
        return self._find(
            "document_versions",
            "document_id = ?",
            (str(document_id),),
            "version_no ASC, version_id ASC",
        )

    def update_document_version(
        self, version_id: str, changes: Mapping[str, Any], *, expected_version: int
    ) -> Row:
        return self._update(
            "document_versions",
            "version_id",
            str(version_id),
            changes,
            expected_version=expected_version,
        )

    # -- conversations -----------------------------------------------------

    def add_conversation(self, values: Mapping[str, Any]) -> Row:
        payload = dict(values)
        payload.setdefault("conversation_id", new_id("conv"))
        payload.setdefault("created_at", _now_iso())
        payload.setdefault("updated_at", payload.get("created_at") or _now_iso())
        return self._insert("conversations", payload)

    def get_conversation(self, conversation_id: str) -> Row | None:
        return self._get("conversations", "conversation_id", str(conversation_id))

    def get_conversation_by_session(self, session_id: str) -> Row | None:
        found = self._find(
            "conversations", "session_id = ?", (str(session_id),), "created_at ASC", limit=1
        )
        return found[0] if found else None

    def list_conversations(self, limit: int = 50, offset: int = 0) -> list[Row]:
        return self._find(
            "conversations",
            "",
            (),
            "created_at DESC, conversation_id ASC",
            limit=int(limit),
            offset=int(offset),
        )

    def update_conversation(
        self, conversation_id: str, changes: Mapping[str, Any], *, expected_version: int
    ) -> Row:
        return self._update(
            "conversations",
            "conversation_id",
            str(conversation_id),
            changes,
            expected_version=expected_version,
        )

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation, its messages and its queue items, in one transaction.

        The cascade is here rather than in three route handlers because half of
        it is worse than none of it: a conversation row removed while its
        messages stay leaves a message list that no conversation claims and no
        query can exclude.
        """
        identifier = str(conversation_id)
        placeholder = self.dialect.placeholder
        bound = (self.dialect.scalar(identifier),)
        with self.transaction() as cursor:
            cursor.execute(
                f"DELETE FROM {self.dialect.qualify('reading_queue')} "
                f"WHERE conversation_id = {placeholder}",
                bound,
            )
            queue_removed = int(cursor.rowcount or 0)
            cursor.execute(
                f"DELETE FROM {self.dialect.qualify('messages')} "
                f"WHERE conversation_id = {placeholder}",
                bound,
            )
            messages_removed = int(cursor.rowcount or 0)
            cursor.execute(
                f"DELETE FROM {self.dialect.qualify('conversations')} "
                f"WHERE conversation_id = {placeholder}",
                bound,
            )
            removed = int(cursor.rowcount or 0)
        if not removed:
            return False
        logger.info(
            "已删除会话 %s（消息 %s 条，队列 %s 项）。", identifier, messages_removed, queue_removed
        )
        return True

    # -- messages ----------------------------------------------------------

    def append_message(self, values: Mapping[str, Any]) -> Row:
        """Append one message, numbering it within its conversation.

        The number is assigned here because it is a property of the position in
        the conversation, not of the caller's request. Two concurrent Q&A turns
        that both believe they are message 7 would produce a list where the
        order of the answers is decided by rowid, and the transcript would be
        wrong in a way no reader could detect.
        """
        conversation_id = str(values.get("conversation_id") or "").strip()
        if not conversation_id:
            raise MissingColumn("写入 messages 缺少必需列 'conversation_id'。")
        payload = dict(values)
        payload.setdefault("message_id", new_id("msg"))
        payload.setdefault("created_at", _now_iso())
        payload.setdefault("updated_at", payload.get("created_at") or _now_iso())
        with self.transaction() as cursor:
            if self.dialect.advisory_locks:
                self._lock_key(cursor, conversation_id)
            cursor.execute(
                f"SELECT COALESCE(MAX(sequence), 0) FROM {self.dialect.qualify('messages')} "
                f"WHERE conversation_id = {self.dialect.placeholder}",
                (self.dialect.scalar(conversation_id),),
            )
            row = cursor.fetchone()
            if not payload.get("sequence"):
                payload["sequence"] = int(row[0]) + 1 if row else 1
        return self._insert("messages", payload)

    def get_message(self, message_id: str) -> Row | None:
        return self._get("messages", "message_id", str(message_id))

    def list_messages(self, conversation_id: str) -> list[Row]:
        return self._find(
            "messages",
            "conversation_id = ?",
            (str(conversation_id),),
            "sequence ASC, message_id ASC",
        )

    def update_message(
        self, message_id: str, changes: Mapping[str, Any], *, expected_version: int
    ) -> Row:
        return self._update(
            "messages", "message_id", str(message_id), changes, expected_version=expected_version
        )

    def replace_messages(self, conversation_id: str, items: Sequence[Mapping[str, Any]]) -> int:
        """Rebuild a conversation's transcript in one transaction.

        Numbering is recomputed from the incoming order rather than kept, because
        the source of truth for the order is the list the caller passed and a
        sequence number that disagreed with it would be a second, wrong order.
        """
        identifier = str(conversation_id)
        payload: list[dict[str, Any]] = []
        for position, item in enumerate(items):
            row = dict(item)
            row["conversation_id"] = identifier
            row.setdefault("message_id", new_id("msg"))
            row["sequence"] = position + 1
            row.setdefault("created_at", _now_iso())
            row.setdefault("updated_at", row.get("created_at") or _now_iso())
            payload.append(row)
        with self.transaction() as cursor:
            if self.dialect.advisory_locks:
                self._lock_key(cursor, identifier)
            cursor.execute(
                f"DELETE FROM {self.dialect.qualify('messages')} "
                f"WHERE conversation_id = {self.dialect.placeholder}",
                (self.dialect.scalar(identifier),),
            )
            for row in payload:
                self._insert_within(cursor, "messages", row)
        return len(payload)

    # -- reading queue -----------------------------------------------------

    def add_queue_item(self, values: Mapping[str, Any]) -> Row:
        conversation_id = str(values.get("conversation_id") or "").strip()
        if not conversation_id:
            raise MissingColumn("写入 reading_queue 缺少必需列 'conversation_id'。")
        payload = dict(values)
        payload.setdefault("queue_item_id", new_id("queue"))
        payload.setdefault("position", 0)
        payload.setdefault("added_at", _now_iso())
        payload.setdefault("updated_at", payload.get("added_at") or _now_iso())
        return self._insert("reading_queue", payload)

    def get_queue_item(self, queue_item_id: str) -> Row | None:
        return self._get("reading_queue", "queue_item_id", str(queue_item_id))

    def list_queue_items(self, conversation_id: str) -> list[Row]:
        return self._find(
            "reading_queue",
            "conversation_id = ?",
            (str(conversation_id),),
            "position ASC, queue_item_id ASC",
        )

    def update_queue_item(
        self, queue_item_id: str, changes: Mapping[str, Any], *, expected_version: int
    ) -> Row:
        return self._update(
            "reading_queue",
            "queue_item_id",
            str(queue_item_id),
            changes,
            expected_version=expected_version,
        )

    def remove_queue_item(self, queue_item_id: str) -> bool:
        return self._delete_where("reading_queue", "queue_item_id = ?", (str(queue_item_id),)) > 0

    def replace_reading_queue(
        self, conversation_id: str, items: Sequence[Mapping[str, Any]]
    ) -> int:
        """Replace the whole queue in one transaction, returning how many rows.

        A delete followed by inserts, committed together. The two halves cannot
        be observed separately, so a reader never sees an empty queue for a
        conversation that has one, and a failure halfway through leaves the old
        queue intact rather than a truncated one.
        """
        identifier = str(conversation_id)
        payload = []
        for position, item in enumerate(items):
            row = dict(item)
            row["conversation_id"] = identifier
            row.setdefault("queue_item_id", new_id("queue"))
            row["position"] = int(row.get("position", position))
            row.setdefault("added_at", _now_iso())
            row.setdefault("updated_at", _now_iso())
            payload.append(row)
        with self.transaction() as cursor:
            if self.dialect.advisory_locks:
                self._lock_key(cursor, identifier)
            cursor.execute(
                f"DELETE FROM {self.dialect.qualify('reading_queue')} "
                f"WHERE conversation_id = {self.dialect.placeholder}",
                (self.dialect.scalar(identifier),),
            )
            for row in payload:
                self._insert_within(cursor, "reading_queue", row)
        return len(payload)

    def _insert_within(self, cursor: Any, table_name: str, values: Mapping[str, Any]) -> Row:
        """The insert half of :meth:`_insert`, on a cursor the caller owns.

        Split out so :meth:`replace_reading_queue` can do its delete and its
        inserts in one transaction. Re-entering :meth:`_insert` would open a
        nested transaction, and on PostgreSQL that is a savepoint the outer
        statement does not control — the "atomic" replace would be two commits.
        """
        table = self._table(table_name)
        resolved = self._resolve_values(table, values, require=(table.primary_key,))
        columns = [column.name for column in table.columns]
        placeholders = ", ".join(self.dialect.placeholder for _ in columns)
        try:
            cursor.execute(
                f"INSERT INTO {self.dialect.qualify(table.name)} "
                f"({', '.join(columns)}) VALUES ({placeholders})",
                tuple(
                    self.dialect.encode(
                        table.column(name) or Column(name, Kind.TEXT), resolved[name]
                    )
                    for name in columns
                ),
            )
        except Exception as exc:
            if _is_constraint_failure(exc):
                raise ConstraintViolation(table.name, str(exc)) from exc
            raise

    # -- analysis runs -----------------------------------------------------

    def record_analysis_run(
        self, columns: Mapping[str, Any], *, embedding: Mapping[str, Any] | None = None
    ) -> Row:
        """Persist one T1.8 run.

        ``columns`` is what
        :meth:`~paperwhisperer.analysis.paper_facts.AnalysisRun.as_analysis_run_columns`
        returned, verbatim — this method does not rename a key, so the test that
        compares the two is comparing the same strings. Every key in
        :data:`REQUIRED_ANALYSIS_RUN_COLUMNS` must be there, and a key the table
        does not have raises :class:`UnknownColumn`.

        Two costs are kept apart and both are nullable: a run that spent nothing
        because everything was cached (``0.0``) is a different fact from one that
        cannot report its cost (``None``), and rounding the second to the first
        is how a cost dashboard starts lying.
        """
        merged: dict[str, Any] = dict(columns)
        if embedding:
            merged.update(embedding)
        table = self._table("analysis_runs")
        available = set(table.column_names)
        missing = [
            name for name in REQUIRED_ANALYSIS_RUN_COLUMNS if name not in merged
        ]
        if missing:
            raise MissingColumn(
                f"写入 analysis_runs 缺少 T1.8 要求的列：{', '.join(missing)}。"
                "AnalysisRun.as_analysis_run_columns() 与 "
                "EmbeddingUsage.as_analysis_run_columns() 命名的每一列都必须落库。"
            )
        for name in merged:
            if name not in available:
                raise UnknownColumn(
                    f"analysis_runs 没有列 {name!r}（可用列：{', '.join(table.column_names)}）"
                )
        # updated_at mirrors created_at on insert rather than staying at the
        # column default: an empty "last modified" on every run is a row that
        # cannot be sorted by it, and the only writer of this column is
        # _update, which sets it on every patch. A run nobody has amended is by
        # definition as old as it is new.
        merged.setdefault("updated_at", str(merged.get("created_at") or "") or _now_iso())
        return self._insert("analysis_runs", merged)

    def get_analysis_run(self, run_id: str) -> Row | None:
        return self._get("analysis_runs", "run_id", str(run_id))

    def list_analysis_runs(self, document_id: str, limit: int = 50) -> list[Row]:
        return self._find(
            "analysis_runs",
            "document_id = ?",
            (str(document_id),),
            "created_at DESC, run_id ASC",
            limit=int(limit),
        )

    def latest_analysis_run(self, document_id: str) -> Row | None:
        found = self.list_analysis_runs(document_id, limit=1)
        return found[0] if found else None

    def cost_summary(self, document_id: str) -> Row:
        """Totals across a document's runs, unknowns included as unknowns.

        ``llm_cost`` is ``None`` when any run could not report its cost, and the
        legacy columns are summed the same way, so the T1.8 comparison — this
        path against the old one — is answerable from stored rows rather than
        from a measurement somebody took once and put in a commit message.
        """
        runs = self.list_analysis_runs(document_id, limit=0)
        summary: Row = {
            "document_id": str(document_id),
            "runs": len(runs),
            "llm_calls": 0,
            "llm_input_tokens": 0,
            "llm_output_tokens": 0,
            "legacy_llm_calls": 0,
            "legacy_input_tokens": 0,
            "llm_cost": 0.0,
            "legacy_cost_known": True,
        }
        cost_unknown = False
        for run in runs:
            for key in (
                "llm_calls",
                "llm_input_tokens",
                "llm_output_tokens",
                "legacy_llm_calls",
                "legacy_input_tokens",
            ):
                summary[key] += int(run.get(key) or 0)
            if run.get("llm_cost") is None:
                cost_unknown = True
            else:
                summary["llm_cost"] += float(run.get("llm_cost") or 0.0)
        if cost_unknown:
            summary["llm_cost"] = None
        return summary

    # -- cross-table -------------------------------------------------------

    def object_keys_for_document(self, document_id: str) -> list[Row]:
        """Every object key this document's rows reference, with its layer.

        A report, not a delete. The tables do not own the bytes, and a store
        that removed them from here would be making a reachability decision from
        one of the two places that know the answer — which is how an upload
        outlives its row, or a row outlives its upload, and the next parse finds
        neither.
        """
        keys: list[Row] = []
        seen: set[tuple[str, str]] = set()
        versions = self._find(
            "document_versions",
            "document_id = ?",
            (str(document_id),),
            "version_no ASC, version_id ASC",
        )
        for version in versions:
            for column, layer in (
                ("original_key", "original"),
                ("parsed_key", "parsed"),
                ("canonical_key", "canonical"),
            ):
                value = version.get(column)
                if not value or (layer, str(value)) in seen:
                    # De-duplicated because the original key is the same on every
                    # version of a document by design: reporting it once per
                    # version would make a caller that deletes the listed keys
                    # either fail on the second delete or, worse, count the same
                    # upload as N files in a storage report.
                    continue
                seen.add((layer, str(value)))
                keys.append(
                    {
                        "layer": layer,
                        "key": value,
                        "source_table": "document_versions",
                        "source_id": version["version_id"],
                        "version_no": version.get("version_no"),
                    }
                )
        return keys

    def delete_document(self, document_id: str) -> Row:
        """Delete every row belonging to a document. Returns what went by table.

        The object keys are *reported* rather than removed — see
        :meth:`object_keys_for_document` — and the caller is expected to delete
        the bytes itself, in the layer-aware way
        :meth:`~paperwhisperer.infra.object_store.ObjectStore.delete` requires.
        """
        identifier = str(document_id)
        report: Row = {
            "document_id": identifier,
            "document_versions": 0,
            "analysis_runs": 0,
            "conversations": 0,
            "messages": 0,
            "reading_queue": 0,
            "object_keys": self.object_keys_for_document(identifier),
        }
        with self.transaction() as cursor:
            placeholder = self.dialect.placeholder
            bound = (self.dialect.scalar(identifier),)
            conversation_ids: list[str] = []
            cursor.execute(
                f"SELECT conversation_id FROM {self.dialect.qualify('conversations')} "
                f"WHERE document_id = {placeholder}",
                bound,
            )
            for row in cursor.fetchall():
                conversation_ids.append(str(row[0]))
            for orphan in conversation_ids:
                cursor.execute(
                    f"DELETE FROM {self.dialect.qualify('reading_queue')} "
                    f"WHERE conversation_id = {placeholder}",
                    (self.dialect.scalar(orphan),),
                )
                report["reading_queue"] += int(cursor.rowcount or 0)
                cursor.execute(
                    f"DELETE FROM {self.dialect.qualify('messages')} "
                    f"WHERE conversation_id = {placeholder}",
                    (self.dialect.scalar(orphan),),
                )
                report["messages"] += int(cursor.rowcount or 0)
            cursor.execute(
                f"DELETE FROM {self.dialect.qualify('conversations')} "
                f"WHERE document_id = {placeholder}",
                bound,
            )
            report["conversations"] = int(cursor.rowcount or 0)
            cursor.execute(
                f"DELETE FROM {self.dialect.qualify('analysis_runs')} "
                f"WHERE document_id = {placeholder}",
                bound,
            )
            report["analysis_runs"] = int(cursor.rowcount or 0)
            cursor.execute(
                f"DELETE FROM {self.dialect.qualify('document_versions')} "
                f"WHERE document_id = {placeholder}",
                bound,
            )
            report["document_versions"] = int(cursor.rowcount or 0)
        return report

    def describe(self) -> Row:
        """Row counts per table, for a health endpoint.

        Counts and not table lists: a health check that enumerates rows to prove
        it is alive is a health check that goes slow exactly when there is
        something wrong.
        """
        return {
            "schema_version": SCHEMA_VERSION,
            "tier": self.tier,
            "backend": self.backend,
            "tables": {table.name: self._count(table.name) for table in TABLES},
        }

    # -- test and diagnostic helpers ---------------------------------------

    def table_names(self) -> tuple[str, ...]:
        return tuple(table.name for table in TABLES)

    def live_columns(self, table_name: str) -> set[str]:
        """The columns the database actually has for ``table_name``.

        Not on the protocol: it is how a test compares the DDL that was
        *executed* against :data:`TABLES` and against what the analysis module
        demands, rather than comparing two copies of the same declaration in
        Python and calling that verification.
        """
        table = self._table(table_name)
        with self.transaction() as cursor:
            return self.dialect.existing_columns(cursor, table)


class SqliteTableStore(_TableStoreBase):
    """The personal tier's tables, in the same SQLite file as the stores."""

    def __init__(
        self, database: SqliteDatabase, *, owned_bundle: StoreBundle | None = None
    ) -> None:
        super().__init__(database, _SqliteDialect(), owned_bundle=owned_bundle)


class PostgresTableStore(_TableStoreBase):
    """The Hosted tier's tables, in the same PostgreSQL schema as the stores."""

    def __init__(
        self, database: PostgresDatabase, *, owned_bundle: StoreBundle | None = None
    ) -> None:
        super().__init__(database, _PostgresDialect(database), owned_bundle=owned_bundle)


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


#: Every key this module reads, with the default used when it is absent.
#:
#: It is
#: :data:`~paperwhisperer.infra.stores.factory.REQUIRED_CONFIG_KEYS` — the very
#: same dict, not a copy of it. This module reads no key the store factory does
#: not, because it opens the store factory's database: a second table of keys
#: and defaults would be a second contract with ``core/config.py``, and the two
#: would drift the first time a default changed. It is re-exported under its own
#: name so a reader of *this* module can see the whole of what it depends on
#: without following the import.
#:
#: There is deliberately nothing here about *which* tables to create. The set is
#: fixed by the plan, and a configuration knob that drops a table is a knob that
#: produces a database nobody can read back.
TABLE_STORE_CONFIG_KEYS: dict[str, Any] = dict(REQUIRED_CONFIG_KEYS)


def build_table_store(
    settings: Any = None,
    *,
    bundle: StoreBundle | None = None,
) -> TableStore:
    """Build the table store for the configured tier.

    A ``bundle`` is the normal argument and mirrors
    :func:`~paperwhisperer.infra.jobs.build_job_store` exactly: the tables go in
    the same SQLite file, or the same PostgreSQL schema, as the documents, which
    is what "back up one data directory" needs and what makes a document row and
    its analysis run writable in one transaction. Opening a second connection to
    the same file would be a second lock, and a second place for a write to
    fail.

    Without one, a bundle is built and owned by this function — the store's
    ``close()`` then closes it — which is what a script or a test that only wants
    these tables needs.
    """
    if bundle is None:
        bundle = build_stores(settings)
        owned: StoreBundle | None = bundle
    else:
        owned = None
    backend = resolve_backend(settings) if settings is not None else bundle.backend
    if backend == BACKEND_POSTGRES or isinstance(bundle.database, PostgresDatabase):
        store: TableStore = PostgresTableStore(bundle.database, owned_bundle=owned)
    else:
        store = SqliteTableStore(bundle.database, owned_bundle=owned)
    store.initialize()
    logger.info("资产表已就绪：%s", store.describe())
    return store


__all__ = [
    "BACKEND_POSTGRES",
    "EXTRA_ANALYSIS_RUN_COLUMNS",
    "EXTERNAL_TABLES",
    "REQUIRED_ANALYSIS_RUN_COLUMNS",
    "SCHEMA_VERSION",
    "TABLE_BY_NAME",
    "TABLE_STORE_CONFIG_KEYS",
    "TABLES",
    "TIER_HOSTED",
    "TIER_PERSONAL",
    "Column",
    "ConstraintViolation",
    "DuplicateRow",
    "Index",
    "Kind",
    "MissingColumn",
    "PostgresTableStore",
    "SqliteTableStore",
    "Table",
    "TableError",
    "TableStore",
    "UnknownColumn",
    "UnknownRow",
    "VersionConflict",
    "build_table_store",
    "new_id",
]
