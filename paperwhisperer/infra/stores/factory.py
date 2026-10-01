"""Backend selection, in one place, from one decision.

``build_stores`` is the only function in the project that chooses a storage
engine. Every other caller receives a :class:`StoreBundle` and never learns
which one it got. That is not tidiness — it is the requirement. The plan
insists both tiers be reachable through one interface set, and an interface
that every call site can override is not an interface, it is a convention.

**Configuration is read with ``getattr`` and a default for every key.**
:mod:`paperwhisperer.core.config` is owned by another task, and a store
package that imported a name from it would break the moment that module
renamed something. Every key below has a working default, so the store works
on a checkout where nobody has touched the config at all — which is the state
the personal tier is required to work in. The keys to add to
``core/config.py`` are listed in :data:`REQUIRED_CONFIG_KEYS`, so the
contract between the two modules is written down rather than remembered.

**The tier is decided once, loudly, and reported.** A Hosted deployment that
falls back to SQLite by accident would look healthy while quietly writing
every user's data to a container's local disk, where it disappears with the
container. So a missing DSN under ``STORAGE_BACKEND=postgres`` is a hard
error, not a fallback.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from .interfaces import (
    BACKEND_POSTGRES,
    BACKEND_SQLITE,
    STORAGE_BACKENDS,
    TIER_HOSTED,
    TIER_PERSONAL,
    LexicalRetriever,
    MetadataStore,
    StoreCapabilities,
    VectorStore,
)
from .postgres_backend import (
    PostgresDatabase,
    PostgresLexicalStore,
    PostgresMetadataStore,
    PostgresVectorStore,
    default_sqlite_path,
    looks_like_dsn,
)
from .sqlite_backend import (
    LEXICAL_MODE_AUTO,
    LEXICAL_MODES,
    SqliteDatabase,
    SqliteLexicalStore,
    SqliteMetadataStore,
    SqliteVectorStore,
)

logger = logging.getLogger(__name__)

#: Every key this module reads, with the default it uses when the key is
#: absent. This table *is* the contract with ``core/config.py``: whoever adds
#: the keys copies these names, types and defaults, and this module keeps
#: working unchanged either way.
#:
#: ``STORAGE_BACKEND``
#:     ``sqlite`` (default, personal tier) or ``postgres`` (Hosted tier).
#: ``STORAGE_SQLITE_PATH``
#:     Path to the personal tier's database file. Relative paths resolve
#:     against the process working directory, like every other folder in this
#:     project. Default ``context/stores/paperwhisperer.sqlite3``.
#: ``STORAGE_SQLITE_TIMEOUT_SECONDS``
#:     How long a blocked write waits for the lock. Default ``30``.
#: ``STORAGE_LEXICAL_FTS5``
#:     ``auto`` (default, probe), ``on`` (require the accelerator and fail if
#:     it is missing) or ``off`` (always use the scan path).
#: ``STORAGE_PG_DSN``
#:     Hosted tier connection string. Required when the backend is
#:     ``postgres``. Falls back to ``PAPERWHISPERER_PG_DSN``, ``DATABASE_URL``
#:     or ``POSTGRES_DSN``.
#: ``STORAGE_PG_SCHEMA``
#:     Schema holding the store tables. Default ``public``.
#: ``STORAGE_PG_CONNECT_TIMEOUT_SECONDS``
#:     Driver connect timeout. Default ``5``.
#: ``STORAGE_PG_ANN``
#:     Whether to build pgvector's approximate index. Default ``false`` — see
#:     :class:`~.postgres_backend.PostgresVectorStore` for why exact is the
#:     default rather than an oversight.
#:
#: Note what is *not* here: a fuzzy-match threshold. ``pg_trgm`` is installed
#: and indexed so T1.4 can add partial recall, but a threshold in this layer
#: would be a recall policy in the wrong place — and a policy that only the
#: Hosted tier could apply is the drift T1.1 exists to prevent.
REQUIRED_CONFIG_KEYS: dict[str, Any] = {
    "STORAGE_BACKEND": BACKEND_SQLITE,
    "STORAGE_SQLITE_PATH": default_sqlite_path(),
    "STORAGE_SQLITE_TIMEOUT_SECONDS": 30,
    "STORAGE_LEXICAL_FTS5": LEXICAL_MODE_AUTO,
    "STORAGE_PG_DSN": "",
    "STORAGE_PG_SCHEMA": "public",
    "STORAGE_PG_CONNECT_TIMEOUT_SECONDS": 5,
    "STORAGE_PG_ANN": False,
}

_BOOL_TRUE = {"1", "true", "yes", "on"}


def _setting(settings: Any, key: str) -> Any:
    """Read one key from a settings object, falling back to the documented default."""
    value = getattr(settings, key, None)
    if value is None:
        return REQUIRED_CONFIG_KEYS[key]
    return value


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in _BOOL_TRUE


def resolve_backend(settings: Any = None) -> str:
    """Normalize ``STORAGE_BACKEND`` to one of the two known backends."""
    raw = str(_setting(settings, "STORAGE_BACKEND") or "").strip().lower()
    if raw in {"", BACKEND_SQLITE, "file", "local"}:
        return BACKEND_SQLITE
    if raw in {"postgres", "postgresql", "pg"}:
        return BACKEND_POSTGRES
    raise ValueError(
        f"未知的 STORAGE_BACKEND：{raw!r}（可用值：{', '.join(STORAGE_BACKENDS)}）"
    )


def resolve_dsn(settings: Any = None) -> str:
    """The Hosted tier's connection string, from config or the environment."""
    configured = str(_setting(settings, "STORAGE_PG_DSN") or "").strip()
    if configured:
        return configured
    from .postgres_backend import DEFAULT_DSN_ENV_VARS

    for name in DEFAULT_DSN_ENV_VARS:
        candidate = os.environ.get(name, "").strip()
        if candidate:
            return candidate
    return ""


@dataclass(slots=True)
class StoreBundle:
    """The three stores of one tier, plus what that tier turned out to be.

    Kept as a bundle rather than three free variables because they share a
    lifecycle: they are opened together, they are closed together, and a
    document delete has to reach all three. The two convenience methods below
    are the only place that ordering is written down, so an API route does not
    have to remember it — and cannot get it half right on one tier.
    """

    metadata: MetadataStore
    vectors: VectorStore
    lexical: LexicalRetriever
    tier: str = TIER_PERSONAL
    backend: str = BACKEND_SQLITE
    capabilities: StoreCapabilities = field(default_factory=StoreCapabilities)
    #: The object that owns the connections. Closing the bundle closes it;
    #: the stores' own ``close`` methods are no-ops by design.
    database: Any = None

    @property
    def degradations(self) -> tuple[str, ...]:
        return self.capabilities.degradations

    def initialize(self) -> None:
        for store in (self.metadata, self.vectors, self.lexical):
            store.initialize()

    def delete_document(self, document_id: str) -> bool:
        """Delete a document and every index entry that belongs to it.

        The vectors and the lexical rows go first: once the metadata row is
        gone there is nothing left to discover which vectors were orphaned,
        and an orphaned embedding keeps answering queries for a document the
        user deleted.
        """
        self.vectors.delete_document(document_id)
        self.lexical.delete_document(document_id)
        return self.metadata.delete_document(document_id)

    def close(self) -> None:
        """Close the shared connection pool. Idempotent, and never raises."""
        for store in (self.metadata, self.vectors, self.lexical):
            try:
                store.close()
            except Exception:  # pragma: no cover - close must not raise
                logger.warning("关闭存储失败。", exc_info=True)
        if self.database is not None:
            self.database.close()

    def describe(self) -> dict[str, Any]:
        """A machine-readable summary for a health endpoint or a log line.

        ``degradations`` is deliberately part of it rather than a log-only
        detail: "this deployment is slower than the other one" is a fact an
        operator needs before a user reports it.
        """
        return {
            "tier": self.tier,
            "backend": self.backend,
            **self.capabilities.describe(),
        }


def _merge_capabilities(
    tier: str,
    *,
    metadata: StoreCapabilities,
    vectors: StoreCapabilities,
    lexical: StoreCapabilities,
) -> StoreCapabilities:
    """Fold three per-store reports into the one an operator reads.

    A named argument per store, because the two that matter are specific to
    one store: a metadata store has no opinion about how vectors are indexed
    or how text is searched, so asking it and letting its defaults win would
    quietly overwrite the answer that was actually measured. Degradations are
    unioned and de-duplicated, because the same numpy warning arriving from
    two stores is one fact, not two.
    """
    degradations: list[str] = []
    for item in (metadata, vectors, lexical):
        for message in item.degradations:
            if message not in degradations:
                degradations.append(message)
    return StoreCapabilities(
        tier=tier,
        vector_engine=vectors.vector_engine,
        lexical_engine=lexical.lexical_engine,
        scan_only=lexical.scan_only,
        degradations=tuple(degradations),
    )


def build_sqlite_bundle(settings: Any = None) -> StoreBundle:
    """Build the personal tier. Opens a file; starts no service."""
    path = str(_setting(settings, "STORAGE_SQLITE_PATH") or default_sqlite_path())
    timeout = float(_setting(settings, "STORAGE_SQLITE_TIMEOUT_SECONDS") or 30)
    mode = str(_setting(settings, "STORAGE_LEXICAL_FTS5") or LEXICAL_MODE_AUTO).strip().lower()
    if mode not in LEXICAL_MODES:
        raise ValueError(
            f"未知的 STORAGE_LEXICAL_FTS5：{mode!r}（可用值：{', '.join(LEXICAL_MODES)}）"
        )
    database = SqliteDatabase(path, timeout=timeout)
    metadata = SqliteMetadataStore(database)
    vectors = SqliteVectorStore(database)
    lexical = SqliteLexicalStore(database, mode=mode)
    for store in (metadata, vectors, lexical):
        store.initialize()
    bundle = StoreBundle(
        metadata=metadata,
        vectors=vectors,
        lexical=lexical,
        tier=TIER_PERSONAL,
        backend=BACKEND_SQLITE,
        database=database,
    )
    bundle.capabilities = _merge_capabilities(
        TIER_PERSONAL,
        metadata=metadata.capabilities(),
        vectors=vectors.capabilities(),
        lexical=lexical.capabilities(),
    )
    for message in bundle.degradations:
        logger.warning(message)
    return bundle


def build_postgres_bundle(settings: Any = None) -> StoreBundle:
    """Build the Hosted tier. Requires a reachable database."""
    dsn = resolve_dsn(settings)
    if not looks_like_dsn(dsn):
        raise ValueError(
            "STORAGE_BACKEND=postgres 需要配置 STORAGE_PG_DSN"
            "（或环境变量 PAPERWHISPERER_PG_DSN / DATABASE_URL / POSTGRES_DSN）。"
            "个人档无需任何配置，请改用 STORAGE_BACKEND=sqlite。"
        )
    schema = str(_setting(settings, "STORAGE_PG_SCHEMA") or "public")
    timeout = int(_setting(settings, "STORAGE_PG_CONNECT_TIMEOUT_SECONDS") or 5)
    ann = _as_bool(_setting(settings, "STORAGE_PG_ANN"))
    database = PostgresDatabase(dsn, schema=schema, connect_timeout=timeout)
    metadata = PostgresMetadataStore(database)
    vectors = PostgresVectorStore(database, ann=ann)
    lexical = PostgresLexicalStore(database)
    for store in (metadata, vectors, lexical):
        store.initialize()
    bundle = StoreBundle(
        metadata=metadata,
        vectors=vectors,
        lexical=lexical,
        tier=TIER_HOSTED,
        backend=BACKEND_POSTGRES,
        database=database,
    )
    bundle.capabilities = _merge_capabilities(
        TIER_HOSTED,
        metadata=metadata.capabilities(),
        vectors=vectors.capabilities(),
        lexical=lexical.capabilities(),
    )
    return bundle


def build_stores(settings: Any = None) -> StoreBundle:
    """Build the stores for the configured tier.

    ``settings`` is anything with the attributes in :data:`REQUIRED_CONFIG_KEYS`
    — normally :mod:`paperwhisperer.core.config` itself, or the module with
    its attributes monkeypatched in a test. Passing ``None`` falls back to the
    real config, so a caller that has no opinion gets the project's own.
    """
    if settings is None:
        from paperwhisperer.core import config as settings

    backend = resolve_backend(settings)
    builder = build_postgres_bundle if backend == BACKEND_POSTGRES else build_sqlite_bundle
    bundle = builder(settings)
    logger.info("存储后端已就绪：%s", bundle.describe())
    return bundle
