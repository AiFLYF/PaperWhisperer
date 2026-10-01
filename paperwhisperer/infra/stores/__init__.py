"""Storage backends. Both deployment tiers implement the same interfaces.

The package is arranged so that *which* backend is running is the one fact a
caller cannot see:

``interfaces``
    The three contracts (:class:`MetadataStore`, :class:`VectorStore`,
    :class:`LexicalRetriever`) and the values they exchange. No method name or
    docstring here mentions a storage engine — that neutrality is what lets
    the same test suite run against both tiers.
``scoring``
    How a match is normalized, tokenized, scored and ordered. Shared by both
    tiers on purpose; see the module docstring for why.
``sqlite_backend`` / ``postgres_backend``
    The two implementations. Neither imports the other.
``factory``
    :func:`build_stores`, the single place a backend is chosen.

Start at :mod:`paperwhisperer.infra.stores.interfaces` — it explains the
decisions the rest of the package is an implementation of.
"""

from __future__ import annotations

from .factory import (
    REQUIRED_CONFIG_KEYS,
    StoreBundle,
    build_postgres_bundle,
    build_sqlite_bundle,
    build_stores,
    resolve_backend,
    resolve_dsn,
)
from .interfaces import (
    BACKEND_POSTGRES,
    BACKEND_SQLITE,
    DOCUMENT_STATUSES,
    LEXICAL_ENGINE_FTS,
    LEXICAL_ENGINE_SCAN,
    STORAGE_BACKENDS,
    TIER_HOSTED,
    TIER_PERSONAL,
    VECTOR_ENGINE_ANN,
    VECTOR_ENGINE_MATRIX,
    VECTOR_ENGINE_PURE_PYTHON,
    DocumentRecord,
    LexicalHit,
    LexicalRetriever,
    MetadataStore,
    StoreCapabilities,
    StoreCapable,
    VectorEntry,
    VectorHit,
    VectorStore,
)
from .sqlite_backend import (
    LEXICAL_MODE_AUTO,
    LEXICAL_MODE_REQUIRED,
    LEXICAL_MODE_SCAN,
    LEXICAL_MODES,
    SqliteDatabase,
    SqliteLexicalStore,
    SqliteMetadataStore,
    SqliteVectorStore,
)

__all__ = [
    "BACKEND_POSTGRES",
    "BACKEND_SQLITE",
    "DOCUMENT_STATUSES",
    "LEXICAL_ENGINE_FTS",
    "LEXICAL_ENGINE_SCAN",
    "LEXICAL_MODE_AUTO",
    "LEXICAL_MODE_REQUIRED",
    "LEXICAL_MODE_SCAN",
    "LEXICAL_MODES",
    "REQUIRED_CONFIG_KEYS",
    "STORAGE_BACKENDS",
    "TIER_HOSTED",
    "TIER_PERSONAL",
    "VECTOR_ENGINE_ANN",
    "VECTOR_ENGINE_MATRIX",
    "VECTOR_ENGINE_PURE_PYTHON",
    "DocumentRecord",
    "LexicalHit",
    "LexicalRetriever",
    "MetadataStore",
    "SqliteDatabase",
    "SqliteLexicalStore",
    "SqliteMetadataStore",
    "SqliteVectorStore",
    "StoreBundle",
    "StoreCapabilities",
    "StoreCapable",
    "VectorEntry",
    "VectorHit",
    "VectorStore",
    "build_postgres_bundle",
    "build_sqlite_bundle",
    "build_stores",
    "resolve_backend",
    "resolve_dsn",
]
