"""Document ingestion: parsers, canonical schema, quality gate and fallback.

The public surface is deliberately small. Callers want a
:class:`~paperwhisperer.ingestion.schema.ParsedDocument` for a file, so
``ingestion.parse_document`` is the entry point and the rest is implementation
detail that adapters and tests may reach into.

The name is a deliberate collision with the cloud adapter's own
``mineru_cloud.parse_document``: that one is one link in a chain, takes no
privacy decision and is reached only through the gate, while this one is the
whole business layer — it routes, repairs, asks the cloud when policy allows,
picks one whole document and caches it. Importing ``ingestion.parse_document``
can never accidentally reach the cloud one.
"""

from __future__ import annotations

from .orchestrator import (
    CloudFallbackDecision,
    ParseAttempt,
    ParseCache,
    ParseCacheKey,
    ParseOutcome,
    parse_document,
)
from .schema import (
    SCHEMA_VERSION,
    BoundingBox,
    Chunk,
    Element,
    Evidence,
    Fact,
    PaperFacts,
    ParsedDocument,
    SourceLocator,
    SourceUnit,
    Table,
)

__all__ = [
    "SCHEMA_VERSION",
    "BoundingBox",
    "Chunk",
    "CloudFallbackDecision",
    "Element",
    "Evidence",
    "Fact",
    "PaperFacts",
    "ParseAttempt",
    "ParseCache",
    "ParseCacheKey",
    "ParseOutcome",
    "ParsedDocument",
    "SourceLocator",
    "SourceUnit",
    "Table",
    "parse_document",
]
