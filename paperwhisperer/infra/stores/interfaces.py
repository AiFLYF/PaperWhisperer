"""The storage contracts both deployment tiers implement.

T1.1 splits the project into two tiers that must be *behaviourally* the same
product: a personal tier with no Docker at all, and a Hosted tier on
PostgreSQL. The plan's own words are that "两档必须通过同一套接口访问" — both
tiers reach storage through one set of interfaces — and that the guarantee is
made mechanical by running one test suite against each tier.

That guarantee only means something if the interfaces themselves are
neutral. A contract whose method names say ``load_blob`` or whose docstrings
promise WAL semantics would push the personal tier's actual behaviour into
the Hosted tier's tests, and the "same suite twice" would quietly become "the
Hosted suite, plus some SQLite-specific extras nobody compares". So nothing in
the :class:`MetadataStore`, :class:`VectorStore` or :class:`LexicalRetriever`
surface names a storage engine. Where a tier genuinely differs, it says so
through :meth:`capabilities`, which is data rather than contract.

Three interfaces, not one, because the three access patterns have genuinely
different performance shapes. A row lookup by primary key, an approximate
nearest-neighbour search and a full-text query are three different problems,
and merging them into a single "storage" object would push that choice onto
every caller. The types they exchange are the canonical ones from
:mod:`paperwhisperer.ingestion.schema` — a locator stored here must resolve
identically when it is read back, which is the whole reason those types have
exact ``to_dict`` / ``from_dict`` inverses.

Two design decisions carry most of the weight, and both exist to keep the
shared test suite honest:

**Scoring lives here, not in the backends.** A lexical or vector *search*
backend can only narrow candidates; the score a caller finally sees is
computed in Python by :mod:`paperwhisperer.infra.stores.scoring`, identically
for every tier. A backend-native ``ts_rank`` or ``bm25`` would be faster and
would make the two tiers return different numbers for the same corpus, at
which point the shared suite could only assert "both returned something". The
backends are allowed to be good at *finding* candidates; they are not allowed
to disagree about what a candidate scores.

**Degradation is reported, not swallowed.** A build of the personal tier
without a full-text accelerator still answers queries correctly, just by
scanning. Silently, that is a performance regression nobody would ever notice
from a test report. So every store publishes
:meth:`~StoreCapable.capabilities`, the factory collects the differences into
:attr:`StoreBundle.degradations`, and a degraded tier says so in Chinese in
the log — the same voice the parse-quality gate uses for a degraded parse.

What is *not* here: :class:`ObjectStore` and :class:`JobSystem` from the same
plan row. They are T2.1 and T2.2, they depend on the tables this module
introduces, and stubbing protocols for them now would mean two more
implementations the contract checker has to keep honest before anything
implements them. Adding a fourth and fifth interface later costs one file
edit; adding two shells that no backend implements costs a lie in
``tools/check_store_contract.py``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from paperwhisperer.ingestion.schema import Chunk, PaperFacts, ParsedDocument

# --------------------------------------------------------------------------
# Tier / backend vocabulary
# --------------------------------------------------------------------------

#: Deployment tiers. These name *what the user runs*, not which library is
#: underneath, so a tier survives a backend swap and a test can assert on it.
TIER_PERSONAL = "personal"
TIER_HOSTED = "hosted"

#: Selectable backends. ``STORAGE_BACKEND`` accepts exactly these.
BACKEND_SQLITE = "sqlite"
BACKEND_POSTGRES = "postgres"
STORAGE_BACKENDS = (BACKEND_SQLITE, BACKEND_POSTGRES)

#: Vector engines, as reported by :class:`StoreCapabilities`. Deliberately not
#: named after the libraries: the personal tier reports the same engine for a
#: numpy build and a pure-Python build unless it is honest about the cliff.
VECTOR_ENGINE_MATRIX = "matrix"
VECTOR_ENGINE_PURE_PYTHON = "python"
VECTOR_ENGINE_ANN = "ann"

#: Lexical engines. ``scan`` is the honest floor: correct, and linear. Only
#: two are defined because only two exist — a store that narrows candidates
#: with an index, and one that does not. A third name here would be a
#: vocabulary for a backend nobody has written, and an unused constant is
#: one more thing a reader has to wonder about.
LEXICAL_ENGINE_SCAN = "scan"
LEXICAL_ENGINE_FTS = "fts"

#: A document's index state. ``partial`` is a real outcome and not a softer
#: spelling of failure: the document is retrievable, but something was dropped
#: and the caller is expected to show that.
DOCUMENT_STATUSES = ("indexed", "partial", "failed")


# --------------------------------------------------------------------------
# Values exchanged with a store
# --------------------------------------------------------------------------


@dataclass(slots=True)
class DocumentRecord:
    """One document, as the metadata store holds it.

    The canonical :class:`ParsedDocument` rides along rather than being
    projected into loose columns, because the plan's exit criterion for P0 is
    that a locator survives a round trip. Storing the payload whole and
    rehydrating it through ``ParsedDocument.from_dict`` is the only way to
    *test* that, as opposed to asserting that a JSON column holds JSON.

    ``version`` is the optimistic-lock counter T2.2 needs. It is in the
    contract now because a lock that is added later has to guess what the
    initial value was, and every row already written would be wrong.
    """

    document_id: str
    document: ParsedDocument | None = None
    source_filename: str = ""
    media_type: str = ""
    sha256: str = ""
    page_count: int = 0
    parser_name: str = ""
    parser_version: str = ""
    options_hash: str = ""
    status: str = "indexed"
    index_version: str = ""
    created_at: str = ""
    updated_at: str = ""
    version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in DOCUMENT_STATUSES:
            self.status = "indexed"
        if not isinstance(self.metadata, dict):
            self.metadata = {}

    def to_dict(self) -> dict[str, Any]:
        """The wire form. Lossless in both directions, like every canonical type."""
        return {
            "document_id": self.document_id,
            "document": self.document.to_dict() if self.document is not None else None,
            "source_filename": self.source_filename,
            "media_type": self.media_type,
            "sha256": self.sha256,
            "page_count": self.page_count,
            "parser_name": self.parser_name,
            "parser_version": self.parser_version,
            "options_hash": self.options_hash,
            "status": self.status,
            "index_version": self.index_version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DocumentRecord:
        document_payload = payload.get("document")
        return cls(
            document_id=str(payload.get("document_id") or ""),
            document=(
                ParsedDocument.from_dict(dict(document_payload))
                if isinstance(document_payload, Mapping)
                else None
            ),
            source_filename=str(payload.get("source_filename") or ""),
            media_type=str(payload.get("media_type") or ""),
            sha256=str(payload.get("sha256") or ""),
            page_count=int(payload.get("page_count") or 0),
            parser_name=str(payload.get("parser_name") or ""),
            parser_version=str(payload.get("parser_version") or ""),
            options_hash=str(payload.get("options_hash") or ""),
            status=str(payload.get("status") or "indexed"),
            index_version=str(payload.get("index_version") or ""),
            created_at=str(payload.get("created_at") or ""),
            updated_at=str(payload.get("updated_at") or ""),
            version=int(payload.get("version") or 1),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(slots=True)
class VectorEntry:
    """One stored embedding, plus the keys a filtered search needs.

    ``vector`` is a tuple so a caller cannot mutate the store's idea of what
    it holds after handing it over. ``metadata`` is whatever the caller wants
    to filter on later — ``{"heading": "Results"}`` in practice — and it
    round-trips as JSON.
    """

    chunk_id: str
    vector: tuple[float, ...]
    document_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.vector = tuple(float(value) for value in self.vector)
        if not isinstance(self.metadata, dict):
            self.metadata = {}


@dataclass(slots=True, frozen=True)
class VectorHit:
    """One search result: the key, and the score the scorer computed."""

    chunk_id: str
    score: float
    document_id: str = ""


@dataclass(slots=True, frozen=True)
class LexicalHit:
    """One lexical result, with a short excerpt for display without a re-read.

    The snippet is generated by the scorer from the stored text, not by the
    caller, so both tiers cut the same window around the same match. A caller
    that built its own snippet would have to re-read the document to do it,
    which is exactly the round trip this interface exists to avoid.
    """

    chunk_id: str
    score: float
    document_id: str = ""
    snippet: str = ""


@dataclass(slots=True)
class StoreCapabilities:
    """What a store can actually do here, as opposed to what it promises.

    The plan's lexical row is a two-column table whose second column is a
    *fallback* — "构建时探测，可用则启用". Probing is worthless if the answer
    is not recorded anywhere a human will see, so this is the record, and
    :attr:`degradations` is the part that goes in a log and in the UI.
    """

    tier: str = TIER_PERSONAL
    vector_engine: str = VECTOR_ENGINE_MATRIX
    lexical_engine: str = LEXICAL_ENGINE_SCAN
    #: True when the store does no narrowing of its own, i.e. a query is
    #: answered by reading the whole corpus. This is the one performance fact
    #: a user can notice, and it is the reason the Hosted tier exists.
    scan_only: bool = True
    #: Human-readable differences from the fastest available path, most
    #: important first. Empty means nothing was given up.
    degradations: tuple[str, ...] = ()

    def describe(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "vector_engine": self.vector_engine,
            "lexical_engine": self.lexical_engine,
            "scan_only": self.scan_only,
            "degradations": list(self.degradations),
        }


# --------------------------------------------------------------------------
# Contracts
# --------------------------------------------------------------------------


@runtime_checkable
class StoreCapable(Protocol):
    """The one method every store shares, so degradation is always askable.

    It is a separate protocol rather than a method repeated on all three:
    a factory can then accept "anything with capabilities", and the contract
    checker can hold every store to the same reporting obligation without the
    three main protocols inheriting from a common base.
    """

    def capabilities(self) -> StoreCapabilities:
        """Describe what this instance is actually doing.

        Called once at construction by the factory and again on demand by the
        health endpoint. It must not touch the network or the disk — it
        answers from what the constructor already established.
        """
        ...


@runtime_checkable
class MetadataStore(StoreCapable, Protocol):
    """Durable rows: documents, chunks and the extracted fact layer.

    This is the only write path for structured data. A caller that reached
    into the underlying database directly would tie itself to one tier, which
    is the failure the two-tier split exists to prevent; ``tools/
    check_store_contract.py`` exists because a Protocol is documentation until
    something checks it.
    """

    def initialize(self) -> None:
        """Create or migrate whatever this store needs. Idempotent.

        Called once by the factory, and again from tests that want a store
        with no inherited state. "Idempotent" is the whole contract: a
        migration that has to know whether it already ran is a migration
        nobody can safely retry.
        """
        ...

    def upsert_document(self, record: DocumentRecord) -> DocumentRecord:
        """Insert or replace a document row, and return it as stored.

        Returning the stored form rather than echoing the argument is
        deliberate. The store owns the fields a caller does not supply —
        ``version`` and the timestamps — and a caller that trusts its own
        copy would be holding a row that does not exist.
        """
        ...

    def get_document(self, document_id: str) -> DocumentRecord | None:
        """Return the document row, or ``None`` when it is not there.

        A missing row is a ``None`` and not an exception: "this document was
        deleted" is an ordinary outcome of a concurrent request, and the
        callers are easier to read when they have to say so.
        """
        ...

    def list_documents(self, limit: int = 50, offset: int = 0) -> list[DocumentRecord]:
        """List document rows newest first, paged.

        Ordered by creation time and then by id. The id tiebreak is not
        cosmetic: two rows written in the same millisecond are common under
        concurrent writes, and without it the same query returns a different
        page order on the two tiers, which is a pagination bug that only
        appears in production.
        """
        ...

    def delete_document(self, document_id: str) -> bool:
        """Delete a document and everything hanging off it. Idempotent.

        Returns whether a row was actually removed, so a caller can tell a
        delete from a no-op. Chunks, facts and embeddings all go with it:
        leaving an orphan chunk behind is how a re-indexed document starts
        returning the text of the version it replaced.
        """
        ...

    def put_chunk(self, document_id: str, chunk: Chunk) -> Chunk:
        """Store one retrieval unit against a document. Replaces by ``chunk_id``."""
        ...

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Return one chunk, or ``None``."""
        ...

    def list_chunks(self, document_id: str) -> list[Chunk]:
        """Every chunk of a document, in reading order."""
        ...

    def delete_chunks(self, document_id: str) -> int:
        """Drop a document's chunks, returning how many went."""
        ...

    def put_facts(self, document_id: str, facts: PaperFacts) -> PaperFacts:
        """Store the fact layer, replacing any previous extraction.

        Replace, not merge: two extractions of the same document disagree
        often enough that keeping both would mean the fact layer has no
        single answer to the question "what does this paper claim".
        """
        ...

    def get_facts(self, document_id: str) -> PaperFacts | None:
        """Return the fact layer, or ``None`` if it was never extracted."""
        ...

    def count_documents(self) -> int:
        """How many documents are stored."""
        ...

    def close(self) -> None:
        """Release connections. Idempotent, and never raises.

        Called from ``finally`` blocks in tests and from the app's shutdown
        hook. A close that raised would replace the error that caused the
        shutdown with a second, less useful one.
        """
        ...


@runtime_checkable
class VectorStore(StoreCapable, Protocol):
    """Similarity search over embeddings.

    The contract is cosine similarity over unit-length vectors, and it is
    stated in those terms rather than as a distance because a caller should
    not have to know which of the two the store computed internally. Scores
    are therefore comparable across tiers, which is what lets one test assert
    an exact expected score on both.
    """

    def initialize(self) -> None:
        """Create or migrate the vector space. Idempotent."""
        ...

    @property
    def dimension(self) -> int | None:
        """The vector width, or ``None`` before the first write.

        ``None`` rather than ``0``: "no vectors yet" and "vectors of width
        zero" are different states, and conflating them is how an empty index
        reports itself as misconfigured.
        """
        ...

    def upsert_vector(
        self,
        chunk_id: str,
        vector: Sequence[float],
        *,
        document_id: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Store one embedding, replacing any previous one for ``chunk_id``.

        Raises :class:`ValueError` on a vector whose width differs from the
        space already established, and on an all-zero vector. Both are
        refused rather than repaired: a width mismatch means the caller's
        embedding model changed, and silently truncating or padding would
        produce search results that are confidently wrong. A zero vector has
        no direction, so its cosine similarity to anything is undefined —
        and a store that returned ``0.0`` for it would be reporting a number
        it did not compute.
        """
        ...

    def upsert_many(self, entries: Iterable[VectorEntry]) -> int:
        """Write many embeddings, returning how many were written.

        Exists as its own method because the tiers' costs differ by two
        orders of magnitude: one round trip against a server, a loop against a
        file. A caller looping over :meth:`upsert_vector` would work on both
        and be needlessly slow on one.
        """
        ...

    def get_vector(self, chunk_id: str) -> VectorEntry | None:
        """Return one stored embedding, or ``None``."""
        ...

    def delete_vector(self, chunk_id: str) -> bool:
        """Delete one embedding, returning whether it was there."""
        ...

    def delete_document(self, document_id: str) -> int:
        """Delete every embedding belonging to a document, returning the count."""
        ...

    def search(
        self,
        vector: Sequence[float],
        top_k: int = 10,
        *,
        document_ids: Sequence[str] | None = None,
        min_score: float | None = None,
    ) -> list[VectorHit]:
        """Return the ``top_k`` nearest vectors, most similar first.

        Ties break on ``chunk_id`` ascending so the ordering is total and
        both tiers return the same list. A partial order here would make
        ``top_k`` itself unstable at the boundary — the classic case being a
        query whose 10th and 11th results score identically, where the
        personal tier and the Hosted tier keep different chunks.
        """
        ...

    def count_vectors(self) -> int:
        """How many embeddings are stored."""
        ...

    def close(self) -> None:
        """Release connections. Idempotent, and never raises."""
        ...


@runtime_checkable
class LexicalRetriever(StoreCapable, Protocol):
    """Exact and partial term matching over stored text.

    Scoring is token coverage, defined once in
    :mod:`paperwhisperer.infra.stores.scoring` and computed in Python for
    every tier. That is a deliberate trade of some throughput for a guarantee
    the product actually needs: an evidence system whose two deployment tiers
    rank the same corpus differently is one where a retrieval regression
    reproduces only on the machine the owner happened to be running.

    Backends may still narrow candidates with an index, and the personal tier
    does so when its build has one. What they may not do is decide the score.
    """

    def initialize(self) -> None:
        """Create or migrate the search space. Idempotent."""
        ...

    def index_chunk(
        self,
        chunk_id: str,
        text: str,
        *,
        document_id: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Make one chunk's text searchable. Replaces by ``chunk_id``.

        The text is stored normalized, alongside the original. Normalized
        only would make snippets unreadable — case folding and CJK bigramming
        destroy the sentence a reader would want quoted — and the original
        only would make every query a full scan.
        """
        ...

    def delete_chunk(self, chunk_id: str) -> bool:
        """Remove one chunk from the index, returning whether it was there."""
        ...

    def delete_document(self, document_id: str) -> int:
        """Remove every chunk of a document, returning the count."""
        ...

    def search(
        self,
        query: str,
        top_k: int = 10,
        *,
        document_ids: Sequence[str] | None = None,
    ) -> list[LexicalHit]:
        """Return the ``top_k`` best-matching chunks, best first.

        Chunks that share no term with the query are never returned, however
        high ``top_k`` is. Returning low-similarity filler to pad a result
        list would let an unanswered question look answered, which is the
        one outcome the product exists to prevent.
        """
        ...

    def count_chunks(self) -> int:
        """How many chunks are indexed."""
        ...

    def close(self) -> None:
        """Release connections. Idempotent, and never raises."""
        ...
