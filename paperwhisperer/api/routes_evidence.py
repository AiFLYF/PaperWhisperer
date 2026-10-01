"""Evidence API — the endpoint a citation click opens.

One route, and it exists because of a promise. In an Evidence-first product a
citation chip is a claim that clicking it shows the passage behind the
sentence, and the whole API is arranged so that promise is kept or declared
unkeepable — never quietly broken. This module is the HTTP edge of that: it
finds the document, finds the evidence, and hands both to
:func:`paperwhisperer.rag.evidence.resolve_evidence`, which owns every
decision about where the evidence actually points. Keeping the decisions out
of the route is what lets the same resolution be unit-tested without a server
and reused by the T3.3 viewer without a second copy drifting from the first.

**Two id namespaces, one promise.** The path parameter is an *evidence* id, and
that is not a synonym for a chunk id. The retriever mints ``ev_…`` ids around the
chunks it returns and the front end's citation chips carry exactly those, so a
route that resolved its parameter only as a chunk id answered ``404`` to every id
the product actually emits — the central promise failing for the ids it mints
itself, while a suite that only ever addressed ``chunk_*`` stayed green. The route
therefore resolves both namespaces: a chunk id directly, and an evidence id by
re-deriving the retriever's own content address over *this document's* chunks
(:func:`paperwhisperer.rag.evidence.find_chunk_for_evidence_id`). Scoping that
re-derivation to the document named in the URL is what keeps the cross-document
refusal below intact on the new path without a second check.

**The re-derivation's candidate set is a union, not a registry.** An evidence id
is a content address over ``(chunk, plan_signature)``, so resolving one means
retrying it against the signatures the document could have been queried under —
and the registered plans are not all of those. A plan built with
``with_overrides`` has a signature no registry enumerates, and the fact layer
builds one that way for every field it extracts; those are the ids an analysis
card's citation chips would carry. So the document record is passed through to
the resolver, which unions the recorded signatures with the registered ones.
This is recorded on the document rather than learned in-process precisely
because the ids are stored and outlive the process that minted them.

The status codes are the other half of the promise, and they are chosen so
that no two different failures look alike to a client:

* **400** for an id that is not shaped like an id. Rejected before any store
  is touched, because a malformed id cannot name a document and a lookup
  would answer "not found" — turning a client's own bug into a 404 that
  sends it looking for a document that was there all along.
* **404 unknown document**, because the document really is not there.
* **404 unknown evidence**, because the id names nothing at all.
* **404 evidence belonging to another document**, which is a *different*
  answer and gets a different code and a different sentence. Serving another
  document's evidence because the caller guessed an id would be a real
  information leak, and answering it with the same 404 a missing id gets
  would mean a client cannot tell a typo from someone else's citation.

There is deliberately no 500 on the happy path. If resolution cannot produce a
target it says so in the body — the ``visual_targeting`` block — rather than
failing the request, because a citation that cannot be located is a normal,
expected outcome for a degraded parse (the schema has an ``unknown`` locator
kind precisely for it) and not an error. The one thing this API will never do
is return a target it cannot justify.

**The page image.** T3.3's last unmet criterion was a real raster of the page,
and the answer is ``pypdfium2`` behind the optional ``render`` extra. Getting
this right is less about PNG encoding than about **geometry**: a highlight
drawn at the wrong scale, or mirrored about the page's top edge, is worse than
no highlight at all, because it looks correct. So the response states the scale
it rendered at and the viewport it produced, and the client applies exactly two
operations to a ``BoundingBox`` to land on it —
:meth:`paperwhisperer.rag.evidence.page_image_geometry` is the contract, and
``tests/test_page_render.py`` asserts it against real ink on a real page
rather than against a restatement of the formula.

The route renders **unrotated** (a page's ``/Rotate`` is neutralised before
rendering) so that the ``rotation == 0`` transform is correct for every
document and the client has no special case to get wrong. The true rotation is
reported in ``geometry.source_rotation`` rather than hidden, because a sideways
page is a visible, explainable thing and a highlight on the wrong lines is not.

**The 501 branch is still here, and it is load-bearing.** ``pypdfium2`` is an
extra, not a base dependency: the personal tier must run with the smallest
footprint it can, and a rendering nicety is not core. A deployment that never
installed the extra still gets 501 with a machine code and the reason, and
still gets a correct ``view`` built from the parse's own geometry. The client
therefore has three distinguishable outcomes — "here is the page", "this build
cannot rasterise", and "here is the page but no defensible box" — and the
distinctions are the whole reason the codes exist.

**Where the bytes come from, and the gap that is honestly reported.** The
original upload is needed to raster anything, and T2.2's
:class:`~paperwhisperer.infra.object_store.FilesystemObjectStore` is where it
is meant to live — content-addressed in the ``original`` layer, which makes
"find this document's bytes" arithmetic (:meth:`find_by_digest`) rather than a
directory walk. That store is **not yet wired into
:class:`~paperwhisperer.infra.stores.factory.StoreBundle`**, and nothing in
the codebase currently writes an upload into it, so on a fresh checkout the
honest answer for most documents is :data:`ORIGINAL_UNAVAILABLE_REASON`: the
renderer works, the document is there, and the bytes that would prove which
page it is have been deleted. That is reported as its own status rather than
papered over with a blank image, and this route does not block on T2.2 to fix
it: it reads the object store when a root is configured and otherwise falls
back to a path carried on the document record.
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
import re
import threading
from collections import OrderedDict
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, Response

from paperwhisperer.core import config
from paperwhisperer.core.errors import build_error_payload, build_error_response
from paperwhisperer.infra.object_store import (
    LAYER_ORIGINAL,
    FilesystemObjectStore,
    ObjectStoreError,
    digest_key,
)
from paperwhisperer.infra.stores import StoreBundle, build_stores
from paperwhisperer.infra.stores.interfaces import DocumentRecord
from paperwhisperer.ingestion.schema import Chunk, ParsedDocument
from paperwhisperer.rag.evidence import (
    find_chunk_for_evidence_id,
    page_image_geometry,
    page_renderer_name,
    resolve_evidence,
    resolve_page_render_scale,
)

logger = logging.getLogger(__name__)

router = APIRouter()

#: What an id is allowed to look like. Deliberately narrow — letters, digits,
#: and the punctuation the schema's own id prefixes use (``doc_…``, ``ev_…``,
#: ``chunk_…``) — because these come straight off the wire and end up as a
#: store key. ``..`` is rejected on top of the character class: a traversal
#: sequence is built from characters the class already allows, so the class
#: alone would wave it through. Whether a traversal is exploitable is a
#: property of the storage tier, and a rule that depends on which tier is
#: running is not a rule.
_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


def _is_well_formed(value: str) -> bool:
    """Whether a path id could name anything at all."""
    return bool(_ID_PATTERN.match(value)) and ".." not in value


#: The process-wide store bundle, built once on first use.
#:
#: ``build_stores`` is the single place a backend is chosen, so this defers to
#: it rather than picking a database — an evidence request has no business
#: knowing which tier answered it. It is a module global because a store per
#: request would re-open a connection pool per click, and because the two
#: tiers share one lifecycle (they are opened and closed together by
#: :meth:`StoreBundle.close`). Tests reach it through FastAPI's
#: ``dependency_overrides``, so no test ever opens a real database.
_STORES: StoreBundle | None = None


def get_store_bundle() -> StoreBundle:
    """Return the shared store bundle, building it on first call."""
    global _STORES
    if _STORES is None:
        _STORES = build_stores(config)
    return _STORES


def _resolve_chunk(
    stores: StoreBundle,
    document_id: str,
    evidence_id: str,
    chunks: list[Chunk],
    record: DocumentRecord | None = None,
) -> tuple[Chunk | None, str]:
    """Find the chunk this id addresses, in either id namespace.

    Two namespaces, one meaning. The *chunk* id addresses the stored retrieval
    unit; the *evidence* id is the ``ev_…`` name the retriever, the Q&A claims and
    the front end's citation chips all put on the wire. A reader clicks the
    second, so a route that only understood the first answered 404 to every id the
    product actually emits — the "click a citation and open the passage" promise
    failing for the ids it mints. Both are resolved here, in the route, because
    the resolver is a pure function of the chunk and the decisions about *which*
    namespace applies belong at the edge.

    **Order matters, and it is chunk-first.** A chunk id is looked up directly and
    is exact; only if that misses do we try to re-derive the chunk from an evidence
    id. A chunk id can never be mistaken for an evidence id (different shape), so
    this order is not a heuristic — it is simply trying the exact lookup first.

    **The cross-document guarantee survives both branches.** In the chunk branch it
    is the explicit membership test against this document's own chunk list (a
    ``Chunk`` carries no ``document_id``, so membership is the only available
    proof of ownership, and a chunk that exists but is not among them is a
    citation from another paper — refused, and given its own error key so the
    client can tell a typo from someone else's citation). In the evidence branch it
    is structural: :func:`find_chunk_for_evidence_id` is only ever handed *this*
    document's chunks, so a foreign evidence id has no candidate to match and
    resolves to the same not-found a typo gets. Nothing outside the named
    document is ever consulted.

    **``record`` carries the plan signatures this document was retrieved under**,
    and it widens the *candidate set*, not the candidate *chunks*. That is the
    distinction the whole cross-document argument rests on: a signature learned
    from another paper would only add a hash to try, because the chunks remain
    this document's. The record is optional so a caller without one still
    resolves every registered-plan id exactly as before.
    """
    chunk = stores.metadata.get_chunk(evidence_id)
    if chunk is not None:
        owned = any(candidate.chunk_id == evidence_id for candidate in chunks)
        return chunk, "" if owned else "evidence_not_in_document"
    return find_chunk_for_evidence_id(
        evidence_id, chunks, document_record=record
    ), ""


def _evidence_payload(
    stores: StoreBundle, document_id: str, evidence_id: str
) -> tuple[dict | None, str]:
    """Fetch and resolve one evidence, returning ``(payload, error_key)``.

    Exactly one half is ever non-``None``, so the route bails out with a
    single check — the same shape :func:`parse_json_object_request` uses, and
    for the same reason: a lookup that can fail four different ways is much
    easier to read when each failure is a named key rather than an exception
    unwound through a shared frame.

    The id namespaces are reconciled in :func:`_resolve_chunk`; the document
    record is fetched first so an unknown document is reported as such before any
    lookup runs, and the resolver itself stays a pure function of (document, chunk)
    so T3.3 can call it without a store at all.
    """
    record = stores.metadata.get_document(document_id)
    if record is None:
        return None, "document_not_found"

    chunk, error_key = _resolve_chunk(
        stores,
        document_id,
        evidence_id,
        stores.metadata.list_chunks(document_id),
        record,
    )
    if error_key:
        return None, error_key
    if chunk is None:
        return None, "evidence_not_found"

    document = record.document or ParsedDocument(document_id=document_id)
    payload = resolve_evidence(
        evidence_id=evidence_id,
        document=document,
        chunk=chunk,
        document_record=record,
    ).to_dict()
    return payload, ""


#: One Chinese sentence and one machine code per failure. Kept as a table
#: rather than inline at each return so the message a client reads and the
#: code it branches on are written down together and cannot drift apart.
_ERRORS: dict[str, tuple[int, str, str]] = {
    "document_not_found": (404, "文档不存在。", "document_not_found"),
    "evidence_not_found": (404, "该证据不存在。", "evidence_not_found"),
    "evidence_not_in_document": (404, "该证据不属于此文档。", "evidence_not_in_document"),
}

#: The status a page-image request gets when *this build* cannot rasterise. 501
#: rather than 404, 500 or 200 with an empty body, because those three each say
#: something false: 404 says the page is not in the document, 500 says the
#: server broke, and an empty 200 says "here is your page" while handing over
#: nothing. The reader's decision turns on exactly this distinction — "no
#: picture" versus "no highlight" are different problems with different fixes.
#:
#: It is a *branch*, not the whole route. The code is unchanged from the build
#: that had no renderer at all, because a client that learned the code once
#: should not have to learn it again when an operator installs the extra; what
#: changes is which deployments take it.
PAGE_RENDER_STATUS = 501
PAGE_RENDER_CODE = "page_render_unavailable"
PAGE_RENDER_MESSAGE = (
    "本版本没有 PDF 页图渲染能力：运行环境没有安装可选的渲染依赖 pypdfium2"
    "（可安装 `pip install -e \".[render]\"` 获得该能力）。证据面板改用解析得到的"
    "元素坐标重建页内布局来定位，该视图会明确标注它不是原始渲染图。"
)

#: Why a page image could not be produced, when "could not produce a page image"
#: is not by itself specific enough for a client to act on.
#:
#: Both causes below answer 501 with :data:`PAGE_RENDER_CODE`, and that is
#: deliberate rather than a compromise. The code was written down before a
#: renderer existed and is mirrored in the front end as
#: ``PAGE_RENDER_UNAVAILABLE_CODE``; keeping it is what means a deployment that
#: installs the extra, or one that does not, is handled by a client that was
#: written against the code that was already there. From the client's side
#: these are the same outcome — no page image, so draw something else — and
#: inventing a second code under the same status would be a client-facing
#: change made for the benefit of a log line.
#:
#: The distinction is therefore carried in a *reason* alongside the code rather
#: than displacing it, and it is the reason that matters: one of these is fixed
#: by ``pip install -e ".[render]"`` and the other is fixed by the pipeline
#: retaining uploads. A user who is told the first and given the second retries
#: forever. The field is additive, so a client reading only ``code`` is
#: unaffected, and one that learns to read ``reason`` gets the better message.
#:
#: 500 is explicitly *not* the answer, and neither is a blank white PNG. A 500
#: tells the operator something broke, when what happened is that a design
#: decision elsewhere (deleting the upload after analysis) left this route with
#: nothing to draw. A blank PNG is worse than either: it is a 200, it looks
#: like a page, and a highlight placed on it would point at nothing while
#: appearing to be perfectly correct.
ORIGINAL_UNAVAILABLE_REASON = "original_unavailable"
ORIGINAL_UNAVAILABLE_MESSAGE = (
    "该文档的原始文件字节已不在本机，因此无法渲染页图。"
    "解析结果与证据引用不受影响：证据面板仍会显示原文片段、页码与"
    "（如有）高亮框。要恢复页图，需要部署保留用户上传的对象存储 original 层。"
)

#: A page number past the end of the document. 404, not 400: the *page number*
#: is well-formed (page < 1 is the 400), the document is there, and the page
#: named is not. A client that paginates past the end lands here routinely.
PAGE_OUT_OF_RANGE_STATUS = 404
PAGE_OUT_OF_RANGE_CODE = "page_out_of_range"

#: Not a PDF. 415, because the request is well-formed and the resource exists —
#: it is the representation that is wrong, which is exactly what 415 is for and
#: what makes it distinguishable from "this deployment has no renderer".
#:
#: The check is on the *media type and the bytes*, not on the filename. A
#: ``.pdf`` that is really a ZIP is the case that matters: trusting the
#: extension would hand the rasteriser a DOCX and produce either an exception
#: or, worse, a page of a document the user never opened.
UNSUPPORTED_FORMAT_STATUS = 415
UNSUPPORTED_FORMAT_CODE = "page_render_unsupported_format"

#: The renderer is installed but refused these bytes. Kept separate from the 415
#: above because they are different diagnoses: 415 means "not a PDF", this means
#: "a PDF the rasteriser could not open" (encrypted, truncated, or a PDF
#: revision pdfium does not implement). Both are 4xx and both are the client's
#: or the file's problem rather than the server's, but they get different
#: sentences because "upload a different file" and "the server is out of disk"
#: are not the same advice.
RENDER_FAILED_STATUS = 422
RENDER_FAILED_CODE = "page_render_failed"

#: The first eight bytes of every PDF, per the PDF specification. Checked
#: because it is the only format test that cannot be fooled by an extension or
#: by a declared media type.
PDF_MAGIC = b"%PDF-"


def _is_pdf_bytes(payload: bytes) -> bool:
    """Whether these bytes begin with a PDF header.

    Checked at the head rather than scanned for, on purpose. A byte string
    produced by a converter that emits a few bytes of preamble before
    ``%PDF-`` is technically malformed and pdfium will usually reject it
    anyway; searching the whole buffer for the magic instead would accept files
    the renderer cannot open and turn a clear error into a 500 from deep inside
    the rasteriser.
    """
    return payload[: len(PDF_MAGIC)] == PDF_MAGIC


# --------------------------------------------------------------------------
# Resolving the document's ORIGINAL bytes
# --------------------------------------------------------------------------
#
# Everything here exists because a raster is a function of the original file and
# nothing else. There is no way to render page 3 of a document the server no
# longer holds the bytes of, and the temptation — render a white page, or
# re-serialise the canonical document back into something PDF-shaped — is
# exactly the wrong-highlight failure in a new costume.
#
# Two sources are tried, in order, and the order is a judgement about which is
# more likely to be *correct* rather than which is cheaper:
#
# 1. An object key carried on the document record. A store that retains
#    uploads will have written one, and it is authoritative because the
#    ``original`` layer is write-once: the key names bytes that cannot have
#    been rewritten.
# 2. The ``original`` layer addressed by the record's ``sha256``. This is
#    T2.2's content-addressing scheme and it is the preferred lookup even
#    though it needs a configured root, because :meth:`find_by_digest` resolves
#    it arithmetically from the digest — no directory walk, nothing to cache,
#    nothing to invalidate. A digest is also the one identifier that cannot be
#    stale: it is the content.
# 3. A filesystem path on the record. Last, and treated as a hint rather than a
#    fact: it is a path in a directory this process may not control, so the
#    resolved real path is checked to be a readable regular file and the bytes
#    are checked to be a PDF before anything is rendered.


def _original_key_candidates(record: Any) -> list[str]:
    """Object keys this document's original might be filed under.

    The order is longest-specificity first and is not arbitrary: a store that
    keeps both a named key and a content address will have written the named
    one, and looking for the digest first would work but would silently prefer
    a file whose *name* nobody chose over the one a caller actually recorded.
    """
    metadata = getattr(record, "metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    candidates: list[str] = []

    for field in ("original_key", "original_ref", "object_key", "storage_key"):
        value = metadata.get(field)
        if value:
            candidates.append(str(value))
            # A ref may arrive as "layer/key"; the original layer is the only
            # one this route will read, so the key half is what is wanted.
            if "/" in str(value):
                candidates.append(str(value).split("/", 1)[1])

    digest = str(getattr(record, "sha256", "") or "").strip().lower()
    if len(digest) == 64 and all(character in "0123456789abcdef" for character in digest):
        candidates.append(digest_key(digest))
        # A ``.pdf`` suffix is the other spelling, and object_store's own
        # ``find_by_digest`` only tries the layer's declared suffix (empty for
        # ``original``), so the suffixed form is tried here explicitly.
        candidates.append(digest_key(digest, ".pdf"))
    return candidates


def _object_store_for_originals():
    """A filesystem object store over the configured ``original`` root, or ``None``.

    Built per lookup rather than cached at module scope, because the root is
    configuration and a test (or an operator) can change it between requests;
    the construction is two attribute reads and a list comprehension, and
    holding a store whose root no longer matches the setting would be a stale
    object store, which is the exact class of bug this route exists to avoid.

    Returns ``None`` rather than raising when the root is unusable, because "no
    configured store" and "a store that cannot be built" both mean the same
    thing here: this source is unavailable and the next one should be tried.
    """
    root = str(getattr(config, "PAGE_RENDER_ORIGINALS_DIR", "") or "").strip()
    if not root:
        return None
    try:
        store = FilesystemObjectStore(root)
        if not os.path.isdir(store.layer_dir(LAYER_ORIGINAL)):
            return None
        return store
    except (ObjectStoreError, OSError, ValueError):
        logger.debug("object store unusable for original lookup: root=%r", root, exc_info=True)
        return None


def _read_original(record: Any) -> bytes | None:
    """The document's original bytes, or ``None`` when they are not available.

    Returning ``None`` rather than raising is deliberate: every reason the bytes
    are missing — no store configured, no key on the record, the file was
    deleted after analysis — is the *same answer* to the client, and
    :data:`ORIGINAL_UNAVAILABLE_REASON` is the honest thing to say for all of
    them. Distinguishing them in the response would tell a user things about
    the server's storage layout that they cannot act on.

    The bytes are read in full rather than streamed because pdfium needs a
    seekable buffer, and :meth:`PdfDocument.new_native` accepts one. The size
    is bounded by the same ceiling the upload path enforces, so a record whose
    ``original`` key points at something enormous is refused rather than read.
    """
    metadata = getattr(record, "metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}

    store = _object_store_for_originals()
    if store is not None:
        for key in _original_key_candidates(record):
            try:
                if not store.exists(LAYER_ORIGINAL, key):
                    continue
                with store.open_stream(LAYER_ORIGINAL, key) as handle:
                    payload = handle.read(_MAX_ORIGINAL_BYTES + 1)
                if len(payload) <= _MAX_ORIGINAL_BYTES:
                    return payload
                logger.warning("original object too large to render: key=%r", key)
                return None
            except (ObjectStoreError, OSError):
                logger.debug("original object unreadable: key=%r", key, exc_info=True)

    path = str(
        metadata.get("original_path") or metadata.get("file_path") or ""
    ).strip()
    if path and os.path.isfile(path):
        try:
            if os.path.getsize(path) <= _MAX_ORIGINAL_BYTES:
                with open(path, "rb") as handle:
                    return handle.read()
        except OSError:
            logger.debug("original path unreadable: %r", path, exc_info=True)
    return None


#: Ceiling on the bytes read for one render. Matches the upload limit's intent
#: rather than its exact value: this is a guard against a record pointing at a
#: disk image, not a policy, and a document that parsed in the first place is
#: very unlikely to be near it.
_MAX_ORIGINAL_BYTES = 512 * 1024 * 1024


# --------------------------------------------------------------------------
# The render cache
# --------------------------------------------------------------------------
#
# Rendering a page is not free — pdfium parses the cross-reference table for the
# document on open, and a 260-page paper at scale 2 is a few hundred
# milliseconds of work — and a reader who clicks three citations on the same
# page should pay for it once.
#
# **What the key is, and what it deliberately is not.** The key is
# ``(digest of the original bytes, page, scale, renderer version)``. It is not
# the document id, and the difference is the whole safety argument:
#
# * A page raster is a *function of the original file and nothing else*. The
#   parse, the chunking, the index version and the evidence that prompted the
#   request have no influence on a single pixel. So keying on the content
#   digest makes it structurally impossible to serve a render that disagrees
#   with the file it claims to depict — a re-parse, a re-index or a new
#   embedding model all leave the key untouched, correctly.
# * Conversely, keying on ``document_id`` would be *wrong* the moment T2.2's
#   re-parse lands, because a document id survives re-parsing while its content
#   does not. The module already has the reasoning written down for the
#   evidence body — "a shared cache answering a later reader with an earlier
#   reader's page would be the same defect as a wrong highlight, arrived at by
#   a different road" — and the same argument applies here with the digest as
#   the answer rather than ``no-store`` as the answer.
# * The digest is of the bytes *actually read*, not of the digest the record
#   claims. A record whose ``sha256`` disagrees with its object would otherwise
#   poison the cache under a key the bytes do not belong to, and the mismatch
#   would then be served to every later reader of that digest. Hashing what was
#   read is one pass over a file already in memory and makes the key
#   self-verifying.
#
# **Why it is bounded.** A personal-tier process is long-lived and a page render
# is a few hundred kilobytes of PNG. An unbounded cache here is not a cache, it
# is a slow memory leak, so both an entry count and a byte budget are enforced
# and eviction is least-recently-used until both are satisfied. The defaults
# (64 entries, 32 MB) are sized for the working set of an actual reading
# session — a handful of pages of one or two papers — and are configurable
# because a deployment with 400-page technical reports wants more than a
# deployment skimming abstracts.
class _RenderCache:
    """A bounded, content-keyed cache of rendered pages.

    A class rather than a bare ``dict`` because the bound is the interesting
    part and a dict cannot express it. The bookkeeping is three integers and an
    ``OrderedDict``; the value is that the bound is enforced in one place, so
    "we bounded it" is a property of the type rather than a promise in a
    comment.
    """

    def __init__(self, max_entries: int, max_bytes: int) -> None:
        self.max_entries = max(0, int(max_entries))
        self.max_bytes = max(0, int(max_bytes))
        self._entries: OrderedDict[tuple[str, int, float, str], tuple[bytes, dict]] = (
            OrderedDict()
        )
        self._bytes = 0
        #: The one lock. pdfium's document and page handles are not documented
        #: as safe to use from two threads at once, and a FastAPI route runs
        #: sync work in a threadpool, so without this two simultaneous clicks
        #: on the same document race inside the library. The lock is held
        #: across the render rather than around the cache so the render is
        #: serialised too — pdfium calls are milliseconds and a cache that let
        #: two renders of the same page proceed would only save work at the
        #: cost of a crash.
        self._lock = threading.Lock()

    def get(self, key: tuple[str, int, float, str]) -> tuple[bytes, dict] | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            self._entries.move_to_end(key)
            return entry

    def get_or_render(
        self, key: tuple[str, int, float, str], render: Any
    ) -> tuple[bytes, dict]:
        """The cached render, or a fresh one, holding the lock throughout.

        This is the only correct way to reach :func:`_render_page`, and the
        reason is the lock's scope rather than the cache. pdfium's document and
        page handles are not documented as safe to use concurrently, and a
        FastAPI route runs its blocking work in a threadpool — so two readers
        clicking citations in the same document at the same moment is the
        *normal* case, not an exotic one. Holding the lock across the render
        serialises pdfium access, which costs a few milliseconds of
        parallelism on a path that is milliseconds long, and buys the
        elimination of a crash that would surface as a 500 on a working
        deployment.

        Double-checking the cache inside the lock is not redundant with the
        caller's earlier lookup: between the two, another thread may have
        rendered the same page. Reusing its result is the entire point of the
        cache, and re-rendering it would make the cache's hit rate depend on
        timing.

        ``render`` is called with no arguments and is expected to raise on
        failure; nothing is cached when it does, so a transient failure cannot
        be served to a later reader.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
                return entry
            payload, geometry = render()
            self._store(key, payload, geometry)
            return payload, geometry

    def _store(
        self, key: tuple[str, int, float, str], payload: bytes, geometry: dict
    ) -> None:
        """Insert and evict. Assumes the lock is held; kept separate so both
        ``put`` and ``get_or_render`` enforce the bound through one path rather
        than two that can drift."""
        if self.max_entries <= 0 or self.max_bytes <= 0 or len(payload) > self.max_bytes:
            # A single image larger than the whole budget is not cached at
            # all. Evicting everything else to make room for it would turn
            # one oversized request into a cache that is useless for every
            # other document, and holding it alone would make the budget a
            # lie.
            return
        if key in self._entries:
            self._bytes -= len(self._entries[key][0])
            self._entries.pop(key)
        self._entries[key] = (payload, geometry)
        self._bytes += len(payload)
        while self._entries and (
            len(self._entries) > self.max_entries or self._bytes > self.max_bytes
        ):
            _, (evicted, _) = self._entries.popitem(last=False)
            self._bytes -= len(evicted)

    def put(self, key: tuple[str, int, float, str], payload: bytes, geometry: dict) -> None:
        with self._lock:
            self._store(key, payload, geometry)

    def clear(self) -> None:
        """Drop everything. Exists for tests and for an operator's ``/api/health``
        style escape hatch; a cache that cannot be emptied is a cache that
        cannot be reasoned about when it misbehaves."""
        with self._lock:
            self._entries.clear()
            self._bytes = 0

    def __len__(self) -> int:
        return len(self._entries)


def _render_cache() -> _RenderCache:
    """The process-wide render cache, (re)built when the bounds change.

    Read with ``getattr`` for the same reason every other module in this
    project reads its settings that way: the configuration block is owned by
    another task, and a route that raised :class:`AttributeError` because a
    setting had not landed would take a citation click down with it. Comparing
    the configured bounds against the live cache's own is what lets a test
    shrink the cache without reaching into it, and what makes an operator's
    lowered ``PAGE_RENDER_CACHE_MAX_BYTES`` take effect without a restart.
    """
    global _RENDER_CACHE
    max_entries = int(getattr(config, "PAGE_RENDER_CACHE_MAX_ENTRIES", 64))
    max_bytes = int(getattr(config, "PAGE_RENDER_CACHE_MAX_BYTES", 32 * 1024 * 1024))
    if _RENDER_CACHE is None or (
        _RENDER_CACHE.max_entries != max_entries or _RENDER_CACHE.max_bytes != max_bytes
    ):
        _RENDER_CACHE = _RenderCache(max_entries, max_bytes)
    return _RENDER_CACHE


_RENDER_CACHE: _RenderCache | None = None


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _encode_png(bitmap: Any) -> bytes:
    """Encode a rendered bitmap as a PNG.

    Pillow rather than a hand-rolled encoder, and the reason is the one that has
    recurred through this module: a subtly malformed image is not a loud
    failure. A browser given a broken PNG shows a broken-image icon, the
    highlight never appears, and the reader concludes the citation points
    nowhere — which is the exact wrong-highlight outcome the route exists to
    prevent, reached by way of a compression detail. ``PIL.Image.save`` is the
    most-exercised PNG writer in the ecosystem and is already a dependency
    through the front-end toolchain, so the alternative is not "no dependency"
    but "a dependency that is easy to get subtly wrong".

    The bitmap is closed explicitly on the way out. pdfium allocates it outside
    Python's heap, so leaving it to the garbage collector means an allocation
    that is freed whenever the collector happens to run rather than when the
    render finishes — and a page viewer's failure mode under memory pressure is
    an OOM kill, not a slowdown.

    Pillow is imported inside the function rather than at module scope so that a
    personal-tier install without the ``render`` extra never reaches this code
    at all — which is what keeps :func:`page_renderer_name`'s probe honest,
    since a module-level ``import PIL`` would raise at import time on exactly
    the deployments the 501 branch exists to serve.
    """
    # Imported for the failure rather than for the name. ``PdfBitmap.to_pil``
    # imports Pillow itself, so without this line a missing or half-installed
    # encoder would surface as an ``AttributeError`` or a 500 from deep inside
    # the rasteriser instead of the ``ImportError`` the handler answers with a
    # 501.
    import PIL.Image  # noqa: F401

    try:
        image = bitmap.to_pil()
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=False, compress_level=6)
        return buffer.getvalue()
    finally:
        close = getattr(bitmap, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # pragma: no cover - free never raises meaningfully
                logger.debug("closing a rendered bitmap failed", exc_info=True)


def _render_page(pdf_bytes: bytes, page: int, scale: float) -> tuple[bytes, dict]:
    """Render one 1-based page of a PDF, returning ``(png, geometry)``.

    The three things this function is careful about, in the order they matter:

    1. **Rotation is neutralised.** :meth:`pypdfium2.PdfPage.set_rotation` is
       called with ``0`` before rendering. pdfium's ``get_size()`` reports the
       *rotated* viewport, so without this the viewport would be the swapped
       pair for a ``/Rotate 90`` page while every ``BoundingBox`` in this
       project stays in unrotated user space, and the highlight would land on a
       page rotated 90° from the text it names. Rendering unrotated costs the
       reader a sideways page on the rare document that asks for one — a
       visible thing, explained by ``geometry.source_rotation`` — and buys a
       client contract with no special case, which is the trade this module
       makes everywhere else too. Verified against a synthetic fixture at all
       four ``/Rotate`` values, in ``tests/test_page_render.py``.

    2. **The bitmap dimensions are the contract, not the requested scale.**
       pdfium's output size is not exactly ``round(page_pt * scale)`` — a
       595.276pt page at scale 1.0 comes back 596px — so the achieved ratio is
       measured and handed to the client instead of the scale that was asked
       for. A client multiplying by the requested scale is off by a fraction of
       a percent, which is invisible and is also wrong.

    3. **The page number is checked against the document, not assumed.** pdfium
       raises on an out-of-range index, and letting it raise would surface as a
       500. The count is read from the open document so the answer is about
       *this file*, not about a ``page_count`` the record may have recorded
       from a different parse of a different revision.
    """
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(pdf_bytes)
    try:
        count = len(document)
        if page < 1 or page > count:
            raise PageOutOfRange(page=page, page_count=count)

        target = document[page - 1]
        source_rotation = int(target.get_rotation() or 0) % 360
        if source_rotation:
            target.set_rotation(0)

        page_width_pt, page_height_pt = target.get_size()
        bitmap = target.render(scale=scale)
        image_width_px = int(bitmap.width)
        image_height_px = int(bitmap.height)
        png = _encode_png(bitmap)

        geometry = page_image_geometry(
            page_width_pt=page_width_pt,
            page_height_pt=page_height_pt,
            image_width_px=image_width_px,
            image_height_px=image_height_px,
            rotation=source_rotation,
            scale=scale,
        )
        geometry["page_count"] = int(count)
        return png, geometry
    finally:
        close = getattr(document, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # pragma: no cover
                logger.debug("closing a rendered document failed", exc_info=True)


class PageOutOfRange(Exception):
    """The requested page is not in the document. Carries both numbers so the
    handler can say how long the document actually is, which is the one piece of
    information that lets a client recover instead of just reporting a failure."""

    def __init__(self, page: int, page_count: int) -> None:
        super().__init__(f"page {page} is not in a {page_count}-page document")
        self.page = page
        self.page_count = page_count


@router.get("/v1/documents/{document_id}/evidence/{evidence_id}")
async def get_evidence(
    document_id: str,
    evidence_id: str,
    stores: StoreBundle = Depends(get_store_bundle),
):
    """Resolve one citation to the passage, locator and view it supports."""
    if not _is_well_formed(document_id) or not _is_well_formed(evidence_id):
        return build_error_response(
            "文档 ID 或证据 ID 格式不合法。", code="malformed_id"
        )

    payload, error_key = _evidence_payload(stores, document_id, evidence_id)
    if error_key:
        status, message, code = _ERRORS[error_key]
        return build_error_response(message, status_code=status, code=code)

    # ``no-store`` because the body is one reader's citation pointing at one
    # document's coordinates; a shared cache answering a later reader with an
    # earlier reader's page would be the same defect as a wrong highlight,
    # arrived at by a different road.
    return JSONResponse(content=payload, headers={"Cache-Control": "no-store"})


#: The headers that carry the geometry on the image response itself.
#:
#: Duplicated in the body of a ``format=json`` request and in HTTP headers,
#: which looks redundant until you remember that the two are for different
#: clients. The headers are what a ``fetch``-based client reads, and putting
#: the geometry there means it can draw the highlight over the ``<img>`` it
#: already has without a second round trip and without a second URL to keep
#: correct. The JSON form is what a client that wants the geometry *before* the
#: image — to size a container, or to place a box while the image is still in
#: flight — reads, and it is the same numbers from the same code path.
GEOMETRY_HEADERS = {
    "scale_x": "X-Page-Scale-X",
    "scale_y": "X-Page-Scale-Y",
    "page_width_pt": "X-Page-Width-Pt",
    "page_height_pt": "X-Page-Height-Pt",
    "image_width_px": "X-Page-Width-Px",
    "image_height_px": "X-Page-Height-Px",
    "source_rotation": "X-Page-Source-Rotation",
}


def _geometry_headers(geometry: dict, etag: str) -> dict[str, str]:
    """The geometry as headers, plus the validators that make re-fetching cheap."""
    headers = {header: str(geometry[field]) for field, header in GEOMETRY_HEADERS.items()}
    headers["ETag"] = etag
    # ``private`` because the bytes are a user's document, and because a shared
    # cache between two users would be an information leak — but not
    # ``no-store`` as the evidence body uses. The difference is the point: the
    # evidence body is a *resolution decision* that differs per reader, whereas
    # a page render is a pure function of content-addressed bytes, so two
    # readers of the same document legitimately get identical bytes and a
    # private cache holding one cannot be wrong. What a cache must never do is
    # answer for a *different* document, which the digest in the ETag rules out.
    headers["Cache-Control"] = "private, max-age=600"
    return headers


@router.get("/v1/documents/{document_id}/pages/{page}/image")
async def get_page_image(
    document_id: str,
    page: int,
    scale: str | None = None,
    format: str = "png",
    stores: StoreBundle = Depends(get_store_bundle),
):
    """The real rendered page, as a PNG, with the geometry to place a box on it.

    **The order of the checks is the design.** Each one is a different fact, and
    a client acts differently on each, so a client must never be told a cheaper
    answer before an earlier, more specific one:

    1. The id is well formed and the page is at least 1 — 400. FastAPI rejects a
       non-integer ``page`` with a 422 before this body runs, which is right: a
       page of zero is a client bug, not a missing renderer.
    2. The document exists — 404. Checked *before* the capability check, and
       that ordering is not incidental: 501 is a statement about this build, and
       answering "the renderer is missing" for a document that is also missing
       would hide a 404 behind a capability answer and send a client looking
       for a dependency instead of a typo in an id.
    3. This build can rasterise — 501. The path existed and answered this before
       a rasteriser did, so the code, the status and the body are the same ones
       a client already handles; what changed is that a deployment which
       installed the extra now takes a different branch.
    4. The original bytes are on this machine — 501, with ``reason:
       "original_unavailable"`` alongside the code. The renderer works, the
       document is there, and the file that would prove which page this is has
       been deleted. A blank PNG here would be the worst outcome available,
       because it is a 200 and it looks like a page.
    5. The document is a PDF — 415, on the bytes rather than the filename.
    6. The page is in the document — 404, with the real page count in the body.

    ``format=json`` returns the geometry *without* the image, for a client that
    needs the viewport to lay out a container before the bytes arrive. It is the
    same numbers the PNG's headers carry, produced by the same code from the
    same cache entry, so a client cannot end up with a scale from one and a
    viewport from the other.

    **Why this route takes no bounding box.** It is worth stating as a refusal
    rather than left unstated, because it is the obvious next feature and the
    obvious wrong turn. A ``?bbox=`` parameter here would be a second, weaker
    copy of :meth:`BoundingBox.to_query`: it would accept a rectangle this
    module has already decided is undrawable, and it would do so behind a URL
    that looks like a *rendering* rather than a *verdict*. A mirrored box is not
    rescuable by being drawn on a real page — the image makes the mistake
    legible, which is worse, not better. Boxes are placed by the client from the
    coordinates this response states, and only for boxes
    ``visual_targeting`` already said were drawable.
    """
    if not _is_well_formed(document_id):
        return build_error_response("文档 ID 格式不合法。", code="malformed_id")
    if page < 1:
        return build_error_response("页码必须从 1 开始。", code="invalid_page")

    record = stores.metadata.get_document(document_id)
    if record is None:
        status, message, code = _ERRORS["document_not_found"]
        return build_error_response(message, status_code=status, code=code)

    if page_renderer_name() == "none":
        return build_error_response(
            PAGE_RENDER_MESSAGE,
            status_code=PAGE_RENDER_STATUS,
            code=PAGE_RENDER_CODE,
        )

    pdf_bytes = _read_original(record)
    if not pdf_bytes:
        payload = build_error_payload(ORIGINAL_UNAVAILABLE_MESSAGE, code=PAGE_RENDER_CODE)
        # Additive, not a replacement: ``code`` stays the one a client already
        # branches on and ``reason`` is the narrower fact. See the note on
        # ORIGINAL_UNAVAILABLE_REASON for why the distinction does not get a
        # code of its own.
        payload["reason"] = ORIGINAL_UNAVAILABLE_REASON
        return JSONResponse(content=payload, status_code=PAGE_RENDER_STATUS)

    if not _is_pdf_bytes(pdf_bytes):
        return build_error_response(
            f"该文档不是 PDF（原始文件以 {pdf_bytes[:8]!r} 开头），无法渲染页图。",
            status_code=UNSUPPORTED_FORMAT_STATUS,
            code=UNSUPPORTED_FORMAT_CODE,
        )

    requested_scale = resolve_page_render_scale(scale)
    # The digest of the bytes actually read, not the digest the record claims.
    # See ``_RenderCache``'s note: the key has to be self-verifying or a
    # mismatched record poisons every later reader of that digest.
    digest = hashlib.sha256(pdf_bytes).hexdigest()
    cache_key = (digest, page, requested_scale, page_renderer_name())

    try:
        png, geometry = _render_cache().get_or_render(
            cache_key, lambda: _render_page(pdf_bytes, page, requested_scale)
        )
    except PageOutOfRange as exc:
        return build_error_response(
            f"该文档共有 {exc.page_count} 页，请求的第 {exc.page} 页不存在。",
            status_code=PAGE_OUT_OF_RANGE_STATUS,
            code=PAGE_OUT_OF_RANGE_CODE,
        )
    except ImportError:
        # The capability probe said the modules are installed and the import
        # still failed — a broken or half-removed install. This is the same
        # situation the probe exists to describe, so it gets the same answer
        # rather than a 500 that reads as a server bug.
        logger.warning("page renderer import failed after a successful probe", exc_info=True)
        return build_error_response(
            PAGE_RENDER_MESSAGE,
            status_code=PAGE_RENDER_STATUS,
            code=PAGE_RENDER_CODE,
        )
    except Exception as exc:
        # An encrypted, truncated or otherwise unreadable PDF. Logged with the
        # exception and answered as a 422: this is the file's problem, and the
        # client can act on it (ask for another file) in a way it cannot act on
        # a 500.
        logger.warning("page render failed: %s", exc, exc_info=True)
        return build_error_response(
            f"该 PDF 无法被渲染（可能已加密或文件损坏）：{exc}",
            status_code=RENDER_FAILED_STATUS,
            code=RENDER_FAILED_CODE,
        )

    # The ETag is over the key, which is over the content, so it is a strong
    # validator by construction: same ETag means the same pixels.
    etag = (
        f'"{digest[:16]}-{page}-{requested_scale:g}'
        f'-{geometry["image_width_px"]}x{geometry["image_height_px"]}"'
    )
    headers = _geometry_headers(geometry, etag)

    if str(format).strip().lower() == "json":
        return JSONResponse(content={"page": page, **geometry}, headers=headers)

    return Response(
        content=png,
        media_type="image/png",
        headers=headers,
    )
