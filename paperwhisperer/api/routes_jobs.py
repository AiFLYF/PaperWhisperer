"""The job resource: ``/v1/jobs``, and the submit surface that drives ``/api``.

T2.3's title is "长任务出 API" and its subject is the API *request*: parse, index
and analyse are long, and a request that performs them is a request that holds a
connection open for a cloud parse. This module is where the work leaves.

**What the two surfaces are.** The read half — ``GET /v1/jobs/{id}``, ``GET
/v1/jobs/{id}/events``, ``GET /v1/jobs/{id}/stream``, ``POST …/cancel``, ``POST
…/retry`` — is T2.4's frozen contract and is unchanged by T2.3. The submit half
— ``POST /api/jobs/analyze`` and ``POST /api/jobs/ask`` — is new, and it is the
only place in the service where a caller hands work to a worker.

**Why a job and not a thread.** Three reasons, and they are the reason the plan
asked for it:

* *The work outlives the connection.* ``/api/analyze`` used to parse, analyse and
  write a report inside one request, and its ``finally`` deleted the upload. A
  user who closed the tab at minute two got nothing, and the model call that had
  been paid for was thrown away. A job has its bytes in the object store and its
  progress in a table, so the tab may go.
* *A restart should not lose it.* The state machine has a lease, so a worker that
  dies mid-job leaves a row another worker reclaims. A thread per request has no
  such property, and cannot be given one cheaply.
* *The API process should not be the queue.* A request that blocks is a request
  that cannot answer a health check. ``POST /api/jobs/analyze`` returns in
  milliseconds with a job id no matter what the document is.

**Why the synchronous routes still work, and what "works" means.** ``/api/analyze``
and ``/api/analyze/stream`` are what the bundled browser calls, and T2.4's
acceptance criterion is that the existing front end switches over without
breaking. So they were not rewritten into 202s: they *submit the same job* and
then **wait on it**, which is the whole design. The work is in a
:class:`~paperwhisperer.infra.jobs.JobWorker` either way, so:

* the analysis runs once, in one place, under the state machine, and its
  provenance is in a job trail rather than in a log line;
* a load test that watches CPU on the API worker sees the *waiting*, not the
  work — the request occupies a coroutine, and the parse and the model calls
  happen on a worker thread;
* and a client that wants to leave is one query parameter away.

That last point is why both modes share a submit function rather than
approximating each other. ``/api/analyze`` and ``POST /api/jobs/analyze`` are
the same call with a different answer, and there is no second pipeline.

**The file survives because it is in the object store, not in ``UPLOAD_FOLDER``.
** This is the part that bites, and T2.2's ``original`` layer already solves it.
The submit path writes the uploaded bytes to the ``original`` layer under
``digest_key(sha256)`` — content-addressed and write-once, exactly as
``routes_v1`` does it for ``POST /v1/documents`` — and the job's *first* move is
to materialise those bytes into a path only that stage knows about. The request's
``finally`` still deletes the staging upload, which is now a copy rather than the
only copy. A job that starts an hour later still has a file to parse, and the
bytes outlive every process that touched them.

**A key is never in the payload.** The payload is a JSON row in a database, so a
browser-supplied API key written there is a key at rest, in a file that gets
backed up. Instead :func:`register_ephemeral_secret` holds it in a process-local
map keyed by job id: a synchronous caller gets exactly the key they sent, and a
worker in another process — or the same one after a restart — falls back to the
environment and says so in the trail. There is no code path on which a key
supplied over HTTP is written to disk.

**The subscription carries state, and nothing else.**
:class:`~paperwhisperer.infra.jobs.JobSubscription`

``GET /v1/jobs/{id}``
    Where the job is now. One object, no paging, no filters.
``GET /v1/jobs/{id}/events?since_sequence=``
    Everything that has happened since a sequence number, as JSON. This is the
    resync for a client that would rather poll, and it is the same data the
    stream sends.
``GET /v1/jobs/{id}/stream``
    The same feed as SSE, with a snapshot first.

**Why the feed carries state and not content.**
:class:`~paperwhisperer.infra.jobs.JobSubscription` projects every transition
through :meth:`~paperwhisperer.infra.jobs.JobEvent.to_state_dict`, which has no
field for the payload, the stage scratch or the result. A stream that carried the
model output would have to be replayed, and a half-consumed stream is exactly the
thing the Phase 2 exit criterion forbids. A stream of numbered states is
replayable: ``Last-Event-ID`` says where to resume, and everything after it is in
the trail. So every state event is written with an SSE ``id:`` line carrying its
sequence, and :meth:`_resolve_since` reads ``Last-Event-ID`` first — that is the
header the browser's own ``EventSource`` sends back on a reconnect, which means
the resync works without the client having to remember to ask for it.

**The snapshot comes first on every connection, including the first.** A client
that arrives late knows where things stand before it starts reading transitions,
and the snapshot deliberately does *not* advance the cursor: it answers "where am
I", not "what happened", and counting it as an event would put every later
sequence off by one for a client that had not seen the submission.

**``DELETE`` does not cancel a job and ``POST …/cancel`` is not an action
dressed as a resource.** A cancel is a *request* whose answer is "maybe" — the
worker honours it at a cancellable boundary, and may finish first. So the
response is a :class:`~paperwhisperer.infra.jobs.CancelAck` with an ``accepted``
flag and a Chinese sentence saying which of the two happened, rather than a 200
that a client would read as "stopped". A sub-resource (``…/cancel``) rather than
``DELETE /v1/jobs/{id}`` because deleting a job is a different thing this surface
deliberately does not do: the trail is the record of what happened, and
``JOBS_RETENTION_DAYS`` owns its lifetime.

**This module also holds two things both ``/v1`` routers need** — the shared
store/job dependencies and the :class:`ErrorResponse` envelope — because it is
the module the document router imports. Declaring the error envelope once is not
tidiness: the three field names are the frozen part of the contract, and two
declarations of them are two chances to disagree.

Everything here is backend-agnostic. The store arrives through
:func:`paperwhisperer.api.routes_evidence.get_store_bundle`, which is the same
process-wide bundle the evidence API reads, and :func:`build_job_store` puts the
``jobs`` table in the same database. That is what makes the Phase 2 exit
criterion "API 多实例运行状态一致" true by construction rather than by
convention: no job state is ever held in a process's memory, so a second instance
reading the same database reports exactly what the first one wrote.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any

from fastapi import APIRouter, Depends, File, Form, Request, Response, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from paperwhisperer.api.deps import parse_json_object_request
from paperwhisperer.api.routes_evidence import _is_well_formed as evidence_id_guard
from paperwhisperer.api.routes_evidence import get_store_bundle
from paperwhisperer.core import config
from paperwhisperer.core.errors import build_error_response, build_sse_event, build_sse_headers
from paperwhisperer.core.text import (
    build_safe_upload_filename,
    build_unique_storage_path,
    close_upload_file_safely,
    redact_secrets,
    remove_file_safely,
)
from paperwhisperer.documents.uploads import save_upload_file
from paperwhisperer.infra.jobs import (
    JOB_STATES,
    STATE_ANALYZING,
    STATE_MESSAGES,
    STATE_PARSING,
    STATE_QUEUED,
    Job,
    JobContext,
    JobPipeline,
    JobStep,
    JobStore,
    JobSubscription,
    JobWorker,
    StepFailed,
    build_job_store,
    passthrough_step,
)
from paperwhisperer.infra.object_store import LAYER_ORIGINAL, ObjectNotFound, ObjectStore
from paperwhisperer.infra.stores import StoreBundle
from paperwhisperer.llm.prompts import normalize_answer_mode
from paperwhisperer.sessions.store import load_validated_session

logger = logging.getLogger(__name__)

router = APIRouter()

#: How often the subscription asks the store for new events while it is idle.
#: A module constant rather than a config key because it is a property of the
#: feed, not of a deployment: it trades store reads against how quickly a client
#: sees a transition it missed. A test shortens it by patching this attribute.
SSE_POLL_INTERVAL_SECONDS = 0.2

#: How long the response body waits for the producer thread to notice the client
#: has gone. Bounded on purpose — the thread is a daemon and exits on its own
#: within one poll interval, and a close that blocks for that is acceptable while
#: a close that blocks for ever is not.
SSE_DRAIN_TIMEOUT_SECONDS = 3.0

#: The most job stores kept alive at once, keyed by the bundle they were built
#: from. One per process in a deployment; more only when a test overrides the
#: bundle dependency, and the bound keeps that from growing without limit.
MAX_CACHED_JOB_STORES = 8

#: How long a synchronous submit waits for its own job before answering 202.
#: Bounded because a request may not be held for ever: past this the caller is
#: told "still running, here is the id" rather than being left on a socket. The
#: job is unaffected — it keeps running, and the same query that produced the 202
#: will report its result.
SYNC_WAIT_TIMEOUT_SECONDS = 900.0

#: How often a waiting submit re-reads its job's state.
#: The same order as the SSE poll: it is a table read, and the state machine is
#: the only thing that can end the wait.
SYNC_POLL_INTERVAL_SECONDS = 0.2

#: How often ``/api/analyze/stream`` re-reads the job it is following.
#: Slightly faster than the SSE subscription's own interval because this loop is
#: how the browser learns that a *section* has landed, and a section arriving
#: half a second late is a visible stutter in the progress counter — whereas the
#: state feed is read by a client that is deliberately further away.
ANALYSIS_STREAM_POLL_SECONDS = 0.1

#: How long a synchronous submit waits for a background worker before running
#: the job itself. Long enough that a healthy worker has certainly claimed
#: the job already — the poll interval is a second — and short enough that a
#: deployment with ``JOBS_ENABLED`` off answers promptly instead of hanging
#: for the full wait timeout.
SYNC_INLINE_FALLBACK_GRACE_SECONDS = 1.5


# --------------------------------------------------------------------------
# Dependencies
# --------------------------------------------------------------------------


#: Job stores, keyed by ``id()`` of the store bundle they were built from.
#:
#: Keyed by identity rather than a single module global because the bundle is a
#: dependency: a test overrides it, and a single global would then hand the
#: override's caller a store wired to the *overridden* bundle's predecessor —
#: which is a write to a closed database, and one that only shows up on the
#: second test that overrides. The bundle is kept in the value so its ``id()``
#: cannot be recycled by a garbage collection while it is still the key.
_JOB_STORES: dict[int, tuple[StoreBundle, JobStore]] = {}


def get_job_store(bundle: StoreBundle = Depends(get_store_bundle)) -> JobStore:
    """The job store for the process's store bundle, built on first call.

    Passing the bundle in is the supported way to share the connection, and it is
    what puts the ``jobs`` table in the same SQLite file — or the same
    PostgreSQL database — as the documents. Building a store per request would
    open a connection pool per poll, which is the wrong price for a read that
    happens every 200 ms.
    """
    key = id(bundle)
    cached = _JOB_STORES.get(key)
    if cached is not None and cached[0] is bundle:
        return cached[1]

    store = build_job_store(bundle=bundle)
    if len(_JOB_STORES) >= MAX_CACHED_JOB_STORES:
        _JOB_STORES.pop(next(iter(_JOB_STORES)))
    _JOB_STORES[key] = (bundle, store)
    return store


# --------------------------------------------------------------------------
# Job kinds and the pipeline that runs them
# --------------------------------------------------------------------------


#: The kind an uploaded document's parse-and-analyse run is submitted under.
#: A name rather than an enum because ``JobStore`` keeps ``kind`` as a string on
#: purpose: the pipeline is the routing key a worker claims by, and a deployment
#: that wants a second kind of work should be able to name it without a release.
KIND_DOCUMENT_ANALYSIS = "document_analysis"

#: The kind a follow-up question is submitted under. Separate from the document
#: kind because it enters the spine at ``analyzing`` rather than at ``parsing``
#: and because a question cannot outlive the session it retrieves from — the
#: session, not the job row, is what makes it re-runnable.
KIND_QA = "qa_answer"

#: API keys a synchronous caller supplied, keyed by job id, and never written to
#: a row.
#:
#: A key that arrives over HTTP is a secret, and the job payload is a JSON row in
#: a database that gets backed up. Writing a browser's key there would turn a
#: per-request convenience into a credential at rest that outlives the session
#: and travels in the backup. So a synchronous submit parks the key here, in
#: process memory, and the stage that needs it takes it and drops it.
#:
#: The honest cost is stated rather than hidden: this map is per process, so a
#: worker in another instance — or this one after a restart — will not find the
#: key. The stage does not fail on that; it falls back to the environment and
#: says in the trail that it did, because "the deployment's own key" and "the
#: caller's key" are different bills and a user who cannot tell which one they
#: paid for is being asked to guess.
_SECRETS: dict[str, str] = {}
_SECRETS_LOCK = threading.Lock()

#: The most parked keys at once. A key is taken and removed by the stage that
#: uses it, so this only has to survive the gap between submit and claim, but the
#: bound is what stops a stream of never-claimed jobs from growing the map without
#: limit. The oldest goes, which loses a key for a job that has not started yet —
#: the same loss the trail then explains.
MAX_PARKED_SECRETS = 64


def register_ephemeral_secret(job_id: str, secret: str) -> None:
    """Hold a caller-supplied API key in memory until its stage claims the job.

    Empty and short values are not parked. A key shorter than eight characters
    is not a key, and the store for a document whose key is a browser default
    would be a map entry that looks like a credential and is not one.
    """
    text = str(secret or "").strip()
    if len(text) < 8:
        return
    with _SECRETS_LOCK:
        if len(_SECRETS) >= MAX_PARKED_SECRETS:
            _SECRETS.pop(next(iter(_SECRETS)))
        _SECRETS[job_id] = text


def take_ephemeral_secret(job_id: str) -> str:
    """Remove and return a parked key. Single-use by construction.

    Called once, by the stage, before the model is constructed. A key left in the
    map after the work that needed it is a key sitting in a heap waiting for
    something to read it, so taking it out is part of using it rather than a
    tidy-up afterwards.
    """
    with _SECRETS_LOCK:
        return _SECRETS.pop(job_id, "")


#: The finished analysis for a job, handed to the caller that is *waiting* for
#: it. Keyed by job id, process-local, single-use.
#:
#: This exists because of one field: the analysis payload carries the session
#: **token**, and a token in a job row is a credential at rest. So the durable
#: record of a job is a pointer — ``result.session_id`` — and the payload itself
#: travels by a route that does not write anything down. A caller in another
#: process does not get it, which is the honest limit, and it is why
#: ``mode=async`` exists: an async caller never needed the payload, only the id.
_RESULTS: dict[str, dict[str, Any]] = {}
_RESULTS_LOCK = threading.Lock()

#: The most parked results at once. A waiting caller takes its result promptly,
#: so this only has to cover the gap between completion and the response being
#: written, and the bound is what stops a client that disconnects mid-wait from
#: leaving a model answer sitting in a heap.
MAX_PARKED_RESULTS = 16


def register_job_result(job_id: str, result: dict[str, Any]) -> None:
    """Park a finished analysis for the caller waiting on this job."""
    with _RESULTS_LOCK:
        if len(_RESULTS) >= MAX_PARKED_RESULTS:
            _RESULTS.pop(next(iter(_RESULTS)))
        _RESULTS[job_id] = dict(result or {})


def take_job_result(job_id: str) -> dict[str, Any]:
    """Remove and return a parked analysis, or ``{}`` if it is not here.

    ``{}`` is a real answer rather than an error: it means the work finished in
    another process, which a caller that asked to wait has no way to distinguish
    from a caller that asked to wait on a job nobody ran. Both are answered by
    falling back to the job's own state, so a waiting client is never left
    holding a null it has to interpret.
    """
    with _RESULTS_LOCK:
        return _RESULTS.pop(job_id, {})


def job_sections(job: Job) -> dict[str, Any]:
    """The analysis sections a running job has published so far.

    Read from the scratch rather than from the trail because a section is a
    document, not a sentence: it does not fit in an event's ``message`` and does
    not belong in the state stream, which is state. The scratch is the job's own
    durable hand-off and it is written by the stage that produced them, so this
    cannot drift from what the analysis actually did.
    """
    return dict(getattr(job, "context", None) or {}).get("sections") or {}


def resolve_stage_api_key(job_id: str, context: JobContext | None = None) -> str:
    """The key a stage should bill to: the caller's if parked, else the deployment's.

    The trail records which one it was, because a personal deployment whose
    operator supplied ``OPENAI_API_KEY`` and a browser that pasted its own are
    spending two different accounts, and "why was I charged" is a question the
    job should already have answered.
    """
    caller_key = take_ephemeral_secret(job_id)
    if caller_key:
        if context is not None:
            context.log("使用请求方提供的 API Key。")
        return caller_key
    resolved = config.resolve_api_key("")
    if context is not None:
        if resolved:
            context.log("未随请求提供 API Key，改用部署环境的 OPENAI_API_KEY。")
        else:
            context.log("请求与部署环境均未提供 API Key。")
    return str(resolved or "")


#: The stages between parsing and analysing, declared and doing nothing.
#:
#: The state machine's spine is seven states and the pipeline's contiguity check
#: is what makes a missing stage a loud error at construction. The alternative,
#: a ``quality_check -> analyzing`` shortcut, would let a pipeline skip a stage
#: nobody noticed and the trail would show a job that went straight from parsing
#: to analysing as though that were the plan.
#:
#: **``normalizing`` and ``indexing`` are empty as spine steps, not because
#: this pipeline never indexes.** The document *is* chunked and indexed — twice
#: over, in fact, once idempotently at the head of
#: :func:`_analyse_stage` so the fact layer has retrieved evidence to extract
#: from, and once at the tail of
#: :func:`paperwhisperer.analysis.service.finalize_analysis_result` by
#: :func:`paperwhisperer.analysis.service.register_analysis_document` so the
#: published ``document_id`` names a document the Evidence API can find. What
#: these two states decline to be is a *separate step*: both sit before
#: ``analyzing`` in the spine, so a reader would wait on them as an empty
#: boundary stage. The cost is the same index either way, paid once per
#: document because both call sites go through the idempotent
#: :func:`~paperwhisperer.analysis.orchestrator.index_document_for_retrieval`.
#:
#: These two reasons used to say the opposite — that "文档检索由 /v1/documents
#: 负责" — while :func:`~paperwhisperer.analysis.service.attach_document_identity`
#: published a ``/v1/documents``-only ``document_id`` to the browser as though
#: this surface had registered it. Two modules disagreed and the reader's
#: citation chip was what found out: after analyze, ask and a click, the store
#: had never heard of the document and every chip answered ``404
#: document_not_found``. The reason strings are data a job trail renders, so a
#: stale one is not a comment problem; it is the product still asserting
#: something untrue about itself.
DOCUMENT_ANALYSIS_PLACEHOLDERS = (
    ("quality_check", "本任务不建立质量门：解析器自身已按质量阈值决定是否升级。"),
    ("fallback", "本任务的解析阶段已包含解析器自身的升级策略。"),
    ("normalizing", "本阶段不单独切分文档：切分随事实层所需的检索索引一并建立。"),
    ("indexing", "本阶段不单独建索引：分析阶段在抽取事实层前建立检索索引，"
                 "分析完成后再登记文档，两者均为幂等操作。"),
)


def _materialise_original(context: JobContext, key: str) -> str:
    """Write the ``original``-layer bytes to a path only this stage knows about.

    Shared by the two stages because both need a real file and neither may rely
    on the other having left one behind. A stage that depended on a previous
    stage's temporary file would lose the job on a requeue: recovery sends the
    job back to ``queued``, the pipeline starts again at ``parsing``, and the
    path the analysing stage was going to read is whatever the previous attempt
    removed. Each stage materialising its own copy costs one file copy and makes
    the second half of the pipeline re-runnable from nothing.

    The copy is removed by the caller's ``finally``, so the staging file's
    lifetime is the stage's lifetime rather than the job's.
    """
    suffix = str(context.job.payload.get("source_suffix") or "")
    staged = build_unique_storage_path(config.UPLOAD_FOLDER, f"job{context.job.job_id}{suffix}")
    try:
        with (
            resolve_object_store().open_stream(LAYER_ORIGINAL, key) as source,
            open(staged, "wb") as target,
        ):
            while True:
                block = source.read(1 << 20)
                if not block:
                    break
                target.write(block)
    except ObjectNotFound as exc:
        remove_file_safely(staged, "job analysis staging file")
        raise StepFailed("原始文件已不在对象存储中，无法继续。", retryable=False) from exc
    return staged


def _parse_stage(context: JobContext) -> None:
    """Parse the document, from the object store, and record what it cost.

    Three things are deliberate here.

    **The bytes come from the ``original`` layer, not from a path in the
    payload.** The submit route has already deleted its staging upload by the time
    this runs, so a path would be a path to nothing. What is in the payload is
    the object store key, and the bytes under it outlive every process that
    touched them — which is what makes the job re-runnable at all.

    **A partial parse is a partial job, not a failed one.**
    :func:`~paperwhisperer.ingestion.orchestrator.parse_document` answers with a
    :class:`~paperwhisperer.ingestion.orchestrator.ParseOutcome` whose ``status``
    is ``partial`` when it produced usable text with something missing. Routing
    that to ``StepFailed`` would report "the document could not be parsed" about a
    document that was parsed well enough to analyse, which is the "partial success
    shown as total loss" the job state machine's ``partial`` terminal state exists
    to prevent. So the warnings become ``mark_partial`` and the job carries them.

    **The parse is heartbeated around, not just after.** A scanned PDF can sit in
    a force-OCR retry for minutes, and a step that is quiet for longer than the
    lease is a step the next worker will reclaim while it is still running. The
    boundary before the stage has already extended the lease once; this extends it
    again for the duration.
    """
    payload = dict(context.job.payload or {})
    key = str(payload.get("original_key") or "")
    if not key:
        raise StepFailed("任务缺少原始文件的对象存储键，无法解析。", retryable=False)

    staged = _materialise_original(context, key)
    try:
        context.heartbeat()
        # Imported here rather than at module scope: ``routes_v1`` imports this
        # module for the error envelope, so a top-level import in the other
        # direction is a cycle. The import happens once per job, not per request.
        from paperwhisperer.ingestion.orchestrator import parse_document

        outcome = parse_document(staged, media_type=str(payload.get("media_type") or "") or None)
        context.heartbeat()
    finally:
        # This stage's copy. The caller's upload was deleted with the request;
        # this one is the stage's alone and goes with the stage.
        remove_file_safely(staged, "job parsing staging file")

    document = outcome.document if outcome is not None else None
    if outcome is None or not outcome.is_usable or document is None:
        raise StepFailed(
            (outcome.error_message if outcome is not None else "") or "文档解析失败。",
            retryable=False,
        )

    context.update(document_id=str(document.document_id))
    context.update(
        result={
            "source_filename": str(payload.get("source_filename") or ""),
            "parser_name": outcome.parsed_by,
            "parse_status": outcome.status,
            "page_count": int(getattr(document, "page_count", 0) or 0),
            "char_count": int(getattr(document, "char_count", 0) or 0),
            "used_cloud": bool(outcome.used_cloud),
            "touched_network": bool(outcome.touched_network),
            "cache_hit": bool(outcome.cache_hit),
        }
    )
    context.log(outcome.summary_line())
    for reason in list(outcome.warnings)[:3]:
        context.mark_partial(reason)


def _analyse_stage(context: JobContext) -> None:
    """Run the model over the parsed document and persist the analysis.

    **The stage writes the *session*, not the analysis, into the job.** That split
    is the reason a reconnecting client can be answered at all: the job's
    ``result`` names the session, the session holds the text, and a client that
    dropped mid-run re-reads the session rather than asking a job row to carry a
    model's prose — which is the same rule that keeps content off the stream and
    out of ``result`` in ``infra.jobs``.

    **This is the fact layer, not the six-section path.** The analysis itself is
    :func:`~paperwhisperer.analysis.service.analyze_with_fact_layer`, which
    extracts the eight structured facts in one call and renders every card from
    them, so this is the only analysis path in the product that produces a
    ``paper_facts`` payload and an ``analysis_run`` record — the provenance the
    export report promises. The legacy six-section generator
    (:meth:`~paperwhisperer.analysis.orchestrator.DocumentAnalyzer.analyze_stream`)
    is still in :mod:`paperwhisperer.analysis.orchestrator` and still callable,
    but nothing routes to it: the measured comparison on the same document was
    roughly two model calls against thirteen and a five-fold wall-clock
    difference, and the gate comment that used to ask for that comparison before
    this swap ("deliberately left as the caller's decision") has now been
    answered, so the decision is this swap.

    **The index is built here, before the analysis, because the fact layer
    refuses to run without evidence.** :func:`analyze_with_fact_layer` takes an
    ``evidence_provider`` over an *already indexed* document and raises without
    one — a fact layer with no citations renders as though every claim in it had
    been verified. So this stage materialises the document, asks
    :func:`~paperwhisperer.analysis.orchestrator.build_qa_retriever` to index
    it (idempotently, and only when it is not already indexed at the current
    index version), and wraps that retriever in
    :func:`~paperwhisperer.analysis.paper_facts.make_evidence_provider`. That
    ordering reverses the latency contract the six-section path was built
    around — there, indexing deliberately trailed the summary so a reader never
    waited on an embedding — but the trade is different now and unavoidable: the
    evidence *is* the input, so the document has to be retrievable before the
    first token. What is paid here is the same index the finaliser would have
    built anyway, moved earlier rather than duplicated, and it is warm on the
    second analysis of the same document.

    **The parse is a cache hit.** Both the pre-index above and the fact layer's
    own load call :func:`~paperwhisperer.ingestion.orchestrator.parse_document`,
    which is the same call this pipeline's parse stage made and whose cache is
    keyed by the file's content digest. The bytes re-materialised here are the
    bytes it saw, so re-reading them here reads the cache, not a parser. This is
    the same "parse is cache hit" fact the old streaming stage documented; only
    the consumer of the parse changed.

    **Per-section streaming is gone, and that is the accepted cost.** The six
    sections no longer arrive one at a time, so there is nothing to heartbeat
    between and the ``section`` scratch is written once, at the end, with the
    finished set. What the reader sees instead is the whole result after a run
    that takes a fraction of the time; the ``section`` event stream still fires
    (once, with the finished sections) so ``/api/analyze/stream`` keeps its
    event vocabulary. The legacy generator is untouched and still streams, for
    the Q&A and comparison paths that want it.
    """
    payload = dict(context.job.payload or {})
    api_key = resolve_stage_api_key(context.job.job_id, context)
    if not api_key:
        raise StepFailed(
            "缺少 API Key，无法生成分析结果。请提供 api_key 或设置 OPENAI_API_KEY。",
            retryable=False,
        )

    key = str(payload.get("original_key") or "")
    if not key:
        raise StepFailed("任务缺少原始文件的对象存储键，无法分析。", retryable=False)

    # Imported here for the same reason as the parser: the analysis package
    # pulls in the store bundle, and building an analyzer at import time would
    # build it before a test has had a chance to override that bundle.
    from paperwhisperer.analysis.orchestrator import build_qa_retriever
    from paperwhisperer.analysis.paper_facts import make_evidence_provider
    from paperwhisperer.analysis.service import analyze_with_fact_layer
    from paperwhisperer.documents.loader import DocumentLoader

    source_filename = str(payload.get("source_filename") or "document")
    mermaid = bool(payload.get("generate_mermaid", True))
    evaluation = bool(payload.get("generate_evaluation", True))
    brief = bool(payload.get("generate_research_brief", True))

    staged = _materialise_original(context, key)
    try:
        context.heartbeat()
        # Load once, here, through the exact loader the analyzer uses, and keep
        # the result. A parse mints a fresh ``document_id`` whenever the parser
        # does not supply a content-derived one, so a second parse of the same
        # bytes is a second document — and indexing one while extracting from
        # the other would produce a run whose every citation addresses a document
        # the store never registered. The single ``loaded`` below is therefore
        # handed to the fact layer rather than re-derived inside it.
        loaded = DocumentLoader.load_with_outcome(staged)
        document = getattr(loaded.outcome, "document", None)
        if document is None:
            raise StepFailed(
                (loaded.outcome.error_message if loaded.outcome is not None else "")
                or "文档解析失败。",
                retryable=False,
            )
        context.heartbeat()

        # Build the retriever over the indexed document and turn it into the
        # per-field evidence provider the fact layer requires. `build_qa_retriever`
        # indexes only what is not already indexed and caches the retriever, so a
        # repeated analysis of the same document pays neither a second embedding
        # nor a second chunking pass.
        retriever = build_qa_retriever(get_store_bundle(), document=document)
        context.heartbeat()
        evidence_provider = make_evidence_provider(retriever, document.document_id)

        # One call, one result: the fact layer extracts, renders every card, and
        # records the run; `finalize_analysis_result` (called inside
        # `analyze_with_fact_layer`) attaches the run/facts, the document id, the
        # parse provenance and the session.
        result = analyze_with_fact_layer(
            staged,
            source_filename,
            api_key,
            mermaid,
            evaluation,
            session_id=str(context.job.session_id or ""),
            generate_research_brief_bool=brief,
            document_id=document.document_id,
            evidence_provider=evidence_provider,
            store=get_store_bundle(),
            loaded_document=loaded,
        )
        context.heartbeat()
    finally:
        remove_file_safely(staged, "job analysis staging file")

    # The sections the fact layer produced, published into the scratch as a
    # finished set. `job_sections` is what `/api/analyze/stream` reads to emit
    # `section` events, so the event vocabulary is preserved — there is simply
    # one publication instead of one per section.
    sections = dict(result.get("sections") or {})
    context.update(context={"sections": dict(sections)})

    # The hand-off for a synchronous caller, and the reason it has to be
    # process-local: the result carries the session *token*, which is exactly the
    # thing that must never become a database row. A caller that is not in this
    # process gets the job and reads the session instead — see
    # :func:`take_job_result`.
    register_job_result(context.job.job_id, result)

    # What happened to the citation surface, in the job's own words. The
    # analysis succeeded either way -- the notice is on the result the caller
    # receives -- but a job row that said only "分析结果已写入会话" would tell
    # an operator reading the trail that everything worked while every citation
    # in that analysis addressed a document the store had never heard of.
    registration = dict(result.get("document_registration") or {})
    indexed_chunks = int(registration.get("chunk_count") or 0)
    context.update(
        result={
            **(dict(context.job.result or {})),
            "session_id": str(result.get("session_id") or ""),
            "section_count": len(sections),
            "document_registration": str(registration.get("status") or ""),
            "indexed_chunks": indexed_chunks,
        }
    )
    if str(registration.get("status") or "") == "registered":
        context.log(
            f"分析结果已写入会话；文档 {registration.get('document_id') or ''} "
            f"已登记并建立索引（{indexed_chunks} 个片段）。"
        )
    else:
        # Not a failure of the job, and deliberately not `StepFailed`: the
        # summary the reader came for exists and is on the result. The trail
        # says which half of the product is missing, in the words
        # `register_analysis_document` shows the reader.
        context.log(f"分析结果已写入会话。{registration.get('notice') or ''}")


def _answer_stage(context: JobContext) -> None:
    """Answer one question and record the turn in its session.

    The same split as the analysis stage: the job names the turn, the session
    holds it. A client whose stream dropped re-reads ``qa_history`` from the
    session it already holds a token for, which is a read rather than a second
    model call.
    """
    payload = dict(context.job.payload or {})
    api_key = resolve_stage_api_key(context.job.job_id, context)
    if not api_key:
        raise StepFailed("缺少 API Key，无法回答问题。请提供 api_key 或设置 OPENAI_API_KEY。",
                         retryable=False)

    from paperwhisperer.analysis.service import record_qa_turn
    from paperwhisperer.api.routes_qa import (
        _attach_turn_evidence as attach_turn_evidence,
    )
    from paperwhisperer.api.routes_qa import as_answer_result, build_qa_analyzer
    from paperwhisperer.sessions.store import (
        get_session_document_content,
        load_validated_session,
        write_session_payload,
    )

    safe_session_id, session_payload = load_validated_session(
        context.job.session_id, payload.get("session_token"), require_token=True
    )
    analyzer = build_qa_analyzer(
        api_key,
        get_session_document_content(session_payload),
        document_id=str(session_payload.get("document_id") or ""),
    )
    result = as_answer_result(
        analyzer.answer_question(
            str(payload.get("question") or ""),
            history=session_payload.get("qa_history", []),
            answer_mode=str(payload.get("answer_mode") or "evidence"),
        )
    )
    context.heartbeat()
    record_qa_turn(session_payload, str(payload.get("question") or ""), result.answer,
                   str(payload.get("answer_mode") or "evidence"))
    attach_turn_evidence(session_payload, result)
    write_session_payload(safe_session_id, session_payload)

    context.update(
        result={
            "session_id": safe_session_id,
            "confidence": str(result.confidence),
            "claim_count": len(result.claims),
            "citation_count": len(result.citations),
            "answer_chars": len(result.answer),
        }
    )
    context.log("回答已写入会话的问答历史。")


def analysis_pipeline() -> JobPipeline:
    """The ``document_analysis`` pipeline: parse, then analyse, with four blanks.

    Every stage is in JOB_SPINE order, so the pipeline is a valid contiguous
    slice and the trail reads as the work actually happened rather than as a
    graph edge nobody traversed.
    """
    return JobPipeline(
        kind=KIND_DOCUMENT_ANALYSIS,
        steps=(
            JobStep(state=STATE_PARSING, run=_parse_stage, required=True, label="正在解析文档。"),
            *(passthrough_step(state, reason) for state, reason in DOCUMENT_ANALYSIS_PLACEHOLDERS),
            JobStep(
                state=STATE_ANALYZING,
                run=_analyse_stage,
                required=True,
                label="正在生成分析结果。",
            ),
        ),
    )


def qa_pipeline() -> JobPipeline:
    """The ``qa_answer`` pipeline: one stage, entered at ``analyzing``.

    A question retrieves from a session that already exists and re-enters the
    spine at the end, because there is nothing to parse and nothing to index. The
    graph allows ``queued → analyzing`` and that edge is the reason this is one
    step rather than a pipeline with four passthroughs in front of it.
    """
    return JobPipeline(
        kind=KIND_QA,
        steps=(
            JobStep(
                state=STATE_ANALYZING,
                run=_answer_stage,
                required=True,
                label="正在回答问题。",
            ),
        ),
    )


#: Every kind this service knows how to run, built once.
#:
#: A dict rather than a chain of ``if kind ==`` in the worker, because the worker
#: claims by kind and a kind it has no pipeline for must be a job it declines
#: rather than a job it runs with somebody else's stages.
PIPELINES: dict[str, JobPipeline] = {
    KIND_DOCUMENT_ANALYSIS: analysis_pipeline(),
    KIND_QA: qa_pipeline(),
}


# --------------------------------------------------------------------------
# The background worker
# --------------------------------------------------------------------------

#: The process's workers, keyed by the job kind each one drains.
#:
#: **An entry is removed only once that worker's thread has actually stopped.**
#: The obvious shape — snapshot the values, ``clear()``, then stop each — forgets
#: a worker before it has finished stopping, and that single reordering is the
#: whole defect. Once the handle is gone, a join that timed out has left a live
#: thread that *no* shutdown path can reach again: not this one, not the next
#: lifespan's, not a test's teardown. The bounded wait silently becomes an
#: unbounded orphan, and the process's only evidence that it happened — an empty
#: registry read as "drained" — is gone with the entry.
_WORKERS: dict[str, JobWorker] = {}
#: Workers displaced from :data:`_WORKERS` that are still running, and are
#: therefore still somebody's problem.
#:
#: They get here when a *new* lifespan starts while a previous worker's thread
#: has not yet finished the step it was in. ``_WORKERS`` is keyed by kind and so
#: can hold only one worker per kind, so a displaced live thread cannot stay in
#: it — but dropping it would be exactly the defect above, one level down, so it
#: moves here instead: still reachable, still re-joinable, and counted by every
#: question a caller can ask about the shutdown.
_STRANDED: list[JobWorker] = []
_WORKER_LOCK = threading.RLock()


def start_job_worker(store: JobStore) -> list[JobWorker]:
    """Start one worker per known kind, and recover the queue first.

    ``recover_stale`` before ``start`` rather than relying on the worker's first
    tick: the tick does it too, but a tick happens after a poll interval, and
    "the queue is stuck until the poll interval elapses" is a worse thing to
    write down than a call that is plainly there. Recovery is one indexed
    statement and it is the release gate's sentence as a function.

    **A worker that is still running is reused, and one that has stopped is
    replaced.** Reuse is the ordinary case — a second lifespan on the same
    application object, or a test that enters a ``TestClient`` twice — and it is
    what keeps two threads from ever claiming the same queue. Replacement is
    equally necessary: a registered worker whose thread has since exited is a
    worker that drains nothing, and handing it back would let a process report a
    started queue while nothing was consuming it. A running worker bound to a
    *different* store is a third case, and it is a real one in a test session
    where a previous fixture's bundle is still being drained; it is moved to
    :data:`_STRANDED` so it stays reachable, and a fresh worker takes its kind.
    """
    started: list[JobWorker] = []
    with _WORKER_LOCK:
        global _WORKERS_STARTED_AT
        _WORKERS_STARTED_AT = time.monotonic()
        for kind, pipeline in PIPELINES.items():
            existing = _WORKERS.get(kind)
            if existing is not None and existing.running:
                if existing.store is store:
                    started.append(existing)
                    continue
                # Registered but still running against a different store: a new
                # lifespan came up while the previous one's step had not
                # finished. The kind slot can only hold one worker, so the live
                # thread moves to `_STRANDED` rather than being overwritten —
                # overwriting it would be the same defect one level down.
                _STRANDED.append(existing)
                logger.warning(
                    "任务种类 %s 的 worker %s 仍在运行，且绑定的存储与本次启动不同；"
                    "已把它移入 _STRANDED（仍可被 stop_job_workers 回收），"
                    "并为新存储另起一个 worker。",
                    kind,
                    existing.worker_id,
                )
            # A registered worker whose thread has already exited is simply
            # replaced: the branch above did not `continue`, so the new worker
            # takes the slot. Reusing it would hand back a worker that drains
            # nothing while the process reported a started queue.
            worker = JobWorker(store, pipeline, settings=config)
            _WORKERS[kind] = worker
            worker.start()
            started.append(worker)
    return started


def stop_job_workers(timeout: float = 5.0) -> list[JobWorker]:
    """Stop every worker this process started. Idempotent.

    Called from the lifespan teardown, and idempotent because a lifespan that
    raises during startup still runs its ``finally`` and a shutdown path that
    stops twice is not a bug. A worker that is mid-parse when it is stopped leaves
    the job leased, and the next process recovers it — which is the same path a
    crash takes, so a clean shutdown and a kill are the same thing to the
    database, which is the property that makes the recovery test meaningful.

    **A worker is only forgotten once it has really stopped.** The join is
    attempted first and the entry is removed afterwards, so a stop whose budget
    expired leaves a live, *reachable* handle: the next shutdown call — another
    lifespan teardown, a test's fixture, a later test module in the same session
    — re-joins it with a fresh budget and collects it. That is what makes a
    failed shutdown recoverable rather than terminal, and it is why the registry
    is worth having at all: it is the answer to "did shutdown actually happen?".

    Those that do not stop are logged at ERROR here rather than at DEBUG, because
    the caller is the lifespan teardown, which is about to report the process as
    stopped and cannot be expected to second-guess it. :func:`stuck_workers` is
    the programmatic form of the same question, and it is the only honest way
    for a caller to distinguish this shutdown from a clean one.

    Returns every worker this call asked to stop — the ones that stopped *and*
    the ones that did not — because a caller counting "how many workers did this
    process have" wants the attempt set. Which of them actually stopped is
    answered by :func:`stuck_workers`, not by the length of the list.
    """
    with _WORKER_LOCK:
        pending = list(_WORKERS.items())
    stuck: list[JobWorker] = []
    for kind, worker in pending:
        if worker.stop(timeout=timeout):
            with _WORKER_LOCK:
                # Only drop the entry if it is still *this* worker. A concurrent
                # `start_job_worker` may have replaced a worker that had already
                # exited; removing by key alone would delete the live one.
                if _WORKERS.get(kind) is worker:
                    del _WORKERS[kind]
        else:
            stuck.append(worker)
    for worker in stranded_workers():
        if not worker.stop(timeout=timeout):
            stuck.append(worker)
        else:
            with _WORKER_LOCK:
                if worker in _STRANDED:
                    _STRANDED.remove(worker)
    if stuck:
        logger.error(
            "有 %d 个任务 worker 未能在 %.1fs 内停止（%s）；它们仍留在 worker 注册表中，"
            "可被下一次 stop_job_workers() 再次回收。正在执行的任务会保留租约，"
            "并由下一次启动的恢复流程接管。",
            len(stuck),
            float(timeout),
            "、".join(sorted(worker.worker_id for worker in stuck)),
        )
    return [worker for _kind, worker in pending]


def stranded_workers() -> list[JobWorker]:
    """Workers displaced from the registry that are still running.

    Public because "is there a live thread nobody owns" is a question about the
    process, not about one worker's kind, and because a test asserting on
    teardown has to be able to see one to assert that it is gone.
    """
    with _WORKER_LOCK:
        return list(_STRANDED)


def stuck_workers() -> list[JobWorker]:
    """The workers whose threads are still running after a shutdown.

    The question a clean stop and a failed stop used to answer identically. A
    caller that wants to *be sure* the process drained — a health check, a test
    asserting teardown, an operator reading a log — asks this, and an empty list
    is now evidence rather than a tautology. It is a read, not a second stop:
    nothing here joins, clears, or reorders anything.
    """
    with _WORKER_LOCK:
        return [
            worker
            for worker in list(_WORKERS.values()) + list(_STRANDED)
            if worker.running
        ]


def active_workers() -> list[JobWorker]:
    """The workers currently running. For a shutdown log line and for tests.

    A worker that could not be stopped is still running, and is still counted:
    reporting the process drained while a thread is draining the queue is the
    symptom this registry was reordered to remove. Stranded workers are included
    for the same reason — a displaced live thread is more certainly doing work
    than an idle registered one.
    """
    with _WORKER_LOCK:
        return list(_WORKERS.values()) + list(_STRANDED)


def resolve_object_store() -> ObjectStore:
    """The object store, deferred so this module's import graph stays acyclic.

    ``routes_v1`` imports this module for the error envelope and imports nothing
    from ``object_store`` that this needs at import time, but a top-level import
    of ``routes_v1`` here would close the loop. Resolving per call is cheap: the
    store is a process singleton behind that function.

    This is the accessor the *stages* use, because a stage runs on a worker
    thread with no request and therefore no dependency to resolve. The routes use
    :func:`get_object_store` instead, which is a ``Depends`` and can be
    overridden; both bottom out in the same process singleton in a real
    deployment, which is what makes them the same store rather than merely two
    stores that happen to agree.
    """
    from paperwhisperer.api.routes_v1 import get_object_store

    return get_object_store()


def get_object_store() -> ObjectStore:
    """The object store as a dependency, so a test can point it at a temp dir.

    A route cannot call :func:`resolve_object_store` directly: it would then
    build a store against the developer's real ``context/objects`` while a test
    asserted against a temporary one, and the failure would be a write to
    somebody's data directory rather than a visible error. As a ``Depends`` it is
    overridable, and the stages — which cannot be overridden that way — are
    pointed at the same place by the fixture overriding
    :func:`resolve_object_store`.
    """
    return resolve_object_store()


# --------------------------------------------------------------------------
# Submitting work: the helpers both surfaces share
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Submitting work
# --------------------------------------------------------------------------


class SubmitResponse(BaseModel):
    """What a submission hands back, in both modes.

    ``job_id`` is the one field a client must be able to rely on, and it is here
    in every mode including the synchronous one — so a caller that learns to save
    it is not learning something that stops working when the work finishes faster.
    ``deduplicated`` says this submission matched an ``idempotency_key`` that had
    already been used, which is the difference between "your job is running" and
    "the job you already asked for is running".
    """

    job_id: str
    kind: str
    state: str
    deduplicated: bool
    job: JobState
    #: Where to watch. A path rather than a URL because this module does not know
    #: what prefix the application is mounted under, and a client that guesses the
    #: prefix is the one thing a hard-coded base URL would break.
    status_path: str = ""
    stream_path: str = ""


def submit_paths(job_id: str) -> dict[str, str]:
    """The two paths a freshly submitted job is watched through."""
    return {"status_path": f"/v1/jobs/{job_id}", "stream_path": f"/v1/jobs/{job_id}/stream"}


def submit_response(submitted: Any, job: Job) -> SubmitResponse:
    """Assemble the wire form, taking the state from the row rather than the result.

    ``SubmitResult`` carries the job as it was *submitted* — ``queued`` — while
    ``job`` is the row after the worker has already had a go at it. Reading the
    state off the submitted copy would report a job as queued that is in fact
    running, so the fresher row wins.
    """
    return SubmitResponse(
        job_id=str(submitted.job_id),
        kind=str(job.kind),
        state=str(job.state),
        deduplicated=bool(getattr(submitted, "deduplicated", False)),
        job=job_state(job),
        **submit_paths(str(submitted.job_id)),
    )


def wait_for_job(
    store: JobStore,
    job_id: str,
    *,
    timeout: float = SYNC_WAIT_TIMEOUT_SECONDS,
    poll_interval: float = SYNC_POLL_INTERVAL_SECONDS,
) -> Job:
    """Block on a job's state, on this thread, until it settles or time runs out.

    Deliberately a poll of the store rather than a wait on a condition variable.
    The state machine is in a database and the work is on another thread; a
    caller that waited on an in-process event would be waiting on *this* process's
    memory, which is exactly the coupling the Phase 2 exit criterion "API 多实例
    运行状态一致" forbids — the same answer would come back only while the process
    that submitted the job was still alive. A poll answers from the one place both
    instances can see.

    ``asyncio.sleep`` rather than ``time.sleep`` would be wrong here: this runs on
    a worker thread via ``to_thread``, and blocking the event loop from a thread
    that exists to avoid blocking the event loop would be a strange way to keep
    the API responsive.
    """
    deadline = time.monotonic() + max(0.0, float(timeout))
    while True:
        job = store.get_job(job_id)
        if job is not None and job.terminal:
            return job
        if time.monotonic() >= deadline:
            # Out of patience, not out of work. The job keeps going and the
            # client keeps the id, so the answer to a request that gave up is
            # the same answer an async submit would have given in the first
            # place: here is the job, go and watch it.
            return job if job is not None else store.get_job(job_id)
        time.sleep(max(0.01, float(poll_interval)))


def run_job_if_idle(store: JobStore, pipeline: JobPipeline, job_id: str) -> bool:
    """Claim and run a still-queued job on a thread. ``True`` if it was claimed.

    The safety net under every caller that waits on a job rather than serving it.
    A synchronous submit cannot assume a background worker exists — ``JOBS_ENABLED``
    is documented as the switch back to synchronous handling, and a process can be
    started without its lifespan — and a caller that merely waited would hang until
    its timeout in both cases.

    Claiming is a compare-and-set on ``claim_next``, so this is not a race with
    the background worker: if it got there first this claims nothing and returns
    ``False``, and the work runs exactly once. The one thing it must never do is
    *skip* the wait, so it is only ever called with the job still ``queued``.
    """
    job = store.get_job(job_id)
    if job is None or job.state != STATE_QUEUED:
        return False
    worker = JobWorker(store, pipeline, lease_seconds=60.0)
    return worker.run_once() is not None


def should_run_inline(grace: float = 0.0) -> bool:
    """Whether a waiting caller may claim the job itself, and after how long.

    **Immediately, when this process has no background worker.** That is the whole
    point of the check: with nothing else able to claim the job, waiting first is
    not politeness towards another worker, it is the request sitting idle for a
    grace period that can only ever expire. ``JOBS_ENABLED=false`` and a process
    started without its lifespan both land here, and both deserve an answer now.

    **After the grace period, when one is running.** A live worker polls on a
    second; giving it that long to make the claim is what keeps the two from
    racing, and it is also what stops a queue of submissions from having every one
    of them attempt a claim the worker is about to make anyway.
    """
    if not active_workers():
        return True
    return time.monotonic() - _WORKERS_STARTED_AT >= max(0.0, float(grace))


#: When the workers were started, for :func:`should_run_inline`. A process with
#: workers always pays the grace; a process without them never does.
_WORKERS_STARTED_AT = time.monotonic()


async def wait_for_job_async(
    store: JobStore,
    job_id: str,
    *,
    timeout: float | None = None,
    poll_interval: float | None = None,
    pipeline: JobPipeline | None = None,
    grace: float | None = None,
) -> Job:
    """The awaiting submit, yielding the event loop rather than occupying it.

    This is what the acceptance criterion "负载测试下 API worker 不被长任务占用"
    actually measures. A request that called :func:`wait_for_job` directly would
    pin one of the process's worker threads for the length of a document, and a
    thread pool with a handful of threads is a service that stops answering
    health checks under load — occupied by *waiting*, which is the distinction
    that matters. Here the coroutine sleeps, the parse and the model calls happen
    on a :class:`~paperwhisperer.infra.jobs.JobWorker` thread, and every other
    request in the process proceeds.

    **The inline fallback is not a second pipeline; it is the same one, run by
    the caller when nobody else will.** A synchronous submit that assumes a
    background worker exists is a synchronous submit that hangs forever in the two
    deployments where it does not: ``JOBS_ENABLED=false``, which is documented as
    the way back to synchronous handling, and a process started without its
    lifespan. So after ``grace`` seconds with the job still ``queued``, this claims
    and runs it through :meth:`~paperwhisperer.infra.jobs.JobWorker.run_once` on
    a thread.

    That is safe against the background worker rather than a race with it, because
    ``claim_next`` is a compare-and-set: if the background worker got there first
    this attempt claims nothing and the loop simply keeps waiting. Two workers
    cannot run the same job, which is the property the lease exists to guarantee.
    """
    # Resolved from the module attributes *inside* the function rather than as
    # default arguments. A default is evaluated once, when the ``def`` runs, so a
    # test that patched ``SYNC_WAIT_TIMEOUT_SECONDS`` to shorten a wait would
    # patch a name this function no longer reads — and then wait the full fifteen
    # minutes, which is how a fast test becomes a hung one.
    timeout = SYNC_WAIT_TIMEOUT_SECONDS if timeout is None else timeout
    poll_interval = SYNC_POLL_INTERVAL_SECONDS if poll_interval is None else poll_interval
    grace = SYNC_INLINE_FALLBACK_GRACE_SECONDS if grace is None else grace
    deadline = time.monotonic() + max(0.0, float(timeout))
    tried_fallback = False

    while True:
        job = store.get_job(job_id)
        if job is not None and job.terminal:
            return job

        if (
            pipeline is not None
            and not tried_fallback
            and job is not None
            and job.state == STATE_QUEUED
            and should_run_inline(grace)
        ):
            tried_fallback = True
            await asyncio.to_thread(run_job_if_idle, store, pipeline, job_id)

        if time.monotonic() >= deadline:
            return job if job is not None else store.get_job(job_id)
        await asyncio.sleep(max(0.01, float(poll_interval)))


#: How the caller asks for its work. ``async`` is the mode T2.3 exists to enable;
#: ``sync`` is the one the bundled browser needs and is the default, because a
#: default that broke the front end would be a default that broke the product.
#:
#: Read from a form field or a query parameter and normalised here, so a caller
#: writing ``mode=async``, ``mode=ASYNC`` or ``async=true`` all get the same
#: answer rather than a silent 422 that looks like the job system is broken.
ASYNC_ALIASES = frozenset({"async", "background", "submit", "job", "true", "1", "yes"})
SYNC_ALIASES = frozenset({"sync", "synchronous", "wait", "false", "0", "no", ""})


def wants_async(value: str) -> bool:
    """Whether a request asked to be answered now rather than when the work ends.

    Anything unrecognised is a *synchronous* request, not a rejection. The
    conservative reading is the one that cannot surprise: a client that sent a
    mode this build does not know still gets the analysis, which is what it got
    before this parameter existed, and a client that genuinely wanted to leave
    gets a job id it can watch — never an error page.
    """
    return str(value or "").strip().lower() in ASYNC_ALIASES


def store_upload_as_original(
    objects: ObjectStore,
    file_path: str,
    content_type: str = "",
) -> tuple[str, bool]:
    """Write an upload into ``original`` and return ``(key, newly_written)``.

    Delegated to :func:`paperwhisperer.api.routes_v1.store_original` rather than
    restated, because that function exists to get one thing right — naming the
    key from the content digest so ``find_by_digest`` can find it again — and a
    second implementation would be a second answer to a question the v1 surface has
    already settled.

    The boolean is ``not deduplicated``, i.e. "these bytes were not already here".
    It is what lets a submit say the upload was stored rather than the truth, which
    is that two identical uploads share one object.
    """
    from paperwhisperer.api.routes_v1 import store_original

    reference = store_original(objects, file_path, content_type)
    return str(reference.key), not bool(reference.deduplicated)


def analysis_idempotency_key(
    original_key: str,
    session_id: str,
    *,
    generate_mermaid: bool,
    generate_evaluation: bool,
    generate_research_brief: bool,
) -> str:
    """The key that makes a repeated submission deduplicate, derived not random.

    Composed of the *content digest* and the *options*, and of nothing else. A
    timestamp would deduplicate nothing while looking as though it did, which is
    the failure ``JobStore.submit``'s own docstring warns about by refusing to
    generate one.

    Two uploads of the same bytes with the same switches and the same session are
    the same request sent twice — a double-clicked button, a retried fetch, a
    client that reconnected before it had read the first response — and answering
    both with one job is the difference between one parse and two. Two *different*
    documents, or the same document analysed into a *different* session, are
    different requests, and the key says so.

    The session id is in the key on purpose. Without it, analysing the same
    paper twice into two different sessions would deduplicate into one, and the
    second reader would find the first reader's session where their own should
    be.
    """
    parts = [
        KIND_DOCUMENT_ANALYSIS,
        str(original_key or ""),
        str(session_id or ""),
        f"mermaid={int(bool(generate_mermaid))}",
        f"evaluation={int(bool(generate_evaluation))}",
        f"brief={int(bool(generate_research_brief))}",
    ]
    return "|".join(parts)


def submit_analysis_job(
    store: JobStore,
    *,
    original_key: str,
    original_filename: str,
    content_type: str = "",
    api_key: str = "",
    session_id: str = "",
    generate_mermaid: str | None = None,
    generate_evaluation: str | None = None,
    generate_research_brief: str | None = None,
) -> tuple[Any, Job]:
    """Write the one row that means "somebody should do this analysis".

    Shared by ``POST /api/analyze``, ``POST /api/analyze/stream`` and
    ``POST /api/jobs/analyze`` so that all three describe the same work
    identically. Three copies of a payload builder is three chances to disagree
    about what a job *is*, and the disagreement would not surface as an error —
    only as a job that cannot be deduplicated against its own twin.

    The payload carries no file bytes, no staging path and no API key: a key
    name, some booleans, a filename and the object store key. The path is
    excluded on purpose because the submit route deletes its upload in a
    ``finally`` before the job is claimed, so a path in the payload would be a
    path to nothing by the time anyone read it.

    ``idempotency_key`` is passed rather than left empty, which is the one
    deliberate exception to "a key nobody chose is a key nobody reasoned about":
    this key is composed from what the request actually asked for.
    """
    from paperwhisperer.api.routes_v1 import original_suffix

    mermaid = config.parse_bool_value(generate_mermaid, default=True)
    evaluation = config.parse_bool_value(generate_evaluation, default=True)
    brief = config.parse_bool_value(generate_research_brief, default=True)

    submitted = store.submit(
        KIND_DOCUMENT_ANALYSIS,
        {
            "original_key": str(original_key or ""),
            "source_filename": str(original_filename or ""),
            "source_suffix": original_suffix(original_filename),
            "media_type": str(content_type or ""),
            "generate_mermaid": mermaid,
            "generate_evaluation": evaluation,
            "generate_research_brief": brief,
        },
        idempotency_key=analysis_idempotency_key(
            original_key,
            session_id,
            generate_mermaid=mermaid,
            generate_evaluation=evaluation,
            generate_research_brief=brief,
        ),
        session_id=str(session_id or ""),
        message=STATE_MESSAGES[STATE_QUEUED],
    )
    # Parked, not persisted. See ``_SECRETS``: a key in the payload would be a
    # key in a backed-up database.
    register_ephemeral_secret(str(submitted.job_id), api_key)
    job = store.get_job(str(submitted.job_id)) or submitted.job
    return submitted, job


# --------------------------------------------------------------------------
# The error envelope, shared by every /v1 router
# --------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    """Every failure on ``/v1``, in three fields.

    ``error`` is Chinese prose written for a person and may be reworded; ``code``
    is the API and is frozen. A client branches on ``code`` and never on
    ``error``, which is why the two are separate keys rather than one string a
    client has to pattern-match.
    """

    error: str = Field(description="面向用户的中文说明。措辞可能变化，不要用于程序判断。")
    code: str = Field(description="稳定的机器可读错误码。客户端只应依赖这一个字段。")
    timestamp: str = Field(description="服务端生成该错误的时间（ISO 8601，秒精度）。")


def error_responses(*specs: tuple[int, str]) -> dict[int | str, dict[str, Any]]:
    """Build a FastAPI ``responses=`` mapping from ``(status, description)`` pairs.

    Every operation on this surface declares its failures here rather than
    leaving them to the default 422 FastAPI invents for a malformed body. The
    reason to bother: a document that a client believes exists, an operation the
    deployment's privacy policy forbids, and a typo in an id are three different
    client problems, and an OpenAPI document that describes none of them is a
    document nobody can build a client against.
    """
    return {
        status: {"model": ErrorResponse, "description": description}
        for status, description in specs
    }


#: The ``/v1/jobs`` error table. Written as data, at one place, so the code a
#: client branches on and the sentence a human reads cannot drift apart.
JOB_ERRORS: dict[str, tuple[int, str, str]] = {
    "malformed_id": (400, "任务 ID 格式不合法。", "malformed_id"),
    "invalid_since": (400, "since_sequence 必须是不小于 0 的整数。", "invalid_since"),
    "invalid_state_filter": (400, "未知的任务状态过滤值。", "invalid_state_filter"),
    "job_not_found": (404, "任务不存在。", "job_not_found"),
    "job_not_retryable": (409, "只有失败且可重试的任务才能重新入队。", "job_not_retryable"),
    "job_store_unavailable": (503, "任务存储不可用。", "job_store_unavailable"),
}


def job_error_response(key: str):
    """Build the response for one row of :data:`JOB_ERRORS`.

    A lookup rather than three arguments, so a caller cannot raise "not found"
    with the status of "not retryable" — and so the code and the sentence a
    client reads stay written down together.
    """
    status, message, code = JOB_ERRORS[key]
    return build_error_response(message, status_code=status, code=code)


# --------------------------------------------------------------------------
# The frozen shapes
# --------------------------------------------------------------------------


class JobState(BaseModel):
    """One job, as this surface reports it.

    Deliberately a superset of
    :meth:`~paperwhisperer.infra.jobs.Job.to_state_dict` — it adds
    ``document_id``, ``session_id``, ``partial_reasons``, ``error_message``,
    ``started_at``, ``finished_at`` and ``result`` — because a *polling* client
    needs those and a poll is a snapshot, not a replay. The stream does not send
    them: :class:`JobEventState` is a different, smaller shape, and the
    difference is the contract. A reconnecting subscriber rebuilds the truth from
    the state events plus one ``GET``; a subscriber cannot rebuild the result
    from the stream, and that is by design, because the result is fetched from
    here and not replayed.
    """

    job_id: str
    kind: str
    state: str
    message: str
    terminal: bool
    partial: bool
    partial_reasons: list[str] = Field(default_factory=list)
    retryable: bool
    cancel_requested: bool
    attempts: int
    max_attempts: int
    #: Who holds the lease and until when. Present because
    #: :meth:`~paperwhisperer.infra.jobs.Job.to_state_dict` projects them: a
    #: job whose worker died keeps saying ``parsing`` until its lease deadline
    #: passes, so without these two fields a polling client cannot tell it from
    #: one a healthy worker is actively parsing. Both are absolute values
    #: already on the row rather than anything derived from the server clock,
    #: so two polls of an unchanged job stay byte-identical.
    lease_owner: str = ""
    lease_expires_at: float = 0.0
    sequence: int
    document_id: str = ""
    session_id: str = ""
    error_message: str = ""
    started_at: str = ""
    finished_at: str = ""
    created_at: str = ""
    updated_at: str = ""
    #: The stage hand-off summary. Never the document text and never model
    #: output: ``jobs.Job`` says as much about this field, and the reason is the
    #: same one that keeps content off the stream.
    result: dict[str, Any] = Field(default_factory=dict)


class JobEventState(BaseModel):
    """One transition, as the stream and the event feed send it.

    The exact fields of
    :meth:`~paperwhisperer.infra.jobs.JobEvent.to_state_dict`, and the reason it
    is spelled out here is that these names are the frozen part of the resync
    contract. ``sequence`` is the cursor; ``state`` is where the job is now;
    ``previous_state`` and ``changed`` are what it moved from, so a client that
    was away can say what it missed without re-deriving anything.
    """

    job_id: str
    sequence: int
    state: str
    previous_state: str
    changed: bool
    message: str
    note: str
    terminal: bool
    attempt: int
    at: str


class JobEventFeed(BaseModel):
    """The resync answer: everything since a sequence number, and where we are."""

    job_id: str
    since: int
    sequence: int
    events: list[JobEventState]


class JobList(BaseModel):
    """A page of jobs, plus the queue's shape.

    ``by_state`` is here so that "is anything stuck?" is one cheap request rather
    than a page of jobs to count, which matters when the question is asked
    because something looks wrong.
    """

    items: list[JobState]
    count: int
    limit: int
    offset: int
    by_state: dict[str, int] = Field(default_factory=dict)


class CancelAckResponse(BaseModel):
    """The honest answer to "did you cancel it".

    ``accepted`` is ``False`` both for a job that had already finished and for
    one whose current step has not reached a cancellable boundary. The message
    says which, because the user has to know whether the work they were paying
    for is still going.
    """

    job_id: str
    state: str
    cancel_requested: bool
    accepted: bool
    message: str
    job: JobState


# --------------------------------------------------------------------------
# Projections
# --------------------------------------------------------------------------


def job_state(job: Job) -> dict[str, Any]:
    """The wire form of one job.

    Built from the job's own state projection plus the six fields a poll needs
    and a stream does not, so the two shapes cannot be confused for one another.
    Key order is the model's field order, which keeps the JSON byte-identical
    between two calls on the same job — which is what makes a
    "did anything change?" comparison on the raw body meaningful.
    """
    state = dict(job.to_state_dict())
    state.update(
        {
            "partial_reasons": list(job.partial_reasons),
            "document_id": str(job.document_id),
            "session_id": str(job.session_id),
            "error_message": str(job.error_message),
            "started_at": str(job.started_at),
            "finished_at": str(job.finished_at),
            "result": dict(job.result),
        }
    )
    return state


def _is_well_formed(value: str) -> bool:
    """Whether a job id could name anything at all.

    Delegated to the evidence router's guard rather than restated. The rule is
    about which strings are allowed to become store keys, and a second copy of it
    would be a second opinion about exactly that — the two would drift, and the
    drift would only surface as a query against a key built from a path, on an id
    that one router accepted and the other did not.
    """
    return evidence_id_guard(value)


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


@router.get(
    "/v1/jobs",
    summary="列出任务",
    tags=["jobs"],
    response_model=JobList,
    responses=error_responses(
        (400, "状态过滤值不是已知状态。"),
        (503, "任务存储不可用。"),
    ),
)
async def list_jobs(
    state: str = "",
    kind: str = "",
    document_id: str = "",
    limit: int = 50,
    offset: int = 0,
    store: JobStore = Depends(get_job_store),
):
    """List jobs newest first, with the queue's per-state counts alongside.

    ``state`` is validated against the state machine's own table rather than
    accepted freely, because a filter that matches nothing and a filter that
    matches no such state produce identical empty lists — and "no such state" is
    the client's bug while "nothing queued" is the answer.
    """
    wanted = str(state or "").strip().lower()
    if wanted and wanted not in JOB_STATES:
        return job_error_response("invalid_state_filter")

    window = config.clamp_int_value(limit, 50, min_value=1, max_value=200)
    try:
        jobs = store.list_jobs(
            state=wanted,
            kind=str(kind or "").strip(),
            document_id=str(document_id or "").strip(),
            limit=window,
            offset=max(0, int(offset)),
        )
        counts = store.count_by_state()
    except Exception:  # noqa: BLE001 - a store outage is a 503, not a 500
        logger.exception("Job listing failed")
        return job_error_response("job_store_unavailable")

    return {
        "items": [job_state(job) for job in jobs],
        "count": len(jobs),
        "limit": window,
        "offset": max(0, int(offset)),
        "by_state": {str(key): int(value) for key, value in (counts or {}).items()},
    }


@router.get(
    "/v1/jobs/{job_id}",
    summary="读取任务状态",
    tags=["jobs"],
    response_model=JobState,
    responses=error_responses(
        (400, "任务 ID 格式不合法。"),
        (404, "任务不存在。"),
        (503, "任务存储不可用。"),
    ),
)
async def get_job(job_id: str, store: JobStore = Depends(get_job_store)):
    """One job's current state — the answer a reconnecting client asks first.

    State, and the summary the stages left behind. Never the payload, the scratch
    or the document: those are the work, and this surface reports work.
    """
    if not _is_well_formed(job_id):
        return job_error_response("malformed_id")

    try:
        job = store.get_job(job_id)
    except Exception:  # noqa: BLE001 - a store outage is a 503, not a 500
        logger.exception("Job lookup failed for %s", job_id)
        return job_error_response("job_store_unavailable")

    if job is None:
        return job_error_response("job_not_found")
    return job_state(job)


@router.get(
    "/v1/jobs/{job_id}/events",
    summary="按序号补齐任务事件",
    tags=["jobs"],
    response_model=JobEventFeed,
    responses=error_responses(
        (400, "任务 ID 格式不合法，或 since_sequence 不是不小于 0 的整数。"),
        (404, "任务不存在。"),
        (503, "任务存储不可用。"),
    ),
)
async def get_job_events(
    job_id: str,
    since_sequence: int = 0,
    store: JobStore = Depends(get_job_store),
):
    """Every transition after ``since_sequence``, oldest first.

    The polling half of the resync contract, and the answer a client gets when it
    would rather not hold a stream open. Two consecutive calls partition the
    trail: no gap, no duplicate, and the returned ``sequence`` is the cursor to
    send next time. Passing ``since_sequence=0`` replays the whole trail, which is
    what a client attaching to a job that has been running for a while needs.
    """
    if not _is_well_formed(job_id):
        return job_error_response("malformed_id")
    if int(since_sequence) < 0:
        return job_error_response("invalid_since")

    try:
        subscription = JobSubscription(store, job_id, since_sequence=int(since_sequence))
        if subscription.snapshot() is None:
            return job_error_response("job_not_found")
        events = subscription.poll()
    except Exception:  # noqa: BLE001 - a store outage is a 503, not a 500
        logger.exception("Job event feed failed for %s", job_id)
        return job_error_response("job_store_unavailable")

    return {
        "job_id": job_id,
        "since": int(since_sequence),
        "sequence": int(subscription.cursor),
        "events": events,
    }


def _resolve_since(header_value: str | None, query_value: int) -> int:
    """Where a stream should resume from.

    ``Last-Event-ID`` wins over the query parameter, because that is the header
    the browser's own ``EventSource`` sends back on a reconnect and honouring it
    is what makes the resync work without the client doing anything clever. The
    query parameter exists for clients that are not ``EventSource`` — a test, a
    server-side poller, a ``curl`` a human typed.

    A non-numeric header is the client's bug and is answered as one. Guessing
    (treating it as 0 and replaying everything) would be the friendlier reading
    and the wrong one: a client whose cursor silently rewinds believes it has
    seen a transition it has not, and the bug then lives in the client.
    """
    raw = str(header_value or "").strip()
    if not raw:
        return max(0, int(query_value))
    try:
        return max(0, int(raw))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Last-Event-ID 必须是整数：{raw!r}") from exc


def _state_event(payload: dict[str, Any]) -> str:
    """Encode one state transition, with its sequence as the SSE event id.

    The ``id:`` line is what makes the stream resumable. Without it the browser
    cannot remember where it stopped, and the whole resync contract would depend
    on the client tracking a number that only appears inside a JSON body.
    """
    return f"id: {int(payload.get('sequence') or 0)}\n{build_sse_event('state', payload)}"


@router.get(
    "/v1/jobs/{job_id}/stream",
    summary="订阅任务状态（SSE）",
    tags=["jobs"],
    response_class=StreamingResponse,
    responses={
        **error_responses(
            (400, "任务 ID 格式不合法，或 Last-Event-ID 不是不小于 0 的整数。"),
            (404, "任务不存在。"),
            (503, "任务存储不可用。"),
        ),
        200: {
            "content": {"text/event-stream": {}},
            "description": (
                "事件流。第一个事件名为 `snapshot`（任务当前状态），其后每个事件名为 "
                "`state`（一次状态迁移，带 `id:` 序号行）。断线后带 "
                "`Last-Event-ID` 重连即可从断点续传。终态事件之后流结束。"
            ),
        },
    },
)
async def stream_job_events(
    job_id: str,
    request: Request,
    since_sequence: int = 0,
    store: JobStore = Depends(get_job_store),
):
    """Stream one job's state, resumably.

    The stream is written here rather than through
    :func:`~paperwhisperer.api.deps.pump_sync_events` for one reason: that helper
    waits unconditionally for its producer when the consumer goes away, and this
    feed's producer is an open-ended poll that will never end on its own. So the
    same shape is used — a blocking generator on a worker thread, an async queue
    between — with two differences that the disconnection requires: the consumer
    checks whether the client is still there while the feed is idle, and the
    producer is handed a ``stop_event`` so it stops polling rather than being
    waited for.
    """
    if not _is_well_formed(job_id):
        return job_error_response("malformed_id")
    try:
        since = _resolve_since(request.headers.get("last-event-id"), since_sequence)
    except ValueError:
        return job_error_response("invalid_since")

    stop_event = threading.Event()
    subscription = JobSubscription(
        store,
        job_id,
        since_sequence=since,
        poll_interval=SSE_POLL_INTERVAL_SECONDS,
        stop_event=stop_event,
    )
    try:
        snapshot = subscription.snapshot()
    except Exception:  # noqa: BLE001 - a store outage is a 503, not a 500
        logger.exception("Job snapshot failed for %s", job_id)
        return job_error_response("job_store_unavailable")

    if snapshot is None:
        return job_error_response("job_not_found")

    async def event_stream():
        # The snapshot goes out before anything is read, and before the producer
        # thread starts: a client that arrives late should not have to wait out a
        # poll interval to learn where the job is.
        yield build_sse_event("snapshot", snapshot)

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        sentinel = object()

        def produce() -> None:
            try:
                for event in subscription.stream():
                    loop.call_soon_threadsafe(queue.put_nowait, event)
            except BaseException as exc:  # noqa: BLE001 - forwarded to the consumer
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        worker = loop.run_in_executor(None, produce)
        try:
            while True:
                try:
                    item = await asyncio.wait_for(
                        queue.get(), timeout=SSE_POLL_INTERVAL_SECONDS
                    )
                except asyncio.TimeoutError:
                    # Nothing happened. This is the only moment the connection can
                    # be checked without an event to piggyback on, and it is where
                    # a client that closed its tab is noticed.
                    if await request.is_disconnected():
                        return
                    continue
                if item is sentinel:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield _state_event(item)
                if item.get("terminal"):
                    return
        finally:
            # Set before the wait, not after: the producer only checks the event
            # between polls, so setting it first is what makes the wait short.
            stop_event.set()
            try:
                await asyncio.wait_for(worker, timeout=SSE_DRAIN_TIMEOUT_SECONDS)
            except Exception:  # noqa: BLE001 - teardown must not raise
                logger.debug("Job stream producer outlived the client; it will stop.")

    return StreamingResponse(
        event_stream(), media_type="text/event-stream", headers=build_sse_headers()
    )


@router.post(
    "/v1/jobs/{job_id}/cancel",
    summary="请求取消任务",
    tags=["jobs"],
    response_model=CancelAckResponse,
    responses=error_responses(
        (400, "任务 ID 格式不合法。"),
        (404, "任务不存在。"),
        (503, "任务存储不可用。"),
    ),
)
async def cancel_job(job_id: str, store: JobStore = Depends(get_job_store)):
    """Ask a job to stop, and say honestly whether that will do anything.

    Idempotent, on purpose: cancelling twice is not a mistake, it is a client
    retrying a request whose response it never saw. The store settles a job that
    has not started from the queue rather than claiming it first, so "I
    cancelled it" is true immediately for a job that had not begun.
    """
    if not _is_well_formed(job_id):
        return job_error_response("malformed_id")

    try:
        if store.get_job(job_id) is None:
            return job_error_response("job_not_found")
        ack = store.request_cancel(job_id)
    except Exception:  # noqa: BLE001 - a store outage is a 503, not a 500
        logger.exception("Job cancel failed for %s", job_id)
        return job_error_response("job_store_unavailable")

    return {**ack.to_dict(), "job": job_state(ack.job)}


@router.post(
    "/v1/jobs/{job_id}/retry",
    summary="重新入队失败任务",
    tags=["jobs"],
    response_model=JobState,
    responses=error_responses(
        (400, "任务 ID 格式不合法。"),
        (404, "任务不存在。"),
        (409, "任务当前状态不可重试。"),
        (503, "任务存储不可用。"),
    ),
)
async def retry_job(job_id: str, store: JobStore = Depends(get_job_store)):
    """Put a failed job back on the queue.

    The only caller of the ``failed → queued`` edge, which is why it is here and
    not inside the worker: retrying is a decision somebody makes after reading
    why the job failed, and a worker that quietly retried its own failures would
    make that decision invisible. A job that did not fail cannot be retried, and
    says so with ``job_not_retryable`` rather than re-running finished work.
    """
    if not _is_well_formed(job_id):
        return job_error_response("malformed_id")

    try:
        job = store.get_job(job_id)
        if job is None:
            return job_error_response("job_not_found")
        if job.state != "failed":
            return job_error_response("job_not_retryable")
        return job_state(store.retry(job_id))
    except Exception:  # noqa: BLE001 - a store outage is a 503, not a 500
        logger.exception("Job retry failed for %s", job_id)
        return job_error_response("job_store_unavailable")


# --------------------------------------------------------------------------
# Submitting work: the routes
# --------------------------------------------------------------------------

@router.post(
    "/api/jobs/analyze",
    summary="提交解析与分析任务",
    tags=["jobs"],
    status_code=202,
    response_model=SubmitResponse,
    responses=error_responses(
        (400, "请求不合法：缺少文件、文件类型不支持，或会话令牌无效。"),
        (404, "会话不存在。"),
        (500, "上传过程中发生内部错误。"),
        (503, "任务存储不可用。"),
    ),
)
async def submit_analysis(
    response: Response,
    file: UploadFile | None = File(None),
    api_key: str = Form(""),
    generate_mermaid: str | None = Form(None),
    generate_evaluation: str | None = Form(None),
    generate_research_brief: str | None = Form(None),
    session_id: str = Form(""),
    session_token: str = Form(""),
    mode: str = Form("async"),
    store: JobStore = Depends(get_job_store),
    objects: ObjectStore = Depends(get_object_store),
):
    """Submit an uploaded document for parse and analysis, and return a job id.

    This is the endpoint T2.3 adds: it answers in milliseconds with the id of a
    job somebody else will do, and it is the one a load test should be pointed at.
    Nothing here parses, indexes or calls a model — the request's whole job is to
    validate the upload, put its bytes somewhere that outlives it, and write one
    row.

    ``POST /api/analyze`` and this route are the same submission. The difference
    is only what happens after the row is written: that route waits for the job,
    this one returns. Sharing :func:`submit_analysis_job` is what keeps them from
    drifting into two pipelines, which is the failure mode a "just this once"
    duplicate of a submit path always eventually reaches.
    """
    # Deferred: ``routes_v1`` imports this module for the error envelope and the
    # upload contract, so the import has to point one way only. ``validate_upload``
    # is the *versioned* surface's check on purpose — the codes it returns
    # (``missing_file``, ``unsupported_file_type``) are ones the front end already
    # branches on, and reusing it is what keeps the two routers from drifting into
    # different messages under the same code.
    from paperwhisperer.api.routes_v1 import validate_upload

    invalid = validate_upload(file)
    if invalid is not None:
        return invalid

    safe_session_id = ""
    if str(session_id or "").strip():
        try:
            safe_session_id, _payload = load_validated_session(
                session_id, session_token, require_token=True
            )
        except PermissionError as exc:
            return build_error_response(
                redact_secrets(str(exc)), status_code=403, code="session_token_invalid"
            )
        except ValueError as exc:
            return build_error_response(
                redact_secrets(str(exc)), status_code=404, code="session_not_found"
            )

    file_path: str | None = None
    try:
        original_filename = build_safe_upload_filename(file.filename)
        file_path = build_unique_storage_path(config.UPLOAD_FOLDER, original_filename)
        await save_upload_file(file, file_path, config.MAX_UPLOAD_BYTES)

        # The bytes go to the object store *before* the job is submitted, and
        # the job is submitted before this request returns. That ordering is the
        # whole of "the file survives the request": the ``finally`` below deletes
        # the staging upload, and what survives is the object-store copy the job
        # is about to read.
        original_key, _newly_written = store_upload_as_original(
            objects, file_path, file.content_type or ""
        )
    except ValueError as exc:
        return build_error_response(redact_secrets(str(exc)), code="invalid_request")
    except Exception:  # noqa: BLE001 - surfaced as a 500, never a stack trace
        logger.exception("Storing the uploaded original failed")
        return build_error_response(
            "上传文件保存失败。", status_code=500, code="document_upload_failed"
        )
    finally:
        await close_upload_file_safely(file, "analysis job upload file")
        remove_file_safely(file_path, "analysis job upload file")

    try:
        submitted, job = submit_analysis_job(
            store,
            original_key=original_key,
            original_filename=original_filename,
            content_type=file.content_type or "",
            api_key=api_key,
            session_id=safe_session_id,
            generate_mermaid=generate_mermaid,
            generate_evaluation=generate_evaluation,
            generate_research_brief=generate_research_brief,
        )
    except Exception:  # noqa: BLE001 - a store outage is a 503, not a 500
        logger.exception("Submitting the analysis job failed")
        return job_error_response("job_store_unavailable")

    if wants_async(mode):
        return submit_response(submitted, job)

    # Synchronous mode. The job is already submitted and already running; the
    # request now waits on the state machine rather than doing the work itself,
    # and the event loop stays free for everything else in the process.
    settled = await wait_for_job_async(
        store, str(submitted.job_id), pipeline=PIPELINES[KIND_DOCUMENT_ANALYSIS]
    )
    response.status_code = 200 if settled.terminal and settled.state != "failed" else 202
    return submit_response(submitted, settled)


@router.post(
    "/api/jobs/ask",
    summary="提交问答任务",
    tags=["jobs"],
    status_code=202,
    response_model=SubmitResponse,
    responses=error_responses(
        (400, "请求不合法：缺少问题或会话令牌无效。"),
        (403, "会话令牌无效。"),
        (404, "会话不存在。"),
        (503, "任务存储不可用。"),
    ),
)
async def submit_question(
    response: Response,
    request: Request,
    mode: str = "",
    store: JobStore = Depends(get_job_store),
):
    """Submit a follow-up question, and return a job id.

    The question is not a document, so nothing is uploaded and the object store
    is not involved: the session the question retrieves from already exists and is
    addressed by the id and token in the body.

    ``mode=async`` is the point of the route. The streaming question the browser
    uses carries the answer as text deltas, and a client that drops mid-answer
    loses the half it had not received — a browser refresh during a long answer
    throws the work away. Here the answer outlives the connection because the job
    is durable and the answer is written to the session's ``qa_history``, so a
    client that comes back reads the turn it already paid for instead of asking
    the model again.
    """
    data, error_response = await parse_json_object_request(request)
    if error_response is not None:
        return error_response

    question = str(data.get("question") or "").strip()
    if not question:
        return build_error_response("Please enter a question.", code="missing_question")

    try:
        safe_session_id, _payload = load_validated_session(
            data.get("session_id"), data.get("session_token"), require_token=True
        )
    except PermissionError as exc:
        return build_error_response(
            redact_secrets(str(exc)), status_code=403, code="invalid_session_token"
        )
    except ValueError as exc:
        return build_error_response(
            redact_secrets(str(exc)), status_code=404, code="session_not_found"
        )

    try:
        submitted = store.submit(
            KIND_QA,
            {
                "question": question,
                "answer_mode": normalize_answer_mode(data.get("answer_mode")),
                "session_token": str(data.get("session_token") or ""),
            },
            session_id=safe_session_id,
            message=STATE_MESSAGES[STATE_QUEUED],
        )
    except Exception:  # noqa: BLE001 - a store outage is a 503, not a 500
        logger.exception("Submitting the question job failed")
        return job_error_response("job_store_unavailable")

    register_ephemeral_secret(str(submitted.job_id), str(data.get("api_key") or ""))

    if wants_async(mode):
        return submit_response(submitted, submitted.job)

    settled = await wait_for_job_async(
        store, str(submitted.job_id), pipeline=PIPELINES[KIND_QA]
    )
    response.status_code = 200 if settled.terminal and settled.state != "failed" else 202
    return submit_response(submitted, settled)
