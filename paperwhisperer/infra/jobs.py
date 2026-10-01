"""Durable jobs: one table, one state machine, one in-process worker.

T2.1's job is to take the long work — parse, quality gate, cloud fallback,
normalize, index, analyze — out of the request that triggers it, and to make
that safe for a single-user deployment. Decision D2 fixes the shape: the
personal tier runs an in-process worker against a job table in the same
database as everything else, with no Docker and no broker. Celery plus Redis
is P4 work for a multi-instance deployment, and building it now would mean
shipping a queue nobody needs and a second place for a job to be.

**The job table lives in the store tier, not beside it.** The obvious
alternative is a new module with its own connection and its own ``jobs.db``.
That is wrong twice: T2.6 asks for a backup of *one* data directory, and a
document row and the job that produced it would stop being writable in one
transaction. So :func:`build_job_store` takes the
:class:`~paperwhisperer.infra.stores.factory.StoreBundle` that is already open
and puts the table in the same file (or the same server, on the Hosted tier).
T2.2's table list already names ``jobs``; this module owns that one table and
its DDL, and T2.2 owns everything else.

**The state machine is the deliverable, so it is enforced, not documented.**
:data:`JOB_TRANSITIONS` is the whole graph, and every write that changes a
job's state goes through :func:`assert_transition`. A handler cannot return a
job to ``queued`` after it completed, a cancelled job cannot resume, and a
pipeline that skips a stage in the middle of the spine is rejected when it is
built rather than when it misbehaves in production. Both tiers run the same
Python table and the same assertion, and ``tests/test_jobs.py`` runs one
contract body against both — the arrangement T1.1 established, because a state
machine checked on one engine is a state machine with one set of rules.

The spine is ``queued → parsing → quality_check → fallback → normalizing →
indexing → analyzing`` and every terminal outcome — ``completed``,
``partial``, ``failed``, ``cancelled`` — is reachable from every spine state,
because a pipeline is a *prefix* of the spine and its last stage goes straight
to an outcome. That is what lets a re-index pipeline be ``indexing`` alone
and a document pipeline be six stages, without the graph growing a special case
for either.

The plan writes the fallback stage as ``fallback?``, and the question mark is
the interesting part. It could mean the *edge* is optional — and then the
graph would need a ``quality_check → normalizing`` shortcut, which is exactly
the kind of hole that lets a buggy handler skip the quality gate and look
fine. So the question mark means the *stage* is conditional: the stage is
always entered, and it decides internally whether to escalate, recording why
in the trail. A parse that came back clean walks through ``fallback`` having
done nothing, and the trail says so. The alternative — a graph with a hole in
it — makes "did this document get a quality check?" answerable only by reading
the handler.

**Recovery is the acceptance criterion, so it is a lease, not a hope.** Every
running job carries ``lease_owner`` and ``lease_expires_at``. The worker
refreshes the lease at each cancellable boundary; a worker that dies stops
refreshing, and the next worker's :func:`recover_stale_jobs` pass either
requeues the job or, if it has used up its attempts, moves it to ``failed``
with ``retryable`` set. A job stuck in ``running`` forever — the failure this
task exists to prevent — is unreachable, because the only way out of a running
state is a transition the worker makes, and a dead worker makes none.

**Cancellation is cooperative and the module says so.** ``cancel_requested`` is
a flag, not a signal: the worker checks it at boundaries. A boundary check
cannot interrupt work in progress, so a step sitting in a long cloud parse or
a long LLM call runs to the end of that call, and the job may even reach
``completed`` if the cancel lands during its final step. Pretending otherwise
would be the worse failure: a user who is told "cancelled" while the bill for
that cloud parse keeps arriving has been lied to. The honest API is
:class:`CancelAck`, which says whether the request was accepted, and
``accepted`` is ``False`` for a job that had already finished.

**A subscription carries state, and nothing else.** :class:`JobSubscription`
projects each transition through :meth:`JobEvent.to_state_dict`, which has no
field for the payload, the stage scratch or the result. The reason is
recovery, not bandwidth: a browser that reconnects mid-run must be able to
reconstruct the truth from the job's state, so the state has to be complete on
its own. Pushing document text or model output down a stream nobody can
replay produces exactly the failure the plan's exit criterion names — a
reconnected client that missed three transitions and cannot tell.

**Two things this module deliberately does not know about.** It does not import
:mod:`paperwhisperer.ingestion` and it does not build a document pipeline. The
stages are named here because the state machine has to name them, but the work
in them belongs to
:func:`~paperwhisperer.ingestion.orchestrator.parse_document` (with its
idempotent cache and its privacy gate) and to T2.3's handler, which calls it.
A job system that reimplemented the parse chain would own a second copy of
the escalation policy, and the copy would be the one nobody reads.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

from .stores.factory import (
    StoreBundle,
    build_stores,
    resolve_backend,
)
from .stores.interfaces import (
    BACKEND_POSTGRES,
    TIER_HOSTED,
    TIER_PERSONAL,
    StoreCapabilities,
    StoreCapable,
)
from .stores.postgres_backend import SORT_KEY, PostgresDatabase
from .stores.sqlite_backend import SqliteDatabase

logger = logging.getLogger(__name__)


def _observability() -> Any:
    """The observability facade, imported late and defensively.

    Imported inside the call rather than at module scope because
    :mod:`paperwhisperer.core.observability` is the process-wide facade whose
    *default* is that collection is off, and a module-level import here would
    make ``infra.jobs`` — imported by the store tier, the API and the CLI —
    the thing that decides whether instrumentation is on. The
    ``ImportError`` guard exists for the same reason the ``record_*`` helpers
    check ``enabled`` first: instrumentation that can raise on the error path is
    worse than no instrumentation, because it turns a queue that was working
    into a queue that raises on every transition.
    """
    try:
        from paperwhisperer.core import observability
    except Exception:  # noqa: BLE001 - never fail a job write over telemetry
        return None
    return observability


def _record_transition(updated: Job, *, event: JobEvent | None = None) -> None:
    """Count one committed state change on the ``jobs`` metric family.

    Called *after* the row is committed, never before. A counter that
    increments on an attempt and not on a commit is a counter that reports
    work that did not happen — which is the one thing a recovery metric must
    not do, because ``jobs.recovered_total`` is read as "how many workers died
    while holding work". Counting a transition that then lost its
    compare-and-set would report a crash recovery that was rolled back, and an
    operator paging on that number would page on a number that is not real.

    ``note`` is what distinguishes a recovery from an ordinary step: the
    helpers in :mod:`paperwhisperer.core.observability` read
    ``note.startswith("lease_expired")`` and count that on its own counter. So
    the notes are the contract, not the prose.

    Best-effort by construction: every ``record_*`` helper returns
    immediately when collection is off, so the cost of an uninstrumented
    deployment is one attribute lookup, and a telemetry failure can never take
    the queue down with it.
    """
    facade = _observability()
    if facade is None:
        return
    try:
        facade.record_job_transition(updated, event=event)
        if updated.terminal:
            started = _iso_epoch(updated.started_at)
            finished = _iso_epoch(updated.finished_at or updated.updated_at)
            if started is not None and finished is not None:
                facade.record_job_duration(
                    updated, duration_ms=max(0.0, (finished - started) * 1000.0)
                )
    except Exception:  # noqa: BLE001 - telemetry must never fail a job write
        logger.debug("记录任务指标失败，已忽略。", exc_info=True)


def _record_submitted(result: SubmitResult) -> None:
    """Count one submission — fresh or deduplicated — on the metrics family.

    Takes the :class:`SubmitResult` rather than the :class:`Job` because the
    ``deduplicated`` flag is on the wrapper and not on the job, and it is the
    flag that decides the metric's label. Same best-effort contract as
    :func:`_record_transition`: it is called after the row is committed and it
    can never raise into the caller's submit.
    """
    facade = _observability()
    if facade is None:
        return
    try:
        facade.record_job_submitted(result)
    except Exception:  # noqa: BLE001 - telemetry must never fail a submit
        logger.debug("记录任务提交指标失败，已忽略。", exc_info=True)


def _record_queue_depth(counts: Mapping[str, int]) -> None:
    """Overwrite the queue-depth gauges from a :meth:`JobStore.count_by_state`.

    On every count rather than on an interval, because the gauge is a *set* of
    a census rather than an accumulation, and a queue that only updates its
    depth when somebody happens to look is a queue whose depth is wrong exactly
    when it matters. The helper is a no-op while collection is off.
    """
    facade = _observability()
    if facade is None:
        return
    try:
        facade.record_job_queue(counts)
    except Exception:  # noqa: BLE001 - telemetry must never fail a read
        logger.debug("记录任务队列深度失败，已忽略。", exc_info=True)


def _retention_cutoff(retention_days: int | None, now: float | None) -> str:
    """The ISO stamp a terminal job must predate to be swept.

    ``now`` defaults to the wall clock and ``retention_days`` to
    ``JOBS_RETENTION_DAYS``. A negative retention is clamped to zero rather
    than rejected: "sweep everything already finished" is a legitimate thing
    for an operator to ask for on a personal machine, and raising here would
    make that a configuration error instead of a choice.

    The comparison happens against ``finished_at``, falling back to
    ``updated_at`` for a terminal row with no finish stamp. Both are written by
    :func:`_now_iso` as fixed-width UTC to the second, so a string comparison
    orders them chronologically — the same reasoning as the lease columns
    being epoch floats, applied to the stamps instead.

    The one consequence of second resolution worth stating: at
    ``retention_days=0`` the cutoff is the current second, and a job that
    finished *within* that second is not yet older than it, so it survives
    until the clock ticks over. That is the honest direction to be wrong in —
    a job kept a moment longer than configured costs a row, while a job deleted
    a moment early costs a result the user may still poll for.
    """
    days = int(
        retention_days
        if retention_days is not None
        else REQUIRED_CONFIG_KEYS["JOBS_RETENTION_DAYS"]
    )
    moment = time.time() if now is None else float(now)
    stamp = datetime.fromtimestamp(moment - max(0, days) * 86400.0, timezone.utc)
    return stamp.isoformat(timespec="seconds")


def _terminal_ids_before(rows: Sequence[Any], cutoff: str) -> list[str]:
    """Which of ``rows`` finished before ``cutoff``, by job id.

    Shared by both tiers because the deadline cannot be expressed in SQL —
    ``finished_at`` lives inside the payload column on each of them — and
    "the personal tier filters in Python and the Hosted tier filters in SQL" is
    exactly the drift this repository keeps getting bitten by. One function, two
    translations of it.

    ``finished_at`` is preferred over ``updated_at`` because a job is retained
    for its *result*, and the result stopped changing the moment it finished.
    Keying on ``updated_at`` alone would let a terminal row that something
    keeps touching outlive the window forever, which is the leak the sweep
    exists to close. A row with neither stamp is kept rather than deleted on a
    guess: an unknown age is not evidence of an old one.

    The payload is coerced the same way :meth:`PostgresJobStore._job` does,
    and for the same reason: the personal tier's column holds TEXT so
    ``json.loads`` is needed, while the Hosted tier's is ``jsonb`` and the
    driver has already handed back a ``dict``. Assuming either one shape is how
    a sweep that works perfectly on SQLite raises ``TypeError`` on the first
    Hosted sweep, fourteen days into a deployment nobody was watching.
    """
    doomed: list[str] = []
    for row in rows:
        payload = row[0]
        if isinstance(payload, (str, bytes)):
            payload = json.loads(payload)
        job = Job.from_dict(payload or {})
        stamp = str(job.finished_at or job.updated_at)
        if stamp and stamp < cutoff:
            doomed.append(job.job_id)
    return doomed


def _iso_epoch(value: str) -> float | None:
    """Parse one of this module's ISO stamps to epoch seconds, or ``None``.

    ``None`` rather than a zero for a stamp it cannot read: a duration
    computed against a zero epoch is a number of decades, and it would land in
    the histogram as the slowest job the deployment has ever run. An unreadable
    stamp means "not known", and a metric that skips an unknown is more useful
    than one that invents a value for it.
    """
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def _now_iso() -> str:
    """UTC, second resolution, same shape the store backends write.

    UTC rather than the local time :func:`paperwhisperer.core.text.now_iso`
    returns, because a job row and a document row are compared and sorted
    against each other, and the local-time convention is one that only makes
    sense for a session file read by the process that wrote it. Lease
    deadlines are not timestamps at all — they are epoch floats, compared with
    ``<`` in SQL on both tiers, because "expired" is a question about a
    duration and not about a calendar.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _dumps(payload: Any) -> str:
    """Serialize a job row, refusing values JSON cannot represent.

    ``allow_nan=False`` is the whole body of this function, and it is here
    because of a divergence rather than out of caution. JSON has no ``NaN``,
    no ``Infinity`` and no ``-Infinity``; they are a Python extension that
    ``json.dumps`` emits by default. SQLite does not care — it stores the
    text ``{"v":NaN}`` and hands it back, and Python's own reader accepts it
    again, so on the personal tier a payload carrying a non-finite float
    round-trips through what is nominally a JSON column and looks fine.
    PostgreSQL's ``jsonb`` is a real parser and refuses it outright:
    ``invalid input syntax for type json / DETAIL: Token "NaN" is invalid``.

    So the two tiers disagreed about whether a job carrying one could exist
    at all, and the personal tier was the wrong one: what it stored is not
    JSON, so it cannot be read by a backup tool, an ``awk`` pipeline, or the
    Hosted tier a user migrates to, and the value has no meaning to begin
    with — a NaN is not a score. Refusing it on both tiers, with the same
    error, is the fix; silently coercing it to a string or to null would
    hand back a number the caller never sent.

    Both tiers call this one function, so the two answers cannot drift.
    """
    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


# --------------------------------------------------------------------------
# The state machine
# --------------------------------------------------------------------------

#: The queued state. The only state a client can observe that is not leased.
STATE_QUEUED = "queued"
#: Parse the uploaded file through the ingestion orchestrator.
STATE_PARSING = "parsing"
#: Grade the parse and decide whether the fallback stage needs to escalate.
STATE_QUALITY_CHECK = "quality_check"
#: The optional escalation: local force-OCR repair, then cloud fallback under
#: the privacy gate. Always entered, sometimes inert — see the module docstring.
STATE_FALLBACK = "fallback"
#: Flatten the parsed document into chunks and facts.
STATE_NORMALIZING = "normalizing"
#: Write vectors and lexical entries.
STATE_INDEXING = "indexing"
#: Run the LLM analysis.
STATE_ANALYZING = "analyzing"

#: Everything worked.
STATE_COMPLETED = "completed"
#: Something worked and something did not. A real, common outcome: a document
#: that parsed with quality warnings, or an analysis where one section failed.
#: Collapsing it into ``failed`` is how partial success gets reported to a user
#: as a total loss, which is the one thing they cannot recover from.
STATE_PARTIAL = "partial"
#: Nothing usable was produced.
STATE_FAILED = "failed"
#: The user asked to stop and the worker honoured it at a boundary.
STATE_CANCELLED = "cancelled"

#: The spine, in order. A pipeline is a contiguous slice of this tuple that
#: starts at :data:`STATE_QUEUED` or later.
JOB_SPINE: tuple[str, ...] = (
    STATE_QUEUED,
    STATE_PARSING,
    STATE_QUALITY_CHECK,
    STATE_FALLBACK,
    STATE_NORMALIZING,
    STATE_INDEXING,
    STATE_ANALYZING,
)

#: The four ways a job ends.
TERMINAL_STATES: frozenset[str] = frozenset(
    {STATE_COMPLETED, STATE_PARTIAL, STATE_FAILED, STATE_CANCELLED}
)
#: Every state, for ``by_state`` reports and validation.
JOB_STATES: tuple[str, ...] = JOB_SPINE + (
    STATE_COMPLETED,
    STATE_PARTIAL,
    STATE_FAILED,
    STATE_CANCELLED,
)
#: The states a job can only be *taken out of* by a worker or by recovery.
RUNNING_STATES: frozenset[str] = frozenset(JOB_SPINE) - {STATE_QUEUED}

#: The graph. Every running state reaches every terminal state, because a
#: pipeline is a prefix of the spine and its last stage goes straight to an
#: outcome; and every running state reaches :data:`STATE_QUEUED` *only* through
#: recovery, which is why that edge is a separate entry in
#: :data:`RECOVERY_EDGES` rather than an ordinary one.
_OUTCOMES = frozenset(TERMINAL_STATES)
JOB_TRANSITIONS: dict[str, frozenset[str]] = {
    # A job may enter the spine at its first stage, which is how a re-index
    # pipeline starts at `indexing` instead of running the four stages before
    # it. It may not enter at `queued`, and it may not jump straight to an
    # outcome from a running stage.
    STATE_QUEUED: frozenset(
        {STATE_PARSING, STATE_QUALITY_CHECK, STATE_NORMALIZING, STATE_INDEXING, STATE_ANALYZING}
    )
    # A job can be cancelled or fail before it is ever claimed, so the
    # outcomes are reachable from here too. A *completed* job is not: nothing
    # reaches `completed` without a stage having run, because an empty
    # pipeline is rejected when it is built.
    | _OUTCOMES,
    # In the middle of the spine, only the next stage — plus the requeue edge
    # below. A handler that forgets to run the quality gate gets
    # `IllegalTransition` rather than a document that was indexed without ever
    # being graded.
    STATE_PARSING: frozenset({STATE_QUALITY_CHECK, STATE_QUEUED}) | _OUTCOMES,
    STATE_QUALITY_CHECK: frozenset(
        {STATE_FALLBACK, STATE_NORMALIZING, STATE_QUEUED}
    )
    | _OUTCOMES,
    STATE_FALLBACK: frozenset({STATE_NORMALIZING, STATE_QUEUED}) | _OUTCOMES,
    STATE_NORMALIZING: frozenset({STATE_INDEXING, STATE_QUEUED}) | _OUTCOMES,
    STATE_INDEXING: frozenset({STATE_ANALYZING, STATE_QUEUED}) | _OUTCOMES,
    STATE_ANALYZING: frozenset({STATE_QUEUED}) | _OUTCOMES,
    # Terminal means terminal. `failed` is the single exception and it exists
    # on purpose: a job that failed for a retryable reason has to be
    # submittable again, and `retry()` is the only thing that takes that edge.
    STATE_COMPLETED: frozenset(),
    STATE_PARTIAL: frozenset(),
    STATE_FAILED: frozenset({STATE_QUEUED}),
    STATE_CANCELLED: frozenset(),
}

#: The edges that lead back to :data:`STATE_QUEUED`, derived from the table so
#: it cannot drift from it: the recovery pass takes the running-state ones, and
#: :meth:`JobStore.retry` takes the ``failed`` one.
#:
#: They are in the graph rather than beside it for one reason — the write is
#: the same write, and a second code path that "usually" requeues correctly is
#: not enforcement. What keeps a *step* from taking them is that a step is
#: handed a :class:`JobContext`, which has no ``transition`` at all: a stage can
#: publish progress, note a fact and request a boundary, and cannot move the
#: job anywhere.
RECOVERY_EDGES: frozenset[tuple[str, str]] = frozenset(
    (state, STATE_QUEUED) for state in JOB_TRANSITIONS if STATE_QUEUED in JOB_TRANSITIONS[state]
)

#: What each spine state says, in the user's language. A job that is stuck at
#: ``parsing`` on somebody's screen should say what parsing is, and the only
#: place that text can live without drifting is next to the state name.
STATE_MESSAGES: dict[str, str] = {
    STATE_QUEUED: "任务已提交，等待 worker 领取。",
    STATE_PARSING: "正在解析文档。",
    STATE_QUALITY_CHECK: "正在评估解析质量。",
    STATE_FALLBACK: "正在评估是否需要回退解析。",
    STATE_NORMALIZING: "正在切分并规范化文档。",
    STATE_INDEXING: "正在写入向量与全文索引。",
    STATE_ANALYZING: "正在生成分析结果。",
    STATE_COMPLETED: "任务已完成。",
    STATE_PARTIAL: "任务部分完成，存在未成功的步骤。",
    STATE_FAILED: "任务失败。",
    STATE_CANCELLED: "任务已取消。",
}


def can_transition(source: str, target: str) -> bool:
    """Whether ``source → target`` is in the graph. Unknown states are ``False``.

    ``False`` rather than an exception for an *unknown* state, so a caller can
    use this to ask about a value that came out of a database rather than out
    of this module. :func:`assert_transition` is the enforcing form.
    """
    return target in JOB_TRANSITIONS.get(source, frozenset())


def is_terminal(state: str) -> bool:
    return state in TERMINAL_STATES


def is_running(state: str) -> bool:
    """Whether a worker is expected to be holding this job right now."""
    return state in RUNNING_STATES


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class JobError(Exception):
    """Base for everything this module raises, so a caller can catch one type."""


class JobNotFound(JobError):
    """No row with that id. ``None`` is for a getter; this is for a mutation."""


class IllegalTransition(JobError):
    """A write asked for a state change the graph does not contain.

    The message names both states, because the two questions a reader has are
    always the same: what was it, and what was someone trying to do to it.
    """

    def __init__(self, source: str, target: str, job_id: str = "") -> None:
        self.source = source
        self.target = target
        self.job_id = job_id
        prefix = f"任务 {job_id} " if job_id else ""
        allowed = sorted(JOB_TRANSITIONS.get(source, frozenset())) or ["（无）"]
        super().__init__(
            f"{prefix}不允许的状态变更：{source} → {target}。"
            f"从 {source} 只能进入 {', '.join(allowed)}。"
        )


class JobConflict(JobError):
    """Someone else moved the job between the read and the write.

    Raised instead of silently overwriting. The two writers are a worker and an
    API caller — a cancel arriving while a step is finishing its own write — and
    last-writer-wins there would silently drop a cancel request.
    """

    def __init__(self, job_id: str, expected: str, actual: str) -> None:
        self.job_id = job_id
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"任务 {job_id} 已被其他写入方更新：期望状态 {expected}，实际状态 {actual}。"
            "请重新读取任务状态后重试。"
        )


class JobCancelled(Exception):
    """Raised by :meth:`JobContext.boundary` when a cancel is pending.

    Deliberately not a :class:`JobError`. It is the pipeline's normal control
    flow — a boundary check that found a cancel and asked the step to stop —
    and a handler that catches ``JobError`` broadly must not swallow it and
    carry on doing the work the user just asked it to stop doing.
    """


class StepError(Exception):
    """Base for the three ways a step reports a problem to the worker."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class StepSkipped(StepError):
    """This step had nothing to do, and that is the right outcome.

    The trail records why. An optional stage that silently did nothing is
    indistinguishable from a stage that was never reached, and the user asking
    "did my document go to the cloud?" is entitled to an answer.
    """


class StepDegraded(StepError):
    """The step produced usable output and also a problem.

    This is what makes ``partial`` real. A parse that came back with quality
    warnings, or an analysis where one section failed, degrades rather than
    fails, and the job ends ``partial`` with the reason attached.
    """


class StepFailed(StepError):
    """The step could not produce its output.

    ``retryable`` says whether re-submitting could plausibly work. A document
    that failed to download is not retryable; a provider that was rate-limited
    is, and that is the difference between a job the user can re-run and one
    they would just re-run into the same wall.

    ``cloud_outcome`` is optional and defaults to empty, so a step that has no
    cloud involvement raises exactly as it did before. When a step *does* know
    the cloud is the reason — it has read
    :attr:`~paperwhisperer.ingestion.gateway.ParseResult.cloud_outcome` — it
    passes it here, and :func:`_settle` persists it onto the job in the same
    write as the error message. The default is deliberately "no opinion" rather
    than a guess: a step that does not know must not invent a reason, because a
    wrong one would be reported to the user as though it were measured.
    """

    def __init__(
        self, message: str, *, retryable: bool = False, cloud_outcome: str = ""
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.cloud_outcome = str(cloud_outcome or "")


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

#: Every key this module reads, with the default it uses when the key is
#: absent, and with the reason it exists. This table *is* the contract with
#: :mod:`paperwhisperer.core.config`, which this task does not own: whoever adds
#: the keys copies these names, types and defaults, and this module keeps
#: working unchanged either way. Every key has a working default so that a
#: checkout nobody has configured can still run a job — which is the state the
#: personal tier is required to work in.
#:
#: ``JOBS_ENABLED``
#:     Read by the app's lifespan hook (T2.3), not by this module: it decides
#:     whether :meth:`JobWorker.start` is called at all. Default ``True``. It
#:     exists so a read-only deployment or a test can serve the API without a
#:     background thread appearing that nothing is watching.
#: ``JOBS_POLL_INTERVAL_SECONDS``
#:     How long the worker's loop waits when it found nothing. Default ``0.5``.
#:     A condition variable would be faster and would not work across
#:     processes; the queue is a personal-scale queue and the loop is idle 99%
#:     of the time, so a half-second of latency is free.
#: ``JOBS_LEASE_SECONDS``
#:     How long a claimed job's lease lasts. Default ``300``. This is a
#:     dead-worker detector, not a progress guarantee: a step that runs longer
#:     than this must call :meth:`JobContext.heartbeat`, or another worker will
#:     conclude it died. On the personal tier there is no other worker, so the
#:     failure mode is a slow requeue, not a double run; T4.5 is where
#:     multi-worker correctness gets settled.
#: ``JOBS_MAX_ATTEMPTS``
#:     How many times a job may be claimed before recovery gives up and moves
#:     it to ``failed``. Default ``3``.
#: ``JOBS_WORKER_ID``
#:     Identity recorded in ``lease_owner``, read by
#:     :func:`default_worker_id` when a :class:`JobWorker` is constructed
#:     without an explicit id. Default: hostname, pid and a short random
#:     suffix — exactly the granularity needed to tell "the worker that held
#:     this is gone" from "some worker holds this", and to tell two workers in
#:     one process apart.
#: ``JOBS_RETENTION_DAYS``
#:     How long a terminal job is kept before
#:     :meth:`JobStore.sweep_terminal` deletes it. Default ``14``.
#:
#:     The number lives here because only this module knows what a job row
#:     means, and the *sweep* lives here too — a :class:`JobWorker` runs it on
#:     its first tick and thereafter at most once per window. It was previously
#:     declared here and in ``core/config.py`` and read by nothing, which is a
#:     slow disk leak wearing the costume of a configured guarantee: an
#:     operator reads ``JOBS_RETENTION_DAYS=14`` as "this queue is bounded" and
#:     it is not, because every job keeps its payload, result and event trail
#:     for as long as the process runs. Retention across the *other* tables is
#:     still T2.2's business; this column is the job table's own.
REQUIRED_CONFIG_KEYS: dict[str, Any] = {
    "JOBS_ENABLED": True,
    # Integer seconds, matching the integer settings in ``core/config.py``.
    # These are the values config declares, and config is read with ``getattr``
    # against this dict, so the two must agree or the declared default is a lie
    # that only shows up as a behaviour difference nobody can trace.
    "JOBS_POLL_INTERVAL_SECONDS": 1,
    "JOBS_LEASE_SECONDS": 300,
    "JOBS_MAX_ATTEMPTS": 3,
    "JOBS_WORKER_ID": "",
    "JOBS_RETENTION_DAYS": 14,
}

#: Floor on any lease, in seconds. A lease of zero would mean "already expired"
#: the instant it was written, and a lease that short turns a slow write into a
#: duplicate run. One second is the cheapest margin that covers both. It also
#: means no test can expire a lease by waiting less than a second, which is why
#: the tests ask the question with a clock (``recover_stale(now=...)``) instead
#: of a sleep.
MIN_LEASE_SECONDS = 1.0


def default_worker_id(settings: Any = None) -> str:
    """A worker identity that changes when the process does.

    Hostname and pid, not a uuid: the point of the string is to distinguish
    *this* worker from *another* one, including after a restart, and a uuid
    would make a restarted worker look like a stranger to its own leases. It
    is generated per :class:`JobWorker`, not once per module, so two workers in
    one process still do not collide.
    """
    configured = str(_setting(settings, "JOBS_WORKER_ID") or "").strip()
    if configured:
        return configured
    return f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def _setting(settings: Any, key: str) -> Any:
    value = getattr(settings, key, None)
    return REQUIRED_CONFIG_KEYS[key] if value is None else value


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


def assert_transition(source: str, target: str, job_id: str = "") -> None:
    """Raise :class:`IllegalTransition` unless the graph allows the edge.

    Every state-changing write calls this. The check lives here, once, rather
    than in each backend's SQL, because a check written twice is a check that
    will eventually be written differently in the two places — and the whole
    value of the two-tier split is that it is not.
    """
    if not can_transition(source, target):
        raise IllegalTransition(source, target, job_id)


@dataclass(slots=True)
class JobEvent:
    """One thing that happened to a job.

    A *transition* is a state change; a *lease* is not, and both are recorded
    here so the trail answers "what happened to this job" in one place.
    :attr:`changed` is the distinction — a reclaim after a dead worker has
    ``from_state == to_state`` and is still worth a line, because a job that
    silently changed hands is a job nobody is watching.

    ``note`` is the machine-readable half (an error code, a recovery reason)
    and ``message`` is the half a user reads. They are separate because they
    are written at different moments by different code: the message by the
    worker that knows what it just did, the note by the code path that decided
    to do it.
    """

    sequence: int = 0
    job_id: str = ""
    from_state: str = STATE_QUEUED
    to_state: str = STATE_QUEUED
    at: str = ""
    message: str = ""
    note: str = ""
    attempt: int = 0

    @property
    def changed(self) -> bool:
        return self.from_state != self.to_state

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": int(self.sequence),
            "job_id": str(self.job_id),
            "from_state": str(self.from_state),
            "to_state": str(self.to_state),
            "at": str(self.at),
            "message": str(self.message),
            "note": str(self.note),
            "attempt": int(self.attempt),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> JobEvent:
        return cls(
            sequence=int(payload.get("sequence") or 0),
            job_id=str(payload.get("job_id") or ""),
            from_state=str(payload.get("from_state") or STATE_QUEUED),
            to_state=str(payload.get("to_state") or STATE_QUEUED),
            at=str(payload.get("at") or ""),
            message=str(payload.get("message") or ""),
            note=str(payload.get("note") or ""),
            attempt=int(payload.get("attempt") or 0),
        )

    def to_state_dict(self) -> dict[str, Any]:
        """The projection a subscriber is allowed to see.

        No payload, no context, no result, no error. The subscriber's job is to
        answer "where is this job now", and a client that can rebuild the truth
        from state alone can reconnect and re-ask. A client that also received
        the model output would have to decide what to do with a half-consumed
        stream, which is the situation the plan's exit criterion forbids.
        """
        return {
            "job_id": self.job_id,
            "sequence": int(self.sequence),
            "state": self.to_state,
            "previous_state": self.from_state,
            "changed": self.changed,
            "message": self.message,
            "note": self.note,
            "terminal": self.to_state in TERMINAL_STATES,
            "attempt": int(self.attempt),
            "at": self.at,
        }


@dataclass(slots=True)
class Job:
    """One durable unit of long-running work.

    Three payload-shaped fields, and the split is the point:

    ``payload``
        The immutable input, as the submitter wrote it. Nothing the pipeline
        does may change it, so a retry of the same job runs the same work.
    ``context``
        The scratch handed from stage to stage — the parsed document on its way
        to the indexer, the index summary on its way to the analyzer. Not
        part of the contract and not part of the result; if it grows into
        something the API should serve, that is T2.2's ``document_versions``
        table, not this field.
    ``result``
        The user-facing summary, written by the last stage: counts, ids, a
        pointer to the analysis run. Not the analysis text. The text belongs to
        the analysis run T2.2 stores; duplicating it here would give two
        copies with one write path and no rule about which is current.

    ``events`` is the whole trail, in one row, committed with the state change
    that produced it. A separate ``job_events`` table would be the more
    conventional shape and it is wrong here for a specific reason: a
    reconnecting subscriber must not be able to observe a state that has no
    corresponding event, or a transition that has no state. One row, one
    write, one transaction. The trail is bounded by construction — a job
    appends at most a handful of events per stage — so it does not need
    trimming, and not trimming it is what keeps the resync guarantee
    unconditional.
    """

    job_id: str
    kind: str
    state: str = STATE_QUEUED
    idempotency_key: str = ""
    document_id: str = ""
    session_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    error_message: str = ""
    #: Why the cloud said no, when the cloud is the reason this job failed.
    #: One of ``quota_exhausted``, ``timed_out``, ``blocked``, ``failed`` — see
    #: :attr:`paperwhisperer.ingestion.gateway.ParseResult.cloud_outcome`.
    #:
    #: It exists because ``error_message`` cannot answer the question a reader
    #: actually has. "The cloud parse was refused" and "the cloud provider ran
    #: out of quota" arrive as the same sentence, and they are three different
    #: actions — top up the quota, try again later, or ask an administrator to
    #: allow this document to leave the machine. A reader who cannot tell them
    #: apart will do the wrong one, so the machine-readable half is carried
    #: alongside the prose rather than being left to be inferred from it.
    #:
    #: **It is persisted, not derived, and it rides in the payload like every
    #: other field.** Both tiers store the whole ``to_dict()``, so adding it
    #: here is what makes it survive to the route that reads it; a value written
    #: on one tier and not the other is the exact defect this repository keeps
    #: rediscovering, and keeping it inside ``to_dict``/``from_dict`` is what
    #: makes that impossible rather than merely unlikely. It is not a column,
    #: because nothing filters or orders on it — it is read only by a route
    #: deciding which error code to hand a browser.
    cloud_outcome: str = ""
    partial: bool = False
    partial_reasons: list[str] = field(default_factory=list)
    cancel_requested: bool = False
    cancel_requested_at: str = ""
    retryable: bool = False
    attempts: int = 0
    max_attempts: int = int(REQUIRED_CONFIG_KEYS["JOBS_MAX_ATTEMPTS"])
    lease_owner: str = ""
    lease_expires_at: float = 0.0
    started_at: str = ""
    finished_at: str = ""
    created_at: str = ""
    updated_at: str = ""
    sequence: int = 0
    events: list[JobEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.state not in JOB_TRANSITIONS:
            raise ValueError(
                f"未知的任务状态：{self.state!r}（可用值：{', '.join(JOB_STATES)}）"
            )
        if not isinstance(self.payload, dict):
            self.payload = {}
        if not isinstance(self.context, dict):
            self.context = {}
        if not isinstance(self.result, dict):
            self.result = {}
        if not isinstance(self.partial_reasons, list):
            self.partial_reasons = []
        self.cancel_requested = bool(self.cancel_requested)

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def is_leased(self) -> bool:
        return bool(self.lease_owner) and self.lease_expires_at > 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "state": self.state,
            "idempotency_key": self.idempotency_key,
            "document_id": self.document_id,
            "session_id": self.session_id,
            "payload": dict(self.payload),
            "context": dict(self.context),
            "result": dict(self.result),
            "message": self.message,
            "error_message": self.error_message,
            "cloud_outcome": self.cloud_outcome,
            "partial": self.partial,
            "partial_reasons": list(self.partial_reasons),
            "cancel_requested": self.cancel_requested,
            "cancel_requested_at": self.cancel_requested_at,
            "retryable": self.retryable,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "lease_owner": self.lease_owner,
            "lease_expires_at": float(self.lease_expires_at),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "sequence": self.sequence,
            "events": [event.to_dict() for event in self.events],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Job:
        return cls(
            job_id=str(payload.get("job_id") or ""),
            kind=str(payload.get("kind") or ""),
            state=str(payload.get("state") or STATE_QUEUED),
            idempotency_key=str(payload.get("idempotency_key") or ""),
            document_id=str(payload.get("document_id") or ""),
            session_id=str(payload.get("session_id") or ""),
            payload=dict(payload.get("payload") or {}),
            context=dict(payload.get("context") or {}),
            result=dict(payload.get("result") or {}),
            message=str(payload.get("message") or ""),
            error_message=str(payload.get("error_message") or ""),
            cloud_outcome=str(payload.get("cloud_outcome") or ""),
            partial=bool(payload.get("partial")),
            partial_reasons=list(payload.get("partial_reasons") or []),
            cancel_requested=bool(payload.get("cancel_requested")),
            cancel_requested_at=str(payload.get("cancel_requested_at") or ""),
            retryable=bool(payload.get("retryable")),
            attempts=int(payload.get("attempts") or 0),
            max_attempts=int(payload.get("max_attempts") or 0),
            lease_owner=str(payload.get("lease_owner") or ""),
            lease_expires_at=float(payload.get("lease_expires_at") or 0.0),
            started_at=str(payload.get("started_at") or ""),
            finished_at=str(payload.get("finished_at") or ""),
            created_at=str(payload.get("created_at") or ""),
            updated_at=str(payload.get("updated_at") or ""),
            sequence=int(payload.get("sequence") or 0),
            events=[JobEvent.from_dict(item) for item in payload.get("events") or []],
        )

    def to_state_dict(self) -> dict[str, Any]:
        """The state-only view, for a UI that polls and for a health check.

        Built from the job's own fields rather than from the trail so that a
        caller who has never seen an event still gets an answer.

        **The lease is projected, and it is the only reason the "is it stuck"
        question is answerable at all.** Recovery is deadline-driven: a worker
        that dies holding the shipped 300-second lease leaves the job untouched
        until that deadline passes, so for that whole window the job's *state*
        is indistinguishable from one a healthy worker is actively parsing.
        Without ``lease_owner`` and ``lease_expires_at`` here, those two jobs
        render as byte-identical dicts differing only in ``job_id``, and a
        subscriber has no way to know which one it is looking at. With them, a
        client can say "leased by <worker> until <t>" and either believe the
        holder or, once ``t`` has passed, know the holder is gone.

        Both fields are absolute values already stored on the row rather than
        anything derived from the current clock. That is deliberate: this
        projection is compared for equality between two polls (a UI asking
        "did anything change?") and :func:`routes_jobs.job_state` promises
        byte-identical JSON across two calls on the same job. A
        ``seconds_remaining`` computed from ``time.time()`` would tick between
        those two calls and break that promise for no gain — the client can
        subtract the two clocks itself, and the one it already has is the one
        it drew the page with.
        """
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "state": self.state,
            "message": self.message or STATE_MESSAGES.get(self.state, ""),
            "terminal": self.terminal,
            "partial": self.partial,
            "retryable": self.retryable,
            "cancel_requested": self.cancel_requested,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            # Who holds the lease, and until when. Empty/zero when nobody does,
            # which is the ordinary case for a queued or terminal job — and a
            # client can tell "not claimed" from "claimed and stalled" because
            # an unleased job reports an empty owner rather than no key at all.
            "lease_owner": self.lease_owner,
            "lease_expires_at": float(self.lease_expires_at),
            "sequence": self.sequence,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def summary_line(self) -> str:
        """One Chinese sentence naming the job and where it got to.

        The same contract as
        :meth:`~paperwhisperer.ingestion.orchestrator.ParseOutcome.summary_line`:
        unambiguous on its own, because it is the line a log or a CLI prints
        when nobody is watching a progress bar.
        """
        where = self.message or STATE_MESSAGES.get(self.state, self.state)
        if self.partial and self.partial_reasons:
            detail = "；".join(self.partial_reasons[:3])
            return f"任务 {self.job_id} 部分完成（{self.state}）：{where} 受影响：{detail}"
        if self.error_message:
            return f"任务 {self.job_id} 失败：{self.error_message}"
        return f"任务 {self.job_id}（{self.state}）：{where}"


@dataclass(frozen=True, slots=True)
class SubmitResult:
    """What ``submit`` returned, and whether it ran anything.

    ``deduplicated`` is on the result rather than buried in a log because the
    caller has to be able to tell the user what happened. A second click on
    "Analyse" that silently reuses a job in progress is a *good* outcome — it
    is the whole point of the idempotency key, because the alternative is
    paying for a second cloud parse of the same file — but it is only good if
    the user is told, and a response that is byte-identical to a fresh
    submission is a response that tells them nothing.
    """

    job: Job
    deduplicated: bool

    @property
    def job_id(self) -> str:
        return self.job.job_id

    @property
    def notice(self) -> str:
        if not self.deduplicated:
            return ""
        return (
            f"检测到重复提交（幂等键 {self.job.idempotency_key or '未设置'}），"
            f"已复用既有任务 {self.job.job_id}，当前状态：{self.job.state}。"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "job": self.job.to_state_dict(),
            "deduplicated": self.deduplicated,
            "notice": self.notice,
        }


@dataclass(frozen=True, slots=True)
class CancelAck:
    """The honest answer to "did you cancel it".

    ``accepted`` is ``False`` in two different situations, and the message says
    which: the job had already finished (there is nothing left to cancel), or
    the job is in the middle of a step whose boundary has not been reached yet
    (the flag is recorded and the worker will honour it). Presenting the second
    as a completed cancel would be a lie the user discovers when the bill for
    that cloud parse arrives.
    """

    job: Job
    accepted: bool
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job.job_id,
            "state": self.job.state,
            "cancel_requested": self.job.cancel_requested,
            "accepted": self.accepted,
            "message": self.message,
        }


# --------------------------------------------------------------------------
# The store contract
# --------------------------------------------------------------------------


@runtime_checkable
class JobStore(StoreCapable, Protocol):
    """Durable job rows, and the only way to move a job between states.

    No member name and no member docstring names a storage engine, for the
    reason T1.1 wrote down at length: the two tiers are only the same product
    if a caller cannot tell from the contract which one it got, and a contract
    docstring that names an engine is the first step towards a signature that
    requires one. The tier is a property of the *instance*, reported by
    :meth:`~.stores.interfaces.StoreCapable.capabilities`, and nowhere else.
    Where the two tiers genuinely differ — how a claim excludes the job another
    worker is holding — the difference is described by what it achieves rather
    than by what it is done with.
    """

    def initialize(self) -> None:
        """Create or migrate the ``jobs`` table. Idempotent.

        Same contract as the other stores' ``initialize``: a migration that has
        to know whether it already ran is a migration nobody can safely retry,
        and the factory calls this on every boot.
        """
        ...

    def submit(
        self,
        kind: str,
        payload: Mapping[str, Any] | None = None,
        *,
        idempotency_key: str = "",
        document_id: str = "",
        session_id: str = "",
        max_attempts: int | None = None,
        message: str = "",
    ) -> SubmitResult:
        """Enqueue a job, or return the existing one for ``idempotency_key``.

        Deduplication happens here, in the store, and not in the caller,
        because a check-then-insert in the API layer is a race: two double
        clicks arrive on two threads and both see an empty table. The store
        makes the decision inside the transaction that inserts the row, so at
        most one of them can win.

        An empty key disables deduplication entirely. That is a deliberate
        opt-out rather than a generated key, because a key nobody chose is a
        key nobody reasoned about, and a key derived from a timestamp would
        deduplicate nothing while looking like it deduplicated something.
        """
        ...

    def get_job(self, job_id: str) -> Job | None:
        """The row, or ``None``. A missing job is an ordinary outcome."""
        ...

    def list_jobs(
        self,
        *,
        state: str = "",
        kind: str = "",
        document_id: str = "",
        limit: int = 50,
        offset: int = 0,
    ) -> list[Job]:
        """Newest first, ordered by ``created_at`` then ``job_id``.

        The tiebreak is not cosmetic: two jobs submitted in the same second is
        the normal case on the personal tier, and without a total order the
        two tiers page the same queue differently.
        """
        ...

    def request_cancel(self, job_id: str) -> CancelAck:
        """Write ``cancel_requested`` and say whether that will do anything.

        Idempotent: cancelling a job that is already cancelled is not an error,
        it is the client retrying a request whose response it never saw.
        """
        ...

    def apply_queued_cancellations(self, limit: int = 50) -> list[Job]:
        """Cancel queued jobs whose cancel has been requested. Returns what went.

        A job nobody has started needs no worker's cooperation to stop, so the
        worker settles those from the queue rather than claiming them and
        immediately dropping them. It costs one indexed statement per tick and
        it keeps "I cancelled it" true for a job that had not begun.
        """
        ...

    def claim_next(
        self,
        worker_id: str,
        lease_seconds: float,
        *,
        now: float | None = None,
        kind: str = "",
    ) -> Job | None:
        """Take the oldest queued job, lease it to ``worker_id``, return it.

        The claim does not change the state. ``queued`` is where a job waits,
        and a claimed job is still waiting until the worker enters its first
        stage — which is a real, recoverable state (``lease_owner`` set, state
        still ``queued``) and the reason
        :meth:`apply_queued_cancellations` exists.

        The claim is a compare-and-set, and it also refuses any job whose lease
        is still live, because a claimed job still reads as ``queued`` and
        would otherwise be taken off the queue a second time. Both tiers get
        the exclusion from the engine's own write serialisation — which is
        what the plan's risk table means by the personal tier's single-writer
        constraint — so a second worker cannot run the same work, or bill the
        same external call, twice.
        """
        ...

    def transition(
        self,
        job_id: str,
        target: str,
        *,
        expected_state: str | None = None,
        message: str = "",
        note: str = "",
        patch: Mapping[str, Any] | None = None,
    ) -> Job:
        """Move a job to ``target``, recording the move. Idempotency aside.

        The graph in :data:`JOB_TRANSITIONS` is checked before anything is
        written, and the write is a compare-and-set on the state the caller
        expected to find. A caller that passes ``expected_state`` is asserting
        "I still hold this job"; losing that assertion raises
        :class:`JobConflict` rather than overwriting, because the thing that
        raced it is usually a cancel arriving mid-step.
        """
        ...

    def patch(self, job_id: str, changes: Mapping[str, Any]) -> Job:
        """Write fields without changing the state and without an event.

        For the scratch a step accumulates and the lease a step refreshes. No
        event, because nothing a subscriber watches moved — and a subscriber
        that woke up for every scratch write would be woken up constantly by a
        job that is doing exactly what it is supposed to be doing.
        """
        ...

    def retry(self, job_id: str) -> Job:
        """Put a failed job back on the queue, and reset its attempt count.

        The only caller of the ``failed → queued`` edge. It exists as a separate
        method rather than as ``transition(job_id, "queued")`` so that a caller
        cannot take that edge by accident from a state that is not a failure —
        and so the attempt reset, which is the whole reason a retry is not the
        same as a recovery, happens in exactly one place.
        """
        ...

    def recover_stale(self, *, now: float | None = None) -> list[Job]:
        """Requeue or fail every job whose lease has expired. Returns what moved.

        The release gate "worker/API 重启后任务可恢复或明确进入可重试终态" is
        this method. Called on the worker's first tick and on every tick after,
        because the moment a worker learns a job is orphaned is the moment it
        should be running it.
        """
        ...

    def events_since(self, job_id: str, since_sequence: int = 0) -> list[JobEvent]:
        """Trail entries with ``sequence > since_sequence``, oldest first.

        The resync primitive. A client that reconnects passes the last sequence
        it saw and gets exactly what it missed — including, on a job that
        finished while it was away, the terminal event.
        """
        ...

    def count_by_state(self) -> dict[str, int]:
        """How many jobs sit in each state, including the empty ones as zero.

        The zeroes are there so a health endpoint can render a queue panel
        without inventing a bar for a state the query did not mention.
        """
        ...

    def sweep_terminal(
        self, *, retention_days: int | None = None, now: float | None = None
    ) -> int:
        """Delete terminal jobs whose result is older than the retention window.

        ``JOBS_RETENTION_DAYS`` is declared in
        :data:`REQUIRED_CONFIG_KEYS` and in ``core/config.py``, and until this
        method existed nothing ever read it. A setting that is declared and
        never enforced is not a slow leak by accident — it is a claim the
        operator reads as a guarantee, so the queue grows forever on a
        deployment nobody is watching, and every job row keeps its payload, its
        result and its whole event trail alive for as long as the process runs.

        **Only terminal jobs, and only ones past the deadline.** A non-terminal
        job is somebody's work in progress and is never swept, however old it
        is — a queue whose janitor deletes stalled jobs is a queue that deletes
        the work instead of reporting that it is stuck. ``failed`` jobs are
        terminal and are swept like any other ending, which is the point of the
        gate they went through: the trail exists to explain a failure while
        somebody is still looking, not to be a permanent record.

        ``retention_days`` defaults to the configured value; pass an explicit
        number to sweep on a different schedule. Returns how many rows were
        deleted, so a caller can log it rather than sweep silently.
        """
        ...

    def close(self) -> None:
        """Release connections. Idempotent, and never raises."""
        ...


# --------------------------------------------------------------------------
# Store implementation, both tiers
# --------------------------------------------------------------------------

_SQLITE_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS jobs (
        job_id TEXT PRIMARY KEY,
        state TEXT NOT NULL,
        kind TEXT NOT NULL,
        idempotency_key TEXT NOT NULL DEFAULT '',
        document_id TEXT NOT NULL DEFAULT '',
        cancel_requested INTEGER NOT NULL DEFAULT 0,
        lease_expires_at REAL NOT NULL DEFAULT 0,
        sequence INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        payload TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state, created_at, job_id)",
    "CREATE INDEX IF NOT EXISTS idx_jobs_document ON jobs(document_id)",
    "CREATE INDEX IF NOT EXISTS idx_jobs_lease ON jobs(state, lease_expires_at)",
    # The dedupe guarantee, enforced by the database rather than by the
    # caller's read-then-write. Partial because an empty key means "no
    # deduplication", and without the predicate every job without a key would
    # collide with every other one on the second insert.
    """
    CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_idempotency
    ON jobs(idempotency_key) WHERE idempotency_key <> ''
    """,
)

#: The same table on the Hosted tier. ``DOUBLE PRECISION`` for the lease
#: because SQL compares it with ``<`` on both tiers, and a ``TIMESTAMPTZ``
#: column compared against a Python float would be a comparison of two
#: different questions. The partial unique index is spelled identically
#: because both engines support it, which is what lets the check-then-insert
#: in :meth:`PostgresJobStore.submit` be the same shape as the SQLite one.
#: ``{table}`` is the schema-qualified table; ``{index}`` is its bare name.
#: They are separate placeholders because PostgreSQL accepts a qualified name
#: for a table and rejects one for an index -- and a single placeholder produced
#: ``CREATE INDEX public.jobs_state``, so the Hosted tier could not create its
#: job table at all. Nothing in the suite caught it: the Hosted half of the job
#: contract skips wherever there is no database, and the SQLite path never sees
#: a schema qualifier. A bug that only exists on the tier nobody runs.
#:
#: Every TEXT sort key is declared ``COLLATE "C"``, and every ``ORDER BY`` on
#: one repeats the collation. The reason is in
#: :data:`~paperwhisperer.infra.stores.postgres_backend.SORT_KEY`, and for the
#: queue it is not cosmetic: ``claim_next`` takes ``ORDER BY created_at ASC,
#: job_id ASC LIMIT 1``, so the collation decides *which job the worker runs
#: next*. Under this project's own compose file -- which initialises the
#: cluster as ``en_US.utf8`` -- that order differed from SQLite's byte order for
#: any two ids differing only in case, so a queue holding ``a1b2c3`` and
#: ``A1b2c3`` drained in a different sequence on each tier. Same rows, same
#: query, different worker. The unit test that exists for exactly this --
#: ``test_claiming_drains_the_queue_in_a_total_order`` -- compares the two
#: jobs against each other rather than against a fixed expectation, so it
#: passed on both tiers while the two disagreed with each other.
_POSTGRES_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS {table} (
        job_id {sort_key} PRIMARY KEY,
        state {sort_key} NOT NULL,
        kind {sort_key} NOT NULL,
        idempotency_key {sort_key} NOT NULL DEFAULT '',
        document_id {sort_key} NOT NULL DEFAULT '',
        cancel_requested BOOLEAN NOT NULL DEFAULT FALSE,
        lease_expires_at DOUBLE PRECISION NOT NULL DEFAULT 0,
        sequence INTEGER NOT NULL DEFAULT 0,
        created_at {sort_key} NOT NULL,
        updated_at {sort_key} NOT NULL,
        payload JSONB NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS {index}_state ON {table} "
    '(state COLLATE "C", created_at COLLATE "C", job_id COLLATE "C")',
    "CREATE INDEX IF NOT EXISTS {index}_document ON {table} (document_id COLLATE \"C\")",
    "CREATE INDEX IF NOT EXISTS {index}_lease ON {table} "
    '(state COLLATE "C", lease_expires_at)',
    """
    CREATE UNIQUE INDEX IF NOT EXISTS {index}_idempotency
    ON {table} (idempotency_key COLLATE "C") WHERE idempotency_key <> ''
    """,
)

#: The columns written from the job's own fields, and the attribute each one
#: mirrors. A column that can disagree with the payload is a second source of
#: truth, so these are copies made on write and read only as sort keys or
#: filters — the rule the store tier set for the same reason. ``payload`` is
#: the record; the rest exist because SQL filters and orders on them.
_COLUMN_FIELDS = (
    ("state", "state"),
    ("kind", "kind"),
    ("idempotency_key", "idempotency_key"),
    ("document_id", "document_id"),
    ("cancel_requested", "cancel_requested"),
    ("lease_expires_at", "lease_expires_at"),
    ("sequence", "sequence"),
    ("created_at", "created_at"),
    ("updated_at", "updated_at"),
)


def _new_job(
    kind: str,
    payload: Mapping[str, Any] | None,
    *,
    idempotency_key: str,
    document_id: str,
    session_id: str,
    max_attempts: int,
    message: str,
) -> Job:
    """Build a fresh, unsubmitted job with its first trail entry.

    The submission event is written here rather than by the store so that the
    row and its trail are always consistent: there is no code path that
    produces a job with an empty trail, and a subscriber that attaches to a
    brand-new job can still see it being created.
    """
    if not str(kind or "").strip():
        raise ValueError("任务类型 kind 不能为空。")
    now_text = _now_iso()
    job = Job(
        job_id=uuid.uuid4().hex,
        kind=str(kind).strip(),
        state=STATE_QUEUED,
        idempotency_key=str(idempotency_key or "").strip(),
        document_id=str(document_id or "").strip(),
        session_id=str(session_id or "").strip(),
        payload=deepcopy(dict(payload or {})),
        message=str(message or STATE_MESSAGES[STATE_QUEUED]),
        max_attempts=max(1, int(max_attempts or 1)),
        created_at=now_text,
        updated_at=now_text,
        sequence=1,
    )
    job.events = [
        JobEvent(
            sequence=1,
            job_id=job.job_id,
            from_state=STATE_QUEUED,
            to_state=STATE_QUEUED,
            at=now_text,
            message=job.message,
            note="submitted",
            attempt=0,
        )
    ]
    return job


def _rollback(connection: Any) -> None:
    """Abandon a transaction, tolerating a rollback that is not needed.

    Called from the ``except`` of a write. A rollback that raises would
    replace the real error — a unique-index violation, a serialization failure
    — with "cannot rollback - no transaction is active", which is a strictly
    worse message about a strictly worse place to be looking.
    """
    try:
        connection.execute("ROLLBACK")
    except Exception:  # pragma: no cover - defensive; the original error matters more
        logger.debug("回滚任务事务失败（可能没有活动事务）。", exc_info=True)


def _column_values(job: Job) -> tuple[Any, ...]:
    """The non-payload columns, in the order every statement binds them.

    One function because three statements in two backends bind the same tuple
    in the same order, and an INSERT whose columns and values drift apart is
    a row whose ``state`` column says something other than what its payload
    says — the one bug that would make the two tiers disagree while both
    "worked".
    """
    return (
        job.state,
        job.kind,
        job.idempotency_key,
        job.document_id,
        # A real bool, not int(bool(...)). SQLite stores it as 0/1 without
        # complaint because bool is an int subclass, so the coercion was never
        # needed there -- but PostgreSQL declares this column BOOLEAN and
        # rejects an int parameter outright: "column cancel_requested is of
        # type boolean but expression is of type smallint". The two backends
        # share this tuple, so a convenience for one tier was a hard failure on
        # the other, visible only with a real server.
        job.cancel_requested,
        float(job.lease_expires_at),
        int(job.sequence),
        job.created_at,
        job.updated_at,
    )


def _apply_transition(
    job: Job,
    target: str,
    *,
    message: str,
    note: str,
    patch: Mapping[str, Any] | None,
    expected_state: str | None,
) -> Job:
    """The whole transition, in Python, shared by both tiers.

    One function rather than one per backend because the parts that must
    agree — the graph check, the event, the sequence number, the timestamps, the
    lease release on a terminal move — are the parts that are easy to write two
    slightly different ways. The backends are left with a row write and a
    compare-and-set, which is the part that genuinely differs.
    """
    source = job.state
    if expected_state is not None and source != expected_state:
        raise JobConflict(job.job_id, expected_state, source)
    assert_transition(source, target, job.job_id)

    changes = dict(patch or {})
    updated = Job.from_dict(job.to_dict())
    for name, value in changes.items():
        setattr(updated, name, value)
    updated.state = target
    updated.sequence = job.sequence + 1
    updated.updated_at = _now_iso()
    updated.message = str(message or STATE_MESSAGES.get(target, ""))
    if target in TERMINAL_STATES:
        updated.lease_owner = ""
        updated.lease_expires_at = 0.0
        updated.finished_at = updated.updated_at
    # The base list for the new event is whatever the caller patched in when it
    # has one, so that a worker holding accumulated trail lines does not lose
    # them to a re-read. The trade is explicit: a concurrent writer's event
    # could be overwritten in the same window, and the only writer that can
    # interleave here is an API cancel, which records no event of its own.
    base_events = list(changes.get("events") or job.events)
    updated.events = base_events + [
        JobEvent(
            sequence=updated.sequence,
            job_id=updated.job_id,
            from_state=source,
            to_state=target,
            at=updated.updated_at,
            message=updated.message,
            note=str(note or ""),
            attempt=updated.attempts,
        )
    ]
    return updated


def _terminal_patch(
    target: str,
    *,
    error: str = "",
    retryable: bool = False,
    cloud_outcome: str = "",
) -> dict[str, Any]:
    """The fields that differ between a good ending and a bad one.

    Computed before the write so the verdict and the state land in one
    transaction. Doing it afterwards would mean a crash between the two leaves
    a ``failed`` job with no error message — which is a failed job whose cause
    nobody can find, on exactly the run where finding it mattered. Only
    ``failed`` can be retryable: a job that finished did not need to be.

    ``cloud_outcome`` is persisted on the same transaction and by the same
    mechanism as ``error_message`` — both are payload fields carried in
    ``to_dict``, so both tiers store them and neither can have one without the
    other. That is the reason it is written here rather than left to the caller
    to patch in afterwards: a verdict split across two writes is a verdict that
    can survive a crash with only half of itself, which is the failure this
    whole function exists to prevent.

    A good ending clears it, exactly as it clears ``error_message``. A job that
    succeeded cannot be carrying a cloud refusal from a stage it recovered from,
    and leaving one there would let a stale reason outlive the failure it
    explained.
    """
    if target in {STATE_COMPLETED, STATE_PARTIAL, STATE_CANCELLED}:
        return {
            "retryable": False,
            "error_message": str(error or ""),
            "cloud_outcome": str(cloud_outcome or ""),
        }
    return {
        "retryable": bool(retryable),
        "error_message": str(error or STATE_MESSAGES[STATE_FAILED]),
        "cloud_outcome": str(cloud_outcome or ""),
    }


class _JobStoreBase:
    """Shared plumbing: the logger, the tier, and a per-backend capability."""

    tier = TIER_PERSONAL

    def __init__(
        self, *, lease_seconds: float | None = None, owned_bundle: StoreBundle | None = None
    ) -> None:
        self._lease_seconds = (
            float(REQUIRED_CONFIG_KEYS["JOBS_LEASE_SECONDS"])
            if lease_seconds is None
            else float(lease_seconds)
        )
        #: Set only when this store built its own store bundle (the no-argument
        #: path of :func:`build_job_store`). A caller that passed a bundle in
        #: keeps ownership of it, exactly as with the store tier.
        self._owned_bundle = owned_bundle
        self._capabilities = StoreCapabilities(tier=self.tier)

    def capabilities(self) -> StoreCapabilities:
        """A job store has no opinion about vectors or full-text search.

        Same reason the metadata store reports only its tier: the factory
        assembles the composite picture from all three, and a default here
        would read as a measurement nobody made.
        """
        return self._capabilities

    def close(self) -> None:
        """Close the owned bundle if there is one. Idempotent, and never raises.

        No-op in the normal case, because the job table shares a connection
        with the documents and closing it from here would close theirs too —
        the same reason the store tier's per-store ``close`` is a no-op.
        """
        bundle, self._owned_bundle = self._owned_bundle, None
        if bundle is None:
            return
        try:
            bundle.close()
        except Exception:  # pragma: no cover - close must not raise
            logger.warning("关闭任务存储失败。", exc_info=True)

    # -- helpers shared with the worker -------------------------------------

    def _recover_stale(
        self, fetched: Sequence[Job], *, now: float
    ) -> list[Job]:
        """The recovery decision, in Python, for rows the SQL handed back.

        The SQL narrows the candidates — non-terminal, leased, expired — and
        this decides what happens to each. The decision needs the attempt count
        against ``max_attempts``, which lives in the payload, so it is made
        once here rather than expressed twice in two dialects of SQL. Two
        copies of "when does a job give up" is exactly the pair that drifts.
        """
        moved: list[Job] = []
        for job in fetched:
            if job.terminal or not job.lease_owner:
                continue
            if job.state == STATE_QUEUED:
                # The retry budget is checked FIRST, before the lease release.
                # Releasing unconditionally meant a worker that kept dying
                # between claiming and entering its first step requeued the job
                # forever: `claim_next` never refuses an over-budget job, so
                # the two rules together guaranteed an infinite reclaim loop
                # that looks like progress in the queue depth and never
                # terminates. The release gate below already did this check for
                # running jobs; a job is "running" the moment it is claimed, so
                # the same rule applies here.
                if job.attempts >= job.max_attempts:
                    moved.append(
                        self.transition(
                            job.job_id,
                            STATE_FAILED,
                            message=(
                                f"worker 在领取后、首个步骤前中断，已重试 {job.attempts} 次，"
                                "任务进入可重试终态。"
                            ),
                            note="lease_expired_before_first_step_attempts_exhausted",
                            expected_state=job.state,
                            patch=_terminal_patch(
                                STATE_FAILED,
                                error=(
                                    f"任务已被领取 {job.attempts} 次且每次都在开始执行前中断，"
                                    "已达重试上限。"
                                ),
                                retryable=True,
                            ),
                        )
                    )
                    continue

                # Claimed and then abandoned before the first stage. There is
                # no stage to redo, so this is a lease release rather than a
                # requeue, and the attempt it already consumed stands.
                # `queued → queued` is deliberately not an edge in the graph —
                # see SqliteJobStore._write_raw for why the event is written
                # here instead of through transition().
                released = Job.from_dict(job.to_dict())
                released.lease_owner = ""
                released.lease_expires_at = 0.0
                released.sequence = job.sequence + 1
                released.updated_at = _now_iso()
                released.events = list(job.events) + [
                    JobEvent(
                        sequence=released.sequence,
                        job_id=released.job_id,
                        from_state=STATE_QUEUED,
                        to_state=STATE_QUEUED,
                        at=released.updated_at,
                        message="上一个 worker 领取后中断，任务已重新回到队列。",
                        note=f"lease_expired:{job.lease_owner}",
                        attempt=released.attempts,
                    )
                ]
                written = self._write_raw(released)
                # This path commits through `_write_raw` rather than
                # `transition()`, so it does not get counted by the call site
                # in the store's `transition()`. It is still a recovery — the
                # note is `lease_expired:<owner>`, which is exactly the prefix
                # the counter keys on — and leaving it uncounted would mean the
                # one recovery a user is most likely to hit (a worker that dies
                # in the claim window, repeatedly) is the one that stays at
                # zero on the dashboard.
                _record_transition(written, event=written.events[-1] if written.events else None)
                moved.append(written)
                continue

            if job.attempts >= job.max_attempts:
                exhausted = self.transition(
                    job.job_id,
                    STATE_FAILED,
                    message=(
                        f"worker 意外中断，已重试 {job.attempts} 次，"
                        "任务进入可重试终态。"
                    ),
                    note="lease_expired_attempts_exhausted",
                    expected_state=job.state,
                    patch=_terminal_patch(
                        STATE_FAILED,
                        error=(
                            f"任务在第 {job.attempts} 次执行中被 worker 中断，已达重试上限。"
                        ),
                        retryable=True,
                    ),
                )
                moved.append(exhausted)
                continue

            requeued = self.transition(
                job.job_id,
                STATE_QUEUED,
                message=(
                    f"检测到上一个 worker（{job.lease_owner}）已中断，"
                    "任务重新排队等待执行。"
                ),
                note="lease_expired",
                expected_state=job.state,
                patch={"lease_owner": "", "lease_expires_at": 0.0},
            )
            moved.append(requeued)
        if moved:
            logger.info("恢复了 %d 个中断的任务。", len(moved))
        return moved


class SqliteJobStore(_JobStoreBase):
    """The personal tier's job table, in the same SQLite file as the documents.

    Every claim and every transition is one transaction. That is not a
    performance choice, it is what makes the queue correct: SQLite serialises
    writers, so a claim that read the head of the queue and then wrote the
    lease is a claim no other thread can interleave with. The Hosted tier gets
    the same property from ``FOR UPDATE SKIP LOCKED``; the risk table's
    "SQLite 单写者并发" note is the constraint both are written around.
    """

    tier = TIER_PERSONAL

    def __init__(
        self,
        database: SqliteDatabase,
        *,
        lease_seconds: float | None = None,
        owned_bundle: StoreBundle | None = None,
    ) -> None:
        super().__init__(lease_seconds=lease_seconds, owned_bundle=owned_bundle)
        self.database = database

    @property
    def connection(self):
        return self.database.connection()

    def initialize(self) -> None:
        for statement in _SQLITE_SCHEMA:
            self.connection.execute(statement)

    # -- writes --------------------------------------------------------------

    def submit(
        self,
        kind: str,
        payload: Mapping[str, Any] | None = None,
        *,
        idempotency_key: str = "",
        document_id: str = "",
        session_id: str = "",
        max_attempts: int | None = None,
        message: str = "",
    ) -> SubmitResult:
        key = str(idempotency_key or "").strip()
        job = _new_job(
            kind,
            payload,
            idempotency_key=key,
            document_id=document_id,
            session_id=session_id,
            max_attempts=int(max_attempts or REQUIRED_CONFIG_KEYS["JOBS_MAX_ATTEMPTS"]),
            message=message,
        )
        connection = self.connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            existing = self._row_by_key(connection, key)
            if existing is not None:
                connection.execute("COMMIT")
                # Counted here as well as on the fresh-submit path, and
                # deliberately: the dedup case is a *good* outcome — it is what
                # stopped a double-click paying for a second cloud parse — and
                # a counter that omitted it would report fewer submissions
                # than the API actually received. The metric labels it
                # `deduplicated=true` rather than dropping it.
                duplicate = SubmitResult(Job.from_dict(existing), deduplicated=True)
                _record_submitted(duplicate)
                return duplicate
            self._insert(connection, job)
            connection.execute("COMMIT")
        except Exception:
            _rollback(connection)
            raise
        result = SubmitResult(job, deduplicated=False)
        _record_submitted(result)
        return result

    def _row_by_key(self, connection, key: str) -> dict[str, Any] | None:
        if not key:
            return None
        row = connection.execute(
            "SELECT payload FROM jobs WHERE idempotency_key = ?", (key,)
        ).fetchone()
        return json.loads(row[0]) if row is not None else None

    def _insert(self, connection, job: Job) -> None:
        connection.execute(
            "INSERT INTO jobs(job_id, state, kind, idempotency_key, document_id, "
            "cancel_requested, lease_expires_at, sequence, created_at, updated_at, payload) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (job.job_id, *_column_values(job), _dumps(job.to_dict())),
        )

    def _write_raw(self, job: Job) -> Job:
        """Write a fully-formed job without inventing an event.

        Only for the two places that append their own event after the fact
        (recovery's lease release), where routing through
        :meth:`transition` would be impossible — ``queued → queued`` is not an
        edge in the graph, and pretending it were would put a hole in the one
        structure this task exists to keep closed.
        """
        self.connection.execute(
            "UPDATE jobs SET state = ?, kind = ?, idempotency_key = ?, document_id = ?, "
            "cancel_requested = ?, lease_expires_at = ?, sequence = ?, created_at = ?, "
            "updated_at = ?, payload = ? WHERE job_id = ?",
            (*_column_values(job), _dumps(job.to_dict()), job.job_id),
        )
        return job

    def transition(
        self,
        job_id: str,
        target: str,
        *,
        expected_state: str | None = None,
        message: str = "",
        note: str = "",
        patch: Mapping[str, Any] | None = None,
    ) -> Job:
        job = self.get_job(job_id)
        if job is None:
            raise JobNotFound(f"任务 {job_id} 不存在。")
        updated = _apply_transition(
            job,
            target,
            message=message,
            note=note,
            patch=patch,
            expected_state=expected_state,
        )
        connection = self.connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            cursor = connection.execute(
                "UPDATE jobs SET state = ?, lease_expires_at = ?, sequence = ?, "
                "updated_at = ?, payload = ? WHERE job_id = ? AND state = ?",
                (
                    updated.state,
                    float(updated.lease_expires_at),
                    int(updated.sequence),
                    updated.updated_at,
                    _dumps(updated.to_dict()),
                    job.job_id,
                    job.state,
                ),
            )
            lost = not cursor.rowcount
            # Every exit from here ends the transaction — including the one
            # where we did not win. A lost compare-and-set is a *normal* outcome
            # (an API cancel arriving mid-step, or `recover_stale` requeueing a
            # job whose lease lapsed mid-parse), and it is the outcome the
            # lease and `JobConflict` exist to produce.
            #
            # Falling through without COMMIT or ROLLBACK here leaves the
            # transaction open on this thread's connection, and SQLite holds the
            # write lock for exactly as long as it stays open. Two things then
            # follow, and neither is a race that more care could have won:
            #
            # * this connection can never open another transaction, so every
            #   later write on it raises "cannot start a transaction within a
            #   transaction" — for the life of the thread; and
            # * it holds the single writer lock, so every *other* thread's
            #   write blocks until `busy_timeout` and then raises "database is
            #   locked".
            #
            # That second one is why a lost CAS on one request turned into 500s
            # on unrelated uploads: a 9-page parse outruns the 60s lease
            # `routes_v1.run_parse_job` ships, the background worker's
            # `recover_stale` requeues the job mid-parse, and the request
            # thread's terminal transition is the write that loses. ROLLBACK
            # releases the lock at once, and the follow-up read below then sees
            # committed state rather than reading through a transaction nobody
            # is going to close.
            if not lost:
                connection.execute("COMMIT")
            else:
                # `_rollback` rather than a bare ROLLBACK, for the reason its
                # own docstring gives: a rollback that raises would replace the
                # `JobConflict` this branch exists to raise with a worse
                # message about a worse place.
                _rollback(connection)
        except Exception:
            _rollback(connection)
            raise
        if not lost:
            _record_transition(
                updated, event=updated.events[-1] if updated.events else None
            )
            return updated
        current = self.get_job(job_id)
        actual = current.state if current is not None else "（已删除）"
        raise JobConflict(job_id, job.state, actual)

    def patch(self, job_id: str, changes: Mapping[str, Any]) -> Job:
        job = self.get_job(job_id)
        if job is None:
            raise JobNotFound(f"任务 {job_id} 不存在。")
        updated = Job.from_dict(job.to_dict())
        for name, value in changes.items():
            setattr(updated, name, value)
        updated.updated_at = _now_iso()
        # `sequence` is written here as well as in the payload: a step's
        # trail lines advance it without changing the state, and a column that
        # lagged its own payload would make `events_since` skip entries on one
        # tier and not the other.
        self.connection.execute(
            "UPDATE jobs SET cancel_requested = ?, lease_expires_at = ?, sequence = ?, "
            "updated_at = ?, payload = ? WHERE job_id = ?",
            (
                int(bool(updated.cancel_requested)),
                float(updated.lease_expires_at),
                int(updated.sequence),
                updated.updated_at,
                _dumps(updated.to_dict()),
                job.job_id,
            ),
        )
        return updated

    def retry(self, job_id: str) -> Job:
        job = self.get_job(job_id)
        if job is None:
            raise JobNotFound(f"任务 {job_id} 不存在。")
        if job.state != STATE_FAILED:
            raise IllegalTransition(job.state, STATE_QUEUED, job.job_id)
        return self.transition(
            job_id,
            STATE_QUEUED,
            message="任务已重新排队，等待重试。",
            note="retry_requested",
            patch={
                "attempts": 0,
                "retryable": False,
                "error_message": "",
                # Cleared with the message it qualified: a requeued job must not
                # still be carrying the previous run's cloud refusal, or the
                # next reader is told a reason for a run that has not happened
                # yet. Both tiers clear it, so neither can keep a stale reason.
                "cloud_outcome": "",
                "finished_at": "",
            },
            expected_state=STATE_FAILED,
        )

    def claim_next(
        self,
        worker_id: str,
        lease_seconds: float,
        *,
        now: float | None = None,
        kind: str = "",
    ) -> Job | None:
        moment = time.time() if now is None else float(now)
        expires = moment + max(MIN_LEASE_SECONDS, float(lease_seconds))
        connection = self.connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            # Never claim a job whose lease is still live. A claimed job is
            # still `queued` until its worker enters the first stage, so
            # without this a second worker would take the same job off the
            # queue, bump its attempt count and run the same cloud parse
            # twice — which is the exact loss the idempotency key exists to
            # prevent, arriving through the other door.
            sql = "SELECT payload FROM jobs WHERE state = ?"
            params: list[Any] = [STATE_QUEUED]
            if kind:
                sql += " AND kind = ?"
                params.append(str(kind))
            sql += " AND (lease_expires_at <= 0 OR lease_expires_at < ?)"
            sql += " ORDER BY created_at ASC, job_id ASC LIMIT 1"
            params.append(moment)
            row = connection.execute(sql, params).fetchone()
            if row is None:
                connection.execute("COMMIT")
                return None
            job = Job.from_dict(json.loads(row[0]))
            job.lease_owner = str(worker_id)
            job.lease_expires_at = expires
            job.attempts += 1
            if not job.started_at:
                job.started_at = _now_iso()
            job.updated_at = _now_iso()
            job.events = list(job.events) + [
                JobEvent(
                    sequence=job.sequence + 1,
                    job_id=job.job_id,
                    from_state=job.state,
                    to_state=job.state,
                    at=job.updated_at,
                    message=f"任务已被 worker {worker_id} 领取（第 {job.attempts} 次尝试）。",
                    note="claimed",
                    attempt=job.attempts,
                )
            ]
            job.sequence += 1
            self._write_raw(job)
            connection.execute("COMMIT")
        except Exception:
            _rollback(connection)
            raise
        return job

    def request_cancel(self, job_id: str) -> CancelAck:
        job = self.get_job(job_id)
        if job is None:
            raise JobNotFound(f"任务 {job_id} 不存在。")
        if job.terminal:
            return CancelAck(
                job=job,
                accepted=False,
                message=(
                    f"任务已处于终态（{job.state}），取消请求未生效；"
                    "若结果不完整，请重新提交任务。"
                ),
            )
        if not job.cancel_requested:
            job = self.patch(
                job_id,
                {"cancel_requested": True, "cancel_requested_at": _now_iso()},
            )
        return CancelAck(
            job=job,
            accepted=True,
            message=(
                "已登记取消请求。worker 会在下一个可取消边界停止任务；"
                "正在进行的那一步不会被强制中断。"
            ),
        )

    def apply_queued_cancellations(self, limit: int = 50) -> list[Job]:
        rows = self.connection.execute(
            "SELECT payload FROM jobs WHERE state = ? AND cancel_requested = 1 "
            "ORDER BY created_at ASC, job_id ASC LIMIT ?",
            (STATE_QUEUED, max(1, int(limit))),
        ).fetchall()
        settled: list[Job] = []
        for row in rows:
            job = Job.from_dict(json.loads(row[0]))
            settled.append(
                self.transition(
                    job.job_id,
                    STATE_CANCELLED,
                    message="任务在开始执行前被取消。",
                    note="cancelled_before_start",
                    expected_state=STATE_QUEUED,
                )
            )
        return settled

    def recover_stale(self, *, now: float | None = None) -> list[Job]:
        moment = time.time() if now is None else float(now)
        # No state filter. A `queued` job with an expired lease is a job a dead
        # worker claimed and never entered its first stage for, and leaving it
        # leased would block the queue behind a lease nobody holds — the exact
        # "stuck in running forever" failure in a quieter costume. `queued`
        # with no lease has `lease_expires_at = 0` and is excluded anyway.
        rows = self.connection.execute(
            "SELECT payload FROM jobs WHERE lease_expires_at > 0 AND lease_expires_at < ? "
            "ORDER BY lease_expires_at ASC, job_id ASC",
            (moment,),
        ).fetchall()
        fetched = [Job.from_dict(json.loads(row[0])) for row in rows]
        return self._recover_stale(fetched, now=moment)

    # -- reads ---------------------------------------------------------------

    def get_job(self, job_id: str) -> Job | None:
        row = self.connection.execute(
            "SELECT payload FROM jobs WHERE job_id = ?", (str(job_id),)
        ).fetchone()
        return Job.from_dict(json.loads(row[0])) if row is not None else None

    def list_jobs(
        self,
        *,
        state: str = "",
        kind: str = "",
        document_id: str = "",
        limit: int = 50,
        offset: int = 0,
    ) -> list[Job]:
        sql = "SELECT payload FROM jobs"
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in (("state", state), ("kind", kind), ("document_id", document_id)):
            if value:
                clauses.append(f"{column} = ?")
                params.append(str(value))
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC, job_id ASC LIMIT ? OFFSET ?"
        params.extend([max(0, int(limit)), max(0, int(offset))])
        return [Job.from_dict(json.loads(row[0])) for row in self.connection.execute(sql, params)]

    def events_since(self, job_id: str, since_sequence: int = 0) -> list[JobEvent]:
        job = self.get_job(job_id)
        if job is None:
            return []
        return [event for event in job.events if event.sequence > int(since_sequence)]

    def count_by_state(self) -> dict[str, int]:
        rows = self.connection.execute(
            "SELECT state, COUNT(*) FROM jobs GROUP BY state"
        ).fetchall()
        counts = {state: 0 for state in JOB_STATES}
        for state, total in rows:
            counts[str(state)] = int(total)
        _record_queue_depth(counts)
        return counts

    def sweep_terminal(
        self, *, retention_days: int | None = None, now: float | None = None
    ) -> int:
        cutoff = _retention_cutoff(retention_days, now)
        # `finished_at` is a payload field, not a column on either tier, so the
        # deadline cannot be pushed into SQL the way the lease filter is. The
        # `state IN (...)` predicate still is — it is what keeps the candidate
        # set to the terminal rows and nothing else — and the timestamp test
        # runs in Python against each candidate's own stamp. See
        # :meth:`PostgresJobStore.sweep_terminal` for the same shape.
        rows = self.connection.execute(
            "SELECT payload FROM jobs WHERE state IN "
            f"({', '.join('?' for _ in TERMINAL_STATES)})",
            list(TERMINAL_STATES),
        ).fetchall()
        doomed = _terminal_ids_before(rows, cutoff)
        if not doomed:
            return 0
        connection = self.connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(
                f"DELETE FROM jobs WHERE job_id IN ({', '.join('?' for _ in doomed)})",
                doomed,
            )
            connection.execute("COMMIT")
        except Exception:
            _rollback(connection)
            raise
        logger.info("按保留期清除了 %d 个已结束的任务。", len(doomed))
        return len(doomed)


class PostgresJobStore(_JobStoreBase):
    """The Hosted tier's job table, in the same database as the documents.

    Written to mirror :class:`SqliteJobStore` statement for statement, because
    "the same suite runs on both tiers" is only true if the second
    implementation is a translation and not a redesign. Where a statement has
    to differ, the difference is the smallest one the engine allows and it is
    commented at the call site:

    * ``FOR UPDATE SKIP LOCKED`` replaces the transaction's exclusive write
      lock, so two workers can claim two different jobs at the same time and
      never the same one.
    * ``pg_advisory_xact_lock(hashtext(key))`` guards the idempotency
      check-then-insert, because ``FOR UPDATE`` locks rows that exist and two
      submissions of the same key both see nothing.
    * ``RETURNING payload`` makes the read and the write one round trip, which
      is why the compare-and-set does not need a follow-up ``SELECT``.

    None of it can be exercised here — there is no server and no driver — so
    the shared contract in ``tests/test_jobs.py`` is written once and run
    against this class whenever ``PAPERWHISPERER_TEST_PG_DSN`` is set. That
    skip is the honest state of the tier here, and the reason this class is a
    translation of a tested one rather than an independent design.
    """

    tier = TIER_HOSTED
    TABLE = "store_jobs"

    def __init__(
        self,
        database: PostgresDatabase,
        *,
        lease_seconds: float | None = None,
        owned_bundle: StoreBundle | None = None,
    ) -> None:
        super().__init__(lease_seconds=lease_seconds, owned_bundle=owned_bundle)
        self.database = database

    @property
    def table(self) -> str:
        return self.database.qualified(self.TABLE)

    def initialize(self) -> None:
        with self.database.cursor() as cursor:
            cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {self.database.schema}")
            for statement in _POSTGRES_SCHEMA:
                cursor.execute(
                    statement.format(
                        table=self.table,
                        # Postgres index names live in the table's own
                        # schema; a qualified index name is a syntax error.
                        index=self.table.rsplit(".", 1)[-1],
                        sort_key=SORT_KEY,
                    )
                )

    # -- writes --------------------------------------------------------------

    def submit(
        self,
        kind: str,
        payload: Mapping[str, Any] | None = None,
        *,
        idempotency_key: str = "",
        document_id: str = "",
        session_id: str = "",
        max_attempts: int | None = None,
        message: str = "",
    ) -> SubmitResult:
        key = str(idempotency_key or "").strip()
        job = _new_job(
            kind,
            payload,
            idempotency_key=key,
            document_id=document_id,
            session_id=session_id,
            max_attempts=int(max_attempts or REQUIRED_CONFIG_KEYS["JOBS_MAX_ATTEMPTS"]),
            message=message,
        )
        with self.database.cursor() as cursor:
            if key:
                # A transaction-level advisory lock keyed on the idempotency
                # key — the same technique the store backends use for document
                # upserts, and for the same reason: `FOR UPDATE` cannot lock a
                # row that does not exist yet, so two concurrent submissions
                # of the same key would both find nothing and both insert.
                cursor.execute(
                    "SELECT pg_advisory_xact_lock(hashtext(%s)::bigint)", (key,)
                )
                row = cursor.execute(
                    f"SELECT payload FROM {self.table} WHERE idempotency_key = %s", (key,)
                ).fetchone()
                if row is not None:
                    # Same two call sites as the personal tier, in the same
                    # order: the deduplicated outcome is counted before it is
                    # returned and the fresh one after the INSERT commits.
                    duplicate = SubmitResult(self._job(row[0]), deduplicated=True)
                    _record_submitted(duplicate)
                    return duplicate
            cursor.execute(
                f"""
                INSERT INTO {self.table}(job_id, state, kind, idempotency_key, document_id,
                                         cancel_requested, lease_expires_at, sequence,
                                         created_at, updated_at, payload)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                """,
                (job.job_id, *_column_values(job), _dumps(job.to_dict())),
            )
        submitted = SubmitResult(job, deduplicated=False)
        _record_submitted(submitted)
        return submitted

    @staticmethod
    def _job(payload: Any) -> Job:
        if isinstance(payload, (str, bytes)):
            payload = json.loads(payload)
        return Job.from_dict(payload or {})

    def _write_raw(self, job: Job) -> Job:
        """Write a fully-formed job without inventing an event. See the SQLite one."""
        with self.database.cursor() as cursor:
            cursor.execute(
                f"""
                UPDATE {self.table} SET state = %s, kind = %s, idempotency_key = %s,
                    document_id = %s, cancel_requested = %s, lease_expires_at = %s,
                    sequence = %s, created_at = %s, updated_at = %s, payload = %s::jsonb
                WHERE job_id = %s
                """,
                (*_column_values(job), _dumps(job.to_dict()), job.job_id),
            )
        return job

    def transition(
        self,
        job_id: str,
        target: str,
        *,
        expected_state: str | None = None,
        message: str = "",
        note: str = "",
        patch: Mapping[str, Any] | None = None,
    ) -> Job:
        job = self.get_job(job_id)
        if job is None:
            raise JobNotFound(f"任务 {job_id} 不存在。")
        updated = _apply_transition(
            job,
            target,
            message=message,
            note=note,
            patch=patch,
            expected_state=expected_state,
        )
        with self.database.cursor() as cursor:
            # Compare-and-set on the state the caller expected. RETURNING
            # distinguishes "I moved it" from "someone else did" without a
            # second round trip, which is the same fact the SQLite rowcount
            # gives.
            row = cursor.execute(
                f"""
                UPDATE {self.table} SET state = %s, lease_expires_at = %s, sequence = %s,
                    updated_at = %s, payload = %s::jsonb
                WHERE job_id = %s AND state = %s
                RETURNING state
                """,
                (
                    updated.state,
                    float(updated.lease_expires_at),
                    int(updated.sequence),
                    updated.updated_at,
                    _dumps(updated.to_dict()),
                    job.job_id,
                    job.state,
                ),
            ).fetchone()
        if row is None:
            current = self.get_job(job_id)
            actual = current.state if current is not None else "（已删除）"
            raise JobConflict(job_id, job.state, actual)
        # Same call site as the personal tier's, and for the same reason: the
        # counter moves on a committed row, never on an attempt that lost its
        # compare-and-set.
        _record_transition(
            updated, event=updated.events[-1] if updated.events else None
        )
        return updated

    def patch(self, job_id: str, changes: Mapping[str, Any]) -> Job:
        job = self.get_job(job_id)
        if job is None:
            raise JobNotFound(f"任务 {job_id} 不存在。")
        updated = Job.from_dict(job.to_dict())
        for name, value in changes.items():
            setattr(updated, name, value)
        updated.updated_at = _now_iso()
        with self.database.cursor() as cursor:
            cursor.execute(
                f"""
                UPDATE {self.table} SET cancel_requested = %s, lease_expires_at = %s,
                    sequence = %s, updated_at = %s, payload = %s::jsonb WHERE job_id = %s
                """,
                (
                    updated.cancel_requested,  # a bool: this column is BOOLEAN
                    float(updated.lease_expires_at),
                    int(updated.sequence),
                    updated.updated_at,
                    _dumps(updated.to_dict()),
                    job.job_id,
                ),
            )
        return updated

    def retry(self, job_id: str) -> Job:
        job = self.get_job(job_id)
        if job is None:
            raise JobNotFound(f"任务 {job_id} 不存在。")
        if job.state != STATE_FAILED:
            raise IllegalTransition(job.state, STATE_QUEUED, job.job_id)
        return self.transition(
            job_id,
            STATE_QUEUED,
            message="任务已重新排队，等待重试。",
            note="retry_requested",
            # `cloud_outcome` is cleared alongside `error_message`, on this tier
            # exactly as on the personal one, for the same reason: a requeued
            # job must not still be carrying the previous run's cloud refusal.
            patch={
                "attempts": 0,
                "retryable": False,
                "error_message": "",
                "cloud_outcome": "",
                "finished_at": "",
            },
            expected_state=STATE_FAILED,
        )

    def claim_next(
        self,
        worker_id: str,
        lease_seconds: float,
        *,
        now: float | None = None,
        kind: str = "",
    ) -> Job | None:
        moment = time.time() if now is None else float(now)
        expires = moment + max(MIN_LEASE_SECONDS, float(lease_seconds))
        with self.database.cursor() as cursor:
            # The lease filter is the same rule as on the personal tier: a
            # claimed job is still `queued` until its worker enters the first
            # stage, so a second worker must not take it off the queue and run
            # the same work twice. SKIP LOCKED is what makes several workers
            # share one queue at all — a row another worker is already
            # claiming is skipped rather than waited on, so a slow claim never
            # becomes a queue-wide stall.
            sql = f"SELECT payload FROM {self.table} WHERE state = %s"
            params: list[Any] = [STATE_QUEUED]
            if kind:
                sql += " AND kind = %s"
                params.append(str(kind))
            sql += " AND (lease_expires_at <= 0 OR lease_expires_at < %s)"
            sql += (
                ' ORDER BY created_at COLLATE "C" ASC, job_id COLLATE "C" ASC '
                "LIMIT 1 FOR UPDATE SKIP LOCKED"
            )
            params.append(moment)
            row = cursor.execute(sql, params).fetchone()
            if row is None:
                return None
            job = self._job(row[0])
            job.lease_owner = str(worker_id)
            job.lease_expires_at = expires
            job.attempts += 1
            if not job.started_at:
                job.started_at = _now_iso()
            job.updated_at = _now_iso()
            job.events = list(job.events) + [
                JobEvent(
                    sequence=job.sequence + 1,
                    job_id=job.job_id,
                    from_state=job.state,
                    to_state=job.state,
                    at=job.updated_at,
                    message=f"任务已被 worker {worker_id} 领取（第 {job.attempts} 次尝试）。",
                    note="claimed",
                    attempt=job.attempts,
                )
            ]
            job.sequence += 1
            cursor.execute(
                f"""
                UPDATE {self.table} SET lease_expires_at = %s, sequence = %s,
                    updated_at = %s, payload = %s::jsonb WHERE job_id = %s
                """,
                (
                    float(job.lease_expires_at),
                    int(job.sequence),
                    job.updated_at,
                    _dumps(job.to_dict()),
                    job.job_id,
                ),
            )
        return job

    def request_cancel(self, job_id: str) -> CancelAck:
        job = self.get_job(job_id)
        if job is None:
            raise JobNotFound(f"任务 {job_id} 不存在。")
        if job.terminal:
            return CancelAck(
                job=job,
                accepted=False,
                message=(
                    f"任务已处于终态（{job.state}），取消请求未生效；"
                    "若结果不完整，请重新提交任务。"
                ),
            )
        if not job.cancel_requested:
            job = self.patch(
                job_id, {"cancel_requested": True, "cancel_requested_at": _now_iso()}
            )
        return CancelAck(
            job=job,
            accepted=True,
            message=(
                "已登记取消请求。worker 会在下一个可取消边界停止任务；"
                "正在进行的那一步不会被强制中断。"
            ),
        )

    def apply_queued_cancellations(self, limit: int = 50) -> list[Job]:
        with self.database.cursor() as cursor:
            rows = cursor.execute(
                f"""
                SELECT payload FROM {self.table}
                WHERE state = %s AND cancel_requested = TRUE
                ORDER BY created_at COLLATE "C" ASC, job_id COLLATE "C" ASC
                LIMIT %s FOR UPDATE SKIP LOCKED
                """,
                (STATE_QUEUED, max(1, int(limit))),
            ).fetchall()
        settled: list[Job] = []
        for row in rows:
            job = self._job(row[0])
            settled.append(
                self.transition(
                    job.job_id,
                    STATE_CANCELLED,
                    message="任务在开始执行前被取消。",
                    note="cancelled_before_start",
                    expected_state=STATE_QUEUED,
                )
            )
        return settled

    def recover_stale(self, *, now: float | None = None) -> list[Job]:
        moment = time.time() if now is None else float(now)
        with self.database.cursor() as cursor:
            rows = cursor.execute(
                f"""
                SELECT payload FROM {self.table}
                WHERE lease_expires_at > 0 AND lease_expires_at < %s
                ORDER BY lease_expires_at ASC, job_id COLLATE "C" ASC
                """,
                (moment,),
            ).fetchall()
        fetched = [self._job(row[0]) for row in rows]
        return self._recover_stale(fetched, now=moment)

    # -- reads ---------------------------------------------------------------

    def get_job(self, job_id: str) -> Job | None:
        with self.database.cursor() as cursor:
            row = cursor.execute(
                f"SELECT payload FROM {self.table} WHERE job_id = %s", (str(job_id),)
            ).fetchone()
        return self._job(row[0]) if row is not None else None

    def list_jobs(
        self,
        *,
        state: str = "",
        kind: str = "",
        document_id: str = "",
        limit: int = 50,
        offset: int = 0,
    ) -> list[Job]:
        sql = f"SELECT payload FROM {self.table}"
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in (("state", state), ("kind", kind), ("document_id", document_id)):
            if value:
                clauses.append(f"{column} = %s")
                params.append(str(value))
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += (
            ' ORDER BY created_at COLLATE "C" DESC, job_id COLLATE "C" ASC '
            "LIMIT %s OFFSET %s"
        )
        params.extend([max(0, int(limit)), max(0, int(offset))])
        with self.database.cursor() as cursor:
            rows = cursor.execute(sql, params).fetchall()
        return [self._job(row[0]) for row in rows]

    def events_since(self, job_id: str, since_sequence: int = 0) -> list[JobEvent]:
        job = self.get_job(job_id)
        if job is None:
            return []
        return [event for event in job.events if event.sequence > int(since_sequence)]

    def count_by_state(self) -> dict[str, int]:
        with self.database.cursor() as cursor:
            rows = cursor.execute(
                f"SELECT state, COUNT(*) FROM {self.table} GROUP BY state"
            ).fetchall()
        counts = {state: 0 for state in JOB_STATES}
        for state, total in rows:
            counts[str(state)] = int(total)
        _record_queue_depth(counts)
        return counts

    def sweep_terminal(
        self, *, retention_days: int | None = None, now: float | None = None
    ) -> int:
        cutoff = _retention_cutoff(retention_days, now)
        # The same statement as the personal tier's, statement for statement.
        # `finished_at` is a payload field rather than a column on this tier,
        # so unlike `lease_expires_at` it cannot be compared in SQL — the filter
        # runs on `state` alone here and the timestamp test is applied in
        # Python against each candidate's own stamp. The terminal-state
        # predicate is still in SQL because it is what keeps the candidate set
        # small enough to read.
        with self.database.cursor() as cursor:
            rows = cursor.execute(
                f"SELECT payload FROM {self.table} WHERE state = ANY(%s)",
                (list(TERMINAL_STATES),),
            ).fetchall()
            doomed = _terminal_ids_before(rows, cutoff)
            if not doomed:
                return 0
            cursor.execute(
                f"DELETE FROM {self.table} WHERE job_id = ANY(%s)", (doomed,)
            )
        logger.info("按保留期清除了 %d 个已结束的任务。", len(doomed))
        return len(doomed)


def build_job_store(
    settings: Any = None,
    *,
    bundle: StoreBundle | None = None,
    lease_seconds: float | None = None,
) -> JobStore:
    """The one place a job store's backend is chosen, mirroring ``build_stores``.

    A ``bundle`` is the normal argument and the recommended one: it puts the
    ``jobs`` table in the same SQLite file, or the same PostgreSQL database, as
    the documents, which is what T2.2's "one data directory to back up" needs.

    Without one, a bundle is built and owned by this function — the store's
    ``close()`` closes it. That path exists so a script or a test can have a job
    queue without knowing anything about the store tier, and it is why the
    ``StoreBundle`` is kept on the instance rather than thrown away.
    """
    if bundle is None:
        bundle = build_stores(settings)
        owned: StoreBundle | None = bundle
    else:
        owned = None
    backend = resolve_backend(settings) if settings is not None else bundle.backend
    if backend == BACKEND_POSTGRES or isinstance(bundle.database, PostgresDatabase):
        store: JobStore = PostgresJobStore(
            bundle.database, lease_seconds=lease_seconds, owned_bundle=owned
        )
    else:
        store = SqliteJobStore(
            bundle.database, lease_seconds=lease_seconds, owned_bundle=owned
        )
    store.initialize()
    return store


# --------------------------------------------------------------------------
# Pipelines
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class JobStep:
    """One stage: the state it occupies, and the work that happens there.

    ``required`` decides what a failure means. A required step that fails ends
    the job as ``failed``; an optional one that fails or degrades ends it as
    ``partial``. That flag is the only difference between "this produced
    nothing" and "this produced something with a hole in it", and it is data
    rather than a rule inferred from the step's name, because a step that
    cleans up after a failure is a different kind of step from one that is
    load-bearing.
    """

    state: str
    run: Callable[[JobContext], Any]
    required: bool = True
    label: str = ""

    def __post_init__(self) -> None:
        if self.state not in RUNNING_STATES:
            raise ValueError(
                f"步骤状态 {self.state!r} 不是运行态"
                f"（可用值：{', '.join(sorted(RUNNING_STATES))}）。"
                "queued 不是步骤状态：任务在进入第一个步骤之前一直处于 queued。"
            )

    @property
    def message(self) -> str:
        return self.label or STATE_MESSAGES.get(self.state, "")


def passthrough_step(state: str, reason: str, *, required: bool = False) -> JobStep:
    """A stage that is declared and does nothing, and says why.

    The plan writes the fallback stage as optional, and the state machine here
    has no optional edge — so a pipeline that does not need cloud escalation
    still enters ``fallback`` and this is what it finds there. The alternative,
    a graph with a ``quality_check → normalizing`` shortcut, would let a
    handler that forgot the quality gate pass silently, and "was this document
    graded?" is not a question the system should answer by inference.
    """

    def _skip(context: JobContext) -> None:
        raise StepSkipped(reason)

    return JobStep(state=state, label=reason, required=required, run=_skip)


@dataclass(frozen=True, slots=True)
class JobPipeline:
    """An ordered, validated sequence of stages for one kind of job.

    The contiguity check at construction is the point. A pipeline whose states
    are not a contiguous run of :data:`JOB_SPINE` could only be run by making
    a transition the graph forbids, so it would fail on its first job — at
    runtime, under a user, possibly after a cloud parse had already been paid
    for. Rejecting it here means the mistake is a stack trace in a test.

    ``kind`` is the routing key: a worker claims jobs of its own kind, so
    ``document_analysis`` and ``reindex`` can have different pipelines and
    different workers without either one running the other's stages.
    """

    kind: str
    steps: tuple[JobStep, ...]

    def __post_init__(self) -> None:
        if not str(self.kind or "").strip():
            raise ValueError("任务管线必须有 kind。")
        if not self.steps:
            raise ValueError(f"任务管线 {self.kind} 至少需要一个步骤。")
        positions = [JOB_SPINE.index(step.state) for step in self.steps]
        # `strict=False` is not sloppiness: the second list is a *suffix* of the
        # first by construction, so the length difference is the point. The
        # pairs walked are the ones that need to be adjacent.
        for earlier, later, before, after in zip(
            positions, positions[1:], self.steps, self.steps[1:], strict=False
        ):
            if later != earlier + 1:
                raise ValueError(
                    f"任务管线 {self.kind} 的阶段不连续：{before.state} 之后不能是 "
                    f"{after.state}。步骤必须是 {JOB_SPINE} 的一段连续切片"
                    "（可以用 passthrough_step 显式占位）。"
                )

    @property
    def states(self) -> tuple[str, ...]:
        return tuple(step.state for step in self.steps)

    @property
    def first_state(self) -> str:
        return self.steps[0].state


@dataclass(slots=True)
class JobContext:
    """What a step is handed, and the only things it is allowed to do.

    A step gets the job's scratch, a way to publish progress, and a boundary
    check. It does *not* get the store's whole surface, and that is deliberate:
    a handler that can call ``transition`` can leave the state machine from
    the inside, and the enforcement this task is about only means something if
    the pipeline's own transitions are the only ones reachable from a step.
    """

    job: Job
    store: JobStore
    worker_id: str
    lease_seconds: float
    #: The reasons this job degraded, kept here rather than only on the job so
    #: a re-read after a transition cannot quietly reset them.
    partial_reasons: list[str] = field(default_factory=list)
    #: The state of the stage currently running, so a trail line written from
    #: inside a step names the stage rather than guessing.
    step_state: str = STATE_QUEUED

    # -- progress ------------------------------------------------------------

    #: The only fields a stage may write. ``state`` is not among them, and
    #: neither is anything to do with the lease or the attempt count: a stage
    #: that could move the job would be able to leave the state machine from
    #: the inside, and every assertion about illegal transitions assumes the
    #: pipeline is the only thing moving the job.
    WRITABLE = ("context", "result", "message", "document_id")

    def update(self, **changes: Any) -> None:
        """Merge this stage's contribution into the job's scratch.

        ``context`` and ``result`` are *merged*, not replaced, because the
        point of the scratch is the hand-off: the parser leaves the parsed
        document for the indexer, the indexer leaves its counts for the
        analyzer, and neither should have to know what the other wrote. A
        replacing setter would make every stage's output a function of the
        stages before it having not clobbered anything, which is not a
        property anyone would design on purpose.

        In memory, not written: a step that is halfway through three hundred
        chunks should not be three hundred transactions, and the scratch is
        worthless if the job dies before the end. The next :meth:`heartbeat`
        or :meth:`boundary` persists it — which is also exactly the point at
        which a crash loses it, honestly.
        """
        for name, value in changes.items():
            if name not in self.WRITABLE:
                raise AttributeError(
                    f"步骤不能直接写入 Job.{name}"
                    f"（可写入：{', '.join(self.WRITABLE)}）。"
                )
            current = getattr(self.job, name)
            if isinstance(current, dict) and isinstance(value, dict):
                merged = dict(current)
                merged.update(value)
                setattr(self.job, name, merged)
            else:
                setattr(self.job, name, value)

    def mark_partial(self, reason: str) -> None:
        """Record that this job produced something and also a problem.

        The job will end ``partial`` whatever the remaining stages do, because
        a later stage succeeding does not un-drop an earlier one. Collected as
        reasons rather than as a boolean so the terminal state can say *what*
        was affected, which is the difference between a message a user can act
        on and a shrug.
        """
        text = str(reason or "").strip()
        if text and text not in self.partial_reasons:
            self.partial_reasons.append(text)
        self.job.partial = True

    def log(self, message: str) -> None:
        """Append a line to the trail without moving the state.

        For the things a user would ask about that are not state changes — a
        cache hit, which parser won, whether the cloud was refused. The trail
        is the only place that survives a restart, so anything worth answering
        a question about has to go in it. Accumulated in memory and written by
        the next :meth:`heartbeat`; see :meth:`_apply_transition` for why that
        is safe.

        The sequence advances here too, and that is not cosmetic: a
        subscriber's resync cursor is a sequence number, so two entries
        sharing one would let a reconnecting client silently skip the second.
        Sequences are unique and monotonic for the life of a job, which is the
        whole property :class:`JobSubscription` rests on.
        """
        self.job.sequence += 1
        self.job.events = list(self.job.events) + [
            JobEvent(
                sequence=self.job.sequence,
                job_id=self.job.job_id,
                from_state=self.step_state,
                to_state=self.job.state,
                at=_now_iso(),
                message=str(message or ""),
                note="note",
                attempt=self.job.attempts,
            )
        ]

    # -- the boundary --------------------------------------------------------

    def heartbeat(self) -> Job:
        """Publish progress and extend the lease. Cheap; call it often.

        The lease is a dead-worker detector, and a detector that stops being
        fed is a detector that will eventually report a live worker as dead. A
        step that runs for minutes — which in this project means one blocked
        on a cloud parse or an LLM call — calls this while it waits, and the
        cost is one small row write.

        No-op on a terminal job: a lease on a ``completed`` job would be a
        lie about who is holding it, and the recovery query would then keep
        fetching a row it has to discard.
        """
        if self.job.terminal:
            return self.job
        now = time.time()
        changes: dict[str, Any] = {
            "lease_owner": self.worker_id,
            "lease_expires_at": now + max(MIN_LEASE_SECONDS, float(self.lease_seconds)),
            "context": dict(self.job.context),
            "result": dict(self.job.result),
            "message": self.job.message,
            "partial": bool(self.job.partial),
            "partial_reasons": list(self.partial_reasons),
            "events": list(self.job.events),
            "sequence": int(self.job.sequence),
        }
        self.job = self.store.patch(self.job.job_id, changes)
        return self.job

    def boundary(self) -> Job:
        """The cancellable boundary: extend the lease, then honour a cancel.

        This is the whole of cooperative cancellation, and it is worth being
        precise about what it is not: **it cannot interrupt work in progress.**
        A step that has already entered a cloud parse will finish that parse.
        The check is placed where a step is between units of work, which is the
        only place a cooperative check can go without a signal handler, a
        thread kill or a parser that can be aborted — none of which exist in
        the libraries this project calls, and all of which would leave a
        half-written index or an unpaid-for parse behind.

        The honest consequence, and it is tested: a cancel that arrives during
        the *final* step finds no boundary left to check, and the job reaches
        ``completed``. Reporting that as ``cancelled`` would be a lie about
        work that was actually finished.
        """
        self.heartbeat()
        if self.job.cancel_requested:
            raise JobCancelled(f"任务 {self.job.job_id} 收到取消请求，在可取消边界停止。")
        return self.job


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------


class JobWorker:
    """One in-process worker: claim, run, recover, repeat.

    A thread in the application process, because decision D2 says the personal
    tier starts no service. There is no ``run_forever`` a caller has to
    remember to start and a separate ``stop`` that can be forgotten, because a
    queue nobody started is a queue that silently does nothing: :meth:`start`
    is idempotent, and :meth:`drain` runs the same loop synchronously for a
    test, a script or a one-shot command.

    **Every tick recovers before it claims.** Not occasionally — every tick.
    A worker that starts and finds an orphaned job would leave it orphaned
    until the next restart, and "the queue is stuck until you reboot" is the
    failure the release gate names. Recovery is one indexed statement and it
    makes the queue self-healing within one poll interval.

    There is deliberately no worker-level retry cap. How many times a job may
    be attempted is a property of the job — re-indexing a large document and
    refreshing a metadata row do not deserve the same budget — so it is set per
    submission, and a worker-wide default would be a second place for that
    number to live.
    """

    def __init__(
        self,
        store: JobStore,
        pipeline: JobPipeline,
        *,
        worker_id: str | None = None,
        lease_seconds: float | None = None,
        poll_interval: float | None = None,
        settings: Any = None,
    ) -> None:
        self.store = store
        self.pipeline = pipeline
        self.worker_id = worker_id or default_worker_id(settings)
        # Explicit argument, then the settings object, then the shipped
        # default — in that order, so a test can shorten a lease without
        # monkeypatching a module another task owns.
        self.lease_seconds = float(
            lease_seconds
            if lease_seconds is not None
            else _setting(settings, "JOBS_LEASE_SECONDS")
        )
        self.poll_interval = float(
            poll_interval
            if poll_interval is not None
            else _setting(settings, "JOBS_POLL_INTERVAL_SECONDS")
        )
        # Retention is the worker's to enforce, because the worker is the only
        # thing in the process that runs on a schedule nobody has to remember.
        # Stored rather than read at sweep time so the value is fixed for the
        # life of the worker, and so a sweep cannot change schedule halfway.
        self.retention_days = int(
            _setting(settings, "JOBS_RETENTION_DAYS") or 0
        )
        self._last_sweep = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lifecycle_lock = threading.RLock()

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Start the background thread. Idempotent, and never blocks.

        The thread is a daemon, so a test that forgets to stop it cannot hang
        the suite; and ``stop()`` is idempotent, so a shutdown path that calls
        it twice is not a bug. A worker that is already running is left alone
        rather than replaced, because two threads claiming from one pipeline is
        the double-run this task exists to prevent.
        """
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._loop, name=f"job-worker-{self.worker_id}", daemon=True
            )
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Ask the loop to finish the step it is in, then join. Idempotent.

        ``timeout`` is a bound on waiting for a *running step*, not on
        interrupting one. A worker stopped mid-parse leaves the job leased and
        the next worker recovers it — which is the same path a crash takes, and
        is tested as one.
        """
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    def __enter__(self) -> JobWorker:
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                ran = self.tick() is not None
            except Exception:  # noqa: BLE001 - the loop must outlive one job
                logger.exception("任务 worker 循环异常，循环继续运行。")
                ran = False
            if ran:
                # Drain the backlog before sleeping. One job per poll interval
                # would make a queue of twenty take ten seconds to start, which
                # a user submitting twenty documents would read as the system
                # ignoring them.
                continue
            self._stop.wait(self.poll_interval)

    # -- the work ------------------------------------------------------------

    def tick(self) -> Job | None:
        """Recover, settle queued cancels, then claim and run one job."""
        self.store.recover_stale()
        self.store.apply_queued_cancellations()
        self.sweep_if_due()
        return self.run_once()

    def sweep_if_due(self, *, now: float | None = None) -> int:
        """Run the retention sweep at most once per retention window.

        Called from the tick, but rate-limited: a ``DELETE`` on every poll
        would mean a write transaction every ``poll_interval`` forever, on a
        queue that is idle 99% of the time, to delete nothing. The interval is
        the retention window itself rather than a separate knob — sweeping
        hourly when the window is fourteen days only buys a slightly fresher
        answer to a question nobody is asking, and a shorter period than the
        window can do is bounded by that same window's own deadline.

        The first tick sweeps immediately rather than waiting a full window,
        so a process that starts on a deployment whose last process had not run
        in weeks clears the backlog it inherited rather than keeping it for
        another fortnight.
        """
        moment = time.monotonic() if now is None else float(now)
        # At least a poll interval apart, so two workers (one per kind) racing
        # the same queue do not both sweep the same rows on the same tick, and
        # at most one retention window apart, so the deadline is never missed
        # by more than the window it is measured in.
        period = max(float(self.poll_interval), float(self.retention_days) * 86400.0)
        if self._last_sweep and moment - self._last_sweep < period:
            return 0
        self._last_sweep = moment
        return self.store.sweep_terminal(retention_days=self.retention_days)

    def run_once(self) -> Job | None:
        """Claim one job of this pipeline's kind and run it to a terminal state.

        Returns the finished job, or ``None`` when the queue was empty. The
        whole pipeline runs inside this call — there is no partially-run job
        left in memory, which is what makes the process crash mid-job the only
        way a job can be interrupted, and what makes recovery's job easy: the
        database is the entire record of where the work had got to.
        """
        claimed = self.store.claim_next(
            self.worker_id, self.lease_seconds, kind=self.pipeline.kind
        )
        if claimed is None:
            return None
        return self._run_claimed(claimed)

    def drain(self, timeout: float = 30.0) -> int:
        """Run until the queue is empty. Returns how many jobs finished.

        For tests, scripts and a one-shot CLI. It is the same loop the thread
        runs, called synchronously, so a test that drains is testing the real
        path rather than a stand-in for it.
        """
        deadline = time.monotonic() + float(timeout)
        finished = 0
        while time.monotonic() < deadline:
            if self.store.recover_stale() or self.store.apply_queued_cancellations():
                continue
            if self.run_once() is None:
                return finished
            finished += 1
        logger.warning("drain 在 %.1fs 内未清空队列，已完成 %d 个任务。", timeout, finished)
        return finished

    def wait_for(self, job_id: str, timeout: float = 30.0, interval: float = 0.02) -> Job | None:
        """Block until a job reaches a terminal state. Returns it, or ``None``.

        A test helper that lives here rather than in the test file, because
        "wait for the job" is something every future test of this subsystem
        needs and re-implementing it in each one is how a test suite starts
        inventing its own synchronisation.
        """
        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            job = self.store.get_job(job_id)
            if job is not None and job.terminal:
                return job
            time.sleep(max(0.001, float(interval)))
        return self.store.get_job(job_id)

    def _run_claimed(self, job: Job) -> Job:
        """The pipeline, the boundaries, and the four ways it can end."""
        if job.cancel_requested:
            # The claim happened, the work did not. A user who cancelled a
            # job that had not started still gets `cancelled`, not a job that
            # ran a moment and then stopped.
            return _settle(
                self.store,
                job.job_id,
                STATE_CANCELLED,
                message="任务在开始执行前被取消。",
                note="cancelled_before_start",
                expected_state=STATE_QUEUED,
            )

        context = JobContext(
            job=job,
            store=self.store,
            worker_id=self.worker_id,
            lease_seconds=self.lease_seconds,
            step_state=self.pipeline.first_state,
        )
        for step in self.pipeline.steps:
            try:
                # Entering a stage *is* the boundary before it: the lease is
                # extended and any pending cancel is honoured before the work
                # starts, not after it finishes.
                context.boundary()
            except JobCancelled as exc:
                return _settle(
                    self.store,
                    job.job_id,
                    STATE_CANCELLED,
                    message=str(exc),
                    note="cancelled_at_boundary",
                    expected_state=context.job.state,
                )
            _settle(
                self.store,
                job.job_id,
                step.state,
                message=step.message,
                note="step_enter",
                expected_state=context.job.state,
            )
            # Re-read after the transition, so the step sees the row as stored
            # rather than the copy the claim returned and the boundary patched.
            stored = self.store.get_job(job.job_id)
            if stored is not None:
                context.job = stored
            context.partial_reasons = list(context.job.partial_reasons)
            context.step_state = step.state

            settled = self._run_step(context, step)
            if settled is not None:
                return settled
            # Publish whatever the step accumulated. Written per stage, not
            # per field, so a step doing thousands of small updates costs one
            # transaction — and a crash loses the stage, not the whole run.
            context.heartbeat()

        target = STATE_PARTIAL if context.job.partial else STATE_COMPLETED
        return _settle(
            self.store,
            job.job_id,
            target,
            message=(
                f"任务部分完成：{'；'.join(context.partial_reasons)}"
                if context.job.partial
                else STATE_MESSAGES[STATE_COMPLETED]
            ),
            note="partial" if context.job.partial else "completed",
            expected_state=context.job.state,
            patch={"partial_reasons": list(context.partial_reasons)},
        )

    def _run_step(self, context: JobContext, step: JobStep) -> Job | None:
        """Run one stage. Returns the settled job if the job ended here."""
        store = self.store
        job_id = context.job.job_id
        state = context.job.state
        try:
            step.run(context)
        except JobCancelled as exc:
            return _settle(
                store,
                job_id,
                STATE_CANCELLED,
                message=str(exc),
                note="cancelled_by_step",
                expected_state=state,
            )
        except StepSkipped as exc:
            # Recorded, and the job carries on. A stage that decided there was
            # nothing to do is an outcome, not an absence: "the cloud fallback
            # was considered and refused" and "the cloud fallback never ran"
            # are different answers to a question users do ask.
            context.log(f"{step.state}：跳过（{exc.message}）")
        except StepDegraded as exc:
            context.mark_partial(f"{step.state}：{exc.message}")
        except StepFailed as exc:
            if step.required:
                return _settle(
                    store,
                    job_id,
                    STATE_FAILED,
                    message=f"{step.state} 失败：{exc.message}",
                    note="step_failed",
                    expected_state=state,
                    error=exc.message,
                    retryable=exc.retryable,
                    cloud_outcome=exc.cloud_outcome,
                )
            context.mark_partial(f"{step.state}：{exc.message}")
        except Exception as exc:  # noqa: BLE001 - one bad job is not a dead worker
            logger.exception("任务 %s 的步骤 %s 抛出未处理异常。", job_id, step.state)
            return _settle(
                store,
                job_id,
                STATE_FAILED,
                message=f"{step.state} 抛出未处理异常：{exc}",
                note="step_raised",
                expected_state=state,
                error=str(exc),
                retryable=False,
            )
        return None


def _settle(
    store: JobStore,
    job_id: str,
    target: str,
    *,
    message: str,
    note: str,
    expected_state: str | None = None,
    error: str = "",
    retryable: bool = False,
    cloud_outcome: str = "",
    patch: Mapping[str, Any] | None = None,
) -> Job:
    """Move a job to a terminal state and settle its verdict, in one write.

    The verdict fields ride in the same ``patch`` as the state change, so a
    crash between the two cannot leave a ``failed`` job with no reason attached
    — which is a failed job whose cause nobody can find, on exactly the run
    where finding it mattered.

    ``cloud_outcome`` is threaded rather than left for the caller to patch in,
    so that it is written by :func:`_terminal_patch` in the same transaction as
    ``error_message``. A caller that patched it separately could interleave a
    crash between the two writes and leave a failed job naming a cloud refusal
    it never actually reported.
    """
    merged = dict(patch or {})
    merged.update(
        _terminal_patch(
            target, error=error, retryable=retryable, cloud_outcome=cloud_outcome
        )
    )
    return store.transition(
        job_id,
        target,
        message=message,
        note=note,
        expected_state=expected_state,
        patch=merged,
    )


def recover_stale_jobs(store: JobStore, *, now: float | None = None) -> list[Job]:
    """Free-function form of :meth:`JobStore.recover_stale`, for a cold start.

    An operator who restarts the app and wants to know what happened to the
    queue before any worker exists calls this. It is the release gate's
    sentence, as a function someone can call in a shell.
    """
    return store.recover_stale(now=now)


# --------------------------------------------------------------------------
# Subscriptions
# --------------------------------------------------------------------------


class JobSubscription:
    """A cursor over one job's state, and nothing else.

    The design question this class answers is what a client does when the
    connection drops in the middle of a run. The answer it makes possible is:
    come back, pass the last sequence it saw, and read the rest. That works
    only because the feed is state — every transition is a row, numbered, and
    the numbers do not repeat or skip — so a client that missed three events
    can say exactly which ones.

    The obvious alternative, a broadcast queue in the worker's memory, is
    cheaper and useless: a client that was not connected at the moment a
    transition happened has no way to learn that it happened, which is the
    failure the plan calls "刷新/断线不丢任务". Polling a table is how a
    subscriber survives the process it was watching being restarted.

    ``poll_interval`` defaults to a fifth of a second, and the loop is
    interruptible through ``stop_event`` so a request handler that is closing
    a connection does not have to wait it out.
    """

    def __init__(
        self,
        store: JobStore,
        job_id: str,
        *,
        since_sequence: int = 0,
        poll_interval: float = 0.2,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.store = store
        self.job_id = str(job_id)
        self.cursor = int(since_sequence)
        self.poll_interval = float(poll_interval)
        self.stop_event = stop_event

    def snapshot(self) -> dict[str, Any] | None:
        """The current state, as one state dict, or ``None`` if no such job.

        Delivered first on every (re)connect, so a client that arrives late
        knows where things stand before it starts reading transitions. It does
        not advance the cursor: a snapshot is an answer to "where am I", not an
        event, and counting it as one would make every later sequence off by
        one for a client that had not seen the submission event.
        """
        job = self.store.get_job(self.job_id)
        return job.to_state_dict() if job is not None else None

    def poll(self) -> list[dict[str, Any]]:
        """Every transition since the last call, oldest first.

        Advances the cursor, so two consecutive calls partition the trail with
        no gap and no duplicate. Returns state dicts only.
        """
        events = self.store.events_since(self.job_id, self.cursor)
        if events:
            self.cursor = events[-1].sequence
        return [event.to_state_dict() for event in events]

    def stream(self) -> Iterator[dict[str, Any]]:
        """Yield state dicts until the job reaches a terminal state.

        A terminal event ends the stream. A job that is *already* terminal
        yields its remaining events — which for a fresh subscription is the
        whole trail — and then stops, so a client that reconnects after the
        fact still learns how the run ended instead of hanging on a job that
        will never move again.
        """
        while True:
            events = self.poll()
            for event in events:
                yield event
                if event.get("terminal"):
                    return
            if self.stop_event is not None and self.stop_event.is_set():
                return
            if not events:
                time.sleep(self.poll_interval)


__all__ = [
    "JOB_SPINE",
    "JOB_STATES",
    "JOB_TRANSITIONS",
    "RECOVERY_EDGES",
    "REQUIRED_CONFIG_KEYS",
    "RUNNING_STATES",
    "STATE_ANALYZING",
    "STATE_CANCELLED",
    "STATE_COMPLETED",
    "STATE_FAILED",
    "STATE_FALLBACK",
    "STATE_INDEXING",
    "STATE_MESSAGES",
    "STATE_NORMALIZING",
    "STATE_PARTIAL",
    "STATE_PARSING",
    "STATE_QUALITY_CHECK",
    "STATE_QUEUED",
    "TERMINAL_STATES",
    "CancelAck",
    "IllegalTransition",
    "Job",
    "JobCancelled",
    "JobConflict",
    "JobContext",
    "JobError",
    "JobEvent",
    "JobNotFound",
    "JobPipeline",
    "JobStep",
    "JobStore",
    "JobSubscription",
    "JobWorker",
    "MIN_LEASE_SECONDS",
    "PostgresJobStore",
    "SqliteJobStore",
    "StepDegraded",
    "StepError",
    "StepFailed",
    "StepSkipped",
    "SubmitResult",
    "assert_transition",
    "build_job_store",
    "can_transition",
    "default_worker_id",
    "is_running",
    "is_terminal",
    "passthrough_step",
    "recover_stale_jobs",
]
