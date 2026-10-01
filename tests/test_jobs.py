"""The job system, and the machinery that lets one suite run against both tiers.

T2.1's acceptance criterion is not "the worker works". It is "worker 重启后
任务恢复或进入可重试终态" — a worker that dies mid-job leaves a job the next
worker picks up, and a job stuck running forever is impossible. Those are
statements about durable state, so the tests that pin them have to survive
the process being killed rather than a fixture being torn down. Hence three
shapes of test in this file:

``JobStoreContract``
    Behaviour every tier must exhibit, with no tier in its name and no ``if``
    on the backend. It is the definition of "the two tiers are the same
    product", and anything it has to bend for is a bug in one of them. T1.1
    established the arrangement and the reason for it: two hand-copied files
    drift within a week, and the acceptance criterion then keeps passing on the
    tier nobody is running. ``TestJobSqlite`` and ``TestJobPostgres`` are
    *empty* subclasses of it — empty on purpose, because a subclass that adds a
    test is a test only one tier runs.
``test_recovery_*``
    A worker that is genuinely killed. Not a mock, not a simulated crash: a
    child process runs a real worker against a real database file and calls
    ``os._exit`` in the middle of a stage, so nothing unwinds, no ``finally``
    runs and no test fixture gets the chance to tidy up. The parent then asserts
    what the corpse left behind and what the next worker does about it.
``everything else``
    The state machine, the worker loop, cancellation, idempotency and the
    state-only subscription, on the personal tier with a real SQLite file.

The Hosted tier skips cleanly and says what to do.
``PAPERWHISPERER_TEST_PG_DSN`` is the switch; without it — and there is no
server and no driver on this machine — the PostgreSQL class is skipped rather
than failed, with a reason naming the variable, because a skip nobody can act
on is a test that stopped being run.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest

from paperwhisperer.infra import jobs as J
from paperwhisperer.infra.stores import (
    TIER_HOSTED,
    TIER_PERSONAL,
    StoreBundle,
    build_stores,
)

#: Set this to run the Hosted half of every contract test.
#:
#:     docker compose up -d db
#:     PAPERWHISPERER_TEST_PG_DSN=postgresql://paperwhisperer:paperwhisperer@localhost:5432/
#:         paperwhisperer /home/ubuntu/miniconda3/bin/python3 -m pytest tests/test_jobs.py
#:
#: ``docker compose up -d db`` in this repository brings up exactly that
#: database, with pgvector already installed.
PG_DSN_ENV = "PAPERWHISPERER_TEST_PG_DSN"

#: Where the project lives, for the child process that gets killed below. It
#: imports the package by absolute path rather than relying on inheriting this
#: process's ``sys.path``, because a subprocess started from a test should not
#: depend on how the test itself happened to be launched.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: How far into the future to look when deciding a lease has expired. Tests do
#: not sleep for a lease to lapse: they ask the question with a clock, because a
#: test that sleeps to prove a deadline passed is a test that is either slow or
#: flaky, and usually both.
FUTURE = 3600.0


# --------------------------------------------------------------------------
# Settings stand-in
# --------------------------------------------------------------------------


@dataclass
class _Settings:
    """The subset of configuration the store and job factories read.

    A private object rather than monkeypatching
    :mod:`paperwhisperer.core.config`, which another task owns and which a
    test in this file must not be reaching into. Every field carries the
    shipped default, so a test that overrides one thing is still exercising
    real defaults for everything else.
    """

    STORAGE_BACKEND: str = "sqlite"
    STORAGE_SQLITE_PATH: str = "context/stores/paperwhisperer.sqlite3"
    STORAGE_SQLITE_TIMEOUT_SECONDS: int = 30
    STORAGE_LEXICAL_FTS5: str = "auto"
    STORAGE_PG_DSN: str = ""
    STORAGE_PG_SCHEMA: str = "public"
    STORAGE_PG_CONNECT_TIMEOUT_SECONDS: int = 5
    STORAGE_PG_ANN: bool = False


# --------------------------------------------------------------------------
# Hosted-tier availability
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def postgres_availability() -> tuple[bool, str]:
    """Whether the Hosted tier can be reached, and why not if it cannot.

    Checked once per session, and only when the DSN environment variable is
    set. Probing a database nobody asked for would make an offline test suite
    slow in a way that reads as a hang, and the reason string has to be useful:
    someone reading a CI log should be able to fix the skip without opening this
    file.
    """
    dsn = os.environ.get(PG_DSN_ENV, "").strip()
    if not dsn:
        return False, (
            f"未设置 {PG_DSN_ENV}。运行 docker compose up -d db 后设置 "
            f"{PG_DSN_ENV}=postgresql://paperwhisperer:paperwhisperer@localhost:5432/"
            "paperwhisperer 即可启用 Hosted 档位测试。"
        )
    try:
        module = importlib.import_module("paperwhisperer.infra.stores.postgres_backend")
    except RuntimeError as exc:
        return False, f"PostgreSQL 驱动不可用：{exc}"
    database = module.PostgresDatabase(dsn, connect_timeout=5)
    try:
        with database.cursor() as cursor:
            cursor.execute("SELECT 1")
    except Exception as exc:  # noqa: BLE001 - any failure means "not reachable"
        return False, f"无法连接 {PG_DSN_ENV}={dsn}：{exc}"
    finally:
        database.close()
    return True, dsn


PG_AVAILABLE, PG_REASON = postgres_availability()


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------


@dataclass
class JobHarness:
    """One tier's job store, plus the teardown that belongs to it."""

    tier: str
    store: Any
    bundle: StoreBundle
    _cleanup: list[Any] = field(default_factory=list)

    def close(self) -> None:
        # Cleanups FIRST: they hold a reference to the bundle's own database
        # and need a live connection to drop their schema. Closing the bundle
        # before them raises "database connection is closed" during teardown,
        # which reads as a product failure and is not one — and it is invisible
        # on the personal tier, whose cleanup touches only a temp directory.
        for action in reversed(self._cleanup):
            action()
        self.store.close()
        self.bundle.close()


def open_store(tier: str, tmp_path: Any, name: str = "jobs.db") -> JobHarness:
    """Build an isolated job store for ``tier``.

    ``name`` is the database file for the personal tier, and the recovery tests
    need it because they hand the same path to a child process — a child
    pointed at a different file would find an empty queue and never die, which
    is a very confusing way to fail.

    Isolation is per test. The personal tier gets its own file under
    ``tmp_path``; the Hosted tier gets its own PostgreSQL *schema*, dropped on
    teardown — a schema rather than a table, because a run creates a table and
    four indexes and ``DROP TABLE`` would leave the indexes to collide with the
    next test's names.

    The bundle is built and handed over rather than letting
    :func:`~paperwhisperer.infra.jobs.build_job_store` build one, because
    passing a bundle in is the supported way to share the connection — and
    using it here means the contract tests exercise that path rather than a
    convenience branch.
    """
    if tier == TIER_HOSTED:
        schema = f"job_test_{uuid.uuid4().hex[:12]}"
        bundle = build_stores(
            _Settings(
                STORAGE_BACKEND="postgres",
                STORAGE_PG_DSN=os.environ[PG_DSN_ENV],
                STORAGE_PG_SCHEMA=schema,
            )
        )
        harness = JobHarness(tier=tier, bundle=bundle, store=J.build_job_store(bundle=bundle))
        database = bundle.database

        def _drop() -> None:
            with database.cursor() as cursor:
                cursor.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")

        harness._cleanup.append(_drop)
        return harness

    bundle = build_stores(
        _Settings(STORAGE_BACKEND="sqlite", STORAGE_SQLITE_PATH=str(tmp_path / name))
    )
    return JobHarness(tier=tier, bundle=bundle, store=J.build_job_store(bundle=bundle))


class JobStoreContract:
    """Behaviour every tier must exhibit.

    No engine name appears in any assertion here, and that is the point: a
    contract that had to tolerate an engine difference would be a record of the
    two tiers disagreeing, which is the thing this class exists to prevent.
    """

    #: Overridden by the two empty subclasses below.
    tier: str = TIER_PERSONAL

    @pytest.fixture
    def store(self, tmp_path):
        if self.tier == TIER_HOSTED and not PG_AVAILABLE:
            pytest.skip(PG_REASON)
        harness = open_store(self.tier, tmp_path)
        try:
            yield harness.store
        finally:
            harness.close()

    # -- submission ----------------------------------------------------------

    def test_submit_returns_a_queued_job_with_a_submission_trail_entry(self, store):
        result = store.submit("document_analysis", {"file_path": "/tmp/a.pdf"})
        job = result.job
        assert not result.deduplicated
        assert job.state == J.STATE_QUEUED
        assert job.kind == "document_analysis"
        assert job.payload == {"file_path": "/tmp/a.pdf"}
        assert job.attempts == 0
        assert [event.note for event in job.events] == ["submitted"]
        # A trail entry at all, not an empty list: a subscriber that attaches
        # to a brand-new job has to be able to see that it was created.
        assert job.events[0].sequence == job.sequence
        assert store.get_job(job.job_id).to_dict() == job.to_dict()

    def test_a_second_submission_with_the_same_key_returns_the_first_job(self, store):
        first = store.submit("document_analysis", {"file_path": "a.pdf"}, idempotency_key="k")
        second = store.submit("document_analysis", {"file_path": "a.pdf"}, idempotency_key="k")
        assert second.deduplicated
        assert second.job_id == first.job_id
        assert store.count_by_state()[J.STATE_QUEUED] == 1

    def test_a_duplicate_submission_is_visible_as_a_duplicate(self, store):
        store.submit("document_analysis", {"file_path": "a.pdf"}, idempotency_key="k")
        duplicate = store.submit("document_analysis", {"file_path": "a.pdf"}, idempotency_key="k")
        assert duplicate.deduplicated is True
        # The point of the flag: a response byte-identical to a fresh
        # submission would tell a user who double-clicked nothing at all.
        assert "重复提交" in duplicate.notice
        assert duplicate.notice != ""
        assert duplicate.to_dict()["deduplicated"] is True

    def test_a_duplicate_is_not_counted_as_a_new_job_or_a_new_attempt(self, store):
        first = store.submit("k", idempotency_key="same")
        store.claim_next("w1", 60, kind="k")
        store.claim_next("w2", 60, kind="k")
        again = store.submit("k", idempotency_key="same")
        assert again.job_id == first.job_id
        # The job is in flight and was claimed exactly once. A second claim is
        # the thing that would pay for a second cloud parse.
        assert again.job.attempts == 1
        assert again.job.state == J.STATE_QUEUED

    def test_an_empty_idempotency_key_does_not_deduplicate(self, store):
        first = store.submit("k", idempotency_key="")
        second = store.submit("k", idempotency_key="")
        assert not second.deduplicated
        assert second.job_id != first.job_id

    def test_different_keys_produce_different_jobs(self, store):
        first = store.submit("k", idempotency_key="a")
        second = store.submit("k", idempotency_key="b")
        assert not second.deduplicated
        assert first.job_id != second.job_id

    def test_a_duplicate_submission_after_completion_still_reuses_the_job(self, store):
        first = store.submit("k", idempotency_key="done")
        store.transition(first.job_id, J.STATE_COMPLETED)
        again = store.submit("k", idempotency_key="done")
        # A key means "this exact work". Same key, same job, whatever state it
        # is in — otherwise a user who resubmits a failed document pays twice
        # for a failure they already paid for once.
        assert again.deduplicated
        assert again.job.state == J.STATE_COMPLETED

    # -- the state machine ---------------------------------------------------

    def test_a_completed_job_cannot_return_to_parsing(self, store):
        job = store.submit("k").job
        store.transition(job.job_id, J.STATE_COMPLETED)
        with pytest.raises(J.IllegalTransition):
            store.transition(job.job_id, J.STATE_PARSING)
        assert store.get_job(job.job_id).state == J.STATE_COMPLETED

    def test_a_cancelled_job_cannot_resume(self, store):
        job = store.submit("k").job
        store.transition(job.job_id, J.STATE_CANCELLED)
        for target in (J.STATE_QUEUED, J.STATE_PARSING, J.STATE_COMPLETED):
            with pytest.raises(J.IllegalTransition):
                store.transition(job.job_id, target)

    def test_a_partial_job_is_not_retryable_by_transition(self, store):
        job = store.submit("k").job
        store.transition(job.job_id, J.STATE_PARTIAL)
        with pytest.raises(J.IllegalTransition):
            store.transition(job.job_id, J.STATE_QUEUED)
        with pytest.raises(J.IllegalTransition):
            store.transition(job.job_id, J.STATE_ANALYZING)

    def test_a_running_job_may_not_skip_a_stage(self, store):
        job = store.submit("k").job
        store.transition(job.job_id, J.STATE_PARSING)
        with pytest.raises(J.IllegalTransition):
            store.transition(job.job_id, J.STATE_INDEXING)
        assert store.get_job(job.job_id).state == J.STATE_PARSING

    def test_a_job_may_enter_the_spine_at_a_later_stage(self, store):
        # A re-index has no parse to do, so the graph has to allow entering at
        # `indexing` without a special case for re-index.
        job = store.submit("reindex").job
        assert store.transition(job.job_id, J.STATE_INDEXING).state == J.STATE_INDEXING

    def test_an_illegal_transition_is_rejected_before_anything_is_written(self, store):
        job = store.submit("k").job
        before = store.get_job(job.job_id).to_dict()
        with pytest.raises(J.IllegalTransition):
            store.transition(job.job_id, J.STATE_FALLBACK, message="不应该写入")
        # Not even the message reached the row: a refused write that half
        # succeeded would leave a job that says something the state machine
        # never allowed.
        assert store.get_job(job.job_id).to_dict() == before

    def test_transitioning_a_job_that_does_not_exist_raises(self, store):
        with pytest.raises(J.JobNotFound):
            store.transition("no-such-job", J.STATE_PARSING)

    def test_a_transition_rejects_a_state_that_moved_underneath_the_caller(self, store):
        job = store.submit("k").job
        store.transition(job.job_id, J.STATE_PARSING)
        # A second writer — the cancel, arriving mid-step — has the row now.
        store.transition(job.job_id, J.STATE_QUALITY_CHECK)
        with pytest.raises(J.JobConflict):
            store.transition(job.job_id, J.STATE_FALLBACK, expected_state=J.STATE_PARSING)
        assert store.get_job(job.job_id).state == J.STATE_QUALITY_CHECK

    # -- claiming ------------------------------------------------------------

    def test_claiming_leases_the_job_without_moving_it(self, store):
        job = store.submit("k").job
        claimed = store.claim_next("worker-1", 60, kind="k")
        assert claimed.job_id == job.job_id
        # Still `queued`: a claimed job is one that is waiting until the worker
        # enters its first stage, and conflating the two would make "claimed"
        # and "running" indistinguishable to a user.
        assert claimed.state == J.STATE_QUEUED
        assert claimed.lease_owner == "worker-1"
        assert claimed.lease_expires_at > 0
        assert claimed.is_leased
        assert claimed.attempts == 1
        # A second worker must not be able to take it: the job is off the queue
        # even though its state still says `queued`, and taking it again would
        # run the same work — and bill the same cloud parse — twice.
        assert store.claim_next("worker-2", 60, kind="k") is None

    def test_claiming_drains_the_queue_in_a_total_order(self, store):
        store.submit("k", {"n": 1})
        store.submit("k", {"n": 2})
        first = store.claim_next("worker-1", 60, kind="k")
        second = store.claim_next("worker-1", 60, kind="k")
        # The order is (created_at, job_id), which is a *total* order rather
        # than submission order: two jobs created in the same second — the
        # normal case on the personal tier — are separated by their ids. What
        # the contract promises is a deterministic order, and that both tiers
        # produce the same one, which is what stops a paginated queue from
        # meaning two different things depending on where it runs.
        assert (first.created_at, first.job_id) <= (second.created_at, second.job_id)
        assert {first.job_id, second.job_id} == {
            job.job_id for job in store.list_jobs(kind="k")
        }
        assert store.claim_next("worker-1", 60, kind="k") is None

    def test_claiming_only_takes_the_kind_it_was_asked_for(self, store):
        store.submit("reindex")
        assert store.claim_next("worker-1", 60, kind="document_analysis") is None
        assert store.claim_next("worker-1", 60, kind="reindex") is not None

    def test_a_claim_records_the_attempt_in_the_trail(self, store):
        store.submit("k")
        claimed = store.claim_next("worker-1", 60, kind="k")
        assert claimed.events[-1].note == "claimed"
        assert claimed.events[-1].attempt == 1
        assert claimed.events[-1].changed is False
        assert claimed.started_at

    # -- cancellation --------------------------------------------------------

    def test_cancelling_a_queued_job_settles_it_without_a_worker(self, store):
        job = store.submit("k").job
        ack = store.request_cancel(job.job_id)
        assert ack.accepted
        assert ack.job.cancel_requested
        # Nothing has to cooperate when nothing has started, so the queue
        # settles it on the next pass rather than letting a worker claim a job
        # the user already stopped.
        settled = store.apply_queued_cancellations()
        assert [item.job_id for item in settled] == [job.job_id]
        assert store.get_job(job.job_id).state == J.STATE_CANCELLED

    def test_cancelling_a_running_job_only_sets_the_flag(self, store):
        job = store.submit("k").job
        store.claim_next("worker-1", 60, kind="k")
        store.transition(job.job_id, J.STATE_PARSING)
        ack = store.request_cancel(job.job_id)
        assert ack.accepted
        # The flag, and nothing else: a running job is stopped by the worker
        # reaching a boundary, not by this call reaching into its state.
        assert store.get_job(job.job_id).state == J.STATE_PARSING
        assert store.get_job(job.job_id).cancel_requested
        assert "边界" in ack.message

    def test_cancelling_a_finished_job_says_it_was_too_late(self, store):
        job = store.submit("k").job
        store.transition(job.job_id, J.STATE_COMPLETED)
        ack = store.request_cancel(job.job_id)
        assert ack.accepted is False
        assert "终态" in ack.message
        assert ack.to_dict()["accepted"] is False

    def test_cancelling_twice_is_not_an_error(self, store):
        job = store.submit("k").job
        first = store.request_cancel(job.job_id)
        second = store.request_cancel(job.job_id)
        assert first.accepted and second.accepted
        assert second.job.cancel_requested_at == first.job.cancel_requested_at

    def test_cancelling_a_job_that_does_not_exist_raises(self, store):
        with pytest.raises(J.JobNotFound):
            store.request_cancel("no-such-job")

    def test_settling_queued_cancellations_leaves_unrelated_jobs_alone(self, store):
        cancelled = store.submit("k").job
        other = store.submit("k").job
        store.request_cancel(cancelled.job_id)
        store.apply_queued_cancellations()
        assert store.get_job(other.job_id).state == J.STATE_QUEUED
        assert store.apply_queued_cancellations() == []

    # -- recovery ------------------------------------------------------------

    def test_an_expired_lease_requeues_the_job_and_clears_the_owner(self, store):
        job = store.submit("k", max_attempts=3).job
        store.claim_next("dead-worker", 60, kind="k")
        store.transition(job.job_id, J.STATE_PARSING)
        moved = store.recover_stale(now=time.time() + FUTURE)
        recovered = store.get_job(job.job_id)
        assert [item.job_id for item in moved] == [job.job_id]
        assert recovered.state == J.STATE_QUEUED
        assert recovered.lease_owner == ""
        assert recovered.lease_expires_at == 0
        # The attempt it already burned stands: a job that is being retried
        # must not look like it never ran, or the retry cap is meaningless.
        assert recovered.attempts == 1
        assert "dead-worker" in recovered.message

    def test_an_expired_lease_after_the_last_attempt_fails_retryably(self, store):
        job = store.submit("k", max_attempts=1).job
        store.claim_next("dead-worker", 60, kind="k")
        store.transition(job.job_id, J.STATE_PARSING)
        store.recover_stale(now=time.time() + FUTURE)
        exhausted = store.get_job(job.job_id)
        # The other half of the release gate: recovery *or* an explicit
        # retryable terminal state. This is the second one.
        assert exhausted.state == J.STATE_FAILED
        assert exhausted.retryable is True
        assert "中断" in exhausted.error_message

    def test_recovery_releases_a_lease_held_on_a_job_that_never_started(self, store):
        job = store.submit("k").job
        store.claim_next("dead-worker", 60, kind="k")
        store.recover_stale(now=time.time() + FUTURE)
        released = store.get_job(job.job_id)
        # Claimed and abandoned before the first stage. Leaving it leased would
        # block the queue behind a lease nobody holds, which is the same
        # "stuck forever" failure in a quieter costume.
        assert released.state == J.STATE_QUEUED
        assert released.lease_owner == ""
        assert released.events[-1].note.startswith("lease_expired")
        assert store.claim_next("worker-2", 60, kind="k").job_id == job.job_id

    def test_recovery_ignores_a_live_lease(self, store):
        job = store.submit("k").job
        store.claim_next("live-worker", 60, kind="k")
        store.transition(job.job_id, J.STATE_PARSING)
        assert store.recover_stale(now=time.time()) == []
        assert store.get_job(job.job_id).state == J.STATE_PARSING

    def test_recovery_never_touches_an_unclaimed_queued_job(self, store):
        store.submit("k")
        assert store.recover_stale(now=time.time() + FUTURE) == []
        assert store.count_by_state()[J.STATE_QUEUED] == 1

    def test_recovery_never_resurrects_a_finished_job(self, store):
        job = store.submit("k").job
        store.claim_next("worker-1", 60, kind="k")
        store.transition(job.job_id, J.STATE_COMPLETED)
        assert store.recover_stale(now=time.time() + FUTURE) == []
        finished = store.get_job(job.job_id)
        assert finished.state == J.STATE_COMPLETED
        # And no lease is left on it. A terminal row that still claims to be
        # held by a worker is a lie about who is working on it, and the
        # recovery query would keep fetching a row it has to discard.
        assert finished.is_leased is False
        assert store.count_by_state()[J.STATE_COMPLETED] == 1

    def test_retry_puts_a_retryable_failure_back_and_resets_the_attempt_count(self, store):
        job = store.submit("k", max_attempts=1).job
        store.claim_next("worker-1", 60, kind="k")
        store.transition(job.job_id, J.STATE_FAILED, patch={"retryable": True})
        retried = store.retry(job.job_id)
        assert retried.state == J.STATE_QUEUED
        # The reset is the whole difference between a retry and a recovery, and
        # it is why `retry` is a method and not just another transition.
        assert retried.attempts == 0
        assert retried.retryable is False
        assert retried.finished_at == ""

    def test_retry_refuses_a_job_that_did_not_fail(self, store):
        job = store.submit("k").job
        store.transition(job.job_id, J.STATE_COMPLETED)
        with pytest.raises(J.IllegalTransition):
            store.retry(job.job_id)

    def test_a_failure_keeps_its_cloud_outcome_next_to_the_error_message(self, store):
        """Why a cloud reason is persisted, and why it must survive the write.

        Five different cloud endings used to reach a browser as one sentence, so
        a reader who ran out of quota was told to try again later and a
        document an administrator had refused to let leave the machine was
        told the same thing again. The machine-readable half has to be *stored*,
        not reconstructed by whoever reads the job, because the two are written
        together and a reader that derives one from the other has to guess.

        So this asserts both halves land in the same write and both come back.
        The values are the ones ingestion actually produces, plus one this
        module has never heard of: the store must carry whatever it is given
        rather than validating against a list that lives three modules away,
        because a value the store silently drops is a value the reader is told
        is absent — which is the generic code, and the bug.
        """
        for outcome in ("quota_exhausted", "timed_out", "blocked", "failed", "partial",
                        "success", "a-future-outcome-nobody-predicted", ""):
            job = store.submit("k").job
            store.claim_next("worker-1", 60, kind="k")
            store.transition(job.job_id, J.STATE_PARSING, expected_state=J.STATE_QUEUED)
            store.transition(
                job.job_id,
                J.STATE_FAILED,
                message="cloud refused",
                patch={
                    "retryable": True,
                    "error_message": "云端解析被拒绝。",
                    "cloud_outcome": outcome,
                },
            )
            stored = store.get_job(job.job_id)
            assert stored.cloud_outcome == outcome
            # The two are independent fields, not one derived from the other:
            # a failure with a reason and no message is still readable, and a
            # message with no reason must not acquire one.
            assert stored.error_message == "云端解析被拒绝。"

    def test_a_failure_with_no_cloud_reason_reports_none_rather_than_guessing(self, store):
        job = store.submit("k").job
        store.claim_next("worker-1", 60, kind="k")
        store.transition(job.job_id, J.STATE_PARSING, expected_state=J.STATE_QUEUED)
        store.transition(job.job_id, J.STATE_FAILED, message="本地解析失败。")
        stored = store.get_job(job.job_id)
        assert stored.state == J.STATE_FAILED
        # A transition that names no cloud reason carries none. `transition` is
        # the low-level write and does not invent a verdict — the worker settles
        # one through `_terminal_patch`, and that is where a reason would come
        # from — so "absent" here has to stay absent.
        assert stored.cloud_outcome == ""

    def test_a_requeued_job_no_longer_carries_the_previous_cloud_reason(self, store):
        """The stale-value guard, and the reason retry touches this field at all.

        ``retry`` clears ``error_message``; clearing only half of the verdict
        would leave a job sitting in the queue still named by the failure it was
        requeued to escape. A reader polling that job would be told it ran out
        of quota before it has run a second time.
        """
        job = store.submit("k", max_attempts=1).job
        store.claim_next("worker-1", 60, kind="k")
        store.transition(
            job.job_id,
            J.STATE_FAILED,
            patch={
                "retryable": True,
                "error_message": "云端配额已用尽。",
                "cloud_outcome": "quota_exhausted",
            },
        )
        retried = store.retry(job.job_id)
        assert retried.state == J.STATE_QUEUED
        assert retried.error_message == ""
        assert retried.cloud_outcome == ""

    # -- the trail and the resync cursor -------------------------------------

    def test_the_trail_is_ordered_and_its_sequences_are_unique(self, store):
        job = store.submit("k").job
        store.claim_next("worker-1", 60, kind="k")
        store.transition(job.job_id, J.STATE_PARSING)
        store.transition(job.job_id, J.STATE_COMPLETED)
        stored = store.get_job(job.job_id)
        sequences = [event.sequence for event in stored.events]
        assert sequences == sorted(sequences)
        # Unique, not merely sorted: a resync cursor is a sequence number, and
        # two entries sharing one lets a reconnecting client skip the second.
        assert len(sequences) == len(set(sequences))
        assert stored.sequence == sequences[-1]

    def test_events_since_returns_exactly_what_a_resuming_client_missed(self, store):
        job = store.submit("k").job
        store.claim_next("worker-1", 60, kind="k")
        first = store.events_since(job.job_id, 0)
        store.transition(job.job_id, J.STATE_PARSING)
        store.transition(job.job_id, J.STATE_COMPLETED)
        seen = first[-1].sequence
        missed = store.events_since(job.job_id, seen)
        assert [event.to_state for event in missed] == [J.STATE_PARSING, J.STATE_COMPLETED]
        assert store.events_since(job.job_id, store.get_job(job.job_id).sequence) == []

    def test_events_since_an_unknown_job_is_empty(self, store):
        assert store.events_since("no-such-job", 0) == []

    def test_patch_writes_without_moving_the_state_or_adding_an_entry(self, store):
        job = store.submit("k").job
        before = len(store.get_job(job.job_id).events)
        patched = store.patch(job.job_id, {"context": {"chunks": 12}})
        assert patched.state == J.STATE_QUEUED
        assert patched.context == {"chunks": 12}
        # No event: nothing a subscriber watches moved, and waking a subscriber
        # for every scratch write would mean waking it constantly.
        assert len(store.get_job(job.job_id).events) == before
        assert store.get_job(job.job_id).context == {"chunks": 12}

    # -- listing -------------------------------------------------------------

    def test_list_jobs_filters_and_orders_newest_first(self, store):
        store.submit("k", {"n": 1})
        store.submit("other", {"n": 2})
        listed = store.list_jobs()
        # Newest first, and total. The tiebreak is not cosmetic: two rows
        # written in the same second are the normal case on the personal tier,
        # and without a total order the same query pages differently on the
        # two tiers — a pagination bug that only reproduces in production.
        by_time = sorted(listed, key=lambda job: job.created_at, reverse=True)
        assert listed == sorted(by_time, key=lambda job: job.job_id)
        assert [job.kind for job in store.list_jobs(kind="k")] == ["k"]
        assert len(store.list_jobs(state=J.STATE_QUEUED)) == 2
        assert store.list_jobs(state=J.STATE_COMPLETED) == []
        assert store.list_jobs(document_id="doc-9") == []

    def test_list_jobs_pages_deterministically(self, store):
        for index in range(5):
            store.submit("k", {"n": index})
        everything = [job.job_id for job in store.list_jobs()]
        first_page = [job.job_id for job in store.list_jobs(limit=2)]
        second_page = [job.job_id for job in store.list_jobs(limit=2, offset=2)]
        # A total order, not a partial one. Five jobs submitted in the same
        # second is the normal case on the personal tier, so the tiebreak is
        # what decides the page — and it has to be the same tiebreak on both
        # tiers or a paginated queue differs between them.
        assert first_page + second_page == everything[:4]
        assert not set(first_page) & set(second_page)
        assert [job.job_id for job in store.list_jobs()] == everything
        assert store.list_jobs(limit=0) == []

    def test_count_by_state_reports_every_state_including_the_empty_ones(self, store):
        counts = store.count_by_state()
        assert set(counts) == set(J.JOB_STATES)
        assert all(value == 0 for value in counts.values())
        store.submit("k")
        assert store.count_by_state()[J.STATE_QUEUED] == 1

    def test_get_job_returns_none_for_an_unknown_id(self, store):
        assert store.get_job("no-such-job") is None

    # -- round trip ----------------------------------------------------------

    def test_the_job_survives_the_row_intact(self, store):
        payload = {"file_path": "/tmp/a.pdf", "options": {"force_ocr": True}, "n": [1, 2]}
        job = store.submit("k", payload, document_id="doc-1", session_id="s-1").job
        store.claim_next("worker-1", 60, kind="k")
        store.patch(job.job_id, {"context": {"parsed": {"pages": 10}}})
        stored = store.get_job(job.job_id)
        # A document payload with CJK in it and a nested list, byte for byte.
        # A job that came back equivalent-but-not-equal would mean the
        # serialiser and the rehydrator had drifted apart.
        assert stored.payload == payload
        assert stored.context == {"parsed": {"pages": 10}}
        assert stored.document_id == "doc-1"
        assert stored.session_id == "s-1"
        # Through a full JSON round trip, so the assertion is about the
        # serialiser and not about a copy that never left memory.
        assert J.Job.from_dict(json.loads(json.dumps(stored.to_dict()))).to_dict() == stored.to_dict()

    def test_a_subscriber_projection_carries_no_payload(self, store):
        secret = "SENTINEL-DO-NOT-STREAM"
        job = store.submit("k", {"file_path": secret}).job
        store.transition(job.job_id, J.STATE_COMPLETED, patch={"result": {"answer": secret}})
        state = store.get_job(job.job_id).to_state_dict()
        assert secret not in json.dumps(state, ensure_ascii=False)
        # The whole point of the projection: state answers "where am I", and a
        # client that also got the payload would have to decide what to do
        # with half a stream.
        assert "payload" not in state
        assert "result" not in state
        assert "context" not in state


class TestJobSqlite(JobStoreContract):
    """The personal tier. Empty on purpose — see the module docstring."""

    tier = TIER_PERSONAL


@pytest.mark.skipif(not PG_AVAILABLE, reason=PG_REASON)
class TestJobPostgres(JobStoreContract):
    """The Hosted tier. Empty on purpose — see the module docstring."""

    tier = TIER_HOSTED


# --------------------------------------------------------------------------
# The state machine, on its own
# --------------------------------------------------------------------------


class TestStateMachine:
    """The graph, independent of any store.

    These are the assertions that would be false if someone edited
    :data:`~paperwhisperer.infra.jobs.JOB_TRANSITIONS` to make a test pass,
    which is exactly why they are here rather than only in the contract.
    """

    def test_the_spine_is_the_chain_the_plan_writes(self):
        # verbatim from T2.1: queued → parsing → quality_check → fallback? →
        # normalizing → indexing → analyzing
        assert J.JOB_SPINE == (
            "queued",
            "parsing",
            "quality_check",
            "fallback",
            "normalizing",
            "indexing",
            "analyzing",
        )

    def test_the_four_outcomes_are_terminal_and_the_spines_states_are_not(self):
        assert {"completed", "partial", "failed", "cancelled"} == J.TERMINAL_STATES
        for state in J.RUNNING_STATES:
            assert not J.is_terminal(state)
            assert J.is_running(state)
        for state in J.TERMINAL_STATES:
            assert J.is_terminal(state)
            assert not J.is_running(state)
        # `queued` is neither: it is where a job waits, and it is the one
        # state a worker is allowed to leave without having done anything.
        assert not J.is_running(J.STATE_QUEUED)
        assert not J.is_terminal(J.STATE_QUEUED)

    @pytest.mark.parametrize("state", ["completed", "partial", "cancelled"])
    def test_a_finished_job_has_no_way_out(self, state):
        assert J.JOB_TRANSITIONS[state] == frozenset()
        for target in J.JOB_STATES:
            assert not J.can_transition(state, target)

    def test_only_failed_may_be_requeued_directly(self):
        # Every other terminal is final, so "retry" is a different action and
        # not a spelling of "run it again".
        assert J.JOB_TRANSITIONS["failed"] == frozenset({"queued"})
        for edge in J.RECOVERY_EDGES:
            source, target = edge
            assert target == "queued"
            assert source in J.TERMINAL_STATES or source in J.RUNNING_STATES
        assert ("failed", "queued") in J.RECOVERY_EDGES
        for state in J.RUNNING_STATES:
            assert (state, "queued") in J.RECOVERY_EDGES

    @pytest.mark.parametrize(
        ("source", "target"),
        [
            ("completed", "parsing"),
            ("partial", "queued"),
            ("cancelled", "queued"),
            ("parsing", "indexing"),
            ("quality_check", "analyzing"),
            # `queued` may not jump *into* the middle of the spine's optional
            # stage, though it may start at a later one. The asymmetry is
            # deliberate: a pipeline declares where it begins, but nothing may
            # arrive in `fallback` without a quality verdict to justify it.
            ("queued", "fallback"),
        ],
    )
    def test_illegal_edges_are_refused(self, source, target):
        with pytest.raises(J.IllegalTransition):
            J.assert_transition(source, target)

    def test_every_running_state_can_reach_every_outcome(self):
        # A pipeline is a prefix of the spine, so its last stage has to be able
        # to finish. If a future edit dropped one of these, a one-stage re-index
        # pipeline would raise at its first transition.
        for state in J.JOB_SPINE:
            assert J.JOB_TRANSITIONS[state] >= J.TERMINAL_STATES

    def test_an_unknown_state_is_refused_rather_than_guessed(self):
        assert not J.can_transition("running", "parsing")
        with pytest.raises(J.IllegalTransition):
            J.assert_transition("running", "parsing")

    def test_every_state_the_code_can_write_is_in_the_table(self):
        assert set(J.JOB_SPINE) | J.TERMINAL_STATES == set(J.JOB_TRANSITIONS)
        assert set(J.JOB_TRANSITIONS) == set(J.JOB_STATES)

    def test_the_illegal_transition_message_names_both_states(self):
        with pytest.raises(J.IllegalTransition) as caught:
            J.assert_transition("completed", "parsing", job_id="abc123")
        message = str(caught.value)
        assert "completed" in message and "parsing" in message
        assert "abc123" in message


class TestPipelines:
    """A pipeline is validated when it is built, not when it misbehaves."""

    def test_a_pipeline_that_skips_a_stage_is_rejected_when_it_is_built(self):
        with pytest.raises(ValueError, match="不连续"):
            J.JobPipeline(
                kind="document_analysis",
                steps=(
                    J.JobStep(state=J.STATE_PARSING, run=lambda ctx: None),
                    J.JobStep(state=J.STATE_INDEXING, run=lambda ctx: None),
                ),
            )

    def test_a_pipeline_may_start_at_a_later_stage(self):
        # Re-indexing has no parse to do. Rather than a special case in the
        # graph, the pipeline declares its entry point.
        pipeline = J.JobPipeline(
            kind="reindex",
            steps=(J.JobStep(state=J.STATE_INDEXING, run=lambda ctx: None),),
        )
        assert pipeline.states == ("indexing",)
        assert pipeline.first_state == "indexing"

    def test_a_pipeline_needs_a_kind_and_at_least_one_step(self):
        with pytest.raises(ValueError, match="kind"):
            J.JobPipeline(kind="", steps=(J.JobStep(state=J.STATE_PARSING, run=lambda ctx: None),))
        with pytest.raises(ValueError, match="至少需要一个步骤"):
            J.JobPipeline(kind="k", steps=())

    def test_a_step_cannot_be_queued(self):
        with pytest.raises(ValueError, match="不是运行态"):
            J.JobStep(state=J.STATE_QUEUED, run=lambda ctx: None)

    def test_a_step_is_handed_a_context_and_not_the_store(self):
        # The enforcement of the graph, where it actually matters: a stage can
        # publish progress and ask for a boundary, and has no way at all to
        # move the job. A step that could call `transition` could leave the
        # state machine from the inside, and none of the assertions about
        # illegal transitions would mean anything.
        assert not hasattr(J.JobContext, "transition")
        assert not hasattr(J.JobContext, "claim_next")
        assert not hasattr(J.JobContext, "recover_stale")
        assert hasattr(J.JobContext, "boundary")
        assert hasattr(J.JobContext, "heartbeat")

    def test_a_step_cannot_reach_the_private_attributes_of_the_job(self, worker_store):
        captured: list[str] = []

        def probe(context: J.JobContext) -> None:
            with pytest.raises(AttributeError):
                context.update(state=J.STATE_COMPLETED)
            with pytest.raises(AttributeError):
                context.update(lease_owner="me")
            captured.append(context.job.job_id)

        store = worker_store
        pipeline = J.JobPipeline(
            kind="k", steps=(J.JobStep(state=J.STATE_PARSING, run=probe),)
        )
        job = store.submit("k").job
        J.JobWorker(store, pipeline, lease_seconds=30).run_once()
        assert captured == [job.job_id]
        assert store.get_job(job.job_id).state == J.STATE_COMPLETED


# --------------------------------------------------------------------------
# The worker
# --------------------------------------------------------------------------


@pytest.fixture
def worker_store(tmp_path):
    """A personal-tier job store on a real file, isolated per test."""
    harness = open_store(TIER_PERSONAL, tmp_path)
    try:
        yield harness.store
    finally:
        harness.close()


def states_of(store, job_id: str) -> list[str]:
    """The state changes a client would have seen, in order."""
    return [event.to_state for event in store.events_since(job_id, 0) if event.changed]


def document_pipeline(**overrides: Any) -> J.JobPipeline:
    """The six-stage document pipeline, with each stage's body replaceable.

    Built from ``passthrough_step`` for every stage a test does not care about,
    so each test's pipeline stays one line long and the spine is not restated
    in a dozen places. ``overrides`` maps a state to the body to run there.
    """
    bodies: dict[str, Any] = {
        J.STATE_PARSING: lambda ctx: ctx.update(context={"parsed": True}),
        J.STATE_QUALITY_CHECK: lambda ctx: ctx.log("质量评估：PASS"),
        J.STATE_FALLBACK: None,
        J.STATE_NORMALIZING: lambda ctx: ctx.update(context={"chunks": 12}),
        J.STATE_INDEXING: lambda ctx: ctx.update(context={"vectors": 12}),
        J.STATE_ANALYZING: lambda ctx: ctx.update(result={"analysis_id": "run-1"}),
    }
    bodies.update(overrides)
    steps = tuple(
        J.passthrough_step(state, "该阶段无需执行。")
        if bodies[state] is None
        else J.JobStep(state=state, run=bodies[state])
        for state in J.JOB_SPINE[1:]
    )
    return J.JobPipeline(kind="document_analysis", steps=steps)


class TestWorker:
    def test_a_failed_stage_that_names_a_cloud_outcome_records_it_on_the_job(
        self, worker_store
    ):
        """The whole path, not just the store: StepFailed → _settle → the row.

        The store contract proves the field survives a write; this proves a real
        stage can actually *set* it. That gap matters because without it the
        field would be persisted correctly and always empty — which is
        indistinguishable, from the reader's side, from the bug it was added to
        fix.
        """

        def refuse(context):
            raise J.StepFailed(
                "云端配额已用尽。", retryable=True, cloud_outcome="quota_exhausted"
            )

        store = worker_store
        job = store.submit("document_analysis").job
        J.JobWorker(
            store,
            document_pipeline(**{J.STATE_PARSING: refuse}),
            lease_seconds=30,
        ).drain()

        finished = store.get_job(job.job_id)
        assert finished.state == J.STATE_FAILED
        assert finished.cloud_outcome == "quota_exhausted"
        # The prose the user reads and the code the front end branches on are
        # two views of one verdict, written in the same transaction.
        assert finished.error_message == "云端配额已用尽。"
        assert finished.retryable is True

    def test_a_stage_that_fails_locally_reports_no_cloud_outcome(self, worker_store):
        """Absent must mean absent, not "the last thing that went wrong".

        A stage with no cloud involvement leaves the field empty. If it
        inherited a reason from somewhere, a purely local failure would be
        reported to the user as though a provider had refused the document.
        """

        def fail_locally(context):
            raise J.StepFailed("本地解析失败。", retryable=False)

        store = worker_store
        job = store.submit("document_analysis").job
        J.JobWorker(
            store, document_pipeline(**{J.STATE_PARSING: fail_locally}), lease_seconds=30
        ).drain()
        finished = store.get_job(job.job_id)
        assert finished.state == J.STATE_FAILED
        assert finished.cloud_outcome == ""
        assert finished.error_message == "本地解析失败。"

    def test_a_clean_run_walks_the_whole_spine_and_completes(self, worker_store):
        store = worker_store
        job = store.submit("document_analysis", {"file_path": "a.pdf"}).job
        assert J.JobWorker(store, document_pipeline(), lease_seconds=30).drain() == 1
        finished = store.get_job(job.job_id)
        assert finished.state == J.STATE_COMPLETED
        assert finished.terminal
        assert finished.attempts == 1
        assert finished.finished_at
        assert states_of(store, job.job_id) == [
            J.STATE_PARSING,
            J.STATE_QUALITY_CHECK,
            J.STATE_FALLBACK,
            J.STATE_NORMALIZING,
            J.STATE_INDEXING,
            J.STATE_ANALYZING,
            J.STATE_COMPLETED,
        ]
        # The scratch each stage handed to the next one survived.
        assert finished.context == {"parsed": True, "chunks": 12, "vectors": 12}
        assert finished.result == {"analysis_id": "run-1"}

    def test_the_optional_fallback_stage_is_entered_and_records_why_it_did_nothing(
        self, worker_store
    ):
        # This is the plan's `fallback?`. The stage is always entered, and it
        # says in the trail that the escalation was not needed — which is the
        # difference between "the cloud was considered and refused" and "the
        # cloud stage never ran". The graph has no optional edge precisely so
        # that a handler cannot skip the quality gate by accident.
        store = worker_store
        job = store.submit("document_analysis").job
        J.JobWorker(store, document_pipeline(), lease_seconds=30).run_once()
        finished = store.get_job(job.job_id)
        assert J.STATE_FALLBACK in states_of(store, job.job_id)
        assert finished.state == J.STATE_COMPLETED
        skipped = [event for event in finished.events if event.note == "note" and "跳过" in event.message]
        assert any("fallback" in event.message for event in skipped)

    def test_a_degraded_stage_makes_the_job_partial(self, worker_store):
        def warn(ctx: J.JobContext) -> None:
            ctx.update(context={"parsed": True})
            raise J.StepDegraded("解析质量为 WARN，部分引用可能无法定位")

        store = worker_store
        job = store.submit("document_analysis").job
        J.JobWorker(
            store, document_pipeline(**{J.STATE_PARSING: warn}), lease_seconds=30
        ).run_once()
        finished = store.get_job(job.job_id)
        # `partial` and not `failed`: the document was parsed, indexed and
        # analyzed, and reporting that as a failure is how partial success
        # becomes a false one.
        assert finished.state == J.STATE_PARTIAL
        assert finished.partial is True
        assert finished.retryable is False
        assert any("WARN" in reason for reason in finished.partial_reasons)
        assert "部分完成" in finished.message
        assert finished.result == {"analysis_id": "run-1"}

    def test_an_optional_stage_that_fails_degrades_instead_of_failing(self, worker_store):
        def optional_broken(ctx: J.JobContext) -> None:
            raise J.StepFailed("可选增强不可用")

        pipeline = J.JobPipeline(
            kind="document_analysis",
            steps=(
                J.JobStep(state=J.STATE_PARSING, run=lambda ctx: ctx.update(context={"p": 1})),
                J.JobStep(state=J.STATE_QUALITY_CHECK, run=optional_broken, required=False),
                J.JobStep(state=J.STATE_FALLBACK, run=lambda ctx: None),
            ),
        )
        store = worker_store
        job = store.submit("document_analysis").job
        J.JobWorker(store, pipeline, lease_seconds=30).run_once()
        finished = store.get_job(job.job_id)
        assert finished.state == J.STATE_PARTIAL
        assert finished.partial_reasons
        # The pipeline carried on: the stage after the broken one ran.
        assert J.STATE_FALLBACK in states_of(store, job.job_id)

    def test_a_required_stage_that_fails_fails_the_job(self, worker_store):
        def broken(ctx: J.JobContext) -> None:
            raise J.StepFailed("无法连接云端解析服务", retryable=True)

        store = worker_store
        job = store.submit("document_analysis").job
        J.JobWorker(
            store, document_pipeline(**{J.STATE_PARSING: broken}), lease_seconds=30
        ).run_once()
        finished = store.get_job(job.job_id)
        assert finished.state == J.STATE_FAILED
        assert finished.retryable is True
        assert finished.error_message == "无法连接云端解析服务"
        # The stages after the failure were not entered: running the indexer
        # over a document that was never parsed would only produce noise.
        assert J.STATE_INDEXING not in states_of(store, job.job_id)
        assert J.STATE_COMPLETED not in states_of(store, job.job_id)

    def test_an_unexpected_exception_fails_the_job_and_does_not_kill_the_worker(
        self, worker_store
    ):
        store = worker_store
        pipeline = document_pipeline(
            **{J.STATE_PARSING: lambda ctx: (_ for _ in ()).throw(RuntimeError("boom"))}
        )
        first = store.submit("document_analysis").job
        J.JobWorker(store, pipeline, lease_seconds=30).run_once()
        assert store.get_job(first.job_id).state == J.STATE_FAILED
        assert "boom" in store.get_job(first.job_id).error_message
        # The worker is still usable: one poisonous job is not a dead queue.
        second = store.submit("document_analysis").job
        J.JobWorker(store, document_pipeline(), lease_seconds=30).run_once()
        assert store.get_job(second.job_id).state == J.STATE_COMPLETED

    def test_the_worker_drains_a_backlog_without_sleeping_between_jobs(self, worker_store):
        store = worker_store
        ids = [store.submit("document_analysis").job.job_id for _ in range(5)]
        assert J.JobWorker(store, document_pipeline(), lease_seconds=30).drain() == 5
        assert {store.get_job(job_id).state for job_id in ids} == {J.STATE_COMPLETED}

    def test_drain_returns_zero_on_an_empty_queue(self, worker_store):
        assert J.JobWorker(worker_store, document_pipeline()).drain() == 0

    def test_a_background_worker_runs_jobs_and_stops_cleanly(self, worker_store):
        store = worker_store
        job = store.submit("document_analysis").job
        with J.JobWorker(store, document_pipeline(), lease_seconds=30, poll_interval=0.01) as worker:
            finished = worker.wait_for(job.job_id, timeout=10.0)
        assert finished.state == J.STATE_COMPLETED
        worker.stop()
        worker.stop()  # idempotent: a shutdown path may call it twice

    def test_wait_for_reports_a_job_that_is_still_running(self, worker_store):
        store = worker_store
        job = store.submit("document_analysis").job
        assert J.JobWorker(store, document_pipeline()).wait_for(job.job_id, timeout=0.1).state == (
            J.STATE_QUEUED
        )


# --------------------------------------------------------------------------
# The configuration contract with core/config.py
# --------------------------------------------------------------------------


class TestConfiguration:
    """``REQUIRED_CONFIG_KEYS`` is a promise to another module, so it is tested.

    :mod:`paperwhisperer.core.config` is owned by another task and is not
    monkeypatched from here. This module reads every key with ``getattr`` and a
    documented default, which is what lets the job queue work on a checkout
    nobody has configured — the state the personal tier is required to work in.
    The table is the handover, so the promise it makes is worth an assertion.
    """

    def test_every_documented_key_has_a_working_default(self):
        for key in (
            "JOBS_ENABLED",
            "JOBS_POLL_INTERVAL_SECONDS",
            "JOBS_LEASE_SECONDS",
            "JOBS_MAX_ATTEMPTS",
            "JOBS_WORKER_ID",
            "JOBS_RETENTION_DAYS",
        ):
            assert key in J.REQUIRED_CONFIG_KEYS
        # A settings object with no job keys at all is the shipping state, and
        # a worker built from it must still run.
        class _Empty:
            pass

        assert J.JobWorker(None, document_pipeline()).lease_seconds == (
            J.REQUIRED_CONFIG_KEYS["JOBS_LEASE_SECONDS"]
        )
        assert J.default_worker_id(_Empty())
        assert J.REQUIRED_CONFIG_KEYS["JOBS_MAX_ATTEMPTS"] >= 1

    def test_a_configured_worker_id_is_used(self):
        class _Settings:
            JOBS_WORKER_ID = "operator-chosen"
            JOBS_LEASE_SECONDS = 12.0
            JOBS_POLL_INTERVAL_SECONDS = 0.25

        worker = J.JobWorker(None, document_pipeline(), settings=_Settings())
        assert worker.worker_id == "operator-chosen"
        assert worker.lease_seconds == 12.0
        assert worker.poll_interval == 0.25

    def test_an_explicit_worker_id_beats_the_configured_one(self):
        class _Settings:
            JOBS_WORKER_ID = "from-config"

        assert J.JobWorker(
            None, document_pipeline(), worker_id="explicit", settings=_Settings()
        ).worker_id == "explicit"

    def test_two_workers_in_one_process_do_not_share_an_identity(self):
        # One worker per pipeline is the normal shape, and two of them must not
        # be able to look at each other's leases and conclude they are the
        # same holder.
        first = J.JobWorker(None, document_pipeline())
        second = J.JobWorker(None, document_pipeline())
        assert first.worker_id != second.worker_id

    def test_a_job_without_a_kind_is_refused(self, worker_store):
        # A job with no kind is one no worker could ever claim, and it would
        # sit in the queue forever looking like a stuck job.
        with pytest.raises(ValueError, match="kind"):
            worker_store.submit("")
        with pytest.raises(ValueError, match="kind"):
            worker_store.submit("   ")


# --------------------------------------------------------------------------
# Idempotency under concurrency
# --------------------------------------------------------------------------


class TestIdempotencyUnderLoad:
    def test_eight_threads_submitting_the_same_key_produce_one_job(self, worker_store):
        # The check-then-insert in an API handler is a race, and this is the
        # race: eight double-clicks arriving on eight threads. Deduplication
        # lives inside the store's transaction, so the database decides, and
        # the answer is one row rather than whichever thread answered last.
        store = worker_store
        barrier = threading.Barrier(8)
        results: list[Any] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def submit() -> None:
            try:
                barrier.wait(timeout=10)
                outcome = store.submit(
                    "document_analysis", {"file_path": "a.pdf"}, idempotency_key="double-click"
                )
                with lock:
                    results.append(outcome)
            except Exception as exc:  # noqa: BLE001 - the assertion is on the outcome
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=submit) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)

        assert not errors
        assert len(results) == 8
        assert len({outcome.job_id for outcome in results}) == 1
        assert sum(1 for outcome in results if not outcome.deduplicated) == 1
        assert store.count_by_state()[J.STATE_QUEUED] == 1

    def test_a_keyed_job_is_claimed_by_exactly_one_worker(self, worker_store):
        # The other half of "重复提交去重": even if two workers wake at once,
        # one job is one claim, so a second cloud parse is not paid for.
        store = worker_store
        job = store.submit("document_analysis", idempotency_key="once").job
        barrier = threading.Barrier(4)
        claims: list[Any] = []
        lock = threading.Lock()

        def claim(index: int) -> None:
            barrier.wait(timeout=10)
            outcome = store.claim_next(f"worker-{index}", 60, kind="document_analysis")
            with lock:
                claims.append(outcome)

        threads = [threading.Thread(target=claim, args=(index,)) for index in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)

        winners = [outcome for outcome in claims if outcome is not None]
        assert [outcome.job_id for outcome in winners] == [job.job_id]
        assert store.get_job(job.job_id).attempts == 1


# --------------------------------------------------------------------------
# Cancellation
# --------------------------------------------------------------------------


class TestCancellation:
    """Cooperative, and honest about what cooperative means.

    The three tests below are the three cases there are: a cancel that lands
    before a stage starts, a cancel that lands during a stage and is honoured
    at the next boundary, and a cancel that lands during the *last* stage and
    therefore never gets a boundary to act on. The third one is the interesting
    one, and the module's promise is that it is reported as ``completed`` rather
    than dressed up as a cancellation the user did not get.
    """

    def test_a_cancel_lands_at_the_next_boundary_and_stops_the_job(self, worker_store):
        store = worker_store
        seen: list[str] = []

        def parsing(ctx: J.JobContext) -> None:
            seen.append("parsing ran")
            ctx.update(context={"parsed": True})

        def quality(ctx: J.JobContext) -> None:
            # Somebody cancels while this stage is running, and the stage asks
            # for a boundary when it next has one.
            store.request_cancel(ctx.job.job_id)
            seen.append("quality ran")
            ctx.boundary()
            seen.append("quality finished")  # unreachable; the boundary raises

        pipeline = J.JobPipeline(
            kind="document_analysis",
            steps=(
                J.JobStep(state=J.STATE_PARSING, run=parsing),
                J.JobStep(state=J.STATE_QUALITY_CHECK, run=quality),
                J.JobStep(state=J.STATE_FALLBACK, run=lambda ctx: None),
            ),
        )
        job = store.submit("document_analysis").job
        J.JobWorker(store, pipeline, lease_seconds=30).run_once()
        finished = store.get_job(job.job_id)
        assert seen == ["parsing ran", "quality ran"]
        assert finished.state == J.STATE_CANCELLED
        assert finished.cancel_requested is True
        # The stage after the cancelled one was not entered.
        assert J.STATE_FALLBACK not in states_of(store, job.job_id)

    def test_a_cancel_during_the_last_stage_completes_the_job(self, worker_store):
        store = worker_store

        def parsing(ctx: J.JobContext) -> None:
            ctx.update(context={"p": 1})

        def final(ctx: J.JobContext) -> None:
            # The request arrives while the last stage is working. There is no
            # boundary after this point — by construction, the terminal
            # transition is not a cancellation boundary, because the work *was*
            # finished and calling that "cancelled" would be a lie about it.
            store.request_cancel(ctx.job.job_id)
            ctx.update(result={"analysis_id": "run-1"})

        pipeline = J.JobPipeline(
            kind="document_analysis",
            steps=(
                J.JobStep(state=J.STATE_PARSING, run=parsing),
                J.JobStep(state=J.STATE_QUALITY_CHECK, run=final),
            ),
        )
        job = store.submit("document_analysis").job
        J.JobWorker(store, pipeline, lease_seconds=30).run_once()
        finished = store.get_job(job.job_id)
        assert finished.state == J.STATE_COMPLETED
        assert finished.result == {"analysis_id": "run-1"}
        # The flag is still on the row: the cancel was genuinely requested, it
        # simply arrived too late to do anything. Both facts are recorded,
        # because "it says cancelled but it says completed too" is at least
        # honest, and a false `cancelled` would not be.
        assert finished.cancel_requested is True

    def test_a_cancel_does_not_interrupt_work_already_in_progress(self, worker_store):
        store = worker_store
        progress: list[str] = []

        def slow(ctx: J.JobContext) -> None:
            # Stands in for a long cloud parse or a long LLM call. The request
            # is made from another thread, in the middle, and the work runs to
            # the end of the step regardless.
            def cancel_midway() -> None:
                store.request_cancel(ctx.job.job_id)

            timer = threading.Timer(0.02, cancel_midway)
            timer.start()
            for step in range(5):
                time.sleep(0.01)
                progress.append(f"chunk {step}")
            timer.join()
            # Only now, at the boundary, does the cancel take effect.
            ctx.boundary()

        pipeline = J.JobPipeline(
            kind="document_analysis", steps=(J.JobStep(state=J.STATE_PARSING, run=slow),)
        )
        job = store.submit("document_analysis").job
        J.JobWorker(store, pipeline, lease_seconds=60).run_once()
        # The honest claim of a cooperative cancel: every unit of work that had
        # started was finished. A cancellation that could interrupt this would
        # need to abort a parser mid-write, and would leave a half-indexed
        # document behind instead of a cancelled job.
        assert progress == [f"chunk {step}" for step in range(5)]
        assert store.get_job(job.job_id).state == J.STATE_CANCELLED

    def test_a_job_cancelled_while_queued_never_runs_a_stage(self, worker_store):
        store = worker_store
        ran: list[str] = []
        pipeline = document_pipeline(**{J.STATE_PARSING: lambda ctx: ran.append("ran")})
        job = store.submit("document_analysis").job
        ack = store.request_cancel(job.job_id)
        assert ack.accepted
        J.JobWorker(store, pipeline, lease_seconds=30).drain()
        assert ran == []
        assert store.get_job(job.job_id).state == J.STATE_CANCELLED

    def test_a_cancelled_job_never_resumes_after_a_restart(self, worker_store):
        store = worker_store
        job = store.submit("document_analysis").job
        store.transition(job.job_id, J.STATE_CANCELLED)
        assert J.JobWorker(store, document_pipeline(), lease_seconds=30).drain() == 0
        assert store.get_job(job.job_id).state == J.STATE_CANCELLED


# --------------------------------------------------------------------------
# Recovery: a worker that is really killed
# --------------------------------------------------------------------------

#: A worker that dies where a worker dies. It claims a real job from a real
#: database file, enters the ``parsing`` stage, writes — proving the row is
#: committed and durable — and then calls ``os._exit``. No exception, no
#: ``finally``, no ``atexit``, no graceful close: the process image simply
#: stops existing, which is the only kind of death the release gate is about.
_DYING_WORKER = '''
import os
import sys

sys.path.insert(0, {project_root!r})

from paperwhisperer.infra import jobs as J
from paperwhisperer.infra.stores import build_sqlite_bundle


class _Settings:
    STORAGE_BACKEND = "sqlite"
    STORAGE_SQLITE_PATH = sys.argv[1]
    STORAGE_SQLITE_TIMEOUT_SECONDS = 30
    STORAGE_LEXICAL_FTS5 = "auto"


def die_midway(context):
    context.update(context={{"parsed_bytes": 4096}})
    context.heartbeat()
    with open(sys.argv[2], "w", encoding="utf-8") as handle:
        handle.write(context.job.job_id)
        handle.flush()
        os.fsync(handle.fileno())
    os._exit(9)


def main() -> None:
    bundle = build_sqlite_bundle(_Settings())
    store = J.build_job_store(bundle=bundle)
    pipeline = J.JobPipeline(
        kind="document_analysis",
        steps=(J.JobStep(state=J.STATE_PARSING, run=die_midway),),
    )
    worker = J.JobWorker(store, pipeline, worker_id="doomed-worker", lease_seconds=1)
    worker.run_once()
    raise SystemExit("the worker was supposed to die")


main()
'''


def kill_a_worker_on_job(db_path: str, marker_path: Any) -> subprocess.CompletedProcess:
    """Run a worker in its own process and kill it inside the first stage.

    Returns the completed process so the caller can assert on the exit code —
    9 rather than 0 is the proof that the process died rather than returning.
    """
    script = Path(db_path).parent / "dying_worker.py"
    script.write_text(
        _DYING_WORKER.format(project_root=str(PROJECT_ROOT)), encoding="utf-8"
    )
    return subprocess.run(
        [sys.executable, str(script), str(db_path), str(marker_path)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


class TestRecoveryFromAKilledWorker:
    """The release gate: "worker/API 重启后任务可恢复或明确进入可重试终态".

    Nothing here is mocked. A real process claims a real row in a real SQLite
    file and is killed at a real point, and the assertions are about what this
    process can see afterwards.
    """

    @pytest.fixture
    def corpse(self, tmp_path):
        """A job a worker genuinely died holding, and the store that can see it.

        The parent and the child share one database *file*, which is the whole
        point: the child's writes were committed before the process stopped
        existing, and a different process can read them. A child pointed at its
        own file would find an empty queue and never die, which is a very
        confusing way to fail.
        """
        harness = open_store(TIER_PERSONAL, tmp_path, name="killed.db")
        store = harness.store
        marker = tmp_path / "reached.txt"
        job = store.submit("document_analysis", {"file_path": "a.pdf"}, max_attempts=3).job
        result = kill_a_worker_on_job(tmp_path / "killed.db", marker)
        try:
            # It really did die, and it really did get into the stage. Without
            # both of these, every assertion below could pass for the wrong
            # reason — most of all by never having exercised a crash at all.
            assert result.returncode == 9, f"子进程未按预期退出：{result.stderr}"
            assert marker.read_text(encoding="utf-8") == job.job_id
            # What the corpse left behind. A different process, a different
            # connection, the same committed bytes.
            dead = store.get_job(job.job_id)
            assert dead.state == J.STATE_PARSING
            assert dead.lease_owner == "doomed-worker"
            assert dead.lease_expires_at > 0
            assert dead.attempts == 1
            assert dead.context == {"parsed_bytes": 4096}
            yield store, job
        finally:
            harness.close()

    def test_a_killed_worker_leaves_a_job_that_claimed_by_nobody_can_see(self, corpse):
        # The state the release gate exists to prevent: a job in a running
        # state whose owner no longer exists. It is invisible to the queue —
        # nothing can claim it — which is exactly why recovery has to go
        # looking for it rather than waiting for it to come back.
        store, _job = corpse
        assert store.claim_next("healthy-worker", 60, kind="document_analysis") is None
        assert store.count_by_state()[J.STATE_PARSING] == 1

    def test_recovery_requeues_the_job_and_the_next_worker_finishes_it(self, corpse):
        store, job = corpse
        moved = store.recover_stale(now=time.time() + FUTURE)
        assert [item.job_id for item in moved] == [job.job_id]
        assert store.get_job(job.job_id).state == J.STATE_QUEUED
        # And the point of requeueing it: a fresh worker in this process picks
        # it up and carries it to the end. The job is not lost, and it is not
        # sitting waiting for a restart that may never come.
        assert J.JobWorker(store, document_pipeline(), lease_seconds=30).drain() == 1
        finished = store.get_job(job.job_id)
        assert finished.state == J.STATE_COMPLETED
        assert finished.attempts == 2
        # The trail still says who died, so a user asking "why did this take
        # twice?" gets an answer from the job itself.
        assert any("doomed-worker" in event.message for event in finished.events)

    def test_a_dead_job_is_recovered_by_the_next_tick_without_being_asked(self, worker_store):
        # The same dead state as the fixture above, with the deadline *moved*
        # rather than slept through. The killed process is what proves a real
        # crash leaves this state; this is about *where* recovery is invoked,
        # and a test that waits out a real lease to check that is a slow,
        # flaky test. Recovery is part of every tick, not a thing an operator
        # has to remember to run: a worker that started and left an orphaned
        # job alone would leave it orphaned until the next restart, which is
        # the failure this whole mechanism exists to remove.
        store = worker_store
        job = store.submit("document_analysis", max_attempts=3).job
        store.claim_next("doomed-worker", 60, kind="document_analysis")
        store.transition(job.job_id, J.STATE_PARSING)
        store.patch(job.job_id, {"lease_expires_at": time.time() - 1})
        # Nobody can claim it: it is in a running state with a live-looking
        # owner, and the queue has no way to know better.
        assert store.claim_next("healthy-worker", 60, kind="document_analysis") is None
        assert J.JobWorker(store, document_pipeline(), lease_seconds=30).tick() is not None
        finished = store.get_job(job.job_id)
        assert finished.state == J.STATE_COMPLETED
        assert finished.attempts == 2


class TestRecoveryWhenTheRetryBudgetIsGone:
    def test_a_worker_that_dies_on_the_last_attempt_ends_in_a_retryable_state(self, tmp_path):
        harness = open_store(TIER_PERSONAL, tmp_path, name="killed.db")
        try:
            store = harness.store
            job = store.submit("document_analysis", max_attempts=1).job
            result = kill_a_worker_on_job(tmp_path / "killed.db", tmp_path / "reached.txt")
            assert result.returncode == 9, result.stderr
            store.recover_stale(now=time.time() + FUTURE)
            exhausted = store.get_job(job.job_id)
            # The other half of the gate's "或": not recoverable, so explicitly
            # retryable. A user can re-run it, and nothing pretends otherwise.
            assert exhausted.state == J.STATE_FAILED
            assert exhausted.retryable is True
            assert "中断" in exhausted.error_message
            assert store.claim_next("healthy-worker", 60, kind="document_analysis") is None
            # And retryable means retryable, not merely labelled: the job runs.
            store.retry(job.job_id)
            assert J.JobWorker(store, document_pipeline(), lease_seconds=30).drain() == 1
            assert store.get_job(job.job_id).state == J.STATE_COMPLETED
        finally:
            harness.close()

    def test_a_worker_that_dies_holding_a_lease_on_a_job_it_never_started(self, tmp_path):
        # The quieter version of the same failure: claimed, then dead before
        # the first stage. Left alone it would sit leased forever and block the
        # queue behind a lease nobody holds.
        store = open_store(TIER_PERSONAL, tmp_path).store
        job = store.submit("document_analysis").job
        store.claim_next("doomed-worker", 60, kind="document_analysis")
        assert store.claim_next("other", 60, kind="document_analysis") is None
        store.recover_stale(now=time.time() + FUTURE)
        released = store.get_job(job.job_id)
        assert released.state == J.STATE_QUEUED
        assert released.lease_owner == ""
        assert store.claim_next("healthy-worker", 60, kind="document_analysis") is not None


# --------------------------------------------------------------------------
# The state-only subscription
# --------------------------------------------------------------------------


class TestSubscription:
    """SSE subscribes to state. The assertions here are about what is *absent*.

    A subscriber that also received the document text and the model output
    could not be resynced after a dropped connection, because it could not tell
    which half of the stream it had. So the test that matters is the one that
    puts a sentinel in the payload and the result and then never finds it.
    """

    SECRET = "SENTINEL-这一段不应出现在状态流里"

    def test_no_state_event_ever_carries_the_payload_or_the_result(self, worker_store):
        store = worker_store
        job = store.submit(
            "document_analysis", {"file_path": f"/tmp/{self.SECRET}.pdf"}
        ).job
        pipeline = document_pipeline(
            **{J.STATE_PARSING: lambda ctx: ctx.update(result={"answer": self.SECRET})}
        )
        with J.JobWorker(store, pipeline, lease_seconds=30) as worker:
            worker.wait_for(job.job_id, timeout=10.0)
        events = store.events_since(job.job_id, 0)
        assert events
        for event in events:
            rendered = json.dumps(event.to_state_dict(), ensure_ascii=False)
            assert self.SECRET not in rendered
            assert "payload" not in event.to_state_dict()
            assert "result" not in event.to_state_dict()

    def test_a_snapshot_does_not_advance_the_resync_cursor(self, worker_store):
        # A snapshot answers "where am I"; it is not an event. Counting it as
        # one would make every later sequence off by one for a client that had
        # not yet seen the submission event.
        store = worker_store
        job = store.submit("document_analysis").job
        snapshot = J.JobSubscription(store, job.job_id).snapshot()
        assert snapshot["state"] == J.STATE_QUEUED
        assert len(J.JobSubscription(store, job.job_id).poll()) == len(store.events_since(job.job_id, 0))

    def test_a_snapshot_of_a_job_that_does_not_exist_is_none(self, worker_store):
        assert J.JobSubscription(worker_store, "no-such-job").snapshot() is None

    def test_poll_partitions_the_trail_with_no_gap_and_no_duplicate(self, worker_store):
        store = worker_store
        job = store.submit("document_analysis").job
        J.JobWorker(store, document_pipeline(), lease_seconds=30).drain()
        everything = store.events_since(job.job_id, 0)
        subscription = J.JobSubscription(store, job.job_id)
        first = subscription.poll()
        assert subscription.poll() == []
        assert [event["sequence"] for event in first] == [
            event.sequence for event in everything
        ]

    def test_a_client_that_drops_and_reconnects_sees_exactly_what_it_missed(
        self, worker_store
    ):
        store = worker_store
        job = store.submit("document_analysis").job
        first_page = J.JobSubscription(store, job.job_id).poll()
        cursor = first_page[-1]["sequence"]
        J.JobWorker(store, document_pipeline(), lease_seconds=30).drain()
        resumed = J.JobSubscription(store, job.job_id, since_sequence=cursor)
        # Where is it now, without reading the trail: the answer to "where am
        # I" is available even to a client that missed the entire run.
        assert resumed.snapshot()["state"] == J.STATE_COMPLETED
        missed = resumed.poll()
        assert missed
        assert missed[-1]["state"] == J.STATE_COMPLETED
        # Together the two pages are the whole trail, with nothing lost and
        # nothing delivered twice. This is "刷新/断线不丢任务".
        seen = [event["sequence"] for event in first_page] + [
            event["sequence"] for event in missed
        ]
        assert seen == [event.sequence for event in store.events_since(job.job_id, 0)]
        assert len(seen) == len(set(seen))

    def test_a_refresh_of_the_browser_does_not_resubmit_anything(self, worker_store):
        # The user reloads the page. There is nothing to resubmit: the job is
        # in the database, and the reloaded page re-reads it.
        store = worker_store
        job = store.submit("document_analysis", idempotency_key="refresh-me").job
        J.JobWorker(store, document_pipeline(), lease_seconds=30).drain()
        before = store.count_by_state()
        for _ in range(3):
            assert J.JobSubscription(store, job.job_id).snapshot()["state"] == (
                J.STATE_COMPLETED
            )
        # Even a client that wrongly resubmits gets the same job back rather
        # than a second run — see TestIdempotencyUnderLoad.
        again = store.submit("document_analysis", idempotency_key="refresh-me")
        assert again.deduplicated and again.job_id == job.job_id
        assert store.count_by_state() == before

    def test_the_stream_ends_at_the_terminal_event(self, worker_store):
        store = worker_store
        job = store.submit("document_analysis").job
        with J.JobWorker(store, document_pipeline(), lease_seconds=30, poll_interval=0.01) as w:
            events = list(
                J.JobSubscription(store, job.job_id, poll_interval=0.01).stream()
            )
            w.stop()
        assert events[-1]["state"] == J.STATE_COMPLETED
        assert events[-1]["terminal"] is True
        assert all(not event["terminal"] for event in events[:-1])

    def test_a_stream_that_attaches_to_a_finished_job_does_not_hang(self, worker_store):
        # A browser that reconnects after the fact must still learn how the run
        # ended, rather than waiting on a job that will never move again.
        store = worker_store
        job = store.submit("document_analysis").job
        J.JobWorker(store, document_pipeline(), lease_seconds=30).drain()
        events = list(J.JobSubscription(store, job.job_id, poll_interval=0.01).stream())
        assert events[-1]["state"] == J.STATE_COMPLETED
        assert [event["state"] for event in events][:1] == [J.STATE_QUEUED]

    def test_a_stream_of_an_unknown_job_ends_when_asked_to_stop(self, worker_store):
        stop = threading.Event()
        stop.set()
        subscription = J.JobSubscription(worker_store, "no-such-job", stop_event=stop)
        # No crash and no busy hang: an SSE handler for a job that was deleted
        # mid-connection has to be able to close the stream.
        assert list(subscription.stream()) == []
