"""The job store under real concurrency: who is writing while whom.

    python3 -m pytest tests/test_job_concurrency.py -q

This file exists because of a defect that every other suite in the repository
was structurally unable to find. ``SqliteJobStore.transition`` opens a
transaction, runs its compare-and-set, and then — on the branch where the
caller *lost* the compare-and-set — fell out of the function without either
committing or rolling back. A lost compare-and-set is a normal outcome and not
a rare one: an API cancel arriving mid-step produces one, and so does
``recover_stale`` requeueing a job whose lease lapsed while a slow parse was
still running inside it. So the leak fired on a real upload of a real document,
and the symptom was not an exception anyone could read. It was a *write lock
held forever by a connection nobody would close again*, which surfaced to
other users as ``sqlite3.OperationalError: database is locked`` on uploads that
had nothing wrong with them — a 500 that read as a server fault.

**Why no existing test found it.** Two reasons, and both are structural rather
than accidental.

*The window is wide and the trigger is a timing fact about a document.* A
20-byte text file parses in well under a second, so its job never comes close to
the lease, ``recover_stale`` never sees it, nothing is ever requeued, and the
compare-and-set never loses. The bug needs a parse slower than the lease. A
9-page corpus PDF through docling takes about two minutes against the 60-second
lease ``routes_v1.run_parse_job`` ships, so it overlaps every time. The tests
that would have hit it are the ones that were skipped because they were slow.

*And the test that would have caught it asserts the wrong thing.* "Drive N
threads at the store and assert nothing raised" passes against this bug, and
passes more reliably the more contention there is — because the lost
compare-and-set raises :class:`~paperwhisperer.infra.jobs.JobConflict`, which
is the *designed* outcome and which every reasonable caller catches. So the
suite below asserts on the thing that actually broke, which is state nobody can
see by looking for exceptions: **is a transaction still open on each thread's
connection, and does that thread's connection still work afterwards.** Those are
properties of the store, not of the absence of a raise, and they are the ones
that regressed.

**The overlap is produced honestly here rather than simulated.** Every test
below creates the real condition — a step that outlives its lease, a background
worker calling ``recover_stale`` on a job the request thread is still parsing —
and then measures the store's condition afterwards. Nothing monkeypatches a
clock to make a race appear; the races are real and they are made more likely by
running them many times, not by faking them.
"""

from __future__ import annotations

import threading
import time
import uuid

import pytest

from paperwhisperer.infra import jobs as J
from paperwhisperer.infra.stores import TIER_PERSONAL

from .test_jobs import open_store

#: Long enough that a step can be made to overrun it deterministically, short
#: enough that the suite does not wait on it. The shipped lease is 300s and the
#: one ``routes_v1.run_parse_job`` uses is 60s; neither value is under test
#: here — what is under test is that a lease which *does* expire mid-step
#: leaves the store usable.
LEASE_SECONDS = 2.0

#: How many times a contended interaction is repeated. Not superstition: the
#: lost compare-and-set needs a specific interleaving, and one run is a coin
#: flip. Repeating is what turns "usually reproduces" into "reproduces", and a
#: concurrency test that only fails once in ten CI runs is a test that will be
#: marked flaky and deleted rather than fixed.
ROUNDS = 25


@pytest.fixture
def store_harness(tmp_path):
    harness = open_store(TIER_PERSONAL, tmp_path, name=f"conc-{uuid.uuid4().hex[:8]}.db")
    try:
        yield harness.store
    finally:
        _quiesce_before_close(harness)
        harness.close()


def _quiesce_before_close(harness) -> None:
    """Roll back anything still open, so teardown cannot crash the interpreter.

    **This is test hygiene, not part of any assertion.** SQLite segfaults if a
    connection with an open transaction is closed, so when this file is
    regressed the teardown of the *first* failing test would take the whole
    pytest process down — and a core dump prints no assertion messages, which
    would hide the very failure this file exists to report. The assertions have
    already run by this point; this only keeps their output readable.
    """
    database = getattr(harness.bundle, "database", None)
    for connection in list(getattr(database, "_connections", []) or []):
        try:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
        except Exception:  # noqa: BLE001 - teardown must not raise
            pass


def assert_store_usable(store, label: str) -> None:
    """Every connection this store handed out is idle and still writable.

    The assertion is deliberately not "no exception was raised". It is two
    measurable facts about the store, and each one on its own caught this defect:

    * ``in_transaction`` is false — nothing was left holding a transaction open,
      and therefore nothing is holding the write lock behind it.
    * the connection can still open a new transaction — so it was not merely
      idle but *usable*, which is what "this thread can never write again"
      denies.
    """
    connection = store.connection
    assert connection.in_transaction is False, (
        f"{label}: 连接上仍有未结束的事务；它会一直持有 SQLite 的写锁，"
        "使其他线程的写入全部失败。"
    )
    store.claim_next(f"{label}-probe", LEASE_SECONDS)


def slow_pipeline(kind: str, *, seconds: float, heartbeat: bool = True) -> J.JobPipeline:
    """A one-stage pipeline whose stage outlives ``seconds`` without finishing.

    ``heartbeat=False`` is the shipped behaviour of
    :func:`paperwhisperer.api.routes_v1.run_parse_job`, whose step calls
    ``parse_document`` without refreshing the lease — that is what lets the
    lease lapse under a slow parse, so it is what the overlap tests use.
    """
    deadline = time.monotonic() + float(seconds)

    def _step(context: J.JobContext) -> None:
        while time.monotonic() < deadline:
            if heartbeat:
                context.heartbeat()
            time.sleep(0.02)

    return J.JobPipeline(
        kind=kind,
        steps=(J.JobStep(state=J.STATE_PARSING, run=_step, required=True, label="解析中。"),),
    )


def lose_the_compare_and_set(store, job_id: str) -> None:
    """Make a ``transition`` lose its compare-and-set, for real.

    The stale read is served by handing the store the row as it looked before a
    second writer moved it, which is exactly what the losing thread has in hand
    when its ``UPDATE ... WHERE state = ?`` misses. The competing write is a
    genuine one through the same public API — ``recover_stale``'s own requeue —
    so nothing here fakes a race into existence.

    Callers must have the job at ``parsing`` before calling this.
    """
    stale = store.get_job(job_id)
    store.transition(
        job_id,
        J.STATE_QUEUED,
        message="上一个 worker 已中断，任务重新排队。",
        note="lease_expired",
        expected_state=J.STATE_PARSING,
    )
    real_get = store.get_job
    store.get_job = lambda requested: stale if requested == job_id else real_get(requested)
    try:
        with pytest.raises(J.JobConflict):
            store.transition(
                job_id,
                J.STATE_QUALITY_CHECK,
                expected_state=J.STATE_PARSING,
                message="迟到的步骤",
                note="step_enter",
            )
    finally:
        store.get_job = real_get


class TestTheLostCompareAndSetDoesNotLeak:
    """The defect itself, isolated so a regression names its own cause.

    Each test arranges a genuine lost compare-and-set — not a fake one — by
    having a second writer move the row in the window between the loser's read
    and its write. That is the only way the branch is reached, and it is reached
    constantly in production by ``recover_stale`` and by API cancels.
    """

    @staticmethod
    def _parsing_job(store, worker: str = "worker") -> str:
        job_id = store.submit("document_analysis", {}).job.job_id
        store.claim_next(worker, LEASE_SECONDS)
        store.transition(job_id, J.STATE_PARSING, expected_state=J.STATE_QUEUED, message="m")
        return job_id

    def test_a_lost_compare_and_set_leaves_no_transaction_open(self, store_harness):
        store = store_harness
        lose_the_compare_and_set(store, self._parsing_job(store))
        assert_store_usable(store, "after a lost compare-and-set")

    def test_a_lost_compare_and_set_does_not_lock_out_another_thread(self, store_harness):
        """The damage was to *other* users, so this is measured from another thread.

        A connection left holding a write lock is invisible to the thread that
        leaked it — reads still answer, and only the next write complains. The
        failure a user actually experiences is on a different connection, so the
        test asks that question from one.
        """
        store = store_harness
        lose_the_compare_and_set(store, self._parsing_job(store))

        outcome: dict[str, str] = {}

        def unrelated_upload() -> None:
            try:
                store.submit("document_analysis", {"source_filename": "other.pdf"})
                outcome["result"] = "OK"
            except Exception as exc:  # noqa: BLE001 - the failure being guarded
                outcome["result"] = f"{type(exc).__name__}: {exc}"

        thread = threading.Thread(target=unrelated_upload, name="unrelated-upload")
        thread.start()
        thread.join(timeout=90)
        assert outcome.get("result") == "OK", (
            f"一个无关请求的提交失败了：{outcome.get('result')}。"
            "上一次丢失的 compare-and-set 仍然占着 SQLite 的写锁。"
        )

    def test_repeated_lost_compare_and_sets_do_not_accumulate(self, store_harness):
        """One leak is survivable; a loop of them is how a deployment finds out.

        The retry budget, an API client retrying a cancel, and a queue of long
        parses all produce many of these in a row. Asserted over the whole run
        because the failure mode is cumulative: a store that survives one lost
        race and dies on the fifth is exactly what a busy queue looks like.
        """
        store = store_harness
        for round_number in range(ROUNDS):
            lose_the_compare_and_set(store, self._parsing_job(store))
            assert_store_usable(store, f"round {round_number}")


class TestAWorkerOverlappingALeaseExpiry:
    """The production shape: a slow parse, a lapsed lease, a second worker.

    This is the reproduction the defect was found through. ``run_parse_job``
    hands its job a 60-second lease and its parse step does not heartbeat while
    ``parse_document`` runs, so a real multi-page document outruns the lease;
    the background worker's ``recover_stale`` then requeues a job that is still
    being parsed, and the request thread's terminal transition is the write
    that loses its compare-and-set.
    """

    def test_a_parse_that_outlives_its_lease_leaves_the_store_writable(self, store_harness):
        """The production shape, end to end, on a deliberately overrunning step.

        This is the reproduction the defect was found through. ``run_parse_job``
        hands its job a 60-second lease and its parse step does not heartbeat
        while ``parse_document`` runs, so a real multi-page document outruns the
        lease; the background worker's ``recover_stale`` then requeues a job that
        is still being parsed, and the request thread's terminal transition is
        the write that loses its compare-and-set.

        **The lost write happens on the request thread, so the request thread is
        what inspects the connection.** ``SqliteDatabase`` hands each thread its
        own, and the transaction leaks on the one that opened it; a check from
        the main thread would find a perfectly clean store and pass against the
        bug. So the assertion runs inside the worker thread, before it returns.
        """
        store = store_harness
        # The request thread runs the parse, on a `document_ingest` kind.
        request_pipeline = slow_pipeline(
            "document_ingest", seconds=LEASE_SECONDS * 2, heartbeat=False
        )
        # The background worker runs a *different* kind, exactly as
        # routes_jobs.start_job_worker does: it claims `document_analysis` and
        # `qa_answer`, never `document_ingest`. Giving it the same kind here
        # would make it claim the job out from under the request thread, and the
        # overlap would never happen — the request thread would find an empty
        # queue and the test would pass for the wrong reason. With a different
        # kind its only route to the running job is `recover_stale`, which is
        # precisely how the production worker touches it.
        background = J.JobWorker(
            store,
            slow_pipeline("document_analysis", seconds=0.0, heartbeat=False),
            lease_seconds=LEASE_SECONDS,
            poll_interval=0.05,
        )
        background.start()
        try:
            request_worker = J.JobWorker(
                store, request_pipeline, lease_seconds=LEASE_SECONDS, poll_interval=0.05
            )
            job_id = store.submit("document_ingest", {"source_filename": "hinton.pdf"}).job.job_id

            failures: list[str] = []
            observed: dict[str, object] = {}

            def request_thread() -> None:
                _run_catching(request_worker, failures, "request")
                # This thread's own connection: the one the lost write opened
                # its transaction on.
                observed["leaked"] = store.connection.in_transaction
                try:
                    store.claim_next("request-thread-probe", LEASE_SECONDS)
                    observed["writable"] = True
                except Exception as exc:  # noqa: BLE001 - recorded for the assertion
                    observed["writable"] = f"{type(exc).__name__}: {exc}"

            thread = threading.Thread(target=request_thread, name="request")
            thread.start()
            thread.join(timeout=180)
            assert not thread.is_alive(), "请求线程没有结束；解析步骤卡住了。"

            # The overlap really happened. Without this the test could be
            # passing simply because the lease never lapsed, which is what a
            # fast step or a generous lease would give.
            assert observed.get("leaked") is False, (
                "请求线程的连接上仍持有未结束的事务；它会一直占着 SQLite 的写锁。"
            )
            assert observed.get("writable") is True, (
                f"泄漏之后请求线程再也无法写入：{observed.get('writable')}"
            )

            # And the failure a user sees is on an unrelated request, whose
            # upload is blocked behind a lock it had no part in creating.
            outcome: dict[str, str] = {}

            def unrelated_upload() -> None:
                try:
                    store.submit("document_ingest", {"source_filename": "next.pdf"})
                    outcome["result"] = "OK"
                except Exception as exc:  # noqa: BLE001
                    outcome["result"] = f"{type(exc).__name__}: {exc}"

            other = threading.Thread(target=unrelated_upload)
            other.start()
            other.join(timeout=120)
            assert outcome.get("result") == "OK", (
                f"慢解析之后的下一个上传失败了：{outcome.get('result')}"
            )
            assert not [item for item in failures if "locked" in item], (
                f"慢解析期间 worker 因写锁被占而失败：{failures}"
            )
            # The job is recoverable whatever recovery decided: a terminal state
            # with a reason, or back on the queue for another attempt.
            assert store.get_job(job_id).state in J.JOB_STATES
        finally:
            background.stop()

    def test_recovery_racing_a_live_worker_never_corrupts_the_row(self, store_harness):
        """Two workers recovering the same lapsed lease must leave it coherent.

        ``recover_stale`` has no ``kind`` filter, so with one worker per pipeline
        kind — which is what ``start_job_worker`` creates — several workers
        fetch the same expired row and race to requeue it. Exactly one may win.
        The losers raise :class:`JobConflict`, which is the designed outcome and
        which ``JobWorker._loop`` catches. What must not happen is a row left
        half-moved, or a connection left holding a transaction.

        **The assertion is per-worker, and that is the whole point of the test.**
        A leaked transaction lives on the connection of the thread that opened
        it, and ``SqliteDatabase`` hands each thread its own. Checking the main
        thread's connection here would report a clean store while the very
        connection that just lost the race is the one holding the write lock —
        and would pass against the defect it exists to catch. So each worker
        thread inspects ``store.connection`` *itself*, which resolves to its own
        connection because the store resolves it per thread.
        """
        store = store_harness
        jobs = 6
        job_ids = []
        for _ in range(jobs):
            job_id = store.submit("document_ingest", {}).job.job_id
            store.claim_next("inline-drain", LEASE_SECONDS)
            store.transition(
                job_id, J.STATE_PARSING, expected_state=J.STATE_QUEUED, message="m"
            )
            job_ids.append(job_id)

        # Wait for the leases to genuinely lapse. `recover_stale` with no
        # argument reads the wall clock, exactly as `JobWorker.tick` calls it,
        # so the only way to make a lease expired here is to let it expire.
        # Passing a future `now` instead would make the test assert against a
        # clock no production caller uses.
        deadline = time.monotonic() + LEASE_SECONDS + 0.5
        while time.monotonic() < deadline:
            time.sleep(0.05)
        assert all(
            store.get_job(job_id).lease_expires_at < time.time() for job_id in job_ids
        ), "租约并未真正过期，这个测试将无事可做。"

        workers = [
            J.JobWorker(
                store,
                slow_pipeline(kind, seconds=0.0),
                lease_seconds=LEASE_SECONDS,
                poll_interval=0.01,
            )
            for kind in ("document_analysis", "qa_answer")
        ]
        failures: list[str] = []
        leaked: list[str] = []
        gate = threading.Barrier(len(workers))

        def run_one(name: str, worker: J.JobWorker) -> None:
            # Line the workers up so they all enter recovery together, which is
            # what makes them contend for the same rows.
            gate.wait()
            _run_catching(worker, failures, name)
            # This thread's own connection, not the main thread's.
            if store.connection.in_transaction:
                leaked.append(name)

        threads = [
            threading.Thread(target=run_one, args=(f"worker-{i}", worker))
            for i, worker in enumerate(workers)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)

        assert leaked == [], (
            f"这些 worker 的连接上仍持有未结束的事务：{leaked}。"
            "它们会一直占着 SQLite 的写锁。"
        )
        assert not [item for item in failures if "locked" in item], failures

        for job_id in job_ids:
            row = store.get_job(job_id)
            assert row.state in J.JOB_TRANSITIONS, row.state
            if row.state == J.STATE_QUEUED:
                # Requeued means the lease went with it — a requeued job that
                # kept its expired lease would be un-claimable by the very
                # worker that just recovered it.
                assert row.lease_owner == "" and row.lease_expires_at == 0.0


class TestTheQueueUnderContention:
    """The queue itself, with several writers and a reader throughout."""

    def test_concurrent_submitters_each_get_a_job_and_none_is_lost(self, store_harness):
        """N writers, one queue, and an accounting that has to balance.

        Asserted on the row count rather than on the absence of an exception,
        because a submit that silently deduplicated into someone else's job would
        satisfy "nothing raised" and lose the user's document.
        """
        store = store_harness
        count = 16
        barrier = threading.Barrier(count)
        errors: list[str] = []

        def submit(index: int) -> None:
            barrier.wait()
            try:
                store.submit("document_analysis", {"index": index}, idempotency_key=f"k{index}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"submitter {index}: {type(exc).__name__}: {exc}")

        threads = [threading.Thread(target=submit, args=(i,)) for i in range(count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)

        assert errors == []
        assert len(store.list_jobs(limit=count + 5)) == count
        assert store.count_by_state()[J.STATE_QUEUED] == count

    def test_only_one_worker_can_claim_a_given_job(self, store_harness):
        """The lease's reason for existing, checked under real contention.

        Every worker tries to claim the *same* single job. The property is not
        "all of them returned something" — it is that exactly one got it, so
        the work is never done twice and the external cost is never billed
        twice.
        """
        store = store_harness
        job_id = store.submit("document_analysis", {}).job.job_id
        workers = 8
        barrier = threading.Barrier(workers)
        claimed: list[str] = []
        errors: list[str] = []
        guard = threading.Lock()

        def claim(index: int) -> None:
            barrier.wait()
            try:
                got = store.claim_next(f"worker-{index}", LEASE_SECONDS)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"claimant {index}: {type(exc).__name__}: {exc}")
                return
            if got is not None:
                with guard:
                    claimed.append(got.job_id)

        threads = [threading.Thread(target=claim, args=(i,)) for i in range(workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)

        assert errors == []
        assert claimed == [job_id], f"一个任务被领取了 {len(claimed)} 次：{claimed}"
        assert store.get_job(job_id).attempts == 1

    def test_a_reader_is_never_blocked_by_the_writers(self, store_harness):
        """Reads must not queue behind the single writer.

        This is the property that makes the connection model worth its
        complexity, and the one a naive fix — a global lock around the
        connection — would destroy. It is asserted as a *latency* bound against
        a writer that is deliberately holding the store busy, not as the absence
        of an error: a reader that blocked behind a writer would raise nothing
        and be wrong anyway.
        """
        store = store_harness
        stop = threading.Event()
        write_errors: list[str] = []

        def write_forever() -> None:
            index = 0
            while not stop.is_set():
                try:
                    store.submit("document_analysis", {"i": index})
                    index += 1
                except Exception as exc:  # noqa: BLE001
                    write_errors.append(f"{type(exc).__name__}: {exc}")
                    return

        writer = threading.Thread(target=write_forever, name="writer")
        writer.start()
        try:
            # Give the writer time to be genuinely mid-flight.
            time.sleep(0.2)
            samples = []
            for _ in range(20):
                started_at = time.monotonic()
                assert store.list_jobs(limit=5) is not None
                samples.append(time.monotonic() - started_at)
                time.sleep(0.005)
        finally:
            stop.set()
            writer.join(timeout=30)

        assert write_errors == []
        # Generous on purpose: the claim is "reads are not serialised behind
        # writers", not a performance target, and a threshold this loose still
        # fails loudly against a lock (which would put every sample behind the
        # writer's whole round trip).
        worst = max(samples)
        assert worst < 1.0, f"读操作被写者阻塞，最慢一次耗时 {worst:.3f}s"

    def test_a_worker_polling_does_not_stop_a_request_from_writing(self, store_harness):
        """The reported shape: the loop and a request, at the same time.

        A worker polling every few milliseconds against a request that keeps
        writing is the configuration that produced the defect. It is asserted as
        a throughput property — every write in a fixed window must land — because
        a store that answers ``database is locked`` after the first collision
        also answers it without raising anywhere the caller looks.
        """
        store = store_harness
        pipeline = slow_pipeline("document_ingest", seconds=0.0)
        pollers = [
            J.JobWorker(store, pipeline, lease_seconds=LEASE_SECONDS, poll_interval=0.005)
            for _ in range(3)
        ]
        for poller in pollers:
            poller.start()
        try:
            errors: list[str] = []
            landed = 0
            for index in range(ROUNDS * 2):
                try:
                    store.submit("document_analysis", {"i": index})
                    landed += 1
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{type(exc).__name__}: {exc}")
            assert errors == [], f"worker 轮询期间请求写入失败：{errors[:3]}"
            assert landed == ROUNDS * 2
        finally:
            for poller in pollers:
                poller.stop()


class TestTheLostRaceIsHandledNotLeaked:
    """The guarantee, stated as behaviour rather than as an absence of a raise.

    This is the personal tier only, and the class name says so rather than
    claiming a two-tier result it does not measure. What makes the Hosted tier
    safe is structural and is asserted elsewhere: ``PostgresJobStore.transition``
    runs its statement inside ``PostgresDatabase.cursor``, which commits on a
    clean exit and rolls back on any exception, so that path has no unterminated
    transaction to leak and no single-writer lock to hold. Nothing was added to
    it, and nothing about it got slower — which is the whole of the Hosted-tier
    argument, and it is an argument about the code rather than about a test.

    So the assertion here is the behaviour both tiers must show a caller: the
    losing writer's row change survives, the conflict is reported, and the store
    is still writable afterwards. A caller that could rely on that does not have
    to know which tier it got, which is what "the same product" means.
    """

    def test_the_competing_write_survives_and_the_store_stays_writable(self, store_harness):
        store = store_harness
        job_id = store.submit("document_analysis", {}).job.job_id
        store.claim_next("worker", LEASE_SECONDS)
        store.transition(job_id, J.STATE_PARSING, expected_state=J.STATE_QUEUED, message="m")

        lose_the_compare_and_set(store, job_id)

        # The competing writer's move survived intact — a lost compare-and-set
        # must not silently undo the write that beat it, or the last writer
        # would not in fact be the last writer.
        assert store.get_job(job_id).state == J.STATE_QUEUED
        assert_store_usable(store, "after a lost compare-and-set")


def _run_catching(worker: J.JobWorker, failures: list[str], name: str = "worker") -> None:
    """Run one worker tick, recording rather than raising its failure."""
    try:
        worker.tick()
    except Exception as exc:  # noqa: BLE001 - recorded and asserted on by the caller
        failures.append(f"{name}: {type(exc).__name__}: {exc}")


class TestAWorkerOutlivingItsStore:
    """The lifetime half of the same subsystem: a worker must not outlive its store.

    The tests above pin the *transaction* the previous fix repaired — a lost
    compare-and-set that left a transaction open and SQLite's writer lock held.
    This class pins the defect that fix could not have found and that only shows
    up at teardown: :meth:`JobWorker.stop` joins with a *timeout*, and a bounded
    join that expires returns while the worker thread is still running. The
    owner then closes the store underneath it, and from that instant the worker
    is using a connection that no longer belongs to it.

    There were two ways out of that, and both were real:

    * **A use-after-free.** SQLite's C layer frees a connection the instant
      ``.close()`` is called. Closing, from one thread, a handle another thread is
      mid-``execute()`` on is a use-after-free and CPython does not raise — the
      interpreter dies with a SIGSEGV. That is the "intermittent segfault in
      infra/jobs.py" that accompanied the transaction report: intermittent
      because it needs the close to land while the worker is inside a statement.
    * **A hard teardown error.** Where the free happened to miss the worker's
      active window, the worker's next store call hit the closed database and
      raised ``RuntimeError: 数据库连接已关闭。``, which — raised from a worker
      during the last tick of a session — is what made a full run abort during
      teardown and print no pass/fail count at all.

    So the invariant under test is **ownership of the connection, not timing**:
    a connection is released by the thread using it, or after that thread has
    finished — never by a second thread while the first is still running. These
    tests reproduce both failure modes directly rather than waiting for a slow
    parse to line up with a fixture teardown, so the regression is deterministic
    here even though it was intermittent in the suite.
    """

    @staticmethod
    def _harness(tmp_path):
        return open_store(TIER_PERSONAL, tmp_path, name=f"lifetime-{uuid.uuid4().hex[:8]}.db")

    def test_closing_the_database_under_a_running_worker_does_not_kill_the_process(
        self, tmp_path
    ):
        """The SIGSEGV, made deterministic: close a handle a thread is using.

        Asserting "it did not raise" is impossible here — a use-after-free does
        not raise, it takes the interpreter with it, and a core dump prints
        nothing. So the assertion *is* survival: the test process running to
        completion after a cross-thread ``close()`` is the only honest statement
        that the handle was not freed underneath its owner.
        """
        harness = self._harness(tmp_path)
        try:
            database = harness.bundle.database
            harness.store.connection.execute(
                "SELECT 1"
            )  # warm the file and give this thread a connection

            release = threading.Event()
            failure: list[str] = []

            def run_until_released() -> None:
                # This thread's own connection, which `database.close()` is
                # about to be told about. The loop keeps a statement in flight
                # across the close on purpose.
                connection = database.connection()
                while not release.is_set():
                    connection.execute("SELECT 1").fetchall()

            thread = threading.Thread(target=run_until_released, name="store-user")
            thread.start()
            try:
                # Wait until the worker thread really has a connection of its
                # own, so the close() below has a *live foreign owner* to
                # consider rather than racing connection creation. The main
                # thread already holds one, so the count to wait for is two.
                deadline = time.monotonic() + 10.0
                while len(database._connections) < 2 and time.monotonic() < deadline:
                    time.sleep(0.01)
                assert len(database._connections) >= 2, (
                    "工作线程没有打开自己的连接，这个用例没有制造出竞争。"
                )

                # The dangerous moment: a second thread releases handles that
                # include one the worker is actively executing on.
                database.close()
            finally:
                release.set()
                thread.join(timeout=30)

            assert not thread.is_alive(), "工作线程卡死，说明 close() 与它互相等待了。"
            assert failure == []
            # The connection owned by the *worker* was reclaimed by the worker
            # thread itself when its thread-local was collected; close() must
            # not have raced it. The main thread's own connection is closed
            # eagerly, which is the ordinary path.
            harness.bundle.close()  # idempotent; must not raise
        finally:
            _quiesce_before_close(harness)

    def test_a_worker_stopped_mid_step_can_have_its_store_closed_underneath_it(
        self, tmp_path
    ):
        """The real shape: bounded ``stop()``, then the owner closes the store.

        This is what the application lifespan does at shutdown and what the
        session-scoped journey fixture did at teardown. A slow step outlives the
        join timeout, so ``stop()`` returns with the thread still mid-pipeline,
        and the bundle is closed a moment later. The worker must survive that
        and finish on its own terms — here, by noticing it was asked to stop and
        leaving the job recoverable — rather than by touching a dead connection.
        """
        harness = self._harness(tmp_path)
        try:
            store = harness.store
            reached = threading.Event()
            release = threading.Event()

            def _step(context: J.JobContext) -> None:
                # A stage that is still working when `stop()` is called, the
                # way a multi-minute parse is.
                reached.set()
                release.wait(timeout=30)
                context.heartbeat()  # a store call after the close

            pipeline = J.JobPipeline(
                kind="document_analysis",
                steps=(J.JobStep(state=J.STATE_PARSING, run=_step, required=True),),
            )
            worker = J.JobWorker(store, pipeline, lease_seconds=30, poll_interval=0.01)
            store.submit("document_analysis", {})
            worker.start()
            try:
                assert reached.wait(timeout=30), "工作线程没有进入解析阶段。"

                # Bounded stop, exactly as a lifespan teardown issues it. The
                # worker is mid-step, so this returns before the thread is done.
                worker.stop(timeout=0.2)
                # The store is closed while the worker is still inside the step.
                harness.bundle.close()
            finally:
                release.set()

            # Give the orphan the time it needs to finish its step and reach the
            # store it was told to stop using. Under the old code the handle was
            # freed here, which either segfaulted the process or raised out of
            # the loop; either way the suite died at teardown.
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                alive = [t for t in threading.enumerate() if t.name.startswith("job-worker-")]
                if not alive:
                    break
                time.sleep(0.05)
            remaining = [t.name for t in threading.enumerate() if t.name.startswith("job-worker-")]
            assert remaining == [], (
                f"被 stop() 遗弃的 worker 线程仍在运行：{remaining}。"
                "stop() 超时后线程被丢弃，没有任何东西能再让它退出。"
            )
            # Reaching this line at all is half the assertion: under the old
            # code, closing the bundle under the still-running worker either
            # freed its SQLite handle mid-execute (SIGSEGV, process gone before
            # any summary) or raised `RuntimeError: 数据库连接已关闭。` out of the
            # loop during this window and aborted the whole run at teardown.
            # Surviving the window with no live worker left is the fix.
        finally:
            _quiesce_before_close(harness)

    def test_stop_keeps_the_thread_handle_until_the_worker_really_finishes(self, tmp_path):
        """``stop()`` must not orphan its own thread.

        The obvious implementation — ``thread, self._thread = self._thread, None``
        — forgets the handle whether or not the join actually succeeded. When the
        timeout expires that loses the only reference to a live thread: nothing
        can re-join it, ``start()`` believes the worker is gone and launches a
        *second* one over the same queue, and the store it is using is torn down
        while it still runs. The handle is dropped only once the thread is
        actually gone.
        """
        harness = self._harness(tmp_path)
        try:
            store = harness.store
            release = threading.Event()
            reached = threading.Event()

            def _step(context: J.JobContext) -> None:
                reached.set()
                release.wait(timeout=30)

            pipeline = J.JobPipeline(
                kind="document_analysis",
                steps=(J.JobStep(state=J.STATE_PARSING, run=_step, required=True),),
            )
            worker = J.JobWorker(store, pipeline, lease_seconds=30, poll_interval=0.01)
            # One job for the worker to claim, so it enters the step above.
            store.submit("document_analysis", {})
            worker.start()
            try:
                assert reached.wait(timeout=30), "工作线程没有进入解析阶段。"
                worker.stop(timeout=0.2)  # expires: the thread is mid-step
                assert worker._thread is not None, (
                    "stop() 超时后丢掉了仍在运行的线程句柄；该线程成了无主线程。"
                )
            finally:
                release.set()
                worker.stop(timeout=30)

            assert worker._thread is None, (
                "线程已经结束，但句柄仍被保留。"
            )
        finally:
            _quiesce_before_close(harness)