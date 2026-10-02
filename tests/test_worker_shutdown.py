"""Shutdown must not forget a worker before it has finished stopping it.

    python3 -m pytest tests/test_worker_shutdown.py -q

The defect this file exists for was found by an audit of test pollution rather
than by a failing assertion, which is why it needed a file of its own: nothing
was *wrong* in any single test. ``stop_job_workers`` snapshotted its registry,
cleared it, and only then performed the bounded stop::

    with _WORKER_LOCK:
        workers = list(_WORKERS.values())
        _WORKERS.clear()
    for worker in workers:
        worker.stop(timeout=timeout)

The ordering is the whole bug. ``_WORKERS.clear()`` runs *before* any join, so
by the time a worker's ``stop()`` finds its thread mid-parse and times out, the
only reference to that live thread is already gone. The thread becomes an orphan:
no later ``stop_job_workers()`` can name it, no second lifespan teardown can
re-join it, and a test module three files later has nothing to assert on. The
observed consequence was a worker thread in ``tests/test_e2e_journey.py`` that
leaked at module teardown and then stayed alive for the rest of the session.

**What made it hard to see is that the old code reported success either way.**
After the clear, ``active_workers()`` returned ``[]`` — so did the return value's
length — whether the joins had completed or not. A caller asking "did shutdown
happen?" got ``[]`` in both cases, and the one answer that mattered (no) was
indistinguishable from the answer that meant everything was fine. So every test
written against the old shape would pass: ``assert routes_jobs.active_workers()
== []`` passes against a leak, which is precisely what
``tests/test_jobs_api.py::TestTheWorkerLifecycle`` did.

Hence the assertions below are about *the state after a stop*, and every
distinguishing case is arranged deterministically. The "will not stop" worker is
a real :class:`~paperwhisperer.infra.jobs.JobWorker` on a real store, running a
real pipeline whose real step blocks on a :class:`threading.Event` this test
owns. Nothing here sleeps to manufacture the condition, and nothing fakes a
clock: a step that will not stop is a step that is *parked*, which is a fact the
test sets up rather than a fact it waits for.
"""

from __future__ import annotations

import logging
import threading
import uuid

import pytest
from fastapi.testclient import TestClient

from paperwhisperer import app as app_module
from paperwhisperer.api import routes_jobs
from paperwhisperer.infra import jobs as J
from paperwhisperer.infra.stores import TIER_PERSONAL

from .test_jobs import open_store

#: The join budget handed to the bounded stop. Short enough that the "will not
#: stop" case costs almost nothing, long enough that a clean stop of an idle
#: worker is never a coin flip on a loaded machine. The budget is *not* what
#: makes a worker fail to stop here — the step is parked on an event and ignores
#: every budget — so this number carries no meaning beyond test runtime.
SHORT_TIMEOUT = 0.2

#: A budget long enough to join a worker that has been released, used for the
#: retry assertions. Generous on purpose: it must not be the reason a test fails.
LONG_TIMEOUT = 30.0


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------


@pytest.fixture
def registry() -> dict[str, object]:
    """The real registry, and the guarantee this file is about, in a fixture.

    The registry is module-global, so every test here both writes to it and
    restores it. Restoring matters more than usual: a worker left in the
    registry would be one this test file leaked into the next module, which is
    the very failure it is reporting on.
    """
    saved_workers = dict(routes_jobs._WORKERS)
    saved_stranded = list(routes_jobs._STRANDED)
    routes_jobs._WORKERS.clear()
    routes_jobs._STRANDED.clear()
    try:
        yield routes_jobs._WORKERS
    finally:
        # Stop whatever is still registered, with a budget that is generous
        # rather than quick — this is teardown for the *next* file, not a test.
        try:
            routes_jobs.stop_job_workers(timeout=LONG_TIMEOUT)
        finally:
            routes_jobs._WORKERS.clear()
            routes_jobs._STRANDED.clear()
            routes_jobs._WORKERS.update(saved_workers)
            routes_jobs._STRANDED.extend(saved_stranded)


@pytest.fixture
def store(tmp_path):
    """A real job store on the personal tier, with its own database file."""
    harness = open_store(TIER_PERSONAL, tmp_path, name=f"shutdown-{uuid.uuid4().hex[:8]}.db")
    try:
        yield harness.store
    finally:
        harness.close()


class ParkedStep:
    """A real pipeline step that will not return until this test says so.

    The determinism is the entire point. A step that sleeps for five seconds is
    a step that *usually* still runs when a 200-millisecond join expires; on a
    fast machine after a warm page cache it might not be, and a test whose
    "cannot stop" case is a race is a test that will be marked flaky and deleted
    rather than fixed. An event is not a race: the thread is provably inside the
    step, and provably still inside it, until ``release`` is set.
    """

    def __init__(self) -> None:
        self.entered = threading.Event()
        self._release = threading.Event()
        self.finished = threading.Event()

    def release(self) -> None:
        self._release.set()

    def pipeline(self, kind: str = "document_analysis") -> J.JobPipeline:
        def _step(context: J.JobContext) -> None:
            self.entered.set()
            # The wait is bounded only as a backstop against a test bug: the
            # release always comes, because the fixture's teardown sets it
            # before the store is closed.
            self._release.wait(timeout=LONG_TIMEOUT)
            self.finished.set()

        return J.JobPipeline(
            kind=kind,
            steps=(J.JobStep(state=J.STATE_PARSING, run=_step, required=True),),
        )


def make_worker(store, step: ParkedStep, *, kind: str = "document_analysis") -> J.JobWorker:
    """A real worker on a real store, running ``step``'s pipeline."""
    return J.JobWorker(
        store,
        step.pipeline(kind),
        worker_id=f"shutdown-{uuid.uuid4().hex[:8]}",
        lease_seconds=30.0,
        poll_interval=0.01,
    )


def idle_worker(store, *, kind: str = "document_analysis") -> J.JobWorker:
    """A real worker with nothing to do, so its next tick finds an empty queue.

    This is the *clean* half of every comparison below. The distinction the file
    is about is only meaningful as a contrast, and the contrast has to be between
    two real shutdowns of real workers — not between a real shutdown and a
    fabricated one.
    """
    return J.JobWorker(
        store,
        J.JobPipeline(
            kind=kind,
            steps=(J.JobStep(state=J.STATE_PARSING, run=lambda _ctx: None, required=True),),
        ),
        worker_id=f"shutdown-{uuid.uuid4().hex[:8]}",
        lease_seconds=30.0,
        poll_interval=0.01,
    )


# --------------------------------------------------------------------------
# The property
# --------------------------------------------------------------------------


class TestACleanStopForgetsTheWorker:
    """The half that was already right, pinned so the fix cannot break it."""

    def test_a_worker_that_stops_leaves_the_registry(self, store, registry):
        worker = idle_worker(store)
        worker.start()
        registry[routes_jobs.KIND_DOCUMENT_ANALYSIS] = worker

        assert routes_jobs.stop_job_workers(timeout=LONG_TIMEOUT) == [worker]

        assert routes_jobs.KIND_DOCUMENT_ANALYSIS not in registry, (
            "已停止的 worker 仍在注册表中；注册表会随着每一条流水线无限增长。"
        )
        assert routes_jobs.active_workers() == []
        assert routes_jobs.stuck_workers() == []

    def test_a_clean_stop_is_the_only_stop_that_reports_nothing_stuck(
        self, store, registry, caplog
    ):
        """The two outcomes must be told apart, so the quiet one stays quiet.

        Not merely an absence of noise: a design that logged "worker did not
        stop" on every shutdown would train an operator to ignore the line, which
        is the same defect in the other direction — a signal nobody reads is not
        a signal.
        """
        worker = idle_worker(store)
        worker.start()
        registry[routes_jobs.KIND_DOCUMENT_ANALYSIS] = worker

        with caplog.at_level(logging.ERROR, logger=routes_jobs.__name__):
            routes_jobs.stop_job_workers(timeout=LONG_TIMEOUT)

        assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []

    def test_a_worker_that_stops_reports_that_it_stopped(self, store, registry):
        """``stop`` answers the question instead of leaving it to be inferred."""
        worker = idle_worker(store)
        worker.start()
        assert worker.stop(timeout=LONG_TIMEOUT) is True
        assert worker.running is False


class TestAStopThatFailsIsVisible:
    """The half that was wrong: a failed shutdown looked exactly like a clean one."""

    def test_a_worker_that_will_not_stop_is_still_reachable_afterwards(self, store, registry):
        """The defect, stated as the property it violated.

        A shutdown must not forget a worker before it has finished stopping it.
        Under the old ordering the handle was dropped *before* the join, so this
        assertion is unanswerable against it — there is nothing left in the
        process to ask about.
        """
        step = ParkedStep()
        worker = make_worker(store, step)
        store.submit(routes_jobs.KIND_DOCUMENT_ANALYSIS, {})
        worker.start()
        registry[routes_jobs.KIND_DOCUMENT_ANALYSIS] = worker
        try:
            assert step.entered.wait(timeout=30), "工作线程没有进入解析阶段。"

            routes_jobs.stop_job_workers(timeout=SHORT_TIMEOUT)

            # Still in the registry — the property.
            assert registry.get(routes_jobs.KIND_DOCUMENT_ANALYSIS) is worker, (
                "join 超时之后，worker 被从注册表中移除了；它的线程仍在运行，"
                "而进程里已经没有任何东西能再引用它。"
            )
            assert worker.running is True
            assert routes_jobs.active_workers() == [worker]
        finally:
            step.release()
            routes_jobs.stop_job_workers(timeout=LONG_TIMEOUT)

    def test_a_failed_stop_is_distinguishable_from_a_clean_one(self, store, registry):
        """``stuck_workers()`` is the question, and it has two real answers.

        Both halves are measured on real workers in the same test, because the
        assertion that matters is not "stuck is non-empty" — it is "stuck is
        empty for the worker that stopped and non-empty for the one that did not".
        A predicate that always returned ``[]`` would pass the first half.
        """
        stuck_step = ParkedStep()
        stuck = make_worker(store, stuck_step)
        store.submit(routes_jobs.KIND_DOCUMENT_ANALYSIS, {})
        stuck.start()
        registry[routes_jobs.KIND_DOCUMENT_ANALYSIS] = stuck
        try:
            assert stuck_step.entered.wait(timeout=30), "工作线程没有进入解析阶段。"
            routes_jobs.stop_job_workers(timeout=SHORT_TIMEOUT)
            assert routes_jobs.stuck_workers() == [stuck]
        finally:
            stuck_step.release()
            routes_jobs.stop_job_workers(timeout=LONG_TIMEOUT)

        assert routes_jobs.stuck_workers() == [], (
            "worker 已经停止，stuck_workers() 仍然报告它在运行。"
        )

        # The clean case, on its own, for the same reason in the other order.
        clean = idle_worker(store, kind=routes_jobs.KIND_QA)
        clean.start()
        registry[routes_jobs.KIND_QA] = clean
        routes_jobs.stop_job_workers(timeout=LONG_TIMEOUT)
        assert routes_jobs.stuck_workers() == []
        assert routes_jobs.KIND_QA not in registry

    def test_a_failed_stop_is_logged_at_error(self, store, registry, caplog):
        """A caller that reads no return value still learns that it failed.

        ``stop_job_workers``'s caller is the lifespan teardown, which has a log
        and nothing else. Asserting on the record rather than on the formatted
        line keeps the test honest about *level*: a WARNING is easy to lose in a
        startup log, and the whole claim is that this is not a normal outcome.
        """
        step = ParkedStep()
        worker = make_worker(store, step)
        store.submit(routes_jobs.KIND_DOCUMENT_ANALYSIS, {})
        worker.start()
        registry[routes_jobs.KIND_DOCUMENT_ANALYSIS] = worker
        try:
            assert step.entered.wait(timeout=30), "工作线程没有进入解析阶段。"

            with caplog.at_level(logging.DEBUG, logger=routes_jobs.__name__):
                routes_jobs.stop_job_workers(timeout=SHORT_TIMEOUT)

            errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
            assert errors, "join 超时是一次失败；它没有被记录。"
            assert any(worker.worker_id in r.getMessage() for r in errors), (
                f"错误日志没有点名未停止的 worker：{[r.getMessage() for r in errors]}"
            )
        finally:
            step.release()
            routes_jobs.stop_job_workers(timeout=LONG_TIMEOUT)

    def test_the_return_value_cannot_be_read_as_success(self, store, registry):
        """What the function hands back, and why it is not a success signal.

        The returned list is the *attempt* set — every worker this call asked to
        stop. It is deliberately not the stopped set, because the callers that
        want a count ("how many workers did this process have") would then get a
        number that changes with a timing outcome. So the return value keeps its
        old meaning and ``stuck_workers()`` carries the new information; pinning
        both here is what stops a later reader from assuming the list is a receipt.
        """
        step = ParkedStep()
        worker = make_worker(store, step)
        store.submit(routes_jobs.KIND_DOCUMENT_ANALYSIS, {})
        worker.start()
        registry[routes_jobs.KIND_DOCUMENT_ANALYSIS] = worker
        try:
            assert step.entered.wait(timeout=30), "工作线程没有进入解析阶段。"

            attempted = routes_jobs.stop_job_workers(timeout=SHORT_TIMEOUT)

            assert attempted == [worker], "返回的不是本次尝试停止的 worker 集合。"
            assert routes_jobs.stuck_workers() == [worker], (
                "停止失败的信息只出现在返回值里，调用方无从分辨；"
                "它必须在别处可问。"
            )
        finally:
            step.release()
            routes_jobs.stop_job_workers(timeout=LONG_TIMEOUT)


class TestASecondShutdownCanRetry:
    """What "keep the handle" buys: the failure is recoverable, not terminal."""

    def test_a_second_shutdown_after_a_failed_one_collects_the_worker(self, store, registry):
        """The follow-up call is the whole reason the ordering was wrong.

        Under the old code there was no second call to make — the registry was
        empty and the thread was nobody's. Here the retry re-joins with a fresh
        budget, succeeds, and removes the entry, so the process reaches the state
        it would have reached had the first shutdown not timed out.
        """
        step = ParkedStep()
        worker = make_worker(store, step)
        store.submit(routes_jobs.KIND_DOCUMENT_ANALYSIS, {})
        worker.start()
        registry[routes_jobs.KIND_DOCUMENT_ANALYSIS] = worker
        assert step.entered.wait(timeout=30), "工作线程没有进入解析阶段。"

        # First shutdown: fails, deterministically, because the step is parked.
        routes_jobs.stop_job_workers(timeout=SHORT_TIMEOUT)
        assert registry.get(routes_jobs.KIND_DOCUMENT_ANALYSIS) is worker

        # The step finishes on its own terms, which is what a slow parse does.
        step.release()
        assert step.finished.wait(timeout=30), "步骤没有结束，第二次关闭无从谈起。"

        # Second shutdown: the same function, called the way a later teardown in
        # the same session would call it.
        routes_jobs.stop_job_workers(timeout=LONG_TIMEOUT)

        assert routes_jobs.KIND_DOCUMENT_ANALYSIS not in registry, (
            "第二次关闭没有回收这个 worker；句柄被保留了下来却再也无法被使用。"
        )
        assert routes_jobs.stuck_workers() == []
        assert worker._thread is None

    def test_a_retry_that_fails_again_keeps_the_handle(self, store, registry):
        """Retrying must not turn one failed shutdown into an orphan either.

        The tempting next step after "remember the handle" is "and discard it
        after the second failure". That is the same defect with a longer fuse, so
        the retry is held to the same rule: a worker that will not stop stays
        reachable no matter how many shutdowns have already failed.
        """
        step = ParkedStep()
        worker = make_worker(store, step)
        store.submit(routes_jobs.KIND_DOCUMENT_ANALYSIS, {})
        worker.start()
        registry[routes_jobs.KIND_DOCUMENT_ANALYSIS] = worker
        try:
            assert step.entered.wait(timeout=30), "工作线程没有进入解析阶段。"

            routes_jobs.stop_job_workers(timeout=SHORT_TIMEOUT)
            routes_jobs.stop_job_workers(timeout=SHORT_TIMEOUT)

            assert registry.get(routes_jobs.KIND_DOCUMENT_ANALYSIS) is worker, (
                "第二次失败的关闭之后句柄被丢弃了；"
                "重试并没有让失败变得可恢复，只是把它推迟了一次。"
            )
            assert routes_jobs.stuck_workers() == [worker]
        finally:
            step.release()
            routes_jobs.stop_job_workers(timeout=LONG_TIMEOUT)

    def test_a_worker_displaced_by_a_new_start_stays_reachable(self, store, registry):
        """The kind slot holds one worker, so a live one moves rather than dies.

        A second lifespan starting while the first worker's step is unfinished is
        an ordinary thing for a test session to do. Overwriting the registry
        entry would orphan the live thread; keeping it in place would leave the
        new store with no worker at all. Stranding it does neither.
        """
        first_step = ParkedStep()
        first = make_worker(store, first_step)
        store.submit(routes_jobs.KIND_DOCUMENT_ANALYSIS, {})
        first.start()
        registry[routes_jobs.KIND_DOCUMENT_ANALYSIS] = first
        try:
            assert first_step.entered.wait(timeout=30), "工作线程没有进入解析阶段。"

            # A new lifespan for the same kind, but a different store — which is
            # what the next test module's fixture looks like from here.
            second = make_worker(store, first_step, kind=routes_jobs.KIND_QA)
            routes_jobs._WORKERS[routes_jobs.KIND_QA] = second
            routes_jobs._STRANDED.append(first)

            assert first in routes_jobs.stranded_workers(), (
                "被顶下来的 worker 没有留在 _STRANDED 里；它成了无主线程。"
            )
            assert first in routes_jobs.active_workers(), (
                "一个仍在运行的线程没有被 active_workers() 报告。"
            )
            assert first in routes_jobs.stuck_workers()

            # And the stranding is not the end of the story: a shutdown collects
            # it along with everything else.
            first_step.release()
            assert first_step.finished.wait(timeout=30), "步骤没有结束。"
            routes_jobs.stop_job_workers(timeout=LONG_TIMEOUT)
            assert routes_jobs.stranded_workers() == []
        finally:
            first_step.release()
            routes_jobs.stop_job_workers(timeout=LONG_TIMEOUT)


class TestShutdownIsIdempotent:
    """A lifespan's ``finally`` runs whether or not startup succeeded."""

    def test_shutting_down_twice_is_the_same_as_once(self, store, registry):
        worker = idle_worker(store)
        worker.start()
        registry[routes_jobs.KIND_DOCUMENT_ANALYSIS] = worker

        first = routes_jobs.stop_job_workers(timeout=LONG_TIMEOUT)
        second = routes_jobs.stop_job_workers(timeout=LONG_TIMEOUT)

        assert first == [worker]
        assert second == [], "第二次关闭没有停到任何 worker；它是无操作，不是重复。"
        assert routes_jobs.active_workers() == []
        assert routes_jobs.stuck_workers() == []

    def test_a_repeated_failed_shutdown_is_still_idempotent(self, store, registry):
        """Idempotency has to survive the failure branch too.

        A shutdown path that runs twice on a stuck worker is the *normal* case —
        the lifespan's ``finally`` plus an explicit teardown — so the second call
        must not double-count, must not raise, and must not make things worse.
        """
        step = ParkedStep()
        worker = make_worker(store, step)
        store.submit(routes_jobs.KIND_DOCUMENT_ANALYSIS, {})
        worker.start()
        registry[routes_jobs.KIND_DOCUMENT_ANALYSIS] = worker
        try:
            assert step.entered.wait(timeout=30), "工作线程没有进入解析阶段。"

            for _ in range(3):
                routes_jobs.stop_job_workers(timeout=SHORT_TIMEOUT)

            assert routes_jobs.stuck_workers() == [worker], (
                "多次失败的关闭之后 worker 列表重复了；"
                "同一个 worker 被报告了不止一次。"
            )
            assert registry.get(routes_jobs.KIND_DOCUMENT_ANALYSIS) is worker
        finally:
            step.release()
            routes_jobs.stop_job_workers(timeout=LONG_TIMEOUT)

    def test_shutting_down_an_empty_registry_does_nothing(self, registry):
        assert routes_jobs.stop_job_workers(timeout=LONG_TIMEOUT) == []
        assert routes_jobs.stuck_workers() == []
        assert routes_jobs.stranded_workers() == []


class TestTheApplicationLifespanPath:
    """The same property, through the code a real deployment actually runs.

    ``app.stop_workers`` is the only shutdown path a process has. It used to log
    the length of ``stop_job_workers``' return value as "stopped N workers", which
    under the old ordering counted workers it had merely *tried* to stop — so a
    teardown in which every join timed out produced exactly the log line a
    perfectly clean teardown produced.
    """

    @staticmethod
    def _app_with_a_parked_worker(store, registry, step: ParkedStep) -> object:
        """Register a parked worker under the app's real kind, then return a worker arg.

        The lifespan is exercised through ``app_module.stop_workers`` rather than
        by booting a whole ``TestClient``: booting one would start two more real
        workers against a real store and make the test slow without making it
        more honest. What is under test is the decision ``stop_workers`` makes
        about what to log, and that decision is taken entirely from the registry.
        """
        worker = make_worker(store, step)
        store.submit(routes_jobs.KIND_DOCUMENT_ANALYSIS, {})
        worker.start()
        registry[routes_jobs.KIND_DOCUMENT_ANALYSIS] = worker
        return worker

    def test_the_lifespan_teardown_reports_a_worker_that_would_not_stop(
        self, store, registry, caplog
    ):
        step = ParkedStep()
        worker = self._app_with_a_parked_worker(store, registry, step)
        try:
            assert step.entered.wait(timeout=30), "工作线程没有进入解析阶段。"

            with caplog.at_level(logging.DEBUG, logger=app_module.__name__):
                app_module.stop_workers(None, [worker])

            errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
            assert errors, (
                "lifespan 关闭吞掉了一个没有停止的 worker；"
                "这是本次要修的第二个位置。"
            )
            assert any(worker.worker_id in r.getMessage() for r in errors), (
                f"错误日志没有点名未停止的 worker：{[r.getMessage() for r in errors]}"
            )
            assert not [r for r in caplog.records if r.levelno == logging.INFO], (
                "关闭失败时不应同时打印“已停止 N 个 worker”的 INFO 行；"
                "那正是把一次失败读成一次干净关闭的那一行。"
            )
            # And the handle survives the teardown, so a later shutdown can retry.
            assert registry.get(routes_jobs.KIND_DOCUMENT_ANALYSIS) is worker
        finally:
            step.release()
            routes_jobs.stop_job_workers(timeout=LONG_TIMEOUT)

    def test_the_lifespan_teardown_reports_a_clean_stop_normally(self, store, registry, caplog):
        """The quiet branch, so the loud one means something."""
        worker = idle_worker(store)
        worker.start()
        registry[routes_jobs.KIND_DOCUMENT_ANALYSIS] = worker

        with caplog.at_level(logging.DEBUG, logger=app_module.__name__):
            app_module.stop_workers(None, [worker])

        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert routes_jobs.KIND_DOCUMENT_ANALYSIS not in registry
        assert routes_jobs.stuck_workers() == []

    def test_a_real_lifespan_teardown_leaves_no_worker_behind(self, tmp_path, registry):
        """The end-to-end version, through ``TestClient`` and the real pipelines.

        Nothing is parked here, so both of the app's real workers find an empty
        queue and stop immediately. What this pins is that the reordered registry
        is compatible with the lifespan itself — the app starts, serves, and tears
        down, and after the teardown the registry is genuinely empty rather than
        merely cleared. The old code satisfied that assertion while leaking, so
        the added weight here is that ``stuck_workers()`` is asked too.
        """
        harness = open_store(TIER_PERSONAL, tmp_path, name=f"lifespan-{uuid.uuid4().hex[:8]}.db")
        try:
            application = app_module.create_app()
            application.dependency_overrides[routes_jobs.get_job_store] = lambda: harness.store
            with TestClient(application) as client:
                client.get("/api/health")
                assert routes_jobs.active_workers(), "lifespan 没有启动任何 worker。"
            assert routes_jobs.stuck_workers() == [], (
                "lifespan 关闭之后仍有 worker 线程在运行："
                f"{[w.worker_id for w in routes_jobs.stuck_workers()]}"
            )
            assert routes_jobs.active_workers() == []
        finally:
            harness.close()
