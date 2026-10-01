"""The worker starts in a *real* deployment, and a submitted job runs.

Every other suite that exercised the lifespan installed a
``dependency_overrides`` entry for :func:`routes_jobs.get_job_store`, which is
why the bug these tests exist for survived a large green suite: the branch only
tests take was covered, and the branch every real deployment takes was not. So
the tests here are built around one rule — **prefer the real resolution path,
and where an override is needed, say which seam it replaces and why.**

The properties under test:

1. with *no* overrides at all, the lifespan starts a worker;
2. the store the worker drains is the same store the routes read, resolved
   through whichever seam a caller overrode;
3. a job submitted into that store reaches ``completed`` on its own, with no
   request waiting on it;
4. the supported default configuration logs **no** ERROR at startup, and a
   genuinely broken store still logs one that names the backend and the
   consequence.

No test in this file calls a model, opens a socket, or reaches a developer's
real data directory.
"""

from __future__ import annotations

import logging

import pytest

from paperwhisperer.api import routes_evidence, routes_jobs
from paperwhisperer.app import _resolve_worker_store, create_app, start_workers
from paperwhisperer.core import config
from paperwhisperer.infra import jobs as J


@pytest.fixture(autouse=True)
def probe_store(tmp_path, monkeypatch):
    """An empty, temporary store for the whole module, and no shared singletons.

    :func:`routes_evidence.get_store_bundle` and :func:`routes_jobs.get_job_store`
    are process-wide singletons that build from ``config`` on first use and then
    cache forever. Left alone, the first test here would construct a bundle
    against the repository's real ``context/stores/`` path and every later test
    — and every later *suite* — would inherit it. Both are cleared on entry and
    restored on exit, so a worker started here writes only to ``tmp_path``.
    """
    monkeypatch.setattr(
        config, "STORAGE_SQLITE_PATH", str(tmp_path / "stores" / "probe.sqlite3")
    )
    monkeypatch.setattr(config, "STORAGE_BACKEND", "sqlite")
    previous_stores = getattr(routes_evidence, "_STORES", None)
    previous_jobs = dict(getattr(routes_jobs, "_JOB_STORES", {}) or {})
    routes_evidence._STORES = None
    routes_jobs._JOB_STORES = {}
    yield
    routes_evidence._STORES = previous_stores
    routes_jobs._JOB_STORES = previous_jobs
    routes_jobs.stop_job_workers()


@pytest.fixture
def probe_pipeline(monkeypatch):
    """A job kind whose stages do nothing, so a worker can run it offline.

    Registered *before* ``start_workers`` runs, because the startup path walks
    :data:`routes_jobs.PIPELINES` once and starts a worker per kind found there.
    """
    pipeline = J.JobPipeline(
        kind="startup_probe",
        steps=(
            J.passthrough_step(J.STATE_PARSING, "startup_probe: 不解析。"),
            J.passthrough_step(J.STATE_QUALITY_CHECK, "startup_probe: 不评估。"),
        ),
    )
    monkeypatch.setitem(routes_jobs.PIPELINES, "startup_probe", pipeline)
    return pipeline


# --------------------------------------------------------------------------
# The regression: the branch a real deployment takes
# --------------------------------------------------------------------------


def test_the_lifespan_starts_a_worker_with_no_dependency_overrides(probe_pipeline):
    """The bug, stated as a test.

    A real application object has an **empty** ``dependency_overrides`` table —
    the table only ever gets entries from a test. So "no override present" is
    not an edge case to be tolerant of, it is the normal deployment, and it was
    the one path that raised ``AttributeError``.

    The assertion is on workers actually running, not on a return value: a
    ``start_workers`` that returned a list without starting threads would pass
    a weaker test and fail every user.
    """
    application = create_app()
    assert application.dependency_overrides == {}, (
        "a real application must have no overrides; if this fails, the test is "
        "no longer testing the branch it exists to test"
    )

    started = start_workers(application)
    try:
        assert started, "no worker started with an empty override table"
        assert sorted(w.pipeline.kind for w in started) == sorted(routes_jobs.PIPELINES)
        for worker in started:
            assert worker._thread is not None and worker._thread.is_alive()
    finally:
        routes_jobs.stop_job_workers()


def test_a_submitted_job_runs_to_completion_with_no_request_waiting(probe_pipeline):
    """Not "the worker started" — "the worker did the work".

    The distinction is the whole investigation. A worker that logs and does
    nothing passes any assertion about threads being alive, and this is the
    assertion that distinguishes the two: the job is submitted directly against
    the store, nothing is polling on the caller's behalf, and only the worker
    thread can move it.
    """
    application = create_app()
    started = start_workers(application)
    assert started

    store = _resolve_worker_store(application)
    job_id = store.submit("startup_probe", {"probe": True}).job_id

    deadline = 30.0
    import time

    limit = time.monotonic() + deadline
    job = None
    while time.monotonic() < limit:
        job = store.get_job(job_id)
        if job is not None and job.terminal:
            break
        time.sleep(0.05)

    assert job is not None, "the job vanished from the store"
    assert job.state == J.STATE_COMPLETED, f"worker did not finish it: {job.summary_line()}"
    assert job.attempts == 1, "the job should have been claimed exactly once"
    # A terminal job must not still be leased: a lease left behind makes the
    # recovery pass keep fetching a row nobody is working on.
    assert job.lease_owner == ""
    # The trail has to show the work actually happening, not a job teleporting
    # from queued to completed.
    states = [event.to_state for event in job.events]
    assert J.STATE_PARSING in states, f"stages were skipped: {states}"
    assert J.STATE_QUALITY_CHECK in states, f"stages were skipped: {states}"
    assert states[-1] == J.STATE_COMPLETED


# --------------------------------------------------------------------------
# Why the old code failed, pinned so it cannot regress
# --------------------------------------------------------------------------


def test_calling_get_job_store_without_a_bundle_is_what_broke():
    """The mechanism, asserted rather than described.

    ``get_job_store``'s ``bundle`` parameter defaults to
    ``Depends(get_store_bundle)`` — a marker object, not a bundle. Calling it
    bare therefore hands the marker to ``build_job_store``, which then reads
    ``bundle.backend`` and raises ``AttributeError``.

    This test pins *why* the resolution has to pass a bundle explicitly. If a
    future FastAPI version changed the default, this test would start failing
    and the comment in ``_resolve_worker_store`` would need re-examining; that
    is the correct direction for it to fail in, because the alternative is a
    silent change of meaning in a function that decides where jobs are read.
    """
    import fastapi

    default = routes_jobs.get_job_store.__defaults__[0]
    assert isinstance(default, fastapi.params.Depends), (
        "get_job_store's bundle default is no longer a Depends marker; "
        "re-check how _resolve_worker_store must call it"
    )
    with pytest.raises(AttributeError):
        routes_jobs.get_job_store()


def test_an_override_of_the_inner_bundle_seam_is_honoured(tmp_path):
    """A caller may override ``get_store_bundle`` alone and get what it asked for.

    This is the second resolution branch, and it is the one that keeps the
    worker's store and the routes' store identical. The override returns a real
    :class:`StoreBundle` pointed at its own temporary file, and the assertion is
    that a job written through the resolved store lands in *that* file — which
    is the actual guarantee the original docstring was reaching for when it said
    the worker and the routes must be talking about the same database.

    Before the fix there was no such thing as this branch: the code only ever
    consulted the ``get_job_store`` override, so a caller who had overridden the
    one level down — which is what a test that wants a temporary database
    naturally does — got a worker pointed somewhere else entirely.
    """
    from paperwhisperer.infra.stores.factory import build_stores

    class _Settings:
        STORAGE_BACKEND = "sqlite"
        STORAGE_SQLITE_PATH = str(tmp_path / "overridden.sqlite3")
        STORAGE_SQLITE_TIMEOUT_SECONDS = 30
        STORAGE_LEXICAL_FTS5 = "auto"

    bundle = build_stores(_Settings())

    application = create_app()
    application.dependency_overrides[routes_evidence.get_store_bundle] = lambda: bundle
    try:
        store = _resolve_worker_store(application)
        store.submit("startup_probe", {})
        # Read it back through the same bundle a request would have used.
        from paperwhisperer.infra.jobs import build_job_store

        assert build_job_store(bundle=bundle).list_jobs() != []
        assert (tmp_path / "overridden.sqlite3").exists()
    finally:
        application.dependency_overrides.pop(routes_evidence.get_store_bundle, None)
        bundle.close()


def test_an_override_of_get_job_store_is_still_the_thing_returned():
    """The outer override keeps priority: it is the most specific seam a caller can override."""
    marker = object()
    application = create_app()
    application.dependency_overrides[routes_jobs.get_job_store] = lambda: marker
    try:
        assert _resolve_worker_store(application) is marker
    finally:
        application.dependency_overrides.pop(routes_jobs.get_job_store, None)


# --------------------------------------------------------------------------
# The log: quiet when fine, loud when broken
# --------------------------------------------------------------------------


def test_a_supported_startup_logs_no_error(probe_pipeline, caplog):
    """An ERROR on every boot of a supported configuration is itself a defect.

    The specific failure this guards against is a process that is *completely*
    healthy — health returns 200, every route answers — logging an ERROR line
    that says its job queue will never be drained. An operator who sees that
    once learns to ignore ERROR lines from this service, and then does not see
    the one that matters.
    """
    application = create_app()
    with caplog.at_level(logging.DEBUG, logger="paperwhisperer"):
        started = start_workers(application)
    try:
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert started, "nothing started, so this test proved nothing"
        assert errors == [], (
            "a healthy startup logged ERROR: "
            + "; ".join(record.getMessage() for record in errors)
        )
    finally:
        routes_jobs.stop_job_workers()


def test_an_unreachable_store_logs_an_error_that_names_backend_and_consequence(caplog):
    """When the store really is broken, ERROR is right — and the message must help.

    Two things are asserted, and the second is the one that was missing: the
    level, and that the message tells the operator *what to do*. The old text
    said only that long tasks would stay queued, which describes the symptom in
    the passive voice and names neither the cause nor the consequence.

    The override raises rather than returns ``None`` on purpose. ``None`` is not
    a broken store: :func:`build_job_store` treats a ``None`` bundle as "build me
    one" and cheerfully opens a fresh database, so an override returning ``None``
    would test a code path that cannot occur and quietly pass. A store that is
    *configured* and cannot be *opened* — the Hosted tier's database refusing
    connections — raises, and that is the case this log line is for.
    """
    application = create_app()

    def _unreachable():
        raise ConnectionError("could not connect to server: database is down")

    application.dependency_overrides[routes_evidence.get_store_bundle] = _unreachable

    try:
        with caplog.at_level(logging.DEBUG, logger="paperwhisperer.app"):
            started = start_workers(application)
        assert started == []
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(errors) == 1, f"expected exactly one ERROR, got {errors}"
        message = errors[0].getMessage()
        # Names the backend, so "the database is unreachable" is distinguishable
        # from "the code is broken".
        assert config.STORAGE_BACKEND in message
        # Names the consequence, so the reader knows users are affected.
        assert "queued" in message or "队列" in message
        # And keeps the cause: the exception text is what tells an operator
        # whether this is a config mistake or a database outage.
        assert "database is down" in message
    finally:
        application.dependency_overrides.pop(routes_evidence.get_store_bundle, None)


def test_jobs_disabled_stays_below_error(probe_pipeline, caplog):
    """Turning the queue off is a decision, and a decision is not an incident."""
    application = create_app()
    original = config.JOBS_ENABLED
    config.JOBS_ENABLED = False
    try:
        with caplog.at_level(logging.DEBUG, logger="paperwhisperer"):
            assert start_workers(application) == []
        assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []
    finally:
        config.JOBS_ENABLED = original