#!/usr/bin/env python3
"""Prove the job worker actually runs, in the deployment you point it at.

The question this answers is the one a log line cannot: **not** "did the worker
start" but "did a submitted job get claimed, run, and reach a terminal state".
The startup log is evidence that a thread was spawned; this is evidence that it
did the work. Those are different claims, and only the second one matters.

**Why this exists rather than another test.** A test asserts the wiring; this
asserts the deployment. The bug that prompted it lived in
``paperwhisperer.app.start_workers`` and was invisible to the entire test suite
because every test that exercised the lifespan installed a
``dependency_overrides`` entry for ``routes_jobs.get_job_store`` — so the suite
exercised the branch that only tests take, and the branch every real deployment
takes was never run. A probe that starts the *real* application through the
*real* lifespan, with no overrides and no monkeypatching, is the thing that
would have caught it, and it is worth having as a command an operator can run
against their own deployment.

**What it deliberately does not do.** It never calls a model and never uploads
a document. The submission uses :func:`probe_pipeline`, a two-stage
passthrough pipeline registered into :data:`routes_jobs.PIPELINES` for the
duration of the run. That is enough to exercise everything the worker does that
matters — claim, lease, the state machine, the event trail, the terminal
transition — without touching the network, so this is safe to run against a
live deployment and cheap enough to run as a smoke test.

Usage::

    # Personal tier (SQLite) — the shipped default, no configuration needed.
    python tools/worker_probe.py

    # Hosted tier. Uses a throwaway schema and drops it afterwards.
    PAPERWHISPERER_TEST_PG_DSN=postgresql://paperwhisperer:paperwhisperer@localhost:5432/\
paperwhisperer python tools/worker_probe.py --tier hosted

Exit status is ``0`` only if a job reached a terminal state. That is the whole
contract: a probe that exits ``0`` because the worker *started* is the same bug
in a different costume.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

if __package__ in (None, ""):  # run as a script rather than ``-m``
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paperwhisperer.api import routes_jobs  # noqa: E402
from paperwhisperer.core import config  # noqa: E402
from paperwhisperer.infra import jobs as J  # noqa: E402

#: The kind the probe submits under. Distinct from every shipped kind so a
#: probe job can never be claimed by — or confused with — a real one.
PROBE_KIND = "worker_probe"


def probe_pipeline() -> J.JobPipeline:
    """Two real stages that do nothing, so the state machine really runs.

    ``passthrough_step`` is used rather than a hand-rolled no-op so the probe
    traverses the same code a production pipeline does: the stage is entered,
    it raises :class:`~paperwhisperer.infra.jobs.StepSkipped`, the worker
    records the skip in the trail and carries on, and the job ends
    ``completed`` because nothing marked it partial. ``parsing`` and
    ``quality_check`` are adjacent on the spine, so the contiguity check the
    constructor enforces is satisfied without a fake edge — and that check is
    load-bearing here: an earlier draft of this file reached for ``indexing``
    instead and was correctly rejected, which is the pipeline guard doing its
    job on the one caller that bypasses the application entirely.
    """
    return J.JobPipeline(
        kind=PROBE_KIND,
        steps=(
            J.passthrough_step(
                J.STATE_PARSING, "worker_probe: 探测任务不解析任何文件。"
            ),
            J.passthrough_step(
                J.STATE_QUALITY_CHECK, "worker_probe: 探测任务不评估任何解析质量。"
            ),
        ),
    )


class ProbeOutcome:
    """What the run found, rendered on its own so exit status and prose agree."""

    def __init__(self, ok: bool, lines: list[str]) -> None:
        self.ok = ok
        self.lines = lines


def _start_workers(application) -> tuple[list, object]:
    """Run the real lifespan entry point and hand back what it started.

    ``start_workers`` is called directly rather than through a ``TestClient``
    so the probe does not need an HTTP stack, and so the thing under test is
    unambiguously the same function the application's lifespan calls.
    """
    from paperwhisperer.app import start_workers

    return start_workers(application), application


def run_probe(*, tier: str, timeout: float, workdir: Path | None = None) -> ProbeOutcome:
    """Start the workers, submit one probe job, and report whether it finished."""
    lines: list[str] = []
    started_warnings: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if record.levelno >= logging.WARNING:
                started_warnings.append(f"{record.levelname}: {record.getMessage()}")

    capture = _Capture()
    logging.getLogger("paperwhisperer").addHandler(capture)
    logging.getLogger("paperwhisperer").setLevel(logging.INFO)

    restore = _configure_tier(tier, workdir)
    cleanup_schema = None
    application = None
    try:
        from paperwhisperer.app import create_app

        application = create_app()

        # Registered *before* the workers start, and that ordering is the whole
        # point: ``start_job_worker`` walks ``PIPELINES`` once, at startup, and
        # starts one worker per kind it finds there. A probe kind registered
        # afterwards has no worker to claim it, and the job sits at ``queued``
        # forever — which is precisely the "starts, logs, and does nothing"
        # outcome this tool exists to detect, so the probe would have reported
        # its own construction error as a broken worker.
        if PROBE_KIND not in routes_jobs.PIPELINES:
            routes_jobs.PIPELINES[PROBE_KIND] = probe_pipeline()

        started, _ = _start_workers(application)
        kinds = sorted(w.pipeline.kind for w in started)
        lines.append(f"workers started: {kinds or '(none)'}")

        if not started:
            lines.append("FAIL: no worker started, so nothing can ever drain the queue.")
            lines.append("      Startup log above is the reason; see report for the cause.")
            return ProbeOutcome(False, lines)

        if PROBE_KIND not in kinds:
            lines.append(f"FAIL: no worker was started for the probe kind {PROBE_KIND!r}.")
            return ProbeOutcome(False, lines)

        # Resolve the store through the *same* helper the startup path used, so
        # the probe submits into the database the worker is draining. Calling
        # ``routes_jobs.get_job_store()`` bare would reintroduce the very bug this
        # probe exists to detect: its ``bundle`` parameter defaults to a
        # ``Depends`` marker, not a bundle.
        from paperwhisperer.app import _resolve_worker_store

        store = _resolve_worker_store(application)
        result = store.submit(PROBE_KIND, {"submitted_by": "tools/worker_probe.py"})
        job_id = result.job_id
        lines.append(f"submitted job {job_id} as kind {PROBE_KIND!r}")

        deadline = time.monotonic() + max(1.0, float(timeout))
        job = None
        while time.monotonic() < deadline:
            job = store.get_job(job_id)
            if job is not None and job.terminal:
                break
            time.sleep(0.05)

        if job is None:
            lines.append(f"FAIL: job {job_id} vanished from the store.")
            return ProbeOutcome(False, lines)

        lines.append(f"final state: {job.state} (attempts={job.attempts})")
        lines.append("trail: " + " -> ".join(event.to_state for event in job.events))

        ok = job.state == J.STATE_COMPLETED and job.lease_owner == ""
        if not ok:
            lines.append(f"FAIL: expected {J.STATE_COMPLETED}, got {job.state}.")
        if job.lease_owner:
            lines.append(f"FAIL: job still leased by {job.lease_owner!r}.")
        if not ok:
            return ProbeOutcome(False, lines)

        # A worker that completes a job but keeps its lease is still a bug, and
        # retention/queue-depth behaviour is only real if the worker ticks.
        counts = store.count_by_state()
        lines.append(f"queue depth: {counts}")
        return ProbeOutcome(True, lines)
    finally:
        routes_jobs.stop_job_workers()
        if application is not None:
            from paperwhisperer.app import stop_workers

            stop_workers(application, [])
        routes_jobs.PIPELINES.pop(PROBE_KIND, None)
        logging.getLogger("paperwhisperer").removeHandler(capture)
        cleanup_schema = restore()
        if cleanup_schema:
            lines.append(f"cleaned up schema {cleanup_schema}")


def _configure_tier(tier: str, workdir: Path | None):
    """Point the store tier at the requested backend. Returns a restore callable.

    The personal tier gets a scratch SQLite file rather than the operator's real
    database: the probe writes job rows, and a probe should not leave rows in
    the data directory it is trying to prove something about.
    """
    if tier != "hosted":
        root = workdir or Path(
            os.environ.get("PAPERWHISPERER_PROBE_DIR")
            or tempfile.mkdtemp(prefix="worker-probe-")
        )
        root.mkdir(parents=True, exist_ok=True)
        previous_path = config.STORAGE_SQLITE_PATH
        previous_backend = config.STORAGE_BACKEND
        config.STORAGE_SQLITE_PATH = str(root / "probe.sqlite3")
        config.STORAGE_BACKEND = "sqlite"

        def _restore_sqlite() -> str:
            config.STORAGE_SQLITE_PATH = previous_path
            config.STORAGE_BACKEND = previous_backend
            return ""

        return _restore_sqlite

    dsn = os.environ.get("PAPERWHISPERER_TEST_PG_DSN", "").strip()
    if not dsn:
        raise SystemExit(
            "Hosted 档位需要 PAPERWHISPERER_TEST_PG_DSN，"
            "例如 postgresql://paperwhisperer:paperwhisperer@localhost:5432/paperwhisperer"
        )
    schema = f"worker_probe_{uuid.uuid4().hex[:12]}"
    config.STORAGE_BACKEND = "postgres"
    config.STORAGE_PG_DSN = dsn
    config.STORAGE_PG_SCHEMA = schema

    def _restore_pg() -> str:
        _drop_schema(dsn, schema)
        return schema

    return _restore_pg


def _drop_schema(dsn: str, schema: str) -> None:
    """Remove a probe schema. Best-effort: a probe must not fail on cleanup."""
    try:
        import psycopg

        with psycopg.connect(dsn, autocommit=True) as connection:  # noqa: F841
            connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
    except Exception as exc:  # noqa: BLE001 - cleanup is not the test
        logging.getLogger(__name__).warning("清理探测 schema %s 失败：%s", schema, exc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prove the job worker runs.")
    parser.add_argument(
        "--tier", choices=("personal", "hosted"), default="personal",
        help="personal = SQLite (default); hosted = PostgreSQL via PAPERWHISPERER_TEST_PG_DSN",
    )
    parser.add_argument(
        "--timeout", type=float, default=30.0,
        help="How long to wait for the probe job to reach a terminal state.",
    )
    args = parser.parse_args(argv)

    outcome = run_probe(tier=args.tier, timeout=args.timeout)
    for line in outcome.lines:
        print(line)
    print("RESULT:", "PASS — a submitted job was claimed and completed." if outcome.ok
          else "FAIL — the worker did not complete a submitted job.")
    return 0 if outcome.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())