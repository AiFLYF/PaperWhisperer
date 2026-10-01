#!/usr/bin/env python3
"""Drive each T4.4 failure scenario for real, and report whether it is observable.

    python tools/fault_inject.py --list
    python tools/fault_inject.py worker-restart
    python tools/fault_inject.py --all
    python tools/fault_inject.py --all --json

T4.4's acceptance criterion is "故障场景均有明确可观察行为，无静默丢失" — every
failure scenario has a defined, observable behaviour, and nothing is lost
silently. A criterion like that cannot be discharged by a test suite that
asserts the happy path, because the failure being absent is indistinguishable
from the failure being unhandled. So this tool *causes* each failure and then
asks the only question that matters: **after this, can anybody tell what
happened?**

A scenario counts as observable only if all three of these hold:

1. The failure is *reached* — the tool asserts it actually got there, because a
   scenario that did not fire passes every assertion written about it.
2. The outcome is *visible* to somebody with the access the product gives them:
   a job row, an SSE event, an API response, an error message. A log line in a
   file nobody opens is a record, not an observation.
3. The outcome is *distinguishable* from the healthy case. A degraded request
   that looks exactly like a fast one has not degraded observably; it has
   degraded invisibly, which is the failure this task exists to name.

Anything that fails those is reported in the ``silent`` list, and the process
exits non-zero. A tool that always exits 0 is a tool that has learned to say
"fine" about everything.

**The faults are real.** A worker restart is a real child process calling
``os._exit`` in the middle of a real stage against a real SQLite file — the
approach ``tests/test_jobs.py`` already established, generalised here to the
death points that file does not cover. A MinerU failure is a real HTTP server
on a real socket returning a real 429 / 402 / hanging, not a monkeypatched
``urlopen``. A database lock is a real second connection holding
``BEGIN EXCLUSIVE``. A "duplicate submission" is six real processes racing the
same idempotency key. A fault-injection test that mocks the fault is a test of
the mock, and every one of these could have been mocked in about a line.

**It is importable, and the tests import it.** ``tests/test_fault_injection.py``
asserts against the same functions this CLI runs, so the scenarios that ship in
the suite and the scenarios an operator can reproduce by hand are the same code
— one implementation, two entry points, no second copy to drift.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paperwhisperer.infra import jobs as J  # noqa: E402
from paperwhisperer.infra.stores import build_stores  # noqa: E402

#: Used to decide whether a dead lease has expired without sleeping 300 seconds.
#: Tests that sleep out a real lease are slow *and* flaky; asking the question
#: with a clock is the arrangement ``tests/test_jobs.py`` established.
FUTURE = 3600.0


# --------------------------------------------------------------------------
# What a scenario reports back
# --------------------------------------------------------------------------


@dataclasses.dataclass
class ScenarioResult:
    """One scenario's verdict, in the shape the acceptance criterion asks for.

    ``observable`` is the headline and is computed, not asserted: a scenario
    reports ``True`` only when it reached the fault *and* the evidence it
    collected covers every dimension it claimed to. That is deliberate — a
    hand-written ``observable=True`` in the return statement is exactly the
    kind of thing that survives a regression it should have caught.
    """

    scenario: str
    #: What the product promises will happen.
    expected: str
    #: What actually happened, as the user or operator would see it.
    observed: str
    #: The machine-checkable facts behind ``observed``.
    evidence: dict[str, Any] = dataclasses.field(default_factory=dict)
    #: The checks ``expected`` promised, and whether each held.
    checks: dict[str, bool] = dataclasses.field(default_factory=dict)
    #: Things a user or operator cannot see. Non-empty means the scenario failed
    #: the acceptance criterion even if every other check passed.
    silent: list[str] = dataclasses.field(default_factory=list)
    #: A failure so severe the scenario did not even reach its assertion.
    fatal: str = ""

    @property
    def reached(self) -> bool:
        """Whether the fault itself was provoked.

        Checked separately from the other assertions because a scenario that
        silently failed to reproduce is the most dangerous kind of green test:
        every assertion after it passes for the wrong reason.
        """
        return self.checks.get("reached", False)

    @property
    def observable(self) -> bool:
        return bool(
            self.reached
            and not self.fatal
            and all(self.checks.values())
            and not self.silent
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "expected": self.expected,
            "observed": self.observed,
            "evidence": self.evidence,
            "checks": self.checks,
            "silent": list(self.silent),
            "fatal": self.fatal,
            "reached": self.reached,
            "observable": self.observable,
        }


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------


@dataclasses.dataclass
class _Settings:
    """The configuration slice the store factory reads.

    A private object rather than the application's settings object: this tool
    must not depend on, or disturb, whatever the developer's ``.env`` says, and
    a test that inherits ``STORAGE_SQLITE_PATH`` from a real checkout writes
    into the developer's actual database.
    """

    STORAGE_BACKEND: str = "sqlite"
    STORAGE_LEXICAL_FTS5: str = "auto"
    STORAGE_SQLITE_PATH: str = ""
    STORAGE_SQLITE_TIMEOUT_SECONDS: int = 30


def open_store(db_path: str) -> Any:
    """A personal-tier job store on a real file at ``db_path``.

    Returns the store, not the bundle, because every scenario here needs only
    the store and three of them hand the *path* to a child process — a child
    pointed at a different file would find an empty queue and never die, which
    is a very confusing way to pass a recovery test.
    """
    bundle = build_stores(_Settings(STORAGE_SQLITE_PATH=str(db_path)))
    return J.build_job_store(bundle=bundle)


def document_pipeline(**overrides: Any) -> J.JobPipeline:
    """The six-stage spine, with each stage's body replaceable.

    Every stage not named in ``overrides`` is a :func:`J.passthrough_step` that
    records "该阶段无需执行。" rather than a silent no-op — a stage that did
    nothing and said so is a different answer from a stage that was never
    reached, and a scenario that runs this pipeline needs both to be visible.
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


def recovered_worker(store: Any, worker_id: str = "healthy-worker") -> J.JobWorker:
    """A fresh worker on the same store — what an operator gets after a restart."""
    return J.JobWorker(store, document_pipeline(), worker_id=worker_id, lease_seconds=30)


def scratch(workdir: Path) -> Path:
    """The scenario's own directory, created.

    Each scenario runs against its own SQLite file, and the reason is not
    tidiness: a shared file would let one scenario's leftover job be claimed by
    another scenario's worker, and the resulting failure looks exactly like the
    bug the scenario was written to catch.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    return workdir


# --------------------------------------------------------------------------
# The child process that really dies
# --------------------------------------------------------------------------

#: A worker that dies at a named point, in its own process, against a real
#: database file. ``os._exit`` rather than ``sys.exit``: nothing unwinds, no
#: ``finally`` runs, no ``atexit`` fires, and the process image simply stops
#: existing. A graceful shutdown is not the failure the release gate is about.
#:
#: Three death points, because "crashed" is three different events with three
#: different recovery paths and only one of them was covered before:
#:
#: ``mid-stage``
#:     Inside a step, after writing something durable. The common case.
#: ``before-first-step``
#:     Claimed, then dead before entering anything. The job reads as ``queued``
#:     but carries a lease, which is the quietest of the three.
#: ``far-lease``
#:     Dead while holding a lease whose deadline is 300 seconds away — the
#:     shipped ``JOBS_LEASE_SECONDS`` default.
_DYING_WORKER = '''
import os
import sys

sys.path.insert(0, {project_root!r})

from paperwhisperer.infra import jobs as J
from paperwhisperer.infra.stores import build_stores


class _Settings:
    """Accepts keywords because the parent passes the database path in.

    A plain class rather than a dataclass because the child must not depend on
    anything the parent imported — a child that fails on an import is a child
    that never reaches the fault, and "exit code 1" is indistinguishable from
    "the worker returned cleanly" unless the test checks for 9.
    """

    STORAGE_BACKEND = "sqlite"
    STORAGE_LEXICAL_FTS5 = "auto"
    STORAGE_SQLITE_TIMEOUT_SECONDS = 30

    def __init__(self, **overrides):
        for name, value in overrides.items():
            setattr(self, name, value)


def mark(path, job_id):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(job_id + "\\n")
        handle.flush()
        os.fsync(handle.fileno())


def main():
    db_path, marker, where, kind = sys.argv[1:5]
    store = J.build_job_store(bundle=build_stores(_Settings(STORAGE_SQLITE_PATH=db_path)))

    if where == "before-first-step":
        # Claim and stop. No stage, no transition, no heartbeat: the job is
        # leased and still reads as queued.
        claimed = store.claim_next("doomed-worker", 300, kind=kind)
        if claimed is None:
            raise SystemExit(2)
        mark(marker, claimed.job_id)
        os._exit(9)

    if where == "far-lease":
        claimed = store.claim_next("doomed-worker", 300, kind=kind)
        store.transition(claimed.job_id, J.STATE_PARSING)
        store.patch(claimed.job_id, {{"lease_expires_at": __import__("time").time() + 300}})
        mark(marker, claimed.job_id)
        os._exit(9)

    def die_midway(context):
        context.update(context={{"parsed_bytes": 4096}})
        context.heartbeat()
        mark(marker, context.job.job_id)
        os._exit(9)

    pipeline = J.JobPipeline(
        kind=kind, steps=(J.JobStep(state=J.STATE_PARSING, run=die_midway),)
    )
    J.JobWorker(store, pipeline, worker_id="doomed-worker", lease_seconds=1).run_once()
    raise SystemExit("the worker was supposed to die")


main()
'''


def kill_a_worker_at(db_path: str, marker_path: Any, where: str,
                     kind: str = "document_analysis") -> subprocess.CompletedProcess:
    """Run a worker in its own process and kill it at ``where``.

    Returns the completed process so the caller can assert the exit code is 9
    rather than 0 — a worker that returned cleanly is not a worker that died,
    and every downstream assertion would then pass for the wrong reason.
    """
    script = Path(db_path).with_name(f"dying_worker_{where}.py")
    script.write_text(
        _DYING_WORKER.format(project_root=str(PROJECT_ROOT)), encoding="utf-8"
    )
    return subprocess.run(
        [sys.executable, str(script), str(db_path), str(marker_path), where, kind],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


# --------------------------------------------------------------------------
# Scenario 1 — worker restart
# --------------------------------------------------------------------------


def scenario_worker_restart(workdir: Path) -> ScenarioResult:
    """Kill a worker at three points, and ask whether each death is visible.

    This is the release gate: "worker/API 重启后任务可恢复或明确进入可重试终态".
    Three death points, because the gate's promise is two-ended — recoverable
    *or* explicitly retryable-terminal — and which end you get depends entirely
    on the state the corpse left behind. The mid-stage case was already covered
    by ``tests/test_jobs.py``; the other two were open, and they behave
    differently enough that "the recovery test passes" says nothing about them.

    **One database file per death point.** A shared queue would let the healthy
    worker in case (a) drain the corpse from case (b), and the resulting failure
    looks exactly like the bug case (b) exists to catch.
    """
    result = ScenarioResult(
        scenario="worker-restart",
        expected=(
            "任务在 worker 中断后可恢复或明确进入可重试终态，"
            "且用户能分辨『worker 还活着在跑』与『worker 已死』"
        ),
        observed="",
    )
    workdir = scratch(workdir)
    try:
        # -- (a) died inside a stage: recovered and completed ---------------
        mid_db = workdir / "mid_stage.db"
        mid_store = open_store(str(mid_db))
        mid_job = mid_store.submit("document_analysis", {"file_path": "a.pdf"},
                                   max_attempts=2).job
        mid_proc = kill_a_worker_at(str(mid_db), workdir / "mid_marker.txt", "mid-stage")
        corpse = mid_store.get_job(mid_job.job_id)
        result.evidence["mid_stage"] = {
            "exit_code": mid_proc.returncode,
            "state_left": corpse.state,
            "lease_owner": corpse.lease_owner,
            "context_written": corpse.context,
        }
        result.checks["reached"] = mid_proc.returncode == 9
        result.checks["the_claim_is_invisible_to_the_queue"] = (
            mid_store.claim_next("probe", 60, kind="document_analysis") is None
        )
        # A real clock is the wrong question here: the dead worker held a 1s
        # lease and this test must not depend on how fast the machine is.
        mid_store.recover_stale(now=time.time() + FUTURE)
        requeued = mid_store.get_job(mid_job.job_id)
        finished_count = recovered_worker(mid_store).drain()
        finished = mid_store.get_job(mid_job.job_id)
        result.evidence["mid_stage_after_recovery"] = {
            "requeued_to": requeued.state,
            "drained": finished_count,
            "final": finished.state,
            "attempts": finished.attempts,
        }
        result.checks["mid_stage_recovers_and_completes"] = (
            requeued.state == J.STATE_QUEUED
            and finished_count == 1
            and finished.state == J.STATE_COMPLETED
        )
        result.checks["mid_stage_trail_names_the_dead_worker"] = any(
            "doomed-worker" in event.message for event in finished.events
        )
        mid_store.close()

        # -- (b) died between the claim and the first stage ------------------
        gap_db = workdir / "claim_gap.db"
        gap_store = open_store(str(gap_db))
        gap_job = gap_store.submit("document_analysis", {"file_path": "a.pdf"},
                                   max_attempts=2).job
        gap_proc = kill_a_worker_at(str(gap_db), workdir / "gap_marker.txt", "before-first-step")
        leased = gap_store.get_job(gap_job.job_id)
        result.evidence["claim_gap_corpse"] = {
            "exit_code": gap_proc.returncode,
            "state_left": leased.state,
            "lease_owner": leased.lease_owner,
            "attempts": leased.attempts,
        }
        result.checks["reached"] = result.checks["reached"] and gap_proc.returncode == 9
        result.checks["a_claimed_but_unstarted_job_is_queued"] = leased.state == J.STATE_QUEUED
        gap_store.recover_stale(now=time.time() + FUTURE)
        released = gap_store.get_job(gap_job.job_id)
        result.evidence["claim_gap_after_recovery"] = {
            "state": released.state,
            "lease_owner": released.lease_owner,
            "attempts": released.attempts,
            "note": released.events[-1].note,
        }
        result.checks["claim_gap_lease_is_released"] = (
            released.state == J.STATE_QUEUED and not released.lease_owner
        )
        # The attempt the dead worker consumed stands, and recovery does not
        # compare it against ``max_attempts`` on this path. So a worker that
        # keeps dying in this window is re-claimed forever: neither exit the
        # gate offers. The trail is honest about each death — this is not a
        # silent loss of *information*, it is an unbounded wait.
        cycles = 0
        probe = released
        while probe.state == J.STATE_QUEUED and cycles < 8:
            gap_store.claim_next("doomed-again", 1, kind="document_analysis")
            gap_store.patch(gap_job.job_id, {"lease_expires_at": time.time() - 1})
            gap_store.recover_stale(now=time.time() + FUTURE)
            probe = gap_store.get_job(gap_job.job_id)
            cycles += 1
        result.evidence["claim_gap_repeat"] = {
            "cycles": cycles,
            "state": probe.state,
            "attempts": probe.attempts,
            "max_attempts": probe.max_attempts,
            "terminal": probe.terminal,
        }
        result.checks["claim_gap_reaches_a_terminal_state"] = probe.terminal
        if not probe.terminal:
            result.silent.append(
                f"worker 反复死在 claim 与首个 step 之间时，任务被无限重新领取："
                f"{cycles} 次中断后 attempts={probe.attempts} 已远超 "
                f"max_attempts={probe.max_attempts}，state 仍为 {probe.state}，"
                "recovery 的这一分支根本不检查重试预算——release gate 说的"
                "『可恢复或明确进入可重试终态』两个出口都不成立。"
            )
        gap_store.close()

        # -- (c) dead holding a lease with the shipped 300s deadline ---------
        far_db = workdir / "far_lease.db"
        far_store = open_store(str(far_db))
        far_job = far_store.submit("document_analysis", {"file_path": "a.pdf"},
                                   max_attempts=2).job
        far_proc = kill_a_worker_at(str(far_db), workdir / "far_marker.txt", "far-lease")
        far = far_store.get_job(far_job.job_id)
        result.evidence["far_lease_corpse"] = {
            "exit_code": far_proc.returncode,
            "state_left": far.state,
            "lease_owner": far.lease_owner,
            "seconds_until_the_lease_expires": round(far.lease_expires_at - time.time()),
        }
        result.checks["reached"] = result.checks["reached"] and far_proc.returncode == 9
        started = time.monotonic()
        moved_far = far_store.recover_stale()  # the real clock, not a faked one
        elapsed = time.monotonic() - started
        result.evidence["far_lease_recover_stale"] = {
            "moved": len(moved_far),
            "seconds": round(elapsed, 4),
        }
        result.checks["far_lease_waits_for_its_deadline"] = (
            len(moved_far) == 0 and elapsed < 1.0
        )
        result.checks["nothing_can_claim_it_in_the_meantime"] = (
            far_store.claim_next("healthy", 60, kind="document_analysis") is None
        )
        # Is the wait observable? A live worker's job and a dead worker's job
        # differ only in the lease, and the lease is not in the projection a
        # subscriber sees. That is the question this whole scenario turns on.
        # Submitted with the same ``max_attempts`` as the corpse, so the two
        # projections are comparable field for field rather than differing on
        # a setting this comparison has nothing to do with.
        live = far_store.submit(
            "document_analysis", {"file_path": "b.pdf"}, max_attempts=far_job.max_attempts
        ).job
        far_store.claim_next("live-worker", 300, kind="document_analysis")
        far_store.transition(live.job_id, J.STATE_PARSING)
        live_view = far_store.get_job(live.job_id).to_state_dict()
        dead_view = far.to_state_dict()
        # Compared field by field, and with the volatile fields excluded — and
        # the exclusion is stated rather than hidden, because "the two dicts are
        # not equal" is satisfied by a timestamp and would hide the finding.
        #
        # ``job_id`` is the identity the client subscribes by and is *supposed*
        # to differ. ``created_at``/``updated_at``/``sequence`` are a
        # wall-clock stamp and a cursor: two jobs created a second apart differ
        # in them for reasons that have nothing to do with whether a worker is
        # alive. Every remaining field is something a client could branch on,
        # and none of them may differ.
        volatile = {"job_id", "created_at", "updated_at", "sequence"}
        differences = {
            key: [live_view.get(key), dead_view.get(key)]
            for key in set(live_view) | set(dead_view)
            if live_view.get(key) != dead_view.get(key)
        }
        lease_keys = sorted(
            {"lease_owner", "lease_expires_at"} & set(live_view)
        )
        result.evidence["far_lease_projection"] = {
            "keys": sorted(live_view),
            "fields_excluded_as_volatile": sorted(volatile),
            "lease_fields_present": lease_keys,
            "differences": differences,
            "differences_that_could_mean_something": sorted(set(differences) - volatile),
        }
        semantic_differences = sorted(set(differences) - volatile)
        # True when the two can be told apart, which is what a subscriber
        # needs. Today the only field that differs is the job id, and the
        # lease fields are absent entirely — so this is False, and the
        # branch below records why.
        result.checks["a_dead_job_is_distinguishable_from_a_live_one"] = bool(
            semantic_differences or lease_keys
        )
        if not semantic_differences:
            result.silent.append(
                "一个被死 worker 持有的任务与一个正被活 worker 执行的任务，"
                f"对订阅者呈现完全相同的 state 字典（state={dead_view['state']!r}，"
                f"message={dead_view['message']!r}）：租约与持有者不在状态投影里。"
                f"用户要等 {far.lease_expires_at - time.time():.0f}s 才会看到任何变化，"
                "而这整段时间里它和一个正在解析的任务无法区分。"
            )
        # Nothing is lost once the deadline passes — that half of the gate holds.
        far_store.patch(far_job.job_id, {"lease_expires_at": time.time() - 1})
        far_store.recover_stale(now=time.time() + FUTURE)
        result.evidence["far_lease_after_the_deadline"] = far_store.get_job(
            far_job.job_id
        ).state
        result.checks["far_lease_recovers_once_the_deadline_passes"] = (
            far_store.get_job(far_job.job_id).state == J.STATE_QUEUED
        )
        far_store.close()

        result.observed = (
            f"3 个真实子进程均以 os._exit(9) 死亡。步骤内崩溃可恢复并完成"
            f"（{finished.state}，attempts={finished.attempts}，"
            f"trail 记录了死掉的 worker）；claim 后、首个 step 前崩溃时租约被释放"
            f"但重试预算不被检查，{cycles} 轮后 attempts={probe.attempts} 仍非终态；"
            f"持有 300s 租约崩溃时 recover_stale 在 {elapsed:.4f}s 内无动作，"
            f"且该任务与活 worker 执行的任务状态投影除 job_id 外逐字段相同。"
        )
    except Exception as exc:  # noqa: BLE001 - the tool reports, it does not crash
        result.fatal = f"{type(exc).__name__}: {exc}"
        result.observed = f"场景未能完成：{result.fatal}"
    return result


# --------------------------------------------------------------------------


def scenario_worker_race(workdir: Path) -> ScenarioResult:
    """Six real processes claim from one queue at the same instant.

    The question the release gate's double-spend concern asks: if two workers
    wake together, does exactly one get the job? Getting it wrong is not a
    crash — it is a second cloud parse billed twice for the same document, which
    is the exact loss the idempotency key exists to prevent arriving through the
    other door.
    """
    result = ScenarioResult(
        scenario="worker-race",
        expected="同一时刻多个 worker 竞争时，恰好一个领取成功，attempts 只 +1",
        observed="",
    )
    workdir = scratch(workdir)
    db = workdir / "race.db"
    store = open_store(str(db))
    try:
        for index in range(12):
            store.submit("document_analysis", {"n": index}, max_attempts=3)
        queued = store.count_by_state()[J.STATE_QUEUED]
        # A barrier, because "start them in a loop" is not a race — it is a
        # queue. Every process blocks until every other one is also ready.
        script = workdir / "racer.py"
        script.write_text(
            _RACER.format(project_root=str(PROJECT_ROOT), db_path=str(db)),
            encoding="utf-8",
        )
        procs = [
            subprocess.Popen(
                [sys.executable, str(script), str(index)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for index in range(6)
        ]
        outputs = [proc.communicate(timeout=120) for proc in procs]
        claims: list[str] = []
        errors: list[str] = []
        for out, err in outputs:
            if out.strip():
                claims.extend(json.loads(out.strip()).get("claimed") or [])
            if err.strip():
                errors.append(err.strip().splitlines()[-1])
        duplicates = len(claims) - len(set(claims))
        attempts = {
            job_id: store.get_job(job_id).attempts for job_id in set(claims)
        }
        result.evidence = {
            "queued_before": queued,
            "processes": len(procs),
            "exit_codes": [proc.returncode for proc in procs],
            "claims": claims,
            "distinct_claims": len(set(claims)),
            "duplicate_claims": duplicates,
            "attempts_after": attempts,
            "errors": errors,
        }
        result.checks["reached"] = len(procs) == 6 and any(claims)
        result.checks["exactly_one_worker_per_job"] = duplicates == 0
        result.checks["no_double_spend"] = all(value == 1 for value in attempts.values())
        result.checks["no_process_crashed"] = not errors
        result.observed = (
            f"{len(procs)} 个独立进程同时争抢 {queued} 个排队任务，"
            f"成功领取 {len(claims)} 个，其中重复领取 {duplicates} 次，"
            f"每个任务的 attempts 均为 {sorted(set(attempts.values())) or '—'}。"
        )
    except Exception as exc:  # noqa: BLE001 - the tool reports, it does not crash
        result.fatal = f"{type(exc).__name__}: {exc}"
        result.observed = f"场景未能完成：{result.fatal}"
    finally:
        store.close()
    return result


_RACER = '''
import json
import sys
import threading

sys.path.insert(0, {project_root!r})

from paperwhisperer.infra import jobs as J
from paperwhisperer.infra.stores import build_stores


class _Settings:
    STORAGE_BACKEND = "sqlite"
    STORAGE_LEXICAL_FTS5 = "auto"
    STORAGE_SQLITE_PATH = {db_path!r}
    STORAGE_SQLITE_TIMEOUT_SECONDS = 30


def main():
    index = sys.argv[1]
    store = J.build_job_store(bundle=build_stores(_Settings()))
    barrier = threading.Barrier(2)
    claimed = []
    lock = threading.Lock()

    def grab():
        barrier.wait(timeout=30)
        # Two claims in one process, from two threads, against one file: the
        # store is reached from a second connection, not the first thread's.
        got = store.claim_next("worker-" + index, 60, kind="document_analysis")
        with lock:
            if got is not None:
                claimed.append(got.job_id)

    threads = [threading.Thread(target=grab) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    print(json.dumps({{"claimed": claimed}}))


main()
'''


# --------------------------------------------------------------------------
# Scenario 2 — database connection exhaustion
# --------------------------------------------------------------------------


def scenario_db_exhaustion(workdir: Path) -> ScenarioResult:
    """Take the write lock away and see what the job store does.

    The personal tier is SQLite: one writer, and the plan's own risk table says
    so. The question this scenario answers is not "how fast can it go" but
    "**when the lock is gone, does the system say so**". A write that blocks
    until ``busy_timeout`` and then raises is a visible failure. A write that
    blocks until ``busy_timeout`` and then returns success is not.
    """
    result = ScenarioResult(
        scenario="db-exhaustion",
        expected="写锁被占用时写入要么成功要么抛出明确错误，绝不静默丢弃",
        observed="",
    )
    workdir = scratch(workdir)
    db = workdir / "exhaustion.db"
    store = open_store(str(db))
    try:
        keepalive = store.submit("document_analysis", {"file_path": "a.pdf"}).job

        # -- (a) contention that resolves: does the write actually land? -----
        # 16 threads x 20 transactions of real work against one real file.
        # The point is not throughput, it is that every committed write is
        # readable afterwards — the difference between "slow" and "lost".
        def burst(worker_index: int, done: list[int], failed: list[str]) -> None:
            for item in range(20):
                try:
                    store.submit("document_analysis", {"w": worker_index, "i": item})
                    done.append(1)
                except Exception as exc:  # noqa: BLE001 - the assertion is the count
                    failed.append(f"{type(exc).__name__}: {exc}")

        done: list[int] = []
        failed: list[str] = []
        threads = [threading.Thread(target=burst, args=(n, done, failed)) for n in range(16)]
        started = time.monotonic()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)
        burst_seconds = time.monotonic() - started
        total_rows = sum(store.count_by_state().values())
        result.evidence["burst_writes"] = 16 * 20
        result.evidence["burst_committed"] = len(done)
        result.evidence["burst_failed"] = len(failed)
        result.evidence["burst_seconds"] = round(burst_seconds, 3)
        result.evidence["burst_writes_per_second"] = round(len(done) / max(burst_seconds, 1e-9))
        result.evidence["rows_in_table"] = total_rows
        result.checks["reached"] = len(done) > 0
        result.checks["no_write_silently_dropped"] = (
            len(done) + len(failed) == 16 * 20 and len(failed) == 0
        )
        result.checks["every_committed_write_is_readable"] = total_rows >= len(done)

        # -- (b) the lock never comes back: what does a write do? ------------
        # A *separate* connection holding BEGIN EXCLUSIVE is the real shape of
        # this failure — a stale lock, a long batch write, an fsync stall.
        # A short busy_timeout keeps the scenario fast; the shipped default is
        # 30s and the behaviour is the same, only later.
        store.connection.execute("PRAGMA busy_timeout=250")
        hog = sqlite3.connect(str(db), isolation_level=None, timeout=30)
        hog.execute("PRAGMA busy_timeout=30000")
        hog.execute("BEGIN EXCLUSIVE")
        hog.execute(
            "INSERT INTO jobs(job_id, state, kind, payload, created_at, updated_at) "
            "VALUES('lock-hog', 'queued', 'document_analysis', '{}', 'x', 'x')"
        )
        locked_at = time.monotonic()
        error: str = ""
        try:
            store.submit("document_analysis", {"file_path": "blocked.pdf"})
        except Exception as exc:  # noqa: BLE001 - the exception IS the observation
            error = f"{type(exc).__name__}: {exc}"
        blocked_for = time.monotonic() - locked_at
        result.evidence["blocked_seconds_before_raising"] = round(blocked_for, 3)
        result.evidence["exception"] = error
        result.checks["reached"] = result.checks["reached"] and bool(error)
        result.checks["lock_contention_raises_loudly"] = bool(error)
        result.checks["it_is_sqlite_not_a_generic_error"] = "locked" in error.lower()
        if not error:
            result.silent.append(
                "写锁被占用时 submit() 仍然返回成功——写入被静默丢弃，"
                "调用方以为任务已入队。"
            )
        # And the store must still be usable the moment the lock clears, with
        # nothing half-written left behind.
        hog.execute("ROLLBACK")
        hog.close()
        recovered = store.submit("document_analysis", {"file_path": "after.pdf"})
        result.evidence["usable_after_the_lock_cleared"] = recovered.job.state
        result.evidence["original_job_survived"] = store.get_job(keepalive.job_id) is not None
        result.checks["store_usable_after"] = recovered.job.state == J.STATE_QUEUED
        result.checks["no_orphaned_rows"] = store.count_by_state()[J.STATE_QUEUED] >= 1

        result.observed = (
            f"并发 16 线程 x 20 次写：{len(done)} 次提交、{len(failed)} 次失败，"
            f"耗时 {burst_seconds:.2f}s（{len(done) / max(burst_seconds, 1e-9):.0f} 写/秒），"
            f"表内 {total_rows} 行，读回数量与提交数量一致。"
            f"写锁被另一个连接独占时，submit 在 {blocked_for:.2f}s 后抛出 "
            f"{error or '（无异常）'}，锁释放后存储立即可用。"
        )
    except Exception as exc:  # noqa: BLE001 - the tool reports, it does not crash
        result.fatal = f"{type(exc).__name__}: {exc}"
        result.observed = f"场景未能完成：{result.fatal}"
    finally:
        store.close()
    return result


# --------------------------------------------------------------------------
# Scenario 3 — MinerU: 429, quota exhaustion, timeout
# --------------------------------------------------------------------------


def scenario_mineru_failures(workdir: Path) -> ScenarioResult:
    """Three real HTTP failures from a real server, and three distinct answers.

    These are three different problems wearing the same "the cloud call did not
    work" costume, and they must not collapse into one:

    * **429** — the provider asked us to slow down. Retrying is right, the
      message must not claim a quota problem, and the job is worth re-running.
    * **402 / quota exhausted** — the account is out of credit. Retrying wastes
      the user's time and burns a quota check that will also fail. The message
      must tell them to top up.
    * **timeout** — the cloud is slower than the deadline. The work may well
      still finish; the honest answer is "stopped waiting", not "failed".

    The server here is a real ``ThreadingHTTPServer`` on a real socket. Every
    HTTP status, every ``Retry-After`` header and every timeout is produced by
    the network stack, not by a replacement ``urlopen``.
    """
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from paperwhisperer.core import config
    from paperwhisperer.ingestion.adapters import mineru_cloud as M

    result = ScenarioResult(
        scenario="mineru-failures",
        expected=(
            "429（限流，可重试）、额度耗尽（不可重试，需充值）、超时（可稍后重试）"
            "三种失败各自映射到不同的结果与不同的用户可见文案"
        ),
        observed="",
    )
    workdir = scratch(workdir)
    state = {"mode": "429", "calls": 0}
    audit = workdir / "mineru_audit.jsonl"
    document = workdir / "paper.pdf"
    document.write_bytes(b"%PDF-1.4\n% tiny\n")
    #: Never downloaded — the partial-recovery path is decided on the *URL* being
    #: present, and the download leg is guarded by an SSRF check that rightly
    #: refuses a loopback host. Asserting the decision point rather than
    #: disabling the check is the honest version of this test.
    result_zip = "https://cdn.example.invalid/result.zip"

    class Handler(BaseHTTPRequestHandler):
        """Answers with the failure under test. No framework, no mocking."""

        protocol_version = "HTTP/1.1"

        def log_message(self, *_args: Any) -> None:  # noqa: D102 - silence the test
            return

        def _reply(self, status: int, body: bytes = b"", headers: dict | None = None):
            self.send_response(status)
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body:
                self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's name
            length = int(self.headers.get("content-length", 0) or 0)
            if length:
                self.rfile.read(length)
            state["calls"] += 1
            mode = state["mode"]
            if mode == "429":
                self._reply(429, b"", {"Retry-After": "1"})
            elif mode == "402":
                self._reply(402, b"")
            elif mode == "business-quota":
                self._reply(
                    200,
                    json.dumps({"code": 1002, "msg": "insufficient balance"}).encode(),
                    {"Content-Type": "application/json"},
                )
            elif mode == "business-generic":
                self._reply(
                    200,
                    json.dumps({"code": 1001, "msg": "file not found"}).encode(),
                    {"Content-Type": "application/json"},
                )
            else:  # "ok"
                self._reply(
                    200,
                    json.dumps(
                        {"code": 0, "data": {"batch_id": "b1", "file_urls": ["https://oss.example/x"]}}
                    ).encode(),
                    {"Content-Type": "application/json"},
                )

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's name
            """The poll leg. ``pending`` forever is what a slow cloud job looks like."""
            state["calls"] += 1
            if state["mode"] == "pending":
                self._reply(
                    200,
                    json.dumps(
                        {"code": 0, "data": {"extract_result": [{"state": "pending"}]}}
                    ).encode(),
                    {"Content-Type": "application/json"},
                )
            elif state["mode"] == "pending-then-done":
                # The partial-recovery path: the result landed after we gave up
                # waiting, and the second lookup finds it.
                if state["calls"] > 2:
                    self._reply(
                        200,
                        json.dumps(
                            {
                                "code": 0,
                                "data": {
                                    "extract_result": [
                                        {"state": "done", "full_zip_url": result_zip}
                                    ]
                                },
                            }
                        ).encode(),
                        {"Content-Type": "application/json"},
                    )
                else:
                    self._reply(
                        200,
                        json.dumps(
                            {"code": 0, "data": {"extract_result": [{"state": "pending"}]}}
                        ).encode(),
                        {"Content-Type": "application/json"},
                    )
            elif state["mode"] == "cloud-failed":
                self._reply(
                    200,
                    json.dumps(
                        {"code": 0, "data": {"extract_result": [{"state": "failed"}]}}
                    ).encode(),
                    {"Content-Type": "application/json"},
                )
            else:
                self._reply(404, b"")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"

    # Config is set on the module for the duration and restored after, because
    # ``_load_settings`` reads module attributes and this tool must not leave a
    # loopback base URL in somebody's running process.
    saved = {
        name: getattr(config, name)
        for name in (
            "DOCUMENT_PRIVACY_MODE",
            "DOCUMENT_CLOUD_CONSENT_GRANTED",
            "MINERU_API_TOKEN",
            "MINERU_API_BASE_URL",
            "MINERU_MAX_RETRIES",
            "MINERU_POLL_INTERVAL_SECONDS",
            "MINERU_POLL_TIMEOUT_SECONDS",
        )
    }
    config.DOCUMENT_PRIVACY_MODE = "allow_cloud"
    config.DOCUMENT_CLOUD_CONSENT_GRANTED = True
    config.MINERU_API_TOKEN = "fault-injection-token"
    config.MINERU_API_BASE_URL = base_url
    config.MINERU_MAX_RETRIES = 2
    config.MINERU_POLL_INTERVAL_SECONDS = 1
    config.MINERU_POLL_TIMEOUT_SECONDS = 1

    try:
        outcomes: dict[str, dict[str, Any]] = {}
        # The sleeper is recorded rather than slept, so the backoff schedule is
        # asserted without the test waiting out a real ``Retry-After``.
        slept: list[float] = []

        def probe(mode: str) -> dict[str, Any]:
            state["mode"] = mode
            state["calls"] = 0
            slept.clear()
            settings = M._load_settings()
            try:
                M.request_upload_urls(
                    filename="paper.pdf", settings=settings, sleeper=slept.append
                )
                return {"outcome": "success", "code": "", "message": "", "calls": state["calls"],
                        "slept": list(slept)}
            except M.MineruError as exc:
                return {
                    "outcome": "failed",
                    "code": exc.code,
                    "status": exc.status,
                    "message": str(exc),
                    "calls": state["calls"],
                    "slept": list(slept),
                }

        for mode in ("429", "402", "business-quota", "business-generic", "ok"):
            outcomes[mode] = probe(mode)
        result.evidence["outcomes"] = outcomes

        result.checks["reached"] = outcomes["429"]["outcome"] == "failed"
        result.checks["429_is_retried"] = outcomes["429"]["calls"] > 1
        result.checks["429_honours_retry_after"] = 1.0 in outcomes["429"]["slept"]
        result.checks["429_is_not_reported_as_quota"] = (
            outcomes["429"]["code"] != "mineru_quota_exhausted"
        )
        result.checks["quota_is_not_retried"] = outcomes["402"]["calls"] == 1
        result.checks["quota_has_its_own_code"] = (
            outcomes["402"]["code"] == "mineru_quota_exhausted"
        )
        result.checks["business_quota_is_short_circuited"] = (
            outcomes["business-quota"]["calls"] == 1
            and outcomes["business-quota"]["code"] == "mineru_quota_exhausted"
        )
        result.checks["a_non_quota_business_error_is_not_quota"] = (
            outcomes["business-generic"]["code"] != "mineru_quota_exhausted"
        )
        result.checks["a_healthy_call_still_succeeds"] = outcomes["ok"]["outcome"] == "success"
        result.checks["the_three_messages_differ"] = len(
            {
                outcomes["429"]["message"],
                outcomes["402"]["message"],
                "MinerU 解析在 1 秒内未完成，已放弃等待。",
            }
        ) == 3

        # -- the third failure: the cloud is slower than the deadline --------
        # Driven through the real poll loop against a server that answers 200
        # with ``pending`` forever. ``poll_batch`` returns ``None`` rather than
        # raising, and that ``None`` is the whole design: it is the difference
        # between "degraded" and "failed", and it is what lets the caller look
        # once more before giving up.
        settings = M._load_settings()
        state["mode"] = "ok"
        batch, _upload = M.request_upload_urls(
            filename="paper.pdf", settings=settings, sleeper=lambda _s: None
        )
        timed = dataclasses.replace(settings, poll_interval_seconds=1, poll_timeout_seconds=1)

        state["mode"] = "pending"
        state["calls"] = 0
        poll_started = time.monotonic()
        poll_result = M.poll_batch(batch, settings=timed, sleeper=time.sleep)
        poll_seconds = time.monotonic() - poll_started
        recovered_url = M.fetch_partial_result_url(batch, settings=timed, sleeper=time.sleep)
        result.evidence["poll_timeout"] = {
            "poll_batch_returned": poll_result,
            "seconds_before_giving_up": round(poll_seconds, 2),
            "second_lookup_found": recovered_url,
        }
        result.checks["poll_timeout_returns_none_not_an_error"] = poll_result is None
        result.checks["poll_timeout_actually_waits_out_its_deadline"] = poll_seconds >= 0.9
        result.checks["nothing_to_recover_means_no_partial"] = recovered_url is None

        # The same server, except the result lands a moment after we stop
        # waiting. This is the degradation that makes the timeout worth having
        # at all: the user gets their document and is told it may be incomplete.
        state["mode"] = "pending-then-done"
        state["calls"] = 0
        late_poll = M.poll_batch(batch, settings=timed, sleeper=time.sleep)
        late_recovered = M.fetch_partial_result_url(batch, settings=timed, sleeper=time.sleep)
        result.evidence["late_result_recovery"] = {
            "poll_batch_returned": late_poll,
            "second_lookup_found": bool(late_recovered),
        }
        result.checks["a_late_result_is_recovered_as_partial"] = (
            late_poll is None and bool(late_recovered)
        )

        # A cloud job that reports its own failure is a fourth answer, and it
        # must not be dressed up as a timeout.
        state["mode"] = "cloud-failed"
        cloud_error: dict[str, Any] = {}
        try:
            M.poll_batch(batch, settings=timed, sleeper=time.sleep)
        except M.MineruError as exc:
            cloud_error = {"code": exc.code, "message": str(exc)}
        result.evidence["cloud_failure_state"] = cloud_error
        result.checks["a_cloud_failure_is_not_reported_as_a_timeout"] = (
            cloud_error.get("code") == "mineru_task_failed"
        )

        # And the audit record a quota exhaustion produces, end to end through
        # ``parse_document`` — the one place the audit is written.
        state["mode"] = "402"
        state["calls"] = 0
        _doc, record = M.parse_document(
            str(document),
            source_filename="paper.pdf",
            trigger_reason="fallback",
            sleeper=lambda _s: None,
            audit_log_path=str(audit),
        )
        result.evidence["audit_record"] = record.to_dict()
        result.evidence["audit_lines"] = audit.read_text(encoding="utf-8").count("\n")
        result.checks["quota_reaches_the_audit_log_as_quota"] = (
            record.outcome == "quota_exhausted"
        )
        result.checks["quota_makes_exactly_one_call_end_to_end"] = state["calls"] == 1
        result.checks["the_audit_never_carries_the_token"] = (
            "fault-injection-token" not in json.dumps(record.to_dict(), ensure_ascii=False)
        )

        # -- the layer where the three answers stop being distinguishable ----
        # The adapter separates 429 / quota / timeout correctly, which is the
        # half that is easy. The other half is whether that separation survives
        # the trip to whoever reads the result: the orchestrator's cloud bridge
        # is the only place it can be lost, and it is the only place a UI's
        # "充值" button or its "retry" button can be wired from.
        from paperwhisperer.analysis.orchestrator import build_parse_provenance
        from paperwhisperer.ingestion import orchestrator as O

        def user_facing(outcome: str, message: str) -> dict[str, Any]:
            """What the ingestion layer hands upward for one adapter outcome."""

            class _Inner:
                name = "mineru_cloud"
                version = "test"

                def parse(self, _path: str, _options: Any) -> tuple[Any, Any]:
                    class _Record:
                        pass

                    record = _Record()
                    record.outcome = outcome
                    record.message = message
                    return None, record

            bridge = O._CloudParserBridge(
                _Inner(), source_filename="paper.pdf", trigger_reason="fallback"
            )
            bridge.gate = O.CloudFallbackDecision(allowed=True, reason_code="allow_cloud",
                                                 message="")
            parsed = bridge.parse(str(document), None)
            # ``ParseResult`` has no ``to_dict`` — the wire shape is assembled by
            # the caller — so the fields a caller could branch on are read off
            # the dataclass directly. That is the point being measured: which
            # machine-readable facts exist at all, not how they are serialised.
            rendered = {
                "status": parsed.status,
                "error_message": parsed.error_message,
                "warnings": list(parsed.warnings),
                "diagnostics": list(parsed.diagnostics),
            }
            rendered["provenance"] = build_parse_provenance(
                O.ParseOutcome(
                    status=parsed.status,
                    attempts=[],
                    warnings=list(parsed.warnings),
                    error_message=parsed.error_message,
                    cloud_attempted=True,
                )
            )
            return rendered

        surfaced = {
            "429": user_facing("failed", outcomes["429"]["message"]),
            "quota": user_facing("quota_exhausted", outcomes["402"]["message"]),
            "timeout": user_facing(
                "timed_out", "MinerU 解析在 1 秒内未完成，已放弃等待。"
            ),
        }
        result.evidence["user_facing"] = {
            name: {
                "status": payload.get("status"),
                "error_message": payload.get("error_message"),
                "diagnostics": payload.get("diagnostics"),
                "provenance_keys": sorted(payload["provenance"]),
            }
            for name, payload in surfaced.items()
        }
        statuses = {payload.get("status") for payload in surfaced.values()}
        result.checks["the_adapter_outcomes_reach_the_caller_as_diagnostics"] = all(
            any(
                item.get("cloud_outcome") == expected
                for item in payload.get("diagnostics") or []
            )
            for expected, payload in (
                ("failed", surfaced["429"]),
                ("quota_exhausted", surfaced["quota"]),
                ("timed_out", surfaced["timeout"]),
            )
        )
        result.checks["the_three_share_one_status"] = len(statuses) == 1
        result.checks["a_ui_can_branch_on_the_outcome"] = len(
            {
                json.dumps(
                    {
                        key: payload.get(key)
                        for key in ("status", "diagnostics")
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                for payload in surfaced.values()
            }
        ) == 3
        provenance_carries_outcome = all(
            "cloud_outcome" in json.dumps(payload["provenance"], ensure_ascii=False)
            for payload in surfaced.values()
        )
        if len(statuses) == 1 and not provenance_carries_outcome:
            result.silent.append(
                f"三种 MinerU 失败在适配器层区分正确（各自的 code 与文案都不同），"
                f"但穿过编排层的云端桥之后全部塌缩成同一个 status={next(iter(statuses))!r}；"
                "机器可读的 cloud_outcome 只留在 ParseResult.diagnostics 里，"
                "而 build_parse_provenance 构造的用户可见载荷（parse 溯源块）"
                "既不含该字段、也不含 error_message——"
                "界面无法按『需要充值』/『可重试』/『稍后再看』分支处理，"
                "只能显示一段中文散文，用户要靠读懂『额度不足』四个字才知道发生了什么。"
            )
        result.checks["cloud_outcome_reaches_the_user_payload"] = provenance_carries_outcome
        result.checks["the_error_message_reaches_the_user_payload"] = all(
            "error_message" in payload["provenance"] for payload in surfaced.values()
        )

        result.observed = (
            f"429：{outcomes['429']['calls']} 次调用后失败，"
            f"退避 {outcomes['429']['slept']}（遵守 Retry-After），"
            f"code={outcomes['429']['code']}，文案「{outcomes['429']['message']}」；"
            f"额度耗尽：{outcomes['402']['calls']} 次调用（不重试），"
            f"code={outcomes['402']['code']}，文案「{outcomes['402']['message']}」；"
            f"轮询超时：{poll_seconds:.1f}s 后 poll_batch 返回 None，补查可捡回则降级为 "
            f"partial，否则报 timed_out；云端自报 failed 不会被误当成超时；"
            f"业务码 1002 同样短路为额度耗尽，1001 不误判。"
            f"但三者穿过编排层后 status 均为 {sorted(statuses)[0]!r}，"
            "只有 diagnostics 里的 cloud_outcome 还能区分。"
        )
    except Exception as exc:  # noqa: BLE001 - the tool reports, it does not crash
        result.fatal = f"{type(exc).__name__}: {exc}"
        result.observed = f"场景未能完成：{result.fatal}"
    finally:
        server.shutdown()
        for name, value in saved.items():
            setattr(config, name, value)
    return result


# --------------------------------------------------------------------------
# Scenario 4 — SSE reconnect
# --------------------------------------------------------------------------


def scenario_sse_reconnect(workdir: Path) -> ScenarioResult:
    """Drop a subscriber mid-run, reconnect with a cursor, and check for gaps.

    The Phase 2 exit criterion is "浏览器刷新/SSE 断开不丢任务". It is really a
    question about :class:`~paperwhisperer.infra.jobs.JobSubscription`, and it
    is answerable only because the feed is *state*: numbered, monotonic, and
    carrying no payload. So the test that matters is a client that misses
    several transitions — not one, several — and still reconstructs the truth.
    """
    result = ScenarioResult(
        scenario="sse-reconnect",
        expected=(
            "客户端断线后凭 since_sequence 游标重连，恰好补齐错过的状态变更（含终态），"
            "既不丢也不重；且状态流不含任何内容载荷"
        ),
        observed="",
    )
    workdir = scratch(workdir)
    db = workdir / "reconnect.db"
    store = open_store(str(db))
    try:
        job = store.submit("document_analysis", {"file_path": "a.pdf"}).job
        subscription = J.JobSubscription(store, job.job_id, poll_interval=0.01)
        snapshot = subscription.snapshot()
        result.evidence["snapshot_on_first_connect"] = snapshot
        first_page = subscription.poll()
        result.evidence["first_page"] = [event["state"] for event in first_page]
        cursor = first_page[-1]["sequence"]

        # The connection drops here. The subscription object is simply not
        # polled again: no stop_event, no teardown, nothing on the other side
        # aware that a client existed. Whatever happens next happens without it.
        unreached_before_drop = len(store.events_since(job.job_id, cursor))
        recovered_worker(store).drain()
        unreached_after = len(store.events_since(job.job_id, cursor))
        result.evidence["events_unreached_before_the_drop"] = unreached_before_drop
        result.checks["the_client_really_was_behind"] = unreached_after > 2

        full_trail = store.events_since(job.job_id, 0)
        resumed = J.JobSubscription(store, job.job_id, since_sequence=cursor, poll_interval=0.01)
        result.evidence["snapshot_on_reconnect"] = resumed.snapshot()
        missed = resumed.poll()
        result.evidence["missed_states"] = [event["state"] for event in missed]
        result.evidence["missed_count"] = len(missed)
        result.evidence["transitions_missed_while_away"] = len(
            [event for event in full_trail if event.sequence > cursor and event.changed]
        )

        seen = [event["sequence"] for event in first_page] + [
            event["sequence"] for event in missed
        ]
        result.checks["reached"] = bool(first_page) and len(missed) > 2
        result.checks["the_gap_is_recovered_exactly"] = seen == [
            event.sequence for event in full_trail
        ]
        result.checks["nothing_is_delivered_twice"] = len(seen) == len(set(seen))
        result.checks["the_terminal_event_arrives"] = bool(missed) and missed[-1]["terminal"]
        result.checks["the_snapshot_says_where_it_got_to"] = (
            resumed.snapshot()["state"] == J.STATE_COMPLETED
        )
        # A client that missed the terminal event must still learn how the run
        # ended, rather than hanging on a job that will never move again.
        finished_stream = list(
            J.JobSubscription(store, job.job_id, poll_interval=0.01).stream()
        )
        result.checks["a_late_subscriber_learns_the_ending"] = (
            bool(finished_stream) and finished_stream[-1]["terminal"]
        )
        # The reason the whole design works: state alone is enough.
        payload_leak = json.dumps(
            [event.to_state_dict() for event in full_trail], ensure_ascii=False
        )
        result.checks["no_payload_in_the_state_feed"] = "a.pdf" not in payload_leak
        result.checks["no_result_in_the_state_feed"] = "analysis_id" not in payload_leak

        result.observed = (
            f"客户端读到 {len(first_page)} 条后断线（游标 sequence={cursor}），"
            f"期间任务走完 {result.evidence['transitions_missed_while_away']} 次状态变更；"
            f"重连后 snapshot 直接给出 {resumed.snapshot()['state']}，"
            f"补发 {len(missed)} 条且以终态结束，两页序号拼接后与完整 trail 完全一致"
            f"（无丢失、无重复）。"
        )
    except Exception as exc:  # noqa: BLE001 - the tool reports, it does not crash
        result.fatal = f"{type(exc).__name__}: {exc}"
        result.observed = f"场景未能完成：{result.fatal}"
    finally:
        store.close()
    return result


def scenario_sse_disconnect_cleanup() -> ScenarioResult:
    """Close an SSE stream mid-flight and check the cleanup and the error path.

    Separate from the resync scenario on purpose: resync is about the client
    coming *back*, and this is about the client going *away* and the work it
    started finishing anyway. Both matter, and they fail for different reasons.
    """
    import asyncio

    from paperwhisperer.api.deps import pump_sync_events

    result = ScenarioResult(
        scenario="sse-disconnect",
        expected="客户端断开后 cleanup 必定执行，临时文件被删除，producer 跑完不泄漏",
        observed="",
    )
    try:
        state = {"cleaned": 0, "producer_finished": 0, "slept": []}

        def produce():
            yield {"type": "start", "data": {"session_id": "s-1"}}
            for index in range(3):
                time.sleep(0.05)
                yield {"type": "section", "data": {"n": index}}
            state["producer_finished"] += 1
            yield {"type": "done", "data": {"answer": "x"}}

        async def scenario() -> dict[str, Any]:
            stream = pump_sync_events(
                produce,
                lambda event: f"event: {event['type']}\ndata: {{}}\n\n",
                cleanup=lambda: state.__setitem__("cleaned", state["cleaned"] + 1),
            )
            first = await stream.__anext__()
            await stream.__anext__()
            started = time.monotonic()
            await stream.aclose()  # what Starlette does when the client vanishes
            return {
                "first_event": first,
                "aclose_seconds": round(time.monotonic() - started, 3),
            }

        result.evidence.update(asyncio.run(scenario()))
        result.evidence["cleanup_ran"] = state["cleaned"]
        result.evidence["producer_ran_to_completion"] = state["producer_finished"]
        result.checks["reached"] = state["cleaned"] == 1
        result.checks["cleanup_runs_on_disconnect"] = state["cleaned"] == 1
        result.checks["the_work_is_not_abandoned_mid_flight"] = state[
            "producer_finished"
        ] == 1
        result.checks["aclose_does_not_return_before_the_work_does"] = (
            result.evidence["aclose_seconds"] >= 0.1
        )
        result.observed = (
            f"客户端读到第 2 条事件后断开：cleanup 执行 {state['cleaned']} 次，"
            f"producer 仍跑完（落库 {state['producer_finished']}），"
            f"aclose 等待了 {result.evidence['aclose_seconds']}s 才返回——"
            "断开不会中断正在进行的分析，结果不会丢失。"
        )
    except Exception as exc:  # noqa: BLE001 - the tool reports, it does not crash
        result.fatal = f"{type(exc).__name__}: {exc}"
        result.observed = f"场景未能完成：{result.fatal}"
    return result


# --------------------------------------------------------------------------
# Scenario 5 — user cancellation
# --------------------------------------------------------------------------


def scenario_user_cancel(workdir: Path) -> ScenarioResult:
    """Cancel at three points, and check the answer tells the truth each time.

    Cancellation here is cooperative and the module says so. The honest cases
    are the interesting ones: a cancel that is *accepted but not yet honoured*,
    and a cancel that arrives during the final stage and therefore cannot be
    honoured — reported as ``completed``, because the work really was finished
    and calling that "cancelled" would be a lie the user discovers on the bill.
    """
    result = ScenarioResult(
        scenario="user-cancel",
        expected=(
            "取消被如实记录：未开始→cancelled；步骤间→在边界停止并 cancelled；"
            "最后一步→completed 且 cancel_requested=True（不谎称取消）"
        ),
        observed="",
    )
    workdir = scratch(workdir)
    db = workdir / "cancel.db"
    store = open_store(str(db))
    try:
        # -- (a) cancelled while queued: never runs a single stage ----------
        ran: list[str] = []
        queued = store.submit("document_analysis").job
        pipeline = document_pipeline(**{J.STATE_PARSING: lambda ctx: ran.append("parsing")})
        ack = store.request_cancel(queued.job_id)
        result.evidence["queued_ack"] = ack.to_dict()
        J.JobWorker(store, pipeline, lease_seconds=30).drain()
        queued_done = store.get_job(queued.job_id)
        result.evidence["queued_final"] = {
            "state": queued_done.state,
            "stages_run": ran,
        }
        result.checks["reached"] = ack.accepted and not ran
        result.checks["a_queued_cancel_never_runs_a_stage"] = not ran
        result.checks["a_queued_cancel_ends_cancelled"] = (
            queued_done.state == J.STATE_CANCELLED
        )

        # -- (b) cancelled mid-run: honoured at the next boundary ------------
        def parsing(ctx: J.JobContext) -> None:
            ctx.update(context={"parsed": True})

        def quality(ctx: J.JobContext) -> None:
            # Somebody cancels while this stage is working. The stage asks for a
            # boundary when it next has one, which is the only place a
            # cooperative check can go.
            store.request_cancel(ctx.job.job_id)
            ctx.boundary()

        mid = store.submit("document_analysis").job
        J.JobWorker(
            store,
            J.JobPipeline(
                kind="document_analysis",
                steps=(
                    J.JobStep(state=J.STATE_PARSING, run=parsing),
                    J.JobStep(state=J.STATE_QUALITY_CHECK, run=quality),
                    J.JobStep(state=J.STATE_FALLBACK, run=lambda ctx: None),
                ),
            ),
            lease_seconds=30,
        ).run_once()
        mid_done = store.get_job(mid.job_id)
        result.evidence["mid_run_final"] = {
            "state": mid_done.state,
            "cancel_requested": mid_done.cancel_requested,
            "stages_entered": [
                event.to_state for event in store.events_since(mid.job_id, 0) if event.changed
            ],
        }
        result.checks["a_mid_run_cancel_ends_cancelled"] = (
            mid_done.state == J.STATE_CANCELLED
        )
        result.checks["the_next_stage_does_not_start"] = (
            J.STATE_FALLBACK not in mid_done.message
            and J.STATE_FALLBACK
            not in [e.to_state for e in store.events_since(mid.job_id, 0)]
        )

        # -- (c) cancelled during the LAST stage: the work was done ---------
        def final(ctx: J.JobContext) -> None:
            store.request_cancel(ctx.job.job_id)
            ctx.update(result={"analysis_id": "run-1"})

        late = store.submit("document_analysis").job
        J.JobWorker(
            store,
            J.JobPipeline(
                kind="document_analysis",
                steps=(
                    J.JobStep(state=J.STATE_PARSING, run=lambda ctx: None),
                    J.JobStep(state=J.STATE_QUALITY_CHECK, run=final),
                ),
            ),
            lease_seconds=30,
        ).run_once()
        late_done = store.get_job(late.job_id)
        result.evidence["late_final"] = {
            "state": late_done.state,
            "cancel_requested": late_done.cancel_requested,
            "result": late_done.result,
        }
        result.checks["a_late_cancel_reports_completed_not_cancelled"] = (
            late_done.state == J.STATE_COMPLETED and late_done.cancel_requested
        )

        # -- (d) cancelling a finished job says so, rather than pretending ---
        after = store.request_cancel(late.job_id)
        result.evidence["cancel_after_finish"] = after.to_dict()
        result.checks["a_finished_job_reports_the_cancel_was_not_accepted"] = (
            after.accepted is False
        )
        result.checks["and_explains_why"] = "终态" in after.message

        result.observed = (
            f"排队中取消：{queued_done.state}，未执行任何步骤（ran={ran}），"
            f"ack.accepted={ack.accepted}；"
            f"运行中取消：{mid_done.state}，后续阶段未进入，"
            f"轨迹记录 {result.evidence['mid_run_final']['stages_entered']}；"
            f"最后一步取消：{late_done.state} 且 cancel_requested="
            f"{late_done.cancel_requested}（工作已完成，不谎称取消）；"
            f"对已完成任务再次取消：accepted={after.accepted}，文案「{after.message}」。"
        )
    except Exception as exc:  # noqa: BLE001 - the tool reports, it does not crash
        result.fatal = f"{type(exc).__name__}: {exc}"
        result.observed = f"场景未能完成：{result.fatal}"
    finally:
        store.close()
    return result


# --------------------------------------------------------------------------
# Scenario 6 — duplicate submission
# --------------------------------------------------------------------------


def scenario_duplicate_submit(workdir: Path) -> ScenarioResult:
    """Six real processes double-click the same button.

    Two properties, and the second is the one that is easy to miss: the
    duplicate must be *distinguishable*. A response byte-identical to a fresh
    submission tells a user who double-clicked nothing at all — the
    deduplication worked and the person never found out.
    """
    result = ScenarioResult(
        scenario="duplicate-submit",
        expected=(
            "并发重复提交只产生一个任务、只被领取一次，"
            "且重复响应带 deduplicated 标记与中文提示，与新提交可区分"
        ),
        observed="",
    )
    workdir = scratch(workdir)
    db = workdir / "duplicate.db"
    store = open_store(str(db))
    try:
        first = store.submit(
            "document_analysis", {"file_path": "a.pdf"}, idempotency_key="double-click"
        )
        script = workdir / "submitter.py"
        script.write_text(
            _SUBMITTER.format(project_root=str(PROJECT_ROOT), db_path=str(db)),
            encoding="utf-8",
        )
        procs = [
            subprocess.Popen(
                [sys.executable, str(script), str(index)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for index in range(6)
        ]
        outputs = [proc.communicate(timeout=120) for proc in procs]
        replies = [json.loads(out.strip()) for out, _ in outputs if out.strip()]
        errors = [err.strip() for _, err in outputs if err.strip()]

        job_ids = {reply["job_id"] for reply in replies}
        fresh = [reply for reply in replies if not reply["deduplicated"]]
        notices = {reply["notice"] for reply in replies if reply["notice"]}
        again = store.submit(
            "document_analysis", {"file_path": "a.pdf"}, idempotency_key="double-click"
        )
        result.evidence = {
            "processes": len(procs),
            "exit_codes": [proc.returncode for proc in procs],
            "job_ids": sorted(job_ids),
            "fresh_submissions": len(fresh),
            "queued_rows": store.count_by_state()[J.STATE_QUEUED],
            "sample_notice": sorted(notices)[0] if notices else "",
            "distinct_notices": len(notices),
            "errors": errors,
        }
        result.checks["reached"] = len(replies) == 6
        result.checks["one_job_id_for_all_six"] = len(job_ids) == 1
        result.checks["all_reuse_the_original"] = job_ids == {first.job_id}
        result.checks["one_row_in_the_table"] = store.count_by_state()[J.STATE_QUEUED] == 1
        result.checks["the_duplicate_is_flagged"] = all(
            reply["deduplicated"] for reply in replies
        )
        result.checks["it_carries_a_user_readable_notice"] = bool(notices) and all(
            "重复提交" in notice for notice in notices
        )
        result.checks["the_duplicate_differs_from_a_fresh_one"] = (
            again.deduplicated
            and bool(again.notice)
            and not first.deduplicated
            and first.notice == ""
        )
        # And the thing deduplication exists to prevent: a second claim, hence
        # a second bill for the same cloud parse.
        store.claim_next("w1", 60, kind="document_analysis")
        second_claim = store.claim_next("w2", 60, kind="document_analysis")
        result.evidence["second_claim"] = (
            second_claim.job_id if second_claim is not None else None
        )
        result.evidence["attempts"] = store.get_job(first.job_id).attempts
        result.checks["claimed_exactly_once"] = second_claim is None
        result.checks["attempts_is_one"] = store.get_job(first.job_id).attempts == 1

        result.observed = (
            f"{len(replies)} 个独立进程并发提交同一幂等键，得到 {len(job_ids)} 个 job_id"
            f"（均为 {sorted(job_ids)[0][:8]}…），队列表中只有 "
            f"{store.count_by_state()[J.STATE_QUEUED]} 行，attempts="
            f"{store.get_job(first.job_id).attempts}（第二次领取返回 "
            f"{second_claim!r}）。重复响应带 deduplicated=true 与提示"
            f"「{sorted(notices)[0][:40] if notices else ''}…」，与新提交的 notice=\"\" 可区分。"
        )
    except Exception as exc:  # noqa: BLE001 - the tool reports, it does not crash
        result.fatal = f"{type(exc).__name__}: {exc}"
        result.observed = f"场景未能完成：{result.fatal}"
    finally:
        store.close()
    return result


_SUBMITTER = '''
import json
import sys

sys.path.insert(0, {project_root!r})

from paperwhisperer.infra import jobs as J
from paperwhisperer.infra.stores import build_stores


class _Settings:
    STORAGE_BACKEND = "sqlite"
    STORAGE_LEXICAL_FTS5 = "auto"
    STORAGE_SQLITE_PATH = {db_path!r}
    STORAGE_SQLITE_TIMEOUT_SECONDS = 30


def main():
    index = sys.argv[1]
    store = J.build_job_store(bundle=build_stores(_Settings()))
    result = store.submit(
        "document_analysis", {{"file_path": "a.pdf"}}, idempotency_key="double-click"
    )
    print(json.dumps({{
        "worker": index,
        "job_id": result.job_id,
        "deduplicated": result.deduplicated,
        "notice": result.notice,
    }}))


main()
'''


# --------------------------------------------------------------------------
# Registry and CLI
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Scenario 7 — does a recovery make any noise?
# --------------------------------------------------------------------------


def scenario_recovery_signal(workdir: Path) -> ScenarioResult:
    """Recover a job for real, then ask whether anything *counted* it.

    The release gate says a killed worker's task must be recoverable, and
    :mod:`paperwhisperer.infra.jobs` delivers that: the row is requeued, the
    trail names the dead worker, and the next worker finishes the job. What this
    scenario asks is the question the gate does not: **if that happened at 3am,
    would anything have noticed?**

    A recovery that is written to a database nobody queries and a log nobody
    tails has not really been surfaced. The ``jobs`` metric family exists
    precisely for this — ``jobs.recovered_total`` is documented in
    :mod:`paperwhisperer.core.observability` as "the loudest number in this
    family" — so the check is whether a real recovery moves it.
    """
    from paperwhisperer.core import observability as O

    result = ScenarioResult(
        scenario="recovery-signal",
        expected="一次真实的 worker 崩溃恢复会让 jobs.recovered_total 等指标 +1",
        observed="",
    )
    db = workdir / "signal.db"
    store = open_store(str(db))
    # The facade is a process-wide singleton and a test that leaves it enabled
    # changes what every later test in this session measures.
    was_enabled = O.observability.enabled
    try:
        workdir = scratch(workdir)
        O.observability.enabled = True
        O.observability.metrics.reset()

        job = store.submit("document_analysis", {"file_path": "a.pdf"}, max_attempts=2).job
        result.evidence["metric_names"] = {
            "recovered": O.observability.metrics.jobs_recovered.name,
            "transitions": O.observability.metrics.jobs_transitions.name,
            "terminal": O.observability.metrics.jobs_terminal.name,
        }
        result.evidence["before"] = {
            "recovered": O.observability.metrics.jobs_recovered.total,
            "transitions": O.observability.metrics.jobs_transitions.total,
            "terminal": O.observability.metrics.jobs_terminal.total,
        }

        # A real recovery, through the real store: claim, enter a stage, then
        # let the lease lapse. No worker process is killed here because the kill
        # is already covered by ``worker-restart``; what is under test here is
        # whether the recovery is *instrumented*, and that is a property of the
        # code around the recovery rather than of how the corpse was made.
        store.claim_next("doomed-worker", 1, kind="document_analysis")
        store.transition(job.job_id, J.STATE_PARSING)
        store.patch(job.job_id, {"lease_expires_at": time.time() - 1})
        moved = store.recover_stale(now=time.time() + FUTURE)
        recovered_worker(store).drain()

        after = {
            "recovered": O.observability.metrics.jobs_recovered.total,
            "transitions": O.observability.metrics.jobs_transitions.total,
            "terminal": O.observability.metrics.jobs_terminal.total,
        }
        result.evidence["recovery_moved_rows"] = [item.job_id for item in moved]
        result.evidence["after"] = after
        result.evidence["observability_enabled_by_default"] = was_enabled
        # "Reached" is about the fault, not about the answer: the recovery
        # happened and moved rows. Whether a counter moved is the finding.
        result.checks["reached"] = bool(moved)
        result.checks["a_recovery_moves_the_recovered_counter"] = (
            after["recovered"] > result.evidence["before"]["recovered"]
        )
        result.checks["a_recovery_moves_the_transition_counter"] = (
            after["transitions"] > result.evidence["before"]["transitions"]
        )
        result.checks["a_terminal_move_moves_the_terminal_counter"] = (
            after["terminal"] > result.evidence["before"]["terminal"]
        )
        if after == result.evidence["before"]:
            result.silent.append(
                "一次完整的『崩溃 → 恢复 → 重跑 → 完成』之后，"
                f"jobs.recovered_total / jobs.transitions_total / jobs.terminal_total "
                f"三个计数器全部纹丝不动（{after}）。"
                "指标已声明、可被 Prometheus 抓取，却没有任何调用点写入它们："
                "运维看板上会永远显示 0 次恢复，而不是『没有人恢复过』。"
            )
        if not was_enabled:
            result.silent.append(
                "可观测性默认关闭（observability.enabled = False），"
                "且应用启动路径上没有任何 configure() 调用——"
                "即使指标被正确写入，默认部署也一条都不会上报。"
            )
        result.observed = (
            f"恢复确实发生（{len(moved)} 个任务被回收并重跑至 "
            f"{store.get_job(job.job_id).state}），但指标计数 "
            f"recovered/transitions/terminal 全部为 "
            f"{after['recovered']}/{after['transitions']}/{after['terminal']}；"
            f"observability 默认 enabled={was_enabled}。"
        )
    except Exception as exc:  # noqa: BLE001 - the tool reports, it does not crash
        result.fatal = f"{type(exc).__name__}: {exc}"
        result.observed = f"场景未能完成：{result.fatal}"
    finally:
        O.observability.enabled = was_enabled
        O.observability.metrics.reset()
        store.close()
    return result


#: Every named scenario in T4.4, and where it came from. The plan's row reads
#: "worker 重启、DB 连接耗尽、MinerU 429/超时、SSE 断线重连、用户取消、重复提交"
#: — six items, seven scenarios, because "worker 重启" has three distinct death
#: points that recover through different code paths and only the mid-stage one
#: was covered.
SCENARIOS: dict[str, Callable[[Path], ScenarioResult]] = {
    "worker-restart": scenario_worker_restart,
    "worker-race": scenario_worker_race,
    "db-exhaustion": scenario_db_exhaustion,
    "mineru-failures": scenario_mineru_failures,
    "sse-reconnect": scenario_sse_reconnect,
    "sse-disconnect": lambda _workdir: scenario_sse_disconnect_cleanup(),
    "user-cancel": scenario_user_cancel,
    "duplicate-submit": scenario_duplicate_submit,
    "recovery-signal": scenario_recovery_signal,
}


def run_scenario(name: str, workdir: Path | None = None) -> ScenarioResult:
    """Run one scenario. The tests call this, so there is one implementation."""
    if name not in SCENARIOS:
        raise KeyError(f"未知场景：{name}（可用：{', '.join(sorted(SCENARIOS))}）")
    if workdir is None:
        with tempfile.TemporaryDirectory(prefix="pw-fault-") as temporary:
            return SCENARIOS[name](Path(temporary))
    workdir.mkdir(parents=True, exist_ok=True)
    return SCENARIOS[name](workdir)


def run_all(workdir: Path | None = None) -> list[ScenarioResult]:
    """Every scenario, each in its own directory so their databases cannot meet."""
    if workdir is None:
        with tempfile.TemporaryDirectory(prefix="pw-fault-") as temporary:
            root = Path(temporary)
            return [SCENARIOS[name](root / name) for name in SCENARIOS]
    return [SCENARIOS[name](workdir / name) for name in SCENARIOS]


def _render(result: ScenarioResult) -> str:
    lines = [
        f"── {result.scenario} "
        + ("可观察" if result.observable else "存在静默丢失"),
        f"   预期：{result.expected}",
        f"   实测：{result.observed}",
    ]
    for name, ok in result.checks.items():
        lines.append(f"   [{'PASS' if ok else 'FAIL'}] {name}")
    for item in result.silent:
        lines.append(f"   [SILENT] {item}")
    if result.fatal:
        lines.append(f"   [FATAL] {result.fatal}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="触发 T4.4 的每个故障场景，并检查其行为是否可观察。",
    )
    parser.add_argument(
        "scenario",
        nargs="?",
        choices=sorted(SCENARIOS),
        help="要运行的场景名称（省略则配合 --all 运行全部）",
    )
    parser.add_argument("--all", action="store_true", help="运行全部场景")
    parser.add_argument("--list", action="store_true", help="列出场景名称")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出")
    parser.add_argument(
        "--workdir", default="", help="把临时数据库写到这个目录（默认用系统临时目录）"
    )
    args = parser.parse_args(argv)

    if args.list:
        for name in sorted(SCENARIOS):
            print(name)
        return 0
    if not args.all and not args.scenario:
        parser.error("请指定一个场景，或使用 --all / --list")

    root = Path(args.workdir) if args.workdir else None
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
    if args.all:
        results = run_all(root)
    else:
        results = [run_scenario(args.scenario, (root / args.scenario) if root else None)]

    if args.json:
        print(json.dumps([result.to_dict() for result in results], ensure_ascii=False, indent=2))
    else:
        print("\n\n".join(_render(result) for result in results))
        passed = [r for r in results if r.observable]
        print(
            f"\n{len(passed)}/{len(results)} 个场景的失败可观察。"
            + ("" if len(passed) == len(results) else " 其余场景存在静默丢失，见上方 SILENT 行。")
        )
    # Non-zero when any scenario failed the acceptance criterion. A tool that
    # always exits 0 has learned to say "fine" about everything.
    return 0 if all(result.observable for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
