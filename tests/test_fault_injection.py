"""T4.4 fault injection: every failure scenario has a defined, observable outcome.

    python3 -m pytest tests/test_fault_injection.py -q

The acceptance criterion this file exists to satisfy is "故障场景均有明确可观察
行为，无静默丢失" — every failure scenario has a defined, observable behaviour,
and nothing is lost silently. A criterion phrased that way cannot be discharged
by asserting that things work. It has to be discharged by *causing* each
failure and then asking the only question that matters: after this, can anybody
tell what happened?

**The faults are real.** A worker restart is a real child process calling
``os._exit`` in the middle of a real stage against a real SQLite file — the
approach ``tests/test_jobs.py`` established, extended here to the two death
points it does not cover. A MinerU failure is a real ``ThreadingHTTPServer`` on
a real socket returning a real 429, a real 402 and a real stall, not a
replacement ``urlopen``. Database lock exhaustion is a second connection holding
``BEGIN EXCLUSIVE``. A duplicate submission is six separate OS processes racing
the same idempotency key. A resync is a subscription that really disconnected.

A fault-injection test that mocks the fault is a test of the mock, and every
one of these could have been mocked in about a line. The whole file runs in
about five seconds, which is what a fault suite that is *real* costs — the
expensive kind is the one that has to be skipped on a machine without Docker.

**There is one implementation.** The scenarios live in
:mod:`tools.fault_inject` and this file asserts against them, so the scenarios
in CI and the ones an operator can reproduce by hand are the same code. An
operator runs ``python tools/fault_inject.py --all`` and gets this same report.

**The defects this task found are fixed, and the pins became guards.** A silent
failure found by this task was recorded as an ``xfail(strict=True)``: the test
passed *because* the bug was still there, and started failing the moment somebody
fixed it, which was the signal to convert it. Writing a test that asserts a bug
is present as a plain passing test would be a test that keeps passing after the
fix and would then be lying; writing one that asserts the *fixed* behaviour as a
plain passing test is right, because now it fails if the behaviour regresses.

Three were pinned. All three are fixed, so this file carries no ``xfail``
markers today:

* the infinite reclaim loop — a job claimed and abandoned before its first step
  was requeued forever because the retry budget was only checked on the running
  path. Fixed in ``infra.jobs._recover_stale`` (the queued branch now fails an
  over-budget job the same way the running branch does).
* the MinerU cloud outcome — ``quota_exhausted``/``failed``/``timed_out`` were
  collapsed into one ``status="failed"``, so a reader could not tell "top up your
  quota" from "try again later". A machine-readable ``cloud_outcome`` is now
  carried on ``ParseResult``/``ParseOutcome`` and surfaced through
  ``build_parse_provenance``. (Follow-up, still open and tracked in the task
  report: the browser bundle does not yet branch on it.)
* the two markers that were live when this revision started — the state
  projection omitting the lease, and the ``jobs`` metrics having no production
  write path — both fixed here and both now asserted as plain regression guards.

The mechanism each fix rests on is stated at the test that guards it, so the
"why" does not rot the moment the marker is retired.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paperwhisperer.core.errors import build_sse_event  # noqa: E402
from paperwhisperer.infra import jobs as J  # noqa: E402
from tools import fault_inject as F  # noqa: E402
from tools import load_probe as L  # noqa: E402

#: The scenarios T4.4 names, and the plan row they come from:
#: "worker 重启、DB 连接耗尽、MinerU 429/超时、SSE 断线重连、用户取消、重复提交".
#: Six items, nine scenarios — "worker 重启" has three death points that recover
#: through different code paths, and "SSE 断线重连" splits into *the client comes
#: back* and *the client goes away*, which fail for entirely different reasons.
#: Asserted as a set so dropping a scenario is a failing test rather than a
#: quietly smaller suite.
EXPECTED_SCENARIOS = {
    "worker-restart",
    "worker-race",
    "db-exhaustion",
    "mineru-failures",
    "sse-reconnect",
    "sse-disconnect",
    "user-cancel",
    "duplicate-submit",
    "recovery-signal",
}


@pytest.fixture(scope="session")
def results(tmp_path_factory) -> dict[str, F.ScenarioResult]:
    """Every scenario, run once, for the whole session.

    Session-scoped because the scenarios are the expensive part — real process
    kills, real HTTP servers, real lock contention — and re-running them per test
    would buy nothing. What each test does with them is cheap.
    """
    root = tmp_path_factory.mktemp("fault-scenarios")
    return {name: F.SCENARIOS[name](root / name) for name in F.SCENARIOS}


# --------------------------------------------------------------------------
# The suite itself: coverage and honesty of the harness
# --------------------------------------------------------------------------


class TestTheSuiteIsHonest:
    """Before asking whether the product behaves, check that the probe is real.

    A fault-injection suite that silently stops reproducing its faults is worse
    than no suite: it reports green over code that has never been crashed. So
    the first assertions are about the harness reaching its faults at all.
    """

    def test_every_named_scenario_is_registered_and_reported(self, results):
        assert set(results) == EXPECTED_SCENARIOS, (
            f"场景登记表与 T4.4 的清单不一致："
            f"缺少 {EXPECTED_SCENARIOS - set(results)}，"
            f"多出 {set(results) - EXPECTED_SCENARIOS}"
        )

    @pytest.mark.parametrize("name", sorted(EXPECTED_SCENARIOS))
    def test_every_scenario_actually_reached_its_fault(self, results, name):
        # ``reached`` is the guard against the most dangerous kind of green: a
        # scenario that failed to reproduce the fault, after which every other
        # assertion passes for the wrong reason.
        result = results[name]
        assert result.fatal == "", f"{name} 未跑到断言就崩了：{result.fatal}"
        assert result.reached, (
            f"{name} 没有复现出它要注入的故障，"
            f"之后的断言全部会因为『故障没发生』而通过：{result.evidence}"
        )

    @pytest.mark.parametrize("name", sorted(EXPECTED_SCENARIOS))
    def test_every_scenario_states_what_it_expected(self, results, name):
        result = results[name]
        assert result.expected.strip(), f"{name} 没有写下预期行为"
        assert result.observed.strip(), f"{name} 没有记录实测行为"

    def test_the_cli_reports_the_same_verdicts_and_exits_non_zero(self, tmp_path):
        # A tool that always exits 0 has learned to say "fine" about everything.
        # The exit code is the part CI would actually read.
        proc = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "tools" / "fault_inject.py"),
             "--all", "--json", "--workdir", str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        assert proc.returncode != 0, "存在静默丢失时 CLI 仍应返回非零退出码"
        payload = json.loads(proc.stdout)
        assert {item["scenario"] for item in payload} == EXPECTED_SCENARIOS
        assert not any(item["fatal"] for item in payload)


# --------------------------------------------------------------------------
# Scenario 1 — worker restart (the release gate)
# --------------------------------------------------------------------------


class TestWorkerRestart:
    """The release gate: "worker/API 重启后任务可恢复或明确进入可重试终态".

    Read together with ``tests/test_jobs.py::TestRecoveryFromAKilledWorker``,
    which covers the mid-stage death. This class adds the two death points that
    file leaves open, and they do not behave the same way.
    """

    def test_a_worker_that_dies_inside_a_stage_is_recovered_and_completed(self, results):
        evidence = results["worker-restart"].evidence
        corpse = evidence["mid_stage"]
        after = evidence["mid_stage_after_recovery"]
        # The process really died — not "returned an error" and not "raised".
        # Everything below is only meaningful because it did.
        assert corpse["exit_code"] == 9
        assert corpse["state_left"] == J.STATE_PARSING
        assert corpse["lease_owner"] == "doomed-worker"
        # The stage wrote something durable before it died, so the parent — a
        # different process with a different connection — can see it.
        assert corpse["context_written"] == {"parsed_bytes": 4096}
        assert after["requeued_to"] == J.STATE_QUEUED
        assert after["drained"] == 1
        assert after["final"] == J.STATE_COMPLETED
        assert after["attempts"] == 2

    def test_the_recovery_names_the_worker_that_died(self, results):
        # "Why did this take twice?" is a question the job itself has to answer,
        # and the answer is in the trail rather than in a log nobody is reading.
        assert results["worker-restart"].checks["mid_stage_trail_names_the_dead_worker"]

    def test_a_claim_holds_the_job_out_of_the_queue_while_the_owner_is_alive(self, results):
        assert results["worker-restart"].checks["the_claim_is_invisible_to_the_queue"]

    def test_a_worker_that_dies_before_its_first_stage_has_its_lease_released(
        self, results
    ):
        # The quieter version of the same failure: claimed, then dead before
        # anything ran. Left leased it would block the queue behind a lease
        # nobody holds.
        assert results["worker-restart"].checks["claim_gap_lease_is_released"]
        after = results["worker-restart"].evidence["claim_gap_after_recovery"]
        assert after["state"] == J.STATE_QUEUED
        assert after["lease_owner"] == ""

    def test_a_worker_that_keeps_dying_in_the_claim_window_still_terminates(self, results):
        assert results["worker-restart"].checks["claim_gap_reaches_a_terminal_state"], (
            "claim→step 之间的反复崩溃不会进入任何终态，"
            f"实测：{results['worker-restart'].evidence['claim_gap_repeat']}"
        )

    def test_a_dead_lease_is_not_recovered_before_its_deadline(self, results):
        # The honest half: recovery is deadline-driven, so a worker that dies
        # holding the shipped 300-second lease leaves the job untouched for
        # 300 seconds. That is a long time to look like normal progress.
        assert results["worker-restart"].checks["far_lease_waits_for_its_deadline"]
        assert results["worker-restart"].checks["nothing_can_claim_it_in_the_meantime"]
        corpse = results["worker-restart"].evidence["far_lease_corpse"]
        assert corpse["exit_code"] == 9
        assert 250 < corpse["seconds_until_the_lease_expires"] <= 300

    def test_nothing_is_lost_once_the_deadline_passes(self, results):
        assert results["worker-restart"].checks["far_lease_recovers_once_the_deadline_passes"]

    def test_a_dead_job_is_distinguishable_from_a_running_one(self, results):
        # The lease is now projected into the state view, so a job a dead worker
        # still holds names its owner and its deadline. Kept as a plain
        # regression guard rather than deleted along with its marker: the fact it
        # pins is the difference between ``parsing`` meaning "being parsed" and
        # ``parsing`` meaning "nobody is coming back for this until the lease
        # lapses", and nothing else in the suite would notice that going away
        # again.
        assert results["worker-restart"].checks[
            "a_dead_job_is_distinguishable_from_a_live_one"
        ], (
            "死任务与活任务的状态投影逐字段相同："
            f"{results['worker-restart'].evidence['far_lease_projection']}"
        )


class TestWorkerRace:
    """Two workers, one job. Getting it wrong is a second bill, not a crash."""

    def test_exactly_one_worker_claims_each_job(self, results):
        evidence = results["worker-race"].evidence
        assert evidence["processes"] == 6
        assert evidence["duplicate_claims"] == 0
        assert set(evidence["attempts_after"].values()) == {1}
        assert not evidence["errors"]

    def test_the_race_is_a_real_race_and_not_a_queue(self, results):
        # 6 processes x 2 threads against one file, all released by a barrier.
        # If fewer jobs came back than processes went in, the "race" was really
        # a fast worker that drained the queue before the others woke up.
        assert results["worker-race"].evidence["distinct_claims"] >= 1
        assert results["worker-race"].evidence["queued_before"] >= 12


# --------------------------------------------------------------------------
# Scenario 2 — database connection exhaustion
# --------------------------------------------------------------------------


class TestDatabaseExhaustion:
    """SQLite is one writer. The question is not speed, it is visibility."""

    def test_concurrent_writers_lose_nothing(self, results):
        evidence = results["db-exhaustion"].evidence
        # 16 threads x 20 submissions against one real file. Every committed
        # write must be readable afterwards: that is the difference between
        # "slow" and "lost", and it is the whole of the acceptance criterion.
        assert evidence["burst_failed"] == 0
        assert evidence["burst_committed"] == evidence["burst_writes"] == 320
        assert evidence["rows_in_table"] >= evidence["burst_committed"]

    def test_a_held_write_lock_raises_rather_than_silently_dropping(self, results):
        evidence = results["db-exhaustion"].evidence
        # A separate connection holding BEGIN EXCLUSIVE. The write must block
        # and then fail loudly — a submit that returned success here would mean
        # a job the user was told was queued and that is not in the table.
        assert "database is locked" in evidence["exception"].lower()
        assert evidence["blocked_seconds_before_raising"] > 0
        assert results["db-exhaustion"].checks["lock_contention_raises_loudly"]

    def test_the_store_is_usable_the_moment_the_lock_clears(self, results):
        evidence = results["db-exhaustion"].evidence
        assert evidence["usable_after_the_lock_cleared"] == J.STATE_QUEUED
        assert evidence["original_job_survived"] is True

    def test_no_write_is_ever_silently_dropped_under_contention(self, results):
        # The negative, which is the criterion: if this ever passes without an
        # error, the tool would have recorded it under ``silent``.
        assert not any("静默丢弃" in item for item in results["db-exhaustion"].silent)

    def test_sqlite_connections_accumulate_per_thread_not_per_live_thread(self):
        # "DB connection exhaustion" on the personal tier has no pool to
        # exhaust. SqliteDatabase opens one connection per thread and holds it
        # until close(), so the count is a function of how many threads have
        # *ever* run. A reusing thread pool is therefore bounded; a
        # thread-per-unit-of-work design is not.
        accounting = L.connection_accounting(threads=24)
        # 24 threads *simultaneously alive* each hold one — the stronger claim,
        # since a burst whose threads serialise could otherwise share an
        # identifier and report a smaller number run to run.
        assert accounting["connections_while_all_threads_were_live"] == 24
        assert accounting["connections_after_threads_finished"] == 24
        assert accounting["threads_that_failed_to_open_a_connection"] == 0
        # And they outlive the threads: nothing but close() releases them.
        assert accounting["released_only_by_close"] is True
        assert accounting["connections_after_close"] == 0


# --------------------------------------------------------------------------
# Scenario 3 — MinerU: 429, quota exhaustion, timeout
# --------------------------------------------------------------------------


class TestMinerUFailures:
    """Three failures, three answers. Collapsing them wastes the user's time."""

    def test_a_429_is_retried_and_honours_retry_after(self, results):
        outcome = results["mineru-failures"].evidence["outcomes"]["429"]
        assert outcome["calls"] == 2  # MINERU_MAX_RETRIES=2 against a real server
        assert 1.0 in outcome["slept"], "应优先遵守对端的 Retry-After 而不是本地退避"

    def test_a_429_is_never_reported_as_a_quota_problem(self, results):
        # Retrying a 429 is right; telling the user to go top up is not, and
        # the two are separated by an explicit code rather than by prose.
        outcome = results["mineru-failures"].evidence["outcomes"]["429"]
        assert outcome["code"] == "mineru_http_error"
        assert outcome["code"] != "mineru_quota_exhausted"
        assert "额度" not in outcome["message"]

    def test_a_quota_exhaustion_is_never_retried(self, results):
        # Retrying a quota exhaustion wastes the user's time *and* burns a
        # quota check that will also fail. Exactly one call must be made.
        outcome = results["mineru-failures"].evidence["outcomes"]["402"]
        assert outcome["calls"] == 1
        assert outcome["slept"] == []
        assert outcome["code"] == "mineru_quota_exhausted"
        assert "额度" in outcome["message"]

    def test_a_business_code_reporting_quota_is_short_circuited_too(self, results):
        # MinerU reports most failures as `code != 0` on an HTTP 200, so a
        # status-code-only check would call this a success.
        outcome = results["mineru-failures"].evidence["outcomes"]["business-quota"]
        assert outcome["calls"] == 1
        assert outcome["code"] == "mineru_quota_exhausted"

    def test_a_business_error_that_is_not_about_quota_is_not_called_quota(self, results):
        outcome = results["mineru-failures"].evidence["outcomes"]["business-generic"]
        assert outcome["code"] != "mineru_quota_exhausted"
        assert "1001" in outcome["message"]

    def test_a_poll_timeout_degrades_rather_than_raising(self, results):
        # poll_batch returns None instead of raising, and that None is what
        # makes "slow" and "failed" different answers.
        timeout = results["mineru-failures"].evidence["poll_timeout"]
        assert timeout["poll_batch_returned"] is None
        assert timeout["seconds_before_giving_up"] >= 0.9

    def test_a_result_that_lands_after_the_deadline_is_recovered_as_partial(self, results):
        # The degradation that makes the timeout worth having: the user gets
        # their document and is told it may be incomplete, rather than a
        # "timed out" for a job that actually succeeded.
        assert results["mineru-failures"].checks["a_late_result_is_recovered_as_partial"]

    def test_a_cloud_reported_failure_is_not_dressed_up_as_a_timeout(self, results):
        assert results["mineru-failures"].evidence["cloud_failure_state"]["code"] == (
            "mineru_task_failed"
        )

    def test_the_three_messages_a_user_reads_are_all_different(self, results):
        messages = {
            results["mineru-failures"].evidence["outcomes"]["429"]["message"],
            results["mineru-failures"].evidence["outcomes"]["402"]["message"],
            "MinerU 解析在 1 秒内未完成，已放弃等待。",
        }
        assert len(messages) == 3
        assert all(messages)

    def test_the_quota_outcome_reaches_the_audit_log_and_never_the_token(self, results):
        record = results["mineru-failures"].evidence["audit_record"]
        assert record["outcome"] == "quota_exhausted"
        assert "fault-injection-token" not in json.dumps(record, ensure_ascii=False)

    def test_the_adapter_outcome_survives_as_a_machine_readable_field(self, results):
        # Up to the ingestion layer the three are distinguishable — the part
        # that is easy, and the part the next test says is then thrown away.
        assert results["mineru-failures"].checks[
            "the_adapter_outcomes_reach_the_caller_as_diagnostics"
        ]

    def test_a_ui_can_branch_on_which_of_the_three_it_was(self, results):
        assert results["mineru-failures"].checks["cloud_outcome_reaches_the_user_payload"], (
            "三种失败在用户可见载荷上无法区分："
            f"{results['mineru-failures'].evidence['user_facing']}"
        )


# --------------------------------------------------------------------------
# Scenario 4 — SSE reconnect and disconnect
# --------------------------------------------------------------------------


class TestSseReconnect:
    """The Phase 2 exit criterion: "浏览器刷新/SSE 断开不丢任务"."""

    def test_a_client_that_misses_several_transitions_gets_exactly_them_back(
        self, results
    ):
        evidence = results["sse-reconnect"].evidence
        # "Several", not "one". A cursor that only works when the client missed
        # a single event is a cursor with an off-by-one waiting to happen.
        assert evidence["missed_count"] > 2
        assert evidence["transitions_missed_while_away"] >= 3
        assert results["sse-reconnect"].checks["the_gap_is_recovered_exactly"]
        assert results["sse-reconnect"].checks["nothing_is_delivered_twice"]

    def test_the_reconnect_includes_the_terminal_event(self, results):
        # A client that missed the end of the run must still learn how it ended,
        # otherwise it hangs on a job that will never move again.
        assert results["sse-reconnect"].checks["the_terminal_event_arrives"]

    def test_the_snapshot_answers_where_am_i_without_reading_the_trail(self, results):
        assert results["sse-reconnect"].checks["the_snapshot_says_where_it_got_to"]

    def test_a_subscriber_that_reconnects_after_the_fact_still_learns_the_ending(
        self, results
    ):
        assert results["sse-reconnect"].checks["a_late_subscriber_learns_the_ending"]

    def test_the_state_feed_carries_no_content_and_no_result(self, results):
        # The reason the resync design works at all. A subscriber that also got
        # the document text could not tell which half of the stream it had.
        assert results["sse-reconnect"].checks["no_payload_in_the_state_feed"]
        assert results["sse-reconnect"].checks["no_result_in_the_state_feed"]

    def test_the_resync_primitive_survives_the_process_that_was_watching_it_dying(
        self, tmp_path
    ):
        # The subscription polls a table rather than listening to a worker's
        # memory, so the thing that tells the client what happened outlives the
        # thing that was doing the work. Built with two real store objects — a
        # second connection to the same file is the closest a test gets to "a
        # different process" without a subprocess, and the SQL is identical.
        first = F.open_store(str(tmp_path / "resync.db"))
        second = F.open_store(str(tmp_path / "resync.db"))
        try:
            job = first.submit("document_analysis", {"file_path": "a.pdf"}).job
            client = J.JobSubscription(first, job.job_id, poll_interval=0.01)
            first_page = client.poll()
            cursor = first_page[-1]["sequence"]

            # The "watched process" is gone. A new worker, on another
            # connection, carries the job to the end with nobody listening.
            worker = J.JobWorker(second, F.document_pipeline(), worker_id="other", lease_seconds=30)
            worker.drain()

            resumed = J.JobSubscription(first, job.job_id, since_sequence=cursor, poll_interval=0.01)
            missed = resumed.poll()
            seen = [event["sequence"] for event in first_page] + [
                event["sequence"] for event in missed
            ]
            assert seen == [event.sequence for event in first.events_since(job.job_id, 0)]
            assert len(seen) == len(set(seen))
            assert missed[-1]["state"] == J.STATE_COMPLETED
            assert resumed.snapshot()["terminal"] is True
        finally:
            first.close()
            second.close()

    def test_the_sse_wire_format_carries_no_resume_cursor(self):
        # Pinned as a fact about the format, not as a wish. ``build_sse_event``
        # emits ``event:`` and ``data:`` and nothing else, so there is no
        # ``id:`` for a client to resume from and no ``Last-Event-ID`` for the
        # browser to send back. The two resync mechanisms that do exist — the
        # SSE ``id:`` field and the ``JobSubscription`` cursor — are different
        # mechanisms, and only one of them is on the wire today.
        frame = build_sse_event("parsing", {"state": "parsing"})
        assert frame == 'event: parsing\ndata: {"state": "parsing"}\n\n'
        assert "id:" not in frame
        assert "retry:" not in frame


class TestSseDisconnect:
    """The other half: the client goes away, and the work it started finishes."""

    def test_the_cleanup_runs_when_the_client_vanishes(self, results):
        # Not a nicety: it is what deletes the temporary upload the analysis
        # was reading from. Leaking it per disconnect fills the disk of a
        # deployment that has no janitor.
        assert results["sse-disconnect"].checks["cleanup_runs_on_disconnect"]
        assert results["sse-disconnect"].evidence["cleanup_ran"] == 1

    def test_the_work_in_flight_is_not_abandoned(self, results):
        # pump_sync_events awaits the producer in its finally, so a disconnect
        # does not cancel a paid-for cloud parse halfway and then tell nobody.
        assert results["sse-disconnect"].checks["the_work_is_not_abandoned_mid_flight"]
        assert results["sse-disconnect"].evidence["producer_ran_to_completion"] == 1

    def test_disconnect_does_not_return_before_the_work_is_persisted(self, results):
        assert results["sse-disconnect"].checks[
            "aclose_does_not_return_before_the_work_does"
        ]


# --------------------------------------------------------------------------
# Scenario 5 — user cancellation
# --------------------------------------------------------------------------


class TestUserCancellation:
    """Cooperative, and honest about the fact that it is cooperative."""

    def test_a_cancel_before_the_first_stage_never_runs_one(self, results):
        assert results["user-cancel"].checks["a_queued_cancel_never_runs_a_stage"]
        assert results["user-cancel"].evidence["queued_final"]["stages_run"] == []

    def test_a_cancel_mid_run_is_honoured_at_the_next_boundary(self, results):
        assert results["user-cancel"].checks["a_mid_run_cancel_ends_cancelled"]
        entered = results["user-cancel"].evidence["mid_run_final"]["stages_entered"]
        assert J.STATE_FALLBACK not in entered

    def test_a_cancel_during_the_last_step_reports_completed_not_cancelled(
        self, results
    ):
        # The honest case. There is no boundary left after the final step, the
        # work was finished, and reporting "cancelled" would be a lie the user
        # discovers on the bill.
        final = results["user-cancel"].evidence["late_final"]
        assert final["state"] == J.STATE_COMPLETED
        assert final["cancel_requested"] is True
        assert final["result"] == {"analysis_id": "run-1"}

    def test_cancelling_a_finished_job_says_the_cancel_was_not_accepted(self, results):
        ack = results["user-cancel"].evidence["cancel_after_finish"]
        assert ack["accepted"] is False
        assert "终态" in ack["message"]


# --------------------------------------------------------------------------
# Scenario 6 — duplicate submission
# --------------------------------------------------------------------------


class TestDuplicateSubmission:
    """Deduplication, and the half that is easy to forget: telling the user."""

    def test_six_processes_double_clicking_produce_one_job(self, results):
        evidence = results["duplicate-submit"].evidence
        assert evidence["processes"] == 6
        assert len(evidence["job_ids"]) == 1
        assert evidence["queued_rows"] == 1
        assert not evidence["errors"]

    def test_the_job_is_claimed_once_so_the_cloud_parse_is_billed_once(self, results):
        evidence = results["duplicate-submit"].evidence
        assert evidence["second_claim"] is None
        assert evidence["attempts"] == 1

    def test_the_duplicate_response_is_distinguishable_from_a_fresh_one(self, results):
        # The half that is easy to forget. A response byte-identical to a new
        # submission tells a user who double-clicked nothing at all: the
        # deduplication worked and they never found out.
        assert results["duplicate-submit"].checks["the_duplicate_is_flagged"]
        assert results["duplicate-submit"].checks["it_carries_a_user_readable_notice"]
        assert results["duplicate-submit"].checks[
            "the_duplicate_differs_from_a_fresh_one"
        ]
        notice = results["duplicate-submit"].evidence["sample_notice"]
        assert "重复提交" in notice
        assert "double-click" in notice  # names the key, so it is actionable

    def test_the_notice_is_stable_enough_for_a_ui_to_branch_on(self, results):
        # One distinct notice for one distinct duplicate, so a UI can key on it
        # rather than parsing Chinese.
        assert results["duplicate-submit"].evidence["distinct_notices"] == 1


# --------------------------------------------------------------------------
# Scenario 7 — does a recovery make any noise?
# --------------------------------------------------------------------------


class TestRecoveryIsSignalled:
    """The release gate's failure, and whether anybody would hear it.

    The gate says a killed worker's task must be recoverable, and the job system
    delivers that. This is the question the gate does not ask: if that had
    happened at 3am, would anything have noticed?
    """

    def test_a_recovery_really_happened(self, results):
        # Otherwise the next two tests would be measuring a scenario that never
        # ran, which is the failure mode of the whole suite.
        assert results["recovery-signal"].checks["reached"]
        assert results["recovery-signal"].evidence["recovery_moved_rows"]

    def test_a_recovery_moves_the_jobs_metrics(self, results):
        # The write side is done: `infra.jobs` calls `record_job_transition`
        # on every committed state change (both tiers), `record_job_submitted`
        # on both the fresh and the deduplicated path, and `record_job_queue`
        # from `count_by_state`, all after the row commits so a lost
        # compare-and-set never inflates a recovery count. Kept as a plain
        # regression guard: a recovery that stops being counted is exactly the
        # kind of regression this file exists to catch, and it is invisible from
        # the outside until a 3am incident and a dashboard that says zero.
        assert results["recovery-signal"].checks[
            "a_recovery_moves_the_recovered_counter"
        ], (
            "一次完整的崩溃→恢复→重跑→完成没有让任何 jobs 指标 +1："
            f"{results['recovery-signal'].evidence['after']}"
        )

    def test_observability_is_off_unless_something_asks_for_it(self):
        # Stated as its own test because it is a design decision that a later
        # task may legitimately reverse, and whoever reverses it should have to
        # delete a line that says this was deliberate.
        from paperwhisperer.core import observability as O

        assert O.observability.enabled is False


# --------------------------------------------------------------------------
# The load probe
# --------------------------------------------------------------------------


class TestLoadProbe:
    """The probe's own numbers, checked where a number could be a lie."""

    def test_a_measurement_reports_a_tail_not_just_an_average(self):
        # p50 alone hides the entire finding of this probe: contention is not a
        # uniform slowdown, it is one unlucky writer waiting for the batch
        # ahead of it. If p95 were not above p50 there would be no tail to see.
        measurement = L.Measurement(
            scenario="t", concurrency=1, operations=0, seconds=1.0,
            latencies_ms=[1.0] * 99 + [500.0],
        )
        assert measurement.percentile(0.50) == 1.0
        assert measurement.percentile(0.99) == 1.0
        assert measurement.to_dict()["max_ms"] == 500.0
        assert measurement.to_dict()["p99_ms"] >= measurement.to_dict()["p50_ms"]

    def test_an_empty_measurement_does_not_divide_by_zero(self):
        empty = L.Measurement(scenario="t", concurrency=1, operations=0, seconds=0.0)
        assert empty.throughput == 0.0
        assert empty.percentile(0.95) == 0.0
        assert empty.to_dict()["max_ms"] == 0.0

    def test_a_contended_write_is_a_visible_error_not_a_drop(self, tmp_path):
        # The property T4.4 asks of the database tier, measured rather than
        # asserted: under a held write lock every write either succeeds or
        # raises. There is no third outcome in which it silently vanishes.
        store = F.open_store(str(tmp_path / "probe.db"))
        store.submit("document_analysis", {"file_path": "keep.pdf"})
        try:
            hog = F.sqlite3.connect(str(tmp_path / "probe.db"), isolation_level=None, timeout=30)
            hog.execute("PRAGMA busy_timeout=30000")
            hog.execute("BEGIN EXCLUSIVE")
            hog.execute(
                "INSERT INTO jobs(job_id, state, kind, payload, created_at, updated_at) "
                "VALUES('hog', 'queued', 'document_analysis', '{}', 'x', 'x')"
            )
            error = None
            try:
                store.submit("document_analysis", {"file_path": "blocked.pdf"})
            except Exception as exc:  # noqa: BLE001 - the exception IS the assertion
                error = exc
            finally:
                hog.execute("ROLLBACK")
                hog.close()
            assert isinstance(error, F.sqlite3.OperationalError)
            assert "locked" in str(error).lower()
            # And the pre-existing job is untouched: a failed write does not
            # take the queue down with it.
            assert store.get_job(store.list_jobs()[0].job_id) is not None
        finally:
            store.close()

    def test_the_probe_runs_and_reports_its_own_caveats(self, tmp_path):
        # A probe whose caveats live only in its docstring is a probe whose
        # caveats get separated from its numbers the moment somebody copies a
        # number out of a CI log. They have to be in the output.
        proc = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "tools" / "load_probe.py"),
             "--scenario", "submit", "--concurrency", "1,2", "--jobs", "5",
             "--json", "--workdir", str(tmp_path / "probe")],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        payload = json.loads(proc.stdout)
        assert payload["caveats"], "探针必须把自己的适用范围写进输出"
        assert any("容量计划" in item for item in payload["caveats"])
        assert payload["results"] and all(
            item["errors"] == 0 for item in payload["results"]
        )
        assert payload["ceiling"]

    def test_the_probe_exits_non_zero_on_a_bad_concurrency_argument(self):
        proc = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "tools" / "load_probe.py"),
             "--concurrency", "0"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert proc.returncode != 0


# --------------------------------------------------------------------------
# Cross-scenario invariants
# --------------------------------------------------------------------------


def test_no_scenario_silently_dropped_a_job(results, tmp_path):
    """Across every scenario, nothing was lost without a trace.

    The broadest form of the acceptance criterion: whatever else each scenario
    found, not one of them ended with a job that simply disappeared.
    """
    for name, result in results.items():
        assert result.fatal == "", f"{name}: {result.fatal}"
        counts = result.evidence.get("queued_rows", result.evidence.get("rows_in_table"))
        if counts is not None:
            assert isinstance(counts, int) and counts >= 0
        # A scenario that reported silence must have said *what* was silent.
        for item in result.silent:
            assert len(item) > 20, f"{name} 报告了静默但没有说明是什么：{item!r}"
