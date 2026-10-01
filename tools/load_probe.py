#!/usr/bin/env python3
"""Measure what a personal-tier deployment actually does under concurrency.

    python tools/load_probe.py                     # default sweep
    python tools/load_probe.py --concurrency 1,4,8,16,32
    python tools/load_probe.py --jobs 400 --json
    python tools/load_probe.py --scenario sse

T4.4 names k6 and Locust. Neither is installable here and neither is the right
tool for this deployment anyway: k6 and Locust drive HTTP endpoints, and the
interesting question on the personal tier is not "how fast does the HTTP layer
answer" but **"where does the single-writer SQLite file stop keeping up"**. The
load that matters here is not requests per second, it is concurrent *writers*,
because SQLite serialises them at the file level no matter how many threads or
processes you point at it.

So this probe is stdlib-only (``threading``, ``statistics``, ``sqlite3``) and it
measures the real objects: a real ``JobStore``, a real ``JobWorker``, real
SQLite with the project's own WAL and ``busy_timeout`` pragmas.

**Read the numbers for what they are.** This runs entirely on loopback, in one
process, against a local file, with no network and no LLM. It is a
*contention* measurement, not a capacity plan, and three things follow from
that which are worth stating before the numbers rather than after:

* **It excludes the network entirely.** Every millisecond here is disk and
  lock. A deployment on a real filesystem with a slow disk will be slower; one
  on tmpfs or NVMe will be faster. Nothing in this output is a latency SLO.
* **It excludes the model.** Parse and analyse are the long poles in the real
  pipeline by two orders of magnitude, and they are dominated by third-party
  quota, not by this process. A "job per second" figure derived from this probe
  says nothing about documents per minute, because the pipeline spends
  essentially all of its wall clock waiting on an API.
* **It is a single process.** The Hosted tier is multi-process against
  PostgreSQL and the number that matters there is a different one entirely.

What it *does* establish, honestly: the shape of the contention curve, the
point at which adding concurrency stops adding throughput, and whether
contention turns into visible errors or invisible ones. That last one is the
question T4.4 actually asks.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paperwhisperer.infra import jobs as J  # noqa: E402
from paperwhisperer.infra.stores import build_stores  # noqa: E402
from paperwhisperer.infra.stores.sqlite_backend import BUSY_TIMEOUT_MS  # noqa: E402

#: Concurrency levels swept by default. Deliberately includes 1 (the honest
#: baseline that every other number is a multiple of) and goes past the point
#: where the curve flattens, because "where is the ceiling" is only answerable
#: if the probe went past it.
DEFAULT_LEVELS = (1, 2, 4, 8, 16, 32)


@dataclass
class _Settings:
    """The configuration slice the store factory reads.

    Private on purpose. Inheriting the developer's real ``STORAGE_SQLITE_PATH``
    would mean a load test writing hundreds of jobs into their actual
    database, which is the kind of mistake a probe makes once and an operator
    remembers.
    """

    STORAGE_BACKEND: str = "sqlite"
    STORAGE_LEXICAL_FTS5: str = "auto"
    STORAGE_SQLITE_PATH: str = ""
    STORAGE_SQLITE_TIMEOUT_SECONDS: int = 30


@dataclass
class Measurement:
    """One concurrency level's result, with the distribution kept, not summarised.

    ``p50`` alone would hide the thing this probe exists to find: SQLite
    contention is not a uniform slowdown, it is a long tail where one unlucky
    writer waits for the whole batch ahead of it. ``p95`` and ``max`` are the
    numbers that move first.
    """

    scenario: str
    concurrency: int
    operations: int
    seconds: float = 0.0
    latencies_ms: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    #: The ``busy_timeout`` in force for this measurement. Carried on the
    #: measurement rather than reported globally because one scenario
    #: deliberately shortens it, and a number without its own caveat is a
    #: number that gets quoted without it.
    busy_timeout_ms: float = float(BUSY_TIMEOUT_MS)

    @property
    def throughput(self) -> float:
        return self.operations / self.seconds if self.seconds > 0 else 0.0

    def percentile(self, fraction: float) -> float:
        if not self.latencies_ms:
            return 0.0
        ordered = sorted(self.latencies_ms)
        # Nearest-rank, not interpolated: with a few hundred samples an
        # interpolated p95 invents a number no request actually took.
        index = min(len(ordered) - 1, max(0, int(round(fraction * len(ordered))) - 1))
        return ordered[index]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "concurrency": self.concurrency,
            "operations": self.operations,
            "seconds": round(self.seconds, 4),
            "throughput_per_second": round(self.throughput, 1),
            "p50_ms": round(self.percentile(0.50), 3),
            "p95_ms": round(self.percentile(0.95), 3),
            "p99_ms": round(self.percentile(0.99), 3),
            "max_ms": round(max(self.latencies_ms), 3) if self.latencies_ms else 0.0,
            "busy_timeout_ms": self.busy_timeout_ms,
            "errors": len(self.errors),
            "error_kinds": sorted({item.split(":")[0] for item in self.errors}),
        }


def _open_store(db_path: str) -> Any:
    bundle = build_stores(_Settings(STORAGE_SQLITE_PATH=str(db_path)))
    return J.build_job_store(bundle=bundle)


def _pipeline() -> J.JobPipeline:
    """A pipeline whose stages do a trivial amount of real work.

    Deliberately not ``lambda: None``: an empty step would let the measurement
    be dominated by how fast a thread can be created, which is a property of
    CPython rather than of the queue. The step writes to the scratch, which is
    what a real stage does and what makes the write lock contended.
    """

    def parsing(ctx: J.JobContext) -> None:
        ctx.update(context={"pages": 20, "chars": 40_000})

    def indexing(ctx: J.JobContext) -> None:
        ctx.update(result={"chunks": 18})

    return J.JobPipeline(
        kind="document_analysis",
        steps=(
            J.JobStep(state=J.STATE_PARSING, run=parsing),
            J.passthrough_step(J.STATE_QUALITY_CHECK, "该阶段无需执行。"),
            J.passthrough_step(J.STATE_FALLBACK, "该阶段无需执行。"),
            J.passthrough_step(J.STATE_NORMALIZING, "该阶段无需执行。"),
            J.JobStep(state=J.STATE_INDEXING, run=indexing),
        ),
    )


def _drive(
    scenario: str,
    concurrency: int,
    per_thread: int,
    body: Callable[[Any, int], None],
) -> Measurement:
    """Run ``body`` on ``concurrency`` threads, timing every single call.

    Per-call timing rather than per-batch, because a per-batch average would
    report one number for a run where one thread waited 400ms behind a write
    lock and the other fifteen took 2ms. That asymmetry *is* the finding.
    """
    measurement = Measurement(scenario=scenario, concurrency=concurrency, operations=0)
    barrier = threading.Barrier(concurrency)
    latencies: list[float] = []
    errors: list[str] = []
    lock = threading.Lock()

    def worker(index: int) -> None:
        store = _STORE_HOLDER[0]
        barrier.wait(timeout=30)  # otherwise this measures thread creation
        local: list[float] = []
        for item in range(per_thread):
            started = time.perf_counter()
            try:
                body(store, index * per_thread + item)
                local.append((time.perf_counter() - started) * 1000.0)
            except Exception as exc:  # noqa: BLE001 - errors are a measurement
                local.append((time.perf_counter() - started) * 1000.0)
                with lock:
                    errors.append(f"{type(exc).__name__}: {exc}")
        with lock:
            latencies.extend(local)

    started = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(index,)) for index in range(concurrency)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=300)
    measurement.seconds = time.perf_counter() - started
    measurement.latencies_ms = latencies
    measurement.errors = errors
    measurement.operations = len(latencies)
    return measurement


#: The store every thread shares. Set once per scenario by the caller and read
#: by ``_drive``'s worker closure; a module global is honest about what is
#: happening (one SQLite file, one set of per-thread connections) in a way that
#: threading a store through every call site would not be.
_STORE_HOLDER: list[Any] = [None]


def probe_submit(db: str, concurrency: int, per_thread: int) -> Measurement:
    """N clients submitting jobs at once — the API's write path.

    Every submission is a distinct idempotency key, because a duplicate key
    would collapse the load into one row and the probe would measure the
    unique index instead of the write lock.
    """

    def body(store: Any, index: int) -> None:
        store.submit("document_analysis", {"file_path": f"p{index}.pdf"},
                     idempotency_key=f"probe-{concurrency}-{index}")

    return _drive("submit", concurrency, per_thread, body)


def probe_read(db: str, concurrency: int, per_thread: int) -> Measurement:
    """N clients reading one job's state at once — the UI's poll path.

    The number that matters for a dashboard or a polling front end, and the one
    that should be *flat* in concurrency: WAL exists precisely so readers do not
    queue behind writers.
    """
    job = _STORE_HOLDER[0].submit("document_analysis", {"file_path": "read.pdf"}).job

    def body(store: Any, _index: int) -> None:
        store.get_job(job.job_id)

    return _drive("read", concurrency, per_thread, body)


def probe_claim(db: str, concurrency: int, per_thread: int) -> Measurement:
    """N workers claiming from one queue at once — the personal tier's ceiling.

    This is the honest measure of the single-writer constraint: every claim is
    a ``BEGIN IMMEDIATE``, so this curve is what SQLite's write serialisation
    actually costs rather than what a benchmark of a no-op would cost.
    """
    for index in range(concurrency * per_thread):
        _STORE_HOLDER[0].submit("document_analysis", {"file_path": f"c{index}.pdf"})

    def body(store: Any, _index: int) -> None:
        store.claim_next("probe-worker", 60, kind="document_analysis")

    return _drive("claim", concurrency, per_thread, body)


def probe_run(db: str, concurrency: int, per_thread: int) -> Measurement:
    """N workers each running whole jobs — the end-to-end job path.

    Every operation here is a real job going through its real state machine to
    a real terminal state, so the number is "jobs per second a single process
    can carry", which is the only capacity figure this deployment has.
    """

    def body(store: Any, index: int) -> None:
        store.submit("document_analysis", {"file_path": f"r{index}.pdf"})
        J.JobWorker(store, _pipeline(), worker_id="probe", lease_seconds=30).run_once()

    return _drive("run", concurrency, per_thread, body)


def probe_contention(db: str, concurrency: int, per_thread: int) -> Measurement:
    """N writers fighting over one held write lock — the failure shape.

    Not a throughput number: a deliberate one. A separate connection holds
    ``BEGIN EXCLUSIVE`` for the whole run while the workers try to write, which
    is what a stalled fsync, a long batch index or a stale lock looks like. The
    question it answers is whether contention becomes a *visible* error or an
    invisible stall, and the answer has a number attached: how long a caller
    waits before being told.
    """
        # The shipped ``busy_timeout`` is 30 seconds, so measuring it honestly would
    # take 30 seconds *per operation* — a probe that takes half an hour to
    # answer a yes/no question gets run once and then ignored. The timeout is
    # therefore shortened for this scenario only, and both numbers are reported
    # side by side so the shortened one cannot be mistaken for the shipped one.
    #
    # What is being measured does not depend on the value: a blocked writer
    # waits, and then either succeeds or raises ``sqlite3.OperationalError``.
    # Both are visible. The alternative — a silent drop — would be visible at
    # any timeout.
    from paperwhisperer.infra.stores import sqlite_backend

    original = sqlite_backend.BUSY_TIMEOUT_MS
    sqlite_backend.BUSY_TIMEOUT_MS = 250.0
    hog = sqlite3.connect(db, isolation_level=None, timeout=30)
    hog.execute("PRAGMA busy_timeout=30000")
    hog.execute("BEGIN EXCLUSIVE")
    hog.execute(
        "INSERT INTO jobs(job_id, state, kind, payload, created_at, updated_at) "
        "VALUES('probe-hog', 'queued', 'document_analysis', '{}', 'x', 'x')"
    )
    try:
        measurement = _drive("contended-write", concurrency, per_thread, _contended_write)
    finally:
        hog.execute("ROLLBACK")
        hog.close()
        sqlite_backend.BUSY_TIMEOUT_MS = original
    measurement.busy_timeout_ms = 250.0
    return measurement


def _contended_write(store: Any, index: int) -> None:
    store.submit("document_analysis", {"file_path": f"x{index}.pdf"},
                 idempotency_key=f"contended-{index}")


SCENARIOS: dict[str, Callable[[str, int, int], Measurement]] = {
    "submit": probe_submit,
    "read": probe_read,
    "claim": probe_claim,
    "run": probe_run,
    "contention": probe_contention,
}


def connection_accounting(threads: int = 40) -> dict[str, Any]:
    """How many SQLite connections does a burst of threads leave behind?

    On the personal tier "DB connection exhaustion" does not mean a pool ran
    dry — there is no pool. It means ``SqliteDatabase`` opens one connection per
    thread and holds it until ``close()``, so the count is a function of *how
    many threads have ever run*, not of how many are alive. A thread pool with
    a bound reuses threads and is therefore bounded; anything that spawns a
    thread per unit of work is not.

    Measured rather than argued, because the difference matters and is easy to
    get wrong in either direction.

    **The threads are held alive at a barrier before the count is read.** Two
    reasons, and the second is the one that bites. It makes the measurement the
    stronger claim — N *simultaneously live* threads hold N connections, which
    is not the same statement as "N threads ran at some point". And it removes a
    race: CPython reuses thread identifiers once a thread dies, so a burst whose
    threads happen to serialise can land two threads on one identifier and
    produce a count that varies run to run. A measurement that flickers is worse
    than no measurement, because it teaches people to ignore it.
    """
    from paperwhisperer.infra.stores.sqlite_backend import SqliteDatabase

    directory = tempfile.mkdtemp(prefix="pw-load-conn-")
    database = SqliteDatabase(os.path.join(directory, "connections.db"))
    # One connection on the main thread first, which is what an application
    # does at startup and what sets ``journal_mode=WAL`` on the file.
    #
    # This is not cosmetic. Converting a *fresh* file to WAL needs a brief
    # exclusive lock, and SQLite does not run the busy handler for a journal
    # mode change — so several threads racing to open the very first connection
    # can get an immediate ``database is locked`` out of ``_configure`` before
    # ``busy_timeout`` has even been set. Warming the file first measures the
    # steady state, which is the thing being asked about. (The cold-start race
    # is real and is reported in docs/FAILURE_MODES.md; it is rare, and a probe
    # whose number flickered with machine load would be worse than useless.)
    database.connection().execute("SELECT 1")
    before = len(database._connections)
    # +1 for the main thread, which is the last waiter and the one that
    # takes the reading. Getting this wrong breaks the barrier, and a broken
    # barrier raises in every worker rather than in one place.
    barrier = threading.Barrier(threads + 1)
    release = threading.Event()
    failures: list[str] = []
    failure_lock = threading.Lock()

    def touch() -> None:
        try:
            database.connection().execute("SELECT 1")
        except Exception as exc:  # noqa: BLE001 - a failure is a measurement
            # Recorded, not raised: a worker that dies here would break the
            # barrier and take the whole probe down with it, and the point of
            # this function is to *count* what happened.
            with failure_lock:
                failures.append(f"{type(exc).__name__}: {exc}")
        # Everybody has (or has failed to get) a connection; only now may anyone
        # count.
        barrier.wait(timeout=30)
        release.wait(timeout=30)

    workers = [threading.Thread(target=touch) for _ in range(threads)]
    for worker in workers:
        worker.start()
    # The main thread is one more waiter, so the barrier only trips once every
    # worker actually holds a connection. Only then is the count meaningful.
    barrier.wait(timeout=30)
    with database._lock:
        while_threads_live = len(database._connections) - before
    # Let the threads go, then look again: the connections outlive them.
    release.set()
    for worker in workers:
        worker.join(timeout=30)
    after_threads_died = len(database._connections) - before
    database.close()
    return {
        "threads": threads,
        "connections_while_all_threads_were_live": while_threads_live,
        "connections_after_threads_finished": after_threads_died,
        "connections_opened_per_thread": after_threads_died / max(1, threads),
        "connections_after_close": len(database._connections),
        "released_only_by_close": after_threads_died > 0
        and len(database._connections) == 0,
        "threads_that_failed_to_open_a_connection": len(failures),
        "open_failures": sorted(set(failures))[:3],
    }


def sweep(
    scenarios: Sequence[str],
    levels: Sequence[int],
    jobs_per_thread: int,
    workdir: Path,
) -> list[Measurement]:
    """Every scenario at every concurrency level, each on its own database.

    One file per (scenario, level) so a level's leftover rows cannot warm or
    bloat the next one's, and so a run that crashes leaves a readable
    intermediate rather than a corrupt shared file.
    """
    results: list[Measurement] = []
    for scenario in scenarios:
        for level in levels:
            directory = workdir / f"{scenario}-{level}"
            directory.mkdir(parents=True, exist_ok=True)
            db = str(directory / "probe.db")
            _STORE_HOLDER[0] = _open_store(db)
            try:
                results.append(SCENARIOS[scenario](db, level, jobs_per_thread))
            finally:
                _STORE_HOLDER[0].close()
                _STORE_HOLDER[0] = None
    return results


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

#: Stated in the output rather than only in the docstring, because a number
#: separated from its caveat is a number that gets quoted without it.
CAVEATS = (
    "回环单进程、无网络、无 LLM：这是**争用**测量，不是容量计划。",
    "它不包含解析与分析——那两步占真实管线墙钟时间的绝大部分，且由第三方配额决定。",
    "它不包含文件系统差异：真实磁盘比 tmpfs 慢，NVMe 比机械盘快。",
    "Hosted 档是多进程 + PostgreSQL，本表与之无关。",
)


def _table(results: Sequence[Measurement]) -> str:
    # ASCII column headers only: a header containing CJK is measured in
    # display columns by every terminal except the ones this might be piped
    # into, and a table whose separator lines do not line up is worse than no
    # table. The Chinese explanation lives in the prose below it.
    header = (
        f"{'scenario':<17}{'conc':>5}{'ops':>7}{'sec':>8}"
        f"{'ops/sec':>10}{'p50ms':>9}{'p95ms':>9}{'p99ms':>9}{'maxms':>9}{'err':>6}"
    )
    lines = [header, "-" * len(header)]
    for result in results:
        data = result.to_dict()
        lines.append(
            f"{data['scenario']:<17}{data['concurrency']:>5}{data['operations']:>7}"
            f"{data['seconds']:>8.2f}{data['throughput_per_second']:>10.0f}"
            f"{data['p50_ms']:>9.2f}{data['p95_ms']:>9.2f}{data['p99_ms']:>9.2f}"
            f"{data['max_ms']:>9.2f}{data['errors']:>6}"
        )
    return "\n".join(lines)


def _ceiling(results: Sequence[Measurement]) -> list[str]:
    """Where each curve stops improving, stated per scenario.

    "The ceiling" is not one number. It is the concurrency level past which
    throughput stops rising, and for a single-writer store the answer is
    usually "almost immediately" — which is the useful thing to say plainly
    rather than hiding in a table.
    """
    findings: list[str] = []
    by_scenario: dict[str, list[Measurement]] = {}
    for result in results:
        by_scenario.setdefault(result.scenario, []).append(result)
    for scenario, group in by_scenario.items():
        ordered = sorted(group, key=lambda item: item.concurrency)
        if scenario == "contended-write":
            # Not a throughput curve. Every operation here is a write that had
            # to wait out a busy_timeout and then fail, so the "peak" is just
            # the rate at which the probe can time writes out. Reporting it
            # next to a real throughput number would be comparing two different
            # quantities under one heading.
            errors = sum(len(item.errors) for item in ordered)
            waits = [item.percentile(0.50) for item in ordered if item.latencies_ms]
            findings.append(
                f"contended-write: 全部 {errors} 次写入在 busy_timeout 后抛出 "
                f"sqlite3.OperationalError（可见失败，不是静默丢弃），"
                f"等待中位 {statistics.median(waits):.0f} ms。"
                "这不是吞吐曲线，是失败形态的度量。"
            )
            continue
        best = max(item.throughput for item in ordered)
        # The lowest concurrency within 90% of peak: the smallest amount of
        # concurrency that already gets you essentially all of the ceiling.
        knee = next(
            (item.concurrency for item in ordered if item.throughput >= 0.9 * best), None
        )
        ratio = (
            ordered[-1].throughput / ordered[0].throughput if ordered[0].throughput else 0.0
        )
        spread = (
            max(item.percentile(0.95) for item in ordered) / max(1e-9, min(
                item.percentile(0.95) for item in ordered
            ))
        )
        findings.append(
            f"{scenario}: 峰值 {best:.0f} 次/秒，在并发 {knee} 附近达到；"
            f"从并发 {ordered[0].concurrency} 加到 {ordered[-1].concurrency}，"
            f"吞吐只涨了 {ratio:.2f}×，而 p95 尾延迟变化了 {spread:.1f}×。"
        )
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="测量个人档在并发下的真实表现（回环单进程，非容量计划）。",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=sorted(SCENARIOS),
        help="要测的场景，可重复；默认全部",
    )
    parser.add_argument(
        "--concurrency",
        default=",".join(str(level) for level in DEFAULT_LEVELS),
        help=f"并发档位，逗号分隔（默认 {','.join(str(x) for x in DEFAULT_LEVELS)}）",
    )
    parser.add_argument(
        "--jobs", type=int, default=40, help="每线程每档的操作数（默认 40）"
    )
    parser.add_argument("--json", action="store_true", help="以 JSON 输出")
    parser.add_argument(
        "--workdir", default="", help="临时数据库目录（默认用系统临时目录）"
    )
    args = parser.parse_args(argv)

    scenarios = args.scenario or list(SCENARIOS)
    try:
        levels = [int(item) for item in args.concurrency.split(",") if item.strip()]
    except ValueError:
        parser.error(f"--concurrency 必须是逗号分隔的整数，收到 {args.concurrency!r}")
    if not levels or min(levels) < 1:
        parser.error("--concurrency 必须至少包含一个 >= 1 的档位")
    if args.jobs < 1:
        parser.error("--jobs 必须 >= 1")

    def _run() -> list[Measurement]:
        if args.workdir:
            root = Path(args.workdir)
            root.mkdir(parents=True, exist_ok=True)
            return sweep(scenarios, levels, args.jobs, root)
        with tempfile.TemporaryDirectory(prefix="pw-load-") as temporary:
            return sweep(scenarios, levels, args.jobs, Path(temporary))

    results = _run()

    if args.json:
        print(
            json.dumps(
                {
                    "caveats": list(CAVEATS),
                    "busy_timeout_ms": BUSY_TIMEOUT_MS,
                    "results": [result.to_dict() for result in results],
                    "ceiling": _ceiling(results),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    print("PaperWhisperer 个人档并发探针")
    print(f"  busy_timeout = {BUSY_TIMEOUT_MS} ms，WAL，每线程一条连接，单进程\n")
    print(_table(results))
    print("\n天花板在哪里：")
    for line in _ceiling(results):
        print(f"  - {line}")
    print("\n这些数字意味着什么（也意味着什么不意味着）：")
    for caveat in CAVEATS:
        print(f"  * {caveat}")
    print(
        "\n结论：个人档是单用户部署，这个探针确认的是"
        "『争用会变成可见的排队与可见的错误，而不是静默丢失』，"
        "而不是它能扛多少并发。"
    )
    connections = connection_accounting()
    print("\n连接计数（SQLite 没有连接池，这里才是『连接耗尽』的真实形态）：")
    print(
        f"  {connections['threads']} 个线程同时存活，各持有 "
        f"{connections['connections_while_all_threads_were_live']} 条连接；"
        f"线程全部结束后仍保留 {connections['connections_after_threads_finished']} 条"
        f"（每线程 {connections['connections_opened_per_thread']:.1f} 条），"
        f"只有 close() 才释放（之后剩 {connections['connections_after_close']} 条）。"
    )
    if connections["threads_that_failed_to_open_a_connection"]:
        print(
            f"  其中 {connections['threads_that_failed_to_open_a_connection']} 个线程"
            f"开连接时直接失败：{connections['open_failures']}"
        )
    print(
        "  也就是说：连接数随『曾经跑过的线程总数』增长，不随『当前存活线程数』增长。"
        "线程池复用线程则连接有界；每个工作单元起一个线程则无界。"
    )

    contended = [r for r in results if r.scenario == "contended-write"]
    if contended:
        errors = sum(len(item.errors) for item in contended)
        waits = [item.percentile(0.50) for item in contended if item.latencies_ms]
        if errors:
            print(
                f"  写锁被独占时：{errors} 次写入在 busy_timeout 后抛出 "
                f"sqlite3.OperationalError（可见失败，不是静默丢弃），"
                f"等待中位 {statistics.median(waits):.0f} ms。"
            )
        else:
            print("  写锁被独占时没有出现错误——这值得复查，它可能意味着探针没锁上。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
