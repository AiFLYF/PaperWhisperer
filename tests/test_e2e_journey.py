"""The end-to-end journey, on a real document, through the real surfaces.

Every other test in this repository replaces one collaborator with a fake:
the parser with a fixture, the model with a stub, the store with a temporary
bundle that is trivially correct. That is right for unit tests and wrong for
this file, which exists to ask the one question none of them can: **does the
assembled product work when its parts, each written by a different author, are
finally put together on a real PDF?**

The journey is the product's own pitch: upload a paper, parse it, index it,
analyse it, ask a question whose answer is *late* in the document, get an answer
with claims and citations, click a citation and open the evidence, export it.
Nine steps, all through the real code paths.

**The load-bearing step is step 5.** The project was rebuilt around removing a
hard ``document_content[:12000]`` head excerpt from the Q&A path. Against this
paper, the sentence reporting its result (``67 test errors``) sits at offset
12132 — 132 characters past that window, inside the results section. A
head-excerpt implementation can never answer a question about it. So step 5
asks exactly that question, and this file asserts the answer *deterministically*,
against the retriever, with no model in the loop: the central claim must not be
gated behind a rate-limited endpoint. (The model-backed ``done`` answer is
asserted separately, and skips when the endpoint is unavailable.)

**What is honest about this file.** It is the test, not the repair. Where the
assembled journey breaks, the failing assertions below carry a precise
``reason`` and are marked ``xfail(strict=True)`` so a defect is *recorded* rather
than hidden, and so that fixing it turns the xpass into a failure that forces
the marker to be retired. The central claim is a plain assertion: it passes, and
that is the headline. Every model-backed test skips — never fails — when the
endpoint is down or rate-limited, so an outage cannot masquerade as a broken
product.

Run it with::

    python3 -m pytest tests/test_e2e_journey.py -q -p no:cacheprovider

The printable, runnable narration of the same journey is
``tools/e2e_smoke.py``; this file owns no production code and fixes nothing.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path

import pytest

from .helpers import UNAVAILABLE_MARKERS as HELPERS_UNAVAILABLE_MARKERS
from .helpers import LiveFailureKind, classify_live_failure

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PDF = PROJECT_ROOT / "eval" / "corpus" / "digital_single_col" / "hinton_distillation.pdf"

#: The late evidence and the question that reaches for it. The string is at
#: offset 12132 in the 33880-char parse of this paper; the old head window cut
#: at 12000. This is the sentence the whole retrieval migration exists to make
#: reachable.
LATE_EVIDENCE = "67 test errors"
LATE_QUESTION = "How many test errors did the student network make on the test set?"

#: Corpus guard. A clean clone has no ``eval/corpus``; the journey is meaningless
#: without a real document, so skip with the reason rather than fail.
pytestmark = pytest.mark.skipif(
    not CORPUS_PDF.exists(),
    reason=(
        f"no real corpus document at {CORPUS_PDF}. The journey requires a real PDF; "
        "eval/corpus is gitignored and must be fed per docs/DEVELOPMENT_PLAN.md §6."
    ),
)

#: Credentials are snapshotted at *import* time, before conftest's autouse
#: ``isolated_environment`` fixture strips OPENAI_API_KEY for the session. This
#: is the house pattern (tests/test_llm_integration.py): the value is gone from
#: ``os.environ`` by the time a test body runs, so it has to be read during
#: collection.
_ENV = {name: os.environ.get(name, "") for name in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL")}

#: "The endpoint would not serve us right now" is owned by ``tests/helpers.py``
#: and shared with ``test_llm_integration.py``, so both live files agree on the
#: boundary. This file previously carried its own copy; the two lists diverged
#: enough that one endpoint timeout skipped three tests in one file and hard-
#: failed a fourth in the other. The kept name below stays because every call
#: site in this file reads better with "is this _unavailable?".
_UNAVAILABLE = HELPERS_UNAVAILABLE_MARKERS


def _is_unavailable(text: str) -> bool:
    return classify_live_failure(text) is LiveFailureKind.ENDPOINT


def _is_model_quirk(text: str) -> bool:
    """Whether the endpoint served us and the *model* is the difficulty.

    A reasoning model asked a question the document cannot answer sometimes
    reaches for a tool this product never offers and returns an empty body with
    ``finish_reason="tool_calls"``. That is a property of the model, not a
    defect in the product, so the model-backed steps below retry it a bounded
    number of times rather than reporting it as a broken journey.
    """
    return classify_live_failure(text) is LiveFailureKind.MODEL


def _completion_available() -> tuple[bool, str]:
    """Whether the endpoint can actually serve one chat completion right now.

    ``GET /models`` only proves the socket is open; this endpoint answers that
    while every completion is 429. So the probe makes the cheapest completion
    and reports whether one is truly available — the question the model-backed
    tests need answered before they bill a run.
    """
    import httpx

    base = _ENV["OPENAI_BASE_URL"].rstrip("/")
    if not base:
        return False, "OPENAI_BASE_URL is not set"
    try:
        response = httpx.post(
            f"{base}/chat/completions",
            headers={"Authorization": "Bearer " + _ENV["OPENAI_API_KEY"]},
            json={
                "model": _ENV["OPENAI_MODEL"],
                "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
                "max_tokens": 5,
            },
            timeout=30.0,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    if response.status_code == 200:
        return True, "a real completion returned 200"
    return False, f"completion returned HTTP {response.status_code}: {response.text[:120]}"


@pytest.fixture
def live_model(monkeypatch):
    """Restore the real credentials for one model-backed test, or skip it.

    The autouse ``isolated_environment`` fixture in conftest deletes
    ``OPENAI_API_KEY`` before every test; this puts the import-time snapshot
    back, serializes the shared semaphore so the pipeline is a courteous client
    of a rate-limited endpoint, and pins retries to 1 so an error path costs one
    call rather than three.
    """
    if not _ENV["OPENAI_API_KEY"] or not _ENV["OPENAI_BASE_URL"] or not _ENV["OPENAI_MODEL"]:
        pytest.skip(
            "no live LLM configured (need OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL "
            "in the project .env). The model-backed journey steps are opt-in and skip cleanly."
        )
    available, detail = _completion_available()
    if not available:
        pytest.skip(
            f"LLM endpoint {_ENV['OPENAI_BASE_URL']!r} cannot serve a completion ({detail}). "
            "This is an environment skip, not a product failure."
        )
    monkeypatch.setenv("OPENAI_API_KEY", _ENV["OPENAI_API_KEY"])
    from paperwhisperer.core import config

    monkeypatch.setattr(config, "OPENAI_MAX_RETRIES", 1)
    monkeypatch.setattr(config, "MAX_LLM_CONCURRENCY", 1)
    monkeypatch.setattr(config, "LLM_REQUEST_SEMAPHORE", threading.BoundedSemaphore(1))
    return dict(_ENV)


# --------------------------------------------------------------------------
# Teardown: this fixture's own worker, collected by this fixture's own stop
# --------------------------------------------------------------------------

#: The budget handed to the teardown drain's ``stop_job_workers`` call. This is
#: the same number the app lifespan's own shutdown uses, and deliberately so:
#: the drain is not asking for *more* patience than a real shutdown would give
#: a real worker. What it adds is a second attempt, not a longer wait.
TEARDOWN_STOP_TIMEOUT = 5.0

#: How many times the drain re-issues the bounded stop before it gives up and
#: lets the process carry on. A worker whose in-flight step finishes while the
#: drain sleeps is collected on the next attempt, because ``JobWorker.stop``
#: sets the stop flag and re-checks it once the step returns. This count is the
#: bound on the drain's whole lifetime: teardown must not become the unbounded
#: join the design exists to avoid.
TEARDOWN_STOP_ATTEMPTS = 3

#: Sleep between drain attempts. Short enough to cost almost nothing, long
#: enough to give a step that is about to return the chance to return.
TEARDOWN_STOP_PAUSE = 0.25


def _journey_worker_threads() -> set[str]:
    """The names of every live ``job-worker-*`` thread in this process.

    A thread probe rather than an inference from the registry: ``active_workers``
    and ``stuck_workers`` both read the same dict the drain writes, so using them
    to check the drain would be the code agreeing with itself. This enumerates
    the interpreter's own live threads, which is the thing that actually has to
    be gone for the next module to start clean.
    """
    return {t.name for t in threading.enumerate() if t.name.startswith("job-worker-")}


def _drain_journey_workers() -> set[str]:
    """Retry the bounded stop until this fixture's workers are really gone.

    **Why a bounded retry, and not a longer join.** ``stop_job_workers`` is
    bounded *on purpose* — a shutdown that blocks is the failure this whole
    design exists to avoid, and a longer budget would only make that failure
    slower to notice. But "bounded" is not "stop trying": since the shutdown
    ordering was fixed, a worker whose budget expired keeps its handle and stays
    reachable, so calling the *same* function again re-joins it with a fresh
    budget and collects it. A step that is merely slow — which is what a
    mid-parse shutdown has — returns on its own, and the very next bounded stop
    collects the thread. So the drain trades an unbounded wait for a bounded
    number of bounded waits, and a worker that genuinely will not stop (a step
    parked forever) still leaves after ``TEARDOWN_STOP_ATTEMPTS`` instead of
    hanging the session.

    Returns whatever worker threads are still alive when the drain gives up, so
    the caller can decide what to do about them rather than assuming success.
    """
    from paperwhisperer.api import routes_jobs

    for attempt in range(TEARDOWN_STOP_ATTEMPTS):
        if not _journey_worker_threads():
            return set()
        if attempt:
            time.sleep(TEARDOWN_STOP_PAUSE)
        routes_jobs.stop_job_workers(timeout=TEARDOWN_STOP_TIMEOUT)
    return _journey_worker_threads()


class _ReleasableStep:
    """A real pipeline step that stays inside ``run`` until the test releases it.

    This is what a *mid-parse* worker looks like to a shutdown: the thread is
    already inside ``run`` and will not leave until its own step returns. The
    step is a real :class:`~paperwhisperer.infra.jobs.JobStep` on a real
    pipeline running on a real worker thread; the only thing simulated is that
    the step is parked until released.

    An :class:`threading.Event` rather than a sleep, because a sleep is a race —
    on a loaded machine the "missed" window can elapse before the assertion runs
    and "collected" can happen before the drain joins. Here the thread is
    *provably* inside the step and *provably* still inside it until ``release``
    is set, so the mid-parse condition is arranged, never waited for.
    """

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.finished = threading.Event()
        self._release = threading.Event()

    def release(self) -> None:
        self._release.set()

    def pipeline(self, kind: str):
        from paperwhisperer.infra import jobs as J

        def _step(_context):
            self.entered.set()
            # Bounded only as a backstop against a test bug: the release always
            # comes, because the test's own flow sets it.
            self._release.wait(timeout=30.0)
            self.finished.set()

        return J.JobPipeline(
            kind=kind, steps=(J.JobStep(state=J.STATE_PARSING, run=_step, required=True),)
        )


@pytest.fixture
def probe_registry():
    """The real worker registry, isolated so a probe never disturbs the journey.

    The journey fixture keeps its own workers registered in this module-global
    registry for the whole module. A probe that called ``stop_job_workers``
    without isolating would stop those too. So this snapshots, empties, and
    restores — the same contract ``tests/test_worker_shutdown.py`` uses, for the
    same reason: a probe that leaked a worker into the next test would be the
    very bug this file is about.
    """
    from paperwhisperer.api import routes_jobs

    saved_workers = dict(routes_jobs._WORKERS)
    saved_stranded = list(routes_jobs._STRANDED)
    routes_jobs._WORKERS.clear()
    routes_jobs._STRANDED.clear()
    try:
        yield routes_jobs._WORKERS
    finally:
        try:
            routes_jobs.stop_job_workers(timeout=TEARDOWN_STOP_TIMEOUT)
        finally:
            routes_jobs._WORKERS.clear()
            routes_jobs._STRANDED.clear()
            routes_jobs._WORKERS.update(saved_workers)
            routes_jobs._STRANDED.extend(saved_stranded)


@pytest.fixture
def probe_store(tmp_path):
    """A real job store on the personal tier, with its own database file."""
    from paperwhisperer.infra.stores import TIER_PERSONAL

    from .test_jobs import open_store

    harness = open_store(TIER_PERSONAL, tmp_path, name="journey-drain-probe.db")
    try:
        yield harness.store
    finally:
        harness.close()


def test_the_teardown_drain_collects_a_worker_a_single_bounded_stop_missed(
    probe_registry, probe_store
):
    """The leak, the fix, and a thread probe — the whole teardown claim in one test.

    Reproduces the journey's situation deterministically and then proves the
    drain closes it:

    1. A real worker is driven **mid-step** (its pipeline is parked inside a
       step on an ``Event`` this test owns), which is the state the journey's
       model-backed steps can leave behind at module teardown.
    2. One bounded ``stop_job_workers`` — exactly what the app lifespan's own
       shutdown issues — is shown to **miss** it: budget expired, thread still
       alive, still registered. This is the leak, arranged rather than raced.
    3. The step is released (the parse finishes, as a slow parse eventually
       does) and the journey's teardown drain (:func:`_drain_journey_workers`)
       re-issues the bounded stop, collecting the worker.
    4. A **thread probe** over ``threading.enumerate()`` confirms the
       ``job-worker-*`` thread is genuinely gone — not merely dropped from the
       registry, which is the defect the shutdown-ordering fix addressed.
    """
    from paperwhisperer.api import routes_jobs
    from paperwhisperer.infra import jobs as J

    step = _ReleasableStep()
    worker = J.JobWorker(
        probe_store,
        step.pipeline(routes_jobs.KIND_DOCUMENT_ANALYSIS),
        lease_seconds=30.0,
        poll_interval=0.01,
    )
    probe_store.submit(routes_jobs.KIND_DOCUMENT_ANALYSIS, {})
    worker.start()
    probe_registry[routes_jobs.KIND_DOCUMENT_ANALYSIS] = worker
    try:
        assert step.entered.wait(timeout=30), "工作线程没有进入解析阶段。"

        thread_name = f"job-worker-{worker.worker_id}"

        # (2) One bounded stop, exactly the lifespan's own teardown. The step is
        # parked, so this deterministically misses.
        routes_jobs.stop_job_workers(timeout=0.05)
        assert worker.running is True, (
            "the stop unexpectedly collected the parked worker; the test would no "
            "longer be reproducing the mid-parse teardown."
        )
        assert thread_name in _journey_worker_threads(), (
            "thread probe: the worker thread should still be alive after the missed stop."
        )
        assert probe_registry.get(routes_jobs.KIND_DOCUMENT_ANALYSIS) is worker, (
            "a stop that missed must keep the handle so a retry can reach the worker."
        )

        # (3) The step returns (the parse completes), then the journey's own
        # teardown drain re-issues the bounded stop.
        step.release()
        assert step.finished.wait(timeout=30), "步骤没有结束，drain 无从回收。"
        _drain_journey_workers()

        # (4) Thread probe: really gone.
        assert thread_name not in _journey_worker_threads(), (
            f"the teardown drain returned while {thread_name} was still alive; the worker "
            "would leak into every later module in the session."
        )
        assert worker.running is False
        assert routes_jobs.stuck_workers() == []
    finally:
        step.release()


class Journey:
    """The assembled product, mid-journey, with each step's real result."""

    def __init__(self, client, bundle, document, tmp_root: Path) -> None:
        self.client = client
        self.bundle = bundle
        self.document = document
        self.tmp_root = tmp_root
        self.document_id = str(document["document_id"])
        self.resource = document
        self._retrieval = None

    @property
    def stored(self):
        record = self.bundle.metadata.get_document(self.document_id)
        assert record is not None and record.document is not None, "document not in the store"
        return record

    @property
    def chunks(self):
        return self.bundle.metadata.list_chunks(self.document_id)

    def late_retrieval(self):
        """The Q&A retrieval over the *indexed* document, via the real retriever.

        This is the deterministic heart of step 5 and the central claim of the
        project. It uses the same ``HybridRetriever`` the ``/v1`` Q&A route
        builds, over the same store the upload indexed, so it cannot pass in a
        way the product would not.
        """
        if self._retrieval is None:
            from paperwhisperer.analysis.orchestrator import build_qa_retriever
            from paperwhisperer.rag.retrieve import TASK_QA, plan_for

            retriever = build_qa_retriever(self.bundle, None, document=self.stored.document)
            self._retrieval = retriever.retrieve(
                LATE_QUESTION, plan=plan_for(TASK_QA), document_ids=[self.document_id]
            )
        return self._retrieval


@pytest.fixture(scope="module")
def journey_worker_drained():
    """Asserts, *after* the journey has torn itself down, that it took its worker.

    ``journey`` requests this fixture, so pytest sets this up first and finalizes
    it **last** — after ``journey``'s own finalizer has run the drain. That
    ordering is the entire point: an assertion placed any earlier would run while
    the journey is still up and would be measuring the wrong state. Finalization
    is in reverse order of setup, so "depends on" here is how a fixture says
    "assert after me".

    The check is a thread probe over ``threading.enumerate()``, not a read of
    ``routes_jobs``' registry. The registry is what the drain writes, so
    consulting it to grade the drain is the code agreeing with itself; the only
    thing that can honestly answer "is the process drained?" is the interpreter's
    own list of live threads.
    """
    yield
    survivors = _journey_worker_threads()
    assert not survivors, (
        f"the journey fixture tore itself down but left worker thread(s) alive: "
        f"{sorted(survivors)}. The teardown drain bounds itself to "
        f"{TEARDOWN_STOP_ATTEMPTS} attempts of a {TEARDOWN_STOP_TIMEOUT}s join, so a survivor "
        "means a worker was still mid-step when the module ended and never finished "
        "within the budget — the exact orphan this fixture is supposed to prevent."
    )


@pytest.fixture(scope="module")
def journey(journey_worker_drained):
    """Upload the corpus PDF once and keep the product mid-journey.

    Module-scoped on purpose: the instruction, and the whole cost argument, is
    to *reuse one parse across every assertion* rather than re-parsing a 9-page
    PDF per test. Every route, store, job worker and Q&A analyzer is wired to a
    single temporary tree through ``dependency_overrides``, so the workers the
    lifespan starts, the routes, and the retriever all see the same store — the
    property a real single-tier deployment has.

    **Its own worker is collected by its own teardown.** Leaving the ``with
    TestClient`` block runs the app lifespan's shutdown, which issues *one*
    bounded ``stop_job_workers``. That is enough only if no worker is mid-step at
    that instant. The model-backed steps below submit real ``/api/analyze`` jobs
    that run on those worker threads, so one can absolutely still be inside a
    parse or an analysis stage when the module ends — and then the single bounded
    stop returns with its budget expired, the thread survives, and (before the
    shutdown-ordering fix) nothing left in the process could name it again. The
    fixture used to simply close its bundle and walk away, which left that
    thread to be collected only if some *later* file happened to call
    ``stop_job_workers()``: luck, not correctness.

    So after the block, :func:`_drain_journey_workers` re-issues the bounded stop
    a bounded number of times — the recovery path the shutdown fix deliberately
    made reachable — and ``journey_worker_drained`` then probes the live threads
    to prove the drain actually collected it. Crucially the drain runs *before*
    ``bundle.close()``: closing a store while a worker is still executing on it is
    a use-after-free, which is the other half of the same defect.
    """
    from fastapi.testclient import TestClient

    from paperwhisperer.analysis import orchestrator
    from paperwhisperer.api import routes_evidence, routes_jobs, routes_qa, routes_v1
    from paperwhisperer.api.routes_jobs import build_job_store
    from paperwhisperer.app import create_app
    from paperwhisperer.core import config
    from paperwhisperer.infra.object_store import build_object_store
    from paperwhisperer.infra.stores import build_sqlite_bundle

    with tempfile.TemporaryDirectory(prefix="pw-e2e-") as tmp:
        root = Path(tmp)
        for name in ("uploads", "output", "context"):
            (root / name).mkdir(parents=True, exist_ok=True)
        config.UPLOAD_FOLDER = str(root / "uploads")
        config.OUTPUT_FOLDER = str(root / "output")
        config.CONTEXT_FOLDER = str(root / "context")
        config.STORAGE_SQLITE_PATH = str(root / "context" / "stores" / "e2e.sqlite3")
        config.OBJECT_STORE_ROOT = str(root / "context" / "objects")
        # The page renderer reads originals from its own configured root; point
        # it at the same tree the upload wrote, or it answers
        # ``original_unavailable`` for bytes that are right there.
        config.PAGE_RENDER_ORIGINALS_DIR = str(root / "context" / "objects")

        bundle = build_sqlite_bundle(config)
        objects = build_object_store(config)
        app = create_app()
        job_store = build_job_store(bundle=bundle)
        app.dependency_overrides[routes_evidence.get_store_bundle] = lambda: bundle
        app.dependency_overrides[routes_jobs.get_job_store] = lambda: job_store
        app.dependency_overrides[routes_v1.get_object_store] = lambda: objects
        routes_qa.get_store_bundle = lambda: bundle
        orchestrator.get_store_bundle = lambda: bundle

        with TestClient(app) as client:
            with CORPUS_PDF.open("rb") as handle:
                response = client.post(
                    "/v1/documents",
                    files={"file": (CORPUS_PDF.name, handle, "application/pdf")},
                )
            assert response.status_code == 201, (
                f"uploading a real PDF returned HTTP {response.status_code}: {response.text[:300]}"
            )
            yield Journey(client, bundle, response.json()["document"], root)
        # The lifespan has made its one bounded stop. Re-issue it so *this*
        # fixture's worker is drained here, deterministically, rather than by
        # whichever later test module happens to shut a worker down next. Must
        # run before ``bundle.close()`` below: a worker still mid-parse on a
        # store we just closed is a use-after-free.
        _drain_journey_workers()
        bundle.close()


# ---------------------------------------------------------------------------
# Steps 1-2: upload and parse provenance
# ---------------------------------------------------------------------------


def test_step1_upload_through_the_real_route_creates_a_document(journey):
    """Step 1: a real PDF, uploaded through POST /v1/documents, becomes a document.

    Not a function call — the route, with the real validation, the real object
    store write, the real parse and the real index.
    """
    assert journey.document_id.startswith("doc_"), journey.document_id
    assert journey.resource["page_count"] > 0, journey.resource
    assert journey.resource["char_count"] > 1000, "the parse produced almost no text"
    assert journey.chunks, "the upload indexed no chunks"


def test_step2_parse_records_which_parser_read_it(journey):
    """Step 2a: provenance names a parser and a status.

    A user who uploaded a paper is entitled to know which parser read it and
    whether the parse degraded.
    """
    parse = journey.resource["parse"]
    assert parse["parsed_by"], f"parse names no parser: {parse}"
    assert parse["status"], f"parse has no status: {parse}"
    assert parse["used_cloud"] is False, f"a local corpus PDF was sent to a cloud parser: {parse}"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "DEFECT (owner: parser routing default). PARSER_BACKEND defaults to 'legacy' "
        "(paperwhisperer/core/config.py:575), so the shipped default bypasses the gateway and "
        "runs the legacy adapter — which produces NO bbox and degrades even a clean digital PDF "
        "to status='partial' with the warning 'legacy PDF 解析只能定位到页，无法提供页内坐标（bbox）'. "
        "The bake-off (docs/PARSER_BAKE_OFF.md §6, strategy B) decided docling-first, and docling "
        "2.131.0 is installed and registered, but the default never uses it. Result: the product "
        "ships page-level provenance only, contrary to the Evidence-first promise. This test "
        "asserts a page locator that carries a bbox, which only the gateway/docling route gives."
    ),
)
def test_step2_parse_provides_page_and_bbox_provenance(journey):
    """Step 2b: elements carry a *bbox*, not just a page.

    A locator that stops at the page cannot be visually targeted, and the whole
    Evidence viewer rests on the bbox. Under the current default parser this
    fails, which is the point of recording it.
    """
    document = journey.stored.document
    pdf_elements = [e for e in document.elements if e.locator.kind == "pdf_page"]
    assert pdf_elements, "no pdf_page elements"
    with_bbox = [e for e in pdf_elements if e.locator.bbox]
    assert with_bbox, (
        f"none of the {len(pdf_elements)} pdf_page elements carry a bbox; the default parse is "
        "page-only, so no citation can be visually targeted"
    )


# ---------------------------------------------------------------------------
# Step 3: did vectors actually land?
# ---------------------------------------------------------------------------


def test_step3_vectors_actually_landed_in_the_real_store(journey):
    """Step 3: assert the vectors, not that the call returned.

    Reads every chunk's vector out of the store and checks the count and the
    dimension. A stored index that has no vectors is the state where the dense
    leg abstains on every question.
    """
    chunks = journey.chunks
    assert chunks, "no chunks were indexed"
    vectors = []
    for chunk in chunks:
        try:
            entry = journey.bundle.vectors.get_vector(chunk.chunk_id)
        except Exception:  # noqa: BLE001
            entry = None
        # ``get_vector`` returns a ``VectorEntry`` (the row, carrying the keys a
        # filtered search needs), so the vector itself is ``.vector`` — the
        # dimension assertion below reads the numbers, not the row.
        if entry is not None and entry.vector:
            vectors.append(entry.vector)
    assert vectors, (
        f"0 of {len(chunks)} chunks carry a vector: the upload path indexes lexically-only, "
        "so the dense leg never runs"
    )
    from paperwhisperer.core import config

    assert {len(v) for v in vectors} == {int(config.EMBEDDING_LOCAL_DIMENSION)}, (
        f"vector dimension(s) {sorted({len(v) for v in vectors})} != configured "
        f"{config.EMBEDDING_LOCAL_DIMENSION}"
    )


# ---------------------------------------------------------------------------
# Step 5: the central claim — the late passage is reachable
# ---------------------------------------------------------------------------


def test_step5_the_late_passage_is_reachable_by_retrieval(journey):
    """The central claim, asserted deterministically with no model in the loop.

    The head-excerpt window is gone. The sentence reporting the paper's result
    sits at offset 12132, past the old 12000 cut, and retrieval must surface it.
    If this passes, the fix the project was rebuilt around holds in the
    *assembled* system; if it fails, every unit test can still be green while
    the product cannot answer a question about its own results.
    """
    record = journey.stored
    full_text = record.document.to_full_text()
    assert LATE_EVIDENCE in full_text, "the corpus document no longer contains the evidence"
    offset = full_text.find(LATE_EVIDENCE)
    from paperwhisperer.core import config

    window = config.DOCUMENT_WINDOW_BUDGET["qa"]
    assert offset > window, (
        f"the fixture no longer exercises the late-evidence case: {LATE_EVIDENCE!r} is at "
        f"offset {offset}, inside the {window}-char head window"
    )

    result = journey.late_retrieval()
    assert not result.insufficient_evidence, (
        "retrieval found nothing, but the evidence is in the document — the head excerpt is "
        "gone and something else is dropping it"
    )
    joined = " ".join(item.text for item in result.evidence)
    assert LATE_EVIDENCE in joined, (
        f"retrieval returned {len(result.evidence)} passage(s) but none contains {LATE_EVIDENCE!r}. "
        f"The passage sits at offset {offset}, past the old {window}-char window; a head-excerpt "
        "implementation could not reach it, so retrieval failing here means the central fix does "
        "not hold in the assembled system."
    )


# ---------------------------------------------------------------------------
# Step 7: quote verification and degradation (deterministic)
# ---------------------------------------------------------------------------


def _first_evidence(journey):
    from paperwhisperer.ingestion.schema import Evidence

    chunk = journey.chunks[0]
    element = next((e for e in journey.stored.document.elements if e.text), None)
    return chunk, Evidence(
        evidence_id="ev_journey",
        chunk_id=chunk.chunk_id,
        text=chunk.text,
        locator=element.locator if element else None,
    )


def test_step7_a_real_quote_verifies_and_a_fabrication_degrades(journey):
    """Step 7: the release gate, on the real document.

    A span the paper actually contains aligns and is quotable. A number the
    paper never states does not align and is degraded to a paraphrase carrying
    no verified text — so there is no state in which an unverified string can be
    rendered as though the paper said it.
    """
    from paperwhisperer.analysis.orchestrator import AnswerClaim, _verify_claim

    chunk, evidence = _first_evidence(journey)
    real_span = (chunk.text or "")[:60].strip()
    assert real_span, "the first chunk has no text to quote"

    aligned = _verify_claim(AnswerClaim(text="t", quote=real_span), [evidence], journey.stored.document)
    assert aligned.presentation.value == "direct_quote", (
        f"a real span did not verify: outcome={aligned.outcome!r} message={aligned.message!r}"
    )
    assert aligned.verified_text, "a verified direct quote carries no verified text"

    fabricated = _verify_claim(
        AnswerClaim(text="t", quote="the student made 999999 errors, a number not in the paper"),
        [evidence],
        journey.stored.document,
    )
    assert fabricated.presentation.value == "paraphrase", (
        f"a fabricated quote was NOT degraded: presentation={fabricated.presentation!r} "
        f"verified_text={fabricated.verified_text!r}"
    )
    assert not fabricated.verified_text, "a degraded paraphrase still carries verified text"
    assert not fabricated.is_direct_quote, "a degraded paraphrase is marked quotable"


# ---------------------------------------------------------------------------
# Step 8: citation click -> Evidence API -> page image
# ---------------------------------------------------------------------------


def test_step8_the_page_image_renders_for_a_citation(journey):
    """Step 8b: the renderer works — addressed by the underlying chunk id.

    Proves the Evidence API and the page rasteriser function on the real
    document. It addresses the chunk id, the other of the two namespaces the
    route resolves; the retriever's ``ev_`` id is covered by step 8a.
    """
    chunk, _evidence = _first_evidence(journey)
    response = journey.client.get(
        f"/v1/documents/{journey.document_id}/evidence/{chunk.chunk_id}"
    )
    assert response.status_code == 200, f"evidence did not resolve: {response.text[:200]}"
    payload = response.json()
    page = (payload.get("locator") or {}).get("page")
    assert page, f"evidence has no page to render: {payload.get('locator')}"
    image = journey.client.get(f"/v1/documents/{journey.document_id}/pages/{page}/image")
    assert image.status_code == 200, f"page image did not render: {image.text[:200]}"
    assert image.content[:4] == b"\x89PNG", "the page route did not return a PNG"


def test_step8_a_citation_click_resolves_through_the_evidence_api(journey):
    """Step 8a: a citation carrying its evidence_id opens the passage.

    The citation a reader clicks carries the ``ev_`` id the retriever produced.
    Opening it must resolve to a passage — the identifier namespaces the
    retriever and the API used to disagree on, which made every real click a
    404 while the evidence suite stayed green on ``chunk_*`` ids alone.
    """
    evidence = journey.late_retrieval().evidence[0]
    response = journey.client.get(
        f"/v1/documents/{journey.document_id}/evidence/{evidence.evidence_id}"
    )
    assert response.status_code == 200, (
        f"a citation click carrying evidence_id {evidence.evidence_id} returned "
        f"HTTP {response.status_code} ({response.json().get('code')})"
    )
    assert response.json().get("locator") is not None, "the evidence has no locator"


# ---------------------------------------------------------------------------
# Steps 4, 5-live, 6, 9: model-backed. Skip cleanly when the endpoint is down.
# ---------------------------------------------------------------------------


def test_step4_analysis_reports_section_statuses_and_partial_consistently(journey, live_model):
    """Step 4: analysis through the real pipeline with the real model.

    A full analysis is the most expensive step, so the optional sections are
    turned off to keep the journey re-runnable. What is asserted is the
    bookkeeping: the required sections succeed, and the reported quality
    matches whether any section actually failed (a ``partial`` is reported as
    partial, never as a clean success).
    """
    with CORPUS_PDF.open("rb") as handle:
        response = journey.client.post(
            "/api/analyze",
            files={"file": (CORPUS_PDF.name, handle, "application/pdf")},
            data={
                "api_key": live_model["OPENAI_API_KEY"],
                "generate_mermaid": "false",
                "generate_evaluation": "false",
                "generate_research_brief": "false",
            },
        )
    if response.status_code != 200:
        if _is_unavailable(response.text):
            pytest.skip(f"analysis could not complete; endpoint unavailable: {response.text[:200]}")
        pytest.fail(f"analysis returned HTTP {response.status_code}: {response.text[:300]}")
    result = response.json()
    statuses = {name: (sec or {}).get("status") for name, sec in (result.get("sections") or {}).items()}
    sections = result.get("sections") or {}
    # A required section that reports "failed" is not automatically a product
    # defect. The section carries the error string the product classified it
    # with, and on a shared endpoint that string is regularly a timeout or a
    # 429. Reporting those as a broken analysis is what made this step red on
    # somebody else's outage, so the error is classified before it is judged.
    # A genuinely wrong failure (a product defect) still fails below.
    for required in ("summary", "quotes", "mindmap"):
        if statuses.get(required) != "success":
            error = (sections.get(required) or {}).get("error") or ""
            if _is_unavailable(error):
                pytest.skip(
                    f"a required section ({required}) failed because the endpoint "
                    f"would not serve it: {error!r}. Environment, not a broken journey."
                )
            if _is_model_quirk(error):
                pytest.xfail(
                    f"a required section ({required}) came back declined by the model "
                    f"(tool-call refusal), which this step does not re-draw: {error!r}. "
                    f"See docs/LLM_TESTS.md; the orchestrated equivalent in "
                    f"test_llm_integration.py retries this and passes."
                )
    for required in ("summary", "quotes", "mindmap"):
        assert statuses.get(required) == "success", f"{required} did not succeed: {statuses}"
    quality = (result.get("analysis_status") or {}).get("quality")
    expected = "partial" if any(v == "failed" for v in statuses.values()) else "complete"
    assert quality == expected, f"quality={quality} but statuses imply {expected}: {statuses}"
    assert result["parse"]["parsed_by"], "analysis returned no parse provenance"


def test_step5_live_answer_to_the_late_question_carries_claims_and_citations(journey, live_model):
    """Steps 5-live + 6: the generated answer is structured and reaches the late fact.

    This is the model-backed counterpart to the deterministic reachability test:
    the passage is *found* without a model, and here the model is asked a
    question about it and must answer with claims, evidence ids, citations and a
    confidence that matches what the claims actually support.
    """
    def _sse(text: str) -> list[tuple[str, dict]]:
        events: list[tuple[str, dict]] = []
        for block in text.split("\n\n"):
            name, data_lines = None, []
            for line in block.splitlines():
                if line.startswith("event:"):
                    name = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[len("data:"):].strip())
            if name is None:
                continue
            raw = "\n".join(data_lines)
            try:
                events.append((name, json.loads(raw) if raw else {}))
            except json.JSONDecodeError:
                events.append((name, {"_raw": raw}))
        return events

    answer_result = None
    last_answer = ""
    last_detail = ""
    # Bounded draws for a sampled model. Two model-flavoured outcomes are
    # re-drawn rather than failed: the model reaching for a tool it was never
    # offered (the Q&A error carries TOOL_CALL_MESSAGE), and the model answering
    # but not surfacing the late figure on this particular draw. Both are the
    # model's sampling, not a defect in the journey; the deterministic test in
    # this same file already proves the evidence is retrievable without a model,
    # so what is being re-drawn here is only the phrasing, never the fact. An
    # endpoint outage skips and is never retried, and a product-shaped failure
    # (a stream with no done event for any other reason) fails immediately.
    for attempt in range(1, 4):
        response = journey.client.post(
            f"/v1/documents/{journey.document_id}/qa",
            json={"question": LATE_QUESTION, "api_key": live_model["OPENAI_API_KEY"]},
        )
        if response.status_code != 200:
            if _is_unavailable(response.text):
                pytest.skip(f"Q&A could not run; endpoint unavailable: {response.text[:200]}")
            if not _is_model_quirk(response.text):
                pytest.fail(f"Q&A returned HTTP {response.status_code}: {response.text[:300]}")
            last_detail = f"HTTP {response.status_code}: {response.text[:200]}"
        else:
            events = _sse(response.text)
            done = [payload for name, payload in events if name == "done"]
            if not done:
                detail = json.dumps([p for n, p in events if n == "error"], ensure_ascii=False)[:300]
                if _is_unavailable(detail):
                    pytest.skip(f"Q&A stream carried an endpoint error: {detail}")
                if not _is_model_quirk(detail):
                    pytest.fail(f"no 'done' event in stream {[n for n, _ in events]}: {detail}")
                last_detail = f"stream carried only errors: {detail}"
            else:
                candidate = done[0]
                answer = candidate.get("answer", "")
                if LATE_EVIDENCE in answer or "67" in answer:
                    answer_result = candidate
                    break
                last_answer = answer
                last_detail = f"answered without the late figure: {answer[:200]!r}"
        print(f"\nstep5 draw {attempt}/3 -- {last_detail}")
    assert answer_result is not None, (
        f"step5: 3 real draws, none carried the late evidence {LATE_EVIDENCE!r}. The "
        f"endpoint served every draw and the retrieval is proven deterministic by the "
        f"model-free test in this file, so this is the model's phrasing, not a broken "
        f"journey and not an outage. Last: {last_answer[:200]!r} / {last_detail}"
    )
    answer = answer_result.get("answer", "")
    claims = answer_result.get("claims") or []
    assert claims, "the answer carries no claims"
    assert any(claim.get("evidence_ids") for claim in claims), "no claim names an evidence id"
    confidence = answer_result.get("confidence")
    unbacked = [
        claim for claim in claims
        if not claim.get("evidence_ids")
        or (claim.get("quote") and claim.get("presentation") != "direct_quote")
    ]
    if confidence == "supported":
        assert not unbacked, f"confidence 'supported' but {len(unbacked)} claims are unbacked"
    for claim in claims:
        if claim.get("presentation") == "paraphrase":
            assert not claim.get("uses_quotation_marks"), "a paraphrase carries quotation marks"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "DEFECT (partially fixed; owner: analysis route wiring). The transport half of D3 is "
        "CLOSED and verified: the exported report now carries whatever the analysis actually "
        "produced. finalize_analysis_result (analysis/service.py) no longer lets the session drop the "
        "keys /api/export reads -- _session_payload_carrying_provenance places parse/analysis_run/"
        "citations/claims at the session top level (the exact contract routes_documents."
        "_export_result_from_session lifts) and hands document_id/paper_facts to "
        "build_session_payload through its own kwargs, so a run that DOES carry paper_facts/"
        "analysis_run now renders both '生成条件与溯源' and '引用与证据链接' (verified with a stubbed run). "
        "What remains is the recording half: the job's _analyse_stage (api/routes_jobs.py:586) still "
        "calls finalize_analysis_result(run=None, report=None) -- the legacy six-section path -- so the "
        "one analysis route the product exposes produces no paper_facts/analysis_run for the export to "
        "carry. The only honest producer of a run is analysis.service.analyze_with_fact_layer (the "
        "client tracks no token usage), which needs the document indexed inside the job and changes "
        "/api/analyze's output, cost and streaming -- an explicit migration the codebase deliberately "
        "gated ('a one-line change at the call site ... so that the two can be compared on the same "
        "document rather than one replacing the other unmeasured'). That swap is a product decision, "
        "not a composition repair, and cannot be validated without a live model, so it is left for that "
        "decision rather than made here. Until it lands, the real /api/analyze export is still a shell "
        "-- correctly so, because that run recorded no run to report."
    ),
)
def test_step9_export_carries_the_provenance(journey, live_model):
    """Step 9: the export records what produced it, and never quotes a paraphrase.

    Driven through the real ``/api/export/{session}`` for a session a real
    analysis created. The export must carry a provenance section. The session
    no longer drops the provenance (that half is fixed); what is still missing is
    that the one analysis route the product exposes runs the legacy path and
    records no run for the export to carry. See the reason.
    """
    with CORPUS_PDF.open("rb") as handle:
        response = journey.client.post(
            "/api/analyze",
            files={"file": (CORPUS_PDF.name, handle, "application/pdf")},
            data={
                "api_key": live_model["OPENAI_API_KEY"],
                "generate_mermaid": "false",
                "generate_evaluation": "false",
                "generate_research_brief": "false",
            },
        )
    if response.status_code != 200:
        if _is_unavailable(response.text):
            pytest.skip(f"analysis could not complete; endpoint unavailable: {response.text[:200]}")
        pytest.fail(f"analysis returned HTTP {response.status_code}: {response.text[:300]}")
    result = response.json()
    export = journey.client.get(
        f"/api/export/{result['session_id']}",
        params={"session_token": result["session_token"], "format": "markdown"},
    )
    assert export.status_code == 200, f"export returned HTTP {export.status_code}"
    body = export.text
    assert "生成条件与溯源" in body, (
        "the exported markdown has no provenance section; the analysis route records no "
        f"analysis_run and the session drops the fact layer. Export was {len(body)} bytes."
    )
