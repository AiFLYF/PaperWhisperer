"""The submit/query surface, asserted through the real application.

T2.3 is "长任务出 API | 解析/索引/分析全部经 job 系统；API 只提交与查询", and its
subject is a claim about *where work happens*, not about a response shape. A
behaviour test that only checks a 202 would pass against a route that submitted
a job and then did the work anyway, so this file is built around the question
"did the work happen in the request?" and the evidence is always one of three
things: a stub that records when it was called, a job row whose state says where
the work had got to, or a process that genuinely died.

The tests are grouped by what they hold the design to:

*Submission.* An async submit returns in milliseconds with a job id, and the stub
analyzer has not been touched when the response is written. The same submission
drained returns the analysis, because T2.4's criterion is that the existing front
end switches over without breaking.
*The state machine.* A job walks the real spine to a terminal state, and a
partial parse ends the job ``partial`` rather than failing it.
*Durability.* A worker is killed inside the analysing stage in a **real
subprocess** — the same technique ``tests/test_jobs.py`` uses, pointed at the
API's pipeline rather than a synthetic one — and this process recovers the job
and finishes it.
*Resync.* A client that disconnects mid-stream comes back with
``Last-Event-ID`` and is told exactly what it missed.
*Concurrency and cancellation.* A duplicated submission is one job, and a cancel
is accepted and observed.

No LLM, no network, no MinerU. The parser, the analyzer and the finaliser are
stubs reached through the *modules* the stages import them from, which is what
makes these tests about the pipeline's plumbing rather than about any model's
opinions. The store is a real SQLite file on ``tmp_path`` and the object store a
real directory tree, because "the job table lives in the same database as the
documents" and "the uploaded bytes outlive the request" are claims about files.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from paperwhisperer.analysis import orchestrator as analysis_orchestrator
from paperwhisperer.analysis import service as analysis_service
from paperwhisperer.api import routes_evidence, routes_jobs, routes_v1
from paperwhisperer.app import app
from paperwhisperer.core import config
from paperwhisperer.infra import jobs as J
from paperwhisperer.infra.object_store import FilesystemObjectStore
from paperwhisperer.infra.stores import build_sqlite_bundle
from paperwhisperer.ingestion import gateway
from paperwhisperer.ingestion import orchestrator as ingest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: How far into the future to look when deciding a lease has expired. The
#: recovery tests ask the question with a clock rather than sleeping out a real
#: lease: a test that sleeps to prove a deadline passed is slow *and* flaky.
FUTURE = 3600.0

#: A credential the process does not hold, so only the pattern pass of
#: ``redact_secrets`` can catch it. This is the shape of a key a user pasted
#: into a document, or a presigned signature minted by a third party.
SENTINEL = "sk-T2.3-SENTINEL-KEY-MUST-NOT-LEAK"


# --------------------------------------------------------------------------
# Stubs
# --------------------------------------------------------------------------


@dataclass
class ParseCall:
    """One recorded parse, so a test can say *when* it happened."""

    file_path: str
    body: str = ""


class StubParseOutcome:
    """The bits of ``ParseOutcome`` the parse stage reads.

    A stub rather than a real ``ParseOutcome`` so a test can say "this document
    parses partially" without arranging for a parser that genuinely degrades,
    which would be a second thing under test in a test about the job system.
    """

    def __init__(self, *, warnings: list[str] | None = None, usable: bool = True) -> None:
        self.status = "partial" if warnings else "success"
        self.warnings = list(warnings or [])
        self.error_message = "" if usable else "模拟的解析失败。"
        self.used_cloud = False
        self.touched_network = False
        self.cache_hit = False
        self.parsed_by = "stub"
        self.document = SimpleNamespace(
            document_id="doc_stub",
            page_count=2,
            char_count=1200,
            source_filename="paper.txt",
        ) if usable else None

    def summary_line(self) -> str:
        return "由 stub 解析。"

    @property
    def is_usable(self) -> bool:
        return self.document is not None


class StubParser:
    """A parser the gateway accepts, which records being asked to run."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def supports(self, media_type: str) -> bool:
        return True

    def parse(self, file_path, options=None):
        self.calls.append(str(file_path))
        raise AssertionError("the job's parse stage should have been stubbed out")


class StubAnalyzer:
    """A ``DocumentAnalyzer`` that answers without a model.

    Recording *when* it was constructed is the point of this file: the claim
    under test is that the analysis did not happen inside the request, and the
    only honest evidence for that is a stub that would have noticed.
    """

    instances: list[StubAnalyzer] = []
    calls: list[str] = []

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.version = "stub-1.0"
        StubAnalyzer.instances.append(self)

    def analyze_stream(self, file_path, *args, **kwargs):
        StubAnalyzer.calls.append(str(file_path))
        yield {"type": "section", "name": "summary", "section": {"summary": "要点"}}
        yield {"type": "section", "name": "evaluation", "section": {"evaluation": "评价"}}
        yield {"type": "done", "result": {"sections": {"summary": {}, "evaluation": {}}}}


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def bundle(tmp_path):
    settings = SimpleNamespace(
        STORAGE_SQLITE_PATH=str(tmp_path / "jobs_api.sqlite3"),
        STORAGE_SQLITE_TIMEOUT_SECONDS=30,
        STORAGE_LEXICAL_FTS5="auto",
    )
    built = build_sqlite_bundle(settings)
    yield built
    built.close()


@pytest.fixture
def objects(tmp_path):
    store = FilesystemObjectStore(tmp_path / "objects", fsync=False)
    store.initialize()
    yield store
    store.close()


@pytest.fixture
def job_store(bundle):
    return J.build_job_store(bundle=bundle)


@pytest.fixture
def stubbed_pipeline(monkeypatch, tmp_path):
    """Stub the parse, the analyzer and the finaliser; isolate the runtime dirs.

    Each stage imports its collaborator *inside* the function, from the module
    it names, so replacing the module attribute is enough and is also the most
    honest seam available: it proves the stage is calling the function it claims
    to rather than a local copy of it.

    The runtime folders go to ``tmp_path`` because the stages materialise a real
    file out of the object store, and a test must never write to a developer's
    uploads directory.
    """
    monkeypatch.setattr(config, "UPLOAD_FOLDER", str(tmp_path / "uploads"))
    monkeypatch.setattr(config, "OUTPUT_FOLDER", str(tmp_path / "output"))
    monkeypatch.setattr(config, "CONTEXT_FOLDER", str(tmp_path / "context"))
    for name in ("uploads", "output", "context"):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "PARSER_CACHE_ENABLED", False)
    monkeypatch.setattr(config, "PARSER_BACKEND", "gateway")

    calls = ParseCall(file_path="", body="")

    def fake_parse(file_path, **kwargs):
        calls.file_path = str(file_path)
        calls.body = Path(str(file_path)).read_text(encoding="utf-8", errors="ignore")
        return StubParseOutcome()

    monkeypatch.setattr(ingest, "parse_document", fake_parse)
    monkeypatch.setattr(gateway, "build_default_parsers", lambda: [StubParser()])
    monkeypatch.setattr(routes_v1, "build_default_parsers", lambda: [StubParser()])

    StubAnalyzer.instances = []
    StubAnalyzer.calls = []
    monkeypatch.setattr(analysis_orchestrator, "DocumentAnalyzer", StubAnalyzer)

    def fake_finalize(*, result, analyzer, original_filename, **kwargs):
        # A constant, not the session id it was handed: ``build_session_id("")``
        # mints a real one for an anonymous upload, and echoing it would make
        # this assertion pass without proving the stub ran.
        return {
            "session_id": "sess_stub",
            "session_token": "token-not-a-real-secret",
            "source_filename": original_filename,
            "sections": result.get("sections") or {},
            "citations": [],
            "claims": [],
        }

    monkeypatch.setattr(analysis_service, "finalize_analysis_result", fake_finalize)
    return calls


@pytest.fixture
def api(bundle, objects, job_store, stubbed_pipeline, monkeypatch):
    """The real application with only the three storage seams replaced.

    Deliberately **not** entering the lifespan, so no background worker is
    running and a job submitted in one test is still ``queued`` when the response
    is written. That is what makes "the work did not happen in the request"
    checkable at all: with a live worker, a fast stub would have finished during
    the request and the test would be asserting nothing. The worker-driven
    behaviour is covered separately, by a client that *does* run the lifespan.
    """
    overrides = {
        routes_evidence.get_store_bundle: lambda: bundle,
        routes_v1.get_object_store: lambda: objects,
        routes_jobs.get_object_store: lambda: objects,
        routes_jobs.get_job_store: lambda: job_store,
    }
    monkeypatch.setattr(routes_jobs, "resolve_object_store", lambda: objects)
    app.dependency_overrides.update(overrides)
    try:
        yield TestClient(app)
    finally:
        for dependency in overrides:
            app.dependency_overrides.pop(dependency, None)


@pytest.fixture
def live_api(bundle, objects, job_store, stubbed_pipeline, monkeypatch):
    """The real application *with* its lifespan, so a worker is genuinely running.

    This is the fixture that tests the Phase 2 criterion itself: a job submitted
    to it is picked up by a thread the application started, not by the request.
    """
    monkeypatch.setattr(config, "JOBS_POLL_INTERVAL_SECONDS", 1)
    overrides = {
        routes_evidence.get_store_bundle: lambda: bundle,
        routes_v1.get_object_store: lambda: objects,
        routes_jobs.get_object_store: lambda: objects,
        routes_jobs.get_job_store: lambda: job_store,
    }
    monkeypatch.setattr(routes_jobs, "resolve_object_store", lambda: objects)
    app.dependency_overrides.update(overrides)
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        routes_jobs.stop_job_workers()
        for dependency in overrides:
            app.dependency_overrides.pop(dependency, None)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def upload(client, name: str = "paper.txt", body: bytes = b"a document about testing", **form):
    """POST a multipart upload the way a browser would."""
    return client.post(
        "/api/jobs/analyze",
        files={"file": (name, io.BytesIO(body), "text/plain")},
        data=form,
    )


def upload_analyze(client, name: str = "paper.txt", body: bytes = b"a document about testing",
                   **form):
    """POST to the legacy synchronous route, which the bundled browser calls."""
    return client.post("/api/analyze", files={"file": (name, io.BytesIO(body), "text/plain")},
                       data=form)


def sse_events(body: str) -> list[tuple[str, str, dict]]:
    """Decode an SSE body into ``[(event_id, event_name, payload), ...]``.

    The id is returned separately because it is the contract: it is what a
    reconnecting client sends back as ``Last-Event-ID``, and a test that read
    the sequence out of the JSON payload would pass even if the ``id:`` line
    had been dropped from the wire.
    """
    events: list[tuple[str, str, dict]] = []
    for block in body.split("\n\n"):
        if not block.strip():
            continue
        event_id = name = data = ""
        for line in block.splitlines():
            if line.startswith("id:"):
                event_id = line[len("id:"):].strip()
            elif line.startswith("event:"):
                name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data = line[len("data:"):].strip()
        if name:
            events.append((event_id, name, json.loads(data) if data else {}))
    return events


def drain(job_store, job_id: str, timeout: float = 30.0) -> J.Job:
    """Run the real pipeline synchronously and return the settled job.

    The same ``JobWorker.drain`` the lifespan's thread calls, called from here
    instead — so a test that drains is exercising the production path rather than
    a stand-in for it.
    """
    worker = J.JobWorker(job_store, routes_jobs.analysis_pipeline(), lease_seconds=30)
    worker.drain(timeout=timeout)
    return job_store.get_job(job_id)


# --------------------------------------------------------------------------
# Submit: a job id, and no work
# --------------------------------------------------------------------------


class TestSubmission:
    def test_an_async_submit_returns_a_job_id_and_answers_immediately(self, api, job_store):
        response = upload(api, api_key="sk-live-0000000000000000")

        assert response.status_code == 202, response.text
        body = response.json()
        # The id is the whole product of this route. A client that can only get
        # this far still has something it can poll, stream and cancel.
        assert body["job_id"]
        assert body["status_path"] == f"/v1/jobs/{body['job_id']}"
        assert body["stream_path"] == f"/v1/jobs/{body['job_id']}/stream"
        assert body["kind"] == routes_jobs.KIND_DOCUMENT_ANALYSIS

    def test_the_work_does_not_happen_in_the_request(self, api, job_store):
        """The load-bearing assertion of T2.3, and it is about *timing*.

        If the route parsed or analysed inside the request, the stub would have
        been called by the time the response body was written. It has not, and
        the job is still ``queued`` — which is the same fact said twice, once
        through the collaborator and once through the state machine, because
        either one alone could be a lie.
        """
        response = upload(api, api_key="sk-live-0000000000000000")
        assert response.status_code == 202

        assert StubAnalyzer.instances == [], "analyzer was constructed inside the request"
        assert StubAnalyzer.calls == [], "analysis ran inside the request"
        job = job_store.get_job(response.json()["job_id"])
        assert job.state == J.STATE_QUEUED

    def test_the_uploaded_bytes_outlive_the_request(self, api, objects, job_store):
        """The file has to still be there when the job runs, an hour later.

        The submit route deletes its staging upload in a ``finally``, exactly as
        it always did. The assertion is therefore not "the file is still on
        disk" but "the bytes are in the object store's ``original`` layer", which
        is the copy the job reads and the only one that survives.
        """
        body = b"the exact bytes of the upload, which must survive the request"
        response = upload(api, body=body)
        job = job_store.get_job(response.json()["job_id"])

        key = job.payload["original_key"]
        assert objects.exists(routes_jobs.LAYER_ORIGINAL, key)
        assert objects.get(routes_jobs.LAYER_ORIGINAL, key) == body
        # And the staging copy really is gone, so the test is not passing because
        # a file was accidentally left behind on the local disk.
        assert list(Path(config.UPLOAD_FOLDER).glob("*.txt")) == []

    def test_the_payload_carries_no_bytes_and_no_api_key(self, api, job_store):
        """A payload is a JSON row in a database. A key written there is a key
        at rest, in a file that gets backed up.

        The key is parked in process memory instead, so the same request still
        works — but the durable record of it is a boolean, not a secret.
        """
        secret = "sk-live-9999999999999999"
        response = upload(api, api_key=secret)
        job = job_store.get_job(response.json()["job_id"])

        assert secret not in json.dumps(job.payload, ensure_ascii=False)
        assert secret not in json.dumps(job.context, ensure_ascii=False)
        assert secret not in json.dumps(job.result, ensure_ascii=False)
        # Only a name, an option flag and an object store key.
        assert set(job.payload) == {
            "original_key", "source_filename", "source_suffix", "media_type",
            "generate_mermaid", "generate_evaluation", "generate_research_brief",
        }

    def test_a_browser_key_is_recovered_for_the_stage_that_needs_it(self, api, job_store):
        """The parked key has to actually reach the stage, or the feature is a
        security fix that also breaks the product."""
        response = upload(api, api_key="sk-live-4242424242424242")
        job = job_store.get_job(response.json()["job_id"])
        assert routes_jobs.take_ephemeral_secret(job.job_id) == "sk-live-4242424242424242"
        # Single-use: a key left in the map is a key waiting to be read by
        # something. Draining it again returns nothing.
        assert routes_jobs.take_ephemeral_secret(job.job_id) == ""


# --------------------------------------------------------------------------
# The synchronous path still returns an analysis
# --------------------------------------------------------------------------


class TestTheSynchronousPath:
    def test_the_legacy_route_still_returns_an_analysis(self, api, job_store):
        """T2.4's criterion is that the existing front end switches over without
        breaking, and the front end renders its workspace from this body. So the
        response must still be the analysis, and it must still be a 200.
        """
        response = upload_analyze(api, api_key="sk-live-0000000000000000")

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["session_id"] == "sess_stub"
        assert "summary" in body["sections"]
        # Additive only: a client reading the analysis sees what it saw before.
        assert body["job_id"]
        assert body["job_state"] == J.STATE_COMPLETED

    def test_the_synchronous_path_uses_the_same_job(self, api, job_store):
        """One submission, one pipeline, one job — not "the old way" and "the new
        way". If the sync route had its own copy of the work, this is where two
        pipelines would have drifted.
        """
        upload_analyze(api, api_key="sk-live-0000000000000000")
        jobs = job_store.list_jobs(kind=routes_jobs.KIND_DOCUMENT_ANALYSIS)
        assert len(jobs) == 1
        assert jobs[0].state == J.STATE_COMPLETED

    def test_the_synchronous_route_answers_202_when_the_work_is_still_going(
        self, api, job_store, monkeypatch
    ):
        """A request may not be held for ever, and running out of patience is not
        the same as running out of work. The answer is the same 202 an async
        submit would have given, plus the id to watch — not a fake failure.
        """
        monkeypatch.setattr(routes_jobs, "SYNC_WAIT_TIMEOUT_SECONDS", 0.0)
        response = upload_analyze(api, api_key="sk-live-0000000000000000")

        assert response.status_code == 202
        assert response.json()["job_id"]
        assert response.json()["state"] == J.STATE_QUEUED

    def test_the_streaming_route_still_emits_the_events_the_browser_consumes(
        self, api, job_store
    ):
        """``analyze.js`` dispatches on ``start``/``section``/``done``. Those three
        names and their shapes are the front end's contract, so they are asserted
        rather than assumed — a rename here breaks every section render.
        """
        response = api.post(
            "/api/analyze/stream",
            files={"file": ("paper.txt", io.BytesIO(b"a document"), "text/plain")},
            data={"api_key": "sk-live-0000000000000000"},
        )
        assert response.status_code == 200, response.text
        events = sse_events(response.text)
        names = [name for _id, name, _payload in events]

        assert names[0] == "start"
        assert "done" in names
        done = next(payload for _id, name, payload in events if name == "done")
        assert done["session_id"] == "sess_stub"

    def test_the_start_event_carries_the_job_id_so_a_client_can_come_back(
        self, api, job_store
    ):
        """The re-attach handle. Without it the stream is a one-way door: a tab
        that closed at minute two has nothing to come back to.
        """
        response = api.post(
            "/api/analyze/stream",
            files={"file": ("paper.txt", io.BytesIO(b"a document"), "text/plain")},
            data={"api_key": "sk-live-0000000000000000"},
        )
        events = sse_events(response.text)
        start = next(payload for _id, name, payload in events if name == "start")

        job_id = start["job_id"]
        assert start["job_stream_path"] == f"/v1/jobs/{job_id}/stream"
        # And the handle is real: the job it names exists and reached a terminal
        # state on its own.
        assert job_store.get_job(job_id).state == J.STATE_COMPLETED


# --------------------------------------------------------------------------
# The state machine
# --------------------------------------------------------------------------


class TestTheStateMachine:
    def test_a_job_walks_the_spine_to_completed(self, api, job_store):
        job_id = upload(api, api_key="sk-live-0000000000000000").json()["job_id"]
        job = drain(job_store, job_id)

        assert job.state == J.STATE_COMPLETED
        assert job.terminal
        # Every stage the pipeline declares, in order, and nothing skipped. The
        # submission event is ``to_state='queued'`` and is dropped here: what is
        # under test is the spine the job walked, not the row it started from.
        walked = [e.to_state for e in job.events if e.to_state != J.STATE_QUEUED]
        for declared in routes_jobs.analysis_pipeline().states:
            assert declared in walked, f"{declared} 没有出现在任务轨迹中"
        assert walked.index(J.STATE_PARSING) < walked.index(J.STATE_ANALYZING)
        assert walked[-1] == J.STATE_COMPLETED

    def test_a_partial_parse_ends_the_job_partial_rather_than_failing_it(
        self, api, job_store, monkeypatch
    ):
        """A parse that produced usable text with something missing is a partial
        success. Routing it to ``failed`` would report "this document could not
        be parsed" about a document that was parsed well enough to analyse.
        """
        monkeypatch.setattr(
            ingest, "parse_document",
            lambda path, **kw: StubParseOutcome(warnings=["第 3 页未提取到文本。"]),
        )
        job_id = upload(api, api_key="sk-live-0000000000000000").json()["job_id"]
        job = drain(job_store, job_id)

        assert job.state == J.STATE_PARTIAL
        assert job.partial is True
        assert any("第 3 页" in reason for reason in job.partial_reasons)

    def test_an_unusable_parse_fails_the_job_with_a_readable_reason(
        self, api, job_store, monkeypatch
    ):
        monkeypatch.setattr(
            ingest, "parse_document", lambda path, **kw: StubParseOutcome(usable=False)
        )
        job_id = upload(api, api_key="sk-live-0000000000000000").json()["job_id"]
        job = drain(job_store, job_id)

        assert job.state == J.STATE_FAILED
        assert job.error_message == "模拟的解析失败。"
        # Not retryable: the same bytes will fail the same way, and a retry
        # would bill for the same answer.
        assert job.retryable is False

    def test_the_job_records_the_parse_facts_and_never_the_document(
        self, api, job_store
    ):
        """``result`` is a pointer and a set of counts. A job row that grew the
        document text would be a second copy of the analysis with no rule about
        which one is current.
        """
        job_id = upload(api, api_key="sk-live-0000000000000000").json()["job_id"]
        result = drain(job_store, job_id).result

        assert result["page_count"] == 2
        assert result["char_count"] == 1200
        assert result["parser_name"] == "stub"
        assert result["session_id"] == "sess_stub"
        assert not any("text" in key for key in result)

    def test_a_missing_object_makes_the_job_fail_rather_than_crash_the_worker(
        self, api, job_store, objects
    ):
        """The stage must not raise out of the pipeline: an unhandled exception
        in a step is caught by the worker, but the ``ObjectNotFound`` here is an
        expected operational outcome and gets a Chinese sentence rather than a
        stack trace in the job's error field.
        """
        job_id = upload(api, api_key="sk-live-0000000000000000").json()["job_id"]
        job = job_store.get_job(job_id)
        objects.delete(routes_jobs.LAYER_ORIGINAL, job.payload["original_key"], force=True)

        settled = drain(job_store, job_id)
        assert settled.state == J.STATE_FAILED
        assert "对象存储" in settled.error_message


# --------------------------------------------------------------------------
# Durability: a worker that dies
# --------------------------------------------------------------------------

#: A child process that claims a real job in a real database and dies inside the
#: parsing stage, with the object store and the upload folder pointed at the
#: parent's temporary directories so that the bytes it was about to read really
#: are there.
_DYING_WORKER = '''
import os, sys

# The package by absolute path rather than by inheriting this process's
# ``sys.path``: a subprocess started from a test must not depend on how the test
# itself happened to be launched.
sys.path.insert(0, {project_root!r})

os.environ["STORAGE_SQLITE_PATH"] = sys.argv[1]
os.environ["OBJECT_STORE_ROOT"] = sys.argv[2]
os.environ["UPLOAD_FOLDER"] = sys.argv[3]
os.environ["PARSER_CACHE_ENABLED"] = "0"
os.environ["PARSER_BACKEND"] = "gateway"

from paperwhisperer.api import routes_jobs
from paperwhisperer.core import config
from paperwhisperer.ingestion import orchestrator as ingest
from paperwhisperer.infra import jobs as J
from paperwhisperer.infra.stores import build_sqlite_bundle


def die_mid_parse(file_path, **kwargs):
    """Enter the stage, prove the bytes were reachable, then stop existing.

    ``os._exit`` rather than an exception, because an exception would unwind into
    the worker's handler and let it settle the job — which is the graceful path,
    not the one under test. What is under test is what a *killed* worker leaves
    behind in the database.

    The path is written out before exiting, so the parent can assert that the
    stage really was reading the object store's copy of the upload rather than a
    file left over on the local disk.
    """
    with open(sys.argv[4], "w", encoding="utf-8") as handle:
        handle.write(str(file_path))
        handle.flush()
        os.fsync(handle.fileno())
    os._exit(9)


ingest.parse_document = die_mid_parse
config.OBJECT_STORE_ROOT = os.environ["OBJECT_STORE_ROOT"]

store = J.build_job_store(bundle=build_sqlite_bundle(config))
worker = J.JobWorker(store, routes_jobs.analysis_pipeline(), worker_id="doomed-api-worker",
                     lease_seconds=1)
worker.run_once()
raise SystemExit("the worker was supposed to die")
'''


def kill_a_worker_inside_a_stage(db_path, objects_root, uploads_root, marker_path):
    """Run a worker in its own process and kill it inside the parsing stage."""
    script = Path(uploads_root).parent / "dying_api_worker.py"
    script.write_text(
        _DYING_WORKER.format(project_root=str(PROJECT_ROOT)), encoding="utf-8"
    )
    return subprocess.run(
        [sys.executable, str(script), str(db_path), str(objects_root),
         str(uploads_root), str(marker_path)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


class TestRecoveryFromAKilledApiWorker:
    """The release gate again, pointed at the API's own pipeline.

    ``tests/test_jobs.py`` kills a worker inside a synthetic step it defines
    itself. That proves the lease and the recovery pass work; it does not prove
    that *this* module's stages are on the other side of a real kill. The job
    here is submitted through the HTTP surface, dies inside
    ``routes_jobs._parse_stage``, and is finished by a worker in this process —
    so the test covers the pipeline, the object-store materialisation and the
    recovery together, which is the combination the exit criterion is about.
    """

    @pytest.fixture
    def corpse(self, api, job_store, bundle, objects, tmp_path):
        job_id = upload(api, api_key="sk-live-0000000000000000").json()["job_id"]
        marker = tmp_path / "reached.txt"
        result = kill_a_worker_inside_a_stage(
            bundle.database.path,
            str(tmp_path / "objects"),
            str(tmp_path / "uploads"),
            marker,
        )
        # Both of these, or every assertion below could pass for the wrong
        # reason — most of all by never having exercised a crash at all.
        assert result.returncode == 9, f"子进程未按预期退出：{result.stderr}"
        assert marker.exists(), "子进程没有进入解析阶段"
        # It really was reading the *stored original*: the path it was given
        # exists and holds the uploaded bytes, which is the whole of "the file
        # survives the request" seen from the other side.
        staged = Path(marker.read_text(encoding="utf-8"))
        assert staged.exists()
        assert staged.read_bytes() == b"a document about testing"
        yield job_store, job_id
        routes_jobs.stop_job_workers()

    def test_a_killed_worker_leaves_a_job_this_process_can_see(self, corpse):
        job_store, job_id = corpse
        dead = job_store.get_job(job_id)
        assert dead.state == J.STATE_PARSING
        assert dead.lease_owner == "doomed-api-worker"
        # Invisible to the queue: nothing can claim it, which is exactly why
        # recovery has to go looking rather than wait for it to come back.
        assert job_store.claim_next("healthy", 60,
                                    kind=routes_jobs.KIND_DOCUMENT_ANALYSIS) is None

    def test_recovery_requeues_it_and_this_process_finishes_it(self, corpse):
        job_store, job_id = corpse
        moved = job_store.recover_stale(now=time.time() + FUTURE)
        assert [item.job_id for item in moved] == [job_id]
        assert job_store.get_job(job_id).state == J.STATE_QUEUED

        finished = drain(job_store, job_id)
        assert finished.state == J.STATE_COMPLETED
        assert finished.attempts == 2
        # The trail still names who died, so "why did this take twice?" is
        # answerable from the job itself.
        assert any("doomed-api-worker" in event.message for event in finished.events)


# --------------------------------------------------------------------------
# The lifespan actually runs a worker
# --------------------------------------------------------------------------

#: The budget of one bounded teardown stop. The same number the app lifespan's
#: own shutdown uses, so an isolated worker is stopped the way production stops
#: one rather than the way only a test would.
STOP_TIMEOUT = 5.0

#: How many times the isolation below re-issues that bounded stop before it
#: gives up. Bounded because a shutdown that blocks is the failure this design
#: exists to avoid; repeated because a worker whose budget expired stays
#: *reachable*, so the next call re-joins it with a fresh budget and collects
#: it. Three attempts of five seconds is enough for any step that is slow
#: rather than stuck, and a genuinely stuck thread still lets the test run.
STOP_ATTEMPTS = 3


def job_worker_threads() -> set[str]:
    """The names of every live ``job-worker-*`` thread in this process.

    A thread probe rather than a read of the registry, and the reason is the
    whole subject of this file's worker tests. ``active_workers()`` and
    ``stuck_workers()`` both read the dict the teardown writes, so asserting
    against them is the code agreeing with itself: a registry entry whose
    thread has already exited is indistinguishable, there, from a thread that
    is genuinely still draining the queue. Only the interpreter's own list of
    live threads can tell the two apart.
    """
    return {thread.name for thread in threading.enumerate()
            if thread.name.startswith("job-worker-")}


@pytest.fixture
def isolated_worker_registry():
    """The process's worker registry, drained, for the length of one test.

    **Why these tests need it at all.** :data:`routes_jobs._WORKERS` is a module
    global, so a worker registered by an *earlier test module* is still there
    when this class runs — and a worker whose in-flight step outlived that
    module's shutdown stays there for the rest of the session, reachable but
    not stopped. Asserting ``active_workers() == []`` without arranging for it
    therefore asserted something about the run's history rather than about this
    test: it passed when the file ran alone and failed in the full suite,
    because a worker in ``tests/test_e2e_journey.py`` was mid-analysis when
    that module ended. That is order-dependence, and a test whose pass/fail is
    a function of which module ran before it pins nothing.

    **Drained rather than cleared.** The registry is emptied by re-issuing the
    bounded stop, not by dropping the entries: an entry removed while its
    thread is alive is precisely the orphan commit ``341792e`` was about, and a
    fixture must not be the thing that reintroduces it. Whatever cannot be
    stopped within the attempts is cleared anyway so this class's assertions
    stay about its own workers — and a thread that survives that is still
    reported by :func:`job_worker_threads`, which is the honest observable
    every other file in this area uses too.

    **Nothing is restored on the way out.** The previous modules' workers have
    been asked to stop by now, and handing a still-live handle back to the
    registry would re-poison whichever module runs next — which is the leak
    this fixture exists to keep from spreading.
    """
    def drain() -> None:
        for _attempt in range(STOP_ATTEMPTS):
            routes_jobs.stop_job_workers(timeout=STOP_TIMEOUT)
            if not job_worker_threads():
                break

    drain()
    routes_jobs._WORKERS.clear()
    routes_jobs._STRANDED.clear()
    try:
        yield routes_jobs._WORKERS
    finally:
        drain()
        routes_jobs._WORKERS.clear()
        routes_jobs._STRANDED.clear()


class TestTheWorkerLifecycle:
    def test_a_submitted_job_is_claimed_by_the_applications_own_worker(self, live_api, job_store):
        """The gap T2.3 was opened to close: a complete job engine with nothing
        driving it. Here a background thread the *lifespan* started picks the job
        up with no request waiting for it.
        """
        assert routes_jobs.active_workers(), "JOBS_ENABLED did not start a worker"

        job_id = upload(live_api, api_key="sk-live-0000000000000000").json()["job_id"]
        job = job_store.get_job(job_id)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not job.terminal:
            time.sleep(0.05)
            job = job_store.get_job(job_id)

        assert job.terminal, "the lifespan's worker never claimed the job"
        assert job.state == J.STATE_COMPLETED
        assert StubAnalyzer.calls, "the work never happened"

    def test_the_worker_is_stopped_on_teardown(self, bundle, objects, job_store,
                                               stubbed_pipeline, monkeypatch,
                                               isolated_worker_registry):
        """A worker that is not stopped on shutdown is a leaked thread, and a
        leaked worker holding a lease is worse than useless — the next process
        recovers its job and both processes bill for the same cloud parse.

        **The assertion is about the threads, not about the registry.** The
        failure this names is a thread that outlived the lifespan, and the only
        honest evidence about threads is the interpreter's own list of them: a
        registry that has forgotten a live worker reads exactly like a registry
        that has finished with one. So the workers are named while they are up,
        and their absence is proved by the probe.
        """
        monkeypatch.setattr(config, "JOBS_POLL_INTERVAL_SECONDS", 1)
        overrides = {
            routes_evidence.get_store_bundle: lambda: bundle,
            routes_v1.get_object_store: lambda: objects,
            routes_jobs.get_object_store: lambda: objects,
            routes_jobs.get_job_store: lambda: job_store,
        }
        monkeypatch.setattr(routes_jobs, "resolve_object_store", lambda: objects)
        app.dependency_overrides.update(overrides)
        try:
            with TestClient(app):
                started = routes_jobs.active_workers()
                assert started, "the lifespan started no worker to stop"
                # Proof it was *this* test's worker and not a stale entry: a real
                # thread with this name existed while the app was up.
                assert job_worker_threads(), "lifespan 注册了 worker，但没有线程"
                thread_names = {f"job-worker-{worker.worker_id}" for worker in started}
                assert thread_names & job_worker_threads(), (
                    "lifespan 启动的 worker 没有对应的线程"
                )

            survivors = job_worker_threads() & thread_names
            assert not survivors, (
                f"lifespan 退出后线程仍然存活：{sorted(survivors)}。"
                "一个没有被停止的 worker 会继续占用租约。"
            )
            # And the registry agrees, which is the question an operator asks.
            # Asserted *after* the probe rather than instead of it.
            assert routes_jobs.active_workers() == []
            assert routes_jobs.stuck_workers() == []
        finally:
            routes_jobs.stop_job_workers()
            for dependency in overrides:
                app.dependency_overrides.pop(dependency, None)

    def test_jobs_disabled_starts_no_worker(self, bundle, objects, job_store,
                                            stubbed_pipeline, monkeypatch,
                                            isolated_worker_registry):
        """``JOBS_ENABLED`` is documented as the switch back to synchronous
        handling while diagnosing something, so it has to actually leave the
        worker unstarted rather than half-starting it.

        Pinned as *no new thread*, not only as an empty registry. A worker that
        was started and then quietly stopped would still leave an empty registry
        behind, and that is a deployment paying for a thread it was told had
        been switched off.
        """
        monkeypatch.setattr(config, "JOBS_ENABLED", False)
        overrides = {
            routes_evidence.get_store_bundle: lambda: bundle,
            routes_v1.get_object_store: lambda: objects,
            routes_jobs.get_object_store: lambda: objects,
            routes_jobs.get_job_store: lambda: job_store,
        }
        monkeypatch.setattr(routes_jobs, "resolve_object_store", lambda: objects)
        app.dependency_overrides.update(overrides)
        before = job_worker_threads()
        try:
            with TestClient(app):
                assert routes_jobs.active_workers() == []
                assert job_worker_threads() == before, (
                    f"JOBS_ENABLED=False 仍然启动了线程："
                    f"{sorted(job_worker_threads() - before)}"
                )
        finally:
            routes_jobs.stop_job_workers()
            for dependency in overrides:
                app.dependency_overrides.pop(dependency, None)


# --------------------------------------------------------------------------
# Resync
# --------------------------------------------------------------------------


class TestResync:
    def test_a_subscriber_sees_state_and_a_reconnect_says_what_it_missed(
        self, api, job_store
    ):
        """The Phase 2 exit criterion "浏览器刷新/SSE 断开不丢任务", over the wire.

        A client reads the feed, drops the connection, and comes back with the
        sequence number it last saw. The two halves together assert the property
        that matters: the client can say *exactly* which transitions it missed,
        rather than having to re-read everything and guess.
        """
        job_id = upload(api, api_key="sk-live-0000000000000000").json()["job_id"]
        # Settle it first, so the whole trail exists and a fresh subscriber has
        # something to resync against.
        drain(job_store, job_id)

        first = api.get(f"/v1/jobs/{job_id}/stream")
        events = sse_events(first.text)
        names = [name for _id, name, _payload in events]
        assert names[0] == "snapshot"
        assert "state" in names
        # The sequence rides on the SSE ``id:`` line, not only inside the JSON:
        # the client's cursor has to survive a reconnect without parsing a body.
        state_ids = [event_id for event_id, name, _p in events if name == "state"]
        assert state_ids and all(value.isdigit() for value in state_ids)

        seen_through = int(state_ids[0])
        missed = api.get(
            f"/v1/jobs/{job_id}/events", params={"since_sequence": seen_through}
        ).json()
        assert missed["events"], "the feed reported nothing new to tell a reconnecting client"
        assert all(int(e["sequence"]) > seen_through for e in missed["events"])
        assert missed["sequence"] >= max(int(e["sequence"]) for e in missed["events"])

    def test_a_reconnect_with_last_event_id_resumes_rather_than_replays(
        self, api, job_store
    ):
        """``Last-Event-ID`` is the header a browser's own ``EventSource`` sends
        back. Honouring it is what makes the resync work without the client
        having to remember to ask for it.
        """
        job_id = upload(api, api_key="sk-live-0000000000000000").json()["job_id"]
        drain(job_store, job_id)
        full = sse_events(api.get(f"/v1/jobs/{job_id}/stream").text)
        cursor = [int(i) for i, n, _p in full if n == "state"][0]

        resumed = sse_events(
            api.get(
                f"/v1/jobs/{job_id}/stream", headers={"Last-Event-ID": str(cursor)}
            ).text
        )
        sequences = [int(p["sequence"]) for _i, n, p in resumed if n == "state"]
        assert sequences, "the reconnect replayed nothing and was told nothing"
        assert all(sequence > cursor for sequence in sequences)
        # And the snapshot still comes first on a reconnect: a client that was
        # away still needs to know where things stand.
        assert resumed[0][1] == "snapshot"

    def test_the_event_feed_partitions_the_trail_with_no_gap_and_no_duplicate(
        self, api, job_store
    ):
        """Two consecutive polls must exactly partition the transitions. A
        duplicate is as wrong as a gap: a client that sees a state twice cannot
        tell a retry from a second attempt.
        """
        job_id = upload(api, api_key="sk-live-0000000000000000").json()["job_id"]
        drain(job_store, job_id)

        everything = job_store.events_since(job_id, 0)
        assert len(everything) >= 4, "轨迹太短，测不出分段"

        # Partition the way a client actually does it: read everything, then
        # read again from the cursor the first read handed back. Two consecutive
        # polls must be disjoint and, between them, complete.
        first = api.get(f"/v1/jobs/{job_id}/events").json()
        second = api.get(
            f"/v1/jobs/{job_id}/events",
            params={"since_sequence": first["sequence"]},
        ).json()
        joined = [e["sequence"] for e in first["events"] + second["events"]]

        assert joined == sorted(set(joined)), "两次轮询之间有重复"
        assert joined == sorted(e.sequence for e in everything), "有一次状态迁移丢失了"
        # And the cursor really does advance, which is what makes the second poll
        # cheap rather than a full replay.
        assert second["sequence"] >= first["sequence"]

    def test_a_job_survives_a_dropped_stream_and_still_answers(self, api, job_store):
        """A client whose tab closed mid-analysis did not throw the work away.
        The job is on a worker thread and its progress is in a table, so the
        client's whole recovery is one ``GET``.
        """
        response = api.post(
            "/api/analyze/stream",
            files={"file": ("paper.txt", io.BytesIO(b"a document"), "text/plain")},
            data={"api_key": "sk-live-0000000000000000"},
        )
        job_id = sse_events(response.text)[0][2]["job_id"]

        # The client "disconnected" — it simply stopped reading. The job is
        # unaffected, and a second caller can be told exactly how it ended.
        settled = job_store.get_job(job_id)
        assert settled.terminal
        state = api.get(f"/v1/jobs/{job_id}").json()
        assert state["state"] == J.STATE_COMPLETED
        # And the pointer to the answer survived, which is what makes the
        # recovery a read rather than a second model call.
        assert state["result"]["session_id"] == "sess_stub"


# --------------------------------------------------------------------------
# Deduplication, cancellation, and the store seam
# --------------------------------------------------------------------------


class TestDeduplication:
    def test_the_same_upload_twice_is_one_job(self, api, job_store):
        """A double-clicked button, a retried fetch, a client that reconnected
        before it read the first response. All three are one request, and one
        request should be one parse.
        """
        body = b"identical bytes, submitted twice"
        first = upload(api, body=body, api_key="sk-live-0000000000000000")
        second = upload(api, body=body, api_key="sk-live-0000000000000000")

        assert first.json()["job_id"] == second.json()["job_id"]
        assert second.json()["deduplicated"] is True
        assert len(job_store.list_jobs(kind=routes_jobs.KIND_DOCUMENT_ANALYSIS)) == 1

    def test_different_options_are_different_jobs(self, api, job_store):
        """The key is composed of the content digest *and* the options. If it
        were only the digest, asking for the same paper without a mind map would
        silently return the run that had one.
        """
        body = b"identical bytes, different options"
        first = upload(api, body=body, api_key="sk-live-0000000000000000")
        second = upload(api, body=body, api_key="sk-live-0000000000000000",
                        generate_mermaid="false")

        assert first.json()["job_id"] != second.json()["job_id"]
        assert second.json()["deduplicated"] is False

    def test_different_content_is_always_a_different_job(self, api, job_store):
        first = upload(api, body=b"one paper", api_key="sk-live-0000000000000000")
        second = upload(api, body=b"a different paper", api_key="sk-live-0000000000000000")
        assert first.json()["job_id"] != second.json()["job_id"]


class TestCancellation:
    def test_a_cancel_is_accepted_and_honoured_before_the_work_starts(
        self, api, job_store
    ):
        """Cooperative, and honest about it: a job nobody has started needs no
        worker's cooperation to stop, so "I cancelled it" is true immediately.
        """
        job_id = upload(api, api_key="sk-live-0000000000000000").json()["job_id"]
        ack = api.post(f"/v1/jobs/{job_id}/cancel").json()

        assert ack["cancel_requested"] is True
        assert ack["accepted"] is True
        assert ack["message"]

        # And the worker, when it arrives, finds the job already settled from
        # the queue rather than claiming it and dropping it.
        job_store.apply_queued_cancellations()
        assert job_store.get_job(job_id).state == J.STATE_CANCELLED
        assert StubAnalyzer.calls == [], "a cancelled job still spent money"

    def test_cancelling_twice_is_not_an_error(self, api, job_store):
        """It is a client retrying a request whose response it never saw.
        """
        job_id = upload(api, api_key="sk-live-0000000000000000").json()["job_id"]
        first = api.post(f"/v1/jobs/{job_id}/cancel")
        second = api.post(f"/v1/jobs/{job_id}/cancel")

        assert first.status_code == 200
        assert second.status_code == 200
        assert second.json()["cancel_requested"] is True

    def test_a_cancel_arriving_after_completion_says_so(self, api, job_store):
        """The honest answer, rather than a 200 a client would read as "stopped".
        A user who is told "cancelled" while the bill for that run keeps arriving
        has been lied to.
        """
        job_id = upload(api, api_key="sk-live-0000000000000000").json()["job_id"]
        drain(job_store, job_id)

        ack = api.post(f"/v1/jobs/{job_id}/cancel").json()
        assert ack["accepted"] is False
        assert job_store.get_job(job_id).state == J.STATE_COMPLETED


class TestCredentialLeaks:
    """The findings T2.3 closed, asserted on the paths that were open.

    ``tests/test_ingestion_security.py`` records these as ``OPEN_FINDINGS``; this
    is the same property asserted at the routers rather than at the helper, so
    that a future route that reaches for ``str(exc)`` directly fails here.
    """

    def test_a_leaking_exception_does_not_reach_the_papers_routes(self, api, monkeypatch):
        from paperwhisperer.api import routes_papers

        monkeypatch.setattr(
            routes_papers, "search_papers",
            lambda **kw: (_ for _ in ()).throw(RuntimeError(f"upstream said: invalid key {SENTINEL}")),
        )
        response = api.post("/api/search-papers", json={"query": "graphs"})
        assert SENTINEL not in response.text
        assert response.json()["code"] == "paper_search_failed"

    def test_a_leaking_exception_does_not_reach_the_qa_routes(self, api, monkeypatch):
        from paperwhisperer.sessions import store as sessions

        monkeypatch.setattr(
            sessions, "load_validated_session",
            lambda *a, **kw: (_ for _ in ()).throw(
                ValueError(f"upstream said: invalid key {SENTINEL}")
            ),
        )
        response = api.post("/api/ask", json={"question": "why?", "api_key": "sk-live-0000000000000000"})
        assert SENTINEL not in response.text
        assert response.json()["code"] == "invalid_request"

    def test_the_code_survives_redaction_so_a_client_can_still_branch(
        self, api, monkeypatch
    ):
        """Redaction changes the prose, never the API. A client that branches on
        ``code`` must be unaffected by the fix.
        """
        from paperwhisperer.api import routes_papers

        monkeypatch.setattr(
            routes_papers, "search_papers",
            lambda **kw: (_ for _ in ()).throw(RuntimeError(f"invalid key {SENTINEL}")),
        )
        response = api.post("/api/search-papers", json={"query": "graphs"})
        body = response.json()
        assert body["code"] == "paper_search_failed"
        assert body["error"]
        assert body["timestamp"]
