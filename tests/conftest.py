"""Shared pytest fixtures.

Two rules shape everything here:

1. **No test may touch the real runtime folders.** ``config`` resolves
   ``UPLOAD_FOLDER`` / ``OUTPUT_FOLDER`` / ``CONTEXT_FOLDER`` as plain module
   attributes, so ``monkeypatch.setattr(config, ...)`` redirects every reader
   at once — including the ones living deep inside ``sessions.store``.
2. **No test may read the developer's real credentials.** ``.env`` is loaded at
   import time and this repository ships one, so the API-key variables are
   stripped for the whole session unless a test asks for them back.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from paperwhisperer.app import app  # noqa: E402
from paperwhisperer.core import config  # noqa: E402
from paperwhisperer.remote import ssrf  # noqa: E402

from .helpers import FakeResponse, FakeUploadFile  # noqa: E402

#: Environment variables that would otherwise leak the developer's own
#: credentials into a test run and turn an offline test into a live API call.
CREDENTIAL_ENV_VARS = (
    "OPENAI_API_KEY",
    "SEMANTIC_SCHOLAR_API_KEY",
)


#: Tests that are *allowed* to reach a real model endpoint opt in by restoring
#: a real ``OPENAI_API_KEY`` into the environment (see ``test_llm_integration``
#: and ``test_e2e_journey``). The hermetic suite never does: ``isolated_environment``
#: strips the key, so its presence at the moment an analyzer would be built is a
#: reliable, opt-in signal that distinguishes the two populations.
def _live_model_opted_in() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


@pytest.fixture
def isolated_job_stores(tmp_path, monkeypatch):
    """Give one test its own, empty job store, object store and database.

    T2.3 made ``/api/analyze`` durable: the request writes the upload into the
    object store's ``original`` layer and submits a job row, and the job's stages
    later materialise those bytes back out and re-read them. ``runtime_dirs``
    already redirects the three runtime folders, but the job store, the object
    store and their database are reached through **process-wide singletons**
    (:func:`routes_evidence.get_store_bundle`, :func:`routes_v1.get_object_store`,
    :func:`routes_jobs.get_job_store`) that are built from ``config`` on first
    use and cached for the rest of the process, and ``STORAGE_SQLITE_PATH`` /
    ``OBJECT_STORE_ROOT`` default to paths inside the repository's ``context/``.
    Two consequences, both real:

    * a test that submits a job writes a row into the *developer's* database, and
      leaves it there for the next run; and
    * because a submission is deduplicated by content digest + session + options
      (:func:`routes_jobs.analysis_idempotency_key`), a later run reuses the
      previous run's job — which has already consumed its single-use API key and
      already had its result taken — so the request answers ``missing api key`` or
      a bare ``202`` instead of the analysis the test is asserting on. That is
      how the "hostile session id" case became order- and run-dependent.

    Pointing both backends at ``tmp_path`` and clearing the singletons makes the
    test start from an empty store, so nothing is written outside ``tmp_path`` and
    no job can be observed across tests. The configuration and the singletons are
    restored on teardown (like the process-global teardown in
    ``tests/test_observability.py``), because leaving a ``tmp_path``-backed store
    installed would point a later test at a directory that no longer exists.

    **Opt-in, not autouse.** The defaults are themselves part of the contract for
    a few suites (``test_backup_restore`` snapshots the configured database;
    ``test_asset_model`` asserts that the default object-store root follows
    ``CONTEXT_FOLDER``), and an autouse override would break them for a reason that
    has nothing to do with what they are checking. A suite reaches job submission
    only by calling ``/api/analyze``, ``/api/analyze/stream`` or
    ``/api/jobs/analyze``, so the fixture is requested by exactly the suites that
    do, and everything else keeps the shipped configuration.
    """
    from paperwhisperer.api import routes_evidence, routes_jobs, routes_v1

    previous = {
        "evidence_stores": getattr(routes_evidence, "_STORES", None),
        "v1_objects": getattr(routes_v1, "_OBJECTS", None),
        "jobs_stores": dict(getattr(routes_jobs, "_JOB_STORES", {}) or {}),
    }
    monkeypatch.setattr(config, "STORAGE_SQLITE_PATH", str(tmp_path / "stores" / "test.sqlite3"))
    monkeypatch.setattr(config, "OBJECT_STORE_ROOT", str(tmp_path / "objects"), raising=False)
    routes_evidence._STORES = None
    routes_v1._OBJECTS = None
    routes_jobs._JOB_STORES = {}
    yield
    routes_evidence._STORES = previous["evidence_stores"]
    routes_v1._OBJECTS = previous["v1_objects"]
    routes_jobs._JOB_STORES = previous["jobs_stores"]


@pytest.fixture(autouse=True)
def no_unstubbed_job_stage_model(monkeypatch):
    """Make a real model completion impossible for every hermetic test.

    **Why this exists.** T2.3 moved the analysis out of the request and into a
    durable job, so ``/api/analyze`` no longer constructs an analyzer in the
    route. The job's analysing stage
    (:func:`paperwhisperer.api.routes_jobs._analyse_stage`) imports
    ``DocumentAnalyzer`` **at call time** from
    :mod:`paperwhisperer.analysis.orchestrator` — the seam that is actually
    executed. Suites that stubbed ``DocumentAnalyzer`` *in the route module* had
    their stub silently bypassed, the real analyzer was built, and the first
    thing it did with the configured ``OPENAI_BASE_URL`` was open a socket: an
    offline test run became a live call against whatever endpoint the
    developer's ``.env`` pointed at. That is the defect this fixture closes.

    **How it is neutralised.** The analyzer the job stage would construct is
    replaced by a tripwire that raises *before* any network I/O, unless the
    running test has deliberately restored a real credential
    (:func:`_live_model_opted_in`) — the opt-in the two live suites already
    perform. A test that stubs this seam is unaffected: the per-test stub is
    applied after this fixture and therefore wins for the duration of the test.
    A test that forgets to stub now fails loudly and immediately instead of
    quietly reaching the shared, rate-limited endpoint.

    **Attribute reads are forwarded, not blocked.** Only *construction* is
    intercepted. The tripwire proxies every other attribute to the real class,
    because the production code itself reaches the class through this very
    module global — ``orchestrator.py`` calls ``DocumentAnalyzer.
    _strip_mermaid_fence`` from inside ``_normalize_mermaid_source`` — and
    several suites read the analyzer's classmethods and staticmethods
    (``test_orchestrator.py`` binds them at import). A tripwire that hid the
    class would break those for reasons that have nothing to do with a live
    call, which is the difference between a test that says "you forgot to stub"
    and one that says "this suite is broken".

    **Teardown.** Installed with ``monkeypatch``, so the attribute is restored
    when the test ends. This matters because the replacement is on a shared
    module: a leak here would blank out the real analyzer for every later test
    in the session, not just this one. (``tests/test_observability.py`` asserts
    the same property for the process-global logging singletons.)
    """
    from paperwhisperer.analysis import orchestrator as analysis_orchestrator

    real_analyzer = analysis_orchestrator.DocumentAnalyzer

    class _TripwireMeta(type):
        """Metaclass: refuse construction, forward every attribute read.

        Class-level attribute lookup (``DocumentAnalyzer._strip_mermaid_fence``)
        is resolved by the *metaclass*, so the forwarding has to live here; a
        ``__getattr__`` in the tripwire's own class body would only ever see
        instance lookups and would leave every classmethod and staticmethod
        unreadable. The production module itself reaches the class through this
        global (``orchestrator.py`` calls ``DocumentAnalyzer.
        _strip_mermaid_fence`` from inside ``_normalize_mermaid_source``), so a
        tripwire that hid the class would break it for reasons that have nothing
        to do with a live call.
        """

        def __call__(cls, *args, **kwargs):
            if _live_model_opted_in():
                # An explicitly-live test (see _live_model_opted_in). Build the
                # real analyzer so those tests still exercise the product.
                return real_analyzer(*args, **kwargs)
            raise AssertionError(
                "the job's analysing stage tried to construct a real "
                "DocumentAnalyzer. T2.3 moved analysis into a job, so the "
                "analyzer is built at call time from "
                "paperwhisperer.analysis.orchestrator — patching the route "
                "module no longer intercepts it. Stub "
                "'paperwhisperer.analysis.orchestrator.DocumentAnalyzer' (see "
                "tests/test_jobs_api.py::stubbed_pipeline), or restore a real "
                "OPENAI_API_KEY to opt into a live model."
            )

        def __getattr__(cls, name):
            return getattr(real_analyzer, name)

    _JobStageAnalyzerTripwire = _TripwireMeta("_JobStageAnalyzerTripwire", (object,), {})

    monkeypatch.setattr(analysis_orchestrator, "DocumentAnalyzer", _JobStageAnalyzerTripwire)


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    """Run every test as if no credentials were configured."""
    for name in CREDENTIAL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def no_leaked_observability_singleton():
    """Hand every test the observability singleton in its documented default state.

    ``paperwhisperer.core.observability.observability`` is a process-wide object
    with three mutable pieces: ``enabled``, a bounded ``trace`` ring buffer, and
    the ``metrics`` registry. Unlike ``config`` (which tests redirect with
    ``monkeypatch``) nothing rewinds these on its own, so any test that turns
    collection on to look at it changes what every test after it measures. That
    is not hypothetical here: ``tools/fault_inject.py`` is imported *in process*
    by ``tests/test_fault_injection.py``, and its ``recovery-signal`` scenario
    sets ``enabled = True``, drives ten real job transitions through the real
    store to prove ``jobs.recovered_total`` moves, and then restores ``enabled``
    and calls ``metrics.reset()`` in its ``finally``. It never clears
    ``trace``. The leak is invisible in every way except the one that matters:
    ``enabled`` reads ``False``, every counter reads ``0``, the ten
    ``job.submitted``/``job.transition`` events sit in the buffer, and
    ``tests/test_observability.py::TestItIsOffUntilAskedFor::
    test_importing_the_module_changes_nothing`` — the test whose entire job is
    to catch exactly this — fails with ``assert 10 == 0`` in a full run and
    passes alone.

    **Restore to the baseline rather than snapshotting the prior contents.**
    ``shutdown()`` clears the logging state but deliberately leaves the buffer
    alone, because in production a shutdown should not throw away the trace you
    were about to ship. Putting it back the way it was would need the recorder's
    and every counter's private series, and it would be the wrong contract
    anyway: the singleton is documented as **off until asked for**, so "clean"
    is the only state a test is entitled to hand the next one. A test that wants
    it configured calls ``configure()`` itself and is responsible for its own
    teardown, which is what ``tests/test_observability.py`` does.

    ``exporter`` is included because it is the same class of leak by a
    different route: ``configure(exporter_path=...)`` assigns onto the
    singleton, and a test that pointed it at a ``tmp_path`` file would leave
    every later flush writing to a directory that no longer exists.

    Autouse, and installed here rather than in ``test_observability.py``,
    because the leak is caused by a *different* file: the file that leaves the
    mess is not the file that trips over it, and a guard that only protects the
    suite it lives in does not protect the suite.
    """
    from paperwhisperer.core import observability as observability_module

    facade = observability_module.observability
    try:
        yield
    finally:
        # Logging first: shutdown() removes the handler and restores the record
        # factory, and the trace/metric reset below is what the module
        # deliberately does *not* do on its own.
        observability_module.shutdown()
        facade.trace.clear()
        facade.metrics.reset()
        facade.exporter = observability_module.NullExporter()
        facade.enabled = False
        assert facade.enabled is False, "the observability singleton was left enabled"
        assert len(facade.trace) == 0, "trace events leaked into the next test"
        assert facade.metrics.parsing_documents.total == 0, "metrics leaked into the next test"


@pytest.fixture(autouse=True)
def clean_hostname_cache():
    """The SSRF verdict cache is process-global; never let it bleed across tests."""
    ssrf.clear_hostname_cache()
    yield
    ssrf.clear_hostname_cache()


@pytest.fixture
def client():
    """A ``TestClient`` bound to the real application wiring."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def runtime_dirs(tmp_path, monkeypatch):
    """Point all three runtime folders at ``tmp_path`` and create them."""
    # Maps the folder name used everywhere else onto its config attribute —
    # "uploads" is the odd one out (UPLOAD_FOLDER, not UPLOADS_FOLDER).
    config_attr_by_folder = {
        "uploads": "UPLOAD_FOLDER",
        "output": "OUTPUT_FOLDER",
        "context": "CONTEXT_FOLDER",
    }
    folders = {}
    for folder_name, config_attr in config_attr_by_folder.items():
        path = tmp_path / folder_name
        path.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(config, config_attr, str(path))
        folders[folder_name] = path
    return folders


@pytest.fixture
def public_example_urls(monkeypatch):
    """Allow only ``https://example.com`` through the SSRF gate."""
    from paperwhisperer.remote import download

    monkeypatch.setattr(
        download,
        "is_public_http_url",
        lambda raw_url: str(raw_url).startswith("https://example.com"),
    )


@pytest.fixture
def patch_remote_response(monkeypatch):
    """Return a callable that swaps the remote-download transport for a fake.

    The seam is ``download._urlopen_pinned``, the one place a hop leaves the
    process. It moved off ``urllib.request.urlopen`` so the connection can dial
    a vetted address rather than resolving the hostname again at connect time;
    ``_urlopen_pinned`` is that connection's only entry point.
    """
    from paperwhisperer.remote import download

    def _patch(response):
        monkeypatch.setattr(
            download, "_urlopen_pinned", lambda *args, **kwargs: response
        )
        return response

    return _patch


@pytest.fixture
def fake_response():
    """Factory for :class:`helpers.FakeResponse`."""
    return FakeResponse


@pytest.fixture
def fake_upload_file():
    """Factory for :class:`helpers.FakeUploadFile`."""
    return FakeUploadFile


@pytest.fixture
def session_factory(runtime_dirs):
    """Persist a session and hand back ``(session_id, token)``."""
    from paperwhisperer.core.text import build_session_expiry
    from paperwhisperer.sessions.store import hash_session_token, write_session_payload

    counter = {"value": 0}

    def _create(document_content="document text", token=None, **extra):
        counter["value"] += 1
        session_id = f"session{counter['value']}"
        resolved_token = token if token is not None else f"token{counter['value']}"
        payload = {
            "expires_at": build_session_expiry(),
            "document_content": document_content,
            "qa_history": [],
            "paper_search": {},
            "session_auth": {"token_hash": hash_session_token(resolved_token)},
        }
        payload.update(extra)
        write_session_payload(session_id, payload)
        return session_id, resolved_token

    return _create


def read_sse_events(text: str) -> list[tuple[str, dict]]:
    """Parse an SSE body into ``[(event_name, payload), ...]``.

    Streaming responses are the API's main output format, so tests assert on
    decoded events instead of substrings — that keeps them honest when field
    names change and strict when behaviour does not.
    """
    events: list[tuple[str, dict]] = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        name = None
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())
        if name is None:
            continue
        raw = "\n".join(data_lines)
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            payload = {"_raw": raw}
        events.append((name, payload))
    return events


@pytest.fixture
def sse_events():
    return read_sse_events


@pytest.fixture
def project_root():
    return PROJECT_ROOT


@pytest.fixture
def no_sleep(monkeypatch):
    """Make retry backoff instant so retry tests stay fast."""
    import time as time_module

    monkeypatch.setattr(time_module, "sleep", lambda _seconds: None)


@pytest.fixture
def api_key_env(monkeypatch):
    """Opt a test back into having a server-level API key.

    Requesting the fixture is enough — it sets the variable on setup. The
    previous version returned a setter *function*, which meant every test that
    merely listed it as a dependency silently ran with no key at all and got
    ``missing_api_key`` instead of the behaviour it was asserting.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return "test-key"


def require_local_embedder() -> None:
    """Skip unless the *real* local embedding provider is loadable here.

    The dense leg's floors are cosines against one specific model, so a test that
    exercises them honestly cannot run against a stand-in. When that model is
    absent — a fresh clone with no download yet, an offline machine, a CI runner
    without the extra — the correct outcome is a skip, not a failure: nothing
    about the code under test has been shown to be wrong, the environment simply
    cannot supply a required external resource.

    This mirrors the guard ``tests/test_retrieval_floor.py`` has always used
    (``a missing extra is a skip, not a failure``). It lives here so every suite
    that genuinely needs the real embedder shares one definition of "usable"
    rather than each re-inventing a probe — and so none of them reports a red
    suite that is really just a green machine with no model on it.

    It is a no-op whenever the model is present: callers then run their full,
    unmodified assertions.
    """
    from paperwhisperer.llm.embeddings_local import LocalEmbeddingProvider

    try:
        LocalEmbeddingProvider().embed_query("a probe")
    except Exception as exc:  # noqa: BLE001 - a missing/unusable model is a skip
        pytest.skip(f"the local embedding provider is not usable here: {exc}")


@pytest.fixture
def local_embedder_usable():
    """Requestable no-op guard for suites that need the real embedder.

    Depending on this is enough — it runs :func:`require_local_embedder` at
    setup and skips the test if the model is absent. Put it on the module (as an
    ``autouse`` fixture) or on the specific fixture that reaches the embedder.
    """
    require_local_embedder()
    return True
