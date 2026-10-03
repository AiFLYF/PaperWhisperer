"""Shared API route behaviour.

Routes are exercised through the real ``TestClient`` wiring with only the two
things that would otherwise reach the network replaced: the LLM transport and
the remote-download layer. Session storage is redirected to ``tmp_path`` so a
test run cannot read or clobber the developer's real context folder.
"""

from __future__ import annotations

import asyncio
import io
import threading
import time
from types import SimpleNamespace

import pytest

from paperwhisperer.analysis import sections
from paperwhisperer.api.deps import pump_sync_events
from paperwhisperer.core import config
from paperwhisperer.llm import prompts


async def drain(stream) -> None:
    """Consume an async iterator to completion, discarding items."""
    async for _ in stream:
        pass


class RecordingAnalyzer:
    """Stand-in for ``DocumentAnalyzer`` that records how it was called."""

    instances: list[RecordingAnalyzer] = []
    sections_to_fail: set[str] = set()
    sections_to_be_empty: set[str] = set()
    answer_text = "answer body"
    chunks = ("第一段", "第二段")
    rewrite_meta = None
    recommendation = None
    analysis_result = None

    def __init__(self, api_key):
        self.api_key = api_key
        self.document_content = ""
        self.version = "test-version"
        self.calls: list[tuple] = []
        self.stream_calls: list[tuple] = []
        RecordingAnalyzer.instances.append(self)

    @classmethod
    def reset(cls, **overrides):
        cls.instances = []
        cls.sections_to_fail = set()
        cls.sections_to_be_empty = set()
        cls.answer_text = "answer body"
        cls.chunks = ("第一段", "第二段")
        cls.rewrite_meta = None
        cls.recommendation = None
        cls.analysis_result = None
        for name, value in overrides.items():
            setattr(cls, name, value)

    @classmethod
    def last(cls):
        """The most recently constructed stub.

        A ``property`` would not work here: tests hold the *class* (that is
        what the ``analyzer`` fixture yields), and reading a property off a
        class hands back the property object rather than a stub.
        """
        return cls.instances[-1]

    # -- analysis -------------------------------------------------------

    def _section_body(self, name):
        if name in self.sections_to_fail:
            raise RuntimeError(f"{name} 生成失败")
        return f"{name} 内容"

    def _build_result(self, mermaid=True, evaluation=True, brief=True):
        if self.analysis_result is not None:
            return self.analysis_result
        switched_off = {
            "mermaid": not mermaid,
            "evaluation": not evaluation,
            "research_brief": not brief,
        }
        empty = set(self.sections_to_be_empty)
        sections_map = {}
        for name in sections.SECTION_KEYS:
            if switched_off.get(name):
                status, content = "disabled", ""
            elif name in empty:
                status, content = "empty", ""
            else:
                status, content = "success", f"{name} 内容"
            sections_map[name] = {"status": status, "content": content,
                                  "error": "", "retryable": False}
        result = {"char_count": 10, "sections": sections_map}
        for name in sections.SECTION_KEYS:
            result[name] = sections_map[name]["content"]
        return result

    def analyze(self, file_path, mermaid=True, evaluation=True, brief=True):
        self.calls.append(("analyze", file_path, mermaid, evaluation, brief))
        for name in sections.SECTION_KEYS:
            enabled = {
                "mermaid": mermaid, "evaluation": evaluation, "research_brief": brief,
            }.get(name, True)
            if not enabled:
                continue
            self._section_body(name)
        return self._build_result(mermaid, evaluation, brief)

    def analyze_stream(self, file_path, mermaid=True, evaluation=True, brief=True):
        self.calls.append(("analyze_stream", file_path, mermaid, evaluation, brief))
        for name in sections.SECTION_KEYS:
            enabled = {
                "mermaid": mermaid, "evaluation": evaluation, "research_brief": brief,
            }.get(name, True)
            if not enabled:
                continue
            try:
                content = self._section_body(name)
                yield {"type": "section", "name": name,
                       "section": {"status": "success", "content": content,
                                   "error": "", "retryable": False}}
            except RuntimeError as exc:
                yield {"type": "section", "name": name,
                       "section": {"status": "failed", "content": "",
                                   "error": str(exc), "retryable": True}}

        yield {"type": "done", "result": self._build_result(mermaid, evaluation, brief)}

    # -- Q&A ------------------------------------------------------------

    def answer_question(self, question, history=None, answer_mode="evidence"):
        self.calls.append(("answer", question, list(history or []), answer_mode))
        return self.answer_text

    def stream_answer_question(self, question, history=None, answer_mode="evidence"):
        self.stream_calls.append(("stream", question, list(history or []), answer_mode))
        yield from self.chunks

    # -- search ---------------------------------------------------------

    def rewrite_search_query(self, query, context_text=""):
        self.calls.append(("rewrite", query, context_text))
        if self.rewrite_meta is not None:
            return self.rewrite_meta
        return {"original_query": query, "rewritten_query": f"rewritten:{query}",
                "topics": ["t"], "reason": "因为", "model": "cheap-model"}

    def recommend_papers(self, content, limit=None, search_fn=None):
        self.calls.append(("recommend", content, limit))
        if self.recommendation is not None:
            return self.recommendation
        return {"original_query": "orig", "query": "rewritten", "topics": ["t"],
                "reason": "r", "rewrite_model": "m",
                "items": [{"title": "Paper A", "url": "https://example.com/a"}],
                "errors": []}


def _stub_fact_layer_seams(monkeypatch):
    """Drive the migrated ``/api/analyze`` stage through stubs, not the network.

    The analysing stage
    (:func:`paperwhisperer.api.routes_jobs._analyse_stage`) was moved off the
    legacy six-section generator (``DocumentAnalyzer.analyze_stream``) onto the
    fact layer, and it now runs four collaborators *before* it produces any
    content, each imported at call time from the module it names:

    1. ``DocumentLoader.load_with_outcome`` — the real parse (and the one parse
       whose ``document_id`` the rest of the stage must reuse),
    2. ``orchestrator.build_qa_retriever`` — the pre-index the fact layer
       refuses to run without,
    3. ``paper_facts.make_evidence_provider`` — the per-field citations wrapper,
    4. ``service.analyze_with_fact_layer`` — the analysis itself.

    Patching only ``DocumentAnalyzer`` (which the fixture below also does) is no
    longer enough: with the real ``analyze_with_fact_layer`` in place it builds a
    *real* ``DocumentAnalyzer`` and runs the real DAG, and step 2 builds a dense
    retriever — an embedding pass that, on a machine without the local model
    cached, either hangs for minutes on download retries or fails the job. So
    every one of the four is replaced here.

    **The last stub is deliberately shallow, and deliberately calls the real
    finaliser.** It builds a ``RecordingAnalyzer`` (so ``analyzer.last()`` keeps
    working) and produces its content through the same
    :meth:`RecordingAnalyzer._build_result` helper the legacy path used, so the
    three switches still decide the section statuses exactly as they did. Then
    it hands off to the *real*
    :func:`paperwhisperer.analysis.service.finalize_analysis_result`, because
    that is where the product's real behaviour lives for everything these tests
    are actually about: the session id is sanitised, the session token is minted,
    the session is persisted, the Markdown report is written, and the result
    gains the 200-vs-422 semantics the route answers with. Stubbing the finaliser
    too would turn these into tests of a fake.

    ``register_analysis_document`` is the one thing inside the finaliser that
    reaches outside the process (it indexes the document for the citation API,
    which embeds), and none of these tests are about citations — so it is
    replaced with a no-op that reports the same ``registered`` shape the real one
    would, keeping the run hermetic and offline. Everything after it is real.
    """
    from paperwhisperer.analysis import orchestrator as analysis_orchestrator
    from paperwhisperer.analysis import paper_facts as analysis_paper_facts
    from paperwhisperer.analysis import service as analysis_service
    from paperwhisperer.documents.loader import DocumentLoader, LoadedDocument

    # (1) The parse. A stub outcome rather than a real ``ParseOutcome`` because
    # the stage reads exactly three things off it — ``.document`` (for
    # ``document_id``), ``.error_message`` (only when there is no document) and
    # nothing else — and arranging a genuine parser to run would be a second
    # thing under test in a test about the HTTP route.
    def fake_load_with_outcome(file_path):
        return LoadedDocument(
            text="stub document body",
            outcome=SimpleNamespace(
                document=SimpleNamespace(
                    document_id="doc_stub", page_count=1, char_count=18,
                    source_filename="paper.txt",
                ),
                error_message="",
                used_cloud=False,
            ),
        )

    monkeypatch.setattr(DocumentLoader, "load_with_outcome",
                        staticmethod(fake_load_with_outcome))

    # (2) The pre-index and (3) the evidence provider: dummies, because a real
    # retriever embeds the document and the fact layer's citations are not what
    # any test here is about.
    monkeypatch.setattr(
        analysis_orchestrator, "build_qa_retriever",
        lambda bundle, document=None: SimpleNamespace(indexed=document),
    )
    monkeypatch.setattr(
        analysis_paper_facts, "make_evidence_provider",
        lambda retriever, document_id, **kwargs: SimpleNamespace(document_id=document_id),
    )

    # The citation index, the one part of the finaliser that leaves the process.
    def fake_register_analysis_document(result, parse_outcome=None):
        result["document_registration"] = {
            "status": "registered",
            "document_id": str(result.get("document_id") or "doc_stub"),
            "chunk_count": 0,
            "index_version": "test",
            "code": "",
            "notice": "",
        }
        return result

    monkeypatch.setattr(analysis_service, "register_analysis_document",
                        fake_register_analysis_document)

    # (4) The analysis. The signature is the real one, so a caller that started
    # passing a new argument would be caught here rather than silently dropped.
    def fake_analyze_with_fact_layer(
        file_path, original_filename, api_key, generate_mermaid_bool,
        generate_evaluation_bool, session_id, generate_research_brief_bool=True,
        **kwargs,
    ):
        stub = RecordingAnalyzer(api_key)
        stub.calls.append((
            "analyze_with_fact_layer", str(file_path), generate_mermaid_bool,
            generate_evaluation_bool, generate_research_brief_bool,
        ))
        result = stub._build_result(
            generate_mermaid_bool, generate_evaluation_bool, generate_research_brief_bool
        )
        # ``run``/``report`` stay ``None``: the fact layer's provenance payload is
        # covered where it is actually produced (``test_fact_layer_route.py``),
        # and leaving them off keeps this stub from having to fabricate a
        # ``PaperFacts`` the assertions here never look at.
        return analysis_service.finalize_analysis_result(
            result=result,
            analyzer=stub,
            original_filename=original_filename,
            generate_evaluation_bool=generate_evaluation_bool,
            session_id=session_id,
            generate_research_brief_bool=generate_research_brief_bool,
            run=None,
            report=None,
        )

    monkeypatch.setattr(analysis_service, "analyze_with_fact_layer",
                        fake_analyze_with_fact_layer)


@pytest.fixture
def analyzer(monkeypatch, isolated_job_stores):
    """Swap in ``RecordingAnalyzer`` and hand back the class for configuration.

    ``DocumentAnalyzer`` has to be replaced in every module that binds the name,
    and the one that matters most is :mod:`paperwhisperer.analysis.orchestrator`:
    since T2.3 ``/api/analyze`` does not build the analyzer in the route at all —
    it submits a job, and the job's analysing stage
    (:func:`paperwhisperer.api.routes_jobs._analyse_stage`) imports
    ``DocumentAnalyzer`` **at call time** from that module. Patching only the
    route modules therefore let the real analyzer through, and the first thing
    it did with the configured ``OPENAI_BASE_URL`` was open a socket to the
    shared endpoint. The other four modules are still patched because
    ``/api/import-paper`` and the Q&A / papers routes call
    ``analysis.service.analyze_saved_file`` (and bind the name themselves).

    ``tests/conftest.py::no_unstubbed_job_stage_model`` is the suite-wide
    backstop for the same seam; this per-test stub is applied after it and so
    wins for the duration of the test.

    **The analysis seam itself has since moved again.** That stage no longer
    calls ``analyze_stream`` on the analyzer at all — it runs the fact layer,
    which is a *function* seam four collaborators deep. So
    :func:`_stub_fact_layer_seams` installs those four replacements as well; the
    ``DocumentAnalyzer`` patches below stay because the Q&A / papers routes still
    construct the analyzer directly, and the stub above still builds a
    ``RecordingAnalyzer`` so ``analyzer.last()`` keeps meaning something for
    every test in this file.

    ``isolated_job_stores`` is requested because every test here drives
    ``/api/analyze`` (or its streaming sibling), which since T2.3 writes a job
    row and an ``original`` object through process-wide store singletons. Without
    it those land in the developer's own database and a re-run can deduplicate to
    the previous run's job — whose single-use key is already spent — so the
    response stops being the analysis this file asserts on.
    """
    RecordingAnalyzer.reset()
    for module in (
        "paperwhisperer.analysis.orchestrator",
        "paperwhisperer.analysis.service",
        "paperwhisperer.api.routes_documents",
        "paperwhisperer.api.routes_qa",
        "paperwhisperer.api.routes_papers",
    ):
        monkeypatch.setattr(f"{module}.DocumentAnalyzer", RecordingAnalyzer)
    _stub_fact_layer_seams(monkeypatch)
    yield RecordingAnalyzer
    RecordingAnalyzer.reset()


def upload_bytes(name="paper.txt", body=b"plain text body"):
    """Build the ``files=`` mapping for a multipart upload."""
    return {"file": (name, io.BytesIO(body), "text/plain")}


def form(**values):
    """Build the multipart ``data=`` mapping, dropping unset values."""
    return {key: str(value) for key, value in values.items() if value is not None}


#: A caller-supplied API key that a real browser would send.
#:
#: Since T2.3 the analysis runs inside a job, and a caller-supplied key is only
#: parked for that job's stage when it is at least eight characters
#: (:func:`paperwhisperer.api.routes_jobs.register_ephemeral_secret` — a short
#: "key" is a browser default, not a credential). A one-character placeholder
#: is therefore not a key the stage will accept, and the job answers
#: ``missing api key``. That is correct product behaviour, not something these
#: tests should paper over, so the placeholder is a realistic length here and
#: every assertion below is unchanged.
BROWSER_KEY = "sk-test-browser-key"


@pytest.fixture
def analysis_client(client, analyzer, runtime_dirs):
    """A ``TestClient`` with a stubbed analyzer and isolated storage."""
    return client


class TestAnalyzeRoute:
    def test_a_valid_upload_returns_the_analysis_result(self, analysis_client):
        response = analysis_client.post(
            "/api/analyze", files=upload_bytes(), data=form(api_key=BROWSER_KEY)
        )
        assert response.status_code == 200
        assert response.json()["summary"] == "summary 内容"

    def test_no_file_is_a_client_error(self, analysis_client):
        response = analysis_client.post("/api/analyze", data=form(api_key=BROWSER_KEY))
        assert response.status_code == 400
        assert response.json()["code"] == "missing_file"

    def test_an_empty_filename_is_rejected(self, analysis_client):
        response = analysis_client.post(
            "/api/analyze", files={"file": ("", io.BytesIO(b"x"), "text/plain")},
            data=form(api_key=BROWSER_KEY),
        )
        # The request is turned away before the route body runs: with no
        # filename the multipart parser hands FastAPI a plain string, so
        # parameter validation rejects the body with 422. What matters is
        # that a nameless upload can never reach the analyzer, not which
        # layer says so. No stub was constructed at all, so ``instances``
        # is empty.
        assert response.status_code in {400, 422}
        assert RecordingAnalyzer.instances == []

    def test_an_unsupported_extension_names_the_supported_ones(self, analysis_client):
        response = analysis_client.post(
            "/api/analyze", files=upload_bytes("malware.exe", b"MZ"),
            data=form(api_key=BROWSER_KEY),
        )
        assert response.status_code == 400
        assert response.json()["code"] == "unsupported_file_type"

    def test_an_oversized_upload_is_refused_and_leaves_nothing_behind(self, analysis_client, runtime_dirs):
        oversized = b"a" * (config.MAX_CONTENT_LENGTH + 1)
        response = analysis_client.post(
            "/api/analyze", files=upload_bytes("big.txt", oversized), data=form(api_key=BROWSER_KEY)
        )
        assert response.status_code in {400, 413}
        assert list(runtime_dirs["uploads"].iterdir()) == []

    def test_the_uploaded_file_is_deleted_after_analysis(self, analysis_client, runtime_dirs):
        analysis_client.post("/api/analyze", files=upload_bytes(), data=form(api_key=BROWSER_KEY))
        assert list(runtime_dirs["uploads"].iterdir()) == []

    @pytest.mark.parametrize(
        ("field", "position"),
        [("generate_mermaid", 2), ("generate_evaluation", 3), ("generate_research_brief", 4)],
    )
    def test_optional_section_switches_reach_the_analyzer(
        self, analysis_client, analyzer, field, position
    ):
        analysis_client.post(
            "/api/analyze", files=upload_bytes(), data=form(api_key=BROWSER_KEY, **{field: "false"})
        )
        call = analyzer.last().calls[0]
        # The analyser runs inside the job's analysing stage, which drives the
        # fact-layer entry (``analyze_with_fact_layer``) rather than the legacy
        # ``analyze_stream`` generator. The switch positions are the same on both
        # methods, which is what this test is pinning: a switch the reader turned
        # off has to arrive ``False`` at the position the seam reads it from.
        assert call[0] == "analyze_with_fact_layer"
        assert call[position] is False

    def test_the_switches_default_to_on(self, analysis_client, analyzer):
        analysis_client.post("/api/analyze", files=upload_bytes(), data=form(api_key=BROWSER_KEY))
        assert analyzer.last().calls[0][2:5] == (True, True, True)

    def test_a_hostile_session_id_is_sanitised_before_use(self, analysis_client, runtime_dirs):
        response = analysis_client.post(
            "/api/analyze", files=upload_bytes(), data=form(api_key=BROWSER_KEY, session_id="../../etc")
        )
        assert response.status_code == 200
        assert not (runtime_dirs["context"].parent / "etc").exists()

    def test_a_model_failure_is_reported_as_a_server_error(self, client, analyzer, runtime_dirs,
                                                           monkeypatch):
        def explode(*args, **kwargs):
            raise RuntimeError("模型不可用")

        # The analysis runs inside the job's analysing stage and reaches the model
        # through the fact-layer entry, so a model failure is injected at that
        # seam and surfaces as a *failed job* rather than an exception in the
        # request. The route's answer for a failed job is 422
        # ``document_analysis_failed`` — see ``routes_documents.analyze``. What
        # this test guards is unchanged by the migration: the failure is reported
        # (never a 200), it carries the analysis-failure code, and the staging
        # upload is still cleaned up.
        monkeypatch.setattr(
            "paperwhisperer.analysis.service.analyze_with_fact_layer", explode
        )
        response = client.post("/api/analyze", files=upload_bytes(), data=form(api_key=BROWSER_KEY))
        assert response.status_code == 422
        assert response.json()["code"] == "document_analysis_failed"
        assert list(runtime_dirs["uploads"].iterdir()) == []

    def test_a_value_error_is_reported_as_a_client_error(self, client, analyzer, runtime_dirs,
                                                         monkeypatch):
        def bad(*args, **kwargs):
            raise ValueError("文件内容无法解析")

        # The upload-save phase is the one that still runs inside the request,
        # and it is still where a ValueError means "the client's upload is
        # unusable" — hence 400 ``invalid_request``. This keeps the original
        # client-error contract intact and asserts it against a seam the route
        # actually executes (``save_upload_file`` is called directly here).
        monkeypatch.setattr("paperwhisperer.api.routes_documents.save_upload_file", bad)
        response = client.post("/api/analyze", files=upload_bytes(), data=form(api_key=BROWSER_KEY))
        assert response.status_code == 400
        assert response.json()["code"] == "invalid_request"


class TestAnalyzeStreamRoute:
    def test_the_response_is_an_event_stream(self, analysis_client):
        response = analysis_client.post(
            "/api/analyze/stream", files=upload_bytes(), data=form(api_key=BROWSER_KEY)
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

    def test_the_stream_opens_with_a_session_and_finishes_with_a_result(
        self, analysis_client, sse_events
    ):
        response = analysis_client.post(
            "/api/analyze/stream", files=upload_bytes(), data=form(api_key=BROWSER_KEY)
        )
        events = sse_events(response.text)
        assert events[0][0] == "start"
        assert events[-1][0] == "done"
        assert events[-1][1]["summary"] == "summary 内容"

    def test_every_planned_section_arrives_as_its_own_event(self, analysis_client, sse_events):
        response = analysis_client.post(
            "/api/analyze/stream", files=upload_bytes(), data=form(api_key=BROWSER_KEY)
        )
        names = [p["name"] for name, p in sse_events(response.text) if name == "section"]
        assert sorted(names) == sorted(sections.SECTION_KEYS)

    def test_a_switched_off_section_arrives_as_a_disabled_event(self, analysis_client, sse_events):
        response = analysis_client.post(
            "/api/analyze/stream", files=upload_bytes(),
            data=form(api_key=BROWSER_KEY, generate_evaluation="false"),
        )
        events = sse_events(response.text)
        evaluation = [p["section"] for name, p in events
                      if name == "section" and p["name"] == "evaluation"]
        # A switched-off section now ARRIVES as an event with status ``disabled``
        # and no content — the legacy path simply omitted it. This is a strictly
        # stronger contract than the absence it replaced: the client is told the
        # card was turned off on purpose, rather than having to guess whether the
        # empty card means "failed" or "you chose this". It is exactly why
        # ``_result_from_outcome`` distinguishes ``disabled`` from ``empty`` — the
        # latter would offer the reader a follow-up about a card their own click
        # suppressed. The migrated stage publishes all sections in one shot, so
        # every key appears.
        assert len(evaluation) == 1
        assert evaluation[0]["status"] == "disabled"
        assert evaluation[0]["content"] == ""
        assert events[-1][0] == "done"

    def test_a_section_without_content_is_degraded_but_not_fatal(self, analysis_client, analyzer,
                                                              sse_events):
        # The fact layer has no per-section ``failed`` event any more: it extracts
        # the eight facts in one call and renders every card from them, so a card
        # it could not produce is ``empty`` (present, empty, not an error) rather
        # than ``failed``, and the whole run still finishes. This test pins that
        # real behaviour — the honest replacement for the old
        # ``sections_to_fail -> failed`` event. Making ``quotes`` come back empty
        # is the product's genuine way of saying "this section has no content",
        # and the contract under test is that such a run is *degraded, not
        # fatal*: every section still arrives, the stream still ends in ``done``,
        # and the reader gets the sections that did work.
        analyzer.sections_to_be_empty = {"quotes"}
        response = analysis_client.post(
            "/api/analyze/stream", files=upload_bytes(), data=form(api_key=BROWSER_KEY)
        )
        events = sse_events(response.text)
        by_name = {p["name"]: p["section"] for name, p in events if name == "section"}
        assert sorted(by_name) == sorted(sections.SECTION_KEYS)
        # The degraded section is reported as empty — present, no content, not an
        # error — and no section carries a ``failed`` status the product can no
        # longer emit.
        assert by_name["quotes"]["status"] == "empty"
        assert by_name["quotes"]["content"] == ""
        assert not [n for n, sec in by_name.items() if sec["status"] == "failed"]
        assert by_name["summary"]["status"] == "success"
        assert events[-1][0] == "done"

    def test_a_missing_api_key_is_refused_before_the_stream_opens(self, analysis_client):
        response = analysis_client.post("/api/analyze/stream", files=upload_bytes())
        assert response.status_code in {200, 400}
        if response.status_code == 400:
            assert response.json()["code"] == "missing_api_key"
        else:
            # An SSE body can only report the problem as an event.
            assert "error" in response.text

    def test_the_upload_is_removed_once_the_stream_finishes(self, analysis_client, runtime_dirs):
        analysis_client.post("/api/analyze/stream", files=upload_bytes(), data=form(api_key=BROWSER_KEY))
        assert list(runtime_dirs["uploads"].iterdir()) == []

    def test_a_producer_failure_becomes_an_error_event(self, client, analyzer, runtime_dirs,
                                                       monkeypatch):
        def explode(*args, **kwargs):
            raise RuntimeError("流式分析崩溃")

        # The stream is a view of a durable job, so a producer failure is now a
        # failed job rather than a dead generator. The seam that produces the
        # sections is the fact-layer entry the job stage reads at call time; the
        # route turns the failed job into an ``error`` event, which is the same
        # event vocabulary the browser already consumes.
        monkeypatch.setattr(
            "paperwhisperer.analysis.service.analyze_with_fact_layer", explode
        )
        response = client.post(
            "/api/analyze/stream", files=upload_bytes(), data=form(api_key=BROWSER_KEY)
        )
        assert "error" in response.text
        assert list(runtime_dirs["uploads"].iterdir()) == []

    def test_no_file_is_a_client_error(self, analysis_client):
        response = analysis_client.post("/api/analyze/stream", data=form(api_key=BROWSER_KEY))
        assert response.status_code == 400


class TestAskRoute:
    @pytest.fixture
    def session(self, session_factory, runtime_dirs, api_key_env):
        return session_factory(document_content="document text")

    def test_a_answered_question_returns_the_answer(self, client, analyzer, session):
        session_id, token = session
        response = client.post(
            "/api/ask",
            json={"question": "问题", "session_id": session_id, "session_token": token},
        )
        assert response.status_code == 200
        assert response.json()["answer"] == "answer body"

    def test_a_blank_question_is_refused(self, client, analyzer, session):
        session_id, token = session
        response = client.post(
            "/api/ask", json={"question": "   ", "session_id": session_id, "session_token": token}
        )
        assert response.status_code == 400
        assert response.json()["code"] == "missing_question"

    def test_a_missing_api_key_is_refused(self, client, analyzer, session, monkeypatch):
        # The session fixture supplies a key; this case is about its absence.
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        session_id, token = session
        response = client.post(
            "/api/ask", json={"question": "问题", "session_id": session_id, "session_token": token}
        )
        assert response.status_code == 400
        assert response.json()["code"] == "missing_api_key"

    def test_a_wrong_token_is_a_403_not_a_500(self, client, analyzer, session):
        session_id, _ = session
        response = client.post(
            "/api/ask", json={"question": "问题", "session_id": session_id, "session_token": "wrong"}
        )
        assert response.status_code == 403
        assert response.json()["code"] == "invalid_session_token"

    def test_an_unknown_session_is_a_client_error(self, client, analyzer, session):
        _, token = session
        response = client.post(
            "/api/ask", json={"question": "问题", "session_id": "nope", "session_token": token}
        )
        assert response.status_code == 400
        assert response.json()["code"] == "invalid_request"

    def test_an_unknown_answer_mode_falls_back_to_the_default(self, client, analyzer, session):
        session_id, token = session
        client.post(
            "/api/ask",
            json={"question": "问题", "session_id": session_id, "session_token": token,
                  "answer_mode": "not-a-mode"},
        )
        assert analyzer.last().calls[0][3] == prompts.DEFAULT_ANSWER_MODE

    @pytest.mark.parametrize("mode", sorted(prompts.ANSWER_MODES))
    def test_every_declared_answer_mode_survives_normalisation(
        self, client, analyzer, session, mode
    ):
        session_id, token = session
        client.post(
            "/api/ask",
            json={"question": "问题", "session_id": session_id, "session_token": token,
                  "answer_mode": mode},
        )
        assert analyzer.last().calls[0][3] == mode

    def test_a_missing_answer_mode_uses_the_default(self, client, analyzer, session):
        session_id, token = session
        client.post(
            "/api/ask",
            json={"question": "问题", "session_id": session_id, "session_token": token},
        )
        assert analyzer.last().calls[0][3] == prompts.DEFAULT_ANSWER_MODE

    def test_the_turn_is_recorded_in_the_session(self, client, analyzer, session, runtime_dirs):
        import json

        session_id, token = session
        client.post(
            "/api/ask", json={"question": "问题", "session_id": session_id, "session_token": token}
        )
        stored = json.loads((runtime_dirs["context"] / f"{session_id}.json").read_text("utf-8"))
        assert stored["qa_history"][-1]["question"] == "问题"
        assert stored["qa_history"][-1]["answer"] == "answer body"

    def test_prior_history_is_forwarded_to_the_analyzer(self, client, analyzer, session_factory, api_key_env):
        session_id, token = session_factory(
            qa_history=[{"question": "旧问题", "answer": "旧答案", "answer_mode": "evidence"}]
        )
        client.post(
            "/api/ask", json={"question": "新问题", "session_id": session_id, "session_token": token}
        )
        assert analyzer.last().calls[0][2] == [{"question": "旧问题", "answer": "旧答案", "answer_mode": "evidence"}]

    def test_a_malformed_body_is_refused(self, client, analyzer, session):
        session_id, token = session
        response = client.post(
            "/api/ask",
            content=b"{not json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 400
        assert response.json()["code"] == "invalid_json"


class TestAskStreamRoute:
    @pytest.fixture
    def session(self, session_factory, runtime_dirs, api_key_env):
        return session_factory(document_content="document text")

    def test_the_deltas_reassemble_into_the_final_answer(self, client, analyzer, session, sse_events):
        session_id, token = session
        response = client.post(
            "/api/ask/stream",
            json={"question": "问题", "session_id": session_id, "session_token": token},
        )
        events = sse_events(response.text)
        assert events[0][0] == "start"
        streamed = "".join(p["text"] for name, p in events if name == "delta")
        assert events[-1][0] == "done"
        assert streamed == events[-1][1]["answer"]
        assert streamed == "".join(analyzer.chunks)

    def test_a_failed_producer_becomes_an_error_event(self, client, analyzer, session, sse_events, monkeypatch):
        session_id, token = session

        def explode(api_key):
            raise RuntimeError("流式问答崩溃")
            yield  # pragma: no cover

        monkeypatch.setattr("paperwhisperer.api.routes_qa.DocumentAnalyzer", explode)
        response = client.post(
            "/api/ask/stream",
            json={"question": "问题", "session_id": session_id, "session_token": token},
        )
        assert sse_events(response.text)[-1][0] == "error"

    def test_the_turn_is_recorded_after_streaming(self, client, analyzer, session, runtime_dirs):
        import json

        session_id, token = session
        client.post(
            "/api/ask/stream",
            json={"question": "问题", "session_id": session_id, "session_token": token},
        )
        stored = json.loads((runtime_dirs["context"] / f"{session_id}.json").read_text("utf-8"))
        assert stored["qa_history"][-1]["answer"] == "".join(analyzer.chunks)

    def test_a_wrong_token_is_refused_before_the_stream_opens(self, client, analyzer, session):
        session_id, _ = session
        response = client.post(
            "/api/ask/stream", json={"question": "问题", "session_id": session_id, "session_token": "bad"}
        )
        assert response.status_code == 403
        assert response.json()["code"] == "invalid_session_token"



class TestEventLoopIsNotBlocked:
    """The SSE pump must keep the loop free while a producer blocks.

    Regression coverage for the fix that moved blocking LLM generators onto a
    worker thread.

    The measurement is taken **from inside the producer**: it records how many
    ticks a free-running coroutine managed to accumulate while it was blocked.
    That number can only be positive if the producer was on a worker thread
    and the loop stayed available to run other work. Were the producer
    iterated on the loop thread, the ticker could not be scheduled at all
    during the block and the recorded value would be exactly zero.
    """

    #: Ticks the producer waits for. Small on purpose: a healthy run satisfies
    #: it almost immediately, so only a regression pays the full timeout.
    TICKS_REQUIRED = 5
    BLOCK_TIMEOUT_SECONDS = 5

    def test_the_loop_keeps_running_while_the_producer_blocks(self):
        async def scenario():
            started = threading.Event()
            ticks: list[int] = []
            observed: list[int] = []

            def produce():
                started.set()
                deadline = time.monotonic() + self.BLOCK_TIMEOUT_SECONDS
                while len(ticks) < self.TICKS_REQUIRED and time.monotonic() < deadline:
                    time.sleep(0.001)
                observed.append(len(ticks))
                yield {"type": "delta", "data": {"text": "x"}}

            async def ticker():
                while True:
                    ticks.append(1)
                    await asyncio.sleep(0)

            ticker_task = asyncio.create_task(ticker())
            try:
                consumer = asyncio.create_task(
                    drain(pump_sync_events(produce, lambda e: e))
                )
                assert await asyncio.to_thread(started.wait, 5), "producer never started"
                await asyncio.wait_for(consumer, timeout=15)
            finally:
                ticker_task.cancel()
            return observed

        observed = asyncio.run(scenario())
        # ``>=`` rather than ``==``: the ticker is a free-running loop, so a
        # healthy run overshoots the threshold. Zero is the only real failure
        # signal — that means the ticker was never scheduled at all.
        assert observed and observed[0] >= self.TICKS_REQUIRED, (
            f"producer saw {observed} loop ticks while blocking, "
            f"expected at least {self.TICKS_REQUIRED} — the event loop was blocked"
        )

    def test_cleanup_runs_even_when_the_consumer_stops_early(self):
        async def scenario():
            cleaned = []
            release = threading.Event()

            def produce():
                yield {"type": "delta", "data": {"text": "a"}}
                release.wait(timeout=5)
                yield {"type": "delta", "data": {"text": "b"}}

            stream = pump_sync_events(produce, lambda e: e, cleanup=lambda: cleaned.append(1))
            first = await stream.__anext__()
            assert first == {"type": "delta", "data": {"text": "a"}}
            await stream.aclose()
            return cleaned

        assert asyncio.run(scenario()) == [1]









