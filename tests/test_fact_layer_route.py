"""The analysis route now runs the fact layer, and the visible fields survive.

``/api/analyze`` reaches its analysis through
:func:`paperwhisperer.analysis.service.analyze_with_fact_layer` (called by the
job stage :func:`paperwhisperer.api.routes_jobs._analyse_stage`) rather than
through the old six-section generator. That is the whole point of the migration:
the fact layer is the only path that records ``paper_facts`` / ``analysis_run``,
the provenance the export report promises.

Swapping an analysis path is exactly the kind of change that is invisible until
the interface goes blank, so these tests pin the *result contract* the front end
depends on, driven with a stub model and a real (tiny) parse. They are the
hermetic guard: the e2e suite proves it against a live model, and these prove it
does not regress between runs. **No live model, no endpoint** — the analyzer is
replaced by a stub whose client answers from a fixed payload.

Two things are pinned here, and they are the two halves of the migration:

* the fields the interface reads are still present *and populated* —
  ``next_actions``, six ``suggested_questions``, ``elapsed_seconds`` — because
  the fact layer's own result shape returns an empty action list and four
  questions, and the migration ports the old path's derivation rather than
  leaving the reader with nothing to click;
* the three section switches are honoured, because the fact layer's pipeline
  used to render the Mermaid and research-brief cards unconditionally even when
  the caller had switched them off.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from paperwhisperer.analysis import paper_facts, sections, service
from paperwhisperer.core import config
from paperwhisperer.documents.loader import DocumentLoader
from paperwhisperer.ingestion.schema import Evidence, SourceLocator

# -- a stub model that answers the two prompts the fact layer sends ---------


def _extraction_payload() -> str:
    return json.dumps(
        {
            "research_question": "Does graph-structured retrieval help RAG?",
            "contributions": "A graph index that routes queries to subgraphs.",
            "method": "We build a graph-structured index and route a query.",
            "datasets": "ImageNet and WebQuestions.",
            "metrics": "F1 and exact-match accuracy.",
            "main_results": "Improves exact-match accuracy by 6.3 points.",
            "limitations": "We do not study multilingual settings.",
            "reproducibility": "Code is released under a permissive licence.",
            "quotes": [
                {
                    "field": "method",
                    "text": "We build a graph-structured index",
                    "quote": "We build a graph-structured index",
                }
            ],
        },
        ensure_ascii=False,
    )


def _appraisal_payload() -> str:
    return json.dumps({"strengths": ["局部隔离检索误差"], "weaknesses": ["未做同预算对比"]},
                      ensure_ascii=False)


class StubLLMClient:
    """A ``client`` stand-in the analyzer hands to the pipeline. No sockets."""

    def __init__(self) -> None:
        self.complete_calls: list[str] = []
        self.model = "stub-model"

    def complete(self, system_prompt, user_prompt, *args, **kwargs) -> str:
        self.complete_calls.append(user_prompt)
        if "critical_appraisal_evidence" in user_prompt:
            return _appraisal_payload()
        return _extraction_payload()


class StubAnalyzer:
    """The ``DocumentAnalyzer`` :func:`analyze_with_fact_layer` constructs.

    It carries just enough surface for the fact-layer entry point: a ``client``
    (with a ``complete`` the pipeline can call and a ``model`` name for the run
    record) and a ``version`` the report header reads. Every attribute the
    function *sets* (``document_content`` / ``parse_outcome``) is a plain
    attribute, so the "adopt the caller's parse" path is exercised for real.
    """

    instances: list[StubAnalyzer] = []

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.client = StubLLMClient()
        self.version = "stub-1.0"
        self.document_content = ""
        self.parse_outcome = None
        StubAnalyzer.instances.append(self)


def _provider(evidence):
    """A provider returning the same real evidence pool for every field."""

    def provider(field_name, document_id):
        return tuple(evidence), (f"rr_{field_name}",)

    return provider


def _evidence_pool() -> tuple[Evidence, ...]:
    texts = {
        "ev_intro": "Retrieval-augmented generation prepends the query to the document.",
        "ev_method": "We build a graph-structured index and route a query to a subgraph.",
        "ev_results": "RAG-Fusion improves exact-match accuracy on NQ by 6.3 points.",
        "ev_limits": "We do not study multilingual settings and routing doubles latency.",
        "ev_code": "Code is released under a permissive licence with all hyperparameters.",
    }
    return tuple(
        Evidence(
            evidence_id=evidence_id,
            chunk_id=f"chunk_{evidence_id}",
            text=text,
            locator=SourceLocator(kind="pdf_page", page=index + 1),
            heading_path=("Introduction",),
        )
        for index, (evidence_id, text) in enumerate(texts.items())
    )


@pytest.fixture
def stub_analyzer(monkeypatch):
    """Replace the analyzer the fact-layer entry point constructs."""
    StubAnalyzer.instances = []
    monkeypatch.setattr(service, "DocumentAnalyzer", StubAnalyzer)
    return StubAnalyzer


@pytest.fixture
def benign_registration(monkeypatch):
    """Skip the document registration: it is covered by the citation suites.

    :func:`finalize_analysis_result` calls
    :func:`register_analysis_document` to index the document for the Evidence
    API. That path writes vectors and is the subject of
    ``tests/test_citation_click.py``; stubbing it here keeps these tests
    hermetic and single-purpose, without hiding the field the route reads.
    """

    def fake_register(result, parse_outcome):
        result["document_registration"] = {
            "status": "registered",
            "document_id": str(getattr(parse_outcome.document, "document_id", "") or ""),
            "chunk_count": 3,
            "index_version": "iv-test",
            "code": "",
            "notice": "",
        }
        return result

    monkeypatch.setattr(service, "register_analysis_document", fake_register)


#: The body the import test serves over the fake download, and the body the
#: parse fixture writes: the same paper, so the two routes are analysing the
#: same bytes.
PAPER_TEXT = (
    "Graph-Structured Retrieval\n\n"
    "Introduction\n\n"
    "Retrieval-augmented generation prepends the query to the document.\n\n"
    "Method\n\n"
    "We build a graph-structured index and route a query to a subgraph.\n"
    "The method improves exact-match accuracy on NQ by 6.3 points.\n"
    "We do not study multilingual settings. Code is released under a licence.\n"
)


@pytest.fixture
def paper(tmp_path):
    """A real, tiny document — parsed for real, so the identity path is real."""
    path = tmp_path / "fact_layer_paper.txt"
    path.write_text(PAPER_TEXT, encoding="utf-8")
    return path


@pytest.fixture
def loaded(paper):
    """The one parse every caller here shares — one identity, as in production."""
    return DocumentLoader.load_with_outcome(str(paper))


def _run(loaded, *, mermaid=True, evaluation=True, brief=True):
    return service.analyze_with_fact_layer(
        "fact_layer_paper.txt",
        "fact_layer_paper.txt",
        "stub-key",
        mermaid,
        evaluation,
        "session-under-test",
        brief,
        document_id=loaded.outcome.document.document_id,
        evidence_provider=_provider(_evidence_pool()),
        loaded_document=loaded,
    )


# -- the fields the interface reads -----------------------------------------


class TestThePortedFields:
    """The visible contract the swap must not blank out."""

    def test_the_fact_layer_route_still_offers_next_actions(
        self, stub_analyzer, benign_registration, loaded
    ):
        result = _run(loaded)
        # The old path produced a populated action list; the fact layer's own
        # result shape returns []. The migration ports the derivation, so this
        # must be non-empty — a reader with nothing to click is the regression
        # this test exists to prevent.
        assert result["next_actions"], "the fact-layer route returned no next actions"
        for action in result["next_actions"]:
            assert action.get("label") and action.get("prompt")

    def test_the_fact_layer_route_still_suggests_six_questions(
        self, stub_analyzer, benign_registration, loaded
    ):
        result = _run(loaded)
        # The old path produced six; the fact layer hardcoded four. Pinned to
        # six so the port is visible as a number, not just "non-empty".
        assert len(result["suggested_questions"]) == 6

    def test_the_fact_layer_route_records_elapsed_seconds(
        self, stub_analyzer, benign_registration, loaded
    ):
        result = _run(loaded)
        # Present and a number. The fact layer measures no wall-clock, so the
        # route stamps it; without it the interface renders "N/A".
        assert isinstance(result["elapsed_seconds"], (int, float))
        assert result["elapsed_seconds"] >= 0.0

    def test_the_other_keys_the_interface_reads_are_still_present(
        self, stub_analyzer, benign_registration, loaded
    ):
        result = _run(loaded)
        for key in ("summary", "sections", "char_count", "session_id", "document_id",
                    "parse"):
            assert key in result, f"the front end reads {key!r} and it is gone"
        assert result["summary"]
        assert result["sections"]["summary"]["content"] == result["summary"]
        # The provenance this migration exists to produce.
        assert result.get("paper_facts"), "the fact layer recorded no paper_facts"
        assert result.get("analysis_run"), "the fact layer recorded no analysis_run"


# -- the three section switches --------------------------------------------


class TestTheThreeFlagsAreHonoured:
    """The fact layer used to render Mermaid and the brief unconditionally."""

    def test_switching_everything_off_renders_no_those_cards(
        self, stub_analyzer, benign_registration, loaded
    ):
        result = _run(loaded, mermaid=False, evaluation=False, brief=False)
        assert result["mermaid"] == ""
        assert result["evaluation"] == ""
        assert result["research_brief"] == ""
        statuses = result["sections"]
        assert statuses["mermaid"]["status"] == "disabled"
        assert statuses["evaluation"]["status"] == "disabled"
        assert statuses["research_brief"]["status"] == "disabled"

    def test_switching_the_brief_off_does_not_offer_to_complete_it(
        self, stub_analyzer, benign_registration, loaded
    ):
        # A section the reader switched off is "disabled", not "empty"; the old
        # derivation must not then offer an action about filling in a gap the
        # reader closed themselves.
        result = _run(loaded, mermaid=False, evaluation=False, brief=False)
        labels = {action["label"] for action in result["next_actions"]}
        assert "复现路线" not in labels
        assert "找后续工作" not in labels

    def test_switching_everything_on_still_renders_them(
        self, stub_analyzer, benign_registration, loaded
    ):
        result = _run(loaded)
        assert result["mermaid"], "the Mermaid card was dropped even though it was asked for"
        assert result["research_brief"], "the brief was dropped even though it was asked for"
        assert result["sections"]["mermaid"]["status"] == "success"

    def test_the_evaluation_flag_gates_the_second_model_call(
        self, stub_analyzer, benign_registration, loaded
    ):
        # evaluation is the one switch that also gates a model call, so it is
        # the one whose cost must actually change.
        StubAnalyzer.instances.clear()
        _run(loaded, evaluation=False)
        without = len(StubAnalyzer.instances[-1].client.complete_calls)
        _run(loaded, evaluation=True)
        with_eval = len(StubAnalyzer.instances[-1].client.complete_calls)
        assert with_eval > without, "the appraisal call ran even with evaluation off"


# -- the pipeline itself, isolated from the result shape --------------------


class TestThePipelineHonoursItsFlags:
    """The same three switches at the layer that renders the cards."""

    @pytest.fixture
    def outcome(self, stub_analyzer):
        client = StubLLMClient()
        pipeline = paper_facts.FactLayerPipeline(
            complete=client.complete,
            evidence_provider=_provider(_evidence_pool()),
            model="stub-model",
            document_id="doc_stub",
        )
        from paperwhisperer.ingestion.schema import Element, ParsedDocument

        document = ParsedDocument(
            document_id="doc_stub",
            source_filename="paper.pdf",
            page_count=3,
            elements=[
                Element(role="heading", text="1 Introduction", order=0, level=1),
                Element(role="heading", text="2 Method", order=1, level=1),
                Element(role="heading", text="3 Results", order=2, level=1),
            ],
        )
        return pipeline.run(
            document=document,
            generate_evaluation=True,
            generate_mermaid=False,
            generate_research_brief=False,
        )

    def test_the_cards_the_flags_switched_off_are_empty(self, outcome):
        assert outcome.cards["mermaid"] == ""
        assert outcome.cards["research_brief"] == ""
        # The one that was not switched off is still rendered.
        assert outcome.cards["summary"]

    def test_the_section_keys_are_still_all_present(self, outcome):
        # Every key is present (empty when off) so the result shape the API
        # returns does not depend on which switches the caller passed.
        for key in sections.SECTION_KEYS:
            assert key in outcome.cards


def test_analyze_with_fact_layer_still_refuses_without_an_evidence_provider(
    paper, stub_analyzer, monkeypatch
):
    # Unchanged by the migration: a fact layer with no citations is refused
    # rather than guessed, and the loaded document does not change that.
    monkeypatch.setattr(config, "OUTPUT_FOLDER", str(paper.parent))
    loaded = DocumentLoader.load_with_outcome(str(paper))
    with pytest.raises(ValueError, match="evidence_provider"):
        service.analyze_with_fact_layer(
            "paper.txt", "paper.txt", "stub-key", True, True, "sid",
            loaded_document=loaded,
        )

# -- the second way a paper reaches the fact layer ---------------------------


class TestImportPaperRecordsTheSameProvenance:
    """``/api/import-paper`` is a second front door to the same analysis.

    Two routes that both claim to "analyse this paper" must not produce
    different quality. Importing by URL used to call the legacy six-section
    generator, which records no run and no facts — so the provenance the export
    report promises was silently absent for every imported paper, with no error
    to notice it by.

    What is stubbed here is the *network*, not the behaviour. The download is
    served by the existing ``fake_response`` pattern, the retriever and the
    evidence provider are replaced because the real ones embed (and the local
    model is not cached offline), and ``register_analysis_document`` is a no-op
    because it indexes for the citation API. The parse, the fact layer, and the
    **real** ``finalize_analysis_result`` all run for real, which is what makes
    the 200 and the two provenance fields on the response worth asserting.
    """

    @pytest.fixture
    def import_client(
        self, monkeypatch, client, runtime_dirs, isolated_job_stores,
        public_example_urls, patch_remote_response, fake_response,
        stub_analyzer, benign_registration,
    ):
        """A ``TestClient`` whose import route can run the fact layer offline."""
        from paperwhisperer.analysis import orchestrator as analysis_orchestrator

        # The two retrieval collaborators, imported by the route at call time
        # from the module it names, so patching the module attribute is the
        # seam. The evidence provider answers from the real ``Evidence`` pool
        # the rest of this file uses, so the citations the run records address
        # something.
        monkeypatch.setattr(
            analysis_orchestrator, "build_qa_retriever",
            lambda bundle, document=None: SimpleNamespace(indexed=document),
        )
        monkeypatch.setattr(
            paper_facts, "make_evidence_provider",
            lambda retriever, document_id, **kwargs: _provider(_evidence_pool()),
        )

        # The download itself stays real (``download_remote_paper`` writes the
        # bytes into the redirected upload folder and enforces the byte
        # ceiling); only ``urlopen`` is replaced.
        patch_remote_response(
            fake_response(
                PAPER_TEXT.encode("utf-8"),
                content_type="text/plain",
                # ``stream_remote_paper`` names the local file after the *final*
                # URL (its redirect target), so the fake's ``geturl`` — not the
                # ``pdf_url`` in the request body — is what decides the download
                # is a ``.txt`` and not a ``.pdf`` the text does not match.
                url="https://example.com/paper.txt",
            )
        )
        return client

    def test_an_imported_paper_answers_with_its_facts_and_its_run(self, import_client):
        response = import_client.post(
            "/api/import-paper",
            json={
                "title": "Graph-Structured Retrieval",
                "pdf_url": "https://example.com/paper.txt",
                "api_key": "sk-test-import-key",
                "session_id": "import-session",
            },
        )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body.get("paper_facts"), "an imported paper recorded no paper_facts"
        assert body.get("analysis_run"), "an imported paper recorded no analysis_run"
        # Not just present: the run must name the document the parse produced.
        # A migration that let the route re-parse inside the analysis would
        # index one identity and quote against another, and this is where that
        # would show up.
        assert body["analysis_run"]["document_id"] == body["document_id"]
        assert body["document_id"]

    def test_the_import_still_answers_with_the_fields_the_reader_reads(self, import_client):
        # The provenance arrived at the cost of nothing else: the session and
        # the six sections are still real, because the real finaliser ran.
        response = import_client.post(
            "/api/import-paper",
            json={
                "title": "Graph-Structured Retrieval",
                "pdf_url": "https://example.com/paper.txt",
                "api_key": "sk-test-import-key",
                "session_id": "import-session",
            },
        )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["summary"]
        assert body["session_id"] == "import-session"
        assert body["suggested_questions"]
        for key in sections.SECTION_KEYS:
            assert key in body["sections"]
