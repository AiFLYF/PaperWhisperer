"""The contracts module, wired into the boundaries it was written for.

``tests/test_ingestion_contracts.py`` proves the Pydantic models do what their
docstrings claim. It cannot prove they are *used*: every one of its113 tests
would still pass with the entire module deleted, which is precisely the state
this file was written to end. That is not a hypothetical — the module landed
complete, fully tested, and imported by nothing, and 113 green tests were the
only symptom.

So everything here is about the seam rather than the models. Three questions,
and one guard:

* does a malformed adapter result become a *diagnosable* gateway failure,
  naming the field, instead of a document that cites the wrong page — or
  instead of a crash that takes down the request?
* is a malformed upload rejected at the edge, with a message naming the field,
  before the file is ever stored?
* does a valid document still pass through the whole wired path unchanged?

The fourth question is the one that keeps the other three honest: a set of
tests that would all still pass if the wiring were removed. Every wiring test
below is therefore written so that deleting the call makes it *fail* — by
spying on the contract to prove it was called at all, and by asserting on the
*rejection* rather than only on the acceptance.

The schema fixes that these tests also cover — the malformed
``schema_version`` and the ``relative`` bbox flag — are asserted here at their
boundaries because that is where they have to hold. Their unit behaviour
belongs to ``test_ingestion_schema.py``, which does not own them.
"""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from paperwhisperer.analysis import orchestrator as analysis_orchestrator
from paperwhisperer.core import config
from paperwhisperer.ingestion.contracts import (
    BoundingBoxContract,
    ParsedDocumentContract,
    SourceLocatorContract,
    UploadRequestContract,
)
from paperwhisperer.ingestion.gateway import (
    PARSE_FAILED,
    PARSE_SUCCESS,
    ParseOptions,
    ParseResult,
    ParserRouter,
)
from paperwhisperer.ingestion.schema import (
    SCHEMA_VERSION,
    BoundingBox,
    Element,
    ParsedDocument,
    SourceLocator,
    Table,
)

PDF_MEDIA_TYPE = "application/pdf"
TXT_MEDIA_TYPE = "text/plain"


class ImmediateExecutor:
    """Runs the submitted callable inline; the timeout path is not under test."""

    def submit(self, fn, /, *args, **kwargs):
        class _Done:
            @staticmethod
            def result(timeout=None):
                return fn(*args, **kwargs)

            @staticmethod
            def cancel():
                return False

        return _Done()

    def shutdown(self, wait=True, *, cancel_futures=False):
        return None


class FakeParser:
    """A parser whose every observable the test sets."""

    def __init__(self, name, *, version="1.0", outcome=None, media_types=(PDF_MEDIA_TYPE,)):
        self.name = name
        self.version = version
        self.outcome = outcome
        self.media_types = tuple(media_types)
        self.calls: list = []

    def supports(self, media_type):
        self.calls.append(("supports", media_type))
        return media_type in self.media_types

    def parse(self, file_path, options):
        self.calls.append(("parse", file_path))
        if callable(self.outcome):
            return self.outcome(file_path, options)
        return self.outcome


def good_document(**overrides) -> ParsedDocument:
    """A document that satisfies every contract, built the way an adapter would.

    "Every contract" includes the quality gate (T0.6), because since T0.7 the
    gateway runs it over every usable parse and the contract layer's promise —
    a valid document comes back unchanged — is only meaningful for a document
    the whole pipeline accepts. A one-sentence, one-page fixture used to sit
    *below* the density rule's floor, so it described a scan that never produced
    a text layer rather than a good document, and the tests below were passing
    for a reason that had nothing to do with what they claimed to pin. The
    second paragraph is what keeps this fixture on the right side of that
    threshold; it carries no assertion of its own.
    """
    document = ParsedDocument(
        source_filename="paper.pdf",
        media_type=PDF_MEDIA_TYPE,
        sha256="abc123",
        page_count=1,
        parser_name="legacy",
        parser_version="1.0",
        elements=[
            Element(
                role="paragraph",
                text="A resolvable sentence on page 1.",
                locator=SourceLocator(kind="pdf_page", page=1),
            ),
            Element(
                role="paragraph",
                text=(
                    "The rest of the page carries ordinary prose so that the density "
                    "gate sees a page a reader could actually work from, rather than "
                    "the sparse output a failed text extraction leaves behind. Nothing "
                    "downstream reads this paragraph: it exists so that the document "
                    "as a whole clears every quality rule, and the only thing under "
                    "test in the contract cases is what the router does to a document "
                    "that was already good when it arrived."
                ),
                locator=SourceLocator(kind="pdf_page", page=1),
            ),
            Element(
                role="table",
                table=Table(rows=[["Metric", "Score"], ["BLEU", "27.3"]], header_rows=1),
                locator=SourceLocator(
                    kind="pdf_page", page=1, bbox=BoundingBox(72.0, 640.0, 300.0, 700.0)
                ),
            ),
        ],
    )
    for key, value in overrides.items():
        setattr(document, key, value)
    return document


def _element(document: ParsedDocument, text: str) -> Element:
    """The element carrying ``text``.

    Located by content rather than by index on purpose. A ``pdf_page`` with no
    page number sorts to the *front* of the document — ``sort_key`` reads a
    missing page as ``0`` — so an index-based assertion here would be pinning
    an accident of ordering that has nothing to do with the repair.
    """
    for element in document.elements:
        if element.text == text:
            return element
    raise AssertionError(f"no element with text {text!r}")


def good_result(**overrides) -> ParseResult:
    document = overrides.pop("document", None) or good_document()
    return ParseResult.from_document(document, **overrides)


def router_for(*parsers) -> ParserRouter:
    return ParserRouter(list(parsers), executor_factory=ImmediateExecutor)


def _install_stub(monkeypatch, routes, fake_analyze):
    """Neutralise the whole post-validation analysis pipeline for one test.

    Since T2.3 ``/api/analyze`` does not call ``analyze_saved_file`` in the
    route: it submits a durable job, and the job's stages
    (:mod:`paperwhisperer.api.routes_jobs`) import their collaborators **at call
    time**. Patching ``analyze_saved_file`` on the route module therefore
    intercepted nothing — the real parser and the real analyzer both ran, and
    the request made a live call to the configured model endpoint.

    Two stubs are installed here, mirroring
    ``tests/test_jobs_api.py::stubbed_pipeline``:

    * the parse stage reads ``parse_document`` from
      :mod:`paperwhisperer.ingestion.orchestrator`;
    * the analysing stage is handed the four seams listed in
      :func:`_stub_fact_layer_seams` below.

    ``DocumentAnalyzer`` is still replaced on the module the stage used to read
    it from. Nothing on the ``/api/analyze`` path constructs one any more, so
    this is belt-and-braces rather than load-bearing — but it is what the
    tripwire in
    :meth:`TestMalformedUploadIsRejectedAtTheEdge.test_a_rejected_upload_never_reaches_the_analyzer`
    patches over, and removing it would turn that test into a no-op rather than
    into a different test.

    These tests are about the upload edge and the contract wiring, not about
    the parser, the embedder or the model; all three were always stubbed (as one
    ``analyze_saved_file``) and are stubbed again, here at the seams that are
    actually executed.
    """
    from paperwhisperer.ingestion import orchestrator as ingestion_orchestrator
    from paperwhisperer.ingestion.schema import ParsedDocument

    def fake_parse(file_path, media_type=None, **kwargs):
        return _StubParseOutcome(
            document=ParsedDocument(
                source_filename=str(file_path),
                media_type=str(media_type or "text/plain"),
                elements=[],
            )
        )

    monkeypatch.setattr(ingestion_orchestrator, "parse_document", fake_parse)
    monkeypatch.setattr(routes, "DocumentAnalyzer", _StubAnalyzer)
    _stub_fact_layer_seams(monkeypatch)


def _stub_fact_layer_seams(monkeypatch):
    """Drive the migrated ``/api/analyze`` analysing stage through stubs.

    The analysing stage (:func:`paperwhisperer.api.routes_jobs._analyse_stage`)
    was moved off the legacy six-section generator
    (``DocumentAnalyzer.analyze_stream``) onto the fact layer, and it now runs
    four collaborators *before* it produces any content, each imported at call
    time from the module it names:

    1. ``DocumentLoader.load_with_outcome`` — the load, and the one parse whose
       ``document_id`` the rest of the stage must reuse;
    2. ``orchestrator.build_qa_retriever`` — the pre-index the fact layer
       refuses to run without;
    3. ``paper_facts.make_evidence_provider`` — the per-field citations wrapper;
    4. ``service.analyze_with_fact_layer`` — the analysis itself.

    Patching only ``DocumentAnalyzer`` is no longer enough, and the symptom it
    produced was a *misleading* one rather than a loud one: with the real
    ``analyze_with_fact_layer`` in place it builds a real ``DocumentAnalyzer``
    and runs the real DAG, and step 2 builds a dense retriever — an embedding
    pass that, with no local model cached, fails the job, and the route answers
    ``422`` for an upload these tests assert is a ``200``. A 422 that is really
    an embedding download is exactly the kind of failure a test that only reads
    a status code will happily encode as "the contract refuses this".

    **The last stub is deliberately shallow, and deliberately calls the real
    finaliser.** It builds the same minimal analysis body the legacy
    ``_StubAnalyzer.analyze_stream`` used to yield — the ``summary`` / ``char_count``
    / ``sections`` these edge tests assert on — and then hands off to the *real*
    :func:`paperwhisperer.analysis.service.finalize_analysis_result`, because
    that is where the product's real behaviour lives for everything these tests
    are about: the session id is sanitised, the session token is minted, the
    session is persisted, the Markdown report is written, and the result gains
    the 200-vs-422 semantics the route answers with. Stubbing the finaliser too
    would turn these into tests of a fake.

    ``register_analysis_document`` is the one thing inside the finaliser that
    reaches outside the process (it indexes the document for the citation API,
    which embeds), and none of these tests are about citations — so it is
    replaced with a no-op reporting the same ``registered`` shape the real one
    would, keeping the run hermetic and offline. Everything after it is real.
    """
    from paperwhisperer.analysis import orchestrator as analysis_orchestrator
    from paperwhisperer.analysis import paper_facts as analysis_paper_facts
    from paperwhisperer.analysis import service as analysis_service
    from paperwhisperer.documents.loader import DocumentLoader, LoadedDocument

    # (1) The load. A stub outcome rather than a real ``ParseOutcome`` because
    # the stage reads exactly three things off it — ``.document`` (for
    # ``document_id``), ``.error_message`` (only when there is no document) and
    # nothing else — and arranging a genuine parser to run would be a second
    # thing under test in a test about the HTTP route. Note the deliberate
    # mismatch with the *parse* stub installed by ``_install_stub``: the parse
    # stage goes through ``ingestion.orchestrator.parse_document`` and the
    # analysing stage does not, it goes through this loader. Two seams, one
    # stubbed load each, and leaving either real means a parser reading a
    # ``b"body"`` upload (which is what the PDF-alias test sends).
    def fake_load_with_outcome(file_path):
        return LoadedDocument(
            text="document text",
            outcome=SimpleNamespace(
                document=SimpleNamespace(
                    document_id="doc_stub",
                    page_count=1,
                    char_count=len("document text"),
                    source_filename=str(file_path),
                ),
                error_message="",
                used_cloud=False,
            ),
        )

    monkeypatch.setattr(
        DocumentLoader, "load_with_outcome", staticmethod(fake_load_with_outcome)
    )

    # (2) The pre-index and (3) the evidence provider: dummies, because a real
    # retriever embeds the document and the fact layer's citations are not what
    # any test in this file is about.
    monkeypatch.setattr(
        analysis_orchestrator,
        "build_qa_retriever",
        lambda bundle, document=None: SimpleNamespace(indexed=document),
    )
    monkeypatch.setattr(
        analysis_paper_facts,
        "make_evidence_provider",
        lambda retriever, document_id, **kwargs: SimpleNamespace(
            document_id=document_id
        ),
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

    monkeypatch.setattr(
        analysis_service, "register_analysis_document", fake_register_analysis_document
    )

    # (4) The analysis. The signature is the real one — including the
    # keyword-only arguments the stage passes — so a caller that started passing
    # a new one raises here instead of having it silently dropped by a
    # ``**kwargs``.
    def fake_analyze_with_fact_layer(
        file_path,
        original_filename,
        api_key,
        generate_mermaid_bool,
        generate_evaluation_bool,
        session_id,
        generate_research_brief_bool=True,
        *,
        document_id="",
        evidence_provider=None,
        store=None,
        loaded_document=None,
    ):
        analyzer = _StubAnalyzer(api_key)
        result = {
            "summary": "ok",
            "char_count": len(analyzer.document_content),
            "source_filename": str(file_path),
            "sections": {
                "summary": {
                    "status": "success",
                    "content": "ok",
                    "error": "",
                    "retryable": False,
                }
            },
        }
        # ``run``/``report`` stay ``None``: the fact layer's provenance payload
        # is covered where it is actually produced
        # (``tests/test_fact_layer_route.py``), and leaving them off keeps this
        # stub from having to fabricate a ``PaperFacts`` these assertions never
        # look at. ``finalize_analysis_result`` accepts exactly that — it is how
        # the legacy six-section path calls it.
        return analysis_service.finalize_analysis_result(
            result=result,
            analyzer=analyzer,
            original_filename=original_filename,
            generate_evaluation_bool=generate_evaluation_bool,
            session_id=session_id,
            generate_research_brief_bool=generate_research_brief_bool,
            run=None,
            report=None,
        )

    monkeypatch.setattr(
        analysis_service, "analyze_with_fact_layer", fake_analyze_with_fact_layer
    )


def _upload_client(monkeypatch):
    """A TestClient over the real app, with the analyzer stubbed for its lifetime.

    The stub is installed with ``monkeypatch.setattr`` (see
    :func:`_install_stub`) and is on the seam the job's analysing stage reads
    from :mod:`paperwhisperer.analysis.orchestrator`. A ``TestClient`` built
    from the real ``app`` is the only way to exercise the route as deployed.
    Calling the route function directly would skip the multipart parsing that
    produces ``file.filename`` and ``file.content_type`` in the first place,
    which are exactly the two values under test.
    """
    from fastapi.testclient import TestClient

    from paperwhisperer.app import app

    _install_stub(monkeypatch, analysis_orchestrator, None)
    return TestClient(app)


#: A caller-supplied API key that a real browser would send. Since T2.3 a
#: caller-supplied key only reaches the job's analysing stage when it is at
#: least eight characters (:func:`paperwhisperer.api.routes_jobs.
#: register_ephemeral_secret`), so a one-character placeholder would not be a
#: key the stage accepts. Realistic length here; no assertion is weakened.
BROWSER_KEY = "sk-test-browser-key"


class _StubAnalyzer:
    """A ``DocumentAnalyzer`` with no model behind it, for the finaliser.

    Nothing on the ``/api/analyze`` path constructs one any more — the analysing
    stage is the fact layer, and ``_stub_fact_layer_seams`` builds this class
    itself to stand in for the analyzer ``finalize_analysis_result`` wants to
    read ``.document_content`` and ``.version`` off. ``analyze_stream`` was the
    seam the route used before the migration and is kept only so this class
    still describes the whole legacy surface; the ``analyze`` entry point stays
    because the legacy generator itself is still callable, and a stub that only
    had the method the route used would be a stub that fails for the wrong
    reason the moment anything else reaches for it.
    """

    version = "test-version"

    def __init__(self, api_key):
        self.api_key = api_key
        self.document_content = "document text"

    def analyze(self, file_path, mermaid=True, evaluation=True, brief=True):
        return {"summary": "ok", "source_filename": str(file_path)}

    def analyze_stream(self, file_path, mermaid=True, evaluation=True, brief=True):
        yield {
            "type": "section",
            "name": "summary",
            "section": {"status": "success", "content": "ok", "error": "", "retryable": False},
        }
        yield {
            "type": "done",
            "result": {"summary": "ok", "char_count": 10, "source_filename": str(file_path)},
        }


class _StubParseOutcome:
    """The bits of ``ParseOutcome`` the job's parse stage reads.

    A usable, successful outcome built around a real minimal ``ParsedDocument``,
    so the stage proceeds exactly as it would after a genuine local parse. The
    parser and the model are both stubbed here; the pipeline's own bookkeeping
    (which outcome fields it copies into the job, which warnings it records) is
    left real.
    """

    status = "success"
    parsed_by = "stub"
    used_cloud = False
    touched_network = False
    cache_hit = False
    warnings: list = []
    error_message = ""

    def __init__(self, document):
        self.document = document

    @property
    def is_usable(self) -> bool:
        return True

    def summary_line(self) -> str:
        return "由 stub 解析。"


@pytest.fixture
def upload_client(monkeypatch, isolated_job_stores):
    """A TestClient whose analysis call is stubbed, restored on teardown.

    The stub is installed with ``monkeypatch.setattr`` rather than by assigning
    the module attribute directly. A direct assignment to a session-wide
    singleton has to be undone by hand, and a hand-written undo that does not
    run — because a test called the helper without requesting this fixture, or
    because the teardown order changed — leaves ``fake_analyze`` installed for
    every later test in the session. That is not hypothetical: it happened
    here, and the upload-limit suite, which stubs ``DocumentAnalyzer`` in
    ``analysis.service`` a completely different way, silently ran against this
    stub instead of its own and read a summary of "ok". ``monkeypatch`` removes
    the possibility by making the undo unconditional.
    """

    yield _upload_client(monkeypatch)


def build_sample_documents(root):
    """Write one real file per format the legacy adapter claims, return the paths.

    Real bytes rather than mocks, for the same reason the legacy adapter's own
    tests build real files: the point is to prove that turning validation on
    does not break a format that works today, and a mock document cannot fail
    in any of the ways a real ``.pptx`` fails.
    """
    from docx import Document as DocxDocument
    from pptx import Presentation
    from pptx.util import Inches

    root = Path(root)
    text = root / "notes.txt"
    text.write_text("First paragraph.\n\nSecond paragraph, longer than the first.\n",
                    encoding="utf-8")

    markdown = root / "notes.md"
    markdown.write_text(
        "# Findings\n\n"
        "Prose under a level-one heading, long enough to be a paragraph.\n\n"
        "## Method\n\n"
        "```python\n# a comment, not a heading\nprint('x')\n```\n\n"
        "Body under the section.\n",
        encoding="utf-8",
    )

    docx_path = root / "report.docx"
    document = DocxDocument()
    document.add_heading("Introduction", level=1)
    document.add_paragraph("Body text under the heading.")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Metric"
    table.cell(0, 1).text = "Score"
    table.cell(1, 0).text = "BLEU"
    table.cell(1, 1).text = "27.3"
    document.save(docx_path)

    pptx_path = root / "deck.pptx"
    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[5])
    slide.shapes.title.text = "A title"
    slide.shapes.add_textbox(
        Inches(1), Inches(2), Inches(3), Inches(1)
    ).text_frame.text = "Body"
    presentation.save(pptx_path)

    return [text, markdown, docx_path, pptx_path]


@pytest.fixture(autouse=True)
def _gateway_backend(monkeypatch):
    """Pin the feature flag so the registry is the default one for these tests."""
    monkeypatch.setattr(config, "PARSER_BACKEND", "gateway", raising=False)
    monkeypatch.setattr(config, "PARSER_PRIMARY", "", raising=False)
    monkeypatch.setattr(config, "PARSER_FALLBACK_CHAIN", (), raising=False)


# --------------------------------------------------------------------------
# Guard: the module is actually reached
# --------------------------------------------------------------------------


class TestTheWiringIsLive:
    """Tests that fail if the contracts are disconnected again.

    This is the class the whole file exists for. The rest of the file asserts
    behaviour; this one asserts *reachability*, which is the property the
    previous state of the repository lacked. Each test spies on the contract
    and requires evidence that it ran — because a test that merely exercises a
    code path is satisfied by any code that happens to produce the same
    observable, including code with the validation deleted.
    """

    def test_the_gateway_validates_through_the_contract_and_not_around_it(
        self, monkeypatch
    ):
        # The strongest form of "it is wired": point the contract at a
        # recognisable document and require the router to have handed that
        # exact object to it. Deleting the validation leaves ``seen`` empty and
        # this fails, which a behavioural test would not.
        seen: list[dict] = []
        original = ParsedDocumentContract.model_validate

        def spy(payload, *args, **kwargs):
            seen.append(payload)
            return original(payload, *args, **kwargs)

        monkeypatch.setattr(ParsedDocumentContract, "model_validate", spy)

        router = router_for(FakeParser("only", outcome=good_result()))
        result = router.parse("a.pdf", PDF_MEDIA_TYPE)

        assert result.status == PARSE_SUCCESS
        assert len(seen) == 1
        # The payload is the document's own serialized form, so the spy has
        # seen exactly the bytes the parse cache would be handed.
        assert seen[0]["document_id"] == result.document.document_id
        assert [item["text"] for item in seen[0]["elements"]] == [
            element.text for element in result.document.elements
        ]

    def test_an_upload_is_validated_through_the_contract(
        self, monkeypatch, runtime_dirs, upload_client
    ):
        # Same guard for the API edge. The contract is called with the filename
        # and the declared type the client actually sent, so a route that
        # stopped consulting it fails here even though the legacy
        # ``is_allowed_file`` check would still happily accept ``paper.txt``.
        seen: list[dict] = []
        original = UploadRequestContract.__init__

        def spy(self, **kwargs):
            seen.append(kwargs)
            original(self, **kwargs)

        monkeypatch.setattr(UploadRequestContract, "__init__", spy)

        # Through the fixture, never through ``_upload_client()`` directly.
        # That helper installs a stub on the route module as a global
        # assignment; only the fixture restores it. Calling the helper from a
        # test that does not request the fixture left ``fake_analyze``
        # installed for the rest of the session, and every later test that
        # stubs ``DocumentAnalyzer`` in ``analysis.service`` — the way the
        # upload-limit suite does — then silently exercised this stub instead
        # of its own, and read a "summary" of "ok".
        response = upload_client.post(
            "/api/analyze",
            files={"file": ("paper.txt", io.BytesIO(b"body"), TXT_MEDIA_TYPE)},
            data={"api_key": BROWSER_KEY},
        )

        assert response.status_code == 200
        assert seen, "the upload route stopped consulting the contract"
        assert seen[0]["filename"] == "paper.txt"
        assert seen[0]["media_type"] == TXT_MEDIA_TYPE

    def test_the_gateway_module_documents_the_contract_as_a_boundary(self):
        # Guards against a future "just add a quick check here" that
        # re-duplicates the contract by hand. The router is allowed to reach
        # for ``ParsedDocumentContract``; it is not allowed to grow a private
        # substitute, and the module docstring is where that decision has to
        # be written down for the next person to find.
        import paperwhisperer.ingestion.gateway as gateway

        docstring = gateway.__doc__ or ""
        assert "contracts" in docstring
        assert "ParsedDocumentContract" in docstring


# --------------------------------------------------------------------------
# Boundary 1: adapter output entering the gateway
# --------------------------------------------------------------------------


class TestMalformedAdapterOutputBecomesADiagnosableFailure:
    """A broken adapter result is refused, named, and never raised.

    The three assertions that matter are separate on purpose. A ``failed``
    status alone proves the router did *something*; the message proves a human
    can act on it; and the absence of an exception proves the router's
    no-raise guarantee survived the addition.
    """

    def _parse_with(self, document, *, second=None):
        parsers = [FakeParser("broken", outcome=ParseResult.from_document(document))]
        if second is not None:
            parsers.append(FakeParser("healthy", outcome=second))
        return router_for(*parsers).parse("a.pdf", PDF_MEDIA_TYPE)

    def test_a_locator_missing_its_page_is_refused_with_the_field_named(self):
        document = good_document()
        # Exactly the MinerU shape: a pdf_page that promises a page and
        # supplies none. Not reachable through ``_repair_known_locator_defects``
        # here because the repair is exercised separately; this pins that a
        # page number that is *wrong* rather than absent is not survivable.
        document.elements[0].locator = SourceLocator(kind="pdf_page", page=0)

        result = self._parse_with(document)

        assert result.status == PARSE_FAILED
        assert result.document is None
        assert result.is_usable is False
        assert "elements[0]" in result.error_message
        assert "index must be >= 1" in result.error_message

    def test_the_message_names_the_parser_so_the_user_knows_it_is_not_their_file(self):
        document = good_document()
        document.elements[0].locator = SourceLocator(kind="pdf_page", page=0)

        result = self._parse_with(document)

        assert "broken" in result.error_message
        assert "结构" in result.error_message or "规范" in result.error_message

    def test_a_table_role_with_no_grid_is_refused(self):
        # The flattening failure the quality gate can only count, and a table
        # that arrived as prose is invisible to every downstream check once it
        # is labelled ``table``.
        document = good_document(elements=[
            Element(role="table", text="Metric | BLEU", locator=SourceLocator(kind="pdf_page", page=1))
        ])

        result = self._parse_with(document)

        assert result.status == PARSE_FAILED
        assert "elements[0]" in result.error_message
        assert "grid" in result.error_message

    def test_a_locator_carrying_another_kinds_field_is_refused(self):
        document = good_document()
        # A pdf_page carrying a slide number: the corruption that survives
        # every consumer, because all of them dispatch on ``kind``.
        document.elements[0].locator = SourceLocator(kind="pdf_page", page=1, slide=3)

        result = self._parse_with(document)

        assert result.status == PARSE_FAILED
        assert "does not belong" in result.error_message

    def test_a_mirrored_bounding_box_is_refused(self):
        document = good_document()
        # The adapter normalized it, so this has to be built past the
        # constructor — which is exactly the shape a buggy converter produces
        # and every consumer would then draw as a highlight in the wrong place.
        box = BoundingBox(300.0, 700.0, 72.0, 640.0)
        box.x0, box.x1 = box.x1, box.x0
        document.elements[1].locator = SourceLocator(kind="pdf_page", page=1, bbox=box)

        result = self._parse_with(document)

        assert result.status == PARSE_FAILED
        assert "not ordered" in result.error_message

    def test_a_malformed_schema_version_on_an_adapter_document_is_refused(self):
        document = good_document(schema_version="1")

        result = self._parse_with(document)

        assert result.status == PARSE_FAILED
        assert "schema_version" in result.error_message

    def test_the_failure_does_not_raise_and_the_chain_continues_to_the_next_parser(self):
        # The router's no-raise guarantee under the new code path. A contract
        # violation is a worse *parse*, not a worse request, so a healthy
        # backend behind a broken one must still be reached — that is what
        # makes adding validation here safe rather than a new outage.
        broken = good_document()
        broken.elements[0].locator = SourceLocator(kind="pdf_page", page=0)

        result = self._parse_with(broken, second=good_result())

        assert result.status == PARSE_SUCCESS
        assert result.document is not None
        attempted = [entry["parser"] for entry in result.diagnostics if entry["attempted"]]
        assert attempted == ["broken", "healthy"]
        # And the reason the first one was dropped is still on the record.
        assert "index must be >= 1" in result.diagnostics[0]["error"]

    def test_the_refusal_is_reachable_in_the_log_even_when_the_result_is_taken_from_a_later_parser(
        self, caplog
    ):
        broken = good_document()
        broken.elements[0].locator = SourceLocator(kind="pdf_page", page=0)

        with caplog.at_level("WARNING"):
            self._parse_with(broken, second=good_result())

        assert any("contract validation" in record.getMessage()
                   for record in caplog.records)


class TestAMalformedDocumentNeverBecomesAnEmptySuccess:
    """A refusal must not be mistaken for a document that simply has no text.

    The failure mode this closes is subtle and was live before validation: an
    element list that is empty or a locator that resolves nowhere still yields a
    ``ParseResult`` with a document, so every downstream stage proceeds and the
    user is shown a completed analysis of nothing. A refused document is
    ``document is None``, which is unambiguous.
    """

    def test_a_refused_document_is_never_attached_to_the_result(self):
        document = good_document()
        document.elements[0].locator = SourceLocator(kind="pdf_page", page=0)

        result = router_for(
            FakeParser("broken", outcome=ParseResult.from_document(document))
        ).parse("a.pdf", PDF_MEDIA_TYPE)

        assert result.status == PARSE_FAILED
        assert result.document is None
        assert result.quality is None

    def test_an_empty_but_valid_document_is_accepted_as_an_empty_document(self):
        # The counterpart, so the previous test is not merely "empty is bad".
        # An adapter that legitimately found nothing still returns a usable
        # result: the quality gate, not the contract, is what judges that.
        empty = ParsedDocument(source_filename="blank.pdf", media_type=PDF_MEDIA_TYPE)

        result = router_for(
            FakeParser("blank", outcome=ParseResult.from_document(empty))
        ).parse("a.pdf", PDF_MEDIA_TYPE)

        assert result.status == PARSE_SUCCESS
        assert result.document is not None
        assert result.document.elements == []


# --------------------------------------------------------------------------
# Boundary 2: the user-supplied metadata at the API edge
class TestMalformedUploadIsRejectedAtTheEdge:
    """The two client-controlled inputs, refused before anything is stored.

    Each case is one the pre-existing ``is_allowed_file`` check passes, which is
    the point: these are precisely the rejections that were missing.
    """

    @pytest.mark.parametrize(
        ("filename", "content_type", "expected_field", "expected_fragment"),
        [
            ("../../etc/passwd.txt", TXT_MEDIA_TYPE, "filename", "bare name"),
            ("notes.txt", "image/png", "media_type", "unsupported media type"),
            ("paper.pdf", "application/x-msdownload", "media_type", "unsupported media type"),
        ],
    )
    def test_a_malformed_upload_is_refused_naming_the_field(
        self, upload_client, runtime_dirs, filename, content_type,
        expected_field, expected_fragment,
    ):
        response = upload_client.post(
            "/api/analyze",
            files={"file": (filename, io.BytesIO(b"body"), content_type)},
            data={"api_key": BROWSER_KEY},
        )

        assert response.status_code == 400
        payload = response.json()
        assert payload["code"] == "invalid_upload_metadata"
        assert expected_field in payload["error"]
        assert expected_fragment in payload["error"]
        # Refused before storage: a rejected upload leaves nothing behind.
        assert list(runtime_dirs["uploads"].iterdir()) == []

    def test_a_valid_upload_still_reaches_the_analysis(self, upload_client):
        response = upload_client.post(
            "/api/analyze",
            files={"file": ("paper.txt", io.BytesIO(b"body"), TXT_MEDIA_TYPE)},
            data={"api_key": BROWSER_KEY},
        )

        assert response.status_code == 200
        assert response.json()["summary"] == "ok"

    def test_the_browser_pdf_alias_is_accepted_rather_than_refused(self, upload_client):
        # ``application/x-pdf`` is what browsers actually send for a PDF. A
        # strict reading of "the declared type must be a known IANA type" would
        # turn every real browser upload into a 400, which is why the contract
        # normalizes this spelling rather than refusing it.
        response = upload_client.post(
            "/api/analyze",
            files={"file": ("paper.pdf", io.BytesIO(b"%PDF-1.4"), "application/x-pdf")},
            data={"api_key": BROWSER_KEY},
        )

        assert response.status_code == 200

    def test_the_legacy_error_codes_survive_the_contract_layer(self, upload_client):
        # The front end branches on these codes. Adding a stricter check
        # underneath must not change the answer to a question it already asked,
        # which is why the contract is layered *after* the legacy checks rather
        # than replacing them.
        response = upload_client.post(
            "/api/analyze",
            files={"file": ("malware.exe", io.BytesIO(b"MZ"), "application/octet-stream")},
            data={"api_key": BROWSER_KEY},
        )

        assert response.status_code == 400
        assert response.json()["code"] == "unsupported_file_type"

    def test_a_rejected_upload_never_reaches_the_analyzer(
        self, upload_client, runtime_dirs, monkeypatch
    ):
        # The contract runs inside ``_validate_upload``, before the file is
        # written to disk and before any job is submitted. Asserting the upload
        # folder is empty (above) proves the write did not happen; this proves
        # the expensive path after it did not run either.
        #
        # The tripwire has to sit on the seam the job *actually* executes, and
        # that moved. Before the fact-layer migration the analysing stage
        # instantiated ``analysis.orchestrator.DocumentAnalyzer``, so exploding
        # that class was the assertion. It instantiates nothing now: it reads
        # the loader, the retriever and ``service.analyze_with_fact_layer``. An
        # upload that reached a job would now fail on the *load*, before the
        # analyzer name was ever spoken of, so a tripwire left on
        # ``DocumentAnalyzer`` would pass no matter what the route did — the
        # worst kind of still-green test. So the loader and the analysis entry
        # point are both armed here, plus the old class for the case where a
        # future route goes back to constructing one.
        from paperwhisperer.analysis import service as analysis_service
        from paperwhisperer.documents.loader import DocumentLoader

        def explode(*args, **kwargs):
            raise AssertionError("the analysis was reached for a rejected upload")

        monkeypatch.setattr(analysis_orchestrator, "DocumentAnalyzer", explode)
        monkeypatch.setattr(
            DocumentLoader, "load_with_outcome", staticmethod(explode)
        )
        monkeypatch.setattr(analysis_service, "analyze_with_fact_layer", explode)

        response = upload_client.post(
            "/api/analyze",
            files={"file": ("notes.txt", io.BytesIO(b"body"), "image/png")},
            data={"api_key": BROWSER_KEY},
        )

        assert response.status_code == 400
        assert response.json()["code"] == "invalid_upload_metadata"


# --------------------------------------------------------------------------
# Boundary 1 again: the known producer defect, handled loudly
# --------------------------------------------------------------------------


class TestTheKnownProducerDefectIsRepairedLoudly:
    """``mineru_cloud.py`` spells ``pdf_page`` even when there is no page.

    The contract is right to refuse it and this does not weaken the contract.
    What it does is downgrade the *one* defect that has an honest alternative
    spelling, and record every single repair, so that losing a page index is
    visible as a warning rather than as a citation that quietly stops resolving.
    """

    def _mineru_shaped(self):
        document = ParsedDocument(
            source_filename="cloud.pdf",
            media_type=PDF_MEDIA_TYPE,
            parser_name="mineru_cloud",
            elements=[
                Element(
                    role="paragraph",
                    text="This one has a page.",
                    locator=SourceLocator(kind="pdf_page", page=1),
                ),
                Element(
                    role="paragraph",
                    text="This one lost its page index.",
                    locator=SourceLocator(kind="pdf_page", page=None),
                ),
            ],
        )
        document.build_units()
        return document

    def _parse(self):
        return router_for(
            FakeParser("mineru_cloud", outcome=ParseResult.from_document(self._mineru_shaped()))
        ).parse("a.pdf", PDF_MEDIA_TYPE)

    def test_the_parse_is_not_discarded(self):
        result = self._parse()

        # The other half of the document was fine. Throwing all of it away over
        # one missing page index is a worse outcome than the defect itself.
        assert result.status == PARSE_SUCCESS
        assert result.document is not None
        assert {element.text for element in result.document.elements} == {
            "This one has a page.",
            "This one lost its page index.",
        }

    def test_the_defective_locator_is_downgraded_to_the_honest_kind(self):
        result = self._parse()

        repaired = _element(result.document, "This one lost its page index.").locator
        assert repaired.kind == "unknown"
        assert repaired.page is None
        assert repaired.is_resolvable is False

    def test_the_element_that_did_have_a_page_keeps_it(self):
        result = self._parse()

        intact = _element(result.document, "This one has a page.").locator
        assert intact.kind == "pdf_page"
        assert intact.page == 1
        assert intact.is_resolvable is True

    def test_the_repair_is_recorded_as_a_warning_naming_the_position(self):
        result = self._parse()

        assert any("elements[" in warning and "unknown" in warning
                   for warning in result.warnings)

    def test_the_repair_is_recorded_as_structured_evidence(self):
        result = self._parse()

        repairs = result.contract_repairs
        # Two: the element and the unit ``build_units`` derived from it. Both
        # carried the defect, and leaving the unit unrepaired would hand the
        # contract a page-less pdf_page it would then refuse.
        assert [entry["where"] for entry in repairs] == ["elements[0]", "units[0]"]
        assert all(entry["from_kind"] == "pdf_page" for entry in repairs)
        assert all(entry["field"] == "page" for entry in repairs)

    def test_the_absence_survives_serialization_as_an_absence(self):
        # The repair must not be a lie the cache then re-reads as truth: a
        # round-tripped document has to still say "unknown", not have the page
        # invented back into it.
        result = self._parse()

        restored = ParsedDocument.from_dict(result.document.to_dict())
        assert _element(restored, "This one lost its page index.").locator.kind == "unknown"

    def test_a_unit_carrying_the_same_defect_is_repaired_too(self):
        result = self._parse()

        assert all(unit.locator.kind != "pdf_page" or unit.locator.page is not None
                   for unit in result.document.units)

    def test_a_wrong_page_number_is_still_refused_and_never_repaired(self):  # noqa: D401
        # The asymmetry is the whole design. A page number that is *absent* has
        # one honest value; a page number that is *wrong* has none, and a
        # document whose citations point at the wrong page must fail rather
        # than be quietly downgraded into something that looks fine.
        document = self._mineru_shaped()
        document.elements[0].locator = SourceLocator(kind="pdf_page", page=0)

        result = router_for(
            FakeParser("mineru_cloud", outcome=ParseResult.from_document(document))
        ).parse("a.pdf", PDF_MEDIA_TYPE)

        assert result.status == PARSE_FAILED
        assert "index must be >= 1" in result.error_message

    def test_the_contract_itself_still_refuses_the_unrepaired_shape(self):
        # Proof that the tolerance lives in the gateway and not in the
        # contract. If this ever starts passing, the fix that was made here was
        # the wrong one.
        with pytest.raises(ValidationError, match="must carry 'page'"):
            SourceLocatorContract(kind="pdf_page")

        with pytest.raises(ValidationError, match="must carry 'page'"):
            ParsedDocumentContract.from_dataclass(self._mineru_shaped())


# --------------------------------------------------------------------------
# A valid document passes through the whole path unchanged
# --------------------------------------------------------------------------


class TestAValidDocumentIsUntouchedByTheWiring:
    """The regression net for everything above.

    Validation that quietly altered a good document would be worse than no
    validation: the document would be wrong and there would be nothing to see.
    These assert the *identity* of what comes out, not merely that it succeeded.
    """

    def test_the_document_object_is_returned_unchanged(self):
        document = good_document()
        before = document.to_dict()

        result = router_for(
            FakeParser("legacy", outcome=ParseResult.from_document(document))
        ).parse("a.pdf", PDF_MEDIA_TYPE)

        assert result.status == PARSE_SUCCESS
        assert result.document is document
        assert result.document.to_dict() == before

    def test_nothing_is_repaired_and_no_warning_is_added(self):
        result = router_for(
            FakeParser("legacy", outcome=good_result())
        ).parse("a.pdf", PDF_MEDIA_TYPE)

        assert result.contract_repairs == []
        assert result.warnings == []


class TestAQualityFailureIsVisible:
    """The other half of the promise above, and the reason the gateway warns.

    ``TestAValidDocumentIsUntouchedByTheWiring`` pins that a good document is
    left alone. This pins that a document the quality gate *fails* is not left
    silently alone: ``parse`` succeeds, the status never changes, and the reader
    is told anyway. A parse that returned a degraded document with no warning
    anywhere is the ``silent_truncation`` gate's whole subject — the analysis
    would confidently summarise half a paper while the screen said nothing.

    Both channels are asserted because they are read by different people:
    ``result.warnings`` is what the caller surfaces, ``document.warnings`` is
    what survives into the stored parse and what the gate itself consults.
    Asserting only one leaves the other able to drift away unnoticed.
    """

    def test_a_failed_rule_reaches_both_warning_channels(self):
        # A one-page, one-sentence document is what a scan that never produced
        # a text layer looks like: it parses, it is contract-valid, and it is
        # empty enough for the density rule to fail it.
        document = good_document(page_count=1, elements=[
            Element(
                role="paragraph",
                text="A resolvable sentence on page 1.",
                locator=SourceLocator(kind="pdf_page", page=1),
            ),
        ])

        result = router_for(
            FakeParser("legacy", outcome=ParseResult.from_document(document))
        ).parse("a.pdf", PDF_MEDIA_TYPE)

        assert result.status == PARSE_SUCCESS
        assert result.quality is not None
        assert "text_density" in result.quality.failed_rules
        assert any("text_density" in warning for warning in result.warnings)
        assert any("text_density" in warning for warning in document.warnings)

    def test_a_clean_document_gains_no_warning_at_all(self):
        # The negative case, so the warning above cannot be produced by simply
        # always warning. ``good_document`` clears every rule; the density rule
        # is the one that would fire if it did not.
        document = good_document()

        result = router_for(
            FakeParser("legacy", outcome=ParseResult.from_document(document))
        ).parse("a.pdf", PDF_MEDIA_TYPE)

        assert result.quality is not None
        assert not result.quality.failed_rules
        assert result.warnings == []
        assert document.warnings == []

    def test_a_warning_rule_does_not_alarm_the_reader(self):
        # ``repeated_furniture`` fires on any paper whose header repeats on
        # every page, which is most papers. Surfacing WARN as "your document
        # may be incomplete" would train readers to ignore the sentence, so it
        # stays on the report for whoever looks. Furniture is ``role``
        # ``header``/``footer`` — it is excluded from ``body_elements``, so the
        # repeated lines contribute nothing to the density rule and the only
        # rule that can fire here is the furniture one.
        repeated = "Page 3 of 12 — the same line appears at the top of every page."
        document = good_document(elements=[
            Element(role="header", text=repeated, locator=SourceLocator(kind="pdf_page", page=1)),
            Element(role="header", text=repeated, locator=SourceLocator(kind="pdf_page", page=1)),
            Element(role="header", text=repeated, locator=SourceLocator(kind="pdf_page", page=1)),
            Element(
                role="paragraph",
                text=(
                    "The body text of the page, which is long enough on its own that "
                    "the density rule is satisfied without any help from the three "
                    "repeated lines above it, and so the only rule that could fire "
                    "here is the furniture one."
                ),
                locator=SourceLocator(kind="pdf_page", page=1),
            ),
        ])

        result = router_for(
            FakeParser("legacy", outcome=ParseResult.from_document(document))
        ).parse("a.pdf", PDF_MEDIA_TYPE)

        assert "repeated_furniture" in result.quality.warned_rules
        assert not result.quality.failed_rules
        assert result.warnings == []

    def test_the_legacy_adapter_end_to_end_still_validates(self, tmp_path):
        # The real producer, not a hand-built document. Every format the
        # legacy adapter claims is exercised, because a contract that rejects
        # one of them is a contract that would take a working deployment's
        # uploads down the moment validation was switched on.
        from paperwhisperer.ingestion.adapters.legacy_loader import LegacyDocumentParser

        for path in build_sample_documents(tmp_path):
            result = LegacyDocumentParser().parse(str(path), ParseOptions())
            assert result.is_usable, path.name
            ParsedDocumentContract.from_dataclass(result.document)

    def test_an_absolute_bounding_box_survives_the_contract_byte_for_byte(self):
        box = BoundingBox(72.0, 640.0, 300.0, 700.0)

        assert box.relative is False
        assert box.to_dict() == {"x0": 72.0, "y0": 640.0, "x1": 300.0, "y1": 700.0}
        assert BoundingBoxContract.from_dataclass(box).to_dataclass() == box


# --------------------------------------------------------------------------
# The two schema fixes, asserted at the boundaries that use them
# --------------------------------------------------------------------------


class TestTheSchemaFixesHoldAtTheBoundary:
    """The dataclass and the contract must agree about a version string.

    They did not, and the disagreement was invisible: ``from_dict`` compared
    ``"1".split(".")[0]`` — which is ``"1"`` — against the supported major and
    called the result compatible. The contract refused it. Whichever one a
    reader happened to use decided whether a document was readable, which is
    the definition of two gates disagreeing.
    """

    @pytest.mark.parametrize("bad", ["1", "1.0.0", "v1.0", "one point oh", "1.x"])
    def test_the_dataclass_reader_refuses_a_malformed_version(self, bad):
        with pytest.raises(ValueError, match="[Mm]alformed"):
            ParsedDocument.from_dict({"schema_version": bad, "elements": []})

    @pytest.mark.parametrize("bad", ["1", "1.0.0", "v1.0", "one point oh", "1.x"])
    def test_the_contract_refuses_exactly_the_same_versions(self, bad):
        with pytest.raises(ValidationError, match="malformed schema_version"):
            ParsedDocumentContract(
                schema_version=bad, document_id="doc_1"
            )

    def test_a_minor_bump_still_reads_in_both(self):
        assert ParsedDocument.from_dict({"schema_version": "1.7", "elements": []})
        assert ParsedDocumentContract(schema_version="1.7", document_id="doc_1")

    def test_a_future_major_is_still_refused_in_both(self):
        with pytest.raises(ValueError, match="[Uu]nsupported"):
            ParsedDocument.from_dict({"schema_version": "9.0", "elements": []})
        with pytest.raises(ValidationError, match="unsupported schema version"):
            ParsedDocumentContract(schema_version="9.0", document_id="doc_1")

    def test_the_current_version_round_trips(self):
        assert SCHEMA_VERSION == "1.0"
        assert ParsedDocument.from_dict({"elements": []}).schema_version == SCHEMA_VERSION

    def test_the_gateway_refuses_a_document_carrying_a_malformed_version(self):
        result = router_for(
            FakeParser("sloppy", outcome=ParseResult.from_document(good_document(
                schema_version="1"
            )))
        ).parse("a.pdf", PDF_MEDIA_TYPE)

        assert result.status == PARSE_FAILED
        assert "schema_version" in result.error_message


class TestTheRelativeBoxIsRecordedNotSilentlyAccepted:
    """A mirrored rectangle has to be distinguishable from an absolute one.

    The adapter's warning says "some boxes are approximate". What the code that
    is about to *draw* the highlight needs is to know that *this* box is. The
    flag is what carries that, and ``to_query`` is the method whose entire
    output feeds a renderer — so it is the method that has to refuse.
    """

    def test_a_mirrored_box_is_flagged_relative(self):
        # What ``docling_adapter.to_bounding_box`` produces for a TOPLEFT box
        # with no page height: the y axis is negated rather than mirrored.
        box = BoundingBox(x0=10.0, y0=-100.0, x1=300.0, y1=-50.0)

        assert box.relative is True

    def test_a_degraded_box_flush_with_the_top_of_the_page_is_still_flagged(self):
        # The edge that a "negative y1" test would miss: a TOPLEFT rectangle
        # flush with the top of the page (t=0) negates to y1 == 0, so the upper
        # edge is legitimately zero and only the lower edge gives it away.
        assert BoundingBox(10.0, -100.0, 300.0, 0.0).relative is True

    def test_an_absolute_box_sitting_on_the_page_origin_is_not_flagged(self):
        # The other side of that edge: y0 == 0 is a legal absolute position —
        # the bottom margin of the page — and flagging it would mark most of
        # a correctly-parsed document approximate.
        assert BoundingBox(10.0, 0.0, 300.0, 40.0).relative is False

    def test_an_absolute_box_is_not_flagged(self):
        assert BoundingBox(72.0, 640.0, 300.0, 700.0).relative is False

    def test_the_flag_round_trips_through_serialization(self):
        box = BoundingBox(10.0, -100.0, 300.0, -50.0)

        assert BoundingBox.from_dict(box.to_dict()) == box
        assert box.to_dict()["relative"] is True

    def test_an_absolute_box_serializes_identically_to_a_pre_flag_build(self):
        # Byte compatibility is the reason the flag is pruned at its default:
        # a document written by this build must be indistinguishable from one
        # written before the field existed, so an older reader is unaffected.
        assert BoundingBox(10.0, 20.0, 110.0, 60.0).to_dict() == {
            "x0": 10.0, "y0": 20.0, "x1": 110.0, "y1": 60.0,
        }

    def test_to_query_refuses_a_relative_box(self):
        # The payoff. A fragment built from a mirrored rectangle draws a
        # highlight over the wrong lines while looking entirely correct, and
        # nothing downstream can tell.
        box = BoundingBox(10.0, -100.0, 300.0, -50.0)

        with pytest.raises(ValueError, match="relative"):
            box.to_query()

    def test_to_query_still_works_for_an_absolute_box(self):
        assert BoundingBox(10, 20, 110, 60).to_query() == "xywh=10,20,100,40"

    def test_the_flag_survives_the_contract(self):
        box = BoundingBox(10.0, -100.0, 300.0, -50.0)

        assert BoundingBoxContract.from_dataclass(box).relative is True
        assert BoundingBoxContract.from_dataclass(box).to_dataclass() == box

    def test_a_relative_box_is_accepted_by_the_contract_because_it_is_honest(self):
        # Not a rejection case. The box is correctly *flagged*; refusing it
        # would throw away real geometry, and the flag is what stops it being
        # rendered. The failure this prevents is the flag being dropped, not
        # the box existing.
        contract = BoundingBoxContract(x0=10.0, y0=-100.0, x1=300.0, y1=-50.0, relative=True)

        assert contract.relative is True
