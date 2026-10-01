"""The frozen ``/v1`` surface, asserted through the real application.

T2.4's acceptance criterion is "现有前端无破坏性切换；OpenAPI 生成", and its
subject — "对外契约冻结" — is a claim about the future: once this ships, a client
may depend on the field names, the codes and the status codes. A test suite that
asserts behaviour is the wrong instrument for that, because behaviour survives a
rename. So this file has three kinds of test and the first kind is the one that
does the work:

*Pin the shape.* ``test_*_field_names_are_frozen`` asserts the exact key set of
every resource, both the polling shape and the SSE shape, and the schema's
``properties`` are compared against the keys a live response actually carries.
A field that disappears, or an ``OpenAPI`` document that describes a field the
route stopped returning, fails here rather than in somebody's client.
*Pin the codes.* Every documented failure has a test that reaches it through the
HTTP surface and asserts the ``code`` — not the prose. The prose is Chinese and
may be reworded; the code is the API.
*Pin the resync.* A job's state over SSE, then a disconnect and a reconnect that
says which events it missed, because "浏览器刷新/SSE 断开不丢任务" is the Phase 2 exit
criterion and it is only testable through the wire.

No LLM, no network, no MinerU. The parsers are a stub registered through the
gateway's own factory and the Q&A analyzer is a stub, so the tests exercise the
routing, the state machine, the store and the serialisation and nothing else.
The store is a real SQLite file on ``tmp_path`` and the object store is a real
directory tree, because "the job table lives in the same database as the
documents" is a claim about files.
"""

from __future__ import annotations

import io
import json
import os
import re
from types import SimpleNamespace

import anyio
import pytest

from paperwhisperer.analysis.orchestrator import CONFIDENCE_SUPPORTED, AnswerResult
from paperwhisperer.api import routes_documents, routes_evidence, routes_jobs, routes_v1
from paperwhisperer.app import app
from paperwhisperer.core import config
from paperwhisperer.infra import jobs as J
from paperwhisperer.infra.object_store import FilesystemObjectStore
from paperwhisperer.infra.stores import build_sqlite_bundle
from paperwhisperer.infra.stores.interfaces import DocumentRecord
from paperwhisperer.ingestion import gateway
from paperwhisperer.ingestion.gateway import PARSE_SUCCESS, ParseResult
from paperwhisperer.ingestion.schema import (
    Element,
    ParsedDocument,
    SourceLocator,
)
from paperwhisperer.rag.evidence import page_renderer_name
from paperwhisperer.sessions.store import PAPER_RECORD_CLEANED_KEYS

#: Two credentials an upstream library might echo back, one for each pass of the
#: redaction filter. ``PATTERN_SENTINEL`` is recognisable by shape — the filter's
#: pattern pass catches it even though the process never held it, which is the
#: case of a presigned URL or a key the user pasted in. ``LITERAL_SENTINEL`` is
#: unrecognisable by shape and is only caught because the process is holding it,
#: which is the case of our own key coming back inside an HTTP error.
PATTERN_SENTINEL = "sk-SENTINEL-ROUTE-TEST"
LITERAL_SENTINEL = "literal-sentinel-value-9f3c-do-not-leak"

# --------------------------------------------------------------------------
# Stubs
# --------------------------------------------------------------------------


class RegisteredParsers:
    """The parsers this deployment claims to have, and what they were asked.

    Two of them, because the reparse menu names two: ``docling`` (local) and
    ``mineru`` (cloud). A fixture that registered only the local one would make
    every cloud test a test of "the adapter is missing" instead of a test of the
    privacy policy, which is the thing those tests are for.
    """

    def __init__(self, names: tuple[str, ...] = ("docling", "mineru_cloud")) -> None:
        self.by_name = {name: FakeParser(name) for name in names}
        self.all = list(self.by_name.values())

    @property
    def docling(self) -> FakeParser:
        return self.by_name["docling"]

    @property
    def mineru(self) -> FakeParser:
        return self.by_name["mineru_cloud"]

    @property
    def calls(self) -> list[dict]:
        return [call for parser in self.all for call in parser.calls]

    def last_call(self) -> dict | None:
        for parser in reversed(self.all):
            if parser.calls:
                return parser.calls[-1]
        return None


class FakeParser:
    """A parser that reads text and records exactly how it was called.

    Recording ``options`` is the point. "Force OCR was routed to the right
    parser" is a claim about two things at once — which parser ran, and with
    which options — and asserting only the outcome would pass if the route had
    quietly ignored the request and got a usable document anyway.
    """

    def __init__(self, name: str = "docling", version: str = "1.2.3") -> None:
        self.name = name
        self.version = version
        self.calls: list[dict] = []

    def supports(self, media_type: str) -> bool:
        return True

    def parse(self, file_path, options=None):
        with open(file_path, encoding="utf-8", errors="ignore") as handle:
            text = handle.read()
        self.calls.append(
            {
                "file_name": os.path.basename(str(file_path)),
                "force_ocr": bool(getattr(options, "force_ocr", False)),
                "options_hash": options.compute_options_hash() if options is not None else "",
            }
        )
        # Dense enough to pass the quality gate. A stub that fails on
        # ``text_density`` is not a neutral stub: the orchestrator would then
        # escalate — a force-OCR retry, then a cloud stage — and every test that
        # meant to exercise one parser would be measuring the escalation policy
        # instead.
        body = (text + " " + "lorem ipsum dolor sit amet " * 200).strip()[:6000]
        half = max(1, len(body) // 2)
        document = ParsedDocument(
            document_id=f"doc_{self.name}",
            source_filename=os.path.basename(str(file_path)),
            media_type="application/pdf",
            page_count=2,
            parser_name=self.name,
            parser_version=self.version,
            elements=[
                Element(
                    role="text",
                    text=body[:half],
                    locator=SourceLocator(kind="pdf_page", page=1),
                ),
                Element(
                    role="text",
                    text=body[half:],
                    locator=SourceLocator(kind="pdf_page", page=2),
                ),
            ],
        )
        return ParseResult(
            status=PARSE_SUCCESS,
            document=document,
            parser_name=self.name,
            parser_version=self.version,
            options_hash=document.options_hash,
        )


class FailingParser(FakeParser):
    """A parser that never produces a document, for the 422 path."""

    def parse(self, file_path, options=None):
        self.calls.append({"file_name": os.path.basename(str(file_path)), "force_ocr": False})
        return ParseResult.failure("模拟的解析失败。")


class StubAnalyzer:
    """Stands in for ``DocumentAnalyzer`` on the Q&A route."""

    instances: list[StubAnalyzer] = []

    def __init__(self, api_key: str, document_content: str = "") -> None:
        self.api_key = api_key
        self.document_content = document_content
        self.qa_document = None
        self.questions: list[str] = []
        StubAnalyzer.instances.append(self)

    def stream_answer_question(self, question, history=None, answer_mode="evidence"):
        self.questions.append(question)
        yield "第一段答案"
        yield "，第二段"
        yield AnswerResult(answer="第一段答案，第二段", confidence=CONFIDENCE_SUPPORTED)


class LeakyAnalyzer(StubAnalyzer):
    """An analyzer whose exception message carries a credential."""

    def stream_answer_question(self, question, history=None, answer_mode="evidence"):
        raise RuntimeError(f"upstream said: invalid key {PATTERN_SENTINEL}")


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def bundle(tmp_path):
    settings = SimpleNamespace(
        STORAGE_SQLITE_PATH=str(tmp_path / "v1.sqlite3"),
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
def v1_client(client, bundle, objects, job_store):
    """The real application, with only the three storage seams replaced.

    The application object is the real one on purpose: registering the routers in
    ``app.py`` is part of this task, and a test that built its own ``FastAPI``
    would keep passing if that registration were missing. Only the store, the
    object store and the job store are overridden, because those are the ones a
    test must never let reach a developer's real data directory.
    """
    overrides = {
        routes_evidence.get_store_bundle: lambda: bundle,
        routes_v1.get_object_store: lambda: objects,
        routes_jobs.get_job_store: lambda: job_store,
    }
    app.dependency_overrides.update(overrides)
    try:
        yield client
    finally:
        for dependency in overrides:
            app.dependency_overrides.pop(dependency, None)


def _register(monkeypatch, registered: RegisteredParsers) -> RegisteredParsers:
    monkeypatch.setattr(gateway, "build_default_parsers", lambda: registered.all)
    monkeypatch.setattr(routes_v1, "build_default_parsers", lambda: registered.all)
    return registered


@pytest.fixture
def parser(monkeypatch):
    """A registered parser set, with the parse cache off so calls really run.

    The cache is off because a cache *hit* is the one path on which the fake
    parser is not called, and a test asserting "this parser ran" would then pass
    on the strength of a file another test wrote.
    """
    monkeypatch.setattr(config, "PARSER_CACHE_ENABLED", False)
    monkeypatch.setattr(config, "PARSER_BACKEND", "gateway")
    return _register(monkeypatch, RegisteredParsers())


@pytest.fixture
def local_only_parsers(monkeypatch):
    """A deployment whose cloud adapter is not installed."""
    monkeypatch.setattr(config, "PARSER_CACHE_ENABLED", False)
    monkeypatch.setattr(config, "PARSER_BACKEND", "gateway")
    return _register(monkeypatch, RegisteredParsers(("docling",)))


def upload(client, name: str = "paper.txt", body: bytes = b"first page text second page text"):
    """POST a multipart upload the way a browser would."""
    return client.post(
        "/v1/documents",
        files={"file": (name, io.BytesIO(body), "text/plain")},
    )


@pytest.fixture
def document_id(v1_client, parser):
    """One uploaded, parsed, indexed document."""
    response = upload(v1_client)
    assert response.status_code == 201, response.text
    return response.json()["document"]["document_id"]


# --------------------------------------------------------------------------
# The frozen field names
# --------------------------------------------------------------------------


def _response_schema(openapi, operation, status):
    """The declared schema for one response, following a ``$ref`` if needed."""
    response = openapi["paths"][operation["_path"]][operation["_method"]]["responses"][status]
    schema = response.get("content", {}).get("application/json", {}).get("schema", {})
    if "$ref" in schema:
        return openapi["components"]["schemas"][schema["$ref"].rsplit("/", 1)[-1]]
    return schema


def test_document_resource_field_names_are_frozen(v1_client, parser):
    """The document resource's keys are a promise; assert the whole set."""
    document = upload(v1_client).json()["document"]
    assert set(document) == {
        "document_id",
        "source_filename",
        "media_type",
        "sha256",
        "page_count",
        "status",
        "parser_name",
        "parser_version",
        "options_hash",
        "index_version",
        "element_count",
        "chunk_count",
        "char_count",
        "original_available",
        "created_at",
        "updated_at",
        "version",
        "parse",
        "links",
    }
    assert set(document["links"]) == {"self", "metadata", "pages", "page_image", "qa", "reparse"}


def test_parse_report_field_names_are_frozen(v1_client, parser):
    """The provenance block is shared with the legacy analysis payload.

    Pinned against ``PARSE_REPORT_KEYS`` rather than a literal so that a
    deliberate addition is a one-line change in two places — the projection and
    the test — instead of a test that has to be rewritten or, worse, one that is
    loosened to ``issubset``.
    """
    document = upload(v1_client).json()["document"]
    assert set(document["parse"]) == set(routes_v1.PARSE_REPORT_KEYS)
    assert set(routes_v1.ParseReport.model_fields) == set(routes_v1.PARSE_REPORT_KEYS)

    metadata = v1_client.get(f"/v1/documents/{document['document_id']}/metadata")
    assert metadata.status_code == 200
    assert set(metadata.json()) == set(routes_v1.PARSE_REPORT_KEYS)


def test_job_state_field_names_are_frozen(v1_client, parser, job_store):
    """The polling shape and the SSE shape are different, and both are pinned."""
    job_id = upload(v1_client).json()["job"]["job_id"]

    state = v1_client.get(f"/v1/jobs/{job_id}").json()
    assert set(state) == set(routes_jobs.JobState.model_fields)

    # The stream's event is deliberately *not* the same shape: no result, no
    # error, no ids for the owning document. A subscriber rebuilds the truth from
    # these plus one GET; it cannot rebuild the result, by design.
    with v1_client.stream("GET", f"/v1/jobs/{job_id}/stream") as response:
        payload = _first_state_event(response)
    assert set(payload) == set(routes_jobs.JobEventState.model_fields)
    assert set(payload) != set(state)


def test_openapi_schema_matches_what_the_routes_return(v1_client, parser):
    """A contract document that does not match the code is worse than none.

    Each response schema is compared against the keys a live response actually
    carries, so a field added to a route without being added to the model fails
    here. ``response_model`` makes FastAPI *filter* undeclared fields, so this is
    the only thing standing between a projection and a silently dropped key.
    """
    document = upload(v1_client).json()
    document_id = document["document"]["document_id"]

    cases = [
        ({"_path": "/v1/documents", "_method": "get"}, "200", lambda: v1_client.get("/v1/documents")),
        (
            {"_path": "/v1/documents", "_method": "post"},
            "201",
            lambda: upload(v1_client),
        ),
        (
            {"_path": "/v1/documents/{document_id}", "_method": "get"},
            "200",
            lambda: v1_client.get(f"/v1/documents/{document_id}"),
        ),
        (
            {"_path": "/v1/documents/{document_id}/pages", "_method": "get"},
            "200",
            lambda: v1_client.get(f"/v1/documents/{document_id}/pages"),
        ),
        (
            {"_path": "/v1/documents/{document_id}/reparse", "_method": "post"},
            "200",
            lambda: v1_client.post(f"/v1/documents/{document_id}/reparse", json={"force_ocr": True}),
        ),
        (
            {"_path": "/v1/jobs", "_method": "get"},
            "200",
            lambda: v1_client.get("/v1/jobs"),
        ),
    ]

    openapi = app.openapi()
    for operation, status, call in cases:
        response = call()
        assert response.status_code == int(status), (operation, response.text)
        schema = _response_schema(openapi, operation, status)
        declared = set(schema.get("properties") or {})
        assert declared, f"{operation} declares no response schema"
        assert set(response.json()) == declared, operation


# --------------------------------------------------------------------------
# The documents surface
# --------------------------------------------------------------------------


def test_upload_creates_a_document_and_a_job(v1_client, parser):
    """The create is a document *and* the job that produced it."""
    response = upload(v1_client)
    assert response.status_code == 201
    body = response.json()
    document = body["document"]

    assert document["source_filename"] == "paper.txt"
    assert document["page_count"] == 2
    assert document["element_count"] == 2
    assert document["chunk_count"] >= 1
    assert document["original_available"] is True
    assert document["parse"]["parsed_by"] == "docling"
    assert document["parse"]["touched_network"] is False
    assert body["original_stored"] is True

    # The job is a real resource, not a field the route invented: asking for it
    # by id is the same request a browser would make after a refresh.
    job = v1_client.get(f"/v1/jobs/{body['job']['job_id']}")
    assert job.status_code == 200
    assert job.json()["state"] in {"completed", "partial"}
    assert job.json()["result"]["document_id"] == document["document_id"]


def test_upload_stores_the_original_so_a_reparse_can_read_it(v1_client, parser, objects):
    """The legacy route deletes its upload; this one must not."""
    from paperwhisperer.infra.object_store import LAYER_ORIGINAL

    document = upload(v1_client).json()["document"]
    reference = objects.find_by_digest(LAYER_ORIGINAL, document["sha256"])
    assert reference is not None
    assert objects.get(LAYER_ORIGINAL, reference.key).startswith(b"first page text")


def test_upload_rejects_a_missing_file(v1_client):
    response = v1_client.post("/v1/documents", data={})
    assert response.status_code == 400
    assert response.json()["code"] == "missing_file"


def test_upload_rejects_an_unsupported_type(v1_client):
    response = upload(v1_client, name="payload.exe", body=b"MZ")
    assert response.status_code == 400
    assert response.json()["code"] == "unsupported_file_type"


def test_upload_rejects_a_filename_with_a_path(v1_client):
    """The contract layer catches what the extension check cannot."""
    response = v1_client.post(
        "/v1/documents",
        files={"file": ("../../escape.txt", io.BytesIO(b"body"), "text/plain")},
    )
    assert response.status_code == 400
    assert response.json()["code"] == "invalid_upload_metadata"


def test_a_malformed_document_id_is_a_400_not_a_404(v1_client):
    response = v1_client.get("/v1/documents/not..a..document")
    assert response.status_code == 400
    assert response.json()["code"] == "malformed_id"


def test_an_unknown_document_is_a_404(v1_client):
    response = v1_client.get("/v1/documents/doc_missing")
    assert response.status_code == 404
    assert response.json()["code"] == "document_not_found"


def test_pages_list_the_document_and_advertise_the_missing_image_route(
    v1_client, parser, document_id
):
    """The 501 is part of the contract, and the listing declares it up front."""
    response = v1_client.get(f"/v1/documents/{document_id}/pages")
    assert response.status_code == 200
    body = response.json()

    assert set(body) == {
        "document_id",
        "media_type",
        "page_count",
        "locator_kind",
        "page_image",
        "pages",
    }
    assert body["locator_kind"] == "pdf_page"
    assert [page["page"] for page in body["pages"]] == [1, 2]
    assert set(body["pages"][0]) == {
        "page",
        "element_count",
        "char_count",
        "has_table",
        "has_formula",
        "has_figure",
    }
    assert body["page_image"]["url_template"] == (
        f"/v1/documents/{document_id}/pages/{{page}}/image"
    )
    # The build's capability, read from the same function the route acts on, and
    # the 501 it would answer with if the capability were missing.
    assert body["page_image"]["renderer"] == page_renderer_name()
    assert body["page_image"]["available"] is (page_renderer_name() != "none")
    assert body["page_image"]["unavailable_status"] == 501
    assert body["page_image"]["unavailable_code"] == "page_render_unavailable"
    assert body["page_image"]["unavailable_message"]


def test_the_page_image_route_still_answers_501(v1_client, parser, document_id):
    """The evidence router owns that URL; this surface must not have replaced it.

    Asserted through the real application because the claim is about routing: a
    second copy of the route registered here would shadow it and answer 200 or
    404, and the reader would be told "no highlight" when the truth is "no
    renderer".
    """
    response = v1_client.get(f"/v1/documents/{document_id}/pages/1/image")
    if page_renderer_name() == "none":
        assert response.status_code == 501
        assert response.json()["code"] == "page_render_unavailable"
        assert response.json()["error"]
    else:
        # The renderer is installed here, so the route is allowed to succeed —
        # and what is asserted either way is that the code the front end mirrors
        # is the one it answers with when it cannot.
        assert response.status_code in {200, 501}


def test_a_document_record_with_no_pages_lists_no_pages(v1_client, bundle):
    """An empty list is honest; an invented page 1 is not."""
    document = ParsedDocument(document_id="doc_empty", source_filename="x.txt")
    bundle.metadata.upsert_document(
        DocumentRecord(
            document_id="doc_empty",
            document=document,
            source_filename="x.txt",
            media_type="text/plain",
        )
    )
    body = v1_client.get("/v1/documents/doc_empty/pages").json()
    assert body["pages"] == []
    assert body["locator_kind"] == ""


# --------------------------------------------------------------------------
# Register
# --------------------------------------------------------------------------


def _canonical_payload(document_id: str = "doc_external") -> dict:
    return ParsedDocument(
        document_id=document_id,
        source_filename="outside.pdf",
        media_type="application/pdf",
        page_count=1,
        elements=[
            Element(
                role="text",
                text="externally parsed",
                locator=SourceLocator(kind="pdf_page", page=1),
            )
        ],
    ).to_dict()


def test_register_adopts_a_parsed_document(v1_client):
    response = v1_client.post(
        "/v1/documents/register", json={"document": _canonical_payload()}
    )
    assert response.status_code == 201
    body = response.json()
    assert body["document"]["document_id"] == "doc_external"
    assert body["chunk_count"] >= 1
    # A document nobody uploaded has no original, and says so rather than
    # pretending: `used_cloud: false` for a parse that happened somewhere unknown
    # would be a claim this service cannot make.
    assert body["document"]["original_available"] is False
    assert body["document"]["parse"]["parsed_by"] == ""
    assert "未记录" in body["document"]["parse"]["summary"]


def test_register_rejects_a_second_registration(v1_client):
    payload = {"document": _canonical_payload()}
    assert v1_client.post("/v1/documents/register", json=payload).status_code == 201
    again = v1_client.post("/v1/documents/register", json=payload)
    assert again.status_code == 409
    assert again.json()["code"] == "document_exists"


def test_register_rejects_a_payload_that_is_not_a_parsed_document(v1_client):
    """``from_dict`` fills defaults in, so the id has to be demanded explicitly.

    Without that check ``{"nope": 1}`` would be accepted, given a freshly minted
    id, and registered — and the client would be holding an id it never sent.
    """
    response = v1_client.post("/v1/documents/register", json={"document": {"nope": 1}})
    assert response.status_code == 400
    assert response.json()["code"] == "invalid_document_payload"
    assert "document_id" in response.json()["error"]


def test_register_rejects_a_schema_version_it_cannot_read(v1_client):
    """A locator from a schema this build does not understand may mean anything."""
    payload = _canonical_payload()
    payload["schema_version"] = "2.0"
    response = v1_client.post("/v1/documents/register", json={"document": payload})
    assert response.status_code == 400
    assert response.json()["code"] == "invalid_document_payload"


def test_register_rejects_a_body_that_is_not_an_object(v1_client):
    response = v1_client.post(
        "/v1/documents/register",
        content=b"[1, 2, 3]",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 400
    assert response.json()["code"] == "json_not_object"


# --------------------------------------------------------------------------
# Reparse, and the privacy gate
# --------------------------------------------------------------------------


def test_reparse_routes_force_ocr_to_the_named_parser(v1_client, parser, document_id):
    """Both halves of "routed to the right parser": the parser and the options."""
    before = len(parser.calls)
    response = v1_client.post(
        f"/v1/documents/{document_id}/reparse",
        json={"parser": "docling_ocr", "force_ocr": True},
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert len(parser.calls) == before + 1
    assert parser.docling.calls, "the named parser did not run"
    assert parser.mineru.calls == [], "the cloud parser ran on a local request"
    assert parser.last_call()["force_ocr"] is True
    assert body["parse"]["parsed_by"] == "docling"
    assert body["document"]["document_id"] == document_id
    assert body["cloud_requested"] is False


def test_reparse_without_force_ocr_does_not_force_it(v1_client, parser, document_id):
    """A checkbox the user did not tick must not be honoured anyway."""
    v1_client.post(f"/v1/documents/{document_id}/reparse", json={"parser": "docling"})
    assert parser.last_call()["force_ocr"] is False


def test_reparse_keeps_the_document_id_and_bumps_the_version(v1_client, parser, document_id):
    """A reparse replaces this document, it does not mint a new one.

    Minting a new id would make the operation the caller asked for impossible to
    follow, and every citation already held would point at a document that can no
    longer be named.
    """
    before = v1_client.get(f"/v1/documents/{document_id}").json()["version"]
    body = v1_client.post(
        f"/v1/documents/{document_id}/reparse", json={"force_ocr": True}
    ).json()
    assert body["document_id"] == document_id
    assert body["document"]["document_id"] == document_id
    assert body["document"]["version"] > before


def test_reparse_refuses_a_cloud_parser_under_local_only(v1_client, parser, document_id,
                                                         monkeypatch):
    """The refusal names *which* policy refused, and nothing is sent."""
    monkeypatch.setattr(config, "DOCUMENT_PRIVACY_MODE", "local_only")
    monkeypatch.setattr(config, "DOCUMENT_CLOUD_CONSENT_GRANTED", True)
    before = len(parser.calls)

    response = v1_client.post(
        f"/v1/documents/{document_id}/reparse", json={"parser": "mineru_cloud"}
    )
    assert response.status_code == 403
    body = response.json()
    assert body["code"] == "privacy_mode_local_only"
    assert "local_only" in body["error"]
    assert len(parser.calls) == before, "the document was read before the refusal"


def test_reparse_refuses_without_consent_when_the_mode_allows(
    v1_client, parser, document_id, monkeypatch
):
    """The second of the two factors is refused with a different code.

    Same status, different code, because the fix is different: an administrator
    changes the mode, the user grants consent. One code would send people to
    change the setting that was never the problem.
    """
    monkeypatch.setattr(config, "DOCUMENT_PRIVACY_MODE", "allow_cloud")
    monkeypatch.setattr(config, "DOCUMENT_CLOUD_CONSENT_GRANTED", False)

    response = v1_client.post(
        f"/v1/documents/{document_id}/reparse", json={"parser": "mineru_cloud"}
    )
    assert response.status_code == 403
    assert response.json()["code"] == "consent_not_granted"


def test_reparse_accepts_a_cloud_parser_when_the_policy_allows(
    v1_client, parser, document_id, monkeypatch
):
    """Accepted, and honest about what was actually done.

    The local chain wins when it can, so ``used_cloud`` is false and the notice
    says the document was not uploaded. Reporting ``used_cloud: true`` to match
    the request would be inventing the one fact this product's provenance exists
    to keep true.
    """
    monkeypatch.setattr(config, "DOCUMENT_PRIVACY_MODE", "allow_cloud")
    monkeypatch.setattr(config, "DOCUMENT_CLOUD_CONSENT_GRANTED", True)

    response = v1_client.post(
        f"/v1/documents/{document_id}/reparse", json={"parser": "mineru_cloud"}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["cloud_requested"] is True
    assert body["parse"]["used_cloud"] is False
    assert body["parse"]["touched_network"] is False
    assert "未上传" in body["notice"]
    # Stronger than the payload: the cloud parser was never invoked at all. The
    # escalation only reaches it when the local chain produced nothing usable,
    # and here it did — so an allowed cloud request is not a licence to upload a
    # document that was already read.
    assert parser.mineru.calls == []


def test_reparse_rejects_an_unknown_parser(v1_client, parser, document_id):
    response = v1_client.post(
        f"/v1/documents/{document_id}/reparse", json={"parser": "gpt-vision"}
    )
    assert response.status_code == 400
    assert response.json()["code"] == "unknown_parser"


def test_reparse_reports_an_unavailable_parser_as_409(v1_client, document_id, monkeypatch):
    """A parser the rollback flag has switched off is a capability answer.

    Not a 422 "no parsers available": the request was well formed, the deployment
    simply cannot honour it, and the message says which setting to change.
    """
    monkeypatch.setattr(config, "PARSER_BACKEND", "legacy")
    _register(monkeypatch, RegisteredParsers())

    response = v1_client.post(f"/v1/documents/{document_id}/reparse", json={"parser": "docling"})
    assert response.status_code == 409
    assert response.json()["code"] == "parser_unavailable"
    assert "PARSER_BACKEND" in response.json()["error"]


def test_reparse_of_a_registered_document_is_a_definite_409(v1_client):
    """No original means 409 with a code, not a 500 and not a silent success."""
    v1_client.post("/v1/documents/register", json={"document": _canonical_payload()})
    response = v1_client.post("/v1/documents/doc_external/reparse", json={"force_ocr": True})
    assert response.status_code == 409
    assert response.json()["code"] == "original_unavailable"
    assert "重新上传" in response.json()["error"]


def test_a_parse_that_produces_nothing_is_422_not_500(v1_client, monkeypatch):
    """A failure the pipeline understood is a 422 with the pipeline's own words."""
    monkeypatch.setattr(config, "PARSER_CACHE_ENABLED", False)
    monkeypatch.setattr(config, "PARSER_BACKEND", "gateway")
    failing = RegisteredParsers(("docling",))
    failing.by_name["docling"] = FailingParser("docling")
    failing.all = list(failing.by_name.values())
    _register(monkeypatch, failing)

    response = upload(v1_client)
    assert response.status_code == 422
    assert response.json()["code"] == "document_parse_failed"
    assert "模拟的解析失败" in response.json()["error"]


# --------------------------------------------------------------------------
# Delete
# --------------------------------------------------------------------------


def test_delete_removes_the_document_its_index_and_its_original(v1_client, parser, bundle):
    document = upload(v1_client).json()["document"]
    response = v1_client.delete(f"/v1/documents/{document['document_id']}")
    assert response.status_code == 200
    assert response.json() == {
        "document_id": document["document_id"],
        "deleted": True,
        "original_deleted": True,
    }
    assert bundle.metadata.get_document(document["document_id"]) is None
    assert bundle.metadata.list_chunks(document["document_id"]) == []
    assert v1_client.get(f"/v1/documents/{document['document_id']}").status_code == 404


def test_delete_refuses_while_a_job_for_the_document_is_running(
    v1_client, parser, document_id, job_store
):
    """Otherwise the job's next stage would write the document back."""
    job_store.submit("document_reparse", {}, document_id=document_id)
    response = v1_client.delete(f"/v1/documents/{document_id}")
    assert response.status_code == 409
    assert response.json()["code"] == "document_busy"
    assert v1_client.get(f"/v1/documents/{document_id}").status_code == 200


# --------------------------------------------------------------------------
# Jobs: state, resync, cancel
# --------------------------------------------------------------------------


def _sse_events(body: str) -> list[tuple[str, str, dict]]:
    """Decode an SSE body into ``[(event_id, event_name, payload), ...]``.

    The id is returned separately because it is the contract: it is what the
    client sends back as ``Last-Event-ID``, and a test that read the sequence out
    of the JSON payload instead would pass even if the ``id:`` line had been
    dropped from the wire.
    """
    events: list[tuple[str, str, dict]] = []
    for block in body.split("\n\n"):
        if not block.strip():
            continue
        event_id = ""
        name = ""
        data = ""
        for line in block.splitlines():
            if line.startswith("id:"):
                event_id = line[len("id:") :].strip()
            elif line.startswith("event:"):
                name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data = line[len("data:") :].strip()
        if not name:
            continue
        events.append((event_id, name, json.loads(data) if data else {}))
    return events


def _first_state_event(response) -> dict:
    """Read the first ``state`` event out of a streaming response, then stop.

    Skips the ``snapshot``, which is a different shape on purpose: a helper that
    returned whichever event came first would silently compare the polling shape
    against the streaming shape and pass or fail for the wrong reason.
    """
    name = ""
    for line in response.iter_lines():
        if line.startswith("event:"):
            name = line[len("event:") :].strip()
        elif line.startswith("data:") and name == "state":
            return json.loads(line[len("data:") :])
    return {}


def _stream_then_disconnect(
    application, path: str, *, state_events: int, headers: dict | None = None, timeout: float = 15.0
) -> int:
    """Run the ASGI application directly and drop the connection mid-stream.

    ``TestClient`` cannot express the one thing this test needs. Its transport
    runs the whole application to completion before returning a response, and its
    ``receive`` only produces ``http.disconnect`` *after* the response has
    finished — so "the client went away while the stream was still open" is not a
    question it can be asked, and a test that pretended otherwise would assert
    something the harness never did.

    Driving the ASGI callable directly is not a stand-in: the same router, the
    same ``StreamingResponse`` and the same cancellation run, and the
    ``http.disconnect`` is a real message delivered to a real listener. Returns
    the highest sequence the client managed to read before the connection went.
    """
    state_chunks: list[str] = []

    def _highest_state_sequence(text: str) -> int:
        highest = 0
        for block in text.split("\n\n"):
            if "event: state" not in block:
                continue
            match = re.search(r'"sequence":\s*(\d+)', block)
            if match:
                highest = max(highest, int(match.group(1)))
        return highest

    async def drive() -> int:
        disconnect = anyio.Event()
        chunks: list[str] = []

        async def receive() -> dict:
            await disconnect.wait()
            return {"type": "http.disconnect"}

        async def send(message: dict) -> None:
            if message.get("type") != "http.response.body":
                return
            body = message.get("body", b"") or b""
            chunks.append(body.decode("utf-8", "replace"))
            if b"event: state" in body:
                state_chunks.append(body.decode("utf-8", "replace"))
            if len(state_chunks) >= state_events:
                # The reader has what it needed and closes the tab.
                disconnect.set()

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode("utf-8"),
            "root_path": "",
            "query_string": b"",
            "headers": [
                (name.lower().encode("latin-1"), value.encode("latin-1"))
                for name, value in (headers or {}).items()
            ],
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
            "state": {},
        }
        # The timeout is a backstop, not the mechanism: the disconnect above
        # ends the stream, and a test that needed the backstop would be testing
        # the backstop.
        with anyio.move_on_after(timeout):
            await application(scope, receive, send)
        return _highest_state_sequence("".join(chunks))

    return anyio.run(drive)


def test_a_job_streams_a_snapshot_then_its_transitions(v1_client, parser):
    """Snapshot first, then every event, ending on the terminal one.

    The ``id:`` line is asserted alongside the event names because it is the
    other half of the resync contract: without it the sequence exists only inside
    a JSON body, and a browser's ``EventSource`` has nowhere to put it.
    """
    job_id = upload(v1_client).json()["job"]["job_id"]
    events = _sse_events(v1_client.get(f"/v1/jobs/{job_id}/stream").text)

    names = [name for _id, name, _payload in events]
    assert names[0] == "snapshot"
    assert "state" in names[1:]
    assert names[-1] == "state"

    states = [payload for _id, name, payload in events if name == "state"]
    assert [event["sequence"] for event in states] == sorted(
        event["sequence"] for event in states
    )
    assert states[-1]["terminal"] is True
    assert states[-1]["state"] in {"completed", "partial"}

    # Every state event carries its sequence as the SSE id, and the snapshot
    # carries none, because a snapshot is an answer and not an event.
    assert [event_id for event_id, name, _ in events if name == "state"] == [
        str(event["sequence"]) for event in states
    ]
    assert [event_id for event_id, name, _ in events if name == "snapshot"] == [""]


def test_a_disconnected_client_resyncs_with_last_event_id(v1_client, job_store):
    """The Phase 2 exit criterion, over the wire, with a real disconnection.

    Read the stream until the job has been seen in ``parsing``, drop the
    connection, let the work continue, then reconnect with ``Last-Event-ID`` —
    the header a browser's own ``EventSource`` sends on a reconnect — and assert
    the reconnect reports the current state *and* only the events that happened
    after the cursor.

    The disconnection is real, not a simulation: see
    :func:`_stream_then_disconnect`, which drives the ASGI application directly
    because ``TestClient`` cannot express "the client went away while the stream
    was still open" (it runs the whole application to completion and only then
    produces ``http.disconnect``). The code under test is the same either way:
    the same router, the same ``StreamingResponse``, the same cancellation.
    """
    job_store.submit("document_ingest", {"document_id": "doc_x"})
    job_id = job_store.list_jobs(kind="document_ingest")[0].job_id
    job_store.transition(job_id, J.STATE_PARSING, message="正在解析文档。")

    seen_through = _stream_then_disconnect(app, f"/v1/jobs/{job_id}/stream", state_events=2)
    assert seen_through == 2, "the first connection did not reach the parsing event"

    # The client is gone; the job is not. This is the window the exit criterion
    # is about.
    job_store.transition(job_id, J.STATE_QUALITY_CHECK, message="正在评估解析质量。")
    job_store.transition(job_id, J.STATE_COMPLETED, message="任务已完成。")

    resumed = v1_client.get(
        f"/v1/jobs/{job_id}/stream", headers={"Last-Event-ID": str(seen_through)}
    )
    assert resumed.status_code == 200
    events = _sse_events(resumed.text)
    by_name = {name: payload for _id, name, payload in events}

    # The snapshot says where the job is now, and is not itself an event.
    assert by_name["snapshot"]["state"] == J.STATE_COMPLETED
    assert by_name["snapshot"]["terminal"] is True
    assert [event_id for event_id, name, _ in events if name == "snapshot"] == [""]

    # And the events are exactly the ones the client missed — no gap, no replay.
    states = [payload for _id, name, payload in events if name == "state"]
    assert [event["sequence"] for event in states] == [seen_through + 1, seen_through + 2]
    assert [event["state"] for event in states] == [J.STATE_QUALITY_CHECK, J.STATE_COMPLETED]
    assert states[-1]["terminal"] is True
    assert states[0]["previous_state"] == J.STATE_PARSING


def test_a_reconnecting_client_can_ask_for_the_whole_trail(v1_client, job_store):
    """``Last-Event-ID: 0`` is a client that missed everything, not one that
    saw nothing because the job was new — the two must be indistinguishable to
    the server, because from the trail they are."""
    job_store.submit("document_ingest", {})
    job_id = job_store.list_jobs(kind="document_ingest")[0].job_id
    job_store.transition(job_id, J.STATE_PARSING)
    job_store.transition(job_id, J.STATE_COMPLETED)

    events = _sse_events(
        v1_client.get(f"/v1/jobs/{job_id}/stream", headers={"Last-Event-ID": "0"}).text
    )
    states = [payload for _id, name, payload in events if name == "state"]
    assert [event["sequence"] for event in states] == [1, 2, 3]


def test_the_event_feed_resyncs_from_a_sequence(v1_client, job_store):
    """The polling half of the same contract, and what a test can read without SSE."""
    job_store.submit("document_ingest", {})
    job_id = job_store.list_jobs(kind="document_ingest")[0].job_id
    job_store.transition(job_id, J.STATE_PARSING)
    job_store.transition(job_id, J.STATE_COMPLETED)

    everything = v1_client.get(f"/v1/jobs/{job_id}/events").json()
    assert everything["since"] == 0
    assert [event["sequence"] for event in everything["events"]] == [1, 2, 3]
    assert everything["sequence"] == 3

    tail = v1_client.get(f"/v1/jobs/{job_id}/events?since_sequence=2").json()
    assert [event["sequence"] for event in tail["events"]] == [3]
    assert tail["sequence"] == 3


def test_the_event_feed_rejects_a_negative_cursor(v1_client, job_store):
    job_store.submit("document_ingest", {})
    job_id = job_store.list_jobs(kind="document_ingest")[0].job_id
    response = v1_client.get(f"/v1/jobs/{job_id}/events?since_sequence=-1")
    assert response.status_code == 400
    assert response.json()["code"] == "invalid_since"


def test_a_non_numeric_last_event_id_is_the_clients_bug(v1_client, job_store):
    """Rewinding to 0 would replay everything and hide which events were seen."""
    job_store.submit("document_ingest", {})
    job_id = job_store.list_jobs(kind="document_ingest")[0].job_id
    response = v1_client.get(
        f"/v1/jobs/{job_id}/stream", headers={"Last-Event-ID": "yesterday"}
    )
    assert response.status_code == 400
    assert response.json()["code"] == "invalid_since"


def test_the_stream_of_an_unknown_job_is_a_404(v1_client):
    response = v1_client.get("/v1/jobs/deadbeef/stream")
    assert response.status_code == 404
    assert response.json()["code"] == "job_not_found"


def test_a_malformed_job_id_is_a_400(v1_client):
    response = v1_client.get("/v1/jobs/not..a..job")
    assert response.status_code == 400
    assert response.json()["code"] == "malformed_id"


def test_job_listing_reports_the_queue_shape(v1_client, job_store):
    job_store.submit("document_ingest", {"document_id": "doc_1"})
    job_store.submit("document_ingest", {"document_id": "doc_2"})
    body = v1_client.get("/v1/jobs?kind=document_ingest").json()
    assert body["count"] == 2
    assert body["by_state"].get("queued") == 2
    assert all(item["kind"] == "document_ingest" for item in body["items"])


def test_job_listing_rejects_an_unknown_state(v1_client):
    response = v1_client.get("/v1/jobs?state=vibing")
    assert response.status_code == 400
    assert response.json()["code"] == "invalid_state_filter"


def test_cancelling_a_queued_job_is_accepted_and_idempotent(v1_client, job_store):
    """A job nobody started needs no worker's cooperation to stop."""
    job_id = job_store.submit("document_ingest", {}).job_id
    first = v1_client.post(f"/v1/jobs/{job_id}/cancel")
    assert first.status_code == 200
    assert first.json()["accepted"] is True
    assert first.json()["cancel_requested"] is True
    assert set(first.json()) == {"job_id", "state", "cancel_requested", "accepted", "message", "job"}

    again = v1_client.post(f"/v1/jobs/{job_id}/cancel")
    assert again.status_code == 200
    assert again.json()["cancel_requested"] is True


def test_cancelling_an_unknown_job_is_a_404(v1_client):
    response = v1_client.post("/v1/jobs/nosuchjob/cancel")
    assert response.status_code == 404
    assert response.json()["code"] == "job_not_found"


def test_retry_only_applies_to_a_failed_job(v1_client, job_store):
    job_id = job_store.submit("document_ingest", {}).job_id
    refused = v1_client.post(f"/v1/jobs/{job_id}/retry")
    assert refused.status_code == 409
    assert refused.json()["code"] == "job_not_retryable"

    job_store.transition(job_id, J.STATE_PARSING)
    job_store.transition(
        job_id, J.STATE_FAILED, message="模拟失败。", patch={"error_message": "模拟失败"}
    )
    accepted = v1_client.post(f"/v1/jobs/{job_id}/retry")
    assert accepted.status_code == 200
    assert accepted.json()["state"] == J.STATE_QUEUED


# --------------------------------------------------------------------------
# Questions
# --------------------------------------------------------------------------


def test_the_document_qa_streams_start_deltas_and_done(v1_client, parser, document_id,
                                                       monkeypatch):
    """Content, not state — the distinction from the job feed is the point."""
    monkeypatch.setattr(routes_v1, "build_qa_analyzer", StubAnalyzer)
    response = v1_client.post(
        f"/v1/documents/{document_id}/qa",
        json={"question": "这篇论文的方法是什么？", "api_key": "test-key"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    blocks = [block for block in response.text.split("\n\n") if block.strip()]
    names = [block.split("event: ", 1)[1].split("\n", 1)[0] for block in blocks]
    assert names == ["start", "delta", "delta", "done"]

    start = json.loads(blocks[0].split("data: ", 1)[1])
    assert start["document_id"] == document_id
    assert start["locator_kind"] == "pdf_page"

    done = json.loads(blocks[-1].split("data: ", 1)[1])
    assert done["answer"] == "第一段答案，第二段"
    assert done["confidence"] == CONFIDENCE_SUPPORTED
    # The whole AnswerResult, so the claims and citations survive the stream.
    assert {"answer", "claims", "citations", "confidence"} <= set(done)


def test_the_qa_route_reports_the_missing_inputs(v1_client, parser, document_id, monkeypatch):
    monkeypatch.setattr(routes_v1, "build_qa_analyzer", StubAnalyzer)
    base = f"/v1/documents/{document_id}/qa"

    assert v1_client.post(base, json={"api_key": "k"}).json()["code"] == "missing_question"
    assert v1_client.post(base, json={"question": "q"}).json()["code"] == "missing_api_key"
    missing = v1_client.post("/v1/documents/doc_nope/qa", json={"question": "q", "api_key": "k"})
    assert missing.status_code == 404
    assert missing.json()["code"] == "document_not_found"


def test_the_qa_route_rejects_a_session_it_cannot_prove(
    runtime_dirs, v1_client, parser, document_id, session_factory, monkeypatch
):
    """A session that exists but whose token is wrong is 403, not a silent 404.

    A 404 here would be a *different* bug answered correctly: a client with a
    stale token would be told the session is gone, and would re-analyse a
    document it already had, rather than being told the token is the problem.
    """
    monkeypatch.setattr(routes_v1, "build_qa_analyzer", StubAnalyzer)
    session_id, _token = session_factory("document text")

    response = v1_client.post(
        f"/v1/documents/{document_id}/qa",
        json={
            "question": "q",
            "api_key": "k",
            "session_id": session_id,
            "session_token": "wrong",
        },
    )
    assert response.status_code == 403
    assert response.json()["code"] == "session_token_invalid"


def test_the_qa_route_reports_a_session_that_does_not_exist(v1_client, parser, document_id):
    response = v1_client.post(
        f"/v1/documents/{document_id}/qa",
        json={"question": "q", "api_key": "k", "session_id": "nosuchsession"},
    )
    assert response.status_code == 404
    assert response.json()["code"] == "session_not_found"


def test_an_exception_message_never_reaches_the_browser(v1_client, parser, document_id,
                                                        monkeypatch):
    """Both passes of the redaction filter, on this surface's two error paths.

    The text of an exception is written by whatever raised it, which for a cloud
    parse or a model call is a library that has just handled a credential. Two
    shapes are checked because the filter has two passes and either alone would be
    half a defence: one catches a credential by its *shape* even when the process
    never held it, the other catches our own key coming back inside an HTTP error
    however it happens to be spelled.

    The ``code`` is asserted in the same breath, because a client that could no
    longer branch on what happened would be a bad trade for a client that cannot
    see a key.
    """
    monkeypatch.setenv("MINERU_API_TOKEN", LITERAL_SENTINEL)
    monkeypatch.setattr(routes_v1, "build_qa_analyzer", LeakyAnalyzer)

    body = v1_client.post(
        f"/v1/documents/{document_id}/qa", json={"question": "q", "api_key": "k"}
    ).text
    assert PATTERN_SENTINEL not in body
    assert "streaming_question_failed" in body

    def _leaky_search(query, limit=None):
        raise RuntimeError(f"upstream said: token {LITERAL_SENTINEL} rejected")

    monkeypatch.setattr(routes_v1, "search_papers", _leaky_search)
    response = v1_client.post("/v1/search/papers", json={"query": "x"})
    assert response.status_code == 500
    assert LITERAL_SENTINEL not in response.text
    assert response.json()["code"] == "search_failed"


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------


@pytest.fixture
def fake_search(monkeypatch):
    """Replace the network-bound search with a fixed, known-shaped result."""
    calls: list[dict] = []

    def _search(query, limit=None):
        calls.append({"query": query, "limit": limit})
        return {
            "query": query,
            "items": [
                {
                    "source": "arXiv",
                    "paper_id": "2401.00001",
                    "title": "A Paper",
                    "abstract": "An abstract.",
                    "authors": [{"name": "A. Author"}],
                    "year": 2024,
                    "venue": "arXiv",
                    "url": "https://example.com/a",
                    "pdf_url": "https://example.com/a.pdf",
                    "saved_at": "2026-01-01T00:00:00",
                    "doi": "",
                    "arxiv_id": "2401.00001",
                    "s2_paper_id": "",
                    "openalex_id": "",
                    "identity_key": "arxiv:2401.00001",
                }
            ],
            "errors": ["Semantic Scholar: rate limited"],
            "possible_duplicates": [{"left": "A Paper", "right": "A Paper "}],
            "merged": [],
        }

    monkeypatch.setattr(routes_v1, "search_papers", _search)
    return calls


def test_search_returns_the_frozen_shape(v1_client, fake_search):
    response = v1_client.post("/v1/search/papers", json={"query": "retrieval"})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == set(routes_v1.PaperSearchResponse.model_fields)
    assert body["query"] == "retrieval"
    assert body["original_query"] == "retrieval"
    assert body["count"] == 1
    assert body["errors"] == ["Semantic Scholar: rate limited"]
    assert body["possible_duplicates"]


def test_search_item_fields_are_the_search_layers_registry(v1_client, fake_search):
    """The paper record's field set belongs to ``search/``, and is asserted here.

    Compared against the registry rather than a literal so that T3.1 adding an
    identifier fails this test instead of quietly making the frozen model wrong.
    """
    item = v1_client.post("/v1/search/papers", json={"query": "x"}).json()["items"][0]
    assert set(item) == set(PAPER_RECORD_CLEANED_KEYS)
    assert set(routes_v1.PaperRecord.model_fields) == set(PAPER_RECORD_CLEANED_KEYS)


def test_search_without_a_query_is_a_400(v1_client, fake_search):
    response = v1_client.post("/v1/search/papers", json={"query": "   "})
    assert response.status_code == 400
    assert response.json()["code"] == "missing_search_query"


def test_search_reports_that_it_did_not_rewrite(v1_client, fake_search):
    """No API key in the test environment, so the query is searched as written."""
    body = v1_client.post("/v1/search/papers", json={"query": "as-is"}).json()
    assert body["query"] == body["original_query"] == "as-is"
    assert body["reason"]


# --------------------------------------------------------------------------
# OpenAPI
# --------------------------------------------------------------------------


def test_every_operation_this_task_added_is_documented(v1_client):
    """Every operation has a summary and a described failure.

    The failures matter as much as the successes: a client built from this
    document has to know that a malformed id is a 400 and an absent document is a
    404, because those are two different bugs on its side. Scoped to the two
    routers this task added rather than to every ``/v1`` path, because the
    evidence routes belong to T1.5 and are not this contract to declare.
    """
    openapi = app.openapi()
    owned: list[tuple[str, str, str]] = []
    for module in (routes_v1, routes_jobs):
        for route in module.router.routes:
            if route.path.startswith("/v1"):
                owned.append((route.path, route.methods - {"HEAD"}, route.summary or ""))

    assert len(owned) == 16, owned
    for path, methods, summary in owned:
        assert summary, f"{path} has no summary"
        for method in methods:
            operation = openapi["paths"][path][method.lower()]
            assert operation.get("summary"), f"{method} {path} has no summary in the schema"

            described = set(operation.get("responses") or {})
            assert {"200", "201"} & described, f"{method} {path} documents no success"
            # 422 is FastAPI's own answer for a malformed body and is documented
            # by the framework on every operation that takes one.
            failures = described - {"200", "201", "202", "422"}
            assert failures, f"{method} {path} documents no failure"
            for status in failures:
                content = operation["responses"][status]["content"]["application/json"]
                assert content["schema"]["$ref"].endswith("/ErrorResponse"), (method, path, status)
                assert operation["responses"][status]["description"]


def test_an_uninstalled_cloud_parser_is_a_capability_answer(v1_client, document_id,
                                                           local_only_parsers, monkeypatch):
    """Distinct from the policy refusal, and reported as a conflict.

    Two different client problems that both look like "the cloud did not work":
    the deployment forbids it, and the deployment cannot do it. Conflating them
    sends people to change a consent switch that is already on.

    **The policy has to be granted for this to be the question it claims to
    ask.** The route resolves a cloud selection as *policy* first and as
    *availability* second (see the ordering note in
    :func:`routes_v1.resolve_reparse_parser`'s caller), because a deployment
    that forbids egress never registers the cloud adapter anyway — so under
    the default ``local_only`` this request is a 403 ``privacy_mode_local_only``
    and the capability check is never reached. That ordering is the product's
    deliberate answer to a different question, and
    ``test_reparse_refuses_a_cloud_parser_under_local_only`` is where it is
    pinned. Granting the policy here is what makes the adapter's absence the
    *only* remaining reason for the refusal, which is the distinction the test
    exists to draw; without it the test asserts nothing about capability at
    all, and it would keep passing even if the catalog lookup were deleted.
    """
    monkeypatch.setattr(config, "DOCUMENT_PRIVACY_MODE", "allow_cloud")
    monkeypatch.setattr(config, "DOCUMENT_CLOUD_CONSENT_GRANTED", True)

    response = v1_client.post(
        f"/v1/documents/{document_id}/reparse", json={"parser": "mineru_cloud"}
    )
    assert response.status_code == 409
    body = response.json()
    assert body["code"] == "parser_unavailable"
    assert "docling" in body["error"], (
        "the answer must name what this deployment *does* have, so the reader "
        "can tell a missing adapter from a mis-typed name"
    )


def test_the_openapi_document_resolves_every_reference(v1_client):
    """A dangling ``$ref`` renders as a broken page, and nothing warns."""
    openapi = app.openapi()
    schemas = openapi.get("components", {}).get("schemas", {})
    for name, schema in schemas.items():
        for reference in _references(schema):
            assert reference in schemas, f"{name} points at a missing {reference}"


def _references(node) -> list[str]:
    found: list[str] = []
    if isinstance(node, dict):
        if "$ref" in node and isinstance(node["$ref"], str):
            found.append(node["$ref"].rsplit("/", 1)[-1])
        for value in node.values():
            found.extend(_references(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_references(item))
    return found


def test_the_error_tables_are_internally_consistent(v1_client):
    """The tables are the contract, so the tables have to agree with themselves.

    Three properties a client relies on and none of which a per-code test can see:
    a code is a lowercase token a client can switch on; a code declared once is
    not declared twice with two different statuses (which would make the status a
    function of which table answered); and the documented code is the one the
    table row says, not a message that happens to mention it.
    """
    tables = {**routes_v1.DOCUMENT_ERRORS, **routes_jobs.JOB_ERRORS}

    seen: dict[str, tuple[int, str, str]] = {}
    for key, row in tables.items():
        status, message, code = row
        assert key == code, f"{key} is documented under a different code: {code}"
        assert 400 <= status < 600, (key, status)
        assert message.strip(), f"{key} has no message for the human reading it"
        assert code.replace("_", "").isalnum() and code.islower(), code
        assert code not in seen, f"{code} is declared twice"
        seen[code] = row

    # Every error body the surface builds carries exactly the three envelope
    # fields, because the envelope is declared once and every route uses it.
    envelope = set(routes_jobs.ErrorResponse.model_fields)
    assert envelope == {"error", "code", "timestamp"}
    response = v1_client.get("/v1/documents/doc_missing")
    assert set(response.json()) == envelope


# --------------------------------------------------------------------------
# The legacy surface still answers
# --------------------------------------------------------------------------


def test_legacy_routes_still_answer(v1_client, runtime_dirs):
    """Acceptance: 现有前端无破坏性切换.

    T2.4 adds a surface; it does not replace one. The browser calls ``/api/*``,
    so every one of these is a statement about a shipped client continuing to
    work — including the error codes, which is why the assertions are on ``code``
    rather than on prose.
    """
    health = v1_client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    assert v1_client.post("/api/analyze", data={}).json()["code"] == "missing_file"
    assert (
        v1_client.post(
            "/api/analyze", files={"file": ("x.exe", io.BytesIO(b"MZ"), "application/octet-stream")}
        ).json()["code"]
        == "unsupported_file_type"
    )
    assert v1_client.post("/api/ask", json={}).json()["code"] == "missing_question"
    assert v1_client.post("/api/search-papers", json={}).json()["code"] == "missing_search_query"
    assert v1_client.get("/api/export/nope?format=docx").json()["code"] == "session_not_found"


def test_legacy_upload_validation_is_the_v1_validation(v1_client):
    """One implementation behind two routes, and the codes still match.

    Asserted across the two routers rather than once, because the failure this
    guards against is a *divergence*: one router gaining a check the other does
    not have, which reads as "the v1 surface is stricter" until a client is
    switched.
    """
    from fastapi import UploadFile as StarletteUploadFile
    from starlette.datastructures import Headers

    def upload(name: str):
        return StarletteUploadFile(
            file=io.BytesIO(b"x"),
            filename=name,
            headers=Headers({"content-type": "text/plain"}),
        )

    for validate in (routes_documents._validate_upload, routes_v1.validate_upload):
        assert validate(upload("a.exe")) is not None
        assert validate(None) is not None
        assert validate(upload("a.txt")) is None

    response = v1_client.post(
        "/api/analyze", files={"file": ("../../escape.txt", io.BytesIO(b"x"), "text/plain")}
    )
    assert response.json()["code"] == "invalid_upload_metadata"


def test_the_legacy_surface_is_still_in_the_openapi_document(v1_client):
    """The contract document covers both halves and says which is frozen."""
    openapi = app.openapi()
    legacy = {path for path in openapi["paths"] if path.startswith("/api/")}
    assert {"/api/analyze", "/api/ask", "/api/recommend-papers"} <= legacy
    assert "冻结" in openapi["info"]["description"]
    assert "/v1" in openapi["info"]["description"]


def test_evidence_router_is_untouched_by_the_v1_surface(v1_client, parser, document_id):
    """The evidence route the document panel depends on still answers."""
    seed = v1_client.get(f"/v1/documents/{document_id}").json()
    assert seed["document_id"] == document_id
    response = v1_client.get(f"/v1/documents/{document_id}/evidence/ev_nope")
    assert response.status_code == 404
    assert response.json()["code"] in {"evidence_not_found", "document_not_found"}
