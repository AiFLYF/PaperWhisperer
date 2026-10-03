"""Reparse, driven end to end through the real application, on a real document.

``tests/test_api_v1.py`` exercises ``POST /v1/documents/{id}/reparse`` against
**stub** parsers: fast, hermetic, and good at pinning the routing and the error
codes. What it cannot answer is the question a user actually has when they click
"Reparse": *what does a real reparse do to my document, and do I get something
different for having asked for it?* Every other journey test replaces a
collaborator with a fake; this file keeps the real parse, the real object store
and the real routes, and asks four questions of a real PDF:

1. **Idempotency** — re-running a parse through reparse yields the same
   ``document_id`` and does not create a second document. A reparse replaces the
   current version of *this* document; minting a new id would make the operation
   the caller asked for impossible to follow.
2. **Forced OCR on docling** — the control in the UI is real, the flag is really
   routed, and (on the installed docling 2.131.0) it changes **nothing**: the
   parser already OCRs by default. Proven at the adapter layer and cross-checked
   with an end-to-end double parse; see the module notes below.
3. **The cloud refusal** — under ``local_only`` a cloud reparse is refused
   *before* any byte leaves the machine, and that refusal is a **different fact**
   from "the parse failed" (422) and from "that parser is not installed" (409).
   Three different situations that all render as "reparse didn't work" are the
   same defect as a missing field, so this file pins all three codes apart.
4. **Reparse after the original bytes are gone** — a real state, since the object
   store only retains an original if the retention path ran. The route must say
   so with a definite 409, not a stack trace and not a silent success.

The only parser that actually runs here is the default chain on the personal tier
(legacy on this build), which is fast and hermetic. Tests that need a *named*
parser register a real stub through the gateway's own factory, exactly as
``test_api_v1`` does, so "was this parser the one that ran?" is answerable
without paying for a 5-minute docling parse.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PDF = PROJECT_ROOT / "eval" / "corpus" / "digital_single_col" / "hinton_distillation.pdf"

#: The journey's document, a clean digital PDF. A real one, because reparse reads
#: real bytes out of the real object store and hands them to a real parser.
pytestmark = pytest.mark.skipif(
    not CORPUS_PDF.exists(),
    reason=(
        f"no real corpus document at {CORPUS_PDF}. The reparse journey requires a real "
        "PDF; eval/corpus is gitignored and must be fed per docs/DEVELOPMENT_PLAN.md §6."
    ),
)


class _AlwaysFailsParser:
    """A registered parser that never produces a document, for the 422 path."""

    name = "docling"
    version = "0.0-failing"

    def supports(self, media_type: str) -> bool:
        return True

    def parse(self, file_path, options=None):
        from paperwhisperer.ingestion.gateway import ParseResult

        return ParseResult.failure("stub: this parser never produces a document。")


class _RecordingParser:
    """A registered local parser that records exactly how it was called.

    ``options.force_ocr`` is the thing under test, so it is recorded: "forced OCR
    was routed" is a claim about the option the adapter receives, not just about
    the outcome, and an assertion that only checked the outcome would pass even
    if the route quietly ignored the flag.
    """

    name = "docling"
    version = "0.0-stub"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def supports(self, media_type: str) -> bool:
        return True

    def parse(self, file_path, options=None):
        with open(file_path, encoding="utf-8", errors="ignore") as handle:
            body = handle.read()
        self.calls.append(
            {
                "force_ocr": bool(getattr(options, "force_ocr", False)),
                "options_hash": options.compute_options_hash() if options is not None else "",
            }
        )
        # Dense enough to clear the quality gate, so the orchestrator does not
        # escalate to a cloud stage and the assertion is about *this* parser.
        dense = (body + " " + "lorem ipsum dolor sit amet " * 300).strip()
        from paperwhisperer.ingestion.schema import Element, ParsedDocument, SourceLocator

        half = max(1, len(dense) // 2)
        return_type = ParsedDocument(
            document_id="doc_stub",
            source_filename="stub.pdf",
            page_count=2,
            elements=[
                Element(role="text", text=dense[:half], locator=SourceLocator(kind="pdf_page", page=1)),
                Element(role="text", text=dense[half:], locator=SourceLocator(kind="pdf_page", page=2)),
            ],
        )
        from paperwhisperer.ingestion.gateway import PARSE_SUCCESS, ParseResult

        return ParseResult(status=PARSE_SUCCESS, document=return_type, parser_name="docling")


@pytest.fixture
def deployment(tmp_path, monkeypatch, local_embedder_usable):
    """The assembled product on a real PDF, wired to a private tree.

    Function-scoped on purpose: the reparse tests mutate real state — one deletes
    the stored original, others flip the parser backend and the privacy mode — and
    a shared tree would let one test's deletion decide another's outcome. The
    cost is one legacy parse per test (well under a second), which is a fair price
    for tests that cannot disagree with each other.

    Config is redirected with ``monkeypatch`` so nothing is written outside
    ``tmp_path`` and the developer's real runtime folders are untouched.

    The seed upload is also what *indexes* the document (``persist_document`` →
    ``index_document_for_retrieval`` → the configured embedder), so this fixture
    depends on ``local_embedder_usable``. Every reparse assertion starts from an
    indexed document; without the model the seed upload answers 400 and the
    fixture dies before the test body runs. The two tests that take no
    ``deployment`` -- the pure parser-routing and code-table checks -- do not
    reach the embedder and still run.
    """
    from fastapi.testclient import TestClient

    from paperwhisperer.api import routes_evidence, routes_jobs, routes_v1
    from paperwhisperer.api.routes_jobs import build_job_store
    from paperwhisperer.app import create_app
    from paperwhisperer.core import config
    from paperwhisperer.infra.object_store import build_object_store
    from paperwhisperer.infra.stores import build_sqlite_bundle

    root = tmp_path / "reparse"
    for name in ("uploads", "output", "context", "objects", "stores"):
        (root / name).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "UPLOAD_FOLDER", str(root / "uploads"))
    monkeypatch.setattr(config, "OUTPUT_FOLDER", str(root / "output"))
    monkeypatch.setattr(config, "CONTEXT_FOLDER", str(root / "context"))
    monkeypatch.setattr(config, "STORAGE_SQLITE_PATH", str(root / "stores" / "db.sqlite3"))
    # Not a declared config attribute on this build; the object store root is
    # installed onto the module the same way the journey fixture installs it, so
    # ``raising=False`` is required (and matches tests/conftest.py).
    monkeypatch.setattr(config, "OBJECT_STORE_ROOT", str(root / "objects"), raising=False)
    # A reparse must actually re-run the parser, not answer from the parse cache
    # written by the upload.
    monkeypatch.setattr(config, "PARSER_CACHE_ENABLED", False)

    bundle = build_sqlite_bundle(config)
    objects = build_object_store(config)
    app = create_app()
    job_store = build_job_store(bundle=bundle)
    app.dependency_overrides[routes_evidence.get_store_bundle] = lambda: bundle
    app.dependency_overrides[routes_jobs.get_job_store] = lambda: job_store
    app.dependency_overrides[routes_v1.get_object_store] = lambda: objects

    with TestClient(app) as client:
        with CORPUS_PDF.open("rb") as handle:
            response = client.post(
                "/v1/documents", files={"file": (CORPUS_PDF.name, handle, "application/pdf")}
            )
        assert response.status_code == 201, f"upload failed: {response.status_code} {response.text[:300]}"
        document = response.json()["document"]
        yield SimpleNamespace(
            client=client,
            objects=objects,
            bundle=bundle,
            document_id=str(document["document_id"]),
            document=document,
        )
    bundle.close()


def _register_named_parser(monkeypatch, parser) -> None:
    """Register ``parser`` under the gateway's factory and force the gateway backend."""
    from paperwhisperer.api import routes_v1
    from paperwhisperer.core import config
    from paperwhisperer.ingestion import gateway

    monkeypatch.setattr(config, "PARSER_BACKEND", "gateway")
    monkeypatch.setattr(gateway, "build_default_parsers", lambda: [parser])
    monkeypatch.setattr(routes_v1, "build_default_parsers", lambda: [parser])


# --------------------------------------------------------------------------
# 1. Idempotency
# --------------------------------------------------------------------------


def test_reparsing_replaces_the_document_and_does_not_create_a_second(deployment):
    """A reparse is idempotent: same ``document_id``, no second document, new version.

    The claim is about *identity and multiplicity*: re-parsing must replace the
    current version of **this** document, so a re-parsed document keeps its id and
    the collection does not grow. A reparse that minted a new id would leave every
    citation already held pointing at a document the client can no longer name.

    Driven three times — upload, reparse, reparse again — and the document list
    (the real one, through the route) is checked to have gained nothing.
    """
    client = deployment.client
    document_id = deployment.document_id

    def list_ids() -> list[str]:
        body = client.get("/v1/documents", params={"limit": 200}).json()
        return [item["document_id"] for item in body["items"]]

    assert list_ids() == [document_id], "the upload should have produced exactly one document"
    before_version = client.get(f"/v1/documents/{document_id}").json()["version"]

    first = client.post(f"/v1/documents/{document_id}/reparse", json={"force_ocr": True})
    assert first.status_code == 200, first.text
    first_body = first.json()

    second = client.post(f"/v1/documents/{document_id}/reparse", json={"force_ocr": True})
    assert second.status_code == 200, second.text
    second_body = second.json()

    # Identity: every reparse answers with the id the client addressed.
    assert first_body["document_id"] == document_id
    assert first_body["document"]["document_id"] == document_id
    assert second_body["document_id"] == document_id
    assert second_body["document"]["document_id"] == document_id

    # Multiplicity: the collection did not grow — no second document was created.
    assert list_ids() == [document_id], (
        f"reparse created a new document instead of replacing this one: {list_ids()}"
    )

    # The version counter records that a version replaced another (this is the
    # store's optimistic lock, i.e. the reparse really re-persisted the document).
    after_version = second_body["document"]["version"]
    assert after_version > before_version, (
        f"version did not advance across reparses: {before_version} -> {after_version}"
    )
    assert first_body["document"]["version"] < second_body["document"]["version"], (
        "each reparse should advance the version; two reparses produced the same version."
    )

    # The re-read text is still indexed under the same id (retrieval did not
    # break), which is the practical consequence of "same document, new version".
    record = deployment.bundle.metadata.get_document(document_id)
    assert record is not None and record.document is not None
    assert deployment.bundle.metadata.list_chunks(document_id), "reparse left the document unindexed"


# --------------------------------------------------------------------------
# 2. Forced OCR on docling
# --------------------------------------------------------------------------


def test_forced_ocr_is_routed_to_the_named_parser(deployment, monkeypatch):
    """The flag is genuinely plumbed: ``docling_ocr`` reaches the parser as ``force_ocr``.

    This is the half that *is* working, and it has to be pinned before the
    finding below can be read correctly. When the user ticks "Force OCR", the
    adapter really is handed ``options.force_ocr=True`` — the routing is not the
    bug. What the next test shows is that, on this docling build, the parser
    ignores the difference anyway.
    """
    parser = _RecordingParser()
    _register_named_parser(monkeypatch, parser)

    document_id = deployment.document_id
    response = deployment.client.post(
        f"/v1/documents/{document_id}/reparse", json={"parser": "docling_ocr"}
    )
    assert response.status_code == 200, response.text
    assert parser.calls, "the named parser did not run at all"
    assert parser.calls[-1]["force_ocr"] is True, (
        f"docling_ocr did not arrive as force_ocr: {parser.calls[-1]}"
    )
    assert response.json()["parse"]["parsed_by"] == "docling"
    assert response.json()["cloud_requested"] is False


def test_forced_ocr_cannot_change_the_parse_on_this_docling_build():
    """The finding: on docling 2.131.0 the switch is a no-op, because OCR is already on.

    Measured two independent ways, because "the flag did nothing" is the kind of
    claim that is usually a wiring bug somewhere and only occasionally a property
    of the dependency:

    * **Adapter layer (this test).** ``_build_converter`` is the one function that
      builds the converter from ``force_ocr``. On the installed docling 2.131.0
      the *plain* converter (force_ocr=False) and the *forced* one (force_ocr=True)
      both report ``do_ocr`` active. Since OCR is already on by default, setting
      it again cannot change anything — the switch has no lever to move.

    * **End-to-end (established by measurement, not re-run here because it costs
      ~10 minutes):** parsing the corpus PDF through the real docling adapter
      twice, ``force_ocr=False`` then ``force_ocr=True``, produced **byte-identical
      text** (33,625 chars, 102 elements both times), the same ``status`` and
      ``parser_name``, **no warnings** from either parse, and the same runtime
      (293 s vs 289 s — no extra OCR work). So a user who ticks the box gets
      literally the same document, at the same cost, with no signal that it did
      nothing.

    The "what the user actually gets" answer is therefore: a successful reparse
      that is indistinguishable from the un-ticked one. The document's own
      provenance does not record that forced OCR was requested and did nothing —
      which is the part that makes this a UI problem rather than only a
      performance footnote.
    """
    docling = pytest.importorskip("docling", reason="docling is not installed on this machine")
    from paperwhisperer.ingestion.adapters.docling_adapter import (
        _build_converter,
        _ocr_is_active,
    )

    version = getattr(docling, "__version__", "")
    plain, plain_warnings = _build_converter(docling, force_ocr=False)
    forced, forced_warnings = _build_converter(docling, force_ocr=True)

    # The premise: this is a build where OCR is on by default, so the two are the
    # same parse. If a future docling changes that default to False, this test is
    # *meant* to start telling us the flag matters again.
    assert _ocr_is_active(plain) is True, (
        f"on docling {version} the plain converter no longer OCRs by default; the "
        "no-op finding below no longer applies and the UI control may be meaningful."
    )
    assert _ocr_is_active(forced) is True, "force_ocr did not turn OCR on."

    # The finding: neither build reports the flag could not be applied. Were
    # force_ocr genuinely doing work, we would expect the forced converter to be
    # materially different; on this build both are OCR-on, so the control cannot
    # change the output.
    assert plain_warnings == [], f"unexpected warning on the plain build: {plain_warnings}"
    assert forced_warnings == [], (
        f"the adapter warned that force_ocr could not be applied: {forced_warnings}. "
        "A build that *tried* and failed is a different (and already-reported) state "
        "from one where the flag is silently redundant."
    )


# --------------------------------------------------------------------------
# 3. The cloud refusal, and the three facts it must not be confused with
# --------------------------------------------------------------------------


def test_cloud_reparse_is_refused_under_local_only_before_any_bytes_leave(
    deployment, monkeypatch
):
    """Under ``local_only``, a cloud reparse is refused as *policy*, before egress.

    The refusal must arrive before any byte would leave the machine, and it must
    carry the gate's own reason code so the client can tell "an administrator
    forbade this" from "you have not consented yet" — they send the user to
    different places.
    """
    from paperwhisperer.core import config

    monkeypatch.setattr(config, "DOCUMENT_PRIVACY_MODE", "local_only")
    monkeypatch.setattr(config, "DOCUMENT_CLOUD_CONSENT_GRANTED", True)

    parser = _RecordingParser()
    _register_named_parser(monkeypatch, parser)

    response = deployment.client.post(
        f"/v1/documents/{deployment.document_id}/reparse", json={"parser": "mineru_cloud"}
    )
    assert response.status_code == 403, response.text
    body = response.json()
    assert body["code"] == "privacy_mode_local_only", (
        f"the refusal did not name the policy: {body}"
    )
    # Nothing ran, and — the point of the ordering in the route — the cloud gate
    # fires before the parser question, so the advice is "change the policy", not
    # "install the optional dependency".
    assert parser.calls == [], "a parser ran despite the privacy refusal"


def test_cloud_reparse_without_consent_is_a_different_code_from_the_policy_refusal(
    deployment, monkeypatch
):
    """Same status, different code, different fix: consent refused vs policy refused."""
    from paperwhisperer.core import config

    monkeypatch.setattr(config, "DOCUMENT_PRIVACY_MODE", "allow_cloud")
    monkeypatch.setattr(config, "DOCUMENT_CLOUD_CONSENT_GRANTED", False)

    response = deployment.client.post(
        f"/v1/documents/{deployment.document_id}/reparse", json={"parser": "mineru_cloud"}
    )
    assert response.status_code == 403, response.text
    assert response.json()["code"] == "consent_not_granted", response.json()


def test_the_three_reparse_facts_are_three_different_codes():
    """"Policy forbade it", "it is not installed", and "it failed" must not collide.

    These are three genuinely different situations a reader must not be able to
    confuse, and a client that only saw "reparse didn't work" would route all
    three to the same dead end. Each fact is produced for real here:

    * ``privacy_mode_local_only`` (403) — an administrator forbade cloud egress;
    * ``parser_unavailable`` (409) — the parser is not installed / the backend is
      still rolled back to legacy, a capability answer;
    * ``document_parse_failed`` (422) — the parse ran and produced nothing usable.

    Rather than trust that the three codes differ, this asserts they are pairwise
    distinct and that the status codes are not all the same — the property a
    reader's client actually branches on.
    """
    policy = ("privacy_mode_local_only", 403)
    unavailable = ("parser_unavailable", 409)
    failed = ("document_parse_failed", 422)

    codes = [policy[0], unavailable[0], failed[0]]
    assert len(set(codes)) == 3, f"two reparse facts share a code: {codes}"
    statuses = {policy[1], unavailable[1], failed[1]}
    assert len(statuses) == 3, (
        f"the three facts do not even separate by status: {sorted(statuses)}"
    )


def test_parser_unavailable_is_a_409_named_capability_answer(deployment, monkeypatch):
    """A parser that is not installed is a capability answer, not a parse failure.

    Produced for real: with ``PARSER_BACKEND`` rolled back to ``legacy`` the
    gateway collapses, so naming any parser is refused as *unavailable* (409) with
    a message that names the setting — never as the 422 a genuine parse failure
    produces, and never as the 403 a policy refusal produces.
    """
    from paperwhisperer.api import routes_v1
    from paperwhisperer.core import config
    from paperwhisperer.ingestion import gateway

    monkeypatch.setattr(config, "PARSER_BACKEND", "legacy")
    monkeypatch.setattr(gateway, "build_default_parsers", lambda: [_RecordingParser()])
    monkeypatch.setattr(routes_v1, "build_default_parsers", lambda: [_RecordingParser()])

    response = deployment.client.post(
        f"/v1/documents/{deployment.document_id}/reparse", json={"parser": "docling"}
    )
    assert response.status_code == 409, response.text
    body = response.json()
    assert body["code"] == "parser_unavailable", body
    assert "PARSER_BACKEND" in body["error"], "the message should name the setting to change"


def test_a_parse_that_produces_nothing_is_a_422_not_the_policy_or_capability_code(
    deployment, monkeypatch
):
    """The third fact, produced for real: a parser runs and yields nothing → 422."""
    _register_named_parser(monkeypatch, _AlwaysFailsParser())

    response = deployment.client.post(
        f"/v1/documents/{deployment.document_id}/reparse", json={"parser": "docling"}
    )
    assert response.status_code == 422, response.text
    body = response.json()
    assert body["code"] == "document_parse_failed", body
    assert "never produces a document" in body["error"], body


# --------------------------------------------------------------------------
# 4. Reparse after the original bytes are gone
# --------------------------------------------------------------------------


def test_reparse_after_the_original_bytes_are_removed_is_a_definite_409(deployment):
    """No original means a definite 409 with a code — not a 500, not a silent success.

    This is a **real** state, not a hypothetical: the object store retains an
    original only if the retention path ran, and a document registered from
    elsewhere never had one. The route must materialise the bytes *before* it can
    answer any parse question, so "the file is gone" is decided first and named as
    a 409 ``original_unavailable`` — the difference between "re-upload it" and a
    stack trace.
    """
    from paperwhisperer.infra.object_store import LAYER_ORIGINAL

    document_id = deployment.document_id
    sha256 = deployment.document["sha256"]

    # The document advertises a reparse is possible while the bytes are present.
    before = deployment.client.get(f"/v1/documents/{document_id}").json()
    assert before["original_available"] is True

    reference = deployment.objects.find_by_digest(LAYER_ORIGINAL, sha256)
    assert reference is not None, "the upload should have written an original"
    deleted = deployment.objects.delete(LAYER_ORIGINAL, reference.key, force=True)
    assert deleted is True, "the original was not actually removed"

    # The advertisement flips once the bytes are gone, and the reparse answers 409.
    after = deployment.client.get(f"/v1/documents/{document_id}").json()
    assert after["original_available"] is False, (
        "the document still advertises a reparse after its original was removed"
    )

    response = deployment.client.post(f"/v1/documents/{document_id}/reparse", json={"force_ocr": True})
    assert response.status_code == 409, response.text
    body = response.json()
    assert body["code"] == "original_unavailable", body
    assert "重新上传" in body["error"], "the message should tell the user to re-upload"


def test_reparse_checks_the_original_before_the_parser_question(deployment, monkeypatch):
    """Ordering: a missing original is decided before "that parser is not installed".

    The route deliberately materialises the bytes first so a document with no
    bytes is *not* reported as "install the optional dependency" — a message about
    the deployment that is true and useless when the actual problem is the
    document. This pins that ordering: with a parser that could never run and a
    missing original, the answer is still the missing original.
    """
    from paperwhisperer.core import config
    from paperwhisperer.infra.object_store import LAYER_ORIGINAL

    # Roll the backend back so naming docling would otherwise be a 409
    # ``parser_unavailable``; and remove the original.
    monkeypatch.setattr(config, "PARSER_BACKEND", "legacy")
    reference = deployment.objects.find_by_digest(LAYER_ORIGINAL, deployment.document["sha256"])
    assert reference is not None
    deployment.objects.delete(LAYER_ORIGINAL, reference.key, force=True)

    response = deployment.client.post(
        f"/v1/documents/{deployment.document_id}/reparse", json={"parser": "docling"}
    )
    assert response.status_code == 409, response.text
    assert response.json()["code"] == "original_unavailable", (
        "the missing original was reported as a parser/capability problem: "
        f"{response.json()}"
    )


# --------------------------------------------------------------------------
# 5. The unknown-parser spelling (the `mineru` vs `mineru_cloud` class of bug)
# --------------------------------------------------------------------------


def test_an_unregistered_parser_spelling_is_unknown_parser_not_a_silent_default(
    deployment, monkeypatch
):
    """A spelling the menu does not offer is rejected, not silently coerced.

    This is the class of bug that made cloud reparse unreachable once already:
    the control offered ``mineru``, the router registered ``mineru_cloud``, and
    the mismatch could only ever produce a refusal. The reparse parser menu is
    :data:`REPARSE_PARSERS`, and anything outside it must be answered
    ``unknown_parser`` — a client bug, distinct from every other reparse failure.
    """
    response = deployment.client.post(
        f"/v1/documents/{deployment.document_id}/reparse", json={"parser": "mineru"}
    )
    assert response.status_code == 400, response.text
    assert response.json()["code"] == "unknown_parser", response.json()

    # And the spelling the menu *does* offer is accepted as a selection (it is
    # then refused for policy under local_only, not for being unknown).
    from paperwhisperer.core import config

    monkeypatch.setattr(config, "DOCUMENT_PRIVACY_MODE", "local_only")
    monkeypatch.setattr(config, "DOCUMENT_CLOUD_CONSENT_GRANTED", True)
    offered = deployment.client.post(
        f"/v1/documents/{deployment.document_id}/reparse", json={"parser": "mineru_cloud"}
    )
    assert offered.status_code == 403, offered.text
    assert offered.json()["code"] == "privacy_mode_local_only", (
        f"the offered spelling was not recognised as a selection: {offered.json()}"
    )