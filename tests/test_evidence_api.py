"""The Evidence API, asserted as the behaviour a citation click depends on.

T1.5's acceptance criterion is behavioural — "clicking a citation opens the
evidence; the locator is 100% resolvable or explicitly marked as having no
visual targeting capability" — so every assertion here goes through the real
router with a ``TestClient``. Nothing calls ``resolve_locator`` directly,
because the failure this API exists to prevent is a *wrong* answer reaching a
reader, and "the function returned a plausible object" is a much weaker
statement than "the bytes a browser receives say this, and not that".

The one deliberate exception is the app under test. ``app.py`` is not wired
for this router yet (that registration is a two-line change reported
separately), so these tests build a FastAPI app that includes
``routes_evidence.router`` exactly the way ``create_app`` will, and override
the store dependency so no test ever opens a real database. Everything below
the mount point — routing, the id guard, the lookup, the resolution, the JSON
— is the production code path.

The store dependency is overridden rather than monkeypatched so the seam being
exercised is the one FastAPI actually uses. A test that reached into the
module to swap ``_STORES`` would pass even if the route wired the dependency
up wrong; ``dependency_overrides`` fails loudly in that case instead.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from paperwhisperer.api import routes_evidence
from paperwhisperer.infra.stores import build_sqlite_bundle
from paperwhisperer.infra.stores.interfaces import DocumentRecord
from paperwhisperer.ingestion.schema import (
    LOCATOR_KINDS,
    BoundingBox,
    Chunk,
    Element,
    ParsedDocument,
    SourceLocator,
    Table,
)

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    """A personal-tier store on a throwaway SQLite file."""
    settings = SimpleNamespace(
        STORAGE_SQLITE_PATH=str(tmp_path / "evidence.sqlite3"),
        STORAGE_SQLITE_TIMEOUT_SECONDS=30,
        STORAGE_LEXICAL_FTS5="auto",
    )
    bundle = build_sqlite_bundle(settings)
    yield bundle
    bundle.close()


@pytest.fixture
def client(store):
    """A TestClient over an app that includes the evidence router.

    Shadows the suite-wide ``client`` fixture on purpose: that one is bound to
    the real application, which does not mount this router yet, and quietly
    inheriting it would mean these tests assert against the wrong app.
    """
    application = FastAPI()
    application.include_router(routes_evidence.router)
    application.dependency_overrides[routes_evidence.get_store_bundle] = lambda: store
    with TestClient(application) as test_client:
        yield test_client


# --------------------------------------------------------------------------
# Seed helpers
# --------------------------------------------------------------------------


def _table_element(document_id: str, *, page: int = 3, cell_range: str = "") -> Element:
    """A table element whose grid has a value the test can find by address."""
    table = Table(
        rows=[
            ["Model", "BLEU", "Latency"],
            ["Baseline", "27.1", "120ms"],
            ["Ours", "31.4", "95ms"],
        ],
        header_rows=1,
        caption="Main results",
    )
    return Element(
        element_id=f"{document_id}_table",
        role="table",
        table=table,
        locator=SourceLocator(kind="pdf_page", page=page, cell_range=cell_range or None),
    )


def _seed_document(
    store,
    document_id: str = "doc_1",
    *,
    index_version: str = "v2",
    table_page: int = 3,
) -> ParsedDocument:
    """Store a document with a prose element and a table element."""
    table_element = _table_element(document_id, page=table_page)
    prose_element = Element(
        element_id=f"{document_id}_prose",
        role="text",
        text="The dominant sequence models rely on recurrent or convolutional networks.",
        locator=SourceLocator(kind="pdf_page", page=2, bbox=BoundingBox(50.0, 600.0, 550.0, 640.0)),
    )
    document = ParsedDocument(
        document_id=document_id,
        source_filename="paper.pdf",
        media_type="application/pdf",
        page_count=8,
        elements=[prose_element, table_element],
    )
    store.metadata.upsert_document(
        DocumentRecord(
            document_id=document_id,
            document=document,
            source_filename="paper.pdf",
            media_type="application/pdf",
            page_count=8,
            index_version=index_version,
        )
    )
    return document


def _seed_chunk(
    store,
    document_id: str,
    chunk_id: str,
    locator: SourceLocator,
    *,
    text: str = "The dominant sequence models rely on recurrent or convolutional networks.",
    element_ids: tuple[str, ...] = (),
    index_version: str = "v2",
) -> Chunk:
    """Store one retrieval unit — the thing an evidence id addresses."""
    chunk = Chunk(
        chunk_id=chunk_id,
        text=text,
        element_ids=element_ids,
        locator=locator,
        index_version=index_version,
    )
    store.metadata.put_chunk(document_id, chunk)
    return chunk


def _retrieve_evidence(store, document_id: str, question: str, task: str = "qa"):
    """Run the real retriever and return the evidence it actually produced.

    The ids this returns are the ones a reader's citation chip carries: minted by
    :func:`paperwhisperer.rag.retrieve._evidence_id` over a real chunk and a real
    plan, never hand-assembled here. That distinction is the whole point — the
    original defect survived a fully green suite because every test addressed the
    ``chunk_*`` id form, so a test that invents its own ``ev_…`` string would
    prove only that the test agrees with itself.

    ``embeddings=None`` and ``reranker=None`` keep this to the lexical and fuzzy
    legs and keep a model out of it, so the ids are the same content-addressed
    ones a Q&A turn produces regardless of which legs happened to run.
    """
    from paperwhisperer.rag.retrieve import HybridRetriever, plan_for

    retriever = HybridRetriever(store, embeddings=None, reranker=None)
    result = retriever.retrieve(question, plan=plan_for(task), document_ids=[document_id])
    assert result.evidence, "the seeded document was not retrievable; the fixture is wrong"
    return result


def _retrieve_evidence_under_plan(store, document_id: str, question: str, plan):
    """Run the real retriever under a caller-constructed plan.

    The same drive as :func:`_retrieve_evidence`, but the plan is passed in
    rather than looked up by task name, because "is this plan in the registry"
    is the entire question the ``ev_`` tests below are about.
    """
    from paperwhisperer.rag.retrieve import HybridRetriever

    retriever = HybridRetriever(store, embeddings=None, reranker=None)
    result = retriever.retrieve(question, plan=plan, document_ids=[document_id])
    assert result.evidence, "the seeded document was not retrievable; the fixture is wrong"
    return result


def _seed_retrievable_document(store, document_id: str = "doc_1") -> list[Chunk]:
    """A document whose chunks are in both the metadata and lexical stores.

    Indexed lexically as well as in metadata because the retriever's lexical and
    fuzzy legs read the lexical store — without it a retrieval finds nothing and
    the evidence id under test would be one this fixture could not have produced.

    **Chunk ids are namespaced by document, and that is load-bearing rather than
    tidy.** ``put_chunk`` upserts on ``chunk_id`` alone (``ON CONFLICT(chunk_id)
    DO UPDATE SET document_id = excluded.document_id``), so two documents seeded
    with the same chunk ids do not coexist — the second silently *takes them
    over*, and the first is left with an empty chunk list. A cross-document test
    built on that fixture still passes, for the wrong reason: with no chunks of
    its own, the first document resolves nothing, so a foreign id not resolving
    proves only that an empty store is empty. Namespacing is what makes the
    candidate set non-trivial, and a non-trivial candidate set is the only thing
    that can demonstrate isolation.
    """
    _seed_document(store, document_id=document_id)
    passages = [
        ("chunk_seq", "The dominant sequence models rely on recurrent or convolutional networks."),
        ("chunk_attn", "The attention mechanism lets a model weigh tokens against each other."),
        ("chunk_speed", "Recurrent models are slow because they unroll the sequence step by step."),
    ]
    chunks = []
    for index, (suffix, text) in enumerate(passages, start=1):
        chunk_id = f"{document_id}_{suffix}"
        chunk = _seed_chunk(
            store,
            document_id,
            chunk_id,
            SourceLocator(kind="pdf_page", page=index),
            text=text,
        )
        store.lexical.index_chunk(chunk_id, text, document_id=document_id)
        chunks.append(chunk)
    return chunks


def _get(client, document_id: str, evidence_id: str):
    return client.get(f"/v1/documents/{document_id}/evidence/{evidence_id}")


# --------------------------------------------------------------------------
# Per-locator-kind resolution
# --------------------------------------------------------------------------


def test_pdf_page_with_absolute_bbox_emits_a_highlight_matching_the_stored_box(
    client, store
):
    """A resolvable box becomes a pdf.js fragment built from the stored numbers."""
    _seed_document(store)
    bbox = BoundingBox(x0=61.5, y0=142.25, x1=361.5, y1=190.75)
    _seed_chunk(store, "doc_1", "chunk_pdf", SourceLocator(kind="pdf_page", page=7, bbox=bbox))

    response = _get(client, "doc_1", "chunk_pdf")
    assert response.status_code == 200
    targeting = response.json()["visual_targeting"]

    assert targeting["available"] is True
    assert targeting["capability"] == "pdf_page_highlight"
    # The fragment is the stored box, not a rounded or re-derived one. A
    # viewer that draws this rectangle must draw the one the parser measured.
    assert targeting["highlight"] == bbox.to_query() == "xywh=61.5,142.25,300,48.5"
    # The page number is passed through verbatim — never shifted by one to
    # reconcile 0- and 1-based conventions, because "p. 7" highlighting p. 6
    # is the exact wrong-location failure this API is built to avoid.
    assert targeting["target"]["page"] == 7
    assert targeting["page_image"]["page"] == 7
    assert targeting["page_image"]["rendered"] is False


def test_pdf_page_with_a_relative_bbox_is_marked_and_emits_no_highlight(client, store):
    """A mirrored box is refused outright; the page is still honest."""
    _seed_document(store)
    # y0 < 0 is what BoundingBox itself uses to flag a mirrored box, so this
    # is the real shape a TOPLEFT-converting adapter produces.
    _seed_chunk(
        store,
        "doc_1",
        "chunk_rel",
        SourceLocator(
            kind="pdf_page",
            page=4,
            bbox=BoundingBox(x0=40.0, y0=-320.0, x1=300.0, y1=-280.0),
        ),
    )

    response = _get(client, "doc_1", "chunk_rel")
    assert response.status_code == 200
    targeting = response.json()["visual_targeting"]

    assert targeting["available"] is False
    assert targeting["reason_code"] == "relative_bbox"
    assert targeting["highlight"] is None
    # No fragment anywhere in the body — a client that scans the payload for
    # a box to draw must find nothing to draw.
    assert "xywh=" not in response.text
    # The page is a true fact and is still offered; only the highlight is
    # withheld.
    assert targeting["target"]["page"] == 4


def test_pdf_page_without_a_bbox_is_marked_missing(client, store):
    """A page with no box cannot pinpoint the passage, and says so."""
    _seed_document(store)
    _seed_chunk(store, "doc_1", "chunk_nobox", SourceLocator(kind="pdf_page", page=5))

    response = _get(client, "doc_1", "chunk_nobox")
    targeting = response.json()["visual_targeting"]

    assert targeting["available"] is False
    assert targeting["reason_code"] == "missing_bbox"
    assert targeting["highlight"] is None
    assert targeting["target"]["page"] == 5


def test_pdf_page_without_a_page_is_marked_missing_page(client, store):
    """A box with no page names a region of nothing, so nothing is returned."""
    _seed_document(store)
    _seed_chunk(
        store, "doc_1", "chunk_nopage", SourceLocator(kind="pdf_page", bbox=BoundingBox(1, 2, 3, 4))
    )

    response = _get(client, "doc_1", "chunk_nopage")
    targeting = response.json()["visual_targeting"]

    assert targeting["available"] is False
    assert targeting["reason_code"] == "missing_page"
    assert targeting["target"] is None
    assert targeting["page_image"] is None
    assert targeting["highlight"] is None


def test_slide_resolves_to_the_stored_slide(client, store):
    _seed_document(store)
    _seed_chunk(store, "doc_1", "chunk_slide", SourceLocator(kind="slide", slide=3))

    response = _get(client, "doc_1", "chunk_slide")
    targeting = response.json()["visual_targeting"]

    assert targeting["available"] is True
    assert targeting["capability"] == "slide_jump"
    assert targeting["target"]["slide"] == 3
    assert targeting["page_image"]["slide"] == 3


def test_sheet_range_resolves_with_its_cell_range(client, store):
    _seed_document(store)
    _seed_chunk(
        store,
        "doc_1",
        "chunk_sheet",
        SourceLocator(kind="sheet_range", sheet="Results", cell_range="B2:C3"),
    )

    response = _get(client, "doc_1", "chunk_sheet")
    targeting = response.json()["visual_targeting"]

    assert targeting["available"] is True
    assert targeting["capability"] == "sheet_cells"
    assert targeting["target"]["sheet"] == "Results"
    assert targeting["target"]["cell_range"] == "B2:C3"


def test_docx_element_resolves_to_paragraph_and_heading_path(client, store):
    _seed_document(store)
    _seed_chunk(
        store,
        "doc_1",
        "chunk_docx",
        SourceLocator(kind="docx_element", paragraph=5, heading_path=("Method", "Setup")),
    )

    response = _get(client, "doc_1", "chunk_docx")
    targeting = response.json()["visual_targeting"]

    assert targeting["available"] is True
    assert targeting["capability"] == "element_jump"
    assert targeting["target"]["paragraph"] == 5
    assert targeting["target"]["heading_path"] == ["Method", "Setup"]


def test_txt_offset_resolves_to_the_verbatim_offsets(client, store):
    _seed_document(store)
    _seed_chunk(
        store,
        "doc_1",
        "chunk_txt",
        SourceLocator(kind="txt_offset", offset_start=10, offset_end=40),
    )

    response = _get(client, "doc_1", "chunk_txt")
    targeting = response.json()["visual_targeting"]

    assert targeting["available"] is True
    assert targeting["capability"] == "offset_jump"
    # The offsets are echoed exactly; a well-meaning off-by-one would land a
    # reader one character from the sentence.
    assert targeting["target"]["offset_start"] == 10
    assert targeting["target"]["offset_end"] == 40


def test_unknown_locator_is_explicitly_marked_with_no_target(client, store):
    """The canonical "no visual targeting" answer, in full."""
    _seed_document(store)
    _seed_chunk(store, "doc_1", "chunk_unknown", SourceLocator(kind="unknown"))

    response = _get(client, "doc_1", "chunk_unknown")
    assert response.status_code == 200
    body = response.json()
    targeting = body["visual_targeting"]

    assert targeting["available"] is False
    assert targeting["capability"] is None
    assert targeting["reason_code"] == "unknown_locator"
    assert targeting["highlight"] is None
    assert targeting["target"] is None
    assert targeting["page_image"] is None
    # A Chinese sentence a user can read, not just a code.
    assert "没有可用的视觉定位能力" in targeting["reason"]
    # The passage itself is still there — a degraded locator degrades the
    # *position*, not the text.
    assert body["snippet"]["text"]


@pytest.mark.parametrize("kind", LOCATOR_KINDS)
def test_every_locator_kind_produces_a_definite_answer(client, store, kind):
    """Resolution is total: no kind falls through to an empty object.

    The property under test is that a well-formed locator of *any* schema
    kind comes back either available or with a reason code — never as a
    success response that simply omits the targeting block, which a client
    would read as "loaded, nothing to show".
    """
    document = _seed_document(store)
    _seed_chunk(store, "doc_1", "chunk_kind", SourceLocator(kind=kind, page=1, slide=1, sheet="S"))

    response = _get(client, "doc_1", "chunk_kind")
    assert response.status_code == 200
    targeting = response.json()["visual_targeting"]
    # A targeting block is present for every kind…
    assert "available" in targeting
    # …and it is either usable or explicitly explained.
    assert targeting["available"] is True or targeting["reason_code"]
    assert document.document_id == "doc_1"


# --------------------------------------------------------------------------
# Never the wrong location
# --------------------------------------------------------------------------


def test_highlight_comes_from_the_evidence_locator_not_a_sibling_element(client, store):
    """A box that belongs to another element must never be borrowed.

    The document's prose element carries a box at y=600; the chunk's own
    locator carries a different one. The rendered highlight must be the
    chunk's — the numbers are the whole identity of the rectangle, and
    swapping in a plausible neighbour's is the silent wrong-page bug.
    """
    _seed_document(store)
    own = BoundingBox(x0=100.0, y0=200.0, x1=400.0, y1=230.0)
    _seed_chunk(store, "doc_1", "chunk_own", SourceLocator(kind="pdf_page", page=2, bbox=own))

    targeting = _get(client, "doc_1", "chunk_own").json()["visual_targeting"]
    assert targeting["highlight"] == own.to_query()
    # The prose element's box is at y=600; it must not appear anywhere.
    assert "600" not in targeting["highlight"]


def test_a_box_is_not_borrowed_from_the_document_when_the_evidence_has_none(client, store):
    """No bbox on the locator means no highlight, even though elements have one.

    The elements in the document do carry resolvable boxes. Reaching for one
    would produce a highlight that looks perfect and belongs to a different
    element, so the API refuses instead and names the gap.
    """
    _seed_document(store)
    _seed_chunk(store, "doc_1", "chunk_bare", SourceLocator(kind="pdf_page", page=2))

    response = _get(client, "doc_1", "chunk_bare")
    targeting = response.json()["visual_targeting"]
    assert targeting["available"] is False
    assert targeting["reason_code"] == "missing_bbox"
    assert targeting["highlight"] is None


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------


def test_table_evidence_returns_addressable_html_with_the_cited_cell_identifiable(
    client, store
):
    """A table comes back as real HTML whose cells carry stable addresses."""
    _seed_document(store)
    _seed_chunk(
        store,
        "doc_1",
        "chunk_table",
        SourceLocator(
            kind="pdf_page",
            page=3,
            bbox=BoundingBox(10, 10, 500, 400),
            cell_range="R2C3",
        ),
        text="| Model | BLEU | Latency | ...",
        element_ids=("doc_1_table",),
    )

    response = _get(client, "doc_1", "chunk_table")
    assert response.status_code == 200
    table = response.json()["table"]

    assert table is not None
    # Real HTML, not a markdown blob in a <pre>.
    assert "<table>" in table["html"] and "<td" in table["html"] and "<th" in table["html"]
    # Every cell is addressable by the notation the citation verifier emits,
    # so a verifier's "R2C3" is a selector rather than a string.
    assert 'data-cell="R2C3"' in table["html"]
    assert 'data-cell="R1C1"' in table["html"]
    # The cited cell is both named and marked in the markup.
    assert table["cited_cell"] == "R2C3"
    assert 'data-cited="true"' in table["html"]
    # And it is the cell carrying the value, not a different one. Row 2 is
    # "Baseline / 27.1 / 120ms", so R2C3 is the latency column.
    cited_cell_html = table["html"].split('data-cell="R2C3"')[1]
    assert "120ms" in cited_cell_html.split("</td>")[0]
    # The table is the one the chunk was built from, not a lookup of the
    # nearest grid in the document.
    assert table["element_id"] == "doc_1_table"


def test_table_with_no_cited_cell_still_addresses_every_cell(client, store):
    """Without a named cell the grid is still fully addressable, just unmarked."""
    _seed_document(store)
    _seed_chunk(
        store,
        "doc_1",
        "chunk_table2",
        SourceLocator(kind="pdf_page", page=3),
        element_ids=("doc_1_table",),
    )

    table = _get(client, "doc_1", "chunk_table2").json()["table"]
    assert table["cited_cell"] == ""
    assert 'data-cell="R3C2"' in table["html"]
    assert "data-cited" not in table["html"]


# --------------------------------------------------------------------------
# Snippet consistency
# --------------------------------------------------------------------------


def test_snippet_reconciles_with_the_stored_chunk_text_by_offsets(client, store):
    """Slicing the stored chunk with the returned offsets reproduces the snippet.

    A quote is verified against a character range in the stored text, so the
    snippet must be a literal slice of that same text — no re-cleaning, no
    re-wrapping, no ellipsis that shifts a character.
    """
    text = "Line one stays put.  Line two\twith a tab and  two spaces."
    _seed_document(store)
    chunk = _seed_chunk(
        store, "doc_1", "chunk_snip", SourceLocator(kind="slide", slide=1), text=text
    )

    snippet = _get(client, "doc_1", "chunk_snip").json()["snippet"]
    assert snippet["source"] == "chunk_verbatim_text"
    assert snippet["source_id"] == chunk.chunk_id
    assert snippet["truncated"] is False
    # The contract: source[start:end] == snippet.
    assert chunk.verbatim_text[snippet["start"] : snippet["end"]] == snippet["text"]
    assert snippet["text"] == text


def test_snippet_uses_the_verbatim_body_when_a_context_prefix_is_present(client, store):
    """A chunk with a heading prefix still reports the body, and still reconciles.

    ``verbatim_text`` is the only text a direct quote is checked against, so a
    snippet that included the generated heading would show characters that
    the verifier never saw.
    """
    _seed_document(store)
    body = "The verbatim sentence the citation was checked against."
    chunk = Chunk(
        chunk_id="chunk_prefix",
        text="Method / Setup\n\n" + body,
        context_prefix="Method / Setup\n\n",
        locator=SourceLocator(kind="slide", slide=2),
        index_version="v2",
    )
    store.metadata.put_chunk("doc_1", chunk)

    snippet = _get(client, "doc_1", "chunk_prefix").json()["snippet"]
    assert snippet["text"] == body
    assert chunk.verbatim_text[snippet["start"] : snippet["end"]] == snippet["text"]


# --------------------------------------------------------------------------
# Version provenance
# --------------------------------------------------------------------------


def test_document_version_mismatch_is_surfaced_not_silently_served(client, store):
    """Evidence built under a different index version is labelled stale."""
    _seed_document(store, index_version="v2")
    _seed_chunk(
        store,
        "doc_1",
        "chunk_stale",
        SourceLocator(kind="pdf_page", page=1, bbox=BoundingBox(1, 1, 2, 2)),
        index_version="v1",
    )

    response = _get(client, "doc_1", "chunk_stale")
    assert response.status_code == 200
    body = response.json()
    version = body["document_version"]

    assert version["provenance"] == "mismatched"
    assert version["stale"] is True
    assert version["evidence_index_version"] == "v1"
    assert version["document_index_version"] == "v2"
    # The user is told, in a warning, rather than handed a stale box quietly.
    assert body["warnings"], "a version mismatch must produce a visible warning"
    assert "v1" in body["warnings"][0] and "v2" in body["warnings"][0]


def test_matching_versions_report_no_warning(client, store):
    _seed_document(store, index_version="v2")
    _seed_chunk(
        store, "doc_1", "chunk_ok", SourceLocator(kind="slide", slide=1), index_version="v2"
    )

    body = _get(client, "doc_1", "chunk_ok").json()
    assert body["document_version"]["provenance"] == "matched"
    assert body["document_version"]["stale"] is False
    assert body["warnings"] == []


def test_missing_index_version_is_reported_unverifiable_not_matched(client, store):
    """An empty version must not read as agreement just because "" == ""."""
    _seed_document(store, index_version="v2")
    _seed_chunk(
        store, "doc_1", "chunk_noiv", SourceLocator(kind="slide", slide=1), index_version=""
    )

    body = _get(client, "doc_1", "chunk_noiv").json()
    assert body["document_version"]["provenance"] == "unverifiable"
    assert body["warnings"]


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


def test_unknown_document_is_404(client, store):
    response = _get(client, "doc_missing", "chunk_pdf")
    assert response.status_code == 404
    assert response.json()["code"] == "document_not_found"


def test_unknown_evidence_id_is_404(client, store):
    _seed_document(store)
    response = _get(client, "doc_1", "chunk_never_stored")
    assert response.status_code == 404
    body = response.json()
    assert body["code"] == "evidence_not_found"
    assert "不存在" in body["error"]


def test_evidence_from_another_document_is_404_and_distinguishable(client, store):
    """Someone else's citation is refused, and the client can tell it apart."""
    _seed_document(store, document_id="doc_1")
    _seed_document(store, document_id="doc_2")
    _seed_chunk(store, "doc_2", "chunk_other", SourceLocator(kind="slide", slide=1))

    foreign = _get(client, "doc_1", "chunk_other")
    missing = _get(client, "doc_1", "not_a_real_chunk")

    # Both are 404, but they are not the same answer.
    assert foreign.status_code == missing.status_code == 404
    assert foreign.json()["code"] == "evidence_not_in_document"
    assert missing.json()["code"] == "evidence_not_found"
    assert foreign.json()["error"] != missing.json()["error"]
    # And the foreign chunk's text never leaks into doc_1's response.
    assert "slide 1" not in foreign.text


@pytest.mark.parametrize(
    "bad_evidence_id",
    ["bad!id", "bad id", "x" * 129],
)
def test_malformed_evidence_id_is_400(client, store, bad_evidence_id):
    _seed_document(store)
    response = client.get(f"/v1/documents/doc_1/evidence/{bad_evidence_id}")
    assert response.status_code == 400
    assert response.json()["code"] == "malformed_id"


def test_malformed_document_id_is_400(client, store):
    response = client.get("/v1/documents/bad!doc/evidence/chunk_x")
    assert response.status_code == 400
    assert response.json()["code"] == "malformed_id"


def test_a_well_formed_id_that_does_not_exist_is_404_not_400(client, store):
    """Validation and lookup are separate verdicts, and both are reachable."""
    _seed_document(store)
    response = _get(client, "doc_1", "chunk_totally_absent")
    assert response.status_code == 404
    assert response.json()["code"] == "evidence_not_found"


# --------------------------------------------------------------------------
# The two id namespaces (D1)
# --------------------------------------------------------------------------
#
# Everything above addresses a chunk id, because that is the only id form the
# API used to accept — and a suite that only ever uses the form a component
# happens to accept cannot see a mismatch between what a component emits and
# what another accepts. The retriever, the Q&A claims and the front end all emit
# `ev_…`; the tests below therefore mint their ids the way production does, by
# running the retriever, and then check both namespaces and the boundary between
# them.


def test_a_retriever_minted_evidence_id_resolves_through_the_api(client, store):
    """The regression the journey recorded: a real citation click 404s.

    Driven end to end from the retriever rather than from a hand-written id, so
    the assertion is about the id the product actually puts on the wire.
    """
    _seed_retrievable_document(store, "doc_1")
    evidence = _retrieve_evidence(store, "doc_1", "how do attention models weigh tokens").evidence[0]

    response = _get(client, "doc_1", evidence.evidence_id)

    assert response.status_code == 200, (
        f"a citation click carrying the retriever's own id {evidence.evidence_id} returned "
        f"HTTP {response.status_code} ({response.json().get('code')})"
    )
    payload = response.json()
    assert payload["evidence_id"] == evidence.evidence_id
    assert payload["chunk_id"] == evidence.chunk_id
    assert payload["document_id"] == "doc_1"
    assert payload["locator"], "the resolved evidence carries no locator"
    assert payload["snippet"]["text"], "the resolved evidence carries no passage"


def test_both_namespaces_resolve_to_the_same_chunk(client, store):
    """The chunk id and the evidence id name one passage, and the API says so.

    A citation chip carries the evidence id and the API's own callers use the
    chunk id; if they disagreed about which chunk either names, one of them
    would be a citation pointing somewhere other than where the other points.
    """
    _seed_retrievable_document(store, "doc_1")
    evidence = _retrieve_evidence(store, "doc_1", "how do attention models weigh tokens").evidence[0]

    by_evidence = _get(client, "doc_1", evidence.evidence_id)
    by_chunk = _get(client, "doc_1", evidence.chunk_id)

    assert by_evidence.status_code == by_chunk.status_code == 200
    assert by_evidence.json()["chunk_id"] == by_chunk.json()["chunk_id"]
    assert by_evidence.json()["locator"] == by_chunk.json()["locator"]
    assert by_evidence.json()["snippet"] == by_chunk.json()["snippet"]


def test_a_derived_plans_id_resolves_in_a_process_that_never_ran_the_retrieval(
    client, store, tmp_path
):
    """The real restart, not a simulated one.

    A second store bundle is opened on the same SQLite file and a second
    ``TestClient`` is built over it. The retriever that minted the id is gone
    along with every in-process registry it could have taught, so the only thing
    that can answer is what was written to the document. This is the test that
    fails the loudest if discovery ever moves back into memory.
    """
    from paperwhisperer.infra.stores import build_sqlite_bundle
    from paperwhisperer.rag.retrieve import plan_for

    _seed_retrievable_document(store, "doc_1")
    plan = plan_for("qa").with_overrides(final_k=3, candidate_depth=8)
    evidence = _retrieve_evidence_under_plan(
        store, "doc_1", "how do attention models weigh tokens", plan
    ).evidence[0]
    store.close()

    reopened = build_sqlite_bundle(
        SimpleNamespace(
            STORAGE_SQLITE_PATH=str(tmp_path / "evidence.sqlite3"),
            STORAGE_SQLITE_TIMEOUT_SECONDS=30,
            STORAGE_LEXICAL_FTS5="auto",
        )
    )
    application = FastAPI()
    application.include_router(routes_evidence.router)
    application.dependency_overrides[routes_evidence.get_store_bundle] = lambda: reopened
    try:
        with TestClient(application) as fresh_client:
            response = _get(fresh_client, "doc_1", evidence.evidence_id)
        assert response.status_code == 200, (
            f"a stored citation ({evidence.evidence_id}) did not resolve from a fresh "
            f"process: HTTP {response.status_code} ({response.json().get('code')})"
        )
        assert response.json()["chunk_id"] == evidence.chunk_id
    finally:
        reopened.close()


def test_the_recorded_signature_set_stays_bounded(client, store):
    """Discovery must not become an unbounded per-document log.

    A caller that derives a fresh plan per request — a different ``final_k``
    each time — would otherwise grow this list without limit, and the cost of
    resolving one citation would grow with it. The cap is what keeps that cost
    proportional to the chunk count, which is what the re-derivation always cost.
    """
    from paperwhisperer.rag import evidence as evidence_module
    from paperwhisperer.rag.retrieve import (
        PLAN_SIGNATURES_MAX,
        plan_for,
    )

    _seed_retrievable_document(store, "doc_1")
    base = plan_for("qa")
    for index in range(PLAN_SIGNATURES_MAX + 20):
        _retrieve_evidence_under_plan(
            store,
            "doc_1",
            "how do attention models weigh tokens",
            base.with_overrides(final_k=1 + index, candidate_depth=100 + index),
        )

    record = store.metadata.get_document("doc_1")
    recorded = evidence_module.document_plan_signatures(record)
    assert len(recorded) == PLAN_SIGNATURES_MAX, (
        f"expected the set to be capped at {PLAN_SIGNATURES_MAX}, got {len(recorded)}"
    )
    # And it is still a working candidate set, not a truncated husk.
    assert recorded == tuple(sorted(set(recorded)))


def test_an_evidence_id_from_another_document_is_404_and_leaks_nothing(client, store):
    """The cross-document refusal holds on the new namespace too.

    An evidence id is a *correct* id — it was minted by the retriever, for a
    real passage — so this is the case the chunk-id membership test could not
    reach: the caller has not guessed anything, they have simply named someone
    else's document. It still has to be a 404, and the other document's text
    still must not appear.
    """
    _seed_retrievable_document(store, "doc_1")
    _seed_retrievable_document(store, "doc_2")
    foreign = _retrieve_evidence(store, "doc_2", "how do attention models weigh tokens").evidence[0]

    response = _get(client, "doc_1", foreign.evidence_id)

    assert response.status_code == 404, (
        "an evidence id minted for another document resolved; the re-derivation "
        "escaped the document named in the URL"
    )
    assert response.json()["code"] in {"evidence_not_found", "evidence_not_in_document"}
    assert foreign.text[:40] not in response.text
    assert "attention" not in response.text


def test_an_unminted_but_well_formed_evidence_id_is_404(client, store):
    """Right shape, wrong content: an address nobody minted is not an address.

    The resolver matches by re-deriving, so an ``ev_…`` string that no retrieval
    produced has no candidate and is simply not found. It is deliberately *not* a
    400: it is shaped like an id, so it is a lookup that found nothing.
    """
    _seed_retrievable_document(store, "doc_1")
    response = _get(client, "doc_1", "ev_" + "0" * 16)
    assert response.status_code == 404
    assert response.json()["code"] == "evidence_not_found"


# --------------------------------------------------------------------------
# Non-registered plans (the population that still hides behind the Q&A tests)
# --------------------------------------------------------------------------
#
# Every `ev_…` test above retrieves under `plan_for("qa")` — a plan that is in
# the registry, so its signature is one the resolver can enumerate. That is the
# population which *works*, and it is precisely the population that cannot see
# the defect this section is about.
#
# The fact layer is the other producer: `make_evidence_provider` retrieves under
# `plan.with_overrides(final_k=..., candidate_depth=...)`, and `with_overrides`
# returns a plan whose signature is in no registry. Its evidence ids are the
# ones an analysis card would put on a citation chip, so "clicking an analysis
# card's citation" is the case under test here — driven the way the click would
# drive it, not simulated.


def test_a_non_registered_plans_evidence_id_resolves_through_the_api(client, store):
    """The day an analysis card's citation is wired up, this is the click.

    Not `plan_for("qa")` with a changed number attached — a *derived* plan, of
    the kind `paperwhisperer.analysis.paper_facts.make_evidence_provider` builds
    for every fact field. If resolution only knows the registered signatures,
    this id has no candidate and 404s, which is the D1 failure arriving again
    through the second door.
    """
    from paperwhisperer.rag.retrieve import PLANS, plan_for

    _seed_retrievable_document(store, "doc_1")
    base = plan_for("qa")
    plan = base.with_overrides(final_k=3, candidate_depth=8)
    assert plan.signature() not in {p.signature() for p in PLANS.values()}, (
        "this plan is registered, so the test is no longer exercising the "
        "population that fails"
    )

    evidence = _retrieve_evidence_under_plan(
        store, "doc_1", "how do attention models weigh tokens", plan
    ).evidence[0]

    response = _get(client, "doc_1", evidence.evidence_id)

    assert response.status_code == 200, (
        f"a citation minted under a derived plan ({evidence.evidence_id}) returned "
        f"HTTP {response.status_code} ({response.json().get('code')}); the resolver "
        "enumerates only registered plan signatures"
    )
    payload = response.json()
    assert payload["evidence_id"] == evidence.evidence_id
    assert payload["chunk_id"] == evidence.chunk_id
    assert payload["snippet"]["text"]


def test_a_non_registered_plans_id_resolves_without_the_registry(client, store):
    """Resolution must not depend on what happens to be in memory.

    This is the restart property. The ids are stored — `paper_facts` writes them
    into `PaperFacts.evidence_ids`, and a session carries them — so the id a
    reader clicks outlives the process that minted it. A resolver that only knew
    about signatures learned in-process would work for a citation minted this
    request-cycle and 404 for every stored one, which is D1 again with a
    narrower trigger.
    """
    from paperwhisperer.rag import evidence as evidence_module
    from paperwhisperer.rag.retrieve import plan_for

    _seed_retrievable_document(store, "doc_1")
    plan = plan_for("qa").with_overrides(final_k=3, candidate_depth=8)
    signature = plan.signature()
    assert signature not in evidence_module.known_plan_signatures(), (
        "the plan is registered; this test would prove nothing"
    )

    evidence = _retrieve_evidence_under_plan(
        store, "doc_1", "how do attention models weigh tokens", plan
    ).evidence[0]

    # Nothing has been added to the registry, and the signature is still not in
    # it after the retrieval: the answer cannot be coming from there.
    assert signature not in evidence_module.known_plan_signatures()

    # It is on the *stored record*, which is what survives a restart. Pinned
    # here so a change that made discovery in-process-only — working for a
    # citation minted this request-cycle and 404ing every stored one — fails
    # loudly rather than passing on the strength of the minting process being
    # the same one that resolves.
    stored = store.metadata.get_document("doc_1")
    assert signature in evidence_module.document_plan_signatures(stored)

    response = _get(client, "doc_1", evidence.evidence_id)
    assert response.status_code == 200, (
        f"a citation minted under a derived plan ({evidence.evidence_id}) returned "
        f"HTTP {response.status_code} ({response.json().get('code')})"
    )
    assert response.json()["chunk_id"] == evidence.chunk_id


def test_a_non_registered_plans_id_from_another_document_is_404_and_leaks_nothing(
    client, store
):
    """The cross-document guarantee must not weaken because resolution broadened.

    This is the security-relevant property, and the one that made the candidate
    set safe to widen: an id that is *correct* — minted by the real retriever, for
    a real passage, under a real derived plan — must still resolve to nothing when
    it names another paper, and the other paper's text must not appear in the
    refusal.
    """
    from paperwhisperer.rag import evidence as evidence_module
    from paperwhisperer.rag.retrieve import plan_for

    _seed_retrievable_document(store, "doc_1")
    _seed_retrievable_document(store, "doc_2")
    plan = plan_for("qa").with_overrides(final_k=3, candidate_depth=8)
    foreign = _retrieve_evidence_under_plan(
        store, "doc_2", "how do attention models weigh tokens", plan
    ).evidence[0]

    # doc_2's retrieval recorded the signature on doc_2, so the id is now
    # *fully re-derivable*: the signature is in hand and only the chunk is
    # missing. That is the strongest form of this test — it fails for the one
    # reason isolation is supposed to hold, and for no other.
    assert plan.signature() in evidence_module.document_plan_signatures(
        store.metadata.get_document("doc_2")
    )
    assert foreign.evidence_id == evidence_module.evidence_id_for(
        foreign.chunk_id, plan.signature()
    )
    # And it would genuinely resolve if handed the other document's chunks,
    # which is the proof that the 404 below is the isolation doing its job
    # rather than an id that simply cannot be re-derived.
    assert (
        evidence_module.find_chunk_for_evidence_id(
            foreign.evidence_id,
            store.metadata.list_chunks("doc_2"),
            plan_signatures=[plan.signature()],
        )
        is not None
    )

    response = _get(client, "doc_1", foreign.evidence_id)

    assert response.status_code == 404, (
        "a derived-plan evidence id minted for another document resolved; the "
        "re-derivation escaped the document named in the URL"
    )
    assert response.json()["code"] in {"evidence_not_found", "evidence_not_in_document"}
    assert foreign.text[:40] not in response.text
    assert "attention" not in response.text


def test_evidence_id_resolution_agrees_with_the_retrievers_own_minter():
    """Pin the mirror.

    :func:`paperwhisperer.rag.evidence.evidence_id_for` restates the retriever's
    content address so that the resolution module does not have to import the
    retrieval engine. A mirror that drifts would resolve nothing while every test
    that goes through the retriever still passes, so the two are pinned together
    here for every plan a deployment can actually retrieve under.
    """
    from paperwhisperer.rag import evidence as evidence_module
    from paperwhisperer.rag.retrieve import PLANS, _evidence_id

    assert PLANS, "no retrieval plans registered; this test would prove nothing"
    for task, plan in PLANS.items():
        for chunk_id in ("chunk_a", "chunk_b"):
            assert evidence_module.evidence_id_for(chunk_id, plan.signature()) == _evidence_id(
                chunk_id, plan.signature()
            ), f"the resolver's mirror drifted from the retriever for plan {task!r}"
    assert set(evidence_module.known_plan_signatures()) == {
        plan.signature() for plan in PLANS.values()
    }


# --------------------------------------------------------------------------
# The documented degraded shape
# --------------------------------------------------------------------------


def test_no_visual_targeting_response_has_a_shape_a_client_cannot_misread(client, store):
    """Pin the degraded contract: available False, capability None, no highlight.

    This is the "or explicitly marked" branch of the release gate. The point
    of the assertion is that ``available`` is *literally* ``False`` — not
    absent, not ``None`` — so a client that reads truthiness cannot mistake
    "cannot locate this" for "loaded, nothing to show".
    """
    _seed_document(store)
    _seed_chunk(store, "doc_1", "chunk_degraded", SourceLocator(kind="unknown"))

    targeting = _get(client, "doc_1", "chunk_degraded").json()["visual_targeting"]

    assert targeting["available"] is False
    assert targeting["capability"] is None
    assert targeting["highlight"] is None
    assert targeting["reason_code"] == "unknown_locator"
    assert targeting["reason"]  # non-empty, user-readable Chinese
    # The keys are all present, so a client reading the block by key does not
    # have to guess whether a missing key means "no box" or "old build".
    assert set(targeting) == {
        "available",
        "capability",
        "highlight",
        "reason_code",
        "reason",
        "target",
        "page_image",
    }
