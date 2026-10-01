"""What every tier's metadata store must do, asserted once.

The contract lives in a plain class with no backend in its name and no
``if`` on the backend. ``TestMetadataSqlite`` and ``TestMetadataPostgres``
below are empty subclasses that differ by one line, and that is the whole
point: the two tiers cannot drift, because there is only one copy of the
assertions to drift from. A tier-specific behaviour gets a named test
function in that tier's own file, where nobody mistakes it for the contract.

The failure paths are here on purpose. A store that is only ever exercised
by writing a document and reading it back is a store whose delete, update
and concurrency behaviour is untested exactly where it breaks.
"""

from __future__ import annotations

import json
import threading

import pytest

from paperwhisperer.ingestion.schema import Chunk, ParsedDocument

from .test_stores_tiers import (
    TIER_HOSTED,
    TIER_PERSONAL,
    canonical_json,
    open_harness,
    requires_postgres,
    sample_chunk,
    sample_document,
    sample_facts,
    sample_record,
)


@pytest.fixture
def harness(request, tmp_path):
    """The store bundle for whichever tier the test class names.

    The class attribute is the whole mechanism: ``tier = "sqlite"`` on one
    subclass and ``tier = "hosted"`` on the other, so the two differ by one
    line and cannot drift in their assertions.
    """
    yield from open_harness(request.node.cls.tier, tmp_path)


class StoreContract:
    """Behaviour that must be identical on every tier.

    The name is the mechanism that keeps this from being collected a third
    time. pytest's default ``python_classes`` is ``Test*``, so a contract
    base called ``StoreContract`` is never run on its own — which matters,
    because a base with no tier attribute would silently default to the
    personal one and produce a copy of the suite that only ever checks one
    tier. Only the two named subclasses below run it.
    """

    # -- documents ---------------------------------------------------------

    def test_a_document_survives_the_round_trip_unchanged(self, harness):
        """A locator written today must still resolve when read next month.

        Compared two ways on purpose. Deep equality catches a dropped field;
        the serialized comparison catches one that is present but ordered or
        formatted differently. The schema module's own claim is that
        ``to_dict`` / ``from_dict`` are exact inverses, and a store that
        round-trips through a text column is the first place that claim is
        actually at risk.
        """
        # One document object, used for both sides. Calling the factory twice
        # would give the two copies different generated element ids, and the
        # comparison would fail for a reason that has nothing to do with the
        # store.
        original = sample_document("doc-round-trip")
        record = sample_record("doc-round-trip")
        record.document = original
        stored = harness.metadata.upsert_document(record)
        assert stored.document_id == "doc-round-trip"

        loaded = harness.metadata.get_document("doc-round-trip")
        assert loaded is not None
        assert loaded.document is not None
        assert loaded.document.to_dict() == original.to_dict()
        assert canonical_json(loaded.document.to_dict()) == canonical_json(original.to_dict())

    def test_every_locator_field_survives_individually(self, harness):
        """Field-by-field, because one wrong field is one wrong citation.

        A single dict comparison says *that* something differs. This says
        *which* locator field, which is the difference between a test failure
        and a debugging session.
        """
        harness.metadata.upsert_document(sample_record("doc-fields"))
        loaded = harness.metadata.get_document("doc-fields")
        locators = [
            element.locator
            for element in (loaded.document.elements if loaded and loaded.document else [])
        ]
        # Keyed by page rather than by kind: the fixture carries two
        # ``pdf_page`` locators, and collapsing them into a dict entry would
        # test one of them and call it the other.
        assert {locator.page for locator in locators if locator.kind == "pdf_page"} == {1, 2}
        boxed = [item for item in locators if item.bbox is not None][0]
        assert (boxed.page, boxed.bbox.x0, boxed.bbox.y0) == (2, 61.5, 142.25)
        assert (boxed.bbox.x1, boxed.bbox.y1, boxed.bbox.relative) == (361.5, 190.75, False)
        slides = [item for item in locators if item.kind == "slide"]
        assert (slides[0].slide, slides[0].element) == (4, 11)
        offsets = [item for item in locators if item.kind == "txt_offset"]
        assert (offsets[0].offset_start, offsets[0].offset_end) == (4120, 4180)
        # And the units table, which is what an evidence request resolves
        # against rather than the element table.
        assert loaded.document.units[0].locator.page == 2
        assert loaded.document.units[0].label == "Page 2"

    def test_heading_path_survives_as_an_ordered_tuple(self, harness):
        harness.metadata.upsert_document(sample_record("doc-headings"))
        loaded = harness.metadata.get_document("doc-headings")
        headings = [element.heading_path for element in loaded.document.elements]
        assert ("Introduction", "Background") in headings

    def test_a_missing_document_is_none_and_not_an_error(self, harness):
        assert harness.metadata.get_document("doc-never-existed") is None

    def test_writing_the_same_document_twice_updates_it(self, harness):
        first = harness.metadata.upsert_document(sample_record("doc-update"))
        assert first.version == 1

        changed = sample_record("doc-update")
        changed.document.elements[0].text = "Attention Is All You Need (revised)"
        changed.metadata["revision"] = 2
        second = harness.metadata.upsert_document(changed)

        # The version is the store's, not the caller's: T2.2's optimistic
        # lock depends on it being bumped on every write, and a test that
        # only checked the text would let a store that never bumps it pass.
        assert second.version == 2
        assert second.metadata["revision"] == 2

        loaded = harness.metadata.get_document("doc-update")
        assert loaded.version == 2
        assert loaded.document.elements[0].text == "Attention Is All You Need (revised)"
        assert harness.metadata.count_documents() == 1

    def test_documents_list_newest_first_with_a_stable_tiebreak(self, harness):
        for index in range(3):
            record = sample_record(f"doc-{index}")
            record.created_at = f"2026-09-2{index}T00:00:00+00:00"
            harness.metadata.upsert_document(record)

        listed = [record.document_id for record in harness.metadata.list_documents(limit=2)]
        assert listed == ["doc-2", "doc-1"]

        # Two rows written in the same second are ordinary under a concurrent
        # index. Without the id tiebreak the two tiers can page the same data
        # in different orders, and the difference only shows up in production.
        for suffix in ("a", "b"):
            record = sample_record(f"doc-tie-{suffix}")
            record.created_at = "2026-09-29T12:00:00+00:00"
            harness.metadata.upsert_document(record)
        tied = [
            record.document_id
            for record in harness.metadata.list_documents(limit=2)
            if record.document_id.startswith("doc-tie")
        ]
        assert tied == ["doc-tie-a", "doc-tie-b"]

    def test_deleting_twice_says_which_it_was(self, harness):
        harness.metadata.upsert_document(sample_record("doc-delete"))
        assert harness.metadata.delete_document("doc-delete") is True
        assert harness.metadata.delete_document("doc-delete") is False
        assert harness.metadata.get_document("doc-delete") is None

    def test_deleting_a_document_takes_its_chunks_and_facts(self, harness):
        harness.metadata.upsert_document(sample_record("doc-orphan"))
        harness.metadata.put_chunk("doc-orphan", sample_chunk("chunk-orphan"))
        harness.metadata.put_facts("doc-orphan", sample_facts())

        harness.metadata.delete_document("doc-orphan")

        assert harness.metadata.get_chunk("chunk-orphan") is None
        assert harness.metadata.get_facts("doc-orphan") is None
        assert harness.metadata.list_chunks("doc-orphan") == []

    # -- chunks ------------------------------------------------------------

    def test_a_chunk_survives_the_round_trip_with_its_locator(self, harness):
        harness.metadata.upsert_document(sample_record("doc-chunks"))
        original = sample_chunk("chunk-round")
        harness.metadata.put_chunk("doc-chunks", original)

        loaded = harness.metadata.get_chunk("chunk-round")
        assert loaded is not None
        assert loaded.to_dict() == original.to_dict()
        assert loaded.locator.page == 2
        assert loaded.locator.bbox.x0 == 61.5
        assert loaded.heading_path == ("Introduction", "Background")
        assert loaded.element_ids == ("el-1", "el-2")

    def test_chunks_come_back_in_reading_order(self, harness):
        harness.metadata.upsert_document(sample_record("doc-order"))
        for order in (2, 0, 1):
            harness.metadata.put_chunk("doc-order", sample_chunk(f"chunk-{order}", order))
        assert [chunk.chunk_id for chunk in harness.metadata.list_chunks("doc-order")] == [
            "chunk-0",
            "chunk-1",
            "chunk-2",
        ]

    def test_storing_a_chunk_twice_replaces_it(self, harness):
        harness.metadata.upsert_document(sample_record("doc-replace"))
        harness.metadata.put_chunk("doc-replace", sample_chunk("chunk-replace"))
        updated = Chunk.from_dict(sample_chunk("chunk-replace").to_dict())
        updated.text = "重写后的段落。"
        harness.metadata.put_chunk("doc-replace", updated)
        assert harness.metadata.get_chunk("chunk-replace").text == "重写后的段落。"

    def test_deleting_chunks_reports_how_many(self, harness):
        harness.metadata.upsert_document(sample_record("doc-dropchunks"))
        for index in range(3):
            harness.metadata.put_chunk("doc-dropchunks", sample_chunk(f"chunk-{index}", index))
        assert harness.metadata.delete_chunks("doc-dropchunks") == 3
        assert harness.metadata.delete_chunks("doc-dropchunks") == 0

    # -- facts -------------------------------------------------------------

    def test_the_fact_layer_survives_the_round_trip(self, harness):
        harness.metadata.upsert_document(sample_record("doc-facts"))
        original = sample_facts()
        harness.metadata.put_facts("doc-facts", original)

        loaded = harness.metadata.get_facts("doc-facts")
        assert loaded is not None
        assert loaded.to_dict() == original.to_dict()
        # A fact with a value but no citation is still not grounded. The
        # distinction is what lets the UI say "not stated in this paper"
        # instead of rendering an empty section that reads like an oversight,
        # so the ungrounded case has to survive the round trip intact.
        assert loaded.limitations[0].is_grounded is False
        assert loaded.limitations[0].extraction_status == "extracted"
        assert loaded.research_question.is_grounded is True
        assert loaded.research_question.evidence_ids == ("chunk-1",)
        expected_coverage = sum(1 for fact in loaded.all_facts() if fact.is_grounded) / len(
            loaded.all_facts()
        )
        assert loaded.coverage == pytest.approx(expected_coverage)
        assert 0.0 < loaded.coverage < 1.0

    def test_extracting_facts_twice_replaces_rather_than_accumulates(self, harness):
        harness.metadata.upsert_document(sample_record("doc-refacts"))
        harness.metadata.put_facts("doc-refacts", sample_facts())

        second = sample_facts()
        second.contributions = []
        second.metrics = []
        harness.metadata.put_facts("doc-refacts", second)

        loaded = harness.metadata.get_facts("doc-refacts")
        assert loaded.contributions == []
        assert loaded.metrics == []

    def test_facts_for_a_document_that_was_never_extracted(self, harness):
        assert harness.metadata.get_facts("doc-no-facts") is None

    # -- lifecycle ---------------------------------------------------------

    def test_initialize_is_idempotent(self, harness):
        harness.metadata.upsert_document(sample_record("doc-idem"))
        harness.metadata.initialize()
        harness.metadata.initialize()
        assert harness.metadata.count_documents() == 1

    def test_concurrent_writes_all_land(self, harness):
        """Every write survives, and none of them half-applied.

        Threads rather than processes because the tier the personal one runs
        on cannot fork cheaply and because the failure this guards is a
        lost update, not a visibility problem. Each thread writes its own
        document *and* reads the count back, so a store that serialised
        writes but lost one would be caught by the final count rather than by
        a lock assertion.
        """
        workers = 8
        per_worker = 6
        errors: list[BaseException] = []
        barrier = threading.Barrier(workers)

        def _write(worker: int) -> None:
            try:
                barrier.wait(timeout=10)
                for index in range(per_worker):
                    identifier = f"doc-w{worker}-{index}"
                    harness.metadata.upsert_document(sample_record(identifier))
                    harness.metadata.put_chunk(identifier, sample_chunk(f"chunk-{worker}-{index}"))
            except BaseException as exc:  # noqa: BLE001 - reported below, not swallowed
                errors.append(exc)

        threads = [threading.Thread(target=_write, args=(worker,)) for worker in range(workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)

        assert errors == []
        assert harness.metadata.count_documents() == workers * per_worker
        for worker in range(workers):
            chunks = harness.metadata.list_chunks(f"doc-w{worker}-0")
            assert len(chunks) == 1

    def test_a_record_without_a_document_still_round_trips(self, harness):
        """The payload is optional; the row is not.

        A document record can be written before a parse finishes — which is
        exactly when a job system wants to record that a document exists —
        so a store that required the payload would lose the one row written
        at the most interesting moment.
        """
        from paperwhisperer.infra.stores import DocumentRecord

        harness.metadata.upsert_document(
            DocumentRecord(document_id="doc-empty", source_filename="pending.pdf")
        )
        loaded = harness.metadata.get_document("doc-empty")
        assert loaded.source_filename == "pending.pdf"
        assert loaded.document_id == "doc-empty"

    def test_an_empty_document_id_is_refused(self, harness):
        from paperwhisperer.infra.stores import DocumentRecord

        with pytest.raises(ValueError, match="document_id"):
            harness.metadata.upsert_document(DocumentRecord(document_id="  "))

    def test_an_unknown_status_degrades_to_indexed(self, harness):
        from paperwhisperer.infra.stores import DocumentRecord

        record = DocumentRecord(document_id="doc-status", status="not-a-status")
        assert record.status == "indexed"
        assert json.dumps(record.to_dict())

    def test_a_document_with_no_elements_is_storable(self, harness):
        from paperwhisperer.infra.stores import DocumentRecord

        harness.metadata.upsert_document(
            DocumentRecord(
                document_id="doc-noel", document=ParsedDocument(document_id="doc-noel")
            )
        )
        loaded = harness.metadata.get_document("doc-noel")
        assert loaded.document.elements == []
        assert loaded.document.locator_coverage == 0.0


class TestMetadataSqlite(StoreContract):
    tier = TIER_PERSONAL


@requires_postgres
class TestMetadataPostgres(StoreContract):
    tier = TIER_HOSTED
