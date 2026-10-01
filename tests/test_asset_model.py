"""T2.2's asset model, asserted once and run against both tiers.

The plan's acceptance criteria for this task are three sentences, and each one
names a thing that can only be believed if it is tested:

*迁移脚本幂等*
    Idempotence of the schema and of the session migration. Both are tested by
    running the thing twice and comparing the result to the first run's — a
    table count, a ``migrated_at`` timestamp — rather than by asserting that a
    function was called once.
*并发读写无 lost update*
    A real concurrency test with real threads, a real barrier, and a real
    compare-and-set. The failure being prevented is silent by construction: two
    requests read version 1, both write, one wins, and the UI shows the loser's
    data until the next read. A claim in a docstring cannot prevent that, and a
    test that only checks ``VersionConflict`` is *raised* has not shown that two
    writers cannot both succeed.
*旧 session 首次读取自动迁移*
    A migration that is lossless, idempotent, and leaves the file alone. The
    third property is the one that is easy to assert badly — "the file still
    exists" is satisfied by a migration that also truncated it — so the file's
    bytes and its modification time are compared before and after.

**Why the contract is one class and the two subclasses are empty.** T1.1
established the arrangement and the reasoning has not changed: two hand-copied
files drift within a week, one gains a test the other does not, and the
acceptance criterion keeps passing on the tier nobody is running. So the
behaviour lives in :class:`AssetModelContract`, and
:class:`TestAssetModelPersonal` / :class:`TestAssetModelHosted` carry nothing but
a tier name. Anything a single tier must do is a function outside the contract,
named so that nobody can mistake it for shared behaviour.

What is *not* in the contract: the object store is the local filesystem on both
tiers today, because S3 is P4. That is not an exemption — the object store is
reached through the same :class:`~paperwhisperer.infra.object_store.ObjectStore`
protocol on both tiers, and a filesystem implementation on both is what makes
"the Hosted tier uses the same interface" true rather than aspirational. The
Hosted *tables* differ for real, which is where the concurrency and migration
tests earn their second run.

The skip is actionable. ``PAPERWHISPERER_TEST_PG_DSN`` is the switch, the same
variable the store tier's suite uses, and the reason string names the compose
command — because a skip nobody can act on is a test that stopped being run.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import threading
from dataclasses import dataclass
from typing import Any

import pytest

from paperwhisperer.core import config as core_config
from paperwhisperer.infra import object_store as object_store_module
from paperwhisperer.infra import tables as tables_module
from paperwhisperer.infra.object_store import (
    LAYER_CANONICAL,
    LAYER_DERIVED,
    LAYER_ORIGINAL,
    LAYER_PARSED,
    LAYER_SPECS,
    LAYERS,
    FilesystemObjectStore,
    ImmutableObject,
    LayerMismatch,
    ObjectNotFound,
    ObjectRef,
    ObjectStore,
    UnsafeKey,
    build_object_store,
    digest_from_key,
    digest_key,
    validate_object_key,
)
from paperwhisperer.infra.tables import (
    EXTERNAL_TABLES,
    EXTRA_ANALYSIS_RUN_COLUMNS,
    REQUIRED_ANALYSIS_RUN_COLUMNS,
    TABLE_BY_NAME,
    TABLES,
    DuplicateRow,
    MissingColumn,
    TableStore,
    UnknownColumn,
    VersionConflict,
    build_table_store,
)
from paperwhisperer.ingestion.schema import (
    BoundingBox,
    Element,
    ParsedDocument,
    SourceLocator,
)

from .test_stores_tiers import (
    TIER_HOSTED,
    TIER_PERSONAL,
    open_bundle,
    requires_postgres,
)

PG_DSN_ENV = "PAPERWHISPERER_TEST_PG_DSN"

#: The bytes of a "user's upload". Not a real PDF: the object store does not
#: parse anything, and a fixture that had to be a valid PDF would make these
#: tests fail for Docling reasons when the subject is the store.
UPLOAD_BYTES = b"%PDF-1.7\n% a user's original upload, which must never change\n"
REPARSE_BYTES = json.dumps(
    {"schema_version": "1", "note": "re-parsed with a different parser"},
    ensure_ascii=False,
).encode("utf-8")


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------


@dataclass
class _ObjectSettings:
    """The object store's configuration keys, and nothing else.

    A stand-in rather than the real :mod:`paperwhisperer.core.config`, for the
    reason ``tests/test_stores_tiers.py`` gives: the real config is a module,
    and an object that cannot be a neighbour cannot leak.
    """

    OBJECT_STORE_BACKEND: str = "filesystem"
    OBJECT_STORE_ROOT: str = "context/objects"
    OBJECT_STORE_FSYNC: bool = True


@dataclass
class AssetModel:
    """One tier's three things that must behave the same: tables, objects, bundle.

    A bundle rather than free variables because they share a lifecycle and,
    more importantly, share a *database*: the tables live in the same SQLite file
    as the store tier's documents, and that is what makes "back up one data
    directory" a true sentence.
    """

    tier: str
    tables: TableStore
    objects: Any
    bundle: Any

    def close(self) -> None:
        self.tables.close()
        self.objects.close()


def open_asset_model(tier: str, tmp_path: Any):
    """Build one tier's asset model, then tear it down. The one implementation.

    The store bundle is borrowed from ``tests/test_stores_tiers.open_harness``
    rather than rebuilt, so the Hosted tier gets the same per-test schema and the
    same teardown the store suite relies on: a Hosted test that leaked a schema
    would fail the *next* test with an index-name collision, and the failure
    would point at the wrong file.
    """
    if tier == TIER_HOSTED:
        pytest.skip(
            f"未设置 {PG_DSN_ENV}。运行 docker compose up -d db 后设置 "
            f"{PG_DSN_ENV}=postgresql://paperwhisperer:paperwhisperer@localhost:5432/"
            "paperwhisperer 即可启用 Hosted 档位测试。"
        )
    # open_bundle, not open_harness: the harness is a generator that tears the
    # schema down in its own finally, and this fixture has real setup of its own
    # to do between opening the bundle and yielding. Reusing open_bundle is how
    # the Hosted tier gets the same per-test schema teardown the store suite
    # relies on, rather than a second implementation of it that would rot.
    harness = open_bundle(tier, tmp_path)
    model = AssetModel(
        tier=tier,
        tables=build_table_store(bundle=harness.bundle),
        objects=build_object_store(_ObjectSettings(OBJECT_STORE_ROOT=str(tmp_path / "objects"))),
        bundle=harness.bundle,
    )
    try:
        yield model
    finally:
        model.close()
        harness.close()


@pytest.fixture
def model(request, tmp_path):
    """The asset model for whichever tier the test class names.

    ``request.node.cls.tier`` is the whole mechanism: a class attribute on each
    empty subclass, so the two tiers differ by one line and cannot differ by one
    test.
    """
    yield from open_asset_model(request.node.cls.tier, tmp_path)


@pytest.fixture
def personal_model(tmp_path):
    """A personal-tier asset model for a test that is not a tier contract.

    Cascade behaviour is the same on both tiers, but these are not contract tests
    — they would be a second copy of the contract body with a different name,
    which is the drift the two-subclass arrangement exists to prevent. So they
    are written once against the personal tier, and the Hosted half of that
    behaviour is covered by the contract's own delete tests.
    """
    yield from open_asset_model(TIER_PERSONAL, tmp_path)


# --------------------------------------------------------------------------
# Fixtures: values
# --------------------------------------------------------------------------


def canonical_bytes(document: ParsedDocument) -> bytes:
    """A canonical document as the canonical layer stores it.

    Serialized the way both store backends serialize, so the bytes are what a
    real canonical object would be. A test that stored a dict here would pass on
    a store that silently mangled a locator.
    """
    return json.dumps(
        document.to_dict(), sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def sample_document(document_id: str = "doc-asset") -> ParsedDocument:
    """A canonical document with a resolvable locator, for the layer tests.

    A real :class:`ParsedDocument` rather than a dict, because the canonical
    layer's job is to hold exactly this object and a test that stores a dict
    would pass on a store that silently mangled it.
    """
    return ParsedDocument(
        document_id=document_id,
        source_filename="论文.pdf",
        media_type="application/pdf",
        sha256="c" * 64,
        page_count=2,
        elements=[
            Element(
                role="paragraph",
                text="本文提出一种新的架构。",
                order=0,
                locator=SourceLocator(
                    kind="pdf_page",
                    page=1,
                    bbox=BoundingBox(x0=10.0, y0=20.0, x1=300.0, y1=40.0),
                ),
            )
        ],
        parser_name="docling",
        parser_version="2.0",
        options_hash="cafe",
        created_at="2026-09-29T00:00:00+00:00",
    )


def sample_analysis_run_columns() -> dict[str, Any]:
    """Exactly what :meth:`AnalysisRun.as_analysis_run_columns` returns.

    Built by *calling the method* on a real ``AnalysisRun`` rather than by
    restating its keys. If T1.8 renames or adds a column, this fixture changes
    with it and the write fails — which is the point. A hand-written dict of the
    same keys would keep passing forever while the real producer drifted, and
    the drift would only be visible in production.
    """
    from paperwhisperer.analysis.paper_facts import AnalysisRun

    run = AnalysisRun(
        run_id="run-1",
        document_id="doc-asset",
        created_at="2026-09-29T00:00:00+00:00",
        model="gpt-4o-mini",
        prompt_version="facts-v1",
        evidence_version="ev-1",
        fact_count=8,
        grounded_fact_count=6,
        coverage=0.75,
        not_stated_fields=("datasets",),
        degradations=("事实层有 1 个字段未在文中陈述。",),
        retrieval_record_ids=("rec-1", "rec-2"),
        legacy_llm_calls=9,
        legacy_input_tokens=12345,
    )
    run.add_call(
        "fact_layer",
        "system prompt",
        "user prompt",
        "some output",
        model="gpt-4o-mini",
        price_in=0.15,
        price_out=0.6,
    )
    return run.as_analysis_run_columns()


def sample_embedding_columns() -> dict[str, Any]:
    """Exactly what :meth:`EmbeddingUsage.as_analysis_run_columns` returns."""
    from paperwhisperer.llm.embeddings import EmbeddingUsage

    return EmbeddingUsage(
        provider="openai",
        model="text-embedding-3-small",
        dimension=1536,
        index_version="openai:text-embedding-3-small@1",
        requests=2,
        texts=120,
        cache_hits=100,
        input_tokens=20480,
        tokens_reported=True,
        price_per_mtok=0.02,
    ).as_analysis_run_columns()


def read_bytes(path: str) -> bytes:
    """Read a file's bytes through a context manager.

    The tests that use this are asserting that a file was *not* rewritten, so
    they read it directly rather than through the object store: a store that
    served a cached value would pass a test written against the store.
    """
    with open(path, "rb") as handle:
        return handle.read()


def write_session_file(directory: Any, session_id: str, payload: dict) -> str:
    """Write a session file the way the app does, and return its path.

    Through the real writer rather than ``json.dump``, so the bytes on disk are
    the bytes the migration will meet in production, and so the test is not
    asserting that a migration can read a file only this test could write.
    """
    from paperwhisperer.core.text import atomic_write_json

    path = os.path.join(str(directory), f"{session_id}.json")
    atomic_write_json(path, payload)
    return path


def legacy_session_payload(session_id: str, **overrides: Any) -> dict:
    """A session file of the shape the pre-T2.2 build wrote.

    Deliberately carrying the awkward parts: a QA turn with an ``answer_mode``, a
    reading-queue item with identity fields, a token hash, and a
    ``suggested_questions`` list. A fixture trimmed to the easy fields would let
    a migration that dropped half the payload pass.
    """
    payload = {
        "session_id": session_id,
        "source_filename": "迁移测试.pdf",
        "generated_at": "2026-09-28T10:00:00",
        "created_at": "2026-09-28T10:00:00",
        "updated_at": "2026-09-28T10:05:00",
        "expires_at": "2099-01-01T00:00:00",
        "document_content": "全文内容，不应丢失。",
        "document_excerpt": "全文内容…",
        "qa_history": [
            {
                "question": "这篇论文的核心贡献是什么？",
                "answer": "提出一种纯注意力架构。",
                "answer_mode": "direct_quote",
                "timestamp": "2026-09-28T10:01:00",
            },
            {
                "question": "用了什么数据集？",
                "answer": "文中未说明。",
                "answer_mode": "insufficient_evidence",
                "timestamp": "2026-09-28T10:02:00",
            },
        ],
        "paper_search": {
            "last_query": "transformer",
            "last_results": [],
            "last_recommendation": {},
            "reading_queue": [
                {
                    "title": "Attention Is All You Need",
                    "doi": "10.5555/3295222.3295349",
                    "arxiv_id": "1706.03762",
                    "year": 2017,
                    "venue": "NeurIPS",
                    "authors": ["Vaswani", "Shazeer"],
                    "url": "https://example.org/attention",
                    "saved_at": "2026-09-28T10:03:00",
                }
            ],
        },
        "session_auth": {"token_hash": hashlib.sha256(b"token-1").hexdigest()},
        "analysis": {
            "summary": "摘要：一篇关于注意力的论文。",
            "quotes": "",
            "mindmap": "",
            "mermaid": "graph TD;",
            "evaluation": "",
            "research_brief": "",
            "sections": {"方法": "多头自注意力"},
            "char_count": 4321,
            "suggested_questions": ["它的复杂度是多少？"],
            "next_actions": [],
            "analysis_status": {"summary": "done"},
        },
    }
    payload.update(overrides)
    return payload


# --------------------------------------------------------------------------
# The contract, written once
# --------------------------------------------------------------------------


class AssetModelContract:
    """Behaviour that must be identical on every tier."""

    # -- layering ----------------------------------------------------------

    def test_the_four_layers_are_separate_directories(self, model):
        """Sibling roots, not prefixes of one root.

        This is the property the whole taxonomy rests on. A store with one root
        and four prefixes still needs every read to name a layer correctly, and
        a bug in that naming would be a wrong *answer* rather than a missing
        one. Four directories make a wrong answer impossible: there is no
        canonical object for a derived key to be found under.
        """
        directories = {
            layer: os.path.realpath(model.objects.layer_dir(layer)) for layer in LAYERS
        }
        assert len(set(directories.values())) == len(LAYERS)
        for layer, path in directories.items():
            assert os.path.isdir(path), f"{layer} 层目录不存在：{path}"
        assert all(
            os.path.dirname(path) == os.path.realpath(model.objects.root)
            for path in directories.values()
        )

    def test_a_canonical_read_never_returns_a_derived_object(self, model):
        """The failure the layering exists to prevent, asserted byte for byte.

        The same key is written into both layers. The canonical read returns the
        canonical bytes, the listing of the canonical layer does not mention the
        derived object, and the derived one is still readable through its own
        layer. If any of those three were true the other two would have to be
        false, which is the point of writing the test this way.
        """
        model.objects.put(LAYER_CANONICAL, b"canonical bytes", key="same.json")
        model.objects.put(LAYER_DERIVED, b"derived bytes", key="same.json")

        assert model.objects.get(LAYER_CANONICAL, "same.json") == b"canonical bytes"
        assert model.objects.get(LAYER_DERIVED, "same.json") == b"derived bytes"
        assert [ref.key for ref in model.objects.list_objects(LAYER_CANONICAL)] == ["same.json"]
        assert model.objects.head(LAYER_CANONICAL, "same.json").sha256 == hashlib.sha256(
            b"canonical bytes"
        ).hexdigest()

    def test_a_ref_cannot_be_read_as_a_different_layer(self, model):
        """The second half of the taxonomy, at the API boundary.

        Isolation by directory stops a *key* from crossing layers. This stops a
        *ref* from crossing, which is the case that matters in calling code: a
        function that received a derived artifact and needs the canonical one
        would otherwise extract the key, call ``get`` and be handed the wrong
        bytes with no error at all.
        """
        derived = model.objects.put(LAYER_DERIVED, b"derived", key="report.json")

        with pytest.raises(LayerMismatch):
            model.objects.get_ref(derived, expect_layer=LAYER_CANONICAL)
        # Readable through its own layer: a ref is not a token that only opens
        # one particular door, it is an address that says which door it is.
        assert model.objects.get_ref(derived) == b"derived"
        canonical = model.objects.put(LAYER_CANONICAL, b"canonical", key="report.json")
        assert model.objects.get_ref(canonical, expect_layer=LAYER_CANONICAL) == b"canonical"

    def test_a_missing_object_is_an_error_not_an_empty_answer(self, model):
        with pytest.raises(ObjectNotFound):
            model.objects.get(LAYER_CANONICAL, "nothing-here.json")
        assert model.objects.head(LAYER_CANONICAL, "nothing-here.json") is None
        assert model.objects.exists(LAYER_CANONICAL, "nothing-here.json") is False

    def test_keys_cannot_escape_their_layer(self, model):
        """A key is a path on this tier, so a traversal would write outside the
        data directory the owner is told to back up.

        The same key is an opaque name on S3, so a store that accepted one would
        mean the two tiers disagree about which object a key names — which is
        why the refusals are here rather than in the P4 implementation.
        """
        for key in ("../escape.json", "a/../../escape.json", "/absolute.json", "a//b.json", ""):
            with pytest.raises(UnsafeKey):
                model.objects.put(LAYER_DERIVED, b"x", key=key)
        # A CJK filename is a legal key: the validation is structural, and a
        # store that refused them would push callers into escaping.
        assert model.objects.put(LAYER_DERIVED, b"x", key="导出/摘要.json").key == "导出/摘要.json"

    def test_a_failed_write_leaves_no_partial_object(self, model):
        """Atomicity, tested at the point of failure rather than described.

        A generator that raises half-way through a stream is a real event — a
        disk filling up, a client disconnecting mid-upload, an interrupt — and
        the guarantee is that no object appears under its final name and no
        staging file is left behind. ``list_objects`` skips staging files
        precisely so that this assertion is meaningful: a leftover temp file that
        the listing ignored would still be a leftover temp file.
        """
        staged = model.objects._stage  # the one private call, to drive a failing stream

        def _exploding() -> Any:
            yield b"first half "
            raise OSError("模拟磁盘写满。")

        with pytest.raises(OSError):
            staged(LAYER_CANONICAL, _exploding())
        assert model.objects.list_objects(LAYER_CANONICAL) == []
        assert model.objects.usage()[LAYER_CANONICAL].objects == 0

    # -- the original is never touched -------------------------------------

    def test_a_reparse_replaces_the_canonical_version_and_not_the_upload(self, model):
        """The layering's reason for existing, asserted three ways at once.

        Bytes, modification time and object count. The modification time is the
        one that is not obvious: it changes only if something actually wrote the
        file, so a re-parse that rewrote the user's upload with identical content
        would leave the *bytes* looking untouched while proving nothing. The
        count catches the opposite failure — a re-parse that "safely" wrote a
        second copy under a new name, which is how a storage bill doubles with
        no visible symptom.
        """
        upload = model.objects.put(LAYER_ORIGINAL, UPLOAD_BYTES, suffix=".pdf")
        path = model.objects.local_path(upload.layer, upload.key)
        before_bytes = read_bytes(path)
        before_mtime = os.stat(path).st_mtime_ns
        before_usage = model.objects.usage()[LAYER_ORIGINAL]

        first = model.objects.put(LAYER_CANONICAL, canonical_bytes(sample_document()), suffix=".json")
        first_version = model.tables.add_document_version(
            {
                "document_id": "doc-asset",
                "sha256": upload.sha256,
                "original_key": upload.key,
                "canonical_key": first.key,
                "parser_name": "docling",
                "parse_status": "success",
            }
        )
        # A re-parse: a different parser, a different canonical document, the
        # same upload.
        reparsed = sample_document()
        reparsed.parser_name = "mineru"
        second = model.objects.put(LAYER_CANONICAL, canonical_bytes(reparsed), suffix=".json")
        second_version = model.tables.add_document_version(
            {
                "document_id": "doc-asset",
                "sha256": upload.sha256,
                "original_key": upload.key,
                "canonical_key": second.key,
                "parser_name": "mineru",
                "parse_status": "success",
                "used_cloud": True,
            }
        )

        assert first.key != second.key
        assert read_bytes(path) == before_bytes == UPLOAD_BYTES
        assert os.stat(path).st_mtime_ns == before_mtime
        assert model.objects.usage()[LAYER_ORIGINAL] == before_usage
        # The history is a table's job, not a directory's: both versions exist
        # as rows, and exactly one is current.
        versions = model.tables.list_document_versions("doc-asset")
        assert [row["version_no"] for row in versions] == [1, 2]
        assert [row["is_current"] for row in versions] == [False, True]
        assert model.tables.get_current_document_version("doc-asset")["version_id"] == (
            second_version["version_id"]
        )
        assert first_version["canonical_key"] != second_version["canonical_key"]
        # Both canonical objects are still readable, because a user comparing two
        # parses is a thing T3.4 asks for.
        assert model.objects.get(LAYER_CANONICAL, first.key) is not None

    def test_the_original_layer_is_write_once_and_force_deleteable(self, model):
        model.objects.put(LAYER_ORIGINAL, UPLOAD_BYTES, suffix=".pdf")
        # Same bytes under a name the caller chose: the content digest matches,
        # so this is a no-op rather than a refusal. Content addressing is what
        # makes a re-upload of the same file free.
        again = model.objects.put(LAYER_ORIGINAL, UPLOAD_BYTES, suffix=".pdf")
        assert again.deduplicated is True

        with pytest.raises(ImmutableObject):
            model.objects.put(LAYER_ORIGINAL, b"different bytes", key=again.key)
        with pytest.raises(ImmutableObject):
            model.objects.delete(LAYER_ORIGINAL, again.key)
        # A genuine delete request is allowed to be explicit.
        assert model.objects.delete(LAYER_ORIGINAL, again.key, force=True) is True
        assert model.objects.exists(LAYER_ORIGINAL, again.key) is False
        # The other layers are replaceable, which is what a re-parse needs.
        model.objects.put(LAYER_CANONICAL, b"a", key="c.json")
        assert model.objects.put(LAYER_CANONICAL, b"b", key="c.json").size == 1

    def test_content_addressing_stores_identical_bytes_once(self, model):
        """Dedupe, asserted by the object's *modification time*.

        Counting files would also pass if the store wrote a second copy and then
        deleted it. The modification time only stays put if the write never
        happened, which is the property that makes "a re-parse did not touch the
        upload" checkable rather than merely likely.
        """
        first = model.objects.put(LAYER_PARSED, b"same bytes", suffix=".json")
        path = model.objects.local_path(first.layer, first.key)
        mtime = os.stat(path).st_mtime_ns
        usage = model.objects.usage()[LAYER_PARSED]

        second = model.objects.put(LAYER_PARSED, b"same bytes", suffix=".json")
        assert second.key == first.key
        assert second.sha256 == first.sha256
        assert second.deduplicated is True
        assert os.stat(path).st_mtime_ns == mtime
        assert model.objects.usage()[LAYER_PARSED] == usage
        assert model.objects.usage()[LAYER_PARSED].objects == 1

        different = model.objects.put(LAYER_PARSED, b"other bytes", suffix=".json")
        assert different.key != first.key
        assert different.deduplicated is False
        assert model.objects.usage()[LAYER_PARSED].objects == 2

    def test_a_digest_finds_its_object_without_a_listing(self, model):
        """Content addressing is what makes the strict layering affordable.

        A store where "do we already hold these bytes" meant a directory walk
        would push every caller into scanning, and a caller that scans eventually
        caches the answer and serves it stale. Asserted through the public
        method rather than by timing anything: the key is arithmetic, so the
        lookup is one existence check and the result is exact.
        """
        written = model.objects.put(LAYER_CANONICAL, b"canonical", suffix=".json")
        found = model.objects.find_by_digest(LAYER_CANONICAL, written.sha256)
        assert found is not None
        assert found.key == written.key
        assert model.objects.find_by_digest(LAYER_DERIVED, written.sha256) is None
        assert model.objects.find_by_digest(LAYER_CANONICAL, "0" * 64) is None

    # -- optimistic locking ------------------------------------------------

    def test_two_writers_at_the_same_version_cannot_both_win(self, model):
        """The acceptance criterion, with real threads and a real barrier.

        What this proves, precisely: every thread reads the row *first* (before
        the barrier), so all of them hold the same ``version``; the barrier
        releases them together so the writes genuinely race; and exactly one
        write succeeds. If the compare-and-set were not there — if the update
        were a plain ``UPDATE`` or a read-modify-write in Python — all eight
        writes would report success and the final row would be whichever one
        happened to be written last, with seven callers believing their data was
        stored. That is the silent failure the version column exists to prevent,
        and it cannot be seen in a sequential test.

        The winner is not asserted to be any particular thread. Which thread wins
        is the database's business; that exactly one does is ours.
        """
        conversation = model.tables.add_conversation({"session_id": "s-conc", "title": "base"})
        start_version = conversation["version"]
        writers = 8
        barrier = threading.Barrier(writers)
        outcomes: list[Any] = [None] * writers

        def _write(index: int) -> None:
            # Read before the barrier: every thread must be holding the same
            # version, or the test would be proving nothing.
            read = model.tables.get_conversation(conversation["conversation_id"])
            assert read["version"] == start_version
            try:
                barrier.wait(timeout=30)
            except threading.BrokenBarrierError:  # pragma: no cover - only on a hang
                outcomes[index] = ("barrier", None, None)
                return
            try:
                stored = model.tables.update_conversation(
                    conversation["conversation_id"],
                    {"title": f"writer-{index}"},
                    expected_version=read["version"],
                )
                outcomes[index] = ("ok", stored["version"], index)
            except VersionConflict as exc:
                outcomes[index] = ("conflict", exc.expected, exc.actual)

        threads = [threading.Thread(target=_write, args=(index,)) for index in range(writers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)

        assert all(outcome is not None for outcome in outcomes), outcomes
        winners = [item for item in outcomes if item[0] == "ok"]
        conflicts = [item for item in outcomes if item[0] == "conflict"]
        assert len(winners) == 1, f"应当恰好一个写入成功，实际：{outcomes}"
        assert len(conflicts) == writers - 1
        # Every loser is told what it lost to, which is what lets a caller decide
        # between re-reading and reporting a conflict rather than guessing.
        for _status, expected, actual in conflicts:
            assert expected == start_version
            assert actual == start_version + 1

        final = model.tables.get_conversation(conversation["conversation_id"])
        assert final["version"] == start_version + 1
        assert final["title"] == f"writer-{winners[0][2]}"
        # And the loser's data is nowhere: the row holds the winner's write, not
        # a mixture and not the last writer by accident.
        assert final["title"].startswith("writer-")

    def test_a_conflict_is_recoverable_by_re_reading(self, model):
        """The other half of the lock: it has to be a retry, not a dead end.

        A lock that made a writer give up would be a mutex, and a mutex is the
        wrong tool for a web request — the answer to "somebody else changed this
        row" is to look again and merge, which is what a caller can do here
        because the conflict told it which version it lost to.
        """
        conversation = model.tables.add_conversation({"session_id": "s-retry", "title": "base"})
        identifier = conversation["conversation_id"]

        model.tables.update_conversation(identifier, {"title": "first"}, expected_version=1)
        with pytest.raises(VersionConflict):
            model.tables.update_conversation(identifier, {"title": "second"}, expected_version=1)

        reloaded = model.tables.get_conversation(identifier)
        assert reloaded["title"] == "first"
        assert reloaded["version"] == 2
        retried = model.tables.update_conversation(
            identifier, {"title": "second"}, expected_version=reloaded["version"]
        )
        assert retried["title"] == "second"
        assert retried["version"] == 3

    def test_an_update_touches_only_the_columns_it_names(self, model):
        """The bug a default-filling update would cause, asserted.

        An update that fills unspecified columns with their defaults would erase a
        conversation's payload, token hash and migration bookkeeping while
        reporting success. Every value the row did not name must come back
        exactly as it was.
        """
        payload = {"document_content": "全文", "custom": [1, 2, 3]}
        conversation = model.tables.add_conversation(
            {
                "session_id": "s-partial",
                "title": "原始标题",
                "token_hash": "hash-1",
                "payload": payload,
            }
        )
        updated = model.tables.update_conversation(
            conversation["conversation_id"],
            {"title": "新标题"},
            expected_version=conversation["version"],
        )
        assert updated["title"] == "新标题"
        assert updated["payload"] == payload
        assert updated["token_hash"] == "hash-1"
        assert updated["source_migrated"] is False

    def test_a_caller_may_not_set_the_version_itself(self, model):
        """The lock cannot be defeated by passing a field that looks editable.

        ``version`` is a column, and a column a caller can write is a column
        somebody eventually writes. Accepting it would be a way to set the counter
        forward and overwrite somebody else's change with the version you wanted
        it to be — the same silent loss the lock prevents, one argument away.
        """
        conversation = model.tables.add_conversation({"session_id": "s-version", "title": "t"})
        with pytest.raises(UnknownColumn):
            model.tables.update_conversation(
                conversation["conversation_id"],
                {"version": 99},
                expected_version=conversation["version"],
            )
        assert model.tables.get_conversation(conversation["conversation_id"])["version"] == 1

    def test_updating_a_deleted_row_says_so_rather_than_conflicting(self, model):
        """The two failure modes mean opposite things, so they are not merged.

        A version conflict tells a caller to re-read and merge. A row that is
        gone tells it the thing it was editing no longer exists, and sending that
        caller into a retry loop against a row that will never come back is how a
        delete turns into a hang.
        """
        from paperwhisperer.infra.tables import UnknownRow

        conversation = model.tables.add_conversation({"session_id": "s-gone"})
        identifier = conversation["conversation_id"]
        assert model.tables.delete_conversation(identifier) is True
        assert model.tables.delete_conversation(identifier) is False
        with pytest.raises(UnknownRow):
            model.tables.update_conversation(identifier, {"title": "x"}, expected_version=1)

    def test_a_duplicate_key_is_refused(self, model):
        conversation = model.tables.add_conversation(
            {"conversation_id": "conv-fixed", "session_id": "s-dup"}
        )
        assert conversation["conversation_id"] == "conv-fixed"
        with pytest.raises(DuplicateRow):
            model.tables.add_conversation({"conversation_id": "conv-fixed", "session_id": "other"})

    # -- schema ------------------------------------------------------------

    def test_initialize_is_idempotent(self, model):
        """Acceptance criterion: 迁移脚本幂等.

        Run twice and compare the *live* schema, not the declaration — the point
        is that the second run found everything already there and changed
        nothing, which is what lets an operator run it because they are unsure.
        """
        before = {name: model.tables.live_columns(name) for name in model.tables.table_names()}
        model.tables.initialize()
        model.tables.initialize()
        after = {name: model.tables.live_columns(name) for name in model.tables.table_names()}
        assert after == before
        assert set(after) == {table.name for table in TABLES}

    def test_initialize_adds_a_missing_column_without_losing_rows(self, model):
        """A migration has to work on a database an older build made.

        Simulated by dropping a column and re-initializing. The row survives and
        the column comes back with its default: that is the whole difference
        between a migration and a bootstrap, and it is why there is no migration
        ledger anywhere in this module.
        """
        model.tables.add_conversation({"session_id": "s-additive", "title": "保留我"})
        table_name = "conversations"
        qualified = model.tables.dialect.qualify(table_name)
        with model.tables.read() as cursor:
            # Both engines spell this the same, which is the point of the
            # assertion: a column disappears from a live table and the migration
            # has to put it back on either one.
            cursor.execute(f"ALTER TABLE {qualified} DROP COLUMN title")
        assert "title" not in model.tables.live_columns(table_name)

        model.tables.initialize()
        assert "title" in model.tables.live_columns(table_name)
        conversation = model.tables.get_conversation_by_session("s-additive")
        assert conversation is not None
        assert conversation["title"] == ""

    def test_a_failed_write_leaves_no_partial_row(self, model):
        """A write that fails halfway must leave the previous state intact.

        The failure is the *second* item of a queue replacement, so the delete
        and the first insert have already run when the constraint refuses it. If
        the two were not in one transaction the user would come back to a
        half-empty reading queue, having been told nothing. The old queue being
        intact afterwards is the assertion; the raised error is incidental.
        """
        conversation = model.tables.add_conversation({"session_id": "s-atomic"})
        identifier = conversation["conversation_id"]
        model.tables.replace_reading_queue(
            identifier,
            [{"title": "保留的论文", "doi": "10.1/keep"}],
        )
        assert [item["title"] for item in model.tables.list_queue_items(identifier)] == [
            "保留的论文"
        ]

        from paperwhisperer.infra.tables import ConstraintViolation

        with pytest.raises(ConstraintViolation):
            model.tables.replace_reading_queue(
                identifier,
                [
                    {"title": "新的第一篇"},
                    {"title": "会被拒绝的", "status": "not-a-status"},
                ],
            )

        remaining = model.tables.list_queue_items(identifier)
        assert [item["title"] for item in remaining] == ["保留的论文"]
        assert len(remaining) == 1

    def test_a_failed_message_rebuild_leaves_the_transcript_intact(self, model):
        """The same guarantee for the transcript, whose loss is more visible.

        A transcript that came back empty would look exactly like a model that
        answered nothing, which is the one outcome the whole product exists to
        avoid — so the rebuild is atomic for the same reason the queue's is.
        """
        conversation = model.tables.add_conversation({"session_id": "s-transcript"})
        identifier = conversation["conversation_id"]
        model.tables.replace_messages(
            identifier,
            [{"role": "user", "content": "问题一"}, {"role": "assistant", "content": "回答一"}],
        )
        from paperwhisperer.infra.tables import ConstraintViolation

        with pytest.raises(ConstraintViolation):
            model.tables.replace_messages(
                identifier,
                [
                    {"role": "user", "content": "新问题"},
                    {"role": "not-a-role", "content": "坏行"},
                ],
            )
        assert [row["content"] for row in model.tables.list_messages(identifier)] == [
            "问题一",
            "回答一",
        ]

    def test_messages_are_numbered_in_reading_order(self, model):
        conversation = model.tables.add_conversation({"session_id": "s-order"})
        identifier = conversation["conversation_id"]
        for content in ("一", "二", "三"):
            model.tables.append_message(
                {"conversation_id": identifier, "role": "user", "content": content}
            )
        messages = model.tables.list_messages(identifier)
        assert [row["sequence"] for row in messages] == [1, 2, 3]
        assert [row["content"] for row in messages] == ["一", "二", "三"]
        rebuilt = model.tables.replace_messages(
            identifier,
            [{"role": "user", "content": "x"}, {"role": "user", "content": "y"}],
        )
        assert rebuilt == 2
        assert [row["sequence"] for row in model.tables.list_messages(identifier)] == [1, 2]

    def test_the_reading_queue_keeps_the_client_s_order(self, model):
        conversation = model.tables.add_conversation({"session_id": "s-queue"})
        identifier = conversation["conversation_id"]
        titles = ["第三篇", "第一篇", "第二篇"]
        assert model.tables.replace_reading_queue(
            identifier, [{"title": title} for title in titles]
        ) == 3
        assert [item["title"] for item in model.tables.list_queue_items(identifier)] == titles
        # Replacing with a shorter list is a whole replacement, not an append.
        model.tables.replace_reading_queue(identifier, [{"title": "只剩一篇"}])
        assert [item["title"] for item in model.tables.list_queue_items(identifier)] == ["只剩一篇"]

    # -- T1.8's analysis run ------------------------------------------------

    def test_an_analysis_run_takes_every_column_the_producers_name(self, model):
        """T1.8's row, written from the dicts the producers actually return.

        Not a hand-written dict of the same keys: both column sets are built by
        calling ``as_analysis_run_columns`` on a real ``AnalysisRun`` and a real
        ``EmbeddingUsage``, so a column added upstream without a home here fails
        this test instead of disappearing. Every value is read back, including
        the JSON arrays and the nullable costs, because a cost column that comes
        back as ``0.0`` when it was written as ``None`` is a cost dashboard that
        lies.
        """
        columns = sample_analysis_run_columns()
        embedding = sample_embedding_columns()
        assert set(REQUIRED_ANALYSIS_RUN_COLUMNS) == set(columns) | set(embedding)

        model.tables.record_analysis_run(columns, embedding=embedding)
        read = model.tables.get_analysis_run(columns["run_id"])
        assert read is not None
        for name, value in columns.items():
            if isinstance(value, float):
                assert read[name] == pytest.approx(value), f"列 {name} 往返后不一致"
            else:
                assert read[name] == value, f"列 {name} 往返后不一致"
        for name, value in embedding.items():
            assert read[name] == value, f"embedding 列 {name} 往返后不一致"
        assert read["version"] == 1
        assert read["updated_at"]
        assert model.tables.latest_analysis_run("doc-asset")["run_id"] == columns["run_id"]

    def test_an_analysis_run_missing_a_column_is_refused(self, model):
        """A dropped column has to fail at the write, loudly.

        The alternative — filling it with the column's default — is how a cost
        ledger starts reporting zeros for runs whose cost was never known. Both
        failure directions are tested: a required column absent, and a column the
        table does not have.
        """
        columns = sample_analysis_run_columns()
        missing = dict(columns)
        missing.pop("evidence_version")
        with pytest.raises(MissingColumn) as excinfo:
            model.tables.record_analysis_run(missing, embedding=sample_embedding_columns())
        assert "evidence_version" in str(excinfo.value)

        unknown = dict(columns)
        unknown["a_column_nobody_declared"] = 1
        with pytest.raises(UnknownColumn):
            model.tables.record_analysis_run(unknown, embedding=sample_embedding_columns())
        assert model.tables.get_analysis_run(columns["run_id"]) is None

    def test_the_cost_summary_keeps_an_unknown_cost_unknown(self, model):
        """T1.8's acceptance criterion has to be answerable from stored rows.

        The comparison is the point: ``legacy_llm_calls`` next to ``llm_calls``
        on the same row, summed across runs. And an unknown cost stays unknown
        rather than being rounded to a small number, because a partial sum of a
        partly-unknown total is a number that looks like a total.
        """
        known = dict(sample_analysis_run_columns())
        known["run_id"] = "run-known"
        model.tables.record_analysis_run(known, embedding=sample_embedding_columns())
        unknown = dict(known)
        unknown["run_id"] = "run-unknown"
        unknown["llm_cost"] = None
        model.tables.record_analysis_run(unknown, embedding=sample_embedding_columns())

        summary = model.tables.cost_summary("doc-asset")
        assert summary["runs"] == 2
        assert summary["llm_cost"] is None
        assert summary["llm_calls"] == known["llm_calls"] * 2
        assert summary["legacy_llm_calls"] == known["legacy_llm_calls"] * 2
        assert summary["legacy_input_tokens"] == known["legacy_input_tokens"] * 2

    def test_the_schema_gives_every_named_column_a_real_home(self, model):
        """Checked against the live DDL, not against the declaration.

        The declaration is what the module believes; the live column set is what
        the database has after ``initialize()`` ran. Comparing those two, and
        comparing both against the two producer methods, is what makes "no
        column is silently dropped" a checkable property instead of a promise in
        a report.
        """
        live = model.tables.live_columns("analysis_runs")
        declared = set(TABLE_BY_NAME["analysis_runs"].column_names)
        assert live == declared
        assert set(REQUIRED_ANALYSIS_RUN_COLUMNS) <= live
        columns = sample_analysis_run_columns()
        embedding = sample_embedding_columns()
        assert set(columns) | set(embedding) == set(REQUIRED_ANALYSIS_RUN_COLUMNS)
        for name in EXTRA_ANALYSIS_RUN_COLUMNS:
            assert name in live

    # -- session migration --------------------------------------------------

    def test_the_session_migration_is_lossless_and_keeps_the_file(self, model, runtime_dirs):
        """无损迁移, and 原文件保留, in one test because they are one guarantee.

        The file's bytes and its modification time are captured before the
        migration and compared after, and the row's ``payload`` is compared to the
        dict the file parsed to — field for field, including the ones no column
        models. A migration that dropped ``document_content`` or the analysis
        dictionary would still "work" as far as the messages and the queue go,
        which is exactly why the verbatim payload is what is asserted.
        """
        from paperwhisperer.sessions import store as session_store

        session_store.install_table_store(model.tables)
        try:
            session_id = "session-lossless"
            payload = legacy_session_payload(session_id)
            path = write_session_file(runtime_dirs["context"], session_id, payload)
            before_bytes = read_bytes(path)
            before_mtime = os.stat(path).st_mtime_ns

            first = session_store.migrate_session_file(session_id, table_store=model.tables)

            assert first.migrated is True
            assert first.reason == "migrated"
            assert first.source_preserved is True
            assert os.path.isfile(path)
            assert read_bytes(path) == before_bytes
            assert os.stat(path).st_mtime_ns == before_mtime

            row = model.tables.get_conversation(first.conversation_id)
            assert row is not None
            assert row["source_migrated"] is True
            assert row["migrated_at"]
            assert row["payload"] == payload
            assert row["token_hash"] == payload["session_auth"]["token_hash"]
            assert row["title"] == payload["source_filename"]
            assert row["source_sha256"] == hashlib.sha256(before_bytes).hexdigest()

            # The projections are allowed to be lossy; they are not allowed to
            # be wrong about what they do carry.
            messages = model.tables.list_messages(first.conversation_id)
            assert [row_["content"] for row_ in messages] == [
                "这篇论文的核心贡献是什么？",
                "提出一种纯注意力架构。",
                "用了什么数据集？",
                "文中未说明。",
                "摘要：一篇关于注意力的论文。",
            ]
            assert [row_["role"] for row_ in messages] == [
                "user",
                "assistant",
                "user",
                "assistant",
                "assistant",
            ]
            assert messages[1]["answer_mode"] == "direct_quote"
            queue = model.tables.list_queue_items(first.conversation_id)
            assert len(queue) == 1
            assert queue[0]["doi"] == "10.5555/3295222.3295349"
            assert queue[0]["payload"]["authors"] == ["Vaswani", "Shazeer"]
        finally:
            session_store.reset_table_store()

    def test_the_migration_happens_on_first_read_and_only_once(self, model, runtime_dirs):
        """旧 session 首次读取自动迁移, and 一次性.

        The trigger is the read: no script, no flag, no deployment step. And the
        "once" is proved by the row itself — ``migrated_at`` is written once and
        never again, so a second read that found the same file content produced
        a byte-identical row. Asserting the *row* rather than a call count is the
        point: a call count would pass for a migration that wrote the same values
        twice.
        """
        from paperwhisperer.sessions import store as session_store

        session_store.install_table_store(model.tables)
        try:
            session_id = "session-autoload"
            write_session_file(runtime_dirs["context"], session_id, legacy_session_payload(session_id))

            # Nothing has been migrated yet: the row does not exist before a read.
            assert model.tables.get_conversation(session_id) is None

            loaded = session_store.load_session_payload(session_id)
            assert loaded is not None
            row_after_first = model.tables.get_conversation(session_id)
            assert row_after_first is not None
            assert row_after_first["source_migrated"] is True
            messages_after_first = len(model.tables.list_messages(session_id))

            loaded_again = session_store.load_session_payload(session_id)
            assert loaded_again is not None
            row_after_second = model.tables.get_conversation(session_id)

            assert row_after_second == row_after_first
            assert len(model.tables.list_messages(session_id)) == messages_after_first
            # And the explicit path agrees, because it is the same code.
            again = session_store.migrate_session_file(session_id, table_store=model.tables)
            assert again.migrated is False
            assert again.reason == "already_migrated"
            assert model.tables.get_conversation(session_id) == row_after_first
        finally:
            session_store.reset_table_store()

    def test_a_changed_session_file_is_reprojected_not_duplicated(self, model, runtime_dirs):
        """The file is authoritative, so a later save refreshes the one row.

        A migration that only ever inserted would leave the mirror stale from the
        first question onwards, and a stale mirror is worse than no mirror: it
        answers confidently with the state before the user typed. The assertion
        is that there is still exactly one conversation, its version advanced,
        and the new question is in the transcript.
        """
        from paperwhisperer.sessions import store as session_store

        session_store.install_table_store(model.tables)
        try:
            session_id = "session-refresh"
            payload = legacy_session_payload(session_id)
            write_session_file(runtime_dirs["context"], session_id, payload)
            first = session_store.migrate_session_file(session_id, table_store=model.tables)
            assert first.migrated is True

            # The app appends a turn and saves — no migration call anywhere.
            updated = dict(payload)
            updated["qa_history"] = list(payload["qa_history"]) + [
                {
                    "question": "第三个问题？",
                    "answer": "第三个回答。",
                    "answer_mode": "paraphrase",
                    "timestamp": "2026-09-28T10:06:00",
                }
            ]
            write_session_file(runtime_dirs["context"], session_id, updated)

            second = session_store.migrate_session_file(session_id, table_store=model.tables)
            assert second.migrated is True
            assert second.reason == "refreshed"
            assert len(
                [
                    row
                    for row in model.tables.list_conversations(limit=50, offset=0)
                    if row["conversation_id"] == session_id
                ]
            ) == 1
            contents = [row["content"] for row in model.tables.list_messages(session_id)]
            assert "第三个问题？" in contents
            assert "第三个回答。" in contents
            # Numbered in reading order, not appended after the analysis message.
            assert [row["sequence"] for row in model.tables.list_messages(session_id)] == [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
            ]
        finally:
            session_store.reset_table_store()

    def test_a_malformed_session_still_migrates_what_it_can(self, model, runtime_dirs):
        """A hand-edited file must not lose the conversation row.

        The files are written by this app but may be edited, truncated, or left
        by another build. A migration that raised on an unexpected shape would
        turn a cosmetic problem with one field into a session that is in no
        database at all, so the row is written with whatever is readable and the
        projections come out empty rather than absent.
        """
        from paperwhisperer.sessions import store as session_store

        session_store.install_table_store(model.tables)
        try:
            session_id = "session-odd"
            payload = {
                "session_id": session_id,
                "source_filename": "奇怪的文件.pdf",
                "qa_history": ["不是字典", {"question": "", "answer": ""}],
                "paper_search": "不是字典",
                "session_auth": "不是字典",
            }
            path = os.path.join(runtime_dirs["context"], f"{session_id}.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)

            result = session_store.migrate_session_file(session_id, table_store=model.tables)
            assert result.migrated is True
            assert result.message_count == 0
            assert result.queue_item_count == 0
            row = model.tables.get_conversation(session_id)
            assert row is not None
            assert row["payload"] == payload
            assert row["token_hash"] == ""
            assert os.path.isfile(path)
        finally:
            session_store.reset_table_store()

    def test_a_session_read_survives_a_broken_mirror(self, model, runtime_dirs, monkeypatch):
        """The mirror is an optimisation; the read is the product.

        A mirror that can fail a request is a cache in the wrong place: the file
        is right there. Asserted by making the store raise on every call and
        checking that the session still loads with its content intact.
        """
        from paperwhisperer.sessions import store as session_store

        class _Broken:
            def __getattr__(self, name: str) -> Any:
                raise RuntimeError("镜像不可用")

        session_store.install_table_store(_Broken())
        try:
            session_id = "session-broken"
            write_session_file(runtime_dirs["context"], session_id, legacy_session_payload(session_id))
            loaded = session_store.load_session_payload(session_id)
            assert loaded is not None
            assert loaded["source_filename"] == "迁移测试.pdf"
            assert loaded["qa_history"][0]["question"] == "这篇论文的核心贡献是什么？"
        finally:
            session_store.reset_table_store()

    def test_migrating_every_session_is_idempotent_as_a_whole(self, model, runtime_dirs):
        """The bulk path, run twice, for the operator who is unsure.

        This is the acceptance criterion read literally: somebody who cannot tell
        whether the migration ran runs it again, and the second run must not
        write. There is no ``force`` flag, precisely because a flag that means
        "write again anyway" is how the second run becomes a second write.
        """
        from paperwhisperer.sessions import store as session_store

        session_store.install_table_store(model.tables)
        try:
            for index in range(3):
                session_id = f"session-bulk-{index}"
                write_session_file(
                    runtime_dirs["context"], session_id, legacy_session_payload(session_id)
                )
            first = session_store.migrate_all_sessions(table_store=model.tables)
            assert sum(1 for result in first if result.migrated) == 3
            rows = {
                row["conversation_id"]: row
                for row in model.tables.list_conversations(limit=50, offset=0)
            }
            assert len(rows) == 3

            second = session_store.migrate_all_sessions(table_store=model.tables)
            assert [result.migrated for result in second] == [False, False, False]
            assert {result.reason for result in second} == {"already_migrated"}
            after = {
                row["conversation_id"]: row
                for row in model.tables.list_conversations(limit=50, offset=0)
            }
            assert after == rows
        finally:
            session_store.reset_table_store()

    # -- cross-table -------------------------------------------------------

    def test_object_keys_are_reported_and_not_deleted(self, model):
        """The join between the tables and the store, deliberately one-directional.

        ``delete_document`` says which objects the caller should remove and leaves
        the bytes alone. Deleting from here would make a reachability decision
        from one of the two places that know the answer, and the version history
        is exactly the evidence that the original was never rewritten.
        """
        upload = model.objects.put(LAYER_ORIGINAL, UPLOAD_BYTES, suffix=".pdf")
        canonical = model.objects.put(LAYER_CANONICAL, b"{}", suffix=".json")
        for number in (1, 2):
            model.tables.add_document_version(
                {
                    "document_id": "doc-report",
                    "original_key": upload.key,
                    "canonical_key": canonical.key,
                    "version_no": number,
                }
            )
        keys = model.tables.object_keys_for_document("doc-report")
        # The original is the same key on both versions, so it is reported once:
        # a caller deleting the list must not be told to delete an upload twice.
        assert [(item["layer"], item["key"]) for item in keys] == [
            (LAYER_ORIGINAL, upload.key),
            (LAYER_CANONICAL, canonical.key),
        ]

        report = model.tables.delete_document("doc-report")
        assert report["document_versions"] == 2
        assert report["object_keys"]
        assert model.tables.list_document_versions("doc-report") == []
        # Still there: the caller deletes, with the layer rules the store enforces.
        assert model.objects.exists(LAYER_ORIGINAL, upload.key) is True
        assert model.objects.delete(LAYER_ORIGINAL, upload.key, force=True) is True

    def test_describe_and_close_are_harmless(self, model):
        described = model.tables.describe()
        assert described["tables"].keys() == {table.name for table in TABLES}
        assert set(tables_module.EXTERNAL_TABLES) == {"documents", "chunks", "paper_facts", "jobs"}
        model.tables.close()
        model.tables.close()
        model.objects.close()
        model.objects.close()


class TestAssetModelPersonal(AssetModelContract):
    tier = TIER_PERSONAL


@requires_postgres
class TestAssetModelHosted(AssetModelContract):
    tier = TIER_HOSTED


# --------------------------------------------------------------------------
# Tier-independent: the object store's own rules
# --------------------------------------------------------------------------


class TestLayerTaxonomy:
    """The four layers as data, asserted once.

    The isolation between layers is *enforced* by the store, and it is tested in
    the contract. What is data — and what would rot without a test — is the
    declaration of what each layer is for and which layer it may be computed
    from. A layer added to the enum without a spec would pass every other test
    and leave the next reader with a name and no meaning.
    """

    def test_there_are_exactly_four_layers_in_flow_order(self):
        assert LAYERS == (LAYER_ORIGINAL, LAYER_PARSED, LAYER_CANONICAL, LAYER_DERIVED)
        assert set(LAYER_SPECS) == set(LAYERS)

    def test_every_layer_documents_itself(self):
        for layer, spec in LAYER_SPECS.items():
            assert spec.name == layer
            assert spec.title, f"{layer} 层缺少中文名称"
            assert spec.purpose, f"{layer} 层缺少用途说明"
            assert spec.describe()["name"] == layer

    def test_the_derivation_graph_is_acyclic_and_rooted_at_the_original(self):
        """Every declared edge exists, and no layer can reach itself.

        The point of the check is that the graph is a graph. A cycle in it would
        mean "derived from itself", which is either a derived artifact keeping its
        own inputs alive forever or a loader's recursion, and either way a bug
        that would only show up as a storage bill. Walking it is six lines; a
        hand-checked diagram is six lines that stop being true.
        """
        for layer, spec in LAYER_SPECS.items():
            for parent in spec.derived_from:
                assert parent in LAYER_SPECS, f"{layer} 声明的来源层 {parent} 不存在"
        for start in LAYERS:
            seen: set[str] = set()
            current = start
            while LAYER_SPECS[current].derived_from:
                assert current not in seen, f"{start} 的来源链成环：{sorted(seen)}"
                seen.add(current)
                current = LAYER_SPECS[current].derived_from[0]
            assert current == LAYER_ORIGINAL, f"{start} 的来源链没有落在 original 层"

    def test_the_original_is_the_only_layer_that_cannot_be_replaced(self):
        # It is the one artifact in the store that cannot be produced again: every
        # other layer is a computation over it. That single fact is why put
        # refuses to overwrite it and delete needs `force`.
        assert LAYER_SPECS[LAYER_ORIGINAL].derived_from == ()
        assert LAYER_SPECS[LAYER_ORIGINAL].replaceable is False
        assert all(LAYER_SPECS[layer].replaceable for layer in LAYERS if layer != LAYER_ORIGINAL)

    def test_an_unknown_layer_is_refused_everywhere(self):
        store = FilesystemObjectStore("context/never-created-by-this-test")
        with pytest.raises(object_store_module.UnknownLayer):
            store.layer_dir("scratch")
        with pytest.raises(object_store_module.UnknownLayer):
            ObjectRef(layer="scratch", key="a.json")
        with pytest.raises(object_store_module.UnknownLayer):
            store.list_objects("scratch")


class TestKeyValidation:
    """Keys are the one thing a caller supplies freely, so they are checked hard."""

    @pytest.mark.parametrize(
        "key",
        ["", "   ", "../escape", "a/../../b", "/absolute", "a//b", "./a", "a/./b", "a\\b"],
    )
    def test_dangerous_keys_are_refused(self, key):
        with pytest.raises(UnsafeKey):
            validate_object_key(key)

    @pytest.mark.parametrize("key", ["a.json", "导出/摘要.json", "ab/cd/ef.json", "a-b_c.1.json"])
    def test_ordinary_keys_survive(self, key):
        assert validate_object_key(key) == key

    def test_a_control_character_is_refused(self):
        # A NUL in a name truncates it in the C APIs both tiers end up calling,
        # so "a\x00b" and "a" would be one object.
        with pytest.raises(UnsafeKey):
            validate_object_key("a\x00b")
        with pytest.raises(UnsafeKey):
            validate_object_key("a\nb")

    def test_a_trailing_slash_is_the_same_key(self):
        # Normalising this one away is safe: it cannot change which object is
        # named, it only removes a spelling difference.
        assert validate_object_key("a/b/") == "a/b"

    def test_digest_keys_round_trip(self):
        digest = hashlib.sha256(b"content").hexdigest()
        key = digest_key(digest, ".json")
        assert digest_from_key(key) == digest
        # A caller-named key is not content-addressed and must not claim a digest.
        assert digest_from_key("report.json") == ""
        assert digest_key(digest, ".json") != digest_key(digest, "")
        with pytest.raises(ValueError):
            digest_key("not-a-digest")


class TestObjectStoreFactory:
    """One decision about the backend, made in one place."""

    def test_the_default_is_the_local_filesystem(self, tmp_path):
        store = build_object_store(_ObjectSettings(OBJECT_STORE_ROOT=str(tmp_path / "o")))
        try:
            assert store.capabilities().backend == "filesystem"
            assert store.capabilities().content_addressed is True
            # The Hosted tier's fix is P4's S3, and the fact that these objects
            # are machine-local has to be stated before a user finds it out.
            assert store.capabilities().local is True
            assert store.capabilities().degradations
            assert "S3" in " ".join(store.capabilities().degradations)
        finally:
            store.close()

    def test_the_default_root_follows_the_context_folder(self, tmp_path, monkeypatch):
        # Same reasoning as build_default_parse_cache: a test that redirects the
        # runtime folders must redirect the object store with them, or it writes
        # into the developer's real data directory.
        assert object_store_module.default_object_store_root(_ObjectSettings()) == (
            "context/objects"
        )
        monkeypatch.setattr(core_config, "CONTEXT_FOLDER", str(tmp_path / "ctx"))
        assert object_store_module.default_object_store_root(core_config) == os.path.join(
            str(tmp_path / "ctx"), "objects"
        )

    @pytest.mark.parametrize("raw", ["s3", "minio", "http", "gcs"])
    def test_an_unimplemented_backend_is_refused_rather_than_falling_back(self, raw):
        # The failure this prevents is a Hosted deployment that looks healthy
        # while writing every user's bytes to a container's local disk.
        with pytest.raises(ValueError) as excinfo:
            build_object_store(_ObjectSettings(OBJECT_STORE_BACKEND=raw))
        assert "P4" in str(excinfo.value)

    @pytest.mark.parametrize("raw", ["filesystem", "file", "local", ""])
    def test_the_filesystem_is_reachable_under_its_spellings(self, tmp_path, raw):
        store = build_object_store(
            _ObjectSettings(OBJECT_STORE_BACKEND=raw, OBJECT_STORE_ROOT=str(tmp_path / "objects"))
        )
        store.close()

    def test_the_configuration_contract_is_declared(self):
        # Read with getattr and a default, so the module works on a checkout
        # where core/config.py has never heard of these keys. The keys table is
        # the contract with whoever adds them.
        assert set(object_store_module.OBJECT_STORE_CONFIG_KEYS) == {
            "OBJECT_STORE_BACKEND",
            "OBJECT_STORE_FSYNC",
        }
        assert set(tables_module.TABLE_STORE_CONFIG_KEYS) >= {"STORAGE_BACKEND"}


class TestObjectStoreContractShape:
    """The protocol is a contract only if the implementation has all of it.

    ``runtime_checkable`` answers "do these attributes exist", which catches a
    forgotten method and nothing else. This checks the *shape* of every member
    the protocol declares, so a method renamed on one side fails here instead of
    at the call site on one tier.
    """

    def test_every_protocol_member_is_implemented(self):
        store = FilesystemObjectStore("context/never-created-by-this-test")
        missing = [
            name
            for name in dir(ObjectStore)
            if not name.startswith("_") and not hasattr(store, name)
        ]
        assert not missing, missing
        assert isinstance(store, ObjectStore)

    def test_the_protocol_names_no_storage_engine(self):
        # The property that lets one suite run against both tiers: a method
        # called ``put_local`` or documented in terms of a path is a method the
        # S3 implementation of P4 cannot honour.
        engine_words = ("sqlite", "postgres", "pgvector", "fts5", "filesystem", "local_path")
        offenders = [
            f"{name} 提到了 {word!r}"
            for name in dir(ObjectStore)
            if not name.startswith("_")
            for word in engine_words
            if word in name.lower()
        ]
        assert not offenders, offenders


# --------------------------------------------------------------------------
# Tier-independent: the table layer's shape
# --------------------------------------------------------------------------


class TestTableSchemaIsDeclaredOnce:
    """Why the schema is data and the DDL is generated, asserted.

    A test that compared the DDL against a second hand-written copy of the
    columns would be testing that two transcriptions agree. These compare the
    *declaration* against what the producers require and against what the module
    says it owns, which are two things that can genuinely diverge.
    """

    def test_no_owned_table_is_owned_twice(self):
        owned = {table.name for table in TABLES}
        assert not owned & set(EXTERNAL_TABLES), "本模块不得重建他人拥有的表"
        assert owned == {
            "document_versions",
            "conversations",
            "messages",
            "reading_queue",
            "analysis_runs",
        }

    def test_every_table_has_a_key_a_title_and_defaults(self):
        for table in TABLES:
            assert table.primary_key in table.column_names
            assert table.title and table.purpose
            for column in table.columns:
                if column.name == table.primary_key:
                    continue
                # Every non-key column has a default, because that is what makes
                # `ALTER TABLE ADD COLUMN` possible on SQLite and therefore what
                # makes the migration work on a database an older build created.
                assert column.default is not None or column.nullable, (
                    f"{table.name}.{column.name} 既没有默认值也不可空，"
                    "增量迁移会被 SQLite 拒绝"
                )

    def test_every_table_carries_the_optimistic_lock(self):
        for table in TABLES:
            assert "version" in table.column_names, f"{table.name} 缺少 version 列"
            assert "updated_at" in table.column_names, f"{table.name} 缺少 updated_at 列"
            column = table.column("version")
            assert column.kind == tables_module.Kind.INT
            assert column.default == 1

    def test_the_analysis_run_columns_match_the_producers(self):
        # The whole of requirement 4, checked without a database: the two
        # producer methods and this module's required set are the same 28 names.
        columns = sample_analysis_run_columns()
        embedding = sample_embedding_columns()
        produced = set(columns) | set(embedding)
        assert produced == set(REQUIRED_ANALYSIS_RUN_COLUMNS)
        live = set(TABLE_BY_NAME["analysis_runs"].column_names)
        assert produced <= live
        # Nothing is dropped, and nothing is invented: the extras are exactly the
        # bookkeeping columns the report names.
        assert live - produced == set(EXTRA_ANALYSIS_RUN_COLUMNS)

    def test_the_two_costs_are_nullable(self):
        # 0.0 is "this run cost nothing because everything was cached" and None
        # is "we cannot tell". Collapsing them is how a cost dashboard starts
        # lying, so both columns are nullable and neither has a default.
        for name in ("llm_cost", "embedding_cost"):
            column = TABLE_BY_NAME["analysis_runs"].column(name)
            assert column.nullable is True
            assert column.kind == tables_module.Kind.REAL


class TestOptimisticLockIsRequired:
    """The signature claim, checked on the protocol and both implementations.

    "Callers must state the version they read" is only true if the argument has
    no default. A test that a conflict is *raised* would still pass if
    ``expected_version=None`` meant "skip the check", so the shape is asserted
    too — this is the one property that cannot be verified by exercising the
    happy path.
    """

    @pytest.mark.parametrize(
        "method_name",
        [
            "update_conversation",
            "update_message",
            "update_queue_item",
            "update_document_version",
        ],
    )
    def test_expected_version_has_no_default(self, method_name):
        from paperwhisperer.infra.tables import PostgresTableStore, SqliteTableStore

        for owner in (TableStore, SqliteTableStore, PostgresTableStore):
            parameter = inspect.signature(getattr(owner, method_name)).parameters[
                "expected_version"
            ]
            assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, (
                f"{owner.__name__}.{method_name} 的 expected_version 必须是关键字参数"
            )
            assert parameter.default is inspect.Parameter.empty, (
                f"{owner.__name__}.{method_name} 的 expected_version 不能有默认值，"
                "否则调用方可以跳过乐观锁"
            )

    def test_the_argument_is_named_so_the_error_can_quote_it(self):
        # A caller that forgot it has to be told what to pass, and a TypeError
        # saying "missing required keyword argument" is only useful if the name
        # is the documented one.
        assert "expected_version" in str(inspect.signature(TableStore.update_conversation))

    def test_the_lock_is_one_statement_not_a_read_then_a_write(self):
        # Read the generated SQL rather than trusting the implementation: the
        # property is that the version is compared *in the same UPDATE that
        # writes the data*, so a row can never be observed with new data at an
        # old version.
        source = inspect.getsource(tables_module._TableStoreBase._update)
        assert "version = version + 1" in source
        assert "AND version = " in source


class TestTableIsolation:
    """A conversation delete cascades; a document delete reports."""

    def test_deleting_a_conversation_takes_its_messages_and_queue(self, personal_model):
        conversation = personal_model.tables.add_conversation({"session_id": "s-cascade"})
        identifier = conversation["conversation_id"]
        personal_model.tables.append_message(
            {"conversation_id": identifier, "role": "user", "content": "问题"}
        )
        personal_model.tables.replace_reading_queue(identifier, [{"title": "论文"}])
        assert personal_model.tables.delete_conversation(identifier) is True
        assert personal_model.tables.list_messages(identifier) == []
        assert personal_model.tables.list_queue_items(identifier) == []
        assert personal_model.tables.get_conversation(identifier) is None

    def test_deleting_a_document_leaves_another_document_alone(self, personal_model):
        for document_id in ("doc-keep", "doc-drop"):
            personal_model.tables.add_document_version({"document_id": document_id})
            personal_model.tables.add_conversation(
                {"session_id": f"s-{document_id}", "document_id": document_id}
            )
        report = personal_model.tables.delete_document("doc-drop")
        assert report["document_versions"] == 1
        assert report["conversations"] == 1
        assert personal_model.tables.list_document_versions("doc-drop") == []
        assert len(personal_model.tables.list_document_versions("doc-keep")) == 1
        assert personal_model.tables.get_conversation_by_session("s-doc-keep") is not None


# --------------------------------------------------------------------------
# The session payload round trip: fields another module adds must survive
# --------------------------------------------------------------------------


class TestSessionPayloadPreservesOtherModulesFields:
    """A session is where another module's work gets parked. It may not strip it.

    The failure these exist for is silent by construction. A recommendation item
    is written with a ``reasons`` list explaining why the paper was suggested, the
    write reports success, and the reason is gone by the time the session is read
    back — because the store rebuilt each paper record from a fixed key set. From
    the outside that is indistinguishable from "the system has no opinion", and it
    is invisible to any test that only checks the write path.

    So the tests go the whole way round: write a payload, read it back through the
    real read path, and assert the fields are still there. Every one of them is
    written through ``write_session_payload`` and read through
    ``load_session_payload``, because a test that normalises a dict directly would
    not exercise the round trip that loses the field.
    """

    def _round_trip(self, runtime_dirs, payload: dict) -> dict:
        from paperwhisperer.sessions import store as session_store

        session_id = str(payload.get("session_id") or "session-fields")
        payload.setdefault("session_id", session_id)
        session_store.write_session_payload(session_id, payload)
        loaded = session_store.load_session_payload(session_id)
        assert loaded is not None
        return loaded

    def test_recommendation_reasons_survive_a_write_and_a_read(self, runtime_dirs):
        """The P0: "为什么推荐" must still be there on the second read.

        ``reasons`` is a list of per-signal explanations, and T3.2 marks derived
        ones with ``confidence: "derived"``. Both the list and the confidence
        inside it are asserted, because a normaliser that kept the key and
        emptied the list would pass a test that only checked the key.
        """
        reasons = [
            {"signal": "cited_by", "detail": "被 12 篇论文引用", "confidence": "exact"},
            {"signal": "same_dataset", "detail": "与本文同用 ImageNet", "confidence": "derived"},
        ]
        payload = legacy_session_payload(
            "session-reasons",
            paper_search={
                "last_query": "注意力机制",
                "last_recommendation": {
                    "query": "注意力机制",
                    "items": [
                        {
                            "title": "Attention Is All You Need",
                            "doi": "10.5555/3295222.3295349",
                            "document_id": "doc-abc",
                            "reasons": reasons,
                            "relevance": 0.93,
                        }
                    ],
                },
            },
        )
        loaded = self._round_trip(runtime_dirs, payload)
        item = loaded["paper_search"]["last_recommendation"]["items"][0]
        assert item["reasons"] == reasons
        assert [reason["confidence"] for reason in item["reasons"]] == ["exact", "derived"]
        assert item["document_id"] == "doc-abc"
        assert item["relevance"] == pytest.approx(0.93)

    def test_a_field_nobody_anticipated_still_survives(self, runtime_dirs):
        """The general policy, not the P0's three fields.

        One of them is a key this repository has never heard of. If that survives
        too, the policy is "the store preserves what it does not recognise" rather
        than "there is a list of three fields somebody remembered to add", which
        is the only version of the policy that keeps working.
        """
        payload = legacy_session_payload(
            "session-unknown",
            paper_search={
                "reading_queue": [
                    {
                        "title": "某篇论文",
                        "url": "https://example.org/p",
                        "a_field_from_the_future": {"nested": [1, 2, 3]},
                    }
                ]
            },
        )
        loaded = self._round_trip(runtime_dirs, payload)
        item = loaded["paper_search"]["reading_queue"][0]
        assert item["a_field_from_the_future"] == {"nested": [1, 2, 3]}

    def test_the_cleaning_still_happens(self, runtime_dirs):
        """Carrying extra fields is not an excuse to stop normalising.

        A long abstract is truncated, an untitled record is dropped, and the
        per-list cap holds. Without these assertions a "fix" that simply stopped
        normalising would pass the survival tests, and the session file would grow
        without bound.
        """
        from paperwhisperer.core import config as core_config

        payload = legacy_session_payload(
            "session-cleaning",
            paper_search={
                "reading_queue": [
                    {"title": "长摘要" + "x" * 5000, "venue": "y" * 500},
                    {"title": ""},
                    {"no_title_at_all": True},
                ]
            },
        )
        loaded = self._round_trip(runtime_dirs, payload)
        queue = loaded["paper_search"]["reading_queue"]
        # Two records with no title, so one record survives: an untitled nothing is
        # not a thing the user is going to read.
        assert len(queue) == 1
        # compact_text truncates to its limit and marks the cut with an ellipsis,
        # so the bound is the limit plus that marker. Asserted rather than assumed:
        # a change in how truncation is marked is exactly the kind of detail a
        # stored session should not silently lose.
        assert queue[0]["title"].endswith("...")
        assert 300 <= len(queue[0]["title"]) <= 303
        assert queue[0]["venue"].endswith("...")
        assert 120 <= len(queue[0]["venue"]) <= 123

        # And the cap: a list longer than the limit is truncated, not refused.
        from paperwhisperer.sessions import store as session_store

        many = [
            {"title": f"论文 {index}", "url": f"https://example.org/{index}"}
            for index in range(core_config.READING_QUEUE_LIMIT + 5)
        ]
        capped = session_store._normalize_papers_preserving_extras(
            many, max_items=core_config.READING_QUEUE_LIMIT
        )
        assert len(capped) == core_config.READING_QUEUE_LIMIT

    def test_extras_stay_with_their_own_paper(self, runtime_dirs):
        """Matching by identity, not by position.

        Two papers, only the second with reasons. A normaliser that carried extras
        by list position would give the *first* paper the second paper's
        "why recommended" — a wrong answer that looks entirely right, and the
        worst kind of bug to have in a field whose whole purpose is to be
        trustworthy.
        """
        payload = legacy_session_payload(
            "session-attribution",
            paper_search={
                "reading_queue": [
                    {"title": "第一篇", "url": "https://example.org/a"},
                    {
                        "title": "第二篇",
                        "url": "https://example.org/b",
                        "reasons": [{"signal": "related", "detail": "相关工作"}],
                    },
                ]
            },
        )
        loaded = self._round_trip(runtime_dirs, payload)
        first, second = loaded["paper_search"]["reading_queue"]
        assert "reasons" not in first
        assert second["reasons"] == [{"signal": "related", "detail": "相关工作"}]

    def test_the_document_id_and_fact_layer_survive(self, runtime_dirs):
        """T1.8's output parked in the session, which is what P3 needs to read.

        ``document_id`` is what makes the analysed paper identifiable, and
        ``paper_facts`` is the extracted fact layer. Both reach the API response
        and the export today; the test is that they also survive a store round
        trip, so a reloaded session can still say which paper it is about.
        """
        facts = {
            "research_question": {"value": "能否抛弃循环与卷积？"},
            "datasets": [{"value": "WMT 2014 En-De"}],
        }
        payload = legacy_session_payload(
            "session-facts", document_id="doc-identity-1", paper_facts=facts
        )
        loaded = self._round_trip(runtime_dirs, payload)
        assert loaded["document_id"] == "doc-identity-1"
        assert loaded["paper_facts"] == facts

    def test_a_session_without_them_is_not_given_empty_ones(self, runtime_dirs):
        # The difference that matters: "this session was never told which document
        # it is about" and "this session says the document is ''" are different
        # facts, and inventing the second one would make a missing wire-up look
        # like a working one.
        loaded = self._round_trip(runtime_dirs, legacy_session_payload("session-nodoc"))
        assert "document_id" not in loaded
        assert "paper_facts" not in loaded

    def test_build_session_payload_can_carry_them(self):
        from paperwhisperer.sessions import store as session_store

        payload = session_store.build_session_payload(
            "session-built",
            "a.pdf",
            "正文",
            {"summary": "摘要"},
            "token-x",
            document_id="doc-built",
            paper_facts={"limitations": [{"value": "计算开销大"}]},
        )
        assert payload["document_id"] == "doc-built"
        assert payload["paper_facts"] == {"limitations": [{"value": "计算开销大"}]}
        # And the old five-argument call still produces a payload with neither,
        # rather than raising: a new optional argument must not break its callers.
        without = session_store.build_session_payload("s", "a.pdf", "正文", {}, "t")
        assert "document_id" not in without
        assert "paper_facts" not in without

    def test_the_migration_lifts_the_document_id_into_the_row(self, personal_model, runtime_dirs):
        # The row is the durable copy, and a conversation that cannot be joined to
        # the document it is about is a row a join cannot use.
        from paperwhisperer.sessions import store as session_store

        session_store.install_table_store(personal_model.tables)
        try:
            session_id = "session-lifted"
            payload = legacy_session_payload(
                session_id, document_id="doc-lifted", paper_facts={"limitations": []}
            )
            write_session_file(runtime_dirs["context"], session_id, payload)
            result = session_store.migrate_session_file(session_id, table_store=personal_model.tables)
            row = personal_model.tables.get_conversation(result.conversation_id)
            assert row["document_id"] == "doc-lifted"
            # The fact layer stays in the verbatim payload. Its real home is the
            # store tier's paper_facts table via put_facts, which T1.8 writes; a
            # second copy here would be a second source of truth for the same
            # extraction, and the two would disagree the moment one is re-run.
            assert row["payload"]["paper_facts"] == {"limitations": []}
        finally:
            session_store.reset_table_store()

    def test_the_registry_matches_what_the_search_normaliser_emits(self):
        """The cleaning registry cannot drift away from the thing it describes.

        :data:`PAPER_RECORD_CLEANED_KEYS` names the fields the search layer owns.
        If T3.1 adds an identity field to a normalised record and this registry is
        not updated, the field is carried as an "extra" on every record — harmless
        but wrong — and if a field is ever *removed* from the normaliser while
        still listed here, it is silently dropped again, which is the bug this
        whole policy exists to prevent. So the registry is compared against the
        normaliser's real output.
        """
        from paperwhisperer.search.normalize import normalize_paper_collection
        from paperwhisperer.sessions import store as session_store

        record = normalize_paper_collection(
            [{"title": "某篇论文", "doi": "10.1/x", "reasons": [{"signal": "s"}]}]
        )[0]
        assert set(record) == set(session_store.PAPER_RECORD_CLEANED_KEYS)
        # And the extra is the *only* thing that differs, which is the whole claim.
        extra = set(session_store._normalize_papers_preserving_extras(
            [{"title": "某篇论文", "doi": "10.1/x", "reasons": [{"signal": "s"}]}], max_items=5
        )[0]) - set(record)
        assert extra == {"reasons"}


# --------------------------------------------------------------------------
# The store tier must be unaffected
# --------------------------------------------------------------------------


def test_the_store_contract_checker_still_passes():
    """T2.2 adds a fourth interface; it must not perturb the first three.

    The checker exists because a Protocol is documentation until something walks
    it against the implementations. This module imports from the store tier
    heavily, so a change that quietly satisfied the checker by renaming something
    would be the kind of regression worth catching in this file.
    """
    import subprocess
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = subprocess.run(
        [sys.executable, "tools/check_store_contract.py"],
        capture_output=True,
        text=True,
        cwd=root,
    )
    assert result.returncode == 0, result.stdout + result.stderr
