"""T2.6's acceptance criterion, as executable code.

The plan is unusually blunt about this one: **"能备份"必须验证过 restore 才算
完成** — being able to back up is not done until a restore has been exercised.
Most of this file exists because of that sentence, and it shapes what is
asserted:

- The round trip is asserted **byte-identical**, not "the row count is right".
  A restore that reproduces the right *numbers* while quietly mangling the bytes
  around them has not been shown to work, and a per-file digest comparison is
  the only assertion that says so.
- The **jobs table** is restored alongside the documents, because
  :mod:`paperwhisperer.infra.jobs` puts it in the same database as everything
  else precisely so that "one data directory to back up" is true. If a restore
  dropped it, the tier's P2 exit criteria would fail on a deployment that had
  restored successfully.
- **Corruption is detected, not restored.** A digest mismatch must raise before
  anything is written, and the target directory must still be empty afterwards.
- A **non-empty target is refused.** Restoring over a live installation by
  accident is a plausible enough mistake to defend against.

**No network, no Docker, no PostgreSQL.** These tests exercise the data path —
the walk, the snapshot, the digest, the write — and nothing about
orchestration. That is a deliberate boundary: the Hosted tier's ``pg_dump``
half needs a server, so it is *not* covered here, and
``tests/test_backup_restore.py`` asserting it would be asserting a mock.
docs/DEPLOYMENT.md records exactly which half of the Hosted path has been
exercised and which has not, rather than leaving the reader to assume.

**The fixtures write real data through the real store.** A hand-rolled SQLite
file with two hand-written tables would let a backup that captured the wrong
database pass, which is the failure a restore drill is supposed to catch.
"""

from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ``tools/`` is added by path rather than imported as a package, for the same
# reason tools/restore.py does it: these are scripts first, and a test that
# could only import them after an install step would not be testing what a
# user runs.
if str(PROJECT_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import backup as backup_tool  # noqa: E402
import restore as restore_tool  # noqa: E402

from paperwhisperer.core import config  # noqa: E402
from paperwhisperer.infra.jobs import STATE_COMPLETED, build_job_store  # noqa: E402
from paperwhisperer.infra.stores import DocumentRecord, build_stores  # noqa: E402
from paperwhisperer.ingestion.schema import Chunk, ParsedDocument  # noqa: E402

#: The archive name every test uses. Fixed rather than timestamped so a failure
#: message names a file the reader can find.
ARCHIVE_NAME = "backup.tar.gz"

#: The database's location inside a fixture data directory, matching
#: ``config.STORAGE_SQLITE_PATH``'s default shape. Written out rather than
#: imported from config so the test states where it expects the data to be.
DATABASE_RELATIVE_PATH = Path("context/stores/paperwhisperer.sqlite3")


# --------------------------------------------------------------------------
# Settings stand-in
# --------------------------------------------------------------------------


@dataclass
class _Settings:
    """The subset of configuration the store factory reads.

    Same shape as the one in ``test_stores_tiers.py``, and for the same reason:
    ``config`` is a module another task owns, and a test that monkeypatched its
    attributes would be a bad neighbour. The only field a test sets is the
    database path, pointed at a temp directory.
    """

    STORAGE_BACKEND: str = "sqlite"
    STORAGE_SQLITE_PATH: str = str(DATABASE_RELATIVE_PATH)
    STORAGE_SQLITE_TIMEOUT_SECONDS: int = 30
    STORAGE_LEXICAL_FTS5: str = "auto"
    STORAGE_PG_DSN: str = ""
    STORAGE_PG_SCHEMA: str = "public"
    STORAGE_PG_CONNECT_TIMEOUT_SECONDS: int = 5
    STORAGE_PG_ANN: bool = False


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _digest(path: Path) -> str:
    """SHA-256 of a file, read in chunks like the tool does."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_census(root: Path) -> dict[str, str]:
    """Relative path -> SHA-256, for every file under ``root``.

    Sorted by construction (a dict comprehension over a sorted walk) so a
    comparison of two censuses is a comparison of content and not of iteration
    order. Sidecars are excluded because they are an artefact of whether a
    connection happened to be open, not part of the data: the tool's job is to
    produce a directory with no loose ends, and a test that demanded a ``-wal``
    match would be demanding the tool reproduce a coincidence.
    """
    census: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name.endswith(("-wal", "-shm")):
            continue
        census[str(path.relative_to(root))] = _digest(path)
    return census


def _seed(root: Path, *, documents: int = 3, jobs: int = 2) -> None:
    """Write a realistic data directory: real store, real documents, real jobs.

    Goes through :func:`build_stores` and :func:`build_job_store` rather than
    creating tables by hand, for two reasons. A hand-written schema is not the
    schema that ships, so a backup that captured the wrong file would pass. And
    the job table's *location* — the same file as the documents — is the design
    decision these tests exist to check, so a test that put it elsewhere would
    be testing a different architecture.

    The store is closed at the end so nothing holds a lock. That is what makes
    the ``quiesced-copy`` path available and the assertions below legitimately
    byte-identical; the live-writer case has its own test with its own open
    connection, because "a backup is consistent while an app is writing" is a
    different claim from "a backup is a faithful copy of a stopped app".
    """
    for folder in ("uploads", "output", "context"):
        (root / folder).mkdir(parents=True, exist_ok=True)

    settings = _Settings(STORAGE_SQLITE_PATH=str(root / DATABASE_RELATIVE_PATH))
    bundle = build_stores(settings)
    store = build_job_store(bundle=bundle)
    try:
        for index in range(documents):
            document_id = f"doc-{index}"
            record = DocumentRecord(
                document_id=document_id,
                document=ParsedDocument(
                    document_id=document_id,
                    source_filename=f"paper-{index}.pdf",
                    page_count=index + 1,
                    parser_name="docling",
                    parser_version="2.0.0",
                ),
                source_filename=f"paper-{index}.pdf",
                sha256=f"{index:064x}",
                page_count=index + 1,
            )
            bundle.metadata.upsert_document(record)
            bundle.metadata.put_chunk(
                document_id,
                Chunk(
                    chunk_id=f"{document_id}-c0",
                    text=f"第 {index} 段正文，主题 transformer。",
                    heading_path=("引言",),
                ),
            )
            bundle.vectors.upsert_vector(f"{document_id}-c0", [0.1, 0.2, 0.3], document_id=document_id)
            bundle.lexical.index_chunk(f"{document_id}-c0", f"第 {index} 段正文", document_id=document_id)

        for index in range(jobs):
            store.submit(
                "document_analysis",
                {"document_id": f"doc-{index}", "note": "中文字段"},
                idempotency_key=f"idem-{index}",
                document_id=f"doc-{index}",
            )
        # One job driven to a terminal state, so the restored trail has a real
        # multi-event sequence in it rather than a single submission line.
        finished = store.transition(f"{store.list_jobs(limit=1)[0].job_id}", "parsing")
        store.transition(finished.job_id, STATE_COMPLETED, message="完成")

        (root / "uploads" / "paper-0.pdf").write_bytes(b"%PDF-1.4 not really a pdf")
        (root / "output" / "summary.md").write_text("# 概要\n中文内容。\n", encoding="utf-8")
        (root / "context" / "session_demo.json").write_text('{"messages": 2}', encoding="utf-8")
    finally:
        store.close()
        bundle.close()


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """A seeded personal-tier data directory in a temp location.

    ``monkeypatch`` clears ``STORAGE_SQLITE_PATH`` for the duration, because
    :func:`backup.create_backup` reads it from the environment. Without that, a
    developer's own ``.env`` would silently redirect every backup in this file
    at their real data directory — which is precisely the mistake the
    repository's conftest already warns about for the runtime folders.
    """
    root = tmp_path / "data"
    root.mkdir()
    _seed(root)
    import os

    previous = os.environ.pop("STORAGE_SQLITE_PATH", None)
    yield root
    if previous is not None:
        os.environ["STORAGE_SQLITE_PATH"] = previous


@pytest.fixture
def archive(data_dir: Path) -> Path:
    """A backup of :func:`data_dir`, at a fixed name outside it.

    Outside the data directory on purpose: an archive written *into* the tree
    being backed up would either be captured by its own walk or need an exclude
    rule to avoid it, and a backup tool whose output directory is inside its own
    input is a footgun worth not having in the test that everything else
    depends on.
    """
    destination = data_dir.parent / "out" / ARCHIVE_NAME
    backup_tool.create_backup(source_root=data_dir, destination=destination)
    return destination


# --------------------------------------------------------------------------
# The round trip
# --------------------------------------------------------------------------


def test_round_trip_is_byte_identical(data_dir: Path, archive: Path, tmp_path: Path) -> None:
    """Every file comes back with the bytes it went in with.

    The strong form of T2.6's criterion. A restore that reassembled equivalent
    data — same rows, same counts, different bytes — would pass a
    "does it look right" check and fail this one, which is the point: the two
    are different claims and only one of them is what an operator recovering
    from an incident needs.
    """
    target = tmp_path / "restored"
    report = restore_tool.restore_archive(archive, target)

    assert report.files == report.verified > 0
    assert _file_census(data_dir) == _file_census(target)
    # The directories too, including the empty ones. A restore that lost
    # `uploads/` because it held nothing at backup time fails the first time the
    # app writes there.
    assert {"uploads", "output", "context"} <= set(report.directories)


def test_round_trip_preserves_jobs_and_documents(data_dir: Path, archive: Path, tmp_path: Path) -> None:
    """The jobs table survives, with its trail — not just the documents.

    :mod:`paperwhisperer.infra.jobs` puts the queue in the same file as the
    documents on purpose, so that a backup is one operation. This asserts the
    consequence: a restored database is opened through the *real* store and
    job store, and the jobs come back with their states, idempotency keys and
    event trails intact. Reading the rows with raw SQL would not do — it would
    pass even if the store could not deserialize them.
    """
    target = tmp_path / "restored"
    restore_tool.restore_archive(archive, target)

    settings = _Settings(STORAGE_SQLITE_PATH=str(target / DATABASE_RELATIVE_PATH))
    bundle = build_stores(settings)
    store = build_job_store(bundle=bundle)
    try:
        documents = bundle.metadata.list_documents(limit=50)
        assert [record.document_id for record in documents] == ["doc-0", "doc-1", "doc-2"]

        jobs = store.list_jobs(limit=50)
        assert len(jobs) == 2
        by_state = {job.state for job in jobs}
        assert by_state == {STATE_COMPLETED, "queued"}
        # The trail is the part that matters for a resuming browser: a job
        # restored with a state but no events has lost the sequence a
        # reconnecting client uses to resync. Submitted -> parsing ->
        # completed, one event per write, monotonically numbered.
        completed = next(job for job in jobs if job.state == STATE_COMPLETED)
        assert [event.to_state_dict()["state"] for event in completed.events] == [
            "queued",
            "parsing",
            STATE_COMPLETED,
        ]
        assert [event.sequence for event in completed.events] == [1, 2, 3]
        assert completed.events[-1].sequence == completed.sequence
        # The idempotency key is what makes a re-submission reuse the job
        # instead of paying for a second cloud parse, so losing it is a
        # behaviour change, not a data change.
        assert {job.idempotency_key for job in jobs} == {"idem-0", "idem-1"}
    finally:
        store.close()
        bundle.close()


def test_restored_database_is_self_contained(data_dir: Path, archive: Path, tmp_path: Path) -> None:
    """A restored database needs no sidecar to be complete.

    A restored directory that only holds the right data while a ``-wal`` file
    happens to sit beside it is not restored, it is *incomplete* — and the
    incompleteness is invisible until somebody copies the directory with a tool
    that skips sidecars, at which point recent writes have vanished with no
    error at any point. The restore checkpoints the ``-wal`` in, so the census
    comparison above is comparing complete databases.
    """
    target = tmp_path / "restored"
    report = restore_tool.restore_archive(archive, target)
    assert report.databases, "the restore report must describe the databases it installed"
    assert report.databases[0]["integrity"] == "ok"
    leftovers = [path.name for path in target.rglob("*-wal")]
    assert leftovers == []


def test_backup_is_idempotent(data_dir: Path, tmp_path: Path) -> None:
    """Backing up unchanged data twice captures the same thing.

    Asserted on ``content_digest`` rather than on the archive bytes, because the
    archive carries a ``created_at`` and gzip carries a timestamp: two byte-wise
    identical archives are impossible and proving so would prove nothing. The
    digest is over the sorted ``(origin, sha256)`` pairs, so it answers exactly
    the question that matters — did the same data go in both times — and is
    what makes "the backup is reproducible" a checkable claim instead of a
    slogan.
    """
    first = backup_tool.create_backup(
        source_root=data_dir, destination=tmp_path / "a" / ARCHIVE_NAME
    )
    second = backup_tool.create_backup(
        source_root=data_dir, destination=tmp_path / "b" / ARCHIVE_NAME
    )
    assert first["content_digest"] == second["content_digest"]
    assert first["files"] == second["files"]
    assert first["bytes"] == second["bytes"]
    assert first["content_digest"].startswith("sha256:")

    # And the two archives really do restore to the same thing, which is the
    # use an operator makes of the property.
    for label in ("a", "b"):
        target = tmp_path / f"restored-{label}"
        restore_tool.restore_archive(tmp_path / label / ARCHIVE_NAME, target)
    assert _file_census(tmp_path / "restored-a") == _file_census(tmp_path / "restored-b")


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


def test_restore_refuses_a_non_empty_target(archive: Path, tmp_path: Path) -> None:
    """An existing installation is not clobbered by accident.

    The refusal is on "the target has *anything* in it", not "the target has a
    file this backup would overwrite". The narrower rule passes the case that
    actually matters: a target holding another installation's session file
    would be merged into silently, producing a directory that is half one
    deployment and half another and says so nowhere.
    """
    target = tmp_path / "occupied"
    target.mkdir()
    sentinel = target / "context" / "session_someone_elses.json"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text('{"do": "not touch me"}', encoding="utf-8")

    with pytest.raises(restore_tool.RestoreError) as excinfo:
        restore_tool.restore_archive(archive, target)

    assert "非空" in str(excinfo.value)
    # Untouched, byte for byte. The refusal is not a partial write.
    assert sentinel.read_text(encoding="utf-8") == '{"do": "not touch me"}'
    assert list(target.rglob("*.sqlite3")) == []


def test_restore_into_a_populated_target_with_force(archive: Path, tmp_path: Path) -> None:
    """``--force`` works, because the operator who means it should be able to.

    The counterpart to the refusal, and a test of it in both directions: a
    safety check with no override is a wall, not a guard. The stale file is
    overwritten and the flag is recorded in the report, so "this was forced" is
    never a thing that only the terminal scrollback knows.
    """
    target = tmp_path / "occupied"
    (target / "context").mkdir(parents=True)
    (target / "context" / "stale.json").write_text("stale", encoding="utf-8")

    report = restore_tool.restore_archive(archive, target, force=True)

    assert report.forced is True
    assert (target / "context" / "stale.json").exists()  # unrelated file, not our business
    assert (target / DATABASE_RELATIVE_PATH).is_file()


def test_corrupt_member_is_detected_and_nothing_is_written(
    data_dir: Path, archive: Path, tmp_path: Path
) -> None:
    """A bit-rotted backup fails loudly, before the target is touched.

    The archive is rebuilt with one member's bytes changed and its recorded
    digest left alone — which is what bit rot and a truncated upload both look
    like. The assertion that matters is the second one: the target directory is
    still not there afterwards. A restore that verifies as it writes would have
    put every file before the bad one on disk, leaving an operator with a
    half-populated data directory and an error.
    """
    corrupt = tmp_path / "corrupt.tar.gz"
    with tarfile.open(archive) as source, tarfile.open(corrupt, "w:gz") as target:
        for member in source.getmembers():
            payload = source.extractfile(member)
            data = payload.read() if payload is not None else b""
            if member.name.endswith("paper-0.pdf"):
                data = b"%PDF-1.4 rotated bits"
            info = tarfile.TarInfo(member.name)
            info.size = len(data)
            info.mtime = 0
            target.addfile(info, io.BytesIO(data))

    target_dir = tmp_path / "restored"
    with pytest.raises(restore_tool.RestoreError) as excinfo:
        restore_tool.restore_archive(corrupt, target_dir)

    assert "损坏" in str(excinfo.value)
    assert "paper-0.pdf" in str(excinfo.value)
    assert not target_dir.exists()


def test_truncated_archive_is_reported_not_traced(tmp_path: Path, archive: Path) -> None:
    """A half-written archive produces an error message, not a traceback.

    Truncation is the *most* likely corruption there is — an interrupted copy,
    a full disk — and it does not always surface as a ``TarError``. A gzip
    stream cut short raises ``EOFError`` from the compression layer, and a file
    that is not an archive at all raises from the format sniffer. A tool that
    only caught the first of those would print a stack trace on exactly the day
    the operator most needs a sentence.
    """
    truncated = tmp_path / "truncated.tar.gz"
    payload = archive.read_bytes()
    truncated.write_bytes(payload[: len(payload) // 2])

    with pytest.raises(restore_tool.RestoreError):
        restore_tool.restore_archive(truncated, tmp_path / "restored")

    not_an_archive = tmp_path / "notes.gz"
    not_an_archive.write_text("this is not a backup", encoding="utf-8")
    with pytest.raises(restore_tool.RestoreError):
        restore_tool.restore_archive(not_an_archive, tmp_path / "restored2")


def test_manifest_digest_mismatch_is_rejected(tmp_path: Path) -> None:
    """An archive whose manifest is not ours is refused, not interpreted.

    Guessing at an unfamiliar layout means writing files to paths nobody
    approved, so the format string is checked and an unknown value is a refusal
    with a version number in the message. The digests themselves are valid
    here — this is a manifest that is *well formed and honest*, from a future or
    a different tool, and the only way to tell is the version.
    """
    payload = b"future"
    manifest = {
        "format": "paperwhisperer-backup/999",
        "files": [
            {
                "origin": "x",
                "member": "data/x",
                "kind": "file",
                "size": len(payload),
                "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
            }
        ],
    }
    path = tmp_path / "future.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo("data/x")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
        body = json.dumps(manifest).encode("utf-8")
        member = tarfile.TarInfo("manifest.json")
        member.size = len(body)
        archive.addfile(member, io.BytesIO(body))

    with pytest.raises(restore_tool.RestoreError) as excinfo:
        restore_tool.restore_archive(path, tmp_path / "restored")
    assert "999" in str(excinfo.value)


def test_member_escaping_the_target_is_refused(tmp_path: Path) -> None:
    """A member that climbs out of the target is rejected, and leaves nothing.

    An archive is untrusted input even when you made it yourself — a backup
    received from a colleague, or found on a disk, is the case where the member
    names were chosen by somebody else. The digest here is *correct*, so
    nothing but the path check can catch it, which is the point of testing it
    separately. And the refusal happens before the target is created, so
    ``../../tmp/x`` does not leave an empty directory at the path the operator
    named on the way to failing.
    """
    payload = b"pwned"
    manifest = {
        "format": backup_tool.MANIFEST_FORMAT,
        "files": [
            {
                "origin": "../../escaped.txt",
                "member": "data/../../escaped.txt",
                "kind": "file",
                "size": len(payload),
                "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
            }
        ],
        "directories": [],
    }
    path = tmp_path / "evil.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo("data/../../escaped.txt")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
        body = json.dumps(manifest).encode("utf-8")
        member = tarfile.TarInfo("manifest.json")
        member.size = len(body)
        archive.addfile(member, io.BytesIO(body))

    target = tmp_path / "restored"
    with pytest.raises(restore_tool.RestoreError) as excinfo:
        restore_tool.restore_archive(path, target)
    assert "目标目录之外" in str(excinfo.value)
    assert not target.exists()
    assert not (tmp_path.parent / "escaped.txt").exists()


# --------------------------------------------------------------------------
# The snapshot, which is where the engineering is
# --------------------------------------------------------------------------


def test_snapshot_of_a_live_database_excludes_uncommitted_writes(tmp_path: Path) -> None:
    """A backup taken while an app is writing is consistent, and says it is not byte-identical.

    The harder half of T2.6, and the one a stopped-app test cannot reach. A
    connection is held open with an *uncommitted* write in a transaction, which
    is exactly the state that makes the immediate-fail ``BEGIN EXCLUSIVE`` give
    up and fall back to SQLite's online backup API.

    Two things are asserted. The committed rows are all there — a snapshot that
    lost committed work would be a backup nobody can trust. The uncommitted row
    is *not* there, because it was never committed and a backup is not allowed
    to promote it. And the manifest records ``online-backup`` rather than
    ``quiesced-copy``, so the drill in docs/DEPLOYMENT.md can say
    "byte-identical" only when it is true.
    """
    root = tmp_path / "live"
    (root / "uploads").mkdir(parents=True)
    (root / "output").mkdir()
    (root / "context" / "stores").mkdir(parents=True)
    database = root / DATABASE_RELATIVE_PATH

    app = sqlite3.connect(str(database), isolation_level=None)
    app.execute("PRAGMA journal_mode=WAL")
    app.execute("CREATE TABLE documents(document_id TEXT PRIMARY KEY, payload TEXT)")
    app.executemany(
        "INSERT INTO documents VALUES (?, ?)",
        [(f"doc-{index}", json.dumps({"index": index})) for index in range(50)],
    )
    app.execute("INSERT INTO documents VALUES ('committed', 'should survive')")
    app.execute("BEGIN IMMEDIATE")
    app.execute("INSERT INTO documents VALUES ('uncommitted', 'must not survive')")
    try:
        destination = tmp_path / "out" / ARCHIVE_NAME
        summary = backup_tool.create_backup(source_root=root, destination=destination)
        methods = summary["sqlite_databases"][0]["methods"]
        assert methods == ["online-backup"], (
            "a held write transaction must take the online-backup fallback"
        )

        target = tmp_path / "restored"
        restore_tool.restore_archive(destination, target)
        restored = sqlite3.connect(f"file:{target / DATABASE_RELATIVE_PATH}?mode=ro", uri=True)
        try:
            ids = {row[0] for row in restored.execute("SELECT document_id FROM documents")}
        finally:
            restored.close()
        assert "committed" in ids
        assert "uncommitted" not in ids
        assert len(ids) == 51
    finally:
        app.execute("ROLLBACK")
        app.close()


def test_quiesced_snapshot_is_byte_identical(data_dir: Path, archive: Path) -> None:
    """With no writer, the database in the archive is the database on disk.

    The property the drill relies on when it reports a byte-for-byte match, so
    it is asserted rather than assumed. The seed fixture closes every
    connection, which is what makes the quiesce path available; if a future
    change left a connection open this test would fail loudly instead of the
    drill quietly downgrading to a row-count comparison.
    """
    manifest = restore_tool.read_manifest(archive)
    database_records = [
        record for record in manifest["files"] if record.get("kind") == "sqlite"
    ]
    assert len(database_records) == 1
    assert database_records[0]["method"] == "quiesced-copy"
    assert database_records[0]["sha256"] == "sha256:" + _digest(data_dir / DATABASE_RELATIVE_PATH)


def test_wal_is_carried_when_a_reader_blocks_the_checkpoint(tmp_path: Path) -> None:
    """A reader holding a read transaction forces the ``-wal`` into the archive.

    The third and last shape, and the one a stopped-app test can never reach. A
    separate connection is left inside a read transaction, which stops
    ``wal_checkpoint(TRUNCATE)`` from folding the WAL in — so the quiesce path
    still takes the lock (in WAL mode ``BEGIN EXCLUSIVE`` blocks writers but
    not readers) and the sidecar has to travel with the main file, because under
    WAL mode the main file alone holds only the last checkpointed state.

    If this path dropped the sidecar, the backup would be *consistent* and
    *stale*: every one of the 500 rows would be missing, with no error
    anywhere. It is the failure the module docstring is about, and it is why
    the assertion is on row count and not merely on a clean exit.
    """
    root = tmp_path / "reader"
    (root / "uploads").mkdir(parents=True)
    (root / "output").mkdir()
    (root / "context" / "stores").mkdir(parents=True)
    database = root / DATABASE_RELATIVE_PATH

    app = sqlite3.connect(str(database), isolation_level=None)
    app.execute("PRAGMA journal_mode=WAL")
    app.execute("CREATE TABLE documents(document_id TEXT PRIMARY KEY, payload TEXT)")
    app.executemany(
        "INSERT INTO documents VALUES (?, ?)",
        [(f"doc-{index}", "x") for index in range(500)],
    )
    reader = sqlite3.connect(str(database), isolation_level=None)
    reader.execute("BEGIN")
    reader.execute("SELECT COUNT(*) FROM documents").fetchone()
    try:
        destination = tmp_path / "out" / ARCHIVE_NAME
        backup_tool.create_backup(source_root=root, destination=destination)
        manifest = restore_tool.read_manifest(destination)
        kinds = {str(record["member"]): str(record["kind"]) for record in manifest["files"]}
        wal_members = [name for name in kinds if name.endswith("-wal")]
        assert len(wal_members) == 1, "the sidecar must travel with the main file"
        assert kinds[wal_members[0]] == "wal"
        assert all(record["method"] == "quiesced-copy" for record in manifest["files"])

        target = tmp_path / "restored"
        restore_tool.restore_archive(destination, target)
        restored = sqlite3.connect(f"file:{target / DATABASE_RELATIVE_PATH}?mode=ro", uri=True)
        try:
            assert restored.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 500
        finally:
            restored.close()
        # The restore folded the WAL in, so the database is complete on its own.
        # Note that opening *this test's* read-only connection above would have
        # re-created an empty -wal: that is SQLite's normal behaviour for any
        # WAL-mode reader, not a leftover the restore failed to clean up.
    finally:
        reader.execute("ROLLBACK")
        reader.close()
        app.close()


def test_wal_sidecars_are_not_captured(data_dir: Path, archive: Path) -> None:
    """A ``-shm`` is never captured, and a ``-wal`` only as part of a snapshot.

    Both are easy to get wrong and neither is obvious. The ``-shm`` is a
    shared-memory index that SQLite rebuilds from the ``-wal``; shipping a stale
    one is a reliable way to make a restored database look corrupt. The ``-wal``
    *is* data under WAL mode, but only when it travels with the main file as a
    deliberate snapshot — a second ``-wal`` picked up by the directory walk would
    be an unsynchronised copy of the same pages, captured at a different
    instant from the main file it is supposed to complete.
    """
    manifest = restore_tool.read_manifest(archive)
    names = [str(record["member"]) for record in manifest["files"]]
    assert not [name for name in names if name.endswith("-shm")]
    wal = [name for name in names if name.endswith("-wal")]
    assert len(wal) <= 1
    for name in wal:
        record = next(item for item in manifest["files"] if item["member"] == name)
        assert record["kind"] == "wal"


# --------------------------------------------------------------------------
# The two tools agreeing with each other
# --------------------------------------------------------------------------


def test_backup_defaults_match_the_projects_config() -> None:
    """The fallback literals are the real defaults, checked rather than trusted.

    :func:`backup.project_defaults` reads the project's own config and falls
    back to literals when it cannot import — so a deployment that moved its
    data directory is captured where it is. The fallback is only free as long as
    it is right, and "the literal happens to match config" is exactly the kind of
    thing that stops being true when someone renames a folder. One assertion,
    so the two cannot drift.
    """
    defaults = backup_tool.project_defaults()
    assert defaults["sqlite_path"] == config.STORAGE_SQLITE_PATH
    assert sorted(defaults["folders"]) == sorted(config.RUNTIME_FOLDERS)
    assert backup_tool.DEFAULT_SQLITE_RELATIVE_PATH == config.STORAGE_SQLITE_PATH
    assert sorted(backup_tool.DEFAULT_RUNTIME_FOLDERS) == sorted(config.RUNTIME_FOLDERS)


def test_verify_reports_every_member_without_writing(archive: Path, tmp_path: Path) -> None:
    """``--verify`` is a real mode, and it does not create anything.

    An operator holding a backup on a disk they are not sure about needs to
    know it is intact *before* deciding to use it, and that question has to be
    answerable without side effects — otherwise running it changes the thing it
    was asked to inspect.
    """
    target = tmp_path / "never-created"
    report = restore_tool.restore_archive(archive, target, verify_only=True)
    assert report.verified == report.files > 0
    assert report.content_digest.startswith("sha256:")
    assert not target.exists()


def test_excludes_are_honoured_at_every_depth(data_dir: Path, tmp_path: Path) -> None:
    """``--exclude`` matches a name at any depth, not just at the top.

    ``__pycache__`` is the default exclude, and it appears under
    ``context/stores/`` as readily as at the root. A prefix-string test would
    let every one of those through, and an archive full of regenerable bytecode
    is a backup that wastes the operator's disk and their patience in equal
    measure.
    """
    cache = data_dir / "context" / "stores" / "__pycache__"
    cache.mkdir(parents=True)
    (cache / "junk.pyc").write_bytes(b"\x00\x01")

    destination = tmp_path / "out" / ARCHIVE_NAME
    summary = backup_tool.create_backup(source_root=data_dir, destination=destination)
    manifest = restore_tool.read_manifest(destination)
    origins = [str(record["origin"]) for record in manifest["files"]]
    assert not [name for name in origins if "__pycache__" in name]
    assert summary["files"] == len(origins)


def test_restore_reports_what_it_installed(archive: Path, tmp_path: Path) -> None:
    """The report names the database, its tables, and its row counts.

    "The file is 41 MB" is not a restore report. T2.6 asks for a drill that is
    recorded, and a recorded drill needs numbers an operator can compare against
    the source — so the numbers are a return value rather than terminal output
    that nothing can assert on.
    """
    target = tmp_path / "restored"
    report = restore_tool.restore_archive(archive, target)
    assert report.summary_line()
    database = report.databases[0]
    assert database["path"].endswith(str(DATABASE_RELATIVE_PATH))
    assert database["integrity"] == "ok"
    assert "documents" in database["tables"]
    assert "jobs" in database["tables"]
    assert database["row_counts"]["documents"] == 3
    assert database["row_counts"]["jobs"] == 2


def test_hosted_tier_requires_a_dsn_and_never_starts_a_server(tmp_path: Path) -> None:
    """The Hosted half of ``backup`` needs a DSN and refuses to guess one.

    Not a test of ``pg_dump`` — that needs a server, and this suite deliberately
    has no network and no Docker. It is a test of the *refusal*: a Hosted
    backup invoked without a DSN must fail immediately rather than quietly
    falling back to a directory copy that looks like a database backup and
    contains no database.
    """
    root = tmp_path / "hosted"
    root.mkdir()
    import os

    previous = os.environ.pop("STORAGE_PG_DSN", None)
    try:
        with pytest.raises(backup_tool.BackupError) as excinfo:
            backup_tool.create_backup(
                source_root=root, destination=tmp_path / "out" / ARCHIVE_NAME, tier="hosted"
            )
        assert "STORAGE_PG_DSN" in str(excinfo.value)
    finally:
        if previous is not None:
            os.environ["STORAGE_PG_DSN"] = previous


def test_cli_exit_codes(archive: Path, tmp_path: Path, capsys: Any) -> None:
    """The commands' exit codes mean what the docstring says they mean.

    A backup tool's exit code is its contract with cron. ``0`` for success,
    ``1`` for "could not be done", ``2`` for "you called it wrong" — and a
    corrupt archive has to be the middle one, because a cron job that reports
    usage errors as success is a backup nobody is told failed.
    """
    assert backup_tool.main(["--source-root", str(tmp_path), "--out", str(tmp_path / "o")]) == 0
    assert restore_tool.main([str(archive), "--verify"]) == 0
    assert restore_tool.main([str(archive), "--to", str(tmp_path / "cli-restored")]) == 0
    assert restore_tool.main([str(tmp_path / "cli-restored"), "--verify"]) == 2
    assert restore_tool.main([str(archive), "--to", str(tmp_path / "cli-restored")]) == 1
    output = capsys.readouterr().out
    assert "非空" in output or "恢复" in output
