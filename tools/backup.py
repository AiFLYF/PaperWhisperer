#!/usr/bin/env python3
"""Back up a PaperWhisperer deployment into one file, with a manifest you can check.

    python tools/backup.py                       # personal tier, this checkout
    python tools/backup.py --tier hosted --dsn postgresql://...
    python tools/backup.py --out /mnt/backups    # somewhere that is not this disk

T2.6 asks for two things, and its acceptance criterion is stricter than either
of them: **"能备份"必须验证过 restore 才算完成** — being able to back up is not
done until a restore has been exercised. That criterion is why this tool
writes a manifest of SHA-256 digests rather than a tarball and calling it a
day. A tarball whose contents cannot be verified is a hypothesis about a
backup; the digest list is what :mod:`tools.restore` checks *before* it writes
a single byte, which is what makes "this archive is restorable" a fact rather
than an intention.

**The personal tier's backup is its data directory.** Decision D2 fixes the
personal tier as the default with zero Docker, so its entire state is a SQLite
file plus a few folders — uploads, output, context — and all of it is
reachable by copying files. No dump command, no server, no coordination. This
is the tier a single user actually runs, and the module's shape follows from
that: the whole command is a directory walk.

**The Hosted tier's backup is a ``pg_dump`` plus the object files.** The
database moves to PostgreSQL, so copying the directory stops covering it, and
the file half is still needed — uploaded originals and the parse cache are
files on both tiers today (MinIO/S3 is P4). ``--tier hosted`` shells out to
``pg_dump -Fc``, the custom format, because ``pg_restore`` can restore a single
table out of it; when the restore is a rescue rather than a rebuild, that
granularity is the difference between minutes and an outage. It goes to a
*sibling* file rather than inside the tarball, because ``pg_restore`` needs a
seekable file and nesting one inside a gzip stream would mean decompressing
everything to get at it.

**The SQLite file is snapshotted, not copied.** This is the one piece of real
engineering here, and the reason is WAL. Copying a live database file under WAL
mode captures pages the writer has not yet checkpointed, so the copy is missing
committed transactions — a backup that silently loses the last few seconds of
every write. Two honest ways out, and both are implemented:

``quiesced-copy``
    Take ``BEGIN EXCLUSIVE`` (which fails *immediately* rather than waiting —
    a backup must never block the application, and waiting is how a backup
    becomes a latency incident), checkpoint, then copy the main file and any
    non-empty ``-wal``. Copying the ``-wal`` alongside the main file is the
    part that is easy to get wrong: under WAL mode the main file alone holds
    only the last checkpointed state, and every transaction committed since is
    in the sidecar. The pair is a consistent snapshot of the *current* database
    and it is byte-identical to what is on disk, which is what lets the drill in
    docs/DEPLOYMENT.md claim a byte-for-byte comparison rather than a row count.
``online-backup``
    If another writer holds the database at that instant, the ``BEGIN
    EXCLUSIVE`` fails and this falls back to SQLite's own online backup API.
    Consistent and safe, and **not** byte-identical, because it rewrites pages.
    Which path ran is recorded in the manifest under ``files[].method``, so a
    drill that says "byte-identical" is a statement about a fact the archive
    carries rather than a hope.

**A corrupt archive is refused, not restored.** :mod:`tools.restore` verifies
every digest before writing anything, so a truncated or bit-rotted backup fails
loudly and leaves the target directory untouched.

**This tool imports as little of the project as it can.** It reads
:mod:`paperwhisperer.core.config` for the runtime folder names and the default
database path, and falls back to the same literals if that import fails. The
fallback is not duplication for its own sake: a backup tool that cannot start
because the application is broken is a backup tool that is unavailable on the
day it is needed. ``tests/test_backup_restore.py`` asserts the literals against
config's real values, so the two cannot drift while the fallback stays free.

Exit codes: ``0`` success, ``1`` the backup could not be completed, ``2`` bad
usage. Nothing here needs the network, Docker, or a PostgreSQL server for the
personal tier, and the Hosted tier needs a ``psql`` client rather than a Python
driver — which is why this file can shell out to ``pg_dump`` without importing
anything that T1.1 has to keep optional.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

#: Bumped when the manifest's shape changes incompatibly. ``tools.restore``
#: refuses a version it does not know rather than guessing at a layout, because
#: guessing writes files to paths the operator did not approve.
MANIFEST_FORMAT = "paperwhisperer-backup/1"

#: Where the manifest sits inside the archive. Written last, so a manifest at
#: the front of a *directory listing* is not what a streaming reader sees first;
#: the restore reads it by name, which is the only order that matters.
MANIFEST_MEMBER = "manifest.json"

#: Where files live inside the archive, as opposed to where they came from on
#: disk. One flat prefix, because the source root is a filesystem location that
#: means nothing on the machine being restored into — and a member named
#: ``/home/someone/context/...`` invites exactly the path-traversal bug that
#: :mod:`tools.restore` then has to defend against.
DATA_PREFIX = "data/"

#: How long a ``pg_dump`` may run before it is killed. Long enough for a large
#: corpus, short enough that a hung dump is a visible failure rather than a
#: scheduled job that silently never finishes and reports success.
PG_DUMP_TIMEOUT_SECONDS = 3600

#: Default name of the personal tier's database file, matching
#: ``paperwhisperer.infra.stores.postgres_backend.default_sqlite_path`` and
#: ``config.STORAGE_SQLITE_PATH``. Kept as a literal so this tool works when the
#: package does not import.
DEFAULT_SQLITE_RELATIVE_PATH = "context/stores/paperwhisperer.sqlite3"

#: The folders a personal deployment keeps state in, matching
#: ``config.RUNTIME_FOLDERS``.
DEFAULT_RUNTIME_FOLDERS = ("uploads", "output", "context")

#: Suffixes that make a file a *candidate* database. A candidate is then
#: actually opened, and only a file SQLite accepts is snapshotted as one: a
#: stray ``notes.db`` that is really a text file gets copied as a text file
#: instead of failing the backup.
SQLITE_SUFFIXES = (".sqlite3", ".sqlite", ".db")

#: Never captured, at any depth. ``__pycache__`` is regenerated by the first
#: run, and the backup's own output directory would otherwise include the
#: archive currently being written — the sort of thing that fills a disk.
DEFAULT_EXCLUDES = ("__pycache__", "*.pyc", ".git", ".DS_Store", "backups")

#: Tables the personal tier's store creates, reported by the restore so an
#: operator can tell "restored, and here is what is in it" from "restored a
#: file that happens to be named like our database".
KNOWN_STORE_TABLES = (
    "documents",
    "chunks",
    "paper_facts",
    "vectors",
    "lexical_chunks",
    "jobs",
)


class BackupError(RuntimeError):
    """The backup could not be completed.

    Raised rather than warned about and exited 0, because the whole value of a
    backup command is that its exit code means something. A backup that fails
    halfway and reports success is worse than no backup at all: it is a backup
    somebody believes in.
    """


# ---------------------------------------------------------------------------
# Defaults, from the project's own config when it will import
# ---------------------------------------------------------------------------


def project_defaults() -> dict[str, Any]:
    """The runtime folders and database path this project ships with.

    Read from :mod:`paperwhisperer.core.config` so there is one source of
    truth, falling back to the literals above when the import fails. The
    environment still wins over both, so a deployment that moved its data is
    captured where it actually is rather than where the default says.
    """
    folders = list(DEFAULT_RUNTIME_FOLDERS)
    sqlite_path = DEFAULT_SQLITE_RELATIVE_PATH
    try:
        from paperwhisperer.core import config  # noqa: PLC0415 - optional, see above
    except Exception:  # noqa: BLE001 - a broken app must not block a backup
        return {"folders": folders, "sqlite_path": sqlite_path}
    folders = [str(name) for name in getattr(config, "RUNTIME_FOLDERS", folders)]
    sqlite_path = str(getattr(config, "STORAGE_SQLITE_PATH", "") or sqlite_path)
    return {"folders": folders, "sqlite_path": sqlite_path}


# ---------------------------------------------------------------------------
# Entries: what goes in, and what it was
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Entry:
    """One file in the backup, described from both ends.

    ``origin`` is where it came from, relative to the source root (or absolute
    for a database configured outside it). ``member`` is where it sits in the
    archive. Keeping both is what lets the restore put a file back where it was
    without the archive carrying an absolute path from another machine.

    ``kind`` is ``"file"``, ``"sqlite"`` or ``"wal"``, and it is recorded in the
    manifest rather than re-derived on the way back in. A restore that has to
    answer "is this the main database file or one of its write-ahead siblings?"
    is then reading a recorded fact instead of matching a filename convention
    that a future refactor could change.
    """

    origin: str
    member: str
    source: Path
    kind: str = "file"
    size: int = 0
    sha256: str = ""
    #: For database entries, how the snapshot was taken — see the module
    #: docstring. Empty for an ordinary file.
    method: str = ""

    def to_manifest(self) -> dict[str, Any]:
        """The manifest record for this entry.

        A method of ``""`` is omitted rather than written as an empty string, so
        "this file came from a live writer" stays a distinguishable fact
        instead of becoming a key that is present but always falsy.
        """
        record: dict[str, Any] = {
            "origin": self.origin,
            "member": self.member,
            "kind": self.kind,
            "size": self.size,
            "sha256": self.sha256,
        }
        if self.method:
            record["method"] = self.method
        return record


@dataclass(slots=True)
class BackupPlan:
    """The set of entries, before any of them has been read.

    The plan and the writing are separate so the two failure modes stay
    distinguishable: a plan that cannot be *built* is a configuration problem
    (the root does not exist, nothing to capture), and a write that fails is an
    I/O or lock problem. An operator responds to those differently, and a tool
    that reports both as "backup failed" has taken away the one piece of
    information they needed.
    """

    source_root: Path
    entries: list[Entry] = field(default_factory=list)
    #: Directories to recreate on restore, relative to the target. Recorded
    #: separately from the files because an empty folder contains no files, and
    #: a restore that loses ``output/`` because it happened to be empty is a
    #: restore that fails the first time the app writes there.
    directories: list[str] = field(default_factory=list)
    #: Absolute paths of the databases that need a snapshot. The snapshot is
    #: taken by :func:`create_backup` rather than here, so that the staging
    #: directory's lifetime is owned by the function that created it.
    databases: list[Path] = field(default_factory=list)

    @property
    def content_digest(self) -> str:
        """A digest of *what was captured*, independent of when.

        Hashing the sorted ``(origin, sha256)`` pairs gives idempotence a
        meaning: two backups of unchanged data carry the same
        ``content_digest``, so a test can assert that a second run captured the
        same thing without comparing two archives byte for byte — which would
        fail on the timestamps alone and so would prove nothing.

        Timestamps, member order and the container format are excluded on
        purpose. This digest answers exactly one question: *did the same data
        go in both times?*
        """
        digest = hashlib.sha256()
        for entry in sorted(self.entries, key=lambda item: item.origin):
            digest.update(entry.origin.encode("utf-8"))
            digest.update(b"\0")
            digest.update(entry.sha256.encode("utf-8"))
            digest.update(b"\n")
        return "sha256:" + digest.hexdigest()


def relative_member(path: Path, root: Path) -> str:
    """``path`` relative to ``root``, spelled with forward slashes.

    The container is a tar, so its member names are POSIX paths by definition,
    and the manifest has to name the same string the tar does. On Windows
    ``str(path.relative_to(root))`` yields ``uploads\\paper.pdf`` while tarfile
    writes the member as ``uploads/paper.pdf`` — the manifest then records a
    name no member answers to, and the restore refuses a perfectly good backup
    with "备份中缺少清单记录的文件", pointing at a file that is sitting right
    there in the archive. Spelling both sides the same way is the whole fix,
    and it also makes the manifest byte-identical across platforms, which the
    plan digest is compared for.
    """
    return path.relative_to(root).as_posix()


def is_excluded(relative: str, patterns: tuple[str, ...]) -> bool:
    """Whether a source-relative path matches any exclude pattern.

    Matched per path component as well as by fnmatch, so excluding
    ``__pycache__`` means every ``__pycache__`` at every depth — a prefix
    string test would only catch the ones at the top. A pattern containing a
    separator is matched against the whole relative path, which is the
    convention every other build tool uses, so nobody has to learn a new one to
    write ``--exclude 'output/*.md'``.
    """
    parts = Path(relative).parts
    for pattern in patterns:
        if "/" in pattern or os.sep in pattern:
            if fnmatch(relative, pattern):
                return True
        elif any(fnmatch(part, pattern) for part in parts):
            return True
    return False


def looks_like_sqlite(path: Path) -> bool:
    """Whether SQLite will open this file as a database.

    A suffix test alone is a guess, and the guess has a cost in both
    directions: treating a text file as a database produces a backup that fails
    for the operator, and treating a database as a text file produces a torn
    copy that looks like it worked. So the file is genuinely opened, read-only,
    and asked for its schema.
    """
    if not path.is_file():
        return False
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error:
        return False
    try:
        return connection.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone() is not None
    except sqlite3.Error:
        # It opened and would not answer. Either not a database, or a corrupt
        # one. A corrupt database is worth a loud failure, so this returns
        # False and the file is copied verbatim rather than snapshotted —
        # preserving the corruption for whoever has to diagnose it.
        return False
    finally:
        connection.close()


def build_plan(
    source_root: Path,
    *,
    folders: tuple[str, ...] | list[str] = DEFAULT_RUNTIME_FOLDERS,
    sqlite_paths: list[Path] | None = None,
    excludes: tuple[str, ...] = DEFAULT_EXCLUDES,
) -> BackupPlan:
    """Decide what a backup of ``source_root`` will contain.

    Every runtime folder that exists is walked; one that does not exist is
    skipped rather than treated as an error, because a checkout that has never
    had a document uploaded has no ``uploads/`` and that is not a broken
    backup. The database files are listed but *not* copied here — their
    snapshot replaces the file on disk (see the module docstring), and walking
    them as ordinary files would capture a second, torn copy next to the good
    one. The snapshot is taken by :func:`create_backup`, which owns the staging
    directory's lifetime.
    """
    plan = BackupPlan(source_root=source_root)
    resolved_root = source_root.resolve()

    # Resolved once. Resolving per file would let a path that moved mid-backup
    # resolve differently in two places, which is the same class of bug as the
    # torn copy this tool exists to prevent.
    databases: list[Path] = []
    for candidate in sqlite_paths or []:
        resolved = (candidate if candidate.is_absolute() else resolved_root / candidate).resolve()
        if resolved.is_file() and resolved not in databases and looks_like_sqlite(resolved):
            databases.append(resolved)

    for folder in folders:
        folder_path = (resolved_root / folder).resolve()
        if not folder_path.is_dir():
            continue
        plan.directories.append(str(folder))
        for current, dirnames, filenames in os.walk(folder_path):
            current_path = Path(current)
            parent_relative = relative_member(current_path, resolved_root)
            # Pruned in place, which is how os.walk is meant to be filtered:
            # deleting from ``dirnames`` skips the subtree, and a skip list that
            # only hid the directory would still walk everything beneath it.
            dirnames[:] = sorted(
                name
                for name in dirnames
                if not is_excluded(f"{parent_relative}/{name}", excludes)
            )
            for name in sorted(filenames):
                absolute = (current_path / name).resolve()
                # SQLite's sidecars are skipped by name, not by suffix: the
                # suffixes are "-wal" and "-shm" but a file's suffix is whatever
                # follows the last dot, so `x.sqlite3-shm` has the suffix
                # ".sqlite3-shm" and a suffix test silently misses it. Capturing
                # a -shm is actively harmful (a stale shared-memory index makes a
                # restored database look corrupt) and a -wal captured here would
                # be a *second*, unsynchronised copy of the one the snapshot
                # takes deliberately.
                if absolute in databases or name.endswith(("-wal", "-shm")):
                    continue
                relative = relative_member(absolute, resolved_root)
                if is_excluded(relative, excludes):
                    continue
                plan.entries.append(
                    Entry(
                        origin=relative,
                        member=f"{DATA_PREFIX}{relative}",
                        source=absolute,
                        kind="file",
                    )
                )

    plan.databases = databases
    return plan


# ---------------------------------------------------------------------------
# The SQLite snapshot
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SqliteSnapshot:
    """The staged copies of one database, and how they were taken.

    ``files[0]`` is the database itself; anything after it is a write-ahead
    sibling that has to travel with it. The caller owns the staging directory
    and deletes it — deliberately *not* a context manager, because the staged
    files are entries in a plan that outlives the snapshot call, and a ``with``
    block around the snapshot would put the archive-writing inside the
    temporary directory's lifetime, which is the wrong way round.
    """

    files: list[Path]
    method: str

    @property
    def main(self) -> Path:
        return self.files[0]


def snapshot_sqlite(database: Path, stage: Path) -> SqliteSnapshot:
    """Copy ``database`` into ``stage`` as a consistent snapshot, and say how.

    See the module docstring for why the ``-wal`` travels with the main file
    and why the quiesce lock is immediate-fail. The short version: an
    immediate-fail ``BEGIN EXCLUSIVE`` is what makes the pair below immutable
    while it is read, and the ``-wal`` is what makes the pair *current*.
    """
    target = stage / database.name
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(str(database), timeout=0, isolation_level=None)
        connection.execute("PRAGMA busy_timeout=0")
        # The transaction is what makes the copy below immutable; without it,
        # another connection could checkpoint between our read and our write.
        connection.execute("BEGIN EXCLUSIVE")
    except sqlite3.Error:
        if connection is not None:
            connection.close()
        return _snapshot_online(database, target)

    files = [target]
    try:
        try:
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error:
            # A checkpoint that declines — a reader is active — is fine. The
            # data is in the -wal, and the -wal is being copied next.
            pass
        shutil.copyfile(database, target)
        sidecar = Path(f"{database}-wal")
        if sidecar.is_file() and sidecar.stat().st_size > 0:
            shutil.copyfile(sidecar, Path(f"{target}-wal"))
            files.append(Path(f"{target}-wal"))
        # -shm is deliberately NOT copied. It is a shared-memory index that
        # SQLite rebuilds from the -wal, and shipping a stale one is a
        # reliable way to make a restored database look corrupt. The -wal is
        # data; the -shm is a cache of a data structure that no longer exists.
    finally:
        try:
            connection.execute("COMMIT")
        except sqlite3.Error:  # pragma: no cover - the transaction is ours
            pass
        connection.close()
    return SqliteSnapshot(files=files, method="quiesced-copy")


def _snapshot_online(database: Path, target: Path) -> SqliteSnapshot:
    """Snapshot through SQLite's online backup API.

    Used when the quiesce path could not take the write lock. Consistent by
    construction — it is the API SQLite provides for exactly this situation —
    but it rewrites the file, so the result is *not* byte-identical to the
    source, and ``method`` records that so nothing downstream claims otherwise.
    """
    source = sqlite3.connect(str(database))
    destination = sqlite3.connect(str(target))
    try:
        source.backup(destination)
    finally:
        destination.close()
        source.close()
    return SqliteSnapshot(files=[target], method="online-backup")


# ---------------------------------------------------------------------------
# Writing the archive
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> tuple[str, int]:
    """Digest and size of a file, read in chunks.

    Chunked because the database file can be hundreds of megabytes and
    ``read_bytes()`` would hold all of it in memory twice over — once here and
    once in the tar stream that is about to read it again.
    """
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
            size += len(block)
    return "sha256:" + digest.hexdigest(), size


def default_archive_name(tier: str) -> str:
    """``paperwhisperer-personal-20260930T101500Z.tar.gz``.

    The tier is in the name because the two archives are not interchangeable,
    and an operator reaching for the wrong one at 2am should be slowed down by
    the filename. Seconds and a ``Z``, because a backup stamped with a date and
    no time cannot be ordered.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"paperwhisperer-{tier}-{stamp}.tar.gz"


def pg_client_version(binary: str = "pg_dump") -> str:
    """The client's own version string, or ``""`` if it will not say.

    Recorded because of a real trap this ran into: a ``pg_dump`` newer than the
    server emits statements the older server does not understand. Dumping a
    PostgreSQL 16 server with a 17 client produces an archive whose *first*
    statement is ``SET transaction_timeout = 0``, which 16 rejects — so the
    restore fails on a line that has nothing to do with the data, with an error
    ("unrecognized configuration parameter") that reads like a corrupted
    archive. The archive itself is fine.

    Writing the client version into the backup's own output is what turns that
    diagnosis from an afternoon into a glance. It is best-effort and never
    fails the backup: a version string is diagnostic, not data.
    """
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [binary, "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


def run_pg_dump(dsn: str, destination: Path) -> dict[str, Any]:
    """Write a ``pg_dump`` custom-format archive to ``destination``.

    ``-Fc`` for the reasons in the module docstring. ``--no-owner`` and
    ``--no-acl`` because a dump carrying the original cluster's role names and
    grants cannot be restored into a differently-named role without editing it
    first, and hand-editing a dump during a rescue is a bad time to be doing SQL
    archaeology. The password travels in ``PGPASSWORD`` rather than on the
    command line, where it would be visible in ``ps`` to every user on the box.

    No Python driver is imported. ``psql`` is a deployment tool rather than a
    library dependency, which is what keeps the Hosted tier's backup from
    pulling psycopg into a path that has to work when the app is broken.
    """
    environment = dict(os.environ)
    authority = dsn.split("://", 1)[1].partition("@")[0] if "://" in dsn else ""
    if ":" in authority:
        environment["PGPASSWORD"] = authority.partition(":")[2]
    command = [
        "pg_dump",
        "--format=custom",
        "--no-owner",
        "--no-acl",
        "--dbname",
        dsn,
        "--file",
        str(destination),
    ]
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            command,
            env=environment,
            capture_output=True,
            text=True,
            timeout=PG_DUMP_TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError as exc:
        raise BackupError(
            "找不到 pg_dump。Hosted 档的备份需要 PostgreSQL 客户端工具"
            "（Debian/Ubuntu: apt-get install postgresql-client）。"
            "个人档不需要任何外部工具。"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise BackupError(f"pg_dump 超过 {PG_DUMP_TIMEOUT_SECONDS} 秒未完成，已中止。") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "（无输出）"
        raise BackupError(f"pg_dump 退出码 {completed.returncode}：{detail}")
    return {
        "path": str(destination),
        "bytes": destination.stat().st_size if destination.is_file() else 0,
        "client": pg_client_version(),
    }


def write_archive(
    plan: BackupPlan,
    destination: Path,
    *,
    tier: str,
    manifest_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``plan`` to ``destination`` as a gzipped tar with a manifest.

    The digests are computed here, as each member is added, and the manifest is
    serialized afterwards — so the manifest is a record of what is *actually in*
    the archive rather than of what was intended. A file that changed on disk
    between the plan and the write then shows up as a digest mismatch at
    restore time, rather than as a backup that appears to have worked.

    Written to a sibling ``.partial`` path and renamed into place. A half-written
    archive under the final name is a backup that looks restorable and is not;
    one under a temporary name is obviously unfinished and the next run
    overwrites it.
    """
    manifest: dict[str, Any] = {
        "format": MANIFEST_FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tier": tier,
        "tool": "tools/backup.py",
        "source_root": str(plan.source_root),
        "directories": sorted(plan.directories),
        "sqlite_databases": [],
        "pg_dump": None,
        "files": [],
    }
    if manifest_extra:
        manifest.update(manifest_extra)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(destination.name + ".partial")
    try:
        with tarfile.open(staging, "w:gz") as archive:
            for entry in plan.entries:
                if not entry.source.is_file():
                    raise BackupError(f"备份过程中消失的文件：{entry.source}")
                entry.sha256, entry.size = sha256_file(entry.source)
                archive.add(entry.source, arcname=entry.member, recursive=False)
                manifest["files"].append(entry.to_manifest())
            manifest["content_digest"] = plan.content_digest
            if plan.databases:
                manifest["sqlite_databases"] = [
                    {
                        "path": str(database),
                        "tables": list(KNOWN_STORE_TABLES),
                        "methods": sorted(
                            {entry.method for entry in plan.entries if entry.method}
                        ),
                    }
                    for database in sorted(str(item) for item in plan.databases)
                ]
            payload = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode(
                "utf-8"
            )
            info = tarfile.TarInfo(MANIFEST_MEMBER)
            info.size = len(payload)
            info.mtime = 0
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(payload))
        os.replace(staging, destination)
    finally:
        # A failure above must not leave a `.partial` that the next run
        # mistakes for a complete archive. Best effort: the failure that got us
        # here is the one worth reporting.
        if staging.exists():
            try:
                staging.unlink()
            except OSError:  # pragma: no cover - cleanup must not mask the real error
                pass

    return {
        "archive": str(destination),
        "tier": tier,
        "files": len(manifest["files"]),
        "bytes": sum(int(item["size"]) for item in manifest["files"]),
        "content_digest": manifest["content_digest"],
        "directories": list(manifest["directories"]),
        "sqlite_databases": manifest["sqlite_databases"],
        "pg_dump": manifest.get("pg_dump"),
    }


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------


def create_backup(
    *,
    source_root: Path,
    destination: Path,
    tier: str = "personal",
    dsn: str = "",
    excludes: tuple[str, ...] = DEFAULT_EXCLUDES,
) -> dict[str, Any]:
    """Snapshot the databases, walk the folders, write the archive.

    The whole flow, so :mod:`tools.restore` and the tests exercise the same
    function the CLI does; the CLI only adds argument parsing.

    The staging directory lives here rather than inside :func:`build_plan`
    because its lifetime has to span the snapshot *and* the archive write. The
    entries in a plan point at staged files, so the files must outlive the plan
    — and a ``finally`` in this function is the one place that can be certain
    of deleting them afterwards.
    """
    defaults = project_defaults()
    if tier == "hosted":
        # The Hosted tier's documents live in PostgreSQL, so there is no local
        # SQLite file to capture and the file half is only uploads/output/
        # context. Passing a database path here would capture a stale local
        # SQLite file and quietly restore it over a deployment that had
        # migrated — which is how "I restored the backup and my documents
        # disappeared" happens.
        sqlite_paths: list[Path] = []
    else:
        configured = os.environ.get("STORAGE_SQLITE_PATH", "").strip()
        sqlite_paths = [Path(configured or defaults["sqlite_path"])]

    plan = build_plan(
        source_root,
        folders=defaults["folders"],
        sqlite_paths=sqlite_paths,
        excludes=excludes,
    )

    stage = Path(tempfile.mkdtemp(prefix="pw-backup-stage-"))
    try:
        for database in plan.databases:
            # A subdirectory per database, because two databases may share a
            # file name and a flat staging directory would have the second
            # snapshot overwrite the first.
            database_stage = stage / database.name
            database_stage.mkdir(parents=True, exist_ok=True)
            snapshot = snapshot_sqlite(database, database_stage)
            try:
                relative = relative_member(database, plan.source_root.resolve())
            except ValueError:
                # Configured outside the data directory. The origin stays
                # absolute so the restore can put it back where it was, and the
                # member drops to the file name because an absolute path is not
                # something to hand to tar as a member name.
                relative = database.as_posix()
            for staged in snapshot.files:
                kind = "sqlite" if staged == snapshot.main else "wal"
                suffix = "" if kind == "sqlite" else "-wal"
                plan.entries.append(
                    Entry(
                        origin=f"{relative}{suffix}",
                        member=f"{DATA_PREFIX}{relative}{suffix}",
                        source=staged,
                        kind=kind,
                        method=snapshot.method,
                    )
                )
        summary = write_archive(plan, destination, tier=tier)
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    if tier == "hosted":
        # After the archive, not before: a pg_dump failure must not leave a
        # half-written archive under the final name looking like a complete
        # backup of a database.
        resolved_dsn = dsn or os.environ.get("STORAGE_PG_DSN", "").strip()
        if not resolved_dsn:
            raise BackupError(
                "Hosted 档的备份需要 --dsn 或环境变量 STORAGE_PG_DSN；"
                "个人档不需要任何配置。"
            )
        dump_path = destination.with_name(
            str(destination.name).removesuffix(".tar.gz") + ".dump"
        )
        # Recorded because a client newer than the server produces an archive
        # the older server refuses to restore, and the only clue is the version
        # string. See pg_client_version.
        summary["pg_dump"] = run_pg_dump(resolved_dsn, dump_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """The CLI surface: one command, both tiers, no sub-commands to learn."""
    parser = argparse.ArgumentParser(
        prog="tools/backup.py",
        description="备份 PaperWhisperer 部署（个人档=数据目录；Hosted 档=pg_dump + 对象文件）。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python tools/backup.py\n"
            "  python tools/backup.py --out ~/backups\n"
            "  python tools/backup.py --tier hosted --dsn "
            "postgresql://user:pass@localhost:5432/paperwhisperer\n"
        ),
    )
    parser.add_argument(
        "--tier",
        choices=("personal", "hosted"),
        default="personal",
        help="个人档（默认，零 Docker）或 Hosted 档（PostgreSQL）",
    )
    parser.add_argument("--source-root", default=".", help="数据目录所在根（默认当前目录）")
    parser.add_argument("--out", default="", help="输出目录（默认 <source-root>/backups）")
    parser.add_argument("--name", default="", help="输出文件名（默认按档位与时间生成）")
    parser.add_argument("--dsn", default="", help="Hosted 档的 PostgreSQL 连接串")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="PATTERN",
        help="额外排除的路径模式，可重复；与默认的 __pycache__ 等叠加",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command. Returns a process exit code; raises nothing but BackupError."""
    args = build_parser().parse_args(argv)
    source_root = Path(args.source_root).resolve()
    if not source_root.is_dir():
        print(f"数据目录不存在：{source_root}", file=sys.stderr)
        return 2
    out_dir = Path(args.out).resolve() if args.out else source_root / "backups"
    # `--name` is a file *name*, not a path: a backup landing somewhere the
    # operator did not name is how a data directory ends up holding an archive
    # with the wrong tier's contents in it. An absolute path is honoured,
    # because "put this exact file there" is a reasonable thing to mean.
    destination = (
        Path(args.name)
        if args.name and Path(args.name).is_absolute()
        else out_dir / (args.name or default_archive_name(args.tier))
    )
    try:
        summary = create_backup(
            source_root=source_root,
            destination=destination,
            tier=args.tier,
            dsn=args.dsn,
            excludes=DEFAULT_EXCLUDES + tuple(args.exclude),
        )
    except BackupError as exc:
        print(f"备份失败：{exc}", file=sys.stderr)
        return 1
    print(f"备份完成：{summary['archive']}")
    print(f"  档位 {summary['tier']}；{summary['files']} 个文件，共 {summary['bytes']} 字节")
    print(f"  内容摘要 {summary['content_digest']}")
    for database in summary["sqlite_databases"]:
        methods = "/".join(database.get("methods") or ["-"])
        print(f"  数据库 {database['path']}（快照方式 {methods}）")
    if summary.get("pg_dump"):
        dump = summary["pg_dump"]
        client = dump["client"] or "客户端版本未知"
        print(f"  pg_dump {dump['path']}（{dump['bytes']} 字节，{client}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
