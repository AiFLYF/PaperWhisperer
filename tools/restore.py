#!/usr/bin/env python3
"""Restore a PaperWhisperer backup, and prove it worked before it claims to.

    python tools/restore.py --list backups/paperwhisperer-personal-*.tar.gz
    python tools/restore.py --verify backups/paperwhisperer-personal-*.tar.gz
    python tools/restore.py --to /srv/paperwhisperer backups/....tar.gz
    python tools/restore.py --to /tmp/r --force backups/....tar.gz
    python tools/restore.py --tier hosted --dsn postgresql://... --dump backup.dump

T2.6's acceptance criterion is the whole reason this file exists in the shape
it does: **"能备份"必须验证过 restore 才算完成** — a backup nobody has restored
from is a hypothesis. So this tool is not "untar into a directory"; it is the
half of the round trip that can *disagree* with :mod:`tools.backup`, and the
three ways it is built to disagree are the point.

**Nothing is written until every digest is checked.** The archive carries a
manifest of SHA-256 digests, and this reads the whole archive into a temporary
staging area, verifies every member against it, and only then copies anything
into the target. Restoring the first 900 files of a backup and then failing on
the 901st is the worst outcome available: a half-restored directory that looks
like a deployment, on a day when the operator is trying to recover one.

**A non-empty target is refused, not clobbered.** Overwriting a live
installation's data directory with a backup is a reasonable thing to want and an
unreasonable thing to do by accident, and there is no version of "the file was
newer" that settles it. ``--force`` exists because the operator who means it
should be able to, and should have to say so.

**A path that escapes the target is refused.** Every member's destination is
resolved and checked to be inside the target before it is created. An archive
is an untrusted input even when you made it yourself: a member named
``../../.ssh/authorized_keys`` is the first thing anyone would try with a
downloaded backup, and a restore tool that writes wherever it is told is the
wrong place for that to be caught.

**A restored database is checked as a database, not as a file.** ``PRAGMA
integrity_check`` runs on every restored ``.sqlite3``, the ``-wal`` is
checkpointed into the main file so the restored directory is self-contained
rather than depending on a sidecar that a later copy might drop, and the tables
the store creates are listed. "The file is 41 MB" is not a restore report;
"the documents table has 12 rows and the jobs table has 3" is.

**The Hosted tier's restore runs ``pg_restore``, and says what it did.** Not
tested in this repository's CI — there is no server — and the command says so
rather than implying otherwise. See docs/DEPLOYMENT.md for exactly which half of
the Hosted path has been exercised and which has not.

Exit codes: ``0`` success, ``1`` the restore failed or was refused, ``2`` bad
usage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tarfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

#: Imported by name rather than re-typed, so the two tools cannot disagree
#: about the manifest's format string, the member prefix, or what an error is.
#: ``tools/`` goes on the path explicitly rather than through a package-relative
#: import because these are scripts first: ``python tools/restore.py`` has to
#: work on a checkout with no installation step, and ``tools/`` has no
#: ``__init__.py``.
_TOOLS_DIR = str(Path(__file__).resolve().parent)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from backup import DATA_PREFIX, MANIFEST_FORMAT, MANIFEST_MEMBER, BackupError  # noqa: E402

#: How long a ``pg_restore`` may run before it is killed. A restore that hangs
#: is worse than one that fails, because the operator is watching a database
#: mid-restore and has no way to tell how far it got.
PG_RESTORE_TIMEOUT_SECONDS = 3600


class RestoreError(RuntimeError):
    """The restore was refused or failed.

    Separate from :class:`backup.BackupError` on purpose even though both end
    in exit code 1: the message is what an operator reads, and "the backup you
    asked for is not intact" and "the backup could not be taken" are different
    sentences about different problems.
    """


@dataclass(slots=True)
class RestoreReport:
    """What a restore actually did, in a form a caller can assert on.

    ``verified`` is the count of members whose digest matched the manifest.
    ``databases`` is the per-database result of the integrity check and the
    table census. Both are returned rather than merely printed because the drill
    in docs/DEPLOYMENT.md and the tests need the same numbers, and a report that
    only exists as terminal output cannot be checked by anything.
    """

    archive: str
    target: str
    files: int = 0
    bytes: int = 0
    verified: int = 0
    content_digest: str = ""
    directories: list[str] = field(default_factory=list)
    databases: list[dict[str, Any]] = field(default_factory=list)
    pg_restore: dict[str, Any] | None = None
    forced: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "archive": self.archive,
            "target": self.target,
            "files": self.files,
            "bytes": self.bytes,
            "verified": self.verified,
            "content_digest": self.content_digest,
            "directories": list(self.directories),
            "databases": list(self.databases),
            "pg_restore": self.pg_restore,
            "forced": self.forced,
        }

    def summary_line(self) -> str:
        """One line an operator can read without parsing anything else.

        Same contract as ``Job.summary_line`` in
        :mod:`paperwhisperer.infra.jobs`: unambiguous on its own, because it is
        the line printed when nobody is watching a progress bar.
        """
        head = (
            f"已恢复 {self.files} 个文件（{self.bytes} 字节，校验 {self.verified} 个）"
            f" → {self.target}"
        )
        if not self.databases:
            return head
        tables = "；".join(
            f"{Path(item['path']).name}: {item.get('integrity', '?')}"
            f"（{item.get('table_count', 0)} 张表 / {item.get('row_total', 0)} 行）"
            for item in self.databases
        )
        return f"{head}；数据库 {tables}"


# ---------------------------------------------------------------------------
# Reading and verifying
# ---------------------------------------------------------------------------


def read_manifest(archive: Path) -> dict[str, Any]:
    """Load and sanity-check the manifest from an archive.

    Every check here is a refusal to guess. A missing manifest, an unparseable
    one, an unknown format version, or a member list that is not a list of
    records with a digest and a member name — each of those means the archive is
    not one of ours, or is not intact, and writing files based on what can be
    read out of a structure we do not understand is how a restore turns into an
    incident.
    """
    try:
        with tarfile.open(archive, "r:*") as handle:
            try:
                member = handle.extractfile(MANIFEST_MEMBER)
            except KeyError as exc:
                raise RestoreError(
                    f"备份里没有 {MANIFEST_MEMBER}：{archive} 不是本工具生成的备份，"
                    f"或文件已损坏。"
                ) from exc
            if member is None:
                raise RestoreError(f"备份里的 {MANIFEST_MEMBER} 不可读：{archive}")
            payload = member.read()
    except RestoreError:
        raise
    # Truncated and bit-rotted archives do not all fail the same way. A tar
    # checksum failure is a TarError; a gzip stream that ends before its
    # end-of-stream marker is an EOFError out of the gzip layer; a file that is
    # not compressed at all is a plain OSError on read. Catching only TarError
    # would mean a half-written backup — the *most* likely corruption there is
    # — printed a traceback instead of "the backup is damaged", which is
    # exactly the case the tool exists to explain.
    except (tarfile.TarError, EOFError, OSError, ValueError) as exc:
        raise RestoreError(f"无法打开备份 {archive}：{exc}") from exc

    try:
        manifest = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RestoreError(f"备份清单无法解析（可能已截断或损坏）：{archive}") from exc
    if not isinstance(manifest, dict):
        raise RestoreError("备份清单不是一个对象，拒绝猜测其结构。")
    version = manifest.get("format")
    if version != MANIFEST_FORMAT:
        raise RestoreError(
            f"备份清单格式 {version!r} 不是本工具认识的 {MANIFEST_FORMAT!r}，"
            f"拒绝猜测其结构。请用与该备份相同版本的 tools/backup.py 对应的 restore。"
        )
    if not isinstance(manifest.get("files"), list):
        raise RestoreError("备份清单缺少 files 列表，拒绝继续。")
    return manifest


def verify_archive(archive: Path) -> dict[str, Any]:
    """Check every member against the manifest, without writing anything.

    The whole point of the manifest, and the reason a corrupt backup is
    *detected* rather than restored. Returns the manifest on success; raises
    :class:`RestoreError` on the first member whose digest does not match, with
    the member named — because "the backup is bad" is useless at 2am and
    "context/parse_cache/09/x.json does not match" is a starting point.
    """
    manifest = read_manifest(archive)
    checked = 0
    total_bytes = 0
    try:
        with tarfile.open(archive, "r:*") as handle:
            members = {item.name: item for item in handle.getmembers()}
            # The manifest is expected and carries no digest of its own -- a
            # file cannot record its own hash, so it is trusted as the *root*
            # of the digest list and every other member is checked against it.
            # Removing it here rather than filtering at the comparison is what
            # makes "unrecorded member" mean one thing: something the manifest
            # makes no claim about.
            members.pop(MANIFEST_MEMBER, None)
            for record in manifest["files"]:
                name = record.get("member")
                expected = record.get("sha256")
                if not isinstance(name, str) or not isinstance(expected, str):
                    raise RestoreError(f"备份清单中的文件记录不完整：{record!r}")
                member = members.pop(name, None)
                if member is None or not member.isfile():
                    raise RestoreError(f"备份中缺少清单记录的文件：{name}")
                extracted = handle.extractfile(member)
                if extracted is None:
                    raise RestoreError(f"备份中的文件不可读：{name}")
                digest = hashlib.sha256()
                size = 0
                while True:
                    block = extracted.read(1024 * 1024)
                    if not block:
                        break
                    digest.update(block)
                    size += len(block)
                actual = "sha256:" + digest.hexdigest()
                if actual != expected:
                    raise RestoreError(
                        f"备份已损坏：{name} 的校验和不匹配"
                        f"（清单 {expected}，实际 {actual}）。拒绝恢复。"
                    )
                if int(record.get("size", size)) != size:
                    raise RestoreError(
                        f"备份已损坏：{name} 的大小与清单不符"
                        f"（清单 {record.get('size')}，实际 {size}）。拒绝恢复。"
                    )
                checked += 1
                total_bytes += size
            extra = sorted(members)
            # Extra members are not fatal on their own — a tar can carry
            # directory entries — but a *file* nobody recorded is a file the
            # manifest makes no claim about, and this tool is the one that said
            # "check every digest before writing anything".
            stray_files = [name for name in extra if not name.endswith("/")]
            if stray_files:
                raise RestoreError(
                    f"备份中存在清单未记录的文件：{', '.join(stray_files[:5])}"
                    f"（共 {len(stray_files)} 个）。拒绝恢复。"
                )
    except RestoreError:
        raise
    # Same reasoning as in read_manifest: the archive passed the manifest read
    # above, so a failure here means the member stream itself is damaged. A tar
    # checksum mismatch is a TarError, a short gzip stream an EOFError, and a
    # file that stopped being a file an OSError.
    except (tarfile.TarError, EOFError, OSError, ValueError) as exc:
        raise RestoreError(f"备份 {archive} 已损坏或被截断：{exc}") from exc
    manifest = dict(manifest)
    manifest["_verified"] = checked
    manifest["_bytes"] = total_bytes
    return manifest


def list_contents(archive: Path) -> dict[str, Any]:
    """The manifest, for ``--list``. Reads nothing but the manifest."""
    manifest = read_manifest(archive)
    return {
        "format": manifest["format"],
        "tier": manifest.get("tier", ""),
        "created_at": manifest.get("created_at", ""),
        "source_root": manifest.get("source_root", ""),
        "content_digest": manifest.get("content_digest", ""),
        "files": len(manifest.get("files", [])),
        "bytes": sum(int(item.get("size", 0)) for item in manifest.get("files", [])),
        "directories": manifest.get("directories", []),
        "sqlite_databases": manifest.get("sqlite_databases", []),
        "pg_dump": manifest.get("pg_dump"),
    }


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _is_within(root: Path, candidate: Path) -> bool:
    """Whether ``candidate`` resolves to something inside ``root``.

    Both are resolved first, and the comparison is against ``root`` itself
    rather than a string prefix: ``/tmp/data`` and ``/tmp/data-old`` share a
    string prefix and are different directories, and a restore that confuses
    them writes outside the directory the operator named.
    """
    resolved_root = root.resolve()
    resolved_candidate = candidate.resolve()
    return resolved_candidate == resolved_root or resolved_root in resolved_candidate.parents


def _check_target(target: Path, *, force: bool) -> None:
    """Refuse a target that has something in it, unless ``--force``.

    The check is "anything at all", not "any file this backup would overwrite".
    A target holding a *different* deployment's session file is exactly the case
    that must not be merged into silently: the restore would produce a directory
    that is half one installation and half another, and nothing would ever say
    so.
    """
    if not target.exists():
        return
    if not target.is_dir():
        raise RestoreError(f"恢复目标已存在且不是目录：{target}")
    existing = list(target.iterdir())
    if not existing:
        return
    if not force:
        listing = ", ".join(sorted(item.name for item in existing)[:5])
        raise RestoreError(
            f"恢复目标非空：{target}（{len(existing)} 项，例如 {listing}）。"
            f"拒绝覆盖现有数据。确需覆盖请加 --force，或先选一个空目录。"
        )


def _finalize_database(path: Path) -> dict[str, Any]:
    """Make a restored database self-contained, check it, and census it.

    Three things happen here and all three are load-bearing. The checkpoint
    makes the restored directory *self-contained*: without it the database is
    only complete in the presence of a sidecar, and the next person to copy
    that directory with a tool that skips ``-wal`` files produces a database
    that is missing recent writes — with no error at any point. The integrity
    check turns "the bytes matched the manifest" into "and SQLite agrees these
    bytes are a database", which are different claims: a manifest proves the
    archive is intact, not that what is inside it is usable.

    And then the sidecars are deleted, which is the step that is easy to miss.
    ``wal_checkpoint(TRUNCATE)`` empties the ``-wal``; it does not remove it,
    and SQLite is perfectly happy to leave a zero-byte file next to the
    database. A restored directory carrying an empty ``-wal`` is a directory
    that *looks* like it still owes you some transactions, and the next
    process to open it re-creates the pair anyway. Removing it is only safe
    because the checkpoint reported that every frame was folded in — which is
    checked, rather than assumed, because deleting a ``-wal`` that still holds
    data is the one operation here that would lose committed writes.

    ``-shm`` is removed for a different reason: it is a shared-memory index
    SQLite rebuilds on open, and a stale one restored from an archive is a
    reliable way to make a correct database look corrupt.

    Returns the census, so the caller gets the numbers without opening a second
    connection — see :func:`_census` for why that matters.
    """
    connection = sqlite3.connect(str(path))
    try:
        # One checkpoint, and its result is read: `wal_checkpoint(TRUNCATE)`
        # returns (busy, log_frames, checkpointed_frames), and the middle
        # number is how many frames were still only in the -wal afterwards. A
        # second call would return zeros and prove nothing, which is the trap
        # worth naming: this looks like a belt-and-braces double check and is
        # actually a check that always passes.
        row = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        pending = int(row[1]) if row else 0
        if pending > 0:
            # Frames are still only in the -wal, and the next step deletes that
            # file. Refusing is the loud response, and the only correct one:
            # deleting a -wal that still holds data is the one operation in
            # this file that would lose committed writes.
            raise RestoreError(
                f"恢复后的数据库仍有 {pending} 个未合并的 WAL 帧，拒绝删除其 -wal：{path}"
            )
        verdict = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        if verdict != "ok":
            raise RestoreError(f"恢复后的数据库未通过完整性检查：{path}（{verdict}）")
        census = _census(connection, path)
    finally:
        connection.close()
    _drop_empty_sidecars(path)
    return census


def _drop_empty_sidecars(path: Path) -> None:
    """Remove a ``-wal``/``-shm`` that is empty.

    Only when the size is zero, and only after the caller has confirmed the
    checkpoint folded every frame in. A non-empty ``-wal`` is left exactly where
    it is: it is data, and this is the one place in the file where deleting
    something could lose committed writes.

    ``-shm`` goes for a different reason. It is a shared-memory index SQLite
    rebuilds from the ``-wal`` on open, so a stale one restored from an archive
    is a reliable way to make a correct database look corrupt — and any
    connection opened on it, including the census below, recreates the pair
    anyway. Which is why this runs *after* the census rather than before.
    """
    for suffix in ("-wal", "-shm"):
        sidecar = Path(f"{path}{suffix}")
        try:
            if sidecar.is_file() and sidecar.stat().st_size == 0:
                sidecar.unlink()
        except OSError:  # pragma: no cover - best effort; the data is already safe
            pass


def _census(connection: sqlite3.Connection, path: Path) -> dict[str, Any]:
    """Tables and row counts of a restored database, for the restore report.

    Takes an open connection rather than a path, so the caller keeps one
    connection across the checkpoint, the integrity check and this. Not just
    tidiness: opening a second connection to finish the report would re-create
    the ``-shm`` the caller has just spent effort removing, leaving the restored
    directory with a sidecar the restore claimed it had cleaned up.
    """
    tables = [
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]
    counts: dict[str, int] = {}
    for table in tables:
        # Quoted because a table name is data, not SQL. The names come from
        # sqlite_master rather than from here, so an unexpected one must not
        # be a syntax error.
        quoted = '"' + table.replace('"', '""') + '"'
        try:
            row = connection.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()
            counts[table] = int(row[0])
        except sqlite3.Error:
            # A table this build cannot count (an FTS shadow table, say) is
            # not a reason to call the restore failed.
            counts[table] = -1
    return {
        "path": str(path),
        "integrity": "ok",
        "tables": tables,
        "table_count": len(tables),
        "row_counts": counts,
        "row_total": sum(value for value in counts.values() if value >= 0),
    }


def _reject_escaping_members(manifest: dict[str, Any], target: Path) -> None:
    """Refuse any member or directory that would land outside ``target``.

    An archive is untrusted input even when you made it yourself — a backup
    downloaded from a colleague, or one found on a disk, is exactly the case
    where the member names were chosen by somebody else. ``data/../../.ssh/
    authorized_keys`` is the first thing anyone would try, and a restore tool
    that writes wherever it is told is the wrong place for that to be caught.

    Checked as a whole list before anything is created, so the refusal leaves
    no directory behind at the path the operator named. Raises on the *first*
    offender and names it, because the operator needs to know which one.
    """
    names = [str(record.get("member", "")) for record in manifest.get("files", [])]
    names += [str(name) for name in manifest.get("directories", [])]
    for name in names:
        if not _is_within(target, target / _relative_target(name)):
            raise RestoreError(f"备份中的成员会写到目标目录之外，已拒绝：{name}")


def _materialize(
    manifest: dict[str, Any], archive: Path, target: Path, *, force: bool
) -> RestoreReport:
    """Verify everything, then install it.

    The order is the design. Verification completes over the whole archive
    *before* the target is touched at all, so a corrupt backup is discovered
    while the target is still untouched -- which is the difference between
    "the backup is bad" and "the backup is bad and I have just overwritten your
    database". What remains can only fail on a full disk or a permission
    problem, and those are reported with a count of what had been written.
    """
    verified = verify_archive(archive)
    # Every member's destination is resolved and range-checked *before* the
    # target is created, so a hostile member cannot leave behind even an empty
    # directory at the path the operator named. Checking per member inside the
    # write loop would catch it one file too late.
    _reject_escaping_members(verified, target)
    _check_target(target, force=force)

    target.mkdir(parents=True, exist_ok=True)
    report = RestoreReport(
        archive=str(archive),
        target=str(target),
        verified=int(verified.get("_verified", 0)),
        content_digest=str(verified.get("content_digest", "")),
        forced=force,
    )
    try:
        with tarfile.open(archive, "r:*") as handle:
            for record in verified["files"]:
                member_name = str(record["member"])
                destination = target / _relative_target(member_name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                extracted = handle.extractfile(member_name)
                if extracted is None:
                    raise RestoreError(f"备份中的文件不可读：{member_name}")
                with destination.open("wb") as out:
                    shutil.copyfileobj(extracted, out, 1024 * 1024)
                report.files += 1
                report.bytes += int(record.get("size", 0))

        for directory in verified.get("directories", []):
            # After the files, so a folder recorded but never written still
            # exists -- an empty uploads/ is a valid state for a fresh install.
            # No containment check here: _reject_escaping_members has already
            # cleared the whole list, and a check that could quietly skip an
            # entry would make a rejected restore look like a partial one.
            destination = target / _relative_target(str(directory))
            destination.mkdir(parents=True, exist_ok=True)
            report.directories.append(str(directory))

        for record in verified["files"]:
            if record.get("kind") != "sqlite":
                continue
            database = target / _relative_target(str(record["member"]))
            if not database.is_file():
                continue
            census = _finalize_database(database)
            census["origin"] = record.get("origin", "")
            census["method"] = record.get("method", "")
            report.databases.append(census)
    except (tarfile.TarError, EOFError, OSError, ValueError) as exc:
        # Only reachable on a second read of an archive that passed
        # verification moments ago, so it means the file changed underneath us
        # -- a network mount, or a backup job still writing. Either way the
        # report so far is not a successful restore and must not read as one.
        raise RestoreError(f"恢复过程中备份文件变得不可读：{exc}") from exc
    return report


def _relative_target(member: str) -> Path:
    """The path a member maps to inside the target, as a relative path.

    ``DATA_PREFIX`` is stripped so the archive's flat layout does not become a
    directory on disk. A member whose name is not under the prefix, or that
    tries to escape with ``..``, is returned as-is and then caught by
    :func:`_is_within` — this function does not decide safety, because deciding
    it here and checking it there would be two opinions about the same question.
    """
    name = member
    if name.startswith(DATA_PREFIX):
        name = name[len(DATA_PREFIX) :]
    return Path(name)


def restore_archive(
    archive: str | Path,
    target: str | Path,
    *,
    force: bool = False,
    verify_only: bool = False,
) -> RestoreReport:
    """Restore ``archive`` into ``target``. The whole personal-tier path.

    ``verify_only`` stops after the digest check, which is what ``--verify``
    does: an operator with a backup on a disk they are not sure about needs to
    know it is intact *before* they decide to use it.
    """
    archive_path = Path(archive)
    target_path = Path(target)
    if not archive_path.is_file():
        raise RestoreError(f"备份文件不存在：{archive_path}")
    if verify_only:
        verified = verify_archive(archive_path)
        return RestoreReport(
            archive=str(archive_path),
            target=str(target_path),
            files=len(verified["files"]),
            bytes=int(verified.get("_bytes", 0)),
            verified=int(verified.get("_verified", 0)),
            content_digest=str(verified.get("content_digest", "")),
        )
    return _materialize(read_manifest(archive_path), archive_path, target_path, force=force)


# ---------------------------------------------------------------------------
# The Hosted half
# ---------------------------------------------------------------------------


def run_pg_restore(dsn: str, dump_path: Path, *, clean: bool = False) -> dict[str, Any]:
    """Restore a ``pg_dump`` archive with ``pg_restore``.

    ``--clean --if-exists`` only when asked for, because the default has to be
    the safe one: restoring into a database that already has the schema fails
    loudly on the first conflicting object, which is an operator's decision to
    make rather than this tool's.

    Returns a dict describing what happened rather than raising on a non-zero
    exit, because a partial ``pg_restore`` (``--single-transaction`` off, an
    object that failed) still leaves a report worth reading. ``ok`` is the flag
    to check.
    """
    if not dump_path.is_file():
        raise RestoreError(f"pg_dump 归档不存在：{dump_path}")
    command = ["pg_restore", "--no-owner", "--no-acl", "--dbname", dsn]
    if clean:
        command[1:1] = ["--clean", "--if-exists"]
    command.append(str(dump_path))
    started = datetime.now(timezone.utc)
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            command,
            capture_output=True,
            text=True,
            timeout=PG_RESTORE_TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RestoreError(
            "找不到 pg_restore。Hosted 档的恢复需要 PostgreSQL 客户端工具"
            "（Debian/Ubuntu: apt-get install postgresql-client）。"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RestoreError(
            f"pg_restore 超过 {PG_RESTORE_TIMEOUT_SECONDS} 秒未完成，已中止。"
        ) from exc
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "command": command,
        "dump": str(dump_path),
        "stderr": completed.stderr.strip(),
        "seconds": round((datetime.now(timezone.utc) - started).total_seconds(), 3),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """The CLI surface. The flags that matter: ``--to``, ``--force``, ``--verify``."""
    parser = argparse.ArgumentParser(
        prog="tools/restore.py",
        description="从 tools/backup.py 的备份恢复 PaperWhisperer 数据（先校验，后写入）。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python tools/restore.py --list backups/pw.tar.gz\n"
            "  python tools/restore.py --verify backups/pw.tar.gz\n"
            "  python tools/restore.py --to /srv/paperwhisperer backups/pw.tar.gz\n"
            "  python tools/restore.py --tier hosted --dsn postgresql://... "
            "--dump backups/pw.dump\n"
        ),
    )
    parser.add_argument("archive", nargs="?", default="", help="备份归档（.tar.gz）")
    parser.add_argument("--to", default="", help="恢复目标目录（必须是空目录或不存在）")
    parser.add_argument("--force", action="store_true", help="允许写入非空目录（会覆盖）")
    parser.add_argument("--list", action="store_true", help="只打印清单内容")
    parser.add_argument("--verify", action="store_true", help="只校验，不写入")
    parser.add_argument(
        "--tier",
        choices=("personal", "hosted"),
        default="personal",
        help="personal=数据目录（默认）；hosted=调用 pg_restore",
    )
    parser.add_argument("--dsn", default="", help="Hosted 档的 PostgreSQL 连接串")
    parser.add_argument("--dump", default="", help="Hosted 档的 pg_dump 自定义格式归档")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="pg_restore 时先 DROP 已存在的对象（默认不删）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.tier == "hosted":
        # The same rule tools/backup.py uses to name the dump: strip the whole
        # `.tar.gz`, not just `.gz`. `Path.with_suffix` would give
        # `hosted.tar.dump`, which reads like a third file type and is one more
        # thing to get wrong when an operator is looking for the dump by hand.
        archive_name = args.archive or ""
        default_dump = (
            str(Path(archive_name).with_name(
                Path(archive_name).name.removesuffix(".tar.gz") + ".dump"
            ))
            if archive_name
            else ""
        )
        dump_path = Path(args.dump) if args.dump else Path(default_dump)
        dsn = args.dsn or os.environ.get("STORAGE_PG_DSN", "").strip()
        if not dsn:
            print("Hosted 档恢复需要 --dsn（或环境变量 STORAGE_PG_DSN）。", file=sys.stderr)
            return 2
        try:
            result = run_pg_restore(dsn, dump_path, clean=args.clean)
        except RestoreError as exc:
            print(f"恢复失败：{exc}", file=sys.stderr)
            return 1
        print(f"pg_restore 退出码 {result['returncode']}（耗时 {result['seconds']}s）")
        if result["stderr"]:
            print(f"  stderr: {result['stderr'][:2000]}")
        if not result["ok"]:
            print("  pg_restore 未成功，请检查上面的 stderr。", file=sys.stderr)
        return 0 if result["ok"] else 1

    if not args.archive:
        print("需要备份归档路径（或使用 --list / --verify）。", file=sys.stderr)
        return 2
    archive = Path(args.archive)
    if not archive.is_file():
        print(f"备份文件不存在：{archive}", file=sys.stderr)
        return 2

    try:
        if args.list:
            print(json.dumps(list_contents(archive), ensure_ascii=False, indent=2))
            return 0
        target = args.to or "."
        report = restore_archive(archive, target, force=args.force, verify_only=args.verify)
    except RestoreError as exc:
        print(f"恢复失败：{exc}", file=sys.stderr)
        return 1
    except BackupError as exc:  # pragma: no cover - backup errors cannot reach here
        print(f"恢复失败：{exc}", file=sys.stderr)
        return 1

    if args.verify:
        print(
            f"校验通过：{report.verified} 个文件，{report.bytes} 字节，"
            f"内容摘要 {report.content_digest}"
        )
        return 0
    print(report.summary_line())
    for database in report.databases:
        tables = ", ".join(database.get("tables", [])) or "（无）"
        print(f"  数据库 {database['path']}：{tables}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
