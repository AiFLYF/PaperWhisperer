"""Session persistence, token-based access control, and the one-way database mirror.

A session is a single JSON file under ``context/``. Writes go through
:func:`atomic_write_json` so a reader never observes a half-written file, and
access requires a bearer token whose SHA-256 hash is what gets stored — the
raw token is returned to the client exactly once, at analysis time.

**The JSON file stays the live store; the database is a mirror, and this module
is the migration.** T2.2 moves sessions into the ``conversations`` /
``messages`` / ``reading_queue`` tables, and the plan's wording is the whole
design: 一次性迁移, 兼容读取, 原文件保留. Those three clauses are not decoration
and they are the reason this is not a "load the file, write the row, delete the
file" routine:

*兼容读取* is what makes a one-way migration possible at all. A deployment that
upgrades mid-session has files written by the previous build and rows written by
this one, and the read path has to serve both. So the file remains authoritative
and every read refreshes the row from it; the database is never consulted to
answer a question the file can answer.

*原文件保留* is the only thing that makes a bad migration recoverable. If the
row is wrong, if a future build needs a field this one never modelled, or if the
migration itself has a bug, the bytes are still on disk and can be re-read and
re-projected. A migration that deletes its source is a migration whose failure is
a data loss, and it would be discovered exactly once, on the deployment where the
delete half-worked. So nothing here unlinks a session file, and
:func:`cleanup_expired_sessions` is the only thing in the project that removes
them.

*一次性* is the requirement that is easiest to get wrong, because "once" cannot
mean "the first time it ran" — that is a flag, and a flag is state that can be
lost, set twice, or copied from a test database. It is decided by *content*
instead: the row records the SHA-256 of the file's bytes it was built from, and a
migration whose source digest already matches writes nothing at all. A file that
has not changed therefore migrates once, however many times a session is read,
and a file that *has* changed is re-projected because the file is authoritative
and a stale row would be a lie. That is what makes running the migration twice a
no-op rather than a second write, and it is why a caller who is unsure whether it
ran can simply run it.

**Losslessness is structural, not best-effort.** The conversation row stores the
session payload *verbatim* — the exact dict ``json.load`` produced, before any
normalization in this module — in ``conversations.payload``, next to the
``payload_sha256`` of that dict. Everything else (messages, queue items) is a
*projection* of it, and is therefore allowed to be a lossy convenience. The
distinction matters: a projection that misses a field is a gap a future query
reports, while a verbatim payload means the original value is always recoverable
from the row even if every projection is discarded.

**Migration never breaks a read.** :func:`load_session_payload` calls the
migration after a successful parse and before normalization, inside a
``try``/``except`` that logs and continues. A store that cannot be reached, a
schema that does not exist yet, a payload with an unexpected shape — none of them
may stop a user reading a session they are entitled to, because the file is right
there and the mirror is an optimisation. That is the same best-effort contract
:class:`~paperwhisperer.ingestion.orchestrator.ParseCache` has, and for the same
reason: a mirror must never be the reason a request fails.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any

from paperwhisperer.core import config
from paperwhisperer.core.text import (
    atomic_write_json,
    build_document_excerpt,
    build_session_expiry,
    build_session_id,
    compact_text,
    normalize_next_actions,
    normalize_text_items,
    now_iso,
    parse_iso_datetime,
    remove_file_safely,
)
from paperwhisperer.search.normalize import normalize_paper_collection

logger = logging.getLogger(__name__)

_last_cleanup_at = 0.0
_cleanup_lock = threading.Lock()


def get_session_file_path(session_id: str) -> str:
    return os.path.join(config.CONTEXT_FOLDER, f"{session_id}.json")


def generate_session_token() -> str:
    return secrets.token_urlsafe(24)


def hash_session_token(token) -> str:
    return hashlib.sha256(str(token or "").encode("utf-8")).hexdigest()


def validate_session_token(session_payload: dict, session_token) -> bool:
    """Constant-time token check against the stored hash."""
    expected_hash = str((session_payload.get("session_auth") or {}).get("token_hash") or "")
    provided_hash = hash_session_token(session_token)
    return bool(
        expected_hash and session_token and secrets.compare_digest(expected_hash, provided_hash)
    )


def get_session_document_content(session_payload: dict) -> str:
    """Prefer the full document when persisted, else fall back to the excerpt."""
    content = str(session_payload.get("document_content") or "")
    if content:
        return content
    return str(session_payload.get("document_excerpt") or "")


def write_session_payload(session_id: str, payload) -> None:
    now_text = now_iso()
    if isinstance(payload, dict):
        payload["updated_at"] = now_text
        payload["expires_at"] = build_session_expiry()
    atomic_write_json(get_session_file_path(session_id), payload)


def build_session_payload(
    session_id: str,
    source_filename: str,
    document_content: str,
    analysis: dict,
    session_token: str,
    document_id: str = "",
    paper_facts: dict | None = None,
) -> dict:
    """The session a fresh analysis starts from.

    ``document_id`` and ``paper_facts`` are optional keyword arguments rather
    than new required ones, and both are omitted from the payload when empty.
    That is the whole contract T1.8 needs: a session is where the analysed
    document's identity and its extracted fact layer get parked, and a caller that
    has them can hand them over without this module having to be changed again.

    Why they are parked here at all, when the fact layer has a table of its own
    (``paper_facts`` in the store tier, with ``put_facts``/``get_facts`` on both
    tiers): this is the *session* record, and it is what survives a browser
    refresh, a worker restart and a re-analysis. The row is the durable copy of
    "what the user was looking at", and an answer whose provenance is only in
    another table cannot be explained when that table's index version has moved
    on. T1.8 still owns writing the fact layer itself; this only stops the session
    from being unable to say which paper it is about.
    """
    generated_at = now_iso()
    stored_document_content = (
        document_content if config.SESSION_PERSIST_FULL_DOCUMENT else ""
    )
    payload = {
        "session_id": session_id,
        "source_filename": source_filename,
        "generated_at": generated_at,
        "created_at": generated_at,
        "updated_at": generated_at,
        "expires_at": build_session_expiry(),
        "document_content": stored_document_content,
        "document_excerpt": build_document_excerpt(document_content),
        "qa_history": [],
        "paper_search": {
            "last_query": "",
            "last_results": [],
            "last_recommendation": {},
            "reading_queue": [],
        },
        "session_auth": {"token_hash": hash_session_token(session_token)},
        "analysis": {
            "summary": analysis.get("summary", ""),
            "quotes": analysis.get("quotes", ""),
            "mindmap": analysis.get("mindmap", ""),
            "mermaid": analysis.get("mermaid", ""),
            "evaluation": analysis.get("evaluation", ""),
            "research_brief": analysis.get("research_brief", ""),
            "sections": analysis.get("sections", {}),
            "char_count": analysis.get("char_count", 0),
            "elapsed_seconds": analysis.get("elapsed_seconds"),
            "output_file": analysis.get("output_file", ""),
            "suggested_questions": analysis.get("suggested_questions", []),
            "next_actions": analysis.get("next_actions", []),
            "analysis_status": analysis.get("analysis_status", {}),
        },
    }
    if document_id:
        payload["document_id"] = str(document_id)
    if isinstance(paper_facts, dict) and paper_facts:
        payload["paper_facts"] = dict(paper_facts)
    return payload


#: The fields :func:`~paperwhisperer.search.normalize.normalize_paper_collection`
#: rebuilds a paper record from. It is a *registry of what gets cleaned*, not a
#: whitelist of what may exist: anything outside this set is carried through
#: untouched by :func:`_normalize_papers_preserving_extras`.
#:
#: Kept as data, and kept next to the function that enforces it, so that adding a
#: field to the normaliser is a two-place edit somebody can see from here.
#: Mirroring ``normalize_paper_collection``'s own key set is the price of using it
#: as the cap-and-dedupe step; the alternative is re-implementing it here, which
#: would be a second copy of the identity rules that T3.1 is already changing.
PAPER_RECORD_CLEANED_KEYS: frozenset[str] = frozenset(
    {
        "source",
        "paper_id",
        "title",
        "abstract",
        "authors",
        "year",
        "venue",
        "url",
        "pdf_url",
        "saved_at",
        "doi",
        "arxiv_id",
        "s2_paper_id",
        "openalex_id",
        "identity_key",
    }
)


def _paper_extras_by_identity(values) -> dict[str, dict]:
    """Index each record's *uncleaned* extra fields by its paper identity.

    Extras are attached to the record they arrived on rather than to the record
    that happens to be in the same list position, because the cap and the dedupe
    in :func:`~paperwhisperer.search.normalize.normalize_paper_collection` move
    and drop records. Matching by position would give a paper somebody else's
    "why is this recommended" reasons — a wrong answer that looks right.
    """
    from paperwhisperer.search.identity import identity_from_record

    indexed: dict[str, dict] = {}
    for value in values or []:
        if not isinstance(value, dict):
            continue
        extras = {key: item for key, item in value.items() if key not in PAPER_RECORD_CLEANED_KEYS}
        if not extras:
            continue
        try:
            key = str(identity_from_record(value).key or "")
        except Exception:  # noqa: BLE001 - identity is best-effort here
            key = ""
        if not key:
            # No usable identity: fall back to the title and link, which is what
            # a bare client-side record has. Two records that collide here are
            # two rows the user can see, and the later one's extras win.
            key = f"title:{str(value.get('title') or '').strip()}|url:{str(value.get('url') or '')}"
        indexed.setdefault(key, {}).update(extras)
    return indexed


def _normalize_papers_preserving_extras(values, max_items: int) -> list[dict]:
    """Normalise a paper collection **without discarding fields we do not know**.

    The policy, stated once so that the next person does not have to discover it
    in production:

    **The session store preserves unknown fields.** A field another module adds
    to a paper record — ``reasons`` from the multi-signal recommender,
    ``document_id``, ``paper_facts``, anything — survives a write and a read,
    because the only normalising this module does is to *clean the fields it has a
    rule for* and to cap and dedupe the list. It does not rebuild each record from
    a fixed key set.

    That is the opposite of what the underlying
    :func:`~paperwhisperer.search.normalize.normalize_paper_collection` does, and
    the difference is not cosmetic. That function rebuilds a record from ten
    named fields, which is right for a *search result* — it is a projection of a
    provider's payload into this app's paper shape. It is wrong for a *stored
    session*, because a session is where another module's work gets parked, and a
    projection strips everything the projection did not anticipate. Three fields
    have already been lost this way (``document_id``, ``paper_facts``,
    ``reasons``), and the failure is invisible: the data is written, the write
    reports success, and the reason a paper was recommended is gone by the time
    anybody reads the session back.

    So the rule is the other way round. Known fields are cleaned (truncated to
    their limit, typed, identity-attached); unknown fields are copied verbatim;
    and nothing is dropped except a record with no title, which is a titled
    nothing. A module that needs a field cleaned *differently* adds it to
    :data:`PAPER_RECORD_CLEANED_KEYS` and writes the rule here — which is the one
    place to look, and the reason this helper exists rather than a call to the
    normaliser.
    """
    normalized = normalize_paper_collection(values, max_items=max_items)
    if not isinstance(values, list):
        return normalized
    extras = _paper_extras_by_identity(values)
    if not extras:
        return normalized
    for record in normalized:
        key = str(record.get("identity_key") or "")
        carried = extras.get(key)
        if carried is None:
            fallback = (
                f"title:{str(record.get('title') or '').strip()}"
                f"|url:{str(record.get('url') or '')}"
            )
            carried = extras.get(fallback)
        if carried:
            # Never overwrite a cleaned field with an uncleaned one: the whole
            # value of this function is that the cleaning still happened.
            for extra_key, extra_value in carried.items():
                record.setdefault(extra_key, extra_value)
    return normalized


def _normalize_recommendation(last_recommendation) -> dict:
    if not isinstance(last_recommendation, dict):
        return {}
    last_recommendation["original_query"] =compact_text(last_recommendation.get("original_query"),
        limit=240)
    last_recommendation["query"] = compact_text(last_recommendation.get("query"), limit=240)
    last_recommendation["reason"] = compact_text(last_recommendation.get("reason"), limit=500)
    last_recommendation["rewrite_model"] =compact_text(last_recommendation.get("rewrite_model"),
        limit=120)
    last_recommendation["generated_at"] =compact_text(last_recommendation.get("generated_at"),
        limit=40)
    last_recommendation["items"] = _normalize_papers_preserving_extras(
        last_recommendation.get("items"),
        max_items=config.RECOMMENDATION_RESULT_LIMIT,
    )
    last_recommendation["topics"] = normalize_text_items(
        last_recommendation.get("topics"), max_items=6, item_limit=120
    )
    last_recommendation["errors"] = normalize_text_items(
        last_recommendation.get("errors"), max_items=4, item_limit=240
    )
    return last_recommendation


def _normalize_loaded_payload(payload: dict, session_id: str) -> dict:
    """Coerce a session file into the shape the rest of the app expects.

    Session files are written by this app but may be hand-edited or truncated;
    every field is re-typed here so downstream code can trust the structure.
    """
    document_content = str(payload.get("document_content") or "")
    document_excerpt = str(
        payload.get("document_excerpt") or build_document_excerpt(document_content)
    )
    qa_history = payload.get("qa_history")
    analysis = payload.get("analysis")
    paper_search = payload.get("paper_search")
    session_auth = payload.get("session_auth")

    payload["document_content"] = document_content
    payload["document_excerpt"] = document_excerpt
    payload["qa_history"] = qa_history if isinstance(qa_history, list) else []
    payload["analysis"] = analysis if isinstance(analysis, dict) else {}
    payload["paper_search"] = paper_search if isinstance(paper_search, dict) else {}
    payload["session_auth"] = session_auth if isinstance(session_auth, dict) else {}
    payload["source_filename"] = str(payload.get("source_filename") or "")
    payload["session_id"] = str(payload.get("session_id") or session_id)
    # Re-typed, never dropped. These two are the analysed document's identity and
    # its fact layer; a file that carries them keeps them, and a file that does
    # not is left without them rather than given an empty one, because "this
    # session was never told which document it is about" and "this session says
    # the document is ''" are different facts and only one of them is true.
    if "document_id" in payload:
        payload["document_id"] = str(payload.get("document_id") or "")
    if "paper_facts" in payload and not isinstance(payload["paper_facts"], dict):
        payload["paper_facts"] = {}
    payload.setdefault("generated_at", now_iso())
    payload.setdefault("created_at", payload.get("generated_at") or now_iso())
    payload.setdefault("updated_at", payload.get("generated_at") or now_iso())
    payload.setdefault("expires_at", build_session_expiry())

    payload["paper_search"]["last_query"] = compact_text(
        payload["paper_search"].get("last_query"), limit=240
    )
    payload["paper_search"]["last_results"] = _normalize_papers_preserving_extras(
        payload["paper_search"].get("last_results"),
        max_items=config.PAPER_SEARCH_RESULT_LIMIT,
    )
    payload["paper_search"]["last_recommendation"] = _normalize_recommendation(
        payload["paper_search"].get("last_recommendation")
    )
    payload["paper_search"]["reading_queue"] = _normalize_papers_preserving_extras(
        payload["paper_search"].get("reading_queue"),
        max_items=config.READING_QUEUE_LIMIT,
    )

    payload["session_auth"].setdefault("token_hash", "")

    analysis_map = payload["analysis"]
    if not isinstance(analysis_map.get("sections"), dict):
        analysis_map["sections"] = {}
    analysis_map["suggested_questions"] = normalize_text_items(
        analysis_map.get("suggested_questions"), max_items=6, item_limit=180
    )
    analysis_map["next_actions"] = normalize_next_actions(
        analysis_map.get("next_actions"), max_items=5
    )
    if not isinstance(analysis_map.get("analysis_status"), dict):
        analysis_map["analysis_status"] = {}
    return payload


def load_session_payload(session_id: str):
    """Read a session file, deleting it when unreadable, malformed or expired."""
    session_file = get_session_file_path(session_id)
    if not os.path.exists(session_file):
        return None
    try:
        with open(session_file, encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        remove_file_safely(session_file, "corrupt session file")
        logger.warning("Removed corrupt session file %s: %s", session_file, exc)
        return None
    except UnicodeDecodeError as exc:
        # A file that is not valid UTF-8 is evidence worth keeping, so it is
        # reported rather than removed the way a JSON syntax error is. This path
        # used to be an unhandled exception, which turned one bad file into a 500.
        logger.warning("Session file %s is not valid UTF-8: %s", session_file, exc)
        return None
    except OSError as exc:
        logger.warning("Unable to read session file %s: %s", session_file, exc)
        return None

    if not isinstance(payload, dict):
        remove_file_safely(session_file, "non-object session file")
        logger.warning("Removed non-object session file %s", session_file)
        return None

    expires_at = parse_iso_datetime(payload.get("expires_at"))
    if expires_at and expires_at.timestamp() < time.time():
        remove_file_safely(session_file, "expired session file")
        logger.info("Removed expired session file: %s", session_file)
        return None
    if not expires_at:
        payload["expires_at"] = build_session_expiry()

    # The migration is deliberately placed after the expiry check and before
    # _normalize_loaded_payload: an expired session must not be projected into
    # the database, and the row must hold the file's verbatim payload rather than
    # a normalized one — normalization adds fields, so storing it would make the
    # row's payload_sha256 disagree with the file it claims to mirror.
    migrate_session_on_read(
        session_id,
        payload,
        source_path=session_file,
        source_bytes=_read_source_bytes(session_file, payload),
    )
    return _normalize_loaded_payload(payload, session_id)


def _read_source_bytes(path: str, payload: Any) -> bytes:
    """The session file's exact bytes, for the migration's idempotence key.

    Read separately from the payload rather than in place of it, for one reason:
    ``json.load`` on a text handle cannot give back the bytes it read. Universal
    newline translation means ``text.encode("utf-8")`` is not always the file —
    a file written with CRLF would hash to something its own bytes do not, and the
    row would then be rewritten on every single read, which is precisely the
    "a migration that is not idempotent" failure the digest exists to prevent.
    ``payload_sha256`` covers the content; this covers the file, and the two are
    different questions worth answering separately.

    Falls back to the content digest when the file cannot be read again — a race
    with a writer, a permission change between the two reads. The fallback is
    still stable for unchanged content, so idempotence survives; only the
    byte-exactness is lost, and the next successful read reconciles it with one
    refresh.
    """
    try:
        with open(path, "rb") as handle:
            return handle.read()
    except OSError:
        return json.dumps(
            payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")


def load_validated_session(raw_session_id, session_token, require_token: bool = True):
    """Resolve a session id and enforce token ownership.

    Raises ``ValueError`` when the session is missing and ``PermissionError``
    when the token does not match, so callers can map them to 400 and 403.
    """
    if not raw_session_id:
        raise ValueError("session_id is required.")
    safe_session_id = build_session_id(raw_session_id)
    session_payload = load_session_payload(safe_session_id)
    if not session_payload:
        raise ValueError(
            "Session expired or context not found. Please upload and analyze the file again."
        )
    if require_token and not validate_session_token(session_payload, session_token):
        raise PermissionError(
            "Invalid or missing session token. Please analyze the document again."
        )
    return safe_session_id, session_payload


def cleanup_expired_sessions(force: bool = False) -> None:
    """Best-effort sweep of expired and unreadable session files.

    Rate-limited by ``SESSION_CLEANUP_INTERVAL_SECONDS``; the directory scan
    is skipped entirely when the last pass was recent.
    """
    global _last_cleanup_at

    with _cleanup_lock:
        now_ts = time.time()
        if not force and now_ts - _last_cleanup_at < config.SESSION_CLEANUP_INTERVAL_SECONDS:
            return
        _last_cleanup_at = now_ts

    try:
        session_files = os.listdir(config.CONTEXT_FOLDER)
    except OSError as exc:
        logger.warning("Unable to list session folder %s: %s", config.CONTEXT_FOLDER, exc)
        return

    for name in session_files:
        if not name.endswith(".json"):
            continue
        file_path = os.path.join(config.CONTEXT_FOLDER, name)
        try:
            with open(file_path, encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict):
                raise ValueError("session payload must be an object")
            expires_at = parse_iso_datetime(payload.get("expires_at"))
            if expires_at and expires_at.timestamp() < now_ts:
                remove_file_safely(file_path, "expired session cleanup file")
                logger.info("Removed expired session file: %s", file_path)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            remove_file_safely(file_path, "unreadable session cleanup file")
            logger.warning("Removed unreadable session file %s: %s", file_path, exc)


# --------------------------------------------------------------------------
# One-way migration into the T2.2 tables
# --------------------------------------------------------------------------

#: Every key this module reads for the migration, with its default.
#:
#: ``SESSION_MIGRATION_ENABLED``
#:     Default ``true``. Read with ``getattr`` and a default, like
#:     :data:`~paperwhisperer.infra.stores.factory.REQUIRED_CONFIG_KEYS`, because
#:     :mod:`paperwhisperer.core.config` is owned by another task and importing a
#:     name from it would break the moment that module renamed one. Whoever adds
#:     the key to ``core/config.py`` copies this name and this default.
#:
#: There is no "already migrated" flag in this table on purpose. Whether a
#: migration has run is decided by comparing content, and a flag would be a
#: second answer to that question which could disagree with the first.
SESSION_MIGRATION_CONFIG_KEYS: dict[str, Any] = {
    "SESSION_MIGRATION_ENABLED": True,
}

#: An operator can turn the mirror off without a code change, for a deployment
#: that deliberately wants the JSON files to be the only copy. Read from the
#: environment rather than from config because it is a property of *this*
#: process's environment, and a key in ``.env`` is easy to forget about in the
#: middle of debugging something else.
DISABLE_SESSION_MIGRATION_ENV = "PAPERWHISPERER_DISABLE_SESSION_MIGRATION"

_session_store_lock = threading.Lock()
_session_store: Any = None
_session_store_resolved = False


class _SessionStoreSettings:
    """Adapts the real config to the store factory's ``getattr``-style access.

    A shim rather than a settings object of my own. The store factory and the
    table factory both read their configuration with ``getattr`` and a default,
    and the only correct defaults are the ones
    :data:`~paperwhisperer.infra.stores.factory.REQUIRED_CONFIG_KEYS`
    documents — so this forwards to that table instead of restating it, and a
    change to a default there reaches the session mirror with no edit here.
    """

    def __init__(self, source: Any) -> None:
        self._source = source

    def __getattr__(self, name: str) -> Any:
        from paperwhisperer.infra.stores.factory import REQUIRED_CONFIG_KEYS

        if name in REQUIRED_CONFIG_KEYS:
            return getattr(self._source, name, REQUIRED_CONFIG_KEYS[name])
        raise AttributeError(name)


def install_table_store(store: Any) -> None:
    """Use ``store`` for the session mirror, and stop resolving one lazily.

    The wiring point for the app (T2.4) and for every test that wants a mirror
    without touching the developer's real data directory. Passing ``None``
    disables the mirror entirely, which is also how a caller expresses "this
    deployment runs on files only".
    """
    global _session_store, _session_store_resolved
    with _session_store_lock:
        _session_store = store
        _session_store_resolved = True


def reset_table_store() -> None:
    """Forget the installed store and go back to resolving one lazily."""
    global _session_store, _session_store_resolved
    with _session_store_lock:
        _session_store = None
        _session_store_resolved = False


def _as_bool_env(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def get_table_store() -> Any:
    """The table store the mirror writes to, or ``None`` when unavailable.

    Resolved once per process, on first use, and cached: building the store tier
    probes for FTS5 and for numpy, and doing that inside the first session read
    of a request would put a start-up cost in the request path. Once per process
    is the right frequency — a deployment's storage does not change while it is
    running, and a caller that needs it to is what :func:`install_table_store` is
    for.

    Every failure path returns ``None`` rather than raising. A mirror that cannot
    be reached is a missing optimisation; a session read that raises because the
    mirror is down is a user who cannot read their own work. The reasons are
    logged, because a mirror that silently never ran is indistinguishable from a
    migration that was never needed.
    """
    global _session_store, _session_store_resolved
    with _session_store_lock:
        if _session_store_resolved:
            return _session_store
        _session_store_resolved = True
        enabled = _as_bool_env(
            getattr(
                config,
                "SESSION_MIGRATION_ENABLED",
                SESSION_MIGRATION_CONFIG_KEYS["SESSION_MIGRATION_ENABLED"],
            )
        )
        if not enabled or os.environ.get(DISABLE_SESSION_MIGRATION_ENV, "").strip():
            logger.info("会话数据库镜像已按配置关闭，会话继续以 JSON 文件为准。")
            _session_store = None
            return None
        try:
            from paperwhisperer.infra.tables import build_table_store

            _session_store = build_table_store(_SessionStoreSettings(config))
        except Exception as exc:  # noqa: BLE001 - the mirror is best-effort
            logger.warning("会话数据库镜像不可用，会话继续以 JSON 文件为准：%s", exc)
            _session_store = None
        return _session_store


@dataclass(slots=True)
class SessionMigration:
    """What one migration attempt did, in terms a caller can log or assert on.

    ``reason`` is the machine-readable half and it is deliberately more precise
    than a boolean, because the three interesting outcomes are different: a first
    write (``migrated``), a re-projection after the file changed (``refreshed``),
    and the no-op that makes running the migration twice safe
    (``already_migrated``). A caller that cannot tell them apart cannot tell
    whether the mirror is current.

    ``source_preserved`` is a *verified* claim rather than an intention: the
    file's existence is re-checked after the write. It is in the result because
    the guarantee is the one thing about this migration that cannot be recovered
    if it is violated — a mirror can be rebuilt from a row, but not from
    nothing.
    """

    session_id: str = ""
    conversation_id: str = ""
    migrated: bool = False
    reason: str = ""
    source_sha256: str = ""
    payload_sha256: str = ""
    message_count: int = 0
    queue_item_count: int = 0
    source_file: str = ""
    source_preserved: bool = False

    def summary_line(self) -> str:
        """One Chinese sentence, the way the rest of this project reports."""
        if self.reason == "already_migrated":
            return f"会话 {self.session_id} 已在数据库中，未重复迁移（原文件保留）。"
        if not self.migrated:
            return f"会话 {self.session_id} 未迁移：{self.reason}。"
        verb = "已迁移到数据库" if self.reason == "migrated" else "已按最新文件刷新数据库"
        return (
            f"会话 {self.session_id} {verb}"
            f"（消息 {self.message_count} 条，队列 {self.queue_item_count} 项，原文件保留）。"
        )


def _canonical_json_digest(payload: Any) -> str:
    """SHA-256 of a payload's canonical JSON form.

    Canonical, so that two equal payloads have one digest: the digest answers
    "did the *content* change", and a whitespace or key-order change in the file
    must not look like a new session. Sorted keys, no ASCII escaping and no
    insignificant whitespace — the same form both store backends write, so a
    digest computed here can be compared against one computed there.
    """
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _project_messages(payload: dict) -> list[dict]:
    """Project ``qa_history`` and the analysis summary into message rows.

    A Q&A turn becomes *two* rows, the question and the answer, because that is
    what a transcript is and because T1.7's ``claims[evidence_ids]`` attaches to
    an answer rather than to a question-and-answer blob. One row holding both
    would make "what did the user ask" unqueryable.

    The analysis summary becomes one ``kind="analysis"`` message, and that is the
    only analysis text the projection carries: the full analysis dictionary is in
    ``conversations.payload`` verbatim, so nothing is lost, and a per-section
    message set is T1.8's and T2.4's to design — a migration must not invent a
    shape its consumers have not agreed on.

    Every field is coerced rather than trusted. These files are written by this
    app but may be hand-edited, truncated, or left by a build from another
    project, and a migration is the worst possible place to trust a shape it
    cannot re-derive from the file.
    """
    rows: list[dict] = []
    turns = payload.get("qa_history")
    if isinstance(turns, list):
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            stamp = str(turn.get("timestamp") or "").strip()
            question = str(turn.get("question") or "").strip()
            answer = str(turn.get("answer") or "").strip()
            if question:
                rows.append(
                    {"role": "user", "kind": "qa", "content": question, "created_at": stamp}
                )
            if answer:
                rows.append(
                    {
                        "role": "assistant",
                        "kind": "qa",
                        "content": answer,
                        "answer_mode": str(turn.get("answer_mode") or "").strip(),
                        "confidence": str(turn.get("confidence") or "").strip(),
                        "created_at": stamp,
                    }
                )
    analysis = payload.get("analysis")
    if isinstance(analysis, dict):
        summary = str(analysis.get("summary") or "").strip()
        if summary:
            rows.append({"role": "assistant", "kind": "analysis", "content": summary})
    # An empty created_at would defeat the column's default, and a timestamp in
    # an unknown format would defeat a reader ordering by it. Dropping the key
    # lets both the default and the reader cope.
    for row in rows:
        if not row.get("created_at"):
            row.pop("created_at", None)
    return rows


def _project_queue(payload: dict) -> list[dict]:
    """Project ``paper_search.reading_queue`` into ``reading_queue`` rows.

    Flat identity columns *plus* the whole record in ``payload``, because T3.1
    replaces the identity rules and a queue table that hard-codes today's
    precedence would need migrating the day they change. The flat columns are the
    ones a query needs; ``payload`` is the one that stays correct.
    """
    search = payload.get("paper_search")
    items = search.get("reading_queue") if isinstance(search, dict) else None
    rows: list[dict] = []
    if not isinstance(items, list):
        return rows
    for position, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        authors = item.get("authors")
        rows.append(
            {
                "position": position,
                "title": title,
                "doi": str(item.get("doi") or "").strip(),
                "arxiv_id": str(item.get("arxiv_id") or "").strip(),
                "paper_id": str(item.get("paper_id") or "").strip(),
                "url": str(item.get("url") or "").strip(),
                "pdf_url": str(item.get("pdf_url") or "").strip(),
                "venue": str(item.get("venue") or "").strip(),
                "abstract": str(item.get("abstract") or "").strip(),
                "authors": list(authors) if isinstance(authors, list) else [],
                "payload": dict(item),
                "added_at": str(item.get("saved_at") or "").strip(),
            }
        )
    for row in rows:
        if not row.get("added_at"):
            row.pop("added_at", None)
    return rows


def migrate_session_payload(
    session_id: str,
    payload: dict,
    *,
    source_path: str = "",
    source_bytes: bytes = b"",
    table_store: Any = None,
) -> SessionMigration:
    """Project one session file into the asset tables. Idempotent and lossless.

    The three properties, and how each is obtained:

    *Lossless.* ``conversations.payload`` is the dict this function was handed,
    unmodified — the exact value ``json.load`` produced, before any normalization
    anywhere in this module. Everything else written here (the messages, the
    queue) is a projection of it and is allowed to be lossy. The original file is
    also left exactly where it is: this function never writes to it and never
    unlinks it.

    *Idempotent.* The row records the digest of the source. An unchanged file
    produces the same digest and the function returns before writing anything, so
    a second run is not a second write, and ``migrated_at`` on the row is the
    observable that proves it. A *changed* file is re-projected, because the file
    is authoritative and a stale row is a lie — that is a different operation from
    the first, and is reported as ``refreshed`` so a caller can tell them apart.

    *Marked.* ``conversations.source_migrated`` is set on the row and
    ``migrated_at`` records when, because a reader asking "is this row a migrated
    session or something the app wrote natively" needs the answer on the row
    rather than in a log.

    Nothing here raises for a shape it did not expect. A session file this
    function cannot fully project still gets its conversation row, because a row
    with no messages is a recoverable gap and a lost session is not.
    """
    session = str(session_id or "").strip()
    if not session:
        return SessionMigration(reason="missing_session_id")
    store = table_store if table_store is not None else get_table_store()
    if store is None:
        return SessionMigration(session_id=session, reason="store_unavailable")
    # Imported here, not at module scope, for two reasons. The store tier pulls in
    # numpy probing and a FTS5 probe, and sessions.store is imported by the app
    # on the request path; and the names below are the only ones this module needs
    # from it, so a module-level import would make every session read import a
    # database package it may never use.
    from paperwhisperer.infra.tables import DuplicateRow, UnknownRow, VersionConflict

    data = dict(payload or {})
    source_sha = (
        hashlib.sha256(source_bytes).hexdigest() if source_bytes else _canonical_json_digest(data)
    )
    payload_sha = _canonical_json_digest(data)
    mtime_ns = 0
    if source_path and os.path.isfile(source_path):
        try:
            mtime_ns = int(os.stat(source_path).st_mtime_ns)
        except OSError:
            mtime_ns = 0

    auth = data.get("session_auth")
    token_hash = str(auth.get("token_hash") or "") if isinstance(auth, dict) else ""
    source_filename = str(data.get("source_filename") or "").strip()
    conversation = {
        "session_id": session,
        "title": source_filename or f"会话 {session}",
        "status": "active",
        "token_hash": token_hash,
        "payload": data,
        "payload_sha256": payload_sha,
        "source_path": str(source_path or ""),
        "source_sha256": source_sha,
        "source_mtime_ns": mtime_ns,
        "source_migrated": True,
        "migrated_at": now_iso(),
        "expires_at": str(data.get("expires_at") or ""),
        "created_at": str(data.get("created_at") or data.get("generated_at") or now_iso()),
        "updated_at": str(data.get("updated_at") or now_iso()),
    }
    # Lifted out of the verbatim payload so the row can be joined to the rest of
    # the asset model. The payload still holds the original value, so lifting it
    # costs nothing and a session that never had a document_id is not given one.
    document_id = str(data.get("document_id") or "").strip()
    if document_id:
        conversation["document_id"] = document_id

    existing = store.get_conversation(session)
    if existing is not None and (
        existing.get("source_sha256") == source_sha
        and existing.get("payload_sha256") == payload_sha
    ):
        # The no-op. Every statement below this line is a write, so returning
        # here is what makes a second migration cost nothing and change nothing.
        return SessionMigration(
            session_id=session,
            conversation_id=session,
            migrated=False,
            reason="already_migrated",
            source_sha256=source_sha,
            payload_sha256=payload_sha,
            source_file=str(source_path or ""),
            source_preserved=(os.path.isfile(source_path) if source_path else True),
        )

    messages = _project_messages(data)
    queue = _project_queue(data)
    if existing is None:
        try:
            store.add_conversation({"conversation_id": session, **conversation})
            reason = "migrated"
        except DuplicateRow as exc:
            # Another process projected this session between our read and our
            # write. Not an error: the row that exists is the row we wanted.
            logger.info("会话 %s 已被其他进程迁移：%s", session, exc)
            return SessionMigration(
                session_id=session,
                conversation_id=session,
                migrated=False,
                reason="raced",
                source_sha256=source_sha,
                payload_sha256=payload_sha,
                source_file=str(source_path or ""),
            )
    else:
        changes = {
            key: value
            for key, value in conversation.items()
            if key not in {"session_id", "created_at"}
        }
        try:
            store.update_conversation(
                session, changes, expected_version=int(existing.get("version") or 0)
            )
            reason = "refreshed"
        except (VersionConflict, UnknownRow) as exc:
            # Somebody else is writing this conversation right now. The file is
            # still there and the next read re-projects it, so standing aside is
            # correct and failing the user's read is not.
            logger.info("会话 %s 的数据库镜像暂时未更新：%s", session, exc)
            return SessionMigration(
                session_id=session,
                conversation_id=session,
                migrated=False,
                reason="conflict",
                source_sha256=source_sha,
                payload_sha256=payload_sha,
                source_file=str(source_path or ""),
            )

    store.replace_messages(session, messages)
    store.replace_reading_queue(session, queue)
    preserved = os.path.isfile(source_path) if source_path else True
    if source_path and not preserved:
        logger.error("会话 %s 迁移后原文件消失，这违反迁移契约，请立即备份。", session)
    return SessionMigration(
        session_id=session,
        conversation_id=session,
        migrated=True,
        reason=reason,
        source_sha256=source_sha,
        payload_sha256=payload_sha,
        message_count=len(messages),
        queue_item_count=len(queue),
        source_file=str(source_path or ""),
        source_preserved=preserved,
    )


def migrate_session_file(session_id: str, *, table_store: Any = None) -> SessionMigration:
    """Migrate one session file, reading it the way a request would.

    Reads the file itself rather than calling :func:`load_session_payload`, so
    the payload it stores is the file's verbatim content. Going through the read
    path instead would store a *normalized* payload, and the row's
    ``payload_sha256`` would no longer describe the file the row claims to
    mirror — a difference invisible until somebody tried to verify it.
    """
    path = get_session_file_path(session_id)
    if not os.path.isfile(path):
        return SessionMigration(session_id=str(session_id), reason="missing_source")
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return SessionMigration(
            session_id=str(session_id), source_file=path, reason=f"unreadable: {exc}"
        )
    if not isinstance(payload, dict):
        return SessionMigration(
            session_id=str(session_id), source_file=path, reason="not_an_object"
        )
    return migrate_session_payload(
        session_id,
        payload,
        source_path=path,
        source_bytes=raw,
        table_store=table_store,
    )


def migrate_all_sessions(*, table_store: Any = None) -> list[SessionMigration]:
    """Migrate every session file in the context folder. Idempotent as a whole.

    The bulk entry point for a deployment that wants the mirror populated before
    the first request rather than session by session. It has no ``force`` flag on
    purpose: a flag meaning "write again anyway" is an invitation to write twice
    by accident, and idempotence is the property being asked for. A file that has
    not changed costs one read and one comparison here.
    """
    results: list[SessionMigration] = []
    try:
        names = sorted(os.listdir(config.CONTEXT_FOLDER))
    except OSError as exc:
        logger.warning("无法列出会话目录 %s：%s", config.CONTEXT_FOLDER, exc)
        return results
    for name in names:
        if not name.endswith(".json"):
            continue
        results.append(migrate_session_file(name[: -len(".json")], table_store=table_store))
    written = sum(1 for result in results if result.migrated)
    if written:
        logger.info("会话批量迁移完成：%s/%s 个会话写入数据库。", written, len(results))
    return results


def migrate_session_on_read(
    session_id: str,
    payload: dict,
    *,
    source_path: str = "",
    source_bytes: bytes = b"",
) -> SessionMigration | None:
    """The hook :func:`load_session_payload` calls, and the one that cannot fail.

    Automatic migration on first read is T2.2's acceptance criterion, which means
    it cannot depend on somebody remembering to run a script. It also cannot be
    able to break the read, because the read is the one thing that must work: the
    file is right there, the mirror is an optimisation, and a user who cannot
    read their session because the database was briefly unreachable has been
    failed by a cache. So every exception is caught, logged in Chinese, and the
    read continues.
    """
    try:
        return migrate_session_payload(
            session_id,
            payload,
            source_path=source_path,
            source_bytes=source_bytes,
        )
    except Exception as exc:  # noqa: BLE001 - a mirror must not break a read
        logger.warning("会话 %s 的数据库镜像写入失败，仍以 JSON 文件读取：%s", session_id, exc)
        return None
