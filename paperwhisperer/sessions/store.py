"""Session persistence and token-based access control.

A session is a single JSON file under ``context/``. Writes go through
:func:`atomic_write_json` so a reader never observes a half-written file, and
access requires a bearer token whose SHA-256 hash is what gets stored — the
raw token is returned to the client exactly once, at analysis time.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import threading
import time

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
) -> dict:
    generated_at = now_iso()
    stored_document_content = (
        document_content if config.SESSION_PERSIST_FULL_DOCUMENT else ""
    )
    return {
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
    last_recommendation["items"] = normalize_paper_collection(
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
    payload.setdefault("generated_at", now_iso())
    payload.setdefault("created_at", payload.get("generated_at") or now_iso())
    payload.setdefault("updated_at", payload.get("generated_at") or now_iso())
    payload.setdefault("expires_at", build_session_expiry())

    payload["paper_search"]["last_query"] = compact_text(
        payload["paper_search"].get("last_query"), limit=240
    )
    payload["paper_search"]["last_results"] = normalize_paper_collection(
        payload["paper_search"].get("last_results"),
        max_items=config.PAPER_SEARCH_RESULT_LIMIT,
    )
    payload["paper_search"]["last_recommendation"] = _normalize_recommendation(
        payload["paper_search"].get("last_recommendation")
    )
    payload["paper_search"]["reading_queue"] = normalize_paper_collection(
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

    return _normalize_loaded_payload(payload, session_id)


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
