"""General-purpose text helpers shared across the backend."""

from __future__ import annotations

import json
import logging
import os
import re
import time
import unicodedata
import uuid
from datetime import datetime

from paperwhisperer.core import config

logger = logging.getLogger(__name__)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def parse_iso_datetime(value):
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def build_session_expiry(now=None) -> str:
    base_time = now or datetime.now()
    expiry = datetime.fromtimestamp(base_time.timestamp() + config.SESSION_TTL_SECONDS)
    return expiry.isoformat(timespec="seconds")


def compact_text(text, limit: int = 400) -> str:
    collapsed = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit].rstrip() + "..."


def trim_text_for_log(text, limit: int = 2000) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...[truncated]"


def parse_year(value) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    match = re.search(r"(19|20)\d{2}", text)
    return match.group(0) if match else ""


def build_document_excerpt(content, limit: int = config.DOCUMENT_EXCERPT_LIMIT) -> str:
    """First ``limit`` characters of a document, used as the cached session copy."""
    return (content or "")[:limit]


def clean_extracted_text(text: str) -> str:
    """Normalize raw document text: collapse runs of whitespace, trim lines."""
    if not text:
        return ""
    # Collapse 3+ consecutive newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse runs of whitespace (excluding newlines) into single space
    text = re.sub(r"[^\S\n]+", " ", text)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def normalize_author_list(authors, limit: int = 8) -> list[str]:
    normalized = []
    for author in authors or []:
        if isinstance(author, str):
            name = author.strip()
        elif isinstance(author, dict):
            name = str(author.get("name") or author.get("author") or "").strip()
        else:
            name = str(getattr(author, "name", "") or "").strip()
        if name:
            normalized.append(name)
        if len(normalized) >= limit:
            break
    return normalized


def normalize_text_items(values, max_items: int = 8, item_limit: int = 180) -> list[str]:
    if not isinstance(values, list):
        return []
    items = []
    seen = set()
    for value in values:
        text = compact_text(value, limit=item_limit)
        if not text or text in seen:
            continue
        items.append(text)
        seen.add(text)
        if len(items) >= max_items:
            break
    return items


def normalize_next_actions(values, max_items: int = 5) -> list[dict]:
    if not isinstance(values, list):
        return []
    actions = []
    seen = set()
    for value in values:
        if not isinstance(value, dict):
            continue
        label = compact_text(value.get("label"), limit=40)
        prompt = compact_text(value.get("prompt"), limit=240)
        if not label or not prompt or prompt in seen:
            continue
        actions.append({"label": label, "prompt": prompt})
        seen.add(prompt)
        if len(actions) >= max_items:
            break
    return actions


def extract_message_text(content) -> str:
    """Flatten an OpenAI ``message.content`` that may be a string or a list."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and item.get("text"):
                    text_parts.append(str(item["text"]))
            else:
                item_type = getattr(item, "type", None)
                item_text = getattr(item, "text", None)
                if item_type == "text" and item_text:
                    text_parts.append(str(item_text))
        return "\n".join(text_parts).strip()
    return str(content).strip()


def extract_json_object(text) -> str:
    """Pull the outermost JSON object out of a possibly fenced model reply."""
    raw_text = str(text or "").strip()
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text, flags=re.IGNORECASE)
        raw_text = re.sub(r"\s*```$", "", raw_text)
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1 and end >= start:
        return raw_text[start:end + 1]
    return raw_text


# --------------------------------------------------------------------------
# Filenames and identifiers
# --------------------------------------------------------------------------

WINDOWS_RESERVED_NAMES = frozenset({
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
})


def secure_filename(filename: str) -> str:
    """Reduce an arbitrary filename to a safe, traversal-free basename."""
    value = unicodedata.normalize("NFKD", str(filename)).encode("ascii", "ignore").decode("ascii")
    value = value.replace("/", " ").replace("\\", " ")
    value = "_".join(value.split())
    value = re.sub(r"[^A-Za-z0-9_.-]", "", value)
    value = re.sub(r"_+", "_", value)
    value = value.strip("._")
    if os.name == "nt" and value and value.split(".")[0].upper() in WINDOWS_RESERVED_NAMES:
        value = f"_{value}"
    return value


def is_allowed_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in config.ALLOWED_EXTENSIONS


def sanitize_identifier(raw_value, prefix: str) -> str:
    candidate = secure_filename((raw_value or "").strip())
    return candidate or f"{prefix}_{uuid.uuid4().hex}"


def build_session_id(raw_session_id) -> str:
    return sanitize_identifier(raw_session_id, "session")


def build_unique_storage_path(folder: str, filename: str) -> str:
    root, ext = os.path.splitext(filename)
    return os.path.join(folder, f"{root}_{uuid.uuid4().hex}{ext}")


def build_safe_upload_filename(filename: str) -> str:
    original_ext = os.path.splitext(filename)[1].lower()
    if original_ext not in config.ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type. Please upload one of: {config.SUPPORTED_FILE_TYPES_TEXT}"
        )
    sanitized = secure_filename(filename)
    if not sanitized or not sanitized.lower().endswith(original_ext):
        return f"file_{uuid.uuid4().hex}{original_ext}"
    return sanitized


# --------------------------------------------------------------------------
# Resource cleanup
# --------------------------------------------------------------------------

def remove_file_safely(file_path, description: str = "temporary file") -> None:
    """Delete a file, swallowing any failure.

    This is called from ``finally`` blocks on the streaming paths, so it has to
    be total: an exception escaping here replaces the real result of the
    request and tears down the server mid-response. Locks, read-only mounts and
    sandboxed delete guards have all been observed to fail the removal, and the
    last of those signals refusal with ``SystemExit`` rather than ``OSError``.

    ``KeyboardInterrupt`` is deliberately not caught so Ctrl+C still works.
    """
    if not file_path or not os.path.exists(file_path):
        return
    try:
        os.remove(file_path)
    except (Exception, SystemExit):
        logger.exception("Failed to remove %s: %s", description, file_path)


def close_response_safely(response, description: str = "remote response") -> None:
    """Close a response, swallowing any failure (see ``remove_file_safely``)."""
    if not response:
        return
    try:
        response.close()
    except (Exception, SystemExit):
        logger.exception("Failed to close %s", description)


async def close_upload_file_safely(upload_file, description: str = "upload file") -> None:
    """Close an upload, swallowing any failure (see ``remove_file_safely``)."""
    if not upload_file:
        return
    try:
        await upload_file.close()
    except (Exception, SystemExit):
        logger.exception("Failed to close %s", description)


def atomic_write_json(file_path: str, payload) -> None:
    """Write JSON via a temp file + os.replace so readers never see a partial file."""
    temp_path = f"{file_path}.{uuid.uuid4().hex}.tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, file_path)
    finally:
        remove_file_safely(temp_path, "stale JSON temp file")


# --------------------------------------------------------------------------
# Retry backoff
# --------------------------------------------------------------------------

def get_retry_delay_seconds(attempt, max_delay: int = 6) -> int:
    try:
        attempt_index = max(0, int(attempt))
    except (TypeError, ValueError):
        attempt_index = 0
    try:
        delay_cap = max(1, int(max_delay))
    except (TypeError, ValueError):
        delay_cap = 6
    return min(2 * (attempt_index + 1), delay_cap)


def parse_retry_after_seconds(value, fallback_cap: int = 60):
    text = str(value or "").strip()
    if not text:
        return None
    try:
        seconds = int(float(text))
        return max(0, min(seconds, fallback_cap))
    except (TypeError, ValueError):
        pass
    try:
        from email.utils import parsedate_to_datetime

        target = parsedate_to_datetime(text)
        if target is not None:
            delta = int(target.timestamp() - time.time())
            return max(0, min(delta, fallback_cap))
    except Exception:
        pass
    return None


def compute_retry_after_delay(exc, attempt, max_delay) -> int:
    headers = getattr(exc, "headers", None)
    if headers is not None:
        retry_after = parse_retry_after_seconds(
            headers.get("Retry-After"), fallback_cap=max(max_delay, 60)
        )
        if retry_after is not None:
            return max(1, min(retry_after, max_delay))
    return get_retry_delay_seconds(attempt, max_delay=max_delay)
