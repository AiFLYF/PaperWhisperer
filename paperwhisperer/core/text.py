"""General-purpose text helpers shared across the backend."""

from __future__ import annotations

import json
import logging
import os
import re
import time
import unicodedata
import uuid
from collections.abc import Iterable
from datetime import datetime
from typing import Any

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


def is_windows_reserved_name(value: str) -> bool:
    """True when the *stem* of ``value`` is a Windows device name.

    Windows matches the device name against the part before the first dot, so
    ``con.txt``, ``CON`` and ``Con.pdf`` all resolve to the console device
    rather than to a file. Trailing spaces and dots are stripped first because
    the Win32 layer does the same before matching, which is why ``con. .txt``
    behaves like ``con`` there but not in a naive ``split(".")`` check.

    The comparison is on the uppercased stem, since device names are
    case-insensitive on Windows.
    """
    if not value:
        return False
    stem = value.split(".", 1)[0].strip().rstrip(" .")
    return stem.upper() in WINDOWS_RESERVED_NAMES


def secure_filename(filename: str) -> str:
    """Reduce an arbitrary filename to a safe, traversal-free basename.

    Two independent hazards are removed here, and both are handled regardless
    of the platform this process happens to be running on:

    * **Path traversal.** ``/``, ``\\`` and every other non-portable character
      are deleted rather than resolved, so the result is always a bare
      basename and can never escape the folder it is joined to.
    * **Windows device names.** ``CON``, ``NUL``, ``COM1`` and friends still
      resolve to kernel devices on Windows *even when they carry an
      extension*, and a file that cannot be written is a data-loss bug.

      This used to be guarded by ``os.name == "nt"``, which was wrong in two
      ways. First, the sanitized name is not opened on the machine that
      produced it — it is written to the upload folder, returned to the
      browser, and (via the remote-import path) copied to storage the
      deployment does not control, any of which may be a Windows share.
      Second, ``os.name`` describes the *runtime*, so a Linux deployment
      silently produced ``con.pdf`` and only failed later, at the moment
      something tried to persist it, long after the point where prefixing was
      still free. Prefixing unconditionally makes the output portable.

    The ``_`` prefix also keeps this idempotent: the leading/trailing strip
    runs *before* the reserved-name check, so re-sanitizing an already
    prefixed name returns it unchanged instead of accumulating ``__`` on
    every round trip.
    """
    value = unicodedata.normalize("NFKD", str(filename)).encode("ascii", "ignore").decode("ascii")
    value = value.replace("/", " ").replace("\\", " ")
    value = "_".join(value.split())
    value = re.sub(r"[^A-Za-z0-9_.-]", "", value)
    value = re.sub(r"_+", "_", value)
    value = value.strip("._")
    if is_windows_reserved_name(value):
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


# --------------------------------------------------------------------------
# Secret redaction
# --------------------------------------------------------------------------
# Moved here from ``ingestion/limits.py`` so that there is exactly one
# implementation. Two redactors is not tidiness: a secret one of them
# happens to scrub and the other does not is a disclosure, and the module
# that "owns" the feature is not the only module that logs.
# ``ingestion/limits.py`` re-exports these names, so nothing that already
# imported them from there breaks.

_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # OpenAI-style keys, including the project-scoped ``sk-proj-`` form.
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}")),
    # GitHub tokens, classic and fine-grained.
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{16,}|github_pat_[A-Za-z0-9_]{20,})")),
    # AWS access key ids and the signature/credential query parameters of a
    # presigned URL. The last one matters here specifically: MinerU hands back
    # presigned OSS URLs and a socket-level error that quotes the URL puts a
    # working credential into an exception message.
    ("aws_key_id", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("presigned_query", re.compile(
        r"(?i)\b(X-Amz-(?:Signature|Credential|Security-Token)|Signature|sig|"
        r"X-Goog-Signature|access_token|api_key|apikey)=([^&\s\"']+)"
    )),
    # ``Authorization: Bearer ...`` in a header dump or a log line.
    ("bearer_header", re.compile(r"(?i)\b(Bearer|Basic|Token)\s+([A-Za-z0-9._\-+/=]{12,})")),
)

#: What a redaction replaces a match with. Long enough that a reader can tell
#: something was removed and where, short enough that a log line stays
#: readable, and identical for every secret so the shape of the removal does
#: not itself leak the shape of the credential.
REDACTED = "[REDACTED]"


def _live_secret_values() -> list[str]:
    """Credential values this process is actually holding, longest first.

    Read from the environment rather than from ``core.config`` on purpose:
    ``config`` snapshots the environment at import, so a value monkeypatched
    into ``os.environ`` after that would be invisible to it, and this function
    is called from logging handlers where "invisible" means "not redacted".
    The concrete keys are the two this project reads (see ``.env.example``);
    both are matched by length so a two-character placeholder is not treated as
    a secret and blanked out of every message.
    """
    values: list[str] = []
    for name in (
        "OPENAI_API_KEY",
        "MINERU_API_TOKEN",
        "RERANK_API_KEY",
        "SEMANTIC_SCHOLAR_API_KEY",
        "OPENREX_API_KEY",
        "CORS_API_KEY",
    ):
        value = str(os.environ.get(name, "") or "").strip()
        if len(value) >= 8:
            values.append(value)
    # Longest first: redacting a short value that is a prefix of a long one
    # would leave the tail of the long one behind.
    return sorted(set(values), key=len, reverse=True)


def redact_secrets(text: str, *, extra_values: Iterable[str] = ()) -> str:
    """Remove credential values and credential-shaped text from ``text``.

    Two passes, and the order is the whole design:

    1. **Literal.** Every value the process holds for a known credential
       variable, plus anything the caller supplies, is replaced with
       :data:`REDACTED`. This is the pass that cannot be evaded, because it
       does not have to recognise the credential — it already has it.
    2. **Pattern.** The shapes in :data:`_SECRET_PATTERNS`. This is the pass
       that works when the value is *not* ours: a key the process never held (a
       user's own, pasted into a document), or a presigned URL signature that
       was minted by a third party and is therefore a credential we do not have
       the value of.

    ``extra_values`` exists for the case that motivated the literal pass being
    insufficient on its own: a test that installs a sentinel credential, and a
    redaction filter installed in a child process whose environment was
    already cleared.

    Applied to a message that contained no secret, this returns it unchanged
    apart from the substitution of a key-equals-value pair, which is by design:
    ``api_key=`` in a log line is a leak whether or not the value behind it is
    real.
    """
    if not text:
        return str(text)
    result = str(text)
    for value in [*_live_secret_values(), *(str(item) for item in extra_values)]:
        candidate = str(value or "").strip()
        if len(candidate) < 8:
            # Short values would blank out unrelated text. An 8-character floor
            # is below any credential format in use and above the strings this
            # function would otherwise destroy.
            continue
        result = result.replace(candidate, REDACTED)
    for name, pattern in _SECRET_PATTERNS:
        if name == "presigned_query":
            result = pattern.sub(lambda m: f"{m.group(1)}={REDACTED}", result)
        else:
            result = pattern.sub(REDACTED, result)
    return result


class SecretRedactingFilter(logging.Filter):
    """A logging filter that redacts credentials from records and arguments.

    Exists because *where* a secret gets logged is not knowable in advance. A
    ``logging.Filter`` on the root logger sees every record, including the ones
    produced by third-party libraries, which is the only place a redaction that
    depends on knowing the call site can be reliable.

    It filters ``msg`` and ``args`` separately and then re-flattens, because a
    secret in a lazily-formatted ``%s`` argument is the common case: the
    formatting has not happened yet when the record is created, so redacting
    ``record.msg`` alone would miss it entirely.

    ``exc_info`` is deliberately **not** rewritten. A traceback is the one thing
    in a log that must stay verbatim — it is what makes a bug fixable — and
    scrubbing it would destroy the tool this filter exists to support. The
    documented consequence is that a credential embedded in a *local variable
    name or value inside a frame* is still visible in a traceback, which is why
    the leak paths this closes are ones where the credential reaches a message
    rather than one where it is merely held nearby.
    """

    def __init__(self, name: str = "", extra_values: Iterable[str] = ()) -> None:
        super().__init__(name)
        self._extra = tuple(str(item) for item in extra_values if str(item or "").strip())

    def filter(self, record: logging.LogRecord) -> bool:
        """Redact in place and always return ``True``.

        Returning ``False`` would drop the record, which is a worse outcome
        than a redacted one: a log line that vanished is a failure an operator
        cannot diagnose, and this filter's job is to make the line safe, not to
        decide what is worth saying.
        """
        try:
            if isinstance(record.msg, str):
                record.msg = redact_secrets(record.msg, extra_values=self._extra)
            elif record.msg is not None:
                record.msg = redact_secrets(str(record.msg), extra_values=self._extra)
            if record.args:
                if isinstance(record.args, dict):
                    record.args = {
                        key: self._clean(value) for key, value in record.args.items()
                    }
                else:
                    record.args = tuple(self._clean(value) for value in record.args)
        except Exception:  # pragma: no cover - a filter must never raise
            # A logging filter that raises turns every subsequent log call in
            # the process into an error, including the one that would have
            # reported the original problem.
            return True
        return True

    def _clean(self, value: Any) -> Any:
        if isinstance(value, str):
            return redact_secrets(value, extra_values=self._extra)
        if isinstance(value, BaseException):
            # The single most common way a credential reaches a log: a library
            # exception whose message quotes a request or a URL. Redacting the
            # message of an exception that is about to be formatted into a log
            # line is the difference between a diagnostic and a disclosure.
            try:
                value.args = tuple(
                    redact_secrets(str(item), extra_values=self._extra) for item in value.args
                )
            except Exception:  # pragma: no cover - exotic exception types
                pass
        return value

