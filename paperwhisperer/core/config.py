"""Centralized configuration.

Every environment variable read in this project lives here so that the runtime
contract is documented in exactly one place. Values are resolved at import
time, which keeps the rest of the codebase free of ``os.getenv`` calls.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from env_loader import load_project_env

load_project_env()

APP_NAME = "PaperWhisperer"
APP_VERSION = os.getenv("PAPERWHISPERER_VERSION", "0.9.0").strip() or "0.9.0"
APP_USER_AGENT = f"{APP_NAME}/{APP_VERSION}"

# paperwhisperer/core/config.py -> paperwhisperer/core -> paperwhisperer -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
CONTEXT_FOLDER = "context"

RUNTIME_FOLDERS = (UPLOAD_FOLDER, OUTPUT_FOLDER, CONTEXT_FOLDER)

MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB per uploaded document

STATIC_ICON_CACHE_CONTROL = "public, max-age=86400, immutable"
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}

SUPPORTED_EXTENSIONS = (".txt", ".pdf", ".docx", ".pptx")
ALLOWED_EXTENSIONS = frozenset(SUPPORTED_EXTENSIONS)
SUPPORTED_FILE_TYPES_TEXT = ", ".join(SUPPORTED_EXTENSIONS)

READING_QUEUE_LIMIT = 30


# --------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------

def parse_int_env(name: str, default: int, min_value: int = 1, max_value: int = 32) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(min_value, min(max_value, value))


def parse_bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def parse_bool_value(value, default: bool = False) -> bool:
    """Parse a user-supplied form/JSON value such as ``"true"`` or ``"1"``."""
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def clamp_int_value(value, default: int, min_value: int = 1, max_value: int = 32) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(min_value, min(max_value, parsed))


def resolve_api_key(explicit_key) -> str:
    """Prefer a request-scoped key, fall back to the server-level env var."""
    explicit_text = str(explicit_key or "").strip()
    if explicit_text:
        return explicit_text
    return os.getenv("OPENAI_API_KEY", "").strip()


# --------------------------------------------------------------------------
# LLM runtime
# --------------------------------------------------------------------------

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
OPENAI_REQUEST_TIMEOUT_SECONDS = parse_int_env("OPENAI_REQUEST_TIMEOUT_SECONDS", default=60,
    min_value=5, max_value=600)
OPENAI_MAX_RETRIES = parse_int_env("OPENAI_MAX_RETRIES", default=3, min_value=1, max_value=10)
MAX_LLM_CONCURRENCY = parse_int_env("OPENAI_MAX_CONCURRENCY", default=5, min_value=1, max_value=32)

#: Global cap on in-flight LLM requests across all sessions and workers.
LLM_REQUEST_SEMAPHORE = threading.BoundedSemaphore(MAX_LLM_CONCURRENCY)

LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 4000

#: Prompt windows, in characters, per analysis task.
DOCUMENT_WINDOW_BUDGET = {
    "summary_chunk": 4000,
    "quotes": 15000,
    "mindmap": 10000,
    "mermaid": 4000,
    "evaluation": 15000,
    "research_brief": 18000,
    "qa": 12000,
}

QA_HISTORY_BUDGET = 8000
QA_TURN_BUDGET = 2400
DOCUMENT_EXCERPT_LIMIT = 12000
SEARCH_REWRITE_CONTEXT_LIMIT = 4000

# --------------------------------------------------------------------------
# Paper search
# --------------------------------------------------------------------------

ARXIV_API_URL = "http://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

PAPER_SEARCH_PARALLEL = parse_bool_env("PAPER_SEARCH_PARALLEL", default=True)
PAPER_SEARCH_RESULT_LIMIT = parse_int_env("PAPER_SEARCH_RESULT_LIMIT", default=8, min_value=1,
    max_value=20)
RECOMMENDATION_RESULT_LIMIT = parse_int_env("RECOMMENDATION_RESULT_LIMIT", default=6, min_value=1,
    max_value=20)
PAPER_SEARCH_ENABLE_REWRITE = parse_bool_env("PAPER_SEARCH_ENABLE_REWRITE", default=True)
PAPER_SEARCH_REWRITE_MODEL = os.getenv("PAPER_SEARCH_REWRITE_MODEL", "").strip()

SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
SEMANTIC_SCHOLAR_TIMEOUT_SECONDS = parse_int_env("SEMANTIC_SCHOLAR_TIMEOUT_SECONDS", default=20,
    min_value=5, max_value=120)
SEMANTIC_SCHOLAR_MAX_RETRIES = parse_int_env("SEMANTIC_SCHOLAR_MAX_RETRIES", default=3, min_value=1,
    max_value=6)
SEMANTIC_SCHOLAR_RETRY_MAX_DELAY = parse_int_env("SEMANTIC_SCHOLAR_RETRY_MAX_DELAY", default=20,
    min_value=2, max_value=120)

ARXIV_TIMEOUT_SECONDS = parse_int_env("ARXIV_TIMEOUT_SECONDS", default=30, min_value=5,
    max_value=180)
ARXIV_MAX_RETRIES = parse_int_env("ARXIV_MAX_RETRIES", default=2, min_value=1, max_value=6)
ARXIV_RETRY_MAX_DELAY = parse_int_env("ARXIV_RETRY_MAX_DELAY", default=8, min_value=2, max_value=60)

# --------------------------------------------------------------------------
# Sessions
# --------------------------------------------------------------------------

SESSION_TTL_SECONDS = parse_int_env("SESSION_TTL_SECONDS", default=24 * 60 * 60, min_value=60,
    max_value=30 * 24 * 60 * 60)
SESSION_CLEANUP_INTERVAL_SECONDS = parse_int_env(
    "SESSION_CLEANUP_INTERVAL_SECONDS", default=10 * 60, min_value=60, max_value=24 * 60 * 60
)
SESSION_PERSIST_FULL_DOCUMENT = parse_bool_env("SESSION_PERSIST_FULL_DOCUMENT", default=False)

# --------------------------------------------------------------------------
# Remote document import
# --------------------------------------------------------------------------

REMOTE_IMPORT_TIMEOUT_SECONDS = parse_int_env("REMOTE_IMPORT_TIMEOUT_SECONDS", default=30,
    min_value=5, max_value=180)

PUBLIC_HOSTNAME_CACHE_TTL_SECONDS = 300
PUBLIC_HOSTNAME_CACHE_MAX_SIZE = 512

# --------------------------------------------------------------------------
# Server bootstrap
# --------------------------------------------------------------------------

FASTAPI_HOST = os.getenv("FASTAPI_HOST") or os.getenv("FLASK_HOST", "0.0.0.0")
FASTAPI_PORT = parse_int_env(
    "FASTAPI_PORT",
    default=parse_int_env("FLASK_PORT", default=5000, min_value=1, max_value=65535),
    min_value=1,
    max_value=65535,
)
FASTAPI_RELOAD = parse_bool_env("FASTAPI_RELOAD", default=parse_bool_env("FLASK_DEBUG",
    default=False))

TEMPLATES_DIR = PROJECT_ROOT / "templates"
STATIC_DIR = TEMPLATES_DIR / "static"
LOGO_PATH = PROJECT_ROOT / "logo.ico"


def ensure_runtime_folders() -> None:
    """Create the writable runtime folders the app depends on."""
    for folder in RUNTIME_FOLDERS:
        os.makedirs(folder, exist_ok=True)
