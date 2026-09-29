"""PaperWhisperer web entrypoint.

All application logic lives in the ``paperwhisperer`` package; this module
exists to keep ``python web_app.py`` and ``uvicorn web_app:app`` working.

Legacy top-level names are re-exported so existing scripts and tooling that
imported them from here keep working.
"""

from __future__ import annotations

import logging

import uvicorn

from paperwhisperer.app import app  # noqa: F401  (re-exported for uvicorn)
from paperwhisperer.core import config
from paperwhisperer.core.config import (  # noqa: F401  (re-exported)
    ALLOWED_EXTENSIONS,
    APP_NAME,
    APP_USER_AGENT,
    APP_VERSION,
    MAX_CONTENT_LENGTH,
    PAPER_SEARCH_ENABLE_REWRITE,
    PAPER_SEARCH_RESULT_LIMIT,
    PAPER_SEARCH_REWRITE_MODEL,
    RECOMMENDATION_RESULT_LIMIT,
    SECURITY_HEADERS,
    SESSION_PERSIST_FULL_DOCUMENT,
    SUPPORTED_EXTENSIONS,
    SUPPORTED_FILE_TYPES_TEXT,
    clamp_int_value,
    parse_bool_env,
    parse_bool_value,
    parse_int_env,
    resolve_api_key,
)
from paperwhisperer.core.errors import (  # noqa: F401  (re-exported)
    build_error_payload,
    build_error_response,
    build_sse_event,
    build_sse_headers,
    describe_llm_status_code,
    describe_remote_http_error,
    describe_remote_url_error,
    looks_like_html_response,
)
from paperwhisperer.core.text import (  # noqa: F401  (re-exported)
    build_document_excerpt,
    build_safe_upload_filename,
    build_session_id,
    build_unique_storage_path,
    clean_extracted_text,
    compact_text,
    extract_json_object,
    extract_message_text,
    is_allowed_file,
    normalize_author_list,
    normalize_next_actions,
    normalize_text_items,
    now_iso,
    parse_year,
    remove_file_safely,
    secure_filename,
    trim_text_for_log,
)
from paperwhisperer.documents.loader import DocumentLoader, TextChunker  # noqa: F401
from paperwhisperer.remote.ssrf import (  # noqa: F401  (re-exported)
    is_public_http_url,
    is_public_ip_address,
    resolve_public_hostname,
)
from paperwhisperer.search.normalize import (  # noqa: F401  (re-exported)
    deduplicate_papers,
    normalize_paper_record,
)
from paperwhisperer.search.papers import (  # noqa: F401  (re-exported)
    search_arxiv_papers,
    search_papers,
    search_semantic_scholar_papers,
)
from paperwhisperer.sessions.store import (  # noqa: F401  (re-exported)
    build_session_payload,
    cleanup_expired_sessions,
    generate_session_token,
    get_session_document_content,
    hash_session_token,
    load_session_payload,
    load_validated_session,
    validate_session_token,
    write_session_payload,
)

__all__ = ["app", "main"]


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "web_app:app",
        host=config.FASTAPI_HOST,
        port=config.FASTAPI_PORT,
        reload=config.FASTAPI_RELOAD,
    )


if __name__ == "__main__":
    main()
