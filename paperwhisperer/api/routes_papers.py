"""Paper search, reading queue, recommendation and download routes."""

from __future__ import annotations

import logging
import mimetypes
import urllib.error
import urllib.parse

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from paperwhisperer.analysis.orchestrator import DocumentAnalyzer
from paperwhisperer.api.deps import parse_json_object_request
from paperwhisperer.core import config
from paperwhisperer.core.errors import (
    build_error_response,
    describe_remote_http_error,
    describe_remote_url_error,
    now_iso,
)
from paperwhisperer.core.text import compact_text
from paperwhisperer.remote.download import iter_remote_file_chunks, stream_remote_paper
from paperwhisperer.search.normalize import normalize_paper_collection
from paperwhisperer.search.papers import search_papers
from paperwhisperer.sessions.store import (
    build_document_excerpt,
    cleanup_expired_sessions,
    get_session_document_content,
    load_validated_session,
    write_session_payload,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/download-paper")
async def download_paper(url: str = "", pdf_url: str = "", title: str = ""):
    """Proxy a remote paper download so the browser never hits the origin directly."""
    try:
        response, file_name, content_type, initial_chunk = stream_remote_paper(
            title=title, pdf_url=pdf_url, url=url
        )
        media_type = (
            content_type or mimetypes.guess_type(file_name)[0] or "application/octet-stream"
        )
        quoted_name = urllib.parse.quote(file_name)
        headers = {
            "Content-Disposition": (
                f'attachment; filename="{file_name}"; filename*=UTF-8\'\'{quoted_name}'
            ),
            "X-Content-Type-Options": "nosniff",
        }
        return StreamingResponse(
            iter_remote_file_chunks(response, config.MAX_CONTENT_LENGTH, initial_chunk),
            media_type=media_type,
            headers=headers,
        )
    except ValueError as exc:
        return build_error_response(str(exc), code="invalid_request")
    except urllib.error.HTTPError as exc:
        logger.warning("Paper proxy download failed: %s", exc)
        return build_error_response(describe_remote_http_error(exc.code),
                code="paper_download_failed")
    except urllib.error.URLError as exc:
        logger.warning("Paper proxy download failed: %s", exc)
        return build_error_response(
            describe_remote_url_error(getattr(exc, "reason", exc)), code="paper_download_failed"
        )
    except Exception as exc:
        logger.exception("Paper proxy download failed")
        return build_error_response(str(exc), status_code=500, code="paper_download_failed")


@router.post("/api/search-papers")
async def search_papers_api(request: Request):
    data, error_response = await parse_json_object_request(request)
    if error_response is not None:
        return error_response

    cleanup_expired_sessions()
    query = str(data.get("query") or "").strip()
    if not query:
        return build_error_response("Please enter a search query.", code="missing_search_query")

    limit = data.get("limit") or config.PAPER_SEARCH_RESULT_LIMIT
    raw_session_id = str(data.get("session_id") or "").strip()
    session_token = str(data.get("session_token") or "")
    context_text = str(data.get("context_text") or "")

    try:
        rewrite_meta = {
            "original_query": query,
            "rewritten_query": query,
            "topics": [],
            "reason": "Direct search without AI rewriting.",
            "model": "",
        }
        resolved_api_key = config.resolve_api_key(data.get("api_key", ""))
        session_payload = None
        safe_session_id = ""

        if raw_session_id:
            safe_session_id, session_payload = load_validated_session(
                raw_session_id, session_token, require_token=True
            )

        if config.PAPER_SEARCH_ENABLE_REWRITE and resolved_api_key:
            analyzer = DocumentAnalyzer(resolved_api_key)
            rewrite_context = compact_text(context_text, limit=4000)
            if session_payload and not rewrite_context:
                rewrite_context = build_document_excerpt(
                    get_session_document_content(session_payload), limit=4000
                )
            rewrite_meta = analyzer.rewrite_search_query(query, context_text=rewrite_context)
        elif config.PAPER_SEARCH_ENABLE_REWRITE:
            rewrite_meta["reason"] = (
                "No API key available; used direct search without AI rewriting."
            )

        result = search_papers(rewrite_meta["rewritten_query"], limit)
        result["original_query"] = rewrite_meta.get("original_query", query)
        result["rewritten_query"] = rewrite_meta.get("rewritten_query", result["query"])
        result["topics"] = rewrite_meta.get("topics", [])
        result["reason"] = rewrite_meta.get("reason", "")
        result["rewrite_model"] = rewrite_meta.get("model", "")

        if session_payload:
            session_payload.setdefault("paper_search", {})
            session_payload["paper_search"]["last_query"] = result["rewritten_query"]
            session_payload["paper_search"]["last_results"] = result["items"]
            write_session_payload(safe_session_id, session_payload)
        return result
    except PermissionError as exc:
        return build_error_response(str(exc), status_code=403, code="invalid_session_token")
    except ValueError as exc:
        return build_error_response(str(exc), code="invalid_request")
    except Exception as exc:
        logger.exception("Paper search failed")
        return build_error_response(str(exc), status_code=500, code="paper_search_failed")


@router.post("/api/reading-queue")
async def save_reading_queue(request: Request):
    data, error_response = await parse_json_object_request(request)
    if error_response is not None:
        return error_response

    cleanup_expired_sessions()
    try:
        safe_session_id, session_payload = load_validated_session(
            data.get("session_id"), data.get("session_token"), require_token=True
        )
        reading_queue = normalize_paper_collection(
            data.get("items"), max_items=config.READING_QUEUE_LIMIT
        )
        session_payload.setdefault("paper_search", {})
        session_payload["paper_search"]["reading_queue"] = reading_queue
        write_session_payload(safe_session_id, session_payload)
        return {"items": reading_queue, "count": len(reading_queue)}
    except PermissionError as exc:
        return build_error_response(str(exc), status_code=403, code="invalid_session_token")
    except ValueError as exc:
        return build_error_response(str(exc), code="invalid_request")
    except Exception as exc:
        logger.exception("Reading queue save failed")
        return build_error_response(str(exc), status_code=500, code="reading_queue_save_failed")


@router.post("/api/recommend-papers")
async def recommend_papers_api(request: Request):
    data, error_response = await parse_json_object_request(request)
    if error_response is not None:
        return error_response

    cleanup_expired_sessions()
    raw_session_id = str(data.get("session_id") or "").strip()
    if not raw_session_id:
        return build_error_response("session_id is required.", code="missing_session_id")

    limit = data.get("limit") or config.RECOMMENDATION_RESULT_LIMIT
    resolved_api_key = config.resolve_api_key(data.get("api_key", ""))
    if not resolved_api_key:
        return build_error_response(
            "API key is required. Provide api_key or set OPENAI_API_KEY.",
            code="missing_api_key",
        )

    try:
        safe_session_id, session_payload = load_validated_session(
            raw_session_id, data.get("session_token"), require_token=True
        )
        analyzer = DocumentAnalyzer(resolved_api_key)
        result = analyzer.recommend_papers(
            get_session_document_content(session_payload), limit=limit
        )

        session_payload.setdefault("paper_search", {})
        session_payload["paper_search"]["last_recommendation"] = {
            "original_query": result.get("original_query", ""),
            "query": result.get("query", ""),
            "topics": result.get("topics", []),
            "reason": result.get("reason", ""),
            "rewrite_model": result.get("rewrite_model", ""),
            "items": result.get("items", []),
            "errors": result.get("errors", []),
            "generated_at": now_iso(),
        }
        write_session_payload(safe_session_id, session_payload)
        return result
    except PermissionError as exc:
        return build_error_response(str(exc), status_code=403, code="invalid_session_token")
    except ValueError as exc:
        return build_error_response(str(exc), code="invalid_request")
    except Exception as exc:
        logger.exception("Paper recommendation failed")
        return build_error_response(str(exc), status_code=500, code="paper_recommendation_failed")
