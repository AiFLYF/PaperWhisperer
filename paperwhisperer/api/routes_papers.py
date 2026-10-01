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
from paperwhisperer.core.text import compact_text, redact_secrets
from paperwhisperer.remote.download import iter_remote_file_chunks, stream_remote_paper
from paperwhisperer.search.identity import identity_from_record
from paperwhisperer.search.normalize import normalize_paper_collection
from paperwhisperer.search.papers import search_papers
from paperwhisperer.search.recommend import (
    RecommendationQuery,
    dataset_names_from_facts,
    default_recommendation_sources,
    recommend_papers,
)
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
        # MAX_UPLOAD_BYTES, read live: the proxy must enforce the same ceiling
        # the upload path does, and it must be the ceiling the operator
        # configured rather than the legacy alias's frozen-in-at-import copy.
        return StreamingResponse(
            iter_remote_file_chunks(response, config.MAX_UPLOAD_BYTES, initial_chunk),
            media_type=media_type,
            headers=headers,
        )
    except ValueError as exc:
        return build_error_response(redact_secrets(str(exc)), code="invalid_request")
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
        return build_error_response(
            redact_secrets(str(exc)), status_code=500, code="paper_download_failed"
        )


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
        return build_error_response(
            redact_secrets(str(exc)), status_code=403, code="invalid_session_token"
        )
    except ValueError as exc:
        return build_error_response(redact_secrets(str(exc)), code="invalid_request")
    except Exception as exc:
        logger.exception("Paper search failed")
        return build_error_response(
            redact_secrets(str(exc)), status_code=500, code="paper_search_failed"
        )


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
        return build_error_response(
            redact_secrets(str(exc)), status_code=403, code="invalid_session_token"
        )
    except ValueError as exc:
        return build_error_response(redact_secrets(str(exc)), code="invalid_request")
    except Exception as exc:
        logger.exception("Reading queue save failed")
        return build_error_response(
            redact_secrets(str(exc)), status_code=500, code="reading_queue_save_failed"
        )


@router.post("/api/paper-recommendations")
async def paper_recommendations_api(request: Request):
    """Multi-signal recommendations, each row carrying the signal that made it.

    Kept beside :func:`recommend_papers_api` rather than replacing it, and the
    reason is a persistence gap rather than caution: the session writer
    normalizes ``paper_search.last_recommendation.items`` through
    :func:`paperwhisperer.search.normalize.normalize_paper_collection`, which
    rebuilds each row from a fixed key set. Every ``reasons`` list would be
    stripped on the next load, so a stored copy of this payload would keep the
    papers and lose the entire point of them. Until that writer learns the new
    fields, the response is returned and not persisted.

    **The seed paper is supplied by the client, and that is not an
    oversight.** Four of the five signals are walks of a citation graph, and a
    graph walk needs an address. The analysed *document* has a ``document_id``
    minted by the parser, not a DOI or a provider id, and nothing in the
    session payload carries a bibliographic identity for it — so a
    document-scoped recommendation can only answer the topical signal, and
    would have to report the other four as unavailable with that reason, every
    time, to every user. Asking the client for the record it is looking at (it
    has one: search results and the reading queue both carry the full
    identifier set since T3.1) turns that permanent absence into a request
    parameter. A client with no ``paper`` still gets a response, with the
    citation signals correctly marked unavailable.

    **An API key is optional here, unlike on the legacy route.** The citation
    and relatedness signals need no model, and requiring a key to see them
    would mean that a user without one is shown a "no related papers" answer to
    a question the system had the data to answer. Without a key the topical
    signal falls back to searching the seed's own title and says so, rather
    than pretending an expansion happened.
    """
    data, error_response = await parse_json_object_request(request)
    if error_response is not None:
        return error_response

    cleanup_expired_sessions()
    raw_session_id = str(data.get("session_id") or "").strip()
    if not raw_session_id:
        return build_error_response("session_id is required.", code="missing_session_id")

    limit = data.get("limit") or config.RECOMMENDATION_RESULT_LIMIT

    try:
        _safe_session_id, session_payload = load_validated_session(
            raw_session_id, data.get("session_token"), require_token=True
        )
        seed_record = data.get("paper") if isinstance(data.get("paper"), dict) else {}
        identity = identity_from_record(seed_record)
        if not identity.key and not seed_record.get("title"):
            return build_error_response(
                "需要指定一篇论文（paper）才能做多信号推荐："
                "请从检索结果或阅读队列中选择一篇论文后重试。",
                code="missing_seed_paper",
            )

        # The LLM rewrite is the same one the search route uses; it is what
        # "LLM query expansion" means here, and a second prompt asking the same
        # question would be a second answer to compare.
        rewrite_meta = {
            "rewritten_query": "",
            "topics": [],
            "reason": "未配置 API Key，按论文标题检索。",
            "model": "",
        }
        resolved_api_key = config.resolve_api_key(data.get("api_key", ""))
        if resolved_api_key:
            try:
                excerpt = compact_text(
                    get_session_document_content(session_payload),
                    limit=config.SEARCH_REWRITE_CONTEXT_LIMIT,
                )
                seed_hint = identity.title or "this paper"
                rewrite_meta = DocumentAnalyzer(resolved_api_key).rewrite_search_query(
                    query=f"Find related and follow-up papers for: {seed_hint}",
                    context_text=excerpt,
                )
            except Exception as exc:  # noqa: BLE001 - the graph signals can still answer
                logger.warning("Recommendation query expansion failed: %s", exc)
                rewrite_meta["reason"] = f"查询扩展失败（{exc}），已按论文标题检索。"

        datasets, datasets_declared = _resolve_datasets(data, session_payload)
        query = RecommendationQuery(
            identity=identity,
            expanded_query=str(rewrite_meta.get("rewritten_query") or ""),
            topics=tuple(str(topic) for topic in rewrite_meta.get("topics") or ()),
            datasets=datasets,
            dataset_names_are_declared=datasets_declared,
            limit=config.clamp_int_value(
                limit,
                config.RECOMMENDATION_RESULT_LIMIT,
                min_value=1,
                max_value=config.RECOMMENDATION_RESULT_LIMIT,
            ),
        )
        result = recommend_papers(query, default_recommendation_sources(search_fn=search_papers))
        # Kept so a client written against the legacy route keeps working; the
        # per-signal `reasons` are the reason this route exists.
        result["original_query"] = rewrite_meta.get("original_query", "")
        result["query"] = rewrite_meta.get("rewritten_query", "")
        result["topics"] = rewrite_meta.get("topics", [])
        result["reason"] = rewrite_meta.get("reason", "")
        result["rewrite_model"] = rewrite_meta.get("model", "")
        return result
    except PermissionError as exc:
        return build_error_response(
            redact_secrets(str(exc)), status_code=403, code="invalid_session_token"
        )
    except ValueError as exc:
        return build_error_response(redact_secrets(str(exc)), code="invalid_request")
    except Exception as exc:
        logger.exception("Paper recommendation failed")
        return build_error_response(
            redact_secrets(str(exc)), status_code=500, code="paper_recommendation_failed"
        )


def _resolve_datasets(data: dict, session_payload: dict) -> tuple[tuple[str, ...], bool]:
    """The dataset names to search, and whether they are authoritative.

    A client-supplied ``datasets`` list is treated as declared names. A fact
    layer payload — from the request, or from the session if T1.8's persistence
    gap is ever closed — is scanned by
    :func:`paperwhisperer.search.recommend.dataset_names_from_facts`, which
    reads names out of model prose and is therefore explicitly *not* declared.
    The distinction travels to the response as the ``confidence`` on every
    same-dataset reason, so the UI can say "同名数据集" rather than
    "同数据集" when that is all we proved.

    Today the session branch returns nothing: ``build_session_payload`` copies
    a fixed key set out of ``analysis`` and ``paper_facts`` is not in it. The
    branch is written because the alternative is a route that can only ever see
    datasets a client happened to have, and the gap is reported rather than
    worked around with topic similarity.
    """
    supplied = [str(item).strip() for item in data.get("datasets") or [] if str(item).strip()]
    if supplied:
        return tuple(supplied), True

    facts = data.get("paper_facts")
    if not isinstance(facts, dict):
        facts = (session_payload.get("analysis") or {}).get("paper_facts")
    names = dataset_names_from_facts(facts)
    return names, False


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
        return build_error_response(
            redact_secrets(str(exc)), status_code=403, code="invalid_session_token"
        )
    except ValueError as exc:
        return build_error_response(redact_secrets(str(exc)), code="invalid_request")
    except Exception as exc:
        logger.exception("Paper recommendation failed")
        return build_error_response(
            redact_secrets(str(exc)), status_code=500, code="paper_recommendation_failed"
        )
