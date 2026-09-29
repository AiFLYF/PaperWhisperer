"""Follow-up question routes (non-streaming and SSE)."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from paperwhisperer.analysis.orchestrator import DocumentAnalyzer
from paperwhisperer.analysis.service import record_qa_turn
from paperwhisperer.api.deps import parse_json_object_request, pump_sync_events
from paperwhisperer.core import config
from paperwhisperer.core.errors import (
    build_error_payload,
    build_error_response,
    build_sse_event,
    build_sse_headers,
)
from paperwhisperer.llm.prompts import normalize_answer_mode
from paperwhisperer.sessions.store import (
    cleanup_expired_sessions,
    get_session_document_content,
    load_validated_session,
    write_session_payload,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _parse_ask_request(data: dict):
    """Validate the shared ask payload, returning ``(values, error_response)``."""
    question = str(data.get("question") or "").strip()
    if not question:
        return None, build_error_response("Please enter a question.", code="missing_question")

    resolved_api_key = config.resolve_api_key(data.get("api_key", ""))
    if not resolved_api_key:
        return None, build_error_response(
            "API key is required. Provide api_key or set OPENAI_API_KEY.",
            code="missing_api_key",
        )

    return {
        "question": question,
        "answer_mode": normalize_answer_mode(data.get("answer_mode")),
        "resolved_api_key": resolved_api_key,
    }, None


@router.post("/api/ask")
async def ask_question(request: Request):
    data, error_response = await parse_json_object_request(request)
    if error_response is not None:
        return error_response

    cleanup_expired_sessions()
    values, error_response = _parse_ask_request(data)
    if error_response is not None:
        return error_response

    try:
        safe_session_id, session_payload = load_validated_session(
            data.get("session_id"), data.get("session_token"), require_token=True
        )
        analyzer = DocumentAnalyzer(values["resolved_api_key"])
        analyzer.document_content = get_session_document_content(session_payload)

        started_at = time.time()
        answer = analyzer.answer_question(
            values["question"],
            history=session_payload.get("qa_history", []),
            answer_mode=values["answer_mode"],
        )
        elapsed = time.time() - started_at

        record_qa_turn(session_payload, values["question"], answer, values["answer_mode"])
        write_session_payload(safe_session_id, session_payload)
        logger.info("Q&A completed in %.1fs for session %s", elapsed, safe_session_id)
        return {"answer": answer}
    except PermissionError as exc:
        return build_error_response(str(exc), status_code=403, code="invalid_session_token")
    except ValueError as exc:
        return build_error_response(str(exc), code="invalid_request")
    except Exception as exc:
        logger.exception("Question answering failed")
        return build_error_response(str(exc), status_code=500, code="question_answering_failed")


@router.post("/api/ask/stream")
async def ask_question_stream(request: Request):
    data, error_response = await parse_json_object_request(request)
    if error_response is not None:
        return error_response

    cleanup_expired_sessions()
    values, error_response = _parse_ask_request(data)
    if error_response is not None:
        return error_response

    try:
        safe_session_id, session_payload = load_validated_session(
            data.get("session_id"), data.get("session_token"), require_token=True
        )
    except PermissionError as exc:
        return build_error_response(str(exc), status_code=403, code="invalid_session_token")
    except ValueError as exc:
        return build_error_response(str(exc), code="invalid_request")

    def produce():
        """Blocking generator driven from a worker thread by ``pump_sync_events``."""
        analyzer = DocumentAnalyzer(values["resolved_api_key"])
        analyzer.document_content = get_session_document_content(session_payload)
        yield {"type": "start", "data": {"session_id": safe_session_id}}

        full_answer_parts = []
        for chunk in analyzer.stream_answer_question(
            values["question"],
            history=session_payload.get("qa_history", []),
            answer_mode=values["answer_mode"],
        ):
            full_answer_parts.append(chunk)
            yield {"type": "delta", "data": {"text": chunk}}

        answer = "".join(full_answer_parts)
        record_qa_turn(session_payload, values["question"], answer, values["answer_mode"])
        write_session_payload(safe_session_id, session_payload)
        yield {"type": "done", "data": {"answer": answer}}

    def encode(event: dict) -> str:
        return build_sse_event(event["type"], event["data"])

    async def event_stream():
        try:
            async for chunk in pump_sync_events(lambda: produce(), encode):
                yield chunk
        except Exception as exc:
            logger.exception("Streaming question answering failed")
            yield build_sse_event(
                "error", build_error_payload(str(exc), code="streaming_question_failed")
            )

    return StreamingResponse(
        event_stream(), media_type="text/event-stream", headers=build_sse_headers()
    )
