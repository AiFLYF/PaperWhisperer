"""Document upload and analysis routes."""

from __future__ import annotations

import logging
import urllib.error

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from paperwhisperer.analysis.orchestrator import DocumentAnalyzer
from paperwhisperer.analysis.service import analyze_saved_file, finalize_analysis_result
from paperwhisperer.api.deps import parse_json_object_request, pump_sync_events
from paperwhisperer.core import config
from paperwhisperer.core.errors import (
    build_error_payload,
    build_error_response,
    build_sse_event,
    build_sse_headers,
    describe_remote_http_error,
    describe_remote_url_error,
)
from paperwhisperer.core.text import (
    build_safe_upload_filename,
    build_session_id,
    build_unique_storage_path,
    close_upload_file_safely,
    is_allowed_file,
    remove_file_safely,
)
from paperwhisperer.documents.uploads import save_upload_file
from paperwhisperer.remote.download import download_remote_paper
from paperwhisperer.sessions.store import cleanup_expired_sessions

logger = logging.getLogger(__name__)

router = APIRouter()


def _validate_upload(file: UploadFile):
    """Return an error response for an unusable upload, or ``None`` if it is fine."""
    if file is None:
        return build_error_response("Please upload a file.", code="missing_file")
    if file.filename == "":
        return build_error_response("Please select a file.", code="missing_filename")
    if not is_allowed_file(file.filename):
        return build_error_response(
            f"Unsupported file type. Please upload one of: {config.SUPPORTED_FILE_TYPES_TEXT}",
            code="unsupported_file_type",
        )
    return None


@router.get("/logo.ico")
@router.get("/favicon.ico")
async def favicon():
    return FileResponse(
        config.LOGO_PATH,
        media_type="image/x-icon",
        headers={"Cache-Control": config.STATIC_ICON_CACHE_CONTROL},
    )


@router.post("/api/analyze")
async def analyze(
    file: UploadFile | None = File(None),
    api_key: str = Form(""),
    generate_mermaid: str | None = Form(None),
    generate_evaluation: str | None = Form(None),
    generate_research_brief: str | None = Form(None),
    session_id: str = Form(""),
):
    validation_error = _validate_upload(file)
    if validation_error is not None:
        return validation_error

    file_path = None
    try:
        cleanup_expired_sessions()
        original_filename = build_safe_upload_filename(file.filename)
        file_path = build_unique_storage_path(config.UPLOAD_FOLDER, original_filename)

        await save_upload_file(file, file_path, config.MAX_CONTENT_LENGTH)

        result = analyze_saved_file(
            file_path=file_path,
            original_filename=original_filename,
            api_key=api_key,
            generate_mermaid_bool=config.parse_bool_value(generate_mermaid, default=True),
            generate_evaluation_bool=config.parse_bool_value(generate_evaluation, default=True),
            session_id=session_id,
            generate_research_brief_bool=config.parse_bool_value(
                generate_research_brief, default=True
            ),
        )
        return result
    except ValueError as exc:
        return build_error_response(str(exc), code="invalid_request")
    except Exception as exc:
        logger.exception("Document analysis failed")
        return build_error_response(
            str(exc), status_code=500, code="document_analysis_failed"
        )
    finally:
        await close_upload_file_safely(file, "analysis upload file")
        remove_file_safely(file_path, "analyzed upload file")


@router.post("/api/analyze/stream")
async def analyze_stream(
    file: UploadFile | None = File(None),
    api_key: str = Form(""),
    generate_mermaid: str | None = Form(None),
    generate_evaluation: str | None = Form(None),
    generate_research_brief: str | None = Form(None),
    session_id: str = Form(""),
):
    validation_error = _validate_upload(file)
    if validation_error is not None:
        return validation_error

    cleanup_expired_sessions()
    resolved_api_key = config.resolve_api_key(api_key)
    if not resolved_api_key:
        return build_error_response(
            "API key is required. Provide api_key or set OPENAI_API_KEY.",
            code="missing_api_key",
        )

    original_filename = build_safe_upload_filename(file.filename)
    file_path = build_unique_storage_path(config.UPLOAD_FOLDER, original_filename)
    generate_mermaid_bool = config.parse_bool_value(generate_mermaid, default=True)
    generate_evaluation_bool = config.parse_bool_value(generate_evaluation, default=True)
    generate_research_brief_bool = config.parse_bool_value(generate_research_brief, default=True)

    try:
        await save_upload_file(file, file_path, config.MAX_CONTENT_LENGTH)
    except ValueError as exc:
        return build_error_response(str(exc), code="invalid_request")
    except Exception as exc:
        logger.exception("Streaming analyze upload failed")
        return build_error_response(
            str(exc), status_code=500, code="streaming_upload_failed"
        )
    finally:
        await close_upload_file_safely(file, "streaming analysis upload file")

    def produce():
        """Blocking generator: runs on a worker thread, never on the loop."""
        analyzer = DocumentAnalyzer(resolved_api_key)
        safe_session_id = build_session_id(session_id)
        yield {
            "type": "start",
            "data": {"session_id": safe_session_id, "source_filename": original_filename},
        }

        final_result = None
        for event in analyzer.analyze_stream(
            file_path, generate_mermaid_bool, generate_evaluation_bool, generate_research_brief_bool
        ):
            if event["type"] == "section":
                yield {
                    "type": "section",
                    "data": {"name": event["name"], "section": event["section"]},
                }
            elif event["type"] == "done":
                final_result = event["result"]

        if final_result is None:
            raise RuntimeError("Analysis stream completed without a final result.")

        yield {
            "type": "done",
            "data": finalize_analysis_result(
                result=final_result,
                analyzer=analyzer,
                original_filename=original_filename,
                generate_evaluation_bool=generate_evaluation_bool,
                session_id=safe_session_id,
                generate_research_brief_bool=generate_research_brief_bool,
            ),
        }

    def encode(event: dict) -> str:
        return build_sse_event(event["type"], event["data"])

    async def event_stream():
        try:
            async for chunk in pump_sync_events(
                lambda: produce(),
                encode,
                cleanup=lambda: remove_file_safely(file_path, "streamed analyzed upload file"),
            ):
                yield chunk
        except Exception as exc:
            logger.exception("Streaming document analysis failed")
            yield build_sse_event(
                "error", build_error_payload(str(exc), code="streaming_analysis_failed")
            )
        finally:
            remove_file_safely(file_path, "streamed analyzed upload file")

    return StreamingResponse(event_stream(), media_type="text/event-stream",
            headers=build_sse_headers())


@router.post("/api/import-paper")
async def import_paper(request: Request):
    data, error_response = await parse_json_object_request(request)
    if error_response is not None:
        return error_response

    file_path = None
    try:
        cleanup_expired_sessions()
        title = str(data.get("title") or "").strip()
        url = str(data.get("url") or "").strip()
        pdf_url = str(data.get("pdf_url") or "").strip()
        if not pdf_url and not url:
            return build_error_response(
                "A downloadable paper URL is required.", code="missing_paper_url"
            )

        file_path, original_filename = download_remote_paper(
            title=title, pdf_url=pdf_url, url=url
        )
        result = analyze_saved_file(
            file_path=file_path,
            original_filename=original_filename,
            api_key=str(data.get("api_key") or ""),
            generate_mermaid_bool=config.parse_bool_value(data.get("generate_mermaid"),
                default=True),
            generate_evaluation_bool=config.parse_bool_value(data.get("generate_evaluation"),
                default=True),
            session_id=str(data.get("session_id") or ""),
            generate_research_brief_bool=config.parse_bool_value(
                data.get("generate_research_brief"), default=True
            ),
        )
        return result
    except ValueError as exc:
        return build_error_response(str(exc), code="invalid_request")
    except urllib.error.HTTPError as exc:
        logger.warning("Paper import download failed: %s", exc)
        return build_error_response(
            describe_remote_http_error(exc.code), code="paper_import_failed"
        )
    except urllib.error.URLError as exc:
        logger.warning("Paper import download failed: %s", exc)
        return build_error_response(
            describe_remote_url_error(getattr(exc, "reason", exc)), code="paper_import_failed"
        )
    except Exception as exc:
        logger.exception("Paper import failed")
        return build_error_response(str(exc), status_code=500, code="paper_import_failed")
    finally:
        remove_file_safely(file_path, "imported paper analysis file")
