"""Document upload and analysis routes.

The legacy surface: what the bundled browser calls, kept answering exactly as it
did. T2.4 **adds** a versioned ``/v1`` surface rather than replacing this one, so
nothing here was rewritten to be resource-shaped — a client that is this browser
would break.

**T2.3 moved the work out of these routes, and the wire format stayed put.**
``/api/analyze`` and ``/api/analyze/stream`` used to parse, analyse and write a
report inside one request, and their ``finally`` deleted the upload. They now
submit a job and follow it. The difference a client can see is one optional form
field:

* ``mode=async`` (or ``POST /api/jobs/analyze``) answers 202 with a job id in
  milliseconds. This is the mode the Phase 2 criterion "负载测试下 API worker 不被
  长任务占用" is measured against.
* the default waits for the job and returns the same body it always did.

The two are the *same* submission — :func:`~paperwhisperer.api.routes_jobs.
submit_analysis_job` — so there is one pipeline, one parse and one trail, and a
load test and a browser are driving identical work. The waiting is a poll of the
state machine rather than a call into the analyzer, so a waiting request occupies
a coroutine and not a worker thread.

**What survives the request is the object store, not ``UPLOAD_FOLDER``.** The
``finally`` still deletes the staging upload, because it always did; what a job
reads an hour later is the write-once, content-addressed copy in the
``original`` layer that T2.2 put there and that ``routes_v1`` already uses. The
seams for both are in :mod:`paperwhisperer.api.routes_jobs`.

**Suites that stub this module's analyzer need to move their stub.** The analyzer
is now constructed inside a job stage rather than inside a request, so a
``monkeypatch.setattr(routes_documents, "DocumentAnalyzer", …)`` no longer
intercepts anything. The stage imports its collaborators from
:mod:`paperwhisperer.analysis.orchestrator` and
:mod:`paperwhisperer.analysis.service` at call time, which is the seam to patch
instead; ``tests/test_jobs_api.py``'s ``stubbed_pipeline`` fixture is a worked
example. The name is still importable here so that a stale stub fails as an
assertion about the new architecture rather than as an ``AttributeError`` during
setup.

The one thing that did move earlier is the upload validation. It now lives in
:mod:`paperwhisperer.api.routes_v1` and is called from here, because the upload
edge belongs to the versioned surface and the codes
(``missing_file``/``missing_filename``/``unsupported_file_type``/
``invalid_upload_metadata``) are ones the front end branches on: two copies would
have to agree exactly to keep working, and the natural drift in a copy is a
different message under the same code. The checks themselves are unchanged in
substance — three cheap rejections first, then
:class:`~paperwhisperer.ingestion.contracts.UploadRequestContract` for the two
things they cannot see (a filename carrying a path, and a declared content type
that contradicts the extension) — and the contract still runs *last* so a
genuinely unsupported extension keeps producing the old, familiar code rather
than a generic contract refusal. ``test_contracts_integration.py`` is what holds
that wiring in place; the contract models themselves are tested in
``tests/test_ingestion_contracts.py``.

**An exception's own text is redacted before it reaches a browser.** Every
``str(exc)`` that becomes a response body goes through
:func:`~paperwhisperer.core.text.redact_secrets` first — including the export
failure below, which was an f-string rather than a ``str(exc)`` and so was missed
by the pass that found the rest. The text an upstream library puts in an
exception message is shown to whoever made the request — which for this product
means whoever uploaded the document — and a credential can be in an exception
message: a rejected API key, a presigned URL signature, a connection string. The
``code`` is unaffected, so a client still branches on what happened rather than on
the prose.
"""

from __future__ import annotations

import asyncio
import logging
import time
import urllib.error
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

#: T2.3 moved the analysis out of this module and into the job pipeline, so the
#: routes here no longer construct an analyzer. The name is deliberately still
#: importable from this module: several existing suites install their stub at this
#: seam, and deleting the name turns "this test needs updating for the new
#: architecture" into an ``AttributeError`` at setup that reads like a typo rather
#: than like the architectural change it actually is. Kept as an explicit re-export
#: so that the next reader is not left guessing whether it is still used.
from paperwhisperer.analysis.orchestrator import DocumentAnalyzer  # noqa: F401
from paperwhisperer.analysis.service import (
    render_docx_report,
    render_html_report,
    render_markdown_report,
)
from paperwhisperer.api.deps import parse_json_object_request
from paperwhisperer.api.routes_jobs import (
    ANALYSIS_STREAM_POLL_SECONDS,
    SYNC_INLINE_FALLBACK_GRACE_SECONDS,
    SYNC_WAIT_TIMEOUT_SECONDS,
    JobStore,
    analysis_pipeline,
    get_job_store,
    get_object_store,
    job_sections,
    run_job_if_idle,
    should_run_inline,
    store_upload_as_original,
    submit_analysis_job,
    submit_response,
    take_job_result,
    wait_for_job_async,
    wants_async,
)
from paperwhisperer.api.routes_v1 import validate_upload
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
    redact_secrets,
    remove_file_safely,
)
from paperwhisperer.documents.uploads import save_upload_file
from paperwhisperer.infra.object_store import ObjectStore
from paperwhisperer.remote.download import download_remote_paper
from paperwhisperer.sessions.store import cleanup_expired_sessions, load_validated_session

logger = logging.getLogger(__name__)

router = APIRouter()


def _validate_upload(file: UploadFile):
    """Return an error response for an unusable upload, or ``None`` if it is fine.

    Delegated to :func:`paperwhisperer.api.routes_v1.validate_upload`; see this
    module's docstring for why the check lives on the versioned surface and is
    called from here rather than being kept twice.
    """
    return validate_upload(file)


#: A cloud parse can end five ways, and collapsing them into one
#: ``document_analysis_failed`` leaves the reader unable to tell "top up your
#: quota" from "try again" from "an administrator refused this" — three
#: different actions, one sentence. The interface already has a distinct
#: presentation for each, keyed on these codes.
#:
#: One outcome, one code, one policy, one set of words — whether the code
#: arrives from the API directly or is derived here from a field on an
#: otherwise-successful payload. Minting a second code for the same condition
#: would start a race between two mechanisms the day a route begins emitting
#: it.
_CLOUD_OUTCOME_CODES = {
    "quota_exhausted": "mineru_quota_exhausted",
    "timed_out": "cloud_parse_timed_out",
    "blocked": "cloud_parse_blocked",
    "failed": "cloud_parse_failed",
}


def _cloud_failure_code(job, state: str) -> str:
    """The machine-readable reason a job failed.

    Prefers the cloud outcome when there is one, because it is the only value
    that distinguishes the cases. Falls back to the generic code when the
    outcome is absent (a local parse failed, or the job record predates the
    field), and to ``job_cancelled`` when the failure was a cancellation,
    which is not a failure at all.
    """
    if state == "cancelled":
        return "job_cancelled"
    outcome = str(getattr(job, "cloud_outcome", "") or "").strip().lower()
    return _CLOUD_OUTCOME_CODES.get(outcome, "document_analysis_failed")



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
    mode: str = Form(""),
    store: JobStore = Depends(get_job_store),
    objects: ObjectStore = Depends(get_object_store),
):
    """Analyse an upload — by submitting a job and waiting for it.

    **The work is not in this request any more.** T2.3 moved it into
    :mod:`paperwhisperer.infra.jobs`; this route now validates the upload, puts
    the bytes somewhere they will still exist after the response, writes one job
    row, and then *watches* that row. The parse and the model calls happen on a
    :class:`~paperwhisperer.infra.jobs.JobWorker` thread, and this coroutine
    sleeps while it waits.

    **Why it still waits, when waiting is what T2.3 is about.** The bundled
    browser posts here and renders the workspace from the response body; there is
    no client on the other side that knows what to do with a job id. Rewriting it
    into a 202 would break that browser, and T2.4's acceptance criterion is that
    the front end switches over without breaking. So the two modes differ only in
    *when the answer is written*:

    * ``mode=async`` (or ``POST /api/jobs/analyze``) returns a job id in
      milliseconds and occupies nothing. This is the mode the Phase 2 criterion
      "负载测试下 API worker 不被长任务占用" is measured against.
    * the default waits, and is the only difference between them — the same
      :func:`~paperwhisperer.api.routes_jobs.submit_analysis_job` call, the same
      pipeline, the same one parse.

    **What makes the wait safe is that it costs the process almost nothing.** The
    coroutine yields on every poll, so a hundred concurrent analyses do not
    occupy a hundred threads; and the job is durable, so a client that gives up
    mid-wait has not thrown the work away — it has an id to come back to.
    """
    validation_error = _validate_upload(file)
    if validation_error is not None:
        return validation_error

    file_path = None
    try:
        cleanup_expired_sessions()
        original_filename = build_safe_upload_filename(file.filename)
        file_path = build_unique_storage_path(config.UPLOAD_FOLDER, original_filename)

        # Read MAX_UPLOAD_BYTES live rather than the legacy MAX_CONTENT_LENGTH
        # alias: the alias is a name that only *looks* like the setting, so a
        # test or an operator that tightened MAX_UPLOAD_BYTES would be ignored
        # here and the ceiling would silently stay at the shipped default.
        await save_upload_file(file, file_path, config.MAX_UPLOAD_BYTES)

        # Before the submit, and before the ``finally`` below removes the
        # staging upload. The job reads the bytes from the object store's
        # ``original`` layer; without this the job would be queued against a
        # file that no longer exists the moment this request returns.
        original_key, _newly_written = store_upload_as_original(
            objects, file_path, file.content_type or ""
        )
    except ValueError as exc:
        return build_error_response(redact_secrets(str(exc)), code="invalid_request")
    except Exception as exc:
        logger.exception("Document analysis upload failed")
        return build_error_response(
            redact_secrets(str(exc)), status_code=500, code="document_analysis_failed"
        )
    finally:
        await close_upload_file_safely(file, "analysis upload file")
        remove_file_safely(file_path, "analyzed upload file")

    try:
        submitted, job = submit_analysis_job(
            store,
            original_key=original_key,
            original_filename=original_filename,
            content_type=file.content_type or "",
            api_key=api_key,
            session_id=build_session_id(session_id),
            generate_mermaid=generate_mermaid,
            generate_evaluation=generate_evaluation,
            generate_research_brief=generate_research_brief,
        )
    except Exception:  # noqa: BLE001 - a store outage is a 503, not a 500
        logger.exception("Submitting the analysis job failed")
        return build_error_response(
            "任务提交失败，请稍后重试。", status_code=503, code="job_store_unavailable"
        )

    if wants_async(mode):
        return submit_response(submitted, job)

    settled = await wait_for_job_async(
        store, str(submitted.job_id), pipeline=analysis_pipeline()
    )
    if not settled.terminal:
        # Out of patience, not out of work. Answering 202 with the job is the
        # same answer the async mode gives, and it is honest: the work is
        # genuinely still going and the client now has the id to watch it with.
        return JSONResponse(
            status_code=202, content=submit_response(submitted, settled).model_dump()
        )
    if settled.state in {"failed", "cancelled"}:
        code = _cloud_failure_code(settled, settled.state)
        return build_error_response(
            settled.error_message or "文档分析失败。", status_code=422, code=code
        )

    result = take_job_result(str(submitted.job_id))
    if not result:
        # The job finished, but in another process — so the payload carrying the
        # session *token* was never in this one, by design. The job still names
        # the session, so the client is told where the work went instead of being
        # handed an empty 200.
        return JSONResponse(
            status_code=202,
            content=submit_response(submitted, settled).model_dump(),
        )

    # The analysis, plus the job's identity. The two extra keys are additive:
    # a client reading only the analysis sees exactly what it saw before T2.3.
    return {**result, "job_id": str(submitted.job_id), "job_state": str(settled.state)}


@router.post("/api/analyze/stream")
async def analyze_stream(
    request: Request = None,
    file: UploadFile | None = File(None),
    api_key: str = Form(""),
    generate_mermaid: str | None = Form(None),
    generate_evaluation: str | None = Form(None),
    generate_research_brief: str | None = Form(None),
    session_id: str = Form(""),
    mode: str = Form(""),
    store: JobStore = Depends(get_job_store),
    objects: ObjectStore = Depends(get_object_store),
):
    """Stream an analysis by *following a job*, rather than by owning one.

    **The events are unchanged and that is the point.** The bundled browser
    consumes ``start``, ``section``, ``done`` and ``error`` from this route and
    knows nothing about jobs, so all four are still emitted with the same shapes.
    What changed is where they come from: ``section`` events are now read out of
    the running job's scratch as each section is produced, and ``done`` carries
    the result the job's analysing stage finished. The stream is a *view* of a
    durable job.

    **So a client that drops can come back.** The ``start`` event carries a
    ``job_id`` the browser is free to keep, and every state transition is
    numbered. A client that reconnects to ``GET /v1/jobs/{job_id}/stream`` with
    ``Last-Event-ID`` learns exactly what it missed, and a client that reconnects
    once the job is ``completed`` reads ``result.session_id`` off
    ``GET /v1/jobs/{job_id}`` and finds the finished analysis in the session it
    already holds a token for. Neither costs another model call.

    **The deltas are not replayed, and that is deliberate.** Section payloads are
    model output; replaying them is what makes a half-consumed content stream
    unrecoverable in the first place. The job keeps them in its scratch, so a
    *new* subscriber can read the finished set, but the guarantee is about state
    never being lost rather than about every byte being re-sent — which is the
    guarantee the exit criterion actually asks for ("浏览器刷新/SSE 断开不丢任务").

    ``mode=async`` short-circuits all of it and answers 202 with the job id,
    which is the mode a client that has its own UI for a queue should use.
    """
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

    try:
        await save_upload_file(file, file_path, config.MAX_UPLOAD_BYTES)
        # See ``/api/analyze``: the object-store copy is what the job will read,
        # and it is written before the staging upload is removed below.
        original_key, _newly_written = store_upload_as_original(
            objects, file_path, file.content_type or ""
        )
    except ValueError as exc:
        return build_error_response(redact_secrets(str(exc)), code="invalid_request")
    except Exception as exc:
        logger.exception("Streaming analyze upload failed")
        return build_error_response(
            redact_secrets(str(exc)), status_code=500, code="streaming_upload_failed"
        )
    finally:
        await close_upload_file_safely(file, "streaming analysis upload file")
        remove_file_safely(file_path, "streamed analyzed upload file")

    try:
        submitted, _job = submit_analysis_job(
            store,
            original_key=original_key,
            original_filename=original_filename,
            content_type=file.content_type or "",
            api_key=resolved_api_key,
            session_id=build_session_id(session_id),
            generate_mermaid=generate_mermaid,
            generate_evaluation=generate_evaluation,
            generate_research_brief=generate_research_brief,
        )
    except Exception:  # noqa: BLE001 - a store outage is a 503, not a 500
        logger.exception("Submitting the streaming analysis job failed")
        return build_error_response(
            "任务提交失败，请稍后重试。", status_code=503, code="job_store_unavailable"
        )

    job_id = str(submitted.job_id)
    if wants_async(mode):
        return JSONResponse(status_code=202, content=submit_response(
            submitted, submitted.job).model_dump())

    safe_session_id = build_session_id(session_id)

    async def event_stream():
        """Follow one job, and speak the event vocabulary the browser expects."""
        yield build_sse_event(
            "start",
            {
                "session_id": safe_session_id,
                "source_filename": original_filename,
                # The one addition to this event: enough to come back for.
                "job_id": job_id,
                "job_status_path": f"/v1/jobs/{job_id}",
                "job_stream_path": f"/v1/jobs/{job_id}/stream",
            },
        )

        seen_sections: set[str] = set()
        started_at = time.monotonic()
        deadline = started_at + SYNC_WAIT_TIMEOUT_SECONDS
        # The same safety net the synchronous submit has: if nothing has claimed
        # the job shortly after it was written, this stream claims and runs it
        # itself. Without it, a deployment with no background worker — or a
        # process started without its lifespan — would hold this connection open
        # emitting nothing until the timeout, which is the "spinner for ever" the
        # front end already has code to report as a failure.
        tried_fallback = False

        while True:
            job = store.get_job(job_id)
            if job is None:
                yield build_sse_event(
                    "error",
                    build_error_payload("任务不存在。", code="job_not_found"),
                )
                return

            for name, section in job_sections(job).items():
                if name in seen_sections:
                    continue
                seen_sections.add(name)
                yield build_sse_event("section", {"name": name, "section": section})

            if job.terminal:
                if job.state in {"failed", "cancelled"}:
                    code = _cloud_failure_code(job, job.state)
                    yield build_sse_event(
                        "error",
                        build_error_payload(
                            redact_secrets(job.error_message or "文档分析失败。"), code=code
                        ),
                    )
                    return
                result = take_job_result(job_id)
                if not result:
                    # Finished elsewhere. The browser cannot render what it does
                    # not have, so say where it is instead of streaming nothing —
                    # and the session id comes off the *job row*, which is the
                    # one thing that did survive the trip between processes.
                    yield build_sse_event(
                        "error",
                        build_error_payload(
                            "分析已完成，但结果由另一个进程生成；"
                            f"请通过任务 {job_id} 读取会话 "
                            f"{job.result.get('session_id') or ''}。",
                            code="analysis_result_not_local",
                        ),
                    )
                    return
                yield build_sse_event("done", result)
                return

            if request is not None and await request.is_disconnected():
                # The tab is gone. The job is not: it is on a worker thread, its
                # progress is in a table, and the id is in the ``start`` event
                # the client already read.
                return

            if not tried_fallback and should_run_inline(
                SYNC_INLINE_FALLBACK_GRACE_SECONDS
            ):
                tried_fallback = True
                await asyncio.to_thread(
                    run_job_if_idle, store, analysis_pipeline(), job_id
                )

            if time.monotonic() > deadline:
                yield build_sse_event(
                    "error",
                    build_error_payload(
                        f"等待任务超时，请通过 {job_id} 继续查看进度。",
                        code="analysis_wait_timeout",
                    ),
                )
                return
            await asyncio.sleep(ANALYSIS_STREAM_POLL_SECONDS)

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

        # Imported here for the same reason as in
        # :func:`paperwhisperer.api.routes_jobs._analyse_stage`: the analysis
        # package pulls in the store bundle, and binding these at module load
        # would build that bundle before a test had a chance to override it.
        #
        # This route used to call the legacy ``analyze_saved_file``. That left
        # the second user-facing way to analyze a paper — importing one by URL —
        # on the six-section generator, which records no ``paper_facts`` and no
        # ``analysis_run``, so the provenance the export report promises was
        # simply absent for every imported paper. The four collaborators below
        # are the same four the job stage runs, in the same order and for the
        # same reasons: one parse whose ``document_id`` everything downstream
        # reuses, an indexed retriever, a per-field evidence provider, and then
        # the fact layer itself, which records the run and calls the real
        # finaliser.
        from paperwhisperer.analysis.orchestrator import build_qa_retriever
        from paperwhisperer.analysis.paper_facts import make_evidence_provider
        from paperwhisperer.analysis.service import analyze_with_fact_layer
        from paperwhisperer.api.routes_evidence import get_store_bundle
        from paperwhisperer.documents.loader import DocumentLoader

        loaded = DocumentLoader.load_with_outcome(file_path)
        outcome = getattr(loaded, "outcome", None)
        document = getattr(outcome, "document", None)
        if document is None:
            # An unusable parse is a bad request, not a server fault, and this
            # route already maps ValueError onto ``invalid_request``.
            raise ValueError(
                (getattr(outcome, "error_message", "") or "")
                or "文档解析失败。"
            )

        retriever = build_qa_retriever(get_store_bundle(), document=document)
        evidence_provider = make_evidence_provider(retriever, document.document_id)

        result = analyze_with_fact_layer(
            file_path,
            original_filename,
            str(data.get("api_key") or ""),
            config.parse_bool_value(data.get("generate_mermaid"), default=True),
            config.parse_bool_value(data.get("generate_evaluation"), default=True),
            session_id=str(data.get("session_id") or ""),
            generate_research_brief_bool=config.parse_bool_value(
                data.get("generate_research_brief"), default=True
            ),
            document_id=document.document_id,
            evidence_provider=evidence_provider,
            store=get_store_bundle(),
            loaded_document=loaded,
        )
        return result
    except ValueError as exc:
        return build_error_response(redact_secrets(str(exc)), code="invalid_request")
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
        return build_error_response(
            redact_secrets(str(exc)), status_code=500, code="paper_import_failed"
        )
    finally:
        remove_file_safely(file_path, "imported paper analysis file")


#: Export formats this route will render. The three are not interchangeable:
#: markdown and html are what a reader can diff and a browser can open, and
#: docx is the one that survives being pasted into a document. All three carry
#: the same citation and provenance content, because an export that drops the
#: provenance in one format is an export nobody can trace.
EXPORT_FORMATS = {
    "markdown": ("text/markdown; charset=utf-8", ".md"),
    "html": ("text/html; charset=utf-8", ".html"),
    "docx": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".docx",
    ),
}


def _export_result_from_session(session_payload: dict) -> dict:
    """Rebuild the renderable result from a stored session.

    The session keeps a projection of the analysis, not the analysis object, so
    a field it does not carry simply does not appear in the export. That is the
    correct degradation: the renderers print "（未记录）" for a field they were
    not given, which is honest, rather than inventing one.
    """
    analysis = dict(session_payload.get("analysis") or {})
    for key in ("parse", "analysis_run", "citations", "claims"):
        if key in session_payload:
            analysis[key] = session_payload[key]
    return analysis


@router.get("/api/export/{session_id}")
async def export_session(
    session_id: str,
    request: Request,
    format: str = "markdown",
    session_token: str = "",
    app_base_url: str = "",
):
    """Render a stored analysis as markdown, HTML or DOCX.

    Every format is produced by ``analysis.service``, so the browser-download
    path and this route emit the same content. That matters for the release
    gate: the front end already refuses to put quotation marks around a
    degraded paraphrase, and an export generated by a second code path would
    be exactly where that discipline lapses.
    """
    cleanup_expired_sessions()
    chosen = str(format or "markdown").strip().lower()
    if chosen not in EXPORT_FORMATS:
        return build_error_response(
            f"不支持的导出格式：{chosen}。可用：{', '.join(sorted(EXPORT_FORMATS))}。",
            code="unsupported_export_format",
        )

    try:
        safe_session_id, session_payload = load_validated_session(
            session_id, session_token, require_token=True
        )
    except ValueError:
        return build_error_response("会话不存在或已过期。", code="session_not_found")
    except PermissionError:
        return build_error_response("会话令牌无效。", code="session_forbidden")

    result = _export_result_from_session(session_payload)
    original_filename = str(session_payload.get("source_filename") or "document")
    version = config.APP_VERSION
    base = app_base_url or getattr(config, "EXPORT_APP_BASE_URL", "")
    media_type, suffix = EXPORT_FORMATS[chosen]
    stem = original_filename.rsplit(".", 1)[0] or "analysis"

    try:
        if chosen == "docx":
            body = render_docx_report(
                result, original_filename, version, app_base_url=base
            )
        else:
            render = render_markdown_report if chosen == "markdown" else render_html_report
            body = render(result, original_filename, version, app_base_url=base)
    except Exception as exc:  # noqa: BLE001 - surfaced as a user-facing error
        logger.exception("Export rendering failed for session %s", safe_session_id)
        return build_error_response(
            f"导出失败：{redact_secrets(str(exc))}", code="export_render_failed"
        )

    filename = f"{stem}{suffix}"
    return Response(
        content=body,
        media_type=media_type,
        headers={
            # The filename is quoted so a name containing a comma or a space
            # does not truncate the header, and the encoding is declared
            # because non-ASCII filenames are common here.
            "Content-Disposition": (
                f"attachment; filename=\"{filename}\"; "
                f"filename*=UTF-8\'{quote(filename)}"
            ),
            "Cache-Control": "no-store",
        },
    )
