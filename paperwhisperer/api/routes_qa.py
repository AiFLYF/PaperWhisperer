"""Follow-up question routes (non-streaming and SSE).

Both routes answer the same way, and the way is T1.7: retrieve, then answer from
what came back, and hand the caller the structure — claims, citations,
confidence — rather than a string it would have to take on trust. The two
routes differ only in how the answer reaches the browser, and they share one
production function so the two cannot drift.

Three decisions are worth stating, because each of them is a place where the
obvious implementation is the wrong one.

**The route does not compute confidence, and does not accept the model's.**
:func:`~paperwhisperer.analysis.orchestrator.AnswerResult.confidence` is derived
from the claims' evidence inside the analyzer. The route's only job is
serialising it. A route that re-derived the label would be a second
implementation of the release gate, and the two would disagree exactly when it
mattered.

**The streaming route's ``done`` event carries the whole result, not just the
answer text.** The browser already replaces whatever it rendered from the deltas
with ``payload.answer`` on ``done``, so a ``done`` carrying only the string
would drop the citations on the floor for every streaming answer — the feature
would work on the non-streaming endpoint and silently not work on the one
people actually use.

**The analyzer is constructed with the session's document rather than handed
one.** The session persists text, not a parsed document, so this is the path
where the reconstructed-from-text fallback runs and where a citation's locator
is a character offset instead of a page. That is stated on the response as
``notes`` rather than left for a reader to discover, because a locator that
resolves to page 7 and a locator that resolves to character 4,100 are different
kinds of citation and a UI that renders them identically is lying about one of
them.
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from paperwhisperer.analysis.orchestrator import (
    CONFIDENCE_INSUFFICIENT,
    AnswerResult,
    DocumentAnalyzer,
    get_store_bundle,
)
from paperwhisperer.analysis.service import record_qa_turn
from paperwhisperer.api.deps import parse_json_object_request, pump_sync_events
from paperwhisperer.core import config
from paperwhisperer.core.errors import (
    build_error_payload,
    build_error_response,
    build_sse_event,
    build_sse_headers,
)
from paperwhisperer.core.text import redact_secrets
from paperwhisperer.ingestion.schema import ParsedDocument
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


def _registered_document(stores, document_id: str) -> ParsedDocument | None:
    """The canonical document this session was built from, or ``None``.

    :func:`build_qa_analyzer` resolves this and hands the result to the
    analyzer as ``qa_document``, which is the whole of the fix for the 404 on
    every citation click. See that function for why the session's own
    ``document_id`` is the right thing to retrieve over.

    ``None`` for every reason there is no such document, and they are not
    distinguished here: a session predating this, a document deleted from the
    store, a deployment that never registered one. They share one honest
    outcome — the caller falls back to the text-derived document, which is what
    it did before this existed and which the analyzer's own ``warnings`` already
    describe. Guessing between them would mean inventing a document.

    ``record.document`` is checked for *elements*, not merely for presence:
    ``upsert_document`` backfills an empty ``ParsedDocument`` for a document
    whose payload carried none, and handing that to the retriever would index a
    document with no text and answer every question from an empty scope.
    """
    if not str(document_id or "").strip():
        return None
    try:
        record = stores.metadata.get_document(str(document_id).strip())
    except Exception:  # noqa: BLE001 - a store that cannot answer is "no document"
        logger.warning("读取会话文档失败，Q&A 将退回纯文本重建文档。", exc_info=True)
        return None
    document = getattr(record, "document", None) if record is not None else None
    if document is None or not getattr(document, "elements", None):
        return None
    return document


def build_qa_analyzer(
    api_key: str, document_content: str, *, document_id: str = ""
) -> DocumentAnalyzer:
    """An analyzer wired to retrieve over this session's document.

    The store bundle is attached after construction rather than passed to
    ``__init__`` for two reasons. It is an *optional* collaborator — the
    analyzer falls back to the process-wide bundle when it is absent, and a
    test may inject its own — and assigning it keeps the constructor's required
    argument the only thing a caller has to know. It is also the same style the
    class already uses for ``document_content`` and ``qa_document``: build
    plainly, then attach what this particular turn needs.

    Resolving the bundle here rather than inside the analyzer is deliberate:
    it is the same bundle ``routes_evidence`` resolves, and a document indexed
    by one and a citation opened from the other has to be the *same* document,
    or the chip opens onto a document the store has never heard of.

    No embedder is passed. That is a deliberate default rather than an
    oversight: an embedding call per chunk per question costs money and adds
    latency, and a Q&A turn is still answerable from the lexical leg. The
    retriever records the dense leg's abstention as a degradation, so the fact
    is visible rather than inferred. A deployment that has indexed vectors and
    wants them used sets ``analyzer.embedder``.

    **Which document Q&A retrieves over, and why it is the registered one.**
    A session persists two things about the upload: the flat
    ``document_content`` the analysis prompts consumed, and the ``document_id``
    the analysis result published. They are *not* two interchangeable
    spellings of one upload, and only one of them can own a citation.

    ``document_content`` is a projection. Flattening the parse into prose drops
    the page boundaries, the bounding boxes, the heading tree and the tables, so
    the document rebuilt from it (:func:`parsed_document_from_text`) carries
    ``txt_offset`` locators and nothing else. A citation into that resolves to
    a character range: no page, no highlight, no page image. The registered
    document carries ``pdf_page`` locators and ``BoundingBox``es, and is the
    only one the evidence panel can render.

    They also have different ids, and that is the defect this line fixes. The
    rebuilt document's id is a content hash of the flat text
    (``doc_<sha256(text)>``) while the analysis published the parse's own id, so
    the ``ev_*`` ids the retriever minted were content addresses over *the
    rebuilt document's chunks* — and the evidence route, correctly scoped to
    the document named in the URL, had no candidate to match them against.
    Every citation a reader could click answered ``404 evidence_not_found``,
    in a product whose pitch is "click a citation and see the passage".
    Widening the resolver's candidate set would have "fixed" that by letting
    one document's citation resolve inside another, which is the
    cross-document leak the route exists to prevent; the mis-addressing was on
    this side of the seam.

    So the retrieved document is the one this session actually names. When it is
    present, ``qa_document`` is set and the analyzer prefers it over
    re-deriving anything (``DocumentAnalyzer._qa_document`` checks
    ``qa_document`` first), which also means retrieval indexes *that* document
    and the citations carry real locators. When it is absent — no id, a reaped
    document, a deployment that never registered one — the previous behaviour
    stands, and the analyzer's own ``warnings`` say in as many words that the
    structure is rebuilt from text. That degraded path is why
    ``routes_v1.ask_about_document`` already had to set ``qa_document`` by hand:
    this is the same correction, applied where the id was available and unused.
    """
    analyzer = DocumentAnalyzer(api_key)
    stores = get_store_bundle()
    analyzer.store_bundle = stores
    analyzer.document_content = document_content
    qa_document = _registered_document(stores, document_id)
    if qa_document is not None:
        analyzer.qa_document = qa_document
    return analyzer


def as_answer_result(value) -> AnswerResult:
    """The analyzer's return, narrowed to the one type this route serialises.

    ``DocumentAnalyzer.answer_question`` returns an :class:`AnswerResult`. A
    plain string is accepted anyway and reported as ``insufficient_evidence``
    with no claims, and that is not a courtesy to a caller who forgot to
    migrate — it is the only honest reading of what a bare string is. A string
    carries no evidence behind it, so a confidence of ``supported`` would be
    the route vouching for something it never checked.

    One caller paying for that is a test double or a downstream analyzer
    subclass; the alternative is a 500 on every question, which is a worse way
    to learn about a contract mismatch.
    """
    if isinstance(value, AnswerResult):
        return value
    return AnswerResult(
        answer=str(value or ""),
        confidence=CONFIDENCE_INSUFFICIENT,
    )


def _qa_response(result: AnswerResult) -> dict:
    """The wire shape for an answer. One function, so both routes agree.

    ``answer`` stays at the top level because the session transcript, the
    export preview and the pre-T1.7 clients all read it from there. The rest is
    additive: a client that has never heard of ``claims`` renders exactly what
    it rendered before, and one that has heard of them gets the release gate.
    """
    return result.to_dict()


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
        analyzer = build_qa_analyzer(
            values["resolved_api_key"],
            get_session_document_content(session_payload),
            document_id=str(session_payload.get("document_id") or ""),
        )

        started_at = time.time()
        result = as_answer_result(analyzer.answer_question(
            values["question"],
            history=session_payload.get("qa_history", []),
            answer_mode=values["answer_mode"],
        ))
        elapsed = time.time() - started_at

        record_qa_turn(session_payload, values["question"], result.answer, values["answer_mode"])
        # The stored turn carries the citations and the confidence, not just the
        # prose. Without them a later turn's history — which is the only record
        # of what was established — cannot say which claims were ever verified,
        # and the export preview silently loses the evidence it is supposed to
        # carry into a reviewer's notes.
        _attach_turn_evidence(session_payload, result)
        write_session_payload(safe_session_id, session_payload)
        logger.info(
            "Q&A completed in %.1fs for session %s (confidence=%s, claims=%d)",
            elapsed, safe_session_id, result.confidence, len(result.claims),
        )
        return _qa_response(result)
    except PermissionError as exc:
        return build_error_response(
            redact_secrets(str(exc)), status_code=403, code="invalid_session_token"
        )
    except ValueError as exc:
        return build_error_response(redact_secrets(str(exc)), code="invalid_request")
    except Exception as exc:
        logger.exception("Question answering failed")
        return build_error_response(
            redact_secrets(str(exc)), status_code=500, code="question_answering_failed"
        )


def _attach_turn_evidence(session_payload: dict, result: AnswerResult) -> None:
    """Write the result's structure onto the turn just recorded.

    Keyed off the last element rather than an index passed in, so it cannot
    attach to a different turn if the recording call ever changes shape. A
    missing ``qa_history`` leaves it a no-op rather than raising: a session file
    that has lost its history is already degraded, and a Q&A turn should not be
    the thing that turns a lost transcript into a 500.
    """
    history = session_payload.get("qa_history")
    if not isinstance(history, list) or not history:
        return
    last = history[-1]
    if not isinstance(last, dict):
        return
    last["confidence"] = result.confidence
    last["citations"] = [item.to_dict() for item in result.citations]
    last["claims"] = [item.to_dict() for item in result.claims]


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
        return build_error_response(
            redact_secrets(str(exc)), status_code=403, code="invalid_session_token"
        )
    except ValueError as exc:
        return build_error_response(redact_secrets(str(exc)), code="invalid_request")

    def produce():
        """Blocking generator driven from a worker thread by ``pump_sync_events``."""
        analyzer = build_qa_analyzer(
            values["resolved_api_key"],
            get_session_document_content(session_payload),
            document_id=str(session_payload.get("document_id") or ""),
        )
        yield {"type": "start", "data": {"session_id": safe_session_id}}

        deltas: list[str] = []
        result: AnswerResult | None = None
        for item in analyzer.stream_answer_question(
            values["question"],
            history=session_payload.get("qa_history", []),
            answer_mode=values["answer_mode"],
        ):
            # The generator yields text deltas and then the finished
            # ``AnswerResult``. Distinguishing them by type rather than by
            # position is what lets a change to the number of deltas — which is
            # a rendering detail, and now a derived one — not silently turn the
            # structure into a paragraph of JSON at the end of the stream.
            if isinstance(item, AnswerResult):
                result = item
                continue
            deltas.append(item)
            yield {"type": "delta", "data": {"text": item}}

        # A stream that produced no ``AnswerResult`` is a stream from an
        # analyzer that does not implement the T1.7 contract. The text is
        # already on the reader's screen, so the answer is the text and the
        # confidence is the only honest one available for it: nothing was
        # verified, so nothing can be called supported.
        result = as_answer_result(result if result is not None else "".join(deltas))

        record_qa_turn(session_payload, values["question"], result.answer, values["answer_mode"])
        _attach_turn_evidence(session_payload, result)
        write_session_payload(safe_session_id, session_payload)
        # The whole result, and not just ``answer``. The front end already
        # replaces whatever it rendered from the deltas with ``payload.answer``
        # on ``done``; a payload carrying only the string would drop the
        # citations for every streaming answer, so the feature would work on
        # the non-streaming endpoint and silently not work on the one the
        # browser actually uses. ``answer`` is in ``_qa_response`` and the
        # deltas reassemble to exactly that, so a reader never watches one
        # answer and is then handed another.
        yield {"type": "done", "data": _qa_response(result)}

    def encode(event: dict) -> str:
        return build_sse_event(event["type"], event["data"])

    async def event_stream():
        try:
            async for chunk in pump_sync_events(lambda: produce(), encode):
                yield chunk
        except Exception as exc:
            logger.exception("Streaming question answering failed")
            yield build_sse_event(
                "error", build_error_payload(
                    redact_secrets(str(exc)), code="streaming_question_failed"
                )
            )

    return StreamingResponse(
        event_stream(), media_type="text/event-stream", headers=build_sse_headers()
    )
