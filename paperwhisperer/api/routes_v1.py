"""The ``/v1`` document surface: upload, register, metadata, pages, reparse, delete.

**Why this module exists next to the legacy one rather than replacing it.** T2.4's
acceptance criterion is "现有前端无破坏性切换", which is a statement about
addition. ``/api/analyze``, ``/api/ask``, ``/api/recommend-papers`` and their
siblings are still what the shipped browser calls, they still return what it
expects, and nothing in this module is on their path. Where the two need the same
behaviour they *share* it: :func:`validate_upload` is defined here and
``routes_documents`` calls it, the Q&A producer is the one ``routes_qa`` runs, and
the search is the one ``routes_papers`` runs. A second copy of any of those is a
second answer to a question the product has already answered once.

**Why the surface is resource-shaped and versioned.** ``/v1`` is a promise that
the shape of a response will not change under a client. It is kept honest by
three things and not by discipline: every route declares a ``response_model``
whose fields are the fields the route actually returns; every failure carries a
stable ``code`` beside its Chinese prose; and ``tests/test_api_v1.py`` pins the
field names of each resource, because a field name is the part a client is
allowed to depend on. Chinese is for the person reading the response; ``code`` is
the API.

**The document surface is parse and index, not analysis.** Creating a document
stores the original, parses it, chunks it and makes it retrievable. Running the
model over it is a job kind that T2.3 owns, and no route here takes an ``api_key``
for it — a document resource that quietly billed for a model call would be a
resource nobody could reason about.

**Every parse here is a job, and the job is run in the request.** The submit →
run → settle sequence uses
:class:`~paperwhisperer.infra.jobs.JobPipeline` and
:class:`~paperwhisperer.infra.jobs.JobWorker`, so the states a client watches
over SSE are the state machine's own and the trail is written by the same code
that will write it when a background worker exists. Draining inline rather than
handing the job to a thread is deliberate and temporary: the browser reparse
control (``templates/static/js/parseStatus.js``) renders the parse report from the
response body and then says "Reparse finished", so returning a queued job today
would be a message about work nobody is doing. Moving the drain behind
``JobWorker.start`` changes no wire format — the ``job`` object in the response is
the same one either way.

**Reparse and the privacy gate.** A cloud parser means the document leaves the
machine, so an explicit cloud request is checked against
:func:`~paperwhisperer.ingestion.orchestrator.evaluate_cloud_privacy_gate` —
which *is* ``mineru_cloud.evaluate_privacy_gate``, read through the orchestrator's
import-tolerant wrapper, and is the only privacy gate in the repository. There is
no second one here. The check is a refusal, not a second policy: the orchestrator
would consult the same gate during the default chain anyway, and calling it one
layer up turns "the cloud stage was silently skipped" into a 403 the caller can
act on. The refusal's ``code`` is the gate's own ``reason_code``, so
``privacy_mode_local_only`` and ``consent_not_granted`` are distinguishable
without a client reading a sentence.

**The cloud request is a policy, not a guarantee, and the response says which.**
``parser: "mineru_cloud"`` means *this document may be uploaded, and the cloud may
win*. It does not mean a local parse is thrown away to pay for one: the
orchestrator's escalation only reaches the cloud when the local chain produced
nothing usable, because uploading a document you already parsed correctly is the
cost-and-privacy mistake this product is built against. So the response carries
the real facts — ``parse.used_cloud``, ``parse.cloud_attempted``,
``parse.touched_network`` — plus ``cloud_requested`` and a ``notice`` saying, in
Chinese, that no upload happened. Inventing a `used_cloud: true` to match the
request would be the one failure mode the provenance fields exist to prevent.

**An exception's own text is redacted before it reaches a browser.** Every
``str(exc)`` that becomes a response body goes through
:func:`~paperwhisperer.core.text.redact_secrets` first. The text an upstream
library puts in an exception message is shown to whoever made the request — which
for this product means whoever uploaded the document — and a credential can be in
an exception message: a rejected API key, a presigned URL signature, a connection
string. The ``code`` is unaffected, so a client still branches on what happened
rather than on the prose.

**The pages listing, and the 501 that is not this module's.** ``GET
/v1/documents/{id}/pages`` is new here; ``GET /v1/documents/{id}/pages/{page}/image``
belongs to ``routes_evidence`` and answers 501 with a reason code, on purpose.
The listing therefore advertises that route's URL *and* its unavailability — the
status, the code and the reason — so a client can tell "this build cannot rasterise
pages" from "this page has no highlight" without issuing a request it already
knows the answer to.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, Depends, File, Form, Request, Response, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from paperwhisperer.analysis.orchestrator import (
    AnswerResult,
    index_document_for_retrieval,
)
from paperwhisperer.analysis.service import record_qa_turn
from paperwhisperer.api.deps import parse_json_object_request, pump_sync_events
from paperwhisperer.api.routes_evidence import (
    PAGE_RENDER_CODE,
    PAGE_RENDER_MESSAGE,
    PAGE_RENDER_STATUS,
    get_store_bundle,
)
from paperwhisperer.api.routes_evidence import _is_well_formed as evidence_id_guard
from paperwhisperer.api.routes_jobs import JobState, error_responses, get_job_store, job_state
from paperwhisperer.api.routes_qa import (
    _attach_turn_evidence,
    as_answer_result,
    build_qa_analyzer,
)
from paperwhisperer.core import config
from paperwhisperer.core.errors import (
    build_error_payload,
    build_error_response,
    build_sse_event,
    build_sse_headers,
)
from paperwhisperer.core.text import (
    build_safe_upload_filename,
    build_unique_storage_path,
    close_upload_file_safely,
    is_allowed_file,
    redact_secrets,
    remove_file_safely,
)
from paperwhisperer.documents.uploads import save_upload_file
from paperwhisperer.infra.jobs import (
    STATE_PARSING,
    Job,
    JobPipeline,
    JobStep,
    JobStore,
    JobWorker,
    StepFailed,
)
from paperwhisperer.infra.object_store import (
    LAYER_ORIGINAL,
    ObjectNotFound,
    ObjectRef,
    ObjectStore,
    build_object_store,
    digest_key,
)
from paperwhisperer.infra.stores import StoreBundle
from paperwhisperer.infra.stores.interfaces import DocumentRecord
from paperwhisperer.ingestion.contracts import UploadRequestContract
from paperwhisperer.ingestion.gateway import (
    PARSE_PARTIAL,
    ParseOptions,
    ParserRouter,
    build_default_parsers,
    compute_file_sha256,
)
from paperwhisperer.ingestion.orchestrator import (
    ParseOutcome,
    evaluate_cloud_privacy_gate,
    parse_document,
)
from paperwhisperer.ingestion.schema import ParsedDocument
from paperwhisperer.llm.prompts import normalize_answer_mode
from paperwhisperer.rag.evidence import page_renderer_name
from paperwhisperer.search.papers import search_papers
from paperwhisperer.sessions.store import (
    cleanup_expired_sessions,
    load_validated_session,
    write_session_payload,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# --------------------------------------------------------------------------
# Object store
# --------------------------------------------------------------------------


#: The process-wide object store, built on first call.
#:
#: A module global for the reason ``routes_evidence`` keeps one: the store owns a
#: directory tree, and building it per request would re-resolve the root and
#: re-check the layers on every poll. Tests reach it through
#: ``dependency_overrides``, so no test writes to a developer's real data
#: directory.
_OBJECTS: ObjectStore | None = None


def get_object_store() -> ObjectStore:
    """The shared object store, built on first call.

    :func:`~paperwhisperer.infra.object_store.build_object_store` is the only
    place a backend is chosen, so this defers to it: a reparse has no business
    knowing whether the ``original`` layer is a directory or a bucket.
    """
    global _OBJECTS
    if _OBJECTS is None:
        _OBJECTS = build_object_store(config)
    return _OBJECTS


# --------------------------------------------------------------------------
# Error table
# --------------------------------------------------------------------------


#: One Chinese sentence and one machine code per failure on the document surface.
#: Kept as a table, beside the module that raises them, for the reason
#: ``routes_evidence`` keeps one: the message a client reads and the code it
#: branches on are written down together so they cannot drift apart.
DOCUMENT_ERRORS: dict[str, tuple[int, str, str]] = {
    "malformed_id": (400, "文档 ID 格式不合法。", "malformed_id"),
    "missing_file": (400, "请选择要上传的文件。", "missing_file"),
    "missing_filename": (400, "请选择要上传的文件。", "missing_filename"),
    "unsupported_file_type": (
        400,
        f"不支持的文件类型。请上传以下格式之一：{config.SUPPORTED_FILE_TYPES_TEXT}。",
        "unsupported_file_type",
    ),
    "invalid_upload_metadata": (400, "上传文件的元数据不合法。", "invalid_upload_metadata"),
    "upload_rejected": (400, "上传的文件未通过校验。", "upload_rejected"),
    "document_not_found": (404, "文档不存在。", "document_not_found"),
    "original_unavailable": (
        409,
        "该文档的原始文件不在对象存储中，无法重新解析。"
        "请通过 POST /v1/documents 重新上传后再试。",
        "original_unavailable",
    ),
    "document_parse_failed": (422, "文档解析失败。", "document_parse_failed"),
    "document_busy": (409, "该文档仍有未结束的任务，请先等待或取消。", "document_busy"),
    "document_exists": (409, "同名文档已存在。", "document_exists"),
    "invalid_document_payload": (
        400,
        "提交的解析结果不符合 ParsedDocument 结构。",
        "invalid_document_payload",
    ),
    "unknown_parser": (
        400,
        "未知的解析器选项。可用值：docling、docling_ocr、mineru_cloud；"
        "留空表示使用默认解析链。",
        "unknown_parser",
    ),
    "parser_unavailable": (409, "所选解析器在当前部署中不可用。", "parser_unavailable"),
    "missing_question": (400, "请输入问题。", "missing_question"),
    "missing_api_key": (
        400,
        "缺少 API Key。请在请求中提供 api_key，或设置 OPENAI_API_KEY。",
        "missing_api_key",
    ),
    "missing_search_query": (400, "请输入检索关键词。", "missing_search_query"),
    "search_failed": (500, "论文检索失败。", "search_failed"),
    "session_not_found": (404, "会话不存在或已过期。", "session_not_found"),
    "session_token_invalid": (403, "会话令牌无效。", "session_token_invalid"),
    "storage_unavailable": (503, "存储不可用。", "storage_unavailable"),
    "job_cancelled": (409, "任务已被取消。", "job_cancelled"),
    "document_upload_failed": (500, "上传文档失败。", "document_upload_failed"),
    "reparse_failed": (500, "重新解析失败。", "reparse_failed"),
    # The two refusals below carry the *privacy gate's own* reason codes as the
    # code of the response, rather than a wrapper code plus a prose reason. The
    # reason a cloud parse is refused is load-bearing for the user: one is fixed by
    # an administrator changing DOCUMENT_PRIVACY_MODE, the other by the user
    # granting consent. A client that could only see "forbidden" would send people
    # to change the setting that was never the problem.
    "privacy_mode_local_only": (
        403,
        "云端解析未启用：当前文档隐私模式为 local_only，禁止任何文档出网。"
        "如需启用请联系管理员将 DOCUMENT_PRIVACY_MODE 设置为 allow_cloud。",
        "privacy_mode_local_only",
    ),
    "consent_not_granted": (
        403,
        "云端解析未获授权：隐私模式允许云端，但尚未获得您的显式同意。"
        "请在设置中开启“允许将文档上传到云端解析服务”。",
        "consent_not_granted",
    ),
}


class _Refusal(Exception):
    """A validation failure raised before any work, carrying an error-table key.

    Raised rather than returned because every caller of the resolution helpers
    wraps a block of them and a returned sentinel would have to be threaded
    through each one; the key is what travels, so the route builds the response
    from the same table the rest of the surface uses.
    """

    def __init__(self, key: str, message: str = "") -> None:
        self.key = key
        self.message = message
        super().__init__(key)


def document_error_response(key: str, message: str = ""):
    """Build the response for one row of :data:`DOCUMENT_ERRORS`.

    ``message`` overrides the table's sentence without changing the code, for the
    one case that needs it: a refusal whose *cause* differs (a parser that is not
    installed versus a parser the rollback flag has switched off) while the
    client's correct response is identical.
    """
    status, default_message, code = DOCUMENT_ERRORS[key]
    return build_error_response(message or default_message, status_code=status, code=code)


def refusal_response(refusal: _Refusal):
    """Turn a :class:`_Refusal` into the response its key names.

    One place, because three call sites raise one and each of them used to build
    its own response — and a lookup that returns 403 for an unknown key is a
    silent way to turn a 404 into a 403. An unknown key is therefore answered as
    500 with the key in the code, which is the truth: a code nobody declared is a
    bug in this module, not a decision about the caller's request.
    """
    if refusal.key in DOCUMENT_ERRORS:
        return document_error_response(refusal.key, refusal.message)
    logger.error("undeclared v1 error key: %s", refusal.key)
    return build_error_response(
        refusal.message or "服务端内部错误。", status_code=500, code=refusal.key
    )


# --------------------------------------------------------------------------
# Upload validation (shared with routes_documents)
# --------------------------------------------------------------------------


def describe_upload_contract_error(exc: ValidationError) -> str:
    """Turn a contract rejection into one sentence naming the offending field.

    Moved here from ``routes_documents`` because the upload edge belongs to the
    versioned surface and two copies of this sentence would be two answers to
    "what was wrong with my upload". The field name is the whole point: a client
    that sent ``notes.txt`` with a declared type of ``image/png`` has two
    independent claims to fix.
    """
    first = exc.errors()[0]
    field = ".".join(str(part) for part in first.get("loc", ())) or "upload"
    reason = str(first.get("msg", "invalid value")).strip()
    return f"Invalid upload ({field}): {reason}"


def validate_upload(file: UploadFile | None):
    """Return an error response for an unusable upload, or ``None`` if it is fine.

    The three checks are kept exactly as ``routes_documents`` has them, with the
    codes the front end already branches on, and the contract model is layered on
    top for what they cannot see: a filename carrying a path, and a declared
    content type that contradicts the extension. The contract runs last so a
    genuinely unsupported extension still produces the old, familiar code rather
    than a generic contract refusal.
    """
    if file is None:
        return document_error_response("missing_file")
    if file.filename == "":
        return document_error_response("missing_filename")
    if not is_allowed_file(file.filename):
        return document_error_response("unsupported_file_type")
    try:
        UploadRequestContract(filename=file.filename, media_type=file.content_type or "")
    except ValidationError as exc:
        logger.info("rejected upload: %s", describe_upload_contract_error(exc))
        return document_error_response(
            "invalid_upload_metadata", describe_upload_contract_error(exc)
        )
    return None


# --------------------------------------------------------------------------
# Frozen shapes
# --------------------------------------------------------------------------


#: The key set of a parse report, in the order it is emitted. Declared once so
#: the projection from a live :class:`~paperwhisperer.ingestion.orchestrator.
#: ParseOutcome` and the projection from a *stored* one cannot disagree: a
#: document parsed before a field existed still answers with that field, empty,
#: rather than with a key a client then has to branch on.
PARSE_REPORT_KEYS: tuple[str, ...] = (
    "status",
    "parsed_by",
    "media_type",
    "options_hash",
    "cache_hit",
    "fallback_used",
    "used_cloud",
    "cloud_attempted",
    "touched_network",
    "summary",
    "warnings",
    "error_message",
    "quality",
    "attempts",
    "cloud_refusal",
)

_PROVENANCE_NOT_RECORDED = (
    "该文档的解析来源未被记录：它不是由本服务的 /v1 上传或重新解析接口写入的。"
)


class ParseReport(BaseModel):
    """How one document was read, and whether it left this machine.

    The field names here are the ones ``templates/static/js/parseStatus.js``
    already reads (``status``, ``parsed_by``, ``used_cloud``, ``cloud_attempted``,
    ``touched_network``, ``cache_hit``, ``fallback_used``, ``media_type``,
    ``warnings``, ``summary``, ``quality``), frozen so that a document reported by
    ``/v1`` and a document reported by ``/api/analyze`` look the same to the same
    code.

    ``used_cloud`` means *this* text came from the cloud. ``cloud_attempted`` means
    a request was made and paid for. ``touched_network`` means any attempt was
    allowed to reach the network. Conflating them is how a deployment ends up
    quietly billing for a result it discarded, so all three are reported.
    """

    status: str = ""
    parsed_by: str = ""
    media_type: str = ""
    options_hash: str = ""
    cache_hit: bool = False
    fallback_used: bool = False
    used_cloud: bool = False
    cloud_attempted: bool = False
    touched_network: bool = False
    summary: str = ""
    warnings: list[str] = Field(default_factory=list)
    error_message: str = ""
    quality: dict[str, Any] | None = None
    attempts: list[dict[str, Any]] = Field(default_factory=list)
    cloud_refusal: dict[str, Any] | None = None


class DocumentLinks(BaseModel):
    """The other resources hanging off this document.

    ``page_image`` is a template with a ``{page}`` hole in it, and it is always
    present even though it answers 501: the point of naming the URL is that a
    client can be written against it today and be correct when the renderer lands.
    """

    self: str
    metadata: str
    pages: str
    page_image: str
    qa: str
    reparse: str


class DocumentResource(BaseModel):
    """One document, as this surface reports it.

    Identity and counters, not content. The canonical document is a large object
    and a reader gets it through the evidence API, which is where a single
    addressable element is served; a resource representation that inlined the
    whole parse would make every list request expensive for no reader who wanted
    it. ``parse`` is here because "who read this and did it leave my machine" is
    one question a client asks constantly and should not cost a second request.
    """

    document_id: str
    source_filename: str
    media_type: str
    sha256: str
    page_count: int
    status: str
    parser_name: str
    parser_version: str
    options_hash: str
    index_version: str
    element_count: int
    chunk_count: int
    char_count: int
    #: Whether a reparse can run for this document. ``False`` means the original
    #: bytes are not in the object store, and a reparse will answer 409 rather
    #: than pretend to have a file to re-read.
    original_available: bool
    created_at: str
    updated_at: str
    version: int
    parse: ParseReport
    links: DocumentLinks


class DocumentList(BaseModel):
    """A page of documents, newest first."""

    items: list[DocumentResource]
    count: int
    limit: int
    offset: int


class PageDescriptor(BaseModel):
    """One page of a parsed document, described by what the parser found on it.

    Counts rather than text. A page listing that returned the page's text would
    be a second way to read the document, and this surface is about geometry:
    which pages exist, how much landed on each, and what kind of content it was.
    """

    page: int
    element_count: int
    char_count: int
    has_table: bool
    has_formula: bool
    has_figure: bool


class PageImageAvailability(BaseModel):
    """The state of the page-image route, declared before it is called.

    ``available`` is the *build's* capability, read from
    :func:`~paperwhisperer.rag.evidence.page_renderer_name` — the one function
    that answers "can this process rasterise a page", and the same one the route
    acts on, so the listing and the route cannot be two opinions. It is necessary
    but not sufficient: the document's original bytes must also be in the object
    store, which the document resource reports as ``original_available``. The
    client needs both, and this is where the first one lives.

    The ``unavailable_*`` fields are what a 501 from that route carries, declared
    here so a client can tell "this build cannot rasterise pages" from "this page
    has no highlight" without making a request whose answer it already has — and
    so the day a renderer lands, the client is already pointing at the right URL.
    """

    url_template: str
    #: The rasteriser this process can use, or ``"none"``. A name rather than a
    #: boolean because a deployment that swaps renderer is reporting a fact a
    #: client may want in a support ticket.
    renderer: str
    available: bool
    unavailable_status: int
    unavailable_code: str
    unavailable_message: str


class PageList(BaseModel):
    """The pages a document has, and what is on them.

    ``locator_kind`` is the format's addressing scheme (``pdf_page``, ``slide``,
    ``docx_element``, ``txt_offset``). It is reported because ``page`` below is a
    *page* only for ``pdf_page``: for a deck, ``page`` is a slide number, and for
    a plain text file it is not a page at all and the list is empty. A client that
    read ``pages[0].page`` as a page number without looking would be wrong on two
    of the four formats, and the field is what stops it.
    """

    document_id: str
    media_type: str
    page_count: int
    locator_kind: str
    page_image: PageImageAvailability
    pages: list[PageDescriptor]


class DocumentCreatedResponse(BaseModel):
    """The answer to a create: the document, and the job that produced it."""

    document: DocumentResource
    #: The job that produced the document, in the same shape
    #: ``GET /v1/jobs/{id}`` returns — so a client holding this response already
    #: holds everything it would get by following the id.
    job: JobState
    #: ``False`` when the bytes were already in the object store, so nothing was
    #: written and the upload's modification time is unchanged. The same
    #: observable a re-parse uses to prove it did not touch the user's file.
    original_stored: bool
    notice: str = ""


class DocumentRegisteredResponse(BaseModel):
    """The answer to a register: the document, and how much was indexed."""

    document: DocumentResource
    chunk_count: int
    notice: str = ""


class ReparseResponse(BaseModel):
    """The answer to a reparse, and everything a client needs to render it.

    ``cloud_requested`` plus ``notice`` exist because "the user asked for the
    cloud and the cloud was not used" is a real outcome with a real explanation,
    and a client that only read ``parse.used_cloud`` would have to guess why.
    """

    document_id: str
    document: DocumentResource
    parse: ParseReport
    job: JobState
    cloud_requested: bool
    notice: str = ""


class DocumentDeletedResponse(BaseModel):
    """The answer to a delete: what went, and what else went with it."""

    document_id: str
    deleted: bool
    original_deleted: bool


class PaperRecord(BaseModel):
    """One paper, as the search surface reports it.

    ``extra="allow"`` because the field set belongs to ``search/`` — T3.1 is still
    adding identifiers to it, and dropping a field a provider started sending
    would be a silent regression in a surface whose whole promise is stability.
    The *declared* fields are the ones
    :data:`paperwhisperer.sessions.store.PAPER_RECORD_CLEANED_KEYS` names — the set
    the reading-queue writer rebuilds a paper record from — so this model and that
    registry are the same list, and a test asserts they are. The practical
    consequence is the useful one: a paper found through this surface and saved
    to the reading queue keeps every field it arrived with.
    """

    model_config = ConfigDict(extra="allow")

    source: str = ""
    paper_id: str = ""
    title: str = ""
    abstract: str = ""
    authors: list[dict[str, Any]] = Field(default_factory=list)
    year: int | None = None
    venue: str = ""
    url: str = ""
    pdf_url: str = ""
    saved_at: str = ""
    doi: str = ""
    arxiv_id: str = ""
    s2_paper_id: str = ""
    openalex_id: str = ""
    identity_key: str = ""


class PaperSearchResponse(BaseModel):
    """Search results, plus what the pass decided and what it declined to decide.

    ``possible_duplicates`` are pairs that look like one paper and could not be
    *proven* to be one, and which were therefore kept with both rows visible. A
    wrong merge destroys a paper the user can no longer find; a duplicate costs
    one glance. A client that renders the field turns it into a "these may be the
    same" affordance, and one that ignores it still shows both rows.
    """

    query: str
    original_query: str
    count: int
    items: list[PaperRecord]
    possible_duplicates: list[dict[str, Any]] = Field(default_factory=list)
    merged_duplicates: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    #: Why the query is what it is: rewritten, not rewritten, or the expansion
    #: failed and the raw query was used. A client that rendered "rewritten" over a
    #: failed expansion would be explaining results the user never asked for.
    reason: str = ""


# --------------------------------------------------------------------------
# Projections
# --------------------------------------------------------------------------


def _empty_parse_report(summary: str) -> dict[str, Any]:
    """A parse report with every key present and nothing claimed.

    The honest form of an unrecorded fact. The alternative — omitting the keys —
    is what this module exists to prevent: a client would then have to branch on
    whether ``used_cloud`` is *absent* or ``false``, and the two mean "we did not
    record it" and "it stayed on this machine".
    """
    return {
        "status": "",
        "parsed_by": "",
        "media_type": "",
        "options_hash": "",
        "cache_hit": False,
        "fallback_used": False,
        "used_cloud": False,
        "cloud_attempted": False,
        "touched_network": False,
        "summary": summary,
        "warnings": [],
        "error_message": "",
        "quality": None,
        "attempts": [],
        "cloud_refusal": None,
    }


def parse_report(outcome: ParseOutcome | None) -> dict[str, Any]:
    """The wire form of a live parse outcome.

    :meth:`ParseOutcome.to_dict` with the document removed and ``summary``
    added: the whole parsed document is not part of a status report, and the
    one-sentence summary is what a log line, a CLI or a reader's status card
    needs. Everything else is passed through unchanged, so the field names are the
    ingestion layer's and not a re-spelling of them.
    """
    if outcome is None:
        return _empty_parse_report(_PROVENANCE_NOT_RECORDED)
    payload = {key: value for key, value in outcome.to_dict().items() if key != "document"}
    payload["summary"] = outcome.summary_line()
    report = _empty_parse_report(payload.get("summary") or "")
    for key in PARSE_REPORT_KEYS:
        if key in payload:
            report[key] = payload[key]
    report["warnings"] = list(payload.get("warnings") or [])
    report["attempts"] = list(payload.get("attempts") or [])
    return report


def stored_parse_report(record: DocumentRecord) -> dict[str, Any]:
    """The parse report a stored document carries, or an explicit absence.

    The provenance is written into the record's ``metadata`` under ``parse`` when
    the document is created here. A document adopted from elsewhere has no such
    block, and this reports every key empty with a ``summary`` that says so —
    never a dict of confident-looking falses that a client would render as
    "parsed locally, never uploaded".

    The second fallback is for the case where the block existed and the row has
    since been rewritten: the document's own columns still say which parser read
    it, so they are reported and the ``summary`` says the detail is gone. Blanket
    "not recorded" would be true about the block and misleading about the
    document, and the reader who needs the parser name is the reader who matters
    when a citation looks wrong.
    """
    stored = record.metadata.get("parse") if isinstance(record.metadata, dict) else None
    if isinstance(stored, dict) and stored:
        report = _empty_parse_report(_PROVENANCE_NOT_RECORDED)
        for key in PARSE_REPORT_KEYS:
            if key in stored:
                report[key] = stored[key]
        report["warnings"] = list(report.get("warnings") or [])
        report["attempts"] = list(report.get("attempts") or [])
        return report

    if record.parser_name:
        report = _empty_parse_report(
            f"该文档的详细解析来源未保留；仅能从文档记录中读出解析器 "
            f"{record.parser_name}。"
        )
        report["parsed_by"] = str(record.parser_name)
        report["media_type"] = str(record.media_type)
        report["options_hash"] = str(record.options_hash)
        return report

    return _empty_parse_report(_PROVENANCE_NOT_RECORDED)


def document_links(document_id: str) -> dict[str, str]:
    """The sibling URLs of one document."""
    base = f"/v1/documents/{document_id}"
    return {
        "self": base,
        "metadata": f"{base}/metadata",
        "pages": f"{base}/pages",
        "page_image": f"{base}/pages/{{page}}/image",
        "qa": f"{base}/qa",
        "reparse": f"{base}/reparse",
    }


def document_resource(
    record: DocumentRecord,
    stores: StoreBundle,
    objects: ObjectStore,
    *,
    chunk_count: int | None = None,
) -> dict[str, Any]:
    """The wire form of one document.

    ``parse`` is read from the record's stored provenance, so the resource and
    ``GET /v1/documents/{id}/metadata`` are two projections of one fact and
    cannot disagree. ``original_available`` is a ``stat`` against the object
    store, not a flag: it is what decides whether a reparse will have a file to
    read, and a flag that could be stale would be a reparse that fails on the
    user's word rather than on the store's answer.
    """
    document = record.document
    resolved_chunks = (
        chunk_count
        if chunk_count is not None
        else len(stores.metadata.list_chunks(record.document_id))
    )
    return {
        "document_id": str(record.document_id),
        "source_filename": str(record.source_filename),
        "media_type": str(record.media_type),
        "sha256": str(record.sha256),
        "page_count": int(record.page_count),
        "status": str(record.status),
        "parser_name": str(record.parser_name),
        "parser_version": str(record.parser_version),
        "options_hash": str(record.options_hash),
        "index_version": str(record.index_version),
        "element_count": len(document.elements) if document is not None else 0,
        "chunk_count": int(resolved_chunks),
        "char_count": int(document.char_count) if document is not None else 0,
        "original_available": has_stored_original(objects, record),
        "created_at": str(record.created_at),
        "updated_at": str(record.updated_at),
        "version": int(record.version),
        "parse": stored_parse_report(record),
        "links": document_links(record.document_id),
    }


def page_descriptors(record: DocumentRecord) -> tuple[str, list[dict[str, Any]]]:
    """Group a document's elements by page, and name the addressing scheme.

    The returned kind is the one that decides what ``page`` means. For a deck the
    numbers are slides; for a text file there are no pages and the list is empty,
    which is reported rather than faked with a single "page 1" — a page listing
    that invented a page would be read as "this document has one page of text" by
    a client that has no way to tell the difference.
    """
    document = record.document
    if document is None or not document.elements:
        return "", []

    kind_counts: dict[str, int] = {}
    for element in document.elements:
        kind = str(element.locator.kind or "")
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
    locator_kind = max(kind_counts.items(), key=lambda item: item[1])[0]

    pages: dict[int, dict[str, Any]] = {}
    for element in document.elements:
        locator = element.locator
        if locator_kind == "pdf_page":
            page = locator.page
        elif locator_kind == "slide":
            page = locator.slide
        else:
            page = None
        if page is None:
            continue
        entry = pages.setdefault(
            int(page),
            {
                "page": int(page),
                "element_count": 0,
                "char_count": 0,
                "has_table": False,
                "has_formula": False,
                "has_figure": False,
            },
        )
        entry["element_count"] += 1
        entry["char_count"] += len(element.text or "")
        entry["has_table"] = entry["has_table"] or element.table is not None
        entry["has_formula"] = entry["has_formula"] or bool(element.latex)
        entry["has_figure"] = entry["has_figure"] or element.figure is not None

    return locator_kind, [pages[number] for number in sorted(pages)]


# --------------------------------------------------------------------------
# The original bytes
# --------------------------------------------------------------------------


def original_reference(objects: ObjectStore, sha256: str) -> ObjectRef | None:
    """The stored original for a content digest, or ``None``.

    Content-addressed on purpose: the key is derived from the digest, so this is
    one arithmetic step and one stat rather than a directory walk, and a document
    record only has to carry the hash it already has.
    """
    digest = str(sha256 or "").strip().lower()
    if len(digest) != 64:
        return None
    try:
        return objects.find_by_digest(LAYER_ORIGINAL, digest)
    except Exception:  # noqa: BLE001 - an unreachable store is "not available"
        logger.warning("original layer lookup failed for %s", digest, exc_info=True)
        return None


def store_original(objects: ObjectStore, file_path: str, content_type: str = "") -> ObjectRef:
    """Write an upload into the ``original`` layer, content-addressed, by name.

    The key is chosen here rather than left to
    :meth:`~paperwhisperer.infra.object_store.ObjectStore.put_file`, which derives
    a suffix from the source filename and then produces a key that
    :meth:`~paperwhisperer.infra.object_store.ObjectStore.find_by_digest` cannot
    find — it only probes the layer's canonical suffix. A document whose original
    cannot be looked up by digest cannot be reparsed, and the failure would be a
    409 that says "re-upload it" about a file the user never had to upload twice.
    So the digest is computed here (one streaming pass) and the key is
    ``digest_key(digest)``, which is the exact name ``find_by_digest`` looks for.
    """
    digest = compute_file_sha256(file_path)
    return objects.put_file(
        LAYER_ORIGINAL,
        file_path,
        key=digest_key(digest),
        content_type=content_type,
    )


def stored_original_key(objects: ObjectStore, record: DocumentRecord) -> str:
    """The key this document's original bytes are stored under, or ``""``.

    Two probes, and the second one is a workaround for a gap in the object store
    rather than a policy: :meth:`~paperwhisperer.infra.object_store.ObjectStore.
    find_by_digest` probes only the layer's declared suffix, while
    :meth:`~paperwhisperer.infra.object_store.ObjectStore.put_file` derives a
    suffix from the uploaded filename when the caller names no key. An original
    written from ``paper.pdf`` therefore lands at ``<digest>.pdf`` and is
    invisible to a digest lookup — the evidence router works around the same gap
    the same way.

    The upload path avoids the gap by naming the key itself (see
    :func:`store_original`), so a document this service uploaded is found by the
    first probe. The second exists for a document whose original was written
    *before* that: a reader that could not open it would answer
    ``original_unavailable`` for a file that is right there, which is the one
    answer a user cannot act on. Once the store's lookup learns the fan-out, or
    refuses the fan-out, this collapses to the first probe.
    """
    reference = original_reference(objects, record.sha256)
    if reference is not None:
        return str(reference.key)

    digest = str(record.sha256 or "").strip().lower()
    suffix = original_suffix(record.source_filename)
    if suffix and len(digest) == 64:
        candidate = digest_key(digest, suffix)
        if objects.exists(LAYER_ORIGINAL, candidate):
            return candidate
    return ""


def has_stored_original(objects: ObjectStore, record: DocumentRecord) -> bool:
    """Whether a reparse would have a file to read. One stat, not a read."""
    return bool(stored_original_key(objects, record))


def original_suffix(source_filename: str) -> str:
    """The file extension to keep when materialising an original for a reparse.

    Kept because the parsers route on the extension for anything the declared
    media type does not cover, and a reparse of ``paper.pdf`` must not be handed
    a file called ``paper``.
    """
    _, extension = os.path.splitext(str(source_filename or ""))
    return extension[:16]


def materialise_original(
    objects: ObjectStore, record: DocumentRecord, destination: str
) -> None:
    """Write a document's stored original to ``destination`` for a parser to read.

    Raises :class:`_Refusal` with ``original_unavailable`` when the bytes are not
    there, because a document registered from elsewhere never had an original
    written by this service and re-reading it is not something that can be
    invented. A 409 with that code is the difference between "re-upload it" and a
    stack trace.
    """
    key = stored_original_key(objects, record)
    if not key:
        raise _Refusal("original_unavailable")
    try:
        with (
            objects.open_stream(LAYER_ORIGINAL, key) as source,
            open(destination, "wb") as target,
        ):
            while True:
                block = source.read(1 << 20)
                if not block:
                    break
                target.write(block)
    except ObjectNotFound as exc:
        raise _Refusal("original_unavailable") from exc


# --------------------------------------------------------------------------
# Parser choice
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReparseParser:
    """One entry of the reparse control's parser menu.

    The four values are the four the front end offers
    (``REPARSE_PARSER_OPTIONS`` in ``templates/static/js/config.js``), and the
    table is the server's answer to the same question: which spellings are
    accepted, what they map onto, and — the one that matters — which of them
    leaves the machine. A parser absent from this table cannot be selected at
    all, which is the safe direction: an unlisted parser cannot be chosen by
    accident.
    """

    selection: str
    parser_name: str
    label: str
    force_ocr: bool = False
    leaves_machine: bool = False


REPARSE_PARSERS: dict[str, ReparseParser] = {
    "": ReparseParser(selection="", parser_name="", label="默认解析链"),
    "docling": ReparseParser(selection="docling", parser_name="docling", label="Docling（本地）"),
    "docling_ocr": ReparseParser(
        selection="docling_ocr",
        parser_name="docling",
        label="Docling + 强制 OCR（本地）",
        force_ocr=True,
    ),
    "mineru_cloud": ReparseParser(
        selection="mineru_cloud",
        parser_name="mineru_cloud",
        label="MinerU 云端解析",
        leaves_machine=True,
    ),
}


@dataclass(slots=True)
class ResolvedReparse:
    """What a reparse request resolved to, before anything is read.

    ``router`` is ``None`` for the default chain, which means "let
    :func:`~paperwhisperer.ingestion.orchestrator.parse_document` build the router
    the deployment configured" — passing an empty router there would silently
    override ``PARSER_PRIMARY`` and ``PARSER_FALLBACK_CHAIN`` with nothing.
    """

    chosen: ReparseParser
    router: ParserRouter | None
    options: ParseOptions
    cloud_parser: Any = None
    notice: str = ""


def build_parser_catalog() -> dict[str, Any]:
    """The parsers this deployment can actually run, by name.

    Read from the gateway's own factory rather than from a list written here, so
    an adapter that is missing on this machine changes the catalog and not the
    call sites. A parser that cannot run here is *absent* rather than present and
    failing, which is what makes ``parser_unavailable`` an honest 409.
    """
    return {
        str(getattr(parser, "name", "") or ""): parser
        for parser in build_default_parsers()
    }


def resolve_reparse_parser(selection: str, *, force_ocr: bool) -> ResolvedReparse:
    """Resolve a reparse's parser choice, or raise :class:`_Refusal`.

    Three refusals, each a different client mistake and each a different fix:

    ``unknown_parser``
        The spelling is not in :data:`REPARSE_PARSERS`. A client bug.
    ``parser_unavailable``
        The spelling is right but this deployment cannot run it — the adapter is
        not installed, no token is configured, or ``PARSER_BACKEND`` is still on
        the rollback flag, which collapses the registry to the legacy adapter
        whatever is installed. That last one is checked explicitly because the
        collapse happens inside the router, where a caller would otherwise get a
        generic "no parsers available" for a request it spelled correctly.
    ``parser_unavailable`` for a cloud selection when the privacy gate refuses is
        raised by the route, not here, because the gate is a policy decision with
        its own codes.

    The router is restricted to the one chosen parser when one was named. That is
    what makes "parser choice" real: a router holding the whole chain would
    quietly run something else and report success, and a document that was asked
    to be re-read by Docling and was in fact read by the legacy adapter is a
    provenance record that is wrong in the field a reader is told to trust.
    """
    key = str(selection or "").strip().lower()
    chosen = REPARSE_PARSERS.get(key)
    if chosen is None:
        raise _Refusal("unknown_parser")

    resolved_ocr = bool(force_ocr) or chosen.force_ocr
    options = ParseOptions(force_ocr=resolved_ocr)
    if not chosen.parser_name:
        return ResolvedReparse(
            chosen=chosen,
            router=None,
            options=options,
            notice="按部署配置的默认解析链解析。",
        )

    backend = str(getattr(config, "PARSER_BACKEND", "legacy") or "legacy").strip().lower()
    if backend != "gateway":
        raise _Refusal(
            "parser_unavailable",
            f"当前 PARSER_BACKEND={backend}，解析网关已整体回滚到 legacy 适配器，"
            f"因此无法指定解析器 {key!r}。请改用默认解析链，或由管理员设置 "
            "PARSER_BACKEND=gateway。",
        )

    catalog = build_parser_catalog()
    parser = catalog.get(chosen.parser_name)
    if parser is None:
        # A parser that is not registered is usually the *optional* one: docling
        # pulls in torch, and the cloud adapter is skipped without a token. So
        # the catalog is usually short, and an empty or docling-only registry is
        # the normal case rather than a broken deployment.
        available = "、".join(sorted(name for name in catalog if name)) or (
            "本部署没有注册任何解析器"
        )
        raise _Refusal(
            "parser_unavailable",
            f"所选解析器 {key!r} 在当前部署中不可用（已注册的解析器：{available}）。"
            "如需启用该解析器，请安装对应的可选依赖或配置其凭据；"
            "也可以改用默认解析链。",
        )

    if chosen.leaves_machine:
        return ResolvedReparse(
            chosen=chosen,
            router=None,
            options=options,
            cloud_parser=parser,
            notice="已按云端解析策略处理：只有本地解析链无法产出可用结果时才会出网。",
        )

    return ResolvedReparse(
        chosen=chosen,
        router=ParserRouter([parser]),
        options=options,
        notice=f"已指定本地解析器 {chosen.parser_name}"
        + ("（强制 OCR）。" if resolved_ocr else "。"),
    )


def _require_cloud_permission() -> None:
    """Refuse a cloud parse unless the deployment's own gate allows it.

    This is not a second privacy gate: it *is*
    :func:`~paperwhisperer.ingestion.adapters.mineru_cloud.evaluate_privacy_gate`,
    reached through the orchestrator's wrapper (which only adds import tolerance),
    called one layer earlier than the orchestrator would. The reason to call it
    here rather than let the chain run and skip the stage is the difference in
    what the caller learns: a 403 the user can act on, versus a successful
    response whose parse report happens to mention a refusal somewhere in its
    attempt trail. And the reason the gate's ``reason_code`` becomes the *code* of
    the response is that a client has to be able to tell "an administrator forbade
    this" from "you have not consented yet" — they send the user to different
    places, and a single wrapper code would send them to both.
    """
    decision = evaluate_cloud_privacy_gate()
    if decision.allowed:
        return
    if decision.reason_code == "cloud_adapter_unavailable":
        raise _Refusal("parser_unavailable", decision.message)
    raise _Refusal(decision.reason_code, decision.message)


# --------------------------------------------------------------------------
# Running one parse as a job
# --------------------------------------------------------------------------


@dataclass(slots=True)
class ParseRun:
    """What one parse produced, for the caller and the response that follows.

    A mutable box because the work happens inside a job step on a worker thread
    and the route needs the outcome on this thread afterwards. The alternative —
    reading it back out of the job's ``result`` — would mean reconstructing a
    :class:`ParseOutcome` from a dict, which is the kind of round trip that loses
    a field.
    """

    outcome: ParseOutcome | None = None
    record: DocumentRecord | None = None


def _job_result(outcome: ParseOutcome) -> dict[str, Any]:
    """The stage summary a job hands to its subscriber.

    Counts and flags. Never the document, never the model output: ``jobs.Job``
    says as much about ``result``, and the reason is the same one that keeps
    content off the stream.
    """
    document = outcome.document
    return {
        "document_id": str(getattr(document, "document_id", "") or ""),
        "parser_name": outcome.parsed_by,
        "status": outcome.status,
        "page_count": int(getattr(document, "page_count", 0) or 0),
        "char_count": int(getattr(document, "char_count", 0) or 0),
        "used_cloud": bool(outcome.used_cloud),
        "touched_network": bool(outcome.touched_network),
        "cache_hit": bool(outcome.cache_hit),
    }


def run_parse_job(
    store: JobStore,
    *,
    kind: str,
    payload: dict[str, Any],
    run,
    capture: ParseRun,
    document_id: str = "",
    session_id: str = "",
    lease_seconds: float = 60.0,
    drain_timeout: float = 900.0,
) -> Job:
    """Submit one parse, run it to a terminal state, and return the job.

    The pipeline has a single stage on purpose. The state machine's spine is
    seven states, and a reparse or an upload genuinely traverses most of it — but
    the work inside is
    :func:`~paperwhisperer.ingestion.orchestrator.parse_document` plus
    :func:`~paperwhisperer.analysis.orchestrator.index_document_for_retrieval`, one
    call each. Claiming a transition per sub-step would mean asserting, on every
    request, that those two calls are at a particular point in their own
    internal sequence — and that assertion is a second implementation of the
    escalation policy, in a module whose job is not to know what a parse does.
    So the job says ``parsing``, does the work, and settles; the fine-grained
    trail of what the parse did is in the document's own parse report, which is
    where the front end already reads it from.

    Drained inline. See the module docstring for why, and for why moving it
    behind ``JobWorker.start`` changes no wire format.
    """

    def _step(context) -> None:
        outcome = run()
        capture.outcome = outcome
        if outcome is None or not outcome.is_usable or outcome.document is None:
            message = (outcome.error_message if outcome is not None else "") or "文档解析失败。"
            raise StepFailed(message, retryable=False)
        context.update(document_id=str(outcome.document.document_id))
        context.update(result=_job_result(outcome))
        context.log(outcome.summary_line())
        for reason in list(outcome.warnings)[:3]:
            context.mark_partial(reason)

    pipeline = JobPipeline(
        kind=kind,
        steps=(JobStep(state=STATE_PARSING, run=_step, required=True, label="正在解析文档。"),),
    )
    submitted = store.submit(kind, payload, document_id=document_id, session_id=session_id)
    worker = JobWorker(store, pipeline, lease_seconds=lease_seconds, poll_interval=0.01)
    worker.drain(timeout=drain_timeout)
    return store.get_job(submitted.job_id) or submitted.job


def _unhappy_status(job: Job) -> int | None:
    """The status a finished-but-unsuccessful parse job maps to, or ``None``.

    ``completed`` and ``partial`` are both successes and both return ``None``:
    ``partial`` means the document parsed with visible warnings, and reporting
    that as a failure is the "partial success shown as total loss" the job state
    machine exists to prevent. A parse that produced nothing usable is 422 — the
    request was valid and the document was readable, the pipeline just could not
    read it. A job that was still running when the drain gave up is *not* in this
    table at all: the caller answers 202 with the job, because "still going" is a
    real answer and pretending otherwise is how a UI claims a reparse finished
    before it did.
    """
    if job.state == "cancelled":
        return 409
    if job.state == "failed":
        return 422
    return None


def _parse_failure_response(job: Job, status: int):
    """The response for a job that finished without producing a document.

    The message is the job's own ``error_message`` rather than a sentence
    written here, because the step that failed is the one that knows why: a
    Chinese parse failure from the orchestrator, or a cancellation notice from the
    worker, both belong to the reader and both are more useful than a fixed
    string. The ``code`` is the one a client branches on, and it is the same
    whether the job was cancelled or failed, so a client does not have to read
    the prose to know the document was not updated.
    """
    code = "job_cancelled" if status == 409 else "document_parse_failed"
    return build_error_response(
        job.error_message or "文档解析失败。", status_code=status, code=code
    )


# --------------------------------------------------------------------------
# Routes: documents
# --------------------------------------------------------------------------


@router.get(
    "/v1/documents",
    summary="列出文档",
    tags=["documents"],
    response_model=DocumentList,
    responses=error_responses((503, "存储不可用。")),
)
async def list_documents(
    limit: int = 50,
    offset: int = 0,
    stores: StoreBundle = Depends(get_store_bundle),
    objects: ObjectStore = Depends(get_object_store),
):
    """List documents newest first.

    Two integers and a page, with the same tiebreak the store itself uses, so the
    order is the same on both tiers and a page boundary cannot reshuffle.
    """
    window = config.clamp_int_value(limit, 50, min_value=1, max_value=200)
    start = max(0, int(offset))
    try:
        records = stores.metadata.list_documents(limit=window, offset=start)
        items = [document_resource(record, stores, objects) for record in records]
    except Exception:  # noqa: BLE001 - a store outage is a 503, not a 500
        logger.exception("Document listing failed")
        return document_error_response("storage_unavailable")
    return {"items": items, "count": len(items), "limit": window, "offset": start}


@router.post(
    "/v1/documents",
    summary="上传文档（解析并建立索引）",
    tags=["documents"],
    status_code=201,
    response_model=DocumentCreatedResponse,
    responses=error_responses(
        (400, "请求体不合法、缺少文件、文件类型不支持或元数据不合法。"),
        (404, "会话不存在。"),
        (409, "任务已被取消。"),
        (422, "解析未能产出可用的文档。"),
        (500, "上传过程中发生内部错误。"),
        (503, "存储不可用。"),
    ),
)
async def create_document(
    response: Response,
    file: UploadFile | None = File(None),
    session_id: str = Form(""),
    session_token: str = Form(""),
    stores: StoreBundle = Depends(get_store_bundle),
    objects: ObjectStore = Depends(get_object_store),
    store: JobStore = Depends(get_job_store),
):
    """Create a document from an uploaded file.

    Five things happen, in this order, and the order is the design: validate the
    upload against the same contract the legacy route uses; write the bytes to the
    ``original`` layer, which is write-once and content-addressed; parse; index;
    and record the parse provenance on the document. The original is written
    *before* the parse, so a reparse is always possible afterwards — the legacy
    ``/api/analyze`` deletes its upload, which is why a document analysed through
    it cannot be reparsed and this one can.

    No ``api_key``. This resource is parse and index; the model runs on a job
    kind that T2.3 owns.
    """
    invalid = validate_upload(file)
    if invalid is not None:
        return invalid

    cleanup_expired_sessions()
    safe_session_id = ""
    if str(session_id or "").strip():
        try:
            safe_session_id, _payload = load_validated_session(
                session_id, session_token, require_token=True
            )
        except PermissionError as exc:
            return build_error_response(
                redact_secrets(str(exc)), status_code=403, code="session_token_invalid"
            )
        except ValueError as exc:
            return build_error_response(
                redact_secrets(str(exc)), status_code=404, code="session_not_found"
            )

    file_path: str | None = None
    try:
        original_filename = build_safe_upload_filename(file.filename)
        file_path = build_unique_storage_path(config.UPLOAD_FOLDER, original_filename)
        await save_upload_file(file, file_path, config.MAX_UPLOAD_BYTES)

        stored_original = False
        notice = ""
        try:
            reference = store_original(objects, file_path, file.content_type or "")
            stored_original = not reference.deduplicated
        except Exception:  # noqa: BLE001 - the parse below reads the file on disk
            # Degraded rather than refused. The document is still real and still
            # indexable; what it loses is the ability to be reparsed later, so the
            # notice says so instead of the resource quietly advertising
            # ``original_available`` as if it were a fact rather than a stat.
            logger.exception("Storing the original upload failed")
            notice = " 原始文件未能存入对象存储，因此该文档暂不支持重新解析。"

        capture = ParseRun()
        job = await asyncio.to_thread(
            run_parse_job,
            store,
            kind="document_ingest",
            payload={
                "source_filename": original_filename,
                "force_ocr": False,
                "parser": "",
            },
            run=lambda: parse_document(file_path, media_type=file.content_type or None),
            capture=capture,
            document_id="",
            session_id=safe_session_id,
        )
        unhappy = _unhappy_status(job)
        if unhappy is not None:
            return _parse_failure_response(job, unhappy)
        outcome = capture.outcome
        if outcome is None or not outcome.is_usable or outcome.document is None:
            return build_error_response(
                (outcome.error_message if outcome is not None else "") or "文档解析失败。",
                status_code=422,
                code="document_parse_failed",
            )

        # The parser was handed a staged path, and records the basename of the
        # file it was given. The user uploaded ``paper.txt``; the document must
        # not be recorded as ``paper_<uuid>.txt`` or every export and download
        # names a file the user never saw.
        outcome.document.source_filename = (
            original_filename or outcome.document.source_filename
        )
        record = await asyncio.to_thread(
            persist_document, stores, outcome.document, parse_report(outcome)
        )
        chunks = len(stores.metadata.list_chunks(record.document_id))
        if not job.terminal:
            # The drain gave up before the job settled, which should not happen
            # but is reported rather than hidden: 202 says "still going" and the
            # job is the thing to watch.
            response.status_code = 202
        return {
            "document": document_resource(record, stores, objects, chunk_count=chunks),
            "job": job_state(job),
            "original_stored": stored_original,
            "notice": outcome.summary_line() + notice,
        }
    except ValueError as exc:
        return document_error_response("upload_rejected", redact_secrets(str(exc)))
    except Exception:  # noqa: BLE001 - surfaced as a 500, never a stack trace
        logger.exception("Document upload failed")
        return build_error_response(
            "上传文档失败。", status_code=500, code="document_upload_failed"
        )
    finally:
        await close_upload_file_safely(file, "v1 document upload file")
        remove_file_safely(file_path, "v1 document upload file")


def persist_document(
    stores: StoreBundle,
    document: ParsedDocument,
    report: dict[str, Any],
) -> DocumentRecord:
    """Index a parsed document and record its provenance on it.

    Chunking and indexing go through
    :func:`~paperwhisperer.analysis.orchestrator.index_document_for_retrieval`
    rather than being reimplemented, because that function owns the
    index-version check and the vector/lexical writes; a second indexing path
    would be a second answer to "is this document retrievable yet".

    ``status`` is ``partial`` when the parse came back partial, so the store row
    agrees with the parse report instead of claiming a clean index over a
    degraded parse.
    """
    document_id = str(document.document_id)
    index_document_for_retrieval(stores, document)
    record = stores.metadata.get_document(document_id) or DocumentRecord(document_id=document_id)
    record.document = document
    record.source_filename = str(document.source_filename or record.source_filename)
    record.media_type = str(document.media_type or record.media_type)
    record.sha256 = str(document.sha256 or record.sha256)
    record.page_count = int(document.page_count or record.page_count)
    record.parser_name = str(document.parser_name or record.parser_name)
    record.parser_version = str(document.parser_version or record.parser_version)
    record.options_hash = str(document.options_hash or record.options_hash)
    if not record.index_version:
        chunks = stores.metadata.list_chunks(document_id)
        record.index_version = str(getattr(chunks[0], "index_version", "") if chunks else "")
    if str(report.get("status") or "") == PARSE_PARTIAL:
        # The store row and the parse report have to agree. A row claiming a clean
        # index over a parse that returned partial content is the disagreement
        # that makes a degraded document look fine in a list and look broken when
        # a citation resolves into the gap.
        record.status = "partial"
    record.metadata = {**dict(record.metadata or {}), "parse": dict(report)}
    return stores.metadata.upsert_document(record)


@router.post(
    "/v1/documents/register",
    summary="登记已解析的文档",
    tags=["documents"],
    status_code=201,
    response_model=DocumentRegisteredResponse,
    responses=error_responses(
        (400, "请求体不是 JSON 对象，或其中的 document 不符合 ParsedDocument 结构。"),
        (409, "同 document_id 的文档已存在。"),
        (503, "存储不可用。"),
    ),
)
async def register_document(
    request: Request,
    stores: StoreBundle = Depends(get_store_bundle),
    objects: ObjectStore = Depends(get_object_store),
):
    """Register a document that was parsed somewhere else.

    The partner of ``POST /v1/documents``, and the reason there are two creates
    rather than one: upload takes *bytes* and does the parsing, register takes a
    finished canonical :class:`~paperwhisperer.ingestion.schema.ParsedDocument`
    and adopts it. That is what an ingest that ran offline, a script, or a future
    T2.3 pipeline hands in — and it is why the two are different paths rather than
    one route dispatching on a content type: they need different validation, and
    a single path that guessed would answer 415 to a client with a valid body.

    The document is validated by ``ParsedDocument.from_dict``, which refuses a
    ``schema_version`` this build cannot read rather than accepting locators that
    may mean anything. It is *not* reparsed, so nothing is claimed about how the
    bytes were read: the parse report says so explicitly rather than reporting
    ``used_cloud: false`` for a parse that may have happened anywhere.

    No original is written, so ``original_available`` is ``false`` and a reparse
    of this document answers 409 until it is uploaded through
    ``POST /v1/documents``.
    """
    data, error_response = await parse_json_object_request(request)
    if error_response is not None:
        return error_response

    payload = data.get("document")
    if not isinstance(payload, dict):
        return build_error_response(
            "请求体需要包含 document 字段（ParsedDocument 的 JSON 结构）。",
            code="invalid_document_payload",
        )
    # ``from_dict`` is permissive on purpose — it fills in defaults so an
    # older serialisation still loads — which means an object with none of the
    # document's fields in it would be accepted and given a freshly minted id.
    # A registration that invents its own identity is worse than a refusal: the
    # client would hold an id it never sent, and a second registration of the
    # same parse would create a second document.
    if not str(payload.get("document_id") or "").strip():
        return build_error_response(
            "请求体中的 document 缺少非空的 document_id：登记一个文档需要知道它的身份，"
            "服务端不会替调用方编造。",
            code="invalid_document_payload",
        )

    try:
        document = ParsedDocument.from_dict(payload)
    except (ValueError, TypeError, KeyError) as exc:
        logger.info("rejected document registration: %s", exc)
        return build_error_response(
            f"提交的解析结果不符合 ParsedDocument 结构：{exc}", code="invalid_document_payload"
        )

    if stores.metadata.get_document(document.document_id) is not None:
        return document_error_response("document_exists")

    report = _empty_parse_report(
        "该文档由 /v1/documents/register 登记：解析过程发生在本服务之外，来源未记录。"
    )
    try:
        record = await asyncio.to_thread(persist_document, stores, document, report)
        chunks = len(stores.metadata.list_chunks(record.document_id))
    except Exception:  # noqa: BLE001 - a store outage is a 503, not a 500
        logger.exception("Document registration failed")
        return document_error_response("storage_unavailable")

    return {
        "document": document_resource(record, stores, objects, chunk_count=chunks),
        "chunk_count": chunks,
        "notice": "已登记并建立索引。原始文件未入库，因此该文档暂不支持重新解析。",
    }


def _load_document_or_error(
    document_id: str, stores: StoreBundle
) -> tuple[DocumentRecord | None, Any]:
    """Resolve a document id to a stored record, or the error response for it.

    Two failures, in this order: an id that could not name anything is the
    client's own bug (400), and a well-formed id that names nothing is a real 404.
    Answering the first with the second would send a client looking for a document
    that was never there.
    """
    if not evidence_id_guard(document_id):
        return None, document_error_response("malformed_id")
    record = stores.metadata.get_document(document_id)
    if record is None:
        return None, document_error_response("document_not_found")
    return record, None


@router.get(
    "/v1/documents/{document_id}",
    summary="读取文档",
    tags=["documents"],
    response_model=DocumentResource,
    responses=error_responses(
        (400, "文档 ID 格式不合法。"),
        (404, "文档不存在。"),
        (503, "存储不可用。"),
    ),
)
async def get_document(
    document_id: str,
    stores: StoreBundle = Depends(get_store_bundle),
    objects: ObjectStore = Depends(get_object_store),
):
    """One document's identity, counters, provenance and links."""
    record, error = _load_document_or_error(document_id, stores)
    if error is not None:
        return error
    try:
        return document_resource(record, stores, objects)
    except Exception:  # noqa: BLE001 - a store outage is a 503, not a 500
        logger.exception("Document read failed for %s", document_id)
        return document_error_response("storage_unavailable")


@router.get(
    "/v1/documents/{document_id}/metadata",
    summary="读取文档的解析元数据",
    tags=["documents"],
    response_model=ParseReport,
    responses=error_responses(
        (400, "文档 ID 格式不合法。"),
        (404, "文档不存在。"),
        (503, "存储不可用。"),
    ),
)
async def get_document_metadata(document_id: str, stores: StoreBundle = Depends(get_store_bundle)):
    """How this document was read, and whether it left this machine.

    The same block the document resource carries as ``parse``, served on its own
    so a client can refresh the provenance after a reparse without re-reading the
    whole resource. "Metadata" here means *parse* metadata, not arbitrary
    key/value: T2.2's extensible metadata travels in the document resource's
    counter set, and a bag of arbitrary keys on a frozen surface is a place where
    nothing can be deprecated.

    Every field is present even when the provenance was never recorded, in which
    case ``summary`` says so. An absent signal is reported as absent; it is never
    reported as a reassuring one.
    """
    record, error = _load_document_or_error(document_id, stores)
    if error is not None:
        return error
    return stored_parse_report(record)


@router.get(
    "/v1/documents/{document_id}/pages",
    summary="列出文档的页面",
    tags=["documents"],
    response_model=PageList,
    responses=error_responses(
        (400, "文档 ID 格式不合法。"),
        (404, "文档不存在。"),
        (503, "存储不可用。"),
    ),
)
async def list_document_pages(
    document_id: str,
    stores: StoreBundle = Depends(get_store_bundle),
):
    """What the parser found on each page, and where the page image is not.

    The page-image route is owned by ``routes_evidence``, and it answers 501 with
    a reason code when this build cannot rasterise a page. This listing advertises
    that URL together with the build's actual capability and the 501's status,
    code and reason, so a client can distinguish "this build cannot rasterise
    pages" from "this document has no highlight" without making a request whose
    answer it already has — and so the day a renderer is installed, the client is
    already pointing at the right URL.
    """
    record, error = _load_document_or_error(document_id, stores)
    if error is not None:
        return error

    locator_kind, pages = page_descriptors(record)
    renderer = page_renderer_name()
    return {
        "document_id": document_id,
        "media_type": str(record.media_type),
        "page_count": int(record.page_count),
        "locator_kind": locator_kind,
        "page_image": {
            "url_template": f"/v1/documents/{document_id}/pages/{{page}}/image",
            "renderer": renderer,
            "available": renderer != "none",
            "unavailable_status": PAGE_RENDER_STATUS,
            "unavailable_code": PAGE_RENDER_CODE,
            "unavailable_message": PAGE_RENDER_MESSAGE,
        },
        "pages": pages,
    }


@router.post(
    "/v1/documents/{document_id}/reparse",
    summary="按新选项重新解析文档",
    tags=["documents"],
    response_model=ReparseResponse,
    responses=error_responses(
        (400, "请求体不合法、文档 ID 格式不合法、解析器选项未知，或缺少 API Key。"),
        (403, "隐私策略拒绝云端解析（privacy_mode_local_only / consent_not_granted），"
              "或会话令牌无效。"),
        (404, "文档不存在，或会话不存在。"),
        (409, "原始文件不可用、解析器不可用、任务已取消。"),
        (422, "重新解析未能产出可用的文档。"),
        (500, "重新解析过程中发生内部错误。"),
        (503, "存储不可用。"),
    ),
)
async def reparse_document(
    document_id: str,
    request: Request,
    response: Response,
    stores: StoreBundle = Depends(get_store_bundle),
    objects: ObjectStore = Depends(get_object_store),
    store: JobStore = Depends(get_job_store),
):
    """Re-read a document with forced OCR or a different parser.

    The endpoint ``templates/static/js/parseStatus.js`` already posts to. Body:

    ``parser``
        ``""`` (default chain), ``"docling"``, ``"docling_ocr"`` or
        ``"mineru_cloud"``. Anything else is ``unknown_parser``.
    ``force_ocr``
        Re-read the whole document through OCR. Slower and less accurate on text
        that is already digital, which is why the control warns before it is used.
    ``session_id`` / ``session_token``
        Optional. Recorded on the job for correlation; not required, because the
        document itself carries the text.

    A cloud parser means the document leaves the machine, and that is refused
    unless the deployment's own gate allows it — see :func:`_require_cloud_permission`
    and this module's docstring. A refusal is a 403 carrying the gate's reason
    code, and nothing has been read or sent at that point.

    An explicit local parser replaces the chain rather than being preferred within
    it, so a reparse that asked for Docling cannot quietly succeed on the legacy
    adapter and report a parser the user did not choose.
    """
    data, error_response = await parse_json_object_request(request)
    if error_response is not None:
        return error_response

    record, error = _load_document_or_error(document_id, stores)
    if error is not None:
        return error

    safe_session_id = ""
    if str(data.get("session_id") or "").strip():
        try:
            safe_session_id, _payload = load_validated_session(
                data.get("session_id"), data.get("session_token"), require_token=True
            )
        except PermissionError as exc:
            return build_error_response(
                redact_secrets(str(exc)), status_code=403, code="session_token_invalid"
            )
        except ValueError as exc:
            return build_error_response(
                redact_secrets(str(exc)), status_code=404, code="session_not_found"
            )

    # The original comes first, deliberately. It is a local read and it decides
    # whether *any* reparse is possible, so answering it before the parser
    # question stops a document that has no bytes left from being reported as
    # "that parser is not installed here" — a message about the deployment
    # pointing at a document that is the thing that is missing.
    suffix = original_suffix(record.source_filename)
    file_path = build_unique_storage_path(config.UPLOAD_FOLDER, f"reparse{document_id}{suffix}")
    try:
        materialise_original(objects, record, file_path)
    except _Refusal as refusal:
        remove_file_safely(file_path, "v1 reparse staging file")
        return refusal_response(refusal)
    except Exception:  # noqa: BLE001 - a store outage is a 503, not a 500
        remove_file_safely(file_path, "v1 reparse staging file")
        logger.exception("Reading the stored original failed for %s", document_id)
        return document_error_response("storage_unavailable")

    try:
        selection = str(data.get("parser") or "")
        # A cloud selection is refused by *policy* before it is resolved as
        # *availability*, and the order is what the user sees. Resolving first
        # answers `mineru_cloud` under `local_only` with 409
        # `parser_unavailable` and the advice "install the optional dependency
        # or configure its credentials" — because a deployment that forbids
        # egress never registers the cloud adapter, so the catalog genuinely
        # lacks it. That is true and useless: the operator is sent to install a
        # dependency when an administrator has forbidden cloud parsing, and
        # installing it would not make the request legal. Reading
        # `leaves_machine` off the table needs no adapter, so the gate can run
        # first and answer 403 `privacy_mode_local_only` with the fix that
        # applies. An unlisted selection is not cloud, so it still reaches
        # `resolve_reparse_parser` and is answered `unknown_parser`.
        preselected = REPARSE_PARSERS.get(selection.strip().lower())
        if preselected is not None and preselected.leaves_machine:
            _require_cloud_permission()

        resolved = resolve_reparse_parser(
            selection,
            force_ocr=config.parse_bool_value(data.get("force_ocr"), default=False),
        )
    except _Refusal as refusal:
        remove_file_safely(file_path, "v1 reparse staging file")
        return refusal_response(refusal)

    # Re-checked here on purpose. The gate above already refused a cloud
    # selection, so this cannot be the first refusal — but it is the last line
    # before a parser is constructed and a file is read, and keeping the check
    # adjacent to the work means a future caller that reaches this point some
    # other way is still stopped by policy rather than by luck.
    if resolved.chosen.leaves_machine:
        try:
            _require_cloud_permission()
        except _Refusal as refusal:
            remove_file_safely(file_path, "v1 reparse staging file")
            return refusal_response(refusal)

    try:
        capture = ParseRun()

        def _run() -> ParseOutcome:
            return parse_document(
                file_path,
                media_type=record.media_type or None,
                router=resolved.router,
                options=resolved.options,
                # A named local parser must not be able to escalate: the user
                # asked for that parser, and silently spending a cloud call
                # after the local one failed would be the exact behaviour the
                # reparse control's "this leaves your machine" warning is about.
                allow_cloud_fallback=not resolved.router,
                cloud_parser=resolved.cloud_parser,
            )

        job = await asyncio.to_thread(
            run_parse_job,
            store,
            kind="document_reparse",
            payload={
                "document_id": document_id,
                "parser": resolved.chosen.selection,
                "force_ocr": resolved.options.force_ocr,
            },
            run=_run,
            capture=capture,
            document_id=document_id,
            session_id=safe_session_id,
        )
        unhappy = _unhappy_status(job)
        if unhappy is not None:
            return _parse_failure_response(job, unhappy)
        outcome = capture.outcome
        if outcome is None or not outcome.is_usable or outcome.document is None:
            return build_error_response(
                (outcome.error_message if outcome is not None else "") or "重新解析失败。",
                status_code=422,
                code="document_parse_failed",
            )

        report = parse_report(outcome)
        # The reparsed document keeps the id the client addressed. A reparse
        # replaces the current version of *this* document; minting a new id would
        # make the operation the caller asked for impossible to follow, and the
        # citations it already holds would point at a document they can no longer
        # name. So the record is re-upserted over the old one, and the store's
        # optimistic-lock counter is what records that a version replaced
        # another.
        outcome.document.document_id = document_id
        outcome.document.source_filename = (
            record.source_filename or outcome.document.source_filename
        )
        outcome.document.media_type = record.media_type or outcome.document.media_type
        stored = await asyncio.to_thread(persist_document, stores, outcome.document, report)
        chunks = len(stores.metadata.list_chunks(document_id))
        if not job.terminal:
            response.status_code = 202

        notice = resolved.notice
        if resolved.chosen.leaves_machine and not outcome.used_cloud:
            notice += " 本次未上传：本地解析链已产出可用结果，因此没有调用云端服务。"

        return {
            "document_id": document_id,
            "document": document_resource(stored, stores, objects, chunk_count=chunks),
            "parse": report,
            "job": job_state(job),
            "cloud_requested": bool(resolved.chosen.leaves_machine),
            "notice": notice,
        }
    except Exception:  # noqa: BLE001 - surfaced as a 500, never a stack trace
        logger.exception("Reparse failed for document %s", document_id)
        return build_error_response("重新解析失败。", status_code=500, code="reparse_failed")
    finally:
        remove_file_safely(file_path, "v1 reparse staging file")


@router.delete(
    "/v1/documents/{document_id}",
    summary="删除文档",
    tags=["documents"],
    response_model=DocumentDeletedResponse,
    responses=error_responses(
        (400, "文档 ID 格式不合法。"),
        (404, "文档不存在。"),
        (409, "该文档仍有未结束的任务。"),
        (503, "存储不可用。"),
    ),
)
async def delete_document(
    document_id: str,
    stores: StoreBundle = Depends(get_store_bundle),
    objects: ObjectStore = Depends(get_object_store),
    store: JobStore = Depends(get_job_store),
):
    """Delete a document, its chunks, its index entries and its original.

    Refused while a job for this document is still running. The reason is
    concrete rather than cautious: the job's next stage would write a document
    row and chunks back into a store the user just emptied, and the result would
    be a document the user deleted reappearing with no way to tell that it had
    been deleted at all. The client is told to cancel the job first, and
    ``POST /v1/jobs/{id}/cancel`` is the request that stops it.

    The ``original`` layer needs ``force`` — it is the one object in the store
    that cannot be produced again — and a genuine user-requested delete is exactly
    the case that authorises it.
    """
    record, error = _load_document_or_error(document_id, stores)
    if error is not None:
        return error

    outstanding = [
        job.job_id
        for job in store.list_jobs(document_id=document_id, limit=200)
        if not job.terminal
    ]
    if outstanding:
        return document_error_response("document_busy")

    original_deleted = False
    original_key = stored_original_key(objects, record)
    if original_key:
        try:
            original_deleted = bool(objects.delete(LAYER_ORIGINAL, original_key, force=True))
        except Exception:  # noqa: BLE001 - the metadata delete still has to happen
            logger.warning("Deleting the original for %s failed", document_id, exc_info=True)

    try:
        deleted = stores.delete_document(document_id)
    except Exception:  # noqa: BLE001 - a store outage is a 503, not a 500
        logger.exception("Document delete failed for %s", document_id)
        return document_error_response("storage_unavailable")

    return {
        "document_id": document_id,
        "deleted": bool(deleted),
        "original_deleted": original_deleted,
    }


# --------------------------------------------------------------------------
# Routes: questions
# --------------------------------------------------------------------------


@router.post(
    "/v1/documents/{document_id}/qa",
    summary="就文档提问（SSE）",
    tags=["documents"],
    response_class=StreamingResponse,
    responses={
        **error_responses(
            (400, "请求体不合法、文档 ID 格式不合法、缺少问题或缺少 API Key。"),
            (403, "会话令牌无效。"),
            (404, "文档不存在，或会话不存在。"),
        ),
        200: {
            "content": {"text/event-stream": {}},
            "description": (
                "事件流。`start`（文档与会话信息）、`delta`（答案增量）、`done`"
                "（完整 AnswerResult，含 claims/citations/confidence）、`error`。"
            ),
        },
    },
)
async def ask_about_document(
    document_id: str,
    request: Request,
    stores: StoreBundle = Depends(get_store_bundle),
):
    """Stream an answer about one document.

    Deliberately POST rather than GET: the question is the request, and putting it
    in a query string puts it in access logs, browser history and any proxy's
    ``Referer``. The legacy ``/api/ask/stream`` is POST for the same reason, and
    the producer here is the same one, so the two cannot drift.

    The document is retrieved from the store rather than from the session, which
    is what makes the citations locators instead of character offsets: the
    retriever indexes the document the caller named, so an evidence id resolves to
    a page. The session is optional and, when given, supplies the follow-up
    history and records this turn. ``answer_mode`` is normalised exactly as
    ``/api/ask`` normalises it, so one client cannot ask for a stricter mode on
    one route and get it by accident on the other.

    The stream carries the *answer* — this is content, and the state-only rule
    applies to the job feed, not to a question. The distinction is the point:
    ``/v1/jobs/{id}/stream`` is resumable precisely because it carries no content,
    and this one is not resumable because it carries the answer.
    """
    data, error_response = await parse_json_object_request(request)
    if error_response is not None:
        return error_response

    record, error = _load_document_or_error(document_id, stores)
    if error is not None:
        return error

    question = str(data.get("question") or "").strip()
    if not question:
        return document_error_response("missing_question")
    resolved_api_key = config.resolve_api_key(data.get("api_key", ""))
    if not resolved_api_key:
        return document_error_response("missing_api_key")

    document = record.document or ParsedDocument(document_id=document_id)
    answer_mode = normalize_answer_mode(data.get("answer_mode"))
    history: list = []
    session_payload: dict | None = None
    safe_session_id = ""
    if str(data.get("session_id") or "").strip():
        try:
            safe_session_id, session_payload = load_validated_session(
                data.get("session_id"), data.get("session_token"), require_token=True
            )
        except PermissionError as exc:
            return build_error_response(
                redact_secrets(str(exc)), status_code=403, code="session_token_invalid"
            )
        except ValueError as exc:
            return build_error_response(
                redact_secrets(str(exc)), status_code=404, code="session_not_found"
            )
        history = list(session_payload.get("qa_history") or [])

    cleanup_expired_sessions()

    def produce():
        analyzer = build_qa_analyzer(resolved_api_key, document.to_full_text())
        # The document itself, so retrieval indexes it and every citation is a
        # locator rather than a character offset. A client can tell which it got
        # from ``start.locator_kind``.
        analyzer.qa_document = document
        # Retrieval runs over the bundle *this* route resolved the document in,
        # rather than the Q&A module's own process-wide bundle. They point at the
        # same store in a deployment, but a route that looked the document up in
        # one place and retrieved from another is a route whose answer depends on
        # which global happened to be built first.
        analyzer.store_bundle = stores
        yield {
            "type": "start",
            "data": {
                "session_id": safe_session_id,
                "document_id": document_id,
                "answer_mode": answer_mode,
                "locator_kind": (
                    str(document.elements[0].locator.kind) if document.elements else "unknown"
                ),
            },
        }

        deltas: list[str] = []
        result: AnswerResult | None = None
        for item in analyzer.stream_answer_question(
            question, history=history, answer_mode=answer_mode
        ):
            # Distinguishing the finished result from a delta by type rather than
            # by position, the way routes_qa does: the number of deltas is a
            # rendering detail, and deciding on it would turn a changed stream
            # length into a paragraph of JSON at the end.
            if isinstance(item, AnswerResult):
                result = item
                continue
            deltas.append(str(item))
            yield {"type": "delta", "data": {"text": str(item)}}

        # A stream that produced no AnswerResult came from an analyzer that does
        # not implement the T1.7 contract. The text is already on the reader's
        # screen, so the answer is the text and the confidence is the only honest
        # one available for it: nothing was verified, so nothing is supported.
        result = as_answer_result(result if result is not None else "".join(deltas))
        if session_payload is not None:
            record_qa_turn(session_payload, question, result.answer, answer_mode)
            _attach_turn_evidence(session_payload, result)
            write_session_payload(safe_session_id, session_payload)
        yield {"type": "done", "data": result.to_dict()}

    def encode(event: dict) -> str:
        return build_sse_event(event["type"], event["data"])

    async def event_stream():
        try:
            async for chunk in pump_sync_events(produce, encode):
                yield chunk
        except Exception as exc:  # noqa: BLE001 - surfaced as an SSE error event
            logger.exception("Streaming question answering failed for %s", document_id)
            yield build_sse_event(
                "error",
                build_error_payload(
                    redact_secrets(str(exc)), code="streaming_question_failed"
                ),
            )

    return StreamingResponse(
        event_stream(), media_type="text/event-stream", headers=build_sse_headers()
    )


# --------------------------------------------------------------------------
# Routes: search
# --------------------------------------------------------------------------


@router.post(
    "/v1/search/papers",
    summary="检索论文",
    tags=["search"],
    response_model=PaperSearchResponse,
    responses=error_responses(
        (400, "请求体不合法，或缺少检索关键词。"),
        (500, "检索失败。"),
    ),
)
async def search_papers_v1(request: Request):
    """Search the configured providers and return deduplicated results.

    POST rather than GET, and the reason is cost rather than convention: with
    ``rewrite`` on, this call may spend an LLM request, and a GET is prefetchable
    and cacheable by things that are not the user. A crawler following a link
    would be a way to bill a model for a search nobody asked for. The body's
    ``api_key`` is a second reason and an older one — a credential does not belong
    in a URL, in a ``Referer`` header or in an access log.

    ``rewrite`` (default ``true``) is the LLM query expansion. With no API key, or
    with it off, the query is searched as written; ``query`` is what was actually
    searched and ``original_query`` is what the caller asked, so a client can
    always tell whether it is looking at results for its own words. ``reason``
    says why in one sentence, including when the expansion failed and the raw
    query was used — a client that showed "rewritten" for a failed expansion would
    be explaining results the user never asked for.
    """
    data, error_response = await parse_json_object_request(request)
    if error_response is not None:
        return error_response

    query = str(data.get("query") or "").strip()
    if not query:
        return document_error_response("missing_search_query")

    limit = data.get("limit") or config.PAPER_SEARCH_RESULT_LIMIT
    wants_rewrite = config.parse_bool_value(data.get("rewrite"), default=True)
    resolved_api_key = config.resolve_api_key(data.get("api_key", ""))
    searched = query
    reason = "按原始关键词检索，未使用 AI 改写。"
    if wants_rewrite and resolved_api_key and config.PAPER_SEARCH_ENABLE_REWRITE:
        from paperwhisperer.analysis.orchestrator import DocumentAnalyzer

        try:
            meta = DocumentAnalyzer(resolved_api_key).rewrite_search_query(
                query=query, context_text=str(data.get("context_text") or "")
            )
        except Exception as exc:  # noqa: BLE001 - the raw query can still answer
            logger.warning("Paper search query rewrite failed: %s", exc)
            reason = f"查询扩展失败（{exc}），已按原始关键词检索。"
        else:
            searched = str(meta.get("rewritten_query") or query)
            reason = str(meta.get("reason") or reason)
    elif wants_rewrite:
        reason = "未配置 API Key，按原始关键词检索。"

    try:
        result = search_papers(searched, limit)
    except ValueError as exc:
        return build_error_response(
            redact_secrets(str(exc)), code="missing_search_query"
        )
    except Exception:  # noqa: BLE001 - surfaced as a 500, never a stack trace
        logger.exception("Paper search failed")
        return build_error_response("论文检索失败。", status_code=500, code="search_failed")

    items = list(result.get("items") or [])
    return {
        "query": searched,
        "original_query": query,
        "count": len(items),
        "items": items,
        "possible_duplicates": list(result.get("possible_duplicates") or []),
        "merged_duplicates": list(result.get("merged_duplicates") or result.get("merged") or []),
        "errors": [str(item) for item in (result.get("errors") or [])],
        "reason": reason,
    }
