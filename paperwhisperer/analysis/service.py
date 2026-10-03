"""Analysis service: run an analysis and persist its session.

This is the layer the API routes call. It owns the side effects of an
analysis — writing the Markdown report, minting the session token and storing
the session — so route handlers stay thin.

Two generations of analysis arrive here, and this module is the seam between
them.

**The old path** (:class:`paperwhisperer.analysis.orchestrator.DocumentAnalyzer`)
asks six sections to read the same document independently. It still runs, it
still produces every key the API, the session payload and the templates
already read, and nothing in this file removes it: T1.8's acceptance criterion
is a *comparison* between the two paths, and deleting the baseline before the
comparison has been measured would make the number unreproducible.

**The new path** (:class:`paperwhisperer.analysis.paper_facts.FactLayerPipeline`)
extracts eight structured facts once and renders every card from them. This
module is where that result becomes an API result: the fact layer and its run
record are attached to the same dict the old path produces, under new keys, so
that a client which has never heard of them is unaffected and a client that
renders them gets a consistent set of cards.

**Why the fact layer goes on the result at all**, rather than only into the
stored session: the acceptance criterion is 跨卡片事实一致, and a consistency
guarantee that is only visible in a database row cannot be rendered, cannot be
regression-tested against a fixture, and cannot be checked by the same person
who reads the report. Putting it on the payload makes the guarantee
inspectable, which is the only way it can be maintained.

**What is additive here and what is not.** The report gains a 事实层 section
when the analysis produced one and is byte-identical when it did not. No
existing key changes name, type or meaning. The one place that cannot be done
additively is the session payload, because
:func:`paperwhisperer.sessions.store.build_session_payload` copies a fixed list
of keys and does not know about these two; that is a change to a module this
task does not own, and it is written down below rather than worked around by
writing the keys into the payload dict after the fact, which would produce a
session file whose contents no longer match the function that claims to build
them.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Any
from urllib.parse import quote

from paperwhisperer.analysis import paper_facts
from paperwhisperer.analysis.orchestrator import (
    DocumentAnalyzer,
    build_analysis_metadata,
    build_parse_provenance,
)
from paperwhisperer.analysis.sections import SECTION_KEYS
from paperwhisperer.core import config
from paperwhisperer.core.text import build_session_id, now_iso
from paperwhisperer.ingestion.latex import (
    compare_latex,
    find_inline_math,
    latex_defects,
)
from paperwhisperer.ingestion.schema import Element, PaperFacts
from paperwhisperer.sessions.store import (
    build_document_excerpt,
    build_session_payload,
    generate_session_token,
    get_session_document_content,
    write_session_payload,
)

logger = logging.getLogger(__name__)


def attach_document_identity(
    result: dict,
    parse_outcome: Any = None,
) -> dict:
    """Put the *document's own* id on the result, and never a session id.

    A citation chip addresses ``GET /v1/documents/{document_id}/evidence/
    {evidence_id}``, so an analysis that reaches the front end without a
    document id leaves every chip unaddressable -- the card renders, the chip
    looks clickable, and clicking it has nothing to point at. The failure is
    invisible until somebody clicks a chip, which is exactly the kind of defect
    that ships.

    The id is read from ``ParseOutcome.document.document_id``: the id the
    canonical schema minted for the parsed document. It is deliberately *not*
    the session id. A session is a conversation about a document, a document
    is the file; reusing one for the other produces citations that resolve to
    the wrong place rather than to none, and a citation that resolves to the
    wrong place is the more dangerous of the two because it looks like it
    worked. A caller may pass ``document_id`` explicitly (the fact-layer path
    does, because it needs one before the pipeline runs); otherwise the
    outcome's document is the only source, and an absent one leaves the field
    empty rather than guessing.
    """
    document = getattr(parse_outcome, "document", None)
    document_id = str(getattr(document, "document_id", "") or "")
    if not document_id:
        quality = getattr(parse_outcome, "quality", None)
        document_id = str(getattr(quality, "document_id", "") or "")
    if document_id:
        result["document_id"] = document_id
    return result


def attach_parse_quality(result: dict, parse_outcome: Any = None) -> dict:
    """Add the ``ParseQualityReport`` to the result's parse provenance.

    ``build_parse_provenance`` in :mod:`paperwhisperer.analysis.orchestrator`
    reports who parsed the file and whether it went to the cloud, but it does
    not carry the quality verdict -- and that verdict is part of what T3.4's
    acceptance criterion asks the user to know ("用户能明确知道文档被谁解析、
    是否经过云端"). Without it the parse-status card says "no quality report
    was returned" for every document, which reads as a gap in the parser when
    it is a gap in this function.

    This adds the field at the one seam this task owns, on the result the API
    returns, in the shape the report's own ``to_dict`` produces. The
    alternative -- editing ``build_parse_provenance`` -- is a file this task
    does not own, and the change there is a one-liner that the orchestrator's
    owner should apply when that migration lands:

    .. code-block:: python

        # in paperwhisperer/analysis/orchestrator.py, build_parse_provenance
        "quality": outcome.quality.to_dict() if outcome.quality is not None else None,

    A report is a whole value, not a summary: the quality gate's own docstring
    is explicit that there is no score, and a reader who wants to know *which*
    check failed needs the flags, not a single number. So the full report goes
    on, and an absent one is ``None`` rather than an empty dict that would
    render as a clean bill of health.
    """
    parse = result.get("parse")
    if not isinstance(parse, dict) or parse_outcome is None:
        return result
    quality = getattr(parse_outcome, "quality", None)
    parse["quality"] = quality.to_dict() if quality is not None else None
    return result


#: The three answers ``result["document_registration"]["status"]`` can give.
#:
#: Named rather than spelled inline because a caller branches on them, and a
#: typo in a bare string is a silent behaviour change. ``REGISTERED`` is the
#: only one of the three that means a citation will open.
REGISTRATION_REGISTERED = "registered"
REGISTRATION_SKIPPED = "skipped"
REGISTRATION_FAILED = "failed"

#: What the reader is told when the analysis path could not index the document.
#:
#: Two sentences, and both are load-bearing. The first says what broke; the
#: second says what did **not**, because the reader is looking at a complete
#: summary and the thing that failed is invisible to them. A notice that only
#: named the failure would read as "your analysis is unreliable", which is the
#: one claim this notice must not make -- the analysis completed and is
#: returned in full either way.
REGISTRATION_FAILED_NOTICE = (
    "分析结果本身完整，但这份文档未能写入检索索引：其中的引用可能打不开原文。"
    "可以在索引恢复后重新分析，或改用 /v1/documents 上传同一份文件来建立索引。"
)

#: The notice for a parse that produced no structured document at all.
#:
#: Distinct from :data:`REGISTRATION_FAILED` because nothing was attempted and
#: nothing failed: there was no document to write. Saying "索引失败" here would
#: blame a component that was never asked to do anything.
REGISTRATION_SKIPPED_NOTICE = (
    "本次解析没有产出结构化文档，因此没有可登记的原文：引用无法打开原文。"
)


def register_analysis_document(result: dict, parse_outcome: Any = None) -> dict:
    """Put the analysed document in the store, so the id this result publishes resolves.

    :func:`attach_document_identity` publishes ``ParseOutcome.document.document_id``
    on the result, and its own docstring says why: "a citation chip addresses
    ``GET /v1/documents/{document_id}/evidence/{evidence_id}``". A citation chip
    can only address a document the Evidence API can *find*, and the Evidence API
    only finds documents the store holds. Publishing the id without registering
    the document is therefore not a degraded citation -- it is an unaddressable
    one: after ``/api/analyze`` -> ``/api/ask/stream`` -> click, the store had
    never heard of the document and every chip answered ``404
    document_not_found``.

    **The registration is :func:`paperwhisperer.api.routes_v1.persist_document`,
    not a second implementation of it.** That function already owns the chunk
    write, the lexical write and the vector write, and re-calling it here would
    be a second answer to "is this document retrievable yet". It is imported
    inside this function because ``routes_v1`` imports this module; the same
    cycle already forces every other stage-level import in this codebase to be
    deferred, and this one is no different.

    **Idempotence is inherited, not re-implemented.** ``persist_document`` calls
    ``index_document_for_retrieval``, whose early return -- *before* chunking and
    before any embedding call -- fires when the document is already indexed at
    the current index version and its vectors are present. Re-running an analysis
    over a document that is already registered therefore re-reads a row and
    returns; it does not pay for a second embedding of the same paper. The guard
    lives with the code that would otherwise have to be duplicated, which is the
    only place a reader changing ``index_version`` will look for it.

    **Where it runs is the whole latency contract.** ``finalize_analysis_result``
    calls it after the six sections are generated and the Markdown report is on
    disk, and before the session is written. Not earlier: the job spine runs
    ``indexing`` *before* ``analyzing``, so an embedding pass earlier in the
    pipeline would make the reader wait on it before a single word of their
    summary existed. Not later: the export route renders from the stored session
    rather than from this dict, so a notice added afterwards would never reach
    an exported report whose every citation link is about to 404.

    **It can never fail the analysis.** Every side effect a reader came for is
    already on disk by the time this runs, every exception is caught and logged,
    and the outcome is a notice. There is no path from a failed embedding to a
    missing summary.

    **The notice rides ``result["parse"]["warnings"]`` deliberately.** That list
    is the one channel the shipped interface renders as reader-facing notices --
    ``parseStatus.js`` renders each entry as a ``<li>`` under the Parse Status
    card -- and this task does not own the templates. It is a poor home: the
    failure was not in the parse, and the honest fix is a dedicated field on
    that card. Until then a reader is told the truth rather than left holding a
    chip that silently 404s, which is the trade this function makes knowingly.
    A result with no ``parse`` block at all gets no list entry -- the block is
    not invented for it -- and carries the notice on ``document_registration``
    alone, which is where a client reads it.

    The store is not a parameter. It is resolved from
    :func:`paperwhisperer.api.routes_evidence.get_store_bundle` -- the same
    process-wide bundle the Evidence API itself reads -- because a document
    written into some *other* bundle is a document the citation click still
    cannot find, and taking a bundle as an argument would make it possible to
    get that wrong for the sake of a convenience no caller has.
    """
    # Only ever append to a ``parse`` block that already exists, and for the same
    # reason ``attach_parse_quality`` returns early when it does not: an empty
    # dict is what the orchestrator returns when there was no parse, and a
    # front end reads its emptiness as "no parse information was reported"
    # (``parseStatus.js`` keys on ``Object.keys(parse).length``). Inventing the
    # block here would turn that into a card whose only content is our notice.
    parse = result.get("parse")
    if not isinstance(parse, dict):
        parse = None
    document = getattr(parse_outcome, "document", None)
    document_id = str(getattr(document, "document_id", "") or "")

    if document is None or not document_id:
        # Nothing to write, so nothing failed. Its own status and its own
        # sentence, rather than folded into ``failed``: a caller deciding
        # whether to retry must not retry these two the same way.
        result["document_registration"] = {
            "status": REGISTRATION_SKIPPED,
            "document_id": document_id,
            "chunk_count": 0,
            "index_version": "",
            "code": "no_parsed_document",
            "notice": REGISTRATION_SKIPPED_NOTICE,
        }
        if parse is not None:
            parse.setdefault("warnings", []).append(REGISTRATION_SKIPPED_NOTICE)
        return result

    try:
        from paperwhisperer.api.routes_evidence import get_store_bundle
        from paperwhisperer.api.routes_v1 import parse_report, persist_document

        stores = get_store_bundle()
        record = persist_document(stores, document, parse_report(parse_outcome))
        # The count the reader is given is the store's answer, not the indexer's
        # return value: this is how many passages a citation can actually be
        # resolved against, which is the number that matters here.
        chunks = list(stores.metadata.list_chunks(record.document_id))
    except Exception:  # noqa: BLE001 - an analysis must survive a side task
        logger.warning(
            "分析已完成，但为文档 %s 建立检索索引失败；分析结果本身不受影响。",
            document_id,
            exc_info=True,
        )
        result["document_registration"] = {
            "status": REGISTRATION_FAILED,
            "document_id": document_id,
            "chunk_count": 0,
            "index_version": "",
            "code": "document_indexing_failed",
            "notice": REGISTRATION_FAILED_NOTICE,
        }
        if parse is not None:
            parse.setdefault("warnings", []).append(REGISTRATION_FAILED_NOTICE)
        return result

    result["document_registration"] = {
        "status": REGISTRATION_REGISTERED,
        "document_id": document_id,
        "chunk_count": len(chunks),
        "index_version": str(record.index_version or ""),
        "code": "",
        # No notice on the success branch, and that is information rather than
        # an omission: a reader whose citations open needs nothing said about
        # how they came to open.
        "notice": "",
    }
    return result

#: Where the run record has to be written, and where the fact layer has to be
#: written, once T2.2 lands. Stated here rather than only in
#: :meth:`paperwhisperer.analysis.paper_facts.AnalysisRun.as_analysis_run_columns`
#: because this is the function that owns the *side effect*, and the columns
#: docstring owns the *shape*; a reader looking for the write site should not
#: have to know which module a dataclass lives in.
ANALYSIS_RUN_PERSISTENCE_SITE = """
T2.2 must persist ``AnalysisRun.as_analysis_run_columns()`` as one row per run
in ``analysis_runs``, written by :func:`finalize_analysis_result` at the same
moment it writes the session file. Two things about that write site are load
-bearing.

*The fact layer payload goes on ``documents``, not here.* ``MetadataStore``
already declares ``put_facts`` / ``get_facts`` and the SQLite tier already
implements them, so the fact layer itself is a documents-table concern. This
row is its provenance: which prompt, which model, which evidence version and
what it cost to produce.

*The legacy columns are written on the same row.* ``legacy_llm_calls`` and
``legacy_input_tokens`` are the cost of the path this run replaced, computed
from the same document by :func:`paperwhisperer.analysis.paper_facts.legacy_call_budget`.
Recording them is what turns T1.8's acceptance criterion -- "cost went down,
here is the data" -- from a one-off measurement into something answerable from
production rows forever. A baseline that lives only in a commit message is a
baseline that is wrong the moment the legacy path changes.
"""


# --------------------------------------------------------------------------
# T3.5 — export provenance
# --------------------------------------------------------------------------
#
# An exported report is the one artefact that leaves the application, and it is
# the one nobody re-runs. "导出物可追溯生成条件" therefore has to be satisfied by
# the *file*: someone opening it six months later must be able to tell what
# produced it and from which version of which document, without the server.
#
# Three rules govern everything below, and each of them exists because the
# alternative is a file that looks traceable and is not.
#
# **A citation keeps its identity through the export.** The link is written
# twice — once into the Evidence API, which is machine-checkable, and once into
# the application as a deep link, which is the one a human can follow — and
# both are built from the same path template. A citation rendered as prose
# with no link is a claim with no way back to its source, which is the
# property the whole product is arranged around and the one an export is most
# likely to lose, because a Markdown file has nowhere to put a chip.
#
# **A direct quote stays a direct quote and a paraphrase never becomes one.**
# This is the T1.6 release gate at its weakest link. In the application the
# distinction is enforced by ``claims.js`` refusing to emit the quotation-mark
# constants; an exported file has no such guard, and the reader downstream —
# a co-author, a reviewer's PDF, a ``grep`` for "…" — has no way to tell which
# sentences are the paper's. So every entry in the citation index carries an
# explicit ``presentation``, and the quotation marks are emitted in exactly one
# place, on the branch a span the verifier aligned is the only way to reach.
#
# **Unknown is written, not omitted.** Every provenance field renders as
# ``（未记录）`` when the run did not record it. A field that silently
# disappears reads as "not important"; a field that is explicitly empty reads
# as "this run did not record it", and only the second is information. This
# matters most for the fields T2.2 has not persisted yet — see
# :func:`build_export_provenance` for exactly which ones those are.

#: The Evidence API path, as the exported citation link writes it.
#:
#: The same string lives in ``templates/static/js/config.js`` as
#: ``EVIDENCE_API_PATH`` (with camelCase placeholder names, because JavaScript).
#: There is no shared module across the language boundary, so the two are
#: pinned by a test that reads both files: a link an export advertises and a
#: path the server serves are one contract, and only a test comparing them
#: notices when they part.
EVIDENCE_API_PATH_TEMPLATE = "/v1/documents/{document_id}/evidence/{evidence_id}"

#: Where an application is reachable for the human-facing link. Relative by
#: default — an export that hard-codes ``localhost`` is a link that resolves
#: for exactly one person on exactly one machine — and overridable by the
#: caller, which is what a deployment behind a hostname needs.
EXPORT_APP_BASE_URL = ""

#: What a field the run did not record says in an export. Chinese, because it
#: is user-facing, and written as a value rather than as a gap so a reader can
#: copy the field into a form.
EXPORT_UNKNOWN_VALUE = "（未记录）"

#: Presentation labels, keyed the way ``rag/citations.py`` and ``claims.js``
#: key them. A label an export does not recognise falls through to the raw
#: value and never to the direct-quote one: an unknown presentation is
#: unknown, and unknown is not evidence.
EXPORT_PRESENTATION_LABELS = {
    "direct_quote": "直接引语（已逐字回源校验）",
    "paraphrase": "转述（未逐字校验，不加引号）",
}

EXPORT_STATUS_LABELS = {
    "extracted": "已抽取并有引用支撑",
    "needs_review": "未能通过证据校验，需人工确认",
    "not_found": "论文未说明",
}

#: Mirrors ``EXPORT_FACT_FIELD_LABELS`` in ``config.js``. Presentation only: a
#: field missing from this table still exports under its raw ``field_name``
#: rather than being dropped, because a dropped claim is worse than an
#: untranslated one.
EXPORT_FACT_FIELD_LABELS = {
    "research_question": "研究问题",
    "contributions": "核心贡献",
    "method": "方法",
    "datasets": "数据集",
    "metrics": "评价指标",
    "main_results": "主要结果",
    "limitations": "局限性",
    "reproducibility": "可复现性",
}

#: How a formula in an export may be presented. Same two-branch rule as
#: :data:`EXPORT_PRESENTATION_LABELS`, and the same reason: the T1.6 release
#: gate says a verified direct quote may be marked as the paper's words, and
#: the ``formula_dense`` corpus category exists to test the same gate on the
#: thing it is most often quietly wrong about. A formula the paper contains and
#: a formula the model produced are different claims, and the reader cannot
#: tell them apart from a ``$…$`` on its own.
EXPORT_FORMULA_PRESENTATION_LABELS = {
    "paper_formula": "论文原式（已回源比对）",
    "model_formula": "模型生成的公式（未在论文中回源比对）",
}

#: What an unverified formula is, in a sentence a reader can act on. The point
#: of naming it is that a formula in an export which nobody could check is not
#: a small footnote — it is the reader's only signal that a piece of maths in
#: the summary may be the model's own.
EXPORT_FORMULA_UNVERIFIED_NOTE = (
    "该公式出现在我们的表述里，论文中没有找到与之对应的原式，"
    "因此它不受回源校验保护，请以论文为准。"
)

#: What a formula the source parser could not read looks like, said plainly
#: rather than as a formula that differs from the model's.
EXPORT_FORMULA_UNCOMPARABLE_NOTE = (
    "论文中的这一处公式没有解析出可比较的 LaTeX（可能是以图片形式排版的），"
    "无法判定是否同一个公式。"
)

#: The other way a comparison cannot succeed: the source was transcribed from a
#: text layer that dropped the structure. Named separately from the note above
#: because the reader's next action differs — for a picture the formula is
#: simply not available, whereas here the page shows it perfectly well and only
#: this build's parser lost the subscripts, so a reader with the PDF in front
#: of them can check it and this export cannot.
EXPORT_FORMULA_FLATTENED_NOTE = (
    "论文原式是从 PDF 文本层转写的，下标与上标等结构在解析时已经丢失，"
    "因此无法逐字回源比对。论文里的公式本身是完整的，请以论文为准："
    "这不等于我们生成的公式与论文不同。"
)

#: Defect names as a reader sees them. A malformed formula is still exported —
#: dropping it would produce a file that is silently missing an equation — but
#: it is exported with its problem named, because an equation that will not
#: typeset and an equation that typesets wrong look the same otherwise.
EXPORT_FORMULA_DEFECT_LABELS = {
    "unbalanced_braces": "花括号没有配对",
    "stray_closing_brace": "多出一个右花括号",
    "unclosed_environment": "环境没有 \\end",
    "dangling_command": "末尾是残缺的命令",
    "no_latex_content": "没有可排版的 LaTeX 内容",
    "unbalanced_delimiters": "行内公式的 $ 没有闭合",
    "mismatched_delimiters": "公式的 $ 与 $$ 配对不一致",
    "ambiguous_delimiters": "公式定界符连续出现三个以上，模棱两可",
}

#: Human-facing text for where a formula's display/inline bit came from. A
#: report that says "行内公式" and a report that says "行内公式（按版面推断）" are
#: making different claims, and the second is the honest one whenever the
#: parser did not actually distinguish them.
EXPORT_DISPLAY_SOURCE_LABELS = {
    "backend": "解析器已区分",
    "layout": "按版面推断",
    "latex": "公式自身已标注（\\begin 环境或 $$ 定界符）",
    "unknown": "来源未区分，暂按块级渲染",
}

#: Display, inline, or unknown — the three-row table a reader can be shown
#: without the table claiming a distinction the parser never made.
EXPORT_DISPLAY_LABELS = {
    True: "块级公式",
    False: "行内公式",
}


def _evidence_ids(value: Any) -> list[str]:
    """Normalise an evidence-id field into a clean list of ids.

    Accepts a list, a tuple or a comma-separated string, because all three
    shapes appear in the same payload: ``Fact.evidence_ids`` is a list,
    ``VerifiedQuote.evidence_id`` is a single string, and a model that cited
    two passages for one field writes them comma-separated into that string.
    Reading only the list shape would silently drop every second citation of a
    multi-evidence field, and a dropped citation is a claim whose reader cannot
    check it.
    """
    if value is None:
        return []
    raw = list(value) if isinstance(value, (list, tuple)) else str(value).split(",")
    return [str(item).strip() for item in raw if str(item).strip()]


def _fact_label(field_name: str) -> str:
    return EXPORT_FACT_FIELD_LABELS.get(field_name, field_name or "未命名字段")


def evidence_api_url(evidence_id: str, document_id: str = "", base_url: str = "") -> str:
    """The addressable URL of one citation, for an exported file.

    ``base_url`` prefixes the path; empty (the default) yields a root-relative
    URL, which is the right answer for a file opened on the same host and the
    least-wrong one otherwise — it never claims to live somewhere it might not.
    """
    path = EVIDENCE_API_PATH_TEMPLATE.format(
        document_id=quote(str(document_id or ""), safe=""),
        evidence_id=quote(str(evidence_id or ""), safe=""),
    )
    return f"{base_url.rstrip('/')}{path}" if base_url else path


def evidence_deep_link(evidence_id: str, document_id: str = "", base_url: str = "") -> str:
    """The application URL that opens the evidence panel on this citation.

    The fragment names the two ids and nothing else, matching exactly what
    ``evidence.js``'s ``readDeepLinkTarget`` reads — the parameter *names* are
    the shared contract, and ``EVIDENCE_DEEP_LINK_PARAMS`` in ``config.js`` is
    the other end of it. An export whose links point only at the API is
    checkable but not *usable*, and a reader who cannot follow a link has to
    take the provenance on trust, which is the thing the export exists to
    remove.
    """
    fragment = f"#evidence={quote(str(evidence_id or ''), safe='')}"
    if document_id:
        fragment += f"&document={quote(str(document_id or ''), safe='')}"
    root = base_url or EXPORT_APP_BASE_URL
    return f"{root.rstrip('/')}/{fragment}" if root else f"/{fragment}"


def _iter_result_facts(result: dict) -> list[dict]:
    """Every fact in a result's fact layer, in field order.

    Reads the serialized ``paper_facts`` payload rather than rebuilding a
    :class:`PaperFacts`, because the export must report what the result
    *actually carried* — including the ``__not_stated_fields__`` bookkeeping
    that :meth:`PaperFacts.from_dict` ignores, and including a run that stored
    a field the current schema no longer knows. Round-tripping through the
    typed model would silently drop exactly the unusual payloads an export
    exists to preserve.
    """
    payload = result.get("paper_facts")
    if not isinstance(payload, dict):
        return []
    facts: list[dict] = []
    for key, value in payload.items():
        if key.startswith("__"):
            continue
        items = value if isinstance(value, list) else [value]
        for item in items:
            if isinstance(item, dict):
                facts.append(item)
    return facts


def build_citation_index(
    result: dict, base_url: str = "", app_base_url: str = ""
) -> list[dict[str, Any]]:
    """Every claim in this analysis that carries a citation, with its kind.

    The two presentations arrive in the payload in two different places, and
    this function is the join between them. ``fact_extraction.verified_quotes``
    holds the spans the verifier aligned inside a cited passage — the paper's
    own characters, which may carry quotation marks. ``paper_facts`` holds
    every fact, and a fact's ``value`` is *our* sentence about the paper, which
    may not. A fact is a direct quote exactly when one of its evidence ids
    appears in a verified quote for the same field; otherwise it is a
    paraphrase, and it is labelled as one no matter how quote-shaped it reads.

    An evidence id that resolves to no verified quote is still emitted. Dropping
    it would produce an index that looks complete while omitting precisely the
    entries a reader needs to check, and the dropped citation would be
    invisible rather than flagged.
    """
    document_id = str(result.get("document_id") or "")
    report = result.get("fact_extraction")
    verified = [item for item in (report.get("verified_quotes") or []) if isinstance(item, dict)] \
        if isinstance(report, dict) else []

    # field_name + evidence_id -> the quote the verifier aligned. A model may
    # cite several passages for one field, and any one of them aligning is what
    # makes that field's text quotable for *that* evidence id.
    aligned: dict[tuple[str, str], dict[str, Any]] = {}
    for item in verified:
        field_name = str(item.get("field_name") or "")
        for evidence_id in _evidence_ids(item.get("evidence_id")):
            aligned[(field_name, evidence_id)] = item

    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    def _add(entry: dict[str, Any]) -> None:
        key = (entry["field_name"], entry["presentation"], entry["text"])
        if key not in seen:
            seen.add(key)
            entries.append(entry)

    for item in verified:
        quote = str(item.get("quote") or "").strip()
        if not quote:
            continue
        for evidence_id in _evidence_ids(item.get("evidence_id")):
            _add({
                "field_name": str(item.get("field_name") or ""),
                "label": _fact_label(str(item.get("field_name") or "")),
                "text": quote,
                "presentation": "direct_quote",
                "presentation_label": EXPORT_PRESENTATION_LABELS["direct_quote"],
                "status": "verified",
                "status_label": "已逐字回源校验",
                "evidence_id": evidence_id,
                "api_url": evidence_api_url(evidence_id, document_id, base_url),
                "deep_link": evidence_deep_link(evidence_id, document_id, app_base_url),
            })

    for fact in _iter_result_facts(result):
        text = str(fact.get("value") or "").strip()
        evidence_ids = _evidence_ids(fact.get("evidence_ids"))
        if not text or not evidence_ids:
            continue
        field_name = str(fact.get("field_name") or "")
        status = str(fact.get("extraction_status") or "")
        for evidence_id in evidence_ids:
            match = aligned.get((field_name, evidence_id))
            presentation = "direct_quote" if match else "paraphrase"
            _add({
                "field_name": field_name,
                "label": _fact_label(field_name),
                # A matching verified quote replaces the fact's own wording, so
                # the exported text is the paper's characters rather than ours
                # wearing a quote's label. Falling back to the value while
                # *labelling* it a direct quote is the exact flattening the gate
                # names, so this falls the other way.
                "text": str(match.get("quote") or "").strip() if match else text,
                "presentation": presentation,
                "presentation_label": EXPORT_PRESENTATION_LABELS[presentation],
                "status": status,
                "status_label": EXPORT_STATUS_LABELS.get(status, status or EXPORT_UNKNOWN_VALUE),
                "evidence_id": evidence_id,
                "api_url": evidence_api_url(evidence_id, document_id, base_url),
                "deep_link": evidence_deep_link(evidence_id, document_id, app_base_url),
            })

    return entries


# --------------------------------------------------------------------------
# T0.2 — the formula index, and the gate on it
# --------------------------------------------------------------------------
#
# **A formula is a claim, and it is the claim a language model is most able to
# make by accident.** A quoted sentence can be checked character by character
# against the page. A formula cannot: the parser transliterated Unicode maths
# out of a text layer, the model wrote LaTeX from its own reading of the same
# line, and the two spellings differ in ways that are meaningless to a reader
# and fatal to a byte comparison. A gate that only passes on an exact echo
# fails almost always, and a gate that fails almost always is a gate readers
# learn to ignore — which is worse than having no gate, because it still looks
# like a gate.
#
# So the gate is *comparative*, not *exact*. :mod:`paperwhisperer.ingestion.latex`
# states which differences are noise (ligatures, ``\dfrac`` versus ``\frac``,
# ``α`` versus ``\alpha``, a leaked ``/greaterorequalslant``, a trailing equation
# number) and returns one of four verdicts, of which only ``different`` may be
# reported as an invented formula. An ``uncomparable`` verdict — a formula the
# parser recovered as an image, with no LaTeX to compare — is deliberately
# *not* a failure: reporting it as one would accuse the model of inventing a
# formula the parser simply could not read.
#
# **The mark is structural, exactly as quotation marks are.** In Markdown a
# verified formula is typeset with ``$``/``$$`` delimiters and an unverified one
# is an inline code span; in HTML a verified one is a ``<div>`` carrying
# ``data-presentation="paper_formula"`` and an unverified one is a ``<code>``;
# in DOCX a verified one takes the Quote style and an unverified one does not.
# There is no code path from an unverified formula to a maths delimiter, and
# no code path from an unverified formula to a quote style. The distinction
# survives a stylesheet that fails to load, which is the same property the
# citation gate claims and the same reason it is a property of an element type
# rather than of punctuation.
#
# **A source formula the model never used is still exported.** The section is
# an inventory of the paper's mathematics, not a log of what the model
# happened to repeat. An equation the model ignored is precisely the one a
# reader checking the summary cannot check for themselves.


def _source_formulas(document: Any) -> list[Element]:
    """Every formula element in a parsed document, in reading order."""
    if document is None:
        return []
    elements = getattr(document, "elements", None) or []
    return [element for element in elements if getattr(element, "role", "") == "formula"]


def _formula_entry(
    *,
    latex: str,
    display: bool,
    display_source: str,
    presentation: str,
    status: str,
    verdict: str = "",
    defects: tuple[str, ...] = (),
    note: str = "",
    element_id: str = "",
    citation_label: str = "",
    field_name: str = "",
    evidence_id: str = "",
    document_id: str = "",
    base_url: str = "",
    app_base_url: str = "",
    mentioned: bool = True,
) -> dict[str, Any]:
    """One row of the formula index.

    Every field a renderer needs and nothing a renderer has to compute, so the
    three formats cannot reach different conclusions about the same formula
    from different parts of the payload. The LaTeX a renderer prints is
    ``latex``, verbatim: a renderer that re-derived it, re-delimited it or
    "tidied" it would be a renderer that has opinions about the paper's
    mathematics, and three renderers with three opinions is three exports that
    disagree.
    """
    notes = [note] if note else []
    notes.extend(
        f"{EXPORT_FORMULA_DEFECT_LABELS.get(defect, defect)}"
        for defect in defects
        if defect in EXPORT_FORMULA_DEFECT_LABELS
    )
    return {
        "latex": latex,
        "display": bool(display),
        "display_source": display_source or "unknown",
        "display_label": (
            EXPORT_DISPLAY_LABELS[bool(display)]
            + f"（{EXPORT_DISPLAY_SOURCE_LABELS.get(display_source or 'unknown', '来源未记录')}）"
        ),
        "presentation": presentation,
        "presentation_label": EXPORT_FORMULA_PRESENTATION_LABELS.get(presentation, presentation),
        "status": status,
        "status_label": _FORMULA_STATUS_LABELS.get(status, status),
        "verdict": verdict,
        "defects": tuple(defects),
        "note": note,
        "notes": tuple(notes),
        "element_id": element_id,
        "citation_label": citation_label,
        "field_name": field_name,
        "evidence_id": evidence_id,
        "mentioned": mentioned,
        "api_url": (
            evidence_api_url(evidence_id, document_id, base_url) if evidence_id else ""
        ),
        "deep_link": (
            evidence_deep_link(evidence_id, document_id, app_base_url) if evidence_id else ""
        ),
    }


_FORMULA_STATUS_LABELS = {
    "verified": "已与论文中的原式回源比对",
    "unverified": "未在论文中找到对应原式",
    "uncomparable": "论文原式结构在解析时丢失，未能回源比对",
    "source_only": "论文原式，本次分析未引用",
}


def build_formula_index(
    result: dict,
    *,
    document: Any = None,
    base_url: str = "",
    app_base_url: str = "",
) -> list[dict[str, Any]]:
    """Every formula this analysis can show a reader, and how far it is trusted.

    Two sources, joined. The *document*'s formulas are what the paper actually
    contains. The *model's* formulas are what a summary, a quote or a fact
    value wrote into the answer, found by
    :func:`~paperwhisperer.ingestion.latex.find_inline_math` and never dropped —
    an unbalanced ``$`` yields an entry with a defect, because a formula the
    reader cannot see is worse than one they can see flagged.

    A model formula becomes a ``paper_formula`` **only** when the comparison
    returned ``identical`` or ``equivalent``. When it does, the exported LaTeX
    is the *document's*, not the model's: the paper's characters are the ones
    worth showing, and swapping them in is the formula-index analogue of
    :func:`build_citation_index` replacing a fact's own wording with the
    verified quote.

    ``document`` is optional and its absence is meaningful, not degrading. With
    no document there is nothing to compare against, so no model formula can be
    verified and every one of them exports as unverified. That is the correct
    answer rather than a pessimistic one: a claim that was never checked is
    unverified, and the release gate asks whether it may be *rendered* as
    checked.
    """
    document_id = str(result.get("document_id") or "")
    sources = _source_formulas(document)
    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    def _add(entry: dict[str, Any]) -> None:
        key = (entry["latex"], entry["presentation"], entry["evidence_id"])
        if key in seen:
            return
        seen.add(key)
        entries.append(entry)

    def _match(latex: str) -> tuple[Element | None, str]:
        for source in sources:
            verdict = compare_latex(latex, source.latex)
            if verdict.matches:
                return source, verdict.verdict.value
        return None, ""

    def _has_flattened_source() -> bool:
        """Whether any source formula lost its sub/superscripts in parsing.

        Docling 2.131.0 returns "qi = exp(zi/T) ∑ j exp(zj/T) (1)" for a
        formula whose page shows ``q_i`` — a text layer has no notion of a
        subscript, so the glyph arrives smaller and lower and the structure is
        simply gone. A model that writes ``q_i`` has therefore *recovered*
        what the parser dropped, and reporting it as an invented formula would
        be the gate blaming the model for the parser's own limitation.
        """
        return any(
            element.formula is not None and element.formula.is_structure_flattened
            for element in sources
        )

    # 1. What the model wrote. Verified quotes first, because a formula inside a
    #    span the verifier aligned is the paper's characters by construction and
    #    needs no further permission to be shown as such.
    for text, field_name, evidence_id, in _iter_model_formula_sources(result):
        for span in find_inline_math(text):
            latex = str(span.latex or "").strip()
            if not latex:
                continue
            source, verdict = _match(latex)
            if source is not None:
                _add(_formula_entry(
                    latex=source.latex,
                    display=bool(source.formula.display if source.formula else True),
                    display_source=(
                        source.formula.display_source if source.formula else "unknown"
                    ),
                    presentation="paper_formula",
                    status="verified",
                    verdict=verdict,
                    defects=source.formula.defects if source.formula else (),
                    element_id=source.element_id,
                    citation_label=_citation_label_for(source),
                    field_name=field_name,
                    evidence_id=evidence_id,
                    document_id=document_id,
                    base_url=base_url,
                    app_base_url=app_base_url,
                ))
                continue
            flattened = _has_flattened_source()
            _add(_formula_entry(
                latex=latex,
                display=span.display,
                # The delimiters are an assertion. A model told "行内用 $…$、
                # 块级用 $$…$$" (PROMPT_FORMULA_RULES) is telling us how it read
                # the layout of the line it copied from, so "latex" is the
                # honest source here rather than "unknown" -- and labelling it
                # unknown would produce the contradiction the reader can see:
                # "行内公式（来源未区分，暂按块级渲染）", which states two
                # incompatible things about the same formula.
                display_source="latex",
                presentation="model_formula",
                status="uncomparable" if flattened else "unverified",
                note=(
                    EXPORT_FORMULA_FLATTENED_NOTE if flattened
                    else EXPORT_FORMULA_UNVERIFIED_NOTE
                )
                + (f"（{span.defect}）" if span.defect else ""),
                defects=_span_defects(latex, span.defect),
                field_name=field_name,
                evidence_id=evidence_id,
                document_id=document_id,
                base_url=base_url,
                app_base_url=app_base_url,
            ))

    # 2. What the paper has and the model never used.
    for source in sources:
        if not source.latex.strip():
            continue
        _add(_formula_entry(
            latex=source.latex,
            display=bool(source.formula.display if source.formula else True),
            display_source=source.formula.display_source if source.formula else "unknown",
            presentation="paper_formula",
            status="source_only",
            defects=source.formula.defects if source.formula else (),
            element_id=source.element_id,
            citation_label=_citation_label_for(source),
            document_id=document_id,
            mentioned=False,
        ))
    return entries


def _span_defects(latex: str, delimiter_defect: str = "") -> tuple[str, ...]:
    """The LaTeX defects of a span, plus its delimiter defect if it had one.

    Both kinds travel together because a reader needs both: a formula whose
    braces do not pair is a different problem from one whose ``$`` never
    closed, and the second is a problem the *model* introduced while the first
    may have come from the parser. Collapsing them into one list would make the
    export say "LaTeX 有问题" about a formula that is perfectly well formed and
    merely unterminated.
    """
    defects = list(latex_defects(latex))
    if delimiter_defect and delimiter_defect not in defects:
        defects.append(delimiter_defect)
    return tuple(defects)


def _citation_label_for(element: Element) -> str:
    """``p. 7`` for an element's locator, without reaching into Evidence."""
    from paperwhisperer.ingestion.schema import Evidence

    return Evidence(locator=element.locator).citation_label


def _iter_model_formula_sources(result: dict) -> list[tuple[str, str, str]]:
    """Every stretch of model prose that could contain a formula.

    Yields ``(text, field_name, evidence_id)``. The free-text sections come
    first because they are the largest surface — a summary that reproduces an
    equation the paper never had is the exact failure this index exists to
    catch — and a fact's ``value`` after it, because a fact is a place a model
    is most likely to restate a formula from memory while writing about it.
    """
    for key in ("summary", "evaluation", "research_brief", "mindmap", "quotes"):
        text = result.get(key)
        if isinstance(text, str) and text.strip():
            yield text, key, ""
    for item in _iter_result_facts(result):
        value = item.get("value")
        if not isinstance(value, str) or not value.strip():
            continue
        ids = _evidence_ids(item.get("evidence_ids"))
        evidence_id = ids[0] if ids else ""
        yield value, str(item.get("field_name") or ""), evidence_id


def attach_formula_index(
    result: dict,
    *,
    document: Any = None,
    base_url: str = "",
    app_base_url: str = "",
) -> dict:
    """Put the formula index on the result, where the API and exports read it.

    Same shape of decision as :func:`attach_fact_layer`, and the same
    limitation: :func:`paperwhisperer.sessions.store.build_session_payload`
    copies a fixed list of keys out of the analysis, so this reaches the API
    response and the exported report but **not** the stored session file. That
    is a one-line change in a module this task does not own, and it is written
    down here rather than worked around by writing the key into the payload
    afterwards — which would produce a session file whose contents no longer
    match the function documented to build them.

    The index is attached even when it is empty, because an empty one is
    information: it says this run looked for formulas and found none, which is
    different from this build not looking.
    """
    result["formulas"] = build_formula_index(
        result, document=document, base_url=base_url, app_base_url=app_base_url
    )
    return result


def _result_formulas(result: dict) -> list[dict[str, Any]]:
    """The formula index off a result, tolerating a result that has none.

    Tolerating it is not leniency: the legacy six-section path predates this
    key entirely, and a renderer that raised on its absence would make a
    document analysed last month un-exportable today.
    """
    payload = result.get("formulas")
    if not isinstance(payload, list):
        return []
    return [entry for entry in payload if isinstance(entry, dict)]


def _formulas_section_exists(entries: list[dict[str, Any]]) -> bool:
    return bool(entries)


def build_formula_display_summary(entries: list[dict[str, Any]]) -> dict[str, int]:
    """Counts by block / inline / unknown, for the display-inventory table.

    **The unknown row is the point.** A table with a block row and an inline row
    and no third row reports "all 40 formulas are display" for a document whose
    parser never distinguished a single one of them, and a reader has no way to
    tell that from a document where the parser did. Splitting the block row by
    where the bit came from makes the guess visible instead of letting it pass
    as a finding.
    """
    summary = {"display": 0, "inline": 0, "unknown": 0, "paper": 0, "model": 0}
    for entry in entries:
        if entry.get("display_source") == "unknown":
            summary["unknown"] += 1
        elif entry.get("display"):
            summary["display"] += 1
        else:
            summary["inline"] += 1
        if entry.get("presentation") == "paper_formula":
            summary["paper"] += 1
        else:
            summary["model"] += 1
    return summary


def build_export_provenance(
    result: dict,
    original_filename: str = "",
    version: str = "",
    *,
    document_version: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Everything needed to say what produced this report, and from what.

    The columns come from
    :meth:`paperwhisperer.analysis.paper_facts.AnalysisRun.as_analysis_run_columns`
    — the same fields T2.2 will persist — so an export and a database row
    describe one run in the same vocabulary and a reader holding the row can
    match it to the file.

    **What is genuinely missing is stated rather than filled in.** The analysis
    result carries no ``document_version`` integer: that lives on the
    ``documents`` row, and :func:`finalize_analysis_result` has no store to read
    it from. Rather than write a plausible-looking number, this returns
    ``None`` and the caller may pass the real one — the renderers take it as a
    keyword argument for exactly that. The exported file then says
    ``（未记录）``, which is true, instead of a version number nobody can check.
    The same applies to the embedding cost, which T1.3's gateway records on the
    run record rather than on the result dict.
    """
    run = result.get("analysis_run") if isinstance(result.get("analysis_run"), dict) else {}
    facts = result.get("paper_facts") if isinstance(result.get("paper_facts"), dict) else {}
    parse = result.get("parse") if isinstance(result.get("parse"), dict) else {}
    cost = run.get("llm_cost")

    return {
        "app_version": version or None,
        "source_filename": original_filename or result.get("source_filename") or None,
        "document_id": result.get("document_id") or None,
        "document_version": dict(document_version) if document_version else None,
        "session_id": result.get("session_id") or None,
        "run_id": run.get("run_id") or None,
        "created_at": run.get("created_at") or None,
        "model": run.get("model") or facts.get("extraction_model") or None,
        "prompt_version": run.get("prompt_version") or facts.get("prompt_version") or None,
        "evidence_version": run.get("evidence_version") or facts.get("evidence_version") or None,
        "llm_calls": run.get("llm_calls"),
        "llm_input_tokens": run.get("llm_input_tokens"),
        "llm_output_tokens": run.get("llm_output_tokens"),
        "llm_cost": float(cost) if isinstance(cost, (int, float)) else None,
        "fact_count": run.get("fact_count"),
        "grounded_fact_count": run.get("grounded_fact_count"),
        "fact_coverage": (
            f"{run.get('grounded_fact_count', 0)}/{run.get('fact_count', 0)}"
            if run.get("fact_count")
            else None
        ),
        "not_stated_fields": list(facts.get("__not_stated_fields__") or ()) or None,
        "degradations": list(run.get("degradations") or ()) or None,
        "retrieval_record_ids": list(run.get("retrieval_record_ids") or ()) or None,
        "legacy_llm_calls": run.get("legacy_llm_calls"),
        "legacy_input_tokens": run.get("legacy_input_tokens"),
        "parsed_by": parse.get("parsed_by") or None,
        "media_type": parse.get("media_type") or None,
        "used_cloud": parse.get("used_cloud"),
        "fallback_used": parse.get("fallback_used"),
        "cache_hit": parse.get("cache_hit"),
        "parse_warnings": list(parse.get("warnings") or ()) or None,
        "has_fact_layer": bool(facts),
    }


#: Human-facing labels for the provenance block, in render order. A tuple of
#: pairs rather than a dict because the *order* is the reading order, and a
#: reader looking for the model should not have to scan a sorted list to find
#: it. ``used_cloud`` appears before ``parsed_by`` on purpose: "did this
#: document leave my machine" is the first question anybody asks of a report
#: they did not generate.
_EXPORT_PROVENANCE_ROWS: tuple[tuple[str, str], ...] = (
    ("应用版本", "app_version"),
    ("源文件", "source_filename"),
    ("文档 ID", "document_id"),
    ("文档版本", "document_version"),
    ("会话 ID", "session_id"),
    ("分析运行 ID", "run_id"),
    ("运行时间", "created_at"),
    ("模型", "model"),
    ("提示词版本", "prompt_version"),
    ("证据版本", "evidence_version"),
    ("LLM 调用次数", "llm_calls"),
    ("输入 tokens", "llm_input_tokens"),
    ("输出 tokens", "llm_output_tokens"),
    ("成本", "llm_cost"),
    ("事实覆盖", "fact_coverage"),
    ("论文未说明的字段", "not_stated_fields"),
    ("旧路径对照调用", "legacy_llm_calls"),
    ("旧路径对照 tokens", "legacy_input_tokens"),
    ("是否经过云端解析", "used_cloud"),
    ("是否发生 fallback", "fallback_used"),
    ("是否命中解析缓存", "cache_hit"),
    ("解析器", "parsed_by"),
    ("媒体类型", "media_type"),
    ("解析告警", "parse_warnings"),
    ("降级说明", "degradations"),
)


def _provenance_cell(value: Any) -> str:
    """Render one provenance value, never silently.

    Booleans become 是/否 because a raw ``True`` in a Chinese report reads as a
    Python repr; a mapping becomes readable fields rather than a repr; ``None``
    and the empty string become :data:`EXPORT_UNKNOWN_VALUE`; ``0`` and
    ``False`` do **not**, because a recorded zero is a measurement and a
    recorded ``False`` is an answer, and neither is "unknown".
    """
    if value is None or value == "":
        return EXPORT_UNKNOWN_VALUE
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, dict):
        parts = [f"{key}={val}" for key, val in value.items() if val not in (None, "")]
        return "；".join(parts) if parts else EXPORT_UNKNOWN_VALUE
    if isinstance(value, (list, tuple)):
        return "；".join(str(item) for item in value) if value else EXPORT_UNKNOWN_VALUE
    return str(value)


def render_provenance_markdown(provenance: dict[str, Any]) -> str:
    """The provenance table, in the order :data:`_EXPORT_PROVENANCE_ROWS` sets."""
    lines = [
        "> 这张表记录的是「生成这份文件的条件」。打开一个六个月前的导出件时，",
        "> 它是判断这份结论还能不能信的唯一依据；未记录的字段说明当次运行没有写入该项，",
        "> 而不是该项不重要。",
        "",
        "| 项目 | 值 |",
        "| --- | --- |",
    ]
    for label, key in _EXPORT_PROVENANCE_ROWS:
        lines.append(f"| {label} | {_provenance_cell(provenance.get(key))} |")
    return "\n".join(lines)


def render_citation_index_markdown(entries: list[dict[str, Any]]) -> str:
    """The citation index, one entry per claim that carries a citation.

    The quotation marks are emitted here and only here, on the branch that
    requires ``presentation == "direct_quote"``. A paraphrase entry cannot
    acquire one because there is no code path from a paraphrase to a mark:
    that is the release gate restated as a property of the exporter rather than
    a rule somebody has to remember.

    The table first, the text after. A reader checking provenance wants the
    index — field, kind, link — without scrolling through quotes, and a reader
    checking a specific claim wants the text next to its own evidence id. The
    index satisfies both because both rows and blocks are here, keyed the same
    way.
    """
    if not entries:
        return "本次分析没有产出带引用的结论。"

    lines = [
        "> 直接引语逐字回源校验过；转述是我们的表述，不加引号。",
        "> 两种呈现方式是分开的字段，不靠标点区分。",
        "",
        "| # | 字段 | 呈现方式 | 字段状态 | 证据链接 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for index, entry in enumerate(entries, start=1):
        link = f"[打开]({entry['deep_link']}) · [API]({entry['api_url']})"
        lines.append(
            f"| {index} | {entry['label']} | {entry['presentation_label']} | "
            f"{entry['status_label']} | {link} |"
        )

    lines.extend(["", "### 引用原文与证据链接", ""])
    for index, entry in enumerate(entries, start=1):
        body = f'> “{entry["text"]}”' if entry["presentation"] == "direct_quote" else entry["text"]
        lines.extend([
            f"#### {index}. {entry['label']}",
            "",
            f"- 呈现方式: {entry['presentation_label']}",
            f"- 字段状态: {entry['status_label']}",
            f"- 证据 ID: {entry['evidence_id']}",
            f"- 打开该引用: {entry['deep_link']}",
            f"- Evidence API: {entry['api_url']}",
            "",
            body,
            "",
        ])
    return "\n".join(lines)


# --------------------------------------------------------------------------
# The formula renderers
# --------------------------------------------------------------------------
#
# Three formats, one rule, and the rule is the same rule the citation index
# above is built on: **the mark is emitted on the verified branch only.**
#
# In Markdown a verified formula is typeset with ``$``/``$$`` and an unverified
# one is an inline code span. In HTML a verified formula is a ``<div>`` with
# ``data-presentation="paper_formula"`` and an unverified one is a ``<code>``.
# In DOCX a verified formula takes the Quote style and an unverified one does
# not. A stylesheet that fails to load, a Word reader who ignores the label,
# and a ``grep`` for ``$$`` all reach the same answer, because the answer is in
# the element type rather than in a decoration.
#
# **The same LaTeX string goes into all three.** That is not tidiness: a reader
# who copies the formula out of the Markdown and pastes it into a LaTeX
# compiler must get the expression the paper had, and the three exports must
# not disagree about what that was. Every renderer prints ``entry["latex"]``
# verbatim and adds only delimiters, so the three cannot drift.


#: The only place a maths delimiter is emitted, on the verified branch only.
#:
#: Named and isolated so the test that asserts the gate can read this function
#: and find every ``$`` an export can produce inside it — the same structural
#: assertion ``tests/test_export.py`` makes for the quotation mark.
_FORMULA_DISPLAY_DELIMITER = "$$"
_FORMULA_INLINE_DELIMITER = "$"


def _formula_body_markdown(entry: dict[str, Any]) -> str:
    """A verified formula, typeset. The only place a ``$`` is written.

    Called from the ``paper_formula`` branch of
    :func:`render_formula_index_markdown` and nowhere else, which is what makes
    "an unverified formula can never acquire a maths delimiter" a property of
    the code rather than a rule somebody has to remember.
    """
    delimiter = _FORMULA_DISPLAY_DELIMITER if entry["display"] else _FORMULA_INLINE_DELIMITER
    return f"{delimiter}{entry['latex']}{delimiter}"


def _formula_body_html(entry: dict[str, Any]) -> str:
    """A verified formula as HTML, escaping the LaTeX rather than emitting it.

    Escaped for the same reason every other model string here is: the LaTeX
    came out of a document, and ``\\<`` or a ``<`` in it must not become a tag
    in a file that leaves the application. The delimiters are written outside
    the escaped region because they are ours, not the document's.
    """
    delimiter = _FORMULA_DISPLAY_DELIMITER if entry["display"] else _FORMULA_INLINE_DELIMITER
    return (
        f"{_escape_html(delimiter)}<span class=\"pw-latex\">"
        f"{_escape_html(entry['latex'])}</span>{_escape_html(delimiter)}"
    )


def _formula_metadata_lines(entry: dict[str, Any]) -> list[str]:
    """The rows both formats print above a formula, in the same order."""
    lines = [
        f"- 公式来源: {entry['presentation_label']}",
        f"- 版式: {entry['display_label']}",
        f"- 校验状态: {entry['status_label']}",
    ]
    if entry.get("citation_label"):
        lines.append(f"- 位置: {entry['citation_label']}")
    if entry.get("element_id"):
        lines.append(f"- 元素 ID: {entry['element_id']}")
    if entry.get("evidence_id"):
        lines.append(f"- 证据 ID: {entry['evidence_id']}")
        lines.append(f"- 打开该引用: {entry['deep_link']}")
        lines.append(f"- Evidence API: {entry['api_url']}")
    for note in entry.get("notes") or ():
        lines.append(f"- 注意: {note}")
    return lines


def render_formula_index_markdown(entries: list[dict[str, Any]]) -> str:
    """The formula inventory as a Markdown section.

    Absent rather than empty when there is nothing, matching the rule the two
    traceability sections follow: a heading over an empty table reads as "we
    checked and there was nothing", which is not what a run that never looked
    means.
    """
    if not entries:
        return ""
    summary = build_formula_display_summary(entries)
    lines = [
        "## 公式与回源校验",
        "",
        "> 论文原式逐条与论文比对过；模型生成的公式不加数学定界符，只作行内代码，"
        "两种呈现方式是分开的写法，不靠排版区分。",
        "",
        f"- 论文原式: {summary['paper']} · 模型生成: {summary['model']}",
        f"- 块级: {summary['display']} · 行内: {summary['inline']} · "
        f"来源未区分: {summary['unknown']}",
        "",
        "### 公式清单",
        "",
    ]
    for index, entry in enumerate(entries, start=1):
        lines.append(f"#### {index}. {entry['latex'][:60]}")
        lines.append("")
        lines.extend(_formula_metadata_lines(entry))
        lines.append("")
        # The gate. One branch reaches a maths delimiter; the other cannot.
        if entry["presentation"] == "paper_formula":
            lines.append(_formula_body_markdown(entry))
        else:
            lines.append(f"`{entry['latex']}`")
        lines.append("")
    return "\n".join(lines)


def _formula_body_docx(document: Any, entry: dict[str, Any]) -> Any:
    """A verified formula as a DOCX paragraph. The only Quote-styled one.

    Word cannot typeset LaTeX, so the expression is written out as its source
    and the delimiters are what tell a reader it is a formula at all. The
    label above it says in words which kind it is, for the same reason the
    citation index writes 直接引语 on both branches: a reader who ignores the
    indentation still has the words.
    """
    from docx.shared import Pt

    delimiter = _FORMULA_DISPLAY_DELIMITER if entry["display"] else _FORMULA_INLINE_DELIMITER
    style = _docx_style(document, "quote")
    paragraph = (
        document.add_paragraph(f"{delimiter}{entry['latex']}{delimiter}", style=style)
        if style
        else document.add_paragraph(f"{delimiter}{entry['latex']}{delimiter}")
    )
    for run in paragraph.runs:
        run.font.size = Pt(11)
    return paragraph


def render_formula_index_html(entries: list[dict[str, Any]]) -> list[str]:
    """The formula inventory as HTML fragments.

    ``data-presentation`` is on the wrapper as well as on the class, so the
    distinction survives a stylesheet that never loads — the same reasoning
    ``render_html_traceability`` applies to a direct quote, and the reason the
    unverified branch is a ``<code>`` element rather than a paragraph with a
    class name.
    """
    if not entries:
        return []
    summary = build_formula_display_summary(entries)
    parts = [
        "<h2>公式与回源校验</h2>",
        '<p class="pw-note">论文原式逐条与论文比对过；模型生成的公式不加数学定界符，'
        "只作行内代码，两种呈现方式是分开的元素类型，不靠样式区分。</p>",
        f'<p class="pw-meta">论文原式: {summary["paper"]} · 模型生成: {summary["model"]}'
        f' · 块级: {summary["display"]} · 行内: {summary["inline"]}'
        f' · 来源未区分: {summary["unknown"]}</p>',
        "<h3>公式清单</h3>",
    ]
    for index, entry in enumerate(entries, start=1):
        parts.append(
            f'<div class="pw-formula"'
            f' data-presentation="{_escape_html(entry["presentation"])}"'
            f' data-display="{"block" if entry["display"] else "inline"}">'
        )
        parts.append(f'<p class="pw-meta">{index}. {_escape_html(entry["latex"][:60])}</p>')
        metadata = "".join(
            f"<li>{_escape_html(line[2:])}</li>" for line in _formula_metadata_lines(entry)
        )
        parts.append(f'<ul class="pw-formula-meta">{metadata}</ul>')
        # The gate. Only the paper_formula branch can reach a maths delimiter.
        if entry["presentation"] == "paper_formula":
            parts.append(f'<p class="pw-math">{_formula_body_html(entry)}</p>')
        else:
            parts.append(
                '<p><code class="pw-math-unverified">'
                f'{_escape_html(entry["latex"])}</code></p>'
            )
        parts.append("</div>")
    return parts


def render_formula_index_docx(document: Any, entries: list[dict[str, Any]]) -> None:
    """The formula inventory, appended to a DOCX document in place."""
    if not entries:
        return
    summary = build_formula_display_summary(entries)
    _docx_heading(document, "公式与回源校验")
    _docx_note(
        document,
        "论文原式逐条与论文比对过；模型生成的公式不加数学定界符，只作行内代码。"
        "Word 不能排版 LaTeX，这里写出的是公式源码。",
    )
    _docx_note(
        document,
        f"论文原式: {summary['paper']} · 模型生成: {summary['model']} · "
        f"块级: {summary['display']} · 行内: {summary['inline']} · "
        f"来源未区分: {summary['unknown']}",
    )
    _docx_heading(document, "公式清单")
    for index, entry in enumerate(entries, start=1):
        _docx_note(document, f"{index}. {entry['latex'][:60]}")
        for line in _formula_metadata_lines(entry):
            _docx_note(document, line)
        # The gate. Only the paper_formula branch can take the Quote style.
        if entry["presentation"] == "paper_formula":
            _formula_body_docx(document, entry)
        else:
            paragraph = document.add_paragraph(f"`{entry['latex']}`")
            for run in paragraph.runs:
                run.font.name = "Consolas"


def _neutralise_unverified_math(text: Any) -> str:
    """Render a model-written formula as inline code, in the Markdown report.

    The formula index gates its *own* section, but the model also writes
    mathematics into the summary, the evaluation and the brief, and those are
    rendered verbatim. A ``$$\\frac{a}{b}$$`` the model invented therefore
    appeared typeset in the middle of the 摘要, with the same weight and the
    same delimiters as an equation the paper has — which is precisely the
    failure the release gate is about, reached by a route the gate did not
    cover. A reader has no way to tell that one from a quotation, and the
    quotation they *can* tell apart is the one the gate is strongest on.

    The replacement is inline code rather than a deletion, for the reason every
    other degradation here is a replacement and never a removal: the reader
    still sees exactly the characters the model wrote, in a form that cannot be
    mistaken for the paper's typesetting. Deleting them would be a corruption
    they cannot see.

    **Markdown only, and that is not an oversight.** The HTML renderer escapes
    model text into ``<p>`` elements and the DOCX renderer writes it into
    ordinary paragraphs, so a ``$$`` there is inert — two literal characters,
    not a delimiter. There is no invented formula being typeset in those two
    formats, so there is nothing for this to prevent and a third copy of the
    rule would be a third thing to keep in step.
    """
    source = str(text or "")
    spans = find_inline_math(source)
    if not spans:
        return source
    pieces: list[str] = []
    cursor = 0
    for span in spans:
        pieces.append(source[cursor : span.start])
        latex = span.latex.strip()
        pieces.append(f"`{latex}`" if latex else f"`{span.delimiters}`")
        cursor = span.end
    pieces.append(source[cursor:])
    return "".join(pieces)


def render_markdown_report(
    result: dict,
    original_filename: str,
    version: str,
    *,
    document_version: dict[str, Any] | None = None,
    base_url: str = "",
    app_base_url: str = "",
) -> str:
    """Render the analysis as a standalone Markdown report.

    The 事实层 section is rendered by
    :func:`paperwhisperer.analysis.paper_facts.render_overview` -- the same
    function the overview card uses -- so the exported document cannot say
    something different from the screen. That is not tidiness: the report is
    the artefact people paste into a literature review, and an export that
    quietly reintroduced a second wording of the contribution would undo the
    consistency this task exists to produce, one file download later.

    **T3.5 adds two sections and changes nothing else.** 引用与证据链接 and
    生成条件与溯源 go in near the end, after the analysis and before the flat
    元信息 footer, because that is the reading order: what the paper says, then
    what backs it, then what produced this. Both are absent rather than empty
    when the analysis produced neither a fact layer nor a run record -- the old
    six-section path has neither, and a heading with an empty table under it
    would claim a traceability the file does not have.

    The keyword-only arguments are additive and defaulted, so
    :func:`finalize_analysis_result` is unaffected: it calls this with three
    positional arguments and the report gains its two sections.
    """
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed = result.get("elapsed_seconds", "N/A")

    parts = [
        "# PaperWhisperer 分析报告",
        "",
        f"> 生成时间: {generated_at}",
        f"> 源文件: {original_filename}",
        f"> 耗时: {elapsed}s",
        "",
        "---",
        "",
    ]
    fact_section = render_fact_layer_section(result)
    if fact_section:
        parts.extend([fact_section, "---", ""])
    # Every one of these is the model's own wording, so every one of them goes
    # through _neutralise_unverified_math on the way out. The 引用片段 section
    # is included even though its name promises quotes: the model writes it, and
    # until a span has been aligned by the verifier a sentence in it is a
    # paraphrase wearing a quote's heading.
    parts.extend([
        "## AI 摘要",
        "",
        _neutralise_unverified_math(result.get("summary", "")),
        "",
        "---",
        "",
        "## 引用片段",
        "",
        _neutralise_unverified_math(result.get("quotes", "")),
        "",
        "---",
        "",
        "## 思维导图",
        "",
        _neutralise_unverified_math(result.get("mindmap", "")),
        "",
        "---",
        "",
    ])
    if result.get("mermaid"):
        parts.extend(["## 视觉图谱", "", "```mermaid", result["mermaid"], "```", "", "---", ""])
    if result.get("evaluation"):
        parts.extend([
            f"## 论文评价\n\n{_neutralise_unverified_math(result['evaluation'])}\n\n---\n"
        ])
    if result.get("research_brief"):
        parts.extend([
            f"## 深度阅读简报\n\n{_neutralise_unverified_math(result['research_brief'])}\n\n---\n"
        ])
    parts.extend(render_run_section(result))
    # The formula inventory goes after the analysis and before the traceability
    # tables, for the same reading-order reason those tables come last: what the
    # paper says, then what backs it, then what produced this. It is absent
    # rather than empty when the run found no formulas, so a legacy result --
    # which has no formula index at all -- is byte-identical to what it was.
    formula_section = render_formula_index_markdown(_result_formulas(result))
    if formula_section:
        parts.extend([formula_section, "", "---", ""])
    parts.extend(render_traceability_sections(
        result,
        original_filename,
        version,
        document_version=document_version,
        base_url=base_url,
        app_base_url=app_base_url,
    ))
    parts.extend([f"## 元信息\n\n- 版本: {version}\n- 字符数: {result.get('char_count', 0)}\n"])
    return "".join(parts)


def render_traceability_sections(
    result: dict,
    original_filename: str = "",
    version: str = "",
    *,
    document_version: dict[str, Any] | None = None,
    base_url: str = "",
    app_base_url: str = "",
) -> list[str]:
    """The T3.5 sections, as report fragments, or ``[]`` when there is nothing.

    One function so the Markdown, HTML and DOCX exporters cannot disagree about
    which of the two sections exist. The condition is deliberately the *same*
    for both: a run with a fact layer has claims worth citing, and a run with
    an ``analysis_run`` has conditions worth recording. A run with neither — the
    legacy six-section path — gets no heading at all, because a report that
    carries an empty provenance table reads as "we checked and there was
    nothing" rather than "this build did not record that".

    The asymmetry is worth stating: the citation section is a claim about the
    analysis, so its absence is meaningful; the provenance section is a claim
    about the *exporter*, so its absence is a gap in this build and is
    documented at the bottom of the report instead.
    """
    entries = build_citation_index(result, base_url=base_url, app_base_url=app_base_url)
    provenance = build_export_provenance(
        result, original_filename, version, document_version=document_version
    )
    if not entries and not provenance.get("has_fact_layer"):
        return []

    parts: list[str] = []
    if entries:
        parts.extend([
            "## 引用与证据链接", "",
            render_citation_index_markdown(entries), "", "---", "",
        ])
    if provenance.get("has_fact_layer") or provenance.get("run_id"):
        parts.extend([
            "## 生成条件与溯源", "",
            render_provenance_markdown(provenance), "", "---", "",
        ])
    if not provenance.get("document_version"):
        parts.extend([
            "> 文档版本号未写入本导出件：分析结果里不携带 documents 表的版本行，",
            "> 而本模块没有可读的存储层。文档 ID 已在上方给出，可据此回查版本。", "",
        ])
    return parts


def render_fact_layer_section(result: dict) -> str:
    """The 事实层 block of the report, or ``""`` when there is no fact layer.

    Absent rather than empty on purpose: a report for an analysis run through
    the old path has no fact layer, and printing a heading with nothing under
    it would claim a guarantee the document does not have.
    """
    payload = result.get("paper_facts")
    if not isinstance(payload, dict):
        return ""
    facts = PaperFacts.from_dict(payload)
    not_stated = [
        str(paper_facts.FACT_FIELDS[name]["label"])
        for name in payload.get("__not_stated_fields__") or ()
        if name in paper_facts.FACT_FIELDS
    ]
    body = paper_facts.render_overview(facts)
    if not_stated:
        body = f"{body}\n### 论文未说明的字段\n\n" + "、".join(not_stated) + "\n"
    return f"## 事实层（唯一抽取结果，其余卡片均为其渲染）\n\n{body}"


# --------------------------------------------------------------------------
# T3.5 — the other two export formats
# --------------------------------------------------------------------------
#
# Markdown is the format the pipeline already wrote, so it got the sections
# first. HTML and DOCX exist because "export" means different things to
# different readers: HTML is what gets pasted into a notebook or a wiki, and
# DOCX is what gets emailed to a co-author and commented on in Word. A reader
# in Word is also the reader most likely to treat an exported sentence as a
# quotation, which is why the direct-quote / paraphrase distinction is a
# *structural* property in all three formats rather than a styling one:
#
#   * Markdown — a direct quote is a blockquote, a paraphrase is a paragraph.
#     The mark appears only on the quote branch.
#   * HTML — a direct quote is a ``<blockquote>``, a paraphrase is a ``<p>``,
#     and the enclosing element also carries ``data-presentation`` so the
#     distinction survives a stylesheet that does not load.
#   * DOCX — a direct quote is an indented ``Quote``-styled paragraph, a
#     paraphrase is a body paragraph, and the label above each one says which
#     it is in words.
#
# All three write their provenance through
# :func:`render_provenance_markdown` and the same row table, so the three files
# cannot disagree about what produced them.
#
# **What these renderers deliberately do not do is interpret markdown.** The
# analysis cards are markdown; a renderer that turned them into headings and
# lists would also be a renderer that could be made to emit an anchor or a
# ``<script>`` from model text, and there is no security reason to take that
# risk in a file that leaves the application. The card text is emitted as
# escaped, paragraph-split text: readable, and incapable of being anything but
# text.

_HTML_ESCAPES = {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}


def _escape_html(value: Any) -> str:
    """Escape every character that could open a tag or an attribute.

    Hand-rolled rather than ``html.escape`` because the report is assembled by
    string concatenation, and one place using a different escaper than the rest
    is one missed call. Ampersand first, by construction: this is a single
    pass over the source string, not a chain of replaces that could
    double-escape.
    """
    text = str(value if value is not None else "")
    return "".join(_HTML_ESCAPES.get(char, char) for char in text)


def _html_paragraphs(text: Any) -> str:
    """Model text as escaped paragraphs — never as HTML, never as markdown.

    Split on blank lines only. A single newline stays a newline inside the
    paragraph, because re-flowing it would change the text, and a quote's
    internal line breaks are part of what a reader is checking.
    """
    raw = str(text or "").strip()
    if not raw:
        return '<p class="pw-empty">（本次分析没有这一节的内容）</p>'
    blocks = [block.strip() for block in raw.split("\n\n") if block.strip()]
    return "\n".join(
        "<p>" + _escape_html(block).replace("\n", "<br>\n") + "</p>" for block in blocks
    )


#: A deliberately small stylesheet. The report is a document, not a theme: the
#: colours exist so that a blockquote *looks* like a quote in Word's preview
#: and in a browser's reader mode, and everything still reads correctly if the
#: whole block is deleted.
_EXPORT_HTML_STYLE = """
body{font-family:system-ui,-apple-system,"Segoe UI","Noto Sans CJK SC",sans-serif;
max-width:52rem;margin:0 auto;padding:2rem 1.25rem;line-height:1.7;color:#1c1c1e;background:#fff;}
h1{font-size:1.6rem}h2{font-size:1.15rem;margin-top:2.4rem;b
order-bottom:1px solid #e5e5ea;padding-bottom:.3rem;}
h3{font-size:1rem;margin-top:1.6rem;}
table.pw-provenance{border-collapse:collapse;width:100%;font-size:.9rem;}
table.pw-provenance th,table.pw-provenance td{border:1px solid #d1d1d6;padding:.4rem .6rem;
text-align:left;vertical-align:top;}
table.pw-provenance th{background:#f5f5f7;width:13rem;}
blockquote.pw-citation{border-left:3px solid #0a84ff;margin:
.4rem 0;padding:.2rem .9rem;color:#3a3a3c;}
div.pw-citation[data-presentation="paraphrase"]{border-left:3px solid #ff9f0a;margin:.6rem 0;
padding:.4rem .9rem;border-top:1px solid #e5e5ea;border-righ
t:1px solid #e5e5ea;border-bottom:1px solid #e5e5ea;}
p.pw-meta{font-size:.85rem;color:#6e6e73;margin:.2rem 0 .5rem;}
p.pw-note{font-size:.85rem;color:#6e6e73;border-left:3px solid #d1d1d6;padding-left:.8rem;}
p.pw-empty{color:#8e8e93;font-style:italic;}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.9em;}
div.pw-formula{border-left:3px solid #1c1c1e;margin:.8rem 0;padding:.5rem .9rem;
background:#fafafa;}
div.pw-formula[data-presentation="model_formula"]{border-left:3px solid #ff9f0a;
background:#fff8ee;}
ul.pw-formula-meta{margin:.2rem 0 .4rem;padding-left:1.1rem;font-size:.85rem;color:#6e6e73;}
p.pw-math{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;margin:.3rem 0 0;}
code.pw-math-unverified{color:#8a4b00;background:#fff3e0;padding:.1rem .3rem;}
""".strip()


def render_html_report(
    result: dict,
    original_filename: str,
    version: str,
    *,
    document_version: dict[str, Any] | None = None,
    base_url: str = "",
    app_base_url: str = "",
) -> str:
    """The report as a standalone HTML file.

    A complete document with an inline stylesheet, because the commonest
    destination for an HTML export is an email attachment or a repository — a
    file that depended on this application's CSS would arrive unstyled and
    would lose the blockquote-versus-paragraph distinction entirely, which is
    the one distinction it exists to preserve.
    """
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entries = build_citation_index(result, base_url=base_url, app_base_url=app_base_url)
    provenance = build_export_provenance(
        result, original_filename, version, document_version=document_version
    )

    parts = [
        "<!DOCTYPE html>",
        '<html lang="zh-CN">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{_escape_html(original_filename or 'PaperWhisperer 分析报告')}</title>",
        f"<style>\n{_EXPORT_HTML_STYLE}\n</style>",
        "</head>",
        "<body>",
        "<h1>PaperWhisperer 分析报告</h1>",
        f'<p class="pw-meta">生成时间: {_escape_html(generated_at)} · '
        f'源文件: {_escape_html(original_filename or "（未记录）")} · '
        f'应用版本: {_escape_html(version or "（未记录）")}</p>',
    ]

    def _section(title: str, body: Any) -> None:
        parts.append(f"<h2>{_escape_html(title)}</h2>")
        parts.append(_html_paragraphs(body))

    _section("AI 摘要", result.get("summary"))
    _section("引用片段", result.get("quotes"))
    _section("思维导图", result.get("mindmap"))
    if result.get("evaluation"):
        _section("论文评价", result["evaluation"])
    if result.get("research_brief"):
        _section("深度阅读简报", result["research_brief"])

    if result.get("mermaid"):
        parts.append("<h2>视觉图谱</h2>")
        parts.append(f'<pre class="pw-empty"><code>{_escape_html(result["mermaid"])}</code></pre>')

    fact_section = render_fact_layer_section(result)
    if fact_section:
        parts.append("<h2>事实层（唯一抽取结果，其余卡片均为其渲染）</h2>")
        parts.append(_html_paragraphs(fact_section))

    parts.extend(render_formula_index_html(_result_formulas(result)))
    parts.extend(render_html_traceability(entries, provenance))
    parts.extend(["</body>", "</html>"])
    return "\n".join(parts)


def render_html_traceability(
    entries: list[dict[str, Any]], provenance: dict[str, Any]
) -> list[str]:
    """The citation index and the provenance table, as HTML.

    The direct-quote / paraphrase distinction is carried by the element type
    and by ``data-presentation``, never by a class name a stylesheet can
    override and never by punctuation in the text. That is the same rule
    ``claims.js`` follows in the application, applied to the file.
    """
    parts: list[str] = []
    if entries:
        parts.extend([
            "<h2>引用与证据链接</h2>",
            '<p class="pw-note">直接引语逐字回源校验过；转述是我们的表述，不加引号。'
            "两种呈现方式是分开的元素类型，不靠标点区分。</p>",
        ])
        for index, entry in enumerate(entries, start=1):
            parts.append(
                '<div class="pw-citation"'
                f' data-presentation="{_escape_html(entry["presentation"])}"'
                f' data-evidence-id="{_escape_html(entry["evidence_id"])}">'
            )
            parts.append(
                f'<p class="pw-meta">{index}. {_escape_html(entry["label"])} · '
                f'{_escape_html(entry["presentation_label"])} · '
                f'{_escape_html(entry["status_label"])} · '
                f'<a href="{_escape_html(entry["deep_link"])}">打开该引用</a> · '
                f'<a href="{_escape_html(entry["api_url"])}">Evidence API</a></p>'
            )
            if entry["presentation"] == "direct_quote":
                # The release gate, in the exported bytes: a blockquote only on
                # the verified branch. Split across two literals purely for the
                # line length -- concatenating them changes nothing about the
                # output, and fusing them back would be the same string.
                parts.append(
                    '<blockquote class="pw-citation">“'
                    f'{_escape_html(entry["text"])}”</blockquote>'
                )
            else:
                parts.append(f"<p>{_escape_html(entry['text'])}</p>")
            parts.append("</div>")

    if provenance.get("has_fact_layer") or provenance.get("run_id"):
        parts.append("<h2>生成条件与溯源</h2>")
        parts.append(
            '<p class="pw-note">这张表记录的是「生成这份文件的条件」。'
            "打开一个六个月前的导出件时，它是判断这份结论还能不能信的唯一依据；"
            "未记录的字段说明当次运行没有写入该项，而不是该项不重要。</p>"
        )
        parts.append('<table class="pw-provenance"><tr><th>项目</th><th>值</th></tr>')
        for label, key in _EXPORT_PROVENANCE_ROWS:
            parts.append(
                f'<tr><th scope="row">{_escape_html(label)}</th>'
                f'<td>{_escape_html(_provenance_cell(provenance.get(key)))}</td></tr>'
            )
        parts.append("</table>")

    if not provenance.get("document_version"):
        parts.append(
            '<p class="pw-note">文档版本号未写入本导出件：分析结果里不携带 documents 表的版本行，'
            "而本模块没有可读的存储层。文档 ID 已在上方给出，可据此回查版本。</p>"
        )
    return parts


# --------------------------------------------------------------------------
# DOCX
# --------------------------------------------------------------------------

#: Word's built-in styles, used where they exist. Every one is applied through
#: a ``get`` with a fallback, because a style name is not part of python-docx's
#: compatibility promise across versions and templates: a document that raises
#: because "Intense Quote" is missing in the default template is a document
#: that cannot be exported at all, which is a worse outcome than one that is
#: quoted with a paragraph indent.
_DOCX_STYLE_FALLBACKS = {
    "title": "Title",
    "heading1": "Heading 1",
    "heading2": "Heading 2",
    "heading3": "Heading 3",
    "quote": "Intense Quote",
}


def _docx_style(document: Any, key: str) -> str | None:
    """The named Word style, or ``None`` when this template lacks it."""
    name = _DOCX_STYLE_FALLBACKS.get(key, key)
    try:
        document.styles[name]
    except (KeyError, AttributeError):
        return None
    return name


def _docx_heading(document: Any, text: str, key: str = "heading2") -> Any:
    """Add a heading, degrading to a bold paragraph if the style is missing.

    The degradation is not cosmetic: a report whose section headings are plain
    bold text still tells a reader what the sections are, whereas one whose
    headings raised is a report that does not exist.
    """
    style = _docx_style(document, key)
    paragraph = document.add_paragraph(text, style=style) if style else document.add_paragraph()
    if not style and paragraph.runs:
        paragraph.runs[0].bold = True
    return paragraph


def _docx_note(document: Any, text: str) -> Any:
    """A small grey note — the standing explanation of the file's contents."""
    paragraph = document.add_paragraph()
    run = paragraph.add_run(text)
    run.italic = True
    return paragraph


def render_docx_report(
    result: dict,
    original_filename: str,
    version: str,
    *,
    document_version: dict[str, Any] | None = None,
    base_url: str = "",
    app_base_url: str = "",
) -> bytes:
    """The report as a ``.docx`` file, as bytes.

    Uses ``python-docx``, which is already a declared dependency and is already
    imported by the legacy parser adapter — so the DOCX writer this task needs
    is a dependency the project already chose, and adding a second one (or
    hand-rolling the OOXML) would be introducing risk for no capability.

    Returns bytes rather than writing a file, because the caller decides where
    an export goes: the API wants a ``Response`` body, a test wants something
    it can unzip, and neither wants this function choosing a path.

    The direct-quote / paraphrase distinction is an *indentation and style*
    difference here rather than an element type, because Word has no
    blockquote element — so the written label is what carries it, and it is
    written on both branches, never on the paraphrase branch alone. A
    reader in Word who ignores the label still sees the indentation; a reader
    who greps the text sees the words 直接引语 or 转述 next to it.
    """
    # Imported here, not at module scope: an analysis runs on every request and
    # a DOCX export runs on almost none, so a deployment that never exports one
    # should neither pay for the import nor be broken by a broken ``python-docx``
    # install. ``python-docx`` is a declared dependency (the legacy parser
    # adapter already imports it) — this is about where the cost and the failure
    # land, not about whether it is available.
    from docx import Document
    from docx.shared import Pt

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    document = Document()
    entries = build_citation_index(result, base_url=base_url, app_base_url=app_base_url)
    provenance = build_export_provenance(
        result, original_filename, version, document_version=document_version
    )

    title_style = _docx_style(document, "title")
    if title_style:
        document.add_paragraph("PaperWhisperer 分析报告", style=title_style)
    else:
        _docx_heading(document, "PaperWhisperer 分析报告", "heading1")
    _docx_note(
        document,
        f"生成时间: {generated_at} · 源文件: {original_filename or '（未记录）'} · "
        f"应用版本: {version or '（未记录）'}",
    )

    def _section(title: str, body: Any) -> None:
        _docx_heading(document, title)
        text = str(body or "").strip()
        if not text:
            _docx_note(document, "（本次分析没有这一节的内容）")
            return
        # Blank-line-separated paragraphs, with single newlines kept as line
        # breaks inside one. A quote's internal line breaks are part of what a
        # reader is checking, so re-flowing them would change the text.
        for block in [item.strip() for item in text.split("\n\n") if item.strip()]:
            document.add_paragraph(block)

    _section("AI 摘要", result.get("summary"))
    _section("引用片段", result.get("quotes"))
    _section("思维导图", result.get("mindmap"))
    if result.get("evaluation"):
        _section("论文评价", result["evaluation"])
    if result.get("research_brief"):
        _section("深度阅读简报", result["research_brief"])
    if result.get("mermaid"):
        _docx_heading(document, "视觉图谱")
        document.add_paragraph(str(result["mermaid"]))

    render_formula_index_docx(document, _result_formulas(result))
    render_docx_traceability(document, entries, provenance, Pt)

    from io import BytesIO

    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def render_docx_traceability(
    document: Any,
    entries: list[dict[str, Any]],
    provenance: dict[str, Any],
    Pt: Any = None,
) -> None:
    """The citation index and the provenance table, appended to a document.

    Takes the document rather than returning one so the three renderers share
    the same assembly order, and takes ``Pt`` because ``python-docx``'s size
    type is a module-level import this module defers — passing it in keeps the
    import out of the signature and out of the caller's path.
    """
    if entries:
        _docx_heading(document, "引用与证据链接")
        _docx_note(
            document,
            "直接引语逐字回源校验过；转述是我们的表述，不加引号。"
            "两种呈现方式是分开呈现的样式与文字标签，不靠标点区分。",
        )
        quote_style = _docx_style(document, "quote")
        for index, entry in enumerate(entries, start=1):
            _docx_note(
                document,
                f"{index}. {entry['label']} · {entry['presentation_label']} · "
                f"{entry['status_label']} · 证据 ID: {entry['evidence_id']}",
            )
            _docx_note(document, f"打开该引用: {entry['deep_link']}")
            _docx_note(document, f"Evidence API: {entry['api_url']}")
            if entry["presentation"] == "direct_quote":
                # The quotation marks and the Quote style are both on this
                # branch only. A paraphrase takes neither, so there is no path
                # through this function from an unverified sentence to a mark.
                paragraph = (
                    document.add_paragraph(f"“{entry['text']}”", style=quote_style)
                    if quote_style
                    else document.add_paragraph(f"“{entry['text']}”")
                )
            else:
                paragraph = document.add_paragraph(entry["text"])
            if Pt is not None:
                for run in paragraph.runs:
                    run.font.size = Pt(11)

    if provenance.get("has_fact_layer") or provenance.get("run_id"):
        _docx_heading(document, "生成条件与溯源")
        _docx_note(
            document,
            "这张表记录的是「生成这份文件的条件」。打开一个六个月前的导出件时，"
            "它是判断这份结论还能不能信的唯一依据；未记录的字段说明当次运行没有写入该项，"
            "而不是该项不重要。",
        )
        table = document.add_table(rows=1, cols=2)
        header = table.rows[0].cells
        header[0].text = "项目"
        header[1].text = "值"
        for label, key in _EXPORT_PROVENANCE_ROWS:
            cells = table.add_row().cells
            cells[0].text = label
            cells[1].text = _provenance_cell(provenance.get(key))

    if not provenance.get("document_version"):
        _docx_note(
            document,
            "文档版本号未写入本导出件：分析结果里不携带 documents 表的版本行，"
            "而本模块没有可读的存储层。文档 ID 已在上方给出，可据此回查版本。",
        )


def render_run_section(result: dict) -> list[str]:
    """The provenance and cost footer: which prompt, which model, what it cost.

    In a report rather than only in a database row, because the report is
    what gets forwarded to a co-author six months later, and "this summary came
    from prompt facts-v1-… against evidence evs-… and cost $0.004" is the
    sentence that lets somebody reproduce or disbelieve it.
    """
    payload = result.get("analysis_run")
    if not isinstance(payload, dict):
        return []
    facts = PaperFacts.from_dict(result.get("paper_facts") or {})
    run = paper_facts.AnalysisRun.from_dict(payload)
    run.facts = facts
    cost = run.llm_cost
    lines = [
        "## 事实层溯源",
        "",
        f"- 模型: {run.model or '（未记录）'}",
        f"- 提示词版本: {run.prompt_version}",
        f"- 证据版本: {run.evidence_version}",
        f"- LLM 调用: {run.llm_calls} 次，输入约 {run.input_tokens} tokens",
        f"- 成本: {'未知（未配置单价）' if cost is None else f'{cost:.6f}'}",
        f"- 事实覆盖: {run.grounded_fact_count}/{run.fact_count}",
    ]
    if run.legacy_llm_calls:
        lines.append(
            f"- 旧路径对照: {run.legacy_llm_calls} 次调用、"
            f"约 {run.legacy_input_tokens} tokens（本次 {run.llm_calls} 次）"
        )
    if run.degradations:
        lines.append("- 降级说明:")
        lines.extend(f"  - {item}" for item in run.degradations)
    return lines + ["", "---", ""]


def attach_fact_layer(
    result: dict,
    facts: PaperFacts | None = None,
    run: paper_facts.AnalysisRun | None = None,
    report: paper_facts.ExtractionReport | None = None,
) -> dict:
    """Put the fact layer and its run record on a result dict, in place.

    ``__not_stated_fields__`` is stored alongside the payload because
    ``PaperFacts.to_dict`` has no field for it and the schema does not own
    one. It is the answer to "which fields did the paper not say", which is
    the whole point of the fact layer, so dropping it to keep the payload
    schema-pure would lose the feature. The double underscore marks it as
    run-scoped bookkeeping rather than part of the schema, and
    :func:`PaperFacts.from_dict` ignores it, so a reader that does not know
    about it is unaffected.
    """
    if facts is None and run is None and report is None:
        return result
    if facts is not None:
        payload = facts.to_dict()
        if report is not None:
            payload["__not_stated_fields__"] = list(report.not_stated_fields)
        result["paper_facts"] = payload
    if run is not None:
        result["analysis_run"] = run.to_dict()
    if report is not None:
        result["fact_extraction"] = report.to_dict()
    return result


def finalize_analysis_result(
    result: dict,
    analyzer: DocumentAnalyzer,
    original_filename: str,
    generate_evaluation_bool: bool,
    session_id,
    generate_research_brief_bool: bool = True,
    run: paper_facts.AnalysisRun | None = None,
    report: paper_facts.ExtractionReport | None = None,
) -> dict:
    """Write the report, mint a session token and persist the session.

    The raw token is placed on ``result`` exactly here; only its hash is
    stored server-side.

    ``run`` and ``report`` are optional and default to ``None``, so a caller
    driving the old six-section path is completely unaffected. When they are
    present, the fact layer and the run record go onto the result -- which is
    what the API returns, what the Markdown report renders, and what a
    regression test can assert on.

    **The persistence gap, and where it is now closed.** ``/api/export`` renders
    from the *stored session*, and
    :func:`paperwhisperer.api.routes_documents._export_result_from_session`
    rebuilds the renderable result from the top level of that session. The session
    builder copies a fixed key set under ``analysis`` and places none of
    ``parse`` / ``analysis_run`` / ``citations`` / ``claims`` at the top level, so
    the provenance an analysis produced stopped there: the API response had it,
    the exported artefact did not, and the report rendered as an empty shell with
    no record of what produced it. :func:`_session_payload_carrying_provenance`
    closes that gap at the boundary this function owns, copying each key from the
    result and ``document_id`` / ``paper_facts`` through the builder's own keyword
    arguments — so the export is faithful to whatever the analysis actually
    recorded, and a run that recorded nothing still degrades to "（未记录）"
    rather than gaining an invented one.

    Note what this does and does not do. It does not *create* a run record: the
    fact-layer path (:func:`analyze_with_fact_layer`) is what produces
    ``paper_facts`` / ``analysis_run``, and a caller driving the legacy
    six-section path still has neither — for such a run the export carries the
    parse and document identity it has, and the provenance section is still
    absent, which is the honest report of a run that recorded no run. The gap
    fixed here is the transport, not the recording.

    ``document_id`` and the parse quality are attached here, at the last
    moment before the result leaves this function, for both paths. Doing it
    here rather than in each path's own finalizer is the point: a citation
    chip addresses a document, and a field only one of two paths fills in is a
    bug that appears on exactly the path somebody switched to.

    **The id is attached here and the document is registered here**, one step
    later, because publishing an id the store has never heard of is what made
    every citation the reader could click answer ``404 document_not_found``.
    :func:`register_analysis_document` is called after the session and the
    report are written rather than before, so the registration is the last
    thing paid for and its failure cannot cost the reader their analysis. Both
    paths go through this function, so neither can end up publishing an
    unaddressable document id while the other cannot -- which is the same
    "a field only one of two paths fills in" failure the paragraph above is
    about, one level up.
    """
    safe_session_id = build_session_id(session_id)
    session_token = generate_session_token()
    base_name = os.path.splitext(original_filename)[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        config.OUTPUT_FOLDER, f"{base_name}_analysis_{timestamp}.md"
    )

    attach_fact_layer(result, run=run, report=report, facts=run.facts if run else None)
    parse_outcome = getattr(analyzer, "parse_outcome", None)
    attach_document_identity(result, parse_outcome)
    attach_parse_quality(result, parse_outcome)
    # Before the renderers, and after the document id is on the result, because
    # a formula entry's evidence link needs it. The document is read for its
    # formulas only -- a run whose parser produced no document simply gets an
    # index in which nothing is verified, which is the correct answer rather
    # than a reason to fail an analysis that otherwise succeeded.
    attach_formula_index(result, document=getattr(parse_outcome, "document", None))

    markdown = render_markdown_report(result, original_filename, analyzer.version)
    with open(output_file, "w", encoding="utf-8") as handle:
        handle.write(markdown)

    result["output_file"] = output_file
    result["session_id"] = safe_session_id
    result["session_token"] = session_token
    result["source_filename"] = original_filename

    # After the analysis and after the report, and before the session is
    # written. The ordering is the whole of the latency contract: the spine
    # runs ``indexing`` *before* ``analyzing``, so a registration placed in the
    # indexing stage would make the reader wait on an embedding before a single
    # word of their summary was produced. Here the six sections are already
    # generated and the Markdown report is already on disk, so the embedding is
    # the last thing paid for.
    #
    # Before the session write, so that a registration failure reaches the
    # stored session and therefore ``/api/export`` -- a report whose every
    # citation link 404s, with nothing saying why, is the exact failure this
    # notice exists to prevent, and the export renders from the session rather
    # than from this dict. The registration cannot raise, so nothing a reader
    # came for is at risk from standing here.
    register_analysis_document(result, parse_outcome)

    write_session_payload(
        safe_session_id,
        _session_payload_carrying_provenance(
            build_session_payload(
                session_id=safe_session_id,
                source_filename=original_filename,
                document_content=analyzer.document_content,
                analysis=result,
                session_token=session_token,
                document_id=str(result.get("document_id") or ""),
                paper_facts=result.get("paper_facts")
                if isinstance(result.get("paper_facts"), dict)
                else None,
            ),
            result,
        ),
    )
    return result


#: The keys :func:`paperwhisperer.api.routes_documents._export_result_from_session`
#: lifts from the *top level* of a stored session onto the renderable result,
#: and :attr:`paperwhisperer.infra.stores.interfaces` fact layer: the one that
#: decides whether a citation is rendered at all. Read here rather than
#: hard-coded at the call site so this module and the export route cannot drift
#: about which keys they share a contract over.
_EXPORT_LIFTED_KEYS = ("parse", "analysis_run", "citations", "claims")


def _session_payload_carrying_provenance(payload: dict, result: dict) -> dict:
    """Put the provenance the export route reads onto the session's top level.

    **This is the composition fix for D3.** ``/api/export`` renders from the
    *stored session*, not from the result object, and
    :func:`_export_result_from_session` rebuilds the renderable result by copying
    ``parse`` / ``analysis_run`` / ``citations`` / ``claims`` from the top level of
    that session. ``build_session_payload`` keeps a fixed key set under
    ``analysis`` and does not place any of those four at the top level, so every
    one of them was dropped on the way to the export: the result carried the
    parse, the session did not, and the exported report rendered as a shell with
    no record of what produced it.

    Why the payload is completed here rather than in
    :func:`paperwhisperer.sessions.store.build_session_payload`: this is the one
    function that owns the boundary between "what the analysis produced" and
    "what survives in a session", and the export route's contract is a fixed list
    of top-level keys. Recording them here — with the contract named above, and
    with ``document_id`` and ``paper_facts`` handed to the builder through its own
    documented keyword arguments — keeps the assembled path faithful to the
    result. The alternative, mutating the payload from the call site, is what the
    helper exists to prevent.

    Nothing is invented: each key is copied from the result the analysis actually
    produced, and a key the result does not carry is simply not added, so the
    export's "（未记录）" degradation still applies to a run that recorded nothing.
    """
    analysis = payload.setdefault("analysis", {})
    for key in _EXPORT_LIFTED_KEYS:
        value = result.get(key)
        if value:
            payload[key] = value
    # ``paper_facts`` is the one that decides whether the provenance and citation
    # sections render at all (``has_fact_layer``), and the export route does not
    # lift it from the top level — so it also goes under ``analysis`` where the
    # renderers read it.
    facts = result.get("paper_facts")
    if isinstance(facts, dict) and facts and "paper_facts" not in analysis:
        analysis["paper_facts"] = facts
    return payload


def analyze_saved_file(
    file_path,
    original_filename,
    api_key,
    generate_mermaid_bool,
    generate_evaluation_bool,
    session_id,
    generate_research_brief_bool: bool = True,
) -> dict:
    """Analyze a file on disk and return the finalized, session-backed result.

    The old six-section path, unchanged, and now no longer on the main
    analysis route: ``_analyse_stage`` in :mod:`paperwhisperer.api.routes_jobs`
    calls :func:`analyze_with_fact_layer` instead. What this function is kept
    for is *comparison* — it is what produces the numbers
    :func:`paperwhisperer.analysis.paper_facts.legacy_call_budget` reconstructs,
    and running it against the same document is how the fact layer's cost and
    coverage claims were measured before the swap.

    The gate this docstring used to carry — "swapping it is a one-line change at
    the call site, deliberately left as the caller's decision so that the two can
    be compared on the same document rather than one replacing the other
    unmeasured" — has now been satisfied: the comparison was run on the same
    document (about two model calls against thirteen, roughly a five-fold
    wall-clock difference), and the decision was made in favour of the fact
    layer. The legacy path is deliberately kept callable so that claim stays
    checkable rather than becoming folklore.
    """
    resolved_api_key = config.resolve_api_key(api_key)
    if not resolved_api_key:
        raise ValueError("API key is required. Provide api_key or set OPENAI_API_KEY.")

    analyzer = DocumentAnalyzer(resolved_api_key)
    result = analyzer.analyze(
        file_path,
        generate_mermaid_bool,
        generate_evaluation_bool,
        generate_research_brief_bool,
    )
    return finalize_analysis_result(
        result=result,
        analyzer=analyzer,
        original_filename=original_filename,
        generate_evaluation_bool=generate_evaluation_bool,
        session_id=session_id,
        generate_research_brief_bool=generate_research_brief_bool,
    )


def analyze_with_fact_layer(
    file_path,
    original_filename,
    api_key,
    generate_mermaid_bool: bool,
    generate_evaluation_bool: bool,
    session_id,
    generate_research_brief_bool: bool = True,
    *,
    document_id: str = "",
    evidence_provider: Any | None = None,
    store: Any | None = None,
    loaded_document: Any | None = None,
) -> dict:
    """Analyze a file through the fact layer and return the same result shape.

    The entry point T1.8's design is for, and the one that makes the cost and
    consistency claims true. It returns the same keys the old path returns, so
    every template and every route that reads ``summary``, ``quotes``,
    ``mindmap``, ``mermaid``, ``evaluation``, ``research_brief`` and
    ``sections`` keeps working; what changes is where their text came from.

    ``evidence_provider`` is required rather than optional, and a caller with
    no index gets an error rather than a fact layer extracted from raw
    document text. That is the same refusal the retriever makes for the same
    reason: a fact layer with no citations renders as though every claim in it
    had been verified, and there is no way for a reader to tell the difference.

    ``generate_mermaid_bool``, ``generate_evaluation_bool`` and
    ``generate_research_brief_bool`` are all honoured and all three are passed
    through to the pipeline, which is where they belong. They used to be
    accepted and dropped, with only the evaluation flag reaching
    :meth:`~paperwhisperer.analysis.paper_facts.FactLayerPipeline.run`; a caller
    who switched the Mermaid card off still got one, and one who switched it on
    and paid nothing extra got no way to tell that from a card the run had
    chosen not to build. Keeping the arguments is also what lets the two entry
    points stay substitutable at the call site.

    ``elapsed_seconds`` is measured here rather than left to the caller, because
    this is the only place that knows where the analysis begins and ends. The
    old path stamped it on its own result; a result that carried no such field
    left the interface showing "N/A" for a run that had in fact taken most of a
    minute, which is the same invisible difference the three flags above were.
    The measurement covers the parse and the whole DAG and stops before
    :func:`finalize_analysis_result`, matching where the old path drew its
    boundary -- the reader is being told how long the *analysis* took, not how
    long the session write and the trailing registration added.
    """
    resolved_api_key = config.resolve_api_key(api_key)
    if not resolved_api_key:
        raise ValueError("API key is required. Provide api_key or set OPENAI_API_KEY.")
    if evidence_provider is None:
        raise ValueError(
            "事实层分析需要检索证据（evidence_provider），没有索引时不能抽取事实层："
            "请先建索引，或继续使用旧分析路径。"
        )

    started_at = time.monotonic()
    analyzer = DocumentAnalyzer(resolved_api_key)
    if loaded_document is not None:
        # A caller that already parsed the file — because it had to index the
        # document before it could build an evidence provider — hands the very
        # parse it used back here rather than letting this function parse a
        # second time. That is not a micro-optimisation: a parse mints a fresh
        # ``document_id`` whenever the parser does not supply a content-derived
        # one, so two parses of one file are two documents. Indexing the first
        # and extracting from the second would give a run whose evidence
        # citations address a document the store never registered — the exact
        # unaddressable-citation defect this whole path exists to remove. One
        # parse, one identity, is therefore the only correct arrangement.
        _adopt_loaded_document(analyzer, loaded_document)
        document = getattr(loaded_document.outcome, "document", None)
        if document is None:
            raise ValueError("文档解析结果中没有可用的结构化文档，无法生成章节树。")
    else:
        document = _load_document_once(analyzer, file_path)

    pipeline = paper_facts.FactLayerPipeline(
        complete=analyzer.client.complete,
        evidence_provider=evidence_provider,
        model=getattr(analyzer.client, "model", ""),
        document_id=document_id or str(getattr(document, "document_id", "") or ""),
        embedding_usage=_embedding_usage(store),
    )
    outcome = pipeline.run(
        document=document,
        generate_evaluation=generate_evaluation_bool,
        generate_mermaid=generate_mermaid_bool,
        generate_research_brief=generate_research_brief_bool,
        legacy_content=analyzer.document_content,
        created_at=now_iso(),
    )
    result = _result_from_outcome(
        outcome,
        content=analyzer.document_content,
        parse_outcome=analyzer.parse_outcome,
        generate_mermaid=generate_mermaid_bool,
        generate_evaluation=generate_evaluation_bool,
        generate_research_brief=generate_research_brief_bool,
    )
    result["elapsed_seconds"] = round(time.monotonic() - started_at, 1)
    logger.info(
        "Fact-layer analysis completed in %.1fs for %s (%d facts)",
        result["elapsed_seconds"],
        original_filename,
        len(outcome.facts.all_facts()),
    )
    return finalize_analysis_result(
        result=result,
        analyzer=analyzer,
        original_filename=original_filename,
        generate_evaluation_bool=generate_evaluation_bool,
        session_id=session_id,
        generate_research_brief_bool=generate_research_brief_bool,
        run=outcome.run,
        report=outcome.report,
    )


def _load_document_once(analyzer: DocumentAnalyzer, file_path) -> Any:
    """Parse the file and keep both the text and the canonical document.

    One parse, shared: the same rule the orchestrator's ``_load_document``
    follows, and for the same reason. Two parses of one file can return two
    different documents when the cache is cold and the backend is nondeterministic
    -- and a fact layer extracted from one parse, quoted against another, is a
    citation that points at a page whose text has since changed.
    """
    from paperwhisperer.documents.loader import DocumentLoader

    loaded = DocumentLoader.load_with_outcome(file_path)
    _adopt_loaded_document(analyzer, loaded)
    document = getattr(loaded.outcome, "document", None)
    if document is None:
        raise ValueError("文档解析结果中没有可用的结构化文档，无法生成章节树。")
    return document


def _adopt_loaded_document(analyzer: DocumentAnalyzer, loaded: Any) -> None:
    """Point ``analyzer`` at a parse a caller already performed.

    The two attributes are one decision: the text the prompts see and the
    structured document the identity, the formula index and the citation locators
    come from must be the *same* parse of the *same* bytes. Setting one without
    the other is how a citation ends up addressing a document id that was never
    registered, so this is one function rather than two assignments at each
    call site.
    """
    analyzer.document_content = loaded.text
    analyzer.parse_outcome = loaded.outcome


def _embedding_usage(store: Any) -> Any:
    """The embedding cost of the retrieval that fed this analysis, if any.

    Read from the store's gateway by ``reset_usage`` rather than accumulated by
    hand here, because the gateway already counts every request, token and
    cache hit and a second counter would be a second opinion. ``None`` when
    there is no embedding provider, which is a personal-tier deployment with
    the dense leg off -- and ``None`` means "not measured", so the run record
    says so rather than reporting zero.
    """
    gateway = getattr(store, "embeddings", None) or getattr(store, "gateway", None)
    reset = getattr(gateway, "reset_usage", None)
    if not callable(reset):
        return None
    try:
        usage = reset()
    except Exception as exc:  # noqa: BLE001 - provenance must not fail an analysis
        logger.warning("读取 embedding 用量失败：%s", exc)
        return None
    return usage if getattr(usage, "requests", 0) else None


def _result_from_outcome(
    outcome: paper_facts.FactLayerOutcome,
    *,
    content: str,
    parse_outcome: Any = None,
    generate_mermaid: bool = True,
    generate_evaluation: bool = True,
    generate_research_brief: bool = True,
) -> dict:
    """Shape the DAG's output like an analysis result.

    The card text is the DAG's; the surrounding keys are the orchestrator's,
    built here rather than inside the pipeline so that the pipeline has no
    opinion about the API contract and the contract has no opinion about the
    fact layer. ``sections`` is filled in from the same strings, because the
    templates read it for per-section status and a result whose flattened keys
    and ``sections`` disagreed would be a worse bug than either path had.

    ``quality`` reports ``partial`` when the extraction was degraded, which is
    a real difference from the old path's meaning of the word: here it means
    some fields are unverified or absent, not that a section threw. Both are
    reasons a reader should not take the cards at face value, which is what the
    field is for.

    **The follow-up suggestions are the old path's, derived the old way.** They
    used to be hard-coded here -- four questions and an empty action list --
    which meant this path shipped an interface with no follow-up affordances
    and, worse, disagreed with the path it was written to replace about what
    those affordances were. They are now produced by the very same
    :func:`~paperwhisperer.analysis.orchestrator.build_analysis_metadata` the
    legacy path calls, from the very same ``sections`` statuses, so there is one
    implementation of "what can I usefully ask next" rather than two that
    disagree about it. That is why the counts are what they are: the questions
    grow to six and the actions to five when the sections that justify them are
    present, which is the behaviour the reader already had.

    **A switched-off section is ``disabled``, not ``empty``.** The card is empty
    either way, so the text cannot tell the two apart, but the two mean opposite
    things to the code that reads the statuses: ``empty`` is a section that
    should have produced text and did not, and it is what
    :func:`build_analysis_metadata` turns into a "go and ask about the
    gaps" action. A reader who switched the research brief off and is then
    offered a follow-up about why their research brief is missing has been told
    their own click was a failure. So the flags are threaded in here, where the
    statuses are computed, rather than being re-inferred from an empty string.
    """
    disabled_by_flag = {
        "mermaid": not generate_mermaid,
        "evaluation": not generate_evaluation,
        "research_brief": not generate_research_brief,
    }
    sections_map: dict[str, dict] = {}
    for key in SECTION_KEYS:
        text = outcome.cards.get(key, "")
        status = "disabled" if disabled_by_flag.get(key) else "success" if text else "empty"
        sections_map[key] = {
            "status": status,
            "content": text,
            "error": "",
            "retryable": False,
        }

    # One derivation of the follow-up affordances from the statuses computed
    # above, so the questions, the actions and the quality verdict cannot
    # disagree about which sections actually produced anything.
    metadata = build_analysis_metadata(sections_map)

    result: dict[str, Any] = {
        "char_count": len(content),
        "sections": sections_map,
        "parse": build_parse_provenance(parse_outcome),
    }
    for key in SECTION_KEYS:
        result[key] = outcome.cards.get(key, "")
    # The statuses are computed once, above, and both the quality verdict and
    # the follow-up affordances are derived from that single map -- so the
    # interface can never claim a section is complete while offering the reader
    # an action for filling it in.
    result["analysis_status"] = {
        **metadata["analysis_status"],
        "quality": "partial" if outcome.report.is_degraded else "complete",
    }
    result["suggested_questions"] = metadata["suggested_questions"]
    result["next_actions"] = metadata["next_actions"]
    return attach_fact_layer(
        result, facts=outcome.facts, run=outcome.run, report=outcome.report
    )


def record_qa_turn(session_payload: dict, question: str, answer: str, answer_mode: str) -> None:
    """Append a Q&A turn and refresh the cached document excerpt."""
    qa_history = session_payload.get("qa_history", [])
    qa_history.append(
        {
            "question": question,
            "answer": answer,
            "answer_mode": answer_mode,
            "timestamp": now_iso(),
        }
    )
    session_payload["qa_history"] = qa_history
    session_payload["document_excerpt"] = build_document_excerpt(
        get_session_document_content(session_payload)
    )
