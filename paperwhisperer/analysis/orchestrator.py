"""Document analysis orchestration.

Runs the analysis sections concurrently under a shared LLM concurrency cap and
turns each one into a self-describing result. A section that fails is reported
in place rather than aborting the run; only a total failure of the required
sections is treated as a failed analysis.

The document itself is parsed by the ingestion gateway, once, before any
section runs. Two things about that are deliberate.

**The sections keep taking flat text.** ``ParsedDocument`` is the canonical
form and the one right place for a citation to hang off, but every prompt in
:mod:`paperwhisperer.analysis.sections` is written against prose, and
rewriting six prompts to accept structured elements is T1.x's work, not this
migration's. So the parse happens exactly once, :meth:`ParsedDocument.to_full_text`
flattens it exactly once, and the sections are handed the same string they
have always been handed. The document object does not die at the boundary
though — see below.

**The outcome is kept and published.** A user who uploads a file has a right
to know which parser read it and whether it was sent to a cloud service, and
"which parser" is a fact that only exists *around* the document: the chain may
have tried three parsers, consulted a privacy gate, or answered from a cache.
None of that survives in a flat string, so the :class:`ParseOutcome` is kept on
the analyzer and its user-facing line is attached to the result as ``parse``.
It is additive — every existing key keeps its meaning and its shape.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import logging
import os
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from paperwhisperer.analysis import sections
from paperwhisperer.core import config
from paperwhisperer.core.errors import is_retryable_llm_error
from paperwhisperer.core.text import (
    compact_text,
    trim_text_for_log,
)
from paperwhisperer.documents.loader import DocumentLoader, TextChunker
from paperwhisperer.infra.stores import DocumentRecord, StoreBundle, VectorEntry, build_stores
from paperwhisperer.ingestion import ParseOutcome
from paperwhisperer.ingestion.schema import Element, Evidence, ParsedDocument, SourceLocator
from paperwhisperer.llm import output
from paperwhisperer.llm.client import LLMClient
from paperwhisperer.llm.embeddings import EmbeddingDimensionMismatchError
from paperwhisperer.llm.prompts import (
    ANSWER_MODES,
    DEFAULT_ANSWER_MODE,
    build_stable_system_prompt,
    build_task_user_prompt,
    normalize_answer_mode,
)
from paperwhisperer.rag.chunking import (
    build_index_version,
    chunk_document_from_config,
    index_version_parts,
)
from paperwhisperer.rag.citations import ClaimPresentation, verify_evidence_quote
from paperwhisperer.rag.retrieve import (
    TASK_QA,
    HybridRetriever,
    RetrievalResult,
    plan_for,
)

logger = logging.getLogger(__name__)

#: Section name -> whether it is optional (can be switched off by the caller).
OPTIONAL_SECTIONS = ("mermaid", "evaluation", "research_brief")


def build_section_result(
    status: str,
    content: str = "",
    error: str = "",
    retryable: bool = False,
) -> dict:
    return {
        "status": status,
        "content": content or "",
        "error": error or "",
        "retryable": bool(retryable),
    }


def build_parse_provenance(outcome: ParseOutcome | None) -> dict[str, Any]:
    """Describe how one file was parsed, for the result payload.

    Returns the fields a reader (or a support ticket) needs to answer "who read
    this and did it leave my machine", and nothing that would let a caller
    reconstruct a document they do not have.

    ``used_cloud`` is the field that matters most and is separated from
    ``cloud_attempted`` on purpose. The first means *this* text came from the
    cloud; the second means a request was made and paid for, which may still be
    true of a document that was ultimately parsed locally. Conflating them is
    how a deployment ends up quietly billing users for a result it threw away,
    so both are reported and the difference is left to the reader.

    An empty dict is returned when there is no outcome. That happens when a
    caller assembles a result by hand — the unit tests do exactly that — and it
    is preferable to a dict of confident-looking falses that a template would
    happily render as "parsed by ".
    """
    if outcome is None:
        return {}
    return {
        "summary": outcome.summary_line(),
        "parsed_by": outcome.parsed_by,
        "status": outcome.status,
        "used_cloud": outcome.used_cloud,
        "cloud_attempted": outcome.cloud_attempted,
        # The machine-readable cloud outcome, so the interface can say "top up
        # your quota" and mean it. Without this the three cloud failures are one
        # Chinese sentence and the reader cannot act on any of them; with it
        # they are three different instructions. `error_message` rides along for
        # the same reason — a branch still needs something to show.
        "cloud_outcome": outcome.cloud_outcome,
        "error_message": outcome.error_message,
        "touched_network": outcome.touched_network,
        "cache_hit": outcome.cache_hit,
        "fallback_used": outcome.fallback_used,
        "media_type": outcome.media_type,
        "warnings": list(outcome.warnings),
    }


def build_analysis_metadata(sections_map: dict) -> dict:
    """Derive follow-up suggestions from which sections actually succeeded."""
    section_statuses = {
        name: str((section or {}).get("status") or "empty")
        for name, section in (sections_map or {}).items()
    }
    completed = [
        sections.SECTION_LABELS.get(name, name)
        for name, status in section_statuses.items()
        if status == "success"
    ]
    failed = [
        sections.SECTION_LABELS.get(name, name)
        for name, status in section_statuses.items()
        if status == "failed"
    ]
    disabled = [
        sections.SECTION_LABELS.get(name, name)
        for name, status in section_statuses.items()
        if status == "disabled"
    ]

    suggested_questions = [
        "这篇论文要解决的核心问题是什么？",
        "它的主要方法和创新点分别是什么？",
        "实验或论证最支持哪些结论？",
        "这篇论文有哪些局限性和后续研究方向？",
    ]
    if section_statuses.get("quotes") == "success":
        suggested_questions.append("哪些原文片段最适合在综述或笔记中引用？")
    if section_statuses.get("evaluation") == "success":
        suggested_questions.append("如果我要复现或扩展这篇论文，应该优先关注什么？")
    if section_statuses.get("research_brief") == "success":
        suggested_questions.append("这篇论文最适合放进哪条研究脉络或综述段落？")

    next_actions = [
        {"label": "追问方法细节", "prompt": "请解释这篇论文的方法流程，并指出每一步解决了什么问题。"},
        {"label": "整理局限性", "prompt": "请基于文档总结这篇论文的局限性，并给出可能的改进方向。"},
        {"label": "生成阅读路线", "prompt": "请把这篇论文拆成适合精读的阅读路线和检查清单。"},
    ]
    if section_statuses.get("research_brief") == "success":
        next_actions.extend([
            {"label": "复现路线", "prompt": "请基于深度简报生成一份复现路线，包括数据、实验变量、依赖和风险点。"},
            {"label": "找后续工作", "prompt": "请提炼 5 个英文检索关键词，用于寻找这篇论文的后续工作或相邻研究。"},
        ])
    if section_statuses.get("evaluation") == "disabled":
        next_actions.append({"label": "手动评价", "prompt": "请基于当前文档补充一份批判性评价。"})
    if failed:
        next_actions.append(
            {"label": "补全失败部分", "prompt": f"请重新生成以下分析部分：{', '.join(failed)}。"}
        )

    return {
        "suggested_questions": suggested_questions[:6],
        "next_actions": next_actions[:5],
        "analysis_status": {
            "quality": "partial" if failed else "complete",
            "completed_sections": completed,
            "failed_sections": failed,
            "disabled_sections": disabled,
            "section_statuses": section_statuses,
        },
    }


# --------------------------------------------------------------------------
# The Q&A answer surface (T1.7)
# --------------------------------------------------------------------------

#: The three states an answer can be in, named as the plan's acceptance
#: criterion names them. ``SUPPORTED`` is only reachable when every claim the
#: answer makes is carried by evidence that was retrieved for *this* question
#: and every quoted span in it aligned to that evidence — which is why it is
#: computed from the claims and never read from the model.
CONFIDENCE_SUPPORTED = "supported"
CONFIDENCE_PARTIALLY_SUPPORTED = "partially_supported"
CONFIDENCE_INSUFFICIENT = "insufficient_evidence"

CONFIDENCE_VALUES = (
    CONFIDENCE_SUPPORTED,
    CONFIDENCE_PARTIALLY_SUPPORTED,
    CONFIDENCE_INSUFFICIENT,
)

#: The Chinese sentence for each way retrieval can come back empty. Written
#: per-cause rather than as one generic refusal because "the paper does not say
#: this" and "this deployment has no index" are different facts, and a reader
#: told the first when the truth is the second will conclude the wrong thing
#: about the paper. Keys are the retriever's ``empty_reason`` vocabulary; a
#: reason this table does not name falls back to ``no_candidates``, which is
#: the least specific of them and therefore the least likely to be wrong.
EMPTY_MESSAGES = {
    "no_candidates": "在全文中没有检索到与该问题相关的原文，无法给出有依据的回答。",
    "filtered_out": "本次问答没有命中任何已索引的文档片段，无法给出有依据的回答。",
    "beyond_threshold": "检索到的片段与该问题的相关度不足，无法给出有依据的回答。",
    "legs_skipped": "检索没有执行，无法给出有依据的回答。",
}

#: The refusal for the other way an answer can fail: retrieval *did* find
#: passages, the model *was* asked, and nothing it said could be tied to them.
#: It is a separate sentence from :data:`EMPTY_MESSAGES` because it is a
#: different fact — "we could not find anything" and "we found something and
#: could not use it" leave the reader with completely different next steps.
UNVERIFIED_REFUSAL = (
    "证据校验未通过：检索到了相关片段，但回答中的结论没有可回源校验的依据，"
    "因此不展示该回答。请换一种问法，或直接核对下方列出的证据片段。"
)

#: The reply contract, written into the prompt. A quoted string, not prose
#: about a schema, because the one field that matters most — ``quote`` — has to
#: be a value the verifier can be handed rather than a span a parser has to
#: recover from a sentence the model is also asserting. The sentence asserting
#: it is exactly what the gate exists to distrust.
CLAIM_OUTPUT_CONTRACT = """{
  "answer": "面向用户的回答，先给结论，说明依据来自哪些证据片段。",
  "claims": [
    {
      "text": "该结论的中文表述。",
      "quote": "从证据原文中逐字摘录的句子；若为转述则留空字符串。",
      "evidence_ids": ["支撑该结论的证据编号"]
    }
  ]
}"""


#: Quote pairs stripped from an unverified quotation in the answer prose.
#: Mirrors ``_WRAPPING_QUOTE_PAIRS`` in :mod:`paperwhisperer.rag.citations`, and
#: deliberately so rather than by import: that list is private to a module
#: whose contract is that verification is a value, and reaching past it for a
#: presentational detail would couple this one to its internals. The front end
#: mirrors the same list for the same reason (``templates/static/js/claims.js``).
_WRAPPING_QUOTE_PAIRS = (
    ('"', '"'),
    ("“", "”"),
    ("‘", "’"),
    ("「", "」"),
    ("『", "』"),
    ("《", "》"),
    ("〝", "〞"),
    ("＂", "＂"),
)

#: How many characters of one retrieved passage go into the prompt. A bound and
#: not a share of a global budget: a retrieved passage is the *only* knowledge
#: the answer has, so truncating it to make room for other passages would trade
#: the evidence for the evidence, and the passages that get cut are the long
#: ones — which are disproportionately the experimental sections this task
#: exists to reach.
QA_EVIDENCE_ITEM_CHARS = 3000

#: Ceiling on the whole evidence block, so a pathological retrieval cannot blow
#: the prompt past the model's window. Applied by dropping whole trailing
#: passages rather than by cutting text mid-sentence: a passage cut in half is
#: a passage whose quotes will not align, and a quote that fails to align is
#: downgraded to a paraphrase, so truncating mid-passage would silently convert
#: real direct quotes into unverifiable ones.
QA_EVIDENCE_TOTAL_CHARS = 12000


def build_evidence_answer_prompts(
    *, question: str, evidence: Sequence[Evidence], history_block: str, answer_mode: str
) -> tuple[str, str]:
    """Assemble the answer prompt from retrieved evidence and nothing else.

    Each passage is tagged with the ``ev_`` id the reply is required to cite it
    by, and the reply is required to be JSON: a free-prose answer cannot be
    checked claim by claim, and an answer that cannot be checked is an answer
    that cannot be given confidence. The ``quote`` field is a separate string
    precisely so the verifier can be handed the span the model believes it
    quoted, instead of having to recover it from a sentence it is also
    asserting.
    """
    input_blocks: list[tuple[str, str]] = []
    if evidence:
        input_blocks.append(("retrieved_evidence", render_evidence_block(evidence)))
    if history_block:
        input_blocks.append(("recent_qa_history", history_block))
    input_blocks.append(("user_question", question))

    constraints = (
        "只能依据 retrieved_evidence 中的原文片段回答；"
        "history 只用于理解追问上下文，不能作为依据。"
        "证据中没有的内容必须回答“文档中没有说明”，"
        "不得根据经验、常识或论文题名补全，不得给出证据里没有的数字、章节或结论。\n"
        f"{ANSWER_MODES.get(answer_mode, ANSWER_MODES[DEFAULT_ANSWER_MODE])}"
    )
    return (
        build_stable_system_prompt("qa"),
        build_task_user_prompt(
            task="基于检索到的原文证据回答用户关于当前文档的追问，并给出可回源校验的引用。",
            constraints=constraints,
            output_format=CLAIM_OUTPUT_CONTRACT,
            input_blocks=input_blocks,
        ),
    )


def render_evidence_block(evidence: Sequence[Evidence]) -> str:
    """The retrieved passages as a labelled block, under the stated budgets.

    Every passage carries its id twice over — as a tag and inside the text —
    because the id is what the reply has to reproduce and a model that has to
    copy it out of prose is a model that will get it wrong. Getting it wrong is
    not cosmetic: an id that names nothing resolves to no evidence, and the
    claim is correctly downgraded for citing something the retriever never
    returned.
    """
    blocks: list[str] = []
    used = 0
    for item in evidence:
        text = str(item.text or "").strip()
        if not text:
            continue
        if len(text) > QA_EVIDENCE_ITEM_CHARS:
            text = text[:QA_EVIDENCE_ITEM_CHARS].rstrip() + "…"
        if used + len(text) > QA_EVIDENCE_TOTAL_CHARS:
            break
        used += len(text)
        label = item.citation_label
        heading = " › ".join(item.heading_path) if item.heading_path else ""
        header = f"[{item.evidence_id}]"
        if heading or label:
            header += f" ({heading + ' · ' if heading else ''}{label})"
        blocks.append(f"{header}\n{text}")
    return "\n\n".join(blocks)


def parse_claim_payload(raw: str) -> tuple[str, list[AnswerClaim], list[str]]:
    """Read the model's reply into an answer string, claims, and notes.

    The model is asked for JSON and told that every claim must name an
    ``ev_`` id it was given. This function does not enforce either. It parses
    what arrived and hands it to the verifier, which is where an id that names
    nothing is caught: :func:`_verify_claim` resolves ids against the retrieved
    set and a claim with no resolvable id is unbacked, so a reply that ignores
    the contract produces unbacked claims and therefore a lower confidence
    rather than a silent pass.

    A reply that is not JSON at all becomes a single claim carrying no quote
    and no evidence. It is still reported — the prose is what the reader reads,
    and hiding it because it arrived in the wrong shape would be its own kind of
    dishonesty — but it cannot be supported, and the note says why.

    The parse is delegated to :mod:`paperwhisperer.llm.output`, which is the
    single owner of the failure taxonomy. Two things follow that the hand-rolled
    ``json.loads(extract_json_object(text))`` did not give us. A reply the model
    fenced, punctuated with full-width colons, or prefixed with a sentence of
    explanation is now *recovered* and the recovery is reported, instead of
    being demoted to an unsupported single claim — and because the live endpoint
    fences most of its answers, that is the common case rather than the rare
    one. And the failure notes now name which of the taxonomy's seven outcomes
    occurred, so "the model sent prose" is distinguishable in the UI from "the
    model sent JSON we could not read".
    """
    notes: list[str] = []
    text = str(raw or "").strip()
    if not text:
        return "", [AnswerClaim(text="")], ["模型没有返回任何内容。"]

    parsed = output.parse_json_object(text, label="问答作答")
    if not parsed.ok:
        if parsed.status is output.OutputStatus.WRONG_SHAPE:
            notes.append("模型回复不是约定的 JSON 对象，已按单条未验证声明处理。")
        else:
            notes.append(f"{parsed.user_message()}已按单条未验证声明处理。")
        notes.extend(parsed.degradations())
        return text, [AnswerClaim(text=text)], notes

    # A reply that needed repair is still reported as such even though it
    # parsed, so a reader can tell a confident-looking answer from one whose
    # bytes we had to rewrite.
    notes.extend(parsed.degradations())
    payload = parsed.value if isinstance(parsed.value, dict) else {}

    answer_text = str(payload.get("answer") or "")
    raw_claims = payload.get("claims")
    if not isinstance(raw_claims, list):
        notes.append("模型回复中没有 claims 数组，已按单条未验证声明处理。")
        return answer_text or text, [AnswerClaim(text=answer_text or text)], notes

    claims: list[AnswerClaim] = []
    for item in raw_claims:
        if not isinstance(item, dict):
            continue
        ids = item.get("evidence_ids")
        claims.append(
            AnswerClaim(
                text=str(item.get("text") or ""),
                quote=str(item.get("quote") or ""),
                evidence_ids=tuple(
                    str(value) for value in ids if str(value or "").strip()
                ) if isinstance(ids, list) else (),
            )
        )
    return answer_text, claims, notes


def _verify_claim(
    claim: AnswerClaim, evidence: Sequence[Evidence], document: ParsedDocument | None
) -> AnswerClaim:
    """Check one claim's quotation against the evidence it cites.

    Three outcomes, and the third is the important one. A quote that aligns in
    the evidence the claim named becomes a direct quote carrying the source's
    own characters. A quote that aligns nowhere the claim named becomes a
    paraphrase — the model's wording, with no quotation marks, and a message
    saying why. And a claim that named no evidence at all is left as a
    paraphrase carrying no id, so it can be rendered and cannot be cited.

    The scope is deliberately *only* the evidence the claim itself named.
    Widening it to the whole retrieval would let a fluent sentence that appears
    in some other passage "verify" against a citation pointing at this one,
    which is precisely the failure this release gate exists to prevent: the
    reader follows the chip, lands somewhere else, and concludes the paper said
    something in a context that changes its meaning.
    """
    if not claim.quote or not evidence or document is None:
        # The ids are replaced with the ones that *resolved*, never merged with
        # what the model sent. Keeping an unresolvable id would let a claim
        # cite ``ev_something_the_retriever_never_returned`` and be counted as
        # carried by evidence it does not have, which is the whole fabrication
        # this gate exists to catch. An id that resolved to nothing is dropped
        # here, and the claim is left with no evidence behind it.
        return replace(
            claim,
            evidence_ids=tuple(item.evidence_id for item in evidence),
            citation_label=evidence[0].citation_label if evidence else "",
            locator=evidence[0].locator if evidence else None,
        )

    first_outcome = None
    for item in evidence:
        verdict = verify_evidence_quote(claim.quote, item, document)
        if verdict.aligned:
            return replace(
                claim,
                evidence_ids=tuple(candidate.evidence_id for candidate in evidence),
                presentation=ClaimPresentation.DIRECT_QUOTE,
                verified_text=verdict.verified_text,
                outcome=verdict.outcome.value,
                message=verdict.message,
                locator=verdict.locator or item.locator,
                citation_label=item.citation_label,
            )
        if first_outcome is None:
            first_outcome = (verdict, item)

    verdict, item = first_outcome
    return replace(
        claim,
        evidence_ids=tuple(candidate.evidence_id for candidate in evidence),
        presentation=ClaimPresentation.PARAPHRASE,
        verified_text="",
        outcome=verdict.outcome.value,
        message=verdict.message,
        locator=None,
        citation_label=item.citation_label,
    )


def _derive_confidence(
    claims: Sequence[AnswerClaim], *, relevance: Any = None
) -> str:
    """Confidence, computed from the claims the answer actually carries.

    A claim is *verified* when it is carried by evidence that was retrieved for
    this question, that **cleared the retrieval relevance floor**, and that
    presents no quotation or a quotation that aligned. "Presents no quotation"
    still counts as verified deliberately: a claim is a paraphrase by nature,
    the gate is about claims dressed as quotations, and a rule that made every
    paraphrase unverified would report ``insufficient_evidence`` for a perfectly
    well-sourced answer and teach users to distrust a label that is always wrong.
    That argument is about the *quotation*, and it does not extend to relevance.

    **What was added, and why it is not the same rule.** Before the floor, a
    claim was verified when it cited any id the retriever had minted. That is an
    identity test, and identity is not relevance: a retriever that padded every
    question to ``final_k`` minted eight ids for every question including the
    ones it had no answer to, so a paraphrase resting on one of those was
    labelled ``supported``. The walkthrough reproduced exactly that, with no
    model in the loop. ``relevance`` is the floor's verdict -- which passages
    were allowed to count as evidence at all -- and a claim that cites a passage
    the floor rejected is not verified.

    ``relevance`` is keyword-only and optional purely so this function stays
    callable in isolation; ``None`` means "no floor report available", which
    skips the check rather than failing it. The production call site at
    :meth:`DocumentAnalyzer._compose_answer_result` always passes one.

    The three-way split is then the only thing this could be. All claims
    verified and at least one claim present is ``supported``; some verified is
    ``partially_supported``; none is ``insufficient_evidence``. The model is
    never asked, and the answer's own prose is never consulted — a model asked
    whether its answer is supported will say it is, and a rule that reads that
    answer is a rule that cannot report its own failure.
    """
    if not claims:
        return CONFIDENCE_INSUFFICIENT
    verified = [claim for claim in claims if _claim_is_verified(claim, relevance)]
    if not verified:
        return CONFIDENCE_INSUFFICIENT
    if len(verified) == len(claims):
        return CONFIDENCE_SUPPORTED
    return CONFIDENCE_PARTIALLY_SUPPORTED


def _claim_is_verified(claim: AnswerClaim, relevance: Any = None) -> bool:
    """Whether one claim is carried by real, *relevant* evidence with a valid quotation."""
    if not claim.evidence_ids:
        return False
    # The defect this closes: an id the retriever minted is not a passage that
    # answers the question. Before the relevance floor every query was padded to
    # a full set, so every fabricated-and-supported chain had this door open.
    if relevance is not None and not all(
        relevance.clears(evidence_id) for evidence_id in claim.evidence_ids
    ):
        return False
    if not claim.quote:
        return True
    return claim.is_direct_quote


def _citations_for(
    claims: Sequence[AnswerClaim], result: RetrievalResult
) -> list[AnswerCitation]:
    """The passages the answer's claims point at, deduplicated and ordered.

    Built from the claims rather than from the whole retrieval, because a
    citation the answer does not use is not a citation — it is a suggestion.
    The expansion passages the retriever attached for context are deliberately
    absent: they were *placed*, not found, and a chip that opened one would
    present a passage the system did not retrieve as though it had.
    """
    by_id = {item.evidence_id: item for item in result.evidence}
    citations: list[AnswerCitation] = []
    seen: set[str] = set()
    for claim in claims:
        for evidence_id in claim.evidence_ids:
            if evidence_id in seen:
                continue
            item = by_id.get(evidence_id)
            if item is None:
                continue
            seen.add(evidence_id)
            citations.append(
                AnswerCitation(
                    evidence_id=item.evidence_id,
                    chunk_id=item.chunk_id,
                    citation_label=item.citation_label,
                    locator=item.locator,
                    heading_path=item.heading_path,
                    text=item.text,
                    score=item.score,
                    is_direct_quote=claim.is_direct_quote,
                    presentation=claim.presentation,
                    message=claim.message,
                )
            )
    return citations


def _strip_unverified_quotes(answer: str, claims: Sequence[AnswerClaim]) -> str:
    """Remove the quotation marks from every quotation the verifier rejected.

    Applied to the answer's own prose, not only to the claim list, because the
    prose is what the reader reads first and a quote-shaped sentence with no
    marks is still legible while a fabricated one inside quotation marks is
    exactly the failure being gated. Only spans named by a claim that actually
    failed verification are touched, so a quotation mark the model used for
    emphasis — or one that *did* align — is left exactly as written.
    """
    text = str(answer or "")
    for claim in claims:
        if not claim.quote or claim.is_direct_quote:
            continue
        for opening, closing in _WRAPPING_QUOTE_PAIRS:
            wrapped = f"{opening}{claim.quote}{closing}"
            if wrapped in text:
                text = text.replace(wrapped, claim.quote)
    return text


def _stream_pieces(answer: str, pieces: int = 4) -> list[str]:
    """Split a finished answer into pieces, so the streaming path still streams.

    Cut at sentence terminators, falling back to a hard split for text with none
    (a refusal, a heading, a table row). Splitting on a hard character count
    would cut mid-sentence, and the front end renders these as markdown: a
    piece ending mid-word is a visible artefact of the transport appearing in
    the middle of the reader's answer, which is worse than the streaming being
    slightly lumpy.
    """
    text = str(answer or "")
    if not text:
        return []
    if pieces < 1:
        return [text]
    if len(text) <= pieces * _STREAM_PIECE_CHARS:
        return [text]

    terminators = "。！？\n"
    bounds = [index + 1 for index, char in enumerate(text) if char in terminators]
    if not bounds:
        size = max(1, len(text) // pieces)
        return [text[index : index + size] for index in range(0, len(text), size)]

    cuts: list[int] = []
    for bound in bounds:
        if not cuts or bound - cuts[-1] >= _STREAM_PIECE_CHARS:
            cuts.append(bound)
    parts = [text[start:stop] for start, stop in zip([0, *cuts], cuts, strict=False)]
    tail = text[cuts[-1] :]
    if tail:
        parts.append(tail)
    return [part for part in parts if part]


#: Roughly one clause per piece. A floor, not a target: a short answer is
#: emitted whole rather than padded into a fixed number of chunks, because
#: splitting a two-sentence answer into four events helps nothing.
_STREAM_PIECE_CHARS = 40


def parsed_document_from_text(text: str) -> ParsedDocument:
    """Wrap stored flat text as a ``ParsedDocument`` with resolvable offsets.

    The session persists the document's text, not the parsed object, so a Q&A
    turn that arrives in a later request — which is every turn in production —
    has prose and nothing else. This rebuilds the smallest canonical document
    that prose supports: paragraph-sized elements carrying ``txt_offset``
    locators, which the evidence API and the citation verifier both resolve.

    The honest cost is that structure the parse recovered and the flat
    projection dropped (headings, tables, page boundaries) is not recovered
    here. That is recorded in the document's warnings rather than papered over,
    because a citation into a re-derived document is a weaker claim than one
    into a parsed one and the difference is exactly the kind a reader should
    be able to see.
    """
    text = str(text or "")
    document = ParsedDocument(
        document_id="doc_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
        source_filename="session_document",
        media_type="text/plain",
    )
    if not text.strip():
        document.warnings.append("会话中没有可检索的文档正文。")
        return document

    for start, end in _text_element_spans(text):
        body = text[start:end].strip()
        if not body:
            continue
        document.elements.append(
            Element(
                role="paragraph",
                text=body,
                locator=SourceLocator(kind="txt_offset", offset_start=start, offset_end=end),
            )
        )
    if not document.elements:
        document.warnings.append("会话中的文档正文没有可切分的段落。")
    document.build_units()
    document.warnings.append(
        "问答使用的文档结构由会话保存的纯文本重建，"
        "标题层级、表格结构与页码定位可能不完整。"
    )
    return document


#: Characters per rebuilt element. Matches the legacy plain-text parser's
#: window, for the same reason: plain text carries no paragraph structure worth
#: trusting, so a fixed budget under a typical embedding chunk is the honest
#: unit rather than a boundary invented from blank lines that a PDF export
#: scattered arbitrarily.
_TEXT_ELEMENT_CHARS = 1200


def _text_element_spans(text: str, size: int = _TEXT_ELEMENT_CHARS) -> list[tuple[int, int]]:
    """Split ``text`` into addressable ``(start, end)`` spans.

    Cuts prefer a paragraph or line break near the end of the window so an
    element usually does not stop mid-sentence, but the span boundaries stay
    exact: the offsets are the citation handle, and nudging them to make the
    text prettier would point a citation at the wrong characters.
    """
    if not text:
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            for separator in ("\n\n", "\n", "。", ". "):
                cut = window.rfind(separator)
                if cut > size // 2:
                    end = start + cut + len(separator)
                    break
        spans.append((start, end))
        start = end
    return spans


def _embedding_index_version() -> str:
    """The index version a document chunked right now would carry.

    Read from the same configuration :func:`chunk_document_from_config` reads,
    so the check in :func:`index_document_for_retrieval` compares like with
    like. Resolved with ``getattr`` and documented defaults for the same reason
    the chunker resolves it that way: the configuration block is owned by
    another task, and a module that raised ``AttributeError`` because a setting
    had not landed yet would take Q&A down with it.
    """
    provider = config.embedding_provider_name()
    model = str(getattr(config, "EMBEDDING_MODEL", "") or "")
    dimension = int(getattr(config, "EMBEDDING_DIMENSION", 0) or 0)
    if not model or dimension <= 0:
        # No usable embedding configuration: the caller will get the error from
        # the chunker, which is the one place that can say what is missing.
        return ""
    return build_index_version(provider=provider, model=model, dim=dimension)


def _embedder_identity(embedder: Any) -> dict[str, Any] | None:
    """``{"provider", "model", "dim"}`` the embedder will produce, or ``None``.

    ``None`` means the object does not declare an identity at all, which is not
    the same as declaring a wrong one. The real gateway -- vendored or local --
    carries all three, because it is the same object that stamps
    ``index_version`` onto the ledger and the retrieval record. A bare test
    double often does not, and refusing to index for a duck-typed embedder
    would make this function untestable without a real provider, which is the
    condition it currently suffers from.

    The asymmetry is deliberate and worth being explicit about: an embedder that
    *says* something is held to it, and one that says nothing is trusted. The
    alternative -- requiring all three -- makes "provider agnosticism" a
    property of a test rather than of the product.
    """
    provider = str(getattr(embedder, "provider", "") or "").strip()
    model = str(getattr(embedder, "model", "") or "").strip()
    try:
        dimension = int(getattr(embedder, "dimension", 0) or 0)
    except (TypeError, ValueError):
        return None
    if not (provider and model and dimension > 0):
        return None
    return {"provider": provider, "model": model, "dim": dimension}


def _assert_embedder_matches_index(embedder: Any, index_version: str) -> None:
    """Refuse to write vectors from a space the index does not claim to hold.

    The store can only catch a *dimension* change, because a width is a property
    of a row it already has. A same-width model change is invisible to it: the
    vectors are all the right length, they are just in a different space, and a
    cosine similarity between them is a confident meaningless number. The chunk
    ``index_version`` encodes provider + model + dim precisely so this check has
    something to compare against, and it is compared *before* the first write
    rather than discovered as bad answers days later.

    An embedder that declares no identity is let through, and the reason is in
    :func:`_embedder_identity`. The dimension half of the comparison is not
    optional, though: :meth:`index_document_for_retrieval` checks it separately,
    because ``dimension`` is declared by every embedder that has one, including
    the test doubles.
    """
    identity = _embedder_identity(embedder)
    if identity is None:
        return
    parts = index_version_parts(index_version)
    if not parts:
        return
    for key, claimed in identity.items():
        recorded = parts.get(key, "")
        if str(recorded) != str(claimed):
            raise EmbeddingDimensionMismatchError(
                f"embedding provider 与索引不一致：{key} 索引记录为 {recorded!r}，"
                f"而当前 embedder 声明为 {claimed!r}。两者维度相同时 store 不会报错，"
                "但向量属于不同的空间，检索结果会是无意义的相似度。"
                "请修正配置后重建索引，不要混用不同模型的向量。"
            )


def _embed_chunk_texts(embedder: Any, texts: list[str]) -> list[Sequence[float]]:
    """Embed every chunk in as few provider calls as the embedder allows.

    The batched :meth:`~paperwhisperer.llm.embeddings.EmbeddingGateway.embed_documents`
    path is preferred, and the per-chunk ``embed_document`` is the fallback for
    an embedder that only offers that. The preference is not an optimisation
    detail: ``embed_documents`` is the surface that collapses duplicate texts
    before the request, consults the content-hash cache, and accumulates the
    usage ledger. Calling ``embed_document`` in a loop bypasses all three, so a
    paper that quotes itself would be paid for twice and a run would report no
    cost at all.

    That the per-chunk call used to be the *only* call is why no cost has ever
    been reported for an embedding run in this project: the method it called
    does not exist on the gateway the configuration points at.
    """
    batched = getattr(embedder, "embed_documents", None)
    if callable(batched):
        batch = batched(texts)
        return [tuple(vector) for vector in batch]
    single = getattr(embedder, "embed_document", None)
    if not callable(single):
        raise TypeError(
            "embedder 既没有 embed_documents() 也没有 embed_document()，"
            "无法为 chunk 生成向量。请传入 EmbeddingGateway 的实例"
            "（或任何实现 embed_documents 的对象）。"
        )
    return [tuple(single(text)) for text in texts]


#: The process-wide configured embedder, built on first use.
#:
#: Module-global for the same reason the store bundle and the retriever cache
#: are: a provider is a process resource, and a personal-tier process must not
#: build one per document. Cached so the fastembed model (hundreds of megabytes,
#: loaded once) is shared by every index and every query in the process.
_CONFIGURED_EMBEDDER: Any = None
_CONFIGURED_EMBEDDER_RESOLVED = False


def _configured_embedder() -> Any:
    """The embedder this deployment is configured to use, or ``None``.

    This is the seam that closes D2. The indexer already reads the embedding
    *configuration* (:func:`_embedding_index_version`) to decide what the index
    claims to hold, and it already refuses to write vectors from a space that
    claim does not match (:func:`_assert_embedder_matches_index`) — so obtaining
    the matching provider from that same configuration is the same decision, not
    a new one. What was missing was a *caller*: the upload path
    (:func:`paperwhisperer.api.routes_v1.persist_document`) and the Q&A path both
    pass ``embedder=None`` and nothing in production ever constructed a provider,
    so every uploaded document was indexed lexically-only and the dense leg
    abstained on every question while the index advertised an embedding model it
    had never used.

    Returning ``None`` is a supported answer, not a failure: a deployment with no
    embedding provider configured — or one whose optional extra is not installed —
    is a legitimate personal-tier state and must index lexically rather than
    refuse. The caller logs the reason and continues.
    """
    global _CONFIGURED_EMBEDDER, _CONFIGURED_EMBEDDER_RESOLVED
    if _CONFIGURED_EMBEDDER_RESOLVED:
        return _CONFIGURED_EMBEDDER
    _CONFIGURED_EMBEDDER_RESOLVED = True
    try:
        from paperwhisperer.llm.embeddings_local import build_embedder_from_config

        _CONFIGURED_EMBEDDER = build_embedder_from_config()
    except Exception:  # noqa: BLE001 - a provider that cannot be built is "no provider"
        logger.warning("构建配置的 embedding provider 失败，本次索引按纯词法进行。", exc_info=True)
        _CONFIGURED_EMBEDDER = None
    return _CONFIGURED_EMBEDDER


def index_document_for_retrieval(
    bundle: StoreBundle, document: ParsedDocument, embedder: Any = None
) -> int:
    """Chunk ``document`` into ``bundle`` and make it retrievable. Idempotent.

    The idempotence check runs *before* chunking, and that ordering is the
    point. Checking afterwards means every question re-splits the whole document
    only to discover the result matches what is already stored, so a repeated
    question costs a full chunking pass and, with an embedder, a full batch of
    embedding calls. A Q&A turn is asked about a document that was just
    analysed, so that is the common case rather than an edge one.

    The version string is what makes the check safe: a document indexed under a
    different embedding model has chunk ids from a different identity and is
    re-indexed rather than trusted, because a stale index answered with
    confidence is worse than a slow one.

    **When the caller passes no embedder, the configured one is used.**
    ``embedder=None`` is not "index lexically-only" — it is "I have no provider
    to hand you", and the only place that knows which provider the deployment is
    configured for is this module (see :func:`_configured_embedder`). Resolving it
    here is what makes an uploaded document actually get vectors. A caller that
    genuinely wants a lexical-only index can still have one, by configuring no
    provider — which is a deployment decision rather than a default.

    **A document indexed without vectors is re-indexed when a provider arrives.**
    The early return additionally requires the vectors to be there whenever a
    provider is in play, so a document that was indexed lexically-only once does
    not stay lexical-only forever.

    **The embedder is checked against the index before anything is written.**
    See :func:`_assert_embedder_matches_index`; the store cannot detect a
    same-width model change, so this is the only place it is caught.
    """
    document_id = str(document.document_id)
    if embedder is None:
        embedder = _configured_embedder()
    wanted_version = _embedding_index_version()
    existing = bundle.metadata.get_document(document_id)
    stored_chunks = bundle.metadata.list_chunks(document_id) if existing else []
    if (
        existing
        and wanted_version
        and existing.index_version == wanted_version
        and stored_chunks
        and bundle.lexical.count_chunks() > 0
        and not (embedder is not None and not _has_vectors(bundle, stored_chunks))
    ):
        return len(stored_chunks)

    chunks = chunk_document_from_config(document)
    if not chunks:
        return 0

    _assert_embedder_matches_index(embedder, chunks[0].index_version)

    bundle.metadata.upsert_document(
        DocumentRecord(
            document_id=document_id,
            document=document,
            source_filename=document.source_filename,
            media_type=document.media_type,
            sha256=document.sha256,
            page_count=document.page_count,
            parser_name=document.parser_name,
            index_version=chunks[0].index_version,
        )
    )
    for chunk in chunks:
        metadata = {"heading": chunk.heading_path[-1] if chunk.heading_path else ""}
        bundle.metadata.put_chunk(document_id, chunk)
        bundle.lexical.index_chunk(
            chunk.chunk_id, chunk.text, document_id=document_id, metadata=metadata
        )

    if embedder is not None:
        _store_chunk_vectors(bundle, document_id, chunks, embedder)
    return len(chunks)


def _has_vectors(bundle: StoreBundle, chunks: list) -> bool:
    """Whether this document's chunks already carry vectors.

    Probed on the first chunk rather than counted, deliberately. A count would
    mean a store-level query whose cost grows with the corpus, on a path that
    runs before every Q&A turn to answer a yes/no question; one indexed lookup
    is O(1) and answers the same question, because chunk ids are content-addressed
    over the index version -- so a document indexed under a different embedding
    model has *different* chunk ids, and this probe correctly reports "no
    vectors" for it rather than finding some other version's row.
    """
    if not chunks:
        return False
    try:
        return bundle.vectors.get_vector(chunks[0].chunk_id) is not None
    except Exception:  # noqa: BLE001 - a store that cannot answer says "not indexed"
        logger.warning("读取已存向量失败，将重新索引该文档。", exc_info=True)
        return False


def _store_chunk_vectors(
    bundle: StoreBundle, document_id: str, chunks: list, embedder: Any
) -> None:
    """Embed every chunk and write the vectors, in as few calls as possible.

    The whole document is embedded in one :func:`_embed_chunk_texts` call before
    any vector is written. Two reasons, and the second is the one that matters:
    a per-chunk loop against an HTTP provider is one request per chunk, which
    turns an index rebuild into a rate-limit problem; and embedding everything
    first means a dimension mismatch or an unreachable provider fails *before*
    the store holds a partial index, so the document is either fully indexed or
    not indexed at all. A half-indexed document is the state where a retrieval
    returns some passages from the new space and some from the old one.
    """
    texts = [chunk.text for chunk in chunks]
    vectors = _embed_chunk_texts(embedder, texts)
    if len(vectors) != len(chunks):
        raise ValueError(
            f"embedder 返回了 {len(vectors)} 个向量，但需要 {len(chunks)} 个。"
            "数量对不上意味着向量与 chunk 没有一一对应，写进去就是错位的索引。"
        )
    width = len(vectors[0]) if vectors else 0
    if width and any(len(vector) != width for vector in vectors):
        raise ValueError("embedder 对不同 chunk 返回了不同宽度的向量，无法写入同一个向量空间。")
    entries = [
        VectorEntry(
            chunk_id=chunk.chunk_id,
            vector=tuple(vector),
            document_id=document_id,
            metadata={"heading": chunk.heading_path[-1] if chunk.heading_path else ""},
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    bundle.vectors.upsert_many(entries)


#: The process-wide store bundle, built on first Q&A use. Module-global for the
#: same reason it is in ``routes_evidence``: one connection pool per process
#: rather than one per click or per question, and both tiers share one
#: lifecycle. The Q&A path writes to it (indexing) and the evidence API reads
#: from it, which is why they must be the *same* bundle: a document indexed
#: into one and resolved from the other would be a citation that opens onto a
#: document the store has never heard of.
_QA_STORES: StoreBundle | None = None

#: Retrievers already built, keyed by *both* the store and the document. Both
#: halves are load-bearing. Keying on the document alone was a real bug: a test
#: bundle is closed and torn down between cases while a retriever over it
#: survives in the cache, and the next caller inherits a retriever whose store
#: no longer exists — surfacing as "数据库连接已关闭" from a retrieval that
#: looks like it never ran. Keying on the bundle's identity as well costs one
#: tuple and makes the cache's lifetime the shorter of the two resources'.
#:
#: The retriever holds no query state, so a second request cannot observe
#: anything the first left behind; the per-document trigram index behind the
#: fuzzy leg is the only cache and it lives on the retriever itself.
_QA_RETRIEVERS: dict[tuple[int, str], HybridRetriever] = {}


def get_store_bundle() -> StoreBundle:
    """The shared store bundle, built on first call.

    Defers to :func:`~paperwhisperer.infra.stores.build_stores` rather than
    picking a database, so the Q&A path never learns which tier answered it.
    """
    global _QA_STORES
    if _QA_STORES is None:
        _QA_STORES = build_stores(config)
    return _QA_STORES


def build_qa_retriever(
    bundle: StoreBundle, embedder: Any = None, *, document: ParsedDocument | None = None
) -> HybridRetriever:
    """Index ``document`` if needed and return a retriever over ``bundle``.

    When the caller passes no embedder, the configured one is resolved (see
    :func:`_configured_embedder`) and used both to index and to embed the query,
    so the dense leg runs rather than abstaining. Indexing with vectors but
    querying without them would leave the "hybrid dense + lexical" claim half
    true: the store would hold a vector space the query never consults.
    """
    if embedder is None:
        embedder = _configured_embedder()
    if document is None:
        return HybridRetriever(bundle, embeddings=embedder, reranker=None)

    key = (id(bundle), str(document.document_id))
    cached = _QA_RETRIEVERS.get(key)
    if cached is not None:
        return cached
    index_document_for_retrieval(bundle, document, embedder)
    retriever = HybridRetriever(bundle, embeddings=embedder, reranker=None)
    _QA_RETRIEVERS[key] = retriever
    return retriever


@dataclass(slots=True, frozen=True)
class AnswerClaim:
    """One assertion the answer makes, and the evidence it is carried by.

    A claim is the unit confidence is computed over, and it is deliberately
    *not* a sentence of prose: it is (text, evidence_ids) plus the verdict on
    whatever the model claimed to quote. A claim whose quote aligned carries
    ``presentation = DIRECT_QUOTE`` and its own ``verified_text`` — the
    characters the paper actually wrote, taken from the source rather than from
    the reply. A claim whose quote did not align carries ``PARAPHRASE`` and has
    no quotation marks to render, so there is no state in which an unverified
    string can be shown as though the paper said it.
    """

    text: str
    evidence_ids: tuple[str, ...] = ()
    quote: str = ""
    presentation: ClaimPresentation = ClaimPresentation.PARAPHRASE
    verified_text: str = ""
    outcome: str = ""
    message: str = ""
    locator: SourceLocator | None = None
    citation_label: str = ""

    @property
    def is_direct_quote(self) -> bool:
        """Whether this claim's text may be shown in quotation marks."""
        return self.presentation is ClaimPresentation.DIRECT_QUOTE

    @property
    def uses_quotation_marks(self) -> bool:
        """The name the front end reads. See ``rag.citations`` for why both."""
        return self.is_direct_quote

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "evidence_ids": list(self.evidence_ids),
            "quote": self.quote,
            "presentation": self.presentation.value,
            "uses_quotation_marks": self.uses_quotation_marks,
            "is_direct_quote": self.is_direct_quote,
            "verified_text": self.verified_text,
            "outcome": self.outcome,
            "message": self.message,
            "locator": self.locator.to_dict() if self.locator else None,
            "citation_label": self.citation_label,
        }


@dataclass(slots=True, frozen=True)
class AnswerCitation:
    """One retrieved passage a claim points at, in the shape the UI opens.

    Distinct from :class:`AnswerClaim` because they answer different questions
    and are counted differently. A claim is something the answer *asserts*; a
    citation is somewhere a reader can *go*. Keeping them separate is what lets
    the front end offer a chip for every citable claim while still rendering
    the unverified ones inline as a degraded paraphrase with no chip — a chip
    beside an unverified sentence is a chip the reader reads as a citation.
    """

    evidence_id: str
    chunk_id: str
    citation_label: str
    locator: SourceLocator | None
    heading_path: tuple[str, ...] = ()
    text: str = ""
    score: float = 0.0
    is_direct_quote: bool = False
    presentation: ClaimPresentation = ClaimPresentation.PARAPHRASE
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "chunk_id": self.chunk_id,
            "citation_label": self.citation_label,
            "locator": self.locator.to_dict() if self.locator else None,
            "heading_path": list(self.heading_path),
            "text": self.text,
            "score": self.score,
            "is_direct_quote": self.is_direct_quote,
            "presentation": self.presentation.value,
            "uses_quotation_marks": self.is_direct_quote,
            "message": self.message,
        }


@dataclass(slots=True, frozen=True)
class RetrievalSummary:
    """What retrieval did, in the few fields a reader or a bug report needs.

    Carried on the answer rather than logged because the two questions "why did
    this answer cite that passage" and "why did this answer have no citations
    at all" are both asked of the *response*, and a record that is only in a log
    can answer neither for a user. The full record — every candidate, its legs,
    and why the losers lost — stays on ``RetrievalRecord`` for the retrieval
    side of the system to keep.
    """

    record_id: str = ""
    task: str = TASK_QA
    evidence_count: int = 0
    empty_reason: str = ""
    degradations: tuple[str, ...] = ()
    plan_signature: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "task": self.task,
            "evidence_count": self.evidence_count,
            "empty_reason": self.empty_reason,
            "degradations": list(self.degradations),
            "plan_signature": self.plan_signature,
        }


@dataclass(slots=True, frozen=True)
class AnswerResult:
    """The whole answer, as a value rather than a dict the route happens to return.

    Four fields, and each answers a question a string cannot:

    ``answer``
        The prose, for a reader and for the session transcript.
    ``claims``
        What was actually asserted, each with the evidence carrying it and the
        verdict on any quoted span. The release gate is made of this list.
    ``citations``
        The passages those claims point at, in the shape a chip opens.
    ``confidence``
        One of :data:`CONFIDENCE_VALUES`, **derived from ``claims`` and
        nothing else.** The model is not asked for it and its self-assessment
        is not read, because a model asked whether its own answer is supported
        will say yes, and the failure mode is not a wrong answer — it is no
        answer at all.

    Frozen, so a route cannot mutate a result after deciding how to render it,
    and ``to_dict`` is the only way out, which is what keeps the wire shape in
    one place.
    """

    answer: str
    claims: tuple[AnswerClaim, ...] = ()
    citations: tuple[AnswerCitation, ...] = ()
    confidence: str = CONFIDENCE_INSUFFICIENT
    retrieval: RetrievalSummary = field(default_factory=RetrievalSummary)
    notes: tuple[str, ...] = ()
    answer_mode: str = "evidence"

    @property
    def insufficient_evidence(self) -> bool:
        """Whether this answer is a declared non-answer rather than an answer."""
        return self.confidence == CONFIDENCE_INSUFFICIENT

    def to_dict(self) -> dict[str, Any]:
        """The wire shape. One function, so the route cannot reshape it."""
        return {
            "answer": self.answer,
            "claims": [item.to_dict() for item in self.claims],
            "citations": [item.to_dict() for item in self.citations],
            "confidence": self.confidence,
            "insufficient_evidence": self.insufficient_evidence,
            "retrieval": self.retrieval.to_dict(),
            "notes": list(self.notes),
            "answer_mode": self.answer_mode,
        }


class DocumentAnalyzer:
    """Owns one analysis run: document text, section prompts and LLM dispatch."""

    def __init__(
        self,
        api_key: str,
        *,
        store_bundle: StoreBundle | None = None,
        embedder: Any = None,
        qa_document: ParsedDocument | None = None,
    ):
        self.version = config.APP_VERSION
        self.client = LLMClient(api_key)
        self.chunker = TextChunker(4000, 200)
        self.summary_chunk_workers = min(3, self.client.max_concurrency)
        self.analysis_workers = min(5, self.client.max_concurrency)
        self.document_content = ""
        #: How the current document was parsed. Kept on the instance because
        #: the Q&A and recommendation routes reuse the same analyzer after the
        #: analysis run, and "which parser produced the text I am asking about"
        #: is still the right question to be able to answer there.
        self.parse_outcome: ParseOutcome | None = None
        #: The stores Q&A retrieves from, and the embedding provider whose
        #: vectors index it. Both are injected rather than constructed so a
        #: test can run the whole Q&A path against a temporary store and a
        #: deterministic embedder, and so a deployment with no embedding
        #: endpoint still answers from the lexical leg instead of failing.
        self.store_bundle = store_bundle
        self.embedder = embedder
        #: The parsed document Q&A retrieves over, when the caller has one.
        #: Preferred over re-deriving a document from the flat text, and over
        #: re-parsing: the parse already happened, and a second one could
        #: produce a different document for the same file.
        self.qa_document = qa_document
        self._retriever: HybridRetriever | None = None

    # -- document ----------------------------------------------------------

    def _load_document(self, file_path) -> None:
        """Parse ``file_path`` once, through the gateway, and keep the text.

        One parse per run, shared by every section. The alternative — letting
        each section load the file — would multiply the parse cost by six and,
        worse, could give two sections different text for the same file if the
        backend or the cache changed underneath them. A summary and a mindmap
        that disagree because they read different bytes is a bug no amount of
        prompt tuning fixes.
        """
        loaded = DocumentLoader.load_with_outcome(file_path)
        self.document_content = loaded.text
        self.parse_outcome = loaded.outcome
        # The parsed document is kept for Q&A rather than discarded after the
        # flat text is taken. T1.7 needs elements, locators and chunkable
        # structure to retrieve over and to verify quotes against, and none of
        # those survive ``to_full_text``; re-deriving them from the flat text
        # would give every citation in a follow-up question a weaker locator
        # than the analysis that ran a moment earlier already had.
        if self.qa_document is None:
            self.qa_document = loaded.outcome.document
        logger.info(
            "Parsed %s via %s (%d chars, %s)",
            os.path.basename(str(file_path)),
            loaded.outcome.parsed_by or "unknown parser",
            len(loaded.text),
            "cloud" if loaded.outcome.used_cloud else "local",
        )

    # -- section implementations ------------------------------------------

    def _generate_summary_chunk(self, chunk: str) -> str:
        system_prompt, user_prompt = sections.build_summary_chunk_prompt(chunk)
        return self.client.complete(system_prompt, user_prompt)

    def _merge_summaries(self, summaries: list[str]) -> str | None:
        if not summaries:
            return None
        if len(summaries) == 1:
            return summaries[0]

        combined = "\n\n--- 章节 ---\n\n".join(summaries)
        system_prompt, user_prompt = sections.build_summary_merge_prompt(combined)
        return self.client.complete(system_prompt, user_prompt)

    def generate_summary(self, content: str) -> str:
        chunks = self.chunker.chunk_text(content)
        if len(chunks) == 1:
            return self._generate_summary_chunk(content)

        worker_count = self._get_worker_count(len(chunks), self.summary_chunk_workers)
        if worker_count == 1:
            chunk_summaries = [
                summary for summary in map(self._generate_summary_chunk, chunks) if summary
            ]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
                chunk_summaries = list(
                    filter(None, executor.map(self._generate_summary_chunk, chunks))
                )

        if len(chunk_summaries) > 1:
            return self._merge_summaries(chunk_summaries)
        return chunk_summaries[0] if chunk_summaries else "无法生成摘要"

    def extract_quotes(self, content: str) -> str:
        system_prompt, user_prompt = sections.build_quotes_prompt(content)
        return self.client.complete(system_prompt, user_prompt)

    def generate_mindmap(self, content: str) -> str:
        system_prompt, user_prompt = sections.build_mindmap_prompt(content)
        return self.client.complete(system_prompt, user_prompt)

    def generate_mermaid_mindmap(self, content: str) -> str | None:
        system_prompt, user_prompt = sections.build_mermaid_prompt(content)
        result = self.client.complete(system_prompt, user_prompt)
        if not result:
            return None
        return self._normalize_mermaid_source(result)

    @staticmethod
    def _normalize_mermaid_source(result: str) -> str:
        """Strip preamble so the payload starts at the diagram declaration.

        Models often emit a sentence of explanation before the diagram; the
        frontend renders the string verbatim, so that preamble would break the
        parse. Fenced fallbacks are handled for the rare case where no known
        diagram keyword appears.
        """
        lines = result.strip().split("\n")
        start_index = -1
        for index, line in enumerate(lines):
            if any(line.strip().startswith(prefix) for prefix in sections.MERMAID_DIAGRAM_PREFIXES):
                start_index = index
                break

        if start_index != -1:
            result = "\n".join(lines[start_index:]).strip()
        else:
            match = re.search(
                r"```(?:mermaid)?\s*\n(.*?)\n```", result, re.DOTALL | re.IGNORECASE
            )
            result = match.group(1).strip() if match else "graph TD\n" + result

        if not any(result.startswith(prefix) for prefix in sections.MERMAID_DIAGRAM_PREFIXES):
            result = "graph TD\n" + result
        return DocumentAnalyzer._strip_mermaid_fence(result)

    @staticmethod
    def _strip_mermaid_fence(source: str) -> str:
        """Drop the code fence that follows the diagram body.

        Slicing from the diagram keyword keeps everything after it, including
        the closing ``` of the block the model wrapped the diagram in. Left in
        place that fence reaches the renderer as diagram source and the parse
        fails, so it has to come off here rather than at render time.
        """
        lines = source.split("\n")
        end = len(lines)
        while end > 1 and lines[end - 1].strip() in {"```", "~~~"}:
            end -= 1
        body = "\n".join(lines[:end]).rstrip()
        # A model can also close the fence without a newline before it.
        return re.sub(r"(?:\s*(?:```|~~~))\s*$", "", body)

    def generate_evaluation(self, content: str) -> str:
        system_prompt, user_prompt = sections.build_evaluation_prompt(content)
        return self.client.complete(system_prompt, user_prompt)

    def generate_research_brief(self, content: str) -> str:
        system_prompt, user_prompt = sections.build_research_brief_prompt(content)
        return self.client.complete(system_prompt, user_prompt)

    # -- Q&A ---------------------------------------------------------------
    #
    # Everything below is T1.7. The rule the whole section is built around:
    # **the answer's knowledge source is retrieved evidence or nothing.**
    # The previous implementation sent ``document_content[:12000]`` — a hard
    # prefix of the flat text, measured against a real paper to cut 132
    # characters before the sentence reporting its result, which made every
    # section past the third unreachable. There is deliberately no fallback to
    # that window: if retrieval returns nothing the answer is
    # ``insufficient_evidence``, because a retriever that cannot find evidence
    # and then hands back the document head has invented an answer, which is
    # the single outcome this product exists to prevent.

    @staticmethod
    def _build_history_block(history) -> str:
        history = history or []
        history_sections = []
        used_history_chars = 0

        for turn in reversed(history):
            question_text = trim_text_for_log(turn.get("question", ""), limit=400)
            answer_text = trim_text_for_log(turn.get("answer", ""), limit=800)
            if not question_text and not answer_text:
                continue

            section = f"Q: {question_text}\nA: {answer_text}"
            if len(section) > config.QA_TURN_BUDGET:
                section = section[: config.QA_TURN_BUDGET].rstrip() + "\n...[truncated]"

            if used_history_chars + len(section) > config.QA_HISTORY_BUDGET:
                break

            history_sections.append(section)
            used_history_chars += len(section)

        history_sections.reverse()
        return "\n\n---\n\n".join(history_sections)

    def _retrieve_for_question(self, question) -> RetrievalResult:
        """Run the Q&A retrieval plan over the analyzer's parsed document.

        A missing document is a :class:`ValueError` rather than an empty
        retrieval, and the distinction is the user's. "You have not uploaded
        anything" is something the client can fix and should be told plainly;
        "the paper does not say this" is a finding about the paper. Collapsing
        them into one "insufficient evidence" answer would report a missing
        upload as though the document had been searched and come up empty.
        """
        document = self._require_qa_document()
        retriever = self._qa_retriever()
        return retriever.retrieve(
            question, plan=plan_for(TASK_QA), document_ids=[document.document_id]
        )

    def _require_qa_document(self) -> ParsedDocument:
        """The parsed document Q&A reads, or a ``ValueError`` saying there is none."""
        document = self._qa_document()
        if document is None:
            raise ValueError("没有文档内容，请先上传文档进行分析。")
        return document

    def _qa_document(self) -> ParsedDocument | None:
        """The parsed document Q&A reads, or ``None`` when there is none.

        Prefers the document the gateway already produced, so a Q&A turn after
        an analysis in the same process reuses the parse instead of paying for
        a second one. Falls back to wrapping the flat text when only that is
        available — the session stores text, not a ``ParsedDocument`` — and in
        that case the fallback says so in ``warnings``, because a citation into
        a re-derived document is a weaker claim than one into a parsed one.
        """
        if self.qa_document is not None:
            return self.qa_document
        if self.parse_outcome is not None and self.parse_outcome.document is not None:
            return self.parse_outcome.document
        if not self.document_content:
            return None
        return parsed_document_from_text(self.document_content)

    def _qa_retriever(self) -> HybridRetriever:
        """A retriever over a store that has this analyzer's document indexed.

        The index is built once per analyzer, on first use. Building it is
        cheap relative to the parse it would otherwise repeat, and doing it
        lazily is what lets ``answer_question`` stay a pure function of the
        question: no work happens for a turn that is never asked.
        """
        if self._retriever is not None:
            return self._retriever
        bundle = self.store_bundle if self.store_bundle is not None else get_store_bundle()
        self._retriever = build_qa_retriever(bundle, self.embedder, document=self._qa_document())
        return self._retriever

    def _build_evidence_answer_prompts(
        self, question, history=None, answer_mode="evidence", *, result: RetrievalResult
    ):
        """Assemble the prompt from retrieved evidence and nothing else.

        The evidence block is the retrieved passages, each tagged with the
        id the answer must cite it by. Nothing about the document beyond those
        passages is present, which is the mechanical form of "no head excerpt":
        a fact the retriever did not surface is not in the prompt, so a model
        asked about the experiments section either sees those passages or has
        nothing to answer from.
        """
        history_block = self._build_history_block(history)
        return build_evidence_answer_prompts(
            question=question,
            evidence=result.evidence,
            history_block=history_block,
            answer_mode=normalize_answer_mode(answer_mode),
        )

    def answer_question(self, question, history=None, answer_mode="evidence") -> AnswerResult:
        """Answer one question from retrieved evidence, with its structure.

        Returns an :class:`AnswerResult` rather than a string, because the
        three things the caller actually needs — which claims were made, which
        passages back them, and how far the answer can be trusted — cannot be
        recovered from prose after the fact. The string is still there, on
        ``result.answer``, so a consumer that only wants the text is unchanged.

        No evidence means no model call. See :meth:`_insufficient_answer`.
        """
        result = self._retrieve_for_question(question)
        if result.insufficient_evidence:
            return self._insufficient_answer(question, result, answer_mode)

        system_prompt, user_prompt = self._build_evidence_answer_prompts(
            question, history=history, answer_mode=answer_mode, result=result
        )
        raw = self.client.complete(system_prompt, user_prompt)
        return self._compose_answer_result(
            result, raw, question=question, answer_mode=answer_mode
        )

    def stream_answer_question(self, question, history=None, answer_mode="evidence"):
        """Yield the verified answer in pieces, then the :class:`AnswerResult`.

        The model is streamed at the transport level — that is the call that
        decides latency, retry behaviour and token cost, and none of it is
        affected by what is done with the result — but its output is
        **buffered and verified before any of it reaches the caller.** The
        naive version of this method, forwarding each chunk as it arrives,
        would be a release-gate violation: the model writes a quote it invented
        and the browser has already rendered it in quotation marks before
        anything gets the chance to check. Presentation has to come after
        verification, so the deltas are cut from the finished answer.

        So the pieces are a rendering convenience, not a claim that tokens
        arrived one at a time. They exist so the front end's incremental path
        and the SSE contract keep working unchanged, and a client that would
        rather wait is free to do exactly that.
        """
        result = self._retrieve_for_question(question)
        if result.insufficient_evidence:
            answer = self._insufficient_answer(question, result, answer_mode)
            yield from _stream_pieces(answer.answer)
            yield answer
            return

        system_prompt, user_prompt = self._build_evidence_answer_prompts(
            question, history=history, answer_mode=answer_mode, result=result
        )
        accumulated: list[str] = []
        for chunk in self.client.stream(system_prompt, user_prompt):
            accumulated.append(chunk)
        composed = self._compose_answer_result(
            result, "".join(accumulated), question=question, answer_mode=answer_mode
        )
        yield from _stream_pieces(composed.answer)
        yield composed

    def _insufficient_answer(
        self, question, result: RetrievalResult, answer_mode: str = "evidence"
    ) -> AnswerResult:
        """The answer for a question the retrieved evidence does not cover.

        The model is **not** called. This is the whole release gate, and it is
        here rather than in a post-hoc filter because a filter cannot undo an
        answer: by the time a refusal is being checked for invented numbers,
        the numbers are already written, and the measurement behind this work
        is that the model correctly refuses to invent the *figure* and then
        invents confidently in the hedging around it — asserting the paper
        reported several results and describing a section structure it was
        never shown. Asking a model to answer from no evidence, and then
        checking its answer, is asking it to do the one thing it cannot do.

        So the empty case is answered by a fixed sentence assembled from what
        retrieval actually reported. It is not a template because it cannot be
        a template: there is no model output in it at all, which is the
        property the release gate is made of.
        """
        reason = result.empty_reason
        message = EMPTY_MESSAGES.get(reason) or EMPTY_MESSAGES["no_candidates"]
        return AnswerResult(
            answer=f"证据不足：{message}",
            claims=(),
            citations=(),
            confidence=CONFIDENCE_INSUFFICIENT,
            retrieval=RetrievalSummary(
                record_id=result.record.record_id,
                task=result.record.task,
                evidence_count=0,
                empty_reason=result.empty_reason,
                degradations=result.record.degradations,
                plan_signature=result.record.plan_signature,
            ),
            answer_mode=normalize_answer_mode(answer_mode),
        )

    def _compose_answer_result(
        self, result: RetrievalResult, raw: str, *, question: str, answer_mode: str
    ) -> AnswerResult:
        """Turn a model reply plus the retrieval that framed it into an AnswerResult.

        The order of operations is the design. The reply is parsed into claims,
        each claim's quoted span is verified against the evidence it names, the
        verified text replaces what the model wrote, and only then is
        confidence computed — from the claims that survived, not from anything
        the model said about itself. Confidence is therefore a function of the
        evidence the answer actually carries, which is the only definition of
        "supported" that does not put the model in charge of grading its own
        work.
        """
        answer_text, claims, notes = parse_claim_payload(raw)
        evidence_by_id = {item.evidence_id: item for item in result.evidence}

        verified: list[AnswerClaim] = []
        for claim in claims:
            evidence = [evidence_by_id[item] for item in claim.evidence_ids
                        if item in evidence_by_id]
            verified.append(_verify_claim(claim, evidence, self._qa_document()))

        confidence = _derive_confidence(verified, relevance=result.relevance)
        # A refusal wrapped in invented context is worse than no refusal, so an
        # answer nothing can be tied to is not shown at all — not even as a
        # caveat. The model was given real passages and still produced claims
        # that resolve to nothing, which means its prose is assertion rather
        # than report; rendering it would put a number on screen that the
        # product has just established it cannot vouch for. The passages are
        # still returned, so a reader can go and look for themselves, which is
        # the useful thing to do with a question the product could not answer.
        if confidence == CONFIDENCE_INSUFFICIENT:
            return AnswerResult(
                answer=UNVERIFIED_REFUSAL,
                claims=(),
                citations=(),
                confidence=CONFIDENCE_INSUFFICIENT,
                retrieval=RetrievalSummary(
                    record_id=result.record.record_id,
                    task=result.record.task,
                    evidence_count=len(result.evidence),
                    empty_reason=result.record.empty_reason,
                    degradations=result.record.degradations,
                    plan_signature=result.record.plan_signature,
                ),
                notes=tuple(notes),
                answer_mode=normalize_answer_mode(answer_mode),
            )

        return AnswerResult(
            answer=_strip_unverified_quotes(answer_text, verified),
            claims=tuple(verified),
            citations=tuple(_citations_for(verified, result)),
            confidence=confidence,
            retrieval=RetrievalSummary(
                record_id=result.record.record_id,
                task=result.record.task,
                evidence_count=len(result.evidence),
                empty_reason=result.record.empty_reason,
                degradations=result.record.degradations,
                plan_signature=result.record.plan_signature,
            ),
            notes=tuple(notes),
            answer_mode=normalize_answer_mode(answer_mode),
        )

    # -- search rewrite / recommendation ----------------------------------

    def rewrite_search_query(self, query, context_text: str = "") -> dict:
        clean_query = compact_text(query, limit=240)
        if not clean_query:
            raise ValueError("Please enter a search query.")

        context_excerpt = sections.document_excerpt(
            context_text or "", limit=config.SEARCH_REWRITE_CONTEXT_LIMIT
        )
        system_prompt, user_prompt = sections.build_search_rewrite_prompt(
            clean_query, context_excerpt
        )
        raw_response = self.client.complete(
            system_prompt, user_prompt, model=self.client.search_rewrite_model
        )
        return sections.normalize_search_rewrite(
            raw_response, clean_query, self.client.search_rewrite_model
        )

    def recommend_papers(self, content, limit=None, search_fn=None) -> dict:
        excerpt = sections.document_excerpt(content, limit=12000)
        if not excerpt:
            raise ValueError("Current session does not contain document content.")

        resolved_limit = config.clamp_int_value(
            limit, config.RECOMMENDATION_RESULT_LIMIT, min_value=1,
            max_value=config.RECOMMENDATION_RESULT_LIMIT,
        )
        rewrite_meta = self.rewrite_search_query(
            query="Find closely related follow-up papers for this paper.",
            context_text=excerpt,
        )
        search_result = (search_fn or _default_search)(
            rewrite_meta.get("rewritten_query", ""), resolved_limit
        )
        return {
            "original_query": rewrite_meta.get("original_query", ""),
            "query": rewrite_meta.get("rewritten_query", ""),
            "topics": rewrite_meta.get("topics") or [],
            "reason": rewrite_meta.get("reason", ""),
            "rewrite_model": rewrite_meta.get("model", ""),
            "items": search_result.get("items", []),
            "errors": search_result.get("errors", []),
        }

    # -- orchestration -----------------------------------------------------

    def _get_worker_count(self, task_count: int, configured_workers: int) -> int:
        return max(1, min(task_count, configured_workers))

    def _resolve_section_future(self, future, enabled: bool = True) -> dict:
        if not enabled:
            return build_section_result("disabled")
        try:
            content_value = future.result()
            if not content_value:
                return build_section_result("empty")
            return build_section_result("success", content=content_value)
        except Exception as exc:
            return build_section_result(
                "failed",
                error=str(exc),
                retryable=is_retryable_llm_error(str(exc)),
            )

    def _plan_tasks(self, generate_mermaid, generate_evaluation, generate_research_brief):
        """Return (callable, section_name, enabled) for every section to run."""
        plan = [
            (self.generate_summary, "summary", True),
            (self.extract_quotes, "quotes", True),
            (self.generate_mindmap, "mindmap", True),
        ]
        if generate_mermaid:
            plan.append((self.generate_mermaid_mindmap, "mermaid", True))
        if generate_evaluation:
            plan.append((self.generate_evaluation, "evaluation", True))
        if generate_research_brief:
            plan.append((self.generate_research_brief, "research_brief", True))
        return plan

    @staticmethod
    def _initial_sections(generate_mermaid, generate_evaluation, generate_research_brief) -> dict:
        def status(enabled):
            return build_section_result("pending") if enabled else build_section_result("disabled")

        return {
            "summary": status(True),
            "quotes": status(True),
            "mindmap": status(True),
            "mermaid": status(generate_mermaid),
            "evaluation": status(generate_evaluation),
            "research_brief": status(generate_research_brief),
        }

    def _finalize_analysis_sections(self, content, sections_map) -> dict:
        result = {"char_count": len(content), "sections": sections_map}
        for key in sections.SECTION_KEYS:
            result[key] = (sections_map.get(key) or {}).get("content", "")

        result.update(build_analysis_metadata(sections_map))

        # Additive: no existing key changes name, type or meaning, so the API
        # response, the stored session payload and the templates that already
        # read ``char_count`` and ``sections`` are unaffected. A client that
        # has never heard of ``parse`` ignores it; a client that renders it can
        # now tell the user which parser ran and whether the file was uploaded.
        result["parse"] = build_parse_provenance(self.parse_outcome)

        required = [sections_map.get(key) or {} for key in sections.REQUIRED_SECTION_KEYS]
        if required and all(section.get("status") == "failed" for section in required):
            raise RuntimeError(required[0].get("error") or "核心分析项全部失败")
        return result

    def analyze(
        self,
        file_path,
        generate_mermaid: bool = True,
        generate_evaluation: bool = True,
        generate_research_brief: bool = True,
    ) -> dict:
        """Run the full analysis and return once every section has settled."""
        self._load_document(file_path)
        content = self.document_content

        plan = self._plan_tasks(generate_mermaid, generate_evaluation, generate_research_brief)
        worker_count = self._get_worker_count(len(plan), self.analysis_workers)
        started_at = time.time()

        # Seed with the disabled entries so the map covers every SECTION_KEYS.
        # Building it purely from ``plan`` would leave the switched-off sections
        # missing, and the flattening step in _finalize_analysis_sections indexes
        # all of them.
        section_results = self._initial_sections(
            generate_mermaid, generate_evaluation, generate_research_brief
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(func, content): (name, enabled)
                for func, name, enabled in plan
            }
            for future in concurrent.futures.as_completed(futures):
                name, enabled = futures[future]
                section_results[name] = self._resolve_section_future(future, enabled=enabled)

        result = self._finalize_analysis_sections(content, section_results)
        result["elapsed_seconds"] = round(time.time() - started_at, 1)

        logger.info(
            "Analysis completed in %.1fs for %s (%d chars)",
            result["elapsed_seconds"],
            os.path.basename(file_path),
            len(content),
        )
        return result

    def analyze_stream(
        self,
        file_path,
        generate_mermaid: bool = True,
        generate_evaluation: bool = True,
        generate_research_brief: bool = True,
    ):
        """Yield a ``section`` event per completion, then a final ``done`` event.

        Consumers must run this generator off the event loop: each step is a
        blocking LLM call.
        """
        self._load_document(file_path)
        content = self.document_content

        plan = self._plan_tasks(generate_mermaid, generate_evaluation, generate_research_brief)
        section_results = self._initial_sections(
            generate_mermaid, generate_evaluation, generate_research_brief
        )
        worker_count = self._get_worker_count(len(plan), self.analysis_workers)
        started_at = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(func, content): (name, enabled)
                for func, name, enabled in plan
            }
            for future in concurrent.futures.as_completed(futures):
                name, enabled = futures[future]
                section_result = self._resolve_section_future(future, enabled=enabled)
                section_results[name] = section_result
                yield {"type": "section", "name": name, "section": section_result}

        result = self._finalize_analysis_sections(content, section_results)
        result["elapsed_seconds"] = round(time.time() - started_at, 1)
        logger.info(
            "Streaming analysis completed in %.1fs for %s (%d chars)",
            result["elapsed_seconds"],
            os.path.basename(file_path),
            len(content),
        )
        yield {"type": "done", "result": result}


def _default_search(query, limit):
    """Late import to avoid a cycle between analysis and search packages."""
    from paperwhisperer.search.papers import search_papers

    return search_papers(query, limit)
