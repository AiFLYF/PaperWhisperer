"""The fact layer: one extraction, and every card a rendering of it.

Task T1.8 of the plan, and the fix for a defect that is not really a cost
problem. Before this module existed, one paper was read seven times: nine
chunk-level summary calls, a merge, and five independent section calls, each
with its own reading of the document. That is expensive *and* it is wrong,
because seven independent readings produce seven independent opinions. The
overview's "this paper contributes X" and the brief's "this paper contributes
Y" were not a rendering bug -- they were the design working exactly as written.

So the design here is not "call the model less". It is **one extraction, many
renderings**, and the acceptance criterion (跨卡片事实一致) is a structural
consequence of that rather than a property to be checked afterwards.

**What makes consistency structural.** A card is a function of
:class:`~paperwhisperer.ingestion.schema.PaperFacts` and nothing else. Look at
the signature of :func:`render_overview`, :func:`render_brief`,
:func:`render_mindmap` and :func:`build_quote_card`: one argument, the fact
layer. They are not passed the document, the section text, or an excerpt, so
they *cannot* read the paper and therefore cannot form an opinion about it. A
claim that appears on a card is a claim that is in the fact layer, because the
only way a string gets onto a card is
:func:`claim_text` returning it. That is the whole mechanism; there is no
consistency check anywhere in this module, because there is nothing to check
once the cards are pure functions.

**What makes the fact layer worth trusting.** Three rules, enforced in code
after the model has answered, never by asking the model nicely:

*Every citation resolves.* The model is only ever shown ``[ev_xxxx]`` headers
it is allowed to cite, and every id it returns is looked up in the evidence it
was shown. An id that resolves to nothing is deleted from the fact, and a fact
that loses its last citation is downgraded to ``needs_review`` rather than
kept as a confident claim. An unresolvable citation is worse than no citation
at all, because it renders as a working link to nothing: the reader clicks,
gets an empty panel, and learns nothing about whether the claim was checked.

*Every quote is verified, and the check is not the model's opinion.* The
extraction is asked for a verbatim ``quote`` per fact, and
:func:`_quote_aligned` checks it against the cited passage with
:func:`paperwhisperer.rag.citations.normalize_for_alignment` -- the same
normalisation rules the T1.6 citation verifier uses, applied here to a chunk
of text rather than to an element. No LLM, no network, no prompt asking the
model to confirm its own quotation. A fact whose quote does not align keeps its
value but is marked ``needs_review``, which is what the UI is for.

*Absence is a recorded answer.* A field the paper does not state comes back as
a ``Fact`` with an empty value and ``extraction_status="not_found"``, and for
the five list fields it comes back as a list *containing* that fact rather than
as an empty list. "The paper lists no datasets" and "we failed to extract the
datasets" are otherwise the same empty brackets on screen, and they are not the
same fact. :func:`render_card` then prints the ``not_found`` marker, so the
distinction survives all the way to the card -- which is the last place it could
have been lost.

**The Mermaid first layer is not a model output.** The document already has a
section tree: it came out of the parser as heading elements with levels and
heading paths, and it is the same structure the reader sees. Asking a model to
redraw it cannot add anything and can lose any of it, so
:func:`build_section_tree` and :func:`build_mermaid_sections` read the headings
and emit the diagram. Zero model calls, and the first level of the diagram is
*the paper's* structure rather than a plausible-looking arrangement of it. The
deeper levels are the model's (see
:func:`paperwhisperer.analysis.sections.build_mermaid_deepening_prompt`), and
that prompt is handed the tree as something it may not alter.

**Cost is a number, not an intention.** The plan's acceptance criterion is
"LLM 调用成本较旧路径下降且有对比数据", so this module measures rather than
asserts. :func:`legacy_call_budget` reconstructs the old path's calls and
prompt volume from the *real* chunker and the *real* prompt builders in
:mod:`paperwhisperer.analysis.sections`, and :func:`fact_layer_call_budget`
does the same for this one, over the same input. The comparison is a function
with no numbers baked in, so it cannot rot; the test pins the result as a
number so it cannot silently regress. See :data:`LEGACY_SUMMARY_CHUNK_CHARS`
for why the legacy planner is not simply a constant.

**Where the run record goes.** :class:`AnalysisRun` is a complete, serializable
record of one analysis -- model, prompt version, evidence version, token counts,
cost, and the embedding columns -- and :meth:`AnalysisRun.as_analysis_run_columns`
documents the exact columns T2.2 must persist. The ``analysis_runs`` table does
not exist yet and this module deliberately does not invent a half-table or open
a connection, for the same reason :mod:`paperwhisperer.llm.embeddings` does not:
a second, unofficial cost table is how a project ends up with two cost
histories that disagree. The embedding usage arrives here as an
:class:`~paperwhisperer.llm.embeddings.EmbeddingUsage` and is merged into the
same row's columns, because the dense leg's cost is part of the analysis's cost
and splitting it across two ledgers is how the split becomes invisible.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from paperwhisperer.analysis import sections
from paperwhisperer.core import config
from paperwhisperer.documents.loader import TextChunker
from paperwhisperer.infra.stores import scoring
from paperwhisperer.ingestion.schema import (
    EXTRACTION_STATUSES,
    Element,
    Evidence,
    Fact,
    PaperFacts,
    ParsedDocument,
)
from paperwhisperer.llm import output
from paperwhisperer.llm.embeddings import EmbeddingUsage
from paperwhisperer.rag.chunking import estimate_tokens
from paperwhisperer.rag.citations import normalize_for_alignment
from paperwhisperer.rag.retrieve import (
    TASK_CONCEPT,
    TASK_CONTRIBUTION,
    TASK_EXPERIMENT,
    TASK_REPRODUCTION,
    plan_for,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AnalysisRun",
    "CallBudget",
    "CardClaim",
    "ExtractionReport",
    "FactExtractor",
    "FactLayerOutcome",
    "FactLayerPipeline",
    "FACT_FIELDS",
    "Grounding",
    "ModelCall",
    "NOT_STATED_TEXT",
    "PromptVersionError",
    "SectionNode",
    "VerifiedQuote",
    "build_analysis_run",
    "build_fact_digest",
    "build_mermaid_sections",
    "build_quote_card",
    "build_run_id",
    "build_section_tree",
    "card_claims",
    "claim_text",
    "evidence_version",
    "fact_layer_call_budget",
    "ground_fact",
    "legacy_call_budget",
    "make_evidence_provider",
    "prompt_version",
    "render_brief",
    "render_evaluation",
    "render_mindmap",
    "render_overview",
    "render_section_tree_text",
]

#: The statuses, read from the canonical schema rather than restated.
#:
#: Unpacked by position on purpose. A hard-coded copy of these four strings
#: would keep working after the schema grew a fifth status, and the code that
#: has to make a decision about an unrecognised one -- here,
#: ``Fact.__post_init__``, which silently coerces it to ``not_found`` -- would
#: make that decision without anyone noticing. Unpacking means a schema change
#: stops the build, and a test asserts the two agree.
STATUS_EXTRACTED, STATUS_NOT_FOUND, STATUS_CONFLICTING, STATUS_NEEDS_REVIEW = (
    EXTRACTION_STATUSES
)

#: The statuses a *model* is allowed to return. ``needs_review`` is excluded
#: on purpose: it is the status this module assigns when a claim fails to
#: verify, and a model that could declare its own claim verified would have a
#: field for exactly the thing that must not be self-assessed.
MODEL_STATUSES = (STATUS_EXTRACTED, STATUS_NOT_FOUND, STATUS_CONFLICTING)

#: What a card prints for a field the paper does not state. A user-facing
#: string, so it is a constant rather than something each renderer spells.
NOT_STATED_TEXT = "论文未说明"

#: Prefix for a fact whose value survived but whose evidence did not verify.
#: Printed in front of the value, not in place of it: the claim is still what
#: the model said, and hiding it would make an unverified extraction look like
#: an empty one.
UNVERIFIED_PREFIX = "【未通过证据校验】"

#: The eight fields, with the retrieval task each one is answered from.
#:
#: The mapping is a declared editorial choice rather than a computed one: it
#: encodes where in a paper each kind of claim is written, which is knowledge
#: about papers and not something a query string can discover. It reuses the
#: plan section 6.4 already defined, so the fact layer adds no retrieval
#: strategy of its own -- ``test_every_field_names_a_registered_plan`` asserts
#: that every value here is a plan that exists, which is the failure that would
#: otherwise surface as a ``KeyError`` at the first real paper.
FACT_FIELDS: dict[str, dict[str, Any]] = {
    "research_question": {
        "label": "研究问题",
        "many": False,
        "task": TASK_CONCEPT,
        "query": (
            "研究问题 动机 目标 本文要解决什么问题 why this paper exists "
            "problem motivation objective"
        ),
    },
    "contributions": {
        "label": "核心贡献",
        "many": True,
        "task": TASK_CONTRIBUTION,
        "query": (
            "贡献 创新点 我们提出 本文的主要贡献 novel contribution "
            "we propose our contribution"
        ),
    },
    "method": {
        "label": "方法",
        "many": False,
        "task": TASK_REPRODUCTION,
        "query": (
            "方法 模型 结构 训练流程 algorithm architecture training procedure method"
        ),
    },
    "datasets": {
        "label": "数据集",
        "many": True,
        "task": TASK_EXPERIMENT,
        "query": "数据集 语料 benchmark 训练数据 dataset corpus training data",
    },
    "metrics": {
        "label": "评估指标",
        "many": True,
        "task": TASK_EXPERIMENT,
        "query": "评估指标 评价标准 metric evaluation measure score F1 BLEU accuracy",
    },
    "main_results": {
        "label": "主要结果",
        "many": True,
        "task": TASK_EXPERIMENT,
        "query": "主要结果 实验结果 性能提升 SOTA 消融 main results experimental results",
    },
    "limitations": {
        "label": "局限性",
        "many": True,
        "task": TASK_REPRODUCTION,
        "query": "局限性 不足 失败案例 未来工作 limitation failure case future work",
    },
    "reproducibility": {
        "label": "可复现性",
        "many": False,
        "task": TASK_REPRODUCTION,
        "query": (
            "复现 代码 开源 超参数 实现细节 reproducibility code available "
            "hyperparameters implementation details"
        ),
    },
}

#: Evidence retrieved per field. Small on purpose: the eight passes together
#: share one prompt and therefore one context window, and a generous per-field
#: budget would fill that window with the introduction eight times over while
#: the experimental section goes unread.
EVIDENCE_PER_FIELD = 6

#: The legacy chunk size, read from the orchestrator's own ``TextChunker``
#: construction. Duplicated as a named constant because the cost comparison has
#: to be checkable from this file, and because a silent divergence here would
#: make the before-number wrong in the direction that flatters the change. The
#: test pins it against the orchestrator's real chunker.
LEGACY_SUMMARY_CHUNK_CHARS = 4000
LEGACY_SUMMARY_OVERLAP_CHARS = 200


# --------------------------------------------------------------------------
# Prompt versioning
# --------------------------------------------------------------------------


class PromptVersionError(RuntimeError):
    """Raised when a prompt cannot be versioned.

    A prompt version that silently falls back to ``"unknown"`` is a prompt
    version that cannot answer the only question it exists for -- "was this
    answer produced by the same prompt as the one I am testing?" -- and
    T4.3 requires that question to be answerable for every stored run. The
    failure is therefore raised rather than defaulted.
    """


def _digest(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()[:12]


def prompt_version(
    extraction_template: str | None = None,
    appraisal_template: str | None = None,
) -> str:
    """A version string derived from the prompt text, not from a counter.

    The obvious implementation is ``PROMPT_VERSION = "v1"`` and a manual bump.
    It is wrong in a way that only shows up months later: the day somebody
    edits a constraint without bumping the constant, a regression is then
    indistinguishable from a model change, and the one artifact that could have
    told them apart says the prompt was the same. Hashing the text removes the
    possibility -- the version changes if and only if the prompt does.

    The arguments exist so that a caller (a test, or a deployment overriding a
    template) versions *the prompt it actually used*. The default reads the
    templates from :mod:`paperwhisperer.analysis.sections` at call time rather
    than caching them at import, so that a test which patches a template sees
    the version change.
    """
    extraction_text = (
        sections.FACT_EXTRACTION_TEMPLATE if extraction_template is None
        else extraction_template
    )
    appraisal_text = (
        sections.FACT_APPRAISAL_TEMPLATE if appraisal_template is None
        else appraisal_template
    )
    if not str(extraction_text or "").strip() or not str(appraisal_text or "").strip():
        raise PromptVersionError(
            "fact-layer prompt templates must be non-empty; an unversionable "
            "prompt produces analysis runs that cannot be regression-tested"
        )
    payload = f"{sections.FACT_TASK_KEY}\n{extraction_text}\n--\n{appraisal_text}"
    return f"facts-v1-{_digest(payload)}"


# --------------------------------------------------------------------------
# Evidence grounding
# --------------------------------------------------------------------------


def evidence_version(evidence: Iterable[Evidence], plan_signature: str = "") -> str:
    """A version for the evidence a fact layer was extracted from.

    Content-addressed over the evidence ids and the passages they point at, so
    two runs over the same index under the same plan produce the same version
    and a run over a re-indexed document does not. That is the question
    ``evidence_version`` is on the run record to answer: "is this answer
    comparable with that one?". A timestamp would answer "no" to everything.

    The passage text is hashed rather than the ids alone. Evidence ids are
    themselves content-addressed over ``(chunk_id, plan_signature)``, so
    hashing them is close to hashing the text -- but a caller who built
    evidence by hand, as the tests do, is not obliged to have done that, and a
    version that changes when only the id scheme changes is a version that
    reports a reindex as a different paper.
    """
    rows = sorted(
        f"{item.evidence_id}|{item.chunk_id}|{_digest(item.text)}" for item in evidence or ()
    )
    payload = json.dumps(
        {"plan": plan_signature, "evidence": rows},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return f"evs-{_digest(payload)}"


def _quote_aligned(quote: str, text: str) -> bool:
    """Whether ``quote`` occurs in ``text``, exactly or after the T1.6 rules.

    Reuses :func:`paperwhisperer.rag.citations.normalize_for_alignment` rather
    than writing a comparison, for the reason that module's docstring gives at
    length: the normalisation rules are a closed, written-down list, and a
    second implementation of "roughly the same string" in this project is a
    second opinion that will drift from the first -- and drift in the direction
    of accepting quotes the verifier would reject, which means a fact could be
    marked verified on evidence the rest of the product refuses to quote.

    ``text`` here is an evidence passage rather than an element, which is a
    wider scope than ``verify_quote`` allows. That widening is deliberate and
    bounded: the passage *is* the thing the fact cites, so "is the quote in
    the passage I cited" is the question being asked, and the answer cannot
    point at a different part of the paper.
    """
    needle = str(quote or "").strip()
    if not needle:
        return False
    haystack = str(text or "")
    if not haystack.strip():
        return False
    if needle in haystack:
        return True
    return normalize_for_alignment(needle).text in normalize_for_alignment(haystack).text


def lexical_coverage(value: str, text: str) -> float:
    """How much of ``value``'s vocabulary ``text`` contains, in ``[0, 1]``.

    The project tokenizer, not a private one: CJK runs become character
    bigrams, so a Chinese fact is checked the same way a Chinese query is.

    **The limitation is real and is the reason the quote exists.** This number
    is meaningful when the fact and the passage are in the same script. A
    Chinese paraphrase of an English passage scores near zero, and a near-zero
    score is not evidence of anything. So this is used only as a *secondary*
    check for facts whose quote could not be verified, never as the primary
    one: a cross-lingual fact without a verbatim quote is reported as
    unverifiable rather than quietly accepted on a number that does not apply
    to it.
    """
    query = scoring.LexicalQuery.prepare(value)
    if query.is_empty:
        return 0.0
    return scoring.score_lexical(
        query, scoring.normalize(text), scoring.tokenize(text)
    )


def _not_stated(field_name: str) -> Fact:
    """The recorded answer for a field the paper does not state."""
    return Fact(field_name=field_name, value="", extraction_status=STATUS_NOT_FOUND)


@dataclass(slots=True, frozen=True)
class Grounding:
    """The verdict on one claimed fact: what survived, and what was checked.

    ``verified_quote`` / ``quote_evidence_id`` are the point of the whole
    exercise. A fact's *value* is a Chinese paraphrase and must never be shown
    in quotation marks -- that is the T1.6 release gate, and rendering a
    paraphrase as a quote is precisely the failure the gate exists to stop. The
    *quote* is the model's verbatim span, and this type is the only place
    that span is known to have been found in a specific passage. So the quotes
    card renders ``verified_quote`` against ``quote_evidence_id`` and has
    nothing to fall back on: a fact whose quote did not align contributes no
    quote at all, which is why the card can be trusted to be verbatim.
    """

    fact: Fact
    dropped_evidence_ids: tuple[str, ...] = ()
    verified_quote: str = ""
    quote_evidence_id: str = ""


def ground_fact(
    fact: Fact,
    evidence_by_id: Mapping[str, Evidence],
) -> Grounding:
    """Resolve one claimed fact against real evidence, and say what is left.

    The order of the rules is the design:

    1. A value the model retracted (empty) is ``not_found`` whatever it
       claimed, and any citation attached to an empty value is discarded -- a
       citation to nothing is not support.
    2. A citation that resolves to nothing in ``evidence_by_id`` is deleted.
       Not kept, not renamed, not guessed at: the ids in this pipeline are
       content-addressed, so there is no near-miss to snap to.
    3. What survives determines the status. A value with at least one
       surviving citation and (a verified quote or a fact that shares
       vocabulary with its evidence) is ``extracted``. A value whose quote was
       claimed and does not align, or which has no citations left at all, is
       ``needs_review`` -- the claim is kept because deleting it would hide
       what the model said, and labelled because rendering it as an ordinary
       fact would be a lie about how it was obtained.

    Rule 3's lexical fallback is the cross-lingual escape hatch, and it is a
    narrow one. :func:`lexical_coverage` compares vocabulary, so a Chinese
    paraphrase of an English passage scores near zero no matter how faithful
    it is; a fact that fails it is not thereby false, only unverifiable here,
    and ``needs_review`` says exactly that rather than accepting it on a number
    that does not apply to it.
    """
    value = str(fact.value or "").strip()
    cited = tuple(str(item) for item in fact.evidence_ids or ())
    if not value:
        return Grounding(
            fact=Fact(
                field_name=fact.field_name,
                value="",
                evidence_ids=(),
                extraction_status=STATUS_NOT_FOUND,
                conflicting_values=tuple(fact.conflicting_values or ()),
            ),
            dropped_evidence_ids=cited,
        )

    resolved: list[str] = []
    dropped: list[str] = []
    for evidence_id in cited:
        if evidence_id in evidence_by_id and evidence_id not in resolved:
            resolved.append(evidence_id)
        else:
            dropped.append(evidence_id)

    quote = str(getattr(fact, "quote", "") or "").strip()
    quote_owner = ""
    if quote:
        for evidence_id in resolved:
            if _quote_aligned(quote, evidence_by_id[evidence_id].text):
                quote_owner = evidence_id
                break
        if not quote_owner:
            return Grounding(
                fact=Fact(
                    field_name=fact.field_name,
                    value=value,
                    evidence_ids=tuple(resolved),
                    extraction_status=STATUS_NEEDS_REVIEW,
                    conflicting_values=tuple(fact.conflicting_values or ()),
                ),
                dropped_evidence_ids=tuple(dropped),
            )

    if not resolved:
        return Grounding(
            fact=Fact(
                field_name=fact.field_name,
                value=value,
                evidence_ids=(),
                extraction_status=STATUS_NEEDS_REVIEW,
                conflicting_values=tuple(fact.conflicting_values or ()),
            ),
            dropped_evidence_ids=tuple(dropped),
        )

    coverage = max(
        lexical_coverage(value, evidence_by_id[item].text) for item in resolved
    )
    status = STATUS_EXTRACTED if (quote or coverage > 0.0) else STATUS_NEEDS_REVIEW
    if fact.extraction_status == STATUS_CONFLICTING and status == STATUS_EXTRACTED:
        status = STATUS_CONFLICTING
    return Grounding(
        fact=Fact(
            field_name=fact.field_name,
            value=value,
            evidence_ids=tuple(resolved),
            extraction_status=status,
            conflicting_values=tuple(fact.conflicting_values or ()),
        ),
        dropped_evidence_ids=tuple(dropped),
        verified_quote=quote,
        quote_evidence_id=quote_owner,
    )



@dataclass(slots=True)
class _Claimed:
    """A fact as the model returned it, before grounding.

    ``quote`` is not on the canonical :class:`~paperwhisperer.ingestion.schema.Fact`
    on purpose: the canonical type is the *conclusion*, and the quote is the
    evidence for it -- a field that lives on the conclusion is a field a caller
    could mistake for a verified citation. It is carried here, checked here, and
    either promotes the fact to ``extracted`` or is thrown away.
    """

    field_name: str
    value: str = ""
    evidence_ids: tuple[str, ...] = ()
    status: str = STATUS_NOT_FOUND
    quote: str = ""
    conflicting_values: tuple[str, ...] = ()


def _parse_payload(raw: str) -> output.OutputResult:
    """The model's JSON, or a named verdict explaining why there is none.

    Total rather than raising, and now *named* rather than silently empty. A
    malformed reply is a failed extraction, and the pipeline's contract for a
    failed extraction is a fact layer of eight ``not_found`` fields that renders
    as "the paper does not say" -- a degraded, honest card -- rather than an
    exception that takes the document's analysis down with it.

    The return type is :class:`~paperwhisperer.llm.output.OutputResult` instead
    of a bare mapping so the caller can tell the two failures apart, which is
    the point of the layer. The old ``json.loads(extract_json_object(raw))``
    here returned ``{}`` for *every* failure -- unreadable bytes, a JSON array,
    and a genuinely empty object all reached the caller as the same empty
    mapping -- and ``ExtractionReport`` then reported one undifferentiated
    "parse failed" for all three. Now the report can say which one happened,
    and a reply the model fenced or punctuated with full-width colons is
    recovered and flagged as recovered rather than discarded.
    """
    result = output.parse_json_object(raw, label="事实层抽取")
    if not result.ok:
        logger.warning(
            "事实层抽取返回的不是可解析的 JSON，按“论文未说明”处理：%s",
            result.user_message(),
        )
    return result


def _claimed_from(raw: Any, field_name: str) -> list[_Claimed]:
    """Read one field out of the payload, tolerating the shapes models emit.

    Three shapes are accepted for every field, because which one comes back is
    a property of the model and the sampling temperature, not of the schema: a
    single object, a list of objects, and a bare string. The last is the one
    worth arguing about -- a model that returns ``"method": "we use a graph
    network"`` has produced a value with no citation, and it is accepted as
    exactly that (a ``needs_review`` fact) rather than dropped, because the
    difference between "said badly" and "not said" is one a reader should be
    able to see.
    """
    def one(item: Any) -> _Claimed:
        if not isinstance(item, Mapping):
            text = str(item or "").strip()
            return _Claimed(field_name=field_name, value=text)
        value = str(item.get("value") or "").strip()
        status = str(item.get("status") or item.get("extraction_status") or "").strip()
        if status not in MODEL_STATUSES:
            status = ""
        conflicting = item.get("conflicting_values") or ()
        if isinstance(conflicting, str):
            conflicting = (conflicting,)
        return _Claimed(
            field_name=field_name,
            value=value,
            evidence_ids=tuple(
                str(entry) for entry in (item.get("evidence_ids") or ()) if str(entry).strip()
            ),
            # A model that omits ``status`` has still said something useful: a
            # non-empty value is a claim, an empty one is a record of silence.
            # Defaulting the empty case to ``not_found`` is what makes a field
            # the model left blank read as "not stated" rather than as an
            # extraction that silently did nothing.
            status=status or (STATUS_EXTRACTED if value else STATUS_NOT_FOUND),
            quote=str(item.get("quote") or "").strip(),
            conflicting_values=tuple(str(entry) for entry in conflicting if str(entry).strip()),
        )

    if raw is None:
        return []
    if isinstance(raw, (Mapping, str)):
        claimed = one(raw)
        return [claimed] if (claimed.value or claimed.status) else []
    if isinstance(raw, (list, tuple)):
        return [one(item) for item in raw if item is not None]
    return []


@dataclass(slots=True, frozen=True)
class VerifiedQuote:
    """One span of the paper's own words, and the passage it was found in.

    The two texts are kept side by side because they answer different
    questions and conflating them is the failure the release gate names:
    ``value`` is the claim in our words and may not carry quotation marks,
    while ``quote`` is the paper's characters and may. A quotes card that
    renders ``value`` inside a blockquote produces a paraphrase wearing
    quotation marks, which is precisely what "Direct quote 与 Paraphrase 视觉
    严格区分" forbids -- so this type exists to make the distinction something
    a renderer has to decide about rather than something it can do by accident.
    """

    field_name: str
    value: str
    evidence_id: str
    quote: str


@dataclass(slots=True)
class ExtractionReport:
    """What the extraction did, for the run record and for debugging.

    ``dropped_evidence_ids`` is the load-bearing field. A model that cites an
    id nobody printed is a fact about the prompt or the model, and without
    this counter the only visible symptom is a fact layer quietly missing
    citations -- which looks exactly like a paper that says little. The
    truncations and the parse failure are recorded for the same reason: the
    plan's release gate counts silent truncations, which requires that a
    truncation be able to be loud.
    """

    raw_response: str = ""
    parse_failed: bool = False
    dropped_evidence_ids: tuple[str, ...] = ()
    truncated_evidence: int = 0
    not_stated_fields: tuple[str, ...] = ()
    unverified_fields: tuple[str, ...] = ()
    #: The spans that aligned with a cited passage, in field order. Read by
    #: :func:`build_quote_card`, which has no other source of quoted text and
    #: therefore cannot present a paraphrase as a quotation.
    verified_quotes: tuple[VerifiedQuote, ...] = ()
    #: Messages from stages that ran after the extraction -- the quote
    #: fallback, principally -- so that :meth:`degradations` has one place to
    #: read from rather than each stage reaching into the run record itself.
    notes: tuple[str, ...] = ()
    temperature_applied: bool = True
    temperature_effective: str = ""

    @property
    def is_degraded(self) -> bool:
        return bool(
            self.parse_failed
            or self.dropped_evidence_ids
            or self.truncated_evidence
            or self.unverified_fields
        )

    def degradations(self) -> tuple[str, ...]:
        """Chinese, user-visible, one per thing that went wrong."""
        messages: list[str] = []
        if self.parse_failed:
            messages.append("事实层抽取返回的结构无法解析，八个字段全部标记为“论文未说明”。")
        if self.dropped_evidence_ids:
            messages.append(
                f"抽取引用了 {len(self.dropped_evidence_ids)} 个不存在于本次证据中的 "
                f"evidence_id，已全部丢弃：{'、'.join(self.dropped_evidence_ids[:5])}。"
            )
        if self.truncated_evidence:
            messages.append(
                f"证据片段超出单次抽取的上下文预算，{self.truncated_evidence} 条证据未进入提示。"
            )
        if self.unverified_fields:
            messages.append(
                f"以下字段未能通过证据校验：{'、'.join(self.unverified_fields)}。"
            )
        messages.extend(self.notes)
        return tuple(messages)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parse_failed": self.parse_failed,
            "dropped_evidence_ids": list(self.dropped_evidence_ids),
            "truncated_evidence": self.truncated_evidence,
            "not_stated_fields": list(self.not_stated_fields),
            "unverified_fields": list(self.unverified_fields),
            "verified_quotes": [
                {
                    "field_name": item.field_name,
                    "value": item.value,
                    "evidence_id": item.evidence_id,
                    "quote": item.quote,
                }
                for item in self.verified_quotes
            ],
            "temperature_applied": self.temperature_applied,
            "temperature_effective": self.temperature_effective,
            "degradations": list(self.degradations()),
        }


@dataclass(slots=True)
class FactExtractor:
    """One low-temperature structured call, grounded before it is returned.

    ``complete`` is injected rather than imported so that this class is
    testable without a network, and so that the temperature question has a
    visible answer: :meth:`call` inspects the callable once and passes
    ``temperature`` only if the signature accepts it.
    ``LLMClient.complete`` does not, because it reads
    ``config.LLM_TEMPERATURE`` (0.7) for every call -- so on today's client
    the extraction runs at 0.7 and the run record says so, in
    :attr:`ExtractionReport.temperature_effective`. That is a real gap in
    :mod:`paperwhisperer.llm.client`, and it is recorded rather than worked
    around: the alternative, mutating a module-level constant around the call,
    would change the temperature of every other request in the process.

    The three methods are separate on purpose. ``build_prompt`` /
    :meth:`call` / :meth:`ground` are the pipeline's stages, so the run record
    can attribute the real prompt and its real cost to the extraction, and
    :meth:`extract` is only the composition of the three for a caller that has
    no ledger to keep. Building the prompt in two places instead would be a
    second copy of a contract that decides what the model is allowed to cite.
    """

    complete: Callable[..., str]
    model: str = ""
    temperature: float = sections.FACT_EXTRACTION_TEMPERATURE
    _accepts_temperature: bool | None = field(default=None, repr=False)
    #: Whether the temperature this extractor asked for actually reached the
    #: provider. Set by :meth:`call`, read by :meth:`ground` -- the reply and
    #: the fact about the reply are produced in two different steps, and
    #: stitching the flag on at call time is what keeps the run record honest
    #: about a call it did not itself make.
    _call_applied: bool = field(default=True, repr=False)

    def build_prompt(self, evidence: Sequence[Evidence]) -> tuple[str, str]:
        """The exact prompt this extractor will send, for the cost ledger."""
        return sections.build_fact_extraction_prompt(evidence)

    def call(self, system_prompt: str, user_prompt: str) -> tuple[str, bool]:
        """One completion, plus whether the temperature we asked for was used."""
        if self._accepts_temperature is None:
            try:
                parameters = inspect.signature(self.complete).parameters
            except (TypeError, ValueError):  # pragma: no cover - builtins only
                parameters = {}
            self._accepts_temperature = "temperature" in parameters or any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
        if self._accepts_temperature:
            return (
                str(self.complete(system_prompt, user_prompt, temperature=self.temperature)),
                True,
            )
        return str(self.complete(system_prompt, user_prompt)), False

    def ground(
        self,
        raw: str,
        evidence: Sequence[Evidence],
        *,
        evidence_version_value: str = "",
    ) -> tuple[PaperFacts, ExtractionReport]:
        """Turn one raw reply into a fact layer that has been grounded.

        Total by construction: any failure anywhere produces a fact layer of
        eight ``not_found`` fields plus a loud report, never an exception. A
        paper whose extraction failed has an analysis that renders as "the
        paper does not say", which is a degraded card a user can see, rather
        than a 500 for a document that parsed perfectly well.
        """
        evidence_by_id = {
            str(item.evidence_id): item for item in evidence or () if item.evidence_id
        }
        report = ExtractionReport(raw_response=str(raw or ""))
        report.temperature_applied = self._call_applied
        report.temperature_effective = (
            f"{self.temperature:g}" if self._call_applied
            else f"{getattr(config, 'LLM_TEMPERATURE', 'unknown')}（客户端不支持每次调用设置温度）"
        )
        _, dropped_passages = sections.format_evidence_blocks(
            evidence, sections.FACT_EVIDENCE_BUDGET_CHARS
        )
        report.truncated_evidence = dropped_passages

        if not evidence_by_id:
            # No evidence means nothing to extract from, and asking a model to
            # summarize a document it has not been given is how a fact layer
            # becomes an opinion. Eight not_found facts and a loud report.
            report.not_stated_fields = tuple(sections.FACT_FIELD_ORDER)
            return self._empty(evidence_version_value), report

        parsed = _parse_payload(report.raw_response)
        payload = parsed.value if isinstance(parsed.value, Mapping) else {}
        report.parse_failed = not payload
        # The verdict's own wording goes into ``notes`` so it reaches
        # ``degradations()``. That is what separates "the model sent prose" from
        # "the model sent JSON we repaired" from "the model sent JSON we could
        # not read at all" -- three different defects that used to produce one
        # identical line here.
        report.notes = report.notes + parsed.degradations()
        return self._assemble(payload, evidence_by_id, report, evidence_version_value)

    def extract(
        self,
        evidence: Sequence[Evidence],
        *,
        evidence_version_value: str = "",
    ) -> tuple[PaperFacts, ExtractionReport]:
        """Build, send and ground in one call. The composition of the three.

        Short-circuits before the call when there is nothing to extract from.
        Not an optimisation: a model handed an empty evidence list and asked to
        describe a paper has been asked to describe one from memory, which is
        how an unindexed document produces eight confident facts with no
        citations behind them. Spending a call to get that is spending money to
        buy a wrong answer.
        """
        if not any(item.evidence_id for item in evidence or ()):
            report = ExtractionReport(
                not_stated_fields=tuple(sections.FACT_FIELD_ORDER)
            )
            report.notes = (
                "没有可用的检索证据，本次未调用模型，八个字段全部标记为“论文未说明”。",
            )
            return self._empty(evidence_version_value), report
        system_prompt, user_prompt = self.build_prompt(evidence)
        raw, self._call_applied = self.call(system_prompt, user_prompt)
        return self.ground(raw, evidence, evidence_version_value=evidence_version_value)

    def _empty(self, evidence_version_value: str) -> PaperFacts:
        return PaperFacts(
            extraction_model=self.model,
            prompt_version=prompt_version(),
            evidence_version=evidence_version_value,
        )

    def _assemble(
        self,
        payload: Mapping[str, Any],
        evidence_by_id: Mapping[str, Evidence],
        report: ExtractionReport,
        evidence_version_value: str,
    ) -> tuple[PaperFacts, ExtractionReport]:
        """Ground every field and guarantee all eight are present.

        The presence guarantee is the requirement that a field the paper does
        not mention is *recorded* rather than omitted, and it is enforced here
        rather than trusted from the prompt: for a single field the schema's
        default is already a ``not_found`` fact, and for a list field an
        omitted list is replaced by a list holding one, which is the only way
        ``PaperFacts`` can express "we looked and there was nothing" for a
        multi-valued field.
        """
        values: dict[str, Any] = {}
        dropped_ids: list[str] = []
        not_stated: list[str] = []
        unverified: list[str] = []
        verified: list[VerifiedQuote] = []

        def absorb(field_name: str) -> list[Fact]:
            spec = FACT_FIELDS[field_name]
            claimed = _claimed_from(payload.get(field_name), field_name)
            if not claimed:
                # The model left the field out entirely. Recording that as
                # "not stated" rather than as an absent list is the whole
                # requirement, and it has to happen here -- an empty list is
                # indistinguishable on the card from a field nobody looked at.
                not_stated.append(field_name)
                return [_not_stated(field_name)]

            grounded: list[Fact] = []
            for item in claimed:
                if not item.value and not item.conflicting_values:
                    grounded.append(_not_stated(field_name))
                    not_stated.append(field_name)
                    continue
                candidate = Fact(
                    field_name=field_name,
                    value=item.value,
                    evidence_ids=item.evidence_ids,
                    extraction_status=item.status or STATUS_EXTRACTED,
                    conflicting_values=item.conflicting_values,
                )
                result = ground_fact(
                    _FactWithQuote(candidate, item.quote), evidence_by_id
                )
                dropped_ids.extend(result.dropped_evidence_ids)
                if result.fact.extraction_status == STATUS_NOT_FOUND:
                    not_stated.append(field_name)
                elif result.fact.extraction_status == STATUS_NEEDS_REVIEW:
                    unverified.append(field_name)
                grounded.append(result.fact)
                # Only a quote that aligned with a specific passage reaches
                # the quotes card. A paraphrase is never a quote, and the card
                # has no other text to fall back on -- so an absent verified
                # quote is what keeps that card trustworthy rather than full.
                if result.verified_quote and result.quote_evidence_id:
                    verified.append(
                        VerifiedQuote(
                            field_name=field_name,
                            value=result.fact.value,
                            evidence_id=result.quote_evidence_id,
                            quote=result.verified_quote,
                        )
                    )
            if not grounded:
                not_stated.append(field_name)
                return [_not_stated(field_name)]
            if spec["many"]:
                return grounded
            return grounded[:1]

        for field_name in sections.FACT_FIELD_ORDER:
            values[field_name] = absorb(field_name)

        report.dropped_evidence_ids = tuple(sorted(set(dropped_ids)))
        report.not_stated_fields = tuple(sorted(set(not_stated)))
        report.unverified_fields = tuple(sorted(set(unverified)))
        report.verified_quotes = tuple(verified)

        facts = PaperFacts(
            research_question=values["research_question"][0],
            contributions=values["contributions"],
            method=values["method"][0],
            datasets=values["datasets"],
            metrics=values["metrics"],
            main_results=values["main_results"],
            limitations=values["limitations"],
            reproducibility=values["reproducibility"][0],
            extraction_model=self.model,
            prompt_version=prompt_version(),
            evidence_version=evidence_version_value,
        )
        return facts, report


class _FactWithQuote:
    """A :class:`Fact` with the model's quote attached, for grounding only.

    :func:`ground_fact` reads ``quote`` off whatever it is handed, and this is
    how it gets there without adding a field to the canonical schema. The
    dataclass is thrown away the moment grounding returns; nothing downstream
    ever sees it, which is what keeps ``quote`` a verification input rather
    than a second, unofficial citation field.
    """

    __slots__ = ("fact", "quote")

    def __init__(self, fact: Fact, quote: str):
        self.fact = fact
        self.quote = quote

    def __getattr__(self, name: str) -> Any:
        return getattr(self.fact, name)


# --------------------------------------------------------------------------
# Cards
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CardClaim:
    """One line of a card, and the fact it came from.

    ``claim`` is the *only* text a card is allowed to present as a statement
    about the paper, and it is :func:`claim_text` verbatim. ``evidence_ids``
    travels with it so a card can be rendered with citation chips without
    reaching back into the fact layer and risking a different pairing.
    """

    field_name: str
    label: str
    claim: str
    status: str
    evidence_ids: tuple[str, ...] = ()

    @property
    def is_not_stated(self) -> bool:
        return self.status == STATUS_NOT_FOUND

    @property
    def needs_review(self) -> bool:
        return self.status == STATUS_NEEDS_REVIEW

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_name": self.field_name,
            "label": self.label,
            "claim": self.claim,
            "status": self.status,
            "evidence_ids": list(self.evidence_ids),
        }


def claim_text(fact: Fact) -> str:
    """The single canonical sentence for a fact, shared by every card.

    This function is the consistency guarantee. Two cards that both want to
    state a fact call this, so they cannot state it differently -- not by
    discipline, and not by a test, but because there is one function. The
    ``not_found`` branch is the other half of the requirement: the
    "the paper does not say" marker is produced here, so it reaches every card
    by the same route a value would.
    """
    if fact.extraction_status == STATUS_NOT_FOUND or not str(fact.value or "").strip():
        return NOT_STATED_TEXT
    return str(fact.value).strip()


def _claim(fact: Fact) -> CardClaim:
    field_name = str(fact.field_name or "")
    return CardClaim(
        field_name=field_name,
        label=str(FACT_FIELDS.get(field_name, {}).get("label", field_name)),
        claim=claim_text(fact),
        status=str(fact.extraction_status or STATUS_NOT_FOUND),
        evidence_ids=tuple(str(item) for item in (fact.evidence_ids or ())),
    )


def card_claims(facts: PaperFacts, fields: Sequence[str] | None = None) -> tuple[CardClaim, ...]:
    """Every claim a card about these fields will make, in field order.

    Exposed because the acceptance criterion is *cross-card consistency* and
    the cheapest honest way to test it is to ask what each card says rather
    than to read its Markdown. A card that wanted to introduce a claim of its
    own would have to add it here, and this list is the whole of what any
    card may render.
    """
    wanted = tuple(fields) if fields else sections.FACT_FIELD_ORDER
    claims: list[CardClaim] = []
    for field_name in wanted:
        for fact in _field_facts(facts, field_name):
            claims.append(_claim(fact))
    return tuple(claims)


def _field_facts(facts: PaperFacts, field_name: str) -> list[Fact]:
    """The facts of one field, as a list whether the field holds one or many.

    The uniform return is what lets every renderer iterate all eight fields
    with one loop. A renderer that had to special-case the three single-valued
    fields would be three more places for a fact to be rendered differently
    from the others, and the whole point of this module is that there is one
    rendering.
    """
    value = getattr(facts, field_name, None)
    if isinstance(value, Fact):
        return [value]
    if isinstance(value, (list, tuple)):
        return [item for item in value if isinstance(item, Fact)]
    return []


def _render(title: str, claims: Sequence[CardClaim]) -> str:
    """Claims as Markdown, one heading per field, in the canonical order.

    The review marker goes in front of the value rather than replacing it (see
    :data:`UNVERIFIED_PREFIX`), and the two ``not_found``/``needs_review``
    branches differ only in that marker: a fact that failed verification and a
    fact the paper never stated are *both* statements, and the reader is owed
    the difference between them. Collapsing them into one "empty" branch would
    have made the product quieter and less true at the same time.
    """
    parts: list[str] = [f"## {title}", ""]
    current: str | None = None
    for claim in claims:
        if claim.label != current:
            if current is not None:
                parts.append("")
            current = claim.label
            parts.extend([f"### {claim.label}", ""])
        if claim.needs_review:
            parts.append(f"- {UNVERIFIED_PREFIX}{claim.claim}")
        else:
            parts.append(f"- {claim.claim}")
    return "\n".join(parts).strip() + "\n"


def render_overview(facts: PaperFacts) -> str:
    """概览: every field, every fact, one per line, in the canonical order."""
    return _render("概览", card_claims(facts))


def render_brief(facts: PaperFacts) -> str:
    """深度简报: the same claims, grouped, with the reproducibility check.

    Every claim here is the same :func:`claim_text` output the overview shows.
    The brief is longer because it adds *structure* over the facts -- headings,
    the grouping, a checklist of what a reproducer has to supply -- and no new
    statements about the paper, which is the only thing that could make the two
    cards disagree.
    """
    parts: list[str] = [
        "## 深度阅读简报",
        "",
        "### 事实层（与其他卡片逐字一致）",
        "",
        _claims_body(card_claims(facts)),
        "",
    ]
    checklist = _reproduction_checklist(facts)
    parts.extend(["### 复现检查清单", ""])
    parts.extend(f"- {item}" for item in checklist)
    return "\n".join(parts).strip() + "\n"


def _claims_body(claims: Sequence[CardClaim]) -> str:
    """The grouped claim lines without the ``## title`` that heads a card.

    Split out so that the brief and the overview run the *same* grouping code
    rather than two copies of it that can drift. A brief that formats the
    fields differently from the overview is a brief whose contribution lines
    differ from the overview's, which is the exact inconsistency this module
    exists to prevent, arrived at by the most innocent route available.
    """
    parts: list[str] = []
    current: str | None = None
    for claim in claims:
        if claim.label != current:
            if current is not None:
                parts.append("")
            current = claim.label
            parts.extend([f"#### {claim.label}", ""])
        prefix = UNVERIFIED_PREFIX if claim.needs_review else ""
        parts.append(f"- {prefix}{claim.claim}")
    return "\n".join(parts).strip()


def render_mindmap(facts: PaperFacts) -> str:
    """文本结构: the fact layer as a tree, with no model in the loop.

    The tree shape is the fact layer's own shape -- one branch per field, the
    field's facts as leaves -- so a reader who compares this card with the
    overview sees the same claims in a different arrangement, which is the
    point. An earlier version of this function asked a model to draw the
    paper's logic; the result was a second opinion wearing the costume of a
    structure diagram.
    """
    parts: list[str] = ["## 思维导图", ""]
    root = claim_text(facts.research_question)
    # A paper that never stated its research question still gets a root, and
    # the root is the marker rather than the empty string -- a diagram whose
    # root node is blank is one a reader cannot distinguish from a failed
    # parse.
    parts.append("论文" if root == NOT_STATED_TEXT else root)
    for field_name in sections.FACT_FIELD_ORDER:
        if field_name == "research_question":
            continue
        facts_here = _field_facts(facts, field_name)
        if not facts_here:
            continue
        label = str(FACT_FIELDS[field_name]["label"])
        last_field = field_name == "reproducibility"
        parts.append(f"{'└── ' if last_field else '├── '}{label}")
        for index, fact in enumerate(facts_here):
            glyph = "    " if last_field else "│   "
            connector = "└── " if index == len(facts_here) - 1 else "├── "
            text = claim_text(fact)
            if fact.extraction_status == STATUS_NEEDS_REVIEW:
                text = f"{UNVERIFIED_PREFIX}{text}"
            parts.append(f"{glyph}{connector}{text}")
    return "\n".join(parts).strip() + "\n"


def _reproduction_checklist(facts: PaperFacts) -> list[str]:
    """What a reproducer has to supply, assembled from the fact layer.

    Every line is either a fact's own words or a fixed connective; nothing is
    inferred. That constraint is what makes it safe to show a checklist rather
    than a paragraph: there is no sentence here that the fact layer did not
    already contain.
    """
    lines: list[str] = []
    datasets = [claim_text(item) for item in facts.datasets if claim_text(item) != NOT_STATED_TEXT]
    method = claim_text(facts.method)
    metrics = [claim_text(item) for item in facts.metrics if claim_text(item) != NOT_STATED_TEXT]
    reproducibility = claim_text(facts.reproducibility)
    if datasets:
        lines.append(f"数据：{'；'.join(datasets)}")
    if method != NOT_STATED_TEXT:
        lines.append(f"方法：{method}")
    if metrics:
        lines.append(f"指标：{'；'.join(metrics)}")
    if reproducibility != NOT_STATED_TEXT:
        lines.append(f"复现信息：{reproducibility}")
    if not lines:
        lines.append(NOT_STATED_TEXT)
    return lines


def build_fact_digest(facts: PaperFacts) -> str:
    """The fact layer as a single block of text, for the appraisal prompt.

    Same claims, same wording, one more consumer -- which is the property the
    consistency requirement wants: the evaluation card is built from this text,
    so the evaluation can only restate what every other card already said.
    """
    lines: list[str] = []
    for claim in card_claims(facts):
        suffix = f"（{claim.status}）" if claim.status != STATUS_EXTRACTED else ""
        lines.append(f"- {claim.label}：{claim.claim}{suffix}")
    return "\n".join(lines)


def build_quote_card(
    report: ExtractionReport,
    evidence_by_id: Mapping[str, Evidence],
) -> str:
    """引用片段: the spans the fact layer verified against the source.

    Rendering, not extraction, and rendering the paper's characters rather
    than the fact layer's. Every line is a ``quote`` that
    :func:`_quote_aligned` matched inside the passage it cites, so this is the
    only card in the product whose quoted text is known to be quoted text --
    the section it replaces asked a model to "extract quotable sentences",
    which is a request to *write* sentences, and not one of them was ever
    checked.

    A fact whose quote did not verify contributes nothing here, and there is
    no fallback to its value: a paraphrase printed with quotation marks around
    it is the exact thing the T1.6 release gate exists to prevent, and a card
    that is shorter and true beats one that is longer and unverified.
    """
    lines: list[str] = ["## 引用片段", ""]
    for item in report.verified_quotes:
        evidence = evidence_by_id.get(item.evidence_id)
        if evidence is None:
            continue
        label = str(FACT_FIELDS.get(item.field_name, {}).get("label", item.field_name))
        lines.append(f'> "{item.quote.strip()}"')
        lines.append(f"— {label}（{evidence.citation_label}）")
    if len(lines) == 2:
        lines.append(NOT_STATED_TEXT)
    return "\n".join(lines).strip() + "\n"


#: The appraisal payload's sections, in render order, with their headings.
#: The keys are the ones the appraisal prompt asks for; the headings are the
#: ones a reader looks under. Keeping them as one table means the prompt, the
#: parser and the card cannot disagree about the shape -- the failure being a
#: model that returns ``"critique"`` where the card reads ``"weaknesses"``, so
#: the judgement silently disappears from the UI.
APPRAISAL_SECTIONS = (
    ("strengths", "主要优点"),
    ("weaknesses", "值得质疑之处"),
    ("threats_to_validity", "威胁效度"),
    ("worth_learning", "值得学习的地方"),
    ("open_questions", "后续问题"),
)


def render_evaluation(facts: PaperFacts, appraisal: Mapping[str, Any] | None) -> str:
    """批判评价: the fact layer's own account, plus the model's judgement.

    The structure is the consistency mechanism. 贡献, 方法, 数据集, 指标,
    主要结果 and 局限性 are rendered from the facts, so this card cannot
    disagree with the overview about them even if the model tried: the model's
    text only ever appears under 审稿人视角, which is where a reader looks for
    opinion rather than for the paper's claims.
    """
    parts: list[str] = ["## 论文评价", ""]
    parts.append("### 事实层要点（与其他卡片一致）")
    parts.append("")
    for field_name in sections.FACT_FIELD_ORDER:
        spec = FACT_FIELDS[field_name]
        for fact in _field_facts(facts, field_name):
            parts.append(f"- {spec['label']}：{claim_text(fact)}")
    parts.append("")
    parts.append("### 审稿人视角（仅补充事实层未覆盖的判断）")
    parts.append("")
    listed = False
    for key, title in APPRAISAL_SECTIONS:
        raw_items = (appraisal or {}).get(key) or ()
        items = [str(item).strip() for item in raw_items if str(item).strip()]
        if not items:
            continue
        listed = True
        parts.append(f"**{title}**")
        parts.extend(f"- {item}" for item in items)
        parts.append("")
    if not listed:
        parts.append("（本次未生成审稿视角，事实层要点见上。）")
    return "\n".join(parts).strip() + "\n"


# --------------------------------------------------------------------------
# The section tree, and the Mermaid first layer
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SectionNode:
    """One heading in the document's own structure."""

    title: str
    level: int
    order: int
    children: tuple[SectionNode, ...] = ()

    def walk(self) -> list[SectionNode]:
        """This node and every descendant, in reading order."""
        found = [self]
        for child in self.children:
            found.extend(child.walk())
        return found

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "level": self.level,
            "order": self.order,
            "children": [child.to_dict() for child in self.children],
        }


#: The title used when the document has none. A fixed string rather than an
#: empty node, because a diagram with a nameless root is not renderable and a
#: reader cannot tell an unnamed paper from a failed parse.
DEFAULT_ROOT_TITLE = "论文"


def _heading_level(element: Element) -> int:
    """The heading's level, from the element or from its heading path.

    The fallback matters: adapters differ in whether they set ``level``, but
    they all maintain ``heading_path``, so a document from an adapter that
    omits levels still produces a tree rather than a flat list of roots.
    """
    if element.level:
        return max(1, int(element.level))
    return max(1, len(element.heading_path or ()) or 1)


def build_section_tree(document: ParsedDocument) -> SectionNode:
    """The document's heading structure, read rather than inferred.

    The root is the document's title element when it has one, and otherwise
    the first top-level heading, so a paper that has no separate title page
    still gets a root that is a real heading rather than an invention. Nodes
    nest by level with a stack, which is the only nesting that survives a
    document whose levels skip (a ``##`` with no ``#`` above it).
    """
    elements = [
        element
        for element in document.iter_reading_order()
        if element.role in ("title", "heading") and str(element.text or "").strip()
    ]
    title = ""
    for element in elements:
        if element.role == "title":
            title = str(element.text).strip()
            break

    heading_elements = [element for element in elements if element.role == "heading"]
    if not heading_elements:
        return SectionNode(title=title or DEFAULT_ROOT_TITLE, level=0, order=0)

    root_title = title
    root_level = 1
    if not root_title:
        root_title = str(heading_elements[0].text).strip()
        root_level = _heading_level(heading_elements[0])

    # The tree is assembled as mutable lists and frozen once, at the end.
    # SectionNode is frozen because the whole value it belongs to is an
    # immutable description of a document that is not changing, and building
    # it with copy-on-write reassignment would rebuild every subtree on every
    # heading -- quadratic in the section count for a document that has
    # hundreds of them.
    root: dict[str, Any] = {"title": root_title, "level": root_level, "order": 0,
                            "children": []}
    stack: list[tuple[int, dict[str, Any]]] = [(root_level, root)]
    order = 1
    for element in heading_elements:
        level = _heading_level(element)
        while len(stack) > 1 and stack[-1][0] >= level:
            stack.pop()
        node: dict[str, Any] = {
            "title": str(element.text).strip(),
            "level": level,
            "order": order,
            "children": [],
        }
        stack[-1][1]["children"].append(node)
        stack.append((level, node))
        order += 1
    return _freeze_node(root)


def _freeze_node(node: Mapping[str, Any]) -> SectionNode:
    return SectionNode(
        title=str(node["title"]),
        level=int(node["level"]),
        order=int(node["order"]),
        children=tuple(_freeze_node(child) for child in node["children"]),
    )


def _mermaid_label(title: str, width: int = 40) -> str:
    """A node label that cannot break the diagram.

    Brackets, quotes and newlines are the three characters that turn a heading
    into a parse error inside ``A[text]``. A paper whose section is called
    ``Results [continued]`` is entirely ordinary, and the naive renderer emits
    a diagram that fails to load with no indication why.
    """
    cleaned = " ".join(str(title or "").split())
    cleaned = cleaned.replace("[", "(").replace("]", ")").replace('"', "'")
    if len(cleaned) > width:
        cleaned = cleaned[: width - 1].rstrip() + "…"
    return cleaned or "（未命名）"


def _nodes_within(tree: SectionNode, depth: int) -> list[tuple[str, SectionNode]]:
    """Descendants of ``tree`` down to ``depth`` levels, with their node id.

    Pairs rather than nodes, because a node does not know its parent and
    :class:`SectionNode` is frozen -- an earlier version attached ``parent`` as
    an attribute, which works on a normal object and raises on a frozen slotted
    one. The depth is counted from the root, not from the first heading, so a
    document whose title element is absent and whose first heading is a
    subsection still produces a one-level diagram rather than a two-level one
    that skipped a level it does not have.
    """
    found: list[tuple[str, SectionNode]] = []

    def visit(node: SectionNode, parent_id: str, current_depth: int) -> None:
        for child in node.children:
            child_id = f"n{child.order}"
            found.append((parent_id, child))
            if current_depth < depth:
                visit(child, child_id, current_depth + 1)

    visit(tree, "root", 1)
    return found


def build_mermaid_sections(
    tree: SectionNode,
    depth: int = 1,
    title: str = "",
) -> str:
    """The Mermaid diagram, first layer generated from the section tree.

    ``depth=1`` -- the default, and the case the acceptance criterion names --
    emits the document's own top-level structure and nothing else: one node
    per heading at the level the reader sees, edges from the heading they sit
    under. No model is involved and no call is made; a diagram whose section
    level is a model's guess is a diagram that can disagree with the paper.

    Deeper levels are optional and equally structural: ``depth=2`` adds the
    document's own sub-headings. The model may add *leaves* below that (see
    :func:`paperwhisperer.analysis.sections.build_mermaid_deepening_prompt`) but
    never the sections themselves, so the part that must be right cannot be
    the part that is guessed.
    """
    depth = max(1, int(depth))
    root_label = title or tree.title
    lines = ["graph TD", f'    root["{_mermaid_label(root_label)}"]']
    for parent_id, node in _nodes_within(tree, depth):
        node_id = f"n{node.order}"
        lines.append(f'    {node_id}["{_mermaid_label(node.title)}"]')
        lines.append(f"    {parent_id} --> {node_id}")
    return "\n".join(lines) + "\n"


def render_section_tree_text(tree: SectionNode) -> str:
    """The tree as indented text, for a prompt or a log."""
    lines: list[str] = [tree.title or DEFAULT_ROOT_TITLE]
    for node in tree.walk()[1:]:
        lines.append(f"{'  ' * max(0, node.level - tree.level)}{node.title}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Cost
# --------------------------------------------------------------------------


def _config_float(name: str) -> float | None:
    """Read an optional price, the way ``llm.embeddings`` reads its own.

    ``llm/prompts.py`` and ``core/config.py`` are not owned by this task, so
    the price keys are read through ``getattr`` and default to absent. Absent
    means *unknown*, which cost arithmetic must report as ``None`` and never
    as ``0.0`` -- "we could not tell" and "this was free" are different facts
    and conflating them makes a budget lie.
    """
    raw = getattr(config, name, None)
    if raw in (None, ""):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


@dataclass(slots=True, frozen=True)
class CallBudget:
    """What one analysis path costs, measured rather than estimated-of-opinion.

    ``llm_calls`` is the count the acceptance criterion is about. ``input_tokens``
    is the number that actually tracks money, and it is measured with the same
    estimator (:func:`paperwhisperer.rag.chunking.estimate_tokens`) on both
    paths over the same document, so the comparison cannot be a comparison of
    two different guesses.
    """

    llm_calls: int
    input_tokens: int
    prompt_chars: int
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def calls_per_document(self) -> int:
        return int(self.llm_calls)

    def to_dict(self) -> dict[str, Any]:
        return {
            "llm_calls": self.llm_calls,
            "input_tokens": self.input_tokens,
            "prompt_chars": self.prompt_chars,
            "detail": dict(self.detail),
        }


def legacy_call_budget(
    content: str,
    *,
    generate_mermaid: bool = True,
    generate_evaluation: bool = True,
    generate_research_brief: bool = True,
) -> CallBudget:
    """The old path, counted with its own chunker and its own prompts.

    This is the baseline T1.8's acceptance criterion is measured against, so
    it is reconstructed rather than quoted from a comment. The summary cost is
    the number of chunks the *real* ``TextChunker`` produces for this content
    plus the merge call, and the five section costs are the *real* prompt
    builders -- the same functions the orchestrator calls. If either changes,
    the comparison changes with it, which is the property a hand-written "15
    calls" comment does not have.
    """
    chunker = TextChunker(LEGACY_SUMMARY_CHUNK_CHARS, LEGACY_SUMMARY_OVERLAP_CHARS)
    chunks = chunker.chunk_text(content or "")
    summary_calls = len(chunks) + (1 if len(chunks) > 1 else 0)

    calls = 0
    tokens = 0
    characters = 0

    def add(system_prompt: str, user_prompt: str, count: int = 1) -> None:
        nonlocal calls, tokens, characters
        calls += count
        tokens += count * estimate_tokens(system_prompt) + count * estimate_tokens(user_prompt)
        characters += count * (len(system_prompt) + len(user_prompt))

    for chunk in chunks:
        system_prompt, user_prompt = sections.build_summary_chunk_prompt(chunk)
        add(system_prompt, user_prompt)
    if len(chunks) > 1:
        system_prompt, user_prompt = sections.build_summary_merge_prompt(
            "\n\n--- 章节 ---\n\n".join("summary" for _ in chunks)
        )
        add(system_prompt, user_prompt)

    system_prompt, user_prompt = sections.build_quotes_prompt(content)
    add(system_prompt, user_prompt)
    system_prompt, user_prompt = sections.build_mindmap_prompt(content)
    add(system_prompt, user_prompt)
    if generate_mermaid:
        system_prompt, user_prompt = sections.build_mermaid_prompt(content)
        add(system_prompt, user_prompt)
    if generate_evaluation:
        system_prompt, user_prompt = sections.build_evaluation_prompt(content)
        add(system_prompt, user_prompt)
    if generate_research_brief:
        system_prompt, user_prompt = sections.build_research_brief_prompt(content)
        add(system_prompt, user_prompt)

    return CallBudget(
        llm_calls=calls,
        input_tokens=tokens,
        prompt_chars=characters,
        detail={"summary_calls": summary_calls, "chunks": len(chunks)},
    )


#: How many LLM calls the fact-layer path spends on a document, by stage. One
#: extraction, always. One appraisal, when the evaluation card is wanted. One
#: quote-selection call, only when the extraction produced no verified quote at
#: all. Overview, brief, mindmap and the Mermaid first layer cost nothing,
#: because none of them calls a model.
def fact_layer_call_budget(
    evidence: Sequence[Evidence],
    *,
    generate_evaluation: bool = True,
    quotes_fallback: bool = False,
) -> CallBudget:
    """The new path, counted the same way as :func:`legacy_call_budget`.

    The call count is incremented by the same helper that accumulates the
    tokens, deliberately. An earlier version hard-coded ``calls = 1`` for the
    extraction and then added the appraisal's prompt volume without adding its
    call, so the planner reported one call for a path that makes two -- and the
    acceptance criterion is a number about calls. Counting calls and tokens in
    one place makes it impossible for the two to describe different pipelines.
    """
    calls = 0
    tokens = 0
    characters = 0
    stages: dict[str, int] = {}

    def add(stage: str, system_prompt: str, user_prompt: str) -> None:
        nonlocal calls, tokens, characters
        calls += 1
        stages[stage] = stages.get(stage, 0) + 1
        tokens += estimate_tokens(system_prompt) + estimate_tokens(user_prompt)
        characters += len(system_prompt) + len(user_prompt)

    system_prompt, user_prompt = sections.build_fact_extraction_prompt(evidence)
    add("extraction", system_prompt, user_prompt)
    if generate_evaluation:
        digest = "x" * 512  # a fact layer's rendered length, order of magnitude
        system_prompt, user_prompt = sections.build_fact_appraisal_prompt(digest, evidence)
        add("appraisal", system_prompt, user_prompt)
    if quotes_fallback:
        system_prompt, user_prompt = sections.build_fact_quote_prompt(evidence)
        add("quote_selection", system_prompt, user_prompt)

    return CallBudget(
        llm_calls=calls,
        input_tokens=tokens,
        prompt_chars=characters,
        detail=dict(stages),
    )


@dataclass(slots=True, frozen=True)
class ModelCall:
    """One recorded model call: enough to attribute a cost to a stage.

    ``temperature_applied`` is here rather than only on the extraction report
    because the same question -- "was the thing you asked for actually used?"
    -- has to be answerable for every stage, and a run record that answers it
    for one stage and not another is a record that reads as complete.
    """

    stage: str
    model: str
    prompt_version: str
    input_tokens: int
    output_tokens: int
    cost: float | None
    temperature: float | None
    temperature_applied: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost": self.cost,
            "temperature": self.temperature,
            "temperature_applied": self.temperature_applied,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ModelCall:
        temperature = payload.get("temperature")
        cost = payload.get("cost")
        return cls(
            stage=str(payload.get("stage") or ""),
            model=str(payload.get("model") or ""),
            prompt_version=str(payload.get("prompt_version") or ""),
            input_tokens=int(payload.get("input_tokens") or 0),
            output_tokens=int(payload.get("output_tokens") or 0),
            cost=None if cost is None else float(cost),
            temperature=None if temperature is None else float(temperature),
            temperature_applied=bool(payload.get("temperature_applied")),
        )


@dataclass(slots=True)
class AnalysisRun:
    """One analysis run, complete and serializable, and where T2.2 persists it.

    This is the object the plan's ``analysis_runs`` table is a projection of.
    It exists in full, in Python, before the table exists, for the same reason
    :class:`paperwhisperer.llm.embeddings.EmbeddingUsage` exists without a table:
    a cost record that has to be invented at the moment someone adds the table
    is a cost record that is wrong, because by then the tokens are gone.

    What is on it and why each field earns its place:

    ``model`` / ``prompt_version``
        The two coordinates that make a regression diagnosable. A changed
        answer with the same pair is a retrieval or index problem; a changed
        answer with a changed pair is not, and without both you cannot tell.
    ``evidence_version``
        Whether this answer is comparable with another one at all -- see
        :func:`evidence_version`.
    ``calls`` / ``llm_cost`` / ``embedding_usage``
        The money, split into what the model was told and what it produced, so
        that a run which spent nothing because everything was cached is
        distinguishable from one that spent nothing because the index was empty.
    ``llm_calls`` / ``input_tokens``
        The acceptance criterion for T1.8, recorded per run so the comparison
        can be made on real data later rather than on this module's arithmetic.
    ``degradations``
        What went wrong, in Chinese, for a user or a support ticket. A run
        record with no degradations field is one that reports success for a
        paper that said nothing.
    """

    run_id: str
    document_id: str
    created_at: str
    model: str
    prompt_version: str
    evidence_version: str
    calls: tuple[ModelCall, ...] = ()
    embedding_usage: EmbeddingUsage | None = None
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    fact_count: int = 0
    grounded_fact_count: int = 0
    not_stated_fields: tuple[str, ...] = ()
    coverage: float = 0.0
    legacy_llm_calls: int = 0
    legacy_input_tokens: int = 0
    degradations: tuple[str, ...] = ()
    facts: PaperFacts | None = None
    retrieval_record_ids: tuple[str, ...] = ()

    @property
    def llm_cost(self) -> float | None:
        """What the run cost, or ``None`` if it cannot be known.

        ``None`` the moment any one call reported no usage or no price: a
        partial sum of a partly-unknown total is a number that looks like a
        total, and a cost dashboard that rounds an unknown to a small number is
        worse than one that says it does not know.
        """
        if not self.calls:
            return 0.0
        total = 0.0
        for call in self.calls:
            if call.cost is None:
                return None
            total += call.cost
        return total

    def add_call(
        self,
        stage: str,
        system_prompt: str,
        user_prompt: str,
        output: str,
        *,
        model: str = "",
        temperature: float | None = None,
        temperature_applied: bool = True,
        price_in: float | None = None,
        price_out: float | None = None,
    ) -> ModelCall:
        """Record one call and its cost, from the prompts that were sent.

        Token counts are *estimates* (:func:`paperwhisperer.rag.chunking.estimate_tokens`),
        because ``LLMClient.complete`` returns a string and no usage block.
        That is stated in the field names and in the docstring of
        :mod:`paperwhisperer.llm.client`'s successor rather than left for a
        reader to discover: an estimate presented as a measurement is how a
        cost ledger starts lying. When the client exposes provider usage, this
        method is the single place that has to change.
        """
        input_tokens = estimate_tokens(system_prompt) + estimate_tokens(user_prompt)
        output_tokens = estimate_tokens(output)
        cost: float | None = None
        if price_in is not None or price_out is not None:
            cost = input_tokens * float(price_in or 0.0) / 1_000_000
            cost += output_tokens * float(price_out or 0.0) / 1_000_000
        call = ModelCall(
            stage=stage,
            model=model or self.model,
            prompt_version=self.prompt_version,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            temperature=temperature,
            temperature_applied=temperature_applied,
        )
        self.calls = self.calls + (call,)
        self.llm_calls = len(self.calls)
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        return call

    def as_analysis_run_columns(self) -> dict[str, Any]:
        """The columns T2.2 must create, and the exact values for this row.

        The table does not exist. When it does, one row per analysis run gets
        these, written once by
        :meth:`paperwhisperer.analysis.paper_facts.FactLayerPipeline.run`, plus
        the embedding columns that
        :meth:`paperwhisperer.llm.embeddings.EmbeddingUsage.as_analysis_run_columns`
        already names on the same row.

        ==============================  ==========================================
        column                           source
        ==============================  ==========================================
        ``run_id``                       :attr:`run_id`
        ``document_id``                  :attr:`document_id`
        ``created_at``                   :attr:`created_at`
        ``model``                        :attr:`model`
        ``prompt_version``               :attr:`prompt_version`
        ``evidence_version``             :attr:`evidence_version`
        ``llm_calls``                    :attr:`llm_calls`
        ``llm_input_tokens``             :attr:`input_tokens`
        ``llm_output_tokens``            :attr:`output_tokens`
        ``llm_cost``                     :attr:`llm_cost`
        ``fact_count``                   :attr:`fact_count`
        ``grounded_fact_count``          :attr:`grounded_fact_count`
        ``fact_coverage``                :attr:`coverage`
        ``not_stated_fields``            :attr:`not_stated_fields` (JSON array)
        ``degradations``                 :attr:`degradations` (JSON array)
        ``retrieval_record_ids``         :attr:`retrieval_record_ids` (JSON array)
        ``legacy_llm_calls``             :attr:`legacy_llm_calls`
        ``legacy_input_tokens``          :attr:`legacy_input_tokens`
        ==============================  ==========================================

        The last two exist so that the T1.8 acceptance criterion -- "cost went
        down, here is the data" -- can be answered from stored runs rather
        than from a one-off measurement that nobody can repeat. They are the
        cost of the *other* path recorded next to this one, and they are the
        reason the acceptance criterion is a number that can be re-checked
        against production instead of a claim in a commit message.

        The ``paper_facts`` payload itself belongs on the documents table
        (``MetadataStore.put_facts`` already exists on both tiers); this record
        is the provenance of that payload, not a copy of it.
        """
        return {
            "run_id": self.run_id,
            "document_id": self.document_id,
            "created_at": self.created_at,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "evidence_version": self.evidence_version,
            "llm_calls": self.llm_calls,
            "llm_input_tokens": self.input_tokens,
            "llm_output_tokens": self.output_tokens,
            "llm_cost": self.llm_cost,
            "fact_count": self.fact_count,
            "grounded_fact_count": self.grounded_fact_count,
            "fact_coverage": self.coverage,
            "not_stated_fields": list(self.not_stated_fields),
            "degradations": list(self.degradations),
            "retrieval_record_ids": list(self.retrieval_record_ids),
            "legacy_llm_calls": self.legacy_llm_calls,
            "legacy_input_tokens": self.legacy_input_tokens,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.as_analysis_run_columns()
        payload["calls"] = [call.to_dict() for call in self.calls]
        payload["embedding_usage"] = (
            self.embedding_usage.to_dict() if self.embedding_usage is not None else None
        )
        payload["facts"] = self.facts.to_dict() if self.facts is not None else None
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AnalysisRun:
        usage_payload = payload.get("embedding_usage")
        facts_payload = payload.get("facts")
        not_stated = payload.get("not_stated_fields") or ()
        return cls(
            run_id=str(payload.get("run_id") or ""),
            document_id=str(payload.get("document_id") or ""),
            created_at=str(payload.get("created_at") or ""),
            model=str(payload.get("model") or ""),
            prompt_version=str(payload.get("prompt_version") or ""),
            evidence_version=str(payload.get("evidence_version") or ""),
            calls=tuple(
                ModelCall.from_dict(item)
                for item in payload.get("calls") or ()
                if isinstance(item, Mapping)
            ),
            embedding_usage=(
                EmbeddingUsage(**usage_payload) if isinstance(usage_payload, Mapping) else None
            ),
            llm_calls=int(payload.get("llm_calls") or 0),
            input_tokens=int(payload.get("llm_input_tokens") or 0),
            output_tokens=int(payload.get("llm_output_tokens") or 0),
            fact_count=int(payload.get("fact_count") or 0),
            grounded_fact_count=int(payload.get("grounded_fact_count") or 0),
            not_stated_fields=tuple(str(item) for item in not_stated),
            coverage=float(payload.get("fact_coverage") or 0.0),
            legacy_llm_calls=int(payload.get("legacy_llm_calls") or 0),
            legacy_input_tokens=int(payload.get("legacy_input_tokens") or 0),
            degradations=tuple(
                str(item) for item in payload.get("degradations") or ()
            ),
            facts=(
                PaperFacts.from_dict(facts_payload)
                if isinstance(facts_payload, Mapping)
                else None
            ),
            retrieval_record_ids=tuple(
                str(item) for item in payload.get("retrieval_record_ids") or ()
            ),
        )


# --------------------------------------------------------------------------
# The DAG
# --------------------------------------------------------------------------


@dataclass(slots=True)
class FactLayerOutcome:
    """Everything one fact-layer run produced.

    ``cards`` is keyed by the existing section names
    (:data:`paperwhisperer.analysis.sections.SECTION_KEYS`) so the result shape
    the API, the session payload and the templates already read does not
    change. ``run`` is the cost and provenance record; ``report`` is what went
    wrong and is the reason a degraded run is visible instead of silent.
    """

    facts: PaperFacts
    cards: dict[str, str]
    run: AnalysisRun
    report: ExtractionReport
    evidence: tuple[Evidence, ...] = ()
    appraisal: dict[str, Any] = field(default_factory=dict)
    section_tree: SectionNode | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_facts": self.facts.to_dict(),
            "analysis_run": self.run.to_dict(),
            "extraction": self.report.to_dict(),
            "cards": dict(self.cards),
            "section_tree": (
                self.section_tree.to_dict() if self.section_tree is not None else None
            ),
        }


def build_analysis_run(
    *,
    document_id: str,
    facts: PaperFacts,
    calls: Sequence[ModelCall] = (),
    embedding_usage: EmbeddingUsage | None = None,
    created_at: str = "",
    report: ExtractionReport | None = None,
    legacy_budget: CallBudget | None = None,
    retrieval_record_ids: Sequence[str] = (),
    model: str = "",
) -> AnalysisRun:
    """Assemble the run record for a finished fact layer.

    A function rather than a constructor call because the record is worth
    building even when no model was called at all: a document that produced
    eight ``not_found`` facts is a real analysis run, it cost nothing, and it
    is the run most worth remembering -- "this paper says nothing about its
    reproducibility" is a finding, and it is indistinguishable from a bug
    unless somebody wrote it down.
    """
    all_facts = facts.all_facts()
    run = AnalysisRun(
        run_id=build_run_id(
            document_id=document_id,
            model=model or facts.extraction_model,
            facts=facts,
        ),
        document_id=document_id,
        created_at=created_at,
        model=model or facts.extraction_model,
        prompt_version=facts.prompt_version or prompt_version(),
        evidence_version=facts.evidence_version,
        calls=tuple(calls),
        embedding_usage=embedding_usage,
        llm_calls=len(calls),
        input_tokens=sum(call.input_tokens for call in calls),
        output_tokens=sum(call.output_tokens for call in calls),
        fact_count=len(all_facts),
        grounded_fact_count=sum(1 for fact in all_facts if fact.is_grounded),
        coverage=facts.coverage,
        not_stated_fields=report.not_stated_fields if report else (),
        facts=facts,
        retrieval_record_ids=tuple(str(item) for item in retrieval_record_ids),
    )
    if report is not None:
        run.degradations = tuple(report.degradations())
    if legacy_budget is not None:
        run.legacy_llm_calls = legacy_budget.llm_calls
        run.legacy_input_tokens = legacy_budget.input_tokens
    return run


def build_run_id(*, document_id: str, model: str, facts: PaperFacts) -> str:
    """A content-addressed run id.

    Over the document, the model, the prompt version, the evidence version and
    the facts themselves -- the inputs a reader would compare to decide whether
    two runs are the same analysis. It deliberately excludes any timestamp or
    duration, for the reason :func:`paperwhisperer.rag.retrieve._record_id` gives
    for retrieval records: an id that changed every run could not answer "did
    yesterday's analysis of this paper say the same thing", which is exactly
    the question T4.3's regression gate asks.
    """
    payload = json.dumps(
        {
            "document": document_id,
            "model": model,
            "prompt": facts.prompt_version,
            "evidence": facts.evidence_version,
            "facts": facts.to_dict(),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return "run_" + _digest(payload)


class FactLayerPipeline:
    """One extraction, five renderings, and the run record for all of it.

    The shape is a DAG with exactly one node that costs a model call and one
    that costs a second: ``extraction``, and ``appraisal`` when the evaluation
    card is wanted. Everything else is a pure function of their output. That is
    not primarily an optimisation -- it is what makes the cards agree with
    each other, because there is only one place a fact can be read from.

    ``evidence_provider`` is anything callable as
    ``provider(field_name, document_id) -> (evidence, record_ids)``. The
    production implementation is a thin wrapper over
    :class:`paperwhisperer.rag.retrieve.HybridRetriever`, which is why the
    per-field task types in :data:`FACT_FIELDS` are plan names: this module
    does not have a retrieval strategy of its own, it asks for the one plan
    section 6.4 already defines.

    A required provider rather than an optional one is deliberate. The
    obvious convenience -- "with no provider, fall back to the document text"
    -- is exactly the failure the retriever refuses to commit: a fact layer
    extracted from text the model was handed wholesale is a fact layer with no
    citations, and it would render as though every one of its claims had been
    verified. So a caller with no index gets eight ``not_found`` facts and a
    loud report instead.
    """

    def __init__(
        self,
        *,
        complete: Callable[..., str],
        evidence_provider: Callable[[str, str], tuple[Sequence[Evidence], Sequence[str]]],
        model: str = "",
        document_id: str = "",
        price_in: float | None = None,
        price_out: float | None = None,
        embedding_usage: EmbeddingUsage | None = None,
    ) -> None:
        self.extractor = FactExtractor(complete=complete, model=model)
        self.evidence_provider = evidence_provider
        self.model = model
        self.document_id = document_id
        self.price_in = (
            price_in if price_in is not None else _config_float("LLM_PRICE_PER_MTOK_INPUT")
        )
        self.price_out = (
            price_out if price_out is not None else _config_float("LLM_PRICE_PER_MTOK_OUTPUT")
        )
        self.embedding_usage = embedding_usage

    # -- stages ------------------------------------------------------------

    def gather_evidence(self) -> tuple[list[Evidence], list[str]]:
        """One retrieval pass per field, merged, de-duplicated, in field order.

        De-duplication by evidence id is what keeps the prompt from carrying
        the same passage three times when three fields all boost the
        introduction. Field order is preserved so that a field whose evidence
        falls outside the context budget loses its *last* passages rather than
        its first -- the difference between "the limitations are missing" and
        "the limitations are the one thing this paper never said".
        """
        collected: dict[str, Evidence] = {}
        record_ids: list[str] = []
        for field_name in sections.FACT_FIELD_ORDER:
            try:
                evidence, records = self.evidence_provider(field_name, self.document_id)
            except Exception as exc:  # noqa: BLE001 - a leg must not take the document down
                logger.warning("事实层检索失败（%s）：%s", field_name, exc)
                continue
            for record_id in records or ():
                if str(record_id) not in record_ids:
                    record_ids.append(str(record_id))
            for item in evidence or ():
                if item.evidence_id and item.evidence_id not in collected:
                    collected[item.evidence_id] = item
        return list(collected.values()), record_ids

    def extract(
        self,
        evidence: Sequence[Evidence],
        *,
        evidence_version_value: str = "",
    ) -> tuple[PaperFacts, ExtractionReport, ModelCall]:
        """The single call, plus the record of what it cost.

        Short-circuits on an empty evidence pool for the reason
        :meth:`FactExtractor.extract` gives, and records the fact as a
        zero-token call so that the run's ``llm_calls`` still describes the
        stages the DAG walked rather than the ones that happened to spend
        money.
        """
        if not any(item.evidence_id for item in evidence or ()):
            facts, report = self.extractor.extract(
                evidence, evidence_version_value=evidence_version_value
            )
            return facts, report, ModelCall(
                stage="extraction",
                model=self.model,
                prompt_version=prompt_version(),
                input_tokens=0,
                output_tokens=0,
                cost=0.0,
                temperature=self.extractor.temperature,
                temperature_applied=True,
            )
        system_prompt, user_prompt = self.extractor.build_prompt(evidence)
        raw, applied = self.extractor.call(system_prompt, user_prompt)
        facts, report = self.extractor.ground(
            raw, evidence, evidence_version_value=evidence_version_value
        )
        input_tokens = estimate_tokens(system_prompt) + estimate_tokens(user_prompt)
        output_tokens = estimate_tokens(raw)
        call = ModelCall(
            stage="extraction",
            model=self.model,
            prompt_version=prompt_version(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=self._cost(input_tokens, output_tokens),
            temperature=self.extractor.temperature,
            temperature_applied=applied,
        )
        return facts, report, call

    def _cost(self, input_tokens: int, output_tokens: int) -> float | None:
        if self.price_in is None and self.price_out is None:
            return None
        return (
            input_tokens * float(self.price_in or 0.0)
            + output_tokens * float(self.price_out or 0.0)
        ) / 1_000_000

    def appraise(
        self,
        facts: PaperFacts,
        evidence: Sequence[Evidence],
    ) -> tuple[dict[str, Any], ModelCall | None]:
        """Evaluation = facts + a targeted pass. One call, two narrow inputs.

        The digest is rendered from the facts rather than from the document,
        and that is the mechanism rather than an optimisation: the model is
        asked to add judgement and forbidden to restate a claim, so it has no
        opportunity to form a second opinion about what the paper contributes.
        The evidence is the critical-appraisal slice -- limitations, hedged
        results, setup -- rather than the whole pool, because the whole pool
        was read once already and a second full read is the cost this task
        exists to remove.
        """
        if not evidence and not facts.all_facts():
            return {}, None
        digest = build_fact_digest(facts)
        system_prompt, user_prompt = sections.build_fact_appraisal_prompt(
            digest, self.appraisal_evidence(evidence)
        )
        raw, applied = self.extractor.call(system_prompt, user_prompt)
        input_tokens = estimate_tokens(system_prompt) + estimate_tokens(user_prompt)
        output_tokens = estimate_tokens(raw)
        call = ModelCall(
            stage="appraisal",
            model=self.model,
            prompt_version=prompt_version(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=self._cost(input_tokens, output_tokens),
            temperature=None,
            temperature_applied=applied,
        )
        # The appraisal is optional by design -- ``render_evaluation`` prints a
        # marker when there is no 审稿视角 -- so an unreadable reply degrades to
        # an empty appraisal rather than raising. The verdict is still named in
        # the log, because "the appraisal came back empty" and "the appraisal
        # could not be read" are different bugs.
        parsed = _parse_payload(raw)
        if not parsed.ok:
            logger.warning(
                "审稿视角返回的不是可解析的 JSON，审稿视角留空：%s", parsed.user_message()
            )
        appraisal = parsed.value if isinstance(parsed.value, Mapping) else {}
        return {str(key): value for key, value in appraisal.items()}, call

    @staticmethod
    def appraisal_evidence(evidence: Sequence[Evidence]) -> list[Evidence]:
        """The evidence a critical reading needs, and only that.

        Bounded twice: by what the appraisal is for (the material a structured
        extraction handles worst -- stated limitations, hedged claims, setup)
        and by count. An unbounded slice would quietly become a second full
        read of the document wearing a narrower name, which is the shape of
        regression this task is supposed to prevent.
        """
        needles = (
            "limitation", "we do not", "future work", "threat", "assum",
            "局限", "不足", "未来工作", "假设", "失败",
        )
        chosen = [
            item
            for item in evidence
            if any(needle in str(item.text).casefold() for needle in needles)
        ]
        return chosen[: EVIDENCE_PER_FIELD * 2] or list(evidence)[:EVIDENCE_PER_FIELD]

    def select_quotes(
        self,
        evidence: Sequence[Evidence],
    ) -> tuple[str, ModelCall]:
        """The fallback quote selection. Reached only when nothing verified.

        The quotes card is normally free: it renders the spans the extraction
        already checked. This call exists so that a *required* section does not
        simply go empty when the extraction returned no verifiable span -- an
        empty quotes card is indistinguishable from a bug, and this section is
        in ``REQUIRED_SECTION_KEYS``.

        What it is not allowed to do is invent: the prompt demands verbatim
        spans from the printed evidence and nothing else. Even here the
        alignment requirement is relaxed to "the model was told not to", which
        is why the fallback result is recorded as a distinct stage -- a reader
        of the run record can see at a glance which quotes were verified and
        which were merely requested.
        """
        system_prompt, user_prompt = sections.build_fact_quote_prompt(evidence)
        raw, applied = self.extractor.call(system_prompt, user_prompt)
        input_tokens = estimate_tokens(system_prompt) + estimate_tokens(user_prompt)
        output_tokens = estimate_tokens(raw)
        call = ModelCall(
            stage="quote_selection",
            model=self.model,
            prompt_version=prompt_version(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=self._cost(input_tokens, output_tokens),
            temperature=None,
            temperature_applied=applied,
        )
        return str(raw or ""), call

    # -- run ---------------------------------------------------------------

    def run(
        self,
        document: ParsedDocument | None = None,
        *,
        evidence: Sequence[Evidence] | None = None,
        generate_evaluation: bool = True,
        generate_mermaid: bool = True,
        generate_research_brief: bool = True,
        quote_fallback: bool = True,
        legacy_content: str = "",
        created_at: str = "",
    ) -> FactLayerOutcome:
        """Run the whole thing and return the cards, the facts and the record.

        ``evidence`` short-circuits retrieval for a caller that already has a
        pool; otherwise :meth:`gather_evidence` runs one pass per field.
        ``document`` is used only for the section tree, which is why the
        Mermaid first layer costs nothing: there is no model in that path and
        therefore nothing for a document argument to buy.

        ``generate_mermaid``, ``generate_evaluation`` and
        ``generate_research_brief`` are the caller's three section switches and
        all three are honoured here. They cost nothing to honour -- two of them
        are pure renderings of data the extraction already paid for -- so a
        flag that was accepted and then ignored was a switch that lied to the
        caller: the card appeared, the run was billed for it, and a reader who
        had explicitly switched it off could not tell it from one they had
        asked for. ``evaluation`` is the only one of the three that additionally
        gates a model call, and that is why it is the one that was already
        correct.

        ``quote_fallback`` is the third and last call this path can make. It
        fires only when the extraction verified no span at all, so the common
        case is two calls (extraction and appraisal) against the old path's
        fifteen to seventeen.
        """
        if evidence is None:
            collected, record_ids = self.gather_evidence()
        else:
            collected, record_ids = list(evidence), []
        version = evidence_version(collected)

        facts, report, extraction_call = self.extract(collected, evidence_version_value=version)

        calls = [extraction_call]
        appraisal: dict[str, Any] = {}
        if generate_evaluation:
            payload, appraisal_call = self.appraise(facts, collected)
            appraisal = payload
            if appraisal_call is not None:
                calls.append(appraisal_call)

        tree = build_section_tree(document) if document is not None else None
        evidence_by_id = {item.evidence_id: item for item in collected}
        quotes = build_quote_card(report, evidence_by_id)
        if quote_fallback and not report.verified_quotes and collected:
            fallback_text, fallback_call = self.select_quotes(collected)
            calls.append(fallback_call)
            quotes = fallback_text.strip() or quotes
            # The run record has to say *which* quotes were checked. A card
            # full of unverified spans next to a card of verified ones is
            # exactly the visual distinction T1.9 asks for, and that
            # distinction is only possible if the run says which is which.
            report.notes = report.notes + (
                "事实层没有取得可逐字回源的原文片段，引用片段已降级为"
                "未经验证的模型摘录（本次额外调用了一次模型）。",
            )

        cards = {
            "summary": render_overview(facts),
            "quotes": quotes,
            "mindmap": render_mindmap(facts),
            "mermaid": (
                build_mermaid_sections(tree) if tree is not None and generate_mermaid else ""
            ),
            "evaluation": render_evaluation(facts, appraisal) if generate_evaluation else "",
            "research_brief": render_brief(facts) if generate_research_brief else "",
        }

        run = build_analysis_run(
            document_id=self.document_id,
            facts=facts,
            calls=calls,
            embedding_usage=self.embedding_usage,
            created_at=created_at,
            report=report,
            legacy_budget=legacy_call_budget(legacy_content) if legacy_content else None,
            retrieval_record_ids=record_ids,
            model=self.model,
        )
        return FactLayerOutcome(
            facts=facts,
            cards=cards,
            run=run,
            report=report,
            evidence=tuple(collected),
            appraisal=appraisal,
            section_tree=tree,
        )


def make_evidence_provider(retriever: Any, document_id: str, top_k: int = EVIDENCE_PER_FIELD):
    """Bind a :class:`~paperwhisperer.rag.retrieve.HybridRetriever` into a provider.

    The retriever is passed in rather than constructed here so that this module
    depends on retrieval's *plan names* and not on its stores, its bundles or
    its configuration -- the same reasoning that keeps
    :mod:`paperwhisperer.rag.retrieve` from importing an embedding SDK. A test
    can therefore drive the whole DAG with a callable and no database, and
    production gets the real thing by passing it in.

    The plan's own ``final_k`` is used when the caller does not override it,
    because overriding it downward would silently change what "six passages
    per field" means and the evidence budget is what keeps the single
    extraction call inside its context window.
    """

    def provider(field_name: str, owner_id: str) -> tuple[tuple[Evidence, ...], tuple[str, ...]]:
        spec = FACT_FIELDS[field_name]
        plan = plan_for(str(spec["task"]))
        result = retriever.retrieve(
            str(spec["query"]),
            plan=plan.with_overrides(
                final_k=top_k,
                candidate_depth=max(plan.candidate_depth, top_k),
            ),
            document_ids=[owner_id or document_id],
        )
        return result.evidence, (result.record.record_id,)

    return provider


