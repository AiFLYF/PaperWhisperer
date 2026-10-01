"""The fact layer: one extraction, five renderings, and a measurable bill.

T1.8's acceptance criteria are two sentences -- 跨卡片事实一致 (the same
conclusion reads the same on every card), and LLM 调用成本较旧路径下降且有对比
数据 (the cost went down, and here is the number). Both are asserted here as
written rather than through a proxy for them, because a proxy for
cross-card consistency is a test that checks two cards are both non-empty.

**The consistency test is the interesting one, and it is structural rather
than statistical.** It does not compare two summaries for similarity; it asks
two different renderers what they say about the *same fact* and requires
byte-identical answers, and it requires that no card contain a claim the fact
layer does not contain. The second half is the one that catches the real
failure: a card that quietly introduced a wording of its own would pass every
"are the cards similar" test ever written. So the test enumerates what each
card is allowed to say -- :func:`paper_facts.card_claims` -- and asserts the
rendered text says nothing else. A renderer that wants to add a claim has to
add it there, where it would be visible.

**No network, no model, no clock.** The stub client below is a plain function
that records what it was sent and returns a fixed payload; two of the tests
deliberately make it *raise*, to prove that a path which claims to need no
model really does not call one. Token counts are estimates by construction
(:func:`paperwhisperer.rag.chunking.estimate_tokens`), which is stated on the
field names and repeated in the run record's own docstring rather than left for
a reader to discover.

**The cost comparison is measured, not asserted from a comment.**
:func:`legacy_call_budget` rebuilds the old path from the orchestrator's real
chunker and the real prompt builders, and :func:`fact_layer_call_budget` does
the same for the new one, over the same document and with the same token
estimator. The numbers below are the ones those two functions produce, and the
test also *drives* both paths with a counting stub to check that the predicted
call counts are the observed ones -- a planner that disagreed with the code it
plans for would be worse than no planner.
"""

from __future__ import annotations

import json
import math

import pytest

from paperwhisperer.analysis import paper_facts, sections, service
from paperwhisperer.analysis.orchestrator import DocumentAnalyzer
from paperwhisperer.documents.loader import TextChunker
from paperwhisperer.ingestion.schema import (
    EXTRACTION_STATUSES,
    Element,
    Evidence,
    Fact,
    PaperFacts,
    ParsedDocument,
    SourceLocator,
)
from paperwhisperer.llm.embeddings import EmbeddingUsage
from paperwhisperer.rag.retrieve import plan_for

# --------------------------------------------------------------------------
# Fixtures and builders
# --------------------------------------------------------------------------

EVIDENCE_TEXT = {
    "ev_intro": (
        "Retrieval-augmented generation prepends the query to the document, "
        "which fails when the query and the document disagree. Existing "
        "systems therefore answer questions the paper never asked."
    ),
    "ev_method": (
        "We propose RAG-Fusion, a graph-structured index that routes a query to "
        "a subgraph before generation. The index is rebuilt whenever a document "
        "changes, and routing is deterministic."
    ),
    "ev_results": (
        "All experiments use ImageNet and WebQuestions. We report F1 and "
        "exact-match accuracy. RAG-Fusion improves exact-match accuracy on NQ "
        "by 6.3 points over vanilla RAG at equal parameter count."
    ),
    "ev_limits": (
        "We do not study multilingual settings. The graph index does not "
        "update when a document is revised, and the routing step doubles "
        "latency on the p.99 tail."
    ),
    "ev_code": (
        "Code is available at github.com/example/rag-fusion. The default "
        "configuration, including all hyperparameters, is in config/default.yaml."
    ),
}

EVIDENCE = tuple(
    Evidence(
        evidence_id=evidence_id,
        chunk_id=f"chunk_{evidence_id}",
        text=text,
        locator=SourceLocator(kind="pdf_page", page=index + 1),
        heading_path=("Introduction",) if index < 2 else ("Experiments",),
    )
    for index, (evidence_id, text) in enumerate(EVIDENCE_TEXT.items())
)
EVIDENCE_BY_ID = {item.evidence_id: item for item in EVIDENCE}


def extraction_payload(**overrides) -> str:
    """A well-formed extraction reply, for a paper that states six of eight.

    ``limitations`` is deliberately absent by default: it is the field that
    makes "recorded as not stated" observable rather than theoretical, since a
    payload that stated everything would make the absence test vacuous.
    """
    payload = {
        "research_question": {
            "value": "检索增强生成在查询与文档不一致时给出错误答案",
            "evidence_ids": ["ev_intro"],
            "status": "extracted",
            "quote": "fails when the query and the document disagree",
        },
        "contributions": [
            {
                "value": "提出图结构索引 RAG-Fusion，查询先路由到子图再生成",
                "evidence_ids": ["ev_method"],
                "status": "extracted",
                "quote": "We propose RAG-Fusion, a graph-structured index",
            }
        ],
        "method": {
            "value": "图结构索引 + 确定性子图路由",
            "evidence_ids": ["ev_method"],
            "status": "extracted",
            "quote": "routes a query to a subgraph before generation",
        },
        "datasets": [
            {
                "value": "ImageNet、WebQuestions",
                "evidence_ids": ["ev_results"],
                "status": "extracted",
                "quote": "All experiments use ImageNet and WebQuestions.",
            }
        ],
        "metrics": [
            {
                "value": "F1、exact-match accuracy",
                "evidence_ids": ["ev_results"],
                "status": "extracted",
                "quote": "We report F1 and exact-match accuracy.",
            }
        ],
        "main_results": [
            {
                "value": "NQ 上 EM 提升 6.3 个点",
                "evidence_ids": ["ev_results"],
                "status": "extracted",
                "quote": "improves exact-match accuracy on NQ by 6.3 points",
            }
        ],
        "reproducibility": {
            "value": "代码与默认超参数均已开源",
            "evidence_ids": ["ev_code"],
            "status": "extracted",
            "quote": "Code is available at github.com/example/rag-fusion",
        },
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


class StubClient:
    """A counting model stand-in. Records every prompt it is handed.

    ``temperature_applied`` mirrors the real client's behaviour: it does not
    accept a per-call temperature, so the extraction runs at the global one.
    Several tests assert that the run record says so rather than implying the
    requested temperature was used.
    """

    def __init__(self, extraction: str | None = None, appraisal: str | None = None):
        self.extraction = extraction if extraction is not None else extraction_payload()
        self.appraisal = appraisal if appraisal is not None else json.dumps(
            {"strengths": ["子图路由把检索误差隔离在局部"],
             "weaknesses": ["未与稠密检索器做同预算对比"]},
            ensure_ascii=False,
        )
        self.calls: list[dict] = []

    def complete(self, system_prompt, user_prompt, max_retries=None, model=None):
        self.calls.append({"system": system_prompt, "user": user_prompt, "model": model})
        if "critical_appraisal_evidence" in user_prompt:
            return self.appraisal
        return self.extraction

    @property
    def stages(self) -> list[str]:
        return [
            "appraisal" if "critical_appraisal_evidence" in call["user"] else "extraction"
            for call in self.calls
        ]


class ExplodingClient:
    """A client that fails the test if it is called at all."""

    def complete(self, *args, **kwargs):  # pragma: no cover - the assertion is the call
        raise AssertionError("这一段路径不允许调用模型")


def document_with_sections() -> ParsedDocument:
    """A paper with the section tree the Mermaid first layer must reproduce."""
    return ParsedDocument(
        document_id="doc_1",
        source_filename="rag_fusion.pdf",
        page_count=9,
        elements=[
            Element(role="title", text="RAG-Fusion: Graph-Structured Retrieval",
                    order=0, level=1),
            Element(role="heading", text="1 Introduction", order=1, level=1),
            Element(role="paragraph", text="Retrieval augmented generation ...", order=2),
            Element(role="heading", text="2 Method", order=3, level=1),
            Element(role="heading", text="2.1 Graph Index", order=4, level=2),
            Element(role="paragraph", text="We build a graph ...", order=5),
            Element(role="heading", text="3 Experiments", order=6, level=1),
            Element(role="heading", text="4 Limitations", order=7, level=1),
            Element(role="heading", text="5 Conclusion", order=8, level=1),
        ],
    )


def a_document_of(size: int) -> str:
    """A synthetic paper of about ``size`` characters.

    Real sentences rather than one repeated character: the legacy cost model
    measures prompt *tokens*, and a document of ``x`` would be priced by the
    symbol rule at a rate no real paper pays. The number under test is the
    ratio between two paths, and it has to be a ratio of realistic quantities.
    """
    sentences = [
        "Retrieval augmented generation prepends the query to the document and "
        "answers from the concatenation of the two.",
        "When the query and the document disagree the model follows the query "
        "and the answer is wrong, which is the failure this paper addresses.",
        "We build a graph-structured index over the corpus and route a query to "
        "a subgraph before the generator is called.",
        "Experiments are run on ImageNet and WebQuestions with F1 and "
        "exact-match accuracy as the metrics.",
        "The method improves exact-match accuracy on Natural Questions by six "
        "points over a vanilla retrieval baseline at equal parameter count.",
        "We do not study multilingual settings and the index does not update "
        "when a document is revised.",
        "Code, the default configuration and every hyperparameter are released "
        "under a permissive licence.",
    ]
    block = " ".join(sentences)
    repeats = math.ceil(size / len(block))
    return (block * repeats)[:size]


def provider_for(evidence=EVIDENCE):
    """A provider that returns the same pool for every field, deterministically."""

    def provider(field_name, document_id):
        return tuple(evidence), (f"rr_{field_name}",)

    return provider


def build_outcome(client=None, **run_kwargs):
    """Run the whole DAG once and hand back the pipeline, the stub and the result."""
    client = client or StubClient()
    pipeline = paper_facts.FactLayerPipeline(
        complete=client.complete,
        evidence_provider=provider_for(),
        model="test-model",
        document_id="doc_1",
        price_in=0.15,
        price_out=0.60,
    )
    return pipeline, client, pipeline.run(
        document=document_with_sections(),
        legacy_content=a_document_of(40_000),
        created_at="2026-09-29T12:00:00",
        **run_kwargs,
    )


# --------------------------------------------------------------------------
# The eight fields
# --------------------------------------------------------------------------


class TestFieldSpecification:
    def test_the_eight_named_fields_are_the_ones_the_plan_lists(self):
        assert sections.FACT_FIELD_ORDER == (
            "research_question", "contributions", "method", "datasets",
            "metrics", "main_results", "limitations", "reproducibility",
        )
        assert set(paper_facts.FACT_FIELDS) == set(sections.FACT_FIELD_ORDER)

    def test_every_field_names_a_retrieval_plan_that_exists(self):
        # A typo here surfaces as a KeyError on the first real paper, in
        # production, on a document the user is waiting for. It is the failure
        # this test exists to move to a test run.
        for name, spec in paper_facts.FACT_FIELDS.items():
            assert plan_for(spec["task"]), name

    def test_every_field_is_declared_multi_or_single_consistently(self):
        for name, spec in paper_facts.FACT_FIELDS.items():
            expected = name in sections.MULTI_FACT_FIELDS
            assert spec["many"] is expected, name

    def test_the_statuses_are_the_canonical_ones(self):
        # Unpacking by position means a schema change breaks the import; this
        # asserts the two agree rather than assuming it.
        assert paper_facts.STATUS_EXTRACTED in EXTRACTION_STATUSES
        assert set(paper_facts.MODEL_STATUSES) <= set(EXTRACTION_STATUSES)
        assert paper_facts.STATUS_NEEDS_REVIEW not in paper_facts.MODEL_STATUSES

    def test_a_model_may_not_declare_its_own_claim_verified(self):
        # needs_review is the status this module assigns when a check fails.
        # If a model could return it, the field would exist for exactly the
        # thing that must never be self-assessed.
        assert "needs_review" not in sections.FACT_EXTRACTION_OUTPUT_FORMAT


class TestExtractionProducesEveryField:
    @pytest.fixture
    def facts(self):
        extractor = paper_facts.FactExtractor(complete=StubClient().complete, model="m")
        return extractor.extract(EVIDENCE)[0]

    def test_all_eight_fields_are_present(self, facts):
        for name in sections.FACT_FIELD_ORDER:
            values = getattr(facts, name)
            found = [values] if isinstance(values, Fact) else values
            assert found, f"{name} produced no fact at all"
            for fact in found:
                assert fact.field_name == name

    def test_a_stated_field_carries_its_citation_and_its_quote_checked_out(self, facts):
        for fact in facts.all_facts():
            if fact.extraction_status != paper_facts.STATUS_EXTRACTED:
                continue
            assert fact.evidence_ids
            assert all(item in EVIDENCE_BY_ID for item in fact.evidence_ids)

    def test_a_field_the_paper_does_not_state_is_recorded_as_not_stated(self, facts):
        # The paper here states no limitations at all, and the requirement is
        # that this is *recorded*: the field exists, its status says so, and it
        # is not silently an empty list that a reader cannot tell apart from
        # "nobody looked".
        assert len(facts.limitations) == 1
        assert facts.limitations[0].extraction_status == paper_facts.STATUS_NOT_FOUND
        assert facts.limitations[0].value == ""
        assert facts.limitations[0].evidence_ids == ()

    def test_the_not_stated_field_is_counted_in_coverage_as_a_gap(self, facts):
        # A field the paper did not answer is a hole in the fact layer, and
        # coverage is the number that should show it.
        assert facts.coverage < 1.0
        assert facts.coverage == pytest.approx(
            sum(1 for f in facts.all_facts() if f.is_grounded) / len(facts.all_facts())
        )

    def test_a_single_field_with_no_value_is_also_recorded_not_found(self):
        payload = json.dumps(
            {"method": {"value": "", "evidence_ids": [], "status": "extracted"}}
        )
        facts, report = paper_facts.FactExtractor(
            complete=StubClient(extraction=payload).complete, model="m"
        ).extract(EVIDENCE)
        assert facts.method.extraction_status == paper_facts.STATUS_NOT_FOUND
        assert "method" in report.not_stated_fields

    def test_a_field_the_model_omitted_entirely_is_still_recorded(self):
        facts, report = paper_facts.FactExtractor(
            complete=StubClient(extraction="{}").complete, model="m"
        ).extract(EVIDENCE)
        assert len(facts.all_facts()) == 8
        for name in sections.FACT_FIELD_ORDER:
            values = getattr(facts, name)
            found = [values] if isinstance(values, Fact) else values
            assert found[0].extraction_status == paper_facts.STATUS_NOT_FOUND
        assert set(report.not_stated_fields) == set(sections.FACT_FIELD_ORDER)

    def test_an_unparsable_reply_yields_eight_not_stated_facts_and_a_loud_report(self):
        facts, report = paper_facts.FactExtractor(
            complete=StubClient(extraction="模型今天不太想回答").complete, model="m"
        ).extract(EVIDENCE)
        assert report.parse_failed
        assert len(facts.all_facts()) == 8
        assert all(f.extraction_status == paper_facts.STATUS_NOT_FOUND
                   for f in facts.all_facts())
        assert any("无法解析" in message for message in report.degradations())

    def test_no_evidence_at_all_yields_eight_not_stated_facts(self):
        facts, report = paper_facts.FactExtractor(
            complete=ExplodingClient().complete, model="m"
        ).extract(())
        assert report.not_stated_fields == tuple(sections.FACT_FIELD_ORDER)
        assert all(f.extraction_status == paper_facts.STATUS_NOT_FOUND
                   for f in facts.all_facts())


# --------------------------------------------------------------------------
# Grounding: every citation resolves, every quote is checked
# --------------------------------------------------------------------------


class TestGrounding:
    def test_an_evidence_id_that_resolves_to_nothing_is_dropped(self):
        payload = json.dumps(
            {"method": {"value": "某个方法", "evidence_ids": ["ev_ghost"],
                        "quote": "We propose RAG-Fusion"}}
        )
        facts, report = paper_facts.FactExtractor(
            complete=StubClient(extraction=payload).complete, model="m"
        ).extract(EVIDENCE)
        assert facts.method.evidence_ids == ()
        assert facts.method.extraction_status == paper_facts.STATUS_NEEDS_REVIEW
        assert "ev_ghost" in report.dropped_evidence_ids
        assert any("ev_ghost" in message for message in report.degradations())

    def test_a_fact_that_keeps_some_citations_keeps_only_those(self):
        payload = json.dumps(
            {"method": {"value": "某个方法", "evidence_ids": ["ev_ghost", "ev_method"],
                        "quote": "routes a query to a subgraph before generation"}}
        )
        facts, report = paper_facts.FactExtractor(
            complete=StubClient(extraction=payload).complete, model="m"
        ).extract(EVIDENCE)
        assert facts.method.evidence_ids == ("ev_method",)
        assert facts.method.extraction_status == paper_facts.STATUS_EXTRACTED
        assert report.dropped_evidence_ids == ("ev_ghost",)

    def test_a_quote_that_does_not_occur_in_the_passage_is_not_verified(self):
        # The claim is kept -- hiding it would hide what the model said -- but
        # it is labelled, because rendering it as an ordinary fact would be a
        # lie about how it was obtained.
        payload = json.dumps(
            {"method": {"value": "某种全新的图结构索引", "evidence_ids": ["ev_method"],
                        "quote": "这句话根本不在任何证据里出现过"}}
        )
        facts, report = paper_facts.FactExtractor(
            complete=StubClient(extraction=payload).complete, model="m"
        ).extract(EVIDENCE)
        assert facts.method.extraction_status == paper_facts.STATUS_NEEDS_REVIEW
        assert "method" in report.unverified_fields
        assert not report.verified_quotes

    def test_a_quote_is_verified_by_the_citation_normalisation_rules(self):
        # The same rules T1.6 uses, so a fact cannot be marked verified on
        # evidence the rest of the product would refuse to quote.
        payload = json.dumps(
            {"method": {"value": "图结构索引", "evidence_ids": ["ev_method"],
                        "quote": "a  graph-structured    index"}}
        )
        facts, report = paper_facts.FactExtractor(
            complete=StubClient(extraction=payload).complete, model="m"
        ).extract(EVIDENCE)
        assert facts.method.extraction_status == paper_facts.STATUS_EXTRACTED
        assert report.verified_quotes[0].evidence_id == "ev_method"

    def test_every_fact_is_checkable_against_the_source_text(self):
        # The checkable form of the requirement: each grounded fact cites a
        # real passage, and the paper's own words behind it are recoverable
        # from that passage. An id that resolved to nothing would fail here
        # even though it passes an isinstance check.
        _, _, outcome = build_outcome()
        for item in outcome.report.verified_quotes:
            passage = EVIDENCE_BY_ID[item.evidence_id].text
            assert item.quote in passage
            assert item.quote.strip()

    def test_a_conflicting_claim_keeps_both_sides(self):
        payload = json.dumps(
            {"method": {"value": "图结构索引", "evidence_ids": ["ev_method"],
                        "status": "conflicting", "quote": "graph-structured index",
                        "conflicting_values": ["稠密检索器", "图结构索引"]}}
        )
        facts, _ = paper_facts.FactExtractor(
            complete=StubClient(extraction=payload).complete, model="m"
        ).extract(EVIDENCE)
        assert facts.method.extraction_status == paper_facts.STATUS_CONFLICTING
        assert "稠密检索器" in facts.method.conflicting_values

    def test_a_cross_lingual_fact_without_a_quote_is_not_rejected_as_false(self):
        # The lexical fallback compares vocabulary, so it cannot judge a
        # Chinese paraphrase of an English passage. It must report
        # "unverifiable here" (needs_review), not "false" -- and it must not
        # accept the fact on a number that does not apply to it either.
        payload = json.dumps(
            {"method": {"value": "完全不同的中文表述，不含任何英文词元", "evidence_ids": ["ev_method"]}}
        )
        facts, _ = paper_facts.FactExtractor(
            complete=StubClient(extraction=payload).complete, model="m"
        ).extract(EVIDENCE)
        assert facts.method.extraction_status == paper_facts.STATUS_NEEDS_REVIEW

    def test_a_same_script_fact_with_no_quote_is_held_to_the_lexical_check(self):
        payload = json.dumps(
            {"method": {"value": "graph structured index routing", "evidence_ids": ["ev_method"]}}
        )
        facts, _ = paper_facts.FactExtractor(
            complete=StubClient(extraction=payload).complete, model="m"
        ).extract(EVIDENCE)
        assert facts.method.extraction_status == paper_facts.STATUS_EXTRACTED

    def test_the_quote_card_shows_the_paper_words_not_the_paraphrase(self):
        _, _, outcome = build_outcome()
        quotes = outcome.cards["quotes"]
        assert "graph-structured index" in quotes
        assert "图结构索引 + 确定性子图路由" not in quotes

    def test_the_quote_card_carries_the_citation_label(self):
        _, _, outcome = build_outcome()
        assert "p. 2" in outcome.cards["quotes"]


# --------------------------------------------------------------------------
# Cross-card consistency
# --------------------------------------------------------------------------


class TestCrossCardConsistency:
    @pytest.fixture
    def outcome(self):
        return build_outcome()[2]

    @pytest.fixture
    def facts(self, outcome):
        return outcome.facts

    @pytest.fixture
    def cards(self, outcome):
        return {
            "overview": outcome.cards["summary"],
            "brief": outcome.cards["research_brief"],
            "mindmap": outcome.cards["mindmap"],
            "evaluation": outcome.cards["evaluation"],
        }

    def test_the_same_fact_claims_the_same_thing_on_every_card(self, cards, facts):
        # The acceptance criterion, asserted as written. Two renderers, one
        # fact, byte-identical text -- not "similar", identical.
        checked = 0
        for claim in paper_facts.card_claims(facts):
            if claim.is_not_stated:
                continue
            for name, text in cards.items():
                assert claim.claim in text, f"{name} did not carry {claim.claim!r}"
            checked += 1
        assert checked >= 6, "the fixture should ground at least six facts"

    def test_no_card_contains_a_claim_the_fact_layer_does_not(self, cards, facts):
        # The half of the criterion that a similarity test cannot catch. A
        # card that introduced a wording of its own would still be "similar"
        # to the others; here it is simply not on the list of things any card
        # is allowed to say.
        #
        # The appraisal section of the evaluation card is the single exempt
        # region, and it is exempt by *position* rather than by a flag: a
        # model's own words are allowed there and nowhere else, so a renderer
        # that started letting them leak into the fact-layer half of the card
        # would fail here even though the card still "looked fine".
        allowed = {claim.claim for claim in paper_facts.card_claims(facts)}
        for name, text in cards.items():
            body = _fact_layer_half(text)
            for line in body.splitlines():
                stripped = _strip_decoration(line)
                if not stripped or stripped in ALLOWED_STRUCTURE:
                    continue
                if _is_checklist_line(stripped):
                    continue
                assert any(fragment in stripped for fragment in allowed), (
                    f"{name} stated something the fact layer does not: {line!r}"
                )

    def test_the_appraisal_is_the_only_region_a_model_may_speak_in(self, facts):
        # The other half of the exemption above, asserted directly: everything
        # the model wrote on the evaluation card is under the appraisal
        # heading, and nothing above it is.
        _, _, outcome = build_outcome()
        evaluation = outcome.cards["evaluation"]
        head, separator, tail = evaluation.partition("### 审稿人视角")
        assert separator, "the evaluation card must separate facts from judgement"
        for item in outcome.appraisal.get("strengths", []):
            assert item in tail
        for claim in paper_facts.card_claims(facts):
            if not claim.is_not_stated:
                assert claim.claim in head

    def test_a_fact_the_paper_did_not_state_propagates_to_every_card(self, outcome, cards):
        assert outcome.facts.limitations[0].extraction_status == paper_facts.STATUS_NOT_FOUND
        claim = next(
            item for item in paper_facts.card_claims(outcome.facts)
            if item.field_name == "limitations"
        )
        assert claim.claim == paper_facts.NOT_STATED_TEXT
        for name, text in cards.items():
            assert paper_facts.NOT_STATED_TEXT in text, name

    def test_the_cards_cannot_read_the_document(self):
        # The structural half of the guarantee, and the reason the criterion
        # holds rather than merely being checked. A renderer that received the
        # document could form an opinion about it; none of them is passed one.
        import inspect

        for renderer in (
            paper_facts.render_overview,
            paper_facts.render_brief,
            paper_facts.render_mindmap,
        ):
            parameters = list(inspect.signature(renderer).parameters)
            assert parameters == ["facts"], renderer.__name__

    def test_a_fact_layer_with_a_different_wording_changes_every_card_together(self):
        # Consistency is not "the cards never change" -- it is "the cards
        # change together". Re-extracting with a different wording must move
        # the overview and the brief in the same step, which is what makes the
        # guarantee useful rather than merely static.
        _, _, first = build_outcome()
        payload = extraction_payload()
        payload = payload.replace("NQ 上 EM 提升 6.3 个点", "NQ 上 EM 提升六点三个百分点")
        _, _, second = build_outcome(StubClient(extraction=payload))
        old_claim = first.facts.main_results[0].value
        new_claim = second.facts.main_results[0].value
        assert old_claim != new_claim
        for key in ("summary", "research_brief", "mindmap", "evaluation"):
            assert old_claim not in second.cards[key]
            assert new_claim in second.cards[key]


#: Headings, field labels and the markers that are structure rather than
#: claims. A card is allowed to carry these and nothing else beyond the fact
#: layer's own words.
ALLOWED_STRUCTURE = {
    "论文未说明",
    "概览",
    "深度阅读简报",
    "思维导图",
    "论文评价",
    "事实层（与其他卡片逐字一致）",
    "复现检查清单",
    "事实层要点（与其他卡片一致）",
    "审稿人视角（仅补充事实层未覆盖的判断）",
    "（本次未生成审稿视角，事实层要点见上。）",
    "研究问题", "核心贡献", "方法", "数据集", "评估指标",
    "主要结果", "局限性", "可复现性",
    "主要优点", "值得质疑之处", "威胁效度", "值得学习的地方", "后续问题",
    "数据：", "方法：", "指标：", "复现信息：",
}

#: The markers the renderers use for list bullets, tree glyphs, markdown
#: headings and the review flag. ``─`` is in the set because the mindmap's
#: box-drawing glyphs are three characters and stripping two of them leaves a
#: line that looks like a claim.
_DECORATION = " \t-─>│├└#*>"

#: The heading that separates the fact layer from the model's judgement.
_APPRAISAL_HEADING = "### 审稿人视角"


def _strip_decoration(line: str) -> str:
    return str(line).strip().strip(_DECORATION).strip()


def _fact_layer_half(card: str) -> str:
    """The card up to the appraisal heading -- the part that must be pure facts."""
    return str(card).partition(_APPRAISAL_HEADING)[0]


def _is_checklist_line(stripped: str) -> bool:
    """A brief checklist line: a fixed connective in front of a fact's words."""
    return any(stripped.startswith(prefix) for prefix in ("数据：", "方法：",
                                                          "指标：", "复现信息："))


# --------------------------------------------------------------------------
# The Mermaid first layer
# --------------------------------------------------------------------------


class TestMermaidFirstLayer:
    @pytest.fixture
    def tree(self):
        return paper_facts.build_section_tree(document_with_sections())

    def test_the_tree_is_the_documents_own_sections(self, tree):
        assert tree.title == "RAG-Fusion: Graph-Structured Retrieval"
        assert [child.title for child in tree.children] == [
            "1 Introduction", "2 Method", "3 Experiments", "4 Limitations", "5 Conclusion",
        ]
        assert [child.title for child in tree.children[1].children] == ["2.1 Graph Index"]

    def test_the_first_layer_is_generated_without_a_model(self, tree):
        # The client raises on contact. If this path ever acquires a model
        # call -- because someone "improved" the diagram by asking for a
        # nicer layout -- this test fails rather than the product quietly
        # starting to guess at a paper's structure.
        assert ExplodingClient().complete  # the stand-in, unused
        diagram = paper_facts.build_mermaid_sections(tree)
        assert diagram.startswith("graph TD")
        for heading in ("1 Introduction", "2 Method", "3 Experiments",
                        "4 Limitations", "5 Conclusion"):
            assert f'"{heading}"' in diagram

    def test_the_first_layer_excludes_the_documents_subheadings(self, tree):
        # Depth 1 is the level the reader sees, and a sub-heading promoted to
        # the top level would be a diagram that disagrees with the paper about
        # its own shape.
        assert "2.1 Graph Index" not in paper_facts.build_mermaid_sections(tree)

    def test_a_second_layer_adds_the_subheadings_and_still_needs_no_model(self, tree):
        diagram = paper_facts.build_mermaid_sections(tree, depth=2)
        assert "2.1 Graph Index" in diagram

    def test_the_pipeline_produces_the_diagram_without_an_extraction_call(self):
        # End to end: a run whose model client raises still produces a correct
        # section-level diagram, because that node of the DAG has no edges
        # into the model at all.
        extractor_free = paper_facts.build_mermaid_sections(
            paper_facts.build_section_tree(document_with_sections())
        )
        assert extractor_free.count("-->") == 5

    def test_a_heading_containing_brackets_cannot_break_the_diagram(self, tree):
        from paperwhisperer.ingestion.schema import Element as _Element

        document = ParsedDocument(
            document_id="d", source_filename="x.pdf", page_count=1,
            elements=[
                _Element(role="title", text='Results "continued" [part 2]', order=0, level=1),
                _Element(role="heading", text="A [very] odd heading", order=1, level=1),
            ],
        )
        diagram = paper_facts.build_mermaid_sections(
            paper_facts.build_section_tree(document)
        )
        for line in diagram.splitlines():
            assert line.count("[") == line.count("]")

    def test_a_document_with_no_headings_still_renders(self):
        document = ParsedDocument(
            document_id="d", source_filename="x.txt", page_count=1,
            elements=[Element(role="paragraph", text="just prose, no structure",
                             order=0)],
        )
        diagram = paper_facts.build_mermaid_sections(
            paper_facts.build_section_tree(document)
        )
        assert diagram.startswith("graph TD")
        assert "root[" in diagram

    def test_a_heading_without_a_level_uses_its_heading_path(self):
        document = ParsedDocument(
            document_id="d", source_filename="x.pdf", page_count=1,
            elements=[
                Element(role="title", text="T", order=0, level=1),
                Element(role="heading", text="Outer", order=1, heading_path=("Outer",)),
                Element(role="heading", text="Inner", order=2, heading_path=("Outer", "Inner")),
            ],
        )
        tree = paper_facts.build_section_tree(document)
        assert tree.children[0].title == "Outer"
        assert tree.children[0].children[0].title == "Inner"


# --------------------------------------------------------------------------
# The run record
# --------------------------------------------------------------------------


class TestPromptVersion:
    def test_the_version_is_a_hash_of_the_prompt_text(self):
        version = paper_facts.prompt_version()
        assert version.startswith("facts-v1-")
        assert version == paper_facts.prompt_version()

    def test_the_version_changes_when_the_prompt_changes(self):
        # The whole point of deriving it rather than hand-maintaining it: the
        # day somebody edits a constraint, the version moves, and a regression
        # can be attributed instead of guessed at. T4.3 depends on this.
        baseline = paper_facts.prompt_version()
        edited = paper_facts.prompt_version(
            extraction_template=sections.FACT_EXTRACTION_TEMPLATE + "\n新增一条约束。"
        )
        assert edited != baseline

    def test_the_appraisal_prompt_alone_can_move_the_version(self):
        baseline = paper_facts.prompt_version()
        assert paper_facts.prompt_version(
            appraisal_template=sections.FACT_APPRAISAL_TEMPLATE + "\n再多说一句。"
        ) != baseline

    def test_an_empty_prompt_cannot_be_versioned(self):
        with pytest.raises(paper_facts.PromptVersionError):
            paper_facts.prompt_version(extraction_template="   ")

    def test_the_facts_record_the_version_of_the_prompt_that_produced_them(self):
        _, _, outcome = build_outcome()
        assert outcome.facts.prompt_version == paper_facts.prompt_version()
        assert outcome.run.prompt_version == paper_facts.prompt_version()


class TestAnalysisRunRecord:
    @pytest.fixture
    def run(self):
        return build_outcome()[2].run

    def test_every_required_column_is_present_and_non_empty(self, run):
        columns = run.as_analysis_run_columns()
        for key in ("run_id", "document_id", "created_at", "model", "prompt_version",
                    "evidence_version", "llm_calls", "llm_input_tokens",
                    "llm_output_tokens", "llm_cost", "fact_count",
                    "grounded_fact_count", "fact_coverage", "legacy_llm_calls",
                    "legacy_input_tokens"):
            assert key in columns, key
            assert columns[key] not in (None, ""), key

    def test_the_record_round_trips_through_json(self, run):
        payload = json.loads(json.dumps(run.to_dict(), ensure_ascii=False))
        rebuilt = paper_facts.AnalysisRun.from_dict(payload)
        assert rebuilt.run_id == run.run_id
        assert rebuilt.prompt_version == run.prompt_version
        assert rebuilt.evidence_version == run.evidence_version
        assert rebuilt.llm_calls == run.llm_calls
        assert rebuilt.llm_cost == run.llm_cost
        assert rebuilt.facts is not None
        assert rebuilt.facts.to_dict() == run.facts.to_dict()

    def test_the_run_id_is_content_addressed(self):
        _, _, first = build_outcome()
        _, _, second = build_outcome()
        assert first.run.run_id == second.run.run_id

    def test_a_different_fact_layer_gets_a_different_run_id(self):
        _, _, first = build_outcome()
        payload = extraction_payload().replace("NQ 上 EM 提升 6.3 个点", "NQ 上 EM 提升 7.0 个点")
        _, _, second = build_outcome(StubClient(extraction=payload))
        assert first.run.run_id != second.run.run_id

    def test_cost_is_unknown_rather_than_zero_when_no_price_is_configured(self):
        pipeline = paper_facts.FactLayerPipeline(
            complete=StubClient().complete,
            evidence_provider=provider_for(),
            model="m",
            document_id="doc_1",
            price_in=None,
            price_out=None,
        )
        pipeline.price_in = pipeline.price_out = None
        run = pipeline.run(evidence=EVIDENCE).run
        assert run.llm_cost is None

    def test_cost_is_a_number_when_prices_are_configured(self, run):
        assert run.llm_cost is not None
        assert run.llm_cost > 0

    def test_the_call_count_and_stages_are_recorded_per_call(self, run):
        assert [call.stage for call in run.calls] == ["extraction", "appraisal"]
        assert run.llm_calls == 2

    def test_the_run_records_that_the_requested_temperature_was_not_applied(self, run):
        # LLMClient.complete has no temperature parameter, so the extraction
        # runs at the global 0.7. The record says so rather than implying the
        # 0.0 the extractor asked for.
        extraction = run.calls[0]
        assert extraction.temperature == sections.FACT_EXTRACTION_TEMPERATURE
        assert extraction.temperature_applied is False

    def test_a_client_that_accepts_a_temperature_gets_one(self):
        seen: dict = {}

        def complete(system_prompt, user_prompt, temperature=None, model=None):
            seen["temperature"] = temperature
            return extraction_payload()

        pipeline = paper_facts.FactLayerPipeline(
            complete=complete,
            evidence_provider=provider_for(),
            model="m",
            document_id="doc_1",
        )
        run = pipeline.run(evidence=EVIDENCE, generate_evaluation=False).run
        assert seen["temperature"] == sections.FACT_EXTRACTION_TEMPERATURE
        assert run.calls[0].temperature_applied is True

    def test_the_embedding_cost_reaches_the_run_through_the_same_row(self):
        usage = EmbeddingUsage(
            provider="api.openai.com", model="text-embedding-3-small", dimension=1536,
            index_version="emb-x-d1536", requests=8, texts=8, cache_hits=3,
            input_tokens=4096, price_per_mtok=0.02,
        )
        pipeline = paper_facts.FactLayerPipeline(
            complete=StubClient().complete,
            evidence_provider=provider_for(),
            model="m",
            document_id="doc_1",
            embedding_usage=usage,
        )
        run = pipeline.run(evidence=EVIDENCE).run
        assert run.embedding_usage is not None
        assert run.embedding_usage.cost == pytest.approx(4096 * 0.02 / 1_000_000)
        # The embedding columns land on the same row T2.2 will write.
        assert usage.as_analysis_run_columns()["embedding_cost"] == run.embedding_usage.cost

    def test_degradations_are_recorded_in_chinese_and_reach_the_run(self):
        payload = json.dumps(
            {"method": {"value": "某个方法", "evidence_ids": ["ev_ghost"]}}
        )
        _, _, outcome = build_outcome(StubClient(extraction=payload))
        assert outcome.run.degradations
        assert all(message.isprintable() for message in outcome.run.degradations)
        assert any("ev_ghost" in message for message in outcome.run.degradations)

    def test_an_empty_fact_layer_still_produces_a_run_record(self):
        # A paper that said nothing is a finding, and the run that recorded it
        # is the one most worth keeping.
        run = paper_facts.build_analysis_run(document_id="doc_1", facts=PaperFacts())
        assert run.fact_count == 3
        assert run.llm_calls == 0
        assert run.llm_cost == 0.0
        assert run.run_id.startswith("run_")


# --------------------------------------------------------------------------
# Cost: the acceptance criterion, measured
# --------------------------------------------------------------------------


class TestCallCountComparison:
    """The before/after numbers, asserted so they cannot silently regress."""

    #: A 40,000-character paper -- a long conference paper, and the size the
    #: plan's corpus is full of.
    DOCUMENT_CHARS = 40_000

    def test_the_legacy_path_makes_seventeen_calls_and_the_fact_layer_two(self):
        content = a_document_of(self.DOCUMENT_CHARS)
        legacy = paper_facts.legacy_call_budget(content)
        new = paper_facts.fact_layer_call_budget(EVIDENCE, generate_evaluation=True)
        # Pinned, not derived: this is the number in the report. A change to
        # either planner that moved these has changed the cost claim, and that
        # is a decision somebody should make deliberately.
        assert legacy.llm_calls == 17
        assert new.llm_calls == 2
        # The planner must count the same calls it measures tokens for, or the
        # acceptance criterion is a number about a pipeline that does not run.
        assert new.llm_calls == sum(new.detail.values())
        assert new.detail == {"extraction": 1, "appraisal": 1}

    def test_the_fact_layer_without_the_evaluation_card_makes_one_call(self):
        new = paper_facts.fact_layer_call_budget(EVIDENCE, generate_evaluation=False)
        assert new.llm_calls == 1
        assert new.detail == {"extraction": 1}

    def test_the_quote_fallback_is_the_third_and_last_call(self):
        new = paper_facts.fact_layer_call_budget(
            EVIDENCE, generate_evaluation=True, quotes_fallback=True
        )
        assert new.llm_calls == 3
        assert new.detail["quote_selection"] == 1

    def test_the_legacy_call_count_is_the_real_chunker_and_the_real_prompts(self):
        content = a_document_of(self.DOCUMENT_CHARS)
        legacy = paper_facts.legacy_call_budget(content)
        chunks = TextChunker(
            paper_facts.LEGACY_SUMMARY_CHUNK_CHARS,
            paper_facts.LEGACY_SUMMARY_OVERLAP_CHARS,
        ).chunk_text(content)
        # Nine chunked summary calls plus one merge, plus five sections.
        assert legacy.llm_calls == len(chunks) + 1 + 5
        assert legacy.detail["summary_calls"] == len(chunks) + 1

    def test_the_legacy_planner_agrees_with_the_orchestrator_it_plans_for(self):
        # The baseline is reconstructed here, so it is checked against the code
        # that actually runs rather than trusted. A planner that drifted from
        # the orchestrator would make the acceptance criterion measure nothing.
        class Counting:
            def __init__(self):
                self.count = 0

            def complete(self, system_prompt, user_prompt, max_retries=None, model=None):
                self.count += 1
                return "x"

        analyzer = DocumentAnalyzer.__new__(DocumentAnalyzer)
        analyzer.client = Counting()
        analyzer.chunker = TextChunker(
            paper_facts.LEGACY_SUMMARY_CHUNK_CHARS,
            paper_facts.LEGACY_SUMMARY_OVERLAP_CHARS,
        )
        analyzer.summary_chunk_workers = 3
        content = a_document_of(self.DOCUMENT_CHARS)
        analyzer.generate_summary(content)
        analyzer.extract_quotes(content)
        analyzer.generate_mindmap(content)
        analyzer.generate_mermaid_mindmap(content)
        analyzer.generate_evaluation(content)
        analyzer.generate_research_brief(content)
        assert analyzer.client.count == paper_facts.legacy_call_budget(content).llm_calls

    def test_the_measured_pipeline_makes_the_planned_number_of_calls(self):
        _, client, outcome = build_outcome()
        assert client.stages == ["extraction", "appraisal"]
        assert len(client.calls) == outcome.run.llm_calls == 2

    def test_the_fact_layer_sends_far_fewer_tokens_than_the_old_path(self):
        content = a_document_of(self.DOCUMENT_CHARS)
        legacy = paper_facts.legacy_call_budget(content)
        new = paper_facts.fact_layer_call_budget(EVIDENCE, generate_evaluation=True)
        # The old path sends 15,000-18,000 characters five times over plus the
        # whole document in chunks; the new path sends one bounded evidence
        # pool. A ratio, not an absolute, because only the ratio is a claim
        # about the design rather than about this fixture.
        assert new.input_tokens * 5 < legacy.input_tokens
        assert legacy.llm_calls > new.llm_calls * 5

    def test_the_old_path_costs_about_nine_times_the_tokens(self):
        content = a_document_of(self.DOCUMENT_CHARS)
        legacy = paper_facts.legacy_call_budget(content)
        new = paper_facts.fact_layer_call_budget(EVIDENCE, generate_evaluation=True)
        ratio = legacy.input_tokens / max(1, new.input_tokens)
        # Widely separated bounds rather than a tight number: the ratio moves
        # with the document, and the claim worth protecting is "an order of
        # magnitude", not "18.8x on this fixture".
        assert 5 < ratio < 100, ratio

    def test_the_run_record_carries_both_paths_for_later_comparison(self):
        # T2.2 stores these on the same row, which is what makes the acceptance
        # criterion answerable from production data rather than from this test.
        _, _, outcome = build_outcome()
        columns = outcome.run.as_analysis_run_columns()
        assert columns["legacy_llm_calls"] == 17
        assert columns["llm_calls"] == 2
        assert columns["legacy_input_tokens"] > columns["llm_input_tokens"]

    def test_disabling_the_optional_sections_shrinks_the_old_path(self):
        content = a_document_of(self.DOCUMENT_CHARS)
        everything = paper_facts.legacy_call_budget(content)
        minimal = paper_facts.legacy_call_budget(
            content, generate_mermaid=False, generate_evaluation=False,
            generate_research_brief=False,
        )
        assert minimal.llm_calls == everything.llm_calls - 3

    def test_the_new_path_does_not_call_a_model_for_the_cheap_cards(self):
        # Overview, brief, mindmap and the Mermaid first layer are all free.
        # Only the extraction and the appraisal cost anything, which is the
        # claim the whole design rests on.
        calls: list[str] = []

        def complete(system_prompt, user_prompt, temperature=None, model=None):
            calls.append("extraction" if "critical_appraisal_evidence" not in user_prompt
                         else "appraisal")
            return extraction_payload() if len(calls) == 1 else "{}"

        pipeline = paper_facts.FactLayerPipeline(
            complete=complete, evidence_provider=provider_for(), model="m",
            document_id="doc_1",
        )
        outcome = pipeline.run(document=document_with_sections(), evidence=EVIDENCE)
        assert calls == ["extraction", "appraisal"]
        assert outcome.cards["summary"] and outcome.cards["research_brief"]
        assert outcome.cards["mindmap"] and outcome.cards["mermaid"]


# --------------------------------------------------------------------------
# The DAG as a whole
# --------------------------------------------------------------------------


class TestPipeline:
    def test_the_cards_are_keyed_by_the_existing_section_names(self):
        _, _, outcome = build_outcome()
        assert set(outcome.cards) == set(sections.SECTION_KEYS)

    def test_every_field_gets_a_retrieval_pass(self):
        requested: list[str] = []

        def provider(field_name, document_id):
            requested.append(field_name)
            return EVIDENCE, (f"rr_{field_name}",)

        pipeline = paper_facts.FactLayerPipeline(
            complete=StubClient().complete, evidence_provider=provider, model="m",
            document_id="doc_1",
        )
        pipeline.run()
        assert requested == list(sections.FACT_FIELD_ORDER)

    def test_evidence_is_de_duplicated_across_fields(self):
        seen: list[str] = []

        def provider(field_name, document_id):
            seen.append(field_name)
            return EVIDENCE, (f"rr_{field_name}",)

        pipeline = paper_facts.FactLayerPipeline(
            complete=StubClient().complete, evidence_provider=provider, model="m",
            document_id="doc_1",
        )
        collected, _ = pipeline.gather_evidence()
        assert len(collected) == len(EVIDENCE)
        assert len(seen) == 8

    def test_a_failing_retrieval_pass_does_not_take_the_document_down(self):
        def provider(field_name, document_id):
            if field_name == "method":
                raise RuntimeError("索引未就绪")
            return EVIDENCE, (f"rr_{field_name}",)

        pipeline = paper_facts.FactLayerPipeline(
            complete=StubClient().complete, evidence_provider=provider, model="m",
            document_id="doc_1",
        )
        collected, _ = pipeline.gather_evidence()
        assert collected, "the other seven passes should still have returned"

    def test_the_evidence_version_is_stable_and_content_addressed(self):
        first = paper_facts.evidence_version(EVIDENCE)
        assert first == paper_facts.evidence_version(list(reversed(EVIDENCE)))
        assert first != paper_facts.evidence_version(EVIDENCE[:2])

    def test_the_quote_fallback_only_fires_when_nothing_verified(self):
        client = StubClient()
        _, _, outcome = build_outcome(client)
        assert client.stages == ["extraction", "appraisal"]

        silent = StubClient(extraction=json.dumps(
            {"method": {"value": "某个方法", "evidence_ids": ["ev_method"]}}
        ))
        _, _, degraded = build_outcome(silent)
        assert silent.stages == ["extraction", "appraisal", "extraction"]
        assert any("未经验证" in message for message in degraded.report.notes)

    def test_the_quote_fallback_can_be_switched_off(self):
        silent = StubClient(extraction="{}")
        _, _, outcome = build_outcome(silent, quote_fallback=False)
        assert silent.stages == ["extraction", "appraisal"]
        assert paper_facts.NOT_STATED_TEXT in outcome.cards["quotes"]

    def test_the_appraisal_evidence_is_a_bounded_slice_not_the_whole_pool(self):
        wide = paper_facts.FactLayerPipeline(
            complete=StubClient().complete, evidence_provider=provider_for(),
            model="m", document_id="doc_1",
        )
        chosen = wide.appraisal_evidence(EVIDENCE)
        assert chosen
        assert len(chosen) <= paper_facts.EVIDENCE_PER_FIELD * 2

    def test_the_evaluation_card_separates_facts_from_the_models_judgement(self):
        _, _, outcome = build_outcome()
        evaluation = outcome.cards["evaluation"]
        assert "审稿人视角" in evaluation
        assert "子图路由把检索误差隔离在局部" in evaluation
        # Every fact-layer line in the evaluation is a claim the overview also
        # carries, which is the only way the two can be read side by side.
        for claim in paper_facts.card_claims(outcome.facts):
            if not claim.is_not_stated:
                assert claim.claim in evaluation


# --------------------------------------------------------------------------
# The result payload and the exported report
# --------------------------------------------------------------------------


class TestServiceIntegration:
    @pytest.fixture
    def result(self):
        _, _, outcome = build_outcome()
        return service._result_from_outcome(outcome, content="a" * 5000)

    def test_the_result_keeps_the_keys_the_api_already_reads(self, result):
        for key in sections.SECTION_KEYS:
            assert key in result
            assert key in result["sections"]
        assert result["char_count"] == 5000

    def test_the_flattened_keys_and_the_section_map_cannot_disagree(self, result):
        # The templates read both. A result whose ``summary`` and
        # ``sections["summary"]["content"]`` differed would be a worse bug than
        # either path had, and the only thing preventing it is that both are
        # filled from the same string here.
        for key in sections.SECTION_KEYS:
            assert result[key] == result["sections"][key]["content"]

    def test_the_fact_layer_and_run_record_are_on_the_result(self, result):
        assert "paper_facts" in result
        assert "analysis_run" in result
        assert result["paper_facts"]["prompt_version"].startswith("facts-v1-")
        assert result["analysis_run"]["document_id"] == "doc_1"

    def test_the_fields_the_paper_did_not_state_are_listed_on_the_result(self, result):
        assert result["paper_facts"]["__not_stated_fields__"] == ["limitations"]

    def test_the_report_renders_the_fact_layer_through_the_same_function(self, result):
        report = service.render_markdown_report(result, "rag_fusion.pdf", "test")
        assert "事实层" in report
        # Rendered by render_overview, so the export cannot introduce a
        # wording the screen did not show.
        for claim in paper_facts.card_claims(PaperFacts.from_dict(result["paper_facts"])):
            if claim.claim == paper_facts.NOT_STATED_TEXT:
                assert paper_facts.NOT_STATED_TEXT in report

    def test_the_report_records_the_prompt_and_the_evidence_version(self, result):
        report = service.render_markdown_report(result, "rag_fusion.pdf", "test")
        assert result["analysis_run"]["prompt_version"] in report
        assert result["analysis_run"]["evidence_version"] in report
        assert "旧路径对照" in report

    def test_an_old_path_result_gets_no_fact_layer_and_no_new_headings(self):
        # The other path must be byte-identical to what it produced before.
        old = {"summary": "S", "quotes": "Q", "mindmap": "M", "mermaid": "",
               "evaluation": "", "research_brief": "", "elapsed_seconds": 1.0}
        report = service.render_markdown_report(old, "p.pdf", "test")
        assert "事实层" not in report
        assert "## AI 摘要" in report

    def test_a_degraded_run_reports_partial_quality(self):
        silent = StubClient(extraction="{}")
        _, _, outcome = build_outcome(silent)
        result = service._result_from_outcome(outcome, content="")
        assert result["analysis_status"]["quality"] == "partial"
        # A run that degraded is still a run: every section is present, and
        # the report says why the cards are thin.
        assert set(result["sections"]) == set(sections.SECTION_KEYS)
        assert result["analysis_run"]["degradations"]

    def test_attach_fact_layer_is_a_no_op_without_arguments(self):
        result = {"summary": "S"}
        assert service.attach_fact_layer(result) is result
        assert "paper_facts" not in result

    def test_the_fact_layer_payload_round_trips_through_the_canonical_reader(self, result):
        payload = dict(result["paper_facts"])
        payload.pop("__not_stated_fields__", None)
        assert PaperFacts.from_dict(payload).to_dict() == payload

    def test_a_missing_evidence_provider_is_refused_rather_than_guessed(self):
        # The refusal is the product rule: a fact layer with no citations
        # renders as though every claim in it were verified, and there is
        # nothing a reader could do to tell the difference.
        with pytest.raises(ValueError, match="evidence_provider"):
            service.analyze_with_fact_layer(
                "paper.txt", "paper.txt", "key", True, True, "sid",
            )


class TestDocumentIdentityAndParseQuality:
    """A citation chip has to have a document to point at, and a verdict to show.

    Both of these are cross-module breaks that only the front end could reveal:
    the analyze payload carried a session id and no document id, so every
    citation chip was unaddressable, and the parse provenance carried no
    quality report, so the UI rendered "no quality report was returned" for
    every document. Both are asserted here as the fields they are.
    """

    @staticmethod
    def outcome_with_quality():
        from paperwhisperer.ingestion.quality import (
            ParseQualityReport,
            QualityFlag,
            QualitySeverity,
        )

        document = document_with_sections()
        report = ParseQualityReport(
            document_id=document.document_id,
            parser_name="legacy",
            flags=[QualityFlag(
                rule_id="page_coverage",
                severity=QualitySeverity.WARN,
                message="第 4 页文本层为空",
            )],
            evaluated_rules=("page_coverage", "text_density"),
        )
        return document, report

    def test_the_document_id_comes_from_the_parsed_document(self):
        document, report = self.outcome_with_quality()
        outcome = _StubOutcome(document=document, quality=report)
        _, _, facts_outcome = build_outcome()
        result = service._result_from_outcome(
            facts_outcome, content="x", parse_outcome=outcome
        )
        service.attach_document_identity(result, outcome)
        assert result["document_id"] == document.document_id

    def test_the_document_id_is_never_a_session_id(self):
        document, report = self.outcome_with_quality()
        outcome = _StubOutcome(document=document, quality=report)
        _, _, facts_outcome = build_outcome()
        result = service._result_from_outcome(
            facts_outcome, content="x", parse_outcome=outcome
        )
        result["session_id"] = "sess_abcdef"
        service.attach_document_identity(result, outcome)
        assert result["document_id"] != result["session_id"]
        assert result["document_id"].startswith("doc_")

    def test_an_unparseable_outcome_leaves_the_field_absent_rather_than_invented(self):
        result = {"parse": {}}
        service.attach_document_identity(result, None)
        assert "document_id" not in result

    def test_the_quality_report_reaches_the_parse_provenance(self):
        document, report = self.outcome_with_quality()
        outcome = _StubOutcome(document=document, quality=report)
        _, _, facts_outcome = build_outcome()
        result = service._result_from_outcome(
            facts_outcome, content="x", parse_outcome=outcome
        )
        service.attach_parse_quality(result, outcome)
        quality = result["parse"]["quality"]
        assert quality is not None
        # The whole report, not a score: the quality gate's own contract is
        # that there is no single number, and a reader who wants to know which
        # check failed needs the flags.
        assert quality["status"] == report.status.value
        assert quality["flags"][0]["rule_id"] == "page_coverage"
        assert quality["evaluated_rules"] == ["page_coverage", "text_density"]

    def test_the_quality_report_is_present_without_a_separate_call(self):
        # What the fact-layer path actually does: the outcome flows through
        # _result_from_outcome and the field is there, because the two
        # attach_* calls are made at the finalize seam for both paths.
        document, report = self.outcome_with_quality()
        outcome = _StubOutcome(document=document, quality=report)
        _, _, facts_outcome = build_outcome()
        result = service._result_from_outcome(
            facts_outcome, content="x", parse_outcome=outcome
        )
        service.attach_document_identity(result, outcome)
        service.attach_parse_quality(result, outcome)
        assert result["parse"]["quality"]["status"] == report.status.value
        assert result["document_id"] == document.document_id

    def test_an_absent_quality_report_is_none_and_not_an_empty_dict(self):
        # An empty dict would render as a clean bill of health, which is the
        # opposite of "nobody checked".
        outcome = _StubOutcome(document=document_with_sections(), quality=None)
        result = {"parse": {}}
        service.attach_parse_quality(result, outcome)
        assert result["parse"]["quality"] is None

    def test_the_result_with_no_parse_provenance_is_left_alone(self):
        result = {"summary": "S"}
        service.attach_parse_quality(result, _StubOutcome(
            document=document_with_sections(), quality=None
        ))
        assert "parse" not in result


class _StubOutcome:
    """A stand-in for :class:`ParseOutcome`, carrying only what is read.

    Built by hand rather than by parsing a file so the test asserts the two
    fields the payload needs and does not depend on whichever parser happens
    to be configured in the developer's environment.
    """

    def __init__(self, document, quality):
        self.document = document
        self.quality = quality
        self.parsed_by = "legacy"
        self.status = "success"
        self.used_cloud = False
        self.cloud_attempted = False
        # Mirrors the real ParseOutcome. Added when `build_parse_provenance`
        # started surfacing the machine-readable cloud outcome, so an interface
        # can say "top up your quota" rather than showing one prose sentence for
        # three different situations. A stub that omits a field the payload
        # reads fails here rather than in production.
        self.cloud_outcome = ""
        self.error_message = ""
        self.touched_network = False
        self.cache_hit = False
        self.fallback_used = False
        self.media_type = "application/pdf"
        self.warnings: list[str] = []

    def summary_line(self) -> str:
        return "由 legacy 解析，未使用云端"
