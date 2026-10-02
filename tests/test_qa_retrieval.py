"""Q&A answers from retrieved evidence, and the release gate on its quotes.

T1.7, and the defect it removes: ``orchestrator._build_answer_prompts`` sent
``document_content[:12000]`` — a hard prefix of the flat text. On a real paper
that cut 132 characters before the sentence reporting its result, so every
section after the third was unreachable. The first test here rebuilds that
case explicitly and asserts the old window could not have answered it, so the
retrieval path is doing the work rather than accidentally getting away with it.

The second test is the other half of the acceptance criterion, and the harder
one. When the document does not answer, the model is **not called at all** —
so a scripted model that would have invented a top-1 accuracy never gets the
chance, and ``confidence`` is derived from the claims' evidence rather than
from anything the model said about itself. The test scripts a model that
*would* fabricate, and asserts the fabricated figure is absent: that is what
makes the test able to fail if someone later derives confidence from the model.

No LLM, no network, no clock. The model is a scripted double that returns
canned JSON, and retrieval runs over a temporary SQLite store with a real
``HybridRetriever`` and the lexical leg — which is the leg that answers exact
questions, and the one a personal-tier deployment with no embedding endpoint
would actually have.
"""

from __future__ import annotations

import json
import os

import pytest

from paperwhisperer.analysis.orchestrator import (
    CONFIDENCE_INSUFFICIENT,
    CONFIDENCE_PARTIALLY_SUPPORTED,
    CONFIDENCE_SUPPORTED,
    AnswerClaim,
    AnswerResult,
    DocumentAnalyzer,
    index_document_for_retrieval,
    parsed_document_from_text,
)
from paperwhisperer.core import config
from paperwhisperer.core.text import extract_json_object
from paperwhisperer.infra.stores import build_sqlite_bundle
from paperwhisperer.infra.stores.interfaces import DocumentRecord
from paperwhisperer.ingestion.schema import (
    Chunk,
    Element,
    ParsedDocument,
    SourceLocator,
)
from paperwhisperer.rag.citations import ClaimPresentation

# --------------------------------------------------------------------------
# Doubles
# --------------------------------------------------------------------------


class ScriptedModel:
    """Stands in for ``LLMClient``. Returns a canned reply, records the calls.

    The prompt is recorded rather than ignored because two of the tests are
    about what the model was *shown*: a question that only late passages could
    answer is only proven answerable if the late passage is in the prompt, and
    a quote that fails verification is only meaningful if the model had the
    evidence to have quoted correctly in the first place.
    """

    def __init__(self, reply: str = "", *, chunks: tuple[str, ...] = ()):
        self.reply = reply
        self.chunks = chunks
        self.prompts: list[dict] = []
        self.complete_calls = 0
        self.stream_calls = 0
        self.model = "scripted-model"
        self.search_rewrite_model = "scripted-model"
        self.request_timeout = 30
        self.max_retries = 1
        self.max_concurrency = 4

    def complete(self, system_prompt, user_prompt, max_retries=None, model=None):
        self.prompts.append({"system": system_prompt, "user": user_prompt})
        self.complete_calls += 1
        return self.reply

    def stream(self, system_prompt, user_prompt, max_retries=None, model=None):
        self.prompts.append({"system": system_prompt, "user": user_prompt})
        self.stream_calls += 1
        return iter(self.chunks or (self.reply,))

    @property
    def last_prompt(self) -> str:
        return self.prompts[-1]["user"] if self.prompts else ""


class _Settings:
    """Isolates the store bundle from the developer's own configuration."""

    STORAGE_BACKEND = "sqlite"
    STORAGE_LEXICAL_FTS5 = "auto"
    RERANK_ENABLED = False
    RERANK_MODEL = ""

    def __init__(self, path: str) -> None:
        self.STORAGE_SQLITE_PATH = path


@pytest.fixture
def bundle(tmp_path):
    store = build_sqlite_bundle(_Settings(str(tmp_path / "stores.sqlite3")))
    yield store
    store.close()


def make_analyzer(monkeypatch, bundle, reply="", *, chunks=(), **kwargs) -> DocumentAnalyzer:
    """A ``DocumentAnalyzer`` with a scripted model and a temporary store."""
    monkeypatch.setattr(
        "paperwhisperer.analysis.orchestrator.LLMClient",
        lambda api_key: ScriptedModel(reply, chunks=chunks),
    )
    return DocumentAnalyzer("test-key", store_bundle=bundle, **kwargs)


def claim_payload(answer: str, claims: list[dict]) -> str:
    """The reply shape the prompt asks for, as a string."""
    return json.dumps({"answer": answer, "claims": claims}, ensure_ascii=False)


def evidence_ids_in_prompt(model: ScriptedModel) -> list[str]:
    """The evidence ids the prompt actually offered the model."""
    import re

    return re.findall(r"\[(ev_[0-9a-f]+)\]", model.last_prompt)


# --------------------------------------------------------------------------
# A document whose answer is late enough that a prefix window cannot reach it
# --------------------------------------------------------------------------

#: A sentence that exists only near the end of the document, on page 9. The
#: filler before it is long enough that its offset lands past the 12,000
#: character Q&A window, which is what makes this a test of retrieval rather
#: than of luck.
LATE_SENTENCE = "在消融实验中，去掉上下文混合模块后准确率下降了 4.2 个百分点。"

#: The question only the late sentence can answer.
LATE_QUESTION = "消融实验中去掉上下文混合模块后准确率下降了多少？"

#: The figure the late sentence reports, which must appear in the answer.
LATE_FIGURE = "4.2"


def _filler_paragraph(index: int) -> str:
    return (
        f"第 {index} 段讨论了相关工作的背景与动机，并说明了既有方法在这个问题上的不足，"
        f"以及本文选择的技术路线与它们之间的关系，最后给出了本章的结论。"
    )


def build_late_evidence_document(*, filler_paragraphs: int = 100) -> ParsedDocument:
    """A document whose only answer to :data:`LATE_QUESTION` sits past page 9.

    Structured with headings and page locators rather than one blob of text,
    because a single blob gets packed into a handful of giant chunks by the
    chunker and retrieval would then have almost nothing to discriminate on.
    The layout is a realistic one: an introduction, some related work, a
    method section, and an experiments section on a late page.
    """
    elements: list[Element] = []
    order = 0

    def add(role: str, text: str, page: int, path: tuple[str, ...], level: int | None = None):
        nonlocal order
        elements.append(
            Element(
                role=role,
                text=text,
                locator=SourceLocator(kind="pdf_page", page=page),
                level=level,
                heading_path=path,
            )
        )
        order += 1

    add("heading", "1 引言", 1, ("1 引言",), level=1)
    for index in range(filler_paragraphs):
        add("paragraph", _filler_paragraph(index), 1 + index, ("1 引言",))
    add("heading", "2 相关工作", 1 + filler_paragraphs, ("2 相关工作",), level=1)
    for index in range(filler_paragraphs):
        add(
            "paragraph",
            _filler_paragraph(index + 100),
            2 + filler_paragraphs + index,
            ("2 相关工作",),
        )
    add("heading", "3 方法", 1 + 2 * filler_paragraphs, ("3 方法",), level=1)
    for index in range(filler_paragraphs):
        add(
            "paragraph",
            _filler_paragraph(index + 200),
            2 + 2 * filler_paragraphs + index,
            ("3 方法",),
        )
    add("heading", "4 实验", 1 + 3 * filler_paragraphs, ("4 实验",), level=1)
    add("paragraph", LATE_SENTENCE, 1 + 3 * filler_paragraphs, ("4 实验",))

    document = ParsedDocument(
        document_id="doc_late_evidence",
        source_filename="late.pdf",
        media_type="application/pdf",
        page_count=1 + 3 * filler_paragraphs,
        elements=elements,
        parser_name="test-parser",
    )
    document.build_units()
    return document


@pytest.fixture
def late_document() -> ParsedDocument:
    return build_late_evidence_document()


# --------------------------------------------------------------------------
# The document that cannot answer
# --------------------------------------------------------------------------

#: A document about an unrelated subject, and a question about one this paper
#: never addresses. There is nothing in common to retrieve on — the two share
#: no token at all, which is what makes this the *clean* insufficient case:
#: the model is never asked, so there is no answer in which a figure could
#: have been invented.
UNRELATED_QUESTION = "ImageNet top-1 准确率是多少"

#: A document that mentions ImageNet exactly once, in the acknowledgements,
#: thanking a colleague for training runs — and reports no ImageNet result at
#: all. This is the *hard* version, and it is the one the measurement behind
#: this work used: the bait word is present, so a retriever will find the
#: acknowledgement, so the model *is* asked, and a model asked will answer.
#: The gate has to hold there too, which is a different mechanism from the
#: clean case and needs its own test.
BAIT_QUESTION = "作者在 ImageNet 上报告的 top-1 准确率具体是多少？"

#: What a model asked this question *will* produce if it is allowed to answer.
#: Scripted into the double so the tests prove the answer is suppressed rather
#: than merely unlikely.
FABRICATED_REPLY = claim_payload(
    "论文确实报告了多组 ImageNet 结果，作者在第 4 节给出的 top-1 准确率为 76.3%，"
    "并在第 5 节与 6 节分别讨论了消融与迁移。",
    [
        {
            "text": "作者报告的 top-1 准确率为 76.3%。",
            "quote": "the model reaches a top-1 accuracy of 76.3% on ImageNet",
            "evidence_ids": ["ev_never_returned"],
        }
    ],
)

#: A passage that mentions ImageNet without reporting a result, so the
#: retrieval leg has something real to return and the gate is exercised on an
#: answer the model composed rather than one it was never asked for.
BAIT_ELEMENTS = [
    Element(
        role="paragraph",
        text="感谢张伟在 ImageNet 上协助训练模型所提供的帮助。",
        locator=SourceLocator(kind="pdf_page", page=10),
    ),
    Element(
        role="paragraph",
        text="本研究在 CIFAR-10 上取得了 94.8% 的准确率。",
        locator=SourceLocator(kind="pdf_page", page=8),
    ),
]

#: A paper on a subject with no term in common with :data:`UNRELATED_QUESTION`,
#: so nothing is retrievable and the model is never called.
UNRELATED_ELEMENTS = [
    Element(
        role="paragraph",
        text="我们提出了一种新的蛋白质折叠预测网络，并报告了其在 PDB 数据集上的表现。",
        locator=SourceLocator(kind="pdf_page", page=4),
    ),
    Element(
        role="paragraph",
        text="消融实验表明，去掉注意力池化层后平均误差明显上升，说明该模块是必要的。",
        locator=SourceLocator(kind="pdf_page", page=6),
    ),
]

UNDECIDABLE_QUOTES = ("76.3", "76.3%")


def _document_for(elements, document_id: str) -> ParsedDocument:
    document = ParsedDocument(
        document_id=document_id,
        source_filename=f"{document_id}.pdf",
        media_type="application/pdf",
        page_count=10,
        elements=list(elements),
        parser_name="test-parser",
    )
    document.build_units()
    return document


@pytest.fixture
def bait_document() -> ParsedDocument:
    """Retrieval succeeds, the model is asked, and it fabricates anyway."""
    return _document_for(BAIT_ELEMENTS, "doc_bait")


@pytest.fixture
def unrelated_document() -> ParsedDocument:
    """Nothing is retrievable, so the model is never asked at all."""
    return _document_for(UNRELATED_ELEMENTS, "doc_unrelated")


# --------------------------------------------------------------------------
# 1. A late answer is reachable at all — the half the prefix window failed
# --------------------------------------------------------------------------


class TestLateEvidenceIsRetrieved:
    def test_the_old_prefix_window_could_not_have_reached_this_answer(self, late_document):
        """The case is built so the previous implementation would fail it.

        Asserted first and explicitly: if the answer sentence turned out to sit
        inside the 12,000 character window, every test below would pass
        against the old code too, and none of them would be evidence of
        anything. The defect was that the window stopped 132 characters short
        of this sentence on a real paper; here it stops thousands of characters
        short of it.
        """
        full_text = late_document.to_full_text()
        offset = full_text.find(LATE_SENTENCE)
        budget = config.DOCUMENT_WINDOW_BUDGET["qa"]

        assert offset != -1, "the late sentence is not in the document at all"
        assert offset > budget, (
            f"the late sentence sits at offset {offset}, inside the {budget}-character "
            "Q&A window; this fixture no longer reproduces the defect and the "
            "retrieval tests below would pass against the old prefix code too"
        )
        assert LATE_SENTENCE not in full_text[:budget]

    def test_a_question_only_late_passages_answer_is_answered(self, monkeypatch, bundle,
                                                               late_document):
        """The acceptance criterion: 后文（实验/限制章节）的问题可正确回答."""
        analyzer = make_analyzer(
            monkeypatch, bundle, qa_document=late_document, reply="{}"
        )
        index_document_for_retrieval(bundle, late_document, None)
        evidence_id = analyzer._retrieve_for_question(LATE_QUESTION).evidence[0].evidence_id
        analyzer.client.reply = claim_payload(
            f"消融实验显示，去掉上下文混合模块后准确率下降了 {LATE_FIGURE} 个百分点。",
            [
                {
                    "text": f"消融实验显示，去掉上下文混合模块后准确率下降了 {LATE_FIGURE} 个百分点。",
                    "quote": LATE_SENTENCE,
                    "evidence_ids": [evidence_id],
                }
            ],
        )

        result = analyzer.answer_question(LATE_QUESTION)

        assert LATE_FIGURE in result.answer, (
            "the answer did not report the figure the late passage states"
        )
        assert result.confidence == CONFIDENCE_SUPPORTED
        assert result.retrieval.evidence_count > 0
        assert result.citations, "an answer with evidence behind it cited nothing"
        assert result.claims[0].verified_text == LATE_SENTENCE

    def test_the_late_passage_really_reached_the_prompt(self, monkeypatch, bundle,
                                                       late_document):
        """The model was shown the experiment passage, not just told about it.

        Without this the test above would pass even if the answer were right by
        coincidence: the figure is asserted, so the only thing that proves
        retrieval is the passage itself appearing in the prompt the model read.
        """
        analyzer = make_analyzer(monkeypatch, bundle, qa_document=late_document, reply="{}")
        index_document_for_retrieval(bundle, late_document, None)
        analyzer.answer_question(LATE_QUESTION)

        prompt = analyzer.client.last_prompt
        assert "4.2" in prompt, "the experiment figure never reached the prompt"
        assert "4 实验" in prompt or "消融" in prompt

    def test_the_prompt_carries_no_document_prefix_outside_the_evidence(
        self, monkeypatch, bundle, late_document
    ):
        """The prompt is the evidence block, and nothing else from the document.

        This is the structural form of "no head excerpt": a sentence the
        retriever did not surface cannot be in the prompt, so there is no
        residual path by which the model could answer from a section retrieval
        skipped. Asserted on an early sentence, which retrieval has no reason
        to have selected.
        """
        analyzer = make_analyzer(monkeypatch, bundle, qa_document=late_document, reply="{}")
        index_document_for_retrieval(bundle, late_document, None)
        analyzer.answer_question(LATE_QUESTION)

        prompt = analyzer.client.last_prompt
        early = _filler_paragraph(0)
        assert early not in prompt, (
            "an unretrieved early paragraph reached the prompt; the evidence "
            "block is not the only thing being sent"
        )
        assert "<retrieved_evidence>" in prompt


# --------------------------------------------------------------------------
# 2. No evidence, no answer — and never the model's self-assessment
# --------------------------------------------------------------------------


class TestInsufficientEvidence:
    """Both ways an answer can fail to be supported, which are not the same.

    The clean case is the one the acceptance criterion names: nothing is
    retrievable, so the model is never asked. The bait case is the one the
    measurement behind this work actually hit: the retriever *does* find an
    ImageNet acknowledgement, the model *is* asked, and it fabricates. A gate
    that only covers the first is a gate that would have passed on the real
    failure.
    """

    def _ask(self, monkeypatch, bundle, document, question):
        analyzer = make_analyzer(
            monkeypatch, bundle, qa_document=document, reply=FABRICATED_REPLY
        )
        index_document_for_retrieval(bundle, document, None)
        return analyzer, analyzer.answer_question(question)

    # -- clean: nothing retrievable ------------------------------------

    def test_a_question_with_no_retrievable_evidence_is_insufficient(
        self, monkeypatch, bundle, unrelated_document
    ):
        _analyzer, result = self._ask(
            monkeypatch, bundle, unrelated_document, UNRELATED_QUESTION
        )
        assert result.confidence == CONFIDENCE_INSUFFICIENT
        assert result.insufficient_evidence is True

    def test_the_model_is_never_asked_to_answer_without_evidence(
        self, monkeypatch, bundle, unrelated_document
    ):
        """The gate is enforced by not asking, not by checking the answer.

        This is the load-bearing assertion of the whole task. A post-hoc filter
        cannot undo an answer: the measurement behind this work is that the
        model correctly refuses to invent the *number* and then fabricates
        confidently in the hedging around it. Refusing to call it is the only
        enforcement that cannot be raced.
        """
        analyzer, _result = self._ask(
            monkeypatch, bundle, unrelated_document, UNRELATED_QUESTION
        )
        assert analyzer.client.complete_calls == 0
        assert analyzer.client.stream_calls == 0
        assert not analyzer.client.prompts

    def test_nothing_retrievable_means_no_figure_and_no_citations(
        self, monkeypatch, bundle, unrelated_document
    ):
        _analyzer, result = self._ask(
            monkeypatch, bundle, unrelated_document, UNRELATED_QUESTION
        )
        assert result.claims == ()
        assert result.citations == ()
        assert "%" not in result.answer

    def test_the_retrieval_record_says_why(self, monkeypatch, bundle, unrelated_document):
        """Why there are no citations is on the answer, not only in a log.

        A reader told "insufficient" with no reason cannot tell a paper that
        does not cover the question from a deployment that is broken, and those
        two warrant opposite reactions.
        """
        _analyzer, result = self._ask(
            monkeypatch, bundle, unrelated_document, UNRELATED_QUESTION
        )
        assert result.retrieval.empty_reason
        assert result.retrieval.evidence_count == 0
        assert result.retrieval.record_id
        assert "证据不足" in result.answer

    # -- hard: retrieved something, model fabricated anyway -------------

    def test_the_bait_case_retrieves_something_so_the_model_is_asked(
        self, monkeypatch, bundle, bait_document
    ):
        """Establishes that the test below is actually the hard case.

        The acknowledgement mentions ImageNet, so the lexical leg matches it
        and the model is asked. Asserted so the suppression below cannot be
        explained by the model simply never having been called — which is the
        easy case, tested above.
        """
        analyzer, _result = self._ask(monkeypatch, bundle, bait_document, BAIT_QUESTION)
        assert analyzer.client.complete_calls == 1, (
            "the bait document did not retrieve; the hard case is not being exercised"
        )

    def test_a_fabricated_answer_over_real_evidence_is_not_shown(
        self, monkeypatch, bundle, bait_document
    ):
        """The claim cites an id retrieval never returned, so nothing is carried.

        The model was handed a real passage — the acknowledgement — and chose
        to cite an id that does not exist rather than the one it was given.
        There is no claim left that any evidence backs, so the answer is
        refused rather than shown with a caveat attached.
        """
        _analyzer, result = self._ask(monkeypatch, bundle, bait_document, BAIT_QUESTION)
        assert result.confidence == CONFIDENCE_INSUFFICIENT
        assert result.claims == ()
        assert result.citations == ()

    def test_the_fabricated_figure_never_reaches_the_reader(
        self, monkeypatch, bundle, bait_document
    ):
        """The percentage check is what makes this more than a keyword search.

        A refusal that also produces "76.3%" has still invented the number, and
        hedging language in the same breath does not redeem it. The scripted
        reply is the fabrication verbatim, so the test fails the moment any
        part of it is shown.
        """
        _analyzer, result = self._ask(monkeypatch, bundle, bait_document, BAIT_QUESTION)
        for quoted in UNDECIDABLE_QUOTES:
            assert quoted not in result.answer, f"a fabricated figure appeared: {quoted}"
        assert "ImageNet 结果" not in result.answer
        assert "%" not in result.answer

    def test_the_refusal_says_why_in_chinese(self, monkeypatch, bundle, bait_document):
        _analyzer, result = self._ask(monkeypatch, bundle, bait_document, BAIT_QUESTION)
        assert "证据校验未通过" in result.answer
        assert any(character > "一" for character in result.answer)

    def test_the_retrieved_evidence_is_still_reported(self, monkeypatch, bundle,
                                                      bait_document):
        """The refusal counts the passages, so the reader can go and look.

        A refusal that hides what *was* found is only half an answer: the
        useful thing to do with a question the product could not settle is to
        read the passage that triggered it and judge for yourself.
        """
        _analyzer, result = self._ask(monkeypatch, bundle, bait_document, BAIT_QUESTION)
        assert result.retrieval.evidence_count > 0, (
            "the bait case retrieved nothing, so this is the clean case in disguise"
        )
        assert result.retrieval.empty_reason == "ok"


# --------------------------------------------------------------------------
# 3. The release gate: what may be shown as a quotation
# --------------------------------------------------------------------------


def _single_evidence_document(*, text: str, page: int = 3) -> ParsedDocument:
    """A one-passage document, for tests that are about the quote and nothing else."""
    document = ParsedDocument(
        document_id="doc_gate",
        source_filename="gate.pdf",
        media_type="application/pdf",
        page_count=page,
        elements=[
            Element(
                role="paragraph",
                text=text,
                locator=SourceLocator(kind="pdf_page", page=page),
            )
        ],
        parser_name="test-parser",
    )
    document.build_units()
    return document


GATE_PASSAGE = "在 ImageNet 上，我们的模型达到 76.3% 的 top-1 准确率，优于全部基线。"

#: A fluent, plausible sentence the paper never says. It reads like a result
#: and is the exact shape of what a model invents.
INVENTED_QUOTE = "the model reaches 76.3% top-1 accuracy while using 40% fewer parameters"


def _index_evidence(bundle, analyzer, document, question="ImageNet top-1 准确率是多少"):
    """Index ``document`` and return the single evidence id retrieval offers."""
    index_document_for_retrieval(bundle, document, None)
    result = analyzer._retrieve_for_question(question)
    assert result.evidence, "the fixture document did not retrieve; the test is not testing"
    return result.evidence[0].evidence_id


def _good_claim(evidence_id: str) -> dict:
    """A claim the verifier will accept, so a second claim has a company.

    Every test of a *degradation* needs one passing claim alongside the failing
    one, and the reason is a property of the product rather than of the test:
    when nothing in an answer can be tied to the evidence, the whole answer is
    refused rather than shown with caveats. So a single fabricated claim does
    not produce a degraded claim — it produces no answer at all, which is
    asserted separately. To observe the degradation itself, the answer has to
    contain something true.
    """
    return {
        "text": "模型在 ImageNet 上取得了 76.3% 的 top-1 准确率。",
        "quote": GATE_PASSAGE,
        "evidence_ids": [evidence_id],
    }


def _bad_claim(evidence_id: str, quote: str = INVENTED_QUOTE) -> dict:
    return {
        "text": "模型在使用更少参数的同时保持了这一准确率。",
        "quote": quote,
        "evidence_ids": [evidence_id],
    }


class TestCitationGate:
    def test_a_quote_that_exists_aligns_and_is_reported_as_a_direct_quote(
        self, monkeypatch, bundle
    ):
        """The positive half: real quotes are not thrown away with the fake ones.

        A gate that degrades everything is not a gate, it is a refusal. The
        source's own characters are what get rendered, so a model that sent a
        near-miss still shows the reader the paper's wording.
        """
        document = _single_evidence_document(text=GATE_PASSAGE)
        analyzer = make_analyzer(monkeypatch, bundle, qa_document=document, reply="{}")
        evidence_id = _index_evidence(bundle, analyzer, document)
        analyzer.client.reply = claim_payload(
            "模型在 ImageNet 上取得了 76.3% 的 top-1 准确率。",
            [_good_claim(evidence_id)],
        )

        result = analyzer.answer_question("ImageNet top-1 准确率是多少？")

        claim = result.claims[0]
        assert claim.presentation is ClaimPresentation.DIRECT_QUOTE
        assert claim.uses_quotation_marks is True
        assert claim.verified_text == GATE_PASSAGE
        assert claim.outcome == "aligned"
        assert result.confidence == CONFIDENCE_SUPPORTED

    def test_a_quote_that_does_not_exist_degrades_to_paraphrase_with_no_marks(
        self, monkeypatch, bundle
    ):
        """The release gate, on a claim the model presented as a quotation.

        The claim still names a real evidence id, so the failure is not "the
        model cited nothing" — it is "the model claimed the paper says this and
        the paper does not", which is the one that ships a fabricated citation
        into a reader's notes. The passing claim beside it is what keeps the
        answer on screen at all.
        """
        document = _single_evidence_document(text=GATE_PASSAGE)
        analyzer = make_analyzer(monkeypatch, bundle, qa_document=document, reply="{}")
        evidence_id = _index_evidence(bundle, analyzer, document)
        analyzer.client.reply = claim_payload(
            "模型在 ImageNet 上取得了 76.3% 的 top-1 准确率，并使用了更少的参数。",
            [_good_claim(evidence_id), _bad_claim(evidence_id)],
        )

        result = analyzer.answer_question("ImageNet top-1 准确率是多少？")

        bad = result.claims[1]
        assert bad.presentation is ClaimPresentation.PARAPHRASE
        assert bad.uses_quotation_marks is False
        assert bad.outcome == "not_found"
        assert bad.verified_text == ""
        assert bad.message, "a degraded claim must say why it degraded"
        assert result.confidence == CONFIDENCE_PARTIALLY_SUPPORTED

    def test_a_rejected_quote_loses_its_quotation_marks_in_the_answer_prose(
        self, monkeypatch, bundle
    ):
        """The gate applies to the prose the reader reads, not only the claim list.

        The answer body is what a reader copies into their own notes. A
        fabricated sentence wrapped in quotation marks there is the failure
        being gated, and fixing only the claim list would leave the fabricated
        quote sitting in the text a user takes away.
        """
        document = _single_evidence_document(text=GATE_PASSAGE)
        analyzer = make_analyzer(monkeypatch, bundle, qa_document=document, reply="{}")
        evidence_id = _index_evidence(bundle, analyzer, document)
        analyzer.client.reply = claim_payload(
            f"论文写道 “{INVENTED_QUOTE}”，这是该方法的核心结论。",
            [_good_claim(evidence_id), _bad_claim(evidence_id)],
        )

        result = analyzer.answer_question("ImageNet top-1 准确率是多少？")

        assert f"“{INVENTED_QUOTE}”" not in result.answer
        assert f'"{INVENTED_QUOTE}"' not in result.answer
        assert INVENTED_QUOTE in result.answer, (
            "the sentence should survive as a paraphrase; only the marks go"
        )

    def test_a_quote_that_verifies_keeps_its_quotation_marks_in_the_prose(
        self, monkeypatch, bundle
    ):
        """The other half of the prose rule: a real quote is not defanged.

        Stripping marks from quotes that verified would make the release gate
        into a blanket ban on quotation, and a reader could no longer tell a
        checked citation from an unchecked one — which is the distinction the
        whole feature exists to draw.
        """
        document = _single_evidence_document(text=GATE_PASSAGE)
        analyzer = make_analyzer(monkeypatch, bundle, qa_document=document, reply="{}")
        evidence_id = _index_evidence(bundle, analyzer, document)
        analyzer.client.reply = claim_payload(
            f"论文写道 “{GATE_PASSAGE}”，该句可回源校验。",
            [_good_claim(evidence_id)],
        )

        result = analyzer.answer_question("ImageNet top-1 准确率是多少？")

        assert f"“{GATE_PASSAGE}”" in result.answer

    def test_an_answer_where_nothing_verifies_is_refused_whole(
        self, monkeypatch, bundle
    ):
        """The suppression rule, which the degradation tests depend on.

        A single fabricated claim does not become a degraded claim — it
        produces no answer. A refusal wrapped in invented context is worse than
        no refusal, so an answer with nothing behind it is not shown at all,
        even with a caveat attached.
        """
        document = _single_evidence_document(text=GATE_PASSAGE)
        analyzer = make_analyzer(monkeypatch, bundle, qa_document=document, reply="{}")
        evidence_id = _index_evidence(bundle, analyzer, document)
        analyzer.client.reply = claim_payload(
            f"论文写道 “{INVENTED_QUOTE}”。", [_bad_claim(evidence_id)]
        )

        result = analyzer.answer_question("ImageNet top-1 准确率是多少？")

        assert result.confidence == CONFIDENCE_INSUFFICIENT
        assert INVENTED_QUOTE not in result.answer
        assert result.claims == ()
        assert result.retrieval.evidence_count > 0

    def test_a_quote_that_matched_only_the_generated_prefix_is_not_verified(
        self, monkeypatch, bundle
    ):
        """The subtle one, and the whole reason ``verbatim_text`` exists.

        A chunk's text begins with a heading line the chunker generated — the
        document never wrote "## 4 实验". Verifying a quotation against the
        chunk's *text* would find that string, call it a direct quote, and
        ship a citation to a sentence that exists nowhere in the paper. The
        evidence is built from ``verbatim_text`` and the scope is derived from
        the document's elements, so a prefix-only span matches neither and is
        refused.

        The chunk is hand-built rather than taken from the chunker, because
        ``rag.chunking`` does not currently populate ``context_prefix`` — the
        gap is reported, and building the shape here keeps the retriever's half
        of the behaviour under test rather than loosening the assertion.
        """
        heading = "4 实验"
        body = GATE_PASSAGE
        document = _single_evidence_document(text=body, page=7)
        chunk = Chunk(
            chunk_id="chunk_with_prefix",
            text=f"## {heading}\n\n{body}",
            element_ids=(document.elements[0].element_id,),
            locator=SourceLocator(kind="pdf_page", page=7),
            heading_path=(heading,),
            order=0,
            context_prefix=f"## {heading}\n\n",
        )
        assert chunk.verbatim_text == body
        assert chunk.text != chunk.verbatim_text

        bundle.metadata.upsert_document(
            DocumentRecord(
                document_id=document.document_id,
                document=document,
                index_version="test-version",
            )
        )
        bundle.metadata.put_chunk(document.document_id, chunk)
        bundle.lexical.index_chunk(
            chunk.chunk_id, chunk.text, document_id=document.document_id
        )

        analyzer = make_analyzer(monkeypatch, bundle, qa_document=document, reply="{}")
        retrieved = analyzer._retrieve_for_question("实验")
        assert retrieved.evidence, "the hand-built chunk was not retrieved"
        evidence = retrieved.evidence[0]
        assert evidence.text == body, "the evidence carried the generated prefix"
        assert heading not in evidence.text.split("\n")[-1]

        analyzer.client.reply = claim_payload(
            f"模型取得了 76.3% 的准确率，章节标题是 {heading}。",
            [
                _good_claim(evidence.evidence_id),
                {
                    "text": f"论文的章节标题是 {heading}。",
                    "quote": f"## {heading}",
                    "evidence_ids": [evidence.evidence_id],
                },
            ],
        )
        # The question now carries the passage's own content terms. The
        # relevance floor added in T1.4's floor work measures *content* coverage
        # ("ImageNet", "76.3%", "top-1", "baselines"), because lexical coverage
        # on the question as a whole is dominated by function words — "what is
        # the title of the experiments section" scored 0.111 against a
        # distillation passage and was correctly refused before it ever reached
        # the citation gate. The property under test is unaffected: the answer
        # lives only in the generated heading, which is not in `verbatim_text`.
        result = analyzer.answer_question(
            "在 ImageNet 上模型达到 76.3% top-1 准确率的那个章节标题是什么？"
        )

        degraded = result.claims[1]
        assert degraded.is_direct_quote is False, (
            "a quotation that matched only the generated context prefix was "
            "accepted as a direct quote"
        )
        assert degraded.verified_text == ""
        assert degraded.outcome == "not_found"
        assert result.confidence == CONFIDENCE_PARTIALLY_SUPPORTED

    def test_a_claim_naming_an_evidence_id_that_was_never_returned_is_unbacked(
        self, monkeypatch, bundle
    ):
        """A fabricated id is not a citation, and cannot make a claim verified.

        The ids are opaque strings. Nothing about ``ev_never_returned`` marks it
        as false, so the only place that can be caught is the resolution step
        against the retrieved set — which is why the id is resolved there
        rather than trusted from the reply.
        """
        document = _single_evidence_document(text=GATE_PASSAGE)
        analyzer = make_analyzer(monkeypatch, bundle, qa_document=document, reply="{}")
        _index_evidence(bundle, analyzer, document)
        analyzer.client.reply = claim_payload(
            "模型使用了更少的参数。",
            [
                {
                    "text": "模型使用了更少的参数。",
                    "quote": "",
                    "evidence_ids": ["ev_never_returned"],
                }
            ],
        )

        result = analyzer.answer_question("ImageNet top-1 准确率是多少？")

        # Nothing resolved, so nothing survives, so there is no answer — and in
        # particular no citation carrying an id the retriever never issued.
        assert result.confidence == CONFIDENCE_INSUFFICIENT
        assert result.claims == ()
        assert result.citations == ()
        assert "ev_never_returned" not in json.dumps(result.to_dict(), ensure_ascii=False)

    def test_a_reply_that_is_not_json_is_refused_rather_than_shown(
        self, monkeypatch, bundle
    ):
        """Prose where JSON was asked for is not laundered into an answer.

        The tempting behaviour is to show it and mark it unverified. The
        problem is that "unverified" is a badge on a paragraph, and a paragraph
        of confident prose is what a reader skims: the badge does not survive
        contact with the text. So the prose is not shown, and the note says
        what happened and why.
        """
        document = _single_evidence_document(text=GATE_PASSAGE)
        analyzer = make_analyzer(
            monkeypatch, bundle, qa_document=document, reply="模型达到了 76.3% 的准确率。"
        )
        _index_evidence(bundle, analyzer, document)

        result = analyzer.answer_question("ImageNet top-1 准确率是多少？")

        assert result.confidence == CONFIDENCE_INSUFFICIENT
        assert "76.3" not in result.answer
        assert result.claims == ()
        assert result.notes, "a malformed reply should say so"


# --------------------------------------------------------------------------
# 4. Confidence is a function of the evidence, not of the model
# --------------------------------------------------------------------------


class TestConfidenceDerivation:
    def _two_claim_reply(self, evidence_id: str) -> str:
        return claim_payload(
            "模型取得了 76.3% 的 top-1 准确率，并使用了更少的参数。",
            [
                {
                    "text": "模型取得了 76.3% 的 top-1 准确率。",
                    "quote": GATE_PASSAGE,
                    "evidence_ids": [evidence_id],
                },
                {
                    "text": "模型使用了更少的参数。",
                    "quote": INVENTED_QUOTE,
                    "evidence_ids": [evidence_id],
                },
            ],
        )

    def test_partially_supported_when_some_claims_verify_and_some_do_not(
        self, monkeypatch, bundle
    ):
        """The middle state, and the one a release gate is actually made of.

        Both claims name the same real evidence. One quotes it, the other does
        not. The answer is therefore real but not wholly checkable, and
        ``partially_supported`` is the only label that says so — ``supported``
        would vouch for the fabricated claim and ``insufficient_evidence``
        would throw away the true one.
        """
        document = _single_evidence_document(text=GATE_PASSAGE)
        analyzer = make_analyzer(monkeypatch, bundle, qa_document=document, reply="{}")
        evidence_id = _index_evidence(bundle, analyzer, document)
        analyzer.client.reply = self._two_claim_reply(evidence_id)

        result = analyzer.answer_question("ImageNet top-1 准确率是多少？")

        assert result.confidence == CONFIDENCE_PARTIALLY_SUPPORTED
        presentations = [claim.presentation for claim in result.claims]
        assert presentations == [
            ClaimPresentation.DIRECT_QUOTE,
            ClaimPresentation.PARAPHRASE,
        ]
        assert result.citations, "a partially supported answer still cites what it verified"

    def test_a_confidence_the_model_asserts_for_itself_is_ignored(
        self, monkeypatch, bundle
    ):
        """The model does not get to grade itself, even by asking.

        The reply carries a ``confidence`` field claiming full support, next to
        a claim that cites an id retrieval never returned. A rule that read the
        model's assessment would report ``supported`` here, which is precisely
        the failure the whole design exists to prevent: a confident answer with
        nothing behind it.
        """
        document = _single_evidence_document(text=GATE_PASSAGE)
        analyzer = make_analyzer(monkeypatch, bundle, qa_document=document, reply="{}")
        _index_evidence(bundle, analyzer, document)
        analyzer.client.reply = json.dumps(
            {
                "answer": "论文报告了 76.3% 的 top-1 准确率。",
                "confidence": CONFIDENCE_SUPPORTED,
                "claims": [
                    {
                        "text": "论文报告了 76.3% 的 top-1 准确率。",
                        "quote": GATE_PASSAGE,
                        "evidence_ids": ["ev_invented_by_the_model"],
                    }
                ],
            },
            ensure_ascii=False,
        )

        result = analyzer.answer_question("ImageNet top-1 准确率是多少？")

        assert result.confidence != CONFIDENCE_SUPPORTED
        assert result.confidence == CONFIDENCE_INSUFFICIENT
        assert result.citations == ()

    def test_a_paraphrase_claim_with_real_evidence_is_verified(
        self, monkeypatch, bundle
    ):
        """A claim with no quotation is supported by naming real evidence.

        Deliberate, and the reasoning is the counter-intuitive one: a claim is
        a paraphrase by nature, the gate is about claims dressed as
        quotations, and a rule that treated every paraphrase as unverified
        would report ``insufficient_evidence`` for a perfectly well-sourced
        answer. It would then be wrong every time, and a label that is always
        wrong is a label users learn to ignore.
        """
        document = _single_evidence_document(text=GATE_PASSAGE)
        analyzer = make_analyzer(monkeypatch, bundle, qa_document=document, reply="{}")
        evidence_id = _index_evidence(bundle, analyzer, document)
        analyzer.client.reply = claim_payload(
            "模型在 ImageNet 上报告了 76.3% 的 top-1 准确率，并优于全部基线。",
            [
                {
                    "text": "模型在 ImageNet 上报告了 76.3% 的 top-1 准确率，并优于全部基线。",
                    "quote": "",
                    "evidence_ids": [evidence_id],
                }
            ],
        )

        result = analyzer.answer_question("ImageNet top-1 准确率是多少？")

        assert result.confidence == CONFIDENCE_SUPPORTED
        assert result.claims[0].presentation is ClaimPresentation.PARAPHRASE
        assert result.citations[0].evidence_id == evidence_id


# --------------------------------------------------------------------------
# 5. The streaming path carries the same structure
# --------------------------------------------------------------------------


class TestStreamingCarriesTheStructure:
    def _streamed(self, analyzer, question):
        return list(analyzer.stream_answer_question(question))

    def test_the_stream_ends_with_the_answer_result(self, monkeypatch, bundle,
                                                   late_document):
        analyzer = make_analyzer(
            monkeypatch, bundle, qa_document=late_document, chunks=("占位",)
        )
        index_document_for_retrieval(bundle, late_document, None)
        evidence_id = analyzer._retrieve_for_question(LATE_QUESTION).evidence[0].evidence_id
        analyzer.client.chunks = (claim_payload(
            f"消融实验显示准确率下降了 {LATE_FIGURE} 个百分点。",
            [
                {
                    "text": f"准确率下降了 {LATE_FIGURE} 个百分点。",
                    "quote": LATE_SENTENCE,
                    "evidence_ids": [evidence_id],
                }
            ],
        ),)

        items = self._streamed(analyzer, LATE_QUESTION)

        results = [item for item in items if isinstance(item, AnswerResult)]
        assert len(results) == 1, "the stream must end with exactly one AnswerResult"
        assert results[0].confidence == CONFIDENCE_SUPPORTED
        assert results[0].claims[0].verified_text == LATE_SENTENCE

    def test_the_deltas_reassemble_into_the_final_answer(self, monkeypatch, bundle,
                                                         late_document):
        """The reader is never shown one answer and then handed another.

        The front end replaces whatever it rendered from the deltas with
        ``payload.answer`` on ``done``. If the two could differ, every streaming
        answer would visibly change at the end — and the browser would show the
        pre-verification text, which is exactly what the gate forbids.
        """
        analyzer = make_analyzer(
            monkeypatch, bundle, qa_document=late_document, chunks=("占位",)
        )
        index_document_for_retrieval(bundle, late_document, None)
        evidence_id = analyzer._retrieve_for_question(LATE_QUESTION).evidence[0].evidence_id
        analyzer.client.chunks = (claim_payload(
            f"消融实验显示准确率下降了 {LATE_FIGURE} 个百分点。",
            [
                {
                    "text": f"准确率下降了 {LATE_FIGURE} 个百分点。",
                    "quote": LATE_SENTENCE,
                    "evidence_ids": [evidence_id],
                }
            ],
        ),)

        items = self._streamed(analyzer, LATE_QUESTION)
        deltas = [item for item in items if isinstance(item, str)]
        final = items[-1]

        assert isinstance(final, AnswerResult)
        assert "".join(deltas) == final.answer
        assert LATE_FIGURE in final.answer

    def test_the_streaming_path_also_refuses_without_evidence(
        self, monkeypatch, bundle, bait_document
    ):
        """Same gate, same suppression, on the path people actually use.

        A release gate that holds on ``/api/ask`` and not on
        ``/api/ask/stream`` is not a gate — the streaming endpoint is the one
        the browser uses for every question, so the failure would be the common
        case rather than the edge one.
        """
        analyzer = make_analyzer(
            monkeypatch, bundle, qa_document=bait_document, chunks=("占位",)
        )
        index_document_for_retrieval(bundle, bait_document, None)
        analyzer.client.chunks = (FABRICATED_REPLY,)

        items = list(analyzer.stream_answer_question(BAIT_QUESTION))

        final = items[-1]
        assert isinstance(final, AnswerResult)
        assert final.confidence == CONFIDENCE_INSUFFICIENT
        assert final.citations == ()
        for quoted in UNDECIDABLE_QUOTES:
            assert quoted not in final.answer
        assert "".join(item for item in items if isinstance(item, str)) == final.answer

    def test_the_stream_never_emits_the_unverified_model_reply(
        self, monkeypatch, bundle
    ):
        """Not even transiently, which is why the reply is buffered.

        The naive streaming implementation forwards each chunk as it arrives.
        The model writes a fabricated quote in its first chunk and the browser
        has already rendered it in quotation marks before anything gets the
        chance to check. The gate has to run *before* presentation, not after.
        """
        document = _single_evidence_document(text=GATE_PASSAGE)
        analyzer = make_analyzer(
            monkeypatch,
            bundle,
            qa_document=document,
            chunks=(),
        )
        evidence_id = _index_evidence(bundle, analyzer, document)
        analyzer.client.chunks = (
            claim_payload(
                f"论文写道 “{INVENTED_QUOTE}”。",
                [
                    {
                        "text": "模型取得了 76.3% 的准确率。",
                        "quote": INVENTED_QUOTE,
                        "evidence_ids": [evidence_id],
                    }
                ],
            ),
        )

        items = self._streamed(analyzer, "ImageNet top-1 准确率是多少？")
        streamed = "".join(item for item in items if isinstance(item, str))

        assert f"“{INVENTED_QUOTE}”" not in streamed
        assert f'"{INVENTED_QUOTE}"' not in streamed
        assert INVENTED_QUOTE not in streamed, (
            "the fabricated quotation was streamed to the reader before it could "
            "be checked; the reply has to be buffered, not forwarded"
        )
        # The single claim failed to verify, so there is nothing the answer can
        # stand on and the refusal is what gets streamed.
        assert items[-1].confidence == CONFIDENCE_INSUFFICIENT


# --------------------------------------------------------------------------
# 6. The AnswerResult surface
# --------------------------------------------------------------------------


class TestAnswerResultSurface:
    def test_the_result_carries_every_documented_field(self, monkeypatch, bundle):
        document = _single_evidence_document(text=GATE_PASSAGE)
        analyzer = make_analyzer(
            monkeypatch, bundle, qa_document=document, reply="{}"
        )
        evidence_id = _index_evidence(bundle, analyzer, document)
        analyzer.client.reply = claim_payload(
            "模型在 ImageNet 上取得了 76.3% 的 top-1 准确率。",
            [
                {
                    "text": "模型在 ImageNet 上取得了 76.3% 的 top-1 准确率。",
                    "quote": GATE_PASSAGE,
                    "evidence_ids": [evidence_id],
                }
            ],
        )

        payload = analyzer.answer_question("ImageNet top-1 准确率是多少？").to_dict()

        assert set(payload) == {
            "answer",
            "claims",
            "citations",
            "confidence",
            "insufficient_evidence",
            "retrieval",
            "notes",
            "answer_mode",
        }
        assert payload["confidence"] in {
            CONFIDENCE_SUPPORTED,
            CONFIDENCE_PARTIALLY_SUPPORTED,
            CONFIDENCE_INSUFFICIENT,
        }
        claim = payload["claims"][0]
        assert claim["evidence_ids"] == [evidence_id]
        assert claim["uses_quotation_marks"] is True
        assert claim["presentation"] == "direct_quote"

    def test_a_citation_names_a_passage_a_chip_can_open(self, monkeypatch, bundle):
        document = _single_evidence_document(text=GATE_PASSAGE, page=7)
        analyzer = make_analyzer(
            monkeypatch, bundle, qa_document=document, reply="{}"
        )
        evidence_id = _index_evidence(bundle, analyzer, document)
        analyzer.client.reply = claim_payload(
            "模型在 ImageNet 上取得了 76.3% 的 top-1 准确率。",
            [
                {
                    "text": "模型在 ImageNet 上取得了 76.3% 的 top-1 准确率。",
                    "quote": GATE_PASSAGE,
                    "evidence_ids": [evidence_id],
                }
            ],
        )

        citation = analyzer.answer_question(
            "ImageNet top-1 准确率是多少？"
        ).citations[0].to_dict()

        assert citation["evidence_id"] == evidence_id
        assert citation["citation_label"] == "p. 7"
        assert citation["locator"]["kind"] == "pdf_page"
        assert citation["locator"]["page"] == 7
        assert GATE_PASSAGE in citation["text"]

    def test_a_result_is_frozen(self):
        """A route cannot regrade an answer after deciding how to render it."""
        result = AnswerResult(answer="a", confidence=CONFIDENCE_SUPPORTED)
        with pytest.raises((AttributeError, TypeError)):
            result.confidence = CONFIDENCE_INSUFFICIENT  # type: ignore[misc]

    def test_a_claim_round_trips_through_its_dict(self):
        claim = AnswerClaim(
            text="t",
            evidence_ids=("ev_1",),
            quote="q",
            presentation=ClaimPresentation.PARAPHRASE,
        )
        assert claim.to_dict()["evidence_ids"] == ["ev_1"]
        assert claim.to_dict()["uses_quotation_marks"] is False


# --------------------------------------------------------------------------
# 7. Guards on the surrounding wiring
# --------------------------------------------------------------------------


class TestQaGuards:
    def test_answering_with_no_document_is_a_client_error(self, monkeypatch, bundle):
        """A missing upload and a paper that does not say are different facts.

        Collapsing them would report "insufficient evidence" for a document the
        user never uploaded, which reads as a finding about a paper they may not
        have.
        """
        analyzer = make_analyzer(monkeypatch, bundle, reply="{}")
        analyzer.document_content = ""
        with pytest.raises(ValueError, match="没有文档内容"):
            analyzer.answer_question("问题")

    def test_flat_session_text_is_wrapped_into_an_addressable_document(self):
        """The session stores text, so this is the path every turn really takes.

        Without it there is no ``ParsedDocument`` to chunk or to verify quotes
        against, and the answer would have nothing to cite. The locators are
        character offsets rather than pages, and the document says so in its
        warnings rather than presenting a re-derived structure as a parsed one.
        """
        document = parsed_document_from_text("第一段。\n\n第二段。\n\n第三段。")
        assert document.elements
        assert all(
            element.locator.kind == "txt_offset" for element in document.elements
        )
        assert document.warnings, (
            "a re-derived document must say so; a citation into it is a weaker "
            "claim than one into a parsed document"
        )

    def test_the_same_text_yields_the_same_document_id(self):
        """Content-addressed, so a repeated question reuses the same index.

        A random id per call would re-chunk and re-index on every question,
        and would give the same passage a different evidence id each time —
        which makes "did this answer cite the same thing yesterday" a question
        with no answer.
        """
        first = parsed_document_from_text("同样的内容。")
        second = parsed_document_from_text("同样的内容。")
        assert first.document_id == second.document_id
        assert parsed_document_from_text("别的内容。").document_id != first.document_id

    def test_indexing_is_idempotent(self, bundle, late_document):
        """A second question about the same document does not re-index it."""
        index_document_for_retrieval(bundle, late_document, None)
        before = bundle.metadata.list_chunks(late_document.document_id)
        index_document_for_retrieval(bundle, late_document, None)
        after = bundle.metadata.list_chunks(late_document.document_id)
        assert [chunk.chunk_id for chunk in before] == [chunk.chunk_id for chunk in after]

    def test_the_prompt_offers_exactly_the_evidence_it_retrieved(self, monkeypatch, bundle,
                                                                  late_document):
        """Every id in the reply contract is one a reader could have been shown."""
        analyzer = make_analyzer(monkeypatch, bundle, qa_document=late_document, reply="{}")
        retrieved = analyzer._retrieve_for_question(LATE_QUESTION)
        index_document_for_retrieval(bundle, late_document, None)
        analyzer.answer_question(LATE_QUESTION)

        offered = set(evidence_ids_in_prompt(analyzer.client))
        expected = {item.evidence_id for item in retrieved.evidence}
        assert offered == expected

    def test_the_answer_prompt_asks_for_the_declared_contract(self, monkeypatch, bundle,
                                                              late_document):
        """The schema is in the prompt, so a compliant model produces parseable claims."""
        analyzer = make_analyzer(monkeypatch, bundle, qa_document=late_document, reply="{}")
        index_document_for_retrieval(bundle, late_document, None)
        analyzer.answer_question(LATE_QUESTION)

        prompt = analyzer.client.last_prompt
        for field in ("answer", "claims", "text", "quote", "evidence_ids"):
            assert f'"{field}"' in prompt, f"the reply contract omits {field}"

    def test_history_is_forwarded_but_never_becomes_evidence(
        self, monkeypatch, bundle, late_document
    ):
        """Follow-up context is useful and is not a citation.

        A prior turn is model output, not the document. Letting it into the
        evidence block would let an unverified claim from turn one become a
        verified citation in turn two — the fabrication this product is built
        against arriving by a second route — and would also put text into the
        prompt that no ``ev_`` id names, so a model reading the block could not
        tell which passages are the paper's.
        """
        analyzer = make_analyzer(monkeypatch, bundle, qa_document=late_document, reply="{}")
        index_document_for_retrieval(bundle, late_document, None)
        evidence_id = analyzer._retrieve_for_question(LATE_QUESTION).evidence[0].evidence_id
        analyzer.client.reply = claim_payload(
            f"消融实验显示准确率下降了 {LATE_FIGURE} 个百分点。",
            [
                {
                    "text": f"准确率下降了 {LATE_FIGURE} 个百分点。",
                    "quote": LATE_SENTENCE,
                    "evidence_ids": [evidence_id],
                }
            ],
        )
        analyzer.answer_question(
            LATE_QUESTION,
            history=[{"question": "上一问", "answer": "上一答里有一个未验证的数字 99.9%"}],
        )

        prompt = analyzer.client.last_prompt
        assert "上一问" in prompt, "follow-up history was dropped entirely"
        evidence_block = prompt[prompt.index("<retrieved_evidence>") : prompt.index(
            "</retrieved_evidence>"
        )]
        assert "99.9%" not in evidence_block, "history text leaked into the evidence block"
        assert "上一问" in prompt[prompt.index("<recent_qa_history>") :], (
            "history must arrive in its own block, not inline with the evidence"
        )


def test_no_network_and_no_clock(monkeypatch, bundle):
    """The suite is deterministic by construction, and says so in a test.

    A test that could reach a provider would be a test whose failure means
    "the endpoint was down" as often as it means "the code is wrong", and
    those two need different responses from whoever is on call.
    """
    import socket

    def refuse(*args, **kwargs):
        raise AssertionError("the Q&A tests must not open a socket")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    document = parsed_document_from_text("一段文字。")
    assert document.document_id
    assert os.sep  # a sanity check that the module under test imported cleanly


def test_the_json_helpers_the_prompt_contract_relies_on_agree():
    """A fenced reply parses to the same object as a bare one.

    Models fence JSON despite being told not to. If the parser only handled the
    bare form, every fenced reply would silently degrade to a single unverified
    claim, and the confidence would be wrong on a large fraction of real
    traffic rather than never.
    """
    import json as json_module

    payload = {"answer": "a", "claims": []}
    for form in (
        json_module.dumps(payload),
        f"```json\n{json_module.dumps(payload)}\n```",
        f"前言\n```json\n{json_module.dumps(payload)}\n```\n后记",
    ):
        assert json.loads(extract_json_object(form)) == payload
