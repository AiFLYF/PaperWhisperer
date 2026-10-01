"""The correlation context, the six families, and the trace one document produces.

Task T4.1's acceptance criterion is a sentence, not a dashboard: **任一失败可从
document_id/job_id 追到 parser/fallback/LLM/外部 API** — any failure must be
reachable from a document or job id, through the parser, the fallback, the
LLM and the external API. So this file is arranged around that sentence rather
than around the module's surface, and the test names are the questions a person
would ask at 3am:

``TestTheIdsSurviveTheHop``
    Does the context survive a thread and an async boundary? Not "does
    contextvars work" -- does *this* context, in *this* codebase's two hop
    mechanisms, and is the one place where it does **not** survive (a bare
    ``executor.submit``) proven rather than asserted in a docstring.
``TestTheTraceIsAStory``
    One document that parses, degrades, escalates, indexes, retrieves, answers
    and costs -- traced end to end, in order, and read back in Chinese. This
    is the acceptance criterion, executed against the real
    ``ParseOutcome``, the real ``RetrievalRecord``, the real
    ``verify_quote`` and the real ``Job``.
``TestItIsOffUntilAskedFor``
    The hazard this repository already has one of: a test that leaves the root
    logger or the process-wide record factory configured poisons every test
    after it. Asserted from *both* directions -- nothing happens before
    ``configure``, and the process is byte-for-byte what it was after
    ``shutdown``.

The tests drive real types from the real modules on purpose. A
``SimpleNamespace(document_id="doc_1")`` would make every assertion in this
file pass while the actual ``ParseOutcome`` (which has no ``document_id`` of
its own) recorded nothing -- which is a bug this module had, and which a fake
object could not have caught.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import io
import json
import logging
import threading

import pytest

from paperwhisperer.analysis.paper_facts import ModelCall
from paperwhisperer.core import observability as O
from paperwhisperer.infra.jobs import (
    STATE_COMPLETED,
    STATE_INDEXING,
    STATE_PARSING,
    Job,
    JobEvent,
)
from paperwhisperer.ingestion.gateway import PARSE_SUCCESS
from paperwhisperer.ingestion.orchestrator import (
    ATTEMPT_SKIPPED,
    STAGE_CLOUD,
    STAGE_FORCE_OCR,
    STAGE_PRIMARY,
    ParseAttempt,
    ParseOutcome,
)
from paperwhisperer.ingestion.quality import ParseQualityReport, QualityStatus
from paperwhisperer.ingestion.schema import Element, ParsedDocument, SourceLocator
from paperwhisperer.rag.citations import verify_quote
from paperwhisperer.rag.retrieve import CandidateRecord, LegHit, RetrievalRecord

DOC = "doc_7f3a1c9e2b4d"
JOB = "job_1a2b3c4d5e6f"
RUN = "run_c4d5e6f7a8b9"


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def obs():
    """An enabled facade that is not the process singleton.

    Own instance rather than the module-level one on purpose: the singleton's
    state is process-global, and a test asserting on counters it just
    incremented would pass on a dirty registry and fail on a clean one, with
    the failure depending on test order. The one test that genuinely needs the
    singleton says so in its name.
    """
    return O.Observability(enabled=True, trace_limit=256)


@pytest.fixture(autouse=True)
def _no_leaked_observability():
    """Leave the singleton, the record factory and the package logger untouched.

    The repository already has this hazard once (the autouse fixtures in
    ``conftest.py`` exist because ``config`` is read by ``getattr`` from every
    depth of the codebase). The logging equivalent is worse, because
    ``logging.setLogRecordFactory`` and a handler on the ``paperwhisperer``
    logger are both process-global and neither is visible in a traceback: a
    test that installs them and does not remove them makes every later test
    emit duplicate lines with ids stamped on them, and the *later* tests are the
    ones that get blamed.

    The teardown asserts the restoration rather than assuming it, because a
    teardown that assumes is the thing that breaks silently.
    """
    before_factory = logging.getLogRecordFactory()
    before_handlers = list(logging.getLogger("paperwhisperer").handlers)
    before_level = logging.getLogger("paperwhisperer").level
    yield
    O.shutdown()
    assert logging.getLogRecordFactory() is before_factory, "record factory leaked"
    assert logging.getLogger("paperwhisperer").handlers == before_handlers, "handler leaked"
    assert logging.getLogger("paperwhisperer").level == before_level, "logger level leaked"


# --------------------------------------------------------------------------
# Builders over the real types
# --------------------------------------------------------------------------


def _document(document_id: str = DOC) -> ParsedDocument:
    element = Element(
        role="text",
        text="We evaluate on ImageNet-1K and report top-1 accuracy of 76.8%.",
        locator=SourceLocator(kind="pdf_page", page=7),
        order=7,
    )
    document = ParsedDocument(
        document_id=document_id,
        source_filename="resnet.pdf",
        media_type="application/pdf",
        page_count=12,
        elements=[element],
        parser_name="mineru_cloud",
        parser_version="v1",
    )
    document.build_units()
    return document


def _scanned_pdf_outcome(document: ParsedDocument | None = None) -> ParseOutcome:
    """A PDF that defeated docling, was repaired by the cloud, and degraded.

    Every field is one the real orchestrator would have set: a failed primary,
    a force-OCR stage that was *considered and declined* rather than absent, a
    network-touching cloud attempt that won, and a WARN quality report. This is
    the run the acceptance criterion is about, and the interesting part is that
    it has a failure in it -- which is why it is used for the failure-reachability
    assertions rather than a clean parse.
    """
    return ParseOutcome(
        status=PARSE_SUCCESS,
        document=document if document is not None else _document(),
        quality=ParseQualityReport(
            document_id=document.document_id if document else DOC,
            parser_name="mineru_cloud",
            status=QualityStatus.WARN,
            evaluated_rules=("page_coverage", "min_chars_per_page"),
        ),
        attempts=[
            ParseAttempt(
                stage=STAGE_PRIMARY,
                parser_name="docling",
                parser_version="2.4",
                status="failed",
                attempted=True,
                error_message="文本层为空（扫描件）",
                duration_ms=8120,
            ),
            ParseAttempt(
                stage=STAGE_FORCE_OCR,
                status=ATTEMPT_SKIPPED,
                skipped_reason="ocr_already_enabled",
            ),
            ParseAttempt(
                stage=STAGE_CLOUD,
                parser_name="mineru_cloud",
                parser_version="v1",
                status=PARSE_SUCCESS,
                attempted=True,
                network=True,
                quality_status="warn",
                duration_ms=96400,
            ),
        ],
        fallback_used=True,
        cloud_attempted=True,
        used_cloud=True,
        sha256="c0ffee" * 8,
        media_type="application/pdf",
    )


def _retrieval_record(document_id: str = DOC) -> RetrievalRecord:
    """A retrieval where the dense leg abstained, so the lexical leg answered.

    The degradation string is verbatim in shape from
    :meth:`HybridRetriever._dense_leg`, because the leg-abstention counter reads
    it by prefix and a paraphrased string would silently stop matching.
    """
    return RetrievalRecord(
        record_id="ret_5510",
        query="top-1 accuracy ImageNet-1K",
        task="experiment",
        plan_signature="sig_9a1",
        plan={},
        rrf_k=60,
        leg_weights={"dense": 1.0, "lexical": 1.0, "fuzzy": 0.6},
        candidate_depth=40,
        candidates=(
            CandidateRecord(
                chunk_id="chunk_1", document_id=document_id, rrf_score=0.0325,
                rank=1, legs=(LegHit("lexical", 1, 0.0164),), selected=True,
            ),
            CandidateRecord(
                chunk_id="chunk_2", document_id=document_id, rrf_score=0.0325,
                rank=2, legs=(LegHit("dense", 1, 0.0163),), selected=True,
            ),
            CandidateRecord(
                chunk_id="chunk_9", document_id=document_id, rrf_score=0.0164,
                rank=3, reason="below_final_k",
            ),
        ),
        degradations=(
            "dense 通道前 10 个候选的余弦相似度跨度仅 0.0004（小于 0.02），"
            "对本次 query 没有区分力，该通道弃权。",
        ),
    )


def _job(document_id: str = DOC) -> Job:
    return Job(
        job_id=JOB, kind="document_analysis", state=STATE_PARSING,
        document_id=document_id, attempts=1,
    )


def _full_run(obs: O.Observability, document_id: str = DOC) -> ParseOutcome:
    """Drive every family's instrument from the real types, in chain order.

    The one place the run is assembled, because the test that asserts the order
    of the chain and the test that asserts the counters must be looking at the
    *same* run: a second assembly would be free to differ from the first in
    exactly the field the ordering assertion depends on.
    """
    outcome = _scanned_pdf_outcome(_document(document_id))
    with obs.bind(request_id="req_9a8b", document_id=document_id, job_id=JOB):
        obs.record_parse(outcome, duration_ms=104520.0)

        job = _job(document_id)
        for earlier, later in (
            (STATE_PARSING, "quality_check"),
            ("quality_check", "fallback"),
            ("fallback", STATE_INDEXING),
            (STATE_INDEXING, STATE_COMPLETED),
        ):
            obs.record_job_transition(
                job,
                event=JobEvent(
                    job_id=JOB, from_state=earlier, to_state=later, note="step_enter"
                ),
            )

        obs.record_retrieval(_retrieval_record(document_id), duration_ms=143.0)

        element = _document(document_id).elements[0]
        obs.record_citation_verification(verify_quote("top-1 accuracy of 76.8%", element))
        obs.record_citation_verification(
            verify_quote("we achieve 76.8% top-1 on ImageNet", element)
        )

        obs.record_model_call(
            ModelCall(
                stage="extraction", model="gpt-4o-mini", prompt_version="facts-v1-9ab2",
                input_tokens=18420, output_tokens=2310, cost=None, temperature=0.2,
                temperature_applied=True,
            ),
            duration_ms=21400.0,
        )
        obs.record_cost(
            source="llm", cost=None, document_id=document_id, reason="no_price_configured"
        )
        obs.record_claim_verdicts((_Claim(),), confidence="insufficient_evidence")
    return outcome


class _Claim:
    """The shape ``analysis.orchestrator.AnswerClaim`` presents to a counter.

    A two-field stand-in, because ``record_claim_verdicts`` reads exactly two
    things off a claim -- whether it carries evidence and how it presents -- and
    importing the real dataclass would drag the whole analysis package into a
    test about correlation. What is *not* faked is the value those two fields
    take: ``paraphrase`` is the string
    :class:`rag.citations.ClaimPresentation.PARAPHRASE` actually carries.
    """

    evidence_ids = ()
    presentation = "paraphrase"
    quote = "a fluent sentence that is not in the paper"


# --------------------------------------------------------------------------
# The correlation context
# --------------------------------------------------------------------------


class TestTheIdsSurviveTheHop:
    """The property the whole module is named after, in this codebase's hops."""

    def test_a_thread_hop_keeps_all_four_ids(self, obs):
        """The case the module exists for: a ThreadPoolExecutor loses them.

        ``executor.submit`` runs its callable in a fresh
        :mod:`contextvars` context. This test pins the *fixed* path, and
        :func:`Observability.submit` is what the fix is -- there is no way to
        make a bare ``submit`` behave, and pretending otherwise would be the
        one thing that could make this module a lie.
        """
        captured: list[dict] = []
        with obs.bind(request_id="req_1", job_id="job_1", document_id="doc_1",
                      analysis_run_id="run_1"), \
                concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = obs.submit(pool, lambda: captured.append(O.current_ids().as_dict()))
            future.result(timeout=5)
        assert captured == [{
            "request_id": "req_1", "job_id": "job_1",
            "document_id": "doc_1", "analysis_run_id": "run_1",
        }]

    def test_a_bare_submit_loses_them_which_is_why_submit_exists(self, obs):
        """The hazard, demonstrated rather than asserted in a docstring.

        A test that proves the problem exists is what stops someone from
        "simplifying" :meth:`Observability.submit` back into a plain
        ``executor.submit`` -- a change that would look correct, pass every
        other test in this file, and silently orphan every long job's trace.
        """
        captured: list[dict] = []
        with obs.bind(document_id="doc_1"), \
                concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(lambda: captured.append(O.current_ids().as_dict())).result(timeout=5)
        assert captured == [{}], "a bare submit is expected to drop the context"

    def test_an_async_hop_keeps_them_without_any_help(self, obs):
        """``await`` and ``create_task`` copy the context on their own.

        Worth a test even though it needs nothing from this module, because it
        is the asymmetry that makes :meth:`Observability.submit` necessary: the
        async half of this codebase is already correlated for free, so the bug
        is *only* on the thread side, and knowing which half is broken is what
        tells you where to look.
        """
        seen: list[dict] = []

        async def leaf():
            seen.append(O.current_ids().as_dict())
            await asyncio.sleep(0)
            seen.append(O.current_ids().as_dict())

        async def root():
            with obs.bind(document_id="doc_1", job_id="job_1"):
                await leaf()
                await asyncio.gather(asyncio.create_task(leaf()))

        with obs.bind(document_id="doc_1", job_id="job_1"):
            asyncio.run(root())
        # Four readings: ``leaf`` runs twice and samples before and after its
        # own ``await``. All four identical is the claim -- a task boundary did
        # not quietly reset anything.
        assert seen == [{"document_id": "doc_1", "job_id": "job_1"}] * 4

    def test_two_threads_dont_share_a_correlation(self, obs):
        """A context is copied per call, not shared between concurrent ones.

        The bug this prevents is silent and expensive: two documents analysed
        at once, one of them stamped with the other's id, and a trace query by
        ``document_id`` returning confidently wrong events. The cost of the copy
        is a fraction of a microsecond; the cost of sharing is a support ticket
        that cannot be reproduced.
        """
        found: dict[str, dict] = {}
        barrier = threading.Barrier(2)

        def worker(document_id: str) -> None:
            with obs.bind(document_id=document_id):
                barrier.wait(timeout=5)  # hold the scope open in both threads
                found[document_id] = O.current_ids().as_dict()

        threads = [threading.Thread(target=worker, args=(name,))
                   for name in ("doc_a", "doc_b")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        assert found == {"doc_a": {"document_id": "doc_a"},
                         "doc_b": {"document_id": "doc_b"}}

    def test_a_nested_scope_inherits_and_restores(self, obs):
        with obs.bind(request_id="req_1", document_id="doc_1"):
            with obs.bind(job_id="job_1"):
                assert O.current_ids().as_dict() == {
                    "request_id": "req_1", "job_id": "job_1", "document_id": "doc_1",
                }
            # The inner scope is gone and the outer one is intact -- the thing
            # a token-based API gets wrong when a route returns early.
            assert O.current_ids().as_dict() == {"request_id": "req_1", "document_id": "doc_1"}
        assert O.current_ids().as_dict() == {}

    def test_an_id_can_be_cleared_which_is_not_the_same_as_omitted(self, obs):
        """A worker must be able to drop the request it was claimed under.

        The real case: ``JobWorker.tick`` runs on its own thread, but a worker
        started inside a request's lifetime (a test, a first-run bootstrap, a
        lifespan hook inside a request context) would otherwise spend its life
        writing lines attributed to whichever upload happened to be on the
        stack. ``None`` clears; ``""`` inherits. Conflating them would make the
        common case impossible to spell.
        """
        with obs.bind(request_id="req_1", document_id="doc_1"), obs.bind(request_id=None):
            assert O.current_ids().as_dict() == {"document_id": "doc_1"}

    def test_a_scope_restores_even_when_the_body_raises(self, obs):
        """An exception unwinds the ``finally``; the scope does not leak.

        The inner scope is a separate function so the nesting stays one level
        deep: an exception must leave the *outer* scope intact and visible to
        whatever runs next, which is the whole reason :func:`correlate` is a
        context manager rather than a token a caller has to remember to reset.
        """

        def boom():
            with obs.bind(job_id="job_1"):
                raise RuntimeError("the parse blew up")

        with obs.bind(document_id="doc_1"):
            with pytest.raises(RuntimeError):
                boom()
            assert O.current_ids().as_dict() == {"document_id": "doc_1"}
        assert O.current_ids().as_dict() == {}


class TestResolvingIds:
    def test_the_object_beats_the_context_and_the_context_beats_nothing(self, obs):
        """The composition rule, which is the part that is easy to get wrong.

        ``ParseOutcome`` has no ``document_id``; its ``document`` does. Reading
        only the outcome yields an empty id, and passing that empty value as
        the event's ids would *overwrite* the scope's -- landing a parse event in
        no chain at all while every metric still looked correct.
        """
        with obs.bind(document_id="doc_scope", job_id="job_scope"):
            from_object = O.resolve_ids(_scanned_pdf_outcome())
        assert from_object.document_id == DOC
        assert from_object.job_id == "job_scope"
        # An explicit argument beats both: it is a deliberate statement.
        assert O.resolve_ids(_scanned_pdf_outcome(), document_id="doc_explicit").document_id \
            == "doc_explicit"

    def test_analysis_run_id_is_mapped_from_run_id(self):
        """``AnalysisRun.run_id`` is the ``analysis_run_id``, mapped in one place.

        A call site that guessed would get it wrong invisibly, because the
        field would simply be empty and a trace by analysis run would be empty
        for runs that certainly happened.
        """
        run = ModelCall(stage="extraction", model="m", prompt_version="v",
                        input_tokens=1, output_tokens=1, cost=None, temperature=None,
                        temperature_applied=True)
        assert O.extract_ids(run).analysis_run_id == ""
        assert O.extract_ids({"run_id": RUN}).analysis_run_id == RUN

    def test_a_source_with_no_ids_yields_empty_rather_than_raising(self):
        assert O.extract_ids(object()).is_empty
        assert O.extract_ids(None).is_empty


# --------------------------------------------------------------------------
# The trace
# --------------------------------------------------------------------------


class TestTheTraceIsAStory:
    """The acceptance criterion, executed."""

    def test_one_document_traces_parser_fallback_retrieval_llm_and_cost_in_order(
        self, obs
    ):
        """The whole sentence of the acceptance criterion, in one assertion.

        Ordered by ``seq`` rather than by timestamp, because a same-millisecond
        pair ordering by clock is exactly the probable answer this criterion
        says is not good enough.
        """
        _full_run(obs)
        events = obs.trace.events(document_id=DOC)
        names = [event.name for event in events]

        assert events, "a document that ran produced no trace"
        assert names.index("parse.attempt.primary") < names.index("parse.attempt.cloud")
        assert names.index("parse.attempt.cloud") < names.index("retrieval.finished")
        assert names.index("retrieval.finished") < names.index("llm.call")
        assert names.index("llm.call") < names.index("grounding.answer_verdict")
        assert [event.seq for event in events] == sorted(event.seq for event in events)
        assert {event.family for event in events} >= {
            O.FAMILY_PARSING, O.FAMILY_RAG, O.FAMILY_LLM,
            O.FAMILY_GROUNDING, O.FAMILY_JOBS, O.FAMILY_COST,
        }, "六个指标族都应在这条链路上留下事件"

    def test_the_escalation_to_the_cloud_is_named_not_just_counted(self, obs):
        """Why the document left the machine, in one readable event.

        A counter that says "one cloud parse happened" cannot answer the
        question a user asks, which is *was this necessary*. The escalation
        event names the stage it came from, and the trace is what a reader of
        the trace will look for first.
        """
        _full_run(obs)
        escalated = obs.trace.events(document_id=DOC, name="parse.escalated")
        assert len(escalated) == 1
        assert escalated[0].fields["reason"] == "primary→cloud"

    def test_the_failed_primary_is_reachable_from_the_document_id(self, obs):
        """任一失败可从 document_id 追到 parser.

        The failure is the *docling attempt*, and it is the one event a support
        reply needs: "your scan had no text layer, so we paid a cloud parser."
        Without this the criterion is satisfied only for failures that happened
        at the end of a run.
        """
        _full_run(obs)
        failures = obs.trace.failures(document_id=DOC)
        assert failures, "a parse that failed a stage recorded no failure"
        failure = failures[0]
        assert failure.name == "parse.attempt.primary"
        assert failure.fields["parser"] == "docling"
        assert failure.fields["status"] == "failed"
        assert "扫描件" in failure.error
        assert failure.ids.document_id == DOC

    def test_a_stage_that_never_ran_is_still_recorded(self, obs):
        """"The repair was considered and declined" is a fact about the document.

        The force-OCR stage was skipped, and an absent event would be
        indistinguishable from an orchestration bug. This is the same reason
        ``ParseOutcome`` records skipped stages at all, read from the other
        side.
        """
        _full_run(obs)
        skipped = obs.trace.events(document_id=DOC, name="parse.stage_skipped.force_ocr")
        assert len(skipped) == 1
        assert skipped[0].fields["reason"] == "ocr_already_enabled"

    def test_another_documents_events_never_leak_into_this_one(self, obs):
        """A query that returned everything would look like it worked.

        This is the filter semantics that matter most: matching requires the
        event to *carry* the id, so the events of a second document are not
        "close enough" to appear. Asserted with two real runs, because a
        one-document fixture cannot tell a filter from a no-op.
        """
        _full_run(obs, document_id=DOC)
        _full_run(obs, document_id="doc_other")
        mine = {event.name for event in obs.trace.events(document_id=DOC)}
        theirs = {event.name for event in obs.trace.events(document_id="doc_other")}
        assert mine == theirs, "两条链路的形状相同，泄漏只能通过 id 区分"
        assert len(obs.trace.events(document_id=DOC)) == len(
            obs.trace.events(document_id="doc_other")
        )
        assert not any(
            event.ids.document_id == "doc_other"
            for event in obs.trace.events(document_id=DOC)
        )

    def test_the_narrative_is_readable_and_names_the_whole_story(self, obs):
        """``explain`` is for the person holding the complaint, not for a machine.

        The metrics answer "how often"; this answers "what happened to *this*
        one", which is the question the acceptance criterion is actually about.
        """
        _full_run(obs)
        text = obs.trace.explain(document_id=DOC)
        assert DOC in text
        assert "docling" in text and "mineru_cloud" in text
        assert "文本层为空" in text, "失败的原因必须在叙述里，不只在计数器里"
        assert "retrieval.finished" in text and "llm.call" in text
        assert text.count("\n") >= 10

    def test_an_unknown_id_gets_a_useful_answer_not_a_stack_trace(self, obs):
        _full_run(obs)
        text = obs.trace.explain(document_id="doc_never_existed")
        assert "未找到" in text
        assert "configure()" in text, "空结果最可能的解释是没启用，应当被说出来"

    def test_the_trace_is_bounded_and_the_drops_are_counted(self, obs):
        """A ring buffer, because a personal-tier process is long-lived.

        An unbounded event list is a slow leak whose size is proportional to how
        long the app has been up, discovered on the day it takes the machine
        down. The bound is what makes the memory knowable; counting the drops
        is what stops a trace that lost events from answering "no such event".
        """
        small = O.Observability(enabled=True, trace_limit=8)
        for index in range(20):
            small.emit(f"event_{index}", family=O.FAMILY_PARSING, document_id="doc_1")
        assert len(small.trace) == 8
        assert small.trace.dropped == 12
        names = [event.name for event in small.trace.events(document_id="doc_1")]
        assert "event_0" not in names, "最旧的被丢弃，最新的保留"
        assert "event_19" in names

    def test_an_event_round_trips_through_json(self, obs):
        """A record that cannot be shipped cannot be queried by a collector.

        The Hosted tier's seam hands these to an exporter, and the exporter
        writes JSON lines; a field that only existed in Python would be the one
        somebody needed in production.
        """
        _full_run(obs)
        event = obs.trace.events(document_id=DOC, name="parse.finished")[0]
        restored = json.loads(json.dumps(event.to_dict(), ensure_ascii=False, default=str))
        assert restored["document_id"] == DOC
        assert restored["family"] == O.FAMILY_PARSING
        assert restored["fields"]["parser"] == "mineru_cloud"
        assert restored["fields"]["used_cloud"] is True
        assert restored["duration_ms"] == pytest.approx(104520.0)

    def test_an_arbitrary_object_in_a_field_does_not_break_the_event(self, obs):
        """Lossiness is deliberate: losing a field beats losing the event.

        A caller who passes a whole ``ParsedDocument`` as a field value should
        get its ``repr`` truncated, not a ``TypeError`` in the middle of a
        parse's error path.
        """
        event = obs.trace.record("t", family=O.FAMILY_PARSING, payload=_document())
        assert isinstance(event.fields["payload"], str)
        assert len(event.fields["payload"]) <= 200
        json.dumps(event.to_dict(), default=str)


# --------------------------------------------------------------------------
# The six families
# --------------------------------------------------------------------------


class TestTheSixFamiliesCountWhatTheyClaim:
    def test_a_full_run_produces_the_right_numbers(self, obs):
        """One assertion over the whole counter set, after a real run.

        Written as a table rather than a dozen asserts so that a regression
        shows up as *which* number moved. These are exact values, not ranges:
        the point of a counter is that a reader can predict it, and a test that
        only checked "something was counted" would have caught the two bugs
        that made ``+=`` rebind a local.
        """
        _full_run(obs)
        metrics = obs.metrics
        assert {
            "parsing.documents": metrics.parsing_documents.total,
            "parsing.cloud_egress": metrics.parsing_cloud_egress.total,
            "parsing.fallbacks": metrics.parsing_fallbacks.total,
            "parsing.cache.miss": metrics.parsing_cache.get(result="miss"),
            "parsing.quality.warn": metrics.parsing_quality.get(status="warn"),
            "rag.retrievals": metrics.rag_retrievals.total,
            "rag.legs_abstained": metrics.rag_legs_abstained.total,
            "rag.degradations": metrics.rag_degradations.get(),
            "rag.candidates.selected": metrics.rag_candidates.get(
                task="experiment", selected="true"),
            "rag.candidates.dropped": metrics.rag_candidates.get(
                task="experiment", selected="false", reason="below_final_k"),
            "llm.calls": metrics.llm_calls.total,
            "llm.input_tokens": metrics.llm_tokens.get(direction="input"),
            "llm.output_tokens": metrics.llm_tokens.get(direction="output"),
            "grounding.verified_quotes": metrics.grounding_verified_quotes.total,
            "grounding.degraded": metrics.grounding_degraded_claims.total,
            "grounding.unsupported": metrics.grounding_unsupported_claims.get(
                reason="answer_claim"),
            "grounding.answers": metrics.grounding_answers.total,
            "grounding.insufficient": metrics.grounding_insufficient.total,
            "cost.unpriced": metrics.cost_unpriced.total,
            "jobs.transitions": metrics.jobs_transitions.total,
            "jobs.terminal.completed": metrics.jobs_terminal.get(state=STATE_COMPLETED),
        } == {
            "parsing.documents": 1,
            # One: the answer came from the cloud, so the document left the
            # machine. This is the number a privacy question is answered with.
            "parsing.cloud_egress": 1,
            # One escalation: primary -> cloud. The force-OCR stage was
            # considered and declined, which is not an escalation.
            "parsing.fallbacks": 1,
            "parsing.cache.miss": 1,
            "parsing.quality.warn": 1,
            "rag.retrievals": 1,
            "rag.legs_abstained": 1,
            "rag.degradations": 1,
            "rag.candidates.selected": 2,
            "rag.candidates.dropped": 1,
            "llm.calls": 1,
            "llm.input_tokens": 18420,
            "llm.output_tokens": 2310,
            # One quote aligned, one did not and was demoted. The product's
            # central promise, counted for the first time.
            "grounding.verified_quotes": 1,
            # One, from the demoted quote. The answer's own claims add zero
            # here: the claim that resolved to no evidence at all is counted
            # as *unsupported* instead, because "the reader cannot be sent
            # anywhere" and "the reader is sent somewhere without quotation
            # marks" are different failures with different fixes.
            "grounding.degraded": 1,
            "grounding.unsupported": 1,
            "grounding.answers": 1,
            "grounding.insufficient": 1,
            # Two: the cloud parse (unpriced) and the LLM call (no price
            # configured). Neither is free; both are recorded as unknown.
            "cost.unpriced": 2,
            "jobs.transitions": 4,
            "jobs.terminal.completed": 1,
        }

    def test_cloud_attempted_and_used_cloud_are_counted_separately(self, obs):
        """A paid parse that lost is money spent, and must not read as egress.

        ``used_cloud`` means the answer came from a third party; ``cloud_
        attempted`` means a third party was paid. Collapsing them would either
        over-report privacy incidents or under-report the bill, and both
        numbers are in the trace for exactly that reason.
        """
        document = _document()
        outcome = _scanned_pdf_outcome(document)
        outcome.used_cloud = False      # cloud was called, and the local chain won
        obs.record_parse(outcome)
        assert obs.metrics.parsing_cloud_egress.total == 0, "本地胜出不是出网"
        assert obs.metrics.cost_unpriced.get(source="cloud_parse") == 1, "但钱花了"

    def test_grounding_counts_the_verdicts_and_nothing_else_decides_them(self, obs):
        """The family the plan names and the codebase had no counter for.

        Driven by the real :func:`rag.citations.verify_quote`, so the counts are
        a property of the verifier's real verdicts and not of a hand-picked
        enum value.
        """
        element = _document().elements[0]
        verdicts = {
            "aligned": verify_quote("top-1 accuracy of 76.8%", element),
            "exact but retyped": verify_quote("We evaluate on ImageNet-1K", element),
            "invented": verify_quote("this paper reports 92.4% mAP", element),
            "empty": verify_quote("", element),
        }
        for verdict in verdicts.values():
            obs.record_citation_verification(verdict)

        metrics = obs.metrics
        assert metrics.grounding_verified_quotes.total == 2
        assert metrics.grounding_quote_alignment.get(outcome="aligned") == 2
        assert metrics.grounding_quote_alignment.get(outcome="empty_quote") == 1
        assert metrics.grounding_quote_alignment.get(outcome="not_found") == 1
        # The invented one could not be presented as a quote, so it is counted
        # as demoted rather than silently dropped.
        assert metrics.grounding_degraded_claims.total == 2

    def test_a_fact_layer_run_counts_its_unverified_fields(self, obs):
        """A document that parsed perfectly and produced nothing verified.

        This is the run a dashboard must not report as a success, and it is
        only visible if the family's counters exist at all.
        """
        from paperwhisperer.analysis.paper_facts import ExtractionReport

        report = ExtractionReport(
            parse_failed=False,
            dropped_evidence_ids=("ev_does_not_exist",),
            unverified_fields=("reproducibility",),
            not_stated_fields=("datasets",),
        )
        obs.record_fact_extraction(report, run=None)
        assert obs.metrics.grounding_facts.get(status="needs_review") == 1
        assert obs.metrics.grounding_facts.get(status="not_found", field="datasets") == 1
        assert obs.metrics.grounding_unsupported_claims.get(
            reason="unresolvable_citation") == 1
        assert obs.trace.events(name="grounding.fact_extraction")[0].level == "WARNING"

    def test_a_cost_that_cannot_be_priced_is_never_recorded_as_zero(self, obs):
        """Unknown and free are different facts, and the ledger must keep them apart.

        ``AnalysisRun.llm_cost`` returns ``None`` when any call's price is
        unknown precisely so that a partial sum does not look like a total. A
        gauge that folded ``None`` into ``0.0`` would re-introduce that lie, and
        a budget built on it would under-spend.
        """
        obs.record_cost(source="llm", cost=None, reason="no_price_configured")
        assert obs.metrics.cost_total.total == 0.0
        assert obs.metrics.cost_unpriced.get(
            source="llm", reason="no_price_configured") == 1
        event = obs.trace.events(name="cost.unpriced")[0]
        assert event.level == "WARNING"
        assert event.fields["reason"] == "no_price_configured"

        obs.record_cost(source="llm", cost=0.25, document_id=DOC)
        assert obs.metrics.cost_llm.total == pytest.approx(0.25)
        assert obs.metrics.cost_documents.get(document_id=DOC) == pytest.approx(0.25)

    def test_job_queue_depth_is_a_census_not_an_accumulation(self, obs):
        """A gauge is overwritten by a census, never accumulated into.

        The store's ``count_by_state`` is a snapshot. Summing snapshots produces
        a number that only grows and means nothing -- and a queue panel that
        shows a steadily rising backlog is precisely the alert T4.6 needs to be
        able to trust.
        """
        obs.record_job_queue({"queued": 3, "parsing": 1, "completed": 9})
        assert obs.metrics.jobs_depth.get(state="queued") == 3
        obs.record_job_queue({"queued": 0, "parsing": 1, "completed": 9})
        assert obs.metrics.jobs_depth.get(state="queued") == 0, "普查要覆盖，不是累加"
        assert obs.metrics.jobs_depth.get(state="completed") == 9

    def test_a_recovered_lease_is_loud(self, obs):
        """``recover_stale`` moving a job means somebody died.

        It is the loudest number in the jobs family and it is on its own
        counter, because "somebody died" buried in a transition histogram is a
        number nobody reads and a bug that is exactly the one the release gate
        names.
        """
        obs.record_job_transition(
            _job(),
            event=JobEvent(job_id=JOB, from_state="parsing", to_state="queued",
                           note="lease_expired"),
        )
        assert obs.metrics.jobs_recovered.get(note="lease_expired") == 1

    def test_a_failed_job_is_an_error_event(self, obs):
        obs.record_job_transition(
            _job(),
            event=JobEvent(job_id=JOB, from_state="analyzing", to_state="failed",
                           note="step_failed", attempt=2),
        )
        event = obs.trace.events(name="job.transition")[0]
        assert event.level == "ERROR"
        assert event.fields["status"] == "failed"
        assert event.fields["attempt"] == 2
        assert obs.trace.failures(job_id=JOB)


class TestTheMetricKindsBehave:
    def test_a_counter_sums_and_a_gauge_moves_down(self, obs):
        counter = obs.metrics.parsing_documents
        counter.inc(status="ok", parser="a")
        counter.inc(2, status="ok", parser="a")
        assert counter.get(status="ok", parser="a") == 3
        assert counter.total == 3

        gauge = obs.metrics.jobs_depth
        gauge.set(5, state="queued")
        gauge.set(1, state="queued")
        assert gauge.get(state="queued") == 1, "gauge 必须能下降，否则不是 gauge"

    def test_label_order_does_not_split_a_series(self, obs):
        """``inc(a=1, b=2)`` and ``inc(b=2, a=1)`` are one series.

        Not a micro-optimisation: without the sorted key a label set written in
        a different order silently doubles every count, and the symptom is a
        metric that is wrong in a way nobody can reproduce. Asserted on the
        *number of series* rather than the total, because the total is ``2``
        either way -- two increments of one series and two increments of two
        series sum to the same number, and only the series count tells them
        apart.
        """
        counter = obs.metrics.llm_calls
        counter.inc(stage="extraction", model="m")
        counter.inc(model="m", stage="extraction")
        assert len(counter.to_dict()) == 1, "同一个标签集合被拆成了两个 series"
        assert counter.get(stage="extraction", model="m") == 2

    def test_a_histogram_counts_sum_and_cumulative_buckets(self, obs):
        histogram = obs.metrics.llm_duration
        for value in (5.0, 100.0, 40000.0):
            histogram.observe(value)
        state = histogram.get()
        assert state["count"] == 3
        assert state["sum"] == pytest.approx(40105.0)
        # Cumulative: bucket 100 holds both the 5 and the 100.
        assert state["buckets"][histogram.buckets.index(5.0)] == 1
        assert state["buckets"][histogram.buckets.index(100.0)] == 2
        assert state["buckets"][-1] == 3, "超过所有边界的观测仍计入最高的有限桶"

    def test_cardinality_is_capped_and_the_overflow_is_counted(self, obs):
        """A ``document_id`` in a label must not become a table that never ends.

        The split between aggregate metrics and a per-id trace exists for this,
        and the cap is what enforces it: a caller who ignores that gets a
        counted, visible drop rather than a process that grows with the user's
        library.
        """
        counter = O.Counter("rag.overflow_probe", O.FAMILY_RAG, max_series=4)
        for index in range(10):
            counter.inc(document_id=f"doc_{index}")
        assert counter.total == 4
        assert counter.overflow == 6

    def test_a_metric_outside_the_six_planned_families_is_refused(self, obs):
        """A family nobody planned is a family nobody reads.

        Refused at construction rather than accepted and ignored, so a metric
        invented on a Friday cannot quietly become a third source of truth.
        """
        with pytest.raises(ValueError, match="未规划的族"):
            O.Counter("x.y", "cache_hits")
        assert set(O.FAMILIES) == {"parsing", "rag", "llm", "grounding", "jobs", "cost"}

    def test_prometheus_text_is_the_exposition_format_and_renames_the_dots(self, obs):
        """A Hosted deployment scrapes this with a real Prometheus and no client.

        The format is a stable contract and the library is not, which is why
        the plan's optional Prometheus costs no dependency.
        """
        _full_run(obs)
        text = obs.metrics.prometheus_text()
        assert 'paperwhisperer_parsing_documents_total{parser="mineru_cloud"' in text
        assert 'paperwhisperer_grounding_insufficient_evidence_total' in text
        assert "# TYPE paperwhisperer_llm_duration_ms histogram" in text
        assert 'paperwhisperer_llm_duration_ms_bucket{le="+Inf"}' in text
        # Every line is either a comment or a legal sample.
        for line in text.splitlines():
            if not line or line.startswith("#"):
                continue
            name, _, value = line.rpartition(" ")
            assert name.startswith("paperwhisperer_"), line
            float(value), line

    def test_the_registry_declares_every_family_up_front(self, obs):
        """A zero-valued metric must exist, so a panel can render a bar for it.

        The same argument :meth:`infra.jobs.JobStore.count_by_state` makes: a
        counter that appears the first time it fires makes "no cloud parses
        happened" indistinguishable from "nothing is watching".
        """
        declared = {metric.family for metric in obs.metrics.metrics}
        assert declared == set(O.FAMILIES)


# --------------------------------------------------------------------------
# Structured logging
# --------------------------------------------------------------------------


class TestStructuredLogging:
    def test_a_record_carries_the_ids_as_fields_not_as_text(self, obs):
        """The format exists so ``jq 'select(.document_id=="...")'`` is the query.

        An id interpolated into the message is greppable by a human and
        unqueryable by anything else, and the moment logs are aggregated across
        requests -- the moment this stops being a local dev convenience -- the
        interpolating format stops working.
        """
        stream = io.StringIO()
        obs.install(stream=stream, json_lines=True, level=logging.INFO)
        try:
            with obs.bind(document_id=DOC, job_id=JOB):
                obs.emit("parse.finished", family=O.FAMILY_PARSING, parser="docling")
        finally:
            obs.close()
        payload = json.loads(stream.getvalue().strip().splitlines()[-1])
        assert payload["document_id"] == DOC
        assert payload["job_id"] == JOB
        assert payload["parser"] == "docling"
        assert payload["event"] == "parse.finished"
        assert payload["message"] == "parse.finished"

    def test_the_human_readable_path_still_works(self, obs):
        """A developer running the app locally is a different consumer.

        A local regression to raw JSON would be a real cost for the person most
        likely to spot the bug, and the ids stay in the line because that
        person is about to go looking for the same id.
        """
        stream = io.StringIO()
        obs.install(stream=stream, json_lines=False, level=logging.INFO)
        try:
            with obs.bind(document_id=DOC):
                obs.emit("llm.call", family=O.FAMILY_LLM, model="gpt-4o-mini")
        finally:
            obs.close()
        line = stream.getvalue().strip()
        assert DOC in line
        assert "model=gpt-4o-mini" in line
        with pytest.raises(json.JSONDecodeError):
            json.loads(line)

    def test_an_id_absent_from_a_line_is_an_omitted_key(self, obs):
        """Four keys where two are ``""`` is a line a filter must special-case.

        The absence *is* the information, and omitting the key says it the way
        every log pipeline already understands.
        """
        stream = io.StringIO()
        obs.install(stream=stream, json_lines=True)
        try:
            obs.emit("something", family=O.FAMILY_JOBS)
        finally:
            obs.close()
        payload = json.loads(stream.getvalue().strip())
        assert set(O.ID_FIELDS).isdisjoint(payload)

    def test_an_unrelated_modules_logger_gets_the_ids_too(self, obs):
        """The record factory is global, which is the only hook the stdlib has.

        This is what makes the ids useful without every module adopting
        anything: a module that never calls into this file still has its
        records stamped, because the factory stamps them at creation. Which is
        also why the factory must be restorable, and why this test pairs with
        the teardown assertion.
        """
        stream = io.StringIO()
        obs.install(stream=stream, json_lines=True)
        try:
            with obs.bind(document_id=DOC):
                logging.getLogger("paperwhisperer.some.module").info("普通的一行")
        finally:
            obs.close()
        payload = json.loads(stream.getvalue().strip())
        assert payload["document_id"] == DOC
        assert payload["message"] == "普通的一行"

    def test_slow_successes_are_visible_because_nothing_else_reports_them(self, obs, monkeypatch):
        """A successful call that took 40 seconds is what a user complains about.

        It never appears in an error log, and it is the one signal that a
        timeout threshold is working. The threshold is read per call so a test
        can set it rather than the module owning a magic number.
        """
        from paperwhisperer.core import config

        monkeypatch.setattr(config, "OBSERVABILITY_SLOW_LLM_MS", 1000.0, raising=False)
        obs.record_model_call(_call(), duration_ms=40000.0)
        assert obs.trace.events(name="llm.call")[0].level == "WARNING"
        obs.trace.clear()
        obs.record_model_call(_call(), duration_ms=10.0)
        assert obs.trace.events(name="llm.call")[0].level == "INFO"


def _call() -> ModelCall:
    return ModelCall(stage="extraction", model="m", prompt_version="v", input_tokens=1,
                     output_tokens=1, cost=None, temperature=None, temperature_applied=True)


# --------------------------------------------------------------------------
# Off unless asked for
# --------------------------------------------------------------------------


class TestItIsOffUntilAskedFor:
    """The hazard this file could create for every other test in the suite."""

    def test_importing_the_module_changes_nothing(self):
        """Five modules import this one for the ``record_*`` helpers.

        A module-level facade that started collecting on import would mean
        every deployment that never asked for telemetry began paying for it,
        and every test in the repository would emit a JSON line. Asserted on
        the real singleton rather than a fresh object, because "the constructor
        defaults to disabled" would not catch a ``configure()`` call somewhere
        at import time.
        """
        assert O.observability.enabled is False
        assert len(O.observability.trace) == 0
        assert O.observability.metrics.parsing_documents.total == 0

    def test_a_disabled_facade_costs_a_boolean_test_and_nothing_else(self):
        """Off must be cheap, or nobody will leave it off.

        A disabled ``record_parse`` must not touch the metrics, must not append
        to the trace, and must not raise on an object shaped like nothing --
        because a logging helper that can fail on the error path is worse than
        no logging.
        """
        disabled = O.Observability(enabled=False)
        disabled.record_parse(_scanned_pdf_outcome(), duration_ms=1.0)
        disabled.record_retrieval(_retrieval_record())
        disabled.record_cost(source="llm", cost=1.0)
        assert len(disabled.trace) == 0
        assert disabled.metrics.parsing_documents.total == 0
        assert disabled.emit("x", family=O.FAMILY_PARSING) is None

    def test_install_is_idempotent_so_no_line_is_written_twice(self, obs):
        """A second ``install`` must return the first handler, not add another.

        Two handlers on the ``paperwhisperer`` logger means every line twice,
        and in a test suite that doubles the output of every assertion about
        logs written anywhere after it.
        """
        first = obs.install()
        try:
            second = obs.install()
            assert first is second
            assert logging.getLogger("paperwhisperer").handlers.count(first) == 1
        finally:
            obs.close()

    def test_close_restores_the_handler_level_and_record_factory_exactly(self, obs):
        """The restoration is the point, not a nicety.

        A stray record factory keeps stamping ids onto records for the rest of
        the session: harmless in isolation, quietly wrong in a suite that also
        asserts on log output. The autouse fixture asserts the same thing on
        teardown; this one asserts it with a deliberate change in between, so a
        regression is caught by the test that caused the setup.
        """
        package = logging.getLogger("paperwhisperer")
        original_level = package.level
        original_factory = logging.getLogRecordFactory()
        original_handlers = list(package.handlers)

        obs.install(level=logging.DEBUG)
        assert package.level == logging.DEBUG
        obs.close()

        assert package.level == original_level
        assert package.handlers == original_handlers
        assert logging.getLogRecordFactory() is original_factory
        assert obs.enabled is False

    def test_close_is_idempotent_so_a_fixture_teardown_need_not_know(self, obs):
        obs.install()
        obs.close()
        obs.close()  # must not raise, must not re-remove somebody else's handler

    def test_configure_and_shutdown_round_trip(self, obs, monkeypatch):
        """The public entry points, on the real singleton.

        The only test that touches the singleton deliberately: it configures,
        asserts, and hands the process back through the same path the
        autouse fixture uses, so the ordering of two teardowns cannot leave the
        process configured.
        """
        stream = io.StringIO()
        O.configure(level="INFO", json_lines=True, stream=stream)
        try:
            assert O.observability.enabled is True
            with O.correlate(document_id=DOC):
                O.observability.record_retrieval(_retrieval_record(), duration_ms=5.0)
            assert O.observability.trace.events(document_id=DOC)
            assert DOC in stream.getvalue()
        finally:
            O.shutdown()
        assert O.observability.enabled is False

    def test_a_settings_key_that_does_not_exist_falls_back_to_the_declared_default(self):
        """``config`` is not this task's to change, so the table is the contract.

        Same arrangement :mod:`infra.jobs` and :mod:`analysis.paper_facts` use:
        the reader is ``getattr(config, name, default)`` against
        :data:`OBSERVABILITY_CONFIG_KEYS`, and whoever adds the keys to config
        copies these names and defaults. Absent key, default in force -- and the
        defaults are the conservative ones, so a checkout nobody configured
        behaves exactly as it did before this file existed.
        """
        from paperwhisperer.core import config

        # The keys now exist in config, so the contract is parity, not absence:
        # a default that disagrees with the table is a silent behaviour change
        # for anyone who never sets the variable, and nothing else would catch
        # it.
        mismatched = {
            name: (getattr(config, name, None), default)
            for name, default in O.OBSERVABILITY_CONFIG_KEYS.items()
            if str(getattr(config, name, None)) != str(default)
        }
        assert not mismatched, (
            f"config 与 OBSERVABILITY_CONFIG_KEYS 不一致：{mismatched}"
        )
        assert O.OBSERVABILITY_CONFIG_KEYS["OBSERVABILITY_ENABLED"] is False, (
            "observability must be off until asked for; a default that turns it "
            "on changes behaviour for every existing deployment"
        )


# --------------------------------------------------------------------------
# The exporter seam
# --------------------------------------------------------------------------


class TestTheExporterSeam:
    """An interface with no implementation is an interface nobody exercised."""

    def test_the_seam_receives_both_halves_of_the_state(self, obs):
        """Spans without the counters for the same window make a lying dashboard.

        So ``export`` is handed both, and a caller that only wants one ignores
        the other. The ``NullExporter`` is the default precisely so that the
        path is exercised in every deployment, including the ones that do not
        want an exporter.
        """
        _full_run(obs)
        assert obs.flush() > 0
        assert obs.exporter.batches == 1
        assert obs.exporter.event_count > 0

    def test_a_jsonl_exporter_survives_the_process(self, obs, tmp_path):
        """On the personal tier the alternative to a trace file is a reboot.

        Append mode, one object per line, and the file is written per batch so
        a crash costs a batch rather than the file.
        """
        target = tmp_path / "trace.jsonl"
        obs.exporter = O.JsonlExporter(str(target))
        _full_run(obs)
        assert obs.flush() > 0
        lines = target.read_text(encoding="utf-8").strip().splitlines()
        assert lines, "trace 文件是空的"
        first = json.loads(lines[0])
        assert first["document_id"] == DOC
        assert all(json.loads(line) for line in lines)

    def test_an_exporter_that_raises_cannot_break_the_caller(self, obs):
        """It is called on the request path; an I/O failure there is our bug.

        ``flush`` is the one place this module hands control to somebody else's
        code, and a metrics library whose export can fail a response would be
        worse than no metrics.
        """

        class Exploding(O.Exporter):
            def export(self, events, metrics):
                raise OSError("the collector is down")

        obs.exporter = Exploding()
        _full_run(obs)
        assert obs.flush() == 0
        assert obs.trace.events(document_id=DOC), "导出失败不得影响已经记录的 trace"

    def test_a_corrupt_export_target_is_counted_not_raised(self, obs, tmp_path):
        """A trace file that cannot be written must not take down the process
        that was being diagnosed -- and must not fail silently either."""
        obs.exporter = O.JsonlExporter(str(tmp_path / "missing" / "dir" / "trace.jsonl"))
        _full_run(obs)
        obs.flush()
        assert obs.exporter.corrupt == 1
