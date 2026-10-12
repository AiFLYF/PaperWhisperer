"""Parser gateway: routing, timeout and error mapping, plus the legacy adapter.

The gateway's whole job is to make every backend look alike to its caller, so
the tests are organised around the promises it makes rather than around the
classes it defines:

* a usable parse comes back as a result, never as an exception, and a total
  failure still comes back as a result;
* a usable result also carries the quality report, because a gate that has to
  be asked for is a gate nobody asks for — though the verdict is recorded, not
  enforced;
* parsers are tried in the configured order, and only those that claim the
  media type are asked to run at all;
* the ``PARSER_BACKEND`` flag can collapse the chain to the legacy adapter
  without rebuilding the router;
* a parser that hangs becomes a failed result rather than a hung request;
* the legacy adapter produces canonical documents with honest locators — which
  for PDF means a page and no bounding box.

Fakes are defined here rather than shared through ``conftest`` because they are
only meaningful against this contract; a parser double that outlives its test
ends up encoding assumptions nobody remembers making. The legacy adapter tests
build real files in ``tmp_path`` for the same reason: an adapter's job is
reading real bytes, and a mock of the thing under test would test nothing.
"""

from __future__ import annotations

import concurrent.futures
import time

import pytest

from paperwhisperer.core import config
from paperwhisperer.ingestion.adapters.legacy_loader import LegacyDocumentParser
from paperwhisperer.ingestion.gateway import (
    LEGACY_PARSER_NAME,
    PARSE_FAILED,
    PARSE_PARTIAL,
    PARSE_SUCCESS,
    STATUS_ALIASES,
    ParseOptions,
    ParserError,
    ParseResult,
    ParserRouter,
    ParserTimeout,
    media_type_for_path,
)
from paperwhisperer.ingestion.schema import Element, ParsedDocument, SourceLocator

#: Media type the fakes claim, chosen to match the legacy adapter's table so
#: one constant serves both the router tests and the adapter tests.
PDF_MEDIA_TYPE = "application/pdf"
TXT_MEDIA_TYPE = "text/plain"


def success_result(text: str = "body") -> ParseResult:
    """A usable result, built the way a real adapter would build one."""
    document = ParsedDocument(
        source_filename="a.pdf",
        media_type=PDF_MEDIA_TYPE,
        elements=[
            Element(
                role="paragraph",
                text=text,
                locator=SourceLocator(kind="pdf_page", page=1),
            )
        ],
    )
    return ParseResult.from_document(document)


class FakeParser:
    """A parser double whose every observable is set by the test.

    ``outcome`` is either a :class:`ParseResult` to return or an exception
    instance to raise. Returning and raising are kept distinct on purpose: the
    router's contract says an adapter raises on failure, and a double that only
    knows how to return failures would let a broken mapping pass unnoticed.
    """

    def __init__(
        self,
        name: str,
        *,
        version: str = "1.0",
        outcome=None,
        media_types=(PDF_MEDIA_TYPE,),
        delay: float = 0.0,
        calls: list | None = None,
    ):
        self.name = name
        self.version = version
        self.outcome = outcome
        self.media_types = tuple(media_types)
        self.delay = delay
        self.calls = calls if calls is not None else []

    def supports(self, media_type: str) -> bool:
        self.calls.append(("supports", media_type))
        return media_type in self.media_types

    def parse(self, file_path, options):
        self.calls.append(("parse", file_path))
        if self.delay:
            time.sleep(self.delay)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        if callable(self.outcome):
            return self.outcome(file_path, options)
        return self.outcome


class ImmediateExecutor:
    """An executor that runs work synchronously and never really times out.

    Used for every test that is not *about* the timeout, so the suite does not
    pay for thread creation per assertion and cannot become flaky through
    scheduling. It implements only the two methods the router calls, which is
    the point of the ``ParserExecutor`` protocol being that narrow.
    """

    def __init__(self):
        self.shutdown_calls: list[bool] = []

    def submit(self, fn, /, *args, **kwargs):
        future: concurrent.futures.Future = concurrent.futures.Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # noqa: BLE001 - mirrors a real future
            future.set_exception(exc)
        return future

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        self.shutdown_calls.append(wait)


class ScriptedTimeoutExecutor:
    """An executor whose ``result()`` always raises ``TimeoutError``.

    Deterministic stand-in for "the parser is still running when the budget
    runs out". A real thread plus a real sleep would test the same branch while
    making the suite slower and, on a loaded machine, occasionally ambiguous
    about which timeout fired.
    """

    def __init__(self):
        self.cancelled = False

    def submit(self, fn, /, *args, **kwargs):
        future: concurrent.futures.Future = concurrent.futures.Future()
        # Deliberately never resolved: the work is "in flight" forever.
        return future

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        return None


class TestParseOptions:
    def test_the_same_options_hash_the_same_way(self):
        assert ParseOptions().compute_options_hash() == ParseOptions().compute_options_hash()

    def test_turning_force_ocr_on_changes_the_hash(self):
        plain = ParseOptions(force_ocr=False)
        forced = ParseOptions(force_ocr=True)

        assert plain.compute_options_hash() != forced.compute_options_hash()

    def test_extra_keys_are_part_of_the_hash_regardless_of_insertion_order(self):
        first = ParseOptions(extra={"mode": "fast", "depth": 2})
        second = ParseOptions(extra={"depth": 2, "mode": "fast"})

        assert first.compute_options_hash() == second.compute_options_hash()

    def test_changing_extra_changes_the_hash(self):
        assert (
            ParseOptions(extra={"mode": "fast"}).compute_options_hash()
            != ParseOptions(extra={"mode": "accurate"}).compute_options_hash()
        )

    def test_the_hash_is_short_enough_to_use_as_a_cache_key_component(self):
        digest = ParseOptions().compute_options_hash()

        assert len(digest) == 16
        assert set(digest) <= set("0123456789abcdef")

    def test_rejects_a_non_mapping_extra(self):
        with pytest.raises(ValueError, match="extra must be a dict"):
            ParseOptions(extra=["not", "a", "mapping"])


class TestParseResult:
    def test_from_document_inherits_provenance_from_the_document(self):
        document = ParsedDocument(parser_name="legacy", parser_version="1.0")
        document.options_hash = "abc123"

        result = ParseResult.from_document(document)

        assert result.parser_name == "legacy"
        assert result.parser_version == "1.0"
        assert result.options_hash == "abc123"
        assert result.status == PARSE_SUCCESS

    def test_a_failure_carries_no_document_and_no_quality_report(self):
        result = ParseResult.failure("解析器炸了")

        assert result.status == PARSE_FAILED
        assert result.document is None
        assert result.quality is None
        assert result.error_message == "解析器炸了"

    def test_partial_counts_as_usable_but_success_does_not_leak_into_partial(self):
        assert ParseResult(status=PARSE_PARTIAL).is_usable
        assert not ParseResult.failure("no").is_usable

    def test_an_unknown_status_is_downgraded_to_failed(self):
        result = ParseResult(status="mostly fine")

        assert result.status == PARSE_FAILED
        assert result.error_message

    def test_the_alias_table_names_the_docling_status(self):
        # Pinned by name, not only by behaviour, so that deleting the entry is
        # a red test rather than a silent regression nobody notices until a
        # user reports that every docling parse failed.
        assert STATUS_ALIASES["ok"] == PARSE_SUCCESS

    def test_a_status_spelled_ok_by_an_adapter_is_read_as_success(self):
        # The docling adapter's own result type spells success "ok" and builds
        # a real ParseResult from it. Without the alias, a perfect parse comes
        # back marked as a failure and the user is told their PDF is broken.
        assert ParseResult(status="ok").status == PARSE_SUCCESS

    @pytest.mark.parametrize(
        "status",
        ["OK", "okay", "ok ", " success", "succeeded"],
    )
    def test_a_status_that_only_resembles_the_alias_is_still_a_failure(self, status):
        # The alias is exact. The moment it becomes a case-folding or a prefix
        # match, a typo in an adapter's status string turns into a document
        # nobody checked — which is the failure mode the table exists to avoid.
        result = ParseResult(status=status)

        assert result.status == PARSE_FAILED
        assert result.error_message


class TestAdapterStatusVocabulary:
    """The seam between the docling adapter's ``ok`` and the gateway's statuses.

    These tests are the only thing standing between a future cleanup and a
    silent, total breakage of the docling path, so they are deliberately
    redundant: one pins the table by name, one pins the behaviour it buys, and
    one pins that the table is not a general excuse for loose statuses.
    """

    @pytest.fixture(autouse=True)
    def _gateway_chain(self, gateway_backend):
        """Pin the feature flag for this class; see :func:`gateway_backend`."""

    def test_a_docling_shaped_result_survives_the_router(self):
        # What the docling adapter actually hands over: a gateway ParseResult
        # carrying the status from its own vocabulary, plus a real document.
        def docling_shaped(file_path, options):
            return ParseResult(
                status="ok",
                document=success_result("docling body").document,
            )

        router = ParserRouter(
            [FakeParser("docling", outcome=docling_shaped)],
            executor_factory=ImmediateExecutor,
        )

        result = router.parse("a.pdf", PDF_MEDIA_TYPE)

        # Not just the status: without the alias the document is still there
        # but the result is a failure, so the chain would move on and throw
        # away a perfectly good parse.
        assert result.status == PARSE_SUCCESS
        assert result.is_usable
        assert result.document.elements[0].text == "docling body"

    def test_the_aliased_status_reaches_the_one_success_predicate(self):
        # ``is_usable`` is the only "may I use this as a document" check in the
        # gateway; the docling mirror's own ``ok`` property is not one of them.
        # This asserts the alias and the predicate agree, so standardising on
        # one name does not split the two vocabularies again.
        result = ParseResult(status="ok", document=success_result().document)

        assert result.is_usable
        assert result.is_usable is (result.status in {PARSE_SUCCESS, PARSE_PARTIAL})

    def test_a_usable_result_with_no_document_is_rejected_whatever_it_calls_itself(
        self,
    ):
        # The empty-document guard asks ``is_usable`` rather than comparing
        # against "success", so a partial that dropped everything is caught by
        # the same check instead of reaching a caller as a usable result whose
        # ``document`` is None.
        router = ParserRouter(
            [FakeParser("empty", outcome=lambda path, options: ParseResult(status=PARSE_PARTIAL))],
            executor_factory=ImmediateExecutor,
        )

        result = router.parse("a.pdf", PDF_MEDIA_TYPE)

        assert result.status == PARSE_FAILED
        assert "没有返回任何内容" in result.error_message


class TestParserRouterQualityGate:
    @pytest.fixture(autouse=True)
    def _gateway_chain(self, gateway_backend):
        """Pin the feature flag for this class; see :func:`gateway_backend`."""

    def test_a_usable_result_carries_the_quality_report(self):
        router = ParserRouter(
            [FakeParser("only", outcome=success_result())],
            executor_factory=ImmediateExecutor,
        )

        result = router.parse("a.pdf", PDF_MEDIA_TYPE)

        # Every parse passes through the router, so this is the one place the
        # gate can be guaranteed to have run.
        assert result.quality is not None
        assert result.quality.evaluated_rules
        assert result.quality.flags

    def test_a_failing_verdict_does_not_reject_the_document(self):
        # "Hello world" trips the text-density rule, and rejecting it here
        # would throw away the readable part of a mostly-bad document. Whether
        # to retry with OCR or spend a cloud call is the orchestrator's call.
        router = ParserRouter(
            [FakeParser("only", outcome=success_result("Hello world"))],
            executor_factory=ImmediateExecutor,
        )

        result = router.parse("a.pdf", PDF_MEDIA_TYPE)

        assert result.status == PARSE_SUCCESS
        assert result.quality is not None

    def test_a_failure_is_never_judged(self):
        # There is no document to judge, and a report attached to a failure
        # would read as "the document was bad" rather than "there was none".
        router = ParserRouter(
            [FakeParser("boom", outcome=ParserError("坏了"))],
            executor_factory=ImmediateExecutor,
        )

        assert router.parse("a.pdf", PDF_MEDIA_TYPE).quality is None


@pytest.fixture
def gateway_backend(monkeypatch):
    """Run a test against the gateway chain rather than the rollback path.

    ``PARSER_BACKEND`` ships as ``legacy``: a build where the new backends are
    not installed yet still has to parse a PDF, and defaulting to the new chain
    would break exactly that. So every test says which mode it means instead of
    inheriting whatever ``.env`` happened to contain. A test whose result
    depends on the developer's environment is a test that eventually passes for
    the wrong reason and fails for the wrong reason.

    The classes that exercise dispatch, timeouts and error mapping take this
    implicitly; the feature-flag class sets the value by hand because setting
    it is the thing under test there.
    """
    monkeypatch.setattr(config, "PARSER_BACKEND", "gateway")


class TestParserRouterDispatch:
    @pytest.fixture(autouse=True)
    def _gateway_chain(self, gateway_backend):
        """Pin the feature flag for this class; see :func:`gateway_backend`."""

    def test_returns_the_first_successful_result(self):
        calls: list = []
        router = ParserRouter(
            [
                FakeParser("first", outcome=success_result(), calls=calls),
                FakeParser("second", outcome=success_result(), calls=calls),
            ],
            executor_factory=ImmediateExecutor,
        )

        result = router.parse("a.pdf", PDF_MEDIA_TYPE)

        assert result.status == PARSE_SUCCESS
        # The second parser is never asked, so a paid cloud backend is not
        # called after a local one already succeeded.
        assert ("parse", "a.pdf") in calls
        assert sum(1 for call in calls if call[0] == "parse") == 1

    def test_falls_through_to_the_next_parser_when_one_fails(self):
        calls: list = []
        router = ParserRouter(
            [
                FakeParser("first", outcome=ParserError("坏了"), calls=calls),
                FakeParser("second", outcome=success_result("second body"), calls=calls),
            ],
            executor_factory=ImmediateExecutor,
        )

        result = router.parse("a.pdf", PDF_MEDIA_TYPE)

        assert result.status == PARSE_SUCCESS
        assert result.document.elements[0].text == "second body"
        assert [call[0] for call in calls].count("parse") == 2

    def test_a_partial_result_is_usable_and_is_announced_as_such(self):
        router = ParserRouter(
            [
                FakeParser("partial", outcome=ParseResult(status=PARSE_PARTIAL,
                                                          document=success_result().document)),
                FakeParser("never", outcome=success_result(), calls=[]),
            ],
            executor_factory=ImmediateExecutor,
        )

        result = router.parse("a.pdf", PDF_MEDIA_TYPE)

        assert result.status == PARSE_PARTIAL
        assert result.is_usable
        assert any("不完整" in warning for warning in result.warnings)

    def test_returns_the_last_failure_when_every_parser_fails(self):
        router = ParserRouter(
            [
                FakeParser("first", outcome=ParserError("第一个失败")),
                FakeParser("second", outcome=ParserError("第二个失败")),
            ],
            executor_factory=ImmediateExecutor,
        )

        result = router.parse("a.pdf", PDF_MEDIA_TYPE)

        # No exception escapes: a broken backend is an operational fact, and
        # the caller renders it rather than crashing on it.
        assert result.status == PARSE_FAILED
        assert result.error_message == "第二个失败"
        assert not result.is_usable

    def test_only_parsers_that_claim_the_media_type_are_run(self):
        pdf_calls: list = []
        txt_calls: list = []
        router = ParserRouter(
            [
                FakeParser("txtonly", outcome=success_result(), media_types=(TXT_MEDIA_TYPE,),
                           calls=txt_calls),
                FakeParser("pdfonly", outcome=success_result(), calls=pdf_calls),
            ],
            executor_factory=ImmediateExecutor,
        )

        result = router.parse("a.pdf", PDF_MEDIA_TYPE)

        assert result.parser_name == "pdfonly"
        assert not any(call[0] == "parse" for call in txt_calls)
        assert any(call[0] == "parse" for call in pdf_calls)

    def test_reports_a_coverage_gap_when_no_parser_supports_the_type(self):
        router = ParserRouter(
            [FakeParser("pdfonly", outcome=success_result())],
            executor_factory=ImmediateExecutor,
        )

        result = router.parse("a.zip", "application/zip")

        assert result.status == PARSE_FAILED
        assert "没有解析器支持该文件类型" in result.error_message

    def test_a_parser_whose_supports_check_raises_is_treated_as_unsupported(self):
        class ExplodingSupport(FakeParser):
            def supports(self, media_type: str) -> bool:
                raise RuntimeError("依赖缺失")

        router = ParserRouter(
            [ExplodingSupport("broken"), FakeParser("healthy", outcome=success_result())],
            executor_factory=ImmediateExecutor,
        )

        result = router.parse("a.pdf", PDF_MEDIA_TYPE)

        assert result.status == PARSE_SUCCESS
        assert result.parser_name == "healthy"

    def test_a_parser_returning_the_wrong_type_is_reported_as_a_failure(self):
        router = ParserRouter(
            [FakeParser("sloppy", outcome=lambda path, options: "just a string")],
            executor_factory=ImmediateExecutor,
        )

        result = router.parse("a.pdf", PDF_MEDIA_TYPE)

        assert result.status == PARSE_FAILED
        assert "非 ParseResult" in result.error_message

    def test_a_success_without_a_document_is_treated_as_a_failure(self):
        router = ParserRouter(
            [FakeParser("empty", outcome=lambda path, options: ParseResult())],
            executor_factory=ImmediateExecutor,
        )

        result = router.parse("a.pdf", PDF_MEDIA_TYPE)

        assert result.status == PARSE_FAILED
        assert "没有返回任何内容" in result.error_message

    def test_the_router_stamps_provenance_the_adapter_forgot_to_set(self):
        forgetful_document = success_result().document

        def forgetful(file_path, options):
            # A document with no parser_name/version/options_hash, which is
            # exactly what an adapter that forgets the provenance block returns.
            return ParseResult.from_document(forgetful_document)

        router = ParserRouter(
            [FakeParser("forgetful", version="2.5", outcome=forgetful)],
            executor_factory=ImmediateExecutor,
        )

        result = router.parse("a.pdf", PDF_MEDIA_TYPE, options=ParseOptions())

        assert result.parser_name == "forgetful"
        assert result.parser_version == "2.5"
        assert result.options_hash == ParseOptions().compute_options_hash()

    def test_diagnostics_record_every_attempt_in_order(self):
        router = ParserRouter(
            [
                FakeParser("first", outcome=ParserError("第一个失败")),
                FakeParser("second", outcome=success_result()),
            ],
            executor_factory=ImmediateExecutor,
        )

        result = router.parse("a.pdf", PDF_MEDIA_TYPE)

        attempted = [entry["parser"] for entry in result.diagnostics if entry["attempted"]]
        assert attempted == ["first", "second"]
        assert result.diagnostics[0]["status"] == PARSE_FAILED
        assert result.diagnostics[1]["status"] == PARSE_SUCCESS

    def test_diagnostics_mark_a_skipped_parser_instead_of_dropping_it(self):
        router = ParserRouter(
            [FakeParser("txtonly", media_types=(TXT_MEDIA_TYPE,), outcome=success_result())],
            executor_factory=ImmediateExecutor,
        )

        result = router.parse("a.pdf", PDF_MEDIA_TYPE)

        assert result.diagnostics[0]["attempted"] is False
        assert result.diagnostics[0]["skipped_reason"] == "unsupported_media_type"

    def test_a_parser_is_run_on_a_worker_thread_not_on_the_calling_one(self):
        import threading

        class ThreadRecordingParser(FakeParser):
            def parse(self, file_path, options):
                self.worker_thread = threading.current_thread().ident
                return success_result()

        parser = ThreadRecordingParser("worker")
        router = ParserRouter([parser])

        router.parse("a.pdf", PDF_MEDIA_TYPE)

        assert parser.worker_thread != threading.main_thread().ident


class TestParserRouterFeatureFlag:
    def test_the_legacy_flag_collapses_the_chain_to_the_legacy_parser(
        self, monkeypatch
    ):
        monkeypatch.setattr(config, "PARSER_BACKEND", "legacy")
        legacy_calls: list = []
        router = ParserRouter(
            [
                FakeParser("docling", outcome=success_result(), calls=legacy_calls),
                FakeParser(LEGACY_PARSER_NAME, outcome=success_result(), calls=legacy_calls),
            ],
            executor_factory=ImmediateExecutor,
        )

        result = router.parse("a.pdf", PDF_MEDIA_TYPE)

        assert result.parser_name == LEGACY_PARSER_NAME

    def test_the_legacy_flag_is_read_on_every_call_not_at_construction(
        self, monkeypatch
    ):
        router = ParserRouter(
            [
                FakeParser("docling", outcome=success_result()),
                FakeParser(LEGACY_PARSER_NAME, outcome=success_result()),
            ],
            executor_factory=ImmediateExecutor,
        )
        monkeypatch.setattr(config, "PARSER_BACKEND", "legacy")

        assert router.parse("a.pdf", PDF_MEDIA_TYPE).parser_name == LEGACY_PARSER_NAME

    def test_the_legacy_flag_fails_loudly_when_no_legacy_parser_is_registered(
        self, monkeypatch
    ):
        # A silently empty chain would report "unsupported format" and send the
        # user hunting for an upload problem that does not exist.
        monkeypatch.setattr(config, "PARSER_BACKEND", "legacy")
        router = ParserRouter(
            [FakeParser("docling", outcome=success_result())],
            executor_factory=ImmediateExecutor,
        )

        result = router.parse("a.pdf", PDF_MEDIA_TYPE)

        assert result.status == PARSE_FAILED
        assert "没有可用的文档解析器" in result.error_message

    def test_the_gateway_flag_follows_the_primary_then_the_fallback_chain(
        self, monkeypatch
    ):
        monkeypatch.setattr(config, "PARSER_BACKEND", "gateway")
        monkeypatch.setattr(config, "PARSER_PRIMARY", "beta")
        monkeypatch.setattr(config, "PARSER_FALLBACK_CHAIN", ("alpha", "legacy"))
        calls: list = []
        router = ParserRouter(
            [
                FakeParser("alpha", outcome=ParserError("alpha 挂了"), calls=calls),
                FakeParser(LEGACY_PARSER_NAME, outcome=success_result("legacy body"),
                           calls=calls),
                FakeParser("beta", outcome=success_result("beta body"), calls=calls),
            ],
            executor_factory=ImmediateExecutor,
        )

        result = router.parse("a.pdf", PDF_MEDIA_TYPE)

        assert result.document.elements[0].text == "beta body"
        assert [call[0] for call in calls].count("parse") == 1

    def test_a_registered_parser_absent_from_the_configuration_still_gets_a_turn(
        self, monkeypatch
    ):
        # Config names a parser this build does not ship. Dropping the local
        # one as well would mean an uninstalled docling disables txt parsing.
        monkeypatch.setattr(config, "PARSER_BACKEND", "gateway")
        monkeypatch.setattr(config, "PARSER_PRIMARY", "docling")
        monkeypatch.setattr(config, "PARSER_FALLBACK_CHAIN", ())
        router = ParserRouter(
            [FakeParser(LEGACY_PARSER_NAME, outcome=success_result())],
            executor_factory=ImmediateExecutor,
        )

        result = router.parse("a.pdf", PDF_MEDIA_TYPE)

        assert result.status == PARSE_SUCCESS
        assert result.parser_name == LEGACY_PARSER_NAME

    def test_an_unrecognised_backend_value_falls_back_to_legacy(
        self, monkeypatch
    ):
        monkeypatch.setattr(config, "PARSER_BACKEND", "martian")
        router = ParserRouter(
            [
                FakeParser("docling", outcome=success_result()),
                FakeParser(LEGACY_PARSER_NAME, outcome=success_result()),
            ],
            executor_factory=ImmediateExecutor,
        )

        assert router.parse("a.pdf", PDF_MEDIA_TYPE).parser_name == LEGACY_PARSER_NAME


class TestParserRouterTimeout:
    @pytest.fixture(autouse=True)
    def _gateway_chain(self, gateway_backend):
        """Pin the feature flag for this class; see :func:`gateway_backend`."""

    def test_a_parser_that_overruns_the_budget_becomes_a_failed_result(self):
        router = ParserRouter(
            [FakeParser("slow", outcome=success_result())],
            executor_factory=ScriptedTimeoutExecutor,
        )

        result = router.parse("a.pdf", PDF_MEDIA_TYPE, options=ParseOptions(timeout_seconds=1))

        assert result.status == PARSE_FAILED
        assert "解析超时" in result.error_message
        assert result.parser_name == "slow"

    def test_a_real_hanging_parser_is_abandoned_rather_than_awaited(self):
        # The honest version of the test above: a genuine thread that sleeps
        # past the budget. It confirms the executor is really consulted, and
        # that the router returns while the parser is still running.
        router = ParserRouter([FakeParser("slow", outcome=success_result(), delay=0.6)])

        started = time.monotonic()
        # A budget well inside the parser's sleep. If it were longer than the
        # sleep the parse would simply succeed and this test would assert
        # nothing about timeouts at all.
        result = router.parse(
            "a.pdf", PDF_MEDIA_TYPE, options=ParseOptions(timeout_seconds=0.2)
        )
        elapsed = time.monotonic() - started

        assert result.status == PARSE_FAILED
        assert elapsed < 0.6

    def test_a_zero_budget_means_no_timeout_rather_than_give_up_immediately(self):
        # A missing config value should not read as "give up at once"; the
        # socket convention (zero means wait) is the least surprising reading.
        router = ParserRouter(
            [FakeParser("slow", outcome=success_result(), delay=0.05)],
            options=ParseOptions(timeout_seconds=0),
        )

        result = router.parse("a.pdf", PDF_MEDIA_TYPE, options=ParseOptions(timeout_seconds=0))

        assert result.status == PARSE_SUCCESS

    def test_a_parser_that_raises_parser_timeout_is_reported_as_a_timeout(self):
        router = ParserRouter(
            [FakeParser("slow", outcome=ParserTimeout("自行判定超时"))],
            executor_factory=ImmediateExecutor,
        )

        result = router.parse("a.pdf", PDF_MEDIA_TYPE)

        assert result.status == PARSE_FAILED
        assert "解析超时" in result.error_message

    def test_a_timed_out_parser_does_not_stop_the_chain(self):
        # The chain has to survive a timeout, or one wedged backend takes the
        # whole document down instead of just its own turn.
        created: list = []

        def factory():
            executor = ScriptedTimeoutExecutor() if not created else ImmediateExecutor()
            created.append(executor)
            return executor

        router = ParserRouter(
            [
                FakeParser("slow", outcome=success_result()),
                FakeParser("next", outcome=success_result("next body")),
            ],
            executor_factory=factory,
        )

        result = router.parse("a.pdf", PDF_MEDIA_TYPE, options=ParseOptions(timeout_seconds=1))

        assert result.status == PARSE_SUCCESS
        assert result.document.elements[0].text == "next body"


class TestParserErrorMapping:
    @pytest.fixture(autouse=True)
    def _gateway_chain(self, gateway_backend):
        """Pin the feature flag for this class; see :func:`gateway_backend`."""

    @pytest.mark.parametrize(
        ("raised", "fragment"),
        [
            (ValueError("PDF 读取失败: boom"), "PDF 读取失败"),
            (FileNotFoundError(2, "No such file"), "找不到待解析的文件"),
            (PermissionError(13, "denied"), "没有读取该文件的权限"),
            (MemoryError(), "内存不足"),
            (OSError("disk on fire"), "系统错误"),
            (RuntimeError("weird"), "未预期的错误"),
        ],
    )
    def test_backend_exceptions_become_chinese_failure_messages(self, raised, fragment):
        router = ParserRouter(
            [FakeParser("boom", outcome=raised)],
            executor_factory=ImmediateExecutor,
        )

        result = router.parse("a.pdf", PDF_MEDIA_TYPE)

        assert result.status == PARSE_FAILED
        assert fragment in result.error_message
        assert result.parser_name == "boom"

    def test_an_empty_parser_error_still_produces_a_readable_message(self):
        router = ParserRouter(
            [FakeParser("quiet", outcome=ParserError(""))],
            executor_factory=ImmediateExecutor,
        )

        result = router.parse("a.pdf", PDF_MEDIA_TYPE)

        assert "未提供具体原因" in result.error_message

    def test_a_bare_exception_carries_the_parser_name(self):
        # Without the name, a failure log line cannot say which backend broke.
        router = ParserRouter(
            [FakeParser("mineru_cloud", outcome=RuntimeError("签名错误"))],
            executor_factory=ImmediateExecutor,
        )

        result = router.parse("a.pdf", PDF_MEDIA_TYPE)

        assert "mineru_cloud" in result.error_message


class TestMediaTypeForPath:
    @pytest.mark.parametrize(
        ("file_name", "media_type"),
        [
            ("a.txt", TXT_MEDIA_TYPE),
            ("a.pdf", PDF_MEDIA_TYPE),
            (
                "a.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ),
            (
                "a.pptx",
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ),
        ],
    )
    def test_known_extensions_map_to_their_iana_media_types(self, file_name, media_type):
        assert media_type_for_path(file_name) == media_type

    def test_an_unknown_extension_falls_back_to_a_generic_binary_type(self):
        assert media_type_for_path("a.unknown-ext") == "application/octet-stream"


class TestLegacyAdapterSupports:
    @pytest.mark.parametrize(
        "media_type",
        [
            TXT_MEDIA_TYPE,
            PDF_MEDIA_TYPE,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ],
    )
    def test_claims_every_format_the_legacy_loader_reads(self, media_type):
        assert LegacyDocumentParser().supports(media_type)

    def test_declines_anything_else(self):
        assert not LegacyDocumentParser().supports("application/zip")

    def test_identifies_itself_for_traceability(self):
        parser = LegacyDocumentParser()

        assert parser.name == "legacy"
        assert parser.version == "1.0"


class TestLegacyAdapterTxt:
    def test_slices_plain_text_into_offset_addressed_elements(self, tmp_path):
        target = tmp_path / "a.txt"
        target.write_text("Hello world", encoding="utf-8")

        document = _parse_with_legacy(target)

        assert len(document.elements) == 1
        element = document.elements[0]
        assert element.role == "paragraph"
        assert element.text == "Hello world"
        assert element.locator.kind == "txt_offset"
        assert element.locator.offset_start == 0
        assert element.locator.offset_end == len("Hello world")

    def test_long_text_is_split_into_several_spans_that_cover_it(self, tmp_path):
        target = tmp_path / "long.txt"
        body = "".join(f"line {index}\n" for index in range(2000))
        target.write_text(body, encoding="utf-8")

        document = _parse_with_legacy(target)

        assert len(document.elements) > 1
        assert all(element.locator.offset_end > element.locator.offset_start
                   for element in document.elements)
        assert all(element.locator.is_resolvable for element in document.elements)

    def test_offsets_refer_to_the_cleaned_text_the_pipeline_sees(self, tmp_path):
        # Cleaning removes whitespace, so offsets taken before cleaning would
        # point at the wrong characters — worse than having no offset at all.
        target = tmp_path / "messy.txt"
        target.write_text("Hello  \n\n\n\n  world ", encoding="utf-8")

        document = _parse_with_legacy(target)

        element = document.elements[0]
        assert element.locator.offset_start == 0
        assert element.locator.offset_end == len("Hello\n\nworld")

    def test_an_empty_file_produces_a_degraded_result_not_a_crash(self, tmp_path):
        target = tmp_path / "empty.txt"
        target.write_text("   \n\n  ", encoding="utf-8")

        result = _result_with_legacy(target)

        assert result.status == PARSE_PARTIAL
        assert any("没有解析出任何正文" in warning for warning in result.warnings)


class TestLegacyAdapterPdf:
    def test_each_page_becomes_an_element_located_by_page_without_a_bbox(self, tmp_path):
        # ``PdfWriter`` is reached through the package rather than through a
        # ``pypdf.writer`` submodule: that module path is a pypdf 3.x
        # internal, and importing it skips the test on every current release —
        # which is how a whole PDF code path ends up with no coverage at all.
        PdfWriter = pytest.importorskip("pypdf").PdfWriter

        target = tmp_path / "a.pdf"
        writer = PdfWriter()
        for _ in range(2):
            writer.add_blank_page(width=200, height=200)
        with open(target, "wb") as handle:
            writer.write(handle)

        document = _parse_with_legacy(target)

        # A blank page has no text layer, so there is nothing to assert per page
        # beyond the parse not lying about what it found.
        assert document.page_count == 2
        assert all(element.locator.bbox is None for element in document.elements)

    def test_reports_a_broken_pdf_with_the_legacy_error_message(self, tmp_path):
        pytest.importorskip("pypdf")
        target = tmp_path / "broken.pdf"
        target.write_bytes(b"%PDF-1.7\nnot really a pdf")

        with pytest.raises(ValueError, match="PDF 读取失败"):
            _parse_with_legacy(target)

    def test_a_pdf_with_no_text_layer_is_reported_as_possibly_scanned(self, tmp_path):
        PdfWriter = pytest.importorskip("pypdf").PdfWriter
        target = tmp_path / "scan.pdf"
        pdf = PdfWriter()
        pdf.add_blank_page(width=200, height=200)
        with open(target, "wb") as handle:
            pdf.write(handle)

        result = _result_with_legacy(target)

        assert result.status == PARSE_PARTIAL
        assert any("扫描件" in warning for warning in result.warnings)


class TestLegacyAdapterDocx:
    def test_paragraphs_carry_a_docx_element_locator(self, tmp_path):
        docx = pytest.importorskip("docx")
        target = tmp_path / "a.docx"
        document = docx.Document()
        document.add_paragraph("First paragraph")
        document.add_paragraph("Second paragraph")
        document.save(str(target))

        parsed = _parse_with_legacy(target)

        assert [element.text for element in parsed.elements] == [
            "First paragraph",
            "Second paragraph",
        ]
        assert all(element.locator.kind == "docx_element" for element in parsed.elements)
        assert all(element.locator.paragraph is not None for element in parsed.elements)

    def test_a_table_stays_a_two_dimensional_grid(self, tmp_path):
        # Flattening to "a | b" here would make cross-column retrieval
        # impossible for the rest of the pipeline, so the grid has to survive.
        docx = pytest.importorskip("docx")
        target = tmp_path / "table.docx"
        document = docx.Document()
        table = document.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "Metric"
        table.cell(0, 1).text = "Value"
        table.cell(1, 0).text = "Accuracy"
        table.cell(1, 1).text = "0.91"
        document.save(str(target))

        parsed = _parse_with_legacy(target)

        table_elements = [element for element in parsed.elements if element.role == "table"]
        assert len(table_elements) == 1
        grid = table_elements[0].table
        assert grid is not None
        assert grid.rows == [["Metric", "Value"], ["Accuracy", "0.91"]]
        assert grid.column_count == 2
        assert "Accuracy" in table_elements[0].text

    def test_a_table_keeps_its_place_in_reading_order(self, tmp_path):
        docx = pytest.importorskip("docx")
        target = tmp_path / "mixed.docx"
        document = docx.Document()
        document.add_paragraph("Before the table")
        table = document.add_table(rows=1, cols=2)
        table.cell(0, 0).text = "A"
        table.cell(0, 1).text = "B"
        document.add_paragraph("After the table")
        document.save(str(target))

        parsed = _parse_with_legacy(target)

        assert [element.role for element in parsed.elements] == [
            "paragraph",
            "table",
            "paragraph",
        ]
        # Reading order is what a reader notices first when it is wrong, so the
        # locators have to increase monotonically across the body.
        indices = [element.locator.paragraph for element in parsed.elements]
        assert indices == sorted(indices)

    def test_headings_build_a_heading_path_for_the_paragraphs_beneath_them(
        self, tmp_path
    ):
        docx = pytest.importorskip("docx")
        target = tmp_path / "headings.docx"
        document = docx.Document()
        document.add_paragraph("Methods", style="Heading 1")
        document.add_paragraph("Setup")
        document.add_paragraph("Datasets", style="Heading 2")
        document.add_paragraph("ImageNet")
        document.save(str(target))

        parsed = _parse_with_legacy(target)

        headings = [element for element in parsed.elements if element.role == "heading"]
        assert [element.text for element in headings] == ["Methods", "Datasets"]
        assert [element.level for element in headings] == [1, 2]

        setup = _element_with_text(parsed, "Setup")
        assert setup.heading_path == ("Methods",)
        imagenet = _element_with_text(parsed, "ImageNet")
        assert imagenet.heading_path == ("Methods", "Datasets")

    def test_a_sibling_heading_replaces_the_path_rather_than_appending_to_it(
        self, tmp_path
    ):
        docx = pytest.importorskip("docx")
        target = tmp_path / "siblings.docx"
        document = docx.Document()
        document.add_paragraph("Methods", style="Heading 1")
        document.add_paragraph("Setup")
        document.add_paragraph("Results", style="Heading 1")
        document.add_paragraph("Accuracy")
        document.save(str(target))

        parsed = _parse_with_legacy(target)

        assert _element_with_text(parsed, "Accuracy").heading_path == ("Results",)

    def test_every_element_points_at_the_unit_that_contains_it(self, tmp_path):
        docx = pytest.importorskip("docx")
        target = tmp_path / "a.docx"
        document = docx.Document()
        document.add_paragraph("First")
        document.add_paragraph("Second")
        document.save(str(target))

        parsed = _parse_with_legacy(target)

        unit_ids = {unit.unit_id for unit in parsed.units}
        assert unit_ids
        assert all(element.unit_id in unit_ids for element in parsed.elements)

    def test_the_full_text_projection_still_contains_the_legacy_prose(
        self, tmp_path
    ):
        # The canonical document has to satisfy the existing analysis prompts,
        # so the prose it projects must not be a regression on the old string.
        docx = pytest.importorskip("docx")
        target = tmp_path / "a.docx"
        document = docx.Document()
        document.add_paragraph("First paragraph")
        document.add_paragraph("Second paragraph")
        document.save(str(target))

        parsed = _parse_with_legacy(target)

        text = parsed.to_full_text()
        assert "First paragraph" in text
        assert "Second paragraph" in text


class TestLegacyAdapterPptx:
    def test_the_slide_title_becomes_a_heading_instead_of_a_body_prefix(self, tmp_path):
        # "[Slide 1]" in the prose makes a slide boundary look like a sentence
        # the author wrote; the title belongs in its own element.
        pptx = pytest.importorskip("pptx")
        target = tmp_path / "a.pptx"
        presentation = pptx.Presentation()
        slide = presentation.slides.add_slide(presentation.slide_layouts[5])
        slide.shapes.title.text = "Slide title"
        slide.notes_slide.notes_text_frame.text = "Remember to breathe"
        presentation.save(str(target))

        parsed = _parse_with_legacy(target)

        headings = [element for element in parsed.elements if element.role == "heading"]
        assert [element.text for element in headings] == ["Slide title"]
        assert headings[0].locator.kind == "slide"
        assert headings[0].locator.slide == 1

    def test_slide_elements_are_located_by_slide_and_shape(self, tmp_path):
        pptx = pytest.importorskip("pptx")
        target = tmp_path / "body.pptx"
        presentation = pptx.Presentation()
        slide = presentation.slides.add_slide(presentation.slide_layouts[5])
        slide.shapes.title.text = "Title"
        box = slide.shapes.add_textbox(0, 0, 100, 100)
        box.text_frame.text = "Body copy"
        presentation.save(str(target))

        parsed = _parse_with_legacy(target)

        body = _element_with_text(parsed, "Body copy")
        assert body.locator.kind == "slide"
        assert body.locator.slide == 1
        assert body.locator.element is not None

    def test_speaker_notes_are_separated_from_the_slide_content(self, tmp_path):
        pptx = pytest.importorskip("pptx")
        target = tmp_path / "notes.pptx"
        presentation = pptx.Presentation()
        slide = presentation.slides.add_slide(presentation.slide_layouts[5])
        slide.shapes.title.text = "Title"
        slide.notes_slide.notes_text_frame.text = "Remember to breathe"
        presentation.save(str(target))

        parsed = _parse_with_legacy(target)

        notes = _element_with_text(parsed, "Remember to breathe")
        assert notes.locator.slide == 1
        assert notes.role == "paragraph"

    def test_the_slide_count_is_recorded_as_the_page_count(self, tmp_path):
        pptx = pytest.importorskip("pptx")
        target = tmp_path / "two.pptx"
        presentation = pptx.Presentation()
        for text in ("First", "Second"):
            slide = presentation.slides.add_slide(presentation.slide_layouts[5])
            slide.shapes.title.text = text
        presentation.save(str(target))

        parsed = _parse_with_legacy(target)

        assert parsed.page_count == 2
        assert parsed.elements[0].locator.slide == 1


class TestLegacyAdapterDispatch:
    def test_rejects_an_unsupported_extension_with_the_legacy_message(self, tmp_path):
        # The message is locked down by tests/test_documents.py, so the adapter
        # has to raise the same words or one of the two suites goes red.
        target = tmp_path / "a.exe"
        target.write_bytes(b"binary")

        with pytest.raises(ValueError, match="不支持的文件格式"):
            _parse_with_legacy(target)

    def test_records_the_file_name_media_type_and_content_hash(self, tmp_path):
        target = tmp_path / "a.txt"
        target.write_text("body", encoding="utf-8")

        document = _parse_with_legacy(target)

        assert document.source_filename == "a.txt"
        assert document.media_type == TXT_MEDIA_TYPE
        assert document.sha256
        assert document.parser_name == LEGACY_PARSER_NAME
        assert document.parser_version == "1.0"

    def test_two_parses_of_the_same_file_report_the_same_hash(self, tmp_path):
        # The parse cache is keyed on this hash, so it has to be a property of
        # the bytes rather than of the run.
        target = tmp_path / "a.txt"
        target.write_text("body", encoding="utf-8")

        first = _parse_with_legacy(target)
        second = _parse_with_legacy(target)

        assert first.sha256 == second.sha256
        assert first.options_hash == second.options_hash


class TestRouterWithTheLegacyAdapter:
    def test_the_real_adapter_drives_end_to_end_through_the_router(self, tmp_path):
        monkey_target = tmp_path / "a.txt"
        monkey_target.write_text("Hello world", encoding="utf-8")
        router = ParserRouter([LegacyDocumentParser()], executor_factory=ImmediateExecutor)

        result = router.parse(str(monkey_target), TXT_MEDIA_TYPE)

        assert result.status == PARSE_SUCCESS
        assert result.parser_name == LEGACY_PARSER_NAME
        assert result.document.elements[0].text == "Hello world"
        assert result.duration_ms >= 0

    def test_a_corrupt_file_becomes_a_failed_result_rather_than_an_exception(
        self, tmp_path
    ):
        target = tmp_path / "broken.pdf"
        target.write_bytes(b"%PDF-1.7\nnot really a pdf")
        router = ParserRouter([LegacyDocumentParser()], executor_factory=ImmediateExecutor)

        result = router.parse(str(target), PDF_MEDIA_TYPE)

        assert result.status == PARSE_FAILED
        assert "PDF 读取失败" in result.error_message

    def test_the_default_registry_contains_the_legacy_parser(self):
        # The rollback path depends on this adapter being present without any
        # optional dependency installed. It is asserted as *last* rather than
        # as the only entry: the registry is allowed to grow, but the adapter
        # that must work for all of Phase 0 has to keep the final word.
        from paperwhisperer.ingestion.gateway import build_default_parsers

        names = [parser.name for parser in build_default_parsers()]

        assert names[-1] == LEGACY_PARSER_NAME

    def test_a_backend_that_cannot_run_here_is_left_out_of_the_registry(
        self, monkeypatch
    ):
        # Registering a parser that can only fail would add a skipped attempt
        # to every document's diagnostics and teach the log to expect noise.
        from paperwhisperer.ingestion.gateway import build_default_parsers

        monkeypatch.setattr(config, "DOCUMENT_PRIVACY_MODE", "local_only")
        monkeypatch.setattr(config, "MINERU_API_TOKEN", "")

        names = [parser.name for parser in build_default_parsers()]

        assert "mineru_cloud" not in names

    def test_a_configured_cloud_token_puts_the_cloud_parser_in_the_registry(
        self, monkeypatch
    ):
        from paperwhisperer.ingestion.gateway import build_default_parsers

        monkeypatch.setattr(config, "DOCUMENT_PRIVACY_MODE", "allow_cloud")
        monkeypatch.setattr(config, "MINERU_API_TOKEN", "a-token")

        names = [parser.name for parser in build_default_parsers()]

        assert "mineru_cloud" in names
        # Still last-but-one: a local parse is preferred over a paid upload
        # whenever the configuration does not say otherwise.
        assert names.index("mineru_cloud") < names.index(LEGACY_PARSER_NAME)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _parse_with_legacy(path) -> ParsedDocument:
    """Parse ``path`` with the legacy adapter and return the document."""
    return _result_with_legacy(path).document


def _result_with_legacy(path) -> ParseResult:
    parser = LegacyDocumentParser()
    return parser.parse(str(path), ParseOptions())


def _element_with_text(document: ParsedDocument, text: str):
    for element in document.elements:
        if element.text == text:
            return element
    raise AssertionError(f"no element with text {text!r}")


class TestQualityVerdictIsVisible:
    """A quality FAIL on a usable parse must reach the reader as a warning.

    The silent-truncation gate exists for the case where the analysis
    confidently summarises half a paper and nothing says so. A verdict that is
    only *attached* (``result.quality``) and never turned into a visible warning
    leaves exactly that case open on the product path, so this pins the
    warning on both surfaces a reader or a diagnostic can inspect:
    ``ParsedDocument.warnings`` and ``ParseResult.warnings``.
    """

    @staticmethod
    def _failing_report(rule_id: str = "text_density"):
        class _Report:
            failed_rules = (rule_id,)
            warned_rules = ()

        return _Report()

    def test_a_failing_quality_report_surfaces_a_document_warning(self, monkeypatch):
        from paperwhisperer.ingestion import quality as quality_module

        monkeypatch.setattr(
            quality_module, "evaluate_quality", lambda document: self._failing_report()
        )

        result = success_result("body")
        router = ParserRouter([])
        router._attach_quality(result)

        document_warnings = result.document.warnings
        assert any("质量门禁" in w and "text_density" in w for w in document_warnings), (
            f"the quality FAIL did not reach the document's warnings: {document_warnings}"
        )
        assert any("质量门禁" in w for w in result.warnings), (
            f"the quality FAIL did not reach the result's warnings: {result.warnings}"
        )

    def test_a_clean_quality_report_adds_no_warning(self, monkeypatch):
        class _Report:
            failed_rules = ()
            warned_rules = ()

        from paperwhisperer.ingestion import quality as quality_module

        monkeypatch.setattr(quality_module, "evaluate_quality", lambda document: _Report())

        result = success_result("body")
        router = ParserRouter([])
        before_doc = list(result.document.warnings)
        before_res = list(result.warnings)
        router._attach_quality(result)
        assert result.document.warnings == before_doc
        assert result.warnings == before_res
