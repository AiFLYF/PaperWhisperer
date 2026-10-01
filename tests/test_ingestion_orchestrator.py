"""Fallback orchestration: the chain, the privacy gate, and the parse cache.

The orchestrator is the layer a user is finally accountable to, so these tests
are organised around the promises *it* makes rather than around its classes:

* a clean document stops the chain immediately — no retry, no cloud, no cost;
* a degraded one is repaired locally before anything is uploaded anywhere;
* a refused cloud fallback is a *recorded fact*, not a missing entry;
* ``local_only`` provably makes zero network calls, in both the injected and
  the default wiring;
* a cache hit never re-invokes a parser, which is the only thing standing
  between a user and a second bill for the same document;
* and when two documents compete, exactly one of them is returned whole.

The last one deserves a note, because "whole" is the whole point. A page-level
best-of would produce a document with two reading orders and two locator
namespaces, and a citation built on it would resolve to the wrong place exactly
where the reader clicks. The tests below therefore assert on *which* document
came back — including asserting that the losing parser's text is nowhere in the
winner — rather than only on the winner's quality.

Fakes are local to this file for the reason the gateway tests give theirs:
a parser double that outlives its test ends up encoding assumptions nobody
remembers making. No test here touches the network, docling or a real clock —
``ImmediateExecutor`` runs every parser synchronously so a timeout can never
fire by accident and a stray sleep cannot make the suite flaky.
"""

from __future__ import annotations

import os
import socket
import urllib.request
from dataclasses import dataclass
from typing import Any

import pytest

from paperwhisperer.core import config
from paperwhisperer.ingestion import orchestrator as orch
from paperwhisperer.ingestion.adapters.docling_adapter import normalize_docling_document
from paperwhisperer.ingestion.gateway import (
    PARSE_PARTIAL,
    ParseOptions,
    ParserError,
    ParseResult,
    ParserRouter,
)
from paperwhisperer.ingestion.orchestrator import (
    ATTEMPT_SKIPPED,
    STAGE_CLOUD,
    STAGE_FORCE_OCR,
    STAGE_PRIMARY,
    CloudFallbackDecision,
    ParseCache,
    ParseCacheKey,
    ParseOutcome,
    parse_document,
)
from paperwhisperer.ingestion.quality import QualityStatus
from paperwhisperer.ingestion.schema import (
    LOCATOR_KINDS,
    Element,
    ParsedDocument,
    SourceLocator,
)

PDF = "application/pdf"
TXT = "text/plain"
PPTX = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

ALLOW_CLOUD = CloudFallbackDecision(allowed=True, reason_code="allowed", message="")


# --------------------------------------------------------------------------
# Document builders
# --------------------------------------------------------------------------


def element(text: str, page: int | None = 1) -> Element:
    """A paragraph on a real page, or on no page at all when ``page`` is None.

    A missing page is the interesting case: an adapter that lost page
    boundaries produces exactly this, and the locator-coverage rule is what
    turns it into a FAIL.
    """
    locator = SourceLocator(kind="pdf_page", page=page) if page else SourceLocator()
    return Element(role="paragraph", text=text, locator=locator)


def clean_document(pages: int = 1, source: str = "a.pdf") -> ParsedDocument:
    """A document the quality gate passes: located, dense enough, one page."""
    return ParsedDocument(
        source_filename=source,
        media_type=PDF,
        page_count=pages,
        elements=[element("文" * 200, page=index + 1) for index in range(pages)],
    )


def broken_document(source: str = "a.pdf", pages: int = 2) -> ParsedDocument:
    """A document that FAILs the gate: a lost text layer and a lost page."""
    return ParsedDocument(
        source_filename=source,
        media_type=PDF,
        page_count=pages,
        elements=[element("文" * 200, page=1), element("文" * 200, page=None)],
    )


# --------------------------------------------------------------------------
# Parser doubles
# --------------------------------------------------------------------------


class FakeParser:
    """A local parser double whose every observable the test controls.

    ``outcome`` is a ``ParseResult``, an exception to raise, or a callable
    ``(file_path, options)`` so a test can key behaviour on ``force_ocr`` — the
    only way to tell the repair stage from the primary one.
    """

    def __init__(
        self,
        name: str = "docling",
        *,
        version: str = "1.0",
        outcome: Any = None,
        media_types: tuple[str, ...] = (PDF,),
    ) -> None:
        self.name = name
        self.version = version
        self.outcome = outcome
        self.media_types = tuple(media_types)
        self.calls: list[dict[str, Any]] = []

    def supports(self, media_type: str) -> bool:
        return media_type in self.media_types

    def parse(self, file_path, options) -> ParseResult:
        self.calls.append({"file_path": file_path, "force_ocr": bool(options.force_ocr)})
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        if callable(self.outcome):
            return self.outcome(file_path, options)
        return self.outcome


@dataclass
class FakeAuditRecord:
    """The cloud adapter's audit record, in the shape the bridge reads."""

    outcome: str = "success"
    message: str = ""
    operator: str = "tester"
    timestamp: str = "2026-01-01T00:00:00+00:00"
    source_filename: str = "a.pdf"
    sha256: str = ""
    trigger_reason: str = "fallback"


class FakeCloudParser:
    """A cloud parser double with the shipped adapter's return shape.

    It deliberately returns ``(document | None, record)`` rather than a
    ``ParseResult``, because that is what the orchestrator has to bridge in
    production. A double that returned the gateway's own type would let the
    bridge rot unnoticed.
    """

    def __init__(
        self,
        *,
        document: ParsedDocument | None = None,
        outcome: str = "success",
        message: str = "",
        raise_exc: BaseException | None = None,
        name: str = "mineru_cloud",
        version: str = "1.0.0",
        media_types: tuple[str, ...] = (PDF,),
        raw_result: Any = None,
    ) -> None:
        self.name = name
        self.version = version
        self.document = document
        self.outcome = outcome
        self.message = message
        self.raise_exc = raise_exc
        self.media_types = tuple(media_types)
        self.raw_result = raw_result
        self.calls: list[dict[str, Any]] = []

    def supports(self, media_type: str) -> bool:
        return media_type in self.media_types

    def parse(self, file_path, options):
        self.calls.append({"file_path": file_path, "options": options})
        if self.raise_exc is not None:
            raise self.raise_exc
        if self.raw_result is not None:
            return self.raw_result
        record = FakeAuditRecord(outcome=self.outcome, message=self.message)
        return (self.document, record) if self.document is not None else (None, record)


class ImmediateExecutor:
    """Runs submitted work synchronously; never really times out.

    Every test that is not *about* the timeout uses this, so a hung parser
    cannot be confused with a slow machine and no thread outlives a test.
    """

    def submit(self, fn, /, *args, **kwargs):
        import concurrent.futures

        future: concurrent.futures.Future = concurrent.futures.Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # noqa: BLE001 - mirrors a real future
            future.set_exception(exc)
        return future

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        return None


def router_for(*parsers: FakeParser) -> ParserRouter:
    return ParserRouter(list(parsers), executor_factory=ImmediateExecutor)


def success(document: ParsedDocument) -> ParseResult:
    return ParseResult.from_document(document)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def gateway_wiring(monkeypatch):
    """Pin the configuration every test reasons about.

    ``PARSER_BACKEND`` ships as ``legacy``, which collapses the router's
    registry to a parser named ``legacy`` and would make every orchestration
    test assert against a configuration nobody deploys. The privacy settings
    default to the safest combination, so a test that forgets to set one is
    still running with zero egress available to it.
    """
    monkeypatch.setattr(config, "PARSER_BACKEND", "gateway")
    monkeypatch.setattr(config, "PARSER_PRIMARY", "docling")
    monkeypatch.setattr(config, "PARSER_FALLBACK_CHAIN", ())
    monkeypatch.setattr(config, "PARSER_FORCE_OCR", False)
    monkeypatch.setattr(config, "DOCUMENT_PRIVACY_MODE", "local_only")
    monkeypatch.setattr(config, "DOCUMENT_CLOUD_CONSENT_GRANTED", False)
    monkeypatch.setattr(config, "PARSER_CACHE_ENABLED", True)
    monkeypatch.setattr(config, "PARSER_CACHE_DIR", "")


@pytest.fixture
def no_network(monkeypatch):
    """A tripwire on every egress door, plus a record of any that opened.

    This is the evidence behind the "local-only cloud egress: 0" release gate.
    It is deliberately paranoid and deliberately layered: ``urlopen`` is what
    the adapter actually calls, ``socket.create_connection`` is what urllib
    resolves underneath it, and ``socket.socket`` catches anything that skips
    both. Each one appends to the returned list *before* raising, so a test can
    assert both that the call never happened and that it would have been
    caught.
    """
    seen: list[str] = []

    def tripwire(label: str):
        def _fn(*args, **kwargs):
            seen.append(label)
            raise AssertionError(f"测试中发生了网络访问：{label}；local_only 模式绝不允许出网。")

        return _fn

    monkeypatch.setattr(urllib.request, "urlopen", tripwire("urllib.request.urlopen"))
    monkeypatch.setattr(socket, "create_connection", tripwire("socket.create_connection"))
    monkeypatch.setattr(socket, "socket", tripwire("socket.socket"))
    monkeypatch.setattr(socket, "getaddrinfo", tripwire("socket.getaddrinfo"))
    return seen


@pytest.fixture
def pdf(tmp_path):
    """A real file on disk; the orchestrator hashes bytes and stats the path."""
    path = tmp_path / "a.pdf"
    path.write_bytes(b"%PDF-1.7 fake document bytes")
    return str(path)


@pytest.fixture
def cache_dir(tmp_path):
    return str(tmp_path / "parse_cache")


def stage(outcome: ParseOutcome, name: str) -> Any:
    """The single attempt recorded for one stage."""
    matches = [attempt for attempt in outcome.attempts if attempt.stage == name]
    assert len(matches) == 1, f"stage {name} 应当恰好记录一次，实际 {len(matches)} 次"
    return matches[0]


# --------------------------------------------------------------------------
# Stage 1: a clean document stops everything
# --------------------------------------------------------------------------


def test_clean_primary_parse_stops_the_chain(pdf, cache_dir):
    parser = FakeParser(outcome=success(clean_document()))
    outcome = parse_document(
        pdf,
        router=router_for(parser),
        cache=ParseCache(cache_dir),
        options=ParseOptions(force_ocr=False),
    )

    assert outcome.status == "success"
    assert outcome.is_usable
    assert outcome.parsed_by == "docling"
    assert outcome.quality.status is QualityStatus.PASS
    assert outcome.fallback_used is False
    assert outcome.used_cloud is False
    assert outcome.cloud_attempted is False
    assert outcome.touched_network is False
    assert len(parser.calls) == 1, "干净文档只应解析一次"
    assert stage(outcome, STAGE_FORCE_OCR).skipped_reason == "quality_passed"
    assert stage(outcome, STAGE_CLOUD).skipped_reason == "quality_sufficient"
    assert stage(outcome, STAGE_CLOUD).attempted is False


def test_clean_parse_never_asks_the_privacy_gate(pdf, cache_dir):
    """The cloud is not merely refused under ``local_only``; it is not raised.

    A gate consulted on every document would be a policy check on a path that
    cannot reach the network, which is exactly the kind of coupling that makes
    the local-only guarantee hard to audit later.
    """
    asked: list[bool] = []
    outcome = parse_document(
        pdf,
        router=router_for(FakeParser(outcome=success(clean_document()))),
        cache=ParseCache(cache_dir),
        options=ParseOptions(force_ocr=False),
        privacy_gate=lambda: asked.append(True) or ALLOW_CLOUD,
    )

    assert asked == []
    assert outcome.used_cloud is False


def test_partial_parse_is_usable_and_surfaced(pdf, cache_dir):
    """``partial`` ends the chain but keeps its warnings.

    Refusing a partial parse would throw away a scanned-but-mostly-readable
    PDF a human can still work with; hiding the warnings would be the silent
    truncation the release gates forbid.
    """
    parser = FakeParser(
        outcome=ParseResult(status=PARSE_PARTIAL, document=clean_document()),
    )
    outcome = parse_document(
        pdf,
        router=router_for(parser),
        cache=ParseCache(cache_dir),
        options=ParseOptions(force_ocr=False),
    )

    assert outcome.is_usable
    assert outcome.status == PARSE_PARTIAL
    assert len(parser.calls) == 1
    assert any("不完整" in warning for warning in outcome.warnings)


# --------------------------------------------------------------------------
# Stage 2: local repair
# --------------------------------------------------------------------------


def test_degraded_quality_triggers_a_force_ocr_retry(pdf, cache_dir):
    """A FAIL is repaired locally first, and the repair wins.

    The retry is a *different parse of the whole document*, not a patch: the
    winner is the OCR document and none of the primary's unlocated text
    survives into it.
    """
    primary_doc = broken_document()
    repaired_doc = clean_document(pages=2)

    def outcome_for(_path, options):
        if options.force_ocr:
            return success(repaired_doc)
        return success(primary_doc)

    parser = FakeParser(outcome=outcome_for)
    result = parse_document(
        pdf,
        router=router_for(parser),
        cache=ParseCache(cache_dir),
        options=ParseOptions(force_ocr=False),
    )

    assert [call["force_ocr"] for call in parser.calls] == [False, True]
    assert result.quality.status is QualityStatus.PASS
    assert result.fallback_used is True
    assert result.parsed_by == "docling"
    assert stage(result, STAGE_PRIMARY).quality_status == QualityStatus.FAIL.value
    assert stage(result, STAGE_FORCE_OCR).attempted is True


def test_failed_primary_is_retried_with_force_ocr(pdf, cache_dir):
    """A primary that produced nothing is exactly the case OCR rescues."""
    repaired_doc = clean_document()

    def outcome_for(_path, options):
        if options.force_ocr:
            return success(repaired_doc)
        raise ParserError("PDF 读取失败: 文件没有可提取的文本层。")

    parser = FakeParser(outcome=outcome_for)
    result = parse_document(
        pdf,
        router=router_for(parser),
        cache=ParseCache(cache_dir),
        options=ParseOptions(force_ocr=False),
    )

    assert result.is_usable
    assert [call["force_ocr"] for call in parser.calls] == [False, True]
    assert stage(result, STAGE_PRIMARY).status == "failed"


def test_the_two_stages_use_different_option_hashes(pdf, cache_dir):
    """Force OCR must never be served from the non-OCR cache entry.

    The options hash is the only thing standing between "the repair ran" and
    "the repair silently returned the thing it was meant to fix", so it is
    asserted directly rather than inferred from call counts.
    """
    parser = FakeParser(outcome=success(broken_document()))
    result = parse_document(
        pdf,
        router=router_for(parser),
        cache=ParseCache(cache_dir),
        options=ParseOptions(force_ocr=False),
    )

    primary = stage(result, STAGE_PRIMARY)
    repair = stage(result, STAGE_FORCE_OCR)
    assert primary.options_hash and repair.options_hash
    assert primary.options_hash != repair.options_hash


def test_ocr_retry_can_be_disabled_by_the_caller(pdf, cache_dir):
    parser = FakeParser(outcome=success(broken_document()))
    result = parse_document(
        pdf,
        router=router_for(parser),
        cache=ParseCache(cache_dir),
        options=ParseOptions(force_ocr=False),
        force_ocr_retry=False,
    )

    assert len(parser.calls) == 1
    assert stage(result, STAGE_FORCE_OCR).skipped_reason == "ocr_retry_disabled"
    assert stage(result, STAGE_FORCE_OCR).status == ATTEMPT_SKIPPED


def test_ocr_retry_is_skipped_when_ocr_is_already_on(pdf, cache_dir):
    """Re-running an identical call is a duplicate, not a repair."""
    parser = FakeParser(outcome=success(broken_document()))
    result = parse_document(
        pdf,
        router=router_for(parser),
        cache=ParseCache(cache_dir),
        options=ParseOptions(force_ocr=True),
    )

    assert len(parser.calls) == 1
    assert stage(result, STAGE_FORCE_OCR).skipped_reason == "ocr_already_enabled"


def test_ocr_retry_is_skipped_for_a_format_that_cannot_be_scanned(tmp_path, cache_dir):
    """Asking a TXT parser to OCR is a second full parse of the same text."""
    path = tmp_path / "a.txt"
    path.write_text("plain text body " * 40, encoding="utf-8")
    parser = FakeParser(
        outcome=success(
            ParsedDocument(
                source_filename="a.txt",
                media_type=TXT,
                elements=[element("文本" * 200, page=None)],
            )
        ),
        media_types=(TXT,),
    )
    result = parse_document(
        str(path),
        router=router_for(parser),
        cache=ParseCache(cache_dir),
        options=ParseOptions(force_ocr=False),
    )

    assert len(parser.calls) == 1
    assert stage(result, STAGE_FORCE_OCR).skipped_reason == "media_type_not_ocr_candidate"


# --------------------------------------------------------------------------
# Stage 3: cloud fallback
# --------------------------------------------------------------------------


def test_primary_failure_falls_back_to_cloud_when_policy_allows(pdf, cache_dir):
    """The whole point of the chain: a local dead end is not a user dead end."""
    cloud = FakeCloudParser(document=clean_document())
    parser = FakeParser(outcome=ParserError("PDF 读取失败: 没有文本层。"))
    result = parse_document(
        pdf,
        router=router_for(parser),
        cache=ParseCache(cache_dir),
        options=ParseOptions(force_ocr=False),
        cloud_parser=cloud,
        privacy_gate=lambda: ALLOW_CLOUD,
    )

    assert result.is_usable
    assert result.used_cloud is True
    assert result.cloud_attempted is True
    assert result.touched_network is True
    assert result.parsed_by == "mineru_cloud"
    assert "已上传至云端" in result.summary_line()
    assert len(cloud.calls) == 1
    assert stage(result, STAGE_CLOUD).network is True


def test_allow_cloud_fallback_false_skips_the_gate_entirely(pdf, cache_dir):
    """A deployment can forbid the cloud without touching the mode setting."""
    asked: list[bool] = []
    cloud = FakeCloudParser(document=clean_document())
    result = parse_document(
        pdf,
        router=router_for(FakeParser(outcome=ParserError("没有文本层。"))),
        cache=ParseCache(cache_dir),
        options=ParseOptions(force_ocr=False),
        cloud_parser=cloud,
        allow_cloud_fallback=False,
        privacy_gate=lambda: asked.append(True) or ALLOW_CLOUD,
    )

    assert asked == [], "被禁用的云端回退不应再询问隐私门控"
    assert result.used_cloud is False
    assert cloud.calls == []
    assert stage(result, STAGE_CLOUD).skipped_reason == "cloud_fallback_disabled"


# --------------------------------------------------------------------------
# The local-only release gate
# --------------------------------------------------------------------------


def test_local_only_refuses_cloud_and_makes_zero_network_calls(
    pdf, cache_dir, monkeypatch, no_network
):
    """The P0 gate, proved on the real privacy gate rather than a stub.

    Nothing about the decision is injected here: ``DOCUMENT_PRIVACY_MODE`` is
    left at ``local_only`` and the shipped ``mineru_cloud.evaluate_privacy_gate``
    reads it. The cloud parser double would fail the test if called, and the
    network tripwire covers the case where something other than the cloud
    parser reaches for a socket.
    """
    from paperwhisperer.ingestion.adapters import mineru_cloud

    cloud = FakeCloudParser(document=clean_document())
    monkeypatch.setattr(
        mineru_cloud,
        "parse_document",
        lambda *a, **k: pytest.fail("local_only 模式下不得调用云端解析链路。"),
    )
    monkeypatch.setattr(
        mineru_cloud,
        "request_upload_urls",
        lambda *a, **k: pytest.fail("local_only 模式下不得请求云端上传地址。"),
    )

    result = parse_document(
        pdf,
        router=router_for(FakeParser(outcome=ParserError("没有文本层。"))),
        cache=ParseCache(cache_dir),
        options=ParseOptions(force_ocr=False),
        cloud_parser=cloud,
    )

    assert no_network == [], "local_only 模式下发生了出网"
    assert cloud.calls == [], "云端解析器被调用了"
    assert result.cloud_attempted is False
    assert result.touched_network is False
    assert result.used_cloud is False
    assert stage(result, STAGE_CLOUD).skipped_reason == "privacy_mode_local_only"
    assert result.cloud_refusal is not None
    assert result.cloud_refusal.reason_code == "privacy_mode_local_only"
    assert "local_only" in result.cloud_refusal.message


def test_local_only_proves_zero_egress_on_the_default_wiring(
    pdf, cache_dir, monkeypatch, no_network
):
    """Same guarantee with no cloud parser injected at all.

    This is the path production takes. If the gate were only honoured because a
    test happened to pass a double, a wiring mistake — a default parser built
    before the gate is consulted — would not be caught here.
    """
    from paperwhisperer.ingestion.adapters import mineru_cloud

    monkeypatch.setattr(
        mineru_cloud,
        "parse_document",
        lambda *a, **k: pytest.fail("默认接线也不得在 local_only 下调用云端解析。"),
    )
    monkeypatch.setattr(
        mineru_cloud,
        "MineruCloudParser",
        lambda *a, **k: pytest.fail("local_only 下不应构造云端解析器。"),
    )

    result = parse_document(
        pdf,
        router=router_for(FakeParser(outcome=ParserError("没有文本层。"))),
        cache=ParseCache(cache_dir),
        options=ParseOptions(force_ocr=False),
    )

    assert no_network == []
    assert result.cloud_attempted is False
    assert stage(result, STAGE_CLOUD).skipped_reason == "privacy_mode_local_only"


def test_consent_not_granted_refuses_cloud_with_its_own_reason(
    pdf, cache_dir, monkeypatch, no_network
):
    """Policy-allowed but user-unconsented is a *different* refusal.

    The two refusals send people to two different places to fix something: the
    operator changes the mode, or the user flips a switch. Collapsing them
    would make a user change a setting that was never the problem.
    """
    monkeypatch.setattr(config, "DOCUMENT_PRIVACY_MODE", "allow_cloud")
    monkeypatch.setattr(config, "DOCUMENT_CLOUD_CONSENT_GRANTED", False)

    cloud = FakeCloudParser(document=clean_document())
    result = parse_document(
        pdf,
        router=router_for(FakeParser(outcome=ParserError("没有文本层。"))),
        cache=ParseCache(cache_dir),
        options=ParseOptions(force_ocr=False),
        cloud_parser=cloud,
    )

    assert no_network == []
    assert cloud.calls == []
    assert stage(result, STAGE_CLOUD).skipped_reason == "consent_not_granted"
    assert result.cloud_refusal.reason_code == "consent_not_granted"
    assert result.used_cloud is False


def test_a_raising_privacy_gate_is_treated_as_a_refusal(pdf, cache_dir, no_network):
    """Fail closed. A gate that cannot answer must not answer "yes"."""

    def broken_gate() -> CloudFallbackDecision:
        raise RuntimeError("策略服务不可用")

    cloud = FakeCloudParser(document=clean_document())
    result = parse_document(
        pdf,
        router=router_for(FakeParser(outcome=ParserError("没有文本层。"))),
        cache=ParseCache(cache_dir),
        options=ParseOptions(force_ocr=False),
        cloud_parser=cloud,
        privacy_gate=broken_gate,
    )

    assert no_network == []
    assert cloud.calls == []
    assert stage(result, STAGE_CLOUD).skipped_reason == "privacy_gate_error"
    assert result.cloud_attempted is False


# --------------------------------------------------------------------------
# The cache: never pay twice
# --------------------------------------------------------------------------


def test_cache_hit_does_not_re_invoke_the_parser(pdf, cache_dir):
    parser = FakeParser(outcome=success(clean_document()))
    cache = ParseCache(cache_dir)
    options = ParseOptions(force_ocr=False)

    first = parse_document(pdf, router=router_for(parser), cache=cache, options=options)
    second = parse_document(pdf, router=router_for(parser), cache=cache, options=options)

    assert len(parser.calls) == 1, "同样的字节不应被解析第二次"
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.is_usable
    assert second.parsed_by == "docling"
    assert "来自缓存" in second.summary_line()
    assert stage(second, STAGE_PRIMARY).cache_hit is True


def test_cache_hit_survives_a_fresh_cache_object(pdf, cache_dir):
    """The directory is the durable store, not the in-process mirror.

    A cache that only worked within one process would help a retry inside a
    single request and nothing else; paying twice after a restart is the case
    that actually costs money.
    """
    parser = FakeParser(outcome=success(clean_document()))
    options = ParseOptions(force_ocr=False)

    parse_document(pdf, router=router_for(parser), cache=ParseCache(cache_dir), options=options)
    result = parse_document(
        pdf, router=router_for(parser), cache=ParseCache(cache_dir), options=options
    )

    assert len(parser.calls) == 1
    assert result.cache_hit is True
    assert result.document.elements  # the document is restored, not just flagged


def test_a_cache_hit_never_pays_for_a_second_cloud_parse(pdf, cache_dir):
    """The acceptance criterion: 缓存命中不重复付费解析.

    A cloud fallback is the expensive case — a second call is a second bill —
    so it is the one that must be proven rather than assumed.
    """
    cloud = FakeCloudParser(document=clean_document())
    parser = FakeParser(outcome=ParserError("没有文本层。"))
    options = ParseOptions(force_ocr=False)

    def run() -> ParseOutcome:
        return parse_document(
            pdf,
            router=router_for(parser),
            # A fresh cache object per call: the second call must be served by
            # the directory, not by a memory mirror left over from the first.
            cache=ParseCache(cache_dir),
            options=options,
            cloud_parser=cloud,
            privacy_gate=lambda: ALLOW_CLOUD,
        )

    first = run()
    second = run()

    assert first.used_cloud is True and first.cache_hit is False
    assert second.cache_hit is True
    assert second.used_cloud is True
    assert len(cloud.calls) == 1, "缓存命中后不得再次调用云端解析"
    # The local parser ran again on the second call, and that is correct: it
    # *failed*, and failures are never cached (see the next test). The bill is
    # for the cloud, and the bill did not happen twice.
    assert len(parser.calls) == 4
    assert cache_entry_count(cache_dir) == 1


def test_different_bytes_miss_the_cache(tmp_path, cache_dir):
    parser = FakeParser(outcome=success(clean_document()))
    cache = ParseCache(cache_dir)
    options = ParseOptions(force_ocr=False)

    first_path = tmp_path / "a.pdf"
    first_path.write_bytes(b"%PDF-1.7 first")
    second_path = tmp_path / "b.pdf"
    second_path.write_bytes(b"%PDF-1.7 second, different bytes entirely")

    parse_document(str(first_path), router=router_for(parser), cache=cache, options=options)
    second = parse_document(
        str(second_path), router=router_for(parser), cache=cache, options=options
    )

    assert len(parser.calls) == 2, "不同的字节必须是不同的缓存条目"
    assert second.cache_hit is False


def test_the_same_bytes_under_a_new_name_is_a_hit(tmp_path, cache_dir):
    """The key is the content, not the path.

    Re-uploading a failed file under a corrected name is the most common way a
    user meets the same bytes twice, and making that a miss means paying twice
    for one document.
    """
    parser = FakeParser(outcome=success(clean_document()))
    cache = ParseCache(cache_dir)
    options = ParseOptions(force_ocr=False)

    first = tmp_path / "scan.pdf"
    first.write_bytes(b"%PDF-1.7 identical")
    second = tmp_path / "scan_final.pdf"
    second.write_bytes(b"%PDF-1.7 identical")

    parse_document(str(first), router=router_for(parser), cache=cache, options=options)
    result = parse_document(
        str(second), router=router_for(parser), cache=cache, options=options
    )

    assert len(parser.calls) == 1
    assert result.cache_hit is True


def test_a_new_parser_version_does_not_reuse_the_old_entry(pdf, cache_dir):
    """Upgrading Docling must not serve yesterday's layout forever."""
    cache = ParseCache(cache_dir)
    options = ParseOptions(force_ocr=False)

    parse_document(
        pdf,
        router=router_for(FakeParser(version="1.0", outcome=success(clean_document()))),
        cache=cache,
        options=options,
    )
    upgraded = FakeParser(version="2.0", outcome=success(clean_document()))
    result = parse_document(pdf, router=router_for(upgraded), cache=cache, options=options)

    assert len(upgraded.calls) == 1
    assert result.cache_hit is False
    assert cache_entry_count(cache_dir) == 2


def test_a_corrupt_cache_entry_is_a_miss_not_a_crash(pdf, cache_dir):
    """A truncated file must cost one parse, never the whole request."""
    cache = ParseCache(cache_dir)
    options = ParseOptions(force_ocr=False)
    parser = FakeParser(outcome=success(clean_document()))

    parse_document(pdf, router=router_for(parser), cache=cache, options=options)
    corrupt_every_entry(cache_dir)

    fresh_cache = ParseCache(cache_dir)
    result = parse_document(pdf, router=router_for(parser), cache=fresh_cache, options=options)

    assert result.is_usable
    assert result.cache_hit is False
    assert len(parser.calls) == 2


def test_failures_are_never_cached(tmp_path, cache_dir):
    """A cached failure would pin a transient outage to the document forever."""
    cache = ParseCache(cache_dir)
    options = ParseOptions(force_ocr=False)
    good = tmp_path / "good.pdf"
    good.write_bytes(b"%PDF-1.7 good")
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.7 different bytes")

    parse_document(
        str(good), router=router_for(FakeParser(outcome=success(clean_document()))),
        cache=cache, options=options,
    )
    cache.clear()
    result = parse_document(
        str(bad),
        router=router_for(FakeParser(outcome=ParserError("没有文本层。"))),
        cache=cache,
        options=options,
    )

    assert not result.is_usable
    assert cache_entry_count(cache_dir) == 1, "失败不应写入缓存"


# --------------------------------------------------------------------------
# Whole-document best-of, never a page-level chimera
# --------------------------------------------------------------------------


def test_the_winner_is_one_whole_document_with_no_stitched_pages(pdf, cache_dir):
    """Two parsers competed; exactly one's text is in the result.

    This is the invariant the Evidence-first design rests on. A merged document
    would carry one parser's page-1 text and another's pages 2-9, with two
    incompatible locator namespaces and no single reading order — and every
    citation into it would be a guess.
    """
    primary_text = "旧解析器的第一页文字。" * 20
    repaired_text = "OCR 修正后的第一页文字。" * 20
    second_page_text = "OCR 修正后的第二页文字。" * 20

    # The primary run reads page 1 but loses the page boundary on page 2, so
    # the quality gate FAILs it and the repair stage gets its turn. Page 1's
    # text is deliberately different between the two runs: if the orchestrator
    # ever stitched at page level, the losing parser's text would show up here.
    primary_doc = ParsedDocument(
        source_filename="a.pdf",
        media_type=PDF,
        page_count=2,
        elements=[element(primary_text, page=1), element(primary_text, page=None)],
    )
    repaired_doc = ParsedDocument(
        source_filename="a.pdf",
        media_type=PDF,
        page_count=2,
        elements=[element(repaired_text, page=1), element(second_page_text, page=2)],
    )

    def outcome_for(_path, options):
        return success(repaired_doc if options.force_ocr else primary_doc)

    result = parse_document(
        pdf,
        router=router_for(FakeParser(outcome=outcome_for)),
        cache=ParseCache(cache_dir),
        options=ParseOptions(force_ocr=False),
    )

    text = result.document.to_full_text()
    assert "OCR 修正后的第一页文字。" in text
    assert "OCR 修正后的第二页文字。" in text
    assert "旧解析器的第一页文字。" not in text, "不得把落败解析器的页面拼接进结果"
    assert len(result.document.elements) == len(repaired_doc.elements)
    assert [item.element_id for item in result.document.elements] == [
        item.element_id for item in repaired_doc.elements
    ]
    assert result.parsed_by == "docling"


def test_quality_outranks_text_volume_when_choosing_a_whole_document(pdf, cache_dir):
    """More recovered text does not buy a document whose citations break.

    The local parse yields twice the characters and cannot locate a single
    element; the cloud parse yields less and can. Quality is compared before
    size, and this is the test that says so.
    """
    huge_but_unlocatable = ParsedDocument(
        source_filename="a.pdf",
        media_type=PDF,
        page_count=1,
        elements=[element("无法定位的文字。" * 400, page=None)],
    )
    small_but_located = clean_document()

    parser = FakeParser(
        outcome=lambda _p, _o: success(huge_but_unlocatable),
    )
    cloud = FakeCloudParser(document=small_but_located)
    result = parse_document(
        pdf,
        router=router_for(parser),
        cache=ParseCache(cache_dir),
        options=ParseOptions(force_ocr=False),
        cloud_parser=cloud,
        privacy_gate=lambda: ALLOW_CLOUD,
    )

    assert result.used_cloud is True
    assert result.quality.status is QualityStatus.PASS
    assert result.document.char_count < huge_but_unlocatable.char_count


def test_a_tie_keeps_the_free_primary_parse(pdf, cache_dir):
    """Equal documents must not cost money to compare."""
    cloud = FakeCloudParser(document=clean_document())
    parser = FakeParser(outcome=success(clean_document()))
    result = parse_document(
        pdf,
        router=router_for(parser),
        cache=ParseCache(cache_dir),
        options=ParseOptions(force_ocr=False),
        cloud_parser=cloud,
        privacy_gate=lambda: ALLOW_CLOUD,
        force_ocr_retry=False,
        quality_thresholds=_lenient_thresholds(),
    )

    assert result.used_cloud is False
    assert result.parsed_by == "docling"


# --------------------------------------------------------------------------
# Failure
# --------------------------------------------------------------------------


def test_every_parser_failing_returns_a_clear_message_and_no_document(pdf, cache_dir):
    """No silent empty document, and a message that says what was tried."""
    cloud = FakeCloudParser(
        document=None,
        outcome="failed",
        message="MinerU 额度不足（HTTP 402），请检查账户余额或套餐额度。",
    )
    result = parse_document(
        pdf,
        router=router_for(
            FakeParser(name="docling", outcome=ParserError("docling 未能解析该文档。")),
            FakeParser(name="legacy", outcome=ParserError("legacy 未能解析该文档。")),
        ),
        cache=ParseCache(cache_dir),
        options=ParseOptions(force_ocr=False),
        cloud_parser=cloud,
        privacy_gate=lambda: ALLOW_CLOUD,
    )

    assert result.status == "failed"
    assert not result.is_usable
    assert result.document is None
    assert "文档解析失败" in result.error_message
    assert "docling" in result.error_message
    assert "legacy" in result.error_message
    assert "额度不足" in result.error_message
    assert result.summary_line() == result.error_message
    assert [attempt.stage for attempt in result.attempts] == [
        STAGE_PRIMARY,
        STAGE_FORCE_OCR,
        STAGE_CLOUD,
    ]


def test_a_missing_file_is_a_failure_not_an_exception(tmp_path, cache_dir):
    result = parse_document(
        str(tmp_path / "nope.pdf"),
        router=router_for(FakeParser(outcome=success(clean_document()))),
        cache=ParseCache(cache_dir),
        options=ParseOptions(force_ocr=False),
    )

    assert not result.is_usable
    assert "找不到待解析的文件" in result.error_message
    assert result.attempts == []


def test_a_cloud_parser_that_raises_becomes_a_failed_attempt(pdf, cache_dir):
    cloud = FakeCloudParser(raise_exc=RuntimeError("网络中断"))
    result = parse_document(
        pdf,
        router=router_for(FakeParser(outcome=ParserError("没有文本层。"))),
        cache=ParseCache(cache_dir),
        options=ParseOptions(force_ocr=False),
        cloud_parser=cloud,
        privacy_gate=lambda: ALLOW_CLOUD,
    )

    assert not result.is_usable
    assert "网络中断" in stage(result, STAGE_CLOUD).error_message


def test_a_cloud_parser_returning_a_bare_parse_result_is_accepted(pdf, cache_dir):
    """Defensive normalisation: a future adapter may return the gateway type."""
    document = clean_document()
    cloud = FakeCloudParser(raw_result=ParseResult.from_document(document))
    result = parse_document(
        pdf,
        router=router_for(FakeParser(outcome=ParserError("没有文本层。"))),
        cache=ParseCache(cache_dir),
        options=ParseOptions(force_ocr=False),
        cloud_parser=cloud,
        privacy_gate=lambda: ALLOW_CLOUD,
    )

    assert result.is_usable
    assert result.used_cloud is True
    assert result.parsed_by == "mineru_cloud"


def test_an_unrecognised_cloud_return_is_reported_not_swallowed(pdf, cache_dir):
    cloud = FakeCloudParser(raw_result=42)
    result = parse_document(
        pdf,
        router=router_for(FakeParser(outcome=ParserError("没有文本层。"))),
        cache=ParseCache(cache_dir),
        options=ParseOptions(force_ocr=False),
        cloud_parser=cloud,
        privacy_gate=lambda: ALLOW_CLOUD,
    )

    assert not result.is_usable
    assert "无法识别" in stage(result, STAGE_CLOUD).error_message


# --------------------------------------------------------------------------
# What the user is shown
# --------------------------------------------------------------------------


def test_the_outcome_answers_who_parsed_and_whether_it_touched_the_cloud(
    pdf, cache_dir
):
    """The P0 exit criterion, asserted on the fields that will back the UI."""
    local = parse_document(
        pdf,
        router=router_for(FakeParser(outcome=success(clean_document()))),
        cache=ParseCache(cache_dir),
        options=ParseOptions(force_ocr=False),
    )
    payload = local.to_dict()

    assert payload["parsed_by"] == "docling"
    assert payload["used_cloud"] is False
    assert payload["cloud_attempted"] is False
    assert payload["touched_network"] is False
    assert payload["cloud_refusal"] is None
    assert "未上传至任何云端服务" in local.summary_line()
    assert [stage(local, name).stage for name in (STAGE_PRIMARY, STAGE_FORCE_OCR, STAGE_CLOUD)] == [
        STAGE_PRIMARY,
        STAGE_FORCE_OCR,
        STAGE_CLOUD,
    ]


def test_the_outcome_records_a_refused_cloud_attempt_for_the_ui(pdf, cache_dir):
    result = parse_document(
        pdf,
        router=router_for(FakeParser(outcome=ParserError("没有文本层。"))),
        cache=ParseCache(cache_dir),
        options=ParseOptions(force_ocr=False),
        cloud_parser=FakeCloudParser(document=clean_document()),
    )
    payload = result.to_dict()

    assert payload["cloud_refusal"]["reason_code"] == "privacy_mode_local_only"
    assert payload["cloud_refusal"]["allowed"] is False
    cloud_attempt = next(a for a in payload["attempts"] if a["stage"] == STAGE_CLOUD)
    assert cloud_attempt["skipped_reason"] == "privacy_mode_local_only"
    assert cloud_attempt["network"] is False


def test_a_degraded_but_usable_document_travels_with_its_warnings(pdf, cache_dir):
    """Visible degradation, not silent truncation."""
    result = parse_document(
        pdf,
        router=router_for(FakeParser(outcome=success(broken_document()))),
        cache=ParseCache(cache_dir),
        options=ParseOptions(force_ocr=False),
        cloud_parser=FakeCloudParser(document=None, outcome="failed", message="云端失败。"),
        privacy_gate=lambda: ALLOW_CLOUD,
    )

    assert result.is_usable
    assert any("解析质量为 fail" in warning for warning in result.warnings)
    assert any("locator_coverage" in warning for warning in result.warnings)


# --------------------------------------------------------------------------
# Cache key shape
# --------------------------------------------------------------------------


def test_the_cache_key_covers_all_four_components():
    base = ParseCacheKey("a" * 64, "docling", "1.0", "opt1")
    assert base.filename() == ParseCacheKey("a" * 64, "docling", "1.0", "opt1").filename()
    for other in (
        ParseCacheKey("b" * 64, "docling", "1.0", "opt1"),
        ParseCacheKey("a" * 64, "legacy", "1.0", "opt1"),
        ParseCacheKey("a" * 64, "docling", "2.0", "opt1"),
        ParseCacheKey("a" * 64, "docling", "1.0", "opt2"),
    ):
        assert other.filename() != base.filename()


def test_the_cache_key_survives_a_pathological_parser_name(cache_dir):
    """A parser name is a code constant; an option hash comes from a caller.

    The property that matters is not "the string is clean" but "the resolved
    path stays inside the cache directory" — that is what a hostile name would
    be for.
    """
    cache = ParseCache(cache_dir)
    key = cache.key_for("c" * 64, "../../etc/passwd", "v/1", "opt")
    resolved = os.path.realpath(cache._path_for(key))

    assert resolved.startswith(os.path.realpath(cache_dir) + os.sep)
    assert os.sep not in os.path.basename(resolved)


def test_the_default_cache_follows_configuration(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PARSER_CACHE_ENABLED", False)
    monkeypatch.setattr(config, "PARSER_CACHE_DIR", str(tmp_path / "c"))
    disabled = orch.build_default_parse_cache()
    assert disabled.enabled is False
    assert disabled.get(ParseCacheKey("d" * 64, "docling", "1.0", "o")) is None

    monkeypatch.setattr(config, "PARSER_CACHE_ENABLED", True)
    enabled = orch.build_default_parse_cache()
    assert enabled.enabled is True
    assert enabled.directory == str(tmp_path / "c")


# --------------------------------------------------------------------------
# Locator vocabulary: the file's own units, not the parse's
# --------------------------------------------------------------------------
#
# Every test below drives the real docling normalizer over a hand-written
# ``DoclingDocument.export_to_dict()`` payload, exactly as
# ``tests/test_ingestion_docling.py`` does, and then hands the resulting
# ``ParsedDocument`` to the orchestrator. That is deliberate: the contract under
# test is the *seam* between a page-ifying backend and a layer that knows the
# file was a deck. A document built by hand from ``SourceLocator(kind="pdf_page")``
# would assert the same thing while quietly missing the case that actually
# occurs — docling's own output, with its own reading order and its own boxes.


def _docling_prov(page: int) -> list[dict[str, Any]]:
    """One docling provenance entry: a page and a PDF-space rectangle."""
    return [
        {
            "page": page,
            "bbox": {"l": 72.0, "t": 700.0, "r": 468.0, "b": 640.0, "coord_origin": "BOTTOMLEFT"},
            "span": [0, 12],
        }
    ]


def _docling_text(ref: str, text: str, page: int, label: str = "") -> dict[str, Any]:
    return {
        "self_ref": ref,
        "parent": {"$ref": "body/children/0"},
        "text": text,
        "text_label": label,
        "prov": _docling_prov(page),
    }


def _docling_table(ref: str, page: int, rows: list[list[str]]) -> dict[str, Any]:
    """A docling table with explicit inclusive cell offsets, plus a header row."""
    cells = []
    for row_index, row in enumerate(rows):
        for col_index, value in enumerate(row):
            cells.append({
                "bbox": {"l": 72.0, "t": 700.0, "r": 468.0, "b": 640.0,
                         "coord_origin": "BOTTOMLEFT"},
                "row_span": 1,
                "col_span": 1,
                "start_row_offset_idx": row_index,
                "end_row_offset_idx": row_index,
                "start_col_offset_idx": col_index,
                "end_col_offset_idx": col_index,
                "text": value,
                "column_header": row_index == 0,
                "row_header": False,
            })
    return {
        "self_ref": ref,
        "data": {
            "table_cells": cells,
            "num_rows": len(rows),
            "num_cols": len(rows[0]),
        },
        "prov": _docling_prov(page),
    }


def _docling_payload(filename: str, media_type: str, page_count: int, *items) -> dict[str, Any]:
    """A docling payload for a page-ified document with a simple body tree."""
    return {
        "schema_name": "DoclingDocument",
        "version": "1.7.0",
        "name": filename,
        "origin": {"filename": filename, "mimetype": media_type},
        "pages": {
            str(number): {"size": {"width": 612.0, "height": 792.0}, "image": {"uri": ""}}
            for number in range(1, page_count + 1)
        },
        "body": {
            "self_ref": "body",
            "name": "body",
            "children": [{"$ref": item["self_ref"]} for item in items],
        },
        "groups": [],
        "texts": [item for item in items if "text" in item],
        "tables": [item for item in items if "data" in item],
        "pictures": [],
    }


def _docling_parse(payload: dict[str, Any], media_type: str) -> ParsedDocument:
    """Run the real adapter over the payload, as a docling parse would."""
    return normalize_docling_document(
        payload,
        source_filename=payload["origin"]["filename"],
        media_type=media_type,
        parser_version="0.0-test",
    )


def _upload(tmp_path, name: str) -> str:
    """A real file on disk: the orchestrator stats the path and hashes bytes."""
    path = tmp_path / name
    path.write_bytes(b"not a real office file, only the extension matters here")
    return str(path)


def _parse_office(path: str, document: ParsedDocument, media_type: str, cache_dir: str):
    """Run the orchestrator over a file whose only parser returns ``document``."""
    return parse_document(
        path,
        router=router_for(FakeParser(name="docling", outcome=success(document),
                                     media_types=(media_type,))),
        cache=ParseCache(cache_dir),
        options=ParseOptions(force_ocr=False),
    )


def _deck_payload(slide_pages=(1, 1, 2, 3)) -> dict[str, Any]:
    """A two- and three-slide deck with a heading, body text, a caption and a table."""
    return _docling_payload(
        "deck.pptx",
        PPTX,
        3,
        _docling_text("texts/0", "研究背景" * 60, slide_pages[0], label="title"),
        _docling_text("texts/1", "我们提出了一种新的检索方法。" * 30, slide_pages[1]),
        _docling_text("texts/2", "图 1：整体架构。" * 20, slide_pages[2], label="caption"),
        _docling_table(
            "tables/0", slide_pages[3], [["模型名称", "F1 分数"], ["本文方法", "91.2"]]
        ),
    )


def test_a_docling_parsed_deck_is_retargeted_to_slides(tmp_path, cache_dir):
    """The P0 promise, for a deck: every key element lands on a slide.

    Docling renders a slide onto a page, so page 3 of a converted deck *is*
    slide 3. The orchestrator is the only layer holding both the original file's
    type and the parse, which is why the translation lives here.
    """
    document = _docling_parse(_deck_payload(), PPTX)
    # The precondition this whole section exists for: docling page-ifies a deck.
    assert {element.locator.kind for element in document.elements} == {"pdf_page"}
    assert all(element.locator.bbox is not None for element in document.elements)

    result = _parse_office(_upload(tmp_path, "deck.pptx"), document, PPTX, cache_dir)

    assert result.is_usable
    assert result.parsed_by == "docling"
    for element in result.document.elements:
        locator = element.locator
        assert locator.kind == "slide"
        assert locator.slide is not None
        assert locator.kind in LOCATOR_KINDS
        # The rendered-page rectangle is dropped rather than relabelled: it is
        # geometry in a coordinate space the original .pptx does not have.
        assert locator.bbox is None
        assert locator.page is None
    assert [element.locator.slide for element in result.document.elements] == [1, 1, 2, 3]
    assert result.document.locator_coverage == 1.0


def test_a_retargeted_deck_keeps_units_and_labels_in_the_slide_vocabulary(
    tmp_path, cache_dir
):
    """A deck must not speak two vocabularies: units, labels and elements agree.

    Units are what an evidence request actually resolves to, and they were
    derived from the page locators before the retarget. Rebuilding them is what
    stops a citation from arriving as "p. 2" for a document whose elements say
    "slide 2".
    """
    document = _docling_parse(_deck_payload(), PPTX)
    result = _parse_office(_upload(tmp_path, "deck.pptx"), document, PPTX, cache_dir)

    units = result.document.units
    assert [unit.locator.kind for unit in units] == ["slide", "slide", "slide"]
    assert [unit.label for unit in units] == ["slide 1", "slide 2", "slide 3"]
    for element in result.document.elements:
        assert element.unit_id, "每个元素都必须指回它所属的幻灯片单元"
        assert element.page_number_label.startswith("slide ")


def test_heading_caption_and_table_on_a_deck_all_round_trip_to_the_slide_kind(
    tmp_path, cache_dir
):
    """The three element roles a reader is most likely to cite.

    Roles are what chunking and rendering branch on, so a locator rewrite that
    only worked for prose would still leave figures and tables unresolvable.
    """
    document = _docling_parse(_deck_payload(), PPTX)
    result = _parse_office(_upload(tmp_path, "deck.pptx"), document, PPTX, cache_dir)

    by_role = {element.role: element for element in result.document.elements}
    assert set(by_role) >= {"title", "caption", "table"}
    for role, expected_slide in (("title", 1), ("caption", 2), ("table", 3)):
        locator = by_role[role].locator
        assert locator.kind == "slide", f"{role} 元素应当定位到幻灯片"
        assert locator.slide == expected_slide
    # The grid survives the rewrite; only the locator changed.
    assert by_role["table"].table is not None
    assert by_role["table"].table.rows == [["模型名称", "F1 分数"], ["本文方法", "91.2"]]


def test_a_retargeted_deck_is_not_judged_as_though_it_had_no_pages(tmp_path, cache_dir):
    """The ordering constraint, asserted through its consequence.

    ``quality._page_of`` reads a page only off a ``pdf_page`` locator, so a deck
    retargeted *before* the gate would score zero page coverage against three
    pages and FAIL — turning a perfectly parsed presentation into a paid cloud
    escalation. The gate must see the parse as the backend reported it.
    """
    document = _docling_parse(_deck_payload(), PPTX)
    result = _parse_office(_upload(tmp_path, "deck.pptx"), document, PPTX, cache_dir)

    assert result.quality is not None
    assert result.quality.status is not QualityStatus.FAIL
    assert [flag.rule_id for flag in result.quality.flags if flag.rule_id == "page_coverage"] == []


def test_a_docling_parsed_word_file_keeps_its_page_locator_and_says_why(tmp_path, cache_dir):
    """The negative case for DOCX, and the one that must *not* be papered over.

    A ``docx_element`` locator is a paragraph index into the original
    ``word/document.xml``. A page-ified parse enumerates what survived
    rendering, so its k-th element is not the file's k-th paragraph. Inventing
    the index would produce a citation that looks exact and lands a section
    away, so the honest page stays and the limitation is reported.
    """
    payload = _docling_payload(
        "report.docx",
        DOCX,
        2,
        _docling_text("texts/0", "引言", 1, label="section_header"),
        _docling_text("texts/1", "本文提出一种面向证据的引用方案。" * 30, 1),
        _docling_text("texts/2", "方法", 2, label="section_header"),
        _docling_text("texts/3", "我们使用两阶段解析与质量门控。" * 30, 2),
    )
    document = _docling_parse(payload, DOCX)
    result = _parse_office(_upload(tmp_path, "report.docx"), document, DOCX, cache_dir)

    assert result.is_usable
    for element in result.document.elements:
        assert element.locator.kind == "pdf_page", "不得凭空生成 docx_element 定位"
        assert element.locator.paragraph is None
        assert element.locator.page in {1, 2}
    assert result.document.locator_coverage == 1.0, "诚实降级不得让引用失去落点"
    assert any("段落" in warning for warning in result.warnings)
    assert any("T3.3" in warning for warning in result.warnings)
    # The heading path *was* reported by the parse, so it is kept: it is
    # faithful, and it is what the citation label falls back to.
    assert result.document.elements[-1].heading_path == ("引言", "方法")


def test_a_docling_parsed_workbook_keeps_its_page_locator_and_says_why(tmp_path, cache_dir):
    """The negative case for XLSX, which is a step further from recoverable.

    A ``sheet_range`` locator is a sheet *name* — a string that exists in the
    workbook and in no rendered page. There is no arithmetic from a page number
    to a sheet name, so this is not an index that needs aligning; the value is
    simply absent from the parse.
    """
    payload = _docling_payload(
        "results.xlsx",
        XLSX,
        2,
        _docling_table("tables/0", 1, [["模型", "F1", "EM"], ["本文", "91.2", "88.0"]]),
        _docling_table("tables/1", 2, [["模型", "Recall"], ["基线", "0.71"]]),
    )
    document = _docling_parse(payload, XLSX)
    result = _parse_office(_upload(tmp_path, "results.xlsx"), document, XLSX, cache_dir)

    assert result.is_usable
    for element in result.document.elements:
        assert element.locator.kind == "pdf_page"
        assert element.locator.sheet is None
        assert element.locator.cell_range is None
    assert any("工作表" in warning for warning in result.warnings)


def test_a_deck_with_a_broken_page_sequence_keeps_its_pages_and_warns(tmp_path, cache_dir):
    """The safety net inside the one translation this layer does perform.

    Slides 1 and 3 came back but slide 2 did not. Page 3 is slide 4 in the
    original file, so renumbering it as "slide 3" would be off by one from that
    point on — a locator that looks perfect and resolves to the wrong slide,
    which is the one thing worse than an honest page. The pages are kept and the
    gap is named.
    """
    payload = _deck_payload(slide_pages=(1, 1, 3, 3))
    document = _docling_parse(payload, PPTX)
    result = _parse_office(_upload(tmp_path, "deck.pptx"), document, PPTX, cache_dir)

    assert result.is_usable
    for element in result.document.elements:
        assert element.locator.kind == "pdf_page", "页码不连续时不得改写为幻灯片编号"
    assert any("不连续" in warning for warning in result.warnings)
    assert result.document.locator_coverage == 1.0


def test_a_pdf_is_left_exactly_as_the_parser_reported_it(tmp_path, cache_dir):
    """No media type outside the office table is touched, PDF least of all.

    A PDF's page and box are the file's own coordinates, not a rendering's, so
    the one thing that must never happen is a "correction" to a locator that was
    already right.
    """
    payload = _docling_payload(
        "paper.pdf",
        PDF,
        2,
        _docling_text("texts/0", "摘要" * 40, 1, label="section_header"),
        _docling_text("texts/1", "本文研究文档解析中的引用定位问题。" * 20, 1),
        _docling_text("texts/2", "实验表明该方法在三个数据集上有效。" * 20, 2),
    )
    document = _docling_parse(payload, PDF)
    result = _parse_office(_upload(tmp_path, "paper.pdf"), document, PDF, cache_dir)

    assert result.is_usable
    for element in result.document.elements:
        locator = element.locator
        assert locator.kind == "pdf_page"
        assert locator.slide is None and locator.paragraph is None
        assert locator.bbox is not None, "PDF 的 bbox 是原文件坐标，必须保留"
    assert result.document.warnings == document.warnings


def test_the_retarget_never_mutates_the_cached_document(tmp_path, cache_dir):
    """A cache entry is read again on every later call; rewriting it in place
    would make the second call see an already-converted document while the
    on-disk copy still held page locators — the same bytes, two answers.
    """
    document = _docling_parse(_deck_payload(), PPTX)
    path = _upload(tmp_path, "deck.pptx")
    cache = ParseCache(cache_dir)
    options = ParseOptions(force_ocr=False)

    first = parse_document(
        path,
        router=router_for(FakeParser(outcome=success(document), media_types=(PPTX,))),
        cache=cache,
        options=options,
    )
    second = parse_document(
        path,
        router=router_for(FakeParser(outcome=success(document), media_types=(PPTX,))),
        cache=ParseCache(cache_dir),
        options=options,
    )

    assert first.cache_hit is False and second.cache_hit is True
    assert all(element.locator.kind == "pdf_page" for element in document.elements)
    assert [element.locator.kind for element in second.document.elements] == ["slide"] * 4


def test_a_degradation_warning_is_recorded_once_per_outcome_not_once_per_call(
    tmp_path, cache_dir
):
    """Warnings accumulate on documents; an outcome must not multiply one.

    The retarget copies the document rather than appending to it, which is what
    keeps a cached entry from gaining a fresh copy of the same warning on every
    hit.
    """
    payload = _docling_payload(
        "report.docx", DOCX, 1, _docling_text("texts/0", "正文内容。" * 60, 1)
    )
    document = _docling_parse(payload, DOCX)
    path = _upload(tmp_path, "report.docx")
    options = ParseOptions(force_ocr=False)

    def run() -> ParseOutcome:
        return parse_document(
            path,
            router=router_for(FakeParser(outcome=success(document), media_types=(DOCX,))),
            cache=ParseCache(cache_dir),
            options=options,
        )

    first, second = run(), run()
    matched = [w for w in first.warnings if "段落" in w]
    assert len(matched) == 1
    assert len([w for w in second.warnings if "段落" in w]) == 1


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def cache_entry_count(cache_dir: str) -> int:
    return sum(
        1
        for _, _dirs, files in os.walk(cache_dir)
        for name in files
        if name.endswith(".json")
    )


def corrupt_every_entry(cache_dir: str) -> None:
    """Replace every stored entry with something unparseable."""
    for root, _dirs, files in os.walk(cache_dir):
        for name in files:
            if name.endswith(".json"):
                with open(os.path.join(root, name), "w", encoding="utf-8") as handle:
                    handle.write('{"schema_version": "parse-cache/1", "document": ')


def _lenient_thresholds():
    """Thresholds no ordinary document can fail, to force a quality tie."""
    from paperwhisperer.ingestion.quality import QualityThresholds

    return QualityThresholds(
        min_page_coverage_ratio=0.0,
        max_blank_page_ratio=1.0,
        max_garbled_char_ratio=1.0,
        min_chars_per_page=0,
        min_document_chars=0,
        min_resolvable_locator_ratio=0.0,
    )
