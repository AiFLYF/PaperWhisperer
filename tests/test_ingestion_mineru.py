"""MinerU 云端适配器：全链路 mock、隐私门控与重试降级。

这个模块是全项目唯一主动出网到第三方服务的地方，所以它的测试和其他适配器
反过来：这里真正要证明的不是"能解析出 PDF"，而是**不该出网的时候一个字节都
不出去**。因此测试分成两组语气：

* :class:`TestLocalOnlyEgress` 是 P0 release gate「Local-only cloud egress: 0 次」
  的机器可验证形式。它把传输层换成一个**只要被调用就直接失败**的桩——不是
  "记录下来再断言调用次数为 0"那种可以自己骗自己的写法，而是让出网这个动作
  本身不存在，因此无法被绕过或误报。
* 其余部分用完整的 mock 链路（签名 → PUT → 轮询 → 下载 → normalize）证明功能，
  全部走内存里的假响应，一次真实网络请求都不会发生。

传输层有**两条腿**：签名/轮询/下载走 ``urllib.request.urlopen``，上传走
:func:`_no_redirect_opener` 返回的 ``opener.open``（预签名 URL 经不起一次 30x，
所以上传这一腿显式拒绝跟随重定向）。因此所有测试都通过
:func:`_patch_transport` 一次性把两条腿都换掉——只换 ``urlopen`` 会让上传那
条腿真的打到网络上，而 ``[Errno -2] Name or service not known`` 正是这么来的：
假域名解析不了，于是测试安静地变慢并失败，而不是在桩上失败。

真实 token 的 smoke test 故意不存在：仓库里没有 ``MINERU_API_TOKEN``，而一个
会因为环境变量是否存在而随机跳过或随机失败的测试比没有测试更糟。计划里把它
列为手动触发项，这里保持一致。
"""

from __future__ import annotations

import io
import json
import socket
import urllib.error
import urllib.request
import zipfile
from email.message import Message
from pathlib import Path

import pytest

from paperwhisperer.core import config
from paperwhisperer.ingestion.adapters import mineru_cloud as adapter
from paperwhisperer.ingestion.adapters.mineru_cloud import (
    MAX_ZIP_ENTRIES,
    MAX_ZIP_ENTRY_UNCOMPRESSED_BYTES,
    MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES,
    MineruAuditRecord,
    MineruCloudParser,
    MineruConsentError,
    MineruError,
    _is_safe_member_name,
    _Settings,
    append_audit_record,
    evaluate_privacy_gate,
    extract_result_payload,
    normalize_mineru_payload,
    parse_document,
    read_audit_records,
    resolve_audit_log_path,
    validate_zip_bomb,
)
from paperwhisperer.ingestion.schema import ParsedDocument

from .helpers import FakeResponse

#: 假 MinerU 服务地址。**不是**真实域名：这些测试一旦真的发出请求就会立刻
#: 失败并暴露问题，而不是安静地把文档发给一个真的第三方。
FAKE_BASE_URL = "https://mineru.example.test"

#: 签名上传地址与结果下载地址。两者都要过 SSRF 校验，测试里用 monkeypatch
#: 放行（见 :func:`_allow_all_urls`），不去做真实 DNS 解析。
UPLOAD_URL = "https://upload.mineru.example.test/put/abc123"
RESULT_ZIP_URL = "https://cdn.mineru.example.test/result/batch-1.zip"

SIGNED_BATCH = {"code": 0, "data": {"batch_id": "batch-1", "file_urls": [UPLOAD_URL]}}


class _ExplodingUrlopen:
    """任何一次出网都让测试立刻失败。

    这是本文件的核心安全装置。它不是"记录调用然后断言 0 次"——那种写法在
    有人给桩加了一层缓存或重试时依然会通过，而缓存/重试恰恰是把文档发出去的
    两条最常见的路。让桩直接抛异常，出网这个动作在测试里根本不存在。
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, request, timeout=None, **_kwargs):
        url = getattr(request, "full_url", str(request))
        self.calls.append(url)
        raise AssertionError(f"local_only 模式下不得发起任何网络请求，但收到了 {url}")


def _allow_all_urls(monkeypatch) -> None:
    """放行 SSRF 校验，让测试用假域名而不去解析真实 DNS。"""
    monkeypatch.setattr(adapter, "is_public_http_url", lambda _url: True)


class _FakeOpener:
    """上传那一腿的传输替身：只暴露 ``OpenerDirector.open``。

    只需要一个 ``open(request, timeout=...)``：``_open_with_retry`` 在拿到
    ``opener=`` 时就切到这条路径上（自定义 opener 自带 TLS context，而
    ``OpenerDirector.open`` 不接受 ``context`` 参数），其余 urllib 内部机制在
    被测逻辑之外，因此不复制。
    """

    def __init__(self, transport) -> None:
        self._transport = transport

    def open(self, request, timeout=None):
        return self._transport(request, timeout=timeout)


def _patch_transport(monkeypatch, transport) -> None:
    """把整条链路的**两条腿**都换成同一个假传输。

    只换 ``urllib.request.urlopen`` 是不够的：上传那一步传的是
    ``opener=_no_redirect_opener(...)``，不经过 ``urlopen``。曾经这个文件用
    只换 ``urlopen`` 的写法跑全链路，结果上传那一步真的去连了
    ``upload.mineru.example.test``——15 个测试以
    ``[Errno -2] Name or service not known`` 失败。那种失败模式很坏：它看起来
    像"网络不稳"，不像"测试没接住"。

    拦截点选在 ``_no_redirect_opener`` 而不是
    ``urllib.request.build_opener``：前者是本模块为了"上传不跟随重定向"而
    专门调用的那个函数，是生产代码里可替换的接缝；后者是标准库全局函数，
    换掉它会顺带影响本进程里任何其他用到 urllib 的代码。
    """
    monkeypatch.setattr(urllib.request, "urlopen", transport)
    monkeypatch.setattr(
        adapter, "_no_redirect_opener", lambda *_args, **_kwargs: _FakeOpener(transport)
    )


def _wire_headers(request) -> dict[str, str]:
    """urllib 真正会写到 socket 上的那份头（含 ``do_request_`` 的补充）。

    ``AbstractHTTPHandler.do_request_`` 是进 socket 前的最后一步，也正是
    ``Content-Type`` 被静默补上 ``application/x-www-form-urlencoded`` 的地方。
    只读 ``Request.headers`` 读到的是补充**之前**的状态，用它断言 Content-Type
    会得到一个必然为真的空断言。这个函数跑的就是那个补充步骤——它不开连接，
    只改 ``Request``。
    """
    opener = urllib.request.OpenerDirector()
    handler = urllib.request.AbstractHTTPHandler()
    # do_request_ 会读 ``self.parent.addheaders``，而它只有在挂在某个 opener
    # 上时才有 parent；ProxyHandler 空字典表示"不走代理"，免得它去读环境变量。
    handler.parent = opener
    opener.add_handler(urllib.request.ProxyHandler({}))
    prepared = handler.do_request_(request)
    return {**prepared.headers, **prepared.unredirected_hdrs}


class _CannedResponse:
    """够 urllib 错误链跑完的最小 ``HTTPResponse``。

    只在 :class:`TestPresignedUploadLeg` 里用：那里要让一个**真的**
    ``OpenerDirector`` 处理一次 30x，就需要一个能被 ``HTTPErrorProcessor`` 读出
    ``code``/``msg``/``info()``、并能当 ``HTTPError`` 的 ``fp`` 的东西。开一个
    真 socket 去拿一个 302 是不可接受的——所以只实现被读到的这几个属性。
    """

    def __init__(self, code: int, headers: dict, url: str) -> None:
        self.code = code
        self.status = code
        self.msg = self.reason = "Found"
        self.headers = Message()
        for name, value in headers.items():
            self.headers[name] = value
        self.url = url
        self.host = url

    def info(self) -> Message:
        return self.headers

    def read(self, *_args) -> bytes:
        return b""

    def close(self) -> None:
        return None


class _StubbedTransport(urllib.request.BaseHandler):
    """顶掉 ``http_open``/``https_open``，让一个真实 opener 完全跑在内存里。

    拦截的是**开 socket 的那一个 handler**，而不是 ``OpenerDirector.open``：
    这样 opener 仍然由 ``build_opener`` 真实构造、重定向/错误链也仍然由 urllib
    真实执行，测试要断言的"30x 有没有被重放"才是 urllib 自己的行为，而不是
    测试替身写死的返回值。``handler_order`` 取 400（默认 HTTP/HTTPS handler 是
    500）保证它排在真正会联网的那个前面。
    """

    handler_order = 400

    def __init__(self, response) -> None:
        self.response = response
        #: 被请求过的 URL，按顺序——"有没有第二次"就是重定向有没有被跟随的证据。
        self.seen: list[str] = []
        self.methods: list[str] = []

    def _serve(self, request):
        self.seen.append(request.full_url)
        self.methods.append(request.get_method())
        return self.response

    http_open = _serve
    https_open = _serve


@pytest.fixture(autouse=True)
def no_real_sockets(monkeypatch):
    """把本文件的出网能力整体掐断，而不只是"少patch 一个地方"。

    上面那些传输替身依赖"生产代码仍然经由那两个入口出网"这个前提；一旦有人
    改用 ``http.client``、``requests`` 或某个 SDK，这个前提就破了，替身会被
    绕过，而症状是慢吞吞地等 DNS 超时。这里直接让 socket 层不可用：任何真
    正的网络访问——不管它从哪个模块发起的——都会立刻失败在最近的一层，而不是
    变成一个看起来像 flaky 的超时。

    ``getaddrinfo`` 一起挡掉：正是一次真实 DNS 解析（而不是一次连接）造成了
    ``[Errno -2] Name or service not known``，只挡 connect 的话这个最典型的
    漏网方式仍然会先卡满一次解析超时。
    """

    def _refuse(*args, **_kwargs):
        raise AssertionError("本测试文件不允许发起任何真实网络访问")

    monkeypatch.setattr(socket.socket, "connect", _refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", _refuse)
    monkeypatch.setattr(socket, "create_connection", _refuse)
    monkeypatch.setattr(socket, "getaddrinfo", _refuse)


def _json_response(payload: dict) -> FakeResponse:
    return FakeResponse(
        json.dumps(payload).encode("utf-8"),
        content_type="application/json",
        url="https://mineru.example.test/api",
    )


def _pending() -> dict:
    return {"code": 0, "data": {"extract_result": [{"state": "running"}]}}


def _done() -> dict:
    return {
        "code": 0,
        "data": {"extract_result": [{"state": "done", "full_zip_url": RESULT_ZIP_URL}]},
    }


def _result_zip(markdown: str = "# 标题\n\n正文内容。") -> bytes:
    """构造一个结构与 MinerU 真实结果一致的 zip。"""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "batch-1/content_list.json",
            json.dumps(
                [
                    {"type": "title", "text": "论文标题", "page_idx": 0, "bbox": [10, 10, 300, 40]},
                    {"type": "text", "text": "第一段正文。", "page_idx": 0},
                    {"type": "text", "text": "第二段正文。", "page_idx": 1},
                ],
                ensure_ascii=False,
            ),
        )
        archive.writestr("batch-1/full.md", markdown)
    return buffer.getvalue()


class _FakeClock:
    """假时钟：``sleep`` 只推进时间而不真睡。

    没有它，轮询超时的测试要么真的等满 ``poll_timeout_seconds``，要么只能
    靠``monkeypatch`` 换掉整个 ``poll_batch``——后者等于把要测的逻辑一起
    换掉了。
    """

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


class _MineruStub:
    """按 URL 路由的假 MinerU 服务，记录每一步并可编排失败。

    端点而不是请求序号，因为真实链路的重试发生在**单步内部**：一次 429 之后
    重放的是同一个请求，序号路由会把重试错认成"进入了下一步"。
    """

    def __init__(self, *, poll_states=None, upload_bodies=None) -> None:
        #: 轮询依次返回的状态；用尽后重复最后一个。
        self.poll_states = list(poll_states or ["done"])
        #: (method, url) 摘要，按顺序。链路顺序与重试次数从这里读。
        self.requests: list[tuple[str, str]] = []
        #: 完整的 Request 对象，需要检查 body/headers 时用。
        self.captured: list = []
        self._poll_index = 0
        self.zip_bytes = _result_zip()

    @property
    def steps(self) -> list[str]:
        """按顺序记录链路走了哪几步（重试会重复出现）。"""
        return [method for method, _url in self.requests]

    def find(self, method: str, *, url_contains: str = "") -> list:
        """按方法（可选再按 URL 片段）筛出捕获到的请求。"""
        return [
            request
            for request in self.captured
            if request.get_method() == method
            and url_contains in request.full_url
        ]

    @staticmethod
    def _state_payload(state: str) -> dict:
        """按状态名构造进度响应。

        终态必须**原样**回给适配器而不是统统当"进行中"——否则一个 ``failed``
        会让 ``poll_batch`` 一直轮询到真实的 900 秒截止时间，把整个套件挂住。
        这不是假设出来的麻烦：``TERMINAL_FAILURE_STATES`` 就是靠这些字面量
        判定的。
        """
        if state == "done":
            return _done()
        if state in adapter.TERMINAL_FAILURE_STATES:
            return {"code": 0, "data": {"extract_result": [{"state": state}]}}
        return _pending()

    def __call__(self, request, timeout=None, **_kwargs):
        url = getattr(request, "full_url", str(request))
        method = request.get_method()
        self.requests.append((method, url))
        self.captured.append(request)
        path = url.split("example.test", 1)[-1]
        if path.startswith("/api/v4/file-urls/batch"):
            return _json_response(SIGNED_BATCH)
        if method == "PUT":
            return FakeResponse(b"", content_type=None, url=UPLOAD_URL)
        if path.startswith("/api/v4/extract-results/batch"):
            state = self.poll_states[min(self._poll_index, len(self.poll_states) - 1)]
            self._poll_index += 1
            return _json_response(self._state_payload(state))
        if path.endswith(".zip"):
            return FakeResponse(self.zip_bytes, content_type="application/zip", url=RESULT_ZIP_URL)
        raise AssertionError(f"测试桩收到了意料之外的请求：{method} {url}")


@pytest.fixture
def allow_cloud(monkeypatch):
    """隐私策略与用户同意都放行——只跳过门控，不跳过其他任何检查。

    轮询上限被压到 1 秒是一道**防挂死**的保险：``poll_batch`` 用的是真实的
    ``time.monotonic``，任何一个因为桩写错而永远返回"进行中"的测试都会
    在这里烧满 900 秒的默认上限，把整轮 CI 拖死。压到 1 秒之后，同样的错误
    只是这个测试失败，而不是整个套件卡住。需要精确控制截止时间的测试自己
    构造 ``_Settings`` 并注入假时钟。
    """
    monkeypatch.setattr(config, "DOCUMENT_PRIVACY_MODE", "allow_cloud")
    monkeypatch.setattr(config, "DOCUMENT_CLOUD_CONSENT_GRANTED", True)
    monkeypatch.setattr(config, "MINERU_API_TOKEN", "test-token")
    monkeypatch.setattr(config, "MINERU_API_BASE_URL", FAKE_BASE_URL)
    monkeypatch.setattr(config, "MINERU_POLL_TIMEOUT_SECONDS", 1)
    monkeypatch.setattr(config, "MINERU_POLL_INTERVAL_SECONDS", 0)
    _allow_all_urls(monkeypatch)
    return monkeypatch


@pytest.fixture
def cloud_ready(allow_cloud, tmp_path, monkeypatch):
    """一个可上传的假 PDF + 审计日志路径。审计始终落在 tmp_path，绝不写进仓库。"""
    target = tmp_path / "paper.pdf"
    target.write_bytes(b"%PDF-1.7\nbody")
    return target, str(tmp_path / "mineru_audit.jsonl")


def _parse(target, audit_path, **kwargs):
    kwargs.setdefault("sleeper", lambda _seconds: None)
    return parse_document(str(target), audit_log_path=audit_path, **kwargs)


class TestPrivacyGate:
    """门控是两道的，且两道都不可绕过——只满足一道必须是拒绝。"""

    def test_local_only_blocks_even_with_consent_and_a_token(self, monkeypatch):
        monkeypatch.setattr(config, "DOCUMENT_PRIVACY_MODE", "local_only")
        monkeypatch.setattr(config, "DOCUMENT_CLOUD_CONSENT_GRANTED", True)
        monkeypatch.setattr(config, "MINERU_API_TOKEN", "test-token")

        gate = evaluate_privacy_gate()

        # 策略层先于用户层：系统没开这个口子，用户同意与否都不改变结论。
        assert gate.allowed is False
        assert gate.reason_code == "privacy_mode_local_only"
        assert "local_only" in gate.message

    def test_consent_is_required_even_when_policy_allows(self, monkeypatch):
        monkeypatch.setattr(config, "DOCUMENT_PRIVACY_MODE", "allow_cloud")
        monkeypatch.setattr(config, "DOCUMENT_CLOUD_CONSENT_GRANTED", False)
        monkeypatch.setattr(config, "MINERU_API_TOKEN", "test-token")

        gate = evaluate_privacy_gate()

        # 两道检查分开返回，用户才知道该去改哪一处，而不是两边都改一遍。
        assert gate.allowed is False
        assert gate.reason_code == "consent_not_granted"
        assert "同意" in gate.message

    @pytest.mark.parametrize("mode", ["", "ALLOW_CLOUD", "yes", "allow-cloud", "true", None])
    def test_an_unrecognised_privacy_mode_falls_back_to_local_only(self, monkeypatch, mode):
        """不确定即禁止出网。把一个手滑的配置值宽松解析成 allow_cloud，
        等于默认把隐私门打开了。"""
        monkeypatch.setattr(config, "DOCUMENT_PRIVACY_MODE", mode)
        monkeypatch.setattr(config, "DOCUMENT_CLOUD_CONSENT_GRANTED", True)
        monkeypatch.setattr(config, "MINERU_API_TOKEN", "t")

        settings = adapter._load_settings()
        gate = evaluate_privacy_gate(settings)

        assert settings.privacy_mode == "local_only"
        assert gate.allowed is False
        assert gate.reason_code == "privacy_mode_local_only"

    def test_the_normalisation_happens_inside_the_settings_loader(self, monkeypatch):
        """归一化是本模块的责任，不依赖 config 那边是否也做了同样的事——
        适配器不能因为上游改了实现就悄悄放松门控。"""
        monkeypatch.setattr(config, "DOCUMENT_PRIVACY_MODE", "allow_cloud")
        monkeypatch.setattr(config, "DOCUMENT_CLOUD_CONSENT_GRANTED", True)

        assert adapter._load_settings().privacy_mode == "allow_cloud"

    def test_both_gates_open_allows_the_call(self, allow_cloud):
        gate = evaluate_privacy_gate()

        assert gate.allowed is True
        assert gate.reason_code == "allowed"

    def test_a_whitespace_only_token_counts_as_missing(self, monkeypatch):
        monkeypatch.setattr(config, "DOCUMENT_PRIVACY_MODE", "allow_cloud")
        monkeypatch.setattr(config, "DOCUMENT_CLOUD_CONSENT_GRANTED", True)
        monkeypatch.setattr(config, "MINERU_API_TOKEN", "   ")

        assert adapter._load_settings().token == ""


class TestLocalOnlyEgress:
    """P0 release gate：Local-only cloud egress = 0 次。

    证明方式是把 ``urllib.request.urlopen`` 换成 :class:`_ExplodingUrlopen`——
    被调用即失败——并断言解析仍然返回了一个**结构化结果**。两件事必须同时成立：

    1. 出网点一次都没被碰到（否则测试直接失败，不需要看断言）；
    2. 调用方拿到的是"被拦截"而不是异常或崩溃，因为编排层要靠它静默降级。
    """

    def test_a_local_only_parse_makes_zero_network_calls(self, tmp_path, monkeypatch):
        """两道门里**只**留策略这一道挡路。

        同意必须置为 True：如果同时把两道都关掉，这个测试会因为"用户没同意"
        而被拦下，于是它证明的是第二道门而不是策略门——把第一道门整个删掉它
        照样会绿。这里让策略成为唯一可能拦住这次调用的一道。
        """
        monkeypatch.setattr(config, "DOCUMENT_PRIVACY_MODE", "local_only")
        monkeypatch.setattr(config, "DOCUMENT_CLOUD_CONSENT_GRANTED", True)
        monkeypatch.setattr(config, "MINERU_API_TOKEN", "test-token")
        guard = _ExplodingUrlopen()
        _patch_transport(monkeypatch, guard)
        target = tmp_path / "paper.pdf"
        target.write_bytes(b"%PDF-1.7\n")
        audit = str(tmp_path / "audit.jsonl")

        document, record = _parse(target, audit, trigger_reason="fallback")

        # 关键断言：出网函数一次都没被调用。
        assert guard.calls == []
        # 且链路是干净降级，不是异常。
        assert document is None
        assert record.outcome == "blocked"
        # 确认拦住它的是策略而不是同意——否则这个测试名不副实。
        assert "local_only" in record.message

    def test_a_blank_or_malformed_privacy_mode_also_keeps_egress_at_zero(
        self, tmp_path, monkeypatch
    ):
        """一个手滑的配置值不得被宽松解析成"允许出网"。

        同样把同意设为 True，让策略成为唯一可能拦住调用的一道。
        """
        for bad_mode in ("ALLOW_CLOUD", "allow-cloud", "1", "", "  "):
            monkeypatch.setattr(config, "DOCUMENT_PRIVACY_MODE", bad_mode)
            monkeypatch.setattr(config, "DOCUMENT_CLOUD_CONSENT_GRANTED", True)
            monkeypatch.setattr(config, "MINERU_API_TOKEN", "test-token")
            guard = _ExplodingUrlopen()
            _patch_transport(monkeypatch, guard)
            target = tmp_path / "paper.pdf"
            target.write_bytes(b"%PDF-1.7\n")

            document, record = _parse(target, str(tmp_path / "audit.jsonl"))

            assert guard.calls == [], f"privacy_mode={bad_mode!r} 泄漏了一次出网"
            assert document is None
            assert record.outcome == "blocked"

    def test_local_only_holds_even_when_policy_allows_but_consent_is_missing(
        self, tmp_path, monkeypatch
    ):
        """反向用例：只开策略不开同意，一样零出网。"""
        monkeypatch.setattr(config, "DOCUMENT_PRIVACY_MODE", "allow_cloud")
        monkeypatch.setattr(config, "DOCUMENT_CLOUD_CONSENT_GRANTED", False)
        monkeypatch.setattr(config, "MINERU_API_TOKEN", "test-token")
        guard = _ExplodingUrlopen()
        _patch_transport(monkeypatch, guard)
        target = tmp_path / "paper.pdf"
        target.write_bytes(b"%PDF-1.7\n")

        document, record = _parse(target, str(tmp_path / "audit.jsonl"))

        assert guard.calls == []
        assert record.outcome == "blocked"
        assert record.trigger_reason == "manual"

    def test_the_router_facing_parser_is_also_egress_free_under_local_only(
        self, tmp_path, monkeypatch
    ):
        """走 Parser 的 ``parse()`` 入口也一样——门控在链路最前面，不在某个
        具体的调用点上，所以绕过 :func:`parse_document` 直接用类不会漏出去。"""
        monkeypatch.setattr(config, "DOCUMENT_PRIVACY_MODE", "local_only")
        monkeypatch.setattr(config, "DOCUMENT_CLOUD_CONSENT_GRANTED", True)
        monkeypatch.setattr(config, "MINERU_API_TOKEN", "test-token")
        guard = _ExplodingUrlopen()
        _patch_transport(monkeypatch, guard)
        target = tmp_path / "paper.pdf"
        target.write_bytes(b"%PDF-1.7\n")
        parser = MineruCloudParser(audit_log_path=str(tmp_path / "audit.jsonl"))

        document, record = parser.parse(str(target))

        assert guard.calls == []
        assert document is None
        assert record.outcome == "blocked"
        # 拦住它的是策略门：同意已经给了，所以只可能是 privacy_mode。
        assert "local_only" in record.message

    def test_a_missing_token_fails_before_any_request(self, allow_cloud, tmp_path):
        """没有 token 时连一个注定 401 的请求都不该发——那会白白消耗一次
        额度检查并把一个配置问题伪装成网络问题。"""
        allow_cloud.setattr(config, "MINERU_API_TOKEN", "")
        guard = _ExplodingUrlopen()
        _patch_transport(allow_cloud, guard)
        target = tmp_path / "paper.pdf"
        target.write_bytes(b"%PDF-1.7\n")

        document, record = _parse(target, str(tmp_path / "audit.jsonl"))

        assert guard.calls == []
        assert document is None
        assert record.outcome == "failed"
        assert "MINERU_API_TOKEN" in record.message


class TestFullChain:
    """mock 全链路：签名 → PUT → 轮询 → 下载 → normalize。"""

    def test_the_five_steps_run_in_order_and_produce_a_document(
        self, cloud_ready, monkeypatch
    ):
        target, audit = cloud_ready
        stub = _MineruStub()
        _patch_transport(monkeypatch, stub)

        document, record = _parse(target, audit)

        assert isinstance(document, ParsedDocument)
        assert record.outcome == "success"
        assert document.parser_name == "mineru_cloud"
        assert "论文标题" in document.to_full_text()
        assert "第二段正文。" in document.to_full_text()

    def test_the_signed_url_is_requested_before_the_file_is_uploaded(
        self, cloud_ready, monkeypatch
    ):
        """签名与上传的顺序不能反：先 PUT 就无从得知该 PUT 到哪。"""
        target, audit = cloud_ready
        stub = _MineruStub()
        _patch_transport(monkeypatch, stub)

        _parse(target, audit)

        methods = [method for method, _url in stub.requests]
        assert methods[0] == "POST"
        assert "PUT" in methods
        assert methods.index("POST") < methods.index("PUT")

    def test_the_upload_puts_the_real_file_bytes_to_the_signed_url(
        self, cloud_ready, monkeypatch
    ):
        target, audit = cloud_ready
        stub = _MineruStub()
        _patch_transport(monkeypatch, stub)

        _parse(target, audit)

        put = stub.find("PUT")[0]
        assert put.full_url == UPLOAD_URL
        assert put.data == b"%PDF-1.7\nbody"

    def test_the_upload_url_is_carried_by_an_authorization_header_not_the_body(
        self, cloud_ready, monkeypatch
    ):
        """签名步骤只换 URL 不传内容，所以即使这一步被日志记录也不会泄露正文。"""
        target, audit = cloud_ready
        stub = _MineruStub()
        _patch_transport(monkeypatch, stub)

        _parse(target, audit)

        post = stub.find("POST", url_contains=adapter.FILE_URLS_BATCH_PATH)[0]
        assert post.headers["Authorization"] == "Bearer test-token"
        assert b"%PDF" not in (post.data or b"")

    def test_poll_runs_until_done(self, cloud_ready, monkeypatch):
        target, audit = cloud_ready
        stub = _MineruStub(poll_states=["running", "running", "done"])
        _patch_transport(monkeypatch, stub)

        document, record = _parse(target, audit)

        assert record.outcome == "success"
        assert document is not None
        assert stub.steps.count("GET") >= 3

    def test_a_poll_timeout_that_still_finds_a_result_is_partial_not_failed(
        self, cloud_ready, monkeypatch
    ):
        """降级路径：轮询放弃后补查一次，捡回结果就标 partial。

        这里用桩替换两个步骤而不是靠真实的轮询超时——``parse_document`` 不
        接受时钟注入，让测试真的等满 ``poll_timeout_seconds`` 会让整个套件
        慢上一分钟；而"拿到结果后标 partial"这个判断本身才是要钉住的东西。
        """
        target, audit = cloud_ready
        stub = _MineruStub()
        _patch_transport(monkeypatch, stub)
        monkeypatch.setattr(adapter, "poll_batch", lambda *_a, **_k: None)
        monkeypatch.setattr(adapter, "fetch_partial_result_url", lambda *_a, **_k: RESULT_ZIP_URL)

        document, record = _parse(target, audit)

        # 丢掉一份可能完整的文档、只因为等得不够久，比标成 partial 严重得多。
        assert document is not None
        assert record.outcome == "partial"
        assert "不完整" in record.message

    def test_a_poll_timeout_with_nothing_to_recover_fails_cleanly(
        self, cloud_ready, monkeypatch
    ):
        target, audit = cloud_ready
        _patch_transport(monkeypatch, _MineruStub())
        monkeypatch.setattr(adapter, "poll_batch", lambda *_a, **_k: None)
        monkeypatch.setattr(adapter, "fetch_partial_result_url", lambda *_a, **_k: None)

        document, record = _parse(target, audit)

        assert document is None
        assert record.outcome == "timed_out"
        assert "未完成" in record.message

    def test_poll_batch_gives_up_at_its_own_deadline(self, monkeypatch):
        """``poll_batch`` 超时返回 None 而不是抛异常——"降级"与"失败"的分界。
        假时钟让它在两次轮询内确定性地越过截止时间。"""
        clock = _FakeClock()
        _patch_transport(monkeypatch, lambda *_a, **_k: _json_response(_pending()))
        settings = _Settings(
            privacy_mode="allow_cloud",
            consent_granted=True,
            token="t",
            poll_timeout_seconds=10,
            poll_interval_seconds=5,
        )

        result = adapter.poll_batch(
            "batch-1", settings=settings, sleeper=clock.sleep, monotonic=clock.monotonic
        )

        assert result is None
        # 截止时间在每次请求**之后**才判定，所以最后一次 sleep 会把假时钟
        # 推到恰好等于 deadline，第三次轮询才放弃。
        assert clock.slept == [5, 5]
        assert clock.now == 10

    def test_a_cloud_failure_state_stops_polling_immediately(self, cloud_ready, monkeypatch):
        target, audit = cloud_ready
        stub = _MineruStub(poll_states=["failed"])
        _patch_transport(monkeypatch, stub)

        document, record = _parse(target, audit)

        assert document is None
        assert record.outcome == "failed"
        assert "failed" in record.message

    def test_the_result_zip_is_downloaded_and_normalized(self, cloud_ready, monkeypatch):
        target, audit = cloud_ready
        stub = _MineruStub()
        _patch_transport(monkeypatch, stub)

        document, _record = _parse(target, audit)

        assert stub.requests[-1][1] == RESULT_ZIP_URL
        assert [element.text for element in document.elements][:2] == [
            "论文标题",
            "第一段正文。",
        ]

    def test_the_document_carries_page_provenance_from_the_cloud_result(
        self, cloud_ready, monkeypatch
    ):
        target, audit = cloud_ready
        stub = _MineruStub()
        _patch_transport(monkeypatch, stub)

        document, _record = _parse(target, audit)

        # page_idx 是 0-based，locator 必须是 1-based 的可解析页码。
        pages = [element.locator.page for element in document.elements]
        assert pages == [1, 1, 2]
        assert document.page_count == 2
        assert all(element.locator.is_resolvable for element in document.elements)

    def test_the_batch_body_carries_nothing_but_the_documented_file_name(
        self, cloud_ready, monkeypatch
    ):
        """``file-urls/batch`` 的 ``files[]`` 只认 ``name``（外加可选的
        ``data_id`` / ``is_ocr`` / ``page_ranges``）。

        多发一个自己都没验证过的键不是"多点信息"：对端会把它一起参与签名或
        直接按未知字段处理，表现是 400 或一个查不出原因的签名不匹配，而错误
        信息永远不会指向真正的原因。整份 body 逐键钉住，比断言"没有某一个键"
        更严——将来有人加任何新字段都会在这里失败。
        """
        target, audit = cloud_ready
        stub = _MineruStub()
        _patch_transport(monkeypatch, stub)

        _parse(target, audit)

        post = stub.find("POST", url_contains=adapter.FILE_URLS_BATCH_PATH)[0]
        body = json.loads(post.data.decode("utf-8"))
        assert body == {"files": [{"name": "paper.pdf"}]}
        # 显式再钉一次：这个键曾经被发出去过，而它并不在接口文档里。
        assert "is_oss" not in body["files"][0]


class TestPresignedUploadLeg:
    """上传那一步与前三步不同：它打的是 OSS 这类**预签名**地址。

    预签名 URL 把"请求长什么样"也签了进去，所以这里的三条约束全都不是风格
    偏好，而是"少了它就发不上去"：不带 Content-Type、不跟随重定向、不带
    Authorization。下面的测试逐条钉住它们，因为它们全都是在真实 API 上才会
    暴露的 403/400，而 403 在别处通常意味着"token 没配好"——排查方向会完全
    跑偏。
    """

    def test_the_put_puts_no_media_type_on_the_wire(self, cloud_ready, monkeypatch):
        """上传的 Content-Type 必须是空的——**不能是"看起来差不多"的某个值**。

        Aliyun OSS 的 Signature V1 把 Content-Type 拼进待签串，而 MinerU 签
        这条 URL 时一个 Content-Type 都没有带；多带一个（哪怕是
        ``application/octet-stream``，看起来最"正确"的那个）都会得到
        ``SignatureDoesNotMatch``。因此这里断言的是"空"，不是"某个值"。
        """
        target, _audit = cloud_ready
        settings = _Settings(privacy_mode="allow_cloud", consent_granted=True, token="t")
        stub = _MineruStub()
        _patch_transport(monkeypatch, stub)

        adapter.upload_file_to_signed_url(str(target), UPLOAD_URL, settings=settings)

        # 断言的是 urllib 真正会写到 socket 上的那份头，而不是 ``Request.headers``：
        # 后者还没经过 ``do_request_``，也就是注入 Content-Type 的那一步，读它
        # 会让这条断言变成空话。
        media_type = _wire_headers(stub.find("PUT")[0]).get("Content-type", "")
        assert media_type == "", f"上传带了 Content-Type {media_type!r}，签名一定对不上"

    def test_the_put_carries_no_authorization_header(self, cloud_ready, monkeypatch):
        """签名信息已经在 URL 的 query 里了。

        再挂一个 Bearer 头过去，等于把 MinerU 的 token 交给一个第三方存储主机
        ——而那个主机的日志、缓存和告警系统都不是我们自己控制的。
        """
        target, _audit = cloud_ready
        settings = _Settings(privacy_mode="allow_cloud", consent_granted=True, token="t")
        stub = _MineruStub()
        _patch_transport(monkeypatch, stub)

        adapter.upload_file_to_signed_url(str(target), UPLOAD_URL, settings=settings)

        wire = {name.lower() for name in _wire_headers(stub.find("PUT")[0])}
        assert "authorization" not in wire

    def test_a_bare_put_would_have_gained_a_form_content_type(self):
        """解释那个空字符串为什么不能"顺手删掉"。

        一个带 body 却没有 Content-Type 的 Request 会被
        ``AbstractHTTPHandler`` 静默补上
        ``application/x-www-form-urlencoded``。所以"删掉那行 headers"得到的
        并不是"不带 Content-Type"，而是带了一个更隐蔽的 media type——正是本类
        开头说的那种 403。这条测试把那个注入行为本身钉住：哪天它变了，上面两条
        就会以"莫名其妙"的方式失败，而不是以"签名不匹配"的方式。
        """
        bare = urllib.request.Request(UPLOAD_URL, data=b"%PDF-1.7", method="PUT")
        blanked = urllib.request.Request(
            UPLOAD_URL, data=b"%PDF-1.7", headers={"Content-Type": ""}, method="PUT"
        )

        assert _wire_headers(bare)["Content-type"] == "application/x-www-form-urlencoded"
        assert _wire_headers(blanked)["Content-type"] == ""

    def test_the_upload_goes_through_the_no_redirect_opener_not_urlopen(
        self, cloud_ready, monkeypatch
    ):
        """上传这一腿**不经过** ``urllib.request.urlopen``。

        预签名 URL 的授权信息在 query string 里；交给默认 opener 就等于允许
        urllib 在它认为合适的时候跟随一次 30x，把 PUT 重放到一个通常已经丢掉
        query 的地址上。这里用一个"被调用就失败"的 urlopen 桩把这条约束钉死：
        哪天生产代码退回 urlopen，这个测试立刻红，而不是在真实 API 上变成一个
        403。
        """
        target, _audit = cloud_ready
        settings = _Settings(privacy_mode="allow_cloud", consent_granted=True, token="t")
        built: list[int] = []
        opened: list[str] = []

        def _transport(request, timeout=None):
            opened.append(request.full_url)
            return FakeResponse(b"", content_type=None, url=UPLOAD_URL)

        def _build(*_args, **_kwargs):
            built.append(1)
            return _FakeOpener(_transport)

        guard = _ExplodingUrlopen()
        monkeypatch.setattr(urllib.request, "urlopen", guard)
        monkeypatch.setattr(adapter, "_no_redirect_opener", _build)

        adapter.upload_file_to_signed_url(str(target), UPLOAD_URL, settings=settings)

        assert guard.calls == []
        assert built == [1]
        assert opened == [UPLOAD_URL]

    def test_a_redirect_from_the_upload_url_is_an_error_not_a_replay(
        self, allow_cloud, cloud_ready, monkeypatch
    ):
        """上传地址回 30x 时必须报错，而不是换一个 URL 把 PUT 再发一次。

        这次**不复用** :class:`_FakeOpener`：那样只能证明"适配器收到什么就
        抛什么"，证明不了 urllib 自己会怎么处理 30x。这里用真实的
        :class:`urllib.request.OpenerDirector`（带生产代码那个
        :class:`_NoRedirectHandler`），只把真正开 socket 的那一个 handler 换成
        内存响应——于是 302 是否被重放，完全由 urllib 的错误链决定。
        """
        target, _audit = cloud_ready
        settings = _Settings(privacy_mode="allow_cloud", consent_granted=True, token="t")
        canned = _CannedResponse(
            302, {"Location": "https://elsewhere.mineru.example.test/put/moved"}, UPLOAD_URL
        )
        transport = _StubbedTransport(canned)
        monkeypatch.setattr(
            adapter,
            "_no_redirect_opener",
            lambda *_a, **_k: urllib.request.build_opener(
                adapter._NoRedirectHandler(), transport, urllib.request.HTTPSHandler()
            ),
        )

        with pytest.raises(MineruError) as excinfo:
            adapter.upload_file_to_signed_url(str(target), UPLOAD_URL, settings=settings)

        # 错误必须被翻译成用户能看懂的话，而不是裸奔一个 HTTPError 出去。
        assert excinfo.value.code == "mineru_http_error"
        assert "302" in str(excinfo.value)
        # 关键：只问了原始地址一次。重定向目标一次都没被碰过。
        assert transport.seen == [UPLOAD_URL]
        # 302 不在可重试集合里，所以也不会"重试一次再报"。
        assert 302 not in adapter.RETRYABLE_STATUS_CODES

    @pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
    def test_the_no_redirect_handler_never_builds_a_replayable_request(self, code):
        """直接钉住 handler 本身：任何一种 30x 都不给出新的 Request。

        上一条端到端的测试在 CPython 3.13 上还"碰巧"成立——那一版的默认
        ``HTTPRedirectHandler`` 对非 GET/HEAD 请求本来就会抛 HTTPError，所以即便
        我们忘了传这个 handler，测试也可能是绿的。靠别的版本的巧合来保证安全
        是不行的：这里把"返回 None"这个动作本身 pin 住，于是"我们没有跟随
        重定向"始终是我们自己选的，而不是标准库某条规则的副作用。
        """
        request = urllib.request.Request(UPLOAD_URL, data=b"%PDF", method="PUT")

        assert (
            adapter._NoRedirectHandler().redirect_request(
                request, None, code, "Moved", Message(), "https://elsewhere.example.test/put"
            )
            is None
        )

    def test_a_successful_upload_through_a_real_opener_still_works(
        self, allow_cloud, cloud_ready, monkeypatch
    ):
        """上一条不能只测失败路径。

        用同样的真实 opener + 内存 200 响应跑一次成功上传，确认这条链路上
        我们加的 no-redirect opener 没有顺手把正常路径也掐掉（比如把请求
        变成不可读、或吞掉响应体）。
        """
        target, _audit = cloud_ready
        settings = _Settings(privacy_mode="allow_cloud", consent_granted=True, token="t")
        transport = _StubbedTransport(_CannedResponse(200, {}, UPLOAD_URL))
        monkeypatch.setattr(
            adapter,
            "_no_redirect_opener",
            lambda *_a, **_k: urllib.request.build_opener(
                adapter._NoRedirectHandler(), transport, urllib.request.HTTPSHandler()
            ),
        )

        adapter.upload_file_to_signed_url(str(target), UPLOAD_URL, settings=settings)

        assert transport.seen == [UPLOAD_URL]
        assert transport.methods == ["PUT"]


class TestSsrfAndUploadSafety:
    """上传 URL 与下载地址都来自第三方，因此必须过 SSRF 校验。"""

    def test_a_signed_url_pointing_at_loopback_is_refused(self, allow_cloud, tmp_path):
        """一个被篡改的响应能让本机对 127.0.0.1 或云元数据地址发起 PUT。"""
        import paperwhisperer.ingestion.adapters.mineru_cloud as module

        allow_cloud.setattr(
            module,
            "is_public_http_url",
            lambda url: not url.startswith("http://169.254.169.254"),
        )
        settings = _Settings(privacy_mode="allow_cloud", consent_granted=True, token="t")
        target = tmp_path / "paper.pdf"
        target.write_bytes(b"%PDF-1.7\n")

        with pytest.raises(MineruError) as excinfo:
            adapter.upload_file_to_signed_url(
                str(target), "http://169.254.169.254/latest/meta-data/", settings=settings
            )

        assert excinfo.value.code == "mineru_ssrf_blocked"

    def test_a_result_url_pointing_at_loopback_is_refused(self, allow_cloud):
        allow_cloud.setattr(adapter, "is_public_http_url", lambda _url: False)
        settings = _Settings(privacy_mode="allow_cloud", consent_granted=True, token="t")

        with pytest.raises(MineruError) as excinfo:
            adapter.download_result_zip("http://127.0.0.1:8000/result.zip", settings=settings)

        assert excinfo.value.code == "mineru_ssrf_blocked"

    def test_the_ssrf_check_runs_before_the_file_is_read(self, allow_cloud, tmp_path):
        """顺序反了会先把文档读进内存再拒绝——读本身不外泄，但白做功且
        让人误以为校验没跑。"""
        allow_cloud.setattr(adapter, "is_public_http_url", lambda _url: False)
        settings = _Settings(privacy_mode="allow_cloud", consent_granted=True, token="t")
        missing = tmp_path / "absent.pdf"

        with pytest.raises(MineruError) as excinfo:
            adapter.upload_file_to_signed_url(str(missing), "http://127.0.0.1/x", settings=settings)

        # SSRF 错误，不是"文件不存在"——说明确实在读文件之前就拒了。
        assert excinfo.value.code == "mineru_ssrf_blocked"


class TestRetryAndBackoff:
    """429 / 超时 / 5xx 重试；额度耗尽不重试。"""

    def test_a_429_is_retried_and_then_succeeds(self, cloud_ready, monkeypatch):
        target, audit = cloud_ready
        stub = _MineruStub()
        attempts: list = []

        def _open(request, timeout=None, **kwargs):
            attempts.append(request)
            if len(attempts) == 1:
                headers = Message()
                headers["Retry-After"] = "1"
                raise urllib.error.HTTPError(
                    request.full_url, 429, "Too Many Requests", headers, None
                )
            return stub(request, timeout=timeout, **kwargs)

        _patch_transport(monkeypatch, _open)

        document, record = _parse(target, audit, sleeper=lambda _s: None)

        assert record.outcome == "success"
        assert document is not None
        # 重放的是**同一个请求**：同一个 URL、同一个方法。签名这一步先吃了一次
        # 429、第二次才拿到凭证。
        signups = [a for a in attempts if a.get_method() == "POST"]
        assert len(signups) == 2
        assert signups[0].full_url == signups[1].full_url

    def test_a_429_honours_retry_after_rather_than_the_local_backoff(
        self, cloud_ready, monkeypatch
    ):
        target, audit = cloud_ready
        slept: list[float] = []
        calls = {"n": 0}

        def _open(request, timeout=None, **_kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                headers = Message()
                headers["Retry-After"] = "7"
                raise urllib.error.HTTPError(
                    request.full_url, 429, "Too Many Requests", headers, None
                )
            return _json_response(SIGNED_BATCH)

        _patch_transport(monkeypatch, _open)

        _parse(target, audit, sleeper=slept.append)

        # 对端最清楚自己多久恢复，听它的，而不是用本地指数退盖过去。
        assert slept[0] == 7
        assert adapter.RETRY_MAX_DELAY_SECONDS >= 7

    def test_a_timeout_is_retried(self, cloud_ready, monkeypatch):
        target, audit = cloud_ready
        calls = {"n": 0}
        stub = _MineruStub()

        def _open(request, timeout=None, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise TimeoutError("timed out")
            return stub(request, timeout=timeout, **kwargs)

        _patch_transport(monkeypatch, _open)

        document, record = _parse(target, audit, sleeper=lambda _s: None)

        assert record.outcome == "success"
        assert calls["n"] > 1

    def test_retries_are_bounded_and_the_last_error_surfaces(self, cloud_ready, monkeypatch):
        """退避必须有上限：把重试摊平成对第三方的持续压力比失败更糟。"""
        target, audit = cloud_ready
        monkeypatch.setattr(config, "MINERU_MAX_RETRIES", 3)
        calls = {"n": 0}

        def _open(request, timeout=None, **_kwargs):
            calls["n"] += 1
            headers = Message()
            raise urllib.error.HTTPError(request.full_url, 503, "Unavailable", headers, None)

        _patch_transport(monkeypatch, _open)
        slept: list[float] = []

        document, record = _parse(target, audit, sleeper=slept.append)

        assert document is None
        assert record.outcome == "failed"
        # 3 次重试 = 3 次尝试，不是 3+1。
        assert calls["n"] == 3
        assert max(slept) <= adapter.RETRY_MAX_DELAY_SECONDS

    def test_a_client_error_is_not_retried(self, cloud_ready, monkeypatch):
        """413 说明请求本身就不对，重放只会再花一次调用方的时间预算。"""
        target, audit = cloud_ready
        calls = {"n": 0}

        def _open(request, timeout=None, **_kwargs):
            calls["n"] += 1
            raise urllib.error.HTTPError(request.full_url, 413, "Too Large", Message(), None)

        _patch_transport(monkeypatch, _open)

        document, record = _parse(target, audit, sleeper=lambda _s: None)

        assert document is None
        assert calls["n"] == 1
        assert "413" in record.message

    def test_a_quota_402_is_not_retried_and_is_audited_as_quota(
        self, cloud_ready, monkeypatch
    ):
        """余额不会因为重试而补充，重试只会让用户多等一轮并再消耗一次检查。"""
        target, audit = cloud_ready
        calls = {"n": 0}

        def _open(request, timeout=None, **_kwargs):
            calls["n"] += 1
            raise urllib.error.HTTPError(request.full_url, 402, "Payment Required", Message(), None)

        _patch_transport(monkeypatch, _open)

        document, record = _parse(target, audit, sleeper=lambda _s: None)

        assert document is None
        assert calls["n"] == 1
        # 额度耗尽必须是可区分的 outcome，编排层据此提示充值而不是静默换 parser。
        assert record.outcome == "quota_exhausted"
        assert "额度" in record.message

    def test_a_business_code_reporting_quota_short_circuits_to_quota_exhausted(
        self, cloud_ready, monkeypatch
    ):
        """MinerU 的业务错误是 HTTP 200 + code != 0，只看状态码会把它们
        全当成成功。"""
        target, audit = cloud_ready
        stub = _MineruStub()
        calls = {"n": 0}

        def _open(request, timeout=None, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return _json_response({"code": 1002, "msg": "insufficient balance"})
            return stub(request, timeout=timeout, **kwargs)

        _patch_transport(monkeypatch, _open)

        document, record = _parse(target, audit, sleeper=lambda _s: None)

        assert document is None
        assert record.outcome == "quota_exhausted"
        assert calls["n"] == 1

    def test_a_non_quota_business_error_is_a_plain_failure(self, cloud_ready, monkeypatch):
        target, audit = cloud_ready
        stub = _MineruStub()
        calls = {"n": 0}

        def _open(request, timeout=None, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return _json_response({"code": 1001, "msg": "file not found"})
            return stub(request, timeout=timeout, **kwargs)

        _patch_transport(monkeypatch, _open)

        document, record = _parse(target, audit, sleeper=lambda _s: None)

        assert document is None
        assert record.outcome == "failed"
        assert "1001" in record.message

    def test_an_invalid_token_is_reported_as_a_config_problem(self, cloud_ready, monkeypatch):
        target, audit = cloud_ready

        def _open(request, timeout=None, **_kwargs):
            raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", Message(), None)

        _patch_transport(monkeypatch, _open)

        _document, record = _parse(target, audit, sleeper=lambda _s: None)

        assert "401" in record.message
        assert "Token" in record.message


class TestAudit:
    """每次调用写一条审计：操作者、时间、文件、触发原因、结果。"""

    def test_a_successful_call_writes_one_record(self, cloud_ready, monkeypatch):
        target, audit = cloud_ready
        _patch_transport(monkeypatch, _MineruStub())

        _parse(target, audit, trigger_reason="fallback")

        records = read_audit_records(audit)
        assert len(records) == 1
        assert records[0]["outcome"] == "success"
        assert records[0]["trigger_reason"] == "fallback"
        assert records[0]["source_filename"] == "paper.pdf"
        assert records[0]["timestamp"]

    def test_the_record_names_the_operator_file_and_digest(self, cloud_ready, monkeypatch):
        import hashlib

        target, audit = cloud_ready
        _patch_transport(monkeypatch, _MineruStub())

        _parse(target, audit, operator="alice@example.com")

        record = read_audit_records(audit)[0]
        assert record["operator"] == "alice@example.com"
        expected = hashlib.sha256(b"%PDF-1.7\nbody").hexdigest()
        assert record["sha256"] == expected
        assert record["parser_name"] == "mineru_cloud"

    def test_the_audit_never_contains_the_token(self, cloud_ready, monkeypatch):
        """审计日志的读取范围远大于 secret 的读取范围，token 出现在里面就是
        一次泄密。"""
        target, audit = cloud_ready
        _patch_transport(monkeypatch, _MineruStub())

        _parse(target, audit)

        assert "test-token" not in Path(audit).read_text(encoding="utf-8")

    @pytest.mark.parametrize(
        "outcome_setup",
        ["blocked", "no_token", "http_error"],
    )
    def test_a_failed_call_is_audited_too(self, cloud_ready, monkeypatch, outcome_setup):
        target, audit = cloud_ready
        if outcome_setup == "blocked":
            monkeypatch.setattr(config, "DOCUMENT_PRIVACY_MODE", "local_only")
        elif outcome_setup == "no_token":
            monkeypatch.setattr(config, "MINERU_API_TOKEN", "")
        else:
            def _open(request, timeout=None, **_kwargs):
                raise urllib.error.HTTPError(request.full_url, 400, "Bad Request", Message(), None)

            _patch_transport(monkeypatch, _open)

        _parse(target, audit)

        records = read_audit_records(audit)
        assert len(records) == 1
        assert records[0]["outcome"] != "success"
        assert records[0]["message"]

    def test_a_blocked_call_is_audited_as_blocked(self, tmp_path, monkeypatch):
        """被隐私策略挡住恰恰是运维最需要看到的一类记录；只记成功会把它抹掉。"""
        monkeypatch.setattr(config, "DOCUMENT_PRIVACY_MODE", "local_only")
        target = tmp_path / "paper.pdf"
        target.write_bytes(b"%PDF-1.7\n")
        audit = str(tmp_path / "audit.jsonl")

        _parse(target, audit, trigger_reason="manual")

        record = read_audit_records(audit)[0]
        assert record["outcome"] == "blocked"
        assert "local_only" in record["message"]

    def test_an_unexpected_exception_is_audited_rather_than_raised(self, cloud_ready, monkeypatch):
        """业务层的 fallback 几乎总在 try/except 里，异常会让"云端失败"和
        "代码 bug" 走同一条路。"""
        target, audit = cloud_ready

        def _open(request, timeout=None, **_kwargs):
            raise ValueError("something nobody planned for")

        _patch_transport(monkeypatch, _open)

        document, record = _parse(target, audit, sleeper=lambda _s: None)

        assert document is None
        assert record.outcome == "failed"
        assert "未预期错误" in record.message

    def test_an_unknown_trigger_reason_falls_back_to_manual(self, cloud_ready, monkeypatch):
        target, audit = cloud_ready
        _patch_transport(monkeypatch, _MineruStub())

        _parse(target, audit, trigger_reason="not-a-real-reason")

        assert read_audit_records(audit)[0]["trigger_reason"] == "manual"

    def test_audit_appends_rather_than_replacing(self, tmp_path):
        """用 atomic_write_json 写日志等于每次只留最后一条，审计就没了。"""
        path = str(tmp_path / "audit.jsonl")
        for index in range(3):
            append_audit_record(
                MineruAuditRecord(
                    timestamp=f"2026-01-01T00:00:0{index}",
                    operator="alice",
                    source_filename=f"f{index}.pdf",
                    sha256="x",
                    trigger_reason="manual",
                    outcome="success",
                ),
                log_path=path,
            )

        assert len(read_audit_records(path)) == 3

    def test_a_half_written_line_does_not_make_the_log_unreadable(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        append_audit_record(
            MineruAuditRecord(
                timestamp="t",
                operator="a",
                source_filename="f.pdf",
                sha256="x",
                trigger_reason="manual",
                outcome="success",
            ),
            log_path=str(path),
        )
        with open(path, "a", encoding="utf-8") as handle:
            handle.write('{"truncated": ')

        assert len(read_audit_records(str(path))) == 1

    def test_a_write_failure_does_not_fail_the_parse(self, cloud_ready, monkeypatch, tmp_path):
        """因为只读文件系统或磁盘满而让用户的解析请求报错，代价远大于丢一条审计。

        审计路径故意指向一个**已存在的文件**当目录用：``open(..., "a")`` 会
        稳定地抛 ``NotADirectoryError``，是真实的 ``OSError``，不需要任何替身。
        """
        target, _audit = cloud_ready
        _patch_transport(monkeypatch, _MineruStub())
        blocker = tmp_path / "not-a-directory"
        blocker.write_text("我是一个文件，不是目录", encoding="utf-8")
        unwritable = str(blocker / "nested" / "audit.jsonl")

        document, record = _parse(target, unwritable)

        assert document is not None
        assert record.outcome == "success"

    def test_the_audit_lives_beside_the_artifact_dir_not_inside_it(self, tmp_path, monkeypatch):
        """artifact 目录的语义是"可删除的中间产物"，清理脚本会整目录删掉；
        审计放一起会被一起清掉。"""
        artifact_dir = tmp_path / "context" / "parser_artifacts"
        artifact_dir.mkdir(parents=True)
        monkeypatch.setattr(config, "PARSER_ARTIFACT_DIR", str(artifact_dir))

        path = Path(resolve_audit_log_path())

        assert path.parent == artifact_dir.parent
        assert path.name == "mineru_audit.jsonl"

    def test_the_operator_falls_back_to_the_environment(self, monkeypatch):
        monkeypatch.setenv("PAPERWHISPERER_OPERATOR", "carol")

        assert adapter._resolve_operator(None) == "carol"

    def test_the_operator_is_never_derived_from_a_secret(self, monkeypatch):
        """审计里出现一个 API key 比出现 unknown 严重得多。"""
        monkeypatch.delenv("PAPERWHISPERER_OPERATOR", raising=False)
        monkeypatch.delenv("USER", raising=False)
        monkeypatch.delenv("USERNAME", raising=False)
        monkeypatch.setattr(config, "MINERU_API_TOKEN", "secret-token")

        assert adapter._resolve_operator(None) == "unknown"

    def test_an_unreadable_source_file_still_produces_an_audit_record(self, tmp_path, monkeypatch):
        """审计不能因为源文件读不到就整个失败——sha256 记空串即可。"""
        monkeypatch.setattr(config, "DOCUMENT_PRIVACY_MODE", "local_only")
        audit = str(tmp_path / "audit.jsonl")

        _parse(tmp_path / "does-not-exist.pdf", audit, source_filename="missing.pdf")

        record = read_audit_records(audit)[0]
        assert record["sha256"] == ""
        assert record["source_filename"] == "missing.pdf"


class TestZipDefenses:
    """结果 zip 来自第三方，所以三个维度都要卡，路径穿越也要挡。"""

    @pytest.mark.parametrize(
        "name",
        ["../../etc/passwd", "/etc/passwd", "..", "a/../b", "C:/Windows/system32", "a/.."],
    )
    def test_a_traversing_member_name_is_rejected(self, name):
        assert _is_safe_member_name(name) is False

    @pytest.mark.parametrize("name", ["full.md", "a/b/content_list.json", "out.md"])
    def test_an_ordinary_member_name_is_accepted(self, name):
        assert _is_safe_member_name(name) is True

    def test_a_traversing_member_is_never_read_into_the_result(self):
        """即使它带着 .md 后缀，穿越路径也不该进结果。"""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("../../etc/passwd", "root:x:0:0")
            archive.writestr("ok.md", "# 正常内容")

        payload = extract_result_payload(buffer.getvalue())

        assert payload["markdown"] == "# 正常内容"
        assert "root:x" not in payload["markdown"]

    def test_too_many_entries_is_rejected(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for index in range(MAX_ZIP_ENTRIES + 1):
                archive.writestr(f"f{index}.md", "x")

        with pytest.raises(MineruError) as excinfo:
            extract_result_payload(buffer.getvalue())

        # 海量小条目打满的是内存/inode，不是单条目大小。
        assert excinfo.value.code == "mineru_zip_bomb"
        assert "条目过多" in str(excinfo.value)

    def test_an_oversized_declared_entry_is_rejected_before_it_is_decompressed(self):
        """声明 4GB 的条目必须在**声明阶段**被拒，而不是先花掉 4GB 内存
        再发现。"""
        buffer = _zip_with_declared_size("big.md", MAX_ZIP_ENTRY_UNCOMPRESSED_BYTES + 1)

        with pytest.raises(MineruError) as excinfo:
            validate_zip_bomb(buffer)

        assert excinfo.value.code == "mineru_zip_bomb"
        assert "声明解压后超过" in str(excinfo.value)

    def test_a_declared_total_larger_than_the_cap_is_rejected(self):
        """单看每个都不大但累加超限，是叠加密那一类。"""
        forged = _zip_exceeding_total_cap(MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES)

        with pytest.raises(MineruError) as excinfo:
            validate_zip_bomb(forged)

        assert excinfo.value.code == "mineru_zip_bomb"
        assert "总解压大小" in str(excinfo.value)

    def test_a_nested_zip_is_not_reopened(self):
        """内层 zip 再解一层是下一层攻击面，这里按名字白名单取，从不递归。"""
        inner = io.BytesIO()
        with zipfile.ZipFile(inner, "w") as archive:
            archive.writestr("inner.md", "# 嵌套")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("outer.zip", inner.getvalue())
            archive.writestr("full.md", "# 外层")

        payload = extract_result_payload(buffer.getvalue())

        assert payload["markdown"] == "# 外层"
        assert "嵌套" not in payload["markdown"]

    def test_a_corrupt_archive_is_reported_not_crashed(self):
        with pytest.raises(MineruError) as excinfo:
            extract_result_payload(b"this is not a zip file at all")

        assert excinfo.value.code == "mineru_bad_zip"

    def test_an_archive_with_no_recognisable_content_is_an_explicit_error(self):
        """返回空结果会让下游以为"解析成功但文档是空的"，必须说清楚。"""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("layout.png", b"\x89PNG")

        with pytest.raises(MineruError) as excinfo:
            extract_result_payload(buffer.getvalue())

        assert excinfo.value.code == "mineru_empty_result"

    def test_a_result_larger_than_the_download_cap_is_aborted(self, monkeypatch):
        """不信 Content-Length：实际流量才是唯一可信的量。"""
        settings = _Settings(privacy_mode="allow_cloud", consent_granted=True, token="t")
        monkeypatch.setattr(adapter, "is_public_http_url", lambda _url: True)
        _patch_transport(
            monkeypatch,
            lambda *_args, **_kwargs: FakeResponse(
                b"0" * (adapter.MAX_RESULT_ZIP_BYTES + 1), url=RESULT_ZIP_URL
            ),
        )

        with pytest.raises(MineruError) as excinfo:
            adapter.download_result_zip(RESULT_ZIP_URL, settings=settings, sleeper=lambda _s: None)

        assert excinfo.value.code == "mineru_response_too_large"


class TestNormalize:
    """normalize 是纯函数——最容易出错也最难在真实链路上调试的一段。"""

    def test_a_table_keeps_its_grid(self):
        payload = {
            "content_list": [
                {
                    "type": "table",
                    "table_body": "<tr><td>指标</td><td>值</td></tr>"
                    "<tr><td>Accuracy</td><td>0.91</td></tr>",
                    "table_caption": "主要结果",
                    "page_idx": 0,
                }
            ]
        }

        document = normalize_mineru_payload(payload)
        element = document.elements[0]

        # 拍平成一段文本之后，跨列检索就不可能了。
        assert element.role == "table"
        assert element.table.rows == [["指标", "值"], ["Accuracy", "0.91"]]
        assert element.table.caption == "主要结果"

    def test_a_table_as_the_first_element_does_not_crash(self):
        """content_list 第一项是表格是常见的；曾经因为 text 只在部分分支赋值
        而直接 UnboundLocalError。"""
        payload = {"content_list": [{"type": "table", "table_body": "<tr><td>a</td></tr>"}]}

        document = normalize_mineru_payload(payload)

        assert document.elements[0].role == "table"
        assert document.elements[0].table.rows == [["a"]]

    def test_a_table_does_not_inherit_the_previous_paragraph_text(self):
        """表格元素曾经会沿用上一段正文的 text，于是正文重复一次、
        表格自己的内容反而丢失。"""
        payload = {
            "content_list": [
                {"type": "text", "text": "上一段正文。"},
                {"type": "table", "table_body": "<tr><td>a</td><td>b</td></tr>"},
            ]
        }

        document = normalize_mineru_payload(payload)
        table_element = document.elements[1]

        assert "上一段正文。" not in table_element.text
        assert table_element.text == table_element.table.to_markdown()

    def test_a_formula_keeps_its_latex_and_does_not_inherit_previous_text(self):
        payload = {
            "content_list": [
                {"type": "text", "text": "上一段正文。"},
                {"type": "equation", "text": "$E = mc^2$"},
            ]
        }

        document = normalize_mineru_payload(payload)
        formula = document.elements[1]

        assert formula.role == "formula"
        assert formula.latex == "E = mc^2"
        assert "上一段正文。" not in formula.text

    def test_text_level_marks_a_heading(self):
        payload = {
            "content_list": [
                {"type": "text", "text": "论文标题", "text_level": 1},
                {"type": "text", "text": "方法", "text_level": 2},
                {"type": "text", "text": "我们做了实验。"},
            ]
        }

        document = normalize_mineru_payload(payload)

        assert [element.role for element in document.elements] == [
            "title",
            "heading",
            "paragraph",
        ]
        assert document.elements[2].heading_path == ("论文标题", "方法")

    def test_a_bbox_is_normalized_to_positive_extent(self):
        payload = {
            "content_list": [
                {"type": "text", "text": "x", "page_idx": 0, "bbox": [300, 40, 10, 10]}
            ]
        }

        document = normalize_mineru_payload(payload)
        box = document.elements[0].locator.bbox

        assert (box.x0, box.y0) == (10.0, 10.0)
        assert (box.x1, box.y1) == (300.0, 40.0)

    @pytest.mark.parametrize("bbox", [None, [], [1, 2], ["a", 2, 3, 4], "not a list"])
    def test_an_unusable_bbox_yields_no_box(self, bbox):
        # 伪造一个框会让 evidence viewer 高亮到无关文本上。
        payload = {"content_list": [{"type": "text", "text": "x", "bbox": bbox}]}

        document = normalize_mineru_payload(payload)

        assert document.elements[0].locator.bbox is None

    def test_a_figure_takes_its_caption(self):
        payload = {"content_list": [{"type": "image", "img_caption": ["图 1", "架构图"]}]}

        document = normalize_mineru_payload(payload)

        assert document.elements[0].role == "figure"
        assert document.elements[0].text == "图 1 架构图"

    def test_a_page_number_becomes_furniture(self):
        payload = {"content_list": [{"type": "page_number", "text": "1"}]}

        document = normalize_mineru_payload(payload)

        assert document.elements[0].role == "footer"
        assert document.elements[0].is_furniture

    def test_an_unknown_type_degrades_to_text(self):
        """MinerU 加一个新 type 不该让整篇解析失败。"""
        payload = {"content_list": [{"type": "some_future_block", "text": "内容"}]}

        document = normalize_mineru_payload(payload)

        assert document.elements[0].role in {"text", "paragraph"}
        assert document.elements[0].text == "内容"

    def test_an_empty_shell_element_is_dropped(self):
        """空壳元素会让 char_count 与 locator 覆盖率失真。"""
        payload = {"content_list": [{"type": "text", "text": "   "}, {"type": "text", "text": "有内容"}]}

        document = normalize_mineru_payload(payload)

        assert [element.text for element in document.elements] == ["有内容"]

    def test_markdown_is_used_when_there_is_no_content_list(self):
        """降级路径只有 txt_offset 定位，但至少标题层级保住了。"""
        payload = {"markdown": "# 标题\n\n第一段。\n\n## 小节\n\n第二段。"}

        document = normalize_mineru_payload(payload)

        assert [element.role for element in document.elements] == [
            "title",
            "paragraph",
            "heading",
            "paragraph",
        ]
        assert all(
            element.locator.kind == "txt_offset" and element.locator.is_resolvable
            for element in document.elements
        )

    def test_an_empty_result_says_so_instead_of_looking_clean(self):
        document = normalize_mineru_payload({})

        assert document.elements == []
        assert any("未返回任何可用内容" in warning for warning in document.warnings)

    def test_page_count_is_derived_from_the_result(self):
        payload = {"content_list": [{"type": "text", "text": "x", "page_idx": 4}]}

        document = normalize_mineru_payload(payload)

        assert document.page_count == 5

    def test_an_element_with_a_page_index_is_still_a_pdf_page(self):
        """有页码的照旧是 ``pdf_page``——新规则只针对"没有页码"这一种情况。

        钉住它是因为修复很容易过头：把 ``page=None`` 一律降级成 ``unknown``，
        看起来"更诚实"，却会毁掉所有页码信息完整的解析结果，让 evidence viewer
        在最常见的场景下退化成什么都定位不了。
        """
        payload = {
            "content_list": [
                {"type": "text", "text": "x", "page_idx": 2, "bbox": [1, 2, 300, 40]}
            ]
        }

        locator = normalize_mineru_payload(payload).elements[0].locator

        assert locator.kind == "pdf_page"
        assert locator.page == 3  # page_idx 是 0-based，locator 是 1-based
        assert locator.is_resolvable is True
        assert (locator.bbox.x0, locator.bbox.y0) == (1.0, 2.0)
        assert (locator.bbox.x1, locator.bbox.y1) == (300.0, 40.0)

    def test_an_element_without_a_page_index_is_not_declared_as_a_pdf_page(self):
        """没有 ``page_idx`` 时必须说"我不知道"，而不是声明一个空页码。

        曾经这里无条件写 ``kind="pdf_page"``、``page=None``，于是 locator 声称
        自己是一个 PDF 页定位却解析不出任何页码：``is_resolvable`` 勉强返回
        False，但 ``kind`` 本身是假的，下游按 kind 分支的代码会走进"我有页码"
        的分支然后什么都拿不到——契约层只能事后把整个元素降级，掩盖了生产端的
        这一个 bug。正确的做法是在生产端就说实话。
        """
        payload = {"content_list": [{"type": "text", "text": "x"}]}

        locator = normalize_mineru_payload(payload).elements[0].locator

        assert locator.kind == "unknown"
        assert locator.page is None
        assert locator.is_resolvable is False

    def test_a_pageless_element_keeps_the_bbox_it_does_have(self):
        """丢页码是诚实的，丢坐标不是：bbox 是实打实拿到的信息。

        ``bbox`` 在 ``pdf_page`` 和 ``unknown`` 两种 kind 下都是合法字段，所以
        "没有页码"不构成"连框一起扔掉"的理由——扔掉它会让本来能高亮的位置变成
        高亮不了。
        """
        payload = {"content_list": [{"type": "text", "text": "x", "bbox": [10, 10, 300, 40]}]}

        locator = normalize_mineru_payload(payload).elements[0].locator

        assert locator.kind == "unknown"
        assert (locator.bbox.x0, locator.bbox.y0) == (10.0, 10.0)
        assert (locator.bbox.x1, locator.bbox.y1) == (300.0, 40.0)

    def test_a_page_idx_of_zero_is_a_page_not_a_missing_page(self):
        """``page_idx=0`` 落在 falsy 分支上的经典陷阱。

        0 是**第一页**，不是"没有页码"。用 ``if page:`` 判断时它和 ``None`` 走
        同一条路，于是整篇只有第一页的文档会被降级成 unknown。
        """
        payload = {"content_list": [{"type": "text", "text": "x", "page_idx": 0}]}

        locator = normalize_mineru_payload(payload).elements[0].locator

        assert locator.kind == "pdf_page"
        assert locator.page == 1

    def test_a_negative_page_index_does_not_become_page_zero(self):
        """``page_idx=-1`` 是脏数据，不能被当成"第 0 页"——那会指向真实存在的
        某一页，让引用指向错误的位置。"""
        payload = {"content_list": [{"type": "text", "text": "x", "page_idx": -1}]}

        locator = normalize_mineru_payload(payload).elements[0].locator

        assert locator.kind == "unknown"
        assert locator.page is None

    def test_an_impossible_page_number_is_warned_about_not_silently_accepted(self):
        payload = {
            "content_list": [{"type": "text", "text": "x", "page_idx": 9}],
        }

        document = normalize_mineru_payload(payload, page_count=3)

        assert any("超出" in warning for warning in document.warnings)

    def test_the_result_is_a_serializable_parsed_document(self):
        payload = {
            "content_list": [
                {"type": "title", "text": "标题", "page_idx": 0},
                {"type": "table", "table_body": "<tr><td>a</td></tr>", "page_idx": 0},
            ]
        }

        document = normalize_mineru_payload(payload, source_filename="p.pdf", sha256="abc")
        restored = ParsedDocument.from_dict(document.to_dict())

        assert restored.elements[0].role == "title"
        assert restored.elements[1].table.rows == [["a"]]
        assert restored.sha256 == "abc"

    def test_normalize_never_touches_the_network(self, monkeypatch):
        """这是它被写成纯函数的原因：网络完全解耦才能用手工 payload 直接断言。"""
        guard = _ExplodingUrlopen()
        _patch_transport(monkeypatch, guard)

        normalize_mineru_payload({"content_list": [{"type": "text", "text": "x"}]})

        assert guard.calls == []


class TestParserSurface:
    """``MineruCloudParser`` 是网关看到的那个对象。"""

    @pytest.mark.parametrize(
        ("media_type", "expected"),
        [
            ("application/pdf", True),
            ("application/pdf; charset=binary", True),
            ("APPLICATION/PDF", True),
            ("image/png", True),
            ("image/jpeg", True),
            # DOCX/PPTX 是结构化格式，本地解析同样准确，而发出去是隐私代价
            # 最高的一种情形。
            ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", False),
            ("text/plain", False),
            ("", False),
        ],
    )
    def test_supports_only_the_formats_worth_sending(self, media_type, expected):
        assert MineruCloudParser().supports(media_type) is expected

    def test_it_reports_a_name_and_a_version_without_a_token(self):
        parser = MineruCloudParser()

        assert parser.name == "mineru_cloud"
        assert isinstance(parser.version, str) and parser.version

    def test_it_satisfies_the_gateway_parser_protocol(self):
        """结构化地钉住网关要求的那个接口面。

        这里用鸭子类型而不是 ``isinstance(parser, DocumentParser)``：那个协议
        标了 ``@runtime_checkable``，而 ``runtime_checkable`` **只检查方法是否
        存在**，不检查签名也不检查返回值——所以它对一个返回类型完全不对的
        parser 同样会返回 True。真正把关的是
        ``ParserRouter._normalize`` 里的 ``isinstance(raw, ParseResult)``，
        下面的 ``test_the_parse_signature_is_callable_by_the_router`` 才是与
        它对齐的那条。
        """
        from paperwhisperer.ingestion.gateway import DocumentParser

        parser = MineruCloudParser()

        assert isinstance(parser.name, str) and parser.name
        assert isinstance(parser.version, str) and parser.version
        assert callable(parser.supports) and callable(parser.parse)
        # 协议确实接受它——但这个断言很弱，见上面的说明。
        assert isinstance(parser, DocumentParser)

    def test_the_parse_signature_is_callable_by_the_router(self):
        """网关是 ``executor.submit(parser.parse, file_path, options)``——
        两个位置参数、都能按位置传进去。签名对不上会在这里就抛 TypeError，
        而不是等到解析时才炸。"""
        import inspect

        signature = inspect.signature(MineruCloudParser.parse)

        assert list(signature.parameters) == ["self", "file_path", "options"]
        assert signature.parameters["file_path"].default is inspect.Parameter.empty
        # options 允许缺省，router 两种调用方式都得能用。
        assert signature.parameters["options"].default is None

    def test_the_gateway_can_actually_dispatch_to_it(self):
        """``supports`` 必须能在不碰文件的前提下回答"是不是我的格式"——
        网关对每个候选 parser 都会问一次，开一次文件来回答是荒谬的。"""
        parser = MineruCloudParser()

        assert parser.supports("application/pdf") is True
        assert parser.supports("text/plain") is False
        # 问两次答案一致：它必须是纯粹的、无状态的判断。
        assert parser.supports("application/pdf") is True

    def test_parse_returns_the_document_and_its_audit_record(self, cloud_ready, monkeypatch):
        target, audit = cloud_ready
        _patch_transport(monkeypatch, _MineruStub())
        parser = MineruCloudParser(audit_log_path=audit, sleeper=lambda _s: None)

        document, record = parser.parse(str(target), None)

        assert isinstance(document, ParsedDocument)
        assert isinstance(record, MineruAuditRecord)
        assert record.outcome == "success"

    def test_parse_reads_its_options_from_a_dict(self, cloud_ready, monkeypatch):
        """options 可能是 dict、dataclass 或 None，三种都要工作。"""
        target, audit = cloud_ready
        _patch_transport(monkeypatch, _MineruStub())
        parser = MineruCloudParser(audit_log_path=audit, sleeper=lambda _s: None)

        _document, record = parser.parse(
            str(target),
            {"source_filename": "重命名.pdf", "media_type": "image/png", "trigger_reason": "forced"},
        )

        assert record.source_filename == "重命名.pdf"
        assert record.trigger_reason == "forced"

    def test_parse_reads_its_options_from_an_object(self, cloud_ready, monkeypatch):
        target, audit = cloud_ready
        _patch_transport(monkeypatch, _MineruStub())
        parser = MineruCloudParser(audit_log_path=audit, sleeper=lambda _s: None)

        class _Options:
            source_filename = "来自对象.pdf"
            trigger_reason = "fallback"

        _document, record = parser.parse(str(target), _Options())

        assert record.source_filename == "来自对象.pdf"
        assert record.trigger_reason == "fallback"

    def test_parse_survives_options_being_none(self, cloud_ready, monkeypatch):
        target, audit = cloud_ready
        _patch_transport(monkeypatch, _MineruStub())
        parser = MineruCloudParser(audit_log_path=audit, sleeper=lambda _s: None)

        document, _record = parser.parse(str(target))

        assert isinstance(document, ParsedDocument)

    def test_the_constructor_operator_wins_over_options(self, cloud_ready, monkeypatch):
        """请求级构造的 operator 代表已认证的会话身份，不该被 options 覆盖。"""
        target, audit = cloud_ready
        _patch_transport(monkeypatch, _MineruStub())
        parser = MineruCloudParser(
            operator="session-user", audit_log_path=audit, sleeper=lambda _s: None
        )

        _document, record = parser.parse(str(target), {"operator": "spoofed"})

        assert record.operator == "session-user"


class TestErrorMapping:
    def test_the_business_code_shell_is_checked_even_on_http_200(self):
        """只看状态码会把所有业务失败当成成功——这是 MinerU 最容易踩的坑。"""
        with pytest.raises(MineruError) as excinfo:
            adapter._check_api_code({"code": 1001, "msg": "文件不存在"}, "申请上传凭证")

        assert excinfo.value.code == "mineru_business_error"
        assert "1001" in str(excinfo.value)

    def test_a_zero_code_passes_through(self):
        payload = {"code": 0, "data": {"batch_id": "b"}}

        assert adapter._check_api_code(payload, "申请上传凭证") is payload

    def test_a_non_json_response_is_reported(self, monkeypatch):
        monkeypatch.setattr(adapter, "is_public_http_url", lambda _url: True)
        _patch_transport(
            monkeypatch,
            lambda *_a, **_k: FakeResponse(b"<html>maintenance</html>", url=FAKE_BASE_URL),
        )
        settings = _Settings(privacy_mode="allow_cloud", consent_granted=True, token="t")

        with pytest.raises(MineruError) as excinfo:
            adapter.request_upload_urls(filename="a.pdf", settings=settings, sleeper=lambda _s: None)

        assert excinfo.value.code == "mineru_bad_json"

    def test_an_incomplete_batch_response_is_rejected(self, monkeypatch):
        _patch_transport(
            monkeypatch,
            lambda *_a, **_k: _json_response({"code": 0, "data": {"batch_id": "b"}}),
        )
        settings = _Settings(privacy_mode="allow_cloud", consent_granted=True, token="t")

        with pytest.raises(MineruError) as excinfo:
            adapter.request_upload_urls(filename="a.pdf", settings=settings, sleeper=lambda _s: None)

        assert excinfo.value.code == "mineru_incomplete_batch"

    def test_a_missing_result_url_on_done_is_an_error_not_a_silent_success(self, monkeypatch):
        _patch_transport(
            monkeypatch,
            lambda *_a, **_k: _json_response(
                {"code": 0, "data": {"extract_result": [{"state": "done"}]}}
            ),
        )
        settings = _Settings(privacy_mode="allow_cloud", consent_granted=True, token="t")

        with pytest.raises(MineruError) as excinfo:
            adapter.poll_batch("batch-1", settings=settings, sleeper=lambda _s: None)

        assert excinfo.value.code == "mineru_missing_result_url"

    def test_a_consent_error_is_distinguishable_from_a_cloud_failure(self):
        """编排层需要区分"被隐私挡住，换个本地 parser 就能继续"和
        "云端真的出错了"——前者应静默降级。"""
        assert issubclass(MineruConsentError, MineruError)
        assert MineruConsentError is not MineruError

    @pytest.mark.parametrize("state", ["failed", "error", "cancelled", "canceled"])
    def test_every_terminal_failure_state_stops_polling(self, monkeypatch, state):
        calls = {"n": 0}

        def _open(*_args, **_kwargs):
            calls["n"] += 1
            return _json_response(
                {"code": 0, "data": {"extract_result": [{"state": state}]}}
            )

        _patch_transport(monkeypatch, _open)
        settings = _Settings(privacy_mode="allow_cloud", consent_granted=True, token="t")

        with pytest.raises(MineruError) as excinfo:
            adapter.poll_batch("batch-1", settings=settings, sleeper=lambda _s: None)

        assert excinfo.value.code == "mineru_task_failed"
        assert calls["n"] == 1


class TestModuleBoundaries:
    def test_the_module_never_imports_a_heavy_parser_at_scope(self):
        """把这条钉住，免得以后有人顺手加一行 ``import docling``，
        结果为了测试一个纯 urllib 模块先装一整套 torch。"""
        source = Path(adapter.__file__).read_text(encoding="utf-8")

        for line in source.splitlines():
            if line.startswith(("import docling", "from docling", "import requests", "import httpx")):
                assert line.startswith((" ", "\t")), line

    def test_the_advertised_resource_limits_are_the_ones_the_tests_assert(self):
        """上限写在注释里、代码里各一份时会漂；测试引用的是代码里那份。"""
        assert MAX_ZIP_ENTRIES == adapter.MAX_ZIP_ENTRIES
        assert MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES == adapter.MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES
        assert MAX_ZIP_ENTRY_UNCOMPRESSED_BYTES == adapter.MAX_ZIP_ENTRY_UNCOMPRESSED_BYTES
        assert adapter.MAX_RESULT_ZIP_BYTES > 0


# --------------------------------------------------------------------------
# 构造带伪造声明值的 zip
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# 构造带伪造声明值的 zip
# --------------------------------------------------------------------------

#: zip 中央目录文件头的签名。
_CENTRAL_SIGNATURE = b"PK\x01\x02"
#: 中央目录头里"解压后大小"字段相对签名的偏移（4 字节小端）。
_CENTRAL_UNCOMPRESSED_SIZE_OFFSET = 24
#: 文件名长度字段相对签名的偏移（2 字节小端）。
_CENTRAL_NAME_LENGTH_OFFSET = 28
#: 文件名相对签名的偏移。
_CENTRAL_NAME_OFFSET = 46


def _zip_with_declared_size(name: str, declared_size: int) -> zipfile.ZipFile:
    """造一个**中央目录里声明**了超大解压大小的 zip。

    zip 格式不强制声明值与实际内容一致，所以这正是 zip bomb 的真实形状：
    它必须在声明阶段就被拒，而不是先花掉 4GB 内存再发现。
    """
    return _forge_declared_sizes(_real_zip({name: b"small"}), {name: declared_size})


def _real_zip(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return buffer.getvalue()


def _forge_declared_sizes(payload: bytes, sizes: dict[str, int]) -> zipfile.ZipFile:
    """把中央目录里指定条目的"解压后大小"改写掉。

    直接构造这样的 zip 做不到——``writestr`` 永远按真实内容写大小——所以
    测试只能在字节层面改，而字节层面也正是真实攻击者的做法。改完之后
    ``infolist()`` 读到的就是伪造值，CRC 因此不再匹配，但``validate_zip_bomb``
    只读中央目录，这正是它应该成立的地方。
    """
    raw = bytearray(payload)
    wanted = dict(sizes)
    offset = 0
    while True:
        index = bytes(raw).find(_CENTRAL_SIGNATURE, offset)
        if index == -1:
            break
        name_length = int.from_bytes(
            raw[index + _CENTRAL_NAME_LENGTH_OFFSET : index + _CENTRAL_NAME_LENGTH_OFFSET + 2],
            "little",
        )
        name = bytes(
            raw[index + _CENTRAL_NAME_OFFSET : index + _CENTRAL_NAME_OFFSET + name_length]
        ).decode("utf-8")
        if name in wanted:
            start = index + _CENTRAL_UNCOMPRESSED_SIZE_OFFSET
            raw[start : start + 4] = wanted[name].to_bytes(4, "little")
            del wanted[name]
        offset = index + _CENTRAL_NAME_OFFSET + name_length
    assert not wanted, f"这些条目没能在中央目录里找到：{sorted(wanted)}"
    return zipfile.ZipFile(io.BytesIO(bytes(raw)))


def _zip_exceeding_total_cap(cap: int) -> zipfile.ZipFile:
    """造一个"每个条目都不超限、但加起来超限"的 zip。

    这是叠加密那一类，只卡单条目上限会漏掉它，所以三个维度必须各自有测试。
    条目数按单条目上限反推，保证**先撞上总上限而不是单条目上限**——否则这条
    测试会在错误的分支上通过。条目内容保持很小：上限看的是**声明值**，而
    声明值正是攻击者能任意伪造的那一部分。
    """
    per_entry = MAX_ZIP_ENTRY_UNCOMPRESSED_BYTES - 1
    names = [f"f{index}.md" for index in range(cap // per_entry + 1)]
    assert per_entry * len(names) > cap
    assert per_entry < MAX_ZIP_ENTRY_UNCOMPRESSED_BYTES
    return _forge_declared_sizes(
        _real_zip({name: b"x" for name in names}), {name: per_entry for name in names}
    )
