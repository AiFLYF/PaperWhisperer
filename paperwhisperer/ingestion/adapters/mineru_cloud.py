"""MinerU 云端解析适配器：全项目唯一主动出网到第三方服务的模块。

因为它是唯一出网点，这个模块的第一职责不是"把 PDF 解析出来"，而是
**保证不该出网的时候一个字节都不出网**。隐私门控因此被放在
:meth:`MineruCloudParser.parse` 的最前面，早于任何 ``urllib`` 调用、任何
文件读取、任何 sha256 计算之外的准备工作。

四道约束，按执行顺序：

1. **策略层** ``DOCUMENT_PRIVACY_MODE == "allow_cloud"``。默认 ``local_only``，
   非法值也回落 ``local_only``（不确定即禁止出网）。
2. **用户层** ``DOCUMENT_CLOUD_CONSENT_GRANTED is True``。策略允许只说明
   "系统允许"，用户同意才说明"这一次可以"。
3. **凭据层** ``MINERU_API_TOKEN`` 非空。没配 token 就明确失败，而不是发一个
   注定 401 的请求去浪费一次额度。
4. **审计层** 无论成功、失败还是被拦截，都写一条 :class:`MineruAuditRecord`。
   被拦截也写（``outcome="blocked"``），因为"用户点了云解析但被隐私策略挡住"
   正是运维最需要看到的一类记录；只记成功会把这部分完全抹掉。

审计里**绝不写 token**：``describe_parser_settings`` 同样只暴露
``mineru_configured`` 布尔值，两处保持一致，否则审计日志本身就成了泄密面。

全链路（计划指定）::

    file-urls/batch 签名 → PUT 上传 → 轮询任务状态 → 下载结果 zip → normalize

其中两处安全设计值得单独说明：

- **上传 URL 来自第三方响应**，所以 PUT 之前必须过
  :func:`~paperwhisperer.remote.ssrf.is_public_http_url`。否则一个被篡改或
  恶意构造的 ``url`` 就能让本机对内网发起请求（SSRF）。zip 下载 URL 同理。
- **结果 zip 来自第三方**，所以解压前限制条目数、单条目解压大小、总解压大小。
  计划把 zip bomb 防护列在 P4，但这里的输入是外部 zip，防护必须在这里就做。

降级策略：429 / 超时 / 5xx 退避重试（优先读 ``Retry-After``）；**明确的额度
耗尽不重试**——重试只会再消耗一次时间并得到同样的 4xx，而且会让用户等更久。
轮询总超时返回 ``partial`` 而不是 ``failed``，只要已经拿到任何部分结果；
详见 :func:`poll_batch` 的 docstring。
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any

from paperwhisperer.core import config
from paperwhisperer.core.text import (
    close_response_safely,
    compute_retry_after_delay,
    get_retry_delay_seconds,
    now_iso,
)
from paperwhisperer.ingestion.schema import (
    BoundingBox,
    Element,
    ParsedDocument,
    SourceLocator,
    Table,
)
from paperwhisperer.remote.ssrf import is_public_http_url

logger = logging.getLogger(__name__)

PARSER_NAME = "mineru_cloud"
PARSER_VERSION = "1.0.0"

#: MinerU v4 端点。签名与查询分开：签名给上传 URL，查询按 batch 轮询。
FILE_URLS_BATCH_PATH = "/api/v4/file-urls/batch"
EXTRACT_RESULTS_BATCH_PATH = "/api/v4/extract-results/batch"

#: 触发原因写入审计。区分"用户主动要点云端"和"本地解析失败自动降级"，
#: 因为两者的隐私敏感度与费用归属不同。
TRIGGER_REASONS = ("fallback", "forced", "manual")

# --------------------------------------------------------------------------
# 资源上限
# --------------------------------------------------------------------------

#: 结果 zip 的下载体积上限。MinerU 的 markdown + content_list 通常在几 MB，
#: 64MB 已有两个数量级余量。
MAX_RESULT_ZIP_BYTES = 64 * 1024 * 1024

#: zip 防护上限。三个维度都要卡：只看条目数挡不住"少量条目超大"，只看总大小
#: 挡不住"海量小条目"把内存和 inode 打满。
MAX_ZIP_ENTRIES = 512
MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
MAX_ZIP_ENTRY_UNCOMPRESSED_BYTES = 64 * 1024 * 1024

#: 只按名字白名单取 zip 内的两个文件，从不写盘，因此路径穿越（``../``）与
#: 嵌套压缩（内层 zip 再解）都不存在可利用面；这里仍做校验是为了让"以后有人
#: 改成全量落盘"时不会静默变成漏洞。
ALLOWED_ZIP_SUFFIXES = (".md", "_content_list.json", "content_list.json")

#: 重试退避上限。指数退避封顶，防止 429 时把请求摊平成对第三方的持续压力。
RETRY_MAX_DELAY_SECONDS = 30

#: 可重试的 HTTP 状态码。429 由对端明确要求降速；5xx 是瞬时故障。
RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})

#: 额度耗尽的特征词。命中即判定为"重试无用"，因为余额不会因为重试而补充。
QUOTA_ERROR_HINTS = (
    "quota",
    "额度",
    "余额",
    "余额不足",
    "insufficient",
    "exceeded",
    "超出",
    "payment",
    "billing",
    "arrears",
    "欠费",
)

#: 明确表示"任务在云端失败"的状态值。轮询到这些值就不再等。
TERMINAL_FAILURE_STATES = frozenset({"failed", "error", "cancelled", "canceled"})

_SLEEPER_TYPE = Callable[[float], None]


class MineruError(RuntimeError):
    """MinerU 适配器的统一失败类型。

    urllib 抛出的 ``HTTPError`` / ``URLError`` / ``socket.timeout`` 一律在模块
    内部被翻译成这个异常（带中文消息和 ``code``），不让传输层异常裸奔到业务层
    ——调用方需要的是一个能直接展示给用户的原因，而不是 urllib 的英文栈。
    """

    def __init__(self, message: str, code: str = "mineru_error", status: int | None = None):
        super().__init__(message)
        self.code = code
        self.status = status


class MineruConsentError(MineruError):
    """隐私策略或用户同意未通过，因此完全没有发生网络调用。

    单独一个类型而不是复用 :class:`MineruError`，是因为调用方（编排层）需要
    区分"被隐私挡住，换个本地 parser 就能继续"和"云端真的出错了，换 parser
    也会失败"——前者应该静默降级，后者应该提示用户。
    """


# --------------------------------------------------------------------------
# 审计
# --------------------------------------------------------------------------


@dataclass(slots=True)
class MineruAuditRecord:
    """一次云端解析尝试的审计条目。

    字段选择遵循"事后能回答三个问题"：谁（``operator``）、什么时候
    （``timestamp``）、把什么发给了谁（``source_filename`` + ``sha256``），
    外加最关键的两条——``trigger_reason``（为什么走云端）和 ``outcome``
    （结果如何）。**没有 ``token`` 字段**，这是刻意的：审计日志的读取范围
    远大于 secret 的读取范围。
    """

    timestamp: str
    operator: str
    source_filename: str
    sha256: str
    trigger_reason: str
    outcome: str
    parser_version: str = PARSER_VERSION
    parser_name: str = PARSER_NAME
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "operator": self.operator,
            "source_filename": self.source_filename,
            "sha256": self.sha256,
            "trigger_reason": self.trigger_reason,
            "outcome": self.outcome,
            "parser_name": self.parser_name,
            "parser_version": self.parser_version,
            "message": self.message,
        }


def resolve_audit_log_path() -> str:
    """审计日志路径：``PARSER_ARTIFACT_DIR`` 旁边的 ``mineru_audit.jsonl``。

    放在 artifact 目录的**同级**而不是里面，因为 artifact 目录的语义是
    "可删除的解析中间产物"（清理脚本会整目录删掉），而审计日志的语义是
    "必须留存的合规记录"。放一起会导致清理时把合规记录一起清掉。
    """
    artifact_dir = str(getattr(config, "PARSER_ARTIFACT_DIR", "") or "context/parser_artifacts")
    parent = os.path.dirname(artifact_dir.rstrip("/\\")) or "."
    return os.path.join(parent, "mineru_audit.jsonl")


def append_audit_record(
    record: MineruAuditRecord,
    *,
    log_path: str | None = None,
) -> bool:
    """向 JSONL 审计日志追加一条记录，返回是否成功。

    **写入失败不让解析失败。** 审计是"应该做"而不是"必须做"的前置条件：
    如果因为只读文件系统或磁盘满而让用户的解析请求报错，代价远大于丢一条
    审计。容错风格对齐 :func:`~paperwhisperer.core.text.remove_file_safely`
    ——捕获（含 ``SystemExit``）、记 logger、不向上抛。

    这里没有用 :func:`~paperwhisperer.core.text.atomic_write_json`，因为它是
    *替换*语义（``os.replace``），用它写日志等于每次只保留最后一条，审计就
    没了。改为以 ``O_APPEND`` 单次写入整行：POSIX 下小于 ``PIPE_BUF`` 的
    append 对多个写者是原子的，不会出现半行交错；这正是 JSONL 审计需要的
    保证，比"临时文件 + 替换"更贴合 append-only 的语义。
    """
    target = log_path or resolve_audit_log_path()
    line = json.dumps(record.to_dict(), ensure_ascii=False) + "\n"
    try:
        parent = os.path.dirname(target)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(target, "a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        return True
    except (Exception, SystemExit):
        logger.exception("写入 MinerU 审计日志失败: %s", target)
        return False


def read_audit_records(log_path: str) -> list[dict[str, Any]]:
    """读回审计记录。跳过写了一半的行而不是让整份日志不可读。"""
    records: list[dict[str, Any]] = []
    if not os.path.exists(log_path):
        return records
    with open(log_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("跳过一行无法解析的 MinerU 审计记录")
    return records


def _resolve_operator(explicit_operator: str | None = None) -> str:
    """尽力取操作者标识，取不到记 ``unknown``。

    优先级：显式传入（请求上下文里的会话用户）→ ``PAPERWHISPERER_OPERATOR``
    环境变量 → 系统用户名。**绝不用 token 或任何 secret 兜底**——审计里出现
    一个 API key 比出现 ``unknown`` 严重得多。
    """
    for candidate in (
        explicit_operator,
        os.environ.get("PAPERWHISPERER_OPERATOR"),
        os.environ.get("USER"),
        os.environ.get("USERNAME"),
    ):
        text = str(candidate or "").strip()
        if text:
            return text
    return "unknown"


# --------------------------------------------------------------------------
# 配置读取
# --------------------------------------------------------------------------


@dataclass(slots=True)
class _Settings:
    """一次调用内固定的配置快照。

    全部用 ``getattr`` 带默认值读取，因为 ``core/config.py`` 与本模块并行
    开发：字段还没落地时本模块必须仍能 import 并工作（落到安全默认值
    ``local_only`` + 无 token = 永不出网）。
    """

    privacy_mode: str = "local_only"
    consent_granted: bool = False
    #: ``repr=False`` is not decoration. Without it ``repr(_Settings(...))``
    #: prints the token, and this object reaches a traceback or a log line the
    #: moment anything below it raises -- which is exactly when nobody is
    #: reading carefully. A credential that appears in a stack trace is a
    #: credential in whatever collects them.
    token: str = field(default="", repr=False)
    base_url: str = "https://mineru.net"
    timeout_seconds: int = 60
    max_retries: int = 3
    poll_interval_seconds: int = 5
    poll_timeout_seconds: int = 900


def _load_settings() -> _Settings:
    privacy_mode = str(getattr(config, "DOCUMENT_PRIVACY_MODE", "local_only") or "local_only")
    if privacy_mode not in {"local_only", "allow_cloud"}:
        privacy_mode = "local_only"
    return _Settings(
        privacy_mode=privacy_mode,
        consent_granted=bool(getattr(config, "DOCUMENT_CLOUD_CONSENT_GRANTED", False)),
        token=str(getattr(config, "MINERU_API_TOKEN", "") or "").strip(),
        base_url=str(
            getattr(config, "MINERU_API_BASE_URL", "") or "https://mineru.net"
        ).rstrip("/"),
        timeout_seconds=max(1, int(getattr(config, "MINERU_TIMEOUT_SECONDS", 60) or 60)),
        max_retries=max(1, int(getattr(config, "MINERU_MAX_RETRIES", 3) or 3)),
        poll_interval_seconds=max(
            0, int(getattr(config, "MINERU_POLL_INTERVAL_SECONDS", 5) or 5)
        ),
        poll_timeout_seconds=max(
            1, int(getattr(config, "MINERU_POLL_TIMEOUT_SECONDS", 900) or 900)
        ),
    )


@dataclass(slots=True)
class PrivacyGate:
    """隐私门控结论。"""

    allowed: bool
    reason_code: str
    message: str


def evaluate_privacy_gate(settings: _Settings | None = None) -> PrivacyGate:
    """判断本次是否允许出网。**必须在任何网络调用之前调用。**

    两道检查分开返回，因为用户要能分辨该去哪儿改：策略不允许要去管理员那里
    改 ``DOCUMENT_PRIVACY_MODE``；策略允许但用户没同意要在界面上开同意开关。
    合成一句"云解析未启用"会让用户两边都改一遍还是不知道哪儿错了。
    """
    resolved = settings or _load_settings()
    if resolved.privacy_mode != "allow_cloud":
        return PrivacyGate(
            allowed=False,
            reason_code="privacy_mode_local_only",
            message=(
                "云端解析未启用：当前文档隐私模式为 local_only，禁止任何文档出网。"
                "如需启用请联系管理员将 DOCUMENT_PRIVACY_MODE 设置为 allow_cloud。"
            ),
        )
    if not resolved.consent_granted:
        return PrivacyGate(
            allowed=False,
            reason_code="consent_not_granted",
            message=(
                "云端解析未获授权：隐私模式允许云端，但尚未获得您的显式同意。"
                "请在设置中开启“允许将文档上传到云端解析服务”。"
            ),
        )
    return PrivacyGate(allowed=True, reason_code="allowed", message="")


# --------------------------------------------------------------------------
# HTTP 层
# --------------------------------------------------------------------------


def _build_ssl_context() -> ssl.SSLContext:
    """优先用 certifi 的 CA bundle，与 :mod:`paperwhisperer.search.http` 保持一致。"""
    try:
        import certifi
    except ImportError:  # pragma: no cover - certifi ships with requirements
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def _default_sleeper(seconds: float) -> None:
    """默认 sleeper 走 ``time.sleep`` 的属性查找，便于测试 monkeypatch。"""
    time.sleep(seconds)


def _describe_http_error(exc: urllib.error.HTTPError) -> str:
    """把 ``HTTPError`` 变成一句能直接展示的中文原因。"""
    code = exc.code
    try:
        body = exc.read().decode("utf-8", errors="ignore")
    except Exception:  # pragma: no cover - 读 body 失败不应掩盖状态码
        body = ""
    if code in _quota_status_codes():
        return f"MinerU 额度不足（HTTP {code}），请检查账户余额或套餐额度。"
    if code == 401:
        return "MinerU API Token 无效或已过期（HTTP 401），请检查 MINERU_API_TOKEN 配置。"
    if code == 403:
        return "云端解析被拒绝（HTTP 403），请检查 Token 权限与文件访问策略。"
    if code == 404:
        return "云端解析接口不存在（HTTP 404），请检查 MINERU_API_BASE_URL 配置。"
    if code == 429:
        return "云端解析请求过于频繁（HTTP 429），已重试到上限，请稍后再试。"
    if 500 <= code < 600:
        return f"MinerU 服务暂时不可用（HTTP {code}），已重试到上限，请稍后再试。"
    detail = body.strip()[:200]
    suffix = f"：{detail}" if detail else ""
    return f"云端解析失败（HTTP {code}）{suffix}"


def _quota_status_codes() -> frozenset[int]:
    """表示"钱/额度问题"的状态码。402 Payment Required 是明确的信号。"""
    return frozenset({402})


def _http_error_to_mineru_error(exc: urllib.error.HTTPError) -> MineruError:
    """把 ``HTTPError`` 翻成 :class:`MineruError`，**保留额度语义**。

    额度耗尽必须在 HTTP 层就带上 ``mineru_quota_exhausted``：编排层据此决定
    提示用户充值而不是静默换 parser，而它只认 ``code``。只把中文消息写对、
    code 却仍是通用的 ``mineru_http_error``，会让一次 402 在审计里和一次 500
    长得一模一样。
    """
    message = _describe_http_error(exc)
    if exc.code in _quota_status_codes():
        return MineruError(message, code="mineru_quota_exhausted", status=exc.code)
    return MineruError(message, code="mineru_http_error", status=exc.code)


def _looks_like_quota_error(status: int, body: str) -> bool:
    """状态码或响应体命中额度特征即判定额度耗尽（此类错误不重试）。"""
    if status in _quota_status_codes():
        return True
    lowered = str(body or "").lower()
    return any(hint in lowered for hint in QUOTA_ERROR_HINTS)


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse to follow a redirect, by returning ``None`` as urllib expects.

    A presigned OSS URL authenticates the *exact* request that was signed.
    Following a 30x replays the PUT against a different URL — typically one
    that drops the query string — and the signature no longer matches, so the
    upload fails with a 403 that looks like a credentials problem rather than
    a routing one. urllib's default opener follows redirects, so this has to
    be opted into explicitly for the upload leg.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None


def _no_redirect_opener(ssl_context: ssl.SSLContext) -> urllib.request.OpenerDirector:
    """An opener that will not follow redirects but keeps the TLS settings."""
    return urllib.request.build_opener(
        _NoRedirectHandler(),
        urllib.request.HTTPSHandler(context=ssl_context),
    )


def _open_with_retry(
    request: urllib.request.Request,
    *,
    timeout: int,
    max_retries: int,
    sleeper: _SLEEPER_TYPE | None = None,
    ssl_context: ssl.SSLContext | None = None,
    opener: Any = None,
) -> Any:
    """发起一次 HTTP 请求，429 / 瞬时网络错误按指数退避重试。

    与 :func:`paperwhisperer.search.http._open_with_retry` 同一套退避逻辑：
    有 ``Retry-After`` 就听对端的（对端最清楚自己多久恢复），否则用本地指数
    退避。额度类 4xx 不在这里重试——它不是瞬时故障，重试只会让用户多等一轮
    并再消耗一次配额检查。

    ``opener`` 覆盖默认的 ``urlopen``，用于上传那种**不能跟随重定向**的场景。
    """
    sleep = sleeper or _default_sleeper
    attempts = max(1, max_retries)
    last_error: Exception | None = None
    if opener is not None:
        # A custom opener carries its own TLS context (it is built with one),
        # and ``OpenerDirector.open`` does not accept a ``context`` argument.
        send: Any = lambda req, timeout: opener.open(req, timeout=timeout)  # noqa: E731
    else:
        send = lambda req, timeout: urllib.request.urlopen(  # noqa: E731
            req, timeout=timeout, context=ssl_context or _build_ssl_context()
        )
    for attempt in range(attempts):
        try:
            return send(request, timeout)
        except urllib.error.HTTPError as exc:
            last_error = exc
            retryable = exc.code in RETRYABLE_STATUS_CODES
            if retryable and attempt < attempts - 1:
                sleep(compute_retry_after_delay(exc, attempt, RETRY_MAX_DELAY_SECONDS))
                continue
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt < attempts - 1:
                sleep(get_retry_delay_seconds(attempt, max_delay=RETRY_MAX_DELAY_SECONDS))
                continue
            raise
    raise last_error if last_error else MineruError("云端解析请求失败。", code="mineru_no_attempt")


def _read_capped(response: Any, max_bytes: int, description: str) -> bytes:
    """读取响应体，超过 ``max_bytes`` 立即中止。

    读 ``max_bytes + 1`` 字节而不是信任 ``Content-Length``：对端可以撒谎，
    实际流量才是唯一可信的量。
    """
    payload = response.read(max_bytes + 1)
    if payload is None:
        return b""
    if len(payload) > max_bytes:
        raise MineruError(
            f"{description}超过上限（{max_bytes // (1024 * 1024)}MB），已中止下载。",
            code="mineru_response_too_large",
        )
    return payload


def _read_json(response: Any, description: str) -> dict[str, Any]:
    """把响应体解成 JSON。所有协议错误都在这里收口成中文消息。"""
    raw = _read_capped(response, MAX_RESULT_ZIP_BYTES, description)
    if not raw:
        raise MineruError(f"{description}返回了空响应。", code="mineru_empty_response")
    try:
        payload = json.loads(raw.decode("utf-8", errors="ignore"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise MineruError(
            f"{description}返回的不是合法 JSON。", code="mineru_bad_json"
        ) from exc
    if not isinstance(payload, dict):
        raise MineruError(f"{description}返回的 JSON 结构异常。", code="mineru_bad_json_shape")
    return payload


def _check_api_code(payload: dict[str, Any], description: str) -> dict[str, Any]:
    """校验 MinerU 统一响应壳 ``{"code": 0, "data": {...}}``。

    业务错误（额度不足、文件不存在等）都在 ``code != 0`` 里，HTTP 仍是 200，
    所以必须显式检查——只看 HTTP 状态码会把所有业务失败当成成功。
    """
    code = payload.get("code")
    if code is None:
        return payload
    try:
        numeric_code = int(code)
    except (TypeError, ValueError) as exc:
        raise MineruError(
            f"{description}返回了无法识别的业务状态码。", code="mineru_bad_code"
        ) from exc
    if numeric_code == 0:
        return payload
    message = str(payload.get("msg") or payload.get("message") or "").strip()
    if _looks_like_quota_error(numeric_code, message):
        raise MineruError(
            "MinerU 额度不足，无法完成云端解析。请检查账户余额或套餐额度后重试。",
            code="mineru_quota_exhausted",
            status=numeric_code,
        )
    raise MineruError(
        f"{description}失败（业务码 {numeric_code}）{f'：{message}' if message else ''}",
        code="mineru_business_error",
        status=numeric_code,
    )


# --------------------------------------------------------------------------
# 全链路各步骤
# --------------------------------------------------------------------------


def request_upload_urls(
    *,
    filename: str,
    settings: _Settings,
    sleeper: _SLEEPER_TYPE | None = None,
) -> tuple[str, str]:
    """第一步：向 ``/api/v4/file-urls/batch`` 申请上传凭证。

    返回 ``(batch_id, upload_url)``。这里只拿 URL，不传文件内容——真正的字节
    在下一步 PUT 上去，所以"签名"这一步即使被日志记录也不会泄露文档正文。
    """
    url = f"{settings.base_url}{FILE_URLS_BATCH_PATH}"
    # Only `name` is documented on `files[]` (plus optional `data_id` /
    # `is_ocr` / `page_ranges`). An undocumented key is sent as-is and can be
    # reflected into the signature, so it is not sent.
    body = json.dumps({"files": [{"name": filename}]}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.token}",
            "User-Agent": str(getattr(config, "APP_USER_AGENT", "PaperWhisperer")),
        },
        method="POST",
    )
    response = None
    try:
        response = _open_with_retry(
            request,
            timeout=settings.timeout_seconds,
            max_retries=settings.max_retries,
            sleeper=sleeper,
        )
        payload = _check_api_code(_read_json(response, "申请上传凭证"), "申请上传凭证")
    except urllib.error.HTTPError as exc:
        raise _http_error_to_mineru_error(exc) from exc
    except urllib.error.URLError as exc:
        raise MineruError(
            f"连接 MinerU 失败：{getattr(exc, 'reason', exc)}。请检查网络或代理设置。",
            code="mineru_network_error",
        ) from exc
    except (TimeoutError, OSError) as exc:
        raise MineruError(
            f"访问 MinerU 超时或网络异常：{exc}", code="mineru_network_error"
        ) from exc
    finally:
        close_response_safely(response, "MinerU 上传凭证响应")

    data = payload.get("data")
    data = data if isinstance(data, dict) else {}
    batch_id = str(data.get("batch_id") or "").strip()
    file_urls = data.get("file_urls")
    upload_url = str(file_urls[0]).strip() if isinstance(file_urls, list) and file_urls else ""
    if not batch_id or not upload_url:
        raise MineruError(
            "MinerU 返回的批次信息不完整（缺少 batch_id 或上传地址）。",
            code="mineru_incomplete_batch",
        )
    return batch_id, upload_url


def upload_file_to_signed_url(
    file_path: str,
    upload_url: str,
    *,
    settings: _Settings,
    sleeper: _SLEEPER_TYPE | None = None,
) -> None:
    """第二步：把本地文件 PUT 到签名 URL。

    **SSRF 校验在这里，不在请求构造里。** ``upload_url`` 完全来自第三方响应，
    一个被篡改的响应可以让本机对 ``127.0.0.1`` 或云元数据地址发起 PUT。复用
    :func:`is_public_http_url` 而不是自己写一遍判断，是为了和远程导入链路
    共用同一套 DNS 解析 + 全 IP 公网校验（含 rebinding 防护）。
    """
    if not is_public_http_url(upload_url):
        raise MineruError(
            "MinerU 返回的上传地址未通过 SSRF 校验（指向非公网主机），已拒绝上传。",
            code="mineru_ssrf_blocked",
        )
    try:
        with open(file_path, "rb") as handle:
            payload = handle.read()
    except OSError as exc:
        raise MineruError(f"读取待上传文件失败：{exc}", code="mineru_local_read_error") from exc

    # **No Content-Type, and that is load-bearing, not an oversight.**
    # Aliyun OSS Signature V1 bakes Content-Type into the string-to-sign, and
    # the URL was signed by MinerU with no Content-Type at all — so sending one
    # produces a signature mismatch and the upload is rejected with 403
    # ``SignatureDoesNotMatch``. The official example is a bare
    # ``curl -X PUT -T <file> '<url>'``.
    #
    # The empty-string value is how that is expressed with urllib: a Request
    # carrying `data` and no explicit Content-type has one *injected* —
    # ``application/x-www-form-urlencoded`` — by ``AbstractHTTPHandler``,
    # silently. Setting the header to "" satisfies ``has_header()`` so the
    # injection is skipped, and an empty value canonicalises the same as an
    # absent one. Do not "tidy" this into a real media type.
    #
    # Also absent, deliberately: ``Authorization``. The URL carries its own
    # auth, and attaching the Bearer here would hand the MinerU token to a
    # third-party storage host.
    request = urllib.request.Request(
        upload_url,
        data=payload,
        headers={"Content-Type": ""},
        method="PUT",
    )
    response = None
    try:
        response = _open_with_retry(
            request,
            timeout=settings.timeout_seconds,
            max_retries=settings.max_retries,
            sleeper=sleeper,
            opener=_no_redirect_opener(_build_ssl_context()),
        )
        close_response_safely(response, "MinerU 上传响应")
        response = None
    except urllib.error.HTTPError as exc:
        raise _http_error_to_mineru_error(exc) from exc
    except urllib.error.URLError as exc:
        raise MineruError(
            f"上传文件到 MinerU 失败：{getattr(exc, 'reason', exc)}", code="mineru_network_error"
        ) from exc
    except (TimeoutError, OSError) as exc:
        raise MineruError(
            f"上传文件到 MinerU 超时或网络异常：{exc}", code="mineru_network_error"
        ) from exc


def poll_batch(
    batch_id: str,
    *,
    settings: _Settings,
    sleeper: _SLEEPER_TYPE | None = None,
    monotonic: Callable[[], float] | None = None,
) -> str | None:
    """第三步：轮询任务状态直到 ``done``，返回结果 zip 的下载地址。

    **超时返回 ``None`` 而不是抛异常**，这是"降级"与"失败"的分界：调用方
    拿到 ``None`` 后会再查一次结果接口，如果云端其实已经产出（只是慢）就能
    捡回一份完整结果并把这次解析标为 ``partial``；只有连部分结果都拿不到
    才对用户报失败。让轮询函数自己决定"算不算失败"会丢掉这个信息。

    间隔与总超时都从配置取，``sleeper`` 可注入，因此测试不会真的睡。
    """
    sleep = sleeper or _default_sleeper
    clock = monotonic or time.monotonic
    url = f"{settings.base_url}{EXTRACT_RESULTS_BATCH_PATH}/{urllib.parse.quote(batch_id)}"
    deadline = clock() + settings.poll_timeout_seconds

    while True:
        request = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {settings.token}",
                "User-Agent": str(getattr(config, "APP_USER_AGENT", "PaperWhisperer")),
            },
            method="GET",
        )
        response = None
        try:
            response = _open_with_retry(
                request,
                timeout=settings.timeout_seconds,
                max_retries=settings.max_retries,
                sleeper=sleeper,
            )
            payload = _check_api_code(_read_json(response, "查询解析进度"), "查询解析进度")
        except urllib.error.HTTPError as exc:
            raise _http_error_to_mineru_error(exc) from exc
        except urllib.error.URLError as exc:
            raise MineruError(
                f"查询解析进度失败：{getattr(exc, 'reason', exc)}", code="mineru_network_error"
            ) from exc
        except (TimeoutError, OSError) as exc:
            raise MineruError(
                f"查询解析进度超时或网络异常：{exc}", code="mineru_network_error"
            ) from exc
        finally:
            close_response_safely(response, "MinerU 进度响应")

        zip_url, state = _extract_state(payload)
        if state == "done":
            if not zip_url:
                raise MineruError(
                    "MinerU 报告解析完成但未提供结果下载地址。", code="mineru_missing_result_url"
                )
            return zip_url
        if state in TERMINAL_FAILURE_STATES:
            raise MineruError(
                f"MinerU 云端解析失败（状态 {state}），可能是文件损坏或内容无法识别。",
                code="mineru_task_failed",
            )
        if clock() >= deadline:
            logger.warning(
                "MinerU 批次 %s 轮询超时（%ss）", batch_id, settings.poll_timeout_seconds
            )
            return None
        sleep(settings.poll_interval_seconds)


def _extract_state(payload: dict[str, Any]) -> tuple[str, str]:
    """从进度响应里取出 ``(结果 zip 地址, 状态)``。"""
    data = payload.get("data")
    data = data if isinstance(data, dict) else {}
    results = data.get("extract_result")
    first = results[0] if isinstance(results, list) and results else {}
    first = first if isinstance(first, dict) else {}
    zip_url = str(
        first.get("full_zip_url") or first.get("zip_url") or data.get("full_zip_url") or ""
    ).strip()
    state = str(first.get("state") or data.get("state") or "pending").strip().lower()
    return zip_url, state


def fetch_partial_result_url(
    batch_id: str,
    *,
    settings: _Settings,
    sleeper: _SLEEPER_TYPE | None = None,
) -> str | None:
    """轮询超时后再查一次结果接口，尝试捡回已完成的结果。

    存在的意义是 :func:`poll_batch` 返回 ``None`` 时的那条降级路径：云端可能
    在我们放弃轮询之后几十秒才真正完成，这时用户应该拿到文档并被告知"结果
    可能不完整"，而不是一句"解析超时"。
    """
    url = f"{settings.base_url}{EXTRACT_RESULTS_BATCH_PATH}/{urllib.parse.quote(batch_id)}"
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {settings.token}"},
        method="GET",
    )
    response = None
    try:
        response = _open_with_retry(
            request,
            timeout=settings.timeout_seconds,
            max_retries=1,
            sleeper=sleeper,
        )
        payload = _read_json(response, "查询解析进度")
    except urllib.error.HTTPError as exc:
        # 额度耗尽不在这里吞掉：吞掉会被当成"还没好"，最终报成"轮询超时"，
        # 把一个需要用户充钱的确定性失败伪装成一个可以重试的暂时问题。
        if exc.code in _quota_status_codes():
            raise _http_error_to_mineru_error(exc) from exc
        return None
    except (urllib.error.URLError, TimeoutError, OSError):
        return None
    finally:
        close_response_safely(response, "MinerU 补查响应")
    zip_url, state = _extract_state(payload)
    return zip_url if state == "done" and zip_url else None


def download_result_zip(
    zip_url: str,
    *,
    settings: _Settings,
    sleeper: _SLEEPER_TYPE | None = None,
) -> bytes:
    """第四步：下载结果 zip（带 SSRF 校验与体积上限）。"""
    if not is_public_http_url(zip_url):
        raise MineruError(
            "MinerU 结果下载地址未通过 SSRF 校验（指向非公网主机），已拒绝下载。",
            code="mineru_ssrf_blocked",
        )
    request = urllib.request.Request(
        zip_url, headers={"User-Agent": "PaperWhisperer"}, method="GET"
    )
    response = None
    try:
        response = _open_with_retry(
            request,
            timeout=settings.timeout_seconds,
            max_retries=settings.max_retries,
            sleeper=sleeper,
        )
        return _read_capped(response, MAX_RESULT_ZIP_BYTES, "云端解析结果")
    except urllib.error.HTTPError as exc:
        raise _http_error_to_mineru_error(exc) from exc
    except urllib.error.URLError as exc:
        raise MineruError(
            f"下载云端解析结果失败：{getattr(exc, 'reason', exc)}", code="mineru_network_error"
        ) from exc
    except (TimeoutError, OSError) as exc:
        raise MineruError(
            f"下载云端解析结果超时或网络异常：{exc}", code="mineru_network_error"
        ) from exc
    finally:
        close_response_safely(response, "MinerU 结果响应")


# --------------------------------------------------------------------------
# zip 处理
# --------------------------------------------------------------------------


def validate_zip_bomb(archive: zipfile.ZipFile) -> None:
    """在解压**之前**按声明值拒绝恶意 zip。

    三个维度分别对应三种攻击形态：

    - 条目数上限 → 海量小条目（打满内存/inode，或拖死 ``namelist``）。
    - 单条目上限 → 少量条目解压后极大（经典 42.zip 式）。
    - 总解压上限 → 单看每个都不大但累加超限（叠加密）。

    检查的是 zip **中央目录里声明的** ``file_size``，不是实际解压出来的字节
    数。这正是 zip bomb 能骗过只看实际流量的检查的地方：一个声明 4GB、
    实际解压出 4GB 的条目在声明阶段就该被拒，而不是先花掉 4GB 内存再去发现。
    声明值可被伪造（zip 格式不做强制），所以读取时还有
    :func:`_read_zip_member` 的实际字节数兜底。
    """
    entries = archive.infolist()
    if len(entries) > MAX_ZIP_ENTRIES:
        raise MineruError(
            f"云端结果压缩包条目过多（{len(entries)} > {MAX_ZIP_ENTRIES}），疑似恶意压缩包。",
            code="mineru_zip_bomb",
        )
    total = 0
    for entry in entries:
        if entry.is_dir():
            continue
        if entry.file_size > MAX_ZIP_ENTRY_UNCOMPRESSED_BYTES:
            raise MineruError(
                f"云端结果压缩包中 {entry.filename} 声明解压后超过 "
                f"{MAX_ZIP_ENTRY_UNCOMPRESSED_BYTES // (1024 * 1024)}MB，疑似 zip 炸弹。",
                code="mineru_zip_bomb",
            )
        total += entry.file_size
        if total > MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES:
            raise MineruError(
                "云端结果压缩包声明的总解压大小超过 "
                f"{MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES // (1024 * 1024)}MB，疑似 zip 炸弹。",
                code="mineru_zip_bomb",
            )


def _read_zip_member(archive: zipfile.ZipFile, entry: zipfile.ZipInfo) -> bytes:
    """按上限读取一个条目，作为 zip 声明值不可信时的实际字节兜底。"""
    with archive.open(entry) as handle:
        payload = handle.read(MAX_ZIP_ENTRY_UNCOMPRESSED_BYTES + 1)
    if len(payload) > MAX_ZIP_ENTRY_UNCOMPRESSED_BYTES:
        raise MineruError(
            f"云端结果压缩包条目 {entry.filename} 实际解压大小超过上限，疑似 zip 炸弹。",
            code="mineru_zip_bomb",
        )
    return payload


def _is_safe_member_name(name: str) -> bool:
    """拒绝绝对路径与 ``..`` 穿越。

    当前实现只把内容读进内存、从不落盘，所以穿越没有可利用面；保留这道校验
    是为了让"以后有人改成全量解压到临时目录"时不会静默引入漏洞。
    """
    normalized = name.replace("\\", "/")
    if normalized.startswith("/") or ":" in normalized.split("/")[0][1:2]:
        return False
    return ".." not in normalized.split("/")


def extract_result_payload(zip_bytes: bytes) -> dict[str, Any]:
    """从结果 zip 里取出 ``content_list`` 与 markdown。

    按名字白名单取文件，不落盘、不递归解嵌套 zip——于是路径穿越与嵌套压缩
    两种攻击在结构上就没有入口，上面的校验退化为第二道防线。
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile as exc:
        raise MineruError("云端返回的结果不是有效的压缩包。", code="mineru_bad_zip") from exc

    with archive:
        validate_zip_bomb(archive)
        content_list: list[dict[str, Any]] = []
        markdown_parts: list[str] = []
        for entry in archive.infolist():
            if entry.is_dir() or not _is_safe_member_name(entry.filename):
                continue
            lowered = entry.filename.lower()
            if lowered.endswith("content_list.json"):
                raw = _read_zip_member(archive, entry)
                try:
                    parsed = json.loads(raw.decode("utf-8", errors="ignore"))
                except json.JSONDecodeError as exc:
                    raise MineruError(
                        "云端结果中的 content_list.json 解析失败。", code="mineru_bad_content_list"
                    ) from exc
                if isinstance(parsed, list):
                    content_list.extend(item for item in parsed if isinstance(item, dict))
            elif lowered.endswith(".md") and lowered not in ALLOWED_ZIP_SUFFIXES:
                markdown_parts.append(
                    _read_zip_member(archive, entry).decode("utf-8", errors="ignore")
                )

    if not content_list and not markdown_parts:
        raise MineruError(
            "云端结果压缩包中没有可识别的解析内容（content_list.json / markdown）。",
            code="mineru_empty_result",
        )
    return {"content_list": content_list, "markdown": "\n\n".join(markdown_parts).strip()}


# --------------------------------------------------------------------------
# normalize（纯函数）
# --------------------------------------------------------------------------


class _TableHtmlParser(HTMLParser):
    """把 MinerU 的 ``table_body`` HTML 还原成二维网格。

    表格**不能**拍平成一段文本：拍平之后跨列检索就不可能了，而表格里往往
    正是"某指标在某数据集上的数值"这类最该被引用的内容。所以这里逐 ``<tr>``
    逐 ``<td>`` 收集，交给 :class:`~paperwhisperer.ingestion.schema.Table`。
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._current_row: list[str] | None = None
        self._cell_parts: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "tr":
            self._current_row = []
        elif tag in {"td", "th"}:
            self._cell_parts = []
        elif tag == "br" and self._cell_parts is not None:
            self._cell_parts.append(" ")

    def handle_data(self, data: str) -> None:
        if self._cell_parts is not None:
            self._cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._cell_parts is not None:
            cell = re.sub(r"\s+", " ", "".join(self._cell_parts)).strip()
            if self._current_row is None:
                self._current_row = [cell]
            else:
                self._current_row.append(cell)
            self._cell_parts = None
        elif tag == "tr" and self._current_row is not None:
            if any(cell for cell in self._current_row):
                self.rows.append(self._current_row)
            self._current_row = None

    def close(self) -> None:  # pragma: no cover - 委托给父类
        super().close()
        if self._current_row and any(cell for cell in self._current_row):
            self.rows.append(self._current_row)
            self._current_row = None


def _parse_table_html(html_text: str) -> Table:
    """HTML 表格 → :class:`Table`；无法解析出任何行时返回空壳。"""
    parser = _TableHtmlParser()
    try:
        parser.feed(str(html_text or ""))
        parser.close()
    except Exception:  # pragma: no cover - 半截 HTML 不应让整个解析失败
        logger.warning("MinerU 表格 HTML 解析异常，按空表格处理", exc_info=True)
    rows = parser.rows
    if not rows:
        return Table()
    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]
    # MinerU 的表头是 <thead> 里的 <th>；这里无法区分 th/td，退而用首行作表头
    # ——错误方向是把数据行当表头（多一行表头比少一行安全）。
    return Table(rows=padded, header_rows=1)


def _extract_bbox(raw_bbox: Any) -> BoundingBox | None:
    """MinerU 的 bbox 是 ``[x0, y0, x1, y1]``；形状不对就返回 None。"""
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) < 4:
        return None
    try:
        x0, y0, x1, y1 = (float(value) for value in raw_bbox[:4])
    except (TypeError, ValueError):
        return None
    return BoundingBox(x0=min(x0, x1), y0=min(y0, y1), x1=max(x0, x1), y1=max(y0, y1))


def _join_caption(raw_caption: Any) -> str:
    """MinerU 的 caption 是字符串或字符串列表，统一成一行文本。"""
    if isinstance(raw_caption, (list, tuple)):
        return " ".join(str(item).strip() for item in raw_caption if str(item).strip())
    return str(raw_caption or "").strip()


def _role_for_content_type(content_type: str) -> str:
    """MinerU ``type`` → 本项目 ``Element.role``。

    映射到 schema 已有的 role 而不是自造一套：chunking 与渲染都按 role 分支，
    未知 role 会被 :class:`Element` 降级成 ``text``，等于丢掉"这是公式"的信息。
    """
    return {
        "text": "paragraph",
        "title": "title",
        "equation": "formula",
        "equation_inline": "formula",
        "image": "figure",
        "table": "table",
        "list": "list_item",
        "page_number": "footer",
        "header": "header",
        "footer": "footer",
        "page_footnote": "footnote",
        "footnote": "footnote",
        "code": "code",
    }.get(content_type, "text")


def _markdown_to_elements(markdown: str) -> list[Element]:
    """没有 content_list 时的降级：按 markdown 标题与段落切元素。

    只有 ``txt_offset`` 之外的定位能力为零——但这仍好过"整篇塞进一个
    paragraph"，至少标题层级（决定 chunk 边界）保住了。
    """
    elements: list[Element] = []
    offset = 0
    heading_path: tuple[str, ...] = ()
    for block in str(markdown or "").split("\n\n"):
        text = block.strip()
        if not text:
            offset += len(block) + 2
            continue
        stripped = text.lstrip()
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            heading_text = stripped.lstrip("#").strip()
            role = "title" if level <= 1 else "heading"
            heading_path = (*heading_path, heading_text) if role == "heading" else (heading_text,)
            elements.append(
                Element(
                    role=role,
                    text=heading_text,
                    level=max(1, level),
                    heading_path=heading_path,
                    locator=SourceLocator(
                        kind="txt_offset", offset_start=offset, offset_end=offset + len(text)
                    ),
                )
            )
        else:
            elements.append(
                Element(
                    role="paragraph",
                    text=text,
                    heading_path=heading_path,
                    locator=SourceLocator(
                        kind="txt_offset", offset_start=offset, offset_end=offset + len(text)
                    ),
                )
            )
        offset += len(block) + 2
    return elements


def normalize_mineru_payload(
    payload: dict[str, Any],
    *,
    source_filename: str = "",
    media_type: str = "",
    sha256: str = "",
    page_count: int = 0,
    parser_version: str = PARSER_VERSION,
) -> ParsedDocument:
    """把 MinerU 结果转成 :class:`ParsedDocument`——**纯函数，不碰网络**。

    写成纯函数是刻意的：normalize 是最容易出错也最难在真实链路上调试的一段
    （要花额度、几分钟、依赖云端在线），把它与网络完全解耦后就能用手工构造的
    payload 直接断言元素角色、locator 与表格网格。

    ``payload`` 接受 ``{"content_list": [...], "markdown": "..."}``，
    也接受直接的 list（视为 content_list）。
    """
    if isinstance(payload, list):
        content_list: list[dict[str, Any]] = [item for item in payload if isinstance(item, dict)]
        markdown = ""
    else:
        content_list = [
            item for item in (payload.get("content_list") or []) if isinstance(item, dict)
        ]
        markdown = str(payload.get("markdown") or "")

    document = ParsedDocument(
        source_filename=source_filename,
        media_type=media_type,
        sha256=sha256,
        page_count=page_count,
        parser_name=PARSER_NAME,
        parser_version=parser_version,
        created_at=now_iso(),
    )

    elements: list[Element] = []
    heading_path: tuple[str, ...] = ()
    max_page = 0
    for order, item in enumerate(content_list):
        content_type = str(item.get("type") or "text").strip().lower()
        role = _role_for_content_type(content_type)
        text_level = item.get("text_level")
        if role == "paragraph" and isinstance(text_level, int) and text_level > 0:
            # MinerU 用 text_level 表达标题：1 是文档标题，2+ 是各级小标题。
            role = "title" if text_level == 1 else "heading"

        table: Table | None = None
        latex = ""
        # 每一轮都重置：``text`` 曾经只在部分分支里赋值，于是表格与公式分支会
        # 沿用**上一个元素**的 text——表格里重复上一段正文、公式丢失自己的
        # 渲染文本，而 content_list 的第一项是表格时直接 UnboundLocalError。
        # 表格与公式都留空 text，由 :class:`Element` 从 table/latex 投影，
        # 同一段内容因此不会有两个来源互相覆盖。
        text = ""
        if role == "table":
            caption = _join_caption(item.get("table_caption"))
            table = _parse_table_html(str(item.get("table_body") or ""))
            table.caption = caption
        elif role == "formula":
            latex = str(item.get("text") or "").strip().strip("$")
        elif role == "figure":
            text = _join_caption(item.get("img_caption")) or str(item.get("text") or "").strip()
        else:
            text = str(item.get("text") or "").strip()

        # 标题替换整条路径（level 1 之后就是新的文档标题），小标题追加一级。
        if role in {"title", "heading"} and text:
            heading_path = (*heading_path, text) if role == "heading" else (text,)

        page_index = item.get("page_idx")
        page = int(page_index) + 1 if isinstance(page_index, int) and page_index >= 0 else None
        if page:
            max_page = max(max_page, page)

        element = Element(
            role=role,
            text=text,
            heading_path=heading_path,
            level=(
                text_level
                if role in {"title", "heading"} and isinstance(text_level, int)
                else None
            ),
            table=table,
            latex=latex,
            # "unknown" is the honest spelling when the payload carries no page
            # index. Declaring pdf_page with page=None produced a locator that
            # looked resolvable and resolved to nothing; the gateway's contract
            # layer refuses that shape, so fixing it here rather than letting
            # the repair pass paper over a producer bug. The bbox stays as-is:
            # it is permitted on either kind, and dropping it would lose a
            # coordinate we actually have.
            locator=SourceLocator(
                kind="pdf_page" if page else "unknown",
                page=page,
                bbox=_extract_bbox(item.get("bbox")),
            ),
            order=order,
        )
        if not element.text and table is None and not latex:
            # 空壳元素（空图注、无单元格表格）会让 char_count 与 locator
            # 覆盖率统计失真，直接丢弃而不是留一个空引用。
            continue
        elements.append(element)

    if not elements and markdown:
        elements = _markdown_to_elements(markdown)

    document.elements = elements
    if not document.page_count:
        document.page_count = max_page
    if page_count:
        document.page_count = page_count
    if not elements:
        document.warnings.append("云端解析未返回任何可用内容元素。")
    if max_page and page_count and max_page > page_count:
        document.warnings.append(
            f"部分元素页码（最大 {max_page}）超出报告的总页数 {page_count}，请复核解析结果。"
        )
    document.build_units()
    return document


# --------------------------------------------------------------------------
# 顶层入口
# --------------------------------------------------------------------------


def parse_document(
    file_path: str,
    *,
    source_filename: str = "",
    media_type: str = "application/pdf",
    trigger_reason: str = "manual",
    operator: str | None = None,
    sleeper: _SLEEPER_TYPE | None = None,
    audit_log_path: str | None = None,
) -> tuple[ParsedDocument | None, MineruAuditRecord]:
    """完整跑一遍云端解析链路，返回 ``(文档 或 None, 审计记录)``。

    **不抛异常**：隐私拦截、缺 token、额度不足、网络失败全部转成
    ``(None, 审计记录)``，消息在审计记录的 ``message`` 里。理由是调用方
    （编排层）几乎总是在 try/except 里做 fallback，异常会让"云端失败"和
    "代码 bug"走同一条路；返回结构化结果让编排层能按 ``outcome`` 决定
    静默降级还是提示用户。

    审计在这里写、且只在这里写一次，保证"每次调用一条记录"不会因为链路
    中途抛错而漏写。
    """
    settings = _load_settings()
    filename = source_filename or os.path.basename(str(file_path))
    reason = trigger_reason if trigger_reason in TRIGGER_REASONS else "manual"
    who = _resolve_operator(operator)
    digest = _file_sha256(file_path)

    def finish(document: ParsedDocument | None, outcome: str, message: str):
        record = MineruAuditRecord(
            timestamp=now_iso(),
            operator=who,
            source_filename=filename,
            sha256=digest,
            trigger_reason=reason,
            outcome=outcome,
            message=message,
        )
        append_audit_record(record, log_path=audit_log_path)
        return document, record

    # --- 门控：以下所有网络动作都在这行之后 ---------------------------------
    gate = evaluate_privacy_gate(settings)
    if not gate.allowed:
        logger.info("MinerU 调用被拦截（%s）：%s", gate.reason_code, filename)
        return finish(None, "blocked", gate.message)

    if not settings.token:
        message = (
            "未配置 MINERU_API_TOKEN，无法使用 MinerU 云端解析。"
            "请在环境变量中设置 MINERU_API_TOKEN 后重试。"
        )
        return finish(None, "failed", message)

    try:
        batch_id, upload_url = request_upload_urls(
            filename=filename, settings=settings, sleeper=sleeper
        )
        upload_file_to_signed_url(file_path, upload_url, settings=settings, sleeper=sleeper)
        zip_url = poll_batch(batch_id, settings=settings, sleeper=sleeper)
        partial = zip_url is None
        if zip_url is None:
            # 降级路径：轮询超时后再查一次，能捡回就当 partial 成功。
            zip_url = fetch_partial_result_url(batch_id, settings=settings, sleeper=sleeper)
        if zip_url is None:
            message = (
                f"MinerU 解析在 {settings.poll_timeout_seconds} 秒内未完成，已放弃等待。"
                "可稍后在解析记录中查看结果，或改用本地解析器。"
            )
            return finish(None, "timed_out", message)
        zip_bytes = download_result_zip(zip_url, settings=settings, sleeper=sleeper)
        payload = extract_result_payload(zip_bytes)
    except MineruConsentError:
        raise
    except MineruError as exc:
        return finish(None, "quota_exhausted" if exc.code == "mineru_quota_exhausted" else "failed",
                      str(exc))
    except (Exception, SystemExit) as exc:  # 兜底：任何意外都不让 urllib/IO 细节裸奔
        logger.exception("MinerU 云端解析出现未预期错误：%s", filename)
        return finish(None, "failed", f"云端解析出现未预期错误：{exc}")

    document = normalize_mineru_payload(
        payload,
        source_filename=filename,
        media_type=media_type,
        sha256=digest,
    )
    outcome = "partial" if partial else "success"
    message = (
        "云端解析超时，以下内容可能不完整。"
        if partial
        else ""
    )
    return finish(document, outcome, message)


def _file_sha256(file_path: str) -> str:
    """文件 sha256。读不到就记空串——审计不能因为源文件读不到就整个失败。"""
    hasher = hashlib.sha256()
    try:
        with open(file_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
    except OSError:
        logger.warning("无法计算文件摘要用于审计：%s", file_path)
        return ""
    return hasher.hexdigest()


class MineruCloudParser:
    """MinerU 云端 parser。

    通过 duck typing 满足 :class:`~paperwhisperer.ingestion.gateway.DocumentParser`
    协议（``name`` / ``version`` / ``supports`` / ``parse``），**不 import
    gateway**：两者并行开发，import 会让本模块在 gateway 落地前无法加载。
    协议只要求属性与方法签名匹配，duck typing 足够；若 gateway 后续要求
    返回 ``ParseResult``，只需在 :meth:`parse` 里包一层。
    """

    name = PARSER_NAME
    version = PARSER_VERSION

    #: MinerU 的强项是版面与公式识别，因此只接 PDF 与图片。DOCX/PPTX 走本地
    #: 链路——它们是结构化格式，本地解析同样准确，而把它们发出去是隐私
    #: 代价最高的一种情形。
    SUPPORTED_MEDIA_TYPES = ("application/pdf", "image/png", "image/jpeg", "image/webp")

    def __init__(
        self,
        *,
        operator: str | None = None,
        sleeper: _SLEEPER_TYPE | None = None,
        audit_log_path: str | None = None,
    ) -> None:
        self.operator = operator
        self.sleeper = sleeper
        self.audit_log_path = audit_log_path

    def supports(self, media_type: str) -> bool:
        normalized = str(media_type or "").split(";")[0].strip().lower()
        return normalized in self.SUPPORTED_MEDIA_TYPES

    def parse(
        self,
        file_path: str,
        options: Any = None,
    ) -> tuple[ParsedDocument | None, MineruAuditRecord]:
        """解析一个文件。``options`` 未被 gateway 落地前按 duck typing 读取。

        从 options 上取 ``source_filename`` / ``media_type`` / ``trigger_reason`` /
        ``operator``，全部用 ``getattr`` 兜底，因此 options 是 ``None``、
        是个普通 dict、还是带这些属性的 dataclass，都能正常工作。
        """
        options_dict = options if isinstance(options, dict) else {}
        return parse_document(
            str(file_path),
            source_filename=str(
                getattr(options, "source_filename", "") or options_dict.get("source_filename") or ""
            ),
            media_type=str(
                getattr(options, "media_type", "") or options_dict.get("media_type") or ""
            )
            or "application/pdf",
            trigger_reason=str(
                getattr(options, "trigger_reason", "")
                or options_dict.get("trigger_reason")
                or "manual"
            ),
            operator=self.operator or getattr(options, "operator", None),
            sleeper=self.sleeper,
            audit_log_path=self.audit_log_path,
        )
