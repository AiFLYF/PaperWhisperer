"""Centralized configuration.

Every environment variable read in this project lives here so that the runtime
contract is documented in exactly one place. Values are resolved at import
time, which keeps the rest of the codebase free of ``os.getenv`` calls.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from pathlib import Path

from env_loader import load_project_env

load_project_env()

#: Settings are read once at import time, so a rejected value is reported here
#: and nowhere else -- whoever starts the server sees it in the same log they
#: would have looked at for a crash.
logger = logging.getLogger(__name__)

APP_NAME = "PaperWhisperer"
APP_VERSION = os.getenv("PAPERWHISPERER_VERSION", "0.9.0").strip() or "0.9.0"
APP_USER_AGENT = f"{APP_NAME}/{APP_VERSION}"

# paperwhisperer/core/config.py -> paperwhisperer/core -> paperwhisperer -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
CONTEXT_FOLDER = "context"

RUNTIME_FOLDERS = (UPLOAD_FOLDER, OUTPUT_FOLDER, CONTEXT_FOLDER)

STATIC_ICON_CACHE_CONTROL = "public, max-age=86400, immutable"
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}

SUPPORTED_EXTENSIONS = (".txt", ".pdf", ".docx", ".pptx")
ALLOWED_EXTENSIONS = frozenset(SUPPORTED_EXTENSIONS)
SUPPORTED_FILE_TYPES_TEXT = ", ".join(SUPPORTED_EXTENSIONS)

READING_QUEUE_LIMIT = 30


# --------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------

def parse_int_env(name: str, default: int, *, min_value: int = 1, max_value: int) -> int:
    """Read an integer setting, rejecting anything outside ``[min, max]``.

    Out-of-range input is *clamped* rather than rejected outright: a
    ``MAX_UPLOAD_BYTES=0`` or ``FASTAPI_PORT=999999`` in a ``.env`` should
    degrade to a working value, not take the process down at import. Clamping
    is still a validation step, so the substitution is logged -- otherwise a
    typo silently turns a tightened limit back into the default, and the
    operator has no way to tell that the setting they wrote is not the
    setting in force.

    ``max_value`` is a required keyword with no default, deliberately. It used
    to default to 32, which is a plausible ceiling for a retry count and
    nothing else: a setting that needed a larger number and forgot to say so
    was silently clamped to 32, and the only symptom was a warning line in a
    log nobody reads, on a value that was wrong in production. No ceiling here
    is worth guessing -- an upload wants 512 MiB, a port wants 65535, a page
    count wants 5000 -- so requiring the argument turns "I forgot to bound
    this" into a ``TypeError`` at import, which is the same class of failure
    as a syntax error: immediate, total, impossible to ship. ``min_value``
    keeps its default of 1 because 1 is a safe floor for a count, and a wrong
    floor degrades towards "too little" rather than a surprising large value.

    A setting that genuinely has no upper bound can pass
    ``max_value=math.inf``. That is a visible, reviewable choice in the call
    rather than an accident, which is the entire point.
    """
    if max_value < min_value:
        # A source-level bug, not a .env error: the range is unsatisfiable, so
        # every value would clamp to the same wrong number.
        raise ValueError(
            f"{name}: max_value={max_value} is below min_value={min_value}; "
            "this range can never be satisfied."
        )
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not an integer; falling back to the default %r.", name, raw, default
        )
        return default
    clamped = max(min_value, min(max_value, value))
    if clamped != value:
        logger.warning(
            "%s=%r is outside the accepted range [%s, %s]; using %r.",
            name, value, min_value, max_value, clamped,
        )
    return clamped


def parse_bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def parse_bool_value(value, default: bool = False) -> bool:
    """Parse a user-supplied form/JSON value such as ``"true"`` or ``"1"``."""
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def parse_csv_env(name: str, default: str = "") -> tuple[str, ...]:
    """把逗号分隔的环境变量解析成 tuple，忽略空段与首尾空白。"""
    raw = os.getenv(name, "").strip() or default.strip()
    if not raw:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def parse_optional_number_env(name: str, min_value: float, max_value: float) -> float | None:
    """读一个「不设即不覆盖」的可选数值，返回 ``None`` 表示环境变量未设置。

    这与 :func:`parse_int_env` 的区别是刻意的：后者总是给出一个值（连同范围钳制），
    适合配置里确实要有一个数字的项；而下面 :data:`PARSER_QUALITY_*` 阈值必须
    保持「未设置」这个状态，因为真正的默认值属于 ingestion/quality.py 的
    ``QualityThresholds``。如果 config 在这里也写死一份，规则作者调整了阈值
    就会在没有配置项的部署里被静默冻结——两处默认值漂移，运维看不出来。

    读不出来（非数字、NaN/inf、超出 ``[min_value, max_value]``）同样返回
    ``None`` 并告警，即退回库默认值，而不是让一个手滑的值悄悄改变质量判定。
    返回 ``float`` 而不是原样返回字符串：``quality._coerce_like`` 负责按字段
    类型转换（``int(3.0)`` 与 ``3`` 等价），这里只需要保证它拿到的是数字。
    """
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; the built-in default stays in force.", name, raw)
        return None
    if not math.isfinite(value):
        logger.warning("%s=%r is not a finite number; the built-in default stays in force.",
                       name, raw)
        return None
    if not min_value <= value <= max_value:
        logger.warning(
            "%s=%r is outside the accepted range [%s, %s]; the built-in default stays in force.",
            name, value, min_value, max_value,
        )
        return None
    return value


#: 合法的 parser feature flag 取值。
PARSER_BACKEND_VALUES = ("legacy", "gateway")
#: 合法的 docling 运行模式取值。
PARSER_DOCLING_MODES = ("inprocess", "serve")
#: 合法的文档隐私模式取值。
DOCUMENT_PRIVACY_MODES = ("local_only", "allow_cloud")
#: 计划中已知的 parser 名称。用于对 PARSER_PRIMARY 做告警而非强制，
#: 这样自研/评测用 parser 仍然能通过名字路由。
PARSER_NAMES = ("legacy", "docling", "mineru")


def _coerce_choice(value, allowed: tuple[str, ...], default: str, env_name: str = "") -> str:
    """校验枚举型设置：合法值原样返回，非法值回落到 ``default`` 并告警。

    回落而不是抛异常，是因为本模块在进程启动时被导入，``.env`` 里一个拼写
    错误不该让整个服务起不来。但静默回落是更糟的失败模式：运维以为策略生效
    了而实际没有——对 ``PARSER_BACKEND``（以为走了新链路）和
    ``DOCUMENT_PRIVACY_MODE``（以为允许出网或以为禁止出网）尤其致命。
    所以回落时在启动日志里留一行，把「我写的设置没生效」变成可检索的信号。
    """
    text = str(value or "").strip().lower()
    if text in allowed:
        return text
    if text:
        logger.warning(
            "%s=%r is not a valid value; expected one of %s. Falling back to %r.",
            env_name or "value", text, ", ".join(allowed), default,
        )
    return default


def normalize_parser_backend(value) -> str:
    """归一化 parser feature flag；非法值回落到 legacy，绝不抛异常。"""
    return _coerce_choice(value, PARSER_BACKEND_VALUES, "legacy", "PARSER_BACKEND")


def normalize_docling_mode(value) -> str:
    """归一化 docling 运行模式；非法值回落到进程内模式。"""
    return _coerce_choice(value, PARSER_DOCLING_MODES, "inprocess", "PARSER_DOCLING_MODE")


def normalize_privacy_mode(value) -> str:
    """归一化文档隐私模式；非法值回落到 local_only（不确定即禁止出网）。"""
    return _coerce_choice(value, DOCUMENT_PRIVACY_MODES, "local_only", "DOCUMENT_PRIVACY_MODE")


def normalize_parser_name(value, default: str) -> str:
    """校验单个 parser 名称；未知名称保留原值但告警。

    与其它三个 normalizer 的区别是「告警但不替换」：parser 注册表是运行期的，
    评测阶段可能注册一个名字不在 :data:`PARSER_NAMES` 里的自研 parser，
    强行改写会让它无法被路由到。这里只保证拼写错误在日志里可见。
    """
    text = str(value or "").strip().lower() or default
    if text not in PARSER_NAMES:
        logger.warning(
            "PARSER_PRIMARY=%r is not a built-in parser (expected one of %s). "
            "Keeping the value; a custom parser may still be registered under it.",
            text, ", ".join(PARSER_NAMES),
        )
    return text


def normalize_http_url(value, default: str, env_name: str) -> str:
    """校验可出网的服务地址，只接受 http/https。

    这些 URL 会被后端主动请求（docling-serve、MinerU API），所以 scheme 必须
    受限：``file://`` 会把远端解析退化成读本地文件，``ftp://``/``gopher://``
    在部分栈上仍是可用的内网读取原语。留空则回落到内置默认地址。
    """
    text = str(value or "").strip()
    if not text:
        return default
    if not text.lower().startswith(("http://", "https://")):
        logger.warning(
            "%s=%r is not an http(s) URL; falling back to the default %r.",
            env_name, text, default,
        )
        return default
    return text.rstrip("/")


def clamp_int_value(value, default: int, *, min_value: int = 1, max_value: int) -> int:
    """Request-time twin of :func:`parse_int_env`, for caller-supplied values.

    Same argument, same reason it is required: a caller clamping an operator
    input to a window would otherwise inherit the 32 ceiling and clamp a
    legitimate result limit to 32 without a word. Unlike the env version this
    one cannot log -- it runs inside a request -- which makes a wrong ceiling
    strictly worse here, and is the stronger reason to force the call to name
    its own bound.
    """
    if max_value < min_value:
        raise ValueError(
            f"max_value={max_value} is below min_value={min_value}; "
            "this range can never be satisfied."
        )
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(min_value, min(max_value, parsed))


def resolve_api_key(explicit_key) -> str:
    """Prefer a request-scoped key, fall back to the server-level env var."""
    explicit_text = str(explicit_key or "").strip()
    if explicit_text:
        return explicit_text
    return os.getenv("OPENAI_API_KEY", "").strip()


# --------------------------------------------------------------------------
# LLM runtime
# --------------------------------------------------------------------------

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
OPENAI_REQUEST_TIMEOUT_SECONDS = parse_int_env("OPENAI_REQUEST_TIMEOUT_SECONDS", default=60,
    min_value=5, max_value=600)
OPENAI_MAX_RETRIES = parse_int_env("OPENAI_MAX_RETRIES", default=3, min_value=1, max_value=10)
MAX_LLM_CONCURRENCY = parse_int_env("OPENAI_MAX_CONCURRENCY", default=5, min_value=1, max_value=32)

#: Global cap on in-flight LLM requests across all sessions and workers.
LLM_REQUEST_SEMAPHORE = threading.BoundedSemaphore(MAX_LLM_CONCURRENCY)

LLM_TEMPERATURE = 0.7

#: Per-response output budget in tokens, counted against the model's reasoning
#: budget as well as its visible answer -- the two share one allowance.
#:
#: This used to be a bare literal, 4000, which was quietly wrong for any
#: reasoning-capable model: a model that spends 4000 tokens thinking returns
#: *empty content* with ``finish_reason == "length"``, and the user sees a
#: failed section rather than a truncated one. The failure is silent in the
#: worst way -- it does not look like a limit, it looks like the model refusing.
#:
#: The default is 8000, chosen as a budget rather than a wish. It is a ceiling
#: against two real things: the longest structured output the prompts ask for
#: (a mermaid diagram or a full markdown section, on the order of 2000-3000
#: tokens), and the reasoning overhead a reasoning model spends before it emits
#: any of it. Doubling 4000 buys the second without abandoning the first -- it
#: leaves room to think and still enough left to answer in full. It is
#: deliberately nowhere near the 32000 a current model advertises, because a
#: default is the value a deployment gets without anyone having decided on one.
#:
#: Operators pointing this at a reasoning model should raise it explicitly; the
#: upper bound leaves room to do so. Setting it too low is now a reported error
#: naming this variable, not an empty section.
# --------------------------------------------------------------------------
# 后台任务（T2.1 paperwhisperer/infra/jobs.py）
# --------------------------------------------------------------------------
# 键名与默认值由 jobs.REQUIRED_CONFIG_KEYS 声明，改动需两边同步。

#: 把长任务移出 API 请求生命周期。关闭后回到同步处理，便于对照排查。
JOBS_ENABLED = parse_bool_env("JOBS_ENABLED", default=True)
JOBS_POLL_INTERVAL_SECONDS = parse_int_env("JOBS_POLL_INTERVAL_SECONDS", default=1,
                                           min_value=1, max_value=60)
#: 租约时长（秒）。worker 在每个可取消边界续租；超过这个时长没有续租的任务
#: 视为 worker 已死，会被下一个 worker 回收。步骤比它长就必须调
#: ``ctx.heartbeat()``，否则一个正常但缓慢的步骤会被误判为崩溃并重复执行。
#: 用整数秒表达：租约不需要亚秒精度，而配置值用 ``N*1000`` 走同一条已验证的
#: 取整与钳制路径，不必为它再引入一个浮点解析器。
JOBS_LEASE_SECONDS = parse_int_env("JOBS_LEASE_SECONDS", default=300,
                                   min_value=5, max_value=3600)
JOBS_MAX_ATTEMPTS = parse_int_env("JOBS_MAX_ATTEMPTS", default=3,
                                  min_value=1, max_value=20)
#: 留空表示按主机名自动生成。多实例部署时每台机器应能区分。
JOBS_WORKER_ID = os.getenv("JOBS_WORKER_ID", "").strip()
JOBS_RETENTION_DAYS = parse_int_env("JOBS_RETENTION_DAYS", default=14,
                                    min_value=1, max_value=3650)

# --------------------------------------------------------------------------
# 可观测性（T4.1 paperwhisperer/core/observability.py）
# --------------------------------------------------------------------------
# 键名与默认值由 observability.OBSERVABILITY_CONFIG_KEYS 声明，改动需两边同步。
# 默认关闭：可观测性必须在被要求时才开启，且开启后不改变既有行为。

#: 是否启用指标与追踪。关闭时 record_* 调用是空操作（实测 0.09 µs/次）。
OBSERVABILITY_ENABLED = parse_bool_env("OBSERVABILITY_ENABLED", default=False)
#: 上界大写返回：``_coerce_choice`` 归一化为小写比较，而日志级别需要
#: 大写字面量。给 allowed 传大写元组会让每个合法值都判为非法，每次启动
#: 都打一条回落告警——一个"只在启动时出现一次"的错误信号。
OBSERVABILITY_LOG_LEVEL = _coerce_choice(
    os.getenv("OBSERVABILITY_LOG_LEVEL", "INFO"),
    ("debug", "info", "warning", "error"),
    "info",
    "OBSERVABILITY_LOG_LEVEL",
).upper()
#: 是否把日志行渲染成 JSON。关闭则仍带关联 id，但走人类可读的格式。
OBSERVABILITY_JSON_LOGS = parse_bool_env("OBSERVABILITY_JSON_LOGS", default=True)
#: 单个进程内保留的追踪事件数上限。有界是因为个人档进程长驻；丢弃会被计数。
OBSERVABILITY_TRACE_LIMIT = parse_int_env("OBSERVABILITY_TRACE_LIMIT", default=2048,
                                          min_value=32, max_value=65536)
#: 单个指标允许的标签基数上限。超过的观测计入 series_overflow_total，
#: 而不是被静默丢弃或抛出——label 基数是调用方的错误，也是本模块的内存。
OBSERVABILITY_MAX_SERIES = parse_int_env("OBSERVABILITY_MAX_SERIES", default=256,
                                         min_value=8, max_value=8192)
#: 指标导出文件；留空表示不导出（仍可查询 prometheus_text()）。
OBSERVABILITY_EXPORTER_PATH = os.getenv("OBSERVABILITY_EXPORTER_PATH", "").strip()
#: 超过此时长的解析/模型调用会被单独记录一次事件，否则会静默地慢掉。
OBSERVABILITY_SLOW_PARSE_MS = parse_int_env("OBSERVABILITY_SLOW_PARSE_MS", default=120000,
                                            min_value=1000, max_value=3600000)
OBSERVABILITY_SLOW_LLM_MS = parse_int_env("OBSERVABILITY_SLOW_LLM_MS", default=30000,
                                          min_value=1000, max_value=3600000)

#: Absolute base URL this deployment is reached at, used when an export has to
#: emit links a reader will open from somewhere other than the machine that
#: produced the file. Left empty, exports carry relative links — correct for a
#: single-user local install, and the only safe default: a wrong absolute URL
#: produces links that look authoritative and resolve to nothing, which is
#: worse than a relative one that still works when opened next to the app.
EXPORT_APP_BASE_URL = os.getenv("EXPORT_APP_BASE_URL", "").strip().rstrip("/")

LLM_MAX_TOKENS = parse_int_env("OPENAI_LLM_MAX_TOKENS", default=8000, min_value=256,
    max_value=32768)

#: Prompt windows, in characters, per analysis task.
DOCUMENT_WINDOW_BUDGET = {
    "summary_chunk": 4000,
    "quotes": 15000,
    "mindmap": 10000,
    "mermaid": 4000,
    "evaluation": 15000,
    "research_brief": 18000,
    "qa": 12000,
}

QA_HISTORY_BUDGET = 8000
QA_TURN_BUDGET = 2400
DOCUMENT_EXCERPT_LIMIT = 12000
SEARCH_REWRITE_CONTEXT_LIMIT = 4000

# --------------------------------------------------------------------------
# Paper search
# --------------------------------------------------------------------------

ARXIV_API_URL = "http://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

PAPER_SEARCH_PARALLEL = parse_bool_env("PAPER_SEARCH_PARALLEL", default=True)
PAPER_SEARCH_RESULT_LIMIT = parse_int_env("PAPER_SEARCH_RESULT_LIMIT", default=8, min_value=1,
    max_value=20)
RECOMMENDATION_RESULT_LIMIT = parse_int_env("RECOMMENDATION_RESULT_LIMIT", default=6, min_value=1,
    max_value=20)
PAPER_SEARCH_ENABLE_REWRITE = parse_bool_env("PAPER_SEARCH_ENABLE_REWRITE", default=True)
PAPER_SEARCH_REWRITE_MODEL = os.getenv("PAPER_SEARCH_REWRITE_MODEL", "").strip()

SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
SEMANTIC_SCHOLAR_TIMEOUT_SECONDS = parse_int_env("SEMANTIC_SCHOLAR_TIMEOUT_SECONDS", default=20,
    min_value=5, max_value=120)
SEMANTIC_SCHOLAR_MAX_RETRIES = parse_int_env("SEMANTIC_SCHOLAR_MAX_RETRIES", default=3, min_value=1,
    max_value=6)
SEMANTIC_SCHOLAR_RETRY_MAX_DELAY = parse_int_env("SEMANTIC_SCHOLAR_RETRY_MAX_DELAY", default=20,
    min_value=2, max_value=120)

ARXIV_TIMEOUT_SECONDS = parse_int_env("ARXIV_TIMEOUT_SECONDS", default=30, min_value=5,
    max_value=180)
ARXIV_MAX_RETRIES = parse_int_env("ARXIV_MAX_RETRIES", default=2, min_value=1, max_value=6)
ARXIV_RETRY_MAX_DELAY = parse_int_env("ARXIV_RETRY_MAX_DELAY", default=8, min_value=2, max_value=60)

# --------------------------------------------------------------------------
# Sessions
# --------------------------------------------------------------------------

SESSION_TTL_SECONDS = parse_int_env("SESSION_TTL_SECONDS", default=24 * 60 * 60, min_value=60,
    max_value=30 * 24 * 60 * 60)
SESSION_CLEANUP_INTERVAL_SECONDS = parse_int_env(
    "SESSION_CLEANUP_INTERVAL_SECONDS", default=10 * 60, min_value=60, max_value=24 * 60 * 60
)
# --------------------------------------------------------------------------
# Hosted 档鉴权与租户配额（T4.5 paperwhisperer/api/auth.py、tenant.py）
# --------------------------------------------------------------------------
# 默认关闭：个人档是默认部署，不能被一个登录界面挡住。Hosted 档缺少
# AUTH_MODE 时 assert_startup_safe() 会拒绝启动，而不是退化成开放访问。

#: ``disabled``（个人档默认）或 ``static_tokens``。
AUTH_MODE = _coerce_choice(
    os.getenv("AUTH_MODE", "disabled"),
    ("disabled", "static_tokens"),
    "disabled",
    "AUTH_MODE",
)
#: 静态令牌表。**分号**分隔的条目，每条 ``token:tenant_id[:scope1,scope2]``。
#: 不用 parse_csv_env：它按逗号切，而 scope 列表本身用逗号——按逗号切会把
#: 一条条目撕成两半，得到两个都不合法的条目而不是一个。这个格式是 auth.py
#: 定的，不是这里。
AUTH_STATIC_TOKENS = os.getenv("AUTH_STATIC_TOKENS", "").strip()
#: 从 Hosted 主机名推导租户用的域名后缀，留空表示不启用。
AUTH_TENANT_DOMAIN = os.getenv("AUTH_TENANT_DOMAIN", "").strip().lower()
#: 配额窗口（秒）。窗口内累计，窗口过后重置。
QUOTA_WINDOW_SECONDS = parse_int_env("QUOTA_WINDOW_SECONDS", default=86400,
                                    min_value=60, max_value=2592000)
#: 文档数与页数由存储实时推导（删除后会回落），因此不设上限时为 0。
QUOTA_MAX_DOCUMENTS = parse_int_env("QUOTA_MAX_DOCUMENTS", default=0,
                                    min_value=0, max_value=1000000)
QUOTA_MAX_PAGES = parse_int_env("QUOTA_MAX_PAGES", default=0,
                                min_value=0, max_value=100000000)
#: LLM 调用与云端解析是花钱的，计数器无法随删除回落，因此必须设上限。
QUOTA_MAX_LLM_CALLS = parse_int_env("QUOTA_MAX_LLM_CALLS", default=0,
                                    min_value=0, max_value=10000000)
QUOTA_MAX_CLOUD_PARSES = parse_int_env("QUOTA_MAX_CLOUD_PARSES", default=0,
                                       min_value=0, max_value=1000000)

#: Whether a session stores the whole document or only a display excerpt.
#:
#: Defaults to True, and it used to be False. With it off, a session stored an
#: empty ``document_content`` and every later read fell back to
#: ``document_excerpt`` — a truncated prefix. Retrieval (T1.7) can only
#: retrieve what it is given, so a browser refresh silently reintroduced the
#: exact defect the retrieval migration removed: the back half of the paper
#: became unreachable, and the answers said so. The flag also stops meaning
#: what its name says; it reads like an opt-in privacy choice and behaves like
#: a correctness setting.
#:
#: The cost is session file size, which is a real cost and is why T2.2 moves
#: sessions into the database and stores the canonical ``ParsedDocument`` — the
#: thing retrieval actually indexes — rather than a flat text blob.
SESSION_PERSIST_FULL_DOCUMENT = parse_bool_env("SESSION_PERSIST_FULL_DOCUMENT", default=True)

# --------------------------------------------------------------------------
# Remote document import
# --------------------------------------------------------------------------

REMOTE_IMPORT_TIMEOUT_SECONDS = parse_int_env("REMOTE_IMPORT_TIMEOUT_SECONDS", default=30,
    min_value=5, max_value=180)

PUBLIC_HOSTNAME_CACHE_TTL_SECONDS = 300
PUBLIC_HOSTNAME_CACHE_MAX_SIZE = 512

# --------------------------------------------------------------------------
# Document ingestion limits
# --------------------------------------------------------------------------

#: 单个上传文件的体积上限（字节）。同时也是远程下载与 API 上传校验的实际上限：
#: 下面 MAX_CONTENT_LENGTH 指向同一个值，所以改这一个环境变量对所有入口生效。
MAX_UPLOAD_BYTES = parse_int_env(
    "MAX_UPLOAD_BYTES", default=16 * 1024 * 1024, min_value=1024 * 1024, max_value=512 * 1024 * 1024
)

#: 旧名。保留是因为它已经是上传校验、远程下载和现有测试的公开接口，删除会让
#: 调用方全部改一遍而不改变任何行为；定义为 MAX_UPLOAD_BYTES 的别名而不是另一份
#: 字面量，是为了让「.env 里改大了上限但上传仍被 16MB 挡下」这种问题不可能出现。
#: 新代码请直接读 MAX_UPLOAD_BYTES。
MAX_CONTENT_LENGTH = MAX_UPLOAD_BYTES

#: 单个文档允许解析的最大页数，超出的文档在网关层直接拒绝，避免超大文件拖垮进程。
#: 注意：网关尚未实现这个拒绝（见 T0.8 交接说明），目前只有 health 端点会报告它。
MAX_DOCUMENT_PAGES = parse_int_env("MAX_DOCUMENT_PAGES", default=400, min_value=1, max_value=5000)

# --------------------------------------------------------------------------
# Storage backend
# --------------------------------------------------------------------------
# 个人档默认 SQLite、托管档默认 Postgres，这组键此前只存在于
# infra/stores/factory.py 的 REQUIRED_CONFIG_KEYS 兜底表里：设置项存在但没有配置
# 入口，改后端只能改代码。搬到这里之后默认值与原兜底表逐字一致，因此「声明配置」
# 这件事本身不改变任何运行行为，只是让原本不可见的开关变得可见。
# 每个默认值都标注了原表位置，漂移时以两处为准并同步。

#: 合法的存储后端名。
STORAGE_BACKEND_VALUES = ("sqlite", "postgres")
#: 合法的 SQLite 全文检索（FTS5）模式。
STORAGE_LEXICAL_FTS5_VALUES = ("auto", "on", "off")

#: 存储后端：sqlite 或 postgres。非法值回落 sqlite（即 factory.py 的
#: BACKEND_SQLITE），理由与 PARSER_BACKEND 相同：宁可退回已知可用的一档，也不
#: 让一个拼错的部署安静地落到另一个后端上。
STORAGE_BACKEND = _coerce_choice(
    os.getenv("STORAGE_BACKEND"), STORAGE_BACKEND_VALUES, "sqlite", "STORAGE_BACKEND"
)

#: SQLite 文件路径。留空回落到 context/stores/paperwhisperer.sqlite3，即
#: postgres_backend.default_sqlite_path() 的返回值，所以两种装法指向同一个文件。
STORAGE_SQLITE_PATH = (
    os.getenv("STORAGE_SQLITE_PATH", "").strip() or "context/stores/paperwhisperer.sqlite3"
)

#: SQLite 写锁等待超时（秒）。默认 30 覆盖一次较重的索引写入。
STORAGE_SQLITE_TIMEOUT_SECONDS = parse_int_env(
    "STORAGE_SQLITE_TIMEOUT_SECONDS", default=30, min_value=1, max_value=600
)

#: FTS5 模式：auto 按 SQLite 编译能力探测，on 强制要求，off 退回 LIKE。
STORAGE_LEXICAL_FTS5 = _coerce_choice(
    os.getenv("STORAGE_LEXICAL_FTS5"), STORAGE_LEXICAL_FTS5_VALUES, "auto", "STORAGE_LEXICAL_FTS5"
)

#: Postgres DSN。留空表示未配置 Postgres；factory.py 在为空时还会依次尝试
#: PAPERWHISPERER_PG_DSN / DATABASE_URL / POSTGRES_DSN，这个顺序保留不变。
STORAGE_PG_DSN = os.getenv("STORAGE_PG_DSN", "").strip()

#: Postgres schema 名。
STORAGE_PG_SCHEMA = os.getenv("STORAGE_PG_SCHEMA", "").strip() or "public"

#: Postgres 连接超时（秒）。
STORAGE_PG_CONNECT_TIMEOUT_SECONDS = parse_int_env(
    "STORAGE_PG_CONNECT_TIMEOUT_SECONDS", default=5, min_value=1, max_value=120
)

#: 是否启用 pgvector 近似检索（ANN）。默认关闭：托管档的向量检索先走精确扫描，
#: 语料规模上来后再开，且开之前需要确认目标库装了 pgvector 扩展。
STORAGE_PG_ANN = parse_bool_env("STORAGE_PG_ANN", default=False)

# --------------------------------------------------------------------------
# Parser gateway
# --------------------------------------------------------------------------

#: feature flag：legacy 走旧解析链路，gateway 走统一解析网关。非法值回落 legacy。
PARSER_BACKEND = normalize_parser_backend(os.getenv("PARSER_BACKEND", "legacy"))

#: 主路由 parser 名，bake-off 决策落档后调整。未知名字不替换，只告警。
PARSER_PRIMARY = normalize_parser_name(os.getenv("PARSER_PRIMARY", "docling"), "docling")

#: 分级 fallback 链，逗号分隔；空元组表示不做 fallback。
#: Per-format parser preference, keyed by media type.
#:
#: A single global order cannot express the routing decision the bake-off
#: reached (``docs/PARSER_BAKE_OFF.md`` §6). The measured fact is that
#: docling returns a DOCX **successfully** with ``locator_coverage = 0`` on
#: 4/4 documents, while legacy returns 1.0 -- so an exception-based fallback
#: never fires, because nothing failed. The fork has to be declared by
#: format, not inferred from a failure.
#:
#: Parsed from a ``type=parser`` comma-separated list so an operator can
#: override it without editing code.
PARSER_PRIMARY_BY_MEDIA_TYPE: dict[str, str] = {}
for _entry in parse_csv_env("PARSER_PRIMARY_BY_MEDIA_TYPE", default="").__iter__():
    if "=" in _entry:
        _media, _parser = _entry.split("=", 1)
        _media, _parser = _media.strip().lower(), _parser.strip().lower()
        if _media and _parser:
            PARSER_PRIMARY_BY_MEDIA_TYPE[_media] = _parser

#: The DOCX exception, applied when the operator has not said otherwise.
#: Kept as a named constant rather than a literal in the router so that the
#: measured reason travels with the exception.
DOCX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
PARSER_PRIMARY_BY_MEDIA_TYPE.setdefault(DOCX_MEDIA_TYPE, "legacy")

PARSER_FALLBACK_CHAIN = parse_csv_env("PARSER_FALLBACK_CHAIN")

#: 单个 parser 调用的超时时间（秒）。
PARSER_TIMEOUT_SECONDS = parse_int_env("PARSER_TIMEOUT_SECONDS", default=120, min_value=5,
    max_value=1800)

#: 强制走 OCR，用于扫描件与图片型 PDF 的本地修复重试。
PARSER_FORCE_OCR = parse_bool_env("PARSER_FORCE_OCR", default=False)

#: 是否启用解析缓存（按 sha256 + parser + version + options_hash 命中）。
PARSER_CACHE_ENABLED = parse_bool_env("PARSER_CACHE_ENABLED", default=True)

#: parser 原始产物（raw artifact）落盘目录，用于问题回溯与离线复现。
PARSER_ARTIFACT_DIR = os.getenv("PARSER_ARTIFACT_DIR", "").strip() or "context/parser_artifacts"

#: 解析缓存目录。留空表示由 ingestion/orchestrator.py 回落到 context/parse_cache，
#: 这里不填默认值是为了让「缓存放在哪」和「缓存存了什么」各自只有一个来源。
PARSER_CACHE_DIR = os.getenv("PARSER_CACHE_DIR", "").strip()

#: docling 运行模式：inprocess 进程内，serve 走 docling-serve HTTP。
PARSER_DOCLING_MODE = normalize_docling_mode(os.getenv("PARSER_DOCLING_MODE", "inprocess"))

#: docling-serve 服务地址，仅 serve 模式使用。非 http(s) 地址会被拒绝并回落到默认。
PARSER_DOCLING_SERVE_URL = normalize_http_url(
    os.getenv("PARSER_DOCLING_SERVE_URL", ""), "http://localhost:5001", "PARSER_DOCLING_SERVE_URL"
)

# --------------------------------------------------------------------------
# Parse quality gate (T0.6)
# --------------------------------------------------------------------------
# 每个阈值默认都是 None ——「未设置」。这不是省事，而是刻意的：这些阈值的权威
# 默认值在 ingestion/quality.py 的 QualityThresholds 里，规则作者会调整它们
# （min_chars_per_page 之类）。config 若再写死一份，任何一次规则调整都会在没配
# 环境变量的部署里被静默冻结，两处默认值从此漂移。quality.py 用
# ``getattr(config, "PARSER_QUALITY_<字段名大写>", None)`` 读取，None 即回落库
# 默认值，所以这里只需保证「键存在、值可解析、越界有告警」。
#
# 范围按物理意义给死（比率必在 0-1，计数必为非负整数），拦的是手滑而不是调参。

#: 至少产出一个正文元素的页面占比下限；低于此值说明解析中途放弃（最常见的静默截断）。
PARSER_QUALITY_MIN_PAGE_COVERAGE_RATIO = parse_optional_number_env(
    "PARSER_QUALITY_MIN_PAGE_COVERAGE_RATIO", 0.0, 1.0)

#: 无正文的页面占比上限；未 OCR 的扫描件和被压成一栏的双栏版式都在这里露出来。
PARSER_QUALITY_MAX_BLANK_PAGE_RATIO = parse_optional_number_env(
    "PARSER_QUALITY_MAX_BLANK_PAGE_RATIO", 0.0, 1.0)

#: 解码失败字符占比上限。标题里一个 U+FFFD 是噪声，全文二十分之一就是文本层坏了。
PARSER_QUALITY_MAX_GARBLED_CHAR_RATIO = parse_optional_number_env(
    "PARSER_QUALITY_MAX_GARBLED_CHAR_RATIO", 0.0, 1.0)

#: 短行（碎片）占比上限。两端对齐的表格、公式和窄栏都会产生短行，所以该规则只告警。
PARSER_QUALITY_MAX_FRAGMENTED_LINE_RATIO = parse_optional_number_env(
    "PARSER_QUALITY_MAX_FRAGMENTED_LINE_RATIO", 0.0, 1.0)

#: 判定「碎片行」的长度上限（行长度 <= 该值即视为碎片）。
PARSER_QUALITY_MAX_FRAGMENT_LINE_CHARS = parse_optional_number_env(
    "PARSER_QUALITY_MAX_FRAGMENT_LINE_CHARS", 1, 1000)

#: 少于这么多行时，比率只是轶事而非证据，规则弃权而不是对三行摘要报警。
PARSER_QUALITY_MIN_LINES_FOR_RATIO = parse_optional_number_env(
    "PARSER_QUALITY_MIN_LINES_FOR_RATIO", 1, 100000)

#: 页眉/页脚（running furniture）重复多少次开始上报。两页文档重复两次是正常的。
PARSER_QUALITY_MIN_FURNITURE_REPEAT_COUNT = parse_optional_number_env(
    "PARSER_QUALITY_MIN_FURNITURE_REPEAT_COUNT", 1, 10000)

#: 每页正文字符数下限（未分页格式的平均值）。密集论文远超此值，低于则文本层多半没读出来。
PARSER_QUALITY_MIN_CHARS_PER_PAGE = parse_optional_number_env(
    "PARSER_QUALITY_MIN_CHARS_PER_PAGE", 0, 1000000)

#: 无分页格式的全文正文字符数下限。5 个字符的上传不是「降质的论文」，它没有可引用的内容。
PARSER_QUALITY_MIN_DOCUMENT_CHARS = parse_optional_number_env(
    "PARSER_QUALITY_MIN_DOCUMENT_CHARS", 0, 10000000)

#: 空壳表格（只有轮廓没有内容）占比上限。
PARSER_QUALITY_MAX_EMPTY_TABLE_RATIO = parse_optional_number_env(
    "PARSER_QUALITY_MAX_EMPTY_TABLE_RATIO", 0.0, 1.0)

#: 可解析回原文的定位信息占比下限。P0 验收线是 1.0，低于它就是对读者的失信。
PARSER_QUALITY_MIN_RESOLVABLE_LOCATOR_RATIO = parse_optional_number_env(
    "PARSER_QUALITY_MIN_RESOLVABLE_LOCATOR_RATIO", 0.0, 1.0)

#: 逐规则的严重级覆盖，格式 ``rule_id=warn|fail``，逗号分隔。
#: 只能改一条规则报成什么，不能让它「本来不报却报出来」——见
#: quality.QualityThresholds.severity_overrides 的说明。空字符串表示不覆盖。
PARSER_QUALITY_SEVERITY_OVERRIDES = os.getenv("PARSER_QUALITY_SEVERITY_OVERRIDES", "").strip()

# --------------------------------------------------------------------------
# Document privacy
# --------------------------------------------------------------------------

#: local_only 禁止任何云端出网；allow_cloud 才允许云 fallback。非法值回落 local_only。
DOCUMENT_PRIVACY_MODE = normalize_privacy_mode(os.getenv("DOCUMENT_PRIVACY_MODE", "local_only"))

#: 用户是否已显式同意把文档送往云端解析；未同意时即使 privacy 允许也不出网。
DOCUMENT_CLOUD_CONSENT_GRANTED = parse_bool_env("DOCUMENT_CLOUD_CONSENT_GRANTED", default=False)

# --------------------------------------------------------------------------
# MinerU cloud parser
# --------------------------------------------------------------------------

#: MinerU API token，留空表示未配置云解析能力（不发起任何云调用）。
MINERU_API_TOKEN = os.getenv("MINERU_API_TOKEN", "").strip()

#: MinerU API 根地址。非 http(s) 地址会被拒绝并回落到默认。
MINERU_API_BASE_URL = normalize_http_url(
    os.getenv("MINERU_API_BASE_URL", ""), "https://mineru.net", "MINERU_API_BASE_URL"
)

#: 单次 MinerU HTTP 请求超时（秒）。
MINERU_TIMEOUT_SECONDS = parse_int_env("MINERU_TIMEOUT_SECONDS", default=60, min_value=5,
    max_value=600)

#: MinerU 请求失败时的重试次数。
MINERU_MAX_RETRIES = parse_int_env("MINERU_MAX_RETRIES", default=3, min_value=1, max_value=10)

#: 轮询 MinerU 任务状态的间隔（秒）。
MINERU_POLL_INTERVAL_SECONDS = parse_int_env("MINERU_POLL_INTERVAL_SECONDS", default=5, min_value=1,
    max_value=60)

#: 轮询 MinerU 任务状态的总超时（秒），超时视为解析失败。
MINERU_POLL_TIMEOUT_SECONDS = parse_int_env("MINERU_POLL_TIMEOUT_SECONDS", default=900,
    min_value=30, max_value=3600)

# --------------------------------------------------------------------------
# MinerU quota
# --------------------------------------------------------------------------
# 云解析是按次计费的外部资源，而额度耗尽往往先表现为 402/4xx 而不是本地错误。
# 下面两个上限让调用方能在发出请求之前就知道「今天不能再发了」——T0.5 的降级
# 逻辑（额度耗尽不重试、直接降级到本地 parser）需要这两个值来区分「可重试的
# 429」和「不可重试的额度错误」。默认为 0 / 1，即不额外改变当前行为。

#: 单个 MinerU 账号每天允许发起的云解析任务数；0 表示不限制。
MINERU_DAILY_CALL_LIMIT = parse_int_env("MINERU_DAILY_CALL_LIMIT", default=0, min_value=0,
    max_value=100000)

#: 同时在途的 MinerU 任务数。默认串行：云解析成本高，并发放大费用与限流概率。
MINERU_MAX_CONCURRENT_TASKS = parse_int_env("MINERU_MAX_CONCURRENT_TASKS", default=1, min_value=1,
    max_value=8)

# --------------------------------------------------------------------------
# Embedding / rerank（P1 使用，预留配置）
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# 本地 embedding provider（可选 extra，见 D3）
#
# 放在 EMBEDDING_MODEL / DIMENSION / PROVIDER **之前**，因为下面那三个要在
# 本地 provider 启用时改写成它——这就是「切 provider 只需要改配置」的落点。
#
# 规划 D3 写死：LLM / embedding / rerank 全部走 OpenAI 兼容 API，「不做本地
# BGE-M3」。所以下面这一组是**可选开关而不是替代品**，默认关闭：默认路径仍是
# EMBEDDING_API_BASE_URL，本地 provider 只服务三种场景——无网络无 key 的个人
# 部署、必须跑真实 embedding 链路的测试、以及要把两个 provider 放在同一篇
# 文档上对比的选型。默认 EMBEDDING_LOCAL_ENABLED=0，因此 base install 不会
# 因为下面任何一个名字而 import 一个可选依赖。
# --------------------------------------------------------------------------

#: 是否启用本地 embedding provider。0/false 时 llm/embeddings_local.py 的
#: 解析函数直接走 OpenAI 兼容网关，本组其余设置全部不生效。
EMBEDDING_LOCAL_ENABLED = parse_bool_env("EMBEDDING_LOCAL_ENABLED", default=False)

#: 本地 provider 写进 index_version 的名字。必须与
#: llm/embeddings_local.py 的 LOCAL_PROVIDER_NAME 一致：chunking 读的是这
#: 里的 EMBEDDING_PROVIDER，不是那个模块的常量。
EMBEDDING_LOCAL_PROVIDER = os.getenv("EMBEDDING_LOCAL_PROVIDER", "").strip().lower() or (
    "fastembed_local"
)

#: 本地模型名，写入 index_version 参与索引版本编码。默认是一个多语言 MiniLM
#: 而不是纯英文模型：本项目语料是中英混合的（eval/corpus/mixed_zh_en），而纯
#: 英文模型对中文 query 只会给出「对所有段落都差不多」的向量——那正是
#: rag/retrieve.py 的 DENSE_TIE_BAND 要识别的「dense 通道没有区分力」。
EMBEDDING_LOCAL_MODEL = os.getenv("EMBEDDING_LOCAL_MODEL", "").strip() or (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

#: 本地模型权重的缓存目录。留空表示用 fastembed 自己的默认位置；填一个目录是
#: 为了让「首次下载」这件事在部署脚本里可预置，也让容器里没有可写 HOME 的
#: 情况有地方放权重。
EMBEDDING_LOCAL_MODEL_DIR = os.getenv("EMBEDDING_LOCAL_MODEL_DIR", "").strip()

#: ONNX Runtime 的线程数。0 = 交给 fastembed 自行决定。ONNX 内部已经多线程，
#: 这一项只在机器核数很多而 CPU 亲和性设错时才需要调。
EMBEDDING_LOCAL_THREADS = parse_int_env("EMBEDDING_LOCAL_THREADS", default=0, min_value=0,
    max_value=64)

#: 模型是否要求 query 与 passage 使用不同的前缀（E5 / Jina / GTE 这类）。
#: 默认 False，即 query 与文档走同一个 embed()——MiniLM 注册表条目明确写了
#: 无需前缀。打开它会让 llm/embeddings_local.py 改用 query_embed()。
EMBEDDING_LOCAL_ASYMMETRIC = parse_bool_env("EMBEDDING_LOCAL_ASYMMETRIC", default=False)

#: 上面 EMBEDDING_LOCAL_MODEL 的向量宽度**声明值**。存在的原因是
#: config.py 不能在模块级 import 一个可选 extra（而 import 一次 fastembed 要
#: 拉起 ONNX Runtime），而 index_version 必须在 chunking 阶段就写出来——
#: 那时 fastembed 还不该被 import。所以这里声明，llm/embeddings_local.py 在
#: 构造 provider 时拿 fastembed 注册表里的真值核对；不一致直接抛
#: EmbeddingDimensionMismatchError 并把正确的数字写进错误信息。
#:
#: 换模型时这个数必须一起改，否则 provider 拒绝启动——这比「启动成功、索引
#: 里存着一批互相无法比较的向量」要便宜得多。
EMBEDDING_LOCAL_DIMENSION = parse_int_env("EMBEDDING_LOCAL_DIMENSION", default=384,
    min_value=64, max_value=8192)

#: embedding 服务地址，留空时复用 OPENAI_BASE_URL。本地 provider 启用时它不
#: 参与任何请求，只是留在配置里，以便切回远端时无需改动其他项。
EMBEDDING_API_BASE_URL = os.getenv("EMBEDDING_API_BASE_URL", "").strip() or OPENAI_BASE_URL

#: embedding 模型名，写入 index_version。
#:
#: 本地 provider 启用时这个名字**必须**是本地模型名，而不是 env 里可能残留的
#: text-embedding-3-small：chunking 靠它拼 index_version，provider 的向量靠
#: 它认账，两个不一致就是「索引声称一个模型、存着另一个模型的向量」。所以这里
#: 直接取本地设置，而不是让每个调用方各自记住这条约定。
EMBEDDING_MODEL = (
    EMBEDDING_LOCAL_MODEL
    if EMBEDDING_LOCAL_ENABLED
    else (os.getenv("EMBEDDING_MODEL", "").strip() or "text-embedding-3-small")
)

#: embedding 向量维度，写入 index_version；换模型必须换维度并重建索引。
#: 本地 provider 启用时取上面那个声明值，并交给 llm/embeddings_local.py 与
#: fastembed 注册表核对——核对不通过就拒绝，而不是猜一个宽度。
EMBEDDING_DIMENSION = (
    EMBEDDING_LOCAL_DIMENSION
    if EMBEDDING_LOCAL_ENABLED
    else parse_int_env("EMBEDDING_DIMENSION", default=1536, min_value=64, max_value=8192)
)

#: embedding provider 名，写入 index_version 参与索引版本编码。取值开放（各家
#: 自建服务自取名字），所以这里只做小写归一不校验白名单——provider 的真实分支
#: 在 rag/chunking.py，config 不该复制一份会漂移的清单。默认与该文件读不到
#: 配置时的回落值一致。
EMBEDDING_PROVIDER = (
    EMBEDDING_LOCAL_PROVIDER
    if EMBEDDING_LOCAL_ENABLED
    else (os.getenv("EMBEDDING_PROVIDER", "").strip().lower() or "openai_compatible")
)

#: 每次批量 embedding 的文本条数。调大减少请求数，调小降低单次超时风险。
EMBEDDING_BATCH_SIZE = parse_int_env("EMBEDDING_BATCH_SIZE", default=64, min_value=1,
    max_value=2048)

#: 同时在途的 embedding 请求数。默认串行度低于 LLM 侧（5），因为 embedding
#: 网关通常与语料重建一起运行，一次重建会同时打出大量请求。
EMBEDDING_MAX_CONCURRENCY = parse_int_env(
    "EMBEDDING_MAX_CONCURRENCY", default=4, min_value=1, max_value=64
)

#: embedding 请求失败重试次数。
EMBEDDING_MAX_RETRIES = parse_int_env("EMBEDDING_MAX_RETRIES", default=3, min_value=1, max_value=10)

#: 单次 embedding HTTP 请求超时（秒）。
EMBEDDING_REQUEST_TIMEOUT_SECONDS = parse_int_env(
    "EMBEDDING_REQUEST_TIMEOUT_SECONDS", default=60, min_value=1, max_value=600
)

#: embedding 响应缓存的条目上限。条目按 (model, content_hash) 计，进程内 LRU，
#: 所以这个数字同时是内存上界；越大命中越准，代价是常驻内存线性增长。
EMBEDDING_MAX_CACHE_ENTRIES = parse_int_env(
    "EMBEDDING_MAX_CACHE_ENTRIES", default=4096, min_value=1, max_value=1000000
)

#: 每百万 token 的 embedding 价格，用于把调用成本记进 analysis_run。留空
#: （None）表示「不知道价格」而不是「免费」——两者在成本报告里必须可区分，
#: 所以这里用 optional number 而不是 parse_int_env 的 0 默认值。
EMBEDDING_PRICE_PER_MTOK = parse_optional_number_env(
    "EMBEDDING_PRICE_PER_MTOK", 0.0, 1_000_000.0
)

# --------------------------------------------------------------------------
# RAG chunking
# --------------------------------------------------------------------------

#: 单个 chunk 的 token 上限。超过就切分，过小则检索片段失去上下文。
RAG_CHUNK_MAX_TOKENS = parse_int_env("RAG_CHUNK_MAX_TOKENS", default=512, min_value=64,
    max_value=8192)

# --------------------------------------------------------------------------
# Bibliographic enrichment
# --------------------------------------------------------------------------
# enrich.py 此前直接读 os.environ，默认值散在那个模块里。这里把地址与联系邮箱
# 收进统一配置，但默认值与 enrich.py 的原值逐字一致，所以搬动不改变行为。

#: Crossref works API 地址。
CROSSREF_API_URL = normalize_http_url(
    os.getenv("CROSSREF_API_URL", ""), "https://api.crossref.org/works", "CROSSREF_API_URL"
)

#: OpenAlex works API 地址。
OPENALEX_API_URL = normalize_http_url(
    os.getenv("OPENALEX_API_URL", ""), "https://api.openalex.org/works", "OPENALEX_API_URL"
)

#: Crossref polite pool 联系邮箱。留空表示不声明；两家都把它当作限速的礼仪，
#: 设置一个真实邮箱能进更宽松的池子。
ENRICHMENT_CONTACT_EMAIL = os.getenv("ENRICHMENT_CONTACT_EMAIL", "").strip()

#: rerank 默认关闭，是否启用由检索评测决定。
RERANK_ENABLED = parse_bool_env("RERANK_ENABLED", default=False)

#: rerank 模型名，留空表示未选定 provider。
RERANK_MODEL = os.getenv("RERANK_MODEL", "").strip()

# --------------------------------------------------------------------------
# Page rasterisation (T3.3)
# --------------------------------------------------------------------------
# 装 `render` extra 之后，证据面板可以显示 PDF 页面的真实渲染图。这组设置全部
# 是「渲染 nicety」而不是核心功能：没装 extra 时它们一个都不起作用，路由退回
# 501，前端继续用 view 里的坐标重建布局。所以全部保持可选、且都有可用默认值。

#: 页图渲染的默认缩放，即「每个 PDF 点对应多少设备像素」。默认 2.0（144 DPI）：
# 双栏论文的正文在 600px 宽的面板里刚好可读，读者不必缩放；这个数也是
# pypdfium2 文档里的示例值，运维对照上游文档时是对齐的。
PAGE_RENDER_SCALE = (
    parse_optional_number_env("PAGE_RENDER_SCALE", 0.5, 4.0) or 2.0
)

#: 允许请求的最大缩放。这是本项目里唯一一处「请求可以按线上传来的数字分配
# 图像」的地方：一张 A0 海报按 scale=64 渲染约 2.56 亿像素，光原始像素就接近
# 1GB。上限定得既能让 3x 屏拿到 3 倍图，又让最坏情况是一张大图而不是进程被杀。
PAGE_RENDER_MAX_SCALE = (
    parse_optional_number_env("PAGE_RENDER_MAX_SCALE", 0.5, 8.0) or 4.0
)

#: 渲染页图的进程内缓存条目上限。渲染不便宜，但也不该无限增长——个人档是长驻
# 进程，无界的渲染缓存就是内存泄漏。命中键是「原始文件摘要 + 页码 + 缩放」，
# 而不是 document_id：原始字节决定像素，所以换解析（document_id 不变、版本变）
# 不会让缓存失效，也不会让缓存过期。
PAGE_RENDER_CACHE_MAX_ENTRIES = parse_int_env(
    "PAGE_RENDER_CACHE_MAX_ENTRIES", default=64, min_value=0, max_value=4096
)

#: 同一缓存的字节上限。一张 2 倍渲染的 A4 页约 400KB，只按条目数封顶的话
# 4096 条就是 1.6GB；按字节封顶才是内存真正的不变量。两个上限同时生效，
# 哪个先到按哪个淘汰。
PAGE_RENDER_CACHE_MAX_BYTES = parse_int_env(
    "PAGE_RENDER_CACHE_MAX_BYTES", default=32 * 1024 * 1024, min_value=0, max_value=1 << 30
)

#: 对象存储的根目录，T2.2 的 original 层在这里面。页图渲染按文档记录的 sha256
#: 到这一层里按内容寻址找原始字节（find_by_digest 是算术，不是扫描）。
#: 留空回落到 context/objects，与 STORAGE_SQLITE_PATH 的相对路径风格一致。
#:
#: 注意：当前没有任何代码把上传写进这一层，而且 /api/analyze 在分析结束后会删掉
#: 临时上传文件。所以这一项在 T2.2 把 original 层接上之前是「配了也没有东西可
#: 找」的——路由对此会明确报「原始文件不可用」，而不是渲染一张空图。
PAGE_RENDER_ORIGINALS_DIR = (
    os.getenv("PAGE_RENDER_ORIGINALS_DIR", "").strip() or "context/objects"
)

# --------------------------------------------------------------------------
# Server bootstrap
# --------------------------------------------------------------------------

FASTAPI_HOST = os.getenv("FASTAPI_HOST") or os.getenv("FLASK_HOST", "0.0.0.0")
FASTAPI_PORT = parse_int_env(
    "FASTAPI_PORT",
    default=parse_int_env("FLASK_PORT", default=5000, min_value=1, max_value=65535),
    min_value=1,
    max_value=65535,
)
FASTAPI_RELOAD = parse_bool_env("FASTAPI_RELOAD", default=parse_bool_env("FLASK_DEBUG",
    default=False))

TEMPLATES_DIR = PROJECT_ROOT / "templates"
STATIC_DIR = TEMPLATES_DIR / "static"
LOGO_PATH = PROJECT_ROOT / "logo.ico"


def ensure_runtime_folders() -> None:
    """Create the writable runtime folders the app depends on."""
    for folder in RUNTIME_FOLDERS:
        os.makedirs(folder, exist_ok=True)


def _configured_quality_thresholds() -> dict:
    """The quality thresholds this deployment actually overrides.

    Scans the module namespace for ``PARSER_QUALITY_*`` rather than keeping a
    second copy of the list: the thresholds belong to
    ``ingestion.quality.QualityThresholds``, and a second list here would be
    one more place to forget to update when a rule is added.
    """
    prefix = "PARSER_QUALITY_"
    overrides = {
        name[len(prefix):].lower(): value
        for name, value in globals().items()
        if name.startswith(prefix) and value is not None and name != f"{prefix}SEVERITY_OVERRIDES"
    }
    return dict(sorted(overrides.items()))


def _page_renderer_name() -> str:
    """页图渲染后端的真名，或 ``"none"``。

    转发给 :func:`paperwhisperer.rag.evidence.page_renderer_name` 而不是在这里
    自己探测一次，理由是「payload 告诉客户端的能力」和「路由实际会不会渲染」
    必须是同一个答案。两个各自为政的探测会漂移，漂移出来的症状是：payload 说
    有渲染器，路由回 501，客户端把空面板显示成「加载失败」。
    """
    from paperwhisperer.rag.evidence import page_renderer_name

    return page_renderer_name()


def describe_parser_settings() -> dict:
    """导出解析链路配置供 /api/health 与诊断使用，只含非敏感项。

    MINERU_API_TOKEN 只暴露是否已配置，不暴露实际值。
    """
    return {
        "parser_backend": PARSER_BACKEND,
        "parser_primary": PARSER_PRIMARY,
        "parser_fallback_chain": list(PARSER_FALLBACK_CHAIN),
        "parser_timeout_seconds": PARSER_TIMEOUT_SECONDS,
        "parser_force_ocr": PARSER_FORCE_OCR,
        "parser_cache_enabled": PARSER_CACHE_ENABLED,
        "parser_cache_dir": PARSER_CACHE_DIR,
        "parser_artifact_dir": PARSER_ARTIFACT_DIR,
        "parser_docling_mode": PARSER_DOCLING_MODE,
        "parser_docling_serve_url": PARSER_DOCLING_SERVE_URL,
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "max_content_length": MAX_CONTENT_LENGTH,
        "max_document_pages": MAX_DOCUMENT_PAGES,
        "document_privacy_mode": DOCUMENT_PRIVACY_MODE,
        "document_cloud_consent_granted": DOCUMENT_CLOUD_CONSENT_GRANTED,
        "mineru_configured": bool(MINERU_API_TOKEN),
        "mineru_api_base_url": MINERU_API_BASE_URL,
        "mineru_timeout_seconds": MINERU_TIMEOUT_SECONDS,
        "mineru_max_retries": MINERU_MAX_RETRIES,
        "mineru_poll_interval_seconds": MINERU_POLL_INTERVAL_SECONDS,
        "mineru_poll_timeout_seconds": MINERU_POLL_TIMEOUT_SECONDS,
        "mineru_daily_call_limit": MINERU_DAILY_CALL_LIMIT,
        "mineru_max_concurrent_tasks": MINERU_MAX_CONCURRENT_TASKS,
        "embedding_api_base_url": EMBEDDING_API_BASE_URL,
        "llm_max_tokens": LLM_MAX_TOKENS,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "embedding_provider": EMBEDDING_PROVIDER,
        "embedding_batch_size": EMBEDDING_BATCH_SIZE,
        "embedding_max_concurrency": EMBEDDING_MAX_CONCURRENCY,
        "embedding_max_retries": EMBEDDING_MAX_RETRIES,
        "embedding_request_timeout_seconds": EMBEDDING_REQUEST_TIMEOUT_SECONDS,
        "embedding_max_cache_entries": EMBEDDING_MAX_CACHE_ENTRIES,
        "embedding_price_per_mtok": EMBEDDING_PRICE_PER_MTOK,
        # 本地 provider 只报「是否启用 + 模型名」，不报 extra 装没装：那是
        # llm/embeddings_local.py 的 fastembed_installed() 的活，在这份摘要
        # 里再探一次就是第二份会漂移的真相。启用时上面三个 embedding_* 字段
        # 已经是本地模型的 provider/model/dim，index_version 因此会随之改变。
        "embedding_local_enabled": EMBEDDING_LOCAL_ENABLED,
        "embedding_local_model": EMBEDDING_LOCAL_MODEL,
        "rag_chunk_max_tokens": RAG_CHUNK_MAX_TOKENS,
        # 只报「是否配置」而不报值：DSN 里带口令。
        "storage_backend": STORAGE_BACKEND,
        "storage_sqlite_path": STORAGE_SQLITE_PATH,
        "storage_sqlite_timeout_seconds": STORAGE_SQLITE_TIMEOUT_SECONDS,
        "storage_lexical_fts5": STORAGE_LEXICAL_FTS5,
        "storage_pg_configured": bool(STORAGE_PG_DSN),
        "storage_pg_schema": STORAGE_PG_SCHEMA,
        "storage_pg_connect_timeout_seconds": STORAGE_PG_CONNECT_TIMEOUT_SECONDS,
        "storage_pg_ann": STORAGE_PG_ANN,
        "crossref_api_url": CROSSREF_API_URL,
        "openalex_api_url": OPENALEX_API_URL,
        "enrichment_contact_email": ENRICHMENT_CONTACT_EMAIL,
        "rerank_enabled": RERANK_ENABLED,
        "rerank_model": RERANK_MODEL,
        # 页图渲染能力：extra 装了就是 renderer 的名字，没装就是 "none"，
        # 与证据 payload 里的 page_image.renderer 同一个来源，不另立一份真相。
        # 延迟导入：rag.evidence 走 infra.stores.interfaces，而那一层再往下会读
        # config；模块级导入会把这个摘要函数变成一条环。
        "page_render_renderer": _page_renderer_name(),
        "page_render_scale": PAGE_RENDER_SCALE,
        "page_render_max_scale": PAGE_RENDER_MAX_SCALE,
        "page_render_cache_max_entries": PAGE_RENDER_CACHE_MAX_ENTRIES,
        "page_render_cache_max_bytes": PAGE_RENDER_CACHE_MAX_BYTES,
        # 只列出真正被覆盖的阈值：未设置的项在 quality.py 那里用库默认值，
        # 在这里报成 None 会让人误以为「配置写的是 None」。
        "parser_quality_overrides": _configured_quality_thresholds(),
        "parser_quality_severity_overrides": PARSER_QUALITY_SEVERITY_OVERRIDES,
    }
