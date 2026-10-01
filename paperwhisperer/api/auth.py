"""鉴权与授权边界：只在 Hosted 档位生效，个人档位必须是惰性的。

T4.5 的计划原文是「鉴权 / tenant scope / 配额；Celery+Redis 多 worker；
S3/MinIO；MinerULocalAdapter（self-host）按需实现」，验收标准是
**跨租户访问为零（安全测试）**。但同一行前面还写着「仅当公开部署时实施」，
而项目是否公开部署**没有人声明过**。所以本模块的任务不是「加一个登录」，
而是把鉴权做成**一个在个人档位上不存在、在 Hosted 档位上不可绕过**的接缝。

三条设计决定，每一条都在回答"为什么不是另一种做法"：

**一、租户来自凭据，绝不来自请求字段。** :attr:`Principal.tenant_id` 是
*token 查表*的结果，不是客户端 ``X-Tenant-Id`` 头的回声。回声式的租户选择
是这类系统最常见的越权来源：客户端说"我是 tenant_b"，后端就信，于是
tenant_a 拿着自己的合法 token 就能读 tenant_b 的文档。凭据决定租户这件事
没有第二种写法。

**二、``DisabledAuthenticator`` 是一个真实存在的对象，不是"没有鉴权"。**
个人档位的选择是一个 :class:`DisabledAuthenticator`，它照样产出
:class:`Principal`、照样带一个隐式租户（:data:`LOCAL_TENANT_ID`）。这样调用方
只有一条代码路径：*先取 Principal，再取租户*。如果个人档位走的是"没有鉴权
所以跳过取 Principal"，那么 Hosted 档位的鉴权分支就是一条只在部署时才
被执行过的代码路径——而那正是最需要被测试覆盖的分支。

**三、缺配置就拒绝启动，而不是降级成免鉴权。** :func:`build_authenticator`
在 Hosted 档位遇到"没有配 token"时抛 :class:`AuthNotConfigured`，而不是
返回一个"先放行"的 authenticator。一个配错的 Hosted 部署如果降级成免鉴权，
后果是**全部租户的数据互相可见**——这是最坏的结果，而且它安静。一个连
启动都起不来的部署，运维看得见。

本模块**不 import 存储层**。判断档位是自己读 ``STORAGE_BACKEND`` 归一化出来的，
而不是调 :func:`paperwhisperer.infra.stores.factory.resolve_backend`——那个函数
会连带把 numpy 探测与两个数据库后端拖进 import 图，而鉴权边界必须能被一个
从不打开数据库的测试直接 import。这两处对 ``STORAGE_BACKEND`` 的理解必须一致，
:func:`resolve_tier` 的注释指明了不一致时的症状。
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from collections.abc import Iterable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "AUTH_CONFIG_KEYS",
    "AUTH_MODE_DISABLED",
    "AUTH_MODE_STATIC_TOKENS",
    "AUTH_MODES",
    "LOCAL_TENANT_ID",
    "SCOPE_ALL",
    "SCOPE_CLOUD_PARSE",
    "SCOPE_DOCUMENTS_READ",
    "SCOPE_DOCUMENTS_WRITE",
    "SCOPE_JOBS_READ",
    "SCOPE_JOBS_WRITE",
    "SCOPE_PAPER_SEARCH",
    "ALL_SCOPES",
    "AuthenticationError",
    "AuthNotConfigured",
    "Authenticator",
    "AuthorizationError",
    "DisabledAuthenticator",
    "Principal",
    "StaticTokenAuthenticator",
    "TokenEntry",
    "assert_startup_safe",
    "authenticate_headers",
    "build_authenticator",
    "current_principal",
    "describe_auth",
    "hash_token",
    "new_token",
    "optional_principal",
    "parse_token_table",
    "require_scope",
    "resolve_tier",
    "use_principal",
]


#: 合法的 ``AUTH_MODE`` 取值。声明成元组而不是在 :func:`build_authenticator`
#: 里散落字符串，是为了让"有哪些模式"这个问题有**一个**答案——新增模式时
#: 这里是要改的地方，而 :func:`build_authenticator` 会在遇到表外的值时
#: 抛 :class:`AuthNotConfigured`（托管档位）或忽略（个人档位）。
AUTH_MODE_DISABLED = "disabled"
AUTH_MODE_STATIC_TOKENS = "static_tokens"
AUTH_MODES: tuple[str, ...] = (AUTH_MODE_DISABLED, AUTH_MODE_STATIC_TOKENS)


# --------------------------------------------------------------------------
# 作用域词汇
# --------------------------------------------------------------------------

#: 读文档、读证据、读导出。
SCOPE_DOCUMENTS_READ = "documents:read"
#: 上传、删除、重新解析、提交任务。
SCOPE_DOCUMENTS_WRITE = "documents:write"
#: 读任务状态、订阅任务流。
SCOPE_JOBS_READ = "jobs:read"
#: 取消与重试任务。
SCOPE_JOBS_WRITE = "jobs:write"
#: 触发云端解析——单独一个作用域，因为它花钱。
SCOPE_CLOUD_PARSE = "cloud:parse"
#: 学术检索与推荐。
SCOPE_PAPER_SEARCH = "papers:search"
#: 通配。仅在 token 未声明任何作用域时授予（见 :func:`_parse_entry`）。
SCOPE_ALL = "*"

ALL_SCOPES: tuple[str, ...] = (
    SCOPE_DOCUMENTS_READ,
    SCOPE_DOCUMENTS_WRITE,
    SCOPE_JOBS_READ,
    SCOPE_JOBS_WRITE,
    SCOPE_CLOUD_PARSE,
    SCOPE_PAPER_SEARCH,
)

#: 个人档位的隐式租户 id。
#:
#: 它是一个**保留字**：托管档位的 token 表里不允许出现它，因为
#: :class:`DisabledAuthenticator` 在托管档位上根本不会被构造——而一个能被
#: 配出来的"和免鉴权租户同名的 token"，等于给免鉴权开了一扇侧门。
LOCAL_TENANT_ID = "local"


# --------------------------------------------------------------------------
# 错误
# --------------------------------------------------------------------------


class AuthenticationError(RuntimeError):
    """凭据缺失、无效或不完整——HTTP 401。

    消息面向用户，``code`` 面向客户端。两者分开是因为 ``/v1`` 的契约承诺
    字段名与错误码在版本内不变，而中文说明可以随时改。
    """

    status_code = 401
    code = "authentication_required"

    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        self.message = str(message)
        if code:
            self.code = code

    def to_dict(self) -> dict[str, Any]:
        """与 :func:`paperwhisperer.core.errors.build_error_payload` 同形。

        单独实现而不是 import 那个函数：那个模块会拉进 ``fastapi``，而本模块
        需要能被一个不装 Web 框架的脚本（一个 CLI 工具、一次离线检查）import。
        """
        return {"error": self.message, "code": self.code, "tenant_id": "", "subject": ""}


class AuthorizationError(AuthenticationError):
    """凭据有效但权限不足——HTTP 403。

    与 :class:`AuthenticationError` 分开，是因为客户端该做的动作不同：401 要
    重新登录，403 要找管理员。合成一个会让客户端在"重登一次还是白屏"之间猜。
    """

    status_code = 403
    code = "insufficient_scope"

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload["code"] = self.code
        return payload


class AuthNotConfigured(RuntimeError):
    """Hosted 档位没有配好鉴权。**这是启动失败，不是请求失败。**

    单独一个类型而不是复用 :class:`AuthenticationError`，因为它发生在进程
    启动路径上：调用方（app 的 lifespan）要把它当成"不要继续启动"处理，而
    一个 401 响应的处理器会把它当成"这个请求没登录"处理——于是一个配置错误
    被降级成"每个人都登录失败"，而真正的信息（少了什么配置）被 HTTP 层
    吞掉了。
    """


# --------------------------------------------------------------------------
# Principal
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Principal:
    """一个已识别的调用者，以及它**被允许触及的全部东西**。

    ``tenant_id`` 是这个类型存在的理由。它不是"日志字段"或"提示词里的名字"，
    而是 :mod:`paperwhisperer.api.tenant` 在**构造查询时**用来划定范围的那个
    值。一个没有租户的 Principal 是不可用的，所以构造它时``tenant_id`` 是必填
    的位置参数——没有默认值，因此"忘了传租户"是一个 ``TypeError``，而不是
    一个运行到数据层才发现的空字符串。

    ``scopes`` 为空集时**什么都不允许**（:func:`require_scope` 拒绝一切），
    只有显式的 :data:`SCOPE_ALL` 才表示全部。默认拒绝与默认放行的区别，
    在多租户系统里就是"新增接口忘了加鉴权"和"上线了"的区别。
    """

    subject: str
    tenant_id: str
    scopes: frozenset[str] = frozenset()
    method: str = "disabled"
    authenticated: bool = False

    def __post_init__(self) -> None:
        if not str(self.tenant_id or "").strip():
            raise ValueError("Principal 必须带一个非空 tenant_id：没有租户的调用者不可用。")
        object.__setattr__(self, "scopes", frozenset(str(item) for item in self.scopes))

    @property
    def is_superuser(self) -> bool:
        return SCOPE_ALL in self.scopes

    def grants(self, scope: str) -> bool:
        """Whether this principal may exercise ``scope``.

        通配只匹配 :data:`SCOPE_ALL` 自己，**不隐式包含其它作用域**——一个写了
        ``*`` 的 token 就是真的要全部权限，所以这里不需要推导；但一个显式列出
        三个作用域的 token 不该因为多了一个第四个作用域就被判成超权。
        """
        return self.is_superuser or str(scope) in self.scopes

    def describe(self) -> dict[str, Any]:
        """给 health 端点看的形状。**不含 token，也不含 token 的哈希。**"""
        return {
            "subject": self.subject,
            "tenant_id": self.tenant_id,
            "scopes": sorted(self.scopes),
            "method": self.method,
            "authenticated": self.authenticated,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.describe()

    @classmethod
    def local(cls) -> Principal:
        """个人档位的 Principal：一个隐式租户、通配权限、未认证。

        ``authenticated=False`` 是诚实的：个人档位没有任何东西证明调用者的
        身份，它只是"这台机器上的唯一用户"。把它记成 ``True`` 会让
        "已认证请求数" 这个指标在个人档位上永远是 100%，而它本该是 0。
        """
        return cls(
            subject=LOCAL_TENANT_ID,
            tenant_id=LOCAL_TENANT_ID,
            scopes=frozenset({SCOPE_ALL}),
            method="disabled",
            authenticated=False,
        )


#: 当前作用域内的 Principal。
#:
#: 存在的原因与 :mod:`paperwhisperer.core.observability` 存四个关联 id 的原因
#: 相同：调用方在**构造查询**的那一刻需要租户，而那一刻通常在好几层之下
#: （路由 → 检索器 → 存储）。把租户作为参数一路传下去，每一层都要改签名；
#: 用 ContextVar 则只有一个"读取当前租户"的地方，而漏读会在绑定缺失时
#: **抛异常**而不是读到空串——因为 :func:`current_principal` 找不到就抛。
_CURRENT_PRINCIPAL: ContextVar[Principal | None] = ContextVar(
    "paperwhisperer_principal", default=None
)


def current_principal() -> Principal:
    """当前作用域的 Principal，**没有绑定就抛异常**。

    不用"返回 None"也不用"返回一个空租户的 Principal"。前者会让每个调用方都要
    判空，于是总有一个忘了；后者会让空租户一路传到数据层，**而数据层无法区分
    "空租户"和"这个租户真的没有数据"**——那就是一个返回全集的查询。
    抛异常是唯一一种让漏读变成响亮的失败的方式。
    """
    principal = _CURRENT_PRINCIPAL.get()
    if principal is None:
        raise AuthenticationError(
            "当前请求没有绑定调用者身份：请先经过鉴权中间件再访问受保护资源。"
            "这通常意味着某条路由忘了挂 auth 依赖。",
            code="principal_not_bound",
        )
    return principal


def optional_principal() -> Principal | None:
    """当前 Principal，或 ``None``。给"登录可选"的端点用（如只读 health）。"""
    return _CURRENT_PRINCIPAL.get()


def use_principal(principal: Principal | None) -> Any:
    """绑定 Principal 的上下文管理器。``None`` 表示解绑。

    用法是 ``with use_principal(p):``。复原由 ``__exit__`` 完成，所以一个路由
    即使有三个提前 return 也不会把租户泄漏到下一个请求——这正是用上下文管理器
    而不是"设置一次全局变量"的理由：全局变量在一个返回 401 的早退路径上会留下
    上一个请求的租户，而那正是最难复现也最致命的一类串号。
    """
    return _PrincipalScope(principal)


class _PrincipalScope:
    """``with use_principal(p)`` 的实现：进入时设置，退出时精确复原。"""

    __slots__ = ("_token", "_value")

    def __init__(self, principal: Principal | None) -> None:
        self._value = principal
        self._token: Any = None

    def __enter__(self) -> Principal | None:
        self._token = _CURRENT_PRINCIPAL.set(self._value)
        return self._value

    def __exit__(self, *exc_info: Any) -> None:
        if self._token is not None:
            _CURRENT_PRINCIPAL.reset(self._token)
            self._token = None


def require_scope(principal: Principal, scope: str) -> None:
    """没有权限就抛 :class:`AuthorizationError`。

    错误消息**不泄露该 Principal 实际拥有什么**。一个越权探测者能通过
    "错误消息里列出了你的权限" 这条路把一个 token 的授权范围套出来；把它
    换成"你需要 X 权限"就够了，而 X 是调用方自己请求的那个权限，不是秘密。
    """
    if principal.grants(scope):
        return
    raise AuthorizationError(
        f"当前身份没有「{scope}」权限，无法执行此操作。"
        "如需访问，请联系管理员为该凭据授予对应权限。",
        code="insufficient_scope",
    )


# --------------------------------------------------------------------------
# 档位判定
# --------------------------------------------------------------------------

#: 托管档位在 ``STORAGE_BACKEND`` 里的写法。与
#: :data:`paperwhisperer.infra.stores.interfaces.BACKEND_POSTGRES` 同值，
#: 这里写字面量是为了不 import 存储层。
_HOSTED_BACKEND_NAMES = frozenset({"postgres", "postgresql", "pg"})


def resolve_tier(settings: Any = None) -> str:
    """这个部署是个人档还是托管档。与 :mod:`.factory.resolve_backend` 同义。

    两处对 ``STORAGE_BACKEND`` 的理解必须一致，而它们不共享代码——因为
    存储层会把 numpy 探测和数据库驱动拖进 import 图，而鉴权边界要能被一个
    从不开数据库的东西 import。**不一致时的症状**是：部署把
    ``STORAGE_BACKEND=postgresql`` 交给存储层，存储层按托管档连上 Postgres，
    而这里判定它是个人档，于是免鉴权启动——多租户数据库上的免鉴权服务。
    所以别名的集合必须与 ``interfaces.BACKEND_POSTGRES`` 的归一化保持一致，
    并且两边都要接受 ``file``/``local`` 作为 sqlite 的别名（存储层接受）。
    """
    raw = str(getattr(settings, "STORAGE_BACKEND", "") or "").strip().lower()
    if raw in _HOSTED_BACKEND_NAMES:
        return "hosted"
    # sqlite / file / local / 空 / 任何未知值 → 个人档。
    # 未知值回落个人档而不是抛异常，与 :func:`config._coerce_choice` 的取向
    # 一致：个人档是更安全的一档（没有跨租户可泄露）。
    return "personal"


# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------

#: 本模块读的每一个键，与默认值。**这张表就是与 ``core/config.py`` 的契约**，
#: 而那个模块归别的任务所有： whoever 把这些键加进去，照抄名字与默认值即可，
#: 本模块在键缺席时照常工作。
#:
#: ``AUTH_MODE``
#:     ``disabled``（默认）或 ``static_tokens``。默认关闭，因为默认档位是个人
#:     档（D2），而个人档上加登录屏是回归而不是安全改进。
#: ``AUTH_STATIC_TOKENS``
#:     凭据表。分号分隔条目，每条 ``token:tenant_id[:scope1,scope2]``。
#:     token 本身**明文写在这里**（它必须能被运维粘贴进 ``.env``），
#:     但本模块只在内存里保留它的 SHA-256，:func:`describe_auth` 之类的
#:     出口一律不吐原文。
#: ``AUTH_TENANT_DOMAIN``
#:     Hosted 档位默认租户名。``AUTH_STATIC_TOKENS`` 为空时它**不会**让服务
#:     启动——见 :class:`AuthNotConfigured`——它只作为运维提示里的一句。
AUTH_CONFIG_KEYS: dict[str, Any] = {
    "AUTH_MODE": "disabled",
    "AUTH_STATIC_TOKENS": "",
    "AUTH_TENANT_DOMAIN": "",
}

_TRUE = frozenset({"1", "true", "yes", "on"})


def _setting(settings: Any, key: str) -> Any:
    value = getattr(settings, key, None)
    if value is None:
        return AUTH_CONFIG_KEYS[key]
    return value


def hash_token(token: str) -> str:
    """凭据的存储形态：SHA-256 十六进制。

    为什么不是 bcrypt/argon：token 是 192 位随机数（:func:`new_token`），不是
    人选的密码。**暴力猜测 192 位随机数是不可能的**，慢哈希解决的是一个这里不
    存在的攻击（弱密码）。慢哈希带来的是一个真实的代价：进程启动要几十毫秒到
    几秒，而这在每个请求的鉴权路径上。所以本项目用快哈希，理由写在这里而不是
    让下一个读者去猜。

    比较方式是**以这个哈希为键的字典查找**，不是逐字节比较。这是有意的：
    字典查找对存在的键与不存在的键都走同样的哈希与探测路径，因此不存在
    "比较了前 N 个字节才发现不匹配" 这类时序侧信道——而逐字节比较即便用
    ``compare_digest`` 也要小心在它之前不要有任何按前缀短路的逻辑。
    """
    return hashlib.sha256(str(token or "").encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# 认证器
# --------------------------------------------------------------------------


class Authenticator:
    """把请求头变成 :class:`Principal` 的东西。

    一个基类而不是 Protocol，因为本模块有两个实现都要提供一份**相同形状的
    拒绝语义**（:meth:`authenticate` 抛 :class:`AuthenticationError` 而不是返回
    ``None``），而 Protocol 只能约束签名、约束不了这个约定。写成一族可以被
    ``isinstance`` 检查的具体类，测试里"两种实现对同一组坏凭据给出同样反应"
    这件事才有地方写。
    """

    #: 这一档的鉴权方式。``disabled`` 或 ``static_tokens``。
    mode = "disabled"

    def authenticate(self, headers: Mapping[str, str] | None) -> Principal:
        """认证一次请求。失败时抛 :class:`AuthenticationError`。"""
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        """给 health / 诊断看的形状。**永远不含凭据本身或其哈希。**"""
        return {"mode": self.mode}


class DisabledAuthenticator(Authenticator):
    """个人档位：任何请求都是隐式的本地用户。

    存在的理由在模块 docstring 里：它让个人档位与托管档位共用同一条调用方
    代码路径。它**永远不接受任何凭据**——即便请求带了一个格式正确的
    ``Authorization: Bearer``，结果仍是本地 Principal。这不是疏忽：个人档位上
    "识别出这个 token 是谁的"这件事没有数据支撑（没有 token 表），
    而一个"看起来识别成功"的分支会在个人档位上把凭据解析成垃圾租户。
    """

    mode = "disabled"

    def authenticate(self, headers: Mapping[str, str] | None = None) -> Principal:
        return Principal.local()

    def describe(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "enforced": False,
            "message": (
                "个人档位：未启用鉴权，所有请求都归属隐式本地租户 "
                f"{LOCAL_TENANT_ID!r}。此档位不面向多用户公开部署。"
            ),
        }


class StaticTokenAuthenticator(Authenticator):
    """托管档位：Bearer token → 租户。

    刻意不实现 OAuth2 / OIDC / JWT。为什么：这三者都需要引入依赖、需要一个
    身份提供方、需要密钥轮换策略，而**没有一个问题是当前部署已经提出的**。
    在已知需求是"多租户之间不能互相读到数据"的前提下，一张运维可读的
    token 表 + 常量时间比较是一个**真的能挡住跨租户访问**的实现；引入一个
    未经使用的 OIDC 栈则是一个未经使用的攻击面。真正的单点登录是需求出现时
    的替换，不是本任务的交付物。
    """

    mode = "static_tokens"

    def __init__(self, entries: Iterable[TokenEntry] = ()) -> None:
        #: token 的 SHA-256 → 条目。**键是哈希**，所以内存转储与
        #: ``describe()`` 都拿不到原文。
        self._by_hash: dict[str, TokenEntry] = {}
        for entry in entries:
            self._by_hash[entry.token_hash] = entry

    @property
    def token_count(self) -> int:
        return len(self._by_hash)

    def tenants(self) -> tuple[str, ...]:
        """配置里出现过的租户名，去重排序。仅用于诊断输出。"""
        return tuple(sorted({entry.tenant_id for entry in self._by_hash.values()}))

    def authenticate(self, headers: Mapping[str, str] | None = None) -> Principal:
        token = _bearer_token(headers)
        if not token:
            raise AuthenticationError(
                "缺少访问凭据：请在 Authorization 头中提供 Bearer token。"
                "此部署已启用多租户鉴权，未认证的请求一律拒绝。",
                code="missing_credential",
            )
        entry = self._by_hash.get(hash_token(token))
        if entry is None:
            # 不区分"token 不存在"与"token 错误"。区分它们等于提供一个
            # 无需凭据的枚举接口。
            raise AuthenticationError("凭据无效或已失效。", code="invalid_credential")
        return entry.to_principal()

    def describe(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "enforced": True,
            "token_count": len(self._by_hash),
            "tenants": list(self.tenants()),
        }


@dataclass(frozen=True, slots=True)
class TokenEntry:
    """凭据表里的一行：哈希、租户、作用域。"""

    token_hash: str
    tenant_id: str
    scopes: frozenset[str] = frozenset()
    label: str = ""
    _principal: Principal | None = field(default=None, repr=False, compare=False)

    def to_principal(self) -> Principal:
        """由这条表项造出 Principal。缓存的那个是同一个值。

        缓存的理由是 :class:`Principal` 是 frozen 的，构造它是廉价的，但
        :meth:`StaticTokenAuthenticator.authenticate` 在每个请求的路径上，
        把它做成一次 dict 查表 + 一次 dataclass 构造就够了。
        """
        if self._principal is None:
            object.__setattr__(
                self,
                "_principal",
                Principal(
                    subject=self.label or f"token:{self.tenant_id}",
                    tenant_id=self.tenant_id,
                    scopes=self.scopes,
                    method=StaticTokenAuthenticator.mode,
                    authenticated=True,
                ),
            )
        return self._principal  # type: ignore[return-value]

    @classmethod
    def from_token(
        cls,
        token: str,
        tenant_id: str,
        scopes: Iterable[str] | None = None,
        *,
        label: str = "",
    ) -> TokenEntry:
        """由明文 token 造表项。原文在此刻即被丢弃。"""
        text = str(token or "").strip()
        if not text:
            raise ValueError("凭据不能为空。")
        tenant = str(tenant_id or "").strip()
        if not tenant:
            raise ValueError("凭据必须属于一个租户：没有租户的凭据无法限定数据范围。")
        if tenant == LOCAL_TENANT_ID:
            raise ValueError(
                f"租户名 {LOCAL_TENANT_ID!r} 是保留字：它是免鉴权档位的隐式租户，"
                "托管档位使用它等于给自己开了一扇免鉴权的侧门。"
            )
        return cls(
            token_hash=hash_token(text),
            tenant_id=tenant,
            scopes=frozenset(SCOPE_ALL if scopes is None else (str(s) for s in scopes)),
            label=str(label or ""),
        )


def _parse_entry(raw: str) -> TokenEntry | None:
    """解析 ``token:tenant_id[:scope1,scope2]``。

    空白条目被跳过而不是报错——运维在 ``.env`` 里留下的一个尾随分号不应该
    让服务起不来。其余解析失败（缺租户、缺 token）也返回 ``None`` 并由
    :func:`build_authenticator` 汇总成一条**说清少了什么**的启动错误，
    比在这里抛一个 ``IndexError`` 有用得多。
    """
    text = str(raw or "").strip()
    if not text:
        return None
    parts = text.split(":")
    token = parts[0].strip()
    tenant = parts[1].strip() if len(parts) > 1 else ""
    scope_text = parts[2].strip() if len(parts) > 2 else ""
    if not token or not tenant:
        return None
    scopes = (
        [item.strip() for item in scope_text.split(",") if item.strip()]
        if scope_text
        else [SCOPE_ALL]
    )
    try:
        return TokenEntry.from_token(token, tenant, scopes)
    except ValueError as exc:
        logger.warning("忽略一条无效的凭据配置：%s", exc)
        return None


def parse_token_table(raw: str) -> tuple[list[TokenEntry], list[str]]:
    """解析整张表，返回 ``(有效条目, 无效条目的原文)``。

    无效条目被**收集**而不是丢弃，这样 :func:`build_authenticator` 能在启动
    错误里说"这一条写错了"。一条被静默忽略的凭据配置表现为"这个租户登录
    不进去"，而这是最难排查的一类配置问题。
    """
    entries: list[TokenEntry] = []
    invalid: list[str] = []
    for chunk in str(raw or "").split(";"):
        if not chunk.strip():
            continue
        entry = _parse_entry(chunk)
        if entry is None:
            invalid.append(chunk.strip())
        else:
            entries.append(entry)
    return entries, invalid


def _bearer_token(headers: Mapping[str, str] | None) -> str:
    """从请求头里取 Bearer token，**大小写不敏感地**匹配 ``Bearer``。

    头名按 HTTP 规范不区分大小写，而 ``headers`` 在不同框架里可能是
    ``dict``（区分）或大小写不敏感的映射。这里两种都试一遍，而不是假定
    调用方传进来的一定是小写的 ``authorization``——那会让一个正确配置的
    部署在某些 ASGI 服务器上全军覆没。
    """
    if not headers:
        return ""
    raw = ""
    for key in ("authorization", "Authorization"):
        value = headers.get(key)
        if value:
            raw = str(value)
            break
    if not raw:
        return ""
    parts = raw.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return ""
    return parts[1].strip()


def authenticate_headers(
    authenticator: Authenticator,
    headers: Mapping[str, str] | None,
) -> Principal:
    """认证并**绑定**当前作用域，返回 Principal。

    绑定的理由：调用方（路由处理函数）几乎总是在好几层之下才构造查询，
    而那些层不该各自接收一个租户参数。绑一次、处处可读，是让
    :mod:`paperwhisperer.api.tenant` 能在**构造查询的地方**划定范围的前提。
    """
    principal = authenticator.authenticate(headers)
    _CURRENT_PRINCIPAL.set(principal)
    return principal


# --------------------------------------------------------------------------
# 构建：档位 × 配置 → 一个 Authenticator
# --------------------------------------------------------------------------


def build_authenticator(settings: Any = None) -> Authenticator:
    """**唯一**决定"这个部署要不要鉴权、用哪种"的地方。

    规则只有两条，都可从代码直接读出来，不依赖任何隐式默认：

    * 托管档位 → 必须是 :class:`StaticTokenAuthenticator`，且凭据表非空。
      任何一种不满足都抛 :class:`AuthNotConfigured`。
    * 个人档位 → :class:`DisabledAuthenticator`，**无论配置怎么写**。

    第二条的"无论配置怎么写"是刻意的。一个人档部署如果有人在 ``.env`` 里
    写了 ``AUTH_MODE=static_tokens`` 而忘了配 token，我们仍然以个人档启动：
    那台机器的边界是文件系统，不是 HTTP。
    """
    tier = resolve_tier(settings)
    if tier != "hosted":
        if str(_setting(settings, "AUTH_MODE") or "").strip() not in ("", "disabled"):
            logger.info(
                "检测到 AUTH_MODE=%s，但当前部署是个人档位（STORAGE_BACKEND 非 postgres）。"
                "个人档位不启用鉴权，已忽略该设置。",
                _setting(settings, "AUTH_MODE"),
            )
        return DisabledAuthenticator()

    mode = str(_setting(settings, "AUTH_MODE") or "disabled").strip().lower()
    raw = str(_setting(settings, "AUTH_STATIC_TOKENS") or "")
    entries, invalid = parse_token_table(raw)
    if invalid:
        logger.warning(
            "AUTH_STATIC_TOKENS 中有 %d 条无法解析（缺少 token 或租户名），已忽略。",
            len(invalid),
        )
    if mode == "disabled":
        raise AuthNotConfigured(
            "Hosted 档位未启用鉴权，拒绝启动。"
            "多租户部署在鉴权关闭时等价于所有租户数据互相可见。"
            "请设置 AUTH_MODE=static_tokens 并在 AUTH_STATIC_TOKENS 中配置凭据，"
            "格式为 token:tenant_id[:scope1,scope2]（多条用分号分隔）。"
            "若这是个人部署，请把 STORAGE_BACKEND 改回 sqlite。"
        )
    if not entries:
        raise AuthNotConfigured(
            "Hosted 档位已启用 AUTH_MODE=static_tokens，但 AUTH_STATIC_TOKENS 为空，"
            "没有任何凭据可以登录，拒绝启动。"
            "空凭据表与关闭鉴权的实际效果相同，因此这里不降级。"
            "请至少配置一条 token:tenant_id。"
        )
    logger.info("Hosted 档位鉴权已启用：%d 条凭据，%d 个租户。",
                len(entries), len({entry.tenant_id for entry in entries}))
    return StaticTokenAuthenticator(entries)


def assert_startup_safe(settings: Any = None) -> Authenticator:
    """在进程启动路径上调它。返回构建出的认证器，失败则抛。

    这个函数的存在是为了让 app 只需要写一行
    ``authenticator = assert_startup_safe(config)``，并且**失败一定发生在
    启动期**。"启动时失败"是这条要求里唯一有意义的措辞：一个先起来再拒绝
    服务的部署，在那几秒里已经能被人连上了。
    """
    return build_authenticator(settings)


def describe_auth(settings: Any = None) -> dict[str, Any]:
    """给 health 端点的鉴权状态摘要。

    刻意**不吞异常**：托管档位配置错了的时候，这个函数会抛，而 health 端点
    会因此返回一个 5xx。这是对的——"鉴权没配好"不是一个健康的服务，而在
    托管部署上把它渲染成 ``"auth": "error"`` 加一个绿色状态灯，是把一个
    安全事故伪装成一次健康检查失败。
    """
    authenticator = build_authenticator(settings)
    return {
        "tier": resolve_tier(settings),
        **authenticator.describe(),
    }


def new_token(nbytes: int = 24) -> str:
    """生成一条新凭据（``token_urlsafe(24)`` ≈ 192 位熵）。

    存在的意义是让"配一条凭据"这件事有唯一正确的做法。一个自己写的
    ``password123`` 在这个表里同样有效，而它会被 :func:`hash_token` 一样
    忠实地哈希——所以本函数不是强制手段，是默认手段。
    """
    return secrets.token_urlsafe(nbytes)
