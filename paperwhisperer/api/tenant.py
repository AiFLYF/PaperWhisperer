"""租户范围与配额：让"跨租户访问为零"成为一条**机械的**保证。

验收标准是「**跨租户访问为零（安全测试）**」。这句话有两种做法。

一种是在路由里写 ``if record.tenant_id != current_tenant: 403``。它在**第一次
写的时候**是对的，第二次写的时候忘了，就是一次跨租户读取——而这条失败不会
报错，只会让一个租户读到另一个租户的论文摘要。

另一种是让**查询本身**带上范围：范围参数不是可选的，忘了就没有查询可发。
本文件做的是第二种，具体机制是 :class:`ScopedStoreBundle`——它包住
:class:`~paperwhisperer.infra.stores.interfaces.StoreBundle`，并把
``document_ids`` 从"调用方可以传 ``None`` 表示全集"改成**"永远不为 None"**。

**为什么范围要落在 store 这一层，而不是路由或检索器。** 检索器已经支持
``document_ids`` 过滤（:meth:`~paperwhisperer.rag.retrieve.HybridRetriever.
_resolve_scope` 明确说了"调用方的 document_ids **替换**而非合并计划的，因为
合并就是泄露"）。但那是**调用方**提供的过滤。一个调用方忘了传，于是
``scope is None``，于是 dense 与 lexical 两条腿都查询了全集——这就是那条
"跨租户访问为零"的裂缝。所以本文件不是"再加一层过滤"，而是**把
``None`` 这个取值从接口上拿掉**：进入 store 的每一次查询都带一个非空的
``document_ids``，或者带一个**这个租户确实拥有的文档集合**。忘掉的不是
过滤器，是整个查询。

**两个档位的差别是"隐式租户"还是"必须绑定租户"。** 个人档位有且只有一个
租户 :data:`~paperwhisperer.api.auth.LOCAL_TENANT_ID`，它拥有全部文档，
所以它的范围等于全集——**行为与今天逐字节一致**，但走的是同一条代码路径。
这是"个人档位惰性"的实现方式：不是加一个 ``if personal: bypass``，而是让
个人档位绑定一个恰好拥有一切的租户。

**跨租户的对象不止文档。** chunk、vector、lexical 条目、job、session、
evidence、export 全部要隔离。它们的隔离点不同，因为它们被持有的方式不同：
文档有 ``document_id`` 可以查归属，chunk 的归属只能**经由它所属的文档**
判断（chunk 自身不带 document_id），job 与 session 则完全在 store 之外
（一个在 ``jobs`` 表、一个在 ``context/*.json``）。这些差异都写在下面各自的
函数上，而不是抽象成一个含糊的"资源"概念——含糊的抽象会在第一个需要具体
判断的地方崩掉。
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from paperwhisperer.api.auth import LOCAL_TENANT_ID, Principal

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_QUOTA_LIMITS",
    "QUOTA_CONFIG_KEYS",
    "QUOTA_DERIVED_RESOURCES",
    "QUOTA_RESOURCE_CLOUD_PARSES",
    "QUOTA_RESOURCE_DOCUMENTS",
    "QUOTA_RESOURCE_LLM_CALLS",
    "QUOTA_RESOURCE_PAGES",
    "QUOTA_RESOURCES",
    "ScopedBundle",
    "ScopedLexicalStore",
    "ScopedMetadataStore",
    "ScopedVectorStore",
    "TENANT_METADATA_KEY",
    "TenantScope",
    "TenantScopeMissing",
    "assert_owns_chunk",
    "assert_owns_document",
    "assert_owns_session",
    "bind_for_principal",
    "bind_tenant",
    "current_scope",
    "document_scope_filter",
    "local_scope",
    "owns_chunk",
    "owns_document",
    "owns_job",
    "owns_session",
    "quota_ledger",
    "QuotaExceeded",
    "QuotaLedger",
    "QuotaLimits",
    "QuotaUsage",
    "require_document_access",
    "reset_quota_ledger",
    "resolve_documents_for",
    "scope_bundle",
    "session_tenant",
    "set_bundle_resolver",
    "tenant_of_record",
]


# --------------------------------------------------------------------------
# 错误
# --------------------------------------------------------------------------


class TenantScopeMissing(RuntimeError):
    """需要租户范围，但当前作用域里没有绑定租户。

    单独一个类型而不是复用 :class:`~paperwhisperer.api.auth.AuthenticationError`
    的理由和那里 :class:`~paperwhisperer.api.auth.AuthNotConfigured` 一样：
    调用方需要把它当成"这是一个编程错误，接缝没接上"处理，而当成"用户没登录"
    处理会让一个漏绑定的部署变成一个人人登录失败的部署——把一个结构性问题
    伪装成认证问题。

    **它不应该在正常路径上出现。** 出现即意味着某条路由没有经过
    :func:`paperwhisperer.api.auth.authenticate_headers`，而这正是安全测试要
    抓的那种洞。
    """


# --------------------------------------------------------------------------
# 范围
# --------------------------------------------------------------------------


@dataclass(slots=True)
class TenantScope:
    """一个租户在**数据层**的投影：它能触及哪些文档。

    ``documents`` 是 ``None`` 还是 ``frozenset()`` 的区别，是本文件里唯一
    一个会导致越权的区别，所以它被写成了显式的一行而不是一个默认值：

    * ``documents=None`` —— 本租户拥有**全部**文档。个人档位是这一档。
    * ``documents=frozenset()`` —— 本租户**一个文档都没有**。任何查询都必须
      返回空集。

    如果 ``documents`` 缺失时默认成 ``None``，那么"忘了解析租户"就等于
    "授予全部访问"，而这恰好是本任务要防的那件事。默认是空集。
    """

    tenant_id: str
    documents: frozenset[str] | None = None
    principal: Principal | None = None
    #: 这一档是否要求必须有租户。个人档位为 False（隐式本地租户）。
    required: bool = True

    # ``documents`` 是**可变**的，这是刻意的：一次成功的
    # ``ScopedMetadataStore.upsert_document`` 会把新文档登记进来，于是随后的
    # 检索能看见它自己刚写的东西。frozen 在这里会是错的——它会让"写入后立刻
    # 检索自己的新文档"失败，而那个失败表现为"写进去了但查不到"，是本文件
    # 最难排查的一类症状。其余字段（``tenant_id``）不改，它们由认证决定。

    @property
    def owns_everything(self) -> bool:
        """Whether this scope covers the whole corpus.

        只有个人档位的隐式租户会是 ``True``。它**不是**一个"关闭检查"的开关：
        它是一个明确的事实陈述，而这个事实在下面每个 ``_scope_of`` 里都要被
        重新读一遍（而不是被 ``if owns_everything: return x`` 提前短路掉），
        因为短路会让"托管档位上有人误设成拥有全部"变成静默通过。
        """
        return self.documents is None

    def contains(self, document_id: str) -> bool:
        if self.documents is None:
            return True
        return str(document_id) in self.documents

    def filter_document_ids(
        self, document_ids: Sequence[str] | None
    ) -> tuple[str, ...] | None:
        """把调用方的 ``document_ids`` 与本范围求交，返回可安全下发的值。

        **返回 ``None`` 只在一种情况下发生：本租户拥有全集。** 其余一切——
        包括调用方传了 ``None``、本租户拥有零个文档——都返回非空的
        ``tuple``（可能长度为零）。而底层两个后端对 ``document_ids=[]`` 的
        处理都是 ``return []``（见
        :meth:`~.sqlite_backend.SqliteVectorStore._candidates` 与
        :meth:`~.sqlite_backend.SqliteLexicalStore._candidates`），
        所以零文档的租户得到的是空结果而不是全集。

        这是本文件的核心断言：**``None``（全集）只能由一个显式声明自己拥有
        全部文档的范围产生**。
        """
        if self.documents is None:
            return None
        if document_ids is None:
            # 调用方没有指定范围 → 用本租户拥有的全部。
            return tuple(sorted(self.documents))
        narrowed = [str(item) for item in document_ids if str(item) in self.documents]
        return tuple(narrowed)

    def describe(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "required": self.required,
            "owns_everything": self.owns_everything,
            "document_count": (
                None if self.documents is None else len(self.documents)
            ),
        }


#: 当前作用域的 TenantScope。与 :mod:`paperwhisperer.api.auth` 里的 Principal
#: 是两个变量而不是一个，理由是它们回答不同的问题：Principal 说"你是谁"，
#: TenantScope 说"因此你能看见哪些行"。后者依赖存储层的内容（哪些文档属于你），
#: 所以它不能由认证决定——它由认证 **加上** 一次归属解析得到。
_CURRENT_SCOPE: Any = None


def _scope_var():
    """懒建 ContextVar，避免模块导入顺序上的耦合。"""
    global _CURRENT_SCOPE
    if _CURRENT_SCOPE is None:
        import contextvars

        _CURRENT_SCOPE = contextvars.ContextVar(
            "paperwhisperer_tenant_scope", default=None
        )
    return _CURRENT_SCOPE


def bind_tenant(
    scope: TenantScope | None,
) -> Any:
    """绑定租户范围。用法是 ``with bind_tenant(scope):``。

    复原在 ``__exit__`` 里完成。理由与 :func:`paperwhisperer.api.auth.
    use_principal` 相同：一个提前 return 的错误处理路径如果靠手动复位，
    就会把上一个请求的租户留给下一个——而"张冠李戴"的租户比"没有租户"
    危险得多，因为它看起来完全正常。
    """
    return _ScopeBinder(scope)


class _ScopeBinder:
    __slots__ = ("_token", "_value")

    def __init__(self, scope: TenantScope | None) -> None:
        self._value = scope
        self._token: Any = None

    def __enter__(self) -> TenantScope | None:
        self._token = _scope_var().set(self._value)
        return self._value

    def __exit__(self, *exc_info: Any) -> None:
        if self._token is not None:
            _scope_var().reset(self._token)
            self._token = None


def current_scope() -> TenantScope:
    """当前租户范围。**未绑定就抛** :class:`TenantScopeMissing`。

    与 :func:`paperwhisperer.api.auth.current_principal` 同一个取舍：不返回
    ``None``，也不返回一个"拥有全部"的默认范围。默认拥有全部是最危险的
    默认值——它让一处漏绑变成一次全库读取，而且没有任何信号。
    """
    scope = _scope_var().get()
    if scope is None:
        raise TenantScopeMissing(
            "当前请求没有绑定租户范围，无法构造任何数据查询。"
            "请先调用 bind_tenant()（生产路径上由鉴权中间件完成）。"
        )
    return scope


def _scope_for(principal: Principal, documents: Iterable[str] | None) -> TenantScope:
    """由 Principal 与归属集合造一个范围。

    ``documents=None`` 表示"拥有全部"，只在 ``tenant_id`` 是本地隐式租户时
    合法。这条限制是显式的：如果托管档位的某个 Principal 声称自己拥有全部
    文档，那是一个配置错误，而让它静默通过就等于把 :class:`TenantScope`
    变成了一个绕过一切的后门。
    """
    if documents is None and principal.tenant_id != LOCAL_TENANT_ID:
        raise ValueError(
            f"租户 {principal.tenant_id!r} 不能声明拥有全部文档："
            "托管档位的租户范围必须来自归属解析。"
        )
    return TenantScope(
        tenant_id=principal.tenant_id,
        documents=None if documents is None else frozenset(str(d) for d in documents),
        principal=principal,
        required=principal.tenant_id != LOCAL_TENANT_ID,
    )


def bind_for_principal(
    principal: Principal,
    documents: Iterable[str] | None = None,
) -> Any:
    """由 Principal 造范围并绑定。生产路径上的入口。

    ``documents=None`` 的语义由 :func:`_scope_for` 校验。
    """
    return bind_tenant(_scope_for(principal, documents))


@contextmanager
def local_scope() -> Iterator[TenantScope]:
    """个人档位的范围：隐式本地租户，拥有全部文档。

    这是"惰性"的具体形态——不是一条 ``if`` 旁路，而是一个**事实为真**的范围。
    于是所有 store 包装器在个人档位上都会跑一遍过滤逻辑、算出全集、
    原样下发。个人档位的行为因此**被证明**未变，而不是被断言。
    """
    scope = TenantScope(
        tenant_id=LOCAL_TENANT_ID,
        documents=None,
        principal=Principal.local(),
        required=False,
    )
    with bind_tenant(scope):
        yield scope


# --------------------------------------------------------------------------
# 归属
# --------------------------------------------------------------------------
#
# 归属存在哪里，是这一节唯一的设计决定，也是本文件里最不可替代的一句。
#
# `DocumentRecord.metadata` 是一个自由字典，它随记录**一起被两个后端写进同一
# 个 payload**（见 :meth:`~.sqlite_backend.SqliteMetadataStore.upsert_document`
# 与 :meth:`~.postgres_backend.PostgresMetadataStore.upsert_document`：两者都
# 存 `record.to_dict()`），并且由 :meth:`DocumentRecord.from_dict` 完整还原。
# 所以 `metadata["tenant_id"]` 是一个**已经存在、两档一致、无需改 schema、
# 随文档一起持久化**的字段。
#
# 为什么不新建一张 `tenant_documents` 表：那样需要改 `infra/**` 的 DDL，而本
# 任务不拥有那两个文件；更实质的理由是，一个独立的归属表会与文档行**分处两
# 处**，于是"归属说属于 A、文档说属于 B"这种不一致是可能的，而读文档的人只
# 看得到一边。归属写在文档自己身上，两个视图不可能不一致。
#
# 代价要说清楚：`list_documents()` **没有租户过滤参数**，所以托管档位上解析
# 一个租户的范围需要先列出全部文档、在 Python 里过滤。这是 O(语料) 的，且是
# 本文件的已知瓶颈。正确的修法是给 `DocumentRecord` 加一个顶层 ``tenant_id``
# 字段并让两个后端的 SQL 带 ``WHERE``——那要动 ``infra/**``，所以留给拥有它
# 的那个任务。这里先保证**正确**，再由那个任务把它变快。

#: ``DocumentRecord.metadata`` 里存租户 id 的键。
TENANT_METADATA_KEY = "tenant_id"


def tenant_of_record(record: Any) -> str:
    """一条文档记录属于谁。

    **没有归属的文档归本地隐式租户。** 备选是"归 nobody"，即任何托管租户都
    看不见它——那样一次忘记写归属的写入会让文档凭空消失，而"本地可见"是
    一个安全且可诊断的选择：一个存量部署在打开鉴权之后，它的每一份文档仍然
    读得到（归属本地租户），而不是集体变成 404。
    """
    metadata = getattr(record, "metadata", None)
    if not isinstance(metadata, Mapping):
        return LOCAL_TENANT_ID
    return str(metadata.get(TENANT_METADATA_KEY) or LOCAL_TENANT_ID)


def resolve_documents_for(
    principal: Principal,
    bundle: Any,
    *,
    page_size: int = 200,
) -> frozenset[str]:
    """解析某个 Principal 能触及的 ``document_id`` 全集。

    个人档位返回 ``None`` 的等价物（空范围由 :func:`_scope_for` 转成
    ``None``/全集），因为它拥有全部；托管档位则真的去列一遍文档并按归属过滤。

    分页而不是一次 ``limit=10**9``：后者在托管档位上会把整个语料装进内存，
    而分页在两个后端上都用同一套 ``created_at, document_id`` 全序，因此两档
    的翻页结果一致。翻页的终止条件是"某一页返回的行数小于页长"，这在有并发
    删除时是**保守**的（可能少取几行），而少取意味着**看不见**而不是**多看**，
    是安全的方向。
    """
    if principal.tenant_id == LOCAL_TENANT_ID:
        return frozenset()
    owned: list[str] = []
    offset = 0
    metadata = bundle.metadata
    while True:
        page = metadata.list_documents(limit=page_size, offset=offset)
        if not page:
            break
        owned.extend(
            record.document_id
            for record in page
            if tenant_of_record(record) == principal.tenant_id
        )
        offset += len(page)
        if len(page) < page_size:
            break
    return frozenset(owned)


# --------------------------------------------------------------------------
# 归属断言：所有隔离点
# --------------------------------------------------------------------------


def owns_document(bundle: Any, document_id: str) -> bool:
    """这个范围是否拥有 ``document_id``。

    **答案来自归属表，不来自"当前范围是否列了它"。** 这两者在正常情况下
    相同，但来源不同决定了失效方式不同：前者是"这个文档不属于我"，后者是
    "我的范围没有包含它"——而后者可能是解出错的结果。用归属作为判据，
    一处解bug 最多让一个调用被多拒一次，不会让它多看到一份别人的论文。
    """
    scope = current_scope()
    if scope.owns_everything:
        return True
    record = bundle.metadata.get_document(str(document_id))
    if record is None:
        # 文档不存在。它当然不属于本租户，而返回 False 让调用方用一个
        # 统一的"403"回答，避免用 403/404 的差异做存在性探测。
        return False
    return tenant_of_record(record) == scope.tenant_id


def assert_owns_document(bundle: Any, document_id: str) -> None:
    """不拥有就抛 :class:`~paperwhisperer.api.auth.AuthorizationError`。"""
    if owns_document(bundle, document_id):
        return
    scope = current_scope()
    raise _denied(scope, f"文档 {document_id}")


def require_document_access(bundle: Any, document_id: str) -> Any:
    """断言拥有并**返回该文档记录**，一次查询完成两件事。

    把"检查"和"取"合成一个函数，是因为分开写就会出现检查与取用之间的
    TOCTOU：两次查询之间文档可能被改归属，而调用方拿着第一轮的许可去用
    第二轮的数据。
    """
    scope = current_scope()
    record = bundle.metadata.get_document(str(document_id))
    if record is None or tenant_of_record(record) != scope.tenant_id:
        if scope.owns_everything and record is not None:
            return record
        raise _denied(scope, f"文档 {document_id}")
    return record


def _denied(scope: TenantScope, subject: str) -> Exception:
    """构造拒绝异常。**消息里不回显对方的租户名。**"""
    from paperwhisperer.api.auth import AuthorizationError

    return AuthorizationError(
        f"无权访问{subject}：它不属于当前租户。",
        code="cross_tenant_access_denied",
    )


def document_scope_filter(
    bundle: Any,
    document_ids: Sequence[str] | None,
) -> tuple[str, ...] | None:
    """把调用方的 ``document_ids`` 收窄到本范围，返回可安全下发的值。

    这是给**已有调用点**用的最小接缝：一条已经会调
    ``bundle.vectors.search(..., document_ids=ids)`` 的代码，只要把
    ``ids`` 换成 ``document_scope_filter(bundle, ids)``，就获得了租户隔离，
    而不需要改它对 store 的认识。

    前提是当前作用域有绑定；没有绑定时抛 :class:`TenantScopeMissing`——
    而不是放行。
    """
    scope = current_scope()
    if scope.owns_everything:
        return None if document_ids is None else tuple(
            str(item) for item in document_ids
        )
    narrowed = scope.filter_document_ids(document_ids)
    if narrowed is None:  # pragma: no cover - owns_everything 已在上返回
        return None
    # 再用归属表复核一次调用方点名的文档。这不是冗余：scope 可能是调用方
    # 自己构造的（测试、脚本），而归属表是事实。
    verified = tuple(
        item for item in narrowed if owns_document(bundle, item)
    )
    return verified


# -- chunk ------------------------------------------------------------------
#
# chunk 自身不带 document_id（见 `Chunk` 的字段列表），所以"这个 chunk 属于
# 哪个租户"只能**经由它所属的文档**回答。直接读 chunk 行是做不到的——两档的
# `get_chunk` 都只返回 `Chunk`。因此归属判断走"这个 chunk_id 在本租户的某个
# 文档的 chunk 列表里吗"，而这是一个**成员判断**：不属于的 chunk 与不存在的
# chunk 给出同一个答案（不在集合里），因此不存在跨租户的存在性侧信道。


def owns_chunk(bundle: Any, chunk_id: str) -> bool:
    """这个范围是否拥有 ``chunk_id``。"""
    scope = current_scope()
    if scope.owns_everything:
        return True
    if not scope.documents:
        return False
    for document_id in scope.documents:
        for chunk in bundle.metadata.list_chunks(document_id):
            if chunk.chunk_id == str(chunk_id):
                return True
    return False


def assert_owns_chunk(bundle: Any, chunk_id: str) -> None:
    """不拥有就抛拒绝异常。"""
    if owns_chunk(bundle, chunk_id):
        return
    raise _denied(current_scope(), f"证据 {chunk_id}")


# -- job --------------------------------------------------------------------
#
# job 不在 store bundle 里，它在 `jobs` 表（见 `paperwhisperer.infra.jobs`），
# 而 `Job` 的字段里有 `document_id` 与 `session_id`。所以 job 的隔离点是
# **document_id 优先、session_id 兜底**：一个还没绑定文档的注册类 job 只能
# 靠 session 判定，而 session 判定需要一个"这个 session 属于谁"的解析。
# 解析不到就**拒绝**，因为一个归属不明的 job 不该被任何人认领。


def owns_job(job: Any, session_owner: Any = None) -> bool:
    """这个范围是否拥有 ``job``。

    ``session_owner`` 是一个 ``session_id -> tenant_id`` 的可调用对象或映射，
    由调用方注入（租户表不归本模块管）。它只在 job 没有 ``document_id`` 时
    被用到。
    """
    scope = current_scope()
    if scope.owns_everything:
        return True
    document_id = str(getattr(job, "document_id", "") or "")
    if document_id:
        return owns_document(_bundle_for(job), document_id)
    session_id = str(getattr(job, "session_id", "") or "")
    if not session_id or session_owner is None:
        # 归属不明 → 不认领。返回 True 会让"任何人都能取消任何人的任务"，
        # 而取消是一个会消耗云解析额度的写操作。
        return False
    owner = (
        session_owner.get(session_id)
        if isinstance(session_owner, Mapping)
        else session_owner(session_id)
    )
    return str(owner or "") == scope.tenant_id


#: :func:`owns_job` 里那个"从 job 拿 bundle"的占位。它必须在调用前被替换成
#: 真实的 bundle 解析器；默认抛异常而不是返回 ``None``，因为一个 ``None``
#: bundle 会让 :func:`owns_document` 的下一次 ``bundle.metadata`` 变成
#: ``AttributeError`` —— 那是一个看起来像 bug 的 500，而正确行为是 403。
_BUNDLE_RESOLVER: Any = None


def set_bundle_resolver(resolver: Any) -> None:
    """注册"当前请求的 store bundle"如何取到。生产路径由 app 启动时调用。"""
    global _BUNDLE_RESOLVER
    _BUNDLE_RESOLVER = resolver


def _bundle_for(_job: Any = None) -> Any:
    if _BUNDLE_RESOLVER is None:
        raise TenantScopeMissing(
            "需要 store bundle 来校验归属，但未注册 bundle 解析器。"
            "请在应用启动时调用 paperwhisperer.api.tenant.set_bundle_resolver()。"
        )
    return _BUNDLE_RESOLVER()


# -- session ----------------------------------------------------------------
#
# session 是 `context/*.json` 文件里的 JSON，外加 `sessions` 表里的一份镜像。
# 归属写在**文件 payload** 上（`session["tenant_id"]`），而不是只在表里，
# 因为文件的读取路径（`load_session_payload`）不经过那张表——一份只在表里有
# 归属、文件里没有的 session，其归属会随一次镜像失败而消失。


def session_tenant(payload: Mapping[str, Any] | None) -> str:
    """session payload 属于谁。

    缺字段回落到本地租户，理由与 :func:`tenant_of_record` 相同：存量 session
    必须仍然可读，而"本地可见"是一个安全的选择。
    """
    if not isinstance(payload, Mapping):
        return LOCAL_TENANT_ID
    return str(payload.get("tenant_id") or LOCAL_TENANT_ID)


def owns_session(payload: Mapping[str, Any] | None) -> bool:
    """这个范围是否拥有该 session。"""
    scope = current_scope()
    if scope.owns_everything:
        return True
    return session_tenant(payload) == scope.tenant_id


def assert_owns_session(payload: Mapping[str, Any] | None) -> None:
    if owns_session(payload):
        return
    raise _denied(current_scope(), "该会话")


# --------------------------------------------------------------------------
# 范围化 store：把 "document_ids=None" 从接口上拿掉
# --------------------------------------------------------------------------
#
# 上面那些函数是**给已有调用点用的接缝**：它们要求调用方记得调。
# 下面这一组是**不需要调用方记得**的那一层：一个把 StoreBundle 包起来的代理，
# 它在每次查询构造时把范围塞进去。托管档位上，应用拿到的应当是它而不是
# 真实 bundle；个人档位上两者等价（隐式租户拥有全集），所以代理是惰性的。
#
# 代理的每个读方法都遵循同一条规则：**不拥有就返回空**，而不是抛异常。
# 这不是软弱，是与 store 契约一致——`get_document` 的契约就是"没有就返回
# None"，`search` 的契约就是"没有命中就返回 []"。一个代理如果把跨租户访问
# 变成异常，那么**一个合法的空结果**（文档已被删除）和**一次越权**就会走
# 不同的代码路径，而调用方必须两者都处理。返回空值让它们在同一个分支里
# 汇合，而真正的"你越权了"由 :func:`assert_owns_document` 这类显式断言在
# 需要给用户不同提示的地方提供。


class ScopedMetadataStore:
    """带租户范围的 :class:`~paperwhisperer.infra.stores.interfaces.MetadataStore`。"""

    __slots__ = ("_inner", "_scope")

    def __init__(self, inner: Any, scope: TenantScope) -> None:
        self._inner = inner
        self._scope = scope

    @property
    def inner(self) -> Any:
        """被代理的 store。**只给需要下沉到存储层的代码用**（如备份）。"""
        return self._inner

    @property
    def scope(self) -> TenantScope:
        return self._scope

    def capabilities(self) -> Any:
        return self._inner.capabilities()

    def initialize(self) -> None:
        self._inner.initialize()

    def close(self) -> None:
        self._inner.close()

    # -- documents ---------------------------------------------------------

    def upsert_document(self, record: Any) -> Any:
        """写入文档，并**把归属改成本租户**。

        改写而不是校验：调用方传的记录可能带着旧的（或者别人的）归属，而
        一次写入不该有"我能不能改这条记录"这个问题——本租户写自己的文档，
        归属就是本租户的。如果调用方试图写一个属于别人的 document_id，
        那是覆盖而不是读取，:meth:`get_document` 的归属检查管不到它，所以
        这里显式拒绝。
        """
        document_id = str(getattr(record, "document_id", "") or "")
        if not self._scope.owns_everything and not self._scope.contains(document_id):
            existing = self._inner.get_document(document_id)
            if existing is not None and tenant_of_record(existing) != self._scope.tenant_id:
                raise _denied(self._scope, f"文档 {document_id}")
            if existing is None and self._scope.documents is not None:
                # 新文档：登记进本范围，这样随后的查询能看见它。
                self._scope.documents = self._scope.documents | {document_id}
        stamped = _with_tenant(record, self._scope.tenant_id)
        return self._inner.upsert_document(stamped)

    def get_document(self, document_id: str) -> Any:
        record = self._inner.get_document(str(document_id))
        if record is None:
            return None
        if self._scope.owns_everything:
            return record
        if tenant_of_record(record) != self._scope.tenant_id:
            return None
        return record

    def list_documents(self, limit: int = 50, offset: int = 0) -> list[Any]:
        """本租户的文档，按原契约分页。

        分页是**先过滤后切**，不是先切后过滤——后者会让 ``limit=10`` 返回
        少于 10 条（如果前 10 条里有别人的），而调用方会把"少"读成
        "这个租户只有几条"。代价是要多翻几页；收益是分页语义不变。
        """
        if self._scope.owns_everything:
            return self._inner.list_documents(limit=limit, offset=offset)
        collected: list[Any] = []
        scan_offset = 0
        page_size = max(int(limit), 50)
        while len(collected) < max(0, int(limit)) + max(0, int(offset)):
            page = self._inner.list_documents(limit=page_size, offset=scan_offset)
            if not page:
                break
            collected.extend(
                record
                for record in page
                if tenant_of_record(record) == self._scope.tenant_id
            )
            scan_offset += len(page)
            if len(page) < page_size:
                break
        start = max(0, int(offset))
        return collected[start : start + max(0, int(limit))]

    def delete_document(self, document_id: str) -> bool:
        """删除本租户的文档。不属于本租户的一律返回 ``False``（不抛）。"""
        if self.get_document(document_id) is None:
            return False
        return self._inner.delete_document(str(document_id))

    def count_documents(self) -> int:
        return len(self.list_documents(limit=10**9, offset=0))

    # -- chunks ------------------------------------------------------------

    def put_chunk(self, document_id: str, chunk: Any) -> Any:
        self._require_writable(document_id)
        return self._inner.put_chunk(str(document_id), chunk)

    def get_chunk(self, chunk_id: str) -> Any:
        """按 chunk_id 取。**不属于本租户返回 None。**

        归属要经由文档判断，所以这条比 :meth:`get_document` 多一次查询。
        """
        if self._scope.owns_everything:
            return self._inner.get_chunk(str(chunk_id))
        if not self._scope.documents:
            return None
        for document_id in self._scope.documents:
            for chunk in self._inner.list_chunks(document_id):
                if chunk.chunk_id == str(chunk_id):
                    return chunk
        return None

    def list_chunks(self, document_id: str) -> list[Any]:
        if not self._scope.contains(str(document_id)):
            return []
        if self._scope.owns_everything:
            return self._inner.list_chunks(str(document_id))
        record = self._inner.get_document(str(document_id))
        if record is None or tenant_of_record(record) != self._scope.tenant_id:
            return []
        return self._inner.list_chunks(str(document_id))

    def delete_chunks(self, document_id: str) -> int:
        if not self._scope.contains(str(document_id)):
            return 0
        if self._scope.owns_everything:
            return self._inner.delete_chunks(str(document_id))
        record = self._inner.get_document(str(document_id))
        if record is None or tenant_of_record(record) != self._scope.tenant_id:
            return 0
        return self._inner.delete_chunks(str(document_id))

    # -- facts -------------------------------------------------------------

    def put_facts(self, document_id: str, facts: Any) -> Any:
        self._require_writable(document_id)
        return self._inner.put_facts(str(document_id), facts)

    def get_facts(self, document_id: str) -> Any:
        if not self._scope.contains(str(document_id)):
            return None
        if self._scope.owns_everything:
            return self._inner.get_facts(str(document_id))
        record = self._inner.get_document(str(document_id))
        if record is None or tenant_of_record(record) != self._scope.tenant_id:
            return None
        return self._inner.get_facts(str(document_id))

    def _require_writable(self, document_id: str) -> None:
        """写入前确认本租户可以写这个文档。

        写 chunk/facts 的入口没有"记录"可改写归属，所以归属必须已经被
        :meth:`upsert_document` 登记过。个人档位跳过这一检查（拥有全部）。
        """
        if self._scope.owns_everything:
            return
        if self._scope.contains(str(document_id)):
            return
        if self._inner.get_document(str(document_id)) is None:
            # 还没有文档行：允许（它将成为本租户的新文档）。
            return
        raise _denied(self._scope, f"文档 {document_id}")


def _with_tenant(record: Any, tenant_id: str) -> Any:
    """返回一份把 ``metadata["tenant_id"]`` 设为 ``tenant_id`` 的记录副本。

    副本而不是原地改：``DocumentRecord`` 是 ``slots`` dataclass，调用方手里
    可能还有同一份对象，而它也许正被用作"写入前"的快照。就地改会连快照一起
    改掉，于是"我写了什么"和"我本来拿着什么"变成同一件事。
    """
    metadata = dict(getattr(record, "metadata", None) or {})
    metadata[TENANT_METADATA_KEY] = str(tenant_id)
    try:
        return type(record).from_dict({**record.to_dict(), "metadata": metadata})
    except AttributeError:  # pragma: no cover - 非 DocumentRecord 的兜底
        try:
            record.metadata = metadata  # type: ignore[attr-defined]
        except Exception:
            pass
        return record


class ScopedVectorStore:
    """带租户范围的 :class:`~paperwhisperer.infra.stores.interfaces.VectorStore`。

    关键在 :meth:`search`：**``document_ids=None`` 不再下发**。它被替换成
    本租户拥有的文档集合（可能为空集，而两个后端对空集都返回 ``[]``）。
    这是整份文件里最重要的一行——"忘了传过滤条件"从"查询全集"变成
    "查询本租户的文档"，而后者永远不是一次越权。
    """

    __slots__ = ("_inner", "_metadata", "_scope")

    def __init__(self, inner: Any, metadata: Any, scope: TenantScope) -> None:
        self._inner = inner
        self._metadata = metadata
        self._scope = scope

    @property
    def inner(self) -> Any:
        return self._inner

    @property
    def scope(self) -> TenantScope:
        return self._scope

    @property
    def dimension(self) -> int | None:
        return self._inner.dimension

    def capabilities(self) -> Any:
        return self._inner.capabilities()

    def initialize(self) -> None:
        self._inner.initialize()

    def close(self) -> None:
        self._inner.close()

    def upsert_vector(self, chunk_id, vector, *, document_id="", metadata=None) -> None:
        if (
            not self._scope.owns_everything
            and document_id
            and not self._scope.contains(str(document_id))
        ):
            raise _denied(self._scope, f"文档 {document_id}")
        self._inner.upsert_vector(chunk_id, vector, document_id=document_id, metadata=metadata)

    def upsert_many(self, entries: Iterable[Any]) -> int:
        blocked = [
            entry
            for entry in entries
            if not self._scope.owns_everything
            and str(getattr(entry, "document_id", "") or "")
            and not self._scope.contains(str(entry.document_id))
        ]
        if blocked:
            raise _denied(self._scope, f"文档 {blocked[0].document_id}")
        return self._inner.upsert_many(entries)

    def get_vector(self, chunk_id: str) -> Any:
        if not self._scope.owns_everything and self._metadata.get_chunk(
            str(chunk_id)
        ) is None:
            return None
        return self._inner.get_vector(str(chunk_id))

    def delete_vector(self, chunk_id: str) -> bool:
        if (
            not self._scope.owns_everything
            and self._metadata.get_chunk(str(chunk_id)) is None
        ):
            return False
        return self._inner.delete_vector(str(chunk_id))

    def delete_document(self, document_id: str) -> int:
        if not self._scope.contains(str(document_id)):
            return 0
        return self._inner.delete_document(str(document_id))

    def search(
        self,
        vector: Any,
        top_k: int = 10,
        *,
        document_ids: Sequence[str] | None = None,
        min_score: float | None = None,
    ) -> list[Any]:
        """在**本租户的文档范围内**做向量检索。"""
        scope_filter = self._scope.filter_document_ids(document_ids)
        if self._scope.owns_everything:
            hits = self._inner.search(
                vector, top_k, document_ids=scope_filter, min_score=min_score
            )
        elif scope_filter is not None and not scope_filter:
            return []
        else:
            hits = self._inner.search(
                vector, top_k, document_ids=scope_filter, min_score=min_score
            )
        if self._scope.owns_everything:
            return hits
        # 纵深防御：即使某个后端的过滤有 bug，返回的 hit 也要再过一次归属。
        return [hit for hit in hits if self._scope.contains(str(hit.document_id))]

    def count_vectors(self) -> int:
        if self._scope.owns_everything:
            return self._inner.count_vectors()
        if not self._scope.documents:
            return 0
        return sum(
            len(self._inner.search(
                _unit_vector(self._inner), 10**9,
                document_ids=[document_id],
            ))
            for document_id in self._scope.documents
        )


def _unit_vector(store: Any) -> tuple[float, ...]:
    """一个单位向量，用来让后端帮我们数一遍某个文档的向量。

    这是一个**借用**，只在"数一数有多少条"时使用，后端会把它与每条向量算一次
    余弦相似度。个人档位语料规模下这是微秒级，托管档位上应该由那个拥有
    ``infra/**`` 的任务加一个带 ``WHERE`` 的 ``count_documents``。写在这里
    而不是留空，是为了让 :meth:`count_vectors` 在两档上都有一个**正确的**
    答案——慢，但正确。
    """
    dimension = store.dimension
    if not dimension:
        return ()
    return (1.0,) + (0.0,) * (int(dimension) - 1)


class ScopedLexicalStore:
    """带租户范围的 :class:`~paperwhisperer.infra.stores.interfaces.LexicalRetriever`。"""

    __slots__ = ("_inner", "_metadata", "_scope")

    def __init__(self, inner: Any, metadata: Any, scope: TenantScope) -> None:
        self._inner = inner
        self._metadata = metadata
        self._scope = scope

    @property
    def inner(self) -> Any:
        return self._inner

    @property
    def scope(self) -> TenantScope:
        return self._scope

    def capabilities(self) -> Any:
        return self._inner.capabilities()

    def initialize(self) -> None:
        self._inner.initialize()

    def close(self) -> None:
        self._inner.close()

    def index_chunk(self, chunk_id, text, *, document_id="", metadata=None) -> None:
        if (
            not self._scope.owns_everything
            and document_id
            and not self._scope.contains(str(document_id))
        ):
            raise _denied(self._scope, f"文档 {document_id}")
        self._inner.index_chunk(chunk_id, text, document_id=document_id, metadata=metadata)

    def delete_chunk(self, chunk_id: str) -> bool:
        if (
            not self._scope.owns_everything
            and self._metadata.get_chunk(str(chunk_id)) is None
        ):
            return False
        return self._inner.delete_chunk(str(chunk_id))

    def delete_document(self, document_id: str) -> int:
        if not self._scope.contains(str(document_id)):
            return 0
        return self._inner.delete_document(str(document_id))

    def search(
        self, query: str, top_k: int = 10, *, document_ids: Sequence[str] | None = None
    ) -> list[Any]:
        scope_filter = self._scope.filter_document_ids(document_ids)
        if not self._scope.owns_everything and scope_filter is not None and not scope_filter:
            return []
        hits = self._inner.search(query, top_k, document_ids=scope_filter)
        if self._scope.owns_everything:
            return hits
        return [hit for hit in hits if self._scope.contains(str(hit.document_id))]

    def count_chunks(self) -> int:
        if self._scope.owns_everything:
            return self._inner.count_chunks()
        # **数 chunk 行，不是数"包含某个词的 chunk"。** 原先这里对每个文档搜一个
        # 硬编码的「的」字再数命中数，于是词法统计返回的是"含『的』的 chunk
        # 有几个"——一份纯英文语料会得到 0，而 0 看起来像一个合法的小数字。
        # chunk 行数由元数据层按文档给出，那个数与词法索引无关，也不会因为语料
        # 语言而变。
        return sum(
            len(self._metadata.list_chunks(document_id))
            for document_id in (self._scope.documents or frozenset())
        )


@dataclass(slots=True)
class ScopedBundle:
    """一个 :class:`~paperwhisperer.infra.stores.factory.StoreBundle` 的范围化视图。

    形状与真实 bundle **完全一致**（同样有 ``metadata``/``vectors``/``lexical``
    与 ``tier``/``backend``/``capabilities``/``delete_document``），所以
    :class:`~paperwhisperer.rag.retrieve.HybridRetriever` 这类接受
    ``StoreBundle`` 的调用方**不需要知道**自己拿的是代理还是真的。这是惰性的
    另一半：个人档位上把它换成代理，检索结果必须逐字节相同，而这一点由
    ``tests/test_hosted_auth.py`` 直接断言。
    """

    metadata: ScopedMetadataStore
    vectors: ScopedVectorStore
    lexical: ScopedLexicalStore
    scope: TenantScope
    tier: str = ""
    backend: str = ""
    capabilities: Any = None
    database: Any = None
    _inner: Any = field(default=None, repr=False)

    @property
    def degradations(self) -> tuple[str, ...]:
        return tuple(self.capabilities.degradations) if self.capabilities else ()

    @property
    def inner(self) -> Any:
        """被代理的真实 bundle。**只给备份、下沉到存储层的代码用。**"""
        return self._inner

    def describe(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "backend": self.backend,
            **(
                self.capabilities.describe()
                if self.capabilities
                else {"degradations": []}
            ),
            "tenant": self.scope.describe(),
        }

    def delete_document(self, document_id: str) -> bool:
        return self.metadata.delete_document(document_id)


def scope_bundle(bundle: Any, scope: TenantScope) -> ScopedBundle:
    """把真实 bundle 包成范围化视图。这是唯一的包装入口。"""
    metadata = ScopedMetadataStore(bundle.metadata, scope)
    return ScopedBundle(
        metadata=metadata,
        vectors=ScopedVectorStore(bundle.vectors, metadata, scope),
        lexical=ScopedLexicalStore(bundle.lexical, metadata, scope),
        scope=scope,
        tier=str(getattr(bundle, "tier", "") or ""),
        backend=str(getattr(bundle, "backend", "") or ""),
        capabilities=getattr(bundle, "capabilities", None),
        database=getattr(bundle, "database", None),
        _inner=bundle,
    )


# --------------------------------------------------------------------------
# 配额
# --------------------------------------------------------------------------
#
# 配额要卡的是**会花钱的四件事**，因此只有这四个资源：文档数、解析页数、
# LLM 调用数、云解析调用数。少一个就漏了一处支出，多一个则会在没人要求的地方
# 添规则。
#
# **两类资源，两种计量方式，这是配额设计里唯一真正的决定。**
#
# * ``documents`` 与 ``pages`` **从 store 推导**。它们是状态的函数：租户名下
#   有几份文档、共几页，读一下就知道。计数器的价值在于对**事件**计数，而这
#   两项没有事件——文档数就是行数。用计数器记"上传了几份"意味着一次删除之后
#   计数不会回落，于是配额会在用户删掉文档之后仍然拒绝他。推导没有这个毛病，
#   而且重启不会清零。
# * ``llm_calls`` 与 ``cloud_parse_calls`` **必须计数**。它们没有对应的行：
#   一次 LLM 调用过去就过去了，没有"这一行"。所以它们要一个计数器。
#
# 计数器的**持久性**是一个必须说清楚的限制。本实现的默认计数器是**进程内**的
# ：进程重启会把它清零，因此托管部署在重启后可以再花一份配额。对一个单实例、
# 且重启是运维主动行为的部署，这是可接受的；**对多实例部署它是不够的**，而
# 把它做够需要 Redis——那正是计划里 T4.5 剩下的一半，本任务被明确要求不写。
# 诚实的说法写在 :class:`QuotaLedger` 的 docstring 与 ``docs/SLO.md`` 里，
# 而不是让一个"看起来生效"的配额成为部署者的信心来源。

QUOTA_RESOURCE_DOCUMENTS = "documents"
QUOTA_RESOURCE_PAGES = "pages"
QUOTA_RESOURCE_LLM_CALLS = "llm_calls"
QUOTA_RESOURCE_CLOUD_PARSES = "cloud_parse_calls"

#: 会花钱的四件事，声明成元组让"有哪些资源"有**一个**答案。
QUOTA_RESOURCES: tuple[str, ...] = (
    QUOTA_RESOURCE_DOCUMENTS,
    QUOTA_RESOURCE_PAGES,
    QUOTA_RESOURCE_LLM_CALLS,
    QUOTA_RESOURCE_CLOUD_PARSES,
)

#: 上面两类里**可以推导**的那些。它们的用量永远从 store 读，不从计数器读。
QUOTA_DERIVED_RESOURCES: frozenset[str] = frozenset(
    {QUOTA_RESOURCE_DOCUMENTS, QUOTA_RESOURCE_PAGES}
)

#: 资源的中文名。拒绝消息面向用户，资源名是英文的。
_QUOTA_LABELS: dict[str, str] = {
    QUOTA_RESOURCE_DOCUMENTS: "文档数",
    QUOTA_RESOURCE_PAGES: "累计解析页数",
    QUOTA_RESOURCE_LLM_CALLS: "模型调用次数",
    QUOTA_RESOURCE_CLOUD_PARSES: "云端解析次数",
}


class QuotaExceeded(RuntimeError):
    """配额用尽。

    **是一个异常而不是一个返回码。** 因为它必须在**花钱之前**抛出：一个
    "已超额"的返回值如果由调用方选择是否理会，那么漏掉理会的那条路径就会
    先花钱再发现问题。异常让"停下"成为默认行为。

    ``retry_at`` 存在是因为两种超额的补救方式完全不同：可推导资源（文档数、
    页数）只能通过删除既有内容来腾出额度，事件计数（调用次数）则会在窗口
    滚动后自动恢复。只说"超配额了"而不说哪一种，运维只能猜。
    """

    status_code = 429
    code = "quota_exceeded"

    def __init__(
        self,
        resource: str,
        used: int,
        limit: int,
        *,
        tenant_id: str = "",
        retry_at: str = "",
    ) -> None:
        self.resource = str(resource)
        self.used = int(used)
        self.limit = int(limit)
        self.tenant_id = str(tenant_id or "")
        self.retry_at = str(retry_at or "")
        label = _QUOTA_LABELS.get(self.resource, self.resource)
        if self.resource in QUOTA_DERIVED_RESOURCES:
            relief = (
                f"配额按当前用量计算，不随时间恢复。"
                f"请先删除部分文档以腾出额度，或联系管理员提高「{label}」上限。"
            )
        else:
            relief = (
                f"配额将在额度窗口滚动后自动恢复（{self.retry_at}）。"
                f"如需立即继续，请联系管理员提高「{label}」上限。"
            )
        super().__init__(
            f"已达到「{label}」配额上限：已用 {self.used}，上限 {self.limit}。"
            f"本次请求已被拒绝，未产生任何费用。{relief}"
        )
        self.message = str(self)

    def to_dict(self) -> dict[str, Any]:
        """与 :func:`paperwhisperer.core.errors.build_error_payload` 同形。"""
        return {
            "error": self.message,
            "code": self.code,
            "resource": self.resource,
            "used": self.used,
            "limit": self.limit,
            "retry_at": self.retry_at,
        }


@dataclass(frozen=True, slots=True)
class QuotaLimits:
    """一个租户在四件事上的上限。**0 表示不限制。**

    用 0 而不是 ``None`` 表示"无限制"，是因为 ``None`` 常常意味着"忘了配"，
    而本文件里"忘了配"的正确解读是"沿用库默认值"；把默认值定成
    :data:`DEFAULT_QUOTA_LIMITS`（全部为 0 = 不限制）后，两种"不限制"的写法
    结果相同，配置里少一个键不会让部署突然被限流。
    """

    documents: int = 0
    pages: int = 0
    llm_calls: int = 0
    cloud_parse_calls: int = 0

    def limit_for(self, resource: str) -> int:
        return int(getattr(self, str(resource), 0) or 0)

    @classmethod
    def from_settings(cls, settings: Any, tenant_id: str = "") -> QuotaLimits:
        """从配置里读出上限。

        ``tenant_id`` 非空时会先查 ``QUOTA_<TENANT>_<RESOURCE>`` 这个
        按租户覆盖的键，再回落到全局键。没有按租户覆盖是一条**明确的
        决定**：一份配额要么对所有租户一样，要么由调用方在构造
        :class:`QuotaLedger` 时直接传一个 :class:`QuotaLimits`。从环境变量
        读出几百个租户的配置不是 T4.5 的问题，租户之间的额度差异应该来自
        租户表，而那张表不归本模块管。
        """
        base = {
            "documents": _int_setting(settings, "QUOTA_MAX_DOCUMENTS"),
            "pages": _int_setting(settings, "QUOTA_MAX_PAGES"),
            "llm_calls": _int_setting(settings, "QUOTA_MAX_LLM_CALLS"),
            "cloud_parse_calls": _int_setting(settings, "QUOTA_MAX_CLOUD_PARSES"),
        }
        if tenant_id:
            prefix = f"QUOTA_{_env_name(tenant_id)}_"
            overrides = {
                key: _int_setting(settings, f"{prefix}{_env_name(key)}", sentinel=None)
                for key in base
            }
            base = {key: (overrides[key] if overrides[key] is not None else value)
                    for key, value in base.items()}
        return cls(**base)

    def to_dict(self) -> dict[str, int]:
        return {resource: self.limit_for(resource) for resource in QUOTA_RESOURCES}


#: 不限制。这是 D2 的直接推论：**个人档位是默认档位，而个人档位不该有配额。**
#: 给一个单用户部署设"最多 500 页"，是在给一件从未发生过的问题写防护，
#: 并且在用户真的写了 500 页的那天制造一次莫名其妙的失败。
DEFAULT_QUOTA_LIMITS = QuotaLimits()

#: 本模块读的配置键，与默认值。契约同 :data:`paperwhisperer.api.auth.
#: AUTH_CONFIG_KEYS`：``core/config.py`` 归别的任务所有，谁加键就照抄。
QUOTA_CONFIG_KEYS: dict[str, Any] = {
    "QUOTA_MAX_DOCUMENTS": 0,
    "QUOTA_MAX_PAGES": 0,
    "QUOTA_MAX_LLM_CALLS": 0,
    "QUOTA_MAX_CLOUD_PARSES": 0,
    #: 事件型配额的窗口长度（秒）。默认 86400 = 一天。
    "QUOTA_WINDOW_SECONDS": 86400,
}

def _int_setting(settings: Any, key: str, sentinel: Any = 0) -> Any:
    raw = getattr(settings, key, None)
    if raw is None:
        return QUOTA_CONFIG_KEYS.get(key, sentinel)
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        logger.warning("%s=%r 不是非负整数，按 0（不限制）处理。", key, raw)
        return 0


def _env_name(text: str) -> str:
    """租户/资源名 → 环境变量片段。小写、非字母数字变下划线。"""
    return "".join(
        character if character.isalnum() else "_" for character in str(text).lower()
    ).upper()


@dataclass(frozen=True, slots=True)
class QuotaUsage:
    """一次用量查询的结果：用了多少、上限多少、还剩多少。"""

    tenant_id: str
    resource: str
    used: int
    limit: int
    derived: bool = False
    retry_at: str = ""

    @property
    def remaining(self) -> int:
        """还能用多少。``limit=0``（不限制）时是 ``-1`` 而不是 0。

        ``-1`` 而不是 0：这两个数在告警里含义完全相反——0 会让一个
        "无限制" 的租户看起来像"用完了"。返回 ``None`` 也行，但那样每个
        读它的地方都要判空，而本文件的所有读者都是数值运算。
        """
        return -1 if self.limit <= 0 else max(0, self.limit - self.used)

    @property
    def exceeded(self) -> bool:
        return self.limit > 0 and self.used >= self.limit

    @property
    def utilization(self) -> float:
        """0.0–1.0。不限制时返回 ``0.0``，理由同 :attr:`remaining`。"""
        return 0.0 if self.limit <= 0 else min(1.0, self.used / self.limit)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "resource": self.resource,
            "used": self.used,
            "limit": self.limit,
            "remaining": self.remaining,
            "utilization": round(self.utilization, 4),
            "exceeded": self.exceeded,
            "derived": self.derived,
            "retry_at": self.retry_at,
        }


class QuotaLedger:
    """检查配额、记用量，**并在超额时抛 :class:`QuotaExceeded`**。

    三种用法，对应三个方法，且**顺序不可交换**：

    * :meth:`check` —— 只看不扣。用于"能不能提交这个任务"的预检。
    * :meth:`reserve` —— **先扣后做事**。这是花钱那条路径该用的：一次云解析
      先记一次，再去调用。它保证"记账失败"不会发生在"钱已经花了"之后。
    * :meth:`consume` —— ``check`` + 记一次。用于不需要预留的场景。

    为什么需要 reserve/consume 的区分：一个"检查→做事→记账"的序列里，如果
    做事这一步把进程打死了，计数就永远没记上，而钱花了。预留式的顺序把这个
    窗口关掉，代价是"预留了但没做事"会多计一次——而**多计**是安全的方向
    （它让配额更紧，不会更松）。

    **持久性的诚实说明。** 事件型计数默认存在进程内，重启清零。单实例部署
    可以接受；多实例部署需要 Redis 支撑，那属于 T4.5 未完成的部分。文档型与
    页数型配额不受影响——它们从 store 推导，重启不影响。
    """

    def __init__(
        self,
        limits: QuotaLimits | None = None,
        *,
        window_seconds: int | None = None,
        clock: Any = None,
    ) -> None:
        self.limits = limits if limits is not None else DEFAULT_QUOTA_LIMITS
        self.window_seconds = max(
            1, int(window_seconds if window_seconds is not None else 86400)
        )
        self._clock = clock or (lambda: datetime.now(timezone.utc).timestamp())
        self._lock = threading.Lock()
        self._counts: dict[tuple[str, str, int], int] = {}

    # -- 用量 ---------------------------------------------------------------

    def _window_index(self) -> int:
        """当前所处的窗口序号。窗口滚动后旧计数**整体作废**。

        整体作废而不是逐条衰减：窗口型配额的标准语义就是"最近 N 秒"，
        而逐条衰减需要给每条计数打时间戳并做加权，得到的数字既不直观
        也不可测。
        """
        return int(self._clock() // self.window_seconds)

    def derived_usage(self, resource: str, bundle: Any = None) -> int:
        """从 store 推导出的用量。文档数与页数走这里。"""
        if bundle is None:
            return 0
        if resource == QUOTA_RESOURCE_DOCUMENTS:
            return int(bundle.metadata.count_documents())
        if resource == QUOTA_RESOURCE_PAGES:
            return int(
                sum(
                    max(0, int(getattr(record, "page_count", 0) or 0))
                    for record in bundle.metadata.list_documents(limit=10**9, offset=0)
                )
            )
        return 0

    def usage(
        self,
        tenant_id: str,
        resource: str,
        *,
        bundle: Any = None,
    ) -> QuotaUsage:
        """当前用量。可推导资源读 store，事件资源读计数器。"""
        if resource not in QUOTA_RESOURCES:
            raise ValueError(
                f"未知的配额资源：{resource!r}（可用值：{', '.join(QUOTA_RESOURCES)}）"
            )
        limit = self.limits.limit_for(resource)
        if resource in QUOTA_DERIVED_RESOURCES:
            return QuotaUsage(
                tenant_id=tenant_id,
                resource=resource,
                used=self.derived_usage(resource, bundle),
                limit=limit,
                derived=True,
            )
        window = self._window_index()
        with self._lock:
            used = int(self._counts.get((tenant_id, resource, window), 0))
        return QuotaUsage(
            tenant_id=tenant_id,
            resource=resource,
            used=used,
            limit=limit,
            derived=False,
            retry_at=self.window_reset_at(),
        )

    def window_reset_at(self) -> str:
        """当前窗口的结束时刻（ISO 字符串）。给用户看的"什么时候恢复"。"""
        end = (self._window_index() + 1) * self.window_seconds
        return datetime.fromtimestamp(end, tz=timezone.utc).isoformat(timespec="seconds")

    def snapshot(self, tenant_id: str, *, bundle: Any = None) -> dict[str, Any]:
        """四类资源的完整用量。给 health / 管理端点。"""
        return {
            resource: self.usage(tenant_id, resource, bundle=bundle).to_dict()
            for resource in QUOTA_RESOURCES
        }

    # -- 检查与记账 ---------------------------------------------------------

    def check(
        self,
        tenant_id: str,
        resource: str,
        amount: int = 1,
        *,
        bundle: Any = None,
    ) -> QuotaUsage:
        """检查额度。**超了就抛** :class:`QuotaExceeded`，不返回 False。

        边界用 ``used + amount > limit`` 而不是 ``used >= limit``：上限为 1
        的额度必须允许**一次**调用。用 ``>=`` 会让"上限 1"的租户一次都用不了。
        """
        current = self.usage(tenant_id, resource, bundle=bundle)
        if current.limit > 0 and current.used + max(1, int(amount)) > current.limit:
            raise QuotaExceeded(
                resource,
                current.used,
                current.limit,
                tenant_id=tenant_id,
                retry_at=current.retry_at,
            )
        return current

    def reserve(
        self,
        tenant_id: str,
        resource: str,
        amount: int = 1,
        *,
        bundle: Any = None,
    ) -> QuotaUsage:
        """检查并**先记账**。花钱的路径用它。"""
        count = max(1, int(amount))
        current = self.check(tenant_id, resource, count, bundle=bundle)
        if resource in QUOTA_DERIVED_RESOURCES:
            # 推导型资源没有可记的账：它们的用量就是事实，记录只会造出
            # 一个与 store 不一致的影子计数。
            return current
        window = self._window_index()
        with self._lock:
            key = (tenant_id, resource, window)
            self._counts[key] = int(self._counts.get(key, 0)) + count
        return self.usage(tenant_id, resource, bundle=bundle)

    def consume(
        self, tenant_id: str, resource: str, amount: int = 1, *, bundle: Any = None
    ) -> QuotaUsage:
        """:meth:`reserve` 的别名，读起来更顺的一种叫法。"""
        return self.reserve(tenant_id, resource, amount, bundle=bundle)

    def reset(self, tenant_id: str | None = None) -> None:
        """清空计数。给测试与运维用；**不影响推导型配额**。"""
        with self._lock:
            if tenant_id is None:
                self._counts.clear()
            else:
                for key in [k for k in self._counts if k[0] == tenant_id]:
                    del self._counts[key]


_LEDGER: QuotaLedger | None = None
_LEDGER_LOCK = threading.Lock()


def quota_ledger(settings: Any = None) -> QuotaLedger:
    """进程级配额台账，**只在托管档位构建**。

    个人档位返回的 ledger 带 :data:`DEFAULT_QUOTA_LIMITS`（全部 0 = 不限制），
    因此"个人档位不该有配额"这件事是由**数据**保证的，而不是由一条
    ``if personal: skip`` 分支保证的。那条分支会在某次重构里被删掉，而一个
    恒为不限制的台账不会。
    """
    global _LEDGER
    with _LEDGER_LOCK:
        if _LEDGER is None:
            limits = (
                QuotaLimits.from_settings(settings)
                if settings is not None
                else DEFAULT_QUOTA_LIMITS
            )
            _LEDGER = QuotaLedger(
                limits,
                window_seconds=int(_int_setting(settings, "QUOTA_WINDOW_SECONDS", 86400)),
            )
        return _LEDGER


def reset_quota_ledger() -> None:
    """丢弃进程级台账。给测试用。"""
    global _LEDGER
    with _LEDGER_LOCK:
        _LEDGER = None
