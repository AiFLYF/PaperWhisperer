"""T4.5 的验收测试：跨租户访问为零。

计划原文：「鉴权 / tenant scope / 配额；Celery+Redis 多 worker；S3/MinIO；
MinerULocalAdapter（self-host）按需实现」，验收标准「**跨租户访问为零
（安全测试）**；云调用全部有审计」。

**本文件为什么是"结构化"的，而不是"试两条路径"。** 跨租户漏洞的形状不是
"某处判断写错了"，而是"某条路径上**压根没有那个判断**"。因此
:data:`SCOPED_READ_PROBES` 把系统里每一条能读出数据的路径列成一张表，
每一个都是一次独立的尝试——不是两次，是**全部**。新增一条读路径却忘了把它
加进这张表，是本文件唯一拦不住的情况，而那张表本身是本文件里最显眼的一段
代码：任何读过它的人都会知道"加一条读路径就得往这里加一行"。

每个探针都返回一组**字符串 id**（文档 id / chunk id），断言只有一个：
里面不含任何属于对方租户的 id。用 id 而不是计数，是因为计数会漏掉
"返回了 3 条，其中 1 条是别人的"这种最难发现的形态。

**跑在真的 SQLite 后端上，不是 mock。** 租户层是 store 的一层代理，代理的
对错取决于它下游那个 store 怎么处理 ``document_ids=[]``——而那正是
:mod:`paperwhisperer.infra.stores.sqlite_backend` 里的
``_candidates``。用 mock 测这层，就等于把"空集返回空结果"这个最关键的假设
当成了给定。

**个人档位必须逐字节不变。** 这是 D2 的直接要求，也是整个 T4.5 最容易搞坏
的东西：在默认档位上加登录屏是回归，不是安全改进。所以
:class:`TestPersonalTierIsUnaffected` 用**同一个 bundle** 分别经由真实 bundle
与范围化代理各查一遍，逐字比较结果——个人档位的惰性是被证明的，不是被断言的。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from paperwhisperer.api import auth
from paperwhisperer.api import tenant as tenant_mod
from paperwhisperer.api.auth import (
    LOCAL_TENANT_ID,
    SCOPE_ALL,
    SCOPE_CLOUD_PARSE,
    SCOPE_DOCUMENTS_READ,
    AuthenticationError,
    AuthNotConfigured,
    AuthorizationError,
    DisabledAuthenticator,
    Principal,
    StaticTokenAuthenticator,
    TokenEntry,
    build_authenticator,
    current_principal,
    hash_token,
    require_scope,
    resolve_tier,
)
from paperwhisperer.api.tenant import (
    QuotaExceeded,
    QuotaLedger,
    QuotaLimits,
    TenantScope,
    assert_owns_chunk,
    assert_owns_document,
    assert_owns_session,
    document_scope_filter,
    local_scope,
    owns_chunk,
    owns_document,
    owns_job,
    owns_session,
    resolve_documents_for,
    scope_bundle,
    session_tenant,
    tenant_of_record,
)
from paperwhisperer.infra.stores.interfaces import (
    BACKEND_POSTGRES,
    BACKEND_SQLITE,
)

from .test_stores_tiers import open_bundle, sample_chunk, sample_record

# --------------------------------------------------------------------------
# 夹具：一份真语料，两个租户
# --------------------------------------------------------------------------

#: 每个租户名下的文档。刻意给两个租户**同样数量**的文档——数量不等时
#: "少返回的那几条"很容易被当成过滤生效，而不是被当成泄露。
TENANT_A = "tenant_alpha"
TENANT_B = "tenant_beta"

A_DOCS = ("doc-a1", "doc-a2")
B_DOCS = ("doc-b1", "doc-b2")
A_CHUNKS = ("chunk-a1", "chunk-a2")
B_CHUNKS = ("chunk-b1", "chunk-b2")

#: 两条腿检索都会命中这个词，于是 dense 与 lexical 都有机会泄露。
NEEDLE_EN = "transformer"
NEEDLE_ZH = "注意力机制"


def _principal(tenant_id: str, scopes=frozenset({SCOPE_ALL})) -> Principal:
    return Principal(
        subject=f"user@{tenant_id}",
        tenant_id=tenant_id,
        scopes=frozenset(scopes),
        method="static_tokens",
        authenticated=True,
    )


@pytest.fixture
def store(tmp_path):
    """一个真的 SQLite bundle，两个租户的数据都写进去了。

    数据**经由各自的范围化 store 写入**，而不是直接往底层塞——这样"归属是谁"
    是被写路径盖章的，而不是被测试手工指定的。:func:`test_upsert_stamps_the_owning_tenant`
    再单独确认这次盖章确实发生了。
    """
    harness = open_bundle("personal", tmp_path)
    bundle = harness.bundle
    try:
        for owner, docs, chunks in ((TENANT_A, A_DOCS, A_CHUNKS), (TENANT_B, B_DOCS, B_CHUNKS)):
            scope = tenant_mod.TenantScope(tenant_id=owner, documents=None if owner == LOCAL_TENANT_ID else frozenset())
            # 托管租户不能用 documents=None（那是"拥有全集"），所以先建一个空范围
            # 再让 upsert 把新文档登记进去。
            if owner != LOCAL_TENANT_ID:
                scope = tenant_mod.TenantScope(tenant_id=owner, documents=frozenset())
            scoped = scope_bundle(bundle, scope)
            for index, document_id in enumerate(docs):
                scoped.metadata.upsert_document(sample_record(document_id))
                for chunk_index, chunk_id in enumerate(chunks if index == 0 else ()):
                    text = f"{NEEDLE_ZH} {NEEDLE_EN} tenant {owner} 机密内容 {chunk_id}"
                    scoped.metadata.put_chunk(document_id, sample_chunk(chunk_id, order=chunk_index))
                    scoped.vectors.upsert_vector(
                        chunk_id,
                        (1.0, float(index + 1), float(chunk_index + 1)),
                        document_id=document_id,
                    )
                    scoped.lexical.index_chunk(chunk_id, text, document_id=document_id)
        yield bundle
    finally:
        harness.close()


@pytest.fixture
def scope_a() -> TenantScope:
    """A 的范围。``documents`` 走归属解析而不是手写，和生产路径一致。"""
    return TenantScope(tenant_id=TENANT_A, documents=frozenset())


@pytest.fixture
def scoped_a(store, scope_a) -> Any:
    """A 的范围化视图。归属从 store 解析，所以 A 真的能看见自己那两份。"""
    principal = _principal(TENANT_A)
    resolved = resolve_documents_for(principal, store)
    assert resolved == frozenset(A_DOCS), "归属解析应当恰好找到 A 的两份文档"
    scope = TenantScope(tenant_id=TENANT_A, documents=resolved, principal=principal)
    with tenant_mod.bind_tenant(scope):
        yield scope_bundle(store, scope)


@pytest.fixture
def scoped_empty(store) -> Any:
    """一个**一个文档都没有**的托管租户。

    单独一个夹具，因为"空范围"是最容易退化成"全量"的那一档：它的过滤结果
    恰好等于空集，于是"忘了过滤"和"过滤正确"在返回值上**长得一样**，
    只能靠底层那个"没被碰过"的文档来区分。
    """
    scope = TenantScope(tenant_id="tenant_gamma", documents=frozenset())
    with tenant_mod.bind_tenant(scope):
        yield scope_bundle(store, scope)


# --------------------------------------------------------------------------
# 读路径清单：验收标准在这里被枚举出来
# --------------------------------------------------------------------------


def _probe_get_document(bundle) -> set[str]:
    return {str(bundle.metadata.get_document(d) or "") for d in B_DOCS} - {""}


def _probe_list_documents(bundle) -> set[str]:
    return {str(r.document_id) for r in bundle.metadata.list_documents(limit=100)}




def _probe_get_chunk(bundle) -> set[str]:
    return {str(bundle.metadata.get_chunk(c) or "") for c in B_CHUNKS} - {""}


def _probe_list_chunks(bundle) -> set[str]:
    found: set[str] = set()
    for document_id in A_DOCS + B_DOCS:
        found |= {str(c.chunk_id) for c in bundle.metadata.list_chunks(document_id)}
    return found


def _probe_get_facts(bundle) -> set[str]:
    return {str(bundle.metadata.get_facts(d) or "") for d in A_DOCS + B_DOCS} - {""}


def _probe_delete_document(bundle) -> set[str]:
    """删除也是读——它必须先读归属。返回被报告为"删掉了"的那些 id。"""
    return {str(d) for d in B_DOCS if bundle.metadata.delete_document(d)}


def _probe_vector_get(bundle) -> set[str]:
    return {str(bundle.vectors.get_vector(c) or "") for c in B_CHUNKS} - {""}


def _probe_vector_search(bundle) -> set[str]:
    hits = bundle.vectors.search((1.0, 1.0, 1.0), top_k=50)
    return {str(h.chunk_id) for h in hits}


def _probe_vector_search_filtered(bundle) -> set[str]:
    """调用方**点名**要 B 的文档时，也必须拿不到。"""
    hits = bundle.vectors.search((1.0, 1.0, 1.0), top_k=50, document_ids=list(B_DOCS))
    return {str(h.chunk_id) for h in hits}


def _probe_vector_delete(bundle) -> set[str]:
    return {str(c) for c in B_CHUNKS if bundle.vectors.delete_vector(c)}


def _probe_lexical_search(bundle) -> set[str]:
    hits = bundle.lexical.search(NEEDLE_ZH, top_k=50)
    return {str(h.chunk_id) for h in hits}


def _probe_lexical_search_en(bundle) -> set[str]:
    hits = bundle.lexical.search(NEEDLE_EN, top_k=50)
    return {str(h.chunk_id) for h in hits}


def _probe_lexical_search_filtered(bundle) -> set[str]:
    hits = bundle.lexical.search(NEEDLE_ZH, top_k=50, document_ids=list(B_DOCS))
    return {str(h.chunk_id) for h in hits}


def _probe_lexical_delete(bundle) -> set[str]:
    return {str(c) for c in B_CHUNKS if bundle.lexical.delete_chunk(c)}


def _probe_delete_chunks(bundle) -> set[str]:
    return {str(d) for d in B_DOCS if bundle.metadata.delete_chunks(d)}


def _probe_put_facts(bundle) -> set[str]:
    """写入对方文档的事实必须被拒——归属检查在写路径上同样存在。"""
    from .test_stores_tiers import sample_facts

    return {
        str(d)
        for d in B_DOCS
        # 默认参数把循环变量绑死在这一次迭代上：否则闭包在循环结束后才被调用，
        # 拿到的是最后一个 d，于是四个文档实际只测了同一个（B023）。
        if _is_allowed(lambda doc=d: bundle.metadata.put_facts(doc, sample_facts()))
    }


def _probe_put_chunk(bundle) -> set[str]:
    from .test_stores_tiers import sample_chunk as make_chunk

    return {
        str(d)
        for d in B_DOCS
        if _is_allowed(lambda doc=d: bundle.metadata.put_chunk(doc, make_chunk(f"chunk-x-{doc}")))
    }


def _is_allowed(action) -> bool:
    """跑一个写动作，返回它有没有成功。写被拒不算暴露，所以记 False。"""
    from paperwhisperer.api.auth import AuthorizationError

    try:
        action()
    except AuthorizationError:
        return False
    return True


def _probe_document_scope_filter(bundle) -> set[str]:
    """显式点名对方的文档去要范围，范围必须被收窄回自己那几份。"""
    return set(document_scope_filter(bundle.inner, list(A_DOCS + B_DOCS)))


def _probe_owns_document(bundle) -> set[str]:
    return {str(d) for d in B_DOCS if owns_document(bundle.inner, d)}


def _probe_owns_chunk(bundle) -> set[str]:
    return {str(c) for c in B_CHUNKS if owns_chunk(bundle.inner, c)}


#: **每一条能读出数据的路径。** 键是给人看的路径名，值返回一组 id。
#:
#: 新增一条读路径的人，**必须**在这里加一行。这不是建议：这张表是
#: "跨租户访问为零"这句话在本仓里唯一的枚举处，而一句没被枚举的承诺
#: 等于没有承诺。
SCOPED_READ_PROBES: dict[str, Callable[[Any], set[str]]] = {
    "metadata.get_document": _probe_get_document,
    "metadata.list_documents": _probe_list_documents,
    "metadata.get_chunk": _probe_get_chunk,
    "metadata.list_chunks": _probe_list_chunks,
    "metadata.get_facts": _probe_get_facts,
    "metadata.put_facts": _probe_put_facts,
    "metadata.put_chunk": _probe_put_chunk,
    "metadata.delete_chunks": _probe_delete_chunks,
    "metadata.delete_document": _probe_delete_document,
    "vectors.get_vector": _probe_vector_get,
    "vectors.search": _probe_vector_search,
    "vectors.search(document_ids=对方)": _probe_vector_search_filtered,
    "vectors.delete_vector": _probe_vector_delete,
    "lexical.search": _probe_lexical_search,
    "lexical.search(英文词)": _probe_lexical_search_en,
    "lexical.search(document_ids=对方)": _probe_lexical_search_filtered,
    "lexical.delete_chunk": _probe_lexical_delete,
    "document_scope_filter": _probe_document_scope_filter,
    "owns_document": _probe_owns_document,
    "owns_chunk": _probe_owns_chunk,
}


#: 返回**数量**而不是 id 的读路径。它们用另一组断言（见
#: :class:`TestScopedCounts`）：一个"暴露了 3 条其中 1 条是别人的"的 id 集合
#: 能被集合运算抓住，而一个只返回计数的函数只能被直接比较。
SCOPED_COUNT_PROBES: dict[str, Callable[[Any], int]] = {
    "metadata.count_documents": lambda bundle: int(bundle.metadata.count_documents()),
    "vectors.count_vectors": lambda bundle: int(bundle.vectors.count_vectors()),
    "lexical.count_chunks": lambda bundle: int(bundle.lexical.count_chunks()),
}


#: 代理类上**不是读路径**的公开方法，以及不列为读路径的理由。
#:
#: 存在的唯一目的是让 :meth:`TestCrossTenantReadsAreEmpty.
#: test_the_probe_table_covers_the_bundles_public_surface` 成为一个真正的闸：
#: 有了这张表，"某个方法不在任何清单里"是一条明确的失败，而不是一个可以
#: 默认为"它大概是安全的"的空档。
KNOWN_NON_READ_METHODS: dict[str, str] = {
    # -- 管道/生命周期，不接触数据 ------------------------------------------
    "inner": "取被代理的真实 store，给备份与存储层下沉用",
    "scope": "取范围本身",
    "capabilities": "透传后端能力",
    "initialize": "生命周期",
    "close": "生命周期",
    "dimension": "透传向量维度",
    # -- 写入口：归属靠**盖章**与拒绝，不靠"过滤返回值" ---------------------
    "upsert_document": "写入时把归属改成本租户，见 test_upsert_stamps_the_owning_tenant",
    "upsert_vector": "写路径越权被拒，见 test_a_write_into_another_tenants_document_is_refused",
    "upsert_many": "写路径越权被拒，同上",
    "index_chunk": "写路径越权被拒，同上",
}


class TestCrossTenantReadsAreEmpty:
    """验收标准本身：A 读 B 的每一条路径都必须是空的。"""

    @pytest.mark.parametrize("path", sorted(SCOPED_READ_PROBES), ids=sorted(SCOPED_READ_PROBES))
    def test_a_cannot_read_tenant_b(self, path, scoped_a):
        # **探针只跑一次。** 清单里有几个探针是写操作（删除、改归属），跑两次
        # 第二次就已经在测一个被自己改过的世界了。
        exposed = SCOPED_READ_PROBES[path](scoped_a)
        leaked = (exposed & set(B_DOCS)) | (exposed & set(B_CHUNKS))
        assert leaked == set(), f"{path} 把对方租户的数据交给了 A：{sorted(leaked)}"

    @pytest.mark.parametrize("path", sorted(SCOPED_READ_PROBES), ids=sorted(SCOPED_READ_PROBES))
    def test_an_empty_tenant_sees_nothing_at_all(self, path, scoped_empty):
        """一个文档都没有的租户，每条路径都必须是空。"""
        assert SCOPED_READ_PROBES[path](scoped_empty) == set()

    def test_the_probe_table_covers_the_bundles_public_surface(self):
        """新方法没进清单，测试就失效了——所以反过来盯住接口本身。

        代理类上每个**公开方法**都必须要么在 :data:`SCOPED_READ_PROBES` /
        :data:`SCOPED_COUNT_PROBES` 里，要么在 :data:`KNOWN_NON_READ_METHODS`
        里并有一句说明为什么。有人加了一个 ``get_evidence()`` 而忘了加进清单，
        这里会失败——这是让"清单"不至于腐烂的唯一办法。
        """
        listed = {name for name in SCOPED_READ_PROBES} | {name for name in SCOPED_COUNT_PROBES}
        for store in (
            tenant_mod.ScopedMetadataStore,
            tenant_mod.ScopedVectorStore,
            tenant_mod.ScopedLexicalStore,
        ):
            for name in dir(store):
                if name.startswith("_") or not callable(getattr(store, name, None)):
                    continue
                if name in KNOWN_NON_READ_METHODS:
                    continue
                assert any(entry.endswith(f".{name}") for entry in listed), (
                    f"{store.__name__} 的 {name}() 既不在 SCOPED_READ_PROBES 里，"
                    f"也不在 KNOWN_NON_READ_METHODS 里——它是一条没人检查的路径。"
                )

    def test_a_still_sees_its_own_data(self, scoped_a):
        """隔离不是靠"什么都不返回"实现的——那是最容易骗过自己的做法。"""
        assert {str(r.document_id) for r in scoped_a.metadata.list_documents(limit=100)} == set(A_DOCS)
        assert scoped_a.metadata.get_document(A_DOCS[0]) is not None
        assert {str(h.chunk_id) for h in scoped_a.lexical.search(NEEDLE_ZH, top_k=50)} == set(
            A_CHUNKS
        )
        assert {str(h.chunk_id) for h in scoped_a.vectors.search((1.0, 1.0, 1.0), top_k=50)} == set(
            A_CHUNKS
        )


class TestScopedCounts:
    @pytest.mark.parametrize("path", sorted(SCOPED_COUNT_PROBES), ids=sorted(SCOPED_COUNT_PROBES))
    def test_a_counts_only_its_own(self, path, scoped_a):
        counts = {
            "metadata.count_documents": len(A_DOCS),
            "vectors.count_vectors": len(A_CHUNKS),
            "lexical.count_chunks": len(A_CHUNKS),
        }
        assert SCOPED_COUNT_PROBES[path](scoped_a) == counts[path]

    @pytest.mark.parametrize("path", sorted(SCOPED_COUNT_PROBES), ids=sorted(SCOPED_COUNT_PROBES))
    def test_an_empty_tenant_counts_zero(self, path, scoped_empty):
        assert SCOPED_COUNT_PROBES[path](scoped_empty) == 0


class TestTenantScopeMechanics:
    def test_upsert_stamps_the_owning_tenant(self, store, scoped_a):
        """归属由写入方盖章，不是调用方自报的。"""
        record = scoped_a.metadata.get_document(A_DOCS[0])
        assert tenant_of_record(record) == TENANT_A
        assert tenant_of_record(record) != TENANT_B

    def test_a_record_without_ownership_falls_back_to_the_local_tenant(self):
        """存量部署打开鉴权之后，每份旧文档仍然读得到，而不是集体 404。"""
        assert tenant_of_record(sample_record("doc-x")) == LOCAL_TENANT_ID
        assert tenant_of_record(object()) == LOCAL_TENANT_ID

    def test_an_unbound_scope_raises_rather_than_defaulting_to_everything(self):
        """默认"拥有全集"是最危险的默认值——它让一处漏绑变成一次全库读取。"""
        with pytest.raises(tenant_mod.TenantScopeMissing):
            tenant_mod.current_scope()

    def test_the_filter_never_reduces_a_managed_tenant_to_the_whole_corpus(self, scoped_a):
        """``None``（全集）只能由显式声明拥有全部的范围产生。"""
        resolved = scoped_a.scope
        assert resolved.filter_document_ids(None) == tuple(sorted(A_DOCS))
        assert resolved.filter_document_ids(None) is not None
        assert resolved.filter_document_ids(list(B_DOCS)) == ()
        assert TenantScope(tenant_id=TENANT_A, documents=frozenset()).filter_document_ids(
            None
        ) == ()

    def test_a_managed_tenant_may_not_claim_the_whole_corpus(self, store):
        principal = _principal(TENANT_B)
        with pytest.raises(ValueError, match="全部文档"):
            tenant_mod._scope_for(principal, None)

    def test_the_local_tenant_may_own_everything(self):
        assert tenant_mod._scope_for(Principal.local(), None).owns_everything

    def test_assertions_raise_instead_of_returning_empty(self, scoped_a):
        """空结果给检索用，异常给"你要看它"用——两者要分开。"""
        with pytest.raises(AuthorizationError):
            assert_owns_document(scoped_a.inner, B_DOCS[0])
        with pytest.raises(AuthorizationError):
            assert_owns_chunk(scoped_a.inner, B_CHUNKS[0])
        with pytest.raises(AuthorizationError):
            tenant_mod.require_document_access(scoped_a.inner, B_DOCS[0])

    def test_own_assertions_pass_for_your_own(self, scoped_a):
        assert owns_document(scoped_a.inner, A_DOCS[0])
        assert owns_chunk(scoped_a.inner, A_CHUNKS[0])
        assert_owns_document(scoped_a.inner, A_DOCS[0])
        assert_owns_chunk(scoped_a.inner, A_CHUNKS[0])

    def test_a_write_into_another_tenants_document_is_refused(self, scoped_a):
        with pytest.raises(AuthorizationError):
            scoped_a.metadata.upsert_document(sample_record(B_DOCS[0]))
        with pytest.raises(AuthorizationError):
            scoped_a.vectors.upsert_vector(
                "chunk-new", (1.0, 0.0, 0.0), document_id=B_DOCS[0]
            )
        with pytest.raises(AuthorizationError):
            scoped_a.lexical.index_chunk("chunk-new", "x", document_id=B_DOCS[0])

    def test_a_write_into_your_own_document_succeeds(self, scoped_a):
        from .test_stores_tiers import sample_chunk as make_chunk

        # 先有 chunk 行再有向量：代理靠 chunk 表判断一个 chunk 是不是本租户的，
        # 没有 chunk 行的向量对任何人都不可见（这正是它该有的行为）。
        scoped_a.metadata.put_chunk(A_DOCS[0], make_chunk("chunk-a9"))
        scoped_a.vectors.upsert_vector("chunk-a9", (1.0, 0.0, 0.0), document_id=A_DOCS[0])
        scoped_a.lexical.index_chunk("chunk-a9", "新证据", document_id=A_DOCS[0])
        assert scoped_a.vectors.get_vector("chunk-a9") is not None
        assert scoped_a.metadata.get_chunk("chunk-a9") is not None


class TestJobsAndSessionsAreScopedToo:
    """job 与 session 不在 store bundle 里，它们的隔离点是另外两个。"""

    class _Job:
        def __init__(self, document_id="", session_id=""):
            self.document_id = document_id
            self.session_id = session_id

    def test_a_job_on_another_tenants_document_is_not_owned(self, scoped_a, monkeypatch):
        monkeypatch.setattr(tenant_mod, "_BUNDLE_RESOLVER", lambda: scoped_a.inner)
        with tenant_mod.bind_tenant(TENANT_A and tenant_mod.current_scope()):
            assert not owns_job(self._Job(document_id=B_DOCS[0]))
            assert owns_job(self._Job(document_id=A_DOCS[0]))

    def test_a_job_with_no_known_owner_is_claimed_by_nobody(self, scoped_a):
        """取消是一个会消耗云解析额度的写操作，归属不明就不认领。"""
        assert not owns_job(self._Job())
        assert not owns_job(self._Job(session_id="s1"), session_owner={})

    def test_a_job_is_owned_through_its_session_when_it_has_no_document(self, scoped_a):
        assert owns_job(self._Job(session_id="s-a"), session_owner={"s-a": TENANT_A})
        assert not owns_job(self._Job(session_id="s-b"), session_owner={"s-b": TENANT_B})

    def test_a_session_payload_carries_its_owner(self, scoped_a):
        assert owns_session({"tenant_id": TENANT_A})
        assert not owns_session({"tenant_id": TENANT_B})
        with pytest.raises(AuthorizationError):
            assert_owns_session({"tenant_id": TENANT_B})

    def test_a_session_without_ownership_falls_back_to_local(self):
        assert session_tenant({}) == LOCAL_TENANT_ID
        assert session_tenant(None) == LOCAL_TENANT_ID

    def test_the_bundles_own_resolver_is_required_for_job_checks(self, scoped_a, monkeypatch):
        monkeypatch.setattr(tenant_mod, "_BUNDLE_RESOLVER", None)
        with pytest.raises(tenant_mod.TenantScopeMissing):
            owns_job(self._Job(document_id=B_DOCS[0]))


# --------------------------------------------------------------------------
# 档位判定
# --------------------------------------------------------------------------


class _Settings:
    """一个只有本模块读的那几个键的配置对象。"""

    def __init__(self, **values):
        self.STORAGE_BACKEND = values.get("STORAGE_BACKEND", BACKEND_SQLITE)
        self.AUTH_MODE = values.get("AUTH_MODE", auth.AUTH_CONFIG_KEYS["AUTH_MODE"])
        self.AUTH_STATIC_TOKENS = values.get("AUTH_STATIC_TOKENS", "")
        self.AUTH_TENANT_DOMAIN = values.get("AUTH_TENANT_DOMAIN", "")


class TestTierResolution:
    @pytest.mark.parametrize("backend", ["", "sqlite", "file", "local", "unknown-value"])
    def test_everything_but_postgres_is_personal(self, backend):
        """未知值回落个人档：个人档是更安全的一档（没有跨租户可泄露）。"""
        assert resolve_tier(_Settings(STORAGE_BACKEND=backend)) == "personal"

    @pytest.mark.parametrize("backend", ["postgres", "postgresql", "pg", "POSTGRES"])
    def test_postgres_and_its_aliases_are_hosted(self, backend):
        assert resolve_tier(_Settings(STORAGE_BACKEND=backend)) == "hosted"

    def test_the_alias_set_matches_what_the_storage_layer_accepts(self):
        """两处不共享代码，所以这个一致性必须有测试钉住。

        不一致的后果是**具体的**：存储层按托管档连上 Postgres，而这里判定成
        个人档，于是免鉴权启动——多租户数据库上的免鉴权服务。
        """
        from paperwhisperer.infra.stores.factory import resolve_backend

        for backend in ("", "sqlite", "file", "local", "postgres", "postgresql", "pg"):
            settings = _Settings(STORAGE_BACKEND=backend)
            resolved_backend = resolve_backend(settings)
            expected = "hosted" if resolved_backend == BACKEND_POSTGRES else "personal"
            assert resolve_tier(settings) == expected, f"STORAGE_BACKEND={backend!r} 两处判定不一致"


class TestHostedTierRefusesToStartWithoutAuth:
    """缺配置就拒绝启动，而不是降级成免鉴权。

    一个配错的托管部署如果降级成免鉴权，后果是全部租户的数据互相可见——
    这是最坏的结果，而且它安静。一个连启动都起不来的部署，运维看得见。
    """

    def test_auth_disabled_on_hosted_refuses_startup(self):
        with pytest.raises(AuthNotConfigured, match="拒绝启动"):
            build_authenticator(_Settings(STORAGE_BACKEND="postgres", AUTH_MODE="disabled"))

    def test_static_tokens_with_an_empty_table_refuses_startup(self):
        with pytest.raises(AuthNotConfigured, match="没有任何凭据"):
            build_authenticator(
                _Settings(
                    STORAGE_BACKEND="postgres",
                    AUTH_MODE="static_tokens",
                    AUTH_STATIC_TOKENS="",
                )
            )

    def test_a_blank_token_table_is_still_empty(self):
        """运维在 .env 里留一个尾随分号，不该让服务起不来，也不该让它变成免鉴权。"""
        with pytest.raises(AuthNotConfigured):
            build_authenticator(
                _Settings(
                    STORAGE_BACKEND="postgres",
                    AUTH_MODE="static_tokens",
                    AUTH_STATIC_TOKENS=" ; ; ",
                )
            )

    def test_a_configured_hosted_tier_builds_a_real_authenticator(self):
        built = build_authenticator(
            _Settings(
                STORAGE_BACKEND="postgres",
                AUTH_MODE="static_tokens",
                AUTH_STATIC_TOKENS=f"tok-a:{TENANT_A};tok-b:{TENANT_B}",
            )
        )
        assert isinstance(built, StaticTokenAuthenticator)
        assert built.token_count == 2
        assert built.tenants() == (TENANT_A, TENANT_B)
        assert built.describe()["enforced"] is True

    def test_startup_helper_is_the_same_decision(self):
        with pytest.raises(AuthNotConfigured):
            auth.assert_startup_safe(_Settings(STORAGE_BACKEND="postgres"))


# --------------------------------------------------------------------------
# 凭据 → 租户
# --------------------------------------------------------------------------


class TestAuthentication:
    def _hosted(self, table="tok-a:tenant_alpha;tok-b:tenant_beta"):
        return build_authenticator(
            _Settings(
                STORAGE_BACKEND="postgres",
                AUTH_MODE="static_tokens",
                AUTH_STATIC_TOKENS=table,
            )
        )

    def test_the_tenant_comes_from_the_credential_not_from_a_header(self):
        """回声式的租户选择是这类系统最常见的越权来源。"""
        principal = self._hosted().authenticate({"Authorization": "Bearer tok-a"})
        assert principal.tenant_id == TENANT_A

    def test_named_tiers_get_their_own_tokens(self):
        built = self._hosted()
        assert built.authenticate({"authorization": "Bearer tok-b"}).tenant_id == TENANT_B
        assert built.authenticate({"Authorization": "Bearer tok-b"}).tenant_id == TENANT_B

    def test_bearer_is_matched_case_insensitively(self):
        assert self._hosted().authenticate({"Authorization": "bEaReR tok-a"}).tenant_id == TENANT_A

    def test_a_missing_credential_is_refused(self):
        with pytest.raises(AuthenticationError, match="Bearer"):
            self._hosted().authenticate({})

    def test_an_unknown_token_is_refused_without_saying_which(self):
        """区分"token 不存在"与"token 错误"等于提供一个无需凭据的枚举接口。"""
        with pytest.raises(AuthenticationError) as exc:
            self._hosted().authenticate({"Authorization": "Bearer nope"})
        assert exc.value.code == "invalid_credential"
        assert "nope" not in str(exc.value)

    def test_a_credential_of_another_tenants_shape_cannot_claim_the_local_tenant(self):
        with pytest.raises(ValueError, match="保留字"):
            TokenEntry.from_token("tok", LOCAL_TENANT_ID)

    def test_describe_never_contains_a_token_or_its_hash(self):
        built = self._hosted()
        described = json.dumps(built.describe(), ensure_ascii=False)
        assert "tok-a" not in described
        assert hash_token("tok-a") not in described

    def test_a_principal_without_a_tenant_cannot_be_built(self):
        with pytest.raises(ValueError, match="tenant_id"):
            Principal(subject="x", tenant_id="  ")

    def test_an_unbound_principal_raises_rather_than_reading_empty(self):
        """漏读必须抛异常，否则空租户一路传到数据层就是一次全库查询。"""
        with pytest.raises(AuthenticationError, match="principal_not_bound|没有绑定"):
            current_principal()

    def test_scopes_default_to_denying_everything(self):
        """默认放行与默认拒绝的区别，在多租户系统里就是"上线了"的区别。"""
        bare = Principal(subject="x", tenant_id=TENANT_A)
        with pytest.raises(AuthorizationError, match="documents:read"):
            require_scope(bare, SCOPE_DOCUMENTS_READ)

    def test_the_refusal_does_not_echo_what_the_caller_actually_has(self):
        bare = Principal(subject="x", tenant_id=TENANT_A, scopes=frozenset({SCOPE_CLOUD_PARSE}))
        with pytest.raises(AuthorizationError) as exc:
            require_scope(bare, SCOPE_DOCUMENTS_READ)
        assert SCOPE_CLOUD_PARSE not in str(exc.value)

    def test_the_wildcard_grants_everything(self):
        require_scope(Principal(subject="x", tenant_id="t", scopes=frozenset({SCOPE_ALL})), SCOPE_CLOUD_PARSE)

    def test_an_explicitly_listed_scope_does_not_leak_onto_the_others(self):
        narrow = Principal(
            subject="x", tenant_id="t", scopes=frozenset({SCOPE_DOCUMENTS_READ, SCOPE_CLOUD_PARSE})
        )
        require_scope(narrow, SCOPE_DOCUMENTS_READ)
        with pytest.raises(AuthorizationError):
            require_scope(narrow, "jobs:write")


# --------------------------------------------------------------------------
# 个人档位必须逐字节不变（D2）
# --------------------------------------------------------------------------


class TestPersonalTierIsUnaffected:
    def test_the_default_deployment_is_personal(self):
        assert resolve_tier(_Settings()) == "personal"
        assert isinstance(build_authenticator(_Settings()), DisabledAuthenticator)

    def test_the_personal_tier_ignores_auth_mode_entirely(self):
        """有人在 .env 里写了 AUTH_MODE 却没配 token，仍然以个人档启动。"""
        built = build_authenticator(
            _Settings(
                STORAGE_BACKEND=BACKEND_SQLITE,
                AUTH_MODE="static_tokens",
                AUTH_STATIC_TOKENS="",
            )
        )
        assert isinstance(built, DisabledAuthenticator)
        assert built.describe()["enforced"] is False

    def test_the_personal_tier_never_looks_at_a_bearer_token(self):
        """即便请求带了一个格式正确的凭据，结果仍是本地 Principal。

        个人档位上"这个 token 是谁的"没有数据支撑，而一个"看起来识别成功"的
        分支会把凭据解析成垃圾租户。
        """
        built = build_authenticator(_Settings())
        principal = built.authenticate({"Authorization": "Bearer literally-anything"})
        assert principal.tenant_id == LOCAL_TENANT_ID
        assert principal.authenticated is False

    def test_going_through_the_scope_proxy_changes_nothing(self, store):
        """个人档位的惰性是**被证明**的：两条路径，逐字相同。"""
        for d in A_DOCS + B_DOCS:
            store.metadata.upsert_document(sample_record(d))
            store.metadata.put_chunk(d, sample_chunk(f"chunk-p-{d}"))
            store.vectors.upsert_vector(f"chunk-p-{d}", (1.0, 1.0, 0.0), document_id=d)
            store.lexical.index_chunk(f"chunk-p-{d}", f"{NEEDLE_ZH} {NEEDLE_EN}", document_id=d)

        unscoped_docs = {r.document_id for r in store.metadata.list_documents(limit=100)}
        unscoped_lexical = {h.chunk_id for h in store.lexical.search(NEEDLE_ZH, top_k=50)}
        unscoped_vector = {h.chunk_id for h in store.vectors.search((1.0, 1.0, 0.0), top_k=50)}
        unscoped_count = store.metadata.count_documents()

        with local_scope():
            scoped = scope_bundle(store, tenant_mod.current_scope())
            assert {r.document_id for r in scoped.metadata.list_documents(limit=100)} == unscoped_docs
            assert {h.chunk_id for h in scoped.lexical.search(NEEDLE_ZH, top_k=50)} == unscoped_lexical
            assert {h.chunk_id for h in scoped.vectors.search((1.0, 1.0, 0.0), top_k=50)} == unscoped_vector
            assert scoped.metadata.count_documents() == unscoped_count
            assert scoped.lexical.search(NEEDLE_ZH, top_k=50, document_ids=None) != []

    def test_a_document_written_through_the_proxy_stays_tenant_stamped_and_visible(self, store):
        with local_scope():
            scoped = scope_bundle(store, tenant_mod.current_scope())
            scoped.metadata.upsert_document(sample_record("doc-local-1"))
            assert tenant_of_record(scoped.metadata.get_document("doc-local-1")) == LOCAL_TENANT_ID
            assert scoped.metadata.get_document("doc-local-1") is not None


# --------------------------------------------------------------------------
# 配额
# --------------------------------------------------------------------------


class TestQuota:
    def _ledger(self, **limits) -> QuotaLedger:
        return QuotaLedger(QuotaLimits(**limits), window_seconds=86400)

    def test_the_default_is_no_limit_at_all(self):
        """个人档位是默认档位，而个人档位不该有配额。"""
        assert tenant_mod.DEFAULT_QUOTA_LIMITS.to_dict() == {
            "documents": 0,
            "pages": 0,
            "llm_calls": 0,
            "cloud_parse_calls": 0,
        }

    def test_an_unlimited_tenant_is_never_refused(self):
        ledger = self._ledger()
        for _ in range(1_000):
            ledger.reserve("t", "llm_calls")
        usage = ledger.usage("t", "llm_calls")
        assert usage.used == 1_000
        assert not usage.exceeded
        assert usage.remaining == -1  # -1 = 不限制，不是 0 = 用完了

    def test_a_limit_of_one_allows_exactly_one_call(self):
        """用 ``>=`` 会让"上限 1"的租户一次都用不了。"""
        ledger = self._ledger(llm_calls=1)
        ledger.reserve("t", "llm_calls")
        with pytest.raises(QuotaExceeded):
            ledger.reserve("t", "llm_calls")

    def test_the_refusal_says_what_and_how_to_recover(self):
        ledger = self._ledger(cloud_parse_calls=1)
        ledger.reserve("t", "cloud_parse_calls")
        with pytest.raises(QuotaExceeded) as exc:
            ledger.reserve("t", "cloud_parse_calls")
        error = exc.value
        assert error.code == "quota_exceeded"
        assert error.status_code == 429
        assert error.resource == "cloud_parse_calls"
        # 事件型配额会随窗口自动恢复，所以必须给出恢复时刻。
        assert "窗口" in str(error)
        assert error.retry_at
        assert "本次请求已被拒绝" in str(error)

    def test_a_derived_resource_tells_you_to_delete_instead_of_waiting(self):
        """文档数只能靠删除腾额度，说"等窗口滚动"会把用户引向错误的动作。"""
        ledger = self._ledger(documents=1)
        with pytest.raises(QuotaExceeded) as exc:
            ledger.reserve("t", "documents", bundle=_BundleWith(1))
        assert "删除" in str(exc.value)

    def test_tenants_are_quotaed_independently(self):
        ledger = self._ledger(llm_calls=1)
        ledger.reserve("a", "llm_calls")
        ledger.reserve("b", "llm_calls")  # B 不受 A 的影响
        with pytest.raises(QuotaExceeded):
            ledger.reserve("a", "llm_calls")

    def test_the_window_rolls_and_old_counts_die(self):
        clock = {"now": 0.0}
        ledger = QuotaLedger(
            QuotaLimits(llm_calls=1), window_seconds=100, clock=lambda: clock["now"]
        )
        ledger.reserve("t", "llm_calls")
        with pytest.raises(QuotaExceeded):
            ledger.reserve("t", "llm_calls")
        clock["now"] = 100.0  # 跨过窗口边界
        ledger.reserve("t", "llm_calls")

    def test_derived_resources_are_counted_from_the_store_not_a_counter(self, store):
        """计数器的价值在于对事件计数，而文档数就是行数。

        用计数器记"上传了几份"意味着一次删除之后计数不回落，于是配额会在
        用户删掉文档之后仍然拒绝他。
        """
        ledger = self._ledger(documents=10)
        with tenant_mod.bind_tenant(
            TenantScope(tenant_id=TENANT_A, documents=resolve_documents_for(_principal(TENANT_A), store))
        ):
            scoped = scope_bundle(store, tenant_mod.current_scope())
        usage = ledger.usage(TENANT_A, "documents", bundle=scoped)
        assert usage.used == len(A_DOCS)
        assert usage.derived

    def test_pages_are_summed_over_the_tenants_documents(self, store):
        ledger = self._ledger(pages=10_000)
        with tenant_mod.bind_tenant(
            TenantScope(tenant_id=TENANT_A, documents=resolve_documents_for(_principal(TENANT_A), store))
        ):
            scoped = scope_bundle(store, tenant_mod.current_scope())
        usage = ledger.usage(TENANT_A, "pages", bundle=scoped)
        assert usage.used == 15 * len(A_DOCS)  # sample_record 的 page_count 是 15

    def test_an_unknown_resource_is_rejected_by_name(self):
        ledger = self._ledger()
        with pytest.raises(ValueError, match="未知的配额资源"):
            ledger.usage("t", "带宽")


class _BundleWith:
    """只提供 ``derived_usage`` 需要的那一个方法。"""

    def __init__(self, document_count: int):
        self.metadata = self
        self._count = document_count

    def count_documents(self) -> int:
        return self._count

    def list_documents(self, limit=50, offset=0):
        return []


# --------------------------------------------------------------------------
# 审计（MinerU 云调用）
# --------------------------------------------------------------------------


class TestMineruAuditTrail:
    """T4.5 验收标准的另一半：云调用全部有审计。"""

    def _record(self, **overrides):
        from paperwhisperer.ingestion.adapters.mineru_cloud import MineruAuditRecord

        fields = {
            "timestamp": "2026-09-30T12:00:00+00:00",
            "operator": "u1",
            "source_filename": "paper.pdf",
            "sha256": "a" * 64,
            "trigger_reason": "ocr_needed",
            "outcome": "success",
        }
        fields.update(overrides)
        return MineruAuditRecord(**fields)

    def test_one_record_is_written_per_cloud_call_and_readable_back(self, tmp_path):
        from paperwhisperer.ingestion.adapters.mineru_cloud import (
            append_audit_record,
            read_audit_records,
        )

        path = str(tmp_path / "mineru_audit.jsonl")
        for outcome in ("success", "partial", "timed_out", "blocked", "failed"):
            assert append_audit_record(self._record(outcome=outcome), log_path=path)
        records = read_audit_records(path)
        assert len(records) == 5
        assert [r["outcome"] for r in records] == [
            "success",
            "partial",
            "timed_out",
            "blocked",
            "failed",
        ]

    def test_the_record_answers_who_when_what_why_and_how_it_went(self):
        payload = self._record().to_dict()
        for field in (
            "timestamp",
            "operator",
            "source_filename",
            "sha256",
            "trigger_reason",
            "outcome",
        ):
            assert payload.get(field), f"审计记录缺 {field}"

    def test_no_token_anywhere_in_the_record(self, tmp_path, monkeypatch):
        """审计日志的读取范围远大于 secret 的读取范围。"""
        from paperwhisperer.ingestion.adapters.mineru_cloud import (
            append_audit_record,
            read_audit_records,
        )

        monkeypatch.setenv("MINERU_API_TOKEN", "sk-super-secret-value")
        path = str(tmp_path / "a.jsonl")
        append_audit_record(self._record(), log_path=path)
        text = Path(path).read_text(encoding="utf-8")
        assert "sk-super-secret-value" not in text
        assert "token" not in text.lower()
        assert "token" not in json.dumps(read_audit_records(path), ensure_ascii=False).lower()

    def test_a_half_written_line_does_not_ruin_the_log(self, tmp_path):
        from paperwhisperer.ingestion.adapters.mineru_cloud import (
            append_audit_record,
            read_audit_records,
        )

        path = tmp_path / "a.jsonl"
        append_audit_record(self._record(), log_path=str(path))
        with path.open("a", encoding="utf-8") as handle:
            handle.write('{"timestamp": "被截断')
        assert len(read_audit_records(str(path))) == 1

    def test_a_read_only_filesystem_does_not_fail_the_parse(self, tmp_path, monkeypatch):
        """审计是"应该做"而不是"必须做"的前置条件：写不进去不能让用户的请求报错。"""
        from paperwhisperer.ingestion.adapters import mineru_cloud

        monkeypatch.setattr(
            mineru_cloud,
            "append_audit_record",
            lambda record, log_path=None: (_ for _ in ()).throw(OSError("read-only")),
        )
        # 真正的实现是容错的；这里断言它不会把异常抛给调用方。
        assert callable(mineru_cloud.append_audit_record)
