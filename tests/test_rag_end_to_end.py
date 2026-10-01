"""The chain, end to end, over a real corpus: parse → chunk → embed → index → retrieve.

This file exists because the whole vector path was built, reviewed and unit
tested against a provider that had never once answered.
``config.EMBEDDING_API_BASE_URL`` points at an endpoint whose
``/v1/embeddings`` route returns 404, so every layer below was exercised
separately with fakes and never together: the store's cosine, the chunker's
boundaries, the retriever's RRF -- all tested, none composed. The dense leg has
been abstaining for the life of the project and reporting a degradation into a
log line nobody read, and every retrieval number in this repository has been
lexical-only. The gap this closes is not a missing feature; it is a missing
*measurement*.

What is real here and what is not:

* **Real.** The parser (the ingestion gateway, on the personal-tier ``legacy``
  route -- the configured default), the structure-aware chunker, the embedding
  gateway's own batching/cache/usage code, an actual ONNX model loaded from
  actual weights, a real SQLite store, the real hybrid retriever, and questions
  whose answers were read out of the corpus text rather than invented.
* **Not real.** No LLM, so nothing past the retrieval boundary is exercised.
  Embedding a chunk is the part that had never run; answering a question is
  not what this file is about.
* **Deliberately not asserted.** Recall numbers are *recorded* and printed, and
  the assertions are floors and structural facts. A test that asserted an exact
  recall would fail the day someone swaps the model, and the fix would be to
  edit the number -- which is the failure mode a measurement is supposed to
  prevent. The floors are what would catch a broken chain rather than a
  different model.

**What the measurement says, as of this writing.** Over 5 real papers (87
chunks, 5 documents), with the multilingual MiniLM at 384 dimensions:

============================  ========  ========  ================
question set (7 each)         R@1       R@8       dense leg
============================  ========  ========  ================
exact-term, hybrid            0.43      1.00      voted 7/7
exact-term, lexical only      0.86      1.00      --
paraphrase, hybrid            0.29      0.43      voted 7/7
paraphrase, lexical only      0.14      0.29      --
paraphrase, dense only        0.29      0.71      --
============================  ========  ========  ================

**These numbers cannot carry the weight a pass/fail gate would put on them, and
this file no longer asks them to.** Seven questions is not a sample. One
question is 0.14 of recall; the 95% Wilson interval on any of these cells is
roughly ±0.30 wide. The hybrid-vs-lexical gap on paraphrase questions — the one
that was once asserted with a strict ``>`` — rests on **exactly one question**
out of seven flipping (Q5, "which training set was not publicly released");
lexical-only already answers two of the seven on its own, and the dense leg
adds one more. Leave that single question out and the asserted gap goes to
exactly 0.000. An exact McNemar test on the paired outcomes gives **p = 1.000**:
the paired design at n=7 cannot distinguish the two systems at any confidence.
Even the large dense-only effect (0.71 vs 0.29) has only three discordant pairs
and comes out at p = 0.25.

So the assertions in this file are capabilities — the dense leg is present, it
abstains never here, and its votes are load-bearing (turning it on reorders the
fused output on 7/7 paraphrase questions) — and the recall deltas are recorded
in ``docs/RAG_MEASUREMENT.md`` with their sample size and interval attached.
A capability assertion survives a corpus refresh or a model swap; a 0.14
delta on n=7 does not, and asserting it only buys a test that fails at random.

**The measurement is deterministic.** Twelve consecutive runs of this file
produced byte-identical recall tables, and ``embed_query`` returns bit-identical
vectors batched or one-at-a-time (max abs diff 0.0), so the model is *not* a
source of run-to-run variance. What does move is the **corpus**, which is
gitignored and fetched from `https://arxiv.org/pdf/<id>` with no version pin
(``eval/download_arxiv.sh``): a different arXiv revision shifts chunk
boundaries, which moves which chunk counts as gold, which flips a question.
That is the honest explanation for the intermittent
"lexical only 0.43 -> hybrid 0.43" failure — not a regression in the dense leg,
and not model noise. The effect is unchanged and was re-confirmed after the
edit.

Three findings, all worth more than a green checkmark:

**The dense leg earns its cost on paraphrase and pays for it on exact terms.**
On paraphrased questions -- the case it exists for, where the query shares no
rare term with the passage -- adding it lifts recall@8 from 0.29 to 0.43. On
exact-term questions, which is the case the plan explicitly designs the lexical
leg for ("专有名词/DOI/数据集名精确查询稳定"), the same addition drops
recall@1 from 0.86 to 0.43 while recall@8 does not move at all. It is not free
and it is not useless; it is a trade, made silently, at equal leg weights. The
direction of both moves is real; the magnitude is not established at this n.

**The dense leg was never abstaining on this corpus.** The tie-band
abstention in :mod:`paperwhisperer.rag.retrieve` fired 0 times in 14 queries
(measured spread across the tie window: 0.10–0.20, against a 0.02 band — not
close). That is the good news and the bad news: the model has an opinion every
time, and on exact-term questions the opinion is wrong often enough to demote
passages the lexical leg had already ranked first. Those passages were rank 1 by
lexical and *absent from the dense leg's top 40*, so nothing vetoed them -- 39
chunks the model preferred each contributed a rank to the fusion. A
confidence-based abstention catches a model that has no opinion; this model has
a confident wrong one. ``dense_weight`` is the dial, and it is not this file's
to turn.

**This file is not the release gate, and must not be read as one.** T4.3's gate
is Recall@K >= 90% with Chinese, English, table, formula and prose counted
separately, and ``docs/RELEASE_GATES.md`` records that gate as **BLOCKED** on
the absence of a labelled set (``eval/labels/retrieval_run.json``). Nothing here
counts those categories separately, nothing here reaches 90%, and a 7-question
sample cannot stand in for a gate that is, correctly, still blocked. The gap
between this file and the gate is the labelling work, and it is roughly 100
labelled questions to put a ±5% interval on a number near 90%.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from paperwhisperer.core import config
from paperwhisperer.llm.embeddings import (
    EmbeddingBatch,
    EmbeddingCache,
    EmbeddingDimensionMismatchError,
    EmbeddingGateway,
)
from paperwhisperer.llm.embeddings_local import (
    LOCAL_PROVIDER_NAME,
    LocalEmbeddingProvider,
    build_embedder_from_config,
    fastembed_installed,
    local_identity,
)

CORPUS = Path(__file__).resolve().parents[1] / "eval" / "corpus"

#: Five real papers spanning the corpus's shapes: a bilingual synthetic report, a
#: single-column paper, a two-column paper, a formula-dense paper and a
#: bilingual survey. Four of the five chunk into a realistic number of pieces;
#: the fifth chunks to one, which is itself a fact worth having in the store --
#: a document shorter than one token budget is a legitimate index entry, not a
#: bug, and a retriever that only ever sees multi-chunk documents has never been
#: asked to rank against one.
CORPUS_FILES = (
    "mixed_zh_en/tech_report_zh_terms.pdf",
    "digital_single_col/attention_is_all_you_need.pdf",
    "digital_single_col/vit_an_image_is_worth_16x16.pdf",
    "digital_single_col/word2vec.pdf",
    "digital_two_col/resnet.pdf",
)

#: Questions whose answers contain a string that can be *located* in the corpus.
#: Locatable rather than labelled on purpose: a hand-assigned chunk id goes stale
#: the moment a chunk boundary moves, and a test that then fails is a test
#: nobody trusts. The gold string is checked against the indexed chunk text on
#: every run, so a corpus that stopped containing it fails loudly and says so
#: instead of quietly reporting a recall drop.
@dataclass(frozen=True)
class Question:
    text: str
    gold: str
    #: ``"exact"`` questions quote a term the passage contains; ``"paraphrase"``
    #: questions deliberately share no rare term with the answer sentence. The
    #: two sets are the two regimes the retriever's two exact legs and its one
    #: dense leg are each good at, and averaging them hides both.
    kind: str


EXACT_TERM_QUESTIONS = (
    Question(
        "What tokenizer was used for the WMT 2014 English-German training data?",
        "byte-pair encoding",
        "exact",
    ),
    Question(
        "What regularization hurts perplexity but improves BLEU?",
        "label smoothing of value",
        "exact",
    ),
    Question(
        "How is the learning rate scheduled over the course of training?",
        "increasing the learning rate linearly for the first",
        "exact",
    ),
    Question(
        "How does word2vec avoid the cost of a full softmax over the vocabulary?",
        "hierarchical softmax",
        "exact",
    ),
    Question(
        "Which in-house dataset was ViT pre-trained on to approach state of the art?",
        "JFT-300M",
        "exact",
    ),
    Question(
        "What did residual learning do to the top-1 error?",
        "ResNet reduces the top-1 error by 3.5%",
        "exact",
    ),
    # Chinese, because a monolingual model would return its prior for every
    # passage here and the tie-band abstention would (correctly) fire. Asserting
    # that the dense leg *votes* on a Chinese query is asserting that the
    # multilingual choice in config was the right one.
    Question("哪个服务的 p95 延迟最高，主要由哪部分贡献？", "p95 延迟为 8,600 ms", "exact"),
)

PARAPHRASE_QUESTIONS = (
    Question(
        "What wraps each of the two sub-layers so that gradients still flow?",
        "LayerNorm(x + Sublayer(x))",
        "paraphrase",
    ),
    Question(
        "Which kind of skip connection wins once the network gets very deep?",
        "parameter-free, identity shortcuts",
        "paraphrase",
    ),
    Question(
        "For the English-French corpus, what was the size of the joint vocabulary?",
        "shared source- target vocabulary of about 37000",
        "paraphrase",
    ),
    Question(
        "How much better is the deep model than a plain one of similar depth on "
        "single-crop accuracy?",
        "ResNet reduces the top-1 error by 3.5%",
        "paraphrase",
    ),
    Question(
        "Which training set was not publicly released that the authors trained on?",
        "in-house JFT-300M",
        "paraphrase",
    ),
    Question(
        "What happens to the queue backlog over the course of the morning?",
        "队列积压峰值出现在每日 09:00-11:00",
        "paraphrase",
    ),
    Question(
        "What is the plan for the document parsing queue next quarter?",
        "OCR 队列拆分与解析缓存",
        "paraphrase",
    ),
)

#: Ranks kept for a recall@8 measurement. The Q&A plan's ``final_k`` is 8, so
#: recall@8 is "would the answer have been shown to the model at all", which is
#: the question that matters before any reranker or claim builder runs.
TOP_K = 8

#: Set this to 1 to turn a weight-download failure into a failure instead of a
#: skip. A skip is the right default -- a checkout with no network is a
#: supported configuration, not a broken one -- but a machine that *should* have
#: the weights and does not is a bug, and the only way to tell the two apart is
#: to say so out loud.
REQUIRE_WEIGHTS_ENV = "PAPERWHISPERER_REQUIRE_LOCAL_EMBEDDINGS"


def _normalise(text: str) -> str:
    """Whitespace-insensitive text, so a gold phrase spans lines without care.

    The parser emits PDF line breaks mid-sentence, so a phrase that is
    contiguous in a reader's eye is not contiguous in the stored chunk. Removing
    all whitespace is the only comparison that survives that, and it is applied
    to both sides so it cannot manufacture a match that is not there.
    """
    return re.sub(r"\s+", "", str(text or ""))


def _skip_unless_usable() -> None:
    """Skip with a reason that says what to install and what to configure.

    Two distinct reasons, because they are two different situations and
    conflating them produces a green test suite that has proved nothing: the
    extra is not installed (a supported deployment shape), or the extra is
    installed but the weights cannot be fetched (an environment problem, which
    the mirror usually fixes).
    """
    if not fastembed_installed():
        pytest.skip(
            "本地 embedding extra 未安装：安装 paperwhisperer[local-embeddings] 可运行本文件；"
            "不安装是一个受支持的部署形态（D3 的默认路径是 EMBEDDING_API_BASE_URL），"
            "所以这里跳过而不是失败。"
        )


@pytest.fixture(scope="session")
def local_config():
    """Point the chunker's index identity at the local provider.

    ``core.config`` resolves ``EMBEDDING_PROVIDER``/``EMBEDDING_MODEL``/
    ``EMBEDDING_DIMENSION`` to the local provider's values when
    ``EMBEDDING_LOCAL_ENABLED`` is set, and those three are exactly what
    :func:`paperwhisperer.rag.chunking.chunk_document_from_config` reads to
    build ``index_version``. Patching them here rather than passing explicit
    arguments to the chunker is the point of the exercise: it is the
    *configuration* that has to carry the identity, so that a provider switch
    changes the index version without any code change. A test that passed the
    provider explicitly would pass on a build whose config was wrong.

    A session-scoped fixture cannot take pytest's function-scoped ``monkeypatch``
    -- their lifetimes are the wrong way round -- so the patcher is created and
    torn down by hand and the resolved identity is returned for the
    function-scoped tests to read. The yield is what makes the undo happen at
    all; without it these three module attributes would stay patched for every
    test in the session, including the ones asserting the vendored path.
    """
    _skip_unless_usable()
    provider, model, dimension = local_identity()
    patcher = pytest.MonkeyPatch()
    patcher.setattr(config, "EMBEDDING_LOCAL_ENABLED", True)
    patcher.setattr(config, "EMBEDDING_PROVIDER", provider)
    patcher.setattr(config, "EMBEDDING_MODEL", model)
    patcher.setattr(config, "EMBEDDING_DIMENSION", dimension)
    try:
        yield provider, model, dimension
    finally:
        patcher.undo()


@pytest.fixture(scope="session")
def provider(local_config) -> LocalEmbeddingProvider:
    """One real local provider for the whole session.

    Session-scoped because the model load is the expensive part and because
    measuring "how long does indexing take" on a run that reloaded 220 MB of
    ONNX weights five times would measure the wrong thing. A private cache keeps
    one test's texts from becoming another's cache hits.
    """
    try:
        return LocalEmbeddingProvider(cache=EmbeddingCache(max_entries=4096))
    except Exception as exc:  # noqa: BLE001 - the failure is a setup failure
        if os.getenv(REQUIRE_WEIGHTS_ENV) == "1":
            raise
        pytest.skip(
            f"本地 embedding 模型不可用：{exc}\n"
            "首次使用需要下载权重；网络受限时设置 HF_ENDPOINT=https://hf-mirror.com 后重试，"
            f"或设置 {REQUIRE_WEIGHTS_ENV}=1 让这里失败而不是跳过。"
        )


class _Settings:
    """A store configuration that cannot reach a developer database."""

    STORAGE_BACKEND = "sqlite"
    STORAGE_LEXICAL_FTS5 = "auto"
    RERANK_ENABLED = False
    RERANK_MODEL = ""

    def __init__(self, path: str) -> None:
        self.STORAGE_SQLITE_PATH = path


@dataclass
class Indexed:
    """A parsed, chunked, embedded, indexed corpus, plus what it cost."""

    bundle: object
    provider: LocalEmbeddingProvider
    documents: dict[str, str]
    #: The parsed documents themselves, by id. Kept because a test that needs to
    #: re-run indexing needs a ``ParsedDocument`` and re-parsing would be a
    #: second, possibly different, parse -- which is the ambiguity
    #: ``index_document_for_retrieval`` exists to avoid.
    parsed: dict[str, object]
    chunks_by_id: dict[str, object]
    parse_seconds: float
    index_seconds: float
    embed_seconds: float
    batch_count: int
    text_count: int
    input_tokens: int


@pytest.fixture(scope="session")
def indexed(provider, local_config, tmp_path_factory) -> Indexed:
    """Run the whole chain once, over five real papers, and keep the result.

    The import of the parse and indexing functions is deferred to inside the
    fixture so that a session where :func:`_skip_unless_usable` already skipped
    never pays for importing the ingestion stack.
    """
    from paperwhisperer.analysis.orchestrator import index_document_for_retrieval
    from paperwhisperer.infra.stores import build_sqlite_bundle
    from paperwhisperer.ingestion.orchestrator import parse_document

    bundle = build_sqlite_bundle(_Settings(str(tmp_path_factory.mktemp("rag") / "stores.sqlite3")))
    documents: dict[str, str] = {}
    parsed: dict[str, object] = {}
    parse_seconds = 0.0
    index_seconds = 0.0
    usage = provider.reset_usage()
    try:
        for relative in CORPUS_FILES:
            path = CORPUS / relative
            if not path.is_file():
                pytest.skip(f"corpus file is missing: {path}")
            started = time.perf_counter()
            outcome = parse_document(str(path), allow_cloud_fallback=False)
            parse_seconds += time.perf_counter() - started
            assert outcome.document is not None, f"{relative} did not parse"
            started = time.perf_counter()
            count = index_document_for_retrieval(bundle, outcome.document, provider)
            index_seconds += time.perf_counter() - started
            documents[relative] = str(outcome.document.document_id)
            parsed[relative] = outcome.document
            assert count > 0, f"{relative} produced no chunks"
    finally:
        # Read the ledger even on the failure path: a test that fails halfway
        # through indexing still wants the numbers that explain how far it got,
        # and a bare ``usage`` in the try block would not survive the skip.
        usage = provider.reset_usage()

    chunks = {}
    for record in bundle.metadata.list_documents(limit=len(CORPUS_FILES) + 4):
        for chunk in bundle.metadata.list_chunks(record.document_id):
            chunks[chunk.chunk_id] = chunk
    return Indexed(
        bundle=bundle,
        provider=provider,
        documents=documents,
        parsed=parsed,
        chunks_by_id=chunks,
        parse_seconds=parse_seconds,
        index_seconds=index_seconds,
        embed_seconds=provider.embed_seconds,
        batch_count=usage.requests,
        text_count=usage.texts,
        input_tokens=usage.input_tokens,
    )


def _gold_chunk_ids(indexed: Indexed, question: Question) -> tuple[str, ...]:
    """Every chunk that literally contains the answer.

    A *set*, not a single id, and the reason is that real papers repeat
    themselves: ``JFT-300M`` appears in ten separate chunks of the ViT paper and
    ``byte-pair encoding`` in two of the Transformer's. "Did retrieval return a
    passage that answers the question" is a question about any of them. Scoring
    against one arbitrarily-first chunk would report a miss for a retrieval that
    returned the answer sitting at rank 1 -- a false negative in a measurement
    whose entire value is that it can be believed.

    Scanning the stored chunk text rather than the parsed document, because the
    chunk is what the retriever can return: a phrase that is in the document but
    was cut across a boundary is in no chunk, and asking whether retrieval found
    it would be asking about a passage that does not exist.
    """
    needle = _normalise(question.gold)
    return tuple(
        chunk_id
        for chunk_id, chunk in indexed.chunks_by_id.items()
        if needle in _normalise(chunk.text)
    )


def _recall(indexed: Indexed, questions, *, plan_overrides=None, top_k=TOP_K):
    """``(recall@1, recall@8)`` plus the retrieval records, for one leg mix."""
    from paperwhisperer.rag.retrieve import HybridRetriever, plan_for

    retriever = HybridRetriever(indexed.bundle, embeddings=indexed.provider, reranker=None)
    plan = plan_for("qa")
    if plan_overrides:
        plan = plan.with_overrides(**plan_overrides)
    golds = {question.text: _gold_chunk_ids(indexed, question) for question in questions}
    missing = [q.gold for q in questions if not golds[q.text]]
    assert not missing, (
        f"这些答案串在已索引的 chunk 文本里找不到，语料或切分变了：{missing}。"
        "定位不到答案的检索测试等于没有测试，所以这里直接失败而不是记一次未命中。"
    )

    at_one = 0
    at_k = 0
    records = []
    for question in questions:
        result = retriever.retrieve(question.text, plan=plan, top_k=top_k)
        records.append(result.record)
        ranks = [evidence.chunk_id for evidence in result.evidence]
        gold = set(golds[question.text])
        if ranks and ranks[0] in gold:
            at_one += 1
        if gold.intersection(ranks):
            at_k += 1
    count = len(questions)
    return at_one / count, at_k / count, records


def _one_order(indexed: Indexed, question: Question, *, dense_weight: float) -> tuple[str, ...]:
    """The fused top-8 order for one question at one ``dense_weight``.

    Separate from :func:`_recall` because it answers a *capability* question
    ("does turning the dense leg on change the ranking at all?") that needs the
    order, not the hit count. Recall aggregates away exactly the signal this
    checks: a leg can move every result within the window and leave recall
    unchanged, and a leg can leave the order untouched and change nothing that
    anyone would notice either. The order is the thing fusion actually produces,
    so it is the thing worth asserting on.
    """
    from paperwhisperer.rag.retrieve import HybridRetriever, plan_for

    retriever = HybridRetriever(indexed.bundle, embeddings=indexed.provider, reranker=None)
    plan = plan_for("qa").with_overrides(dense_weight=dense_weight)
    result = retriever.retrieve(question.text, plan=plan, top_k=TOP_K)
    return tuple(evidence.chunk_id for evidence in result.evidence)


# --------------------------------------------------------------------------
# The chain
# --------------------------------------------------------------------------


def test_chain_runs_end_to_end_and_finds_the_right_passage(indexed: Indexed):
    """Parse → chunk → embed → index → retrieve, and the answer comes back.

    The load-bearing assertion is that this ran at all: every layer of this path
    had a unit test and the composition had none. The recall floor is a
    backstop against a chain that completes but retrieves nothing -- a silent
    wiring error, a store that accepted writes and returned nothing, a model
    returning a constant vector -- all of which "did not raise" would miss.
    """
    at_one, at_k, _ = _recall(indexed, EXACT_TERM_QUESTIONS)
    assert at_k == 1.0, (
        f"精确词查询的 recall@8 只有 {at_k:.2f}：链路跑完了，但没有把答案段落交出来。"
        "这通常意味着索引里没有向量、维度对不上、或检索范围为空。"
    )
    # A liveness floor, not a quality bar. It is deliberately far below the
    # measured 0.43 and far below what the lexical leg alone reaches (0.86): its
    # job is to catch a chain that completes and returns nothing, nothing first,
    # or the same chunk seven times. Pinning it near the current number would
    # make a model change look like a regression, and the fix would be to edit
    # the number -- which is what a measurement must not become. The real
    # rank-1 trade-off is measured, not asserted, in
    # :func:`test_dense_leg_contribution_versus_lexical_only`.
    assert at_one >= 0.25, (
        f"精确词查询的 recall@1 只有 {at_one:.2f}，这个链路的排序几乎失效了"
        "（见 test_dense_leg_contribution_versus_lexical_only 的实测表）。"
    )

    assert indexed.bundle.lexical.count_chunks() > 50, "语料太小，测不出跨文档排序。"
    assert indexed.bundle.vectors.count_vectors() == indexed.bundle.lexical.count_chunks(), (
        "词法索引与向量索引的 chunk 数不一致：有一侧的写入静默失败了。"
    )


def test_dense_leg_is_live_and_its_vectors_are_usable(indexed: Indexed):
    """The dense leg votes, and what it stored is a real unit vector.

    ``None`` was the value this project has been shipping: the retriever
    appends "未配置 embedding provider，dense 通道未参与本次检索" and answers
    lexically. Asserting the leg is *present* is the point; asserting it is
    present on a Chinese query too is what distinguishes a multilingual model
    from a monolingual one returning its prior.
    """
    from paperwhisperer.rag.retrieve import HybridRetriever, plan_for

    retriever = HybridRetriever(indexed.bundle, embeddings=indexed.provider, reranker=None)
    result = retriever.retrieve(EXACT_TERM_QUESTIONS[0].text, plan=plan_for("qa"), top_k=TOP_K)
    assert not [d for d in result.record.degradations if "dense" in d], (
        f"dense 通道没有参与检索：{result.record.degradations}"
    )
    # The abstention is a *tie*, not an absence. If it ever starts firing on
    # every query the dense leg is decorative again, and the number that says so
    # is the score spread, not the degradation string.
    scores = [
        candidate.legs[0].score
        for candidate in result.record.candidates
        if candidate.legs and candidate.legs[0].leg == "dense"
    ]
    assert scores, "记录里没有 dense 腿的分数，无法判断它是否给出了真实排序。"

    entry = indexed.bundle.vectors.get_vector(
        next(iter(indexed.chunks_by_id))
    )
    assert entry is not None
    assert len(entry.vector) == int(indexed.provider.dimension)
    norm = sum(value * value for value in entry.vector) ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-5), (
        f"存进 store 的向量不是单位向量（模长 {norm:.6f}）。"
        "余弦相似度在单位向量上才等于点积，store 的排序依赖这一点。"
    )


def test_dense_leg_contribution_versus_lexical_only(indexed: Indexed):
    """What the dense leg actually buys, measured on both question regimes.

    The two facts worth protecting here are both about the *leg participating in
    fusion*, not about a measured recall delta:

    * the dense leg is present and abstains never -- on this corpus it has an
      opinion about every question, and every one of those opinions is fused;
    * the dense leg's votes are load-bearing -- adding the leg at
      ``dense_weight > 0`` reorders the fused output relative to
      ``dense_weight = 0`` on every question. This is the capability: a dense leg
      that runs but never changes the answer is decoration.

    The recall numbers are measured and *recorded* (printed, and preserved in
    ``docs/RAG_MEASUREMENT.md``), not asserted. A measured recall delta is not a
    capability this sample can certify, and pretending otherwise is how a
    measurement gets quietly deleted. The numbers stand on their own in the
    report; the test's job is to say the leg is live and fused, which is a fact
    that survives a corpus refresh and a model swap.

    The one assertion that is a performance claim -- exact-term recall@8 must not
    get *worse* -- is kept, because it is a non-regression guard rather than a
    quality bar: it can only fire when something is broken, and it holds today at
    the ceiling (1.00 vs 1.00).
    """
    exact_hybrid = _recall(indexed, EXACT_TERM_QUESTIONS)
    exact_lexical = _recall(indexed, EXACT_TERM_QUESTIONS, plan_overrides={"dense_weight": 0.0})
    paraphrase_hybrid = _recall(indexed, PARAPHRASE_QUESTIONS)
    paraphrase_lexical = _recall(
        indexed, PARAPHRASE_QUESTIONS, plan_overrides={"dense_weight": 0.0}
    )
    paraphrase_dense = _recall(
        indexed,
        PARAPHRASE_QUESTIONS,
        plan_overrides={"lexical_weight": 0.0, "fuzzy_weight": 0.0},
    )

    rows = [
        ("exact-term", "hybrid", exact_hybrid),
        ("exact-term", "lexical only", exact_lexical),
        ("paraphrase", "hybrid", paraphrase_hybrid),
        ("paraphrase", "lexical only", paraphrase_lexical),
        ("paraphrase", "dense only", paraphrase_dense),
    ]
    lines = [
        "",
        f"--- dense leg contribution, {len(indexed.chunks_by_id)} chunks "
        f"over {len(indexed.documents)} documents ---",
        f"{'question set':<14}{'leg mix':<15}{'R@1':>6}{'R@8':>6}",
    ]
    for label, mix, (at_one, at_k, _records) in rows:
        lines.append(f"{label:<14}{mix:<15}{at_one:>6.2f}{at_k:>6.2f}")
    lines.append("--- end ---")
    lines.append(
        f"note: n={len(PARAPHRASE_QUESTIONS)} per set. recall@8 的 Wilson 95% CI "
        "半宽约 ±0.30，一个问题的权重就是 0.14；这些数字记录在 "
        "docs/RAG_MEASUREMENT.md，不作为断言。见 test 的 docstring。"
    )
    print("\n".join(lines))

    # --- Capability assertions. These are facts about the leg, not about a
    # measured delta, and they are what actually protect the dense path.

    abstentions = [
        q.text
        for q in PARAPHRASE_QUESTIONS + EXACT_TERM_QUESTIONS
        if any(
            "dense" in d and ("弃权" in d or "未参与" in d or "失败" in d)
            for d in _recall(indexed, [q])[2][0].degradations
        )
    ]
    assert not abstentions, (
        "dense 通道在这些问题上弃权或降级了："
        f"{[t[:40] for t in abstentions]}。"
        "这比 recall 下降更值得先查——通道根本没投票，测出来的任何差值都是"
        "词法通道自己的数。"
    )

    # The dense leg's votes must be load-bearing: with dense_weight=0 the fused
    # order is the lexical+fuzzy fusion, and turning the leg on must change it.
    # Checked on the paraphrase set, the regime the leg exists for. This is the
    # capability claim that "it participates in fusion" actually means, and it
    # is far more robust than a recall delta: it does not depend on which
    # questions happen to land inside the top-8.
    reordered = 0
    for question in PARAPHRASE_QUESTIONS:
        with_dense = _one_order(indexed, question, dense_weight=1.0)
        without_dense = _one_order(indexed, question, dense_weight=0.0)
        if with_dense != without_dense:
            reordered += 1
    assert reordered == len(PARAPHRASE_QUESTIONS), (
        f"只有 {reordered}/{len(PARAPHRASE_QUESTIONS)} 条改述问题在开启 dense 通道后"
        "改变了融合排序。dense 腿在跑，但没有参与排序——它现在是装饰。"
        "如果刚换了模型，先确认它不是一直在弃权（见上面的 abstentions 断言）。"
    )

    # Non-regression guard on the exact-term set, which the lexical leg owns.
    # This is the one place a performance comparison is legitimate to assert:
    # both arms sit at the 1.00 ceiling, so it can only fire if a change *broke*
    # exact-term retrieval. It is not evidence the dense leg helps.
    assert exact_hybrid[1] >= exact_lexical[1], (
        "加入 dense 通道把精确词查询的 recall@8 弄坏了："
        f"{exact_lexical[1]:.2f} -> {exact_hybrid[1]:.2f}。"
        "词法腿是精确串查询的答案来源，dense 腿只能排序不能替换它。"
    )


# --------------------------------------------------------------------------
# Cost
# --------------------------------------------------------------------------


def test_embedding_and_indexing_are_batched_and_timed(indexed: Indexed):
    """The cost side, measured, with the batching asserted rather than hoped for.

    The per-chunk loop this replaced would have made ``batch_count == text
    count``. Asserting the inequality is the durable version of "batching
    happened": it holds whatever the corpus size, whereas a wall-clock number
    only holds on one machine.

    ``embed_seconds`` is the honest cost of a local provider -- CPU time, not
    money -- and it is why the usage ledger's ``cost`` being ``0.0`` is not the
    whole story. Both are recorded, neither is asserted against a threshold,
    because a threshold on either would be a statement about this machine.
    """
    assert indexed.text_count > 50, "语料太小，批处理与否看不出来。"
    assert indexed.batch_count < indexed.text_count, (
        f"{indexed.text_count} 段文本发了 {indexed.batch_count} 次请求："
        "embedding 没有走批处理路径，每段一次调用。"
    )
    assert indexed.input_tokens > 0, "没有统计到 token，成本账目会是 0 而不是未知。"
    print(
        f"\n--- cost: {len(indexed.documents)} docs, {indexed.text_count} chunks, "
        f"{indexed.batch_count} batched calls, {indexed.input_tokens} tokens ---\n"
        f"    parse    {indexed.parse_seconds:6.2f}s\n"
        f"    index    {indexed.index_seconds:6.2f}s  (of which inference "
        f"{indexed.embed_seconds:.2f}s)\n"
        f"    per chunk{indexed.index_seconds / max(indexed.text_count, 1):6.3f}s\n"
        f"--- end ---"
    )


# --------------------------------------------------------------------------
# Dimension and index identity
# --------------------------------------------------------------------------


def test_index_version_names_the_local_provider_and_model(local_config):
    """The index version moves when the provider or the model moves.

    Plan section 8's risk as code. Both version strings are checked -- the
    gateway's, which the retriever records, and the chunker's, which the store's
    idempotence check compares -- because a change to only one of them is a
    configuration state where the index claims a space it does not hold.
    """
    from paperwhisperer.rag.chunking import build_index_version, index_version_parts

    expected_provider, expected_model, expected_dimension = local_config
    gateway = LocalEmbeddingProvider()
    # The gateway slugs its identity (``_slug`` in llm/embeddings.py), so the
    # underscore in the provider name becomes a hyphen. Asserting on the raw
    # name here would be asserting that the gateway stopped slugging, which is
    # a change to the *other* module's contract and not something this test is
    # entitled to require.
    slug = re.sub(r"[^a-z0-9]+", "-", expected_provider.lower()).strip("-")
    assert gateway.index_version.startswith(f"emb-{slug}-")
    assert f"d{expected_dimension}" in gateway.index_version
    model_slug = re.sub(r"[^a-z0-9]+", "-", expected_model.lower()).strip("-")
    assert model_slug in gateway.index_version

    chunk_version = build_index_version(
        provider=expected_provider, model=expected_model, dim=expected_dimension
    )
    parts = index_version_parts(chunk_version)
    assert parts["provider"] == expected_provider
    assert parts["model"] == expected_model
    assert parts["dim"] == expected_dimension

    # The config the *chunker* reads must agree with the provider that answered.
    # This is the assertion that would have caught a provider switch that left
    # EMBEDDING_MODEL pointing at the old vendor.
    assert gateway.provider == config.EMBEDDING_PROVIDER
    assert gateway.model == config.EMBEDDING_MODEL
    assert gateway.dimension == config.EMBEDDING_DIMENSION

    # And a different provider must produce a different version, or an old index
    # would be trusted rather than rebuilt.
    other = build_index_version(
        provider="openai_compatible", model=expected_model, dim=expected_dimension
    )
    assert other != chunk_version
    with pytest.raises(EmbeddingDimensionMismatchError):
        gateway.assert_same_space(other)


def test_dimension_mismatch_is_refused_loudly(monkeypatch, local_config):
    """A configured width the model does not produce is a refusal, not a guess.

    The alternative -- accepting the configured number and letting the store
    discover the truth at query time -- produces a store full of 384-wide
    vectors labelled ``d1536``, which answers every query with a confident
    meaningless number. Refusing at construction costs one clear message.
    """
    _provider, _model, _dimension = local_config
    monkeypatch.setattr(config, "EMBEDDING_DIMENSION", 1536)
    monkeypatch.setattr(config, "EMBEDDING_LOCAL_ENABLED", False)
    with pytest.raises(EmbeddingDimensionMismatchError) as caught:
        local_identity()
    message = str(caught.value)
    assert "384" in message and "1536" in message, (
        f"错误信息必须写出两个数字，运维才知道要改哪一个：{message}"
    )


def test_loaded_model_width_is_verified_not_assumed(monkeypatch, local_config):
    """The registry is a table; the loaded model is the fact. The fact wins.

    ``fastembed``'s registry declares 384 for this model, and a private ONNX
    build served under a registry name would be 384 too -- until it is not. The
    check here is the one that runs against the loaded session, and it is the
    last line of defence before vectors reach the store.
    """
    provider = LocalEmbeddingProvider()
    assert provider.native_dimension() == int(provider.dimension)

    monkeypatch.setattr(provider, "dimension", 1536)
    with pytest.raises(EmbeddingDimensionMismatchError):
        provider._verify_loaded_width()


# --------------------------------------------------------------------------
# Gateway conformance
# --------------------------------------------------------------------------


def test_local_provider_is_the_existing_gateway(provider: LocalEmbeddingProvider):
    """The contract is the gateway's, not a parallel one.

    Everything asserted here is behaviour the base class already implements, and
    asserting it against the local subclass is the test that a future edit to
    the transport has not quietly stopped being a gateway: the content-hash
    cache, the batch split, the usage ledger, the empty-text refusal and the
    dimension check all belong to
    :class:`paperwhisperer.llm.embeddings.EmbeddingGateway`, and the local
    provider inherits rather than reimplements them.
    """
    assert isinstance(provider, EmbeddingGateway)

    batch = provider.embed_documents(["alpha passage", "beta passage", "alpha passage"])
    assert isinstance(batch, EmbeddingBatch)
    assert len(batch) == 3
    assert batch[0] == batch[2], "同一段文本必须产生同一个向量（缓存键是内容哈希）。"
    assert len(batch[0]) == int(provider.dimension)

    again = provider.embed_documents(["alpha passage", "beta passage"])
    assert again.usage.cache_hits == 2, "重复文本必须零调用命中缓存。"
    assert again.usage.requests == 0, "全部命中缓存时不应产生任何 provider 请求。"
    assert again.usage.input_tokens == 0

    usage = provider.reset_usage()
    assert usage.provider == provider.provider
    assert usage.model == provider.model
    assert usage.dimension == int(provider.dimension)
    assert usage.index_version == provider.index_version
    # A local provider bills no money, and 0.0 says that. The inherited None
    # would say "we could not tell", which is a weaker and different claim.
    assert usage.cost == 0.0
    assert usage.price_per_mtok == 0.0

    with pytest.raises(ValueError, match="空文本"):
        provider.embed_documents(["   "])

    query = provider.embed_query("what is multi-head attention")
    assert len(query) == int(provider.dimension)
    assert isinstance(query, list) and all(isinstance(value, float) for value in query)


def test_provider_choice_is_configuration_not_code(monkeypatch):
    """Two configurations, one business call site, no branching anywhere else.

    T1.3's acceptance criterion -- provider 切换无需改业务代码 -- asserted rather
    than promised. The business call is
    :func:`build_embedder_from_config` followed by ``embed_documents`` /
    ``embed_query``; nothing downstream can tell which provider answered, and
    the switch is two environment values.
    """
    monkeypatch.setattr(config, "EMBEDDING_LOCAL_ENABLED", False)
    vendored = build_embedder_from_config()
    assert isinstance(vendored, EmbeddingGateway)
    assert not isinstance(vendored, LocalEmbeddingProvider)
    assert vendored.index_version.startswith("emb-") and "fastembed" not in vendored.index_version

    if not fastembed_installed():
        # Without the extra, "enabled" must degrade to no dense leg rather than
        # raise or silently reach for a vendor. That is the supported shape of
        # a personal install that set the flag and forgot the package.
        monkeypatch.setattr(config, "EMBEDDING_LOCAL_ENABLED", True)
        assert build_embedder_from_config() is None
        return

    _provider, _model, _dimension = local_identity()
    monkeypatch.setattr(config, "EMBEDDING_LOCAL_ENABLED", True)
    local = build_embedder_from_config()
    assert isinstance(local, LocalEmbeddingProvider)
    assert local.index_version != vendored.index_version


# --------------------------------------------------------------------------
# The indexing seam
# --------------------------------------------------------------------------


def test_a_document_indexed_without_vectors_is_reindexed_when_one_arrives(
    provider, local_config, tmp_path
):
    """``embedder=None`` is a supported call, and it used to be a trapdoor.

    :func:`paperwhisperer.api.routes_v1.persist_document` calls
    ``index_document_for_retrieval(stores, document)`` with no embedder -- the
    upload path has no embedding provider to hand. The idempotence check keyed
    on the index version alone, so the *second* call, this one with a provider,
    found a document whose version matched and returned early without writing a
    single vector. The document stayed lexical-only for the rest of its life,
    the dense leg abstained on every question about it, and the only symptom was
    a degradation string in a retrieval record nobody read.

    This is the specific reason the dense leg had never run in this project, so
    it gets a test rather than a note.
    """
    from paperwhisperer.analysis.orchestrator import index_document_for_retrieval
    from paperwhisperer.infra.stores import build_sqlite_bundle
    from paperwhisperer.ingestion.orchestrator import parse_document

    bundle = build_sqlite_bundle(_Settings(str(tmp_path / "stores.sqlite3")))
    document = parse_document(
        str(CORPUS / CORPUS_FILES[1]), allow_cloud_fallback=False
    ).document

    first = index_document_for_retrieval(bundle, document, None)
    assert first > 1, "这篇论文应该切出多块，单块的语料测不出这个行为。"
    assert bundle.lexical.count_chunks() == first
    # CONTRACT CHANGED. `embedder=None` used to mean "index lexically only" and
    # this assertion pinned it. It now means "resolve the configured provider",
    # because the previous semantics were the trapdoor described above: a caller
    # with no provider in hand left the document lexical-only forever, with the
    # only symptom a degradation string in a record nobody read. Asserting the
    # old behaviour would have reinstated it.
    assert bundle.vectors.count_vectors() == first, (
        "embedder=None 解析的是配置里的 provider，不是'只做词法索引'——"
        "否则调用方不传 embedder 就会让文档永久停留在纯词法状态。"
    )

    second = index_document_for_retrieval(bundle, document, provider)
    assert second == first, "重新索引改变了 chunk 数量，说明两次切分不一致。"
    assert bundle.vectors.count_vectors() == first, (
        "带着 embedder 再索引一次没有保持向量：dense 通道不会上线。"
    )

    # And a third call really is idempotent, which is the whole reason the check
    # exists -- the fix must not turn every Q&A turn into a full re-embed.
    usage_before = provider.usage.texts
    assert index_document_for_retrieval(bundle, document, provider) == first
    assert provider.usage.texts == usage_before, "第三次调用又重新 embedding 了一遍。"


def test_an_embedder_from_another_space_is_refused(indexed: Indexed, tmp_path):
    """Same-width, different model: the one mismatch a store cannot detect.

    A width change makes the store raise, because a width is a property of a row
    it already holds. A model change at the same width is invisible to it -- the
    vectors are all the right length and just belong to a different space, and
    the cosine similarities between them are meaningless rather than merely
    wrong. So the refusal has to happen at the indexing seam, before the first
    write, which is what :func:`_assert_embedder_matches_index` is for.

    The store is fresh rather than the session's, because the check guards a
    *write*. Against an already-indexed document the idempotence check returns
    first and writes nothing, which is safe -- there is nothing to corrupt -- so
    a test run against that bundle would pass without the guard ever running.
    """
    from paperwhisperer.analysis.orchestrator import index_document_for_retrieval
    from paperwhisperer.infra.stores import build_sqlite_bundle

    class Impostor:
        """Same width, different model. The case the store cannot catch."""

        provider = LOCAL_PROVIDER_NAME
        model = "some-other-model-of-identical-width"
        dimension = int(indexed.provider.dimension)

        def embed_documents(self, texts):
            raise AssertionError("必须在写入任何向量之前就拒绝。")

    fresh = build_sqlite_bundle(_Settings(str(tmp_path / "impostor.sqlite3")))
    with pytest.raises(EmbeddingDimensionMismatchError) as caught:
        index_document_for_retrieval(fresh, indexed.parsed[CORPUS_FILES[1]], Impostor())
    assert "some-other-model-of-identical-width" in str(caught.value), (
        "错误信息必须点名那个不相符的 embedder，否则运维无法判断该改配置还是改代码。"
    )
    assert fresh.vectors.count_vectors() == 0, "拒绝之后不应该留下任何向量。"
