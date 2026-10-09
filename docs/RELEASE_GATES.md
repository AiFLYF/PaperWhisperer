# Release Gates / 发布门禁（T4.3）

计划第 7 节的七行表格，全部落成机器可以判定的阈值。本文说明每一门**测什么**、
**怎么判**、**今天能不能跑**，以及两个被阻塞的门禁需要多小的标注工作才能解锁——
以及**除了标注之外还有哪一行代码挡在路上**（§3）。

| 实现 | CLI | 测试 | CI |
|---|---|---|---|
| `eval/gates.py` | `eval/run_gates.py` | `tests/test_release_gates.py` | `.github/workflows/ci.yml` 的 `gates` job |

> **门禁表格是七行，不是六行。** 计划第 7 节的表格有七条
> （直接引语 / 引用定位 / 零出网 / 静默截断 / 引用精度 / Recall@K / 任务恢复）。
> "六"出现在别处（第 2 节 2026-09-29 的六项决策），与门禁清单无关。
> 七条全部实现——为了对上句子里一个数字而删掉一条门禁，比数字对不上更糟。
>
> **可自动判定的有五条，被阻塞的有两条。** 7 − 2 = 5。
> 第 4 节列出哪一条**是评分表而不是测量**——按要求明说。

---

## 0. 一句话结论（本机实跑，56 份语料）

| 门禁 | 判定 |
|---|---|
| `direct_quote_alignment` | **PASS** |
| `local_only_cloud_egress` | **PASS** |
| `task_recovery` | **PASS** |
| `citation_locator_resolvable` | **FAIL——两条路由都失败，原因不同**（见 §2.2） |
| `silent_truncation` | **FAIL——1 处静默截断**（见 §2.4） |
| `citation_precision` | **BLOCKED**（见 §3：既缺人工确认的标注，也被 `eval/gates.py` 里硬编码的 `blocked=True` 挡住） |
| `retrieval_recall_at_k` | **BLOCKED**（见 §3：除标注外还缺一次真实检索运行的 top-K 结果） |

**三 PASS / 两 FAIL / 两 BLOCKED / 零 SKIP。**
两条 FAIL 都是**真实产品缺陷**，不是门禁写错：定位门禁抓到的是
"8 份扫描件在 legacy 下产出为空"与"4 份 DOCX 在 docling 下定位全丢"，
静默截断抓到的是"质量 FAIL 没有出现在面向用户的 warning 里"。
**审计前这三处都是绿的**，理由见 §2.2 与 §2.4 的"审计修正"。

下面各节给出每条判定的依据。

---

## 1. 四种状态，以及为什么是四种

`SKIP` 和 `PASS` 长得一样，是这套机制唯一要防的事。

| 状态 | 含义 | 阻断 CI？ | 阻断发布？ |
|---|---|---|---|
| `PASS` | 跑了，实测达到阈值 | 否 | 否 |
| `FAIL` | 跑了，实测没达到阈值 | **是** | **是** |
| `SKIP` | 跑不了，环境缺 fixture（无 `eval/corpus/`、无 `PAPERWHISPERER_TEST_PG_DSN`） | 否 | **是** |
| `BLOCKED` | 实现了，但没有真值可测（缺标注集）——**今天这两门还额外被 `eval/gates.py` 里一个硬编码的 `blocked=True` 强制走这条路，见 §3** | 否 | **是** |

`SKIP` 与 `BLOCKED` 分开，是因为两者的修法不同：`SKIP` 补一个 fixture 就能跑，
`BLOCKED` 只能靠一次标注工作。两者都不是绿的。

- **构建规则**（`GateRun.exit_code`）：只有 `FAIL` 返回非零。
  语料被 gitignore 了，一个 PR 不该因为"没下载 79 MB PDF"而变红。
- **发布规则**（`--release` / `assert_release_ready`）：`FAIL` 与未判定
  （`SKIP` + `BLOCKED`）都阻断。**今天 `--release` 返回 3**，因为两门被阻塞。
  这不是 bug，是诚实的答案。

判定的唯一出口是纯函数 `decide_status(spec, measurement)`，它只做一件事：
把测量值和阈值比大小。runner 不返回 `PASS`，它返回测量值。
`tests/test_release_gates.py::TestDecisionArithmetic` 直接喂手工构造的测量值，
测的是这段算术，不是某个 runner 今天的输出。

**零样本永远不是通过。** 没有样本 = `SKIP`，任何门禁任何阈值都如此。
没有这一条，一个语料没就位的门禁会报"100% 对齐"，构建变绿——
那正是这套机制要防的事。

---

## 2. 七道门禁

`python -m eval.run_gates --list` 可在任何机器上打印下表（不需要语料/数据库/端点）：

| Gate | 要求 | 阈值 | 怎么测 | 今天 |
|---|---|---|---|---|
| `direct_quote_alignment` | 100%；对不齐降级 paraphrase，不带引号 | `>= 100%` | 语料采样窗口 × `verify_quote` | **可跑** |
| `citation_locator_resolvable` | 100%；或明确标记无视觉定位能力 | `>= 100%` | `ParsedDocument.locator_coverage` + bbox 声明 | **可跑** |
| `local_only_cloud_egress` | 0 次（集成测试验证） | `== 0` | 跑既有零出网测试 | **可跑** |
| `silent_truncation` | 0；截断/低质量必须是可见警告或失败态 | `== 0` | 质量门禁 FAIL × 文档 warning | **可跑** |
| `citation_precision` | ≥95%（beta 目标，按标注集校准） | `>= 95%` | 人工标注的 claims | **阻塞**（见 §3：`blocked=True` 硬编码 + 标注未经人工确认） |
| `retrieval_recall_at_k` | ≥90% 起步，中英、表格/公式/正文分开统计 | `>= 90%` | 标注 query × 检索 top-K | **阻塞**（见 §3：`blocked=True` 硬编码 + 缺标注与检索运行） |
| `task_recovery` | worker/API 重启后任务可恢复或明确进入可重试终态 | `>= 100%` | 跑既有恢复测试 | **可跑** |

### 2.1 直接引语对齐（`direct_quote_alignment`）

**建在既有机制上**：`paperwhisperer.rag.citations.verify_quote`，
即产品渲染路径调用的同一个函数。样本是 `tests/test_citations.py`
语料门禁采样的同一批 5 份固定 fixture、同一组参数
（前 4 页、14 字宽、步长 5），因此两者测的是同一总体。

**两个方向都测，因为单向不是门禁。**

- 逐字窗口（1798 个）必须 `EXACT` 或 `NORMALIZED` 对齐；
- 伪造窗口（1798 个，由逐字窗口改动一个数字或删一个词得到）必须**不对齐**，
  且 `render().uses_quotation_marks is False`——即"降级 paraphrase，不带引号"。

只测第一个方向，一个"什么都能对上"的坏 verifier 就能全绿，而那恰好是最要命的坏法。
只测第二个方向，一个"什么都对不上"的 verifier 也能全绿。**伪造样本是近似命中
（同样的词、同样的顺序、错一处）**，完全无关的字符串任何子串搜索都会拒，
那样的负样本什么也证明不了。

本机实测：逐字 1798/1798 对齐，伪造 1798/1798 被拒并降级；
重排窗口 136 个中 117 对齐、19 个因换行连字符降级（预期内）。

### 2.2 引用定位可解析（`citation_locator_resolvable`）

**建在既有机制上**：判定依据是 `SourceLocator.is_resolvable`，
与 `ParsedDocument.locator_coverage` 用的是同一个定义，
因此门禁不可能和产品对"可解析"的定义产生分歧。

> **审计修正（2026-09-30）：这一门原先测的不是它声称测的东西。**
> 原实现把全语料的正文元素汇总成一个比值 `可解析元素 / 总元素`，
> 在本语料上读数恰好 **1.0000（1237/1237）**。但同时有 **8 份文档产出 0 个正文元素**
> （扫描件与合成 fixture，legacy 无 OCR）。它们同时退出分子与分母，
> 于是**产出为空的那批文档恰恰是压不低这个比数的那批**——
> 15% 的文档什么都没有，门禁报 100%。
> 这是"零样本永远不是通过"在上一层失效：不是没有样本，
> 是样本里悄悄少了最差的那批。
>
> 现在按**文档**判定：一份文档只有在它**每个**正文元素都可解析时才计 1.0，
> 产出 0 个元素的文档计 0。这与 `eval/run_parser_eval.py` 的
> `locator_report()` 完全一致——它对 0 元素文档返回 0.0，
> 对同一语料同一 parser 报 `0.8462`。**门禁与 bake-off 现在报同一个数。**
> 元素级比值仍然打印，但只作对照，不参与判定。

两半都要 1.0：

1. 每个正文元素 `locator.is_resolvable`（**按文档聚合**，含 0 元素文档）；
2. locator 类型属于计划期望带矩形的（`pdf_page`）却没带 bbox 时，
   文档必须**声明**"无视觉定位能力"。没声明的缺口算违规——
   对点击引用的读者来说，没声明的缺口和 bug 长得一模一样。

第 2 半只作用于 `pdf_page`，依据是 P0 退出标准原文：
"100% 有可解析 SourceLocator（PDF 到 page 可用时含 bbox，PPTX 到 slide/element，
DOCX 到 heading/paragraph）"。DOCX 的 `docx_element` 和 PPTX 的 `slide`
**按设计就不带矩形**——那是那些格式完整的定位词汇，不是一份降级的 PDF。
要求 DOCX 声明"没有 bbox"会制造一个计划没要求的失败；
**一个自己发明要求的门禁没人会相信**。

#### 2.2.1 门禁测的是哪个 parser（审计新增）

语料门禁默认 `--parsers legacy`。这是**速度**取舍，不是"legacy 就是线上 parser"的主张：
`PARSER_PRIMARY` 默认 `docling`，且 bake-off §6 已选定 docling 为主。
因此**实际使用的 parser 会打印在每条语料门禁的 evidence 里**，
不让任何读者去猜一个 100% 是出自线上 parser 还是兜底 parser。

**两条路由都不通过，但原因完全不同——这才是这张表值得看的原因：**

| 路由 | 实测 | 失败原因 |
|---|---|---|
| `legacy`（默认） | **FAIL 0.9231**（44/52 份文档） | 8 份扫描件 legacy 无 OCR，产出 **0 个正文元素** |
| `docling,legacy`（bake-off §6 的生产路由） | **FAIL** | 4 份 DOCX 在 docling 下 **定位全丢**，docling 返回 10–20 个元素但 locator 全为 `unknown`（`locator_coverage = 0.0`）；因为它"成功"了，legacy 兜底不触发 |

后一行是本轮审计最重要的产出，它复现了 bake-off §6 自己预言的那个坑：

> "DOCX 上 docling 不抛异常、不报错，只是安静地交出 0 locator——
> **这是一个'成功但不可用'的返回。** 因此 DOCX 的 legacy 兜底必须由
> **格式判定**驱动，不能等失败信号。"

`ParserRouter._ordered_candidates()` 只按 `PARSER_PRIMARY` + `PARSER_FALLBACK_CHAIN`
排序，**完全不看文档格式**（已核实：`PARSER_BACKEND=gateway` 时有序链是
`['docling','legacy']`，对 DOCX 也一样）。所以 §6 要求的那条格式分流**尚未落地**。
实测该分流确有必要：4/4 份 DOCX 上 docling `locator_coverage=0.0`、legacy `1.0000`。

> **`eval/run_parser_eval.py:build_parser_slots` 忽略调用方给的顺序**：
> 它按自己的 `KNOWN_PARSERS` 顺序返回（传 `["docling","legacy"]` 拿到的是
> `legacy` 优先）。bake-off 是并排比较，不受影响；但门禁把顺序当兜底链用，
> 所以 `eval/gates.py` 自行按请求顺序重排。
> **这是一个静默陷阱：任何假设"传进去的顺序就是返回的顺序"的调用方都会踩到**，
> 而且踩到之后拿到的是一个"看起来合理"的结果，不会报错。
> （本轮审计一开始就被它骗过一次：一度把 legacy-优先的结果当成 docling-优先的。）

### 2.3 本地模式零出网（`local_only_cloud_egress`）

**建在既有测试上，而不是另写一份。** 这个性质已经由
`tests/test_ingestion_orchestrator.py` 和 `tests/test_ingestion_mineru.py` 在
node 粒度上证明过了：每个用例都装了 socket 探针
（`urlopen` / `socket.create_connection` / `socket.socket` / `getaddrinfo`），
**记录并抛异常**，出网尝试无法 unnoticed 通过。

在这里重写一份"零出网"会变成第三份需要和前两份同步的东西，
而它们迟早会不一致——**一个会和它声称强制的测试漂移的门禁，比没有门禁更糟**。

阈值 `violations == 0`。分母是真正跑起来的用例数：指定 node id 一个都没收集到时
是 `SKIP` 而不是空洞的通过（证明零出网的测试被改名了，门禁必须说"没跑"）。

### 2.4 静默截断（`silent_truncation`）

测的是**产品对降级的反应**，不是解析器的准确率。语料里少几页不是违规——
那是语料的事。违规是：一份**确实**可测地截断了、却既不带 warning 也不带
质量 flag 的文档，因为那种情况下分析会自信地总结半篇论文而屏幕上什么都不说。

两半必须来自**不同的表面**，这是这条门禁最容易写错的地方：

- `evaluate_quality()` 说**是否坏了**；
- `document.warnings` 说**有没有人发现**。

把"有 quality flag"也算作"可见"，这条门禁就**永远无法失败**——
因为本函数正是算这些 flag 的东西，凡是它能发现的截断必然有 flag，
于是每一例都算"可见"。`tests/test_release_gates.py::test_truncation_gate_fails_on_a_silent_truncation`
就是钉住这一点的。

> **审计修正（2026-09-30）：另一个方向的同一个错误。**
> "有没有人发现"原先实现为 `bool(document.warnings)`。
> 在真实语料上这根本不是一次检验：52 份里有 **37 份只带一条 warning**，
> 即 legacy 适配器对**每一份 PDF 无条件发出**的 bbox 声明
> （"legacy PDF 解析只能定位到页，无法提供页内坐标（bbox）。"）。
> 于是一份丢了 10 页里 9 页的 PDF，会凭一句关于"高亮"的话被判为"已可见"，
> 门禁通过——**而且是因为一个与截断无关的理由通过**。
> 能力声明对每个文档都成立，因此它不能作为"针对本文档报出了什么"的证据。
>
> 现在**先把能力类 warning 减掉再问有没有可见性**（是减，不是换）：
> 同时带 bbox 声明和真实"第 3 页没有可提取文本"的文档仍然算可见，
> 让它可见的是第二条。若只替换不减，这道修正就会矫枉过正成一个无视 warning 的门禁。

分子是**违规数**（0 容错计数门禁），分母是全部解析出的文档数。
若把"没坏的文档数"当分子，一份语料里一处静默截断会被读成一个比率去和 0 比，
算术和阈值就对不上了。

#### 当前唯一的静默截断（真实缺陷）

`hybrid_zh_paper.pdf`：1 页 1 元素 101 字，`text_density` 报 FAIL
（低于 200 字符/页），而它的 warning 只有 bbox 那一条。
**用户上传后看不到任何"这份文档解析质量很差"的提示。**
产品侧修法是把质量 FAIL 也写进面向用户的 warning——
但 `paperwhisperer/**` 不在本任务可写范围内，**已上报，未修复**。
这条 FAIL 不应被登记为已知缺陷抹平：它是真的。

#### 已登记缺陷（`KNOWN_DEFECTS`）

`paperwhisperer/ingestion/quality.py` 不在本任务可写范围内，而其中两条规则
在面向的格式之外会误报。与其让门禁从第一天起就永久变红（那正是门禁失去可信度的开始），
每一条都**精确登记**在此，并且抑制是**严格的**：

| defect_id | 位置 | 误报原因 |
|---|---|---|
| `page-coverage-ignores-non-pdf-locators` | `quality.py:_page_of` | 只识别 `pdf_page`，PPTX 的 `slide` 全返回 None，于是一份 6/6 幻灯片全覆盖的 deck 被判为 0 页覆盖 |
| `text-density-threshold-is-paper-shaped` | `quality.py:TextDensityRule.evaluate` | `min_chars_per_page=200` 是排版论文页的阈值，套到幻灯片上不成立：329 字/6 页必然低于它，但它既没扫描也没丢页 |

抑制只针对**具体 rule_id**，只对**非 PDF 分页格式**生效。
一份真正被截断的 deck 仍会失败（另两个 flag 是真违规时不会被抹掉）。
底层条件一旦修好，抑制自动失效、门禁重新变红，**届时请删除对应条目**——
这与 `eval/injection/README.md` 记录的已知缺陷、以及
`tests/test_ingestion_security.py` 里的 `xfail(strict=True)` 是同一套约定。

**这两条缺陷已上报，未修复**（不在本任务的可写范围内）。

### 2.5 任务恢复（`task_recovery`）

同样**接既有测试**：`tests/test_jobs.py::TestRecoveryFromAKilledWorker`
会 fork 一个子进程、在 stage 中途 `os._exit(9)`，对共享 SQLite 文件恢复；
`TestRecoveryWhenTheRetryBudgetIsGone` 断言重试预算耗尽时进入
`retryable is True` 的终态——正是计划要求的"可恢复**或**明确进入可重试终态"。

两个档位能跑都跑。`PAPERWHISPERER_TEST_PG_DSN` 未设置时，Hosted 半边
在 evidence 里记为**未执行**，不计为通过。

### 2.6 哪一条是"评分表"而不是测量（审计明确回答）

**七条里没有一条是评分表。五条是测量，两条被阻塞。**
按要求把这话说清楚，顺便说清楚"看起来像评分表"的东西在哪。

| 项 | 是测量吗 | 依据 |
|---|---|---|
| `direct_quote_alignment` | **是** | 对真实语料窗口调产品自己的 `verify_quote`，正反两个方向都测 |
| `citation_locator_resolvable` | **是** | 真实语料逐文档聚合，阈值 100% |
| `local_only_cloud_egress` | **是（代理测量）** | 跑真实测试，判定权在那些测试的 tripwire 上 |
| `silent_truncation` | **是** | 真实语料 + 真实质量门禁，可见性取自另一处表面 |
| `task_recovery` | **是（代理测量）** | 跑 `os._exit(9)` 的真实恢复测试 |
| `citation_precision` | **否——且这个 BLOCKED 是被强制出来的，见 §3** | 没有人工确认的标注就没有真值；今天还被一个硬编码布尔挡住了真实读数 |
| `retrieval_recall_at_k` | **否——同上，且另缺检索运行产物** | 同上，另需 `retrieval_run.json`（§3.2） |

三条需要点名的：

1. **两条被阻塞的门禁，如果按"填表"实现，就是评分表。**
   让一个人读完报告、在表格里写下"这条引用我认为有据"，
   然后把表格里的数字当门禁——那是评分表，不是门禁。
   本实现的立场是：**表格是门禁的输入，不是门禁的判定**。
   判定必须由 `decide_status(spec, measurement)` 从测量值算出来。
   人只负责产生**真值**（哪些元素相关、哪条 claim 有据），
   真值一旦落盘，判定就完全机械。**所以今天它们是 BLOCKED，不是 PASS 也不是 FAIL。**

2. **T4.3 的"corpus 扩到 150-200 份"是评分表，本实现刻意不把它做成门禁。**
   它的判定取决于项目所有者有没有下载 PDF——**这测的是待办事项，不是产品**。
   因此 `corpus_stats()` **只报告**（常规在场数 / 清单登记数 / adversarial 数），
   `GateRun` 的失败判定里没有它。见 §4。

3. **`local_only_cloud_egress` 与 `task_recovery` 是"代理测量"，要说清代价。**
   它们的判定权在别人写的测试上。这不是缺陷——重写一份"零出网"只会得到第三份
   需要和前两份同步的东西，而它们迟早会不一致。但代价是具体的：
   - 测试被**改名** → 收集到 0 个用例 → `SKIP`（已处理，会说明原因）；
   - 测试被**删除** → 分母悄悄变小 → 门禁在更小的集合上通过，**这一条没有守卫**。
     目前靠 `EGRESS_NODE_IDS` / `RECOVERY_*_NODES` 常量与实际测试名对齐，
     但没有测试断言这些 node id 仍然存在。**这是一处已知的、小的缺口。**

---

## 3. 两个被阻塞的门禁

`eval/manifest.yaml` 没有逐文档的 ground-truth 映射。
按任务要求：**不发明标签，不用系统自身输出充当真值。**
**一个拿系统自己给自己打分的门禁，比没有门禁更糟。**

harness 已全部实现并有单测：标注格式、加载校验、打分、分类别分解。
模块里**刻意没有**任何回退路径去拿系统输出当真值，也**没有**一个"临时数字"——
那种数字会在真标签到位后被误当成真的。

> ### ⚠️ 修正：解锁这两门**需要改一行代码**，"标签一到即可运行"是不成立的
>
> 本节此前写着"标签一到，门禁无需再改代码即可运行"。**那是错的**，而且错在代码里：
>
> `eval/gates.py:1900`（`citation_precision`）与 `:1910`（`retrieval_recall_at_k`）
> 的 `GateSpec` 把 **`blocked=True` 硬编码**。`run_gate`（`eval/gates.py:2060`）的第一句就是
> `if spec.blocked:` —— 它**短路**，直接构造一个 `STATUS_BLOCKED` 结果，
> 根本不会调用 `measure_precision` / `measure_recall`。
>
> 于是报告里那行"缺标注集 / `retrieval_labels.yaml` 不存在"来自
> `_blocked_measurement`（`:1656`）里对 `measure_precision` 的**试探性调用**，
> **而不是门禁真的缺输入**。换一个不存在的路径，它照样会打印同一句话。
>
> **实测**（本机直接调 spec 自己的测量函数，绕开短路）：
> ```
> >>> gates.measure_precision(ctx)
> Measurement(numerator=2.0, denominator=34.0, ...)   # 2/34 ≈ 5.9%
> ```
> 阈值是 **95%**。也就是说：**这个 BLOCKED 掩着一个真实且很红的读数。**
> （`measure_recall` 则如 §3.2 所述另缺 `retrieval_run.json`，仍会 `MeasurementSkip`——
> 那一门即使改掉布尔也还跑不起来，缺的是第二样东西。）
>
> **所以：把那个布尔翻过来，不会得到一个通过的门禁，会得到一条可见的红。**
> 这正是应该发生的事——`eval/labels/retrieval_labels.yaml` 里现在有 53 行，
> 但它们**全部**是 `assistant_draft` / `human_confirmed: false`，文件头自带
> `!! NOT GATE-GRADE` 横幅（见 §3.4）。5.9% 这个数字量的是"未经人工确认的草稿
> 与系统自身输出的吻合度"，**不是引用精度**，因此它也不构成一个可用的门禁读数。
>
> **本文档只记录这个事实，不改门禁代码。** 该不该翻这个布尔、什么时候翻，
> 是负责人对"我们要不要一道今天必红的门禁"的判断，不是文档能替你做的决定。

所以两门是 `BLOCKED`。**真正的缺失**不是 harness（那部分已就位），而是两样人工产物：
一份**经人工确认**的标注集，以及（仅 Recall@K 需要）一次真实检索运行的 top-K 结果。
下面 §3.1–§3.4 仍然是解锁它们需要做的事。

### 3.1 标注格式

`eval/labels/retrieval_labels.yaml`（进 git——它是门禁的输入，不是运行产物）：

```yaml
schema: paperwhisperer-eval-labels/0.1
# 建议记录：这份标注是对哪个语料版本做的。跨版本比较无效。
corpus_version: sha256:0dfe319761a7cc9d003dd71c00be7ca57f912bf529654b831cd61d5da8f5dda1

queries:
  - id: q001
    document: digital_single_col/attention_is_all_you_need.pdf
    query: "多头注意力是如何计算的？"
    language: zh            # en | zh | mixed  —— 计划要求中英分开统计
    kind: prose             # prose | table | formula —— 计划要求表格/公式/正文分开统计
    relevant_element_ids:   # 真值：正确的检索器至少要返回其中之一
      - el_1a2b3c4d5e6f7081
      - el_9f8e7d6c5b4a3928

claims:
  - id: c001
    query_id: q001
    quote: "我们采用多头注意力而非单头……"
    element_id: el_1a2b3c4d5e6f7081
    supported: true          # 人工判断：该引用真的有据可依
```

两条轴是**独立**的：`lang:*` 与 `kind:*` 各自成桶，不合并。
合并会让"中文正文"和"英文表格"两类查询平均成一个数字，把正在失败的那类藏起来。

`load_labels` 的校验是**响亮**的，因为一份悄悄丢行的标注集会报出一个假数字：

| 情况 | 行为 |
|---|---|
| 文件不存在 | `FileNotFoundError`（这是 `run_gate` 那句 BLOCKED 理由的**来源**，见本节开头的修正——它由 `blocked=True` 的短路打印，不是门禁真的跑了才缺） |
| `schema` 前缀不对 | `ValueError` |
| query 没有 `relevant_element_ids` | `ValueError`——没有真值的查询只会虚高分母 |
| query 的 `language` 不在 `en/zh/mixed` | `ValueError`——无法归类的标签会让该类别悄悄消失 |
| query 的 `kind` 不在 `prose/table/formula` | `ValueError` |
| claim 没有 `supported` | `ValueError`——精度必须由人判断，不能由系统自评 |

### 3.2 Recall@K 还需要第二样东西

即使有了标注，`score_recall_at_k` 还需要**检索实际返回了什么**：
每个 query 的 top-K `element_id` 序列。该产物也尚未纳入仓库
（约定路径 `eval/labels/retrieval_run.json`）。两样都缺，因此 `BLOCKED`。

K 默认 5（`DEFAULT_RECALL_K`）。**K 是计划的参数（"≥90% 起步"），
不是调出来的常数**——看完分数再定 K，这个数字就没有意义了。

### 3.3 最小的解锁工作

按投入产出排序，两步：

**第一步（解锁 `retrieval_recall_at_k`，约半天）**

1. 挑 4 类文档各 1 份——中英混排、英文双栏、表格密集、公式密集
   （语料里都现成：`mixed_zh_en/bilingual_paper_zh_en.pdf`、
   `digital_two_col/batchnorm.pdf`、`table_dense/glue_benchmark.pdf`、
   `formula_dense/maldacena_large_n.pdf`）。
2. 每份写 3 条 query，共 12 条。**每条只标"这一问的答案在哪几段"**，
   不需要写答案、不需要判断难易、不需要给相关性分数。
3. 标 `language` 与 `kind`，并把相关段落的 `element_id` 填进
   `relevant_element_ids`。取 id 的办法：解析该文档后打印
   `[e.element_id for e in document.body_elements]`。
4. 跑一次检索，把每个 query 的 top-5 `element_id` 写进
   `eval/labels/retrieval_run.json`。

12 条样本足以让门禁**开始判定**，但足以估出 90% 这个数目的置信区间还很宽。
它是"门禁开始有牙齿"，不是"这个数可信"。真正可用于 beta 校准的量级见下。

**第二步（解锁 `citation_precision`，约一天）**

1. 复用第一步的 12 条 query，跑完整问答链路得到 claims。
2. 人读每条 claim 与它引用的 evidence，判 `supported: true/false`。
3. 阈值 ≥95% 是 beta 目标，**小标注集上 95% 与 99% 无法区分**。
   要让 95% 有意义，claim 数需要让 95% 的置信区间不跨过阈值：
   100 条 claim 时 5 个不支持样本就正好压线，**至少要 200 条**
   （约 60–70 条 query × 3 条 claim）。

两步都**不需要改门禁代码**：harness、格式校验、打分、分类别分解都已就位并有单测。
但它们也**都还不足以解锁**——除了做出标注，还必须处理本节开头那个硬编码的
`blocked=True`（否则门禁仍然短路），而那一步的结果是一道**会红的**门禁，不是绿的。

### 3.4 现有标注产物的状态 `[代码实测]`

`eval/labels/retrieval_labels.yaml` 已经存在，53 行 query / claim，
由 `tools/label_retrieval.py`（`cce61e6`）产出。但：

- 文件头写着 `!! NOT GATE-GRADE`，`human_confirmed: false`，
  全部 `provenance: assistant_draft`；
- `eval/labels/pending_labels.yaml` 仍为空
  （`{'queries': [], 'claims': []}`），注释写着
  "工具产出但没有人确认的记录……并进去就是让系统给自己打分"；
- 双人一致性：exact-set 0.70，mean window Jaccard 0.821。
  按该 commit 自己的说法，这测的是**标注说明是否无歧义**，不是人工一致性，
  因此是一个乐观上界；
- 规模仍然远远不够：±5% 需要 +78 条，±2% 需要 +589 条。

**所以"标注集缺失"这句话今天仍然成立**——缺的是**人工确认**，不是文件。

---

## 4. 语料规模

| 项 | 值 |
|---|---|
| 现状 | **52 份常规 + 4 份 adversarial = 56**（`eval/manifest.yaml`，12 个类别） |
| `corpus_version` | `sha256:0dfe319761a7cc9d003dd71c00be7ca57f912bf529654b831cd61d5da8f5dda1` |
| T4.3 目标 | 150–200 份 |
| 缺口 | 约 100–150 份 |

**语料计数只报告，不作门禁。** 语料扩充（T0.1 / T4.3）是项目所有者的数据收集任务；
一个因为"还没下载 100 份 PDF"而失败的门禁，测的是待办事项而不是产品。
`corpus_stats()` 报告**清单登记数**与**磁盘在场数**两个数——两者背离
恰恰发生在有人改了清单却没下完语料的时候，只报登记数会让人以为语料是完整的。
`adversarial` 单独计数并单列：它们登记在册但**刻意不进常规批跑**
（设计上就该失败），混进分母会虚报被评测的总体。

两处限制已核实：

- 直接引语门禁**指名** 5 份 fixture，缺任一份是带原因的 `SKIP`，
  而不是悄悄退化成一个更小的 100%；
- 窗口数随页长线性增长，语料扩到 200 份时该门禁只是更慢，**形状不变**。
  这里没有哪个门禁会因为 200 份而撞到上限。

### 扩到 150–200 份时，runner 需要改什么

**结论：不需要改代码。** 逐项核实过：

| 项 | 现状 | 200 份时会怎样 |
|---|---|---|
| `manifest.yaml` 的 `regular_entries` | 无上限，无硬编码数量 | 照常遍历；清单加条目即可 |
| `--limit` | 按 `(类别, 文件)` 排序取前 N | 确定性不变，冒烟仍选中同一批 |
| 语料门禁排序 | `sorted(key=(category, file))` | 与语料大小无关 |
| 引语窗口数 | 随页长线性 | 只是更慢，`_windows` 无上限 |
| 定位/截断门禁 | 逐文档解析并 memoise 一遍 | 时间线性增长 |
| `run_gates.py` 的路径推断 | 无数量假设 | — |

实测时间（52 份，legacy 路由）：定位 ~68 s、引语 ~14 s。
线性外推到 200 份约 **4–5 分钟**，仍属可接受，但**这已经接近 CI 的合理上限**——
若将来想让 CI 定期跑全量语料，建议保留 `--limit` 冒烟 + 本机全量的分工。

**缺什么**（属于项目所有者，不属于门禁代码）：

1. 语料文件本身——`eval/corpus/` 是 gitignore 的本地目录，门禁只按清单读取；
2. 清单条目——每个类别按第 6 节配额补足，**新条目需带 `category`**，
   否则分母算不出分类别的那一套；
3. `corpus_version` 会随清单变更——**已存的标注与报告跨版本不可比**，
   换语料必须重跑 bake-off 并重标（bake-off §6"触发重评的条件"第 1 条）。

---

## 5. CI

`.github/workflows/ci.yml` 新增 `gates` job，与既有 `backend` / `frontend` 并列——
**追加，不替换**。原有的 lint + test + 前端结构检查原样保留。

该 job 跑 `python -m eval.run_gates`，用**构建规则**（只有 `FAIL` 非零）。
它配一个 pgvector 服务容器并设置 `PAPERWHISPERER_TEST_PG_DSN`，
让任务恢复门禁在 PostgreSQL 档位上真正跑起来——
仓库本来就认这个变量（`tests/test_jobs.py:PG_DSN_ENV`），
没设置时那半边会干净地 skip（`postgres_availability()` 只在变量已设置时才探测）。

> **审计修正（2026-09-30）：`hosted` extra 是必需的，不是可选的。**
> 该 job 原本装 `.[dev]`，而 `psycopg` 只在 `hosted` 里。
> 没有驱动时 `postgres_availability()` 返回"驱动不可用"，
> `TestJobPostgres` **全部 skip**——服务容器健康起来、DSN 也设了，
> 而恢复门禁只测了 SQLite，**job 注释却宣称在跑 PostgreSQL 契约**。
> 门禁会诚实地把 Hosted 记为"未执行"，所以它没有说谎；
> 说谎的是注释和**读者对覆盖面的推断**。
> 已改为 `.[dev,hosted]`。
> 本机核对：真实库上 `tests/test_jobs.py::TestJobPostgres` 43 passed。

### 干净克隆上不会因为错误的理由失败

| 缺什么 | 门禁的反应 | 退出码 |
|---|---|---|
| `eval/corpus/`（gitignore） | 三门语料门禁 `SKIP`，并写明按 `eval/README.md` 投喂 | 0 |
| `PAPERWHISPERER_TEST_PG_DSN` | Hosted 档记为未执行；SQLite 档照跑 | 0 |
| `docling`（CI 只装 base+dev+hosted） | 语料门禁默认 `--parsers legacy`，照跑 | 0 |
| LLM 端点 / key | 门禁**不依赖**任何端点——设计如此 | 0 |
| 标注集 | 两门 `BLOCKED`，并写明缺什么真值 | 0 |

**干净克隆上 `gates` job 的实际结果是 3 PASS / 0 FAIL / 0 SKIP / 2 BLOCKED，退出 0。**
零出网与任务恢复两门不依赖语料，在 CI 里真的跑；两门被阻塞的明确记为 BLOCKED。
本机的两处 FAIL（定位 0.9231、静默截断 1 处）**在 CI 上不会出现**，
因为它们依赖 gitignore 的语料——这也正是这些门禁的意义：
**它们在有语料的机器上盯着产品，在没有语料的机器上不装作看过产品。**

`--release` 不在 CI 里跑：它今天必然返回 3（两门被阻塞），
那会让每次提交都变红。它是**发布前**在有语料、有标注的机器上的检查。

---

## 6. 本地运行

**一条命令**（出报告、有真失败时非零退出）：

```bash
make gates          # = python -m eval.run_gates --out eval/reports
```

```bash
# 门禁定义与阈值（不执行任何门禁，任何机器都可以）
make gates-list

# 全跑（需要 eval/corpus/；不需要网络、数据库、LLM 端点）
python -m eval.run_gates

# 按生产路由测（docling 优先，legacy 兜底）——慢约 20 倍但这才是线上语义
python -m eval.run_gates --parsers docling,legacy

# 单门
python -m eval.run_gates --only direct_quote_alignment

# 语料冒烟（确定性：按 (类别, 文件) 排序取前 N）
python -m eval.run_gates --limit 3

# 发布前硬门禁：任何未判定的门禁都阻断（今天返回 3）
make gates-release

# 不跑子进程门禁（最快）
python -m eval.run_gates --skip-subprocess
```

退出码：`0` 无门禁失败 / `1` 有门禁失败 / `2` 命令行用法错误 / `3` `--release` 下未就绪。

Make 目标：`make gates`、`make gates-release`、`make gates-list`。

产物写入 `eval/reports/release_gates.{md,json}`，文件名固定
（与 `run_parser_eval.py` 一致），时间戳在文件内容里。

---

## 7. 现状与未决项

**本机实跑，56 份语料（52 常规 + 4 adversarial），2026-09-30：**

| 状态 | 门禁 |
|---|---|
| PASS | 直接引语、零出网、任务恢复 |
| **FAIL** | **引用定位（两条路由都失败，原因不同）、静默截断（1 处）** |
| BLOCKED | 引用精度、Recall@K |
| SKIP | 无 |

> 审计前的记录是"5 门 PASS，0 门 FAIL"。**那是错的**——
> 定位门禁读 1.0000 是因为它把 8 份零元素文档同时移出了分子和分母，
> 静默截断读 0 违规是因为"有 warning"这个判据被 37 份 PDF 共用的
> bbox 能力声明满足了。两处都是**因为错误的理由变绿**。

**阻塞**：引用精度、Recall@K——§3.3 给出最小解锁工作（标注），
但**解锁不止是标注**：`eval/gates.py` 把这两门的 `blocked=True` 硬编码，
`run_gate` 会短路并打印一句实际上来自短路探测的"缺标注集"。
直接调 `measure_precision` 实测是 **2/34 ≈ 5.9%** 对 95% 阈值——
翻掉那个布尔会得到一道**真实且红**的门禁，不是一道通过的。详见 §3。

**已上报但未修复**（不在本任务可写范围 `paperwhisperer/**`）：

- `paperwhisperer/ingestion/quality.py:_page_of` 不识别非 PDF 分页定位，
  导致全覆盖的 PPTX 被判 0 页覆盖（见 2.4 已登记缺陷）。
- `paperwhisperer/ingestion/quality.py:TextDensityRule.evaluate` 用论文页阈值
  衡量幻灯片密度。
- **静默截断那 1 处**：质量 FAIL 没有写进面向用户的 warning。
  产品侧修法是让 `parse_document` 把质量 FAIL 透出到 `ParseOutcome.warnings`
  （`ParseOutcome` 已有 `quality` 字段，`routes_v1.parse_report` 也会序列化它，
  但**面向用户的 warning 通道里没有它**）。
- **bake-off §6 要求的按格式分叉（DOCX 走 legacy）没有实现**：
  `ParserRouter._ordered_candidates()` 只按 `PARSER_PRIMARY` + `PARSER_FALLBACK_CHAIN`
  排序，**完全不看文档格式**。实测 docling 在 4/4 份 DOCX 上
  `locator_coverage = 0.0`（全部元素 locator 为 `unknown`），
  legacy 为 1.0000。也就是说：**把 `PARSER_BACKEND` 切到 `gateway` 之后，
  DOCX 会退化成"解析成功但一个引用都定位不了"**，
  而这正是 bake-off §6 明确要求用格式分流来避免的情况。
  这条同时污染 §6 自己的 `PARSER_PRIMARY=docling` 决策。
- CI 没有 PostgreSQL 服务容器的是 `backend` job（`gates` job 已配置），
  既有 `backend` job 仍然只跑 SQLite。
