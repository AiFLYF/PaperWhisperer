# SLO 与性能目标（T4.6）

计划原文：「按页数/格式建立 P95 时延目标；queue backlog 与 provider 限速告警」，
验收标准「**SLO 有数据支撑且可告警**」。

这份文件回答两个问题，顺序不能颠倒：

1. **表里哪些数字是量出来的，哪些是定出来的。**
2. **告警能不能真的响。**

目标表本身不在本文件里维护，而在 `paperwhisperer/core/slo.py` 的
`LATENCY_TARGETS`。本文件下面那张表由 `slo.format_target_table()` 生成，
两者不一致时**以代码为准**——因为代码里的每一行都带一个 `evidence_kind` 标记，
而标记是可以在 `pytest` 里被检查的，文档不是。

---

## 1. 结论先行

**24 条时延目标里，只有 3 条有实测支撑。**

这个比例不好看，但它就是当前真实的证据量。把另外 21 条也写成"实测"不会让
服务变快，只会让这 3 条也一起失去可信度——因为读者无法再区分哪一行是量的。

| 分类 | 条数 | 说明 |
|---|---|---|
| **实测（measured）** | 3 | legacy（pypdf）解析 PDF，按页数分三个桶 |
| 目标（target） | 21 | 目标值，无实测支撑，逐条说明原因 |
| 目标 · 样本量不足 | 4 | 桶内样本 < 5，P95 恒等于最大值，不能自称分位数 |
| 目标 · 无法在本环境测 | 14 | docling、MinerU、LLM、检索 |

对应到验收标准的两半：

- **「SLO 有数据支撑」——部分达成。** 解析时延有据可查的只有 legacy 一条路径，
  而它**不是生产默认路由**（默认是 docling）。生产默认路径上的时延目前是目标，
  不是承诺。
- **「可告警」——达成。** 两条告警规则都对着 T4.1 已有的指标判定，且本次补上了
  两个会让告警**永不触发**的缺陷（见第 4 节）。

---

## 2. 哪些是实测，哪些是目标

「实测」= 在真实语料上跑过、留下了可复核的产物。
「目标」= 一个写下来的数字，**还没有**任何一次测量支撑。

| 阶段 | 格式 | 页数 | parser | P95 目标 | 依据 | 样本 |
| --- | --- | --- | --- | --- | --- | --- |
| parse | pdf | 1–10 页 | legacy | 379 ms | **实测** | 22 |
| parse | pdf | 11–30 页 | legacy | 3699 ms | **实测** | 15 |
| parse | pdf | 31–80 页 | legacy | 23621 ms | **实测** | 5 |
| parse | pdf | 81–200 页 | legacy | 6575 ms | 目标 | — |
| parse | pdf | 201–∞ 页 | legacy | 8206 ms | 目标 | — |
| parse | docx | 1–10 页 | legacy | 50 ms | 目标 | — |
| parse | pptx | 1–10 页 | legacy | 50 ms | 目标 | — |
| parse | pdf | 1–10 页 | docling | 2000 ms | 目标 | — |
| parse | pdf | 11–30 页 | docling | 8000 ms | 目标 | — |
| parse | pdf | 31–80 页 | docling | 20000 ms | 目标 | — |
| parse | pdf | 81–200 页 | docling | 40000 ms | 目标 | — |
| parse | pdf | 201–∞ 页 | docling | 90000 ms | 目标 | — |
| parse | docx | 1–10 页 | docling | 4000 ms | 目标 | — |
| parse | pptx | 1–10 页 | docling | 4000 ms | 目标 | — |
| parse | pdf | 1–10 页 | mineru_cloud | 60000 ms | 目标 | — |
| parse | pdf | 11–30 页 | mineru_cloud | 90000 ms | 目标 | — |
| parse | pdf | 31–80 页 | mineru_cloud | 180000 ms | 目标 | — |
| parse | pdf | 81–200 页 | mineru_cloud | 300000 ms | 目标 | — |
| parse | pdf | 201–∞ 页 | mineru_cloud | 600000 ms | 目标 | — |
| retrieve | any | corpus-1k | personal-sqlite | 80 ms | 目标 | — |
| retrieve | any | corpus-3k | personal-sqlite | 110 ms | 目标 | — |
| retrieve | any | corpus-7k | personal-sqlite | 180 ms | 目标 | — |
| retrieve | any | hosted-pgvector | hosted-postgres | 200 ms | 目标 | — |
| llm | any | any | any | 30000 ms | 目标 | — |

### 注意 `l` 桶比 `xl` 桶慢

实测值里 31–80 页（23.6 s）比 81–200 页（6.6 s）**慢一个量级**，而页数更多。
这不是笔误，是语料的性质：31–80 页桶里落进了扫描件与复杂版式，
`l` 桶的 P95 就是那一份最慢的文件；`xl` 桶只有 2 份文件，都是文本层干净的
数字原生 PDF。

**这条曲线的形状不可外推**：它反映的是这一批语料里有什么，不反映"页数与耗时
的关系"。分桶边界也是按语料的页数分布选的（见 `slo.PAGE_BUCKETS` 的注释），
不是按直觉选的。

---

## 3. 实测是怎么做的

- **产物**：`eval/reports/run1-local/parser_eval.json`
  （`corpus_version: sha256:0dfe3197…`，由 `eval/run_parser_eval.py` 产出）
- **语料**：`eval/corpus/` 的 48 份 PDF，其中 45 份解析成功，
  3 份 adversarial（损坏 / 加密 / 伪后缀）按预期失败。页数 1–261。
- **解析器**：`LegacyDocumentParser`（pypdf 文本层），**不含 OCR**。
- **样本量**：每份文档一个样本，共 45 个，按页数分桶。
- **分位数**：`slo.nearest_rank_percentile`（nearest-rank，第 `ceil(q·n)` 个观测值）。
- **环境**：`Linux 6.8 x86_64 / 72 vCPU / 251 GiB RAM / CPython 3.13.13 / 单进程 / 无 GPU`。

### 怎么复核

```bash
python3 eval/run_parser_eval.py --parsers legacy     # 重跑测量
python3 -m pytest tests/test_slo.py -q               # 断言表里每个"实测"都能重算出来
```

`tests/test_slo.py::TestLatencyTargets::test_every_measured_row_recomputes_from_its_artifact`
是这一节真正的保障：它从产物里独立重算每一个 `measured` 行，**样本数与 P95
都要逐字相等**。表里写 379 而产物重算出 380，测试就会红。

这不是装饰性的检查。`LatencyTarget.__post_init__` 还会**在构造时**拒绝两件事：

- 标 `measured` 却没给 `evidence_ref`（没有可复核的产物）；
- 标 `measured` 却样本数 < 5。

### 一个必须知道的限制

`eval/` 目录当前**整体未被 git 跟踪**（`git status` 显示 `?? eval/`）。也就是说
支撑这 3 个实测数字的报告**不在版本管理里**。测试会因此失败并明确提示要
把 `eval/` 纳入版本管理——在那之前，"实测"这三个字在别人 clone 下来之后是
无法复核的。**这是当前 T4.6 最紧要的一个待办。**

### 为什么样本 < 5 就不算实测

nearest-rank 在 `n < 20` 时**总是返回最大值**。`n=2` 的 P95 是两个观测里更大的那个，
`n=1` 的 P95 是那一个观测。那不是分位数，是一个观测值。

所以 81–200 页（n=2）、201+ 页（n=1）、docx（n=4）、pptx（n=3）这四行被降级为
`target`，但**它们的实测值仍然写在 `evidence` 里**（6575 / 8206 / 16 / 9 ms），
没有丢失。补足语料后用同一个脚本重跑即可转回 `measured`。

---

## 4. 告警

两条规则都读 T4.1 已经写好的指标，**本任务不新增任何 metric**。
两套指标里只有一套会被人盯着看。

### queue backlog（`jobs.queue_depth`）

| 状态 | 阈值 | 持续时间 |
|---|---|---|
| `queued` | 20 个 | 300 s |
| `parsing` | 8 个 | 600 s |
| `analyzing` | 8 个 | 600 s |

两个条件同时满足才报：**绝对深度**（队列里有 200 个任务，机器就已经在积压了）
与**持续时间**（刚提交 5 个任务然后开始消费不是积压，而批量上传是这个产品的
正常用法）。只满足一个的规则会在每次批量上传时误报，而每天响十次的告警等于
没有告警。深度 ≥ 阈值 ×5 升级为 `critical`。

计时由调用方维护：`evaluate_backlog(..., above_since=marks)` 会**就地更新**
传入的 `dict`，首次越过阈值时只记时刻不发告警。

### provider 限速（`llm.failures_total` / `parsing.attempts_total`）

判据是**绝对次数 ≥ 3** 且**占比 ≥ 20%**，两个都要。绝对数排除了样本太小带来的
误报，占比排除了"一次就报"——单次 429 是正常现象，重试退避本来就会处理它。

本次补上了两个会让这条告警**永不触发**的缺陷：

1. **标签名读错。** `parsing.attempts_total` 把状态记在 `status=` 上，
   而原实现只按 `outcome=` 查。子集匹配查不到任何 series，于是读到 0——
   而 0 是一个看起来完全正常的数字，**没有任何错误表明它失效了**。
   MinerU 的 429（provider 限速最该抓的一路）因此一次都不会告警。
   现在由 `slo.PROVIDER_RATE_LIMIT_LABELS` 按来源名给出正确标签。
2. **registry 缺指标时崩溃。** `alerts_from_registry` 把 `None` 传给读取函数，
   在一个"还没限速过"的健康部署上抛 `AttributeError`，让 health 端点 500。

### 错误预算

`ErrorBudget` 的三个数（允许的坏事件、已用、剩余）都由同一对
`good`/`total` 推导，因此不可能互相矛盾。允许坏事件数用 `floor`：
`total=100`、目标 99.5% 时允许 0.5 个坏事件，而**允许半个事件的预算不是一个
可以发放的预算**。

真正该看的是 `burn_rate`（消耗速度相对预期的倍数），它是与窗口无关的量：
同样的比例，在 1000 次请求的窗口里可能完全健康，在 10 次请求的窗口里已经烧穿。
只看达标率的告警在小窗口上会疯狂误报。`BURN_RATE_ALERT = 2.0` 对应 Google SRE
workbook 的 fast burn 档位。

---

## 5. 配额的诚实说明

这一节属于 T4.5（配额），放在这里是因为 `QuotaLedger` 的 docstring 明确把
持久性限制指向了本文件。

配额覆盖**会花钱的四件事**：文档数、累计解析页数、模型调用次数、云端解析次数。
分两类计量方式：

- **文档数 / 页数：从 store 推导。** 它们是状态的函数，读一下就知道。计数器的
  价值在于对**事件**计数，而这两项没有事件——用计数器记"上传了几份"意味着
  一次删除之后计数不回落，于是配额会在用户删掉文档之后仍然拒绝他。
- **调用次数：必须计数。** 一次 LLM 调用过去就过去了，没有"这一行"。

**限制：调用次数的计数器是进程内的，进程重启会清零。** 因此托管部署在重启后
可以再花一份配额。对单实例、且重启是运维主动行为的部署可以接受；
**对多实例部署它是不够的**，而做够需要 Redis——那是 T4.5 未完成的部分（第 6 节）。

拒绝行为是一个**异常**而不是一个返回码：它必须在花钱之前抛出。一个"已超额"的
返回值如果由调用方选择是否理会，那么漏掉理会的那条路径就会先花钱再发现问题。

---

## 6. 还没有做的，以及需要什么

| 项 | 状态 | 需要什么 |
|---|---|---|
| 检索时延的实测 | 未测，3 行按目标对待 | 一个检索基准脚本 + 报告提交进仓（与本表同格式），并把 `evidence_ref` 指过去。个人档位是全表扫描，耗时随语料线性增长，因此不存在与语料无关的 P95 |
| docling 解析时延的实测 | 未测，7 行按目标对待 | docling 首次运行需要下载模型权重。本次环境无网络，测不了。T0.9 bake-off 届时有模型与语料，用同一脚本重跑 |
| MinerU 解析时延的实测 | 未测，5 行按目标对待 | 需要有 `MINERU_API_TOKEN` 的环境与付费出网，且必须与本地解析做**同语料**对照 |
| LLM 调用时延 | 1 行目标 | 取决于部署方选的模型与供应商，本项目无法在离线环境测得。上界参考 T4.1 的 `OBSERVABILITY_SLOW_LLM_MS=30000` |
| 配额计数器跨进程持久化 | 未做 | Redis。属 T4.5 剩余部分，**仅当公开部署**时才需要 |
| 更大的页数桶 | 样本不足 | 往 `eval/corpus/` 补 81 页以上的文档，尤其是扫描件与复杂版式 |

### 运行时 SLO 评估

`slo.slo_report()` 拿一批 `SLOSample` 与目标表比对，产出一份达标率与违约清单。
两个刻意的设计：

- **没有目标的样本不计入 `checked`**，只计入 `missing_target`。把"我们没定过
  这个组合"算成"不达标"，会让报告在没有目标的地方报红，而看报告的人会去修
  那个红——真正要修的是"缺一个目标"。
- **检索样本必须自带 `corpus_chunks`。** 不填就返回 `None`（不参与判定），
  而不是回落到最小的语料档。拿 1k 档的目标去判一个 7k 语料的查询是一个
  **必然误报**的规则：语料一变大，每一次正常查询都会被报成超标。

---

## 7. 与其它任务的关系

- 指标来自 T4.1（`paperwhisperer/core/observability.py`）：
  `jobs.queue_depth`、`llm.failures_total`、`parsing.attempts_total`。
  **本任务不新增指标。**
- 配额来自 T4.5（`paperwhisperer/api/tenant.py`），见第 5 节。
- 审计来自 T4.5 / D4（MinerU 云解析），与时延无关，但同属"云调用全部有审计"
  那条验收标准。
- `docs/RELEASE_GATES.md` 里的门槛是**质量**门槛（引用对齐率、召回率等），
  与这里的**时延**门槛是两类东西，不要互相替代。
