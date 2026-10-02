# 标注工作流（Labeling）——两条门禁的输入是怎么造出来的

`docs/RELEASE_GATES.md` 记着两条 `BLOCKED` 的门禁：`citation_precision` 与
`retrieval_recall_at_k`。harness、schema、scorer 都已就位，**缺的是标注**。
本文说明 `tools/label_retrieval.py` 怎么把"一个人标记哪些段落能回答问题"
从一件下午的苦活变成一份可续做、可回改、来源可查的日志，并诚实交代
**本轮样本有多大、它能支撑什么、还差多少**。

一句话现状：**20 条 query、2 个独立标注者、31+33 条 claim 判定已落盘；
按 `eval.gates.load_labels` 原样加载通过；但全部是 AI 起草、没有人确认，
因此仍不可作为门禁真值。** 详见 §5「诚实缺口」。

---

## 0. 这份文档回答什么

| 问题 | 答案在哪 |
|---|---|
| 怎么跑一次标注 | §1 |
| 输出去哪、loader 收不收 | §2 |
| 人和机器的界线画在哪 | §3 |
| 两个标注者一致吗 | §4 |
| 样本能/不能支撑什么 | §5 |
| 每个置信档位还差多少条 | §6 |
| 剩余阻塞项 | §7 |

---

## 1. 工作流

一条命令链，每一步都可单独重跑。**没有任何一步依赖 LLM 端点**
（`:8084` 在争用下持续 429，把标注工具建在它上面等于建在一个会随机
拒绝你的仪器上）。分词、稠密向量走本地 `fastembed`，检索复用生产代码
`paperwhisperer.rag.retrieve` 的 `lexical_query_forms` / `query_coverage` /
`reciprocal_rank_fusion`——预排不是另写一套打分，而是**产品自己的那条
腿**，免得预排和产品对"什么相关"产生分歧。

```bash
# 1) 一次性：解析语料、切窗口、给每条 query 预排候选 → 缓存到 worklist
python tools/label_retrieval.py build --seeds eval/labels/query_seeds.yaml

# 2) 阅读面：打印每条 query 的候选段落（标注者读这个）
python tools/label_retrieval.py dump                    # 全部
python tools/label_retrieval.py dump --query q004       # 单条
python tools/label_retrieval.py dump --language zh      # 按类别

# 3a) 交互标注（可续做；b 回去改主意）
python tools/label_retrieval.py label --annotator human:ann

# 3b) 或：读 dump 后写 answer sheet，批量落盘（AI 起草给人改用这个）
python tools/label_retrieval.py label --annotator agent:a \
    --answers eval/labels/sheet_A.yaml

# 4) 第二轴：claim 是否有据（独立于相关性）
python tools/label_retrieval.py claims --annotator human:ann \
    --candidates-out eval/labels/claim_candidates.json
python tools/label_retrieval.py claims --annotator human:ann \
    --answers eval/labels/sheet_claims.yaml

# 5) 两人一致性
python tools/label_retrieval.py agreement human:a human:b

# 6) 进度 / 导出
python tools/label_retrieval.py status
python tools/label_retrieval.py export --annotator human:ann --annotator agent:a
```

### 交互标注的按键

```
回车   接受预填（记为 human_accepted_prefill，比手打弱一档）
1,3-5  选中候选段落
none   这些候选都不对（0/n 也行）
w 3    看第 3 段全文
s 词   全文搜词，把命中段落加到列表末尾
l zh / k table   改这一问的 language / kind
b      回上一问（带着上次的答案——改主意就靠它）
x      跳过（resumable，status 可见）
q      保存退出
```

### 让"改主意"便宜——这是设计的重点

- **追加日志，不是原地改。** 每一次标记都是 JSONL 里的一行，
  `latest_decisions` 取每条 query 的最后一行做"当前答案"。
  改了主意 → 追加一行 `action: revise`；重看后没改 → 追加一行
  `action: reaffirm`（**重看本身也是信息**，被记录而非丢弃）。
- **坏行不锁死。** 写坏的一行跳过，其余照常读——中断的会话能续。
- **复核模式。** `--review` 只显示已决定过的 query，且预填上次答案，
  一条一条往下过。
- **拖尾审计。** 每条 query 除排序前 8 名外，额外展示 2 段
  **排序没有提名** 的段落（`origin: audit_tail`）。没有它，
  标注者只能在检索器已经提出的候选里挑，recall 会退化成
  "候选列表的 recall"而非"检索器的 recall"。

### 一条 label 花了多久

| | 标注者 A | 标注者 B |
|---|---|---|
| 相关性 20 条 | 46.0 min（138 s/条） | 15.2 min（46 s/条） |
| claim 31 / 33 条 | 8.6 min（17 s/条） | 11.6 min（21 s/条） |
| **合计** | **54.6 min** | **26.8 min** |

差异很大（A 反复核原文、B 更快），均值**相关性 ~92 s/条、claim ~19 s/条**。
`build` 解析 7 份文档 20 条 query 共 ~17 s。**一次性的机器成本 ≈ 20 s；
人的成本是分钟级的，每条 20–140 s。** 这是"100 条 ±5% 门禁"能成立的量级。

---

## 2. 输出：正好是 scorer 读的那一份

`eval/labels/retrieval_labels.yaml`，schema `paperwhisperer-eval-labels/0.1`。
**用 `eval.gates.load_labels` 原样加载通过**（见 §5 的实测输出），
未改一行 loader。字段：`queries[]`（每条带 `language`/`kind`/
`relevant_element_ids`）与 `claims[]`（每条带人工 `supported` 布尔）。

**一条 query 只出一行。** 两个标注者标同一批问题是常态，而 scorer 以 id
为键（recall 查 `retrieved[query_id]`，claim 每 id 一个 `supported`）。
重复 id 会把每条 query 在 recall 分母里数两遍。`collect_export` 因此按 id
**合并**：query 取 `relevant_element_ids` 的**并集**（recall 只问"top-K 里
有没有一段能回答"，某人认可的那段确实能回答；取交集会白扔一个正确标签），
claim 分歧时取**保守值 `supported: false`**（precision 门禁的保守读法是
"未确立"而非"大概行"）。分歧写在行上的 `annotator_element_sets` /
`annotator_votes`，不藏。

> **这不是我加的需求，是 scorer 的既有契约。** `load_labels` 拒绝空
> `relevant_element_ids`，理由是"没有真值的 query 只会虚高分母"——
> 同一原则下，重复 id 同样会让分母失真。两个轴（语言、类别）保持独立，
> 绝不合并。

---

## 3. 人和机器的界线画在哪

工具可以**排、聚类、预填**，但**工具产出的不是标签，直到一个人确认**。
落地成三档记录来源，写进每一行：

| 来源 | 含义 | `human_confirmed` |
|---|---|---|
| `human` | 标注者自己打出了相关集 | `True` |
| `human_accepted_prefill` | 标注者在预填上按了回车 | `True`（但弱一档，单独计数） |
| `assistant_draft` | 助手起草，**没有人确认** | `False` |

机制上兑现，不只是约定：

- `export` **默认拒绝**把 `assistant_draft` 写进门禁输入文件，
  它们落到 `pending_labels.yaml`；要写必须显式
  `--accept-assistant-labels`。
- 覆盖写出去的文件**三重自证**：顶部 `!! NOT GATE-GRADE` 横幅、
  顶层 `human_confirmed: false`、每行 `provenance: assistant_draft`。
- claim 轴更严：answer sheet 里 `supported` 必须是真布尔，
  工具拒绝 `null`——**引用是否有据只有人能判**。
- `skip` 与 `none` 分开：`none`（候选都不对）是**一个决定**，
  `skip`（没判）是**没做决定**，前者是标签，后者不是。

回车为什么单列一档：手打相关集是"决定"，在预填上按回车是"接受"。
都是人的动作，但它们关于"这个标签有多可信"提供的证据不同，所以分开计数
（`status` 里 `sources` 可查）。

---

## 4. 两个标注者的一致性

两位**独立**标注者（`agent:a` / `agent:b`）标了同一批 20 条 query，互不可见。
报告在 `eval/labels/agreement.json`，`agreement` 子命令可复算。

| 指标 | 值 | 读法 |
|---|---|---|
| 两人重叠 query | 20 / 20 | 全重叠 |
| **答案集合完全一致** | **70%**（14/20） | 主要指标 |
| 平均 Jaccard（窗口级） | **0.821** | 交集/并集 |
| 窗口级 Cohen's κ | **−0.155** | ⚠️ 见下 |
| 混淆矩阵 | both=27, only-A=7, only-B=4, both-irrelevant=**0** | |
| 分类（language/kind）分歧 | 5 条（q002/q008/q010/q019/q020） | 其中 q002/q008/q010 是 `formula` vs `prose` 之争 |

**κ 为负、而集合一致率 70%，不是矛盾，是 kappa 的经典 prevalence 悖论**：
混淆矩阵里 `both_irrelevant = 0`——38 个被判定的窗口中两人**至少有一人**
认为相关，几乎不存在"两人都判无关"的窗口，于是期望一致率
`p_e = 0.749` 高于实测一致率 `p_o = 0.711`，κ 变负。这种数据分布下
κ 不该作为主指标；**70% 集合一致率 + 0.821 Jaccard 才是可解释的**。
两人都"宽判"（宁多勿漏）而非互相矛盾。

**分歧集中在语义判断**：q002/q008/q010 的 kind（γ/β 定义、rank-N 关系、
bulk↔CFT 相关子）该算 `formula` 还是 `prose`；q019/q020 的
language（`zh` vs `mixed`）。**元素定位本身分歧不大**——这说明
"哪一段能回答"比"这一问属于哪个桶"客观得多。

> **这是 AI–AI 一致性，不是人–人。** 两个助手用同一套指令、同一份语料、
> 同一个模型家族，不构成独立的人类判断。它测的是**任务说明书是否无歧义**，
> 不能替代真人复核。人–人 κ 通常低于 AI–AI κ，所以**下表只能当乐观上界**。

---

## 5. 样本能支撑什么 / 不能支撑什么

### 实际产出

| 量 | 值 |
|---|---|
| 标注 query | **20**（3 份中英混排 + 4 份真实 PDF，覆盖 en/zh/mixed × prose/table/formula） |
| claim 判定 | A 31 条 + B 33 条 → 合并 **34 个唯一 claim id** |
| 导入标签文件 | **19 queries / 34 claims**（q010 两位标注者都判 none_relevant，见下） |
| 导出文件 `human_confirmed` | **False**（全部 AI 起草，无人确认） |
| claim 精度（保守合并） | 2/34 = **0.059**，Wilson 95% CI [0.016, 0.191] |
| claim 精度（宽松并集，仅参考） | 6/34 = 0.176 |

### 能支撑

- **门禁开始有牙齿**：recall / precision 两条从今天起能算出数字，
  按语言、按类别分开，不再是 `BLOCKED`。
- **候选检索器的问题被量化**：两位标注者**都**发现 GLUE 的 Table 1
  不在预排前 8（q004/q005 都是）——现有排序偏爱正文、漏掉表格本身。
  这是标注工作**顺带产出的检索器诊断**，不是门禁数字。
- **格式被真实 loader 验过**：`eval.gates.load_labels` 原样加载通过。

### 不能支撑

- **不是门禁真值。** 全部 `assistant_draft`，没有人确认。按
  `RELEASE_GATES.md` 的立场，拿系统自己（或系统近亲）产出的标签打分，
  就是系统给自己打分。**这两门在确认之前仍然应该 `BLOCKED`。**
- **20 条撑不起 ±5%。** 见 §6。
- **claim 的 0.059 不是产品精度。** claim 候选是**抽取式 top-2 窗口**
  的引文，是"这条引文能不能回答问题"的代理指标；真实门禁量的是
  完整问答链路生成的引用。即便取最宽松的读法（至少一位标注者认可 =
  6/34 = 0.176），也远低于 95% 阈值——**这说明的是抽取式代理的判据严苛，
  不是说产品引用精度只有 6%**。一个断言没有 LLM 证据链就宣称产品精度
  5.9%，是把代理说成了度量。
- **`none_relevant` 的 query 被排除在分母外。** q010（两条都判"候选里
  没有答案"）**无法写进标签格式**——`load_labels` 拒绝空
  `relevant_element_ids`，理由正是"空真值会让召回分母虚高"。于是
  "答不出来"的 query 从分母消失，**recall 有一个已知的向上偏差**。
  每一处都在导出的 warnings 里逐条列出。

---

## 6. 每个置信档位还差多少条

Recall 门禁（Recall@K ≥ 90%，一侧 95% 区间），与 `docs/RAG_MEASUREMENT.md`
的表一致：

| 目标区间 | 需要 query 数 | 本样本 20 条后**还差** |
|---|---|---|
| ±10% | 25 | **5** |
| **±5%** | **98** | **78** |
| ±2% | 609 | **589** |

Precision 门禁（≥95%）：要让 95% 这个阈值的单侧 95% 下界真的站得住 95%，
claim 数取决于**真实精度**（若真实值就是 95%，永远不够——阈值即真实值时
区间必跨阈值）：

| 假设真实精度 | 需要 claim 数 | 本样本 34 后还差 |
|---|---|---|
| 97% | ~197 | **~163** |
| 98% | ~59 | **~25** |
| 99% | ~17 | 已够 |

**现实的一条线：先把 recall 打到 ±5%（≈98 条 query，每条按 ~90 s，
≈2.5 人时）是一条清晰的里程碑；precision 的 95% 门禁要么等真实精度
明显高于阈值，要么承认这个量级下它就是 Beta 目标。**

---

## 7. 剩余阻塞项（诚实缺口）

1. **没有人确认。** 本样本是 AI 起草。要变成门禁真值，需要一个人
   （`human:xxx`）过一遍并 `export`。工具已把"确认"做成一道命令，
   不需要改代码。
2. **Recall 门禁还缺第二个输入：`retrieval_run.json`。** 有标签也要知道
   检索器**实际返回了哪 K 个 element id**。本轮**刻意没有伪造**这个文件：
   用标注工具自己的候选排序去填它，就是拿 labeller 的排序给 labeller 的
   标签打分——正是这个任务要防的自评。**它必须由产品的检索器产出**
   （对 `worklist.json` 里的 20 条 query 跑一次 `HybridRetriever.retrieve`，
   写 top-K element id）。这一步是产品侧的活，不在本工具的可写范围。
3. **单元素文档的 zh 类别不可分辨。** 4 份 zh/中英混排 PDF 在 legacy
   parser 下每份只解析出 **1 个 element**，其 Recall@K 恒等于"top-K 里有没有
   唯一那一个"——没有区分度。要让 `lang:zh` 桶有牙齿，需要 docling 或更多
   多页中文文档。每条 seed 的 `note` 里都写了这一点。
4. **两轴样本量不同。** recall 20 条、claim 34 条。precision 要 ~200 条
   claim 需要更多 query（或每条多判几条 claim）。工具已支持
   `--per-query`，不需改代码。

---

## 附：一次完整复现

```bash
python tools/label_retrieval.py build --seeds eval/labels/query_seeds.yaml
python tools/label_retrieval.py dump                       # 读候选
# 写 sheet_A.yaml / sheet_B.yaml（各 20 条），source: assistant_draft
python tools/label_retrieval.py label --annotator agent:a --answers eval/labels/sheet_A.yaml
python tools/label_retrieval.py label --annotator agent:b --answers eval/labels/sheet_B.yaml
python tools/label_retrieval.py agreement agent:a agent:b --out eval/labels/agreement.json
python tools/label_retrieval.py export --annotator agent:a --annotator agent:b \
    --accept-assistant-labels          # 无人确认时才用这个覆盖；默认拒绝

# 确认 loader 原样接受：
python -c "from eval.gates import load_labels; from pathlib import Path; \
    L=load_labels(Path('eval/labels/retrieval_labels.yaml')); \
    print(len(L.queries),'queries',len(L.claims),'claims')"
```

**单测**：`python -m pytest tests/test_label_retrieval.py`（31 passed，
含"loader 原样接受导出"、"AI 标签默认不进 gate 文件"、"同 id 合并"等）。