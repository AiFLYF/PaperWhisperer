# Parser bake-off 与路由决策记录（PW-006 / T0.9）

> **状态：已决（DECIDED）。本轮 bake-off 已实测完毕，路由决策见第 6 节。**
>
> **作废声明**：`eval/reports/run1-local/`（2026-09-30 12:20）**整份作废，不得引用**。
> 该次运行测的是一个坏掉的 docling 适配器——它产出 0 个带 bbox 的元素、0 条 SourceLocator，
> 公式全丢、表格维度错误。详见 §0.1。本文档中所有 docling 数字来自重跑。

| 字段 | 值 |
|---|---|
| 语料清单 | `eval/manifest.yaml`（`paperwhisperer-eval-manifest/0.2`，created 2026-09-29） |
| 语料规模 | 52 份常规文档 / 12 类别，另 4 份 `adversarial/`（不进批跑，可 `--include-excluded` 单独跑） |
| `corpus_version` | `sha256:0dfe319761a7cc9d003dd71c00be7ca57f912bf529654b831cd61d5da8f5dda1` |
| 评测脚本 | `eval/run_parser_eval.py`（runner version 1.0） |
| 本轮 bake-off | **已运行**（2026-09-30 18:19–18:52，52/52 份文档 × 2 解析器，0 失败 0 超时） |
| MinerU cloud | **未运行**，上传腿被 provider 拒绝（§3.3），整列留空 |
| docling-serve | **未运行**，本机无 docling-serve 容器 |
| 决策人 | 记录者：T0.9 bake-off 重跑 |
| 决策日期 | 2026-09-30 |

### 本轮实测所用的环境（复现必需）

| 变量 | 值 | 为什么是这个值 |
|---|---|---|
| `PARSER_TIMEOUT_SECONDS` | **900** | 上一轮用默认 120s，docling 有 5 份文档被超时截断，**被记成了解析失败**。抬高后 0 超时（§0.2）。 |
| `PARSER_BACKEND` | `gateway` | 与生产主链路一致 |
| `PARSER_DOCLING_MODE` | `inprocess` | 本机无 docling-serve |
| `HF_ENDPOINT` | `https://hf-mirror.com` | 模型权重下载走镜像 |
| 硬件 | 4× RTX 3090（本轮仅用 CPU，OCR 走 AzureExecutionProvider） | 串行单进程，未开并发 |

---

## 0. 作废声明：上一轮 run1-local 的数字一律不得引用

### 0.1 为什么作废

`eval/reports/run1-local/`（generated 2026-09-30T12:20:09+00:00，
`corpus_version` 与本轮**相同**）记录的 docling 列为：
`page_coverage 0.0000`、`locator 0.0000`、`bbox 0.0000`、`reading_order n/a`、`质量 FAIL 52/52`。

**这整组数字无效。** 同一份 `corpus_version` 下重跑，docling 变为
`page_coverage 0.9991`、`locator 0.9231`、`bbox 0.9231`、`reading_order 1.0000`、`质量 FAIL 7/52`。
两次不是同一件东西——中间适配器被修过。以 `hinton_distillation.pdf` 为例：

| | run1-local（坏） | 本轮（修后） |
|---|---|---|
| 元素数 | 96 | **102** |
| role 分布 | `text 91, table 5` | `heading 20, text 50, list_item 9, footer 9, formula 6, footnote 2, header 1, table 5` |
| 带 bbox | **0 / 96** | **102 / 102** |
| 带 LaTeX 的公式 | 0 | **6** |
| 表格维度 | (5,4)(4,2)(4,4)(13,5)(5,4) — 错 | (4,3)(3,1)(3,3)(12,4) — 与 docling 自报一致 |

> **`corpus_version` 相同而数字不同，是这份文档里最重要的一条可复现性警告：**
> 语料版本固定**不代表**解析器版本固定。bake-off 数字只在 (corpus_version × parser 版本 × 适配器代码)
> 三者同时固定时才可比。任何重跑都必须重新抄全部数字，不能只补差异。

### 0.2 上一轮的"超时"其实是超时，不是解析失败

上一轮 docling 有 5/52 份文档超过 120s 默认超时，被记为失败。本轮把
`PARSER_TIMEOUT_SECONDS` 抬到 900 后，**这 5 份全部成功**，`failure_rate` 归 0：

| 文档 | 本轮实测耗时 | 状态 |
|---|---|---|
| `formula_dense/dbrane_theory_review.pdf` | **424 425 ms（424.4 s）** | success |
| `long_paper/gpt3_language_models.pdf` | 169 021 ms | success |
| `digital_complex_layout/quantum_supremacy_nature_style.pdf` | 154 189 ms | success |
| `table_dense/helm_holistic_eval.pdf` | 151 175 ms | success |
| `formula_dense/green_tao_primes.pdf` | 138 677 ms | success |

**424.4 s 是这份语料上 docling 的真实最坏耗时**，不是超时值。它超过了 `PARSER_TIMEOUT_SECONDS`
的默认值 120s 达 3.5 倍——这是下面路由决策里最贵的一个数字。

### 0.3 仍然缺失的列（本轮没有、也无法产出）

| 列 | 状态 | 原因 |
|---|---|---|
| MinerU cloud | **留空** | 上传腿被 provider 侧拒绝（§3.3），不是"没跑"，是"跑不了"。 |
| docling-serve | **留空** | 本机没有 docling-serve 容器。serve 模式与 inprocess 的质量应当一致，但**未经实测**。 |
| OCR 字符准确率 | **留空** | manifest 无逐条 ground truth 映射（§3.2）。 |

---

## 1. 这份文档要回答什么

P0 退出标准里有一条是"**bake-off 报告与默认路由决策已落档**"。本文件是该条目的落档位置，包含两部分：

1. **指标表**：在同一语料上跑 legacy / Docling / MinerU cloud 的对比数字（第 4 节）。
2. **决策记录**：默认解析器与 fallback 规则（第 6 节）。

### 1.1 可复现性：corpus_version

每份报告的 Markdown 头部、JSON 顶层和 CSV 旁注都带两个版本串，**名称不同、含义不同**：

| 字段 | 来源 | 作用 |
|---|---|---|
| `corpus_version` | `eval/manifest.yaml` 的顶层字段，由 `eval/build_manifest.py` 计算 | **报告声明的语料身份**。填第 4 节表格时抄这一个 |
| `runner_entry_digest` | 评测脚本对**它实际加载到的条目**（`file` + `sha256`）的本地哈希 | **交叉校验，不是第二个身份** |

本轮两次运行（`run2-inprocess` 与 `run3-force-ocr`）的 `runner_entry_digest` 同为
`414ebe9983b807eb9d64c2101cd28c0553fe70a73bb017c3473e39c9ca7ef5c0`，与 run1-local 一致——
即**三次加载到的文件集合完全相同**，差异只来自适配器，不是语料漂移。

`runner_entry_digest` 与 `corpus_version` 在同一份语料上本来就不相等，这是预期行为。
**不要把 `runner_entry_digest` 当作语料版本引用。**

> **跨版本比较是无效的。** `--fail-on-regression` 在 baseline 与本次 `corpus_version` 不一致时
> 会直接判失败并拒绝比较。语料更新后必须在新语料上重跑 baseline。

---

## 2. 怎么把数据跑出来

全部命令在仓库根目录执行。

```bash
PYTHON=/home/ubuntu/miniconda3/bin/python3

# 1) 可用性自检：确认 docling 装没装（注意：--list-parsers 默认只列 legacy，
#    要看 docling 必须显式 --parsers legacy,docling）
$PYTHON -m eval.run_parser_eval --list-parsers --parsers legacy,docling
```

### 2.1 第一步：本地对照（本轮实际执行的命令）

```bash
HF_ENDPOINT=https://hf-mirror.com \
PARSER_TIMEOUT_SECONDS=900 \
PARSER_BACKEND=gateway \
PARSER_DOCLING_MODE=inprocess \
$PYTHON -m eval.run_parser_eval \
    --parsers legacy,docling \
    --out eval/reports/run2-inprocess
```

产出：`eval/reports/run2-inprocess/parser_eval.{md,csv,json}`。本轮耗时约 **32.7 分钟**
（docling 累计 1 899 s，legacy 累计 80 s，串行）。

### 2.2 强制 OCR 对照（扫描件子集）

```bash
HF_ENDPOINT=https://hf-mirror.com PARSER_TIMEOUT_SECONDS=900 \
PARSER_BACKEND=gateway PARSER_DOCLING_MODE=inprocess \
$PYTHON -m eval.run_parser_eval --parsers legacy,docling --force-ocr \
    --categories scanned_clear,scanned_lowq,hybrid_scan_text \
    --out eval/reports/run3-force-ocr
```

### 2.3 adversarial 对照

```bash
HF_ENDPOINT=https://hf-mirror.com PARSER_TIMEOUT_SECONDS=900 \
PARSER_BACKEND=gateway PARSER_DOCLING_MODE=inprocess \
$PYTHON -m eval.run_parser_eval --parsers legacy,docling --include-excluded \
    --out eval/reports/run4-adversarial
```

> `--include-excluded` 是**把 adversarial 加进批跑集合**，不是"只跑 adversarial"——
> 它会连带重跑全部 52 份常规文档（约 33 分钟）。

### 2.4 未执行的两步（原样保留，说明为什么没做）

- **docling-serve 对照**：本机无 docling-serve 容器，无从测起。
- **云端对照**：MinerU 上传腿被拒（§3.3），即使四道门全开也只会得到一列 `unavailable`，
  不会产生真实数字，故未执行，也未消耗任何额度。

### 2.5 冒烟与单类别调参

```bash
$PYTHON -m eval.run_parser_eval --parsers legacy,docling --limit 3 --out /tmp/eval-smoke
$PYTHON -m eval.run_parser_eval --parsers legacy,docling --categories table_dense
$PYTHON -m eval.run_parser_eval --parsers legacy,docling --out eval/reports/run2 \
    --fail-on-regression eval/reports/run1/parser_eval.json
```

Makefile 里有对应目标：`make eval-bakeoff` / `eval-bakeoff-smoke` / `eval-bakeoff-adversarial` /
`eval-bakeoff-parsers`，可用 `RUN=<目录名>` 指定输出目录。

---

## 3. 指标定义（每一项都从 canonical `ParsedDocument` 推导）

所有指标都在 `paperwhisperer/ingestion/schema.py` 的 `ParsedDocument` 上计算。

| 指标 | 定义 | 怎么来的 | 什么时候是 n/a |
|---|---|---|---|
| `page_coverage` | 出现正文的页数 ÷ 解析器自报页数 | body 元素中 **`pdf_page` kind** locator 的 distinct page ÷ `document.page_count` | 解析器自报 `page_count <= 0`（DOCX/PPTX/TXT 中满足者） |
| `reading_order` | 正文在主坐标上单调的比例 | 相邻元素对中"坐标不倒退"的比例 | 可比元素少于 2 个 |
| `tables 2D/total` | 保留行列结构的表格数 ÷ 表格总数 | ≥2 行 ≥2 列、非空壳 → 计 2D；被拍平成文本的单独计 `tables_flattened` | 该类别语料中表格总数为 0 |
| `formulas LaTeX/total` | 带 LaTeX 的公式数 ÷ 公式总数 | `role == "formula"` 或 `Element.latex` 非空 | 该类别语料中公式总数为 0 |
| `locator` | 可解析 `SourceLocator` 的正文元素比例 | 直接用 `ParsedDocument.locator_coverage` | 正文为空（记 0） |
| `bbox` | 上述 locator 中带 bbox 的比例 | `locator.is_resolvable` 且 `locator.bbox is not None` | 无可解析 locator |
| `chars/manifest` | 解析出正文字符数 ÷ manifest 记录字符数 | `document.char_count` ÷ `entry.chars_total` | manifest 未记录文本层 |
| `ms` | 外层墙钟耗时 | `time.monotonic` 覆盖整个适配器调用 | 解析器未运行（记 0） |
| `failure_rate` | 失败数 ÷ 实际尝试数 | 分母只含 success/partial/failed | 未尝试（`None`，**不是 0**） |
| `quality_status` | 质量门总判 | 直接跑 `ingestion.quality.evaluate_quality` | 解析失败 |

### 3.1 读这张表之前必须知道的五件事

1. **`n/a` 与 0 不同。** `n/a` 表示"在这个格式/这份文档上不可测"，`0` 表示"测了，是零"。
2. **`failure_rate` 对没跑过的解析器是 `None` 而不是 `0`。**
3. **`reading_order` 测的是"页面是否乱序"，不是"双栏是否读对"。** 跨栏顺序错误**不会**降低这个指标——
   本轮两个解析器都是 1.0000，但这**不能**用来说明双栏读对了。见 §3.2 的缺口表。
4. **`chars/manifest` 是召回的代理值，不是 ground truth。** 分母来自 pypdf 的文本层提取，
   在缺 ToUnicode CMap 的文档上少算。本轮 PDF 上普遍 > 1（如 `long_paper/planck2018_cosmo_params.pdf`
   legacy 12.10 / docling 11.79），**说明分母偏低，不是解析器多算了**。
5. **⚠️ `page_coverage` 只统计 `pdf_page` 类型的 locator。** 这一点在 PPTX 上直接决定数字的含义，见 §4.2.1。

### 3.2 本轮仍未测的东西（已知缺口）

| 缺口 | 影响 | 补法 |
|---|---|---|
| OCR 字符准确率 | `scanned_clear` / `scanned_lowq` 只能看"有没有文字"，看不出对不对 | `eval/ground_truth/` 有合成扫描件原文，但 manifest **没有逐条 ground truth 映射**。需先补一条 `ground_truth: <file>` 字段 |
| 双栏阅读顺序正确率 | `digital_two_col`（10 份）的核心关注点未被覆盖 | 需人工标注一小份对照集 |
| 跨页表格 | `table_dense` 关注点之一 | 需在 manifest 中标注"哪些表跨页" |
| 端到端检索指标 | bake-off 只到解析层 | 属 T1.2/T1.7 范围，不在 T0.9 |
| docling-serve 模式 | 与 inprocess 的质量差异未测 | 需起本地容器后按 §2 补跑 |

### 3.3 云端链路实测记录（2026-09-29，阻塞 bake-off 的 MinerU 列）

用真实 `MINERU_API_TOKEN` 对 `eval/corpus/digital_single_col/hinton_distillation.pdf`（106 KB / 9 页）跑了一次完整云端链路。**在第二步 PUT 上传处失败**：

| 步骤 | 结果 |
|---|---|
| `POST /api/v4/file-urls/batch` | **200**，token 有效，返回 `batch_id` 与签名上传 URL |
| `PUT <签名 URL>`（上传到 OSS） | **403 `SignatureDoesNotMatch`** |
| `GET /api/v4/extract-results/batch` | 未到达 |
| 下载 + normalize | 未到达 |

同一个签名 URL 用四种方式 PUT 全部返回 OSS 侧同一个 `SignatureDoesNotMatch`（当前实现 /
不带 Content-Type / `application/pdf` / `http.client` 直连绕开 URL 规范化）。
倾向于**账号侧的预签名策略与该上传方式不匹配**，需要 MinerU 侧确认 token 的 API 权限与预签名上传配置。

**因此第 4 节的 MinerU 列整列留空。** 在上传腿打通之前，本仓库的云端代码只有 mock 全链路测试覆盖。
本次 bake-off **未发起任何 MinerU 调用，未产生任何费用**。

---

## 4. 首轮 bake-off 指标

> 语料：`corpus_version sha256:0dfe319761a7cc9d003dd71c00be7ca57f912bf529654b831cd61d5da8f5dda1`。
> 数据源：`eval/reports/run2-inprocess/parser_eval.{md,csv,json}`（2026-09-30T18:52:03+00:00）。
> **本节所有行同属这一个 `corpus_version` 与同一次运行，可横向比较。**
> 单元格格式 `legacy | docling`。

### 4.1 总体（52 份常规文档）

| 指标 | legacy | docling (inprocess) | docling (serve) | MinerU cloud |
|---|---|---|---|---|
| `corpus_version` | `sha256:0dfe31…a1` | `sha256:0dfe31…a1` | **未运行** | **不可运行** |
| 语料文件是否相同 | 同左 | 同左 | — | — |
| 解析器版本 | 1.0 | 2.131.0 | — | — |
| Usable / attempted | 52/52 | 52/52 | — | — |
| 失败率 | **0.0000** | **0.0000** | — | — |
| 平均页覆盖 | **0.7708** | **0.9991** | — | — |
| 平均阅读顺序 | 1.0000 | 1.0000 | — | — |
| 平均 locator 覆盖 | **0.8462** | **0.9231** | — | — |
| 平均 bbox 比例 | **0.0000** | **0.9231** | — | — |
| 表格 2D 比例 | 1.0000（分母仅 **3**） | 0.9639（分母 277） | — | — |
| 公式 LaTeX 比例 | **n/a（公式总数 = 0）** | **1.0000（1160/1160）** | — | — |
| 中位耗时 (ms) | **283** | **14 914** | — | — |
| 质量门 FAIL 数 | **12** | **7** | — | — |
| 质量门 WARN 数 | 1 | 27 | — | — |
| 单次成本 | 0 | 0（CPU/内存） | 0（需自备容器） | **不可测** |

**耗时分布（外层墙钟，串行单进程）**

| | min | 中位 | p90 | max | 合计 |
|---|---|---|---|---|---|
| legacy | 1 ms | 283 ms | 3 218 ms | 25 838 ms | 80 s |
| docling | 207 ms | 14 914 ms | 100 118 ms | **424 425 ms** | **1 899 s** |

> **两个分母陷阱，读表前必须看到：**
> 1. legacy 的"表格 2D = 1.0000"是在 **3 张表**上取得的，且这 3 张全部来自 DOCX。
>    在 5 份 `table_dense` PDF 上 legacy 检测到 **0 张表**。与 docling 的 0.9639 并列比较会得出完全错误的结论。
> 2. legacy 的"公式 LaTeX = n/a"不是"做得好"，是**一个公式都没识别出来**（公式总数 = 0）。
>    docling 在同一语料上识别 1 160 个公式，1 160 个带 LaTeX。

**质量门失败原因分布**

| 解析器 | FAIL 数 | `page_coverage` | `text_density` | `locator_coverage` | `fragmented_lines` |
|---|---|---|---|---|---|
| legacy | 12 | 11 | 12 | 0 | 1 |
| docling | 7 | 0 | 3 | **4** | 0 |

### 4.2 分类别（每格：`legacy | docling`）

| 类别 | 关注点 | 失败率 | 页覆盖 | 阅读顺序 | locator | bbox | 表格 2D | 公式 LaTeX | 中位 ms |
|---|---|---|---|---|---|---|---|---|---|
| `digital_single_col` (5) | 基线质量 | 0.0000 \| 0.0000 | 1.0000 \| 1.0000 | 1.0000 \| 1.0000 | 1.0000 \| 1.0000 | **0.0000 \| 1.0000** | n/a(0) \| 0.9643 (28) | n/a(0) \| 1.0000 (35) | 377 \| 19 526 |
| `digital_two_col` (10) | 阅读顺序恢复 | 0.0000 \| 0.0000 | 1.0000 \| 1.0000 | 1.0000 \| 1.0000 | 1.0000 \| 1.0000 | **0.0000 \| 1.0000** | n/a(0) \| 0.9750 (80) | n/a(0) \| 1.0000 (38) | 312 \| 19 436 |
| `digital_complex_layout` (2) | 阅读顺序极限 | 0.0000 \| 0.0000 | 1.0000 \| 1.0000 | 1.0000 \| 1.0000 | 1.0000 \| 1.0000 | **0.0000 \| 1.0000** | n/a(0) \| 1.0000 (14) | n/a(0) \| 1.0000 (135) | 6 433 \| 85 209 |
| `formula_dense` (5) | LaTeX 保留 | 0.0000 \| 0.0000 | 1.0000 \| 0.9985 | 1.0000 \| 1.0000 | 1.0000 \| 1.0000 | **0.0000 \| 1.0000** | n/a(0) \| 1.0000 (3) | n/a(0) \| **1.0000 (751)** | 675 \| 57 832 |
| `table_dense` (5) | 表格二维结构 | 0.0000 \| 0.0000 | 1.0000 \| 0.9963 | 1.0000 \| 1.0000 | 1.0000 \| 1.0000 | **n/a(0) \| 0.9841 (63)** | n/a(0) \| 1.0000 (46) | 1 540 \| 30 578 |
| `scanned_clear` (5) | OCR 触发 | 0.0000 \| 0.0000 | **0.0000 \| 1.0000** | n/a \| 1.0000 | **0.0000 \| 1.0000** | **0.0000 \| 1.0000** | n/a(0) \| 1.0000 (4) | n/a(0) \| 1.0000 (1) | 2 \| 3 253 |
| `scanned_lowq` (3) | OCR 极限与降级 | 0.0000 \| 0.0000 | **0.0000 \| 1.0000** | n/a \| 1.0000 | **0.0000 \| 1.0000** | **0.0000 \| 1.0000** | n/a(0) \| 1.0000 (2) | n/a(0) \| 1.0000 (1) | 1 \| 3 106 |
| `hybrid_scan_text` (3) | 页级质量判断 | 0.0000 \| 0.0000 | 1.0000 \| 1.0000 | n/a \| 1.0000 | 1.0000 \| 1.0000 | **0.0000 \| 1.0000** | n/a(0) \| **0.5000 (2)** | n/a(0) \| 1.0000 (1) | 4 \| 3 681 |
| `mixed_zh_en` (4) | 多语言 | 0.0000 \| 0.0000 | 1.0000 \| 1.0000 | n/a \| 1.0000 | 1.0000 \| 1.0000 | **0.0000 \| 1.0000** | n/a(0) \| 1.0000 (4) | n/a(0) \| n/a(0) | 8 \| 2 994 |
| `long_paper` (3) | 全文覆盖 | 0.0000 \| 0.0000 | 1.0000 \| 0.9937 | 1.0000 \| 1.0000 | 1.0000 \| 1.0000 | **0.0000 \| 1.0000** | n/a(0) \| 0.9324 (74) | n/a(0) \| 1.0000 (147) | 6 235 \| 100 118 |
| `docx` (4) | element locator | 0.0000 \| 0.0000 | n/a \| n/a | **1.0000 \| n/a** | **1.0000 \| 0.0000** | 0.0000 \| 0.0000 | 1.0000 (3) \| 1.0000 (3) | n/a(0) \| 1.0000 (5) | 13 \| 239 |
| `pptx` (3) | slide locator | 0.0000 \| 0.0000 | **不可比，见 §4.2.1** | 1.0000 \| 1.0000 | 1.0000 \| 1.0000 | **0.0000 \| 1.0000** | n/a(0) \| n/a(0) | n/a(0) \| n/a(0) | 10 \| 259 |

> 括号内是该比率的**分母**（表格数 / 公式数）。`n/a(0)` = 该类别语料里一个表格都没有，
> 比率无从算起——**这不是"解析得完美"，是"什么都没找到"**。

#### 4.2.1 ⚠️ `page_coverage` 在 PPTX 上不可比——上一轮的判断在这里要修正

上一轮把 `pptx page_coverage = 0`（两个解析器都是 0）判为"指标缺口，不是解析器失败"。
**这个判断只对 legacy 成立，对 docling 是错的**，本轮查清了原因：

`page_coverage` 的实现（`eval/run_parser_eval.py:566`）只把 **`kind == "pdf_page"`** 的 locator
计入已覆盖页。两个解析器对 PPTX 用了**不同的 locator 类型**：

| 解析器 | PPTX locator 类型 | `pages_reported` | `pages_covered` | `page_coverage` | `reading_order` 基准 | `locator` |
|---|---|---|---|---|---|---|
| legacy | **`slide`** | 6 / 4 / 8 | 0 | **0.0000** | `slide`，1.0000 | **1.0000** |
| docling | **`pdf_page`** | 6 / 4 / 8 | 6 / 4 / 8 | **1.0000** | `pdf_page`，1.0000 | 1.0000 |

- **legacy 的 0.0000 是指标口径产物，不是缺陷。** legacy 确实定位到了每一页幻灯片
  （`locator_coverage = 1.0000`，`reading_order` 在 `slide` 基准上单调 1.0000），
  只是它的页用 `slide` 表示，指标不认。
- **docling 的 1.0000 是真实测量。** 它把幻灯片也表达成 `pdf_page`，因而落进了指标口径内。

**处理方式**：PPTX 在 `page_coverage` 这一列上**显式排除**，两个数字都不进入跨格式均值。
理由不是"哪个parser对"，而是**两个数字量的不是同一件事**——
把 0 和 1.0000 并排放在一行里，会让读者以为 legacy 漏掉了全部 18 页幻灯片，而它一页都没漏。
其余各列（`reading_order` / `locator` / `bbox`）在 PPTX 上是可比的，docling 的 `bbox = 1.0000` 为 legacy 的 `0.0000`。

> 同理，`mixed_zh_en`（4 份）**每份只有 1 页**，legacy 产出"一个元素一份文档"
> 是语料性质，不是解析缺陷。该类的页覆盖两个解析器都是 1.0000，没有可比性争议。

### 4.3 `adversarial/`（4 份，`--include-excluded`）

数据源：`eval/reports/run4-adversarial/`（2026-09-30T19:50）。这些文件**设计上就应该失败**，
所以不进 §4.1 的失败率，只用来核对错误路径是否明确。

| 文件 | 期望行为 | legacy | docling |
|---|---|---|---|
| `corrupted_truncated.pdf` | 明确报错，不静默返回空文本 | ✅ `failed` / `parse_error`：`ValueError: PDF 读取失败: Stream has ended unexpectedly` | ✅ `failed` / `parse_error`：`docling 解析失败：Conversion failed…` |
| `encrypted_aes128.pdf` | 明确报错，不崩溃 | ✅ `failed` / `parse_error`：`ValueError: PDF 读取失败: File has not been decrypted` | ✅ `failed` / `parse_error`：`docling 解析失败…` |
| `fake_extension_actually_text.pdf` | 识别伪后缀 | ✅ `failed` / `parse_error`：`ValueError: PDF 读取失败: Stream has ended unexpectedly` | ✅ `failed` / `parse_error`：`docling 解析失败…` |
| `zip_bomb_nested.zip` | 拒绝/限制解压 | ✅ `unsupported` / `unsupported_media_type`：`解析器 legacy 不支持 application/zip` | ✅ `unsupported` / `unsupported_media_type`：`解析器 docling 不支持 application/zip` |

**4/4 通过，两个解析器都是。** 没有进程崩溃，没有一份静默返回空文本，
也没有一份把伪后缀当成合法 PDF 放行。`attempted` 由 52 增至 55（新增 3 份可解析的 adversarial），
`usable` 仍为 52——4 份 adversarial 全部未产出可用结果，符合预期。

### 4.4 强制 OCR 对照（`run3-force-ocr`，11 份扫描件子集）

`force_ocr` 在 docling 2.131.0 上**已可用**——上一轮记录的
`AttributeError: 'PdfPipelineOptions' object has no attribute 'backend'` **已修复**，
`_build_converter(docling, force_ocr=True)` 现在返回空 warning 列表。实测结果：

| 指标 | legacy | docling (普通) | docling (force_ocr) |
|---|---|---|---|
| Usable / attempted | 11/11 | 11/11 | 11/11 |
| 页覆盖 | 0.2727 | **1.0000** | **1.0000** |
| locator / bbox | 0.2727 / 0.0000 | 1.0000 / 1.0000 | **1.0000 / 1.0000** |
| 表格 2D | n/a | 0.8750 | 0.8750 |
| 中位耗时 (ms) | 2 | 3 254 | 3 254 |
| 质量门 FAIL | **9** | **0** | **0** |

**关键结论：`force_ocr` 对 docling 是一个 no-op。** 逐份比对字符数，两列**完全相同**：

| 文档 | 普通 docling | force_ocr docling |
|---|---|---|
| `scanned_clear/paper_en_basic.pdf` | 2 331 | 2 331 |
| `scanned_clear/paper_en_formulas.pdf` | 1 292 | 1 292 |
| `scanned_clear/paper_en_twocol.pdf` | 1 279 | 1 279 |
| `scanned_clear/paper_zh_cn.pdf` | 704 | 704 |
| `scanned_clear/report_en_tables.pdf` | 1 290 | 1 290 |
| `scanned_lowq/scan_lowq_blurry.pdf` | 270 | 270 |
| `scanned_lowq/scan_lowq_rotated.pdf` | 2 322 | 2 322 |
| `scanned_lowq/scan_lowq_skew_faint.pdf` | 968 | 968 |
| `hybrid_scan_text/hybrid_en_paper.pdf` | 2 450 | 2 450 |
| `hybrid_scan_text/hybrid_en_tables.pdf` | 1 152 | 1 152 |
| `hybrid_scan_text/hybrid_zh_paper.pdf` | 646 | 646 |

原因是 docling 2.131.0 的转换器**默认就开着 OCR**（`do_ocr=True`），
`force_ocr` 只是把这个默认值再显式写一遍。这条对路由决策有直接影响（§5 策略 C）。

### 4.5 复现性核对：两次独立全量运行

`run2-inprocess`（18:52）与 `run4-adversarial`（19:50）是两次**独立**的全量运行，
语料、解析器、配置全同。逐项比对 summary：

| 指标 | legacy 两次 | docling 两次 | 结论 |
|---|---|---|---|
| `mean_page_coverage` | 0.7708 / 0.7708 | 0.9991 / 0.9991 | **逐位相同** |
| `mean_locator_coverage` | 0.8462 / 0.8462 | 0.9231 / 0.9231 | **逐位相同** |
| `mean_bbox_locator_ratio` | 0.0000 / 0.0000 | 0.9231 / 0.9231 | **逐位相同** |
| `mean_reading_order` | 1.0000 / 1.0000 | 1.0000 / 1.0000 | **逐位相同** |
| `usable / quality_fail` | 52 / 12，两次一致 | 52 / 7，两次一致 | **逐位相同** |
| `median_duration_ms` | 283 / 301（+6%） | 14 914 / 14 554（−2%） | **有抖动** |

**质量类指标在本语料上是确定性的，耗时不是。** 因此引用 bake-off 数字时，
质量列可以逐位对比，而 `ms` 列应视为带 ±几个百分点噪声的量——
本报告第 6 节依赖的 `ms` 结论（中位 53 倍差、最坏 424.4 s）量级远超该噪声，仍然成立。

---

## 5. 候选路由策略

四条候选。每条都有代价，**没有一条是纯赢**。本轮数据对它们的评价：

### 策略 A：legacy 优先 —— **否决**

```
legacy → （失败）→ docling → （失败且策略允许）→ MinerU
```

**否决依据（三条独立证据）：**

1. **legacy 的 `bbox` = 0.0000（52/52 份文档，0 个元素带 bbox）。**
   本产品的核心承诺是 evidence-first——结论能回到"页 + 矩形"并画出高亮。
   legacy 连矩形都没有，引用只能落到页级，**这等于把证据链建在弱地基上**。
2. **legacy 在 8 份扫描件上产出 0 个字符**（`scanned_clear` 5 + `scanned_lowq` 3，
   `page_coverage 0.0000`、`text_density` 全部 FAIL）。这些文档在 legacy 下**根本不可用**，
   不是"质量差"，是"没有内容"。
3. **legacy 检测到 0 个公式、3 张表**（3 张全在 DOCX）。公式密集的论文在 legacy 下
   丢失全部 1 160 个公式。

### 策略 B：docling 优先（含质量 FAIL 后本地 OCR 重试）—— **部分采纳，但主链路去掉 OCR 重试**

- **代价**：docling 是重依赖，最坏 **424.4 s**（§0.2），中位 14.9 s，比 legacy 慢 **53 倍**；
  且 §4.4 证明强制 OCR 重试**没有任何收益**——docling 本来就开着 OCR。
- **适用**：数据支持。docling 在 45 份 PDF 上 **0 次质量门 FAIL**。

### 策略 C：docling 优先 + legacy 兜底（主链路不含强制 OCR）—— **✅ 选定**

在 B 的基础上去掉主链路里的强制 OCR 重试，扫描件走 docling 自身的 OCR 路径（§4.4 已证明等价）。

- **代价**：低质扫描件最弱的一份 `scanned_lowq/scan_lowq_blurry.pdf` 只出 **270 字符**，
  OCR 准确率**未测**（§3.2 缺口）。这条是本策略最大的未验证风险。
- **但兜底链必须按格式分叉**，因为 docling 在 DOCX 上有硬缺陷（见 §5.1）。

### 策略 D：质量门驱动 + 云端按需 —— **暂不采纳，保留架构**

```
主解析器 → 质量门 FAIL → 本地强制 OCR 重试 → 仍 FAIL 且隐私允许 → MinerU → 仍 FAIL → 可见失败态
```

- **代价**：逻辑最复杂；每次云端调用产生费用与出网。
- **本轮无法评估**：MinerU 列不可测（§3.3），**无法判断云端兜底值不值**。
- **结论**：链路**保留**（MinerU 适配器、审计 `mineru_audit.jsonl`、四道门都已就绪），
  但**默认不启用**，且在 MinerU 上传腿打通并实测之前不得写入默认路由。

### 5.1 决定性发现：两个解析器的失败集合**不相交**

这是本轮 bake-off 最重要的结果，也是"bake-off 只确认预期就什么也教不了"的反例：

| 失败集合 | 文档数 | 说明 |
|---|---|---|
| legacy 失败、docling 通过 | **9** | 8 份扫描件 + `hybrid_scan_text/hybrid_zh_paper.pdf` |
| docling 失败、legacy 通过 | **4** | **全部 4 份 DOCX**，`locator_coverage = 0.0000` |
| 两者都失败 | **3** | 全部 3 份 PPTX，均为 `text_density` |
| 两者都通过 | **36** | 其中 docling 27 份带 `repeated_furniture` WARN |

**没有任何一份 PDF 会被 docling 判 FAIL，也没有任何一份 DOCX 会被 legacy 判 FAIL。**
因此默认路由**不能是单一的"谁优先"，必须按格式分叉**（§6）。

### 5.2 五个类别上的尖锐分歧（按分歧幅度排序）

| # | 类别 | legacy | docling | 分歧性质 |
|---|---|---|---|---|
| 1 | **`scanned_clear` + `scanned_lowq`（8 份）** | 0 字符，0 locator，质量门 FAIL | 100% 页覆盖 + 100% bbox，质量门通过 | **能力有无**，不是程度差别。legacy 无 OCR。 |
| 2 | **`docx`（4 份）** | `locator 1.0000`，质量门通过 | **`locator 0.0000`**，4 份全部 FAIL | **docling 的硬缺陷**。DOCX 走 docling 会让全部引用无法解析。 |
| 3 | **`formula_dense`（5 份）** | **0 个公式** | **751 个公式，751 个带 LaTeX** | **能力有无**。 |
| 4 | **`table_dense`（5 份）** | **0 张表** | 63 张，62 张 2D | **能力有无**。 |
| 5 | **`hybrid_scan_text`（3 份）** | `hybrid_zh_paper` 101 字符、碎片化、FAIL | 646 字符，bbox 1.0，通过 | docling 胜；但 docling 表格 2D 仅 **0.5000**（2 张表里 1 张塌了），是它唯一的 2D 短板。 |

**另有三处 docling 的真实代价，不能只记好的：**

- **最坏耗时 424.4 s**（`formula_dense/dbrane_theory_review.pdf`），是 legacy 最坏值 25.8 s 的 **16 倍**。
- **27 份文档触发 `repeated_furniture` WARN**（legacy 只有 1 份）。docling 把页眉页脚保留成了正文，
  会把"一条跑页眉"重复计入证据。
- **3 份重 PDF 漏页**：`dbrane_theory_review` 259/261、`ligo_tests_of_gr` 52/53、
  `helm_holistic_eval` 159/162。页覆盖 0.9991 而非 1.0，缺口真实存在但很小。

---

## 6. 决策记录

**选定策略**：**策略 C（docling 优先 + legacy 兜底），但兜底链按格式分叉，强制 OCR 重试不进主链路。**

**依据**（全部引自第 4 节，同一 `corpus_version sha256:0dfe31…a1`）：

| 决策点 | 数字 |
|---|---|
| 为什么不用 legacy 兜底 PDF | legacy `bbox 0.0000`；8 份扫描件 0 字符；0 公式；`table_dense` 0 表 |
| 为什么 docling 兜底 PDF | 45 份 PDF 上 0 次质量门 FAIL；`page_coverage 0.9991`；`bbox 0.9231` |
| 为什么 DOCX 反而要 legacy 兜底 | docling 在 4/4 份 DOCX 上 `locator_coverage 0.0000`，legacy 为 1.0000 |
| 为什么不进强制 OCR 重试 | §4.4：11 份扫描件 `force_ocr` 与普通模式字符数**逐份完全相同**，docling 默认已开 OCR |
| 为什么暂不上 MinerU | §3.3：上传腿被 provider 侧拒绝，成本/收益均不可测 |

**默认解析器**：`PARSER_PRIMARY = docling`（`PARSER_DOCLING_MODE=inprocess`）

**fallback 链**（`PARSER_FALLBACK_CHAIN`，顺序即尝试顺序）：

```
1. docling (inprocess)                    # PDF / PPTX 主路径
2. docling → 失败 → legacy               # DOCX 专用：legacy 的 locator 1.0000 是 docling 的 0.0000
3. （云端 MinerU）默认关闭，四道门齐开且质量门 FAIL 才允许
```

> **关键实现约束**：fallback 不能只按"解析失败"触发，**必须按格式预置**。
> DOCX 上 docling 不抛异常、不报错，只是安静地交出 0 locator——
> **这是一个"成功但不可用"的返回，质量门能抓到（`locator_coverage` FAIL），但不会触发异常兜底。**
> 因此 DOCX 的 legacy 兜底必须由**格式判定**驱动，不能等失败信号。

**已知代价 / 本决策接受的取舍**：

1. **接受 PDF 最坏 424.4 s 的延迟。** 超过默认 `PARSER_TIMEOUT_SECONDS=120` 的 3.5 倍。
   **必须把生产默认超时抬到 900**（本轮实测即在此值下 0 超时），否则 5 份文档会被超时伪装成解析失败。
   对交互式上传这意味着最坏情况需要异步任务 + 进度反馈，不能同步等待。
2. **接受 27 份 `repeated_furniture` WARN。** 需要在 chunking 阶段用 role 过滤掉
   `header`/`footer`，否则跑页眉会污染证据。
3. **接受 3 份重 PDF 漏 2–3 页。** `page_coverage 0.9991` 不是 1.0；对要求全量引用的场景
   应在 UI 上暴露"本页未被解析"。
4. **接受扫描件 OCR 质量未验证。** 最弱的 `scan_lowq_blurry.pdf` 只有 270 字符，
   准确率无 ground truth 可算（§3.2）。**这是本决策最大的未验证面。**
5. **接受 DOCX 在 docling 上必须先分流。** 复杂度从"一条链"变成"按格式两条链"。

**关于 legacy 的粒度限制（对检索架构的直接影响）**：

legacy **每个非空 PDF 页只产出一个元素**（`legacy_loader.py:231`；本轮 21 份 PDF 实测
`elements == pages`，如 `dbrane_theory_review.pdf` 261 元素 / 261 页、
`gpt3_language_models.pdf` 75 元素 / 75 页）。
一篇 9 页论文 → 9 个元素、每个约 3 700 字符。

**这正是 hybrid 检索器值得付密检索成本的原因**：段落级切分让精确术语命中可以定位到具体段落，
而 legacy 的页级大块会让词法检索的命中粒度粗到无法引用。
已独立测得：**精确术语 R@1 = 0.86（纯词法） vs 0.43（hybrid）**，
而**释义 R@8 = 0.29（纯词法） vs 0.43（hybrid）**——
两条腿确实各有用武之地，不能因为 legacy 快就砍掉密检索。

**触发重评的条件**：

- 语料 `corpus_version` 变更（当前扩容计划到 150–200 份，届时必须全量重跑）
- docling 换大版本（当前 2.131.0；本报告的 docling 数字**只对该版本成立**）
- docling 修复 DOCX 的 `locator_coverage = 0`（届时 DOCX 分流可撤除，兜底链可简化）
- MinerU 上传腿打通并实测（届时策略 D 的成本/收益才可评估）
- OCR 准确率纳入门禁（届时 `scanned_lowq` 的风险面才能关闭）
- 双栏阅读顺序 ground truth 补齐（届时 `reading_order = 1.0000` 才能说明它真的读对了）

### 6.1 填表前必须先回答的问题 —— 逐条作答

- [x] **第 4.1 节各列的 `corpus_version` 是同一个吗？**
  legacy 与 docling 同为 `sha256:0dfe31…a1`，同一进程同一次运行，`runner_entry_digest` 亦相同。
  MinerU / docling-serve 两列**留空**，原因分别是不可运行与未运行。
- [x] **质量门 FAIL 最集中的类别是哪个？原因是什么？**
  legacy：`scanned_clear`(5) + `scanned_lowq`(3) 全部因 `page_coverage` + `text_density` FAIL——
  **没有 OCR，扫出来是 0 字符**，另有 `hybrid_zh_paper` 因 `fragmented_lines` FAIL。
  docling：**`docx`(4) 全部因 `locator_coverage` FAIL**，即"解析了但引用解析不了"；`pptx`(3) 因 `text_density` FAIL。
- [x] **legacy 的 `bbox` 比例是多少？**
  **0.0000**（52/52 份，0 个元素带 bbox）。远低于 1，因此策略 A/B 的差异不只是"质量"，而是**能否画高亮**。
- [x] **云端那一列的实际账单是多少？**
  **不可测。** 本轮未发起任何 MinerU 调用，费用为 0，但这是因为门没开，**不构成云端成本证据**。
- [x] **慢路径的最坏耗时是多少？触及 `PARSER_TIMEOUT_SECONDS` 吗？**
  docling 最坏 **424.4 s**，触及并远超默认 120s。本轮以 900s 运行，0 次超时，`failure_rate 0.0000`。
- [x] **`scanned_lowq` 那一类，所有解析器是不是都失败？**
  **不是。** legacy 3/3 失败（0 字符），docling 3/3 通过（270 / 2 322 / 968 字符，`bbox 1.0000`）。
  这是"需要换方案"的信号，不是"都需要人来看一眼"——
  但**准确率仍未验证**，最弱的 270 字符那一份需人工确认。
- [x] **本地部署（`local_only`）是否仍然完全可用？零出网是否成立？**
  **成立。** 选定策略 C 的主链路与兜底链**全部是本地进程内**（docling inprocess + legacy），
  不含任何出网调用。MinerU 分支默认关闭且需四道门，符合 §5.1 规则 2。

---

## 7. 变更记录

| 日期 | 变更 | 依据 |
|---|---|---|
| 2026-09-29 | 建立本文档模板；评测脚本 `eval/run_parser_eval.py` 就绪（59 项单测）；首轮 bake-off 未运行，决策未做出 | — |
| 2026-09-30 | **首轮 bake-off 运行（`run1-local`），但其 docling 列为坏适配器所测，全部作废** | §0.1 |
| 2026-09-30 | **适配器修复后重跑全量 52 份 × 2 解析器（`run2-inprocess`，32.7 min，0 失败、0 超时）** | §4.1 |
| 2026-09-30 | **强制 OCR 对照（`run3-force-ocr`，11 份）：`force_ocr` 对 docling 2.131.0 是 no-op** | §4.4 |
| 2026-09-30 | **PPTX `page_coverage` 判定为口径不可比，显式排除该列（修正上一轮的"指标缺口"判断）** | §4.2.1 |
| 2026-09-30 | **adversarial 错误路径重跑（`run4-adversarial`）：4/4 符合预期，两解析器均明确报错不崩溃** | §4.3 |
| 2026-09-30 | **复现性核对：第二次独立全量运行，质量类指标逐位相同，`ms` 有 ±2–6% 抖动** | §4.5 |
| 2026-09-30 | **路由决策落档：策略 C + 按格式分叉的兜底链，`PARSER_PRIMARY=docling`，DOCX 例外走 legacy** | §6 |

### 7.1 复现命令

```bash
cd /home/ubuntu/PaperWhisperer_project
PYTHON=/home/ubuntu/miniconda3/bin/python3

# 全量本地对照（第 4.1 / 4.2 节数据源）
HF_ENDPOINT=https://hf-mirror.com PARSER_TIMEOUT_SECONDS=900 \
PARSER_BACKEND=gateway PARSER_DOCLING_MODE=inprocess \
$PYTHON -m eval.run_parser_eval --parsers legacy,docling \
    --out eval/reports/run2-inprocess

# 或用 Makefile
make eval-bakeoff RUN=run2-inprocess
```