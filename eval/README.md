# PaperWhisperer 评测语料（T0.1）

文档解析 bake-off 与 RAG/引用评测用的语料集合。分类标准与配额来自
`docs/DEVELOPMENT_PLAN.md` 第 6 节。

## 目录

| 路径 | 内容 |
|---|---|
| `corpus/` | 52 份文档，按 12 个类别分目录 |
| `corpus/adversarial/` | 4 份异常文件，不参与常规批跑 |
| `manifest.yaml` | 每份文档的类别、来源、sha256、页数/文本层等实测指标 |
| `ground_truth/` | 合成语料的原文文本，用于 OCR 准确率打分 |
| `download_arxiv.sh` | arXiv 语料下载脚本（`--verify` 可离线校验语料完整性） |
| `make_scanned.py` | 扫描件构造（清晰/低质/混合） |
| `make_office.py` | DOCX / PPTX 构造 |
| `make_mixed_and_adversarial.py` | 中英混排 PDF 与异常文件构造 |
| `build_manifest.py` | 扫描语料并生成 manifest.yaml |
| `reports/` | bake-off 产出（`run_parser_eval.py` 写的 parser_eval.{md,csv,json}） |
| `run_parser_eval.py` | 解析器批跑对比 runner（T0.9） |

## 配额达成情况

| 类别 | 配额 | 实际 | 关注点 |
|---|---|---|---|
| digital_single_col | 5 | 5 | 基线质量 |
| digital_two_col | 10 | 10 | 阅读顺序 |
| digital_complex_layout | 2 | 2 | 跨栏图表、脚注 |
| formula_dense | 5 | 5 | 公式保留 |
| table_dense | 5 | 5 | 表格二维结构 |
| scanned_clear | 5 | 5 | OCR 准确率 |
| scanned_lowq | 3 | 3 | OCR 降级行为 |
| hybrid_scan_text | 3 | 3 | 文本层识别 |
| mixed_zh_en | 4 | 4 | 多语言检索 |
| long_paper | 3 | 3 | 全文覆盖 |
| docx | 4 | 4 | 标题/表格/图片/公式定位 |
| pptx | 3 | 3 | 幻灯片/notes/图表 |
| **合计** | **52** | **52** | |

## reports/ 与 bake-off 重跑（2026-09-30）

`reports/` 存放 `run_parser_eval.py` 的产出。**`run1-local/` 已整体作废，不得引用**——
它测的是一个坏掉的 docling 适配器（0 个带 bbox 的元素、0 条 SourceLocator、公式全丢）。
作废原因与逐项对比见 `docs/PARSER_BAKE_OFF.md` 第 0 节。

当前有效的运行：

| 目录 | 内容 | 规模 | 用途 |
|---|---|---|---|
| `run1-local/` | **作废**，见上 | 52 份 | 保留仅为对照，勿引用其 docling 数字 |
| `run2-inprocess/` | legacy × docling 全量对跑 | 52 份 | 决策依据（第 4.1 / 4.2 节） |
| `run3-force-ocr/` | 强制 OCR，扫描件子集 | 11 份 | 证明 `force_ocr` 对 docling 是 no-op（第 4.4 节） |
| `run4-adversarial/` | `--include-excluded` | 52 + 4 | 错误路径（第 7.2 节） |

### 复现

```bash
PYTHON=/home/ubuntu/miniconda3/bin/python3
make eval-bakeoff RUN=run2-inprocess PYTHON=$PYTHON          # ~33 min
make eval-bakeoff-force-ocr RUN=run3 PYTHON=$PYTHON          # ~1 min
make eval-bakeoff-adversarial RUN=run4 PYTHON=$PYTHON        # ~33 min
```

### 三条容易踩的坑

1. **`PARSER_TIMEOUT_SECONDS` 必须抬到 900。** 默认 120s 下 docling 有 5/52 份文档超时，
   而**超时会被记成解析失败**——报告里会出现并非解析器责任的"失败"。
   实测真实最坏耗时为 **424.4 s**（`formula_dense/dbrane_theory_review.pdf`）。
2. **`--list-parsers` 不带 `--parsers` 时只列 legacy。** 要看 docling 状态必须写
   `--list-parsers --parsers legacy,docling`，否则会误判 docling 不可用。
3. **`--include-excluded` 是"把 adversarial 加进批跑"，不是"只跑 adversarial"。**
   它会连带重跑全部 52 份常规文档，因此耗时与全量跑相当（约 33 分钟），不是秒级。
   加超时保护时别按"4 个文件"来估。

### 结果摘要

同一 `corpus_version` 下 legacy 与 docling **失败集合不相交**：
legacy 失败而 docling 通过 9 份（8 份扫描件 + 1 份混合扫描），
docling 失败而 legacy 通过 4 份（全部 DOCX，`locator_coverage = 0`）。
因此默认路由按格式分叉，而非"谁优先"——详见 `docs/PARSER_BAKE_OFF.md` 第 6 节。

## 真实语料与合成语料

- arXiv 下载 30 份（真实版面、真实公式与表格），来源 ID 记录在 manifest 的 `source` 字段
- 合成 22 份：扫描件、中英混排 PDF、DOCX、PPTX、异常文件

合成语料存在的理由不是偷懒，而是可控：扫描件的原文文本是已知的（`ground_truth/`），
可以直接算字符准确率，而真实扫描件没有 ground truth 可对齐。

## 复现

所有脚本都从自身位置推导路径（`__file__` / `BASH_SOURCE`），可直接在任意目录、任意平台运行。

```bash
# 需要外网。默认直连；如需代理：
#   ARXIV_PROXY=http://127.0.0.1:10808 bash download_arxiv.sh
bash download_arxiv.sh

python make_scanned.py
python make_office.py
python make_mixed_and_adversarial.py
python build_manifest.py
```

依赖：`pillow pypdf python-docx python-pptx reportlab pyyaml cryptography`

扫描件需要真实的拉丁与中文字形。可用 `PW_EN_FONT` / `PW_EN_BOLD_FONT` / `PW_ZH_FONT`
指定字体文件；未指定时按候选列表探测（Windows 字体优先，其次 Linux/macOS 常见字体），
找不到时会打印警告而不是静默生成不可辨识的页面。

## 语料版本与可复现性

`manifest.yaml` 顶层有四个容易混淆的标识字段：

| 字段 | 含义 | 会变吗 |
|---|---|---|
| `schema` | manifest **格式**版本，如 `paperwhisperer-eval-manifest/0.2` | 仅在格式变更时变 |
| `created` | 生成日期 | **每次重跑都变**，不能当版本 |
| `corpus_version` | **语料版本**，由内容算出，形如 `sha256:<64位十六进制>` | 仅在语料内容/配额变动时变 |
| `*.sha256` | 每份文件的完整性基准 | 随该文件内容变 |

- `created` 只是"这份 manifest 是哪天生成的"。即使语料一个字节都没动，重跑也会得到新日期，
  所以它标识的是一次生成，而不是一个快照。**不要用 `created` 标注报告中的语料版本。**
- `corpus_version` 才是报告应当引用的语料版本。算法（见 `build_manifest.py:corpus_version`）：
  对「类别名 + 配额」以及「每份文件的相对路径 + 内容 sha256」（批跑语料与 adversarial 一并计入）
  组成规范化文本行后取 sha256。刻意排除 `created`、各条 `note`/`focus` 说明文字与实测字段
  （`bytes`/`pages`/`preview` 等），因为它们是派生量或编辑性内容，纳入会让改一句注释就改变版本。
- 由此保证：**未改动的语料重跑 `build_manifest.py`，`corpus_version` 逐字节不变**；
  改动任意一份文档（包括 adversarial 样本）一个字节、改动配额、增删文件，都会改变它。
- 每条目的 `sha256` 仍是**权威的完整性基准**；`corpus_version` 只是对它们的汇总标量。
  报告应同时引用两者。
- 报告复现方式：在报告中写明 `corpus_version`，并用
  `bash download_arxiv.sh --verify` 确认本地语料与 manifest 一致（离线，不联网）。

`download_arxiv.sh --verify` 会交叉校验：下载清单与 `manifest.yaml` 的 `source` ID 是否一一对应、
磁盘文件是否与记录的 sha256 一致。清单与实际语料不一致时它会直接失败——
这正是"重跑复现步骤不应改动语料"这条约束的守卫。

## 注意事项

- `corpus/` 体积约 79 MB，不进 git（见 .gitignore）；`eval/` 下的脚本、README、
  `manifest.yaml` 与 `ground_truth/` **都进 git**，因为复现与校验语料需要它们。
  `eval/_assets/` 是 DOCX/PPTX 图表的渲染中间产物，同样不进 git。
- `adversarial/zip_bomb_nested.zip` 展开后约 843 KB，用于验证解压限制，不要把它直接放进生产上传接口之外的环境
- 扩展路线：公开发布前把 corpus 扩到 150-200 份。届时 `corpus_version` 会自动改变，
  报告中应引用新值，无需手动维护版本号。
