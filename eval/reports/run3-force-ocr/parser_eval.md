# Parser bake-off report / 解析器对比报告

- Corpus version (manifest `corpus_version`, authoritative): `sha256:0dfe319761a7cc9d003dd71c00be7ca57f912bf529654b831cd61d5da8f5dda1`
- Runner entry digest (local cross-check over loaded (file, sha256) pairs, **not** a corpus identity): `414ebe9983b807eb9d64c2101cd28c0553fe70a73bb017c3473e39c9ca7ef5c0`
- Manifest: `paperwhisperer-eval-manifest/0.2`, created `2026-09-29`
- Generated: 2026-09-30T19:00:13+00:00
- Runner version: 1.0
- Environment: 3.13.13 on Linux-6.8.0-142-generic-x86_64-with-glibc2.39

## Run options / 运行参数

| Option | Value |
|---|---|
| `allow_cloud` | False |
| `categories` | scanned_clear,scanned_lowq,hybrid_scan_text |
| `documents` | 11 |
| `force_ocr` | True |
| `include_excluded` | False |
| `limit` | (all) |
| `manifest` | /home/ubuntu/PaperWhisperer_project/eval/manifest.yaml |
| `parsers` | legacy,docling |

## Parsers that could not run / 无法运行的解析器

All requested parsers ran. 本次请求的解析器全部可运行。

## Parser summary / 解析器汇总

| Parser | Version | Usable / attempted | Failed | Failure rate | Page coverage | Reading order | Locator | bbox | Tables 2D | Formulas LaTeX | Median ms | Quality FAIL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `legacy` | 1.0 | 11/11 | 0 | 0.0000 | 0.2727 | n/a | 0.2727 | 0.0000 | n/a | n/a | 2 | 9 |
| `docling` | 2.131.0 | 11/11 | 0 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.8750 | 1.0000 | 3254 | 0 |

## Per-category / 分类别对比

| Parser / category | Usable | Page coverage | Reading order | Locator | Tables 2D | Formulas LaTeX | Median ms |
|---|---|---|---|---|---|---|---|
| `docling / hybrid_scan_text` | 3/3 | 1.0000 | 1.0000 | 1.0000 | 0.5000 | 1.0000 | 3909 |
| `docling / scanned_clear` | 5/5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 3254 |
| `docling / scanned_lowq` | 3/3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 3170 |
| `legacy / hybrid_scan_text` | 3/3 | 1.0000 | n/a | 1.0000 | n/a | n/a | 5 |
| `legacy / scanned_clear` | 5/5 | 0.0000 | n/a | 0.0000 | n/a | n/a | 2 |
| `legacy / scanned_lowq` | 3/3 | 0.0000 | n/a | 0.0000 | n/a | n/a | 1 |

## Per document / 逐文档明细

### `legacy`

| Document | Category | Status | Page cov | Order | Tables 2D | Formulas LaTeX | Locator | bbox | Chars/manifest | ms | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `hybrid_scan_text/hybrid_en_paper.pdf` | hybrid_scan_text | partial (降级可用) | 1.0000 | n/a | 0/0 | 0/0 | 1.0000 | 0.0000 | 0.9991 | 6 |  |
| `hybrid_scan_text/hybrid_en_tables.pdf` | hybrid_scan_text | partial (降级可用) | 1.0000 | n/a | 0/0 | 0/0 | 1.0000 | 0.0000 | 1.0000 | 5 |  |
| `hybrid_scan_text/hybrid_zh_paper.pdf` | hybrid_scan_text | partial (降级可用) | 1.0000 | n/a | 0/0 | 0/0 | 1.0000 | 0.0000 | 1.0000 | 3 | fragmented_lines;text_density |
| `scanned_clear/paper_en_basic.pdf` | scanned_clear | partial (降级可用) | 0.0000 | n/a | 0/0 | 0/0 | 0.0000 | 0.0000 | n/a | 2 | page_coverage;text_density |
| `scanned_clear/paper_en_formulas.pdf` | scanned_clear | partial (降级可用) | 0.0000 | n/a | 0/0 | 0/0 | 0.0000 | 0.0000 | n/a | 2 | page_coverage;text_density |
| `scanned_clear/paper_en_twocol.pdf` | scanned_clear | partial (降级可用) | 0.0000 | n/a | 0/0 | 0/0 | 0.0000 | 0.0000 | n/a | 3 | page_coverage;text_density |
| `scanned_clear/paper_zh_cn.pdf` | scanned_clear | partial (降级可用) | 0.0000 | n/a | 0/0 | 0/0 | 0.0000 | 0.0000 | n/a | 3 | page_coverage;text_density |
| `scanned_clear/report_en_tables.pdf` | scanned_clear | partial (降级可用) | 0.0000 | n/a | 0/0 | 0/0 | 0.0000 | 0.0000 | n/a | 2 | page_coverage;text_density |
| `scanned_lowq/scan_lowq_blurry.pdf` | scanned_lowq | partial (降级可用) | 0.0000 | n/a | 0/0 | 0/0 | 0.0000 | 0.0000 | n/a | 1 | page_coverage;text_density |
| `scanned_lowq/scan_lowq_rotated.pdf` | scanned_lowq | partial (降级可用) | 0.0000 | n/a | 0/0 | 0/0 | 0.0000 | 0.0000 | n/a | 2 | page_coverage;text_density |
| `scanned_lowq/scan_lowq_skew_faint.pdf` | scanned_lowq | partial (降级可用) | 0.0000 | n/a | 0/0 | 0/0 | 0.0000 | 0.0000 | n/a | 1 | page_coverage;text_density |

### `docling`

| Document | Category | Status | Page cov | Order | Tables 2D | Formulas LaTeX | Locator | bbox | Chars/manifest | ms | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `hybrid_scan_text/hybrid_en_paper.pdf` | hybrid_scan_text | success (成功) | 1.0000 | 1.0000 | 1/1 | 1/1 | 1.0000 | 1.0000 | 1.0957 | 10811 |  |
| `hybrid_scan_text/hybrid_en_tables.pdf` | hybrid_scan_text | success (成功) | 1.0000 | 1.0000 | 0/1 | 0/0 | 1.0000 | 1.0000 | 1.9264 | 3909 |  |
| `hybrid_scan_text/hybrid_zh_paper.pdf` | hybrid_scan_text | success (成功) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 1.0000 | 6.3960 | 3118 |  |
| `scanned_clear/paper_en_basic.pdf` | scanned_clear | success (成功) | 1.0000 | 1.0000 | 1/1 | 1/1 | 1.0000 | 1.0000 | n/a | 4305 |  |
| `scanned_clear/paper_en_formulas.pdf` | scanned_clear | success (成功) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 1.0000 | n/a | 2837 |  |
| `scanned_clear/paper_en_twocol.pdf` | scanned_clear | success (成功) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 1.0000 | n/a | 3191 |  |
| `scanned_clear/paper_zh_cn.pdf` | scanned_clear | success (成功) | 1.0000 | 1.0000 | 1/1 | 0/0 | 1.0000 | 1.0000 | n/a | 3254 |  |
| `scanned_clear/report_en_tables.pdf` | scanned_clear | success (成功) | 1.0000 | 1.0000 | 2/2 | 0/0 | 1.0000 | 1.0000 | n/a | 3605 |  |
| `scanned_lowq/scan_lowq_blurry.pdf` | scanned_lowq | success (成功) | 1.0000 | 1.0000 | 1/1 | 0/0 | 1.0000 | 1.0000 | n/a | 3170 |  |
| `scanned_lowq/scan_lowq_rotated.pdf` | scanned_lowq | success (成功) | 1.0000 | 1.0000 | 1/1 | 1/1 | 1.0000 | 1.0000 | n/a | 3825 |  |
| `scanned_lowq/scan_lowq_skew_faint.pdf` | scanned_lowq | success (成功) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 1.0000 | n/a | 2839 |  |

## How to read these numbers / 指标解读

- `page_coverage`: 出现正文的页数 / 解析器自报的页数；无分页格式记为 n/a。
- `reading_order`: 正文在主坐标（页/幻灯片/段落/字符偏移）上单调的比例；n/a 表示可比元素不足两个。
- `tables 2d/total`: 保留行列结构的表格数 / 表格总数；差额为被拍平成文本的表格。
- `formulas latex/total`: 带 LaTeX 的公式数 / 公式总数。
- `locator`: 可解析 SourceLocator 的正文元素比例（对应 P0 退出标准 100%）。
- `bbox`: 上述 locator 中带 bbox 的比例（PDF 引用高亮的前置条件）。
- `chars/manifest`: 解析出的正文字符数 / manifest 记录的文本层字符数。后者来自 pypdf 文本层，是参考值而非 ground truth。
- `ms`: 外层墙钟耗时，含适配器自身的文件读取与初始化。

