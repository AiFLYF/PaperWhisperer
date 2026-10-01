# Parser bake-off report / 解析器对比报告

- Corpus version (manifest `corpus_version`, authoritative): `sha256:0dfe319761a7cc9d003dd71c00be7ca57f912bf529654b831cd61d5da8f5dda1`
- Runner entry digest (local cross-check over loaded (file, sha256) pairs, **not** a corpus identity): `414ebe9983b807eb9d64c2101cd28c0553fe70a73bb017c3473e39c9ca7ef5c0`
- Manifest: `paperwhisperer-eval-manifest/0.2`, created `2026-09-29`
- Generated: 2026-09-30T18:52:03+00:00
- Runner version: 1.0
- Environment: 3.13.13 on Linux-6.8.0-142-generic-x86_64-with-glibc2.39

## Run options / 运行参数

| Option | Value |
|---|---|
| `allow_cloud` | False |
| `categories` | (all) |
| `documents` | 52 |
| `force_ocr` | False |
| `include_excluded` | False |
| `limit` | (all) |
| `manifest` | /home/ubuntu/PaperWhisperer_project/eval/manifest.yaml |
| `parsers` | legacy,docling |

## Parsers that could not run / 无法运行的解析器

All requested parsers ran. 本次请求的解析器全部可运行。

## Parser summary / 解析器汇总

| Parser | Version | Usable / attempted | Failed | Failure rate | Page coverage | Reading order | Locator | bbox | Tables 2D | Formulas LaTeX | Median ms | Quality FAIL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `legacy` | 1.0 | 52/52 | 0 | 0.0000 | 0.7708 | 1.0000 | 0.8462 | 0.0000 | 1.0000 | n/a | 283 | 12 |
| `docling` | 2.131.0 | 52/52 | 0 | 0.0000 | 0.9991 | 1.0000 | 0.9231 | 0.9231 | 0.9639 | 1.0000 | 14914 | 7 |

## Per-category / 分类别对比

| Parser / category | Usable | Page coverage | Reading order | Locator | Tables 2D | Formulas LaTeX | Median ms |
|---|---|---|---|---|---|---|---|
| `docling / digital_complex_layout` | 2/2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 85210 |
| `docling / digital_single_col` | 5/5 | 1.0000 | 1.0000 | 1.0000 | 0.9643 | 1.0000 | 19526 |
| `docling / digital_two_col` | 10/10 | 1.0000 | 1.0000 | 1.0000 | 0.9750 | 1.0000 | 19436 |
| `docling / docx` | 4/4 | n/a | n/a | 0.0000 | 1.0000 | 1.0000 | 240 |
| `docling / formula_dense` | 5/5 | 0.9985 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 57832 |
| `docling / hybrid_scan_text` | 3/3 | 1.0000 | 1.0000 | 1.0000 | 0.5000 | 1.0000 | 3681 |
| `docling / long_paper` | 3/3 | 0.9937 | 1.0000 | 1.0000 | 0.9324 | 1.0000 | 100118 |
| `docling / mixed_zh_en` | 4/4 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | n/a | 2994 |
| `docling / pptx` | 3/3 | 1.0000 | 1.0000 | 1.0000 | n/a | n/a | 259 |
| `docling / scanned_clear` | 5/5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 3253 |
| `docling / scanned_lowq` | 3/3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 3106 |
| `docling / table_dense` | 5/5 | 0.9963 | 1.0000 | 1.0000 | 0.9841 | 1.0000 | 30578 |
| `legacy / digital_complex_layout` | 2/2 | 1.0000 | 1.0000 | 1.0000 | n/a | n/a | 6434 |
| `legacy / digital_single_col` | 5/5 | 1.0000 | 1.0000 | 1.0000 | n/a | n/a | 377 |
| `legacy / digital_two_col` | 10/10 | 1.0000 | 1.0000 | 1.0000 | n/a | n/a | 312 |
| `legacy / docx` | 4/4 | n/a | 1.0000 | 1.0000 | 1.0000 | n/a | 14 |
| `legacy / formula_dense` | 5/5 | 1.0000 | 1.0000 | 1.0000 | n/a | n/a | 675 |
| `legacy / hybrid_scan_text` | 3/3 | 1.0000 | n/a | 1.0000 | n/a | n/a | 4 |
| `legacy / long_paper` | 3/3 | 1.0000 | 1.0000 | 1.0000 | n/a | n/a | 6235 |
| `legacy / mixed_zh_en` | 4/4 | 1.0000 | n/a | 1.0000 | n/a | n/a | 8 |
| `legacy / pptx` | 3/3 | 0.0000 | 1.0000 | 1.0000 | n/a | n/a | 10 |
| `legacy / scanned_clear` | 5/5 | 0.0000 | n/a | 0.0000 | n/a | n/a | 2 |
| `legacy / scanned_lowq` | 3/3 | 0.0000 | n/a | 0.0000 | n/a | n/a | 1 |
| `legacy / table_dense` | 5/5 | 1.0000 | 1.0000 | 1.0000 | n/a | n/a | 1540 |

## Per document / 逐文档明细

### `legacy`

| Document | Category | Status | Page cov | Order | Tables 2D | Formulas LaTeX | Locator | bbox | Chars/manifest | ms | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `digital_complex_layout/ligo_gw150914_prl.pdf` | digital_complex_layout | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 3.3829 | 1225 |  |
| `digital_complex_layout/quantum_supremacy_nature_style.pdf` | digital_complex_layout | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 12.7420 | 11642 |  |
| `digital_single_col/attention_is_all_you_need.pdf` | digital_single_col | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 2.7044 | 1277 |  |
| `digital_single_col/gan_generative_adversarial.pdf` | digital_single_col | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 1.7437 | 314 |  |
| `digital_single_col/hinton_distillation.pdf` | digital_single_col | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 1.6499 | 377 |  |
| `digital_single_col/vit_an_image_is_worth_16x16.pdf` | digital_single_col | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 3.4309 | 923 |  |
| `digital_single_col/word2vec.pdf` | digital_single_col | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 2.2145 | 237 |  |
| `digital_two_col/batchnorm.pdf` | digital_two_col | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 1.8425 | 546 |  |
| `digital_two_col/faster_rcnn.pdf` | digital_two_col | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 2.7261 | 784 |  |
| `digital_two_col/fpn_feature_pyramid.pdf` | digital_two_col | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 1.9249 | 326 |  |
| `digital_two_col/googlenet_inception.pdf` | digital_two_col | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 2.0684 | 286 |  |
| `digital_two_col/mask_rcnn.pdf` | digital_two_col | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 2.1747 | 3218 |  |
| `digital_two_col/mobilenets.pdf` | digital_two_col | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 1.7519 | 280 |  |
| `digital_two_col/resnet.pdf` | digital_two_col | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 2.5192 | 448 |  |
| `digital_two_col/unet.pdf` | digital_two_col | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 1.6242 | 148 |  |
| `digital_two_col/yolov1.pdf` | digital_two_col | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 1.7828 | 299 |  |
| `digital_two_col/yolov3.pdf` | digital_two_col | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 1.2314 | 178 |  |
| `docx/bilingual_zh_en_notes.docx` | docx | success (成功) | n/a | 1.0000 | 1/1 | 0/0 | 1.0000 | 0.0000 | n/a | 15 |  |
| `docx/figures_with_captions.docx` | docx | success (成功) | n/a | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | n/a | 12 |  |
| `docx/formulas_omml.docx` | docx | success (成功) | n/a | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | n/a | 11 |  |
| `docx/technical_report_headings_tables.docx` | docx | success (成功) | n/a | 1.0000 | 2/2 | 0/0 | 1.0000 | 0.0000 | n/a | 15 |  |
| `formula_dense/dbrane_theory_review.pdf` | formula_dense | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 70.8067 | 8652 |  |
| `formula_dense/green_tao_primes.pdf` | formula_dense | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 8.8787 | 2049 |  |
| `formula_dense/maldacena_large_n.pdf` | formula_dense | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 4.4628 | 608 |  |
| `formula_dense/maldacena_toroidal_compactification.pdf` | formula_dense | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 3.8244 | 675 |  |
| `formula_dense/witten_ads_holography.pdf` | formula_dense | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 2.8159 | 407 |  |
| `hybrid_scan_text/hybrid_en_paper.pdf` | hybrid_scan_text | partial (降级可用) | 1.0000 | n/a | 0/0 | 0/0 | 1.0000 | 0.0000 | 0.9991 | 5 |  |
| `hybrid_scan_text/hybrid_en_tables.pdf` | hybrid_scan_text | partial (降级可用) | 1.0000 | n/a | 0/0 | 0/0 | 1.0000 | 0.0000 | 1.0000 | 4 |  |
| `hybrid_scan_text/hybrid_zh_paper.pdf` | hybrid_scan_text | partial (降级可用) | 1.0000 | n/a | 0/0 | 0/0 | 1.0000 | 0.0000 | 1.0000 | 3 | fragmented_lines;text_density |
| `long_paper/gpt3_language_models.pdf` | long_paper | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 14.7652 | 1198 |  |
| `long_paper/ligo_tests_of_gr.pdf` | long_paper | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 6.8098 | 6235 |  |
| `long_paper/planck2018_cosmo_params.pdf` | long_paper | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 12.1024 | 25838 |  |
| `mixed_zh_en/bilingual_paper_zh_en.pdf` | mixed_zh_en | partial (降级可用) | 1.0000 | n/a | 0/0 | 0/0 | 1.0000 | 0.0000 | 0.9907 | 9 |  |
| `mixed_zh_en/dual_language_survey.pdf` | mixed_zh_en | partial (降级可用) | 1.0000 | n/a | 0/0 | 0/0 | 1.0000 | 0.0000 | 0.9961 | 8 |  |
| `mixed_zh_en/tech_report_zh_terms.pdf` | mixed_zh_en | partial (降级可用) | 1.0000 | n/a | 0/0 | 0/0 | 1.0000 | 0.0000 | 0.9857 | 9 | fragmented_lines |
| `mixed_zh_en/zh_standard_document.pdf` | mixed_zh_en | partial (降级可用) | 1.0000 | n/a | 0/0 | 0/0 | 1.0000 | 0.0000 | 0.9839 | 8 |  |
| `pptx/bilingual_zh_en_deck.pptx` | pptx | success (成功) | 0.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | n/a | 10 | page_coverage;text_density |
| `pptx/deck_with_notes_and_charts.pptx` | pptx | success (成功) | 0.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | n/a | 10 | page_coverage;text_density |
| `pptx/project_review_deck.pptx` | pptx | success (成功) | 0.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | n/a | 9 | page_coverage;text_density |
| `scanned_clear/paper_en_basic.pdf` | scanned_clear | partial (降级可用) | 0.0000 | n/a | 0/0 | 0/0 | 0.0000 | 0.0000 | n/a | 2 | page_coverage;text_density |
| `scanned_clear/paper_en_formulas.pdf` | scanned_clear | partial (降级可用) | 0.0000 | n/a | 0/0 | 0/0 | 0.0000 | 0.0000 | n/a | 2 | page_coverage;text_density |
| `scanned_clear/paper_en_twocol.pdf` | scanned_clear | partial (降级可用) | 0.0000 | n/a | 0/0 | 0/0 | 0.0000 | 0.0000 | n/a | 2 | page_coverage;text_density |
| `scanned_clear/paper_zh_cn.pdf` | scanned_clear | partial (降级可用) | 0.0000 | n/a | 0/0 | 0/0 | 0.0000 | 0.0000 | n/a | 2 | page_coverage;text_density |
| `scanned_clear/report_en_tables.pdf` | scanned_clear | partial (降级可用) | 0.0000 | n/a | 0/0 | 0/0 | 0.0000 | 0.0000 | n/a | 2 | page_coverage;text_density |
| `scanned_lowq/scan_lowq_blurry.pdf` | scanned_lowq | partial (降级可用) | 0.0000 | n/a | 0/0 | 0/0 | 0.0000 | 0.0000 | n/a | 1 | page_coverage;text_density |
| `scanned_lowq/scan_lowq_rotated.pdf` | scanned_lowq | partial (降级可用) | 0.0000 | n/a | 0/0 | 0/0 | 0.0000 | 0.0000 | n/a | 2 | page_coverage;text_density |
| `scanned_lowq/scan_lowq_skew_faint.pdf` | scanned_lowq | partial (降级可用) | 0.0000 | n/a | 0/0 | 0/0 | 0.0000 | 0.0000 | n/a | 1 | page_coverage;text_density |
| `table_dense/efficientnet.pdf` | table_dense | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 1.9685 | 419 |  |
| `table_dense/glue_benchmark.pdf` | table_dense | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 3.6350 | 879 |  |
| `table_dense/gpt4_technical_report.pdf` | table_dense | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 19.6150 | 1894 |  |
| `table_dense/helm_holistic_eval.pdf` | table_dense | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 30.6408 | 6828 |  |
| `table_dense/llama_report.pdf` | table_dense | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 4.6648 | 1540 |  |

### `docling`

| Document | Category | Status | Page cov | Order | Tables 2D | Formulas LaTeX | Locator | bbox | Chars/manifest | ms | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `digital_complex_layout/ligo_gw150914_prl.pdf` | digital_complex_layout | success (成功) | 1.0000 | 1.0000 | 1/1 | 1/1 | 1.0000 | 1.0000 | 3.3705 | 16185 | repeated_furniture |
| `digital_complex_layout/quantum_supremacy_nature_style.pdf` | digital_complex_layout | success (成功) | 1.0000 | 1.0000 | 13/13 | 134/134 | 1.0000 | 1.0000 | 13.0834 | 154234 | fragmented_lines;repeated_furniture |
| `digital_single_col/attention_is_all_you_need.pdf` | digital_single_col | success (成功) | 1.0000 | 1.0000 | 4/4 | 6/6 | 1.0000 | 1.0000 | 2.7396 | 19526 | repeated_furniture |
| `digital_single_col/gan_generative_adversarial.pdf` | digital_single_col | success (成功) | 1.0000 | 1.0000 | 2/2 | 10/10 | 1.0000 | 1.0000 | 1.7228 | 24448 | repeated_furniture |
| `digital_single_col/hinton_distillation.pdf` | digital_single_col | success (成功) | 1.0000 | 1.0000 | 4/5 | 6/6 | 1.0000 | 1.0000 | 1.6274 | 18640 | repeated_furniture |
| `digital_single_col/vit_an_image_is_worth_16x16.pdf` | digital_single_col | success (成功) | 1.0000 | 1.0000 | 9/9 | 8/8 | 1.0000 | 1.0000 | 3.3537 | 33402 | fragmented_lines;repeated_furniture |
| `digital_single_col/word2vec.pdf` | digital_single_col | success (成功) | 1.0000 | 1.0000 | 8/8 | 5/5 | 1.0000 | 1.0000 | 2.2022 | 14142 | repeated_furniture |
| `digital_two_col/batchnorm.pdf` | digital_two_col | success (成功) | 1.0000 | 1.0000 | 3/3 | 19/19 | 1.0000 | 1.0000 | 1.8610 | 25084 | repeated_furniture |
| `digital_two_col/faster_rcnn.pdf` | digital_two_col | success (成功) | 1.0000 | 1.0000 | 11/11 | 2/2 | 1.0000 | 1.0000 | 2.6247 | 33356 | repeated_furniture |
| `digital_two_col/fpn_feature_pyramid.pdf` | digital_two_col | success (成功) | 1.0000 | 1.0000 | 6/6 | 1/1 | 1.0000 | 1.0000 | 1.9205 | 19917 | repeated_furniture |
| `digital_two_col/googlenet_inception.pdf` | digital_two_col | success (成功) | 1.0000 | 1.0000 | 6/6 | 0/0 | 1.0000 | 1.0000 | 2.1033 | 13606 | repeated_furniture |
| `digital_two_col/mask_rcnn.pdf` | digital_two_col | success (成功) | 1.0000 | 1.0000 | 14/14 | 0/0 | 1.0000 | 1.0000 | 2.1044 | 31089 | repeated_furniture |
| `digital_two_col/mobilenets.pdf` | digital_two_col | success (成功) | 1.0000 | 1.0000 | 14/16 | 8/8 | 1.0000 | 1.0000 | 1.7702 | 18955 |  |
| `digital_two_col/resnet.pdf` | digital_two_col | success (成功) | 1.0000 | 1.0000 | 15/15 | 2/2 | 1.0000 | 1.0000 | 2.4572 | 29464 | repeated_furniture |
| `digital_two_col/unet.pdf` | digital_two_col | success (成功) | 1.0000 | 1.0000 | 2/2 | 2/2 | 1.0000 | 1.0000 | 1.6120 | 11137 | repeated_furniture |
| `digital_two_col/yolov1.pdf` | digital_two_col | success (成功) | 1.0000 | 1.0000 | 4/4 | 3/3 | 1.0000 | 1.0000 | 1.7975 | 15687 |  |
| `digital_two_col/yolov3.pdf` | digital_two_col | success (成功) | 1.0000 | 1.0000 | 3/3 | 1/1 | 1.0000 | 1.0000 | 1.2225 | 11322 | repeated_furniture |
| `docx/bilingual_zh_en_notes.docx` | docx | success (成功) | n/a | n/a | 1/1 | 0/0 | 0.0000 | 0.0000 | n/a | 221 | locator_coverage |
| `docx/figures_with_captions.docx` | docx | success (成功) | n/a | n/a | 0/0 | 0/0 | 0.0000 | 0.0000 | n/a | 281 | locator_coverage |
| `docx/formulas_omml.docx` | docx | success (成功) | n/a | n/a | 0/0 | 5/5 | 0.0000 | 0.0000 | n/a | 214 | locator_coverage |
| `docx/technical_report_headings_tables.docx` | docx | success (成功) | n/a | n/a | 2/2 | 0/0 | 0.0000 | 0.0000 | n/a | 258 | locator_coverage |
| `formula_dense/dbrane_theory_review.pdf` | formula_dense | success (成功) | 0.9923 | 1.0000 | 3/3 | 369/369 | 1.0000 | 1.0000 | 71.4136 | 424425 | repeated_furniture |
| `formula_dense/green_tao_primes.pdf` | formula_dense | success (成功) | 1.0000 | 1.0000 | 0/0 | 266/266 | 1.0000 | 1.0000 | 9.0321 | 138677 | repeated_furniture |
| `formula_dense/maldacena_large_n.pdf` | formula_dense | success (成功) | 1.0000 | 1.0000 | 0/0 | 25/25 | 1.0000 | 1.0000 | 4.4666 | 40939 | repeated_furniture |
| `formula_dense/maldacena_toroidal_compactification.pdf` | formula_dense | success (成功) | 1.0000 | 1.0000 | 0/0 | 36/36 | 1.0000 | 1.0000 | 3.8335 | 46985 | repeated_furniture |
| `formula_dense/witten_ads_holography.pdf` | formula_dense | success (成功) | 1.0000 | 1.0000 | 0/0 | 55/55 | 1.0000 | 1.0000 | 2.9467 | 57832 | repeated_furniture |
| `hybrid_scan_text/hybrid_en_paper.pdf` | hybrid_scan_text | success (成功) | 1.0000 | 1.0000 | 1/1 | 1/1 | 1.0000 | 1.0000 | 1.0957 | 4227 |  |
| `hybrid_scan_text/hybrid_en_tables.pdf` | hybrid_scan_text | success (成功) | 1.0000 | 1.0000 | 0/1 | 0/0 | 1.0000 | 1.0000 | 1.9264 | 3681 |  |
| `hybrid_scan_text/hybrid_zh_paper.pdf` | hybrid_scan_text | success (成功) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 1.0000 | 6.3960 | 3303 |  |
| `long_paper/gpt3_language_models.pdf` | long_paper | success (成功) | 1.0000 | 1.0000 | 47/51 | 1/1 | 1.0000 | 1.0000 | 15.5739 | 168976 | repeated_furniture |
| `long_paper/ligo_tests_of_gr.pdf` | long_paper | success (成功) | 0.9811 | 1.0000 | 11/11 | 26/26 | 1.0000 | 1.0000 | 6.8680 | 65274 | fragmented_lines;repeated_furniture |
| `long_paper/planck2018_cosmo_params.pdf` | long_paper | success (成功) | 1.0000 | 1.0000 | 11/12 | 120/120 | 1.0000 | 1.0000 | 11.7868 | 100118 | repeated_furniture |
| `mixed_zh_en/bilingual_paper_zh_en.pdf` | mixed_zh_en | success (成功) | 1.0000 | 1.0000 | 1/1 | 0/0 | 1.0000 | 1.0000 | 0.9907 | 3174 |  |
| `mixed_zh_en/dual_language_survey.pdf` | mixed_zh_en | success (成功) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 1.0000 | 0.9526 | 1648 |  |
| `mixed_zh_en/tech_report_zh_terms.pdf` | mixed_zh_en | success (成功) | 1.0000 | 1.0000 | 2/2 | 0/0 | 1.0000 | 1.0000 | 1.0991 | 4128 |  |
| `mixed_zh_en/zh_standard_document.pdf` | mixed_zh_en | success (成功) | 1.0000 | 1.0000 | 1/1 | 0/0 | 1.0000 | 1.0000 | 0.9650 | 2814 |  |
| `pptx/bilingual_zh_en_deck.pptx` | pptx | success (成功) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 1.0000 | n/a | 259 | text_density |
| `pptx/deck_with_notes_and_charts.pptx` | pptx | success (成功) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 1.0000 | n/a | 207 | text_density |
| `pptx/project_review_deck.pptx` | pptx | success (成功) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 1.0000 | n/a | 287 | text_density |
| `scanned_clear/paper_en_basic.pdf` | scanned_clear | success (成功) | 1.0000 | 1.0000 | 1/1 | 1/1 | 1.0000 | 1.0000 | n/a | 4308 |  |
| `scanned_clear/paper_en_formulas.pdf` | scanned_clear | success (成功) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 1.0000 | n/a | 2739 |  |
| `scanned_clear/paper_en_twocol.pdf` | scanned_clear | success (成功) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 1.0000 | n/a | 3106 |  |
| `scanned_clear/paper_zh_cn.pdf` | scanned_clear | success (成功) | 1.0000 | 1.0000 | 1/1 | 0/0 | 1.0000 | 1.0000 | n/a | 3253 |  |
| `scanned_clear/report_en_tables.pdf` | scanned_clear | success (成功) | 1.0000 | 1.0000 | 2/2 | 0/0 | 1.0000 | 1.0000 | n/a | 4570 |  |
| `scanned_lowq/scan_lowq_blurry.pdf` | scanned_lowq | success (成功) | 1.0000 | 1.0000 | 1/1 | 0/0 | 1.0000 | 1.0000 | n/a | 3106 |  |
| `scanned_lowq/scan_lowq_rotated.pdf` | scanned_lowq | success (成功) | 1.0000 | 1.0000 | 1/1 | 1/1 | 1.0000 | 1.0000 | n/a | 3982 |  |
| `scanned_lowq/scan_lowq_skew_faint.pdf` | scanned_lowq | success (成功) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 1.0000 | n/a | 2711 |  |
| `table_dense/efficientnet.pdf` | table_dense | success (成功) | 1.0000 | 1.0000 | 8/8 | 4/4 | 1.0000 | 1.0000 | 1.9463 | 22661 | fragmented_lines;repeated_furniture |
| `table_dense/glue_benchmark.pdf` | table_dense | success (成功) | 1.0000 | 1.0000 | 7/7 | 0/0 | 1.0000 | 1.0000 | 3.8043 | 19934 | repeated_furniture |
| `table_dense/gpt4_technical_report.pdf` | table_dense | success (成功) | 1.0000 | 1.0000 | 12/13 | 6/6 | 1.0000 | 1.0000 | 19.5720 | 88303 | repeated_furniture |
| `table_dense/helm_holistic_eval.pdf` | table_dense | success (成功) | 0.9815 | 1.0000 | 18/18 | 34/34 | 1.0000 | 1.0000 | 30.3268 | 151216 | repeated_furniture |
| `table_dense/llama_report.pdf` | table_dense | success (成功) | 1.0000 | 1.0000 | 17/17 | 2/2 | 1.0000 | 1.0000 | 4.6478 | 30578 |  |

## How to read these numbers / 指标解读

- `page_coverage`: 出现正文的页数 / 解析器自报的页数；无分页格式记为 n/a。
- `reading_order`: 正文在主坐标（页/幻灯片/段落/字符偏移）上单调的比例；n/a 表示可比元素不足两个。
- `tables 2d/total`: 保留行列结构的表格数 / 表格总数；差额为被拍平成文本的表格。
- `formulas latex/total`: 带 LaTeX 的公式数 / 公式总数。
- `locator`: 可解析 SourceLocator 的正文元素比例（对应 P0 退出标准 100%）。
- `bbox`: 上述 locator 中带 bbox 的比例（PDF 引用高亮的前置条件）。
- `chars/manifest`: 解析出的正文字符数 / manifest 记录的文本层字符数。后者来自 pypdf 文本层，是参考值而非 ground truth。
- `ms`: 外层墙钟耗时，含适配器自身的文件读取与初始化。

