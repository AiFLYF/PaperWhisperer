# Parser bake-off report / 解析器对比报告

- Corpus version (manifest `corpus_version`, authoritative): `sha256:0dfe319761a7cc9d003dd71c00be7ca57f912bf529654b831cd61d5da8f5dda1`
- Runner entry digest (local cross-check over loaded (file, sha256) pairs, **not** a corpus identity): `414ebe9983b807eb9d64c2101cd28c0553fe70a73bb017c3473e39c9ca7ef5c0`
- Manifest: `paperwhisperer-eval-manifest/0.2`, created `2026-09-29`
- Generated: 2026-09-30T12:20:09+00:00
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
| `legacy` | 1.0 | 52/52 | 0 | 0.0000 | 0.7708 | 1.0000 | 0.8462 | 0.0000 | 1.0000 | n/a | 288 | 12 |
| `docling` | 2.131.0 | 52/52 | 0 | 0.0000 | 0.0000 | n/a | 0.0000 | 0.0000 | 1.0000 | n/a | 15516 | 52 |

## Per-category / 分类别对比

| Parser / category | Usable | Page coverage | Reading order | Locator | Tables 2D | Formulas LaTeX | Median ms |
|---|---|---|---|---|---|---|---|
| `docling / digital_complex_layout` | 2/2 | 0.0000 | n/a | 0.0000 | 1.0000 | n/a | 84102 |
| `docling / digital_single_col` | 5/5 | 0.0000 | n/a | 0.0000 | 1.0000 | n/a | 20551 |
| `docling / digital_two_col` | 10/10 | 0.0000 | n/a | 0.0000 | 1.0000 | n/a | 18199 |
| `docling / docx` | 4/4 | n/a | n/a | 0.0000 | 1.0000 | n/a | 226 |
| `docling / formula_dense` | 5/5 | 0.0000 | n/a | 0.0000 | 1.0000 | n/a | 57918 |
| `docling / hybrid_scan_text` | 3/3 | 0.0000 | n/a | 0.0000 | 1.0000 | n/a | 3772 |
| `docling / long_paper` | 3/3 | 0.0000 | n/a | 0.0000 | 1.0000 | n/a | 100486 |
| `docling / mixed_zh_en` | 4/4 | 0.0000 | n/a | 0.0000 | 1.0000 | n/a | 2991 |
| `docling / pptx` | 3/3 | 0.0000 | n/a | 0.0000 | n/a | n/a | 257 |
| `docling / scanned_clear` | 5/5 | 0.0000 | n/a | 0.0000 | 1.0000 | n/a | 3472 |
| `docling / scanned_lowq` | 3/3 | 0.0000 | n/a | 0.0000 | 1.0000 | n/a | 3049 |
| `docling / table_dense` | 5/5 | 0.0000 | n/a | 0.0000 | 1.0000 | n/a | 38800 |
| `legacy / digital_complex_layout` | 2/2 | 1.0000 | 1.0000 | 1.0000 | n/a | n/a | 6218 |
| `legacy / digital_single_col` | 5/5 | 1.0000 | 1.0000 | 1.0000 | n/a | n/a | 379 |
| `legacy / digital_two_col` | 10/10 | 1.0000 | 1.0000 | 1.0000 | n/a | n/a | 369 |
| `legacy / docx` | 4/4 | n/a | 1.0000 | 1.0000 | 1.0000 | n/a | 14 |
| `legacy / formula_dense` | 5/5 | 1.0000 | 1.0000 | 1.0000 | n/a | n/a | 1096 |
| `legacy / hybrid_scan_text` | 3/3 | 1.0000 | n/a | 1.0000 | n/a | n/a | 4 |
| `legacy / long_paper` | 3/3 | 1.0000 | 1.0000 | 1.0000 | n/a | n/a | 6232 |
| `legacy / mixed_zh_en` | 4/4 | 1.0000 | n/a | 1.0000 | n/a | n/a | 8 |
| `legacy / pptx` | 3/3 | 0.0000 | 1.0000 | 1.0000 | n/a | n/a | 9 |
| `legacy / scanned_clear` | 5/5 | 0.0000 | n/a | 0.0000 | n/a | n/a | 2 |
| `legacy / scanned_lowq` | 3/3 | 0.0000 | n/a | 0.0000 | n/a | n/a | 1 |
| `legacy / table_dense` | 5/5 | 1.0000 | 1.0000 | 1.0000 | n/a | n/a | 743 |

## Per document / 逐文档明细

### `legacy`

| Document | Category | Status | Page cov | Order | Tables 2D | Formulas LaTeX | Locator | bbox | Chars/manifest | ms | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `digital_complex_layout/ligo_gw150914_prl.pdf` | digital_complex_layout | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 3.3829 | 1244 |  |
| `digital_complex_layout/quantum_supremacy_nature_style.pdf` | digital_complex_layout | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 12.7420 | 11192 |  |
| `digital_single_col/attention_is_all_you_need.pdf` | digital_single_col | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 2.7044 | 1272 |  |
| `digital_single_col/gan_generative_adversarial.pdf` | digital_single_col | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 1.7437 | 316 |  |
| `digital_single_col/hinton_distillation.pdf` | digital_single_col | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 1.6499 | 379 |  |
| `digital_single_col/vit_an_image_is_worth_16x16.pdf` | digital_single_col | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 3.4309 | 887 |  |
| `digital_single_col/word2vec.pdf` | digital_single_col | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 2.2145 | 233 |  |
| `digital_two_col/batchnorm.pdf` | digital_two_col | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 1.8425 | 566 |  |
| `digital_two_col/faster_rcnn.pdf` | digital_two_col | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 2.7261 | 750 |  |
| `digital_two_col/fpn_feature_pyramid.pdf` | digital_two_col | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 1.9249 | 315 |  |
| `digital_two_col/googlenet_inception.pdf` | digital_two_col | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 2.0684 | 280 |  |
| `digital_two_col/mask_rcnn.pdf` | digital_two_col | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 2.1747 | 3699 |  |
| `digital_two_col/mobilenets.pdf` | digital_two_col | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 1.7519 | 268 |  |
| `digital_two_col/resnet.pdf` | digital_two_col | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 2.5192 | 423 |  |
| `digital_two_col/unet.pdf` | digital_two_col | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 1.6242 | 452 |  |
| `digital_two_col/yolov1.pdf` | digital_two_col | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 1.7828 | 295 |  |
| `digital_two_col/yolov3.pdf` | digital_two_col | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 1.2314 | 174 |  |
| `docx/bilingual_zh_en_notes.docx` | docx | success (成功) | n/a | 1.0000 | 1/1 | 0/0 | 1.0000 | 0.0000 | n/a | 16 |  |
| `docx/figures_with_captions.docx` | docx | success (成功) | n/a | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | n/a | 12 |  |
| `docx/formulas_omml.docx` | docx | success (成功) | n/a | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | n/a | 11 |  |
| `docx/technical_report_headings_tables.docx` | docx | success (成功) | n/a | 1.0000 | 2/2 | 0/0 | 1.0000 | 0.0000 | n/a | 15 |  |
| `formula_dense/dbrane_theory_review.pdf` | formula_dense | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 70.8067 | 8206 |  |
| `formula_dense/green_tao_primes.pdf` | formula_dense | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 8.8787 | 2629 |  |
| `formula_dense/maldacena_large_n.pdf` | formula_dense | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 4.4628 | 589 |  |
| `formula_dense/maldacena_toroidal_compactification.pdf` | formula_dense | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 3.8244 | 1096 |  |
| `formula_dense/witten_ads_holography.pdf` | formula_dense | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 2.8159 | 425 |  |
| `hybrid_scan_text/hybrid_en_paper.pdf` | hybrid_scan_text | partial (降级可用) | 1.0000 | n/a | 0/0 | 0/0 | 1.0000 | 0.0000 | 0.9991 | 5 |  |
| `hybrid_scan_text/hybrid_en_tables.pdf` | hybrid_scan_text | partial (降级可用) | 1.0000 | n/a | 0/0 | 0/0 | 1.0000 | 0.0000 | 1.0000 | 4 |  |
| `hybrid_scan_text/hybrid_zh_paper.pdf` | hybrid_scan_text | partial (降级可用) | 1.0000 | n/a | 0/0 | 0/0 | 1.0000 | 0.0000 | 1.0000 | 3 | fragmented_lines;text_density |
| `long_paper/gpt3_language_models.pdf` | long_paper | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 14.7652 | 1207 |  |
| `long_paper/ligo_tests_of_gr.pdf` | long_paper | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 6.8098 | 6232 |  |
| `long_paper/planck2018_cosmo_params.pdf` | long_paper | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 12.1024 | 23621 |  |
| `mixed_zh_en/bilingual_paper_zh_en.pdf` | mixed_zh_en | partial (降级可用) | 1.0000 | n/a | 0/0 | 0/0 | 1.0000 | 0.0000 | 0.9907 | 10 |  |
| `mixed_zh_en/dual_language_survey.pdf` | mixed_zh_en | partial (降级可用) | 1.0000 | n/a | 0/0 | 0/0 | 1.0000 | 0.0000 | 0.9961 | 8 |  |
| `mixed_zh_en/tech_report_zh_terms.pdf` | mixed_zh_en | partial (降级可用) | 1.0000 | n/a | 0/0 | 0/0 | 1.0000 | 0.0000 | 0.9857 | 9 | fragmented_lines |
| `mixed_zh_en/zh_standard_document.pdf` | mixed_zh_en | partial (降级可用) | 1.0000 | n/a | 0/0 | 0/0 | 1.0000 | 0.0000 | 0.9839 | 8 |  |
| `pptx/bilingual_zh_en_deck.pptx` | pptx | success (成功) | 0.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | n/a | 9 | page_coverage;text_density |
| `pptx/deck_with_notes_and_charts.pptx` | pptx | success (成功) | 0.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | n/a | 9 | page_coverage;text_density |
| `pptx/project_review_deck.pptx` | pptx | success (成功) | 0.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | n/a | 9 | page_coverage;text_density |
| `scanned_clear/paper_en_basic.pdf` | scanned_clear | partial (降级可用) | 0.0000 | n/a | 0/0 | 0/0 | 0.0000 | 0.0000 | n/a | 2 | page_coverage;text_density |
| `scanned_clear/paper_en_formulas.pdf` | scanned_clear | partial (降级可用) | 0.0000 | n/a | 0/0 | 0/0 | 0.0000 | 0.0000 | n/a | 2 | page_coverage;text_density |
| `scanned_clear/paper_en_twocol.pdf` | scanned_clear | partial (降级可用) | 0.0000 | n/a | 0/0 | 0/0 | 0.0000 | 0.0000 | n/a | 2 | page_coverage;text_density |
| `scanned_clear/paper_zh_cn.pdf` | scanned_clear | partial (降级可用) | 0.0000 | n/a | 0/0 | 0/0 | 0.0000 | 0.0000 | n/a | 3 | page_coverage;text_density |
| `scanned_clear/report_en_tables.pdf` | scanned_clear | partial (降级可用) | 0.0000 | n/a | 0/0 | 0/0 | 0.0000 | 0.0000 | n/a | 2 | page_coverage;text_density |
| `scanned_lowq/scan_lowq_blurry.pdf` | scanned_lowq | partial (降级可用) | 0.0000 | n/a | 0/0 | 0/0 | 0.0000 | 0.0000 | n/a | 1 | page_coverage;text_density |
| `scanned_lowq/scan_lowq_rotated.pdf` | scanned_lowq | partial (降级可用) | 0.0000 | n/a | 0/0 | 0/0 | 0.0000 | 0.0000 | n/a | 2 | page_coverage;text_density |
| `scanned_lowq/scan_lowq_skew_faint.pdf` | scanned_lowq | partial (降级可用) | 0.0000 | n/a | 0/0 | 0/0 | 0.0000 | 0.0000 | n/a | 1 | page_coverage;text_density |
| `table_dense/efficientnet.pdf` | table_dense | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 1.9685 | 435 |  |
| `table_dense/glue_benchmark.pdf` | table_dense | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 3.6350 | 356 |  |
| `table_dense/gpt4_technical_report.pdf` | table_dense | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 19.6150 | 1762 |  |
| `table_dense/helm_holistic_eval.pdf` | table_dense | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 30.6408 | 6575 |  |
| `table_dense/llama_report.pdf` | table_dense | partial (降级可用) | 1.0000 | 1.0000 | 0/0 | 0/0 | 1.0000 | 0.0000 | 4.6648 | 743 |  |

### `docling`

| Document | Category | Status | Page cov | Order | Tables 2D | Formulas LaTeX | Locator | bbox | Chars/manifest | ms | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `digital_complex_layout/ligo_gw150914_prl.pdf` | digital_complex_layout | success (成功) | 0.0000 | n/a | 1/1 | 0/0 | 0.0000 | 0.0000 | 3.4110 | 15928 | page_coverage;locator_coverage |
| `digital_complex_layout/quantum_supremacy_nature_style.pdf` | digital_complex_layout | success (成功) | 0.0000 | n/a | 13/13 | 0/0 | 0.0000 | 0.0000 | 12.8114 | 152275 | page_coverage;fragmented_lines;locator_coverage |
| `digital_single_col/attention_is_all_you_need.pdf` | digital_single_col | success (成功) | 0.0000 | n/a | 4/4 | 0/0 | 0.0000 | 0.0000 | 2.8204 | 20551 | page_coverage;locator_coverage |
| `digital_single_col/gan_generative_adversarial.pdf` | digital_single_col | success (成功) | 0.0000 | n/a | 2/2 | 0/0 | 0.0000 | 0.0000 | 1.6836 | 24237 | page_coverage;locator_coverage |
| `digital_single_col/hinton_distillation.pdf` | digital_single_col | success (成功) | 0.0000 | n/a | 5/5 | 0/0 | 0.0000 | 0.0000 | 1.6611 | 18160 | page_coverage;locator_coverage |
| `digital_single_col/vit_an_image_is_worth_16x16.pdf` | digital_single_col | success (成功) | 0.0000 | n/a | 9/9 | 0/0 | 0.0000 | 0.0000 | 3.5061 | 29799 | page_coverage;fragmented_lines;locator_coverage |
| `digital_single_col/word2vec.pdf` | digital_single_col | success (成功) | 0.0000 | n/a | 8/8 | 0/0 | 0.0000 | 0.0000 | 2.3060 | 15104 | page_coverage;locator_coverage |
| `digital_two_col/batchnorm.pdf` | digital_two_col | success (成功) | 0.0000 | n/a | 3/3 | 0/0 | 0.0000 | 0.0000 | 1.8258 | 25171 | page_coverage;locator_coverage |
| `digital_two_col/faster_rcnn.pdf` | digital_two_col | success (成功) | 0.0000 | n/a | 11/11 | 0/0 | 0.0000 | 0.0000 | 2.6898 | 32401 | page_coverage;locator_coverage |
| `digital_two_col/fpn_feature_pyramid.pdf` | digital_two_col | success (成功) | 0.0000 | n/a | 6/6 | 0/0 | 0.0000 | 0.0000 | 1.9799 | 19750 | page_coverage;locator_coverage |
| `digital_two_col/googlenet_inception.pdf` | digital_two_col | success (成功) | 0.0000 | n/a | 6/6 | 0/0 | 0.0000 | 0.0000 | 2.1708 | 13119 | page_coverage;locator_coverage |
| `digital_two_col/mask_rcnn.pdf` | digital_two_col | success (成功) | 0.0000 | n/a | 14/14 | 0/0 | 0.0000 | 0.0000 | 2.1818 | 31975 | page_coverage;locator_coverage |
| `digital_two_col/mobilenets.pdf` | digital_two_col | success (成功) | 0.0000 | n/a | 16/16 | 0/0 | 0.0000 | 0.0000 | 1.8763 | 16648 | page_coverage;locator_coverage |
| `digital_two_col/resnet.pdf` | digital_two_col | success (成功) | 0.0000 | n/a | 15/15 | 0/0 | 0.0000 | 0.0000 | 2.5646 | 26727 | page_coverage;locator_coverage |
| `digital_two_col/unet.pdf` | digital_two_col | success (成功) | 0.0000 | n/a | 2/2 | 0/0 | 0.0000 | 0.0000 | 1.6287 | 10718 | page_coverage;locator_coverage |
| `digital_two_col/yolov1.pdf` | digital_two_col | success (成功) | 0.0000 | n/a | 4/4 | 0/0 | 0.0000 | 0.0000 | 1.8182 | 16136 | page_coverage;locator_coverage |
| `digital_two_col/yolov3.pdf` | digital_two_col | success (成功) | 0.0000 | n/a | 3/3 | 0/0 | 0.0000 | 0.0000 | 1.2717 | 12276 | page_coverage;locator_coverage |
| `docx/bilingual_zh_en_notes.docx` | docx | success (成功) | n/a | n/a | 1/1 | 0/0 | 0.0000 | 0.0000 | n/a | 213 | locator_coverage |
| `docx/figures_with_captions.docx` | docx | success (成功) | n/a | n/a | 0/0 | 0/0 | 0.0000 | 0.0000 | n/a | 268 | locator_coverage |
| `docx/formulas_omml.docx` | docx | success (成功) | n/a | n/a | 0/0 | 0/0 | 0.0000 | 0.0000 | n/a | 202 | locator_coverage |
| `docx/technical_report_headings_tables.docx` | docx | success (成功) | n/a | n/a | 2/2 | 0/0 | 0.0000 | 0.0000 | n/a | 240 | locator_coverage |
| `formula_dense/dbrane_theory_review.pdf` | formula_dense | success (成功) | 0.0000 | n/a | 3/3 | 0/0 | 0.0000 | 0.0000 | 68.0199 | 422350 | page_coverage;locator_coverage |
| `formula_dense/green_tao_primes.pdf` | formula_dense | success (成功) | 0.0000 | n/a | 0/0 | 0/0 | 0.0000 | 0.0000 | 7.6735 | 133043 | page_coverage;locator_coverage |
| `formula_dense/maldacena_large_n.pdf` | formula_dense | success (成功) | 0.0000 | n/a | 0/0 | 0/0 | 0.0000 | 0.0000 | 4.2478 | 39074 | page_coverage;locator_coverage |
| `formula_dense/maldacena_toroidal_compactification.pdf` | formula_dense | success (成功) | 0.0000 | n/a | 0/0 | 0/0 | 0.0000 | 0.0000 | 3.6567 | 44788 | page_coverage;locator_coverage |
| `formula_dense/witten_ads_holography.pdf` | formula_dense | success (成功) | 0.0000 | n/a | 0/0 | 0/0 | 0.0000 | 0.0000 | 2.5256 | 57918 | page_coverage;locator_coverage |
| `hybrid_scan_text/hybrid_en_paper.pdf` | hybrid_scan_text | success (成功) | 0.0000 | n/a | 1/1 | 0/0 | 0.0000 | 0.0000 | 1.1852 | 4490 | page_coverage;locator_coverage |
| `hybrid_scan_text/hybrid_en_tables.pdf` | hybrid_scan_text | success (成功) | 0.0000 | n/a | 1/1 | 0/0 | 0.0000 | 0.0000 | 2.2508 | 3772 | page_coverage;locator_coverage |
| `hybrid_scan_text/hybrid_zh_paper.pdf` | hybrid_scan_text | success (成功) | 0.0000 | n/a | 0/0 | 0/0 | 0.0000 | 0.0000 | 6.3960 | 3152 | page_coverage;locator_coverage |
| `long_paper/gpt3_language_models.pdf` | long_paper | success (成功) | 0.0000 | n/a | 51/51 | 0/0 | 0.0000 | 0.0000 | 16.8008 | 166630 | page_coverage;locator_coverage |
| `long_paper/ligo_tests_of_gr.pdf` | long_paper | success (成功) | 0.0000 | n/a | 11/11 | 0/0 | 0.0000 | 0.0000 | 6.8809 | 72626 | page_coverage;fragmented_lines;locator_coverage |
| `long_paper/planck2018_cosmo_params.pdf` | long_paper | success (成功) | 0.0000 | n/a | 12/12 | 0/0 | 0.0000 | 0.0000 | 11.8740 | 100486 | page_coverage;locator_coverage |
| `mixed_zh_en/bilingual_paper_zh_en.pdf` | mixed_zh_en | success (成功) | 0.0000 | n/a | 1/1 | 0/0 | 0.0000 | 0.0000 | 1.1237 | 3121 | page_coverage;locator_coverage |
| `mixed_zh_en/dual_language_survey.pdf` | mixed_zh_en | success (成功) | 0.0000 | n/a | 0/0 | 0/0 | 0.0000 | 0.0000 | 0.9772 | 1649 | page_coverage;locator_coverage |
| `mixed_zh_en/tech_report_zh_terms.pdf` | mixed_zh_en | success (成功) | 0.0000 | n/a | 2/2 | 0/0 | 0.0000 | 0.0000 | 1.4472 | 4271 | page_coverage;locator_coverage |
| `mixed_zh_en/zh_standard_document.pdf` | mixed_zh_en | success (成功) | 0.0000 | n/a | 1/1 | 0/0 | 0.0000 | 0.0000 | 1.1489 | 2861 | page_coverage;locator_coverage |
| `pptx/bilingual_zh_en_deck.pptx` | pptx | success (成功) | 0.0000 | n/a | 0/0 | 0/0 | 0.0000 | 0.0000 | n/a | 257 | page_coverage;text_density;locator_coverage |
| `pptx/deck_with_notes_and_charts.pptx` | pptx | success (成功) | 0.0000 | n/a | 0/0 | 0/0 | 0.0000 | 0.0000 | n/a | 210 | page_coverage;text_density;locator_coverage |
| `pptx/project_review_deck.pptx` | pptx | success (成功) | 0.0000 | n/a | 0/0 | 0/0 | 0.0000 | 0.0000 | n/a | 281 | page_coverage;text_density;locator_coverage |
| `scanned_clear/paper_en_basic.pdf` | scanned_clear | success (成功) | 0.0000 | n/a | 1/1 | 0/0 | 0.0000 | 0.0000 | n/a | 3939 | page_coverage;locator_coverage |
| `scanned_clear/paper_en_formulas.pdf` | scanned_clear | success (成功) | 0.0000 | n/a | 0/0 | 0/0 | 0.0000 | 0.0000 | n/a | 2779 | page_coverage;locator_coverage |
| `scanned_clear/paper_en_twocol.pdf` | scanned_clear | success (成功) | 0.0000 | n/a | 0/0 | 0/0 | 0.0000 | 0.0000 | n/a | 3285 | page_coverage;locator_coverage |
| `scanned_clear/paper_zh_cn.pdf` | scanned_clear | success (成功) | 0.0000 | n/a | 1/1 | 0/0 | 0.0000 | 0.0000 | n/a | 3474 | page_coverage;locator_coverage |
| `scanned_clear/report_en_tables.pdf` | scanned_clear | success (成功) | 0.0000 | n/a | 2/2 | 0/0 | 0.0000 | 0.0000 | n/a | 3472 | page_coverage;locator_coverage |
| `scanned_lowq/scan_lowq_blurry.pdf` | scanned_lowq | success (成功) | 0.0000 | n/a | 1/1 | 0/0 | 0.0000 | 0.0000 | n/a | 3049 | page_coverage;locator_coverage |
| `scanned_lowq/scan_lowq_rotated.pdf` | scanned_lowq | success (成功) | 0.0000 | n/a | 1/1 | 0/0 | 0.0000 | 0.0000 | n/a | 4273 | page_coverage;locator_coverage |
| `scanned_lowq/scan_lowq_skew_faint.pdf` | scanned_lowq | success (成功) | 0.0000 | n/a | 0/0 | 0/0 | 0.0000 | 0.0000 | n/a | 2837 | page_coverage;locator_coverage |
| `table_dense/efficientnet.pdf` | table_dense | success (成功) | 0.0000 | n/a | 8/8 | 0/0 | 0.0000 | 0.0000 | 2.0931 | 20760 | page_coverage;fragmented_lines;locator_coverage |
| `table_dense/glue_benchmark.pdf` | table_dense | success (成功) | 0.0000 | n/a | 7/7 | 0/0 | 0.0000 | 0.0000 | 3.9684 | 22199 | page_coverage;locator_coverage |
| `table_dense/gpt4_technical_report.pdf` | table_dense | success (成功) | 0.0000 | n/a | 13/13 | 0/0 | 0.0000 | 0.0000 | 19.9808 | 93970 | page_coverage;locator_coverage |
| `table_dense/helm_holistic_eval.pdf` | table_dense | success (成功) | 0.0000 | n/a | 18/18 | 0/0 | 0.0000 | 0.0000 | 31.5438 | 161256 | page_coverage;locator_coverage |
| `table_dense/llama_report.pdf` | table_dense | success (成功) | 0.0000 | n/a | 17/17 | 0/0 | 0.0000 | 0.0000 | 4.8977 | 38800 | page_coverage;locator_coverage |

## How to read these numbers / 指标解读

- `page_coverage`: 出现正文的页数 / 解析器自报的页数；无分页格式记为 n/a。
- `reading_order`: 正文在主坐标（页/幻灯片/段落/字符偏移）上单调的比例；n/a 表示可比元素不足两个。
- `tables 2d/total`: 保留行列结构的表格数 / 表格总数；差额为被拍平成文本的表格。
- `formulas latex/total`: 带 LaTeX 的公式数 / 公式总数。
- `locator`: 可解析 SourceLocator 的正文元素比例（对应 P0 退出标准 100%）。
- `bbox`: 上述 locator 中带 bbox 的比例（PDF 引用高亮的前置条件）。
- `chars/manifest`: 解析出的正文字符数 / manifest 记录的文本层字符数。后者来自 pypdf 文本层，是参考值而非 ground truth。
- `ms`: 外层墙钟耗时，含适配器自身的文件读取与初始化。

