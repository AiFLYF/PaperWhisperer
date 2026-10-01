# -*- coding: utf-8 -*-
"""Scan eval/corpus, measure each file and emit eval/manifest.yaml.

Measurements per file: size, sha256, page/slide count, whether a text layer
exists, image count, first-line preview.

Batch run scope: everything under corpus/<category>/ is part of the regular
parser bake-off; the whole of corpus/adversarial/ is flagged
`batch_excluded: true` and is never swept into that batch. That is two
different reasons bundled under one flag, and conflating them makes the flag
useless for triage:

- corrupted_truncated.pdf, encrypted_aes128.pdf and
  fake_extension_actually_text.pdf are *legitimate documents that must fail*.
  They are excluded from the scoring batch only because an unhandled parse
  error would abort or crash the sweep; they are then run deliberately, one at
  a time, to assert that the parser reports a clean error instead of silently
  returning empty text. Skipping them in a batch is bookkeeping, not a verdict
  that they are invalid test material.
- zip_bomb_nested.zip is a hostile input. It is excluded because letting the
  batch decompress it is itself the failure mode being tested, and it has no
  legitimate parse result to assert.

Every file in the manifest is nevertheless versioned: all of them, excluded
entries included, feed the `corpus_version` hash, so a change to an adversarial
sample still moves the corpus version.

Reproducibility: BASE is derived from this file's own location, so a fresh
clone works from any directory on any platform.
"""
from __future__ import annotations

import hashlib
import os
from datetime import date

import yaml

# Anchor every path to this script's own directory. The corpus is addressed
# relative to the repo checkout, not to whatever machine or working directory
# happened to build it.
BASE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(BASE, "corpus")

# category -> (quota from DEVELOPMENT_PLAN section 6, focus text)
CATEGORIES = {
    "digital_single_col": (5, "基线质量：单栏数字原生 PDF 的解析完整性"),
    "digital_two_col": (10, "阅读顺序恢复：双栏论文的左右列顺序"),
    "digital_complex_layout": (2, "复杂版式：跨栏图表、浮动对象、脚注"),
    "formula_dense": (5, "公式保留：LaTeX/公式是否被破坏或丢失"),
    "table_dense": (5, "表格二维结构：表头、行列、跨页表格"),
    "scanned_clear": (5, "OCR 触发：清晰扫描件的字符准确率与段落恢复"),
    "scanned_lowq": (3, "OCR 极限：倾斜/模糊/低分辨率下的降级行为"),
    "hybrid_scan_text": (3, "扫描图 + 文本层：是否能识别并使用已有文本层"),
    "mixed_zh_en": (4, "多语言检索：中英混排的词法与语义召回"),
    "long_paper": (3, "全文覆盖：长论文与超长参考文献的上下文完整性"),
    "docx": (4, "DOCX：标题层级、表格、图片、公式元素的定位"),
    "pptx": (3, "PPTX：幻灯片、notes、图表与图片文字"),
}

# arxiv id per downloaded filename
ARXIV_SOURCE = {
    "attention_is_all_you_need.pdf": "arxiv:1706.03762",
    "vit_an_image_is_worth_16x16.pdf": "arxiv:2010.11929",
    "word2vec.pdf": "arxiv:1301.3781",
    "gan_generative_adversarial.pdf": "arxiv:1406.2661",
    "hinton_distillation.pdf": "arxiv:1503.02531",
    "resnet.pdf": "arxiv:1512.03385",
    "unet.pdf": "arxiv:1505.04597",
    "mask_rcnn.pdf": "arxiv:1703.06870",
    "yolov1.pdf": "arxiv:1506.02640",
    "faster_rcnn.pdf": "arxiv:1506.01497",
    "yolov3.pdf": "arxiv:1804.02767",
    "fpn_feature_pyramid.pdf": "arxiv:1612.03144",
    "mobilenets.pdf": "arxiv:1704.04861",
    "googlenet_inception.pdf": "arxiv:1409.4842",
    "batchnorm.pdf": "arxiv:1502.03167",
    "ligo_gw150914_prl.pdf": "arxiv:1602.03837",
    "quantum_supremacy_nature_style.pdf": "arxiv:1910.11333",
    "maldacena_toroidal_compactification.pdf": "arxiv:hep-th/9711201",
    "witten_ads_holography.pdf": "arxiv:hep-th/9802109",
    "green_tao_primes.pdf": "arxiv:math/0404188",
    "dbrane_theory_review.pdf": "arxiv:hep-th/9905111",
    "maldacena_large_n.pdf": "arxiv:hep-th/9711200",
    "glue_benchmark.pdf": "arxiv:1804.07461",
    "helm_holistic_eval.pdf": "arxiv:2211.09110",
    "llama_report.pdf": "arxiv:2302.13971",
    "efficientnet.pdf": "arxiv:1905.11946",
    "gpt4_technical_report.pdf": "arxiv:2303.08774",
    "gpt3_language_models.pdf": "arxiv:2005.14165",
    "planck2018_cosmo_params.pdf": "arxiv:1807.06209",
    "ligo_tests_of_gr.pdf": "arxiv:2010.14533",
}

SYNTHETIC_NOTE = {
    "paper_en_basic.pdf": "合成：300 DPI 英文论文扫描件，含标题/章节/结论",
    "paper_zh_cn.pdf": "合成：300 DPI 中文论文扫描件，含表格与公式",
    "paper_en_formulas.pdf": "合成：公式密集扫描件，含积分与 beta 函数",
    "report_en_tables.pdf": "合成：表格密集扫描件，两张多行多列表格",
    "paper_en_twocol.pdf": "合成：双栏扫描件，考察 OCR 阅读顺序",
    "scan_lowq_rotated.pdf": "合成：旋转 2.8 度 + 降采样 + JPEG q30",
    "scan_lowq_blurry.pdf": "合成：中文扫描件，模糊 + 噪声 + JPEG q22",
    "scan_lowq_skew_faint.pdf": "合成：倾斜 -3.5 度 + 淡墨 + 噪声",
    "hybrid_en_paper.pdf": "合成：扫描图 + 不可见文本层（英文）",
    "hybrid_zh_paper.pdf": "合成：中文扫描图 + 英文不可见文本层",
    "hybrid_en_tables.pdf": "合成：表格扫描图 + 不可见文本层",
    "bilingual_paper_zh_en.pdf": "合成：中文论文，含英文摘要与英文术语，含页眉页脚",
    "tech_report_zh_terms.pdf": "合成：中文技术报告，含 SLO 表格与英文指标名",
    "zh_standard_document.pdf": "合成：标准规范文体，条款编号 3.1/3.2 + 中英术语表",
    "dual_language_survey.pdf": "合成：英文正文 + 中文本节要点交替",
    "technical_report_headings_tables.docx": "合成：H1-H3 标题层级 + 两张网格表 + 项目符号列表",
    "figures_with_captions.docx": "合成：2 张图片 + 图题（画布居中）",
    "formulas_omml.docx": "合成：5 段 OMML 公式（上下标、分数、希腊字母）",
    "bilingual_zh_en_notes.docx": "合成：中英混排 DOCX，含中文表格与英文摘要",
    "project_review_deck.pptx": "合成：8 页标题+正文幻灯片",
    "deck_with_notes_and_charts.pptx": "合成：3 页 notes + 2 个图表 + 1 张图片",
    "bilingual_zh_en_deck.pptx": "合成：6 页中英混排幻灯片",
}

ADVERSARIAL_NOTE = {
    "corrupted_truncated.pdf": "有效 PDF 截断至 35% 字节，应报明确解析错误而不是静默返回空文本",
    "encrypted_aes128.pdf": "AES-128 加密 PDF，口令 paperwhisperer-eval，不得使解析进程崩溃",
    "fake_extension_actually_text.pdf": "纯文本文件伪装成 .pdf，仅校验扩展名的实现会误判通过",
    "zip_bomb_nested.zip": "压缩包膨胀样本（6 KB 展开为约 843 KB），检验解压大小限制",
}


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def corpus_version(categories, adversarial):
    """Content-addressed identity of the corpus: ``sha256:<64 hex>``.

    This is what "corpus version" means here, as opposed to `schema` (the
    *format* version of this file) and `created` (the wall-clock date the file
    was generated -- a date changes on every regeneration even when nothing
    about the corpus changed, so it identifies a generation, not a snapshot).

    The digest is taken over the fields that decide what the corpus *is*:
    the category/quota set, and for every file (batch and adversarial alike)
    its path plus its content sha256. Deliberately excluded are `created`, all
    human-readable `note`/`focus` prose, and the measured fields (`bytes`,
    `pages`, `preview`, `chars_total`, `omml_paragraphs`, ...). Those are
    derived or editorial: including them would make the version change on a
    reworded comment or a difference in how many preview characters a
    pypdf release decides to extract, which is noise for the purpose the field
    exists to serve.

    Consequences that matter for reproducible reports:
    - Re-running build_manifest.py against an untouched corpus reproduces the
      same `corpus_version` byte for byte.
    - Changing one byte of any single document -- including an adversarial
      sample -- changes it.
    - Adding or removing a file changes it.
    - Changing a category's quota changes it, so a report cannot silently run
      against a different quota table than the one it was written for.

    The per-entry `sha256` values remain the authoritative integrity anchor;
    this is a single scalar summarising them, and a report should cite both.

    The input order is normalised (categories sorted, files sorted within each
    category) so the digest does not depend on os.listdir or dict ordering.
    """
    lines = []
    for cat in sorted(categories):
        spec = categories[cat]
        lines.append(f"category\t{cat}\t{spec['quota']}")
        for entry in sorted(spec["entries"], key=lambda e: e["file"]):
            lines.append(f"entry\t{entry['file']}\t{entry['sha256']}")
    for entry in sorted(adversarial, key=lambda e: e["file"]):
        lines.append(f"adversarial\t{entry['file']}\t{entry['sha256']}")
    payload = "\n".join(lines) + "\n"
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def measure_pdf(path):
    from pypdf import PdfReader
    info = {"pages": None, "text_layer": None, "images_p1": None, "preview": ""}
    try:
        r = PdfReader(path)
        info["pages"] = len(r.pages)
        p0 = r.pages[0]
        txt = (p0.extract_text() or "").strip()
        info["text_layer"] = bool(txt)
        info["images_p1"] = len(p0.images)
        info["preview"] = " ".join(txt.split())[:90]
        info["chars_total"] = sum(len((pg.extract_text() or "")) for pg in r.pages[:5])
    except Exception as exc:  # corrupted / encrypted samples land here
        info["error"] = f"{type(exc).__name__}: {exc}"[:120]
    return info


def measure_docx(path):
    from docx import Document
    d = Document(path)
    return {
        "paragraphs": len(d.paragraphs),
        "tables": len(d.tables),
        "images": len(d.inline_shapes),
        "omml_paragraphs": sum(1 for p in d.paragraphs if "oMath" in p._p.xml),
    }


def measure_pptx(path):
    from pptx import Presentation
    prs = Presentation(path)
    notes = charts = pics = 0
    for s in prs.slides:
        if s.has_notes_slide and s.notes_slide.notes_text_frame.text.strip():
            notes += 1
        for sh in s.shapes:
            if sh.has_chart:
                charts += 1
            if sh.shape_type == 13:
                pics += 1
    return {"slides": len(prs.slides), "notes_slides": notes, "charts": charts, "pictures": pics}


def main():
    manifest = {
        "schema": "paperwhisperer-eval-manifest/0.2",
        "created": date.today().isoformat(),
        "corpus_root": "corpus",
        "notes": [
            "adversarial/ 下的文件不参与常规批跑，仅用于错误处理与安全测试",
            "synthetic 条目的原文文本保存在 ground_truth/ 下，可用于 OCR 准确率打分",
            "arXiv 条目的来源 ID 记录在 source 字段，便于复现下载",
            "corpus_version 是由内容算出的语料版本（sha256:<hex>），对未改动的语料重跑不变；"
            "created 只是生成日期，不构成版本；每条目的 sha256 仍是完整性基准",
        ],
        "categories": {},
        "adversarial": [],
        "summary": {},
    }

    total = 0
    quota_ok = True
    for cat, (quota, focus) in CATEGORIES.items():
        d = os.path.join(CORPUS, cat)
        entries = []
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            path = os.path.join(d, name)
            if not os.path.isfile(path):
                continue
            entry = {
                "file": f"{cat}/{name}",
                "source": ARXIV_SOURCE.get(name, "synthetic"),
                "bytes": os.path.getsize(path),
                "sha256": sha256_file(path),
            }
            if cat == "docx":
                entry.update(measure_docx(path))
            elif cat == "pptx":
                entry.update(measure_pptx(path))
            else:
                entry.update(measure_pdf(path))
            if name in SYNTHETIC_NOTE:
                entry["note"] = SYNTHETIC_NOTE[name]
            if name in ARXIV_SOURCE:
                entry["note"] = "arXiv 原始 PDF，版面与元素类型真实"
            entries.append(entry)
            total += 1
        manifest["categories"][cat] = {
            "quota": quota,
            "actual": len(entries),
            "focus": focus,
            "entries": entries,
        }
        flag = "OK" if len(entries) >= quota else "SHORT"
        if len(entries) < quota:
            quota_ok = False
        print(f"{cat:24s} quota={quota:2d} actual={len(entries):2d} {flag}")

    adv_dir = os.path.join(CORPUS, "adversarial")
    for name in sorted(os.listdir(adv_dir)):
        path = os.path.join(adv_dir, name)
        if not os.path.isfile(path):
            continue
        manifest["adversarial"].append({
            "file": f"adversarial/{name}",
            "bytes": os.path.getsize(path),
            "sha256": sha256_file(path),
            "batch_excluded": True,
            "note": ADVERSARIAL_NOTE.get(name, ""),
        })

    manifest["summary"] = {
        "total_documents": total,
        "adversarial_files": len(manifest["adversarial"]),
        "quota_met": quota_ok,
        "ground_truth_dir": "ground_truth",
    }
    # Emitted at top level (not nested in summary) so a consumer can read
    # `manifest["corpus_version"]` without knowing anything about the summary
    # block, and so its position is fixed regardless of the summary's shape.
    manifest["corpus_version"] = corpus_version(manifest["categories"], manifest["adversarial"])
    print(f"\ntotal={total} adversarial={len(manifest['adversarial'])} quota_met={quota_ok}")
    print(f"corpus_version={manifest['corpus_version']}")

    out = os.path.join(BASE, "manifest.yaml")
    with open(out, "w", encoding="utf-8") as fh:
        yaml.safe_dump(manifest, fh, allow_unicode=True, sort_keys=False, width=200)
    print(f"written: {out}")


if __name__ == "__main__":
    main()
