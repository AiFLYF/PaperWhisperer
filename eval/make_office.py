# -*- coding: utf-8 -*-
"""Construct DOCX / PPTX eval corpus for PaperWhisperer (T0.1).

docx/  4 files: heading hierarchy + tables, figures with captions,
        OMML formulas, bilingual notes
pptx/  3 files: plain deck, notes + chart + image, bilingual deck

Paths are resolved from this script's own location, so it runs from a fresh
clone in any directory on any platform.
"""
from __future__ import annotations

import io
import os

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import parse_xml
from pptx import Presentation
from pptx.util import Inches as PInches, Pt as PPt
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE
from PIL import Image, ImageDraw, ImageFont

# Anchor every path to this script's own directory. The corpus is addressed
# relative to the repo checkout, not to whatever machine or working directory
# happened to build it.
BASE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(BASE, "corpus")
GT_DIR = os.path.join(BASE, "ground_truth")
TMP = os.path.join(BASE, "_assets")
os.makedirs(TMP, exist_ok=True)
os.makedirs(GT_DIR, exist_ok=True)

# Chart/figure labels are decorative, not ground truth, so a best-effort font
# is fine here -- each call site already falls back to PIL's default. The
# Windows faces stay first so a Windows rebuild matches the committed corpus.
ARIAL = [
    r"C:\Windows\Fonts\arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]
ARIAL = next((p for p in ARIAL if os.path.isfile(p)), ARIAL[0])


# ---------------------------------------------------------------- charts
def make_chart_png(path, title="Accuracy vs. epoch"):
    w, h = 900, 540
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    try:
        f = ImageFont.truetype(ARIAL, 20)
    except Exception:
        f = ImageFont.load_default()
    d.text((30, 20), title, font=f, fill=(0, 0, 0))
    # axes
    x0, y0, x1, y1 = 80, 440, 850, 80
    d.line([(x0, y0), (x1, y0)], fill=(0, 0, 0), width=2)
    d.line([(x0, y0), (x0, y1)], fill=(0, 0, 0), width=2)
    series = {
        "baseline (lr=0.1)": [(0, 55), (1, 66), (2, 73), (3, 78), (4, 80), (5, 81)],
        "rescaled (lr=0.05)": [(0, 55), (1, 70), (2, 79), (3, 84), (4, 87), (5, 88)],
    }
    colors = {"baseline (lr=0.1)": (30, 90, 200), "rescaled (lr=0.05)": (200, 60, 40)}
    for name, pts in series.items():
        px = [(x0 + i * (x1 - x0) / 5, y0 - (v - 50) * (y0 - y1) / 45) for i, v in pts]
        d.line(px, fill=colors[name], width=4)
        for p in px:
            bbox = [p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4]
            d.ellipse(bbox, fill=colors[name])
        d.text((x1 - 250, y1 + 20 + 24 * list(series).index(name)), name, font=f, fill=colors[name])
    d.text((x0, y0 + 12), "epoch", font=f, fill=(0, 0, 0))
    d.text((x0 + 10, y1 - 30), "accuracy (%)", font=f, fill=(0, 0, 0))
    img.save(path, "PNG")
    return path


def make_table_figure_png(path):
    """A figure that *looks* like a result table (tests figure/table separation)."""
    w, h = 800, 420
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    try:
        f = ImageFont.truetype(ARIAL, 18)
    except Exception:
        f = ImageFont.load_default()
    rows = [
        ["model", "params (M)", "top-1", "FLOPs (G)"],
        ["ResNet-50", "25.6", "76.3", "4.1"],
        ["ResNet-101", "44.5", "77.4", "7.9"],
        ["ResNet-152", "60.2", "78.0", "11.6"],
    ]
    cw = 190
    rh = 70
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            x, y = 30 + c * cw, 30 + r * rh
            d.rectangle([x, y, x + cw, y + rh], outline=(60, 60, 60), width=2)
            d.text((x + 10, y + 22), cell, font=f, fill=(0, 0, 0))
    img.save(path, "PNG")
    return path


# ---------------------------------------------------------------- OMML
OMML_NS = 'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'


def omml(inner: str) -> bytes:
    return parse_xml(f'<m:oMath {OMML_NS}>{inner}</m:oMath>')


def omml_run(text: str) -> str:
    return f'<m:r><w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/></w:rPr><m:t>{text}</m:t></m:r>'


def omml_sup(base: str, sup: str) -> str:
    return (f'<m:sSup><m:e>{omml_run(base)}</m:e>'
            f'<m:sup>{omml_run(sup)}</m:sup></m:sSup>')


def omml_frac(num: str, den: str) -> str:
    return (f'<m:f><m:num>{omml_run(num)}</m:num>'
            f'<m:den>{omml_run(den)}</m:den></m:f>')


def add_omml_paragraph(doc, inner):
    p = doc.add_paragraph()
    p._p.append(omml(inner))
    return p


def center(p):
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER


# ---------------------------------------------------------------- DOCX 1
def build_technical_report():
    doc = Document()
    doc.add_heading("Evaluation Protocol for Document Understanding Systems", level=1)
    doc.add_paragraph(
        "This report defines a reproducible evaluation protocol for document "
        "understanding systems, covering input formats, quality gates, and "
        "scoring procedures. It is intended to be executed before every parser "
        "or model upgrade."
    )
    doc.add_heading("1. Scope", level=1)
    doc.add_heading("1.1 Supported Inputs", level=2)
    for item in ["Digital PDF (single and multi column)", "Scanned PDF and images",
                 "Microsoft Word documents (DOCX)", "PowerPoint decks (PPTX)",
                 "Plain text and Markdown"]:
        doc.add_paragraph(item, style="List Bullet")
    doc.add_heading("1.2 Out of Scope", level=2)
    doc.add_paragraph(
        "Spreadsheet semantics and interactive form fields are out of scope for "
        "this revision and are tracked separately."
    )
    doc.add_heading("2. Quality Gates", level=1)
    doc.add_heading("2.1 Hard Failures", level=2)
    doc.add_paragraph(
        "A document fails the gate when page counts disagree with the parsed "
        "output, when extraction raises an unrecoverable error, or when more than "
        "a configured fraction of pages are empty."
    )
    tbl = doc.add_table(rows=1, cols=4)
    tbl.style = "Table Grid"
    hdr = tbl.rows[0].cells
    for i, h in enumerate(["signal", "severity", "threshold", "action"]):
        hdr[i].text = h
    data = [
        ["page count mismatch", "FAIL", "any", "fallback"],
        ["empty page ratio", "FAIL", "> 0.25", "OCR retry"],
        ["garbled char ratio", "WARN", "> 0.02", "OCR retry"],
        ["missing locator", "FAIL", "> 0.10", "reparse"],
    ]
    for row in data:
        cells = tbl.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = v
    doc.add_heading("2.2 Warning Signals", level=2)
    doc.add_paragraph(
        "Warnings do not block ingestion but must be surfaced to the user and "
        "recorded as flags on the document record."
    )
    doc.add_heading("3. Reporting", level=1)
    doc.add_paragraph(
        "Each run produces a per-document report and an aggregate summary across "
        "the corpus. Reports are versioned by corpus revision."
    )
    t2 = doc.add_table(rows=1, cols=3)
    t2.style = "Table Grid"
    for i, h in enumerate(["metric", "target", "current"]):
        t2.rows[0].cells[i].text = h
    for row in [["page coverage", "1.00", "0.97"],
                ["reading order accuracy", "0.95", "0.92"],
                ["table preservation", "0.90", "0.88"]]:
        cells = t2.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = v
    path = os.path.join(CORPUS, "docx", "technical_report_headings_tables.docx")
    doc.save(path)
    return path


# ---------------------------------------------------------------- DOCX 2
def build_figures_doc():
    chart = make_chart_png(os.path.join(TMP, "chart_acc.png"))
    tfig = make_table_figure_png(os.path.join(TMP, "fig_table.png"))
    doc = Document()
    doc.add_heading("Visual Results and Ablations", level=1)
    doc.add_paragraph(
        "Figure 1 shows top-1 accuracy against training epoch for the baseline and "
        "the rescaled variant. Figure 2 summarises parameter counts and compute."
    )
    doc.add_picture(chart, width=Inches(6.0))
    p = doc.add_paragraph("Figure 1: Top-1 accuracy against epoch on the validation split.")
    center(p)
    doc.add_heading("1. Ablation", level=2)
    doc.add_paragraph(
        "Removing the residual rescale reduces final accuracy by 0.6 points. "
        "Removing normalisation is worse and destabilises training."
    )
    doc.add_picture(tfig, width=Inches(5.5))
    p = doc.add_paragraph("Figure 2: Parameter count, accuracy and compute cost by depth.")
    center(p)
    doc.add_heading("2. Discussion", level=2)
    doc.add_paragraph(
        "The benefit of rescaling concentrates in the late phase of training, "
        "consistent with the variance preservation analysis."
    )
    path = os.path.join(CORPUS, "docx", "figures_with_captions.docx")
    doc.save(path)
    return path


# ---------------------------------------------------------------- DOCX 3
def build_formula_doc():
    doc = Document()
    doc.add_heading("Variance Preservation in Residual Networks", level=1)
    doc.add_paragraph(
        "We derive the variance preservation condition for a rescaled residual "
        "block. Let the residual update be written as follows."
    )
    add_omml_paragraph(doc, omml_run("y") + omml_run("=") + omml_run("x") + omml_run("+") +
                       omml_run("F") + omml_run("(") + omml_run("x") + omml_run(",") +
                       omml_run("W") + omml_run(")"))
    doc.add_paragraph(
        "With scalar multipliers on the identity path and the residual branch the "
        "update becomes a convex combination controlled by two parameters."
    )
    add_omml_paragraph(doc, omml_run("x") +
                       omml_sup("l", "+1") + omml_run(" = ") +
                       omml_run("\u03bb") + omml_run("x") +
                       omml_sup("l", "") + omml_run(" + ") +
                       omml_run("\u03bc") + omml_run("F") + omml_run("(") +
                       omml_run("x") + omml_sup("l", "") + omml_run(")"))
    doc.add_paragraph("The variance preservation condition requires that the following hold for every layer.")
    add_omml_paragraph(doc, omml_run("\u03bb") + omml_sup("2", "") + omml_run(" + ") +
                       omml_run("\u03bc") + omml_sup("2", "") + omml_run("Var") +
                       omml_run("[") + omml_run("F") + omml_run("]") + omml_run(" = ") + omml_run("1"))
    doc.add_paragraph("For the scalar quadratic root used in the analysis we have the closed form.")
    add_omml_paragraph(doc, omml_run("x") + omml_run(" = ") +
                       omml_frac("\u2212b \u00b1 \u221a(b\u00b2 \u2212 4ac)", "2a"))
    doc.add_paragraph("The mass energy relation of the running example is also included for parser coverage.")
    add_omml_paragraph(doc, omml_run("E") + omml_run(" = ") + omml_sup("mc", "2"))
    path = os.path.join(CORPUS, "docx", "formulas_omml.docx")
    doc.save(path)
    return path


# ---------------------------------------------------------------- DOCX 4
def build_bilingual_doc():
    doc = Document()
    doc.add_heading("多模态检索系统在企业知识库中的应用评估", level=1)
    p = doc.add_paragraph()
    r = p.add_run("Abstract. ")
    r.bold = True
    p.add_run(
        "We evaluate a multimodal retrieval system (dense embedding + BM25 hybrid) "
        "on an enterprise knowledge base of 128,000 documents. Recall@10 improves "
        "from 0.71 to 0.86 after adding a cross-encoder reranker, while p95 "
        "latency increases by 42 ms."
    )
    doc.add_heading("1. 引言", level=1)
    doc.add_paragraph(
        "企业知识库通常同时包含 PDF 报告、DOCX 制度文档与 PPTX 汇报材料，"
        "单一模态的检索方案难以覆盖全部内容。本文以 hybrid retrieval 为基线，"
        "评估 reranker 与 query expansion 的增量收益。"
    )
    doc.add_heading("2. 实验设置", level=1)
    doc.add_paragraph(
        "语料规模为 128,000 篇文档，其中中文占比 61%，英文占比 34%，"
        "中英混排文档占比 5%。Embedding 维度为 1024，chunk size 为 512 tokens，"
        "overlap 为 64 tokens。"
    )
    tbl = doc.add_table(rows=1, cols=4)
    tbl.style = "Table Grid"
    for i, h in enumerate(["配置", "Recall@10", "MRR", "p95 延迟 (ms)"]):
        tbl.rows[0].cells[i].text = h
    for row in [["dense only", "0.74", "0.52", "118"],
                ["dense + BM25", "0.79", "0.57", "134"],
                ["dense + BM25 + rerank", "0.86", "0.63", "176"],
                ["+ query expansion", "0.87", "0.64", "198"]]:
        cells = tbl.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = v
    doc.add_heading("3. 结论", level=1)
    doc.add_paragraph(
        "在中文语料上，BM25 通道的贡献明显高于英文语料，说明中文分词与"
        "term-level 匹配仍是必要补充。Reranker 的延迟代价在离线场景中可接受。"
    )
    path = os.path.join(CORPUS, "docx", "bilingual_zh_en_notes.docx")
    doc.save(path)
    return path


# ---------------------------------------------------------------- PPTX 1
def build_plain_deck():
    prs = Presentation()
    slides = [
        ("Document Understanding: Parser Gateway Design",
         ["Motivation: one parser is not enough",
          "Goal: canonical document model with provenance",
          "Constraint: privacy mode forbids silent cloud calls"]),
        ("Current State",
         ["Three extractors flatten everything to text",
          "No page, no bbox, no element type",
          "Downstream tasks read the first N characters"]),
        ("Target Architecture",
         ["Parser Gateway with pluggable adapters",
          "Canonical ParsedDocument + SourceLocator",
          "Quality gate drives retry and fallback"]),
        ("Quality Signals",
         ["Empty page ratio", "Garbled characters", "Missing locators", "Page count mismatch"]),
        ("Fallback Ladder",
         ["Primary route (default candidate)",
          "Local repair: force OCR",
          "High quality route only when policy allows"]),
        ("Evaluation Plan",
         ["50 document corpus for the first bake-off",
          "150-200 documents before public release",
          "Regression corpus pinned per parser version"]),
        ("Rollout",
         ["Feature flag keeps the legacy path",
          "Shadow run before switching default",
          "Rollback is a config change"]),
        ("Open Questions",
         ["Chinese lexical retrieval backend",
          "Reranker latency budget",
          "MinerU quota handling"]),
    ]
    for i, (title, bullets) in enumerate(slides):
        if i == 0:
            slide = prs.slides.add_slide(prs.slide_layouts[0])
            slide.shapes.title.text = title
            slide.placeholders[1].text = "\n".join(bullets)
        else:
            slide = prs.slides.add_slide(prs.slide_layouts[1])
            slide.shapes.title.text = title
            slide.placeholders[1].text = "\n".join(bullets)
    path = os.path.join(CORPUS, "pptx", "project_review_deck.pptx")
    prs.save(path)
    return path


# ---------------------------------------------------------------- PPTX 2
def build_notes_chart_deck():
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = "Retrieval Quality by Configuration"
    slide.placeholders[1].text = "Recall@10 and p95 latency across four configurations"

    chart_data = CategoryChartData()
    chart_data.categories = ["dense only", "dense+BM25", "+rerank", "+query expansion"]
    chart_data.add_series("Recall@10", (0.74, 0.79, 0.86, 0.87))
    slide2 = prs.slides.add_slide(prs.slide_layouts[5])
    slide2.shapes.title.text = "Recall by configuration"
    slide2.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, PInches(1), PInches(2),
                            PInches(8), PInches(4.2), chart_data)
    slide2.notes_slide.notes_text_frame.text = (
        "Key point: the reranker contributes +7 points of Recall@10 but costs "
        "roughly 42 ms of p95 latency. Recommendation: enable for offline analysis, "
        "keep disabled for interactive Q&A until the latency budget is revisited."
    )

    slide3 = prs.slides.add_slide(prs.slide_layouts[5])
    slide3.shapes.title.text = "Latency breakdown"
    lat = CategoryChartData()
    lat.categories = ["parse", "embed", "retrieve", "rerank", "generate"]
    lat.add_series("p95 ms", (540, 210, 180, 176, 920))
    slide3.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, PInches(1), PInches(2),
                            PInches(8), PInches(4.2), lat)
    slide3.notes_slide.notes_text_frame.text = (
        "Parsing dominates time to first answer. Caching parse results by content "
        "hash removes most of it on repeated uploads."
    )
    chart = make_chart_png(os.path.join(TMP, "chart_deck.png"))
    slide4 = prs.slides.add_slide(prs.slide_layouts[5])
    slide4.shapes.title.text = "Accuracy curves"
    slide4.shapes.add_picture(chart, PInches(1.2), PInches(1.8), width=PInches(7.5))
    slide4.notes_slide.notes_text_frame.text = (
        "The two curves separate after epoch two; the rescaled variant keeps "
        "improving while the baseline plateaus."
    )
    path = os.path.join(CORPUS, "pptx", "deck_with_notes_and_charts.pptx")
    prs.save(path)
    return path


# ---------------------------------------------------------------- PPTX 3
def build_zh_en_deck():
    prs = Presentation()
    data = [
        ("企业级文档智能：方案评审",
         ["目标：统一解析入口与证据模型",
          "范围：PDF / DOCX / PPTX 三类优先",
          "约束：隐私模式禁止静默云端调用"]),
        ("当前问题诊断",
         ["三个抽取库把文档压成纯文本",
          "缺少页码、元素类型与定位信息",
          "下游任务只看前 N 个字符"]),
        ("目标架构",
         ["Parser Gateway + 可插拔 adapter",
          "Canonical ParsedDocument + SourceLocator",
          "质量门控驱动重试与降级"]),
        ("质量信号",
         ["空页比例", "乱码比例", "定位缺失率", "页数不一致"]),
        ("降级链路",
         ["默认路由（由评测决定）",
          "本地修复：强制 OCR",
          "仅在策略允许时走高质量云端路由"]),
        ("评测计划",
         ["首轮 50 份语料做 bake-off",
          "公开发布前扩展到 150-200 份",
          "每个 parser 版本固定回归集"]),
    ]
    for i, (title, bullets) in enumerate(data):
        layout = prs.slide_layouts[0] if i == 0 else prs.slide_layouts[1]
        slide = prs.slides.add_slide(layout)
        slide.shapes.title.text = title
        slide.placeholders[1].text = "\n".join(bullets)
    path = os.path.join(CORPUS, "pptx", "bilingual_zh_en_deck.pptx")
    prs.save(path)
    return path


def main():
    for fn in (build_technical_report, build_figures_doc, build_formula_doc, build_bilingual_doc,
               build_plain_deck, build_notes_chart_deck, build_zh_en_deck):
        p = fn()
        size = os.path.getsize(p)
        print(f"[ok] {os.path.relpath(p, CORPUS)} ({size // 1024} KB)")


if __name__ == "__main__":
    main()
