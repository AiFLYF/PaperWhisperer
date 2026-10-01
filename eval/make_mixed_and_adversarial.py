# -*- coding: utf-8 -*-
"""Build mixed Chinese/English PDFs and adversarial files for the eval corpus.

mixed_zh_en/   4 bilingual PDFs with real CJK text layers (reportlab STSong-Light)
adversarial/   3+ malformed files: corrupted PDF, encrypted PDF, fake extension,
               nested-zip blowup sample. Never include these in the normal batch run.

Paths are resolved from this script's own location, so it runs from a fresh
clone in any directory on any platform. Note that reportlab supplies the CJK
glyphs here through its built-in STSong-Light CID font, which requires no
system font installation at all.
"""
from __future__ import annotations

import io
import os
import zipfile

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle, ListFlowable, ListItem)

# Anchor every path to this script's own directory. The corpus is addressed
# relative to the repo checkout, not to whatever machine or working directory
# happened to build it.
BASE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(BASE, "corpus")
GT_DIR = os.path.join(BASE, "ground_truth")
os.makedirs(GT_DIR, exist_ok=True)

pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))

ss = getSampleStyleSheet()
STYLE_TITLE = ParagraphStyle("zhTitle", parent=ss["Title"], fontName="STSong-Light", fontSize=17, leading=23)
STYLE_H1 = ParagraphStyle("zhH1", parent=ss["Heading1"], fontName="STSong-Light", fontSize=14, leading=19,
                          spaceBefore=10, spaceAfter=5)
STYLE_H2 = ParagraphStyle("zhH2", parent=ss["Heading2"], fontName="STSong-Light", fontSize=12, leading=17,
                          spaceBefore=8, spaceAfter=4)
STYLE_BODY = ParagraphStyle("zhBody", parent=ss["BodyText"], fontName="STSong-Light", fontSize=10, leading=15,
                            firstLineIndent=20)
STYLE_EN_BODY = ParagraphStyle("enBody", parent=ss["BodyText"], fontName="Helvetica", fontSize=10, leading=14)
STYLE_CAPTION = ParagraphStyle("cap", parent=ss["BodyText"], fontName="STSong-Light", fontSize=9, leading=12,
                               alignment=1, textColor=colors.HexColor("#444444"))


class DocBuilder:
    """Collects flowables and records ground-truth text."""

    def __init__(self):
        self.flow = []
        self.gt = []

    def add(self, flowable, text=None):
        self.flow.append(flowable)
        if text:
            self.gt.append(text)

    def sp(self, h=6):
        self.flow.append(Spacer(1, h))

    def para(self, text, style=STYLE_BODY):
        self.add(Paragraph(text, style), text)


def page_furniture(canvas, doc):
    """Running header + footer: exercises header/footer repetition detection."""
    canvas.saveState()
    canvas.setFont("STSong-Light", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(20 * mm, A4[1] - 12 * mm, "评测语料 · 中英混排 · internal-eval")
    canvas.drawRightString(A4[0] - 20 * mm, A4[1] - 12 * mm, "PaperWhisperer eval corpus")
    canvas.setLineWidth(0.4)
    canvas.line(20 * mm, A4[1] - 14 * mm, A4[0] - 20 * mm, A4[1] - 14 * mm)
    canvas.drawCentredString(A4[0] / 2, 12 * mm, f"第 {doc.page} 页")
    canvas.restoreState()


def build(doc_path, builder):
    doc = BaseDocTemplate(doc_path, pagesize=A4,
                          leftMargin=20 * mm, rightMargin=20 * mm,
                          topMargin=18 * mm, bottomMargin=18 * mm,
                          title="PaperWhisperer mixed zh-en eval document")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=page_furniture)])
    doc.build(builder.flow)


def table_style():
    return TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8e8")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#888888")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ])


# ---------------------------------------------------------------- 1
def bilingual_paper():
    b = DocBuilder()
    b.para("多模态检索系统在企业知识库中的应用评估", STYLE_TITLE)
    b.para("<b>Abstract.</b> We evaluate a multimodal retrieval system combining dense "
           "embedding and BM25 ranking on an enterprise knowledge base of 128,000 documents. "
           "Recall@10 improves from 0.71 to 0.86 after adding a cross-encoder reranker, "
           "while p95 latency increases by 42 ms.", STYLE_EN_BODY)
    b.sp(4)
    b.para("关键词：多模态检索、hybrid retrieval、reranker、企业知识库、Recall@10", STYLE_BODY)
    b.para("1. 引言", STYLE_H1)
    b.para("企业知识库通常同时包含 PDF 研究报告、DOCX 制度文档与 PPTX 汇报材料，单一模态的检索方案"
           "难以覆盖全部内容类型。本文以 hybrid retrieval 为基线，系统评估 reranker 与 "
           "query expansion 两项技术的增量收益，并给出不同延迟预算下的配置建议。", STYLE_BODY)
    b.para("1.1 研究问题", STYLE_H2)
    items = ["在中文为主的企业语料上，BM25 词法通道是否仍然必要？",
             "cross-encoder reranker 的延迟代价是否可接受？",
             "query expansion 在长尾 query 上的收益是否稳定？"]
    b.add(ListFlowable([ListItem(Paragraph(t, STYLE_BODY), leftIndent=14) for t in items],
                       bulletType="1"), "\n".join(items))
    b.para("2. 方法", STYLE_H1)
    b.para("检索流程包含四个阶段：query rewrite、dense retrieval、lexical retrieval 与 "
           "rank fusion。融合采用 reciprocal rank fusion (RRF)，其中 k 取 60。", STYLE_BODY)
    b.para("score(q, d) = alpha * cos(E(q), E(d)) + (1 - alpha) * BM25(q, d)", STYLE_EN_BODY)
    b.para("3. 实验设置", STYLE_H1)
    b.para("语料规模为 128,000 篇文档，其中中文文档占比 61%，英文文档占比 34%，中英混排文档占比 5%。"
           "Embedding 维度为 1024，chunk size 为 512 tokens，overlap 为 64 tokens。", STYLE_BODY)
    data = [["配置", "Recall@10", "MRR", "p95 延迟 (ms)"],
            ["dense only", "0.74", "0.52", "118"],
            ["dense + BM25", "0.79", "0.57", "134"],
            ["dense + BM25 + rerank", "0.86", "0.63", "176"],
            ["+ query expansion", "0.87", "0.64", "198"]]
    tbl = Table(data, colWidths=[62 * mm, 28 * mm, 24 * mm, 32 * mm])
    tbl.setStyle(table_style())
    b.add(tbl, " | ".join(r[0] for r in data) + " :: " + " ; ".join(" / ".join(r) for r in data[1:]))
    b.add(Spacer(1, 4))
    b.para("表 1：四种配置在测试集上的检索质量与延迟", STYLE_CAPTION)
    b.para("4. 结果分析", STYLE_H1)
    b.para("在中文语料上，BM25 通道的贡献明显高于英文语料（Recall@10 提升 0.05 对 0.02），"
           "说明中文分词与 term-level 匹配仍是必要补充。Reranker 带来 0.07 的 Recall@10 提升，"
           "代价是 42 ms 的 p95 延迟增长。", STYLE_BODY)
    b.para("5. 结论", STYLE_H1)
    b.para("对于交互式问答场景建议关闭 reranker 以控制延迟；对于离线批量分析场景建议开启。"
           "后续工作将评估 BGE-M3 sparse retrieval 替代 BM25 的可行性。", STYLE_BODY)
    return b


# ---------------------------------------------------------------- 2
def tech_report():
    b = DocBuilder()
    b.para("2026 年第一季度平台稳定性与容量评估", STYLE_TITLE)
    b.para("本报告汇总 2026 Q1 期间平台各服务的可用性、延迟与容量数据，"
           "所有数值来自生产环境的 Prometheus 指标与 APM trace 采样。", STYLE_BODY)
    b.para("1. 总体指标", STYLE_H1)
    data = [["服务", "可用性", "p50 (ms)", "p95 (ms)", "错误率"],
            ["api-gateway", "99.98%", "42", "187", "0.02%"],
            ["parser-worker", "99.91%", "1,240", "8,600", "0.09%"],
            ["embedding-proxy", "99.95%", "88", "410", "0.05%"],
            ["retrieval", "99.97%", "61", "238", "0.03%"]]
    tbl = Table(data, colWidths=[38 * mm, 24 * mm, 24 * mm, 24 * mm, 24 * mm])
    tbl.setStyle(table_style())
    b.add(tbl, " ; ".join(" / ".join(r) for r in data))
    b.add(Spacer(1, 4))
    b.para("表 1：核心服务 SLO 达成情况（2026 Q1）", STYLE_CAPTION)
    b.para("2. 容量与瓶颈", STYLE_H1)
    b.para("parser-worker 的 p95 延迟为 8,600 ms，主要由 OCR 路径贡献。当前单机 "
           "CPU 利用率为 78%，队列积压峰值出现在每日 09:00-11:00 时段。", STYLE_BODY)
    b.para("2.1 弹性策略建议", STYLE_H2)
    items = ["将 OCR 任务拆分到独立队列，避免阻塞常规解析",
             "对同一 sha256 的重复上传启用解析缓存",
             "在队列积压超过 50 个任务时自动扩容 worker"]
    b.add(ListFlowable([ListItem(Paragraph(t, STYLE_BODY), leftIndent=14) for t in items],
                       bulletType="bullet"), "\n".join(items))
    b.para("3. 风险登记", STYLE_H1)
    data2 = [["风险项", "可能性", "影响", "缓解措施"],
             ["MinerU 额度下调", "中", "高", "本地 OCR fallback"],
             ["embedding provider 超时", "低", "中", "本地缓存 + 重试"],
             ["磁盘水位过高", "中", "中", "retention policy 30 天"]]
    t2 = Table(data2, colWidths=[44 * mm, 22 * mm, 20 * mm, 52 * mm])
    t2.setStyle(table_style())
    b.add(t2, " ; ".join(" / ".join(r) for r in data2))
    b.para("4. 下季度计划", STYLE_H1)
    b.para("优先完成 OCR 队列拆分与解析缓存，两项合计预计将 parser-worker 的 p95 延迟"
           "降低约 40%，并将云解析成本降低 25%。", STYLE_BODY)
    return b


# ---------------------------------------------------------------- 3
def standard_doc():
    b = DocBuilder()
    b.para("信息技术　文档解析质量评测规范", STYLE_TITLE)
    b.para("第 1 部分：术语与评测框架（征求意见稿）", STYLE_H2)
    b.para("1 范围", STYLE_H1)
    b.para("本文件规定了文档解析系统的质量评测术语、评测语料构成、指标定义与判定规则，"
           "适用于数字原生文档与扫描文档的解析质量评测。", STYLE_BODY)
    b.para("2 规范性引用文件", STYLE_H1)
    b.para("下列文件中的内容通过文中的规范性引用而构成本文件必不可少的条款。"
           "其中，注日期的引用文件，仅该日期对应的版本适用于本文件。", STYLE_BODY)
    b.para("3 术语和定义", STYLE_H1)
    terms = [("3.1 解析单元 source unit", "文档中被解析系统识别的最小来源单位，"
              "如 PDF 的页、PPTX 的幻灯片、DOCX 的节。"),
             ("3.2 定位符 source locator", "用于将解析结果中的元素映射回原始文档位置的信息集合。"),
             ("3.3 降级 fallback", "当默认解析路径质量不达标时切换到替代路径的过程。"),
             ("3.4 静默截断 silent truncation", "解析结果丢失原文内容但未向用户提示的情形。")]
    for t, d in terms:
        b.para(t, STYLE_H2)
        b.para(d, STYLE_BODY)
    b.para("4 评测语料构成", STYLE_H1)
    b.para("评测语料应覆盖数字原生 PDF、双栏论文、扫描件、公式密集文档、"
           "表格密集文档、中英混排文档、DOCX、PPTX 与异常文件九类。", STYLE_BODY)
    data = [["类别", "配额", "主要关注点"],
            ["数字原生 PDF", "17", "基线质量与阅读顺序"],
            ["扫描件", "11", "OCR 准确率与降级"],
            ["中英混排", "4", "多语言词法检索"],
            ["异常文件", "3", "错误处理与安全性"]]
    tbl = Table(data, colWidths=[46 * mm, 22 * mm, 66 * mm])
    tbl.setStyle(table_style())
    b.add(tbl, " ; ".join(" / ".join(r) for r in data))
    b.para("5 判定规则", STYLE_H1)
    b.para("出现任一硬失败信号即判定该文档解析失败；出现两个及以上警告信号应触发降级。"
           "静默截断在正式发布版本中的允许次数为零。", STYLE_BODY)
    return b


# ---------------------------------------------------------------- 4
def dual_language_survey():
    b = DocBuilder()
    b.para("A Survey of Evidence-Grounded Document Analysis", STYLE_TITLE)
    b.para("<b>English body with Chinese section summaries.</b> This sample exercises "
           "parsers and retrievers on documents where the primary language alternates "
           "between sections.", STYLE_EN_BODY)
    sections = [
        ("1. Introduction",
         "Evidence-grounded analysis requires that every claim in a generated answer "
         "can be traced back to a concrete location in the source document. Early "
         "summarisation systems flattened documents into plain text and therefore lost "
         "page structure, element types and provenance.",
         "证据驱动的分析要求每条结论都能回溯到原文的具体位置，早期的摘要系统"
         "把文档压平为纯文本，丢失了版面结构、元素类型与来源信息。"),
        ("2. Parsing and Provenance",
         "Modern parser gateways expose a canonical document model with elements, "
         "tables, figures, formulas and source locators. The locator schema differs per "
         "format: PDF pages with bounding boxes, slides for decks, element references "
         "for word processor files.",
         "现代解析网关提供统一的文档模型，包含元素、表格、图片、公式与定位符。"
         "定位语义按格式而异：PDF 用页码与包围盒，演示文稿用幻灯片，文字处理文档用元素引用。"),
        ("3. Retrieval Strategies",
         "Hybrid retrieval combines dense semantic recall with lexical matching. For "
         "mixed Chinese and English corpora, lexical matching on Chinese text needs "
         "substring or trigram strategies rather than whitespace tokenisation alone.",
         "混合检索结合稠密语义召回与词法匹配。对中英文混合语料，中文部分的词法匹配"
         "需要子串或 trigram 策略，不能只依赖空格分词。"),
        ("4. Verification",
         "Direct quotations must be verified programmatically against the canonical "
         "source text. Quotes that fail exact or normalised alignment are downgraded to "
         "paraphrase and must not be rendered with quotation marks.",
         "直接引文必须以程序方式针对规范原文进行校验，无法逐字对齐的内容应降级为"
         "改写，且不得以引号形式呈现。"),
        ("5. Open Problems",
         "Remaining problems include cross-page table recovery, reading order in "
         "three-column layouts, and cost control for on-demand vision models.",
         "尚未解决的问题包括跨页表格还原、三栏版式的阅读顺序恢复，以及按需视觉模型的成本控制。"),
    ]
    for title, en, zh in sections:
        b.para(title, STYLE_H1)
        b.para(en, STYLE_EN_BODY)
        b.para("本节要点：" + zh, STYLE_BODY)
    return b


def write_gt(name, lines):
    with open(os.path.join(GT_DIR, name + ".txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------- adversarial
def build_adversarial():
    adv = os.path.join(CORPUS, "adversarial")
    os.makedirs(adv, exist_ok=True)
    out = []

    # 1. corrupted PDF: truncate a valid PDF
    src = os.path.join(CORPUS, "scanned_clear", "paper_en_basic.pdf")
    with open(src, "rb") as fh:
        data = fh.read()
    trunc = data[: int(len(data) * 0.35)]
    p = os.path.join(adv, "corrupted_truncated.pdf")
    with open(p, "wb") as fh:
        fh.write(trunc)
    out.append(("corrupted_truncated.pdf", "valid PDF truncated to 35% of its bytes; parser must report a clean error"))

    # 2. encrypted PDF
    from pypdf import PdfReader, PdfWriter
    reader = PdfReader(io.BytesIO(data))
    writer = PdfWriter()
    writer.append(reader)
    writer.encrypt("paperwhisperer-eval", algorithm="AES-128")
    p = os.path.join(adv, "encrypted_aes128.pdf")
    with open(p, "wb") as fh:
        writer.write(fh)
    out.append(("encrypted_aes128.pdf", "AES-128 encrypted PDF, password 'paperwhisperer-eval'; must not crash ingestion"))

    # 3. fake extension: text file named .pdf
    p = os.path.join(adv, "fake_extension_actually_text.pdf")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("This is not a PDF at all.\n")
        fh.write("%PDF marker is only mentioned here, not present in the header.\n")
        fh.write("Detecting this requires magic-byte checking, not extension checking.\n")
    out.append(("fake_extension_actually_text.pdf", "plain text with .pdf extension; extension-only validation would pass it"))

    # 4. nested zip blowup sample (harmless: ~1 KB compressed, ~2 MB expanded)
    p = os.path.join(adv, "zip_bomb_nested.zip")
    payload = ("PaperWhisperer eval adversarial sample. " * 900).encode("utf-8")
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as z:
        for i in range(24):
            z.writestr(f"layer_{i:02d}.txt", payload)
    sz_in = os.path.getsize(p)
    with zipfile.ZipFile(p) as z:
        sz_out = sum(i.file_size for i in z.infolist())
    out.append((f"zip_bomb_nested.zip ({sz_in // 1024} KB -> {sz_out // 1024} KB expanded)",
                "nested archive blowup sample; tests decompression limits"))

    return out


def main():
    jobs = [
        ("bilingual_paper_zh_en.pdf", bilingual_paper),
        ("tech_report_zh_terms.pdf", tech_report),
        ("zh_standard_document.pdf", standard_doc),
        ("dual_language_survey.pdf", dual_language_survey),
    ]
    for name, fn in jobs:
        b = fn()
        build(os.path.join(CORPUS, "mixed_zh_en", name), b)
        write_gt(name.replace(".pdf", ""), b.gt)
        print(f"[zh-en] {name} ({os.path.getsize(os.path.join(CORPUS, 'mixed_zh_en', name)) // 1024} KB, {len(b.gt)} gt blocks)")

    for name, note in build_adversarial():
        print(f"[adversarial] {name} -- {note}")


if __name__ == "__main__":
    main()
