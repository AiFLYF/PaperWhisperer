# -*- coding: utf-8 -*-
"""Construct scanned-style PDF corpus for PaperWhisperer eval (T0.1).

Generates:
- scanned_clear/    5 clean 300-DPI image-only PDFs (EN/ZH, formulas, tables, two-column)
- hybrid_scan_text/ 3 image + invisible text layer PDFs
- scanned_lowq/     3 degraded variants (rotation, downscale, noise, JPEG)

Ground truth text is written to eval/ground_truth/<name>.txt so OCR quality
can be scored exactly.

Paths are resolved from this script's own location, so the script runs from a
fresh clone in any directory on any platform. Fonts are the one genuinely
platform-dependent input: a scanned corpus is only meaningful if the glyphs
are real, so the font is resolved from an ordered candidate list rather than
assumed to live in C:\\Windows\\Fonts. Override with the PW_EN_FONT,
PW_EN_BOLD_FONT and PW_ZH_FONT environment variables to pin exact files.
"""
from __future__ import annotations

import io
import os
import random

from PIL import Image, ImageDraw, ImageFilter, ImageFont

# Anchor every path to this script's own directory. The corpus is addressed
# relative to the repo checkout, not to whatever machine or working directory
# happened to build it.
BASE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(BASE, "corpus")
GT_DIR = os.path.join(BASE, "ground_truth")
os.makedirs(GT_DIR, exist_ok=True)


def _pick_font(env_var, candidates):
    """First existing candidate wins; None means "let PIL pick its default"."""
    override = os.environ.get(env_var)
    if override:
        return override
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


# Ordered most-specific-first. The original Windows faces stay first so a
# Windows rebuild is byte-identical to what produced the committed corpus; the
# Linux/macOS entries exist so the same corpus is reproducible elsewhere.
EN_FONT = _pick_font("PW_EN_FONT", [
    r"C:\Windows\Fonts\times.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
])
EN_BOLD = _pick_font("PW_EN_BOLD_FONT", [
    r"C:\Windows\Fonts\timesbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
])
ZH_FONT = _pick_font("PW_ZH_FONT", [
    r"C:\Windows\Fonts\simsun.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
])

A4_W_IN, A4_H_IN = 8.27, 11.69
DPI = 300


def font(path, px):
    # A missing font is a real risk on a non-Windows host, but falling back to
    # PIL's bitmap default would silently produce a corpus of illegible boxes
    # that still passes the file-count check. So: try the resolved path, then
    # any system TrueType face, and warn loudly if neither is available so the
    # failure is attributable to fonts rather than showing up later as a bogus
    # OCR accuracy score.
    if path:
        try:
            return ImageFont.truetype(path, px)
        except OSError:
            pass
    for candidate in ImageFont.truetype_fonts():
        try:
            return ImageFont.truetype(candidate, px)
        except OSError:
            continue
    print(f"WARNING: no usable TrueType font for {path} at {px}px; "
          f"output will be near-illegible. Set PW_EN_FONT / PW_EN_BOLD_FONT / PW_ZH_FONT.")
    return ImageFont.load_default()


def wrap(draw, text, fnt, max_w):
    lines, cur = [], ""
    for ch in text:
        trial = cur + ch
        if draw.textlength(trial, font=fnt) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = ch
    if cur:
        lines.append(cur)
    return lines


def render_page_blocks(page_h_px, page_w_px, blocks, margin=170):
    """blocks: list of (kind, payload) rendered top-down.
    kinds: title, heading, para, formula, spacer, table, twocol_paras
    Returns (image, text_lines_groundtruth)
    """
    img = Image.new("L", (page_w_px, page_h_px), 255)
    draw = ImageDraw.Draw(img)
    y = margin
    gt = []
    for kind, payload in blocks:
        if kind == "spacer":
            y += payload
            continue
        if kind == "title":
            f = font(EN_BOLD, 44)
            for ln in payload.split("\n"):
                w = draw.textlength(ln, font=f)
                draw.text(((page_w_px - w) / 2, y), ln, font=f, fill=20)
                y += 56
                gt.append(ln)
            y += 20
        elif kind == "zh_title":
            f = font(ZH_FONT, 44)
            for ln in payload:
                w = draw.textlength(ln, font=f)
                draw.text(((page_w_px - w) / 2, y), ln, font=f, fill=20)
                y += 56
                gt.append(ln)
            y += 20
        elif kind == "heading":
            f = font(EN_BOLD, 30)
            draw.text((margin, y), payload, font=f, fill=20)
            y += 44
            gt.append(payload)
        elif kind == "zh_heading":
            f = font(ZH_FONT, 30)
            draw.text((margin, y), payload, font=f, fill=20)
            y += 44
            gt.append(payload)
        elif kind == "para":
            f = font(EN_FONT, 24)
            for ln in wrap(draw, payload, f, page_w_px - 2 * margin):
                draw.text((margin, y), ln, font=f, fill=30)
                y += 36
                gt.append(ln)
            y += 16
        elif kind == "zh_para":
            f = font(ZH_FONT, 24)
            for ln in wrap(draw, payload, f, page_w_px - 2 * margin):
                draw.text((margin, y), ln, font=f, fill=30)
                y += 36
                gt.append(ln)
            y += 16
        elif kind == "formula":
            f = font(EN_FONT, 26)
            w = draw.textlength(payload, font=f)
            draw.text(((page_w_px - w) / 2, y), payload, font=f, fill=15)
            y += 40
            gt.append(payload)
        elif kind == "table":
            rows, cols = payload["shape"]
            data = payload["data"]
            cw = (page_w_px - 2 * margin) // cols
            rh = 48
            f = font(EN_FONT, 20)
            # header band
            draw.rectangle([margin, y, margin + cols * cw, y + rh], fill=230)
            for c in range(cols + 1):
                draw.line([(margin + c * cw, y), (margin + c * cw, y + rows * rh)], fill=40, width=2)
            for r in range(rows + 1):
                draw.line([(margin, y + r * rh), (margin + cols * cw, y + r * rh)], fill=40, width=2)
            for r in range(rows):
                for c in range(cols):
                    txt = data[r][c]
                    tw = draw.textlength(txt, font=f)
                    draw.text((margin + c * cw + (cw - tw) / 2, y + r * rh + 12), txt, font=f, fill=20)
                    gt.append(txt)
            y += rows * rh + 24
        elif kind == "twocol_paras":
            f = font(EN_FONT, 24)
            gap = 90
            col_w = (page_w_px - 2 * margin - gap) / 2
            col_h = payload.get("col_h", page_h_px - y - margin)
            x0, x1 = margin, margin + col_w + gap
            yy0 = yy1 = y
            for i, ln in enumerate(payload["left"]):
                if yy0 + 36 > y + col_h:
                    break
                draw.text((x0, yy0), ln, font=f, fill=30)
                yy0 += 36
                gt.append(ln)
            for ln in payload["right"]:
                if yy1 + 36 > y + col_h:
                    break
                draw.text((x1, yy1), ln, font=f, fill=30)
                yy1 += 36
                gt.append(ln)
            y = max(yy0, yy1) + 16
        if y > page_h_px - margin * 0.5:
            break
    return img, gt


def img_to_pdf_bytes(img, dpi=DPI):
    buf = io.BytesIO()
    img.save(buf, format="PDF", resolution=dpi)
    return buf.getvalue()


def save_image_pdf(path, images, dpi=DPI):
    first = images[0]
    rest = images[1:]
    first.save(path, format="PDF", save_all=True, append_images=rest, resolution=dpi)


def degrade(img, rotate_deg, scale, noise, jpeg_q):
    if rotate_deg:
        img = img.rotate(rotate_deg, resample=Image.BICUBIC, fillcolor=255, expand=False)
    if scale != 1.0:
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    if noise:
        px = img.load()
        rand = random.Random(42)
        n = int(img.width * img.height * noise)
        for _ in range(n):
            x, y = rand.randrange(img.width), rand.randrange(img.height)
            v = rand.choice((0, 255))
            px[x, y] = v
        img = img.filter(ImageFilter.GaussianBlur(0.6))
    if jpeg_q:
        buf = io.BytesIO()
        img.convert("L").save(buf, format="JPEG", quality=jpeg_q)
        img = Image.open(buf).convert("L")
    return img


# ---------- minimal raw PDF text layer (invisible) ----------
def text_pdf_bytes(page_w_pt, page_h_pt, lines, margin_pt=54, leading=12, size=10):
    """Build a tiny single-page PDF with invisible text (Tr 3), Helvetica."""
    ops = ["BT", "3 Tr", f"/F1 {size} Tf"]
    y = page_h_pt - margin_pt
    for ln in lines:
        safe = ln.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        ops.append(f"1 0 0 1 {margin_pt} {y:.1f} Tm")
        ops.append(f"({safe}) Tj")
        y -= leading
        if y < margin_pt:
            break
    ops.append("ET")
    content = "\n".join(ops).encode("latin-1", "replace")
    objs = []
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_w_pt} {page_h_pt}] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>".encode())
    objs.append(b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = [0]
    for i, body in enumerate(objs, start=1):
        offsets.append(out.tell())
        out.write(f"{i} 0 obj\n".encode() + body + b"\nendobj\n")
    xref_pos = out.tell()
    n = len(objs) + 1
    out.write(f"xref\n0 {n}\n".encode())
    out.write(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(f"trailer\n<< /Size {n} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF".encode())
    return out.getvalue()


def hybrid_pdf(path, img, text_lines, dpi=DPI):
    """Image-only page + invisible text layer merged via pypdf."""
    from pypdf import PdfReader, PdfWriter
    img_buf = io.BytesIO()
    img.save(img_buf, format="PDF", resolution=dpi)
    img_reader = PdfReader(img_buf)
    page_w_pt = float(img_reader.pages[0].mediabox.width)
    page_h_pt = float(img_reader.pages[0].mediabox.height)
    # scale text layer font to match image resolution roughly
    scale = dpi / 72.0
    tl_bytes = text_pdf_bytes(page_w_pt, page_h_pt, text_lines, margin_pt=int(42 * scale / scale),
                              leading=int(36 * 72 / dpi * 1.0) or 12, size=max(8, int(24 * 72 / dpi)))
    tl_reader = PdfReader(io.BytesIO(tl_bytes))
    base_page = img_reader.pages[0]
    base_page.merge_page(tl_reader.pages[0])
    writer = PdfWriter()
    writer.add_page(base_page)
    with open(path, "wb") as fh:
        writer.write(fh)


# ------------------------------------------------------------------
EN_PAPER = [
    ("title", "A Study of Gradient Dynamics in Deep\nResidual Networks"),
    ("para", "Deep residual networks have become the dominant architecture for a broad range of perception tasks. In this work we revisit the optimization geometry of residual connections and derive improved scaling rules for very deep networks. We train networks of depth 50 to 200 layers on image classification benchmarks and observe consistent gains."),
    ("heading", "1. Introduction"),
    ("para", "Residual learning reframes the mapping problem: instead of fitting a desired underlying mapping H(x) directly, the stacked layers fit a residual F(x) = H(x) - x. This reformulation eases optimization and enables training of networks that are substantially deeper than previously possible. Prior work has studied normalization and initialization; here we focus on the interaction between residual branches and learning-rate schedules."),
    ("heading", "2. Related Work"),
    ("para", "Highway networks introduced gated shortcut connections, while identity shortcuts were popularized by residual networks. Later analyses connected residual flows to ordinary differential equations, motivating neural ODE solvers. Our analysis complements this line by quantifying the variance of gradient updates across depth."),
    ("heading", "3. Method"),
    ("formula", "y_l = x_l + F(x_l, W_l)"),
    ("formula", "x_(l+1) = lambda * x_l + mu * F(x_l, W_l)"),
    ("para", "We introduce two scalar hyper-parameters lambda and mu that control the trade-off between the identity path and the residual branch. When lambda equals one and mu equals one we recover the standard residual block. Setting lambda greater than one effectively rescales the network depth, and we prove a variance preservation condition under which gradients neither vanish nor explode."),
    ("heading", "4. Experiments"),
    ("table", {"shape": (5, 4), "data": [
        ["depth", "lambda", "top-1 err.", "params (M)"],
        ["50", "1.0", "24.7", "25.6"],
        ["101", "1.0", "23.6", "44.5"],
        ["152", "1.0", "23.0", "60.2"],
        ["152", "1.2", "22.4", "60.2"],
    ]}),
    ("para", "Table 1 summarizes classification accuracy. The rescaled variant improves top-1 error by 0.6 points at identical parameter count. Training curves show that the benefit concentrates in the late phase of training, when the identity path dominates gradient flow."),
    ("heading", "5. Conclusion"),
    ("para", "We presented a simple reparameterization of residual networks with provable variance preservation and consistent empirical gains. The approach requires no extra parameters and composes with existing normalization schemes."),
]

ZH_PAPER = [
    ("zh_title", ["深度残差网络中的梯度动力学研究"]),
    ("zh_para", "深度残差网络已成为各类感知任务的主流架构。本文重新审视残差连接的优化几何，推导出适用于极深网络的改进缩放规则。我们在图像分类基准上训练了 50 至 200 层的网络，观察到一致的增益。实验代码与配置文件将随论文一并公开，供社区复现。"),
    ("zh_heading", "1. 引言"),
    ("zh_para", "残差学习将映射问题重新表述为：堆叠层拟合残差 F(x) = H(x) - x，而不是直接拟合目标映射 H(x)。这一重构显著降低了优化难度，使得训练比以往深得多的网络成为可能。本文关注残差分支与学习率调度之间的相互作用，并给出方差保持条件的证明。"),
    ("zh_heading", "2. 方法"),
    ("formula", "y_l = x_l + F(x_l, W_l)"),
    ("zh_para", "我们引入两个标量超参数 lambda 与 mu，控制恒等路径与残差分支之间的权衡。当二者均为 1 时退化为标准残差块。将 lambda 设为大于 1 等效于重新缩放网络深度，我们在方差保持条件下证明了梯度既不消失也不爆炸。"),
    ("zh_heading", "3. 实验"),
    ("table", {"shape": (5, 4), "data": [
        ["深度", "lambda", "top-1 误差", "参数量 (M)"],
        ["50", "1.0", "24.7", "25.6"],
        ["101", "1.0", "23.6", "44.5"],
        ["152", "1.0", "23.0", "60.2"],
        ["152", "1.2", "22.4", "60.2"],
    ]}),
    ("zh_para", "表 1 汇总了分类精度。重新缩放的变体在参数量相同的情况下将 top-1 误差降低了 0.6 个百分点。训练曲线显示收益集中在训练后期，此时恒等路径主导梯度流动。"),
    ("zh_heading", "4. 结论"),
    ("zh_para", "本文提出了深度残差网络的一种简单重参数化方法，具有可证明的方差保持性质与一致的实证增益。该方法不引入额外参数，并可与现有归一化方案组合使用。"),
]

EN_FORMULA_PAPER = [
    ("title", "Functional Integral Representations\nin Quantum Field Theory"),
    ("para", "We review the construction of functional integrals for scalar and gauge fields and derive the one-loop effective action by the background field method. Explicit computations are carried out for the O(N) model in two and four dimensions."),
    ("heading", "1. The Generating Functional"),
    ("formula", "Z[J] = integral D phi exp(i S[phi] + i integral J phi)"),
    ("formula", "S[phi] = integral d^4x (1/2 partial phi partial phi - 1/2 m^2 phi^2 - lambda/4! phi^4)"),
    ("para", "The generating functional Z[J] formally integrates over all field configurations weighted by the exponential of the classical action. Perturbation theory arises from expanding around the stationary point of the action functional."),
    ("heading", "2. Background Field Method"),
    ("formula", "Gamma[phi_B] = -i ln Z[J] - integral J phi_B"),
    ("formula", "d = 4 - epsilon,  lambda_R = lambda mu^epsilon Z_lambda"),
    ("para", "Splitting the field into a classical background and a quantum fluctuation preserves gauge invariance of the effective action order by order. The divergent part of the one-loop determinant yields the beta function of the quartic coupling."),
    ("heading", "3. Renormalization"),
    ("formula", "beta(lambda) = 3 lambda^2 / (16 pi^2) + O(lambda^3)"),
    ("para", "The one-loop beta function shows the theory is not asymptotically free for positive lambda. We compare with exact results available in two dimensions and discuss the resummation of large logarithms."),
]

EN_TABLE_PAPER = [
    ("title", "Benchmark Protocol and Statistical Analysis\nof Language Model Evaluations"),
    ("para", "We present a reproducible evaluation protocol covering twelve benchmarks, five model scales, and three decoding strategies. All runs use fixed seeds, and we report means and 95 percent confidence intervals over five restarts."),
    ("heading", "1. Evaluation Suite"),
    ("table", {"shape": (6, 5), "data": [
        ["benchmark", "metric", "n-shot", "seed set", "pass@1"],
        ["ARC-C", "accuracy", "0-shot", "S1-S5", "0.612"],
        ["HellaSwag", "accuracy", "10-shot", "S1-S5", "0.734"],
        ["MMLU", "accuracy", "5-shot", "S1-S5", "0.589"],
        ["GSM8K", "exact match", "8-shot", "S1-S5", "0.402"],
        ["HumanEval", "pass@1", "0-shot", "S1-S5", "0.317"],
    ]}),
    ("heading", "2. Variance Decomposition"),
    ("table", {"shape": (5, 4), "data": [
        ["source", "variance share", "stdev", "note"],
        ["decoding", "0.41", "1.9 pts", "temp 0.0-0.8"],
        ["restarts", "0.27", "1.2 pts", "seed only"],
        ["prompt fmt", "0.19", "0.9 pts", "whitespace"],
        ["hardware", "0.13", "0.6 pts", "fp16 vs bf16"],
    ]}),
    ("para", "Decoding temperature contributes the largest share of run-to-run variance, followed by random restarts. Prompt formatting effects are smaller but statistically significant across all benchmarks. We recommend reporting interval estimates rather than single-point scores."),
    ("heading", "3. Recommendations"),
    ("para", "Our protocol reduces reported variance by 48 percent relative to common practice. We release all prompts, seeds, and parsing code to make single-number leaderboards directly reproducible."),
]

def two_col_blocks():
    left_intro = ("Deep residual networks have become the dominant architecture for a broad range of "
                  "perception tasks. In this work we revisit the optimization geometry of residual "
                  "connections and derive improved scaling rules for very deep networks. We train "
                  "networks of depth 50 to 200 layers on image classification benchmarks and observe "
                  "consistent gains across all scales tested. The analysis relies on a variance "
                  "preservation condition that we prove in Section 3.")
    right_intro = ("Residual learning reframes the mapping problem: instead of fitting a desired "
                   "underlying mapping H(x) directly, the stacked layers fit a residual F(x) = H(x) - x. "
                   "This reformulation eases optimization and enables training of networks that are "
                   "substantially deeper than previously possible. Prior work has studied normalization "
                   "and initialization; here we focus on the interaction between residual branches and "
                   "learning-rate schedules in the late phase of training.")
    return [
        ("title", "Gradient Dynamics in Very Deep Residual Networks\nA Reproducibility Study"),
        ("twocol_paras", {"left": [], "right": [], "col_h": 0, "_intros": (left_intro, right_intro)}),
    ]


def make_page(w_px, h_px, blocks, zh=False):
    return render_page_blocks(h_px, w_px, blocks)


def build_two_col_page():
    """Custom two-column rendering with wrapped text in both columns."""
    w_px, h_px = int(A4_W_IN * DPI), int(A4_H_IN * DPI)
    img = Image.new("L", (w_px, h_px), 255)
    draw = ImageDraw.Draw(img)
    margin = 170
    gt = []
    f_title = font(EN_BOLD, 42)
    t1 = "Gradient Dynamics in Very Deep Residual Networks"
    t2 = "A Reproducibility Study"
    for t in (t1, t2):
        w = draw.textlength(t, font=f_title)
        draw.text(((w_px - w) / 2, margin), t, font=f_title, fill=20)
        margin += 54
        gt.append(t)
    margin += 30
    left_intro = ("Deep residual networks have become the dominant architecture for a broad range of perception tasks. In this work we revisit the optimization geometry of residual connections and derive improved scaling rules for very deep networks. We train networks of depth 50 to 200 layers on image classification benchmarks and observe consistent gains across all scales tested. The analysis relies on a variance preservation condition that we prove in Section 3. Empirically, the benefit concentrates in the late phase of training when the identity path dominates gradient flow, and the effect strengthens as depth increases beyond one hundred layers.")
    right_intro = ("Residual learning reframes the mapping problem: instead of fitting a desired underlying mapping H(x) directly, the stacked layers fit a residual F(x) = H(x) - x. This reformulation eases optimization and enables training of networks that are substantially deeper than previously possible. Prior work has studied normalization and initialization; here we focus on the interaction between residual branches and learning-rate schedules. We also quantify the variance of gradient updates across depth and relate it to the effective condition number of the Jacobian.")
    f_body = font(EN_FONT, 24)
    gap = 90
    col_w = (w_px - 2 * margin - gap) / 2
    left_lines = wrap(draw, left_intro, f_body, col_w)
    right_lines = wrap(draw, right_intro, f_body, col_w)
    y = margin
    for ln in left_lines:
        draw.text((margin, y), ln, font=f_body, fill=30)
        gt.append(ln)
        y += 36
    y = margin
    x1 = margin + col_w + gap
    for ln in right_lines:
        draw.text((x1, y), ln, font=f_body, fill=30)
        gt.append(ln)
        y += 36
    return img, gt


def main():
    os.makedirs(GT_DIR, exist_ok=True)
    w_px, h_px = int(A4_W_IN * DPI), int(A4_H_IN * DPI)

    def write_gt(name, lines):
        with open(os.path.join(GT_DIR, name + ".txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

    # ---- scanned_clear ----
    jobs = [
        ("paper_en_basic", EN_PAPER, {}),
        ("paper_zh_cn", ZH_PAPER, {}),
        ("paper_en_formulas", EN_FORMULA_PAPER, {}),
        ("report_en_tables", EN_TABLE_PAPER, {}),
    ]
    for name, blocks, _ in jobs:
        img, gt = render_page_blocks(h_px, w_px, blocks)
        save_image_pdf(os.path.join(CORPUS, "scanned_clear", name + ".pdf"), [img], dpi=DPI)
        write_gt(name, gt)
        print(f"[clear] {name}: {len(gt)} gt lines")

    img2, gt2 = build_two_col_page()
    save_image_pdf(os.path.join(CORPUS, "scanned_clear", "paper_en_twocol.pdf"), [img2], dpi=DPI)
    write_gt("paper_en_twocol", gt2)
    print(f"[clear] paper_en_twocol: {len(gt2)} gt lines")

    # ---- scanned_lowq (degraded variants) ----
    lowq = [
        ("scan_lowq_rotated", EN_PAPER, dict(rotate_deg=2.8, scale=0.45, noise=0.004, jpeg_q=30)),
        ("scan_lowq_blurry", ZH_PAPER, dict(rotate_deg=1.2, scale=0.35, noise=0.012, jpeg_q=22)),
        ("scan_lowq_skew_faint", EN_FORMULA_PAPER, dict(rotate_deg=-3.5, scale=0.4, noise=0.006, jpeg_q=26)),
    ]
    for name, blocks, deg in lowq:
        img, gt = render_page_blocks(h_px, w_px, blocks)
        img = degrade(img, **deg)
        save_image_pdf(os.path.join(CORPUS, "scanned_lowq", name + ".pdf"), [img], dpi=int(DPI * deg["scale"] / 0.35 * 0.35) or 110)
        write_gt(name, gt)
        print(f"[lowq] {name}")

    # ---- hybrid_scan_text ----
    hyb = [
        ("hybrid_en_paper", EN_PAPER),
        ("hybrid_zh_paper", ZH_PAPER),
        ("hybrid_en_tables", EN_TABLE_PAPER),
    ]
    for name, blocks in hyb:
        img, gt = render_page_blocks(h_px, w_px, blocks)
        # text layer only supports latin-1; Chinese samples use a subset of lines
        if "zh" in name:
            tl_lines = [ln for ln in gt if all(ord(ch) < 0x80 for ch in ln)][:20]
        else:
            tl_lines = [ln for ln in gt if len(ln) > 3][:40]
        hybrid_pdf(os.path.join(CORPUS, "hybrid_scan_text", name + ".pdf"), img, tl_lines, dpi=DPI)
        write_gt(name, gt)
        print(f"[hybrid] {name}: {len(tl_lines)} text-layer lines")


if __name__ == "__main__":
    main()
