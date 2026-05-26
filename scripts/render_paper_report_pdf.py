#!/usr/bin/env python3
"""Render the Cola adaptive halt Markdown report to PDF."""

from __future__ import annotations

import html
import os
import re
import subprocess
import argparse
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.font_manager import FontProperties


ROOT = Path("/data1/luyifei/drla")
OUT_DIR = ROOT / "outputs" / "paper_report_20260525"
FIG_DIR = OUT_DIR / "figures"
BUILD_DIR = OUT_DIR / "pdf_build"
FONT_DIR = BUILD_DIR / "fonts"

LANG = "en"
REPORT_MD = ROOT / "docs" / "cola_adaptive_halt_paper_report.md"
PREPROCESSED_MD = BUILD_DIR / "cola_adaptive_halt_paper_report.rendered.md"
CSS_PATH = BUILD_DIR / "paper_report.css"
DATA_FLOW_PNG = FIG_DIR / "fig_data_flow.png"
PDF_PATH = OUT_DIR / "cola_adaptive_halt_paper_report.pdf"
FONT_REGULAR = FONT_DIR / "NotoSansCJKsc-Regular.otf"
FONT_BOLD = FONT_DIR / "NotoSansCJKsc-Bold.otf"

FONT_URLS = {
    FONT_REGULAR: "https://github.com/notofonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf",
    FONT_BOLD: "https://github.com/notofonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Bold.otf",
}


DATA_FLOW_NODES = [
    ("Cola DiT rollout block b", 335, 20, 230, 54),
    ("Per-block trace row", 335, 100, 230, 54),
    ("Readiness features\\nprocess/probe/stability\\nlatent stats", 95, 205, 300, 78),
    ("Continuation-risk features\\nprediction-change\\nanswer shape", 505, 205, 300, 78),
    ("Readiness MLP\\np_ready", 145, 335, 200, 68),
    ("Risk MLP\\np_change", 555, 335, 200, 68),
    ("Sequential halt policy", 335, 465, 230, 62),
    ("Guards\\nriskcap04 + content\\nfragment + choice", 310, 565, 280, 78),
    ("Halt decision\\nhalt now\\nor stability/final", 295, 675, 310, 78),
]

DATA_FLOW_NODES_ZH = [
    ("Cola DiT 生成 block b", 335, 20, 230, 54),
    ("每 block trace row", 335, 100, 230, 54),
    ("Readiness 特征\\nprocess/probe/stability\\nlatent stats", 95, 205, 300, 78),
    ("Continuation-risk 特征\\nprediction-change\\nanswer shape", 505, 205, 300, 78),
    ("Readiness MLP\\np_ready", 145, 335, 200, 68),
    ("Risk MLP\\np_change", 555, 335, 200, 68),
    ("顺序早停策略", 335, 465, 230, 62),
    ("安全 guards\\nriskcap04 + content\\nfragment + choice", 310, 565, 280, 78),
    ("早停决策\\n当前 block 停止\\n或退回 stability/final", 295, 675, 310, 78),
]

DATA_FLOW_EDGES = [
    (0, 1),
    (1, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
    (5, 6),
    (6, 7),
    (7, 8),
]


def configure_paths(lang: str, report_md: Path | None, out_pdf: Path | None) -> None:
    global LANG, REPORT_MD, PREPROCESSED_MD, CSS_PATH, DATA_FLOW_PNG, PDF_PATH
    LANG = lang
    suffix = "_zh" if lang == "zh" else ""
    default_report = ROOT / "docs" / f"cola_adaptive_halt_paper_report{suffix}.md"
    default_pdf = OUT_DIR / f"cola_adaptive_halt_paper_report{suffix}.pdf"
    REPORT_MD = (report_md or default_report).resolve()
    PDF_PATH = (out_pdf or default_pdf).resolve()
    PREPROCESSED_MD = BUILD_DIR / f"{REPORT_MD.stem}.rendered.md"
    CSS_PATH = BUILD_DIR / f"paper_report{suffix}.css"
    DATA_FLOW_PNG = FIG_DIR / f"fig_data_flow{suffix}.png"


def ensure_cjk_fonts() -> None:
    if LANG != "zh":
        return
    FONT_DIR.mkdir(parents=True, exist_ok=True)
    for font_path, url in FONT_URLS.items():
        if font_path.exists() and font_path.stat().st_size > 1_000_000:
            continue
        print(f"Downloading Chinese font: {font_path.name}")
        urllib.request.urlretrieve(url, font_path)


def write_data_flow_png() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    nodes = DATA_FLOW_NODES_ZH if LANG == "zh" else DATA_FLOW_NODES
    font_regular = FontProperties(fname=str(FONT_REGULAR)) if LANG == "zh" and FONT_REGULAR.exists() else None
    font_bold = FontProperties(fname=str(FONT_BOLD)) if LANG == "zh" and FONT_BOLD.exists() else None
    fig, ax = plt.subplots(figsize=(6.2, 5.4), dpi=220)
    ax.set_xlim(0, 900)
    ax.set_ylim(780, 0)
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0, 0),
            900,
            780,
            boxstyle="round,pad=0.0,rounding_size=12",
            linewidth=0,
            facecolor="#FAFAFA",
        )
    )

    def anchor(node_idx: int, side: str) -> tuple[float, float]:
        _, x, y, w, h = nodes[node_idx]
        if side == "bottom":
            return x + w / 2, y + h
        if side == "top":
            return x + w / 2, y
        if side == "right":
            return x + w, y + h / 2
        if side == "left":
            return x, y + h / 2
        raise ValueError(side)

    for src_idx, dst_idx in DATA_FLOW_EDGES:
        start = anchor(src_idx, "bottom")
        end = anchor(dst_idx, "top")
        connectionstyle = "arc3,rad=0.0"
        if (src_idx, dst_idx) in {(1, 2), (4, 6)}:
            connectionstyle = "arc3,rad=0.10"
        elif (src_idx, dst_idx) in {(1, 3), (5, 6)}:
            connectionstyle = "arc3,rad=-0.10"
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.7,
                color="#4A5568",
                connectionstyle=connectionstyle,
                shrinkA=4,
                shrinkB=4,
            )
        )

    palette = ["#E8EDF2", "#E8F2EE", "#FFF6DB", "#FDECE7", "#E8EDF2"]
    for idx, (label, x, y, w, h) in enumerate(nodes):
        fill = palette[idx % len(palette)]
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.0,rounding_size=9",
                linewidth=1.2,
                edgecolor="#2D3748",
                facecolor=fill,
            )
        )
        parts = label.split("\\n")
        line_step = 17
        y0 = y + h / 2 - (len(parts) - 1) * (line_step / 2)
        for line_idx, part in enumerate(parts):
            size = 9.4 if line_idx == 0 else 7.5
            weight = "700" if line_idx == 0 else "400"
            font_prop = font_bold if line_idx == 0 and font_bold else font_regular
            ax.text(
                x + w / 2,
                y0 + line_idx * line_step,
                part,
                ha="center",
                va="center",
                fontsize=size,
                fontweight=weight,
                color="#1A202C",
                family="DejaVu Sans" if font_prop is None else None,
                fontproperties=font_prop,
            )
    fig.savefig(DATA_FLOW_PNG, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def image_tag(alt: str, path_text: str) -> str:
    if path_text.startswith("/"):
        source = Path(path_text)
    elif path_text.startswith(("outputs/", "docs/", "scripts/")):
        source = (ROOT / path_text).resolve()
    else:
        source = (REPORT_MD.parent / path_text).resolve()
    safe_alt = alt.replace("[", "(").replace("]", ")")
    relative_source = source.relative_to(ROOT)
    return f"![{safe_alt}]({relative_source.as_posix()})"


def preprocess_markdown() -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    text = REPORT_MD.read_text(encoding="utf-8")

    data_flow_tag = image_tag("Data flow", str(DATA_FLOW_PNG))
    text = re.sub(r"```mermaid\n.*?\n```", data_flow_tag, text, flags=re.DOTALL)

    def replace_image(match: re.Match[str]) -> str:
        return image_tag(match.group(1), match.group(2))

    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", replace_image, text)
    PREPROCESSED_MD.write_text(text, encoding="utf-8")


def write_css() -> None:
    font_face = ""
    body_font = '"DejaVu Serif", "Times New Roman", serif'
    heading_font = '"DejaVu Sans", Arial, sans-serif'
    mono_font = '"DejaVu Sans Mono", Menlo, monospace'
    if LANG == "zh" and FONT_REGULAR.exists() and FONT_BOLD.exists():
        font_face = f"""
@font-face {{
  font-family: "Noto Sans CJK SC";
  src: url("{FONT_REGULAR.as_uri()}") format("opentype");
  font-weight: 400;
}}
@font-face {{
  font-family: "Noto Sans CJK SC";
  src: url("{FONT_BOLD.as_uri()}") format("opentype");
  font-weight: 700;
}}
""".strip()
        body_font = '"Noto Sans CJK SC", "DejaVu Sans", sans-serif'
        heading_font = '"Noto Sans CJK SC", "DejaVu Sans", sans-serif'
        mono_font = '"DejaVu Sans Mono", "Noto Sans CJK SC", monospace'
    css = """
__FONT_FACE__
body {
  color: #111827;
  font-family: __BODY_FONT__;
  font-size: 10.5px;
  line-height: 1.42;
}
h1, h2, h3 {
  color: #111827;
  font-family: __HEADING_FONT__;
  page-break-after: avoid;
}
h1 {
  font-size: 25px;
  margin-top: 0;
  margin-bottom: 12px;
}
h2 {
  font-size: 16px;
  margin-top: 20px;
  border-bottom: 1px solid #E5E7EB;
  padding-bottom: 4px;
}
h3 {
  font-size: 12.5px;
  margin-top: 14px;
}
p, li {
  widows: 2;
  orphans: 2;
}
code {
  font-family: __MONO_FONT__;
  font-size: 9px;
  background: #F3F4F6;
  padding: 1px 3px;
  border-radius: 3px;
}
li code {
  font-size: 7.2px;
  white-space: nowrap;
}
pre {
  background: #F8FAFC;
  border: 1px solid #E5E7EB;
  border-radius: 5px;
  padding: 8px;
  white-space: pre-wrap;
  page-break-inside: avoid;
}
pre code {
  background: transparent;
  padding: 0;
}
table {
  border-collapse: collapse;
  width: 100%;
  table-layout: fixed;
  margin: 8px 0 14px;
  page-break-inside: avoid;
}
th, td {
  border: 1px solid #D1D5DB;
  padding: 4px 5px;
  vertical-align: top;
  word-wrap: break-word;
}
th {
  background: #F3F4F6;
  font-family: __HEADING_FONT__;
  font-weight: 700;
}
img {
  display: block;
  max-width: 96%;
  max-height: 560px;
  margin: 10px auto 14px;
  page-break-inside: avoid;
}
blockquote {
  border-left: 3px solid #D1D5DB;
  margin-left: 0;
  padding-left: 10px;
  color: #374151;
}
a {
  color: #0F766E;
  text-decoration: none;
}
""".strip()
    css = css.replace("__FONT_FACE__", font_face)
    css = css.replace("__BODY_FONT__", body_font)
    css = css.replace("__HEADING_FONT__", heading_font)
    css = css.replace("__MONO_FONT__", mono_font)
    CSS_PATH.write_text(css, encoding="utf-8")


def render_pdf() -> None:
    command = [
        "npx",
        "-y",
        "markdown-pdf",
        str(PREPROCESSED_MD),
        "--cwd",
        str(ROOT),
        "--css-path",
        str(CSS_PATH),
        "--paper-format",
        "A4",
        "--paper-border",
        "16mm",
        "--out",
        str(PDF_PATH),
    ]
    env = os.environ.copy()
    env["OPENSSL_CONF"] = "/dev/null"
    subprocess.run(command, check=True, env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", choices=["en", "zh"], default="en")
    parser.add_argument("--report-md", type=Path, default=None)
    parser.add_argument("--out-pdf", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_paths(args.lang, args.report_md, args.out_pdf)
    ensure_cjk_fonts()
    write_data_flow_png()
    preprocess_markdown()
    write_css()
    render_pdf()
    print(PDF_PATH)


if __name__ == "__main__":
    main()
