"""Render the project summary PDF from reports/project_summary/project_summary.md.

The Markdown file is the single source of the summary's text. This script draws
two figures that combine earlier results (the project pipeline and the data and
evaluation timeline), converts the Markdown subset the summary uses (headings,
paragraphs, lists, pipe tables, images, quotes, code blocks, and inline bold,
italic, code and links) into a styled PDF with reportlab, and points relative
repository links at their GitHub pages. It trains nothing and reads only saved
reports and tables.
"""

from __future__ import annotations

import json
import os
import posixpath
import re
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

SOURCE = ROOT / "reports/project_summary/project_summary.md"
OUTPUT = ROOT / "reports/project_summary/project_summary.pdf"
FIGURES = ROOT / "reports/project_summary/figures"
REPOSITORY = "https://github.com/Erdos-Projects/fall-2026-AI-financial-models/blob/main/"
INK, INK_2, MUTED, RULE, PANEL, ACCENT = "#14213d", "#4a5568", "#898781", "#d9d8d2", "#f3f2ee", "#2a78d6"
STAGE_FILL = {"train": "#e6eefb", "valid": "#fbeee3", "test": "#e4f3ec"}
WINDOW_FILL = {"fit": "#cde2fb", "tune": "#6da7ec", "score": "#1c5cab"}


def register_fonts() -> None:
    fonts = Path(matplotlib.get_data_path()) / "fonts/ttf"
    for name, file in [("Body", "DejaVuSans.ttf"), ("Body-Bold", "DejaVuSans-Bold.ttf"),
                       ("Body-Italic", "DejaVuSans-Oblique.ttf"), ("Body-BoldItalic", "DejaVuSans-BoldOblique.ttf"),
                       ("Mono", "DejaVuSansMono.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(fonts / file)))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic",
                                  boldItalic="Body-BoldItalic")


def styles() -> dict[str, ParagraphStyle]:
    base = ParagraphStyle("body", fontName="Body", fontSize=9.6, leading=14.2, textColor=INK_2, spaceAfter=6.5)
    return {
        "title": ParagraphStyle("title", parent=base, fontName="Body-Bold", fontSize=20, leading=25,
                                textColor=INK, spaceAfter=6),
        "subtitle": ParagraphStyle("subtitle", parent=base, fontSize=10.5, leading=15, textColor=INK_2,
                                   spaceAfter=2),
        "h2": ParagraphStyle("h2", parent=base, fontName="Body-Bold", fontSize=14, leading=18, textColor=INK,
                             spaceBefore=14, spaceAfter=7),
        "h3": ParagraphStyle("h3", parent=base, fontName="Body-Bold", fontSize=11, leading=15, textColor=INK,
                             spaceBefore=9, spaceAfter=4),
        "body": base,
        "bullet": ParagraphStyle("bullet", parent=base, leftIndent=13, bulletIndent=2, spaceAfter=3.2),
        "caption": ParagraphStyle("caption", parent=base, fontName="Body-Italic", fontSize=8.3, leading=11.5,
                                  textColor=MUTED, alignment=TA_CENTER, spaceBefore=3, spaceAfter=10),
        "cell": ParagraphStyle("cell", parent=base, fontSize=7.9, leading=10.2, spaceAfter=0),
        "head": ParagraphStyle("head", parent=base, fontName="Body-Bold", fontSize=7.9, leading=10.2,
                               textColor=INK, spaceAfter=0),
        "quote": ParagraphStyle("quote", parent=base, fontSize=9.6, leading=14.2, textColor=INK, spaceAfter=0),
        "code": ParagraphStyle("code", parent=base, fontName="Mono", fontSize=7.4, leading=10, textColor=INK),
        "toc0": ParagraphStyle("toc0", parent=base, fontSize=9.6, leading=13, leftIndent=0, spaceAfter=1),
    }


# --- figures -----------------------------------------------------------------------------------------------

def pipeline_figure(path: Path) -> None:
    stages = [("Data audit", "5,682 files → 3,137 unique;\n17.6M one-minute bars"),
              ("Preprocessing", "11.5M examples, 15 features;\nchronological 70/15/15"),
              ("Baselines", "persistence, ridge,\n3 MLPs × 3 seeds"),
              ("Architecture study", "8 MLPs × 3 seeds;\ncompact MLP frozen"),
              ("Diagnosis", "MAE worse than persistence;\nloss–metric mismatch"),
              ("Improvement search", "5 methods × 12 proposals\non 3 forward folds"),
              ("Confirmation", "best of each search,\n3 seeds, validation dates"),
              ("Final test", "once, on reserved\nJun–Dec 2020 dates")]
    fig, ax = plt.subplots(figsize=(10.5, 2.9))
    ax.set_xlim(0, 4)
    ax.set_ylim(-0.16, 2)
    ax.axis("off")
    for index, (name, detail) in enumerate(stages):
        row, column = divmod(index, 4)
        x, y = column + 0.04, 1.08 - row * 1.02
        llm = name in ("Improvement search", "Confirmation", "Final test")
        ax.add_patch(FancyBboxPatch((x, y), 0.9, 0.78, boxstyle="round,pad=0.0,rounding_size=0.06",
                                    facecolor="#e6eefb" if llm else "#f3f2ee", edgecolor="#9fbde8" if llm
                                    else "#d9d8d2", linewidth=1))
        ax.text(x + 0.45, y + 0.57, f"{index + 1}. {name}", ha="center", va="center", fontsize=8.8,
                fontweight="bold", color=INK)
        ax.text(x + 0.45, y + 0.25, detail, ha="center", va="center", fontsize=7.6, color=INK_2, linespacing=1.3)
        if column < 3:
            ax.annotate("", (x + 1.0, y + 0.39), (x + 0.92, y + 0.39),
                        arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 1})
    ax.annotate("", (0.49, 0.87), (3.49, 1.07), arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 1,
                                                          "connectionstyle": "arc3,rad=0.0"})
    ax.text(3.98, -0.1, "blue boxes: the LLM-guided improvement experiment", ha="right", va="center",
            fontsize=7.5, color=MUTED)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)


def timeline_figure(path: Path) -> None:
    monthly = pd.read_csv(ROOT / "reports/nifty_eda/tables/monthly_activity.csv")
    monthly["month"] = pd.to_datetime(monthly["month"] + "-01")
    protocol = json.loads((ROOT / "reports/nifty_llm_search/protocol.json").read_text())
    rows = [(f"Search fold {number}", window) for number, window in enumerate(protocol["search"]["windows"], 1)]
    for label, plan in [("Confirmation", "validation"), ("Final test", "test")]:
        result = next(iter(sorted((ROOT / f"reports/nifty_llm_search/results/{plan}/seed42").glob("*.json"))), None)
        if result is not None:
            rows.append((label, json.loads(result.read_text())["windows"][0]))
    splits = {"train": ("2016-12-14", "2019-10-17"), "valid": ("2019-10-18", "2020-05-29"),
              "test": ("2020-06-01", "2020-12-31")}
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10.5, 4.5), sharex=True,
                                      gridspec_kw={"height_ratios": [1.15, 1.35], "hspace": 0.08})
    for name, (start, end) in splits.items():
        for ax in (top, bottom):
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1), color=STAGE_FILL[name],
                       zorder=0, linewidth=0)
    top.bar(monthly["month"] + pd.Timedelta(days=14), monthly["rows"] / 1e6, width=24, color="#7e98b8", zorder=2)
    top.set_ylabel("raw bars per\nmonth (millions)", fontsize=8.5, color=INK_2)
    labels = {"train": "Training\n700 dates · 7.07M examples", "valid": "Validation\n150 dates · 2.01M",
              "test": "Test\n150 dates · 2.39M"}
    for name, (start, end) in splits.items():
        middle = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
        top.text(middle, top.get_ylim()[1] * 1.02, labels[name], ha="center", va="bottom", fontsize=7.8,
                 color=INK, linespacing=1.25)
    for index, (label, window) in enumerate(rows):
        y = len(rows) - 1 - index
        for part in ["fit", "tune", "score"]:
            first, last = pd.Timestamp(window[part]["first_date"]), pd.Timestamp(window[part]["last_date"])
            bottom.barh(y, (last - first).days + 1, left=first, height=0.56, color=WINDOW_FILL[part], zorder=2,
                        edgecolor="white", linewidth=0.6)
        bottom.text(pd.Timestamp("2016-11-25"), y, label, ha="right", va="center", fontsize=8.5, color=INK)
    bottom.set_yticks([])
    bottom.set_ylim(-0.6, len(rows) - 0.4)
    handles = [plt.Rectangle((0, 0), 1, 1, color=WINDOW_FILL[part]) for part in ["fit", "tune", "score"]]
    bottom.legend(handles, ["fit (training)", "tune (early stopping, calibration)", "score (reported)"],
                  loc="lower left", ncols=3, fontsize=8, frameon=False, bbox_to_anchor=(0, -0.42))
    bottom.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    for ax in (top, bottom):
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(RULE)
        ax.tick_params(colors=MUTED, labelsize=8)
    bottom.spines["left"].set_visible(False)
    bottom.set_xlim(pd.Timestamp("2016-12-01"), pd.Timestamp("2021-01-15"))
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)


def slide_search_figure(path: Path) -> None:
    """The search curves as one large-type panel for a 1920×1080 slide."""
    from build_nifty_llm_search_report import read_jsonl, summarize_search

    report = ROOT / "reports/nifty_llm_search"
    base = json.loads((report / "baseline.json").read_text())["objective"]
    series = [("random", "Random search", "#a3a196"), ("bayes", "Bayesian optimization", "#e8833a"),
              ("llm_batch", "LLM, no feedback", "#8fb8ef"), ("llm_feedback", "LLM + feedback", "#1f5fbf")]
    fig, ax = plt.subplots(figsize=(12, 6.4))
    fig.patch.set_facecolor("#f7f6f2")
    ax.set_facecolor("#f7f6f2")
    ends = []
    for arm, label, color in series:
        search = summarize_search(arm, 1, read_jsonl(report / f"trials/{arm}_r1.jsonl"), 12, base)
        y = [0.0, *search["best_so_far"]]
        ax.plot(range(13), y, drawstyle="steps-post", color=color, linewidth=4, solid_capstyle="round")
        ends.append((y[-1], label, color))
    placed = []
    for value, label, color in sorted(ends):
        position = max(value, placed[-1] + 0.05) if placed else value
        placed.append(position)
        ax.plot([12], [value], "o", color=color, markersize=11, markeredgecolor="#f7f6f2", markeredgewidth=2)
        # Each label carries its own color key, because two lines end almost at the same value.
        ax.plot([12.45], [position], "s", color=color, markersize=13, clip_on=False)
        ax.annotate(label, (12.45, position), xytext=(14, 0), textcoords="offset points", va="center",
                    fontsize=19, color="#14213d", annotation_clip=False)
    ax.axhline(0, color="#c3c2b7", linewidth=1.5)
    ax.set_xlim(0, 12.2)
    ax.set_xticks(range(0, 13, 2))
    ax.set_xlabel("proposals evaluated", fontsize=19, color="#4a5568", labelpad=10)
    ax.set_ylabel("best error found vs baseline\nconfig (%), lower is better", fontsize=19, color="#4a5568",
                  labelpad=10)
    ax.tick_params(labelsize=17, colors="#4a5568", length=0)
    ax.grid(axis="y", color="#e1e0d9", linewidth=1)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.25, facecolor=fig.get_facecolor())
    plt.close(fig)


# --- Markdown subset → reportlab flowables ------------------------------------------------------------------

def link_target(target: str) -> str:
    if re.match(r"^[a-z]+://", target):
        return target
    path = posixpath.normpath(posixpath.join(SOURCE.parent.relative_to(ROOT).as_posix(), target))
    return REPOSITORY + path


def inline(text: str) -> str:
    """Convert inline Markdown to reportlab paragraph markup; code spans and links are protected first."""
    saved: list[str] = []

    def keep(markup: str) -> str:
        saved.append(markup)
        return f"\x00{len(saved) - 1}\x00"

    text = re.sub(r"`([^`]+)`", lambda m: keep(f'<font face="Mono">{escape(m.group(1))}</font>'), text)
    text = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", lambda m: keep(
        f'<link href="{escape(link_target(m.group(2)))}" color="{ACCENT}">{inline(m.group(1))}</link>'), text)
    text = escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![*\w])\*(?![\s*])(.+?)(?<![\s*])\*(?![*\w])", r"<i>\1</i>", text)
    return re.sub(r"\x00(\d+)\x00", lambda m: saved[int(m.group(1))], text)


def plain(text: str) -> str:
    return re.sub(r"[*`]|\[([^\]]+)\]\([^)]+\)", r"\1", text)


def column_widths(rows: list[list[str]], width: float, font_size: float) -> list[float]:
    """Every column gets room for its longest word; spare width follows each column's longest cell."""
    glyph = 0.62 * font_size
    columns = list(zip(*rows, strict=True))
    minimum = [glyph * max(len(word) for cell in column for word in plain(cell).split() or [""]) + 9
               for column in columns]
    longest = [max(len(plain(cell)) for cell in column) for column in columns]
    desired = [max(glyph * min(length, 60) + 9, low) for length, low in zip(longest, minimum, strict=True)]
    if sum(desired) <= width:
        return [value * width / sum(desired) for value in desired]
    spare = max(width - sum(minimum), 0)
    extra = [high - low for high, low in zip(desired, minimum, strict=True)]
    return [low + spare * part / sum(extra) for low, part in zip(minimum, extra, strict=True)]


def parse(markdown: str) -> list[tuple]:
    lines, blocks, index = markdown.splitlines(), [], 0
    starts = re.compile(r"^(#{1,3} |!\[|\||> |```|- |\d+\. |<!--)")
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
        elif line.startswith("```"):
            end = next(j for j in range(index + 1, len(lines)) if lines[j].startswith("```"))
            blocks.append(("code", "\n".join(lines[index + 1:end])))
            index = end + 1
        elif line.strip() in ("<!-- pagebreak -->", "<!-- toc -->"):
            blocks.append((line.strip()[5:-4].strip(),))
            index += 1
        elif line.startswith("<!--"):
            index += 1
        elif line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            blocks.append(("heading", level, line[level:].strip()))
            index += 1
        elif line.startswith("!["):
            match = re.match(r"^!\[(.*)\]\((.+?)\)(?:\{width=(\d+)%\})?\s*$", line)
            blocks.append(("image", match.group(1), match.group(2), int(match.group(3) or 100)))
            index += 1
        elif line.startswith("|"):
            rows = []
            while index < len(lines) and lines[index].startswith("|"):
                cells = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
                    rows.append(cells)
                index += 1
            blocks.append(("table", rows))
        elif line.startswith("> "):
            quote = []
            while index < len(lines) and lines[index].startswith(">"):
                quote.append(lines[index][1:].strip())
                index += 1
            blocks.append(("quote", " ".join(quote)))
        elif re.match(r"^(- |\d+\. )", line):
            ordered, items = bool(re.match(r"^\d+\. ", line)), []
            while index < len(lines) and (re.match(r"^(- |\d+\. )", lines[index])
                                          or (lines[index].startswith("  ") and items)):
                if re.match(r"^(- |\d+\. )", lines[index]):
                    items.append(re.sub(r"^(- |\d+\. )", "", lines[index]).strip())
                else:
                    items[-1] += " " + lines[index].strip()
                index += 1
            blocks.append(("list", ordered, items))
        else:
            paragraph = []
            while index < len(lines) and lines[index].strip() and not starts.match(lines[index]):
                paragraph.append(lines[index].strip())
                index += 1
            blocks.append(("para", " ".join(paragraph)))
    return blocks


class SummaryDocument(SimpleDocTemplate):
    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name == "h2":
            text = flowable.getPlainText()
            # Keys come from the heading text, so they are stable across multiBuild passes.
            key = "section-" + re.sub(r"\W+", "-", text).strip("-").lower()
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=0)
            self.notify("TOCEntry", (0, text, self.page, key))


def flowables(blocks: list[tuple], style: dict, width: float, height: float) -> list:
    story, title_done = [], False
    for block in blocks:
        kind = block[0]
        if kind == "heading":
            level, text = block[1], block[2]
            if level == 1 and not title_done:
                story.append(Paragraph(inline(text), style["title"]))
                title_done = True
            else:
                story.append(Paragraph(inline(text), style["h2" if level <= 2 else "h3"]))
        elif kind == "para":
            # Paragraphs between the title and the first heading form the subtitle block.
            after_title = story and isinstance(story[-1], Paragraph) and story[-1].style.name in ("title", "subtitle")
            story.append(Paragraph(inline(block[1]), style["subtitle" if after_title else "body"]))
        elif kind == "list":
            ordered, items = block[1], block[2]
            for number, item in enumerate(items, 1):
                story.append(Paragraph(inline(item), style["bullet"], bulletText=f"{number}." if ordered else "•"))
            story.append(Spacer(1, 3))
        elif kind == "quote":
            box = Table([[Paragraph(inline(block[1]), style["quote"])]], colWidths=[width])
            box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(PANEL)),
                                     ("LINEBEFORE", (0, 0), (0, -1), 2.2, colors.HexColor(ACCENT)),
                                     ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                                     ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
            story += [box, Spacer(1, 8)]
        elif kind == "code":
            box = Table([[Preformatted(block[1], style["code"])]], colWidths=[width])
            box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(PANEL)),
                                     ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 6),
                                     ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
            story += [box, Spacer(1, 8)]
        elif kind == "table":
            rows = block[1]
            widths = column_widths(rows, width, style["cell"].fontSize)
            data = [[Paragraph(inline(cell), style["head" if number == 0 else "cell"]) for cell in row]
                    for number, row in enumerate(rows)]
            table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(PANEL)),
                                       ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor(MUTED)),
                                       ("LINEBELOW", (0, 1), (-1, -1), 0.35, colors.HexColor(RULE)),
                                       ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                       ("TOPPADDING", (0, 0), (-1, -1), 3.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
                                       ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]))
            story += [table, Spacer(1, 9)]
        elif kind == "image":
            caption, target, percent = block[1], block[2], block[3]
            path = (SOURCE.parent / target).resolve()
            image = Image(str(path))
            scale = min(width * percent / 100 / image.imageWidth, height * 0.42 / image.imageHeight)
            image.drawWidth, image.drawHeight = image.imageWidth * scale, image.imageHeight * scale
            story.append(KeepTogether([image, Paragraph(inline(caption), style["caption"])]))
        elif kind == "pagebreak":
            story.append(PageBreak())
        elif kind == "toc":
            toc = TableOfContents()
            toc.levelStyles = [style["toc0"]]
            toc.dotsMinLevel = 0
            story += [Paragraph("Contents", style["h3"]), toc]
    return story


def footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setFont("Body", 7.5)
    canvas.setFillColor(colors.HexColor(MUTED))
    canvas.drawString(document.leftMargin, 1.2 * cm, "LLM-guided model improvement · project summary")
    canvas.drawRightString(document.pagesize[0] - document.rightMargin, 1.2 * cm, str(document.page))
    canvas.restoreState()


def build() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    pipeline_figure(FIGURES / "pipeline.png")
    timeline_figure(FIGURES / "timeline.png")
    slide_search_figure(FIGURES / "slide_search.png")
    register_fonts()
    style = styles()
    document = SummaryDocument(str(OUTPUT), pagesize=letter, leftMargin=2.1 * cm, rightMargin=2.1 * cm,
                               topMargin=1.9 * cm, bottomMargin=2.1 * cm,
                               title="LLM-guided model improvement for next-minute Nifty option returns",
                               author="Fall 2026 AI Financial Models team", subject="Project summary",
                               creator="scripts/build_project_summary.py")
    story = flowables(parse(SOURCE.read_text()), style, document.width, document.height)
    document.multiBuild(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Wrote {OUTPUT.relative_to(ROOT)} ({date.today().isoformat()})")


if __name__ == "__main__":
    build()
