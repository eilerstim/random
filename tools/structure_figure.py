#!/usr/bin/env python3
"""Render a "model file structure" figure as SVG from a small JSON spec.

Every framework deep dive in this repo gets one of these figures, all in the
same visual style (dark navy panel, green container border, gray boxes, red
text for the component that can execute code). Keeping the layout in one
place means a new framework only needs a new spec file.

Usage:
    python tools/structure_figure.py SPEC.json -o OUT.svg

Spec format (see pytorch/figures/pytorch_structure.json for a full example):

    {
      "title":   "PyTorch Model (.pt / .pth) - Zip Archive",
      "caption": "...",                                     # optional italic line under the panel (off by default)
      "container_label": "model/  (archive prefix)",       # optional, top-left inside container
      "columns": [
        {"label": "data.pkl", "width": 500,                # label draws a section box around the stack
         "boxes": [
            {"lines": ["main line", "secondary line"]},
            {"lines": ["dangerous thing"], "danger": true}  # red text
         ]},
        {"width": 320, "boxes": [{"lines": ["Tensor storages", "data/0 ... data/N"]}]}
      ],                                                    # a column without label = one box stretched
      "footer": [ {"lines": ["version"]}, ... ]             # optional row of small boxes, widths follow the text
    }

Only the standard library is used. Text width is estimated (SVG has no
wrapping), and a warning is printed when a line probably overflows its box.
"""

import argparse
import json
import sys
from xml.sax.saxutils import escape

# --- style -------------------------------------------------------------------
PAGE_BG = "#f2f2f2"
PANEL_BG = "#1f2a44"
CONTAINER_BG = "#0e1829"
CONTAINER_STROKE = "#3fa34d"
BOX_STROKE = "#8f99ab"
TEXT = "#dfe4ee"
TEXT_DIM = "#a9b3c6"
DANGER = "#ff5a5f"
CAPTION = "#333333"
FONT = "'Open Sans', 'Noto Sans', 'DejaVu Sans', 'Segoe UI', Helvetica, Arial, sans-serif"

CANVAS_W = 1280
PANEL_MARGIN = 64
PANEL_TOP = 40
CONTAINER_MARGIN = 80
CONTAINER_PAD = 48
SECTION_PAD = 30
COLUMN_GAP = 48
BOX_MAIN_H = 100  # box height with one main + one secondary line
BOX_GAP = 44
FOOTER_H = 84
FOOTER_GAP = 24
MAIN_FS = 21
SUB_FS = 15
FOOTER_MAIN_FS = 18
FOOTER_SUB_FS = 14
FOOTER_TEXT_PAD = 12  # side padding inside a footer box
LABEL_FS = 21
TITLE_FS = 32
CAPTION_FS = 26
CHAR_W = 0.56  # average glyph width / font size for a humanist sans


def est_width(text, fs):
    return len(text) * fs * CHAR_W


def warn_if_overflow(lines, width, where, main_fs=MAIN_FS, sub_fs=SUB_FS):
    for i, line in enumerate(lines):
        fs = main_fs if i == 0 else sub_fs
        if est_width(line, fs) > width - 24:
            print(f"warning: '{line}' probably overflows {where} ({est_width(line, fs):.0f}px > {width - 24}px)", file=sys.stderr)


def text_block(cx, cy, lines, color, bold_first=False, main_fs=MAIN_FS, sub_fs=SUB_FS):
    """Center a block of lines at (cx, cy). First line is the main line."""
    n = len(lines)
    heights = [main_fs] + [sub_fs] * (n - 1)
    gaps = [0] + [8] * (n - 1)
    total = sum(heights) + sum(gaps)
    y = cy - total / 2
    out = []
    for i, line in enumerate(lines):
        fs = heights[i]
        y += gaps[i] + fs
        fill = color if i == 0 else (color if color == DANGER else TEXT_DIM)
        weight = ' font-weight="600"' if (i == 0 and bold_first) else ""
        out.append(
            f'<text x="{cx:.1f}" y="{y - fs * 0.22:.1f}" font-size="{fs}" fill="{fill}"{weight} '
            f'text-anchor="middle" dominant-baseline="auto">{escape(line)}</text>'
        )
    return "\n".join(out)


def box(x, y, w, h, lines, danger=False, rx=8, main_fs=MAIN_FS, sub_fs=SUB_FS):
    color = DANGER if danger else TEXT
    warn_if_overflow(lines, w, f"box '{lines[0]}'", main_fs, sub_fs)
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
        f'fill="{CONTAINER_BG}" stroke="{BOX_STROKE}" stroke-width="1.4"/>\n'
        + text_block(x + w / 2, y + h / 2, lines, color, bold_first=danger, main_fs=main_fs, sub_fs=sub_fs)
    )


def box_height(lines):
    return BOX_MAIN_H + max(0, len(lines) - 2) * (SUB_FS + 8)


def render(spec):
    panel_x, panel_w = PANEL_MARGIN, CANVAS_W - 2 * PANEL_MARGIN
    cont_x = panel_x + CONTAINER_MARGIN
    cont_w = panel_w - 2 * CONTAINER_MARGIN
    inner_w = cont_w - 2 * CONTAINER_PAD

    title_y = PANEL_TOP + 96
    cont_y = title_y + 56
    top = cont_y + CONTAINER_PAD
    parts = []

    if spec.get("container_label"):
        parts.append(
            f'<text x="{cont_x + 28}" y="{cont_y + 34}" font-size="{SUB_FS + 1}" fill="{TEXT_DIM}" '
            f'font-family="monospace">{escape(spec["container_label"])}</text>'
        )
        top += 6

    columns = spec["columns"]
    total_w = sum(c["width"] for c in columns) + COLUMN_GAP * (len(columns) - 1)
    if total_w > inner_w:
        print(f"warning: columns are {total_w}px wide but only {inner_w}px available", file=sys.stderr)
    x = cont_x + CONTAINER_PAD + (inner_w - total_w) / 2

    # Height of the tallest labelled section decides the row height.
    def stack_height(col):
        return sum(box_height(b["lines"]) for b in col["boxes"]) + BOX_GAP * (len(col["boxes"]) - 1)

    labelled = [c for c in columns if c.get("label")]
    row_h = max((stack_height(c) + 2 * SECTION_PAD for c in labelled), default=0)
    row_h = max(row_h, max(stack_height(c) for c in columns))
    label_h = LABEL_FS + 22 if labelled else 0
    row_top = top + label_h

    for col in columns:
        w = col["width"]
        if col.get("label"):
            warn_if_overflow([col["label"]], w, f"label '{col['label']}'")
            parts.append(
                f'<text x="{x + w / 2:.1f}" y="{top + LABEL_FS - 2:.1f}" font-size="{LABEL_FS}" fill="{TEXT}" '
                f'text-anchor="middle">{escape(col["label"])}</text>'
            )
            parts.append(
                f'<rect x="{x:.1f}" y="{row_top:.1f}" width="{w}" height="{row_h:.1f}" rx="6" '
                f'fill="none" stroke="{BOX_STROKE}" stroke-width="1.4"/>'
            )
            # distribute the boxes evenly inside the section
            n = len(col["boxes"])
            heights = [box_height(b["lines"]) for b in col["boxes"]]
            free = row_h - 2 * SECTION_PAD - sum(heights)
            gap = free / (n - 1) if n > 1 else 0
            by = row_top + SECTION_PAD + (free / 2 if n == 1 else 0)
            for b, h in zip(col["boxes"], heights):
                parts.append(box(x + SECTION_PAD, by, w - 2 * SECTION_PAD, h, b["lines"], b.get("danger", False)))
                by += h + gap
        else:
            n = len(col["boxes"])
            if n == 1:
                parts.append(box(x, row_top, w, row_h, col["boxes"][0]["lines"], col["boxes"][0].get("danger", False)))
            else:
                heights = [box_height(b["lines"]) for b in col["boxes"]]
                free = row_h - sum(heights)
                gap = free / (n - 1)
                by = row_top
                for b, h in zip(col["boxes"], heights):
                    parts.append(box(x, by, w, h, b["lines"], b.get("danger", False)))
                    by += h + gap
        x += w + COLUMN_GAP

    y = row_top + row_h
    footer = spec.get("footer") or []
    if footer:
        y += 44
        n = len(footer)
        # Width each box needs for its own text, then scale so the row spans the inner width.
        natural = [
            max(est_width(line, FOOTER_MAIN_FS if i == 0 else FOOTER_SUB_FS) for i, line in enumerate(b["lines"]))
            + 2 * FOOTER_TEXT_PAD
            for b in footer
        ]
        avail = inner_w - FOOTER_GAP * (n - 1)
        scale = avail / sum(natural)
        if scale < 1:
            print(f"warning: footer needs {sum(natural):.0f}px but only {avail:.0f}px available", file=sys.stderr)
        widths = [w * scale for w in natural]
        fx = cont_x + CONTAINER_PAD
        for b, fw in zip(footer, widths):
            parts.append(box(fx, y, fw, FOOTER_H, b["lines"], b.get("danger", False), rx=6,
                             main_fs=FOOTER_MAIN_FS, sub_fs=FOOTER_SUB_FS))
            fx += fw + FOOTER_GAP
        y += FOOTER_H

    cont_h = y + CONTAINER_PAD - cont_y
    panel_h = cont_h + (cont_y - PANEL_TOP) + 72
    caption = spec.get("caption")
    if caption:
        caption_y = PANEL_TOP + panel_h + 74
        canvas_h = caption_y + 44
    else:
        canvas_h = PANEL_TOP + panel_h + PANEL_TOP

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_W}" height="{canvas_h:.0f}" '
        f'viewBox="0 0 {CANVAS_W} {canvas_h:.0f}" font-family="{FONT}">',
        f'<title>{escape(spec["title"])}</title>',
        f'<rect width="{CANVAS_W}" height="{canvas_h:.0f}" fill="{PAGE_BG}"/>',
        f'<rect x="{panel_x}" y="{PANEL_TOP}" width="{panel_w}" height="{panel_h:.0f}" fill="{PANEL_BG}"/>',
        f'<text x="{CANVAS_W / 2}" y="{title_y}" font-size="{TITLE_FS}" fill="{TEXT}" text-anchor="middle">'
        f'{escape(spec["title"])}</text>',
        f'<rect x="{cont_x}" y="{cont_y}" width="{cont_w}" height="{cont_h:.0f}" rx="26" '
        f'fill="{CONTAINER_BG}" stroke="{CONTAINER_STROKE}" stroke-width="2"/>',
        *parts,
    ]
    if caption:
        svg.append(
            f'<text x="{CANVAS_W / 2}" y="{caption_y:.0f}" font-size="{CAPTION_FS}" fill="{CAPTION}" '
            f'font-style="italic" text-anchor="middle">{escape(caption)}</text>'
        )
    svg.append("</svg>")
    return "\n".join(svg) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("spec", help="JSON spec file")
    ap.add_argument("-o", "--out", required=True, help="output .svg path")
    args = ap.parse_args()
    with open(args.spec) as f:
        spec = json.load(f)
    with open(args.out, "w") as f:
        f.write(render(spec))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
