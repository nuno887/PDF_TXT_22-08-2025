#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path
import fitz  # PyMuPDF

# ----------------------- Tunable defaults -----------------------
DEFAULT_MIN_MINOR_SHARE = 0.18   # min share of lines on the smaller side to accept 2 columns
DEFAULT_VERT_OVERLAP     = 0.30  # min vertical overlap (0..1) between left/right blocks for 2 columns
DEFAULT_TOP_PORTION      = 0.65  # portion of the page (top) used to detect Sumário continuation
DEFAULT_SHORT_LEN        = 120   # a "short" line length (characters) typical in Sumário items
DEFAULT_MIN_SHORT_COUNT  = 6     # minimum number of short lines in top portion to keep sumário mode
DEFAULT_SHORT_RATIO      = 0.60  # at least this fraction of top lines must be short
DEFAULT_MAX_LONG_IN_TOP  = 2     # allow up to this many long lines in the top portion


# ---------------------------- Utilities ----------------------------

def page_has_sumario_heading(plain_text: str) -> bool:
    """Detect a 'Sumário' heading on this page (accent tolerant)."""
    t = plain_text.lower()
    return ("sumário" in t) or ("sumario" in t)

def get_page_lines(page) -> list:
    """
    Extract line entries as (y0, x0, y1, text).
    Using PyMuPDF's dict API to stay robust across PDFs.
    """
    out = []
    d = page.get_text("dict")
    for block in d.get("blocks", []):
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            x0 = min(s["bbox"][0] for s in spans)
            y0 = line["bbox"][1]
            y1 = line["bbox"][3]
            text = "".join(s["text"] for s in spans).strip()
            if text:
                out.append((y0, x0, y1, text))
    return out

def choose_adaptive_split_x(lines, page_width) -> float:
    """
    Choose an x split between columns by finding the largest gap
    among distinct x0 positions roughly around the center band.
    """
    xs = sorted(set(round(x0, 1) for _, x0, _, _ in lines))
    if len(xs) < 4:
        return page_width * 0.5
    lo_idx = int(len(xs) * 0.30)
    hi_idx = int(len(xs) * 0.70)
    lo_idx = max(1, min(lo_idx, len(xs) - 2))
    hi_idx = max(lo_idx + 1, min(hi_idx, len(xs) - 1))
    best_gap = 0.0
    best_mid = page_width * 0.5
    for i in range(lo_idx, hi_idx):
        g = xs[i] - xs[i - 1]
        if g > best_gap:
            best_gap = g
            best_mid = (xs[i] + xs[i - 1]) / 2.0
    return best_mid

def vertical_overlap_ratio(ymin_a, ymax_a, ymin_b, ymax_b) -> float:
    """IOU along Y: intersection over union of two vertical ranges."""
    inter = max(0.0, min(ymax_a, ymax_b) - max(ymin_a, ymin_b))
    uni   = max(ymax_a, ymax_b) - min(ymin_a, ymin_b)
    return (inter / uni) if uni > 0 else 0.0

def is_sumario_like_continuation(
    lines,
    page_height,
    top_portion=DEFAULT_TOP_PORTION,
    short_len=DEFAULT_SHORT_LEN,
    min_short_count=DEFAULT_MIN_SHORT_COUNT,
    short_ratio=DEFAULT_SHORT_RATIO,
    max_long_in_top=DEFAULT_MAX_LONG_IN_TOP,
) -> bool:
    """
    Heuristic to decide if this page still looks like Sumário (continuation):
    - In the top portion, most lines are short bullets/entities/actions.
    - There are very few long paragraph lines.
    """
    top_lines = [txt for (y0, x0, y1, txt) in lines if y0 <= page_height * top_portion]
    if not top_lines:
        return False
    short = [txt for txt in top_lines if len(txt) <= short_len]
    long_ = [txt for txt in top_lines if len(txt) > 180]
    if len(short) >= max(min_short_count, int(short_ratio * len(top_lines))) and len(long_) <= max_long_in_top:
        return True
    return False

def detect_layout_two_columns(
    lines,
    page_width,
    min_minor_share=DEFAULT_MIN_MINOR_SHARE,
    min_vert_overlap=DEFAULT_VERT_OVERLAP,
):
    """
    Decide if a page is 1 or 2 columns using:
      - adaptive split
      - require both sides to have enough lines (minor_share)
      - require vertical overlap between sides
    Returns: ("single", None) or ("double", split_x)
    """
    if len(lines) < 6:
        return "single", None

    split_x = choose_adaptive_split_x(lines, page_width)
    left  = [(y0, x0, y1, t) for (y0, x0, y1, t) in lines if x0 < split_x]
    right = [(y0, x0, y1, t) for (y0, x0, y1, t) in lines if x0 >= split_x]

    total = len(lines)
    if total == 0:
        return "single", None

    # Both sides must carry a reasonable share
    min_side_share = min(len(left), len(right)) / total
    if min_side_share < min_minor_share:
        return "single", None

    # Must overlap vertically (prevents Sumário false positives)
    yLmin, yLmax = min(y0 for y0, _, _, _ in left), max(y1 for _, _, y1, _ in left)
    yRmin, yRmax = min(y0 for y0, _, _, _ in right), max(y1 for _, _, y1, _ in right)
    vo = vertical_overlap_ratio(yLmin, yLmax, yRmin, yRmax)
    if vo < min_vert_overlap:
        return "single", None

    return "double", split_x

def extract_single_column(lines) -> str:
    lines_sorted = sorted(lines, key=lambda it: (it[0], it[1]))
    return "\n".join(t for _, _, _, t in lines_sorted).strip()

def extract_two_columns(lines, split_x) -> str:
    left  = [(y0, x0, y1, t) for (y0, x0, y1, t) in lines if x0 < split_x]
    right = [(y0, x0, y1, t) for (y0, x0, y1, t) in lines if x0 >= split_x]
    left.sort(key=lambda it: (it[0], it[1]))
    right.sort(key=lambda it: (it[0], it[1]))
    out = []
    if left:
        out.extend(t for _, _, _, t in left)
        out.append("")  # blank line between columns
    if right:
        out.extend(t for _, _, _, t in right)
    return "\n".join(out).strip()


# --------------------------- Main extraction ---------------------------

def extract_text_from_pdf_file(
    input_pdf: str,
    drop_last_page: bool = False,
    min_minor_share: float = DEFAULT_MIN_MINOR_SHARE,
    min_vert_overlap: float = DEFAULT_VERT_OVERLAP,
) -> str:
    """
    Process one PDF:
      - Optionally drop the last page
      - Keep Sumário across pages (single-column) until it ends
      - Otherwise, detect 1 vs 2 columns and flatten to one stream
    """
    doc = fitz.open(input_pdf)
    pages_idx = list(range(doc.page_count))
    if drop_last_page and len(pages_idx) >= 2:
        pages_idx = pages_idx[:-1]  # keep single-page PDFs intact

    texts = []
    in_sumario = False  # state: we are currently inside the Sumário section

    for i in pages_idx:
        page = doc[i]
        page_width  = page.rect.width
        page_height = page.rect.height

        lines = get_page_lines(page)
        plain = page.get_text().strip()

        # Enter sumário mode when a heading appears
        if page_has_sumario_heading(plain):
            in_sumario = True

        if in_sumario:
            # While it still looks like Sumário in the top portion -> force single-column
            if is_sumario_like_continuation(lines, page_height):
                texts.append(extract_single_column(lines))
                continue
            else:
                # no longer looks like Sumário -> exit mode and proceed normally
                in_sumario = False

        layout, split_x = detect_layout_two_columns(
            lines,
            page_width=page_width,
            min_minor_share=min_minor_share,
            min_vert_overlap=min_vert_overlap,
        )
        if layout == "double" and split_x is not None:
            text = extract_two_columns(lines, split_x)
        else:
            text = extract_single_column(lines)

        texts.append(text)

    doc.close()
    return ("\n\n".join(t for t in texts if t).strip() + ("\n" if texts else ""))


def process_path(input_path: str, output_dir: str, drop_last_page: bool, overwrite: bool):
    os.makedirs(output_dir, exist_ok=True)

    if os.path.isfile(input_path) and input_path.lower().endswith(".pdf"):
        files = [input_path]
    elif os.path.isdir(input_path):
        files = []
        for root, _, names in os.walk(input_path):
            for n in names:
                if n.lower().endswith(".pdf"):
                    files.append(os.path.join(root, n))
    else:
        print(f"Input not found: {input_path}")
        return

    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]
        out_path = os.path.join(output_dir, base + ".txt")
        if os.path.exists(out_path) and not overwrite:
            print(f"[skip] {out_path} exists")
            continue
        try:
            text = extract_text_from_pdf_file(
                f,
                drop_last_page=drop_last_page,
                min_minor_share=DEFAULT_MIN_MINOR_SHARE,
                min_vert_overlap=DEFAULT_VERT_OVERLAP,
            )
            with open(out_path, "w", encoding="utf-8") as h:
                h.write(text)
            print(f"[ok] {f} -> {out_path}")
        except Exception as e:
            print(f"[err] {f}: {e}")


def main():
    ap = argparse.ArgumentParser(
        description="Flatten PDFs (1–2 columns) into single-column text; preserve Sumário across pages; optionally drop the last page."
    )
    ap.add_argument("input", help="PDF file or folder")
    ap.add_argument("-o", "--output", required=True, help="Output folder for .txt files")
    ap.add_argument("--drop-last-page", action="store_true", help="Skip the last page of every PDF")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing .txt files")
    args = ap.parse_args()

    process_path(
        input_path=args.input,
        output_dir=args.output,
        drop_last_page=args.drop_last_page,
        overwrite=args.overwrite,
    )

if __name__ == "__main__":
    main()
