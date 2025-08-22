#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import fitz  # PyMuPDF

# ---------- Tunables (adjust if needed) ----------
MINOR_SHARE = 0.18       # min fraction of lines on the smaller side to accept 2 cols
VERT_OVERLAP = 0.30      # min vertical overlap (0..1) between left/right sides
SHORT_LEN = 120          # characters: a "short" Sumário line
SHORT_RATIO = 0.60       # in Sumário top: ≥60% of lines should be short
MIN_SHORT_COUNT = 5      # ...and at least 5 short lines
CUTOFF_MIN = 0.10        # search cutoffs from 10% ...
CUTOFF_MAX = 0.70        # ... to 70% of page height
CUTOFF_STEP = 0.05       # step by 5%


# ---------------- Core helpers ----------------

def get_page_lines(page):
    """Return [(y0, x0, y1, text)]."""
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


def extract_single(lines):
    lines = sorted(lines, key=lambda it: (it[0], it[1]))
    return "\n".join(t for _, _, _, t in lines).strip()


def choose_adaptive_split_x(lines, page_width):
    """Pick split X via largest gap in distinct x0 values near center."""
    xs = sorted(set(round(x0, 1) for _, x0, _, _ in lines))
    if len(xs) < 4:
        return page_width * 0.5
    lo = max(1, int(len(xs) * 0.30))
    hi = min(len(xs) - 1, max(lo + 1, int(len(xs) * 0.70)))
    best_gap = 0.0
    best_mid = page_width * 0.5
    for i in range(lo, hi):
        gap = xs[i] - xs[i - 1]
        if gap > best_gap:
            best_gap = gap
            best_mid = (xs[i] + xs[i - 1]) / 2.0
    return best_mid


def vertical_overlap_ratio(a_ymin, a_ymax, b_ymin, b_ymax):
    inter = max(0.0, min(a_ymax, b_ymax) - max(a_ymin, b_ymin))
    uni = max(a_ymax, b_ymax) - min(a_ymin, b_ymin)
    return (inter / uni) if uni > 0 else 0.0


def two_col_ok(lines, page_width, minor_share=MINOR_SHARE, vert_overlap=VERT_OVERLAP):
    """Check if a group of lines is convincingly 2-column."""
    if len(lines) < 6:
        return False, None, {"reason": "few_lines", "n": len(lines)}
    split_x = choose_adaptive_split_x(lines, page_width)
    left = [(y0, x0, y1) for (y0, x0, y1, _) in lines if x0 < split_x]
    right = [(y0, x0, y1) for (y0, x0, y1, _) in lines if x0 >= split_x]
    nL, nR = len(left), len(right)
    total = nL + nR
    if total == 0:
        return False, None, {"reason": "no_lines"}
    minor = min(nL, nR) / total
    if minor < minor_share:
        return False, None, {"reason": "minor_share", "minor": round(minor, 3)}
    yLmin, yLmax = min(y for y, _, _ in left), max(y for _, _, y in left)
    yRmin, yRmax = min(y for y, _, _ in right), max(y for _, _, y in right)
    vo = vertical_overlap_ratio(yLmin, yLmax, yRmin, yRmax)
    if vo < vert_overlap:
        return False, None, {"reason": "overlap", "overlap": round(vo, 3)}
    return True, split_x, {"reason": "ok", "nL": nL, "nR": nR, "minor": round(minor, 3), "overlap": round(vo, 3)}


def extract_two_col(lines, split_x):
    left = [(y0, x0, y1, t) for (y0, x0, y1, t) in lines if x0 < split_x]
    right = [(y0, x0, y1, t) for (y0, x0, y1, t) in lines if x0 >= split_x]
    left.sort(key=lambda it: (it[0], it[1]))
    right.sort(key=lambda it: (it[0], it[1]))
    parts = []
    if left:
        parts.extend(t for _, _, _, t in left)
        parts.append("")
    if right:
        parts.extend(t for _, _, _, t in right)
    return "\n".join(parts).strip()


def looks_like_sumario(lines, short_len=SHORT_LEN, short_ratio=SHORT_RATIO, min_short=MIN_SHORT_COUNT):
    """Many short lines, few long paragraphs."""
    if not lines:
        return False
    short = [t for (_, _, _, t) in lines if len(t) <= short_len]
    return len(short) >= max(min_short, int(short_ratio * len(lines)))


def find_dynamic_cutoff(lines, page_h, page_w):
    """
    Search cutoff in [10%, 70%] to maximize:
      top looks like Sumário AND bottom is valid 2-column.
    Returns y_cutoff or None.
    """
    if len(lines) < 8:
        return None
    best_score, best_cut = 0.0, None
    lines_sorted = sorted(lines, key=lambda it: it[0])

    for r in frange(CUTOFF_MIN, CUTOFF_MAX, CUTOFF_STEP):
        ycut = page_h * r
        top = [it for it in lines_sorted if it[0] < ycut]
        bot = [it for it in lines_sorted if it[0] >= ycut]
        if len(top) < 3 or len(bot) < 6:
            continue
        top_ok = looks_like_sumario(top)
        bot_ok, split_x, _ = two_col_ok(bot, page_w)
        if top_ok and bot_ok:
            # score: prefer stronger splits (more lines both sides)
            score = min(len(top) / 10.0, 2.0) + min(len(bot) / 20.0, 2.0)
            if score > best_score:
                best_score, best_cut = score, ycut

    return best_cut


def frange(start, stop, step):
    x = start
    while x <= stop + 1e-9:
        yield round(x, 4)
        x += step


# --------------- Extraction strategies per page ---------------

def extract_page_mixed(page):
    """
    Allow a page to start single-column (Sumário) and switch to 2-column.
    Auto-detects cutoff 10..70% of page height.
    Falls back to single or two-column if no clear split found.
    """
    lines = get_page_lines(page)
    if not lines:
        return ""
    page_w, page_h = page.rect.width, page.rect.height

    # Try mixed split
    ycut = find_dynamic_cutoff(lines, page_h, page_w)
    if ycut is not None:
        top = [it for it in lines if it[0] < ycut]
        bot = [it for it in lines if it[0] >= ycut]
        top_text = extract_single(top)

        ok, split_x, _ = two_col_ok(bot, page_w)
        if ok and split_x is not None:
            bot_text = extract_two_col(bot, split_x)
        else:
            bot_text = extract_single(bot)
        return (top_text + "\n\n" + bot_text).strip()

    # No mixed cutoff → decide whole page
    ok, split_x, _ = two_col_ok(lines, page_w)
    if ok and split_x is not None:
        return extract_two_col(lines, split_x)
    return extract_single(lines)


# ---------------- Batch processing (dir → dir) ----------------

def extract_text_from_single_pdf(input_path: str) -> str:
    """
    First page: usually single (covers/Sumário) → keep simple single.
    Other pages: allow mixed mode (top single, bottom 2-col).
    """
    doc = fitz.open(input_path)
    texts = []
    try:
        n = doc.page_count
        for i in range(n):
            page = doc[i]
            if i == 0:
                lines = get_page_lines(page)
                texts.append(extract_single(lines))
            else:
                texts.append(extract_page_mixed(page))
    finally:
        doc.close()
    return "\n\n".join(t for t in texts if t).strip()


def extract_text_from_pdf_dir(input_dir: str, output_dir: str, recurse=False, overwrite=False):
    os.makedirs(output_dir, exist_ok=True)

    def iter_pdfs(root):
        if recurse:
            for r, _, files in os.walk(root):
                for fn in files:
                    if fn.lower().endswith(".pdf"):
                        yield os.path.join(r, fn)
        else:
            for fn in os.listdir(root):
                if fn.lower().endswith(".pdf"):
                    yield os.path.join(root, fn)

    for pdf_path in iter_pdfs(input_dir):
        base = os.path.splitext(os.path.basename(pdf_path))[0]
        out_path = os.path.join(output_dir, base + ".txt")
        if os.path.exists(out_path) and not overwrite:
            print(f"[skip] {out_path} exists")
            continue
        try:
            text = extract_text_from_single_pdf(pdf_path)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"[ok] {pdf_path} -> {out_path}")
        except Exception as e:
            print(f"[err] {pdf_path}: {e}")


# ----------------------------- CLI -----------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Extract text from PDFs (handles Sumário top→two-column bottom on the same page)."
    )
    ap.add_argument("input_dir", help="Folder with PDFs")
    ap.add_argument("output_dir", help="Folder to write .txt files")
    ap.add_argument("--recurse", action="store_true", help="Process subfolders recursively")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing .txt files")
    args = ap.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Input not found or not a directory: {args.input_dir}")
        return

    extract_text_from_pdf_dir(args.input_dir, args.output_dir, recurse=args.recurse, overwrite=args.overwrite)

if __name__ == "__main__":
    main()

#python pdf_dir_to_txt_mixed.py "C:\Users\nuno.ms.goncalves\Desktop\SQLite_Creation\JORAM" "C:\Users\nuno.ms.goncalves\Desktop\SQLite_Creation\JORAM_TXT" --recurse