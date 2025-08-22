#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import fitz  # PyMuPDF

# ---- Tunables ----
MINOR_SHARE = 0.18     # smaller side must have at least this share to accept 2 cols
VERT_OVERLAP = 0.30    # vertical overlap (0..1) between left/right to accept 2 cols
SHORT_LEN = 120        # "short" line length typical in Sumário bullets
SHORT_RATIO = 0.60     # ≥60% short lines ⇒ Sumário-like
MIN_SHORT = 5          # at least N short lines
CUTOFF_MIN = 0.10      # search cutoffs from 10%...
CUTOFF_MAX = 0.70      # ...to 70% of page height
CUTOFF_STEP = 0.05     # step by 5%


# ---------------- Core helpers ----------------

def get_page_lines(page):
    """Return [(y0, x0, y1, text)] from PyMuPDF dict API."""
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
    """Pick split X via largest gap in distinct x0 values near center band."""
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
    uni   = max(a_ymax, b_ymax) - min(a_ymin, b_ymin)
    return (inter / uni) if uni > 0 else 0.0

def two_col_ok(lines, page_width, minor_share=MINOR_SHARE, vert_overlap=VERT_OVERLAP):
    """Check if a group of lines is convincingly 2-column."""
    if len(lines) < 6:
        return False, None
    split_x = choose_adaptive_split_x(lines, page_width)
    left  = [(y0, x0, y1) for (y0, x0, y1, _) in lines if x0 < split_x]
    right = [(y0, x0, y1) for (y0, x0, y1, _) in lines if x0 >= split_x]
    nL, nR = len(left), len(right)
    total = nL + nR
    if total == 0:
        return False, None
    minor = min(nL, nR) / total
    if minor < minor_share:
        return False, None
    yLmin, yLmax = min(y for y,_,_ in left),  max(y for _,_,y in left)
    yRmin, yRmax = min(y for y,_,_ in right), max(y for _,_,y in right)
    vo = vertical_overlap_ratio(yLmin, yLmax, yRmin, yRmax)
    if vo < vert_overlap:
        return False, None
    return True, split_x

def extract_two_col(lines, split_x):
    left  = [(y0, x0, y1, t) for (y0, x0, y1, t) in lines if x0 < split_x]
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

def looks_like_sumario(lines, short_len=SHORT_LEN, short_ratio=SHORT_RATIO, min_short=MIN_SHORT):
    """Sumário-ish: many short lines, few paragraphs."""
    if not lines:
        return False
    short = [t for (_, _, _, t) in lines if len(t) <= short_len]
    return len(short) >= max(min_short, int(short_ratio * len(lines)))

def find_dynamic_cutoff(lines, page_h, page_w):
    """
    Search cutoff in [10%, 70%] s.t. TOP looks like Sumário and BOTTOM is valid 2-col.
    Return y_cutoff or None.
    """
    if len(lines) < 8:
        return None
    lines_sorted = sorted(lines, key=lambda it: it[0])
    best_score, best_cut = 0.0, None
    for r in frange(CUTOFF_MIN, CUTOFF_MAX, CUTOFF_STEP):
        ycut = page_h * r
        top = [it for it in lines_sorted if it[0] < ycut]
        bot = [it for it in lines_sorted if it[0] >= ycut]
        if len(top) < 3 or len(bot) < 6:
            continue
        if looks_like_sumario(top):
            ok, split_x = two_col_ok(bot, page_w)
            if ok:
                # score prefers having “enough” content on both sides
                score = min(len(top) / 10.0, 2.0) + min(len(bot) / 20.0, 2.0)
                if score > best_score:
                    best_score, best_cut = score, ycut
    return best_cut

def frange(start, stop, step):
    x = start
    while x <= stop + 1e-9:
        yield round(x, 4)
        x += step


# -------- Sumário-first extraction (guarantees order) --------

def extract_text_sumario_then_body(input_pdf: str) -> str:
    """
    1) Accumulate Sumário first (across pages, including top slices).
    2) Accumulate Body after Sumário (including bottom slices and later pages).
    3) Output = Sumário + blank line + Body.
    """
    doc = fitz.open(input_pdf)
    sumario_parts, body_parts = [], []
    in_sumario = False

    try:
        for i in range(doc.page_count):
            page = doc[i]
            page_w, page_h = page.rect.width, page.rect.height
            lines = get_page_lines(page)
            if not lines:
                continue

            plain = page.get_text().lower()
            if ("sumário" in plain) or ("sumario" in plain):
                in_sumario = True  # we saw the heading on this or earlier page

            # Try to split this page: Sumário on top, body below
            ycut = find_dynamic_cutoff(lines, page_h, page_w)

            if in_sumario:
                if ycut is not None:
                    # top → Sumário, bottom → Body
                    top = [it for it in lines if it[0] < ycut]
                    bot = [it for it in lines if it[0] >= ycut]
                    sumario_parts.append(extract_single(top))
                    # bottom may be 2-col or single
                    ok, split_x = two_col_ok(bot, page_w)
                    body_parts.append(extract_two_col(bot, split_x) if ok and split_x else extract_single(bot))
                    # Sumário is done once we found a split
                    in_sumario = False
                else:
                    # Entire page still looks like Sumário or no clear split:
                    # if it looks Sumário-ish, keep adding; else treat whole as single and keep Sumário going
                    if looks_like_sumario(lines):
                        sumario_parts.append(extract_single(lines))
                    else:
                        # ambiguous: be safe and keep gathering into Sumário on early pages
                        sumario_parts.append(extract_single(lines))
            else:
                # After Sumário finished → normal body pages
                ok, split_x = two_col_ok(lines, page_w)
                body_parts.append(extract_two_col(lines, split_x) if ok and split_x else extract_single(lines))

    finally:
        doc.close()

    sumario_text = "\n\n".join(x for x in sumario_parts if x).strip()
    body_text    = "\n\n".join(x for x in body_parts if x).strip()
    if sumario_text and body_text:
        return sumario_text + "\n\n" + body_text + "\n"
    return (sumario_text or body_text) + ("\n" if (sumario_text or body_text) else "")


# ---------------- Batch (dir → dir) ----------------

def process_dir(input_dir: str, output_dir: str, recurse=False, overwrite=False):
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
            text = extract_text_sumario_then_body(pdf_path)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"[ok] {pdf_path} -> {out_path}")
        except Exception as e:
            print(f"[err] {pdf_path}: {e}")


# ----------------------------- CLI -----------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Extract Sumário first (across pages), then body; handles mid-page switch from 1-col to 2-col."
    )
    ap.add_argument("input_dir", help="Folder with PDFs")
    ap.add_argument("output_dir", help="Folder to write .txt files")
    ap.add_argument("--recurse", action="store_true", help="Process subfolders recursively")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing .txt files")
    args = ap.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Input not found or not a directory: {args.input_dir}")
        return

    process_dir(args.input_dir, args.output_dir, recurse=args.recurse, overwrite=args.overwrite)

if __name__ == "__main__":
    main()

#python pdf_sumario_then_body.py "C:\Users\nuno.ms.goncalves\Desktop\SQLite_Creation\JORAM" "C:\Users\nuno.ms.goncalves\Desktop\SQLite_Creation\JORAM_TXT" --recurse
