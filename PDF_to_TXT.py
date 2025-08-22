#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import subprocess
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from sklearn.cluster import KMeans
import difflib


# =========================== OCR fallback ===========================

def has_text_layer(pdf_path: Path, sample_pages: int = 3) -> bool:
    """Quickly check for any extractable words on first N pages."""
    with fitz.open(pdf_path) as doc:
        for i, p in enumerate(doc):
            if i >= sample_pages:
                break
            if p.get_text("words"):
                return True
    return False


def ensure_text_layer(pdf_path: Path, lang: str = "por+eng", jobs: int = 4) -> Path:
    """
    If the PDF has no text layer, run OCRmyPDF to create <name>_ocr.pdf and return that path.
    Otherwise, return the original path.
    """
    pdf_path = Path(pdf_path)
    if has_text_layer(pdf_path):
        return pdf_path

    ocr_out = pdf_path.with_name(pdf_path.stem + "_ocr.pdf")
    cmd = [
        "ocrmypdf",
        "--force-ocr",
        "--language", lang,
        "--rotate-pages", "--deskew", "--clean",
        "--tesseract-timeout", "0",
        "--jobs", str(jobs),
        str(pdf_path), str(ocr_out)
    ]
    subprocess.run(cmd, check=True)
    return ocr_out


# ====================== Column detection & reflow =====================

def load_pages_with_words(pdf_path: Path):
    """Return list[page] where page is list[((x1,y1,x2,y2), text)]."""
    pages = []
    with fitz.open(pdf_path) as doc:
        for p in doc:
            words = p.get_text("words")  # (x1,y1,x2,y2,text,block,line,word)
            pages.append([((w[0], w[1], w[2], w[3]), w[4]) for w in words if w[4].strip()])
    return pages


def group_lines_by_y(sorted_words, y_tol: float = 3.0):
    """Group words into lines by y proximity. Words must be sorted by (y,x)."""
    lines, buf = [], []
    cur_y = None
    for (x1, y1, x2, y2), text in sorted_words:
        if cur_y is None or abs(y1 - cur_y) <= y_tol:
            buf.append((x1, text))
            cur_y = y1 if cur_y is None else (cur_y + y1) / 2.0
        else:
            buf.sort(key=lambda t: t[0])
            lines.append(" ".join(t for _, t in buf))
            buf = [(x1, text)]
            cur_y = y1
    if buf:
        buf.sort(key=lambda t: t[0])
        lines.append(" ".join(t for _, t in buf))
    return lines


def merge_hyphenation(lines):
    """Join lines where previous ends with '-' and next begins lowercase."""
    out = []
    for i, ln in enumerate(lines):
        if i > 0 and out and out[-1].endswith('-') and ln and ln[:1].islower():
            out[-1] = out[-1][:-1] + ln
        else:
            out.append(ln)
    return out


def detect_columns(words, max_cols: int = 2, min_minor_share: float = 0.15):
    """
    Cluster by x-center to detect 1 or 2 columns.
    If a second cluster holds < min_minor_share of words, treat as 1 column.
    Returns list[list[word]] columns ordered left→right.
    """
    if not words:
        return [[]]

    centers_x = np.array([((b[0] + b[2]) / 2.0,) for b, _ in words])
    best = None
    for k in range(1, min(max_cols, len(centers_x)) + 1):
        km = KMeans(n_clusters=k, n_init=5, random_state=0)
        labels = km.fit_predict(centers_x)
        inertia = km.inertia_
        if best is None or inertia < best[0]:
            best = (inertia, labels, km.cluster_centers_.flatten())

    _, labels, centers = best
    # Count per cluster and optionally collapse minor cluster
    counts = np.bincount(labels)
    if len(counts) == 2:
        minor_share = counts.min() / counts.sum()
        if minor_share < min_minor_share:
            # Treat as one column: ignore labels and sort all words by (y,x)
            col = sorted(words, key=lambda w: (w[0][1], w[0][0]))
            return [col]

    # group words by cluster
    clusters = {}
    for idx, lab in enumerate(labels):
        clusters.setdefault(lab, []).append(words[idx])
    # order clusters by center x
    ordered = [clusters[k] for k in sorted(clusters.keys(), key=lambda k: centers[k])]
    # sort each cluster by (y, x)
    for col in ordered:
        col.sort(key=lambda w: (w[0][1], w[0][0]))
    return ordered


def rebuild_page_text(words, newline_after_semicolon: bool = False):
    """
    Convert a page's words into single-column reading order:
    column 1 (top→bottom) then column 2 (top→bottom).
    """
    columns = detect_columns(words, max_cols=2)
    page_lines = []
    for col in columns:
        lines = group_lines_by_y(col, y_tol=3.0)
        lines = merge_hyphenation(lines)
        page_lines.extend(lines)
        page_lines.append("")  # blank line between columns
    text = "\n".join(page_lines).strip("\n")
    if newline_after_semicolon:
        text = re.sub(r';\s*', ';\n', text)
    return text


# ===================== Agnostic header/footer removal =====================

MONTHS_PT = r'janeiro|fevereiro|março|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro'

def _normalize_line(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r'\d', '#', s)                           # mask digits
    s = re.sub(r'\b[ivxlcdm]{1,7}\b', 'rom', s)         # roman numerals
    s = re.sub(fr'\b({MONTHS_PT})\b', 'mes', s)         # months
    s = re.sub(r'\s+', ' ', s)                          # collapse spaces
    s = (s.replace('número', 'numero')
           .replace('nº', 'numero')
           .replace('núm.', 'numero'))
    return s


def _cluster_keys(keys, sim_thresh=0.90):
    """Group near-duplicates using difflib; return list of (rep_key, set(keys))."""
    clusters = []
    for k in keys:
        placed = False
        for i, (rep, members) in enumerate(clusters):
            if difflib.SequenceMatcher(None, k, rep).ratio() >= sim_thresh:
                members.add(k)
                if len(k) < len(rep):
                    clusters[i] = (k, members)
                placed = True
                break
        if not placed:
            clusters.append((k, {k}))
    return clusters


def remove_headers_footers_agnostic(pages_text,
                                    top_lines=4,
                                    bottom_lines=4,
                                    min_share=0.30,
                                    sim_thresh=0.90):
    """
    Remove repeating header/footer lines based on position + repetition.
    """
    n = len(pages_text)
    if n == 0:
        return pages_text

    top_keys_by_page, bot_keys_by_page = [], []
    all_top_norm, all_bot_norm = [], []

    for t in pages_text:
        lines = t.splitlines()
        top = [ln for ln in lines[:top_lines] if ln.strip()]
        bot = [ln for ln in lines[-bottom_lines:] if ln.strip()]
        top_norm = [_normalize_line(ln) for ln in top]
        bot_norm = [_normalize_line(ln) for ln in bot]
        top_keys_by_page.append(top_norm)
        bot_keys_by_page.append(bot_norm)
        all_top_norm.extend(top_norm)
        all_bot_norm.extend(bot_norm)

    top_clusters = _cluster_keys(list(set(all_top_norm)), sim_thresh)
    bot_clusters = _cluster_keys(list(set(all_bot_norm)), sim_thresh)

    def pages_share(clusters, keys_by_page):
        shares = []
        for rep, members in clusters:
            count = 0
            for page_keys in keys_by_page:
                if any(any(difflib.SequenceMatcher(None, k, m).ratio() >= sim_thresh for m in members)
                       for k in page_keys):
                    count += 1
            shares.append((rep, members, count / max(1, len(keys_by_page))))
        return shares

    top_shares = pages_share(top_clusters, top_keys_by_page)
    bot_shares = pages_share(bot_clusters, bot_keys_by_page)

    drop_top = [(rep, members) for rep, members, share in top_shares if share >= min_share]
    drop_bot = [(rep, members) for rep, members, share in bot_shares if share >= min_share]

    def matches_cluster(raw_line, clusters_members):
        norm = _normalize_line(raw_line)
        for _, members in clusters_members:
            if any(difflib.SequenceMatcher(None, norm, m).ratio() >= sim_thresh for m in members):
                return True
        return False

    cleaned_pages = []
    for t in pages_text:
        lines = t.splitlines()
        keep = []
        for i, ln in enumerate(lines):
            is_top_zone = i < top_lines
            is_bot_zone = i >= len(lines) - bottom_lines
            if is_top_zone and matches_cluster(ln, drop_top):
                continue
            if is_bot_zone and matches_cluster(ln, drop_bot):
                continue
            keep.append(ln)
        page_clean = re.sub(r'\n{3,}', '\n\n', "\n".join(keep)).strip("\n")
        cleaned_pages.append(page_clean)
    return cleaned_pages


# ============================== Pipeline ==============================

def pdf_to_single_column_text(pdf_path: Path,
                              use_ocr: bool = True,
                              ocr_lang: str = "por+eng",
                              ocr_jobs: int = 4,
                              remove_headers: bool = True,
                              top_lines: int = 4,
                              bottom_lines: int = 4,
                              min_share: float = 0.30,
                              sim_thresh: float = 0.90,
                              newline_after_semicolon: bool = False) -> str:
    if use_ocr:
        pdf_path = ensure_text_layer(pdf_path, lang=ocr_lang, jobs=ocr_jobs)
    pages = load_pages_with_words(pdf_path)
    pages_text = [rebuild_page_text(words, newline_after_semicolon) for words in pages]
    if remove_headers and len(pages_text) >= 2:
        pages_text = remove_headers_footers_agnostic(
            pages_text,
            top_lines=top_lines,
            bottom_lines=bottom_lines,
            min_share=min_share,
            sim_thresh=sim_thresh
        )
    full = "\n\n".join(pages_text).strip() + "\n"
    return full


def process_path(input_path: Path,
                 output_dir: Path,
                 overwrite: bool,
                 **kwargs):
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            return
        out = output_dir / (input_path.stem + ".txt")
        if out.exists() and not overwrite:
            print(f"[skip] {out} exists")
            return
        try:
            text = pdf_to_single_column_text(input_path, **kwargs)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            print(f"[ok] {input_path.name} -> {out.name}")
        except subprocess.CalledProcessError as e:
            print(f"[ocr err] {input_path}: {e}")
        except Exception as e:
            print(f"[err] {input_path}: {e}")
    else:
        for pdf in input_path.rglob("*.pdf"):
            process_path(pdf, output_dir, overwrite, **kwargs)


def main():
    ap = argparse.ArgumentParser(
        description="Convert PDFs (1–2 columns) into single-column text; auto-OCR if needed; remove repeating headers/footers."
    )
    ap.add_argument("input", help="PDF file or folder")
    ap.add_argument("-o", "--output", help="Output folder for .txt files", required=True)
    ap.add_argument("--no-ocr", action="store_true", help="Disable OCR fallback (default: enabled)")
    ap.add_argument("--ocr-lang", default="por+eng", help="OCR languages (Tesseract), default: por+eng")
    ap.add_argument("--ocr-jobs", type=int, default=4, help="Parallel OCR jobs (default: 4)")
    ap.add_argument("--keep-headers", action="store_true",
                    help="Keep repeating headers/footers (default: remove)")
    ap.add_argument("--top-lines", type=int, default=4, help="Top lines window for header detection")
    ap.add_argument("--bottom-lines", type=int, default=4, help="Bottom lines window for footer detection")
    ap.add_argument("--min-share", type=float, default=0.30, help="Min share of pages to treat line as boilerplate (0.30)")
    ap.add_argument("--sim-thresh", type=float, default=0.90, help="Similarity threshold for clustering (0.90)")
    ap.add_argument("--newline-after-semicolon", action="store_true",
                    help="Insert a newline AFTER each ';' (semicolon kept).")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing .txt files")
    args = ap.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    kwargs = dict(
        use_ocr=not args.no_ocr,
        ocr_lang=args.ocr_lang,
        ocr_jobs=args.ocr_jobs,
        remove_headers=not args.keep_headers,
        top_lines=args.top_lines,
        bottom_lines=args.bottom_lines,
        min_share=args.min_share,
        sim_thresh=args.sim_thresh,
        newline_after_semicolon=args.newline_after_semicolon
    )

    if input_path.exists():
        process_path(input_path, output_dir, overwrite=args.overwrite, **kwargs)
    else:
        print(f"Input not found: {input_path}")


if __name__ == "__main__":
    main()
