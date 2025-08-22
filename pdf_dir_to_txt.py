#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import fitz
import os
import argparse

def extract_text_from_pdf(input_dir: str, output_dir: str, column_split_ratio: float = 0.4, threshold_ratio: float = 0.75, overwrite: bool = False, recurse: bool = False) -> None:
    """
    Processes all PDFs in input_dir and saves extracted text to output_dir.
    - column_split_ratio: horizontal split position (0..1) for two-column pages (default: 0.4)
    - threshold_ratio: if left-side lines / total < threshold -> treat page as double column (default: 0.75)
    - overwrite: if False, skip .txt files that already exist
    - recurse: if True, traverse subfolders
    """
    os.makedirs(output_dir, exist_ok=True)

    def pdf_paths(root):
        if recurse:
            for r, _, files in os.walk(root):
                for fname in files:
                    if fname.lower().endswith(".pdf"):
                        yield os.path.join(r, fname)
        else:
            for fname in os.listdir(root):
                if fname.lower().endswith(".pdf"):
                    yield os.path.join(root, fname)

    for pdf_path in pdf_paths(input_dir):
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        out_path = os.path.join(output_dir, base_name + ".txt")

        if not overwrite and os.path.exists(out_path):
            print(f"[skip] {out_path} exists")
            continue

        try:
            full_text = extract_text_from_single_pdf(pdf_path, column_split_ratio, threshold_ratio)

            # ensure output folder exists (useful when recurse==True and you want a flat output)
            os.makedirs(output_dir, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(full_text)

            print(f"[ok] {pdf_path} -> {out_path}")
        except Exception as e:
            print(f"[err] {pdf_path}: {e}")


def split_pdf_in_memory(input_path: str) -> list:
    doc = fitz.open(input_path)
    pages = []
    for i in range(len(doc)):
        single_page_doc = fitz.open()
        # copy exactly one page into a new in-memory doc
        single_page_doc.insert_pdf(doc, from_page=i, to_page=i)
        pages.append(single_page_doc)
    doc.close()
    return pages


def extract_single_column_page(page) -> str:
    """Extracts plain text from a single-column page."""
    return page.get_text().strip()


def extract_two_column_page(page, column_split_ratio: float = 0.4) -> str:
    """Extracts text from a two-column page: left side first, then right side."""
    mid_x = page.rect.width * column_split_ratio
    left_lines = []
    right_lines = []

    d = page.get_text("dict")
    for block in d.get("blocks", []):
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue

            x0 = min(span["bbox"][0] for span in spans)
            y0 = round(line["bbox"][1], 1)
            text = "".join(span["text"] for span in spans).strip()
            if not text:
                continue

            if x0 < mid_x:
                left_lines.append((y0, x0, text))
            else:
                right_lines.append((y0, x0, text))

    left_lines.sort(key=lambda l: (l[0], l[1]))
    right_lines.sort(key=lambda l: (l[0], l[1]))

    ordered_text = [text for _, _, text in left_lines] + [text for _, _, text in right_lines]
    return "\n".join(ordered_text)


def detect_page_columns(page, column_split_ratio: float = 0.4, threshold_ratio: float = 0.75) -> str:
    """Detects if the page is single or double column based on x0 positions."""
    left_count = 0
    right_count = 0
    mid_x = page.rect.width * column_split_ratio

    d = page.get_text("dict")
    for block in d.get("blocks", []):
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            x0 = min(span["bbox"][0] for span in spans)
            if x0 < mid_x:
                left_count += 1
            else:
                right_count += 1

    total = left_count + right_count
    if total == 0:
        return "single"

    left_ratio = left_count / total
    return "double" if left_ratio < threshold_ratio else "single"


def extract_text_from_single_pdf(input_path: str, column_split_ratio: float = 0.4, threshold_ratio: float = 0.75) -> str:
    """
    Extracts text from a single PDF:
      - First page as single-column
      - Remaining pages auto-detect columns
    """
    extracted_text = []
    pages = split_pdf_in_memory(input_path)

    for i, page_doc in enumerate(pages):
        page = page_doc[0]
        try:
            if i == 0:
                text = extract_single_column_page(page)
            else:
                layout = detect_page_columns(page, column_split_ratio, threshold_ratio)
                if layout == "double":
                    text = extract_two_column_page(page, column_split_ratio)
                else:
                    text = extract_single_column_page(page)
            extracted_text.append(text.strip())
        finally:
            # IMPORTANT: close the single-page document to free memory
            page_doc.close()

    return "\n".join(extracted_text).strip()


def extract_first_page_to_txt(pdf_path: str, output_dir: str) -> str:
    """
    Extracts only the first page of the given PDF to a .txt file in output_dir.
    Returns the path to the created .txt file.
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.splitext(os.path.basename(pdf_path))[0]
    txt_path = os.path.join(output_dir, f"{filename}.txt")

    doc = fitz.open(pdf_path)
    if doc.page_count == 0:
        print(f"⚠️ {filename} is empty, skipping.")
        return ""
    first_page_text = doc[0].get_text()
    doc.close()

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(first_page_text.strip())

    print(f"✅ Saved first page to {txt_path}")
    return txt_path


# ------------------------- Command-line entry -------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch extract text from PDFs in a folder (auto 1/2-column) into .txt files."
    )
    parser.add_argument("input_dir", help="Folder with PDFs")
    parser.add_argument("output_dir", help="Folder to write .txt files")
    parser.add_argument("--column-split", type=float, default=0.4,
                        help="Relative split position for 2-column pages (0..1, default 0.4)")
    parser.add_argument("--threshold", type=float, default=0.75,
                        help="Left/total ratio below which a page is treated as 2 columns (default 0.75)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .txt files")
    parser.add_argument("--recurse", action="store_true", help="Process PDFs in subfolders recursively")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Input not found or not a directory: {args.input_dir}")
        return

    extract_text_from_pdf(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        column_split_ratio=args.column_split,
        threshold_ratio=args.threshold,
        overwrite=args.overwrite,
        recurse=args.recurse
    )

if __name__ == "__main__":
    main()
