# EMMANUEL ATO GAISIE 10022300173
"""
Load election CSV and budget PDF from ./data, extract text, and chunk with a
manual sliding window over whitespace-separated words (treated as tokens).
Chunk size: 500 words, overlap: 50 words. No LangChain / LlamaIndex / RAG libs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import PyPDF2

# Paths relative to this script’s directory
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DEFAULT_CSV = DATA_DIR / "Ghana_Election_Result.csv"
# Set to your PDF filename under data/ when present
DEFAULT_PDF = DATA_DIR / "2025-Budget-Statement-and-Economic-Policy_v4 (1).pdf"

CHUNK_SIZE = 500
OVERLAP = 50


def read_csv_as_text(csv_path: Path) -> str:
    """Read CSV with pandas and flatten to one continuous text blob."""
    df = pd.read_csv(csv_path)
    lines: list[str] = []
    for _, row in df.iterrows():
        parts = [f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])]
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def read_pdf_as_text(pdf_path: Path) -> str:
    """Extract all page text with PyPDF2 (manual loop, no RAG helpers)."""
    parts: list[str] = []
    with pdf_path.open("rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                parts.append(t)
    return "\n".join(parts)


def sliding_window_word_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
) -> list[str]:
    """
    Split text on whitespace into words, then emit overlapping windows.
    Each chunk is up to `chunk_size` words; consecutive windows share `overlap` words.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks: list[str] = []
    i = 0
    while i < len(words):
        window = words[i : i + chunk_size]
        chunks.append(" ".join(window))
        if len(window) < chunk_size:
            break
        i += step
    return chunks


def chunk_source(label: str, text: str) -> list[str]:
    chunks = sliding_window_word_chunks(text)
    print(f"{label}: {len(text):,} chars, {len(text.split()):,} words -> {len(chunks)} chunks")
    return chunks


def main() -> None:
    csv_path = DEFAULT_CSV
    pdf_path = DEFAULT_PDF

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    csv_text = read_csv_as_text(csv_path)
    csv_chunks = chunk_source("CSV", csv_text)

    pdf_chunks: list[str] = []
    if pdf_path.is_file():
        pdf_text = read_pdf_as_text(pdf_path)
        pdf_chunks = chunk_source("PDF", pdf_text)
    else:
        print(f"PDF skipped (file not found): {pdf_path}")

    # Example: first chunk preview only (avoid dumping huge output)
    if csv_chunks:
        preview = csv_chunks[0][:400] + ("…" if len(csv_chunks[0]) > 400 else "")
        print("\nFirst CSV chunk preview:\n", preview, sep="")


if __name__ == "__main__":
    main()
