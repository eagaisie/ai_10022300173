# Emmanuel Ato Gaisie
# 10022300173
# CS4241

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import PyPDF2


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CSV_PATH = DATA_DIR / "Ghana_Election_Result.csv"
PDF_PATH = DATA_DIR / "2025-Budget-Statement-and-Economic-Policy_v4 (1).pdf"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)


def log_message(message: str, log_path: Path) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from a PDF using PyPDF2."""
    parts: list[str] = []
    with pdf_path.open("rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                parts.append(text)
    return "\n\n".join(parts)


def extract_csv_text(csv_path: Path) -> str:
    """Extract CSV content into a paragraph-like text string using pandas."""
    df = pd.read_csv(csv_path)
    rows: list[str] = []
    for _, row in df.iterrows():
        fields = [f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])]
        rows.append(" | ".join(fields))
    # Double line breaks create paragraph-like boundaries for semantic chunking.
    return "\n\n".join(rows)


def semantic_chunk_text(text: str, source_file: str) -> list[dict[str, str]]:
    """
    Semantic chunking based on paragraph boundaries.
    Splits on:
      - double line breaks (\n\n), or
      - paragraph markers like 'Paragraph 3:', 'Para 4:', etc.
    """
    if not text.strip():
        return []

    # Normalize line endings first.
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    # Split by double line breaks OR paragraph markers.
    pattern = r"\n\s*\n|(?=\b(?:Paragraph|Para)\s*\d+\s*[:.-])"
    raw_chunks = re.split(pattern, cleaned)

    chunks: list[dict[str, str]] = []
    for chunk in raw_chunks:
        normalized_chunk = re.sub(r"\s+", " ", chunk).strip()
        if normalized_chunk:
            chunks.append(
                {
                    "text": normalized_chunk,
                    "metadata": f"source={source_file}",
                }
            )
    return chunks


def extract_and_semantic_chunk(pdf_path: Path, csv_path: Path) -> list[dict[str, str]]:
    """Extract text from both files and return semantic chunks with metadata."""
    pdf_text = extract_pdf_text(pdf_path)
    csv_text = extract_csv_text(csv_path)

    pdf_chunks = semantic_chunk_text(pdf_text, source_file=pdf_path.name)
    csv_chunks = semantic_chunk_text(csv_text, source_file=csv_path.name)
    return pdf_chunks + csv_chunks


if __name__ == "__main__":
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"semantic_chunking_{run_id}.log"

    log_message("Starting semantic chunking experiment", log_path)
    log_message(f"PDF path: {PDF_PATH}", log_path)
    log_message(f"CSV path: {CSV_PATH}", log_path)

    all_chunks = extract_and_semantic_chunk(PDF_PATH, CSV_PATH)
    log_message(f"Total semantic chunks: {len(all_chunks)}", log_path)

    pdf_text = extract_pdf_text(PDF_PATH)
    csv_text = extract_csv_text(CSV_PATH)
    log_message(f"Extracted PDF characters: {len(pdf_text):,}", log_path)
    log_message(f"Extracted CSV characters: {len(csv_text):,}", log_path)

    if all_chunks:
        log_message("First chunk example:", log_path)
        log_message(str(all_chunks[0]), log_path)

    sample_path = LOG_DIR / f"semantic_chunking_sample_{run_id}.json"
    sample_payload = {
        "total_chunks": len(all_chunks),
        "sample_chunks": all_chunks[:5],
    }
    sample_path.write_text(
        json.dumps(sample_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log_message(f"Saved sample chunk payload to: {sample_path}", log_path)
    log_message(f"Experiment log saved to: {log_path}", log_path)
