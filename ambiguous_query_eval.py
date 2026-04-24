# Emmanuel Ato Gaisie
# 10022300173
# CS4241

"""
Evaluate ambiguous / misleading queries the same way the Streamlit app does:
  RAG path: pipeline.run_numpy_expanded_rag_query (expand → embed → NumPy top-3 → strict prompt → LLM)
  Baseline: pure LLM (no retrieval), permissive prompt.

Writes machine-readable JSON under logs/ for inclusion in documentation.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from adversarial_eval import call_pure_llm
from pipeline import run_numpy_expanded_rag_query

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Ambiguous (unclear referent), misleading (false presupposition), or underspecified
AMBIGUOUS_QUERIES = [
    {
        "id": "ambiguous_referent",
        "query": "What did he promise about taxes in the document?",
        "note": "No antecedent for 'he'; forces guessing or refusal.",
    },
    {
        "id": "misleading_presupposition",
        "query": "When did construction of the lunar research facility begin under the 2025 budget?",
        "note": "Assumes a 'lunar facility' exists in the budget text.",
    },
    {
        "id": "underspecified",
        "query": "Is it enough?",
        "note": "No subject; retrieval may latch onto arbitrary numeric passages.",
    },
    {
        "id": "false_timeline",
        "query": "Who won the Ghana presidential election in 2030 and by how many votes?",
        "note": "Future year not in dataset; tests fabrication vs abstention.",
    },
    {
        "id": "vague_overlap",
        "query": "What about the debt and the other thing?",
        "note": "Vague; 'other thing' is undefined.",
    },
]


def run_comparison() -> dict:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = LOG_DIR / f"ambiguous_query_eval_{timestamp}.json"
    records = []

    for item in AMBIGUOUS_QUERIES:
        q = item["query"]
        rag = run_numpy_expanded_rag_query(q, top_k=3, memory_buffer=[])
        try:
            pure = call_pure_llm(q)
            pure_ok = True
        except Exception as e:
            pure = f"Pure LLM call failed: {e}"
            pure_ok = False

        top_snippets = [t[:200].replace("\n", " ") + "…" for t in rag.get("retrieved_text_chunks", [])]
        records.append(
            {
                **item,
                "expanded_query": rag.get("expanded_query"),
                "numpy_similarity_scores": rag.get("numpy_similarity_scores"),
                "retrieved_chunk_snippets": top_snippets,
                "rag_response": rag.get("final_response"),
                "rag_api_ok": rag.get("api_call_success"),
                "pure_llm_response": pure,
                "pure_llm_ok": pure_ok,
                "rag_log": rag.get("log_file"),
            }
        )

    report = {
        "methodology": (
            "RAG uses run_numpy_expanded_rag_query (same as Streamlit app): lexical query expansion, "
            "sentence-transformer embeddings, NumPy cosine top-3 over all_chunks.json, strict "
            "context-only prompt with optional memory (empty here). Pure LLM: single user message, "
            "temperature 0.2, instruction to answer even if unsure (encourages hallucination under ambiguity)."
        ),
        "run_timestamp": timestamp,
        "records": records,
    }
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"path": str(out_path), "report": report}


if __name__ == "__main__":
    out = run_comparison()
    print(f"Wrote: {out['path']}")
