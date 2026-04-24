# Emmanuel Ato Gaisie
# 10022300173
# CS4241

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from urllib import error, request

from pipeline import build_llm_chat_headers, resolve_llm_runtime_config, run_rag_query

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

ADVERSARIAL_QUERIES = [
    "Who won the election in 2030?",
    "What was Ghana's inflation rate in 2032 according to the budget statement?",
    "Which candidate got 99% of votes in 2020?",
    "What does the 2025 budget say about building a moon base?",
]


def call_pure_llm(question: str) -> str:
    """
    Baseline LLM call without retrieval context.
    Uses same env vars as pipeline (Groq via ``resolve_llm_runtime_config``).
    """
    api_key, model, api_url = resolve_llm_runtime_config()

    if not api_key:
        raise ValueError("Missing LLM_API_KEY or GROQ_API_KEY environment variable.")
    if not model:
        raise ValueError("Missing LLM_MODEL environment variable.")

    prompt = (
        "Answer the following user question.\n"
        "If unsure, still provide your best possible answer.\n\n"
        f"Question: {question}"
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }

    req = request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers=build_llm_chat_headers(api_key),
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8")
    except error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Pure LLM HTTPError {e.code}: {err_body}") from e
    except error.URLError as e:
        raise RuntimeError(f"Pure LLM connection error: {e}") from e

    parsed = json.loads(body)
    return parsed["choices"][0]["message"]["content"].strip()


def run_adversarial_comparison() -> dict:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_path = LOG_DIR / f"adversarial_eval_{timestamp}.json"

    records = []
    for query in ADVERSARIAL_QUERIES:
        rag_result = run_rag_query(query, top_k=3, memory_buffer=[])
        try:
            pure_llm_answer = call_pure_llm(query)
            pure_llm_ok = True
        except Exception as e:
            pure_llm_answer = f"Pure LLM call failed: {e}"
            pure_llm_ok = False

        records.append(
            {
                "query": query,
                "rag_final_response": rag_result["final_response"],
                "rag_api_call_success": rag_result["api_call_success"],
                "rag_similarity_scores": rag_result["similarity_scores"],
                "rag_prompt": rag_result["exact_prompt_used"],
                "rag_log_file": rag_result["log_file"],
                "pure_llm_response": pure_llm_answer,
                "pure_llm_call_success": pure_llm_ok,
            }
        )

    report = {
        "run_timestamp": timestamp,
        "num_queries": len(ADVERSARIAL_QUERIES),
        "queries": ADVERSARIAL_QUERIES,
        "records": records,
    }
    run_log_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"report_path": str(run_log_path), "report": report}


if __name__ == "__main__":
    output = run_adversarial_comparison()
    print(f"Saved adversarial comparison report to: {output['report_path']}")
