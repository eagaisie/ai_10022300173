from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib import error, request

from retrieval import (
    build_faiss_cosine_index,
    configure_local_model_cache,
    embed_text_chunks,
    faiss_top_k_search,
    load_chunks_from_json,
    numpy_cosine_top_k,
)

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)


def _log(message: str, log_path: Path) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def format_memory_buffer(memory_buffer: list[dict[str, str]], max_turns: int = 3) -> str:
    """
    Keep only last max_turns interactions and format as text for prompt inclusion.
    Each interaction item is expected to have keys: user, assistant.
    """
    recent = memory_buffer[-max_turns:]
    if not recent:
        return "No prior conversation."

    lines: list[str] = []
    for i, turn in enumerate(recent, start=1):
        user_text = turn.get("user", "").strip()
        assistant_text = turn.get("assistant", "").strip()
        lines.append(
            f"[Turn {i}]\nUser: {user_text}\nAssistant: {assistant_text}"
        )
    return "\n\n".join(lines)


def build_grounded_prompt(
    user_query: str,
    retrieved_docs: list[dict[str, Any]],
    memory_buffer: list[dict[str, str]] | None = None,
) -> str:
    context_blocks = []
    for i, doc in enumerate(retrieved_docs, start=1):
        context_blocks.append(f"[Document {i}]\n{doc['text']}")
    context_text = "\n\n".join(context_blocks)

    memory_text = format_memory_buffer(memory_buffer or [], max_turns=3)

    prompt = (
        "You are a strict QA assistant.\n"
        "You MUST answer ONLY using the provided context documents.\n"
        "If the answer is not explicitly present in the context, respond exactly with:\n"
        "\"I do not have enough information in the provided context.\"\n"
        "Do not use prior knowledge, do not infer beyond context, and do not hallucinate.\n\n"
        "Conversation Memory (last 3 turns, for reference only):\n"
        f"{memory_text}\n\n"
        f"User Question:\n{user_query}\n\n"
        f"Context Documents:\n{context_text}\n\n"
        "Return a concise answer grounded only in the context."
    )
    return prompt


def _normalize_env_string(value: str) -> str:
    """Strip whitespace and a single pair of surrounding quotes (common copy/paste mistakes)."""
    s = value.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    return s


GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"


def build_llm_chat_headers(api_key: str) -> dict[str, str]:
    """
    Headers for Groq chat POSTs. Groq sits behind Cloudflare; the default
    ``Python-urllib/…`` user-agent is often blocked (HTTP 403, error **1010**).
    Override with env ``LLM_HTTP_USER_AGENT`` if needed.
    """
    ua = _normalize_env_string(
        os.getenv(
            "LLM_HTTP_USER_AGENT",
            "Mozilla/5.0 (compatible; RAG-Assignment/1.0; +https://streamlit.io)",
        )
    )
    if not ua:
        ua = "Mozilla/5.0 (compatible; RAG-Assignment/1.0)"
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "User-Agent": ua,
    }


def _first_llm_api_key_from_env() -> tuple[str, str]:
    """Return (api_key, env_var_name) from the first non-empty key candidate."""
    for name in ("LLM_API_KEY", "GROQ_API_KEY"):
        v = _normalize_env_string(os.getenv(name, ""))
        if v:
            return v, name
    return "", ""


def resolve_llm_runtime_config() -> tuple[str, str, str]:
    """
    Resolve (api_key, model, chat_completions_url) for **Groq** chat completions
    (OpenAI-compatible JSON over HTTP).

    Key order: ``LLM_API_KEY``, ``GROQ_API_KEY`` (either may hold your ``gsk_`` secret).

    URL: ``LLM_API_URL`` if set, else Groq’s default chat-completions URL.
    """
    api_key, _key_source = _first_llm_api_key_from_env()
    model = _normalize_env_string(os.getenv("LLM_MODEL", ""))
    api_url = _normalize_env_string(os.getenv("LLM_API_URL", "")) or GROQ_CHAT_COMPLETIONS_URL
    return api_key, model, api_url


def call_llm_api(prompt: str, log_path: Path) -> str:
    """
    Calls Groq’s OpenAI-compatible chat-completions endpoint.
    Required env vars:
      - LLM_API_KEY or GROQ_API_KEY (Bearer token, usually ``gsk_...``)
      - LLM_MODEL (Groq model id, e.g. ``llama-3.3-70b-versatile``)
    Optional env vars:
      - LLM_API_URL (default: Groq chat completions URL)
    """
    api_key, model, api_url = resolve_llm_runtime_config()

    if not api_key:
        raise ValueError("Missing LLM_API_KEY or GROQ_API_KEY environment variable.")
    if not model:
        raise ValueError("Missing LLM_MODEL environment variable.")

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }
    data = json.dumps(payload).encode("utf-8")

    _log(f"API call -> {api_url} using model '{model}'", log_path)
    req = request.Request(
        api_url,
        data=data,
        headers=build_llm_chat_headers(api_key),
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8")
    except error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM API HTTPError {e.code}: {err_body}") from e
    except error.URLError as e:
        raise RuntimeError(f"LLM API connection error: {e}") from e

    parsed = json.loads(body)
    content = parsed["choices"][0]["message"]["content"]
    return content.strip()


def expand_user_query(user_query: str) -> str:
    """
    Lightweight manual query expansion to improve retrieval recall.
    """
    expansions = {
        "budget": ["fiscal policy", "government spending", "revenue", "allocation"],
        "election": ["vote", "candidate", "party", "results"],
        "inflation": ["consumer prices", "price growth", "CPI"],
        "debt": ["public debt", "borrowing", "liabilities"],
        "tax": ["taxation", "levy", "revenue mobilization"],
    }
    lowered = user_query.lower()
    terms: list[str] = []
    for keyword, synonyms in expansions.items():
        if keyword in lowered:
            terms.extend(synonyms)

    if not terms:
        return user_query
    return f"{user_query}. Related terms: {', '.join(terms)}."


def run_numpy_expanded_rag_query(
    user_query: str,
    top_k: int = 3,
    memory_buffer: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """
    Pipeline:
      1) expand query
      2) embed expanded query
      3) run pure NumPy cosine top-k search
      4) inject chunks into strict anti-hallucination prompt
      5) return final response + retrieved chunks + scores + exact prompt
    """
    if not user_query.strip():
        raise ValueError("user_query cannot be empty.")

    memory_buffer = memory_buffer or []
    log_path = LOG_DIR / f"numpy_rag_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    _log("Starting NumPy expanded RAG pipeline", log_path)
    _log(f"Original query: {user_query}", log_path)

    configure_local_model_cache()
    chunks = load_chunks_from_json()
    _log(f"Loaded {len(chunks)} chunks", log_path)

    expanded_query = expand_user_query(user_query)
    _log(f"Expanded query: {expanded_query}", log_path)

    doc_embeddings, embedder = embed_text_chunks(chunks)
    query_embedding = embedder.encode([expanded_query], convert_to_numpy=True).astype("float32")

    top_k = max(1, top_k)
    top_indices, top_scores = numpy_cosine_top_k(
        query_vector=query_embedding,
        document_matrix=doc_embeddings,
        k=top_k,
    )

    retrieved_docs = []
    for rank, (idx, score) in enumerate(zip(top_indices, top_scores), start=1):
        retrieved_docs.append(
            {
                "rank": rank,
                "chunk_id": int(idx),
                "score": float(score),
                "text": chunks[int(idx)],
            }
        )
    _log(
        "NumPy retrieved chunk IDs: " + ", ".join(str(item["chunk_id"]) for item in retrieved_docs),
        log_path,
    )
    _log(
        "NumPy similarity scores: " + ", ".join(f"{item['score']:.4f}" for item in retrieved_docs),
        log_path,
    )

    _log(f"Memory turns provided: {len(memory_buffer)}", log_path)
    exact_prompt = build_grounded_prompt(user_query, retrieved_docs, memory_buffer=memory_buffer)
    _log("Built strict grounded prompt", log_path)

    api_call_ok = True
    try:
        final_response = call_llm_api(exact_prompt, log_path)
        _log("LLM response received", log_path)
    except Exception as e:
        api_call_ok = False
        final_response = f"API call failed: {e}"
        _log(final_response, log_path)

    numpy_scores = [item["score"] for item in retrieved_docs]
    chunk_texts = [item["text"] for item in retrieved_docs]

    result = {
        "original_query": user_query,
        "expanded_query": expanded_query,
        "final_llm_response": final_response,
        "final_response": final_response,
        "retrieved_text_chunks": chunk_texts,
        "numpy_similarity_scores": numpy_scores,
        "similarity_scores": numpy_scores,
        "retrieved_documents": retrieved_docs,
        "exact_prompt_sent_to_llm": exact_prompt,
        "exact_prompt_used": exact_prompt,
        "api_call_success": api_call_ok,
        "memory_used": memory_buffer[-3:],
        "log_file": str(log_path),
    }
    result_path = LOG_DIR / f"numpy_rag_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    _log(f"Saved structured result to {result_path}", log_path)
    return result


def run_rag_query(
    user_query: str,
    top_k: int = 3,
    memory_buffer: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """
    End-to-end RAG query:
      1) embed chunks + build FAISS
      2) embed query + retrieve top-k
      3) build strict grounded prompt
      4) execute LLM API call
      5) return response + docs + scores + exact prompt
    """
    if not user_query.strip():
        raise ValueError("user_query cannot be empty.")

    log_path = LOG_DIR / f"rag_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    _log("Starting RAG pipeline", log_path)
    _log(f"Input query: {user_query}", log_path)
    _log("Loading chunks from JSON", log_path)

    configure_local_model_cache()
    chunks = load_chunks_from_json()
    _log(f"Loaded {len(chunks)} chunks", log_path)

    _log("Embedding all chunks", log_path)
    chunk_embeddings, embedder = embed_text_chunks(chunks)
    _log(f"Embeddings shape: {chunk_embeddings.shape}", log_path)

    _log("Building FAISS cosine index", log_path)
    index = build_faiss_cosine_index(chunk_embeddings)

    _log(f"Retrieving top {top_k} chunks with FAISS", log_path)
    retrieved = faiss_top_k_search(
        query=user_query,
        chunks=chunks,
        model=embedder,
        index=index,
        k=top_k,
    )
    retrieved = retrieved[:top_k]
    _log(
        "Retrieved chunk IDs: " + ", ".join(str(item["chunk_id"]) for item in retrieved),
        log_path,
    )
    _log(
        "Similarity scores: " + ", ".join(f"{item['score']:.4f}" for item in retrieved),
        log_path,
    )

    memory_buffer = memory_buffer or []
    _log(f"Memory turns provided: {len(memory_buffer)}", log_path)
    _log("Building strict grounded prompt with memory buffer", log_path)
    exact_prompt = build_grounded_prompt(user_query, retrieved, memory_buffer)

    _log("Executing LLM API call", log_path)
    api_call_ok = True
    try:
        final_response = call_llm_api(exact_prompt, log_path)
        _log("LLM response received", log_path)
    except Exception as e:
        api_call_ok = False
        final_response = f"API call failed: {e}"
        _log(final_response, log_path)

    result = {
        "final_response": final_response,
        "api_call_success": api_call_ok,
        "memory_used": memory_buffer[-3:],
        "retrieved_documents": retrieved,
        "similarity_scores": [item["score"] for item in retrieved],
        "exact_prompt_used": exact_prompt,
        "log_file": str(log_path),
    }

    result_path = LOG_DIR / f"rag_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    _log(f"Saved structured result to {result_path}", log_path)
    _log("RAG pipeline completed", log_path)
    return result


if __name__ == "__main__":
    demo_query = "What are the key priorities in Ghana's 2025 budget statement?"
    output = run_rag_query(demo_query, top_k=3)

    print("\n=== FINAL RESPONSE ===")
    print(output["final_response"])
    print(f"\nAPI call success: {output['api_call_success']}")
    print("\n=== TOP 3 SCORES ===")
    for i, score in enumerate(output["similarity_scores"], start=1):
        print(f"{i}. {score:.4f}")
    print(f"\nLogs written to: {output['log_file']}")
