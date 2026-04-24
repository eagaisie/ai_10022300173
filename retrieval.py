# Emmanuel Ato Gaisie
# 10022300173
# CS4241

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ALL_CHUNKS_PATH = DATA_DIR / "all_chunks.json"


def embed_text_chunks(
    chunks: list[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    cache_dir: str = ".hf_cache",
) -> tuple[np.ndarray, SentenceTransformer]:
    """
    Embed a list of text chunks using sentence-transformers.
    Returns:
      - embeddings: shape (n_chunks, embedding_dim), dtype float32
      - model: loaded SentenceTransformer model (reusable for query embedding)
    """
    if not chunks:
        raise ValueError("chunks list is empty.")

    model = SentenceTransformer(model_name, cache_folder=cache_dir)
    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")
    return embeddings, model


def embed_chunks_to_numpy(
    chunks: list[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    cache_dir: str = ".hf_cache",
) -> np.ndarray:
    """
    Embed a list of text chunks and return only the NumPy matrix.
    Shape: (num_chunks, embedding_dim)
    """
    embeddings, _ = embed_text_chunks(chunks, model_name=model_name, cache_dir=cache_dir)
    return embeddings


def numpy_cosine_top_k(
    query_vector: np.ndarray,
    document_matrix: np.ndarray,
    k: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pure NumPy cosine top-k retrieval.
    Args:
      - query_vector: shape (d,) or (1, d)
      - document_matrix: shape (n, d)
      - k: number of top matches
    Returns:
      - top_indices: shape (k,), indices of best-matching chunks
      - top_scores: shape (k,), cosine similarity scores
    """
    if document_matrix.ndim != 2 or document_matrix.shape[0] == 0:
        raise ValueError("document_matrix must be a non-empty 2D array.")
    if k <= 0:
        raise ValueError("k must be > 0.")

    q = np.asarray(query_vector, dtype=np.float32)
    docs = np.asarray(document_matrix, dtype=np.float32)

    if q.ndim == 2:
        if q.shape[0] != 1:
            raise ValueError("query_vector with 2D shape must be (1, d).")
        q = q[0]
    elif q.ndim != 1:
        raise ValueError("query_vector must be 1D or shape (1, d).")

    if docs.shape[1] != q.shape[0]:
        raise ValueError("Dimension mismatch between query_vector and document_matrix.")

    q_norm = np.linalg.norm(q)
    doc_norms = np.linalg.norm(docs, axis=1)
    denom = np.maximum(doc_norms * q_norm, 1e-12)
    scores = (docs @ q) / denom

    k = min(k, docs.shape[0])
    top_indices = np.argsort(scores)[::-1][:k]
    top_scores = scores[top_indices]
    return top_indices, top_scores


def configure_local_model_cache(cache_dir: str = ".hf_cache") -> None:
    """
    Configure Hugging Face cache directories under the project folder
    to avoid permission issues on restricted home directories.
    """
    base_cache = os.path.abspath(cache_dir)
    os.makedirs(base_cache, exist_ok=True)
    os.environ["HF_HOME"] = base_cache
    os.environ["XDG_CACHE_HOME"] = base_cache
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(base_cache, "hub")
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(base_cache, "transformers")


def build_faiss_cosine_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS index for cosine similarity search.
    Cosine similarity is achieved by L2-normalizing vectors and using inner product.
    """
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("embeddings must be a non-empty 2D array.")

    normalized = embeddings.copy()
    faiss.normalize_L2(normalized)
    index = faiss.IndexFlatIP(normalized.shape[1])
    index.add(normalized)
    return index


def faiss_top_k_search(
    query: str,
    chunks: list[str],
    model: SentenceTransformer,
    index: faiss.IndexFlatIP,
    k: int = 10,
) -> list[dict[str, Any]]:
    """
    Search top-k most similar chunks for a query using cosine similarity.
    Returns list of dicts with rank, chunk_id, score, and text.
    """
    if not query.strip():
        raise ValueError("query is empty.")
    if not chunks:
        raise ValueError("chunks list is empty.")
    if k <= 0:
        raise ValueError("k must be > 0.")

    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_embedding)
    k = min(k, len(chunks))
    scores, indices = index.search(query_embedding, k)

    results: list[dict[str, Any]] = []
    for rank, (chunk_id, score) in enumerate(zip(indices[0], scores[0]), start=1):
        results.append(
            {
                "rank": rank,
                "chunk_id": int(chunk_id),
                "score": float(score),
                "text": chunks[int(chunk_id)],
            }
        )
    return results


def rerank_with_cross_encoder(
    query: str,
    faiss_results: list[dict[str, Any]],
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n: int = 3,
    cache_dir: str = ".hf_cache",
) -> list[dict[str, Any]]:
    """
    Re-rank FAISS results with a CrossEncoder.
    Typical use: pass FAISS top-10 and return top-3 refined results.
    """
    if not faiss_results:
        return []
    if top_n <= 0:
        raise ValueError("top_n must be > 0.")

    reranker = CrossEncoder(reranker_model_name, cache_folder=cache_dir)
    pairs = [(query, item["text"]) for item in faiss_results]
    rerank_scores = reranker.predict(pairs)

    rescored: list[dict[str, Any]] = []
    for item, rerank_score in zip(faiss_results, rerank_scores):
        updated = dict(item)
        updated["rerank_score"] = float(rerank_score)
        rescored.append(updated)

    rescored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return rescored[: min(top_n, len(rescored))]


def load_chunks_from_json(chunks_path: Path = ALL_CHUNKS_PATH) -> list[str]:
    """Load precomputed chunks from JSON file."""
    if not chunks_path.is_file():
        raise FileNotFoundError(
            f"Chunks file not found: {chunks_path}. Run 1_data_prep.py first."
        )
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    if not isinstance(chunks, list) or not all(isinstance(x, str) for x in chunks):
        raise ValueError(f"Invalid chunks file format at {chunks_path}")
    return chunks


if __name__ == "__main__":
    configure_local_model_cache()

    chunks = load_chunks_from_json()
    query_text = "What are the key priorities in Ghana's 2025 budget statement?"

    chunk_embeddings, embedder = embed_text_chunks(chunks)
    faiss_index = build_faiss_cosine_index(chunk_embeddings)

    top10 = faiss_top_k_search(
        query=query_text,
        chunks=chunks,
        model=embedder,
        index=faiss_index,
        k=10,
    )

    top3_reranked = rerank_with_cross_encoder(
        query=query_text,
        faiss_results=top10,
        top_n=3,
    )

    print(f"Loaded {len(chunks)} chunks from {ALL_CHUNKS_PATH}")
    print(f"Query: {query_text}\n")
    print("Top FAISS results:")
    for item in top10:
        snippet = item["text"][:220].replace("\n", " ")
        print(f"[{item['rank']}] score={item['score']:.4f} | {snippet}...")

    print("\nTop re-ranked results:")
    for i, item in enumerate(top3_reranked, start=1):
        snippet = item["text"][:220].replace("\n", " ")
        print(f"[{i}] rerank_score={item['rerank_score']:.4f} | {snippet}...")