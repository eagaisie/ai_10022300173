from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import streamlit as st

from pipeline import run_numpy_expanded_rag_query

st.set_page_config(page_title="Custom RAG Demo", layout="wide")


def _apply_streamlit_secrets_to_environ() -> None:
    """Map Streamlit Cloud (TOML) secrets into os.environ for pipeline.py / HF."""
    try:
        for key, val in dict(st.secrets).items():
            if isinstance(val, str) and val.strip():
                os.environ.setdefault(key, val)
            elif isinstance(val, dict):
                for k2, v2 in val.items():
                    if isinstance(v2, str) and v2.strip():
                        os.environ.setdefault(k2, v2)
    except (FileNotFoundError, RuntimeError, TypeError):
        pass


_apply_streamlit_secrets_to_environ()


def _ensure_chunks_for_cloud() -> None:
    """
    On Streamlit Community Cloud, data/all_chunks.json is usually absent (gitignored).
    Build it once per process from the tracked CSV/PDF under data/.
    """
    root = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("data_prep_bootstrap", root / "1_data_prep.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load 1_data_prep.py for chunk bootstrap.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ensure_all_chunks_json(verbose=False)


@st.cache_resource(show_spinner="Preparing search index (first load may take 1–2 minutes)…")
def _bootstrap_index() -> bool:
    _ensure_chunks_for_cloud()
    return True


_bootstrap_index()

st.title("Custom Retrieval + Prompting Pipeline")
st.caption(
    "NumPy cosine retrieval on expanded queries, strict context-grounded prompting, optional memory."
)

if "memory_buffer" not in st.session_state:
    st.session_state.memory_buffer = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "latest_result" not in st.session_state:
    st.session_state.latest_result = None

if st.button("Clear Chat + Memory"):
    st.session_state.chat_history = []
    st.session_state.memory_buffer = []
    st.session_state.latest_result = None
    st.success("Chat and memory buffer cleared.")

# Render chat transcript
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_query = st.chat_input("Ask a question about the election data or budget PDF")
if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        with st.spinner("Running retrieval + generation pipeline..."):
            try:
                result = run_numpy_expanded_rag_query(
                    user_query,
                    top_k=3,
                    memory_buffer=st.session_state.memory_buffer,
                )
            except Exception as e:
                hint = (
                    "**Something broke before the answer was ready.**\n\n"
                    "Typical fixes:\n"
                    "- **Streamlit Cloud:** **Manage app** → **Settings** → **Secrets** — set `LLM_API_KEY` "
                    "and `LLM_MODEL`, save, then **Reboot app**.\n"
                    "- **Local:** run `export LLM_API_KEY=...` and `export LLM_MODEL=...` in the same shell "
                    "as `streamlit run app.py`.\n"
                    "- If the error mentions **chunks** or **all_chunks.json**, wait for the first-load "
                    "spinner to finish, or run `python 1_data_prep.py` locally.\n"
                    "- If it mentions **Hugging Face** / download, add `HF_TOKEN` in Secrets (Cloud) or "
                    "`export HF_TOKEN=...` locally.\n\n"
                    f"**Technical detail:** `{type(e).__name__}: {e}`"
                )
                st.markdown(hint)
                result = {
                    "original_query": user_query,
                    "expanded_query": user_query,
                    "final_response": hint,
                    "numpy_similarity_scores": [],
                    "retrieved_documents": [],
                    "exact_prompt_sent_to_llm": "",
                    "log_file": "",
                    "api_call_success": False,
                }
            else:
                st.markdown(result["final_response"])
                if not result.get("api_call_success", True):
                    st.warning(
                        "The **LLM** step failed (see the assistant text above). On Streamlit Cloud, "
                        "open **Manage app** → **Settings** → **Secrets**, set `LLM_API_KEY` and "
                        "`LLM_MODEL` exactly (no quotes in the TOML value unless they are part of the key), "
                        "save, then **Reboot app**."
                    )

    st.session_state.chat_history.append(
        {"role": "assistant", "content": result["final_response"]}
    )
    st.session_state.memory_buffer.append(
        {"user": user_query, "assistant": result["final_response"]}
    )
    st.session_state.memory_buffer = st.session_state.memory_buffer[-3:]
    st.session_state.latest_result = result

# Sidebar retrieval details for latest turn
with st.sidebar:
    st.header("Retrieval Details")
    latest = st.session_state.latest_result
    if latest is None:
        st.info("Submit a prompt to see query expansion, retrieval, scores, and the exact LLM prompt.")
    else:
        st.subheader("Original query")
        st.write(latest.get("original_query", ""))

        st.subheader("Expanded query (used for embedding)")
        st.write(latest.get("expanded_query", latest.get("original_query", "")))

        st.subheader("NumPy cosine similarity scores (top 3)")
        st.json(latest.get("numpy_similarity_scores", latest.get("similarity_scores", [])))

        st.subheader("Retrieved chunks")
        docs = latest.get("retrieved_documents") or []
        for i, doc in enumerate(docs, start=1):
            label = f"Chunk {i} | id={doc['chunk_id']} | score={doc['score']:.4f}"
            with st.expander(label):
                st.write(doc["text"])

        st.subheader("Final prompt sent to the LLM")
        st.code(
            latest.get("exact_prompt_sent_to_llm", latest.get("exact_prompt_used", "")),
            language="text",
        )

        st.subheader("Logs")
        st.write(f"Log file: `{latest['log_file']}`")
        st.download_button(
            label="Download Full Result JSON",
            data=json.dumps(latest, indent=2, ensure_ascii=False),
            file_name="rag_result.json",
            mime="application/json",
        )
