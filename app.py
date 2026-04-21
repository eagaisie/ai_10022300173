from __future__ import annotations

import json

import streamlit as st

from pipeline import run_rag_query


st.set_page_config(page_title="Custom RAG Demo", layout="wide")
st.title("Custom Retrieval + Prompting Pipeline")
st.caption("Manual FAISS retrieval with strict context-grounded prompting.")

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
            result = run_rag_query(
                user_query,
                top_k=3,
                memory_buffer=st.session_state.memory_buffer,
            )
        st.markdown(result["final_response"])

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
        st.info("Run a query to view retrieved chunks, scores, and prompt.")
    else:
        st.subheader("Similarity Scores")
        st.json(latest["similarity_scores"])

        st.subheader("Retrieved Chunks")
        for i, doc in enumerate(latest["retrieved_documents"], start=1):
            label = f"Chunk {i} | id={doc['chunk_id']} | score={doc['score']:.4f}"
            with st.expander(label):
                st.write(doc["text"])

        st.subheader("Final Prompt Sent to LLM")
        st.code(latest["exact_prompt_used"])

        st.subheader("Logs")
        st.write(f"Log file: `{latest['log_file']}`")
        st.download_button(
            label="Download Full Result JSON",
            data=json.dumps(latest, indent=2, ensure_ascii=False),
            file_name="rag_result.json",
            mime="application/json",
        )
