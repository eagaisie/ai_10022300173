from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import streamlit as st

from pipeline import run_numpy_expanded_rag_query


def _groq_llm_failure_hint(assistant_text: str) -> str:
    """Tailored follow-up; the assistant bubble already has the raw ``API call failed: …`` line."""
    low = assistant_text.lower()
    if "missing llm_api_key" in low or "missing groq_api_key" in low:
        return (
            "**Secrets missing on this app:** **Manage app** → **Settings** → **Secrets** — add "
            '`GROQ_API_KEY = "gsk_..."` and `LLM_MODEL = "…"` (a valid Groq model id), **Save**, then **Reboot app**.'
        )
    if "401" in assistant_text or "invalid_api_key" in low or "incorrect api key" in low:
        return (
            "**Groq rejected the API key** (wrong or revoked). Create a new key at "
            "[console.groq.com](https://console.groq.com/), paste it into **Secrets**, **Reboot app**."
        )
    if "429" in assistant_text or "rate limit" in low or "quota" in low or "too many requests" in low:
        return "**Rate limit / quota:** wait briefly, try again, or check usage in the Groq console."
    if "403" in assistant_text and "1010" in assistant_text:
        return (
            "**Cloudflare 1010** often means the request was blocked before it reached Groq (common with "
            "default Python clients). The app now sends a normal **User-Agent**; redeploy, retry. If it "
            "persists on **Streamlit Cloud**, try running **locally**, set **`LLM_HTTP_USER_AGENT`** in "
            "Secrets to a current browser UA string, or ask Groq support whether your Cloud egress IP is allowed."
        )
    if "403" in assistant_text:
        return (
            "**HTTP 403:** key scope, account restriction, or edge block. Confirm the key at "
            "[console.groq.com](https://console.groq.com/), check **Manage app → Logs**, try from your laptop "
            "with the same Secrets to see if only Cloud is affected."
        )
    if "model" in low and ("not found" in low or "does not exist" in low or "invalid_model" in low):
        return (
            "**Unknown or retired model id:** set `LLM_MODEL` to a name from "
            "[Groq’s model list](https://console.groq.com/docs/models), **Save**, **Reboot app**."
        )
    return (
        "**LLM step failed** — read the **assistant** line above (starts with `API call failed:`). "
        "For server-side detail: **Manage app** → **Logs**."
    )


st.set_page_config(
    page_title="RAG Research Desk",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _inject_app_styles() -> None:
    st.markdown(
        """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,400;0,600;0,700;1,400&display=swap');
  html, body, [data-testid="stAppViewContainer"] {
    font-family: "Plus Jakarta Sans", "Segoe UI", system-ui, sans-serif !important;
  }
  [data-testid="stAppViewContainer"] > .main {
    background: radial-gradient(1200px 600px at 10% -10%, rgba(124, 58, 237, 0.12), transparent),
                radial-gradient(900px 500px at 100% 0%, rgba(236, 72, 153, 0.1), transparent),
                linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%) !important;
  }
  [data-testid="stSidebar"] {
    background: linear-gradient(195deg, #faf5ff 0%, #ede9fe 55%, #e0e7ff 100%) !important;
    border-right: 2px solid rgba(124, 58, 237, 0.25) !important;
  }
  [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.35rem !important;
  }
  div[data-testid="stChatInput"] {
    border-radius: 18px !important;
    box-shadow: 0 4px 24px rgba(124, 58, 237, 0.15) !important;
    border: 2px solid rgba(124, 58, 237, 0.35) !important;
  }
  .stButton > button {
    border-radius: 12px !important;
    font-weight: 600 !important;
    transition: transform 0.12s ease, box-shadow 0.12s ease !important;
  }
  .stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(124, 58, 237, 0.25) !important;
  }
  .stDownloadButton > button {
    border-radius: 12px !important;
    background: linear-gradient(90deg, #7c3aed, #a855f7) !important;
    color: #fff !important;
    border: none !important;
    font-weight: 600 !important;
  }
  [data-testid="stExpander"] details {
    border-radius: 12px !important;
    border: 1px solid rgba(124, 58, 237, 0.2) !important;
    background: rgba(255, 255, 255, 0.75) !important;
  }
  [data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.85);
    padding: 0.75rem 1rem;
    border-radius: 14px;
    border: 1px solid rgba(124, 58, 237, 0.15);
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.06);
  }
</style>
""",
        unsafe_allow_html=True,
    )


_inject_app_styles()


def _coerce_secret_scalar(val: Any) -> str | None:
    """Streamlit / TOML values may not always pass isinstance(..., str); normalize to str."""
    if val is None or isinstance(val, (dict, list, tuple, set)):
        return None
    s = str(val).strip()
    return s or None


def _flatten_streamlit_secrets(node: Any, out: dict[str, str]) -> None:
    """Recursively collect string secrets from nested TOML tables (any depth)."""
    if not isinstance(node, dict):
        return
    for key, val in node.items():
        if not isinstance(key, str) or not key:
            continue
        if isinstance(val, dict):
            _flatten_streamlit_secrets(val, out)
        else:
            s = _coerce_secret_scalar(val)
            if s is not None:
                out[key] = s


def _apply_streamlit_secrets_direct_keys() -> None:
    """Backup path when `dict(st.secrets)` fails or omits keys; uses Streamlit's mapping API."""
    try:
        sec = st.secrets
    except Exception:
        return
    for key in (
        "LLM_API_KEY",
        "GROQ_API_KEY",
        "LLM_MODEL",
        "LLM_API_URL",
        "LLM_HTTP_USER_AGENT",
        "HF_TOKEN",
    ):
        try:
            raw = sec[key]
        except (KeyError, TypeError):
            continue
        s = _coerce_secret_scalar(raw)
        if s:
            os.environ[key] = s


def _apply_streamlit_secrets_to_environ() -> None:
    """Map Streamlit Cloud (TOML) secrets into os.environ for pipeline.py / HF."""
    try:
        flat: dict[str, str] = {}
        _flatten_streamlit_secrets(dict(st.secrets), flat)
        for key, val in flat.items():
            os.environ[key] = val
    except Exception:
        pass
    _apply_streamlit_secrets_direct_keys()


def _apply_local_secrets_toml_file() -> None:
    """
    Fallback when st.secrets is empty or unavailable: read `.streamlit/secrets.toml`
    (same format as Streamlit; gitignored — safe for local dev).
    Only sets keys that are not already present (Cloud / shell exports win).
    """
    path = Path(__file__).resolve().parent / ".streamlit" / "secrets.toml"
    if not path.is_file():
        return
    try:
        import tomllib
    except ImportError:
        return
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except Exception:
        return
    if not isinstance(data, dict):
        return
    flat: dict[str, str] = {}
    _flatten_streamlit_secrets(data, flat)
    for key, val in flat.items():
        os.environ[key] = val


def _synthesize_llm_env_from_aliases() -> None:
    """Ensure LLM_API_KEY is set when only GROQ_API_KEY is provided; optional model aliases."""
    if not os.getenv("LLM_API_KEY", "").strip():
        v = os.getenv("GROQ_API_KEY", "").strip()
        if v:
            os.environ["LLM_API_KEY"] = v
    if not os.getenv("LLM_MODEL", "").strip():
        v = os.getenv("LLM_MODEL_NAME", "").strip()
        if v:
            os.environ["LLM_MODEL"] = v


def _refresh_llm_env_from_streamlit() -> None:
    """Re-read secrets into os.environ (safe to call every rerun / before each chat)."""
    _apply_streamlit_secrets_to_environ()
    _apply_local_secrets_toml_file()
    _synthesize_llm_env_from_aliases()


_refresh_llm_env_from_streamlit()


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

with st.sidebar:
    st.markdown(
        '<span style="font-size:0.7rem;font-weight:800;letter-spacing:0.14em;color:#6d28d9;'
        'text-transform:uppercase;">LLM connection</span>',
        unsafe_allow_html=True,
    )
    _refresh_llm_env_from_streamlit()
    if os.getenv("LLM_API_KEY", "").strip() or os.getenv("GROQ_API_KEY", "").strip():
        st.success("API key is loaded (hidden). Ready to generate.")
    else:
        st.error(
            "**No API key in the environment.**\n\n"
            "- **Streamlit Community Cloud:** **Manage app** → **Settings** → **Secrets** — add "
            "`LLM_API_KEY` (or `GROQ_API_KEY`) and `LLM_MODEL` (Groq model id). **Save**, then **Reboot app**.\n"
            "- **Local `streamlit run`:** create `.streamlit/secrets.toml` (copy from "
            "`.streamlit/secrets.toml.example` in the repo), or `export GROQ_API_KEY=...` and `export LLM_MODEL=...` in the same terminal."
        )
    _model = os.getenv("LLM_MODEL", "").strip() or "(not set — add LLM_MODEL in Secrets)"
    st.markdown(
        f'<p style="margin:0.35rem 0 0 0;font-size:0.85rem;color:#4c1d95;">Model: '
        f'<code style="background:rgba(124,58,237,0.12);padding:2px 8px;border-radius:8px;">{_model}</code></p>',
        unsafe_allow_html=True,
    )
    st.divider()

st.markdown(
    """
<div style="
  background: linear-gradient(115deg, #5b21b6 0%, #7c3aed 38%, #c026d3 72%, #db2777 100%);
  padding: 1.6rem 1.5rem 1.45rem 1.5rem;
  border-radius: 22px;
  margin: 0 0 1rem 0;
  box-shadow: 0 18px 50px rgba(91, 33, 182, 0.35);
">
  <p style="margin:0 0 0.35rem 0;font-size:0.75rem;font-weight:700;letter-spacing:0.2em;
            color: rgba(255,255,255,0.88);text-transform:uppercase;">Ghana budget · elections RAG</p>
  <h1 style="margin:0;color:#fff;font-size:clamp(1.55rem, 3vw, 2.15rem);font-weight:800;
             letter-spacing:-0.03em;line-height:1.15;">RAG Research Desk</h1>
  <p style="margin:0.65rem 0 0 0;color: rgba(255,255,255,0.92);font-size:1.02rem;line-height:1.45;">
    Query expansion, NumPy cosine retrieval, and strict context-only answers — with a short memory of your last turns.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Retrieval", "NumPy · top-3", help="Pure cosine similarity in embedding space")
with c2:
    st.metric("Grounding", "Strict", help="Answers must cite provided chunks only")
with c3:
    st.metric("Memory", "Last 3 turns", help="Recent Q&A pairs appended to the prompt")

if "memory_buffer" not in st.session_state:
    st.session_state.memory_buffer = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "latest_result" not in st.session_state:
    st.session_state.latest_result = None

btn_col, _ = st.columns([1, 2])
with btn_col:
    if st.button("Clear chat & memory", type="primary", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.memory_buffer = []
        st.session_state.latest_result = None
        st.success("Fresh start — chat and memory cleared.")

st.markdown(
    '<p style="color:#64748b;font-size:0.9rem;margin:0.5rem 0 0.75rem 0;">💬 <strong>Conversation</strong></p>',
    unsafe_allow_html=True,
)

# Render chat transcript
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input("Ask about the 2025 budget, fiscal policy, or election data…")
if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        with st.spinner("Retrieving chunks & calling the language model…"):
            try:
                _refresh_llm_env_from_streamlit()
                result = run_numpy_expanded_rag_query(
                    user_query,
                    top_k=3,
                    memory_buffer=st.session_state.memory_buffer,
                )
            except Exception as e:
                hint = (
                    "**Something broke before the answer was ready.**\n\n"
                    "Typical fixes:\n"
                    "- **Streamlit Cloud:** **Manage app** → **Settings** → **Secrets** — set `GROQ_API_KEY` "
                    "(or `LLM_API_KEY`) and `LLM_MODEL` (Groq model id), save, then **Reboot app**.\n"
                    "- **Local:** run `export GROQ_API_KEY=...` and `export LLM_MODEL=...` in the same shell "
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
                    st.warning(_groq_llm_failure_hint(result.get("final_response", "")))

    st.session_state.chat_history.append(
        {"role": "assistant", "content": result["final_response"]}
    )
    st.session_state.memory_buffer.append(
        {"user": user_query, "assistant": result["final_response"]}
    )
    st.session_state.memory_buffer = st.session_state.memory_buffer[-3:]
    st.session_state.latest_result = result

with st.sidebar:
    st.markdown(
        '<span style="font-size:0.7rem;font-weight:800;letter-spacing:0.14em;color:#be185d;'
        'text-transform:uppercase;">Last answer · debug</span>',
        unsafe_allow_html=True,
    )
    latest = st.session_state.latest_result
    if latest is None:
        st.info("Send a message to see expansion, similarity scores, retrieved chunks, and the exact LLM prompt.")
    else:
        st.markdown(
            '<p style="color:#6d28d9;font-weight:700;font-size:0.9rem;margin:0.5rem 0 0.25rem;">Original query</p>',
            unsafe_allow_html=True,
        )
        st.write(latest.get("original_query", ""))

        st.markdown(
            '<p style="color:#6d28d9;font-weight:700;font-size:0.9rem;margin:0.75rem 0 0.25rem;">'
            "Expanded query <small>(embedding)</small></p>",
            unsafe_allow_html=True,
        )
        st.write(latest.get("expanded_query", latest.get("original_query", "")))

        st.markdown(
            '<p style="color:#6d28d9;font-weight:700;font-size:0.9rem;margin:0.75rem 0 0.25rem;">'
            "Cosine scores · top 3</p>",
            unsafe_allow_html=True,
        )
        st.json(latest.get("numpy_similarity_scores", latest.get("similarity_scores", [])))

        st.markdown(
            '<p style="color:#6d28d9;font-weight:700;font-size:0.9rem;margin:0.75rem 0 0.25rem;">'
            "Retrieved chunks</p>",
            unsafe_allow_html=True,
        )
        docs = latest.get("retrieved_documents") or []
        for i, doc in enumerate(docs, start=1):
            label = f"📄 Chunk {i} · id {doc['chunk_id']} · {doc['score']:.4f}"
            with st.expander(label):
                st.write(doc["text"])

        st.markdown(
            '<p style="color:#6d28d9;font-weight:700;font-size:0.9rem;margin:0.75rem 0 0.25rem;">'
            "Prompt sent to the LLM</p>",
            unsafe_allow_html=True,
        )
        st.code(
            latest.get("exact_prompt_sent_to_llm", latest.get("exact_prompt_used", "")),
            language="text",
        )

        st.markdown(
            '<p style="color:#6d28d9;font-weight:700;font-size:0.9rem;margin:0.75rem 0 0.25rem;">Logs</p>',
            unsafe_allow_html=True,
        )
        st.caption(f"Log file: `{latest.get('log_file', '')}`")
        st.download_button(
            label="⬇️ Download result JSON",
            data=json.dumps(latest, indent=2, ensure_ascii=False),
            file_name="rag_result.json",
            mime="application/json",
            use_container_width=True,
        )
