# Emmanuel Ato Gaisie — 10022300173

# RAG system documentation: ambiguous queries, Streamlit parity, and RAG vs pure LLM

This document records how **misleading and ambiguous** questions are handled, how that matches the **Streamlit** app, and how the **retrieval-augmented** path compares to a **pure LLM** baseline.

---

## 1. System overview (Streamlit-aligned)

The interactive app (`app.py`) sends each user message to:

`pipeline.run_numpy_expanded_rag_query(user_query, top_k=3, memory_buffer=...)`

That pipeline:

1. **Expands** the user query with lightweight keyword-based synonyms (`pipeline.expand_user_query`).
2. **Embeds** the expanded query with **sentence-transformers** (same model as chunk embedding).
3. **Retrieves** the **top 3** chunks by **cosine similarity** using **pure NumPy** (`retrieval.numpy_cosine_top_k`) over the document matrix built from `data/all_chunks.json` (produce chunks with `1_data_prep.py` first).
4. Builds a **strict** prompt: answer **only** from supplied context; otherwise reply exactly with *“I do not have enough information in the provided context.”*
5. Calls the **LLM** via HTTP (OpenAI-compatible chat completions), with optional **last-3-turn memory** folded into the prompt.

The **sidebar** in Streamlit shows: original query, expanded query, NumPy scores, retrieved chunks, and the exact prompt sent to the LLM.

---

## 2. How this evaluation was run (same logic as Streamlit)

Because automated tests cannot drive a browser, retrieval and prompting were exercised with the **same function** the UI calls:

- **Script:** `ambiguous_query_eval.py`
- **RAG path:** `run_numpy_expanded_rag_query(..., memory_buffer=[])` (matches a fresh chat with cleared memory).
- **Pure LLM baseline:** `adversarial_eval.call_pure_llm(query)` — single user message, temperature **0.2**, instruction to answer **even if unsure** (this **increases** the risk of confident guesses on ambiguous input).

Machine-readable output:

- `logs/ambiguous_query_eval_20260421_042550.json` (relative to project root; regenerate with `python ambiguous_query_eval.py`).

**Environment note (this run):** `LLM_API_KEY` / `LLM_MODEL` were **not** set, so **both** the RAG final answer step and the pure LLM call returned the same failure string. **Retrieval still ran**: expanded queries, chunk IDs, NumPy scores, and log files were produced. To capture full natural-language answers for your report, set:

```bash
export LLM_API_KEY="..."
export LLM_MODEL="gpt-4o-mini"   # or your grader-approved model
# optional:
export LLM_API_URL="https://api.openai.com/v1/chat/completions"
python ambiguous_query_eval.py
```

Then paste or diff the new JSON next to this section.

---

## 3. Ambiguous and misleading query battery

| ID | Query | Intent |
|----|--------|--------|
| `ambiguous_referent` | What did **he** promise about taxes in the document? | No referent for “he”; tests grounding vs invention. |
| `misleading_presupposition` | When did construction of the **lunar research facility** begin under the 2025 budget? | False presupposition (no moon base in budget). |
| `underspecified` | Is it enough? | No subject; weak semantic match expected. |
| `false_timeline` | Who won the Ghana presidential election in **2030** …? | Future year; CSV has past years only. |
| `vague_overlap` | What about the debt and the **other thing**? | Undefined “other thing”; partial keyword hit on “debt”. |

---

## 4. Recorded results (run `20260421_042550`)

### 4.1 Retrieval layer (RAG only)

The pure LLM baseline **does not** use these scores or chunks; they illustrate what context the **strict** generator *would* be allowed to use.

| ID | Expanded query (abridged) | NumPy top-3 cosine scores | Comment |
|----|---------------------------|---------------------------|---------|
| `ambiguous_referent` | …Related terms: taxation, levy, revenue mobilization. | **0.449**, **0.410**, **0.393** | Moderate similarity; chunks are budget/tax-adjacent but may not identify “he”. |
| `misleading_presupposition` | …fiscal policy, government spending, revenue, allocation. | **0.367**, **0.357**, **0.355** | “Budget” expansion; retrieved fiscal tables/appendices, **not** lunar construction. |
| `underspecified` | (unchanged) | **0.188**, **0.183**, **0.179** | **Low** scores: embedding search is uncertain; context may be arbitrary social-spend lines. |
| `false_timeline` | …vote, candidate, party, results. | **0.488**, **0.471**, **0.450** | Higher scores: election/budget theme overlap, but **no 2030** in corpus. |
| `vague_overlap` | …public debt, borrowing, liabilities. | **0.445**, **0.436**, **0.433** | Debt-related passages retrieved; “other thing” still undefined. |

**First retrieved snippet (per query, truncated in JSON):**

- **ambiguous_referent:** Revenue measures / fiscal target for 2025…  
- **misleading_presupposition:** Revenue projections and project tables (no moon base).  
- **underspecified:** Social programme spend lines (LEAP, school feeding, etc.).  
- **false_timeline:** Electoral Commission / Ghana Card / budget title chunk mix.  
- **vague_overlap:** Domestic debt / bond reopening / reserve coverage language.

### 4.2 Generator layer (RAG vs pure LLM) — expected behaviour when API is enabled

| Path | Expected behaviour on these queries |
|------|--------------------------------------|
| **RAG (strict)** | If the **answer string** is not clearly supported by the **three** retrieved passages, the model should output the **exact refusal** sentence. That is desirable for `underspecified`, `misleading_presupposition`, and `false_timeline` even when scores are mid-range, because **similarity ≠ entailment**. |
| **Pure LLM** | With “answer even if unsure”, the model often **invents** a plausible entity, date, or event. That is the key **contrast** for the assignment: higher **hallucination risk** without retrieval + strict policy. |

### 4.3 What we observed in the logged run (no API key)

| Query ID | RAG generator output | Pure LLM output |
|------------|---------------------|-----------------|
| All five | `API call failed: Missing LLM_API_KEY environment variable.` | `Pure LLM call failed: Missing LLM_API_KEY environment variable.` |

So for **this** execution, the **comparison of final natural-language answers** is **pending** API credentials. The **retrieval contrast** (scores + chunk focus) is still valid: the RAG stack **always** conditions generation on explicit context once the LLM step runs; the baseline **never** sees those chunks.

---

## 5. How to reproduce in Streamlit

1. Ensure `data/all_chunks.json` exists (`python 1_data_prep.py`).
2. Activate the project venv, e.g. `source env/bin/activate`.
3. Set `LLM_API_KEY` and `LLM_MODEL`.
4. Run: `streamlit run app.py`
5. Paste each query from section 3; compare sidebar (original vs expanded, scores, chunks, prompt) with assistant reply.

---

## 6. Files reference

| File | Role |
|------|------|
| `app.py` | Streamlit chat + sidebar artefact display |
| `pipeline.py` | `run_numpy_expanded_rag_query`, expansion, prompt, LLM HTTP |
| `retrieval.py` | Embeddings, `numpy_cosine_top_k` |
| `ambiguous_query_eval.py` | Batch eval mirroring the app |
| `adversarial_eval.py` | Pure LLM helper + older FAISS-based battery |
| `1_data_prep.py` | Builds `all_chunks.json` |

---

## 7. Summary

- **Ambiguous** queries stress-test **retrieval quality** (e.g. very low scores for “Is it enough?”) and **policy** (strict answer-from-context).
- **Misleading** queries show whether the system **imports a false premise** into the answer; strict RAG is designed to **refuse** when the premise is absent from context, while a **permissive** pure LLM often **confabulates**.
- This document’s **quantitative** retrieval numbers come from **run `20260421_042550`**. **Qualitative** answer comparison should be refreshed after setting API keys and re-running `ambiguous_query_eval.py` or manual Streamlit trials.

---

## 8. Quick start (environment)

```bash
python3 -m venv env
source env/bin/activate   # Windows: env\Scripts\activate
pip install -r requirements.txt
python 1_data_prep.py      # builds data/all_chunks.json (gitignored JSON outputs)
export LLM_API_KEY="..."
export LLM_MODEL="gpt-4o-mini"
streamlit run app.py
```
