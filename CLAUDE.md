# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start

### Prerequisites

- Python 3.12+ (tested on 3.13 and 3.14)
- Ollama running locally (https://ollama.com/download)
- Models pulled: `ollama pull qwen2.5:3b && ollama pull nomic-embed-text`

### Ollama service

```bash
# Start Ollama (must be running before ingest.py or app.py)
ollama serve

# Show loaded models, their context size and GPU/CPU placement
ollama ps

# Pull the required models (once)
ollama pull qwen2.5:3b
ollama pull nomic-embed-text

# List installed models
ollama list

# Windows: after installation Ollama runs as a tray service.
# If it does not, start it manually: ollama serve (or open the Ollama desktop app)
```

### Essential Commands

```bash
# Setup (once)
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate   # macOS/Linux
pip install -r requirements-dev.txt   # runtime deps + pytest + ruff

# Build the vector store from PDFs in documents/
python ingest.py

# Rebuild (after new PDFs, or after changing embedding model / chunking / distance space)
python ingest.py --reset
python ingest.py --reset --skip-failed   # ingest readable PDFs even if some fail

# Start the chatbot UI (http://localhost:8501)
streamlit run app.py

# Tests (offline, no Ollama needed) and lint
pytest
ruff check .
ruff format .
```

---

## Architecture Overview

Two-phase RAG pipeline. Every tunable setting is in `config.py` with a `RAG_*` environment override;
`models.py` builds the Ollama models and the Chroma store from it.

### Phase 1: Ingestion (`ingest.py`)

1. **Load** → `load_pdf` reads each PDF with `pypdf`, one `Document` per page with metadata
   `source` (file name), `page` (1-based physical page), `page_label` (printed label).
   `load_pdfs` aborts with `IngestError` if any PDF is unreadable or has no text, unless `--skip-failed`.
2. **Chunk** → `chunk_documents` splits each page with `RecursiveCharacterTextSplitter`
   (`CHUNK_SIZE=500`, `CHUNK_OVERLAP=100`, in characters) and sets deterministic ids
   `<file>:p<page>:c<index on page>`.
3. **Embed + store** → `nomic-embed-text` (no task prefixes, deliberately), ChromaDB collection
   `rag_documents` in `vector_store/`, created with cosine distance and `embed_model` in its metadata
   (`models.collection_kwargs`).
4. PDFs are loaded and chunked *before* `--reset` deletes the old store.

### Phase 2: Inference (`rag_pipeline.py`, used by `app.py`)

`RAGPipeline.answer(question, history=None) -> Result` is **stateless**; callers own the history as a
list of `(question, answer)` pairs.

1. **Retrieve** → `retrieve()` takes the top `TOP_K=10` chunks and keeps those with cosine similarity
   `>= SCORE_THRESHOLD=0.65`. None left → `refused=True` without an LLM call.
2. **Budget** → `fit_to_budget()` keeps the prompt within `NUM_CTX - NUM_PREDICT` (8192 - 1024): passages
   first (lowest-scoring dropped only on overflow), then the most recent history turns.
3. **Generate** → chat prompt: `SYSTEM_PROMPT` (rules: passages are untrusted data, answer only from them,
   `NO_ANSWER` if insufficient, answer in the question's language, no links), history as real messages,
   then `<passages><passage id=… source=… page=…>…</passage></passages>` and the question last.
   Model: `qwen2.5:3b`, temperature 0.1.
4. **Result** → `Result(answer, refused, sources, chunks, error)`:
   - `sources`: passages sent to the model (`RetrievedChunk`: id, file, page, page_label, text, score); empty on refusal
   - `chunks`: everything retrieved, with scores (for evaluation)
   - `refused`: retrieval found nothing, or the output contained `NO_ANSWER`
   - `error`: `retrieval_failed: …` / `generation_failed: …` / `empty_question`; `answer` is then empty

### UI (`app.py`)

- One `RAGPipeline` shared by all sessions via `st.cache_resource`.
- Chat history lives in `st.session_state.messages`; failed turns are excluded from history.
- Answers and source snippets go through `safe_markdown.neutralize_links` (images → alt text,
  links/URLs → inert code spans) before rendering.
- Sidebar shows the configured models and retrieval settings.

### Error handling

- `ModelUnavailableError` (`models.py`) → Ollama unreachable or a configured model not pulled; raised at startup.
- `EmptyVectorStoreError` → store directory missing or collection empty. Checked before Chroma opens the path,
  because Chroma would silently create an empty store.
- `IncompatibleVectorStoreError` → store built with another distance space or embedding model; re-ingest.
- Per-question failures never raise; they are returned in `Result.error` and logged.

---

## Common Modifications

All of these are environment variables (see `config.py` and the README table); no code change needed.

- **Generation model:** `RAG_LLM_MODEL=qwen2.5:7b` (pull it first).
- **Embedding model:** `RAG_EMBED_MODEL=…`, then `python ingest.py --reset`. The pipeline refuses a store
  built with a different embedding model.
- **Chunking:** `RAG_CHUNK_SIZE`, `RAG_CHUNK_OVERLAP`, then `--reset`. Changes chunk ids, so gold chunk ids
  in evaluation data become invalid (page-level labels stay valid).
- **Retrieval:** `RAG_TOP_K`, `RAG_SCORE_THRESHOLD` (a cosine similarity, 0..1).
- **Context:** `RAG_NUM_CTX`, `RAG_NUM_PREDICT`, `RAG_MAX_HISTORY_TURNS`.
- **Ollama host (Docker):** `RAG_OLLAMA_BASE_URL` or the standard `OLLAMA_HOST`.

---

## Important Notes

- **Decisions:** every non-trivial decision gets an entry in `DECISIONS.md`
  (Decision / Alternatives / Rationale / Consequences). Review findings are in `REVIEW.md`.
- **Language:** all code, comments, docs, UI strings and commit messages are in English. The model answers
  in the language of the question (no language detection in code).
- **Source documents:** English EU regulations (EHDS 2025/327, GDPR 2016/679) in `documents/` (git-ignored).
- **Evaluation baseline:** nomic-embed-text without task prefixes and threshold 0.65 are deliberate baseline
  choices; prefixes and threshold calibration are planned experiments. Do not change them casually.
- **Prompt injection:** mitigated, not solved. qwen2.5:3b still follows strong instructions planted in a
  passage (see D-015). Only ingest trusted PDFs.
- **Tests:** `pytest` runs offline with fake embeddings and a fake LLM (`tests/conftest.py`). Answer quality
  is not covered by tests; it belongs to evaluation.
- **Dependencies:** direct dependencies pinned exactly in `requirements.txt`; no transitive lock yet.
- **Memory:** qwen2.5:3b uses about 3.1 GB with `num_ctx=8192` (2.5 GB at Ollama's default 4096).
- **Vector store:** `vector_store/` is not committed; rebuild with `python ingest.py --reset`.
  Stop the app first on Windows (open files cannot be deleted).
