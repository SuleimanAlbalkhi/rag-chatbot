# Decisions

Log of architecture and project decisions. Every decision gets an entry with
Decision / Alternatives / Rationale / Consequences. Newest entries at the bottom.

---

## D-001 — Generator model: qwen2.5:3b instead of llama3.2:3b

- **Date:** 2026-10-31
- **Status:** Accepted

**Decision**
Use `qwen2.5:3b` (via Ollama) as the answer-generation LLM, replacing `llama3.2:3b`.

**Alternatives**
- Keep `llama3.2:3b` and re-pull it.
- Larger models (`llama3.1:8b`, `qwen2.5:7b`) for better quality.
- A hosted API model (rejected: the project's premise is fully local, no data leaving the machine).

**Rationale**
- `llama3.2:3b` is no longer installed locally; `ollama list` shows only `qwen2.5:3b`.
  The current code fails on every question with `model 'llama3.2:3b' not found (404)`.
- Qwen 2.5 has stronger multilingual ability, which matters because the bot answers
  English and German questions over English source documents.
- Same size class (3B), so RAM/VRAM requirements and latency stay comparable
  (observed: 2.5 GB, 100% GPU, 4096 context).

**Consequences**
- Model name must change in `rag_pipeline.py`, the Streamlit sidebar, README and CLAUDE.md.
  It should move to a single config value so eval, MCP and Docker can override it.
- Prior qualitative impressions of answer quality (based on Llama) are not transferable;
  the planned evaluation will establish a new baseline with qwen2.5:3b.
- Generation model change does not require re-ingestion (embeddings are separate).
- The embedding model `nomic-embed-text` is also not installed and still needs to be
  pulled (or replaced) before ingestion works — separate decision.

---

## D-002 — Source documents: English EU regulations (EHDS 2025/327, GDPR 2016/679)

- **Date:** 2026-10-31
- **Status:** Accepted

**Decision**
The knowledge base consists of the English versions of Regulation (EU) 2025/327
(European Health Data Space, EHDS) and Regulation (EU) 2016/679 (GDPR), as PDFs in `documents/`.

**Alternatives**
- German language versions of the same regulations.
- Both languages ingested side by side.
- Arbitrary internal SME documents (the original README use case).

**Rationale**
- Public, authoritative, stable texts: good for a reproducible evaluation set with
  gold answers and gold citations (article / paragraph / page).
- English keeps documents, prompt and evaluation in one language; embedding models
  are generally strongest on English.
- The two regulations overlap topically (health data is special-category data under
  GDPR Art. 9), which gives realistic hard cases for retrieval and refusal tests.

**Consequences**
- German questions become cross-lingual retrieval (German query vs. English chunks);
  retrieval quality for German questions must be measured separately in evaluation.
- Legal text is long and densely structured (recitals, articles, paragraphs); chunking
  and citation granularity (page vs. article) need to be revisited.
- Citations must point to correct page numbers of the official PDFs, since they are
  checkable by anyone.
- The PDFs are not committed (`documents/` is git-ignored); the exact source URLs and
  versions should be recorded so evaluation is reproducible and the Docker image can fetch them.

---

## D-003 — Embedding model: nomic-embed-text, without task prefixes (baseline)

- **Date:** 2026-10-31
- **Status:** Accepted

**Decision**
Keep `nomic-embed-text` (now pulled) as the embedding model and send raw text, without
the `search_document:` / `search_query:` task prefixes the model was trained with.

**Alternatives**
- Add the prefixes now (fixes review finding M7).
- Switch to another embedding model (e.g. `mxbai-embed-large`, `bge-m3` for stronger multilingual retrieval).

**Rationale**
- Prefixes are the first planned evaluation experiment. The baseline must be the current
  behavior so the effect of prefixes can be measured in isolation.
- Changing the model and prefixes at the same time would make the baseline meaningless.

**Consequences**
- Retrieval quality may be below what the model can deliver until the experiment runs.
- When prefixes are adopted, the store must be re-ingested (document embeddings change).
- Finding M7 is deferred, see D-020.

---

## D-004 — All settings in config.py with environment-variable overrides

- **Date:** 2026-10-31
- **Status:** Accepted

**Decision**
`config.py` is the single source of every tunable setting (paths, collection name, Ollama base
URL, model names, temperature, chunking, top-k, threshold, history length). Each one can be
overridden with a `RAG_*` environment variable. Model and vector-store construction lives in
`models.py`, so ingest and query always use the same embedding model and collection.
The Chroma collection gets an explicit name, `rag_documents`, instead of LangChain's implicit default `langchain`.

**Alternatives**
- `pydantic-settings` or a YAML/TOML config file.
- CLI flags per script.

**Rationale**
- Env vars are what Docker, CI and an MCP server can set without code changes, and eval can
  sweep parameters by setting them per run.
- Plain `os.getenv` needs no extra dependency; the settings are few and flat.
- `RAG_OLLAMA_BASE_URL` is optional: when unset, the ollama client still honors the standard `OLLAMA_HOST`.

**Consequences**
- Settings are read at import time; changing an env var requires a new process (fine for scripts and containers).
- No type validation beyond `int()` / `float()`; a malformed value fails at startup with a Python error.
- Existing vector stores built under the old default collection name are not found;
  re-ingestion is required anyway (see D-012, D-014).

---

## D-005 — langchain-chroma instead of langchain-community; PDF loading via pypdf directly

- **Date:** 2026-10-31
- **Status:** Accepted

**Decision**
Use `langchain_chroma.Chroma` and drop `langchain-community` entirely. Replace `PyPDFLoader`
(the only other community import) with a small loader in `ingest.py` that calls `pypdf.PdfReader`
directly. Also drop the unused `langchain` meta-package.

**Alternatives**
- Keep `langchain-community` only for `PyPDFLoader`.
- `langchain-pymupdf4llm` or `pymupdf` for better text extraction.

**Rationale**
- `langchain-community` is being sunset and its `Chroma` class is deprecated for removal (review H5).
- `PyPDFLoader` is a thin wrapper around `pypdf`; owning the loader gives direct control
  over page metadata (needed for the page-number fix in group 3) at the cost of a few lines.
- Switching extraction libraries would change chunk text and therefore the eval baseline.

**Consequences**
- Text extraction is unchanged (same `pypdf` plain extraction as `PyPDFLoader`).
- Metadata is reduced to what the code sets (`source`, `page`); `PyPDFLoader` extras such as
  `producer` or `total_pages` are gone. Nothing used them.

---

## D-006 — Pin direct dependencies exactly; transitive lock deferred to Docker

- **Date:** 2026-10-31
- **Status:** Accepted

**Decision**
`requirements.txt` pins every direct dependency to an exact version (the versions a fresh
install resolved on 2026-10-31). Transitive dependencies are not locked yet.

**Alternatives**
- Full lock file now (`pip-compile` / `uv lock` / `pip freeze`).
- Version ranges (`>=x,<y`).

**Rationale**
- Exact direct pins remove the main risk found in review (a fresh install pulling a breaking
  LangChain release).
- A full lock is platform- and Python-version-specific. It should be generated for the Python
  version the Docker image will use, not the local Windows Python 3.14.

**Consequences**
- Transitive packages (e.g. onnxruntime, numpy, httpx) can still drift between installs.
- To do in the Docker phase: generate a lock file for the image's Python version.
- Upgrades become deliberate: bump a pin, re-run tests and eval.

---

## D-007 — Startup check that configured Ollama models exist

- **Date:** 2026-10-31
- **Status:** Accepted

**Decision**
Construct `ChatOllama` / `OllamaEmbeddings` with `validate_model_on_init=True` and translate the
failure into a single `ModelUnavailableError` with a readable message. `ingest.py` runs the
check before it touches the existing vector store; the app shows the message and stops.

**Alternatives**
- Own check via `ollama.Client().list()`.
- No check; fail on the first question (previous behavior).

**Rationale**
- The built-in validation already compares against `ollama list` and names the missing model
  and the available ones; reusing it is less code.
- Two gaps had to be closed: the error comes wrapped in a pydantic `ValidationError`, and an
  unreachable Ollama raises the ollama client's builtin `ConnectionError`, which langchain-ollama
  does not map. Both are converted to `ModelUnavailableError`.

**Consequences**
- Startup makes one extra HTTP call to Ollama per model.
- A model that is pulled but broken still only fails on first use.
