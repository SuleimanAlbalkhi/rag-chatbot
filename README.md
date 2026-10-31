# On-Premise RAG Chatbot

> Ask questions about your PDF documents — fully local, no API keys, no cloud, no data leaving your machine.

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1.x-1C3C3C?style=flat)
![Ollama](https://img.shields.io/badge/Ollama-local-black?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

<img width="1912" height="909" alt="image" src="https://github.com/user-attachments/assets/67be64c5-c323-4486-b785-203d54b9c80d" />


---

## Why this project?

Most RAG chatbots require sending your documents to external APIs. This one doesn't.  
It was built with a specific use case in mind: **SMEs handling sensitive internal documents** (contracts, manuals, reports) who cannot or will not use cloud-based AI services.

Everything runs on your hardware. Zero vendor lock-in.

The current knowledge base is the English text of two EU regulations: the European Health Data Space
(Regulation (EU) 2025/327) and the GDPR (Regulation (EU) 2016/679). Any text-based PDFs work.

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│                INGESTION  (once)                 │
│                                                  │
│   PDF files (documents/)                         │
│       │  pypdf, one document per page            │
│       ▼                                          │
│   RecursiveCharacterTextSplitter                 │
│   (500 chars, 100 overlap, per page)             │
│   chunk id = <file>:p<page>:c<index>             │
│       │                                          │
│       ▼                                          │
│   nomic-embed-text  →  ChromaDB (cosine space)   │
└──────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────┐
│             INFERENCE  (per question)            │
│                                                  │
│   Question (+ history owned by the caller)       │
│       │                                          │
│       ▼                                          │
│   nomic-embed-text → ChromaDB top-10             │
│   keep chunks with cosine similarity ≥ 0.65      │
│       │            └─ none → refuse (no LLM call)│
│       ▼                                          │
│   Token budget (num_ctx 8192): passages first,   │
│   then the most recent history turns             │
│       │                                          │
│       ▼                                          │
│   qwen2.5:3b: system rules + <passages> +        │
│   question last  →  answer or NO_ANSWER          │
│       │                                          │
│       ▼                                          │
│   Result(answer, refused, sources, chunks, error)│
└──────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer          | Technology                        |
|----------------|-----------------------------------|
| Language model | qwen2.5:3b via Ollama             |
| Embeddings     | nomic-embed-text via Ollama       |
| Vector store   | ChromaDB (cosine distance)        |
| Orchestration  | LangChain 1.x (core, chroma, ollama) |
| PDF parsing    | pypdf                             |
| UI             | Streamlit                         |
| Tests / lint   | pytest, ruff, GitHub Actions      |

All design decisions, with alternatives and trade-offs, are logged in [DECISIONS.md](DECISIONS.md).
The code review that led to the current design is in [REVIEW.md](REVIEW.md).

---

## Key Design Decisions

**Stateless core** — `RAGPipeline.answer(question, history=None)` returns a structured `Result`
(`answer`, `refused`, `sources`, `chunks` with scores, `error`). The caller owns the conversation history,
which makes the pipeline reproducible for evaluation and safe to share between users.

**Cosine threshold (0.65)** — Only chunks with cosine similarity ≥ 0.65 to the question are used; if none
qualify, the bot refuses without calling the LLM. The value reproduces the previous behavior and will be
calibrated during evaluation. Note: German questions over English documents score noticeably lower.

**Explicit refusals** — The model replies `NO_ANSWER` when the passages do not contain the answer; the
pipeline turns that into `refused=True` with no sources.

**Citations** — Each source is one chunk with its id, file, 1-based physical page (what you type into a PDF
viewer), the printed page label if different, the full text and the similarity score.

**Prompt hardening** — Rules in a system message, document text in delimited numbered passages, question
last. Links and images in answers are rendered inert. This reduces, but does not eliminate, prompt
injection through documents (see DECISIONS.md D-015): only ingest PDFs you trust.

**Language** — The model is told to answer in the language of the question.

**Context window** — `num_ctx` is set explicitly (Ollama's default of 4096 silently truncates long prompts),
and passages plus history are budgeted to fit.

---

## Project Structure

```
rag-chatbot/
├── app.py              # Streamlit UI; keeps chat history in session state
├── rag_pipeline.py     # Stateless retrieval + generation, Result types, prompt, token budget
├── ingest.py           # PDF loading, chunking with stable ids, embedding, ChromaDB storage
├── models.py           # Ollama model and vector-store factories, startup model check
├── config.py           # All settings, overridable with RAG_* environment variables
├── safe_markdown.py    # Neutralizes links/images before rendering model output
├── tests/              # Offline test suite (no Ollama needed)
├── requirements.txt    # Pinned runtime dependencies
├── requirements-dev.txt# pytest + ruff
├── DECISIONS.md        # Decision log
├── REVIEW.md           # Code review findings
└── documents/          # Place your PDF files here (not committed)
```

> `vector_store/` is generated by `ingest.py` and not committed to git.

---

## Setup

### Prerequisites

- Python 3.12+ (tested on 3.13 and 3.14)
- [Ollama](https://ollama.com/download) installed and running locally

### 1. Pull required models

```bash
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

### 2. Clone and install

```bash
git clone https://github.com/SuleimanAlbalkhi/rag-chatbot.git
cd rag-chatbot
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Add your documents

Place one or more text-based `.pdf` files into the `documents/` folder. Scanned PDFs need OCR first.

### 4. Build the vector store

```bash
python ingest.py
```

To rebuild from scratch (after adding PDFs, or after upgrading from an older version of this project):

```bash
python ingest.py --reset
```

Ingestion stops with a report if any PDF cannot be read. To ingest the readable ones anyway:

```bash
python ingest.py --reset --skip-failed
```

Stop the Streamlit app before `--reset`; on Windows the open store cannot be deleted.

### 5. Start the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Configuration

Every setting lives in `config.py` and can be overridden with an environment variable:

| Variable | Default | Meaning |
|---|---|---|
| `RAG_LLM_MODEL` | `qwen2.5:3b` | Ollama generation model |
| `RAG_EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model (changing it requires `--reset`) |
| `RAG_OLLAMA_BASE_URL` | unset | Ollama URL; when unset, `OLLAMA_HOST` or `localhost:11434` |
| `RAG_LLM_TEMPERATURE` | `0.1` | Sampling temperature |
| `RAG_NUM_CTX` | `8192` | Context window requested from Ollama |
| `RAG_NUM_PREDICT` | `1024` | Maximum answer length in tokens |
| `RAG_TOP_K` | `10` | Chunks retrieved per question |
| `RAG_SCORE_THRESHOLD` | `0.65` | Minimum cosine similarity of a chunk |
| `RAG_MAX_HISTORY_TURNS` | `10` | History turns considered (before the token budget) |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | `500` / `100` | Chunking, in characters (changing requires `--reset`) |
| `RAG_DOCUMENTS_DIR` | `documents/` | PDF input folder |
| `RAG_VECTOR_STORE_DIR` | `vector_store/` | ChromaDB folder |
| `RAG_COLLECTION_NAME` | `rag_documents` | ChromaDB collection |

---

## Development

```bash
pip install -r requirements-dev.txt
pytest              # offline: fake embeddings and LLM, no Ollama needed
ruff check .
ruff format .
```

CI (GitHub Actions) runs lint, format check and tests on Python 3.12 and 3.13.

---

## Features

- **Fully local** — no data leaves your machine, no API keys required
- **Source transparency** — every answer lists the exact chunks used, with file, page and similarity score
- **Honest refusals** — refuses when nothing relevant is retrieved or the passages do not contain the answer
- **Multilingual answers** — replies in the language of the question
- **Clear failures** — missing models, an unreachable Ollama, an empty or incompatible vector store and
  unreadable PDFs are reported at startup with what to do
- **Conversation memory** — follow-up questions use recent turns, within a token budget

---

## Limitations

- PDF only — `.docx`, `.txt` not yet supported; scanned PDFs need OCR
- Follow-up questions are retrieved on their own wording, so "and what about paragraph 2?" often finds nothing
- A 3B model can still be steered by instructions planted inside documents; ingest only trusted PDFs
- Cross-lingual retrieval (German question, English documents) is weaker than same-language retrieval
- Response speed depends on hardware (CPU vs. GPU)

---

## Related Project

For a production-ready, multi-tenant version with authentication, PostgreSQL + pgvector, and a Next.js frontend:
→ [Full-Stack RAG SaaS Platform](https://github.com/SuleimanAlbalkhi/Full-Stack-rag-chatbot)

---

## License

MIT
