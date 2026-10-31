# On-Premise RAG Chatbot

> Ask questions about your PDF documents — fully local, no API keys, no cloud, no data leaving your machine.
> Evaluated on a held-out test set over two EU regulations.

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1.x-1C3C3C?style=flat)
![Ollama](https://img.shields.io/badge/Ollama-local-black?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

<img width="1912" height="909" alt="image" src="https://github.com/user-attachments/assets/67be64c5-c323-4486-b785-203d54b9c80d" />


---

## Why this project?

Most RAG chatbots require sending your documents to external APIs. This one doesn't.
It was built for **organisations handling sensitive documents** that cannot or will not use cloud AI services.
Everything runs on your hardware: a 3B generator and an embedding model via Ollama, ChromaDB on disk.

The knowledge base is the English text of two EU regulations: the European Health Data Space
(Regulation (EU) 2025/327, EHDS) and the GDPR (Regulation (EU) 2016/679). Any text-based PDFs work.

The project is as much about **measuring** a small local RAG system honestly as about building it: a code review
([REVIEW.md](REVIEW.md)), a decision log ([DECISIONS.md](DECISIONS.md)), an experiment log
([EXPERIMENTS.md](EXPERIMENTS.md)) and a frozen held-out test set.

---

## Evaluation

### Final result: held-out test set

25 questions never used during development (18 answerable, 7 unanswerable), run once with the frozen
configuration (qwen2.5:3b at temperature 0, nomic-embed-text, cosine threshold 0.65, top 10).
Source: [`eval/runs/holdout/RESULTS.md`](eval/runs/holdout/RESULTS.md).

| Metric | Holdout (25) | Dev, for context (50) |
|---|---|---|
| **Unanswerable questions refused** | **6 of 7 (86%)** | 14 of 15 (93%) |
| **Answerable questions refused (false refusals)** | **7 of 18 (39%)** | 6 of 35 (17%) |
| Answerable questions answered | 11 of 18 (61%) | 29 of 35 (83%) |
| **Answers judged correct** (of answered) | **5 of 11 (45%)** | 14 of 29 (48%) |
| Answers judged faithful to retrieved passages (judge-only) | 7 of 11 (64%) | 16 of 29 (55%) |
| Retrieval Recall@5 (answerable) | 8 of 18 (44%) | 19 of 35 (54%) |
| Retrieval Recall@10 (answerable) | 11 of 18 (61%) | 22 of 35 (63%) |
| Retrieval MRR | 0.341 | 0.482 |
| Recall@10, multi-passage questions | 2 of 5 (40%) | 1 of 10 (10%) |
| Errors | 0 of 25 | 0 of 50 |
| Latency per question (median / p95) | 2.9 s / 8.2 s | 3.8 s / 8.8 s |

Dev numbers come from [`eval/runs/baseline/RESULTS.md`](eval/runs/baseline/RESULTS.md). The dev set was used to
choose the configuration; the holdout was read once, after the configuration was frozen
([DECISIONS.md D-024](DECISIONS.md)). With 18 answerable and 7 unanswerable holdout items, one item is 6 or 14
percentage points, so small differences between the columns are noise.

### Judge validation

Answer correctness is scored by an LLM judge (Claude via the Claude Code CLI, blind, no tools, binary rubric).
Requested as `sonnet`, the CLI reported the model **`claude-opus-5-5`** for every verdict. On 10 dev answers labelled
by hand, the judge's `correct` verdict agreed **9 of 10, Cohen's kappa 0.80**. The one disagreement shows the judge is
slightly **more lenient**: it accepted an answer that gave the GDPR's 72-hour breach rule but left out its
exception. The `faithful` verdict was not human-labelled and is **judge-only**.

### Method

- **Test sets:** questions, reference answers and evidence were **generated with Claude**; the holdout set was
  generated in a separate session that never saw the experiments. Both sets were **verified automatically** against
  the PDFs by `eval/verify_testset.py` (every evidence quote verbatim on its stated page, no 4+ word overlap between
  question and passage) and **reviewed** for duplicates and label quality; the review found 3 holdout questions
  that duplicated dev questions, which were replaced. Three types: single-passage, multi-passage (including
  questions that need both regulations) and unanswerable — mostly *near misses* where the topic is in the
  documents but the specific fact is not.
- **Retrieval** is scored at page level on the unfiltered top 10 (Recall@k, MRR; a multi-passage question counts
  only when all its pages are retrieved). **Refusals** are measured end to end at the configured threshold, and as
  a threshold curve from retrieval scores alone.
- Runs are never overwritten and record config, git commit and test-set hash
  ([`eval/run_eval.py`](eval/run_eval.py), [`eval/report.py`](eval/report.py)).
- Design decisions: [DECISIONS.md](DECISIONS.md) (D-021, D-022, D-024). All runs and their interpretation:
  [EXPERIMENTS.md](EXPERIMENTS.md). Code review that preceded the evaluation: [REVIEW.md](REVIEW.md).

---

## Key findings

- **The similarity threshold cannot separate near-miss questions.** Scores of answerable and unanswerable
  questions overlap: on the holdout, the threshold alone refuses 1 of 7 unanswerable questions at 0.65, and raising
  it to 0.75 still refuses only 1 of 7 while already refusing 12 of 18 answerable ones. Almost all correct
  refusals come from the generator's explicit `NO_ANSWER` (6 of 7 end to end vs. 1 of 7 by threshold).
- **The system is cautious.** It rarely invents an answer to an unanswerable question (1 of 7 holdout, 1 of 15 dev
  were answered), but it often refuses answerable ones (7 of 18 holdout, 6 of 35 dev), and only about half of the
  answers it gives are judged fully correct.
- **The bottleneck is ranking and multi-passage questions.** The right page is often retrieved but ranked low
  (holdout Recall@10 61% vs. Recall@1 17%, MRR 0.341). Questions that need two passages are rarely fully covered
  (Recall@10 1 of 10 on dev, 2 of 5 on holdout).
- **nomic-embed-text task prefixes were measured and rejected.** With `search_document:` / `search_query:` prefixes,
  dev Recall@5 fell from 19 to 17 of 35 and MRR from 0.482 to 0.414
  ([`eval/runs/prefixes/RESULTS.md`](eval/runs/prefixes/RESULTS.md), EXPERIMENTS.md E-001). The option remains,
  off by default.

---

## Limitations

- **Small, synthetic test sets:** 50 dev and 25 holdout questions, generated with Claude rather than collected from
  real users; automatic checks and a review make them accurate, not representative. Differences of a few points
  are noise.
- **3B generator:** qwen2.5:3b is fast and local but over-cautious and often incomplete on multi-part questions.
- **Prompt injection is mitigated, not fixed:** documents are delimited and treated as untrusted, links are rendered
  inert, but a strong instruction planted in a PDF still steers the 3B model ([DECISIONS.md D-015](DECISIONS.md)).
  Only ingest trusted PDFs.
- **Questions come mostly from articles:** recitals and annexes are covered less.
- **Cross-lingual weakness:** German questions over the English documents get lower retrieval scores; the test sets
  are English only.
- **No tracing or monitoring:** there is no request tracing, logging of retrieval scores per request, or usage
  telemetry beyond the evaluation runs.
- Follow-up questions are retrieved on their own wording (no query rewriting), and scanned PDFs need OCR.

---

## Next steps

- **Hybrid search** (BM25 + embeddings): legal text is full of exact terms and article numbers that keyword search
  finds well.
- **Reranker** (cross-encoder) on the top candidates, to fix ranking.
- **Larger generator** (e.g. 7B–8B) to reduce false refusals and incomplete answers.
- **Query rewriting** for follow-up questions and multi-passage questions.
- **MCP server** exposing the stateless `answer()` API ([DECISIONS.md D-025](DECISIONS.md)).
- **Docker** setup with Ollama as a service ([DECISIONS.md D-026](DECISIONS.md)).

Each change should be tried on the dev set first and confirmed on a new, unseen test set.

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

| Layer          | Technology                           |
|----------------|--------------------------------------|
| Language model | qwen2.5:3b via Ollama                |
| Embeddings     | nomic-embed-text via Ollama          |
| Vector store   | ChromaDB (cosine distance)           |
| Orchestration  | LangChain 1.x (core, chroma, ollama) |
| PDF parsing    | pypdf                                |
| UI             | Streamlit                            |
| Evaluation     | own harness, Claude Code CLI judge   |
| Tests / lint   | pytest, ruff, GitHub Actions         |

Design highlights: a **stateless core** (`RAGPipeline.answer(question, history=None)` returns
`Result(answer, refused, sources, chunks, error)`), **explicit refusals** (`NO_ANSWER` → `refused=True`, no
sources), **chunk-level citations** with file, physical page, printed page label and score, and an **explicit
context window** with a token budget. Every decision with alternatives and trade-offs is in
[DECISIONS.md](DECISIONS.md).

---

## Project Structure

```
rag-chatbot/
├── app.py               # Streamlit UI; keeps chat history in session state
├── rag_pipeline.py      # Stateless retrieval + generation, Result types, prompt, token budget
├── ingest.py            # PDF loading, chunking with stable ids, embedding, ChromaDB storage
├── models.py            # Ollama model and vector-store factories, startup model check
├── config.py            # All settings, overridable with RAG_* environment variables
├── safe_markdown.py     # Neutralizes links/images before rendering model output
├── eval/                # Test sets, verifier, run/report/judge scripts, runs/ with all results
├── tests/               # Offline test suite (no Ollama needed)
├── REVIEW.md            # Code review findings
├── DECISIONS.md         # Decision log
├── EXPERIMENTS.md       # Evaluation runs and results
└── documents/           # Place your PDF files here (not committed)
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

## Reproducing the evaluation

```bash
pip install -r requirements-dev.txt
python eval/verify_testset.py eval/dev.jsonl                          # check the test set against the PDFs
python eval/run_eval.py --testset eval/dev.jsonl --run-name my-run    # add --retrieval-only to skip the LLM
python eval/report.py --run my-run                                    # writes eval/runs/my-run/RESULTS.md
python eval/judge.py --run my-run                                     # needs a Claude Code CLI login
```

Run names must be new; existing runs are never overwritten. The holdout set is reserved for a final run
(`--final`) and must not be used for tuning.

---

## Configuration

Every setting lives in `config.py` and can be overridden with an environment variable:

| Variable | Default | Meaning |
|---|---|---|
| `RAG_LLM_MODEL` | `qwen2.5:3b` | Ollama generation model |
| `RAG_EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model (changing it requires `--reset`) |
| `RAG_EMBED_PREFIXES` | off | nomic task prefixes (measured and rejected, E-001; requires its own store) |
| `RAG_OLLAMA_BASE_URL` | unset | Ollama URL; when unset, `OLLAMA_HOST` or `localhost:11434` |
| `RAG_LLM_TEMPERATURE` | `0.1` | Sampling temperature (evaluation runs use 0) |
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

## Related Project

For a production-ready, multi-tenant version with authentication, PostgreSQL + pgvector, and a Next.js frontend:
→ [Full-Stack RAG SaaS Platform](https://github.com/SuleimanAlbalkhi/Full-Stack-rag-chatbot)

---

## License

MIT
