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
