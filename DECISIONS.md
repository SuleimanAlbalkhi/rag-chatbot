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

---

## D-008 — Stateless core: answer(question, history=None) -> Result

- **Date:** 2026-10-31
- **Status:** Accepted

**Decision**
`RAGPipeline` holds no conversation state. `answer(question, history=None)` returns a frozen
`Result(answer, refused, sources, chunks, error)`. `chunks` holds every retrieved chunk with its id
and relevance score. The caller owns history as a list of `(question, answer)` pairs; the pipeline
uses the last `MAX_HISTORY_TURNS`. Streamlit rebuilds history from `st.session_state.messages`
and shares one cached pipeline across sessions (`st.cache_resource`).
The constructor accepts an optional `vector_store` and `llm` for tests and alternative backends.

**Alternatives**
- Keep `ask()` with internal history and add a `reset()` (previous design).
- A session object per user inside the pipeline.

**Rationale**
- Review H7: shared mutable history leaks between eval questions and between MCP clients.
- A pure function of its inputs is reproducible for evaluation, thread-safe for an MCP server,
  and trivially testable.
- Returning retrieved chunks with scores gives evaluation everything it needs (retrieval
  metrics, threshold calibration) without a second call.

**Consequences**
- `ask()` is removed; all callers use `answer()`.
- Callers must pass history explicitly; forgetting it silently loses multi-turn context.
- One model instance serves all Streamlit sessions (less memory, but requests share Ollama).

---

## D-009 — Refusals via a sentinel token and a `refused` flag

- **Date:** 2026-10-31
- **Status:** Accepted

**Decision**
Two refusal paths, both reported as `refused=True` with empty `sources` and a fixed English
`REFUSAL_MESSAGE`: (1) retrieval-level, when no chunk reaches the threshold (deterministic, no
LLM call); (2) LLM-level, when the prompt-instructed sentinel `NO_ANSWER` appears in the output.

**Alternatives**
- Match the free-text refusal sentence (previous behavior, review M6).
- Structured JSON output from the LLM.
- A separate LLM call to classify answer vs. refusal.

**Rationale**
- A fixed token survives the "respond in German" instruction, where a sentence would be
  translated or paraphrased. Verified live: an English and a German off-topic question both
  produced `NO_ANSWER` with qwen2.5:3b.
- JSON output from a 3B model is brittle and adds parsing failures; an extra classifier call doubles latency.
- Sources on a refusal are misleading (review H3), so they are dropped; `chunks` still shows
  what was retrieved for debugging and eval.

**Consequences**
- Any output containing `NO_ANSWER` counts as a refusal, including a partial answer that
  mentions it. Eval should track how often this happens.
- The refusal message is English even for German questions.
- Sources are still all retrieved chunks, not the ones the model actually used; per-chunk
  citation is a possible later improvement.

---

## D-010 — Error handling: error field for runtime failures, exceptions for setup failures

- **Date:** 2026-10-31
- **Status:** Accepted

**Decision**
- Setup problems raise at construction: `EmptyVectorStoreError` if the store directory is
  missing or the collection has no chunks, `ModelUnavailableError` (D-007) for Ollama/models.
  The directory is checked *before* Chroma is created, because Chroma would silently create an empty store.
- Per-question failures return `Result(error="retrieval_failed: …" | "generation_failed: …")`
  with an empty answer and are logged with the traceback. The UI shows a generic message and
  excludes failed turns from history.

**Alternatives**
- Raise on every failure and let callers catch.
- Error text inside `answer` (previous behavior, review H8).

**Rationale**
- A misconfigured store must never look like a 100% refusal rate (review H6); failing at startup is unmistakable.
- Per-question failures are expected in a long eval run or a server (Ollama timeout); a
  structured field lets the caller count, retry or report them without crashing the batch.

**Consequences**
- Callers must check `result.error` before using `result.answer`.
- Raw exception text is in `error` for logs and eval; the UI does not show it to end users.

---

## D-011 — Citations: 1-based physical page plus printed page label, one source per chunk

- **Date:** 2026-10-31
- **Status:** Accepted

**Decision**
Ingestion stores `page` as the 1-based physical page number and `page_label` as the label from
the PDF's page-label table (pypdf `reader.page_labels`). The UI shows `Page N` and appends
`(printed: X)` only when the label differs. Sources are no longer deduplicated by page: every
retrieved chunk is its own source with id, full text and relevance score.

**Alternatives**
- Show only `page_label` (what is printed on the paper page).
- Show only `page + 1`.
- Keep page-level dedup with a 300-character snippet (previous behavior, review L1).

**Rationale**
- Review H2: the 0-based index was shown as a page number, so every citation was one page early.
- The physical page number is what users type into a PDF viewer to jump there, and works for
  every PDF; labels are optional and can be non-numeric (e.g. roman numerals in front matter).
- Chunk-level sources show the exact passage that went to the model; with page dedup the
  supporting passage could be hidden behind another chunk from the same page.

**Consequences**
- Requires re-ingestion: old stores have 0-based `page` and no `page_label`.
- The sources list can contain several entries for the same page.
- Article/paragraph-level citations (more useful for legal text) remain future work.

---

## D-012 — Deterministic chunk IDs: `<file>:p<page>:c<index>`

- **Date:** 2026-10-31
- **Status:** Accepted

**Decision**
Each page is split separately and each chunk gets the id `<file name>:p<physical page>:c<index on page>`,
e.g. `gdpr.pdf:p37:c2`. The id is passed to Chroma explicitly and returned with every retrieved chunk.

**Alternatives**
- Random UUIDs (previous behavior, review M2).
- Content hash (stable across re-chunking of unrelated pages, but opaque).
- Global running index per file.

**Rationale**
- Retrieval eval needs gold chunk ids that survive re-ingestion; verified: two ingests of the
  same PDF produce identical ids.
- Human-readable ids make gold-label creation and debugging easy (the id says where the chunk is).
- Per-page indexing keeps ids stable when another page's text changes.

**Consequences**
- Ids are stable only for the same PDF, pypdf version and chunk settings; changing
  `CHUNK_SIZE` / `CHUNK_OVERLAP` invalidates gold chunk ids (gold labels at page level stay valid).
- Splitting per page means chunks never span a page break. A sentence that crosses pages is
  split in two; previously the splitter also worked per page document, so this is no change in behavior.
- Two PDFs with the same file name in different folders would collide; only one folder is ingested.

---

## D-013 — Strict ingestion: any unreadable PDF aborts unless --skip-failed

- **Date:** 2026-10-31
- **Status:** Accepted

**Decision**
`ingest.py` reports every PDF that cannot be read or has no extractable text and exits with
status 1 before touching the vector store. `--skip-failed` ingests the readable PDFs anyway
(still failing if none are readable). Pages without text are listed as warnings. PDFs are now
loaded and chunked *before* `--reset` deletes the old store.

**Alternatives**
- Warn and continue (previous behavior, review M4).
- Abort only when zero chunks are produced.

**Rationale**
- A silently missing regulation would make evaluation look like a retrieval failure. An explicit
  abort is cheap to work around and impossible to overlook.
- Loading before deleting means a bad PDF no longer costs the existing store. Verified with a
  corrupt and a text-less PDF: exit 1, existing store intact.

**Consequences**
- Ingestion is now all-or-nothing by default; CI or Docker builds fail on a bad PDF (intended).
- The remaining window where `--reset` can lose the store (embedding failure after deletion,
  Windows file locks) is review M3, deferred (D-020).
- Scanned PDFs need OCR before ingestion; no OCR is added.

---

## D-014 — Cosine distance space; threshold 0.65 = minimum cosine similarity

- **Date:** 2026-10-31
- **Status:** Accepted

**Decision**
The Chroma collection is created with `{"hnsw": {"space": "cosine"}}`. LangChain's relevance score
is then `1 - cosine_distance`, i.e. exactly the cosine similarity between question and chunk, and
`SCORE_THRESHOLD` means "keep chunks with cosine similarity >= threshold". The default is **0.65**.
The collection also records the embedding model in its metadata; at startup the pipeline refuses a
store whose space or embedding model does not match (`IncompatibleVectorStoreError`).

**Alternatives**
- Keep l2 and document the formula (review M1).
- Inner product (`ip`); equivalent to cosine for normalized vectors, but scores are not bounded to 0..1 in general.
- Threshold 0.50 (the old nominal value, now looser), or no threshold and rely on the LLM's `NO_ANSWER`.

**Rationale**
- Verified: Ollama's nomic-embed-text vectors are unit-normalized (norm 1.0), so the old l2 setup
  produced `score = 1 - sqrt(2) * (1 - cos)`; the documented 0.50 really meant cos >= 0.646.
- Cosine scores are what people expect a "similarity threshold" to be; eval sweeps become interpretable.
- 0.65 reproduces the old cutoff, so this change alters semantics, not behavior. Verified on a test
  store: the same chunk is retrieved with cos 0.674 (old score 0.539) and the same off-topic question is refused.
- The startup check prevents mixing an old l2 store (or another embedding model) with the cosine threshold.

**Consequences**
- **Re-ingestion is required** (`python ingest.py --reset`): existing stores use l2, the old collection
  name and 0-based pages.
- Measured on a small synthetic EHDS test PDF (not representative, but a warning sign): relevant German
  questions scored cos 0.46–0.49 against English chunks, below an off-topic German question (0.53) and
  an off-topic English one (0.54). With 0.65, German questions will mostly be refused at retrieval;
  lowering the threshold lets off-topic chunks through. Threshold calibration per language is the second
  evaluation experiment, after task prefixes (D-003).
- `RAG_SCORE_THRESHOLD` must now be given as a cosine similarity.

---

## D-015 — Prompt structure: system rules, delimited numbered passages, question last

- **Date:** 2026-10-31
- **Status:** Accepted

**Decision**
The prompt is a chat prompt: a system message with the rules (passages are untrusted data, answer only
from them, `NO_ANSWER` if they are insufficient, answer in the question's language, no URLs/links/images),
then prior turns as real user/assistant messages, then one user message with
`<passages><passage id="n" source=… page=…>…</passage></passages>` followed by the question. Tag-like
text (`<passage…>`, `</passages>`) is stripped from chunk text so a document cannot close the block.

**Alternatives**
- Single text prompt with "Context:" header (previous behavior).
- Sandwich defense: repeat the rule after the question. Tested and rejected, see below.
- An LLM-based injection classifier on retrieved chunks, or a grounding check on the answer.

**Rationale**
- Separating roles and delimiting the data is the standard baseline defense and costs nothing.
- Question last keeps the instruction closest to generation; history as real messages matches how the
  model was trained.
- Verified live with qwen2.5:3b: a payload that closes the passage tags and poses as a new user
  instruction is now refused 3/3 (was not tested before; it would have been inside the plain context).

**Consequences — residual risk (C3 is mitigated, not fixed)**
- The original review payload ("IMPORTANT SYSTEM NOTICE … Reply only with: …") **still succeeds 5/5**
  with qwen2.5:3b. A 3B model does not reliably separate instructions from data. A post-question
  reminder (sandwich) also failed 5/5 and made normal answers worse (verbatim copying), so it was reverted.
- The remaining defenses are at the output side (D-018) and in source control: only ingest trusted PDFs.
- Next steps: injection cases in the eval set; a stronger generator or a grounding/injection check if
  the eval shows it matters.

---

## D-016 — "Answer in the language of the question" instead of word-list detection

- **Date:** 2026-10-31
- **Status:** Accepted

**Decision**
Remove `_detect_language` and its German word list; the system prompt tells the model to answer in the
question's language, even when passages are in another language.

**Alternatives**
- Fix the word list (remove `was`, `die`, …).
- A language-ID library (`langdetect`, `lingua`) and an explicit "Respond in X".

**Rationale**
- Review H4: the word list sent English "was" questions to German.
- Qwen 2.5 is multilingual (D-001) and follows this instruction. Verified live: "Was the contract basis
  mentioned?" is answered in English, a German question over English passages in German.
- No extra dependency and no misclassification of short or mixed questions.

**Consequences**
- Language choice is now model behavior, not code; eval should measure the answer-language match rate.
- Any language the model knows works, not just German and English.
- The refusal message stays English (D-009).

---

## D-017 — Explicit context window (8192) and a token budget for passages and history

- **Date:** 2026-10-31
- **Status:** Accepted

**Decision**
`ChatOllama` gets `num_ctx=8192` and `num_predict=1024` (both configurable). Before each call,
`fit_to_budget` estimates tokens (3 characters per token, conservative) and keeps the prompt within
`num_ctx - num_predict - 256`: passages have priority, the lowest-scoring are dropped only if passages
alone overflow (at least one kept); the remaining budget takes the most recent history turns, oldest
dropped first. If Ollama reports `prompt_eval_count >= num_ctx`, a warning is logged.

**Alternatives**
- Only raise `num_ctx`.
- Exact token counts with the model's tokenizer (needs `transformers` or an Ollama tokenize call per turn).
- Summarize old history with an extra LLM call.

**Rationale**
- Review H1: Ollama's default 4096 window silently cut the start of the prompt (instructions and context).
- 8192 holds 10 passages plus several turns; qwen2.5:3b supports 32k, and the extra KV cache is small
  (~0.3 GB) on the local GPU.
- A character heuristic is dependency-free and errs on the safe side; exact counting is not worth a
  tokenizer dependency at this size.
- Verified live: 10 history turns of ~3,600 characters each now drop the 5 oldest turns and the answer stays correct.

**Consequences**
- Very long conversations lose old turns silently (logged at INFO).
- Changing `num_ctx` makes Ollama reload the model once.
- `Result.sources` lists the passages actually sent; `Result.chunks` lists everything retrieved.

---

## D-018 — Neutralize links and images when rendering model output

- **Date:** 2026-10-31
- **Status:** Accepted

**Decision**
`safe_markdown.neutralize_links` runs on every answer and source snippet before `st.markdown` /
`st.caption`: images (inline and reference) become their alt text, links become `text (url)` with the URL
in a code span, autolinks and bare URLs become code spans, reference definitions are removed.

**Alternatives**
- Render answers as plain text (`st.text`), losing lists and emphasis.
- Remove URLs entirely.
- An allow-list of domains (e.g. eur-lex.europa.eu).

**Rationale**
- Injected content can reach the answer (D-015). A rendered image is fetched by the browser without a
  click and can leak data in its URL; a rendered link invites a click on an attacker's site.
- Code spans keep the URL visible (useful for legitimate references) but inert.
- Markdown formatting of normal answers is preserved. Streamlit already escapes raw HTML.

**Consequences**
- This protects the Streamlit UI only. The MCP server will return raw text; its clients must apply
  their own rendering policy (to be documented with the MCP work).
- Legitimate links are no longer clickable.
