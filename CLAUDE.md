# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start

### Prerequisites

- Python 3.10+
- Ollama running locally (https://ollama.com/download)
- Models pulled: `ollama pull llama3.2:3b && ollama pull nomic-embed-text`

### Ollama-Dienst

```bash
# Ollama starten (muss laufen, bevor ingest.py oder app.py gestartet wird)
ollama serve

# Status prüfen
ollama ps

# Benötigte Modelle laden (einmalig)
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# Geladene Modelle anzeigen
ollama list

# Windows: Ollama läuft nach der Installation automatisch als Dienst im System-Tray.
# Falls nicht, manuell starten: ollama serve (oder die Ollama-Desktop-App öffnen)
```

### Essential Commands

```bash
# Setup (einmalig)
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt

# Vektordatenbank aus PDFs in documents/ aufbauen
python ingest.py

# Neu aufbauen (nach neuen PDFs)
python ingest.py --reset

# Chatbot-UI starten (http://localhost:8501)
streamlit run app.py
```

---

## Architecture Overview

This is a **two-phase RAG pipeline**:

### Phase 1: Ingestion (ingest.py)
Runs once, at setup or when new PDFs arrive.

1. **Load PDFs** → PyPDFLoader reads all `.pdf` files from `documents/`
2. **Chunk** → RecursiveCharacterTextSplitter (500 tokens, 100-token overlap)
3. **Embed** → OllamaEmbeddings (nomic-embed-text model)
4. **Store** → ChromaDB persists to `vector_store/` directory

### Phase 2: Inference (rag_pipeline.py + app.py)
Runs per user question.

1. **UI Input** → Streamlit chat interface
2. **Embed Query** → Same nomic-embed-text model
3. **Retrieve** → ChromaDB similarity search (top-6 documents, k=6)
4. **Augment Prompt** → Add retrieved chunks + chat history to context
5. **Generate** → Llama 3.2 3B produces answer
6. **Track Sources** → Metadata extracted (filename, page) and displayed

### Session Management (app.py)
- Streamlit manages UI state via `st.session_state`
- `RAGPipeline` instance persists across messages
- Chat history (user/assistant pairs) stored in `pipeline.chat_history` (capped at 10 turns)
- Reset button clears both session state and pipeline history

---

## Key Implementation Details

### RAGPipeline Class (rag_pipeline.py)

**Constructor** (`__init__`):
- Initializes ChromaDB retriever from persisted `vector_store/`
- Creates ChatOllama instance (model="llama3.2:3b", temperature=0.1)
- Builds prompt template

**ask() method**:
- Retrieves top-6 similar documents via `retriever.invoke(question)`
- Formats chat history as "Nutzer: ... \n Assistent: ..." string
- Invokes LLM chain with context, history, and question
- Appends (question, response) pair to `chat_history`
- Trims history to last 10 exchanges with `chat_history = chat_history[-MAX_HISTORY:]`
- Extracts sources from document metadata: filename and page number
- Returns dict: `{"answer": str, "sources": [{"file": str, "page": int, "content": str}, ...]}`

### Configuration (config.py)

```python
DOCUMENTS_DIR = Path("documents")  # PDF input folder
VECTOR_STORE_DIR = Path("vector_store")  # ChromaDB persistence
```

These are relative paths and expected at repo root.

### Error Handling

- **Missing vector store** → app.py catches and shows error message
- **Ollama unavailable** → app.py and pipeline both have try/except for connection errors
- **No PDFs found** → ingest.py raises FileNotFoundError
- **Ollama timeout** → Returns error string to user instead of crashing

---

## Common Modifications

### Change LLM Model
In `rag_pipeline.py`, line 33:
```python
self.llm = ChatOllama(model="llama3.1:8b", temperature=0.1)  # Use 8B for better quality
```
Then re-run `python ingest.py --reset` if embeddings model changes.

### Adjust Chunk Size or Overlap
In `ingest.py`, lines 24-28:
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Increase for longer context per chunk
    chunk_overlap=200,  # Increase for more redundancy
    separators=["\n\n", "\n", " ", ""]
)
```

### Retrieve More Documents
In `rag_pipeline.py`, line 32:
```python
self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})  # Increase from 6
```
More docs = slower but potentially more comprehensive answers.

### Extend Chat History Limit
In `rag_pipeline.py`, line 7:
```python
MAX_HISTORY = 20  # Increase from 10
```
More context = slower LLM inference.

### Adjust Temperature
In `rag_pipeline.py`, line 33:
```python
self.llm = ChatOllama(model="llama3.2:3b", temperature=0.7)  # Higher = more creative, lower = deterministic
```

---

## Important Notes

- **Language Mix**: UI and error messages are in German (Streamlit app designed for German-speaking users). Prompt template is English.
- **Memory**: Llama 3.2 3B requires ~6GB RAM; 8B version requires ~16GB.
- **No Tests**: No test suite exists. Manual testing via `streamlit run app.py`.
- **No Linting**: No pre-commit hooks or linting configuration.
- **Vector Store Size**: ChromaDB persists to `vector_store/` (~17MB for one ~8MB PDF). Not committed to git.
