# RAG Chatbot – Local Document Assistant

> Ask questions about your PDF documents using a fully local, free AI pipeline.  
> No API keys. No cloud. No costs. Everything runs on your machine.

---

## What is this?

This project is a **Retrieval-Augmented Generation (RAG)** chatbot built entirely with open-source tools. It lets you upload any PDF document, builds a semantic search index over it, and lets you have a conversation with the content — with source references for every answer.

The chatbot remembers previous questions within a session, so you can ask follow-up questions naturally, just like a real conversation.

---

## Demo

```
You:       What is Retrieval-Augmented Generation?
Assistant: RAG is a technique that combines a retrieval system with a
           language model. Instead of relying solely on the model's
           training data, it first searches a document index for
           relevant passages, then passes those passages to the LLM
           as context to generate a grounded answer.

           📚 Sources: Chapter 1, Page 27 – "Building Your First Chatbot"
```

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Language Model | Llama 3.2 3B via Ollama | Fast, free, runs locally |
| Embeddings | nomic-embed-text via Ollama | High quality, open source |
| Vector Store | ChromaDB | Persistent, easy to use |
| Orchestration | LangChain | Industry standard for LLM apps |
| UI | Streamlit | Clean, Python-native frontend |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     INGESTION (once)                    │
│                                                         │
│   PDF Files  →  Text Chunks  →  Embeddings  →  ChromaDB│
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                  INFERENCE (per question)               │
│                                                         │
│   User Question                                         │
│        │                                                │
│        ▼                                                │
│   Embed Question  →  Similarity Search in ChromaDB      │
│                              │                          │
│                              ▼                          │
│                      Top-K Chunks                       │
│                              │                          │
│                              ▼                          │
│          Prompt = Question + Chunks + Chat History      │
│                              │                          │
│                              ▼                          │
│                    Llama 3.2 (local)                    │
│                              │                          │
│                              ▼                          │
│                   Answer  +  Sources                    │
└─────────────────────────────────────────────────────────┘
```

---

## Features

- **Fully local** – no data ever leaves your machine
- **Conversation memory** – ask follow-up questions naturally
- **Source transparency** – every answer shows which page it came from
- **Any PDF** – works with books, papers, manuals, contracts
- **Reset button** – clear conversation history in one click
- **No API costs** – runs entirely on Ollama + open-source models

---

## Project Structure

```
rag-chatbot/
│
├── app.py              # Streamlit UI and session management
├── ingest.py           # PDF loading, chunking, embedding, ChromaDB storage
├── rag_pipeline.py     # Retrieval logic, prompt, LLM, conversation memory
├── requirements.txt    # Python dependencies
│
├── documents/          # ← Place your PDF files here
└── vector_store/       # Auto-generated ChromaDB index (git-ignored)
```

---

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/download) installed

### 1. Pull the required models

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

### 2. Clone the repository

```bash
git clone https://github.com/SuleimanAlbalkhi/rag-chatbot.git
cd rag-chatbot
```

### 3. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Add your documents

Place one or more `.pdf` files into the `documents/` folder.

### 6. Build the vector store

```bash
python ingest.py
```

This loads your PDFs, splits them into chunks, generates embeddings with `nomic-embed-text`, and stores everything in a local ChromaDB index.

### 7. Start the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser and start chatting.

---

## How RAG Works

Traditional LLMs can only answer from what they learned during training. RAG extends this by giving the model access to your own documents at query time:

1. **Ingestion** – your PDFs are split into overlapping text chunks
2. **Embedding** – each chunk is converted to a vector (a list of numbers capturing its meaning)
3. **Retrieval** – when you ask a question, it's also embedded and the most semantically similar chunks are found
4. **Generation** – the retrieved chunks are given to the LLM as context, so it can answer based on your documents

This means the model doesn't need to be retrained on your data – it just reads the relevant parts on demand.

---

## Limitations

- Only supports PDF files (no `.docx`, `.txt`, etc. yet)
- Response speed depends on your hardware
- The 3B model may struggle with very complex reasoning – a larger model like `llama3.1:8b` improves quality if your RAM allows

---

## License

MIT
