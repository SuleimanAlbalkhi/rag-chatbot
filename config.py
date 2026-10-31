"""Central configuration. Every setting can be overridden with an environment variable."""

import os
from pathlib import Path

_ROOT = Path(__file__).parent


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


# Paths
DOCUMENTS_DIR = Path(_env_str("RAG_DOCUMENTS_DIR", str(_ROOT / "documents")))
VECTOR_STORE_DIR = Path(_env_str("RAG_VECTOR_STORE_DIR", str(_ROOT / "vector_store")))
COLLECTION_NAME = _env_str("RAG_COLLECTION_NAME", "rag_documents")

# Ollama. None lets the ollama client fall back to OLLAMA_HOST / localhost:11434.
OLLAMA_BASE_URL = os.getenv("RAG_OLLAMA_BASE_URL") or None
LLM_MODEL = _env_str("RAG_LLM_MODEL", "qwen2.5:3b")
EMBED_MODEL = _env_str("RAG_EMBED_MODEL", "nomic-embed-text")
LLM_TEMPERATURE = _env_float("RAG_LLM_TEMPERATURE", 0.1)

# Ingestion (sizes are in characters, not tokens)
CHUNK_SIZE = _env_int("RAG_CHUNK_SIZE", 500)
CHUNK_OVERLAP = _env_int("RAG_CHUNK_OVERLAP", 100)

# Retrieval. SCORE_THRESHOLD is a minimum cosine similarity (0..1) between question and chunk.
# 0.65 reproduces the previous l2-based cutoff (see DECISIONS.md D-014); calibrate it in evaluation.
TOP_K = _env_int("RAG_TOP_K", 10)
SCORE_THRESHOLD = _env_float("RAG_SCORE_THRESHOLD", 0.65)

# Conversation
MAX_HISTORY_TURNS = _env_int("RAG_MAX_HISTORY_TURNS", 10)
