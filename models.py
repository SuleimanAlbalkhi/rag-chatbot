"""Factories for the Ollama models and the vector store, built from config.

Model construction validates against the running Ollama instance and fails fast
with ModelUnavailableError if Ollama is unreachable or the model is not pulled.
"""

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from pydantic import ValidationError

import config


class ModelUnavailableError(RuntimeError):
    """Ollama is unreachable or a configured model is not installed."""


def _validated(cls, **kwargs):
    try:
        return cls(validate_model_on_init=True, **kwargs)
    except ValidationError as e:
        # langchain-ollama raises a ValueError inside a pydantic validator; surface its message only.
        inner = e.errors()[0].get("ctx", {}).get("error", e)
        raise ModelUnavailableError(str(inner)) from e
    except ConnectionError as e:
        # Raised by the ollama client itself; langchain-ollama only maps httpx errors.
        raise ModelUnavailableError(f"{e} (base URL: {kwargs.get('base_url') or 'OLLAMA_HOST or default'})") from e


# Task prefixes nomic-embed-text was trained with; only the embedded text gets them, stored chunks stay unchanged.
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


class PrefixedEmbeddings(Embeddings):
    """Adds nomic task prefixes before embedding documents and queries."""

    def __init__(self, inner: Embeddings):
        self.inner = inner

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.inner.embed_documents([DOCUMENT_PREFIX + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_query(QUERY_PREFIX + text)


def get_embeddings() -> Embeddings:
    embeddings = _validated(OllamaEmbeddings, model=config.EMBED_MODEL, base_url=config.OLLAMA_BASE_URL)
    return PrefixedEmbeddings(embeddings) if config.EMBED_PREFIXES else embeddings


def get_llm() -> ChatOllama:
    return _validated(
        ChatOllama,
        model=config.LLM_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=config.LLM_TEMPERATURE,
        num_ctx=config.NUM_CTX,
        num_predict=config.NUM_PREDICT,
    )


# Cosine space makes Chroma's relevance score equal to cosine similarity (1 - cosine distance),
# so SCORE_THRESHOLD reads directly as "cosine similarity >= threshold". Chroma's default is l2.
DISTANCE_SPACE = "cosine"


def collection_kwargs() -> dict:
    """Settings applied when the collection is created; shared by ingest and query."""
    return {
        "collection_name": config.COLLECTION_NAME,
        "persist_directory": str(config.VECTOR_STORE_DIR),
        "collection_configuration": {"hnsw": {"space": DISTANCE_SPACE}},
        "collection_metadata": {"embed_model": config.EMBED_MODEL, "embed_prefixes": config.EMBED_PREFIXES},
    }


def get_vector_store(embeddings: Embeddings) -> Chroma:
    return Chroma(embedding_function=embeddings, **collection_kwargs())
