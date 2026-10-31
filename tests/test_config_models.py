import importlib

import ollama
import pytest

import config
import models


def test_env_vars_override_defaults(monkeypatch):
    monkeypatch.setenv("RAG_LLM_MODEL", "qwen2.5:7b")
    monkeypatch.setenv("RAG_TOP_K", "4")
    monkeypatch.setenv("RAG_SCORE_THRESHOLD", "0.42")
    monkeypatch.setenv("RAG_VECTOR_STORE_DIR", "/data/store")
    try:
        reloaded = importlib.reload(config)
        assert reloaded.LLM_MODEL == "qwen2.5:7b"
        assert reloaded.TOP_K == 4
        assert reloaded.SCORE_THRESHOLD == 0.42
        assert reloaded.VECTOR_STORE_DIR.as_posix() == "/data/store"
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_defaults():
    reloaded = importlib.reload(config)
    assert reloaded.LLM_MODEL == "qwen2.5:3b"
    assert reloaded.EMBED_MODEL == "nomic-embed-text"
    assert reloaded.NUM_CTX == 8192


def test_missing_model_raises_clear_error(monkeypatch):
    monkeypatch.setattr(ollama.Client, "list", lambda self: {"models": [{"model": "qwen2.5:3b"}]})
    with pytest.raises(models.ModelUnavailableError, match="nomic-embed-text") as e:
        models.get_embeddings()
    assert "ollama pull nomic-embed-text" in str(e.value)


def test_installed_model_passes(monkeypatch):
    monkeypatch.setattr(ollama.Client, "list", lambda self: {"models": [{"model": "qwen2.5:3b"}]})
    llm = models.get_llm()
    assert (llm.model, llm.num_ctx, llm.num_predict) == ("qwen2.5:3b", 8192, 1024)


def test_unreachable_ollama_raises_clear_error(monkeypatch):
    monkeypatch.setattr(config, "OLLAMA_BASE_URL", "http://127.0.0.1:1")
    with pytest.raises(models.ModelUnavailableError, match="127.0.0.1:1"):
        models.get_llm()


class RecordingEmbeddings:
    def __init__(self):
        self.seen = []

    def embed_documents(self, texts):
        self.seen += texts
        return [[1.0] for _ in texts]

    def embed_query(self, text):
        self.seen.append(text)
        return [1.0]


def test_prefixed_embeddings_add_nomic_task_prefixes():
    inner = RecordingEmbeddings()
    wrapped = models.PrefixedEmbeddings(inner)
    wrapped.embed_documents(["Article 9 text"])
    wrapped.embed_query("health data?")
    assert inner.seen == ["search_document: Article 9 text", "search_query: health data?"]


def test_get_embeddings_wraps_only_when_enabled(monkeypatch):
    monkeypatch.setattr(ollama.Client, "list", lambda self: {"models": [{"model": "nomic-embed-text:latest"}]})
    assert not isinstance(models.get_embeddings(), models.PrefixedEmbeddings)
    monkeypatch.setattr(config, "EMBED_PREFIXES", True)
    assert isinstance(models.get_embeddings(), models.PrefixedEmbeddings)


def test_prefix_env_var(monkeypatch):
    monkeypatch.setenv("RAG_EMBED_PREFIXES", "1")
    try:
        assert importlib.reload(config).EMBED_PREFIXES is True
    finally:
        monkeypatch.undo()
        importlib.reload(config)
