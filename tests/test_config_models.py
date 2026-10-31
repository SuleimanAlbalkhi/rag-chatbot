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
