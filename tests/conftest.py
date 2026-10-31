"""Shared fixtures. Nothing here talks to Ollama: embeddings and the LLM are fakes."""

import hashlib
import math
import re

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import config
import models


class BagOfWordsEmbeddings(Embeddings):
    """Deterministic, normalized bag-of-words vectors: texts sharing words have higher cosine similarity."""

    dim = 64

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for word in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class AngleEmbeddings(Embeddings):
    """Text 'cos=0.7' embeds to a unit vector with cosine 0.7 to every query (which embeds to [1, 0])."""

    def _vec(self, text: str) -> list[float]:
        if text.startswith("cos="):
            c = float(text[4:])
            return [c, math.sqrt(1 - c * c)]
        return [1.0, 0.0]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class FakeLLM:
    """Records the prompt messages of every call and replies with canned responses (or raises)."""

    def __init__(self, *responses: str | Exception):
        self.responses = list(responses) or ["A grounded answer."]
        self.calls: list[list] = []
        self.runnable = RunnableLambda(self._respond)

    def _respond(self, prompt_value):
        self.calls.append(prompt_value.to_messages())
        reply = self.responses[min(len(self.calls), len(self.responses)) - 1]
        if isinstance(reply, Exception):
            raise reply
        return AIMessage(content=reply)


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Point every path at a temp dir so tests never touch the real store or documents."""
    monkeypatch.setattr(config, "VECTOR_STORE_DIR", tmp_path / "store")
    monkeypatch.setattr(config, "DOCUMENTS_DIR", tmp_path / "docs")
    (tmp_path / "docs").mkdir()
    return tmp_path


@pytest.fixture
def embeddings():
    return BagOfWordsEmbeddings()


@pytest.fixture
def make_store(embeddings):
    """Build a Chroma store from (id, text, metadata) triples with the production collection settings."""

    def _make(items, embedding=None):
        store = models.Chroma(embedding_function=embedding or embeddings, **models.collection_kwargs())
        if items:
            ids, texts, metas = zip(*items, strict=True)
            store.add_texts(list(texts), metadatas=list(metas), ids=list(ids))
        return store

    return _make
