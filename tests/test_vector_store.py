import pytest
from langchain_chroma import Chroma

import config
from rag_pipeline import EmptyVectorStoreError, IncompatibleVectorStoreError, RAGPipeline
from tests.conftest import AngleEmbeddings, FakeLLM


def meta(page):
    return {"source": "doc.pdf", "page": page, "page_label": str(page)}


def test_relevance_score_is_cosine_similarity(make_store):
    store = make_store([(f"c{c}", f"cos={c}", meta(1)) for c in (0.95, 0.7, 0.4)], embedding=AngleEmbeddings())
    scores = {doc.page_content: s for doc, s in store.similarity_search_with_relevance_scores("q", k=3)}
    assert scores == pytest.approx({"cos=0.95": 0.95, "cos=0.7": 0.7, "cos=0.4": 0.4}, abs=1e-4)


def test_threshold_keeps_chunks_at_or_above_cosine(make_store, monkeypatch):
    monkeypatch.setattr(config, "SCORE_THRESHOLD", 0.65)
    texts = ("cos=0.9", "cos=0.66", "cos=0.64", "cos=0.2")
    store = make_store([(t, t, meta(1)) for t in texts], embedding=AngleEmbeddings())
    p = RAGPipeline(vector_store=store, llm=FakeLLM().runnable)
    assert [c.text for c in p.retrieve("question")] == ["cos=0.9", "cos=0.66"]


def test_missing_store_dir_raises_and_is_not_created():
    with pytest.raises(EmptyVectorStoreError, match="ingest.py"):
        RAGPipeline(llm=FakeLLM().runnable)
    assert not config.VECTOR_STORE_DIR.exists()


def test_empty_collection_raises(make_store):
    with pytest.raises(EmptyVectorStoreError, match="empty"):
        RAGPipeline(vector_store=make_store([]), llm=FakeLLM().runnable)


def test_store_built_with_l2_space_is_rejected(embeddings):
    old = Chroma(
        collection_name=config.COLLECTION_NAME,
        persist_directory=str(config.VECTOR_STORE_DIR),
        embedding_function=embeddings,
    )
    old.add_texts(["some text"], metadatas=[meta(1)], ids=["a"])
    with pytest.raises(IncompatibleVectorStoreError, match="space='l2'"):
        RAGPipeline(vector_store=old, llm=FakeLLM().runnable)


def test_store_built_with_other_embedding_model_is_rejected(make_store, monkeypatch):
    store = make_store([("a", "some text", meta(1))])
    monkeypatch.setattr(config, "EMBED_MODEL", "mxbai-embed-large")
    with pytest.raises(IncompatibleVectorStoreError, match="nomic-embed-text"):
        RAGPipeline(vector_store=store, llm=FakeLLM().runnable)


def test_store_built_without_prefixes_is_rejected_when_prefixes_are_on(make_store, monkeypatch):
    store = make_store([("a", "some text", meta(1))])
    monkeypatch.setattr(config, "EMBED_PREFIXES", True)
    with pytest.raises(IncompatibleVectorStoreError, match="embed_prefixes=False"):
        RAGPipeline(vector_store=store, llm=FakeLLM().runnable)


def test_store_built_with_prefixes_is_rejected_when_prefixes_are_off(make_store, monkeypatch):
    monkeypatch.setattr(config, "EMBED_PREFIXES", True)
    store = make_store([("a", "some text", meta(1))])
    monkeypatch.setattr(config, "EMBED_PREFIXES", False)
    with pytest.raises(IncompatibleVectorStoreError, match="embed_prefixes=True"):
        RAGPipeline(vector_store=store, llm=FakeLLM().runnable)


def test_legacy_store_without_prefix_metadata_counts_as_unprefixed(embeddings):
    legacy = Chroma(
        collection_name=config.COLLECTION_NAME,
        persist_directory=str(config.VECTOR_STORE_DIR),
        embedding_function=embeddings,
        collection_configuration={"hnsw": {"space": "cosine"}},
        collection_metadata={"embed_model": config.EMBED_MODEL},
    )
    legacy.add_texts(["some text"], metadatas=[meta(1)], ids=["a"])
    RAGPipeline(vector_store=legacy, llm=FakeLLM().runnable)  # no error
