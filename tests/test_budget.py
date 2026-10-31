import pytest

import config
from rag_pipeline import RetrievedChunk, estimate_tokens, fit_to_budget, format_passages


def chunk(i, score, size=500):
    return RetrievedChunk(f"doc.pdf:p{i}:c0", "doc.pdf", i, str(i), "x" * size, score)


@pytest.fixture(autouse=True)
def small_window(monkeypatch):
    monkeypatch.setattr(config, "NUM_CTX", 4096)
    monkeypatch.setattr(config, "NUM_PREDICT", 1024)


def test_everything_fits_unchanged():
    chunks = [chunk(1, 0.9), chunk(2, 0.8)]
    history = [("q1", "a1"), ("q2", "a2")]
    assert fit_to_budget("question", chunks, history) == (chunks, history)


def test_oldest_history_turns_are_dropped_first():
    chunks = [chunk(i, 0.9 - i / 100) for i in range(10)]
    history = [(f"q{i}", "y" * 1500) for i in range(10)]
    passages, kept = fit_to_budget("question", chunks, history)
    assert passages == chunks
    assert 0 < len(kept) < len(history)
    assert kept == history[-len(kept) :]


def test_lowest_scoring_passages_dropped_when_passages_alone_overflow():
    chunks = [chunk(i, s, size=3000) for i, s in enumerate([0.7, 0.9, 0.8, 0.6, 0.75])]
    passages, kept = fit_to_budget("question", chunks, [("q", "y" * 3000)])
    assert [c.score for c in passages] == sorted([c.score for c in passages], reverse=True)
    assert passages[0].score == 0.9 and len(passages) < len(chunks)
    assert kept == []  # no room left for history


def test_at_least_one_passage_is_kept():
    passages, _ = fit_to_budget("question", [chunk(1, 0.9, size=50_000)], [])
    assert len(passages) == 1


def test_result_stays_within_budget():
    chunks = [chunk(i, 0.9) for i in range(10)]
    history = [(f"q{i}", "y" * 2000) for i in range(10)]
    passages, kept = fit_to_budget("question", chunks, history)
    used = estimate_tokens(format_passages(passages)) + sum(estimate_tokens(q) + estimate_tokens(a) for q, a in kept)
    assert used <= config.NUM_CTX - config.NUM_PREDICT
