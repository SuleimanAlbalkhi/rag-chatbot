# Experiments

Log of evaluation runs. Every run is in `eval/runs/<name>/` (`meta.json`, `results.jsonl`, `RESULTS.md`).
All numbers on `eval/dev.jsonl` (50 items: 35 answerable, 15 unanswerable) unless stated otherwise.
Design of the set and the harness: DECISIONS.md D-021, D-022.

---

## E-000 — Baseline (`baseline-retrieval`, `baseline`)

- **Date:** 2026-10-31
- **Setup:** qwen2.5:3b (temperature 0), nomic-embed-text **without** task prefixes, cosine threshold 0.65,
  top-k 10, chunks 500/100, store `vector_store/` (1830 chunks).
- **Provenance note:** both runs ran on commit `0eeb3c2f` with a **dirty working tree**. The uncommitted files
  were only the new evaluation harness (`eval/*.py`) and its tests; application code (`rag_pipeline.py`,
  `models.py`, `config.py`, `ingest.py`) was unchanged from the commit. The harness was committed right after
  the runs without changes to its logic (only the report footnote changed, which does not affect numbers),
  so the runs are reproducible from that commit and were not re-run.

**Retrieval (answerable, unfiltered top 10)**

| Group | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|---|
| All answerable | 10 of 35 (29%) | 15 of 35 (43%) | 19 of 35 (54%) | 22 of 35 (63%) | 0.482 |
| Single | 10 of 25 (40%) | 15 of 25 (60%) | 18 of 25 (72%) | 21 of 25 (84%) | 0.534 |
| Multi | 0 of 10 (0%) | 0 of 10 (0%) | 1 of 10 (10%) | 1 of 10 (10%) | 0.354 |

**Retrieval-level threshold curve**

| Threshold | Correct refusals (unanswerable) | False refusals (answerable) |
|---|---|---|
| 0.40–0.50 | 0 of 15 (0%) | 0 of 35 (0%) |
| 0.55 | 1 of 15 (7%) | 0 of 35 (0%) |
| 0.60 | 2 of 15 (13%) | 0 of 35 (0%) |
| **0.65** (configured) | 4 of 15 (27%) | 1 of 35 (3%) |
| 0.70 | 7 of 15 (47%) | 6 of 35 (17%) |
| 0.75 | 13 of 15 (87%) | 20 of 35 (57%) |
| 0.80 | 13 of 15 (87%) | 29 of 35 (83%) |

**End to end at 0.65 (`baseline`)**: correct refusals 14 of 15 (93%; 4 by threshold, 10 by `NO_ANSWER`),
false refusals 6 of 35 (17%; 1 by threshold, 5 by `NO_ANSWER`), answered 29 of 35 (83%), errors 0 of 50,
latency median 3.8 s, p95 8.8 s. The one answered unanswerable item is D-047 ("16 years old", borrowed from
GDPR Art. 8). Judge not run yet (CLI login expired).

**Observations**: retrieval is the bottleneck (7 of 10 multi items miss one page entirely in the top 10);
the threshold alone hardly separates answerable from unanswerable questions.
