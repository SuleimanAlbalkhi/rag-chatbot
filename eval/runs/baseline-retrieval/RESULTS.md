# Results: baseline-retrieval

- Mode: retrieval-only · test set `eval/dev.jsonl` (50 items, sha256 `b58c4f4f6835`)
- Commit `0eeb3c2f8e` (dirty working tree) · started 2026-10-31T15:25:31+00:00
- Models: `qwen2.5:3b` (temperature 0.0), embeddings `nomic-embed-text`; threshold 0.65, top-k 10, chunks 500/100

## Retrieval (answerable questions, unfiltered top 10)

| Group | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|---|
| all answerable | 10 of 35 (29%) | 15 of 35 (43%) | 19 of 35 (54%) | 22 of 35 (63%) | 0.482 |
| single | 10 of 25 (40%) | 15 of 25 (60%) | 18 of 25 (72%) | 21 of 25 (84%) | 0.534 |
| multi | 0 of 10 (0%) | 0 of 10 (0%) | 1 of 10 (10%) | 1 of 10 (10%) | 0.354 |

Multi items count as a hit only when every gold page is within the top k (so Recall@1 is 0 by construction).

## Threshold curve (retrieval-level refusal: top score below threshold)

| Threshold | Correct refusals (unanswerable) | False refusals (answerable) |
|---|---|---|
| 0.40 | 0 of 15 (0%) | 0 of 35 (0%) |
| 0.45 | 0 of 15 (0%) | 0 of 35 (0%) |
| 0.50 | 0 of 15 (0%) | 0 of 35 (0%) |
| 0.55 | 1 of 15 (7%) | 0 of 35 (0%) |
| 0.60 | 2 of 15 (13%) | 0 of 35 (0%) |
| 0.65 ◀ configured | 4 of 15 (27%) | 1 of 35 (3%) |
| 0.70 | 7 of 15 (47%) | 6 of 35 (17%) |
| 0.75 | 13 of 15 (87%) | 20 of 35 (57%) |
| 0.80 | 13 of 15 (87%) | 29 of 35 (83%) |

The LLM can still refuse (`NO_ANSWER`) above the threshold; that is measured in full runs only.
