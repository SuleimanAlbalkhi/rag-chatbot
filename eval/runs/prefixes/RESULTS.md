# Results: prefixes

- Mode: retrieval-only · test set `eval/dev.jsonl` (50 items, sha256 `b58c4f4f6835`)
- Commit `0f0ca45a70` (dirty working tree) · started 2026-10-31T15:57:25+00:00
- Models: `qwen2.5:3b` (temperature 0.0), embeddings `nomic-embed-text`; threshold 0.65, top-k 10, chunks 500/100

## Retrieval (answerable questions, unfiltered top 10)

| Group | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|---|
| all answerable | 8 of 35 (23%) | 14 of 35 (40%) | 17 of 35 (49%) | 21 of 35 (60%) | 0.414 |
| single | 8 of 25 (32%) | 14 of 25 (56%) | 17 of 25 (68%) | 20 of 25 (80%) | 0.471 |
| multi | 0 of 10 (0%) | 0 of 10 (0%) | 0 of 10 (0%) | 1 of 10 (10%) | 0.272 |

Multi items count as a hit only when every gold page is within the top k (so Recall@1 is 0 by construction).

## Threshold curve (retrieval-level refusal: top score below threshold)

| Threshold | Correct refusals (unanswerable) | False refusals (answerable) |
|---|---|---|
| 0.40 | 0 of 15 (0%) | 0 of 35 (0%) |
| 0.45 | 0 of 15 (0%) | 0 of 35 (0%) |
| 0.50 | 0 of 15 (0%) | 0 of 35 (0%) |
| 0.55 | 0 of 15 (0%) | 0 of 35 (0%) |
| 0.60 | 3 of 15 (20%) | 0 of 35 (0%) |
| 0.65 ◀ configured | 4 of 15 (27%) | 1 of 35 (3%) |
| 0.70 | 8 of 15 (53%) | 4 of 35 (11%) |
| 0.75 | 13 of 15 (87%) | 21 of 35 (60%) |
| 0.80 | 13 of 15 (87%) | 29 of 35 (83%) |

The LLM can still refuse (`NO_ANSWER`) above the threshold; that is measured in full runs only.
