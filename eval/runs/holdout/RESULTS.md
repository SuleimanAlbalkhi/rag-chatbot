# Results: holdout

- Mode: full · test set `eval/holdout/holdout.jsonl` (25 items, sha256 `e4274ed12e9c`)
- Commit `542137b5aa` · started 2026-10-31T18:02:15+00:00
- Models: `qwen2.5:3b` (temperature 0.0), embeddings `nomic-embed-text`; threshold 0.65, top-k 10, chunks 500/100

## Retrieval (answerable questions, unfiltered top 10)

| Group | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|---|
| all answerable | 3 of 18 (17%) | 5 of 18 (28%) | 8 of 18 (44%) | 11 of 18 (61%) | 0.341 |
| single | 3 of 13 (23%) | 4 of 13 (31%) | 7 of 13 (54%) | 9 of 13 (69%) | 0.331 |
| multi | 0 of 5 (0%) | 1 of 5 (20%) | 1 of 5 (20%) | 2 of 5 (40%) | 0.369 |

Multi items count as a hit only when every gold page is within the top k (so Recall@1 is 0 by construction).

## Threshold curve (retrieval-level refusal: top score below threshold)

| Threshold | Correct refusals (unanswerable) | False refusals (answerable) |
|---|---|---|
| 0.40 | 0 of 7 (0%) | 0 of 18 (0%) |
| 0.45 | 0 of 7 (0%) | 0 of 18 (0%) |
| 0.50 | 0 of 7 (0%) | 0 of 18 (0%) |
| 0.55 | 1 of 7 (14%) | 0 of 18 (0%) |
| 0.60 | 1 of 7 (14%) | 0 of 18 (0%) |
| 0.65 ◀ configured | 1 of 7 (14%) | 2 of 18 (11%) |
| 0.70 | 1 of 7 (14%) | 6 of 18 (33%) |
| 0.75 | 1 of 7 (14%) | 12 of 18 (67%) |
| 0.80 | 2 of 7 (29%) | 14 of 18 (78%) |

The LLM can still refuse (`NO_ANSWER`) above the threshold; that is measured in full runs only.

## End-to-end at the configured threshold (0.65)

- Correct refusals (unanswerable): 6 of 7 (86%)
- False refusals (answerable): 7 of 18 (39%)
- Answered (answerable): 11 of 18 (61%)
- Errors: 0 of 25 (0%)
- Latency per question: median 2.9 s, p95 8.2 s

## LLM judge (answered answerable questions)

- Correct: 5 of 11 (45%)
- Faithful to retrieved passages: 7 of 11 (64%)
- Judge model: `claude-opus-5-5` (alias `sonnet`)
