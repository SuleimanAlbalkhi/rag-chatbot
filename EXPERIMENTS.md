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
GDPR Art. 8).

**LLM judge on `baseline`** (29 answered answerable items, `eval/judge.py`, run after a CLI re-login):
correct 14 of 29 (48%), faithful to the retrieved passages 16 of 29 (55%). By type: single 14 of 21 correct and
15 of 21 faithful; multi 0 of 8 correct and 1 of 8 faithful. The judge was called with the alias `sonnet`, but the CLI reported the model id
`claude-opus-5-5` for every verdict; the recorded id is authoritative. Typical failures: missing key facts in
multi answers (retrieval found only one of the two passages), wrong article numbers (D-029), a date present in a
passage but not used (D-022).

**Judge validation** (`eval/judge_agreement.py`, 10 human-labelled items in `eval/runs/baseline/judge_form.csv`):
`correct` agrees on 9 of 10 (90%), Cohen's kappa 0.80. The one disagreement, D-003, is the judge being more
lenient: it accepted the core 72-hour rule although the answer left out the exception (no notification if the
breach is unlikely to result in a risk); the human label is incorrect. So `correct` may be slightly optimistic for
answers that give the rule but omit its conditions. **`faithful` is judge-only**: it was intentionally not
human-labelled, so its numbers are unvalidated.

**Observations**: retrieval is the bottleneck (7 of 10 multi items miss one page entirely in the top 10);
the threshold alone hardly separates answerable from unanswerable questions.

---

## E-001 — nomic task prefixes (`prefixes`) — REJECTED

- **Date:** 2026-10-31
- **Hypothesis:** nomic-embed-text was trained with `search_document: ` / `search_query: ` prefixes; adding them
  improves retrieval (review finding M7, D-003).
- **Change:** `RAG_EMBED_PREFIXES=1` wraps the embeddings (`models.PrefixedEmbeddings`); only the embedded text gets
  the prefix, stored chunk text is unchanged. Ingested into a separate store `vector_store_prefixes/` (same 1830
  chunk ids); the baseline store `vector_store/` is untouched. The collection records `embed_prefixes` and the
  pipeline refuses a store whose setting differs (stores without the key count as unprefixed).
- **Run:** `RAG_EMBED_PREFIXES=1 RAG_VECTOR_STORE_DIR=vector_store_prefixes python eval/run_eval.py
  --testset eval/dev.jsonl --run-name prefixes --retrieval-only`. Everything else as E-000.
- **Provenance:** commit `0f0ca45a` with a dirty tree: the prefix feature itself (`config.py`, `models.py`,
  `rag_pipeline.py`, tests, `.gitignore`) was uncommitted; the changed files are listed in `meta.json`.
- **Acceptance rule (set before the run):** Recall@5 (all answerable) improves by ≥ 2 questions **and** MRR does
  not drop.

| Metric (answerable) | baseline | prefixes | Δ |
|---|---|---|---|
| Recall@1 | 10 of 35 (29%) | 8 of 35 (23%) | −2 |
| Recall@3 | 15 of 35 (43%) | 14 of 35 (40%) | −1 |
| **Recall@5** | **19 of 35 (54%)** | **17 of 35 (49%)** | **−2** |
| Recall@10 | 22 of 35 (63%) | 21 of 35 (60%) | −1 |
| **MRR** | **0.482** | **0.414** | **−0.068** |
| Single Recall@5 / MRR | 18 of 25 / 0.534 | 17 of 25 / 0.471 | −1 / −0.063 |
| Multi Recall@10 / MRR | 1 of 10 / 0.354 | 1 of 10 / 0.272 | 0 / −0.082 |

Retrieval-level threshold curve with prefixes (for reference): correct refusals / false refusals at 0.60: 3 of 15 /
0 of 35; 0.65: 4 of 15 / 1 of 35; 0.70: 8 of 15 / 4 of 35; 0.75: 13 of 15 / 21 of 35.

- **Result:** both conditions fail (Recall@5 −2, MRR −0.068). **Rejected**; the threshold proposal (planned for an
  accepted result) was not made.
- **Per item:** the first relevant rank improved for 8 items and got worse for 11 (e.g. D-010 rank 1 → 9, D-018
  rank 2 → not in top 10; D-023 rank 8 → 3, D-034 multi not found → rank 5). Mean top score barely moved
  (answerable 0.748 → 0.747, unanswerable 0.690 → 0.689), so prefixes neither sharpen nor shift the score scale.
- **Caveat:** with 35 answerable items, a 2-question difference is within noise; the evidence is "no improvement",
  not "prefixes are harmful". The result may differ with other chunk sizes or after query rewriting.

---

## E-002 — FINAL: holdout run (`holdout`)

- **Date:** 2026-10-31
- **Status:** Final result. The configuration was frozen before this run (D-024) and is **not tuned further** on
  these numbers; the holdout set has now been used and cannot serve as an unseen test set again.
- **Setup:** exactly E-000 / D-024 (qwen2.5:3b at temperature 0, nomic-embed-text without prefixes, cosine
  threshold 0.65, top-k 10, chunks 500/100, store `vector_store/`).
- **Run:** `python eval/run_eval.py --testset eval/holdout/holdout.jsonl --run-name holdout --final` (full mode),
  then `python eval/judge.py --run holdout`. Clean tree on commit `542137b5`; no code changes before or after.
- **Test set:** 25 items (18 answerable: 13 single, 5 multi; 7 unanswerable), sha256 `e4274ed12e9c…`.

| | Dev `baseline` (50) | **Holdout `holdout` (25)** |
|---|---|---|
| Recall@1 (answerable) | 10 of 35 (29%) | 3 of 18 (17%) |
| Recall@3 | 15 of 35 (43%) | 5 of 18 (28%) |
| Recall@5 | 19 of 35 (54%) | 8 of 18 (44%) |
| Recall@10 | 22 of 35 (63%) | 11 of 18 (61%) |
| MRR | 0.482 | 0.341 |
| Single / multi Recall@10 | 21 of 25 (84%) / 1 of 10 (10%) | 9 of 13 (69%) / 2 of 5 (40%) |
| Threshold only (0.65): correct / false refusals | 4 of 15 (27%) / 1 of 35 (3%) | 1 of 7 (14%) / 2 of 18 (11%) |
| **End to end: correct refusals (unanswerable)** | 14 of 15 (93%) | **6 of 7 (86%)** |
| **End to end: false refusals (answerable)** | 6 of 35 (17%) | **7 of 18 (39%)** |
| Answered (answerable) | 29 of 35 (83%) | 11 of 18 (61%) |
| Errors | 0 of 50 | 0 of 25 |
| Latency median / p95 | 3.8 s / 8.8 s | 2.9 s / 8.2 s |
| **Judge: correct** (validated, κ 0.80 on dev) | 14 of 29 (48%) | **5 of 11 (45%)** |
| Judge: faithful (judge-only, unvalidated) | 16 of 29 (55%) | 7 of 11 (64%) |
| Judge correct: single / multi | 14 of 21 / 0 of 8 | 3 of 6 / 2 of 5 |

Judge model: alias `sonnet`, CLI-reported id `claude-opus-5-5` (as on dev).

**Findings**
- **False refusals are the main loss:** 7 of 18 answerable holdout questions were refused (dev: 6 of 35). Two by the
  threshold (top scores 0.633 and 0.637, just below 0.65), five by the generator's `NO_ANSWER` although retrieval
  passed the threshold (top scores 0.653–0.723).
- **Ranking is weaker, recall at 10 is not:** MRR 0.48 → 0.34 and Recall@5 54% → 44%, while Recall@10 stays at
  61% vs. 63%: the right page is usually retrieved, but ranked lower.
- **Unanswerable questions are still refused reliably:** 6 of 7. The one miss (H-020) had a top score of 0.821,
  out of reach for any reasonable threshold; nearly all refusals come from the generator, not the threshold
  (threshold alone: 1 of 7).
- **Correctness of answered questions is stable:** 45% vs. 48% on dev.
- **Precision:** with 18 answerable and 7 unanswerable items, one item is 6 or 14 percentage points; dev/holdout
  differences of one or two items are noise. The false-refusal rate is the one clearly worse number.

**Conclusion:** the local 3B pipeline refuses unanswerable questions well (86–93%), but answers only about 60–80% of
answerable ones and gets roughly half of those fully right. The limits are retrieval ranking (multi-passage
questions especially) and an over-cautious generator, not the threshold. Any improvement now needs new dev
experiments and a new, unseen test set for confirmation.
