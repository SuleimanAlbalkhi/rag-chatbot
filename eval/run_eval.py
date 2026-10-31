"""Run the RAG pipeline over a test set and store everything needed to score it later.

Usage: python eval/run_eval.py --testset eval/dev.jsonl --run-name baseline [--retrieval-only]

Per question:
  retrieval  top-10 chunks WITHOUT threshold: id, file, page, score, rank
  full mode  RAGPipeline.answer() at the configured threshold, temperature 0: answer, refused, error,
             source ids, latency

Writes eval/runs/<run-name>/{meta.json, results.jsonl}. An existing run directory is never touched.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from rag_pipeline import RAGPipeline  # noqa: E402

RUNS_DIR = ROOT / "eval" / "runs"
RETRIEVAL_K = 10


def git_state() -> dict:
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    # Uncommitted or untracked files (outside the run outputs) mean the commit alone does not reproduce the run.
    changed = git("status", "--porcelain", "--", ".", ":(exclude)eval/runs").splitlines()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(changed), "changed_files": changed}


def config_snapshot() -> dict:
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(config).items() if k.isupper()}


def retrieve_unfiltered(pipeline: RAGPipeline, question: str) -> list[dict]:
    scored = pipeline.vector_store.similarity_search_with_relevance_scores(question, k=RETRIEVAL_K)
    return [
        {
            "rank": rank,
            "id": doc.id,
            "file": doc.metadata.get("source"),
            "page": doc.metadata.get("page"),
            "score": round(score, 6),
        }
        for rank, (doc, score) in enumerate(scored, 1)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--testset", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--final", action="store_true", help="required to read eval/holdout/ (final run only)")
    args = parser.parse_args()

    if "holdout" in args.testset.resolve().parts and not args.final:
        sys.exit("The holdout set is only read in an explicitly ordered final run (see CLAUDE.md); pass --final.")

    run_dir = RUNS_DIR / args.run_name
    if run_dir.exists():
        sys.exit(f"Run '{args.run_name}' already exists at {run_dir}; runs are never overwritten.")

    testset_bytes = args.testset.read_bytes()
    items = [json.loads(line) for line in testset_bytes.decode("utf-8").splitlines() if line.strip()]

    # Deterministic generation for evaluation; set before the pipeline builds its LLM.
    config.LLM_TEMPERATURE = 0.0
    pipeline = RAGPipeline()

    run_dir.mkdir(parents=True)
    meta = {
        "run_name": args.run_name,
        "mode": "retrieval-only" if args.retrieval_only else "full",
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
        "testset": {
            "path": args.testset.as_posix(),
            "sha256": hashlib.sha256(testset_bytes).hexdigest(),
            "items": len(items),
        },
        "git": git_state(),
        "retrieval_k": RETRIEVAL_K,
        "config": config_snapshot(),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    with (run_dir / "results.jsonl").open("w", encoding="utf-8") as out:
        for n, item in enumerate(items, 1):
            row = {"id": item["id"], "type": item["type"], "retrieved": retrieve_unfiltered(pipeline, item["question"])}
            if not args.retrieval_only:
                start = time.perf_counter()
                result = pipeline.answer(item["question"])
                row.update(
                    latency_s=round(time.perf_counter() - start, 3),
                    answer=result.answer,
                    refused=result.refused,
                    error=result.error,
                    source_ids=[s.id for s in result.sources],
                )
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            status = "" if args.retrieval_only else f" refused={row['refused']} {row['latency_s']}s"
            print(f"[{n}/{len(items)}] {item['id']} top={row['retrieved'][0]['score']:.3f}{status}", flush=True)

    meta["finished"] = datetime.now(UTC).isoformat(timespec="seconds")
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Run saved to {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
