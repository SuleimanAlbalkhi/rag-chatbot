"""Score a run and write eval/runs/<run>/RESULTS.md.

Usage: python eval/report.py --run baseline

Answerable questions: Recall@1/3/5/10 (multi: all gold pages within k) and MRR, from unfiltered top-10 retrieval.
Threshold curve 0.40-0.80: retrieval-level refusals (top score below threshold) on both question groups.
Full runs: refusal rates at the configured threshold, latency median/p95, judge results if judge.jsonl exists.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.metrics import (  # noqa: E402
    fraction,
    gold_pages,
    hit_at_k,
    percentile,
    reciprocal_rank,
    threshold_curve,
    thresholds,
)

RUNS_DIR = ROOT / "eval" / "runs"
KS = (1, 3, 5, 10)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_report(meta: dict, testset: dict[str, dict], rows: list[dict], judged: list[dict] | None) -> str:
    cfg = meta["config"]
    answerable = [r for r in rows if testset[r["id"]]["type"] != "unanswerable"]
    unanswerable = [r for r in rows if testset[r["id"]]["type"] == "unanswerable"]

    def ranked(r):
        return [(c["file"], c["page"]) for c in r["retrieved"]]

    def top(r):
        return r["retrieved"][0]["score"] if r["retrieved"] else None

    lines = [
        f"# Results: {meta['run_name']}",
        "",
        f"- Mode: {meta['mode']} · test set `{meta['testset']['path']}` ({meta['testset']['items']} items, "
        f"sha256 `{meta['testset']['sha256'][:12]}`)",
        f"- Commit `{meta['git']['commit'][:10]}`{' (dirty working tree)' if meta['git']['dirty'] else ''} · "
        f"started {meta['started']}",
        f"- Models: `{cfg['LLM_MODEL']}` (temperature {cfg['LLM_TEMPERATURE']}), embeddings `{cfg['EMBED_MODEL']}`; "
        f"threshold {cfg['SCORE_THRESHOLD']}, top-k {cfg['TOP_K']}, chunks {cfg['CHUNK_SIZE']}/{cfg['CHUNK_OVERLAP']}",
        "",
        "## Retrieval (answerable questions, unfiltered top 10)",
        "",
        "| Group | " + " | ".join(f"Recall@{k}" for k in KS) + " | MRR |",
        "|---|" + "---|" * (len(KS) + 1),
    ]
    for label, group in (
        ("all answerable", answerable),
        *((t, [r for r in answerable if testset[r["id"]]["type"] == t]) for t in ("single", "multi")),
    ):
        if not group:
            continue
        golds = [gold_pages(testset[r["id"]]) for r in group]
        cells = [
            fraction(sum(hit_at_k(ranked(r), g, k) for r, g in zip(group, golds, strict=True)), len(group)) for k in KS
        ]
        mrr = sum(reciprocal_rank(ranked(r), g) for r, g in zip(group, golds, strict=True)) / len(group)
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {mrr:.3f} |")
    lines += [
        "",
        "Multi items count as a hit only when every gold page is within the top k (so Recall@1 is 0 by construction).",
        "",
    ]

    lines += [
        "## Threshold curve (retrieval-level refusal: top score below threshold)",
        "",
        "| Threshold | Correct refusals (unanswerable) | False refusals (answerable) |",
        "|---|---|---|",
    ]
    for row in threshold_curve(map(top, answerable), map(top, unanswerable), thresholds()):
        marker = " ◀ configured" if abs(row["threshold"] - cfg["SCORE_THRESHOLD"]) < 1e-9 else ""
        lines.append(
            f"| {row['threshold']:.2f}{marker} | {fraction(row['correct_refusals'], row['n_unanswerable'])} "
            f"| {fraction(row['false_refusals'], row['n_answerable'])} |"
        )
    lines += ["", "The LLM can still refuse (`NO_ANSWER`) above the threshold; that is measured in full runs only.", ""]

    if meta["mode"] == "full":
        ok = [r for r in rows if not r.get("error")]
        errors = len(rows) - len(ok)
        ans_ok = [r for r in ok if testset[r["id"]]["type"] != "unanswerable"]
        unans_ok = [r for r in ok if testset[r["id"]]["type"] == "unanswerable"]
        latencies = [r["latency_s"] for r in ok]
        p50, p95 = percentile(latencies, 50), percentile(latencies, 95)
        lines += [
            f"## End-to-end at the configured threshold ({cfg['SCORE_THRESHOLD']})",
            "",
            f"- Correct refusals (unanswerable): {fraction(sum(r['refused'] for r in unans_ok), len(unans_ok))}",
            f"- False refusals (answerable): {fraction(sum(r['refused'] for r in ans_ok), len(ans_ok))}",
            f"- Answered (answerable): {fraction(sum(not r['refused'] for r in ans_ok), len(ans_ok))}",
            f"- Errors: {fraction(errors, len(rows))}",
            f"- Latency per question: median {p50:.1f} s, p95 {p95:.1f} s" if latencies else "- Latency: n/a",
            "",
        ]

    if judged is not None:
        n = len(judged)
        lines += [
            "## LLM judge (answered answerable questions)",
            "",
            f"- Correct: {fraction(sum(j['correct'] for j in judged), n)}",
            f"- Faithful to retrieved passages: {fraction(sum(j['faithful'] for j in judged), n)}",
            "- Judge model: "
            + ", ".join(sorted({f"`{j.get('judge_model_id')}` (alias `{j.get('judge_model_alias')}`)" for j in judged}))
            if judged
            else "- No items judged",
            "",
        ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()

    run_dir = RUNS_DIR / args.run
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    if "finished" not in meta:
        sys.exit(f"Run '{args.run}' did not finish.")
    testset_path = ROOT / meta["testset"]["path"]
    if hashlib.sha256(testset_path.read_bytes()).hexdigest() != meta["testset"]["sha256"]:
        sys.exit(f"{testset_path} changed since the run; test sets must never change.")
    testset = {it["id"]: it for it in load_jsonl(testset_path)}
    rows = load_jsonl(run_dir / "results.jsonl")
    judge_path = run_dir / "judge.jsonl"
    judged = load_jsonl(judge_path) if judge_path.exists() else None

    report = build_report(meta, testset, rows, judged)
    (run_dir / "RESULTS.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
