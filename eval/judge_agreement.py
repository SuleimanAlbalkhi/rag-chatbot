"""Agreement between the LLM judge and human labels.

Usage: python eval/judge_agreement.py --run baseline

Reads eval/runs/<run>/judge_form.csv (human_correct / human_faithful filled with 1 or 0) and judge.jsonl,
prints and writes judge_agreement.json with raw agreement and Cohen's kappa per verdict. A verdict whose
human column is empty for every item is reported as "not validated"; a partly filled column is an error.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.metrics import cohens_kappa, fraction  # noqa: E402

RUNS_DIR = ROOT / "eval" / "runs"
LABELS = {"correct": "human_correct (1/0)", "faithful": "human_faithful (1/0)"}


def parse_label(value: str, item_id: str, column: str) -> bool:
    v = value.strip().lower()
    if v in ("1", "true", "yes", "y"):
        return True
    if v in ("0", "false", "no", "n"):
        return False
    raise ValueError(f"{item_id}: '{column}' must be 1 or 0, got {value!r}")


def compare_verdict(rows: list[dict], judge: dict[str, dict], verdict: str, column: str) -> dict | None:
    """Agreement and kappa for one verdict, or None if no item has a human label in `column`."""
    if all(not (r.get(column) or "").strip() for r in rows):
        return None
    human = [parse_label(r.get(column) or "", r["id"], column) for r in rows]
    model = [bool(judge[r["id"]][verdict]) for r in rows]
    return {
        "n": len(rows),
        "agreement": sum(h == m for h, m in zip(human, model, strict=True)),
        "kappa": cohens_kappa(human, model),
        "disagreements": [
            {"id": r["id"], "human": h, "judge": m} for r, h, m in zip(rows, human, model, strict=True) if h != m
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()

    run_dir = RUNS_DIR / args.run
    judge = {
        j["id"]: j
        for j in (json.loads(line) for line in (run_dir / "judge.jsonl").read_text(encoding="utf-8").splitlines())
    }
    with (run_dir / "judge_form.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    result = {"n": len(rows)}
    for verdict, column in LABELS.items():
        try:
            stats = compare_verdict(rows, judge, verdict, column)
        except ValueError as e:
            sys.exit(str(e))
        result[verdict] = stats if stats is not None else "not validated"
        if stats is None:
            print(f"{verdict}: not validated (no human labels)")
            continue
        kappa = stats["kappa"]
        kappa_text = f"{kappa:.2f}" if kappa is not None else "undefined (all labels identical)"
        print(f"{verdict}: agreement {fraction(stats['agreement'], stats['n'])}, Cohen's kappa {kappa_text}")
        for d in stats["disagreements"]:
            print(f"  disagree {d['id']}: human={int(d['human'])} judge={int(d['judge'])} ({judge[d['id']]['reason']})")

    (run_dir / "judge_agreement.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
