"""Verify an evaluation test set against the source PDFs.

Usage: python eval/verify_testset.py eval/dev.jsonl
       python eval/verify_testset.py --holdout eval/holdout/holdout.jsonl

Checks
  (a) every evidence quote appears on its stated page (1-based physical page, as the app cites it)
  (b) schema, ids and type counts, mix rules (per set, see PROFILES):
      dev     D-001…, 25 single / 10 multi / 15 unanswerable, >= 5 cross-document multi, >= 12 near miss
      holdout H-001…, 13 single / 5 multi / 7 unanswerable, >= 3 cross-document multi, >= 6 near miss
  (c) no 4+ consecutive words shared between a question and any of its quotes
  (d) unanswerable items: the terms listed in `note` are searched over the full text of both PDFs;
      hits are reported (not failed) so a human can confirm the fact is really absent

PDFs are loaded with ingest.load_pdf, the same loader the app uses. Text and quotes are compared after
normalization: Unicode NFKC, unified quote characters, soft hyphens and hyphens removed, all whitespace
removed. This tolerates line breaks and end-of-line hyphenation in the PDF text, nothing else.

Writes a human-readable report next to the test set: review_<name>.md (git-ignored).
Exit code 0 only if every check passes.
"""

import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from ingest import load_pdf  # noqa: E402

FILES = ("ehds.pdf", "gdpr.pdf")
TYPES = ("single", "multi", "unanswerable")
PROFILES = {
    "dev": {
        "id_prefix": "D",
        "counts": {"single": 25, "multi": 10, "unanswerable": 15},
        "min_cross_doc_multi": 5,
        "min_near_miss": 12,
        "max_off_topic": 3,
    },
    "holdout": {
        "id_prefix": "H",
        "counts": {"single": 13, "multi": 5, "unanswerable": 7},
        "min_cross_doc_multi": 3,
        "min_near_miss": 6,
        "max_off_topic": 1,
    },
}
MIN_QUOTE_WORDS = 10
MAX_SHARED_RUN = 3  # a run of 4+ identical consecutive words fails check (c)
OFF_TOPIC_PREFIX = "off-topic"
FIELDS = ("id", "question", "type", "ground_truth", "evidence", "note")

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "´": "'"})
_HYPHENS = re.compile(r"[­‐‑‒–\-]")


def compact(text: str) -> str:
    """Comparison form for quote matching: no whitespace, no hyphens, unified quotes."""
    text = unicodedata.normalize("NFKC", text).translate(_QUOTES)
    return re.sub(r"\s+", "", _HYPHENS.sub("", text))


def readable(text: str) -> str:
    """Display/search form: joined end-of-line hyphenation, single spaces, lower case."""
    text = unicodedata.normalize("NFKC", text).translate(_QUOTES).replace("­", "")
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", text).lower().translate(_QUOTES))


def shared_runs(a: str, b: str, n: int) -> set[str]:
    wa, wb = words(a), words(b)
    grams_b = {tuple(wb[i : i + n]) for i in range(len(wb) - n + 1)}
    return {" ".join(wa[i : i + n]) for i in range(len(wa) - n + 1) if tuple(wa[i : i + n]) in grams_b}


def search_terms(note: str) -> list[str]:
    """Terms after 'Searched:' in the note, separated by ';'."""
    m = re.search(r"searched:\s*(.+)$", note, re.IGNORECASE | re.DOTALL)
    return [t.strip().strip(".") for t in m.group(1).split(";") if t.strip().strip(".")] if m else []


def load_pages() -> dict[str, list[str]]:
    pages = {}
    for name in FILES:
        path = config.DOCUMENTS_DIR / name
        if not path.exists():
            sys.exit(f"Missing {path}")
        pages[name] = [d.page_content for d in load_pdf(path)]
    return pages


def term_hits(term: str, pages: dict[str, list[str]], limit: int = 5) -> tuple[int, list[str]]:
    """Whole-word, case-insensitive matches (so 'EUR' does not match 'Europe')."""
    pattern = re.compile(rf"(?<!\w){re.escape(readable(term))}(?!\w)")
    total, snippets = 0, []
    for name, texts in pages.items():
        for i, text in enumerate(texts, 1):
            hay = readable(text)
            for m in pattern.finditer(hay):
                total += 1
                if len(snippets) < limit:
                    s, e = max(0, m.start() - 70), min(len(hay), m.end() + 70)
                    snippets.append(f"{name} p{i}: …{hay[s:e]}…")
    return total, snippets


def main(path: Path, profile: dict) -> int:
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    pages = load_pages()
    compact_pages = {n: [compact(t) for t in ts] for n, ts in pages.items()}

    errors: list[str] = []
    per_item: dict[str, list[str]] = {}
    hits_report: dict[str, list[tuple[str, int, list[str]]]] = {}

    # (b) schema and ids
    for n, it in enumerate(items, 1):
        iid = it.get("id", f"#{n}")
        problems = per_item.setdefault(iid, [])
        missing = [f for f in FIELDS if f not in it]
        if missing:
            problems.append(f"missing fields {missing}")
            continue
        want_id = f"{profile['id_prefix']}-{n:03d}"
        if it["id"] != want_id:
            problems.append(f"id should be {want_id}")
        if it["type"] not in TYPES:
            problems.append(f"unknown type {it['type']!r}")
        if not it["question"].strip() or not it["ground_truth"].strip() or not it["note"].strip():
            problems.append("empty question, ground_truth or note")
        ev = it["evidence"]
        if it["type"] == "unanswerable" and ev:
            problems.append("unanswerable item must have empty evidence")
        if it["type"] == "single" and len(ev) < 1:
            problems.append("single item needs evidence")
        if it["type"] == "multi" and len({(e.get("file"), e.get("page")) for e in ev}) < 2:
            problems.append("multi item needs evidence from at least two places")

        for k, e in enumerate(ev, 1):
            f, page, quote = e.get("file"), e.get("page"), e.get("quote", "")
            if f not in pages:
                problems.append(f"evidence {k}: unknown file {f!r}")
                continue
            if not isinstance(page, int) or not 1 <= page <= len(pages[f]):
                problems.append(f"evidence {k}: page {page!r} out of range 1..{len(pages[f])}")
                continue
            if len(words(quote)) < MIN_QUOTE_WORDS:
                problems.append(f"evidence {k}: quote has {len(words(quote))} words (< {MIN_QUOTE_WORDS})")
            # (a) quote on stated page
            cq = compact(quote)
            if cq not in compact_pages[f][page - 1]:
                found = [i for i, t in enumerate(compact_pages[f], 1) if cq in t]
                where = f"found on page(s) {found}" if found else "not found anywhere in the file"
                problems.append(f"evidence {k}: quote not on {f} p{page} ({where})")
            # (c) paraphrase check
            runs = shared_runs(it["question"], quote, MAX_SHARED_RUN + 1)
            if runs:
                problems.append(f"evidence {k}: question shares {MAX_SHARED_RUN + 1}+ words with quote: {sorted(runs)}")

        # (d) term search for unanswerable
        if it["type"] == "unanswerable":
            terms = search_terms(it["note"])
            if not terms:
                problems.append("unanswerable note must list 'Searched: term; term; ...'")
            hits_report[iid] = [(t, *term_hits(t, pages)) for t in terms]

    # (b) counts and mix
    types = Counter(it.get("type") for it in items)
    for t, want in profile["counts"].items():
        if types[t] != want:
            errors.append(f"type '{t}': {types[t]} items, expected {want}")
    if len(items) != sum(profile["counts"].values()):
        errors.append(f"{len(items)} items, expected {sum(profile['counts'].values())}")

    def files_of(it):
        return {e["file"] for e in it.get("evidence", []) if e.get("file") in pages}

    multi = [it for it in items if it.get("type") == "multi"]
    cross = [it for it in multi if len(files_of(it)) == 2]
    if len(cross) < profile["min_cross_doc_multi"]:
        errors.append(f"cross-document multi items: {len(cross)}, expected >= {profile['min_cross_doc_multi']}")
    unans = [it for it in items if it.get("type") == "unanswerable"]
    off = [it for it in unans if it.get("note", "").lower().startswith(OFF_TOPIC_PREFIX)]
    if len(unans) - len(off) < profile["min_near_miss"]:
        errors.append(f"near-miss unanswerable: {len(unans) - len(off)}, expected >= {profile['min_near_miss']}")
    if len(off) > profile["max_off_topic"]:
        errors.append(f"off-topic unanswerable: {len(off)}, expected <= {profile['max_off_topic']}")
    ids = [it.get("id") for it in items]
    if len(set(ids)) != len(ids):
        errors.append("duplicate ids")

    for iid, problems in per_item.items():
        errors.extend(f"{iid}: {p}" for p in problems)

    # Balance and spread (reported, not enforced)
    by_file = Counter(f for it in items for f in files_of(it))
    spread = Counter()
    for it in items:
        for e in it.get("evidence", []):
            if e.get("file") in pages and isinstance(e.get("page"), int):
                third = min(2, 3 * (e["page"] - 1) // len(pages[e["file"]]))
                spread[(e["file"], ("beginning", "middle", "end")[third])] += 1
    recitals = sum(1 for it in items if re.search(r"recital", it.get("note", ""), re.IGNORECASE))

    write_review(
        path, items, errors, types, len(cross), len(unans) - len(off), len(off), by_file, spread, recitals, hits_report
    )

    print(
        f"{len(items)} items | types {dict(types)} | cross-doc multi {len(cross)} | "
        f"near-miss {len(unans) - len(off)} | off-topic {len(off)} | recital items {recitals}"
    )
    print("evidence per file:", dict(by_file))
    print("evidence spread:", {f"{f} {p}": n for (f, p), n in sorted(spread.items())})
    for iid, rows in hits_report.items():
        print(f"{iid} term hits: " + ", ".join(f"{t!r}={n}" for t, n, _ in rows))
    if errors:
        print(f"\nFAILED ({len(errors)}):")
        for e in errors:
            print("  -", e)
        return 1
    print("\nALL CHECKS PASSED")
    return 0


def write_review(path, items, errors, types, cross, near, off, by_file, spread, recitals, hits_report):
    out = path.with_name(f"review_{path.stem}.md")
    lines = [f"# Review: {path.name}", "", "Generated by `eval/verify_testset.py`. Not committed.", ""]
    lines += ["## Status", "", "**ALL CHECKS PASSED**" if not errors else f"**{len(errors)} problem(s)**", ""]
    lines += [f"- {e}" for e in errors] + ([""] if errors else [])
    lines += [
        "## Composition",
        "",
        f"- Types: {dict(types)}",
        f"- Multi across both documents: {cross}",
        f"- Unanswerable: {near} near miss, {off} off-topic",
        f"- Items referencing recitals: {recitals}",
        f"- Evidence quotes per file: {dict(by_file)}",
        "- Evidence spread: " + ", ".join(f"{f} {p}: {n}" for (f, p), n in sorted(spread.items())),
        "",
        "## Items",
        "",
    ]
    for it in items:
        lines += [
            f"### {it['id']} · {it['type']}",
            "",
            f"**Q:** {it['question']}",
            "",
            f"**Ground truth:** {it['ground_truth']}",
            "",
            f"**Note:** {it['note']}",
            "",
        ]
        for e in it["evidence"]:
            lines.append(f"- `{e['file']}` p{e['page']}: “{e['quote']}”")
        if it["id"] in hits_report:
            lines.append("")
            lines.append("Term search over both PDFs (confirm the fact is absent):")
            lines.append("")
            for term, n, snippets in hits_report[it["id"]]:
                lines.append(f"- `{term}`: {n} hit(s)")
                lines += [f"  - {s}" for s in snippets]
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Review written to {out}")


if __name__ == "__main__":
    args = sys.argv[1:]
    holdout = "--holdout" in args
    if holdout:
        args.remove("--holdout")
    if len(args) != 1:
        sys.exit("Usage: python eval/verify_testset.py [--holdout] <testset.jsonl>")
    sys.exit(main(Path(args[0]), PROFILES["holdout" if holdout else "dev"]))
