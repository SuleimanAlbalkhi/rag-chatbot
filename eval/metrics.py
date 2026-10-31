"""Pure metric functions for the evaluation harness (no I/O, unit-tested in tests/test_eval_metrics.py)."""

import re
from collections.abc import Iterable, Sequence

Page = tuple[str, int]  # (file, 1-based physical page)

# Notes can name an equivalent passage elsewhere, e.g. "GDPR Recital 85 (same rule in Art. 33(1), p52)".
_ALTERNATIVE = re.compile(r"\((?:same rule|list also) in .*?\bp(\d+)\)", re.IGNORECASE)


def gold_pages(item: dict) -> list[set[Page]]:
    """Gold evidence as a list of requirements; each requirement is a set of acceptable pages.

    single: one requirement, satisfied by any evidence page (or an alternative named in the note).
    multi:  one requirement per distinct evidence page; all must be satisfied.
    unanswerable: no requirements.
    """
    pages = list(dict.fromkeys((e["file"], e["page"]) for e in item.get("evidence", [])))
    if not pages:
        return []
    if item["type"] == "multi":
        return [{p} for p in pages]
    file = pages[0][0]
    alternatives = {(file, int(n)) for n in _ALTERNATIVE.findall(item.get("note", ""))}
    return [set(pages) | alternatives]


def first_hit_rank(ranked: Sequence[Page], acceptable: set[Page]) -> int | None:
    """1-based rank of the first retrieved chunk on an acceptable page, or None."""
    return next((i for i, p in enumerate(ranked, 1) if p in acceptable), None)


def hit_at_k(ranked: Sequence[Page], gold: list[set[Page]], k: int) -> bool:
    """True if every gold requirement is met within the top k retrieved chunks."""
    top = ranked[:k]
    return bool(gold) and all(any(p in req for p in top) for req in gold)


def reciprocal_rank(ranked: Sequence[Page], gold: list[set[Page]]) -> float:
    """1 / rank of the first chunk that meets any gold requirement (0 if none in the list)."""
    ranks = [r for r in (first_hit_rank(ranked, req) for req in gold) if r is not None]
    return 1.0 / min(ranks) if ranks else 0.0


def refused_at(top_score: float | None, threshold: float) -> bool:
    """Retrieval-level refusal: no chunk reaches the threshold (as in RAGPipeline.answer)."""
    return top_score is None or top_score < threshold


def threshold_curve(
    top_scores_answerable: Iterable[float | None],
    top_scores_unanswerable: Iterable[float | None],
    thresholds: Iterable[float],
) -> list[dict]:
    """Per threshold: correct refusals on unanswerable and false refusals on answerable questions."""
    ans, unans = list(top_scores_answerable), list(top_scores_unanswerable)
    return [
        {
            "threshold": t,
            "correct_refusals": sum(refused_at(s, t) for s in unans),
            "n_unanswerable": len(unans),
            "false_refusals": sum(refused_at(s, t) for s in ans),
            "n_answerable": len(ans),
        }
        for t in thresholds
    ]


def thresholds(start: float = 0.40, stop: float = 0.80, step: float = 0.05) -> list[float]:
    n = round((stop - start) / step)
    return [round(start + i * step, 2) for i in range(n + 1)]


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolation percentile (q in 0..100), like numpy's default."""
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def fraction(x: int, n: int) -> str:
    """'X of N (pct)' as used throughout the reports."""
    return f"{x} of {n} ({100 * x / n:.0f}%)" if n else f"{x} of 0 (n/a)"


def cohens_kappa(a: Sequence[bool], b: Sequence[bool]) -> float | None:
    """Cohen's kappa for two binary raters. None if undefined (expected agreement is 1)."""
    if len(a) != len(b) or not a:
        raise ValueError("ratings must be non-empty and of equal length")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    if expected == 1:
        return None
    return (observed - expected) / (1 - expected)
