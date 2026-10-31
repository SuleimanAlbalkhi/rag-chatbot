import pytest

from eval.metrics import (
    cohens_kappa,
    fraction,
    gold_pages,
    hit_at_k,
    percentile,
    reciprocal_rank,
    refused_at,
    threshold_curve,
    thresholds,
)

G = "gdpr.pdf"
E = "ehds.pdf"


def single(*pages, note=""):
    return {"type": "single", "evidence": [{"file": f, "page": p} for f, p in pages], "note": note}


def multi(*pages):
    return {"type": "multi", "evidence": [{"file": f, "page": p} for f, p in pages], "note": ""}


def test_gold_single_any_page_and_note_alternative():
    item = single((G, 17), note="GDPR Recital 85 (same rule in Art. 33(1), p52)")
    assert gold_pages(item) == [{(G, 17), (G, 52)}]


def test_gold_single_two_quotes_same_page_is_one_requirement():
    assert gold_pages(single((G, 37), (G, 37))) == [{(G, 37)}]


def test_gold_multi_needs_every_distinct_page():
    assert gold_pages(multi((E, 68), (G, 82))) == [{(E, 68)}, {(G, 82)}]


def test_gold_unanswerable_is_empty():
    assert gold_pages({"type": "unanswerable", "evidence": [], "note": "Searched: x"}) == []


def test_hit_at_k_single():
    gold = [{(G, 52), (G, 17)}]
    ranked = [(G, 1), (G, 2), (G, 52)]
    assert not hit_at_k(ranked, gold, 1)
    assert hit_at_k(ranked, gold, 3)


def test_hit_at_k_multi_requires_all_pages():
    gold = [{(E, 68)}, {(G, 82)}]
    assert not hit_at_k([(E, 68), (E, 1)], gold, 2)
    assert hit_at_k([(E, 68), (E, 1), (G, 82)], gold, 3)


def test_hit_at_k_without_gold_is_false():
    assert not hit_at_k([(G, 1)], [], 10)


def test_reciprocal_rank_uses_first_relevant_chunk():
    assert reciprocal_rank([(G, 1), (G, 52)], [{(G, 52)}]) == 0.5
    assert reciprocal_rank([(E, 68), (G, 82)], [{(G, 82)}, {(E, 68)}]) == 1.0
    assert reciprocal_rank([(G, 1)], [{(G, 52)}]) == 0.0


def test_refused_at_threshold_boundary():
    assert refused_at(0.64, 0.65)
    assert not refused_at(0.65, 0.65)
    assert refused_at(None, 0.40)


def test_threshold_curve_counts():
    curve = threshold_curve([0.9, 0.6, 0.5], [0.7, 0.45], [0.5, 0.65])
    assert curve[0] == {
        "threshold": 0.5,
        "correct_refusals": 1,
        "n_unanswerable": 2,
        "false_refusals": 0,
        "n_answerable": 3,
    }
    assert curve[1]["correct_refusals"] == 1 and curve[1]["false_refusals"] == 2


def test_thresholds_grid():
    assert thresholds() == [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]


def test_percentile():
    assert percentile([], 50) is None
    assert percentile([3.0, 1.0, 2.0], 50) == 2.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 95) == pytest.approx(3.85)


def test_fraction_format():
    assert fraction(3, 12) == "3 of 12 (25%)"
    assert fraction(0, 0) == "0 of 0 (n/a)"


def test_cohens_kappa():
    assert cohens_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert cohens_kappa([True, True, False, False], [True, False, True, False]) == 0.0
    assert cohens_kappa([True, True], [True, True]) is None
    with pytest.raises(ValueError):
        cohens_kappa([True], [True, False])
