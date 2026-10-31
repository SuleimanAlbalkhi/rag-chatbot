import pytest

from eval.judge_agreement import compare_verdict

COLUMN = "human_correct (1/0)"
JUDGE = {"D-001": {"correct": True}, "D-002": {"correct": True}, "D-003": {"correct": False}}


def rows(*labels):
    return [{"id": f"D-00{i}", COLUMN: label} for i, label in enumerate(labels, 1)]


def test_all_empty_column_is_not_validated():
    assert compare_verdict(rows("", " ", ""), JUDGE, "correct", COLUMN) is None


def test_missing_column_is_not_validated():
    assert compare_verdict([{"id": "D-001"}], JUDGE, "correct", COLUMN) is None


def test_partly_filled_column_is_an_error():
    with pytest.raises(ValueError, match="D-002"):
        compare_verdict(rows("1", "", "0"), JUDGE, "correct", COLUMN)


def test_agreement_kappa_and_disagreements():
    stats = compare_verdict(rows("1", "0", "0"), JUDGE, "correct", COLUMN)
    assert stats["n"] == 3 and stats["agreement"] == 2
    assert stats["disagreements"] == [{"id": "D-002", "human": False, "judge": True}]
    assert stats["kappa"] == pytest.approx(0.4)
