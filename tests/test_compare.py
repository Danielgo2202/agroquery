import datetime as dt
from decimal import Decimal

from agroquery.eval.compare import results_match


def test_identical_results_match():
    rows = [("Meta", 10), ("Huila", 7)]
    assert results_match(rows, list(rows))


def test_row_order_ignored_unless_required():
    gold = [("a", 1), ("b", 2)]
    pred = [("b", 2), ("a", 1)]
    assert results_match(gold, pred, order_matters=False)
    assert not results_match(gold, pred, order_matters=True)


def test_column_order_ignored():
    assert results_match([("Meta", 10.0)], [(10, "Meta")])


def test_numeric_tolerance_allows_rounding():
    assert results_match([(23.41667,)], [(23.42,)])
    assert results_match([(Decimal("1503.2456"),)], [(1503.2,)])
    assert not results_match([(23.4,)], [(25.0,)])


def test_row_and_column_counts_must_match():
    assert not results_match([(1,), (2,)], [(1,)])
    assert not results_match([("a", 1)], [("a", 1, "extra")])


def test_dates_and_timestamps_normalized():
    assert results_match([(dt.date(2024, 3, 1),)], [(dt.datetime(2024, 3, 1),)])


def test_booleans_match_zero_one():
    assert results_match([(True, 5.0), (False, 4.0)], [(1, 5.0), (0, 4.0)])


def test_duplicates_counted():
    assert not results_match([("a",), ("a",)], [("a",), ("b",)])


def test_empty_results():
    assert results_match([], [])
    assert not results_match([("a",)], [])
