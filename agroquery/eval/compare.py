"""Execution accuracy: do two result sets contain the same answer?

We compare *results*, not SQL text. Rules:
  * same number of rows and columns;
  * columns may appear in any order (names are ignored);
  * row order matters only when the question asks for an ordering;
  * numbers match within a relative tolerance of 1% or an absolute one of 0.01
    (so a model rounding an average to 2 decimals is still correct);
  * dates and timestamps are compared by value; booleans are treated as 0/1.
"""

from __future__ import annotations

import datetime as dt
import math
from decimal import Decimal
from itertools import permutations

REL_TOL = 1e-2
ABS_TOL = 1e-2
MAX_PERMUTED_COLS = 6
GREEDY_MAX_ROWS = 2_000


def normalize(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, int | float | Decimal):
        return float(v)
    if isinstance(v, dt.datetime):
        if v.time() == dt.time(0) and v.tzinfo is None:
            return v.date().isoformat()
        return v.isoformat(sep=" ")
    if isinstance(v, dt.date):
        return v.isoformat()
    if isinstance(v, dt.timedelta):
        return v.total_seconds()
    if isinstance(v, str):
        return v.strip()
    return str(v)


def values_equal(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        return math.isclose(a, b, rel_tol=REL_TOL, abs_tol=ABS_TOL)
    return a == b


def rows_equal(r1: tuple, r2: tuple) -> bool:
    return all(values_equal(a, b) for a, b in zip(r1, r2, strict=True))


def _sort_key(row: tuple):
    # Coarse key: type tag + rounded value, so near-equal numbers sort together.
    out = []
    for v in row:
        if v is None:
            out.append((0, ""))
        elif isinstance(v, float):
            out.append((1, round(v, 1) if not math.isnan(v) else math.inf))
        else:
            out.append((2, str(v)))
    return out


def _unordered_match(gold: list[tuple], pred: list[tuple]) -> bool:
    if all(rows_equal(a, b) for a, b in zip(sorted(gold, key=_sort_key),
                                             sorted(pred, key=_sort_key), strict=True)):
        return True
    if len(gold) > GREEDY_MAX_ROWS:
        return False
    remaining = list(pred)
    for g in gold:
        for i, p in enumerate(remaining):
            if rows_equal(g, p):
                del remaining[i]
                break
        else:
            return False
    return True


def _column_orders(gold: list[tuple], pred: list[tuple], ncols: int):
    if ncols > MAX_PERMUTED_COLS:
        yield tuple(range(ncols))
        return
    g0, p0 = gold[0], pred[0]

    def compatible(gi: int, pi: int) -> bool:
        a, b = g0[gi], p0[pi]
        return a is None or b is None or isinstance(a, float) == isinstance(b, float)

    for perm in permutations(range(ncols)):
        if all(compatible(gi, pi) for gi, pi in enumerate(perm)):
            yield perm


def results_match(gold_rows, pred_rows, order_matters: bool = False) -> bool:
    gold = [tuple(normalize(v) for v in r) for r in gold_rows]
    pred = [tuple(normalize(v) for v in r) for r in pred_rows]
    if len(gold) != len(pred):
        return False
    if not gold:
        return True
    ncols = len(gold[0])
    if len(pred[0]) != ncols:
        return False
    for perm in _column_orders(gold, pred, ncols):
        permuted = [tuple(r[i] for i in perm) for r in pred]
        if order_matters:
            if all(rows_equal(a, b) for a, b in zip(gold, permuted, strict=True)):
                return True
        elif _unordered_match(gold, permuted):
            return True
    return False
