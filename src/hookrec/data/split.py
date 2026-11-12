"""Yambda Global Temporal Split (GTS), mirroring the official benchmark code.

Source: `benchmarks/yambda/processing/timesplit.py` (`flat_split_train_val_test`, docstring and
the `train_timestamp = test_timestamp - gap_size - val_size - gap_size` line) and
`benchmarks/yambda/constants.py` (GAP_SIZE = 30 min, VAL_SIZE = TEST_SIZE = 1 day,
TEST_TIMESTAMP = LAST_TIMESTAMP - TEST_SIZE) in the HF dataset repo `yandex/yambda`.

The split is anchored at the LAST timestamp, not the first:
  t_test_start = t_last - test_days * day; test = ts >= t_test_start (t_test_end = t_last + 1)
  t_train_end  = t_test_start - gap;       train = ts < t_train_end
  val mirrors it inside train: val = [t_train_end - day, t_train_end),
  val_train = ts < t_val_start - gap  (the official "train" when val_size != 0).
The gap is carved from the train side; the test day is a full day. The official code does not
bound train from below (train = [0, ...)), so `train_days` is only validated (>= 2) and does not
cut the start of history. All boundaries are ints in timestamp units. Raw Yambda timestamps are seconds (on a 5-s grid),
so the defaults are `ts_unit_seconds=1`, day = 86400, gap = 1800.

`test_timestamp` pins the test start explicitly (official `YAMBDA_TEST_TIMESTAMP`); when None it is
`max(ts) - test_days * day`. `train_days` is validated only, kept for interface compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

# constants.py:9-10 in yandex/yambda: TEST_TIMESTAMP = LAST_TIMESTAMP (26_000_000) - TEST_SIZE (1 day)
YAMBDA_TEST_TIMESTAMP = 26_000_000 - 86_400


@dataclass
class Split:
    train: pl.LazyFrame
    val_train: pl.LazyFrame
    val: pl.LazyFrame
    test: pl.LazyFrame
    t_train_end: int
    t_test_start: int
    t_test_end: int
    day_units: int = 86400  # one day in ts units (seconds); sets the val window
    gap_units: int = 1800  # gap in ts units (test_start - train_end)

    @property
    def t_val_start(self) -> int:
        return self.t_train_end - self.day_units

    @property
    def t_val_train_end(self) -> int:
        return self.t_val_start - self.gap_units


def global_temporal_split(
    lf: pl.LazyFrame,
    train_days: int = 300,
    gap_seconds: int = 1800,
    test_days: int = 1,
    ts_unit_seconds: int = 1,
    test_timestamp: int | None = None,
) -> Split:
    if train_days < 2:
        raise ValueError(f"train_days must be >= 2 (need a validation day), got {train_days}")
    t_last = lf.select(pl.col("timestamp").max()).collect(engine="streaming").item()
    if t_last is None:
        raise ValueError("cannot split an empty event table")
    t_last = int(t_last)
    day = 86400 // ts_unit_seconds
    gap = gap_seconds // ts_unit_seconds
    if test_timestamp is None:
        t_test_start = t_last - test_days * day
    else:
        t_test_start = int(test_timestamp)
    t_test_end = max(t_last, t_test_start) + 1
    t_train_end = t_test_start - gap
    t_val_start = t_train_end - day
    t_val_train_end = t_val_start - gap

    ts = pl.col("timestamp")
    return Split(
        train=lf.filter(ts < t_train_end),
        val_train=lf.filter(ts < t_val_train_end),
        val=lf.filter((ts >= t_val_start) & (ts < t_train_end)),
        test=lf.filter((ts >= t_test_start) & (ts < t_test_end)),
        t_train_end=t_train_end,
        t_test_start=t_test_start,
        t_test_end=t_test_end,
        day_units=day,
        gap_units=gap,
    )


def _bounds(lf: pl.LazyFrame) -> tuple[int | None, int | None]:
    ts = pl.col("timestamp")
    return lf.select(ts.min().alias("lo"), ts.max().alias("hi")).collect().row(0)


def assert_no_leakage(split: Split) -> None:
    """Raise ValueError if the split's parts overlap or leave their windows."""
    s = split
    errs = []
    if not s.t_train_end <= s.t_test_start < s.t_test_end:
        errs.append("train must end before the test window")
    if not s.t_val_start < s.t_train_end:
        errs.append("val window must be inside train")

    _, tr_hi = _bounds(s.train)
    if tr_hi is not None and tr_hi >= min(s.t_train_end, s.t_test_start):
        errs.append(f"train reaches the test window (max ts {tr_hi})")
    v_lo, v_hi = _bounds(s.val)
    if v_lo is not None and (v_lo < s.t_val_start or v_hi >= s.t_train_end):
        errs.append("val events fall outside [t_val_start, t_train_end)")
    _, vt_hi = _bounds(s.val_train)
    if vt_hi is not None and vt_hi >= s.t_val_train_end:
        errs.append("val_train reaches the val window")
    te_lo, te_hi = _bounds(s.test)
    if te_lo is not None and (te_lo < s.t_test_start or te_hi >= s.t_test_end):
        errs.append("test events fall outside [t_test_start, t_test_end)")
    if errs:
        raise ValueError("temporal leakage: " + "; ".join(errs))
