"""Yambda Global Temporal Split (GTS).

Paper (arXiv 2505.22238) gives "Train: 300 days, Gap: 30 minutes, Test: 1 day" but does not say
whether the gap is carved out of the test day. We take the gap out of the test day, so the test
window is [t_train_end + gap, t_train_end + 1 day), matching the synthetic fixture window.
Validation repeats the scheme one day earlier: val = [t_train_end - 1d + gap, t_train_end),
val_train = [t0, t_train_end - 1d). Everything at or past t_test_end is dropped from every part.
All boundaries are ints in timestamp units.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass
class Split:
    train: pl.LazyFrame
    val_train: pl.LazyFrame
    val: pl.LazyFrame
    test: pl.LazyFrame
    t_train_end: int
    t_test_start: int
    t_test_end: int
    day_units: int = 86400 // 5  # one day in ts units; sets the val window

    @property
    def t_val_train_end(self) -> int:
        return self.t_train_end - self.day_units

    @property
    def t_val_start(self) -> int:
        return self.t_val_train_end + (self.t_test_start - self.t_train_end)


def global_temporal_split(
    lf: pl.LazyFrame,
    train_days: int = 300,
    gap_seconds: int = 1800,
    test_days: int = 1,
    ts_unit_seconds: int = 5,
) -> Split:
    if train_days < 2:
        raise ValueError(f"train_days must be >= 2 (need a validation day), got {train_days}")
    t0 = lf.select(pl.col("timestamp").min()).collect().item()
    if t0 is None:
        raise ValueError("cannot split an empty event table")
    t0 = int(t0)
    day = 86400 // ts_unit_seconds
    gap = gap_seconds // ts_unit_seconds
    t_train_end = t0 + train_days * day
    t_test_start = t_train_end + gap
    t_test_end = t_train_end + test_days * day
    t_val_train_end = t_train_end - day
    t_val_start = t_val_train_end + gap

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
