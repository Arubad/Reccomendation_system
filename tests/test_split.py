from dataclasses import replace

import polars as pl
import pytest

from hookrec.data.clean import clean_events
from hookrec.data.split import Split, assert_no_leakage, global_temporal_split

DAY = 86400 // 5
GAP = 1800 // 5


@pytest.fixture(scope="module")
def parts(synth):
    lf = clean_events(synth["multi_event"].lazy())
    t_last = lf.select(pl.col("timestamp").max()).collect().item()
    return t_last, global_temporal_split(lf)


def _rng(lf):
    ts = pl.col("timestamp")
    return lf.select(ts.min().alias("lo"), ts.max().alias("hi")).collect().row(0)


def test_split_boundaries(parts):
    t_last, s = parts
    assert s.t_test_start == t_last - DAY
    assert s.t_test_end == t_last + 1
    assert s.t_train_end == s.t_test_start - GAP  # gap sits on the train side
    te_lo, te_hi = _rng(s.test)
    assert te_lo >= s.t_test_start and te_hi == t_last
    tr_lo, tr_hi = _rng(s.train)
    assert tr_hi < s.t_train_end
    assert s.t_train_end - tr_lo >= 299 * DAY  # train keeps the earliest history
    assert s.test.collect().height > 0
    v_lo, v_hi = _rng(s.val)
    assert v_lo >= s.t_train_end - DAY and v_hi < s.t_train_end
    vt_hi = _rng(s.val_train)[1]
    assert vt_hi < s.t_train_end - DAY - GAP
    assert s.t_val_start == s.t_train_end - DAY
    assert s.t_val_train_end == s.t_val_start - GAP
    assert_no_leakage(s)


def test_test_is_full_day_and_gap_is_empty_in_train(parts):
    _, s = parts
    assert s.t_test_end - s.t_test_start == DAY + 1  # [t_last - day, t_last]
    assert s.t_test_start - s.t_train_end == GAP
    ts = pl.col("timestamp")
    in_gap = (ts >= s.t_train_end) & (ts < s.t_test_start)
    for part in (s.train, s.val, s.test):
        assert part.filter(in_gap).collect().height == 0


def test_split_anchored_at_last_ts_not_first():
    lf = pl.DataFrame({"timestamp": [10, 500_000, 600_000]}).lazy()
    s = global_temporal_split(lf)
    assert s.t_test_start == 600_000 - DAY
    assert s.train.collect()["timestamp"].to_list() == [10, 500_000]
    assert s.test.collect()["timestamp"].to_list() == [600_000]


def test_assert_no_leakage_raises(parts):
    _, s = parts
    lf = s.train
    overlap = replace(s, train=lf, val_train=lf, val=lf, test=lf, t_test_start=s.t_train_end - 1)
    with pytest.raises(ValueError):
        assert_no_leakage(overlap)
    # test part holding train events (outside the test window)
    with pytest.raises(ValueError):
        assert_no_leakage(replace(s, test=s.train))
    # val_train overlapping the val window
    with pytest.raises(ValueError):
        assert_no_leakage(replace(s, val_train=s.train))


def test_split_seven_field_construction(parts):
    _, s = parts
    s7 = Split(s.train, s.val_train, s.val, s.test, s.t_train_end, s.t_test_start, s.t_test_end)
    assert s7.t_val_train_end == s.t_val_train_end and s7.t_val_start == s.t_val_start
    assert_no_leakage(s7)


def test_assert_no_leakage_val_branches(parts):
    _, s = parts
    with pytest.raises(ValueError, match="val events"):
        assert_no_leakage(replace(s, val=s.train))
    # val_train must stay before the val window (gap enforced by derived bounds)
    with pytest.raises(ValueError, match="val_train"):
        assert_no_leakage(replace(s, val_train=s.train))


def test_split_input_validation(synth):
    lf = synth["multi_event"].lazy()
    with pytest.raises(ValueError, match="empty"):
        global_temporal_split(lf.head(0))
    with pytest.raises(ValueError, match="train_days"):
        global_temporal_split(lf, train_days=1)
