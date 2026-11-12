from dataclasses import replace

import polars as pl
import pytest

from hookrec.data.clean import clean_events
from hookrec.data.split import assert_no_leakage, global_temporal_split

DAY = 86400 // 5
GAP = 1800 // 5


@pytest.fixture(scope="module")
def parts(synth):
    lf = clean_events(synth["multi_event"].lazy())
    t0 = lf.select(pl.col("timestamp").min()).collect().item()
    return t0, global_temporal_split(lf)


def _rng(lf):
    ts = pl.col("timestamp")
    return lf.select(ts.min().alias("lo"), ts.max().alias("hi")).collect().row(0)


def test_split_boundaries(parts):
    t0, s = parts
    assert s.t_train_end == t0 + 300 * DAY
    assert s.t_test_start == s.t_train_end + GAP
    assert s.t_test_end == s.t_train_end + DAY
    tr_lo, tr_hi = _rng(s.train)
    assert tr_lo == t0 and tr_hi < s.t_train_end
    te_lo, te_hi = _rng(s.test)
    assert te_lo >= s.t_test_start and te_hi < s.t_test_end
    # trailing skip-only events (~day 309) are excluded from every part
    for part in (s.train, s.val_train, s.val, s.test):
        assert _rng(part)[1] < s.t_test_end
    assert s.test.collect().height > 0
    v_lo, v_hi = _rng(s.val)
    assert v_lo >= s.t_train_end - DAY + GAP and v_hi < s.t_train_end
    vt_hi = _rng(s.val_train)[1]
    assert vt_hi < s.t_train_end - DAY
    assert vt_hi + GAP <= v_lo
    assert_no_leakage(s)


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
