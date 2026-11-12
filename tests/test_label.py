import polars as pl

from hookrec.data.label import (
    DISLIKE,
    LIKE,
    LISTEN_FULL,
    LISTEN_SKIP,
    N_RATIO_BUCKETS,
    N_TOKEN_TYPES,
    label_events,
)


def _lab(rows, **kw):
    lf = pl.LazyFrame(
        {
            "event_type": [r[0] for r in rows],
            "played_ratio_pct": pl.Series([r[1] for r in rows], dtype=pl.UInt16),
        }
    )
    return label_events(lf, **kw).collect()


def test_label_thresholds():
    df = _lab([("listen", 49), ("listen", 50), ("like", None), ("dislike", None)])
    assert df["token"].to_list() == [LISTEN_SKIP, LISTEN_FULL, LIKE, DISLIKE]
    assert df["is_positive"].to_list() == [False, True, True, False]
    assert df["token"].dtype == pl.UInt8 and df["ratio_bucket"].dtype == pl.UInt8
    assert (N_TOKEN_TYPES, N_RATIO_BUCKETS) == (5, 6)


def test_ratio_buckets():
    df = _lab([("listen", 0), ("listen", 19), ("listen", 20), ("listen", 100), ("like", None)])
    assert df["ratio_bucket"].to_list() == [1, 1, 2, 5, 0]


def test_custom_threshold():
    df = _lab([("listen", 70)], full_threshold=80)
    assert df["token"].to_list() == [LISTEN_SKIP]


def test_null_ratio_listen_is_skip_bucket_1():
    df = _lab([("listen", None)])
    assert df["token"].to_list() == [LISTEN_SKIP]
    assert df["ratio_bucket"].to_list() == [1]
    assert df["is_positive"].to_list() == [False]


def test_unknown_event_type_not_positive():
    df = _lab([("unlike", None)])
    assert df["is_positive"].to_list() == [False]
