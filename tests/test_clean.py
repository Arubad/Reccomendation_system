import polars as pl

from hookrec.data.clean import clean_events, clean_report, scan_events


def _ev(rows):
    """rows: (uid, item_id, ts, event_type[, ratio, track_len]) -> Yambda-style LazyFrame."""
    out = []
    for r in rows:
        uid, item, ts, et = r[:4]
        ratio = r[4] if len(r) > 4 else None
        tl = r[5] if len(r) > 5 else None
        out.append((uid, item, ts, 1, et, ratio, tl))
    return pl.LazyFrame(
        out,
        schema={
            "uid": pl.UInt32,
            "item_id": pl.UInt32,
            "timestamp": pl.UInt32,
            "is_organic": pl.UInt8,
            "event_type": pl.String,
            "played_ratio_pct": pl.Int32,
            "track_length_seconds": pl.Int32,
        },
        orient="row",
    )


def _survivors(rows, et="like"):
    df = clean_events(_ev(rows)).collect()
    return df.filter(pl.col("event_type") == et)["timestamp"].to_list()


def test_dedupe():
    df = clean_events(_ev([(1, 1, 5, "like"), (1, 1, 5, "like"), (1, 2, 5, "like")])).collect()
    assert len(df) == 2


def test_unlike_cancels_previous_like_only():
    rows = [(1, 7, 1, "like"), (1, 7, 3, "like"), (1, 7, 4, "unlike")]
    assert _survivors(rows) == [1]


def test_unlike_before_any_like_is_dropped_noop():
    rows = [(1, 7, 1, "unlike"), (1, 7, 2, "like")]
    df = clean_events(_ev(rows)).collect()
    assert df["timestamp"].to_list() == [2]
    assert "unlike" not in df["event_type"].to_list()


def test_like_like_unlike_unlike_like_only_last_survives():
    rows = [
        (1, 7, 1, "like"),
        (1, 7, 2, "like"),
        (1, 7, 3, "unlike"),
        (1, 7, 4, "unlike"),
        (1, 7, 5, "like"),
    ]
    assert _survivors(rows) == [5]


def test_unmatched_unlike_does_not_cancel_later_like():
    rows = [(1, 7, 1, "unlike"), (1, 7, 2, "unlike"), (1, 7, 3, "like"), (1, 7, 4, "like")]
    assert _survivors(rows) == [3, 4]


def test_same_timestamp_like_ordered_before_unlike():
    rows = [(1, 7, 3, "unlike"), (1, 7, 3, "like")]
    assert _survivors(rows) == []


def test_cancellation_scoped_per_uid_item_and_pair():
    rows = [
        (1, 7, 1, "like"),
        (2, 7, 2, "unlike"),  # other user: no-op
        (1, 8, 3, "unlike"),  # other item: no-op
        (1, 7, 4, "dislike"),
        (1, 7, 5, "unlike"),  # cancels like, not dislike
        (1, 9, 6, "dislike"),
        (1, 9, 7, "undislike"),
        (1, 9, 8, "like"),
    ]
    df = clean_events(_ev(rows)).collect()
    got = sorted(zip(df["uid"], df["item_id"], df["event_type"], strict=True))
    assert got == [(1, 7, "dislike"), (1, 9, "like")]


def test_invalid_duration_dropped():
    rows = [
        (1, 1, 1, "listen", 50, 200),
        (1, 2, 2, "listen", 50, 0),
        (1, 3, 3, "listen", 50, None),
        (1, 4, 4, "like"),
    ]
    df = clean_events(_ev(rows)).collect()
    assert df["item_id"].to_list() == [1, 4]


def test_played_ratio_clipped():
    rows = [(1, 1, 1, "listen", 250, 100), (1, 2, 2, "listen", -5, 100)]
    df = clean_events(_ev(rows)).collect()
    assert df["played_ratio_pct"].to_list() == [100, 0]


def test_sorted_output():
    rows = [(2, 1, 9, "like"), (1, 1, 5, "like"), (2, 2, 3, "like"), (1, 2, 1, "like")]
    df = clean_events(_ev(rows)).collect()
    assert list(zip(df["uid"], df["timestamp"], strict=True)) == [(1, 1), (1, 5), (2, 3), (2, 9)]


def test_clean_report_counts():
    before = _ev([(1, 1, 1, "like"), (1, 1, 2, "unlike"), (1, 2, 3, "like")])
    rep = clean_report(before, clean_events(before))
    assert rep["rows_before"] == 3
    assert rep["rows_after"] == 1
    assert rep["before_like"] == 2 and rep["after_like"] == 1
    assert rep["before_unlike"] == 1 and "after_unlike" not in rep


def test_scan_and_clean_synth(synth, synth_dir):
    path = synth_dir / "flat" / "50m" / "multi_event.parquet"
    cols = ["uid", "item_id", "timestamp", "event_type"]
    assert scan_events(path, cols).collect_schema().names() == cols
    out = clean_events(scan_events(path)).collect()
    assert len(out) < synth["multi_event"].height
    assert not out["event_type"].is_in(["unlike", "undislike"]).any()
    listens = out.filter(pl.col("event_type") == "listen")
    assert (listens["track_length_seconds"] > 0).all()
    key = out.select("uid", "item_id", "timestamp", "event_type")
    assert not key.is_duplicated().any()
    assert out.equals(out.sort(["uid", "timestamp"], maintain_order=True))
