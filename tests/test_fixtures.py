import polars as pl

DAY = 86400 // 5  # one day in 5-second timestamp units
GAP = 30 * 60 // 5


def _window(me):
    t_last = me["timestamp"].max()
    return me["timestamp"].min(), t_last - DAY, t_last + 1


def test_schema(synth):
    me, emb, mapping = synth["multi_event"], synth["embeddings"], synth["artist_item_mapping"]
    assert me.schema == {
        "uid": pl.UInt32,
        "item_id": pl.UInt32,
        "timestamp": pl.UInt32,
        "is_organic": pl.UInt8,
        "event_type": pl.String,
        "played_ratio_pct": pl.UInt16,
        "track_length_seconds": pl.UInt32,
    }
    assert emb.schema == {
        "item_id": pl.UInt32,
        "embed": pl.List(pl.Float32),
        "normalized_embed": pl.List(pl.Float32),
    }
    assert set(mapping.columns) == {"item_id", "artist_id"}
    assert set(me["event_type"].unique()) == {"listen", "like", "dislike", "unlike", "undislike"}
    is_listen = me["event_type"] == "listen"
    assert me.filter(is_listen)["played_ratio_pct"].null_count() == 0
    assert me.filter(~is_listen)["played_ratio_pct"].null_count() == (~is_listen).sum()

    n_items = mapping["item_id"].n_unique()
    missing = 1 - emb["item_id"].n_unique() / n_items
    assert 0.05 <= missing <= 0.15

    assert me["timestamp"].max() - me["timestamp"].min() >= 301 * DAY - 1


def test_no_events_after_test_day_and_train_before_gap(synth):
    me = synth["multi_event"]
    t0, w_start, w_end = _window(me)
    assert w_end - 1 == me["timestamp"].max()
    before = me.filter(pl.col("timestamp") < w_start)["timestamp"]
    assert before.max() < w_start - GAP and w_start - GAP - t0 >= 299 * DAY


def test_test_window_structure(synth):
    me = synth["multi_event"]
    _, w_start, w_end = _window(me)
    positive = ((pl.col("event_type") == "listen") & (pl.col("played_ratio_pct") >= 50)) | (
        pl.col("event_type") == "like"
    )
    in_win = (pl.col("timestamp") >= w_start) & (pl.col("timestamp") < w_end)

    # >= 3 users with zero positive listens in the test window
    pos_users = me.filter(in_win & positive)["uid"].unique()
    assert me["uid"].n_unique() - pos_users.len() >= 3

    # >= 1 user has only skips (listens below the full threshold)
    per_user = me.group_by("uid").agg(
        (positive.any()).alias("any_pos"), (pl.col("event_type") == "listen").all().alias("listens")
    )
    assert per_user.filter(~pl.col("any_pos") & pl.col("listens")).height >= 1

    # >= 5 items first appear inside the test window
    first = me.group_by("item_id").agg(pl.col("timestamp").min().alias("first"))
    new = first.filter((pl.col("first") >= w_start) & (pl.col("first") < w_end))
    assert new.height >= 5


def test_users_without_train_history(synth):
    me = synth["multi_event"]
    _, w_start, w_end = _window(me)
    per_user = me.group_by("uid").agg(
        pl.col("timestamp").min().alias("lo"), pl.col("timestamp").max().alias("hi")
    )
    only_test = per_user.filter((pl.col("lo") >= w_start) & (pl.col("hi") < w_end))
    assert only_test.height >= 3


def test_listens_have_track_length_mostly(synth):
    listens = synth["multi_event"].filter(pl.col("event_type") == "listen")
    bad = listens.filter(
        pl.col("track_length_seconds").is_null() | (pl.col("track_length_seconds") == 0)
    )
    assert bad.height <= 0.02 * listens.height


def test_audio_embeddings_cluster_by_artist(synth):
    import numpy as np

    emb = synth["embeddings"].join(synth["artist_item_mapping"], on="item_id")
    x = np.array(emb["normalized_embed"].to_list())
    art = emb["artist_id"].to_numpy()
    same = (art[:, None] == art[None, :]) & ~np.eye(len(art), dtype=bool)
    sim = x @ x.T
    assert sim[same].mean() > sim[~same & ~np.eye(len(art), dtype=bool)].mean() + 0.3


def test_synth_dir_layout(synth_dir):
    assert (synth_dir / "flat" / "50m" / "multi_event.parquet").exists()
    assert (synth_dir / "embeddings.parquet").exists()
    assert (synth_dir / "artist_item_mapping.parquet").exists()
    assert pl.read_parquet(synth_dir / "flat" / "50m" / "multi_event.parquet").height > 0
