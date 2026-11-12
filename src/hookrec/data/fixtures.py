"""Small structured synthetic Yambda-like data for tests.

Layout follows the official Yambda split (anchored at the last timestamp, DAY = 17280 units):
  t_last      BASE_TS + n_days*DAY - 1 (the last event; nothing exists after it)
  test        [t_last - DAY, t_last]
  gap         the 30 minutes before the test day (no events)
  train       [BASE_TS, t_last - DAY - 30min), >= 300 days with the default n_days=301
Users prefer 1-2 artist clusters; audio embeddings are artist centroid + noise.
Special users (ids): 0-2 have events only in the test window, 3-5 only skip in the test window,
6 only ever skips, 7+ are regular. The last N_NEW items first appear inside the test window.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

DAY = 86400 // 5
GAP = 30 * 60 // 5
N_NEW = 8
BASE_TS = 100_000
N_SPECIAL = 7  # users 0..6 have scripted roles; regular users start here


def _sample_items(rng, artists: np.ndarray, n_pool: int, n_artists: int) -> np.ndarray:
    """Items of `artists` (item i belongs to artist i % n_artists) among ids < n_pool."""
    counts = (n_pool - artists + n_artists - 1) // n_artists
    return artists + n_artists * (rng.random(len(artists)) * counts).astype(np.int64)


def make_synthetic_yambda(
    n_users: int = 50,
    n_items: int = 200,
    n_days: int = 301,
    n_artists: int = 20,
    emb_dim: int = 16,
    missing_audio_frac: float = 0.1,
    seed: int = 0,
) -> dict[str, pl.DataFrame]:
    assert n_users > N_SPECIAL + 2 and n_days >= 301 and n_items > 4 * N_NEW
    rng = np.random.default_rng(seed)
    t_last = BASE_TS + n_days * DAY - 1
    w_start = t_last - DAY  # test window start (inclusive)
    train_span = w_start - GAP - BASE_TS  # train events lie in [BASE_TS, BASE_TS + train_span)
    n_old = n_items - N_NEW
    uids = np.arange(n_users)

    # preferences: 1-2 artist clusters per user
    pref = rng.integers(0, n_artists, size=(n_users, 2))
    two = rng.random(n_users) < 0.5
    pref[~two, 1] = pref[~two, 0]

    # event counts per user for train and test window
    n_train = rng.integers(30, 60, size=n_users)
    n_train[:3] = 0
    n_win = np.full(n_users, 6)
    n_win[:3] = 10
    u_tr = np.repeat(uids, n_train)
    u_w = np.repeat(uids, n_win)

    def draw(u, n_pool):
        n = len(u)
        in_pref = rng.random(n) < 0.85
        artist = pref[u, rng.integers(0, 2, size=n)]
        item = _sample_items(rng, artist, n_pool, n_artists)
        item = np.where(in_pref, item, rng.integers(0, n_pool, size=n))
        return item, in_pref

    item_tr, pref_tr = draw(u_tr, n_old)
    item_w, pref_w = draw(u_w, n_items)
    ts_tr = BASE_TS + rng.integers(0, train_span, size=len(u_tr))
    ts_w = w_start + rng.integers(0, DAY + 1, size=len(u_w))

    # one forced listen per new item inside the window, by a regular user
    new_items = np.arange(n_old, n_items)
    u_new = N_SPECIAL + 1 + (new_items % (n_users - N_SPECIAL - 1))
    ts_new = w_start + rng.integers(0, DAY + 1, size=N_NEW)
    ts_new[0] = t_last  # pin the last timestamp

    uid = np.concatenate([u_tr, u_w, u_new])
    item = np.concatenate([item_tr, item_w, new_items])
    ts = np.concatenate([ts_tr, ts_w, ts_new])
    is_pref = np.concatenate([pref_tr, pref_w, np.ones(N_NEW, bool)])
    in_window = np.concatenate([np.zeros(len(u_tr), bool), np.ones(len(u_w) + N_NEW, bool)])
    forced_pos = np.zeros(len(uid), bool)
    forced_pos[len(u_tr) + len(u_w) : len(u_tr) + len(u_w) + N_NEW] = True
    ts[np.flatnonzero(u_tr == N_SPECIAL + 1)[0]] = BASE_TS  # pin t0

    # event types: mostly listens, a few likes/dislikes/unlikes/undislikes
    r = rng.random(len(uid))
    etype = np.full(len(uid), "listen", dtype=object)
    etype[r < 0.05] = "like"
    etype[(r >= 0.05) & (r < 0.07)] = "dislike"
    etype[(r >= 0.07) & (r < 0.08)] = "unlike"
    etype[(r >= 0.08) & (r < 0.09)] = "undislike"
    regular_train = np.flatnonzero((uid >= N_SPECIAL) & ~in_window)
    etype[regular_train[:4]] = ["like", "dislike", "unlike", "undislike"]
    etype[forced_pos | (uid < N_SPECIAL)] = "listen"

    ratio = np.where(is_pref, rng.integers(60, 101, len(uid)), rng.integers(0, 40, len(uid)))
    skip_only = (uid == 6) | ((uid >= 3) & (uid <= 5) & in_window)
    ratio = np.where(skip_only, rng.integers(0, 40, len(uid)), ratio)
    ratio = np.where((uid < 3) | forced_pos, rng.integers(60, 101, len(uid)), ratio)
    ratio = np.where(forced_pos, 90, ratio)
    listen = etype == "listen"

    track_len = rng.integers(120, 300, size=n_items)[item].astype(np.int64)
    # a few listens with 0/null duration (cleaned in Task 3)
    cand = np.flatnonzero(listen & (uid >= N_SPECIAL) & ~in_window)
    zero_rows, null_rows = cand[:2], cand[2:4]
    track_len[zero_rows] = 0

    ratio_col = pl.Series("played_ratio_pct", ratio).set(pl.Series(~listen), None)
    tl_col = pl.Series("track_length_seconds", track_len).set(
        pl.Series(np.isin(np.arange(len(uid)), null_rows)), None
    )
    me = pl.DataFrame(
        {
            "uid": pl.Series(uid, dtype=pl.UInt32),
            "item_id": pl.Series(item, dtype=pl.UInt32),
            "timestamp": pl.Series(ts, dtype=pl.UInt32),
            "is_organic": pl.Series(rng.integers(0, 2, size=len(uid)), dtype=pl.UInt8),
            "event_type": pl.Series(etype.astype(str), dtype=pl.String),
            "played_ratio_pct": ratio_col.cast(pl.UInt16),
            "track_length_seconds": tl_col.cast(pl.UInt32),
        }
    )
    # a few exact duplicate rows (cleaned in Task 3), then Yambda-style (uid, ts) order
    me = pl.concat([me, me.head(3)]).sort(["uid", "timestamp"], maintain_order=True)

    # audio embeddings: artist centroid + noise; some items have no audio (never the new ones)
    centroids = rng.normal(size=(n_artists, emb_dim))
    artist_of = np.arange(n_items) % n_artists
    embed = (centroids[artist_of] + 0.3 * rng.normal(size=(n_items, emb_dim))).astype(np.float32)
    normed = embed / np.linalg.norm(embed, axis=1, keepdims=True)
    n_missing = round(missing_audio_frac * n_items)
    missing = rng.choice(n_old, size=n_missing, replace=False)
    keep = np.setdiff1d(np.arange(n_items), missing)
    embeddings = pl.DataFrame(
        {
            "item_id": pl.Series(keep, dtype=pl.UInt32),
            "embed": pl.Series(embed[keep].tolist(), dtype=pl.List(pl.Float32)),
            "normalized_embed": pl.Series(normed[keep].tolist(), dtype=pl.List(pl.Float32)),
        }
    )
    mapping = pl.DataFrame(
        {
            "item_id": pl.Series(np.arange(n_items), dtype=pl.UInt32),
            "artist_id": pl.Series(artist_of, dtype=pl.UInt32),
        }
    )
    return {"multi_event": me, "embeddings": embeddings, "artist_item_mapping": mapping}


def write_yambda_layout(data: dict[str, pl.DataFrame], root: str | Path) -> Path:
    """Write `data` as parquet in the Yambda on-disk layout under `root`."""
    root = Path(root)
    (root / "flat" / "50m").mkdir(parents=True, exist_ok=True)
    data["multi_event"].write_parquet(root / "flat" / "50m" / "multi_event.parquet")
    data["embeddings"].write_parquet(root / "embeddings.parquet")
    data["artist_item_mapping"].write_parquet(root / "artist_item_mapping.parquet")
    return root
