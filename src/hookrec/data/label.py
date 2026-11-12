"""Event -> token labeling."""

from __future__ import annotations

import polars as pl

PAD = 0
LISTEN_FULL = 1
LISTEN_SKIP = 2
LIKE = 3
DISLIKE = 4
N_TOKEN_TYPES = 5
N_RATIO_BUCKETS = 6  # 0 = n/a (non-listen); 1..5 = played-ratio quintiles [0,20),...,[80,100]


def label_events(lf: pl.LazyFrame, full_threshold: int = 50) -> pl.LazyFrame:
    """Add `token` (u8), `ratio_bucket` (u8) and `is_positive` (bool).

    Listens with played_ratio_pct >= full_threshold are LISTEN_FULL, other listens LISTEN_SKIP.
    Positive = LISTEN_FULL or LIKE.
    """
    et = pl.col("event_type").cast(pl.String)
    ratio = pl.col("played_ratio_pct").cast(pl.Int32)
    is_listen = et == "listen"
    token = (
        pl.when(is_listen & (ratio >= full_threshold))
        .then(LISTEN_FULL)
        .when(is_listen)
        .then(LISTEN_SKIP)
        .when(et == "like")
        .then(LIKE)
        .when(et == "dislike")
        .then(DISLIKE)
        .otherwise(None)
        .cast(pl.UInt8)
    )
    bucket = (
        pl.when(is_listen)
        .then((ratio // 20 + 1).clip(upper_bound=5))
        .otherwise(0)
        .cast(pl.UInt8)
    )
    return lf.with_columns(
        token.alias("token"),
        bucket.alias("ratio_bucket"),
        ((token == LISTEN_FULL) | (token == LIKE)).alias("is_positive"),
    )
