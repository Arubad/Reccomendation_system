"""EDA statistics on a cleaned Yambda events LazyFrame (sorted by uid, timestamp).

All aggregations are Polars group_bys; nothing loops over events in Python.
Timestamps are in seconds (on a 5-s grid).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl

TS_UNIT_SECONDS = 1  # raw Yambda timestamps are seconds (official constants.py)
DAY_UNITS = 86400 // TS_UNIT_SECONDS
DEFAULT_SESSION_GAP_SECONDS = 1800.0


def gini(x: np.ndarray) -> float:
    """Gini coefficient of non-negative values (0 = equal, ->1 = concentrated)."""
    x = np.sort(np.asarray(x, dtype=np.float64))
    n, total = len(x), x.sum()
    if n == 0 or total == 0:
        return 0.0
    return float(2 * np.sum(np.arange(1, n + 1) * x) / (n * total) - (n + 1) / n)


def lorenz_curve(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(population share, cumulative value share), both starting at 0 and ending at 1."""
    x = np.sort(np.asarray(x, dtype=np.float64))
    cum = np.concatenate([[0.0], np.cumsum(x)])
    total = cum[-1] if cum[-1] > 0 else 1.0
    return np.linspace(0.0, 1.0, len(cum)), cum / total


def powerlaw_alpha(x: np.ndarray) -> float:
    """Discrete power-law (zeta) MLE for integer counts with xmin = 1.

    Maximises -alpha * sum(ln x) - n * ln(zeta(alpha)). The closed form 1 + n / sum(ln(x / xmin))
    is the continuous MLE and is badly biased for counts at xmin = 1 (about 2.76 for true 2.0).
    """
    from scipy.optimize import minimize_scalar
    from scipy.special import zeta

    x = np.asarray(x, dtype=np.float64)
    x = x[x >= 1]
    if len(x) == 0:
        return float("nan")
    sum_ln, n = np.log(x).sum(), len(x)
    res = minimize_scalar(
        lambda a: a * sum_ln + n * np.log(zeta(a, 1.0)), bounds=(1.01, 6.0), method="bounded"
    )
    return float(res.x)


def entity_counts(lf: pl.LazyFrame) -> tuple[np.ndarray, np.ndarray]:
    """Event counts per item and per user, as two 1-D int arrays."""
    items = lf.group_by("item_id").len().collect()["len"].to_numpy()
    users = lf.group_by("uid").len().collect()["len"].to_numpy()
    return items, users


def popularity_stats(lf: pl.LazyFrame, counts: tuple[np.ndarray, np.ndarray] | None = None) -> dict:
    """Popularity summary; pass precomputed `counts=entity_counts(lf)` to avoid recomputing."""
    item_c, user_c = counts if counts is not None else entity_counts(lf)
    by_type = lf.group_by("event_type").len().collect()
    desc = np.sort(item_c)[::-1]
    total = desc.sum()

    def top_share(frac: float) -> float:
        k = max(1, int(np.ceil(frac * len(desc))))
        return float(desc[:k].sum() / total)

    return {
        "n_users": len(user_c),
        "n_items": len(item_c),
        "n_events": int(total),
        "by_event_type": {str(k): int(v) for k, v in by_type.iter_rows()},
        "item_gini": gini(item_c),
        "user_gini": gini(user_c),
        "top1pct_item_share": top_share(0.01),
        "top10pct_item_share": top_share(0.10),
        "powerlaw_alpha": powerlaw_alpha(item_c),
    }


def played_ratio_hist(lf: pl.LazyFrame, bins: int = 20) -> pl.DataFrame:
    """Listen counts per played-ratio bin (`bin` = lower edge in percent) and organic flag."""
    width = 100 / bins
    return (
        lf.filter((pl.col("event_type") == "listen") & pl.col("played_ratio_pct").is_not_null())
        .with_columns(
            bin=(pl.col("played_ratio_pct").cast(pl.Float64) / width).floor().clip(0, bins - 1)
            * width
        )
        .group_by("bin", pl.col("is_organic").cast(pl.Boolean).alias("organic"))
        .agg(count=pl.len())
        .sort("bin", "organic")
        .select("bin", "organic", "count")
        .collect()
    )


def gap_distribution(lf: pl.LazyFrame, max_users: int | None = None, seed: int = 0) -> np.ndarray:
    """log10 of seconds between consecutive events of a user (zero gaps dropped).

    If `max_users` is set, a fixed random subset of users is used to bound memory.
    """
    lf = lf.select("uid", "timestamp")
    if max_users is not None:
        uids = lf.select("uid").unique().collect()["uid"].sort()
        if len(uids) > max_users:
            keep = np.random.default_rng(seed).choice(len(uids), max_users, replace=False)
            lf = lf.filter(pl.col("uid").is_in(uids.gather(np.sort(keep)).implode()))
    gaps = (
        lf.with_columns(gap=pl.col("timestamp").cast(pl.Int64).diff().over("uid"))
        .filter(pl.col("gap") > 0)
        .select((pl.col("gap") * TS_UNIT_SECONDS).log10())
        .collect()
        .to_series()
    )
    return gaps.to_numpy()


def suggest_session_gap(gaps_log10: np.ndarray) -> float:
    """Seconds at the valley between the two main modes of the log-gap histogram.

    Uses a 100-bin histogram (lightly smoothed). Falls back to 1800 s when there is no clear
    second mode or the valley is not a real dip.
    """
    g = np.asarray(gaps_log10, dtype=np.float64)
    g = g[np.isfinite(g)]
    if len(g) < 100 or g.max() - g.min() < 1e-9:
        return DEFAULT_SESSION_GAP_SECONDS
    counts, edges = np.histogram(g, bins=100)
    h = np.convolve(counts, np.ones(5) / 5, mode="same")
    centers = (edges[:-1] + edges[1:]) / 2
    peaks = [
        i
        for i in range(1, len(h) - 1)
        if h[i] >= h[i - 1] and h[i] > h[i + 1] and h[i] >= 0.05 * h.max()
    ]
    if not peaks:
        return DEFAULT_SESSION_GAP_SECONDS
    p1 = max(peaks, key=lambda i: h[i])
    others = [i for i in peaks if abs(i - p1) >= 10]
    if not others:
        return DEFAULT_SESSION_GAP_SECONDS
    p2 = max(others, key=lambda i: h[i])
    lo, hi = sorted((p1, p2))
    valley = lo + int(np.argmin(h[lo : hi + 1]))
    if h[valley] > 0.9 * min(h[p1], h[p2]):
        return DEFAULT_SESSION_GAP_SECONDS
    return float(10 ** centers[valley])


def _listens(lf: pl.LazyFrame) -> pl.LazyFrame:
    if "event_type" in lf.collect_schema().names():
        return lf.filter(pl.col("event_type") == "listen")
    return lf


def repeat_stats(lf: pl.LazyFrame) -> dict:
    """Re-listen share (item already heard by the user) and consecutive-duplicate share.

    Computed over listens (or all rows if there is no `event_type`). Precondition: `lf` is
    already sorted by (uid, timestamp), as `clean_events` output is.
    """
    r = (
        _listens(lf)
        .select(
            relisten=(pl.int_range(pl.len()).over("uid", "item_id") > 0),
            prev_dup=(pl.col("item_id") == pl.col("item_id").shift(1).over("uid")),
            has_prev=pl.col("item_id").shift(1).over("uid").is_not_null(),
        )
        .select(
            n=pl.len(),
            relisten=pl.col("relisten").sum(),
            dup=pl.col("prev_dup").fill_null(False).sum(),
            n_pairs=pl.col("has_prev").sum(),
        )
        .collect()
        .row(0, named=True)
    )
    return {
        "relisten_share": r["relisten"] / r["n"] if r["n"] else 0.0,
        "consecutive_dup_share": r["dup"] / r["n_pairs"] if r["n_pairs"] else 0.0,
    }


def _decile(rank_col: str, n: int) -> pl.Expr:
    """1..10 decile from a 1-based rank among n entities (10 = highest)."""
    return ((pl.col(rank_col) - 1) * 10 // max(n, 1)).cast(pl.Int32) + 1


def organic_share(
    lf: pl.LazyFrame, by: Literal["day", "user_activity_decile", "item_pop_decile"]
) -> pl.DataFrame:
    """Organic share of events by day, user-activity decile or item-popularity decile."""
    if by == "day":
        t0 = lf.select(pl.col("timestamp").min()).collect().item()
        return (
            lf.group_by(day=(pl.col("timestamp").cast(pl.Int64) - t0) // DAY_UNITS)
            .agg(organic_share=pl.col("is_organic").mean(), events=pl.len())
            .sort("day")
            .collect()
        )
    if by not in ("user_activity_decile", "item_pop_decile"):
        raise ValueError(f"unknown grouping {by!r}")
    key = "uid" if by == "user_activity_decile" else "item_id"
    per = (
        lf.group_by(key)
        .agg(n=pl.len(), org=pl.col("is_organic").sum())
        .sort("n", key)
        .with_columns(rank=pl.int_range(1, pl.len() + 1))
        .collect()
    )
    return (
        per.with_columns(decile=_decile("rank", per.height))
        .group_by("decile")
        .agg(organic_share=pl.col("org").sum() / pl.col("n").sum(), events=pl.col("n").sum())
        .sort("decile")
    )


def daily_stats(lf: pl.LazyFrame, churn_top_k: int = 100) -> pl.DataFrame:
    """Per day: events, newly-appearing items, and popularity churn.

    `churn` is the Jaccard distance between the top-`churn_top_k` items (by events) of
    consecutive days; null on the first day. Day 0 = first day in the data.
    """
    t0 = lf.select(pl.col("timestamp").min()).collect().item()
    day = (pl.col("timestamp").cast(pl.Int64) - t0) // DAY_UNITS
    events = lf.group_by(day=day).agg(events=pl.len())
    first = lf.group_by("item_id").agg(first_ts=pl.col("timestamp").min())
    new = first.group_by(day=(pl.col("first_ts").cast(pl.Int64) - t0) // DAY_UNITS).agg(
        new_items=pl.len()
    )
    out = (
        events.join(new, on="day", how="left")
        .with_columns(pl.col("new_items").fill_null(0))
        .sort("day")
        .collect()
    )
    top = (
        lf.group_by(day=day, item_id=pl.col("item_id"))
        .agg(n=pl.len())
        .sort(["day", "n", "item_id"], descending=[False, True, False])
        .group_by("day", maintain_order=True)
        .head(churn_top_k)
        .group_by("day")
        .agg(items=pl.col("item_id"))
        .collect()
    )
    sets = {d: set(items) for d, items in top.iter_rows()}  # one set per day, not per event
    churn = []
    prev = None
    for d in out["day"].to_list():
        cur = sets.get(d, set())
        union = len(cur | prev) if prev is not None else 0
        churn.append(None if prev is None else (1 - len(cur & prev) / union if union else None))
        prev = cur
    return out.with_columns(pl.Series("churn", churn, dtype=pl.Float64))


def seq_lengths(lf: pl.LazyFrame) -> np.ndarray:
    return lf.group_by("uid").len().collect()["len"].to_numpy()


def seq_length_quantiles(
    lf: pl.LazyFrame, qs: tuple = (0.5, 0.9, 0.95, 0.99), lengths: np.ndarray | None = None
) -> dict:
    """Per-user event-count quantiles (`q50`, `q90`, ...) plus mean and max."""
    if lengths is None:
        lengths = seq_lengths(lf)
    out = {f"q{round(q * 100)}": float(np.quantile(lengths, q)) for q in qs}
    out["mean"] = float(lengths.mean())
    out["max"] = int(lengths.max())
    return out


def embedding_coverage(lf: pl.LazyFrame, emb_item_ids: np.ndarray) -> pl.DataFrame:
    """Audio-embedding coverage by item-popularity decile (10 = most popular)."""
    per = (
        lf.group_by("item_id")
        .agg(n=pl.len())
        .sort("n", "item_id")
        .with_columns(rank=pl.int_range(1, pl.len() + 1))
        .collect()
    )
    covered = pl.Series("emb", np.asarray(emb_item_ids))
    per = per.with_columns(
        decile=_decile("rank", per.height),
        has=pl.col("item_id").is_in(covered.cast(per["item_id"].dtype).implode()),
    )
    return (
        per.group_by("decile")
        .agg(
            n_items=pl.len(),
            coverage=pl.col("has").mean(),
            event_coverage=(pl.col("n") * pl.col("has")).sum() / pl.col("n").sum(),
        )
        .sort("decile")
    )


def like_rate_by_ratio(lf: pl.LazyFrame, bins: int = 10) -> pl.DataFrame:
    """Share of listens whose (user, item) also has a like / dislike, by played-ratio bin."""

    def pairs(event: str, flag: str) -> pl.LazyFrame:
        return (
            lf.filter(pl.col("event_type") == event)
            .select("uid", "item_id")
            .unique()
            .with_columns(pl.lit(True).alias(flag))
        )

    width = 100 / bins
    return (
        lf.filter((pl.col("event_type") == "listen") & pl.col("played_ratio_pct").is_not_null())
        .select(
            "uid",
            "item_id",
            bin=(pl.col("played_ratio_pct").cast(pl.Float64) / width).floor().clip(0, bins - 1)
            * width,
        )
        .join(pairs("like", "liked"), on=["uid", "item_id"], how="left")
        .join(pairs("dislike", "disliked"), on=["uid", "item_id"], how="left")
        .group_by("bin")
        .agg(
            listens=pl.len(),
            like_rate=pl.col("liked").fill_null(False).mean(),
            dislike_rate=pl.col("disliked").fill_null(False).mean(),
        )
        .sort("bin")
        .collect()
    )


def cold_item_share(train_lf: pl.LazyFrame, test_lf: pl.LazyFrame) -> dict:
    """Share of test items (and test events) whose item never occurs in train."""
    train_items = train_lf.select("item_id").unique()
    test = test_lf.group_by("item_id").len()
    unseen = test.join(train_items, on="item_id", how="anti").select(
        n_items=pl.len(), n_events=pl.col("len").sum()
    )
    tot = test.select(n_items=pl.len(), n_events=pl.col("len").sum())
    u = unseen.collect().row(0, named=True)
    t = tot.collect().row(0, named=True)
    return {
        "share_test_items_unseen": u["n_items"] / t["n_items"] if t["n_items"] else 0.0,
        "share_test_events_unseen": (u["n_events"] or 0) / t["n_events"] if t["n_events"] else 0.0,
    }
