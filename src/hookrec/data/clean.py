"""Event loading and cleaning for Yambda (dedupe, bad listens, like/unlike reconciliation)."""

from __future__ import annotations

from pathlib import Path

import polars as pl

DEDUPE_KEY = ["uid", "item_id", "timestamp", "event_type"]
# (positive event, event that cancels it); each pair is reconciled independently
CANCEL_PAIRS = [("like", "unlike"), ("dislike", "undislike")]


def scan_events(path: Path, columns: list[str] | None = None) -> pl.LazyFrame:
    """Lazily scan an events parquet, projecting only `columns` (all if None)."""
    lf = pl.scan_parquet(path)
    return lf.select(columns) if columns else lf


def _drop_cancelled(lf: pl.LazyFrame, pos: str, neg: str) -> pl.LazyFrame:
    """Keep only `pos` rows that survive stack-matched cancellation by `neg` rows.

    Per (uid, item), each `neg` cancels the most recent earlier uncancelled `pos`; a `neg`
    with no such `pos` is a no-op. Vectorised: with x = +1 (pos) / -1 (neg) in time order and
    S = cumsum(x), the clamped depth D = S - min(0, running min S) is the stack depth with
    unmatched negs reflected away. A pos row i is cancelled iff min(D after i) < D_i.
    """
    ev = pl.col("event_type")
    keys = ["uid", "item_id"]
    rows = (
        lf.filter(ev.is_in([pos, neg]))
        # ties at one timestamp: pos before neg
        .with_columns(x=pl.when(ev == pos).then(1).otherwise(-1).cast(pl.Int32))
        .sort(["uid", "item_id", "timestamp", "x"], descending=[False, False, False, True])
        .with_columns(s=pl.col("x").cum_sum().over(keys))
        .with_columns(
            d=pl.col("s") - pl.min_horizontal(pl.lit(0), pl.col("s").cum_min().over(keys))
        )
        .with_columns(
            min_after=pl.col("d").reverse().cum_min().reverse().shift(-1).over(keys),
        )
    )
    cancelled = pl.col("min_after") < pl.col("d")
    return rows.filter((ev == pos) & ~cancelled.fill_null(False)).select(
        lf.collect_schema().names()
    )


def clean_events(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Dedupe, drop invalid listens, reconcile cancellations, drop cancel rows, sort."""
    names = lf.collect_schema().names()
    ev = pl.col("event_type")
    lf = lf.unique(subset=DEDUPE_KEY, keep="first", maintain_order=True)
    bad_listen = (ev == "listen") & (
        pl.col("track_length_seconds").is_null() | (pl.col("track_length_seconds") == 0)
    )
    lf = lf.filter(~bad_listen).with_columns(pl.col("played_ratio_pct").clip(0, 100))

    pair_events = [e for pair in CANCEL_PAIRS for e in pair]
    parts = [lf.filter(~ev.is_in(pair_events))]
    parts += [_drop_cancelled(lf, pos, neg) for pos, neg in CANCEL_PAIRS]
    return pl.concat(parts).select(names).sort(["uid", "timestamp"], maintain_order=True)


def clean_report(
    before: pl.LazyFrame | pl.DataFrame, after: pl.LazyFrame | pl.DataFrame
) -> dict[str, int]:
    """Row counts before/after cleaning, overall and per event type (flat dict of ints)."""
    report: dict[str, int] = {}
    for tag, lf in (("before", before), ("after", after)):
        if isinstance(lf, pl.DataFrame):  # already materialised: no recomputation
            lf = lf.lazy()
        report[f"rows_{tag}"] = lf.select(pl.len()).collect().item()
        counts = lf.group_by("event_type").len().collect()
        for et, n in counts.iter_rows():
            report[f"{tag}_{et}"] = n
    return report
