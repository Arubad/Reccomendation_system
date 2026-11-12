"""Run the full HookSeq EDA: stats JSON + figures.

python scripts/run_eda.py --raw DIR --size 50m --out reports/ [--gap-max-users N]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from hookrec.data.clean import clean_events, clean_report, scan_events
from hookrec.data.split import YAMBDA_TEST_TIMESTAMP, global_temporal_split
from hookrec.eda import plots, stats

KEEP_COLS = ["uid", "item_id", "timestamp", "is_organic", "event_type", "played_ratio_pct"]


def _jsonable(o):
    if isinstance(o, pl.DataFrame):
        return _jsonable(o.to_dicts())
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None  # NaN/inf -> null, never abort the run
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, required=True, help="Yambda root (flat/<size>/...)")
    ap.add_argument("--size", default="50m")
    ap.add_argument("--out", type=Path, default=Path("reports"))
    ap.add_argument("--gap-max-users", type=int, default=None)
    ap.add_argument(
        "--test-timestamp",
        default=None,
        help="test-window start (seconds); 'none' = max-anchored. Default: the official "
        "YAMBDA_TEST_TIMESTAMP for real sizes (50m/500m/5b), max-anchored otherwise",
    )
    args = ap.parse_args(argv)

    fig_dir = args.out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    raw = scan_events(args.raw / "flat" / args.size / "multi_event.parquet")
    cleaned = clean_events(raw)
    # clean once, keep only needed columns in memory for every stat
    df = cleaned.select(KEEP_COLS).collect()
    df = df.with_columns(pl.col("event_type").cast(pl.Categorical))
    lf = df.lazy()
    report = clean_report(raw, df)

    out: dict = {"clean_report": report}
    item_c, user_c = counts = stats.entity_counts(lf)
    out["popularity"] = stats.popularity_stats(lf, counts)
    plots.plot_event_mix(out["popularity"]["by_event_type"], fig_dir / "event_mix.png")
    plots.plot_loglog_counts(item_c, user_c, fig_dir / "loglog_counts.png")
    plots.plot_lorenz(item_c, user_c, fig_dir / "lorenz.png")

    ratio = stats.played_ratio_hist(lf)
    out["played_ratio_hist"] = ratio
    plots.plot_played_ratio(ratio, fig_dir / "played_ratio.png")

    gaps = stats.gap_distribution(lf, max_users=args.gap_max_users)
    session_gap = stats.suggest_session_gap(gaps)
    out["gap"] = {"n_gaps": len(gaps), "suggested_session_gap_seconds": session_gap}
    plots.plot_gaps(gaps, session_gap, fig_dir / "gaps.png")
    del gaps

    out["repeat"] = stats.repeat_stats(lf)
    plots.plot_repeat(out["repeat"], fig_dir / "repeat.png")

    org = {
        "day": stats.organic_share(lf, "day"),
        "user_activity_decile": stats.organic_share(lf, "user_activity_decile"),
        "item_pop_decile": stats.organic_share(lf, "item_pop_decile"),
    }
    out["organic_share"] = org
    plots.plot_organic_share(org, fig_dir / "organic_share.png")

    out["daily"] = stats.daily_stats(lf)
    plots.plot_daily(out["daily"], fig_dir / "daily.png")

    lengths = stats.seq_lengths(lf)
    out["seq_length"] = stats.seq_length_quantiles(lf, lengths=lengths)
    plots.plot_seq_length(lengths, out["seq_length"], fig_dir / "seq_length.png")

    emb = pl.read_parquet(args.raw / "embeddings.parquet", columns=["item_id", "normalized_embed"])
    out["embedding_coverage"] = stats.embedding_coverage(lf, emb["item_id"].to_numpy())
    plots.plot_embedding_coverage(out["embedding_coverage"], fig_dir / "embedding_coverage.png")

    mapping = pl.read_parquet(args.raw / "artist_item_mapping.parquet")
    sample = emb.sample(min(emb.height, 20_000), seed=0).join(mapping, on="item_id", how="inner")
    plots.plot_embedding_map(
        np.asarray(sample["normalized_embed"].to_list(), dtype=np.float32),
        sample["artist_id"].to_numpy(),
        fig_dir / "embedding_map.png",
    )

    out["like_rate_by_ratio"] = stats.like_rate_by_ratio(lf)
    plots.plot_like_rate(out["like_rate_by_ratio"], fig_dir / "like_rate.png")

    if args.test_timestamp is None:
        test_ts = YAMBDA_TEST_TIMESTAMP if args.size in ("50m", "500m", "5b") else None
    else:
        test_ts = None if args.test_timestamp.lower() == "none" else int(args.test_timestamp)
    split = global_temporal_split(
        lf, ts_unit_seconds=stats.TS_UNIT_SECONDS, test_timestamp=test_ts
    )
    out["cold_items"] = stats.cold_item_share(split.train, split.test)
    plots.plot_cold_share(out["cold_items"], fig_dir / "cold_items.png")

    out = _jsonable(out)
    (args.out / "eda_stats.json").write_text(json.dumps(out, indent=2, allow_nan=False))
    return out


if __name__ == "__main__":
    main()
