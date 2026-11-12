"""One matplotlib plot per EDA statistic. Each saves a PNG and closes the figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from hookrec.eda.stats import gini, lorenz_curve


def _save(fig, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_event_mix(by_event_type: dict, out_path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    ax.bar(list(by_event_type), list(by_event_type.values()))
    ax.set_yscale("log")
    ax.set_ylabel("events (log)")
    ax.set_title("Event-type mix")
    _save(fig, out_path)


def plot_loglog_counts(item_counts: np.ndarray, user_counts: np.ndarray, out_path) -> None:
    """Rank-frequency curves (log-log) for items and users."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, c, name in zip(axes, (item_counts, user_counts), ("item", "user")):
        c = np.sort(c)[::-1]
        ax.loglog(np.arange(1, len(c) + 1), c)
        ax.set_xlabel(f"{name} rank")
        ax.set_ylabel("events")
        ax.set_title(f"Events per {name}")
    _save(fig, out_path)


def plot_lorenz(item_counts: np.ndarray, user_counts: np.ndarray, out_path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5), layout="constrained")
    for c, name in ((item_counts, "items"), (user_counts, "users")):
        x, y = lorenz_curve(c)
        ax.plot(x, y, label=f"{name} (Gini {gini(c):.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("cumulative share of entities")
    ax.set_ylabel("cumulative share of events")
    ax.set_title("Lorenz curves")
    ax.legend()
    _save(fig, out_path)


def plot_played_ratio(hist: pl.DataFrame, out_path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    bins = sorted(hist["bin"].unique().to_list())
    step = (bins[1] - bins[0]) if len(bins) > 1 else 5
    for organic, label in ((True, "organic"), (False, "served")):
        d = hist.filter(pl.col("organic") == organic)
        if d.height:
            ax.bar(
                d["bin"].to_numpy() + (step / 2 if organic else 0),
                d["count"].to_numpy(),
                width=step / 2,
                align="edge",
                label=label,
            )
    ax.set_xlabel("played_ratio_pct")
    ax.set_ylabel("listens")
    ax.set_title("Played ratio: organic vs served")
    ax.legend()
    _save(fig, out_path)


def plot_gaps(gaps_log10: np.ndarray, session_gap_seconds: float, out_path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    if len(gaps_log10):
        ax.hist(gaps_log10, bins=100)
    ax.axvline(np.log10(session_gap_seconds), color="r", ls="--", label="suggested gap")
    ax.set_xlabel("log10 seconds between consecutive events")
    ax.set_ylabel("count")
    ax.set_title("Inter-event gaps")
    ax.legend()
    _save(fig, out_path)


def plot_repeat(stats: dict, out_path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
    ax.bar(
        ["re-listen", "consecutive dup"], [stats["relisten_share"], stats["consecutive_dup_share"]]
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("share")
    ax.set_title("Repeat consumption")
    _save(fig, out_path)


def plot_organic_share(by: dict[str, pl.DataFrame], out_path) -> None:
    """`by` maps day / user_activity_decile / item_pop_decile to organic_share frames."""
    fig, axes = plt.subplots(1, len(by), figsize=(4.5 * len(by), 4), layout="constrained")
    for ax, (name, df) in zip(np.atleast_1d(axes), by.items()):
        x = df.columns[0]
        ax.plot(df[x].to_numpy(), df["organic_share"].to_numpy(), marker="o", ms=3)
        ax.set_ylim(0, 1)
        ax.set_xlabel(x)
        ax.set_ylabel("organic share")
        ax.set_title(name)
    _save(fig, out_path)


def plot_daily(df: pl.DataFrame, out_path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True, layout="constrained")
    axes[0].plot(df["day"].to_numpy(), df["events"].to_numpy())
    axes[0].set_ylabel("events")
    axes[1].plot(df["day"].to_numpy(), df["new_items"].to_numpy())
    axes[1].set_ylabel("new items")
    axes[2].plot(df["day"].to_numpy(), df["churn"].fill_null(float("nan")).to_numpy())
    axes[2].set_ylabel("top-100 churn (Jaccard dist.)")
    axes[2].set_xlabel("day")
    axes[0].set_title("Daily volume, item arrivals and popularity churn")
    _save(fig, out_path)


def plot_seq_length(lengths: np.ndarray, quantiles: dict, out_path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    ax.hist(np.log10(np.maximum(lengths, 1)), bins=60)
    for k, v in quantiles.items():
        if k.startswith("q"):
            ax.axvline(np.log10(max(v, 1)), color="r", ls=":", lw=1)
    ax.set_xlabel("log10 events per user")
    ax.set_ylabel("users")
    ax.set_title("Sequence length (dotted: quantiles)")
    _save(fig, out_path)


def plot_embedding_coverage(df: pl.DataFrame, out_path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    ax.plot(df["decile"].to_numpy(), df["coverage"].to_numpy(), marker="o", label="items")
    ax.plot(df["decile"].to_numpy(), df["event_coverage"].to_numpy(), marker="s", label="events")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("item popularity decile (10 = most popular)")
    ax.set_ylabel("share with audio embedding")
    ax.set_title("Audio-embedding coverage")
    ax.legend()
    _save(fig, out_path)


def plot_like_rate(df: pl.DataFrame, out_path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    ax.plot(df["bin"].to_numpy(), df["like_rate"].to_numpy(), marker="o", label="like")
    ax.plot(df["bin"].to_numpy(), df["dislike_rate"].to_numpy(), marker="s", label="dislike")
    ax.set_xlabel("played_ratio_pct (bin start)")
    ax.set_ylabel("share of listens")
    ax.set_title("Like / dislike rate vs played ratio")
    ax.legend()
    _save(fig, out_path)


def plot_cold_share(stats: dict, out_path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
    ax.bar(
        ["test items\nunseen", "test events\nunseen"],
        [stats["share_test_items_unseen"], stats["share_test_events_unseen"]],
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("share")
    ax.set_title("Cold-start magnitude")
    _save(fig, out_path)


def plot_embedding_map(
    emb: np.ndarray, artist_ids: np.ndarray, out_path, method: str = "pca"
) -> None:
    """2-D projection of (a <=20k sample of) item embeddings, coloured by top artists."""
    if method != "pca":
        raise ValueError(f"unsupported method {method!r}; only 'pca' is available")
    from sklearn.decomposition import PCA

    emb = np.asarray(emb, dtype=np.float32)
    artist_ids = np.asarray(artist_ids)
    if len(emb) > 20_000:
        idx = np.random.default_rng(0).choice(len(emb), 20_000, replace=False)
        emb, artist_ids = emb[idx], artist_ids[idx]
    xy = PCA(n_components=2, random_state=0).fit_transform(emb)
    top, counts = np.unique(artist_ids, return_counts=True)
    top = top[np.argsort(-counts)[:10]]
    fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")
    other = ~np.isin(artist_ids, top)
    ax.scatter(xy[other, 0], xy[other, 1], s=3, c="lightgray", label="other")
    for a in top:
        m = artist_ids == a
        ax.scatter(xy[m, 0], xy[m, 1], s=6, label=f"artist {a}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Audio embeddings (PCA), top-10 artists coloured")
    ax.legend(fontsize=6, markerscale=2)
    _save(fig, out_path)
