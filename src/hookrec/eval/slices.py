"""Evaluation slices of the test ground truth, and per-slice metrics on stored top-k lists."""

from __future__ import annotations

import numpy as np
import polars as pl

from hookrec.data.features import ItemFeatures
from hookrec.data.sequences import PackedSequences
from hookrec.data.vocab import ItemVocab
from hookrec.eval.metrics import coverage, ndcg_per_user, recall_per_user

Truth = dict[int, np.ndarray]


def _positive_items_by_user(events: pl.LazyFrame, organic: bool) -> Truth:
    df = (
        events.filter(pl.col("is_positive").cast(pl.Boolean) & (pl.col("is_organic").cast(pl.Boolean) == organic))
        .select("uid", "item_idx")
        .unique()
        .sort("uid", "item_idx")
        .collect()
    )
    out: Truth = {}
    if df.height:
        uids = df["uid"].to_numpy()
        items = df["item_idx"].to_numpy().astype(np.int64)
        cuts = np.flatnonzero(np.diff(uids)) + 1
        starts = np.concatenate([[0], cuts])
        for u, g in zip(uids[starts], np.split(items, cuts)):
            out[int(u)] = g
    return out


def _restrict(truth: Truth, allowed: Truth) -> Truth:
    out: Truth = {}
    for u, items in truth.items():
        keep = items[np.isin(items, allowed.get(u, items[:0]))]
        if len(keep):
            out[u] = keep
    return out


def slice_truth(
    truth: Truth,
    test_events_lf: pl.LazyFrame,
    vocab: ItemVocab,
    features: ItemFeatures,
    seqs_train: PackedSequences,
) -> dict[str, Truth]:
    """Named sub-truths; users whose slice is empty are dropped from that slice.

    organic / served: truth items with >= 1 positive organic / non-organic test event (an item
    may be in both). cold_items: OOV (idx > N) or no ID embedding; warm_items: the rest.
    users_low/mid/high: terciles of train event count over the users in `truth` (ties by uid).
    `features` is part of the interface but unused (cold is defined by the vocab's id_idx).
    """
    n = vocab.n_items
    out: dict[str, Truth] = {"all": {u: t for u, t in truth.items() if len(t)}}
    out["organic"] = _restrict(truth, _positive_items_by_user(test_events_lf, True))
    out["served"] = _restrict(truth, _positive_items_by_user(test_events_lf, False))

    def is_cold(items: np.ndarray) -> np.ndarray:
        return (items > n) | (vocab.id_idx[np.minimum(items, n)] == 0)

    for name, want_cold in (("cold_items", True), ("warm_items", False)):
        sl: Truth = {}
        for u, items in truth.items():
            keep = items[is_cold(items) == want_cold]
            if len(keep):
                sl[u] = keep
        out[name] = sl

    uids = np.array(sorted(truth), dtype=np.int64)
    rows = seqs_train.index_of(uids)
    lens = np.diff(seqs_train.offsets)
    counts = np.where(rows >= 0, lens[np.maximum(rows, 0)] if len(lens) else 0, 0)
    order = np.lexsort((uids, counts))  # by activity, ties by uid
    for name, part in zip(("users_low", "users_mid", "users_high"), np.array_split(order, 3)):
        sl = {}
        for i in part:
            t = truth[int(uids[i])]
            if len(t):
                sl[int(uids[i])] = t
        out[name] = sl
    return out


def evaluate_slices(
    topk: np.ndarray,
    uids: np.ndarray,
    slices: dict[str, Truth],
    n_items: int,
    ks: tuple[int, ...] = (10, 100),
) -> dict[str, dict[str, float]]:
    """Metrics per slice from stored `topk` rows (aligned with `uids`); never re-scores.

    Each slice is scored over its own users that are in `uids`, against its own truth.
    """
    pos = {int(u): i for i, u in enumerate(uids)}
    out: dict[str, dict[str, float]] = {}
    for name, sl in slices.items():
        users = [u for u in sorted(sl) if u in pos and len(sl[u])]
        m: dict[str, float] = {"n_users": float(len(users))}
        if users:
            rows = topk[[pos[u] for u in users]]
            tl = [sl[u] for u in users]
            for k in ks:
                m[f"ndcg@{k}"] = float(ndcg_per_user(rows, tl, k).mean())
                m[f"recall@{k}"] = float(recall_per_user(rows, tl, k).mean())
                m[f"coverage@{k}"] = coverage(rows, n_items, k)
        out[name] = m
    return out
