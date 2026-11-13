"""Evaluation slices of the test ground truth, and per-slice metrics on stored top-k lists."""

from __future__ import annotations

import numpy as np
import polars as pl

from hookrec.data.aggregates import PeriodAggregates
from hookrec.data.features import ItemFeatures
from hookrec.data.sequences import PackedSequences
from hookrec.data.vocab import ItemVocab
from hookrec.eval.metrics import coverage, ndcg_per_user, recall_full_per_user, recall_per_user
from hookrec.eval.protocol import target_filter

Truth = dict[int, np.ndarray]


def _target_items_by_user(events: pl.LazyFrame, organic: bool, target: str) -> Truth:
    df = (
        events.filter(
            target_filter(target) & (pl.col("is_organic").cast(pl.Boolean) == organic)
        )
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


def train_seen_items(seqs_train: PackedSequences, history: PeriodAggregates) -> np.ndarray:
    """bool [N+1]: item has >= 1 event of any kind in the training period (index 0 = False).

    The items of the all-event train sequences (`ProcessedData.seqs("train")`, every event
    type) plus every item with a train positive or Listen+ event (`history` = the full-period
    train aggregates). Exact when the bundle stores full histories (`max_len_store: null`, as
    at 50m); with a cap (500m: 1024) an item whose only train events were truncated away from
    every user's history counts as unseen, so `new_items` can only grow, never miss an item.
    """
    seen = (np.asarray(history.pop_pos) > 0) | (np.asarray(history.pop_lp) > 0)
    items = np.asarray(seqs_train.items)
    seen[items[(items > 0) & (items < len(seen))]] = True
    seen[0] = False
    return seen


def _split_items(truth: Truth, keep_fn) -> Truth:
    out: Truth = {}
    for u, items in truth.items():
        keep = items[keep_fn(items)]
        if len(keep):
            out[u] = keep
    return out


def slice_truth(
    truth: Truth,
    test_events_lf: pl.LazyFrame,
    vocab: ItemVocab,
    features: ItemFeatures,
    history: PeriodAggregates,
    target: str = "listen_plus",
    *,
    seqs_train: PackedSequences,
) -> dict[str, Truth]:
    """Named sub-truths; users whose slice is empty are dropped from that slice.

    organic / served: truth items with >= 1 organic / non-organic test event of the truth's
    `target` kind (`eval.target`; an item may be in both).
    idless_items: OOV (idx > N) or no ID under the *data* vocab (< `data.min_count` train
    positives); warm_items: the rest (a data-vocab ID). Both are fixed by the bundle's vocab,
    so they are the same truth items for every model, whatever its own ID table
    (`model.id_min_count`). Called `cold_items` in eval records before F5 (same definition).
    new_items: OOV or no event of any kind in the training period (`train_seen_items` over
    `seqs_train` and `history`), e.g. items first seen on the test day: never seen by any
    model, a subset of idless_items.
    users_low/mid/high: terciles of the users' Listen+ event count over the full training
    period (`history`, untruncated; ties by uid).
    `features` is part of the interface but unused.
    """
    n = vocab.n_items
    out: dict[str, Truth] = {"all": {u: t for u, t in truth.items() if len(t)}}
    out["organic"] = _restrict(truth, _target_items_by_user(test_events_lf, True, target))
    out["served"] = _restrict(truth, _target_items_by_user(test_events_lf, False, target))

    def is_idless(items: np.ndarray) -> np.ndarray:
        return (items > n) | (vocab.id_idx[np.minimum(items, n)] == 0)

    seen = train_seen_items(seqs_train, history)

    def is_new(items: np.ndarray) -> np.ndarray:
        return (items > n) | ~seen[np.minimum(items, n)]

    out["idless_items"] = _split_items(truth, is_idless)
    out["new_items"] = _split_items(truth, is_new)
    out["warm_items"] = _split_items(truth, lambda items: ~is_idless(items))

    uids = np.array(sorted(truth), dtype=np.int64)
    counts = history.user_counts(uids)
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
                m[f"recall_paper@{k}"] = float(recall_full_per_user(rows, tl, k).mean())
                m[f"coverage@{k}"] = coverage(rows, n_items, k)
        out[name] = m
    return out
