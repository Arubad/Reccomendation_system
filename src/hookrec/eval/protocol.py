"""Evaluation protocol: ground truth from labeled test events, and a batched evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import polars as pl

from hookrec.data.label import LIKE, LISTEN_FULL
from hookrec.data.vocab import ItemVocab
from hookrec.eval.metrics import coverage, ndcg_per_user, recall_per_user


class Recommender(Protocol):
    def recommend(self, uids: np.ndarray, k: int) -> np.ndarray:
        """Item indices, shape (len(uids), k). 0-padding is allowed and never a hit."""
        ...


@dataclass
class EvalResult:
    metrics: dict[str, float]
    per_user: dict[str, np.ndarray]  # "ndcg@k" / "recall@k", aligned with `uids`
    uids: np.ndarray
    topk: np.ndarray  # (len(uids), max(ks))


TARGETS = ("listen_plus", "positive")


def target_filter(target: str = "listen_plus") -> pl.Expr:
    """Which labeled events are ground truth (`eval.target`).

    "listen_plus" (default, the Yambda paper's Listen+): token == LISTEN_FULL, i.e. listens with
    played_ratio_pct >= full_threshold; likes are not targets. "positive": LISTEN_FULL or LIKE.
    """
    if target == "listen_plus":
        return pl.col("token") == LISTEN_FULL
    if target == "positive":
        return pl.col("token").is_in([LISTEN_FULL, LIKE])
    raise ValueError(f"eval.target must be one of {TARGETS}, got {target!r}")


def build_ground_truth(
    test_lf_labeled: pl.LazyFrame, vocab: ItemVocab, target: str = "listen_plus"
) -> dict[int, np.ndarray]:
    """uid -> unique target item indices (see `target_filter`). Users without targets are absent.

    Raw ids outside the vocabulary get reserved indices N+1, N+2, ... (ascending raw id), so they
    can never be recommended and always count as misses (ruling R3).
    """
    pos = (
        test_lf_labeled.filter(target_filter(target))
        .select("uid", "item_id")
        .unique()
        .sort("uid", "item_id")
        .collect()
    )
    if pos.height == 0:
        return {}
    uids = pos["uid"].to_numpy()
    raw = pos["item_id"].to_numpy()
    idx = vocab.encode(raw)
    oov = idx == 0
    if oov.any():
        distinct = np.unique(raw[oov])
        idx[oov] = vocab.n_items + 1 + np.searchsorted(distinct, raw[oov])
    cuts = np.flatnonzero(np.diff(uids)) + 1
    groups = np.split(idx, cuts)
    return {int(u): g for u, g in zip(uids[np.concatenate([[0], cuts])], groups)}


def split_by_history(
    truth: dict[int, np.ndarray], history_uids: np.ndarray
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Split truth into (warm, cold): warm users appear in `history_uids`.

    Preprocessing passes the users with >= 1 Listen+ event in the training period, as in the
    official protocol (its train set is the Listen+-filtered interactions).
    """
    hist = set(np.unique(np.asarray(history_uids)).tolist())
    warm: dict[int, np.ndarray] = {}
    cold: dict[int, np.ndarray] = {}
    for u, t in truth.items():
        (warm if u in hist else cold)[u] = t
    return warm, cold


def count_oov(truth: dict[int, np.ndarray], n_items: int) -> int:
    """Number of ground-truth entries that are out-of-vocabulary placeholders (> N)."""
    return int(sum(int((t > n_items).sum()) for t in truth.values()))


def evaluate(
    rec: Recommender,
    truth: dict[int, np.ndarray],
    n_items: int,
    ks: tuple[int, ...] = (10, 100),
    batch_users: int = 1024,
) -> EvalResult:
    """Evaluate `rec` on users with non-empty truth, in ascending uid order."""
    uids = np.array(sorted(u for u, t in truth.items() if len(t) > 0), dtype=np.int64)
    k_max = max(ks)
    topk = np.zeros((len(uids), k_max), dtype=np.int64)
    for start in range(0, len(uids), batch_users):
        batch = uids[start : start + batch_users]
        out = np.asarray(rec.recommend(batch, k_max))
        if out.shape != (len(batch), k_max):
            raise ValueError(f"recommend returned {out.shape}, expected {(len(batch), k_max)}")
        topk[start : start + len(batch)] = out

    truth_list = [truth[int(u)] for u in uids]
    per_user: dict[str, np.ndarray] = {}
    metrics: dict[str, float] = {}
    for k in ks:
        per_user[f"ndcg@{k}"] = ndcg_per_user(topk, truth_list, k)
        per_user[f"recall@{k}"] = recall_per_user(topk, truth_list, k)
        metrics[f"ndcg@{k}"] = float(per_user[f"ndcg@{k}"].mean()) if len(uids) else 0.0
        metrics[f"recall@{k}"] = float(per_user[f"recall@{k}"].mean()) if len(uids) else 0.0
        metrics[f"coverage@{k}"] = coverage(topk, n_items, k)
    return EvalResult(metrics=metrics, per_user=per_user, uids=uids, topk=topk)
