"""Per-user ranking metrics, matching the official Yambda benchmark.

Source: huggingface.co/datasets/yandex/yambda, benchmarks/yambda/evaluation/metrics.py
(there is no github.com/yandex/yambda; the benchmark code ships inside the HF dataset repo).
Official definitions, with `target_mask[u, j] = 1` iff the item ranked j for user u is a target:

    Recall@k   = mean_u( sum(target_mask[u, :k]) / clamp(num_positives_u, max=k) )
               # users with 0 positives contribute 0 (denominator set to inf)
    NDCG@k     = mean_u( DCG_u@k / IDCG_u@k )              # per-user ratio, then mean
      DCG_u@k  = sum_{j<k} target_mask[u, j] / log2(j + 2)
      IDCG_u@k = DCG of an ideal mask [1]*|T_u| + [0]*..., i.e. sum_{j<min(|T_u|, k)} 1 / log2(j + 2)
               # IDCG == 0 -> ratio 0
    Coverage@k = |unique(ranked.item_ids[:, :k])| / num_item_ids

Recall denominator, and the paper table: the official code divides by min(|T|, k), but the repo
has a later commit "Fix yet another bug in metrics (#24)" and a leftover comment suggesting the
denominator used to be |T|. The paper's Recall column is much closer to hits@k / |T|, though not
equal to it (our NDCG agrees with the paper within 7%; min(|T|, k) recall is 2.0-3.1x the paper's
at k=10, hits / |T| 12-38% above it). So `recall_per_user` is the official definition (`recall@k`) and `recall_full_per_user` is the paper-comparable one (`recall_paper@k`);
both are always reported.

Binary gains, IDCG over min(|truth|, k): identical to the brief's assumptions. One deliberate
deviation: the official Coverage counts the padding id 0 if it appears in `ranked`; here 0 is
padding and never counted as an item.
"""

from __future__ import annotations

import numpy as np


def _check(ranked: np.ndarray, truth: list[np.ndarray], k: int) -> None:
    if ranked.ndim != 2 or ranked.shape[1] < k or k < 1:
        raise ValueError(f"need ranked of shape (U, >= {k}), got {ranked.shape}")
    if len(truth) != ranked.shape[0]:
        raise ValueError(f"{len(truth)} truth entries for {ranked.shape[0]} ranked rows")


def hit_matrix(ranked: np.ndarray, truth: list[np.ndarray], k: int) -> np.ndarray:
    """(U, k) bool: ranked[u, j] is in truth[u]. Item 0 (padding) is never a hit.

    Membership is one `np.isin` over composite (row, item) keys, so there is no per-user loop.
    """
    _check(ranked, truth, k)
    top = np.asarray(ranked[:, :k], dtype=np.int64)
    lens = np.fromiter((len(t) for t in truth), dtype=np.int64, count=len(truth))
    if lens.sum() == 0:
        return np.zeros(top.shape, dtype=bool)
    flat = np.concatenate([np.asarray(t, dtype=np.int64) for t in truth])
    base = max(int(top.max()), int(flat.max())) + 1
    t_keys = np.repeat(np.arange(len(truth), dtype=np.int64), lens) * base + flat
    r_keys = np.arange(len(truth), dtype=np.int64)[:, None] * base + top
    return np.isin(r_keys, t_keys) & (top != 0)


def _n_truth(truth: list[np.ndarray]) -> np.ndarray:
    return np.fromiter((len(t) for t in truth), dtype=np.int64, count=len(truth))


def ndcg_per_user(ranked: np.ndarray, truth: list[np.ndarray], k: int) -> np.ndarray:
    hits = hit_matrix(ranked, truth, k)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = hits @ discounts
    ideal_n = np.minimum(_n_truth(truth), k)
    idcg = np.concatenate([[0.0], np.cumsum(discounts)])[ideal_n]
    return np.divide(dcg, idcg, out=np.zeros_like(dcg), where=idcg > 0)


def recall_per_user(ranked: np.ndarray, truth: list[np.ndarray], k: int) -> np.ndarray:
    hits = hit_matrix(ranked, truth, k).sum(axis=1).astype(np.float64)
    denom = np.minimum(_n_truth(truth), k).astype(np.float64)
    return np.divide(hits, denom, out=np.zeros_like(hits), where=denom > 0)


def recall_full_per_user(ranked: np.ndarray, truth: list[np.ndarray], k: int) -> np.ndarray:
    """hits@k / |T| per user (0 for users with empty truth): the paper-table definition.

    Differs from the official `recall_per_user` (denominator min(|T|, k)) when |T| > k.
    """
    hits = hit_matrix(ranked, truth, k).sum(axis=1).astype(np.float64)
    denom = _n_truth(truth).astype(np.float64)
    return np.divide(hits, denom, out=np.zeros_like(hits), where=denom > 0)


def coverage(ranked: np.ndarray, n_items: int, k: int) -> float:
    if ranked.ndim != 2 or ranked.shape[1] < k or k < 1:
        raise ValueError(f"need ranked of shape (U, >= {k}), got {ranked.shape}")
    uniq = np.unique(ranked[:, :k])
    return float((uniq != 0).sum() / n_items)
