"""Sampled-softmax loss with logQ correction, its log-q builder and a uniform negative sampler."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

_MASK = -1e4  # large negative; logits are float32 so this is finite and exp() underflows to 0
_Q_FLOOR = 1e-12


def sampled_softmax_loss(
    user_vecs: Tensor,
    pos_items: Tensor,
    pos_vecs: Tensor,
    neg_items: Tensor,
    neg_vecs: Tensor,
    log_q: Tensor | None,
    temperature: float = 1.0,
    hard_neg_vecs: Tensor | None = None,
    hard_neg_items: Tensor | None = None,
) -> Tensor:
    """Cross-entropy over [positive, shared sampled negatives, per-row hard negatives].

    user_vecs [M,d]; pos_items [M] (0 = ignored row); pos_vecs [M,d]; neg_items [S] and
    neg_vecs [S,d] are one negative set shared by all rows; hard_neg_vecs [M,H,d] with ids
    hard_neg_items [M,H] (0 = padding). Logits are s/T; with `log_q` [N+1], log q[item] is then
    subtracted (s/T - log q) from the positive and sampled-negative logits (hard negatives are
    not corrected). Negatives equal to the row's positive (accidental hits), padded sampled
    negatives (id 0) and padded hard negatives are masked. Computed in
    float32. Returns the mean over valid rows, or 0.0 * user_vecs.sum() when none are valid.
    """
    u = user_vecs.float()
    pos_logit = (u * pos_vecs.float()).sum(-1, keepdim=True) / temperature
    neg_logit = (u @ neg_vecs.float().T) / temperature
    if log_q is not None:
        lq = log_q.float()
        pos_logit = pos_logit - lq[pos_items][:, None]
        neg_logit = neg_logit - lq[neg_items][None, :]
    neg_bad = (neg_items[None, :] == pos_items[:, None]) | (neg_items[None, :] == 0)
    neg_logit = neg_logit.masked_fill(neg_bad, _MASK)
    cols = [pos_logit, neg_logit]
    if hard_neg_vecs is not None:
        hard_logit = (u[:, None, :] * hard_neg_vecs.float()).sum(-1) / temperature
        if hard_neg_items is not None:
            bad = (hard_neg_items == pos_items[:, None]) | (hard_neg_items == 0)
            hard_logit = hard_logit.masked_fill(bad, _MASK)
        cols.append(hard_logit)
    logits = torch.cat(cols, dim=1)

    valid = pos_items != 0
    n_valid = valid.sum()
    if n_valid == 0:
        return 0.0 * user_vecs.float().sum()
    target = torch.zeros(len(logits), dtype=torch.long, device=logits.device)
    per_row = F.cross_entropy(logits, target, reduction="none")
    return (per_row * valid).sum() / n_valid


def make_log_q(
    pop_train: np.ndarray, mix_uniform: float, n_in_batch: int, n_uniform: int
) -> Tensor:
    """log q[i] for the mixed negative sampler, shape [N+1] (pop_train has N+1 entries).

    Per step, item i is expected to appear among the negatives
        n_in_batch * p_pop[i] + n_uniform / N
    times: in-batch negatives are positives drawn ~ popularity, uniform ones are flat over 1..N.
    p_pop is smoothed, (1 - mix_uniform) * pop/pop.sum() + mix_uniform / N, so never-positive
    items are not -inf. The expected count is proportional to the sampling probability, and
    constant offsets cancel in the softmax, so it is a valid q for logQ correction.
    Floored at 1e-12; index 0 (pad) is set to the floor.
    """
    pop = np.asarray(pop_train, dtype=np.float64)[1:]
    n = len(pop)
    if n == 0:
        raise ValueError("pop_train must contain at least one item (length N+1 with N >= 1)")
    total = pop.sum()
    base = pop / total if total > 0 else np.full(n, 1.0 / n)
    p_pop = (1.0 - mix_uniform) * base + mix_uniform / n
    expected = n_in_batch * p_pop + n_uniform / n
    out = np.empty(n + 1, dtype=np.float64)
    out[0] = _Q_FLOOR
    out[1:] = np.maximum(expected, _Q_FLOOR)
    return torch.from_numpy(np.log(out)).float()


def sample_negatives(n_items: int, n: int, generator: torch.Generator | None) -> Tensor:
    """n item ids drawn uniformly from 1..n_items."""
    return torch.randint(1, n_items + 1, (n,), generator=generator)
