"""Losses: the chunked sampled softmax with logQ correction, the official SASRec BCE loss, the
log-q builder and a uniform negative sampler."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.autograd.function import once_differentiable

_MASK = -1e4  # large negative, finite in fp16 too; exp() of it underflows to 0
_Q_FLOOR = 1e-12
DEFAULT_CHUNK_ROWS = 8192  # rows of the [M, S] negative block materialised at a time


def _matmul_dtype(u: Tensor) -> torch.dtype:
    """dtype of the [M, S] matmuls: fp16 (tensor cores) under CUDA fp16 autocast, else fp32."""
    return torch.float16 if u.is_cuda and torch.is_autocast_enabled("cuda") else torch.float32


class _ChunkedSoftmaxCE(torch.autograd.Function):
    """Mean cross-entropy over valid rows of [pos_logit, (u v^T)/T + neg_bias, hard_logit].

    Never materialises the [M, S+1] logits: rows go in chunks of `chunk_rows`. The loss is a
    scalar, so dL/dlse_i = valid_i / n (n = max(#valid, 1)) is known in the forward pass,
    and each chunk computes its gradients right away from its own logits: one score matmul,
    then e = exp(s - rowmax) in place, then dU = w * (e @ V) and dV += e^T @ (w * U), with the
    row weight w_i = dL/dlse_i * exp(lse_neg_i - lse_i) / (T * sum_j e_ij). Backward only scales
    the saved gradients by the incoming scalar: no recompute, memory O(chunk_rows x S + (M+S) d).

    The [C, S] block is in `mm_dtype` (fp16 on CUDA, so every pass over it moves 2 bytes per
    element); row reductions accumulate in fp32 (u's dtype). Accidental hits are masked with a
    per-chunk broadcast compare; padded negatives come pre-masked in `neg_bias`. A row whose
    negatives are all masked gets lse_neg = -1e4 + log S, so weight exp(lse_neg - lse) = 0 and
    exactly zero gradient, as `masked_fill` gives the reference. For dV the rows w * U are
    divided by max|w| before the fp16 cast (and multiplied back after), so the operands stay
    in fp16's normal range whatever the row count.
    """

    @staticmethod
    def forward(ctx, u, v, pos_logit, hard_logit, neg_bias, pos_items, neg_items, temperature,
                chunk_rows, mm_dtype):  # fmt: skip
        dt = u.dtype
        need_u, need_v, need_pos, need_hard = ctx.needs_input_grad[:4]
        n_neg = len(neg_items)
        valid = (pos_items != 0).to(dt)
        row_w = valid / valid.sum().clamp(min=1)  # dL/dlse_i
        uh = (u / temperature).to(mm_dtype)
        vh = v.to(mm_dtype)
        bias = neg_bias.to(mm_dtype)
        gu = torch.zeros_like(u) if need_u else None
        gv = torch.zeros_like(v) if need_v else None
        gpos = torch.zeros_like(pos_logit) if need_pos else None
        ghard = torch.zeros_like(hard_logit) if need_hard else None
        tiny = torch.finfo(dt).tiny
        loss = u.new_zeros(())
        for r in range(0, len(u), chunk_rows):
            sl = slice(r, r + chunk_rows)
            pos_c = pos_logit[sl]
            cols = [pos_c[:, None]]
            if n_neg:
                e = torch.addmm(bias, uh[sl], vh.T)  # [C, S] = s/T - log q, pads masked
                e.masked_fill_(neg_items[None, :] == pos_items[sl, None], _MASK)  # hits
                m = e.amax(1, keepdim=True)
                e.sub_(m).exp_()  # in (0, 1]; the row max is exactly 1
                z = e.sum(1, dtype=dt)  # >= 1
                lse_neg = m[:, 0].to(dt) + z.log()
                cols.append(lse_neg[:, None])
            if hard_logit is not None:
                cols.append(hard_logit[sl])
            lse = torch.logsumexp(torch.cat(cols, 1), 1)  # [C, 2+H] is small
            a = row_w[sl]
            loss += (a * (lse - pos_c)).sum()
            if need_pos:
                gpos[sl] = a * (torch.exp(pos_c - lse) - 1)
            if need_hard:
                ghard[sl] = a[:, None] * torch.exp(hard_logit[sl] - lse[:, None])
            if n_neg and (need_u or need_v):
                w = a * torch.exp(lse_neg - lse) / (temperature * z)
                if need_u:
                    gu[sl] = (e @ vh).to(dt).mul_(w[:, None])
                if need_v:
                    c = w.abs().amax().clamp_min(tiny)
                    gv += (e.T @ (u[sl] * (w / c)[:, None]).to(mm_dtype)).to(dt).mul_(c)
            if n_neg:
                del e
        ctx.save_for_backward(gu, gv, gpos, ghard)
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, g):
        grads = [x * g if x is not None else None for x in ctx.saved_tensors]
        return (*grads, None, None, None, None, None, None)


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
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> Tensor:
    """Cross-entropy over [positive, shared sampled negatives, per-row hard negatives].

    user_vecs [M,d]; pos_items [M] (0 = ignored row); pos_vecs [M,d]; neg_items [S] and
    neg_vecs [S,d] are one negative set shared by all rows; hard_neg_vecs [M,H,d] with ids
    hard_neg_items [M,H] (0 = padding). Logits are s/T; with `log_q` [N+1], log q[item] is then
    subtracted (s/T - log q) from the positive and sampled-negative logits (hard negatives are
    not corrected). Negatives equal to the row's positive (accidental hits), padded sampled
    negatives (id 0) and padded hard negatives are masked. Returns the mean over valid rows,
    or 0 when none are valid (finite zero gradients). No host sync.

    Computed by `_ChunkedSoftmaxCE` in `chunk_rows`-row chunks, without the [M, S+1] logits.
    Off CUDA autocast everything is float32 and equals the pre-F3 implementation (kept as
    `_reference_sampled_softmax_loss` in tests/test_losses.py) to fp32 rounding. Under CUDA fp16
    autocast the [M, S] block is fp16 (tensor-core matmuls, fp32 row sums): the one intended
    numeric difference, ~2e-4 relative on the loss and ~2e-3 (Frobenius) on the gradients.
    """
    if chunk_rows < 1:
        raise ValueError(f"chunk_rows must be >= 1, got {chunk_rows}")
    mm_dtype = _matmul_dtype(user_vecs)
    with torch.autocast(user_vecs.device.type, enabled=False):
        u = user_vecs.float()
        pos_logit = (u * pos_vecs.float()).sum(-1) / temperature
        lq = log_q.float() if log_q is not None else None
        if lq is not None:
            pos_logit = pos_logit - lq[pos_items]
        neg_bias = -lq[neg_items] if lq is not None else u.new_zeros(len(neg_items))
        neg_bias = neg_bias.masked_fill(neg_items == 0, _MASK)
        hard_logit = None
        if hard_neg_vecs is not None:
            hard_logit = (u[:, None, :] * hard_neg_vecs.float()).sum(-1) / temperature
            if hard_neg_items is not None:
                bad = (hard_neg_items == pos_items[:, None]) | (hard_neg_items == 0)
                hard_logit = hard_logit.masked_fill(bad, _MASK)
        return _ChunkedSoftmaxCE.apply(u, neg_vecs.float(), pos_logit, hard_logit, neg_bias,
                                       pos_items, neg_items, temperature, chunk_rows,
                                       mm_dtype)  # fmt: skip


def bce_loss(user_vecs: Tensor, pos_vecs: Tensor, neg_vecs: Tensor) -> Tensor:
    """The official Yambda SASRec loss: binary cross-entropy, positive score -> 1 and one
    negative score -> 0 per position, averaged over all 2M scores.

    Mirrors `benchmarks/models/sasrec/model.py` (HF `yandex/yambda`), lines 244-265: queries
    `embeddings[mask]` (every non-padded position), positive and negative scores as row dot
    products (`einsum('bd,bd->b')`), then `binary_cross_entropy_with_logits(cat([pos, neg]),
    cat([ones, zeros]))` with the default mean. There is no temperature, no logQ and no
    accidental-hit masking: a negative equal to the positive is kept, as there. The negatives
    (one per position, uniform over 1..N) are drawn by the caller, as `data.py` line 118 does.
    user_vecs, pos_vecs, neg_vecs: [M,d] (valid rows only). Float32; 0 when M = 0.
    """
    with torch.autocast(user_vecs.device.type, enabled=False):
        u = user_vecs.float()
        if len(u) == 0:  # BCE's mean of nothing would be NaN
            return 0.0 * u.sum()
        pos = (u * pos_vecs.float()).sum(-1)
        neg = (u * neg_vecs.float()).sum(-1)
        target = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
        return F.binary_cross_entropy_with_logits(torch.cat([pos, neg]), target)


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
