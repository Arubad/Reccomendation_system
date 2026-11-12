import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from hookrec.models.losses import make_log_q, sample_negatives, sampled_softmax_loss


def _toy(m=4, s=6, d=8, n_items=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(m, d, generator=g)
    pos_items = torch.randint(1, 8, (m,), generator=g)
    neg_items = torch.randint(8, n_items + 1, (s,), generator=g)
    table = torch.randn(n_items + 1, d, generator=g)
    return u, pos_items, table[pos_items], neg_items, table[neg_items], table


def test_uniform_logq_equals_no_logq():
    u, p, pv, n, nv, _ = _toy()
    base = sampled_softmax_loss(u, p, pv, n, nv, None)
    const = sampled_softmax_loss(u, p, pv, n, nv, torch.full((21,), -3.0))
    assert torch.allclose(base, const, atol=1e-6)


def test_logq_reduces_popular_negative_penalty():
    u, p, pv, n, nv, _ = _toy()
    log_q = torch.zeros(21)
    base = sampled_softmax_loss(u, p, pv, n, nv, log_q)
    log_q[n[0]] = 3.0  # one negative is very popular
    corrected = sampled_softmax_loss(u, p, pv, n, nv, log_q)
    assert corrected < base  # its logit shrank, so it penalises the positive less


def test_accidental_hit_masked():
    u, p, pv, n, _nv, table = _toy()
    n2 = n.clone()
    n2[0] = p[0]
    nv2 = table[n2]
    with_hit = sampled_softmax_loss(u[:1], p[:1], pv[:1], n2, nv2, None)
    without = sampled_softmax_loss(u[:1], p[:1], pv[:1], n2[1:], nv2[1:], None)
    assert torch.allclose(with_hit, without, atol=1e-5)


def test_hard_negative_accidental_hit_and_padding_masked():
    u, p, pv, n, nv, table = _toy(m=1)
    hard_items = torch.stack([p[0], torch.tensor(0), torch.tensor(15)])[None]
    hard_vecs = table[hard_items]
    hard_vecs[0, 1] = 5.0 * torch.ones(8)  # padding vec must not matter
    for t in (1.0, 0.1):
        got = sampled_softmax_loss(u, p, pv, n, nv, None, temperature=t,
                                   hard_neg_vecs=hard_vecs, hard_neg_items=hard_items)
        want = sampled_softmax_loss(u, p, pv, n, nv, None, temperature=t,
                                    hard_neg_vecs=hard_vecs[:, 2:],
                                    hard_neg_items=hard_items[:, 2:])
        assert torch.allclose(got, want, atol=1e-5)


def test_all_ignored_rows_zero_loss_finite_grad():
    u, p, pv, n, nv, _ = _toy()
    u = u.clone().requires_grad_(True)
    loss = sampled_softmax_loss(u, torch.zeros_like(p), pv, n, nv, torch.zeros(21))
    assert loss.item() == 0.0
    loss.backward()
    assert torch.isfinite(u.grad).all()


def test_ignored_rows_do_not_affect_loss():
    u, p, pv, n, nv, _ = _toy()
    p2 = p.clone()
    p2[2:] = 0
    full = sampled_softmax_loss(u, p2, pv, n, nv, torch.zeros(21))
    first = sampled_softmax_loss(u[:2], p[:2], pv[:2], n, nv, torch.zeros(21))
    assert torch.allclose(full, first, atol=1e-6)


def test_matches_manual_cross_entropy():
    u, p, pv, n, nv, _ = _toy(n_items=40)
    t = 0.5
    logits = torch.cat([(u * pv).sum(-1, keepdim=True), u @ nv.T], 1) / t
    want = F.cross_entropy(logits, torch.zeros(len(u), dtype=torch.long))
    got = sampled_softmax_loss(u, p, pv, n, nv, None, temperature=t)
    assert torch.allclose(got, want, atol=1e-5)


def test_half_precision_inputs_finite():
    u, p, pv, n, nv, _ = _toy()
    loss = sampled_softmax_loss(u.half(), p, pv.half(), n, nv.half(), torch.zeros(21).half(),
                                temperature=0.05)
    assert loss.dtype == torch.float32 and torch.isfinite(loss)


def test_loss_decreases_on_toy_optimisation():
    torch.manual_seed(0)
    emb = torch.nn.Embedding(31, 8)
    proj = torch.nn.Linear(8, 8)
    opt = torch.optim.Adam(list(emb.parameters()) + list(proj.parameters()), lr=0.05)
    ctx = torch.randint(1, 31, (32,))
    pos = (ctx % 30) + 1
    negs = torch.arange(1, 31)
    losses = []
    for _ in range(50):
        opt.zero_grad()
        loss = sampled_softmax_loss(proj(emb(ctx)), pos, emb(pos), negs, emb(negs), None)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]


def test_make_log_q():
    pop = np.array([7, 10, 30, 60, 0])  # N=4; index 0 (pad) is ignored even if non-zero
    pop_train = pop[1:]
    lq = make_log_q(pop, mix_uniform=0.2, n_in_batch=100, n_uniform=40)
    assert lq.shape == (5,)
    assert lq[0].item() == pytest.approx(math.log(1e-12))
    p = 0.8 * pop_train / pop_train.sum() + 0.2 / 4
    expected = 100 * p + 40 / 4
    assert np.allclose(lq[1:].numpy(), np.log(expected), atol=1e-5)
    assert torch.isfinite(lq).all()


def test_sample_negatives_range_and_determinism():
    a = sample_negatives(50, 1000, torch.Generator().manual_seed(1))
    b = sample_negatives(50, 1000, torch.Generator().manual_seed(1))
    assert torch.equal(a, b) and a.min() >= 1 and a.max() <= 50 and a.dtype == torch.long


def test_logq_applied_after_temperature():
    u, p, pv, n, nv, _ = _toy(n_items=40)
    t = 0.1
    log_q = torch.randn(41)
    logits = torch.cat([(u * pv).sum(-1, keepdim=True) / t - log_q[p][:, None],
                        u @ nv.T / t - log_q[n][None, :]], 1)
    want = F.cross_entropy(logits, torch.zeros(len(u), dtype=torch.long))
    got = sampled_softmax_loss(u, p, pv, n, nv, log_q, temperature=t)
    assert torch.allclose(got, want, atol=1e-4)


def test_pad_negative_masked():
    u, p, pv, n, nv, _ = _toy()
    n2 = torch.cat([n, torch.tensor([0])])
    nv2 = torch.cat([nv, 9.0 * torch.ones(1, 8)])
    got = sampled_softmax_loss(u, p, pv, n2, nv2, None)
    assert torch.allclose(got, sampled_softmax_loss(u, p, pv, n, nv, None), atol=1e-5)


def test_make_log_q_zero_pop_uniform_fallback_and_empty_raises():
    lq = make_log_q(np.zeros(5, dtype=np.int64), 0.2, 100, 40)
    assert torch.isfinite(lq).all()
    assert torch.allclose(lq[1:], lq[1:2].expand(4))
    with pytest.raises(ValueError):
        make_log_q(np.zeros(1, dtype=np.int64), 0.2, 100, 40)


# ---- F3: chunked sampled softmax == the original implementation ---------------------------

_REF_MASK = -1e4


def _reference_sampled_softmax_loss(
    user_vecs, pos_items, pos_vecs, neg_items, neg_vecs, log_q, temperature=1.0,
    hard_neg_vecs=None, hard_neg_items=None,
):  # fmt: skip
    """The pre-F3 `sampled_softmax_loss`, verbatim: full [M, S+1] fp32 logits, then cross-entropy."""
    u = user_vecs.float()
    pos_logit = (u * pos_vecs.float()).sum(-1, keepdim=True) / temperature
    neg_logit = (u @ neg_vecs.float().T) / temperature
    if log_q is not None:
        lq = log_q.float()
        pos_logit = pos_logit - lq[pos_items][:, None]
        neg_logit = neg_logit - lq[neg_items][None, :]
    neg_bad = (neg_items[None, :] == pos_items[:, None]) | (neg_items[None, :] == 0)
    neg_logit = neg_logit.masked_fill(neg_bad, _REF_MASK)
    cols = [pos_logit, neg_logit]
    if hard_neg_vecs is not None:
        hard_logit = (u[:, None, :] * hard_neg_vecs.float()).sum(-1) / temperature
        if hard_neg_items is not None:
            bad = (hard_neg_items == pos_items[:, None]) | (hard_neg_items == 0)
            hard_logit = hard_logit.masked_fill(bad, _REF_MASK)
        cols.append(hard_logit)
    logits = torch.cat(cols, dim=1)
    valid = pos_items != 0
    n_valid = valid.sum()
    if n_valid == 0:
        return 0.0 * user_vecs.float().sum()
    target = torch.zeros(len(logits), dtype=torch.long, device=logits.device)
    per_row = F.cross_entropy(logits, target, reduction="none")
    return (per_row * valid).sum() / n_valid


def _masked_case(m, s, h, d=16, n_items=60, seed=0, scale=1.0):
    """Random inputs that exercise every mask: accidental hits, pad negatives, ignored rows,
    hard-negative hits and hard padding."""
    g = torch.Generator().manual_seed(seed)
    table = torch.randn(n_items + 1, d, generator=g) * scale
    u = torch.randn(m, d, generator=g) * scale
    pos = torch.randint(1, n_items + 1, (m,), generator=g)
    pos[torch.rand(m, generator=g) < 0.2] = 0  # ignored rows
    pos[0] = 1  # at least one valid row
    neg = torch.randint(1, n_items + 1, (s,), generator=g)
    if s:
        neg[torch.rand(s, generator=g) < 0.05] = int(pos[min(1, m - 1)]) or 1  # accidental hits
        neg[: max(1, s // 10)] = 0  # pad negatives
        neg[-1] = pos[0]  # an accidental hit
    hard = hard_vecs = None
    if h:
        hard = torch.randint(0, n_items + 1, (m, h), generator=g)  # 0 = padding
        hard[:, 0] = torch.where(torch.rand(m, generator=g) < 0.3, pos, hard[:, 0])  # hits
        hard_vecs = table[hard].clone()
        hard_vecs[hard == 0] = 7.0  # padded vectors must not matter
    log_q = torch.randn(n_items + 1, generator=g)
    return u, pos, table[pos].clone(), neg, table[neg].clone(), log_q, hard, hard_vecs


def _loss_and_grads(fn, case, log_q_on, t, **kw):
    u, pos, pv, neg, nv, log_q, hard, hv = case
    leaves = [x.clone().requires_grad_(True) for x in (u, pv, nv)]
    hv_leaf = hv.clone().requires_grad_(True) if hv is not None else None
    loss = fn(leaves[0], pos, leaves[1], neg, leaves[2], log_q if log_q_on else None, t,
              hard_neg_vecs=hv_leaf, hard_neg_items=hard, **kw)  # fmt: skip
    loss.backward()
    leaves += [hv_leaf] if hv_leaf is not None else []
    # an unused input (no negatives) has no grad: compare it as zeros
    return loss.detach(), [x.grad if x.grad is not None else torch.zeros_like(x) for x in leaves]


def _rel_err(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


@pytest.mark.parametrize("m,s,h,chunk", [
    (1, 5, 0, 8192), (7, 13, 0, 3), (33, 50, 4, 8), (64, 200, 2, 64), (50, 37, 3, 1),
    (20, 0, 2, 4),
])  # fmt: skip
@pytest.mark.parametrize("t", [1.0, 0.1])
@pytest.mark.parametrize("log_q_on", [False, True])
def test_chunked_loss_matches_reference_values_and_grads(m, s, h, chunk, t, log_q_on):
    case = _masked_case(m, s, h, seed=m * 1000 + s)
    want, want_g = _loss_and_grads(_reference_sampled_softmax_loss, case, log_q_on, t)
    got, got_g = _loss_and_grads(sampled_softmax_loss, case, log_q_on, t, chunk_rows=chunk)
    assert torch.isfinite(got)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)
    assert len(got_g) == len(want_g) == (4 if h else 3)
    for name, a, b in zip(("user", "pos", "neg", "hard"), got_g, want_g):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5, msg=lambda m, n=name: f"{n}: {m}")


def test_chunked_loss_reference_all_ignored_and_all_masked_rows():
    # every row ignored: 0 loss and zero (finite) user grads, as the reference
    case = list(_masked_case(9, 20, 2))
    case[1] = torch.zeros_like(case[1])
    for fn in (_reference_sampled_softmax_loss, sampled_softmax_loss):
        loss, grads = _loss_and_grads(fn, case, True, 0.5)
        assert loss.item() == 0.0
        assert torch.equal(grads[0], torch.zeros_like(grads[0]))
        assert all(torch.isfinite(gr).all() for gr in grads)
    # rows whose sampled negatives are all masked (hits / pads): only pos + hard remain
    u, _pos, _pv, _n, _nv, log_q, hard, hv = _masked_case(4, 6, 2)
    pos = torch.tensor([3, 5, 7, 9])
    neg = torch.tensor([3, 3, 0, 0, 3, 0])
    g = torch.Generator().manual_seed(1)
    pv, nv = torch.randn(4, 16, generator=g), torch.randn(6, 16, generator=g)
    case = (u, pos, pv, neg, nv, log_q, hard, hv)
    want, want_g = _loss_and_grads(_reference_sampled_softmax_loss, case, True, 1.0)
    got, got_g = _loss_and_grads(sampled_softmax_loss, case, True, 1.0, chunk_rows=3)
    torch.testing.assert_close(got, want)
    for a, b in zip(got_g, want_g, strict=True):
        torch.testing.assert_close(a, b)


def test_chunked_loss_large_logits_stay_finite_and_match():
    case = _masked_case(40, 300, 4, scale=3.0, seed=5)  # |s| ~ 150, / T=0.05 -> ~3000
    want, want_g = _loss_and_grads(_reference_sampled_softmax_loss, case, True, 0.05)
    got, got_g = _loss_and_grads(sampled_softmax_loss, case, True, 0.05, chunk_rows=16)
    assert torch.isfinite(got)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-3)
    # fp32 eps x |logit| ~ 1.2e-7 x 3000 = 3.6e-4 bounds any two fp32 orderings after exp()
    for a, b in zip(got_g, want_g, strict=True):
        assert _rel_err(a, b) < 5e-4


def test_chunked_softmax_function_gradcheck_float64():
    from hookrec.models.losses import _MASK, _ChunkedSoftmaxCE

    g = torch.Generator().manual_seed(0)

    def leaf(*shape):
        return torch.randn(*shape, dtype=torch.float64, generator=g, requires_grad=True)

    u, v, pos_logit, hard_logit = leaf(6, 5), leaf(9, 5), leaf(6), leaf(6, 2)
    pos = torch.tensor([1, 2, 3, 0, 5, 6])  # one ignored row
    neg = torch.tensor([1, 2, 0, 4, 5, 6, 7, 8, 9])  # hits for rows 0, 1, 4, 5; one pad
    bias = torch.randn(9, dtype=torch.float64, generator=g).masked_fill(neg == 0, _MASK)
    for hard in (None, hard_logit):
        def f(a, c, p, h=None):
            return _ChunkedSoftmaxCE.apply(a, c, p, h, bias, pos, neg, 0.7, 4, torch.float64)

        args = (u, v, pos_logit) if hard is None else (u, v, pos_logit, hard)
        assert torch.autograd.gradcheck(f, args)


def test_chunked_loss_fp16_matmul_path_is_close(monkeypatch):
    """On CUDA under fp16 autocast the [M, S] block is fp16 (row sums fp32). Emulated on CPU by
    forcing the fp16 path. Expected tolerance vs fp32 (documented in the F3 report): loss
    ~2e-4 relative, gradients ~2e-3 relative (Frobenius)."""
    import hookrec.models.losses as losses_mod

    case = _masked_case(64, 256, 3, seed=3)
    want, want_g = _loss_and_grads(sampled_softmax_loss, case, True, 1.0, chunk_rows=16)
    monkeypatch.setattr(losses_mod, "_matmul_dtype", lambda u: torch.float16)
    got, got_g = _loss_and_grads(sampled_softmax_loss, case, True, 1.0, chunk_rows=16)
    assert got.dtype == torch.float32 and _rel_err(got, want) < 1e-3
    for a, b in zip(got_g, want_g, strict=True):
        assert a.dtype == torch.float32 and _rel_err(a, b) < 5e-3


def test_chunked_loss_matmul_dtype_is_fp32_off_cuda():
    from hookrec.models.losses import _matmul_dtype

    u = torch.zeros(2, 3)
    assert _matmul_dtype(u) == torch.float32
    with torch.autocast("cpu", dtype=torch.bfloat16):  # CPU autocast never lowers the loss
        assert _matmul_dtype(u) == torch.float32
        case = _masked_case(5, 7, 0)
        loss, _ = _loss_and_grads(sampled_softmax_loss, case, False, 1.0)
    assert loss.dtype == torch.float32


# ---- F3: the official SASRec BCE loss ------------------------------------------------------


def test_bce_loss_is_the_official_formula():
    from hookrec.models.losses import bce_loss

    g = torch.Generator().manual_seed(0)
    u, p, n = (torch.randn(11, 8, generator=g) for _ in range(3))
    pos_s, neg_s = (u * p).sum(-1), (u * n).sum(-1)
    want = F.binary_cross_entropy_with_logits(
        torch.cat([pos_s, neg_s]), torch.cat([torch.ones(11), torch.zeros(11)])
    )
    torch.testing.assert_close(bce_loss(u, p, n), want)
    # the mean over 2M logits = the row mean of (softplus(-pos) + softplus(neg)) / 2
    torch.testing.assert_close(bce_loss(u, p, n), (F.softplus(-pos_s) + F.softplus(neg_s)).mean() / 2)
    assert bce_loss(u.half(), p.half(), n.half()).dtype == torch.float32


def test_bce_loss_no_rows_is_zero_with_finite_grad():
    from hookrec.models.losses import bce_loss

    u = torch.zeros(0, 8, requires_grad=True)
    loss = bce_loss(u, torch.zeros(0, 8), torch.zeros(0, 8))
    assert loss.item() == 0.0
    loss.backward()
    assert torch.isfinite(u.grad).all()
