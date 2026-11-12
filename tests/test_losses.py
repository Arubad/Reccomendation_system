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
