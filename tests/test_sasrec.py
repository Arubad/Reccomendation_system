import pytest
import torch

from hookrec.models.sasrec import SASRec

N, D, L = 50, 16, 12


def make(**kw):
    torch.manual_seed(0)
    args = {"n_items": N, "d_model": D, "n_layers": 2, "n_heads": 2, "max_len": 32, "dropout": 0.2}
    args.update(kw)
    return SASRec(**args).eval()


def batch(items):
    items = torch.as_tensor(items, dtype=torch.long)
    z = torch.zeros_like(items)
    return {"items": items, "tokens": z, "organic": z, "ratio": z}


def rand_batch(n_real=(12, 7, 3), length=L, seed=1):
    g = torch.Generator().manual_seed(seed)
    items = torch.zeros(len(n_real), length, dtype=torch.long)
    for i, k in enumerate(n_real):
        if k:
            items[i, length - k :] = torch.randint(1, N + 1, (k,), generator=g)
    return batch(items)


def test_forward_shapes():
    m = make()
    b = rand_batch()
    assert m.encode(b).shape == (3, L, D)
    assert m.user_state(b).shape == (3, D)
    assert m.all_item_repr().shape == (N + 1, D)


def test_causal_no_future_leak():
    m = make()
    b = rand_batch(n_real=(12, 12, 12))
    t = 7
    b2 = {k: v.clone() for k, v in b.items()}
    b2["items"][:, t] = (b2["items"][:, t] % N) + 1
    with torch.no_grad():
        o1, o2 = m.encode(b), m.encode(b2)
    assert torch.allclose(o1[:, :t], o2[:, :t], atol=1e-6)
    assert not torch.allclose(o1[:, t:], o2[:, t:], atol=1e-6)


def test_padding_invariance():
    m = make()
    short = rand_batch(n_real=(5, 3, 8), length=8)["items"]
    long = torch.cat([torch.zeros(3, 10, dtype=torch.long), short], dim=1)
    with torch.no_grad():
        a, b = m.user_state(batch(short)), m.user_state(batch(long))
    assert torch.allclose(a, b, atol=1e-5)


def test_all_pad_row_is_finite():
    m = make()
    b = rand_batch(n_real=(12, 0, 4))
    with torch.no_grad():
        out, st = m.encode(b), m.user_state(b)
    assert torch.isfinite(out).all() and torch.isfinite(st).all()


def test_user_state_is_last_real_position():
    m = make()
    b = rand_batch()
    with torch.no_grad():
        assert torch.equal(m.user_state(b), m.encode(b)[:, -1])


def test_all_item_repr_matches_direct():
    m = make()
    with torch.no_grad():
        full = m.all_item_repr(chunk=7)
        direct = m.item_repr(torch.arange(N + 1))
    assert torch.allclose(full, direct)
    assert not full.requires_grad
    assert torch.count_nonzero(full[0]) == 0


def test_init_and_pad_row():
    m = SASRec(n_items=N, d_model=D, max_len=32)
    assert torch.count_nonzero(m.item_emb.weight[0]) == 0
    assert 0.01 < m.item_emb.weight[1:].std() < 0.03
    for mod in m.modules():
        if isinstance(mod, torch.nn.Linear) and mod.bias is not None:
            assert torch.count_nonzero(mod.bias) == 0


def test_too_long_rejected():
    m = make(max_len=8)
    with pytest.raises(ValueError):
        m.encode(rand_batch(length=12))


def test_train_mode_backward():
    m = make().train()
    m.encode(rand_batch()).sum().backward()
    assert m.item_emb.weight.grad is not None
