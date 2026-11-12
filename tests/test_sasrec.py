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


# ---- R30 / I2: model.arch = official (yambda benchmarks/models/sasrec/model.py) -------------


class _OfficialSASRec(torch.nn.Module):
    """The encoder path of the official SASRecEncoder (HF yandex/yambda,
    benchmarks/models/sasrec/model.py:49-153), trimmed to full-length rows (no padding), which
    makes its packing / create_masked_tensor steps the identity. Test reference only."""

    def __init__(self, num_items, max_len, d, heads, layers, eps=1e-9):
        super().__init__()
        self._item_embeddings = torch.nn.Embedding(num_items + 1, d)  # :68-71
        self._position_embeddings = torch.nn.Embedding(max_len, d)  # :72
        self._layernorm = torch.nn.LayerNorm(d, eps=eps)  # :74
        self._dropout = torch.nn.Dropout(0.0)  # :75
        layer = torch.nn.TransformerEncoderLayer(  # :77-85
            d_model=d, nhead=heads, dim_feedforward=4 * d, dropout=0.0,
            activation=torch.nn.GELU(), layer_norm_eps=eps, batch_first=True,
        )  # fmt: skip
        self._encoder = torch.nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    def forward(self, events):  # [B, L] item ids, every row full length
        b, length = events.shape
        x = self._item_embeddings(events)
        positions = torch.arange(length - 1, -1, -1)[None].tile([b, 1])  # :134-136, newest = 0
        x = x + self._position_embeddings(positions)  # :145
        x = self._dropout(self._layernorm(x))  # :146-147
        causal = torch.tril(torch.ones(length, length)).bool()  # :150
        return self._encoder(src=x, mask=~causal)  # :151-153 (no padding to mask)


def test_official_arch_matches_the_yambda_encoder():
    d, heads, layers, max_len = D, 2, 2, 32
    torch.manual_seed(0)
    ref = _OfficialSASRec(N, max_len, d, heads, layers).eval()
    ours = make(arch="official", dropout=0.0, max_len=max_len)
    sd = {
        "item_emb.weight": ref._item_embeddings.weight,
        # ours anchors the newest event at max_len - 1, theirs at 0: the same table, reversed
        "pos_emb.weight": ref._position_embeddings.weight.flip(0),
        "input_norm.weight": ref._layernorm.weight,
        "input_norm.bias": ref._layernorm.bias,
    }
    sd.update({f"encoder.{k}": v for k, v in ref._encoder.state_dict().items()})
    missing, unexpected = ours.load_state_dict(sd, strict=False)
    assert not missing and not unexpected
    events = torch.randint(1, N + 1, (3, max_len), generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        want = ref(events)
        got = ours.encode(batch(events))
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


def test_official_arch_structure_and_init():
    m = make(arch="official")
    for layer in m.encoder.layers:
        assert layer.norm_first is False and layer.norm1.eps == 1e-9 and layer.norm2.eps == 1e-9
    assert m.input_norm.eps == 1e-9 and isinstance(m.final_norm, torch.nn.Identity)
    for name, p in m.named_parameters():  # model.py:176-188
        if "weight" in name and "norm" in name:
            assert torch.equal(p, torch.ones_like(p)), name
        elif "weight" in name:
            body = p[1:] if name == "item_emb.weight" else p  # row 0 = pad stays zero
            assert body.abs().max() <= 0.04 and body.std() > 0.01, name  # trunc_normal(.02, +-.04)
        else:
            assert torch.equal(p, torch.zeros_like(p)), name
    assert torch.count_nonzero(m.item_emb.weight[0]) == 0
    # no sqrt(d) scaling: the encoder input is item + position embeddings
    b = rand_batch()
    want = m.item_emb(b["items"]) + m.pos_emb(m._positions(L, b["items"].device))
    torch.testing.assert_close(m.input_repr(b), want)
    with torch.no_grad():
        assert torch.isfinite(m.user_state(b)).all()
        assert torch.isfinite(m.encode(rand_batch(n_real=(12, 0, 4)))).all()


def test_default_arch_unchanged_and_unknown_arch_raises():
    a, b = make(), make(arch="hookseq")
    assert a.arch == "hookseq"
    for (k, x), y in zip(a.state_dict().items(), b.state_dict().values(), strict=True):
        assert torch.equal(x, y), k
    assert all(layer.norm_first for layer in a.encoder.layers)
    assert not hasattr(a, "input_norm")
    with pytest.raises(ValueError, match="arch"):
        make(arch="paper")
