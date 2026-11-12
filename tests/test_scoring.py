import numpy as np
import pytest
import torch

from hookrec.models.popularity import MostPop
from hookrec.models.sasrec import SASRec
from hookrec.train.datasets import SeqInferDataset
from hookrec.train.scoring import SeqRecommender, score_topk


def _bruteforce(u, v, k):
    s = (u.double() @ v.double().T).numpy()
    s[:, 0] = -np.inf
    idx = np.arange(s.shape[1])
    order = np.stack([np.lexsort((idx, -row))[:k] for row in s])
    return np.take_along_axis(s, order, 1), order


@pytest.mark.parametrize("chunk", [7, 17, 1000])
def test_score_topk_matches_bruteforce(chunk):
    g = torch.Generator().manual_seed(0)
    # small integer vectors -> many exact ties, which must break towards the lower index
    u = torch.randint(-2, 3, (9, 4), generator=g).float()
    v = torch.randint(-2, 3, (103, 4), generator=g).float()
    v[60] = v[20]
    vals, idx = score_topk(u, v, k=10, item_chunk=chunk)
    ref_vals, ref_idx = _bruteforce(u, v, 10)
    assert idx.dtype == torch.long and idx.shape == (9, 10)
    np.testing.assert_array_equal(idx.numpy(), ref_idx)
    np.testing.assert_allclose(vals.numpy(), ref_vals, rtol=1e-6)


def test_score_topk_many_tie_rows_are_sorted_in_sub_batches():
    from hookrec.train import scoring

    g = torch.Generator().manual_seed(3)
    u = torch.randint(-1, 2, (200, 3), generator=g).float()  # >64 rows with boundary ties
    v = torch.randint(-1, 2, (150, 3), generator=g).float()
    sc = u @ v.T
    sc[:, 0] = float("-inf")
    vals = sc.topk(10, dim=1).values
    thr = vals[:, -1:]
    assert ((sc == thr).sum(1) > (vals == thr).sum(1)).sum() > scoring._TIE_ROWS
    _, idx = score_topk(u, v, k=10, item_chunk=60)
    np.testing.assert_array_equal(idx.numpy(), _bruteforce(u, v, 10)[1])


def test_score_topk_random_floats_and_k_larger_than_chunk():
    g = torch.Generator().manual_seed(1)
    u, v = torch.randn(5, 8, generator=g), torch.randn(301, 8, generator=g)
    vals, idx = score_topk(u, v, k=50, item_chunk=13)
    ref_vals, ref_idx = _bruteforce(u, v, 50)
    np.testing.assert_array_equal(idx.numpy(), ref_idx)
    np.testing.assert_allclose(vals.numpy(), ref_vals, rtol=1e-5, atol=1e-6)


def test_score_topk_never_returns_pad():
    g = torch.Generator().manual_seed(2)
    u = torch.randn(6, 8, generator=g)
    v = torch.randn(40, 8, generator=g)
    v[0] = 1e3 * u.mean(0)  # pad would win every ranking if it were scored
    _, idx = score_topk(u, v, k=39, item_chunk=5)
    assert (idx != 0).all()
    assert all(len(set(r)) == 39 for r in idx.tolist())


def _tiny_model(n_items):
    torch.manual_seed(0)
    return SASRec(n_items, d_model=16, n_layers=1, n_heads=2, max_len=20, dropout=0.0)


def test_seqrecommender_fallback_for_unknown_users(processed_synth):
    seqs = processed_synth.seqs_val_train
    n = processed_synth.vocab.n_items
    model = _tiny_model(n)
    pop = MostPop().fit(seqs, n)
    rec = SeqRecommender(model, seqs, 20, torch.device("cpu"), pop, batch_size=4)
    known = seqs.uids[:10].astype(np.int64)
    unknown = int(seqs.uids.max()) + 999
    uids = np.concatenate([known[:5], [unknown], known[5:]])
    out = rec.recommend(uids, 10)
    assert out.shape == (11, 10) and out.dtype == np.int64
    np.testing.assert_array_equal(out[5], pop.top_items(10))
    assert out.min() >= 1 and out.max() <= n

    # known users: same result as the last-window dataset + user_state + brute force
    ds = SeqInferDataset(seqs, 20)
    rows = [ds[i] for i in range(10)]
    batch = {k: torch.stack([r[k] for r in rows]) for k in ("items", "tokens", "organic", "ratio")}
    model.eval()
    with torch.no_grad():
        ref = _bruteforce(model.user_state(batch), model.all_item_repr(), 10)[1]
    np.testing.assert_array_equal(np.delete(out, 5, axis=0), ref)
