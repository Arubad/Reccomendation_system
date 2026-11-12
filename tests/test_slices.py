import numpy as np
import polars as pl

from hookrec.eval.protocol import evaluate
from hookrec.eval.slices import evaluate_slices, slice_truth
from hookrec.models.popularity import MostPop


def _slices(d):
    return slice_truth(d.truth_test, d.test_events(), d.vocab, d.features, d.seqs_train)


def test_slices_partition_organic_served(processed_synth):
    d = processed_synth
    sl = _slices(d)
    assert sl["all"].keys() == d.truth_test.keys()
    for u, items in d.truth_test.items():
        got = np.union1d(sl["organic"].get(u, items[:0]), sl["served"].get(u, items[:0]))
        # OOV items never appear in test_events (they encode to 0): only in-vocab are covered
        inv = items[items <= d.vocab.n_items]
        assert set(inv.tolist()) <= set(got.tolist())
        assert set(got.tolist()) <= set(items.tolist())


def test_organic_served_come_from_labeled_events(processed_synth):
    d = processed_synth
    sl = _slices(d)
    ev = d.test_events().filter(pl.col("is_positive").cast(pl.Boolean)).collect()
    org = {
        (int(u), int(i))
        for u, i in ev.filter(pl.col("is_organic").cast(pl.Boolean)).select("uid", "item_idx").rows()
    }
    for u, items in sl["organic"].items():
        assert all((u, int(i)) in org for i in items)
    assert sum(len(v) for v in sl["organic"].values()) > 0
    assert sum(len(v) for v in sl["served"].values()) > 0


def test_cold_slice_only_idless_or_unknown(processed_synth):
    d = processed_synth
    n, idx = d.vocab.n_items, d.vocab.id_idx
    sl = _slices(d)
    for items in sl["cold_items"].values():
        assert all(i > n or idx[i] == 0 for i in items)
    for items in sl["warm_items"].values():
        assert all(i <= n and idx[i] != 0 for i in items)
    for u, items in d.truth_test.items():
        got = len(sl["cold_items"].get(u, [])) + len(sl["warm_items"].get(u, []))
        assert got == len(items)


def test_activity_terciles_balanced(processed_synth):
    d = processed_synth
    sl = _slices(d)
    sizes = [len(sl[k]) for k in ("users_low", "users_mid", "users_high")]
    assert sum(sizes) == len(d.truth_test)
    assert max(sizes) - min(sizes) <= 1
    uids = sorted(d.truth_test)
    rows = d.seqs_train.index_of(np.array(uids))
    lens = np.diff(d.seqs_train.offsets)
    act = {u: lens[r] for u, r in zip(uids, rows)}
    assert max(act[u] for u in sl["users_low"]) <= min(act[u] for u in sl["users_mid"])
    assert max(act[u] for u in sl["users_mid"]) <= min(act[u] for u in sl["users_high"])


def test_evaluate_slices_matches_evaluate(processed_synth):
    d = processed_synth
    rec = MostPop().fit(d.seqs_train, d.vocab.n_items)
    res = evaluate(rec, d.truth_test, d.vocab.n_items)
    m = evaluate_slices(res.topk, res.uids, _slices(d), d.vocab.n_items)
    assert abs(m["all"]["ndcg@10"] - res.metrics["ndcg@10"]) < 1e-12
    assert m["all"]["n_users"] == len(res.uids)
    assert m["organic"]["n_users"] <= len(res.uids)
