import importlib.util
from pathlib import Path

import numpy as np
import polars as pl

from hookrec.data.clean import clean_events
from hookrec.data.label import LISTEN_FULL
from hookrec.eval.protocol import evaluate
from hookrec.eval.slices import evaluate_slices, slice_truth, train_seen_items
from hookrec.models.popularity import MostPop

ROOT = Path(__file__).resolve().parents[1]


def _load_script(script):
    spec = importlib.util.spec_from_file_location(script, ROOT / "scripts" / f"{script}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _slices(d):
    return slice_truth(
        d.truth_test, d.test_events(), d.vocab, d.features, d.agg_train, seqs_train=d.seqs("train")
    )


def test_slices_partition_organic_served(processed_synth):
    d = processed_synth
    sl = _slices(d)
    assert sl["all"].keys() == d.truth_test.keys()
    for u, items in d.truth_test.items():
        got = np.union1d(sl["organic"].get(u, items[:0]), sl["served"].get(u, items[:0]))
        # OOV items never appear in test_events (they encode to 0): only in-vocab are covered
        inv = items[items <= d.vocab.n_items]
        assert set(got.tolist()) == set(inv.tolist())  # organic | served == all in-vocab items


def test_organic_served_come_from_labeled_events(processed_synth):
    d = processed_synth
    sl = _slices(d)
    ev = d.test_events().filter(pl.col("token") == LISTEN_FULL).collect()  # Listen+ truth
    org = {
        (int(u), int(i))
        for u, i in ev.filter(pl.col("is_organic").cast(pl.Boolean))
        .select("uid", "item_idx")
        .rows()
    }
    for u, items in sl["organic"].items():
        assert all((u, int(i)) in org for i in items)
    assert sum(len(v) for v in sl["organic"].values()) > 0
    assert sum(len(v) for v in sl["served"].values()) > 0


def test_idless_slice_only_idless_or_unknown(processed_synth):
    d = processed_synth
    n, idx = d.vocab.n_items, d.vocab.id_idx
    sl = _slices(d)
    assert "cold_items" not in sl
    for items in sl["idless_items"].values():
        assert all(i > n or idx[i] == 0 for i in items)
    for items in sl["warm_items"].values():
        assert all(i <= n and idx[i] != 0 for i in items)
    for u, items in d.truth_test.items():
        got = len(sl["idless_items"].get(u, [])) + len(sl["warm_items"].get(u, []))
        assert got == len(items)


def _train_items_brute(synth, d) -> set[int]:
    """Vocab indices of every item with a cleaned train-period event of any kind, from the raw
    synthetic events (the preprocessing path: clean, then ts < t_train_end)."""
    ev = (
        clean_events(synth["multi_event"].lazy())
        .filter(pl.col("timestamp") < d.meta["t_train_end"])
        .select(pl.col("item_id").unique())
        .collect()
    )
    idx = d.vocab.encode(ev["item_id"].to_numpy())
    return set(idx[idx > 0].tolist())


def test_new_items_slice_brute_force(processed_synth, synth):
    d = processed_synth
    n = d.vocab.n_items
    seen = _train_items_brute(synth, d)
    assert np.flatnonzero(train_seen_items(d.seqs("train"), d.agg_train)).tolist() == sorted(seen)
    sl = _slices(d)
    want = {}
    for u, items in d.truth_test.items():
        keep = [int(i) for i in items if i > n or int(i) not in seen]
        if keep:
            want[u] = keep
    assert {u: v.tolist() for u, v in sl["new_items"].items()} == want
    assert sum(len(v) for v in want.values()) > 0, "synthetic test truth should have new items"
    for u, items in sl["new_items"].items():  # never seen -> no ID either
        assert set(items.tolist()) <= set(sl["idless_items"][u].tolist())
    # an item whose only events are in the test day is new
    test_only = {int(i) for i in d.test_events().collect()["item_idx"].unique()} - seen
    truth_items = {int(i) for v in d.truth_test.values() for i in v}
    assert test_only & truth_items
    assert test_only & truth_items <= {int(i) for v in sl["new_items"].values() for i in v}


def test_new_items_share_matches_preprocessing(processed_synth):
    d = processed_synth
    run_eval = _load_script("run_eval")
    got = run_eval.new_item_share(d, train_seen_items(d.seqs("train"), d.agg_train))
    assert got["meta"] == d.meta["share_unknown_test_items"]
    assert abs(got["slice"] - got["meta"]) < 1e-12  # full histories at 50m: exact


def test_activity_terciles_balanced(processed_synth):
    d = processed_synth
    sl = _slices(d)
    sizes = [len(sl[k]) for k in ("users_low", "users_mid", "users_high")]
    assert sum(sizes) == len(d.truth_test)
    assert max(sizes) - min(sizes) <= 1
    uids = sorted(d.truth_test)
    # full-period Listen+ counts (brute force from the CSR rows), not truncated sequences
    dense = d.agg_train.ui.toarray()
    rows = d.agg_train.index_of(np.array(uids))
    act = {u: int(dense[r].sum()) for u, r in zip(uids, rows)}
    assert all(r >= 0 for r in rows)  # every warm user has a Listen+ row
    assert max(act[u] for u in sl["users_low"]) <= min(act[u] for u in sl["users_mid"])
    assert max(act[u] for u in sl["users_mid"]) <= min(act[u] for u in sl["users_high"])


def test_evaluate_slices_matches_evaluate(processed_synth):
    d = processed_synth
    rec = MostPop().fit_counts(d.agg_train.pop_lp)
    res = evaluate(rec, d.truth_test, d.vocab.n_items)
    m = evaluate_slices(res.topk, res.uids, _slices(d), d.vocab.n_items)
    assert abs(m["all"]["ndcg@10"] - res.metrics["ndcg@10"]) < 1e-12
    assert m["all"]["n_users"] == len(res.uids)
    assert m["organic"]["n_users"] <= len(res.uids)
