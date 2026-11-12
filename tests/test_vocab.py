import numpy as np
import polars as pl

from hookrec.data.vocab import ItemVocab, build_vocab


def _lf(items, positive=None):
    d = {"item_id": pl.Series(items, dtype=pl.UInt32)}
    d["is_positive"] = [True] * len(items) if positive is None else positive
    return pl.LazyFrame(d)


def test_vocab_min_count():
    train = _lf([10] * 5 + [20])
    v = build_vocab(train, train, min_count=2)
    a, b = v.encode(np.array([10, 20]))
    assert v.id_idx[a] > 0 and v.id_idx[b] == 0
    assert v.n_items == 2 and v.n_id == 1


def test_vocab_ignores_non_positive_for_id():
    train = _lf([10] * 5, positive=[False] * 5)
    v = build_vocab(train, train, min_count=1)
    assert v.n_id == 0 and v.n_items == 1


def test_vocab_includes_test_only_and_extra_items():
    train = _lf([10, 10])
    all_ev = _lf([10, 10, 99])
    v = build_vocab(all_ev, train, min_count=1, extra_item_ids=np.array([5], dtype=np.uint32))
    assert list(v.raw_ids) == [0, 5, 10, 99]
    i99, i5 = v.encode(np.array([99, 5]))
    assert v.id_idx[i99] == 0 and v.id_idx[i5] == 0
    assert v.id_idx[v.encode(np.array([10]))[0]] == 1


def test_encode_unknown_is_zero():
    train = _lf([10, 20])
    v = build_vocab(train, train, min_count=1)
    out = v.encode(np.array([10, 15, 20, 0, 999], dtype=np.uint32))
    assert list(out) == [1, 0, 2, 0, 0]


def test_roundtrip_save_load(tmp_path):
    train = _lf([10, 10, 20])
    v = build_vocab(train, train, min_count=2)
    v.save(tmp_path / "v.npz")
    w = ItemVocab.load(tmp_path / "v.npz")
    assert np.array_equal(v.raw_ids, w.raw_ids) and np.array_equal(v.id_idx, w.id_idx)
    assert (w.n_items, w.n_id) == (v.n_items, v.n_id)
