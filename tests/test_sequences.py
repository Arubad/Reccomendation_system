import numpy as np
import polars as pl

from hookrec.data.clean import clean_events
from hookrec.data.label import LIKE, LISTEN_FULL, LISTEN_SKIP, label_events
from hookrec.data.sequences import PackedSequences, build_sequences, build_sequences_sharded
from hookrec.data.split import global_temporal_split
from hookrec.data.vocab import build_vocab

FIELDS = ("uids", "offsets", "items", "tokens", "organic", "ratio", "ts")


def _events(rows):
    """rows: (uid, item_id, timestamp, token)."""
    return pl.LazyFrame(
        {
            "uid": pl.Series([r[0] for r in rows], dtype=pl.UInt32),
            "item_id": pl.Series([r[1] for r in rows], dtype=pl.UInt32),
            "timestamp": pl.Series([r[2] for r in rows], dtype=pl.UInt32),
            "is_organic": pl.Series([1] * len(rows), dtype=pl.UInt8),
            "token": pl.Series([r[3] for r in rows], dtype=pl.UInt8),
            "ratio_bucket": pl.Series([r[2] % 6 for r in rows], dtype=pl.UInt8),
            "is_positive": [True] * len(rows),
        }
    )


def _vocab(lf):
    return build_vocab(lf, lf, min_count=1)


def test_build_sequences_order_and_truncation():
    rows = [
        (7, 10, 5, LISTEN_FULL),
        (3, 20, 9, LISTEN_FULL),
        (7, 30, 1, LISTEN_FULL),
        (7, 20, 3, LISTEN_SKIP),
        (3, 10, 2, LIKE),
        (7, 10, 3, LISTEN_FULL),  # ts tie with (7,20,3): original order is kept
    ]
    lf = _events(rows)
    v = _vocab(lf)
    s = build_sequences(lf, v, max_len=None, collapse_consecutive_dups=False)
    assert list(s.uids) == [3, 7] and list(s.offsets) == [0, 2, 6]
    assert list(s.ts) == [2, 9, 1, 3, 3, 5]
    raw = v.raw_ids[s.items]
    assert list(raw[2:]) == [30, 20, 10, 10]
    t = build_sequences(lf, v, max_len=3, collapse_consecutive_dups=False)
    assert list(t.offsets) == [0, 2, 5]
    assert list(t.ts) == [2, 9, 3, 3, 5]  # user 7 keeps its last 3


def test_collapse_dups():
    rows = [
        (1, 10, 1, LISTEN_FULL),
        (1, 10, 2, LISTEN_FULL),
        (1, 10, 3, LISTEN_SKIP),
        (1, 20, 4, LISTEN_SKIP),
        (1, 10, 5, LISTEN_SKIP),
        (2, 10, 6, LISTEN_SKIP),  # same (item, token) as user 1's tail, different user
    ]
    lf = _events(rows)
    s = build_sequences(lf, _vocab(lf), max_len=None, collapse_consecutive_dups=True)
    assert list(s.ts) == [2, 3, 4, 5, 6]
    assert list(s.offsets) == [0, 4, 5]


def test_sharded_equals_unsharded(synth):
    lf = label_events(clean_events(synth["multi_event"].lazy()))
    split = global_temporal_split(lf)
    vocab = build_vocab(lf, split.train, min_count=2)
    for kw in (
        {"max_len": None, "collapse_consecutive_dups": False},
        {"max_len": 20, "collapse_consecutive_dups": True},
    ):
        a = build_sequences(split.train, vocab, **kw)
        b = build_sequences_sharded(lambda: split.train, vocab, n_shards=3, **kw)
        assert len(a) > 3
        for f in FIELDS:
            assert np.array_equal(getattr(a, f), getattr(b, f)), f


def test_roundtrip_mmap(tmp_path):
    rows = [(1, 10, 1, LISTEN_FULL), (1, 20, 2, LIKE), (4, 10, 3, LISTEN_SKIP)]
    lf = _events(rows)
    s = build_sequences(lf, _vocab(lf), max_len=None, collapse_consecutive_dups=False)
    p = tmp_path / "seq"
    s.save(p)
    w = PackedSequences.load(p)
    for f in FIELDS:
        assert isinstance(getattr(w, f), np.memmap), f
        assert np.array_equal(getattr(s, f), getattr(w, f)), f
        assert getattr(w, f).dtype == getattr(s, f).dtype
    assert len(w) == 2
    assert list(w.get(0)["ts"]) == [1, 2]


def test_oov_events_dropped(tmp_path):
    from hookrec.train.datasets import SeqInferDataset

    rows = [(1, 10, 1, LISTEN_FULL), (1, 20, 2, LISTEN_FULL), (1, 77, 3, LISTEN_FULL)]
    lf = _events(rows)
    v = build_vocab(lf.filter(pl.col("item_id") != 77), lf.filter(pl.col("item_id") != 77), 1)
    s = build_sequences(lf, v, max_len=None, collapse_consecutive_dups=False)
    assert s.n_dropped_oov == 1 and list(s.ts) == [1, 2] and (s.items > 0).all()
    x = SeqInferDataset(s, 3)[0]
    assert x["items"].tolist() == [0, int(s.items[0]), int(s.items[1])]


def test_null_organic_and_ratio_filled():
    lf = _events([(1, 10, 1, LISTEN_FULL), (1, 20, 2, LISTEN_FULL)]).with_columns(
        pl.Series("is_organic", [None, 1], dtype=pl.UInt8),
        pl.Series("ratio_bucket", [3, None], dtype=pl.UInt8),
    )
    s = build_sequences(lf, _vocab(lf), max_len=None, collapse_consecutive_dups=False)
    assert list(s.organic) == [0, 1] and list(s.ratio) == [3, 0]


def test_index_of():
    rows = [(1, 10, 1, LISTEN_FULL), (4, 10, 3, LISTEN_SKIP), (9, 10, 3, LISTEN_SKIP)]
    lf = _events(rows)
    s = build_sequences(lf, _vocab(lf), max_len=None, collapse_consecutive_dups=False)
    assert list(s.index_of(np.array([9, 2, 1, 4, 100], dtype=np.uint32))) == [2, -1, 0, 1, -1]


def test_empty_input():
    lf = _events([(1, 10, 1, LISTEN_FULL)])
    s = build_sequences(
        lf.filter(pl.col("uid") == 99), _vocab(lf), max_len=5, collapse_consecutive_dups=True
    )
    assert len(s) == 0 and list(s.offsets) == [0]
