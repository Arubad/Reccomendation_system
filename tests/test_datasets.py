import numpy as np
import torch

from hookrec.data.label import LIKE, LISTEN_FULL, LISTEN_SKIP
from hookrec.data.sequences import PackedSequences
from hookrec.train.datasets import SeqInferDataset, SeqTrainDataset


def _seqs(users):
    """users: {uid: [(item, token), ...]}"""
    uids = np.array(sorted(users), dtype=np.uint32)
    ev = [e for u in uids for e in users[int(u)]]
    offsets = np.concatenate([[0], np.cumsum([len(users[int(u)]) for u in uids])]).astype(np.int64)
    n = len(ev)
    return PackedSequences(
        uids=uids,
        offsets=offsets,
        items=np.array([e[0] for e in ev], dtype=np.int32),
        tokens=np.array([e[1] for e in ev], dtype=np.uint8),
        organic=np.ones(n, dtype=np.uint8),
        ratio=np.full(n, 2, dtype=np.uint8),
        ts=np.arange(n, dtype=np.uint32),
    )


A, B, C = 1, 2, 3


def test_train_dataset_targets_shifted():
    ds = SeqTrainDataset(_seqs({1: [(A, LISTEN_FULL), (B, LISTEN_SKIP), (C, LIKE)]}), 2, seed=0)
    x = ds[0]  # n=3 <= max_len+1, so the window is the whole sequence
    assert x["items"].tolist() == [A, B]
    assert x["target"].tolist() == [0, C]
    assert x["target_is_like"].tolist() == [False, True]
    assert x["target_is_like"].dtype == torch.bool
    for k in ("items", "tokens", "organic", "ratio", "target"):
        assert x[k].dtype == torch.long


def test_left_padding():
    ds = SeqTrainDataset(_seqs({1: [(A, LISTEN_FULL), (B, LISTEN_FULL)]}), 5, seed=0)
    x = ds[0]
    assert x["items"].tolist() == [0, 0, 0, 0, A]
    assert x["tokens"].tolist() == [0, 0, 0, 0, LISTEN_FULL]
    assert x["organic"].tolist() == [0, 0, 0, 0, 1]
    assert x["target"].tolist() == [0, 0, 0, 0, B]
    assert x["target_is_like"].tolist() == [False] * 5


def test_all_negative_user_targets_all_zero():
    ds = SeqTrainDataset(_seqs({1: [(A, LISTEN_SKIP)] * 6, 2: [(A, LISTEN_FULL)]}), 4, seed=0)
    for i in range(len(ds)):
        x = ds[i]
        assert x["target"].sum() == 0 and not x["target_is_like"].any()


def test_train_window_sampling_and_determinism():
    n, L = 30, 4
    seqs = _seqs({1: [(i + 1, LISTEN_FULL) for i in range(n)]})
    ds = SeqTrainDataset(seqs, L, seed=3)
    assert torch.equal(ds[0]["items"], ds[0]["items"])  # same epoch -> same window
    m = n - (L + 1)
    got = []
    for e in range(4000):
        ds.set_epoch(e)
        x = ds[0]
        assert x["items"].shape == (L,)
        assert torch.equal(x["target"][:-1], x["items"][1:])  # contiguous window
        got.append(int(x["items"][0]) - 1)
    got = np.array(got)
    assert got.min() >= 0 and got.max() == m
    assert abs((got == m).mean() - (0.5 + 1 / (m + 2))) < 0.04
    assert got[got < m].mean() > (m - 1) / 2  # weighted to recent
    ds2 = SeqTrainDataset(seqs, L, seed=3)
    ds.set_epoch(5)
    ds2.set_epoch(5)
    assert torch.equal(ds[0]["items"], ds2[0]["items"])


def test_single_event_user_has_zero_targets():
    ds = SeqTrainDataset(_seqs({1: [(A, LISTEN_FULL)]}), 3, seed=0)
    x = ds[0]
    assert x["target"].sum() == 0 and x["items"].tolist() == [0, 0, 0]


def test_infer_dataset_last_window():
    seqs = _seqs({5: [(A, LISTEN_FULL), (B, LISTEN_SKIP), (C, LIKE)], 8: [(B, LISTEN_FULL)]})
    ds = SeqInferDataset(seqs, 2)
    x = ds[0]
    assert x["items"].tolist() == [B, C] and x["tokens"].tolist() == [LISTEN_SKIP, LIKE]
    assert x["uid"].item() == 5 and x["uid"].dtype == torch.long and x["uid"].ndim == 0
    y = ds[1]
    assert y["items"].tolist() == [0, B] and y["uid"].item() == 8
    train_keys = set(SeqTrainDataset(seqs, 2, 0)[0]) - {"target", "target_is_like"}
    assert train_keys == set(x) - {"uid"}
