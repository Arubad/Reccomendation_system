"""Torch datasets over PackedSequences: random-window training, last-window inference.

Pad contract: every field is left-padded with 0 and `items == 0` <=> pad (stored items are >= 1).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from hookrec.data.label import LIKE, LISTEN_FULL
from hookrec.data.sequences import PackedSequences

_INPUT_FIELDS = ("items", "tokens", "organic", "ratio")


def _left_pad(a: np.ndarray, length: int) -> torch.Tensor:
    out = np.zeros(length, dtype=np.int64)
    if len(a):
        out[length - len(a) :] = a
    return torch.from_numpy(out)


class SeqTrainDataset(Dataset):
    def __init__(self, seqs: PackedSequences, max_len: int, seed: int):
        self.seqs, self.max_len, self.seed, self.epoch = seqs, max_len, seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.seqs)

    @staticmethod
    def _start(rng: np.random.Generator, m: int) -> int:
        """Window start in [0, m]: the last window w.p. 0.5, else p(s) proportional to s+1."""
        if m == 0 or rng.random() < 0.5:
            return m
        # inverse CDF: P(S <= s) = (s+1)(s+2) / ((m+1)(m+2))
        u = rng.random() * (m + 1) * (m + 2)
        s = int(np.ceil((-3 + np.sqrt(1 + 4 * u)) / 2))
        return min(max(s, 0), m)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng([self.seed, self.epoch, idx])
        e = self.seqs.get(idx)
        n = len(e["items"])
        w = min(n, self.max_len + 1)
        s = self._start(rng, n - w)
        win = {k: v[s : s + w] for k, v in e.items()}
        out = {k: _left_pad(win[k][:-1], self.max_len) for k in _INPUT_FIELDS}
        tgt_tok = win["tokens"][1:]
        positive = (tgt_tok == LISTEN_FULL) | (tgt_tok == LIKE)
        out["target"] = _left_pad(np.where(positive, win["items"][1:], 0), self.max_len)
        out["target_is_like"] = _left_pad(tgt_tok == LIKE, self.max_len).bool()
        return out


class SeqInferDataset(Dataset):
    def __init__(self, seqs: PackedSequences, max_len: int):
        self.seqs, self.max_len = seqs, max_len

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        e = self.seqs.get(idx)
        out = {k: _left_pad(e[k][-self.max_len :], self.max_len) for k in _INPUT_FIELDS}
        out["uid"] = torch.tensor(int(self.seqs.uids[idx]), dtype=torch.long)
        return out
