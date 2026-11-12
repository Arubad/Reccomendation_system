"""Torch datasets over PackedSequences: random-window training, last-window inference.

Pad contract: every field is left-padded with 0 and `items == 0` <=> pad (stored items are >= 1).
`gap_bucket` is one more such field (see `gap_buckets`), computed over the user's full packed
sequence so a window's first event still gets the true gap to its predecessor.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from hookrec.data.label import LIKE, LISTEN_FULL
from hookrec.data.sequences import PackedSequences

_INPUT_FIELDS = ("items", "tokens", "organic", "ratio")
DEFAULT_SESSION_GAP = 1800  # seconds; `data.session_gap_seconds` overrides (EDA suggestion)
N_GAP_BUCKETS = 8  # index 0 = pad / first-ever event, 1..7 = real gaps


def session_gap_from(cfg: dict) -> int:
    """The session-gap edge: `cfg["data"]["session_gap_seconds"]` if set, else 1800 s."""
    return (cfg.get("data") or {}).get("session_gap_seconds") or DEFAULT_SESSION_GAP


def gap_buckets(gap_seconds: np.ndarray, session_gap: float = DEFAULT_SESSION_GAP) -> np.ndarray:
    """Bucket index (1..7) of each gap in seconds since the user's previous event.

    Edge -> index table (lower edge inclusive, so a gap equal to an edge goes up):

        [0, 30) -> 1    [30, 120) -> 2    [120, 600) -> 3    [600, session_gap) -> 4
        [session_gap, 21600) -> 5    [21600, 86400) -> 6    [86400, inf) -> 7

    The intended edges `[0, 30, 120, 600, session_gap, 6h, 1d, 7d, inf)` make 8 intervals; the
    last two (1 d to 7 d, 7 d and over) are merged so there are exactly 7 real buckets and
    index 0 stays free for pad / first event (`N_GAP_BUCKETS` = 8). `session_gap` must lie
    strictly between 600 and 21600 so the edges stay increasing.
    """
    if not 600 < session_gap < 21600:
        raise ValueError(f"session_gap must lie in (600, 21600) seconds, got {session_gap}")
    edges = np.array([30, 120, 600, session_gap, 21600, 86400])
    return np.searchsorted(edges, np.asarray(gap_seconds), side="right").astype(np.int64) + 1


def _user_gap_bucket(ts: np.ndarray, session_gap: float) -> np.ndarray:
    """[n] bucket of each event of one user; the first event is 0."""
    out = np.zeros(len(ts), dtype=np.int64)
    if len(ts) > 1:
        out[1:] = gap_buckets(np.diff(ts.astype(np.int64)), session_gap)
    return out


def _left_pad(a: np.ndarray, length: int) -> torch.Tensor:
    out = np.zeros(length, dtype=np.int64)
    if len(a):
        out[length - len(a) :] = a
    return torch.from_numpy(out)


class SeqTrainDataset(Dataset):
    def __init__(
        self,
        seqs: PackedSequences,
        max_len: int,
        seed: int,
        session_gap: float = DEFAULT_SESSION_GAP,
    ):
        self.seqs, self.max_len, self.seed, self.epoch = seqs, max_len, seed, 0
        self.session_gap = session_gap
        gap_buckets(np.array([0]), session_gap)  # validate early

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
        win["gap_bucket"] = _user_gap_bucket(e["ts"], self.session_gap)[s : s + w]
        out = {k: _left_pad(win[k][:-1], self.max_len) for k in (*_INPUT_FIELDS, "gap_bucket")}
        tgt_tok = win["tokens"][1:]
        positive = (tgt_tok == LISTEN_FULL) | (tgt_tok == LIKE)
        out["target"] = _left_pad(np.where(positive, win["items"][1:], 0), self.max_len)
        out["target_is_like"] = _left_pad(tgt_tok == LIKE, self.max_len).bool()
        return out


class SeqInferDataset(Dataset):
    def __init__(
        self, seqs: PackedSequences, max_len: int, session_gap: float = DEFAULT_SESSION_GAP
    ):
        self.seqs, self.max_len, self.session_gap = seqs, max_len, session_gap
        gap_buckets(np.array([0]), session_gap)  # validate early

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        e = self.seqs.get(idx)
        out = {k: _left_pad(e[k][-self.max_len :], self.max_len) for k in _INPUT_FIELDS}
        gb = _user_gap_bucket(e["ts"], self.session_gap)
        out["gap_bucket"] = _left_pad(gb[-self.max_len :], self.max_len)
        out["uid"] = torch.tensor(int(self.seqs.uids[idx]), dtype=torch.long)
        return out
