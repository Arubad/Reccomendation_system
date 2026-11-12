"""Torch datasets over PackedSequences: random-window training, last-window inference.

Pad contract: every field is left-padded with 0 and `items == 0` <=> pad (stored items are >= 1).
`gap_bucket` is one more such field (see `gap_buckets`). It is computed over the window plus
the one event before it (`ts[max(s - 1, 0) : s + w]`), so a window's first event still gets the
true gap to its predecessor while the cost stays O(window), not O(history): with
`max_len_store: null` a history can hold tens of thousands of events.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import Dataset

from hookrec.data.label import LIKE, LISTEN_FULL

if TYPE_CHECKING:
    from hookrec.data.sequences import PackedSequences

_INPUT_FIELDS = ("items", "tokens", "organic", "ratio")
DEFAULT_SESSION_GAP = 1800  # seconds; `data.session_gap_seconds` overrides (EDA suggestion)
N_GAP_BUCKETS = 8  # index 0 = pad / first-ever event, 1..7 = real gaps


WINDOW_SAMPLING = ("random", "last")


def session_gap_from(cfg: dict, data_cfg: dict | None = None) -> int:
    """The session-gap edge in seconds.

    Precedence: the training config's `data.session_gap_seconds` (a per-run override), then the
    preprocessing config's top-level `session_gap_seconds` (`data_cfg`, i.e. `ProcessedData.cfg`,
    set in `configs/data_*.yaml` from the EDA), then 1800 s.
    """
    v = (cfg.get("data") or {}).get("session_gap_seconds")
    if v is None and data_cfg is not None:
        v = data_cfg.get("session_gap_seconds")
    return v or DEFAULT_SESSION_GAP


def window_sampling_from(cfg: dict) -> str:
    """`cfg["data"]["window_sampling"]`: "random" (default) or "last" (R24)."""
    v = (cfg.get("data") or {}).get("window_sampling", "random")
    if v not in WINDOW_SAMPLING:
        raise ValueError(f"data.window_sampling must be one of {WINDOW_SAMPLING}, got {v!r}")
    return v


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


def _window_gap_bucket(ts: np.ndarray, s: int, w: int, session_gap: float) -> np.ndarray:
    """[w] buckets of events s..s+w-1 of one user, reading only ts[max(s - 1, 0) : s + w]."""
    lo = max(s - 1, 0)
    gb = _user_gap_bucket(np.asarray(ts[lo : s + w]), session_gap)
    return gb[s - lo :]


def _left_pad(a: np.ndarray, length: int) -> torch.Tensor:
    out = np.zeros(length, dtype=np.int64)
    if len(a):
        out[length - len(a) :] = a
    return torch.from_numpy(out)


class SeqTrainDataset(Dataset):
    """One window of `max_len + 1` events per user per epoch; the target at position t is event
    t+1's item if that event is positive (LISTEN_FULL or LIKE), else 0 (ignored). On Listen+
    sequences (`data.input_filter: listen_plus`) every event is LISTEN_FULL, so the target is
    simply the next item, as in the official SASRec benchmark.

    `window_sampling` (`data.window_sampling`): "random" draws the window start (see `_start`);
    "last" always takes the user's last window, like the official SASRec (one last-200 window
    per user).
    """

    def __init__(
        self,
        seqs: PackedSequences,
        max_len: int,
        seed: int,
        session_gap: float = DEFAULT_SESSION_GAP,
        window_sampling: str = "random",
    ):
        self.seqs, self.max_len, self.seed, self.epoch = seqs, max_len, seed, 0
        self.session_gap = session_gap
        if window_sampling not in WINDOW_SAMPLING:
            raise ValueError(f"window_sampling must be one of {WINDOW_SAMPLING}")
        self.window_sampling = window_sampling
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
        s = n - w if self.window_sampling == "last" else self._start(rng, n - w)
        win = {k: v[s : s + w] for k, v in e.items()}
        win["gap_bucket"] = _window_gap_bucket(e["ts"], s, w, self.session_gap)
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
        w = min(len(e["ts"]), self.max_len)
        gb = _window_gap_bucket(e["ts"], len(e["ts"]) - w, w, self.session_gap)
        out["gap_bucket"] = _left_pad(gb, self.max_len)
        out["uid"] = torch.tensor(int(self.seqs.uids[idx]), dtype=torch.long)
        return out
