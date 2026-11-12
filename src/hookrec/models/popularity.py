"""Popularity baselines and the cold-user fallback helper."""

from __future__ import annotations

import numpy as np

from hookrec.data.label import LIKE, LISTEN_FULL
from hookrec.data.processed import ProcessedData
from hookrec.data.sequences import PackedSequences
from hookrec.models.base import Split, history_for, register

SECONDS_PER_DAY = 86400


def _positive_events(seqs: PackedSequences) -> tuple[np.ndarray, np.ndarray]:
    """(item index, event position) of every positive event (LISTEN_FULL or LIKE)."""
    pos = np.flatnonzero((seqs.tokens == LISTEN_FULL) | (seqs.tokens == LIKE))
    return np.asarray(seqs.items)[pos].astype(np.int64), pos


class _ScoredPop:
    """Shared ranking: the same top-k (by score, ties to the lower index) for every user."""

    scores: np.ndarray  # float64[n_items + 1]; index 0 (pad) is never recommended

    def _rank(self, scores: np.ndarray) -> _ScoredPop:
        self.scores = scores
        scores = scores.copy()
        scores[0] = 0.0
        self._order = np.argsort(-scores, kind="stable")  # stable: ties keep lower index first
        self._order = self._order[scores[self._order] > 0]
        return self

    def top_items(self, k: int) -> np.ndarray:
        out = np.zeros(k, dtype=np.int64)  # 0-padded if fewer than k positive items
        top = self._order[:k]
        out[: len(top)] = top
        return out

    def recommend(self, uids: np.ndarray, k: int) -> np.ndarray:
        return np.tile(self.top_items(k), (len(uids), 1))


class MostPop(_ScoredPop):
    def fit(self, seqs: PackedSequences, n_items: int) -> MostPop:
        items, _ = _positive_events(seqs)
        return self._rank(np.bincount(items, minlength=n_items + 1).astype(np.float64))


class DecayPop(_ScoredPop):
    def __init__(self, half_life_days: float):
        self.half_life_days = float(half_life_days)

    def fit(self, seqs: PackedSequences, n_items: int, t_end: float) -> DecayPop:
        items, pos = _positive_events(seqs)
        age = t_end - np.asarray(seqs.ts)[pos].astype(np.float64)  # seconds (R11)
        w = 0.5 ** (age / (self.half_life_days * SECONDS_PER_DAY))
        return self._rank(np.bincount(items, weights=w, minlength=n_items + 1))


def fill_cold(
    uids: np.ndarray, recs: np.ndarray, fallback_topk: np.ndarray, known_mask: np.ndarray
) -> np.ndarray:
    """Replace the rows of unknown users (`known_mask` False) with `fallback_topk`."""
    out = np.array(recs, copy=True)
    out[~np.asarray(known_mask, dtype=bool)] = fallback_topk[: out.shape[1]]
    return out


@register("mostpop")
def _build_mostpop(data: ProcessedData, cfg: dict, split: Split) -> MostPop:
    seqs, _ = history_for(data, split)
    return MostPop().fit(seqs, data.vocab.n_items)


@register("decaypop")
def _build_decaypop(data: ProcessedData, cfg: dict, split: Split) -> DecayPop:
    seqs, t_end = history_for(data, split)
    return DecayPop(cfg["half_life_days"]).fit(seqs, data.vocab.n_items, t_end)
