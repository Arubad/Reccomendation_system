"""Popularity baselines and the cold-user fallback helper.

The registered builders fit on the full, untruncated training period's Listen+ aggregates
(`hookrec.data.aggregates`, ruling R18), as the official Yambda popularity baselines do on the
Listen+ interactions. The `fit(seqs, ...)` methods count positives (LISTEN_FULL or LIKE) of a
`PackedSequences` and are kept for direct use on small, untruncated histories.
"""

from __future__ import annotations

import numpy as np

from hookrec.data.label import LIKE, LISTEN_FULL
from hookrec.data.processed import ProcessedData
from hookrec.data.sequences import PackedSequences
from hookrec.models.base import Split, aggregates_for, register

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

    def fit_counts(self, counts: np.ndarray) -> MostPop:
        """Rank by per-item counts [N+1] (e.g. a period's `pop_lp`)."""
        return self._rank(np.asarray(counts, dtype=np.float64).copy())


class DecayPop(_ScoredPop):
    """Exponentially time-decayed Listen+ counts, weight 0.5 ** (age / half_life).

    The official Yambda DecayPop (`benchmarks/models/popularity/main.py`) weights each event
    `tau ** (max_train_ts - ts)` with `tau = 0.9 ** (1 / (hours * 3600))`: a continuous decay
    of 0.9 per `hours` window (no flooring), grid hours in {0.5, 1, 2, 3, 6, 12, 24}. That is
    the same family, with half_life = hours * ln 0.5 / ln 0.9 ~= 6.579 * hours; the
    `configs/baselines.yaml` grid holds those equivalents (R25). Two differences remain:
    - age is measured from the period end `t_end`, not the last train event; a constant shift
      that multiplies every score equally and so never changes the ranking;
    - the registry fit (`fit_days`) bins events by day (R18), so every event of a day gets the
      weight of the day's middle. For half-lives well under a day (the official 0.5-3 h
      windows) this is a coarse approximation of the per-event decay; `fit` is exact.
    """

    def __init__(self, half_life_days: float):
        half_life_days = float(half_life_days)
        if not half_life_days > 0:  # also rejects NaN
            raise ValueError(f"half_life_days must be > 0, got {half_life_days}")
        self.half_life_days = half_life_days

    def fit(self, seqs: PackedSequences, n_items: int, t_end: float) -> DecayPop:
        items, pos = _positive_events(seqs)
        age = t_end - np.asarray(seqs.ts)[pos].astype(np.float64)  # seconds (R11)
        w = 0.5 ** (age / (self.half_life_days * SECONDS_PER_DAY))
        return self._rank(np.bincount(items, weights=w, minlength=n_items + 1))

    def fit_days(
        self, item_idx: np.ndarray, day: np.ndarray, count: np.ndarray, n_items: int
    ) -> DecayPop:
        """Fit from per-day counts, day = (t_end - ts) // 86400 (an `item_day_*` aggregate).

        Every event of day d gets the weight of age (d + 0.5) days, the middle of its day, so
        its age differs from the exact per-event age by at most half a day (< 1 day).
        """
        age_days = np.asarray(day, dtype=np.float64) + 0.5
        w = np.asarray(count, dtype=np.float64) * 0.5 ** (age_days / self.half_life_days)
        return self._rank(
            np.bincount(np.asarray(item_idx, dtype=np.int64), weights=w, minlength=n_items + 1)
        )


def fill_cold(
    uids: np.ndarray, recs: np.ndarray, fallback_topk: np.ndarray, known_mask: np.ndarray
) -> np.ndarray:
    """Replace the rows of unknown users (`known_mask` False) with `fallback_topk`.

    A fallback shorter than the rec width is 0-padded (0 is never a hit).
    """
    out = np.array(recs, copy=True)
    k = out.shape[1]
    row = np.zeros(k, dtype=out.dtype)
    row[: min(k, len(fallback_topk))] = fallback_topk[:k]
    out[~np.asarray(known_mask, dtype=bool)] = row
    return out


def mostpop_fallback(data: ProcessedData, split: Split) -> MostPop:
    """MostPop over the full training period's Listen+ counts: the cold-user fallback."""
    return MostPop().fit_counts(aggregates_for(data, split).pop_lp)


@register("mostpop")
def _build_mostpop(data: ProcessedData, cfg: dict, split: Split) -> MostPop:
    return mostpop_fallback(data, split)


@register("decaypop")
def _build_decaypop(data: ProcessedData, cfg: dict, split: Split) -> DecayPop:
    if cfg.get("half_life_days") is None:
        raise ValueError("decaypop needs cfg['half_life_days'] (> 0), but it is missing")
    a = aggregates_for(data, split)
    return DecayPop(cfg["half_life_days"]).fit_days(
        a.day_items, a.day_index, a.day_counts, data.vocab.n_items
    )
