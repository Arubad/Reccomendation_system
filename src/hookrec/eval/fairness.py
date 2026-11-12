"""Exposure fairness metrics on recommendation lists (`topk`: (U, K) item indices).

Item 0 is padding and indices > N are out-of-vocabulary ground-truth placeholders; both are
ignored. `artist_idx` has length N+1 and maps item index -> dense artist index in [0, n_artists).
"""

from __future__ import annotations

import numpy as np

from hookrec.eda.stats import gini


def _valid_items(topk: np.ndarray, n_items: int) -> np.ndarray:
    flat = np.asarray(topk).ravel()
    return flat[(flat >= 1) & (flat <= n_items)]


def _artist_counts(topk: np.ndarray, artist_idx: np.ndarray, n_artists: int | None) -> np.ndarray:
    n_items = len(artist_idx) - 1
    if n_artists is None:
        n_artists = int(artist_idx[1:].max()) + 1 if n_items > 0 else 0
    items = _valid_items(topk, n_items)
    return np.bincount(artist_idx[items], minlength=n_artists)[:n_artists]


def artist_exposure_gini(
    topk: np.ndarray, artist_idx: np.ndarray, n_artists: int | None = None
) -> float:
    """Gini of recommendation counts per artist, over all artists (zero-exposure ones included)."""
    return gini(_artist_counts(topk, artist_idx, n_artists))


def artist_coverage(topk: np.ndarray, artist_idx: np.ndarray, n_artists: int) -> float:
    """Share of artists that appear at least once in `topk`."""
    return float((_artist_counts(topk, artist_idx, n_artists) > 0).sum() / n_artists)


def long_tail_share(topk: np.ndarray, pop_train: np.ndarray, head_frac: float = 0.2) -> float:
    """Share of recommendations outside the head (top `head_frac` of items by train popularity).

    `pop_train` has length N+1 (index 0 unused). Ties in popularity break toward the lower index.
    """
    n_items = len(pop_train) - 1
    items = _valid_items(topk, n_items)
    if len(items) == 0:
        return 0.0
    n_head = max(1, int(np.ceil(head_frac * n_items)))
    order = np.lexsort((np.arange(1, n_items + 1), -np.asarray(pop_train[1:])))
    is_head = np.zeros(n_items + 1, dtype=bool)
    is_head[order[:n_head] + 1] = True
    return float(1.0 - is_head[items].mean())
