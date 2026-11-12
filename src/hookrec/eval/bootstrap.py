"""Percentile bootstrap confidence intervals over per-user metric values."""

from __future__ import annotations

import numpy as np

_CHUNK_CELLS = 20_000_000  # cap on resample-index cells held in memory at once


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """(mean, lo, hi): mean of `values` and the percentile CI of resampled means."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    step = max(1, _CHUNK_CELLS // n)
    means = np.empty(n_boot)
    for start in range(0, n_boot, step):
        m = min(step, n_boot - start)
        means[start : start + m] = values[rng.integers(0, n, size=(m, n))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(values.mean()), float(lo), float(hi)
