import importlib.util
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from hookrec.eda.stats import (
    gini,
    powerlaw_alpha,
    repeat_stats,
    suggest_session_gap,
)

_STAT_KEYS = {
    "clean_report",
    "popularity",
    "played_ratio_hist",
    "gap",
    "repeat",
    "organic_share",
    "daily",
    "seq_length",
    "embedding_coverage",
    "like_rate_by_ratio",
    "cold_items",
}


def test_gini_known_values():
    assert gini(np.array([1, 1, 1, 1])) == 0
    assert gini(np.array([0, 0, 0, 1])) == pytest.approx(0.75)


def test_powerlaw_alpha_recovers():
    x = np.random.default_rng(0).zipf(2.0, 50_000)
    assert powerlaw_alpha(x) == pytest.approx(2.0, abs=0.15)


def test_suggest_session_gap_bimodal():
    rng = np.random.default_rng(0)
    short = rng.normal(np.log10(180), 0.3, 20_000)
    long_ = rng.normal(np.log10(86_400), 0.3, 10_000)
    gap = suggest_session_gap(np.concatenate([short, long_]))
    assert 600 <= gap <= 6 * 3600


def test_suggest_session_gap_fallback():
    assert suggest_session_gap(np.array([])) == 1800
    assert suggest_session_gap(np.random.default_rng(0).normal(3, 0.3, 5000)) == 1800


def test_repeat_stats_simple():
    lf = pl.LazyFrame({"uid": [1, 1, 1, 1], "item_id": [10, 10, 20, 10], "timestamp": [1, 2, 3, 4]})
    r = repeat_stats(lf)
    assert r["consecutive_dup_share"] == pytest.approx(1 / 3)
    assert r["relisten_share"] == pytest.approx(2 / 4)


def test_run_eda_smoke(synth_dir, tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_eda.py"
    spec = importlib.util.spec_from_file_location("run_eda", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    out = tmp_path / "reports"
    mod.main(["--raw", str(synth_dir), "--size", "50m", "--out", str(out)])
    stats = json.loads((out / "eda_stats.json").read_text())
    assert _STAT_KEYS <= set(stats)
    assert {"item_gini", "powerlaw_alpha", "n_items"} <= set(stats["popularity"])
    assert 0 <= stats["cold_items"]["share_test_items_unseen"] <= 1
    assert len(list((out / "figures").glob("*.png"))) >= 8
