import importlib.util
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from hookrec.eda.stats import (
    DAY_UNITS as DAY,
)
from hookrec.eda.stats import (
    daily_stats,
    embedding_coverage,
    gini,
    organic_share,
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


def _load_runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_eda.py"
    spec = importlib.util.spec_from_file_location("run_eda", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_eda_smoke(synth_dir, tmp_path):
    mod = _load_runner()

    out = tmp_path / "reports"
    mod.main(["--raw", str(synth_dir), "--size", "50m", "--out", str(out)])
    stats = json.loads((out / "eda_stats.json").read_text())
    assert _STAT_KEYS <= set(stats)
    assert {"item_gini", "powerlaw_alpha", "n_items"} <= set(stats["popularity"])
    assert 0 <= stats["cold_items"]["share_test_items_unseen"] <= 1
    assert len(list((out / "figures").glob("*.png"))) >= 8


def _ladder() -> pl.LazyFrame:
    """Entity u (1..10) has u events; only entity 10 is organic."""
    n = np.arange(1, 11)
    ids = np.repeat(n, n)
    return pl.LazyFrame(
        {
            "uid": ids,
            "item_id": ids,
            "timestamp": np.arange(len(ids)),
            "is_organic": (ids == 10).astype(int),
        }
    )


def test_organic_share_by_day():
    lf = pl.LazyFrame(
        {"uid": [1, 1, 1], "item_id": [1, 2, 3], "timestamp": [0, 1, DAY], "is_organic": [1, 0, 1]}
    )
    df = organic_share(lf, "day")
    assert df["day"].to_list() == [0, 1]
    assert df["organic_share"].to_list() == [0.5, 1.0]
    assert df["events"].to_list() == [2, 1]


@pytest.mark.parametrize("by", ["user_activity_decile", "item_pop_decile"])
def test_organic_share_by_decile(by):
    df = organic_share(_ladder(), by)
    assert df["decile"].to_list() == list(range(1, 11))
    assert df["organic_share"].to_list() == [0.0] * 9 + [1.0]
    assert df["events"].to_list() == list(range(1, 11))


def test_organic_share_bad_grouping():
    with pytest.raises(ValueError):
        organic_share(_ladder(), "nope")


def test_embedding_coverage_known():
    df = embedding_coverage(_ladder(), np.array([2, 4, 6, 8, 10, 999]))
    assert df["decile"].to_list() == list(range(1, 11))
    assert df["n_items"].to_list() == [1] * 10
    assert df["coverage"].to_list() == [float(i % 2 == 0) for i in range(1, 11)]
    assert df["event_coverage"].to_list() == df["coverage"].to_list()


def test_daily_stats_churn():
    # day 0: items 1,2 ; day 1: items 2,3 -> Jaccard distance 1 - 1/3
    lf = pl.LazyFrame({"uid": [1] * 4, "item_id": [1, 2, 2, 3], "timestamp": [0, 1, DAY, DAY + 1]})
    df = daily_stats(lf)
    assert df["new_items"].to_list() == [2, 1]
    assert df["churn"][0] is None
    assert df["churn"][1] == pytest.approx(2 / 3)


def test_jsonable_maps_nan_inf_to_null():
    out = _load_runner()._jsonable(
        {
            "a": float("nan"),
            "b": np.float64("inf"),
            "c": [np.float32("-inf"), 1.5],
            "d": np.int64(3),
        }
    )
    assert out == {"a": None, "b": None, "c": [None, 1.5], "d": 3}
    json.dumps(out, allow_nan=False)


def test_run_eda_smoke_substructure(synth_dir, tmp_path):
    out = tmp_path / "r"
    stats = _load_runner().main(["--raw", str(synth_dir), "--out", str(out)])
    assert set(stats["gap"]) == {"n_gaps", "suggested_session_gap_seconds"}
    assert set(stats["repeat"]) == {"relisten_share", "consecutive_dup_share"}
    assert set(stats["seq_length"]) == {"q50", "q90", "q95", "q99", "mean", "max"}
    assert set(stats["organic_share"]) == {"day", "user_activity_decile", "item_pop_decile"}
    assert {"churn", "new_items"} <= set(stats["daily"][0])
    assert (out / "figures" / "embedding_map.png").exists()
    cr = stats["clean_report"]
    assert cr["rows_after"] <= cr["rows_before"]
