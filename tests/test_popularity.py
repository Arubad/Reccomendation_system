import numpy as np
import pytest

from hookrec.data.label import LIKE, LISTEN_FULL, LISTEN_SKIP
from hookrec.data.sequences import PackedSequences
from hookrec.eval.protocol import evaluate
from hookrec.models.base import REGISTRY, fit_recommender
from hookrec.models.popularity import DecayPop, MostPop, fill_cold

DAY = 86400


def _seqs(rows):
    """rows: per-user lists of (item, token, ts)."""
    lens = [len(r) for r in rows]
    flat = [e for r in rows for e in r]
    n = len(flat)
    return PackedSequences(
        uids=np.arange(len(rows), dtype=np.uint32),
        offsets=np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
        items=np.array([e[0] for e in flat], dtype=np.int32),
        tokens=np.array([e[1] for e in flat], dtype=np.uint8),
        organic=np.zeros(n, dtype=np.uint8),
        ratio=np.zeros(n, dtype=np.uint8),
        ts=np.array([e[2] for e in flat], dtype=np.uint32),
    )


def test_mostpop_order():
    seqs = _seqs(
        [
            [(3, LISTEN_FULL, 0), (3, LIKE, 1), (2, LISTEN_FULL, 2), (1, LISTEN_FULL, 3)],
            [(3, LISTEN_FULL, 0), (2, LISTEN_FULL, 1), (4, LISTEN_SKIP, 2), (4, LISTEN_SKIP, 3)],
        ]
    )
    m = MostPop().fit(seqs, n_items=5)
    out = m.recommend(np.array([0, 1, 99]), 5)
    assert out.shape == (3, 5) and out.dtype == np.int64
    # counts: 3->3, 2->2, 1->1 (skips ignored); only 3 positive items -> 0-padded
    assert out[0].tolist() == [3, 2, 1, 0, 0]
    assert (out == out[0]).all()


def test_mostpop_tie_breaks_by_lower_index():
    seqs = _seqs([[(4, LISTEN_FULL, 0), (2, LISTEN_FULL, 1), (3, LISTEN_FULL, 2)]])
    out = MostPop().fit(seqs, n_items=5).recommend(np.array([0]), 3)
    assert out[0].tolist() == [2, 3, 4]


def test_decaypop_prefers_recent():
    t_end = 100 * DAY
    seqs = _seqs(
        [
            [(1, LISTEN_FULL, 10 * DAY), (1, LISTEN_FULL, 11 * DAY)],
            [(2, LISTEN_FULL, 98 * DAY), (2, LISTEN_FULL, 99 * DAY)],
        ]
    )
    m = DecayPop(half_life_days=7.0).fit(seqs, n_items=3, t_end=t_end)
    assert m.recommend(np.array([0]), 2)[0].tolist() == [2, 1]
    assert MostPop().fit(seqs, 3).recommend(np.array([0]), 2)[0].tolist() == [1, 2]  # tie -> idx


def test_decaypop_weight_value():
    seqs = _seqs([[(1, LISTEN_FULL, 0)]])
    m = DecayPop(half_life_days=1.0).fit(seqs, n_items=1, t_end=DAY)
    assert m.scores[1] == pytest.approx(0.5)


def test_fill_cold():
    recs = np.array([[5, 6], [7, 8], [9, 1]])
    out = fill_cold(np.array([10, 11, 12]), recs, np.array([1, 2]), np.array([True, False, False]))
    assert out.tolist() == [[5, 6], [1, 2], [1, 2]]
    assert recs[1].tolist() == [7, 8]  # input not mutated


def test_mostpop_eval_on_synth_runs(processed_synth):
    d = processed_synth
    m = fit_recommender("mostpop", d, {}, "val")
    res = evaluate(m, d.truth_val, d.vocab.n_items)
    keys = {"ndcg@10", "ndcg@100", "recall@10", "recall@100", "coverage@10", "coverage@100"}
    assert keys <= set(res.metrics)
    assert all(np.isfinite(res.metrics[k]) for k in keys)


def test_mostpop_cold_users_evaluate(processed_synth):
    d = processed_synth
    assert len(d.truth_test_cold) > 0
    m = fit_recommender("mostpop", d, {}, "test")
    res = evaluate(m, d.truth_test_cold, d.vocab.n_items)
    assert np.isfinite(res.metrics["ndcg@10"])


def test_registry_split_and_errors(processed_synth):
    d = processed_synth
    assert {"mostpop", "decaypop"} <= set(REGISTRY)
    val = fit_recommender("decaypop", d, {"half_life_days": 7}, "val")
    test = fit_recommender("decaypop", d, {"half_life_days": 7}, "test")
    assert val.scores.shape == test.scores.shape == (d.vocab.n_items + 1,)
    with pytest.raises(KeyError):
        fit_recommender("nope", d, {}, "val")
    with pytest.raises(ValueError):
        fit_recommender("mostpop", d, {}, "dev")


def test_decaypop_validates_half_life():
    for bad in (0, -1.0):
        with pytest.raises(ValueError, match="half_life_days"):
            DecayPop(half_life_days=bad)
    with pytest.raises(ValueError, match="half_life_days"):
        DecayPop(half_life_days=float("nan"))


def test_decaypop_builder_missing_half_life(processed_synth):
    with pytest.raises(ValueError, match="half_life_days"):
        fit_recommender("decaypop", processed_synth, {}, "val")


def test_decaypop_per_day_within_half_a_day_of_per_event():
    rng = np.random.default_rng(0)
    t_end, h = 50 * DAY, 3.0
    ts = rng.integers(0, t_end, size=500)
    items = rng.integers(1, 6, size=500)
    exact = np.bincount(items, weights=0.5 ** ((t_end - ts) / (h * DAY)), minlength=6)
    day = (t_end - ts) // DAY
    key, cnt = np.unique(np.stack([items, day]), axis=1, return_counts=True)
    got = DecayPop(h).fit_days(key[0], key[1], cnt, n_items=5).scores
    bound = 0.5 ** (-0.5 / h)  # each event's age is off by at most half a day
    assert got[0] == exact[0] == 0
    assert np.all(got[1:] <= exact[1:] * bound) and np.all(got[1:] >= exact[1:] / bound)
    one = DecayPop(1.0).fit_days(np.array([1]), np.array([0]), np.array([4]), n_items=1)
    assert one.scores[1] == pytest.approx(4 * 0.5**0.5)


def test_registry_baselines_fit_full_period_aggregates(processed_synth):
    d = processed_synth
    for split, agg in (("val", d.agg_val_train), ("test", d.agg_train)):
        m = fit_recommender("mostpop", d, {}, split)
        np.testing.assert_array_equal(m.scores, agg.pop_lp.astype(np.float64))
        dp = fit_recommender("decaypop", d, {"half_life_days": 1e9}, split)
        np.testing.assert_allclose(dp.scores, agg.pop_lp, rtol=1e-6)  # ~no decay = MostPop


def test_decaypop_grid_matches_the_official_windows():
    """R25: official weight 0.9 ** (age / window) for windows 0.5..24 h equals our
    0.5 ** (age / half_life) with half_life = window * ln 0.5 / ln 0.9."""
    from pathlib import Path

    from hookrec.utils.config import load_config

    root = Path(__file__).resolve().parents[1]
    grid = load_config(root / "configs" / "baselines.yaml")["decaypop"]["grid"]["half_life_days"]
    hours = (0.5, 1, 2, 3, 6, 12, 24)
    want = [h / 24 * np.log(0.5) / np.log(0.9) for h in hours]
    assert len(grid) == len(hours) + 1 and grid[-1] == 30
    np.testing.assert_allclose(grid[:-1], want, rtol=0.03)  # 2 significant digits
    age = np.linspace(0, 5 * DAY, 11)
    for h, hl in zip(hours, want):
        np.testing.assert_allclose(0.9 ** (age / (h * 3600)), 0.5 ** (age / (hl * DAY)))


def test_fill_cold_pads_short_fallback():
    recs = np.array([[5, 6, 7], [8, 9, 4]])
    out = fill_cold(np.array([1, 2]), recs, np.array([1, 2]), np.array([True, False]))
    assert out.tolist() == [[5, 6, 7], [1, 2, 0]]
