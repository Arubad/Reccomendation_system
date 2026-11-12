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
