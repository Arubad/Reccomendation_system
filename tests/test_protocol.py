import numpy as np
import polars as pl
import pytest

from hookrec.data.vocab import build_vocab
from hookrec.eval.protocol import EvalResult, build_ground_truth, count_oov, evaluate


def _vocab():
    lf = pl.LazyFrame({"item_id": pl.Series([10, 20, 30], dtype=pl.UInt32), "is_positive": [True] * 3})
    return build_vocab(lf, lf, min_count=1)  # N=3, indices 1..3


def _test_lf():
    return pl.LazyFrame(
        {
            "uid": pl.Series([1, 1, 1, 2, 3, 3, 4, 4], dtype=pl.UInt32),
            "item_id": pl.Series([10, 10, 20, 999, 30, 5, 20, 10], dtype=pl.UInt32),
            "is_positive": [True, True, True, True, True, True, False, False],
        }
    )


def test_build_ground_truth():
    truth = build_ground_truth(_test_lf(), _vocab())
    assert sorted(truth) == [1, 2, 3]  # uid 4 has no positives
    assert sorted(truth[1]) == [1, 2]  # unique
    # OOV raw ids {5, 999} -> N+1=4, N+2=5 by ascending raw id
    assert list(truth[2]) == [5]  # 999 -> 5
    assert sorted(truth[3]) == [3, 4]  # 30 -> 3, 5 -> 4
    assert count_oov(truth, 3) == 2


class Fixed:
    def __init__(self, table):
        self.table, self.calls = table, []

    def recommend(self, uids, k):
        self.calls.append(len(uids))
        return np.stack([self.table[int(u)][:k] for u in uids])


def test_unknown_items_count_as_misses():
    truth = build_ground_truth(_test_lf(), _vocab())
    rec = Fixed({1: np.array([1, 2, 3]), 2: np.array([1, 2, 3]), 3: np.array([3, 1, 2])})
    res = evaluate(rec, truth, n_items=3, ks=(3,))
    assert list(res.uids) == [1, 2, 3]
    assert res.per_user["recall@3"] == pytest.approx([1.0, 0.0, 0.5])


def test_empty_truth_users_excluded_by_evaluate():
    truth = {5: np.array([1]), 2: np.array([], dtype=np.int64), 1: np.array([2])}
    rec = Fixed({1: np.array([2, 0]), 5: np.array([1, 0])})
    res = evaluate(rec, truth, n_items=3, ks=(2,))
    assert list(res.uids) == [1, 5]
    assert res.topk.shape == (2, 2)


def test_evaluate_batches_equal_unbatched():
    rng = np.random.default_rng(0)
    n = 40
    truth = {u: rng.choice(np.arange(1, 30), size=rng.integers(1, 6), replace=False) for u in range(n)}
    table = {u: rng.permutation(np.arange(1, 30))[:20] for u in range(n)}
    fa, fb = Fixed(table), Fixed(table)
    a = evaluate(fa, truth, n_items=29, ks=(5, 20), batch_users=7)
    b = evaluate(fb, truth, n_items=29, ks=(5, 20), batch_users=1000)
    assert isinstance(a, EvalResult)
    assert fa.calls == [7, 7, 7, 7, 7, 5] and fb.calls == [40]
    assert a.metrics == pytest.approx(b.metrics)
    assert set(a.metrics) == {"ndcg@5", "recall@5", "coverage@5", "ndcg@20", "recall@20", "coverage@20"}
    for key in a.per_user:
        assert np.allclose(a.per_user[key], b.per_user[key])
    assert a.topk.shape == (n, 20)
    assert a.metrics["ndcg@5"] == pytest.approx(a.per_user["ndcg@5"].mean())


def test_evaluate_rejects_short_recs():
    with pytest.raises(ValueError):
        evaluate(Fixed({1: np.array([1])}), {1: np.array([1])}, n_items=3, ks=(2,))
