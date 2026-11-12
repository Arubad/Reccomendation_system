import numpy as np
import pytest

from hookrec.eval.bootstrap import bootstrap_ci
from hookrec.eval.metrics import coverage, ndcg_per_user, recall_full_per_user, recall_per_user


def _t(*xs):
    return np.array(xs, dtype=np.int64)


def test_ndcg_hand():
    got = ndcg_per_user(np.array([[1, 3, 2]]), [_t(1, 2)], k=3)
    want = (1 + 1 / np.log2(4)) / (1 + 1 / np.log2(3))
    assert got[0] == pytest.approx(want)


def test_ndcg_perfect_is_one():
    got = ndcg_per_user(np.array([[4, 5, 9], [7, 8, 9]]), [_t(4, 5), _t(7)], k=3)
    assert got == pytest.approx([1.0, 1.0])


def test_ndcg_idcg_capped_by_k():
    # 5 relevant items but k=2 -> ideal is two hits
    got = ndcg_per_user(np.array([[1, 2, 3]]), [_t(1, 2, 3, 4, 5)], k=2)
    assert got[0] == pytest.approx(1.0)


def test_recall_hand():
    got = recall_per_user(np.array([[1, 3, 2], [9, 8, 7]]), [_t(1, 2, 5, 6), _t(1)], k=3)
    # min(|T|, k) denominator: 2/3 for user 0 (4 positives, k=3), 0 for user 1
    assert got == pytest.approx([2 / 3, 0.0])
    assert recall_per_user(np.array([[1, 3, 2]]), [_t(1, 2, 5, 6)], k=2)[0] == pytest.approx(0.5)


def test_padding_is_never_a_hit_and_empty_truth_is_zero():
    ranked = np.array([[0, 0, 0], [1, 0, 0]])
    assert ndcg_per_user(ranked, [_t(1), _t()], k=3) == pytest.approx([0.0, 0.0])
    assert recall_per_user(ranked, [_t(1), _t()], k=3) == pytest.approx([0.0, 0.0])


def test_k_larger_than_ranked_raises():
    with pytest.raises(ValueError):
        ndcg_per_user(np.array([[1, 2]]), [_t(1)], k=3)


def test_coverage():
    ranked = np.array([[1, 2, 0], [2, 3, 0]])
    assert coverage(ranked, n_items=10, k=3) == pytest.approx(3 / 10)  # 0 is not an item
    assert coverage(ranked, n_items=10, k=1) == pytest.approx(2 / 10)


def test_bootstrap_ci_contains_mean():
    v = np.random.default_rng(1).random(500)
    mean, lo, hi = bootstrap_ci(v, n_boot=200)
    assert mean == pytest.approx(v.mean())
    assert lo <= mean <= hi and hi - lo < 0.2
    assert bootstrap_ci(v, n_boot=200) == (mean, lo, hi)  # seeded


def test_bootstrap_ci_empty():
    m, lo, hi = bootstrap_ci(np.array([]))
    assert np.isnan(m) and np.isnan(lo) and np.isnan(hi)


def test_recall_paper_hand():
    ranked = np.array([[1, 2, 3, 99, 98, 97, 96, 95, 94, 93], [9] * 10, [1] * 10])
    truth = [_t(1, 2, 3, *range(100, 117)), _t(1), _t()]
    assert len(truth[0]) == 20
    got = recall_full_per_user(ranked, truth, k=10)
    assert got == pytest.approx([3 / 20, 0.0, 0.0])  # hits / |T|, empty truth is 0
    # official recall divides by min(|T|, k) = 10
    assert recall_per_user(ranked, truth, k=10)[0] == pytest.approx(3 / 10)
