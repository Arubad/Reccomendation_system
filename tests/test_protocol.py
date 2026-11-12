import numpy as np
import polars as pl
import pytest

from hookrec.data.label import LIKE, LISTEN_FULL, LISTEN_SKIP
from hookrec.data.vocab import build_vocab
from hookrec.eval.protocol import EvalResult, build_ground_truth, count_oov, evaluate


def _vocab():
    lf = pl.LazyFrame({"item_id": pl.Series([10, 20, 30], dtype=pl.UInt32), "is_positive": [True] * 3})
    return build_vocab(lf, lf, min_count=1)  # N=3, indices 1..3


def _test_lf(with_likes: bool = False):
    F, S, L = LISTEN_FULL, LISTEN_SKIP, LIKE
    uid = [1, 1, 1, 2, 3, 3, 4, 4]
    item = [10, 10, 20, 999, 30, 5, 20, 10]
    tok = [F, F, F, F, F, F, S, S]
    if with_likes:  # uid 4 likes 30; uid 1 likes 30 (a like-only item); uid 5 only likes
        uid, item, tok = uid + [4, 1, 5], item + [30, 30, 10], tok + [L, L, L]
    return pl.LazyFrame(
        {
            "uid": pl.Series(uid, dtype=pl.UInt32),
            "item_id": pl.Series(item, dtype=pl.UInt32),
            "token": pl.Series(tok, dtype=pl.UInt8),
            "is_positive": [t in (F, L) for t in tok],
        }
    )


def test_build_ground_truth():
    truth = build_ground_truth(_test_lf(), _vocab())
    assert sorted(truth) == [1, 2, 3]  # uid 4 has no Listen+
    assert sorted(truth[1]) == [1, 2]  # unique
    # OOV raw ids {5, 999} -> N+1=4, N+2=5 by ascending raw id
    assert list(truth[2]) == [5]  # 999 -> 5
    assert sorted(truth[3]) == [3, 4]  # 30 -> 3, 5 -> 4
    assert count_oov(truth, 3) == 2


def test_ground_truth_listen_plus_excludes_likes():
    lp = build_ground_truth(_test_lf(with_likes=True), _vocab())
    assert sorted(lp) == [1, 2, 3]  # like-only users 4 and 5 have no Listen+ truth
    assert sorted(lp[1]) == [1, 2]  # the like-only item 30 (idx 3) is not a target
    pos = build_ground_truth(_test_lf(with_likes=True), _vocab(), target="positive")
    assert sorted(pos) == [1, 2, 3, 4, 5]
    assert sorted(pos[1]) == [1, 2, 3] and pos[4].tolist() == [3] and pos[5].tolist() == [1]
    with pytest.raises(ValueError, match="eval.target"):
        build_ground_truth(_test_lf(), _vocab(), target="likes")


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
    assert set(a.metrics) == {
        f"{m}@{k}" for m in ("ndcg", "recall", "recall_paper", "coverage") for k in (5, 20)
    }
    assert set(a.per_user) == {f"{m}@{k}" for m in ("ndcg", "recall", "recall_paper") for k in (5, 20)}
    assert a.metrics["recall_paper@5"] <= a.metrics["recall@5"] + 1e-12
    for key in a.per_user:
        assert np.allclose(a.per_user[key], b.per_user[key])
    assert a.topk.shape == (n, 20)
    assert a.metrics["ndcg@5"] == pytest.approx(a.per_user["ndcg@5"].mean())


def test_evaluate_rejects_short_recs():
    with pytest.raises(ValueError):
        evaluate(Fixed({1: np.array([1])}), {1: np.array([1])}, n_items=3, ks=(2,))


def test_split_by_history_warm_cold():
    from hookrec.eval.protocol import split_by_history

    truth = {1: np.array([5]), 2: np.array([6, 7]), 9: np.array([1])}
    warm, cold = split_by_history(truth, np.array([2, 2, 1, 4]))
    assert set(warm) == {1, 2} and set(cold) == {9}
    assert warm[2].tolist() == [6, 7]
    w2, c2 = split_by_history({}, np.array([], dtype=np.int64))
    assert w2 == {} and c2 == {}
    w3, c3 = split_by_history(truth, np.array([], dtype=np.int64))
    assert w3 == {} and set(c3) == {1, 2, 9}
