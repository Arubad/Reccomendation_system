import numpy as np
import pytest

from hookrec.eval.fairness import artist_coverage, artist_exposure_gini, long_tail_share

# items 1..4 -> artists 0,0,1,2 (index 0 is pad)
ARTIST = np.array([0, 0, 0, 1, 2])


def test_artist_gini_uniform_zero():
    topk = np.array([[1, 3, 4]])  # one exposure per artist
    assert artist_exposure_gini(topk, ARTIST) == pytest.approx(0.0)


def test_artist_gini_includes_zero_exposure_artists_and_ignores_pad_oov():
    topk = np.array([[1, 2, 0, 99]])  # all exposure on artist 0; pad and >N ignored
    # counts [2, 0, 0] -> gini = 2/3
    assert artist_exposure_gini(topk, ARTIST) == pytest.approx(2 / 3)
    assert artist_exposure_gini(topk, ARTIST, n_artists=4) == pytest.approx(3 / 4)


def test_artist_coverage():
    topk = np.array([[1, 2, 0], [3, 0, 0]])
    assert artist_coverage(topk, ARTIST, 3) == pytest.approx(2 / 3)


def test_long_tail_share():
    pop = np.array([0, 10, 5, 5, 1])  # head_frac .5 -> 2 head items: 1, then 2 (tie -> lower idx)
    topk = np.array([[1, 2, 3, 4, 0]])
    assert long_tail_share(topk, pop, head_frac=0.5) == pytest.approx(0.5)
    assert long_tail_share(np.array([[0, 0]]), pop) == 0.0


def test_ignore_zero_excludes_unknown_artist():
    # items 1..5 -> artists 0 (unknown), 1, 1, 2, 3; real artists 1..3
    art = np.array([0, 0, 1, 1, 2, 3])
    topk = np.array([[1, 2, 3, 0, 99]])  # unknown item, artist 1 twice
    # counts over real artists [2, 0, 0] -> gini 2/3; coverage 1/3
    assert artist_exposure_gini(topk, art, 3, ignore_zero=True) == pytest.approx(2 / 3)
    assert artist_coverage(topk, art, 3, ignore_zero=True) == pytest.approx(1 / 3)
    # the last artist (3) is not dropped
    assert artist_coverage(np.array([[5]]), art, 3, ignore_zero=True) == pytest.approx(1 / 3)
    # unknown item alone gives no exposure at all
    assert artist_coverage(np.array([[1]]), art, 3, ignore_zero=True) == 0.0
    assert artist_exposure_gini(topk, art, ignore_zero=True) == pytest.approx(2 / 3)
