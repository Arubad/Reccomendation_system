import numpy as np
import polars as pl
import sklearn.decomposition as skd

from hookrec.data.features import ItemFeatures, build_item_features
from hookrec.data.label import label_events
from hookrec.data.split import global_temporal_split
from hookrec.data.vocab import build_vocab


def _setup(synth):
    lf = label_events(synth["multi_event"].lazy())
    train = global_temporal_split(lf).train
    emb = synth["embeddings"]
    vocab = build_vocab(lf, train, min_count=2, extra_item_ids=emb["item_id"].to_numpy())
    return lf, train, emb, vocab


def test_features_missing_audio(synth):
    _, train, emb, vocab = _setup(synth)
    f = build_item_features(vocab, emb, synth["artist_item_mapping"], train, pca_dim=8)
    assert f.audio.shape == (vocab.n_items + 1, 8) and f.audio.dtype == np.float16
    idx = vocab.encode(emb["item_id"].to_numpy())
    assert f.has_audio[idx].all() and not f.has_audio[0]
    missing = ~f.has_audio
    assert missing.sum() > 1  # pad + catalog items without embeddings
    assert (f.audio[missing] == 0).all()
    assert np.isfinite(f.audio.astype(np.float32)).all()
    assert f.audio[f.has_audio].any()
    assert f.artist_idx[0] == 0 and f.artist_idx.max() == f.n_artists


def test_no_pca_keeps_dim_and_accepts_array_form(synth):
    _, train, emb, vocab = _setup(synth)
    d = len(emb["embed"][0])
    arr = emb.select("item_id", pl.col("embed").list.to_array(d).cast(pl.Array(pl.Float16, d)))
    f = build_item_features(vocab, arr, synth["artist_item_mapping"], train, pca_dim=None)
    assert f.audio.shape[1] == d
    f2 = build_item_features(vocab, emb, synth["artist_item_mapping"], train, pca_dim=d)
    assert f2.audio.shape[1] == d


def test_primary_artist_and_pop(synth):
    _, train, emb, vocab = _setup(synth)
    am = synth["artist_item_mapping"]
    f = build_item_features(vocab, emb, am, train, pca_dim=None)
    first = am.group_by("item_id").agg(pl.col("artist_id").min())
    idx = vocab.encode(first["item_id"].to_numpy())
    a = first["artist_id"].to_numpy()
    known = idx > 0  # items absent from the catalog (no events, no embedding) stay out of the vocab
    gone = first["item_id"].to_numpy()[~known]
    assert not synth["multi_event"]["item_id"].is_in(gone).any()
    assert not emb["item_id"].is_in(gone).any()
    idx, a = idx[known], a[known]
    same_idx = f.artist_idx[idx][:, None] == f.artist_idx[idx][None, :]
    assert np.array_equal(same_idx, a[:, None] == a[None, :])
    assert (f.artist_idx[idx] > 0).all()
    expect = (
        train.filter(pl.col("is_positive")).group_by("item_id").agg(pl.len().alias("n")).collect()
    )
    assert f.pop_train.sum() == expect["n"].sum()
    assert f.pop_train[0] == 0


def test_pca_fit_on_train_only(synth, monkeypatch):
    _, train, emb, vocab = _setup(synth)
    seen = []
    orig_fit = skd.PCA.fit
    orig_ft = skd.PCA.fit_transform

    def fit(self, X, y=None):
        seen.append(len(X))
        return orig_fit(self, X, y)

    def fit_transform(self, X, y=None):
        seen.append(len(X))
        return orig_ft(self, X, y)

    monkeypatch.setattr(skd.PCA, "fit", fit)
    monkeypatch.setattr(skd.PCA, "fit_transform", fit_transform)
    build_item_features(vocab, emb, synth["artist_item_mapping"], train, pca_dim=8)
    train_items = set(train.select(pl.col("item_id").unique()).collect()["item_id"].to_list())
    n_expected = len(train_items & set(emb["item_id"].to_list()))
    assert seen[0] == n_expected
    assert n_expected < emb.height  # some embedded items are not in train


def test_roundtrip_save_load(synth, tmp_path):
    _, train, emb, vocab = _setup(synth)
    f = build_item_features(vocab, emb, synth["artist_item_mapping"], train, pca_dim=8)
    f.save(tmp_path / "f.npz")
    g = ItemFeatures.load(tmp_path / "f.npz")
    assert np.array_equal(f.audio, g.audio) and g.audio.dtype == np.float16
    assert np.array_equal(f.has_audio, g.has_audio)
    assert np.array_equal(f.artist_idx, g.artist_idx) and g.n_artists == f.n_artists
    assert np.array_equal(f.pop_train, g.pop_train)


def test_pca_applies_to_every_item(synth):
    _, train, emb, vocab = _setup(synth)
    f = build_item_features(vocab, emb, synth["artist_item_mapping"], train, pca_dim=8)
    assert f.audio.shape[1] == 8
    assert f.has_audio.sum() == emb.height


def test_duplicate_embedding_ids_keep_first(synth):
    _, train, emb, vocab = _setup(synth)
    dup = pl.concat([emb, emb.head(5).with_columns(pl.col("embed").list.eval(pl.element() * 0))])
    a = build_item_features(vocab, emb, synth["artist_item_mapping"], train, pca_dim=None)
    b = build_item_features(vocab, dup, synth["artist_item_mapping"], train, pca_dim=None)
    assert np.array_equal(a.audio, b.audio)


def test_pca_errors_are_explicit(synth):
    import pytest

    _, train, emb, vocab = _setup(synth)
    am = synth["artist_item_mapping"]
    with pytest.raises(ValueError, match="no train item"):
        build_item_features(vocab, emb.head(0), am, train, pca_dim=8)
    few = emb.head(3)
    with pytest.raises(ValueError, match="fewer than"):
        build_item_features(vocab, few, am, train, pca_dim=8)
