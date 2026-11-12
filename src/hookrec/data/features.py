"""Item features: audio (optionally PCA-reduced), primary artist, train popularity."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.decomposition import PCA

from hookrec.data.vocab import ItemVocab


@dataclass
class ItemFeatures:
    audio: np.ndarray  # f16 (N+1, D), zeros where missing
    has_audio: np.ndarray  # bool (N+1,)
    artist_idx: np.ndarray  # i32 (N+1,), 0 = unknown
    n_artists: int
    pop_train: np.ndarray  # i64 (N+1,), positive train events per item index

    def save(self, path) -> None:
        np.savez(
            Path(path),
            audio=self.audio,
            has_audio=self.has_audio,
            artist_idx=self.artist_idx,
            n_artists=np.int64(self.n_artists),
            pop_train=self.pop_train,
        )

    @classmethod
    def load(cls, path) -> ItemFeatures:
        with np.load(Path(path)) as z:
            return cls(
                audio=z["audio"],
                has_audio=z["has_audio"],
                artist_idx=z["artist_idx"],
                n_artists=int(z["n_artists"]),
                pop_train=z["pop_train"],
            )


F16_MAX = 6e4
PCA_CHUNK = 200_000


def _embedding_matrix(emb_df: pl.DataFrame, column: str) -> np.ndarray:
    """float32 (rows, D) from either a list[f32] or an Array(Float16, D) column, vectorised."""
    col = emb_df[column]
    if isinstance(col.dtype, pl.Array):
        return col.cast(pl.Array(pl.Float32, col.dtype.size)).to_numpy()
    if len(col) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    d = int(col.list.len().first())
    return col.cast(pl.List(pl.Float32)).list.to_array(d).to_numpy()


def _to_f16(x: np.ndarray) -> np.ndarray:
    return np.clip(x, -F16_MAX, F16_MAX).astype(np.float16)


def build_item_features(
    vocab: ItemVocab,
    emb_df: pl.DataFrame,
    artist_df: pl.DataFrame,
    train_lf: pl.LazyFrame,
    pca_dim: int | None = 128,
    seed: int = 0,
    emb_column: str = "embed",
) -> ItemFeatures:
    n1 = vocab.n_items + 1

    # Audio: align embedding rows to catalog indices (rows outside the catalog are dropped,
    # duplicate item ids keep the first row).
    emb_df = emb_df.unique(subset="item_id", keep="first", maintain_order=True)
    emb_idx = vocab.encode(emb_df["item_id"].to_numpy())
    keep = emb_idx > 0
    mat = _embedding_matrix(emb_df, emb_column)
    if not keep.all():
        mat = mat[keep]
        emb_idx = emb_idx[keep]
    if not mat.flags.writeable:  # polars may hand back a read-only view
        mat = mat.copy()
    np.nan_to_num(mat, copy=False, nan=0.0)
    has_audio = np.zeros(n1, dtype=bool)
    has_audio[emb_idx] = True

    # PCA sees only items with >= 1 train event (any kind) and audio, then maps all of them.
    train_ids = train_lf.select(pl.col("item_id").unique()).collect()["item_id"].to_numpy()
    train_idx = vocab.encode(train_ids)
    in_train = np.zeros(n1, dtype=bool)
    in_train[train_idx[train_idx > 0]] = True
    fit_mask = in_train[emb_idx]

    dim = mat.shape[1]
    if pca_dim is not None and pca_dim < dim:
        n_fit = int(fit_mask.sum())
        if n_fit == 0:
            raise ValueError("pca_dim is set but no train item has an audio embedding")
        if n_fit < pca_dim:
            raise ValueError(f"only {n_fit} train items with audio, fewer than pca_dim={pca_dim}")
        pca = PCA(n_components=pca_dim, random_state=seed)
        pca.fit(mat[fit_mask])
        audio = np.zeros((n1, pca_dim), dtype=np.float16)
        for lo in range(0, len(mat), PCA_CHUNK):
            sl = slice(lo, lo + PCA_CHUNK)
            audio[emb_idx[sl]] = _to_f16(pca.transform(mat[sl]))
    else:
        audio = np.zeros((n1, dim), dtype=np.float16)
        audio[emb_idx] = _to_f16(mat)
    if not np.isfinite(audio).all():
        raise ValueError("non-finite values in audio features")

    # Primary artist = smallest artist_id; compact 1..n_artists over catalog items only.
    prim = artist_df.group_by("item_id").agg(pl.col("artist_id").min())
    a_item = vocab.encode(prim["item_id"].to_numpy())
    a_id = prim["artist_id"].to_numpy()
    ok = a_item > 0
    a_item, a_id = a_item[ok], a_id[ok]
    uniq, inv = np.unique(a_id, return_inverse=True)
    artist_idx = np.zeros(n1, dtype=np.int32)
    artist_idx[a_item] = inv.astype(np.int32) + 1

    # Popularity: positive train events per item index.
    pop = (
        train_lf.filter(pl.col("is_positive")).group_by("item_id").agg(pl.len().alias("n")).collect()
    )
    p_idx = vocab.encode(pop["item_id"].to_numpy())
    p_ok = p_idx > 0
    pop_train = np.zeros(n1, dtype=np.int64)
    pop_train[p_idx[p_ok]] = pop["n"].to_numpy()[p_ok]

    return ItemFeatures(
        audio=audio,
        has_audio=has_audio,
        artist_idx=artist_idx,
        n_artists=len(uniq),
        pop_train=pop_train,
    )
