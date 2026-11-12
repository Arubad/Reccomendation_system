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


def _embedding_matrix(emb_df: pl.DataFrame, column: str) -> np.ndarray:
    """float32 (rows, D) from either a list[f32] or an Array(Float16, D) column."""
    col = emb_df[column]
    if isinstance(col.dtype, pl.Array):
        return col.cast(pl.Array(pl.Float32, col.dtype.size)).to_numpy()
    return np.asarray(col.to_list(), dtype=np.float32)


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

    # Audio: align embedding rows to catalog indices (rows outside the catalog are dropped).
    emb_idx = vocab.encode(emb_df["item_id"].to_numpy())
    keep = emb_idx > 0
    mat = np.nan_to_num(_embedding_matrix(emb_df, emb_column)[keep], nan=0.0)
    emb_idx = emb_idx[keep]
    has_audio = np.zeros(n1, dtype=bool)
    has_audio[emb_idx] = True

    # PCA sees only items with >= 1 train event (any kind) and audio, then maps all of them.
    train_ids = train_lf.select(pl.col("item_id").unique()).collect()["item_id"].to_numpy()
    train_idx = vocab.encode(train_ids)
    in_train = np.zeros(n1, dtype=bool)
    in_train[train_idx[train_idx > 0]] = True
    fit_mask = in_train[emb_idx]

    dim = mat.shape[1]
    if pca_dim is not None and pca_dim < dim and fit_mask.any():
        pca = PCA(n_components=pca_dim, random_state=seed)
        pca.fit(mat[fit_mask])
        mat = pca.transform(mat)
        dim = pca_dim
    audio = np.zeros((n1, dim), dtype=np.float16)
    audio[emb_idx] = mat.astype(np.float16)

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
