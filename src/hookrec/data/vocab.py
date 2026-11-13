"""Item vocabulary: catalog indices 1..N (0 = pad) plus compact ID-embedding indices."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import polars as pl


@dataclass
class ItemVocab:
    raw_ids: np.ndarray  # u32, len N+1, raw_ids[0] = 0 (pad), ascending after that
    id_idx: np.ndarray  # i32, len N+1; 0 = no ID embedding, else 1..n_id
    n_items: int
    n_id: int

    def encode(self, raw: np.ndarray) -> np.ndarray:
        """Raw ids -> item indices; unknown ids map to 0. Vectorised.

        Index 0 is the pad only: a real catalog item with raw id 0 gets an index >= 1.
        """
        raw = np.asarray(raw)
        catalog = self.raw_ids[1:]
        if len(catalog) == 0:
            return np.zeros(raw.shape, dtype=np.int64)
        pos = np.minimum(np.searchsorted(catalog, raw), len(catalog) - 1)
        found = catalog[pos] == raw
        return np.where(found, pos + 1, 0).astype(np.int64)

    def save(self, path) -> None:
        np.savez(
            Path(path),
            raw_ids=self.raw_ids,
            id_idx=self.id_idx,
            n_items=np.int64(self.n_items),
            n_id=np.int64(self.n_id),
        )

    @classmethod
    def load(cls, path) -> ItemVocab:
        with np.load(Path(path)) as z:
            return cls(
                raw_ids=z["raw_ids"],
                id_idx=z["id_idx"],
                n_items=int(z["n_items"]),
                n_id=int(z["n_id"]),
            )


def build_vocab(
    all_events_lf: pl.LazyFrame,
    train_lf: pl.LazyFrame,
    min_count: int,
    extra_item_ids: np.ndarray | None = None,
) -> ItemVocab:
    """Catalog = items in any split plus `extra_item_ids`; ID embeddings only for items with
    >= `min_count` positive events in `train_lf` (which must already be labeled)."""
    ids = (
        all_events_lf.select(pl.col("item_id").unique())
        .collect(engine="streaming")["item_id"]
        .to_numpy()
    )
    if extra_item_ids is not None:
        ids = np.concatenate([ids, np.asarray(extra_item_ids)])
    catalog = np.unique(ids.astype(np.uint32))
    raw_ids = np.concatenate([np.zeros(1, dtype=np.uint32), catalog])

    pos = (
        train_lf.filter(pl.col("is_positive"))
        .group_by("item_id")
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") >= min_count)
        .collect(engine="streaming")["item_id"]
        .to_numpy()
    )
    at = np.minimum(np.searchsorted(catalog, pos), max(len(catalog) - 1, 0))
    if len(pos) and not np.array_equal(catalog[at], pos):
        raise ValueError("train positives contain items missing from the catalog")
    eligible = np.zeros(len(raw_ids), dtype=bool)
    eligible[at + 1] = True
    id_idx = np.zeros(len(raw_ids), dtype=np.int32)
    n_id = int(eligible.sum())
    id_idx[eligible] = np.arange(1, n_id + 1, dtype=np.int32)
    return ItemVocab(raw_ids=raw_ids, id_idx=id_idx, n_items=len(catalog), n_id=n_id)


def id_index_from_counts(pos_counts: np.ndarray, min_count: int) -> tuple[np.ndarray, int]:
    """(id_idx i32 [N+1], n_id): items with >= `min_count` positive train events get IDs
    1..n_id in item-index order, the assignment `build_vocab` makes from the same counts.

    `pos_counts` is i64 [N+1]: positive (LISTEN_FULL or LIKE) events per item over the
    vocab's training period `split.train`, i.e. `ProcessedData.agg("train").pop_pos`.
    """
    if isinstance(min_count, bool) or not isinstance(min_count, (int, np.integer)):
        raise TypeError(f"min_count must be an int, got {min_count!r}")
    if min_count < 1:
        raise ValueError(f"min_count must be >= 1, got {min_count}")
    eligible = np.asarray(pos_counts) >= min_count
    eligible[0] = False  # index 0 is the pad
    id_idx = np.zeros(len(eligible), dtype=np.int32)
    n_id = int(eligible.sum())
    id_idx[eligible] = np.arange(1, n_id + 1, dtype=np.int32)
    return id_idx, n_id


def with_id_min_count(vocab: ItemVocab, pos_counts: np.ndarray, min_count: int) -> ItemVocab:
    """`vocab` with its ID table rebuilt at `min_count` (same catalog and raw ids)."""
    if len(pos_counts) != vocab.n_items + 1:
        raise ValueError(f"pos_counts has {len(pos_counts)} rows, expected {vocab.n_items + 1}")
    id_idx, n_id = id_index_from_counts(pos_counts, min_count)
    return replace(vocab, id_idx=id_idx, n_id=n_id)
