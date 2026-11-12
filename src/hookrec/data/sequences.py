"""Packed per-user event sequences (CSR-style arrays), built vectorised from labeled events."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from hookrec.data.vocab import ItemVocab

log = logging.getLogger(__name__)

_FIELDS = ("items", "tokens", "organic", "ratio", "ts")
_DTYPES = {
    "uids": np.uint32,
    "offsets": np.int64,
    "items": np.int32,
    "tokens": np.uint8,
    "organic": np.uint8,
    "ratio": np.uint8,
    "ts": np.uint32,
}


@dataclass
class PackedSequences:
    """CSR-style per-user sequences.

    Pad contract: `items == 0` if and only if the position is padding. Out-of-vocab events are
    dropped at build time (counted in `n_dropped_oov`), so every stored item is >= 1.
    """

    uids: np.ndarray  # u32[U], ascending
    offsets: np.ndarray  # i64[U+1]
    items: np.ndarray  # i32[T] vocab indices
    tokens: np.ndarray  # u8[T]
    organic: np.ndarray  # u8[T]
    ratio: np.ndarray  # u8[T]
    ts: np.ndarray  # u32[T]
    n_dropped_oov: int = 0  # events dropped at build time (item not in vocab); not persisted

    def __len__(self) -> int:
        return len(self.uids)

    def get(self, i: int) -> dict[str, np.ndarray]:
        a, b = int(self.offsets[i]), int(self.offsets[i + 1])
        return {f: getattr(self, f)[a:b] for f in _FIELDS}

    def index_of(self, uids: np.ndarray) -> np.ndarray:
        """Row index of each uid in `self.uids`; -1 if absent."""
        uids = np.asarray(uids)
        if len(self.uids) == 0:
            return np.full(uids.shape, -1, dtype=np.int64)
        pos = np.minimum(np.searchsorted(self.uids, uids), len(self.uids) - 1)
        return np.where(self.uids[pos] == uids, pos, -1).astype(np.int64)

    def save(self, path) -> None:
        """Write a directory with one .npy file per field (so `load` can memory-map them)."""
        d = Path(path)
        d.mkdir(parents=True, exist_ok=True)
        for k in _DTYPES:
            np.save(d / f"{k}.npy", getattr(self, k))

    @classmethod
    def load(cls, path) -> PackedSequences:
        d = Path(path)
        return cls(**{k: np.load(d / f"{k}.npy", mmap_mode="r") for k in _DTYPES})


def _pack(uid, item, token, organic, ratio, ts, max_len, collapse, n_dropped=0):
    """Pack arrays that are already sorted by (uid, chronological order)."""
    cols = [uid, item, token, organic, ratio, ts]
    if collapse and len(uid):  # keep the last event of each run of identical (user, item, token)
        last = np.ones(len(uid), dtype=bool)
        last[:-1] = (uid[1:] != uid[:-1]) | (item[1:] != item[:-1]) | (token[1:] != token[:-1])
        cols = [a[last] for a in cols]
    uids, first, counts = np.unique(cols[0], return_index=True, return_counts=True)
    if max_len is not None:
        pos = np.arange(len(cols[0])) - np.repeat(first, counts)
        keep = pos >= np.repeat(np.maximum(counts - max_len, 0), counts)
        cols = [a[keep] for a in cols]
        counts = np.minimum(counts, max_len)
    offsets = np.zeros(len(uids) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    return PackedSequences(
        uids=uids.astype(np.uint32),
        offsets=offsets,
        n_dropped_oov=n_dropped,
        **{f: a.astype(_DTYPES[f]) for f, a in zip(_FIELDS, cols[1:], strict=True)},
    )


def build_sequences(
    lf: pl.LazyFrame,
    vocab: ItemVocab,
    max_len: int | None = None,
    collapse_consecutive_dups: bool = False,
) -> PackedSequences:
    """Per-user chronological sequences (stable w.r.t. input order on timestamp ties).

    Events whose item is not in the vocab are dropped first (they would encode to the pad
    index 0); a warning is logged and the count is kept in `n_dropped_oov`. Collapsing happens
    before truncation, so `max_len` counts collapsed events. Null organic/ratio become 0.
    """
    df = (
        lf.select("uid", "item_id", "timestamp", "is_organic", "token", "ratio_bucket")
        .filter(pl.col("token").is_not_null())
        .collect()
    )
    uid = df["uid"].to_numpy()
    ts = df["timestamp"].to_numpy()
    order = np.lexsort((ts, uid))  # stable: ties keep original order
    item = vocab.encode(df["item_id"].to_numpy())[order]
    cols = [
        uid[order],
        item,
        df["token"].to_numpy()[order],
        df["is_organic"].fill_null(0).cast(pl.UInt8).to_numpy()[order],
        df["ratio_bucket"].fill_null(0).cast(pl.UInt8).to_numpy()[order],
        ts[order],
    ]
    known = item > 0
    n_dropped = int((~known).sum())
    if n_dropped:
        log.warning("dropped %d events with items outside the vocab", n_dropped)
        cols = [a[known] for a in cols]
    return _pack(*cols, max_len, collapse_consecutive_dups, n_dropped)


def build_sequences_sharded(
    lf_factory: Callable[[], pl.LazyFrame],
    vocab: ItemVocab,
    n_shards: int,
    **kw,
) -> PackedSequences:
    """Same result as `build_sequences`, built one `uid % n_shards` shard at a time."""
    shards = [
        build_sequences(lf_factory().filter(pl.col("uid") % n_shards == s), vocab, **kw)
        for s in range(n_shards)
    ]
    event_base = np.cumsum([0] + [len(s.items) for s in shards[:-1]])
    uids = np.concatenate([s.uids for s in shards])
    counts = np.concatenate([np.diff(s.offsets) for s in shards])
    starts = np.concatenate([s.offsets[:-1] + b for s, b in zip(shards, event_base, strict=True)])
    order = np.argsort(uids, kind="stable")
    counts, starts = counts[order], starts[order]
    offsets = np.zeros(len(uids) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    idx = np.arange(offsets[-1]) - np.repeat(offsets[:-1] - starts, counts)
    cat = {f: np.concatenate([getattr(s, f) for s in shards]) for f in _FIELDS}
    return PackedSequences(
        uids=uids[order],
        offsets=offsets,
        n_dropped_oov=sum(s.n_dropped_oov for s in shards),
        **{f: a[idx] for f, a in cat.items()},
    )
