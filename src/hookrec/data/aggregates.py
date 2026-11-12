"""Full-period interaction aggregates the baselines fit on (no `max_len_store` truncation).

For each training period (`train`, `val_train`) preprocessing stores, from the period's full,
untruncated events:

    ui_{period}.npz        CSR users x items of Listen+ (LISTEN_FULL) counts; row r is uid
                           `uids[r]` (ascending). Only users with >= 1 Listen+ event have a row:
                           they are exactly the period's warm users (official protocol).
    item_day_{period}.npz  sparse (item_idx, day, count) of Listen+ events, day =
                           (t_end - ts) // 86400 (0 = the last day before the period end).
    item_hour_{period}.npz sparse (item_idx, hour, count) of the Listen+ events of the last
                           HOURLY_DAYS (30) days only (age < 30 d, strictly), hour =
                           (t_end - ts) // 3600 < 720.
                           DecayPop decays these per hour and older events per day (R26).
    pop_lp_{period}.npy    i64[N+1] Listen+ events per item.
    pop_pos_{period}.npy   i64[N+1] positive (LISTEN_FULL or LIKE) events per item.

Listen+ is the Yambda paper's "Listen+" interaction (played_ratio_pct >= full_threshold).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

from hookrec.data.label import LIKE, LISTEN_FULL
from hookrec.data.vocab import ItemVocab

SECONDS_PER_DAY = 86400
SECONDS_PER_HOUR = 3600
HOURLY_DAYS = 30  # hourly bins cover [t_end - 30 d, t_end): hours 0 .. 719 (fit in uint16)
PERIODS = ("train", "val_train")


def index_of(sorted_uids: np.ndarray, uids: np.ndarray) -> np.ndarray:
    """Row index of each uid in the ascending array `sorted_uids`; -1 if absent."""
    uids = np.asarray(uids)
    if len(sorted_uids) == 0:
        return np.full(uids.shape, -1, dtype=np.int64)
    pos = np.minimum(np.searchsorted(sorted_uids, uids), len(sorted_uids) - 1)
    return np.where(np.asarray(sorted_uids)[pos] == uids, pos, -1).astype(np.int64)


@dataclass
class PeriodAggregates:
    uids: np.ndarray  # u32[U] ascending: users with >= 1 Listen+ event in the period
    ui: csr_matrix  # [U, N+1] Listen+ counts (int32)
    day_items: np.ndarray  # i32[K]
    day_index: np.ndarray  # i16[K] (t_end - ts) // 86400
    day_counts: np.ndarray  # i32[K]
    hour_items: np.ndarray  # i32[H] (last HOURLY_DAYS days only)
    hour_index: np.ndarray  # u16[H] (t_end - ts) // 3600, < 24 * HOURLY_DAYS
    hour_counts: np.ndarray  # i32[H]
    pop_lp: np.ndarray  # i64[N+1]
    pop_pos: np.ndarray  # i64[N+1]
    t_end: int  # exclusive end of the period (seconds)

    def index_of(self, uids: np.ndarray) -> np.ndarray:
        return index_of(self.uids, uids)

    def user_counts(self, uids: np.ndarray) -> np.ndarray:
        """Listen+ events of each uid over the full period (0 for users without a row)."""
        rows = self.index_of(uids)
        if len(self.uids) == 0:
            return np.zeros(rows.shape, dtype=np.int64)
        per_row = np.asarray(self.ui.sum(axis=1, dtype=np.int64)).ravel()
        return np.where(rows >= 0, per_row[np.maximum(rows, 0)], 0)

    def save(self, d: Path, period: str) -> None:
        d = Path(d)
        np.savez(
            d / f"ui_{period}.npz",
            uids=self.uids, indptr=self.ui.indptr.astype(np.int64),
            indices=self.ui.indices.astype(np.int32), data=self.ui.data.astype(np.int32),
            shape=np.array(self.ui.shape, dtype=np.int64), t_end=np.int64(self.t_end),
        )  # fmt: skip
        np.savez(
            d / f"item_day_{period}.npz",
            item_idx=self.day_items, day=self.day_index, count=self.day_counts,
        )  # fmt: skip
        np.savez(
            d / f"item_hour_{period}.npz",
            item_idx=self.hour_items, hour=self.hour_index, count=self.hour_counts,
        )  # fmt: skip
        np.save(d / f"pop_lp_{period}.npy", self.pop_lp)
        np.save(d / f"pop_pos_{period}.npy", self.pop_pos)

    @classmethod
    def load(cls, d: Path, period: str) -> PeriodAggregates:
        d = Path(d)
        with np.load(d / f"ui_{period}.npz") as z:
            ui = csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
            uids, t_end = z["uids"], int(z["t_end"])
        with np.load(d / f"item_day_{period}.npz") as z:
            day = (z["item_idx"], z["day"], z["count"])
        with np.load(d / f"item_hour_{period}.npz") as z:
            hour = (z["item_idx"], z["hour"], z["count"])
        return cls(
            uids=uids, ui=ui, day_items=day[0], day_index=day[1], day_counts=day[2],
            hour_items=hour[0], hour_index=hour[1], hour_counts=hour[2],
            pop_lp=np.load(d / f"pop_lp_{period}.npy"),
            pop_pos=np.load(d / f"pop_pos_{period}.npy"), t_end=t_end,
        )  # fmt: skip


def _binned(
    vocab: ItemVocab, lf: pl.LazyFrame, bin_expr: pl.Expr
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(item_idx i32, bin, count i32) of `lf`'s events per (item, bin), sorted by (item, bin).

    The bin and the count are typed in polars (the caller picks the narrowest dtype that fits)
    and sorted there: vocab indices ascend with raw ids, so (item_id, bin) order is (item_idx,
    bin) order and no numpy re-sort is needed. `vocab.encode` still makes one int64 temporary
    of the items (cast to int32 right away); bins and counts are never widened.
    """
    df = (
        lf.group_by("item_id", bin_expr.alias("bin"))
        .agg(pl.len().cast(pl.Int32).alias("n"))
        .sort("item_id", "bin")
        .collect(engine="streaming")
    )
    items = vocab.encode(df["item_id"].to_numpy()).astype(np.int32)
    bins, counts = df["bin"].to_numpy(), df["n"].to_numpy()
    del df
    ok = items > 0
    if not ok.all():  # the catalog covers every event item, so this is normally a no-op
        items, bins, counts = items[ok], bins[ok], counts[ok]
    return items, bins, counts


def build_period_aggregates(lf: pl.LazyFrame, vocab: ItemVocab, t_end: int) -> PeriodAggregates:
    """Aggregates of one period from its labeled events (uid, item_id, timestamp, token).

    Streaming group-bys over the period (their outputs are bounded by the number of distinct
    (user, item) / (item, day) / recent (item, hour) pairs and items, not by the event count).
    """
    n1 = vocab.n_items + 1
    ev = lf.select("uid", "item_id", "timestamp", "token")
    lp = ev.filter(pl.col("token") == LISTEN_FULL)

    ui_df = (
        lp.group_by("uid", "item_id")
        .agg(pl.len().alias("n"))
        .sort("uid", "item_id")
        .collect(engine="streaming")
    )
    items = vocab.encode(ui_df["item_id"].to_numpy())
    ok = items > 0
    uids, counts = np.unique(ui_df["uid"].to_numpy()[ok], return_counts=True)
    indptr = np.zeros(len(uids) + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    ui = csr_matrix(
        (ui_df["n"].to_numpy()[ok].astype(np.int32), items[ok].astype(np.int32), indptr),
        shape=(len(uids), n1),
    )
    ui.sort_indices()
    del ui_df, items

    age = pl.lit(t_end, dtype=pl.Int64) - pl.col("timestamp").cast(pl.Int64)
    d_items, d_day, d_n = _binned(vocab, lp, (age // SECONDS_PER_DAY).cast(pl.Int16))
    # age < HOURLY_DAYS days, strictly: an event exactly 30 d old is day 30 (daily part) only
    recent = lp.filter(pl.col("timestamp") > t_end - HOURLY_DAYS * SECONDS_PER_DAY)
    h_items, h_hour, h_n = _binned(vocab, recent, (age // SECONDS_PER_HOUR).cast(pl.UInt16))

    pop = (
        ev.filter(pl.col("token").is_in([LISTEN_FULL, LIKE]))
        .group_by("item_id")
        .agg(pl.len().alias("n"), (pl.col("token") == LISTEN_FULL).sum().alias("n_lp"))
        .collect(engine="streaming")
    )
    idx = vocab.encode(pop["item_id"].to_numpy())
    ok = idx > 0
    pop_lp = np.zeros(n1, dtype=np.int64)
    pop_pos = np.zeros(n1, dtype=np.int64)
    pop_pos[idx[ok]] = pop["n"].to_numpy()[ok]
    pop_lp[idx[ok]] = pop["n_lp"].to_numpy()[ok]
    return PeriodAggregates(
        uids=uids.astype(np.uint32),
        ui=ui,
        day_items=d_items,
        day_index=d_day,
        day_counts=d_n,
        hour_items=h_items,
        hour_index=h_hour,
        hour_counts=h_n,
        pop_lp=pop_lp,
        pop_pos=pop_pos,
        t_end=int(t_end),
    )
