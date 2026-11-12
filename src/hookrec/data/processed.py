"""Cached preprocessing pipeline: raw Yambda -> `ProcessedData` bundle on disk.

Stage A cleans and labels events once per uid shard into `_work/events_{s}.parquet`; every later
stage scans those files lazily, so peak memory stays bounded. `_work/` is deleted at the end.
"""

from __future__ import annotations

import json
import logging
import resource
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from hookrec.data.clean import clean_events, scan_events
from hookrec.data.features import ItemFeatures, build_item_features
from hookrec.data.label import label_events
from hookrec.data.sequences import PackedSequences, build_sequences, build_sequences_sharded
from hookrec.data.split import global_temporal_split
from hookrec.data.vocab import ItemVocab, build_vocab
from hookrec.eval.protocol import build_ground_truth, count_oov, split_by_history
from hookrec.utils.config import config_hash

log = logging.getLogger(__name__)

EVENT_COLS = ["uid", "item_idx", "is_organic", "is_positive", "token", "ts"]
# Columns every downstream stage reads from the cleaned+labeled work files.
WORK_COLS = ["uid", "item_id", "timestamp", "is_organic", "token", "ratio_bucket", "is_positive"]


@dataclass
class ProcessedData:
    cfg: dict
    vocab: ItemVocab
    features: ItemFeatures
    seqs_train: PackedSequences  # train period (test-time history)
    seqs_val_train: PackedSequences  # val_train period (history while tuning)
    truth_val: dict[int, np.ndarray]
    truth_test: dict[int, np.ndarray]
    truth_val_cold: dict[int, np.ndarray]
    truth_test_cold: dict[int, np.ndarray]
    meta: dict
    path: Path

    def val_events(self) -> pl.LazyFrame:
        return pl.scan_parquet(self.path / "val_events.parquet")

    def test_events(self) -> pl.LazyFrame:
        return pl.scan_parquet(self.path / "test_events.parquet")


def peak_rss_mb() -> float:
    """Peak resident set size of this process in MB (ru_maxrss: bytes on macOS, KB on Linux)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024**2 if sys.platform == "darwin" else 1024)


def save_truth(truth: dict[int, np.ndarray], path: Path) -> None:
    uids = np.array(sorted(truth), dtype=np.int64)
    lens = np.array([len(truth[int(u)]) for u in uids], dtype=np.int64)
    offsets = np.zeros(len(uids) + 1, dtype=np.int64)
    np.cumsum(lens, out=offsets[1:])
    items = (
        np.concatenate([truth[int(u)] for u in uids]) if len(uids) else np.zeros(0, dtype=np.int64)
    )
    np.savez(path, uids=uids, offsets=offsets, items=items.astype(np.int64))


def load_truth(path: Path) -> dict[int, np.ndarray]:
    with np.load(path) as z:
        uids, offsets, items = z["uids"], z["offsets"], z["items"]
    return {int(u): items[offsets[i] : offsets[i + 1]] for i, u in enumerate(uids)}


class _Stages:
    """Logs wall time and peak RSS after each stage."""

    def __init__(self) -> None:
        self.t0 = self.last = time.perf_counter()
        self.timings: dict[str, float] = {}

    def done(self, name: str) -> None:
        now = time.perf_counter()
        self.timings[name] = round(now - self.last, 2)
        self.last = now
        log.info("stage %-10s %7.1fs  peak RSS %.0f MB", name, self.timings[name], peak_rss_mb())


def _write_events(df_lf: pl.LazyFrame, vocab: ItemVocab, path: Path) -> int:
    df = df_lf.select(
        "uid",
        pl.col("item_id"),
        "is_organic",
        "is_positive",
        "token",
        pl.col("timestamp").alias("ts"),
    ).collect(engine="streaming")
    idx = vocab.encode(df["item_id"].to_numpy())
    assert (idx > 0).all(), "vocab must cover every event item (catalog = all events)"
    item_idx = pl.Series("item_idx", idx, dtype=pl.Int32)
    # deterministic output: total order (ties broken by item and token)
    df = df.with_columns(item_idx).select(EVENT_COLS).sort("uid", "ts", "item_idx", "token")
    df.write_parquet(path)
    return df.height


def _counts(lf: pl.LazyFrame) -> tuple[int, int]:
    counts = lf.select(pl.col("uid").n_unique(), pl.len()).collect(engine="streaming")
    n_users, n_events = counts.row(0)
    return int(n_users), int(n_events)


def _build(raw_dir: Path, cfg: dict, out: Path) -> None:
    src = Path(raw_dir) / "flat" / cfg["size"] / "multi_event.parquet"
    # `{size}` placeholder: K0 writes the filtered `embeddings_{size}.parquet`
    emb_name = cfg.get("embeddings_file", "embeddings.parquet").format(size=cfg["size"])
    emb_path = Path(raw_dir) / emb_name
    artist_path = Path(raw_dir) / "artist_item_mapping.parquet"
    missing = [str(p) for p in (src, emb_path, artist_path) if not p.exists()]
    if missing:
        raise FileNotFoundError("preprocess inputs not found: " + ", ".join(missing))

    st = _Stages()
    work = out / "_work"
    work.mkdir(parents=True)
    n_shards = int(cfg.get("n_shards", 1))

    # Stage A: clean + label once per uid shard.
    for s in range(n_shards):
        lf = scan_events(src)
        if n_shards > 1:
            lf = lf.filter(pl.col("uid") % n_shards == s)
        label_events(clean_events(lf), cfg["full_threshold"]).select(WORK_COLS).collect(
            engine="streaming"
        ).write_parquet(work / f"events_{s}.parquet")
    st.done("clean")
    all_lf = pl.scan_parquet(work / "events_*.parquet")

    # Split + vocab.
    sp = cfg["split"]
    split = global_temporal_split(
        all_lf,
        train_days=sp["train_days"],
        gap_seconds=sp["gap_seconds"],
        test_days=sp["test_days"],
        test_timestamp=sp.get("test_timestamp"),
    )
    extra = None
    emb_col = cfg.get("emb_column", "normalized_embed")
    if cfg.get("include_embedding_only_items", False):
        emb_ids = pl.scan_parquet(emb_path).select("item_id").collect(engine="streaming")
        extra = emb_ids["item_id"].to_numpy()
    vocab = build_vocab(all_lf, split.train, cfg["min_count"], extra_item_ids=extra)
    st.done("vocab")

    # Features: embeddings are read lazily, restricted to catalog items.
    ids = pl.Series("ids", vocab.raw_ids[1:])
    emb_df = (
        pl.scan_parquet(emb_path)
        .filter(pl.col("item_id").is_in(ids.implode()))
        .select("item_id", emb_col)
        .collect(engine="streaming")
    )
    artist_df = pl.read_parquet(artist_path)
    features = build_item_features(
        vocab, emb_df, artist_df, split.train, cfg["pca_dim"], emb_column=emb_col
    )
    del emb_df
    features.save(out / "features.npz")
    vocab.save(out / "vocab.npz")
    st.done("features")

    # Sequences.
    kw = {
        "max_len": cfg["max_len_store"],
        "collapse_consecutive_dups": cfg["collapse_consecutive_dups"],
    }

    def seqs(lf: pl.LazyFrame, ts_end: int) -> PackedSequences:
        if n_shards > 1:  # shard files already hold one uid shard: only the time bound is left
            return build_sequences_sharded(
                None, vocab, n_shards,
                shard_factory=lambda s: pl.scan_parquet(work / f"events_{s}.parquet").filter(
                    pl.col("timestamp") < ts_end
                ),
                **kw,
            )  # fmt: skip
        return build_sequences(lf, vocab, **kw)

    seqs_train = seqs(split.train, split.t_train_end)
    seqs_train.save(out / "seqs_train")
    seqs_val_train = seqs(split.val_train, split.t_val_train_end)
    seqs_val_train.save(out / "seqs_val_train")
    st.done("sequences")

    # Ground truth (R10): warm users in the main dicts, cold users separately.
    truth_test_all = build_ground_truth(split.test, vocab)
    truth_val_all = build_ground_truth(split.val, vocab)
    truth_test, truth_test_cold = split_by_history(truth_test_all, seqs_train.uids)
    truth_val, truth_val_cold = split_by_history(truth_val_all, seqs_val_train.uids)
    for name, t in (
        ("truth_test", truth_test),
        ("truth_test_cold", truth_test_cold),
        ("truth_val", truth_val),
        ("truth_val_cold", truth_val_cold),
    ):
        save_truth(t, out / f"{name}.npz")

    # R3: share of distinct test-truth items (all users) with no train event of any kind.
    train_items = split.train.select(pl.col("item_id").unique()).collect(engine="streaming")
    train_idx = vocab.encode(train_items["item_id"].to_numpy())
    seen = np.zeros(vocab.n_items + 1, dtype=bool)
    seen[train_idx[train_idx > 0]] = True
    test_items = (
        np.unique(np.concatenate(list(truth_test_all.values())))
        if truth_test_all
        else np.zeros(0, dtype=np.int64)
    )
    unknown = (test_items > vocab.n_items) | ~seen[np.minimum(test_items, vocab.n_items)]
    share_unknown = float(unknown.mean()) if len(test_items) else 0.0
    st.done("truth")

    # R5: labeled, vocab-encoded val/test events.
    _write_events(split.val, vocab, out / "val_events.parquet")
    _write_events(split.test, vocab, out / "test_events.parquet")
    st.done("events")

    n_users, n_events = {}, {}
    for name, lf in (
        ("train", split.train),
        ("val_train", split.val_train),
        ("val", split.val),
        ("test", split.test),
    ):
        n_users[name], n_events[name] = _counts(lf)
    meta = {
        "t_val_train_end": split.t_val_train_end,
        "t_val_start": split.t_val_start,
        "t_train_end": split.t_train_end,
        "t_test_start": split.t_test_start,
        "t_test_end": split.t_test_end,
        "n_users": n_users,
        "n_events": n_events,
        "n_items": vocab.n_items,
        "n_id": vocab.n_id,
        "n_cold_test_users": len(truth_test_cold),
        "n_cold_val_users": len(truth_val_cold),
        "share_unknown_test_items": share_unknown,
        "n_oov_test_items": count_oov(truth_test_all, vocab.n_items),
        "n_dropped_oov": {
            "train": seqs_train.n_dropped_oov,
            "val_train": seqs_val_train.n_dropped_oov,
        },
        "stage_seconds": st.timings,
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "wall_seconds": round(time.perf_counter() - st.t0, 2),
        "config_hash": config_hash(cfg),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    shutil.rmtree(work)


def preprocess(raw_dir: Path, cfg: dict, out_root: Path) -> Path:
    """Build (or reuse) the processed bundle for `cfg`; returns its directory."""
    out = Path(out_root) / f"{cfg['size']}-{config_hash(cfg)}"
    if (out / "_DONE").exists():
        log.info("cache hit: %s", out)
        return out
    if out.exists():  # killed earlier run
        log.warning("removing incomplete run at %s", out)
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True))
    _build(Path(raw_dir), cfg, out)
    (out / "_DONE").write_text("")
    return out


def load_processed(path: Path) -> ProcessedData:
    d = Path(path)
    if not (d / "_DONE").exists():
        raise FileNotFoundError(f"{d} has no _DONE marker: preprocessing incomplete or missing")
    return ProcessedData(
        cfg=yaml.safe_load((d / "config.yaml").read_text()),
        vocab=ItemVocab.load(d / "vocab.npz"),
        features=ItemFeatures.load(d / "features.npz"),
        seqs_train=PackedSequences.load(d / "seqs_train"),
        seqs_val_train=PackedSequences.load(d / "seqs_val_train"),
        truth_val=load_truth(d / "truth_val.npz"),
        truth_test=load_truth(d / "truth_test.npz"),
        truth_val_cold=load_truth(d / "truth_val_cold.npz"),
        truth_test_cold=load_truth(d / "truth_test_cold.npz"),
        meta=json.loads((d / "meta.json").read_text()),
        path=d,
    )
