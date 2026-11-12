"""Yambda download from Hugging Face and streamed embedding filtering.

HF file tree confirmed via list_repo_files("yandex/yambda", repo_type="dataset"):
  root: embeddings.parquet, artist_item_mapping.parquet
  {flat,sequential}/50m/{listens,likes,dislikes,unlikes,undislikes,multi_event}.parquet
(500m and 5b follow the same pattern.) The plan's defaults hold.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
from huggingface_hub import hf_hub_download

REPO_ID = "yandex/yambda"
ROOT_FILES = ("embeddings.parquet", "artist_item_mapping.parquet", "album_item_mapping.parquet")


def download_yambda(
    size: Literal["50m", "500m", "5b"],
    out_dir: Path,
    fmt: Literal["flat", "sequential"] = "flat",
    files: tuple[str, ...] = ("multi_event.parquet",),
    root_files: tuple[str, ...] = ROOT_FILES,
) -> dict[str, Path]:
    """Download requested files (idempotent: existing files are skipped).

    `root_files` are the repo-root files (embeddings, mappings) to fetch alongside.
    Returns {relative_name: local path}, keyed e.g. "flat/50m/multi_event.parquet".
    """
    out_dir = Path(out_dir)
    names = [f"{fmt}/{size}/{f}" for f in files] + list(root_files)
    paths: dict[str, Path] = {}
    for name in names:
        dest = out_dir / name
        if not dest.exists():
            dest = Path(
                hf_hub_download(
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    filename=name,
                    local_dir=str(out_dir),
                )
            )
        paths[name] = dest
    return paths


def subset_item_ids(events_path: Path) -> np.ndarray:
    """Sorted unique item_ids of an events parquet (lazy scan)."""
    df = pl.scan_parquet(events_path).select(pl.col("item_id").unique().sort()).collect()
    return df["item_id"].to_numpy()


def filter_embeddings(
    emb_path: Path,
    item_ids: np.ndarray,
    out_path: Path,
    column: str = "normalized_embed",
    dtype=np.float16,
) -> int:
    """Write parquet (item_id, embed) for only `item_ids`, streaming; returns row count.

    `embed` is stored as a fixed-size Array of `dtype` (float16 by default).
    """
    elem = pl.Float16 if np.dtype(dtype) == np.float16 else pl.Float32
    ids = pl.Series("ids", np.asarray(item_ids))
    src = pl.scan_parquet(emb_path)
    # Cast inside the lazy plan so only the narrow matched rows are ever materialized.
    # D comes from the schema (fixed-size column) or one cheap head(1) row (list column).
    col_type = src.collect_schema()[column]
    if isinstance(col_type, pl.Array):
        dim = col_type.size
    else:
        first = src.select(pl.col(column).list.len()).head(1).collect()
        dim = int(first.item()) if first.height else None
    embed = pl.col(column).cast(pl.Array(elem, dim) if dim else pl.List(elem)).alias("embed")
    df = (
        src.filter(pl.col("item_id").is_in(ids.implode()))
        .select(pl.col("item_id"), embed)
        .collect(engine="streaming")
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    return df.height
