"""Download one Yambda size into a lean raw dir (the K0 stage).

python scripts/run_download.py --size 50m --out DIR [--mirror DIR] [--keep-full-embeddings]

Writes the Yambda layout under --out: `flat/{size}/multi_event.parquet`,
`artist_item_mapping.parquet`, `album_item_mapping.parquet` and `embeddings_{size}.parquet` (only
the subset's items, float16 `embed` column), then the marker `_HOOKREC_RAW_{size}`. The full
`embeddings.parquet` (all items) is deleted afterwards unless --keep-full-embeddings.

--mirror is an existing copy in the Yambda layout (e.g. a public Kaggle dataset): files found
there are copied (events, mappings) or read (full embeddings) instead of downloaded from HF.
Idempotent: finished files are kept, and a dir that already has the marker is not re-fetched.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import polars as pl

from hookrec.data import download as dl

log = logging.getLogger("run_download")

FULL_EMB = "embeddings.parquet"
MAPPINGS = ("artist_item_mapping.parquet", "album_item_mapping.parquet")


def marker_name(size: str) -> str:
    return f"_HOOKREC_RAW_{size}"


def _copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dest)


def _cleanup(out: Path, keep_full: bool) -> None:
    if not keep_full and (out / FULL_EMB).exists():
        (out / FULL_EMB).unlink()
        log.info("removed the full %s (the filtered file is all later stages need)", FULL_EMB)
    shutil.rmtree(out / ".cache", ignore_errors=True)  # hf_hub_download metadata
    for p in out.rglob("*.tmp"):
        p.unlink()


def build_raw(size: str, out: Path, mirror: Path | None = None, keep_full: bool = False) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    marker = out / marker_name(size)
    if marker.exists():
        log.info("%s exists: raw dir already complete", marker)
        _cleanup(out, keep_full)
        return out

    events_name = f"flat/{size}/multi_event.parquet"
    events, filtered = out / events_name, out / f"embeddings_{size}.parquet"
    if mirror is not None:
        for name in (events_name, *MAPPINGS):
            if not (out / name).exists() and (Path(mirror) / name).exists():
                log.info("copying %s from mirror %s", name, mirror)
                _copy(Path(mirror) / name, out / name)
    emb_src = out / FULL_EMB
    if mirror is not None and (Path(mirror) / FULL_EMB).exists():
        emb_src = Path(mirror) / FULL_EMB
    needed = [events, out / MAPPINGS[0]] + ([] if filtered.exists() else [emb_src])
    if not all(p.exists() for p in needed):
        log.info("downloading Yambda %s from Hugging Face (%s) into %s", size, dl.REPO_ID, out)
        dl.download_yambda(size, out)  # skips files already present

    if not filtered.exists():
        ids = dl.subset_item_ids(events)
        tmp = filtered.with_name(filtered.name + ".tmp")
        n = dl.filter_embeddings(emb_src, ids, tmp)
        os.replace(tmp, filtered)
        log.info("embeddings: %d of %d subset items kept (float16) -> %s", n, len(ids), filtered)
    n_emb = pl.scan_parquet(filtered).select(pl.len()).collect().item()
    n_events = pl.scan_parquet(events).select(pl.len()).collect().item()
    meta = {"size": size, "n_events": n_events, "n_embeddings": n_emb, "mirror": str(mirror or "")}
    marker.write_text(json.dumps(meta, indent=2))
    _cleanup(out, keep_full)
    log.info("raw dir complete: %s", json.dumps(meta))
    return out


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", choices=("50m", "500m", "5b"), default="50m")
    ap.add_argument("--out", type=Path, required=True, help="raw dir (Yambda layout)")
    ap.add_argument("--mirror", type=Path, help="existing Yambda copy (read-only) to use first")
    ap.add_argument("--keep-full-embeddings", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    return build_raw(args.size, args.out, args.mirror, args.keep_full_embeddings)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("run_download failed")
        sys.exit(1)
