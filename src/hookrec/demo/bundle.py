"""Self-contained demo bundle: everything the engine needs, with no `ProcessedData`.

Layout of `out_dir`:
    meta.json    both models' cfg.model sections and `data.input_filter`s, session gap, sizes,
                 format version
    arrays.npz   item subset (raw ids, popularity, artist ids, flags), the compact feature rows the
                 HookSeq encoder needs, fp16 item vectors of both models, sample user windows
    hookseq.pt / sasrec.pt   state_dicts over the compact subset

Compact item space: index 0 = pad, 1..M = the subset items in ascending raw-id order, M + 1 =
"unknown" (a session item outside the subset). The unknown row has no ID embedding, no audio and
no artist, so HookSeq encodes it as its learned missing-audio vector, and SASRec as a zero
embedding (it has no unknown row of its own). It is never offered as a recommendation.
Every item of every sample window is in the subset, so sample sessions are encoded exactly.
Sample windows come from HookSeq's sequence set (its `data.input_filter`). A model trained on
Listen+ input only (`input_filter: listen_plus`, the `sasrec` config) sees only the session's
listen_full events (see `DemoEngine`).

Subset = top `n_items` by train popularity, plus the items of the sampled users' windows, plus up
to 500 ID-less items with audio (the new-track-audience tab). Artist ids in the bundle are the
processed data's compact artist indices (0 = unknown), not raw Yambda artist ids.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import torch

from hookrec.train.datasets import gap_buckets, session_gap_from

if TYPE_CHECKING:
    from hookrec.data.processed import ProcessedData

log = logging.getLogger(__name__)

FORMAT_VERSION = 1
SIZE_LIMIT_BYTES = 200 * 1024 * 1024
N_IDLESS = 500
MIN_WINDOW = 5  # sample users have at least this many events when enough such users exist


@dataclass
class Bundle:
    meta: dict
    arrays: dict[str, np.ndarray]
    hookseq_state: dict[str, torch.Tensor]
    sasrec_state: dict[str, torch.Tensor]

    @property
    def n_subset(self) -> int:
        return int(self.meta["n_subset"])


def _load_ckpt(path: Path, expect: str) -> tuple[dict, dict]:
    """(state_dict, full training cfg) of a `model_final.pt` (or the directory holding it)."""
    path = Path(path)
    if path.is_dir():
        path = path / "model_final.pt"
    if not path.exists():
        raise FileNotFoundError(f"{expect} checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    name = payload["cfg"]["model"]["name"]
    if name != expect:
        raise ValueError(f"{path} holds a {name!r} model, expected {expect!r}")
    return payload["model"], payload["cfg"]


def _pick_subset(data: ProcessedData, windows_items: np.ndarray, n_items: int) -> np.ndarray:
    """Sorted vocab indices of the subset (see module docstring)."""
    pop = data.features.pop_train
    top = np.argsort(-pop[1:], kind="stable")[:n_items] + 1  # ties -> lower index
    idless = np.flatnonzero((data.vocab.id_idx == 0) & data.features.has_audio)
    idless = idless[idless > 0]
    idless = idless[np.argsort(-pop[idless], kind="stable")][:N_IDLESS]
    return np.unique(np.concatenate([top, windows_items, idless]))


def input_filter_of(cfg: dict) -> str:
    """A training cfg's `data.input_filter` ("all" when unset; old bundles have no key)."""
    return (cfg.get("data") or {}).get("input_filter", "all")


def _sample_windows(
    data: ProcessedData, n_users: int, max_len: int, seed: int, input_filter: str = "all"
) -> dict:
    """Last `max_len` events of `n_users` random train users, vectorised, in vocab indices."""
    s = data.seqs("train", input_filter)
    offsets = np.asarray(s.offsets)
    lens = np.diff(offsets)
    eligible = np.flatnonzero(lens >= MIN_WINDOW)
    if len(eligible) == 0:
        eligible = np.flatnonzero(lens >= 1)
    if len(eligible) == 0:
        raise ValueError("no user with training history to sample")
    rows = np.sort(
        np.random.RandomState(seed).choice(eligible, min(n_users, len(eligible)), replace=False)
    )
    n = np.minimum(lens[rows], max_len)
    w_off = np.zeros(len(rows) + 1, dtype=np.int64)
    np.cumsum(n, out=w_off[1:])
    flat = np.repeat(offsets[rows + 1] - n - w_off[:-1], n) + np.arange(w_off[-1])
    ts_all = np.asarray(s.ts)
    ts = ts_all[flat].astype(np.int64)
    first_of_user = flat == np.repeat(offsets[rows], n)
    prev = ts_all[np.maximum(flat - 1, 0)].astype(np.int64)
    # the previous event may lie before the window: keep its real gap bucket
    gb = np.where(first_of_user, 0, gap_buckets(ts - prev))
    return {
        "uids": np.asarray(s.uids)[rows].astype(np.int64),
        "offsets": w_off,
        "items_vocab": np.asarray(s.items)[flat].astype(np.int64),
        "tokens": np.asarray(s.tokens)[flat].astype(np.uint8),
        "organic": np.asarray(s.organic)[flat].astype(np.uint8),
        "ratio": np.asarray(s.ratio)[flat].astype(np.uint8),
        "ts": ts.astype(np.uint32),
        "gap_bucket": gb.astype(np.uint8),
    }


def build_models(bundle: Bundle):
    """(HookSeq, SASRec) over the compact subset, in eval mode with frozen weights."""
    from hookrec.models.hookseq import HookSeq
    from hookrec.models.sasrec import SASRec

    a, meta = bundle.arrays, bundle.meta
    m = bundle.n_subset
    # HookSeq reads only these four attributes (a SimpleNamespace avoids importing
    # `hookrec.data.features`, and with it polars and scikit-learn, on the Space)
    feats = SimpleNamespace(
        audio=a["audio"],
        has_audio=a["has_audio_full"],
        artist_idx=a["artist_sub"],
        n_artists=int(meta["n_artists_sub"]),
    )
    # every parameter and persistent buffer is in the state_dict, so loading is strict (the
    # id/audio/artist buffers are non-persistent and rebuilt above from the bundle's arrays)
    hs = HookSeq(
        n_items=m + 1,
        features=feats,
        n_id=int(meta["n_id_sub"]),
        id_idx=a["id_idx_sub"],
        **meta["hookseq_model"],
    )
    hs.load_state_dict(bundle.hookseq_state)
    sas = SASRec(n_items=m + 1, **meta["sasrec_model"])
    sas.load_state_dict(bundle.sasrec_state)
    for model in (hs, sas):
        model.eval().requires_grad_(False)
    return hs, sas


def export_demo_bundle(
    data: ProcessedData,
    hookseq_ckpt: Path,
    sasrec_ckpt: Path,
    out_dir: Path,
    n_users: int = 200,
    n_items: int = 50_000,
    seed: int = 0,
) -> Path:
    """Write the demo bundle for two trained checkpoints (a `model_final.pt` or its directory)."""
    hs_state, hs_full_cfg = _load_ckpt(hookseq_ckpt, "hookseq")
    sas_state, sas_full_cfg = _load_ckpt(sasrec_ckpt, "sasrec")
    hs_cfg = {k: x for k, x in hs_full_cfg["model"].items() if k != "name"}
    sas_cfg = {k: x for k, x in sas_full_cfg["model"].items() if k != "name"}
    v, f = data.vocab, data.features
    n_id_rows, n_sas_rows = (
        hs_state["item_emb.weight"].shape[0],
        sas_state["item_emb.weight"].shape[0],
    )
    if n_id_rows != v.n_id + 1:
        raise ValueError(
            f"hookseq checkpoint has {n_id_rows - 1} ID rows but the processed data has "
            f"n_id={v.n_id}: wrong checkpoint or processed dir"
        )
    if n_sas_rows != v.n_items + 1:
        raise ValueError(
            f"sasrec checkpoint has {n_sas_rows - 1} item rows but the processed data has "
            f"n_items={v.n_items}: wrong checkpoint or processed dir"
        )
    max_len = int(hs_cfg["max_len"])
    win = _sample_windows(data, n_users, max_len, seed, input_filter_of(hs_full_cfg))
    sub = _pick_subset(data, win["items_vocab"], n_items)  # vocab indices, ascending
    m = len(sub)

    def rows(x: np.ndarray) -> np.ndarray:
        """Compact rows of a per-vocab-item array: 0 = pad, 1..m = subset, m + 1 = unknown."""
        out = np.zeros((m + 2, *x.shape[1:]), dtype=x.dtype)
        out[1 : m + 1] = x[sub]
        return out

    id_full = rows(v.id_idx).astype(np.int64)
    has_id = id_full > 0
    id_rows = id_full[has_id]  # original ID rows, ascending
    id_sub = np.zeros(m + 2, dtype=np.int32)
    id_sub[has_id] = np.arange(1, len(id_rows) + 1)
    art_full = rows(f.artist_idx).astype(np.int64)
    art_uniq = np.unique(art_full[art_full > 0])
    art_sub = np.where(art_full > 0, np.searchsorted(art_uniq, art_full) + 1, 0).astype(np.int32)

    hs_state = dict(hs_state)
    hs_state["item_emb.weight"] = hs_state["item_emb.weight"][
        torch.from_numpy(np.concatenate([[0], id_rows]))
    ]
    if "artist_emb.weight" in hs_state:
        if hs_state["artist_emb.weight"].shape[0] != f.n_artists + 1:
            raise ValueError("hookseq artist table does not match the processed data's artists")
        hs_state["artist_emb.weight"] = hs_state["artist_emb.weight"][
            torch.from_numpy(np.concatenate([[0], art_uniq]))
        ]
    sas_w = sas_state["item_emb.weight"]
    sas_state = dict(sas_state)
    sas_state["item_emb.weight"] = torch.cat(
        [sas_w[torch.from_numpy(np.concatenate([[0], sub]))], torch.zeros(1, sas_w.shape[1])]
    )

    arrays = {
        "item_raw_ids": v.raw_ids[sub].astype(np.int64),
        "item_pop": f.pop_train[sub].astype(np.int64),
        "item_artist": f.artist_idx[sub].astype(np.int64),
        "has_id": v.id_idx[sub] > 0,
        "has_audio": f.has_audio[sub],
        "audio": rows(f.audio),
        "has_audio_full": rows(f.has_audio),
        "id_idx_sub": id_sub,
        "artist_sub": art_sub,
        "w_uids": win["uids"],
        "w_offsets": win["offsets"],
        "w_items": v.raw_ids[win["items_vocab"]].astype(np.int64),
        "w_tokens": win["tokens"],
        "w_organic": win["organic"],
        "w_ratio": win["ratio"],
        "w_ts": win["ts"],
        "w_gap_bucket": win["gap_bucket"],
    }
    meta = {
        "format_version": FORMAT_VERSION,
        "n_subset": m,
        "unknown_index": m + 1,
        "n_id_sub": len(id_rows),
        "n_artists_sub": len(art_uniq),
        "max_len": max_len,
        "session_gap": session_gap_from(hs_full_cfg),
        "hookseq_model": hs_cfg,
        "sasrec_model": sas_cfg,
        "input_filter": {
            "hookseq": input_filter_of(hs_full_cfg),
            "sasrec": input_filter_of(sas_full_cfg),
        },
        "n_sample_users": len(win["uids"]),
        "processed_dir": data.path.name,
    }
    hs, sas = build_models(Bundle(meta, arrays, hs_state, sas_state))
    with torch.no_grad():
        vecs = {"vec_hookseq": hs.all_item_repr(), "vec_sasrec": sas.all_item_repr()}
    for name, t in vecs.items():
        t16 = t.float().numpy().astype(np.float16)
        if not np.isfinite(t16).all():
            raise ValueError(f"non-finite values in {name} (overflow in fp16 or NaN weights)")
        arrays[name] = t16

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "arrays.npz", **arrays)
    torch.save(hs_state, out / "hookseq.pt")
    torch.save(sas_state, out / "sasrec.pt")
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    total = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    log.info("demo bundle: %d items, %d sample users, %.1f MB", m, len(win["uids"]), total / 2**20)
    if total >= SIZE_LIMIT_BYTES:
        for p in out.iterdir():  # leave no oversized bundle behind
            if p.name in ("meta.json", "arrays.npz", "hookseq.pt", "sasrec.pt"):
                p.unlink()
        raise RuntimeError(
            f"bundle is {total / 2**20:.0f} MB, over the {SIZE_LIMIT_BYTES // 2**20} MB limit: "
            "lower --n-items"
        )
    return out


def load_bundle(bundle_dir: Path) -> Bundle:
    d = Path(bundle_dir)
    if not (d / "meta.json").exists():
        raise FileNotFoundError(f"{d} is not a demo bundle (no meta.json)")
    meta = json.loads((d / "meta.json").read_text())
    if meta.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported bundle format {meta.get('format_version')}")
    with np.load(d / "arrays.npz") as z:
        arrays = {k: z[k] for k in z.files}
    return Bundle(
        meta,
        arrays,
        torch.load(d / "hookseq.pt", map_location="cpu", weights_only=True),
        torch.load(d / "sasrec.pt", map_location="cpu", weights_only=True),
    )
