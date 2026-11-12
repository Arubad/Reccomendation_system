"""Demo engine: loads a bundle on CPU and scores sessions, on the GPU when one is available.

Nothing touches CUDA at import or construction. The two scoring functions are wrapped with
`spaces.GPU` only when `spaces` is installed (ZeroGPU); inside them the model and tensors move
to CUDA if it is available and the result comes back as numpy. Without CUDA they run on CPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

# isort: off
from hookrec.demo.gpu import _maybe_gpu  # first: `spaces` must be imported before torch

import numpy as np
import torch

# isort: on

from hookrec.data.label import DISLIKE, LIKE, LISTEN_FULL, LISTEN_SKIP
from hookrec.demo.bundle import build_models, load_bundle
from hookrec.train.datasets import gap_buckets
from hookrec.train.scoring import score_topk

EVENT_TOKEN = {
    "listen_full": LISTEN_FULL,
    "listen_skip": LISTEN_SKIP,
    "like": LIKE,
    "dislike": DISLIKE,
}
TOKEN_EVENT = {v: k for k, v in EVENT_TOKEN.items()}
EVENT_RATIO = {"listen_full": 5, "listen_skip": 1, "like": 0, "dislike": 0}  # 0 = n/a
SESSION_SPACING_S = 60  # synthesised seconds between the events of a typed-in session


def _device(device: str | None) -> torch.device:
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def _encode_impl(model, batch: dict, device: str | None = None) -> np.ndarray:
    """User states [B, d] of a batch, as numpy."""
    dev = _device(device)
    on_gpu = dev.type != "cpu"
    if on_gpu:
        model.to(dev)
    try:
        with torch.no_grad():
            return model.user_state({k: v.to(dev) for k, v in batch.items()}).float().cpu().numpy()
    finally:
        if on_gpu:
            model.to("cpu")


def _topk_impl(model, batch: dict, item_vecs, k: int, device: str | None = None):
    """(scores, item indices) numpy [B, k]: top-k items by dot product, pad excluded."""
    dev = _device(device)
    on_gpu = dev.type != "cpu"
    if on_gpu:
        model.to(dev)
    try:
        with torch.no_grad():
            users = model.user_state({key: v.to(dev) for key, v in batch.items()})
            vals, idx = score_topk(users, item_vecs.to(dev), k)
        return vals.cpu().numpy(), idx.cpu().numpy()
    finally:
        if on_gpu:
            model.to("cpu")


_encode = _maybe_gpu(_encode_impl)
_topk = _maybe_gpu(_topk_impl)


class DemoEngine:
    def __init__(self, bundle_dir: Path | str, device: str | None = None):
        b = load_bundle(Path(bundle_dir))
        self.meta, a = b.meta, b.arrays
        self.device = device  # None: use CUDA inside GPU calls when available
        self.max_len = int(self.meta["max_len"])
        self.session_gap = self.meta["session_gap"]
        self.n_subset = int(self.meta["n_subset"])
        self.unknown_index = self.n_subset + 1
        self.raw_ids = a["item_raw_ids"]  # compact index i <-> raw_ids[i - 1]
        self.pop, self.artist = a["item_pop"], a["item_artist"]
        self.has_id, self.has_audio = a["has_id"], a["has_audio"]
        self.models = dict(zip(("hookseq", "sasrec"), build_models(b), strict=True))
        # rows 0..M: pad + subset. The unknown row is never a candidate.
        self._vecs = {
            "hookseq": torch.from_numpy(a["vec_hookseq"][: self.n_subset + 1].astype(np.float32)),
            "sasrec": torch.from_numpy(a["vec_sasrec"][: self.n_subset + 1].astype(np.float32)),
        }
        self._w = {k: a[k] for k in a if k.startswith("w_")}
        self.idless_audio_items = self.raw_ids[~self.has_id & self.has_audio].tolist()
        self._states: np.ndarray | None = None

    # ---- sessions -------------------------------------------------------------------------
    def to_compact(self, raw: np.ndarray) -> np.ndarray:
        """Raw item ids -> compact indices; ids outside the subset map to the unknown index."""
        raw = np.asarray(raw, dtype=np.int64)
        pos = np.minimum(np.searchsorted(self.raw_ids, raw), len(self.raw_ids) - 1)
        return np.where(self.raw_ids[pos] == raw, pos + 1, self.unknown_index)

    def _session_batch(self, session) -> dict[str, torch.Tensor]:
        if len(session) == 0:
            raise ValueError("the session is empty: add at least one event")
        session = list(session)[-self.max_len :]
        events = [e for _, e, _ in session]
        bad = sorted({e for e in events if e not in EVENT_TOKEN})
        if bad:
            raise ValueError(f"unknown event {bad[0]!r}; use one of {sorted(EVENT_TOKEN)}")
        n = len(session)
        gap = np.zeros(n, dtype=np.int64)
        gap[1:] = gap_buckets(np.full(n - 1, SESSION_SPACING_S), self.session_gap)
        cols = {
            "items": self.to_compact([int(i) for i, _, _ in session]),
            "tokens": [EVENT_TOKEN[e] for e in events],
            "organic": [int(bool(o)) for _, _, o in session],
            "ratio": [EVENT_RATIO[e] for e in events],
            "gap_bucket": gap,
        }
        return {k: torch.as_tensor(np.asarray(v, dtype=np.int64))[None] for k, v in cols.items()}

    def recommend(
        self,
        session: list[tuple[int, str, bool]],
        model: Literal["hookseq", "sasrec"],
        k: int = 10,
    ) -> list[dict]:
        """Top-k next tracks for a session; nothing is excluded except pad and the unknown row.

        Session items outside the bundle's item subset are encoded as the unknown item.
        """
        if model not in self.models:
            raise ValueError(f"model must be 'hookseq' or 'sasrec', got {model!r}")
        k = max(1, min(int(k), self.n_subset))
        batch = self._session_batch(session)
        vals, idx = _topk(self.models[model], batch, self._vecs[model], k, self.device)
        return [
            {
                "item_id": int(self.raw_ids[i - 1]),
                "artist_id": int(self.artist[i - 1]),
                "popularity": int(self.pop[i - 1]),
                "score": float(s),
            }
            for s, i in zip(vals[0], idx[0], strict=True)
        ]

    # ---- sample users ---------------------------------------------------------------------
    @property
    def n_sample_users(self) -> int:
        return len(self._w["w_uids"])

    def _window(self, i: int) -> slice:
        if not 0 <= i < self.n_sample_users:
            raise IndexError(f"sample user {i} out of range [0, {self.n_sample_users})")
        o = self._w["w_offsets"]
        return slice(int(o[i]), int(o[i + 1]))

    def sample_user_session(self, i: int, n_last: int | None = 20) -> list[tuple[int, str, bool]]:
        """The last `n_last` events (all if None) of sample user `i` as (item, event, organic)."""
        sl = self._window(i)
        w = self._w
        rows = list(zip(w["w_items"][sl], w["w_tokens"][sl], w["w_organic"][sl], strict=True))
        rows = rows[-n_last:] if n_last else rows
        return [(int(it), TOKEN_EVENT[int(t)], bool(o)) for it, t, o in rows]

    def _sample_states(self) -> np.ndarray:
        """HookSeq states [U, d] of every sample user's stored window (cached)."""
        if self._states is None:
            w, off = self._w, self._w["w_offsets"]
            lens = np.diff(off)
            pos = np.arange(self.max_len)[None, :] - (self.max_len - lens)[:, None]  # < 0: pad
            valid = pos >= 0
            src = np.where(valid, off[1:, None] - lens[:, None] + pos, 0)

            def pad(a):
                return torch.from_numpy(np.where(valid, a[src], 0).astype(np.int64))

            batch = {
                "items": torch.from_numpy(np.where(valid, self.to_compact(w["w_items"])[src], 0)),
                "tokens": pad(w["w_tokens"]),
                "organic": pad(w["w_organic"]),
                "ratio": pad(w["w_ratio"]),
                "gap_bucket": pad(w["w_gap_bucket"]),
            }
            self._states = _encode(self.models["hookseq"], batch, self.device)
        return self._states

    def new_track_audience(self, raw_item_id: int, k: int = 10) -> list[dict]:
        """Sample users most likely to engage with an ID-less track that has audio.

        The track is scored by HookSeq (its vector is audio + artist only) against the states of
        the bundle's sample users. Returns [{uid, score}].
        """
        pos = np.flatnonzero(self.raw_ids == int(raw_item_id))
        if len(pos) == 0:
            raise ValueError(f"track {raw_item_id} is not in the demo bundle")
        i = int(pos[0])
        if self.has_id[i]:
            raise ValueError(f"track {raw_item_id} has an ID embedding: pick an ID-less track")
        if not self.has_audio[i]:
            raise ValueError(f"track {raw_item_id} has no audio embedding")
        scores = self._sample_states() @ self._vecs["hookseq"][i + 1].numpy()
        order = np.argsort(-scores, kind="stable")[: max(1, int(k))]
        uids = self._w["w_uids"]
        return [{"uid": int(uids[j]), "score": float(scores[j])} for j in order]
