"""Full-catalog top-k scoring on the GPU, and a `Recommender` wrapper around sequence models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor, nn

from hookrec.train.datasets import _INPUT_FIELDS, DEFAULT_SESSION_GAP, gap_buckets

if TYPE_CHECKING:
    from hookrec.data.sequences import PackedSequences
    from hookrec.models.popularity import MostPop

_TIE_ROWS = 64  # rows per full stable sort when a top-k cut falls inside a tie


def _sort_desc_low_index_first(vals: Tensor, idx: Tensor) -> tuple[Tensor, Tensor]:
    """Order candidates by (-score, index): sort by index, then a stable sort by score."""
    o = idx.argsort(dim=1)
    vals, idx = vals.gather(1, o), idx.gather(1, o)
    o = vals.sort(dim=1, descending=True, stable=True).indices
    return vals.gather(1, o), idx.gather(1, o)


def _chunk_topk(sc: Tensor, kk: int) -> tuple[Tensor, Tensor]:
    """Exact top-kk of `sc` [U, C] with ties broken by the lower column (local indices)."""
    vals, idx = sc.topk(kk, dim=1)
    # topk only guarantees the *set* above the k-th value; ties at the k-th value may be cut
    # arbitrarily. Rows where the cut fell inside a tie are redone with a stable full sort
    # (exact ties of float dot products are rare, so this is usually zero rows).
    thr = vals[:, -1:]
    ambiguous = (sc == thr).sum(1) > (vals == thr).sum(1)
    if ambiguous.any():
        all_rows = ambiguous.nonzero().squeeze(1)
        for s in range(0, len(all_rows), _TIE_ROWS):  # bounds the full-sort memory
            rows = all_rows[s : s + _TIE_ROWS]
            srt = sc[rows].sort(dim=1, descending=True, stable=True)
            vals[rows], idx[rows] = srt.values[:, :kk], srt.indices[:, :kk]
    return vals, idx


@torch.no_grad()
def score_topk(
    user_vecs: Tensor,
    item_vecs: Tensor,
    k: int,
    item_chunk: int = 500_000,
    exclude_pad: bool = True,
    allowed: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Top-k items by dot product, [U, k] scores and LongTensor ids; ties go to the lower id.

    A running top-k is merged across item chunks, so memory is O(U * item_chunk). Computed in
    float32 on the inputs' device. With `exclude_pad`, item 0 scores -inf (it can only appear
    if k exceeds the number of real items, and 0 is never a hit). `allowed` (bool, one per
    item row) scores the other items -inf (R32); they can only appear if k exceeds the number
    of allowed items, which `SeqRecommender` rejects.
    """
    u = user_vecs.float()
    n_users, dev = u.shape[0], u.device
    if allowed is not None:
        if allowed.shape != (item_vecs.shape[0],):
            raise ValueError(f"allowed has shape {tuple(allowed.shape)}, want ({len(item_vecs)},)")
        blocked = ~allowed.to(dev, torch.bool)
    best_v = torch.full((n_users, k), float("-inf"), device=dev)
    best_i = torch.zeros((n_users, k), dtype=torch.long, device=dev)
    for start in range(0, item_vecs.shape[0], item_chunk):
        sc = u @ item_vecs[start : start + item_chunk].to(dev).float().T
        if exclude_pad and start == 0:
            sc[:, 0] = float("-inf")
        if allowed is not None:
            sc.masked_fill_(blocked[None, start : start + item_chunk], float("-inf"))
        vals, idx = _chunk_topk(sc, min(k, sc.shape[1]))
        cand_v = torch.cat([best_v, vals], dim=1)
        cand_i = torch.cat([best_i, idx + start], dim=1)
        cand_v, cand_i = _sort_desc_low_index_first(cand_v, cand_i)
        best_v, best_i = cand_v[:, :k], cand_i[:, :k]
    return best_v, best_i


class SeqRecommender:
    """`Recommender` over a sequence model: encode each user's last window, score the catalog.

    Users absent from `seqs` get the fallback's top-k. Item representations are computed once
    per instance, so build a new recommender after the weights change. User states use fp16
    autocast on CUDA unless `amp` is False (`train.amp`); on CPU they are always fp32.
    `item_mask` (bool [N+1], `hookrec.models.base.scoring_mask`) restricts the recommendable
    items (R32); None = the full catalog.
    """

    def __init__(
        self,
        model: nn.Module,
        seqs: PackedSequences,
        max_len: int,
        device: torch.device | str,
        fallback: MostPop,
        batch_size: int = 1024,
        item_chunk: int = 500_000,
        session_gap: float = DEFAULT_SESSION_GAP,
        amp: bool = True,
        item_mask: np.ndarray | None = None,
    ):
        self.session_gap, self.amp = session_gap, bool(amp)
        self.model, self.seqs, self.max_len = model, seqs, max_len
        self.device = torch.device(device)
        self.fallback, self.batch_size, self.item_chunk = fallback, batch_size, item_chunk
        self._item_vecs: Tensor | None = None
        self._allowed = None if item_mask is None else torch.from_numpy(np.asarray(item_mask))
        self._n_allowed = None if item_mask is None else int(np.asarray(item_mask)[1:].sum())

    def _windows(self, rows: np.ndarray) -> dict[str, Tensor]:
        """Left-padded last windows of the given rows, as in `SeqInferDataset` (vectorised)."""
        offsets = np.asarray(self.seqs.offsets)
        end = offsets[rows + 1]
        n = np.minimum(end - offsets[rows], self.max_len)
        pos = np.arange(self.max_len)[None, :] - (self.max_len - n)[:, None]  # <0 => pad
        valid = pos >= 0
        src = np.where(valid, end[:, None] - n[:, None] + pos, 0)
        out = {}
        for f in _INPUT_FIELDS:
            a = np.where(valid, np.asarray(getattr(self.seqs, f))[src], 0).astype(np.int64)
            out[f] = torch.from_numpy(a).to(self.device, non_blocking=True)
        # gap to the previous event of the same user, which may lie outside the window
        ts = np.asarray(self.seqs.ts)  # gather first, then cast: O(B*L), not O(T)
        first = src == offsets[rows][:, None]
        gap = ts[src].astype(np.int64) - ts[np.maximum(src - 1, 0)].astype(np.int64)
        gb = np.where(valid & ~first, gap_buckets(gap, self.session_gap), 0)
        out["gap_bucket"] = torch.from_numpy(gb.astype(np.int64)).to(self.device)
        return out

    @torch.no_grad()
    def recommend(self, uids: np.ndarray, k: int) -> np.ndarray:
        if self._n_allowed is not None and k > self._n_allowed:
            raise ValueError(f"k={k} exceeds the {self._n_allowed} items of the scoring catalog")
        rows = self.seqs.index_of(np.asarray(uids))
        known = rows >= 0
        out = np.zeros((len(rows), k), dtype=np.int64)
        was_training = self.model.training
        self.model.eval()
        try:
            if self._item_vecs is None:
                self._item_vecs = self.model.all_item_repr().float()
                if self._allowed is not None:  # once, next to the item vectors
                    self._allowed = self._allowed.to(self._item_vecs.device)
            amp = self.device.type == "cuda" and self.amp
            known_pos = np.flatnonzero(known)
            for s in range(0, len(known_pos), self.batch_size):
                at = known_pos[s : s + self.batch_size]
                with torch.autocast(self.device.type, dtype=torch.float16, enabled=amp):
                    users = self.model.user_state(self._windows(rows[at]))
                _, idx = score_topk(
                    users, self._item_vecs, k, self.item_chunk, allowed=self._allowed
                )
                out[at] = idx.cpu().numpy()
        finally:
            self.model.train(was_training)
        from hookrec.models.popularity import fill_cold

        return fill_cold(uids, out, self.fallback.top_items(k), known)
