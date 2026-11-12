"""Full-catalog top-k scoring on the GPU, and a `Recommender` wrapper around sequence models."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from hookrec.data.sequences import PackedSequences
from hookrec.models.popularity import MostPop, fill_cold
from hookrec.train.datasets import _INPUT_FIELDS


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
        rows = ambiguous.nonzero().squeeze(1)
        s = sc[rows].sort(dim=1, descending=True, stable=True)
        vals[rows], idx[rows] = s.values[:, :kk], s.indices[:, :kk]
    return vals, idx


@torch.no_grad()
def score_topk(
    user_vecs: Tensor,
    item_vecs: Tensor,
    k: int,
    item_chunk: int = 500_000,
    exclude_pad: bool = True,
) -> tuple[Tensor, Tensor]:
    """Top-k items by dot product, [U, k] scores and LongTensor ids; ties go to the lower id.

    A running top-k is merged across item chunks, so memory is O(U * item_chunk). Computed in
    float32 on the inputs' device. With `exclude_pad`, item 0 scores -inf (it can only appear
    if k exceeds the number of real items, and 0 is never a hit).
    """
    u = user_vecs.float()
    n_users, dev = u.shape[0], u.device
    best_v = torch.full((n_users, k), float("-inf"), device=dev)
    best_i = torch.zeros((n_users, k), dtype=torch.long, device=dev)
    for start in range(0, item_vecs.shape[0], item_chunk):
        sc = u @ item_vecs[start : start + item_chunk].to(dev).float().T
        if exclude_pad and start == 0:
            sc[:, 0] = float("-inf")
        vals, idx = _chunk_topk(sc, min(k, sc.shape[1]))
        cand_v = torch.cat([best_v, vals], dim=1)
        cand_i = torch.cat([best_i, idx + start], dim=1)
        cand_v, cand_i = _sort_desc_low_index_first(cand_v, cand_i)
        best_v, best_i = cand_v[:, :k], cand_i[:, :k]
    return best_v, best_i


class SeqRecommender:
    """`Recommender` over a sequence model: encode each user's last window, score the catalog.

    Users absent from `seqs` get the fallback's top-k. Item representations are computed once
    per instance, so build a new recommender after the weights change.
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
    ):
        self.model, self.seqs, self.max_len = model, seqs, max_len
        self.device = torch.device(device)
        self.fallback, self.batch_size, self.item_chunk = fallback, batch_size, item_chunk
        self._item_vecs: Tensor | None = None

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
        return out

    @torch.no_grad()
    def recommend(self, uids: np.ndarray, k: int) -> np.ndarray:
        rows = self.seqs.index_of(np.asarray(uids))
        known = rows >= 0
        out = np.zeros((len(rows), k), dtype=np.int64)
        was_training = self.model.training
        self.model.eval()
        try:
            if self._item_vecs is None:
                self._item_vecs = self.model.all_item_repr().float()
            amp = self.device.type == "cuda"
            known_pos = np.flatnonzero(known)
            for s in range(0, len(known_pos), self.batch_size):
                at = known_pos[s : s + self.batch_size]
                with torch.autocast(self.device.type, dtype=torch.float16, enabled=amp):
                    users = self.model.user_state(self._windows(rows[at]))
                _, idx = score_topk(users, self._item_vecs, k, self.item_chunk)
                out[at] = idx.cpu().numpy()
        finally:
            self.model.train(was_training)
        return fill_cold(uids, out, self.fallback.top_items(k), known)
