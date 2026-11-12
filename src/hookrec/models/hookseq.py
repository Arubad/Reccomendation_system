"""HookSeq: SASRec plus audio, artist, event-type, organic and played-ratio signals.

`item_repr` = ID embedding (randomly dropped in training, so audio/artist carry cold items)
+ audio MLP (a learned vector for items without audio) + artist embedding. `input_repr` adds
per-event token / organic / ratio embeddings. A parameter exists only when its flag is on, so
with every flag off the parameters equal `SASRec(n_items=n_id)`. `use_time_gap` adds a
bucketed seconds-since-previous-event embedding (batch key `gap_bucket`); `like_head` adds a
bilinear like-prediction head (`like_logit`) for the trainer's auxiliary loss.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from hookrec.data.label import N_RATIO_BUCKETS, N_TOKEN_TYPES
from hookrec.models.sasrec import SASRec

if TYPE_CHECKING:
    from hookrec.data.features import ItemFeatures


class HookSeq(SASRec):
    def __init__(
        self,
        n_items: int,
        features: ItemFeatures,
        n_id: int,
        id_idx: np.ndarray,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 2,
        max_len: int = 200,
        dropout: float = 0.2,
        use_audio: bool = True,
        use_artist: bool = True,
        use_event: bool = True,
        use_organic: bool = True,
        use_ratio: bool = True,
        id_dropout: float = 0.2,
        like_head: bool = False,
        use_time_gap: bool = False,
        n_gap_buckets: int = 8,
        arch: str = "hookseq",
    ):
        if arch != "hookseq":  # "official" is the paper-faithful plain SASRec row only (R30)
            raise ValueError(f"HookSeq supports model.arch 'hookseq' only, got {arch!r}")
        # NB: on HookSeq `self.n_items` is n_id (the ID table size, SASRec's constructor arg);
        # `self.n_catalog` is the catalog size N that item_repr / all_item_repr cover.
        # item_emb is the ID table (n_id + 1 rows), not one row per catalog item.
        super().__init__(n_id, d_model, n_layers, n_heads, max_len, dropout)
        id_idx = np.asarray(id_idx)
        n_catalog = len(id_idx) - 1
        if n_catalog != n_items:
            raise ValueError(f"id_idx has {len(id_idx)} rows, expected n_items + 1 = {n_items + 1}")
        if id_idx.min() < 0 or id_idx.max() > n_id:
            raise ValueError(f"id_idx values must lie in [0, n_id={n_id}], got max {id_idx.max()}")
        for name in ("audio", "has_audio", "artist_idx"):
            if len(getattr(features, name)) != n_catalog + 1:
                raise ValueError(f"features.{name} must have n_items + 1 = {n_items + 1} rows")
        self.n_catalog = n_catalog
        self.use_audio, self.use_artist = use_audio, use_artist
        self.use_event, self.use_organic, self.use_ratio = use_event, use_organic, use_ratio
        self.id_dropout = id_dropout
        self.like_head, self.use_time_gap = like_head, use_time_gap
        d = d_model
        # Non-persistent (kept out of checkpoints; run_train rebuilds them from ProcessedData)
        # and only registered when used. With every flag off, an item without an ID
        # (id_idx == 0) has a zero representation (R4: a0 is ID-only; cold items are
        # unscorable, unlike SASRec's row per catalog item).
        self.register_buffer("id_idx", torch.as_tensor(id_idx, dtype=torch.long), persistent=False)
        if use_audio:
            self.register_buffer("audio", torch.as_tensor(features.audio), persistent=False)
            self.register_buffer("has_audio", torch.as_tensor(features.has_audio), persistent=False)
        if use_artist:
            self.register_buffer(
                "artist_idx",
                torch.as_tensor(features.artist_idx, dtype=torch.long),
                persistent=False,
            )
        if use_audio:
            self.audio_mlp = nn.Sequential(
                nn.Linear(features.audio.shape[1], d), nn.GELU(), nn.Linear(d, d)
            )
            self.missing_audio_vec = nn.Parameter(torch.empty(d))
        if use_artist:
            self.artist_emb = nn.Embedding(features.n_artists + 1, d, padding_idx=0)
        if use_event:
            self.token_emb = nn.Embedding(N_TOKEN_TYPES, d, padding_idx=0)
        if use_organic:
            self.organic_emb = nn.Embedding(2, d)
        if use_ratio:
            self.ratio_emb = nn.Embedding(N_RATIO_BUCKETS, d, padding_idx=0)
        # SASRec's own parameters keep their base init (so the all-off model matches it exactly)
        for name in ("audio_mlp", "artist_emb", "token_emb", "organic_emb", "ratio_emb"):
            if hasattr(self, name):
                getattr(self, name).apply(self._init_weights)
        with torch.no_grad():
            for name in ("artist_emb", "token_emb", "ratio_emb"):
                if hasattr(self, name):
                    getattr(self, name).weight[0].zero_()
            if use_audio:
                nn.init.normal_(self.missing_audio_vec, 0.0, 0.02)
        # Created last, so enabling them leaves every other parameter's init draw unchanged.
        if use_time_gap:  # bucket 0 = pad / first event (see datasets.gap_buckets)
            self.gap_emb = nn.Embedding(n_gap_buckets, d, padding_idx=0)
            self.gap_emb.apply(self._init_weights)
            with torch.no_grad():
                self.gap_emb.weight[0].zero_()
        if like_head:  # bilinear "will be liked" score, see like_logit
            self.like_W = nn.Linear(d, d, bias=False)
            self.like_W.apply(self._init_weights)

    def item_repr(self, idx: torch.Tensor) -> torch.Tensor:
        out = self.item_emb(self.id_idx[idx])
        if self.training and self.id_dropout > 0:
            keep = torch.rand(idx.shape, device=idx.device) >= self.id_dropout
            out = out * keep.unsqueeze(-1)  # no rescale: train and eval share semantics
        if self.use_audio:
            audio = self.audio_mlp(self.audio[idx].float())
            out = out + torch.where(
                self.has_audio[idx].unsqueeze(-1), audio, self.missing_audio_vec
            )
        if self.use_artist:
            out = out + self.artist_emb(self.artist_idx[idx])
        return out * (idx != 0).unsqueeze(-1)

    def input_repr(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        items = batch["items"]
        pos = self.pos_emb(self._positions(items.shape[1], items.device))
        x = self.item_repr(items) * math.sqrt(self.d_model) + pos
        extra = None
        for flag, emb, key in (
            (self.use_event, "token_emb", "tokens"),
            (self.use_organic, "organic_emb", "organic"),
            (self.use_ratio, "ratio_emb", "ratio"),
            (self.use_time_gap, "gap_emb", "gap_bucket"),
        ):
            if flag:
                e = getattr(self, emb)(batch[key])
                extra = e if extra is None else extra + e
        if extra is not None:
            x = x + extra * (items != 0).unsqueeze(-1)
        return x

    def like_logit(self, hidden: torch.Tensor, target_vecs: torch.Tensor) -> torch.Tensor:
        """Bilinear like logit <hidden, W target_vecs>, shape = any leading shape of the inputs."""
        return (hidden * self.like_W(target_vecs)).sum(-1)
