"""HookSeq: SASRec plus audio, artist, event-type, organic and played-ratio signals.

`item_repr` = ID embedding (randomly dropped in training, so audio/artist carry cold items)
+ audio MLP (a learned vector for items without audio) + artist embedding. `input_repr` adds
per-event token / organic / ratio embeddings. A parameter exists only when its flag is on, so
with every flag off the parameters equal `SASRec(n_items=n_id)`.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from hookrec.data.features import ItemFeatures
from hookrec.data.label import N_RATIO_BUCKETS, N_TOKEN_TYPES
from hookrec.models.sasrec import SASRec


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
    ):
        # The inherited item_emb is the ID table (n_id + 1 rows), not one row per catalog item.
        super().__init__(n_id, d_model, n_layers, n_heads, max_len, dropout)
        n_catalog = features.audio.shape[0] - 1
        if n_catalog != n_items or len(id_idx) != n_catalog + 1:
            raise ValueError(f"features/id_idx do not match n_items={n_items}")
        self.n_catalog = n_catalog
        self.use_audio, self.use_artist = use_audio, use_artist
        self.use_event, self.use_organic, self.use_ratio = use_event, use_organic, use_ratio
        self.id_dropout = id_dropout
        d = d_model
        # non-persistent: rebuilt from ProcessedData, kept out of checkpoints
        self.register_buffer("id_idx", torch.as_tensor(id_idx, dtype=torch.long), persistent=False)
        self.register_buffer("audio", torch.as_tensor(features.audio), persistent=False)
        self.register_buffer("has_audio", torch.as_tensor(features.has_audio), persistent=False)
        self.register_buffer(
            "artist_idx", torch.as_tensor(features.artist_idx, dtype=torch.long), persistent=False
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
        ):
            if flag:
                e = getattr(self, emb)(batch[key])
                extra = e if extra is None else extra + e
        if extra is not None:
            x = x + extra * (items != 0).unsqueeze(-1)
        return x
