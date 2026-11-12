"""SASRec: causal pre-LN transformer over item-ID embeddings (also the base class of HookSeq).

Batch contract (see hookrec.train.datasets): LongTensors [B, L], left-padded, `items == 0` <=> pad.
Subclasses customise the model by overriding `item_repr` and `input_repr` only.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class SASRec(nn.Module):
    def __init__(
        self,
        n_items: int,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 2,
        max_len: int = 200,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_items, self.d_model, self.max_len, self.n_heads = n_items, d_model, max_len, n_heads
        self.n_catalog = n_items  # items scored by all_item_repr (HookSeq: catalog > ID table)
        # Item embeddings are shared between the input and output (scoring) sides.
        self.item_emb = nn.Embedding(n_items + 1, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,  # standard transformer ratio
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-LN
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(d_model)
        self.apply(self._init_weights)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def item_repr(self, idx: torch.Tensor) -> torch.Tensor:
        """Representation of item ids, [..., d]. HookSeq overrides this."""
        return self.item_emb(idx)

    def _positions(self, length: int, device: torch.device) -> torch.Tensor:
        # Left padding => the last index is always the most recent event. Anchoring it at
        # max_len-1 and counting backwards gives a given recency the same positional
        # embedding whatever the padded length, so extra left padding changes nothing.
        if length > self.max_len:
            raise ValueError(f"sequence length {length} exceeds max_len {self.max_len}")
        return torch.arange(self.max_len - length, self.max_len, device=device)

    def input_repr(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """[B, L, d] encoder input. HookSeq overrides this to add side features."""
        items = batch["items"]
        pos = self.pos_emb(self._positions(items.shape[1], items.device))
        return self.item_repr(items) * math.sqrt(self.d_model) + pos

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """[B, L, d] causal hidden states. Pad-position outputs are finite but meaningless."""
        pad = batch["items"] == 0
        length = pad.shape[1]
        # Block future keys and pad keys, except that every query may always see itself. A pad
        # query (and so an all-pad row) would otherwise softmax over nothing -> NaN, which then
        # leaks into real positions through 0 * NaN in the attention matmul. Real queries never
        # see pad keys (the diagonal exception only matters for pad queries).
        eye = torch.eye(length, dtype=torch.bool, device=pad.device)
        future = torch.triu(torch.ones_like(eye), diagonal=1)
        blocked = future[None] | (pad[:, None, :] & ~eye[None])  # [B, L, L], True = masked
        blocked = blocked.repeat_interleave(self.n_heads, dim=0)  # [B*H, L, L]
        x = self.drop(self.input_repr(batch))
        h = self.encoder(x, mask=blocked)
        return self.final_norm(h)

    def user_state(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """[B, d] hidden state at the last non-pad position (last index for an all-pad row)."""
        h = self.encode(batch)
        real = batch["items"] != 0
        length = real.shape[1]
        idx = torch.arange(length, device=real.device)
        last = (real * idx).amax(dim=1)  # 0 for all-pad rows
        last = torch.where(real.any(dim=1), last, torch.full_like(last, length - 1))
        return h[torch.arange(h.shape[0], device=h.device), last]

    @torch.no_grad()
    def all_item_repr(self, chunk: int = 262144) -> torch.Tensor:
        """[N+1, d] representation of every item (row 0 = pad), computed chunk by chunk."""
        dev = self.item_emb.weight.device
        parts = [
            self.item_repr(torch.arange(s, min(s + chunk, self.n_catalog + 1), device=dev))
            for s in range(0, self.n_catalog + 1, chunk)
        ]
        return torch.cat(parts, dim=0)
