"""Model registry: `fit_recommender(name, data, cfg, split)` dispatches to a registered builder."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from hookrec.data.processed import ProcessedData
from hookrec.data.sequences import PackedSequences
from hookrec.eval.protocol import Recommender

Split = Literal["val", "test"]
Builder = Callable[[ProcessedData, dict, Split], Recommender]

REGISTRY: dict[str, Builder] = {}


def register(name: str) -> Callable[[Builder], Builder]:
    """Decorator adding a builder `(data, cfg, split) -> fitted Recommender` under `name`."""

    def deco(fn: Builder) -> Builder:
        if name in REGISTRY:
            raise ValueError(f"model {name!r} already registered")
        REGISTRY[name] = fn
        return fn

    return deco


def history_for(data: ProcessedData, split: Split) -> tuple[PackedSequences, float]:
    """Fit-time history and its end time: val tunes on `seqs_val_train`, test on `seqs_train`."""
    if split == "val":
        return data.seqs_val_train, data.meta["t_val_train_end"]
    if split == "test":
        return data.seqs_train, data.meta["t_train_end"]
    raise ValueError(f"split must be 'val' or 'test', got {split!r}")


def fit_recommender(name: str, data: ProcessedData, cfg: dict, split: Split) -> Recommender:
    if split not in ("val", "test"):
        raise ValueError(f"split must be 'val' or 'test', got {split!r}")
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[name](data, cfg, split)
