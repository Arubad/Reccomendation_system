"""Model registry: `fit_recommender(name, data, cfg, split)` dispatches to a registered builder.

`build_model(cfg, data)` constructs the (untrained) neural models trained by `hookrec.train`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from hookrec.data.processed import ProcessedData
from hookrec.data.sequences import PackedSequences
from hookrec.eval.protocol import Recommender, evaluate

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


def evaluate_split(rec: Recommender, data: ProcessedData, split: Split) -> dict:
    """Metrics on the split's warm truth plus the cold-user slice (served by `rec`'s fallback)."""
    n = data.vocab.n_items
    warm_truth = data.truth_val if split == "val" else data.truth_test
    cold_truth = data.truth_val_cold if split == "val" else data.truth_test_cold
    warm = evaluate(rec, warm_truth, n)
    cold = evaluate(rec, cold_truth, n).metrics if cold_truth else {}
    return {
        "metrics": warm.metrics,
        "metrics_cold": cold,
        "n_eval_users": len(warm.uids),
        "n_cold_users": len(cold_truth),
    }


def build_model(cfg: dict, data: ProcessedData):
    """Untrained neural sequence model for `cfg["model"]` (torch is imported lazily)."""
    m = {k: v for k, v in cfg["model"].items() if k != "name"}
    name = cfg["model"]["name"]
    if name == "sasrec":
        from hookrec.models.sasrec import SASRec

        return SASRec(n_items=data.vocab.n_items, **m)  # R4: an ID row per catalog item
    raise KeyError(f"unknown neural model {name!r}")


def fit_recommender(name: str, data: ProcessedData, cfg: dict, split: Split) -> Recommender:
    if split not in ("val", "test"):
        raise ValueError(f"split must be 'val' or 'test', got {split!r}")
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[name](data, cfg, split)
