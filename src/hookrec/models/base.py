"""Model registry: `fit_recommender(name, data, cfg, split)` dispatches to a registered builder.

`build_model(cfg, data)` constructs the (untrained) neural models trained by `hookrec.train`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
import polars as pl

from hookrec.data.aggregates import PeriodAggregates
from hookrec.data.label import LISTEN_FULL
from hookrec.data.processed import ProcessedData, check_input_filter
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


def period_of(split: Split) -> str:
    """The training period a split fits on: val tunes on `val_train`, test on `train`."""
    if split == "val":
        return "val_train"
    if split == "test":
        return "train"
    raise ValueError(f"split must be 'val' or 'test', got {split!r}")


def input_filter_of(cfg: dict) -> str:
    """`cfg["data"]["input_filter"]`: "all" (default) or "listen_plus" (R19)."""
    return check_input_filter((cfg.get("data") or {}).get("input_filter", "all"))


def history_for(
    data: ProcessedData, split: Split, input_filter: str = "all"
) -> tuple[PackedSequences, float]:
    """Sequence-model history and its end time for `split`, from the `input_filter` set.

    Sequences are truncated to `max_len_store`; baselines use `aggregates_for` instead.
    """
    period = period_of(split)
    return data.seqs(period, input_filter), data.meta[f"t_{period}_end"]


def aggregates_for(data: ProcessedData, split: Split) -> PeriodAggregates:
    """Full, untruncated training-period aggregates for `split` (baselines, fallback, logQ)."""
    return data.agg(period_of(split))


def listen_plus_items(data: ProcessedData) -> np.ndarray:
    """Sorted item ids (1..N) with any Listen+ (LISTEN_FULL) event in the bundle: the train
    period (which holds val_train and the val day) plus the test day.

    The official SASRec catalog: `preprocess` keeps Listen+ events only (yambda
    benchmarks/models/sasrec/data.py:60-66), then numbers the unique items of that whole frame,
    before the split, so test-day items included (:68-72; `num_items` = their count, :27-28).
    Its negatives are uniform over that numbering (:118) and its eval ranks exactly those rows
    (eval.py `infer_items` = `item_embeddings.weight`). Only the 30-min gap before the test day
    is not in our bundle.
    """
    has = data.agg("train").pop_lp > 0
    test = (
        data.test_events()
        .filter(pl.col("token") == LISTEN_FULL)
        .select(pl.col("item_idx").unique())
        .collect(engine="streaming")["item_idx"]
        .to_numpy()
    )
    has[test] = True
    has[0] = False
    return np.flatnonzero(has)


def evaluate_split(rec: Recommender, data: ProcessedData, split: Split) -> dict:
    """Metrics on the split's warm truth plus the cold-user slice.

    Cold users have no Listen+ event in the training period; `rec` serves them as it would
    any user (its MostPop fallback when it has no input history for them).
    """
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
    if name == "hookseq":
        from hookrec.models.hookseq import HookSeq

        return HookSeq(
            n_items=data.vocab.n_items,
            features=data.features,
            n_id=data.vocab.n_id,
            id_idx=data.vocab.id_idx,
            **m,
        )
    raise KeyError(f"unknown neural model {name!r}")


def fit_recommender(name: str, data: ProcessedData, cfg: dict, split: Split) -> Recommender:
    if split not in ("val", "test"):
        raise ValueError(f"split must be 'val' or 'test', got {split!r}")
    if name not in REGISTRY:
        from hookrec.models import ials, popularity  # noqa: F401  (registers the baselines)
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[name](data, cfg, split)
