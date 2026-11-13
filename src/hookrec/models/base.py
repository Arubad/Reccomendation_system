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
from hookrec.data.vocab import ItemVocab, with_id_min_count
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


def eval_catalog_of(cfg: dict) -> str:
    """The items a run may recommend: "listen_plus" when it trained with
    `loss.neg_catalog: listen_plus`, else "all" (recorded as `extra.eval_catalog`)."""
    return "listen_plus" if (cfg.get("loss") or {}).get("neg_catalog") == "listen_plus" else "all"


def scoring_mask(cfg: dict, data: ProcessedData) -> np.ndarray | None:
    """R32: bool [N+1], True = item may be recommended; None = every item (the default).

    With `neg_catalog: listen_plus` (and the Listen+ input of sasrec_bce, so every positive is
    Listen+ too) the items outside `listen_plus_items` never get a gradient: BCE touches only
    positives and catalog negatives, so their rows stay at random init. They are masked at
    scoring, as the official eval ranks only its Listen+ catalog. Lossless under the default
    `eval.target: listen_plus`: val and test truth are Listen+ events of the train period and
    the test day, all inside the catalog. Coverage keeps the full catalog N as its
    denominator, as for every other run.
    """
    if eval_catalog_of(cfg) != "listen_plus":
        return None
    mask = np.zeros(data.vocab.n_items + 1, dtype=bool)
    mask[listen_plus_items(data)] = True
    return mask


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


# model.* key that sets HookSeq's ID table at model-build time (not a constructor argument)
ID_MIN_COUNT = "id_min_count"


def effective_vocab(data: ProcessedData, model_cfg: dict) -> ItemVocab:
    """The vocab whose ID table a model with `model_cfg` (a cfg's `model` section) uses.

    `model.id_min_count` null or absent (every run before F5): the bundle's own vocab, with IDs
    for items with >= `data.min_count` train positives. An int k (HookSeq only): IDs 1..n_id'
    for the items with >= k train positives, in item-index order, rebuilt from the counts
    `build_vocab` used: positives (LISTEN_FULL or LIKE) over the full train period
    (`split.train`, `agg("train").pop_pos`), for val and test runs alike. So
    k = data.min_count reproduces the stored `id_idx`; catalog and bundle stay the same.
    Every path that builds a HookSeq model (`build_model`: run_train, the trainer's periodic
    eval, run_eval, profile_train; and the demo export) goes through this function, so a
    checkpoint is always loaded with the ID table it was trained with.
    """
    model_cfg = model_cfg or {}
    k = model_cfg.get(ID_MIN_COUNT)
    if k is None:
        return data.vocab
    if model_cfg.get("name") != "hookseq":
        raise ValueError(
            f"model.{ID_MIN_COUNT} is a HookSeq option, got model.name={model_cfg.get('name')!r}"
        )
    return with_id_min_count(data.vocab, data.agg("train").pop_pos, k)


def build_model(cfg: dict, data: ProcessedData):
    """Untrained neural sequence model for `cfg["model"]` (torch is imported lazily)."""
    m = {k: v for k, v in cfg["model"].items() if k not in ("name", ID_MIN_COUNT)}
    name = cfg["model"]["name"]
    vocab = effective_vocab(data, cfg["model"])  # raises on id_min_count outside HookSeq
    if name == "sasrec":
        from hookrec.models.sasrec import SASRec

        return SASRec(n_items=data.vocab.n_items, **m)  # R4: an ID row per catalog item
    if name == "hookseq":
        from hookrec.models.hookseq import HookSeq

        return HookSeq(
            n_items=data.vocab.n_items,
            features=data.features,
            n_id=vocab.n_id,
            id_idx=vocab.id_idx,
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
