"""Run the baselines (MostPop, DecayPop, iALS): validation grid search, then a test refit.

python scripts/run_baselines.py --processed DIR --config configs/baselines.yaml \
    --split {val,test} --out results/ [--models mostpop,decaypop,ials] [--quick]

`--split val` fits every grid point on the val-train history and scores it on the val day.
`--split test` takes, per model, the val run with the best NDCG@10 for the same processed dir,
refits on the full train history and scores the test day (warm users, plus the cold slice).
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
import time
from pathlib import Path

from hookrec.data.processed import ProcessedData, load_processed
from hookrec.eval.protocol import evaluate
from hookrec.models import REGISTRY, fit_recommender
from hookrec.utils.config import load_config
from hookrec.utils.results import load_runs, save_run

log = logging.getLogger("run_baselines")

DEFAULT_MODELS = ("mostpop", "decaypop", "ials")


def grid_points(block: dict, quick: bool) -> list[dict]:
    """Expand a model block into concrete configs: fixed params plus the grid's product."""
    fixed = {k: v for k, v in block.items() if k != "grid"}
    grid = block.get("grid") or {}
    if not grid:
        return [fixed]
    names = list(grid)
    value_lists = [grid[n][:1] if quick else grid[n] for n in names]
    return [{**fixed, **dict(zip(names, vals))} for vals in itertools.product(*value_lists)]


def _evaluate_split(model, data: ProcessedData, split: str) -> dict:
    """Metrics on the warm truth plus the cold-user slice (served by the model's fallback)."""
    n = data.vocab.n_items
    warm_truth = data.truth_val if split == "val" else data.truth_test
    cold_truth = data.truth_val_cold if split == "val" else data.truth_test_cold
    warm = evaluate(model, warm_truth, n)
    cold = evaluate(model, cold_truth, n).metrics if cold_truth else {}
    return {
        "metrics": warm.metrics,
        "metrics_cold": cold,
        "n_eval_users": len(warm.uids),
        "n_cold_users": len(cold_truth),
    }


def _run_one(name: str, cfg: dict, data: ProcessedData, split: str, out: Path) -> dict:
    t0 = time.perf_counter()
    model = fit_recommender(name, data, cfg, split)
    res = _evaluate_split(model, data, split)
    extra = {
        "split": split,
        "processed_dir": data.path.name,
        "dataset_size": data.cfg["size"],
        "wall_time_s": round(time.perf_counter() - t0, 2),
        "n_eval_users": res["n_eval_users"],
        "metrics_cold": res["metrics_cold"],
        "n_cold_users": res["n_cold_users"],
    }
    save_run(out, f"{name}-{split}", cfg, res["metrics"], extra)
    log.info(
        "%s [%s] %s ndcg@10=%.4f (%d users, %d cold)",
        name,
        split,
        cfg,
        res["metrics"]["ndcg@10"],
        res["n_eval_users"],
        res["n_cold_users"],
    )
    return {"model": name, "split": split, "cfg": cfg, "metrics": res["metrics"], **extra}


def _best_val_cfg(out: Path, name: str, processed_dir: str) -> dict:
    runs = [
        r
        for r in load_runs(out)
        if r["run_name"] == f"{name}-val" and r["extra"].get("processed_dir") == processed_dir
    ]
    if not runs:
        raise SystemExit(
            f"no '{name}-val' run for processed dir '{processed_dir}' in {out}; "
            "run with --split val first"
        )
    return max(runs, key=lambda r: r["metrics"]["ndcg@10"])["config"]


def _table(rows: list[dict]) -> str:
    head = "| model | split | config | ndcg@10 | recall@10 | ndcg@100 | recall@100 | cov@10 | cold ndcg@10 |"
    lines = [head, "|" + "---|" * 9]
    for r in rows:
        m, c = r["metrics"], r["metrics_cold"]
        cfg = ", ".join(f"{k}={v}" for k, v in r["cfg"].items())
        cold = f"{c['ndcg@10']:.4f}" if c else "-"
        lines.append(
            f"| {r['model']} | {r['split']} | {cfg or '-'} | {m['ndcg@10']:.4f} | "
            f"{m['recall@10']:.4f} | {m['ndcg@100']:.4f} | {m['recall@100']:.4f} | "
            f"{m['coverage@10']:.4f} | {cold} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> list[dict]:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--split", choices=("val", "test"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--quick", action="store_true", help="first grid value of each parameter")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    cfg = load_config(args.config)
    unknown = [m for m in models if m not in REGISTRY or m not in cfg]
    if unknown:
        raise SystemExit(f"unknown or unconfigured models: {unknown}")

    data = load_processed(args.processed)
    rows: list[dict] = []
    for name in models:
        if args.split == "val":
            points = grid_points(cfg[name], args.quick)
        else:
            points = [_best_val_cfg(args.out, name, data.path.name)]
        log.info("%s: %d config(s) on %s", name, len(points), args.split)
        rows += [_run_one(name, p, data, args.split, args.out) for p in points]

    print(_table(rows))
    return rows


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("run_baselines failed")
        sys.exit(1)
