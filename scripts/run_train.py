"""Train a sequence model, evaluate it, and write a run record.

python scripts/run_train.py --processed DIR --config configs/sasrec.yaml --split {val,test} \
    --ckpt DIR [--resume DIR ...] [--override train.lr=3e-4 ...] [--name NAME] --out results/

`--split val` trains on the val-train history with periodic val NDCG@10 and early stopping, then
evaluates `best.pt`. `--split test` trains on the full train history for the `best_step` of the
best matching val run (same processed dir, same config up to operational keys), with that run's
LR horizon as `schedule_steps`, then evaluates the final weights. With no matching val run it
resumes a checkpoint's own horizon, or fails unless `--allow-no-val` is given. Both write
`{name}-{split}` via `save_run` (warm + cold slices).

Kaggle sessions: if training stops on `train.max_hours`, a checkpoint is written and the script
exits 0 *without* evaluating or recording a run; re-run with the same `--ckpt` (or with the
previous output as `--resume`) to continue.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

from hookrec.data.processed import load_processed
from hookrec.models.base import build_model, evaluate_split, history_for
from hookrec.models.popularity import MostPop
from hookrec.train.checkpoint import load_latest, read_horizon
from hookrec.train.datasets import session_gap_from
from hookrec.train.scoring import SeqRecommender
from hookrec.train.trainer import device_name, train, training_hash
from hookrec.utils.config import load_config
from hookrec.utils.results import best_run, save_run
from hookrec.utils.seed import set_seed

log = logging.getLogger("run_train")


def parse_overrides(items: list[str]) -> dict:
    """`["a.b=1", "c=x"]` -> `{"a.b": 1, "c": "x"}` (values parsed as YAML scalars)."""
    out = {}
    for it in items:
        key, sep, val = it.partition("=")
        if not sep or not key:
            raise SystemExit(f"--override expects key=value, got {it!r}")
        out[key] = yaml.safe_load(val)
    return out


def best_val_horizon(out: Path, name: str, processed_dir: str, cfg: dict) -> tuple[int, int] | None:
    """(best_step, schedule_steps) of the best matching `{name}-val` run, or None.

    A run matches when its config equals `cfg` up to the operational keys (`training_hash`).
    schedule_steps is that val run's own LR horizon, so the test refit follows the same LR
    curve as the val model up to best_step.
    """
    target = training_hash(cfg)
    best = best_run(
        out,
        f"{name}-val",
        processed_dir,
        lambda r: training_hash(r["config"]) == target and r["extra"].get("best_step"),
    )
    if best is None:
        return None
    t = best["config"]["train"]
    return int(best["extra"]["best_step"]), int(t.get("schedule_steps") or t["max_steps"])


def _set_test_horizon(args, cfg: dict, name: str, processed_dir: str) -> None:
    found = best_val_horizon(args.out, name, processed_dir, cfg)
    dirs = [args.ckpt, *args.resume]
    ck = read_horizon(dirs)  # sidecar: no full checkpoint load (~2.3 GB at 500M)
    if found is not None:
        if ck is not None and (ck["max_steps"], ck["schedule_steps"]) != found:
            log.warning(
                "checkpoint %s horizon (%d / %d steps) overrides the val-derived horizon "
                "(%d / %d steps); the resumed run keeps the checkpoint's",
                ck["path"], ck["max_steps"], ck["schedule_steps"], *found,
            )  # fmt: skip
        cfg["train"]["max_steps"], cfg["train"]["schedule_steps"] = found
        log.info("test run: max_steps = best val step %d, schedule_steps = %d", *found)
        return
    if ck is None:  # no sidecar (older checkpoint): fall back to reading the checkpoint itself
        payload = load_latest(dirs)
        if payload is not None and "max_steps" in payload:
            ck = {k: payload[k] for k in ("max_steps", "schedule_steps", "path")}
    if ck is not None:
        log.info(
            "no matching '%s-val' run in %s: resuming %s with its horizon (%d / %d steps)",
            name, args.out, ck["path"], ck["max_steps"], ck["schedule_steps"],
        )  # fmt: skip
        return
    if not args.allow_no_val:
        raise SystemExit(
            f"no matching '{name}-val' run for processed dir '{processed_dir}' in {args.out} and "
            "no checkpoint to resume: run --split val first (same config), or pass "
            "--allow-no-val to train for the configured train.max_steps"
        )
    log.warning(
        "--allow-no-val: training for the configured max_steps=%d", cfg["train"]["max_steps"]
    )


def main(argv: list[str] | None = None) -> Path | None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--split", choices=("val", "test"), required=True)
    ap.add_argument("--ckpt", type=Path, required=True, help="writable checkpoint dir")
    ap.add_argument("--resume", type=Path, action="append", default=[], help="read-only dirs")
    ap.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--name", help="run name (default: config file stem)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--allow-no-val",
        action="store_true",
        help="--split test without a matching val run: use the configured max_steps",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = load_config(args.config, parse_overrides(args.override))
    name = args.name or args.config.stem
    data = load_processed(args.processed)

    if args.split == "test":
        _set_test_horizon(args, cfg, name, data.path.name)

    set_seed(cfg["train"]["seed"])  # model init is part of the run
    model = build_model(cfg, data)
    res = train(cfg, data, model, args.split, args.ckpt, args.resume)
    # record the horizon actually trained (a resumed run takes it from its checkpoint)
    cfg["train"]["max_steps"], cfg["train"]["schedule_steps"] = res.max_steps, res.schedule_steps
    if res.stopped_reason == "max_hours":
        log.warning(
            "stopped on max_hours at step %d: resume from %s to finish", res.steps, args.ckpt
        )
        return None

    device = next(model.parameters()).device
    if args.split == "val" and (args.ckpt / "best.pt").exists():
        model.load_state_dict(
            torch.load(args.ckpt / "best.pt", map_location=device, weights_only=True)["model"]
        )
    seqs, _ = history_for(data, args.split)
    fallback = MostPop().fit(seqs, data.vocab.n_items)
    rec = SeqRecommender(
        model, seqs, cfg["model"]["max_len"], device, fallback, session_gap=session_gap_from(cfg)
    )
    ev = evaluate_split(rec, data, args.split)

    gpu = device_name(device)
    extra = {
        "split": args.split,
        "processed_dir": data.path.name,
        "dataset_size": data.cfg["size"],
        "best_step": res.best_step,
        "steps": res.steps,
        "stopped_reason": res.stopped_reason,
        "val_best_metric": res.best_metric,
        "wall_time_s": round(res.wall_time, 2),
        "steps_per_sec": round(res.steps_per_sec, 3),
        "gpu_name": gpu,
        "gpu_hours": round(res.wall_time / 3600, 4) if device.type == "cuda" else 0.0,
        "metrics_cold": ev["metrics_cold"],
        "n_cold_users": ev["n_cold_users"],
        "n_eval_users": ev["n_eval_users"],
    }
    path = save_run(args.out, f"{name}-{args.split}", cfg, ev["metrics"], extra)
    log.info(
        "%s [%s] ndcg@10=%.4f (%d users, %d cold) -> %s",
        name, args.split, ev["metrics"]["ndcg@10"], ev["n_eval_users"], ev["n_cold_users"], path,
    )  # fmt: skip
    return path


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("run_train failed")
        sys.exit(1)
