"""Fairness, slices and bootstrap CIs for one finished test run.

python scripts/run_eval.py --processed DIR --run results/<run>.json [--ckpt DIR] --out results/

Rebuilds the recommender from the run record: neural runs (config has a `model` key) load
`model_final.pt` from --ckpt; baselines (mostpop/decaypop/ials) are refit on the train history.
Warm test users are scored once; slices, artist-exposure fairness, bootstrap CIs (ndcg@10,
recall@100) and the cold-user metrics all come from that one pass. Writes `{run_name}-eval-*.json`
(via save_run) and `{run_name}-{size}-eval-peruser.npz` (for paired comparisons) into --out.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from hookrec.data.processed import ProcessedData, load_processed
from hookrec.eval.bootstrap import bootstrap_ci
from hookrec.eval.fairness import artist_coverage, artist_exposure_gini, long_tail_share
from hookrec.eval.protocol import Recommender, evaluate
from hookrec.eval.slices import evaluate_slices, slice_truth
from hookrec.models import fit_recommender
from hookrec.models.base import build_model
from hookrec.models.popularity import MostPop
from hookrec.utils.results import save_run

log = logging.getLogger("run_eval")

BASELINES = ("mostpop", "decaypop", "ials")
FAIRNESS_KS = (10, 100)
CI_METRICS = ("ndcg@10", "recall@100")
DRIFT_WARN = 1e-6
# a model trained on CUDA fp16 may be re-scored on CPU, so neural runs get a looser tolerance
DRIFT_TOL_NEURAL = 1e-3
DRIFT_TOL_BASELINE = 1e-4


def build_recommender(run: dict, data: ProcessedData, ckpt: Path | None) -> Recommender:
    cfg, name = run["config"], run["run_name"]
    if "model" in cfg:
        import torch

        from hookrec.models.base import history_for
        from hookrec.train.datasets import session_gap_from
        from hookrec.train.scoring import SeqRecommender

        if ckpt is None or not (Path(ckpt) / "model_final.pt").exists():
            raise SystemExit(f"neural run {name!r} needs --ckpt DIR containing model_final.pt")
        tc = cfg.get("train") or {}
        device = torch.device(tc.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = build_model(cfg, data)
        payload = torch.load(Path(ckpt) / "model_final.pt", map_location=device, weights_only=True)
        model.load_state_dict(payload["model"])
        model.to(device)
        seqs, _ = history_for(data, "test")
        fallback = MostPop().fit(seqs, data.vocab.n_items)
        return SeqRecommender(
            model, seqs, cfg["model"]["max_len"], device, fallback,
            session_gap=session_gap_from(cfg),
        )  # fmt: skip
    model_name = name.rsplit("-", 1)[0]
    if model_name not in BASELINES:
        raise SystemExit(f"cannot rebuild run {name!r}: not a neural run or a known baseline")
    return fit_recommender(model_name, data, cfg, "test")


def check_drift(run: dict, ndcg10: float) -> None:
    """Raise if the rebuilt recommender's ndcg@10 differs from the run record's."""
    recorded = (run.get("metrics") or {}).get("ndcg@10")
    if recorded is None:
        raise RuntimeError(
            f"{run.get('run_name')}: the run record has no metrics['ndcg@10'] to check drift "
            "against (--no-drift-check skips)"
        )
    neural = "model" in run["config"]
    tol = DRIFT_TOL_NEURAL if neural else DRIFT_TOL_BASELINE
    diff = abs(ndcg10 - recorded)
    if neural and DRIFT_WARN < diff <= tol:
        log.warning(
            "drift: re-scored ndcg@10 %.6f vs recorded %.6f (diff %.2g, within tol %g)",
            ndcg10, recorded, diff, tol,
        )  # fmt: skip
    if diff > tol:
        log.error("drift: re-scored ndcg@10 %.6f vs recorded %.6f (tol %g)", ndcg10, recorded, tol)
        raise RuntimeError(
            f"{run['run_name']}: rebuilt model does not reproduce the recorded ndcg@10 "
            f"({ndcg10:.6f} vs {recorded:.6f}); wrong checkpoint or data? (--no-drift-check skips)"
        )


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True, help="results/<run>.json (a test run)")
    ap.add_argument("--ckpt", type=Path, help="dir with model_final.pt (neural runs)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--no-drift-check",
        action="store_true",
        help="skip the check that re-scored ndcg@10 matches the run record",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    run = json.loads(args.run.read_text())
    name = run["run_name"]
    if (run.get("extra") or {}).get("split") != "test":
        raise SystemExit(f"{args.run} is not a test run (extra.split != 'test')")
    data = load_processed(args.processed)
    if run["extra"].get("processed_dir") not in (None, data.path.name):
        raise SystemExit(
            f"run was made on {run['extra']['processed_dir']} but --processed is {data.path.name}"
        )
    n = data.vocab.n_items

    rec = build_recommender(run, data, args.ckpt)
    warm = evaluate(rec, data.truth_test, n)  # the one scoring pass
    log.info("%s: scored %d warm test users", name, len(warm.uids))
    if not args.no_drift_check:
        check_drift(run, warm.metrics["ndcg@10"])

    slices = slice_truth(
        data.truth_test, data.test_events(), data.vocab, data.features, data.seqs_train
    )
    slice_metrics = evaluate_slices(warm.topk, warm.uids, slices, n)

    fairness: dict[str, float] = {}
    feat = data.features
    for k in FAIRNESS_KS:
        top = warm.topk[:, :k]
        # artist 0 = unknown: excluded from exposure counts and from the denominator
        fairness[f"artist_gini@{k}"] = artist_exposure_gini(
            top, feat.artist_idx, feat.n_artists, ignore_zero=True
        )
        fairness[f"artist_coverage@{k}"] = artist_coverage(
            top, feat.artist_idx, feat.n_artists, ignore_zero=True
        )
        fairness[f"long_tail_share@{k}"] = long_tail_share(top, feat.pop_train)

    ci = {m: list(bootstrap_ci(warm.per_user[m], seed=0)) for m in CI_METRICS}

    cold: dict = {}
    if data.truth_test_cold:
        cold = evaluate(rec, data.truth_test_cold, n).metrics
    log.info("cold test users: %d", len(data.truth_test_cold))

    extra = {
        "split": "test",
        "kind": "eval",
        "source_run": name,
        "processed_dir": data.path.name,
        "dataset_size": data.cfg["size"],
        "n_eval_users": len(warm.uids),
        "n_cold_users": len(data.truth_test_cold),
        "metrics_cold": cold,
        "slices": slice_metrics,
        "fairness": fairness,
        "ci": ci,
    }
    peruser = Path(args.out) / f"{name}-{data.cfg['size']}-eval-peruser.npz"
    extra["peruser_file"] = peruser.name
    path = save_run(args.out, f"{name}-eval", run["config"], warm.metrics, extra)
    np.savez(
        peruser,
        uids=warm.uids,
        **{m: warm.per_user[m] for m in CI_METRICS},
    )
    log.info("ndcg@10 %.4f [%.4f, %.4f] -> %s", *ci["ndcg@10"], path)
    return path


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("run_eval failed")
        sys.exit(1)
