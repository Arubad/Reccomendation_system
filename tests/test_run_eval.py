import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TINY = {
    "model.d_model": 16,
    "model.n_layers": 1,
    "model.max_len": 20,
    "model.dropout": 0.1,
    "train.batch_size": 4,
    "train.warmup_steps": 100,
    "train.max_steps": 10,
    "train.eval_every_steps": 10,
    "train.ckpt_every_steps": 5,
    "train.num_workers": 0,
    "train.log_every_steps": 10,
    "train.device": "cpu",
    "loss.n_uniform_negatives": 32,
    "loss.n_in_batch_negatives": 32,
}


def _load(script):
    spec = importlib.util.spec_from_file_location(script, ROOT / "scripts" / f"{script}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _check(path, data, out, name):
    rec = json.loads(Path(path).read_text())
    assert rec["run_name"] == f"{name}-eval"
    ex = rec["extra"]
    assert ex["n_eval_users"] == len(data.truth_test)
    assert ex["n_cold_users"] == len(data.truth_test_cold)
    assert {"all", "organic", "served", "cold_items", "warm_items", "users_low"} <= set(
        ex["slices"]
    )
    assert ex["slices"]["all"]["ndcg@10"] == rec["metrics"]["ndcg@10"]
    for k in (10, 100):
        for f in ("artist_gini", "artist_coverage", "long_tail_share"):
            assert 0.0 <= ex["fairness"][f"{f}@{k}"] <= 1.0
    for m in ("ndcg@10", "recall@100"):
        mean, lo, hi = ex["ci"][m]
        assert lo <= mean <= hi
    with np.load(Path(out) / f"{name}-eval-peruser.npz") as z:
        assert len(z["uids"]) == len(z["ndcg@10"]) == len(z["recall@100"]) == ex["n_eval_users"]


def test_run_eval_neural_smoke(processed_synth, tmp_path):
    train = _load("run_train")
    ev = _load("run_eval")
    over = [a for k, v in TINY.items() for a in ("--override", f"{k}={v}")]
    cfg = str(ROOT / "configs" / "sasrec.yaml")
    run_path = train.main(
        ["--processed", str(processed_synth.path), "--config", cfg, "--split", "test",
         "--allow-no-val", "--ckpt", str(tmp_path / "ck"), "--out", str(tmp_path / "r"), *over]
    )  # fmt: skip
    path = ev.main(
        ["--processed", str(processed_synth.path), "--run", str(run_path),
         "--ckpt", str(tmp_path / "ck"), "--out", str(tmp_path / "r")]
    )  # fmt: skip
    _check(path, processed_synth, tmp_path / "r", "sasrec-test")
    # the eval record must not leak into the headline table
    from hookrec.eval.report import write_readme_tables

    text = write_readme_tables(tmp_path / "r", tmp_path / "res.md").read_text()
    assert "sasrec-test-eval" not in text.split("## Slices")[0]
    assert "### sasrec-test" in text


def test_run_eval_baseline_smoke(processed_synth, tmp_path):
    base = _load("run_baselines")
    ev = _load("run_eval")
    cfg = str(ROOT / "configs" / "baselines.yaml")
    common = ["--processed", str(processed_synth.path), "--config", cfg, "--models", "mostpop",
              "--out", str(tmp_path / "r")]  # fmt: skip
    base.main([*common, "--split", "val"])
    base.main([*common, "--split", "test"])
    (run_json,) = (tmp_path / "r").glob("mostpop-test-2*.json")
    path = ev.main(
        ["--processed", str(processed_synth.path), "--run", str(run_json),
         "--out", str(tmp_path / "r")]
    )  # fmt: skip
    _check(path, processed_synth, tmp_path / "r", "mostpop-test")
