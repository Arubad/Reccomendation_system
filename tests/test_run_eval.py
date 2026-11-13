import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

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


def _check(path, data, out, name, run_json):
    rec = json.loads(Path(path).read_text())
    assert rec["run_name"] == f"{name}-eval"
    ex = rec["extra"]
    assert ex["n_eval_users"] == len(data.truth_test)
    assert ex["n_cold_users"] == len(data.truth_test_cold)
    names = {"all", "organic", "served", "idless_items", "new_items", "warm_items", "users_low"}
    assert names <= set(ex["slices"]) and "cold_items" not in ex["slices"]
    assert ex["new_items_share"]["slice"] == ex["new_items_share"]["meta"]
    assert ex["slices"]["all"]["ndcg@10"] == rec["metrics"]["ndcg@10"]
    for k in (10, 100):
        for f in ("artist_gini", "artist_coverage", "long_tail_share"):
            assert 0.0 <= ex["fairness"][f"{f}@{k}"] <= 1.0
    for m in ("ndcg@10", "recall@100"):
        mean, lo, hi = ex["ci"][m]
        assert lo <= mean <= hi
    recorded = json.loads(Path(run_json).read_text())["metrics"]["ndcg@10"]
    assert abs(rec["metrics"]["ndcg@10"] - recorded) < 1e-6  # drift check held
    with np.load(Path(out) / f"{name}-{data.cfg['size']}-eval-peruser.npz") as z:
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
    _check(path, processed_synth, tmp_path / "r", "sasrec-test", run_path)
    # the eval record must not leak into the headline table
    from hookrec.eval.report import write_readme_tables

    text = write_readme_tables(tmp_path / "r", tmp_path / "res.md").read_text()
    assert "sasrec-test-eval" not in text.split("## Slices")[0]
    assert "### sasrec-test" in text


def test_run_eval_masks_like_run_train_for_listen_plus_catalog(processed_synth, tmp_path):
    """R32: run_eval rebuilds the same catalog mask (the drift check then holds) and records it."""
    import hookrec.train.scoring as scoring_mod

    train = _load("run_train")
    ev = _load("run_eval")
    tiny = {**TINY, "loss.type": "bce", "loss.neg_catalog": "listen_plus"}
    over = [a for k, v in tiny.items() for a in ("--override", f"{k}={v}")]
    cfg = str(ROOT / "configs" / "sasrec.yaml")
    run_path = train.main(
        ["--processed", str(processed_synth.path), "--config", cfg, "--split", "test",
         "--allow-no-val", "--ckpt", str(tmp_path / "ck"), "--out", str(tmp_path / "r"), *over]
    )  # fmt: skip
    assert json.loads(Path(run_path).read_text())["extra"]["eval_catalog"] == "listen_plus"
    seen = []
    real_init = scoring_mod.SeqRecommender.__init__

    def init(self, *a, **kw):
        seen.append(kw.get("item_mask"))
        real_init(self, *a, **kw)

    scoring_mod.SeqRecommender.__init__ = init
    try:
        path = ev.main(
            ["--processed", str(processed_synth.path), "--run", str(run_path),
             "--ckpt", str(tmp_path / "ck"), "--out", str(tmp_path / "r")]
        )  # fmt: skip
    finally:
        scoring_mod.SeqRecommender.__init__ = real_init
    assert seen and all(m is not None for m in seen)
    assert json.loads(Path(path).read_text())["extra"]["eval_catalog"] == "listen_plus"


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
    _check(path, processed_synth, tmp_path / "r", "mostpop-test", run_json)


def test_run_eval_drift_and_dir_mismatch(processed_synth, tmp_path):
    base = _load("run_baselines")
    ev = _load("run_eval")
    cfg = str(ROOT / "configs" / "baselines.yaml")
    common = ["--processed", str(processed_synth.path), "--config", cfg, "--models", "mostpop",
              "--out", str(tmp_path / "r")]  # fmt: skip
    base.main([*common, "--split", "val"])
    base.main([*common, "--split", "test"])
    (run_json,) = (tmp_path / "r").glob("mostpop-test-2*.json")
    rec = json.loads(run_json.read_text())
    args = ["--processed", str(processed_synth.path), "--out", str(tmp_path / "o")]

    drifted = tmp_path / "drift.json"
    drifted.write_text(json.dumps({**rec, "metrics": {**rec["metrics"], "ndcg@10": 0.5}}))
    with pytest.raises(RuntimeError, match="reproduce"):
        ev.main([*args, "--run", str(drifted)])
    assert ev.main([*args, "--run", str(drifted), "--no-drift-check"]).exists()

    other = tmp_path / "other.json"
    other.write_text(json.dumps({**rec, "extra": {**rec["extra"], "processed_dir": "elsewhere"}}))
    with pytest.raises(SystemExit):
        ev.main([*args, "--run", str(other)])


def test_run_eval_ials_is_deterministic(processed_synth, tmp_path):
    base = _load("run_baselines")
    ev = _load("run_eval")
    cfg = str(ROOT / "configs" / "baselines.yaml")
    common = ["--processed", str(processed_synth.path), "--config", cfg, "--models", "ials",
              "--quick", "--out", str(tmp_path / "r")]  # fmt: skip
    base.main([*common, "--split", "val"])
    base.main([*common, "--split", "test"])
    (run_json,) = (tmp_path / "r").glob("ials-test-2*.json")
    path = ev.main(["--processed", str(processed_synth.path), "--run", str(run_json),
                    "--out", str(tmp_path / "r")])  # fmt: skip
    assert path.exists()  # drift check (1e-4) passed on the refit


def test_check_drift_tolerances_and_missing_metric(caplog):
    ev = _load("run_eval")
    neural = {"run_name": "n-test", "config": {"model": {}}, "metrics": {"ndcg@10": 0.1}}
    base = {"run_name": "mostpop-test", "config": {}, "metrics": {"ndcg@10": 0.1}}
    with caplog.at_level("WARNING"):
        ev.check_drift(neural, 0.1 + 5e-4)  # fp16-GPU vs CPU: within 1e-3
    assert "drift" in caplog.text
    with pytest.raises(RuntimeError, match="reproduce"):
        ev.check_drift(neural, 0.1 + 5e-3)
    with pytest.raises(RuntimeError, match="reproduce"):
        ev.check_drift(base, 0.1 + 5e-4)  # baselines keep 1e-4
    with pytest.raises(RuntimeError, match="ndcg@10"):
        ev.check_drift({"run_name": "x", "config": {}, "metrics": {}}, 0.1)
