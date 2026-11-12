import copy
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from hookrec.eval.protocol import evaluate
from hookrec.models.base import build_model
from hookrec.models.popularity import MostPop
from hookrec.train.scoring import SeqRecommender
from hookrec.train.trainer import TrainResult, hard_negatives, train
from hookrec.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
TINY = {
    "model.d_model": 16,
    "model.n_layers": 1,
    "model.max_len": 20,
    "model.dropout": 0.1,
    "train.batch_size": 4,
    "train.lr": 1e-3,
    "train.warmup_steps": 100,
    "train.max_steps": 20,
    "train.eval_every_steps": 10,
    "train.ckpt_every_steps": 5,
    "train.num_workers": 0,
    "train.log_every_steps": 10,
    "train.device": "cpu",
    "loss.n_uniform_negatives": 32,
    "loss.n_in_batch_negatives": 32,
}


def tiny_cfg(**over):
    return load_config(ROOT / "configs" / "sasrec.yaml", {**TINY, **over})


def _fresh(cfg, data):
    torch.manual_seed(0)
    return build_model(cfg, data)


def _ckpts(d):
    return sorted(p.name for p in Path(d).glob("ckpt-*.pt"))


def test_resume_continues_step_count(processed_synth, tmp_path):
    data = processed_synth
    straight = _fresh(tiny_cfg(), data)
    r = train(tiny_cfg(), data, straight, "test", tmp_path / "straight")
    assert (r.steps, r.stopped_reason) == (20, "max_steps")

    # same run in two sessions: stop at 10 (mid-epoch), resume to 20 from a read-only dir
    first = tmp_path / "first"
    r1 = train(tiny_cfg(**{"train.max_steps": 10}), data, _fresh(tiny_cfg(), data), "test", first)
    assert r1.steps == 10 and _ckpts(first) == ["ckpt-00000005.pt", "ckpt-00000010.pt"]
    before = {p.name: p.stat().st_mtime_ns for p in first.iterdir()}
    resumed = _fresh(tiny_cfg(), data)
    r2 = train(tiny_cfg(), data, resumed, "test", tmp_path / "second", resume_dirs=[first])
    assert r2.steps == 20 and r2.stopped_reason == "max_steps"
    assert {p.name: p.stat().st_mtime_ns for p in first.iterdir()} == before  # never written
    assert _ckpts(tmp_path / "second") == ["ckpt-00000015.pt", "ckpt-00000020.pt"]
    # exact continuation: optimizer, scheduler, sampler position and rng states restored
    for a, b in zip(straight.state_dict().values(), resumed.state_dict().values(), strict=True):
        assert torch.equal(a, b)
    assert math.isnan(r2.best_metric)
    assert (tmp_path / "second" / "model_final.pt").exists()


def test_max_hours_zero_stops_and_checkpoints(processed_synth, tmp_path):
    cfg = tiny_cfg(**{"train.max_hours": 0.0})
    r = train(cfg, processed_synth, _fresh(cfg, processed_synth), "val", tmp_path)
    assert isinstance(r, TrainResult)
    assert (r.stopped_reason, r.steps) == ("max_hours", 0)
    assert _ckpts(tmp_path) == ["ckpt-00000000.pt"]
    assert not (tmp_path / "model_final.pt").exists()  # unfinished: resume next session


def test_val_early_stopping_and_best_checkpoint(processed_synth, tmp_path):
    # lr=0 -> the metric never improves after the first eval -> converged after `patience` evals
    cfg = tiny_cfg(**{"train.lr": 0.0, "train.patience": 2, "train.max_steps": 1000})
    r = train(cfg, processed_synth, _fresh(cfg, processed_synth), "val", tmp_path)
    assert r.stopped_reason == "converged"
    assert r.best_step == 10 and r.steps == 30
    best = torch.load(tmp_path / "best.pt", weights_only=False)
    assert best["step"] == 10 and best["metric"] == pytest.approx(r.best_metric)
    # re-launching a converged run (e.g. a chained Kaggle session) must not train further
    again = train(cfg, processed_synth, _fresh(cfg, processed_synth), "val", tmp_path)
    assert (again.stopped_reason, again.steps, again.best_step) == ("converged", 30, 10)


def test_hard_negatives_are_most_recent_dislikes_left_padded():
    from hookrec.data.label import DISLIKE, LISTEN_FULL

    items = torch.tensor([[0, 5, 6, 7, 8, 9], [0, 0, 0, 0, 3, 4]])
    tokens = torch.tensor(
        [[0, DISLIKE, DISLIKE, LISTEN_FULL, DISLIKE, DISLIKE], [0, 0, 0, 0, DISLIKE, LISTEN_FULL]]
    )
    out = hard_negatives(items, tokens, n_hard=2)
    assert out.tolist() == [[8, 9], [0, 3]]


def test_logq_and_hard_negative_losses_are_finite(processed_synth, tmp_path):
    cfg = tiny_cfg(**{"loss.logq": True, "loss.dislike_hard_negatives": True, "train.max_steps": 6})
    r = train(cfg, processed_synth, _fresh(cfg, processed_synth), "test", tmp_path)
    assert r.steps == 6 and np.isfinite(r.final_loss)


def test_train_overfits_synth(processed_synth, tmp_path):
    data = processed_synth
    over = {
        "train.batch_size": 16,
        "train.lr": 5e-3,
        "train.warmup_steps": 20,
        "train.max_steps": 300,
        "train.eval_every_steps": 50,
        "train.ckpt_every_steps": 100,
        "train.patience": 100,
    }
    cfg = tiny_cfg(**over)
    model = _fresh(cfg, data)
    r = train(cfg, data, model, "val", tmp_path)
    assert r.steps == 300
    seqs = data.seqs_val_train
    n = data.vocab.n_items
    pop = MostPop().fit(seqs, n)
    rec = SeqRecommender(model, seqs, 20, torch.device("cpu"), pop)
    ours = evaluate(rec, data.truth_val, n).metrics["ndcg@10"]
    base = evaluate(pop, data.truth_val, n).metrics["ndcg@10"]
    assert r.best_metric >= ours - 1e-9
    assert ours > base, (ours, base)


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_train", ROOT / "scripts" / "run_train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_train_smoke(processed_synth, tmp_path):
    mod = _load_runner()
    out = tmp_path / "results"
    over = [a for k, v in TINY.items() for a in ("--override", f"{k}={v}")]
    cfg = str(ROOT / "configs" / "sasrec.yaml")
    base = ["--processed", str(processed_synth.path), "--config", cfg, "--out", str(out), *over]
    mod.main([*base, "--split", "val", "--ckpt", str(tmp_path / "ck_val")])
    mod.main([*base, "--split", "test", "--ckpt", str(tmp_path / "ck_test")])
    keys = {
        "split", "processed_dir", "best_step", "steps", "stopped_reason", "wall_time_s",
        "steps_per_sec", "gpu_name", "gpu_hours", "metrics_cold", "n_cold_users", "n_eval_users",
    }  # fmt: skip
    recs = {}
    for split in ("val", "test"):
        files = list(out.glob(f"sasrec-{split}-*.json"))
        assert len(files) == 1, split
        recs[split] = json.loads(files[0].read_text())
        assert keys <= set(recs[split]["extra"])
        assert recs[split]["extra"]["split"] == split
        assert "ndcg@10" in recs[split]["metrics"]
        assert recs[split]["extra"]["n_cold_users"] == len(
            getattr(processed_synth, f"truth_{split}_cold")
        )
    best_step = recs["val"]["extra"]["best_step"]
    assert recs["test"]["config"]["train"]["max_steps"] == best_step
    assert recs["test"]["extra"]["steps"] == best_step
    assert (tmp_path / "ck_test" / "model_final.pt").exists()


def test_best_val_run_matches_config_without_max_steps(tmp_path):
    from hookrec.utils.results import save_run

    mod = _load_runner()
    cfg = tiny_cfg()
    other = copy.deepcopy(cfg)
    other["model"]["d_model"] = 32
    longer = copy.deepcopy(cfg)
    longer["train"]["max_steps"] = 999
    save_run(tmp_path, "sasrec-val", cfg, {"ndcg@10": 0.1}, {"processed_dir": "d", "best_step": 4})
    save_run(
        tmp_path, "sasrec-val", longer, {"ndcg@10": 0.2}, {"processed_dir": "d", "best_step": 8}
    )
    save_run(
        tmp_path, "sasrec-val", other, {"ndcg@10": 0.9}, {"processed_dir": "d", "best_step": 1}
    )
    save_run(tmp_path, "sasrec-val", cfg, {"ndcg@10": 0.9}, {"processed_dir": "x", "best_step": 2})
    assert mod.best_val_step(tmp_path, "sasrec", "d", cfg) == 8
    assert mod.best_val_step(tmp_path, "sasrec", "none", cfg) is None
