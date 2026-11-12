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


class _Killed(Exception):
    pass


def test_resume_continues_step_count(processed_synth, tmp_path, monkeypatch):
    import hookrec.train.trainer as trainer_mod

    data = processed_synth
    straight = _fresh(tiny_cfg(), data)
    r = train(tiny_cfg(), data, straight, "test", tmp_path / "straight")
    assert (r.steps, r.stopped_reason) == (20, "max_steps")

    # the same run killed during step 11 (mid-epoch), then resumed from a read-only dir
    real_loss, calls = trainer_mod.sampled_softmax_loss, [0]

    def dying_loss(*a, **kw):
        calls[0] += 1
        if calls[0] == 11:
            raise _Killed
        return real_loss(*a, **kw)

    first = tmp_path / "first"
    monkeypatch.setattr(trainer_mod, "sampled_softmax_loss", dying_loss)
    with pytest.raises(_Killed):
        train(tiny_cfg(), data, _fresh(tiny_cfg(), data), "test", first)
    monkeypatch.setattr(trainer_mod, "sampled_softmax_loss", real_loss)
    assert _ckpts(first) == ["ckpt-00000005.pt", "ckpt-00000010.pt"]
    before = {p.name: p.stat().st_mtime_ns for p in first.iterdir()}
    resumed = _fresh(tiny_cfg(), data)
    # operational keys may change between sessions; the persisted horizon (20) beats 50
    cfg2 = tiny_cfg(**{"train.max_hours": 7.0, "train.max_steps": 50})
    r2 = train(cfg2, data, resumed, "test", tmp_path / "second", resume_dirs=[first])
    assert (r2.steps, r2.max_steps, r2.schedule_steps) == (20, 20, 20)
    assert r2.stopped_reason == "max_steps"
    assert {p.name: p.stat().st_mtime_ns for p in first.iterdir()} == before  # never written
    assert _ckpts(tmp_path / "second") == ["ckpt-00000015.pt", "ckpt-00000020.pt"]
    # exact continuation: optimizer, scheduler, sampler position and rng states restored
    for a, b in zip(straight.state_dict().values(), resumed.state_dict().values(), strict=True):
        assert torch.equal(a, b)
    assert math.isnan(r2.best_metric)
    assert (tmp_path / "second" / "model_final.pt").exists()


def test_resume_refuses_a_different_training_config(processed_synth, tmp_path):
    cfg = tiny_cfg(**{"train.max_steps": 5})
    train(cfg, processed_synth, _fresh(cfg, processed_synth), "test", tmp_path)
    other = tiny_cfg(**{"train.lr": 3e-3})
    with pytest.raises(ValueError, match="different config"):
        train(other, processed_synth, _fresh(other, processed_synth), "test", tmp_path)


def test_schedule_steps_sets_the_lr_horizon(processed_synth, tmp_path):
    from hookrec.train.checkpoint import load_latest
    from hookrec.train.trainer import warmup_cosine

    over = {"train.max_steps": 5, "train.schedule_steps": 100, "train.warmup_steps": 2}
    cfg = tiny_cfg(**over)
    r = train(cfg, processed_synth, _fresh(cfg, processed_synth), "test", tmp_path)
    assert (r.steps, r.schedule_steps) == (5, 100)
    payload = load_latest(tmp_path)
    assert (payload["max_steps"], payload["schedule_steps"]) == (5, 100)
    lr = payload["optimizer"]["param_groups"][0]["lr"]
    assert lr == pytest.approx(1e-3 * warmup_cosine(2, 100)(5))  # not annealed to 0 at step 5


def test_final_eval_after_last_checkpoint_is_checkpointed(processed_synth, tmp_path):
    from hookrec.train.checkpoint import load_latest

    cfg = tiny_cfg(**{"train.max_steps": 25})  # ckpt at 25 in the loop, final eval after it
    train(cfg, processed_synth, _fresh(cfg, processed_synth), "val", tmp_path)
    assert load_latest(tmp_path)["last_eval_step"] == 25


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


def _runner_args(processed_synth, out, *extra):
    over = [a for k, v in TINY.items() for a in ("--override", f"{k}={v}")]
    cfg = str(ROOT / "configs" / "sasrec.yaml")
    base = ["--processed", str(processed_synth.path), "--config", cfg, "--out", str(out)]
    return [*base, *over, *extra]


def _record(out, split):
    files = list(Path(out).glob(f"sasrec-{split}-*.json"))
    assert len(files) == 1, split
    return json.loads(files[0].read_text())


def test_run_train_smoke(processed_synth, tmp_path):
    mod = _load_runner()
    out = tmp_path / "results"
    mod.main(_runner_args(processed_synth, out, "--split", "val", "--ckpt", str(tmp_path / "cv")))
    mod.main(_runner_args(processed_synth, out, "--split", "test", "--ckpt", str(tmp_path / "ct")))
    keys = {
        "split", "processed_dir", "best_step", "steps", "stopped_reason", "wall_time_s",
        "steps_per_sec", "gpu_name", "gpu_hours", "metrics_cold", "n_cold_users", "n_eval_users",
    }  # fmt: skip
    recs = {}
    for split in ("val", "test"):
        recs[split] = _record(out, split)
        assert keys <= set(recs[split]["extra"])
        assert recs[split]["extra"]["split"] == split
        assert "ndcg@10" in recs[split]["metrics"]
        n_cold = len(getattr(processed_synth, f"truth_{split}_cold"))
        assert recs[split]["extra"]["n_cold_users"] == n_cold
    best_step = recs["val"]["extra"]["best_step"]
    assert recs["test"]["config"]["train"]["max_steps"] == best_step
    assert recs["test"]["config"]["train"]["schedule_steps"] == 20  # the val run's horizon
    assert recs["test"]["extra"]["steps"] == best_step
    assert (tmp_path / "ct" / "model_final.pt").exists()


def test_run_train_is_seeded(processed_synth, tmp_path):
    mod = _load_runner()
    for i in (1, 2):
        out, ck = tmp_path / f"out{i}", str(tmp_path / f"ck{i}")
        mod.main(_runner_args(processed_synth, out, "--split", "val", "--ckpt", ck))
    a, b = _record(tmp_path / "out1", "val"), _record(tmp_path / "out2", "val")
    assert a["extra"]["best_step"] == b["extra"]["best_step"]
    assert a["metrics"] == b["metrics"]


def test_run_train_test_resume_keeps_horizon_without_val_record(processed_synth, tmp_path):
    from hookrec.train.checkpoint import load_latest

    mod = _load_runner()
    out, ck = tmp_path / "results", str(tmp_path / "ct")
    mod.main(_runner_args(processed_synth, out, "--split", "val", "--ckpt", str(tmp_path / "cv")))
    best_step = _record(out, "val")["extra"]["best_step"]
    # session 1: horizon from the val record, stopped at once by max_hours=0
    args = _runner_args(processed_synth, out, "--split", "test", "--ckpt", ck)
    assert mod.main([*args, "--override", "train.max_hours=0"]) is None
    payload = load_latest(Path(ck))
    assert (payload["step"], payload["max_steps"], payload["schedule_steps"]) == (0, best_step, 20)
    # session 2: an --out without the val record; the checkpoint's horizon is used
    out2 = tmp_path / "results2"
    mod.main(_runner_args(processed_synth, out2, "--split", "test", "--ckpt", ck))
    rec = _record(out2, "test")
    assert rec["extra"]["steps"] == best_step
    assert rec["config"]["train"]["max_steps"] == best_step
    assert rec["config"]["train"]["schedule_steps"] == 20


def test_run_train_test_without_val_run_fails_unless_allowed(processed_synth, tmp_path):
    mod = _load_runner()
    ck = str(tmp_path / "c")
    args = _runner_args(processed_synth, tmp_path / "o", "--split", "test", "--ckpt", ck)
    with pytest.raises(SystemExit, match="--allow-no-val"):
        mod.main(args)
    mod.main([*args, "--allow-no-val"])
    assert _record(tmp_path / "o", "test")["extra"]["steps"] == 20


def test_best_val_run_matches_config_up_to_operational_keys(tmp_path):
    from hookrec.utils.results import save_run

    mod = _load_runner()
    cfg = tiny_cfg()
    other = copy.deepcopy(cfg)
    other["model"]["d_model"] = 32
    longer = copy.deepcopy(cfg)
    longer["train"].update(max_steps=999, num_workers=3, max_hours=1.0)
    rec = {"processed_dir": "d"}
    save_run(tmp_path, "sasrec-val", cfg, {"ndcg@10": 0.1}, {**rec, "best_step": 4})
    save_run(tmp_path, "sasrec-val", longer, {"ndcg@10": 0.2}, {**rec, "best_step": 8})
    save_run(tmp_path, "sasrec-val", other, {"ndcg@10": 0.9}, {**rec, "best_step": 1})
    save_run(tmp_path, "sasrec-val", cfg, {"ndcg@10": 0.9}, {"processed_dir": "x", "best_step": 2})
    assert mod.best_val_horizon(tmp_path, "sasrec", "d", cfg) == (8, 999)
    assert mod.best_val_horizon(tmp_path, "sasrec", "none", cfg) is None


def test_session_start_env_sets_max_hours_clock(processed_synth, tmp_path, monkeypatch):
    import time

    monkeypatch.setenv("HOOKREC_SESSION_START", str(time.time() - 10 * 3600))  # started 10 h ago
    cfg = tiny_cfg(**{"train.max_hours": 9.0})
    r = train(cfg, processed_synth, _fresh(cfg, processed_synth), "val", tmp_path)
    assert (r.stopped_reason, r.steps) == ("max_hours", 0)
    monkeypatch.setenv("HOOKREC_SESSION_START", str(time.time()))
    r = train(cfg, processed_synth, _fresh(cfg, processed_synth), "val", tmp_path / "b")
    assert r.stopped_reason != "max_hours"


def test_checkpoint_horizon_sidecar(processed_synth, tmp_path, monkeypatch):
    from hookrec.train import checkpoint as ck

    cfg = tiny_cfg(**{"train.max_steps": 10, "train.max_hours": 0.0})
    cfg["train"]["schedule_steps"] = 77
    train(cfg, processed_synth, _fresh(cfg, processed_synth), "val", tmp_path)
    h = ck.read_horizon([tmp_path])
    assert (h["max_steps"], h["schedule_steps"]) == (10, 77)
    assert sorted(p.name for p in tmp_path.glob("ckpt-*.json")) == ["ckpt-00000000.json"]
    # reading the horizon must not deserialize the checkpoint
    monkeypatch.setattr(ck.torch, "load", lambda *a, **k: pytest.fail("loaded checkpoint"))
    assert ck.read_horizon([tmp_path])["max_steps"] == 10
    # no sidecar (older checkpoint) -> None, so the caller can fall back
    for p in tmp_path.glob("ckpt-*.json"):
        p.unlink()
    assert ck.read_horizon([tmp_path]) is None
