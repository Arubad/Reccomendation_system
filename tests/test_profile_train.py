"""scripts/profile_train.py: runs on CPU, emits every key, and its step equals the trainer's."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from hookrec.models.base import build_model
from hookrec.train.trainer import train
from hookrec.utils.config import load_config
from hookrec.utils.seed import set_seed

ROOT = Path(__file__).resolve().parents[1]
TINY = [
    "model.d_model=16", "model.n_layers=1", "model.n_heads=1", "model.max_len=16",
    "train.batch_size=4", "train.num_workers=0", "train.device=cpu", "train.warmup_steps=2",
    "loss.n_uniform_negatives=16", "loss.n_in_batch_negatives=8",
]  # fmt: skip
HOOKSEQ = ["model.like_head=true", "model.use_time_gap=true", "loss.like_weight=0.2"]


def _tool():
    spec = importlib.util.spec_from_file_location(
        "profile_train", ROOT / "scripts" / "profile_train.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # its dataclass resolves annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


def test_profile_tool_runs_and_emits_every_key(processed_synth, tmp_path):
    out = tmp_path / "prof"
    argv = ["--processed", str(processed_synth.path), "--config", str(ROOT / "configs/sasrec.yaml"),
            "--steps", "3", "--warmup", "2", "--throughput-steps", "2", "--profile-steps", "2",
            "--loader-batches", "3", "--loader-workers", "0", "--out", str(out)]  # fmt: skip
    for o in TINY:
        argv += ["--override", o]
    _tool().main(argv)
    r = json.loads((out / "profile.json").read_text())
    assert (out / "summary.txt").read_text().startswith("profile_train: sasrec")
    for k in ("config", "env", "setup_s", "shapes", "warmup", "phased", "throughput", "memory",
              "profiler", "eval", "loader", "wall_s"):  # fmt: skip
        assert k in r, k
    phases = r["phased"]["phases"]
    assert set(phases) == {
        "data_wait", "h2d", "encode", "negatives_item_repr", "loss_fwd", "backward", "optimizer"
    }  # fmt: skip
    assert all(v["mean_s"] >= 0 for v in phases.values())
    assert r["phased"]["steps"] == 3 and len(r["warmup"]["step_s"]) == 2
    assert 0 < r["phased"]["gpu_util_proxy"] <= 1 and r["phased"]["steps_per_sec"] > 0
    assert r["throughput"]["steps"] == 2 and r["throughput"]["steps_per_sec"] > 0
    assert r["shapes"]["M_mean"] > 0 and r["shapes"]["S_mean"] == 24
    assert len(r["profiler"]["top25_by_cpu_time"]) > 0
    assert len(r["profiler"]["top25_by_device_time"]) > 0
    assert {"hk::encode", "hk::loss_fwd", "hk::backward"} <= set(r["profiler"]["phase_ranges"])
    assert r["eval"]["eval_s"] > 0 and r["eval"]["n_users"] > 0
    assert r["loader"]["nw0_pin0"]["batches_per_s"] > 0
    assert "skipped" in r["loader"]["nw0_pin1"]  # pin_memory is CUDA-only
    for k in ("pin_memory", "persistent_workers", "prefetch_factor", "num_workers"):
        assert k in r["config"]
    assert r["memory"] is None  # CPU


@pytest.mark.parametrize(
    "config,extra",
    [
        ("sasrec.yaml", []),
        ("hookseq.yaml", HOOKSEQ),
        ("hookseq.yaml", [*HOOKSEQ, "loss.max_rows=6", "loss.chunk_rows=5"]),
        ("sasrec_bce.yaml", ["train.warmup_steps=0"]),
        ("ablations/a8_full_all_ids.yaml", HOOKSEQ),  # F5: rebuilt ID table (id_min_count: 1)
    ],
    ids=["sasrec", "hookseq", "hookseq_max_rows", "sasrec_bce", "a8_all_ids"],
)
def test_profiled_step_matches_the_trainer(processed_synth, tmp_path, config, extra):
    """The tool's phase-annotated step is the trainer's step: same weights after k steps."""
    tool = _tool()
    data, k = processed_synth, 14  # crosses an epoch boundary on the synth data
    over = tool.parse_overrides(
        [*TINY, *extra, f"train.max_steps={k}", "train.ckpt_every_steps=100"]
    )
    cfg = load_config(ROOT / "configs" / config, over)

    set_seed(cfg["train"]["seed"])
    ref = build_model(cfg, data)
    res = train(cfg, data, ref, "test", tmp_path / "ck")
    assert res.steps == k

    set_seed(cfg["train"]["seed"])
    model = build_model(cfg, data)
    ctx = tool.setup(cfg, data, model, "test")
    assert ctx.sampler.batches_per_epoch < k
    _, _ = tool.run_steps(ctx, tool.batches(ctx), k, "timed")
    for (name, a), b in zip(ref.state_dict().items(), model.state_dict().values(), strict=True):
        assert torch.equal(a, b), name
    assert ctx.sched.last_epoch == k
