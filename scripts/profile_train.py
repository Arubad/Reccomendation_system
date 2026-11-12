"""Profile the sequence-model training step phase by phase (a diagnostic; trains nothing to keep).

python scripts/profile_train.py --processed DIR --config configs/sasrec.yaml \
    [--override train.batch_size=128 ...] --steps 300 --warmup 20 --out DIR \
    [--split val] [--throughput-steps N] [--profile-steps 20] [--loader-batches 50] \
    [--loader-workers 0,2,4] [--no-eval]

It sets up training exactly as `hookrec.train.trainer.train` does (same seeds, model, optimizer,
scheduler, GradScaler, negative sampler, dataset, sampler and DataLoader settings, no resume)
and runs `train_step` below, a phase-annotated copy of the trainer's step (tests/
test_profile_train.py checks that both give bit-identical weights). Then, in order:

1. warmup: W steps, each phase timed (per-step times kept, to expose the cold start);
2. phased: N steps with `torch.cuda.synchronize()` at every phase boundary (only here):
   data_wait, h2d, encode, negatives_item_repr, loss_fwd, backward, optimizer;
3. throughput: steps without any extra sync (the real steps/s), GPU utilisation via NVML
   when available;
4. torch.profiler over --profile-steps steps (phases as record_function ranges);
5. one periodic eval, timed as the trainer runs it (`SeqRecommender` + `evaluate`, ndcg@10);
6. DataLoader-only throughput for each num_workers in --loader-workers x pin_memory on/off.

Writes `profile.json` and `summary.txt` to --out, and prints the summary.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import platform
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_train import parse_overrides

from hookrec.data.processed import load_processed
from hookrec.eval.protocol import evaluate
from hookrec.models.base import (
    aggregates_for,
    build_model,
    history_for,
    input_filter_of,
)
from hookrec.models.losses import make_log_q, sample_negatives, sampled_softmax_loss
from hookrec.models.popularity import mostpop_fallback
from hookrec.train.datasets import (
    SeqTrainDataset,
    session_gap_from,
    window_sampling_from,
)
from hookrec.train.scoring import SeqRecommender
from hookrec.train.trainer import (
    _EpochIndexed,
    _eval_truth,
    _ResumableSampler,
    device_name,
    hard_negatives,
    param_groups,
    warmup_cosine,
)
from hookrec.utils.config import load_config
from hookrec.utils.seed import set_seed

log = logging.getLogger("profile_train")

PHASES = (
    "data_wait", "h2d", "encode", "negatives_item_repr", "loss_fwd", "backward", "optimizer",
)  # fmt: skip
ENV_VARS = (
    "CUDA_MODULE_LOADING", "CUDA_CACHE_MAXSIZE", "CUDA_CACHE_PATH", "CUDA_CACHE_DISABLE",
    "PYTORCH_CUDA_ALLOC_CONF", "OMP_NUM_THREADS",
)  # fmt: skip
COMPUTE_PHASES = ("encode", "negatives_item_repr", "loss_fwd", "backward", "optimizer")


# --- phase timer ---------------------------------------------------------------------------


class Phases:
    """Phase boundaries: `start(name)` closes the open phase and opens `name`.

    mode "timed": synchronize the device at each boundary and record the elapsed seconds;
    "annotate": open a `record_function("hk::<name>")` range (for torch.profiler);
    "off": nothing (the unsynchronised throughput run).
    """

    def __init__(self, mode: str, device: torch.device):
        self.mode, self.device = mode, device
        self.cur: str | None = None
        self.t0 = 0.0
        self.rf = None
        self.step_times: dict[str, float] = {}

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def start(self, name: str | None) -> None:
        if self.mode == "timed":
            self._sync()
            now = time.perf_counter()
            if self.cur is not None:
                self.step_times[self.cur] = self.step_times.get(self.cur, 0.0) + now - self.t0
            self.t0 = now
        elif self.mode == "annotate":
            if self.rf is not None:
                self.rf.__exit__(None, None, None)
                self.rf = None
            if name is not None:
                self.rf = torch.profiler.record_function(f"hk::{name}")
                self.rf.__enter__()
        self.cur = name

    def stop(self) -> None:
        self.start(None)

    def take(self) -> dict[str, float]:
        out, self.step_times = self.step_times, {}
        return out


# --- training context: the trainer's setup -------------------------------------------------


@dataclass
class Ctx:
    cfg: dict
    model: nn.Module
    device: torch.device
    amp: bool
    opt: torch.optim.Optimizer
    sched: LambdaLR
    scaler: torch.amp.GradScaler
    gen: torch.Generator
    log_q: Tensor | None
    n_items: int
    n_ib: int
    n_uni: int
    n_hard: int
    like_weight: float
    max_grad_norm: float
    temperature: float
    seqs: object
    ds: _EpochIndexed
    sampler: _ResumableSampler
    loader: DataLoader
    session_gap: float
    fallback: object
    eval_truth: dict
    epoch: int = 0
    epoch_batch: int = 0
    parts: dict = field(default_factory=lambda: {"rank": float("nan"), "like": 0.0})
    stats: dict = field(default_factory=lambda: {"M": [], "S": []})


def make_loader(ctx_like, num_workers: int, pin_memory: bool) -> DataLoader:
    """The trainer's DataLoader (only num_workers / pin_memory may differ)."""
    return DataLoader(
        ctx_like["ds"],
        batch_size=ctx_like["batch_size"],
        sampler=ctx_like["sampler"],
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        generator=torch.Generator().manual_seed(ctx_like["seed"]),
    )


def setup(cfg: dict, data, model: nn.Module, split: str) -> Ctx:
    """Mirror of `trainer.train` up to the training loop, for a fresh (non-resumed) run."""
    tc, lc, max_len = cfg["train"], cfg["loss"], cfg["model"]["max_len"]
    device = torch.device(tc.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = device.type == "cuda" and bool(tc.get("amp", True))
    set_seed(tc["seed"])
    input_filter = input_filter_of(cfg)
    seqs, _ = history_for(data, split, input_filter)
    n_items = data.vocab.n_items
    n_ib, n_uni = lc.get("n_in_batch_negatives", 4096), lc["n_uniform_negatives"]
    n_hard = lc.get("n_hard", 4) if lc.get("dislike_hard_negatives") else 0
    session_gap = session_gap_from(cfg, data.cfg)
    like_weight = lc.get("like_weight", 0.0) if getattr(model, "like_head", False) else 0.0
    schedule_steps = tc.get("schedule_steps") or tc["max_steps"]

    model.to(device).train()
    opt = torch.optim.AdamW(param_groups(model, tc["weight_decay"]), lr=tc["lr"])
    sched = LambdaLR(opt, warmup_cosine(tc["warmup_steps"], schedule_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    gen = torch.Generator().manual_seed(tc["seed"])
    fallback = mostpop_fallback(data, split)
    log_q = None
    if lc.get("logq"):
        agg = aggregates_for(data, split)
        pop = agg.pop_lp if input_filter == "listen_plus" else agg.pop_pos
        log_q = make_log_q(pop, lc.get("mix_uniform", 0.0), n_ib, n_uni).to(device)
    eval_truth = _eval_truth(data.truth_val, tc.get("eval_max_users"), tc["seed"])

    ds = _EpochIndexed(
        SeqTrainDataset(seqs, max_len, tc["seed"], session_gap, window_sampling_from(cfg))
    )
    sampler = _ResumableSampler(len(seqs), tc["batch_size"], tc["seed"])
    if sampler.batches_per_epoch == 0:
        raise ValueError(f"{len(seqs)} sequences cannot fill one batch of {tc['batch_size']}")
    spec = {"ds": ds, "sampler": sampler, "batch_size": tc["batch_size"], "seed": tc["seed"]}
    loader = make_loader(spec, tc.get("num_workers", 0), device.type == "cuda")
    return Ctx(
        cfg=cfg, model=model, device=device, amp=amp, opt=opt, sched=sched, scaler=scaler,
        gen=gen, log_q=log_q, n_items=n_items, n_ib=n_ib, n_uni=n_uni, n_hard=n_hard,
        like_weight=like_weight, max_grad_norm=tc["max_grad_norm"],
        temperature=lc.get("temperature", 1.0), seqs=seqs, ds=ds, sampler=sampler,
        loader=loader, session_gap=session_gap, fallback=fallback, eval_truth=eval_truth,
    )  # fmt: skip


def batches(ctx: Ctx):
    """Endless batches with the trainer's epoch handling (a new loader iterator per epoch)."""
    while True:
        ctx.sampler.set_position(ctx.epoch, ctx.epoch_batch)
        for b in ctx.loader:
            ctx.epoch_batch += 1
            yield b
        ctx.epoch += 1
        ctx.epoch_batch = 0


def train_step(ctx: Ctx, batch: dict[str, Tensor], ph: Phases) -> Tensor:
    """`trainer.train`'s `train_step`, statement for statement, with phase boundaries."""
    model, device, amp, gen = ctx.model, ctx.device, ctx.amp, ctx.gen
    n_items, n_uni, n_ib, n_hard = ctx.n_items, ctx.n_uni, ctx.n_ib, ctx.n_hard
    opt, scaler, sched = ctx.opt, ctx.scaler, ctx.sched
    ph.start("h2d")
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
        ph.start("encode")
        hidden = model.encode(batch)
        ph.start("negatives_item_repr")
        valid = batch["target"] != 0
        user_vecs, pos_items = hidden[valid], batch["target"][valid]
        neg = [sample_negatives(n_items, n_uni, gen).to(device)]
        if n_ib and len(pos_items):
            neg.insert(
                0, pos_items[torch.randint(len(pos_items), (n_ib,), generator=gen).to(device)]
            )
        neg_items = torch.cat(neg)
        pos_vecs, neg_vecs = model.item_repr(pos_items), model.item_repr(neg_items)
        like_y = batch["target_is_like"][valid]
        hard_items = hard_vecs = None
        if n_hard:
            rows = valid.nonzero()[:, 0]
            hard_items = hard_negatives(batch["items"], batch["tokens"], n_hard)[rows]
            hard_vecs = model.item_repr(hard_items).float()
    ctx.stats["M"].append(len(pos_items))
    ctx.stats["S"].append(len(neg_items) + (n_hard or 0))
    ph.start("loss_fwd")
    with torch.autocast(device.type, enabled=False):
        out = sampled_softmax_loss(
            user_vecs.float(), pos_items, pos_vecs.float(), neg_items, neg_vecs.float(),
            ctx.log_q, ctx.temperature, hard_neg_vecs=hard_vecs, hard_neg_items=hard_items,
        )  # fmt: skip
        ctx.parts["rank"] = out.detach()
        if ctx.like_weight:
            logits = model.like_logit(user_vecs.float(), pos_vecs.float())
            like_loss = F.binary_cross_entropy_with_logits(
                logits, like_y.float(), reduction="sum"
            ) / max(len(like_y), 1)
            ctx.parts["like"] = like_loss.detach()
            out = out + ctx.like_weight * like_loss
    ph.start("backward")
    opt.zero_grad(set_to_none=True)
    scaler.scale(out).backward()
    ph.start("optimizer")
    scaler.unscale_(opt)
    nn.utils.clip_grad_norm_(model.parameters(), ctx.max_grad_norm)
    scale_before = scaler.get_scale()
    scaler.step(opt)
    scaler.update()
    if scaler.get_scale() >= scale_before:
        sched.step()
    ph.stop()
    return out.detach()


def run_steps(ctx: Ctx, it, n: int, mode: str) -> tuple[list[dict[str, float]], float]:
    """n steps; per-step phase seconds (empty dicts unless timed) and the wall time."""
    ph = Phases(mode, ctx.device)
    per_step = []
    t0 = time.perf_counter()
    for _ in range(n):
        ph.start("data_wait")
        batch = next(it)
        train_step(ctx, batch, ph)
        per_step.append(ph.take())
    if ctx.device.type == "cuda":
        torch.cuda.synchronize(ctx.device)
    return per_step, time.perf_counter() - t0


# --- reporting helpers ---------------------------------------------------------------------


def _phase_stats(per_step: list[dict[str, float]]) -> dict:
    out = {}
    for p in PHASES:
        v = np.array([s.get(p, 0.0) for s in per_step])
        out[p] = {
            "mean_s": float(v.mean()) if len(v) else float("nan"),
            "p50_s": float(np.median(v)) if len(v) else float("nan"),
            "p95_s": float(np.percentile(v, 95)) if len(v) else float("nan"),
            "max_s": float(v.max()) if len(v) else float("nan"),
        }
    total = sum(out[p]["mean_s"] for p in PHASES)
    compute = sum(out[p]["mean_s"] for p in COMPUTE_PHASES)
    for p in PHASES:
        out[p]["share"] = out[p]["mean_s"] / total if total > 0 else float("nan")
    return {
        "phases": out,
        "step_mean_s": total,
        "steps_per_sec": 1.0 / total if total > 0 else float("nan"),
        "gpu_util_proxy": compute / total if total > 0 else float("nan"),
    }


def _memory(device: torch.device) -> dict | None:
    if device.type != "cuda":
        return None
    s = torch.cuda.memory_stats(device)
    return {
        "max_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
        "max_reserved_gb": torch.cuda.max_memory_reserved(device) / 1e9,
        "total_gb": torch.cuda.get_device_properties(device).total_memory / 1e9,
        "num_alloc_retries": int(s.get("num_alloc_retries", 0)),
        "num_ooms": int(s.get("num_ooms", 0)),
        "num_device_alloc": int(s.get("num_device_alloc", 0)),
        "num_device_free": int(s.get("num_device_free", 0)),
    }


def _env(device: torch.device) -> dict:
    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": device_name(device),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "env": {k: os.environ.get(k) for k in ENV_VARS},
        "capability": None,
        "arch_list": None,
        "native_sass_for_device": None,
    }
    if device.type == "cuda":
        cap = torch.cuda.get_device_capability(device)
        arch = torch.cuda.get_arch_list()
        env.update(
            capability=f"sm_{cap[0]}{cap[1]}",
            arch_list=arch,
            # no native SASS => every kernel is JIT-compiled from PTX on first use (cold start)
            native_sass_for_device=f"sm_{cap[0]}{cap[1]}" in arch,
        )
    return env


class _GpuUtil:
    """Background NVML sampler of GPU utilisation (%); a no-op without CUDA or pynvml."""

    def __init__(self, device: torch.device, period: float = 0.1):
        self.samples: list[float] = []
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread = None
        self.device, self.period = device, period

    def __enter__(self):
        if self.device.type != "cuda":
            self.error = "no cuda"
            return self
        try:
            torch.cuda.utilization(self.device)
        except Exception as e:  # noqa: BLE001  (pynvml missing, NVML errors: report, never fail)
            self.error = f"{type(e).__name__}: {e}"
            return self

        def loop():
            while not self._stop.is_set():
                try:
                    self.samples.append(float(torch.cuda.utilization(self.device)))
                except Exception:  # noqa: BLE001  (a diagnostic must not kill the run)
                    return
                self._stop.wait(self.period)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def result(self) -> dict:
        if not self.samples:
            return {"mean_pct": None, "n_samples": 0, "error": self.error}
        return {"mean_pct": float(np.mean(self.samples)), "n_samples": len(self.samples),
                "error": None}  # fmt: skip


def _attr(e, *names, default=0.0) -> float:
    for n in names:
        if hasattr(e, n):
            return float(getattr(e, n))
    return default


def run_profiler(ctx: Ctx, it, n: int) -> dict:
    acts = [torch.profiler.ProfilerActivity.CPU]
    if ctx.device.type == "cuda":
        acts.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=acts) as prof:
        _, wall = run_steps(ctx, it, n, "annotate")
    ka = prof.key_averages()
    rows = []
    for e in ka:
        rows.append(
            {
                "name": e.key,
                "count": int(e.count),
                "cpu_time_total_us": _attr(e, "cpu_time_total"),
                "self_cpu_time_total_us": _attr(e, "self_cpu_time_total"),
                "device_time_total_us": _attr(e, "device_time_total", "cuda_time_total"),
                "self_device_time_total_us": _attr(
                    e, "self_device_time_total", "self_cuda_time_total"
                ),
            }
        )
    by_dev = sorted(rows, key=lambda r: -r["self_device_time_total_us"])[:25]
    by_cpu = sorted(rows, key=lambda r: -r["self_cpu_time_total_us"])[:25]
    phases = {r["name"]: r for r in rows if r["name"].startswith("hk::")}
    dev_key = "self_device_time_total" if ctx.device.type == "cuda" else "self_cpu_time_total"
    tables = {
        "by_device_time": ka.table(sort_by=dev_key, row_limit=25),
        "by_cpu_time": ka.table(sort_by="self_cpu_time_total", row_limit=25),
    }
    return {
        "steps": n,
        "wall_s": wall,
        "top25_by_device_time": by_dev,
        "top25_by_cpu_time": by_cpu,
        "phase_ranges": phases,
        "tables": tables,
    }


def timed_eval(ctx: Ctx) -> dict:
    """One periodic eval, built exactly as `trainer.train.run_eval` builds it."""
    tc = ctx.cfg["train"]

    def sync():
        if ctx.device.type == "cuda":
            torch.cuda.synchronize(ctx.device)

    sync()
    t0 = time.perf_counter()
    was = ctx.model.training
    ctx.model.eval()
    with torch.no_grad():
        ctx.model.all_item_repr()
    ctx.model.train(was)
    sync()
    t_repr = time.perf_counter() - t0
    rec = SeqRecommender(
        ctx.model, ctx.seqs, ctx.cfg["model"]["max_len"], ctx.device, ctx.fallback,
        tc.get("eval_batch_size", 1024), session_gap=ctx.session_gap,
        amp=bool(tc.get("amp", True)),
    )  # fmt: skip
    t0 = time.perf_counter()
    metric = evaluate(rec, ctx.eval_truth, ctx.n_items, ks=(10,)).metrics["ndcg@10"]
    sync()
    return {
        "eval_s": time.perf_counter() - t0,
        "all_item_repr_s": t_repr,
        "n_users": len(ctx.eval_truth),
        "n_catalog": int(getattr(ctx.model, "n_catalog", ctx.n_items)),
        "ndcg@10": float(metric),
        "eval_every_steps": tc["eval_every_steps"],
    }


def loader_bench(ctx: Ctx, workers: list[int], n_batches: int) -> dict:
    """DataLoader-only batches/s for each num_workers x pin_memory (no model work)."""
    tc = ctx.cfg["train"]
    out = {}
    n_get = min(len(ctx.ds), 500)
    idx = np.random.default_rng(0).integers(0, len(ctx.ds.ds), n_get)
    t0 = time.perf_counter()
    for i in idx:
        ctx.ds[int(i)]
    out["getitem_us"] = (time.perf_counter() - t0) / n_get * 1e6
    for nw in workers:
        for pin in (False, True):
            key = f"nw{nw}_pin{int(pin)}"
            if pin and ctx.device.type != "cuda":
                out[key] = {"skipped": "pin_memory needs CUDA"}
                continue
            sampler = _ResumableSampler(len(ctx.seqs), tc["batch_size"], tc["seed"])
            spec = {"ds": ctx.ds, "sampler": sampler, "batch_size": tc["batch_size"],
                    "seed": tc["seed"]}  # fmt: skip
            loader = make_loader(spec, nw, pin)
            t0 = time.perf_counter()
            got, first, t1, epochs = 0, None, None, 0
            while got < n_batches:
                sampler.set_position(epochs, 0)
                for _ in loader:
                    got += 1
                    if first is None:
                        first = time.perf_counter() - t0
                        t1 = time.perf_counter()
                    if got >= n_batches:
                        break
                epochs += 1
            dt = time.perf_counter() - t1
            rate = (got - 1) / dt if got > 1 and dt > 0 else float("nan")
            out[key] = {
                "first_batch_s": first,
                "batches_per_s": rate,
                "samples_per_s": rate * tc["batch_size"],
                "batches": got,
            }
            del loader
            gc.collect()
    return out


# --- summary -------------------------------------------------------------------------------


def _fmt_summary(r: dict) -> str:
    L = []
    c = r["config"]
    L.append(f"profile_train: {c['name']} on {r['env']['device']} (torch {r['env']['torch']}, "
             f"cuda {r['env']['torch_cuda']}, {r['env']['capability']})")  # fmt: skip
    L.append(
        f"batch {c['batch_size']} x max_len {c['max_len']}, catalog {c['n_items']}, "
        f"negatives {c['n_in_batch']} in-batch + {c['n_uniform']} uniform + {c['n_hard']} hard, "
        f"logq {c['logq']}, input {c['input_filter']}, windows {c['window_sampling']}, "
        f"workers {c['num_workers']} pin {c['pin_memory']} prefetch {c['prefetch_factor']}"
    )
    s = r["shapes"]
    L.append(
        f"M (valid rows) mean {s['M_mean']:.0f}  S {s['S_mean']:.0f}  logits fp32 "
        f"{s['logits_gb_fp32']:.2f} GB per [M, S+1] tensor"
    )
    if r["env"]["native_sass_for_device"] is False:
        L.append(f"WARNING: torch has no native SASS for {r['env']['capability']} "
                 f"(arch list {r['env']['arch_list']}): kernels are JIT-compiled from PTX")  # fmt: skip
    L.append(f"setup: {json.dumps({k: round(v, 2) for k, v in r['setup_s'].items()})}")
    w = r["warmup"]
    if w["step_s"]:
        L.append(
            f"warmup {len(w['step_s'])} steps: first {w['step_s'][0]:.2f} s "
            f"(phases {json.dumps({k: round(v, 2) for k, v in w['first_step_phases'].items()})}),"
            f" rest mean {np.mean(w['step_s'][1:]) if len(w['step_s']) > 1 else float('nan'):.3f} s"
        )
    p = r["phased"]
    L.append(f"phased ({p['steps']} synced steps): {p['step_mean_s'] * 1e3:.1f} ms/step = "
             f"{p['steps_per_sec']:.2f} steps/s, compute/total {p['gpu_util_proxy']:.2f}")  # fmt: skip
    for name in PHASES:
        q = p["phases"][name]
        L.append(f"  {name:<20} {q['mean_s'] * 1e3:9.2f} ms  ({q['share'] * 100:5.1f}%)  "
                 f"p95 {q['p95_s'] * 1e3:8.2f} ms  max {q['max_s'] * 1e3:8.2f} ms")  # fmt: skip
    t = r["throughput"]
    L.append(f"throughput ({t['steps']} unsynced steps): {t['steps_per_sec']:.2f} steps/s, "
             f"GPU util {t['gpu_util']['mean_pct']}")  # fmt: skip
    if r["memory"]:
        L.append(f"memory: {json.dumps(r['memory'])}")
    if r["profiler"]:
        n = r["profiler"]["steps"]
        L.append(f"torch.profiler over {n} steps; top ops by self device time:")
        L.append(r["profiler"]["tables"]["by_device_time"])
        L.append("top ops by self CPU time:")
        L.append(r["profiler"]["tables"]["by_cpu_time"])
    e = r["eval"]
    if e:
        L.append(f"eval: {e['eval_s']:.2f} s for {e['n_users']} users x {e['n_catalog']} items "
                 f"(all_item_repr {e['all_item_repr_s']:.2f} s), every {e['eval_every_steps']} "
                 f"steps; ndcg@10 {e['ndcg@10']:.4f}")  # fmt: skip
    L.append(f"loader-only (getitem {r['loader']['getitem_us']:.0f} us/sample):")
    for k, v in r["loader"].items():
        if k == "getitem_us":
            continue
        if "skipped" in v:
            L.append(f"  {k:<10} skipped ({v['skipped']})")
        else:
            L.append(f"  {k:<10} {v['batches_per_s']:8.1f} batches/s  first batch "
                     f"{v['first_batch_s']:.2f} s")  # fmt: skip
    L.append(f"total wall {r['wall_s']:.1f} s")
    return "\n".join(L)


# --- main ----------------------------------------------------------------------------------


def profile(args) -> dict:
    t_start = time.perf_counter()
    overrides = parse_overrides(args.override)
    cfg = load_config(args.config, overrides)
    t0 = time.perf_counter()
    data = load_processed(args.processed)
    t_data = time.perf_counter() - t0
    t0 = time.perf_counter()
    set_seed(cfg["train"]["seed"])  # as run_train: model init is part of the run
    model = build_model(cfg, data)
    t_model = time.perf_counter() - t0
    t0 = time.perf_counter()
    ctx = setup(cfg, data, model, args.split)
    t_setup = time.perf_counter() - t0
    log.info("profiling %s on %s", args.config, device_name(ctx.device))
    tc, lc = cfg["train"], cfg["loss"]
    it = batches(ctx)
    t0 = time.perf_counter()
    first_batch = next(it)
    t_first = time.perf_counter() - t0

    def chained(rest):
        yield first_batch
        yield from rest

    it = chained(it)

    # 1. warmup (timed per phase, kept per step)
    warm, _ = run_steps(ctx, it, args.warmup, "timed")
    if warm:
        warm[0]["data_wait"] = warm[0].get("data_wait", 0.0) + t_first
    # 2. phased
    phased, _ = run_steps(ctx, it, args.steps, "timed")
    ph = _phase_stats(phased)
    m_phased = ctx.stats["M"][-len(phased) :] if phased else ctx.stats["M"]
    s_phased = ctx.stats["S"][-len(phased) :] if phased else ctx.stats["S"]
    # 3. throughput, no extra syncs
    n_tp = args.throughput_steps if args.throughput_steps is not None else args.steps
    with _GpuUtil(ctx.device) as gu:
        _, wall = run_steps(ctx, it, n_tp, "off")
    throughput = {
        "steps": n_tp,
        "wall_s": wall,
        "steps_per_sec": n_tp / wall if wall > 0 and n_tp else float("nan"),
        "gpu_util": gu.result(),
    }
    memory = _memory(ctx.device)
    # 4. torch.profiler
    prof = run_profiler(ctx, it, args.profile_steps) if args.profile_steps > 0 else None
    # 5. eval
    ev = None if args.no_eval else timed_eval(ctx)
    # 6. DataLoader-only
    del it
    gc.collect()
    workers = [int(x) for x in args.loader_workers.split(",") if x.strip()]
    loader = loader_bench(ctx, workers, args.loader_batches)

    M = float(np.mean(m_phased)) if m_phased else float("nan")
    S = float(np.mean(s_phased)) if s_phased else float("nan")
    result = {
        "config": {
            "name": Path(args.config).stem,
            "path": str(args.config),
            "overrides": args.override,
            "split": args.split,
            "batch_size": tc["batch_size"],
            "max_len": cfg["model"]["max_len"],
            "d_model": cfg["model"]["d_model"],
            "n_items": ctx.n_items,
            "n_catalog": int(getattr(model, "n_catalog", ctx.n_items)),
            "n_params": int(sum(p.numel() for p in model.parameters())),
            "n_in_batch": ctx.n_ib,
            "n_uniform": ctx.n_uni,
            "n_hard": ctx.n_hard,
            "logq": bool(lc.get("logq")),
            "amp": ctx.amp,
            "input_filter": input_filter_of(cfg),
            "window_sampling": window_sampling_from(cfg),
            "num_workers": tc.get("num_workers", 0),
            "pin_memory": ctx.loader.pin_memory,
            "persistent_workers": ctx.loader.persistent_workers,
            "prefetch_factor": ctx.loader.prefetch_factor,
            "n_seqs": len(ctx.seqs),
            "batches_per_epoch": ctx.sampler.batches_per_epoch,
        },
        "env": _env(ctx.device),
        "setup_s": {
            "load_processed": t_data,
            "build_model": t_model,
            "trainer_setup": t_setup,
            "first_batch": t_first,
        },
        "shapes": {
            "M_mean": M,
            "S_mean": S,
            "logits_gb_fp32": M * (S + 1) * 4 / 1e9 if not math.isnan(M) else float("nan"),
        },
        "warmup": {
            "step_s": [sum(s.values()) for s in warm],
            "first_step_phases": warm[0] if warm else {},
            "phases_sum_s": {p: float(sum(s.get(p, 0.0) for s in warm)) for p in PHASES},
        },
        "phased": {"steps": len(phased), **ph},
        "throughput": throughput,
        "memory": memory,
        "profiler": prof,
        "eval": ev,
        "loader": loader,
        "wall_s": time.perf_counter() - t_start,
    }
    return result


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--split", choices=("val", "test"), default="val")
    ap.add_argument("--steps", type=int, default=300, help="phased (synchronised) steps")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--throughput-steps", type=int, default=None, help="default: --steps")
    ap.add_argument("--profile-steps", type=int, default=20, help="0 = skip torch.profiler")
    ap.add_argument("--loader-batches", type=int, default=50)
    ap.add_argument("--loader-workers", default="0,2,4", help="comma-separated num_workers")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    r = profile(args)
    args.out.mkdir(parents=True, exist_ok=True)
    summary = _fmt_summary(r)
    (args.out / "profile.json").write_text(json.dumps(r, indent=1, default=str) + "\n")
    (args.out / "summary.txt").write_text(summary + "\n")
    print(summary, flush=True)
    return r


if __name__ == "__main__":
    main()
