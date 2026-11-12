"""Resumable sampled-softmax trainer for sequence models (fp16 AMP on CUDA, fp32 on CPU).

split="val": train on `seqs_val_train`, evaluate warm NDCG@10 on `truth_val` every
`eval_every_steps`, early-stop after `patience` evals without improvement, keep `best.pt`.
split="test": train on `seqs_train` for exactly `max_steps` steps, no evaluation.

Every `ckpt_every_steps`, and whenever training stops, an atomic `ckpt-{step:08d}.pt` is written
to `ckpt_dir` (2 newest kept). A run resumes from the newest good checkpoint in
`[ckpt_dir, *resume_dirs]`; resume dirs are only ever read (on Kaggle they are read-only inputs).
"""

from __future__ import annotations

import copy
import logging
import math
import os
import shutil
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, Sampler

from hookrec.data.label import DISLIKE
from hookrec.data.processed import ProcessedData
from hookrec.eval.protocol import evaluate
from hookrec.models.base import history_for
from hookrec.models.losses import make_log_q, sample_negatives, sampled_softmax_loss
from hookrec.models.popularity import MostPop
from hookrec.train.checkpoint import (
    atomic_save,
    capture_rng_states,
    ckpt_name,
    load_latest,
    prune_checkpoints,
    restore_rng_states,
    save_checkpoint,
)
from hookrec.train.datasets import SeqTrainDataset, session_gap_from
from hookrec.train.scoring import SeqRecommender
from hookrec.utils.config import config_hash
from hookrec.utils.seed import set_seed

log = logging.getLogger(__name__)

# `max_hours` bounds the Kaggle session, so it is measured from the session start: the unix time
# in env HOOKREC_SESSION_START (exported by the notebooks' first cell) if set, else from when
# this process imported the trainer (includes data loading, not just the `train` call).
_PROCESS_T0 = time.monotonic()
_warned_bad_session_start: set[str] = set()


def session_elapsed_seconds() -> float:
    start = os.environ.get("HOOKREC_SESSION_START")
    if start:
        try:
            return time.time() - float(start)
        except ValueError:  # never crash mid-training over a bad env var
            if start not in _warned_bad_session_start:
                _warned_bad_session_start.add(start)
                log.warning(
                    "ignoring HOOKREC_SESSION_START=%r (not unix seconds); "
                    "measuring max_hours from process start",
                    start,
                )
    return time.monotonic() - _PROCESS_T0

# train.* keys that do not change what is learned (or that define the horizon, which is
# persisted separately); they are excluded from `training_hash`.
OPERATIONAL_KEYS = frozenset(
    {
        "max_hours", "num_workers", "device", "log_every_steps", "eval_max_users",
        "eval_batch_size", "ckpt_every_steps", "max_steps", "schedule_steps",
    }
)  # fmt: skip


def training_hash(cfg: dict) -> str:
    """Config hash ignoring OPERATIONAL_KEYS: resume compatibility and the val<->test match."""
    c = copy.deepcopy(cfg)
    for k in OPERATIONAL_KEYS:
        c.get("train", {}).pop(k, None)
    return config_hash(c)


@dataclass
class TrainResult:
    best_metric: float  # best periodic val ndcg@10; NaN for split="test"
    steps: int
    wall_time: float  # training seconds, summed over all resumed sessions
    stopped_reason: Literal["converged", "max_hours", "max_steps"]
    best_step: int  # step of best.pt (val); the final step (test)
    steps_per_sec: float  # this session
    final_loss: float
    max_steps: int  # horizon actually used (from the checkpoint when resumed)
    schedule_steps: int


def device_name(device: torch.device) -> str:
    return torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"


def warmup_cosine(warmup_steps: int, total_steps: int):
    """LR multiplier: linear warmup over `warmup_steps`, then cosine decay to 0 at `total_steps`."""

    def f(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return f


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """AdamW groups: no weight decay on embeddings, biases and LayerNorm parameters."""
    no_decay = set()
    for m in model.modules():
        if isinstance(m, (nn.Embedding, nn.LayerNorm)):
            no_decay.update(id(p) for p in m.parameters(recurse=False))
    decay, rest = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (rest if id(p) in no_decay or name.endswith("bias") else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": rest, "weight_decay": 0.0}]


def hard_negatives(items: Tensor, tokens: Tensor, n_hard: int) -> Tensor:
    """[B, n_hard] ids of each row's `n_hard` most recent DISLIKEs in its window, left-padded 0.

    Window-level, not position-causal: a position may get a dislike that comes after it. That
    is acceptable for negatives (it leaks nothing about the positive target).
    """
    dis = (tokens == DISLIKE) & (items != 0)
    rank = dis.flip(1).cumsum(1).flip(1)  # 1 = most recent dislike at or after the position
    sel = dis & (rank <= n_hard)
    slot = torch.where(sel, n_hard - rank, torch.full_like(rank, n_hard))  # n_hard = dump column
    out = torch.zeros(items.shape[0], n_hard + 1, dtype=items.dtype, device=items.device)
    out.scatter_(1, slot, torch.where(sel, items, torch.zeros_like(items)))
    return out[:, :n_hard]


class _EpochIndexed(Dataset):
    """Global index e * n + i -> sample i of epoch e, with the epoch set where it is loaded.

    Calling `SeqTrainDataset.set_epoch` in the main process would not reach persistent
    DataLoader workers (each holds its own copy), so the epoch travels inside the index.
    """

    def __init__(self, ds: SeqTrainDataset):
        self.ds = ds

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int) -> dict[str, Tensor]:
        epoch, idx = divmod(int(i), len(self.ds))
        self.ds.set_epoch(epoch)
        return self.ds[idx]


class _ResumableSampler(Sampler):
    """Seeded per-epoch permutation (full batches only) that can start mid-epoch."""

    def __init__(self, n: int, batch_size: int, seed: int):
        self.n, self.batch_size, self.seed = n, batch_size, seed
        self.batches_per_epoch = n // batch_size
        self.epoch = self.start_batch = 0

    def set_position(self, epoch: int, start_batch: int) -> None:
        self.epoch, self.start_batch = epoch, start_batch

    def __len__(self) -> int:
        return max(0, self.batches_per_epoch - self.start_batch) * self.batch_size

    def __iter__(self):
        perm = np.random.default_rng([self.seed, self.epoch]).permutation(self.n)
        stop = self.batches_per_epoch * self.batch_size
        base = self.epoch * self.n
        return iter((base + perm[self.start_batch * self.batch_size : stop]).tolist())


def _eval_truth(truth: dict[int, np.ndarray], max_users: int | None, seed: int) -> dict:
    if not max_users or max_users >= len(truth):
        return truth
    uids = np.random.default_rng(seed).choice(np.array(sorted(truth)), max_users, replace=False)
    return {int(u): truth[int(u)] for u in uids}


def train(
    cfg: dict,
    data: ProcessedData,
    model: nn.Module,
    split: Literal["val", "test"],
    ckpt_dir: Path,
    resume_dirs: Sequence[Path] = (),
) -> TrainResult:
    tc, lc, max_len = cfg["train"], cfg["loss"], cfg["model"]["max_len"]
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(tc.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = device.type == "cuda"
    log.info("training %s on %s (split=%s)", cfg["model"]["name"], device_name(device), split)

    set_seed(tc["seed"])
    seqs, _ = history_for(data, split)
    n_items = data.vocab.n_items
    eval_every, ckpt_every = tc["eval_every_steps"], tc["ckpt_every_steps"]
    log_every = tc.get("log_every_steps", 100)
    n_ib, n_uni = lc.get("n_in_batch_negatives", 4096), lc["n_uniform_negatives"]
    n_hard = lc.get("n_hard", 4) if lc.get("dislike_hard_negatives") else 0
    session_gap = session_gap_from(cfg)
    like_weight = lc.get("like_weight", 0.0) if getattr(model, "like_head", False) else 0.0

    cfg_hash = training_hash(cfg)
    payload = load_latest([ckpt_dir, *resume_dirs])
    if payload is not None:
        if payload.get("split", split) != split:
            raise ValueError(
                f"{payload['path']} is a split={payload['split']!r} checkpoint; "
                f"use a separate --ckpt dir for split={split!r}"
            )
        if payload.get("config_hash", cfg_hash) != cfg_hash:
            raise ValueError(
                f"{payload['path']} was trained with a different config "
                f"(hash {payload['config_hash']} != {cfg_hash}); use a fresh --ckpt dir"
            )
    # The training horizon is persisted: a resumed run always finishes the run it started.
    if payload is not None and "max_steps" in payload:
        max_steps, schedule_steps = payload["max_steps"], payload["schedule_steps"]
    else:
        max_steps = tc["max_steps"]
        schedule_steps = tc.get("schedule_steps") or max_steps

    model.to(device).train()
    opt = torch.optim.AdamW(param_groups(model, tc["weight_decay"]), lr=tc["lr"])
    sched = LambdaLR(opt, warmup_cosine(tc["warmup_steps"], schedule_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    gen = torch.Generator().manual_seed(tc["seed"])  # negative sampling
    fallback = MostPop().fit(seqs, n_items)
    log_q = None
    if lc.get("logq"):
        # popularity of the training period actually used: val recounts its own history
        pop = fallback.scores if split == "val" else data.features.pop_train
        log_q = make_log_q(pop, lc.get("mix_uniform", 0.0), n_ib, n_uni).to(device)
    eval_truth = _eval_truth(data.truth_val, tc.get("eval_max_users"), tc["seed"])

    st = {
        "epoch": 0,
        "epoch_batch": 0,
        "best_metric": float("-inf"),
        "best_step": 0,
        "bad_evals": 0,
        "last_eval_step": -1,
        "train_seconds": 0.0,
    }
    step = 0
    best_path = ckpt_dir / "best.pt"
    if payload is None:
        if best_path.exists():
            log.warning("fresh run: removing stale %s", best_path)
            best_path.unlink()
    else:
        model.load_state_dict(payload["model"])
        opt.load_state_dict(payload["optimizer"])
        sched.load_state_dict(payload["scheduler"])
        scaler.load_state_dict(payload["scaler"])
        restore_rng_states(payload["rng_states"])
        gen.set_state(payload["rng_states"]["sampler"])
        step = payload["step"]
        st.update({k: payload[k] for k in st if k in payload})
        src_best = Path(payload["path"]).parent / "best.pt"
        if src_best.parent.resolve() != ckpt_dir.resolve() and src_best.exists():
            shutil.copyfile(src_best, ckpt_dir / "best.tmp")
            os.replace(ckpt_dir / "best.tmp", best_path)
        if split == "val" and best_path.exists():  # best.pt may postdate the checkpoint
            b = torch.load(best_path, map_location="cpu", weights_only=True)
            if b["metric"] > st["best_metric"]:
                st["best_metric"], st["best_step"] = b["metric"], b["step"]
        log.info(
            "resumed from %s at step %d (max_steps %d, schedule_steps %d)",
            payload["path"], step, max_steps, schedule_steps,
        )  # fmt: skip
        del payload

    ds = _EpochIndexed(SeqTrainDataset(seqs, max_len, tc["seed"], session_gap))
    sampler = _ResumableSampler(len(seqs), tc["batch_size"], tc["seed"])
    if sampler.batches_per_epoch == 0:
        raise ValueError(f"{len(seqs)} sequences cannot fill one batch of {tc['batch_size']}")
    nw = tc.get("num_workers", 0)
    loader = DataLoader(
        ds,
        batch_size=tc["batch_size"],
        sampler=sampler,
        drop_last=True,
        num_workers=nw,
        pin_memory=amp,
        persistent_workers=nw > 0,
        generator=torch.Generator().manual_seed(tc["seed"]),  # keep the global RNG untouched
    )

    t_session, steps_session, last_ckpt = time.monotonic(), 0, -1
    dirty = False  # early-stopping state changed since the last checkpoint
    seconds_before = st["train_seconds"]
    loss = torch.tensor(float("nan"))
    parts = {"rank": float("nan"), "like": 0.0}  # components of the last step's loss

    def stop_reason() -> str | None:
        if split == "val" and st["bad_evals"] >= tc["patience"]:  # also on re-launch after it
            return "converged"
        if step >= max_steps:
            return "max_steps"
        if session_elapsed_seconds() > tc["max_hours"] * 3600:
            return "max_hours"
        return None

    def run_eval() -> None:
        nonlocal dirty
        rec = SeqRecommender(
            model, seqs, max_len, device, fallback, tc.get("eval_batch_size", 1024),
            session_gap=session_gap,
        )  # fmt: skip
        metric = evaluate(rec, eval_truth, n_items, ks=(10,)).metrics["ndcg@10"]
        st["last_eval_step"] = step
        dirty = True
        if metric > st["best_metric"]:
            st.update(best_metric=metric, best_step=step, bad_evals=0)
            atomic_save({"model": model.state_dict(), "step": step, "metric": metric}, best_path)
        elif step > st["best_step"]:  # replayed evals before an adopted best.pt don't count
            st["bad_evals"] += 1
        log.info(
            "eval step %d ndcg@10 %.5f (best %.5f @ %d, %d bad)",
            step, metric, st["best_metric"], st["best_step"], st["bad_evals"],
        )  # fmt: skip

    def checkpoint() -> None:
        nonlocal last_ckpt, dirty
        rng = capture_rng_states()
        rng["sampler"] = gen.get_state()
        st["train_seconds"] = seconds_before + time.monotonic() - t_session
        extra = {k: v for k, v in st.items() if k not in ("epoch", "best_metric")}
        save_checkpoint(
            ckpt_dir / ckpt_name(step), model, opt, sched, scaler, step, st["epoch"],
            st["best_metric"], rng, split=split, config_hash=cfg_hash, max_steps=max_steps,
            schedule_steps=schedule_steps, **extra,
        )  # fmt: skip
        prune_checkpoints(ckpt_dir, keep=2)
        last_ckpt, dirty = step, False

    def train_step(batch: dict[str, Tensor]) -> Tensor:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            hidden = model.encode(batch)
            valid = batch["target"] != 0
            user_vecs, pos_items = hidden[valid], batch["target"][valid]
            neg = [sample_negatives(n_items, n_uni, gen).to(device)]
            if n_ib and len(pos_items):  # in-batch: positives drawn with replacement ~ popularity
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
        with torch.autocast(device.type, enabled=False):  # logits / softmax in fp32
            out = sampled_softmax_loss(
                user_vecs.float(), pos_items, pos_vecs.float(), neg_items, neg_vecs.float(),
                log_q, lc.get("temperature", 1.0), hard_neg_vecs=hard_vecs,
                hard_neg_items=hard_items,
            )  # fmt: skip
            parts["rank"] = out.detach()
            if like_weight:  # multi-task: does the target event get a like? (positions with target)
                logits = model.like_logit(user_vecs.float(), pos_vecs.float())
                # sum / max(n, 1): an empty batch gives 0 (BCE's mean would be NaN)
                like_loss = F.binary_cross_entropy_with_logits(
                    logits, like_y.float(), reduction="sum"
                ) / max(len(like_y), 1)
                parts["like"] = like_loss.detach()
                out = out + like_weight * like_loss
        opt.zero_grad(set_to_none=True)
        scaler.scale(out).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), tc["max_grad_norm"])
        scale_before = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        if scaler.get_scale() >= scale_before:  # a dropped scale = inf/NaN grads, step skipped
            sched.step()
        return out.detach()

    reason = stop_reason()
    while reason is None:
        sampler.set_position(st["epoch"], st["epoch_batch"])
        for batch in loader:
            loss = train_step(batch)
            step += 1
            st["epoch_batch"] += 1
            steps_session += 1
            if step % log_every == 0:
                sps = steps_session / max(time.monotonic() - t_session, 1e-9)
                log.info(
                    "step %d loss %.4f (rank %.4f like %.4f) lr %.2e %.2f steps/s ETA %.2f h",
                    step,
                    float(loss),
                    float(parts["rank"]),
                    float(parts["like"]),
                    sched.get_last_lr()[0],
                    sps,
                    (max_steps - step) / sps / 3600,
                )
            if split == "val" and step % eval_every == 0:
                run_eval()
            if step % ckpt_every == 0:
                checkpoint()
            reason = stop_reason()
            if reason:
                break
        else:
            st["epoch"] += 1
            st["epoch_batch"] = 0

    if split == "val" and reason == "max_steps" and st["last_eval_step"] != step:
        run_eval()
    if last_ckpt != step or dirty:
        checkpoint()
    if reason != "max_hours":
        atomic_save(
            {"model": model.state_dict(), "cfg": cfg, "step": step}, ckpt_dir / "model_final.pt"
        )
    sps = steps_session / max(time.monotonic() - t_session, 1e-9)
    log.info("stopped (%s) at step %d, %.2f steps/s this session", reason, step, sps)

    val_best = st["best_metric"] if split == "val" and st["last_eval_step"] >= 0 else float("nan")
    return TrainResult(
        best_metric=val_best,
        steps=step,
        wall_time=st["train_seconds"],
        stopped_reason=reason,
        best_step=st["best_step"] if split == "val" else step,
        steps_per_sec=sps,
        final_loss=float(loss),
        max_steps=max_steps,
        schedule_steps=schedule_steps,
    )
