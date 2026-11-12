"""Atomic training checkpoints (tmp file + `os.replace`) and crash-tolerant resume.

Checkpoints are trusted files written by this code, so they are loaded with
`weights_only=False` (they carry python/numpy RNG states). Never point resume at untrusted dirs.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger(__name__)

REQUIRED_KEYS = frozenset(
    {"model", "optimizer", "scheduler", "scaler", "step", "epoch", "best_metric", "rng_states"}
)
_CKPT_RE = re.compile(r"^ckpt-(\d+)\.pt$")


def ckpt_name(step: int) -> str:
    return f"ckpt-{step:08d}.pt"


def atomic_save(obj, path: Path) -> None:
    """torch.save to a `.tmp` sibling, fsync, then `os.replace`: a kill never leaves `path` torn."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _sidecar(ckpt: Path) -> Path:
    return ckpt.with_suffix(".json")


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def read_horizon(dirs) -> dict | None:
    """`{step, max_steps, schedule_steps, path}` of the newest checkpoint from its sidecar.

    Cheap: never loads a checkpoint. None when the newest checkpoint has no readable sidecar
    (written by an older version, or killed between the two writes); callers fall back to
    `load_latest`.
    """
    dirs = [dirs] if isinstance(dirs, (str, Path)) else list(dirs)
    cands = _candidates(dirs)
    if not cands:
        return None
    path = cands[0][1]
    try:
        h = json.loads(_sidecar(path).read_text())
        out = {k: int(h[k]) for k in ("step", "max_steps", "schedule_steps")}
    except (OSError, ValueError, KeyError, TypeError):
        return None
    out["path"] = str(path)
    return out


def capture_rng_states() -> dict:
    states = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state_all()
    return states


def restore_rng_states(states: dict) -> None:
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch"])
    if "cuda" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["cuda"])


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    step: int,
    epoch: int,
    best_metric: float,
    rng_states: dict,
    **extra,
) -> None:
    """Atomically write a resumable checkpoint; `extra` (e.g. best_step, bad_evals) is stored too."""
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "step": int(step),
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "rng_states": rng_states,
        **extra,
    }
    atomic_save(payload, Path(path))
    # Small sidecar so the horizon can be read without deserializing the (GB-sized) checkpoint.
    horizon = {k: int(extra[k]) for k in ("max_steps", "schedule_steps") if k in extra}
    if horizon:
        _atomic_write_text(_sidecar(Path(path)), json.dumps({"step": int(step), **horizon}))


def _candidates(dirs: Sequence[Path]) -> list[tuple[int, Path]]:
    found = []
    for d in dirs:
        if not Path(d).is_dir():
            continue
        for p in Path(d).iterdir():
            m = _CKPT_RE.match(p.name)
            if m:
                found.append((int(m.group(1)), p))
    # highest step first; on equal steps the earlier dir in `dirs` wins (stable sort)
    return sorted(found, key=lambda t: -t[0])


def load_latest(dirs) -> dict | None:
    """Payload of the newest loadable checkpoint across `dirs` (a dir or a list), else None.

    Files that fail to load or lack the required keys (a torn copy, a foreign file) are skipped
    with a warning. The payload gets a `path` key naming the file it came from.
    """
    dirs = [dirs] if isinstance(dirs, (str, Path)) else list(dirs)
    for _, p in _candidates(dirs):
        try:
            payload = torch.load(p, map_location="cpu", weights_only=False)
        except Exception as e:  # noqa: BLE001 - any unreadable file is just skipped
            log.warning("skipping unreadable checkpoint %s: %s", p, e)
            continue
        if not isinstance(payload, dict) or not REQUIRED_KEYS <= payload.keys():
            log.warning("skipping checkpoint %s: missing keys", p)
            continue
        payload["path"] = str(p)
        return payload
    return None


def prune_checkpoints(ckpt_dir: Path, keep: int = 2) -> None:
    """Delete all but the `keep` newest step checkpoints (and stale .tmp files) in `ckpt_dir`."""
    for _, p in _candidates([ckpt_dir])[keep:]:
        p.unlink(missing_ok=True)
        _sidecar(p).unlink(missing_ok=True)
    for p in (*Path(ckpt_dir).glob("ckpt-*.tmp"), *Path(ckpt_dir).glob("ckpt-*.json.tmp")):
        p.unlink(missing_ok=True)
