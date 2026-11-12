"""Helpers for the headless Kaggle notebooks (K0-K5): input discovery, logged subprocesses, disk.

Notebooks run as "Save Version -> Save & Run All" with nobody watching, so every helper either
works or raises with a message that says what to attach or fix. `!cmd` shell magics never raise
on failure; `run_logged` does.
"""

from __future__ import annotations

import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

INPUT_ROOT = "/kaggle/input"
WORKING = "/kaggle/working"


def find_inputs(marker: str, root: str | Path = INPUT_ROOT) -> list[Path]:
    """Sorted dirs under `root` that contain `marker` (a file name, glob, or relative path).

    For a multi-part marker such as "flat/50m/multi_event.parquet" the dir it is relative to is
    returned (here: the Yambda root).
    """
    root = Path(root)
    if not root.is_dir():
        return []
    up = len(Path(marker).parts) - 1
    return sorted({p.parents[up] for p in root.rglob(marker)})


def _listing(root: Path, depth: int = 2) -> str:
    if not root.is_dir():
        return f"{root} does not exist (no inputs attached?)"
    lines = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if len(rel.parts) <= depth:
            lines.append(f"  {rel}{'/' if p.is_dir() else ''}")
    return "\n".join(lines[:60]) or f"  ({root} is empty)"


def find_one(
    marker: str, root: str | Path = INPUT_ROOT, required: bool = True, hint: str = ""
) -> Path | None:
    """The first dir from `find_inputs`; raises FileNotFoundError naming `marker` if none."""
    found = find_inputs(marker, root)
    if not found:
        if not required:
            return None
        raise FileNotFoundError(
            f"no attached input contains {marker!r} under {root}. {hint}\n"
            f"What is under {root}:\n{_listing(Path(root))}"
        )
    if len(found) > 1:
        print(f"note: {len(found)} inputs contain {marker!r}; using {found[0]}", flush=True)
    return found[0]


def find_processed(size: str, root: str | Path = INPUT_ROOT) -> Path:
    """The one finished processed bundle (`{size}-{hash}/_DONE`) under `root`."""
    found = [p for p in find_inputs("_DONE", root) if p.name.startswith(f"{size}-")]
    if not found:
        raise FileNotFoundError(
            f"no finished processed bundle '{size}-*/_DONE' under {root}: attach the K2 output "
            f"for SIZE={size!r}.\nWhat is under {root}:\n{_listing(Path(root))}"
        )
    if len(found) > 1:
        raise RuntimeError(
            f"{len(found)} processed bundles for {size}: {[str(p) for p in found]}. "
            "Attach only the K2 output that matches this run."
        )
    return found[0]


def run_logged(cmd: list, env: dict | None = None, cwd: str | Path | None = None) -> None:
    """Run `cmd`, streaming merged stdout/stderr line by line; RuntimeError on non-zero exit."""
    cmd = [str(c) for c in cmd]
    full_env = {**os.environ, "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg", **(env or {})}
    print(f"$ {shlex.join(cmd)}", flush=True)
    t0 = time.monotonic()
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
        env=full_env,
        cwd=cwd,
    ) as p:
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
        code = p.wait()
    took = time.monotonic() - t0
    if code != 0:
        raise RuntimeError(
            f"{shlex.join(cmd[:2])} ... exited {code} after {took:.0f} s (log above)"
        )
    print(f"[ok in {took / 60:.1f} min]", flush=True)


def session_env() -> str:
    """Export HOOKREC_SESSION_START (unix seconds) unless set; the trainer's max_hours clock."""
    return os.environ.setdefault("HOOKREC_SESSION_START", str(int(time.time())))


def dir_size_bytes(path: str | Path) -> int:
    total = 0
    stack = [Path(path)]
    while stack:
        try:
            with os.scandir(stack.pop()) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(Path(e.path))
                        elif e.is_file(follow_symlinks=False):
                            total += e.stat(follow_symlinks=False).st_size
                    except OSError:
                        pass
        except OSError:
            pass
    return total


def check_disk(path: str | Path = WORKING, warn_gb: float = 18.0) -> float:
    """Print the size of `path` (GB) and warn above `warn_gb` (Kaggle's working disk is 20 GB)."""
    gb = dir_size_bytes(path) / 1e9
    print(f"{path}: {gb:.2f} GB used", flush=True)
    if gb > warn_gb:
        print(
            f"WARNING: {path} holds {gb:.1f} GB (> {warn_gb} GB of Kaggle's 20 GB): the output "
            "may fail to save. Delete stale checkpoints or split the stage.",
            flush=True,
        )
    return gb


def _meminfo() -> tuple[float | None, float | None]:
    try:
        kv = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, _, v = line.partition(":")
            kv[k] = int(v.split()[0]) * 1024
        return kv["MemTotal"] / 1e9, kv.get("MemAvailable", kv.get("MemFree", 0)) / 1e9
    except (OSError, KeyError, ValueError, IndexError):
        pass
    try:
        import psutil

        vm = psutil.virtual_memory()
        return vm.total / 1e9, vm.available / 1e9
    except ImportError:
        return None, None


def env_report(work: str | Path = WORKING) -> dict:
    """Print and return python/torch versions, GPUs, RAM and free disk on `work` (or cwd)."""
    rep: dict = {"python": platform.python_version()}
    try:
        import torch

        rep["torch"] = torch.__version__
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        rep["gpus"] = [torch.cuda.get_device_name(i) for i in range(n)] or ["cpu"]
    except ImportError:
        rep["torch"], rep["gpus"] = None, ["cpu"]
    total, avail = _meminfo()
    rep["ram_total_gb"] = round(total, 2) if total else None
    rep["ram_available_gb"] = round(avail, 2) if avail else None
    disk = Path(work) if Path(work).exists() else Path.cwd()
    rep["disk_free_gb"] = round(shutil.disk_usage(disk).free / 1e9, 2)
    rep["cpus"] = os.cpu_count()
    print(json.dumps(rep, indent=2), flush=True)
    return rep


def has_run(results_dir: str | Path, run_name: str, processed_dir: str) -> bool:
    """True if `results_dir` holds a `run_name` record for `processed_dir` (bad JSONs ignored)."""
    d = Path(results_dir)
    for p in sorted(d.glob("*.json")) if d.is_dir() else []:
        try:
            r = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if (
            r.get("run_name") == run_name
            and (r.get("extra") or {}).get("processed_dir") == processed_dir
        ):
            return True
    return False
