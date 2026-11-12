"""Run-record JSON files: one per training/eval run, written under a results directory."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from hookrec.utils.config import config_hash
from hookrec.utils.gitinfo import git_sha


def save_run(
    out_dir: Path, run_name: str, cfg: dict, metrics: dict, extra: dict | None = None
) -> Path:
    """Write `{run_name}-{YYYYmmdd-HHMMSS}.json` (suffix `-1`, `-2`, ... on collision)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    record = {
        "run_name": run_name,
        "timestamp": now.isoformat(timespec="microseconds"),
        "git_sha": git_sha(),
        "config_hash": config_hash(cfg),
        "config": cfg,
        "metrics": metrics,
        "extra": extra or {},
    }
    text = json.dumps(_clean(record), indent=2, allow_nan=False)
    stem = f"{run_name}-{now.strftime('%Y%m%d-%H%M%S')}"
    path = out_dir / f"{stem}.json"
    with tempfile.NamedTemporaryFile("w", dir=out_dir, delete=False, suffix=".tmp") as fd:
        tmp = Path(fd.name)
        try:
            fd.write(text)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    try:
        n = 0
        while True:
            try:
                os.link(tmp, path)  # publish without overwriting
                break
            except FileExistsError:
                n += 1
                path = out_dir / f"{stem}-{n}.json"
            except OSError:  # hard links unsupported: exclusive create + write
                try:
                    with open(path, "x") as f:
                        f.write(text)
                    break
                except FileExistsError:
                    n += 1
                    path = out_dir / f"{stem}-{n}.json"
    finally:
        tmp.unlink(missing_ok=True)
    return path


def _clean(o):
    """JSON-safe copy: numpy -> python, non-finite floats -> None, keys -> str, other -> str."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_clean(v) for v in o]
    if hasattr(o, "tolist"):  # numpy scalars / arrays
        return _clean(o.tolist())
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if o is None or isinstance(o, (bool, int, str)):
        return o
    return str(o)


def load_runs(dir) -> list[dict]:
    """All run records in `dir`, sorted by timestamp (empty list if the directory is missing)."""
    d = Path(dir)
    if not d.is_dir():
        return []
    runs = [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]
    return sorted(runs, key=lambda r: r["timestamp"])


def best_run(
    dir,
    run_name: str,
    processed_dir: str,
    where: Callable[[dict], bool] | None = None,
    metric: str = "ndcg@10",
) -> dict | None:
    """The `run_name` record for `processed_dir` with the highest `metric` (None if none match).

    `where` filters further; missing/null metrics rank last.
    """
    runs = [
        r
        for r in load_runs(dir)
        if r["run_name"] == run_name
        and r["extra"].get("processed_dir") == processed_dir
        and (where is None or where(r))
    ]
    if not runs:
        return None
    return max(runs, key=lambda r: r["metrics"].get(metric) or float("-inf"))
