"""Run-record JSON files: one per training/eval run, written under a results directory."""

from __future__ import annotations

import json
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
    stem = f"{run_name}-{now.strftime('%Y%m%d-%H%M%S')}"
    path = out_dir / f"{stem}.json"
    n = 0
    while True:
        try:
            with open(path, "x") as f:
                json.dump(record, f, indent=2, default=_jsonable)
            return path
        except FileExistsError:
            n += 1
            path = out_dir / f"{stem}-{n}.json"


def _jsonable(o):
    if hasattr(o, "tolist"):  # numpy scalars / arrays
        return o.tolist()
    return str(o)


def load_runs(dir) -> list[dict]:
    """All run records in `dir`, sorted by timestamp (empty list if the directory is missing)."""
    d = Path(dir)
    if not d.is_dir():
        return []
    runs = [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]
    return sorted(runs, key=lambda r: r["timestamp"])
