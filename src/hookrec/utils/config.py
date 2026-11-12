"""YAML config loading with `extends` inheritance and dotted-key overrides."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml


def _merge(base: dict, child: dict) -> dict:
    out = copy.deepcopy(base)
    for key, val in child.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _load(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("extends", None)
    if parent is not None:
        cfg = _merge(_load(path.parent / parent), cfg)
    return cfg


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    """Load YAML at `path`, resolve `extends` (child wins), apply dotted-key overrides."""
    cfg = _load(Path(path))
    for dotted, value in (overrides or {}).items():
        *parents, leaf = dotted.split(".")
        node = cfg
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value
    return cfg


def config_hash(cfg: dict) -> str:
    """First 10 hex chars of sha256 of the canonical (sorted-key) JSON."""
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:10]
