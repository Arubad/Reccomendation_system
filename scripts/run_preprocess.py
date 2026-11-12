"""Run the cached preprocessing pipeline.

python scripts/run_preprocess.py --config configs/data_50m.yaml --raw DIR --out DIR \
    [--override min_count=2 --override split.test_timestamp=null]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from hookrec.data.processed import preprocess
from hookrec.utils.config import load_config


def _parse_overrides(items: list[str]) -> dict:
    out = {}
    for item in items:
        key, sep, val = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"--override expects key=value, got {item!r}")
        out[key] = yaml.safe_load(val)
    return out


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True, help="Yambda root (flat/<size>/...)")
    ap.add_argument("--out", type=Path, required=True, help="root for processed bundles")
    ap.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config(args.config, _parse_overrides(args.override))
    out = preprocess(args.raw, cfg, args.out)
    meta = json.loads((out / "meta.json").read_text())
    print(f"processed dir: {out}")
    print(json.dumps(meta, indent=2))
    return out


if __name__ == "__main__":
    main()
