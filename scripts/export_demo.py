"""Export the self-contained demo bundle from trained HookSeq and SASRec checkpoints.

python scripts/export_demo.py --processed DIR --hookseq-ckpt DIR_OR_model_final.pt \
    --sasrec-ckpt DIR_OR_model_final.pt --out bundle/ [--n-users 200 --n-items 50000]

The bundle (< 200 MB) needs no processed data: upload it to a Hugging Face model repo and point
the Space's HOOKSEQ_BUNDLE_REPO at it (docs/hf_space.md).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from hookrec.data.processed import load_processed
from hookrec.demo.bundle import export_demo_bundle

log = logging.getLogger("export_demo")


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed", type=Path, required=True)
    ap.add_argument("--hookseq-ckpt", type=Path, required=True)
    ap.add_argument("--sasrec-ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-users", type=int, default=200)
    ap.add_argument("--n-items", type=int, default=50_000)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    data = load_processed(args.processed)
    out = export_demo_bundle(
        data, args.hookseq_ckpt, args.sasrec_ckpt, args.out, args.n_users, args.n_items
    )
    log.info("wrote %s", out)
    return out


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("export_demo failed")
        sys.exit(1)
