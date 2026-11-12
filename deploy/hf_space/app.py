"""Hugging Face Space entry point (Gradio, ZeroGPU).

The demo bundle comes from a local `bundle/` directory next to this file if present, else from
the Hugging Face model repo named by the HOOKSEQ_BUNDLE_REPO environment variable.
"""

import os
from pathlib import Path

try:  # on ZeroGPU, `spaces` must be imported before torch
    import spaces  # noqa: F401
except ImportError:
    pass

HERE = Path(__file__).resolve().parent


def resolve_bundle(here: Path = HERE) -> Path:
    local = here / "bundle"
    if (local / "meta.json").exists():
        return local
    repo = os.environ.get("HOOKSEQ_BUNDLE_REPO")
    if not repo:
        raise RuntimeError(
            "no bundle: put one in bundle/ next to app.py, or set HOOKSEQ_BUNDLE_REPO to the "
            "Hugging Face model repo that holds it (see docs/hf_space.md)"
        )
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=repo, repo_type="model"))


def build_demo(bundle_dir: Path | None = None):
    from hookrec.demo.app import build_app
    from hookrec.demo.engine import DemoEngine

    return build_app(DemoEngine(bundle_dir or resolve_bundle()))


if __name__ == "__main__":
    build_demo().launch()
