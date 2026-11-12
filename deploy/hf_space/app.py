"""Hugging Face Space entry point (Gradio, ZeroGPU).

The demo bundle comes from a local `bundle/` directory next to this file if present, else from
the Hugging Face model repo named by the HOOKSEQ_BUNDLE_REPO environment variable.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

try:  # on ZeroGPU, `spaces` must be imported before torch
    import spaces  # noqa: F401
except ImportError:
    pass

HERE = Path(__file__).resolve().parent

# hookrec is installed WITHOUT its dependencies: its training stack (polars, scikit-learn, scipy,
# matplotlib, implicit, pyarrow) is not needed by the demo modules, which import only numpy and
# torch. pip cannot express --no-deps in requirements.txt, so it is done here, once.
# Pin HOOKREC_REF to a tag (or full commit SHA), never a moving branch: create the tag with
# `git tag v0.1-demo && git push origin v0.1-demo` in the code repo, then use that name here.
HOOKREC_REPO = "git+https://github.com/Arubad/Reccomendation_system.git"
HOOKREC_REF = "v0.1-demo"  # TODO create this tag (or put a commit SHA) before deploying


def ensure_hookrec(ref: str = HOOKREC_REF) -> None:
    if importlib.util.find_spec("hookrec") is not None:
        return
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", f"{HOOKREC_REPO}@{ref}"],
        check=True,
    )


ensure_hookrec()


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
