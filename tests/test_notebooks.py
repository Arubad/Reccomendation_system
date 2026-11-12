"""Generated Kaggle notebooks: structure, sync with the generator, and a local end-to-end run.

The end-to-end test executes each notebook's own cells (params, discovery, work, summary; the
clone/install cell is skipped) with the /kaggle paths redirected to tmp dirs, chaining every
stage's output into the next stage's inputs exactly as Kaggle would mount them.
"""

import ast
import importlib.util
import json
import os
import re
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks" / "kaggle"
NAMES = ["k0_download", "k1_eda", "k2_preprocess", "k3_baselines", "k4_train", "k5_eval_report"]
CPU_ONLY = {"k0_download", "k1_eda", "k2_preprocess", "k3_baselines"}


def _nb(name):
    return json.loads((NB_DIR / f"{name}.ipynb").read_text())


def _code(nb) -> list[str]:
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _gen():
    spec = importlib.util.spec_from_file_location(
        "make_notebooks", ROOT / "scripts" / "make_notebooks.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("name", NAMES)
def test_generated_notebooks_valid(name):
    nb = _nb(name)
    code = _code(nb)
    assert 1 <= len(code) <= 6
    meta = nb["metadata"]
    assert meta["kernelspec"]["name"] == "python3"
    k = meta["kaggle"]
    assert k["isInternetEnabled"] is True
    if name in CPU_ONLY or name == "k5_eval_report":
        assert k["accelerator"] == "none" and k["isGpuEnabled"] is False
    if name == "k4_train":
        assert k["accelerator"] != "none" and k["isGpuEnabled"] is True
    src = "\n".join(code)
    scripts = set(re.findall(r"scripts/(run_\w+\.py)", src))
    assert scripts, "notebook calls no script"
    for s in scripts:
        script = ROOT / "scripts" / s
        assert script.exists()
        # every CLI flag the notebook passes must be one the script defines
        text = script.read_text()
        for cell in code:
            if s in cell:
                for flag in re.findall(r"\"(--[a-z][a-z-]*)\"", cell):
                    assert f'"{flag}"' in text, f"{name}: {s} has no flag {flag}"
    for cell in code:
        ast.parse(cell, feature_version=(3, 10))  # Kaggle runs Python 3.10+
        for line in cell.splitlines():
            assert not line.lstrip().startswith(("!", "%")), f"{name}: shell magic {line!r}"
    params = code[0]
    for p in ("REPO_URL", "GIT_REF", "SIZE", "HOOKREC_SESSION_START"):
        assert p in params
    assert "input(" not in src


def test_notebooks_up_to_date(tmp_path):
    params = _code(_nb("k0_download"))[0]
    url = re.search(r'^REPO_URL = "(.*)"', params, re.MULTILINE).group(1)
    ref = re.search(r'^GIT_REF = "(.*)"', params, re.MULTILINE).group(1)
    _gen().main(["--out", str(tmp_path), "--repo-url", url, "--git-ref", ref])
    got = sorted(p.name for p in tmp_path.glob("*.ipynb"))
    assert got == sorted(f"{n}.ipynb" for n in NAMES)
    for n in got:
        assert (tmp_path / n).read_bytes() == (NB_DIR / n).read_bytes(), f"{n} is stale"


# --- end-to-end: the notebooks' own cells on synthetic data -------------------------------

TINY_TRAIN = [
    "train.max_steps=6", "train.warmup_steps=1", "train.eval_every_steps=3",
    "train.ckpt_every_steps=3", "train.batch_size=8", "train.num_workers=0",
    "train.device=cpu", "loss.n_uniform_negatives=16", "loss.n_in_batch_negatives=8",
    "model.d_model=16", "model.n_heads=1", "model.n_layers=1", "model.max_len=16",
]  # fmt: skip


@pytest.fixture
def kaggle_fs(tmp_path, synth_dir):
    """A fake /kaggle: input/ holds a public 'mirror' of Yambda; returns a stage runner."""
    saved = os.environ.get("HOOKREC_SESSION_START")
    inputs = tmp_path / "input"
    shutil.copytree(synth_dir, inputs / "yambda-mirror")

    def run(name, attach_as=None, **params):
        work = tmp_path / f"work-{name}-{len(list(tmp_path.glob('work-*')))}"
        g = {"__name__": "__notebook__"}
        code = _code(_nb(name))
        os.environ.pop("HOOKREC_SESSION_START", None)
        exec(code[0], g)  # noqa: S102  (params cell, as written)
        g.update(
            INPUT_ROOT=str(inputs), WORK=str(work), OUT=str(work / "out"), REPO=str(ROOT), **params
        )
        for cell in code[2:]:  # skip clone + pip install
            exec(cell, g)  # noqa: S102
        if attach_as:  # "Add Input -> Notebook output" mounts the whole output dir
            shutil.copytree(work, inputs / attach_as)
        return g, work / "out"

    yield run
    if saved is None:
        os.environ.pop("HOOKREC_SESSION_START", None)
    else:
        os.environ["HOOKREC_SESSION_START"] = saved


def _runs(results: Path, prefix: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(results.glob(f"{prefix}*.json"))]


def test_pipeline_end_to_end(kaggle_fs, capsys):
    over_data = ["min_count=2", "split.test_timestamp=null"]
    _, out = kaggle_fs("k0_download", attach_as="k0")
    assert (out / "raw" / "_HOOKREC_RAW_50m").exists()
    assert "mirror" in capsys.readouterr().out  # no HF download: the attached mirror was used

    _, out = kaggle_fs("k1_eda", TEST_TIMESTAMP="none")
    assert (out / "reports" / "_HOOKREC_EDA").exists()
    assert (out / "reports" / "eda_stats.json").exists()

    _, out = kaggle_fs("k2_preprocess", attach_as="k2", OVERRIDES=over_data)
    assert len(list((out / "processed").glob("50m-*/_DONE"))) == 1

    _, out = kaggle_fs("k3_baselines", attach_as="k3", MODELS="mostpop,decaypop", QUICK=True)
    assert (out / "results" / "_HOOKREC_RESULTS_baselines").exists()
    assert _runs(out / "results", "mostpop-test") and _runs(out / "results", "decaypop-test")

    # K4 session 1 is cut by max_hours: no record, a clean RESUME NEEDED, a checkpoint saved
    cut = TINY_TRAIN + ["train.max_hours=0"]
    g, out = kaggle_fs("k4_train", attach_as="k4-v1", REQUIRE_GPU=False, OVERRIDES=cut)
    assert g["RESUME_NEEDED"] is True
    assert "RESUME NEEDED" in capsys.readouterr().out
    assert not _runs(out / "results", "a0_sasrec-")
    assert (out / "ckpt" / "a0_sasrec" / "_HOOKREC_CKPT").exists()
    assert list((out / "ckpt" / "a0_sasrec" / "val").glob("ckpt-*.pt"))

    # session 2 has v1 attached: it resumes val, then trains test
    g, out = kaggle_fs("k4_train", attach_as="k4-v2", REQUIRE_GPU=False, OVERRIDES=TINY_TRAIN)
    assert g["RESUME_NEEDED"] is False
    log = capsys.readouterr().out
    assert "--resume" in log and "resumed from" in log  # the v1 checkpoint was loaded
    val, test = _runs(out / "results", "a0_sasrec-val"), _runs(out / "results", "a0_sasrec-test")
    assert len(val) == 1 and len(test) == 1 and val[0]["extra"]["steps"] == 6
    assert (out / "ckpt" / "a0_sasrec" / "test" / "model_final.pt").exists()
    # a finished split drops its step checkpoints (resume state; ~2.3 GB each at 500M)
    assert not list((out / "ckpt" / "a0_sasrec").rglob("ckpt-*"))
    assert (out / "ckpt" / "a0_sasrec" / "val" / "best.pt").exists()

    # session 3 with v2 attached: both records already exist, so nothing is retrained
    g, out = kaggle_fs("k4_train", REQUIRE_GPU=False, OVERRIDES=TINY_TRAIN)
    assert "skip" in capsys.readouterr().out.lower()
    assert len(_runs(out / "results", "a0_sasrec-test")) == 1
    assert not (out / "ckpt" / "a0_sasrec" / "test").exists()

    g, out = kaggle_fs("k5_eval_report")
    evals = {r["extra"]["source_run"] for r in _runs(out / "results", "")
             if r["extra"].get("kind") == "eval"}  # fmt: skip
    assert evals == {"mostpop-test", "decaypop-test", "a0_sasrec-test"}
    md = (out / "reports" / "results.md").read_text()
    assert "a0_sasrec" in md and "mostpop" in md
    assert "# Results" in capsys.readouterr().out
