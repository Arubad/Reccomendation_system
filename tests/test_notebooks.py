"""Generated Kaggle notebooks: structure, sync with the generator, and a local end-to-end run.

The end-to-end test executes each notebook's own cells (params, discovery, work, summary; the
clone/install cell is skipped) with the /kaggle paths redirected to tmp dirs, chaining every
stage's output into the next stage's inputs exactly as Kaggle would mount them. The
clone/install cell has its own tests against a local bare repo.
"""

import ast
import base64
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks" / "kaggle"
NAMES = [
    "k0_download", "k1_eda", "k2_preprocess", "k3_baselines", "k4_train", "k4p_profile",
    "k5_eval_report",
]  # fmt: skip
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
    if name in ("k4_train", "k4p_profile"):
        assert k["accelerator"] != "none" and k["isGpuEnabled"] is True
    src = "\n".join(code)
    scripts = set(re.findall(r"scripts/((?:run_\w+|profile_train)\.py)", src))
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


def test_every_notebook_including_k4p_is_pinned_to_the_git_ref():
    """v0.3.1 contains the profiler, so K4P follows --git-ref like K0-K5 (no dev branch)."""
    gen = _gen()
    assert gen.DEFAULT_GIT_REF == "v0.3.1"
    assert not hasattr(gen, "PROFILE_GIT_REF")
    nbs = gen.notebooks("https://example.invalid/r.git", "some-tag")
    for name, nb in nbs.items():
        assert re.search(r'^GIT_REF = "some-tag"', _code(nb)[0], re.MULTILINE), name
    md = "".join(nbs["k4p_profile"]["cells"][0]["source"])
    assert "hookseq-impl" not in md
    shipped = sorted(NB_DIR.glob("*.ipynb"))
    assert len(shipped) == len(nbs)
    for p in shipped:  # the committed files, K4P included
        params = _code(json.loads(p.read_text()))[0]
        assert re.search(r'^GIT_REF = "v0.3.1"', params, re.MULTILINE), p.name


def test_k4p_profiles_both_configs(processed_synth, tmp_path):
    """K4P's own cells (clone skipped) on the synthetic bundle, on CPU with tiny shapes."""
    inputs = tmp_path / "input"
    shutil.copytree(processed_synth.path, inputs / "k2" / "processed" / processed_synth.path.name)
    saved = os.environ.get("HOOKREC_SESSION_START")
    g = {"__name__": "__notebook__"}
    code = _code(_nb("k4p_profile"))
    try:
        exec(code[0], g)  # noqa: S102
        assert g["CONFIGS"] == ["configs/sasrec.yaml", "configs/hookseq.yaml"]
        size = processed_synth.path.name.split("-")[0]
        tiny = [o for o in TINY_TRAIN if not o.startswith("train.max_steps")]
        g.update(
            INPUT_ROOT=str(inputs), WORK=str(tmp_path / "work"), OUT=str(tmp_path / "work/out"),
            REPO=str(ROOT), SIZE=size, REQUIRE_GPU=False, STEPS=2, WARMUP=1,
            THROUGHPUT_STEPS=1, PROFILE_STEPS=1, LOADER_WORKERS="0", OVERRIDES=tiny,
        )  # fmt: skip
        for cell in code[2:]:
            exec(cell, g)  # noqa: S102
    finally:
        if saved is None:
            os.environ.pop("HOOKREC_SESSION_START", None)
        else:
            os.environ["HOOKREC_SESSION_START"] = saved
    prof = tmp_path / "work" / "out" / "profile"
    assert (prof / "_HOOKREC_PROFILE").exists()
    for name in ("sasrec", "hookseq"):
        r = json.loads((prof / name / "profile.json").read_text())
        assert r["config"]["name"] == name and r["phased"]["steps"] == 2


def test_notebooks_up_to_date(tmp_path):
    params = _code(_nb("k0_download"))[0]
    url = re.search(r'^REPO_URL = "(.*)"', params, re.MULTILINE).group(1)
    ref = re.search(r'^GIT_REF = "(.*)"', params, re.MULTILINE).group(1)
    _gen().main(["--out", str(tmp_path), "--repo-url", url, "--git-ref", ref])
    got = sorted(p.name for p in tmp_path.glob("*.ipynb"))
    assert got == sorted(f"{n}.ipynb" for n in NAMES)
    for n in got:
        assert (tmp_path / n).read_bytes() == (NB_DIR / n).read_bytes(), f"{n} is stale"


def test_polars_floor_matches_pyproject():
    m = re.search(r'"polars>=([\d.]+)"', (ROOT / "pyproject.toml").read_text())
    assert m and m.group(1) == _gen().POLARS_FLOOR


# --- the clone + install cell -------------------------------------------------------------

FAKE_TOKEN = "ghp_FAKE0123456789abcdefTOKEN"
FAKE_URL = "https://example.invalid/hookseq.git"


def _git(*args, cwd=None):
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=cwd, check=True, capture_output=True,
    )  # fmt: skip


@pytest.fixture
def install_cell(tmp_path, monkeypatch):
    """Runs cells 1-2 against a local bare repo that git serves as FAKE_URL (insteadOf), with a
    stub kaggle_secrets returning FAKE_TOKEN and `pip install -e` replaced by a no-op (the dev
    venv's own editable install must not be repointed at the tmp clone)."""
    src = tmp_path / "src-repo"
    (src / "src" / "hookrec").mkdir(parents=True)
    (src / "src" / "hookrec" / "__init__.py").write_text("MARK = 'cloned'\n")
    _git("init", "-q", "-b", "main", str(src))
    _git("add", "-A", cwd=src)
    _git("commit", "-q", "-m", "init", cwd=src)
    bare = tmp_path / "bare.git"
    _git("clone", "-q", "--bare", str(src), str(bare))
    home = tmp_path / "home"
    home.mkdir()
    (home / ".gitconfig").write_text(f'[url "{bare}"]\n\tinsteadOf = {FAKE_URL}\n')
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    monkeypatch.setenv("HOOKREC_SESSION_START", "1")  # recorded, so the cell's value is undone
    monkeypatch.delenv("HOOKREC_SESSION_START")

    secrets = types.ModuleType("kaggle_secrets")

    class UserSecretsClient:
        def get_secret(self, label):
            assert label == "GITHUB_TOKEN"
            return FAKE_TOKEN

    secrets.UserSecretsClient = UserSecretsClient
    monkeypatch.setitem(sys.modules, "kaggle_secrets", secrets)
    monkeypatch.setattr(sys, "path", list(sys.path))  # the cell prepends REPO/src
    code = _code(_nb("k0_download"))
    pip_e = '"-m", "pip", "install", "-q", "-e", REPO]'
    assert pip_e in code[1]
    install = code[1].replace(pip_e, '"-c", "print(\'pip stub\')"]')

    def run(git_ref="main", polars_min=None):
        g = {"__name__": "__notebook__"}
        exec(code[0], g)  # noqa: S102
        repo = tmp_path / "work" / "repo"
        g.update(REPO_URL=FAKE_URL, GIT_REF=git_ref, REPO=str(repo))
        cell = install
        if polars_min is not None:
            assert "< (2, 1):" in cell
            cell = cell.replace("< (2, 1):", f"< {polars_min!r}:")
        exec(cell, g)  # noqa: S102
        return g, repo

    return run


def test_install_cell_clones_without_leaking_the_token(install_cell, capsys):
    g, repo = install_cell()
    assert re.fullmatch(r"[0-9a-f]{40}", g["SHA"])
    assert sys.path[0] == str(repo / "src")
    out = subprocess.run(
        [sys.executable, "-c", "import hookrec; print(hookrec.MARK, hookrec.__file__)"],
        env={**os.environ, "PYTHONPATH": str(repo / "src")}, capture_output=True, text=True,
        check=True,
    ).stdout  # fmt: skip
    assert out.startswith("cloned ") and str(repo) in out
    basic = base64.b64encode(f"x-access-token:{FAKE_TOKEN}".encode())
    for p in (repo / ".git").rglob("*"):  # config, logs/HEAD, logs/refs/..., packed-refs, ...
        if p.is_file():
            data = p.read_bytes()
            assert FAKE_TOKEN.encode() not in data and basic not in data, p
    assert (repo / ".git" / "config").read_text().count(FAKE_URL) == 1
    assert list((repo / ".git" / "logs").rglob("*"))  # the reflogs were checked
    assert FAKE_TOKEN not in capsys.readouterr().out


def test_install_cell_bad_ref_fails_with_token_masked(install_cell):
    with pytest.raises(RuntimeError, match="no-such-branch") as e:
        install_cell(git_ref="no-such-branch")
    assert FAKE_TOKEN not in str(e.value) and "GITHUB_TOKEN" in str(e.value)


def test_install_cell_rejects_old_polars(install_cell):
    with pytest.raises(RuntimeError, match=r"polars .* needs polars>="):
        install_cell(polars_min=(99, 0))


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


def test_k5_500m_without_gpu_fails_fast(kaggle_fs, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="SIZE='500m' needs the GPU"):
        kaggle_fs("k5_eval_report", SIZE="500m")


def test_pipeline_end_to_end(kaggle_fs, tmp_path, capsys):
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
    g, out = kaggle_fs(
        "k4_train", attach_as="k4-v1", REQUIRE_GPU=False, OVERRIDES=cut,
        CONFIG="configs/ablations/a0_sasrec.yaml", MIN_START_MINUTES=0,
    )  # a plain string CONFIG still works (older single-config notebooks)
    assert g["RESUME_NEEDED"] is True
    assert "RESUME NEEDED" in capsys.readouterr().out
    assert not _runs(out / "results", "a0_sasrec-")
    assert (out / "ckpt" / "a0_sasrec" / "_HOOKREC_CKPT").exists()
    assert list((out / "ckpt" / "a0_sasrec" / "val").glob("ckpt-*.pt"))

    # session 2 has v1 attached: it resumes val, then trains test
    g, out = kaggle_fs(
        "k4_train", attach_as="k4-v2", REQUIRE_GPU=False, OVERRIDES=TINY_TRAIN,
        CONFIGS=["configs/ablations/a0_sasrec.yaml"],
    )
    assert g["RESUME_NEEDED"] is False
    log = capsys.readouterr().out
    assert "--resume" in log and "resumed from" in log  # the v1 checkpoint was loaded
    val, test = _runs(out / "results", "a0_sasrec-val"), _runs(out / "results", "a0_sasrec-test")
    assert len(val) == 1 and len(test) == 1 and val[0]["extra"]["steps"] == 6
    assert (out / "ckpt" / "a0_sasrec" / "test" / "model_final.pt").exists()
    # a finished split drops its step checkpoints (resume state; ~2.3 GB each at 500M)
    assert not list((out / "ckpt" / "a0_sasrec").rglob("ckpt-*"))
    assert (out / "ckpt" / "a0_sasrec" / "val" / "best.pt").exists()

    # session 3 with v1 + v2 attached: both records exist, so nothing is retrained, but the
    # finished splits' weights are carried forward with the records
    g, out = kaggle_fs(
        "k4_train", attach_as="k4-v3", REQUIRE_GPU=False, OVERRIDES=TINY_TRAIN,
        CONFIGS=["configs/ablations/a0_sasrec.yaml"],
    )
    assert "skip" in capsys.readouterr().out.lower()
    assert len(_runs(out / "results", "a0_sasrec-test")) == 1
    ck = out / "ckpt" / "a0_sasrec"
    assert (ck / "test" / "model_final.pt").exists() and (ck / "val" / "best.pt").exists()
    assert not list(ck.rglob("ckpt-*"))

    # K5 with only the newest K4 output attached (v3 never trained anything itself)
    shutil.rmtree(tmp_path / "input" / "k4-v1")
    shutil.rmtree(tmp_path / "input" / "k4-v2")
    g, out = kaggle_fs("k5_eval_report")
    evals = {r["extra"]["source_run"] for r in _runs(out / "results", "")
             if r["extra"].get("kind") == "eval"}  # fmt: skip
    assert evals == {"mostpop-test", "decaypop-test", "a0_sasrec-test"}
    md = (out / "reports" / "results.md").read_text()
    assert "a0_sasrec" in md and "mostpop" in md
    assert "# Results" in capsys.readouterr().out


# --- K4 trains a list of configs ----------------------------------------------------------

A0 = "configs/ablations/a0_sasrec.yaml"


def _tiny_cfg(tmp_path, stem, max_hours):
    """A config that inherits a0 (absolute `extends`) with its own train.max_hours."""
    p = tmp_path / "cfgs" / f"{stem}.yaml"
    p.parent.mkdir(exist_ok=True)
    p.write_text(f"extends: {ROOT / A0}\ntrain: {{max_hours: {max_hours}}}\n")
    return str(p)


@pytest.fixture
def k2_ready(kaggle_fs):
    kaggle_fs("k0_download", attach_as="k0")
    kaggle_fs("k2_preprocess", attach_as="k2", OVERRIDES=["min_count=2", "split.test_timestamp=null"])
    return kaggle_fs


def test_k4_params_default_to_the_ablation_list():
    g = {}
    exec(_code(_nb("k4_train"))[0], g)  # noqa: S102
    assert g["CONFIGS"] == [
        "configs/ablations/a0_sasrec.yaml", "configs/ablations/a1_audio.yaml",
        "configs/ablations/a2_audio_event_organic.yaml", "configs/ablations/a3_full_logq.yaml",
        "configs/ablations/a4_full_dislike.yaml",
    ]  # fmt: skip
    assert g["MIN_START_MINUTES"] == 40 and "CONFIG" not in g
    md = "".join(_nb("k4_train")["cells"][0]["source"])
    assert "CONFIG =" in md and "MIN_START_MINUTES" in md and "at most 2" in md


def test_k4_list_budget_cut_then_resume_across_versions(k2_ready, tmp_path, capsys):
    a, b, c = (_tiny_cfg(tmp_path, n, h) for n, h in (("tiny_a", 8), ("tiny_b", 0), ("tiny_c", 8)))
    kw = {"REQUIRE_GPU": False, "OVERRIDES": TINY_TRAIN, "MIN_START_MINUTES": 0}
    # v1: tiny_a completes; tiny_b is cut by max_hours (RESUME NEEDED); tiny_c is never started
    g, out = k2_ready("k4_train", attach_as="k4-v1", CONFIGS=[a, b, c], **kw)
    log = capsys.readouterr().out
    assert g["RESUME_NEEDED"] is True and "RESUME NEEDED" in log
    assert "Still pending: tiny_b-val, tiny_b-test, tiny_c-val, tiny_c-test" in log
    assert "tiny_a: val, test done" in log
    assert "| tiny_a | done |" in log and "| tiny_b | pending |" in log
    assert len(_runs(out / "results", "tiny_a-")) == 2 and not _runs(out / "results", "tiny_b-")
    assert list((out / "ckpt" / "tiny_b" / "val").glob("ckpt-*.pt"))
    assert not (out / "ckpt" / "tiny_c").exists()
    assert (out / "ckpt" / "tiny_a" / "test" / "model_final.pt").exists()

    # v2 (v1 attached, tiny_b now has budget): tiny_a skipped and carried forward, tiny_b
    # resumes from its checkpoint, tiny_c starts fresh
    _tiny_cfg(tmp_path, "tiny_b", 8)
    g, out = k2_ready("k4_train", attach_as="k4-v2", CONFIGS=[a, b, c], **kw)
    log = capsys.readouterr().out
    assert g["RESUME_NEEDED"] is False
    assert "tiny_a-val: a record already exists, skipping" in log
    assert "resumed from" in log
    for n in ("tiny_a", "tiny_b", "tiny_c"):
        assert len(_runs(out / "results", f"{n}-val")) == 1, n
        assert len(_runs(out / "results", f"{n}-test")) == 1, n
        assert (out / "ckpt" / n / "test" / "model_final.pt").exists(), n
        assert not list((out / "ckpt" / n).rglob("ckpt-*")), n
        assert f"{n}: val, test done" in log
    assert _runs(out / "results", "tiny_b-val")[0]["extra"]["steps"] == 6
    assert "| tiny_c | done |" in log


def test_k4_guard_skips_start_when_budget_is_nearly_spent(k2_ready, monkeypatch, capsys):
    from hookrec.train import trainer

    # 7.5 h into the default 8 h budget: 30 min left < MIN_START_MINUTES (40)
    monkeypatch.setattr(trainer, "session_elapsed_seconds", lambda: 7.5 * 3600)
    g, out = k2_ready(
        "k4_train", REQUIRE_GPU=False, OVERRIDES=TINY_TRAIN, CONFIGS=[A0, "configs/ablations/a1_audio.yaml"]
    )
    log = capsys.readouterr().out
    assert g["RESUME_NEEDED"] is True and "RESUME NEEDED" in log
    assert "30 min" in log and "MIN_START_MINUTES" in log
    assert "Still pending: a0_sasrec-val, a0_sasrec-test, a1_audio-val, a1_audio-test" in log
    assert not _runs(out / "results", "a0_sasrec")
    assert not list((out / "ckpt").rglob("*.pt"))


def test_k4_plain_string_config_is_a_one_item_list(k2_ready, monkeypatch):
    from hookrec.train import trainer

    monkeypatch.setattr(trainer, "session_elapsed_seconds", lambda: 8 * 3600)
    g, _ = k2_ready("k4_train", REQUIRE_GPU=False, OVERRIDES=TINY_TRAIN, CONFIG=A0)
    assert g["CONFIGS"] == [A0] and [p["name"] for p in g["PLAN"]] == ["a0_sasrec"]


def test_k4_failure_fails_the_version_and_stops_the_loop(k2_ready, tmp_path, capsys):
    a, b = _tiny_cfg(tmp_path, "tiny_a", 8), _tiny_cfg(tmp_path, "tiny_b", 8)
    with pytest.raises(RuntimeError, match=r"(?s)K4 failed on tiny_a-val.*not started"):
        k2_ready(
            "k4_train", REQUIRE_GPU=False, CONFIGS=[a, b], MIN_START_MINUTES=0,
            OVERRIDES=[*TINY_TRAIN, "train.device=bogus"],
        )  # fmt: skip
    log = capsys.readouterr().out
    assert "| tiny_a | failed |" in log and "| tiny_b | pending |" in log
    assert "tiny_b-val" not in log.split("$ ")[-1] and not list(tmp_path.rglob("tiny_b-val-*.json"))


def test_k3_sets_single_thread_blas():
    code = "\n".join(_code(_nb("k3_baselines")))
    assert "OPENBLAS_NUM_THREADS" in code and "MKL_NUM_THREADS" in code
    assert "env=BLAS_ENV" in code
