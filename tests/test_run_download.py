import importlib.util
import json
import shutil
from pathlib import Path

import polars as pl
import pytest

from hookrec.data import download as dl
from hookrec.data.processed import load_processed, preprocess
from hookrec.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fake_hf(synth_dir, monkeypatch):
    """hf_hub_download replaced by a copy from the synthetic Yambda dir; records filenames."""
    calls = []

    def fake(repo_id, filename, repo_type, local_dir):
        calls.append(filename)
        dest = Path(local_dir) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        src = synth_dir / filename
        if src.exists():
            shutil.copyfile(src, dest)
        else:  # album_item_mapping is not in the fixture
            pl.DataFrame({"item_id": [1], "album_id": [1]}).write_parquet(dest)
        (Path(local_dir) / ".cache" / "huggingface").mkdir(parents=True, exist_ok=True)
        return str(dest)

    monkeypatch.setattr(dl, "hf_hub_download", fake)
    return calls


def _check_raw(out: Path, synth_dir: Path):
    assert (out / "_HOOKREC_RAW_50m").exists()
    assert (out / "flat/50m/multi_event.parquet").exists()
    assert (out / "artist_item_mapping.parquet").exists()
    assert not (out / "embeddings.parquet").exists()  # full file removed: output stays lean
    assert not (out / ".cache").exists()
    emb = pl.read_parquet(out / "embeddings_50m.parquet")
    assert emb.columns == ["item_id", "embed"] and isinstance(emb.schema["embed"], pl.Array)
    assert emb.schema["embed"].inner == pl.Float16
    event_items = set(pl.read_parquet(out / "flat/50m/multi_event.parquet")["item_id"].to_list())
    assert set(emb["item_id"].to_list()) <= event_items
    meta = json.loads((out / "_HOOKREC_RAW_50m").read_text())
    assert meta["size"] == "50m" and meta["n_embeddings"] == emb.height
    assert not list(out.rglob("*.tmp"))


def test_run_download_hf_idempotent(fake_hf, synth_dir, tmp_path):
    mod = _load("run_download")
    out = tmp_path / "raw"
    mod.main(["--size", "50m", "--out", str(out)])
    _check_raw(out, synth_dir)
    assert "flat/50m/multi_event.parquet" in fake_hf and "embeddings.parquet" in fake_hf
    n = len(fake_hf)
    mod.main(["--size", "50m", "--out", str(out)])  # complete: nothing is fetched again
    assert len(fake_hf) == n
    _check_raw(out, synth_dir)


def test_run_download_resumes_partial(fake_hf, synth_dir, tmp_path):
    mod = _load("run_download")
    out = tmp_path / "raw"
    (out / "flat/50m").mkdir(parents=True)
    shutil.copyfile(
        synth_dir / "flat/50m/multi_event.parquet", out / "flat/50m/multi_event.parquet"
    )
    mod.main(["--size", "50m", "--out", str(out)])
    assert "flat/50m/multi_event.parquet" not in fake_hf
    _check_raw(out, synth_dir)


def test_run_download_from_mirror_skips_hf(fake_hf, synth_dir, tmp_path):
    mod = _load("run_download")
    out = tmp_path / "raw"
    mod.main(["--size", "50m", "--out", str(out), "--mirror", str(synth_dir)])
    assert fake_hf == []
    _check_raw(out, synth_dir)
    assert (synth_dir / "embeddings.parquet").exists()  # the read-only mirror is untouched


def test_k0_output_feeds_eda_and_preprocess(fake_hf, synth_dir, tmp_path):
    """The stage contract: K1 and K2 run on K0's lean raw dir with the shipped config."""
    out = tmp_path / "raw"
    _load("run_download").main(["--size", "50m", "--out", str(out)])
    _load("run_eda").main(
        [
            "--raw", str(out), "--size", "50m", "--out", str(tmp_path / "reports"),
            "--test-timestamp", "none",
            "--embeddings-file", "embeddings_50m.parquet", "--emb-column", "embed",
        ]
    )  # fmt: skip
    assert (tmp_path / "reports" / "eda_stats.json").exists()
    cfg = load_config(ROOT / "configs" / "data_50m.yaml")
    cfg["split"]["test_timestamp"] = None
    cfg["min_count"] = 2
    pd_ = load_processed(preprocess(out, cfg, tmp_path / "processed"))
    assert pd_.features.has_audio[1:].any()  # audio came from the filtered f16 file
