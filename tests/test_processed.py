import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml

import hookrec.data.processed as proc
from hookrec.data.processed import load_processed, preprocess
from hookrec.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = ROOT / "configs"


def _cfg(**kw):
    cfg = load_config(CFG_DIR / "data_50m.yaml")
    cfg["split"]["test_timestamp"] = None  # synth fixture is max-anchored
    cfg["min_count"] = 2
    cfg.update(kw)
    return cfg


@pytest.fixture(scope="module")
def built(processed_synth):
    return processed_synth.path, processed_synth


def test_preprocess_end_to_end_synth(built):
    d, pd_ = built
    n1 = pd_.vocab.n_items + 1
    assert pd_.features.audio.shape[0] == n1 == len(pd_.vocab.raw_ids)
    assert len(pd_.features.pop_train) == n1
    assert len(pd_.seqs_train) > 0 and len(pd_.seqs_val_train) > 0
    assert pd_.seqs_train.items.max() <= pd_.vocab.n_items
    assert len(pd_.truth_test) > 0 and len(pd_.truth_val) > 0
    m = pd_.meta
    assert m["n_cold_test_users"] >= 3
    assert m["share_unknown_test_items"] > 0
    for k in ("peak_rss_mb", "wall_seconds", "config_hash", "n_oov_test_items", "n_dropped_oov"):
        assert k in m
    assert m["n_items"] == pd_.vocab.n_items
    assert pd_.val_events().collect().columns == [
        "uid", "item_idx", "is_organic", "is_positive", "token", "ts"
    ]
    te = pd_.test_events().collect()
    assert te.height == m["n_events"]["test"] and te["item_idx"].min() >= 0
    assert (d / "config.yaml").exists()


def test_cold_users_split(built):
    _, pd_ = built
    assert set(pd_.truth_test_cold) >= {0, 1, 2}
    assert not set(pd_.truth_test_cold) & set(pd_.truth_test)
    assert not set(pd_.truth_test_cold) & set(pd_.seqs_train.uids.tolist())
    assert set(pd_.truth_test) <= set(pd_.seqs_train.uids.tolist())
    assert pd_.meta["n_cold_test_users"] == len(pd_.truth_test_cold)


def test_work_dir_removed(built):
    d, _ = built
    assert not (d / "_work").exists()
    assert (d / "_DONE").exists()


def test_cache_hit_skips_work(synth_dir, tmp_path, monkeypatch):
    cfg = _cfg()
    d1 = preprocess(synth_dir, cfg, tmp_path)
    calls = []
    monkeypatch.setattr(proc, "build_sequences", lambda *a, **k: calls.append(1))
    d2 = preprocess(synth_dir, cfg, tmp_path)
    assert d1 == d2 and not calls


def test_different_config_different_dir(synth_dir, tmp_path):
    a = preprocess(synth_dir, _cfg(), tmp_path)
    b = preprocess(synth_dir, _cfg(max_len_store=64), tmp_path)
    assert a != b and (a / "_DONE").exists() and (b / "_DONE").exists()


def test_killed_run_is_rebuilt(synth_dir, tmp_path):
    cfg = _cfg()
    d = preprocess(synth_dir, cfg, tmp_path)
    (d / "_DONE").unlink()
    (d / "junk.txt").write_text("partial")
    d2 = preprocess(synth_dir, cfg, tmp_path)
    assert d2 == d and (d / "_DONE").exists() and not (d / "junk.txt").exists()
    load_processed(d)


def test_load_requires_done(tmp_path):
    (tmp_path / "x").mkdir()
    with pytest.raises(FileNotFoundError, match="_DONE"):
        load_processed(tmp_path / "x")


def test_sharded_matches_unsharded(synth_dir, tmp_path, built):
    _, ref = built
    d = preprocess(synth_dir, _cfg(n_shards=2), tmp_path)
    got = load_processed(d)
    for a, b in ((ref.seqs_train, got.seqs_train), (ref.seqs_val_train, got.seqs_val_train)):
        for f in ("uids", "offsets", "items", "tokens", "organic", "ratio", "ts"):
            assert np.array_equal(getattr(a, f), getattr(b, f)), f
    for name in ("truth_test", "truth_test_cold", "truth_val", "truth_val_cold"):
        x, y = getattr(ref, name), getattr(got, name)
        assert x.keys() == y.keys()
        assert all(np.array_equal(x[u], y[u]) for u in x)
    assert ref.vocab.n_items == got.vocab.n_items
    assert np.array_equal(ref.vocab.raw_ids, got.vocab.raw_ids)
    assert np.array_equal(ref.features.pop_train, got.features.pop_train)


def test_cli_override(synth_dir, tmp_path, capsys):
    path = ROOT / "scripts" / "run_preprocess.py"
    spec = importlib.util.spec_from_file_location("run_preprocess", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    d = mod.main(
        [
            "--config", str(CFG_DIR / "data_50m.yaml"),
            "--raw", str(synth_dir),
            "--out", str(tmp_path),
            "--override", "min_count=2",
            "--override", "split.test_timestamp=null",
        ]
    )
    assert (Path(d) / "_DONE").exists()
    assert str(d) in capsys.readouterr().out
    saved = yaml.safe_load((Path(d) / "config.yaml").read_text())
    assert saved["min_count"] == 2
    assert saved["split"]["test_timestamp"] is None


def test_val_warm_cold_disjoint(built):
    _, pd_ = built
    assert pd_.truth_val
    assert set(pd_.truth_val) <= set(pd_.seqs_val_train.uids.tolist())
    assert not set(pd_.truth_val) & set(pd_.truth_val_cold)


def test_missing_inputs_listed_in_one_error(tmp_path):
    cfg = _cfg()
    with pytest.raises(FileNotFoundError) as e:
        preprocess(tmp_path / "nope", cfg, tmp_path / "out")
    msg = str(e.value)
    for name in ("multi_event.parquet", "embeddings_50m.parquet", "artist_item_mapping.parquet"):
        assert name in msg


def test_val_test_events_sorted(built):
    _, pd_ = built
    for lf in (pd_.val_events(), pd_.test_events()):
        df = lf.collect()
        assert df.equals(df.sort("uid", "ts", maintain_order=True))
        assert df["item_idx"].min() > 0
