import json

from hookrec.utils.results import load_runs, save_run


def test_save_run_has_git_sha(tmp_path):
    p = save_run(tmp_path, "sasrec", {"a": 1}, {"ndcg@10": 0.1}, extra={"gpu": "T4"})
    d = json.loads(p.read_text())
    assert set(d) == {"run_name", "timestamp", "git_sha", "config_hash", "config", "metrics", "extra"}
    assert d["git_sha"] and d["extra"] == {"gpu": "T4"}
    assert p.name.startswith("sasrec-") and p.suffix == ".json"


def test_save_run_collision_and_load_sorted(tmp_path):
    paths = {save_run(tmp_path, "r", {}, {"x": float(i)}) for i in range(3)}
    assert len(paths) == 3
    runs = load_runs(tmp_path)
    assert len(runs) == 3
    assert [r["timestamp"] for r in runs] == sorted(r["timestamp"] for r in runs)
    assert load_runs(tmp_path / "missing") == []


def test_save_run_serialises_numpy_path_nan_and_int_keys(tmp_path):
    from pathlib import Path

    import numpy as np

    cfg = {"dir": Path("/x/y"), "lr": np.float32(0.5)}
    metrics = {"ndcg@10": float("nan"), "inf": float("inf"), "by_uid": {np.int64(3): np.float32(0.25)}}
    p = save_run(tmp_path, "r", cfg, metrics)
    d = json.loads(p.read_text())
    assert d["config"] == {"dir": "/x/y", "lr": 0.5}
    assert d["metrics"] == {"ndcg@10": None, "inf": None, "by_uid": {"3": 0.25}}
    assert not list(tmp_path.glob("*.tmp"))


def test_two_saves_same_second_are_distinct_and_no_tmp_left(tmp_path):
    a = save_run(tmp_path, "r", {}, {"x": 1.0})
    b = save_run(tmp_path, "r", {}, {"x": 2.0})
    assert a != b and a.exists() and b.exists()
    assert json.loads(a.read_text())["metrics"] == {"x": 1.0}
    assert not list(tmp_path.glob("*.tmp"))
