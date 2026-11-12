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
