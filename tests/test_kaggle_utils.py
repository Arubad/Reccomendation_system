import json
import os
import sys
import time

import pytest

from hookrec.utils import kaggle


def _tree(root):
    (root / "k0-out" / "raw").mkdir(parents=True)
    (root / "k0-out" / "raw" / "_HOOKREC_RAW_50m").write_text("{}")
    (root / "mirror" / "yambda" / "flat" / "50m").mkdir(parents=True)
    (root / "mirror" / "yambda" / "flat" / "50m" / "multi_event.parquet").write_bytes(b"x")
    for name in ("50m-aaa", "50m-bbb", "500m-ccc"):
        (root / "k2" / "processed" / name).mkdir(parents=True)
    for name in ("50m-aaa", "500m-ccc"):
        (root / "k2" / "processed" / name / "_DONE").write_text("")
    return root


def test_find_inputs_marker_and_nested_path(tmp_path):
    root = _tree(tmp_path)
    assert kaggle.find_inputs("_HOOKREC_RAW_50m", root) == [root / "k0-out" / "raw"]
    # a multi-part marker returns the dir it is relative to (the Yambda root)
    got = kaggle.find_inputs("flat/50m/multi_event.parquet", root)
    assert got == [root / "mirror" / "yambda"]
    assert kaggle.find_inputs("_DONE", root) == [
        root / "k2" / "processed" / "500m-ccc",
        root / "k2" / "processed" / "50m-aaa",
    ]
    assert kaggle.find_inputs("_HOOKREC_RAW_500m", root) == []
    assert kaggle.find_inputs("_DONE", tmp_path / "missing") == []


def test_find_inputs_glob_marker(tmp_path):
    for name in ("a0_sasrec", "baselines"):
        (tmp_path / name / "results").mkdir(parents=True)
        (tmp_path / name / "results" / f"_HOOKREC_RESULTS_{name}").write_text("")
    assert kaggle.find_inputs("_HOOKREC_RESULTS_*", tmp_path) == [
        tmp_path / "a0_sasrec" / "results",
        tmp_path / "baselines" / "results",
    ]


def test_find_one_errors_name_marker_and_list_root(tmp_path):
    root = _tree(tmp_path)
    assert kaggle.find_one("_HOOKREC_RAW_50m", root) == root / "k0-out" / "raw"
    assert kaggle.find_one("_HOOKREC_EDA", root, required=False) is None
    with pytest.raises(FileNotFoundError) as e:
        kaggle.find_one("_HOOKREC_EDA", root, hint="attach the K1 output")
    msg = str(e.value)
    assert "_HOOKREC_EDA" in msg and "k0-out" in msg and "attach the K1 output" in msg


def test_find_processed_by_size(tmp_path):
    root = _tree(tmp_path)
    assert kaggle.find_processed("50m", root) == root / "k2" / "processed" / "50m-aaa"
    assert kaggle.find_processed("500m", root) == root / "k2" / "processed" / "500m-ccc"
    with pytest.raises(FileNotFoundError, match="5b"):
        kaggle.find_processed("5b", root)
    (root / "k2" / "processed" / "50m-bbb" / "_DONE").write_text("")
    with pytest.raises(RuntimeError, match="50m-bbb"):
        kaggle.find_processed("50m", root)


def test_run_logged_streams_and_raises(tmp_path, capsys):
    kaggle.run_logged(
        [sys.executable, "-c", "print('one'); import sys; print('two', file=sys.stderr)"]
    )
    out = capsys.readouterr().out
    assert out.index("one") < out.index("two")  # stderr merged into the same stream
    with pytest.raises(RuntimeError, match="exited 3"):
        kaggle.run_logged([sys.executable, "-c", "import sys; print('bye'); sys.exit(3)"])
    assert "bye" in capsys.readouterr().out


def test_run_logged_env_and_cwd(tmp_path, capsys):
    code = "import os; print(os.getcwd()); print(os.environ['HOOKREC_X'], os.environ['MPLBACKEND'])"
    kaggle.run_logged([sys.executable, "-c", code], env={"HOOKREC_X": "42"}, cwd=tmp_path)
    out = capsys.readouterr().out
    assert os.path.realpath(tmp_path) in out and "42 Agg" in out


def test_run_logged_streams_before_exit(monkeypatch):
    stamps = []

    class Rec:
        def write(self, s):
            if s.strip():
                stamps.append((s.strip(), time.monotonic()))

        def flush(self):
            pass

    monkeypatch.setattr(sys, "stdout", Rec())
    code = "import time; print('early', flush=True); time.sleep(1.0); print('late')"
    kaggle.run_logged([sys.executable, "-c", code])
    t = dict(stamps)
    assert t["late"] - t["early"] > 0.5  # 'early' reached the log while the child still ran


def test_session_env_idempotent(monkeypatch):
    monkeypatch.delenv("HOOKREC_SESSION_START", raising=False)
    v1 = kaggle.session_env()
    assert os.environ["HOOKREC_SESSION_START"] == v1 and abs(int(v1) - time.time()) < 5
    assert kaggle.session_env() == v1
    monkeypatch.setenv("HOOKREC_SESSION_START", "123")
    assert kaggle.session_env() == "123"


def test_check_disk_and_env_report(tmp_path, capsys):
    (tmp_path / "f.bin").write_bytes(b"\0" * 2048)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "g.bin").write_bytes(b"\0" * 1024)
    gb = kaggle.check_disk(tmp_path, warn_gb=18.0)
    assert gb == pytest.approx(3072 / 1e9)
    assert "WARNING" not in capsys.readouterr().out
    kaggle.check_disk(tmp_path, warn_gb=1e-9)
    assert "WARNING" in capsys.readouterr().out
    rep = kaggle.env_report(tmp_path)
    assert {"python", "torch", "gpus", "ram_total_gb", "ram_available_gb", "disk_free_gb"} <= set(
        rep
    )
    assert rep["disk_free_gb"] > 0
    assert "python" in capsys.readouterr().out


def test_has_run(tmp_path):
    rec = {"run_name": "a0_sasrec-val", "extra": {"processed_dir": "50m-aaa"}}
    (tmp_path / "a0_sasrec-val-20261112-000000.json").write_text(json.dumps(rec))
    (tmp_path / "broken.json").write_text("{not json")
    assert kaggle.has_run(tmp_path, "a0_sasrec-val", "50m-aaa")
    assert not kaggle.has_run(tmp_path, "a0_sasrec-test", "50m-aaa")
    assert not kaggle.has_run(tmp_path, "a0_sasrec-val", "50m-zzz")
    assert not kaggle.has_run(tmp_path / "missing", "a0_sasrec-val", "50m-aaa")
