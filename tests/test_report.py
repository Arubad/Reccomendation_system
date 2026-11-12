import json
from pathlib import Path

import numpy as np

from hookrec.eval.report import (
    ablation_table,
    fairness_table,
    paired_diff_ci,
    results_table,
    slice_table,
    write_readme_tables,
)

M = ("ndcg@10", "ndcg@100", "recall@10", "recall@100", "coverage@10")


def _run(name, ndcg10, ts, size="50m", split="test", extra=None):
    return {
        "run_name": f"{name}-{split}",
        "timestamp": ts,
        "metrics": {m: ndcg10 for m in M},
        "extra": {"split": split, "dataset_size": size, **(extra or {})},
    }


def test_results_table_contains_paper_row():
    t = results_table([_run("hookseq", 0.1, "2026-01-01")], size="50m")
    assert "hookseq" in t and "0.1000" in t
    assert "SASRec (paper)" in t and "0.0748" in t
    assert "MostPop (paper)" in t


def test_results_table_latest_per_name_and_size():
    runs = [
        _run("hookseq", 0.1, "2026-01-01"),
        _run("hookseq", 0.3, "2026-01-02"),
        _run("hookseq", 0.9, "2026-01-03", size="500m"),
        _run("hookseq", 0.8, "2026-01-04", split="val"),
    ]
    t = results_table(runs, size="50m")
    assert "0.3000" in t
    assert not any(v in t for v in ("0.1000", "0.9000", "0.8000"))


def test_results_table_empty_does_not_crash():
    assert "not yet run" in results_table([], size="50m")
    assert "not yet run" in results_table([], size="zz")


def test_ablation_table_ordering():
    pairs = [("a3_full_logq", 0.4), ("a0_sasrec", 0.1), ("a10_x", 0.5), ("a1_audio", 0.2),
             ("hookseq", 0.9), ("sasrec", 0.9)]  # fmt: skip
    t = ablation_table([_run(n, v, "2026-01-01") for n, v in pairs])
    order = [t.index(k) for k in ("a0_sasrec", "a1_audio", "a3_full_logq", "a10_x")]
    assert order == sorted(order)
    assert "hookseq" not in t and "| sasrec" not in t
    assert "+0.1000" in t  # a1 - a0


def test_ablation_table_missing_a0_and_empty():
    assert "n/a" in ablation_table([_run("a1_audio", 0.2, "2026-01-01")])
    assert "not yet run" in ablation_table([])


def test_slice_fairness_tables_handle_missing():
    assert "not yet run" in slice_table(None)
    assert "not yet run" in fairness_table([])
    r = {
        "run_name": "x-test-eval",
        "extra": {
            "slices": {"all": {"n_users": 5, "ndcg@10": 0.5}},
            "fairness": {"artist_gini@10": 0.25},
            "ci": {"ndcg@10": [0.5, 0.4, 0.6]},
        },
    }
    assert "0.5000" in slice_table(r)
    f = fairness_table([r])
    assert "0.2500" in f and "[0.4000, 0.6000]" in f


def _npz(path, uids, **vals):
    np.savez(path, uids=np.asarray(uids), **vals)


def test_paired_diff_identical_is_zero(tmp_path):
    v = np.random.default_rng(0).random(50)
    _npz(tmp_path / "a.npz", np.arange(50), **{"ndcg@10": v})
    _npz(tmp_path / "b.npz", np.arange(50), **{"ndcg@10": v})
    assert paired_diff_ci(tmp_path / "a.npz", tmp_path / "b.npz", "ndcg@10") == (0.0, 0.0, 0.0)


def test_paired_diff_common_uids_only(tmp_path):
    _npz(tmp_path / "a.npz", [1, 2, 3, 9], **{"ndcg@10": np.array([1.0, 1.0, 1.0, 5.0])})
    _npz(tmp_path / "b.npz", [3, 2, 1, 7], **{"ndcg@10": np.array([0.0, 0.0, 0.0, 5.0])})
    mean, lo, hi = paired_diff_ci(tmp_path / "a.npz", tmp_path / "b.npz", "ndcg@10")
    assert mean == lo == hi == 1.0


def _write(d: Path, rec: dict, i: int):
    rec = {**rec, "git_sha": "x", "config_hash": "x", "config": {}}
    (d / f"{rec['run_name']}-{i}.json").write_text(json.dumps(rec))


def _eval_rec(name, v, ts, size="50m", pdir="p1"):
    r = _run(name, v, ts, size=size)
    r["run_name"] = f"{name}-test-eval"
    r["extra"].update(
        kind="eval",
        processed_dir=pdir,
        slices={"all": {"n_users": 3, "ndcg@10": v}},
        fairness={"artist_gini@10": 0.5},
        ci={"ndcg@10": [v, v - 0.1, v + 0.1]},
    )
    return r


def _peruser(res, name, size, shift):
    base = np.random.default_rng(1).random(40)
    both = {"ndcg@10": base + shift, "recall@100": base + shift}
    _npz(res / f"{name}-test-{size}-eval-peruser.npz", np.arange(40), **both)


def test_write_readme_tables(tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    for i, (n, v) in enumerate([("a0_sasrec", 0.1), ("a1_audio", 0.2), ("hookseq", 0.3)]):
        _write(res, _run(n, v, f"2026-01-0{i + 1}"), i)
    _write(res, _eval_rec("hookseq", 0.3, "2026-01-05"), 8)
    out = tmp_path / "reports" / "results.md"
    first = write_readme_tables(res, out).read_text()
    for h in ("Headline", "Ablations", "Slices", "Fairness", "Paired difference"):
        assert h in first
    assert "HookSeq vs SASRec: not yet run" in first
    assert "SASRec (paper)" in first

    _write(res, _eval_rec("a0_sasrec", 0.2, "2026-01-06"), 9)
    _peruser(res, "hookseq", "50m", 0.1)
    _peruser(res, "a0_sasrec", "50m", 0.0)
    text = write_readme_tables(res, out).read_text()
    paired = text.split("## Paired")[1]
    assert "+0.1000" in paired and "not yet run" not in paired


def test_paired_section_per_size_and_processed_dir(tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    # 50m: consistent; 500m: processed_dir mismatch between the two sides
    _write(res, _eval_rec("hookseq", 0.3, "2026-01-05"), 1)
    _write(res, _eval_rec("a0_sasrec", 0.2, "2026-01-05"), 2)
    _write(res, _eval_rec("hookseq", 0.4, "2026-01-05", size="500m", pdir="pA"), 3)
    _write(res, _eval_rec("a0_sasrec", 0.2, "2026-01-05", size="500m", pdir="pB"), 4)
    for size in ("50m", "500m"):
        _peruser(res, "hookseq", size, 0.1)
        _peruser(res, "a0_sasrec", size, 0.0)
    text = write_readme_tables(res, tmp_path / "o.md").read_text()
    assert "## Headline (500m" in text and "## Headline (50m" in text
    paired = text.split("## Paired")[1]
    p500, p50 = paired.split("### 500m")[1].split("### 50m")
    assert "+0.1000" in p50
    assert "processed_dir mismatch" in p500


def test_ablation_table_test_suffix_and_mixed_sizes():
    runs = [
        _run("a0_sasrec", 0.1, "2026-01-01"),
        _run("a1_audio", 0.2, "2026-01-01"),
        _run("a1_audio", 0.7, "2026-01-02", size="500m"),
        _run("a0_sasrec", 0.6, "2026-01-02", size="500m"),
    ]
    t = ablation_table(runs, size="50m")
    assert "| a0_sasrec |" in t and "-test" not in t
    assert "0.7000" not in t and "+0.1000" in t
    t5 = ablation_table(runs, size="500m")
    assert "0.7000" in t5 and "0.2000" not in t5


def test_write_readme_tables_survives_bad_files(tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    _write(res, _run("hookseq", 0.3, "2026-01-01"), 0)
    _write(res, _eval_rec("hookseq", 0.3, "2026-01-05"), 1)
    _write(res, _eval_rec("a0_sasrec", 0.2, "2026-01-05"), 2)
    (res / "broken-1.json").write_text("{not json")
    (res / "nokeys-1.json").write_text(json.dumps({"hello": 1}))
    bad = _eval_rec("weird", 0.5, "2026-01-06")
    bad["extra"]["slices"] = {"all": "oops"}
    bad["extra"]["fairness"] = {"artist_gini@10": "x"}
    bad["extra"]["ci"] = {"ndcg@10": [1]}
    _write(res, bad, 3)
    (res / "hookseq-test-50m-eval-peruser.npz").write_bytes(b"garbage")
    _peruser(res, "a0_sasrec", "50m", 0.0)
    text = write_readme_tables(res, tmp_path / "o.md").read_text()
    assert "Headline" in text
    assert "not available" in text.split("## Paired")[1]


def test_write_readme_tables_empty_dir(tmp_path):
    text = write_readme_tables(tmp_path / "nope", tmp_path / "o.md").read_text()
    assert "not yet run" in text


def test_safe_load_runs_treats_null_extra_as_empty(tmp_path):
    from hookrec.eval.report import _safe_load_runs

    (tmp_path / "a.json").write_text(
        json.dumps({"run_name": "a", "timestamp": "t", "metrics": {}, "extra": None})
    )
    (run,) = _safe_load_runs(tmp_path)
    assert run["extra"] == {}
