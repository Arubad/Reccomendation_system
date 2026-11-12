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


def test_write_readme_tables(tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    for i, (n, v) in enumerate([("a0_sasrec", 0.1), ("a1_audio", 0.2), ("hookseq", 0.3)]):
        _write(res, _run(n, v, f"2026-01-0{i + 1}"), i)
    ev = _run("hookseq", 0.3, "2026-01-05")
    ev["run_name"] = "hookseq-test-eval"
    ev["extra"].update(
        kind="eval",
        slices={"all": {"n_users": 3, "ndcg@10": 0.3}},
        fairness={"artist_gini@10": 0.5},
        ci={"ndcg@10": [0.3, 0.2, 0.4]},
    )
    _write(res, ev, 9)
    out = tmp_path / "reports" / "results.md"
    first = write_readme_tables(res, out).read_text()
    for h in ("Headline", "Ablations", "Slices", "Fairness", "Paired difference"):
        assert h in first
    assert "HookSeq vs SASRec: not yet run" in first
    assert "SASRec (paper)" in first

    base = np.random.default_rng(1).random(40)
    both = {"ndcg@10": base + 0.1, "recall@100": base + 0.1}
    _npz(res / "hookseq-test-eval-peruser.npz", np.arange(40), **both)
    _npz(
        res / "a0_sasrec-test-eval-peruser.npz",
        np.arange(40),
        **{"ndcg@10": base, "recall@100": base},
    )
    text = write_readme_tables(res, out).read_text()
    paired = text.split("## Paired")[1]
    assert "+0.1000" in paired and "not yet run" not in paired


def test_write_readme_tables_empty_dir(tmp_path):
    text = write_readme_tables(tmp_path / "nope", tmp_path / "o.md").read_text()
    assert "not yet run" in text
