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
    assert "## Headline (50m, test split, Listen+ truth and warm users)" in first
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


def test_paired_section_prefers_sasrec_and_renders_both(tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    _write(res, _eval_rec("hookseq", 0.3, "2026-01-05"), 1)
    _write(res, _eval_rec("a0_sasrec", 0.2, "2026-01-05"), 2)
    _peruser(res, "hookseq", "50m", 0.1)
    _peruser(res, "a0_sasrec", "50m", 0.05)
    only_a0 = write_readme_tables(res, tmp_path / "o.md").read_text().split("## Paired")[1]
    assert "a0 control" in only_a0 and "+0.0500" in only_a0 and "vs sasrec (" not in only_a0

    _write(res, _eval_rec("sasrec", 0.25, "2026-01-05"), 3)
    _peruser(res, "sasrec", "50m", 0.0)
    paired = write_readme_tables(res, tmp_path / "o.md").read_text().split("## Paired")[1]
    i_sas = paired.index("| HookSeq (hookseq) vs sasrec (strong SASRec, Listen+ input)")
    i_a0 = paired.index("vs a0_sasrec (a0 control")
    assert i_sas < i_a0  # the paper-comparable baseline comes first
    assert "ndcg@10 (hookseq - sasrec) | +0.1000" in paired
    assert "ndcg@10 (hookseq - a0_sasrec) | +0.0500" in paired
    assert "recall@100 (hookseq - sasrec)" in paired


def test_quick_records_excluded_from_tables():
    runs = [
        _run("mostpop", 0.2, "2026-01-01"),
        _run("mostpop", 0.7, "2026-01-02", extra={"quick": True}),
        _run("a0_sasrec", 0.1, "2026-01-01"),
        _run("a1_audio", 0.6, "2026-01-03", extra={"quick": True}),
    ]
    t = results_table(runs, size="50m")
    assert "0.2000" in t and "0.7000" not in t
    a = ablation_table(runs, size="50m")
    assert "a0_sasrec" in a and "a1_audio" not in a


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


def test_paper_rows_under_recall_paper_and_old_records_render():
    from hookrec.eval.report import DEFAULT_METRICS, RECALL_FOOTNOTE, paper_rows

    assert "recall_paper@10" in DEFAULT_METRICS and "recall_paper@100" in DEFAULT_METRICS
    pr = paper_rows("50m")["SASRec"]
    assert pr["recall_paper@10"] == 0.0325 and "recall@10" not in pr
    old = _run("hookseq", 0.1, "2026-01-01")  # no recall_paper in its metrics
    t = results_table([old], size="50m")
    header = [c.strip() for c in t.splitlines()[0].strip("|").split("|")]
    row = next(ln for ln in t.splitlines() if ln.startswith("| SASRec (paper)"))
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[header.index("recall_paper@10")] == "0.0325"
    assert cells[header.index("recall@10")] == "-"
    hs = next(ln for ln in t.splitlines() if ln.startswith("| hookseq"))
    assert hs.split("|")[header.index("recall_paper@10") + 1].strip() == "-"  # old record: no crash
    assert "min(|T|, k)" in RECALL_FOOTNOTE


def test_readme_has_recall_footnote(tmp_path):
    res = tmp_path / "res"
    res.mkdir()
    (res / "a.json").write_text(json.dumps(_run("hookseq", 0.1, "2026-01-01")))
    md = write_readme_tables(res, tmp_path / "R.md").read_text()
    assert "recall_paper@k" in md and "hits@k / |T|" in md


# ---- F5: the HookSeq stand-in of the paired section, slice names -----------------------------


def _hs_eval(name, v, model="hookseq", peruser=None):
    r = _eval_rec(name, v, "2026-01-05")
    r["config"] = {"model": {"name": model}}
    if peruser:
        r["extra"]["peruser_file"] = peruser
    return r


def _write_cfg(d: Path, rec: dict, i: int):  # keeps rec["config"] (`_write` blanks it)
    rec = {**rec, "git_sha": "x", "config_hash": "x"}
    (d / f"{rec['run_name']}-{i}.json").write_text(json.dumps(rec))


def _paired(res, tmp_path):
    return write_readme_tables(res, tmp_path / "o.md").read_text().split("## Paired")[1]


def test_paired_falls_back_to_a4_and_labels_it(tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    # no run named hookseq: a4_full_dislike stands in, even though a3 scores higher
    _write_cfg(res, _hs_eval("a4_full_dislike", 0.30), 1)
    _write_cfg(res, _hs_eval("a3_full_logq", 0.35), 2)
    _write_cfg(res, _hs_eval("a0_sasrec", 0.20), 3)
    _write_cfg(res, _hs_eval("sasrec", 0.25, model="sasrec"), 4)
    _peruser(res, "a4_full_dislike", "50m", 0.1)
    _peruser(res, "a3_full_logq", "50m", 0.3)
    _peruser(res, "a0_sasrec", "50m", 0.05)
    _peruser(res, "sasrec", "50m", 0.0)
    p = _paired(res, tmp_path)
    assert "not yet run" not in p
    assert "_HookSeq = `a4_full_dislike` (no run named hookseq" in p
    assert "| HookSeq (a4_full_dislike) vs sasrec (strong SASRec, Listen+ input" in p
    assert "ndcg@10 (a4_full_dislike - sasrec) | +0.1000" in p
    assert "ndcg@10 (a4_full_dislike - a0_sasrec) | +0.0500" in p
    assert "a3_full_logq" not in p


def test_paired_exact_hookseq_run_wins(tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    _write_cfg(res, _hs_eval("hookseq", 0.1), 1)
    _write_cfg(res, _hs_eval("a4_full_dislike", 0.9), 2)
    _write_cfg(res, _hs_eval("sasrec", 0.1, model="sasrec"), 3)
    _peruser(res, "hookseq", "50m", 0.2)
    _peruser(res, "a4_full_dislike", "50m", 0.9)
    _peruser(res, "sasrec", "50m", 0.0)
    p = _paired(res, tmp_path)
    assert "_HookSeq = `hookseq` (the run named hookseq)" in p
    assert "ndcg@10 (hookseq - sasrec) | +0.2000" in p


def test_paired_best_hookseq_model_run_and_peruser_file(tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    # neither hookseq nor a4: the best hookseq-model eval. a0_sasrec (a paired baseline) and
    # non-hookseq models are never candidates, however well they score
    _write_cfg(res, _hs_eval("a1_audio", 0.20), 1)
    _write_cfg(res, _hs_eval("a7_full_no_content", 0.30, peruser="custom.npz"), 2)
    _write_cfg(res, _hs_eval("a0_sasrec", 0.90), 3)
    _write_cfg(res, _hs_eval("other", 0.95, model="sasrec"), 4)
    _write_cfg(res, _hs_eval("sasrec", 0.10, model="sasrec"), 5)
    base = np.random.default_rng(1).random(40)
    _npz(res / "custom.npz", np.arange(40), **{"ndcg@10": base + 0.3, "recall@100": base})
    _peruser(res, "sasrec", "50m", 0.0)
    _peruser(res, "a0_sasrec", "50m", 0.0)
    p = _paired(res, tmp_path)
    assert "_HookSeq = `a7_full_no_content` (no hookseq or a4_full_dislike run; best" in p
    assert "ndcg@10 (a7_full_no_content - sasrec) | +0.3000" in p
    assert "recall@100 (a7_full_no_content - sasrec) | +0.0000" in p


def test_paired_no_hookseq_family_run_is_not_yet_run(tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    _write_cfg(res, _hs_eval("a0_sasrec", 0.2), 1)
    _write_cfg(res, _hs_eval("sasrec", 0.1, model="sasrec"), 2)
    _peruser(res, "a0_sasrec", "50m", 0.0)
    _peruser(res, "sasrec", "50m", 0.0)
    assert "HookSeq vs SASRec: not yet run" in _paired(res, tmp_path)


def test_slice_table_renames_legacy_cold_items_and_footnote(tmp_path):
    r = _eval_rec("a4_full_dislike", 0.3, "2026-01-05")
    r["extra"]["slices"] = {
        "all": {"n_users": 3, "ndcg@10": 0.3},
        "cold_items": {"n_users": 1, "ndcg@10": 0.01},
    }
    t = slice_table(r)
    assert "| idless_items | 1 |" in t and "cold_items" not in t
    res = tmp_path / "results"
    res.mkdir()
    _write(res, r, 1)
    text = write_readme_tables(res, tmp_path / "o.md").read_text()
    assert "`new_items`" in text and "`idless_items`" in text


# ---- Task 20: paired rows for the follow-up controls a7 / a8 ---------------------------------


def test_paired_adds_a7_a8_rows_against_both_baselines(tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    # a4 stands in for HookSeq; a7 and a8 get their own paired rows vs sasrec and a0_sasrec
    _write_cfg(res, _hs_eval("a4_full_dislike", 0.30), 1)
    _write_cfg(res, _hs_eval("a7_full_no_content", 0.40), 2)
    _write_cfg(res, _hs_eval("a8_full_all_ids", 0.35), 3)
    _write_cfg(res, _hs_eval("a0_sasrec", 0.20), 4)
    _write_cfg(res, _hs_eval("sasrec", 0.25, model="sasrec"), 5)
    _peruser(res, "a4_full_dislike", "50m", 0.1)
    _peruser(res, "a7_full_no_content", "50m", 0.3)
    _peruser(res, "a8_full_all_ids", "50m", 0.2)
    _peruser(res, "a0_sasrec", "50m", 0.05)
    _peruser(res, "sasrec", "50m", 0.0)
    p = _paired(res, tmp_path)
    assert "_HookSeq = `a4_full_dislike`" in p
    assert "ndcg@10 (a4_full_dislike - sasrec) | +0.1000" in p
    assert "| a7_full_no_content vs sasrec (strong SASRec, Listen+ input" in p
    assert "ndcg@10 (a7_full_no_content - sasrec) | +0.3000" in p
    assert "ndcg@10 (a7_full_no_content - a0_sasrec) | +0.2500" in p
    assert "recall@100 (a8_full_all_ids - sasrec) | +0.2000" in p
    assert "ndcg@10 (a8_full_all_ids - a0_sasrec) | +0.1500" in p
    # HookSeq rows first, then a7, then a8
    assert p.index("(a4_full_dislike - sasrec)") < p.index("(a7_full_no_content - sasrec)")
    assert p.index("(a7_full_no_content - sasrec)") < p.index("(a8_full_all_ids - sasrec)")


def test_paired_extra_rows_skip_missing_npz_and_chosen_run(tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    # only a8 and a7 at this size: a8 (best) stands in for HookSeq and is not repeated as an
    # extra row; a7 has an eval record but no per-user npz, so it gets no rows at all
    _write_cfg(res, _hs_eval("a8_full_all_ids", 0.35), 1)
    _write_cfg(res, _hs_eval("a7_full_no_content", 0.30), 2)
    _write_cfg(res, _hs_eval("sasrec", 0.25, model="sasrec"), 3)
    _peruser(res, "a8_full_all_ids", "50m", 0.2)
    _peruser(res, "sasrec", "50m", 0.0)
    p = _paired(res, tmp_path)
    assert "_HookSeq = `a8_full_all_ids`" in p
    assert p.count("ndcg@10 (a8_full_all_ids - sasrec)") == 1
    assert "a7_full_no_content" not in p


def test_paired_caption_names_the_only_hookseq_run(tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    # a single HookSeq-model run at the size: the caption says so instead of "best of"
    _write_cfg(res, _hs_eval("a8_full_all_ids", 0.35), 1)
    _write_cfg(res, _hs_eval("sasrec", 0.25, model="sasrec"), 2)
    _peruser(res, "a8_full_all_ids", "50m", 0.2)
    _peruser(res, "sasrec", "50m", 0.0)
    p = _paired(res, tmp_path)
    assert "_HookSeq = `a8_full_all_ids` (the only HookSeq-model run at this size)._" in p
    assert "best test ndcg@10" not in p


def test_paired_extra_run_processed_dir_mismatch_row(tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    _write_cfg(res, _hs_eval("a4_full_dislike", 0.30), 1)
    a7 = _hs_eval("a7_full_no_content", 0.40)
    a7["extra"]["processed_dir"] = "other"
    _write_cfg(res, a7, 2)
    _write_cfg(res, _hs_eval("sasrec", 0.25, model="sasrec"), 3)
    for name, shift in (("a4_full_dislike", 0.1), ("a7_full_no_content", 0.3), ("sasrec", 0.0)):
        _peruser(res, name, "50m", shift)
    p = _paired(res, tmp_path)
    assert "| a7_full_no_content vs sasrec (strong SASRec, Listen+ input) | not available " \
        "(processed_dir mismatch) |" in p
    assert "ndcg@10 (a4_full_dislike - sasrec) | +0.1000" in p
