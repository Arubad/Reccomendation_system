import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from hookrec.eval.protocol import evaluate
from hookrec.models.base import fit_recommender
from hookrec.models.ials import IALS
from hookrec.models.popularity import MostPop
from hookrec.utils.results import load_runs

ROOT = Path(__file__).resolve().parents[1]
CFG = {
    "factors": 16,
    "regularization": 0.1,
    "alpha": 10,
    "iterations": 5,
    "use_gpu": False,
    "seed": 0,
}


def test_ials_beats_mostpop_on_synth(processed_synth):
    data = processed_synth
    ials = fit_recommender("ials", data, CFG, "val")
    pop = fit_recommender("mostpop", data, {}, "val")
    n = data.vocab.n_items
    m_i = evaluate(ials, data.truth_val, n).metrics["ndcg@10"]
    m_p = evaluate(pop, data.truth_val, n).metrics["ndcg@10"]
    assert m_i > m_p, (m_i, m_p)


def test_ials_handles_unseen_users(processed_synth):
    data = processed_synth
    seqs = data.seqs_val_train
    n = data.vocab.n_items
    model = IALS(factors=8, iterations=2, use_gpu=False).fit(seqs, n)
    pop = MostPop().fit(seqs, n)
    known = int(seqs.uids[0])
    unknown = int(seqs.uids.max()) + 12345
    out = model.recommend(np.array([known, unknown]), 10)
    assert out.shape == (2, 10) and out.dtype == np.int64
    assert out.min() >= 0 and out.max() <= n
    np.testing.assert_array_equal(out[1], pop.top_items(10))
    assert (out[0] > 0).any()


def test_ials_registry_fits_listen_plus_aggregates(processed_synth):
    data = processed_synth
    agg = data.agg_val_train
    model = fit_recommender("ials", data, {**CFG, "iterations": 2}, "val")
    np.testing.assert_array_equal(model._uids, agg.uids)
    assert (model._user_items != 0).nnz == agg.ui.nnz
    np.testing.assert_allclose(model._user_items.data, 1.0 + 10 * agg.ui.data)
    pop = MostPop().fit_counts(agg.pop_lp)
    # users without Listen+ in val_train (cold) get MostPop over Listen+ counts
    cold = np.setdiff1d(np.asarray(data.seqs_val_train.uids), agg.uids)[:1]
    if len(cold) == 0:
        cold = np.array([int(agg.uids.max()) + 1])
    np.testing.assert_array_equal(model.recommend(cold, 10)[0], pop.top_items(10))


def _load_runner():
    path = ROOT / "scripts" / "run_baselines.py"
    spec = importlib.util.spec_from_file_location("run_baselines", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_baselines_smoke(processed_synth, tmp_path):
    mod = _load_runner()
    cfg = str(ROOT / "configs" / "baselines.yaml")
    base = ["--processed", str(processed_synth.path), "--config", cfg, "--out", str(tmp_path)]
    mod.main([*base, "--split", "val", "--quick"])
    mod.main([*base, "--split", "test", "--quick"])
    keys = {"split", "processed_dir", "wall_time_s", "n_eval_users", "metrics_cold", "n_cold_users"}
    for model in ("mostpop", "decaypop", "ials"):
        for split in ("val", "test"):
            files = list(tmp_path.glob(f"{model}-{split}-*.json"))
            assert len(files) == 1, (model, split)
            rec = json.loads(files[0].read_text())
            assert keys <= set(rec["extra"])
            assert rec["extra"]["split"] == split
            assert rec["extra"]["processed_dir"] == processed_synth.path.name
            assert "ndcg@10" in rec["metrics"]


def test_ials_positive_less_user_gets_fallback(processed_synth):
    from hookrec.data.label import LISTEN_SKIP
    from hookrec.data.sequences import PackedSequences

    s = processed_synth.seqs_val_train
    n = processed_synth.vocab.n_items
    # append a skip-only user to the history
    new_uid = int(s.uids.max()) + 1
    cat = np.concatenate
    seqs = PackedSequences(
        uids=cat([s.uids, [new_uid]]).astype(np.uint32),
        offsets=cat([s.offsets, [s.offsets[-1] + 2]]).astype(np.int64),
        items=cat([s.items, [1, 2]]).astype(np.int32),
        tokens=cat([s.tokens, [LISTEN_SKIP, LISTEN_SKIP]]).astype(np.uint8),
        organic=cat([s.organic, [0, 0]]).astype(np.uint8),
        ratio=cat([s.ratio, [0, 0]]).astype(np.uint8),
        ts=cat([s.ts, [0, 0]]).astype(np.uint32),
    )
    model = IALS(factors=8, iterations=2, use_gpu=False).fit(seqs, n)
    out = model.recommend(np.array([new_uid]), 10)
    np.testing.assert_array_equal(out[0], MostPop().fit(seqs, n).top_items(10))


def _seed_val(out, name, ndcg, pdir, alpha, quick=False):
    from hookrec.utils.results import save_run

    save_run(
        out, f"{name}-val", {"alpha": alpha}, {"ndcg@10": ndcg},
        {"processed_dir": pdir, "quick": quick},
    )  # fmt: skip


def test_best_val_selection(tmp_path):
    mod = _load_runner()
    _seed_val(tmp_path, "ials", 0.1, "d1", 1)
    _seed_val(tmp_path, "ials", 0.3, "d1", 40)
    _seed_val(tmp_path, "ials", 0.2, "d1", 10)
    _seed_val(tmp_path, "ials", 0.9, "other", 99)
    _seed_val(tmp_path, "ials", 0.8, "d1", 7, quick=True)
    assert mod._best_val_cfg(tmp_path, "ials", "d1") == {"alpha": 40}
    assert mod._best_val_cfg(tmp_path, "ials", "d1", quick=True) == {"alpha": 7}


def test_test_split_uses_winning_val_config(processed_synth, tmp_path):
    mod = _load_runner()
    cfg = str(ROOT / "configs" / "baselines.yaml")
    base = ["--processed", str(processed_synth.path), "--config", cfg, "--out", str(tmp_path)]
    mod.main([*base, "--split", "val", "--models", "decaypop"])
    runs = [r for r in load_runs(tmp_path) if r["run_name"] == "decaypop-val"]
    assert len(runs) == 4
    best = max(runs, key=lambda r: r["metrics"]["ndcg@10"])["config"]
    mod.main([*base, "--split", "test", "--models", "decaypop"])
    test_run = [r for r in load_runs(tmp_path) if r["run_name"] == "decaypop-test"]
    assert len(test_run) == 1 and test_run[0]["config"] == best


def test_test_split_without_val_run_fails(processed_synth, tmp_path):
    mod = _load_runner()
    cfg = str(ROOT / "configs" / "baselines.yaml")
    argv = ["--processed", str(processed_synth.path), "--config", cfg, "--out", str(tmp_path)]
    with pytest.raises(SystemExit) as e:
        mod.main([*argv, "--split", "test", "--models", "mostpop"])
    assert "--split val first" in str(e.value.code)
