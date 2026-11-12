import importlib.util
import json
from pathlib import Path

import numpy as np

from hookrec.eval.protocol import evaluate
from hookrec.models.base import fit_recommender
from hookrec.models.ials import IALS
from hookrec.models.popularity import MostPop

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
