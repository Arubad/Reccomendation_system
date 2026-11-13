"""F5: `model.id_min_count` rebuilds HookSeq's ID table at model-build time (a8_full_all_ids)."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from hookrec.data.vocab import id_index_from_counts, with_id_min_count
from hookrec.demo.bundle import export_demo_bundle
from hookrec.demo.engine import DemoEngine
from hookrec.models.base import build_model, effective_vocab
from hookrec.models.hookseq import HookSeq
from hookrec.train.trainer import train, training_hash
from hookrec.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
A8 = ROOT / "configs" / "ablations" / "a8_full_all_ids.yaml"
TINY = {
    "model.d_model": 16,
    "model.n_layers": 1,
    "model.max_len": 20,
    "model.dropout": 0.1,
    "train.batch_size": 4,
    "train.lr": 1e-3,
    "train.warmup_steps": 10,
    "train.max_steps": 15,
    "train.eval_every_steps": 10,
    "train.ckpt_every_steps": 5,
    "train.num_workers": 0,
    "train.log_every_steps": 10,
    "train.device": "cpu",
    "loss.n_uniform_negatives": 32,
    "loss.n_in_batch_negatives": 32,
}
# training_hash of every config at v0.3.1 (fe52a09): real Kaggle results were recorded under
# these, so K4's has_run and the val -> test match must keep finding them
HASHES_V031 = {
    "sasrec.yaml": "cc15ffa34a",
    "sasrec_bce.yaml": "7b948b6652",
    "hookseq.yaml": "7f6d08a169",
    "ablations/a0_sasrec.yaml": "6a62e9926b",
    "ablations/a1_audio.yaml": "8a3624f00d",
    "ablations/a2_audio_event_organic.yaml": "28f0d964c3",
    "ablations/a3_full_logq.yaml": "cbdfc0b730",
    "ablations/a4_full_dislike.yaml": "7f6d08a169",
    "ablations/a5_like_head.yaml": "89b8910ac2",
    "ablations/a6_time_gap.yaml": "b03f75d77f",
}


def _load(script):
    spec = importlib.util.spec_from_file_location(script, ROOT / "scripts" / f"{script}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("rel", sorted(HASHES_V031))
def test_existing_training_hashes_unchanged(rel):
    cfg = load_config(ROOT / "configs" / rel)
    assert "id_min_count" not in cfg["model"]
    assert training_hash(cfg) == HASHES_V031[rel]
    explicit_null = copy.deepcopy(cfg)
    explicit_null["model"]["id_min_count"] = None  # null = the key's absence
    assert training_hash(explicit_null) == HASHES_V031[rel]
    cfg["model"]["id_min_count"] = 1
    assert training_hash(cfg) != HASHES_V031[rel]


def test_min_count_reproduces_stored_vocab(processed_synth):
    d = processed_synth
    k = d.cfg["min_count"]
    # build_vocab counts `is_positive` (LISTEN_FULL or LIKE) over split.train; so do pop_pos
    # (aggregates, token in {LISTEN_FULL, LIKE}) and features.pop_train
    assert np.array_equal(d.agg("train").pop_pos, d.features.pop_train)
    v = with_id_min_count(d.vocab, d.agg("train").pop_pos, k)
    assert v.n_id == d.vocab.n_id and np.array_equal(v.id_idx, d.vocab.id_idx)
    assert v.id_idx.dtype == d.vocab.id_idx.dtype
    eff = effective_vocab(d, {"name": "hookseq", "id_min_count": k})
    assert np.array_equal(eff.id_idx, d.vocab.id_idx)


def test_id_min_count_one_gives_every_trained_item_an_id(processed_synth):
    d = processed_synth
    pos = d.agg("train").pop_pos
    v = effective_vocab(d, {"name": "hookseq", "id_min_count": 1})
    assert v.n_id == int((pos[1:] > 0).sum()) > d.vocab.n_id
    has = np.flatnonzero(v.id_idx)
    assert np.array_equal(has, np.flatnonzero(pos > 0))
    assert np.array_equal(v.id_idx[has], np.arange(1, v.n_id + 1))  # item-index order
    assert v.id_idx[0] == 0 and v.n_items == d.vocab.n_items
    assert np.array_equal(v.raw_ids, d.vocab.raw_ids)
    assert d.vocab.n_id < v.n_id  # the bundle's vocab is untouched
    # nested thresholds
    v3 = effective_vocab(d, {"name": "hookseq", "id_min_count": 3})
    assert set(np.flatnonzero(v3.id_idx)) <= set(np.flatnonzero(v.id_idx))


def test_effective_vocab_defaults_and_validation(processed_synth):
    d = processed_synth
    assert effective_vocab(d, {"name": "hookseq"}) is d.vocab
    assert effective_vocab(d, {"name": "hookseq", "id_min_count": None}) is d.vocab
    with pytest.raises(ValueError, match="HookSeq option"):
        effective_vocab(d, {"name": "sasrec", "id_min_count": 1})
    for bad, err in ((0, ValueError), (True, TypeError), ("5", TypeError), (1.0, TypeError)):
        with pytest.raises(err):
            effective_vocab(d, {"name": "hookseq", "id_min_count": bad})
    with pytest.raises(ValueError, match="rows"):
        with_id_min_count(d.vocab, np.zeros(3, dtype=np.int64), 1)
    idx, n = id_index_from_counts(np.array([9, 0, 2, 1, 5]), 2)  # index 0 (pad) never gets one
    assert idx.tolist() == [0, 0, 1, 0, 2] and n == 2


def test_build_model_uses_the_rebuilt_table(processed_synth):
    d = processed_synth
    cfg = load_config(A8, TINY)
    m = build_model(cfg, d)
    n_id = effective_vocab(d, cfg["model"]).n_id
    assert isinstance(m, HookSeq) and m.item_emb.num_embeddings == n_id + 1
    assert m.id_idx.max().item() == n_id
    default = build_model(load_config(ROOT / "configs" / "hookseq.yaml", TINY), d)
    assert default.item_emb.num_embeddings == d.vocab.n_id + 1
    sas = load_config(ROOT / "configs" / "sasrec.yaml", {**TINY, "model.id_min_count": 1})
    with pytest.raises(ValueError, match="HookSeq option"):
        build_model(sas, d)


@pytest.fixture(scope="module")
def a8_run(processed_synth, tmp_path_factory):
    """A tiny a8 test run through run_train (ID dropout and dislike hard negatives on)."""
    tmp = tmp_path_factory.mktemp("a8")
    over = [a for k, v in TINY.items() for a in ("--override", f"{k}={v}")]
    run_path = _load("run_train").main(
        ["--processed", str(processed_synth.path), "--config", str(A8), "--split", "test",
         "--allow-no-val", "--ckpt", str(tmp / "ck"), "--out", str(tmp / "r"), *over]
    )  # fmt: skip
    return tmp, Path(run_path)


def test_a8_trains_and_run_eval_reproduces_it(processed_synth, a8_run):
    d = processed_synth
    tmp, run_path = a8_run
    rec = json.loads(run_path.read_text())
    cfg = rec["config"]
    assert cfg["model"]["id_min_count"] == 1 and cfg["loss"]["dislike_hard_negatives"]
    assert cfg["model"]["id_dropout"] > 0
    state = torch.load(tmp / "ck" / "model_final.pt", weights_only=True)["model"]
    v = effective_vocab(d, cfg["model"])
    assert state["item_emb.weight"].shape[0] == v.n_id + 1
    # the tail rows (1 <= positives < data.min_count) exist only under a8, and they trained
    torch.manual_seed(cfg["train"]["seed"])
    init = build_model(cfg, d).state_dict()["item_emb.weight"]
    pos = d.agg("train").pop_pos
    tail = v.id_idx[(pos > 0) & (pos < d.cfg["min_count"])]
    assert len(tail) and not torch.equal(state["item_emb.weight"][tail], init[tail])

    path = _load("run_eval").main(
        ["--processed", str(d.path), "--run", str(run_path), "--ckpt", str(tmp / "ck"),
         "--out", str(tmp / "r")]
    )  # fmt: skip
    ev = json.loads(Path(path).read_text())
    for m, x in rec["metrics"].items():
        assert ev["metrics"][m] == pytest.approx(x, abs=1e-9), m  # same model, same scores
    # the item slices stay on the data vocab: idless_items is not empty though a8 has more IDs
    n = d.vocab.n_items
    n_idless = sum(
        bool(((t > n) | (d.vocab.id_idx[np.minimum(t, n)] == 0)).any())
        for t in d.truth_test.values()
    )
    assert n_idless and ev["extra"]["slices"]["idless_items"]["n_users"] == n_idless


def test_a8_demo_export_matches_the_full_model(processed_synth, a8_run, tmp_path):
    d = processed_synth
    tmp, run_path = a8_run
    cfg = json.loads(run_path.read_text())["config"]
    sas_cfg = load_config(ROOT / "configs" / "sasrec.yaml", {**TINY, "train.max_steps": 2})
    torch.manual_seed(0)
    train(sas_cfg, d, build_model(sas_cfg, d), "test", tmp_path / "sas")
    out = export_demo_bundle(d, tmp / "ck", tmp_path / "sas", tmp_path / "b", 12, 15)
    meta = json.loads((out / "meta.json").read_text())
    assert meta["hookseq_model"]["id_min_count"] == 1
    engine = DemoEngine(out)
    try:
        full = build_model(cfg, d)
        full.load_state_dict(
            torch.load(tmp / "ck" / "model_final.pt", weights_only=True)["model"]
        )
        full.eval()
        sub = d.vocab.encode(engine.raw_ids)
        v = effective_vocab(d, cfg["model"])
        assert np.array_equal(engine.has_id, v.id_idx[sub] > 0)  # the model's own ID table
        with torch.no_grad():
            want = full.all_item_repr()[torch.from_numpy(sub)]
        assert torch.allclose(engine._vecs["hookseq"][1:], want, atol=2e-3)  # fp16 storage
        s = engine.sample_user_session(0, n_last=None)
        batch = engine._session_batch(s)
        full_batch = {k: x.clone() for k, x in batch.items()}
        full_batch["items"] = torch.from_numpy(d.vocab.encode(np.array([i for i, _, _ in s])))[None]
        with torch.no_grad():
            h_full = full.user_state(full_batch)
            h_demo = engine.models["hookseq"].user_state(batch)
            assert torch.allclose(h_full, h_demo, atol=1e-5)
            scores_full = (h_full @ want.T).squeeze(0)
            scores_demo = (h_demo @ engine.models["hookseq"].all_item_repr()[1:-1].T).squeeze(0)
        assert torch.allclose(scores_full, scores_demo, atol=1e-4)
    finally:
        engine.close()
