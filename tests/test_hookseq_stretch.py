import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from hookrec.data.label import LIKE, LISTEN_FULL
from hookrec.data.sequences import PackedSequences
from hookrec.models.base import build_model
from hookrec.models.hookseq import HookSeq
from hookrec.train.datasets import SeqInferDataset, SeqTrainDataset, gap_buckets
from hookrec.train.scoring import SeqRecommender
from hookrec.train.trainer import train
from hookrec.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
TINY = {
    "model.d_model": 16,
    "model.n_layers": 1,
    "model.max_len": 20,
    "model.dropout": 0.1,
    "train.batch_size": 4,
    "train.warmup_steps": 100,
    "train.max_steps": 20,
    "train.eval_every_steps": 10,
    "train.ckpt_every_steps": 5,
    "train.num_workers": 0,
    "train.log_every_steps": 10,
    "train.device": "cpu",
    "loss.n_uniform_negatives": 32,
    "loss.n_in_batch_negatives": 32,
}


def hook(data, **kw):
    torch.manual_seed(0)
    args = {"d_model": 16, "n_layers": 1, "n_heads": 2, "max_len": 20, "dropout": 0.1}
    args.update(kw)
    v = data.vocab
    return HookSeq(
        n_items=v.n_items, features=data.features, n_id=v.n_id, id_idx=v.id_idx, **args
    ).eval()


def test_gap_buckets_edges():
    # [0,30)->1  [30,120)->2  [120,600)->3  [600,1800)->4  [1800,21600)->5
    # [21600,86400)->6  [86400,inf)->7   (bucket 0 = pad / first event, never produced here)
    gaps = [0, 1, 29, 30, 119, 120, 599, 600, 1799, 1800, 21599, 21600, 86399, 86400, 604799]
    gaps += [604800, 10**9]
    want = [1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7, 7]
    assert gap_buckets(np.array(gaps)).tolist() == want
    # the session-gap edge moves
    assert gap_buckets(np.array([899, 900, 1799, 1800]), session_gap=900).tolist() == [4, 5, 5, 5]
    for bad in (600, 21600, 100):
        with pytest.raises(ValueError):
            gap_buckets(np.array([1]), session_gap=bad)


def _seqs_with_ts(users):
    """users: {uid: [ts, ...]} -> PackedSequences (items cycle 1..3, all LISTEN_FULL)."""
    uids = np.array(sorted(users), dtype=np.uint32)
    offsets = np.concatenate([[0], np.cumsum([len(users[int(u)]) for u in uids])]).astype(np.int64)
    ts = np.array([t for u in uids for t in users[int(u)]], dtype=np.uint32)
    n = len(ts)
    return PackedSequences(
        uids=uids,
        offsets=offsets,
        items=(np.arange(n) % 3 + 1).astype(np.int32),
        tokens=np.full(n, LISTEN_FULL, dtype=np.uint8),
        organic=np.ones(n, dtype=np.uint8),
        ratio=np.full(n, 2, dtype=np.uint8),
        ts=ts,
    )


def _ts_users():
    rng = np.random.default_rng(0)
    gaps = rng.choice([5, 60, 300, 1000, 5000, 50000, 200000, 900000], size=(3, 30))
    users = {u + 1: np.cumsum(gaps[u]).tolist() for u in range(3)}
    users[4] = [100, 105]  # shorter than the window
    users[5] = [777]  # first-ever event only
    return users


def test_gap_bucket_uses_full_sequence_and_pads_with_zero():
    seqs = _seqs_with_ts({1: [100, 110, 5000, 5010, 5040]})  # gaps: -, 10, 4890, 10, 30
    x = SeqInferDataset(seqs, 3)[0]  # window = last 3 events; the predecessor is outside it
    assert x["gap_bucket"].tolist() == [5, 1, 2]
    assert x["gap_bucket"].dtype == torch.long
    # first-ever event is bucket 0, like the left pad
    assert SeqInferDataset(seqs, 8)[0]["gap_bucket"].tolist() == [0, 0, 0, 0, 1, 5, 1, 2]


def test_gap_bucket_matches_between_infer_dataset_and_recommender_windows():
    seqs = _seqs_with_ts(_ts_users())
    ds = SeqInferDataset(seqs, 8, session_gap=1200)
    rec = SeqRecommender(None, seqs, 8, "cpu", None, session_gap=1200)
    w = rec._windows(np.arange(len(seqs)))
    for k in ("items", "gap_bucket"):
        assert torch.equal(w[k], torch.stack([ds[i][k] for i in range(len(seqs))])), k
    assert (w["gap_bucket"] > 0).any() and (w["gap_bucket"][w["items"] == 0] == 0).all()
    # user 5 has a single event: bucket 0 at its only position
    assert w["gap_bucket"][4].tolist() == [0] * 8
    # the window start of a long user carries a real (non-zero) gap
    assert w["gap_bucket"][0, 0] > 0


def test_flags_off_unchanged_outputs(processed_synth):
    # reference values from Task 15's HookSeq (commit c6b0da4), recorded before this change
    ref = hook(processed_synth)
    g = torch.Generator().manual_seed(1)
    n = processed_synth.vocab.n_items
    b = {
        "items": torch.randint(1, n + 1, (3, 20), generator=g),
        "tokens": torch.randint(1, 4, (3, 20), generator=g),
        "organic": torch.randint(0, 2, (3, 20), generator=g),
        "ratio": torch.randint(1, 4, (3, 20), generator=g),
    }
    b["gap_bucket"] = torch.randint(1, 8, (3, 20), generator=g)
    names = {k for k, _ in ref.named_parameters()}
    assert not any("gap_emb" in k or "like_W" in k for k in names)
    assert not hasattr(ref, "gap_emb") and not hasattr(ref, "like_W")
    with torch.no_grad():
        h = ref.encode(b)
        # a like head adds a parameter but must not move the encoder
        like = hook(processed_synth, like_head=True)
        assert torch.equal(like.encode(b), h)
        # the gap embedding is read only when enabled
        gap = hook(processed_synth, use_time_gap=True)
        assert not torch.equal(gap.encode(b), h)
        b2 = {**b, "gap_bucket": torch.randint(1, 8, (3, 20), generator=g)}
        assert torch.equal(ref.encode(b2), h)
        assert not torch.equal(gap.encode(b2), gap.encode(b))
    assert gap.gap_emb.weight.shape == (8, 16) and gap.gap_emb.weight[0].abs().sum() == 0
    assert like.like_W.weight.shape == (16, 16) and like.like_W.bias is None
    hid, tv = torch.randn(2, 5, 16), torch.randn(2, 5, 16)
    want = (hid * like.like_W(tv)).sum(-1)
    assert torch.equal(like.like_logit(hid, tv), want) and want.shape == (2, 5)


def _cfg(like_head, like_weight):
    cfg = load_config(ROOT / "configs" / "ablations" / "a4_full_dislike.yaml", TINY)
    cfg["model"]["like_head"] = like_head
    cfg["loss"]["like_weight"] = like_weight
    return cfg


def _train(data, tmp_path, tag, like_head, like_weight):
    cfg = _cfg(like_head, like_weight)
    torch.manual_seed(0)
    model = build_model(cfg, data)
    w0 = model.like_W.weight.clone() if like_head else None
    res = train(cfg, data, model, "test", tmp_path / tag)
    return res, model, w0


def test_like_head_loss_added_only_when_enabled(processed_synth, tmp_path):
    off, _, _ = _train(processed_synth, tmp_path, "off", False, 0.5)  # weight ignored: no head
    zero, m_zero, w0 = _train(processed_synth, tmp_path, "zero", True, 0.0)
    on, m_on, w1 = _train(processed_synth, tmp_path, "on", True, 0.5)
    # head present but weight 0: identical to no head, and the head is never trained
    assert zero.final_loss == off.final_loss
    assert torch.equal(m_zero.like_W.weight, w0)
    # weight > 0 adds the BCE term and trains the head
    assert on.final_loss > off.final_loss
    assert not torch.equal(m_on.like_W.weight, w1)
    assert np.isfinite(on.final_loss)


def test_train_dataset_emits_like_targets_for_bce():
    seqs = _seqs_with_ts({1: [10, 20, 30, 40]})
    seqs.tokens[:] = [LISTEN_FULL, LIKE, LISTEN_FULL, LIKE]
    x = SeqTrainDataset(seqs, 3, seed=0)[0]
    assert x["target_is_like"].tolist() == [True, False, True]
    assert x["gap_bucket"].tolist() == [0, 1, 1]


def test_a5_a6_configs_form_a_cumulative_ladder():
    a4 = load_config(ROOT / "configs" / "ablations" / "a4_full_dislike.yaml")
    a5 = load_config(ROOT / "configs" / "ablations" / "a5_like_head.yaml")
    a6 = load_config(ROOT / "configs" / "ablations" / "a6_time_gap.yaml")
    base = load_config(ROOT / "configs" / "sasrec.yaml")
    assert base["loss"]["like_weight"] == 0.0
    assert not a4["model"].get("like_head") and not a4["model"].get("use_time_gap")
    assert a5["model"]["like_head"] and not a5["model"].get("use_time_gap")
    assert a5["loss"]["like_weight"] == 0.2 and a5["loss"]["logq"]
    assert a6["model"]["like_head"] and a6["model"]["use_time_gap"]
    assert a6["loss"]["like_weight"] == 0.2 and a6["loss"]["dislike_hard_negatives"]


def test_run_train_smoke_on_a6(processed_synth, tmp_path):
    spec = importlib.util.spec_from_file_location("run_train", ROOT / "scripts" / "run_train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    over = [a for k, v in TINY.items() for a in ("--override", f"{k}={v}")]
    cfg = str(ROOT / "configs" / "ablations" / "a6_time_gap.yaml")
    base = ["--processed", str(processed_synth.path), "--config", cfg, "--split", "val"]
    path = mod.main([*base, "--ckpt", str(tmp_path / "c"), "--out", str(tmp_path / "r"), *over])
    rec = json.loads(Path(path).read_text())
    assert rec["config"]["model"]["use_time_gap"] and rec["config"]["model"]["like_head"]
    assert "ndcg@10" in rec["metrics"]
