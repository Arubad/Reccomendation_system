import dataclasses
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from hookrec.eval.protocol import evaluate
from hookrec.models.base import build_model, history_for, input_filter_of
from hookrec.models.hookseq import HookSeq
from hookrec.models.popularity import mostpop_fallback
from hookrec.models.sasrec import SASRec
from hookrec.train.scoring import SeqRecommender
from hookrec.train.trainer import train
from hookrec.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
ABLATIONS = sorted((ROOT / "configs" / "ablations").glob("*.yaml"))
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
FLAGS = ("use_audio", "use_artist", "use_event", "use_organic", "use_ratio")


def hook(data, **kw):
    torch.manual_seed(0)
    args = {"d_model": 16, "n_layers": 1, "n_heads": 2, "max_len": 20, "dropout": 0.1}
    args.update(kw)
    v = data.vocab
    return HookSeq(
        n_items=v.n_items, features=data.features, n_id=v.n_id, id_idx=v.id_idx, **args
    ).eval()


def idless(data, with_audio):
    f, v = data.features, data.vocab
    mask = (v.id_idx == 0) & (np.arange(len(v.id_idx)) > 0) & (f.has_audio == with_audio)
    assert mask.any(), "synth lacks this kind of item"
    return int(np.flatnonzero(mask)[0])


def test_idless_item_has_finite_repr_from_audio(processed_synth):
    m = hook(processed_synth)
    i = idless(processed_synth, True)
    r = m.item_repr(torch.tensor([i]))[0]
    assert torch.isfinite(r).all() and r.abs().sum() > 0
    id_only = hook(processed_synth, use_audio=False, use_artist=False)
    assert id_only.item_repr(torch.tensor([i])).abs().sum() == 0  # no ID, no audio => nothing


def test_no_id_no_audio_uses_fallback_and_artist_finite(processed_synth):
    m = hook(processed_synth)
    i = idless(processed_synth, False)
    r = m.item_repr(torch.tensor([i]))[0]
    expected = m.missing_audio_vec + m.artist_emb(m.artist_idx[i])
    assert torch.isfinite(r).all()
    assert torch.allclose(r, expected, atol=1e-6)
    assert (m.item_repr(torch.tensor([0]))[0] == 0).all()  # pad stays zero


def test_id_dropout_only_in_train(processed_synth):
    m = hook(processed_synth, use_audio=False, use_artist=False, id_dropout=0.5)
    ids = torch.nonzero(m.id_idx)[:, 0].repeat(4)
    with torch.no_grad():
        assert torch.equal(m.item_repr(ids), m.item_repr(ids))
        assert (m.item_repr(ids) != 0).any(-1).all()
        m.train()
        a, b = m.item_repr(ids), m.item_repr(ids)
        assert not torch.equal(a, b)
        dropped = (a == 0).all(-1)
        assert dropped.any() and not dropped.all()


def test_all_flags_off_param_set_equals_sasrec(processed_synth):
    v = processed_synth.vocab
    m = hook(processed_synth, **dict.fromkeys(FLAGS, False))
    s = SASRec(n_items=v.n_id, d_model=16, n_layers=1, n_heads=2, max_len=20, dropout=0.1)

    def shapes(x):
        return {k: tuple(p.shape) for k, p in x.state_dict().items()}

    assert shapes(m) == shapes(s)
    # same seed => identical init values, not just identical shapes
    torch.manual_seed(0)
    s = SASRec(n_items=v.n_id, d_model=16, n_layers=1, n_heads=2, max_len=20, dropout=0.1)
    ms, ss = m.state_dict(), s.state_dict()
    assert all(torch.equal(ms[k], ss[k]) for k in ss)
    assert m.all_item_repr().shape == (v.n_items + 1, 16)


def test_buffers_are_not_checkpointed_and_flag_params_exist(processed_synth):
    names = set(hook(processed_synth).state_dict())
    assert not {"id_idx", "audio", "has_audio", "artist_idx"} & names
    for n in ("audio_mlp.0.weight", "missing_audio_vec", "artist_emb.weight", "token_emb.weight",
              "organic_emb.weight", "ratio_emb.weight"):  # fmt: skip
        assert n in names


def _batch(data, tokens, organic, ratio):
    items = torch.randint(1, data.vocab.n_items + 1, (2, 8))
    z = torch.zeros_like(items)
    return {"items": items, "tokens": z + tokens, "organic": z + organic, "ratio": z + ratio}


def test_side_signals_change_the_encoding(processed_synth):
    m = hook(processed_synth)
    b0 = _batch(processed_synth, 1, 0, 1)
    b1 = dict(b0, tokens=b0["tokens"] + 1, organic=b0["organic"] + 1, ratio=b0["ratio"] + 2)
    assert not torch.allclose(m.encode(b0), m.encode(b1))
    pad = {k: torch.zeros(1, 8, dtype=torch.long) for k in b0}
    assert torch.isfinite(m.encode(pad)).all()


def test_autocast_forward_is_finite(processed_synth):
    m = hook(processed_synth)
    b = _batch(processed_synth, 1, 0, 1)
    for dtype in (torch.float16, torch.bfloat16):  # CPU fp16 autocast may be unsupported
        try:
            with torch.no_grad(), torch.autocast("cpu", dtype=dtype):
                h, r = m.encode(b), m.item_repr(b["items"])
            break
        except RuntimeError:
            continue
    else:
        pytest.fail("no autocast dtype works on CPU")
    assert torch.isfinite(h.float()).all() and torch.isfinite(r.float()).all()


@pytest.mark.parametrize("path", ABLATIONS, ids=lambda p: p.stem)
def test_every_ablation_config_builds(path, processed_synth):
    cfg = load_config(path, TINY)
    assert cfg["model"]["name"] == "hookseq"
    m = build_model(cfg, processed_synth)
    assert isinstance(m, HookSeq)
    assert all(getattr(m, f) == cfg["model"][f] for f in FLAGS)


def test_ablation_ladder():
    got = {}
    for p in ABLATIONS:
        c = load_config(p)
        flags = tuple(c["model"][f] for f in FLAGS)
        got[p.stem] = (flags, c["loss"]["logq"], c["loss"]["dislike_hard_negatives"])
    off, on = (False,) * 5, (True,) * 5
    assert len(got) == 7
    assert got["a0_sasrec"] == (off, False, False)
    assert got["a1_audio"] == ((True, True, False, False, False), False, False)
    assert got["a2_audio_event_organic"] == (on, False, False)
    assert got["a3_full_logq"] == (on, True, False)
    full = load_config(ROOT / "configs" / "hookseq.yaml")
    assert full["model"]["id_dropout"] == 0.2
    assert all(full["model"][f] for f in FLAGS)
    assert full["loss"]["logq"] and full["loss"]["dislike_hard_negatives"]
    assert got["a4_full_dislike"] == (on, True, True)


def test_run_train_smoke_on_a2(processed_synth, tmp_path):
    spec = importlib.util.spec_from_file_location("run_train", ROOT / "scripts" / "run_train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    over = [a for k, v in TINY.items() for a in ("--override", f"{k}={v}")]
    cfg = str(ROOT / "configs" / "ablations" / "a2_audio_event_organic.yaml")
    base = ["--processed", str(processed_synth.path), "--config", cfg, "--split", "val"]
    path = mod.main([*base, "--ckpt", str(tmp_path / "c"), "--out", str(tmp_path / "r"), *over])
    rec = json.loads(Path(path).read_text())
    assert rec["config"]["model"]["name"] == "hookseq" and "ndcg@10" in rec["metrics"]


def _cold_ndcg(model, data, cfg):
    # synth's cold items (no ID row) only appear in the test window's ground truth
    seqs, _ = history_for(data, "test", input_filter_of(cfg))  # the set the model trained on
    n = data.vocab.n_items
    rec = SeqRecommender(model, seqs, 20, torch.device("cpu"), mostpop_fallback(data, "test"))
    cold = data.vocab.id_idx == 0
    truth = {u: t[cold[t]] for u, t in data.truth_test.items()}
    truth = {u: t for u, t in truth.items() if len(t)}
    assert truth, "synth has no cold ground-truth items"
    return evaluate(rec, truth, n, ks=(10,)).metrics["ndcg@10"]


def test_audio_path_beats_untrained_idonly_and_sasrec_on_cold_items(processed_synth, tmp_path):
    """Cold items (no ID row) are reachable only via audio/artist. Isolate the audio path
    (artist off): trained audio-only must clearly beat the same net untrained, an ID-only
    HookSeq, SASRec and the same net trained on audio rows permuted across items (so the
    signal is the audio content, not merely having an audio row); the full model must beat
    SASRec."""
    data = processed_synth
    # shuffled-audio control: permute the audio rows among the items that have audio
    f = data.features
    have = np.flatnonzero(f.has_audio)
    audio = np.array(f.audio, copy=True)
    audio[have] = audio[np.random.default_rng(0).permutation(have)]
    shuffled = dataclasses.replace(data, features=dataclasses.replace(f, audio=audio))
    over = {**TINY, "train.batch_size": 16, "train.lr": 5e-3, "train.warmup_steps": 20,
            "train.max_steps": 300, "train.eval_every_steps": 1000, "train.ckpt_every_steps": 1000,
            "train.patience": 100, "loss.logq": False, "loss.dislike_hard_negatives": False}  # fmt: skip
    audio_only = {"model.use_artist": False}
    runs = {  # name: (base config, overrides, trained?, data)
        "sasrec": ("sasrec", {}, True, data),
        "full": ("hookseq", {}, True, data),
        "audio_only": ("hookseq", audio_only, True, data),
        "audio_only_untrained": ("hookseq", audio_only, False, data),
        "audio_only_shuffled": ("hookseq", audio_only, True, shuffled),
        "id_only": ("hookseq", {"model.use_artist": False, "model.use_audio": False}, True, data),
    }
    scores = {}
    for name, (base, extra, trained, d) in runs.items():
        cfg = load_config(ROOT / "configs" / f"{base}.yaml", {**over, **extra})
        torch.manual_seed(0)
        model = build_model(cfg, d)
        if trained:
            train(cfg, d, model, "test", tmp_path / name)
        scores[name] = _cold_ndcg(model.eval(), d, cfg)
    margin = 0.03  # audio-only scored 0.055-0.064 over seeds 0-2; the others score 0.0
    for other in ("audio_only_untrained", "audio_only_shuffled", "id_only", "sasrec"):
        assert scores["audio_only"] > scores[other] + margin, scores
    assert scores["full"] > scores["sasrec"] + margin, scores
