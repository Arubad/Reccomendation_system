import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from hookrec.demo import gpu
from hookrec.demo.bundle import export_demo_bundle
from hookrec.demo.engine import DemoEngine
from hookrec.models.base import build_model
from hookrec.train.trainer import train
from hookrec.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
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


@pytest.fixture(scope="session")
def tiny_ckpts(processed_synth, tmp_path_factory):
    out = {}
    for name in ("sasrec", "hookseq"):
        cfg = load_config(ROOT / "configs" / f"{name}.yaml", TINY)
        torch.manual_seed(0)
        d = tmp_path_factory.mktemp(f"ckpt_{name}")
        train(cfg, processed_synth, build_model(cfg, processed_synth), "test", d)
        out[name] = d
    return out


@pytest.fixture(scope="session")
def bundle_dir(processed_synth, tiny_ckpts, tmp_path_factory):
    return export_demo_bundle(
        processed_synth,
        tiny_ckpts["hookseq"],
        tiny_ckpts["sasrec"],
        tmp_path_factory.mktemp("bundle"),
        n_users=12,
        n_items=15,
    )


@pytest.fixture(scope="session")
def engine(bundle_dir):
    return DemoEngine(bundle_dir)


def test_bundle_size_limit(bundle_dir):
    total = sum(p.stat().st_size for p in bundle_dir.iterdir())
    assert 0 < total < 200 * 1024 * 1024
    assert {p.name for p in bundle_dir.iterdir()} == {
        "meta.json", "arrays.npz", "hookseq.pt", "sasrec.pt",
    }  # fmt: skip


@pytest.mark.parametrize("model", ["hookseq", "sasrec"])
def test_recommend_returns_k_items_not_in_pad(engine, model):
    session = engine.sample_user_session(0)
    recs = engine.recommend(session, model, k=10)
    assert len(recs) == 10
    assert all(r["item_id"] in set(engine.raw_ids.tolist()) for r in recs)
    assert all(set(r) == {"item_id", "artist_id", "popularity", "score"} for r in recs)
    scores = [r["score"] for r in recs]
    assert scores == sorted(scores, reverse=True) and np.isfinite(scores).all()
    assert len({r["item_id"] for r in recs}) == 10
    # k beyond the subset is clamped; nothing but real items comes back
    assert len(engine.recommend(session, model, k=10**6)) == engine.n_subset


def test_organic_toggle_changes_hookseq_scores(engine):
    item = int(engine.raw_ids[0])
    a = [(item, "listen_full", True), (int(engine.raw_ids[1]), "listen_full", True)]
    b = [a[0], (a[1][0], "listen_full", False)]
    k = engine.n_subset

    def by_item(s, model="hookseq"):
        return {r["item_id"]: r["score"] for r in engine.recommend(s, model, k)}

    sa, sb = by_item(a), by_item(b)
    assert sa.keys() == sb.keys()
    assert max(abs(sa[i] - sb[i]) for i in sa) > 1e-6
    # SASRec ignores the organic flag
    assert by_item(a, "sasrec") == by_item(b, "sasrec")


def test_unknown_items_and_bad_input(engine):
    recs = engine.recommend([(2**40, "listen_skip", False)], "hookseq", k=3)  # not in the subset
    assert len(recs) == 3
    with pytest.raises(ValueError, match="empty"):
        engine.recommend([], "hookseq")
    with pytest.raises(ValueError, match="unknown event"):
        engine.recommend([(int(engine.raw_ids[0]), "skipp", False)], "hookseq")
    with pytest.raises(ValueError, match="model"):
        engine.recommend([(int(engine.raw_ids[0]), "like", False)], "ials")


def test_new_track_audience_for_idless_item(engine):
    assert engine.idless_audio_items, "synthetic data should have ID-less items with audio"
    track = engine.idless_audio_items[0]
    aud = engine.new_track_audience(track, k=5)
    assert len(aud) == 5
    uids = {int(u) for u in engine._w["w_uids"]}
    assert all(a["uid"] in uids for a in aud)
    scores = [a["score"] for a in aud]
    assert scores == sorted(scores, reverse=True) and np.isfinite(scores).all()
    with_id = int(engine.raw_ids[engine.has_id][0])
    with pytest.raises(ValueError, match="ID-less"):
        engine.new_track_audience(with_id)
    with pytest.raises(ValueError, match="not in the demo bundle"):
        engine.new_track_audience(2**40)


def test_bundle_matches_the_full_models(processed_synth, tiny_ckpts, engine):
    """Compact item vectors and session states equal the full models' (strict, exact)."""
    data = processed_synth
    cfg = load_config(ROOT / "configs" / "hookseq.yaml", TINY)
    full = build_model(cfg, data)
    full.load_state_dict(
        torch.load(tiny_ckpts["hookseq"] / "model_final.pt", weights_only=True)["model"]
    )
    full.eval()
    sub = data.vocab.encode(engine.raw_ids)
    with torch.no_grad():
        want = full.all_item_repr()[torch.from_numpy(sub)]
    got = engine._vecs["hookseq"][1:]
    assert torch.allclose(got, want, atol=2e-3)  # fp16 storage
    # a sample window encoded from raw ids == the full model on the processed window
    s = engine.sample_user_session(0, n_last=None)
    batch = engine._session_batch(s)
    vocab_items = data.vocab.encode(np.array([i for i, _, _ in s]))
    full_batch = {k: v.clone() for k, v in batch.items()}
    full_batch["items"] = torch.from_numpy(vocab_items)[None]
    with torch.no_grad():
        assert torch.allclose(
            full.user_state(full_batch), engine.models["hookseq"].user_state(batch), atol=1e-5
        )


def test_export_script(processed_synth, tiny_ckpts, tmp_path):
    spec = importlib.util.spec_from_file_location(
        "export_demo", ROOT / "scripts" / "export_demo.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = mod.main(
        [
            "--processed", str(processed_synth.path),
            "--hookseq-ckpt", str(tiny_ckpts["hookseq"] / "model_final.pt"),
            "--sasrec-ckpt", str(tiny_ckpts["sasrec"]),
            "--out", str(tmp_path / "b"),
            "--n-users", "4", "--n-items", "10",
        ]
    )  # fmt: skip
    assert DemoEngine(out).n_sample_users == 4
    with pytest.raises(ValueError, match="holds a 'sasrec' model"):
        mod.main(
            [
                "--processed", str(processed_synth.path),
                "--hookseq-ckpt", str(tiny_ckpts["sasrec"]),
                "--sasrec-ckpt", str(tiny_ckpts["sasrec"]),
                "--out", str(tmp_path / "bad"),
            ]
        )  # fmt: skip


def test_build_app_constructs(engine):
    pytest.importorskip("gradio")
    from hookrec.demo.app import build_app, parse_session, recommend_both

    demo = build_app(engine)
    assert demo is not None
    rows = [[int(engine.raw_ids[0]), "listen_full", True], [None, None, None]]
    assert parse_session(rows) == [(int(engine.raw_ids[0]), "listen_full", True)]
    hs, sas, msg = recommend_both(engine, rows, 5)
    assert len(hs) == len(sas) == 5 and "1 events" in msg
    assert recommend_both(engine, [[1, "nope", True]], 5)[2].startswith("**Error")


def test_maybe_gpu_identity_without_spaces(monkeypatch):
    def f(x):
        return x

    monkeypatch.setattr(gpu, "_spaces", None)
    assert gpu._maybe_gpu(f) is f
    calls = []

    class FakeSpaces:
        @staticmethod
        def GPU(fn):
            calls.append(fn)
            return lambda *a: ("gpu", fn(*a))

    monkeypatch.setattr(gpu, "_spaces", FakeSpaces)
    assert gpu._maybe_gpu(f)(3) == ("gpu", 3) and calls == [f]


def test_hf_space_app_builds_with_local_bundle(bundle_dir, tmp_path, monkeypatch):
    pytest.importorskip("gradio")
    import gradio as gr

    kit = tmp_path / "space"
    kit.mkdir()
    (kit / "app.py").write_text((ROOT / "deploy" / "hf_space" / "app.py").read_text())
    (kit / "bundle").symlink_to(bundle_dir, target_is_directory=True)
    spec = importlib.util.spec_from_file_location("space_app", kit / "app.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["space_app"] = mod
    spec.loader.exec_module(mod)
    assert mod.resolve_bundle(kit) == kit / "bundle"
    launched = []
    monkeypatch.setattr(gr.Blocks, "launch", lambda self, *a, **k: launched.append(self))
    mod.build_demo(mod.resolve_bundle(kit)).launch()
    assert len(launched) == 1
    monkeypatch.delenv("HOOKSEQ_BUNDLE_REPO", raising=False)
    with pytest.raises(RuntimeError, match="HOOKSEQ_BUNDLE_REPO"):
        mod.resolve_bundle(tmp_path)
    fetched = []
    monkeypatch.setenv("HOOKSEQ_BUNDLE_REPO", "me/bundle")
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", lambda **kw: fetched.append(kw) or str(bundle_dir)
    )
    assert mod.resolve_bundle(tmp_path) == bundle_dir
    assert fetched == [{"repo_id": "me/bundle", "repo_type": "model"}]


def test_bundle_matches_the_full_sasrec(processed_synth, tiny_ckpts, engine):
    data = processed_synth
    cfg = load_config(ROOT / "configs" / "sasrec.yaml", TINY)
    full = build_model(cfg, data)
    full.load_state_dict(
        torch.load(tiny_ckpts["sasrec"] / "model_final.pt", weights_only=True)["model"]
    )
    full.eval()
    sub = data.vocab.encode(engine.raw_ids)
    with torch.no_grad():
        want = full.all_item_repr()[torch.from_numpy(sub)]
    assert torch.allclose(engine._vecs["sasrec"][1:], want, atol=2e-3)  # fp16 storage
    s = engine.sample_user_session(1, n_last=None)
    batch = engine._session_batch(s)
    full_batch = {k: v.clone() for k, v in batch.items()}
    full_batch["items"] = torch.from_numpy(data.vocab.encode(np.array([i for i, _, _ in s])))[None]
    with torch.no_grad():
        assert torch.allclose(
            full.user_state(full_batch), engine.models["sasrec"].user_state(batch), atol=1e-5
        )


def test_audience_scores_are_state_dot_vector(engine):
    track = engine.idless_audio_items[0]
    i = int(np.flatnonzero(engine.raw_ids == track)[0])
    hs = engine.models["hookseq"]
    assert int(hs.id_idx[i + 1]) == 0  # no ID row: the vector is audio + artist only
    with torch.no_grad():
        idx = torch.tensor([i + 1])
        audio = hs.audio_mlp(hs.audio[idx].float())
        want = audio + hs.artist_emb(hs.artist_idx[idx])
        assert torch.allclose(hs.item_repr(idx), want, atol=1e-6)
    states = engine._sample_states()
    aud = engine.new_track_audience(track, k=engine.n_sample_users)
    assert len(aud) == engine.n_sample_users
    by_uid = {int(u): j for j, u in enumerate(engine._w["w_uids"])}
    ref = states @ engine._vecs["hookseq"][i + 1].numpy()
    for a in aud:
        assert a["score"] == pytest.approx(float(ref[by_uid[a["uid"]]]), abs=1e-6)


def test_subset_rule(processed_synth, engine):
    data, raw = processed_synth, set(engine.raw_ids.tolist())
    assert set(engine._w["w_items"].tolist()) <= raw  # every window item is in the subset
    f, v = data.features, data.vocab
    idless = np.flatnonzero((v.id_idx == 0) & f.has_audio)
    idless = idless[idless > 0]
    assert len(idless) and {int(v.raw_ids[i]) for i in idless[:500]} <= raw
    assert set(engine.idless_audio_items) >= {int(v.raw_ids[i]) for i in idless[:500]}
    top = np.argsort(-f.pop_train[1:], kind="stable")[:15] + 1
    assert {int(v.raw_ids[i]) for i in top} <= raw


def test_parse_session_ranges_and_overflow():
    pytest.importorskip("gradio")
    from hookrec.demo.app import parse_session

    for bad in (float("inf"), 2**40, -1):
        with pytest.raises(ValueError, match="track id"):
            parse_session([[bad, "like", True]])


def test_oversized_bundle_is_removed(processed_synth, tiny_ckpts, tmp_path, monkeypatch):
    import hookrec.demo.bundle as b

    monkeypatch.setattr(b, "SIZE_LIMIT_BYTES", 1)
    with pytest.raises(RuntimeError, match="limit"):
        export_demo_bundle(
            processed_synth, tiny_ckpts["hookseq"], tiny_ckpts["sasrec"], tmp_path / "x", 4, 10
        )
    assert list((tmp_path / "x").iterdir()) == []


def test_demo_imports_stay_lean():
    """The Space installs hookrec without its training deps: the demo must not need them."""
    import subprocess

    code = (
        "import sys, hookrec.demo.engine, hookrec.demo.app, hookrec.models.hookseq, "
        "hookrec.models.sasrec\n"
        "bad = [m for m in ('implicit','sklearn','scipy','matplotlib','pyarrow','polars','yaml') "
        "if m in sys.modules]\nprint(bad)"
    )
    out = subprocess.run(
        [sys.executable, "-I", "-c", code], check=False, capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"


def test_ensure_hookrec_installs_without_deps(monkeypatch):
    spec = importlib.util.spec_from_file_location("space_app2", ROOT / "deploy/hf_space/app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    calls = []
    monkeypatch.setattr(mod.subprocess, "run", lambda cmd, check: calls.append(cmd))
    mod.ensure_hookrec("v9")  # hookrec is importable here: nothing to do
    assert calls == []
    monkeypatch.setattr(mod.importlib.util, "find_spec", lambda name: None)
    mod.ensure_hookrec("v9")
    assert "--no-deps" in calls[0] and calls[0][-1].endswith("Reccomendation_system.git@v9")


def test_zerogpu_placement_is_once_at_construction(bundle_dir, engine, monkeypatch):
    """With `spaces` active the engine places models once and the GPU calls never move them."""
    import hookrec.demo.engine as eng

    class FakeSpaces:
        @staticmethod
        def GPU(fn):
            return fn

    monkeypatch.setattr(gpu, "_spaces", FakeSpaces)
    monkeypatch.setattr(eng, "_ZEROGPU_DEVICE", "cpu")  # no CUDA here: stand in for the emulation
    moves = []
    real_to = torch.nn.Module.to
    monkeypatch.setattr(
        torch.nn.Module, "to", lambda self, *a, **k: moves.append(a) or real_to(self, *a, **k)
    )
    e = eng.DemoEngine(bundle_dir)
    placed = len(moves)
    assert placed == 2  # one move per model, at construction
    session = engine.sample_user_session(0)
    assert e.recommend(session, "hookseq", 5) == engine.recommend(session, "hookseq", 5)
    e.new_track_audience(e.idless_audio_items[0], 3)
    assert len(moves) == placed  # nothing moves inside the scoring calls
