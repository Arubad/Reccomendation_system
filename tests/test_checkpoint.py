import torch

from hookrec.train.checkpoint import capture_rng_states, load_latest, save_checkpoint


def _save(path, step):
    model = torch.nn.Linear(3, 2)
    opt = torch.optim.AdamW(model.parameters())
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    save_checkpoint(path, model, opt, sched, scaler, step, 0, 0.5, capture_rng_states())


def test_atomic_checkpoint_survives_partial_write(tmp_path):
    _save(tmp_path / "ckpt-00000005.pt", 5)
    good = (tmp_path / "ckpt-00000005.pt").read_bytes()
    assert not list(tmp_path.glob("*.tmp"))
    (tmp_path / "ckpt-00000012.tmp").write_bytes(b"garbage from a killed save")
    (tmp_path / "ckpt-00000010.pt").write_bytes(good[: len(good) // 2])  # truncated, newer
    torch.save({"model": {}}, tmp_path / "ckpt-00000009.pt")  # loads, but lacks keys
    payload = load_latest(tmp_path)
    assert payload is not None and payload["step"] == 5
    assert payload["path"] == str(tmp_path / "ckpt-00000005.pt")


def test_load_latest_searches_all_dirs_highest_step_first(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _save(a / "ckpt-00000003.pt", 3)
    _save(b / "ckpt-00000007.pt", 7)
    assert load_latest([a, b])["step"] == 7
    assert load_latest([a, tmp_path / "missing"])["step"] == 3
    assert load_latest([tmp_path / "missing"]) is None
