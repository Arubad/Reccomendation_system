import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

import hookrec.data.processed as proc
from hookrec.data.processed import load_processed, preprocess
from hookrec.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = ROOT / "configs"


def _cfg(**kw):
    cfg = load_config(CFG_DIR / "data_50m.yaml")
    cfg["split"]["test_timestamp"] = None  # synth fixture is max-anchored
    cfg["min_count"] = 2
    cfg.update(kw)
    return cfg


@pytest.fixture(scope="module")
def built(processed_synth):
    return processed_synth.path, processed_synth


def test_preprocess_end_to_end_synth(built):
    d, pd_ = built
    n1 = pd_.vocab.n_items + 1
    assert pd_.features.audio.shape[0] == n1 == len(pd_.vocab.raw_ids)
    assert len(pd_.features.pop_train) == n1
    assert len(pd_.seqs_train) > 0 and len(pd_.seqs_val_train) > 0
    assert pd_.seqs_train.items.max() <= pd_.vocab.n_items
    assert len(pd_.truth_test) > 0 and len(pd_.truth_val) > 0
    m = pd_.meta
    assert m["n_cold_test_users"] >= 3
    assert m["share_unknown_test_items"] > 0
    for k in ("peak_rss_mb", "wall_seconds", "config_hash", "n_oov_test_items", "n_dropped_oov"):
        assert k in m
    assert m["n_items"] == pd_.vocab.n_items
    assert pd_.val_events().collect().columns == [
        "uid", "item_idx", "is_organic", "is_positive", "token", "ts"
    ]
    te = pd_.test_events().collect()
    assert te.height == m["n_events"]["test"] and te["item_idx"].min() >= 0
    assert (d / "config.yaml").exists()


def test_cold_users_split(built):
    _, pd_ = built
    assert set(pd_.truth_test_cold) >= {0, 1, 2}
    assert not set(pd_.truth_test_cold) & set(pd_.truth_test)
    assert not set(pd_.truth_test_cold) & set(pd_.seqs_train.uids.tolist())
    assert set(pd_.truth_test) <= set(pd_.seqs_train.uids.tolist())
    assert pd_.meta["n_cold_test_users"] == len(pd_.truth_test_cold)


def test_work_dir_removed(built):
    d, _ = built
    assert not (d / "_work").exists()
    assert (d / "_DONE").exists()


def test_cache_hit_skips_work(synth_dir, tmp_path, monkeypatch):
    cfg = _cfg()
    d1 = preprocess(synth_dir, cfg, tmp_path)
    calls = []
    monkeypatch.setattr(proc, "build_sequences", lambda *a, **k: calls.append(1))
    d2 = preprocess(synth_dir, cfg, tmp_path)
    assert d1 == d2 and not calls


def test_different_config_different_dir(synth_dir, tmp_path):
    a = preprocess(synth_dir, _cfg(), tmp_path)
    b = preprocess(synth_dir, _cfg(max_len_store=64), tmp_path)
    assert a != b and (a / "_DONE").exists() and (b / "_DONE").exists()


def test_killed_run_is_rebuilt(synth_dir, tmp_path):
    cfg = _cfg()
    d = preprocess(synth_dir, cfg, tmp_path)
    (d / "_DONE").unlink()
    (d / "junk.txt").write_text("partial")
    d2 = preprocess(synth_dir, cfg, tmp_path)
    assert d2 == d and (d / "_DONE").exists() and not (d / "junk.txt").exists()
    load_processed(d)


def test_load_requires_done(tmp_path):
    (tmp_path / "x").mkdir()
    with pytest.raises(FileNotFoundError, match="_DONE"):
        load_processed(tmp_path / "x")


def test_sharded_matches_unsharded(synth_dir, tmp_path, built):
    _, ref = built
    d = preprocess(synth_dir, _cfg(n_shards=2), tmp_path)
    got = load_processed(d)
    for name in ("seqs_train", "seqs_val_train", "seqs_train_lp", "seqs_val_train_lp"):
        a, b = getattr(ref, name), getattr(got, name)
        for f in ("uids", "offsets", "items", "tokens", "organic", "ratio", "ts"):
            assert np.array_equal(getattr(a, f), getattr(b, f)), (name, f)
    for period in ("train", "val_train"):
        a, b = ref.agg(period), got.agg(period)
        assert a.t_end == b.t_end and np.array_equal(a.uids, b.uids)
        assert (a.ui != b.ui).nnz == 0 and a.ui.shape == b.ui.shape
        for f in ("day_items", "day_index", "day_counts", "pop_lp", "pop_pos"):
            assert np.array_equal(getattr(a, f), getattr(b, f)), (period, f)
        for f in ("ui", "item_day"):  # the files themselves, too
            name = f"{f}_{period}.npz"
            with np.load(ref.path / name) as x, np.load(d / name) as y:
                assert x.files == y.files
                assert all(np.array_equal(x[k], y[k]) for k in x.files), (f, period)
        for f in ("pop_lp", "pop_pos"):
            name = f"{f}_{period}.npy"
            assert np.array_equal(np.load(ref.path / name), np.load(d / name)), name
    for name in ("truth_test", "truth_test_cold", "truth_val", "truth_val_cold"):
        x, y = getattr(ref, name), getattr(got, name)
        assert x.keys() == y.keys()
        assert all(np.array_equal(x[u], y[u]) for u in x)
    assert ref.vocab.n_items == got.vocab.n_items
    assert np.array_equal(ref.vocab.raw_ids, got.vocab.raw_ids)
    assert np.array_equal(ref.features.pop_train, got.features.pop_train)


def test_cli_override(synth_dir, tmp_path, capsys):
    path = ROOT / "scripts" / "run_preprocess.py"
    spec = importlib.util.spec_from_file_location("run_preprocess", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    d = mod.main(
        [
            "--config", str(CFG_DIR / "data_50m.yaml"),
            "--raw", str(synth_dir),
            "--out", str(tmp_path),
            "--override", "min_count=2",
            "--override", "split.test_timestamp=null",
        ]
    )
    assert (Path(d) / "_DONE").exists()
    assert str(d) in capsys.readouterr().out
    saved = yaml.safe_load((Path(d) / "config.yaml").read_text())
    assert saved["min_count"] == 2
    assert saved["split"]["test_timestamp"] is None


def test_val_warm_cold_disjoint(built):
    _, pd_ = built
    assert pd_.truth_val
    assert set(pd_.truth_val) <= set(pd_.seqs_val_train.uids.tolist())
    assert not set(pd_.truth_val) & set(pd_.truth_val_cold)


def test_missing_inputs_listed_in_one_error(tmp_path):
    cfg = _cfg()
    with pytest.raises(FileNotFoundError) as e:
        preprocess(tmp_path / "nope", cfg, tmp_path / "out")
    msg = str(e.value)
    for name in ("multi_event.parquet", "embeddings_50m.parquet", "artist_item_mapping.parquet"):
        assert name in msg


def test_val_test_events_sorted(built):
    _, pd_ = built
    for lf in (pd_.val_events(), pd_.test_events()):
        df = lf.collect()
        assert df.equals(df.sort("uid", "ts", maintain_order=True))
        assert df["item_idx"].min() > 0


# ---- F1: Listen+ protocol, full-period aggregates, Listen+ sequences, pipeline version ------


def _labeled_train_rows(synth_dir, cfg, t_end):
    from hookrec.data.clean import clean_events, scan_events
    from hookrec.data.label import label_events

    src = Path(synth_dir) / "flat" / "50m" / "multi_event.parquet"
    lf = label_events(clean_events(scan_events(src)), cfg["full_threshold"])
    return lf.filter(pl.col("timestamp") < t_end).select("uid", "item_id", "timestamp", "token")


def test_aggregates_equal_brute_force(synth_dir, built):
    from hookrec.data.label import LIKE, LISTEN_FULL

    _, pd_ = built
    v = pd_.vocab
    for period in ("train", "val_train"):
        t_end = pd_.meta[f"t_{period}_end"]
        rows = _labeled_train_rows(synth_dir, pd_.cfg, t_end).collect().rows()
        enc = {i: int(v.encode(np.array([i]))[0]) for _, i, _, _ in rows}
        lp = [(u, enc[i], t) for u, i, t, k in rows if k == LISTEN_FULL]
        pos = [enc[i] for _, i, _, k in rows if k in (LISTEN_FULL, LIKE)]
        a = pd_.agg(period)
        assert a.t_end == t_end
        assert a.uids.tolist() == sorted({u for u, _, _ in lp})
        dense = a.ui.toarray()
        got = {(int(a.uids[r]), int(c)): int(dense[r, c]) for r, c in zip(*np.nonzero(dense))}
        assert got == dict(Counter((u, i) for u, i, _ in lp))
        got_day = {
            (int(i), int(d)): int(n) for i, d, n in zip(a.day_items, a.day_index, a.day_counts)
        }
        assert got_day == dict(Counter((i, (t_end - t) // 86400) for _, i, t in lp))
        n1 = v.n_items + 1
        assert a.pop_lp.tolist() == np.bincount([i for _, i, _ in lp], minlength=n1).tolist()
        assert a.pop_pos.tolist() == np.bincount(pos, minlength=n1).tolist()
    # the full-period positive counts are the features' train popularity
    assert np.array_equal(pd_.agg_train.pop_pos, pd_.features.pop_train)


def test_listen_plus_sequences(built):
    from hookrec.data.label import LISTEN_FULL

    _, pd_ = built
    for period in ("train", "val_train"):
        s, s_all = pd_.seqs(period, "listen_plus"), pd_.seqs(period, "all")
        assert len(s) and (np.asarray(s.tokens) == LISTEN_FULL).all()
        assert s_all is getattr(pd_, f"seqs_{period}")
        assert s is getattr(pd_, f"seqs_{period}_lp")
        # users with a Listen+ sequence are exactly the aggregates' (warm) users
        assert np.array_equal(s.uids, pd_.agg(period).uids)
    with pytest.raises(ValueError, match="input_filter"):
        pd_.seqs("train", "likes")


def test_truth_is_listen_plus_only(built):
    from hookrec.data.label import LIKE, LISTEN_FULL

    _, pd_ = built
    assert pd_.meta["eval_target"] == "listen_plus"
    te = pd_.test_events().collect()
    lp = set(te.filter(pl.col("token") == LISTEN_FULL).select("uid", "item_idx").rows())
    like_only = set(te.filter(pl.col("token") == LIKE).select("uid", "item_idx").rows()) - lp
    assert like_only, "synth has no like-only test item"
    truth = {**pd_.truth_test, **pd_.truth_test_cold}
    for u, i in like_only:
        assert i not in truth.get(u, np.array([], dtype=np.int64))
    got = {(u, int(i)) for u, t in truth.items() for i in t}
    assert got == lp


def test_positive_target_includes_likes(synth_dir, tmp_path, built):
    _, ref = built
    got = load_processed(preprocess(synth_dir, _cfg(eval={"target": "positive"}), tmp_path))
    assert got.meta["eval_target"] == "positive"
    n_ref = sum(len(t) for t in (*ref.truth_test.values(), *ref.truth_test_cold.values()))
    n_got = sum(len(t) for t in (*got.truth_test.values(), *got.truth_test_cold.values()))
    assert n_got > n_ref
    # warm users stay defined by Listen+ history, whatever the target
    assert set(got.truth_test) <= set(got.agg_train.uids.tolist())


def test_pipeline_version_changes_dir_name(synth_dir, tmp_path, monkeypatch):
    a = preprocess(synth_dir, _cfg(), tmp_path)
    v = proc.PIPELINE_VERSION
    assert json.loads((a / "meta.json").read_text())["pipeline_version"] == v
    assert yaml.safe_load((a / "config.yaml").read_text())["pipeline_version"] == v
    monkeypatch.setattr(proc, "PIPELINE_VERSION", v + 1)
    b = preprocess(synth_dir, _cfg(), tmp_path)
    assert a != b and a.name.split("-")[0] == b.name.split("-")[0] == "50m"
    with pytest.raises(ValueError, match="pipeline version"):
        load_processed(a)  # an older bundle is never silently reused


HEAVY_UID, HEAVY_ITEM, FILLER_ITEM = 10_000, 1, 2


@pytest.fixture(scope="module")
def custom(synth, tmp_path_factory):
    """Synth plus (a) a user with > max_len_store events whose OLD Listen+ events make
    HEAVY_ITEM the most popular item, and (b) a user whose train history has no Listen+."""
    from hookrec.data.fixtures import BASE_TS, DAY, write_yambda_layout

    me = synth["multi_event"]
    t_test = int(me["timestamp"].max()) - DAY
    # (b): a regular user with a Listen+ test event loses every train Listen+ (-> skips)
    test_lp = me.filter(
        (pl.col("timestamp") >= t_test) & (pl.col("event_type") == "listen")
        & (pl.col("played_ratio_pct") >= 50) & (pl.col("uid") >= 7)
    )  # fmt: skip
    nolp = int(test_lp["uid"].min())
    me = me.with_columns(
        pl.when((pl.col("uid") == nolp) & (pl.col("timestamp") < t_test))
        .then(pl.col("played_ratio_pct").clip(upper_bound=10))
        .otherwise(pl.col("played_ratio_pct"))
        .alias("played_ratio_pct")
    )
    n_old, n_new = 300, 600  # 300 old Listen+ of HEAVY_ITEM, then 600 skips of FILLER_ITEM
    n = n_old + n_new
    heavy = pl.DataFrame(
        {
            "uid": pl.Series([HEAVY_UID] * n, dtype=pl.UInt32),
            "item_id": pl.Series([HEAVY_ITEM] * n_old + [FILLER_ITEM] * n_new, dtype=pl.UInt32),
            "timestamp": pl.Series(BASE_TS + DAY + 60 * np.arange(n), dtype=pl.UInt32),
            "is_organic": pl.Series([1] * n, dtype=pl.UInt8),
            "event_type": pl.Series(["listen"] * n, dtype=pl.String),
            "played_ratio_pct": pl.Series([90] * n_old + [10] * n_new, dtype=pl.UInt16),
            "track_length_seconds": pl.Series([200] * n, dtype=pl.UInt32),
        }
    )
    data = {**synth, "multi_event": pl.concat([me, heavy]).sort("uid", "timestamp")}
    root = write_yambda_layout(data, tmp_path_factory.mktemp("yambda_custom"))
    # a capped history (as at 500M) is what makes truncation matter
    d = preprocess(root, _cfg(max_len_store=512), tmp_path_factory.mktemp("processed_custom"))
    return load_processed(d), nolp


def test_mostpop_counts_listen_plus_older_than_max_len_store(custom):
    from hookrec.models.base import fit_recommender
    from hookrec.models.popularity import MostPop

    d, _ = custom
    heavy = int(d.vocab.encode(np.array([HEAVY_ITEM]))[0])
    s = d.seqs_train
    row = int(s.index_of(np.array([HEAVY_UID]))[0])
    window = np.asarray(s.items)[s.offsets[row] : s.offsets[row + 1]]
    assert len(window) == d.cfg["max_len_store"] and heavy not in window  # truncated away
    for split in ("val", "test"):
        assert fit_recommender("mostpop", d, {}, split).top_items(1)[0] == heavy, split
    # a fit on the truncated sequences (the old behaviour) would not rank it first
    assert MostPop().fit(d.seqs_train, d.vocab.n_items).top_items(1)[0] != heavy
    # the Listen+ sequences keep the old Listen+ events (filtered before truncation)
    lp = d.seqs_train_lp
    r = int(lp.index_of(np.array([HEAVY_UID]))[0])
    assert (np.asarray(lp.items)[lp.offsets[r] : lp.offsets[r + 1]] == heavy).sum() == 300


def test_user_without_train_listen_plus_is_cold(custom):
    d, nolp = custom
    assert nolp in d.truth_test_cold and nolp not in d.truth_test
    assert nolp in d.seqs_train.uids.tolist()  # it has (non-Listen+) history
    assert nolp not in d.agg_train.uids.tolist()
    assert set(d.truth_test) <= set(d.agg_train.uids.tolist())
