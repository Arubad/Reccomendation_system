import numpy as np
import polars as pl

from hookrec.data import download as dl


def test_filter_embeddings_subset(synth_dir, tmp_path):
    emb_path = synth_dir / "embeddings.parquet"
    src_df = pl.read_parquet(emb_path)
    have = set(src_df["item_id"].to_list())
    ids = np.arange(50)
    out = tmp_path / "sub.parquet"
    n = dl.filter_embeddings(emb_path, ids, out)
    df = pl.read_parquet(out)
    expected = len(have & set(ids.tolist()))
    assert 0 < expected < 50
    assert n == expected == df.height
    assert set(df["item_id"].to_list()) <= set(ids.tolist())
    assert df.columns == ["item_id", "embed"]
    assert df.schema["embed"] == pl.Array(pl.Float16, 16)
    src = src_df.filter(pl.col("item_id") == df["item_id"][0])
    np.testing.assert_allclose(
        df["embed"][0].to_numpy(), np.array(src["normalized_embed"][0]), atol=1e-2
    )


def test_subset_item_ids(synth_dir):
    path = synth_dir / "flat/50m/multi_event.parquet"
    ids = dl.subset_item_ids(path)
    full = pl.read_parquet(path)["item_id"]
    assert np.array_equal(ids, np.sort(full.unique().to_numpy()))


def test_download_skips_existing(tmp_path, monkeypatch):
    calls = []

    def fake(repo_id, filename, repo_type, local_dir):
        calls.append(filename)
        p = tmp_path / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
        return str(p)

    monkeypatch.setattr(dl, "hf_hub_download", fake)
    pre = tmp_path / "embeddings.parquet"
    pre.write_bytes(b"old")
    paths = dl.download_yambda("50m", tmp_path)
    assert "embeddings.parquet" not in calls
    assert "flat/50m/multi_event.parquet" in calls
    assert paths["embeddings.parquet"] == pre and pre.read_bytes() == b"old"
    n = len(calls)
    dl.download_yambda("50m", tmp_path)
    assert len(calls) == n


def test_filter_embeddings_f16_and_empty_match(synth_dir, tmp_path):
    out = tmp_path / "none.parquet"
    assert dl.filter_embeddings(synth_dir / "embeddings.parquet", np.array([10**9]), out) == 0
    assert pl.read_parquet(out).height == 0
