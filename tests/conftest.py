from pathlib import Path

import pytest

from hookrec.data.fixtures import make_synthetic_yambda, write_yambda_layout
from hookrec.data.processed import load_processed, preprocess
from hookrec.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def synth():
    return make_synthetic_yambda()


@pytest.fixture(scope="session")
def synth_dir(synth, tmp_path_factory):
    """Synthetic data written in the Yambda on-disk layout; returns the root dir."""
    root = tmp_path_factory.mktemp("yambda")
    write_yambda_layout(synth, root)
    return root


@pytest.fixture(scope="session")
def processed_synth(synth_dir, tmp_path_factory):
    """`ProcessedData` built once from the synthetic dataset (min_count=2, max-anchored split)."""
    cfg = load_config(ROOT / "configs" / "data_50m.yaml")
    cfg["split"]["test_timestamp"] = None
    cfg["min_count"] = 2
    out = tmp_path_factory.mktemp("processed_synth")
    return load_processed(preprocess(synth_dir, cfg, out))
