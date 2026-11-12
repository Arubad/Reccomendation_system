import pytest

from hookrec.data.fixtures import make_synthetic_yambda, write_yambda_layout


@pytest.fixture(scope="session")
def synth():
    return make_synthetic_yambda()


@pytest.fixture(scope="session")
def synth_dir(synth, tmp_path_factory):
    """Synthetic data written in the Yambda on-disk layout; returns the root dir."""
    root = tmp_path_factory.mktemp("yambda")
    write_yambda_layout(synth, root)
    return root
