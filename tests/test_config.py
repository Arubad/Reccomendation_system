import yaml

from hookrec.utils.config import config_hash, load_config


def test_extends_and_overrides(tmp_path):
    (tmp_path / "base.yaml").write_text(yaml.safe_dump({"a": 1, "m": {"d": 8, "l": 2}}))
    (tmp_path / "child.yaml").write_text(yaml.safe_dump({"extends": "base.yaml", "m": {"d": 16}}))
    cfg = load_config(tmp_path / "child.yaml", {"m.l": 4})
    assert cfg == {"a": 1, "m": {"d": 16, "l": 4}}


def test_config_hash_stable():
    a = {"x": 1, "y": {"p": 2, "q": 3}}
    b = {"y": {"q": 3, "p": 2}, "x": 1}
    assert config_hash(a) == config_hash(b)
    assert len(config_hash(a)) == 10
