import pytest

from preflight.config import Settings, load_settings


def test_defaults(tmp_path):
    s = Settings(data_dir=tmp_path)
    assert s.theta_high > s.theta_low
    assert s.cache_rule_for("anthropic").read_mult == 0.10
    assert s.cache_rule_for("unknown-provider").read_mult == 0.50


def test_yaml_load(tmp_path):
    cfg = tmp_path / "preflight.yaml"
    cfg.write_text(f"data_dir: {tmp_path / 'data'}\ntheta_high: 0.99\nepsilon: 0\n")
    s = load_settings(cfg)
    assert s.theta_high == 0.99
    assert (tmp_path / "data").is_dir()


def test_invalid_fixed_action(tmp_path):
    with pytest.raises(ValueError):
        Settings(data_dir=tmp_path, fixed_action="A9")


def test_composite_fixed_action_allowed(tmp_path):
    s = Settings(data_dir=tmp_path, fixed_action="A2A3")
    assert s.fixed_action == "A2A3"


def test_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("PREFLIGHT_TAU", "0.42")
    s = Settings(data_dir=tmp_path)
    assert s.tau == 0.42
