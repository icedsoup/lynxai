from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from lynxai.environments import Platform, Smoke2DConfig
from lynxai.utils.config import load_config
from lynxai.utils.smoke_config import default_config_text, load_smoke_config


def save(tmp_path, raw):
    path = tmp_path / "smoke.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_bundled_and_checkout_defaults_agree():
    assert load_smoke_config().environment == Smoke2DConfig()
    source = Path(__file__).resolve().parents[1] / "configs/smoke2d.yaml"
    assert load_smoke_config(source) == load_smoke_config()
    assert source.read_text(encoding="utf-8") == default_config_text()


def test_partial_defaults_and_custom_values(tmp_path):
    minimal = load_smoke_config(save(tmp_path, {"schema_version": 1}))
    assert minimal == load_smoke_config()
    raw = yaml.safe_load(default_config_text())
    raw["experiment"].update(name="custom", seed=0, artifact_root="other")
    raw["environment"].update(max_steps=10, gravity=-0.5)
    cfg = load_smoke_config(save(tmp_path, raw))
    assert cfg.environment.max_steps == 10
    assert cfg.environment.gravity == -0.5
    assert cfg.experiment.seed == 0
    assert cfg.experiment.artifact_root == "other"


@pytest.mark.parametrize("raw", [None, [], "text", 4, {}, {"schema_version": True}, {"schema_version": 2}, {"schema_version": "1"}])
def test_invalid_root_and_version(tmp_path, raw):
    with pytest.raises(ValueError):
        load_smoke_config(save(tmp_path, raw))


@pytest.mark.parametrize("section,key,value", [
    (None, "extra", 1), (None, "training", {}),
    ("experiment", "extra", 1), ("environment", "extra", 1),
    ("environment", "id", "cs2"),
    ("experiment", "name", ""), ("experiment", "name", 1),
    ("experiment", "artifact_root", None), ("experiment", "artifact_root", " "),
    ("experiment", "seed", -1), ("experiment", "seed", True), ("experiment", "seed", 1.0),
    ("environment", "platforms", []), ("environment", "platforms", {}),
    ("environment", "platforms", [None]),
    ("environment", "platforms", [{"x_start": -2, "x_end": 6}]),
    ("environment", "platforms", [{"x_start": -2, "x_end": 6, "y": 0, "extra": 1}]),
])
def test_invalid_schema_values(tmp_path, section, key, value):
    raw = yaml.safe_load(default_config_text())
    (raw if section is None else raw[section])[key] = value
    with pytest.raises(ValueError):
        load_smoke_config(save(tmp_path, raw))


@pytest.mark.parametrize("section", ["environment", "experiment"])
@pytest.mark.parametrize("value", [None, [], 3, "invalid"])
def test_sections_require_mappings(tmp_path, section, value):
    with pytest.raises(ValueError, match="mapping"):
        load_smoke_config(save(tmp_path, {"schema_version": 1, section: value}))


@pytest.mark.parametrize("kwargs", [
    {"gravity": 0}, {"gravity": 1}, {"gravity": "-1"}, {"gravity": True},
    {"jump_velocity": 0}, {"move_speed": -1}, {"goal_radius": 0},
    {"max_steps": 0}, {"max_steps": True}, {"max_steps": 1.0},
    {"goal_x": float("nan")}, {"gravity": float("inf")}, {"fall_y": float("-inf")},
    {"platforms": ()}, {"platforms": ("bad",)},
    {"platforms": (Platform(0, 6, 0),)},
    {"platforms": (Platform(-2, 6, -10),)},
    {"platforms": (Platform(-2, 6, 0), Platform(5, 9, 0))},
    {"platforms": (Platform(-2, 6, 0), Platform(6, 9, 0))},
])
def test_physics_validation_shared_by_direct_and_yaml_construction(tmp_path, kwargs):
    with pytest.raises(ValueError):
        Smoke2DConfig(**kwargs)
    raw = {"schema_version": 1, "environment": dict(kwargs)}
    if "platforms" in kwargs:
        raw["environment"]["platforms"] = [asdict(p) if isinstance(p, Platform) else p for p in kwargs["platforms"]]
    with pytest.raises(ValueError):
        load_smoke_config(save(tmp_path, raw))


@pytest.mark.parametrize("args", [(1, 0, 0), (0, 0, 0), (-2, 6, float("nan")), ("-2", 6, 0)])
def test_platform_validation(args):
    with pytest.raises(ValueError):
        Platform(*args)


def test_cs2_scaffold_stays_separate():
    source = Path(__file__).resolve().parents[1] / "configs/baseline.yaml"
    assert load_config(source).experiment.name == "lynxai-baseline"
    with pytest.raises(ValueError):
        load_smoke_config(source)
