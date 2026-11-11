from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from lynxai.experiments.config import Checkpoints, Evaluation, Experiment, Resources, Training, from_resolved, load_experiment
from lynxai.evaluation.core import evaluate
from lynxai.artifacts.store import atomic_json, read_json


@pytest.mark.parametrize("settings", [
    {"total_steps": 3}, {"batch_size": 63}, {"batch_size": True}, {"n_steps": 1},
    {"learning_rate": float("nan")}, {"learning_rate": 0}, {"gamma": 2},
    {"ent_coef": -1}, {"hidden_sizes": []}, {"hidden_sizes": [True]},
    {"device": "cuda"}, {"normalize_advantage": 1}, {"target_kl": 0},
])
def test_training_validation(settings):
    with pytest.raises(ValueError):
        Training(**settings)


def test_resolved_config_survives_moved_sources(tmp_path):
    directory = tmp_path / "configs"
    directory.mkdir()
    (directory / "task.yaml").write_text("schema_version: 1\nenvironment:\n  max_steps: 30\n")
    config = directory / "experiment.yaml"
    config.write_text("schema_version: 1\ntask_config: task.yaml\n")
    resolved = load_experiment(config)
    assert resolved.task.environment.max_steps == 30
    saved = tmp_path / "resolved.yaml"
    saved.write_text(yaml.safe_dump(resolved.resolved()))
    config.unlink()
    (directory / "task.yaml").unlink()
    assert load_experiment(saved) == resolved
    assert from_resolved(resolved.resolved()) == resolved


@pytest.mark.parametrize("raw", [
    {}, {"schema_version": True}, {"schema_version": 1, "surprise": 2},
    {"schema_version": 1, "training": {"surprise": 2}},
    {"schema_version": 1, "task_config": 4}, {"schema_version": 1, "task_config": None},
    {"schema_version": 1, "task": {"environment": {}}},
])
def test_invalid_yaml(tmp_path, raw):
    file = tmp_path / "invalid.yaml"
    file.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        load_experiment(file)


def test_suites_and_limits():
    with pytest.raises(ValueError):
        Evaluation(validation_seeds=[1], test_seeds=[1])
    with pytest.raises(ValueError):
        Evaluation(validation_seeds=[1, 1])
    with pytest.raises(ValueError):
        Resources(max_seconds=float("inf"))
    with pytest.raises(ValueError):
        Checkpoints(max_bytes=0)


def test_baselines_share_starts_and_have_expected_outcomes(tmp_path):
    cfg = replace(Experiment(), evaluation=Evaluation(validation_seeds=list(range(10000, 10010))))
    scripted = evaluate(cfg, tmp_path, split="validation", baseline="scripted")
    noop = evaluate(cfg, tmp_path, split="validation", baseline="noop")
    assert scripted["success_rate"] == 1
    assert noop["timeouts"] == 10 and noop["complete"]
    random_a = evaluate(cfg, tmp_path, split="validation", baseline="random")
    random_b = evaluate(cfg, tmp_path, split="validation", baseline="random")
    for key in ("goals", "falls", "timeouts", "mean_return", "mean_length"):
        assert random_a[key] == random_b[key]
    for result in (scripted, noop, random_a):
        assert read_json(Path(result["path"]) / "manifest.json")["seeds"] == cfg.evaluation.validation_seeds


def test_invalid_attempts_remain_in_denominator(tmp_path, monkeypatch):
    from lynxai.evaluation.core import Baseline
    def broken(self, observation):
        raise ValueError("injected")
    monkeypatch.setattr(Baseline, "action", broken)
    cfg = replace(Experiment(), evaluation=Evaluation(validation_seeds=[1, 2]))
    result = evaluate(cfg, tmp_path, split="validation", baseline="random")
    assert result["invalid_episodes"] == 2
    assert result["expected_episodes"] == 2 and result["success_rate"] == 0
    assert not result["complete"]


def test_atomic_manifest_failure_preserves_previous(tmp_path, monkeypatch):
    import lynxai.artifacts.store as store
    target = tmp_path / "manifest.json"
    atomic_json(target, {"status": "valid"})
    def denied(*args):
        raise OSError("injected")
    monkeypatch.setattr(store.os, "replace", denied)
    with pytest.raises(OSError):
        atomic_json(target, {"status": "broken"})
    assert read_json(target) == {"status": "valid"}
    assert list(tmp_path.glob("*.tmp")) == []
