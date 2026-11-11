from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from lynxai.environments import Platform, Smoke2DConfig
from lynxai.utils.smoke_config import SmokeConfig, SmokeExperiment, load_smoke_config
from lynxai.utils.validation import finite_number


def mapping(value: Any, cls: type, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} field names must be strings")
    unknown = value.keys() - {item.name for item in fields(cls)}
    if unknown:
        raise ValueError(f"{label}: unknown fields {sorted(unknown)}")
    return value


def positive_int(value: Any, label: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")


def seeds(value: Any, label: str) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a nonempty list")
    if any(type(seed) is not int or not 0 <= seed < 2**31 for seed in value):
        raise ValueError(f"{label} requires integer seeds in [0, 2**31)")
    if len(value) != len(set(value)):
        raise ValueError(f"{label} must contain distinct seeds")


@dataclass(frozen=True)
class Training:
    total_steps: int = 65536
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    learning_rate: float = 0.0003
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_sizes: list[int] = field(default_factory=lambda: [64, 64])
    device: str = "cpu"
    torch_threads: int = 1
    normalize_advantage: bool = True
    clip_range_vf: None = None
    target_kl: float | None = None
    use_sde: bool = False
    sde_sample_freq: int = -1
    stats_window_size: int = 100
    ortho_init: bool = True
    adam_eps: float = 0.00001

    def __post_init__(self):
        for name in ("total_steps", "n_steps", "batch_size", "n_epochs", "torch_threads", "stats_window_size"):
            positive_int(getattr(self, name), name)
        if self.n_steps < 2 or self.batch_size < 2 or self.n_steps % self.batch_size:
            raise ValueError("batch_size must divide n_steps and both must be at least 2")
        if self.total_steps % self.n_steps:
            raise ValueError("total_steps must be a multiple of n_steps")
        for name in ("learning_rate", "clip_range", "max_grad_norm", "adam_eps"):
            finite_number(getattr(self, name), name)
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.clip_range >= 1:
            raise ValueError("clip_range must be less than 1")
        for name in ("gamma", "gae_lambda"):
            finite_number(getattr(self, name), name)
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        for name in ("ent_coef", "vf_coef"):
            finite_number(getattr(self, name), name)
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if not isinstance(self.hidden_sizes, list) or not self.hidden_sizes:
            raise ValueError("hidden_sizes must be a nonempty list")
        for size in self.hidden_sizes:
            positive_int(size, "hidden size")
        for name in ("normalize_advantage", "ortho_init"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if self.target_kl is not None:
            finite_number(self.target_kl, "target_kl")
            if self.target_kl <= 0:
                raise ValueError("target_kl must be positive")
        if self.device != "cpu" or self.clip_range_vf is not None or self.use_sde is not False or self.sde_sample_freq != -1:
            raise ValueError("M1 supports CPU, no value clipping, and categorical actions without SDE")


@dataclass(frozen=True)
class Evaluation:
    validation_seeds: list[int] = field(default_factory=lambda: list(range(10000, 10100)))
    test_seeds: list[int] = field(default_factory=lambda: list(range(20000, 20100)))
    action_seed: int = 1337

    def __post_init__(self):
        seeds(self.validation_seeds, "validation_seeds")
        seeds(self.test_seeds, "test_seeds")
        seeds([self.action_seed], "action_seed")
        if set(self.validation_seeds) & set(self.test_seeds):
            raise ValueError("validation and test seeds must be disjoint")


@dataclass(frozen=True)
class Checkpoints:
    every_updates: int = 10
    max_bytes: int = 1073741824

    def __post_init__(self):
        positive_int(self.every_updates, "every_updates")
        positive_int(self.max_bytes, "max_bytes")


@dataclass(frozen=True)
class Resources:
    max_seconds: float = 900

    def __post_init__(self):
        finite_number(self.max_seconds, "max_seconds")
        if self.max_seconds <= 0:
            raise ValueError("max_seconds must be positive")


@dataclass(frozen=True)
class Experiment:
    schema_version: int = 1
    name: str = "smoke-ppo"
    seed: int = 42
    artifact_root: str = "./artifacts"
    task: SmokeConfig = field(default_factory=SmokeConfig)
    training: Training = field(default_factory=Training)
    evaluation: Evaluation = field(default_factory=Evaluation)
    checkpoints: Checkpoints = field(default_factory=Checkpoints)
    resources: Resources = field(default_factory=Resources)

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("M1 schema_version must be 1")
        SmokeExperiment(self.name, self.seed, self.artifact_root)
        seeds([self.seed], "seed")

    def resolved(self) -> dict:
        return asdict(self)


def from_resolved(raw: dict) -> Experiment:
    data = dict(mapping(raw, Experiment, "experiment"))
    if "task" in data:
        task = dict(mapping(data.pop("task"), SmokeConfig, "task"))
        env = dict(mapping(task.pop("environment", None), Smoke2DConfig, "environment"))
        platforms = env.get("platforms")
        if not isinstance(platforms, (list, tuple)):
            raise ValueError("saved environment.platforms must be a sequence")
        env["platforms"] = tuple(Platform(**mapping(p, Platform, "platform")) for p in platforms)
        task["environment"] = Smoke2DConfig(**env)
        task["experiment"] = SmokeExperiment(**mapping(task.get("experiment"), SmokeExperiment, "task.experiment"))
        if type(task.get("schema_version")) is not int or task.get("schema_version") != 1 or task.get("environment_id") != "smoke2d_v1":
            raise ValueError("unsupported saved task")
        data["task"] = SmokeConfig(**task)
    for name, cls in (("training", Training), ("evaluation", Evaluation), ("checkpoints", Checkpoints), ("resources", Resources)):
        if name in data:
            data[name] = cls(**mapping(data[name], cls, name))
    return Experiment(**data)


def load_experiment(path: str | Path | None = None) -> Experiment:
    if path is None:
        return Experiment()
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid experiment YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("experiment must be a mapping")
    raw = dict(raw)
    if "schema_version" not in raw:
        raise ValueError("schema_version is required")
    if "task_config" in raw:
        reference = raw.pop("task_config")
        if "task" in raw or not isinstance(reference, str) or not reference.strip():
            raise ValueError("task_config must be a path and cannot accompany task")
        raw["task"] = asdict(load_smoke_config(path.parent / reference))
    return from_resolved(raw)
