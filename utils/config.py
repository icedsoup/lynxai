from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


@dataclass
class ExperimentConfig:
    name: str = "lynxai-baseline"
    seed: int = 1337


@dataclass
class ObservationConfig:
    width: int = 84
    height: int = 84
    grayscale: bool = True
    history_length: int = 4
    channels: int = 1
    frame_skip: int = 1
    dtype: str = "float32"


@dataclass
class ActionsConfig:
    names: list[str] = field(
        default_factory=lambda: [
            "move_left",
            "move_right",
            "move_forward",
            "move_back",
            "jump",
            "shoot",
        ]
    )
    discrete: bool = True
    continuous: bool = False


@dataclass
class RewardConfig:
    kill: float = 10.0
    death: float = -10.0
    damage: float = 0.05
    survival: float = 0.01
    invalid_action_penalty: float = -0.01
    time_step: float = -0.001


@dataclass
class TrainingConfig:
    algorithm: str = "ppo"
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_steps: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    checkpoint_dir: str = "checkpoints"


@dataclass
class Config:
    experiment: ExperimentConfig
    observation: ObservationConfig
    actions: ActionsConfig
    reward: RewardConfig
    training: TrainingConfig


def _as_dataclass(data: dict[str, Any], cls: type[T]) -> T:
    return cls(**data)


def load_config(path: str | Path) -> Config:
    """Load a YAML experiment configuration file into a typed config object."""
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    if not isinstance(raw, dict):
        raise TypeError(f"Configuration root must be a mapping, got {type(raw).__name__}")

    return Config(
        experiment=_as_dataclass(raw.get("experiment", {}), ExperimentConfig),
        observation=_as_dataclass(raw.get("observation", {}), ObservationConfig),
        actions=_as_dataclass(raw.get("actions", {}), ActionsConfig),
        reward=_as_dataclass(raw.get("reward", {}), RewardConfig),
        training=_as_dataclass(raw.get("training", {}), TrainingConfig),
    )


__all__ = [
    "Config",
    "ExperimentConfig",
    "ObservationConfig",
    "ActionsConfig",
    "RewardConfig",
    "TrainingConfig",
    "load_config",
]
