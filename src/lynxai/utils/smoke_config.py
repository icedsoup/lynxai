"""Strict, versioned Smoke2D configuration, separate from the CS2 scaffold."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from lynxai.environments import Platform, Smoke2DConfig


@dataclass(frozen=True)
class SmokeExperiment:
    name: str = "smoke2d"
    seed: int = 1337
    artifact_root: str = "./artifacts"

    def __post_init__(self) -> None:
        for name in ("name", "artifact_root"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or "\0" in value:
                raise ValueError(f"experiment.{name} must be a nonempty string without NUL")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("experiment.seed must be a nonnegative integer")


@dataclass(frozen=True)
class SmokeConfig:
    schema_version: int = 1
    experiment: SmokeExperiment = field(default_factory=SmokeExperiment)
    environment_id: str = "smoke2d_v1"
    environment: Smoke2DConfig = field(default_factory=Smoke2DConfig)


def default_config_text() -> str:
    return files("lynxai").joinpath("resources/smoke2d.yaml").read_text(encoding="utf-8")


def _mapping(value: Any, allowed: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    unknown = value.keys() - allowed
    if unknown:
        raise ValueError(f"{name}: unknown fields {sorted(map(str, unknown))}")
    return value


def load_smoke_config(path: str | Path | None = None) -> SmokeConfig:
    text = default_config_text() if path is None else Path(path).read_text(encoding="utf-8")
    raw = _mapping(yaml.safe_load(text), {"schema_version", "experiment", "environment"}, "config")
    version = raw.get("schema_version")
    if type(version) is not int or version != 1:
        raise ValueError("schema_version must be 1")
    experiment = _mapping(raw.get("experiment", {}), {"name", "seed", "artifact_root"}, "experiment")
    env = dict(_mapping(raw.get("environment", {}), {f.name for f in fields(Smoke2DConfig)} | {"id"}, "environment"))
    env_id = env.pop("id", "smoke2d_v1")
    if env_id != "smoke2d_v1":
        raise ValueError("environment.id must be smoke2d_v1")
    if "platforms" in env:
        if not isinstance(env["platforms"], list) or not env["platforms"]:
            raise ValueError("environment.platforms must be a nonempty list")
        platforms = []
        for value in env["platforms"]:
            platform = _mapping(value, {"x_start", "x_end", "y"}, "platform")
            if set(platform) != {"x_start", "x_end", "y"}:
                raise ValueError("each platform requires x_start, x_end, and y")
            platforms.append(Platform(**platform))
        env["platforms"] = tuple(platforms)
    return SmokeConfig(version, SmokeExperiment(**experiment), env_id, Smoke2DConfig(**env))
