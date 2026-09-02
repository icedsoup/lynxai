"""tiny player -> platforms -> goal environment, used only to validate the
observation -> model -> action -> environment -> reward -> training loop
before touching CS2. Not meant to be an interesting game."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .base import Environment

ACTIONS = ("left", "right", "jump", "noop")
ACTION_INDEX = {name: i for i, name in enumerate(ACTIONS)}


@dataclass(frozen=True)
class Platform:
    x_start: float
    x_end: float
    y: float


@dataclass(frozen=True)
class Smoke2DConfig:
    gravity: float = -1.0
    jump_velocity: float = 4.0
    move_speed: float = 1.0
    max_steps: int = 300
    fall_y: float = -10.0
    goal_x: float = 20.0
    goal_radius: float = 1.0
    platforms: tuple[Platform, ...] = field(
        default_factory=lambda: (
            Platform(x_start=-2.0, x_end=6.0, y=0.0),
            Platform(x_start=9.0, x_end=40.0, y=0.0),
        )
    )


class Smoke2DEnv(Environment[np.ndarray, int]):
    """Player starts on a platform, must jump a gap, and walk to a goal
    zone on the far platform. Falling into the gap ends the episode with
    a penalty; reaching the goal ends it with a reward.
    """

    def __init__(self, config: Smoke2DConfig | None = None):
        self.config = config or Smoke2DConfig()
        self._x = 0.0
        self._y = 0.0
        self._vy = 0.0
        self._steps = 0
        self._done = False

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        self._x = float(rng.uniform(-1.0, 1.0))
        self._y = self._ground_height_at(self._x)
        self._vy = 0.0
        self._steps = 0
        self._done = False
        return self._observation()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if self._done:
            raise RuntimeError("step() called after episode ended; call reset() first")

        cfg = self.config
        name = ACTIONS[action]
        self._steps += 1

        on_ground_before = self._on_ground()
        prev_y = self._y

        if name == "jump" and on_ground_before:
            self._vy = cfg.jump_velocity

        dx = cfg.move_speed if name == "right" else (-cfg.move_speed if name == "left" else 0.0)
        self._x += dx

        self._vy += cfg.gravity
        self._y += self._vy

        # landing is only valid when we're crossing the ground surface from
        # above this step -- otherwise an agent that fell past a platform's
        # height before reaching it horizontally would incorrectly "catch"
        # on that platform instead of continuing to fall.
        ground_y = self._ground_height_at(self._x)
        landed = prev_y >= ground_y and self._y <= ground_y and self._vy <= 0.0
        if landed:
            self._y = ground_y
            self._vy = 0.0

        reward = -0.01
        done = False
        info: dict[str, Any] = {}

        reached_goal = abs(self._x - cfg.goal_x) <= cfg.goal_radius and self._on_ground()
        fell = self._y < cfg.fall_y
        timed_out = self._steps >= cfg.max_steps

        if reached_goal:
            reward += 10.0
            done = True
            info["outcome"] = "goal"
        elif fell:
            reward -= 10.0
            done = True
            info["outcome"] = "fell"
        elif timed_out:
            reward -= 1.0
            done = True
            info["outcome"] = "timeout"

        self._done = done
        return self._observation(), reward, done, info

    def _ground_height_at(self, x: float) -> float:
        for platform in self.config.platforms:
            if platform.x_start <= x <= platform.x_end:
                return platform.y
        return self.config.fall_y - 1.0

    def _on_ground(self) -> bool:
        return self._y <= self._ground_height_at(self._x) + 1e-6

    def _observation(self) -> np.ndarray:
        cfg = self.config
        return np.array(
            [self._x, self._y, self._vy, float(self._on_ground()), cfg.goal_x - self._x],
            dtype=np.float32,
        )


__all__ = ["Smoke2DEnv", "Smoke2DConfig", "Platform", "ACTIONS", "ACTION_INDEX"]