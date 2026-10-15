"""tiny player -> platforms -> goal environment, used only to validate the
observation -> model -> action -> environment -> reward -> training loop
before touching CS2. Not meant to be an interesting game."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from lynxai.utils.validation import finite_number

ACTIONS = ("left", "right", "jump", "noop")
ACTION_INDEX = {name: i for i, name in enumerate(ACTIONS)}


@dataclass(frozen=True)
class Platform:
    x_start: float
    x_end: float
    y: float

    def __post_init__(self) -> None:
        for name in ("x_start", "x_end", "y"):
            finite_number(getattr(self, name), f"platform.{name}")
        if self.x_start >= self.x_end:
            raise ValueError("platform.x_start must be less than x_end")


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

    def __post_init__(self) -> None:
        for name in ("gravity", "jump_velocity", "move_speed", "fall_y", "goal_x", "goal_radius"):
            finite_number(getattr(self, name), name)
        if self.gravity >= 0:
            raise ValueError("gravity must be negative")
        for name in ("jump_velocity", "move_speed", "goal_radius"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if type(self.max_steps) is not int or self.max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        if not isinstance(self.platforms, tuple) or not self.platforms:
            raise ValueError("platforms must be a nonempty tuple of Platform objects")
        if any(not isinstance(p, Platform) for p in self.platforms):
            raise ValueError("platforms must contain Platform objects")
        ordered = sorted(self.platforms, key=lambda p: p.x_start)
        if any(a.x_end >= b.x_start for a, b in zip(ordered, ordered[1:])):
            raise ValueError("platforms must not overlap or share endpoints")
        if any(p.y <= self.fall_y for p in self.platforms):
            raise ValueError("platforms must be above fall_y")
        if not any(p.x_start <= -1 and p.x_end >= 1 for p in self.platforms):
            raise ValueError("one platform must cover the entire reset interval [-1, 1]")


class Smoke2DEnv(gym.Env[np.ndarray, int]):
    """Player starts on a platform, must jump a gap, and walk to a goal
    zone on the far platform. Falling into the gap ends the episode with
    a penalty; reaching the goal ends it with a reward.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Smoke2DConfig | None = None):
        self.config = config or Smoke2DConfig()
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf, 0, -np.inf], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf, 1, np.inf], dtype=np.float32),
            dtype=np.float32,
        )
        self._needs_reset = True
        self._x = 0.0
        self._y = 0.0
        self._vy = 0.0
        self._steps = 0
        self._done = False

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if options:
            raise ValueError("Smoke2D reset does not support options")
        super().reset(seed=seed)
        self._x = float(self.np_random.uniform(-1.0, 1.0))
        self._y = self._ground_height_at(self._x)
        self._vy = 0.0
        self._steps = 0
        self._done = False
        self._needs_reset = False
        return self._observation(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._needs_reset or self._done:
            raise RuntimeError("step() requires an active episode; call reset() first")

        if isinstance(action, (bool, np.bool_)) or not self.action_space.contains(action):
            raise ValueError(f"action must be an integer in [0, {len(ACTIONS) - 1}], got {action!r}")
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
        terminated = False
        truncated = False
        info: dict[str, Any] = {}

        reached_goal = abs(self._x - cfg.goal_x) <= cfg.goal_radius and self._on_ground()
        fell = self._y < cfg.fall_y
        timed_out = self._steps >= cfg.max_steps

        if reached_goal:
            reward += 10.0
            terminated = True
            info["outcome"] = "goal"
        elif fell:
            reward -= 10.0
            terminated = True
            info["outcome"] = "fell"
        elif timed_out:
            reward -= 1.0
            truncated = True
            info["outcome"] = "timeout"

        self._done = terminated or truncated
        return self._observation(), reward, terminated, truncated, info

    def close(self) -> None:
        self._needs_reset = True

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
