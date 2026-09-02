from __future__ import annotations

import numpy as np
import pytest

from environments import ACTION_INDEX, Smoke2DEnv


def test_reset_returns_expected_shape_and_dtype() -> None:
    env = Smoke2DEnv()
    obs = env.reset(seed=0)
    assert obs.shape == (5,)
    assert obs.dtype == np.float32


def test_walking_off_the_edge_without_jumping_falls() -> None:
    env = Smoke2DEnv()
    env.reset(seed=0)
    outcome = None
    for _ in range(300):
        _, _, done, info = env.step(ACTION_INDEX["right"])
        if done:
            outcome = info["outcome"]
            break
    assert outcome == "fell"


def test_jumping_the_gap_reaches_the_goal() -> None:
    for seed in range(5):
        env = Smoke2DEnv()
        obs = env.reset(seed=seed)
        jumped = False
        outcome = None
        for _ in range(300):
            x, _, _, on_ground, dist_to_goal = obs
            if on_ground and not jumped and x >= 5.0:
                jumped = True
                action = ACTION_INDEX["jump"]
            else:
                action = ACTION_INDEX["right"]
            obs, _, done, info = env.step(action)
            if done:
                outcome = info["outcome"]
                break
        assert outcome == "goal", f"seed {seed}: expected goal, got {outcome}"


def test_standing_still_times_out() -> None:
    env = Smoke2DEnv()
    env.reset(seed=0)
    outcome = None
    for _ in range(300):
        _, _, done, info = env.step(ACTION_INDEX["noop"])
        if done:
            outcome = info["outcome"]
            break
    assert outcome == "timeout"


def test_step_after_done_raises() -> None:
    env = Smoke2DEnv()
    env.reset(seed=0)
    with pytest.raises(RuntimeError):
        for _ in range(400):
            env.step(ACTION_INDEX["right"])