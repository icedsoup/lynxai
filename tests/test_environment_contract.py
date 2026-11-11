from dataclasses import replace

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from lynxai.environments import ACTION_INDEX, Smoke2DConfig, Smoke2DEnv


def test_gymnasium_contract():
    env = Smoke2DEnv()
    check_env(env, skip_render_check=True)
    assert env.action_space.n == 4


def test_seeded_reset_and_rng_continuation():
    a, b = Smoke2DEnv(), Smoke2DEnv()
    first, _ = a.reset(seed=42)
    np.testing.assert_array_equal(first, b.reset(seed=42)[0])
    next_a, _ = a.reset()
    np.testing.assert_array_equal(next_a, b.reset()[0])
    assert not np.array_equal(first, next_a)
    np.testing.assert_array_equal(first, a.reset(seed=42)[0])


def test_spaces_contain_random_rollouts_and_terminal_observations():
    env = Smoke2DEnv()
    env.action_space.seed(3)
    for seed in range(10):
        obs, info = env.reset(seed=seed)
        assert info == {} and env.observation_space.contains(obs)
        for _ in range(env.config.max_steps):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            assert env.observation_space.contains(obs)
            assert np.isfinite(obs).all() and np.isfinite(reward)
            if terminated or truncated:
                assert info["outcome"] in {"goal", "fell", "timeout"}
                break
        else:
            pytest.fail("episode did not finish within its cap")


@pytest.mark.parametrize("action", [-1, 4, 1.0, True, np.bool_(True), "right", None, [1], np.array([1])])
def test_invalid_actions_do_not_change_state(action):
    env, control = Smoke2DEnv(), Smoke2DEnv()
    env.reset(seed=10)
    control.reset(seed=10)
    with pytest.raises(ValueError, match="action must"):
        env.step(action)
    got = env.step(np.int64(ACTION_INDEX["right"]))
    expected = control.step(ACTION_INDEX["right"])
    np.testing.assert_array_equal(got[0], expected[0])
    assert got[1:] == expected[1:]


def test_reset_required_and_cleanup_idempotent():
    env = Smoke2DEnv(Smoke2DConfig(max_steps=1))
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)
    env.reset(seed=1)
    env.step(3)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(3)
    env.close()
    env.close()
    with pytest.raises(RuntimeError, match="reset"):
        env.step(3)
    env.reset(seed=1)
    env.step(3)


@pytest.mark.parametrize("outcome", ["goal", "fell", "timeout"])
def test_terminal_flags_rewards_and_limit_precedence(outcome):
    # Discover the unmodified trajectory's terminal step, then place the cap there.
    cfg = Smoke2DConfig(goal_x=0.0) if outcome == "goal" else Smoke2DConfig()
    action = ACTION_INDEX["noop" if outcome != "fell" else "right"]
    reference = Smoke2DEnv(cfg)
    reference.reset(seed=0)
    for count in range(1, cfg.max_steps + 1):
        _, _, term, trunc, info = reference.step(action)
        if term or trunc:
            break
    env = Smoke2DEnv(replace(cfg, max_steps=count))
    env.reset(seed=0)
    for _ in range(count):
        _, reward, terminated, truncated, info = env.step(action)
    assert info == {"outcome": outcome}
    assert terminated is (outcome != "timeout")
    assert truncated is (outcome == "timeout")
    assert reward == pytest.approx({"goal": 9.99, "fell": -10.01, "timeout": -1.01}[outcome])


def test_reject_reset_options():
    with pytest.raises(ValueError, match="options"):
        Smoke2DEnv().reset(options={"scenario_id": "unimplemented"})


def test_sb3_checker_and_vector_timeout():
    sb3_checker = pytest.importorskip("stable_baselines3.common.env_checker")
    vec_module = pytest.importorskip("stable_baselines3.common.vec_env")
    sb3_checker.check_env(Smoke2DEnv())
    vec = vec_module.DummyVecEnv([lambda: Smoke2DEnv(Smoke2DConfig(max_steps=1))])
    try:
        vec.seed(0)
        initial = vec.reset()
        _, rewards, dones, infos = vec.step([ACTION_INDEX["noop"]])
        assert bool(dones[0])
        assert rewards[0] == pytest.approx(-1.01)
        assert infos[0]["TimeLimit.truncated"] is True
        assert infos[0]["outcome"] == "timeout"
        np.testing.assert_array_equal(infos[0]["terminal_observation"], initial[0])
        assert vec.envs[0].observation_space.contains(infos[0]["terminal_observation"])
    finally:
        vec.close()
