from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import signal
import time

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import FloatSchedule
from stable_baselines3.common.vec_env import DummyVecEnv

from lynxai.artifacts.store import Run
from lynxai.environments import Smoke2DEnv
from lynxai.experiments.config import Experiment, load_experiment
from lynxai.training.checkpoints import CheckpointStore, StorageLimit, compatible, restore_rng


class BoundaryStop(Exception):
    pass


class EpisodeRecorder(gym.Wrapper):
    def __init__(self, config, run):
        super().__init__(Smoke2DEnv(config.task.environment))
        self.reset_rng = np.random.default_rng(config.seed)
        self.excluded = set(config.evaluation.validation_seeds + config.evaluation.test_seeds)
        self.run = run
        self.episode_seed = None
        self.length = 0
        self.total = 0.0
        self.active = False
        self.episode_started = 0.0
        self.completed_episodes = []

    def reset(self, **kwargs):
        seed = int(self.reset_rng.integers(0, 2**31))
        while seed in self.excluded:
            seed = int(self.reset_rng.integers(0, 2**31))
        self.episode_seed, self.length, self.total = seed, 0, 0.0
        self.active = True
        self.episode_started = time.monotonic()
        self.run.event("episode_start", reset_seed=seed)
        return self.env.reset(seed=seed)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if not np.isfinite(observation).all() or not math.isfinite(reward):
            raise ValueError("Nonfinite training transition")
        self.length += 1
        self.total += float(reward)
        if terminated or truncated:
            self.active = False
            episode = {"reset_seed": self.episode_seed, "outcome": info["outcome"],
                       "episode_return": self.total, "length": self.length, "terminated": bool(terminated),
                       "truncated": bool(truncated), "duration_seconds": time.monotonic() - self.episode_started}
            self.run.metric("episode", **episode)
            self.completed_episodes.append(episode)
        return observation, reward, terminated, truncated, info

    def partial_episode(self):
        return {"reset_seed": self.episode_seed, "length": self.length, "return": self.total} if self.active else None


class RecordedPPO(PPO):
    completed_updates = 0
    optimizer_steps = 0
    after_update = None

    def train(self):
        super().train()
        self.completed_updates += 1
        if self.after_update is not None:
            self.after_update()


class CollectionTiming(BaseCallback):
    def __init__(self, run):
        super().__init__()
        self.run = run
        self.started = None
        self.seconds = 0.0

    def _on_rollout_start(self):
        self.started = time.monotonic()
        self.run.update(status="collecting")

    def _on_rollout_end(self):
        self.seconds = time.monotonic() - self.started
        self.run.update(status="updating")

    def _on_step(self):
        return True


def train(config: Experiment, *, artifact_root=None, resume: Path | None = None):
    parent = None
    record = None
    if resume is not None:
        record = compatible(resume)
        parent = {"run_id": record["run_id"], "checkpoint_id": record["id"]}
    run = Run(config, "resume" if resume else "train", artifact_root, parent)
    started = time.monotonic()
    env = EpisodeRecorder(config, run)
    vector = DummyVecEnv([lambda: env])
    model = None
    stop_requests = 0
    previous_signal = signal.getsignal(signal.SIGINT)
    def interrupt(signum, frame):
        nonlocal stop_requests
        stop_requests += 1
        if stop_requests > 1:
            raise KeyboardInterrupt
    signal.signal(signal.SIGINT, interrupt)
    store = CheckpointStore(run, config)
    stop_reason = None
    base_steps = record["cumulative_steps"] if record else 0
    base_updates = record["completed_updates"] if record else 0
    base_optimizer = record["optimizer_steps"] if record else 0
    try:
        run.update(status="initializing")
        torch.set_num_threads(config.training.torch_threads)
        settings = config.training
        if resume:
            saved = load_experiment(resume / "config.resolved.yaml")
            expected = replace(saved, name=config.name, artifact_root=config.artifact_root,
                               training=replace(saved.training, total_steps=settings.total_steps,
                                                learning_rate=settings.learning_rate, ent_coef=settings.ent_coef))
            if config != expected:
                raise ValueError("Resume may change only name, artifact root, additional steps, learning rate, and entropy coefficient")
            model = RecordedPPO.load(resume / "model.zip", env=vector, device="cpu", force_reset=True)
            model.completed_updates = record["completed_updates"]
            model.optimizer_steps = record["optimizer_steps"]
            model.learning_rate = settings.learning_rate
            model.lr_schedule = FloatSchedule(settings.learning_rate)
            for group in model.policy.optimizer.param_groups:
                group["lr"] = settings.learning_rate
            model.ent_coef = settings.ent_coef
            state = torch.load(resume / "state.pt", map_location="cpu", weights_only=False)
            restore_rng(state, env.reset_rng)
            run.event("resume_boundary", discarded_episode=record["partial_episode"], source=parent,
                      overrides={"learning_rate": {"old": saved.training.learning_rate, "new": settings.learning_rate},
                                 "ent_coef": {"old": saved.training.ent_coef, "new": settings.ent_coef}})
        else:
            arguments = {name: getattr(settings, name) for name in (
                "n_steps", "batch_size", "n_epochs", "learning_rate", "gamma", "gae_lambda",
                "clip_range", "ent_coef", "vf_coef", "max_grad_norm", "normalize_advantage",
                "clip_range_vf", "target_kl", "use_sde", "sde_sample_freq", "stats_window_size")}
            model = RecordedPPO("MlpPolicy", vector, seed=config.seed, device="cpu", verbose=0,
                                policy_kwargs={"net_arch": {"pi": settings.hidden_sizes, "vf": settings.hidden_sizes},
                                               "activation_fn": torch.nn.Tanh, "ortho_init": settings.ortho_init,
                                               "optimizer_kwargs": {"eps": settings.adam_eps}}, **arguments)
        model.set_logger(configure(str(run.path / "tensorboard"), ["tensorboard"]))
        timing = CollectionTiming(run)
        phase_started = time.monotonic()
        def optimizer_step(optimizer, args, kwargs):
            model.optimizer_steps += 1
        hook = model.policy.optimizer.register_step_post_hook(optimizer_step)
        def after_update():
            nonlocal phase_started, stop_reason
            now = time.monotonic()
            elapsed = now - started
            local_steps = model.num_timesteps - base_steps
            local_updates = model.completed_updates - base_updates
            run.update(local_steps=local_steps, cumulative_steps=model.num_timesteps,
                       completed_updates=model.completed_updates, optimizer_steps=model.optimizer_steps,
                       local_updates=local_updates, local_optimizer_steps=model.optimizer_steps-base_optimizer,
                       elapsed_seconds=elapsed)
            diagnostics = {}
            for key, value in model.logger.name_to_value.items():
                if key.startswith("train/") and isinstance(value, (int, float, np.number)):
                    diagnostics[key] = float(value) if math.isfinite(value) else None
            duration = now - phase_started
            optimization_seconds = max(0.0, duration - timing.seconds)
            episodes = env.completed_episodes
            env.completed_episodes = []
            outcomes = {name: sum(episode["outcome"] == name for episode in episodes)
                        for name in ("goal", "fell", "timeout")}
            mean_return = float(np.mean([episode["episode_return"] for episode in episodes])) if episodes else None
            run.metric("update", cumulative_steps=model.num_timesteps, local_steps=local_steps,
                       completed_updates=model.completed_updates, optimizer_steps=model.optimizer_steps,
                       sb3_epochs=model._n_updates, collection_seconds=timing.seconds,
                       optimization_seconds=optimization_seconds,
                       collection_steps_per_second=settings.n_steps / max(timing.seconds, 1e-9),
                       update_steps_per_second=settings.n_steps / max(duration, 1e-9),
                       elapsed_seconds=elapsed, diagnostics=diagnostics, outcomes=outcomes,
                       mean_episode_return=mean_return)
            for name, count in outcomes.items():
                model.logger.record(f"episode/{name}", count)
            if mean_return is not None:
                model.logger.record("episode/mean_return", mean_return)
            model.logger.dump(model.num_timesteps)
            final = local_steps >= settings.total_steps
            stop_reason = "user_stop" if stop_requests else ("wall_clock_limit" if elapsed >= config.resources.max_seconds else None)
            if final or stop_reason or local_updates % config.checkpoints.every_updates == 0:
                run.update(status="evaluating")
                store.save(model, env, "final" if final else ("stop" if stop_reason else "periodic"))
            if final:
                stop_reason = None
            elif not stop_reason:
                stop_reason = "user_stop" if stop_requests else ("wall_clock_limit" if time.monotonic()-started >= config.resources.max_seconds else None)
                if stop_reason:
                    store.save(model, env, "stop")
            if stop_reason:
                raise BoundaryStop
            phase_started = time.monotonic()
        model.after_update = after_update
        run.update(cumulative_steps=base_steps, completed_updates=base_updates, optimizer_steps=base_optimizer)
        store.save(model, env, "initial")
        if stop_requests or time.monotonic()-started >= config.resources.max_seconds:
            stop_reason = "user_stop" if stop_requests else "wall_clock_limit"
            raise BoundaryStop
        phase_started = time.monotonic()
        model.learn(total_timesteps=settings.total_steps, callback=timing, reset_num_timesteps=False)
        hook.remove()
        run.update(status="completed", reason="step_budget", elapsed_seconds=time.monotonic()-started)
    except BoundaryStop:
        run.update(status="stopped", reason=stop_reason, elapsed_seconds=time.monotonic()-started)
    except StorageLimit as exc:
        run.event("storage_limit", message=str(exc))
        run.update(status="stopped", reason="storage_limit", elapsed_seconds=time.monotonic()-started)
    except KeyboardInterrupt:
        run.update(status="stopped", reason="forced_interrupt", elapsed_seconds=time.monotonic()-started)
    except Exception as exc:
        run.event("failure", error=f"{type(exc).__name__}: {exc}")
        run.update(status="failed", reason=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.monotonic()-started)
        raise
    finally:
        signal.signal(signal.SIGINT, previous_signal)
        vector.close()
    return run
