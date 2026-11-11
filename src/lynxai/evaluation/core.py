from __future__ import annotations

import math
from pathlib import Path
import time

import numpy as np

from lynxai.artifacts.store import append_json, atomic_json, identity, utc
from lynxai.environments import ACTION_INDEX, Smoke2DEnv


class Baseline:
    def __init__(self, kind: str, seed: int):
        if kind not in ("noop", "random", "scripted"):
            raise ValueError(f"Unknown baseline: {kind}")
        self.kind = kind
        self.rng = np.random.default_rng(seed)
        self.jumped = False

    def reset(self):
        self.jumped = False

    def action(self, observation):
        if self.kind == "noop":
            return ACTION_INDEX["noop"]
        if self.kind == "random":
            return int(self.rng.integers(4))
        if observation[3] and not self.jumped and observation[0] >= 5:
            self.jumped = True
            return ACTION_INDEX["jump"]
        return ACTION_INDEX["right"]


class Learned:
    def __init__(self, model):
        self.model = model

    def reset(self):
        pass

    def action(self, observation):
        return int(self.model.predict(observation, deterministic=True)[0])


def summarize(records: list[dict], expected: int) -> dict:
    valid = [record for record in records if record["valid"]]
    successes = [record for record in valid if record["outcome"] == "goal"]
    n = len(valid)
    rate = len(successes) / expected if expected else 0.0
    z = 1.959963984540054
    if n:
        p = len(successes) / n
        denominator = 1 + z*z/n
        center = (p + z*z/(2*n)) / denominator
        margin = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
        interval = [max(0.0, center-margin), min(1.0, center+margin)]
    else:
        interval = None
    def mean(values):
        return float(np.mean(values)) if values else None
    return {"expected_episodes": expected, "attempted_episodes": len(records), "valid_episodes": n,
            "invalid_episodes": len(records)-n, "complete": n == expected and len(records) == expected,
            "goals": len(successes), "falls": sum(r["outcome"] == "fell" for r in valid),
            "timeouts": sum(r["outcome"] == "timeout" for r in valid), "success_rate": rate,
            "episode_success_wilson95": interval, "mean_return": mean([r["return"] for r in valid]),
            "mean_length": mean([r["length"] for r in valid]),
            "successful_mean_length": mean([r["length"] for r in successes])}


def evaluate(config, directory: Path, *, split: str, baseline=None, model=None, checkpoint=None):
    suite = getattr(config.evaluation, f"{split}_seeds")
    evaluation_id = identity()
    destination = directory / evaluation_id
    destination.mkdir(parents=True)
    record = {"schema_version": 1, "id": evaluation_id, "created_at": utc(), "split": split,
              "seeds": suite, "baseline": baseline, "checkpoint": checkpoint,
              "action_seed": config.evaluation.action_seed if baseline == "random" else None,
              "action_selection": "uniform" if baseline == "random" else "deterministic", "status": "running"}
    atomic_json(destination / "manifest.json", record)
    policy = Baseline(baseline, config.evaluation.action_seed) if baseline else Learned(model)
    env = Smoke2DEnv(config.task.environment)
    episodes = []
    try:
        for index, seed in enumerate(suite):
            started = time.monotonic()
            episode = {"index": index, "reset_seed": seed, "action_seed": record["action_seed"],
                       "valid": False, "outcome": "invalid", "return": 0.0, "length": 0,
                       "terminated": False, "truncated": False}
            try:
                observation, _ = env.reset(seed=seed)
                policy.reset()
                for step in range(config.task.environment.max_steps):
                    observation, reward, terminated, truncated, info = env.step(policy.action(observation))
                    if not np.isfinite(observation).all() or not math.isfinite(reward):
                        raise ValueError("Nonfinite transition")
                    episode["return"] += float(reward)
                    episode["length"] = step + 1
                    if terminated or truncated:
                        episode.update(valid=True, outcome=info["outcome"], terminated=bool(terminated), truncated=bool(truncated))
                        break
                if not episode["valid"]:
                    raise ValueError("Episode exceeded its declared cap")
            except Exception as exc:
                episode["error"] = f"{type(exc).__name__}: {exc}"
            episode["duration_seconds"] = time.monotonic() - started
            episodes.append(episode)
            append_json(destination / "episodes.jsonl", episode)
    finally:
        env.close()
        summary = summarize(episodes, len(suite))
        atomic_json(destination / "summary.json", summary)
        record["status"] = "completed" if summary["complete"] else "failed"
        atomic_json(destination / "manifest.json", record)
    return {"path": str(destination), "id": evaluation_id, **summary}
