from __future__ import annotations

from importlib.metadata import version
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import uuid

import numpy as np
import torch
import yaml

from lynxai.artifacts.store import CONTRACT, atomic_json, digest, read_json, utc, verify_checkpoint
from lynxai.environments import Smoke2DEnv
from lynxai.evaluation.core import Baseline


class StorageLimit(RuntimeError):
    pass


def worker(path: Path, operation="verify", split="validation", destination=None):
    args = [sys.executable, "-m", "lynxai.training.worker", operation, str(path)]
    if operation == "evaluate":
        args.extend([split, str(destination)])
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"Checkpoint worker failed: {result.stderr.strip()}")
    return json.loads(result.stdout.splitlines()[-1])


def rng_state(reset_rng):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "reset_generator": reset_rng.bit_generator.state}


def restore_rng(state, reset_rng):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    reset_rng.bit_generator.state = state["reset_generator"]


def prediction_probe(model, config):
    environment = Smoke2DEnv(config.task.environment)
    controller = Baseline("scripted", 0)
    observations = []
    try:
        for seed in (17, 29, 43):
            obs, _ = environment.reset(seed=seed)
            controller.reset()
            for _ in range(min(config.task.environment.max_steps, 40)):
                observations.append(obs.tolist())
                obs, _, terminated, truncated, _ = environment.step(controller.action(obs))
                if terminated or truncated:
                    observations.append(obs.tolist())
                    break
    finally:
        environment.close()
    actions = model.predict(np.asarray(observations, dtype=np.float32), deterministic=True)[0]
    return {"observations": observations, "actions": actions.tolist()}


class CheckpointStore:
    def __init__(self, run, config):
        self.run = run
        self.config = config
        self.saved = {}

    def save(self, model, environment, reason):
        steps = model.num_timesteps
        if steps in self.saved:
            path = self.saved[steps]
            self.run.event("checkpoint_role", checkpoint=path.name, role=reason)
            return path
        checkpoint_id = f"step-{steps:012d}-{uuid.uuid4().hex[:8]}"
        stage = self.run.path / "checkpoints" / (".staging-" + uuid.uuid4().hex)
        stage.mkdir()
        destination = stage.with_name(checkpoint_id)
        state = rng_state(environment.reset_rng)
        try:
            model.save(stage / "model.zip", exclude=["after_update"])
            torch.save(state, stage / "state.pt")
            (stage / "config.resolved.yaml").write_text(yaml.safe_dump(self.config.resolved(), sort_keys=False), encoding="utf-8")
            atomic_json(stage / "probe.json", prediction_probe(model, self.config))
            metadata = {"schema_version": 1, "id": checkpoint_id, "run_id": self.run.id,
                        "label": f"{self.config.name} Â· {steps:,} steps", "created_at": utc(), "reason": reason,
                        "contract": CONTRACT, "cumulative_steps": steps, "completed_updates": model.completed_updates,
                        "optimizer_steps": model.optimizer_steps, "sb3_epochs": model._n_updates,
                        "partial_episode": environment.partial_episode(),
                        "runtime": {name: version(name) for name in ("stable-baselines3", "torch", "numpy", "gymnasium")},
                        "checksums": {name: digest(stage / name) for name in ("model.zip", "state.pt", "config.resolved.yaml", "probe.json")}}
            atomic_json(stage / "checkpoint.json", metadata)
            required = sum(p.stat().st_size for p in stage.iterdir())
            existing = sum(p.stat().st_size for p in destination.parent.rglob("*") if p.is_file() and stage not in p.parents)
            if required + existing > self.config.checkpoints.max_bytes:
                raise StorageLimit("Checkpoint storage cap would be exceeded")
            worker(stage)
            stage.rename(destination)
            self.saved[steps] = destination
            self.run.update(latest=checkpoint_id)
            self.run.event("checkpoint_saved", checkpoint=checkpoint_id, role=reason, steps=steps)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
            restore_rng(state, environment.reset_rng)
        result = worker(destination, "evaluate", destination=self.run.path / "evaluations")
        self.run.event("validation", checkpoint=checkpoint_id, result=result)
        if not result["complete"]:
            raise RuntimeError("Checkpoint validation contains invalid or missing attempts")
        best = self.run.manifest.get("best_score")
        if best is None or result["success_rate"] > best:
            self.run.update(best=checkpoint_id, best_score=result["success_rate"])
        return destination


def compatible(path: Path):
    record = verify_checkpoint(path)
    for name, saved in record["runtime"].items():
        if version(name) != saved:
            raise ValueError(f"Resume requires {name} {saved}, installed {version(name)}")
    return record
