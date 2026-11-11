from __future__ import annotations

import hashlib
import json
from pathlib import Path
import signal
import sys

import numpy as np
import torch
from stable_baselines3 import PPO

from lynxai.artifacts.store import verify_checkpoint
from lynxai.evaluation.core import evaluate
from lynxai.experiments.config import load_experiment


def main():
    operation, checkpoint = sys.argv[1:3]
    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    path = Path(checkpoint)
    record = verify_checkpoint(path)
    config = load_experiment(path / "config.resolved.yaml")
    torch.set_num_threads(config.training.torch_threads)
    model = PPO.load(path / "model.zip", device="cpu")
    probe = json.loads((path / "probe.json").read_text(encoding="utf-8"))
    observations = np.asarray(probe["observations"], dtype=np.float32)
    actions = model.predict(observations, deterministic=True)[0]
    if actions.tolist() != probe["actions"]:
        raise ValueError("Reloaded deterministic predictions differ")
    if operation == "verify":
        print(json.dumps({"verified": True}))
        return
    split, destination = sys.argv[3:5]
    before = hashlib.sha256(b"".join(tensor.detach().cpu().numpy().tobytes() for tensor in model.policy.state_dict().values())).hexdigest()
    result = evaluate(config, Path(destination), split=split, model=model,
                      checkpoint={"run_id": record["run_id"], "id": record["id"]})
    after = hashlib.sha256(b"".join(tensor.detach().cpu().numpy().tobytes() for tensor in model.policy.state_dict().values())).hexdigest()
    if before != after:
        raise ValueError("Evaluation modified model parameters")
    result["parameters_unchanged"] = True
    print(json.dumps(result))


if __name__ == "__main__":
    main()
