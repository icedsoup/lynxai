from dataclasses import replace
from pathlib import Path
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")
pytest.importorskip("tensorboard")

from lynxai.artifacts.store import Run, atomic_json, digest, read_json, read_lines, verify_checkpoint
from lynxai.experiments.config import Checkpoints, Evaluation, Experiment, Training, load_experiment
from lynxai.experiments.operations import inspect, report
from lynxai.training.checkpoints import worker
from lynxai.training.runner import EpisodeRecorder, train


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("training")
    config = Experiment(name="test-training", training=Training(total_steps=256, n_steps=128, batch_size=64, n_epochs=2),
                        evaluation=Evaluation(validation_seeds=[30000, 30001], test_seeds=[40000, 40001]))
    run = train(config, artifact_root=root)
    return root, config, run


def test_real_updates_and_immutable_evaluation(trained):
    from stable_baselines3 import PPO
    root, config, run = trained
    assert run.manifest["status"] == "completed"
    assert run.manifest["local_steps"] == 256
    assert run.manifest["completed_updates"] == 2
    assert run.manifest["optimizer_steps"] == 8
    metrics = read_lines(run.path / "metrics.jsonl")
    updates = [row for row in metrics if row["kind"] == "update"]
    assert len(updates) == 2
    assert sum(sum(row["outcomes"].values()) for row in updates) == sum(row["kind"] == "episode" for row in metrics)
    checkpoints = sorted((run.path / "checkpoints").glob("step-*"))
    initial = PPO.load(checkpoints[0] / "model.zip")
    final = PPO.load(checkpoints[-1] / "model.zip")
    assert any(not torch.equal(a, b) for a, b in zip(initial.policy.parameters(), final.policy.parameters()))
    checkpoint = checkpoints[-1]
    before_hash = digest(checkpoint / "model.zip")
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    result = worker(checkpoint, "evaluate", "test", run.path / "evaluations")
    assert result["parameters_unchanged"] and result["complete"]
    assert digest(checkpoint / "model.zip") == before_hash
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)
    record = verify_checkpoint(checkpoint)
    assert record["completed_updates"] == 2 and record["optimizer_steps"] == 8
    assert worker(checkpoint)["verified"]


def test_resume_restores_optimizer_and_allows_overrides(trained, monkeypatch):
    from stable_baselines3 import PPO
    root, config, source = trained
    checkpoint = source.path / "checkpoints" / source.manifest["latest"]
    old_hash = digest(checkpoint / "model.zip")
    saved_model = PPO.load(checkpoint / "model.zip")
    old_step = max(int(state["step"]) for state in saved_model.policy.optimizer.state.values())
    resumed_cfg = replace(config, name="branch", training=replace(config.training, total_steps=128, learning_rate=0.0001, ent_coef=0.02))
    resumed = train(resumed_cfg, artifact_root=root, resume=checkpoint)
    assert resumed.manifest["parent"]["checkpoint_id"] == checkpoint.name
    assert resumed.manifest["local_steps"] == 128 and resumed.manifest["cumulative_steps"] == 384
    assert resumed.manifest["completed_updates"] == 3 and resumed.manifest["optimizer_steps"] == 12
    saved = resumed.path / "checkpoints" / resumed.manifest["latest"]
    model = PPO.load(saved / "model.zip")
    assert model.ent_coef == 0.02 and model.lr_schedule(0.5) == 0.0001
    assert all(group["lr"] == 0.0001 for group in model.policy.optimizer.param_groups)
    assert max(int(state["step"]) for state in model.policy.optimizer.state.values()) == old_step + 4
    events = read_lines(resumed.path / "events.jsonl")
    assert any(event["kind"] == "resume_boundary" for event in events)
    assert digest(checkpoint / "model.zip") == old_hash
    tree = inspect(source.id, root, tree=True)["tree"]
    assert resumed.id in [child["id"] for child in tree["children"]]
    result = report([source.id, resumed.id], root)
    assert len(result["runs"]) == 2
    assert (source.path / "report/report.md").exists()


def test_corrupt_checkpoint_rejected(trained, tmp_path):
    import shutil
    root, config, run = trained
    source = run.path / "checkpoints" / run.manifest["latest"]
    target = tmp_path / "copy"
    shutil.copytree(source, target)
    (target / "model.zip").write_bytes(b"broken")
    with pytest.raises(ValueError, match="checksum"):
        verify_checkpoint(target)
    record = read_json(target / "checkpoint.json")
    record["contract"]["environment"] = "other"
    atomic_json(target / "checkpoint.json", record)
    with pytest.raises(ValueError, match="contract"):
        verify_checkpoint(target)


def test_reset_stream_excludes_evaluation_and_restores(tmp_path):
    cfg = Experiment()
    run = Run(cfg, "test", tmp_path)
    a = EpisodeRecorder(cfg, run)
    for _ in range(50):
        a.reset()
        assert a.episode_seed not in a.excluded
    state = a.reset_rng.bit_generator.state
    a.reset()
    expected = a.episode_seed
    b = EpisodeRecorder(cfg, run)
    b.reset_rng.bit_generator.state = state
    b.reset()
    assert b.episode_seed == expected
    assert b.length == 0


def test_storage_limit_leaves_no_published_checkpoint(tmp_path, monkeypatch):
    import lynxai.training.checkpoints as storage
    cfg = Experiment(training=Training(total_steps=128, n_steps=128), checkpoints=Checkpoints(max_bytes=1))
    monkeypatch.setattr(storage, "worker", lambda *a, **kw: pytest.fail("storage must be checked before verification"))
    run = train(cfg, artifact_root=tmp_path)
    assert run.manifest["reason"] == "storage_limit"
    assert run.manifest["latest"] is None
    assert list((run.path / "checkpoints").iterdir()) == []


def test_post_update_stop_and_forced_interrupt(tmp_path, monkeypatch):
    import lynxai.training.runner as runner
    import lynxai.training.checkpoints as storage
    monkeypatch.setattr(storage, "worker", lambda *a, **kw: {"verified": True, "complete": True, "success_rate": 0.0})
    cfg = Experiment(training=Training(total_steps=512, n_steps=128, batch_size=64, n_epochs=2))
    original_train = runner.RecordedPPO.train
    def stop_after_update(self):
        runner.signal.getsignal(runner.signal.SIGINT)(runner.signal.SIGINT, None)
        original_train(self)
    monkeypatch.setattr(runner.RecordedPPO, "train", stop_after_update)
    normal = train(cfg, artifact_root=tmp_path)
    assert normal.manifest["reason"] == "user_stop"
    assert normal.manifest["local_steps"] == 128
    saved = read_json(normal.path / "checkpoints" / normal.manifest["latest"] / "checkpoint.json")
    assert saved["optimizer_steps"] == 4
    def force_before_update(self):
        handler = runner.signal.getsignal(runner.signal.SIGINT)
        handler(runner.signal.SIGINT, None)
        handler(runner.signal.SIGINT, None)
    monkeypatch.setattr(runner.RecordedPPO, "train", force_before_update)
    forced = train(cfg, artifact_root=tmp_path)
    assert forced.manifest["reason"] == "forced_interrupt"
    saved = read_json(forced.path / "checkpoints" / forced.manifest["latest"] / "checkpoint.json")
    assert saved["cumulative_steps"] == 0


def test_failed_publish_keeps_initial_snapshot(tmp_path, monkeypatch):
    import lynxai.training.checkpoints as storage
    calls = 0
    def verification(path, operation="verify", **kwargs):
        nonlocal calls
        if operation == "verify":
            calls += 1
            if calls == 2:
                raise RuntimeError("injected failed load")
            return {"verified": True}
        return {"complete": True, "success_rate": 0.0}
    monkeypatch.setattr(storage, "worker", verification)
    cfg = Experiment(training=Training(total_steps=128, n_steps=128))
    with pytest.raises(RuntimeError, match="failed load"):
        train(cfg, artifact_root=tmp_path)
    manifest_path = next((tmp_path / "runs").glob("*/manifest.json"))
    record = read_json(manifest_path)
    assert record["status"] == "failed"
    published = list((manifest_path.parent / "checkpoints").iterdir())
    assert len(published) == 1 and published[0].name == record["latest"]


def test_stop_during_periodic_validation_reuses_boundary(tmp_path, monkeypatch):
    import signal
    import lynxai.training.checkpoints as storage

    def verification(path, operation="verify", **kwargs):
        if operation == "evaluate" and path.name.startswith("step-000000000128-"):
            signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
        return {"verified": True, "complete": True, "success_rate": 0.0}

    monkeypatch.setattr(storage, "worker", verification)
    config = Experiment(training=Training(total_steps=256, n_steps=128), checkpoints=Checkpoints(every_updates=1))
    run = train(config, artifact_root=tmp_path)
    assert run.manifest["reason"] == "user_stop"
    assert run.manifest["local_steps"] == 128
    result = inspect(run.id, tmp_path)
    assert len(result["checkpoints"]) == 2
    assert result["checkpoints"][-1]["roles"] == ["latest", "periodic", "stop"]
