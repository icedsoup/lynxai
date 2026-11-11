from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import socket
import statistics
import time

from lynxai.artifacts.store import Run, atomic_json, checkpoint_path, read_json, read_lines, run_path
from lynxai.evaluation.core import evaluate
from lynxai.experiments.config import Evaluation, Experiment, Training, load_experiment
from lynxai.utils.paths import resolve_artifact_root


def evaluate_run(*, baseline=None, checkpoint=None, config=None, split="validation", artifact_root=None, action_seed=None):
    root = resolve_artifact_root(artifact_root)
    parent = None
    path = None
    if checkpoint:
        path = checkpoint_path(root, checkpoint)
        cfg = load_experiment(path / "config.resolved.yaml")
        metadata = read_json(path / "checkpoint.json")
        parent = {"run_id": metadata["run_id"], "checkpoint_id": metadata["id"]}
    else:
        cfg = load_experiment(config)
    if action_seed is not None:
        cfg = replace(cfg, evaluation=replace(cfg.evaluation, action_seed=action_seed))
    cfg = replace(cfg, name=f"{baseline or 'checkpoint'}-{split}")
    started = time.monotonic()
    run = Run(cfg, "evaluate", artifact_root, parent)
    try:
        run.update(status="evaluating", split=split, baseline=baseline)
        if path:
            from lynxai.training.checkpoints import worker
            result = worker(path, "evaluate", split, run.path / "evaluations")
        else:
            result = evaluate(cfg, run.path / "evaluations", split=split, baseline=baseline)
        run.update(status="completed" if result["complete"] else "failed", result=result,
                   reason="invalid_attempts" if not result["complete"] else None, elapsed_seconds=time.monotonic() - started)
        return run
    except Exception as exc:
        run.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
        raise


def process_alive(pid):
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5
        try:
            status = wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(status))) and status.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def inspect(reference, artifact_root=None, tree=False):
    root = resolve_artifact_root(artifact_root)
    directory = run_path(root, reference)
    manifest = read_json(directory / "manifest.json")
    observed = manifest["status"]
    if observed not in ("completed", "stopped", "failed") and manifest["host"] == socket.gethostname() and not process_alive(manifest["pid"]):
        observed = "interrupted"
    checkpoints = []
    events = read_lines(directory / "events.jsonl")
    scores = {event["checkpoint"]: event["result"] for event in events if event.get("kind") == "validation"}
    for path in sorted((directory / "checkpoints").glob("*/checkpoint.json")):
        if path.parent.name.startswith("."):
            continue
        metadata = read_json(path)
        roles = [metadata["reason"]]
        roles.extend(event["role"] for event in events if event.get("kind") == "checkpoint_role" and event["checkpoint"] == metadata["id"])
        for role in ("latest", "best"):
            if metadata["id"] == manifest.get(role):
                roles.append(role)
        checkpoints.append({**metadata, "roles": sorted(set(roles)), "validation": scores.get(metadata["id"])})
    result = {**manifest, "observed_status": observed, "path": str(directory), "checkpoints": checkpoints}
    if tree:
        manifests = [read_json(path) for path in (directory.parent).glob("*/manifest.json")]
        by_id = {item["id"]: item for item in manifests}
        ancestor = manifest
        seen = set()
        while ancestor.get("parent") and ancestor["id"] not in seen:
            seen.add(ancestor["id"])
            parent = by_id.get(ancestor["parent"]["run_id"])
            if parent is None:
                break
            ancestor = parent
        def branch(item, visited):
            if item["id"] in visited:
                return {"id": item["id"], "cycle": True}
            visited = visited | {item["id"]}
            return {"id": item["id"], "name": item["name"], "operation": item["operation"], "parent": item.get("parent"),
                    "children": [branch(child, visited) for child in manifests
                                 if child.get("parent", {}) and child["parent"]["run_id"] == item["id"]]}
        result["tree"] = branch(ancestor, set())
    return result


def print_inspection(result, as_json=False):
    if as_json:
        print(json.dumps(result, indent=2))
        return
    print(f"{result['name']} [{result['id']}] — {result['observed_status']}")
    print(f"Progress: {result['local_steps']:,} local / {result['cumulative_steps']:,} cumulative steps")
    for checkpoint in result["checkpoints"]:
        validation = checkpoint["validation"]
        score = f"{validation['success_rate']:.1%}" if validation else "unscored"
        print(f"  {checkpoint['id']}  {checkpoint['label']}  [{', '.join(checkpoint['roles'])}]  {score}")
    def show(node, depth=0):
        parent = node.get("parent")
        origin = f" from {parent['checkpoint_id']}" if parent else ""
        print("  " * depth + f"- {node['name']} [{node['id']}]{origin}")
        for child in node.get("children", []):
            show(child, depth+1)
    if "tree" in result:
        show(result["tree"])


def report(references, artifact_root=None):
    runs = [inspect(reference, artifact_root) for reference in references]
    rows = []
    for run in runs:
        for path in sorted((Path(run["path"]) / "evaluations").glob("*/summary.json")):
            manifest = read_json(path.parent / "manifest.json")
            summary = read_json(path)
            rows.append({"run_id": run["id"], "name": run["name"], "seed": run["seed"],
                         "status": run["observed_status"], "evaluation_id": manifest["id"], "split": manifest["split"],
                         "baseline": manifest["baseline"], "checkpoint": manifest["checkpoint"], "seeds": manifest["seeds"],
                         "action_seed": manifest["action_seed"], **summary})
    candidates = {}
    for row in rows:
        checkpoint = row["checkpoint"]
        if row["split"] == "test" and checkpoint and row["complete"]:
            key = (checkpoint["run_id"], checkpoint["id"], tuple(row["seeds"]))
            candidates.setdefault(key, row)
    policies = list(candidates.values())
    aggregates = None
    if policies and len({row["seed"] for row in policies}) == len(policies) and len({tuple(row["seeds"]) for row in policies}) == 1:
        rates = [row["success_rate"] for row in policies]
        aggregates = {"mean": statistics.mean(rates), "minimum": min(rates), "maximum": max(rates),
                      "by_seed": {row["seed"]: row["success_rate"] for row in policies}}
    result = {"schema_version": 1, "runs": runs, "evaluations": rows, "test_policy_scores": aggregates,
              "uncertainty_note": "Wilson intervals describe episode success within a fixed policy and reset distribution. Seed-level means and ranges describe the listed training runs. Repeated evaluations are not independent training seeds."}
    destination = Path(runs[0]["path"]) / "report"
    atomic_json(destination / "summary.json", result)
    lines = ["# Smoke2D experiment report", "", result["uncertainty_note"], "",
             "| Run | Split | Controller/checkpoint | Success | Falls | Timeouts | Invalid | Mean return | Mean steps | Successful steps | Episode Wilson 95% |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        controller = row["baseline"] or row["checkpoint"]["id"]
        if row["action_seed"] is not None:
            controller += f" (action seed {row['action_seed']})"
        interval = row["episode_success_wilson95"]
        uncertainty = f"{interval[0]:.1%}–{interval[1]:.1%}" if interval else "—"
        values = [f"{row['name']} ({row['seed']})", row["split"], controller,
                  f"{row['goals']}/{row['expected_episodes']}", row["falls"], row["timeouts"], row["invalid_episodes"],
                  row["mean_return"], row["mean_length"], row["successful_mean_length"], uncertainty]
        cells = [f"{value:.2f}" if isinstance(value, float) else str(value) if value is not None else "—" for value in values]
        lines.append("| " + " | ".join(cell.replace("|", r"\|").replace("\n", " ") for cell in cells) + " |")
    if aggregates:
        lines.extend(["", f"Test success across training seeds averaged {aggregates['mean']:.1%}, "
                      f"with a range of {aggregates['minimum']:.1%}–{aggregates['maximum']:.1%}."])
    for run in runs:
        lines.extend(["", f"## {run['name']} — {run['id']}", "",
                      f"Status: {run['observed_status']}. Reason: {run.get('reason', '')}.",
                      f"Steps: {run['local_steps']} local, {run['cumulative_steps']} cumulative. Elapsed seconds: {run.get('elapsed_seconds', 0):.2f}.",
                      f"Parent: {json.dumps(run['parent'])}.",
                      f"Configuration: [resolved YAML]({Path(run['path']).as_uri()}/config.resolved.yaml).",
                      f"Artifacts: {run['path']}"])
        tensorboard = Path(run["path"]) / "tensorboard"
        if tensorboard.is_dir():
            lines.append(f"Learning curves: [TensorBoard logs]({tensorboard.as_uri()}).")
        for checkpoint in run["checkpoints"]:
            checkpoint_dir = Path(run["path"]) / "checkpoints" / checkpoint["id"]
            roles = ", ".join(checkpoint["roles"])
            lines.append(f"- [{checkpoint['id']}]({checkpoint_dir.as_uri()}) ({roles})")
    (destination / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(destination / "report.md")
    return result


def smoke_test(artifact_root=None):
    from lynxai.training.runner import train
    cfg = Experiment(name="smoke-test", training=Training(total_steps=256, n_steps=128, batch_size=64, n_epochs=2),
                     evaluation=Evaluation(validation_seeds=list(range(30000, 30005)), test_seeds=list(range(40000, 40005))))
    initial = train(cfg, artifact_root=artifact_root)
    if initial.manifest["status"] != "completed" or initial.manifest["optimizer_steps"] == 0:
        raise RuntimeError("Training smoke test did not complete real updates")
    path = initial.path / "checkpoints" / initial.manifest["latest"]
    frozen = evaluate_run(checkpoint=str(path), split="test", artifact_root=artifact_root)
    if frozen.manifest["status"] != "completed" or not frozen.manifest["result"]["parameters_unchanged"]:
        raise RuntimeError("Frozen evaluation smoke test failed")
    resumed = train(replace(cfg, name="smoke-resume", training=replace(cfg.training, total_steps=128, learning_rate=0.0002, ent_coef=0.01)),
                    artifact_root=artifact_root, resume=path)
    if resumed.manifest["status"] != "completed" or resumed.manifest["cumulative_steps"] != 384:
        raise RuntimeError("Resume smoke test failed")
    report([str(initial.path), str(resumed.path), str(frozen.path)], artifact_root)
    return resumed
