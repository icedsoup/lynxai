from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import uuid
import zipfile

import yaml

from lynxai.experiments.config import Experiment
from lynxai.utils.paths import resolve_artifact_root

CONTRACT = {"environment": "smoke2d_v1", "observation": "numeric5_v1", "actions": ["left", "right", "jump", "noop"], "reward": "smoke_reward_v1"}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def identity() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:12]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def append_json(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")
        stream.flush()


def read_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            result.append({"kind": "invalid_record", "raw": line})
    return result


def provenance(directory: Path) -> dict:
    packages = {}
    for package in ("lynxai", "numpy", "gymnasium", "stable-baselines3", "torch", "tensorboard"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    root = Path(__file__).resolve().parents[3]
    result = {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
              "cpu": platform.processor(), "hostname": socket.gethostname(), "packages": packages,
              "commit": None, "dirty": None, "lock_sha256": None}
    if (root / "pyproject.toml").exists():
        def git(*args):
            return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=10)
        try:
            commit = git("rev-parse", "HEAD")
            status = git("status", "--porcelain")
        except (OSError, subprocess.TimeoutExpired) as exc:
            result["git_error"] = str(exc)
            commit = status = None
        if commit and status and commit.returncode == 0 and status.returncode == 0:
            result.update(commit=commit.stdout.strip(), dirty=bool(status.stdout.strip()))
            (directory / "source.diff").write_text(git("diff", "HEAD", "--", "src", "configs", "tests", "scripts", "pyproject.toml", "uv.lock").stdout, encoding="utf-8")
        lock = root / "uv.lock"
        if lock.exists():
            result["lock_sha256"] = digest(lock)
        with zipfile.ZipFile(directory / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for folder in ("src", "configs", "tests", "scripts"):
                for path in sorted((root / folder).rglob("*")):
                    if path.is_file() and "__pycache__" not in path.parts and path.suffix in (".py", ".yaml"):
                        archive.write(path, path.relative_to(root))
            for name in ("pyproject.toml", "uv.lock", "README.md"):
                if (root / name).exists():
                    archive.write(root / name, name)
        result["source_sha256"] = digest(directory / "source.zip")
    return result


class Run:
    def __init__(self, config: Experiment, operation: str, artifact_root=None, parent=None):
        self.root = resolve_artifact_root(artifact_root, config.artifact_root)
        self.id = identity()
        self.path = self.root / "runs" / self.id
        self.path.mkdir(parents=True)
        for name in ("checkpoints", "evaluations", "report"):
            (self.path / name).mkdir()
        (self.path / "config.resolved.yaml").write_text(yaml.safe_dump(config.resolved(), sort_keys=False), encoding="utf-8")
        self.manifest = {"schema_version": 1, "id": self.id, "name": config.name, "operation": operation,
                         "status": "created", "created_at": utc(), "updated_at": utc(), "pid": os.getpid(),
                         "host": socket.gethostname(), "parent": parent, "contract": CONTRACT, "seed": config.seed,
                         "config_sha256": digest(self.path / "config.resolved.yaml"), "latest": None, "best": None,
                         "local_steps": 0, "cumulative_steps": 0, "completed_updates": 0, "optimizer_steps": 0,
                         "provenance": provenance(self.path)}
        self.update()
        print(f"Run {self.id} â€” {config.name}\nArtifacts: {self.path}", flush=True)

    def update(self, **values):
        self.manifest.update(values, updated_at=utc())
        atomic_json(self.path / "manifest.json", self.manifest)

    def event(self, kind: str, **values):
        append_json(self.path / "events.jsonl", {"time": utc(), "kind": kind, **values})

    def metric(self, kind: str, **values):
        append_json(self.path / "metrics.jsonl", {"time": utc(), "kind": kind, **values})


def run_path(root: Path, reference: str) -> Path:
    supplied = Path(reference)
    path = supplied if supplied.is_dir() else root / "runs" / reference
    if not (path / "manifest.json").is_file():
        raise ValueError(f"Unknown run: {reference}")
    return path.resolve()


def checkpoint_path(root: Path, reference: str) -> Path:
    supplied = Path(reference)
    if supplied.is_dir():
        path = supplied
    else:
        parts = reference.replace("\\", "/").split("/")
        if len(parts) != 2 or any(part in ("", ".", "..") for part in parts):
            raise ValueError("Checkpoint must be a directory or RUN_ID/CHECKPOINT_ID")
        path = root / "runs" / parts[0] / "checkpoints" / parts[1]
    if not (path / "checkpoint.json").is_file():
        raise ValueError(f"Unknown checkpoint: {reference}")
    return path.resolve()


def verify_checkpoint(path: Path) -> dict:
    record = read_json(path / "checkpoint.json")
    if record.get("schema_version") != 1 or record.get("contract") != CONTRACT:
        raise ValueError("Incompatible checkpoint contract")
    required = {"model.zip", "state.pt", "config.resolved.yaml", "probe.json"}
    checksums = record.get("checksums", {})
    if set(checksums) != required:
        raise ValueError("Incomplete checkpoint checksums")
    for name, expected in checksums.items():
        if digest(path / name) != expected:
            raise ValueError(f"Checkpoint checksum mismatch: {name}")
    return record
