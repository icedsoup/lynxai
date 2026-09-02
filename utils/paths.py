from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CLIPS_DIR = PROJECT_ROOT / "clips"
LOG_DIR = PROJECT_ROOT / "logs"


class RunPathSet:
    """run-scoped directory helpers"""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

    def run(self, run_id: str) -> Path:
        return self.base_dir / run_id

    def ensure(self, run_id: str) -> Path:
        path = self.run(run_id)
        path.mkdir(parents=True, exist_ok=True)
        return path


runs = RunPathSet(RUNS_DIR)
checkpoints = RunPathSet(CHECKPOINT_DIR)
clips = RunPathSet(CLIPS_DIR)
logs = RunPathSet(LOG_DIR)


def ensure_directories() -> None:
    """make runtime dirs"""
    for path in (DATA_DIR, RUNS_DIR, CHECKPOINT_DIR, CLIPS_DIR, LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)


__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "RUNS_DIR",
    "CHECKPOINT_DIR",
    "CLIPS_DIR",
    "LOG_DIR",
    "runs",
    "checkpoints",
    "clips",
    "logs",
    "ensure_directories",
    "RunPathSet",
]
