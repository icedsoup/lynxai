from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from lynxai.utils.config import load_config
from lynxai.utils.paths import CHECKPOINT_DIR, CLIPS_DIR, DATA_DIR, LOG_DIR, PROJECT_ROOT, RUNS_DIR, ensure_directories


def _check_python() -> tuple[bool, str]:
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    ok = sys.version_info >= (3, 10)
    return ok, version


def _check_torch() -> tuple[bool, str]:
    try:
        import torch

        return True, f"torch {torch.__version__}"
    except Exception as exc:
        return False, str(exc)


def _check_cuda() -> tuple[bool, str]:
    try:
        import torch

        available = bool(torch.cuda.is_available())
        if available:
            return True, f"CUDA available ({torch.cuda.device_count()} device(s))"
        return True, "CUDA unavailable (CPU-only environment)"
    except Exception as exc:
        return False, str(exc)


def _check_paths() -> tuple[bool, str]:
    required = [DATA_DIR, RUNS_DIR, CHECKPOINT_DIR, CLIPS_DIR, LOG_DIR]
    try:
        for path in required:
            path.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                raise FileNotFoundError(f"missing: {path}")
            if not os.access(path, os.W_OK):
                raise PermissionError(f"not writable: {path}")
        return True, ", ".join(str(p.relative_to(PROJECT_ROOT)) for p in required)
    except Exception as exc:
        return False, str(exc)


def _check_baseline() -> tuple[bool, str]:
    config_path = PROJECT_ROOT / "configs" / "baseline.yaml"
    try:
        cfg = load_config(config_path)
        return True, f"{cfg.experiment.name} @ {cfg.training.algorithm}"
    except Exception as exc:
        return False, str(exc)


def _render_table(rows: list[tuple[str, str, str]]) -> str:
    headers = ("Check", "Status", "Details")
    widths = [len(h) for h in headers]
    for label, status, detail in rows:
        widths[0] = max(widths[0], len(label))
        widths[1] = max(widths[1], len(status))
        widths[2] = max(widths[2], len(detail))

    border = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    lines = [border]
    lines.append(f"| {headers[0]:<{widths[0]}} | {headers[1]:<{widths[1]}} | {headers[2]:<{widths[2]}} |")
    lines.append(border)
    for label, status, detail in rows:
        lines.append(f"| {label:<{widths[0]}} | {status:<{widths[1]}} | {detail:<{widths[2]}} |")
    lines.append(border)
    return "\n".join(lines)


def doctor() -> int:
    checks: list[tuple[str, str, str]] = []

    ok, value = _check_python()
    checks.append(("Python version", "PASS" if ok else "FAIL", value))

    ok, value = _check_torch()
    checks.append(("PyTorch import", "PASS" if ok else "FAIL", value))

    ok, value = _check_cuda()
    checks.append(("GPU visibility", "PASS" if ok else "FAIL", value))

    ok, value = _check_paths()
    checks.append(("Project dirs", "PASS" if ok else "FAIL", value))

    ok, value = _check_baseline()
    checks.append(("baseline.yaml", "PASS" if ok else "FAIL", value))

    print(_render_table(checks))
    return 0 if all(status == "PASS" for _, status, _ in checks) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lynxai", description="lynxai project tools")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("doctor", help="run environment checks")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "doctor":
        return doctor()

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
