from __future__ import annotations

import argparse
import sys
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryFile
from typing import Callable

from lynxai.utils.paths import resolve_artifact_root


def _writable(root: Path) -> str:
    root.mkdir(parents=True, exist_ok=True)
    # Probe actual writes: os.access does not reliably reflect Windows ACLs.
    with TemporaryFile(dir=root) as probe:
        probe.write(b"lynxai doctor")
        probe.flush()
    return str(root)


def _dependency(module: str) -> str:
    dependency = import_module(module)
    return f"{module} {dependency.__version__}"


def doctor(
    profile: str = "core",
    config: str | Path | None = None,
    artifact_root: str | Path | None = None,
) -> int:
    failed = False

    def check(label: str, operation: Callable[[], object]) -> object | None:
        nonlocal failed
        try:
            result = operation()
        except Exception as exc:
            failed = True
            print(f"FAIL  {label}: {exc}")
            return None
        print(f"PASS  {label}: {result}")
        return result

    def python_version() -> str:
        if sys.version_info < (3, 12):
            raise RuntimeError("Python 3.12 or newer required; use uv sync --python 3.12")
        return sys.version.split()[0]

    check("Python", python_version)

    def resource() -> str:
        from lynxai.utils.smoke_config import default_config_text

        default_config_text()
        return "bundled smoke2d.yaml"

    check("Package resources", resource)

    def configuration():
        from lynxai.utils.smoke_config import load_smoke_config

        return load_smoke_config(config)

    cfg = check("Smoke configuration", configuration)
    if cfg is not None:
        def storage() -> str:
            return _writable(resolve_artifact_root(artifact_root, cfg.experiment.artifact_root))

        check("Artifact root", storage)

        def environment() -> str:
            from gymnasium.utils.env_checker import check_env
            from lynxai.environments import Smoke2DEnv

            env = Smoke2DEnv(cfg.environment)
            try:
                check_env(env, skip_render_check=True)
            finally:
                env.close()
            return "Gymnasium contract verified (rendering unavailable)"

        check("Smoke2D", environment)
    else:
        print("SKIP  Artifact root and Smoke2D: correct the configuration first")

    if profile == "training":
        torch = check("PyTorch", lambda: _dependency("torch"))
        check("Stable-Baselines3", lambda: _dependency("stable_baselines3"))
        if torch is not None:
            def cuda() -> str:
                available = import_module("torch").cuda.is_available()
                return "CUDA available" if available else "CUDA unavailable; CPU is supported"

            check("Compute (informational)", cuda)
        if failed:
            print("Training dependencies: run uv sync --locked --extra ppo")
    print("INFO  CS2 readiness: unverified; no live host checks performed")
    return int(failed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lynxai", description="lynxai project tools")
    subparsers = parser.add_subparsers(dest="command")
    diagnostics = subparsers.add_parser("doctor", help="check core or training prerequisites")
    diagnostics.add_argument("--profile", choices=("core", "training"), default="core")
    diagnostics.add_argument("--config", type=Path, help="Smoke2D YAML; defaults to bundled config")
    diagnostics.add_argument("--artifact-root", type=Path, help="override configured artifact root")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return doctor(args.profile, args.config, args.artifact_root)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
