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
    parser = argparse.ArgumentParser(prog="lynxai", description="lynxai experiment tools")
    subparsers = parser.add_subparsers(dest="command")
    diagnostics = subparsers.add_parser("doctor", help="check core or training prerequisites")
    diagnostics.add_argument("--profile", choices=("core", "training"), default="core")
    diagnostics.add_argument("--config", type=Path)
    diagnostics.add_argument("--artifact-root", type=Path)
    training = subparsers.add_parser("train", help="train or resume a Smoke2D policy")
    source = training.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path)
    source.add_argument("--resume")
    training.add_argument("--seed", type=int)
    training.add_argument("--name")
    training.add_argument("--additional-steps", type=int)
    training.add_argument("--learning-rate", type=float)
    training.add_argument("--ent-coef", type=float)
    training.add_argument("--artifact-root", type=Path)
    evaluation = subparsers.add_parser("evaluate", help="evaluate a baseline or frozen checkpoint")
    controller = evaluation.add_mutually_exclusive_group(required=True)
    controller.add_argument("--baseline", choices=("noop", "random", "scripted"))
    controller.add_argument("--checkpoint")
    evaluation.add_argument("--config", type=Path)
    evaluation.add_argument("--split", choices=("validation", "test"), default="validation")
    evaluation.add_argument("--action-seed", type=int)
    evaluation.add_argument("--artifact-root", type=Path)
    inspection = subparsers.add_parser("inspect-run", help="inspect progress, checkpoints, and ancestry")
    inspection.add_argument("run_id")
    inspection.add_argument("--tree", action="store_true")
    inspection.add_argument("--json", action="store_true")
    inspection.add_argument("--artifact-root", type=Path)
    reporting = subparsers.add_parser("report", help="create a comparison report")
    reporting.add_argument("run_ids", nargs="+")
    reporting.add_argument("--artifact-root", type=Path)
    smoke = subparsers.add_parser("smoke-test", help="verify training, checkpoints, evaluation, and resume")
    smoke.add_argument("--artifact-root", type=Path)
    return parser


def experiment_command(args) -> int:
    from dataclasses import replace
    from lynxai.artifacts.store import checkpoint_path
    from lynxai.experiments.config import load_experiment
    from lynxai.experiments import operations

    if args.command == "train":
        from lynxai.training.runner import train
        if args.resume:
            if args.additional_steps is None or args.seed is not None:
                raise ValueError("Resume requires --additional-steps and preserves the saved seed")
            path = checkpoint_path(resolve_artifact_root(args.artifact_root), args.resume)
            config = load_experiment(path / "config.resolved.yaml")
            changes = {"total_steps": args.additional_steps}
            for field in ("learning_rate", "ent_coef"):
                if getattr(args, field) is not None:
                    changes[field] = getattr(args, field)
            config = replace(config, name=args.name or config.name + "-resume", training=replace(config.training, **changes))
        else:
            if args.additional_steps is not None or args.learning_rate is not None or args.ent_coef is not None:
                raise ValueError("Set fresh-run hyperparameters in the experiment YAML")
            path = None
            config = load_experiment(args.config)
            config = replace(config, name=args.name if args.name is not None else config.name,
                             seed=args.seed if args.seed is not None else config.seed)
        run = train(config, artifact_root=args.artifact_root, resume=path)
        return 0 if run.manifest["status"] == "completed" else 3
    if args.command == "evaluate":
        if args.checkpoint and (args.config is not None or args.action_seed is not None):
            raise ValueError("Checkpoint evaluation uses its saved configuration and deterministic actions")
        run = operations.evaluate_run(baseline=args.baseline, checkpoint=args.checkpoint, config=args.config,
                                      split=args.split, artifact_root=args.artifact_root, action_seed=args.action_seed)
        print(f"Goal success: {run.manifest['result']['success_rate']:.1%}")
        return 0 if run.manifest["status"] == "completed" else 1
    if args.command == "inspect-run":
        operations.print_inspection(operations.inspect(args.run_id, args.artifact_root, args.tree), args.json)
    elif args.command == "report":
        operations.report(args.run_ids, args.artifact_root)
    elif args.command == "smoke-test":
        operations.smoke_test(args.artifact_root)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return doctor(args.profile, args.config, args.artifact_root)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        return experiment_command(args)
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
