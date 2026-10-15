"""Install the wheel with locked dependencies and test it outside the checkout.

Run with `uv run --locked --extra ppo python scripts/check_wheel.py` after building.
Uses an isolated temporary venv; requires uv on PATH. Works on Windows and Linux.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    wheels = sorted((root / "dist").glob("lynxai-*.whl"))
    if len(wheels) != 1:
        raise SystemExit("Expected one lynxai wheel in dist; build into an empty dist first")
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)

    def run(*args: str, cwd: Path = root) -> None:
        subprocess.run(args, cwd=cwd, env=environment, check=True)

    with tempfile.TemporaryDirectory(prefix="lynxai-wheel-") as temporary:
        directory = Path(temporary)
        venv = directory / "venv"
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        entry = venv / ("Scripts/lynxai.exe" if os.name == "nt" else "bin/lynxai")
        requirements = directory / "pylock.toml"
        run("uv", "export", "--locked", "--extra", "ppo", "--no-dev", "--no-emit-project",
            "--format", "pylock.toml", "--output-file", str(requirements), "--quiet")
        run("uv", "venv", "--python", "3.12", str(venv))
        run("uv", "pip", "sync", "--python", str(python), str(requirements))
        run("uv", "pip", "install", "--python", str(python), "--no-deps", str(wheels[0]))
        run(str(python), "-I", "-c", '''
from pathlib import Path
import sys
import lynxai
from lynxai.environments import ACTION_INDEX, Smoke2DEnv
from lynxai.utils.smoke_config import load_smoke_config
assert Path(lynxai.__file__).is_relative_to(Path(sys.prefix)), lynxai.__file__
assert not Path('src').exists()
cfg = load_smoke_config()
env = Smoke2DEnv(cfg.environment)
obs, _ = env.reset(seed=cfg.experiment.seed)
jumped = False
for _ in range(cfg.environment.max_steps):
    action = 'jump' if obs[3] and not jumped and obs[0] >= 5 else 'right'
    jumped |= action == 'jump'
    obs, _, terminated, truncated, info = env.step(ACTION_INDEX[action])
    if terminated or truncated:
        assert terminated and not truncated and info['outcome'] == 'goal', info
        break
else:
    raise AssertionError('scripted episode failed to finish')
env.close()
assert 'torch' not in sys.modules
print('Installed wheel: bundled config, namespaced imports, scripted goal verified')
''', cwd=directory)
        run(str(python), "-I", "-m", "lynxai", "doctor", cwd=directory)
        run(str(entry), "doctor", "--profile", "training", cwd=directory)
        print("Wheel verification passed outside the checkout")


if __name__ == "__main__":
    main()
