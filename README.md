# lynxai

A Python research platform working toward controlled Counter-Strike 2 learning
experiments. The current release is the **M0 foundation**: an installable Smoke2D
simulation, strict configuration, and diagnostics. Training, evaluation runs,
checkpoints, a GUI, and live CS2 integration are future milestones. See [plan.md](plan.md).

## Setup and checks

Use Python **3.12** and [uv](https://docs.astral.sh/uv/getting-started/installation/).
The CI-tested uv version is **0.11.13**. From the repository:

```sh
uv sync --locked --python 3.12
uv run --locked lynxai doctor
uv sync --locked --python 3.12 --extra ppo
uv run --locked --extra ppo lynxai doctor --profile training
uv run --locked --extra ppo pytest -q
uv run --locked --extra ppo python -m build
uv run --locked --extra ppo python scripts/check_wheel.py
```

The default sync installs core and development/test tools. `--extra ppo` also
installs Stable-Baselines3 for compatibility checks; M0 does not train a policy.
Include `--extra ppo` on subsequent uv commands to retain that extra.
Commit intentional dependency updates with `uv.lock`; ordinary setup and CI use
`--locked` so an outdated lock fails instead of silently resolving new versions.

The uv project selects CPU PyTorch builds on Windows and Linux using an explicit
[PyTorch package index](https://docs.astral.sh/uv/guides/integration/pytorch/).
CUDA is optional and its absence does not fail diagnostics. GPU builds and host
calibration will need a separately validated dependency setup during CS2 integration.
The project metadata requires Python 3.12 or newer; 3.12 is the validated runtime.

`python -m build` creates a source archive and builds the wheel from that archive.
The wheel check installs locked dependencies plus the wheel into a temporary,
isolated environment. It runs a scripted episode and both module/console CLI
entry points from outside the repository. Keep one wheel in `dist` when running it.
Windows and Linux CI run the same checks, without a game or GPU.

## Smoke2D API

```python
from lynxai.environments import ACTION_INDEX, Smoke2DEnv

env = Smoke2DEnv()
obs, info = env.reset(seed=1337)
obs, reward, terminated, truncated, info = env.step(ACTION_INDEX["right"])
env.close()
```

Imports migrated from `environments` to `lynxai.environments`; the old namespace
and abstract base are removed. Reset now returns `(obs, info)` and step returns
`(obs, reward, terminated, truncated, info)`. Call reset before stepping and after
either ending flag. Cleanup is safe to repeat; reset permits reuse after cleanup.

Observations are float32 vectors `[x, y, vertical_velocity, grounded, goal_x - x]`.
The grounded component lies in `[0, 1]`; other components have unbounded declared
spaces. Gymnasium's checker emits expected warnings about those infinite bounds.
Actions retain the indices `left=0`, `right=1`, `jump=2`, `noop=3`.
Negative/out-of-range actions, booleans, and noninteger actions are rejected.

Physics and rewards retain the original behavior: `-0.01` per step, with `+10`
on goal, `-10` on fall, or `-1` on timeout. Goal/fall terminate; timeout truncates.
Goal takes precedence over fall, and both take precedence over the time limit.
The terminal observation is returned unchanged by any automatic reset in the raw
environment; SB3's vector wrapper exposes it through `terminal_observation`.
Rendering and reset options are unsupported.

## Configuration and storage

```sh
uv run --locked lynxai doctor --config configs/smoke2d.yaml --artifact-root ./local-artifacts
```

`load_smoke_config(path)` in `lynxai.utils.smoke_config` returns a typed smoke
configuration. Omitting the path loads the bundled default resource. The editable
example in `configs/smoke2d.yaml` is checked against that resource by tests.
`schema_version: 1` is required; omitted experiment/environment fields use defaults.
Resolution is defaults, YAML, then the explicit artifact-root CLI override.

The loader rejects unknown fields, invalid types/ranges, nonfinite numbers, and
unsupported environment IDs. Platforms must be nonempty, nonoverlapping, above
the fall threshold, and one must cover the full reset interval `[-1, 1]`.
Direct `Smoke2DConfig` and `Platform` construction applies the same physics and
geometry checks. Observations, actions, and rewards are fixed for this task.
`configs/baseline.yaml` and the legacy `load_config` remain a separate CS2 design
scaffold and are not accepted as smoke experiments.

Artifact roots resolve at invocation time: CLI `--artifact-root`, then
`experiment.artifact_root`, then `./artifacts`. Relative paths use the current
working directory, including when YAML lives elsewhere. Doctor reports the
absolute path and performs an actual write probe; imports create no directories.
M0 creates no run manifests, metrics, or checkpoints.

Doctor exits `0` when required checks pass and `1` when a check fails. The core
profile checks Python, packaged config, validated task config, storage, and the
Gymnasium environment contract. Training additionally checks PyTorch and SB3
imports and reports compute availability. Missing optional training dependencies
do not prevent core imports or core checks.

## Game-host readiness

No live CS2 capability has been tested. A successful doctor command establishes
development readiness only. M2 must select and document the game host OS, GPU,
display configuration, supported capture/input mechanisms, outcome detection,
and reliable scenario resets. Local Windows CPU checks and hosted Linux CI do
not establish those capabilities.
