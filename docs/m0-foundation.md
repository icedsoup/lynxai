# M0 foundation completion evidence

Validated October 14, 2026 on the local Windows x86-64 CPU environment.
M0 implementation is complete. Hosted Windows/Linux CI is configured in
`.github/workflows/ci.yml`; its results have not been observed in this session.
Linux execution, CUDA, CS2 host integration, and learning results are unverified.

## Implemented behavior

- Smoke2D ships inside `lynxai.environments` with Gymnasium reset/step/space and
  seeded RNG contracts, terminal/truncation separation, action validation, and cleanup.
- Original physics, action indices, rewards, and outcome precedence are preserved.
- `configs/smoke2d.yaml` and its packaged resource define the fixed smoke task.
  Strict version/type/range/key/platform validation also applies to direct physics config.
  The legacy CS2 scaffold is kept separate.
- Diagnostics use package resources and invocation-relative configurable storage;
  importing modules creates no directories and does not import optional training tools.
- Python 3.12 plus `uv.lock` defines the development/test/PPO environment.
  CPU PyTorch is explicitly selected for Windows/Linux. Build tools are pinned.
- No run records, trainer, checkpoint manager, GUI, or game adapter was introduced.

## Validation

| Command | Observed result |
|---|---|
| `uv sync --locked --python 3.12 --extra ppo` | Locked environment installed successfully |
| `uv run --locked --extra ppo pytest -q` | 95 passed, no skipped tests |
| `uv run --locked --extra ppo python -m build` | Source archive and wheel built; wheel built from source archive |
| `uv run --locked --extra ppo python scripts/check_wheel.py` | Fresh venv outside checkout: imports, packaged YAML, scripted goal, module core doctor, console training doctor all passed |
| `git diff --check` | Passed |

Test coverage includes seeded reset continuation, observation containment, NumPy
integer actions, invalid-action state preservation, cleanup and reset boundaries,
goal/fall precedence at the time limit, Gymnasium and SB3 checkers, SB3 vector
terminal observations and timeout metadata, direct/YAML validation, path precedence,
missing training dependencies, invalid/missing config, and failed storage writes.

The wheel probe removes `PYTHONPATH`/`PYTHONHOME`, uses isolated Python mode for
module checks, and asserts imports originate inside its fresh venv. Dependencies
are exported as `pylock.toml` so the CPU wheel sources and hashes survive export.
Both entry points run with a temporary working directory outside the checkout.

Gymnasium reports expected warnings for the intentionally unbounded continuous
observation components. These are documented space choices, not failed checks.

## Validated versions

- Python 3.12.13; uv 0.11.13
- NumPy 2.5.4; PyYAML 6.0.3
- Gymnasium 1.4.0; Stable-Baselines3 2.9.0
- PyTorch 2.14.1+cpu; CUDA unavailable and optional
- pytest 9.1.1; build 1.6.1
- Isolated build backend: setuptools 84.0.0; wheel 0.48.0

Full transitive versions and distribution hashes are recorded in `uv.lock`.

## Next milestone

Begin M1 with minimal run records and shared random/scripted baseline evaluation.
Only then add PPO training, checkpoints, and frozen evaluation. No learning claim
is implied by M0 compatibility checks or by a scripted controller solving the task.
