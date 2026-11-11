# lynxai

I’m building lynxai to run reinforcement learning experiments in Counter-Strike 2. The first test is a tiny Gymnasium environment called Smoke2D, where an agent has to jump a gap to reach the goal. It gives me a quick way to test training, checkpointing, and evaluation before bringing CS2 into the loop.

The scripted controller cleared the gap on all 100 validation starts. Both PPO pilots scored 0/100. Getting a learned policy across reliably is the next step.

## Try it

With Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/) installed, run these from the repo:

```sh
uv sync --locked --python 3.12 --extra ppo
uv run --locked --extra ppo lynxai evaluate --baseline scripted --config configs/smoke-ppo.yaml --split validation
uv run --locked --extra ppo lynxai smoke-test
```

The smoke test trains briefly, reloads a checkpoint, evaluates it, and resumes training. To run a full pilot and view the training curves:

```sh
uv run --locked --extra ppo lynxai train --config configs/smoke-ppo.yaml --seed 42 --name gap-pilot
uv run --locked --extra ppo tensorboard --logdir artifacts/runs
```

Run `uv run --locked --extra ppo pytest -q` to run the test suite.

## Roadmap

- [x] Build the Smoke2D environment and reference controllers.
- [x] Add PPO training, checkpoint resume, and repeatable evaluation.
- [ ] Train a policy that reliably clears the gap on held-out starts.
- [ ] Run a repeatable training and evaluation loop in a controlled CS2 scenario.
- [ ] Compare learning from demonstrations with training from scratch.
