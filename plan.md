# Project Bible: Teaching an AI to Play CS2

> A long-term portfolio/research project for building, training, evaluating, and visualizing a reinforcement-learning agent for a Counter-Strike-style FPS, with CS2 as the real target and a tiny 2D game used only as a smoke test.
>
> **Core philosophy:** get to the real CS2 experimentation quickly, keep the first agent extremely small, make the training stack reproducible, and grow the project through modular experiments instead of replacing the whole system every time a new idea appears.

---

## 0. The one-page vision

The project starts simple:

**Observation**
- A low-resolution image or short sequence of images from a controlled CS2 test environment.

**Actions**
- `W`
- `A`
- `S`
- `D`
- `LMB`
- optionally `RMB` once the basics work

**Reward**
- Positive reward for kills.
- Negative reward for dying.
- Later: round outcome, damage, survival, objective progress, prediction accuracy, etc.

**Learning**
- Start with behavioral cloning from human demonstrations.
- Then fine-tune/improve with reinforcement learning.
- PPO is a strong first RL algorithm to investigate; it can initially be used through an existing implementation, then implemented yourself later if desired.

**Engineering**
- Python is the glue/orchestration language.
- PyTorch is the core neural-network/ML framework.
- Gymnasium-style environment interfaces keep the agent independent from the game.
- OpenCV/Python image processing can handle perception preprocessing.
- TensorBoard or Weights & Biases can handle experiment metrics.
- A custom GUI sits on top of the same backend used by the CLI.

**Long-term capabilities**
- Automatic highlights/clips.
- Full checkpointing and restore/time travel.
- Training/evaluation toggles.
- Overnight automated training in a controlled environment.
- Spectator/demonstration learning.
- Player-behavior prediction.
- Skill-specific curriculum training.
- Scenario generation.
- Memory/temporal models.
- Self-play between agents in a controlled environment.
- Human-vs-AI behavioral analysis.

---

# 1. Project goals

## Primary goal

Build a visually understandable, reproducible AI system that can progressively learn useful FPS behaviors from observations and rewards, while documenting *why* each design choice works or fails.

The strongest portfolio framing is not:

> â€œI made a CS2 bot.â€

It is:

> **â€œI designed, trained, evaluated, and iteratively improved a reinforcement-learning agent for an FPS, studying the effects of observation design, action-space design, reward functions, imitation learning, temporal modeling, and prediction.â€**

That framing makes the project feel like a serious ML systems/research project rather than a one-off game script.

## Secondary goals

1. Make experiments extremely easy to run.
2. Make every trained agent reproducible.
3. Make it obvious how the agent improved over time.
4. Make failure easy to inspect rather than mysterious.
5. Build a clean GUI that makes the training process fun to watch.
6. Produce automatic videos, graphs, and experiment summaries for the portfolio.

---

# 2. Testing strategy: controlled first, online-adjacent later

The project should be designed around **controlled experimentation first**, but the architecture should leave room for later online-oriented testing that does not put an autonomous agent into public matchmaking or attempt to bypass anti-cheat.

Think of testing as four tiers:

```text
Tier 1  â†’  2D smoke test
Tier 2  â†’  controlled CS2 / practice / custom environment
Tier 3  â†’  replays, demos, spectating, and human-gameplay analysis
Tier 4  â†’  sanctioned/private/experimental online-style testing with humans in the loop, when permitted
```

The important distinction is that the project should **never make anti-cheat evasion a design requirement**. Do not optimize the agent to hide autonomous behavior, bypass detection, or gain an unfair advantage in public matchmaking. Human-like behavior can still be studied as an ML research topic, and real human games can still be useful as observational/training data where collection and use are permitted.

This boundary is also excellent engineering practice. Controlled environments let you reset scenarios, freeze variables, generate repeatable tests, collect precise state information, and compare checkpoints fairly. Online-style testing can then be treated as a later validation layer rather than the foundation of training.

Valve currently provides Source 2 tools for community map makers and artists, and its official resources include Counter-Strike Workshop tooling. Valveâ€™s current CS2 materials also continue to expose game/map scripting updates, making controlled workshop/practice-oriented environments a natural place to experiment. îˆ€citeîˆ‚turn748072search1îˆ‚turn748072search2îˆ‚turn748072search0îˆ

Valveâ€™s developer documentation also describes practice/LAN-oriented commands and demo playback capabilities that can be useful when designing isolated evaluation and observation workflows. îˆ€citeîˆ‚turn748072search9îˆ

**Project rule:** build and train in a controlled environment; treat online-oriented testing as an optional, carefully bounded validation tier, and keep a manual/human-in-the-loop path available at all times.

---

# 3. The architecture to aim for

Keep the system modular from the beginning.

```text
                         â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
                         â”‚    Control GUI       â”‚
                         â”‚  start / stop /     â”‚
                         â”‚  checkpoints / clips â”‚
                         â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                                    â”‚
                         â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
                         â”‚   Experiment Manager â”‚
                         â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                                    â”‚
        â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
        â”‚                           â”‚                            â”‚
        â–¼                           â–¼                            â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”          â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”          â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ Observation   â”‚          â”‚ Agent / Policy  â”‚          â”‚ Reward / Events â”‚
â”‚ Adapter        â”‚â”€â”€â”€â”€â”€â”€â”€â”€â”€â–¶â”‚ PyTorch         â”‚â—€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”‚ Adapter         â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”˜          â””â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”˜          â””â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
        â”‚                            â”‚                           â”‚
        â–¼                            â–¼                           â–¼
   screen / state                 action                    reward
        â”‚                            â”‚                           â”‚
        â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                                     â–¼
                              â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
                              â”‚ Replay /      â”‚
                              â”‚ Rollout Store â”‚
                              â””â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”˜
                                      â–¼
                              â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
                              â”‚ Trainer        â”‚
                              â”‚ PPO / future  â”‚
                              â”‚ algorithms    â”‚
                              â””â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”˜
                                      â–¼
                               checkpoint.pt
```

The single most important architectural decision is this:

> **The neural network should not know what CS2 is.**

It should know how to consume an observation and output an action.

The CS2-specific details should live in adapters.

For example:

```text
Agent
 â”œâ”€â”€ ObservationAdapter
 â”œâ”€â”€ ActionAdapter
 â”œâ”€â”€ RewardAdapter
 â””â”€â”€ EnvironmentAdapter
```

That means later you can swap:

```text
CS2
â†“
2D smoke test
â†“
custom FPS sandbox
â†“
another controlled environment
```

without rewriting the brain.

---

# 4. Recommended tech stack

## Core

- Python
- PyTorch
- Gymnasium-style environment API
- PPO initially
- OpenCV for image processing where useful
- NumPy
- Git + GitHub

## RL

Start with an established PPO implementation so you can focus on the environment and experiments.

Later, once the system works:

> implement PPO yourself in PyTorch.

That gives the project a nice learning progression:

1. Use PPO.
2. Understand PPO.
3. Implement PPO.
4. Compare your implementation with the established implementation.

## Experiment tracking

Use one of:

- TensorBoard for a lightweight local setup.
- Weights & Biases for richer experiment management and visual comparison.

The specific tool matters less than the discipline:

> **Every experiment should automatically log configuration, metrics, checkpoint names, and results.**

## GUI

Pick a Python GUI technology that you are comfortable maintaining. Good candidates include:

- PySide6/Qt for a polished desktop application.
- Tkinter if you want minimal dependencies.

For this project, **PySide6 is the most attractive long-term choice** because you want a clean dashboard with live charts, tabs, controls, logs, video previews, checkpoint browsing, and experiment comparison.

## CLI

Use a single command-line entry point, for example:

```bash
python -m lynxai
```

or, once the package is installed:

```bash
lynxai
```

The CLI should be clean and useful, not a wall of debug output.

---

# 5. Project folder structure

Use one project root and never scatter outputs across random directories.

Recommended:

```text
cs2-ai/
â”‚
â”œâ”€â”€ README.md
â”œâ”€â”€ pyproject.toml
â”œâ”€â”€ uv.lock                    # if using uv
â”œâ”€â”€ .gitignore
â”œâ”€â”€ LICENSE
â”œâ”€â”€ docs/
â”‚   â”œâ”€â”€ project_bible.md
â”‚   â”œâ”€â”€ architecture.md
â”‚   â”œâ”€â”€ experiments.md
â”‚   â””â”€â”€ safety.md
â”‚
â”œâ”€â”€ src/
â”‚   â””â”€â”€ lynxai/
â”‚       â”œâ”€â”€ __init__.py
â”‚       â”œâ”€â”€ __main__.py
â”‚       â”‚
â”‚       â”œâ”€â”€ app/
â”‚       â”‚   â”œâ”€â”€ cli.py
â”‚       â”‚   â”œâ”€â”€ gui.py
â”‚       â”‚   â””â”€â”€ commands.py
â”‚       â”‚
â”‚       â”œâ”€â”€ agents/
â”‚       â”‚   â”œâ”€â”€ base.py
â”‚       â”‚   â”œâ”€â”€ policy.py
â”‚       â”‚   â”œâ”€â”€ vision_encoder.py
â”‚       â”‚   â”œâ”€â”€ memory.py
â”‚       â”‚   â””â”€â”€ prediction.py
â”‚       â”‚
â”‚       â”œâ”€â”€ environments/
â”‚       â”‚   â”œâ”€â”€ base.py
â”‚       â”‚   â”œâ”€â”€ smoke2d.py
â”‚       â”‚   â””â”€â”€ cs2_controlled.py
â”‚       â”‚
â”‚       â”œâ”€â”€ adapters/
â”‚       â”‚   â”œâ”€â”€ observation.py
â”‚       â”‚   â”œâ”€â”€ action.py
â”‚       â”‚   â”œâ”€â”€ reward.py
â”‚       â”‚   â””â”€â”€ events.py
â”‚       â”‚
â”‚       â”œâ”€â”€ training/
â”‚       â”‚   â”œâ”€â”€ trainer.py
â”‚       â”‚   â”œâ”€â”€ rollout.py
â”‚       â”‚   â”œâ”€â”€ evaluation.py
â”‚       â”‚   â”œâ”€â”€ curriculum.py
â”‚       â”‚   â””â”€â”€ self_play.py
â”‚       â”‚
â”‚       â”œâ”€â”€ recording/
â”‚       â”‚   â”œâ”€â”€ recorder.py
â”‚       â”‚   â”œâ”€â”€ highlight_detector.py
â”‚       â”‚   â””â”€â”€ clipper.py
â”‚       â”‚
â”‚       â”œâ”€â”€ checkpoints/
â”‚       â”‚   â”œâ”€â”€ manager.py
â”‚       â”‚   â””â”€â”€ manifest.py
â”‚       â”‚
â”‚       â”œâ”€â”€ data/
â”‚       â”‚   â”œâ”€â”€ demonstrations.py
â”‚       â”‚   â”œâ”€â”€ datasets.py
â”‚       â”‚   â””â”€â”€ schemas.py
â”‚       â”‚
â”‚       â”œâ”€â”€ metrics/
â”‚       â”‚   â”œâ”€â”€ logger.py
â”‚       â”‚   â”œâ”€â”€ statistics.py
â”‚       â”‚   â””â”€â”€ reports.py
â”‚       â”‚
â”‚       â””â”€â”€ utils/
â”‚           â”œâ”€â”€ config.py
â”‚           â”œâ”€â”€ paths.py
â”‚           â””â”€â”€ time.py
â”‚
â”œâ”€â”€ configs/
â”‚   â”œâ”€â”€ smoke_test.yaml
â”‚   â”œâ”€â”€ baseline.yaml
â”‚   â”œâ”€â”€ imitation.yaml
â”‚   â”œâ”€â”€ rl.yaml
â”‚   â””â”€â”€ prediction.yaml
â”‚
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ raw/
â”‚   â”œâ”€â”€ demonstrations/
â”‚   â”œâ”€â”€ processed/
â”‚   â””â”€â”€ scenarios/
â”‚
â”œâ”€â”€ runs/
â”‚   â”œâ”€â”€ active/
â”‚   â””â”€â”€ archive/
â”‚
â”œâ”€â”€ checkpoints/
â”‚   â”œâ”€â”€ best/
â”‚   â””â”€â”€ archive/
â”‚
â”œâ”€â”€ clips/
â”‚   â”œâ”€â”€ highlights/
â”‚   â””â”€â”€ manual/
â”‚
â”œâ”€â”€ logs/
â”‚
â”œâ”€â”€ reports/
â”‚
â””â”€â”€ tests/
    â”œâ”€â”€ test_agent.py
    â”œâ”€â”€ test_environment.py
    â”œâ”€â”€ test_reward.py
    â”œâ”€â”€ test_checkpoint.py
    â””â”€â”€ test_paths.py
```

## Why this layout is good

`src/` = code.

`configs/` = experiment definitions.

`data/` = datasets and raw observations.

`runs/` = each training/evaluation run.

`checkpoints/` = model snapshots.

`clips/` = automatically generated videos.

`reports/` = charts and summaries.

`tests/` = protection against accidental breakage.

The project becomes much easier to reason about when code, data, experiments, and generated media never get mixed together.

---

# 6. Make the project path configurable

Do not hard-code things like:

```text
C:\Users\you\Desktop\...
```

Instead, have one project-path/config system.

For example:

```text
PROJECT_ROOT
DATA_DIR
RUNS_DIR
CHECKPOINT_DIR
CLIPS_DIR
LOG_DIR
REPORT_DIR
```

Everything should be derived from the root.

Example conceptual API:

```python
paths.runs.run(run_id)
paths.checkpoints.run(run_id)
paths.clips.run(run_id)
```

That prevents path bugs when you move the project to another machine.

---

# 7. The 2D game: use it only as a smoke test

You explicitly do **not** want to spend significant time here.

That is the correct decision.

Use AI assistance to make a tiny 2D game whose entire purpose is to validate that:

```text
observation
â†’ model
â†’ action
â†’ environment
â†’ reward
â†’ training update
```

actually works.

The game can be extremely simple:

```text
player â†’ platforms â†’ target
```

Required abilities:

- Move left/right.
- Jump.
- Avoid falling.
- Reach a goal.

That's it.

## What the smoke test must prove

Before touching the real CS2 environment, demonstrate that:

1. The environment exposes observations correctly.
2. The agent receives actions correctly.
3. Rewards are computed correctly.
4. Episodes reset correctly.
5. The trainer updates the model.
6. Checkpoints save and load.
7. Evaluation mode freezes the model.
8. The GUI can start/stop training.
9. Metrics are logged.
10. The CLI can launch a run.

Do **not** spend weeks polishing the 2D game.

Once these ten things work, move to the actual CS2 experiment.

---

# 8. First CS2 milestone: build the control loop, not the AI

Your first real milestone is not:

> â€œAI plays CS2.â€

It is:

> **â€œMy environment adapter can reliably observe, act, reset, and report outcomes in a controlled CS2 test scenario.â€**

Think of it as a game-environment API.

Conceptually:

```python
observation = env.reset()

while not done:
    action = agent.act(observation)
    observation, reward, done, info = env.step(action)
```

This abstraction is enormously important.

Your agent should never care whether `env.step()` is backed by:

- the 2D smoke test,
- a controlled CS2 scenario,
- or a future custom environment.

---

# 9. First real agent: ridiculously small

Do not start with grenades.

Do not start with the bomb.

Do not start with prediction.

Do not start with a giant model.

Start with:

### Observation

A low-resolution frame, potentially downsampled aggressively.

For example, something in the rough neighborhood of:

```text
160 Ã— 90
```

is worth investigating as an initial experiment.

The exact resolution should be treated as a parameter, not a permanent decision.

### Actions

```text
W
A
S
D
LMB
```

Then optionally:

```text
RMB
```

The initial action space should stay tiny.

---

# 10. Why the first model does not need to be huge

A common misconception is:

> â€œTo understand a 3D FPS, I need an enormous AI model.â€

Not necessarily.

You can build a specialized model whose task is much narrower than general visual intelligence.

The conceptual pipeline is:

```text
low-res frame
      â†“
small CNN
      â†“
visual feature vector
      â†“
policy/value network
      â†“
action probabilities
```

Example:

```text
W       0.09
A       0.04
S       0.02
D       0.58
LMB     0.24
RMB     0.03
```

The model does not need to describe the image in English.

It only needs to learn a representation useful for action selection.

The hard problems are likely to be:

- learning useful behavior,
- gathering enough experience,
- reward design,
- temporal reasoning,
- and stable training.

---

# 11. Screen processing and latency

Latency matters more for this project than raw parameter count.

The loop is roughly:

```text
capture
  â†“
preprocess
  â†“
model inference
  â†“
action decision
  â†“
environment input
```

Keep the system measurable.

Log timings such as:

```text
capture_ms
preprocess_ms
inference_ms
action_ms
total_loop_ms
```

Do not guess where the bottleneck is.

Measure it.

## Important idea: action frequency is an experiment

The agent does not necessarily have to make a new decision every possible rendered frame.

Test different control rates and measure:

- reward,
- stability,
- aim performance,
- computational cost.

For example, compare:

```text
10 decisions/sec
20 decisions/sec
30 decisions/sec
60 decisions/sec
```

These are experiment values, not requirements.

The interesting question is:

> **What decision frequency gives the best performance per unit of compute?**

---

# 12. One computer or two?

Start with one.

A second computer should be an optimization, not a prerequisite.

Potential architecture later:

```text
Computer A
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
controlled environment
screen/state collection

      â”‚
      â”‚ local network
      â–¼

Computer B
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
GPU inference/training
```

Benefits:

- isolates GPU workloads,
- leaves the game machine more resources,
- opens the door to multiple training workers.

Costs:

- network latency,
- synchronization complexity,
- more moving pieces,
- harder debugging.

So the rule is:

> **Make the single-machine version correct before distributing it.**

---

# 13. Reward design: start simple, then improve

Your proposed first reward function is excellent for a first experiment:

```text
kill   +10
 death -10
```

Use a small number of reward signals initially so you know what caused learning to change.

## The reward-hacking problem

The agent might discover an unintended strategy.

For example:

> repeatedly perform some useless action that accidentally correlates with reward.

That is not necessarily a failure of the project. It is a valuable finding.

Document it.

Then fix the reward function and compare the new run against the old run.

## Later reward terms

Possible additions:

```text
kill                   +10
round win              +20
survival                +small positive signal
useful damage           +small positive signal
death                  -10
round loss              -small negative signal
objective progress     +small positive signal
objective completion   +large positive signal
```

Avoid stuffing every conceivable reward into the system at once.

Add one change at a time.

---

# 14. Reward shaping experiments

This should eventually become a major research section of the project.

Compare:

### Experiment A

```text
reward = kills - deaths
```

### Experiment B

```text
reward = kills + survival - deaths
```

### Experiment C

```text
reward = kills + damage + survival - deaths
```

### Experiment D

```text
reward = kills + objective_progress + round_result + survival - deaths
```

Then compare learning curves.

The question becomes:

> **How does the reward definition change what the agent actually learns?**

That is an excellent portfolio question.

---

# 15. Human demonstrations: the first big upgrade

You had an excellent idea: record yourself playing and use that data.

Each demonstration should conceptually contain:

```text
frame / frame sequence
+
action
+
timestamp
+
optional game-event metadata
```

For example:

```text
frame_001 â†’ W
frame_002 â†’ W
frame_003 â†’ W + LMB
frame_004 â†’ D
frame_005 â†’ D + LMB
```

This creates a behavioral dataset.

## Behavioral cloning

Train a model to predict human actions from observations.

That produces an initial policy that is not random.

Then use RL to improve it.

```text
human demonstrations
        â†“
behavioral cloning
        â†“
initial policy
        â†“
reinforcement learning
        â†“
improved policy
```

This is one of the strongest ideas in the entire project because it addresses a fundamental problem:

> random exploration in a complicated FPS can be extremely inefficient.

Human demonstrations provide a useful starting point.

---

# 16. Spectating as additional experience

Another idea from the project discussion: when the agent is dead and can spectate a player or another permitted training agent in the controlled environment, treat that as a separate kind of data.

Conceptually:

```text
agent dies
   â†“
spectator / demonstration phase
   â†“
observe another player/agent
   â†“
collect frames + actions where available
   â†“
add to demonstration dataset
```

The key is to distinguish the sources.

Tag data as:

```text
source = human_self
source = human_other
source = scripted_agent
source = learned_agent
```

Then you can measure whether mixing sources helps.

Potential research comparison:

```text
RL only
vs.
Human imitation + RL
vs.
Human + spectator demonstrations + RL
```

That is a genuinely interesting study.

---

# 17. Human-like behavior: do it as measurement, not anti-cheat evasion

A later project goal can be:

> **â€œCan the learned agent produce movement statistics similar to humans?â€**

Collect human behavior statistics such as:

- reaction times,
- mouse velocity,
- mouse acceleration,
- correction frequency,
- pause durations,
- movement-direction changes,
- action burst lengths,
- time between shots.

Then compare distributions:

```text
human distribution
vs.
AI distribution
```

You can define a research objective around reducing the statistical difference between the two.

That is a legitimate behavioral-modeling question.

Do not turn human-like behavior into an anti-cheat-evasion objective. The interesting research question is whether the learned policy reproduces human behavioral statistics, not whether it can disguise autonomous play.

---

# 18. Automatic highlights / clips

This should be built surprisingly early because it makes training more fun and makes failures easier to investigate.

## Rolling buffer

Continuously retain a short rolling video buffer.

When an event occurs, save:

```text
N seconds before event
+
event
+
N seconds after event
```

Possible triggers:

- kill,
- multi-kill,
- round win,
- unusual reward spike,
- prediction success,
- new personal record,
- objective completion,
- unusually long survival,
- surprising behavior.

## Pair every clip with metadata

For each highlight save something like:

```json
{
  "run_id": "2026-09-01_001",
  "checkpoint": 47,
  "episode": 18241,
  "event": "kill",
  "reward": 10.0,
  "prediction_confidence": 0.72,
  "model_version": "agent_0047"
}
```

This allows the GUI to say:

> â€œCheckpoint 47 â€” prediction success â€” watch clip.â€

That is fantastic for debugging and presentation.

---

# 19. Checkpoints: save everything important

Do not only save:

```text
model.pt
```

A serious checkpoint should preserve enough information to reproduce the experiment state.

Recommended conceptual structure:

```text
checkpoint_0047/
â”œâ”€â”€ model.pt
â”œâ”€â”€ optimizer.pt
â”œâ”€â”€ scheduler.pt
â”œâ”€â”€ config.yaml
â”œâ”€â”€ training_state.json
â”œâ”€â”€ metrics.json
â”œâ”€â”€ rng_state.bin
â”œâ”€â”€ environment_info.json
â””â”€â”€ manifest.json
```

The exact files can change by framework, but the idea stays the same.

## Checkpoint metadata should include

- model architecture,
- model version,
- observation size,
- action-space definition,
- reward definition,
- environment version,
- code version/commit hash,
- training step,
- episode count,
- cumulative reward,
- evaluation metrics,
- timestamp.

## Always have

### Best checkpoint

The best performing model by a chosen evaluation metric.

### Latest checkpoint

Most recent model.

### Archived checkpoints

Older snapshots preserved for comparison.

---

# 20. â€œTime travelâ€ checkpoint comparison

This was one of the best ideas from the discussion.

Build a way to run the **same scenario** with multiple checkpoints.

For example:

```text
Scenario: enemy appears at doorway

Checkpoint 01 â†’ misses / does not react
Checkpoint 10 â†’ notices enemy
Checkpoint 20 â†’ moves toward enemy
Checkpoint 30 â†’ aims
Checkpoint 40 â†’ shoots
Checkpoint 50 â†’ anticipates peek
```

Now your project can literally show:

> **â€œThis is what the same agent looked like at different stages of training.â€**

That is far more convincing than a single final score.

---

# 21. Evaluation must be separate from training

Create two explicit modes.

## Training mode

```text
play
â†’ collect experience
â†’ update weights
â†’ save checkpoint
```

## Evaluation mode

```text
play
â†’ collect metrics
â†’ DO NOT update weights
```

This distinction is crucial.

Otherwise you cannot tell whether a model improved because:

- it was already good,
- it happened to get an easy scenario,
- or it actually learned.

## Useful automation

Support a workflow like:

```text
Train for N episodes
        â†“
freeze model
        â†“
run evaluation suite
        â†“
log metrics
        â†“
save checkpoint
        â†“
select best/latest
```

---

# 22. Overnight training

Once the pipeline is reliable, automated long-running training becomes one of the project's biggest strengths.

Example workflow:

```text
22:00
start experiment

â†“

collect episodes
collect metrics
save checkpoints
save clips

â†“

02:00
run evaluation

â†“

03:00
train/update model

â†“

06:00
final evaluation

â†“

08:00
GUI shows summary
```

However, never rely on â€œit ran all nightâ€ as proof that training was successful.

Add automatic health checks.

For example:

```text
If reward has not improved
for a configured number of episodes:

pause
save checkpoint
mark run as stalled
```

This prevents wasting many hours on a broken experiment.

---

# 23. Train while playing: online learning toggle

Eventually add a simple switch:

```text
[ Training ON ]
```

or:

```text
[ Evaluation / Frozen ]
```

In training mode:

```text
play â†’ update â†’ continue
```

In evaluation mode:

```text
play â†’ log â†’ continue
```

Also add:

```text
Train N episodes â†’ Evaluate N episodes â†’ repeat
```

That creates a clean learning/evaluation cycle.

---

# 24. Adaptive curriculum

Once the agent has enough skills, stop giving it only random training.

Measure weaknesses.

Example:

```text
Aim accuracy            83%
Movement                91%
Close-range combat      47%
Long-range combat       31%
Grenade usage           12%
Prediction              55%
```

Then automatically generate more situations matching the weakest skills.

That gives you:

> **AI that practices what it is bad at.**

This is a natural extension of curriculum learning.

---

# 25. Scenario generation

Build a library of controlled situations.

Example:

```yaml
scenario:
  name: left_peek
  player_spawn: A
  enemy_spawn: B
  behavior: aggressive_peek
  objective: eliminate_target
```

Then generate variations.

Possible dimensions:

- enemy location,
- starting position,
- distance,
- direction of movement,
- cover,
- timing,
- number of targets,
- objective state.

This lets you create thousands of controlled test cases.

It also makes evaluation repeatable.

---

# 26. Player prediction

This is a major long-term upgrade and deserves its own subsystem.

Instead of only asking:

> â€œWhat should I do now?â€

ask:

> **â€œWhat will the opponent probably do next?â€**

Example sequence:

```text
t-3  enemy moves left
t-2  enemy moves left
t-1  enemy approaches corner
t0   enemy reaches corner
```

Prediction head:

```text
continue left     72%
stop              18%
reverse            10%
```

Then the policy receives both current perception and predicted behavior.

Conceptually:

```text
                observation
                     â†“
               vision encoder
                     â†“
             â”Œâ”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”
             â†“                â†“
        behavior head     action head
             â†“                â†“
       prediction         decision
             â””â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                     â†“
                   action
```

---

# 27. Train prediction separately first

Do not immediately entangle prediction with the main RL policy.

First build a standalone prediction experiment:

```text
sequence of observations
        â†“
predict next movement
```

Measure:

- next-direction accuracy,
- next-position error,
- prediction horizon,
- calibration of confidence.

Once that works, feed its representation into the decision policy.

This lets you answer:

> â€œDoes prediction actually improve decision-making?â€

---

# 28. Temporal reasoning and memory

A single frame may not contain enough information.

A player might have just disappeared behind cover.

The agent may need to remember:

> â€œI saw that player move behind that wall two seconds ago.â€

So later introduce temporal models.

Potential approaches:

- frame stacking,
- GRU/LSTM,
- temporal convolution,
- small transformer.

Start with the simplest one.

A useful progression is:

```text
single frame
â†’ stacked frames
â†’ recurrent state
â†’ transformer-style sequence model
```

Then compare performance and compute cost.

---

# 29. Vision vs structured state

This is another excellent experiment.

### Agent A: image only

```text
screen â†’ CNN â†’ policy
```

### Agent B: structured state

```text
player position
enemy position
velocity
health
ammo
objective
â†’ policy
```

### Agent C: hybrid

```text
screen
+
structured signals
â†’ policy
```

Then compare:

- training speed,
- final performance,
- generalization,
- model size,
- inference latency.

The question becomes:

> **How much does an explicit world representation help an agent learn?**

This can become a strong section of the research report.

---

# 30. Action-space progression

Use a deliberate expansion strategy.

## Version 1

```text
W A S D
```

## Version 2

```text
W A S D + LMB
```

## Version 3

```text
W A S D + LMB + RMB
```

## Version 4

Add controlled aiming/mouse actions.

## Version 5

Add additional mechanics relevant to the controlled scenario.

## Version 6

Add objective actions.

The important idea is:

> **Do not increase the action space until the current action space is understood.**

An expanded action space changes the learning problem dramatically.

---

# 31. Advanced objectives

Later, in controlled scenarios, introduce objective-level behavior:

- attack objective,
- defend objective,
- plant objective,
- defuse objective,
- save resources,
- coordinate with a teammate.

The agent then progresses from:

```text
shooting
```

to:

```text
gameplay
```

That is the real long-term goal.

---

# 32. Modular skill system

A future architecture can separate skills:

```text
                 Vision
                   â†“
              World model
                   â†“
       â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
       â†“           â†“            â†“
    Combat      Movement    Prediction
       â”‚           â”‚            â”‚
       â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                   â†“
                 Policy
                   â†“
                 Action
```

This may not be needed early.

Treat it as an advanced research direction, not a starting requirement.

---

# 33. AI vs AI / self-play

Once the fundamentals work, create controlled agents with different styles.

Example:

```text
Agent A = aggressive
Agent B = defensive
```

Then measure:

- win rate,
- survival,
- decision diversity,
- adaptation.

Eventually you can explore self-play:

```text
Agent A
   â†•
Agent B
   â†•
Agent C
```

This creates a pathway toward studying whether agents develop stronger behavior when trained against changing opponents.

Again: keep this inside a controlled environment.

---

# 34. Model uncertainty and action probabilities

Do not only output the selected action.

Log the model's action distribution.

Example:

```text
W       0.08
A       0.03
S       0.01
D       0.72
LMB     0.14
RMB     0.02
```

This is useful for:

- debugging,
- visualizing decisions,
- detecting uncertainty,
- studying exploration,
- producing portfolio visuals.

The GUI can show the distribution live.

---

# 35. Build a beautiful GUI

The GUI should be one of the project's headline features.

Think of it as a lab control center, not just a settings window.

## Main dashboard

Show:

```text
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ lynxai lab                                         â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚                                                    â”‚
â”‚  LIVE VIEW                 CURRENT RUN             â”‚
â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”      Checkpoint: 047         â”‚
â”‚  â”‚                 â”‚      Episode: 18,241         â”‚
â”‚  â”‚    GAME VIEW    â”‚      Reward: +137            â”‚
â”‚  â”‚                 â”‚      Kills: 17               â”‚
â”‚  â”‚                 â”‚      Deaths: 9               â”‚
â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜      K/D: 1.89              â”‚
â”‚                                                    â”‚
â”‚  ACTION PROBS             TRAINING                â”‚
â”‚  W   â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆ              [ TRAINING ON ]         â”‚
â”‚  A   â–ˆâ–ˆ                   [ PAUSE ]              â”‚
â”‚  S   â–ˆ                    [ STOP ]               â”‚
â”‚  D   â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆ            [ EVALUATE ]            â”‚
â”‚  LMB â–ˆâ–ˆâ–ˆâ–ˆ                                          â”‚
â”‚  RMB â–ˆ                                             â”‚
â”‚                                                    â”‚
â”‚  [ CHECKPOINTS ] [ HIGHLIGHTS ] [ EXPERIMENTS ]  â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

## Recommended tabs

### Live

- live game view,
- current action,
- action probabilities,
- reward events,
- current episode.

### Training

- training controls,
- learning rate,
- batch/rollout settings,
- steps,
- loss curves,
- reward curves.

### Checkpoints

- list checkpoints,
- view metadata,
- restore,
- compare.

### Highlights

- automatically generated clips,
- filter by event,
- open clip,
- inspect metadata.

### Experiments

- compare runs,
- compare models,
- compare reward functions.

### Evaluation

- launch repeatable test suites,
- compare results.

### Settings

- project paths,
- capture settings,
- logging settings,
- compute settings.

---

# 36. CLI design

The CLI should be short and predictable.

Example command set:

```bash
lynxai doctor
lynxai smoke-test
lynxai train --config configs/baseline.yaml
lynxai evaluate --checkpoint checkpoints/best/...
lynxai inspect-run RUN_ID
lynxai compare RUN_A RUN_B
lynxai list-checkpoints
lynxai restore CHECKPOINT_ID
lynxai generate-report RUN_ID
lynxai clips --run RUN_ID
```

## `doctor`

This should check:

- Python version,
- PyTorch availability,
- GPU visibility,
- required dependencies,
- project directories,
- write permissions,
- model/config compatibility.

Output should feel professional:

```text
lynxai doctor
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Python            OK
PyTorch           OK
GPU               OK
Project paths     OK
Config files      OK
Smoke environment OK

Ready.
```

---

# 37. Configuration files

Do not bury experiment settings inside Python code.

Use config files.

Example conceptual config:

```yaml
experiment:
  name: baseline_kill_death
  seed: 42

observation:
  width: 160
  height: 90
  grayscale: false
  frame_stack: 1

actions:
  - W
  - A
  - S
  - D
  - LMB

reward:
  kill: 10
  death: -10

training:
  algorithm: ppo
  total_steps: 1000000
  learning_rate: 0.0003

logging:
  save_every_steps: 10000
  clip_highlights: true
```

Then an experiment is defined by a config plus a code version.

---

# 38. Reproducibility

Every run should have:

```text
run_id
code_commit
config_hash
random_seed
model_version
environment_version
```

This means you can answer:

> â€œWhat exactly produced this checkpoint?â€

without guessing.

---

# 39. Experiment naming

Use names that are human-readable.

Example:

```text
2026-09-01_baseline_kd_smallcnn
2026-09-02_bc_wasd_lmb
2026-09-03_bc_plus_ppo
2026-09-05_reward_damage
2026-09-07_temporal_stack4
```

Avoid names like:

```text
run_final_final2_realfinal_v7
```

Use machine-readable IDs plus a human-readable experiment name.

---

# 40. Metrics dashboard

Track multiple levels of metrics.

## Raw gameplay metrics

- kills,
- deaths,
- K/D,
- damage,
- survival time,
- round results.

## Agent metrics

- policy loss,
- value loss,
- entropy,
- action distribution,
- gradient statistics,
- inference latency.

## Research metrics

- imitation accuracy,
- prediction accuracy,
- generalization to unseen scenarios,
- performance vs checkpoint,
- reward efficiency.

## Systems metrics

- FPS,
- GPU utilization,
- CPU utilization,
- memory usage,
- capture latency,
- inference latency,
- end-to-end loop latency.

Do not optimize blindly.

Measure first.

---

# 41. Build an evaluation suite

A final score is not enough.

Create standardized tests.

Example:

```text
Evaluation Suite A
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Movement          10 scenarios
Aim               10 scenarios
Target reaction   10 scenarios
Combat            20 scenarios
Prediction        20 scenarios
```

Each model gets the same scenarios.

Then the GUI can produce:

```text
Checkpoint 01
Movement      22%
Combat         3%
Prediction     â€”

Checkpoint 20
Movement      81%
Combat        47%
Prediction     â€”

Checkpoint 47
Movement      92%
Combat        73%
Prediction    65%
```

This is how you demonstrate genuine improvement.

---

# 42. â€œCool eventâ€ detector

The highlight system can grow into a general anomaly detector.

Define an event score:

```text
cool_score = weighted combination of:
    reward spike
    rare event
    unusual accuracy
    prediction success
    new record
```

Then only clip the strongest moments.

You can also store â€œinteresting but not highlight-worthyâ€ events for analysis.

This separates:

```text
highlights/
analysis_events/
```

---

# 43. Automatic report generation

Every experiment should optionally generate:

```text
reports/
â””â”€â”€ 2026-09-03_bc_plus_ppo/
    â”œâ”€â”€ summary.html
    â”œâ”€â”€ reward_curve.png
    â”œâ”€â”€ evaluation.png
    â”œâ”€â”€ latency.png
    â”œâ”€â”€ checkpoint_comparison.png
    â””â”€â”€ metrics.csv
```

The report should include:

- experiment goal,
- configuration,
- model,
- reward function,
- training duration,
- metrics,
- charts,
- best checkpoint,
- notes,
- highlight clips.

That turns every experiment into a reusable research artifact.

---

# 44. Portfolio presentation

The GitHub README should tell a story.

## Suggested structure

### Hero section

> **Learning to Play an FPS with Reinforcement Learning**

A one-paragraph explanation and a GIF/video.

### System diagram

Show:

```text
observation â†’ vision â†’ policy â†’ action â†’ environment â†’ reward
```

### Training progression

Show clips from:

```text
Episode 1
Episode 500
Episode 5,000
Episode 20,000
```

### Experiments

Show tables and graphs.

### Human imitation

Show how demonstrations affected learning.

### Prediction

Show successful future-behavior predictions.

### Checkpoints

Show the same scenario across checkpoints.

### Engineering

Explain:

- architecture,
- latency,
- data pipeline,
- checkpointing,
- GUI,
- reproducibility.

### Lessons learned

Be honest about failure.

Examples:

> â€œThe first reward function produced a stationary strategy.â€

> â€œIncreasing image resolution improved perception but made training less efficient.â€

> â€œBehavioral cloning greatly improved initial exploration.â€

That kind of writing makes the project feel credible.

---

# 45. Suggested development roadmap

The roadmap should be aggressive about reaching the real target.

## Phase 0 â€” Setup

Build:

- repo,
- Python package,
- config system,
- path manager,
- CLI skeleton,
- GUI skeleton,
- logging.

Deliverable:

```text
lynxai doctor
```

works.

---

## Phase 1 â€” 2D smoke test

Use AI assistance to generate the smallest possible 2D platformer.

Implement:

- environment API,
- actions,
- observations,
- rewards,
- training loop,
- checkpoint save/load,
- GUI start/stop.

Deliverable:

> Agent learns a trivial behavior end-to-end.

Then move on.

---

## Phase 2 â€” Controlled CS2 integration

Goal:

> **Observe + act + reset + measure.**

Do not optimize the AI yet.

Deliverable:

```text
controlled environment
        â†“
Python adapter
        â†“
correct observations/actions/events
```

Keep training and repeatable evaluation in the controlled environment. Online-oriented validation, when appropriate, belongs in a separate optional tier and should not become a training dependency.

---

## Phase 3 â€” Random baseline

Before RL, create dumb baselines.

Examples:

```text
random movement
random shooting
simple scripted movement
```

These are necessary because they provide lower-bound comparisons.

Deliverable:

> A baseline report.

---

## Phase 4 â€” First RL agent

Start with:

```text
WASD + LMB
```

Reward:

```text
kill +10
death -10
```

Use a small CNN and PPO.

Deliverable:

> A measurable learning curve.

It does not need to be good yet.

---

## Phase 5 â€” Human demonstrations

Record yourself.

Create behavioral-cloning dataset.

Train imitation model.

Then compare:

```text
RL from scratch
vs.
imitation only
vs.
imitation + RL
```

Deliverable:

> First major research result.

---

## Phase 6 â€” Automatic highlights + checkpoint browser

Add:

- rolling buffer,
- highlight detection,
- automatic clip saving,
- checkpoint metadata,
- checkpoint comparison.

Deliverable:

> â€œWatch the agent improve.â€

---

## Phase 7 â€” Performance optimization

Profile:

- capture,
- preprocessing,
- inference,
- environment loop.

Test:

- observation resolution,
- action frequency,
- model size.

Deliverable:

> Performance/accuracy trade-off report.

---

## Phase 8 â€” Evaluation suite

Build repeatable scenarios.

Create checkpoint-vs-checkpoint testing.

Deliverable:

> Standardized model comparison.

---

## Phase 9 â€” Prediction

Train a separate behavior-prediction model.

Deliverable:

> Measured next-action/next-movement prediction accuracy.

---

## Phase 10 â€” Temporal memory

Test:

```text
single frame
vs.
frame stack
vs.
recurrent memory
```

Deliverable:

> Study of temporal information.

---

## Phase 11 â€” Adaptive curriculum

Measure weaknesses.

Generate more difficult scenarios around them.

Deliverable:

> Agent automatically focuses training on weaknesses.

---

## Phase 12 â€” Advanced controlled objectives

Introduce additional mechanics/objectives in the controlled environment.

Deliverable:

> Agent moves from combat-only behavior toward objective-oriented behavior.

---

## Phase 13 â€” Self-play / AI-vs-AI

Train and evaluate agents against each other.

Deliverable:

> Study of adaptation and emergent strategy in a controlled environment.

---

# 46. The ideal initial implementation

Do not build the entire system above on day one.

Your first useful version should be approximately:

```text
lynxai/
â”œâ”€â”€ src/
â”‚   â””â”€â”€ lynxai/
â”‚       â”œâ”€â”€ agent.py
â”‚       â”œâ”€â”€ environment.py
â”‚       â”œâ”€â”€ trainer.py
â”‚       â”œâ”€â”€ config.py
â”‚       â””â”€â”€ cli.py
â”‚
â”œâ”€â”€ configs/
â”‚   â””â”€â”€ baseline.yaml
â”‚
â”œâ”€â”€ runs/
â”œâ”€â”€ checkpoints/
â”œâ”€â”€ clips/
â””â”€â”€ README.md
```

Then expand the structure as features become real.

Avoid building empty folders for every future idea before you need them.

---

# 47. First coding milestone checklist

When you begin the real implementation, follow this order.

### Step 1
Create project root.

### Step 2
Create Python package.

### Step 3
Implement configuration loading.

### Step 4
Implement path manager.

### Step 5
Implement CLI.

### Step 6
Implement environment interface.

### Step 7
Build the tiny 2D smoke environment.

### Step 8
Train a trivial agent.

### Step 9
Checkpoint it.

### Step 10
Reload it.

### Step 11
Run frozen evaluation.

### Step 12
Implement the controlled CS2 environment adapter.

### Step 13
Test observations/actions/events manually.

### Step 14
Add the first CNN policy.

### Step 15
Train the simplest RL agent.

### Step 16
Add automatic metrics.

### Step 17
Add checkpoints.

### Step 18
Add the GUI.

### Step 19
Add human demonstration capture.

### Step 20
Add imitation-learning pretraining.

Do not move to complicated features until the previous step is observable and reproducible.

---

# 48. Debugging rules

These rules will save enormous amounts of time.

## Rule 1
When behavior is bad, first verify the environment.

## Rule 2
When training is bad, verify rewards before changing the network.

## Rule 3
When performance is slow, profile before shrinking everything.

## Rule 4
When results improve, reproduce them before celebrating.

## Rule 5
Never change five variables in one experiment.

## Rule 6
Keep a checkpoint before every major change.

## Rule 7
Always have a baseline.

## Rule 8
Always have a frozen evaluation mode.

---

# 49. Common failure modes to expect

## Failure: agent learns nothing

Check:

1. Can the agent actually act?
2. Are observations correct?
3. Are rewards firing?
4. Is the episode resetting?
5. Is the optimizer updating?
6. Are gradients nonzero?
7. Is the action space too large?

---

## Failure: agent finds a stupid exploit

That is reward hacking.

Save the run.

Name it something like:

```text
reward_hacking_stationary_strategy
```

Document it.

Then change the reward or environment.

It becomes part of the project story.

---

## Failure: agent overfits

Test it on scenarios it never trained on.

This is why the evaluation suite and scenario generator matter.

---

## Failure: inference is too slow

Profile:

```text
capture
preprocess
model
action
```

Then test:

- lower resolution,
- smaller model,
- lower decision frequency,
- GPU inference,
- more efficient preprocessing.

Only consider a second computer after the single-machine architecture is already correct.

---

# 50. What â€œsuccessâ€ looks like at each level

## Level 1

```text
Agent can learn a tiny 2D task.
```

## Level 2

```text
System can reliably observe and control the controlled CS2 environment.
```

## Level 3

```text
Agent learns a measurable behavior from reward.
```

## Level 4

```text
Human demonstrations improve the starting policy.
```

## Level 5

```text
Agent improves through RL after imitation learning.
```

## Level 6

```text
Agent's performance improves reproducibly across evaluation scenarios.
```

## Level 7

```text
Agent predicts opponent behavior above a useful baseline.
```

## Level 8

```text
Prediction improves decision-making.
```

## Level 9

```text
Agent handles more complex controlled objectives.
```

## Level 10

```text
The whole pipeline is polished enough to demonstrate as a serious portfolio project.
```

You do **not** need Level 10 to have a great project.

Level 4â€“6 alone could already be a very strong portfolio piece if the experiments are well documented.

---

# 51. The â€œresearch questionsâ€ list

Whenever you do not know what to work on next, pick one question.

### Observation

- How does image resolution affect performance?
- Does grayscale hurt or help?
- Does frame stacking help?
- Does a hybrid state representation help?

### Action

- How often should the agent act?
- How does increasing the action space affect sample efficiency?
- Does hierarchical control help?

### Reward

- Is sparse reward enough?
- Does reward shaping accelerate learning?
- What reward exploits appear?

### Imitation

- Does behavioral cloning speed up RL?
- How much human data is enough?
- Does spectator data help?

### Prediction

- How predictable are opponent movement patterns?
- Does prediction improve combat performance?
- How far into the future can the agent predict usefully?

### Memory

- Does temporal memory improve decisions?
- How long should the memory horizon be?

### Curriculum

- Does adaptive training beat random training?
- Does targeted weakness practice reduce total training time?

### Generalization

- Does training on one environment transfer to another?
- Does a model trained on limited scenarios generalize to new scenarios?

### Systems

- What is the best performance/latency trade-off?
- How much does a second machine help?
- How many environment workers can one GPU support?

These questions can keep the project alive for a very long time.

---

# 52. One especially strong experiment matrix

Eventually compare these four agents:

```text
A: RL from scratch

B: Behavioral cloning

C: Behavioral cloning + RL

D: Behavioral cloning + RL + prediction
```

Then evaluate all four on the same test suite.

This one table could become the centerpiece of the portfolio.

Example:

| Agent | Training efficiency | Combat | Prediction | Generalization |
|---|---:|---:|---:|---:|
| RL | baseline | baseline | â€” | baseline |
| BC | high | medium | â€” | medium |
| BC + RL | high | high | â€” | high |
| BC + RL + prediction | high | very high | high | very high |

The actual values must come from your experiments.

---

# 53. Another very strong experiment: model size

Compare:

```text
small CNN
medium CNN
large CNN
```

Measure:

- reward,
- accuracy,
- inference time,
- GPU memory,
- training speed.

The research question:

> **Does a larger model actually improve the agent enough to justify the extra compute?**

This directly answers your original concern about whether the AI must be huge.

---

# 54. Another strong experiment: latency vs performance

Build a graph of:

```text
decision frequency
vs.
performance
```

You may discover something surprising:

> More frequent decisions are not always better if the model becomes noisy or the training becomes unstable.

That is a valuable systems finding.

---

# 55. Another strong experiment: human vs AI behavior

Collect human gameplay data.

Compare:

```text
reaction time
mouse velocity
movement transitions
shooting cadence
correction patterns
```

Visualize distributions.

The point is not to fool an anti-cheat system.

The point is to study:

> **How close is learned behavior to real human behavior?**

That is an interesting ML/behavioral modeling result on its own.

---

# 56. GUI workflow should feel like a real product

The user experience should be:

### Create experiment

```text
New Experiment

Name: baseline_kd
Model: Small CNN
Actions: WASD + LMB
Reward: Kill/Death
Observation: 160Ã—90
```

Click:

**Create**

### Run

Click:

**Start Training**

Watch:

- live metrics,
- live preview,
- current action,
- reward events,
- GPU usage.

### Checkpoint

The app automatically saves.

### Something cool happens

The app says:

> **HIGHLIGHT SAVED**

### Later

Open:

**Checkpoints â†’ Compare 001 vs 047**

### Generate report

Click:

**Generate Report**

And the project creates the graph/report/video summary automatically.

That is the kind of polish that turns a raw ML experiment into a memorable portfolio project.

---

# 57. GUI notifications worth having

Useful events:

```text
Training started
Checkpoint saved
New best reward
New best evaluation
Training stalled
Highlight detected
Experiment complete
Evaluation complete
```

Avoid constant noisy notifications.

Make important events obvious.

---

# 58. Training safety / recovery

Long-running jobs should recover gracefully.

Build:

- automatic checkpoints,
- periodic metrics flushes,
- clean shutdown handling,
- restart-from-checkpoint support,
- crash recovery where practical.

If the process dies after six hours, you should not lose six hours of work.

The ideal mental model is:

> **The experiment can be stopped and resumed at almost any checkpoint.**

---

# 59. Data management

Use clear data labels.

```text
data/raw/
    untouched source data

data/processed/
    normalized/transformed data

data/demonstrations/
    human behavior datasets

data/scenarios/
    generated controlled scenarios
```

Never overwrite raw data.

That makes later experiments much easier.

---

# 60. Version control strategy

Git should track:

- source code,
- configs,
- docs,
- small metadata.

Do not dump huge videos/models into Git.

Use appropriate artifact storage for large outputs.

Every checkpoint should still contain a code version/hash so that it can be traced back to the exact source revision.

---

# 61. The â€œdo not build yetâ€ list

This list matters because the project has a huge feature tree.

Do **not** start with:

- giant transformer,
- complicated world model,
- multi-agent coordination,
- full bomb logic,
- all weapons,
- sophisticated prediction,
- self-play,
- enormous GUI,
- two-computer distributed training.

Start with:

```text
small observation
small action space
simple reward
simple model
controlled environment
clean evaluation
```

Then grow.

---

# 62. The long-term dream architecture

When everything has matured, the project may look like:

```text
                           â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
                           â”‚      GUI         â”‚
                           â”‚  Lab / Monitor   â”‚
                           â””â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                                    â”‚
                           â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
                           â”‚ Experiment       â”‚
                           â”‚ Manager          â”‚
                           â””â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                                    â”‚
                 â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
                 â”‚                  â”‚                   â”‚
                 â–¼                  â–¼                   â–¼
          Demonstrations      Scenario Engine      Live Environment
                 â”‚                  â”‚                   â”‚
                 â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”˜
                              â–¼                  â–¼
                         Data Store        Observations
                              â”‚                  â”‚
                              â””â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                                       â–¼
                                Vision Encoder
                                       â”‚
                            â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
                            â–¼                      â–¼
                      World / Memory         Prediction
                            â”‚                      â”‚
                            â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                                       â–¼
                                   Policy
                                       â”‚
                                       â–¼
                                    Action
                                       â”‚
                                       â–¼
                                  Environment
                                       â”‚
                                       â–¼
                                    Reward
                                       â”‚
                                       â–¼
                                    Trainer
                                       â”‚
                                       â–¼
                                  Checkpoint
                                       â”‚
                           â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
                           â–¼                       â–¼
                      Evaluation              Highlights
```

That is the end-state, not the starting point.

---

# 63. What to do the moment you start

Your first working sprint should have one goal:

> **Get the entire pipeline working with the smallest possible environment.**

Do this in order:

```text
1. Create repo
2. Create Python project
3. Create config system
4. Create path system
5. Create CLI
6. Create minimal GUI shell
7. Create environment interface
8. Have AI generate tiny 2D game
9. Connect agent to 2D game
10. Train successfully
11. Save checkpoint
12. Reload checkpoint
13. Evaluate with frozen weights
14. Build first CS2 controlled environment adapter
15. Verify observation/action/reward loop
16. Train first tiny CS2 agent
```

At that point, **you have officially started the real project.**

Everything after that is an upgrade.

---

# 64. The most important mindset for this project

You are not trying to build the final super-agent immediately.

You are building a **platform for experiments**.

Every layer should make the next experiment easier.

The project should feel like:

```text
Experiment
   â†“
Result
   â†“
Checkpoint
   â†“
Insight
   â†“
Next experiment
```

not:

```text
write huge AI
   â†“
run
   â†“
hope
```

The strongest possible outcome is not necessarily an agent that becomes unbelievably strong.

A strong outcome is:

> **You can clearly demonstrate how the agent learned, what helped, what failed, why it failed, and how each experiment changed the system.**

That story is the portfolio.

---

# 65. Final project feature list

Keep this as the master backlog.

## Core

- [ ] Python project structure
- [ ] Config system
- [ ] Path manager
- [ ] Clean CLI
- [ ] GUI
- [ ] Environment abstraction
- [ ] Observation adapter
- [ ] Action adapter
- [ ] Reward adapter
- [ ] Event detection
- [ ] Logging

## ML

- [ ] Small CNN
- [ ] PPO baseline
- [ ] Behavioral cloning
- [ ] PPO fine-tuning
- [ ] Temporal model
- [ ] Prediction model
- [ ] Adaptive curriculum
- [ ] Hybrid state model
- [ ] Self-play

## Data

- [ ] Human demonstrations
- [ ] Spectator demonstrations
- [ ] Scenario dataset
- [ ] Processed dataset
- [ ] Dataset metadata

## Training

- [ ] Training mode
- [ ] Evaluation mode
- [ ] Train/evaluate cycles
- [ ] Overnight mode
- [ ] Automatic stall detection
- [ ] Resume from checkpoint
- [ ] Automatic best-model selection

## Checkpoints

- [ ] Save model
- [ ] Save optimizer
- [ ] Save config
- [ ] Save metrics
- [ ] Save environment/version info
- [ ] Save code revision
- [ ] Checkpoint browser
- [ ] Restore
- [ ] Compare checkpoints
- [ ] Same-scenario checkpoint replay

## Highlights

- [ ] Rolling video buffer
- [ ] Kill clips
- [ ] Multi-kill clips
- [ ] Reward-spike clips
- [ ] Prediction-success clips
- [ ] Round-win clips
- [ ] Metadata alongside clips
- [ ] Highlight browser

## Evaluation

- [ ] Standard scenarios
- [ ] Generalization tests
- [ ] Model comparison
- [ ] Reward-function comparison
- [ ] Observation-resolution comparison
- [ ] Latency comparison
- [ ] Model-size comparison

## Advanced gameplay

- [ ] Better aiming
- [ ] RMB
- [ ] Additional controlled mechanics
- [ ] Grenades
- [ ] Objective behavior
- [ ] Team/coordination research

## Research

- [ ] Reward hacking analysis
- [ ] Human-vs-AI behavior comparison
- [ ] Prediction accuracy
- [ ] Temporal modeling
- [ ] Curriculum learning
- [ ] Generalization
- [ ] Model efficiency
- [ ] Self-play

---

# 66. The single sentence to keep coming back to

> **Build the smallest system that can answer the next interesting question.**

That principle will keep this project manageable even as the feature list becomes enormous.

---

## References / current CS2 resources

Valve currently describes Counter-Strike 2 as running on Source 2 and notes that Source 2 tools are available for community map makers and artists. îˆ€citeîˆ‚turn748072search1îˆ

Valve's official Counter-Strike Workshop resources provide documentation and access points for community content creation, including map tooling resources. îˆ€citeîˆ‚turn748072search2îˆ‚turn748072search11îˆ

Valve's current CS2 update notes show continued additions and changes to map scripting and game-state-related systems, reinforcing the need to version and pin your environment assumptions when doing long-running experiments. îˆ€citeîˆ‚turn748072search0îˆ‚turn748072search10îˆ

Valve's developer documentation also documents LAN/practice-oriented tooling and demo playback concepts that can be useful when designing isolated testing/evaluation workflows. îˆ€citeîˆ‚turn748072search9îˆ
