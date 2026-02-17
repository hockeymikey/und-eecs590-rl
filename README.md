# EECS 590: Reinforcement Learning

Course repository for mini assignments and versioned RL project work at the University of North Dakota (Spring 2026).

## Repository Layout

```text
.
├── M1/                         # Original Mini 1 submission artifacts
│   └── code/                   # Legacy Mini 1 prototype scripts
├── M2/                         # Original Mini 2 submission artifacts
├── legacy/                     # Legacy stubs retained for traceability
├── src/
│   └── rl590/                  # Refactored V1-ready RL package
│       ├── agents/             # Agent config + train/eval wrapper
│       ├── dp/                 # DP algorithms (VI/PI/Q-policy-improvement)
│       ├── envs/               # Windy Chasm tabular MDP
│       ├── model/              # Belief updates for learned tabular MDP model
│       └── utils/              # Rendering + plotting helpers
├── scripts/
│   └── run_windy.py            # Main CLI entrypoint
├── docs/
│   ├── v1.md                   # V1 assignment description
│   ├── rubric_v1_checklist.md  # Rubric mapping and remaining gaps
│   └── notes/                  # Scratch notes + templates
├── artifacts/                  # Saved models and plots
├── notebooks/                  # Notebook workspace
├── tests/                      # Smoke tests
└── requirements.txt
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train / Evaluate

Train with default policy iteration, save artifacts, print policy/value tables:

```bash
python3 scripts/run_windy.py train
```

Train with Q-value policy improvement:

```bash
python3 scripts/run_windy.py train --algorithm q_policy_iteration
```

Evaluate a saved model:

```bash
python3 scripts/run_windy.py eval --model-path artifacts/windy_best_policy.npz --render
```

Bootstrap a model belief from sampled transitions, then plan on the learned model:

```bash
python3 scripts/run_windy.py bootstrap-model --algorithm policy_iteration --render
```

Note: `bootstrap-model` performance depends on transition coverage. Increase `--bootstrap-episodes` for better estimates.

Compare bootstrap quality across episode budgets and write a CSV:

```bash
python3 scripts/compare_belief_bootstrap.py \
  --episode-grid 20,50,100,200,500 \
  --num-seeds 5 \
  --output artifacts/bootstrap_sweep.csv
```

## Notes Workflow

Use `docs/notes/` for session scratch notes and concept summaries.
Start from templates in `docs/notes/templates/`.

## Legacy Assignment Files

`M1/` and `M2/` are preserved as historical submission artifacts.
`M1/code/` and `legacy/` contain older prototypes/stubs.
`src/rl590/` is the cleaned V1 baseline moving forward.

## Citations / Collaboration

Will be added in future iterations as necessary.