# Version 1 Rubric Checklist

Mapping `v1.MD` rubric items to concrete repo artifacts.

## 1. GitHub Repository Formatting
- Status: In progress
- Evidence:
  - Stable top-level structure in `README.md`
  - Core code under `src/rl590/`
  - Scripts under `scripts/`
  - Notes/docs under `docs/`
- Gap:
  - Not a full cookiecutter clone yet (intentional low-risk incremental refactor).

## 2. Documentation / Organization
- Status: Mostly covered
- Evidence:
  - `README.md` includes setup and run commands.
  - `requirements.txt` added.
  - Notes workflow in `docs/notes/README.md`.
- Gap:
  - README citations/collaboration section should be filled with class-specific content.

## 3. Sufficient MDP Representation
- Status: Covered (tabular model + belief updates)
- Evidence:
  - State/action spaces and transition tensor in `src/rl590/envs/windy_chasm.py`.
  - Expected reward table in `src/rl590/envs/windy_chasm.py`.
  - Prior-belief updates from observed transitions in `src/rl590/model/belief.py`.
  - End-to-end belief bootstrap workflow in `src/rl590/cli.py` (`bootstrap-model` command).

## 4. DP Implementation
- Status: Covered for V1 expectation
- Evidence:
  - Value Iteration in `src/rl590/dp/planning.py`.
  - Policy Iteration + policy improvement from values in `src/rl590/dp/planning.py`.
  - Q-value policy improvement path in `src/rl590/dp/planning.py`.
- Gap:
  - No TD(lambda) fallback implementation yet.

## 5. Framework for Agent Implementation
- Status: Covered (tabular planning agent)
- Evidence:
  - Hyperparameterized agent config in `src/rl590/agents/planning_agent.py`.
  - Train/evaluate CLI in `src/rl590/cli.py` and `scripts/run_windy.py`.
  - Saved policy/value artifacts in `artifacts/windy_best_policy.npz`.
- Gap:
  - No learned function approximation agent yet (future versions).

## Recommended Soon (from rubric's non-graded list)
- Monte Carlo methods
- TD(n), TD(lambda)
- Sarsa(n), Sarsa(lambda)
- Exploration strategies and Q-learning
- Optional logging integrations (TensorBoard/WandB)
