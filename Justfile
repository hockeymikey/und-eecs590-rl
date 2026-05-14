# rl590-minis command shortcuts. Run `just` (no args) to list recipes.
#
# Recipes assume:
# - `uv` is installed and on PATH.
# - The project venv has been synced (`just setup` does this).
#
# `uv run` resolves the project venv automatically — no manual activation.

# Default: list every recipe with its docstring.
default:
    @just --list

# ─── Setup ────────────────────────────────────────────────────────────

# Sync the project venv from pyproject.toml + uv.lock.
setup:
    uv sync

# ─── Class work (V1 / V2 tabular) ─────────────────────────────────────

# Train on Windy Chasm with the given DP algorithm.
windy-train algorithm="policy_iteration":
    uv run python scripts/run_windy.py train --algorithm {{algorithm}}

# Evaluate a saved Windy Chasm policy artifact.
windy-eval model="artifacts/windy_best_policy.npz":
    uv run python scripts/run_windy.py eval --model-path {{model}}

# Belief-bootstrap workflow: fit a tabular model from rollouts, plan on it.
windy-bootstrap-model algorithm="policy_iteration" episodes="200":
    uv run python scripts/run_windy.py bootstrap-model \
        --algorithm {{algorithm}} --bootstrap-episodes {{episodes}}

# Run the classical-algorithms demo (MC, TD, SARSA, Q-learning).
demo-classical:
    uv run python scripts/demo_classical.py

# Run the smoke tests.
test:
    uv run pytest tests/

# ─── Foundation env (in-tree, no external deps) ────────────────────────

# Verify ZambGymEnv imports and steps through a few actions.
gym-smoke:
    uv run python -c "import numpy as np; from rl590.envs.zamb_gym import ZambGymEnv; \
        e = ZambGymEnv(max_steps=20); o, _ = e.reset(seed=0); \
        print('obs shape:', o.shape); \
        o, r, t, tr, info = e.step(np.array([0.5, 0.1], dtype=np.float32)); \
        print('reward:', r, 'components:', info['reward_components'])"

# Roll out the coverage-path teacher for N steps; print return + idx progress.
gym-teacher-smoke steps="400":
    uv run python scripts/smoke_zamb_gym_teacher.py --steps {{steps}}

# Fit the BC actor on the coverage-path teacher; logs to TensorBoard.
gym-bc episodes="5" epochs="10":
    uv run python scripts/bc_warmstart_zamb_gym.py \
        --teacher-npz assets/teacher/coverage_path_v1.npz \
        --episodes {{episodes}} --max-episode-steps 400 \
        --epochs {{epochs}}

# Train PPO from scratch on ZambGymEnv. Logs to TensorBoard.
gym-ppo-scratch timesteps="50000":
    uv run python scripts/demo_zamb_gym_ppo.py \
        --total-timesteps {{timesteps}} \
        --rollout-steps 1024 --max-episode-steps 400 \
        --run-tag scratch

# Train PPO with BC warmstart initialization. Logs to TensorBoard.
gym-ppo-bc timesteps="50000" bc_path="training_runs/zamb_gym_bc_v1/actor_init.pt":
    uv run python scripts/demo_zamb_gym_ppo.py \
        --total-timesteps {{timesteps}} \
        --rollout-steps 1024 --max-episode-steps 400 \
        --bc-init {{bc_path}} \
        --run-tag bcinit

# Launch TensorBoard for BC + PPO logs (bind_all=1 for SSH/remote viewing).
gym-tb port="6006" bind_all="":
    uv run tensorboard --logdir_spec \
        bc:training_runs/zamb_gym_bc_v1/tb,ppo:training_runs/zamb_gym_ppo_v1/tb \
        --port {{port}} --host {{ if bind_all == "" { "localhost" } else { "0.0.0.0" } }}
