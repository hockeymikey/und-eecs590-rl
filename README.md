# EECS 590 RL — Mini-project repo

University of North Dakota, Spring 2026. Version-graded mini-project repo for EECS 590 *Reinforcement Learning*. The codebase covers the V1/V2 tabular content (DP, MC, TD(n)/TD(λ), SARSA, Q-learning) and the V3 capstone: an autonomous-Zamboni foundation environment with a BC-warmstarted, from-scratch PPO trained against both a Gymnasium image-obs env and an Isaac Sim variant.

**Instructor:** Dr. Alex Lowenstein. **Authorship:** solo work; AI-assistance disclosure in [`docs/AI_usage.md`](docs/AI_usage.md).

Everything is in-tree under `src/rl590/`. No sibling research repos need to be cloned for the graded surface.

## Setup

```bash
uv sync                  # creates the venv from pyproject.toml + uv.lock
```

That's it for the gym-side workflow (tabular + foundation env + BC + from-scratch PPO).

The Isaac Sim variant runs inside the Isaac Lab 2.3.2 container and has its own non-trivial setup path on Arch Linux. If you want to reproduce the Isaac runs, start with `docs/isaacsim_setup_issues.md` — it documents the full setup including the upstream bugs you'll hit and the workarounds I'm using.

## Running things

All recipes are in the `Justfile`. `just` with no arguments lists everything.

### Tabular (V1 / V2)

```bash
just windy-train policy_iteration    # DP on WindyChasm
just windy-eval                      # evaluate the saved policy
just windy-bootstrap-model           # belief-model bootstrap workflow
just demo-classical                  # MC, TD, SARSA, Q-learning demo
just test                            # smoke tests
```

### Foundation env — gym side (V3)

```bash
just gym-smoke                       # import + step ZambGymEnv
just gym-teacher-smoke               # roll out the coverage-path teacher
just gym-bc                          # fit BC actor on the teacher
just gym-ppo-scratch                 # train PPO from scratch
just gym-ppo-bc                      # train PPO with BC warmstart
just gym-tb                          # TensorBoard for BC + PPO runs
```

### Foundation env — Isaac side (V3)

The Isaac scripts live in `scripts/` and run inside the Isaac Lab container; see `docs/isaacsim_setup_issues.md` for the setup and `docs/decisions.md` (Isaac Sim section) for what they actually do.

## Documentation

The narrative lives under `docs/`. In priority order for a reviewer:

| Doc | What's in it |
|---|---|
| [`docs/decisions.md`](docs/decisions.md) | What I chose to implement, what I chose not to, and why. Most authoritative narrative. |
| [`docs/technical-challenges.md`](docs/technical-challenges.md) | Bugs, surprises, debugging stories — including the PPO Run 1/2 collapse and the reward-shape bug. |
| [`docs/zamb_gym_env.md`](docs/zamb_gym_env.md) | Foundation env reference: observation, action, reward, test recipes. |
| [`docs/isaacsim_setup_issues.md`](docs/isaacsim_setup_issues.md) | Isaac Lab 2.3.2 on Arch Linux — every upstream bug encountered and the fix. |
| [`docs/v1.md`](docs/v1.md) / [`docs/v2.md`](docs/v2.md) / [`docs/v3.md`](docs/v3.md) | Course assignment specs for each version. |

## Citations and AI usage

See [`CITATIONS.md`](CITATIONS.md) for the full citation list (textbook, PPO/GAE papers, domain references, software dependencies) and [`docs/AI_usage.md`](docs/AI_usage.md) for the AI-assistance disclosure. Short version: Claude was used as a code-review, debugging, and documentation-drafting collaborator; all implementation, training, debugging decisions, and conclusions are mine.