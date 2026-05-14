# Implementation Decisions

This document records what I chose to implement, what I chose not to, and why. Per V3 §Final Content, the decisions are mine alone — they may not all be reasonable, but they are deliberate.

## Foundation environment

The capstone environment is a continuous-control ice resurfacer (a Gymnasium-compatible "Zamboni" env) implemented in this repository at `src/rl590/envs/zamb_gym.py`. It is a stripped-down adaptation of a private prototype; the broader prototype stays outside this course repo because it is used for work beyond EECS 590. The Minis copy is self-contained — there are no imports from that workspace — and uses course-specific reward coefficients and a deliberately simpler linear refreeze placeholder (see `src/rl590/envs/surface.py`).

Observation is a 3-channel `(H, W, C)` image at nav-grid resolution: damage magnitude (normalised), agent-pose hot pixel, and refreeze-progress in `[0, 1]`. Action is continuous `(throttle, steering)` in `[-1, 1]`. This shape rules out tabular methods entirely and pushes toward CNN-based function approximation with a continuous-action policy gradient method.

A second, Isaac-Sim-based env lives at `src/rl590/envs/zamb_isaac.py` — a single-Zamboni rigid body on a lowpoly NHL rink with kinematic root-velocity drive and a damage-decrement reward on a 1 cm truth grid. The Isaac env imports `isaaclab.*` and only resolves inside the Isaac Lab 2.3.2 container; container setup is documented in `docs/isaacsim_setup_issues.md`. The image-CNN gym env and the Isaac env are interchangeable from the from-scratch PPO's perspective via the `IsaacSingleEnvWrapper` (`src/rl590/envs/wrappers.py`).

## Algorithms implemented

### Tabular (V1 / V2)

| Algorithm | Module | Notes |
|---|---|---|
| Value Iteration | `src/rl590/dp/planning.py` | Bellman-optimality updates |
| Policy Iteration | `src/rl590/dp/planning.py` | Both value-based and Q-based improvement paths |
| Monte Carlo | `src/rl590/classical/mc.py` | First-visit and every-visit |
| TD(n), TD(λ) | `src/rl590/classical/td.py` | Forward and backward views; backward uses accumulating eligibility traces with n-step truncation |
| SARSA(n), SARSA(λ) | `src/rl590/classical/sarsa.py` | Same forward/backward duality |
| Q-learning | `src/rl590/classical/qlearning.py` | Off-policy TD(0) |
| Tabular belief estimator | `src/rl590/model/belief.py` | Counts-based P̂(s'∣s,a) and R̂; planner bootstraps from the learned model |

Tested on `WindyChasm` (140 states, 3 actions) and `ReactorEnv` (20 states, 5 actions). These satisfy V1/V2 rubric content and exercise the underlying mechanics; they do not run on the foundation env.

### Deep RL (capstone)

PPO is the primary deep RL algorithm.

PPO appears in two places: a from-scratch course implementation under `src/rl590/deep/ppo.py` (clipped surrogate, GAE, entropy bonus, Gaussian policy with tanh squashing + log-prob change-of-variables correction), and `stable-baselines3` PPO for the Isaac Lab vectorized smoke run. The from-scratch implementation is the main graded gym-side path (`scripts/demo_zamb_gym_ppo.py`) and is also wired to the Isaac env through `scripts/demo_zamb_isaac_ppo.py`. SB3 is used where it buys platform integration rather than algorithmic novelty: Isaac Lab already exposes an SB3 vector-env wrapper, so using it there isolates Isaac setup and environment mechanics from PPO-loop debugging.

#### Why PPO

| Algorithm | Verdict | Reasoning |
|---|---|---|
| DQN | Skip | Discrete-only; binning throttle/steering loses precision in a domain where smooth control matters. |
| REINFORCE | Skip | High variance on long episodes — entire good rollouts get drowned by one collision. |
| Vanilla A-C | Skip | No trust region. Vulnerable to policy collapse on a collision-heavy env (and Run 1 confirmed this for any unconstrained continuous PG). |
| DDPG | Skip | Deterministic policy + single critic = brittle; sparse collision penalties cause value overestimation. |
| TD3 | Skip | Strong candidate, but off-policy + a `(H, W, ~5)` image observation would need a multi-GB replay buffer per run. |
| **PPO** | **Pick** | On-policy avoids the replay-buffer memory blowup; clipped objective + entropy bonus survived a regime where vanilla PG collapsed (Run 1 → Run 2, see `docs/technical-challenges.md`); GAE is a tunable bias-variance knob. |
| TRPO | Skip | Same trust-region benefit as PPO at significantly higher implementation cost (Fisher information matrix + conjugate gradient); no clear win for this env. |
| SAC | Skip | The strongest second choice — but off-policy with the same replay-buffer problem as TD3, plus auto-temperature tuning complexity. If PPO had failed unrecoverably I would have implemented SAC; it didn't, so SAC stayed unimplemented. |

### Imitation / warmstart

I implemented behaviour-cloning warmstart from a deterministic coverage-path teacher (`src/rl590/training/bc.py`, `scripts/bc_warmstart_zamb_gym.py`). The teacher's path is loaded from a precomputed npz artifact (`assets/teacher/coverage_path_v1.npz`); the path-generation algorithm itself lives outside the submission, so only the data is consumed here. A small pure-pursuit follower (`src/rl590/envs/teacher.py`) replays it as `(throttle, steering)` actions.

The standard `imitation` library was rejected after profiling — its `flatten_trajectories` materialises both `obs` and `next_obs` as full arrays, which exceeded available RAM for image-observation rollouts at the scale I needed (200 episodes × 400 steps). The hand-rolled BC streams observations to disk via `np.memmap` so peak RSS stays at one frame.

The first BC run on the prototype env failed (loss flat at ~prior log-prob across all epochs, eval indistinguishable from random). Diagnosed as the observation missing pose information that the teacher uses, which collapsed many distinct teacher actions onto the same student observation; addressed by including the agent-pose channel in the observation (now part of `ZambGymEnv._get_obs`).

#### V3 BC vs. PPO-from-scratch comparison

Single-seed run, 50k env steps each, identical PPO hyperparameters (`--rollout-steps 1024`, `--max-episode-steps 400`, `ent_coef=0.01`). Figures in `docs/figures/`:

- **`bc-loss.png`** and **`bc-nll.png`** — BC training loss and per-sample NLL both drop from ~0.22 to ~0.01 over 10 epochs (loss is what the optimizer minimizes, NLL is the clean data term; the curves track each other up to a small additive constant). The student fits the teacher's action distribution cleanly. This part works.
- **`train-entropy.png`** — Both PPO runs hold entropy in `[2.78, 2.86]` throughout. Neither collapses to the deterministic regime that wrecked Run 1 (see `technical-challenges.md`). `bcinit` drifts down faster than `scratch` (2.84 → 2.78 vs 2.85 flat) — consistent with the BC-initialised actor committing harder to its narrow starting mode and PPO refining around it.
- **`rollout-mean_return_last10.png`** — Both runs converge to ~3.9-4.0k return at 50k steps. The early-rollout BC advantage seen in the 2k smoke is gone by the time PPO converges. *On this seed, BC warmstart does not yield a measurable final-performance gain.*

**Why I think this happens** — three things, probably together:

1. **The teacher npz is one fixed trajectory.** Every BC episode rolls out the same clockwise lawnmower coverage path from the global-nearest anchor point. So BC fits a single trajectory shape and learns it well; it cannot exceed what it imitates.
2. **The reward landscape has a broad "decent coverage" plateau.** Many policies score in the 3.5-4.5k band — the lawnmower the teacher demonstrates is one of them, but so are policies PPO from scratch discovers via its own exploration. Once both reach the plateau, the difference washes out.
3. **PPO from BC-init may need to *unlearn* before it can explore.** The BC actor is initialised in a tight basin around the teacher's action distribution; PPO has to spend gradient steps loosening that basin before it can find anything beyond the lawnmower. The entropy curve is consistent with this — `bcinit` is becoming *more* committed over training, not less.

**Future direction (not in this submission, but the planned next step):** generate a *set* of teacher paths from an offline planner that varies per seed and per scenario — different demand maps, different start cells, different perimeter-lap directions. BC over a path *distribution* would fit a *conditional* policy keyed on the observation rather than a single trajectory, which is the prerequisite for BC to teach behaviours PPO wouldn't rediscover on its own (prioritising higher-demand patches, handling start poses far from a path's "natural" entry, etc.). The current single-path teacher is too easy for unguided PPO to match.

I'm keeping the warmstart in the repo as evidence the pipeline works end-to-end (teacher → BC → PPO fine-tune, with TensorBoard scalars across the whole chain). The honest read on the comparison itself is "no final-performance gain at this seed with this single-path teacher" — which is a more interesting decisions.md entry than a fake win would be.

## Reward and observation design

The submitted `ZambGymEnv` reward is brush-footprint based:

1. **Damage reduction.** Credit is proportional to millimetres of damage removed under the brush footprint.
2. **Coverage.** Newly swept nav cells receive a small bonus based on the brush mask, not the chassis centre trace.
3. **Refreeze readiness.** Newly crossed cells that were ready (`refreeze_progress >= REFREEZE_READY_THRESHOLD`) receive a bonus; cells crossed too early receive a penalty.
4. **Operational penalties.** Wall contacts and per-step cost keep the policy from exploiting stationary or board-riding behaviours.

The observation is intentionally image-only for the gym path: `(damage_norm, agent_pose, refreeze_progress)`. I kept the pose as a hot-pixel channel rather than a separate vector so the same CNN actor/critic can consume the whole state without a second feature tower.

This design was motivated by the reward-shape bug documented in `docs/technical-challenges.md`: an earlier prototype run rewarded centre-cell visitation strongly enough that it was not actually measuring resurfacing. The submitted env avoids that failure mode by tying the dominant reward terms to the brush footprint.

## Workflow and tooling decisions

- **TensorBoard, not Weights & Biases.** Keeps logs local and easy to archive with the submission.
- **`uv`, not `pip` or `poetry`.** Fast lockfile-backed setup with one command (`uv sync`).
- **`just` recipes, not a Makefile.** Discoverable (`just` lists everything), no tab/whitespace footguns, plays cleanly with `uv run`.
- **Training outputs stay out of git.** `training_runs/`, checkpoints, and replay buffers are ignored; summary figures that matter for the report are copied into `docs/figures/`.
- **Explicit seed flags.** The PPO, BC, and smoke scripts expose `--seed` so the single-seed results in the submission can be rerun and extended into multi-seed sweeps later.

## Post-V2 topics — explicit choices

V3 Final Content asks for decisions on content addressed in class since V2.

- **Bayesian hyperparameter tuning.** Skipped for this submission. PPO sensitivity surfaced clearly in the Run 1 collapse → Run 2 entropy fix, so a small Optuna/skopt sweep over entropy coefficient, learning rate, clip range, and target KL is the highest-value next experiment, but I prioritized a reproducible env + BC/PPO pipeline over adding another tuning layer.
- **POMDPs / belief networks for deep RL.** Skipped. The gym observation exposes the nav-grid damage layer, refreeze progress, and agent pose; that is enough state for the course-scale controller. Adding a recurrent or temporal-feature head would cost training compute without addressing the main failure modes observed here.
- **Bayesian belief updates over MDPs.** Implemented (V1) at the tabular level — see `src/rl590/model/belief.py`. Not lifted to the deep setting; Bayesian deep RL is its own research area and out of scope here.
- **Multi-agent / coordination / swarm.** Skipped. Single-protagonist environment; coordination is structurally inapplicable.
- **Isaac Sim / hardware bridging.** Implemented as a single-Zamboni toy environment at `src/rl590/envs/zamb_isaac.py` on Isaac Lab 2.3.2 / Isaac Sim 5.1.0. The env spawns a Zamboni rigid body on a lowpoly NHL rink, applies kinematic root-velocity control from a continuous `(forward, yaw-rate)` action, and computes reward as damage decrement under the chassis footprint on a 1 cm truth grid (deliberately simpler than the gym env reward — see Foundation environment section). Trained with SB3 PPO (`scripts/train_zamb_isaac_sb3.py`) for 50k timesteps over 8 vectorized envs as a platform-validation smoke run; generated checkpoints and curves live under `training_runs/` locally and are intentionally not versioned. Structurally compatible with the from-scratch `rl590.deep.ppo.PPOAgent` via the `IsaacSingleEnvWrapper` in `src/rl590/envs/wrappers.py` (Isaac Lab is N-env-vectorized by default; the wrapper squeezes the leading dim to feed the from-scratch single-env PPO loop — `scripts/demo_zamb_isaac_ppo.py`). The visible class deliverable is the Isaac Sim attempt itself — getting Isaac Lab running on Arch Linux + recent NVIDIA driver was a multi-hour engineering exercise documented in `docs/isaacsim_setup_issues.md` and summarised in `docs/technical-challenges.md`. This matches V3's framing that "struggling with Isaac Sim coordination beats an afternoon training run on a polished gym env."

## Removed content

Per V3 §Structural Readiness, the M1–M4 mini-project source trees and PDFs were removed before final submission. V1 and V2 graded against those directories on this same repository.

This is the one place the V3 spec contradicts V1/V2: those earlier versions explicitly required the mini directories on the public repo, and V3 explicitly says to remove them. I'm following V3 because it's the most recent instruction, and noting the contradiction here for transparency rather than silently dropping the directories.

## Future work

- **Seed-varied offline teacher.** Single-path BC plateaued at the same return as PPO-from-scratch (see V3 figures discussion above). Next iteration: emit N teacher paths from an offline planner with per-seed variation (different demand maps, start cells, perimeter directions) and BC across the distribution, so the student learns a conditional policy rather than memorising one trajectory.
- **Multi-seed PPO sweep on `ZambGymEnv`.** The BC-vs-scratch comparison was one seed; the convergence story would be much stronger as mean ± std across 3-5 seeds. Same recipe via `just gym-ppo-scratch` / `just gym-ppo-bc` with `--seed N` overrides.
- **Bayesian hyperparameter sweep.** Entropy coefficient and learning rate matter enough here to justify a small Optuna/skopt pass once the single-seed submission pipeline is frozen.

## Citations and collaboration

See `CITATIONS.md` for sources and `docs/AI_usage.md` for the AI-assistance disclosure. AI was used collaboratively for code review, debugging, design discussions, and documentation drafting. All implementation, training, debugging decisions, and conclusions are mine.
