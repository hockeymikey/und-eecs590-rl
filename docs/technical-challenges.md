# Technical Challenges and Surprises

A running log of bugs, sticking points, and unexpected behavior encountered
during development.

## V1

- **Transition probability normalization**: Early WindyChasm implementation had
  wind shift probabilities that didn't sum to 1.0 for edge columns. The fix
  was clamping wind-shifted columns to grid bounds and accumulating probability
  mass on boundary states.

- **Belief model convergence**: The bootstrap-model approach needs a surprising
  number of exploration episodes (~200+) to achieve decent state-action coverage
  on the 140-state WindyChasm. With fewer episodes, the learned transition model
  has entire rows of zero probabilities, causing planning to produce degenerate
  policies.

## V2

### Classical Algorithms

- **Forward-view requires full episodes**: Initially tried to implement TD(λ)
  forward-view with online updates, which is impossible — you need the complete
  trajectory to compute the λ-weighted return. This is the fundamental
  distinction between forward and backward views. Backward-view uses eligibility
  traces to achieve the same result online.

- **TD prediction vs control**: The TD(n) and TD(λ) modules learn V(s), not
  Q(s,a) directly. When first testing on WindyChasm, the returns were negative
  because the Q-table (maintained as a side-effect for action selection) wasn't
  the primary learning target. SARSA and Q-learning learn Q directly and
  perform significantly better for control tasks.

- **Eligibility trace truncation for n-step backward view**: Implementing the
  n-step backward view required tracking the "age" of each trace to zero it
  after n steps. Without truncation, you get the full TD(λ) backward-view
  instead. This is a subtle but important distinction — the n-step truncation
  is what makes backward-view equivalent to the n-step forward-view
  specifically, rather than the λ-return.

- **Environment interface mismatch**: WindyChasm didn't have a `reset()` method
  (used `start_index` property instead), while ReactorEnv did. Added `reset()`
  to WindyChasm so all classical agents work uniformly across environments.

- **On-policy vs off-policy convergence**: Q-learning (off-policy) converges to
  the optimal Q* regardless of the exploration policy, while SARSA (on-policy)
  converges to the Q-values of the ε-greedy policy it follows. In practice,
  Q-learning slightly outperforms SARSA on WindyChasm (eval return 78.7 vs 75.0)
  with 500 episodes, but SARSA(λ) backward-view beats Q-learning on the Reactor
  (513.7 vs 480.4) likely due to the credit assignment benefit of traces in the
  longer-horizon Reactor episodes.

### Foundation Environment (Zamboni)

- **Observation space**: The Zamboni environment outputs 2-channel images
  (129x304x2) — ice roughness map and agent position heatmap. This rules out
  tabular methods entirely and requires CNN-based function approximation.

- **Continuous action space**: Throttle and steering are both continuous in
  [-1, 1]. This eliminates DQN (discrete actions only) as a primary algorithm
  choice and points toward policy gradient methods (PPO, SAC, DDPG, TD3).

- **Separate codebases**: The Zamboni environment lives in a separate research
  repository since it will be used for a paper beyond this course. Integrated
  via `pip install -e` to keep codebases independent while allowing imports.

### Neural Network Architecture

- **CNN flattened size computation**: The flattened output size of the conv
  layers depends on the input dimensions and stride/padding choices. Rather than
  computing this by hand (error-prone with 3 conv layers), we run a dummy
  forward pass through the conv stack during `__init__` to determine the size
  automatically. This pattern comes from the stable-baselines3 codebase.

- **Channels-last vs channels-first**: Gymnasium environments output
  observations in channels-last format (H, W, C) but PyTorch convolutions
  expect channels-first (C, H, W). The CNN handles this by detecting the
  format at runtime and permuting if needed. Forgetting this permutation
  is a common source of silent bugs where training runs but learns nothing.

### Deep RL (PPO)

- **Tanh squashing log-probability correction**: When sampling continuous
  actions from a Gaussian and squashing through tanh to bound them to [-1, 1],
  the log-probability must be corrected for the change of variables:
  `log π(a|s) = log N(u|μ,σ) - Σ log(1 - tanh²(u))`. Missing this correction
  biases the policy gradient — the agent would over-explore near the action
  bounds where tanh compresses the distribution.

- **GAE is just TD(λ) on advantages**: Implementing Generalized Advantage
  Estimation was straightforward once the connection to TD(λ) backward-view
  clicked. The backward pass through the rollout buffer (`δ + γλ * gae`) is
  exactly an eligibility trace accumulation over TD errors, applied to
  advantage estimation instead of value prediction.

- **Separate actor and critic CNNs**: Sharing a single CNN between actor and
  critic seems efficient but can cause training instability — the critic's
  gradient signal can interfere with the features the actor needs. Using
  separate CNN weights for each (~10.6M total parameters) is the safer default.

- **numpy float32 not JSON serializable**: When saving training logs, numpy
  float32 values from episode returns and rollout stats can't be directly
  serialized to JSON. Required a recursive converter to cast numpy scalars
  to Python floats before `json.dump()`.

### Saliency Analysis

- **Actor returns a tuple**: The GaussianActor's `forward()` returns
  `(mean, log_std)` not a single tensor. The saliency functions initially
  assumed a single tensor output and crashed on the actor. Fixed by detecting
  tuple outputs and extracting the first element (the action means).

- **Grad-CAM requires hook registration**: Unlike vanilla gradients which just
  need `requires_grad_(True)` on the input, Grad-CAM needs forward and backward
  hooks on a specific conv layer to capture activations and their gradients.
  These hooks must be removed after use to avoid memory leaks in long-running
  training loops.

## Capstone — Foundation Env Training Runs

Headless overnight PPO runs on `RinkEnv` (rinkgym repo, separate codebase) using
SB3 with a custom CNN extractor. Three stories, in order of discovery.

### Run 1 — Vanilla PPO Policy Collapse

- **Symptom**: At ~770k of 1.5M steps, ep_rew_mean plateaued near 30 and
  diagnostic metrics went off the rails: `std=0.00123`, `approx_kl=2.29`,
  `clip_fraction=0.82`. Healthy targets are `std > 0.1`, `approx_kl ~0.01–0.05`,
  `clip_fraction ~0.1–0.3`.
- **Cause**: SB3's default `ent_coef=0.0` provides no pressure to keep a
  continuous-action Gaussian policy stochastic. Once the policy locked onto a
  locally-good action sequence, std collapsed to zero and exploration stopped.
  Without exploration, every gradient update made huge policy jumps (kl=2.29,
  ~100x the safe threshold) that degraded performance further.
- **Lesson**: For continuous control with PPO, never accept the SB3 default
  `ent_coef`. The classic recipe is `ent_coef=0.01` plus a `target_kl` guardrail.

### Run 2 — Entropy-Regularized Fix

- **Changes from Run 1**: `ent_coef=0.01` (entropy bonus to keep policy
  stochastic) and `target_kl=0.05` (early-stop the gradient pass if approx_kl
  exceeds the threshold; belt-and-suspenders against another collapse).
  Everything else identical to Run 1.
- **Result**: Stable training. Late-run metrics: `std=1.13`, `approx_kl=0.04`,
  `clip_fraction=0.21`, `ep_rew_mean ~33`. The guardrail fired routinely
  ("Early stopping at step 1/2/3 due to reaching max kl: 0.08") which is the
  intended behavior — it's preventing exactly what wrecked Run 1.
- **Caveat**: Deterministic eval (single seed) returned reward 24.8, *below*
  the stochastic training mean of 33. Two interpretations: (a) single-seed
  variance, or (b) the policy is leaning on its action noise and hasn't
  committed to a clean strategy. Discovering Run 3's reward bug subsumed this.
- **Operational notes**: Resuming from a checkpoint with `PPO.load(...)` plus
  `learn(reset_num_timesteps=False)` keeps the step counter going so logs and
  checkpoint filenames continue the series naturally.

### Reward-Function Shape Bug (the real finding)

- **Symptom**: Best handcoded baseline (`coverage_path`) reaches reward ~53
  while reporting only `improvement_pct ~16–26%` and `visited_nav_pct ~1.58%`.
  These numbers don't add up — a coverage policy that supposedly resurfaces
  16% of the rink should be scoring better than a PPO policy that doesn't.
- **Cause**: `RinkEnv._compute_reward()` (rink_env.py:694) hands out +0.1 for
  visiting a new cell in `visited_mask` and -0.05 for repeats. But
  `visited_mask` is a *center-point trace* — it only marks the single nav-grid
  cell under the Zamboni's center, not the actual brush-width footprint
  (`brush_visited_mask`). Driving a wide methodical resurface pattern earns
  no more reward than zigzagging through the same unique cells. Worse, the
  ice-flattening signal (`total_elevation_change / 1000.0`) is in the 1e-6
  per-step range — rounding error compared to the +0.1 cell-visit bonus.
- **Math check**: `coverage_path` visited ~620 unique center cells × 0.1
  = +62, minus ~6 time penalty = 56. Reported total: 53. So even the "good"
  baseline is being measured almost entirely on path diversity, not on actual
  ice coverage.
- **Why this matters**: This bug is already documented as Item 4 in
  rinkgym's `docs/gymnasium-api-conformance.md` (the `visited_mask` vs
  `brush_visited_mask` distinction in metrics) — but it's *also* in the reward
  function, not just the metrics. Both Run 1 and Run 2 were optimizing the
  wrong objective. The "fix" for the next run is to (a) swap the exploration
  bonus to `brush_visited_mask`, (b) weight `total_elevation_change` enough
  that it's a real signal rather than rounding error, and (c) audit the
  `collision_penalty` sign — the contract should clearly penalize wall hits.
- **Lesson**: Before tuning RL hyperparameters, *read the reward function*.
  A policy can only learn to maximize what you actually reward, and "what you
  actually reward" can drift from "what you mean to reward" silently.
