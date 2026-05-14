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

- **Observation space**: `ZambGymEnv` outputs a 3-channel image at nav-grid
  resolution (~129×304×3 by default): damage magnitude (normalised), agent-
  pose hot pixel, and refreeze-progress in `[0, 1]`. This rules out tabular
  methods entirely and requires CNN-based function approximation.

- **Continuous action space**: Throttle and steering are both continuous in
  [-1, 1]. This eliminates DQN (discrete actions only) as a primary algorithm
  choice and points toward policy gradient methods (PPO, SAC, DDPG, TD3).

- **Adaptation boundary**: The Minis env is a stripped-down, modified
  adaptation of a private prototype that lives outside this course repo.
  The submitted copy is self-contained (no imports back into that workspace),
  uses a deliberately simpler linear refreeze placeholder, and pins
  course-specific reward coefficients. The frozen coverage-path teacher npz at
  `assets/teacher/coverage_path_v1.npz` is the only artefact crossing the
  workspace boundary, and only as data — the path-generation algorithm
  itself stays out of this repo.

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

Headless PPO runs on an earlier prototype environment using SB3 with a custom
CNN extractor. Three stories, in order of discovery.

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
- **Cause**: the prototype reward handed out +0.1 for visiting a new cell in
  `visited_mask` and -0.05 for repeats. But
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
- **Why this matters**: The same center-trace-vs-brush-footprint distinction
  had already shown up in metrics, but it was also present in the reward
  function. Both Run 1 and Run 2 were optimizing the wrong objective. The fix
  for the submitted env is to reward brush-footprint coverage directly, weight
  damage reduction enough to be a real signal, and keep wall contacts clearly
  negative.
- **Lesson**: Before tuning RL hyperparameters, *read the reward function*.
  A policy can only learn to maximize what you actually reward, and "what you
  actually reward" can drift from "what you mean to reward" silently.

## V3 — Isaac Sim platform debugging

Setting up Isaac Lab 2.3.2 on Arch Linux for the V3 Isaac Sim attempt
surfaced four distinct upstream bugs in the NVIDIA/Omniverse stack. Each
took meaningful debugging time; together they consumed most of a day. Full
narrative and exact commands live in `docs/isaacsim_setup_issues.md`; summaries
here.

- **Isaac Lab `flatdict==4.0.1` build failure on modern pip**. The main
  `isaaclab` package pulls `flatdict==4.0.1` (last released 2019) as a
  transitive dependency. Modern pip's isolated build environment doesn't
  ship `pkg_resources` in its default setuptools, and flatdict's `setup.py`
  imports it. The build fails. **The killer detail**: this failure aborts
  the `isaaclab` install silently — `./isaaclab.sh --install` continues
  through the *other* subpackages (`isaaclab_rl`, `isaaclab_tasks`,
  `isaaclab_assets`, `isaaclab_mimic`) and reports success, leaving you
  with everything installed *except* the top-level `isaaclab` module that
  every entry-point script needs. Symptom: `ModuleNotFoundError: No module
  named 'isaaclab'` after a "successful" install. **Fix**: downgrade pip
  and setuptools before the install (`pip==23 setuptools==65`), pre-install
  flatdict in that older env, then run `./isaaclab.sh --install`. Tracked
  upstream as isaac-sim/IsaacLab #4576 (proposal to drop flatdict) and #4577
  (the build failure itself).
- **NVIDIA Container Toolkit 1.19 fails to mount the Vulkan ICD on Arch**.
  With `NVIDIA_DRIVER_CAPABILITIES=all` set, the toolkit mounts the NVIDIA
  GPU userspace libraries into the container (so `nvidia-smi` works) but
  *doesn't* mount `/usr/share/vulkan/icd.d/nvidia_icd.json`. Inside the
  container Vulkan loader can't find the NVIDIA ICD; Isaac Sim dies at
  `vkCreateInstance failed. Vulkan 1.1 is not supported`. The Vulkan
  `.so` files are mounted; only the ICD JSON descriptor is missing.
  Confusingly, the error message points at the driver, not at the
  container-mount layer. **Fix**: add an explicit bind mount in
  `docker-compose.local.yaml` for `nvidia_icd.json` (and the matching EGL
  vendor descriptor). One yaml line; took hours to identify.
- **NVIDIA driver 595.x not validated for Isaac Sim 5.1**. The headline bug.
  After fixing flatdict and the Vulkan ICD mount, Isaac Sim loaded all
  ~140 of its Kit extensions, reached "app ready", then segfaulted in
  `librtx.scenedb.plugin.so` (the RTX scene-database plugin) with a null-
  pointer write inside a `std::vector::_M_realloc_insert` call —
  `error 6 at 0x3380`, the classic "deref a null pointer at a struct
  member offset." Wasted hours theorizing about laptop-Ampere feature
  enumeration, PCIe Gen 1 link width, IOMMU, and hybrid-GPU Vulkan device
  selection. The actual cause was much simpler: NVIDIA driver 595.x is
  outside the validated driver list for Isaac Sim 5.1; confirmed in a
  github issue thread reporting the same crash on RTX 4070, 5070 Ti, 5080,
  and 5090 — i.e., not hardware-specific. **Fix**: downgrade to driver
  `580.65.06` (Linux) via the Arch Linux Archive
  (`archive.archlinux.org/packages/n/nvidia-utils/`), pin in
  `pacman.conf` so `pacman -Syu` doesn't re-upgrade. **Lesson**: when a
  closed-source binary segfaults in a way that looks hardware-specific,
  search for reports on workstation hardware too. Mobile-GPU theories
  burned a lot of time on what turned out to be a universal driver
  incompatibility.
- **Bind-mount + editable install shadowing**. Isaac Lab's docker-compose
  bind-mounts the host's `source/` directory into the container at
  `/workspace/isaaclab/source/`. The Dockerfile runs
  `./isaaclab.sh --install` at build time, which writes `.egg-info`
  directories into `source/`. At runtime, those build-time `.egg-info`s
  get shadowed by the host bind mount, and editable-install metadata in
  the container's site-packages points at paths that no longer match the
  source layout — especially if you switched IsaacLab git tags between
  build and run. Net effect: install metadata exists but resolves to
  empty/wrong dirs. **Lesson**: with bind-mounted source dirs, editable
  installs that write metadata into the source tree at build time are
  fragile across container restarts. Either re-run the install at runtime
  (and let the metadata land on the bind-mounted host filesystem so it
  persists), or build the install into a non-mounted path inside the
  container image.

The four bugs above are also why I chose to keep the Isaac Sim attempt
*present but deliberately minimal* in the V3 submission. The platform-
debugging effort matches V3.md's explicit framing that "a student
struggling to implement basic functionality in Isaac sim will perform
better than a student who took an afternoon to train a single super-human
agent for a gym-like environment." The actual Isaac env code is small; what
made it a multi-day effort was the path *to* running code, not writing it.
