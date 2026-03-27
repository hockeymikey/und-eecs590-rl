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
