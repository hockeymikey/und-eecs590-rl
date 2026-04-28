# EECS 590: Reinforcement Learning

Course repository for mini assignments and versioned RL project work at the University of North Dakota (Spring 2026).

## Foundation Environment

The foundation environment for this project is an **autonomous ice resurfacer
(Zamboni)** built as a custom Gymnasium environment. It lives in a separate
research repository: https://gitbay.hockeymikey.com/hockeymikey/gym-zamboni

Install as an editable dependency:

```bash
git clone https://gitbay.hockeymikey.com/hockeymikey/gym-zamboni.git
pip install -e gym-zamboni/
```

**Environment characteristics:**
- **Observation**: 2-channel image (129x304x2) — ice roughness/elevation map + agent position heatmap
- **Action space**: Continuous (throttle, steering), each in [-1, 1]
- **Physics**: Dynamic bicycle model with ice friction, RK4 integration
- **Reward**: Ice improvement bonus + exploration coverage - time penalty - collision penalty
- **Goal**: Learn to resurface a full ice rink efficiently while avoiding walls

## Repository Layout

```text
.
├── src/rl590/                      # Core RL library
│   ├── classical/                  # Model-free tabular algorithms
│   │   ├── mc.py                   # Monte Carlo (first-visit / every-visit)
│   │   ├── td.py                   # TD(n) and TD(λ), forward + backward views
│   │   ├── sarsa.py                # SARSA(n) and SARSA(λ), forward + backward views
│   │   └── qlearning.py            # Q-learning (off-policy TD(0))
│   ├── deep/                       # Deep RL algorithms (PPO, etc.)
│   ├── networks/                   # Neural network architectures (CNNs for Zamboni)
│   ├── buffers/                    # Replay buffer and experience storage
│   ├── visualization/              # Saliency analysis and plotting
│   ├── dp/                         # Dynamic programming (VI / PI)
│   ├── envs/                       # Tabular environments (WindyChasm)
│   ├── agents/                     # Agent wrappers (planning agent)
│   ├── model/                      # Bayesian belief model for bootstrap RL
│   └── utils/                      # Rendering + plotting helpers
├── M1/, M2/, M3/                   # Mini project submissions (historical)
├── checkpoints/                    # Saved NN weights (gitignored, local only)
├── replay_data/                    # Raw experience data (gitignored, local only)
├── scripts/                        # CLI entry points
├── docs/                           # Assignment specs, technical challenges
├── tests/                          # Smoke tests
└── notebooks/                      # Experiment notebooks
```

## Setup

```bash
# Using uv (recommended)
uv sync

# Or using pip
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Install foundation environment (separate repo)
git clone https://gitbay.hockeymikey.com/hockeymikey/gym-zamboni.git
pip install -e gym-zamboni/
```

## Classical Algorithms

All classical algorithms work with any tabular environment providing `reset()`,
`simulate_step()`, `num_states`, and `num_actions`. Tested on WindyChasm (140
states, 3 actions) and ReactorEnv (20 states, 5 actions).

```python
from rl590.classical import MCAgent, MCConfig
from rl590.envs.windy_chasm import WindyChasmMDP

env = WindyChasmMDP()
agent = MCAgent(env, MCConfig(episodes=2000, epsilon_decay=0.999))
agent.train()
print(agent.evaluate())
```

**Implemented algorithms:**

| Algorithm | Module | Variants |
|-----------|--------|----------|
| Monte Carlo | `mc.py` | First-visit, every-visit |
| TD(n) | `td.py` | Forward-view, backward-view (with traces) |
| TD(λ) | `td.py` | Forward-view, backward-view (with traces) |
| SARSA(n) | `sarsa.py` | Forward-view, backward-view (with traces) |
| SARSA(λ) | `sarsa.py` | Forward-view, backward-view (with traces) |
| Q-learning | `qlearning.py` | Off-policy TD(0) |

## Deep RL Algorithm Justification

### Choice: PPO (Proximal Policy Optimization)

PPO is the primary deep RL algorithm for the Zamboni foundation environment.
This choice is driven by the environment's characteristics: **continuous action
space** (throttle + steering) and **high-dimensional image observations**
(129x304x2).

**Comparison to all candidate algorithms:**

| Algorithm | Fits Zamboni? | Reasoning |
|-----------|--------------|-----------|
| **DQN** | Poor | Requires discrete actions. Would need to discretize throttle/steering into bins, losing precision in a domain where smooth continuous control matters (e.g., gradual turns around rink corners). |
| **REINFORCE** | Marginal | Handles continuous actions but has high variance due to Monte Carlo returns. The Zamboni's long episodes (2000 steps) make variance a serious problem — entire episodes of good behavior get drowned by one bad collision. |
| **Vanilla Actor-Critic** | Okay | Reduces REINFORCE's variance with a critic baseline, but lacks any trust region mechanism. Unbounded policy updates can destabilize training, especially with the Zamboni's complex reward landscape (ice improvement vs. collision avoidance vs. coverage). |
| **DDPG** | Decent | Designed for continuous control, but deterministic policy + single critic makes it brittle. Known for sensitivity to hyperparameters and overestimation bias. The Zamboni's sparse collision penalties can cause catastrophic value overestimation. |
| **TD3** | Good | Fixes DDPG's overestimation with twin critics and delayed updates. A strong candidate, but off-policy methods require large replay buffers for the Zamboni's high-dimensional observations (each transition stores a 129x304x2 image). |
| **PPO** | Very Good | On-policy, so no replay buffer memory concerns for image observations. Clipped surrogate objective prevents destructive policy updates. Robust to hyperparameter choices — critical when training is expensive (Zamboni physics simulation is slow). GAE provides tunable bias-variance tradeoff. Industry-proven for robotics and continuous control. |
| **TRPO** | Good | Similar trust-region benefits to PPO but requires computing the Fisher information matrix and conjugate gradient optimization — significantly more complex to implement correctly with no clear performance advantage over PPO for this domain. |
| **SAC** | Very Good | Maximum entropy framework naturally encourages exploration, valuable for rink coverage. However, off-policy (large replay buffer for images) and adds temperature auto-tuning complexity. Strong second choice if PPO underperforms. |

**Why PPO wins for the Zamboni:**
1. **On-policy** — avoids storing millions of 129x304x2 images in a replay buffer
2. **Stable** — clipped objective prevents the catastrophic policy collapses that plague DDPG/vanilla PG in collision-heavy environments
3. **Continuous actions** — Gaussian policy outputs mean and std for throttle/steering
4. **Proven for robotics** — used in OpenAI's robotic manipulation, locomotion tasks with similar continuous control characteristics
5. **Manageable cons** — lower sample efficiency than off-policy methods, but the Zamboni simulator is deterministic and parallelizable

## Dynamic Programming (V1)

Train with policy iteration on WindyChasm:

```bash
python3 scripts/run_windy.py train --algorithm policy_iteration
```

Evaluate a saved model:

```bash
python3 scripts/run_windy.py eval --model-path artifacts/windy_best_policy.npz
```

Bootstrap from sampled transitions:

```bash
python3 scripts/run_windy.py bootstrap-model --algorithm policy_iteration
```

## Legacy Files

`M1/`, `M2/`, `M3/` are preserved as historical mini-project submissions.
`src/rl590/` is the consolidated library used for all version updates.

## Citations / Collaboration

- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.)
- Schulman et al., *Proximal Policy Optimization Algorithms* (2017)
- Farama Foundation, Gymnasium
- 
