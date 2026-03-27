"""Q-learning: off-policy TD(0) control.

Update rule:
    Q(S, A) ← Q(S, A) + α [R + γ max_a Q(S', a) - Q(S, A)]

Key property: the update target uses max over next actions, NOT the
action actually taken by the policy. This makes it off-policy — the
behavior policy (ε-greedy) can differ from the target policy (greedy).

Compared to SARSA:
    - SARSA updates toward Q(S', A') where A' is the action actually taken
    - Q-learning updates toward max_a Q(S', a) regardless of what A' was
    - Q-learning converges to Q* (optimal) even under ε-greedy exploration
    - SARSA converges to the Q-values of the ε-greedy policy itself
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict

import numpy as np
from numpy.random import Generator, default_rng


@dataclass
class QLearningConfig:
    """Hyperparameters for Q-learning."""

    alpha: float = 0.1         # step size
    gamma: float = 0.95        # discount factor
    epsilon: float = 0.1       # exploration rate
    epsilon_decay: float = 1.0
    epsilon_min: float = 0.01

    episodes: int = 2000
    max_steps_per_episode: int = 200
    eval_episodes: int = 50
    seed: int = 0


class QLearningAgent:
    """Tabular Q-learning agent.

    Works with any environment providing:
        num_states, num_actions, gamma,
        reset(rng) -> state, simulate_step(state, action, rng) -> (s', r, done)
    """

    def __init__(self, env, config: QLearningConfig | None = None) -> None:
        self.env = env
        self.config = config or QLearningConfig()
        self.Q = np.zeros((env.num_states, env.num_actions))

        self.episode_returns: List[float] = []
        self.episode_lengths: List[int] = []

    def _epsilon_greedy(self, state: int, epsilon: float, rng: Generator) -> int:
        if rng.random() < epsilon:
            return int(rng.integers(self.env.num_actions))
        return int(np.argmax(self.Q[state]))

    def train(self) -> Dict[str, float | int | str]:
        """Train using Q-learning.

        At each step:
            1. Pick action A from S using ε-greedy (behavior policy)
            2. Take action, observe R, S'
            3. Update: Q(S,A) += α[R + γ·max_a Q(S',a) - Q(S,A)]
               Note: we use MAX — this is what makes it off-policy
            4. S ← S'
        """
        cfg = self.config
        rng = default_rng(cfg.seed)
        eps = cfg.epsilon
        alpha = cfg.alpha

        for ep in range(cfg.episodes):
            state = self.env.reset(rng)
            total_reward = 0.0

            for step in range(cfg.max_steps_per_episode):
                action = self._epsilon_greedy(state, eps, rng)
                next_state, reward, done = self.env.simulate_step(
                    state, action, rng
                )
                total_reward += reward

                # Off-policy TD(0) update: target uses max over next actions
                td_target = reward + cfg.gamma * np.max(self.Q[next_state]) * (1 - done)
                td_error = td_target - self.Q[state, action]
                self.Q[state, action] += alpha * td_error

                state = next_state
                if done:
                    break

            self.episode_returns.append(total_reward)
            self.episode_lengths.append(step + 1)
            eps = max(cfg.epsilon_min, eps * cfg.epsilon_decay)

        return {
            "algorithm": "qlearning",
            "episodes": len(self.episode_returns),
            "mean_return_last100": float(np.mean(self.episode_returns[-100:])),
        }

    def evaluate(self, episodes: int | None = None, seed: int = 9999) -> Dict[str, float]:
        """Evaluate greedy policy (ε=0)."""
        n = episodes or self.config.eval_episodes
        rng = default_rng(seed)
        returns = []
        lengths = []

        for _ in range(n):
            state = self.env.reset(rng)
            total_reward = 0.0
            for step in range(self.config.max_steps_per_episode):
                action = int(np.argmax(self.Q[state]))
                next_state, reward, done = self.env.simulate_step(
                    state, action, rng
                )
                total_reward += reward
                state = next_state
                if done:
                    break
            returns.append(total_reward)
            lengths.append(step + 1)

        return {
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "mean_length": float(np.mean(lengths)),
        }

    def policy(self) -> np.ndarray:
        """Extract greedy policy from Q-table."""
        return np.argmax(self.Q, axis=1)

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out,
            Q=self.Q,
            episode_returns=np.array(self.episode_returns),
            episode_lengths=np.array(self.episode_lengths),
            metadata=np.array(json.dumps(asdict(self.config))),
        )
        return out

    def load(self, path: str | Path) -> None:
        data = np.load(path, allow_pickle=False)
        self.Q = data["Q"]
        self.episode_returns = data["episode_returns"].tolist()
        self.episode_lengths = data["episode_lengths"].tolist()
