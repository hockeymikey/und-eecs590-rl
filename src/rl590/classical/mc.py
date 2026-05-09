"""Monte Carlo methods for tabular environments.

First-visit MC: for each (state, action) pair, only the FIRST time it
appears in an episode counts toward the return average.

Every-visit MC: every occurrence of (state, action) in an episode
contributes a return sample.

Both require completing full episodes before updating — no bootstrapping.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict

import numpy as np
from numpy.random import Generator, default_rng

from rl590.primitives import MCMethod, NpzKey


@dataclass
class MCConfig:
    """Hyperparameters for Monte Carlo control."""

    method: str = MCMethod.FIRST_VISIT

    # Learning
    gamma: float = 0.95
    epsilon: float = 0.1       # exploration rate for ε-greedy
    epsilon_decay: float = 1.0  # multiplicative decay per episode
    epsilon_min: float = 0.01

    # Training
    episodes: int = 2000
    max_steps_per_episode: int = 200
    eval_episodes: int = 50
    seed: int = 0


class MCAgent:
    """Tabular Monte Carlo control agent.

    Works with any environment that provides:
        - num_states: int
        - num_actions: int
        - gamma: float
        - reset(rng) -> int  (initial state index)
        - simulate_step(state, action, rng) -> (next_state, reward, done)
    """

    def __init__(self, env, config: MCConfig | None = None) -> None:
        self.env = env
        self.config = config or MCConfig()

        # Q-table and visit counts for averaging returns
        self.Q = np.zeros((env.num_states, env.num_actions))
        self._returns_sum = np.zeros((env.num_states, env.num_actions))
        self._returns_count = np.zeros((env.num_states, env.num_actions))

        # Episode tracking
        self.episode_returns: List[float] = []
        self.episode_lengths: List[int] = []

    def _epsilon_greedy(self, state: int, epsilon: float, rng: Generator) -> int:
        """Select action using ε-greedy policy derived from Q."""
        if rng.random() < epsilon:
            return int(rng.integers(self.env.num_actions))
        return int(np.argmax(self.Q[state]))

    def train(self) -> Dict[str, float | int | str]:
        """Run MC control for the configured number of episodes.

        The core loop:
        1. Generate a complete episode using ε-greedy policy
        2. Walk backward through the episode computing returns
        3. Update Q for each (s, a) pair visited (first-visit or every-visit)
        """
        cfg = self.config
        rng = default_rng(cfg.seed)
        eps = cfg.epsilon
        first_visit = cfg.method == MCMethod.FIRST_VISIT

        for ep in range(cfg.episodes):
            # --- Step 1: Generate a complete episode ---
            trajectory = []  # list of (state, action, reward)
            state = self.env.reset(rng)

            for step in range(cfg.max_steps_per_episode):
                action = self._epsilon_greedy(state, eps, rng)
                next_state, reward, done = self.env.simulate_step(state, action, rng)
                trajectory.append((state, action, reward))
                state = next_state
                if done:
                    break

            # --- Step 2: Walk backward, compute returns ---
            G = 0.0
            # Track which (s, a) pairs we've already seen this episode
            # (only matters for first-visit)
            visited = set()

            for t in range(len(trajectory) - 1, -1, -1):
                s_t, a_t, r_t = trajectory[t]
                G = r_t + cfg.gamma * G

                pair = (s_t, a_t)
                if first_visit and pair in visited:
                    # Skip — we only use the first occurrence
                    continue
                visited.add(pair)

                # --- Step 3: Update Q using incremental mean ---
                self._returns_sum[s_t, a_t] += G
                self._returns_count[s_t, a_t] += 1
                self.Q[s_t, a_t] = (
                    self._returns_sum[s_t, a_t] / self._returns_count[s_t, a_t]
                )

            # Track episode stats
            episode_return = sum(r for _, _, r in trajectory)
            self.episode_returns.append(episode_return)
            self.episode_lengths.append(len(trajectory))

            # Decay exploration
            eps = max(cfg.epsilon_min, eps * cfg.epsilon_decay)

        return {
            "algorithm": f"mc_{cfg.method}",
            "episodes": len(self.episode_returns),
            "mean_return_last100": float(np.mean(self.episode_returns[-100:])),
        }

    def evaluate(self, episodes: int | None = None, seed: int = 9999) -> Dict[str, float]:
        """Evaluate the greedy policy (ε=0)."""
        n = episodes or self.config.eval_episodes
        rng = default_rng(seed)
        returns = []
        lengths = []

        for _ in range(n):
            state = self.env.reset(rng)
            total_reward = 0.0
            for step in range(self.config.max_steps_per_episode):
                action = int(np.argmax(self.Q[state]))
                next_state, reward, done = self.env.simulate_step(state, action, rng)
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
        """Persist Q-table and training history."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out,
            **{
                NpzKey.Q: self.Q,
                NpzKey.RETURNS_SUM: self._returns_sum,
                NpzKey.RETURNS_COUNT: self._returns_count,
                NpzKey.EPISODE_RETURNS: np.array(self.episode_returns),
                NpzKey.EPISODE_LENGTHS: np.array(self.episode_lengths),
                NpzKey.METADATA: np.array(json.dumps(asdict(self.config))),
            },
        )
        return out

    def load(self, path: str | Path) -> None:
        """Load a previously saved agent."""
        data = np.load(path, allow_pickle=False)
        self.Q = data[NpzKey.Q]
        self._returns_sum = data[NpzKey.RETURNS_SUM]
        self._returns_count = data[NpzKey.RETURNS_COUNT]
        self.episode_returns = data[NpzKey.EPISODE_RETURNS].tolist()
        self.episode_lengths = data[NpzKey.EPISODE_LENGTHS].tolist()
