"""Tabular TD agents: SARSA(λ) with replacing traces and Q-learning.

Both operate on discretised observation bins from ReactorEnv.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from numpy.random import Generator, default_rng

from reactor_env import ReactorEnv


@dataclass
class TDConfig:
    algorithm: str = "sarsa_lambda"  # "sarsa_lambda" or "qlearning"

    # Learning
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon: float = 0.1
    epsilon_decay: float = 1.0  # multiplicative per-episode decay
    epsilon_min: float = 0.01

    # Eligibility traces (SARSA only)
    lam: float = 0.8
    trace_type: str = "replacing"  # "replacing" or "accumulating"

    # Training
    episodes: int = 2000
    max_steps_per_episode: int = 200
    eval_episodes: int = 50
    seed: int = 0

    # Non-stationarity investigation (Q4b)
    alpha_decay: float = 0.0  # if >0, α decays by this factor per episode


class TDAgent:
    """Tabular TD agent for the reactor environment."""

    def __init__(self, env: ReactorEnv, config: TDConfig | None = None) -> None:
        self.env = env
        self.config = config or TDConfig()
        self.Q = np.zeros((env.num_states, env.num_actions))

        # Episode tracking
        self.episode_returns: List[float] = []
        self.episode_meltdowns: List[bool] = []
        self.episode_lengths: List[int] = []

    # ------------------------------------------------------------------
    # ε-greedy policy
    # ------------------------------------------------------------------

    def _epsilon_greedy(self, state: int, epsilon: float, rng: Generator) -> int:
        if rng.random() < epsilon:
            return rng.integers(self.env.num_actions)
        return int(np.argmax(self.Q[state]))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, float | int | str]:
        """Run training and return summary statistics."""
        if self.config.algorithm == "sarsa_lambda":
            self._train_sarsa_lambda()
        elif self.config.algorithm == "qlearning":
            self._train_qlearning()
        else:
            raise ValueError(f"Unknown algorithm: {self.config.algorithm}")

        return {
            "algorithm": self.config.algorithm,
            "episodes": len(self.episode_returns),
            "mean_return_last100": float(np.mean(self.episode_returns[-100:])),
            "meltdown_rate_last100": float(np.mean(self.episode_meltdowns[-100:])),
        }

    def _train_qlearning(self) -> None:
        cfg = self.config
        rng = default_rng(cfg.seed)
        eps = cfg.epsilon
        alpha = cfg.alpha

        for ep in range(cfg.episodes):
            state = self.env.reset(rng)
            total_reward = 0.0
            meltdown = False

            for step in range(cfg.max_steps_per_episode):
                action = self._epsilon_greedy(state, eps, rng)
                next_state, reward, done = self.env.simulate_step(state, action, rng)
                total_reward += reward

                # TD(0) off-policy update
                td_target = reward + cfg.gamma * np.max(self.Q[next_state]) * (1 - done)
                td_error = td_target - self.Q[state, action]
                self.Q[state, action] += alpha * td_error

                state = next_state
                if done:
                    # Check if meltdown (env hit mu_max)
                    if step < cfg.max_steps_per_episode - 1:
                        meltdown = True
                    break

            self.episode_returns.append(total_reward)
            self.episode_meltdowns.append(meltdown)
            self.episode_lengths.append(step + 1)

            eps = max(cfg.epsilon_min, eps * cfg.epsilon_decay)
            if cfg.alpha_decay > 0:
                alpha = alpha * (1.0 - cfg.alpha_decay)

    def _train_sarsa_lambda(self) -> None:
        cfg = self.config
        rng = default_rng(cfg.seed)
        eps = cfg.epsilon
        alpha = cfg.alpha

        for ep in range(cfg.episodes):
            state = self.env.reset(rng)
            action = self._epsilon_greedy(state, eps, rng)
            E = np.zeros_like(self.Q)  # eligibility traces
            total_reward = 0.0
            meltdown = False

            for step in range(cfg.max_steps_per_episode):
                next_state, reward, done = self.env.simulate_step(state, action, rng)
                total_reward += reward
                next_action = self._epsilon_greedy(next_state, eps, rng)

                # TD error
                q_next = self.Q[next_state, next_action] * (1 - done)
                td_error = reward + cfg.gamma * q_next - self.Q[state, action]

                # Update traces
                if cfg.trace_type == "replacing":
                    E[state, action] = 1.0
                else:  # accumulating
                    E[state, action] += 1.0

                # Update Q and decay traces
                self.Q += alpha * td_error * E
                E *= cfg.gamma * cfg.lam

                state = next_state
                action = next_action

                if done:
                    if step < cfg.max_steps_per_episode - 1:
                        meltdown = True
                    break

            self.episode_returns.append(total_reward)
            self.episode_meltdowns.append(meltdown)
            self.episode_lengths.append(step + 1)

            eps = max(cfg.epsilon_min, eps * cfg.epsilon_decay)
            if cfg.alpha_decay > 0:
                alpha = alpha * (1.0 - cfg.alpha_decay)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, episodes: int | None = None, seed: int = 9999) -> Dict[str, float]:
        """Evaluate greedy policy (ε=0)."""
        n = episodes or self.config.eval_episodes
        rng = default_rng(seed)
        returns, meltdowns, lengths = [], [], []

        for _ in range(n):
            state = self.env.reset(rng)
            total_reward = 0.0
            meltdown = False
            for step in range(self.config.max_steps_per_episode):
                action = int(np.argmax(self.Q[state]))
                next_state, reward, done = self.env.simulate_step(state, action, rng)
                total_reward += reward
                state = next_state
                if done:
                    if step < self.config.max_steps_per_episode - 1:
                        meltdown = True
                    break
            returns.append(total_reward)
            meltdowns.append(meltdown)
            lengths.append(step + 1)

        return {
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "meltdown_rate": float(np.mean(meltdowns)),
            "mean_length": float(np.mean(lengths)),
        }

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "Q": self.Q,
            "episode_returns": np.array(self.episode_returns),
            "episode_meltdowns": np.array(self.episode_meltdowns, dtype=np.int8),
            "episode_lengths": np.array(self.episode_lengths),
            "metadata": np.array(json.dumps(asdict(self.config))),
        }
        np.savez(out, **payload)
        return out

    def load(self, path: str | Path) -> None:
        data = np.load(path, allow_pickle=False)
        self.Q = data["Q"]
        self.episode_returns = data["episode_returns"].tolist()
        self.episode_meltdowns = [bool(m) for m in data["episode_meltdowns"]]
        self.episode_lengths = data["episode_lengths"].tolist()
