"""Linear function approximation agent with RBF features.

Semi-gradient SARSA(λ) for the reactor environment (Challenge Q4a).
Q̂(z, a; w) = w^T φ(z, a) where φ uses radial basis functions tiled
over the observation space with one-hot action encoding.
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
class FAConfig:
    """Configuration for the linear FA agent."""

    # RBF parameters
    n_rbf_centers: int = 20  # number of RBF centres along obs axis
    rbf_width: float = 0.0  # auto-set if 0 (based on centre spacing)

    # Learning
    alpha: float = 0.001  # smaller step size for FA
    gamma: float = 0.95
    epsilon: float = 0.1
    epsilon_decay: float = 1.0
    epsilon_min: float = 0.01

    # Traces
    lam: float = 0.8

    # Training
    episodes: int = 2000
    max_steps_per_episode: int = 200
    eval_episodes: int = 50
    seed: int = 0


class FAAgent:
    """Linear function approximation agent with RBF features."""

    def __init__(self, env: ReactorEnv, config: FAConfig | None = None) -> None:
        self.env = env
        self.config = config or FAConfig()
        cfg = self.config

        # RBF centres evenly spaced across observation range
        self._centers = np.linspace(
            env.cfg.mu_min, env.cfg.mu_max, cfg.n_rbf_centers
        )
        # Auto-set width based on spacing
        if cfg.rbf_width <= 0:
            spacing = self._centers[1] - self._centers[0]
            self._width = spacing  # σ of each RBF
        else:
            self._width = cfg.rbf_width

        # Weight vector: one block of n_rbf_centers per action
        self.n_features = cfg.n_rbf_centers * env.num_actions
        self.w = np.zeros(self.n_features)

        # Tracking
        self.episode_returns: List[float] = []
        self.episode_meltdowns: List[bool] = []
        self.episode_lengths: List[int] = []

    # ------------------------------------------------------------------
    # Feature construction
    # ------------------------------------------------------------------

    def _rbf_activations(self, z: float) -> np.ndarray:
        """Compute RBF activations for a continuous observation value."""
        return np.exp(-0.5 * ((z - self._centers) / self._width) ** 2)

    def _features(self, obs_bin: int, action_idx: int) -> np.ndarray:
        """Construct feature vector φ(z, a).

        The vector has length n_rbf_centers * num_actions.
        Only the block for the given action is nonzero (action tiling).
        """
        z = self.env.bin_centers()[obs_bin]
        phi = np.zeros(self.n_features)
        start = action_idx * self.config.n_rbf_centers
        end = start + self.config.n_rbf_centers
        phi[start:end] = self._rbf_activations(z)
        return phi

    def _q_hat(self, obs_bin: int, action_idx: int) -> float:
        """Q̂(z, a; w) = w^T φ(z, a)."""
        return float(self.w @ self._features(obs_bin, action_idx))

    def _q_all_actions(self, obs_bin: int) -> np.ndarray:
        """Q̂ for all actions at a given obs bin."""
        return np.array([self._q_hat(obs_bin, a) for a in range(self.env.num_actions)])

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------

    def _epsilon_greedy(self, obs_bin: int, epsilon: float, rng: Generator) -> int:
        if rng.random() < epsilon:
            return rng.integers(self.env.num_actions)
        return int(np.argmax(self._q_all_actions(obs_bin)))

    # ------------------------------------------------------------------
    # Training: semi-gradient SARSA(λ)
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, float | int | str]:
        cfg = self.config
        rng = default_rng(cfg.seed)
        eps = cfg.epsilon

        for ep in range(cfg.episodes):
            state = self.env.reset(rng)
            action = self._epsilon_greedy(state, eps, rng)
            e = np.zeros(self.n_features)  # eligibility trace
            total_reward = 0.0
            meltdown = False

            for step in range(cfg.max_steps_per_episode):
                next_state, reward, done = self.env.simulate_step(state, action, rng)
                total_reward += reward
                next_action = self._epsilon_greedy(next_state, eps, rng)

                q_next = self._q_hat(next_state, next_action) * (1 - done)
                td_error = reward + cfg.gamma * q_next - self._q_hat(state, action)

                # Update trace (replacing: cap feature contribution at 1)
                phi = self._features(state, action)
                e = cfg.gamma * cfg.lam * e + phi

                # Semi-gradient update
                self.w += cfg.alpha * td_error * e

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

        return {
            "algorithm": "fa_sarsa_lambda",
            "episodes": len(self.episode_returns),
            "mean_return_last100": float(np.mean(self.episode_returns[-100:])),
            "meltdown_rate_last100": float(np.mean(self.episode_meltdowns[-100:])),
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, episodes: int | None = None, seed: int = 9999) -> Dict[str, float]:
        n = episodes or self.config.eval_episodes
        rng = default_rng(seed)
        returns, meltdowns, lengths = [], [], []

        for _ in range(n):
            state = self.env.reset(rng)
            total_reward = 0.0
            meltdown = False
            for step in range(self.config.max_steps_per_episode):
                action = int(np.argmax(self._q_all_actions(state)))
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
    # Q-table equivalent (for plotting compatibility)
    # ------------------------------------------------------------------

    def q_table(self) -> np.ndarray:
        """Build a (num_states, num_actions) table of Q̂ values."""
        Q = np.zeros((self.env.num_states, self.env.num_actions))
        for s in range(self.env.num_states):
            for a in range(self.env.num_actions):
                Q[s, a] = self._q_hat(s, a)
        return Q

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out,
            w=self.w,
            centers=self._centers,
            episode_returns=np.array(self.episode_returns),
            episode_meltdowns=np.array(self.episode_meltdowns, dtype=np.int8),
            episode_lengths=np.array(self.episode_lengths),
            metadata=np.array(json.dumps(asdict(self.config))),
        )
        return out

    def load(self, path: str | Path) -> None:
        data = np.load(path, allow_pickle=False)
        self.w = data["w"]
        self._centers = data["centers"]
        self.episode_returns = data["episode_returns"].tolist()
        self.episode_meltdowns = [bool(m) for m in data["episode_meltdowns"]]
        self.episode_lengths = data["episode_lengths"].tolist()
