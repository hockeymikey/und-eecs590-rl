"""Nuclear Reactor Cadmium Rod Control Environment.

POMDP-like environment where the agent controls cadmium rods to keep
reactor reactivity in a productive range while avoiding meltdown.
The agent observes only noisy, discretized readings of the true reactivity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from numpy.random import Generator, default_rng


@dataclass
class ReactorConfig:
    """All tuneable environment parameters."""

    # Reactivity bounds
    mu_min: float = 0.0
    mu_max: float = 10.0

    # Productive range
    mu_lo: float = 3.0
    mu_hi: float = 7.0

    # Drift threshold
    mu_hot: float = 6.0

    # Action parameters
    k: int = 2  # actions in {-k, ..., +k}
    alpha: float = 0.5  # rod effectiveness per increment

    # Drift magnitude
    delta: float = 0.2

    # Noise
    sigma_obs: float = 0.5  # observation noise σ
    sigma_process: float = 0.3  # process noise σ_T
    sigma_reward: float = 0.5  # reward noise σ_R

    # Reward
    meltdown_penalty: float = -50.0
    rod_cost: float = 0.1

    # Discretisation
    n_bins: int = 20

    # Episode
    horizon: int = 200
    gamma: float = 0.95

    # Initial condition: μ_0 drawn from U[mu_min, mu_min + init_spread]
    init_spread: float = 1.0


class ReactorEnv:
    """Reactor cadmium-rod control environment.

    Hidden state: true mean reactivity μ_t ∈ [μ_min, μ_max].
    Observation: noisy reading z_t ~ N(μ_t, σ²), discretised into bins.
    """

    def __init__(self, cfg: ReactorConfig | None = None) -> None:
        self.cfg = cfg or ReactorConfig()
        c = self.cfg

        # Precompute bin edges and centres
        self._bin_edges = np.linspace(c.mu_min, c.mu_max, c.n_bins + 1)
        self._bin_centers = 0.5 * (self._bin_edges[:-1] + self._bin_edges[1:])

        # Action list: [-k, ..., 0, ..., +k]
        self._actions = list(range(-c.k, c.k + 1))

        # Public attributes used by agents
        self.num_states: int = c.n_bins
        self.num_actions: int = len(self._actions)
        self.gamma: float = c.gamma
        self.start_index: int = 0  # updated each reset

        # Internal state
        self._mu: float = c.mu_min
        self._step: int = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def action_value(self, idx: int) -> int:
        """Map action index to signed rod increment."""
        return self._actions[idx]

    def obs_to_bin(self, z: float) -> int:
        """Discretise a continuous observation into a bin index."""
        idx = int(np.digitize(z, self._bin_edges)) - 1
        return max(0, min(self.cfg.n_bins - 1, idx))

    def bin_centers(self) -> np.ndarray:
        return self._bin_centers.copy()

    def action_labels(self) -> list[str]:
        return [str(a) for a in self._actions]

    def is_terminal_state(self, obs_bin: int) -> bool:
        """Heuristic: the highest bin maps to the meltdown region."""
        return obs_bin == self.cfg.n_bins - 1

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, rng: Generator | None = None) -> int:
        """Start a new episode.  Returns initial observation bin."""
        if rng is None:
            rng = default_rng()
        c = self.cfg
        self._mu = c.mu_min + rng.uniform() * c.init_spread
        self._step = 0

        z0 = rng.normal(self._mu, c.sigma_obs)
        obs_bin = self.obs_to_bin(z0)
        self.start_index = obs_bin
        return obs_bin

    def simulate_step(
        self, obs_bin: int, action_idx: int, rng: Generator | None = None
    ) -> Tuple[int, float, bool]:
        """Take one step.

        Parameters
        ----------
        obs_bin : int
            Current observation bin (kept for API compatibility; ignored
            internally because the env tracks hidden μ).
        action_idx : int
            Index into the action list.
        rng : Generator, optional
            Numpy random generator.

        Returns
        -------
        next_obs_bin : int
        reward : float
        done : bool
        """
        if rng is None:
            rng = default_rng()
        c = self.cfg
        a = self.action_value(action_idx)

        # --- Dynamics: μ_{t+1} = clip(μ_t - α·a_t + d(μ_t) + ε_t) ---
        drift = c.delta if self._mu >= c.mu_hot else 0.0
        process_noise = rng.normal(0.0, c.sigma_process)
        mu_next = self._mu - c.alpha * a + drift + process_noise
        mu_next = np.clip(mu_next, c.mu_min, c.mu_max)

        # --- Reward (based on μ_t before transition, per assignment) ---
        reward_mean = self._expected_reward(self._mu, a)
        reward = rng.normal(reward_mean, c.sigma_reward)

        # --- Update hidden state ---
        self._mu = mu_next
        self._step += 1

        # --- Meltdown check ---
        done = False
        if self._mu >= c.mu_max:
            reward = rng.normal(c.meltdown_penalty, c.sigma_reward)
            done = True

        # --- Horizon check ---
        if self._step >= c.horizon:
            done = True

        # --- Observation ---
        z_next = rng.normal(self._mu, c.sigma_obs)
        next_obs_bin = self.obs_to_bin(z_next)

        return next_obs_bin, reward, done

    # ------------------------------------------------------------------
    # Reward helper
    # ------------------------------------------------------------------

    def _expected_reward(self, mu: float, a: int) -> float:
        """Expected reward R(μ, a) before adding noise."""
        c = self.cfg
        if mu >= c.mu_max:
            return c.meltdown_penalty
        elif c.mu_lo <= mu <= c.mu_hi:
            power = mu - c.mu_lo  # w(μ) = μ - μ_lo
            return power - c.rod_cost * abs(a)
        else:
            # Too cold — no power generation
            return -c.rod_cost * abs(a)
