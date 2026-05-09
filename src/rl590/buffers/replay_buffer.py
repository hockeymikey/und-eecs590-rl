"""Replay buffer for off-policy deep RL algorithms.

A replay buffer stores transitions (s, a, r, s', done) from environment
interactions. Off-policy algorithms like DQN, DDPG, TD3, and SAC sample
random minibatches from this buffer during training.

Why a replay buffer?
    - Breaks temporal correlations: consecutive transitions are highly
      correlated, which hurts gradient descent. Random sampling decorrelates.
    - Sample efficiency: each transition can be used for many gradient steps,
      unlike on-policy methods (PPO) which discard data after each update.
    - Circular/ring buffer: once full, new transitions overwrite the oldest,
      keeping memory bounded while favoring recent experience.

Note: PPO does NOT use this buffer (it's on-policy and uses RolloutBuffer
instead). This is for future off-policy algorithms (DQN, SAC, TD3, etc.)
and for the replay data storage requirement.

Storage format on disk:
    replay_data/{algorithm}/{task}/{timestamp}_{policy_version}.npz
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
from numpy.random import Generator, default_rng

from rl590.primitives import BatchKey, NpzKey


@dataclass
class ReplayBufferConfig:
    """Configuration for the replay buffer."""

    max_size: int = 100_000          # max transitions stored
    obs_shape: tuple = (129, 304, 2)  # observation shape
    action_dim: int = 2               # continuous action dimensions


class ReplayBuffer:
    """Fixed-size circular replay buffer.

    Stores transitions in pre-allocated numpy arrays for efficiency.
    When full, new transitions overwrite the oldest (FIFO).

    Parameters
    ----------
    config : ReplayBufferConfig
        Buffer capacity and data shapes.
    """

    def __init__(self, config: ReplayBufferConfig | None = None) -> None:
        self.config = config or ReplayBufferConfig()
        cfg = self.config

        # Pre-allocate arrays
        self.observations = np.zeros((cfg.max_size, *cfg.obs_shape), dtype=np.float32)
        self.actions = np.zeros((cfg.max_size, cfg.action_dim), dtype=np.float32)
        self.rewards = np.zeros(cfg.max_size, dtype=np.float32)
        self.next_observations = np.zeros((cfg.max_size, *cfg.obs_shape), dtype=np.float32)
        self.dones = np.zeros(cfg.max_size, dtype=np.bool_)

        self._pos = 0       # next write position
        self._size = 0       # current number of stored transitions
        self._total_added = 0  # lifetime count (for stats)

    def __len__(self) -> int:
        return self._size

    @property
    def is_full(self) -> bool:
        return self._size >= self.config.max_size

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store a single transition.

        Overwrites the oldest transition if the buffer is full.
        """
        self.observations[self._pos] = obs
        self.actions[self._pos] = action
        self.rewards[self._pos] = reward
        self.next_observations[self._pos] = next_obs
        self.dones[self._pos] = done

        self._pos = (self._pos + 1) % self.config.max_size
        self._size = min(self._size + 1, self.config.max_size)
        self._total_added += 1

    def sample(
        self, batch_size: int, rng: Generator | None = None
    ) -> Dict[str, np.ndarray]:
        """Sample a random minibatch of transitions.

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample.
        rng : Generator, optional
            Numpy random generator for reproducibility.

        Returns
        -------
        dict with keys: obs, actions, rewards, next_obs, dones
            Each value is a numpy array with batch_size rows.
        """
        assert self._size >= batch_size, (
            f"Not enough transitions ({self._size}) to sample batch of {batch_size}"
        )
        if rng is None:
            rng = default_rng()

        indices = rng.integers(0, self._size, size=batch_size)
        return {
            BatchKey.OBS: self.observations[indices],
            BatchKey.ACTIONS: self.actions[indices],
            BatchKey.REWARDS: self.rewards[indices],
            BatchKey.NEXT_OBS: self.next_observations[indices],
            BatchKey.DONES: self.dones[indices],
        }

    def save(self, path: str | Path, metadata: dict | None = None) -> Path:
        """Save buffer contents to a .npz file.

        Only saves the filled portion of the buffer (not empty slots).

        Parameters
        ----------
        path : str or Path
            Output file path (without .npz extension).
        metadata : dict, optional
            Extra info to store (algorithm name, policy version, etc.)
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        size = self._size
        meta = {
            "size": size,
            "pos": self._pos,
            "total_added": self._total_added,
            "max_size": self.config.max_size,
            **(metadata or {}),
        }

        np.savez_compressed(
            out,
            **{
                NpzKey.OBSERVATIONS: self.observations[:size],
                NpzKey.ACTIONS: self.actions[:size],
                NpzKey.REWARDS: self.rewards[:size],
                NpzKey.NEXT_OBSERVATIONS: self.next_observations[:size],
                NpzKey.DONES: self.dones[:size],
                NpzKey.METADATA: np.array(json.dumps(meta)),
            },
        )
        return out

    def load(self, path: str | Path) -> dict:
        """Load buffer contents from a .npz file.

        Returns the metadata dict stored with the file.
        """
        data = np.load(path, allow_pickle=False)
        size = len(data[NpzKey.REWARDS])

        assert size <= self.config.max_size, (
            f"Saved buffer ({size}) exceeds configured max_size ({self.config.max_size})"
        )

        self.observations[:size] = data[NpzKey.OBSERVATIONS]
        self.actions[:size] = data[NpzKey.ACTIONS]
        self.rewards[:size] = data[NpzKey.REWARDS]
        self.next_observations[:size] = data[NpzKey.NEXT_OBSERVATIONS]
        self.dones[:size] = data[NpzKey.DONES]

        meta = json.loads(str(data[NpzKey.METADATA]))
        self._size = size
        self._pos = meta.get("pos", size % self.config.max_size)
        self._total_added = meta.get("total_added", size)

        return meta

    def stats(self) -> Dict[str, int | float]:
        """Return buffer statistics."""
        obs_bytes = self.observations[:self._size].nbytes
        total_bytes = (
            obs_bytes * 2  # obs + next_obs
            + self.actions[:self._size].nbytes
            + self.rewards[:self._size].nbytes
            + self.dones[:self._size].nbytes
        )
        return {
            "size": self._size,
            "capacity": self.config.max_size,
            "utilization": self._size / self.config.max_size,
            "total_added": self._total_added,
            "memory_mb": total_bytes / (1024 * 1024),
        }
