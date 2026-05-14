"""Gymnasium wrappers around ZambIsaacEnv.

These are host-safe — they don't import isaaclab. Use:

  - `IsaacSingleEnvWrapper` to expose a num_envs=1 ZambIsaacEnv as a single-env
    Gymnasium env, suitable for the from-scratch PPO in `rl590.deep.ppo`.
  - For SB3 with N parallel envs, use Isaac Lab's built-in
    `isaaclab_rl.sb3.Sb3VecEnvWrapper` directly on the env. No custom
    wrapper needed.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces


class IsaacSingleEnvWrapper(gym.Env):
    """Adapt a num_envs=1 ZambIsaacEnv to single-env Gymnasium API.

    Squeezes the leading env dimension, converts torch ↔ numpy, and reshapes
    the flat observation back to (H, W, C) for image-CNN policies.
    """

    metadata = {"render_modes": []}

    def __init__(self, isaac_env: Any, obs_hwc: tuple[int, int, int]):
        if isaac_env.num_envs != 1:
            raise ValueError(
                f"IsaacSingleEnvWrapper requires num_envs=1, got {isaac_env.num_envs}"
            )
        self._env = isaac_env
        self._h, self._w, self._c = obs_hwc

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._h, self._w, self._c), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(isaac_env.cfg.action_space,), dtype=np.float32
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None and hasattr(self._env, "seed"):
            self._env.seed(seed)
        obs_dict, info = self._env.reset()
        return self._unwrap_obs(obs_dict), info

    def step(self, action: np.ndarray):
        action_t = torch.as_tensor(
            action, dtype=torch.float32, device=self._env.device
        ).unsqueeze(0)
        obs_dict, reward, terminated, truncated, info = self._env.step(action_t)
        return (
            self._unwrap_obs(obs_dict),
            float(reward[0].item()),
            bool(terminated[0].item()),
            bool(truncated[0].item()),
            info,
        )

    def close(self):
        self._env.close()

    def _unwrap_obs(self, obs: Any) -> np.ndarray:
        if isinstance(obs, dict):
            obs = obs["policy"]
        if isinstance(obs, torch.Tensor):
            obs = obs.detach().cpu().numpy()
        return obs.reshape(self._h, self._w, self._c).astype(np.float32)