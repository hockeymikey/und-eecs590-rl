"""Critic network for value function estimation.

The critic estimates V(s) — the expected discounted return from state s
under the current policy. This is used in PPO to:
    1. Compute advantages: Â_t = R_t + γV(S_{t+1}) - V(S_t) (via GAE)
    2. Provide a baseline to reduce variance in policy gradient updates
    3. The critic loss is MSE between predicted V(s) and the actual return

The critic shares the same CNN architecture as the actor (same observations)
but has its own weights and a scalar output instead of action dimensions.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .cnn import ZamboniCNN


class Critic(nn.Module):
    """Value function V(s) network.

    Parameters
    ----------
    cnn : ZamboniCNN
        Feature extractor (can be shared with actor or independent).
    hidden_dim : int
        Width of the MLP head between features and value output.
    """

    def __init__(self, cnn: ZamboniCNN, hidden_dim: int = 64) -> None:
        super().__init__()
        self.cnn = cnn

        # MLP head: features → hidden → scalar value
        self.mlp = nn.Sequential(
            nn.Linear(cnn.features_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Estimate V(s).

        Parameters
        ----------
        obs : Tensor of shape (batch, H, W, C) or (batch, C, H, W)

        Returns
        -------
        value : Tensor of shape (batch,)
        """
        features = self.cnn(obs)
        return self.mlp(features).squeeze(-1)
