"""CNN feature extractor for the Zamboni foundation environment.

The Zamboni observation is a 2-channel image (H=129, W=304):
    Channel 0: Ice roughness/elevation map (where the damage is)
    Channel 1: Agent position heatmap (where the Zamboni is)

Why a CNN?
    - The input is spatial data — damage patterns, wall locations, and
      agent position all have spatial structure that convolutions capture.
    - Translation equivariance: a patch of damaged ice looks the same
      whether it's in the corner or center of the rink.
    - Parameter efficiency: a fully-connected layer on 129*304*2 = 78,432
      inputs would be enormous. Conv layers share weights across positions.

Architecture choices:
    - 3 conv layers with increasing filters (16 → 32 → 64) to build up
      from low-level edges to higher-level spatial features.
    - Stride-2 in early layers for spatial downsampling (more efficient
      than max pooling for RL where we care about spatial precision).
    - ReLU activations throughout (standard, stable for RL).
    - Final linear layer projects flattened conv features to a fixed-size
      feature vector that actor and critic heads consume.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ZamboniCNN(nn.Module):
    """Feature extractor for 2-channel (H, W, 2) Zamboni observations.

    Accepts observations in channels-last format (H, W, C) as produced
    by the Gymnasium environment, and permutes to channels-first (C, H, W)
    for PyTorch convolutions internally.

    Parameters
    ----------
    obs_height : int
        Height of the observation image (default 129).
    obs_width : int
        Width of the observation image (default 304).
    n_channels : int
        Number of input channels (default 2).
    features_dim : int
        Size of the output feature vector (default 128).
    """

    def __init__(
        self,
        obs_height: int = 129,
        obs_width: int = 304,
        n_channels: int = 2,
        features_dim: int = 128,
    ) -> None:
        super().__init__()
        self.obs_height = obs_height
        self.obs_width = obs_width
        self.n_channels = n_channels
        self.features_dim = features_dim

        # Convolutional backbone
        # Input: (batch, 2, 129, 304)
        # After conv1 (stride=2, 5x5): (batch, 16, 63, 151)
        # After conv2 (stride=2, 3x3): (batch, 32, 32, 76)
        # After conv3 (stride=2, 3x3): (batch, 64, 16, 38)
        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute the flattened size by running a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, obs_height, obs_width)
            n_flat = self.conv(dummy).shape[1]

        # Project to feature vector
        self.fc = nn.Sequential(
            nn.Linear(n_flat, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract features from observation.

        Parameters
        ----------
        obs : Tensor
            Shape (batch, H, W, C) channels-last or (batch, C, H, W) channels-first.

        Returns
        -------
        Tensor of shape (batch, features_dim)
        """
        # Handle channels-last input from Gymnasium
        if obs.dim() == 4 and obs.shape[-1] == self.n_channels:
            obs = obs.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        elif obs.dim() == 3 and obs.shape[-1] == self.n_channels:
            obs = obs.permute(2, 0, 1).unsqueeze(0)  # (H, W, C) -> (1, C, H, W)

        return self.fc(self.conv(obs))
