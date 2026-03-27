"""Gaussian actor network for continuous action spaces.

For the Zamboni, the actor outputs a probability distribution over
2 continuous actions: throttle ∈ [-1, 1] and steering ∈ [-1, 1].

It does this by outputting a mean (μ) and log standard deviation (log σ)
for each action dimension. Actions are sampled from:
    a ~ N(μ, σ²)
then squashed through tanh to stay in [-1, 1].

Why Gaussian?
    - Standard choice for continuous control in policy gradient methods.
    - The log_std is learnable, allowing the policy to become more
      deterministic as it gains confidence.
    - The tanh squashing ensures actions stay in bounds while preserving
      differentiability for backpropagation.

Why separate log_std?
    - log_std can be state-independent (a single learnable parameter per
      action dim) or state-dependent (output of the network). We use
      state-independent for stability, which is the PPO default.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from .cnn import ZamboniCNN


class GaussianActor(nn.Module):
    """Actor network that outputs a squashed Gaussian policy.

    Parameters
    ----------
    cnn : ZamboniCNN
        Shared feature extractor (or a fresh one).
    action_dim : int
        Number of continuous action dimensions (default 2: throttle + steering).
    hidden_dim : int
        Width of the MLP head between features and action output.
    init_log_std : float
        Initial value for the log standard deviation (default 0 → σ=1).
    """

    def __init__(
        self,
        cnn: ZamboniCNN,
        action_dim: int = 2,
        hidden_dim: int = 64,
        init_log_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.cnn = cnn
        self.action_dim = action_dim

        # MLP head: features → hidden → action means
        self.mlp = nn.Sequential(
            nn.Linear(cnn.features_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # State-independent log standard deviation (one per action dim)
        # Learned during training — starts at init_log_std
        self.log_std = nn.Parameter(torch.full((action_dim,), init_log_std))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute action mean and log_std from observation.

        Returns
        -------
        mean : Tensor of shape (batch, action_dim)
        log_std : Tensor of shape (action_dim,), broadcast to batch
        """
        features = self.cnn(obs)
        mean = self.mlp(features)
        return mean, self.log_std.expand_as(mean)

    def get_distribution(self, obs: torch.Tensor) -> Normal:
        """Build the Gaussian distribution for the given observation."""
        mean, log_std = self.forward(obs)
        return Normal(mean, log_std.exp())

    def sample_action(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample an action and compute its log-probability.

        Actions are squashed through tanh to enforce [-1, 1] bounds.
        The log-probability is corrected for the tanh transformation.

        Returns
        -------
        action : Tensor of shape (batch, action_dim), in [-1, 1]
        log_prob : Tensor of shape (batch,), summed across action dims
        """
        dist = self.get_distribution(obs)

        # Sample from Gaussian (with reparameterization trick for gradients)
        raw_action = dist.rsample()

        # Squash through tanh to get bounded action
        action = torch.tanh(raw_action)

        # Correct log-prob for the tanh squashing:
        # log π(a|s) = log N(u|μ,σ) - Σ log(1 - tanh²(u))
        log_prob = dist.log_prob(raw_action) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)  # sum across action dimensions

        return action, log_prob

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute log-probability of a given action under current policy.

        Used during PPO updates to compute the importance ratio π_new/π_old.

        Parameters
        ----------
        obs : Tensor (batch, H, W, C) or (batch, C, H, W)
        action : Tensor (batch, action_dim), already tanh-squashed

        Returns
        -------
        log_prob : Tensor (batch,)
        """
        dist = self.get_distribution(obs)

        # Invert tanh to recover the raw (pre-squash) action
        raw_action = torch.atanh(action.clamp(-0.999, 0.999))

        log_prob = dist.log_prob(raw_action) - torch.log(1 - action.pow(2) + 1e-6)
        return log_prob.sum(dim=-1)

    def entropy(self, obs: torch.Tensor) -> torch.Tensor:
        """Approximate entropy of the policy at given observations.

        Returns
        -------
        entropy : Tensor (batch,)
        """
        dist = self.get_distribution(obs)
        return dist.entropy().sum(dim=-1)
