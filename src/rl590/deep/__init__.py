"""Deep reinforcement learning algorithms.

Implements policy gradient and actor-critic methods for environments
with large or continuous state/action spaces.
"""

from .ppo import PPOAgent, PPOConfig

__all__ = ["PPOAgent", "PPOConfig"]
