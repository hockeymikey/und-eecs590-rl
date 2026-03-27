"""EECS 590 Reinforcement Learning — core RL library.

V1: Dynamic programming, tabular environments, model-based belief.
V2: Classical model-free algorithms, deep RL (PPO), CNN architectures,
    replay buffers, saliency visualization.
"""

# V1 — tabular MDP and planning
from .envs.windy_chasm import WindyChasmMDP
from .agents.planning_agent import PlanningAgent, AgentConfig
from .model.belief import TabularModelBelief

# V2 — classical model-free
from .classical import MCAgent, QLearningAgent, TDPredictionAgent, SarsaAgent

# V2 — deep RL
from .deep import PPOAgent, PPOConfig

# V2 — neural networks
from .networks import ZamboniCNN, GaussianActor, Critic

# V2 — replay buffer
from .buffers import ReplayBuffer, ReplayBufferConfig

__all__ = [
    # V1
    "WindyChasmMDP", "PlanningAgent", "AgentConfig", "TabularModelBelief",
    # V2 classical
    "MCAgent", "QLearningAgent", "TDPredictionAgent", "SarsaAgent",
    # V2 deep
    "PPOAgent", "PPOConfig",
    # V2 networks
    "ZamboniCNN", "GaussianActor", "Critic",
    # V2 buffers
    "ReplayBuffer", "ReplayBufferConfig",
]
