"""Neural network architectures for the Zamboni foundation environment.

CNN-based feature extractors for 2-channel image observations
(ice roughness map + agent position heatmap), plus actor and critic heads
for continuous control (throttle + steering).
"""

from .cnn import ZamboniCNN
from .actor import GaussianActor
from .critic import Critic

__all__ = ["ZamboniCNN", "GaussianActor", "Critic"]
