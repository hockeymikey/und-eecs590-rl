"""Visualization utilities: saliency analysis, rasterization, and rendering."""

from .saliency import vanilla_gradient_saliency, gradcam, plot_saliency
from .rasterize import (
    capture_zamboni_frame,
    record_zamboni_episode,
    render_tabular_heatmap,
    render_observation_channels,
)

__all__ = [
    "vanilla_gradient_saliency", "gradcam", "plot_saliency",
    "capture_zamboni_frame", "record_zamboni_episode",
    "render_tabular_heatmap", "render_observation_channels",
]
