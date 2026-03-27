"""Saliency analysis for CNN-based RL agents.

Saliency maps answer: "which pixels does the network care about most?"

For the Zamboni, this tells us:
    - Is the agent looking at damaged ice? (channel 0: roughness)
    - Is it looking at its own position? (channel 1: position heatmap)
    - Is it focusing on walls/boundaries?
    - Is it attending to already-resurfaced areas?

Two methods implemented:

1. Vanilla Gradient Saliency (Simonyan et al., 2013)
    Compute ∂output/∂input — the gradient of the network output with
    respect to each input pixel. Large gradient = small change in that
    pixel would significantly change the output. Simple and fast.

2. Grad-CAM (Selvaraju et al., 2017)
    Highlights which spatial regions in the LAST conv layer activate
    most strongly for the output. More interpretable than raw gradients
    because it shows coarse attention regions rather than noisy per-pixel
    sensitivity. Works by weighting activation maps by their gradient
    importance, then upsampling to input resolution.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from rl590.networks import ZamboniCNN, GaussianActor, Critic


def vanilla_gradient_saliency(
    network: torch.nn.Module,
    obs: np.ndarray,
    output_index: int | None = None,
) -> np.ndarray:
    """Compute vanilla gradient saliency map.

    How it works:
        1. Feed observation through the network
        2. Pick a scalar output (e.g., value, or one action mean)
        3. Backpropagate to get ∂output/∂input
        4. Take absolute value of gradients — magnitude = importance

    Parameters
    ----------
    network : nn.Module
        The network to analyze (critic for V(s), actor for action means).
    obs : ndarray
        Single observation, shape (H, W, C).
    output_index : int, optional
        Which output scalar to differentiate. For critic (1 output), use None.
        For actor (2 outputs: throttle, steering), use 0 or 1.

    Returns
    -------
    saliency : ndarray of shape (H, W), values in [0, 1]
        Normalized saliency map — bright = high importance.
    """
    device = next(network.parameters()).device

    # Prepare input with gradient tracking
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    obs_t.requires_grad_(True)

    # Forward pass — handle networks that return tuples (e.g., actor returns (mean, log_std))
    raw_output = network(obs_t)
    output = raw_output[0] if isinstance(raw_output, tuple) else raw_output

    # Select which output to differentiate
    if output.dim() == 1 or (output.dim() == 2 and output.shape[1] == 1):
        # Scalar output (critic)
        target = output.squeeze()
    elif output_index is not None:
        # Multi-dim output (actor mean) — pick one action
        target = output[0, output_index]
    else:
        # Default: sum all outputs
        target = output.sum()

    # Backward pass — compute gradients w.r.t. input pixels
    target.backward()

    # Saliency = absolute value of gradient, max across channels
    grad = obs_t.grad.detach().cpu().numpy()[0]  # (H, W, C)
    saliency = np.max(np.abs(grad), axis=-1)  # (H, W)

    # Normalize to [0, 1]
    s_min, s_max = saliency.min(), saliency.max()
    if s_max > s_min:
        saliency = (saliency - s_min) / (s_max - s_min)

    return saliency


def gradcam(
    network: torch.nn.Module,
    obs: np.ndarray,
    conv_layer: torch.nn.Module | None = None,
    output_index: int | None = None,
) -> np.ndarray:
    """Compute Grad-CAM saliency map.

    How it works:
        1. Hook into the last conv layer to capture activations and gradients
        2. Forward pass to get the output
        3. Backward pass to get gradients flowing into that conv layer
        4. Global-average-pool the gradients to get per-channel weights
        5. Weighted sum of activation maps = coarse attention heatmap
        6. ReLU (we only care about features with positive influence)
        7. Upsample to input resolution

    Parameters
    ----------
    network : nn.Module
        Network to analyze. Must contain conv layers (e.g., ZamboniCNN).
    obs : ndarray
        Single observation, shape (H, W, C).
    conv_layer : nn.Module, optional
        Which conv layer to hook. If None, finds the last Conv2d.
    output_index : int, optional
        Which output to differentiate (same as vanilla_gradient_saliency).

    Returns
    -------
    saliency : ndarray of shape (H, W), values in [0, 1]
    """
    device = next(network.parameters()).device

    # Find the last Conv2d layer if not specified
    if conv_layer is None:
        for module in reversed(list(network.modules())):
            if isinstance(module, torch.nn.Conv2d):
                conv_layer = module
                break
        if conv_layer is None:
            raise ValueError("No Conv2d layer found in network")

    # Storage for hook outputs
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    # Register hooks
    fwd_handle = conv_layer.register_forward_hook(forward_hook)
    bwd_handle = conv_layer.register_full_backward_hook(backward_hook)

    try:
        # Forward pass — handle tuple outputs (e.g., actor)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        raw_output = network(obs_t)
        output = raw_output[0] if isinstance(raw_output, tuple) else raw_output

        # Select target output
        if output.dim() == 1 or (output.dim() == 2 and output.shape[1] == 1):
            target = output.squeeze()
        elif output_index is not None:
            target = output[0, output_index]
        else:
            target = output.sum()

        # Backward pass
        network.zero_grad()
        target.backward()

        # Grad-CAM computation
        act = activations[0]   # (1, C, H', W')
        grad = gradients[0]    # (1, C, H', W')

        # Global average pool gradients → per-channel importance weights
        weights = grad.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * act).sum(dim=1, keepdim=True)  # (1, 1, H', W')
        cam = F.relu(cam)  # only positive influence

        # Upsample to input resolution
        cam = F.interpolate(
            cam, size=(obs.shape[0], obs.shape[1]), mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        c_min, c_max = cam.min(), cam.max()
        if c_max > c_min:
            cam = (cam - c_min) / (c_max - c_min)

        return cam

    finally:
        fwd_handle.remove()
        bwd_handle.remove()


def plot_saliency(
    obs: np.ndarray,
    saliency: np.ndarray,
    title: str = "Saliency Map",
    output_path: str | Path | None = None,
    channel_names: tuple[str, ...] = ("Ice Roughness", "Agent Position"),
) -> None:
    """Plot observation channels alongside saliency heatmap overlay.

    Produces a figure with N+2 panels:
        - One panel per observation channel
        - The saliency heatmap alone
        - The saliency overlaid on the first channel (ice roughness)

    Parameters
    ----------
    obs : ndarray of shape (H, W, C)
    saliency : ndarray of shape (H, W), values in [0, 1]
    title : str
    output_path : Path, optional — saves to file if provided
    channel_names : tuple of str — names for each channel
    """
    n_channels = obs.shape[-1]
    n_panels = n_channels + 2  # channels + saliency + overlay

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))

    # Plot each channel
    for i in range(n_channels):
        name = channel_names[i] if i < len(channel_names) else f"Channel {i}"
        axes[i].imshow(obs[:, :, i], cmap="gray")
        axes[i].set_title(name)
        axes[i].axis("off")

    # Saliency heatmap
    axes[n_channels].imshow(saliency, cmap="hot")
    axes[n_channels].set_title("Saliency")
    axes[n_channels].axis("off")

    # Overlay: saliency on top of first channel
    axes[n_channels + 1].imshow(obs[:, :, 0], cmap="gray")
    axes[n_channels + 1].imshow(saliency, cmap="hot", alpha=0.5)
    axes[n_channels + 1].set_title("Overlay (Ch0 + Saliency)")
    axes[n_channels + 1].axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved saliency plot to {out}")

    plt.close(fig)
    return fig
