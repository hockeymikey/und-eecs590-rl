"""Plotting utilities for the reactor control experiments."""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np


def _smooth(data: Sequence[float], window: int) -> np.ndarray:
    """Simple moving-average smoother."""
    arr = np.asarray(data, dtype=float)
    if window <= 1 or len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


# ------------------------------------------------------------------
# Learning curves
# ------------------------------------------------------------------

def plot_learning_curves(
    episode_returns_list: List[List[float]],
    labels: List[str],
    window: int = 50,
    title: str = "Learning Curves",
    ylabel: str = "Episode Return",
    output_path: str | None = None,
) -> None:
    """Overlay smoothed learning curves for multiple runs/agents."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    for returns, label in zip(episode_returns_list, labels):
        smoothed = _smooth(returns, window)
        episodes = np.arange(window - 1, window - 1 + len(smoothed))
        ax.plot(episodes, smoothed, label=label)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


# ------------------------------------------------------------------
# Q-function heatmap
# ------------------------------------------------------------------

def plot_q_heatmap(
    Q: np.ndarray,
    env,
    title: str = "Q-function Heatmap",
    output_path: str | None = None,
) -> None:
    """2-D heatmap of Q[obs_bin, action] with diverging colourmap."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 8), dpi=150)
    centers = env.bin_centers()
    action_labels = env.action_labels()

    im = ax.imshow(
        Q,
        aspect="auto",
        origin="lower",
        cmap="RdYlGn",
        interpolation="nearest",
    )
    ax.set_xticks(range(len(action_labels)))
    ax.set_xticklabels(action_labels)
    ax.set_xlabel("Action (rod increments)")

    # Show a subset of y-tick labels for readability
    ytick_step = max(1, len(centers) // 10)
    yticks = list(range(0, len(centers), ytick_step))
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{centers[i]:.1f}" for i in yticks])
    ax.set_ylabel("Observation bin centre (reactivity)")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Q-value")
    fig.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


# ------------------------------------------------------------------
# Algorithm comparison (multi-panel)
# ------------------------------------------------------------------

def plot_algorithm_comparison(
    results: dict,
    window: int = 50,
    output_path: str | None = None,
) -> None:
    """Multi-panel figure: learning curves + rolling meltdown rate.

    Parameters
    ----------
    results : dict
        Keys are algorithm labels, values are dicts with at least
        'episode_returns' and 'episode_meltdowns' lists.
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    for label, data in results.items():
        returns = data["episode_returns"]
        meltdowns = [float(m) for m in data["episode_meltdowns"]]
        smoothed_ret = _smooth(returns, window)
        smoothed_melt = _smooth(meltdowns, window)
        episodes = np.arange(window - 1, window - 1 + len(smoothed_ret))

        ax1.plot(episodes, smoothed_ret, label=label)
        ax2.plot(episodes, smoothed_melt, label=label)

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Episode Return")
    ax1.set_title("Learning Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Meltdown Rate")
    ax2.set_title("Rolling Meltdown Rate")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


# ------------------------------------------------------------------
# Meltdown rate over training
# ------------------------------------------------------------------

def plot_meltdown_curve(
    meltdown_lists: List[List[bool]],
    labels: List[str],
    window: int = 50,
    output_path: str | None = None,
) -> None:
    """Plot rolling meltdown rate for one or more agents."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    for meltdowns, label in zip(meltdown_lists, labels):
        smoothed = _smooth([float(m) for m in meltdowns], window)
        episodes = np.arange(window - 1, window - 1 + len(smoothed))
        ax.plot(episodes, smoothed, label=label)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Meltdown Rate (rolling)")
    ax.set_title("Meltdown Rate During Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)
