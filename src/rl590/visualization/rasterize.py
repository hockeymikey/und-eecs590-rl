"""Rasterization utilities for rendering environment states as images.

Produces static frame captures from environments for inspection,
documentation, and debugging. Supports:
    - Zamboni-compatible envs with a matplotlib-backed ``render()`` method:
      full rink visualization with ice quality heatmap, vehicle position,
      and rink markings.
    - Tabular envs (WindyChasm): grid-based policy/value heatmaps.

Usage:
    # Zamboni: capture a single frame
    from rl590.visualization.rasterize import capture_zamboni_frame
    frame = capture_zamboni_frame(env)

    # Zamboni: record an episode as a sequence of frames
    from rl590.visualization.rasterize import record_zamboni_episode
    record_zamboni_episode(env, agent, output_dir="artifacts/episode_frames")

    # WindyChasm: render policy and value grids as images
    from rl590.visualization.rasterize import render_tabular_heatmap
    render_tabular_heatmap(env, V, policy, output_path="artifacts/policy_heatmap.png")
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving
import matplotlib.pyplot as plt


# Zamboni rasterization

def capture_zamboni_frame(env, output_path: str | Path | None = None) -> np.ndarray:
    """Capture the current environment state as an RGB image array.

    Calls env.render() to produce the matplotlib figure, then rasterizes
    the canvas to a numpy array. Optionally saves to disk.

    Parameters
    ----------
    env : object
        Environment instance with ``render()`` and a matplotlib ``fig`` attr.
    output_path : str or Path, optional
        If provided, saves the frame as a PNG.

    Returns
    -------
    frame : ndarray of shape (H, W, 3), dtype uint8
    """
    env.render()

    fig = env.fig
    if fig is None:
        raise RuntimeError("env.render() did not create a figure. Call env.reset() first.")

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    frame = np.asarray(buf)[:, :, :3].copy()  # drop alpha channel

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved frame to {out}")

    return frame


def record_zamboni_episode(
    env,
    get_action: Callable[[np.ndarray], np.ndarray],
    output_dir: str | Path = "artifacts/episode_frames",
    max_steps: int = 500,
    save_every: int = 10,
    seed: int = 42,
) -> list[Path]:
    """Record an episode as a sequence of saved frames.

    Parameters
    ----------
    env : object
        Environment instance with ``render()`` and a matplotlib ``fig`` attr.
    get_action : callable
        Function that takes an observation and returns an action.
        e.g., agent.predict or lambda obs: np.zeros(2)
    output_dir : str or Path
        Directory to save frame PNGs.
    max_steps : int
        Maximum steps to record.
    save_every : int
        Save a frame every N steps (reduces disk usage).
    seed : int
        Random seed for the episode.

    Returns
    -------
    saved_paths : list of Path
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    obs, _ = env.reset(seed=seed)
    saved = []

    for step in range(max_steps):
        action = get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        if step % save_every == 0:
            frame_path = out_dir / f"frame_{step:05d}.png"
            capture_zamboni_frame(env, frame_path)
            saved.append(frame_path)

        if terminated or truncated:
            # Capture final frame
            frame_path = out_dir / f"frame_{step:05d}_final.png"
            capture_zamboni_frame(env, frame_path)
            saved.append(frame_path)
            break

    env.close()
    print(f"Recorded {len(saved)} frames to {out_dir}/")
    return saved


# Tabular environment rasterization

def render_tabular_heatmap(
    env,
    values: np.ndarray,
    policy: np.ndarray | None = None,
    title: str = "Value Function",
    output_path: str | Path | None = None,
) -> None:
    """Render a tabular environment's value function as a colored grid.

    Produces a heatmap of V(s) with optional policy arrows overlaid.
    Works with WindyChasm or any env with index_to_state(idx) -> (row, col).

    Parameters
    ----------
    env : WindyChasmMDP or similar
        Must have: rows, cols, num_states, index_to_state(), is_terminal_state()
    values : ndarray of shape (num_states,)
        Value for each state (V or max Q).
    policy : ndarray of shape (num_states,), optional
        Action index for each state. If provided, arrows are drawn.
    title : str
    output_path : str or Path, optional
    """
    grid = np.full((env.rows, env.cols), np.nan)
    for s in range(env.num_states):
        r, c = env.index_to_state(s)
        if not env.is_terminal_state(s):
            grid[r, c] = values[s]

    fig, ax = plt.subplots(1, 1, figsize=(max(env.cols * 0.8, 6), max(env.rows * 0.4, 8)))

    # Heatmap
    cmap = plt.cm.RdYlGn
    cmap.set_bad("gray", alpha=0.3)  # terminal states shown in gray
    im = ax.imshow(grid, cmap=cmap, aspect="auto", origin="upper")
    plt.colorbar(im, ax=ax, shrink=0.6, label="Value")

    # Policy arrows
    if policy is not None:
        action_dx = {0: (0, -0.3), 1: (-0.3, 0), 2: (0.3, 0)}  # forward, left, right
        for s in range(env.num_states):
            if env.is_terminal_state(s):
                continue
            r, c = env.index_to_state(s)
            a = int(policy[s])
            if a in action_dx:
                dy, dx = action_dx[a]
                ax.annotate("", xy=(c + dx, r + dy), xytext=(c, r),
                            arrowprops=dict(arrowstyle="->", color="black", lw=1.2))

    # Mark special states
    if hasattr(env, "start_state"):
        sr, sc = env.start_state
        ax.plot(sc, sr, "bs", markersize=10, label="Start")
    if hasattr(env, "goal_state"):
        gr, gc = env.goal_state
        ax.plot(gc, gr, "r*", markersize=14, label="Goal")

    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.legend(loc="upper right")

    plt.tight_layout()

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved heatmap to {out}")

    plt.close(fig)


def render_observation_channels(
    obs: np.ndarray,
    channel_names: tuple[str, ...] = ("Ice Roughness", "Agent Position"),
    title: str = "Observation",
    output_path: str | Path | None = None,
) -> None:
    """Render each channel of an observation image as a separate subplot.

    Useful for inspecting what the CNN receives as input.

    Parameters
    ----------
    obs : ndarray of shape (H, W, C)
    channel_names : tuple of str
    title : str
    output_path : str or Path, optional
    """
    n_channels = obs.shape[-1]
    fig, axes = plt.subplots(1, n_channels, figsize=(6 * n_channels, 5))
    if n_channels == 1:
        axes = [axes]

    for i in range(n_channels):
        name = channel_names[i] if i < len(channel_names) else f"Channel {i}"
        im = axes[i].imshow(obs[:, :, i], cmap="viridis")
        axes[i].set_title(name)
        axes[i].axis("off")
        plt.colorbar(im, ax=axes[i], shrink=0.6)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved observation plot to {out}")

    plt.close(fig)
