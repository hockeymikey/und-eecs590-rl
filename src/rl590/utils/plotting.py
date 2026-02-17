from __future__ import annotations

from pathlib import Path


def plot_convergence(deltas: list[float], output_path: str | None = None) -> None:
    """Plot convergence deltas for iterative DP algorithms."""
    if not deltas:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting.") from exc

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(deltas)
    ax.set_title("Convergence Delta Per Iteration")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Max |V_new - V_old| or |Q_new - Q_old|")
    ax.grid(True, alpha=0.3)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)
