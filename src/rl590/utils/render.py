from __future__ import annotations

import numpy as np


def render_policy_ascii(env, policy: np.ndarray, show_start: bool = True) -> str:
    symbols = env.action_symbols()
    lines = []
    lines.append(f"Policy (start={env.start_state}, goal={env.goal_state})")
    lines.append("Legend: ^ Forward, < Left, > Right, X Wall, G Goal, S Start")
    lines.append("-" * 48)

    for row in range(env.rows):
        row_cells = []
        for col in range(env.cols):
            state = env.state_to_index(row, col)

            if col == 0 or col == env.cols - 1:
                token = "X"
            elif (row, col) == env.goal_state:
                token = "G"
            elif show_start and (row, col) == env.start_state:
                token = "S"
            else:
                action = int(policy[state])
                token = symbols[action]

            row_cells.append(f"{token:>2}")

        lines.append(f"Row {row:02d} | {' '.join(row_cells)}")

    lines.append("-" * 48)
    return "\n".join(lines)


def render_values_ascii(env, values: np.ndarray) -> str:
    lines = ["State Values", "-" * 48]

    for row in range(env.rows):
        row_vals = []
        for col in range(env.cols):
            state = env.state_to_index(row, col)
            if col == 0 or col == env.cols - 1:
                row_vals.append("   X   ")
            elif (row, col) == env.goal_state:
                row_vals.append("   G   ")
            else:
                row_vals.append(f"{values[state]:6.1f}")
        lines.append(f"Row {row:02d} | {' '.join(row_vals)}")

    lines.append("-" * 48)
    return "\n".join(lines)
