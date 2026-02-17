from __future__ import annotations

import numpy as np
from numpy.random import Generator, default_rng


class WindyChasmMDP:
    """Tabular MDP for the windy chasm environment used in Mini 2."""

    ACTION_FORWARD = 0
    ACTION_LEFT = 1
    ACTION_RIGHT = 2

    def __init__(
        self,
        p_center: float = 0.1,
        goal_col: int = 3,
        crash_penalty: float = -100.0,
        step_reward: float = -1.0,
        goal_bonus: float = 100.0,
        gamma: float = 0.9,
    ) -> None:
        self.rows = 20
        self.cols = 7
        self.start_state = (0, 3)
        self.goal_state = (19, goal_col)
        self.p_center = p_center
        self.crash_penalty = crash_penalty
        self.step_reward = step_reward
        self.goal_bonus = goal_bonus
        self.gamma = gamma

        self.num_states = self.rows * self.cols
        self.num_actions = 3

        # Transition tensor: P[a, s, s_next]
        self.P = np.zeros((self.num_actions, self.num_states, self.num_states), dtype=float)
        # Expected immediate rewards: R[s, a]
        self.R = np.zeros((self.num_states, self.num_actions), dtype=float)

        self._build_transitions()
        self._build_rewards()

    def state_to_index(self, row: int, col: int) -> int:
        return row * self.cols + col

    def index_to_state(self, index: int) -> tuple[int, int]:
        return index // self.cols, index % self.cols

    @property
    def start_index(self) -> int:
        return self.state_to_index(*self.start_state)

    def is_terminal_state(self, state_index: int) -> bool:
        row, col = self.index_to_state(state_index)
        return col == 0 or col == self.cols - 1 or (row, col) == self.goal_state

    def action_symbols(self) -> dict[int, str]:
        return {
            self.ACTION_FORWARD: "^",
            self.ACTION_LEFT: "<",
            self.ACTION_RIGHT: ">",
        }

    def _wind_probability(self, col: int) -> float:
        # p(j) = p_center ^ (1 / (1 + (j - 3)^2))
        dist_sq = (col - 3) ** 2
        exponent = 1.0 / (1.0 + dist_sq)
        return self.p_center**exponent

    def _build_transitions(self) -> None:
        for state in range(self.num_states):
            row, col = self.index_to_state(state)

            if self.is_terminal_state(state):
                self.P[:, state, state] = 1.0
                continue

            for action in range(self.num_actions):
                next_row, next_col = row, col

                if action == self.ACTION_FORWARD:
                    next_row = min(next_row + 1, self.rows - 1)
                elif action == self.ACTION_LEFT:
                    next_col -= 1
                elif action == self.ACTION_RIGHT:
                    next_col += 1

                # Immediate collision checks before wind.
                next_col = max(0, min(self.cols - 1, next_col))

                immediate_idx = self.state_to_index(next_row, next_col)
                if self.is_terminal_state(immediate_idx):
                    self.P[action, state, immediate_idx] = 1.0
                    continue

                p = self._wind_probability(next_col)
                shifts = {
                    0: (1.0 - p) * (1.0 - p**2),
                    -1: p / 2.0,
                    1: p / 2.0,
                    -2: ((1.0 - p) * p**2) / 2.0,
                    2: ((1.0 - p) * p**2) / 2.0,
                }

                for delta, prob in shifts.items():
                    if prob == 0.0:
                        continue
                    wind_col = max(0, min(self.cols - 1, next_col + delta))
                    wind_idx = self.state_to_index(next_row, wind_col)
                    self.P[action, state, wind_idx] += prob

        row_sums = np.sum(self.P, axis=2)
        if not np.allclose(row_sums, 1.0, atol=1e-9):
            raise ValueError("Transition probabilities do not sum to 1 for all (action, state).")

    def _build_rewards(self) -> None:
        for state in range(self.num_states):
            if self.is_terminal_state(state):
                continue

            for action in range(self.num_actions):
                expected_reward = 0.0
                next_probs = self.P[action, state]
                next_states = np.flatnonzero(next_probs > 0)

                for next_state in next_states:
                    prob = next_probs[next_state]
                    reward = self.step_reward

                    row, col = self.index_to_state(next_state)
                    if (row, col) == self.goal_state:
                        reward += self.goal_bonus
                    elif col == 0 or col == self.cols - 1:
                        reward += self.crash_penalty

                    expected_reward += prob * reward

                self.R[state, action] = expected_reward

    def simulate_step(
        self,
        state_index: int,
        action: int,
        rng: Generator | None = None,
    ) -> tuple[int, float, bool]:
        if rng is None:
            rng = default_rng()

        if self.is_terminal_state(state_index):
            return state_index, 0.0, True

        probs = self.P[action, state_index]
        next_state = int(rng.choice(self.num_states, p=probs))

        row, col = self.index_to_state(next_state)
        reward = self.step_reward
        done = False

        if (row, col) == self.goal_state:
            reward += self.goal_bonus
            done = True
        elif col == 0 or col == self.cols - 1:
            reward += self.crash_penalty
            done = True

        return next_state, reward, done
