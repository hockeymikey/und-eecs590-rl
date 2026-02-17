from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.random import Generator, default_rng


@dataclass
class BeliefConfig:
    transition_prior: float = 1e-3
    reward_prior_mean: float = 0.0
    reward_prior_count: float = 0.0


class TabularModelBelief:
    """Counts-based belief over transitions and rewards for tabular MDPs."""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        config: BeliefConfig | None = None,
    ) -> None:
        self.num_states = num_states
        self.num_actions = num_actions
        self.config = config or BeliefConfig()

        self.transition_counts = np.full(
            (num_actions, num_states, num_states),
            self.config.transition_prior,
            dtype=float,
        )

        base_reward_count = self.config.reward_prior_count
        base_reward_sum = self.config.reward_prior_mean * base_reward_count

        self.reward_counts = np.full(
            (num_actions, num_states, num_states),
            base_reward_count,
            dtype=float,
        )
        self.reward_sums = np.full(
            (num_actions, num_states, num_states),
            base_reward_sum,
            dtype=float,
        )

        self.num_updates = 0

    def update(self, state: int, action: int, next_state: int, reward: float) -> None:
        self.transition_counts[action, state, next_state] += 1.0
        self.reward_counts[action, state, next_state] += 1.0
        self.reward_sums[action, state, next_state] += reward
        self.num_updates += 1

    def update_batch(self, transitions: list[tuple[int, int, int, float]]) -> None:
        for state, action, next_state, reward in transitions:
            self.update(state, action, next_state, reward)

    def transition_probabilities(self) -> np.ndarray:
        totals = np.sum(self.transition_counts, axis=2, keepdims=True)
        probs = np.divide(
            self.transition_counts,
            totals,
            out=np.zeros_like(self.transition_counts),
            where=totals > 0,
        )
        zero_rows = np.squeeze(totals == 0, axis=2)
        if np.any(zero_rows):
            probs[zero_rows] = 1.0 / self.num_states
        return probs

    def reward_means(self) -> np.ndarray:
        default = np.full_like(self.reward_sums, self.config.reward_prior_mean, dtype=float)
        return np.divide(
            self.reward_sums,
            self.reward_counts,
            out=default,
            where=self.reward_counts > 0,
        )

    def expected_reward_table(self) -> np.ndarray:
        probs = self.transition_probabilities()
        reward_means = self.reward_means()
        reward_as = np.einsum("ask,ask->as", probs, reward_means)
        return reward_as.T  # [state, action]

    def estimated_mdp(self) -> tuple[np.ndarray, np.ndarray]:
        probs = self.transition_probabilities()
        rewards = self.expected_reward_table()
        return probs, rewards

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out,
            transition_counts=self.transition_counts,
            reward_counts=self.reward_counts,
            reward_sums=self.reward_sums,
            num_updates=np.array([self.num_updates], dtype=int),
        )
        return out

    def load(self, path: str | Path) -> None:
        data = np.load(path, allow_pickle=False)
        self.transition_counts = data["transition_counts"]
        self.reward_counts = data["reward_counts"]
        self.reward_sums = data["reward_sums"]
        self.num_updates = int(data["num_updates"][0]) if "num_updates" in data.files else 0


def collect_transitions(
    env,
    episodes: int,
    max_steps_per_episode: int,
    epsilon: float,
    seed: int,
    policy: np.ndarray | None = None,
) -> list[tuple[int, int, int, float]]:
    rng: Generator = default_rng(seed)
    transitions: list[tuple[int, int, int, float]] = []

    for _ in range(episodes):
        state = env.start_index

        for _ in range(max_steps_per_episode):
            choose_random = policy is None or rng.random() < epsilon
            if choose_random:
                action = int(rng.integers(0, env.num_actions))
            else:
                action = int(policy[state])

            next_state, reward, done = env.simulate_step(state, action, rng=rng)
            transitions.append((state, action, next_state, float(reward)))
            state = next_state

            if done:
                break

    return transitions


def evaluate_policy(
    env,
    policy: np.ndarray,
    episodes: int,
    max_steps_per_episode: int,
    seed: int,
) -> dict[str, float]:
    rng: Generator = default_rng(seed)
    returns = []
    successes = 0

    for _ in range(episodes):
        state = env.start_index
        total_reward = 0.0

        for _ in range(max_steps_per_episode):
            action = int(policy[state])
            next_state, reward, done = env.simulate_step(state, action, rng=rng)
            total_reward += reward
            state = next_state

            if done:
                if env.index_to_state(state) == env.goal_state:
                    successes += 1
                break

        returns.append(total_reward)

    return {
        "episodes": float(episodes),
        "success_rate": successes / episodes,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
    }
