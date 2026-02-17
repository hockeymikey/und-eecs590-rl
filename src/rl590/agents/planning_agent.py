from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from rl590.dp.planning import policy_iteration, q_value_policy_iteration, value_iteration


@dataclass
class AgentConfig:
    algorithm: str = "policy_iteration"
    epsilon: float = 1e-8
    max_iterations: int = 10_000
    episodes: int = 20
    max_steps_per_episode: int = 200
    seed: int = 0


class PlanningAgent:
    def __init__(self, env, config: AgentConfig | None = None) -> None:
        self.env = env
        self.config = config or AgentConfig()

        self.policy: np.ndarray | None = None
        self.V: np.ndarray | None = None
        self.Q: np.ndarray | None = None
        self.deltas: list[float] = []
        self.train_iterations: int = 0

    def train(self) -> dict[str, float | int | str]:
        algo = self.config.algorithm

        if algo == "value_iteration":
            V, Q, policy, deltas, iterations = value_iteration(
                self.env.P,
                self.env.R,
                self.env.gamma,
                epsilon=self.config.epsilon,
                max_iterations=self.config.max_iterations,
            )
        elif algo == "policy_iteration":
            V, Q, policy, deltas, iterations = policy_iteration(
                self.env.P,
                self.env.R,
                self.env.gamma,
                epsilon=self.config.epsilon,
                max_iterations=self.config.max_iterations,
            )
        elif algo == "q_policy_iteration":
            V, Q, policy, deltas, iterations = q_value_policy_iteration(
                self.env.P,
                self.env.R,
                self.env.gamma,
                epsilon=self.config.epsilon,
                max_iterations=self.config.max_iterations,
            )
        else:
            raise ValueError(
                f"Unsupported algorithm: {algo}. "
                "Expected one of value_iteration, policy_iteration, q_policy_iteration."
            )

        self.V = V
        self.Q = Q
        self.policy = policy
        self.deltas = deltas
        self.train_iterations = iterations

        return {
            "algorithm": algo,
            "iterations": iterations,
            "final_delta": deltas[-1] if deltas else 0.0,
        }

    def evaluate(self) -> dict[str, float]:
        if self.policy is None:
            raise RuntimeError("Call train() or load() before evaluate().")

        rng = np.random.default_rng(self.config.seed)
        returns = []
        successes = 0

        for _ in range(self.config.episodes):
            state = self.env.start_index
            total_reward = 0.0

            for _ in range(self.config.max_steps_per_episode):
                action = int(self.policy[state])
                state, reward, done = self.env.simulate_step(state, action, rng=rng)
                total_reward += reward

                if done:
                    if self.env.index_to_state(state) == self.env.goal_state:
                        successes += 1
                    break

            returns.append(total_reward)

        return {
            "episodes": float(self.config.episodes),
            "success_rate": successes / self.config.episodes,
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
        }

    def save(self, path: str | Path) -> Path:
        if self.policy is None or self.V is None:
            raise RuntimeError("No trained model to save.")

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, np.ndarray] = {
            "policy": self.policy,
            "V": self.V,
            "metadata": np.array(json.dumps(asdict(self.config))),
        }
        if self.Q is not None:
            payload["Q"] = self.Q

        np.savez(out, **payload)
        return out

    def load(self, path: str | Path) -> None:
        data = np.load(path, allow_pickle=False)
        self.policy = data["policy"]
        self.V = data["V"]
        self.Q = data["Q"] if "Q" in data.files else None
