"""SARSA methods: on-policy TD control learning Q(s, a).

The name "SARSA" comes from the update tuple: (S, A, R, S', A').
Unlike Q-learning (which uses max_a Q(S', a)), SARSA uses the action
A' that the policy *actually picks* in state S'. This makes it on-policy:
it learns the value of the policy it's currently following.

SARSA(n) forward-view:
    Use n real rewards then bootstrap from Q(S_{t+n}, A_{t+n}):
    G(n)_t = R_{t+1} + γR_{t+2} + ... + γ^n Q(S_{t+n}, A_{t+n})

SARSA(n) backward-view:
    Eligibility traces over (state, action) pairs, truncated after n steps.

SARSA(λ) forward-view:
    λ-weighted average of all n-step SARSA returns. Needs full episode.

SARSA(λ) backward-view:
    Online updates with eligibility traces decayed by γλ. The version
    already implemented in M3/td_agent.py — now generalized here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict

import numpy as np
from numpy.random import Generator, default_rng

from rl590.primitives import NpzKey, SarsaAlgorithm, TraceType


@dataclass
class SarsaConfig:
    """Hyperparameters for SARSA variants."""

    algorithm: str = SarsaAlgorithm.SARSA_LAMBDA_BACKWARD

    # SARSA parameters
    n: int = 4                 # lookahead steps for SARSA(n)
    lam: float = 0.8          # λ for SARSA(λ)
    alpha: float = 0.1        # step size
    gamma: float = 0.95       # discount factor

    # Exploration
    epsilon: float = 0.1
    epsilon_decay: float = 1.0
    epsilon_min: float = 0.01

    # Trace type for backward views
    trace_type: str = TraceType.REPLACING

    # Training
    episodes: int = 2000
    max_steps_per_episode: int = 200
    eval_episodes: int = 50
    seed: int = 0


class SarsaAgent:
    """Tabular SARSA agent — learns Q(s, a) on-policy.

    Works with any environment providing:
        num_states, num_actions, gamma,
        reset(rng) -> state, simulate_step(state, action, rng) -> (s', r, done)
    """

    def __init__(self, env, config: SarsaConfig | None = None) -> None:
        self.env = env
        self.config = config or SarsaConfig()
        self.Q = np.zeros((env.num_states, env.num_actions))

        self.episode_returns: List[float] = []
        self.episode_lengths: List[int] = []

    def _epsilon_greedy(self, state: int, epsilon: float, rng: Generator) -> int:
        if rng.random() < epsilon:
            return int(rng.integers(self.env.num_actions))
        return int(np.argmax(self.Q[state]))

    # ------------------------------------------------------------------
    # Training dispatcher
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, float | int | str]:
        algo = self.config.algorithm
        dispatch = {
            SarsaAlgorithm.SARSA_N_FORWARD: self._train_sarsa_n_forward,
            SarsaAlgorithm.SARSA_N_BACKWARD: self._train_sarsa_n_backward,
            SarsaAlgorithm.SARSA_LAMBDA_FORWARD: self._train_sarsa_lambda_forward,
            SarsaAlgorithm.SARSA_LAMBDA_BACKWARD: self._train_sarsa_lambda_backward,
        }
        if algo not in dispatch:
            raise ValueError(f"Unknown algorithm: {algo}")
        dispatch[algo]()
        return {
            "algorithm": algo,
            "episodes": len(self.episode_returns),
            "mean_return_last100": float(np.mean(self.episode_returns[-100:])),
        }

    # ------------------------------------------------------------------
    # SARSA(n) Forward-View
    # ------------------------------------------------------------------

    def _train_sarsa_n_forward(self) -> None:
        """SARSA(n) forward-view: n-step on-policy returns.

        Like TD(n) forward but for Q-values. The bootstrap target uses
        the action the policy actually chose:
            G(n)_t = R_{t+1} + γR_{t+2} + ... + γ^n Q(S_{t+n}, A_{t+n})

        Must buffer n steps of experience before updating step t.
        """
        cfg = self.config
        rng = default_rng(cfg.seed)
        eps = cfg.epsilon

        for ep in range(cfg.episodes):
            # Collect full episode trajectory with actions
            states = []
            actions = []
            rewards = []
            state = self.env.reset(rng)

            for step in range(cfg.max_steps_per_episode):
                action = self._epsilon_greedy(state, eps, rng)
                next_state, reward, done = self.env.simulate_step(state, action, rng)
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                state = next_state
                if done:
                    break

            T = len(states)

            # Update Q for each timestep using n-step return
            for t in range(T):
                lookahead = min(cfg.n, T - t)

                # Sum discounted rewards for the next n steps
                G_n = 0.0
                for k in range(lookahead):
                    G_n += (cfg.gamma ** k) * rewards[t + k]

                # Bootstrap from Q(S_{t+n}, A_{t+n}) if episode hasn't ended
                if t + lookahead < T:
                    G_n += (cfg.gamma ** lookahead) * self.Q[
                        states[t + lookahead], actions[t + lookahead]
                    ]
                # If done, no bootstrap (terminal return is 0)

                # Update Q
                self.Q[states[t], actions[t]] += cfg.alpha * (
                    G_n - self.Q[states[t], actions[t]]
                )

            self.episode_returns.append(sum(rewards))
            self.episode_lengths.append(T)
            eps = max(cfg.epsilon_min, eps * cfg.epsilon_decay)

    # ------------------------------------------------------------------
    # SARSA(n) Backward-View
    # ------------------------------------------------------------------

    def _train_sarsa_n_backward(self) -> None:
        """SARSA(n) backward-view: eligibility traces truncated at n steps.

        Same as SARSA(λ) backward but traces are hard-zeroed after n steps
        instead of soft-decayed by λ. At each step:
            1. TD error: δ = R + γQ(S', A') - Q(S, A)
            2. Bump trace: e(S, A) += 1
            3. Update all: Q += α·δ·e
            4. Decay traces: e *= γ
            5. Zero traces older than n steps
        """
        cfg = self.config
        rng = default_rng(cfg.seed)
        eps = cfg.epsilon

        for ep in range(cfg.episodes):
            state = self.env.reset(rng)
            action = self._epsilon_greedy(state, eps, rng)
            traces = np.zeros_like(self.Q)
            # Track age of each trace for n-truncation
            trace_age = np.full_like(self.Q, cfg.n + 1, dtype=int)
            total_reward = 0.0

            for step in range(cfg.max_steps_per_episode):
                next_state, reward, done = self.env.simulate_step(state, action, rng)
                total_reward += reward
                next_action = self._epsilon_greedy(next_state, eps, rng)

                # On-policy TD error: uses A' (the action we actually pick)
                q_next = self.Q[next_state, next_action] * (1 - done)
                td_error = reward + cfg.gamma * q_next - self.Q[state, action]

                # Update trace
                if cfg.trace_type == TraceType.REPLACING:
                    traces[state, action] = 1.0
                else:
                    traces[state, action] += 1.0
                trace_age[state, action] = 0

                # Update Q for all (s, a) pairs with active traces
                self.Q += cfg.alpha * td_error * traces

                # Age and decay traces
                traces *= cfg.gamma
                trace_age += 1
                traces[trace_age >= cfg.n] = 0.0

                state = next_state
                action = next_action
                if done:
                    break

            self.episode_returns.append(total_reward)
            self.episode_lengths.append(step + 1)
            eps = max(cfg.epsilon_min, eps * cfg.epsilon_decay)

    # ------------------------------------------------------------------
    # SARSA(λ) Forward-View
    # ------------------------------------------------------------------

    def _train_sarsa_lambda_forward(self) -> None:
        """SARSA(λ) forward-view: λ-weighted blend of n-step SARSA returns.

        Same recursive trick as TD(λ) forward, but using Q:
            G(λ)_t = R_{t+1} + γ[(1-λ)Q(S_{t+1}, A_{t+1}) + λ G(λ)_{t+1}]

        Requires the full episode trajectory.
        """
        cfg = self.config
        rng = default_rng(cfg.seed)
        eps = cfg.epsilon

        for ep in range(cfg.episodes):
            # Collect full episode
            states = []
            actions = []
            rewards = []
            state = self.env.reset(rng)

            for step in range(cfg.max_steps_per_episode):
                action = self._epsilon_greedy(state, eps, rng)
                next_state, reward, done = self.env.simulate_step(state, action, rng)
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                state = next_state
                if done:
                    break

            T = len(states)

            # Compute λ-returns backward through the episode
            # G(λ)_t = R_{t+1} + γ[(1-λ)Q(S_{t+1},A_{t+1}) + λ G(λ)_{t+1}]
            g_lambda_next = 0.0  # G(λ) for the step after the last one

            for t in range(T - 1, -1, -1):
                if t == T - 1:
                    # Last step: if episode ended (done), next Q is 0
                    # Otherwise bootstrap from terminal state Q
                    next_q = 0.0 if done else self.Q[state, self._epsilon_greedy(state, eps, rng)]
                else:
                    next_q = self.Q[states[t + 1], actions[t + 1]]

                g_lambda = rewards[t] + cfg.gamma * (
                    (1 - cfg.lam) * next_q + cfg.lam * g_lambda_next
                )
                g_lambda_next = g_lambda

                # Update Q
                self.Q[states[t], actions[t]] += cfg.alpha * (
                    g_lambda - self.Q[states[t], actions[t]]
                )

            self.episode_returns.append(sum(rewards))
            self.episode_lengths.append(T)
            eps = max(cfg.epsilon_min, eps * cfg.epsilon_decay)

    # ------------------------------------------------------------------
    # SARSA(λ) Backward-View
    # ------------------------------------------------------------------

    def _train_sarsa_lambda_backward(self) -> None:
        """SARSA(λ) backward-view: online with eligibility traces.

        The classic version — at each step:
            1. Pick A' from S' using ε-greedy (on-policy)
            2. TD error: δ = R + γQ(S', A') - Q(S, A)
            3. Bump trace: e(S, A) += 1 (or = 1 for replacing)
            4. Update ALL (s,a): Q(s,a) += α·δ·e(s,a)
            5. Decay ALL traces: e *= γλ

        This is equivalent to the forward-view over a complete episode.
        """
        cfg = self.config
        rng = default_rng(cfg.seed)
        eps = cfg.epsilon

        for ep in range(cfg.episodes):
            state = self.env.reset(rng)
            action = self._epsilon_greedy(state, eps, rng)
            traces = np.zeros_like(self.Q)
            total_reward = 0.0

            for step in range(cfg.max_steps_per_episode):
                next_state, reward, done = self.env.simulate_step(state, action, rng)
                total_reward += reward
                next_action = self._epsilon_greedy(next_state, eps, rng)

                # On-policy TD error
                q_next = self.Q[next_state, next_action] * (1 - done)
                td_error = reward + cfg.gamma * q_next - self.Q[state, action]

                # Update trace for current (state, action)
                if cfg.trace_type == TraceType.REPLACING:
                    traces[state, action] = 1.0
                else:
                    traces[state, action] += 1.0

                # Update Q and decay traces
                self.Q += cfg.alpha * td_error * traces
                traces *= cfg.gamma * cfg.lam

                state = next_state
                action = next_action
                if done:
                    break

            self.episode_returns.append(total_reward)
            self.episode_lengths.append(step + 1)
            eps = max(cfg.epsilon_min, eps * cfg.epsilon_decay)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, episodes: int | None = None, seed: int = 9999) -> Dict[str, float]:
        """Evaluate greedy policy (ε=0)."""
        n = episodes or self.config.eval_episodes
        rng = default_rng(seed)
        returns = []
        lengths = []

        for _ in range(n):
            state = self.env.reset(rng)
            total_reward = 0.0
            for step in range(self.config.max_steps_per_episode):
                action = int(np.argmax(self.Q[state]))
                next_state, reward, done = self.env.simulate_step(state, action, rng)
                total_reward += reward
                state = next_state
                if done:
                    break
            returns.append(total_reward)
            lengths.append(step + 1)

        return {
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "mean_length": float(np.mean(lengths)),
        }

    def policy(self) -> np.ndarray:
        """Extract greedy policy from Q-table."""
        return np.argmax(self.Q, axis=1)

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out,
            **{
                NpzKey.Q: self.Q,
                NpzKey.EPISODE_RETURNS: np.array(self.episode_returns),
                NpzKey.EPISODE_LENGTHS: np.array(self.episode_lengths),
                NpzKey.METADATA: np.array(json.dumps(asdict(self.config))),
            },
        )
        return out

    def load(self, path: str | Path) -> None:
        data = np.load(path, allow_pickle=False)
        self.Q = data[NpzKey.Q]
        self.episode_returns = data[NpzKey.EPISODE_RETURNS].tolist()
        self.episode_lengths = data[NpzKey.EPISODE_LENGTHS].tolist()
