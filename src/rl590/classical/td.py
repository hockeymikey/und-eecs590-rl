"""Temporal Difference methods for value prediction (V).

TD(n) forward-view:
    Compute the n-step return by looking ahead n real rewards, then bootstrap:
    G(n)_t = R_{t+1} + γR_{t+2} + ... + γ^(n-1)R_{t+n} + γ^n V(S_{t+n})
    Then update V(S_t) toward that target. Requires buffering n steps.

TD(n) backward-view:
    Equivalent result via eligibility traces truncated after n steps.
    Update online using one-step TD errors spread backward through traces.

TD(λ) forward-view:
    Blend ALL n-step returns with exponentially decaying weights:
    G(λ)_t = (1-λ) Σ λ^(n-1) G(n)_t  +  λ^(T-t-1) G_t
    Requires the full episode trajectory.

TD(λ) backward-view:
    Online updates with eligibility traces decayed by γλ each step.
    Mathematically equivalent to forward-view over a complete episode.

All methods here learn V(s) under a fixed policy (ε-greedy derived from a
separate Q-table, or purely exploratory). For control (learning Q), see
sarsa.py and qlearning.py.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict

import numpy as np
from numpy.random import Generator, default_rng

from rl590.primitives import NpzKey, TDPredictionAlgorithm, TraceType


@dataclass
class TDConfig:
    """Hyperparameters for TD prediction."""

    algorithm: str = TDPredictionAlgorithm.TD_N_FORWARD

    # TD parameters
    n: int = 4                 # lookahead steps for TD(n)
    lam: float = 0.8          # λ for TD(λ)
    alpha: float = 0.01       # step size
    gamma: float = 0.95       # discount factor

    # Behavior policy (ε-greedy exploration)
    epsilon: float = 0.1
    epsilon_decay: float = 1.0
    epsilon_min: float = 0.01

    # Trace type for backward views
    trace_type: str = TraceType.ACCUMULATING

    # Training
    episodes: int = 2000
    max_steps_per_episode: int = 200
    eval_episodes: int = 50
    seed: int = 0


class TDPredictionAgent:
    """Tabular TD prediction agent — learns V(s).

    Also maintains a Q-table for ε-greedy action selection, updated
    by treating Q(s,a) ≈ R(s,a) + γ V(s'). This lets us derive a
    behavior policy while the core learning target is V.

    Works with any environment providing:
        num_states, num_actions, gamma,
        reset(rng) -> state, simulate_step(state, action, rng) -> (s', r, done)
    """

    def __init__(self, env, config: TDConfig | None = None) -> None:
        self.env = env
        self.config = config or TDConfig()
        self.V = np.zeros(env.num_states)
        self.Q = np.zeros((env.num_states, env.num_actions))

        self.episode_returns: List[float] = []
        self.episode_lengths: List[int] = []

    def _epsilon_greedy(self, state: int, epsilon: float, rng: Generator) -> int:
        if rng.random() < epsilon:
            return int(rng.integers(self.env.num_actions))
        return int(np.argmax(self.Q[state]))

    def _update_q_from_v(self, state: int, action: int, reward: float,
                         next_state: int, done: bool) -> None:
        """Keep Q roughly in sync with V for action selection."""
        target = reward + self.config.gamma * self.V[next_state] * (1 - done)
        self.Q[state, action] += self.config.alpha * (target - self.Q[state, action])

    # ------------------------------------------------------------------
    # Training dispatcher
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, float | int | str]:
        algo = self.config.algorithm
        dispatch = {
            TDPredictionAlgorithm.TD_N_FORWARD: self._train_td_n_forward,
            TDPredictionAlgorithm.TD_N_BACKWARD: self._train_td_n_backward,
            TDPredictionAlgorithm.TD_LAMBDA_FORWARD: self._train_td_lambda_forward,
            TDPredictionAlgorithm.TD_LAMBDA_BACKWARD: self._train_td_lambda_backward,
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
    # TD(n) Forward-View
    # ------------------------------------------------------------------

    def _train_td_n_forward(self) -> None:
        """TD(n) forward-view: compute n-step returns, then update.

        For each step t, the n-step return is:
            G(n)_t = R_{t+1} + γR_{t+2} + ... + γ^(n-1)R_{t+n} + γ^n V(S_{t+n})

        We buffer the trajectory and update V(S_t) once we have n future
        steps (or the episode ends).
        """
        cfg = self.config
        rng = default_rng(cfg.seed)
        eps = cfg.epsilon

        for ep in range(cfg.episodes):
            # Collect full episode trajectory
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
                self._update_q_from_v(state, action, reward, next_state, done)
                state = next_state
                if done:
                    break

            T = len(states)  # episode length
            terminal_state = state  # final state after last transition

            # Now update V for each timestep using the n-step return
            for t in range(T):
                # How many real rewards can we use? Min of n and remaining steps.
                lookahead = min(cfg.n, T - t)

                # Build the n-step return: sum of discounted rewards
                G_n = 0.0
                for k in range(lookahead):
                    G_n += (cfg.gamma ** k) * rewards[t + k]

                # Bootstrap from V at the state we land on after n steps
                # If we reached the end of the episode, no bootstrap needed
                if t + lookahead < T:
                    G_n += (cfg.gamma ** lookahead) * self.V[states[t + lookahead]]
                elif t + lookahead == T and not done:
                    # Episode truncated (hit max steps), bootstrap from final state
                    G_n += (cfg.gamma ** lookahead) * self.V[terminal_state]
                # else: episode ended (done=True), no bootstrap

                # Update V
                self.V[states[t]] += cfg.alpha * (G_n - self.V[states[t]])

            self.episode_returns.append(sum(rewards))
            self.episode_lengths.append(T)
            eps = max(cfg.epsilon_min, eps * cfg.epsilon_decay)

    # ------------------------------------------------------------------
    # TD(n) Backward-View
    # ------------------------------------------------------------------

    def _train_td_n_backward(self) -> None:
        """TD(n) backward-view: eligibility traces with n-step truncation.

        At each step:
            1. Compute one-step TD error: δ_t = R + γV(S') - V(S)
            2. Set trace for current state: e(S) += 1 (or = 1 for replacing)
            3. Update ALL states: V(s) += α·δ·e(s)
            4. Decay traces: e(s) *= γ
            5. Zero out traces older than n steps (truncation)

        The truncation makes this equivalent to the n-step forward view.
        Without truncation (letting traces decay naturally with λ), you
        get TD(λ) backward-view instead.
        """
        cfg = self.config
        rng = default_rng(cfg.seed)
        eps = cfg.epsilon

        for ep in range(cfg.episodes):
            state = self.env.reset(rng)
            traces = np.zeros(self.env.num_states)
            # Track when each state's trace was last set, for n-truncation
            trace_age = np.full(self.env.num_states, cfg.n + 1, dtype=int)
            total_reward = 0.0

            for step in range(cfg.max_steps_per_episode):
                action = self._epsilon_greedy(state, eps, rng)
                next_state, reward, done = self.env.simulate_step(state, action, rng)
                total_reward += reward
                self._update_q_from_v(state, action, reward, next_state, done)

                # TD error (one-step)
                td_error = reward + cfg.gamma * self.V[next_state] * (1 - done) - self.V[state]

                # Update trace for current state
                if cfg.trace_type == TraceType.REPLACING:
                    traces[state] = 1.0
                else:
                    traces[state] += 1.0
                trace_age[state] = 0

                # Update V for all states with active traces
                self.V += cfg.alpha * td_error * traces

                # Age and decay traces
                traces *= cfg.gamma
                trace_age += 1

                # Truncate: zero out traces older than n steps
                traces[trace_age >= cfg.n] = 0.0

                state = next_state
                if done:
                    break

            self.episode_returns.append(total_reward)
            self.episode_lengths.append(step + 1)
            eps = max(cfg.epsilon_min, eps * cfg.epsilon_decay)

    # ------------------------------------------------------------------
    # TD(λ) Forward-View
    # ------------------------------------------------------------------

    def _train_td_lambda_forward(self) -> None:
        """TD(λ) forward-view: λ-weighted average of all n-step returns.

        For each step t, the λ-return is:
            G(λ)_t = (1-λ) Σ_{n=1}^{T-t-1} λ^(n-1) G(n)_t  +  λ^(T-t-1) G_t

        where G_t is the full Monte Carlo return.

        This blends short-term (TD-like) and long-term (MC-like) targets.
        Must wait for the full episode to compute.

        We compute this efficiently by working backward through the episode.
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
                self._update_q_from_v(state, action, reward, next_state, done)
                state = next_state
                if done:
                    break

            T = len(states)
            terminal_state = state

            # Compute λ-return for each timestep
            # Work backward: G(λ)_t = R_{t+1} + γ[(1-λ)V(S_{t+1}) + λ G(λ)_{t+1}]
            # This recursive form is efficient and equivalent to the weighted sum.
            lambda_returns = np.zeros(T)

            # Bootstrap: G(λ)_T = V(S_T) if not terminal, else 0
            if done:
                g_lambda_next = 0.0
            else:
                g_lambda_next = self.V[terminal_state]

            for t in range(T - 1, -1, -1):
                if t == T - 1:
                    # Last step: next state is terminal_state
                    next_v = 0.0 if done else self.V[terminal_state]
                else:
                    next_v = self.V[states[t + 1]]

                # G(λ)_t = R_{t+1} + γ·[(1-λ)·V(S_{t+1}) + λ·G(λ)_{t+1}]
                g_lambda = rewards[t] + cfg.gamma * (
                    (1 - cfg.lam) * next_v + cfg.lam * g_lambda_next
                )
                lambda_returns[t] = g_lambda
                g_lambda_next = g_lambda

            # Update V for each state visited
            for t in range(T):
                self.V[states[t]] += cfg.alpha * (lambda_returns[t] - self.V[states[t]])

            self.episode_returns.append(sum(rewards))
            self.episode_lengths.append(T)
            eps = max(cfg.epsilon_min, eps * cfg.epsilon_decay)

    # ------------------------------------------------------------------
    # TD(λ) Backward-View
    # ------------------------------------------------------------------

    def _train_td_lambda_backward(self) -> None:
        """TD(λ) backward-view: online updates with eligibility traces.

        At each step:
            1. Compute one-step TD error: δ = R + γV(S') - V(S)
            2. Bump trace for current state: e(S) += 1
            3. Update ALL states: V(s) += α·δ·e(s)
            4. Decay ALL traces: e(s) *= γλ

        The trace e(s) remembers how recently/frequently state s was visited.
        Recent states get more credit for the current TD error.

        This is mathematically equivalent to the forward-view over a
        complete episode, but can update at every single step (online).
        """
        cfg = self.config
        rng = default_rng(cfg.seed)
        eps = cfg.epsilon

        for ep in range(cfg.episodes):
            state = self.env.reset(rng)
            traces = np.zeros(self.env.num_states)
            total_reward = 0.0

            for step in range(cfg.max_steps_per_episode):
                action = self._epsilon_greedy(state, eps, rng)
                next_state, reward, done = self.env.simulate_step(state, action, rng)
                total_reward += reward
                self._update_q_from_v(state, action, reward, next_state, done)

                # One-step TD error
                td_error = reward + cfg.gamma * self.V[next_state] * (1 - done) - self.V[state]

                # Update trace for current state
                if cfg.trace_type == TraceType.REPLACING:
                    traces[state] = 1.0
                else:
                    traces[state] += 1.0

                # Update V for all states proportional to their trace
                self.V += cfg.alpha * td_error * traces

                # Decay all traces by γλ
                traces *= cfg.gamma * cfg.lam

                state = next_state
                if done:
                    break

            self.episode_returns.append(total_reward)
            self.episode_lengths.append(step + 1)
            eps = max(cfg.epsilon_min, eps * cfg.epsilon_decay)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, episodes: int | None = None, seed: int = 9999) -> Dict[str, float]:
        """Evaluate greedy policy derived from Q."""
        n = episodes or self.config.eval_episodes
        rng = default_rng(seed)
        returns = []

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

        return {
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
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
                NpzKey.V: self.V,
                NpzKey.Q: self.Q,
                NpzKey.EPISODE_RETURNS: np.array(self.episode_returns),
                NpzKey.EPISODE_LENGTHS: np.array(self.episode_lengths),
                NpzKey.METADATA: np.array(json.dumps(asdict(self.config))),
            },
        )
        return out

    def load(self, path: str | Path) -> None:
        data = np.load(path, allow_pickle=False)
        self.V = data[NpzKey.V]
        self.Q = data[NpzKey.Q]
        self.episode_returns = data[NpzKey.EPISODE_RETURNS].tolist()
        self.episode_lengths = data[NpzKey.EPISODE_LENGTHS].tolist()
