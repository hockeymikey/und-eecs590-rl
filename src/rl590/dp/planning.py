from __future__ import annotations

import numpy as np


def _q_from_v(P: np.ndarray, R: np.ndarray, gamma: float, V: np.ndarray) -> np.ndarray:
    """Compute Q[s, a] from V and tabular model P[a, s, s_next], R[s, a]."""
    expected_future = np.einsum("ask,k->as", P, V).T
    return R + gamma * expected_future


def policy_evaluation(
    P: np.ndarray,
    R: np.ndarray,
    gamma: float,
    policy: np.ndarray,
    epsilon: float = 1e-8,
    max_iterations: int = 10_000,
) -> tuple[np.ndarray, list[float]]:
    num_states = R.shape[0]
    V = np.zeros(num_states, dtype=float)
    deltas: list[float] = []

    for _ in range(max_iterations):
        q_values = _q_from_v(P, R, gamma, V)
        updated = q_values[np.arange(num_states), policy]
        delta = float(np.max(np.abs(updated - V)))
        V = updated
        deltas.append(delta)

        if delta < epsilon:
            break

    return V, deltas


def policy_improvement(
    P: np.ndarray,
    R: np.ndarray,
    gamma: float,
    V: np.ndarray,
    policy: np.ndarray,
) -> tuple[np.ndarray, bool]:
    q_values = _q_from_v(P, R, gamma, V)
    greedy = np.argmax(q_values, axis=1)
    stable = np.array_equal(greedy, policy)
    return greedy, stable


def q_policy_improvement(Q: np.ndarray, policy: np.ndarray) -> tuple[np.ndarray, bool]:
    greedy = np.argmax(Q, axis=1)
    stable = np.array_equal(greedy, policy)
    return greedy, stable


def value_iteration(
    P: np.ndarray,
    R: np.ndarray,
    gamma: float,
    epsilon: float = 1e-8,
    max_iterations: int = 10_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float], int]:
    num_states = R.shape[0]
    V = np.zeros(num_states, dtype=float)
    deltas: list[float] = []

    iterations = 0
    for iteration in range(max_iterations):
        q_values = _q_from_v(P, R, gamma, V)
        updated = np.max(q_values, axis=1)
        delta = float(np.max(np.abs(updated - V)))
        V = updated
        deltas.append(delta)
        iterations = iteration + 1

        if delta < epsilon:
            break

    Q = _q_from_v(P, R, gamma, V)
    policy = np.argmax(Q, axis=1)
    return V, Q, policy, deltas, iterations


def policy_iteration(
    P: np.ndarray,
    R: np.ndarray,
    gamma: float,
    epsilon: float = 1e-8,
    max_iterations: int = 1_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float], int]:
    num_states = R.shape[0]
    policy = np.zeros(num_states, dtype=int)
    eval_deltas: list[float] = []

    iterations = 0
    for iteration in range(max_iterations):
        V, deltas = policy_evaluation(P, R, gamma, policy, epsilon=epsilon)
        eval_deltas.extend(deltas)

        improved_policy, stable = policy_improvement(P, R, gamma, V, policy)
        policy = improved_policy
        iterations = iteration + 1

        if stable:
            break

    Q = _q_from_v(P, R, gamma, V)
    return V, Q, policy, eval_deltas, iterations


def q_value_policy_iteration(
    P: np.ndarray,
    R: np.ndarray,
    gamma: float,
    epsilon: float = 1e-8,
    max_iterations: int = 5_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float], int]:
    """
    Policy iteration variant where policy improvement is driven directly by Q-values.
    Q(s, a) <- R(s, a) + gamma * E[V_pi(s')],  V_pi(s') = Q(s', pi(s')).
    """
    num_states, num_actions = R.shape
    Q = np.zeros((num_states, num_actions), dtype=float)
    policy = np.zeros(num_states, dtype=int)

    deltas: list[float] = []
    iterations = 0

    for iteration in range(max_iterations):
        prior_Q = Q.copy()
        v_under_policy = Q[np.arange(num_states), policy]

        expected_future = np.einsum("ask,k->as", P, v_under_policy).T
        Q = R + gamma * expected_future

        improved_policy, stable = q_policy_improvement(Q, policy)
        policy = improved_policy

        delta = float(np.max(np.abs(Q - prior_Q)))
        deltas.append(delta)
        iterations = iteration + 1

        if stable and delta < epsilon:
            break

    V = Q[np.arange(num_states), policy]
    return V, Q, policy, deltas, iterations
