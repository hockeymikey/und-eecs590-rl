import numpy as np
from abc import ABC, abstractmethod

class MarkovDecisionProcess(ABC):
    """
    Abstract Base Class for Markov Decision Processes.
    """
    def __init__(self, num_states, num_actions, gamma=0.9):
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma

        # P[action, current_state, next_state]
        self.P = None
        # R[current_state, action] (Expected reward)
        self.R = None

        # The calculated policy: policy[state] -> action_index
        self.policy = np.zeros(num_states, dtype=int)
        # The calculated values: V[state] -> float
        self.V = np.zeros(num_states)

    @abstractmethod
    def generate_transitions(self):
        """Constructs self.P"""
        pass

    @abstractmethod
    def generate_rewards(self):
        """Constructs self.R"""
        pass

    def value_iteration(self, epsilon=1e-6, max_iter=10000):
        """
        Solves for V* using the Bellman Optimality Equation[cite: 11].
        V(s) = max_a [ R(s,a) + gamma * sum(P(s'|s,a) * V(s')) ]
        """
        self.V = np.zeros(self.num_states)

        for i in range(max_iter):
            prev_V = self.V.copy()

            # Vectorized Bellman Update:
            # 1. Compute expected future value for all (s, a): P @ V -> shape (A, S)
            # 2. Add expected immediate reward: R.T -> shape (A, S)
            # 3. Result is Q_values of shape (A, S)
            future_values = self.P @ self.V
            Q_values = self.R.T + self.gamma * future_values

            # Max over actions (axis 0) to get V(s)
            self.V = np.max(Q_values, axis=0)

            # Check convergence
            if np.max(np.abs(self.V - prev_V)) < epsilon:
                # Extract the optimal policy
                self.policy = np.argmax(Q_values, axis=0)
                return self.V

        print("Warning: Value Iteration did not converge.")
        return self.V