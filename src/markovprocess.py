import numpy as np
from abc import ABC, abstractmethod

class MarkovProcess(ABC):
    def __init__(self, num_states, gamma=0.9):
        self.num_states = num_states
        self.gamma = gamma
        self.P = None # Transition Matrix
        self.R = None # Reward Vector
        self.V = None # Value Vector

    @abstractmethod
    def generate_transitions(self):
        """Force child classes to define how P is built."""
        pass

    @abstractmethod
    def generate_rewards(self):
        """Force child classes to define how R is built."""
        pass

    def solve(self):
        """
        The 'Unifying Method'  logic goes here.
        Since P and R are just matrices now, we can solve V
        without knowing if this is a grid or a graph.
        """
        I = np.eye(self.num_states)
        try:
            # Using the inversion method (Step 3) [cite: 6]
            self.V = np.linalg.inv(I - self.gamma * self.P) @ self.R
            return self.V
        except np.linalg.LinAlgError:
            print("Matrix singular. Use iterative method.")
            return None