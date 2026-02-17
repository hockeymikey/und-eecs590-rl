import numpy as np
from abc import ABC, abstractmethod

class MarkovDecisionProcess(ABC):
    def __init__(self, num_states, num_actions, gamma=0.9):
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma

        # P becomes a 3D tensor: P[action, current_state, next_state]
        # Or a dictionary: P[action] = Matrix(N x N)
        self.P = None
        self.R = None # R can be R[s] or R[s, a] depending on definition

    @abstractmethod
    def generate_transitions(self):
        """Define P(s' | s, a)"""
        pass

    # ...