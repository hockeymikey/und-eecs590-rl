import numpy as np
from mdp import MarkovDecisionProcess

class WindyChasm(MarkovDecisionProcess):
    def __init__(self, p_center=0.1, goal_row=None, crash_reward=-100, step_reward=-1):
        """
        Grid: 20 rows (0-19), 7 cols (0-6).
        Start: (0, 3) [cite: 14]
        Goal: (19, n) [cite: 14]
        Crash: j <= 0 or j >= 6 [cite: 27]
        """
        self.rows = 20
        self.cols = 7
        self.start_pos = (0, 3)
        self.target_goal = (19, goal_row) if goal_row is not None else (19, 3)
        self.p_center = p_center

        # Rewards [cite: 32-35]
        self.r_crash = crash_reward
        self.r_goal = 100 # Assumed +R
        self.r_step = step_reward

        # Actions: 0=Forward, 1=Left, 2=Right [cite: 16]
        super().__init__(num_states=self.rows * self.cols, num_actions=3, gamma=0.9)

        self.generate_transitions()
        self.generate_rewards()

    def _to_idx(self, i, j):
        return i * self.cols + j

    def _from_idx(self, idx):
        return idx // self.cols, idx % self.cols

    def _get_wind_prob(self, j):
        """
        Calculates p(j) based on distance from center j=3 [cite: 23-24].
        Formula: p(j) = B^E(j) where B = p(3) and E(j) = 1 / (1 + (j-3)^2)
        """
        dist_sq = (j - 3) ** 2
        exponent = 1.0 / (1.0 + dist_sq)
        return self.p_center ** exponent

    def generate_transitions(self):
        # Initialize P: (Actions, States, Next_States)
        self.P = np.zeros((self.num_actions, self.num_states, self.num_states))

        for s in range(self.num_states):
            i, j = self._from_idx(s)

            # --- ABSORBING STATES ---
            # If in a wall (0, 6) or at Goal, we stay there forever (Game Over)
            if j == 0 or j == 6 or (i, j) == self.target_goal:
                self.P[:, s, s] = 1.0
                continue

            for a in range(self.num_actions):
                # --- PHASE 1: Deterministic Action [cite: 18] ---
                next_i, next_j = i, j
                if a == 0:   next_i += 1  # Forward (Increase i)
                elif a == 1: next_j -= 1  # Left (Decrease j)
                elif a == 2: next_j += 1  # Right (Increase j)

                # Check for immediate wall crash before wind
                next_i = min(next_i, 19) # Clamp forward progress
                if next_j <= 0: next_j = 0 # Crash Left
                if next_j >= 6: next_j = 6 # Crash Right

                # If we already crashed or hit goal, wind doesn't matter
                if next_j == 0 or next_j == 6 or (next_i, next_j) == self.target_goal:
                    dest_idx = self._to_idx(next_i, next_j)
                    self.P[a, s, dest_idx] = 1.0
                    continue

                # --- PHASE 2: Wind Effects [cite: 19-22] ---
                # Wind applies based on the position AFTER the move
                p = self._get_wind_prob(next_j)

                # Probabilities for wind shifts:
                # 1. Stay: (1-p)(1-p^2)
                # 2. +/- 1: p (split 50/50)
                # 3. +/- 2: (1-p)p^2 (split 50/50)

                shifts = {
                    0:  (1 - p) * (1 - p**2),
                    -1: p / 2.0,
                    1:  p / 2.0,
                    -2: ((1 - p) * p**2) / 2.0,
                    2:  ((1 - p) * p**2) / 2.0
                }

                for delta, prob in shifts.items():
                    if prob == 0: continue

                    wind_i = next_i
                    wind_j = next_j + delta

                    # Check bounds/crashes again after wind
                    if wind_j <= 0: wind_j = 0
                    if wind_j >= 6: wind_j = 6

                    final_idx = self._to_idx(wind_i, wind_j)
                    self.P[a, s, final_idx] += prob

    def generate_rewards(self):
        # R[s, a] = Sum(P(s'|s,a) * Reward(s')) [cite: 9]
        self.R = np.zeros((self.num_states, self.num_actions))

        for s in range(self.num_states):
            i, j = self._from_idx(s)

            # No rewards if already done
            if j == 0 or j == 6 or (i, j) == self.target_goal:
                continue

            for a in range(self.num_actions):
                expected_r = 0
                for s_prime in range(self.num_states):
                    prob = self.P[a, s, s_prime]
                    if prob > 0:
                        i_p, j_p = self._from_idx(s_prime)

                        r = self.r_step # Step cost [cite: 35]

                        if (i_p, j_p) == self.target_goal:
                            r += self.r_goal # Goal bonus [cite: 33]
                        elif j_p == 0 or j_p == 6:
                            r += self.r_crash # Crash penalty [cite: 34]

                        expected_r += prob * r

                self.R[s, a] = expected_r