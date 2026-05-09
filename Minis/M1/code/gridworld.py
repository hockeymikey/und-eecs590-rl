import numpy as np

from markovprocess import MarkovProcess


class GridWorld(MarkovProcess):
    def __init__(self, grid_input, gamma=0.9):
        # CHECK: Is the input a string mask? If so, parse it.
        if isinstance(grid_input, str):
            self.grid = self._parse_mask(grid_input)
        else:
            # Assume it's already a list/array
            self.grid = np.array(grid_input)

        rows, cols = self.grid.shape
        num_states = rows * cols

        super().__init__(num_states, gamma)

        # Build Matrices
        self.R = self.generate_rewards()
        self.P = self.generate_transitions(rows, cols)

    def _parse_mask(self, mask_str):
        """Helper: Converts string mask to numpy array"""
        lines = [line.strip() for line in mask_str.strip().split('\n')]
        matrix = []
        for line in lines:
            if line:
                # Handle 'inf' or numbers
                row = []
                for x in line.split():
                    if x == "-inf" or x == "-∞":
                        row.append(float('-inf'))
                    else:
                        row.append(float(x))
                matrix.append(row)
        return np.array(matrix)

    def generate_rewards(self):
        return self.grid.flatten()

    def generate_transitions(self, rows, cols):
        P = np.zeros((self.num_states, self.num_states))
        for r in range(rows):
            for c in range(cols):
                current_idx = r * cols + c

                # If this state is a Wall (-inf), it traps you (or you can't leave)
                # For this HW, let's assume you can't be IN a wall, 
                # but if you start there, stay there.
                if self.grid[r, c] == float('-inf'):
                    P[current_idx, current_idx] = 1.0
                    continue

                neighbors = self._get_valid_neighbors(r, c, rows, cols)
                if not neighbors:
                    P[current_idx, current_idx] = 1.0
                else:
                    prob = 1.0 / len(neighbors)
                    for (nr, nc) in neighbors:
                        next_idx = nr * cols + nc
                        P[current_idx, next_idx] = prob
        return P

    def _get_valid_neighbors(self, r, c, max_r, max_c):
        moves = [(-1,0), (1,0), (0,-1), (0,1)] # U, D, L, R
        valid = []
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            # Check bounds
            if 0 <= nr < max_r and 0 <= nc < max_c:
                # Check for Walls (-inf) in neighbors
                if self.grid[nr, nc] != float('-inf'):
                    valid.append((nr, nc))
        return valid