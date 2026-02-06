# Define a simple test mask
from gridworld import GridWorld

mask_str = """
0 0 0
0 100 0
0 0 0
"""

# Experiment Loop
gammas = [0.1, 0.5, 0.9, 0.99]

print(f"{'Gamma':<10} | {'Goal Value':<12} | {'Neighbor Value':<12}")
print("-" * 40)

for g in gammas:
    # Initialize world with current gamma
    world = GridWorld(mask_str, gamma=g)

    # Solve
    V = world.solve()

    # Extract values (Center is goal, (0,1) is a neighbor)
    goal_val = V[4] # Center of 3x3
    neighbor_val = V[1] # Top middle

    print(f"{g:<10} | {goal_val:.2f}         | {neighbor_val:.2f}")