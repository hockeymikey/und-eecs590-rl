from windy_world import WindyChasm
import numpy as np

def print_policy(world):
    print(f"\nPolicy Visualization (Start: {world.start_pos}, Goal: {world.target_goal})")
    print("Symbols: ^ = Forward, < = Left, > = Right, X = Wall/Crash, G = Goal")
    print("-" * 40)

    # Text visualization
    # Actions: 0=F, 1=L, 2=R
    chars = {0: '^', 1: '<', 2: '>'}

    for i in range(world.rows):
        row_str = f"Row {i:02d} |"
        for j in range(world.cols):
            state_idx = world._to_idx(i, j)

            if j == 0 or j == 6:
                row_str += "  X " # Wall
            elif (i, j) == world.target_goal:
                row_str += "  G "
            else:
                action = world.policy[state_idx]
                row_str += f"  {chars[action]} "
        print(row_str + "|")
    print("-" * 40)

def main():
    # --- Q1 Part 1: Standard Parameters ---
    print(">>> Experiment 1: Standard Wind (p=0.1)")
    env1 = WindyChasm(p_center=0.1, goal_row=3)
    env1.value_iteration()
    print_policy(env1)

    # --- Q1 Part 2: Increased Wind ---
    # "How does the policy change as the wind parameter p increases?"
    print("\n>>> Experiment 2: High Wind (p=0.5)")
    env2 = WindyChasm(p_center=0.5, goal_row=3)
    env2.value_iteration()
    print_policy(env2)

    # --- Q1 Part 3: Reduced Crash Penalty ---
    # "What about setting the crashing reward to -1?"
    print("\n>>> Experiment 3: Low Risk Aversion (Crash Reward = -1)")
    env3 = WindyChasm(p_center=0.1, crash_reward=-1)
    env3.value_iteration()
    print_policy(env3)

if __name__ == "__main__":
    main()