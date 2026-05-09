# Mini Project 2: MDP and Windy Chasm
**Name:** Michael Hajostek

I'll be honest, this isn't my best work and I was very confused because you haven't posted any of the lectures yet so there is no way for me to really learn. I have been going through David Silver's lecture recently, pretty far through it but more to do. There is a more recent series DeepMind put out 3 years vs the 9yo one of David Silver's.

## Problem 1: Navigating A Windy Chasm
I formulated the problem as an MDP defined by the tuple $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$.

### 1. Methodology
I utilized **Value Iteration** to solve for the optimal policy. I chose this over matrix inversion because the action space introduces complexity that makes iterative updates more scalable (complexity $O(S^2 A)$ per iteration).

The update rule used was:
$$V_{k+1}(s) = \max_{a} \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V_k(s')]$$

### 2. Policy Analysis (Q1)
* **Standard Behavior ($p=0.1$):**
  The drone generally prefers the center lane ($j=3$). When it deviates to $j=2$ or $j=4$, it uses the **Left** or **Right** actions immediately to correct back to the center. This minimizes the risk of the wind pushing it into the walls ($j=0$ or $j=6$).

* **High Wind ($p=0.5$):**
  As $p$ increases, the stochasticity makes the center dangerous because a "strong wind" event (moving 2 spaces) becomes more likely. The policy becomes extremely conservative, often trying to "fight" the wind proactively even if it slows forward progress.

* **Low Crash Penalty ($r=-1$):**
  When the penalty for crashing is merely -1 (same as a step cost), the drone takes massive risks. It behaves almost as if the walls don't exist, prioritizing the shortest path (Straight Forward) regardless of current wind conditions, because crashing is no worse than taking a single step.

### 3. Code Implementation
The source code consists of `mdp.py` (the solver) and `windy_world.py` (the environment). The `main.py` script generates the ASCII policies below.

```
>>> Experiment 1: Standard Wind (p=0.1)

Policy Visualization (Start: (0, 3), Goal: (19, 3))
Symbols: ^ = Forward, < = Left, > = Right, X = Wall/Crash, G = Goal
----------------------------------------
Row 00 |  X   >   >   ^   <   <   X |
Row 01 |  X   >   >   ^   <   <   X |
Row 02 |  X   >   >   ^   <   <   X |
Row 03 |  X   >   >   ^   <   <   X |
Row 04 |  X   >   >   ^   <   <   X |
Row 05 |  X   >   >   ^   <   <   X |
Row 06 |  X   >   >   ^   <   <   X |
Row 07 |  X   >   >   ^   <   <   X |
Row 08 |  X   >   >   ^   <   <   X |
Row 09 |  X   >   >   ^   <   <   X |
Row 10 |  X   >   >   ^   <   <   X |
Row 11 |  X   >   >   ^   <   <   X |
Row 12 |  X   >   >   ^   <   <   X |
Row 13 |  X   >   >   ^   <   <   X |
Row 14 |  X   >   >   ^   <   <   X |
Row 15 |  X   >   >   ^   <   <   X |
Row 16 |  X   >   >   ^   <   <   X |
Row 17 |  X   >   >   ^   <   <   X |
Row 18 |  X   >   >   ^   <   <   X |
Row 19 |  X   >   >   G   <   <   X |
----------------------------------------

>>> Experiment 2: High Wind (p=0.5)

Policy Visualization (Start: (0, 3), Goal: (19, 3))
Symbols: ^ = Forward, < = Left, > = Right, X = Wall/Crash, G = Goal
----------------------------------------
Row 00 |  X   >   >   ^   <   <   X |
Row 01 |  X   >   >   ^   <   <   X |
Row 02 |  X   >   >   ^   <   <   X |
Row 03 |  X   >   >   ^   <   <   X |
Row 04 |  X   >   >   ^   <   <   X |
Row 05 |  X   >   >   ^   <   <   X |
Row 06 |  X   >   >   ^   <   <   X |
Row 07 |  X   >   >   ^   <   <   X |
Row 08 |  X   >   >   ^   <   <   X |
Row 09 |  X   >   >   ^   <   <   X |
Row 10 |  X   >   >   ^   <   <   X |
Row 11 |  X   >   >   ^   <   <   X |
Row 12 |  X   >   >   ^   <   <   X |
Row 13 |  X   >   >   ^   <   <   X |
Row 14 |  X   >   >   ^   <   <   X |
Row 15 |  X   >   ^   ^   ^   <   X |
Row 16 |  X   >   ^   ^   ^   <   X |
Row 17 |  X   >   ^   ^   ^   <   X |
Row 18 |  X   >   >   ^   <   <   X |
Row 19 |  X   >   >   G   <   <   X |
----------------------------------------

>>> Experiment 3: Low Risk Aversion (Crash Reward = -1)

Policy Visualization (Start: (0, 3), Goal: (19, 3))
Symbols: ^ = Forward, < = Left, > = Right, X = Wall/Crash, G = Goal
----------------------------------------
Row 00 |  X   >   ^   ^   ^   <   X |
Row 01 |  X   >   ^   ^   ^   <   X |
Row 02 |  X   >   >   ^   <   <   X |
Row 03 |  X   >   >   ^   <   <   X |
Row 04 |  X   >   >   ^   <   <   X |
Row 05 |  X   >   >   ^   <   <   X |
Row 06 |  X   >   >   ^   <   <   X |
Row 07 |  X   >   >   ^   <   <   X |
Row 08 |  X   >   >   ^   <   <   X |
Row 09 |  X   >   >   ^   <   <   X |
Row 10 |  X   >   >   ^   <   <   X |
Row 11 |  X   >   >   ^   <   <   X |
Row 12 |  X   >   >   ^   <   <   X |
Row 13 |  X   >   >   ^   <   <   X |
Row 14 |  X   >   >   ^   <   <   X |
Row 15 |  X   >   >   ^   <   <   X |
Row 16 |  X   >   >   ^   <   <   X |
Row 17 |  X   >   >   ^   <   <   X |
Row 18 |  X   >   >   ^   <   <   X |
Row 19 |  X   >   >   G   <   <   X |
----------------------------------------
```

### Q. b

The drone preference for the center of the chasm.

The wind functio nis lowest at the center and increases towards the walls.
Meaining that the gust and risk near the edges is highest, so it maximizes its distance
from the fatal "crash" states.

### Q. c

The policy becomes more conservative. With high p, the strong wind events increases.
The safe corridor narrows so the risk is higher meaning the agent will force to be more 
aggressive to stay in the middle.

### Q. D

It makes the agent risk neutral or risk seeking even. There is no risk of dying so it won't try to stay in the middle
and will take more risks near the edges.