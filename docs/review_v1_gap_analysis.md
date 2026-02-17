# V1 Gap Analysis (Code + Rubric)

## High Priority Gaps

1. Citation/collaboration section is still a placeholder.
   - Evidence: `README.md:70` and `README.md:72`
   - Impact: rubric explicitly calls this out under documentation.

## Medium Priority Gaps

1. Legacy M2 script still demonstrates value iteration only.
   - Evidence: `M2/main.py:32`, `M2/main.py:39`, `M2/main.py:46`
   - Impact: could understate current DP coverage if reviewer only checks legacy folder.

2. Legacy Mini 1 draft file is not executable.
   - Evidence: `M1/code/q7-2.py:3` (`np` undefined)
   - Impact: not part of active framework, but still visible noise if whole-repo scripts are run.

3. V2-prep algorithms remain missing (non-blocking for V1 grading).
   - Evidence: requested list in `v1.MD:15` through `v1.MD:27`
   - Missing: Monte Carlo, TD(n), TD(lambda), Sarsa(n), Sarsa(lambda), Q-learning.

## Covered in Current Refactor

1. Dependency declaration: `requirements.txt`
2. Clean active package layout: `src/rl590/`
3. DP coverage:
   - Value iteration: `src/rl590/dp/planning.py:56`
   - Policy iteration + value-based improvement: `src/rl590/dp/planning.py:84`
   - Q-value policy improvement: `src/rl590/dp/planning.py:111`
4. Agent framework with hyperparameters + train/eval + saved artifacts:
   - Config: `src/rl590/agents/planning_agent.py:12`
   - Train/eval: `src/rl590/agents/planning_agent.py:33` and `src/rl590/agents/planning_agent.py:78`
   - Save/load policy-value artifacts: `src/rl590/agents/planning_agent.py:109` and `src/rl590/agents/planning_agent.py:127`
   - CLI entrypoint: `src/rl590/cli.py:50`
5. Notes system and templates:
   - `docs/notes/README.md`
   - `docs/notes/templates/daily-note.md`
   - `docs/notes/templates/concept-note.md`
6. Model belief updates (new):
   - Belief estimator with transition/reward updates: `src/rl590/model/belief.py`
   - Belief bootstrap CLI path: `src/rl590/cli.py` (`bootstrap-model`)
