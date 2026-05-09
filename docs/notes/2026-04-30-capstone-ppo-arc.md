# Daily Note - 2026-04-30

## Focus
- Wrap up overnight PPO training on the foundation env (rinkgym RinkEnv).
- Decide whether to do another training run, or call it and start the writeup.

## Quick Scratch

The capstone arc so far is three connected stories. Captured in detail in
`docs/technical-challenges.md` under "Capstone — Foundation Env Training Runs";
this note is the chronological log with raw numbers.

### Run 1 (vanilla SB3 PPO defaults)
- Started ~01:33 on 2026-04-29, killed at 750k of 1.5M steps after observing
  policy collapse.
- Key metrics at collapse: `std=0.00123`, `approx_kl=2.29`,
  `clip_fraction=0.82`, `ep_rew_mean ~30` (plateaued).
- Diagnosis: default `ent_coef=0.0` doesn't prevent Gaussian std from
  collapsing on continuous-control tasks.

### Run 2 (entropy-regularized, with KL guardrail)
- Two-line diff from Run 1: `ent_coef=0.01`, `target_kl=0.05`.
- Run 2 first half (`ppo_v2_20260429T100022`) hit 500k steps before stopping.
  Resumed from the 500k checkpoint via `train_ppo_v2_resume.py`.
- Final run (`ppo_v2_resume_20260430T011531`) reached the 1.5M target.
- Late metrics: `std=1.13`, `approx_kl=0.04`, `clip_fraction=0.21`,
  `ep_rew_mean ~33`. Healthy throughout; `target_kl` guardrail fired
  routinely (working as intended).
- Deterministic eval (one seed): `steps=400, reward=24.813`.
- Comparison reference printed by the script: random≈7, Run 1 PPO≈30,
  coverage_path≈53.

### Reward shape investigation
- Triggered by the question "shouldn't reward be higher if we want complete
  coverage?" — instead of sweeping checkpoints, audited the reward.
- `RinkEnv._compute_reward` (rink_env.py:694) gives +0.1 per *new center-cell*
  visit, -0.05 per repeat. Uses `visited_mask` (center trace) not
  `brush_visited_mask` (actual coverage).
- Elevation reward (`total_elevation_change / 1000.0`) is in the 1e-6 range
  per step — drowned out by the +0.1 cell-visit bonus.
- Coverage_path's score of 53 is essentially "+0.1 × 620 unique cells - time
  penalty" — almost zero credit for actual resurfacing.
- Already-documented in rinkgym's `gymnasium-api-conformance.md` Item 4 as
  a metrics issue; turns out it's also in the reward.

## Experiments Run

- **Run 1 (collapsed reference)**: `python -m rinkgym.training.train_ppo`
  - Hyperparameters: SB3 defaults except lr=3e-4, n_steps=2048, batch_size=64,
    n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2.
  - Result: collapse at ~770k. Last checkpoint 750k.

- **Run 2 (entropy fix, two segments)**:
  `python -m rinkgym.training.train_ppo_v2`
  then resumed via
  `python -m rinkgym.training.train_ppo_v2_resume`.
  - Hyperparameters: same as Run 1 plus `ent_coef=0.01`, `target_kl=0.05`.
  - Result: 1.5M steps reached. Stable training. Deterministic eval = 24.813.

- **Reward inspection** (no run): read `_compute_reward` and traced where the
  numbers come from for the coverage_path baseline.

## Open Questions

- [ ] Sign of `collision_penalty` in `_apply_collision_response_v2` — verify
  it's negative (penalty, not reward) when fixing the reward.
- [ ] How to weight elevation change vs cell-coverage so neither dominates.
  Probably want elevation change to be the dominant signal with coverage as
  a shaping aid, not the reverse.
- [ ] "Higher demand areas" — does the env expose a per-cell demand surface
  we can scale the elevation reward by? If yes, that's the natural place for
  "reward hitting higher demand areas more." If not, this becomes a rinkgym
  feature add.
- [ ] Whether to run a Run 3 with the fixed reward, or accept that the
  current story (Run 1 collapse → Run 2 fix → reward bug discovery) is
  already a complete capstone narrative. Capstone due 2026-05-07.

## Next Actions

- [ ] Start a fresh session in the rinkgym repo to refine the reward
  function. Scope: rewrite `_compute_reward` to use `brush_visited_mask`,
  weight elevation change so it's a real signal, audit collision sign,
  consider demand-weighted shaping. Keep the Minis session focused on the
  capstone writeup.
- [ ] After reward fix, decide whether time permits a Run 3 before the
  capstone deadline.
- [ ] Begin drafting capstone writeup using the three-story arc; the
  "reward-shape bug discovered through diagnostic inspection" story is
  the most compelling part and should be foregrounded, not buried.
