# `ZambGymEnv` — the foundation gym env

`src/rl590/envs/zamb_gym.py` implements a continuous-control ice-resurfacing
environment. It is a stripped-down, modified adaptation of a private prototype;
the Minis copy is self-contained (no imports back into that workspace) and uses
a deliberately simpler linear refreeze placeholder plus course-specific reward
coefficients.

This page is the reference for poking at the env to verify correctness.

## Spaces

| Component | Type | Shape / range |
|---|---|---|
| Observation | `gym.spaces.Box(float32)` | `(H, W, 3)` channels-last; default `(129, 304, 3)` |
| Action | `gym.spaces.Box(float32)` | `(2,)` in `[-1.0, 1.0]` |

### Observation channels (channels-last)

| Channel | Source | Range | Description |
|---|---|---|---|
| 0 | `surface.damage_mm / damage_max_mm` | `[0, 1]` | Per-cell damage magnitude, clamped & normalised. |
| 1 | one-hot agent-pose | `{0, 1}` | A single hot pixel at the nav cell containing the chassis center. |
| 2 | `surface.refreeze_progress` | `[0, 1]` | Linear refreeze counter; `>= REFREEZE_READY_THRESHOLD` means "ready". |

Cells outside the rink mask are 0 in all channels.

### Action

`np.ndarray` of shape `(2,)` and dtype `float32`:

- `action[0]` — throttle, mapped to `[-deceleration_limit, +acceleration_limit]` × `vehicle.weight_kg` for the drive force.
- `action[1]` — steering, mapped to `[-max_steering_angle, +max_steering_angle]` via the vehicle spec, with steering-rate limiting in the dynamics step.

## Reward

Computed in `ZambGymEnv._compute_reward`. Components:

| Name | Coefficient (class attr) | Default | Triggered by |
|---|---|---|---|
| `damage` | `COEF_DAMAGE_REDUCTION` | `1.0` | mm of damage cleared under the brush footprint this step. |
| `coverage` | `COEF_COVERAGE` | `0.001` | Nav cells newly under the brush footprint this step. |
| `refreeze` | `COEF_REFREEZE_BONUS` | `0.05` | Newly-crossed cells whose `refreeze_progress >= REFREEZE_READY_THRESHOLD`. |
| `redisturb` | `COEF_REDISTURB_PENALTY` | `-0.1` | Newly-crossed cells whose `refreeze_progress < REFREEZE_READY_THRESHOLD`. |
| `collision` | (from contact response) | — | Wall contacts: `-0.05` per colliding corner, clamped to `[-1.0, 0.0]`. |
| `step` | `COEF_STEP_PENALTY` | `-0.001` | Constant per-step penalty. |

Reward terms `refreeze` and `redisturb` only fire on **newly-crossed** cells
(`footprint & ~swept_mask`), not on every revisit. Override the coefficients
by subclassing or by setting them on an instance — they are class attributes.

The reward value and per-component breakdown are also returned in `info`:

```python
obs, reward, terminated, truncated, info = env.step(action)
info["reward_components"]  # dict with all six terms
```

## Episode termination

- `terminated=True` when `step_count >= max_steps` (passed in at construction).
- `truncated` is always `False` from inside the env — wrap with
  `gymnasium.wrappers.TimeLimit` if you want truncation semantics.

There is no out-of-bounds termination — wall contacts are softly resolved by
pushing the vehicle back along the board normal and damping velocity.

## Physics

- `vehicle: Zamboni552` — geometry spec lives in `src/rl590/envs/dynamics.py`.
- `dynamics_model: DynamicBicycleModel` — force-based bicycle model with linear
  tire forces, RK4 integration. Only the bicycle model is exposed in this
  course copy.
- Time step `dt = 0.5 s`.
- Surface state advances via `step_surface()` from `surface.py`:
  - Every step inside the rink mask: `refreeze_progress += REFREEZE_RATE` (0.01),
    clamped at 1.0.
  - Cells under the brush footprint also: damage reduced by `shave_depth_mm`
    (computed from speed/heading alignment/local damage), and
    `refreeze_progress` reset to 0.

## How to test it

### 1. Import & roundtrip

```fish
cd Minis
.venv/bin/python -c "
import numpy as np
from rl590.envs.zamb_gym import ZambGymEnv
e = ZambGymEnv(max_steps=20)
obs, info = e.reset(seed=0)
print('obs shape:', obs.shape, 'dtype:', obs.dtype)
print('action space:', e.action_space)
obs, r, term, trunc, info = e.step(np.array([0.5, 0.1], dtype=np.float32))
print('reward:', r, 'components:', info['reward_components'])
"
```

Expected: obs shape `(129, 304, 3)`, six reward-component keys (`damage`,
`coverage`, `refreeze`, `redisturb`, `collision`, `step`).

### 2. Determinism check

```fish
.venv/bin/python -c "
import numpy as np
from rl590.envs.zamb_gym import ZambGymEnv
def roll(seed):
    e = ZambGymEnv(max_steps=50)
    obs, _ = e.reset(seed=seed)
    rng = np.random.default_rng(seed)
    rets = 0.0
    for _ in range(50):
        a = rng.uniform(-1, 1, size=2).astype(np.float32)
        obs, r, term, trunc, _ = e.step(a)
        rets += r
        if term or trunc: break
    return rets
print(roll(0), roll(0), roll(1))  # first two must match, third can differ
"
```

### 3. Observation channels look right

```fish
.venv/bin/python -c "
import numpy as np
from rl590.envs.zamb_gym import ZambGymEnv
e = ZambGymEnv(max_steps=10)
obs, _ = e.reset(seed=0)
print('damage channel  range:', obs[..., 0].min(), obs[..., 0].max())
print('agent  channel  sum:  ', obs[..., 1].sum(), '(should be 1.0)')
print('refreeze channel range:', obs[..., 2].min(), obs[..., 2].max())
"
```

Expected: damage in `[0, ~0.2]` (initial uniform damage is 1.0 mm, normalised
by `damage_max_mm=5.0`); agent channel sums to `1.0`; refreeze channel starts
at `1.0` everywhere inside the rink.

### 4. Reward shaping smoke

Drive forward in a straight-ish line for 100 steps; episode return should be
strongly positive (the agent is clearing fresh damage).

```fish
.venv/bin/python -c "
import numpy as np
from rl590.envs.zamb_gym import ZambGymEnv
e = ZambGymEnv(max_steps=100)
obs, _ = e.reset(seed=0)
ret = 0.0; ncoll = 0
for s in range(100):
    obs, r, term, trunc, info = e.step(np.array([0.6, 0.0], dtype=np.float32))
    ret += r
    if info['reward_components']['collision'] < 0: ncoll += 1
    if term or trunc: break
print(f'return={ret:.1f}  collision_steps={ncoll}')
"
```

### 5. Run a tiny PPO smoke (TensorBoard for curves)

```fish
just gym-ppo-scratch 2048
```

(Recipe parameters are positional. `just --list` shows defaults like
`gym-ppo-scratch timesteps="50000"`; the `timesteps="50000"` is the *display*
of the default value, not the literal argument — type just the number.)

Outputs land in `training_runs/zamb_gym_ppo_v1/`:
- `actor.pt`, `critic.pt`, `optimizer.pt`, `hparams.json`, `training_log.json` — checkpoints
- `tb/<run-tag>/` — TensorBoard event files (per-rollout scalars: `rollout/mean_return`,
  `rollout/mean_return_last10`, `rollout/mean_length`, `rollout/episodes_completed`,
  `train/policy_loss`, `train/value_loss`, `train/entropy`)

To view the curves:

```fish
just gym-tb            # http://localhost:6006
```

Mean episode return over the smoke run should rise across the four rollouts.

### 6. BC + PPO end-to-end

```fish
# 1. drop coverage_path_v1.npz into assets/teacher/

# 2. fit BC actor (TB scalars: bc/loss, bc/nll, bc/entropy)
just gym-bc

# 3. PPO from scratch (baseline)
just gym-ppo-scratch

# 4. PPO with BC warmstart (compare to the scratch run in TensorBoard)
just gym-ppo-bc

# 5. view both runs side by side
just gym-tb
```

The two PPO runs land in separate `tb/<run-tag>/` subdirs (`scratch` and
`bcinit`), so TensorBoard plots them as separate curves on the same axes.

## Key class attributes you can tune

```python
ZambGymEnv.COEF_COVERAGE = 0.001
ZambGymEnv.COEF_DAMAGE_REDUCTION = 1.0
ZambGymEnv.COEF_REFREEZE_BONUS = 0.05
ZambGymEnv.COEF_REDISTURB_PENALTY = -0.1
ZambGymEnv.COEF_STEP_PENALTY = -0.001
```

From `surface.py`:

```python
REFREEZE_RATE = 0.01            # progress increment per step
REFREEZE_READY_THRESHOLD = 0.8  # ready when progress >= this
```

From the internal cfg (`ZambGymEnv.cfg`):

```python
rink_width_m = 60.96
rink_height_m = 25.9
corner_radius_m = 8.53
nav_pixels_per_meter = 5      # → nav grid (129, 304)
dt = 0.5                       # seconds per env step
damage_max_mm = 5.0            # normalises the damage observation channel
```

## What was deliberately changed from the prototype

- Renamed the environment class to `ZambGymEnv`.
- `AdaptiveState` → `IceSurfaceState`; `DegradationGenerator` → `DamageGenerator`.
- `brush_visited_mask` → `swept_mask`; `visited_mask` → `agent_visited`.
- Replaced the calibrated refreeze model with a deliberately simple linear
  placeholder (`REFREEZE_RATE`, `REFREEZE_READY_THRESHOLD`).
- Course-specific reward coefficients (see table above).
- Dropped the loose-snow channel and snow-removal reward term.
- Dropped the Dict observation (image + 6-vector); image-only now, with the
  agent pose encoded as a one-hot channel.
- Dropped the pluggable-dynamics interface; `DynamicBicycleModel` is hardcoded.
- Dropped the multi-mode initial-damage selector; `UniformDamageGenerator` is
  hardcoded.
- Dropped the matplotlib `render()` path.

## What stayed close to the prototype

- Action shape and continuous control semantics.
- `Zamboni552` vehicle spec (it's a real machine model).
- Dynamic bicycle force/slip equations and RK4 integration.
- Brush-footprint mask raster.
- Speed × heading-alignment × ice-condition shave-depth formula.
