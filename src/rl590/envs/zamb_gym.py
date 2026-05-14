"""ZambGymEnv — continuous-control ice-resurfacing gym env.

Self-contained course variant of the ice-resurfacing prototype. Keeps the
vehicle dynamics and brush-footprint mechanics, but omits the broader
prototype's optional surface channels, configurable damage modes, and dict
observation. Uses the simplified linear refreeze model from ``surface.py`` and
course-specific reward coefficients.

Action:
    Box(2,) in [-1, 1]: (throttle, steering).
Observation:
    Box(H, W, 3): image with channels (damage_norm, agent_pose,
    refreeze_progress) at nav-grid resolution.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl590.envs.dynamics import (
    DynamicBicycleModel,
    Zamboni552,
    VehicleAction,
    VehicleDynamicsParams,
    VehicleState,
)
from rl590.envs.surface import (
    REFREEZE_READY_THRESHOLD,
    IceSurfaceState,
    RinkDimensions,
    UniformDamageGenerator,
    UniformRandomStartPoseSampler,
    create_rink_mask,
    get_wall_normal_angle,
    step_surface,
)


# Blade and collision helpers.


def _compute_blade_params(
    vx: float,
    vy: float,
    vehicle: Zamboni552,
    base_fill_depth: float = 1.0,
    optimal_speed: float = 1.5,
) -> dict[str, float]:
    """Speed- and sideslip-dependent blade parameters."""
    speed = math.hypot(vx, vy)
    if speed < 1e-6:
        shave_multiplier = 2.0
    else:
        shave_multiplier = min(optimal_speed / speed, 2.0)

    beta = abs(math.atan2(vy, max(abs(vx), 1e-9)))
    cos_beta = abs(math.cos(beta))
    min_width_frac = 0.3
    effective_width = vehicle.brush_width * max(cos_beta, min_width_frac)
    effective_fill_depth = base_fill_depth * shave_multiplier

    return {
        "shave_multiplier": float(shave_multiplier),
        "effective_width": float(effective_width),
        "effective_fill_depth": float(effective_fill_depth),
        "speed": float(speed),
        "sideslip_angle": float(beta),
    }


def _shave_depth(
    old_state: VehicleState,
    new_state: VehicleState,
    dt: float,
    local_damage_mm: float,
) -> dict[str, float]:
    """Per-step shave depth, modulated by speed, heading alignment, and
    local roughness."""
    dx = new_state.x - old_state.x
    dy = new_state.y - old_state.y
    travel = float(math.hypot(dx, dy))
    speed_mps = travel / max(dt, 1e-9)

    if speed_mps < 0.05:
        return {"speed_mps": speed_mps, "alignment": 0.0, "shave_depth_mm": 0.0}

    nominal = 1.2
    sigma = 0.9
    speed_eff = math.exp(-((speed_mps - nominal) ** 2) / (2.0 * sigma * sigma))
    speed_eff = float(np.clip(0.15 + 0.85 * speed_eff, 0.0, 1.0))

    motion_dir = np.array([dx, dy]) / max(travel, 1e-9)
    angle_diff = (new_state.theta - old_state.theta + math.pi) % (2 * math.pi) - math.pi
    mid_theta = old_state.theta + 0.5 * angle_diff
    heading = np.array([math.cos(mid_theta), math.sin(mid_theta)])
    alignment = float(np.clip(abs(float(np.dot(motion_dir, heading))), 0.0, 1.0))
    angle_eff = 0.2 + 0.8 * alignment

    ice_factor = float(np.clip(1.1 - 0.08 * local_damage_mm, 0.55, 1.10))
    base_depth_mm = 1.8
    shave = max(0.0, base_depth_mm * speed_eff * angle_eff * ice_factor)
    return {
        "speed_mps": speed_mps,
        "alignment": alignment,
        "shave_depth_mm": float(shave),
    }


def _sample_box_corners(
    x: float, y: float, theta: float, half_l: float, half_w: float
) -> np.ndarray:
    """Return world-frame coordinates of the four vehicle-box corners."""
    c, s = math.cos(theta), math.sin(theta)
    local = np.array(
        [
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, half_w],
            [-half_l, -half_w],
            [half_l, 0.0],  # front-center
        ]
    )
    world = np.empty_like(local)
    world[:, 0] = x + local[:, 0] * c - local[:, 1] * s
    world[:, 1] = y + local[:, 0] * s + local[:, 1] * c
    return world


def _check_collisions(
    state: VehicleState,
    half_l: float,
    half_w: float,
    rink_mask: np.ndarray,
    pixels_per_meter: float,
) -> list[tuple[float, float]]:
    """Return any vehicle-box corners outside the rink mask."""
    pts = _sample_box_corners(state.x, state.y, state.theta, half_l, half_w)
    mask_h, mask_w = rink_mask.shape
    out = []
    for wx, wy in pts:
        px = int(math.floor(wx * pixels_per_meter))
        py = int(math.floor(wy * pixels_per_meter))
        if px < 0 or px >= mask_w or py < 0 or py >= mask_h or not rink_mask[py, px]:
            out.append((float(wx), float(wy)))
    return out


# Env


@dataclass
class _ZambGymCfg:
    """Internal config bundle. Captured here so tests / scripts can tweak
    without subclassing the env."""

    rink_width_m: float = 60.96
    rink_height_m: float = 25.9
    corner_radius_m: float = 8.53
    nav_pixels_per_meter: int = 5
    dt: float = 0.5
    max_steps: int = 2000
    damage_max_mm: float = 5.0


class ZambGymEnv(gym.Env):
    """Continuous-control ice-resurfacing environment.

    Image observation with three channels stacked at nav-grid resolution.
    Reward favours damage reduction and crossings of refreeze-ready cells;
    penalises crossings of not-yet-ready cells and applies a small step
    penalty.

    Reward coefficients are course-specific.
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    # Reward coefficients for the course environment.
    COEF_COVERAGE = 0.001              # per nav cell newly under brush
    COEF_DAMAGE_REDUCTION = 1.0        # per mm of footprint damage cleared
    COEF_REFREEZE_BONUS = 0.05         # for crossing a ready cell
    COEF_REDISTURB_PENALTY = -0.1      # for crossing a not-yet-ready cell
    COEF_STEP_PENALTY = -0.001

    def __init__(
        self,
        max_steps: int | None = 2000,
        ignore_collisions: bool = False,
        damage_depth_mm: float = 1.0,
        seed: int | None = None,
    ):
        super().__init__()

        cfg = _ZambGymCfg(max_steps=max_steps if max_steps is not None else 2000)
        self.cfg = cfg
        self.rink_width_m = cfg.rink_width_m
        self.rink_height_m = cfg.rink_height_m
        self.corner_radius_m = cfg.corner_radius_m
        self.nav_pixels_per_meter = cfg.nav_pixels_per_meter
        self.dt = cfg.dt
        self.max_steps = max_steps
        self.ignore_collisions = ignore_collisions

        self.vehicle = Zamboni552()
        self._dyn_params = VehicleDynamicsParams.from_vehicle(self.vehicle)
        self.dynamics_model = DynamicBicycleModel()

        self.nav_grid_size = (
            int(cfg.rink_height_m * cfg.nav_pixels_per_meter),
            int(cfg.rink_width_m * cfg.nav_pixels_per_meter),
        )
        self.rink_mask = create_rink_mask(
            self.nav_grid_size, cfg.corner_radius_m * cfg.nav_pixels_per_meter
        )

        self.damage_generator = UniformDamageGenerator(depth_mm=damage_depth_mm)
        self.start_pose_sampler = UniformRandomStartPoseSampler()

        self._vehicle_state = VehicleState()
        self.surface: IceSurfaceState | None = None
        self.swept_mask: np.ndarray | None = None      # was brush_visited_mask
        self.agent_visited: np.ndarray | None = None    # was visited_mask
        self.step_count = 0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        h, w = self.nav_grid_size
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(h, w, 3), dtype=np.float32
        )

        self._last_blade = {"speed_mps": 0.0, "alignment": 0.0, "shave_depth_mm": 0.0}

        if seed is not None:
            self.reset(seed=seed)

    # Gym API

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        rng = self.np_random

        damage = self.damage_generator.generate(
            gt_shape=self.nav_grid_size,
            rink_mask=self.rink_mask,
            rng=rng,
        )
        self.surface = IceSurfaceState(
            damage_mm=damage.astype(np.float32),
            refreeze_progress=np.ones(self.nav_grid_size, dtype=np.float32),
            rink_mask=self.rink_mask,
            damage_max_mm=float(self.cfg.damage_max_mm),
        )
        self.surface.validate()

        self.swept_mask = np.zeros(self.nav_grid_size, dtype=bool)
        self.agent_visited = np.zeros(self.nav_grid_size, dtype=bool)
        self.step_count = 0

        pose = self.start_pose_sampler.sample(
            nav_rink_mask=self.rink_mask,
            nav_pixels_per_meter=self.nav_pixels_per_meter,
            rng=rng,
        )
        self._vehicle_state = VehicleState(
            x=float(pose.x), y=float(pose.y), theta=float(pose.theta)
        )

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        t_start = time.perf_counter()

        old_state = self._vehicle_state.copy()

        vehicle_action = VehicleAction(
            throttle=float(action[0]),
            steering=float(action[1]),
        )
        self._vehicle_state = self.dynamics_model.step(
            self._vehicle_state,
            vehicle_action,
            self.dt,
            self.vehicle,
            self._dyn_params,
        )

        # Collision handling (simple corner check + velocity damping).
        half_l = self.vehicle.length / 2.0
        half_w = self.vehicle.width / 2.0
        collision_penalty = 0.0
        if not self.ignore_collisions:
            colliding = _check_collisions(
                self._vehicle_state,
                half_l,
                half_w,
                self.rink_mask,
                self.nav_pixels_per_meter,
            )
            if colliding:
                collision_penalty = self._apply_simple_collision_response(
                    old_state, colliding
                )

        # Brush footprint mask (nav grid) — used for damage / surface update.
        footprint = self._brush_footprint_mask() & self.rink_mask

        # Blade shave depth depends on speed & heading alignment & local damage.
        if footprint.any():
            local_damage = float(self.surface.damage_mm[footprint].mean())
        else:
            local_damage = 0.0
        blade_metrics = _shave_depth(old_state, self._vehicle_state, self.dt, local_damage)
        self._last_blade = blade_metrics
        shave_depth_mm = blade_metrics["shave_depth_mm"]

        # Newly-crossed cells this step (footprint cells not yet in swept mask).
        newly_crossed = footprint & ~self.swept_mask
        new_brush_cells = int(np.sum(newly_crossed))

        # Capture pre-step damage/refreeze on the *newly crossed* cells.
        before_damage = self.surface.damage_mm[footprint].copy() if footprint.any() else np.zeros(0)
        refreeze_on_new = self.surface.refreeze_progress[newly_crossed].copy() if newly_crossed.any() else np.zeros(0)

        self.surface = step_surface(
            self.surface, footprint_mask=footprint, shave_depth_mm=shave_depth_mm
        )

        self.swept_mask |= footprint

        # Agent center cell.
        nav_y = int(self._vehicle_state.y * self.nav_pixels_per_meter)
        nav_x = int(self._vehicle_state.x * self.nav_pixels_per_meter)
        if 0 <= nav_y < self.nav_grid_size[0] and 0 <= nav_x < self.nav_grid_size[1]:
            self.agent_visited[nav_y, nav_x] = True

        reward, components = self._compute_reward(
            damage_before=before_damage,
            damage_after=self.surface.damage_mm[footprint] if footprint.any() else np.zeros(0),
            refreeze_on_new=refreeze_on_new,
            new_brush_cells=new_brush_cells,
            collision_penalty=collision_penalty,
        )

        self.step_count += 1
        if self.max_steps is None:
            terminated = False
        else:
            terminated = bool(self.step_count >= self.max_steps)
        truncated = False

        info = {
            "vx": self._vehicle_state.vx,
            "vy": self._vehicle_state.vy,
            "omega": self._vehicle_state.omega,
            "blade_speed_mps": blade_metrics["speed_mps"],
            "blade_alignment": blade_metrics["alignment"],
            "blade_shave_depth_mm": blade_metrics["shave_depth_mm"],
            "reward_components": components,
            "step_ms": (time.perf_counter() - t_start) * 1000.0,
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def close(self):
        pass

    # Helpers

    def _brush_footprint_mask(self) -> np.ndarray:
        """Rasterise the rectangle (length × brush_width) centred at the
        current pose into the nav grid."""
        cx, cy, theta = self._vehicle_state.x, self._vehicle_state.y, self._vehicle_state.theta
        half_l = self.vehicle.length / 2.0
        half_w = self.vehicle.brush_width / 2.0
        ppm = float(self.nav_pixels_per_meter)
        rows, cols = self.nav_grid_size
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        half_diag = math.hypot(half_l, half_w)
        cx_px, cy_px = cx * ppm, cy * ppm
        col_lo = max(0, int(cx_px - half_diag * ppm) - 2)
        col_hi = min(cols, int(cx_px + half_diag * ppm) + 3)
        row_lo = max(0, int(cy_px - half_diag * ppm) - 2)
        row_hi = min(rows, int(cy_px + half_diag * ppm) + 3)

        mask = np.zeros((rows, cols), dtype=bool)
        if col_lo >= col_hi or row_lo >= row_hi:
            return mask

        col_idx = np.arange(col_lo, col_hi)
        row_idx = np.arange(row_lo, row_hi)
        col_grid, row_grid = np.meshgrid(col_idx, row_idx)
        dx = (col_grid + 0.5) / ppm - cx
        dy = (row_grid + 0.5) / ppm - cy
        local_lon = dx * cos_t + dy * sin_t
        local_lat = -dx * sin_t + dy * cos_t
        mask[row_lo:row_hi, col_lo:col_hi] = (
            (np.abs(local_lon) <= half_l) & (np.abs(local_lat) <= half_w)
        )
        return mask

    def _apply_simple_collision_response(
        self,
        old_state: VehicleState,
        colliding_points: list[tuple[float, float]],
    ) -> float:
        """Resolve a wall contact by moving the vehicle back along the
        contact-point displacement direction, damping velocity, and
        returning a small negative penalty."""
        rink_dims = RinkDimensions(
            width_m=self.rink_width_m,
            height_m=self.rink_height_m,
            corner_radius_m=self.corner_radius_m,
        )

        normals = []
        for pt in colliding_points:
            angle = get_wall_normal_angle(pt, rink_dims)
            if angle is None:
                continue
            normals.append(np.array([math.cos(angle), math.sin(angle)]))
        if not normals:
            dx = old_state.x - self._vehicle_state.x
            dy = old_state.y - self._vehicle_state.y
            mag = math.hypot(dx, dy)
            board_normal = (
                np.array([dx / mag, dy / mag]) if mag > 1e-8 else np.array([0.0, 1.0])
            )
        else:
            stacked = np.sum(normals, axis=0)
            mag = float(np.linalg.norm(stacked))
            board_normal = stacked / mag if mag > 1e-8 else normals[0]

        # Push back along the board normal.
        push = 0.05
        self._vehicle_state.x += float(board_normal[0]) * push
        self._vehicle_state.y += float(board_normal[1]) * push

        # Damp velocities and yaw.
        self._vehicle_state.vx *= 0.3
        self._vehicle_state.vy *= 0.3
        self._vehicle_state.omega *= 0.3

        # Per-corner penalty, capped.
        penalty = -0.05 * len(colliding_points)
        return float(np.clip(penalty, -1.0, 0.0))

    def _compute_reward(
        self,
        damage_before: np.ndarray,
        damage_after: np.ndarray,
        refreeze_on_new: np.ndarray,
        new_brush_cells: int,
        collision_penalty: float,
    ):
        """Per-step reward.

        Components:
          - ``damage``: total damage (mm) cleared under the brush footprint.
          - ``coverage``: nav cells newly under the brush this step.
          - ``refreeze``: bonus for newly-crossed cells that were *ready*
            (refreeze_progress >= threshold) before being disturbed.
          - ``redisturb``: penalty for newly-crossed cells that were *not*
            yet ready (refreeze_progress < threshold).
          - ``collision``: penalty from the contact response.
          - ``step``: small constant penalty per step.

        Only newly-crossed cells contribute the refreeze/redisturb terms —
        revisiting a cell does not re-award or re-penalise.
        """
        damage_cleared = float((damage_before - damage_after).sum()) if damage_before.size else 0.0
        if refreeze_on_new.size > 0:
            ready = refreeze_on_new >= REFREEZE_READY_THRESHOLD
            refreeze_bonus = float(ready.sum())
            redisturb = float((~ready).sum())
        else:
            refreeze_bonus = 0.0
            redisturb = 0.0

        components = {
            "damage": self.COEF_DAMAGE_REDUCTION * damage_cleared,
            "coverage": self.COEF_COVERAGE * new_brush_cells,
            "refreeze": self.COEF_REFREEZE_BONUS * refreeze_bonus,
            "redisturb": self.COEF_REDISTURB_PENALTY * redisturb,
            "collision": float(collision_penalty),
            "step": float(self.COEF_STEP_PENALTY),
        }
        return sum(components.values()), components

    def _get_obs(self) -> np.ndarray:
        damage_norm = np.clip(self.surface.damage_mm / self.surface.damage_max_mm, 0.0, 1.0)

        agent_pos = np.zeros(self.nav_grid_size, dtype=np.float32)
        nav_y = int(self._vehicle_state.y * self.nav_pixels_per_meter)
        nav_x = int(self._vehicle_state.x * self.nav_pixels_per_meter)
        if 0 <= nav_y < self.nav_grid_size[0] and 0 <= nav_x < self.nav_grid_size[1]:
            agent_pos[nav_y, nav_x] = 1.0

        refreeze = self.surface.refreeze_progress.astype(np.float32)

        return np.stack([damage_norm.astype(np.float32), agent_pos, refreeze], axis=-1)

    # Backward-compatible state accessors

    @property
    def x(self) -> float:
        return self._vehicle_state.x

    @property
    def y(self) -> float:
        return self._vehicle_state.y

    @property
    def theta(self) -> float:
        return self._vehicle_state.theta

    @property
    def vehicle_state(self) -> VehicleState:
        return self._vehicle_state


__all__ = ["ZambGymEnv"]
