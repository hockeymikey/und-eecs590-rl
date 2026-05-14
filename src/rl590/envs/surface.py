"""Ice-surface state, damage generators, and refreeze dynamics.

The course env keeps only the two surface layers needed for the V3
experiments: ``damage`` and ``refreeze_progress``. The refreeze update is a
deliberately simple linear model that preserves the surface-readiness framing
without bringing in calibrated heat-transfer details.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import math

import numpy as np


# Refreeze constants

REFREEZE_RATE = 0.01            # progress increment per env step
REFREEZE_READY_THRESHOLD = 0.8   # cell is "ready" when progress >= this


# Rink geometry / scenario spec


@dataclass(frozen=True)
class RinkDimensions:
    """Top-down rink dimensions in metres."""

    width_m: float
    height_m: float
    corner_radius_m: float


STANDARD_NHL_RINK = RinkDimensions(
    width_m=60.96,
    height_m=25.9,
    corner_radius_m=8.53,
)


@dataclass(frozen=True)
class GridSpec:
    """Grid resolution for truth- and nav-grid layers."""

    truth_pixels_per_meter: float
    planning_pixels_per_meter: float | None = None

    @property
    def truth_cell_size_m(self) -> float:
        return 1.0 / self.truth_pixels_per_meter

    @property
    def planning_cell_size_m(self) -> float | None:
        if self.planning_pixels_per_meter is None:
            return None
        return 1.0 / self.planning_pixels_per_meter


@dataclass(frozen=True)
class ScenarioSpec:
    """Minimal scenario descriptor."""

    name: str
    rink: RinkDimensions = STANDARD_NHL_RINK
    vehicle_name: str = "zamboni_552"
    grid: GridSpec | None = None
    description: str | None = None


# Rounded-corner rink mask


def create_rink_mask(
    grid_shape: tuple[int, int],
    corner_radius_px: float,
) -> np.ndarray:
    """Boolean mask of the rounded-corner rink footprint.

    ``grid_shape`` is ``(height, width)`` in pixels; ``corner_radius_px`` is
    the corner radius in the same pixel units.
    """
    height, width = grid_shape
    y, x = np.ogrid[:height, :width]
    mask = np.ones(grid_shape, dtype=bool)

    centers = [
        (corner_radius_px, corner_radius_px),
        (width - corner_radius_px, corner_radius_px),
        (corner_radius_px, height - corner_radius_px),
        (width - corner_radius_px, height - corner_radius_px),
    ]
    quadrants = [
        (x < centers[0][0]) & (y < centers[0][1]),
        (x >= centers[1][0]) & (y < centers[1][1]),
        (x < centers[2][0]) & (y >= centers[2][1]),
        (x >= centers[3][0]) & (y >= centers[3][1]),
    ]
    for (cx, cy), in_quadrant in zip(centers, quadrants):
        outside_arc = (x - cx) ** 2 + (y - cy) ** 2 > corner_radius_px ** 2
        mask[in_quadrant & outside_arc] = False
    return mask


def get_wall_normal_angle(
    point_m: tuple[float, float],
    rink: RinkDimensions,
) -> float | None:
    """Return the inward wall-normal angle (radians) at the given rink
    boundary point, or ``None`` if the point is not near a wall."""
    x, y = point_m
    w, h, r = rink.width_m, rink.height_m, rink.corner_radius_m

    if y < r and x < r:
        return math.atan2(r - y, r - x)
    if y < r and x > w - r:
        return math.atan2(r - y, (w - r) - x)
    if y > h - r and x < r:
        return math.atan2((h - r) - y, r - x)
    if y > h - r and (x > w - r or x >= w):
        return math.atan2((h - r) - y, (w - r) - x)

    if y < r:
        return math.pi / 2
    if y > h - r:
        return 3 * math.pi / 2
    if x < r:
        return 0.0
    if x > w - r:
        return math.pi
    return None


# Runtime surface state


@dataclass
class IceSurfaceState:
    """Mutable runtime surface state on the nav grid.

    Two scalar layers per cell:
      - ``damage_mm``: magnitude of accumulated rut / cut depth, non-negative
      - ``refreeze_progress``: in ``[0, 1]``; 0 = freshly disturbed, 1 = ready

    Plus the boolean nav-grid ``rink_mask`` and a normalisation constant
    ``damage_max_mm`` used to scale damage into the observation.
    """

    damage_mm: np.ndarray
    refreeze_progress: np.ndarray
    rink_mask: np.ndarray
    damage_max_mm: float

    def validate(self) -> None:
        shape = self.rink_mask.shape
        for name, arr in (
            ("damage_mm", self.damage_mm),
            ("refreeze_progress", self.refreeze_progress),
        ):
            if arr.shape != shape:
                raise ValueError(f"{name} shape {arr.shape} != mask shape {shape}")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} has non-finite values")
        if np.any(self.damage_mm < 0):
            raise ValueError("damage_mm must be non-negative")
        if np.any(self.refreeze_progress < 0.0) or np.any(self.refreeze_progress > 1.0):
            raise ValueError("refreeze_progress must be in [0, 1]")

    def ready_mask(self) -> np.ndarray:
        """Boolean mask of cells whose refreeze_progress is at/above the
        ready threshold (and inside the rink)."""
        return self.rink_mask & (self.refreeze_progress >= REFREEZE_READY_THRESHOLD)


# Damage generators


class DamageGenerator(ABC):
    """Interface for generating an initial damage layer."""

    @abstractmethod
    def generate(
        self,
        *,
        gt_shape: tuple[int, int],
        rink_mask: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return a damage-magnitude array (non-negative, mm)."""
        raise NotImplementedError


@dataclass(frozen=True)
class UniformDamageGenerator(DamageGenerator):
    """Uniform damage layer inside the rink mask."""

    depth_mm: float = 1.0

    def generate(
        self,
        *,
        gt_shape: tuple[int, int],
        rink_mask: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        del rng
        cut = np.zeros(gt_shape, dtype=np.float32)
        cut[rink_mask] = float(self.depth_mm)
        return cut


@dataclass(frozen=True)
class SkateCutsDamageGenerator(DamageGenerator):
    """Random-walk skate-cut damage generator. Magnitudes in mm."""

    num_skaters: int = 10
    num_steps: int = 100
    cut_depth_range_mm: tuple[float, float] = (0.5, 3.0)
    velocity_noise: float = 0.5
    velocity_clip: float = 5.0

    def generate(
        self,
        *,
        gt_shape: tuple[int, int],
        rink_mask: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        cut = np.zeros(gt_shape, dtype=np.float32)
        valid_starts = np.argwhere(rink_mask)
        if len(valid_starts) == 0:
            return cut

        depth_lo, depth_hi = self.cut_depth_range_mm

        for _ in range(int(self.num_skaters)):
            start_idx = int(rng.integers(0, len(valid_starts)))
            pos = valid_starts[start_idx].astype(np.float32)
            velocity = rng.standard_normal(2)
            for _ in range(int(self.num_steps)):
                velocity += rng.standard_normal(2) * self.velocity_noise
                velocity = np.clip(velocity, -self.velocity_clip, self.velocity_clip)
                pos += velocity
                px, py = int(pos[0]), int(pos[1])
                if (
                    0 <= px < gt_shape[0]
                    and 0 <= py < gt_shape[1]
                    and bool(rink_mask[px, py])
                ):
                    depth = float(rng.uniform(depth_lo, depth_hi))
                    cut[px, py] = max(cut[px, py], depth)

        cut[~rink_mask] = 0.0
        return cut


# Start-pose samplers


@dataclass(frozen=True)
class StartPose:
    x: float
    y: float
    theta: float


class StartPoseSampler(ABC):
    @abstractmethod
    def sample(
        self,
        *,
        nav_rink_mask: np.ndarray,
        nav_pixels_per_meter: float,
        rng: np.random.Generator,
    ) -> StartPose:
        raise NotImplementedError


@dataclass(frozen=True)
class UniformRandomStartPoseSampler(StartPoseSampler):
    """Sample a start pose uniformly over the nav-grid rink-mask cells."""

    def sample(
        self,
        *,
        nav_rink_mask: np.ndarray,
        nav_pixels_per_meter: float,
        rng: np.random.Generator,
    ) -> StartPose:
        valid_starts = np.argwhere(nav_rink_mask)
        if len(valid_starts) == 0:
            raise ValueError("nav_rink_mask has no valid start cells")
        idx = int(rng.integers(0, len(valid_starts)))
        sy, sx = valid_starts[idx]
        x = (float(sx) + 0.5) / float(nav_pixels_per_meter)
        y = (float(sy) + 0.5) / float(nav_pixels_per_meter)
        theta = float(rng.uniform(0.0, 2.0 * math.pi))
        return StartPose(x=x, y=y, theta=theta)


# Surface update


def step_surface(
    state: IceSurfaceState,
    *,
    footprint_mask: np.ndarray,
    shave_depth_mm: float,
) -> IceSurfaceState:
    """Advance the surface by one env step.

    Update rules (intentionally simple):

      - For cells *outside* the brush footprint: refreeze_progress increases
        by ``REFREEZE_RATE`` per step until clamped at 1.0. Damage is
        unchanged.
      - For cells *inside* the footprint: damage is reduced by
        ``shave_depth_mm`` (clamped at 0), and refreeze_progress is reset
        to 0 — the resurfacer has just disturbed the surface, so the
        refreeze timer starts over.
    """
    new_damage = state.damage_mm.copy()
    new_refreeze = state.refreeze_progress.copy()

    # Passive refreeze everywhere inside the rink.
    inside = state.rink_mask
    new_refreeze[inside] = np.minimum(new_refreeze[inside] + REFREEZE_RATE, 1.0)

    # Footprint: reduce damage and reset the refreeze counter on disturbance.
    fp = footprint_mask & state.rink_mask
    new_damage[fp] = np.clip(new_damage[fp] - shave_depth_mm, 0.0, None)
    new_refreeze[fp] = 0.0

    return IceSurfaceState(
        damage_mm=new_damage.astype(np.float32),
        refreeze_progress=new_refreeze.astype(np.float32),
        rink_mask=state.rink_mask,
        damage_max_mm=state.damage_max_mm,
    )


__all__ = [
    "REFREEZE_RATE",
    "REFREEZE_READY_THRESHOLD",
    "RinkDimensions",
    "STANDARD_NHL_RINK",
    "GridSpec",
    "ScenarioSpec",
    "create_rink_mask",
    "get_wall_normal_angle",
    "IceSurfaceState",
    "DamageGenerator",
    "UniformDamageGenerator",
    "SkateCutsDamageGenerator",
    "StartPose",
    "StartPoseSampler",
    "UniformRandomStartPoseSampler",
    "step_surface",
]
