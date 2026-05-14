"""Coverage-path waypoint follower.

Loads a precomputed `coverage_path_v1.npz` blob and produces (throttle,
steering) actions that drive the env's vehicle toward the next waypoint
ahead. The path-generation algorithm itself lives outside this repo —
only the data (waypoints) is consumed here.

Expected npz schema (produced by the offline path exporter):

    waypoints  : (N, 3) float32 — x, y, yaw at each step (metres, radians)
    scenario_name, seed, rink_w, rink_h, corner_r, brush_width,
    turn_radius, n_perimeter_laps, resolution, clockwise,
    cw_half_band_mode, export_timestamp, export_format_version

Only ``waypoints`` is required at runtime; the rest is metadata.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CoveragePathTeacher:
    """Nearest-waypoint-ahead pure-pursuit-ish controller.

    Loads the path on construction and exposes ``act(state)`` which returns
    an ``(action, info)`` tuple. ``state`` is anything that provides ``x``,
    ``y``, ``theta`` attributes (e.g. a ``VehicleState`` or the env itself).
    """

    waypoints: np.ndarray              # (N, 3): x, y, yaw
    lookahead_m: float = 1.5           # how far ahead to look on the path
    throttle_scale: float = 0.6        # forward throttle when on path
    arrival_radius_m: float = 0.6      # waypoint considered "passed" within this
    max_steering_deg: float = 30.0     # cap on steering output magnitude

    _idx: int = 0                       # current waypoint pointer
    _anchored: bool = False              # whether the pointer has been anchored to the agent

    @classmethod
    def from_npz(cls, path: str | Path, **overrides) -> "CoveragePathTeacher":
        data = np.load(path, allow_pickle=False)
        if "waypoints" not in data.files:
            raise ValueError(f"{path}: missing required 'waypoints' array")
        wpts = np.asarray(data["waypoints"], dtype=np.float32)
        if wpts.ndim != 2 or wpts.shape[1] < 2:
            raise ValueError(f"{path}: waypoints must be (N, 2) or (N, 3)")
        if wpts.shape[1] == 2:
            yaw = np.zeros((wpts.shape[0], 1), dtype=np.float32)
            wpts = np.concatenate([wpts, yaw], axis=1)
        return cls(waypoints=wpts, **overrides)

    def reset(self) -> None:
        self._idx = 0
        self._anchored = False

    def _advance_to_nearest(self, x: float, y: float) -> None:
        """Snap the path pointer forward to the closest waypoint.

        On the first call, search the entire path so the agent picks up at
        the globally nearest waypoint (handles env start poses far from
        the path's first point). Subsequent calls only search a short
        window forward since the path is roughly monotonic.
        """
        if self._idx >= len(self.waypoints):
            return
        if not self._anchored:
            d2 = (self.waypoints[:, 0] - x) ** 2 + (self.waypoints[:, 1] - y) ** 2
            self._idx = int(np.argmin(d2))
            self._anchored = True
            return
        window = min(len(self.waypoints) - self._idx, 50)
        seg = self.waypoints[self._idx : self._idx + window]
        d2 = (seg[:, 0] - x) ** 2 + (seg[:, 1] - y) ** 2
        nearest = int(np.argmin(d2))
        if math.sqrt(float(d2[nearest])) < self.arrival_radius_m:
            self._idx += nearest + 1
        else:
            self._idx += nearest

    def _lookahead_waypoint(self, x: float, y: float) -> np.ndarray:
        """Return a waypoint roughly ``lookahead_m`` metres ahead of (x, y)."""
        if self._idx >= len(self.waypoints):
            return self.waypoints[-1]
        accum = 0.0
        prev = np.array([x, y], dtype=np.float32)
        for i in range(self._idx, len(self.waypoints)):
            curr = self.waypoints[i, :2]
            accum += float(np.linalg.norm(curr - prev))
            prev = curr
            if accum >= self.lookahead_m:
                return self.waypoints[i]
        return self.waypoints[-1]

    def act(self, state) -> tuple[np.ndarray, dict]:
        """Return a ``(throttle, steering)`` action toward the next waypoint."""
        x = float(state.x)
        y = float(state.y)
        theta = float(state.theta)

        self._advance_to_nearest(x, y)
        target = self._lookahead_waypoint(x, y)
        tx, ty = float(target[0]), float(target[1])

        # Steering = signed angle to the target.
        dx, dy = tx - x, ty - y
        target_heading = math.atan2(dy, dx)
        steering_rad = (target_heading - theta + math.pi) % (2 * math.pi) - math.pi
        max_steer = math.radians(self.max_steering_deg)
        steering_norm = float(np.clip(steering_rad / max_steer, -1.0, 1.0))

        # Throttle = scaled forward; ease off when the heading error is large.
        heading_err = abs(steering_rad)
        throttle = self.throttle_scale * max(0.1, math.cos(min(heading_err, math.pi / 2)))
        action = np.array([throttle, steering_norm], dtype=np.float32)

        info = {
            "waypoint_idx": int(self._idx),
            "target_xy": (tx, ty),
            "heading_error_rad": float(steering_rad),
            "remaining_waypoints": int(max(0, len(self.waypoints) - self._idx)),
        }
        return action, info

    @property
    def finished(self) -> bool:
        return self._idx >= len(self.waypoints)


__all__ = ["CoveragePathTeacher"]
