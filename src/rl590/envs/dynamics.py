"""Vehicle, action, and dynamics model for the Zamboni gym env.

Self-contained course variant with the resurfacer vehicle spec, force-based
bicycle dynamics, and derived parameters needed by ``ZambGymEnv``. Only
``DynamicBicycleModel`` is exposed here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# Vehicle geometry / spec


class Zamboni552:
    """Design-oriented model of the Zamboni 552 AC / electric platform.

    The raw dimensions and manufacturer-known values are the source of
    truth; the helper properties derive the planner/dynamics quantities the
    rest of the env expects.
    """

    model_name: str = "zamboni_552ac"

    # Raw dimensions (meters) and operating weights (kg).
    length: float = 4.04
    height: float = 2.26
    wheelbase: float = 1.96
    wheel_width: float = 1.37
    turning_radius_conditioner: float = 4.87
    blade_width: float = 2.13
    blade_length: float = 0.127
    brush_width: float = 2.13
    weight_full_water: float = 4037.0
    weight_empty_water: float = 3311.0
    traction_motor_hp: float = 24.0
    front_overhang_m: float = 0.50

    max_speed: float = 3.0           # m/s
    max_reverse_speed: float = 1.5   # m/s
    steering_rate_limit: float = math.radians(45.0)

    @property
    def weight_kg(self) -> float:
        return self.weight_full_water

    @property
    def width(self) -> float:
        return 0.5 * (self.brush_width + self.wheel_width)

    @property
    def front_overhang(self) -> float:
        return self.front_overhang_m

    @property
    def rear_overhang(self) -> float:
        return self.length - self.wheelbase - self.front_overhang_m

    @property
    def traction_motor_kw(self) -> float:
        return 0.745699872 * self.traction_motor_hp

    @property
    def turn_radius(self) -> float:
        """Reference (rear-axle) minimum turn radius, derived from the
        conditioner's outer turn radius."""
        outer = self.turning_radius_conditioner
        swept = self.brush_width
        longitudinal = abs(self.rear_overhang)
        lateral = math.sqrt(outer * outer - longitudinal * longitudinal)
        return lateral - 0.5 * swept

    @property
    def max_steering_angle(self) -> float:
        return math.atan(self.wheelbase / self.turn_radius)


# Derived dynamics parameters


@dataclass(frozen=True)
class VehicleDynamicsParams:
    """Derived params for the bicycle dynamics."""

    surface_friction: float
    acceleration_limit: float
    deceleration_limit: float
    cornering_stiffness_front: float
    cornering_stiffness_rear: float
    moment_of_inertia: float
    lf: float
    lr: float
    cg_from_rear_axle: float

    @classmethod
    def from_vehicle(
        cls,
        vehicle: Zamboni552,
        *,
        reference_surface_friction: float = 0.05,
        cornering_saturation_angle: float = math.radians(5.0),
    ) -> "VehicleDynamicsParams":
        cg = 0.5 * (vehicle.wheelbase + vehicle.front_overhang - vehicle.rear_overhang)
        lr = cg
        lf = vehicle.wheelbase - cg
        rear_load_frac = lf / vehicle.wheelbase

        traction = reference_surface_friction * 9.81 * rear_load_frac
        kw = vehicle.traction_motor_kw
        power = (kw * 1000.0) / (vehicle.weight_kg * vehicle.max_speed)
        accel = min(traction, power)
        decel = reference_surface_friction * 9.81

        front_normal = vehicle.weight_kg * 9.81 * lr / vehicle.wheelbase
        rear_normal = vehicle.weight_kg * 9.81 * lf / vehicle.wheelbase
        c_front = front_normal / cornering_saturation_angle
        c_rear = rear_normal / cornering_saturation_angle

        moi = vehicle.weight_kg * (vehicle.length**2 + vehicle.width**2) / 12.0

        return cls(
            surface_friction=float(reference_surface_friction),
            acceleration_limit=float(accel),
            deceleration_limit=float(decel),
            cornering_stiffness_front=float(c_front),
            cornering_stiffness_rear=float(c_rear),
            moment_of_inertia=float(moi),
            lf=float(lf),
            lr=float(lr),
            cg_from_rear_axle=float(cg),
        )


# Vehicle state and action


@dataclass
class VehicleState:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    omega: float = 0.0
    steering_angle: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array(
            [self.x, self.y, self.theta, self.vx, self.vy, self.omega],
            dtype=np.float64,
        )

    @classmethod
    def from_array(cls, arr: np.ndarray, steering_angle: float = 0.0) -> "VehicleState":
        return cls(
            x=float(arr[0]),
            y=float(arr[1]),
            theta=float(arr[2]),
            vx=float(arr[3]),
            vy=float(arr[4]),
            omega=float(arr[5]),
            steering_angle=float(steering_angle),
        )

    def copy(self) -> "VehicleState":
        return VehicleState(
            x=self.x,
            y=self.y,
            theta=self.theta,
            vx=self.vx,
            vy=self.vy,
            omega=self.omega,
            steering_angle=self.steering_angle,
        )


@dataclass
class VehicleAction:
    throttle: float = 0.0
    steering: float = 0.0


# Bicycle dynamics


class DynamicBicycleModel:
    """Force-based dynamic bicycle model with linear tire forces.

    Captures lateral slip and friction-limited behavior on a low-friction
    surface (ice). RK4 integration; clamps speed and normalizes heading.
    """

    def step(
        self,
        state: VehicleState,
        action: VehicleAction,
        dt: float,
        vehicle: Zamboni552,
        params: VehicleDynamicsParams,
    ) -> VehicleState:
        desired_steering = action.steering * vehicle.max_steering_angle
        new_steering = self._apply_steering_rate_limit(
            state.steering_angle, desired_steering, dt, vehicle
        )

        if action.throttle >= 0:
            drive_force = action.throttle * params.acceleration_limit * vehicle.weight_kg
        else:
            drive_force = action.throttle * params.deceleration_limit * vehicle.weight_kg

        mass = vehicle.weight_kg
        inertia_z = params.moment_of_inertia
        lf = params.lf
        lr = params.lr
        cornering_front = params.cornering_stiffness_front
        cornering_rear = params.cornering_stiffness_rear
        mu = params.surface_friction
        gravity = 9.81

        fz_front = mass * gravity * lr / (lf + lr)
        fz_rear = mass * gravity * lf / (lf + lr)

        max_fy_front = mu * fz_front
        max_fy_rear = mu * fz_rear
        max_fx_rear = mu * fz_rear
        drive_force = float(np.clip(drive_force, -max_fx_rear, max_fx_rear))

        delta = new_steering
        s = state.to_array()

        def derivs(st: np.ndarray) -> np.ndarray:
            _, _, theta, vx, vy, omega = st
            speed = float(np.sqrt(vx * vx + vy * vy))

            if speed < 0.5:
                ax = drive_force / mass
                if abs(vx) > 0.01:
                    omega_approx = vx * np.tan(delta) / (lf + lr)
                else:
                    omega_approx = 0.0
                return np.array(
                    [
                        vx * np.cos(theta) - vy * np.sin(theta),
                        vx * np.sin(theta) + vy * np.cos(theta),
                        omega_approx,
                        ax,
                        -vy * 2.0,
                        (omega_approx - omega) * 2.0,
                    ]
                )

            alpha_f = delta - np.arctan2(vy + lf * omega, abs(vx))
            alpha_r = -np.arctan2(vy - lr * omega, abs(vx))

            fy_front = np.clip(cornering_front * alpha_f, -max_fy_front, max_fy_front)
            fy_rear = np.clip(cornering_rear * alpha_r, -max_fy_rear, max_fy_rear)

            rear_force_mag = float(np.sqrt(drive_force * drive_force + fy_rear * fy_rear))
            if rear_force_mag > max_fy_rear:
                scale = max_fy_rear / rear_force_mag
                fy_rear *= scale

            ax = (drive_force - fy_front * np.sin(delta)) / mass + vy * omega
            ay = (fy_rear + fy_front * np.cos(delta)) / mass - vx * omega
            omega_dot = (lf * fy_front * np.cos(delta) - lr * fy_rear) / inertia_z

            return np.array(
                [
                    vx * np.cos(theta) - vy * np.sin(theta),
                    vx * np.sin(theta) + vy * np.cos(theta),
                    omega,
                    ax,
                    ay,
                    omega_dot,
                ]
            )

        k1 = derivs(s)
        k2 = derivs(s + 0.5 * dt * k1)
        k3 = derivs(s + 0.5 * dt * k2)
        k4 = derivs(s + dt * k3)

        new_s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        new_s[3] = self._clamp_speed(new_s[3], vehicle)
        new_s[2] = self._normalize_angle(new_s[2])

        return VehicleState.from_array(new_s, steering_angle=new_steering)

    @staticmethod
    def _apply_steering_rate_limit(
        current: float, desired: float, dt: float, vehicle: Zamboni552
    ) -> float:
        max_change = vehicle.steering_rate_limit * dt
        delta = float(np.clip(desired - current, -max_change, max_change))
        return float(
            np.clip(
                current + delta,
                -vehicle.max_steering_angle,
                vehicle.max_steering_angle,
            )
        )

    @staticmethod
    def _clamp_speed(vx: float, vehicle: Zamboni552) -> float:
        return float(np.clip(vx, -vehicle.max_reverse_speed, vehicle.max_speed))

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        return float((angle + np.pi) % (2 * np.pi) - np.pi)


__all__ = [
    "Zamboni552",
    "VehicleDynamicsParams",
    "VehicleState",
    "VehicleAction",
    "DynamicBicycleModel",
]
