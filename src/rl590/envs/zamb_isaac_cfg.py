"""Config dataclasses for ZambIsaacEnv.

Targets Isaac Lab 2.x / Isaac Sim 4.5+ (package path `isaaclab.*`, not the
older `omni.isaac.lab.*`). Imports resolve inside the Isaac Lab container
only.
"""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.schemas import (
    CollisionPropertiesCfg,
    MassPropertiesCfg,
    RigidBodyPropertiesCfg,
)
from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab.utils import configclass


@configclass
class ZambSceneCfg(InteractiveSceneCfg):
    """Scene: rink (static) + Zamboni (rigid body)."""

    num_envs: int = 8
    env_spacing: float = 80.0  # rink is 60×30, leaves a margin between envs

    zamboni: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Zamboni",
        spawn=UsdFileCfg(
            usd_path=MISSING,
            rigid_props=RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_linear_velocity=5.0,
                max_angular_velocity=2.0,
                linear_damping=0.5,
                angular_damping=0.5,
            ),
            mass_props=MassPropertiesCfg(mass=4500.0),
            collision_props=CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-25.0, 0.0, 1.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


@configclass
class ZambIsaacEnvCfg(DirectRLEnvCfg):
    """Top-level env config. Set asset paths via the constructor or by
    overriding `zamboni_usd` / `rink_usd` after instantiation."""

    decimation: int = 4
    episode_length_s: float = 20.0
    action_space: int = 2
    observation_space: int = MISSING
    state_space: int = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=4,
    )
    scene: ZambSceneCfg = ZambSceneCfg(num_envs=8, env_spacing=80.0)

    zamboni_usd: str = MISSING
    rink_usd: str = MISSING

    max_lin_vel: float = 2.0
    max_ang_vel: float = 0.6

    truth_cell_size_m: float = 0.01
    nav_cell_size_m: float = 0.20
    rink_extent_m: tuple[float, float] = (60.0, 30.0)
    obs_patch_cells: int = 32
    obs_channels: int = 2
    damage_device: str = "cpu"

    reward_damage_decrement_coef: float = 1.0
    reward_step_penalty: float = -0.001
    reward_oob_penalty: float = -1.0

    def __post_init__(self):
        h = self.obs_patch_cells
        w = self.obs_patch_cells
        c = self.obs_channels
        self.observation_space = h * w * c

        if self.zamboni_usd is not MISSING:
            self.scene.zamboni.spawn.usd_path = self.zamboni_usd