"""ZambIsaacEnv — single-Zamboni resurfacing toy on Isaac Lab.

Targets Isaac Lab 2.x / Isaac Sim 4.5+. Runs only inside the Isaac Lab
container.

Design:
  - Kinematic-style root velocity drive (no articulated wheels). Action is
    (forward_throttle, yaw_rate) in [-1, 1]; scaled to (max_lin_vel,
    max_ang_vel) before being written to PhysX as a body-frame velocity.
  - Truth damage tensor at 1 cm cells. Observation is block-pooled to nav
    resolution (20 cm cells) and cropped to a local
    obs_patch_cells × obs_patch_cells window around the chassis.
  - Reward = damage decrement under chassis footprint this step, with a
    small step penalty and a one-shot OOB penalty. Deliberately simple.
"""

from __future__ import annotations

import torch

from isaaclab.assets import RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import UsdFileCfg, spawn_from_usd
from isaaclab.utils.math import euler_xyz_from_quat

from rl590.envs.zamb_isaac_cfg import ZambIsaacEnvCfg


class ZambIsaacEnv(DirectRLEnv):
    cfg: ZambIsaacEnvCfg

    def __init__(self, cfg: ZambIsaacEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        rink_x, rink_y = cfg.rink_extent_m
        self.truth_h = int(round(rink_y / cfg.truth_cell_size_m))
        self.truth_w = int(round(rink_x / cfg.truth_cell_size_m))
        self.nav_h = int(round(rink_y / cfg.nav_cell_size_m))
        self.nav_w = int(round(rink_x / cfg.nav_cell_size_m))

        damage_device = torch.device(cfg.damage_device)
        self.damage = torch.ones(
            self.num_envs, self.truth_h, self.truth_w,
            dtype=torch.float32, device=damage_device,
        )

        self._actions = torch.zeros(self.num_envs, cfg.action_space, device=self.device)

    def _setup_scene(self):
        spawn_from_usd("/World/Rink", UsdFileCfg(usd_path=self.cfg.rink_usd))

        self.zamb = RigidObject(self.cfg.scene.zamboni)
        self.scene.rigid_objects["zamboni"] = self.zamb

        from isaaclab.sim.spawners.lights import DomeLightCfg
        DomeLightCfg(intensity=2000.0).func("/World/Light", DomeLightCfg(intensity=2000.0))

        self.scene.clone_environments(copy_from_source=False)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clamp(-1.0, 1.0)

    def _apply_action(self):
        v = self._actions[:, 0] * self.cfg.max_lin_vel
        omega = self._actions[:, 1] * self.cfg.max_ang_vel

        quat_w = self.zamb.data.root_quat_w
        _, _, yaw = euler_xyz_from_quat(quat_w)
        vx = v * torch.cos(yaw)
        vy = v * torch.sin(yaw)
        zeros = torch.zeros_like(v)

        velocities = torch.stack([vx, vy, zeros, zeros, zeros, omega], dim=-1)
        self.zamb.write_root_velocity_to_sim(velocities)

    def _get_observations(self) -> dict:
        cfg = self.cfg
        N = self.num_envs

        pos = self.zamb.data.root_pos_w[:, :2]
        pos = pos - self.scene.env_origins[:, :2]

        block_h = self.truth_h // self.nav_h
        block_w = self.truth_w // self.nav_w
        damage_dev = self.damage.to(self.device, non_blocking=True)
        nav = damage_dev.view(N, self.nav_h, block_h, self.nav_w, block_w).mean(dim=(2, 4))

        patch = self._extract_local_patch(nav, pos)

        pose_mask = torch.zeros_like(patch)
        center = cfg.obs_patch_cells // 2
        pose_mask[:, center, center] = 1.0

        obs = torch.stack([patch, pose_mask], dim=-1)
        return {"policy": obs.flatten(start_dim=1)}

    def _extract_local_patch(self, nav: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        P = cfg.obs_patch_cells
        N, H, W = nav.shape

        cx = ((pos[:, 0] / cfg.nav_cell_size_m) + W / 2).long()
        cy = ((pos[:, 1] / cfg.nav_cell_size_m) + H / 2).long()

        half = P // 2
        cx = cx.clamp(half, W - half)
        cy = cy.clamp(half, H - half)

        patches = torch.empty(N, P, P, dtype=nav.dtype, device=nav.device)
        for i in range(N):
            patches[i] = nav[i, cy[i] - half : cy[i] + half, cx[i] - half : cx[i] + half]
        return patches

    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg
        N = self.num_envs

        pos = self.zamb.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
        fp_x_cells = int(round(5.0 / cfg.truth_cell_size_m))
        fp_y_cells = int(round(2.5 / cfg.truth_cell_size_m))
        cx = ((pos[:, 0] / cfg.truth_cell_size_m) + self.truth_w / 2).long()
        cy = ((pos[:, 1] / cfg.truth_cell_size_m) + self.truth_h / 2).long()
        cx = cx.clamp(fp_x_cells // 2, self.truth_w - fp_x_cells // 2)
        cy = cy.clamp(fp_y_cells // 2, self.truth_h - fp_y_cells // 2)

        rewards = torch.zeros(N, device=self.device)
        for i in range(N):
            sl_y = slice(cy[i].item() - fp_y_cells // 2, cy[i].item() + fp_y_cells // 2)
            sl_x = slice(cx[i].item() - fp_x_cells // 2, cx[i].item() + fp_x_cells // 2)
            patch = self.damage[i, sl_y, sl_x]
            decrement = patch.clone()
            self.damage[i, sl_y, sl_x] = 0.0
            rewards[i] = decrement.sum() * cfg.reward_damage_decrement_coef

        rewards = rewards + cfg.reward_step_penalty
        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        truncated = self.episode_length_buf >= self.max_episode_length

        pos = self.zamb.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
        half_x = cfg.rink_extent_m[0] / 2.0
        half_y = cfg.rink_extent_m[1] / 2.0
        terminated = (pos[:, 0].abs() > half_x) | (pos[:, 1].abs() > half_y)
        return terminated, truncated

    def _reset_idx(self, env_ids):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.zamb._ALL_INDICES
        super()._reset_idx(env_ids)

        self.damage[env_ids] = 1.0

        default_root_state = self.zamb.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self.zamb.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.zamb.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)