"""Train ZambIsaacEnv with the from-scratch PPO from rl590.deep.ppo.

Single-env Gymnasium API path. Demonstrates that the from-scratch PPO
implementation in this repo also runs on the Isaac Sim env via the
single-env wrapper.

Run inside the Isaac Lab container:

  python scripts/demo_zamb_isaac_ppo.py \
      --zamboni-usd /workspace/Minis/assets/zamboni.usd \
      --rink-usd /workspace/Minis/assets/ice.usd \
      --total-timesteps 50000
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zamboni-usd", required=True)
    parser.add_argument("--rink-usd", required=True)
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/minis_ppo_zamb"))
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args()

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app

    from rl590.deep.ppo import PPOAgent, PPOConfig

    from rl590.envs.wrappers import IsaacSingleEnvWrapper
    from rl590.envs.zamb_isaac import ZambIsaacEnv
    from rl590.envs.zamb_isaac_cfg import ZambIsaacEnvCfg

    cfg = ZambIsaacEnvCfg(
        zamboni_usd=args.zamboni_usd,
        rink_usd=args.rink_usd,
    )
    cfg.scene.num_envs = 1
    cfg.__post_init__()

    isaac_env = ZambIsaacEnv(cfg)
    wrapped = IsaacSingleEnvWrapper(
        isaac_env,
        obs_hwc=(cfg.obs_patch_cells, cfg.obs_patch_cells, cfg.obs_channels),
    )

    ppo_cfg = PPOConfig(
        obs_height=cfg.obs_patch_cells,
        obs_width=cfg.obs_patch_cells,
        n_channels=cfg.obs_channels,
        action_dim=cfg.action_space,
        rollout_steps=args.rollout_steps,
        total_timesteps=args.total_timesteps,
        max_steps_per_episode=int(cfg.episode_length_s / (cfg.sim.dt * cfg.decimation)),
        seed=args.seed,
    )

    agent = PPOAgent(wrapped, ppo_cfg)
    stats = agent.train()

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    agent.save_checkpoint(args.checkpoint_dir)

    print(f"\nFinal: total_steps={stats['total_steps']:,} "
          f"episodes={stats['total_episodes']} "
          f"mean_return_last10={stats['mean_return_last10']:.2f}")

    isaac_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
