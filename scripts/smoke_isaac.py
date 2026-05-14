"""First-run smoke test for ZambIsaacEnv.

Spawns one env, resets it, and runs 100 zero-action steps. Prints obs
shape, reward stats, and final chassis position.

Run inside the Isaac Lab container:

  python scripts/smoke_isaac.py \
      --zamboni-usd /workspace/Minis/assets/zamboni.usd \
      --rink-usd /workspace/Minis/assets/ice.usd
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zamboni-usd", required=True)
    parser.add_argument("--rink-usd", required=True)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args()

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app

    import torch

    from rl590.envs.zamb_isaac import ZambIsaacEnv
    from rl590.envs.zamb_isaac_cfg import ZambIsaacEnvCfg

    cfg = ZambIsaacEnvCfg(
        zamboni_usd=args.zamboni_usd,
        rink_usd=args.rink_usd,
    )
    cfg.scene.num_envs = args.num_envs
    cfg.__post_init__()

    env = ZambIsaacEnv(cfg)

    obs_dict, info = env.reset()
    obs = obs_dict["policy"]
    print(f"obs shape (flat): {tuple(obs.shape)}")
    print(f"  reshaped to (P, P, C): "
          f"({cfg.obs_patch_cells}, {cfg.obs_patch_cells}, {cfg.obs_channels})")

    rewards = []
    zero_action = torch.zeros(args.num_envs, cfg.action_space, device=env.device)
    for step in range(args.steps):
        obs_dict, reward, terminated, truncated, info = env.step(zero_action)
        rewards.append(reward.mean().item())
        if terminated.any() or truncated.any():
            print(f"  step {step}: terminated={terminated.tolist()} truncated={truncated.tolist()}")

    pos = env.zamb.data.root_pos_w
    print(f"final chassis pos (world): {pos[0].tolist()}")
    print(f"reward mean over {args.steps} zero-action steps: {sum(rewards)/len(rewards):+.4f}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
