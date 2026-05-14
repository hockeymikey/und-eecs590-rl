"""SB3 PPO training over N parallel ZambIsaacEnv instances.

Isaac platform-validation path. Uses Isaac Lab's built-in `Sb3VecEnvWrapper`
to adapt the multi-env DirectRLEnv to SB3's VecEnv API.

Run inside the Isaac Lab container:

  python scripts/train_zamb_isaac_sb3.py \
      --zamboni-usd /workspace/Minis/assets/zamboni.usd \
      --rink-usd /workspace/Minis/assets/ice.usd \
      --num-envs 8 \
      --total-timesteps 200000
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zamboni-usd", required=True)
    parser.add_argument("--rink-usd", required=True)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("training_runs/zamb_isaac_sb3_v1"))
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args()

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app

    from isaaclab_rl.sb3 import Sb3VecEnvWrapper
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback

    from rl590.envs.zamb_isaac import ZambIsaacEnv
    from rl590.envs.zamb_isaac_cfg import ZambIsaacEnvCfg

    cfg = ZambIsaacEnvCfg(
        zamboni_usd=args.zamboni_usd,
        rink_usd=args.rink_usd,
    )
    cfg.scene.num_envs = args.num_envs
    cfg.__post_init__()

    env = ZambIsaacEnv(cfg)
    vec_env = Sb3VecEnvWrapper(env)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(args.checkpoint_dir / "tb"),
        n_steps=512,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        learning_rate=3e-4,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=max(10_000 // args.num_envs, 1),
        save_path=str(args.checkpoint_dir),
        name_prefix="ppo_zamb_isaac",
    )

    model.learn(total_timesteps=args.total_timesteps, callback=ckpt_cb)
    model.save(args.checkpoint_dir / "final.zip")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
