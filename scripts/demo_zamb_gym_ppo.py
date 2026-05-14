#!/usr/bin/env python3
"""Train the from-scratch PPO on ``ZambGymEnv``.

Mirrors ``demo_ppo.py`` but points at the real ice-resurfacing gym env.
Supports an optional ``--bc-init <path>`` flag that loads BC-warmstarted
actor weights from a ``.pt`` file before training begins.

Per-rollout metrics are written to TensorBoard under
``<checkpoint-dir>/tb/<run-tag>/``. Launch TensorBoard separately to view
them.
"""

from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from rl590.deep.ppo import PPOAgent, PPOConfig
from rl590.envs.zamb_gym import ZambGymEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int, default=50_000)
    p.add_argument("--rollout-steps", type=int, default=1024)
    p.add_argument("--max-episode-steps", type=int, default=400)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bc-init", type=Path, default=None,
                   help="Path to a BC-warmstarted actor .pt to load before PPO.")
    p.add_argument("--checkpoint-dir", type=Path,
                   default=Path("training_runs/zamb_gym_ppo_v1"))
    p.add_argument("--run-tag", type=str, default=None,
                   help="TB subdir under <checkpoint-dir>/tb/ (default: timestamped).")
    p.add_argument("--log-interval", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    env = ZambGymEnv(max_steps=args.max_episode_steps)
    obs_h, obs_w, obs_c = env.observation_space.shape

    config = PPOConfig(
        obs_height=obs_h,
        obs_width=obs_w,
        n_channels=obs_c,
        action_dim=2,
        rollout_steps=args.rollout_steps,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        entropy_coef=args.ent_coef,
        total_timesteps=args.total_timesteps,
        max_steps_per_episode=args.max_episode_steps,
        log_interval=args.log_interval,
        seed=args.seed,
    )

    run_tag = args.run_tag or _dt.datetime.now().strftime("ppo_%Y%m%d_%H%M%S")
    if args.bc_init is not None:
        run_tag = f"{run_tag}_bcinit"
    tb_dir = args.checkpoint_dir / "tb" / run_tag
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))

    agent = PPOAgent(env, config, tb_writer=writer)

    device = "CUDA" if torch.cuda.is_available() else "CPU"
    actor_params = sum(p.numel() for p in agent.actor.parameters())
    critic_params = sum(p.numel() for p in agent.critic.parameters())
    print(f"Device: {device}")
    print(f"Obs shape: ({obs_h}, {obs_w}, {obs_c})")
    print(f"Actor parameters:  {actor_params:,}")
    print(f"Critic parameters: {critic_params:,}")
    print(f"Rollout steps: {config.rollout_steps}")
    print(f"Total timesteps: {config.total_timesteps:,}")
    print(f"TensorBoard dir:  {tb_dir}")

    if args.bc_init is not None:
        if not args.bc_init.exists():
            raise SystemExit(f"--bc-init path does not exist: {args.bc_init}")
        state = torch.load(args.bc_init, map_location=agent.device, weights_only=True)
        agent.actor.load_state_dict(state)
        print(f"[bc-init] loaded actor weights from {args.bc_init}")

    stats = agent.train()
    print(
        f"Training complete: total_steps={stats['total_steps']:,} "
        f"episodes={stats['total_episodes']} "
        f"mean_return_last10={stats['mean_return_last10']:.2f}"
    )

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    agent.save_checkpoint(args.checkpoint_dir)
    print(f"Checkpoint saved to {args.checkpoint_dir}")

    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
