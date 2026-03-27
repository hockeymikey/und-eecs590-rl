#!/usr/bin/env python3
"""Demo: Run PPO on a simple continuous control environment.

Uses a dummy environment with image observations and continuous actions
to demonstrate that the PPO implementation works end-to-end. For the
real Zamboni, install the foundation environment and adjust the obs dims.

The dummy env rewards the agent for taking small actions (penalizes
large throttle/steering). A successful PPO should learn to output
near-zero actions.
"""
import numpy as np
import torch

from rl590.deep.ppo import PPOAgent, PPOConfig


class DemoContinuousEnv:
    """Simple continuous control env with image-like observations.

    Observation: random (H, W, C) image (simulates noisy sensor input).
    Actions: 2D continuous in [-1, 1] (throttle, steering).
    Reward: -|action|^2 (learn to take small actions) + small bonus per step.
    """

    def __init__(self, h=32, w=64, c=2, max_steps=100):
        self.h, self.w, self.c = h, w, c
        self.max_steps = max_steps
        self._step = 0
        self._rng = np.random.default_rng()

    def reset(self, seed=None):
        self._step = 0
        self._rng = np.random.default_rng(seed)
        obs = self._rng.standard_normal((self.h, self.w, self.c)).astype(np.float32)
        return obs, {}

    def step(self, action):
        self._step += 1
        obs = self._rng.standard_normal((self.h, self.w, self.c)).astype(np.float32)
        # Reward: penalize large actions, bonus for surviving
        reward = -float(np.sum(action ** 2)) + 0.1
        terminated = False
        truncated = self._step >= self.max_steps
        return obs, reward, terminated, truncated, {}


def main():
    print("=" * 60)
    print("PPO Demo — Continuous Control with Image Observations")
    print("=" * 60)
    print()

    H, W, C = 32, 64, 2
    env = DemoContinuousEnv(h=H, w=W, c=C, max_steps=100)

    config = PPOConfig(
        obs_height=H, obs_width=W, n_channels=C, action_dim=2,
        features_dim=64, hidden_dim=32,
        rollout_steps=512,
        n_epochs=5,
        batch_size=64,
        total_timesteps=5_000,
        max_steps_per_episode=100,
        log_interval=1,
        learning_rate=3e-4,
    )

    agent = PPOAgent(env, config)

    device = "CUDA" if torch.cuda.is_available() else "CPU"
    actor_params = sum(p.numel() for p in agent.actor.parameters())
    critic_params = sum(p.numel() for p in agent.critic.parameters())
    print(f"Device: {device}")
    print(f"Actor parameters:  {actor_params:,}")
    print(f"Critic parameters: {critic_params:,}")
    print(f"Rollout steps: {config.rollout_steps}")
    print(f"Total timesteps: {config.total_timesteps:,}")
    print()

    stats = agent.train()

    print()
    print("-" * 60)
    print(f"Training complete!")
    print(f"  Total steps:    {stats['total_steps']:,}")
    print(f"  Total episodes: {stats['total_episodes']}")
    print(f"  Mean return (last 10): {stats['mean_return_last10']:.2f}")

    # Evaluate
    print()
    print("Evaluating deterministic policy...")
    eval_stats = agent.evaluate(n_episodes=10)
    print(f"  Mean return: {eval_stats['mean_return']:.2f} +/- {eval_stats['std_return']:.2f}")
    print(f"  Mean length: {eval_stats['mean_length']:.1f}")

    # Show sample actions
    print()
    print("Sample deterministic actions:")
    obs, _ = env.reset(seed=42)
    for i in range(5):
        action = agent.predict(obs, deterministic=True)
        print(f"  Step {i}: throttle={action[0]:+.3f}, steering={action[1]:+.3f}")
        obs, _, _, _, _ = env.step(action)

    # Save checkpoint
    ckpt_path = "checkpoints/ppo/demo_run"
    agent.save_checkpoint(ckpt_path)
    print(f"\nCheckpoint saved to {ckpt_path}/")
    print(f"  Files: actor.pt, critic.pt, optimizer.pt, hparams.json, training_log.json")


if __name__ == "__main__":
    main()
