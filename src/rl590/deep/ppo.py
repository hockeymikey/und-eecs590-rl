"""Proximal Policy Optimization (PPO) for continuous control.

PPO is an on-policy, actor-critic, policy gradient algorithm. The key
idea is the *clipped surrogate objective* — it limits how much the policy
can change in a single update, preventing the destructive large steps
that plague vanilla policy gradients.

The training loop repeats:
    1. COLLECT — run the current policy in the env for N steps, storing
       transitions (obs, action, reward, done, log_prob, value).
    2. COMPUTE ADVANTAGES — use Generalized Advantage Estimation (GAE)
       to compute how much better each action was than expected.
    3. UPDATE — run K epochs of minibatch gradient descent on the
       collected data, optimizing the clipped policy loss + value loss
       + entropy bonus.

Key concepts:
    - GAE(λ): the same λ-return idea from TD(λ), applied to advantages.
      Â_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
      where δ_t = r_t + γV(s_{t+1}) - V(s_t) is the TD error.

    - Clipped objective: the ratio r_t = π_new(a|s) / π_old(a|s) measures
      how much the policy changed. PPO clips this ratio to [1-ε, 1+ε]
      so the policy can't jump too far in one update.

    - Entropy bonus: encourages exploration by penalizing policies that
      become too deterministic too quickly.

Reference: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from rl590.networks import ZamboniCNN, GaussianActor, Critic
from rl590.primitives import BatchKey


@dataclass
class PPOConfig:
    """PPO hyperparameters.

    These defaults are standard values from the PPO paper and stable-baselines3.
    """

    # Rollout
    rollout_steps: int = 2048     # steps to collect per rollout before updating
    gamma: float = 0.99           # discount factor
    gae_lambda: float = 0.95      # GAE λ — bias/variance tradeoff for advantages

    # PPO update
    n_epochs: int = 10            # gradient epochs per rollout
    batch_size: int = 64          # minibatch size within each epoch
    clip_epsilon: float = 0.2     # clipping range for policy ratio
    clip_value: bool = True       # also clip value function updates
    max_grad_norm: float = 0.5    # gradient clipping (L2 norm)

    # Loss coefficients
    value_coef: float = 0.5       # weight of critic loss in total loss
    entropy_coef: float = 0.01    # weight of entropy bonus

    # Optimizer
    learning_rate: float = 3e-4

    # Network architecture
    features_dim: int = 128       # CNN output features
    hidden_dim: int = 64          # MLP head hidden layer width

    # Training
    total_timesteps: int = 100_000
    max_steps_per_episode: int = 2000
    log_interval: int = 5         # print stats every N rollouts
    seed: int = 0

    # Environment (observation shape)
    obs_height: int = 129
    obs_width: int = 304
    n_channels: int = 2
    action_dim: int = 2


class RolloutBuffer:
    """Stores one rollout of experience for PPO updates.

    PPO is on-policy: we collect data with the current policy, use it
    for a few epochs of updates, then throw it away and collect fresh data.
    This is NOT a replay buffer — the data is used once and discarded.
    """

    def __init__(self) -> None:
        self.observations: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []

        # Computed after rollout
        self.advantages: np.ndarray | None = None
        self.returns: np.ndarray | None = None

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ) -> None:
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float) -> None:
        """Compute Generalized Advantage Estimation.

        GAE is TD(λ) applied to advantage estimation:
            δ_t = r_t + γV(s_{t+1}) - V(s_t)              (TD error)
            Â_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ... (GAE)

        The return target for the critic is:
            G_t = Â_t + V(s_t)

        We compute this backward through the rollout, same as TD(λ)
        backward-view but without updating a value function — just
        computing the advantage estimates.

        Parameters
        ----------
        last_value : float
            V(s_T) — the value estimate of the state after the last step.
            Used to bootstrap if the episode didn't end.
        gamma : float
            Discount factor.
        gae_lambda : float
            λ for GAE. 0 = TD(0) advantage (high bias, low variance).
            1 = MC advantage (low bias, high variance). 0.95 is typical.
        """
        n = len(self.rewards)
        self.advantages = np.zeros(n, dtype=np.float32)
        self.returns = np.zeros(n, dtype=np.float32)

        # Walk backward through the rollout
        gae = 0.0
        next_value = last_value
        for t in range(n - 1, -1, -1):
            # If this step ended an episode, next_value should be 0
            # (no future rewards from a terminal state)
            if self.dones[t]:
                next_value = 0.0
                gae = 0.0

            # TD error for this step
            delta = self.rewards[t] + gamma * next_value - self.values[t]

            # GAE accumulation (same as eligibility trace update)
            gae = delta + gamma * gae_lambda * gae

            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]  # G_t = Â_t + V(s_t)

            next_value = self.values[t]

    def get_batches(self, batch_size: int, device: torch.device):
        """Yield randomized minibatches for one epoch of PPO updates.

        Yields
        ------
        dict with keys: obs, actions, old_log_probs, advantages, returns
            All as tensors on the specified device.
        """
        n = len(self.rewards)
        indices = np.random.permutation(n)

        obs_array = np.array(self.observations, dtype=np.float32)
        actions_array = np.array(self.actions, dtype=np.float32)
        log_probs_array = np.array(self.log_probs, dtype=np.float32)

        # Normalize advantages (reduces variance, standard practice)
        adv = self.advantages.copy()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]

            yield {
                BatchKey.OBS: torch.tensor(obs_array[idx], device=device),
                BatchKey.ACTIONS: torch.tensor(actions_array[idx], device=device),
                BatchKey.OLD_LOG_PROBS: torch.tensor(log_probs_array[idx], device=device),
                BatchKey.ADVANTAGES: torch.tensor(adv[idx], device=device),
                BatchKey.RETURNS: torch.tensor(self.returns[idx], device=device),
            }

    def clear(self) -> None:
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
        self.advantages = None
        self.returns = None

    def __len__(self) -> int:
        return len(self.rewards)


class PPOAgent:
    """PPO agent for continuous control with CNN observations.

    Parameters
    ----------
    env : gymnasium.Env or compatible
        Must provide: reset() -> (obs, info), step(action) -> (obs, r, term, trunc, info)
        Observation shape: (H, W, C)
        Action space: continuous, shape (action_dim,)
    config : PPOConfig
        Hyperparameters.
    """

    def __init__(self, env, config: PPOConfig | None = None, tb_writer=None) -> None:
        self.env = env
        self.config = config or PPOConfig()
        self.tb_writer = tb_writer
        cfg = self.config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build networks — separate CNNs for actor and critic
        # (separate weights lets them learn different features)
        actor_cnn = ZamboniCNN(
            cfg.obs_height, cfg.obs_width, cfg.n_channels, cfg.features_dim
        )
        critic_cnn = ZamboniCNN(
            cfg.obs_height, cfg.obs_width, cfg.n_channels, cfg.features_dim
        )
        self.actor = GaussianActor(
            actor_cnn, cfg.action_dim, cfg.hidden_dim
        ).to(self.device)
        self.critic = Critic(
            critic_cnn, cfg.hidden_dim
        ).to(self.device)

        # Single optimizer for both networks (standard PPO practice)
        self.optimizer = Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=cfg.learning_rate,
        )

        self.buffer = RolloutBuffer()

        # Training tracking
        self.total_steps = 0
        self.episode_returns: List[float] = []
        self.episode_lengths: List[int] = []
        self.rollout_stats: List[Dict] = []

    @torch.no_grad()
    def _get_value(self, obs: np.ndarray) -> float:
        """Get V(s) for a single observation."""
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.critic(obs_t).item()

    @torch.no_grad()
    def _get_action(self, obs: np.ndarray) -> tuple[np.ndarray, float]:
        """Sample action and log-prob from current policy."""
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, log_prob = self.actor.sample_action(obs_t)
        return action.cpu().numpy()[0], log_prob.item()

    def collect_rollout(self) -> Dict[str, float]:
        """Run the current policy for rollout_steps, storing transitions.

        This is step 1 of PPO: gather experience with the current policy.
        The data will be used for several epochs of updates, then discarded.

        Returns
        -------
        stats : dict with episode return/length info from this rollout
        """
        cfg = self.config
        self.buffer.clear()

        obs, _ = self.env.reset()
        episode_return = 0.0
        episode_length = 0
        rollout_returns = []
        rollout_lengths = []

        for step in range(cfg.rollout_steps):
            # Get action and value from current networks
            action, log_prob = self._get_action(obs)
            value = self._get_value(obs)

            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            episode_return += reward
            episode_length += 1
            self.total_steps += 1

            # Store transition
            self.buffer.add(obs, action, reward, done, log_prob, value)

            obs = next_obs

            if done:
                rollout_returns.append(episode_return)
                rollout_lengths.append(episode_length)
                self.episode_returns.append(episode_return)
                self.episode_lengths.append(episode_length)
                obs, _ = self.env.reset()
                episode_return = 0.0
                episode_length = 0

        # Bootstrap value for the last state (if episode didn't end)
        last_value = self._get_value(obs) if not done else 0.0

        # Compute GAE advantages
        self.buffer.compute_gae(last_value, cfg.gamma, cfg.gae_lambda)

        return {
            "episodes_completed": len(rollout_returns),
            "mean_return": float(np.mean(rollout_returns)) if rollout_returns else 0.0,
            "mean_length": float(np.mean(rollout_lengths)) if rollout_lengths else 0.0,
        }

    def update(self) -> Dict[str, float]:
        """Run K epochs of minibatch PPO updates on the collected rollout.

        This is steps 2-3 of PPO:
            - For each epoch, shuffle the rollout data into minibatches
            - For each minibatch, compute and optimize the PPO loss

        The loss has three components:
            L = L_policy + c1 * L_value - c2 * L_entropy

        Returns
        -------
        stats : dict with mean losses across all updates
        """
        cfg = self.config
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch in range(cfg.n_epochs):
            for batch in self.buffer.get_batches(cfg.batch_size, self.device):
                obs = batch[BatchKey.OBS]
                actions = batch[BatchKey.ACTIONS]
                old_log_probs = batch[BatchKey.OLD_LOG_PROBS]
                advantages = batch[BatchKey.ADVANTAGES]
                returns = batch[BatchKey.RETURNS]

                # Clipped surrogate policy loss.
                new_log_probs = self.actor.log_prob(obs, actions)

                # Importance ratio: how much has the policy changed?
                # r_t = π_new(a|s) / π_old(a|s) = exp(log_new - log_old)
                ratio = torch.exp(new_log_probs - old_log_probs)

                # Clipped objective: min of unclipped and clipped
                # This prevents the ratio from going too far from 1.0
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_epsilon, 1.0 + cfg.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Critic regression target.
                values = self.critic(obs)
                value_loss = nn.functional.mse_loss(values, returns)

                # Entropy bonus encourages exploration.
                entropy = self.actor.entropy(obs).mean()

                loss = (
                    policy_loss
                    + cfg.value_coef * value_loss
                    - cfg.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                # Clip gradients to prevent exploding gradients
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    cfg.max_grad_norm,
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        return {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
            "n_updates": n_updates,
        }

    def train(self) -> Dict[str, float | int]:
        """Main PPO training loop.

        Alternates between collecting rollouts and updating networks
        until total_timesteps is reached.
        """
        cfg = self.config
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        n_rollouts = 0

        while self.total_steps < cfg.total_timesteps:
            rollout_stats = self.collect_rollout()
            n_rollouts += 1

            # Advantages are computed at the end of rollout collection.
            update_stats = self.update()

            self.rollout_stats.append({**rollout_stats, **update_stats})

            if self.tb_writer is not None:
                step = self.total_steps
                recent = self.episode_returns[-10:] if self.episode_returns else [0.0]
                self.tb_writer.add_scalar("rollout/mean_return", rollout_stats["mean_return"], step)
                self.tb_writer.add_scalar("rollout/mean_return_last10", float(np.mean(recent)), step)
                self.tb_writer.add_scalar("rollout/mean_length", rollout_stats["mean_length"], step)
                self.tb_writer.add_scalar("rollout/episodes_completed", rollout_stats["episodes_completed"], step)
                self.tb_writer.add_scalar("train/policy_loss", update_stats["policy_loss"], step)
                self.tb_writer.add_scalar("train/value_loss", update_stats["value_loss"], step)
                self.tb_writer.add_scalar("train/entropy", update_stats["entropy"], step)

            # Log progress
            if n_rollouts % cfg.log_interval == 0:
                recent = self.episode_returns[-10:] if self.episode_returns else [0]
                print(
                    f"Rollout {n_rollouts} | "
                    f"Steps: {self.total_steps:,} | "
                    f"Mean return (last 10): {np.mean(recent):.1f} | "
                    f"Policy loss: {update_stats['policy_loss']:.4f} | "
                    f"Value loss: {update_stats['value_loss']:.4f} | "
                    f"Entropy: {update_stats['entropy']:.3f}"
                )

        return {
            "total_steps": self.total_steps,
            "total_episodes": len(self.episode_returns),
            "mean_return_last10": float(np.mean(self.episode_returns[-10:])) if self.episode_returns else 0.0,
            "n_rollouts": n_rollouts,
        }

    @torch.no_grad()
    def predict(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Get action for a single observation.

        Parameters
        ----------
        obs : ndarray of shape (H, W, C)
        deterministic : bool
            If True, return the mean action (no sampling).

        Returns
        -------
        action : ndarray of shape (action_dim,)
        """
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if deterministic:
            mean, _ = self.actor(obs_t)
            return torch.tanh(mean).cpu().numpy()[0]
        action, _ = self.actor.sample_action(obs_t)
        return action.cpu().numpy()[0]

    def evaluate(self, n_episodes: int = 10, seed: int = 9999) -> Dict[str, float]:
        """Evaluate deterministic policy."""
        returns = []
        lengths = []
        rng = np.random.default_rng(seed)

        for _ in range(n_episodes):
            # Use seed for reproducibility if env supports it
            obs, _ = self.env.reset(seed=int(rng.integers(100000)))
            total_reward = 0.0
            for step in range(self.config.max_steps_per_episode):
                action = self.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
            returns.append(total_reward)
            lengths.append(step + 1)

        return {
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "mean_length": float(np.mean(lengths)),
        }

    def save_checkpoint(self, path: str | Path) -> Path:
        """Save actor, critic, optimizer state, and hyperparameters."""
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)

        torch.save(self.actor.state_dict(), out / "actor.pt")
        torch.save(self.critic.state_dict(), out / "critic.pt")
        torch.save(self.optimizer.state_dict(), out / "optimizer.pt")

        # Save hyperparameters and training stats
        meta = {
            "config": asdict(self.config),
            "total_steps": self.total_steps,
            "total_episodes": len(self.episode_returns),
            "device": str(self.device),
        }
        with open(out / "hparams.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Save training log (convert numpy scalars to Python floats)
        def _to_python(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, dict):
                return {k: _to_python(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_python(v) for v in obj]
            return obj

        log = _to_python({
            "episode_returns": self.episode_returns,
            "episode_lengths": self.episode_lengths,
            "rollout_stats": self.rollout_stats,
        })
        with open(out / "training_log.json", "w") as f:
            json.dump(log, f)

        return out

    def load_checkpoint(self, path: str | Path) -> None:
        """Load actor, critic, and optimizer state from a checkpoint."""
        p = Path(path)
        self.actor.load_state_dict(
            torch.load(p / "actor.pt", map_location=self.device, weights_only=True)
        )
        self.critic.load_state_dict(
            torch.load(p / "critic.pt", map_location=self.device, weights_only=True)
        )
        if (p / "optimizer.pt").exists():
            self.optimizer.load_state_dict(
                torch.load(p / "optimizer.pt", map_location=self.device, weights_only=True)
            )

        if (p / "training_log.json").exists():
            with open(p / "training_log.json") as f:
                log = json.load(f)
            self.episode_returns = log.get("episode_returns", [])
            self.episode_lengths = log.get("episode_lengths", [])
            self.rollout_stats = log.get("rollout_stats", [])
