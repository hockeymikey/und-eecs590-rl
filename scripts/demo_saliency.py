#!/usr/bin/env python3
"""Demo: Generate saliency maps for a PPO agent's networks.

Creates vanilla gradient and Grad-CAM visualizations showing what
the CNN attends to when making decisions. Saves plots to artifacts/.
"""
import numpy as np
import torch

from rl590.networks import ZamboniCNN, GaussianActor, Critic
from rl590.visualization import vanilla_gradient_saliency, gradcam, plot_saliency


def main():
    print("=" * 60)
    print("Saliency Analysis Demo")
    print("=" * 60)
    print()

    # Use Zamboni-like dimensions
    H, W, C = 129, 304, 2

    # Build networks
    print("Building networks...")
    actor_cnn = ZamboniCNN(H, W, C, features_dim=128)
    actor = GaussianActor(actor_cnn, action_dim=2, hidden_dim=64)
    critic_cnn = ZamboniCNN(H, W, C, features_dim=128)
    critic = Critic(critic_cnn, hidden_dim=64)

    # Create a synthetic observation that looks like Zamboni input
    # Channel 0: "ice roughness" — mostly smooth with a damaged patch
    # Channel 1: "agent position" — a blob in the lower-left
    print("Creating synthetic Zamboni-like observation...")
    obs = np.zeros((H, W, C), dtype=np.float32)

    # Ice roughness: smooth background with a damaged region
    obs[:, :, 0] = np.random.randn(H, W) * 0.1  # background noise
    obs[40:80, 100:200, 0] = np.random.randn(40, 100) * 2.0  # damaged patch

    # Agent position: Gaussian blob at (100, 80)
    yy, xx = np.mgrid[0:H, 0:W]
    obs[:, :, 1] = np.exp(-((yy - 100)**2 + (xx - 80)**2) / (2 * 20**2))

    # Generate saliency maps
    print()
    print("Computing saliency maps...")

    # Critic — what does the value function attend to?
    sal_critic_grad = vanilla_gradient_saliency(critic, obs)
    sal_critic_cam = gradcam(critic, obs)
    print(f"  Critic vanilla gradient: non-zero pixels = {(sal_critic_grad > 0.01).sum()}/{H*W}")
    print(f"  Critic Grad-CAM: non-zero pixels = {(sal_critic_cam > 0.01).sum()}/{H*W}")

    # Actor — what does the policy attend to for each action?
    sal_throttle = vanilla_gradient_saliency(actor, obs, output_index=0)
    sal_steering = vanilla_gradient_saliency(actor, obs, output_index=1)
    sal_actor_cam = gradcam(actor, obs, output_index=0)
    print(f"  Actor throttle gradient: non-zero pixels = {(sal_throttle > 0.01).sum()}/{H*W}")
    print(f"  Actor steering gradient: non-zero pixels = {(sal_steering > 0.01).sum()}/{H*W}")
    print(f"  Actor Grad-CAM: non-zero pixels = {(sal_actor_cam > 0.01).sum()}/{H*W}")

    # Save plots
    print()
    print("Saving saliency plots to artifacts/...")

    plot_saliency(obs, sal_critic_grad,
                  title="Critic — Vanilla Gradient Saliency",
                  output_path="artifacts/saliency_critic_gradient.png")

    plot_saliency(obs, sal_critic_cam,
                  title="Critic — Grad-CAM",
                  output_path="artifacts/saliency_critic_gradcam.png")

    plot_saliency(obs, sal_throttle,
                  title="Actor (Throttle) — Vanilla Gradient",
                  output_path="artifacts/saliency_actor_throttle.png")

    plot_saliency(obs, sal_steering,
                  title="Actor (Steering) — Vanilla Gradient",
                  output_path="artifacts/saliency_actor_steering.png")

    plot_saliency(obs, sal_actor_cam,
                  title="Actor — Grad-CAM",
                  output_path="artifacts/saliency_actor_gradcam.png")

    print()
    print("Done! Check artifacts/ for the generated plots.")
    print()
    print("Note: These are from a randomly initialized network, so the")
    print("saliency patterns won't be meaningful yet. After training PPO")
    print("on the Zamboni, re-run this with the trained checkpoint to see")
    print("what the network has actually learned to attend to.")


if __name__ == "__main__":
    main()
