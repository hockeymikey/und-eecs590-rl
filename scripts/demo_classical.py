#!/usr/bin/env python3
"""Demo: Run all classical algorithms on WindyChasm and compare results.

Shows a side-by-side comparison of MC, Q-learning, TD, and SARSA variants
with training curves and final evaluation performance.
"""
import sys
sys.path.insert(0, "M3")

import numpy as np
from rl590.envs.windy_chasm import WindyChasmMDP
from rl590.classical.mc import MCAgent, MCConfig
from rl590.classical.qlearning import QLearningAgent, QLearningConfig
from rl590.classical.td import TDPredictionAgent, TDConfig
from rl590.classical.sarsa import SarsaAgent, SarsaConfig


def run_experiment(name, agent_cls, config, env_fn):
    env = env_fn()
    agent = agent_cls(env, config)
    stats = agent.train()
    ev = agent.evaluate()
    return name, stats, ev


def main():
    EPISODES = 1000
    DECAY = 0.998

    env_fn = WindyChasmMDP
    print("=" * 70)
    print("Classical RL Algorithm Comparison — WindyChasm (140 states, 3 actions)")
    print("=" * 70)
    print(f"Training episodes: {EPISODES}  |  ε-decay: {DECAY}")
    print()

    experiments = [
        ("MC First-Visit", MCAgent, MCConfig(
            episodes=EPISODES, epsilon_decay=DECAY)),
        ("Q-Learning", QLearningAgent, QLearningConfig(
            episodes=EPISODES, epsilon_decay=DECAY)),
        ("TD(4) Forward", TDPredictionAgent, TDConfig(
            algorithm="td_n_forward", n=4, episodes=EPISODES, epsilon_decay=DECAY)),
        ("TD(4) Backward", TDPredictionAgent, TDConfig(
            algorithm="td_n_backward", n=4, episodes=EPISODES, epsilon_decay=DECAY)),
        ("TD(λ) Forward", TDPredictionAgent, TDConfig(
            algorithm="td_lambda_forward", lam=0.8, episodes=EPISODES, epsilon_decay=DECAY)),
        ("TD(λ) Backward", TDPredictionAgent, TDConfig(
            algorithm="td_lambda_backward", lam=0.8, episodes=EPISODES, epsilon_decay=DECAY)),
        ("SARSA(4) Forward", SarsaAgent, SarsaConfig(
            algorithm="sarsa_n_forward", n=4, episodes=EPISODES, epsilon_decay=DECAY)),
        ("SARSA(4) Backward", SarsaAgent, SarsaConfig(
            algorithm="sarsa_n_backward", n=4, episodes=EPISODES, epsilon_decay=DECAY)),
        ("SARSA(λ) Forward", SarsaAgent, SarsaConfig(
            algorithm="sarsa_lambda_forward", lam=0.8, episodes=EPISODES, epsilon_decay=DECAY)),
        ("SARSA(λ) Backward", SarsaAgent, SarsaConfig(
            algorithm="sarsa_lambda_backward", lam=0.8, episodes=EPISODES, epsilon_decay=DECAY)),
    ]

    results = []
    for name, cls, cfg in experiments:
        print(f"  Training {name}...", end="", flush=True)
        name, stats, ev = run_experiment(name, cls, cfg, env_fn)
        results.append((name, stats, ev))
        print(f"  done (eval return: {ev['mean_return']:.1f})")

    # Print comparison table
    print()
    print("-" * 70)
    print(f"{'Algorithm':<25s} {'Train (last 100)':>16s} {'Eval Return':>14s} {'Eval Std':>10s}")
    print("-" * 70)
    for name, stats, ev in results:
        print(f"{name:<25s} {stats['mean_return_last100']:>16.2f} {ev['mean_return']:>14.2f} {ev.get('std_return', 0):>10.2f}")
    print("-" * 70)

    # Also run on ReactorEnv
    from reactor_env import ReactorEnv
    print()
    print("=" * 70)
    print("Classical RL Algorithm Comparison — ReactorEnv (20 states, 5 actions)")
    print("=" * 70)
    print()

    reactor_experiments = [
        ("MC First-Visit", MCAgent, MCConfig(
            episodes=EPISODES, epsilon_decay=DECAY, gamma=0.95)),
        ("Q-Learning", QLearningAgent, QLearningConfig(
            episodes=EPISODES, epsilon_decay=DECAY, gamma=0.95)),
        ("SARSA(λ) Backward", SarsaAgent, SarsaConfig(
            algorithm="sarsa_lambda_backward", lam=0.8,
            episodes=EPISODES, epsilon_decay=DECAY, gamma=0.95)),
    ]

    results2 = []
    for name, cls, cfg in reactor_experiments:
        print(f"  Training {name}...", end="", flush=True)
        name, stats, ev = run_experiment(name, cls, cfg, ReactorEnv)
        results2.append((name, stats, ev))
        print(f"  done (eval return: {ev['mean_return']:.1f})")

    print()
    print("-" * 70)
    print(f"{'Algorithm':<25s} {'Train (last 100)':>16s} {'Eval Return':>14s}")
    print("-" * 70)
    for name, stats, ev in results2:
        print(f"{name:<25s} {stats['mean_return_last100']:>16.2f} {ev['mean_return']:>14.2f}")
    print("-" * 70)


if __name__ == "__main__":
    main()
