#!/usr/bin/env python3
"""Experiment runner for Mini 3 Problem 1: Reactor control.

Subcommands
-----------
noise-sweep     Q3a  – learning curves at different σ_obs
q-heatmap       Q3b  – Q-function heatmap
compare-algos   Q3c  – SARSA(λ) vs Q-learning comparison
fa-compare      Q4a  – linear FA vs tabular
nonstationarity Q4b  – fixed LR vs decaying LR
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from reactor_env import ReactorConfig, ReactorEnv
from td_agent import TDConfig, TDAgent
from fa_agent import FAConfig, FAAgent
from plots import (
    plot_learning_curves,
    plot_q_heatmap,
    plot_algorithm_comparison,
    plot_meltdown_curve,
)


def _ensure_dir(d: str) -> str:
    Path(d).mkdir(parents=True, exist_ok=True)
    return d


# ======================================================================
# Q3a: Noise sweep
# ======================================================================

def cmd_noise_sweep(args: argparse.Namespace) -> None:
    """Train at multiple observation noise levels and plot learning curves."""
    sigma_values = [0.5, 1.5]
    all_returns = []
    labels = []

    for sigma in sigma_values:
        print(f"Training SARSA(λ) with σ_obs={sigma} ...")
        cfg_env = ReactorConfig(sigma_obs=sigma)
        env = ReactorEnv(cfg_env)
        cfg_td = TDConfig(
            algorithm="sarsa_lambda",
            episodes=args.episodes,
            seed=args.seed,
        )
        agent = TDAgent(env, cfg_td)
        summary = agent.train()
        print(f"  Done: {summary}")
        all_returns.append(agent.episode_returns)
        labels.append(f"σ_obs = {sigma}")

        if args.output_dir:
            agent.save(os.path.join(args.output_dir, f"sarsa_sigma{sigma}.npz"))

    out = os.path.join(args.output_dir, "noise_sweep.png") if args.output_dir else None
    plot_learning_curves(all_returns, labels, window=args.window, output_path=out,
                         title="Q3a: Learning Curves at Different Noise Levels")

    # Also plot meltdown rates
    # Re-extract meltdown lists
    # (we need to re-train or store — let's just train fresh Q-learning too for comparison)
    # For simplicity, use the agents we already have — but we only have SARSA.
    # Let's also do Q-learning at each noise level.
    all_returns_ql = []
    labels_ql = []
    for sigma in sigma_values:
        print(f"Training Q-learning with σ_obs={sigma} ...")
        cfg_env = ReactorConfig(sigma_obs=sigma)
        env = ReactorEnv(cfg_env)
        cfg_td = TDConfig(
            algorithm="qlearning",
            episodes=args.episodes,
            seed=args.seed,
        )
        agent = TDAgent(env, cfg_td)
        agent.train()
        all_returns_ql.append(agent.episode_returns)
        labels_ql.append(f"Q-learning, σ_obs = {sigma}")

    out2 = os.path.join(args.output_dir, "noise_sweep_both.png") if args.output_dir else None
    plot_learning_curves(
        all_returns + all_returns_ql,
        [f"SARSA(λ), {l}" for l in labels] + labels_ql,
        window=args.window,
        output_path=out2,
        title="Q3a: SARSA(λ) and Q-learning at Different Noise Levels",
    )
    print("Noise sweep complete.")


# ======================================================================
# Q3b: Q-function heatmap
# ======================================================================

def cmd_q_heatmap(args: argparse.Namespace) -> None:
    """Train and render Q-function heatmap."""
    env = ReactorEnv(ReactorConfig())
    cfg_td = TDConfig(
        algorithm=args.algorithm,
        episodes=args.episodes,
        seed=args.seed,
    )
    agent = TDAgent(env, cfg_td)
    print(f"Training {args.algorithm} for {args.episodes} episodes ...")
    summary = agent.train()
    print(f"  Done: {summary}")
    eval_result = agent.evaluate()
    print(f"  Eval: {eval_result}")

    out = os.path.join(args.output_dir, "q_heatmap.png") if args.output_dir else None
    plot_q_heatmap(agent.Q, env, title=f"Q3b: Q-function ({args.algorithm})", output_path=out)

    if args.output_dir:
        agent.save(os.path.join(args.output_dir, f"{args.algorithm}_trained.npz"))
    print("Q-heatmap complete.")


# ======================================================================
# Q3c: Algorithm comparison
# ======================================================================

def cmd_compare_algos(args: argparse.Namespace) -> None:
    """Compare SARSA(λ) vs Q-learning."""
    results = {}

    for algo in ["sarsa_lambda", "qlearning"]:
        print(f"Training {algo} ...")
        env = ReactorEnv(ReactorConfig())
        cfg_td = TDConfig(
            algorithm=algo,
            episodes=args.episodes,
            seed=args.seed,
        )
        agent = TDAgent(env, cfg_td)
        summary = agent.train()
        eval_result = agent.evaluate()
        print(f"  {algo}: {summary}")
        print(f"  Eval: {eval_result}")

        label = "SARSA(λ)" if algo == "sarsa_lambda" else "Q-learning"
        results[label] = {
            "episode_returns": agent.episode_returns,
            "episode_meltdowns": agent.episode_meltdowns,
        }

        if args.output_dir:
            agent.save(os.path.join(args.output_dir, f"{algo}_trained.npz"))

    out = os.path.join(args.output_dir, "algo_comparison.png") if args.output_dir else None
    plot_algorithm_comparison(results, window=args.window, output_path=out)

    # Also do λ=0 vs λ=0.8 comparison
    results_lam = {}
    for lam in [0.0, 0.8]:
        print(f"Training SARSA(λ={lam}) ...")
        env = ReactorEnv(ReactorConfig())
        cfg_td = TDConfig(
            algorithm="sarsa_lambda",
            lam=lam,
            episodes=args.episodes,
            seed=args.seed,
        )
        agent = TDAgent(env, cfg_td)
        agent.train()
        label = f"SARSA(λ={lam})"
        results_lam[label] = {
            "episode_returns": agent.episode_returns,
            "episode_meltdowns": agent.episode_meltdowns,
        }

    out_lam = os.path.join(args.output_dir, "lambda_comparison.png") if args.output_dir else None
    plot_algorithm_comparison(results_lam, window=args.window, output_path=out_lam)
    print("Algorithm comparison complete.")


# ======================================================================
# Q4a: FA vs tabular
# ======================================================================

def cmd_fa_compare(args: argparse.Namespace) -> None:
    """Compare linear FA agent vs tabular SARSA(λ)."""
    env_tab = ReactorEnv(ReactorConfig())
    env_fa = ReactorEnv(ReactorConfig())

    print("Training tabular SARSA(λ) ...")
    td_cfg = TDConfig(algorithm="sarsa_lambda", episodes=args.episodes, seed=args.seed)
    tab_agent = TDAgent(env_tab, td_cfg)
    tab_agent.train()
    tab_eval = tab_agent.evaluate()
    print(f"  Tabular eval: {tab_eval}")

    print("Training FA SARSA(λ) ...")
    fa_cfg = FAConfig(episodes=args.episodes, seed=args.seed)
    fa_agent = FAAgent(env_fa, fa_cfg)
    fa_agent.train()
    fa_eval = fa_agent.evaluate()
    print(f"  FA eval: {fa_eval}")

    out = os.path.join(args.output_dir, "fa_vs_tabular.png") if args.output_dir else None
    plot_learning_curves(
        [tab_agent.episode_returns, fa_agent.episode_returns],
        ["Tabular SARSA(λ)", "Linear FA SARSA(λ)"],
        window=args.window,
        output_path=out,
        title="Q4a: Tabular vs Linear FA",
    )

    # Also plot Q heatmaps side by side
    if args.output_dir:
        plot_q_heatmap(
            tab_agent.Q, env_tab,
            title="Tabular Q-function",
            output_path=os.path.join(args.output_dir, "q_heatmap_tabular.png"),
        )
        plot_q_heatmap(
            fa_agent.q_table(), env_fa,
            title="FA Q-function (projected to table)",
            output_path=os.path.join(args.output_dir, "q_heatmap_fa.png"),
        )
    print("FA comparison complete.")


# ======================================================================
# Q4b: Non-stationarity
# ======================================================================

def cmd_nonstationarity(args: argparse.Namespace) -> None:
    """Compare fixed LR vs decaying LR for non-stationarity analysis."""
    results = {}

    for label, alpha_decay in [("Fixed α=0.1", 0.0), ("Decaying α", 0.001)]:
        print(f"Training Q-learning ({label}) ...")
        env = ReactorEnv(ReactorConfig())
        cfg_td = TDConfig(
            algorithm="qlearning",
            alpha_decay=alpha_decay,
            episodes=args.episodes,
            seed=args.seed,
        )
        agent = TDAgent(env, cfg_td)
        agent.train()
        eval_result = agent.evaluate()
        print(f"  {label}: eval={eval_result}")
        results[label] = {
            "episode_returns": agent.episode_returns,
            "episode_meltdowns": agent.episode_meltdowns,
        }

    out = os.path.join(args.output_dir, "nonstationarity.png") if args.output_dir else None
    plot_algorithm_comparison(results, window=args.window, output_path=out)
    print("Non-stationarity analysis complete.")


# ======================================================================
# CLI
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mini 3 Problem 1: Reactor control experiments"
    )
    parser.add_argument("--output-dir", "-o", default="artifacts/reactor",
                        help="Directory for output plots/models")
    parser.add_argument("--episodes", "-e", type=int, default=2000)
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--window", "-w", type=int, default=50,
                        help="Smoothing window for learning curves")

    sub = parser.add_subparsers(dest="command", required=True)

    # noise-sweep
    sub.add_parser("noise-sweep", help="Q3a: noise level comparison")

    # q-heatmap
    p_hm = sub.add_parser("q-heatmap", help="Q3b: Q-function heatmap")
    p_hm.add_argument("--algorithm", default="sarsa_lambda",
                       choices=["sarsa_lambda", "qlearning"])

    # compare-algos
    sub.add_parser("compare-algos", help="Q3c: SARSA(λ) vs Q-learning")

    # fa-compare
    sub.add_parser("fa-compare", help="Q4a: FA vs tabular")

    # nonstationarity
    sub.add_parser("nonstationarity", help="Q4b: fixed vs decaying LR")

    # all
    sub.add_parser("all", help="Run all experiments")

    args = parser.parse_args()
    _ensure_dir(args.output_dir)

    commands = {
        "noise-sweep": cmd_noise_sweep,
        "q-heatmap": cmd_q_heatmap,
        "compare-algos": cmd_compare_algos,
        "fa-compare": cmd_fa_compare,
        "nonstationarity": cmd_nonstationarity,
    }

    if args.command == "all":
        for name, fn in commands.items():
            print(f"\n{'='*60}")
            print(f"  Running: {name}")
            print(f"{'='*60}\n")
            fn(args)
    else:
        commands[args.command](args)


if __name__ == "__main__":
    main()
