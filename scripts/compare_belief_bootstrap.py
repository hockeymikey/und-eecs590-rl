#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import pathlib
import statistics
import sys
from collections import defaultdict
from typing import Callable

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rl590.dp.planning import policy_iteration, q_value_policy_iteration, value_iteration
from rl590.envs.windy_chasm import WindyChasmMDP
from rl590.model.belief import BeliefConfig, TabularModelBelief, collect_transitions, evaluate_policy


PlannerFn = Callable[[object, object, float, float, int], tuple[object, object, object, list[float], int]]


def _parse_episode_grid(value: str) -> list[int]:
    grid = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not grid:
        raise ValueError("episode-grid cannot be empty")
    if any(v <= 0 for v in grid):
        raise ValueError("episode-grid must contain positive integers")
    return grid


def _planner_for_algorithm(name: str) -> PlannerFn:
    if name == "value_iteration":
        return value_iteration
    if name == "policy_iteration":
        return policy_iteration
    if name == "q_policy_iteration":
        return q_value_policy_iteration
    raise ValueError(f"Unsupported algorithm: {name}")


def _run_single_trial(
    bootstrap_episodes: int,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, float | int | str]:
    env = WindyChasmMDP(
        p_center=args.p_center,
        goal_col=args.goal_col,
        crash_penalty=args.crash_penalty,
        step_reward=args.step_reward,
        goal_bonus=args.goal_bonus,
        gamma=args.gamma,
    )

    belief = TabularModelBelief(
        num_states=env.num_states,
        num_actions=env.num_actions,
        config=BeliefConfig(
            transition_prior=args.transition_prior,
            reward_prior_mean=args.reward_prior_mean,
            reward_prior_count=args.reward_prior_count,
        ),
    )

    transitions = collect_transitions(
        env,
        episodes=bootstrap_episodes,
        max_steps_per_episode=args.bootstrap_max_steps,
        epsilon=args.exploration_epsilon,
        seed=seed,
        policy=None,
    )
    belief.update_batch(transitions)
    P_hat, R_hat = belief.estimated_mdp()

    planner = _planner_for_algorithm(args.algorithm)
    _, _, policy, deltas, planner_iterations = planner(
        P_hat,
        R_hat,
        env.gamma,
        args.epsilon,
        args.max_iterations,
    )

    eval_stats = evaluate_policy(
        env,
        policy=policy,
        episodes=args.eval_episodes,
        max_steps_per_episode=args.eval_max_steps,
        seed=seed + 10_000,
    )

    unique_sa = len({(s, a) for s, a, _, _ in transitions})
    unique_sas = len({(s, a, s_next) for s, a, s_next, _ in transitions})

    return {
        "algorithm": args.algorithm,
        "bootstrap_episodes": bootstrap_episodes,
        "seed": seed,
        "observed_transitions": len(transitions),
        "belief_updates": belief.num_updates,
        "unique_sa": unique_sa,
        "unique_sas": unique_sas,
        "planner_iterations": planner_iterations,
        "planner_final_delta": deltas[-1] if deltas else 0.0,
        "eval_success_rate": eval_stats["success_rate"],
        "eval_mean_return": eval_stats["mean_return"],
        "eval_std_return": eval_stats["std_return"],
    }


def _write_csv(rows: list[dict[str, float | int | str]], output_path: pathlib.Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "algorithm",
        "bootstrap_episodes",
        "seed",
        "observed_transitions",
        "belief_updates",
        "unique_sa",
        "unique_sas",
        "planner_iterations",
        "planner_final_delta",
        "eval_success_rate",
        "eval_mean_return",
        "eval_std_return",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(rows: list[dict[str, float | int | str]]) -> None:
    grouped: dict[int, list[dict[str, float | int | str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["bootstrap_episodes"])] .append(row)

    print("\nBootstrap Sweep Summary")
    print("episodes | success(mean+-std) | return(mean+-std) | transitions(mean)")
    print("-" * 75)

    for episodes in sorted(grouped):
        batch = grouped[episodes]
        success_values = [float(r["eval_success_rate"]) for r in batch]
        return_values = [float(r["eval_mean_return"]) for r in batch]
        transition_values = [float(r["observed_transitions"]) for r in batch]

        success_mean = statistics.fmean(success_values)
        success_std = statistics.pstdev(success_values) if len(success_values) > 1 else 0.0

        return_mean = statistics.fmean(return_values)
        return_std = statistics.pstdev(return_values) if len(return_values) > 1 else 0.0

        transition_mean = statistics.fmean(transition_values)

        print(
            f"{episodes:8d} | "
            f"{success_mean:6.3f} +- {success_std:6.3f} | "
            f"{return_mean:8.2f} +- {return_std:8.2f} | "
            f"{transition_mean:10.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare model-belief bootstrap quality across episode budgets and save CSV results."
    )
    parser.add_argument("--algorithm", default="policy_iteration", choices=["value_iteration", "policy_iteration", "q_policy_iteration"])
    parser.add_argument("--episode-grid", default="20,50,100,200,500,1000", help="Comma-separated bootstrap episode counts")
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--output", default="artifacts/bootstrap_sweep.csv")

    parser.add_argument("--bootstrap-max-steps", type=int, default=200)
    parser.add_argument("--exploration-epsilon", type=float, default=1.0)
    parser.add_argument("--transition-prior", type=float, default=1e-3)
    parser.add_argument("--reward-prior-mean", type=float, default=0.0)
    parser.add_argument("--reward-prior-count", type=float, default=0.0)

    parser.add_argument("--eval-episodes", type=int, default=40)
    parser.add_argument("--eval-max-steps", type=int, default=200)

    parser.add_argument("--p-center", type=float, default=0.1)
    parser.add_argument("--goal-col", type=int, default=3)
    parser.add_argument("--crash-penalty", type=float, default=-100.0)
    parser.add_argument("--step-reward", type=float, default=-1.0)
    parser.add_argument("--goal-bonus", type=float, default=100.0)
    parser.add_argument("--gamma", type=float, default=0.9)

    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--max-iterations", type=int, default=10_000)

    args = parser.parse_args()

    if args.num_seeds <= 0:
        raise ValueError("num-seeds must be positive")

    grid = _parse_episode_grid(args.episode_grid)

    rows: list[dict[str, float | int | str]] = []
    seed_values = [args.seed_start + i for i in range(args.num_seeds)]

    print("Running bootstrap sweep...")
    print(f"algorithm={args.algorithm}, episodes={grid}, seeds={seed_values}")

    for episode_budget in grid:
        for seed in seed_values:
            row = _run_single_trial(
                bootstrap_episodes=episode_budget,
                seed=seed,
                args=args,
            )
            rows.append(row)
            print(
                "trial "
                f"episodes={episode_budget}, seed={seed}, "
                f"success={float(row['eval_success_rate']):.3f}, "
                f"return={float(row['eval_mean_return']):.2f}"
            )

    output_path = pathlib.Path(args.output)
    _write_csv(rows, output_path)
    _print_summary(rows)

    print(f"\nSaved CSV: {output_path}")


if __name__ == "__main__":
    main()
