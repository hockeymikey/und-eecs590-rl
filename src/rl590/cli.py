from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rl590.agents.planning_agent import AgentConfig, PlanningAgent
from rl590.envs.windy_chasm import WindyChasmMDP
from rl590.dp.planning import policy_iteration, q_value_policy_iteration, value_iteration
from rl590.model.belief import BeliefConfig, TabularModelBelief, collect_transitions, evaluate_policy
from rl590.utils.plotting import plot_convergence
from rl590.utils.render import render_policy_ascii, render_values_ascii


def _common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--algorithm", default="policy_iteration", choices=["value_iteration", "policy_iteration", "q_policy_iteration"])
    parser.add_argument("--p-center", type=float, default=0.1)
    parser.add_argument("--goal-col", type=int, default=3)
    parser.add_argument("--crash-penalty", type=float, default=-100.0)
    parser.add_argument("--step-reward", type=float, default=-1.0)
    parser.add_argument("--goal-bonus", type=float, default=100.0)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--max-iterations", type=int, default=10_000)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)


def _build_env(args: argparse.Namespace) -> WindyChasmMDP:
    return WindyChasmMDP(
        p_center=args.p_center,
        goal_col=args.goal_col,
        crash_penalty=args.crash_penalty,
        step_reward=args.step_reward,
        goal_bonus=args.goal_bonus,
        gamma=args.gamma,
    )


def _build_agent(env: WindyChasmMDP, args: argparse.Namespace) -> PlanningAgent:
    cfg = AgentConfig(
        algorithm=args.algorithm,
        epsilon=args.epsilon,
        max_iterations=args.max_iterations,
        episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        seed=args.seed,
    )
    return PlanningAgent(env, cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate tabular planning agents for Windy Chasm.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a planning agent and save artifacts.")
    _common_args(train_parser)
    train_parser.add_argument("--model-path", default="artifacts/windy_best_policy.npz")
    train_parser.add_argument("--plot-path", default="artifacts/convergence.png")
    train_parser.add_argument("--no-plot", action="store_true")
    train_parser.add_argument("--no-render", action="store_true")

    eval_parser = subparsers.add_parser("eval", help="Evaluate an existing policy/value artifact.")
    _common_args(eval_parser)
    eval_parser.add_argument("--model-path", default="artifacts/windy_best_policy.npz")
    eval_parser.add_argument("--render", action="store_true")

    belief_parser = subparsers.add_parser(
        "bootstrap-model",
        help="Learn a model belief from sampled transitions, then plan on the learned model.",
    )
    _common_args(belief_parser)
    belief_parser.add_argument("--bootstrap-episodes", type=int, default=200)
    belief_parser.add_argument("--bootstrap-max-steps", type=int, default=200)
    belief_parser.add_argument("--exploration-epsilon", type=float, default=1.0)
    belief_parser.add_argument("--transition-prior", type=float, default=1e-3)
    belief_parser.add_argument("--reward-prior-mean", type=float, default=0.0)
    belief_parser.add_argument("--reward-prior-count", type=float, default=0.0)
    belief_parser.add_argument("--belief-path", default="artifacts/windy_belief.npz")
    belief_parser.add_argument("--model-path", default="artifacts/windy_from_belief_policy.npz")
    belief_parser.add_argument("--render", action="store_true")

    args = parser.parse_args()
    env = _build_env(args)
    agent = _build_agent(env, args)

    if args.command == "train":
        train_stats = agent.train()
        eval_stats = agent.evaluate()

        model_path = Path(args.model_path)
        saved = agent.save(model_path)

        print("Train Stats:")
        for key, value in train_stats.items():
            print(f"  {key}: {value}")

        print("Eval Stats:")
        for key, value in eval_stats.items():
            print(f"  {key}: {value}")

        print(f"Saved model artifact: {saved}")

        if not args.no_render:
            print()
            print(render_policy_ascii(env, agent.policy))
            print()
            print(render_values_ascii(env, agent.V))

        if not args.no_plot:
            try:
                plot_convergence(agent.deltas, output_path=args.plot_path)
                print(f"Saved convergence plot: {args.plot_path}")
            except RuntimeError as exc:
                print(f"Plot skipped: {exc}")

    elif args.command == "eval":
        model_path = Path(args.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {model_path}. Run train first."
            )

        agent.load(model_path)
        eval_stats = agent.evaluate()

        print(f"Loaded model artifact: {model_path}")
        print("Eval Stats:")
        for key, value in eval_stats.items():
            print(f"  {key}: {value}")

        if args.render:
            print()
            print(render_policy_ascii(env, agent.policy))

    elif args.command == "bootstrap-model":
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
            episodes=args.bootstrap_episodes,
            max_steps_per_episode=args.bootstrap_max_steps,
            epsilon=args.exploration_epsilon,
            seed=args.seed,
            policy=None,
        )
        belief.update_batch(transitions)
        belief_path = belief.save(args.belief_path)

        P_hat, R_hat = belief.estimated_mdp()

        if args.algorithm == "value_iteration":
            V, Q, policy, deltas, iterations = value_iteration(
                P_hat,
                R_hat,
                env.gamma,
                epsilon=args.epsilon,
                max_iterations=args.max_iterations,
            )
        elif args.algorithm == "policy_iteration":
            V, Q, policy, deltas, iterations = policy_iteration(
                P_hat,
                R_hat,
                env.gamma,
                epsilon=args.epsilon,
                max_iterations=args.max_iterations,
            )
        elif args.algorithm == "q_policy_iteration":
            V, Q, policy, deltas, iterations = q_value_policy_iteration(
                P_hat,
                R_hat,
                env.gamma,
                epsilon=args.epsilon,
                max_iterations=args.max_iterations,
            )
        else:
            raise ValueError(f"Unsupported algorithm: {args.algorithm}")

        np.savez(
            args.model_path,
            policy=policy,
            V=V,
            Q=Q,
            metadata=np.array([f"learned_from_belief_updates={belief.num_updates}"]),
        )

        eval_stats = evaluate_policy(
            env,
            policy=policy,
            episodes=args.episodes,
            max_steps_per_episode=args.max_steps,
            seed=args.seed,
        )

        print("Belief Bootstrap Stats:")
        print(f"  observed_transitions: {len(transitions)}")
        print(f"  belief_updates: {belief.num_updates}")
        print(f"  planner_iterations: {iterations}")
        print(f"  final_delta: {deltas[-1] if deltas else 0.0}")
        print(f"  saved_belief: {belief_path}")
        print(f"  saved_model: {args.model_path}")

        print("Eval Stats (policy planned on learned model, executed on true env):")
        for key, value in eval_stats.items():
            print(f"  {key}: {value}")

        if args.render:
            print()
            print(render_policy_ascii(env, policy))


if __name__ == "__main__":
    main()
