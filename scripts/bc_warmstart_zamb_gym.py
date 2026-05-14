#!/usr/bin/env python3
"""Run BC on the coverage-path teacher and save initial actor weights.

Loads the precomputed teacher path from ``assets/teacher/coverage_path_v1.npz``,
rolls it out on ``ZambGymEnv``, fits the from-scratch ``GaussianActor`` to
the teacher's actions, and writes the trained actor's state_dict to
``training_runs/zamb_gym_bc_v1/actor_init.pt``.

Per-epoch metrics are logged to TensorBoard under
``<output-pt-dir>/tb/<run-tag>/``.

Thin entry-point around ``rl590.training.bc.run_bc_warmstart``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path

from rl590.training.bc import BCConfig, run_bc_warmstart


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--teacher-npz",
        type=Path,
        default=Path("assets/teacher/coverage_path_v1.npz"),
    )
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--max-episode-steps", type=int, default=400)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--ent-weight", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--output-pt",
        type=Path,
        default=Path("training_runs/zamb_gym_bc_v1/actor_init.pt"),
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--run-tag", type=str, default=None,
                   help="TB subdir name (default: timestamped).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_tag = args.run_tag or _dt.datetime.now().strftime("bc_%Y%m%d_%H%M%S")
    tb_dir = args.output_pt.parent / "tb" / run_tag

    cfg = BCConfig(
        teacher_npz=args.teacher_npz,
        episodes=args.episodes,
        max_episode_steps=args.max_episode_steps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        ent_weight=args.ent_weight,
        device=args.device,
        seed=args.seed,
        output_pt=args.output_pt,
        tb_dir=tb_dir,
    )
    run_bc_warmstart(cfg)


if __name__ == "__main__":
    main()
