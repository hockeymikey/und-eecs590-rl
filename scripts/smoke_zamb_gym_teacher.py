#!/usr/bin/env python3
"""Smoke-test the coverage-path teacher on ``ZambGymEnv``.

Rolls out the precomputed teacher for ``--steps`` env steps starting from a
random pose; reports return, waypoint progress, and any early termination.
Does not save anything.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rl590.envs.teacher import CoveragePathTeacher
from rl590.envs.zamb_gym import ZambGymEnv


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-npz", type=Path,
                   default=Path("assets/teacher/coverage_path_v1.npz"))
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    teacher = CoveragePathTeacher.from_npz(args.teacher_npz)
    env = ZambGymEnv(max_steps=args.steps)
    obs, _ = env.reset(seed=args.seed)
    print(
        f"agent start pose: x={env.x:.2f} y={env.y:.2f} theta={env.theta:.2f}"
    )
    print(f"path: {len(teacher.waypoints)} waypoints")

    ret = 0.0
    idx_start = None
    last_step = 0
    for s in range(args.steps):
        action, t_info = teacher.act(env.vehicle_state)
        if idx_start is None:
            idx_start = t_info["waypoint_idx"]
        obs, r, term, trunc, _info = env.step(action)
        ret += r
        last_step = s
        if term or trunc or teacher.finished:
            print(f"stopped at step {s}: term={term} trunc={trunc} "
                  f"teacher_finished={teacher.finished}")
            break

    print(
        f"teacher idx: {idx_start} -> {teacher._idx} / {len(teacher.waypoints)}"
    )
    print(f"return over {last_step+1} steps: {ret:+.2f}")


if __name__ == "__main__":
    main()
