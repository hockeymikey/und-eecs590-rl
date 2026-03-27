#!/usr/bin/env python3
"""Replay data management script.

Scans the replay_data/ directory, reports storage usage per algorithm
and task, and can prune old experience to stay within a budget.

Usage:
    # Report current storage usage
    python scripts/manage_replay.py status

    # Prune oldest files to stay under 500MB total
    python scripts/manage_replay.py prune --max-size-mb 500

    # Prune a specific algorithm's data to keep only the newest N files
    python scripts/manage_replay.py prune --algorithm ppo --keep-newest 5

    # Dry run (show what would be deleted without deleting)
    python scripts/manage_replay.py prune --max-size-mb 500 --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPLAY_DIR = Path(__file__).resolve().parent.parent / "replay_data"


def scan_replay_dir(base: Path) -> list[dict]:
    """Scan replay_data/ and return info about each .npz file."""
    files = []
    for npz_path in sorted(base.rglob("*.npz")):
        rel = npz_path.relative_to(base)
        parts = rel.parts  # e.g. ("ppo", "zamboni", "20260326_v1.npz")
        algorithm = parts[0] if len(parts) > 1 else "unknown"
        task = parts[1] if len(parts) > 2 else "unknown"
        size_mb = npz_path.stat().st_size / (1024 * 1024)
        mtime = npz_path.stat().st_mtime

        files.append({
            "path": npz_path,
            "relative": str(rel),
            "algorithm": algorithm,
            "task": task,
            "size_mb": size_mb,
            "mtime": mtime,
        })
    return files


def cmd_status(args: argparse.Namespace) -> None:
    """Print storage usage report."""
    files = scan_replay_dir(REPLAY_DIR)

    if not files:
        print(f"No replay data found in {REPLAY_DIR}")
        return

    total_mb = sum(f["size_mb"] for f in files)

    # Group by algorithm
    by_algo: dict[str, list[dict]] = {}
    for f in files:
        by_algo.setdefault(f["algorithm"], []).append(f)

    print(f"Replay data directory: {REPLAY_DIR}")
    print(f"Total files: {len(files)}")
    print(f"Total size: {total_mb:.1f} MB")
    print()

    for algo, algo_files in sorted(by_algo.items()):
        algo_mb = sum(f["size_mb"] for f in algo_files)
        print(f"  {algo}/")
        # Group by task
        by_task: dict[str, list[dict]] = {}
        for f in algo_files:
            by_task.setdefault(f["task"], []).append(f)
        for task, task_files in sorted(by_task.items()):
            task_mb = sum(f["size_mb"] for f in task_files)
            print(f"    {task}/  ({len(task_files)} files, {task_mb:.1f} MB)")
        print(f"    subtotal: {algo_mb:.1f} MB")
        print()


def cmd_prune(args: argparse.Namespace) -> None:
    """Delete oldest replay files to stay within budget."""
    files = scan_replay_dir(REPLAY_DIR)

    if not files:
        print("No replay data to prune.")
        return

    # Filter by algorithm if specified
    if args.algorithm:
        files = [f for f in files if f["algorithm"] == args.algorithm]
        if not files:
            print(f"No replay data for algorithm '{args.algorithm}'")
            return

    # Sort by modification time (oldest first)
    files.sort(key=lambda f: f["mtime"])

    to_delete = []

    if args.keep_newest:
        # Keep only the N newest files
        if len(files) > args.keep_newest:
            to_delete = files[: len(files) - args.keep_newest]
    elif args.max_size_mb:
        # Delete oldest until total is under budget
        total_mb = sum(f["size_mb"] for f in files)
        for f in files:
            if total_mb <= args.max_size_mb:
                break
            to_delete.append(f)
            total_mb -= f["size_mb"]

    if not to_delete:
        print("Nothing to prune — within budget.")
        return

    freed_mb = sum(f["size_mb"] for f in to_delete)
    print(f"{'Would delete' if args.dry_run else 'Deleting'} "
          f"{len(to_delete)} files ({freed_mb:.1f} MB)")

    for f in to_delete:
        print(f"  {'[DRY RUN] ' if args.dry_run else ''}rm {f['relative']} ({f['size_mb']:.1f} MB)")
        if not args.dry_run:
            f["path"].unlink()

    if not args.dry_run:
        # Clean up empty directories
        for dirpath, dirnames, filenames in os.walk(REPLAY_DIR, topdown=False):
            p = Path(dirpath)
            if p != REPLAY_DIR and not any(p.iterdir()):
                p.rmdir()


def main():
    parser = argparse.ArgumentParser(description="Manage replay buffer storage")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Report storage usage")

    prune_p = sub.add_parser("prune", help="Delete old replay data")
    prune_p.add_argument("--max-size-mb", type=float, help="Max total size in MB")
    prune_p.add_argument("--keep-newest", type=int, help="Keep only N newest files")
    prune_p.add_argument("--algorithm", type=str, help="Only prune this algorithm")
    prune_p.add_argument("--dry-run", action="store_true", help="Show what would be deleted")

    args = parser.parse_args()
    if args.command == "status":
        cmd_status(args)
    elif args.command == "prune":
        cmd_prune(args)


if __name__ == "__main__":
    main()
