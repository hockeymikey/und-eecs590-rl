#!/usr/bin/env python3
"""Demo: Replay buffer usage and management.

Shows how the replay buffer stores transitions, samples batches,
saves/loads from disk, and how the management script works.
"""
import numpy as np

from rl590.buffers import ReplayBuffer, ReplayBufferConfig


def main():
    print("=" * 60)
    print("Replay Buffer Demo")
    print("=" * 60)
    print()

    # Create a buffer with small observations for demo
    cfg = ReplayBufferConfig(max_size=500, obs_shape=(8, 16, 2), action_dim=2)
    buf = ReplayBuffer(cfg)

    print(f"Buffer capacity: {cfg.max_size}")
    print(f"Observation shape: {cfg.obs_shape}")
    print(f"Action dim: {cfg.action_dim}")
    print()

    # Fill with synthetic transitions
    rng = np.random.default_rng(42)
    print("Filling buffer with 750 transitions (overflows by 250)...")
    for i in range(750):
        obs = rng.standard_normal(cfg.obs_shape).astype(np.float32)
        action = rng.standard_normal(cfg.action_dim).astype(np.float32)
        reward = float(rng.standard_normal())
        next_obs = rng.standard_normal(cfg.obs_shape).astype(np.float32)
        done = bool(rng.random() > 0.95)
        buf.add(obs, action, reward, next_obs, done)

    stats = buf.stats()
    print(f"\nBuffer stats:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")

    # Sample a batch
    print(f"\nSampling a minibatch of 32...")
    batch = buf.sample(32, rng)
    for k, v in batch.items():
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

    # Save to disk
    save_path = "replay_data/demo/test"
    print(f"\nSaving buffer to {save_path}.npz...")
    buf.save(save_path, metadata={"algorithm": "demo", "task": "test", "policy_version": "v1"})

    # Load into new buffer
    buf2 = ReplayBuffer(cfg)
    meta = buf2.load(save_path + ".npz")
    print(f"Loaded buffer: {len(buf2)} transitions")
    print(f"Metadata: {meta}")

    # Verify data integrity
    b1 = buf.sample(10, np.random.default_rng(0))
    b2 = buf2.sample(10, np.random.default_rng(0))
    match = np.allclose(b1["obs"], b2["obs"])
    print(f"Data integrity check: {'PASS' if match else 'FAIL'}")

    # Cleanup
    import os
    os.remove(save_path + ".npz")
    os.rmdir("replay_data/demo")
    print("\nCleaned up demo files.")

    # Show management script usage
    print()
    print("-" * 60)
    print("Replay management script usage:")
    print("  python scripts/manage_replay.py status")
    print("  python scripts/manage_replay.py prune --max-size-mb 500")
    print("  python scripts/manage_replay.py prune --keep-newest 5 --dry-run")


if __name__ == "__main__":
    main()
