from enum import StrEnum


class NpzKey(StrEnum):
    """Field names used in `.npz` artifacts (saved policies, Q-tables, replay data, beliefs)."""

    # Planning / control artifacts
    POLICY = "policy"
    V = "V"
    Q = "Q"
    METADATA = "metadata"

    # Episode logs
    EPISODE_RETURNS = "episode_returns"
    EPISODE_LENGTHS = "episode_lengths"

    # Monte Carlo bookkeeping
    RETURNS_SUM = "returns_sum"
    RETURNS_COUNT = "returns_count"

    # Replay-buffer transitions
    OBSERVATIONS = "observations"
    ACTIONS = "actions"
    REWARDS = "rewards"
    NEXT_OBSERVATIONS = "next_observations"
    DONES = "dones"

    # Tabular model belief
    TRANSITION_COUNTS = "transition_counts"
    REWARD_COUNTS = "reward_counts"
    REWARD_SUMS = "reward_sums"
    NUM_UPDATES = "num_updates"


class BatchKey(StrEnum):
    """Field names used in in-memory batch dicts (replay sampling, PPO minibatches)."""

    # Replay-buffer sample dict (note: shorter aliases than the on-disk npz keys)
    OBS = "obs"
    ACTIONS = "actions"
    REWARDS = "rewards"
    NEXT_OBS = "next_obs"
    DONES = "dones"

    # PPO minibatch
    OLD_LOG_PROBS = "old_log_probs"
    ADVANTAGES = "advantages"
    RETURNS = "returns"