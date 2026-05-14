"""Behaviour-cloning warmstart for the from-scratch PPO actor.

Rolls out a deterministic coverage-path teacher on ``ZambGymEnv``, streams
the (obs, action) transitions to disk via ``np.memmap`` so peak RSS stays
at one frame, then fits the from-scratch ``GaussianActor`` to the
teacher's actions with negative-log-likelihood loss (Gaussian mean target
with a frozen log_std). The trained actor's weights are saved to a
``.pt`` file that ``demo_zamb_gym_ppo.py --bc-init <path>`` can load.

The memmap streaming path is the load-bearing detail: it keeps peak RSS near
one frame even for image-observation datasets with tens of thousands of steps.
"""

from __future__ import annotations

import argparse
import gc
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch as th
import torch.nn as nn

from rl590.envs.teacher import CoveragePathTeacher
from rl590.envs.zamb_gym import ZambGymEnv
from rl590.networks import GaussianActor, ZamboniCNN


# Top-level orchestration


@dataclass
class BCConfig:
    teacher_npz: Path
    episodes: int = 5
    max_episode_steps: int = 400
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    ent_weight: float = 0.001
    action_clip: float = 0.95
    freeze_log_std: bool = True
    device: str = "auto"
    seed: int = 0
    output_pt: Path = Path("training_runs/zamb_gym_bc_v1/actor_init.pt")
    scratch_dir: Path | None = None
    tb_dir: Path | None = None


def _resolve_device(name: str) -> th.device:
    if name == "auto":
        return th.device("cuda" if th.cuda.is_available() else "cpu")
    return th.device(name)


def _build_actor(env: ZambGymEnv, features_dim: int = 128) -> GaussianActor:
    h, w, c = env.observation_space.shape
    cnn = ZamboniCNN(
        obs_height=h, obs_width=w, n_channels=c, features_dim=features_dim
    )
    return GaussianActor(cnn=cnn, action_dim=env.action_space.shape[0])


# Rollout phase


def _rollout_teacher(
    teacher_npz: Path,
    episodes: int,
    max_episode_steps: int,
    scratch_dir: Path,
    obs_shape: tuple[int, ...],
    action_clip: float,
    seed_base: int,
) -> tuple[np.memmap, np.ndarray]:
    """Stream teacher transitions to a memmap. Returns (obs_view, actions)."""
    scratch_dir.mkdir(parents=True, exist_ok=True)
    capacity = episodes * max_episode_steps
    obs_path = scratch_dir / "obs.f32"
    act_path = scratch_dir / "acts.f32"

    obs_mm = np.memmap(
        obs_path, dtype=np.float32, mode="w+", shape=(capacity, *obs_shape)
    )
    acts_mm = np.memmap(act_path, dtype=np.float32, mode="w+", shape=(capacity, 2))

    env = ZambGymEnv(max_steps=max_episode_steps)
    n_filled = 0
    for ep in range(episodes):
        seed = seed_base + ep
        obs, _info = env.reset(seed=seed)
        teacher = CoveragePathTeacher.from_npz(teacher_npz)

        for _ in range(max_episode_steps):
            action, _t_info = teacher.act(env.vehicle_state)
            action = np.asarray(action, dtype=np.float32).reshape(2)
            if action_clip < 1.0:
                action = np.clip(action, -action_clip, action_clip)

            obs_mm[n_filled] = np.asarray(obs, dtype=np.float32)
            acts_mm[n_filled] = action
            n_filled += 1

            obs, _r, terminated, truncated, _info = env.step(action)
            if terminated or truncated or teacher.finished:
                break

        if (ep + 1) % 5 == 0 or ep == episodes - 1:
            print(
                f"[rollout] episode {ep+1}/{episodes} transitions={n_filled}",
                flush=True,
            )

    env.close()
    obs_mm.flush()
    acts_mm.flush()
    if n_filled == 0:
        raise RuntimeError("teacher produced zero transitions")

    obs_view = np.memmap(
        obs_path, dtype=np.float32, mode="r", shape=(n_filled, *obs_shape)
    )
    acts_view = np.memmap(
        act_path, dtype=np.float32, mode="r", shape=(n_filled, 2)
    )
    return obs_view, np.asarray(acts_view)


# BC training loop


class BCWarmstart:
    """Hand-rolled BC trainer for the from-scratch ``GaussianActor``.

    Loss: ``-log N(a_teacher | mu(s), sigma) - ent_weight * H[pi(s)]``,
    with ``sigma`` frozen at its init by default so fine-tune exploration
    isn't crippled.
    """

    def __init__(
        self,
        actor: GaussianActor,
        learning_rate: float,
        ent_weight: float,
        freeze_log_std: bool,
        device: th.device,
        seed: int,
        tb_writer=None,
    ) -> None:
        self.actor = actor.to(device)
        self.device = device
        self.ent_weight = ent_weight
        self.freeze_log_std = freeze_log_std
        self._rng = np.random.default_rng(seed)
        self.tb_writer = tb_writer

        if freeze_log_std:
            self.actor.log_std.requires_grad_(False)
            trainable = [p for p in self.actor.parameters() if p.requires_grad]
        else:
            trainable = list(self.actor.parameters())
        self.optimizer = th.optim.Adam(trainable, lr=learning_rate)

    def fit(
        self,
        obs_view: np.ndarray,
        actions: np.ndarray,
        epochs: int,
        batch_size: int,
    ) -> dict:
        n = obs_view.shape[0]
        self.actor.train()
        history: list[dict] = []
        for epoch in range(epochs):
            order = self._rng.permutation(n)
            total_loss = 0.0
            total_nlp = 0.0
            total_ent = 0.0
            n_batches = 0
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                obs_batch = th.as_tensor(
                    np.asarray(obs_view[idx], dtype=np.float32), device=self.device
                )
                act_batch = th.as_tensor(
                    np.asarray(actions[idx], dtype=np.float32), device=self.device
                )
                # Use raw-Gaussian log-prob on the teacher action; the
                # actor's tanh-squashed log_prob path needs a finite
                # pre-tanh inverse and the teacher actions may saturate.
                mean, log_std = self.actor(obs_batch)
                std = log_std.exp()
                # Treat each action dim as independent Gaussian.
                nll = 0.5 * ((act_batch - mean) / std).pow(2) + log_std
                nll = nll.sum(dim=-1).mean()

                ent = self.actor.entropy(obs_batch).mean()
                loss = nll - self.ent_weight * ent

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer.step()

                total_loss += float(loss.detach().item())
                total_nlp += float(nll.detach().item())
                total_ent += float(ent.detach().item())
                n_batches += 1

            row = {
                "epoch": epoch + 1,
                "loss": total_loss / max(n_batches, 1),
                "nll": total_nlp / max(n_batches, 1),
                "entropy": total_ent / max(n_batches, 1),
            }
            history.append(row)
            if self.tb_writer is not None:
                self.tb_writer.add_scalar("bc/loss", row["loss"], row["epoch"])
                self.tb_writer.add_scalar("bc/nll", row["nll"], row["epoch"])
                self.tb_writer.add_scalar("bc/entropy", row["entropy"], row["epoch"])
            print(
                f"[bc] epoch {row['epoch']}/{epochs} "
                f"loss={row['loss']:.4f} nll={row['nll']:.4f} "
                f"entropy={row['entropy']:.4f}",
                flush=True,
            )
        self.actor.eval()
        return {"history": history}

    def save_actor(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        th.save(self.actor.state_dict(), path)


def run_bc_warmstart(cfg: BCConfig) -> BCWarmstart:
    from torch.utils.tensorboard import SummaryWriter

    device = _resolve_device(cfg.device)
    print(f"[setup] device={device}", flush=True)

    tb_writer = None
    if cfg.tb_dir is not None:
        Path(cfg.tb_dir).mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=str(cfg.tb_dir))
        print(f"[setup] tb_dir={cfg.tb_dir}", flush=True)

    probe_env = ZambGymEnv(max_steps=cfg.max_episode_steps)
    obs_shape = probe_env.observation_space.shape
    probe_env.close()
    del probe_env
    gc.collect()

    cleanup = cfg.scratch_dir is None
    scratch_dir = cfg.scratch_dir or Path(tempfile.mkdtemp(prefix="bc_warmstart_"))
    print(f"[setup] obs_shape={obs_shape} scratch_dir={scratch_dir}", flush=True)

    try:
        print(
            f"[rollout] streaming {cfg.episodes} teacher episodes "
            f"× {cfg.max_episode_steps} steps",
            flush=True,
        )
        obs_view, acts = _rollout_teacher(
            teacher_npz=cfg.teacher_npz,
            episodes=cfg.episodes,
            max_episode_steps=cfg.max_episode_steps,
            scratch_dir=scratch_dir,
            obs_shape=obs_shape,
            action_clip=cfg.action_clip,
            seed_base=cfg.seed,
        )
        n = obs_view.shape[0]
        print(f"[rollout] {n} transitions captured", flush=True)

        bc_env = ZambGymEnv(max_steps=cfg.max_episode_steps)
        actor = _build_actor(bc_env)
        bc_env.close()

        trainer = BCWarmstart(
            actor=actor,
            learning_rate=cfg.learning_rate,
            ent_weight=cfg.ent_weight,
            freeze_log_std=cfg.freeze_log_std,
            device=device,
            seed=cfg.seed,
            tb_writer=tb_writer,
        )
        trainer.fit(
            obs_view=obs_view,
            actions=acts,
            epochs=cfg.epochs,
            batch_size=min(cfg.batch_size, n),
        )
        trainer.save_actor(cfg.output_pt)
        print(f"[save] wrote {cfg.output_pt}", flush=True)
        return trainer
    finally:
        if cleanup:
            shutil.rmtree(scratch_dir, ignore_errors=True)
        if tb_writer is not None:
            tb_writer.flush()
            tb_writer.close()


# CLI


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BC warmstart for the from-scratch PPO actor.")
    p.add_argument("--teacher-npz", required=True, type=Path)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--max-episode-steps", type=int, default=400)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--ent-weight", type=float, default=0.001)
    p.add_argument("--action-clip", type=float, default=0.95)
    p.add_argument("--freeze-log-std", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-pt", type=Path,
                   default=Path("training_runs/zamb_gym_bc_v1/actor_init.pt"))
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = BCConfig(
        teacher_npz=args.teacher_npz,
        episodes=args.episodes,
        max_episode_steps=args.max_episode_steps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        ent_weight=args.ent_weight,
        action_clip=args.action_clip,
        freeze_log_std=args.freeze_log_std,
        device=args.device,
        seed=args.seed,
        output_pt=args.output_pt,
    )
    run_bc_warmstart(cfg)


if __name__ == "__main__":
    main()
