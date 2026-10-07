"""Reward-only adaptation of the existing pusher to approach handover poses.

The navigator is not replayed for every rollout: randomized rear reset states
cover its handover region. No demonstration actions or supervised targets exist.
"""

import hashlib
import os
import time
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import torch

from .api import HexapodPushEnv
from .config import STAND_HEIGHT, PushConfig, rotation
from .mappo import load_checkpoint, train_rollouts
from .rewards import PushRewardWeights
from .types import Info, ResetResult

RESET_DISTRIBUTION = dict(
    name="approach_handover_v1",
    original_reset_fraction=0.25,
    cargo_yaw_degrees=30,
    rear_x_m=-1.05,
    rear_x_jitter_m=0.10,
    slot_jitter_m=0.08,
    robot_yaw_jitter_rad=0.15,
)


class HandoverPushEnv(HexapodPushEnv):
    """元の押す環境に、回り込み直後を想定した初期配置を加える。"""

    @property
    def provenance(self) -> Info:
        return {**super().provenance, "reset_distribution": dict(RESET_DISTRIBUTION)}

    def reset(self, *, seed=None, options=None) -> ResetResult:
        """元の配置を25%残し、残りを引き継ぎ位置付近に初期化する。"""
        obs, info = super().reset(seed=seed, options=options)
        env = self.core
        if env.rng.random() < RESET_DISTRIBUTION["original_reset_fraction"]:
            return obs, info
        yaw = env.goal_yaw + env.rng.uniform(-np.pi / 6, np.pi / 6)
        env.data.qpos[env.cargo_q + 3 : env.cargo_q + 7] = [
            np.cos(yaw / 2),
            0,
            0,
            np.sin(yaw / 2),
        ]
        for i, q in enumerate(env.root_q):
            xy = rotation(yaw) @ [
                env.rng.uniform(-1.15, -0.95),
                env.cfg.slots[i] + env.rng.uniform(-0.08, 0.08),
            ]
            heading = yaw + env.rng.uniform(-0.15, 0.15)
            env.data.qpos[q : q + 7] = [
                *xy,
                STAND_HEIGHT,
                np.cos(heading / 2),
                0,
                0,
                np.sin(heading / 2),
            ]
        mujoco.mj_forward(env.model, env.data)
        return self._format_observation(env.observe()), self._episode_info(env.info())


def worker(cfg: PushConfig, asset_root: str | Path | None) -> HandoverPushEnv:
    torch.set_num_threads(1)
    return HandoverPushEnv(cfg, asset_root, flatten=False)


def compose(
    navigation_checkpoint: str | Path,
    pushing_checkpoint: str | Path,
    output: str | Path,
) -> Path:
    """Bind learned actors after evaluation; preserve the navigation weights."""
    from .approach_training import load_approach_checkpoint

    _, _, nav_saved = load_approach_checkpoint(navigation_checkpoint)
    _, push_saved = load_checkpoint(pushing_checkpoint)
    _, original_push = load_checkpoint(nav_saved["pushing_checkpoint"])
    original_config = asdict(PushConfig.from_checkpoint(original_push["config"]))
    new_config = asdict(PushConfig.from_checkpoint(push_saved["config"]))
    original_config.pop("reward_weights")
    new_config.pop("reward_weights")
    if original_config != new_config:
        raise ValueError("The two pushers must share the same physical configuration")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    pushing_checkpoint = Path(pushing_checkpoint).resolve()
    # Resolve relative to the composed checkpoint, so the run can be moved.
    nav_saved["pushing_checkpoint"] = os.path.relpath(
        pushing_checkpoint, output.resolve().parent
    )
    nav_saved["pushing_sha256"] = hashlib.sha256(
        pushing_checkpoint.read_bytes()
    ).hexdigest()
    nav_saved["composition"] = dict(
        navigation_source=str(Path(navigation_checkpoint).resolve()),
        pushing_source=str(pushing_checkpoint),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(nav_saved, output)
    return output


def train_handover(
    checkpoint: str | Path,
    output: str | Path,
    iterations: int = 150,
    num_envs: int = 16,
    horizon: int = 64,
    seed: int = 20261009,
    asset_root: str | Path | None = None,
    *,
    reward_weights: PushRewardWeights | None = None,
) -> Path:
    """Continue MAPPO from a pusher, sampling the navigator's handover region."""
    if min(iterations, num_envs, horizon) < 1 or num_envs * horizon < 2:
        raise ValueError("Positive sizes and at least two transitions required")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    _, saved = load_checkpoint(checkpoint)
    cfg = PushConfig.from_checkpoint(saved["config"])
    if reward_weights is not None:
        cfg = replace(cfg, reward_weights=reward_weights)
    if cfg.num_robots != 2 or cfg.shape != "T":
        raise ValueError("Expected a two-robot T pusher")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    factories = [partial(worker, cfg, asset_root) for _ in range(num_envs)]
    envs = gym.vector.AsyncVectorEnv(
        factories,
        context="spawn",
        copy=True,
        autoreset_mode=gym.vector.AutoresetMode.DISABLED,
    )
    try:
        return train_rollouts(
            envs=envs,
            cfg=cfg,
            output=output,
            iterations=iterations,
            num_envs=num_envs,
            horizon=horizon,
            epochs=4,
            minibatch=256,
            seed=seed,
            resume=checkpoint,
            started_at=start,
            allow_reward_change=reward_weights is not None,
        )
    finally:
        envs.close()
