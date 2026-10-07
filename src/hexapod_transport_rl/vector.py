"""Standard Gymnasium vector environments with explicit episode resets."""

from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch

from .api import HexapodPushEnv
from .config import PushConfig


def _worker_env(kwargs: dict[str, Any], torch_threads: int | None) -> HexapodPushEnv:
    if torch_threads is not None:
        torch.set_num_threads(torch_threads)
    return HexapodPushEnv(**kwargs)


def make_vector_env(
    num_envs: int,
    config: PushConfig | dict[str, Any] | None = None,
    asset_root: str | Path | None = None,
    *,
    asynchronous: bool = False,
    torch_threads: int | None = 1,
    **kwargs,
) -> gym.vector.SyncVectorEnv | gym.vector.AsyncVectorEnv:
    """Batch *worlds*, each containing N robots, using CPU sync or spawn workers.

    Autoreset is disabled: ``step`` always returns real final observations.
    Before stepping again, call ``reset(options={"reset_mask": done})`` if any
    world ended. Returned reset observations include unchanged active worlds.
    torch_threads sets intra-op CPU threads in each executing process (including
    the caller with the synchronous backend); pass None to leave them unchanged.
    """
    if not isinstance(num_envs, int) or isinstance(num_envs, bool) or num_envs < 1:
        raise ValueError("num_envs must be a positive integer")
    if torch_threads is not None and (
        not isinstance(torch_threads, int) or torch_threads < 1
    ):
        raise ValueError("torch_threads must be a positive integer or None")
    env_kwargs = dict(config=config, asset_root=asset_root, **kwargs)
    factories = [
        partial(_worker_env, env_kwargs, torch_threads) for _ in range(num_envs)
    ]
    common = dict(autoreset_mode=gym.vector.AutoresetMode.DISABLED, copy=True)
    if asynchronous:
        if kwargs.get("render_mode") == "human":
            raise ValueError("Use render_mode=None or rgb_array for subprocess worlds")
        return gym.vector.AsyncVectorEnv(factories, context="spawn", **common)
    return gym.vector.SyncVectorEnv(factories, **common)
