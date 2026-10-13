"""Held-out crossing episodes and a movie of the same learned policy."""

import json
import math
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type

from .navigation import NavigationConfig, NavigationEnv
from .torchrl_mappo import load_mappo, policy_action


def _native_episode(actor, cfg, seed, video_path=None):
    env = NavigationEnv(cfg, render_mode="rgb_array_list" if video_path else None)
    writer = imageio.get_writer(str(video_path), fps=25) if video_path else None
    try:
        observation, info = env.reset(seed=seed)
        initial_distance = info["distance"]
        minimum = info["min_separation"]
        contact_seconds = 0.0
        positions = [info["robot_positions"].tolist()]
        if writer:
            for frame in env.render():
                writer.append_data(frame)
        for _ in range(math.ceil(cfg.episode_seconds / cfg.dt)):
            observation, _, terminated, truncated, info = env.step(
                policy_action(actor, observation)
            )
            minimum = min(minimum, info["min_separation"])
            contact_seconds += info["robot_collision_fraction"] * cfg.dt
            positions.append(info["robot_positions"].tolist())
            if writer:
                for frame in env.render():
                    writer.append_data(frame)
            if terminated or truncated:
                break
        return dict(
            seed=seed,
            is_success=info["is_success"],
            failed=info["failed"],
            initial_distance_m=initial_distance,
            final_distance_m=info["distance"],
            min_separation_m=minimum,
            contact_seconds=contact_seconds,
            episode_seconds=env.steps * cfg.dt,
            goals=info["goals"].tolist(),
            positions=positions,
        )
    finally:
        if writer:
            writer.close()
        env.close()


def _warp_episodes(actor, cfg, episodes, seed):
    from .warp_navigation import WarpNavigationWorlds

    worlds = WarpNavigationWorlds(cfg, episodes)
    device = worlds.device
    try:
        # Match the exact native seed states, rather than comparing two RNGs.
        qpos, qvel, goals, initial = [], [], [], []
        for index in range(episodes):
            _, info = worlds.reference.reset(seed=seed + index)
            qpos.append(worlds.reference.data.qpos.copy())
            qvel.append(worlds.reference.data.qvel.copy())
            goals.append(worlds.reference.goals.copy())
            initial.append(info["distance"])
        worlds.physics.qpos.copy_(torch.as_tensor(np.array(qpos), device=device))
        worlds.physics.qvel.copy_(torch.as_tensor(np.array(qvel), device=device))
        worlds.goals.copy_(torch.as_tensor(np.array(goals), device=device))
        worlds.physics.forward()
        active = torch.ones(episodes, dtype=torch.bool, device=device)
        minimum = worlds.measurements()[4].min(-1).values.clone()
        contacts = torch.zeros(episodes, device=device)
        rows = [None] * episodes
        actor = actor.to(device)
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            for step in range(math.ceil(cfg.episode_seconds / cfg.dt)):
                td = TensorDict(
                    {
                        "agents": TensorDict(
                            {"observation": worlds.observation()},
                            batch_size=[episodes, cfg.num_robots],
                        )
                    },
                    batch_size=[episodes],
                    device=device,
                )
                action = actor(td)["agents", "action"]
                _, _, terminated, truncated, info = worlds.step(action)
                minimum = torch.where(
                    active, torch.minimum(minimum, info["min_separation"]), minimum
                )
                contacts += active * info["robot_collision_fraction"] * cfg.dt
                ended = active & (terminated | truncated)
                for index in ended.nonzero().flatten().tolist():
                    rows[index] = dict(
                        seed=seed + index,
                        is_success=bool(info["is_success"][index]),
                        failed=bool(info["failed"][index]),
                        initial_distance_m=initial[index],
                        final_distance_m=float(info["distance"][index]),
                        min_separation_m=float(minimum[index]),
                        contact_seconds=float(contacts[index]),
                        episode_seconds=(step + 1) * cfg.dt,
                    )
                active &= ~ended
                if not bool(active.any()):
                    break
        worlds.physics.check()
        return rows
    finally:
        worlds.close()


def evaluate_navigation(
    checkpoint, *, episodes=8, seed=98000, video=True, backend="auto"
):
    if episodes < 1 or backend not in ("auto", "cpu", "warp"):
        raise ValueError("Use a positive episode count and cpu, warp or auto backend")
    started = time.monotonic()
    checkpoint = Path(checkpoint)
    actor, _, saved = load_mappo(checkpoint)
    cfg = NavigationConfig.from_checkpoint(saved["navigation_config"])
    if backend == "auto":
        backend = "warp" if torch.cuda.is_available() else "cpu"
    rows = (
        _warp_episodes(actor, cfg, episodes, seed)
        if backend == "warp"
        else [_native_episode(actor, cfg, seed + index) for index in range(episodes)]
    )
    report = dict(
        task="navigation",
        backend=backend,
        episodes=rows,
        successes=sum(row["is_success"] for row in rows),
        success_rate=sum(row["is_success"] for row in rows) / episodes,
        collision_free_successes=sum(
            row["is_success"] and row["contact_seconds"] == 0 for row in rows
        ),
        mean_final_distance_m=float(np.mean([row["final_distance_m"] for row in rows])),
        mean_contact_seconds=float(np.mean([row["contact_seconds"] for row in rows])),
    )
    if video:
        video_path = checkpoint.parent / "navigation.mp4"
        video_row = _native_episode(actor.cpu(), cfg, seed, video_path)
        report.update(
            video_path=str(video_path.resolve()),
            video_backend="cpu",
            video_success=video_row["is_success"],
            video_episode=video_row,
        )
    report["evaluation_and_video_seconds"] = time.monotonic() - started
    (checkpoint.parent / "evaluation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report
