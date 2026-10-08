"""Evaluate learned pose pushing and two independent rule-based baselines.

These controllers are used only for evaluation. They never initialize a learned
policy or contribute actions, trajectories or labels to MAPPO training.
"""

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import partial
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

from .config import command_to_action, rotation, wrap
from .pose_push import PosePushConfig, PosePushEnv
from .torchrl_mappo import POSE_FORMAT, load_mappo, policy_action


def rule_action(core, policy="feedback"):
    """A forward-only baseline or proportional pose feedback, in API coordinates."""
    if policy == "forward":
        vx = 0.0 if core.info()["distance"] < core.cfg.position_tolerance else 0.12
        return np.tile([vx / 0.20, 0, 0], (core.cfg.num_robots, 1)).astype(np.float32)
    if policy != "feedback":
        raise ValueError("Rule baseline must be forward or feedback")
    cargo_rotation = rotation(core.cargo_yaw)
    error = (core.goal - core.cargo_xy) @ cargo_rotation
    angle = wrap(core.goal_yaw - core.cargo_yaw)
    velocity = np.clip(error * 0.8, [-0.08, -0.08], [0.15, 0.08])
    angular = float(np.clip(angle * 1.0, -0.35, 0.35))
    relative = (core.robot_xy - core.cargo_xy) @ cargo_rotation
    headings = wrap(core.yaw(core.base_ids) - core.cargo_yaw)
    commands = []
    for i, slot in enumerate(core.cfg.slots):
        formation_error = (
            np.array([core.cfg.rear_face - core.cfg.push_gap, slot]) - relative[i]
        )
        local_velocity = velocity + [-angular * slot, angular * relative[i, 0]]
        local_velocity += 0.4 * formation_error
        body_velocity = local_velocity @ rotation(headings[i])
        commands.append([*body_velocity, angular - 1.5 * headings[i]])
    actions = command_to_action(commands)
    actions[:, 1:] *= np.where(core.cfg.slots >= 0, 1, -1)[:, None]
    return actions.astype(np.float32)


def _run_batch(config, checkpoint, policy, seeds, video_dir, asset_root, width, height):
    torch.set_num_threads(1)
    actor = None
    if policy == "learned":
        actor, _, saved = load_mappo(checkpoint)
        if saved["format"] != POSE_FORMAT:
            raise ValueError("Pose evaluation requires a learned pose checkpoint")
    env = PosePushEnv(
        config,
        asset_root,
        render_mode="rgb_array" if video_dir else None,
        width=width,
        height=height,
    )
    if actor is not None:
        provenance = saved["run"]["provenance"]
        if any(
            env.provenance[key] != provenance[key]
            for key in ("low_level_sha256", "robot_xml_sha256")
        ):
            raise ValueError("Evaluation walking/robot assets differ from training")
    rows = []
    try:
        for seed in seeds:
            observation, initial = env.reset(seed=seed)
            impulses = {
                name: np.zeros(config.num_robots)
                for name in ("body", "leg_link", "foot")
            }
            robot_contact = False
            frames = [env.render()] if video_dir else None
            terminated = truncated = False
            total_reward = 0.0
            trace = []
            while not (terminated or truncated):
                action = (
                    policy_action(actor, observation)
                    if actor is not None
                    else rule_action(env.core, policy)
                )
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                for name in impulses:
                    impulses[name] += info[f"{name}_normal_impulse_ns"]
                robot_contact |= info["robot_collision_fraction"] > 0
                if frames is not None:
                    frames.append(env.render())
                    trace.append(
                        dict(
                            time=info["elapsed_seconds"],
                            cargo_xy=info["cargo_xy"],
                            cargo_yaw=env.core.cargo_yaw,
                            robot_xy=env.core.robot_xy.tolist(),
                            action=action.tolist(),
                            distance=info["distance"],
                            yaw_error=info["yaw_error"],
                        )
                    )
            row = {
                key: info[key]
                for key in (
                    "success",
                    "distance",
                    "yaw_error",
                    "footprint_error_m",
                    "elapsed_seconds",
                    "cargo_speed",
                    "cargo_yaw_speed",
                    "robot_fall",
                    "cargo_fall",
                    "out_of_bounds",
                )
            }
            row.update(
                seed=seed,
                initial_distance=initial["distance"],
                initial_yaw_error=initial["yaw_error"],
                total_reward=total_reward,
                body_contact=bool(impulses["body"].sum() > 1e-6),
                robot_contact=bool(robot_contact),
                total_cargo_normal_impulse_ns={
                    k: v.tolist() for k, v in impulses.items()
                },
            )
            rows.append(row)
            if frames is not None:
                import mediapy as media

                directory = Path(video_dir)
                directory.mkdir(parents=True, exist_ok=True)
                media.write_video(
                    directory / f"seed_{seed}.mp4", frames, fps=round(1 / config.dt)
                )
                (directory / f"seed_{seed}_trace.json").write_text(
                    json.dumps(trace, indent=2) + "\n"
                )
    finally:
        env.close()
    return rows


def evaluate_pose(
    checkpoint=None,
    *,
    config=None,
    policy="learned",
    episodes=20,
    seed=80000,
    workers=2,
    output=None,
    video_dir=None,
    asset_root=None,
    video_width=640,
    video_height=480,
):
    """Compare controllers on identical unseen seeds, using fixed final tolerances.

    With a checkpoint, config defaults to the saved task's final accuracy, even
    when its curriculum never reached that accuracy. Rules require an explicit
    config. Recording runs sequentially and returns the same physical metrics.
    """
    if min(episodes, workers) < 1:
        raise ValueError("Episode and worker counts must be positive")
    if config is None:
        if checkpoint is None:
            config = PosePushConfig()
        else:
            _, _, saved = load_mappo(checkpoint)
            if saved["format"] != POSE_FORMAT:
                raise ValueError("Pose evaluation requires a pose checkpoint")
            config = PosePushConfig.from_checkpoint(saved["pose_config"])
    if not isinstance(config, PosePushConfig):
        raise TypeError("Pose evaluation requires PosePushConfig")
    if config.dt != 0.2 and video_dir:
        raise ValueError("Pose videos currently use the standard 0.2 s control step")
    call = partial(
        _run_batch,
        config,
        checkpoint,
        policy,
        video_dir=video_dir,
        asset_root=asset_root,
        width=video_width,
        height=video_height,
    )
    seeds = list(range(seed, seed + episodes))
    if workers == 1 or video_dir:
        rows = call(seeds=seeds)
    else:
        groups = [seeds[i::workers] for i in range(min(workers, episodes))]
        with ProcessPoolExecutor(
            max_workers=len(groups), mp_context=get_context("spawn")
        ) as pool:
            rows = [row for batch in pool.map(call, groups) for row in batch]
        rows.sort(key=lambda row: row["seed"])
    report = dict(
        policy=policy,
        config=asdict(config),
        seed=seed,
        episodes=rows,
        demonstration_actions_used=False,
        successes=sum(r["success"] for r in rows),
        success_rate=float(np.mean([r["success"] for r in rows])),
        mean_distance_m=float(np.mean([r["distance"] for r in rows])),
        mean_yaw_error_degrees=float(
            np.rad2deg(np.mean([r["yaw_error"] for r in rows]))
        ),
        mean_footprint_error_m=float(np.mean([r["footprint_error_m"] for r in rows])),
        body_contact_episodes=sum(r["body_contact"] for r in rows),
        robot_contact_episodes=sum(r["robot_contact"] for r in rows),
        falls=sum(r["robot_fall"] or r["cargo_fall"] for r in rows),
    )
    if checkpoint:
        report["checkpoint"] = str(checkpoint)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
    return report
