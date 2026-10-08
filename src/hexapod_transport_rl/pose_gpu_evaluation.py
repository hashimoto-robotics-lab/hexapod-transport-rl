"""Evaluate learned policies in parallel, using the native evaluator's reset cases."""

from math import ceil

import torch
from tensordict import TensorDict

from .torchrl_env import TorchRLTransportEnv
from .torchrl_mappo import POSE_FORMAT, load_mappo
from .warp_pose import cargo_coordinates


@torch.inference_mode()
def evaluate_warp(config, checkpoint, seeds, asset_root):
    """Run one episode per lane; completed lanes never contribute again.

    Only initial physical states are copied from native MuJoCo. Every action
    comes from the actor, and all subsequent states are simulated by Warp.
    """
    env = TorchRLTransportEnv(
        config, num_envs=len(seeds), asset_root=asset_root, backend="warp"
    )
    try:
        actor, _, saved = load_mappo(checkpoint, device=env.device)
        if saved["format"] != POSE_FORMAT:
            raise ValueError("Pose evaluation requires a learned pose checkpoint")
        if any(
            env.provenance[key] != saved["run"]["provenance"][key]
            for key in ("low_level_sha256", "robot_xml_sha256")
        ):
            raise ValueError("Evaluation walking/robot assets differ from training")
        env.reset()
        worlds = env.worlds
        initial = []
        for lane, seed in enumerate(seeds):
            _, info = worlds.reference.reset(seed=seed)
            core = worlds.reference.core
            for key in ("qpos", "qvel"):
                getattr(worlds.physics, key)[lane].copy_(
                    torch.as_tensor(getattr(core.data, key), device=env.device)
                )
            worlds.goal[lane].copy_(torch.as_tensor(core.goal, device=env.device))
            worlds.goal_yaw[lane] = core.goal_yaw
            initial.append(info)
        worlds.physics.forward()

        count = len(seeds)
        finished = torch.zeros(count, dtype=torch.bool, device=env.device)
        integral = torch.zeros(count, 2, device=env.device)
        path = torch.zeros(count, device=env.device)
        rewards = torch.zeros_like(path)
        collisions = torch.zeros_like(finished)
        impulses = torch.zeros(count, 3, config.num_robots, device=env.device)
        previous_xy = worlds.coordinates()[0].clone()
        endpoints = torch.tensor(
            [
                [0, -config.t_arm_length],
                [0, config.t_arm_length],
                [config.t_stem_length, 0],
            ],
            device=env.device,
        ).expand(count, -1, -1)
        # Save each lane's first terminal state, before resetting finished lanes.
        terminal = {}
        for step in range(ceil(config.episode_seconds / config.dt)):
            active = ~finished
            td = TensorDict(
                {"agents": {"observation": worlds.observe()}},
                batch_size=[count],
                device=env.device,
            )
            action = actor.get_dist(td).deterministic_sample
            _, reward, terminated, truncated, info = worlds.step(action)
            cargo_xy, yaw, _, _ = worlds.coordinates()
            integral[:, 0] += active * config.dt * info["distance"]
            integral[:, 1] += active * config.dt * info["yaw_error"]
            path += active * torch.linalg.vector_norm(cargo_xy - previous_xy, dim=-1)
            previous_xy.copy_(cargo_xy)
            rewards += active * reward
            contacts = worlds.physics.contacts
            impulses += active[:, None, None] * contacts.normal_impulses
            collisions |= active & (contacts.robot_collision_steps > 0)
            actual = cargo_coordinates(endpoints, -yaw) + cargo_xy[:, None]
            target = (
                cargo_coordinates(endpoints, -worlds.goal_yaw) + worlds.goal[:, None]
            )
            measurements = {
                key: info[key]
                for key in (
                    "success",
                    "distance",
                    "yaw_error",
                    "cargo_speed",
                    "cargo_yaw_speed",
                    "robot_fall",
                    "cargo_fall",
                    "out_of_bounds",
                )
            }
            measurements.update(
                elapsed_seconds=torch.full_like(path, (step + 1) * config.dt),
                footprint_error_m=((actual - target).square().sum(-1).mean(-1)).sqrt(),
            )
            done = terminated | truncated
            new_done = done & active
            for key, value in measurements.items():
                if key not in terminal:
                    terminal[key] = torch.zeros_like(value)
                terminal[key] = torch.where(new_done, value, terminal[key])
            finished |= done
            if bool(finished.all()):
                break
            if bool(done.any()):
                worlds.reset(options={"reset_mask": done})
                previous_xy.copy_(worlds.coordinates()[0])
        worlds.physics.check()
        if not bool(finished.all()):
            raise RuntimeError("Evaluation budget ended before all lanes terminated")
        terminal = {key: value.cpu().tolist() for key, value in terminal.items()}
        integral, path = integral.cpu().tolist(), path.cpu().tolist()
        rewards, impulses = rewards.cpu().tolist(), impulses.cpu().numpy()
        collisions = collisions.cpu().tolist()
        rows = []
        for lane, seed in enumerate(seeds):
            row = {key: value[lane] for key, value in terminal.items()}
            row.update(
                seed=seed,
                initial_distance=initial[lane]["distance"],
                initial_yaw_error=initial[lane]["yaw_error"],
                total_reward=rewards[lane],
                position_error_integral_m_s=integral[lane][0],
                yaw_error_integral_rad_s=integral[lane][1],
                cargo_path_m=path[lane],
                time_to_success_or_limit_s=(
                    row["elapsed_seconds"] if row["success"] else config.episode_seconds
                ),
                body_contact=bool(impulses[lane, 0].sum() > 1e-6),
                robot_contact=collisions[lane],
                total_cargo_normal_impulse_ns={
                    name: impulses[lane, index].tolist()
                    for index, name in enumerate(("body", "leg_link", "foot"))
                },
            )
            rows.append(row)
        return rows
    finally:
        env.close()
