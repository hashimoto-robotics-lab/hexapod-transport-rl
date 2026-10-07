"""Record velocity commands sent to the frozen walking controller, without training."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

from hexapod_transport_rl import PushConfig, PushEnv
from hexapod_transport_rl.config import (
    COMMAND_HIGH,
    COMMAND_LOW,
    command_to_action,
    wrap,
)


def record_commands(commands: list[dict], output: Path) -> dict:
    """Play body-frame [vx, vy, yaw_rate] commands and record actual robot motion."""
    torch.set_num_threads(1)
    duration = sum(item["seconds"] for item in commands)
    env = PushEnv(PushConfig(shape="T", episode_seconds=duration + 1))
    # Reuse the exact transport physics/controller. Keep the other robot, cargo,
    # and goal marker away from this introductory walking scene.
    env.data.qpos[env.root_q[0] : env.root_q[0] + 2] = [-0.5, -0.2]
    env.data.qpos[env.root_q[1] : env.root_q[1] + 2] = [5, 4]
    env.data.qpos[env.cargo_q : env.cargo_q + 2] = [6, -4]
    env.model.body_pos[env.model.body("goal").id, :2] = [6, 4]
    mujoco.mj_forward(env.model, env.data)
    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[3] = 0  # Hide diagnostic collision shapes.
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, 0.1]
    camera.distance = 2.6
    camera.azimuth = 135
    camera.elevation = -50
    output.mkdir(parents=True, exist_ok=True)
    video_path = output / "walking_commands.mp4"
    records = []
    with (
        mujoco.Renderer(env.model, height=480, width=640) as renderer,
        imageio.get_writer(str(video_path), fps=5) as writer,
    ):
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        for item in commands:
            velocity = np.asarray(item["velocity"], dtype=np.float32)
            action = command_to_action([velocity, [0, 0, 0]])
            start_xy = env.robot_xy[0]
            start_yaw = float(env.yaw(env.base_ids)[0])
            for _ in range(round(item["seconds"] / env.cfg.dt)):
                _, _, terminated, truncated, info = env.step(action)
                renderer.update_scene(
                    env.data, camera=camera, scene_option=scene_option
                )
                writer.append_data(renderer.render())
                if terminated or truncated:
                    raise RuntimeError(f"歩行体験が途中で終了しました: {info}")
            record = dict(
                label=item["label"],
                command=velocity.tolist(),
                seconds=item["seconds"],
                start_xy_m=start_xy.tolist(),
                end_xy_m=env.robot_xy[0].tolist(),
                yaw_change_rad=float(wrap(env.yaw(env.base_ids)[0] - start_yaw)),
            )
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    result = dict(
        controller="frozen_learned_walking_policy",
        walking_policy_sha256=env.policy_sha256,
        training_performed=False,
        coordinate_frame="robot body: +X forward, +Y left, positive yaw counterclockwise",
        commands=records,
    )
    (output / "commands.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--commands", required=True, help="JSON list of velocity commands"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    commands = json.loads(args.commands)
    if not isinstance(commands, list) or not commands:
        raise ValueError("指令を1つ以上指定してください。")
    for item in commands:
        velocity = np.asarray(item["velocity"], dtype=float)
        seconds = item["seconds"]
        if (
            velocity.shape != (3,)
            or not np.isfinite(velocity).all()
            or not np.isfinite(seconds)
            or not 0.2 <= seconds <= 10
            or not np.isclose(seconds / 0.2, round(seconds / 0.2))
        ):
            raise ValueError("速度は3成分、時間は0.2秒刻みで0.2〜10秒にしてください。")
        if np.any(velocity < COMMAND_LOW) or np.any(velocity > COMMAND_HIGH):
            raise ValueError("速度指令が歩行モデルの範囲を超えています。")
    record_commands(commands, args.output)


if __name__ == "__main__":
    main()
