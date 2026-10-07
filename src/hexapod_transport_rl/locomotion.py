"""Shared deployment of the frozen walking policy for walking and cargo tasks."""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import mujoco
import numpy as np
import torch
from numpy.typing import ArrayLike

from .config import (
    COMMAND_HIGH,
    COMMAND_LOW,
    CONTROL_DT,
    JOINT_NAMES,
    MODEL_RELATIVE_PATH,
    PHYSICS_DT,
    PHYSICS_STEPS,
    POLICY_RELATIVE_PATH,
    STAND,
)
from .motor import voltage_command


class WalkingController:
    """Keep the 25 Hz policy and 200 Hz motor loop identical in both scenes."""

    def __init__(
        self,
        asset_root: Path,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        num_robots: int,
    ) -> None:
        self.asset_root, self.model, self.data = asset_root, model, data
        self.num_robots = num_robots
        self._load_locomotion_policy()
        self._index_model()
        self.reset()

    def reset(self) -> None:
        self.data.qpos[self.joint_q] = STAND
        self.data.ctrl[self.actuators] = 0
        self.previous_action = np.zeros(
            (self.num_robots, len(JOINT_NAMES)), dtype=np.float32
        )

    def _load_locomotion_policy(self) -> None:
        """Validate the source deployment contract before loading frozen weights."""
        policy_dir = self.asset_root / POLICY_RELATIVE_PATH
        metadata = json.loads((policy_dir / "policy.json").read_text())
        policy_path = policy_dir / "controller.ts"
        self.policy_sha256 = hashlib.sha256(policy_path.read_bytes()).hexdigest()
        if self.policy_sha256 != metadata["sha256"]["controller.ts"]:
            raise ValueError("Locomotion controller hash differs from policy.json")
        if tuple(metadata["joint_names"]) != JOINT_NAMES:
            raise ValueError("Locomotion joint order differs from hexapod")
        if (
            metadata["actor_input"]["shape"] != ["batch", 63]
            or metadata["control_hz"] != 1 / CONTROL_DT
            or metadata["physics_hz_training"] != 1 / PHYSICS_DT
            or metadata["action_scale_rad"] != 0.3
            or metadata["forward_axis"] != "+X"
            or metadata["action_clip"] != [-3.0, 3.0]
            or tuple(metadata["action_joint_names"]) != JOINT_NAMES
            or not np.allclose(metadata["reference_joint_position_rad"], STAND)
            or not np.allclose(
                metadata["command_ranges"],
                np.stack([COMMAND_LOW, COMMAND_HIGH], axis=-1),
            )
        ):
            raise ValueError(
                "Expected hexapod 63-input IMU policy at 25 Hz with +X forward"
            )
        self.policy_metadata = metadata
        self.model_sha256 = hashlib.sha256(
            (self.asset_root / MODEL_RELATIVE_PATH).read_bytes()
        ).hexdigest()
        self.controller = torch.jit.load(str(policy_path), map_location="cpu").eval()

    def _index_model(self) -> None:
        """Cache each robot's MuJoCo addresses; qpos and qvel layouts differ."""
        n = self.num_robots
        self.base_ids = np.array([self.model.body(f"r{i}/base").id for i in range(n)])
        self.gyro_indices = np.array(
            [
                self.model.sensor(f"r{i}/imu_ang_vel").adr[0] + np.arange(3)
                for i in range(n)
            ]
        )
        self.root_q = np.array(
            [self.model.joint(f"r{i}/floating_base_joint").qposadr[0] for i in range(n)]
        )
        self.root_v = np.array(
            [self.model.joint(f"r{i}/floating_base_joint").dofadr[0] for i in range(n)]
        )
        self.joint_q = np.array(
            [
                [self.model.joint(f"r{i}/{j}").qposadr[0] for j in JOINT_NAMES]
                for i in range(n)
            ]
        )
        self.joint_v = np.array(
            [
                [self.model.joint(f"r{i}/{j}").dofadr[0] for j in JOINT_NAMES]
                for i in range(n)
            ]
        )
        self.actuators = np.array(
            [
                [self.model.actuator(f"r{i}/{j}").id for j in JOINT_NAMES]
                for i in range(n)
            ]
        )

    def inputs(self, commands: ArrayLike) -> tuple[torch.Tensor, ...]:
        """Match the exported controller: gyro in IMU axes, gravity in body axes."""
        gravity = -self.data.xmat[self.base_ids].reshape(-1, 3, 3)[:, 2, :]
        values = (
            self.data.qpos[self.joint_q],
            self.data.qvel[self.joint_v],
            self.previous_action,
            commands,
            self.data.sensordata[self.gyro_indices],
            gravity,
        )
        return tuple(torch.from_numpy(np.asarray(v, dtype=np.float32)) for v in values)

    @torch.inference_mode()
    def advance(
        self,
        commands: np.ndarray,
        control_steps: int,
        on_control_step: Callable[[], None] | None = None,
        on_physics_step: Callable[[], None] | None = None,
    ) -> None:
        """Hold velocity commands; refresh joint targets at 25 Hz, voltage at 200 Hz."""
        for _ in range(control_steps):
            mujoco.mj_forward(self.model, self.data)
            target, previous = self.controller(*self.inputs(commands))
            if not torch.isfinite(target).all() or not torch.isfinite(previous).all():
                raise FloatingPointError("Non-finite walking controller output")
            self.previous_action = previous.numpy().copy()
            targets = target.numpy()
            for _ in range(PHYSICS_STEPS):
                self.data.ctrl[self.actuators] = voltage_command(
                    targets, self.data.qpos[self.joint_q]
                )
                mujoco.mj_step(self.model, self.data)
                if on_physics_step is not None:
                    on_physics_step()
            if on_control_step is not None:
                mujoco.mj_forward(self.model, self.data)
                on_control_step()
        # Derived transforms otherwise lag qpos by one physics tick.
        mujoco.mj_forward(self.model, self.data)
        if (
            not np.isfinite(self.data.qpos).all()
            or not np.isfinite(self.data.qvel).all()
        ):
            raise FloatingPointError("Non-finite MuJoCo state")
