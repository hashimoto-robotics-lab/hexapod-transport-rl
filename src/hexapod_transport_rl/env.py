"""Physical team environment beneath the Gymnasium adapter.

A high-level step runs the frozen walking policy, advances MuJoCo, measures
contacts, then computes termination and a shared team reward. See docs/code-guide.md.
"""

from collections.abc import Callable
from pathlib import Path

import mujoco
import numpy as np
import torch
from numpy.typing import ArrayLike

from .config import (
    COMMAND_LIMIT,
    PHYSICS_STEPS,
    STAND_HEIGHT,
    PushConfig,
    action_to_command,
    find_asset_root,
    rotation,
    wrap,
)
from .contacts import BODY, ContactTracker
from .locomotion import WalkingController
from .model import build_model
from .types import Info, Observation, ResetResult, StepResult

# Physical failure/settling thresholds; distances in m and speeds in m/s or rad/s.
MIN_ROBOT_HEIGHT = 0.08
MIN_ROBOT_UPRIGHT = 0.5
MIN_CARGO_UPRIGHT = 0.7
MAX_SETTLED_SPEED = 0.06
MAX_SETTLED_YAW_SPEED = 0.1


class PushEnv:
    """One MuJoCo world containing all N robots and a shared cargo.

    Observations/actions have shape (N,D)/(N,3). Low-level joint state and
    previous walking actions are kept separately for each robot. Public model
    indices (root_q, root_v, joint_q, joint_v, actuators) remain available for
    diagnostics; q means a qpos address and v means a qvel address.
    """

    def __init__(
        self, cfg: PushConfig | None = None, asset_root: str | Path | None = None
    ) -> None:
        self.cfg = cfg or PushConfig()
        self.asset_root = (
            Path(asset_root) if asset_root else find_asset_root()
        ).resolve()
        self.model = build_model(self.asset_root, self.cfg)
        self.data = mujoco.MjData(self.model)
        self.walking = WalkingController(
            self.asset_root, self.model, self.data, self.cfg.num_robots
        )
        self._index_model()
        self.contacts = ContactTracker(self.model, self.cfg.num_robots)
        # Preserve the diagnostic geom lookup attributes of the original API.
        self.geom_owner = self.contacts.geom_owner
        self.geom_kind = self.contacts.geom_kind
        self.rng = np.random.default_rng()
        self.reset(seed=0)
        self.obs_dim = self.observe().shape[-1]
        self.state_dim = self.cfg.num_robots * self.obs_dim

    def _index_model(self) -> None:
        """Expose robot diagnostics and index the cargo separately from walking."""
        self.base_ids = self.walking.base_ids
        self.gyro_indices = self.walking.gyro_indices
        self.root_q, self.root_v = self.walking.root_q, self.walking.root_v
        self.joint_q, self.joint_v = self.walking.joint_q, self.walking.joint_v
        self.actuators = self.walking.actuators
        self.controller = self.walking.controller
        self.policy_metadata = self.walking.policy_metadata
        self.policy_sha256 = self.walking.policy_sha256
        self.model_sha256 = self.walking.model_sha256
        self.cargo_id = self.model.body("cargo").id
        self.cargo_q = self.model.joint("cargo_free").qposadr[0]
        self.cargo_v = self.model.joint("cargo_free").dofadr[0]

    @property
    def previous_action(self) -> np.ndarray:
        return self.walking.previous_action

    @property
    def cargo_xy(self) -> np.ndarray:
        return self.data.xpos[self.cargo_id, :2].copy()

    @property
    def robot_xy(self) -> np.ndarray:
        return self.data.xpos[self.base_ids, :2].copy()

    def yaw(self, ids: ArrayLike) -> np.ndarray:
        matrices = self.data.xmat[ids].reshape(-1, 3, 3)
        return np.arctan2(matrices[:, 1, 0], matrices[:, 0, 0])

    @property
    def cargo_yaw(self) -> float:
        return float(self.yaw([self.cargo_id])[0])

    def push_positions(self) -> np.ndarray:
        """Desired robot centers (N,2) behind the cargo, in world coordinates."""
        # Front feet stand 0.258 m ahead along +X. Keep the body behind the box.
        local = np.column_stack(
            (
                np.full(self.cfg.num_robots, self.cfg.rear_face - self.cfg.push_gap),
                self.cfg.slots,
            )
        )
        return local @ rotation(self.cargo_yaw).T + self.cargo_xy

    def reset(self, seed: int | None = None, randomize: bool = False) -> ResetResult:
        """Initialize an episode; preserve the RNG stream when seed is None."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        cargo_heading = self._reset_cargo_and_goal(randomize)
        self._reset_robot_poses(cargo_heading, randomize)
        self.walking.reset()
        self.last_commands = np.zeros((self.cfg.num_robots, 3), dtype=np.float32)
        self.steps = 0
        self.success_time = 0.0
        self.done = False
        mujoco.mj_forward(self.model, self.data)
        self.initial_xy = self.cargo_xy
        return self.observe(), self.info()

    def _reset_cargo_and_goal(self, randomize: bool) -> float:
        """Sample the common task heading and place cargo plus goal marker."""
        heading = self.rng.uniform(-np.pi, np.pi) if randomize else 0.0
        cargo_heading = heading + (self.rng.uniform(-0.12, 0.12) if randomize else 0)
        self.goal_yaw = heading
        local_goal = [
            self.cfg.goal_distance,
            self.rng.uniform(-0.20, 0.20) if randomize else 0,
        ]
        self.goal = rotation(heading) @ local_goal
        goal_id = self.model.body("goal").id
        self.model.body_pos[goal_id] = (*self.goal, self.cfg.cargo_height / 2)
        self.model.body_quat[goal_id] = (np.cos(heading / 2), 0, 0, np.sin(heading / 2))
        cq = self.cargo_q
        self.data.qpos[cq : cq + 7] = (
            0,
            0,
            self.cfg.cargo_height / 2 + 0.002,
            np.cos(cargo_heading / 2),
            0,
            0,
            np.sin(cargo_heading / 2),
        )
        return cargo_heading

    def _reset_robot_poses(self, cargo_heading: float, randomize: bool) -> None:
        """Place robots behind their assigned cargo slots, with optional jitter."""
        for i, q in enumerate(self.root_q):
            xy = rotation(cargo_heading) @ np.array(
                [self.cfg.rear_face - self.cfg.push_gap - 0.10, self.cfg.slots[i]]
            )
            yaw = cargo_heading
            if randomize:
                xy += self.rng.uniform(-0.025, 0.025, 2)
                yaw += self.rng.uniform(-0.06, 0.06)
            self.data.qpos[q : q + 7] = (
                *xy,
                STAND_HEIGHT,
                np.cos(yaw / 2),
                0,
                0,
                np.sin(yaw / 2),
            )

    def observe(self) -> Observation:
        """Local observations (N, 16 + 4*(N-1)), in stable robot order.

        Own features occupy 0:16; each peer adds relative XY/3 m and yaw sin/cos.
        Position and velocity vectors use the observing robot's body frame.
        The feature order is part of the checkpoint contract (docs/api.md).
        """
        result = []
        robot_xy, yaws = self.robot_xy, self.yaw(self.base_ids)
        cargo_yaw = self.cargo_yaw
        for i, yaw in enumerate(yaws):
            body_rotation = rotation(yaw)
            velocity = (
                self.data.qvel[self.root_v[i] : self.root_v[i] + 2] @ body_rotation
            )
            own = np.concatenate(
                (
                    (self.cargo_xy - robot_xy[i]) @ body_rotation / 3,
                    (self.goal - self.cargo_xy) @ body_rotation / 3,
                    velocity,
                    [self.data.qvel[self.root_v[i] + 5]],
                    [np.sin(cargo_yaw - yaw), np.cos(cargo_yaw - yaw)],
                    [np.sin(self.goal_yaw - yaw), np.cos(self.goal_yaw - yaw)],
                    self.last_commands[i] / COMMAND_LIMIT,  # last normalized command
                    [self.data.xmat[self.base_ids[i], 8]],
                    [self.cfg.slots[i] / self.cfg.width],
                )
            )
            peers = [
                np.concatenate(
                    (
                        (robot_xy[j] - robot_xy[i]) @ body_rotation / 3,
                        [np.sin(yaws[j] - yaw), np.cos(yaws[j] - yaw)],
                    )
                )
                for j in range(self.cfg.num_robots)
                if j != i
            ]
            result.append(np.concatenate((own, *peers)))
        return np.asarray(result, dtype=np.float32)

    def state(self) -> Observation:
        """Central critic receives all agent observations in stable robot order."""
        return self.observe().ravel()

    def locomotion_inputs(self, commands: ArrayLike) -> tuple[torch.Tensor, ...]:
        """Controller input diagnostics; normal callers use step()."""
        return self.walking.inputs(commands)

    def info(self) -> Info:
        """Current cargo metrics, before step-specific reward/contact diagnostics."""
        return {
            "distance": float(np.linalg.norm(self.goal - self.cargo_xy)),
            "yaw_error": float(abs(wrap(self.goal_yaw - self.cargo_yaw))),
            "cargo_xy": self.cargo_xy.tolist(),
            "cargo_displacement": float(
                np.linalg.norm(self.cargo_xy - self.initial_xy)
            ),
            "success": False,
            "elapsed_seconds": self.steps * self.cfg.dt,
        }

    @torch.inference_mode()
    def step(
        self, action: ArrayLike, on_control_step: Callable[[], None] | None = None
    ) -> StepResult:
        """Advance one high-level action (N,3), returning a shared team reward.

        on_control_step runs after each walking-policy tick (25 Hz by default),
        e.g. for video capture. Ended episodes require an explicit reset.
        """
        if self.done:
            raise RuntimeError("Episode is over; call reset before step")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.cfg.num_robots, 3) or not np.isfinite(action).all():
            raise ValueError("Expected finite action with shape (num_robots, 3)")
        commands = action_to_command(action)
        before = self.info()
        previous_approach = self._mean_approach_distance()

        self._advance_physics(commands, on_control_step)
        self.steps += 1
        info = self.info()
        terminated, truncated = self._update_episode_status(info)
        reward_terms = self._reward_terms(before, info, previous_approach, commands)

        self.last_commands = commands
        self.done = terminated or truncated
        physics_steps = PHYSICS_STEPS * self.cfg.high_level_decimation
        info.update(self.contacts.as_info(physics_steps))
        info["reward_terms"] = reward_terms
        reward = float(sum(reward_terms.values()))
        return self.observe(), reward, terminated, truncated, info

    def _advance_physics(
        self, commands: np.ndarray, on_control_step: Callable[[], None] | None
    ) -> None:
        self.contacts.reset()
        self.walking.advance(
            commands,
            self.cfg.high_level_decimation,
            on_control_step=on_control_step,
            on_physics_step=lambda: self.contacts.record(self.model, self.data),
        )

    def _update_episode_status(self, info: Info) -> tuple[bool, bool]:
        """Update the success hold timer and append failure reasons to info.

        A true success/failure takes precedence over a simultaneous time limit.
        Only time limits bootstrap their final state during training.
        """
        robot_fall = bool(
            np.any(self.data.xpos[self.base_ids, 2] < MIN_ROBOT_HEIGHT)
            or np.any(self.data.xmat[self.base_ids, 8] < MIN_ROBOT_UPRIGHT)
        )
        cargo_fall = bool(self.data.xmat[self.cargo_id, 8] < MIN_CARGO_UPRIGHT)
        outside = bool(
            np.any(np.linalg.norm(self.robot_xy, axis=1) > self.cfg.arena_radius)
            or np.linalg.norm(self.cargo_xy) > self.cfg.arena_radius
        )
        failed = robot_fall or cargo_fall or outside
        speed = np.linalg.norm(self.data.qvel[self.cargo_v : self.cargo_v + 2])
        settled = (
            info["distance"] < self.cfg.position_tolerance
            and info["yaw_error"] < self.cfg.yaw_tolerance
            and speed < MAX_SETTLED_SPEED
            and abs(self.data.qvel[self.cargo_v + 5]) < MAX_SETTLED_YAW_SPEED
            and not failed
        )
        self.success_time = self.success_time + self.cfg.dt if settled else 0
        success = self.success_time + 1e-8 >= self.cfg.success_hold_seconds
        terminated = bool(success or failed)
        truncated = bool(
            self.steps * self.cfg.dt + 1e-8 >= self.cfg.episode_seconds
            and not terminated
        )
        info.update(
            success=bool(success),
            failed=failed,
            robot_fall=robot_fall,
            cargo_fall=cargo_fall,
            out_of_bounds=outside,
        )
        return terminated, truncated

    def _mean_approach_distance(self) -> float:
        return np.linalg.norm(self.push_positions() - self.robot_xy, axis=1).mean()

    def _reward_terms(
        self, before: Info, info: Info, previous_approach: float, commands: np.ndarray
    ) -> dict[str, float]:
        """Shared reward components, evaluated before updating last_commands.

        Distances are in meters; yaw errors in radians. Contact/time penalties
        scale with dt, and body contact is penalized so legs do the pushing.
        """
        physics_steps = PHYSICS_STEPS * self.cfg.high_level_decimation
        approach = self._mean_approach_distance()
        if info["success"]:
            terminal_reward = self.cfg.reward_weights.success
        elif info["failed"]:
            terminal_reward = self.cfg.reward_weights.failure
        else:
            terminal_reward = 0.0
        terms = {
            "progress": self.cfg.reward_weights.progress
            * (before["distance"] - info["distance"]),
            "orientation": self.cfg.reward_weights.orientation
            * (before["yaw_error"] - info["yaw_error"]),
            "approach": self.cfg.reward_weights.approach
            * (previous_approach - approach),
            "command_change": -self.cfg.reward_weights.command_change
            * float(np.mean(((commands - self.last_commands) / COMMAND_LIMIT) ** 2)),
            "time": -self.cfg.reward_weights.time * self.cfg.dt,
            "collision": -self.cfg.reward_weights.robot_contact
            * self.cfg.dt
            * self.contacts.robot_collision_steps
            / physics_steps,
            "body_cargo_contact": -self.cfg.reward_weights.body_contact
            * self.cfg.dt
            * float(self.contacts.part_contact_counts[BODY].mean())
            / physics_steps,
            "terminal": terminal_reward,
        }
        return terms
