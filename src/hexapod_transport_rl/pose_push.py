"""Approaching, pushing, rotation and settling, learned from team rewards.

The T origin is the junction of its two bars. Robot observations and actions
use a mirrored coordinate convention to share the same actor between both sides.
No controller or demonstration produces learning actions.
"""

from dataclasses import asdict, dataclass, field, replace
from functools import partial
from math import radians

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from .api import HexapodPushEnv
from .config import (
    COMMAND_LIMIT,
    STAND_HEIGHT,
    PushConfig,
    command_to_action,
    rotation,
    wrap,
)
from .contacts import BODY
from .env import PushEnv
from .pose_rewards import pose_reward_terms
from .push_approach import PushApproachDistance
from .rewards import PoseRewardWeights

POSE_ENV_ID = "HexapodPosePush-v0"
POSE_COMMAND_LIMIT = (0.20, 0.20, 0.60)


@dataclass(frozen=True)
class PosePushConfig(PushConfig):
    """Goal pose and reset variations for the short physical pushing task."""

    shape: str = "T"
    goal_distance: float = 0.4
    episode_seconds: float = 40.0
    position_tolerance: float = 0.08
    yaw_tolerance: float = radians(5)
    success_hold_seconds: float = 1.0
    push_gap: float = 0.20
    max_yaw_degrees: float = 30.0
    min_yaw_degrees: float = 5.0
    goal_lateral_range: float = 0.08
    position_jitter: float = 0.04
    yaw_jitter: float = 0.12
    robot_start: str = "goal_side"
    start_clearance: float = 0.55
    rear_start_fraction: float = 0.0
    start_angle_degrees: float = 180.0
    approach_metric: str = "collision_free"
    approach_clearance: float = 0.4
    approach_penetration_cost: float = 10.0
    action_frame: str = "cargo"
    cargo_command_limits: tuple[float, float, float] = POSE_COMMAND_LIMIT
    shaping_discount: float = 0.99
    reward_weights: PoseRewardWeights = field(default_factory=PoseRewardWeights)

    def __post_init__(self):
        object.__setattr__(
            self, "cargo_command_limits", tuple(self.cargo_command_limits)
        )
        if (
            len(self.cargo_command_limits) != 3
            or not np.isfinite(self.cargo_command_limits).all()
            or min(self.cargo_command_limits) <= 0
        ):
            raise ValueError("Three positive finite cargo command limits are required")
        if isinstance(self.reward_weights, dict):
            object.__setattr__(
                self, "reward_weights", PoseRewardWeights(**self.reward_weights)
            )
        super().__post_init__()
        if self.shape != "T" or self.t_geometry == "legacy":
            raise ValueError("Pose pushing uses a T with a centerline-junction origin")
        if self.robot_start not in ("goal_side", "cargo_rear"):
            raise ValueError("robot_start must be goal_side or cargo_rear")
        if self.approach_metric not in ("euclidean", "collision_free"):
            raise ValueError("Unknown approach distance metric")
        if self.action_frame not in ("body", "cargo"):
            raise ValueError("action_frame must be body or cargo")
        if not np.isfinite(self.approach_clearance) or self.approach_clearance <= 0:
            raise ValueError("approach_clearance must be positive and finite")
        if (
            not np.isfinite(self.approach_penetration_cost)
            or self.approach_penetration_cost < 1
        ):
            raise ValueError("approach_penetration_cost must be at least one")
        if not 0 <= self.rear_start_fraction <= 1:
            raise ValueError("rear_start_fraction must be in [0,1]")
        if not 0 <= self.start_angle_degrees <= 180:
            raise ValueError("start_angle_degrees must be in [0,180]")
        if not np.isfinite(self.start_clearance) or self.start_clearance <= 0:
            raise ValueError("start_clearance must be positive and finite")
        values = (
            self.max_yaw_degrees,
            self.min_yaw_degrees,
            self.goal_lateral_range,
            self.position_jitter,
            self.yaw_jitter,
        )
        if (
            not np.isfinite(values).all()
            or min(values) < 0
            or not 0 <= self.shaping_discount <= 1
        ):
            raise ValueError("Pose reset ranges must be finite and nonnegative")
        if self.min_yaw_degrees > self.max_yaw_degrees or self.max_yaw_degrees > 90:
            raise ValueError("Yaw reset range must satisfy 0 <= min <= max <= 90")

    @property
    def start_distance(self):
        """Robot-center distance along cargo-to-goal direction at reset."""
        return max(self.t_stem_length, self.goal_distance) + self.start_clearance

    def start_pose(self, slot, *, rear=False):
        """Local XY and heading for a reset; 180 degrees is the final goal side."""
        if rear or self.robot_start == "cargo_rear":
            return self.rear_face - self.push_gap - 0.10, slot, 0.0
        if self.start_angle_degrees == 180:
            return self.start_distance, slot, np.pi
        angle = radians(self.start_angle_degrees)
        radius = 0.4 + (self.start_distance - 0.4) * self.start_angle_degrees / 180
        side = 1 if slot >= 0 else -1
        inner_slot = min(abs(peer) for peer in self.slots if (peer >= 0) == (slot >= 0))
        # Preserve spacing within each side when there are three or four robots.
        lateral = (
            radius * np.sin(angle)
            + inner_slot * abs(np.cos(angle))
            + abs(slot)
            - inner_slot
        )
        return (
            -radius * np.cos(angle),
            side * lateral,
            -side * angle,
        )

    @classmethod
    def from_checkpoint(cls, saved: dict):
        """Keep the geometry and near-cargo starts of older pose checkpoints."""
        saved = dict(saved)
        if "reward_weights" in saved:
            saved["reward_weights"] = {
                "approach_error": 0.0,
                "push_heading": 0.0,
                "orientation_error": 0.0,
                **saved["reward_weights"],
            }
        return cls(
            **{
                "t_geometry": "equal_arms",
                "robot_start": "cargo_rear",
                "approach_metric": "euclidean",
                "approach_clearance": 0.16,
                "approach_penetration_cost": 1.0,
                "action_frame": "body",
                "cargo_command_limits": (0.15, 0.15, 0.60),
                **saved,
            }
        )


@dataclass(frozen=True)
class PoseStage:
    """Reset difficulty and success accuracy, applied at the next reset."""

    name: str
    distance: float
    max_yaw_degrees: float
    position_tolerance: float
    yaw_tolerance: float
    hold_seconds: float
    rear_start_fraction: float | None = None
    start_angle_degrees: float | None = None

    def apply(self, config: PosePushConfig):
        start = {
            key: value
            for key, value in (
                ("rear_start_fraction", self.rear_start_fraction),
                ("start_angle_degrees", self.start_angle_degrees),
            )
            if value is not None
        }
        return replace(
            config,
            goal_distance=self.distance,
            max_yaw_degrees=self.max_yaw_degrees,
            position_tolerance=self.position_tolerance,
            yaw_tolerance=self.yaw_tolerance,
            success_hold_seconds=self.hold_seconds,
            **start,
        )


POSE_STAGES = (
    *(
        PoseStage(
            f"approach_{angle}", 0.3, 15.0, 0.08, radians(8), 0.6, 0.0, float(angle)
        )
        for angle in (0, 45, 90, 135, 180)
    ),
    PoseStage("transport", 0.4, 30.0, 0.08, radians(5), 0.6),
    PoseStage("settle", 0.4, 30.0, 0.08, radians(5), 1.0),
)


def observe_pose(core: PushEnv) -> np.ndarray:
    """(N, 18+4*(N-1)): own state, target, cargo velocity and relative peers.

    Local X/Y are cargo coordinates. The right side's Y and yaw signs are
    reflected; reflected actions are converted back before physical execution.
    Position is scaled by 1 m, velocity by walking-command limits.
    """
    cargo_rotation = rotation(core.cargo_yaw)
    positions = (core.robot_xy - core.cargo_xy) @ cargo_rotation
    headings = wrap(core.yaw(core.base_ids) - core.cargo_yaw)
    goal_xy = (core.goal - core.cargo_xy) @ cargo_rotation
    goal_yaw = wrap(core.goal_yaw - core.cargo_yaw)
    cargo_velocity = core.data.qvel[core.cargo_v : core.cargo_v + 2] @ cargo_rotation
    rows = []
    last_commands = core.last_commands.copy()
    if core.cfg.action_frame == "cargo":
        for i, heading in enumerate(headings):
            last_commands[i, :2] = last_commands[i, :2] @ rotation(heading).T
    for i, slot in enumerate(core.cfg.slots):
        side = 1.0 if slot >= 0 else -1.0
        signs = np.array([1, side, side])
        velocity = core.data.qvel[core.root_v[i] : core.root_v[i] + 2] @ cargo_rotation
        own = np.concatenate(
            (
                positions[i] * [1, side],
                [np.sin(headings[i]) * side, np.cos(headings[i])],
                np.r_[velocity, core.data.qvel[core.root_v[i] + 5]]
                * signs
                / COMMAND_LIMIT,
                last_commands[i] * signs / COMMAND_LIMIT,
                goal_xy * [1, side],
                [np.sin(goal_yaw) * side, np.cos(goal_yaw)],
                np.r_[cargo_velocity, core.data.qvel[core.cargo_v + 5]]
                * signs
                / COMMAND_LIMIT,
                [abs(slot) / core.cfg.width],
            )
        )
        peers = [
            np.r_[
                (positions[j] - positions[i]) * [1, side],
                np.sin(headings[j] - headings[i]) * side,
                np.cos(headings[j] - headings[i]),
            ]
            for j in range(core.cfg.num_robots)
            if j != i
        ]
        rows.append(np.concatenate((own, *peers)))
    return np.asarray(rows, dtype=np.float32)


class PosePhysics(PushEnv):
    """Same articulated dynamics; reset and rewards describe target-pose pushing."""

    def __init__(self, cfg=None, asset_root=None):
        cfg = cfg or PosePushConfig()
        self.approach_distance = PushApproachDistance(cfg)
        super().__init__(cfg, asset_root)

    def _mean_approach_distance(self):
        if self.cfg.approach_metric == "euclidean":
            return super()._mean_approach_distance()
        positions = (self.robot_xy - self.cargo_xy) @ rotation(self.cargo_yaw)
        return float(self.approach_distance(positions).mean())

    def _reset_cargo_and_goal(self, randomize):
        heading = self.rng.uniform(-np.pi, np.pi) if randomize else 0.0
        angle = np.deg2rad(self.cfg.max_yaw_degrees)
        if randomize:
            angle = np.deg2rad(
                self.rng.uniform(self.cfg.min_yaw_degrees, self.cfg.max_yaw_degrees)
            )
            angle *= self.rng.choice([-1, 1])
        self.goal_yaw = heading + angle
        lateral = (
            self.rng.uniform(-self.cfg.goal_lateral_range, self.cfg.goal_lateral_range)
            if randomize
            else 0
        )
        self.goal = rotation(heading) @ [self.cfg.goal_distance, lateral]
        self.model.body("goal").pos[:] = (*self.goal, self.cfg.cargo_height / 2)
        self.model.body("goal").quat[:] = (
            np.cos(self.goal_yaw / 2),
            0,
            0,
            np.sin(self.goal_yaw / 2),
        )
        self.data.qpos[self.cargo_q : self.cargo_q + 7] = (
            0,
            0,
            self.cfg.cargo_height / 2 + 0.002,
            np.cos(heading / 2),
            0,
            0,
            np.sin(heading / 2),
        )
        return heading

    def _reset_robot_poses(self, cargo_heading, randomize):
        rear = self.cfg.robot_start == "cargo_rear" or self.cfg.rear_start_fraction == 1
        if self.cfg.robot_start == "goal_side" and 0 < self.cfg.rear_start_fraction < 1:
            rear = (
                self.rng.random() < self.cfg.rear_start_fraction
                if randomize
                else self.cfg.rear_start_fraction >= 0.5
            )
        direction = cargo_heading if rear else np.arctan2(self.goal[1], self.goal[0])
        if not rear and self.cfg.start_angle_degrees != 180:
            direction = (
                cargo_heading
                + wrap(direction - cargo_heading) * self.cfg.start_angle_degrees / 180
            )
        for slot, q in zip(self.cfg.slots, self.root_q, strict=True):
            x, y, heading = self.cfg.start_pose(slot, rear=rear)
            xy = rotation(direction) @ [x, y]
            yaw = direction + heading
            if randomize:
                xy += self.rng.uniform(
                    -self.cfg.position_jitter, self.cfg.position_jitter, 2
                )
                yaw += self.rng.uniform(-self.cfg.yaw_jitter, self.cfg.yaw_jitter)
            self.data.qpos[q : q + 7] = (
                *xy,
                STAND_HEIGHT,
                np.cos(yaw / 2),
                0,
                0,
                np.sin(yaw / 2),
            )

    def observe(self):
        return observe_pose(self)

    def info(self):
        info = super().info()
        positions = (self.robot_xy - self.cargo_xy) @ rotation(self.cargo_yaw)
        headings = wrap(self.yaw(self.base_ids) - self.cargo_yaw)
        behind = np.clip(-positions[:, 0] / 0.4, 0, 1)
        info["push_heading_error"] = float((behind * (1 - np.cos(headings)) / 2).mean())
        info.update(
            cargo_speed=float(
                np.linalg.norm(self.data.qvel[self.cargo_v : self.cargo_v + 2])
            ),
            cargo_yaw_speed=float(abs(self.data.qvel[self.cargo_v + 5])),
        )
        # The three endpoint errors reveal the visible mismatch of both T pieces.
        endpoints = np.array(
            [
                [0, -self.cfg.t_arm_length],
                [0, self.cfg.t_arm_length],
                [self.cfg.t_stem_length, 0],
            ]
        )
        actual = endpoints @ rotation(self.cargo_yaw).T + self.cargo_xy
        target = endpoints @ rotation(self.goal_yaw).T + self.goal
        info["footprint_error_m"] = float(
            np.sqrt(np.mean(np.sum((actual - target) ** 2, axis=1)))
        )
        return info

    def _reward_terms(self, before, info, previous_approach, commands):
        before = {**before, "approach": previous_approach}
        after = {**info, "approach": self._mean_approach_distance()}
        terms = pose_reward_terms(
            self.cfg,
            before,
            after,
            commands,
            self.last_commands,
            COMMAND_LIMIT,
            self.contacts.part_contact_counts[BODY],
            self.contacts.robot_collision_steps,
            self.cfg.position_tolerance,
            self.cfg.yaw_tolerance,
            math=np,
        )
        return {key: float(value) for key, value in terms.items()}


class PosePushEnv(HexapodPushEnv):
    """Gymnasium API for learning the entire short pushing-and-stopping action."""

    core_type = PosePhysics

    def __init__(self, config=None, asset_root=None, **kwargs):
        if isinstance(config, dict):
            config = PosePushConfig(**config)
        config = config or PosePushConfig()
        if not isinstance(config, PosePushConfig):
            raise TypeError("Pose pushing requires PosePushConfig")
        self.base_config = config
        self.pending_config = config
        self.stage_name = "evaluation"
        super().__init__(config, asset_root, flatten=False, **kwargs)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, (config.num_robots, self.core.obs_dim), np.float32
        )

    @property
    def provenance(self):
        return {**super().provenance, "pose_config": asdict(self.base_config)}

    def set_stage(self, stage: PoseStage):
        self.pending_config = stage.apply(self.base_config)
        self.stage_name = stage.name

    def _configure_camera(self, camera):
        if self.config.robot_start == "goal_side":
            direction = self.core.goal / np.linalg.norm(self.core.goal)
            camera.lookat[:] = [*(direction * self.config.start_distance / 2), 0.15]
            camera.distance = max(4.5, self.config.width + 2.5)
            camera.azimuth, camera.elevation = 135, -50
            return
        camera.lookat[:] = [*(self.core.goal / 2), 0.15]
        camera.distance = max(3.8, self.config.width + 2.0)
        camera.azimuth, camera.elevation = 135, -50

    def reset(self, *, seed=None, options=None):
        self.config = self.core.cfg = self.pending_config
        return super().reset(seed=seed, options=options)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action):
            raise ValueError("Pose action must have shape (N,3) and values in [-1,1]")
        physical = action.copy()
        sides = np.where(self.config.slots >= 0, 1, -1)
        physical[:, 1:] *= sides[:, None]
        if self.config.action_frame == "cargo":
            commands = physical * np.asarray(
                self.config.cargo_command_limits, dtype=np.float32
            )
            headings = wrap(self.core.yaw(self.core.base_ids) - self.core.cargo_yaw)
            for i, heading in enumerate(headings):
                commands[i, :2] = commands[i, :2] @ rotation(heading)
            physical = command_to_action(commands)
        return super().step(physical)


def _worker_pose(config, asset_root, torch_threads):
    torch.set_num_threads(torch_threads)
    return PosePushEnv(config, asset_root)


def make_pose_vector(config, num_envs=2, asset_root=None, *, asynchronous=True):
    factories = [partial(_worker_pose, config, asset_root, 1) for _ in range(num_envs)]
    kwargs = dict(autoreset_mode=gym.vector.AutoresetMode.DISABLED, copy=True)
    if asynchronous:
        return gym.vector.AsyncVectorEnv(factories, context="spawn", **kwargs)
    return gym.vector.SyncVectorEnv(factories, **kwargs)


if POSE_ENV_ID not in gym.registry:
    gym.register(
        id=POSE_ENV_ID, entry_point="hexapod_transport_rl.pose_push:PosePushEnv"
    )
