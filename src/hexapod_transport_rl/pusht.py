"""Push-T conventions adapted to a team of articulated hexapods.

The official state/keypoint layouts, world XY target actions and coverage
criterion are retained. Robot heading and peer poses are the multi-robot
extensions. Target tracking uses only robot state, never cargo or goal state.
"""

from dataclasses import dataclass, field
from math import pi

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .api import HexapodPushEnv
from .config import COMMAND_HIGH, COMMAND_LOW, command_to_action, rotation
from .contacts import BODY
from .env import PushEnv
from .pose_push import PosePhysics, PosePushConfig, PosePushEnv
from .pusht_geometry import coverage_numpy, local_rectangles, transform_rectangles

PUSHT_ENV_ID = "HexapodPushT-v0"


@dataclass(frozen=True)
class PushTRewardWeights:
    """Official coverage reward by default; opt-in study terms are extensions."""

    coverage: float = 1.0
    time: float = 0.0
    body_contact: float = 0.0
    robot_contact: float = 0.0
    command_change: float = 0.0
    failure: float = 0.0
    success: float = 0.0

    def __post_init__(self):
        if not np.isfinite(list(vars(self).values())).all():
            raise ValueError("Reward coefficients must be finite")


@dataclass(frozen=True)
class PushTConfig(PosePushConfig):
    """Official-style task with fixed goal, requested layout and original gait."""

    task_type: str = "pusht"
    obs_type: str = "state"
    goal_pose: tuple[float, float, float] = (0.4, 0.0, pi / 4)
    coverage_threshold: float = 0.95
    workspace_half_extent: float = 4.0
    target_position_gain: float = 1.0
    target_velocity_damping: float = 0.2
    target_heading_gain: float = 1.5
    approach_metric: str = "euclidean"
    action_frame: str = "body"
    reward_weights: PushTRewardWeights = field(default_factory=PushTRewardWeights)

    def __post_init__(self):
        object.__setattr__(self, "goal_pose", tuple(self.goal_pose))
        if isinstance(self.reward_weights, dict):
            object.__setattr__(
                self, "reward_weights", PushTRewardWeights(**self.reward_weights)
            )
        super().__post_init__()
        if self.task_type != "pusht" or self.obs_type not in ("state", "keypoints"):
            raise ValueError("Push-T requires state or keypoints observations")
        if len(self.goal_pose) != 3 or not np.isfinite(self.goal_pose).all():
            raise ValueError("Goal pose requires finite x, y and yaw")
        if not 0 < self.coverage_threshold < 1:
            raise ValueError("A strict coverage threshold must be in (0,1)")
        if (
            np.linalg.norm(self.goal_pose[:2])
            + max(self.t_arm_length, self.t_stem_length)
            >= self.arena_radius
        ):
            raise ValueError("Fixed goal footprint must fit within the arena")
        gains = (
            self.workspace_half_extent,
            self.target_position_gain,
            self.target_heading_gain,
        )
        if (
            not np.isfinite(gains).all()
            or min(gains) <= 0
            or not np.isfinite(self.target_velocity_damping)
            or self.target_velocity_damping < 0
        ):
            raise ValueError("Workspace and tracking gains must be positive")
        if self.action_frame != "body":
            raise ValueError("XY target tracking sends body-frame gait commands")

    @property
    def observation_dim(self):
        return (3 if self.obs_type == "state" else 16) + 3 * self.num_robots


def target_position_commands(
    config, targets, positions, headings, velocity, *, math, low=None, high=None
):
    """Position servo, analogous to the official pusher PD controller.

    The gait accepts bounded velocities. Orient toward the requested target
    and damp translational motion. There are no waypoints or obstacle rules.
    """
    error = targets - positions
    world = (
        config.target_position_gain * error - config.target_velocity_damping * velocity
    )
    c, s = math.cos(headings), math.sin(headings)
    vx = world[..., 0] * c + world[..., 1] * s
    vy = -world[..., 0] * s + world[..., 1] * c
    desired_yaw = (
        math.atan2(error[..., 1], error[..., 0])
        if math is not np
        else np.arctan2(error[..., 1], error[..., 0])
    )
    delta = desired_yaw - headings
    yaw = config.target_heading_gain * (
        math.atan2(math.sin(delta), math.cos(delta))
        if math is not np
        else np.arctan2(np.sin(delta), np.cos(delta))
    )
    distance = math.sqrt((error * error).sum(axis=-1))
    yaw = math.where(distance > 0.05, yaw, 0.0)
    commands = (
        math.stack((vx, vy, yaw), axis=-1)
        if math is np
        else math.stack((vx, vy, yaw), dim=-1)
    )
    if math is np:
        return np.clip(commands, COMMAND_LOW, COMMAND_HIGH)
    low = commands.new_tensor(COMMAND_LOW) if low is None else low
    high = commands.new_tensor(COMMAND_HIGH) if high is None else high
    return commands.clamp(low, high)


def pusht_reward_terms(
    config,
    info,
    commands,
    previous,
    body_ticks,
    robot_ticks,
    *,
    math,
    command_limits=None,
):
    weights = config.reward_weights
    coverage = info["coverage"] / config.coverage_threshold
    coverage = math.clip(coverage, 0, 1) if math is np else coverage.clamp(0, 1)
    ticks = config.high_level_decimation * 8
    limits = (
        command_limits
        if command_limits is not None
        else (
            np.maximum(-COMMAND_LOW, COMMAND_HIGH)
            if math is np
            else commands.new_tensor(np.maximum(-COMMAND_LOW, COMMAND_HIGH))
        )
    )
    return dict(
        coverage=weights.coverage * coverage,
        time=-weights.time * config.dt + coverage * 0,
        body_contact=-weights.body_contact
        * config.dt
        * body_ticks.mean(axis=-1)
        / ticks,
        robot_contact=-weights.robot_contact * config.dt * robot_ticks / ticks,
        command_change=-weights.command_change
        * (((commands - previous) / limits) ** 2).mean(axis=(-2, -1)),
        failure=math.where(info["failed"], weights.failure, 0.0),
        success=math.where(info["success"], weights.success, 0.0),
    )


class PushTPhysics(PosePhysics):
    action_dim = 2

    def __init__(self, cfg=None, asset_root=None):
        cfg = cfg or PushTConfig()
        self.rectangles = local_rectangles(cfg)
        # The coverage task has no approach-path reward or visibility graph.
        PushEnv.__init__(self, cfg, asset_root)

    def _reset_robot_poses(self, cargo_heading, randomize):
        super()._reset_robot_poses(cargo_heading, randomize)
        delta = self.cfg.goal_pose[2] - self.goal_yaw
        transform = rotation(delta)
        fixed_xy = np.array(self.cfg.goal_pose[:2])
        for q in [self.cargo_q, *self.root_q]:
            self.data.qpos[q : q + 2] = (
                self.data.qpos[q : q + 2] - self.goal
            ) @ transform.T + fixed_xy
            yaw = 2 * np.arctan2(self.data.qpos[q + 6], self.data.qpos[q + 3]) + delta
            self.data.qpos[q + 3 : q + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.goal, self.goal_yaw = fixed_xy, self.cfg.goal_pose[2]
        self.model.body("goal").pos[:] = (*self.goal, self.cfg.cargo_height / 2)
        self.model.body("goal").quat[:] = [
            np.cos(self.goal_yaw / 2),
            0,
            0,
            np.sin(self.goal_yaw / 2),
        ]

    def observe(self):
        robot_xy = self.robot_xy
        headings = self.yaw(self.base_ids) % (2 * np.pi)
        block = np.r_[self.cargo_xy, self.cargo_yaw % (2 * np.pi)]
        keypoints = transform_rectangles(
            self.rectangles, self.cargo_xy, self.cargo_yaw
        ).ravel()
        rows = []
        for i in range(self.cfg.num_robots):
            standard = (
                np.r_[robot_xy[i], block]
                if self.cfg.obs_type == "state"
                else np.r_[keypoints, robot_xy[i]]
            )
            peers = [
                np.r_[robot_xy[j], headings[j]]
                for j in range(self.cfg.num_robots)
                if j != i
            ]
            rows.append(np.concatenate((standard, [headings[i]], *peers)))
        return np.asarray(rows, dtype=np.float32)

    def info(self):
        info = super().info()
        info["agent_pos"] = self.robot_xy.copy()
        info["block_pose"] = np.r_[self.cargo_xy, self.cargo_yaw]
        info["goal_pose"] = np.r_[self.goal, self.goal_yaw]
        info["coverage"] = coverage_numpy(
            self.rectangles, self.cargo_xy, self.cargo_yaw, self.goal, self.goal_yaw
        )
        return info

    def _update_episode_status(self, info):
        super()._update_episode_status(info)
        success = info["coverage"] > self.cfg.coverage_threshold and not info["failed"]
        info["success"] = bool(success)
        terminated = bool(success or info["failed"])
        truncated = (
            self.steps * self.cfg.dt + 1e-8 >= self.cfg.episode_seconds
            and not terminated
        )
        return terminated, bool(truncated)

    def _reward_terms(self, before, info, previous_approach, commands):
        terms = pusht_reward_terms(
            self.cfg,
            info,
            commands,
            self.last_commands,
            self.contacts.part_contact_counts[BODY],
            self.contacts.robot_collision_steps,
            math=np,
        )
        return {key: float(value) for key, value in terms.items()}

    def step(self, targets, on_control_step=None):
        targets = np.asarray(targets, dtype=np.float32)
        if targets.shape != (self.cfg.num_robots, 2) or not np.isfinite(targets).all():
            raise ValueError("XY targets must have shape (num_robots,2)")
        velocity = self.data.qvel[self.root_v[:, None] + np.arange(2)]
        commands = target_position_commands(
            self.cfg, targets, self.robot_xy, self.yaw(self.base_ids), velocity, math=np
        )
        return PushEnv.step(self, command_to_action(commands), on_control_step)


class PushTEnv(PosePushEnv):
    """Gym API: actions are world XY targets in meters, shape (N,2)."""

    core_type = PushTPhysics

    def __init__(self, config=None, asset_root=None, **kwargs):
        config = (
            PushTConfig(**config)
            if isinstance(config, dict)
            else config or PushTConfig()
        )
        if not isinstance(config, PushTConfig):
            raise TypeError("PushTEnv requires PushTConfig")
        super().__init__(config, asset_root, **kwargs)
        self.action_space = spaces.Box(
            -config.workspace_half_extent,
            config.workspace_half_extent,
            (config.num_robots, 2),
            np.float32,
        )

    def step(self, action):
        return HexapodPushEnv.step(self, action)


if PUSHT_ENV_ID not in gym.registry:
    gym.register(id=PUSHT_ENV_ID, entry_point="hexapod_transport_rl.pusht:PushTEnv")
