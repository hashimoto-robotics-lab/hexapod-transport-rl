"""Cylinder position task using the original articulated hexapod and walker.

The cargo must settle within 5 cm of its goal for 0.4 s. Contact forces from
the robot drive all cargo motion; cargo orientation is not part of success.
"""

from dataclasses import dataclass, replace
from math import pi, radians, sqrt

import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import Bounded, Categorical, Composite, Unbounded
from torchrl.envs import EnvBase

from .api import HexapodPushEnv
from .config import (
    COMMAND_HIGH,
    COMMAND_LOW,
    STAND_HEIGHT,
    PushConfig,
    command_to_action,
    rotation,
    wrap,
)
from .env import MAX_SETTLED_SPEED, PushEnv

DIRECTIONS = (
    (0.0, 0.0),
    (1.0, 0.0),
    (1 / sqrt(2), 1 / sqrt(2)),
    (0.0, 1.0),
    (-1 / sqrt(2), 1 / sqrt(2)),
    (-1.0, 0.0),
    (-1 / sqrt(2), -1 / sqrt(2)),
    (0.0, -1.0),
    (1 / sqrt(2), -1 / sqrt(2)),
)


@dataclass(frozen=True)
class SinglePushConfig(PushConfig):
    num_robots: int = 1
    shape: str = "cylinder"
    cargo_mass: float = 2.0
    cargo_friction: float = 0.3
    cargo_radius: float = 0.25
    goal_distance: float = 0.25
    episode_seconds: float = 40.0
    high_level_decimation: int = 5
    position_tolerance: float = 0.05
    yaw_tolerance: float = radians(5)
    success_hold_seconds: float = 0.4
    arena_radius: float = 8.0
    push_gap: float = 0.28
    max_yaw_degrees: float = 5.0
    min_yaw_degrees: float = 0.0
    goal_lateral_range: float = 0.02
    position_jitter: float = 0.02
    yaw_jitter: float = 0.12
    robot_start: str = "goal_side"
    start_clearance: float = 0.55
    rear_start_fraction: float = 0.0
    start_angle_degrees: float = 0.0
    approach_clearance: float = 0.4
    goal_pose: tuple[float, float, float] = (0.4, 0.0, pi / 4)
    target_heading_gain: float = 1.5
    goal_mode: str = "position"
    direction_speed: float = 0.15
    approach_strategy: str = "surface"
    rear_approach_gain: float = 2.0
    approach_cost_weight: float = 0.0
    goal_distance_cost_weight: float = 1.0
    success_bonus: float = 10.0
    spawn_mode: str = "fixed"
    spawn_radius_min: float = 0.65
    spawn_radius_max: float = 1.0
    spawn_azimuth_degrees: float | None = None
    spawn_heading_mode: str = "toward_cargo"
    spawn_heading_offset_degrees: float | None = None
    goal_position_mode: str = "fixed"
    goal_radius_min: float = 0.3
    goal_radius_max: float = 0.6
    goal_azimuth_degrees: float | None = None
    policy_rotation_mode: str = "none"

    def __post_init__(self):
        if self.shape == "cylinder" and self.goal_mode != "position":
            raise ValueError("Cylinder pushing supports position goals only")
        super().__post_init__()
        object.__setattr__(self, "goal_pose", tuple(self.goal_pose))
        if len(self.goal_pose) != 3 or not np.isfinite(self.goal_pose).all():
            raise ValueError("Goal pose requires finite x, y and yaw")
        if np.linalg.norm(self.goal_pose[:2]) + self.cargo_radius >= self.arena_radius:
            raise ValueError("Fixed goal footprint must fit within the arena")
        if self.robot_start not in ("goal_side", "cargo_rear"):
            raise ValueError("robot_start must be goal_side or cargo_rear")
        if not 0 <= self.rear_start_fraction <= 1:
            raise ValueError("rear_start_fraction must be in [0,1]")
        if not 0 <= self.start_angle_degrees <= 180:
            raise ValueError("start_angle_degrees must be in [0,180]")
        if (
            not np.isfinite(
                [
                    self.start_clearance,
                    self.approach_clearance,
                    self.target_heading_gain,
                ]
            ).all()
            or min(
                self.start_clearance, self.approach_clearance, self.target_heading_gain
            )
            <= 0
        ):
            raise ValueError("Clearances and heading gain must be positive and finite")
        ranges = (
            self.min_yaw_degrees,
            self.max_yaw_degrees,
            self.goal_lateral_range,
            self.position_jitter,
            self.yaw_jitter,
        )
        if (
            not np.isfinite(ranges).all()
            or min(ranges) < 0
            or not self.min_yaw_degrees <= self.max_yaw_degrees <= 90
        ):
            raise ValueError(
                "Reset ranges must be nonnegative; yaw must satisfy min <= max <= 90"
            )
        if self.num_robots != 1:
            raise ValueError("Single-robot PPO requires one hexapod")
        if not np.isfinite(self.direction_speed) or not 0 < self.direction_speed <= 0.2:
            raise ValueError("Direction speed must be in (0, 0.2] m/s")
        if self.goal_mode not in ("position", "coverage"):
            raise ValueError("goal_mode must be position or coverage")
        if self.approach_strategy not in ("surface", "rear_contact"):
            raise ValueError("Unknown cylinder approach strategy")
        if self.approach_strategy == "rear_contact" and self.shape != "cylinder":
            raise ValueError("Rear contact shaping requires a cylinder")
        if not np.isfinite(self.rear_approach_gain) or self.rear_approach_gain < 0:
            raise ValueError("Rear approach gain must be finite and nonnegative")
        if not np.isfinite(self.approach_cost_weight) or self.approach_cost_weight < 0:
            raise ValueError("Approach state cost must be finite and nonnegative")
        if (
            not np.isfinite(self.goal_distance_cost_weight)
            or self.goal_distance_cost_weight < 0
        ):
            raise ValueError("Goal distance cost must be finite and nonnegative")
        if self.spawn_mode not in ("fixed", "around"):
            raise ValueError("spawn_mode must be fixed or around")
        if self.spawn_heading_mode not in ("toward_cargo", "uniform"):
            raise ValueError("Spawn heading mode must be toward_cargo or uniform")
        if self.spawn_heading_mode == "uniform" and self.spawn_mode != "around":
            raise ValueError("Uniform spawn heading requires around spawning")
        if self.spawn_heading_offset_degrees is not None and (
            not np.isfinite(self.spawn_heading_offset_degrees)
            or not 0 <= self.spawn_heading_offset_degrees < 360
        ):
            raise ValueError("Spawn heading offset must be in [0,360)")
        if not np.isfinite(self.success_bonus) or self.success_bonus < 0:
            raise ValueError("Success bonus must be finite and nonnegative")
        if self.spawn_mode == "around" and self.shape != "cylinder":
            raise ValueError("Around spawning requires a cylinder")
        if (
            not np.isfinite([self.spawn_radius_min, self.spawn_radius_max]).all()
            or not 0 < self.spawn_radius_min <= self.spawn_radius_max
            or (
                self.spawn_mode == "around"
                and self.spawn_radius_min <= self.cargo_radius + 0.3
            )
        ):
            raise ValueError(
                "Spawn radii must be ordered and clear of the cylinder and legs"
            )
        if self.spawn_azimuth_degrees is not None and (
            not np.isfinite(self.spawn_azimuth_degrees)
            or not 0 <= self.spawn_azimuth_degrees < 360
        ):
            raise ValueError("Spawn azimuth must be in [0,360)")
        if self.goal_position_mode not in ("fixed", "uniform"):
            raise ValueError("Goal position mode must be fixed or uniform")
        if self.policy_rotation_mode not in ("none", "goal_octant"):
            raise ValueError("Policy rotation mode must be none or goal_octant")
        if self.policy_rotation_mode == "goal_octant" and self.shape != "cylinder":
            raise ValueError("Goal-octant rotation requires a circular cylinder")
        if self.goal_position_mode == "uniform" and (
            self.shape != "cylinder" or self.spawn_mode != "around"
        ):
            raise ValueError("Uniform goals require around-spawn cylinder pushing")
        if (
            not np.isfinite([self.goal_radius_min, self.goal_radius_max]).all()
            or not self.position_tolerance
            < self.goal_radius_min
            <= self.goal_radius_max
            or self.goal_radius_max + self.cargo_radius + self.start_distance
            >= self.arena_radius
        ):
            raise ValueError(
                "Goal radii must be ordered, outside tolerance and in the arena"
            )
        if self.goal_azimuth_degrees is not None and (
            not np.isfinite(self.goal_azimuth_degrees)
            or not 0 <= self.goal_azimuth_degrees < 360
        ):
            raise ValueError("Goal azimuth must be in [0,360)")

    @property
    def start_distance(self):
        if self.shape == "cylinder":
            return max(self.cargo_radius, self.goal_distance) + self.start_clearance
        return super().start_distance

    def start_pose(self, slot, *, rear=False):
        angle = np.deg2rad(self.start_angle_degrees)
        radius = self.cargo_radius + self.push_gap + 0.1
        if rear or self.robot_start == "cargo_rear":
            return (-radius, slot, 0.0)
        if self.start_angle_degrees == 180:
            return (self.start_distance, slot, np.pi)
        side = 1 if slot >= 0 else -1
        return (-radius * np.cos(angle), side * radius * np.sin(angle), -side * angle)

    @classmethod
    def from_checkpoint(cls, saved):
        """Read previous cylinder checkpoints while retiring unused task fields."""
        return cls(
            **{
                k: v
                for k, v in saved.items()
                if k
                not in (
                    "cargo_width",
                    "t_geometry",
                    "reward_weights",
                    "approach_metric",
                    "approach_penetration_cost",
                    "action_frame",
                    "cargo_command_limits",
                    "shaping_discount",
                    "task_type",
                    "obs_type",
                    "policy_features",
                    "policy_goal_angle_gain",
                    "policy_goal_position_gain",
                    "coverage_threshold",
                    "workspace_half_extent",
                    "target_position_gain",
                    "target_velocity_damping",
                    "precision_scale",
                    "yaw_precision_scale",
                )
            }
        )


def circle_coverage(distance, radius, math):
    """Exact overlap / goal area for two equal upright circular footprints."""
    fraction = distance / (2 * radius)
    fraction = np.clip(fraction, 0, 1) if math is np else fraction.clamp(0, 1)
    acos = np.arccos if math is np else torch.acos
    return 2 / np.pi * (acos(fraction) - fraction * math.sqrt(1 - fraction**2))


@dataclass(frozen=True)
class SinglePushResetStage:
    """Only future reset locations change; physics, actions and success stay fixed."""

    name: str
    rear_start_fraction: float

    def apply(self, config):
        return replace(config, rear_start_fraction=self.rear_start_fraction)


@dataclass(frozen=True)
class SinglePushAzimuthStage:
    """Fix a reset azimuth for an angular audit; None restores uniform sampling."""

    name: str
    azimuth_degrees: float | None

    def apply(self, config):
        return replace(config, spawn_azimuth_degrees=self.azimuth_degrees)


@dataclass(frozen=True)
class SinglePushPoseStage:
    """Force reset position angle and heading relative to facing the cylinder."""

    name: str
    azimuth_degrees: float | None
    heading_offset_degrees: float | None

    def apply(self, config):
        return replace(
            config,
            spawn_azimuth_degrees=self.azimuth_degrees,
            spawn_heading_offset_degrees=self.heading_offset_degrees,
        )


@dataclass(frozen=True)
class SinglePushGoalStage:
    """Fix only the initial world cargo-to-goal angle for a directional audit."""

    name: str
    azimuth_degrees: float | None

    def apply(self, config):
        return replace(config, goal_azimuth_degrees=self.azimuth_degrees)


def random_goal_delta(config, uniform, math):
    """Area-uniform goal annulus, independent of robot position and heading."""
    angle = uniform[..., 0] * (2 * np.pi)
    if config.goal_azimuth_degrees is not None:
        angle = angle * 0 + config.goal_azimuth_degrees * np.pi / 180
    radius = math.sqrt(
        config.goal_radius_min**2
        + uniform[..., 1] * (config.goal_radius_max**2 - config.goal_radius_min**2)
    )
    stack = np.stack if math is np else torch.stack
    return stack((radius * math.cos(angle), radius * math.sin(angle)), -1)


def around_start_pose(config, goal_delta, uniform, math):
    """Area-uniform annulus, angle-uniform azimuth; 0=rear, 180=goal side.

    This helper is used only at reset. No cargo/goal tracker supplies actions.
    uniform[..., 0:3] independently select azimuth, radius and initial yaw.
    """
    angle = uniform[..., 0] * (2 * np.pi)
    if config.spawn_azimuth_degrees is not None:
        angle = angle * 0 + config.spawn_azimuth_degrees * np.pi / 180
    radius = math.sqrt(
        config.spawn_radius_min**2
        + uniform[..., 1] * (config.spawn_radius_max**2 - config.spawn_radius_min**2)
    )
    length = math.sqrt((goal_delta * goal_delta).sum(axis=-1))
    length = np.maximum(length, 1e-08) if math is np else length.clamp_min(1e-08)
    direction = goal_delta / length[..., None]
    c, s = (math.cos(angle), math.sin(angle))
    x = radius * (-c * direction[..., 0] - s * direction[..., 1])
    y = radius * (-c * direction[..., 1] + s * direction[..., 0])
    atan2 = np.arctan2 if math is np else torch.atan2
    toward = atan2(-y, -x)
    if config.spawn_heading_mode == "uniform":
        yaw = (2 * uniform[..., 2] - 1) * np.pi
    else:
        yaw = toward + (2 * uniform[..., 2] - 1) * config.yaw_jitter
    if config.spawn_heading_offset_degrees is not None:
        yaw = toward + config.spawn_heading_offset_degrees * np.pi / 180
    xy = np.stack((x, y), -1) if math is np else torch.stack((x, y), -1)
    return (xy, yaw)


def cylinder_approach(config, relative, goal_delta, math):
    """Scalar state cost coupling the pushing side to the actual surface gap.

    No direction, waypoint, or action is supplied to the actor. The side cost
    fades within 10 cm of the goal, where the object-to-goal direction changes
    rapidly and precise position/stop control should dominate.
    """
    radius = math.sqrt((relative * relative).sum(axis=-1))
    gap = radius - config.cargo_radius
    gap = np.maximum(gap, 0) if math is np else gap.clamp_min(0)
    if config.approach_strategy == "surface":
        return gap
    distance = math.sqrt((goal_delta * goal_delta).sum(axis=-1))
    denominator = radius * distance
    denominator = (
        np.maximum(denominator, 1e-08) if math is np else denominator.clamp_min(1e-08)
    )
    cosine = -(relative * goal_delta).sum(axis=-1) / denominator
    cosine = np.clip(cosine, -1, 1) if math is np else cosine.clamp(-1, 1)
    angle = np.arccos(cosine) if math is np else cosine.acos()
    near_goal = distance / (2 * config.position_tolerance)
    near_goal = np.clip(near_goal, 0, 1) if math is np else near_goal.clamp(0, 1)
    return (
        gap
        + config.rear_approach_gain
        * (config.cargo_radius + config.approach_clearance)
        * angle
        * near_goal
    )


def direction_commands(config, action, headings, math, *, low=None, high=None):
    """Convert a world motion direction using only the robot's own heading.

    Turning toward a motion command is independent of cargo/goal state.
    Stop sends zero translation and zero yaw command to the frozen walker.
    """
    magnitude = math.sqrt((action * action).sum(axis=-1, keepdims=True))
    denominator = np.maximum(magnitude, 1) if math is np else magnitude.clamp_min(1)
    world = action / denominator * config.direction_speed
    c, s = (math.cos(headings), math.sin(headings))
    vx = world[..., 0] * c + world[..., 1] * s
    vy = -world[..., 0] * s + world[..., 1] * c
    atan2 = np.arctan2 if math is np else torch.atan2
    desired = atan2(world[..., 1], world[..., 0])
    delta = desired - headings
    yaw = config.target_heading_gain * atan2(math.sin(delta), math.cos(delta))
    yaw = math.where(magnitude[..., 0] > 1e-06, yaw, 0.0)
    commands = (
        np.stack((vx, vy, yaw), -1) if math is np else torch.stack((vx, vy, yaw), -1)
    )
    if math is np:
        return np.clip(commands, COMMAND_LOW, COMMAND_HIGH)
    low = commands.new_tensor(COMMAND_LOW) if low is None else low
    high = commands.new_tensor(COMMAND_HIGH) if high is None else high
    return commands.clamp(low, high)


def position_reward(config, before, after, math):
    """State progress only: no suggested actions, routes or contact labels."""
    distance = after["distance"]
    return (
        20 * (before["distance"] - distance)
        + 2 * (before["approach"] - after["approach"])
        - (config.goal_distance_cost_weight * distance + 0.02) * config.dt
        - config.approach_cost_weight * after["approach"] * config.dt
        + math.where(after["success"], config.success_bonus, 0.0)
        - math.where(after["failed"], 10.0, 0.0)
    )


class SinglePushPhysics(PushEnv):
    def __init__(self, cfg=None, asset_root=None):
        cfg = cfg or SinglePushConfig()
        PushEnv.__init__(self, cfg, asset_root)

    action_dim = 2

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

    def _reset_initial_robot_poses(self, cargo_heading, randomize):
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

    def _reset_fixed_frame(self, cargo_heading, randomize):
        self._reset_initial_robot_poses(cargo_heading, randomize)
        delta = self.cfg.goal_pose[2] - self.goal_yaw
        transform = rotation(delta)
        fixed_xy = np.array(self.cfg.goal_pose[:2])
        for q in [self.cargo_q, *self.root_q]:
            self.data.qpos[q : q + 2] = (
                self.data.qpos[q : q + 2] - self.goal
            ) @ transform.T + fixed_xy
            yaw = 2 * np.arctan2(self.data.qpos[q + 6], self.data.qpos[q + 3]) + delta
            self.data.qpos[q + 3 : q + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.goal, self.goal_yaw = (fixed_xy, self.cfg.goal_pose[2])
        self.model.body("goal").pos[:] = (*self.goal, self.cfg.cargo_height / 2)
        self.model.body("goal").quat[:] = [
            np.cos(self.goal_yaw / 2),
            0,
            0,
            np.sin(self.goal_yaw / 2),
        ]

    def _reset_robot_poses(self, cargo_heading, randomize):
        self._reset_fixed_frame(cargo_heading, randomize)
        if self.cfg.spawn_mode != "around":
            return
        cargo = self.data.qpos[self.cargo_q : self.cargo_q + 2]
        if self.cfg.goal_position_mode == "uniform":
            goal_uniform = self.rng.random(2) if randomize else np.array([0, 0.5])
            self.goal = cargo + random_goal_delta(self.cfg, goal_uniform, np)
            self.model.body("goal").pos[:] = (*self.goal, self.cfg.cargo_height / 2)
        uniform = self.rng.random(3) if randomize else np.array([0, 0.5, 0.5])
        relative, yaw = around_start_pose(self.cfg, self.goal - cargo, uniform, np)
        q = self.root_q[0]
        self.data.qpos[q : q + 7] = (
            *cargo + relative,
            STAND_HEIGHT,
            np.cos(yaw / 2),
            0,
            0,
            np.sin(yaw / 2),
        )

    def _mean_approach_distance(self):
        return float(
            cylinder_approach(
                self.cfg,
                self.robot_xy[0] - self.cargo_xy,
                self.goal - self.cargo_xy,
                np,
            )
        )

    def observe(self):
        return np.asarray(
            [
                np.r_[
                    self.robot_xy[0],
                    self.cargo_xy,
                    self.cargo_yaw,
                    self.yaw(self.base_ids)[0],
                ]
            ],
            dtype=np.float32,
        )

    def info(self):
        info = PushEnv.info(self)
        info.update(
            agent_pos=self.robot_xy.copy(),
            block_pose=np.r_[self.cargo_xy, self.cargo_yaw],
            goal_pose=np.r_[self.goal, self.goal_yaw],
            coverage=float(
                circle_coverage(info["distance"], self.cfg.cargo_radius, np)
            ),
            approach=self._mean_approach_distance(),
            cargo_speed=float(
                np.linalg.norm(self.data.qvel[self.cargo_v : self.cargo_v + 2])
            ),
            cargo_yaw_speed=float(abs(self.data.qvel[self.cargo_v + 5])),
        )
        return info

    def _update_episode_status(self, info):
        previous_hold = self.success_time
        terminated, truncated = super()._update_episode_status(info)
        settled = (
            info["distance"] < self.cfg.position_tolerance
            and info["cargo_speed"] < MAX_SETTLED_SPEED
            and (not info["failed"])
        )
        self.success_time = previous_hold + self.cfg.dt if settled else 0.0
        success = self.success_time + 1e-08 >= self.cfg.success_hold_seconds
        info["success"] = bool(success)
        terminated = bool(success or info["failed"])
        truncated = self.steps * self.cfg.dt + 1e-08 >= self.cfg.episode_seconds
        return (terminated, bool(truncated and (not terminated)))

    def step(self, action, on_control_step=None):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (1, 2) or not np.isfinite(action).all():
            raise ValueError("Expected one finite XY motion command")
        commands = direction_commands(self.cfg, action, self.yaw(self.base_ids), np)
        return PushEnv.step(self, command_to_action(commands), on_control_step)

    def _reward_terms(self, before, info, previous_approach, commands):
        return {"position_task": float(position_reward(self.cfg, before, info, np))}


class SinglePushGymEnv(HexapodPushEnv):
    """Internal continuous-direction physics; TorchRL exposes nine choices."""

    core_type = SinglePushPhysics

    def __init__(self, config=None, asset_root=None, **kwargs):
        config = config or SinglePushConfig()
        self.base_config = self.pending_config = config
        self.stage_name = "evaluation"
        super().__init__(config, asset_root, flatten=False, **kwargs)
        self.action_space = gym.spaces.Box(-1, 1, (1, 2), np.float32)

    def _configure_camera(self, camera):
        camera.lookat[:] = [*(self.core.initial_xy + self.core.goal) / 2, 0.12]
        camera.distance = (
            3.6
            if self.config.start_angle_degrees >= 90
            or self.config.spawn_mode == "around"
            else 2.5
        )
        camera.azimuth, camera.elevation = (135, -50)

    @property
    def provenance(self):
        return {
            **super().provenance,
            "action_type": "world-direction-to-original-walking-commands",
            "walking_model_loaded": True,
        }

    def set_stage(self, stage):
        self.pending_config = stage.apply(self.base_config)
        self.stage_name = stage.name

    def reset(self, *, seed=None, options=None):
        self.config = self.core.cfg = self.pending_config
        return super().reset(seed=seed, options=options)


class SinglePushEnv(EnvBase):
    """Flat (N,14) observations and scalar categorical or continuous XY actions.

    N is the number of independent worlds. There is no multi-agent dimension.
    The original robot, walking network, 18 joints, motor model and 200 Hz
    contact physics are used without cargo forces.
    """

    def __init__(
        self,
        config=None,
        *,
        action_mode="discrete",
        num_envs=2,
        backend="cpu",
        render_mode=None,
    ):
        config = config or SinglePushConfig()
        if not isinstance(config, SinglePushConfig):
            raise TypeError("SinglePushEnv requires SinglePushConfig")
        if action_mode not in ("discrete", "continuous") or num_envs < 1:
            raise ValueError("Use discrete/continuous actions and positive num_envs")
        if backend == "auto":
            backend = "warp" if torch.cuda.is_available() else "cpu"
        if backend not in ("cpu", "warp"):
            raise ValueError("backend must be cpu, warp or auto")
        device = torch.device("cuda:0" if backend == "warp" else "cpu")
        super().__init__(device=device, batch_size=[num_envs])
        self.config, self.action_mode, self.backend = (config, action_mode, backend)
        self.directions = torch.tensor(DIRECTIONS, device=device)
        self._reset_seed = None
        if backend == "warp":
            from .warp_single_push import WarpSinglePushWorlds

            self.worlds = WarpSinglePushWorlds(config, num_envs, device=device)
            self.provenance = self.worlds.provenance
        else:
            self.worlds = gym.vector.SyncVectorEnv(
                [
                    lambda: SinglePushGymEnv(config, render_mode=render_mode)
                    for _ in range(num_envs)
                ],
                autoreset_mode=gym.vector.AutoresetMode.DISABLED,
            )
            self.provenance = self.worlds.call("provenance")[0]
        self.is_closed = False
        self.observation_spec = Composite(
            observation=Unbounded(shape=(num_envs, 14), device=device),
            **{
                key: Unbounded(shape=(num_envs, 1), device=device)
                for key in ("distance", "coverage", "approach")
            },
            success=Categorical(
                2, shape=(num_envs, 1), dtype=torch.bool, device=device
            ),
            shape=(num_envs,),
            device=device,
        )
        self.action_spec = (
            Categorical(9, shape=(num_envs,), device=device)
            if action_mode == "discrete"
            else Bounded(-1.0, 1.0, shape=(num_envs, 2), device=device)
        )
        self.reward_spec = Unbounded(shape=(num_envs, 1), device=device)
        self.done_spec = Composite(
            **{
                key: Categorical(
                    2, shape=(num_envs, 1), dtype=torch.bool, device=device
                )
                for key in ("done", "terminated", "truncated")
            },
            shape=(num_envs,),
            device=device,
        )

    def _observation(self, raw, info):
        raw = torch.as_tensor(raw, device=self.device)[:, 0]
        if self.backend == "warp":
            worlds = self.worlds
            velocity = worlds.physics.qvel
            rv, cv = (int(worlds.reference.core.root_v[0]), worlds.cargo_v)
            own_velocity = velocity[:, rv : rv + 2]
            cargo_velocity = velocity[:, cv : cv + 2]
            yaw_velocity = velocity[:, cv + 5 : cv + 6]
            own_yaw_velocity = velocity[:, rv + 5 : rv + 6]
        else:
            own_velocity = torch.tensor(
                np.stack(
                    [
                        e.core.data.qvel[e.core.root_v[0] : e.core.root_v[0] + 2]
                        for e in self.worlds.envs
                    ]
                ),
                dtype=torch.float32,
            )
            cargo_velocity = torch.tensor(
                np.stack(
                    [
                        e.core.data.qvel[e.core.cargo_v : e.core.cargo_v + 2]
                        for e in self.worlds.envs
                    ]
                ),
                dtype=torch.float32,
            )
            yaw_velocity = torch.tensor(
                [[e.core.data.qvel[e.core.cargo_v + 5]] for e in self.worlds.envs],
                dtype=torch.float32,
            )
            own_yaw_velocity = torch.tensor(
                [[e.core.data.qvel[e.core.root_v[0] + 5]] for e in self.worlds.envs],
                dtype=torch.float32,
            )
        goal = (
            self.worlds.goal
            if self.backend == "warp"
            else raw.new_tensor(np.stack([e.core.goal for e in self.worlds.envs]))
        )
        delta = self.config.goal_pose[2] - raw[:, 4]
        delta = torch.zeros_like(delta)
        yaw_velocity = torch.zeros_like(yaw_velocity)
        observation = torch.cat(
            (
                (raw[:, :2] - raw[:, 2:4]) / self.config.width,
                (goal - raw[:, 2:4]) / self.config.width,
                torch.stack((delta.sin(), delta.cos()), -1),
                own_velocity / 0.2,
                cargo_velocity / 0.2,
                yaw_velocity,
                torch.stack((raw[:, 5].sin(), raw[:, 5].cos()), -1),
                own_yaw_velocity,
            ),
            -1,
        ).clone()
        result = TensorDict({"observation": observation}, self.batch_size, self.device)
        for key in ("distance", "coverage", "approach"):
            result[key] = (
                torch.as_tensor(info[key], dtype=torch.float32, device=self.device)
                .reshape(-1, 1)
                .clone()
            )
        result["success"] = (
            torch.as_tensor(
                info.get("is_success", info.get("success")),
                device=self.device,
                dtype=torch.bool,
            )
            .reshape(-1, 1)
            .clone()
        )
        return result

    def _reset(self, tensordict=None, **kwargs):
        mask = tensordict.get("_reset") if tensordict is not None else None
        options = None
        if mask is not None:
            mask = mask.reshape(-1)
            options = {"reset_mask": mask if self.backend == "warp" else mask.numpy()}
        raw, info = self.worlds.reset(seed=self._reset_seed, options=options)
        self._reset_seed = None
        result = self._observation(raw, info)
        for key in ("done", "terminated", "truncated"):
            result[key] = torch.zeros(
                (*self.batch_size, 1), device=self.device, dtype=torch.bool
            )
        return result

    def _step(self, tensordict):
        action = tensordict["action"].detach()
        force = (
            self.directions[action]
            if self.action_mode == "discrete"
            else action.clamp(-1, 1)
        ).unsqueeze(1)
        if self.backend == "cpu":
            force = force.numpy()
        raw, reward, terminated, truncated, info = self.worlds.step(force)
        result = self._observation(raw, info)
        result["reward"] = (
            torch.as_tensor(reward, device=self.device, dtype=torch.float32)
            .reshape(-1, 1)
            .clone()
        )
        for key, value in (("terminated", terminated), ("truncated", truncated)):
            result[key] = (
                torch.as_tensor(value, device=self.device).reshape(-1, 1).clone()
            )
        result["done"] = result["terminated"] | result["truncated"]
        return result

    def _set_seed(self, seed):
        self._reset_seed = seed

    def check_physics(self):
        if self.backend == "warp":
            self.worlds.physics.check()

    def set_reset_stage(self, stage):
        self.worlds.call("set_stage", stage)

    def close(self, *, raise_if_closed=True):
        if not self.is_closed:
            self.worlds.close()
        return super().close(raise_if_closed=raise_if_closed)
