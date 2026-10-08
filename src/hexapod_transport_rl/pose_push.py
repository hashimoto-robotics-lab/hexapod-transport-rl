"""Short-distance translation, rotation and settling, learned from team rewards.

The T origin is the junction of three equal arms. Robot observations and actions
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
from .config import COMMAND_LIMIT, PushConfig, rotation, wrap
from .contacts import BODY
from .env import PushEnv
from .pose_rewards import pose_reward_terms
from .rewards import PoseRewardWeights

POSE_ENV_ID = "HexapodPosePush-v0"


@dataclass(frozen=True)
class PosePushConfig(PushConfig):
    """Goal pose and reset variations for the short physical pushing task."""

    shape: str = "T"
    goal_distance: float = 0.4
    episode_seconds: float = 12.0
    position_tolerance: float = 0.08
    yaw_tolerance: float = radians(5)
    success_hold_seconds: float = 1.0
    push_gap: float = 0.20
    max_yaw_degrees: float = 30.0
    min_yaw_degrees: float = 5.0
    goal_lateral_range: float = 0.08
    position_jitter: float = 0.04
    yaw_jitter: float = 0.12
    shaping_discount: float = 0.99
    reward_weights: PoseRewardWeights = field(default_factory=PoseRewardWeights)

    def __post_init__(self):
        if isinstance(self.reward_weights, dict):
            object.__setattr__(
                self, "reward_weights", PoseRewardWeights(**self.reward_weights)
            )
        super().__post_init__()
        if self.shape != "T" or self.t_geometry != "equal_arms":
            raise ValueError("Pose pushing uses the equal-arm T")
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


@dataclass(frozen=True)
class PoseStage:
    """Reset difficulty and success accuracy, applied at the next reset."""

    name: str
    distance: float
    max_yaw_degrees: float
    position_tolerance: float
    yaw_tolerance: float
    hold_seconds: float

    def apply(self, config: PosePushConfig):
        return replace(
            config,
            goal_distance=self.distance,
            max_yaw_degrees=self.max_yaw_degrees,
            position_tolerance=self.position_tolerance,
            yaw_tolerance=self.yaw_tolerance,
            success_hold_seconds=self.hold_seconds,
        )


POSE_STAGES = (
    PoseStage("rotate", 0.3, 15.0, 0.08, radians(8), 0.6),
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
                core.last_commands[i] * signs / COMMAND_LIMIT,
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
        super()._reset_robot_poses(cargo_heading, False)
        if randomize:
            for q in self.root_q:
                self.data.qpos[q : q + 2] += self.rng.uniform(
                    -self.cfg.position_jitter, self.cfg.position_jitter, 2
                )
                yaw = cargo_heading + self.rng.uniform(
                    -self.cfg.yaw_jitter, self.cfg.yaw_jitter
                )
                self.data.qpos[q + 3 : q + 7] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))

    def observe(self):
        return observe_pose(self)

    def info(self):
        info = super().info()
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
                [self.cfg.t_arm_length, 0],
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
