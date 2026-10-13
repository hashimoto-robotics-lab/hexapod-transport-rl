"""Cooperative crossing: reach individual goals while leaving room for peers.

Only the high-level velocity policy is learned. MuJoCo, motors and the frozen
25 Hz walking controller are the same as in the pushing experiment.
"""

from dataclasses import asdict, dataclass, field
from functools import partial

import gymnasium as gym
import mujoco
import numpy as np

from .config import CONTROL_DT, PHYSICS_STEPS, action_to_command, find_asset_root
from .contacts import ContactTracker
from .walking import WalkingSimulation

NAVIGATION_ENV_ID = "HexapodNavigation-v0"


@dataclass(frozen=True)
class NavigationRewardWeights:
    progress: float = 20.0
    distance: float = 0.05
    separation: float = 2.0
    collision: float = 1.0
    arrival: float = 0.5
    success: float = 5.0
    failure: float = -5.0

    def __post_init__(self):
        if not all(np.isfinite(value) for value in asdict(self).values()):
            raise ValueError("Reward weights must be finite")


@dataclass(frozen=True)
class NavigationConfig:
    num_robots: int = 2
    start_radius: float = 0.95
    position_jitter: float = 0.08
    yaw_jitter: float = 0.2
    goal_tolerance: float = 0.30
    safe_distance: float = 0.75
    hold_seconds: float = 0.8
    episode_seconds: float = 32.0
    high_level_decimation: int = 10
    reward_weights: NavigationRewardWeights = field(
        default_factory=NavigationRewardWeights
    )
    task_type: str = "navigation"

    def __post_init__(self):
        if (
            isinstance(self.num_robots, bool)
            or not isinstance(self.num_robots, int)
            or self.num_robots not in (2, 3, 4)
        ):
            raise ValueError("Navigation supports two to four robots")
        if (
            isinstance(self.high_level_decimation, bool)
            or not isinstance(self.high_level_decimation, int)
            or self.high_level_decimation < 1
            or self.task_type != "navigation"
        ):
            raise ValueError("Invalid navigation control settings")
        for name in (
            "start_radius",
            "goal_tolerance",
            "safe_distance",
            "episode_seconds",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("position_jitter", "yaw_jitter", "hold_seconds"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative and finite")

    @property
    def dt(self):
        return CONTROL_DT * self.high_level_decimation

    @property
    def observation_dim(self):
        return 7 + 4 * (self.num_robots - 1)

    @classmethod
    def from_checkpoint(cls, saved):
        saved = dict(saved)
        saved["reward_weights"] = NavigationRewardWeights(**saved["reward_weights"])
        return cls(**saved)


def navigation_observation(xy, yaw, velocity, goals, commands, cfg, xp):
    """Own goal/velocity and peer position/velocity, all in the observer's frame.

    Works with NumPy and Torch, including leading batch axes. No assigned path,
    optimal action, robot ID or teacher policy is included in the observation.
    """
    c, s = xp.cos(yaw), xp.sin(yaw)

    def local(vector):
        return xp.stack(
            (
                vector[..., 0] * c + vector[..., 1] * s,
                -vector[..., 0] * s + vector[..., 1] * c,
            ),
            -1,
        )

    parts = [
        local(goals - xy) / (2 * cfg.start_radius),
        local(velocity) / 0.2,
        commands,
    ]
    for offset in range(1, cfg.num_robots):
        # Cyclic ordering gives each shared actor the same self/peer convention.
        parts.extend(
            (
                local(xp.roll(xy, -offset, -2) - xy) / (2 * cfg.start_radius),
                local(xp.roll(velocity, -offset, -2)) / 0.2,
            )
        )
    return xp.concatenate(parts, -1) if xp is np else xp.cat(parts, -1)


def navigation_reward(
    before, distances, pair_distances, collision, success, failed, cfg, xp
):
    """Shared team reward; every term measures the resulting physical state."""
    weights = cfg.reward_weights
    separation = (
        xp.maximum(1 - pair_distances / cfg.safe_distance, 0)
        if xp is np
        else (1 - pair_distances / cfg.safe_distance).clamp_min(0)
    )
    return (
        weights.progress * (before.mean(-1) - distances.mean(-1))
        - weights.distance * distances.mean(-1) * cfg.dt
        - weights.separation * (separation**2).mean(-1) * cfg.dt
        - weights.collision * collision * cfg.dt
        + weights.arrival * ((distances < cfg.goal_tolerance) * 1.0).mean(-1) * cfg.dt
        + weights.success * success
        + weights.failure * failed
    )


class NavigationEnv(gym.Env):
    """Gym API: normalized body velocities (N,3), shared scalar team reward.

    Actions are [forward, lateral, yaw] in [-1,1]. Zero means stop; each component
    scales to the frozen walker's physical command limits. Goals lie opposite
    each robot's start on a circle. Their paths intersect in the center.
    """

    metadata = {"render_modes": ["rgb_array", "rgb_array_list"], "render_fps": 25}

    def __init__(self, config=None, asset_root=None, *, render_mode=None):
        if asset_root is not None and str(asset_root) != str(find_asset_root()):
            raise ValueError("Navigation uses the packaged walking assets")
        if render_mode not in (None, "rgb_array", "rgb_array_list"):
            raise ValueError("Unsupported render mode")
        self.cfg = config or NavigationConfig()
        self.render_mode = render_mode
        self.sim = WalkingSimulation(self.cfg.num_robots)
        self.model, self.data, self.walking = (
            self.sim.model,
            self.sim.data,
            self.sim._walking,
        )
        for name in (
            "base_ids",
            "gyro_indices",
            "root_q",
            "root_v",
            "joint_q",
            "joint_v",
            "actuators",
            "controller",
        ):
            setattr(self, name, getattr(self.walking, name))
        self.contacts = ContactTracker(self.model, self.cfg.num_robots)
        self.action_space = gym.spaces.Box(-1, 1, (self.cfg.num_robots, 3), np.float32)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (self.cfg.num_robots, self.cfg.observation_dim), np.float32
        )
        self.goals = np.zeros((self.cfg.num_robots, 2))
        self.last_commands = np.zeros((self.cfg.num_robots, 3), np.float32)
        self.frames = []
        self.reset(seed=0)

    @property
    def provenance(self):
        return dict(
            navigation_config=asdict(self.cfg),
            low_level_sha256=self.walking.policy_sha256,
            robot_xml_sha256=self.walking.model_sha256,
            env_version=NAVIGATION_ENV_ID,
        )

    def measurements(self):
        xy, yaw = self.sim.positions, self.sim.headings
        velocity = self.data.qvel[self.root_v[:, None] + np.arange(2)]
        distances = np.linalg.norm(self.goals - xy, axis=-1)
        a, b = np.triu_indices(self.cfg.num_robots, 1)
        pairs = np.linalg.norm(xy[a] - xy[b], axis=-1)
        return xy, yaw, velocity, distances, pairs

    def observation(self):
        xy, yaw, velocity, _, _ = self.measurements()
        return navigation_observation(
            xy, yaw, velocity, self.goals, self.normalized_commands, self.cfg, np
        ).astype(np.float32)

    def info(self):
        xy, _, _, distances, pairs = self.measurements()
        return dict(
            is_success=self.is_success,
            distance=float(distances.mean()),
            min_separation=float(pairs.min()),
            robot_positions=xy,
            goals=self.goals.copy(),
            distances=distances,
            failed=self.failed,
            robot_collision_fraction=self.contacts.robot_collision_steps
            / (PHYSICS_STEPS * self.cfg.high_level_decimation),
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        angle = self.np_random.uniform(-np.pi, np.pi)
        angles = (
            angle + np.arange(self.cfg.num_robots) * 2 * np.pi / self.cfg.num_robots
        )
        start = self.cfg.start_radius * np.column_stack(
            (np.cos(angles), np.sin(angles))
        )
        self.goals = -start
        xy = start + self.np_random.uniform(
            -self.cfg.position_jitter, self.cfg.position_jitter, start.shape
        )
        yaw = (
            angles
            + np.pi
            + self.np_random.uniform(
                -self.cfg.yaw_jitter, self.cfg.yaw_jitter, len(angles)
            )
        )
        self.sim.reset(poses=np.column_stack((xy, yaw)))
        self.contacts.reset()
        self.last_commands.fill(0)
        self.normalized_commands = np.zeros_like(self.last_commands)
        self.steps, self.success_time = 0, 0.0
        self.is_success = self.failed = self.done = False
        self.frames = []
        if self.render_mode == "rgb_array_list":
            self.frames.append(self._render_frame())
        return self.observation(), self.info()

    def step(self, action):
        if self.done:
            raise RuntimeError("Reset before stepping a finished episode")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            raise ValueError("Expected finite (num_robots,3) velocity actions")
        before = self.measurements()[3]
        self.normalized_commands = np.clip(action, -1, 1)
        self.last_commands = action_to_command(action)
        self.contacts.reset()
        self.walking.advance(
            self.last_commands,
            self.cfg.high_level_decimation,
            on_physics_step=lambda: self.contacts.record(self.model, self.data),
            on_control_step=self._record,
        )
        self.steps += 1
        xy, _, _, distances, pairs = self.measurements()
        self.failed = bool(
            np.any(self.data.xpos[self.base_ids, 2] < 0.08)
            or np.any(self.data.xmat[self.base_ids, 8] < 0.5)
            or np.any(np.abs(xy) > 3)
        )
        self.success_time = (
            self.success_time + self.cfg.dt
            if np.all(distances < self.cfg.goal_tolerance)
            else 0.0
        )
        self.is_success = bool(
            np.all(distances < self.cfg.goal_tolerance)
            and self.success_time + 1e-8 >= self.cfg.hold_seconds
            and not self.failed
        )
        terminated = self.failed or self.is_success
        truncated = self.steps * self.cfg.dt >= self.cfg.episode_seconds
        self.done = terminated or truncated
        info = self.info()
        reward = navigation_reward(
            before,
            distances,
            pairs,
            info["robot_collision_fraction"],
            self.is_success,
            self.failed,
            self.cfg,
            np,
        )
        return self.observation(), float(reward), terminated, truncated, info

    def _record(self):
        if self.render_mode == "rgb_array_list":
            self.frames.append(self._render_frame())

    def _render_frame(self):
        if self.sim._renderer is None:
            self.sim._renderer = mujoco.Renderer(self.model, height=288, width=384)
        renderer = self.sim._renderer
        camera = mujoco.MjvCamera()
        camera.lookat[:] = (0, 0, 0.1)
        camera.distance, camera.azimuth, camera.elevation = 3.8, 90, -60
        renderer.update_scene(self.data, camera)
        for xy in self.goals:
            geom = renderer.scene.geoms[renderer.scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_CYLINDER,
                np.array([self.cfg.goal_tolerance, self.cfg.goal_tolerance, 0.001]),
                np.array([*xy, 0.003]),
                np.eye(3).ravel(),
                np.array([0.25, 0.8, 0.35, 0.35], np.float32),
            )
            renderer.scene.ngeom += 1
        return renderer.render().copy()

    def render(self):
        if self.render_mode == "rgb_array_list":
            frames, self.frames = self.frames, []
            return frames
        return self._render_frame() if self.render_mode == "rgb_array" else None

    def close(self):
        self.sim.close()


def make_navigation_vector(config, num_envs, *, asynchronous=False):
    factories = [partial(NavigationEnv, config) for _ in range(num_envs)]
    kwargs = dict(autoreset_mode=gym.vector.AutoresetMode.DISABLED)
    if asynchronous:
        return gym.vector.AsyncVectorEnv(factories, context="spawn", **kwargs)
    return gym.vector.SyncVectorEnv(factories, **kwargs)


if NAVIGATION_ENV_ID not in gym.registry:
    gym.register(
        NAVIGATION_ENV_ID, entry_point="hexapod_transport_rl.navigation:NavigationEnv"
    )
