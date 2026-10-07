"""Short navigation episodes before handing over to a frozen pushing actor.

Only reset edits poses. Actors learn from reward, without demonstration actions.
Deployed actors receive geometry/velocity observations and choose all velocities.
"""

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .config import COMMAND_LIMIT, STAND_HEIGHT, PushConfig, rotation, wrap
from .env import PushEnv
from .types import FloatArray, Info, Observation, ResetResult, StepResult

SIDES = np.array([-1, 1])  # robot_0は右側、robot_1は左側を担当する。
OBS_DIM = 10
CURRICULUM_LAYOUTS = ("near", "rear", "side", "front")
LAYOUT_TIME_LIMITS = {"near": 10, "rear": 20, "side": 40, "front": 65}

# T座標での担当位置。左右を反転した観測では、両機が同じ+Y位置を目指す。
HANDOVER_POSITION = np.array([-1.05, 0.325])
HANDOVER_POSITION_TOLERANCE_M = 0.095
HANDOVER_YAW_TOLERANCE_RAD = 0.15
HANDOVER_HOLD_STEPS = 2

# 報酬距離の計算で使う、Tと機体の接触を避けるための余白付き領域 [m]。
CLEARANCE_LOWER = np.array([-0.90, -1.05])
CLEARANCE_UPPER = np.array([0.55, 1.05])


@dataclass(frozen=True)
class ApproachRewardWeights:
    """回り込み報酬の係数。報酬設計の実験ではここを変更する。"""

    progress: float = 6.0
    time: float = 0.04
    robot_contact: float = 4.0
    body_contact: float = 6.0
    cargo_displacement: float = 0.1
    success: float = 15.0
    failure: float = -10.0


REWARD_WEIGHTS = ApproachRewardWeights()


@dataclass(frozen=True)
class ApproachConfig:
    """回り込みの初期配置と制限時間。角度はdegrees/radiansの区別に注意。"""

    seconds: float = 65.0
    max_yaw_degrees: float = 30.0
    position_jitter: float = 0.08
    yaw_jitter: float = 0.15
    layout: str = "mixed"
    easier_reset_fraction: float = 0.0

    def __post_init__(self) -> None:
        if self.layout not in ("near", "front", "side", "rear", "mixed"):
            raise ValueError("Unknown approach layout")
        values = (
            self.seconds,
            self.max_yaw_degrees,
            self.position_jitter,
            self.yaw_jitter,
        )
        if not np.isfinite(values).all() or min(values) < 0 or self.seconds <= 0:
            raise ValueError("Approach settings must be finite and nonnegative")
        if not 0 <= self.easier_reset_fraction <= 1:
            raise ValueError("easier_reset_fraction must be in [0,1]")


def cargo_frame(env: PushEnv) -> tuple[np.ndarray, np.ndarray]:
    """Reflect each robot's assigned side to +Y; neither side is privileged."""
    positions = (env.robot_xy - env.cargo_xy) @ rotation(env.cargo_yaw)
    positions[:, 1] *= SIDES
    yaws = wrap(env.yaw(env.base_ids) - env.cargo_yaw) * SIDES
    return positions, yaws


def observe_approach(env: PushEnv) -> Observation:
    """各機の観測 (2,10)。T基準の位置と向き、機体速度、前回指令。

    0:2=位置/2m、2:4=yaw差のsin/cos、4:7=速度、7:10=正規化指令。
    左右・旋回の符号を反転し、共有actorに同じ役割として入力する。
    """
    positions, yaws = cargo_frame(env)
    world_yaws = env.yaw(env.base_ids)
    velocities = env.data.qvel[env.root_v[:, None] + np.array([0, 1])]
    cos_yaw, sin_yaw = np.cos(world_yaws), np.sin(world_yaws)
    body_velocity = np.column_stack(
        (
            cos_yaw * velocities[:, 0] + sin_yaw * velocities[:, 1],
            -sin_yaw * velocities[:, 0] + cos_yaw * velocities[:, 1],
            env.data.qvel[env.root_v + 5],
        )
    )
    result = np.column_stack(
        (
            positions / 2,
            np.sin(yaws),
            np.cos(yaws),
            body_velocity,
            env.last_commands / COMMAND_LIMIT,
        )
    )
    result[:, [5, 6, 8, 9]] *= SIDES[:, None]
    return result.astype(np.float32)


def physical_actions(canonical_actions: FloatArray) -> FloatArray:
    """共有actorの行動 (2,3) を各機の実際の前後・左右・旋回方向へ戻す。"""
    actions = np.asarray(canonical_actions, dtype=np.float32).copy()
    actions[:, 1:] *= SIDES[:, None]
    return actions


def approach_ready(env: PushEnv) -> np.ndarray:
    """各機が押し開始位置と向きに入ったかを返す bool (2,)。"""
    positions, yaws = cargo_frame(env)
    return (
        np.linalg.norm(positions - HANDOVER_POSITION, axis=1)
        < HANDOVER_POSITION_TOLERANCE_M
    ) & (np.abs(yaws) < HANDOVER_YAW_TOLERANCE_RAD)


def segment_clear(start: np.ndarray, end: np.ndarray) -> bool:
    """Whether a segment avoids the open, clearance-expanded cargo rectangle."""
    low, high = CLEARANCE_LOWER, CLEARANCE_UPPER
    direction = end - start
    enter, leave = 0.0, 1.0
    for axis in range(2):
        if abs(direction[axis]) < 1e-9:
            if start[axis] <= low[axis] + 1e-8 or start[axis] >= high[axis] - 1e-8:
                return True
        else:
            times = sorted(
                [
                    (low[axis] - start[axis]) / direction[axis],
                    (high[axis] - start[axis]) / direction[axis],
                ]
            )
            enter, leave = max(enter, times[0]), min(leave, times[1])
    return enter >= leave - 1e-8


def route_distance(position: np.ndarray) -> float:
    """Shortest distance via visible corners on the assigned side, for reward.

    Unlike a straight-line potential, this gives credit for initially moving
    outward to clear the T. It supplies no actions or waypoints to the actor.
    """
    target = HANDOVER_POSITION
    rear = np.array([CLEARANCE_LOWER[0], CLEARANCE_UPPER[1]])
    front = CLEARANCE_UPPER
    candidates = []
    for end, tail in (
        (target, 0.0),
        (rear, np.linalg.norm(rear - target)),
        (front, np.linalg.norm(front - rear) + np.linalg.norm(rear - target)),
    ):
        if segment_clear(position, end):
            candidates.append(np.linalg.norm(position - end) + tail)
    if candidates:
        return min(candidates)
    # An inside point pays for leaving the margin before continuing around it.
    # Penalizing penetration avoids a shortcut through the expanded rectangle.
    exits = (
        np.array([CLEARANCE_LOWER[0], position[1]]),
        np.array([CLEARANCE_UPPER[0], position[1]]),
        np.array([position[0], CLEARANCE_UPPER[1]]),
    )
    return min(
        5 * np.linalg.norm(position - point) + route_distance(point) for point in exits
    )


def route_potential(env: PushEnv) -> float:
    """報酬に使う負の残距離。後方へ来るほど向きの整列も評価する。"""
    positions, yaws = cargo_frame(env)
    return -float(
        np.mean(
            [
                route_distance(position)
                + 0.25 * abs(yaw) * np.clip((-position[0] - 0.7) / 0.4, 0, 1)
                for position, yaw in zip(positions, yaws, strict=True)
            ]
        )
    )


def reset_approach(
    env: PushEnv, config: ApproachConfig, seed: int | None = None
) -> Observation:
    """Random headings and starts at reset; no scripted motion or external forces."""
    if env.cfg.num_robots != 2 or env.cfg.shape != "T":
        raise ValueError("Approach task requires two hexapod robots and a T")
    env.reset(seed=seed, randomize=True)
    yaw = env.goal_yaw + env.rng.uniform(-1, 1) * np.radians(config.max_yaw_degrees)
    env.data.qpos[env.cargo_q + 3 : env.cargo_q + 7] = (
        np.cos(yaw / 2),
        0,
        0,
        np.sin(yaw / 2),
    )
    layout = config.layout
    if layout == "mixed":
        layout = env.rng.choice(["front", "side", "rear"], p=[0.5, 0.25, 0.25])
    level = CURRICULUM_LAYOUTS.index(layout)
    if level > 0 and env.rng.random() < config.easier_reset_fraction:
        layout = env.rng.choice(CURRICULUM_LAYOUTS[:level])
    env.approach_layout = layout
    for robot_index, address in enumerate(env.root_q):
        xy, heading = _sample_start_pose(env.rng, layout, SIDES[robot_index])
        xy += env.rng.uniform(-config.position_jitter, config.position_jitter, 2)
        heading += yaw + env.rng.uniform(-config.yaw_jitter, config.yaw_jitter)
        xy = rotation(yaw) @ xy
        env.data.qpos[address : address + 7] = (
            *xy,
            STAND_HEIGHT,
            np.cos(heading / 2),
            0,
            0,
            np.sin(heading / 2),
        )
    mujoco.mj_forward(env.model, env.data)
    return observe_approach(env)


def _sample_start_pose(
    rng: np.random.Generator, layout: str, side: int
) -> tuple[np.ndarray, float]:
    """初期配置ごとの位置 [m] と向き [rad]。ばらつきの付加前の値。"""
    if layout == "front":
        return np.array([0.65, 0.65 * side]), side * np.pi / 2
    if layout == "side":
        return (
            np.array([rng.uniform(-0.7, 0.6), 1.15 * side]),
            side * rng.uniform(np.pi / 2, np.pi),
        )
    if layout == "rear":
        return np.array([-1.40, 0.325 * side]), rng.uniform(-0.8, 0.8)
    return np.array([-1.25, 0.325 * side]), rng.uniform(-0.2, 0.2)


def _approach_reward(
    env: PushEnv, previous_potential: float, info: Info, success: bool
) -> float:
    """前後の状態から、進捗報酬と各ペナルティを同じ順序で合算する。"""
    progress_reward = REWARD_WEIGHTS.progress * (
        route_potential(env) - previous_potential
    )
    time_penalty = REWARD_WEIGHTS.time * env.cfg.dt
    robot_contact_penalty = (
        REWARD_WEIGHTS.robot_contact * env.cfg.dt * info["robot_collision_fraction"]
    )
    body_contact_penalty = (
        REWARD_WEIGHTS.body_contact
        * env.cfg.dt
        * np.mean(info["body_contact_fraction"])
    )
    displacement_penalty = (
        REWARD_WEIGHTS.cargo_displacement * info["cargo_displacement"]
    )
    if success:
        terminal_reward = REWARD_WEIGHTS.success
    elif info["failed"]:
        terminal_reward = REWARD_WEIGHTS.failure
    else:
        terminal_reward = 0.0
    return float(
        progress_reward
        - time_penalty
        - robot_contact_penalty
        - body_contact_penalty
        - displacement_penalty
        + terminal_reward
    )


class ApproachEnv(gym.Env):
    """Navigation-only MARL task: finish when both robots can start pushing.

    Stopping at handover avoids repeatedly simulating the already learned push.
    The physical time step and foot-contact model match the original task.
    """

    metadata = {"render_modes": []}

    def __init__(
        self, config: ApproachConfig | None = None, asset_root: str | Path | None = None
    ) -> None:
        self.config = config or ApproachConfig()
        self.core = PushEnv(
            PushConfig(shape="T", episode_seconds=self.config.seconds), asset_root
        )
        self.observation_space = spaces.Box(-np.inf, np.inf, (2, OBS_DIM), np.float32)
        self.action_space = spaces.Box(-1, 1, (2, 3), np.float32)
        self.ready_steps = 0

    @property
    def provenance(self) -> Info:
        return dict(
            asset_root=str(self.core.asset_root),
            low_level_sha256=self.core.policy_sha256,
            robot_xml_sha256=self.core.model_sha256,
            approach_config=asdict(self.config),
        )

    def set_layout(self, layout: str) -> None:
        self.config = replace(self.config, layout=layout)

    def reset(self, *, seed=None, options=None) -> ResetResult:
        super().reset(seed=seed)
        if seed is not None:
            self.core.rng = np.random.default_rng(seed)
        self.ready_steps = 0
        obs = reset_approach(self.core, self.config)
        return obs, {"is_success": False}

    def step(self, action: FloatArray) -> StepResult:
        """行動の検証 → 物理更新 → 整列判定 → 報酬 → 終了判定。"""
        if not self.action_space.contains(np.asarray(action, dtype=np.float32)):
            raise ValueError("Expected canonical actions (2,3) in [-1,1]")
        previous_potential = route_potential(self.core)
        _, _, terminated, truncated, info = self.core.step(physical_actions(action))
        ready = bool(approach_ready(self.core).all())
        self.ready_steps = self.ready_steps + 1 if ready else 0
        success = self.ready_steps >= HANDOVER_HOLD_STEPS and not info["failed"]
        reward = _approach_reward(self.core, previous_potential, info, success)
        terminated = bool(terminated or success)
        limit = min(
            self.config.seconds,
            LAYOUT_TIME_LIMITS[self.core.approach_layout],
        )
        truncated = bool(
            (truncated or self.core.steps * self.core.cfg.dt >= limit)
            and not terminated
        )
        self.core.done = terminated or truncated
        info.update(is_success=success, approach_success=success)
        return observe_approach(self.core), float(reward), terminated, truncated, info


def worker_env(
    config: ApproachConfig, asset_root: str | Path | None = None
) -> ApproachEnv:
    import torch

    torch.set_num_threads(1)
    return ApproachEnv(config, asset_root)


def make_approach_vector(
    config: ApproachConfig,
    num_envs: int = 8,
    asset_root: str | Path | None = None,
    asynchronous: bool = True,
) -> gym.vector.VectorEnv:
    """各世界に2台を置き、独立した世界をまとめて進める。"""
    from functools import partial

    factories = [partial(worker_env, config, asset_root) for _ in range(num_envs)]
    kwargs = dict(autoreset_mode=gym.vector.AutoresetMode.DISABLED, copy=True)
    if asynchronous:
        return gym.vector.AsyncVectorEnv(factories, context="spawn", **kwargs)
    return gym.vector.SyncVectorEnv(factories, **kwargs)
