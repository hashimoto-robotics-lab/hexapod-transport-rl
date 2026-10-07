"""Task settings and the fixed deployment contract of the learned walking policy.

PushConfig is serialized into checkpoints: keep field names/defaults compatible.
Lengths are meters, angles radians, time seconds, mass kilograms.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from .rewards import PushRewardWeights
from .types import FloatArray

# Joint order and neutral pose must match the exported walking controller.
LEGS = ("LF", "LM", "LH", "RF", "RM", "RH")
JOINT_NAMES = tuple(
    f"{leg}_{part}_joint" for leg in LEGS for part in ("coxa", "femur", "tibia")
)
STAND = np.tile([0.0, 0.45, 0.90], 6)
STAND_HEIGHT = 0.16
# Body-frame velocity limits: forward, lateral, yaw. Reverse speed is asymmetric.
COMMAND_LOW = np.array([-0.15, -0.10, -0.60], dtype=np.float32)
COMMAND_HIGH = np.array([0.20, 0.10, 0.60], dtype=np.float32)
COMMAND_LIMIT = np.maximum(-COMMAND_LOW, COMMAND_HIGH)
# One walking tick is eight physics ticks; a high-level step holds several ticks.
PHYSICS_DT = 0.005
CONTROL_DT = 0.04
PHYSICS_STEPS = 8
ENV_VERSION = "hexapod-leg-push-v1"
LEGACY_ENV_VERSION = "sixtrail-leg-push-v1"
MODEL_RELATIVE_PATH = "assets/robot/robot.xml"
POLICY_RELATIVE_PATH = "assets/locomotion"


def action_to_command(action: ArrayLike) -> FloatArray:
    """Map normalized (...,3) actions to [vx m/s, vy m/s, yaw rad/s].

    Scale each sign separately so a zero action remains a stop command.
    """
    action = np.clip(np.asarray(action, dtype=np.float32), -1, 1)
    return action * np.where(action >= 0, COMMAND_HIGH, -COMMAND_LOW)


def command_to_action(command: ArrayLike) -> FloatArray:
    """Convert physical velocity commands to clipped normalized (...,3) actions."""
    command = np.asarray(command, dtype=np.float32)
    return np.clip(command / np.where(command >= 0, COMMAND_HIGH, -COMMAND_LOW), -1, 1)


def find_asset_root() -> Path:
    """インストールしたパッケージに同梱したロボットと歩行モデルを使う。"""
    root = Path(__file__).resolve().parent
    if not (root / MODEL_RELATIVE_PATH).is_file():
        raise FileNotFoundError(
            "Bundled robot assets are missing; reinstall the package"
        )
    return root


@dataclass(frozen=True)
class PushConfig:
    """Physical task and episode settings, shared by all API adapters.

    ``shape`` defaults to a box; use "T" explicitly for T-shaped cargo.
    ``high_level_decimation`` is walking-policy ticks per high-level action.
    ``push_gap`` is rear cargo face to desired robot center, not foot clearance.
    """

    # Team and cargo.
    num_robots: int = 2
    shape: str = "box"
    cargo_mass: float = 2.0
    cargo_friction: float = 0.3
    # Goal, episode horizon and control timing.
    goal_distance: float = 2.0
    episode_seconds: float = 40.0
    high_level_decimation: int = 5
    # Success thresholds: m, rad, seconds held inside tolerance.
    position_tolerance: float = 0.18
    yaw_tolerance: float = 0.25
    success_hold_seconds: float = 0.5
    # Arena and desired pushing formation, in meters.
    arena_radius: float = 8.0
    push_gap: float = 0.30
    reward_weights: PushRewardWeights = field(default_factory=PushRewardWeights)

    def __post_init__(self) -> None:
        if isinstance(self.reward_weights, dict):
            object.__setattr__(
                self, "reward_weights", PushRewardWeights(**self.reward_weights)
            )
        if not isinstance(self.num_robots, int) or not 2 <= self.num_robots <= 4:
            raise ValueError("num_robots must be an integer between 2 and 4")
        if self.shape not in ("box", "T"):
            raise ValueError("shape must be box or T")
        if (
            not isinstance(self.high_level_decimation, int)
            or self.high_level_decimation < 1
        ):
            raise ValueError("high_level_decimation must be a positive integer")
        values = (
            self.cargo_mass,
            self.cargo_friction,
            self.goal_distance,
            self.episode_seconds,
            self.position_tolerance,
            self.yaw_tolerance,
            self.success_hold_seconds,
            self.arena_radius,
            self.push_gap,
        )
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError(
                "Mass, friction, distances, tolerances and durations must be positive and finite"
            )
        if self.goal_distance + self.width / 2 + 1 >= self.arena_radius:
            raise ValueError("Goal must fit within the arena")

    @property
    def dt(self) -> float:
        """Simulated seconds per high-level step (default 0.2 s)."""
        return CONTROL_DT * self.high_level_decimation

    @property
    def width(self) -> float:
        """Cargo extent along its local Y axis, wide enough for N robots."""
        return self.num_robots * 0.65

    @property
    def depth(self) -> float:
        """Cargo extent along its local X axis."""
        return 0.45 if self.shape == "box" else 0.7

    @property
    def cargo_height(self) -> float:
        return 0.25

    @property
    def slots(self) -> np.ndarray:
        """Assigned lateral push offsets (N,) in cargo coordinates, in meters."""
        return np.linspace(
            -0.325 * (self.num_robots - 1),
            0.325 * (self.num_robots - 1),
            self.num_robots,
        )


def rotation(yaw: float) -> np.ndarray:
    """2D rotation; a world-frame row vector @ rotation(yaw) is body-frame."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def wrap(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap a radian angle or array to [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi
