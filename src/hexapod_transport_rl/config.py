"""Task settings and the fixed deployment contract of the learned walking policy.

PushConfig is serialized into checkpoints: keep field names/defaults compatible.
Lengths are meters, angles radians, time seconds, mass kilograms.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from .types import FloatArray

LEGS = ("LF", "LM", "LH", "RF", "RM", "RH")
JOINT_NAMES = tuple(
    (f"{leg}_{part}_joint" for leg in LEGS for part in ("coxa", "femur", "tibia"))
)
STAND = np.tile([0.0, 0.45, 0.9], 6)
STAND_HEIGHT = 0.16
COMMAND_LOW = np.array([-0.15, -0.1, -0.6], dtype=np.float32)
COMMAND_HIGH = np.array([0.2, 0.1, 0.6], dtype=np.float32)
COMMAND_LIMIT = np.maximum(-COMMAND_LOW, COMMAND_HIGH)
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
    num_robots: int = 1
    shape: str = "cylinder"

    def __post_init__(self):
        if self.num_robots != 1 or self.shape != "cylinder":
            raise ValueError("This package supports one hexapod and cylindrical cargo")
        values = (
            self.cargo_mass,
            self.cargo_friction,
            self.cargo_radius,
            self.goal_distance,
            self.episode_seconds,
            self.position_tolerance,
            self.success_hold_seconds,
            self.arena_radius,
            self.push_gap,
        )
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError(
                "Physical dimensions, mass, friction and durations must be positive and finite"
            )
        if (
            not isinstance(self.high_level_decimation, int)
            or self.high_level_decimation < 1
        ):
            raise ValueError("high_level_decimation must be a positive integer")

    @property
    def width(self):
        return 2 * self.cargo_radius

    @property
    def depth(self):
        return self.width

    @property
    def rear_face(self):
        return -self.cargo_radius

    @property
    def dt(self) -> float:
        """Simulated seconds per high-level step (default 0.2 s)."""
        return CONTROL_DT * self.high_level_decimation

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
    c, s = (np.cos(yaw), np.sin(yaw))
    return np.array([[c, -s], [s, c]])


def wrap(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap a radian angle or array to [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi
