"""Experiment parameters and the reward actually used to learn navigation.

Coefficients belong to each environment, so parallel experiments cannot change
one another's rewards. Negative terminal failure rewards retain their sign.
"""

from dataclasses import astuple, dataclass
from math import isfinite


@dataclass(frozen=True)
class ApproachRewardWeights:
    """回り込み報酬。ペナルティ係数は正、失敗時の報酬は負で指定する。"""

    progress: float = 6.0
    time: float = 0.04
    robot_contact: float = 4.0
    body_contact: float = 6.0
    cargo_displacement: float = 0.1
    success: float = 15.0
    failure: float = -10.0

    def __post_init__(self):
        if not all(isfinite(value) for value in astuple(self)):
            raise ValueError("Reward coefficients must be finite")


@dataclass(frozen=True)
class PushRewardWeights:
    """運搬報酬。距離はm、向きはrad、時間・接触はdtに比例する。"""

    progress: float = 8.0
    approach: float = 0.5
    command_change: float = 0.01
    time: float = 0.01
    robot_contact: float = 0.5
    success: float = 20.0
    failure: float = -10.0
    orientation: float = 1.0
    body_contact: float = 1.0

    def __post_init__(self):
        if not all(isfinite(value) for value in astuple(self)):
            raise ValueError("Reward coefficients must be finite")


def approach_reward_terms(
    weights: ApproachRewardWeights,
    *,
    progress: float,
    dt: float,
    robot_contact: float,
    body_contact: float,
    cargo_displacement: float,
    success: bool,
    failed: bool,
) -> dict[str, float]:
    """実際の環境が使う式。progressは経路距離の減少、接触は時間割合。

    チーム報酬はこの辞書の値の合計。式そのものを研究する場合は、
    この関数を変更して新しい実験として学習する。
    """
    return {
        "progress": weights.progress * progress,
        "time": -weights.time * dt,
        "robot_contact": -weights.robot_contact * dt * robot_contact,
        "body_contact": -weights.body_contact * dt * body_contact,
        "cargo_displacement": -weights.cargo_displacement * cargo_displacement,
        "terminal": weights.success if success else weights.failure if failed else 0.0,
    }


@dataclass(frozen=True)
class PoseRewardWeights:
    """Target-pose improvements and settling; independent of actions from demos."""

    position: float = 4.0
    position_error: float = 8.0
    orientation: float = 3.0
    orientation_error: float = 0.0
    approach: float = 4.0
    approach_error: float = 1.0
    push_heading: float = 2.0
    settling: float = 1.0
    command_change: float = 0.01
    time: float = 0.02
    robot_contact: float = 0.5
    body_contact: float = 6.0
    success: float = 50.0
    failure: float = -500.0

    def __post_init__(self):
        if not all(isfinite(value) for value in astuple(self)):
            raise ValueError("Reward coefficients must be finite")
