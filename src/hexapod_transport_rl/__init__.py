"""Student API for single-hexapod cylinder pushing with TorchRL PPO."""

from .single_push import SinglePushConfig, SinglePushEnv
from .student import (
    ContinuousVelocity,
    CylinderPushTask,
    DiscreteDirections,
    ObservationTerm,
    PushState,
    RewardTerm,
    TrainingRun,
    available_observations,
    available_rewards,
    compare,
    evaluate,
    load_task,
    load_task_file,
    record_video,
    train,
)
from .torchrl_ppo import PPOSettings

__all__ = [
    "CylinderPushTask",
    "ObservationTerm",
    "RewardTerm",
    "DiscreteDirections",
    "ContinuousVelocity",
    "PushState",
    "TrainingRun",
    "PPOSettings",
    "train",
    "evaluate",
    "compare",
    "record_video",
    "load_task",
    "load_task_file",
    "available_observations",
    "available_rewards",
    "SinglePushConfig",
    "SinglePushEnv",
]
