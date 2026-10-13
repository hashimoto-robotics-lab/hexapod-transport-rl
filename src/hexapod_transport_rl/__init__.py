"""Cooperative navigation and pushing with articulated hexapod robots."""

from .api import GYM_ENV_ID, HexapodPushEnv
from .approach import APPROACH_ENV_ID, ApproachConfig, ApproachEnv
from .approach_evaluation import evaluate as evaluate_transport
from .approach_training import train_approach
from .config import PushConfig
from .env import PushEnv
from .evaluation import evaluate_mappo
from .navigation import (
    NAVIGATION_ENV_ID,
    NavigationConfig,
    NavigationEnv,
    NavigationRewardWeights,
)
from .pose_evaluation import evaluate_pose, rule_action
from .pose_push import POSE_ENV_ID, POSE_STAGES, PosePushConfig, PosePushEnv, PoseStage
from .pose_training import PoseCurriculum
from .pusht import PUSHT_ENV_ID, PushTConfig, PushTEnv, PushTRewardWeights
from .rewards import ApproachRewardWeights, PoseRewardWeights, PushRewardWeights
from .torchrl_env import TorchRLTransportEnv
from .torchrl_mappo import (
    MAPPOSettings,
    anneal_exploration,
    load_mappo,
    make_mappo_loss,
    make_mappo_networks,
    make_pusht_networks,
    policy_action,
    save_mappo,
)
from .torchrl_training import ApproachCurriculum
from .training import train_mappo
from .vector import make_vector_env
from .walking import WalkingSimulation
from .walking_env import WALKING_ENV_ID, WalkingEnv

__all__ = [
    "NavigationConfig",
    "NavigationEnv",
    "NavigationRewardWeights",
    "NAVIGATION_ENV_ID",
    "PushTConfig",
    "train_mappo",
    "evaluate_mappo",
    "PushTEnv",
    "PushTRewardWeights",
    "PUSHT_ENV_ID",
    "WalkingSimulation",
    "PosePushConfig",
    "PoseCurriculum",
    "evaluate_pose",
    "rule_action",
    "PosePushEnv",
    "PoseStage",
    "POSE_STAGES",
    "POSE_ENV_ID",
    "PoseRewardWeights",
    "TorchRLTransportEnv",
    "MAPPOSettings",
    "make_mappo_networks",
    "make_pusht_networks",
    "anneal_exploration",
    "make_mappo_loss",
    "policy_action",
    "save_mappo",
    "load_mappo",
    "ApproachCurriculum",
    "WalkingEnv",
    "WALKING_ENV_ID",
    "APPROACH_ENV_ID",
    "ApproachConfig",
    "ApproachEnv",
    "ApproachRewardWeights",
    "PushRewardWeights",
    "train_approach",
    "evaluate_transport",
    "PushConfig",
    "PushEnv",
    "GYM_ENV_ID",
    "HexapodPushEnv",
    "make_vector_env",
]
