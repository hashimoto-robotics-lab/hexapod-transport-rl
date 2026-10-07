"""Shared local actor and central critic built from TorchRL's MARL modules.

Losses, GAE, sampling and collection are supplied by TorchRL. This module only
defines network architecture, Gymnasium inference and checkpoint metadata.
"""

import hashlib
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torchrl
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.envs.utils import ExplorationType
from torchrl.modules import (
    MultiAgentMLP,
    PopArtValueNorm,
    ProbabilisticActor,
    TanhNormal,
)
from torchrl.objectives import MAPPOLoss

from .approach import ApproachConfig
from .config import PushConfig
from .mappo import load_checkpoint
from .pose_push import PosePushConfig

FORMAT = "hexapod-approach-torchrl-mappo-v1"
POSE_FORMAT = "hexapod-pose-push-torchrl-mappo-v1"
NETWORK = dict(hidden_units=[128, 128], initial_log_std=-1.0)


def _native_metadata(value):
    """Keep checkpoint metadata compatible with weights_only=True loading."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _native_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_native_metadata(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_native_metadata(item) for item in value)
    return value


@dataclass(frozen=True)
class MAPPOSettings:
    """Common update settings for paired reward experiments."""

    epochs: int = 4
    minibatch_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coeff: float = 0.005
    critic_coeff: float = 0.5
    max_grad_norm: float = 0.5
    initial_log_std: float = -1.0
    value_normalization: bool = False
    final_log_std: float | None = None

    def __post_init__(self):
        if not np.isfinite(self.initial_log_std):
            raise ValueError("Initial log standard deviation must be finite")
        if self.final_log_std is not None and (
            not np.isfinite(self.final_log_std)
            or self.final_log_std > self.initial_log_std
        ):
            raise ValueError("Final exploration scale must be finite and no larger")
        if self.epochs < 1 or self.minibatch_size < 2:
            raise ValueError(
                "Positive epochs and at least two minibatch worlds required"
            )
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("Discount and GAE lambda must be in [0,1]")
        values = (
            self.learning_rate,
            self.clip_epsilon,
            self.entropy_coeff,
            self.critic_coeff,
            self.max_grad_norm,
        )
        if not np.isfinite(values).all() or min(values) < 0 or self.learning_rate == 0:
            raise ValueError("MAPPO settings must be finite and nonnegative")


def make_mappo_loss(
    actor, critic, settings: MAPPOSettings, *, value_normalizer_state=None
):
    """Configure TorchRL's MAPPO loss and multi-agent GAE, without reimplementing them."""
    loss = MAPPOLoss(
        actor,
        critic,
        functional=False,
        clip_epsilon=settings.clip_epsilon,
        entropy_coeff=settings.entropy_coeff,
        critic_coeff=settings.critic_coeff,
        loss_critic_type="smooth_l1",
        value_norm=PopArtValueNorm(shape=1) if settings.value_normalization else None,
    )
    if value_normalizer_state is not None:
        if loss.value_norm is None:
            raise ValueError("Normalizer state requires value_normalization=True")
        loss.value_norm.load_state_dict(value_normalizer_state)
    loss.set_keys(
        value=("agents", "state_value"),
        action=("agents", "action"),
        sample_log_prob=("agents", "sample_log_prob"),
        advantage=("agents", "advantage"),
        value_target=("agents", "value_target"),
    )
    loss.make_value_estimator(gamma=settings.gamma, lmbda=settings.gae_lambda)
    return loss.to(next(actor.parameters()).device)


class SharedScale(nn.Module):
    """Learn the same three exploration scales for every robot."""

    def __init__(self, initial_log_std=NETWORK["initial_log_std"]):
        super().__init__()
        if not np.isfinite(initial_log_std):
            raise ValueError("Initial log standard deviation must be finite")
        self.log_std = nn.Parameter(torch.full((3,), float(initial_log_std)))

    def forward(self, loc):
        return loc, self.log_std.exp().expand_as(loc)


@torch.no_grad()
def anneal_exploration(actor, settings: MAPPOSettings, progress: float):
    """Cap Gaussian noise after PPO updates, before the next collection.

    Progress is the fraction of the fixed training budget already collected.
    The learned mean still supplies every action; no controller is introduced.
    """
    if settings.final_log_std is None:
        return
    fraction = float(np.clip(progress, 0, 1))
    cap = settings.initial_log_std + fraction * (
        settings.final_log_std - settings.initial_log_std
    )
    for module in actor.modules():
        if isinstance(module, SharedScale):
            module.log_std.clamp_(max=cap)


def make_mappo_networks(
    num_robots=2, obs_dim=10, *, initial_log_std=NETWORK["initial_log_std"]
):
    """Return TorchRL actor and critic; each robot has three bounded actions."""
    actor_mlp = MultiAgentMLP(
        n_agent_inputs=obs_dim,
        n_agent_outputs=3,
        n_agents=num_robots,
        centralized=False,
        share_params=True,
        num_cells=NETWORK["hidden_units"],
        activation_class=nn.Tanh,
    )
    actor = ProbabilisticActor(
        module=TensorDictModule(
            nn.Sequential(actor_mlp, SharedScale(initial_log_std)),
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "loc"), ("agents", "scale")],
        ),
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[("agents", "action")],
        distribution_class=TanhNormal,
        distribution_kwargs={"low": -1.0, "high": 1.0, "event_dims": 1},
        default_interaction_type=ExplorationType.RANDOM,
        return_log_prob=True,
        log_prob_key=("agents", "sample_log_prob"),
    )
    critic = TensorDictModule(
        MultiAgentMLP(
            n_agent_inputs=obs_dim,
            n_agent_outputs=1,
            n_agents=num_robots,
            centralized=True,
            share_params=True,
            num_cells=NETWORK["hidden_units"],
            activation_class=nn.Tanh,
        ),
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "state_value")],
    )
    return actor, critic


@torch.inference_mode()
def policy_action(actor, observation) -> np.ndarray:
    """Deterministic normalized (N,3) commands from a Gymnasium observation.

    TanhNormal's deterministic sample is tanh(loc), not its analytic mean.
    This keeps evaluation actions bounded without adding any navigation rules.
    """
    td = TensorDict(
        {
            "agents": {
                "observation": torch.as_tensor(
                    observation,
                    dtype=torch.float32,
                    device=next(actor.parameters()).device,
                )
            }
        },
        batch_size=list(np.shape(observation)[:-2]),
    )
    return actor.get_dist(td).deterministic_sample.cpu().numpy()


def save_mappo(
    path: str | Path,
    actor,
    critic,
    optimizer,
    *,
    config: ApproachConfig | PosePushConfig,
    pushing_checkpoint: str | Path | None = None,
    provenance: dict,
    training: dict,
    curriculum: dict,
    loss=None,
) -> Path:
    """Save a pose policy, or a historical approach policy with its frozen pusher."""
    path = Path(path).resolve()
    normalizer = None if loss is None else loss.value_norm
    if training.get("settings", {}).get("value_normalization") and normalizer is None:
        raise ValueError("Pass loss to save its value normalizer for resuming")
    if isinstance(config, PosePushConfig):
        if PosePushConfig(**provenance["pose_config"]) != config:
            raise ValueError("Saved pose config differs from the training world")
        extra = dict(pose_config=asdict(config))
        file_format, num_robots = POSE_FORMAT, config.num_robots
        obs_dim = 18 + 4 * (num_robots - 1)
    else:
        pushing_checkpoint = Path(pushing_checkpoint).resolve()
        actual_config = ApproachConfig(**provenance["approach_config"])
        if replace(config, layout=actual_config.layout) != actual_config:
            raise ValueError(
                "Saved reward/reset settings differ from the training world"
            )
        with torch.random.fork_rng(devices=[]):
            _, pushing_saved = load_checkpoint(pushing_checkpoint)
        pushing_config = PushConfig.from_checkpoint(pushing_saved["config"])
        if pushing_config.num_robots != 2 or pushing_config.shape != "T":
            raise ValueError("Expected the frozen two-robot T pusher")
        for key in ("low_level_sha256", "robot_xml_sha256"):
            if provenance[key] != pushing_saved[key]:
                raise ValueError("Navigation and pushing assets differ")
        extra = dict(
            approach_config=asdict(config),
            pushing_checkpoint=os.path.relpath(pushing_checkpoint, path.parent),
            pushing_sha256=hashlib.sha256(pushing_checkpoint.read_bytes()).hexdigest(),
        )
        file_format, num_robots, obs_dim = FORMAT, 2, 10
    saved = dict(
        format=file_format,
        torchrl=torchrl.__version__,
        network=NETWORK,
        num_robots=num_robots,
        obs_dim=obs_dim,
        actor=actor.state_dict(),
        critic=critic.state_dict(),
        optimizer=optimizer.state_dict(),
        training=training,
        curriculum=curriculum,
        run=dict(provenance=provenance, algorithm="TorchRL MAPPO"),
        torch_rng_state=torch.get_rng_state(),
        cuda_rng_state=(
            torch.cuda.get_rng_state(next(actor.parameters()).device)
            if next(actor.parameters()).is_cuda
            else None
        ),
        value_normalizer=None if normalizer is None else normalizer.state_dict(),
        **extra,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(_native_metadata(saved), temporary)
    temporary.replace(path)
    return path


def load_mappo(path: str | Path, *, device="cpu"):
    """Load TorchRL weights and metadata; historical approach files verify their pusher."""
    path = Path(path).resolve()
    saved = torch.load(path, weights_only=True, map_location="cpu")
    if saved.get("network") != NETWORK:
        raise ValueError("Unrecognized TorchRL network architecture")
    if saved.get("format") == POSE_FORMAT:
        config = PosePushConfig(**saved["pose_config"])
        if saved.get("num_robots") != config.num_robots or saved.get(
            "obs_dim"
        ) != 18 + 4 * (config.num_robots - 1):
            raise ValueError("Pose checkpoint has an inconsistent observation layout")
    elif saved.get("format") == FORMAT:
        if saved.get("num_robots") != 2 or saved.get("obs_dim") != 10:
            raise ValueError("Expected a two-robot TorchRL approach checkpoint")
        pusher = (path.parent / saved["pushing_checkpoint"]).resolve()
        if hashlib.sha256(pusher.read_bytes()).hexdigest() != saved["pushing_sha256"]:
            raise ValueError("Frozen pushing checkpoint has changed")
        saved["pushing_checkpoint"] = str(pusher)
    else:
        raise ValueError("Unrecognized TorchRL checkpoint format")
    actor, critic = make_mappo_networks(saved["num_robots"], saved["obs_dim"])
    actor.load_state_dict(saved["actor"])
    critic.load_state_dict(saved["critic"])
    if any(
        not torch.isfinite(v).all()
        for model in (actor, critic)
        for v in model.state_dict().values()
    ):
        raise ValueError("Non-finite MAPPO weights")
    return actor.to(device).eval(), critic.to(device).eval(), saved


class TorchRLNavigator:
    """Use the TorchRL actor in the unchanged transport evaluation pipeline."""

    def __init__(self, actor):
        self.actor = actor

    def act(self, observations):
        return policy_action(self.actor, observations)


def load_torchrl_transport(path: str | Path):
    actor, _, saved = load_mappo(path)
    pusher, _ = load_checkpoint(saved["pushing_checkpoint"])
    return TorchRLNavigator(actor), pusher, saved
