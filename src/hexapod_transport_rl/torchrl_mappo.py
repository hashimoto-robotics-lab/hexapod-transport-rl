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
from .velocity_distribution import VelocityDistribution

FORMAT = "hexapod-approach-torchrl-mappo-v1"
POSE_FORMAT = "hexapod-pose-push-torchrl-mappo-v1"
PUSHT_FORMAT = "hexapod-pusht-torchrl-mappo-v1"
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
    fixed_exploration: bool = False

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
    for module in actor.modules():
        if isinstance(module, SharedScale):
            module.log_std.requires_grad_(not settings.fixed_exploration)
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
    """Learn one exploration scale per action component, shared across robots."""

    def __init__(self, initial_log_std=NETWORK["initial_log_std"], action_dim=3):
        super().__init__()
        if not np.isfinite(initial_log_std):
            raise ValueError("Initial log standard deviation must be finite")
        self.log_std = nn.Parameter(torch.full((action_dim,), float(initial_log_std)))

    def forward(self, loc):
        return loc, self.log_std.exp().expand_as(loc)


class PositionTargetActor(nn.Module):
    """Explore around own XY while still emitting absolute world XY targets.

    The learned residual is unconstrained before tanh. Only own position centers
    its distribution; there is no cargo-dependent action or suggested route.
    """

    def __init__(self, network, position_index, low, high, initial_log_std):
        super().__init__()
        self.network = network
        self.scale = SharedScale(initial_log_std, 2)
        self.position_index = position_index
        self.midpoint, self.radius = (high + low) / 2, (high - low) / 2

    def forward(self, observation):
        position = observation[..., self.position_index : self.position_index + 2]
        normalized = ((position - self.midpoint) / self.radius).clamp(-0.999, 0.999)
        loc = normalized.atanh() + self.network(observation)
        return self.scale(loc)


class InputScale(nn.Module):
    """Normalize network inputs; the public observation keeps meters and radians."""

    def __init__(self, scale):
        super().__init__()
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))

    def forward(self, observation):
        return observation * self.scale


@torch.no_grad()
def anneal_exploration(
    actor, settings: MAPPOSettings, progress: float, *, restart=False
):
    """Cap Gaussian noise after PPO updates, before the next collection.

    Progress may describe the full budget or the current curriculum stage.
    Restart reopens exploration at a stage change, preserving learned means.
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
            if restart:
                module.log_std.fill_(settings.initial_log_std)
            module.log_std.clamp_(max=cap)


def make_mappo_networks(
    num_robots=2,
    obs_dim=10,
    *,
    initial_log_std=NETWORK["initial_log_std"],
    action_grid=None,
    action_dim=3,
    action_low=-1.0,
    action_high=1.0,
    position_index=None,
    observation_scale=None,
):
    """Return a shared actor and central critic with explicit action dimensions.

    An action_grid selects velocity candidates with categorical probabilities.
    Omitting it preserves the continuous Gaussian policy used by older models.
    """
    if (
        action_dim < 1
        or not np.isfinite([action_low, action_high]).all()
        or action_low >= action_high
    ):
        raise ValueError("Action dimensions and bounds must be valid")
    if action_grid is not None and (
        action_dim != 3 or action_low != -1 or action_high != 1
    ):
        raise ValueError("Velocity grids require three normalized velocity commands")
    if position_index is not None and (
        action_dim != 2
        or action_grid is not None
        or not 0 <= position_index <= obs_dim - 2
    ):
        raise ValueError("Position-centered targets require own XY and two actions")
    if observation_scale is not None and (
        len(observation_scale) != obs_dim
        or not np.isfinite(observation_scale).all()
        or min(observation_scale) <= 0
    ):
        raise ValueError(
            "Input scales must be positive and match observation dimensions"
        )
    actor_mlp = MultiAgentMLP(
        n_agent_inputs=obs_dim,
        n_agent_outputs=action_dim if action_grid is None else 3 * len(action_grid),
        n_agents=num_robots,
        centralized=False,
        share_params=True,
        num_cells=NETWORK["hidden_units"],
        activation_class=nn.Tanh,
    )
    # Small random output weights and zero biases avoid a directional bias at reset.
    with torch.no_grad():
        nn.init.orthogonal_(actor_mlp.params["4", "weight"], gain=0.01)
        actor_mlp.params["4", "bias"].zero_()
    actor_backbone = (
        actor_mlp
        if observation_scale is None
        else nn.Sequential(InputScale(observation_scale), actor_mlp)
    )
    if action_grid is None:
        actor_module = (
            nn.Sequential(actor_backbone, SharedScale(initial_log_std, action_dim))
            if position_index is None
            else PositionTargetActor(
                actor_backbone, position_index, action_low, action_high, initial_log_std
            )
        )
        module = TensorDictModule(
            actor_module,
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "loc"), ("agents", "scale")],
        )
        parameters = [("agents", "loc"), ("agents", "scale")]
        distribution, arguments = (
            TanhNormal,
            {"low": action_low, "high": action_high, "event_dims": 1},
        )
    else:
        action_grid = tuple(float(value) for value in action_grid)
        if (
            len(action_grid) < 3
            or not np.isfinite(action_grid).all()
            or min(action_grid) < -1
            or max(action_grid) > 1
            or 0 not in action_grid
            or any(a >= b for a, b in zip(action_grid, action_grid[1:], strict=False))
        ):
            raise ValueError(
                "Velocity choices must be ordered in [-1,1] and include zero"
            )
        with torch.no_grad():
            # A weak stop preference gives a neutral initial mode, with broad sampling.
            bias = actor_mlp.params["4", "bias"].reshape(3, len(action_grid))
            bias[:, action_grid.index(0)].fill_(0.05)
        module = TensorDictModule(
            actor_backbone,
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "logits")],
        )
        parameters = [("agents", "logits")]
        distribution, arguments = VelocityDistribution, {"grid": action_grid}
    actor = ProbabilisticActor(
        module=module,
        in_keys=parameters,
        out_keys=[("agents", "action")],
        distribution_class=distribution,
        distribution_kwargs=arguments,
        default_interaction_type=ExplorationType.RANDOM,
        return_log_prob=True,
        log_prob_key=("agents", "sample_log_prob"),
    )
    actor.action_grid = action_grid
    actor.initial_log_std = float(initial_log_std)
    actor.position_index = position_index
    actor.observation_scale = observation_scale
    actor.action_dim, actor.action_low, actor.action_high = (
        action_dim,
        action_low,
        action_high,
    )
    critic_mlp = MultiAgentMLP(
        n_agent_inputs=obs_dim,
        n_agent_outputs=1,
        n_agents=num_robots,
        centralized=True,
        share_params=True,
        num_cells=NETWORK["hidden_units"],
        activation_class=nn.Tanh,
    )
    critic = TensorDictModule(
        critic_mlp
        if observation_scale is None
        else nn.Sequential(InputScale(observation_scale), critic_mlp),
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "state_value")],
    )
    return actor, critic


def make_pusht_networks(config, *, device="cpu", initial_log_std=-1.8, normalize=False):
    """Build the current Push-T actor/critic directly from the task config."""
    scale = [1 / config.workspace_half_extent] * config.observation_dim
    angular_start = 5 if config.obs_type == "state" else 18
    for index in range(angular_start, config.observation_dim, 3):
        scale[index] = 1 / (2 * np.pi)
    if config.obs_type == "state":
        scale[4] = 1 / (2 * np.pi)
    actor, critic = make_mappo_networks(
        config.num_robots,
        config.observation_dim,
        action_dim=2,
        action_low=-config.workspace_half_extent,
        action_high=config.workspace_half_extent,
        position_index=0 if config.obs_type == "state" else 16,
        initial_log_std=initial_log_std,
        observation_scale=scale if normalize else None,
    )
    return actor.to(device), critic.to(device)


@torch.inference_mode()
def policy_action(actor, observation) -> np.ndarray:
    """Deterministic (N,A) actions in the actor's configured Gymnasium units.

    Gaussian policies use tanh(loc); velocity-choice policies use the most
    likely command on each axis. Neither adds navigation or stopping rules.
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
        if PosePushConfig.from_checkpoint(provenance["pose_config"]) != config:
            raise ValueError("Saved pose config differs from the training world")
        extra = dict(pose_config=asdict(config))
        file_format, num_robots = POSE_FORMAT, config.num_robots
        if getattr(config, "task_type", None) == "pusht":
            file_format, obs_dim = PUSHT_FORMAT, config.observation_dim
            if (actor.action_dim, actor.action_low, actor.action_high) != (
                2,
                -config.workspace_half_extent,
                config.workspace_half_extent,
            ):
                raise ValueError(
                    "Push-T checkpoint requires physical XY target actions"
                )
        else:
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
        network={
            **NETWORK,
            "initial_log_std": actor.initial_log_std,
            **(
                {"observation_scale": list(actor.observation_scale)}
                if getattr(actor, "observation_scale", None) is not None
                else {}
            ),
            **(
                {"position_index": actor.position_index}
                if getattr(actor, "position_index", None) is not None
                else {}
            ),
            **(
                {
                    "action_dim": actor.action_dim,
                    "action_low": actor.action_low,
                    "action_high": actor.action_high,
                }
                if actor.action_dim != 3
                or actor.action_low != -1
                or actor.action_high != 1
                else {}
            ),
            **(
                {"action_grid": list(actor.action_grid)}
                if getattr(actor, "action_grid", None) is not None
                else {}
            ),
        },
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
    network = saved.get("network", {})
    if not isinstance(network, dict) or {
        key: value
        for key, value in network.items()
        if key
        not in (
            "action_grid",
            "action_dim",
            "action_low",
            "action_high",
            "position_index",
            "observation_scale",
            "initial_log_std",
        )
    } != {"hidden_units": NETWORK["hidden_units"]}:
        raise ValueError("Unrecognized TorchRL network architecture")
    if saved.get("format") == PUSHT_FORMAT:
        config = PosePushConfig.from_checkpoint(saved["pose_config"])
        if (
            getattr(config, "task_type", None) != "pusht"
            or saved.get("num_robots") != config.num_robots
            or saved.get("obs_dim") != config.observation_dim
            or network.get("action_dim") != 2
            or (network.get("action_low"), network.get("action_high"))
            != (-config.workspace_half_extent, config.workspace_half_extent)
            or network.get("position_index")
            not in (None, 0 if config.obs_type == "state" else 16)
        ):
            raise ValueError(
                "Push-T checkpoint has an inconsistent task or action layout"
            )
    elif saved.get("format") == POSE_FORMAT:
        config = PosePushConfig.from_checkpoint(saved["pose_config"])
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
    actor, critic = make_mappo_networks(
        saved["num_robots"],
        saved["obs_dim"],
        initial_log_std=network["initial_log_std"],
        action_grid=network.get("action_grid"),
        action_dim=network.get("action_dim", 3),
        action_low=network.get("action_low", -1),
        action_high=network.get("action_high", 1),
        position_index=network.get("position_index"),
        observation_scale=network.get("observation_scale"),
    )
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
