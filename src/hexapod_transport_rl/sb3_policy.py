"""Network structure for SB3 PPO; all rollout collection and PPO updates stay in SB3.

The actor sees one robot at a time with shared weights and exploration scales.
The critic sees the complete team. SB3 uses a joint-action probability ratio,
whereas the historical MAPPO implementation clipped each robot's ratio.
"""

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import create_mlp
from torch import nn

HIDDEN_UNITS = 128


class TeamFeatures(nn.Module):
    """Local actor features and centralized critic features from flat observations."""

    def __init__(self, observation_dim: int, num_robots: int):
        super().__init__()
        self.num_robots = num_robots
        self.local_dim = observation_dim // num_robots
        self.latent_dim_pi = HIDDEN_UNITS * num_robots
        self.latent_dim_vf = HIDDEN_UNITS
        self.actor = nn.Sequential(*create_mlp(self.local_dim, -1, [128, 128], nn.Tanh))
        self.critic = nn.Sequential(
            *create_mlp(observation_dim, -1, [128, 128], nn.Tanh)
        )

    def forward_actor(self, observation: torch.Tensor) -> torch.Tensor:
        local = observation.reshape(-1, self.num_robots, self.local_dim)
        return self.actor(local).flatten(start_dim=1)

    def forward_critic(self, observation: torch.Tensor) -> torch.Tensor:
        return self.critic(observation)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(observation), self.forward_critic(observation)


class SharedActionHead(nn.Module):
    def __init__(self, num_robots: int):
        super().__init__()
        self.num_robots = num_robots
        self.linear = nn.Linear(HIDDEN_UNITS, 3)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features.reshape(-1, self.num_robots, HIDDEN_UNITS)).flatten(
            1
        )


class SharedGaussian(DiagGaussianDistribution):
    """SB3's Gaussian distribution with the same three scales for every robot."""

    def __init__(self, num_robots: int):
        super().__init__(3 * num_robots)
        self.num_robots = num_robots

    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0):
        return SharedActionHead(self.num_robots), nn.Parameter(
            torch.full((3,), log_std_init)
        )

    def proba_distribution(self, mean_actions: torch.Tensor, log_std: torch.Tensor):
        return super().proba_distribution(mean_actions, log_std.repeat(self.num_robots))


class SharedTeamPolicy(ActorCriticPolicy):
    """Use as PPO(SharedTeamPolicy, env, policy_kwargs={'num_robots': 2}).

    Requires flat team observations/actions. Gaussian actions are clipped by SB3
    to the environment's bounds; the old tanh-transformed policy is not reused.
    """

    def __init__(
        self, observation_space, action_space, lr_schedule, *, num_robots=2, **kwargs
    ):
        if (
            not isinstance(observation_space, spaces.Box)
            or len(observation_space.shape) != 1
        ):
            raise ValueError("SharedTeamPolicy requires flat Box observations")
        if observation_space.shape[0] % num_robots or action_space.shape != (
            num_robots * 3,
        ):
            raise ValueError(
                "Expected one local observation and three actions per robot"
            )
        if kwargs.get("use_sde", False):
            raise ValueError("SharedTeamPolicy uses diagonal Gaussian exploration")
        self.num_robots = num_robots
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = TeamFeatures(self.features_dim, self.num_robots)

    def _build(self, lr_schedule):
        self.action_dist = SharedGaussian(self.num_robots)
        super()._build(lr_schedule)

    def _get_constructor_parameters(self):
        return {**super()._get_constructor_parameters(), "num_robots": self.num_robots}


class SB3Navigator:
    """Adapter for replay with the existing frozen pusher, without changing actions."""

    def __init__(self, model):
        self.model = model

    def act(self, observations) -> np.ndarray:
        observation = np.asarray(observations, dtype=np.float32)
        action, _ = self.model.predict(observation.reshape(-1), deterministic=True)
        return action.reshape(2, 3)
