"""Expose the existing Gymnasium physics as a batched TorchRL MARL environment.

The world still returns one team reward. Robot observations/actions retain an
agent dimension; final observations are kept until TorchRL requests a reset.
"""

from pathlib import Path

import torch
from tensordict import TensorDict
from torchrl.data import Bounded, Categorical, Composite, Unbounded
from torchrl.envs import EnvBase

from .approach import ApproachConfig, make_approach_vector
from .config import PushConfig
from .vector import make_vector_env


class TorchRLTransportEnv(EnvBase):
    """CPU MuJoCo worlds, optionally collected in independent spawn workers.

    ``config=ApproachConfig(...)`` selects two-robot navigation;
    ``config=PushConfig(...)`` selects pushing with two to four robots.
    ``num_envs`` counts independent worlds, not robots.
    """

    def __init__(
        self,
        config: ApproachConfig | PushConfig,
        num_envs: int = 2,
        *,
        asynchronous: bool = True,
        asset_root: str | Path | None = None,
    ):
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        super().__init__(device="cpu", batch_size=[num_envs])
        if isinstance(config, ApproachConfig):
            self.worlds = make_approach_vector(
                config, num_envs, asset_root, asynchronous=asynchronous
            )
        elif isinstance(config, PushConfig):
            self.worlds = make_vector_env(
                num_envs,
                config,
                asset_root,
                asynchronous=asynchronous,
                flatten=False,
            )
        else:
            raise TypeError("config must be ApproachConfig or PushConfig")
        self.num_robots, self.obs_dim = self.worlds.single_observation_space.shape
        self.asynchronous = asynchronous
        self.config = config
        self.provenance = self.worlds.call("provenance")[0]
        self._reset_seed = None
        self.observation_spec = Composite(
            agents=Composite(
                observation=Unbounded(shape=(num_envs, self.num_robots, self.obs_dim)),
                shape=(num_envs, self.num_robots),
            ),
            success=Categorical(2, shape=(num_envs, 1), dtype=torch.bool),
            shape=(num_envs,),
        )
        self.action_spec = Composite(
            agents=Composite(
                action=Bounded(-1, 1, shape=(num_envs, self.num_robots, 3)),
                shape=(num_envs, self.num_robots),
            ),
            shape=(num_envs,),
        )
        self.reward_spec = Unbounded(shape=(num_envs, 1))
        self.done_spec = Composite(
            **{
                key: Categorical(2, shape=(num_envs, 1), dtype=torch.bool)
                for key in ("done", "terminated", "truncated")
            },
            shape=(num_envs,),
        )

    def _observation(self, observation, info) -> TensorDict:
        return TensorDict(
            {
                "agents": TensorDict(
                    {"observation": torch.as_tensor(observation.copy())},
                    batch_size=[self.worlds.num_envs, self.num_robots],
                ),
                "success": torch.as_tensor(info["is_success"]).reshape(-1, 1),
            },
            batch_size=self.batch_size,
            device=self.device,
        )

    def _reset(self, tensordict=None, **kwargs):
        reset = tensordict.get("_reset", None) if tensordict is not None else None
        options = (
            {"reset_mask": reset.reshape(-1).numpy()} if reset is not None else None
        )
        observation, info = self.worlds.reset(seed=self._reset_seed, options=options)
        self._reset_seed = None
        result = self._observation(observation, info)
        for key in ("done", "terminated", "truncated"):
            result[key] = torch.zeros((*self.batch_size, 1), dtype=torch.bool)
        return result

    def _step(self, tensordict):
        action = tensordict["agents", "action"].detach().cpu().numpy()
        observation, reward, terminated, truncated, info = self.worlds.step(action)
        result = self._observation(observation, info)
        result["reward"] = torch.as_tensor(reward, dtype=torch.float32).reshape(-1, 1)
        result["terminated"] = torch.as_tensor(terminated).reshape(-1, 1)
        result["truncated"] = torch.as_tensor(truncated).reshape(-1, 1)
        result["done"] = result["terminated"] | result["truncated"]
        return result

    def _set_seed(self, seed):
        self._reset_seed = seed

    def set_layout(self, layout: str):
        """Apply curriculum changes at later resets, preserving active episodes."""
        if not isinstance(self.config, ApproachConfig):
            raise TypeError("Only approach environments have a layout curriculum")
        self.worlds.call("set_layout", layout)

    def close(self, *, raise_if_closed=True):
        if not self.is_closed:
            self.worlds.close()
        return super().close(raise_if_closed=raise_if_closed)
