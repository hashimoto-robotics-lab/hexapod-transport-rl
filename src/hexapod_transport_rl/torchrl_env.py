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
from .pose_push import PosePushConfig, make_pose_vector
from .vector import make_vector_env


class TorchRLTransportEnv(EnvBase):
    """Native CPU worlds or batched MuJoCo Warp physics on NVIDIA GPUs.

    ``config=PosePushConfig(...)`` selects target-pose pushing (two to four robots).
    ``PushConfig`` selects basic pushing; ``ApproachConfig`` is historical navigation.
    ``num_envs`` counts independent worlds, not robots.
    ``backend="auto"`` selects Warp with CUDA, otherwise native CPU MuJoCo.
    Warp keeps observations, commands, walking and rewards on the GPU.
    """

    def __init__(
        self,
        config: ApproachConfig | PushConfig,
        num_envs: int = 2,
        *,
        asynchronous: bool = True,
        asset_root: str | Path | None = None,
        backend: str = "cpu",
        device: str | torch.device | None = None,
    ):
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        if backend == "auto":
            backend = "warp" if torch.cuda.is_available() else "cpu"
        if backend not in ("cpu", "warp"):
            raise ValueError("backend must be cpu, warp or auto")
        self.backend = backend
        device = torch.device(device or ("cuda:0" if backend == "warp" else "cpu"))
        if backend == "warp" and (
            device.type != "cuda" or not torch.cuda.is_available()
        ):
            raise ValueError("MuJoCo Warp requires an available NVIDIA CUDA device")
        if backend == "cpu" and device.type != "cpu":
            raise ValueError("Native MuJoCo worlds use device=cpu")
        super().__init__(device=device, batch_size=[num_envs])
        if backend == "warp":
            if not isinstance(config, PosePushConfig):
                raise TypeError("Warp currently supports PosePushConfig")
            from .warp_pose import WarpPoseWorlds

            self.worlds = WarpPoseWorlds(config, num_envs, asset_root, device)
            asynchronous = False
        elif isinstance(config, ApproachConfig):
            self.worlds = make_approach_vector(
                config, num_envs, asset_root, asynchronous=asynchronous
            )
        elif isinstance(config, PosePushConfig):
            self.worlds = make_pose_vector(
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
                observation=Unbounded(
                    shape=(num_envs, self.num_robots, self.obs_dim), device=device
                ),
                shape=(num_envs, self.num_robots),
                device=device,
            ),
            success=Categorical(
                2, shape=(num_envs, 1), dtype=torch.bool, device=device
            ),
            shape=(num_envs,),
            device=device,
        )
        self.action_spec = Composite(
            agents=Composite(
                action=Bounded(
                    -1, 1, shape=(num_envs, self.num_robots, 3), device=device
                ),
                shape=(num_envs, self.num_robots),
                device=device,
            ),
            shape=(num_envs,),
            device=device,
        )
        self.reward_spec = Unbounded(shape=(num_envs, 1), device=device)
        self.done_spec = Composite(
            **{
                key: Categorical(
                    2, shape=(num_envs, 1), dtype=torch.bool, device=device
                )
                for key in ("done", "terminated", "truncated")
            },
            shape=(num_envs,),
            device=device,
        )

    def _observation(self, observation, info) -> TensorDict:
        return TensorDict(
            {
                "agents": TensorDict(
                    {
                        "observation": torch.as_tensor(
                            observation, device=self.device
                        ).clone()
                    },
                    batch_size=[self.worlds.num_envs, self.num_robots],
                ),
                "success": torch.as_tensor(
                    info["is_success"], device=self.device
                ).reshape(-1, 1),
            },
            batch_size=self.batch_size,
            device=self.device,
        )

    def _reset(self, tensordict=None, **kwargs):
        reset = tensordict.get("_reset", None) if tensordict is not None else None
        options = (
            {
                "reset_mask": reset.reshape(-1)
                if self.backend == "warp"
                else reset.reshape(-1).cpu().numpy()
            }
            if reset is not None
            else None
        )
        observation, info = self.worlds.reset(seed=self._reset_seed, options=options)
        self._reset_seed = None
        result = self._observation(observation, info)
        for key in ("done", "terminated", "truncated"):
            result[key] = torch.zeros(
                (*self.batch_size, 1), dtype=torch.bool, device=self.device
            )
        return result

    def _step(self, tensordict):
        action = tensordict["agents", "action"].detach()
        if self.backend == "cpu":
            action = action.cpu().numpy()
        observation, reward, terminated, truncated, info = self.worlds.step(action)
        result = self._observation(observation, info)
        result["reward"] = torch.as_tensor(
            reward, dtype=torch.float32, device=self.device
        ).reshape(-1, 1)
        result["terminated"] = torch.as_tensor(terminated, device=self.device).reshape(
            -1, 1
        )
        result["truncated"] = torch.as_tensor(truncated, device=self.device).reshape(
            -1, 1
        )
        result["done"] = result["terminated"] | result["truncated"]
        return result

    def _set_seed(self, seed):
        self._reset_seed = seed

    def set_layout(self, layout: str):
        """Apply curriculum changes at later resets, preserving active episodes."""
        if not isinstance(self.config, ApproachConfig):
            raise TypeError("Only approach environments have a layout curriculum")
        self.worlds.call("set_layout", layout)

    def set_stage(self, stage):
        """Change pose-task reset difficulty without editing active worlds."""
        if not isinstance(self.config, PosePushConfig):
            raise TypeError("Stages require PosePushConfig")
        self.worlds.call("set_stage", stage)

    def check_physics(self):
        """Check finite states and GPU contact capacity at a rollout boundary."""
        if self.backend == "warp":
            self.worlds.physics.check()

    def render(self, world=0):
        """Render an actual Warp lane for inspection, outside the learning loop."""
        if self.backend != "warp":
            raise ValueError("Use a Gym environment for native MuJoCo rendering")
        return self.worlds.render(world)

    def close(self, *, raise_if_closed=True):
        if not self.is_closed:
            self.worlds.close()
        return super().close(raise_if_closed=raise_if_closed)
