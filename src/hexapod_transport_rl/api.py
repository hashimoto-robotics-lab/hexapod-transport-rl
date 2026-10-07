"""Standard RL interfaces over the shared, physically interacting robot team."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from numpy.typing import ArrayLike

from .config import ENV_VERSION, PushConfig
from .env import PushEnv
from .types import Info, Observation, ResetResult, StepResult

GYM_ENV_ID = "HexapodPush-v0"


class HexapodPushEnv(gym.Env):
    """One Gymnasium agent controls the complete robot team.

    By default observations/actions are flat (N*D,)/(N*3,) for PPO/SAC.
    ``flatten=False`` retains (N,D)/(N,3) for shared-policy MARL collectors.
    Episodes never reset implicitly. ``state()`` remains the final state after
    termination/truncation, so timeout bootstrapping is possible before reset.
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 5}

    def __init__(
        self,
        config: PushConfig | dict[str, Any] | None = None,
        asset_root: str | Path | None = None,
        *,
        randomize: bool = True,
        flatten: bool = True,
        render_mode: str | None = None,
        width: int = 640,
        height: int = 480,
    ) -> None:
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError(f"Unsupported render_mode: {render_mode!r}")
        if not isinstance(randomize, bool) or not isinstance(flatten, bool):
            raise TypeError("randomize and flatten must be bool")
        if (
            not isinstance(width, int)
            or not isinstance(height, int)
            or min(width, height) < 1
        ):
            raise ValueError("width and height must be positive integers")
        if isinstance(config, dict):
            config = PushConfig(**config)
        if config is not None and not isinstance(config, PushConfig):
            raise TypeError("config must be PushConfig, dict, or None")
        self.core = PushEnv(config, asset_root)
        self.config = self.core.cfg
        self.num_robots = self.config.num_robots
        self.agent_ids = tuple(f"robot_{i}" for i in range(self.num_robots))
        self.randomize = randomize
        self.flatten = flatten
        self.render_mode = render_mode
        self.width, self.height = width, height
        self.metadata = {
            **self.metadata,
            "render_fps": max(1, round(1 / self.config.dt)),
        }
        obs_shape = (
            (self.core.state_dim,) if flatten else (self.num_robots, self.core.obs_dim)
        )
        action_shape = (3 * self.num_robots,) if flatten else (self.num_robots, 3)
        self.observation_space = spaces.Box(-np.inf, np.inf, obs_shape, np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, action_shape, np.float32)
        self.state_space = spaces.Box(
            -np.inf, np.inf, (self.core.state_dim,), np.float32
        )
        self._has_reset = False
        self._closed = False
        self._renderer = None
        self._viewer = None
        self._episode_return = 0.0

    @property
    def provenance(self) -> Info:
        """Source controller/model identifiers for reproducible training records."""
        return {
            "env_version": ENV_VERSION,
            "asset_root": str(self.core.asset_root),
            "low_level_sha256": self.core.policy_sha256,
            "robot_xml_sha256": self.core.model_sha256,
            "walking_checkpoint_iteration": self.core.policy_metadata[
                "checkpoint_iteration"
            ],
        }

    def _require_ready(self) -> None:
        if self._closed:
            raise RuntimeError("Environment is closed; create a new instance")
        if not self._has_reset:
            raise gym.error.ResetNeeded("Call reset() before using the environment")

    def _format_observation(self, obs: Observation) -> Observation:
        return obs.reshape(self.observation_space.shape).copy()

    def _episode_info(
        self,
        info: Info,
        reward: float = 0.0,
        terminated: bool = False,
        truncated: bool = False,
    ) -> Info:
        """Copy diagnostics so later resets cannot mutate a stored transition."""
        info = deepcopy(info)
        reason = None
        if terminated:
            reason = "failure"
            for cause in ("success", "robot_fall", "cargo_fall", "out_of_bounds"):
                if info.get(cause):
                    reason = cause
                    break
        elif truncated:
            reason = "time_limit"
        info.update(
            is_success=bool(info["success"]),
            team_reward=float(reward),
            episode_return=self._episode_return,
            elapsed_steps=self.core.steps,
            termination_reason=reason,
            state=self.state(),
            commands=self.core.last_commands.copy(),
        )
        return info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> ResetResult:
        """Start an episode; options['randomize'] overrides only this reset."""
        if self._closed:
            raise RuntimeError("Environment is closed; create a new instance")
        options = {} if options is None else dict(options)
        # Ignore unrelated options supplied by general-purpose RL wrappers.
        randomize = options.get("randomize", self.randomize)
        if not isinstance(randomize, bool):
            raise TypeError("options['randomize'] must be bool")
        super().reset(seed=seed)
        # Use Gymnasium's RNG as the core's RNG, including seed=None continuation.
        self.core.rng = self.np_random
        obs, info = self.core.reset(randomize=randomize)
        self._has_reset = True
        self._episode_return = 0.0
        if self.render_mode == "human":
            self.render()
        return self._format_observation(obs), self._episode_info(info)

    def step(self, action: ArrayLike) -> StepResult:
        """Validate the team action and return the unreset final transition."""
        self._require_ready()
        if self.core.done:
            raise gym.error.ResetNeeded("Episode ended; call reset() before step()")
        action = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action):
            raise ValueError(
                f"Action must be finite, in [-1,1], with shape {self.action_space.shape}"
            )
        obs, reward, terminated, truncated, info = self.core.step(
            action.reshape(self.num_robots, 3)
        )
        self._episode_return += reward
        if self.render_mode == "human":
            self.render()
        return (
            self._format_observation(obs),
            reward,
            terminated,
            truncated,
            self._episode_info(info, reward, terminated, truncated),
        )

    def state(self) -> Observation:
        """Centralized critic input: all local observations, in robot index order."""
        self._require_ready()
        return self.core.state().copy()

    def _configure_camera(self, camera: mujoco.MjvCamera) -> None:
        camera.lookat[:] = [*(self.core.goal / 2), 0.15]
        # Include robots behind the cargo as well as the goal, even at reset.
        camera.distance = max(
            6.0,
            self.core.cfg.goal_distance + self.core.cfg.depth + 2,
            self.core.cfg.width + 3,
        )
        camera.azimuth, camera.elevation = 135, -50

    def render(self) -> np.ndarray | None:
        """Render in the constructor-selected mode; RGB frames are uint8 (H,W,3)."""
        self._require_ready()
        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb()
        return None

    def _render_human(self) -> None:
        if self._viewer is None:
            import mujoco.viewer as mj_viewer

            self._viewer = mj_viewer.launch_passive(self.core.model, self.core.data)
        if self._viewer.is_running():
            self._configure_camera(self._viewer.cam)
            self._viewer.sync()
        return None

    def _render_rgb(self) -> np.ndarray:
        if self._renderer is None:
            self.core.model.vis.global_.offwidth = max(
                self.width, self.core.model.vis.global_.offwidth
            )
            self.core.model.vis.global_.offheight = max(
                self.height, self.core.model.vis.global_.offheight
            )
            self._renderer = mujoco.Renderer(
                self.core.model, height=self.height, width=self.width
            )
        camera = mujoco.MjvCamera()
        self._configure_camera(camera)
        option = mujoco.MjvOption()
        option.geomgroup[3] = 0
        self._renderer.update_scene(self.core.data, camera=camera, scene_option=option)
        return self._renderer.render().copy()

    def close(self) -> None:
        if self._closed:
            return
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._closed = True


if GYM_ENV_ID not in gym.registry:
    gym.register(id=GYM_ENV_ID, entry_point="hexapod_transport_rl.api:HexapodPushEnv")
