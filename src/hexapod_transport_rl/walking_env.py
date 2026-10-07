"""Gymnasium command demonstration above the frozen walking controller."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import ArrayLike

from .config import COMMAND_HIGH, COMMAND_LOW
from .types import Info, Observation, ResetResult, StepResult
from .walking import WalkingFallError, WalkingSimulation

WALKING_ENV_ID = "HexapodWalking-v0"


class WalkingEnv(gym.Env):
    """Send physical body-frame velocities to 1–4 robots every 0.2 seconds.

    Actions are (N,3) [vx m/s, vy m/s, yaw_rate rad/s]. Observations are (N,6)
    [world x, world y, heading, last vx command, last vy command, last yaw command].
    This demonstrates a pretrained walker, so its reward is always zero.
    Falling terminates an episode; the duration limit truncates it.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 5}
    dt = 0.2

    def __init__(
        self,
        num_robots: int = 1,
        *,
        episode_seconds: float = 10.0,
        render_mode: str | None = None,
    ) -> None:
        if render_mode not in (None, "rgb_array"):
            raise ValueError("WalkingEnv supports render_mode='rgb_array' or None")
        if not np.isfinite(episode_seconds) or episode_seconds <= 0:
            raise ValueError("episode_seconds must be positive and finite")
        steps = round(episode_seconds / self.dt)
        if steps < 1 or not np.isclose(steps * self.dt, episode_seconds):
            raise ValueError("episode_seconds must be a multiple of 0.2 seconds")
        self.sim = WalkingSimulation(num_robots=num_robots)
        self.num_robots = num_robots
        self.render_mode = render_mode
        self.max_steps = steps
        self.action_space = spaces.Box(
            np.tile(COMMAND_LOW, (num_robots, 1)),
            np.tile(COMMAND_HIGH, (num_robots, 1)),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            -np.inf, np.inf, (num_robots, 6), np.float32
        )
        self._has_reset = False
        self._closed = False
        self._done = False
        self._steps = 0

    def _observation(self) -> Observation:
        return np.column_stack(
            (self.sim.positions, self.sim.headings, self.sim.velocities)
        ).astype(np.float32)

    def _info(self, failed: bool = False) -> Info:
        return dict(
            positions=self.sim.positions,
            headings=self.sim.headings,
            commands=self.sim.velocities,
            elapsed_seconds=self.sim.time,
            elapsed_steps=self._steps,
            failed=failed,
            robot_fall=failed,
            reward_terms={},
            termination_reason="robot_fall"
            if failed
            else "time_limit"
            if self._done
            else None,
        )

    def _require_ready(self) -> None:
        if self._closed:
            raise RuntimeError("Environment is closed; create a new instance")
        if not self._has_reset:
            raise gym.error.ResetNeeded("Call reset() before using the environment")

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> ResetResult:
        if self._closed:
            raise RuntimeError("Environment is closed; create a new instance")
        super().reset(seed=seed)
        self.sim.reset(poses=(options or {}).get("poses"))
        self._has_reset = True
        self._done = False
        self._steps = 0
        return self._observation(), self._info()

    def step(self, action: ArrayLike) -> StepResult:
        self._require_ready()
        if self._done:
            raise gym.error.ResetNeeded("Episode ended; call reset() before step()")
        action = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action):
            raise ValueError(
                "Expected physical velocity commands (N,3) within action_space"
            )
        for robot_id, (vx, vy, yaw_rate) in enumerate(action):
            self.sim.set_velocity(robot_id, vx=vx, vy=vy, yaw_rate=yaw_rate)
        failed = False
        try:
            self.sim.run_for(seconds=self.dt)
        except WalkingFallError:
            failed = True
        self._steps += 1
        truncated = self._steps >= self.max_steps and not failed
        self._done = failed or truncated
        return self._observation(), 0.0, failed, truncated, self._info(failed)

    def render(self) -> np.ndarray | None:
        self._require_ready()
        return self.sim.render() if self.render_mode == "rgb_array" else None

    def close(self) -> None:
        self.sim.close()
        self._closed = True


if WALKING_ENV_ID not in gym.registry:
    gym.register(
        id=WALKING_ENV_ID, entry_point="hexapod_transport_rl.walking_env:WalkingEnv"
    )
