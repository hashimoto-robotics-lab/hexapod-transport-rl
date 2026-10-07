"""Student API: physical velocity commands for 1–4 pretrained hexapods."""

from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from numpy.typing import ArrayLike

from .config import COMMAND_HIGH, COMMAND_LOW, CONTROL_DT, STAND_HEIGHT, find_asset_root
from .locomotion import WalkingController
from .model import build_walking_model


class WalkingFallError(RuntimeError):
    """The command demonstration stopped because a robot fell."""


class WalkingSimulation:
    """A floor and robots, without cargo, rewards, or a training algorithm.

    set_velocity() replaces one robot's body-frame command, in m/s and rad/s.
    run_for() advances all robots together, retaining their commands until changed.
    Omitted velocity components are zero. Robot IDs are 0 through num_robots-1.
    Use a with block to release optional video resources, including on failure.
    """

    def __init__(
        self,
        num_robots: int = 1,
        *,
        record: bool = False,
        video_path: str | Path | None = None,
    ) -> None:
        if (
            isinstance(num_robots, bool)
            or not isinstance(num_robots, int)
            or not 1 <= num_robots <= 4
        ):
            raise ValueError("num_robots must be an integer from 1 to 4")
        self.num_robots = num_robots
        self.model = build_walking_model(find_asset_root(), num_robots)
        self.data = mujoco.MjData(self.model)
        self._walking = WalkingController(
            find_asset_root(), self.model, self.data, num_robots
        )
        self._commands = np.zeros((num_robots, 3), dtype=np.float32)
        self._closed = False
        self._renderer = None
        self._writer = None
        self._record = record
        self._frames: list[np.ndarray] = []
        self._recording_started = False
        self._video_path = Path(video_path) if video_path is not None else None
        self._recording_ticks = 0
        self.reset()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Simulation is closed; create a new instance")

    def _robot_id(self, robot_id: int) -> int:
        if (
            isinstance(robot_id, bool)
            or not isinstance(robot_id, int)
            or not 0 <= robot_id < self.num_robots
        ):
            raise ValueError(f"robot_id must be 0 through {self.num_robots - 1}")
        return robot_id

    @property
    def time(self) -> float:
        """Elapsed simulation time in seconds, excluding video encoding time."""
        self._require_open()
        return float(self.data.time)

    @property
    def positions(self) -> np.ndarray:
        """A copy of all robots' world XY coordinates in meters, shape (N,2)."""
        self._require_open()
        return self.data.xpos[self._walking.base_ids, :2].copy()

    @property
    def headings(self) -> np.ndarray:
        """World heading in radians for each robot, shape (N,)."""
        self._require_open()
        matrices = self.data.xmat[self._walking.base_ids].reshape(-1, 3, 3)
        return np.arctan2(matrices[:, 1, 0], matrices[:, 0, 0])

    @property
    def velocities(self) -> np.ndarray:
        """A copy of the current target velocities [vx, vy, yaw_rate], shape (N,3)."""
        self._require_open()
        return self._commands.copy()

    @property
    def frames(self) -> list[np.ndarray]:
        """Recorded 640×480 RGB frames at 5 fps, available after close().

        Set record=True to collect frames for mediapy.show_video(frames, fps=5).
        Returns a new list referencing the recorded images. reset() starts a new
        episode but preserves the recording, just as it does for video_path.
        """
        return self._frames.copy()

    def reset(self, *, poses: ArrayLike | None = None) -> None:
        """Stop all robots and place them at world [x m, y m, yaw rad] poses.

        Defaults to a row spaced 1.2 m apart, with room for lateral motion. Validation precedes any state change.
        """
        self._require_open()
        if poses is None:
            poses = np.zeros((self.num_robots, 3))
            poses[:, 1] = 1.2 * (np.arange(self.num_robots) - (self.num_robots - 1) / 2)
        poses = np.asarray(poses, dtype=float)
        if poses.shape != (self.num_robots, 3) or not np.isfinite(poses).all():
            raise ValueError("poses must be finite [x, y, yaw] rows, one per robot")
        mujoco.mj_resetData(self.model, self.data)
        for root, (x, y, yaw) in zip(self._walking.root_q, poses, strict=True):
            self.data.qpos[root : root + 7] = (
                x,
                y,
                STAND_HEIGHT,
                np.cos(yaw / 2),
                0,
                0,
                np.sin(yaw / 2),
            )
        self._walking.reset()
        self._commands.fill(0)
        mujoco.mj_forward(self.model, self.data)

    def set_velocity(
        self, robot_id: int = 0, *, vx: float = 0, vy: float = 0, yaw_rate: float = 0
    ) -> None:
        """Set a persistent body-frame command; simulation time does not advance.

        vx: forward [-0.15,0.20] m/s; vy: left [-0.10,0.10] m/s;
        yaw_rate: counterclockwise [-0.60,0.60] rad/s. Excess values raise an error.
        """
        self._require_open()
        robot_id = self._robot_id(robot_id)
        velocity = np.asarray([vx, vy, yaw_rate], dtype=np.float32)
        if (
            not np.isfinite(velocity).all()
            or np.any(velocity < COMMAND_LOW)
            or np.any(velocity > COMMAND_HIGH)
        ):
            raise ValueError(
                "Velocity must be finite and within the walking model's limits"
            )
        self._commands[robot_id] = velocity

    def stop(self, robot_id: int | None = None) -> None:
        """Send zero velocity to one robot, or all robots when ID is omitted.

        This changes the target command; actual deceleration requires run_for().
        """
        self._require_open()
        if robot_id is None:
            self._commands.fill(0)
        else:
            self._commands[self._robot_id(robot_id)] = 0

    def run_for(self, *, seconds: float) -> None:
        """Advance all robots with their retained commands, in 0.04 s increments."""
        self._require_open()
        if not np.isfinite(seconds) or seconds <= 0:
            raise ValueError("seconds must be positive and finite")
        ticks = round(seconds / CONTROL_DT)
        if ticks < 1 or not np.isclose(ticks * CONTROL_DT, seconds, rtol=0, atol=1e-8):
            raise ValueError("seconds must be a multiple of 0.04 s")
        # Start recording only when time is advanced; invalid commands cannot
        # create video files or partially advance the simulation.
        if (
            self._record or self._video_path is not None
        ) and not self._recording_started:
            if self._video_path is not None:
                self._video_path.parent.mkdir(parents=True, exist_ok=True)
                self._writer = imageio.get_writer(str(self._video_path), fps=5)
            self._record_frame()
            self._recording_started = True
        self._walking.advance(self._commands, ticks, on_control_step=self._after_tick)

    def _after_tick(self) -> None:
        ids = self._walking.base_ids
        if np.any(self.data.xpos[ids, 2] < 0.08) or np.any(
            self.data.xmat[ids, 8] < 0.5
        ):
            raise WalkingFallError(
                "A robot fell; reset() before trying another command"
            )
        self._recording_ticks += 1
        if self._recording_started and self._recording_ticks % 5 == 0:
            self._record_frame()

    def _record_frame(self) -> None:
        """Render once for both notebook frames and optional streaming MP4."""
        frame = self.render()
        if self._record:
            self._frames.append(frame)
        if self._writer is not None:
            self._writer.append_data(frame)

    def render(self) -> np.ndarray:
        """Return a 640×480 RGB image, with original robot colors and a grid floor."""
        self._require_open()
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        camera = mujoco.MjvCamera()
        positions = self.positions
        camera.lookat[:] = [*positions.mean(axis=0), 0.1]
        camera.distance = max(2.6, float(np.ptp(positions, axis=0).max()) + 2)
        camera.azimuth, camera.elevation = 135, -50
        option = mujoco.MjvOption()
        option.geomgroup[3] = 0
        self._renderer.update_scene(self.data, camera=camera, scene_option=option)
        return self._renderer.render().copy()

    def close(self) -> None:
        """Release video resources; calling close() again is harmless."""
        if self._closed:
            return
        try:
            if self._writer is not None:
                self._writer.close()
        finally:
            if self._renderer is not None:
                self._renderer.close()
            self._closed = True

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
