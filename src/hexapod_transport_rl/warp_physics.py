"""Original robot model and frozen walker, batched on a shared CUDA stream.

The motor remains MuJoCo's native DC motor: identical voltage feedback at
200 Hz and frozen walking targets at 25 Hz. A CUDA graph contains one complete
high-level physical step, including contact measurements.
"""

import time

import mujoco_warp as mjw
import torch
import warp as wp

from .config import PHYSICS_STEPS
from .motor import SUPPLY_VOLTAGE, VOLTAGE_GAIN
from .warp_contacts import WarpContactTracker


class WarpPhysics:
    """One compiled MuJoCo model and independent GPU states for every world."""

    def __init__(self, reference, num_worlds, device):
        wp.init()
        self.device = torch.device(device)
        self.stream = torch.cuda.Stream(device=self.device)
        self.warp_stream = wp.stream_from_torch(self.stream)
        self.num_worlds, self.num_robots = num_worlds, reference.cfg.num_robots
        self.control_steps = reference.cfg.high_level_decimation
        self.controller = reference.controller.to(self.device).eval()
        self.compile_seconds = 0.0
        with wp.ScopedDevice(str(self.device)):
            self.model = mjw.put_model(reference.model)
            self.data = mjw.put_data(
                reference.model,
                reference.data,
                nworld=num_worlds,
                nconmax=128 * self.num_robots,
                njmax=256 * self.num_robots,
            )
            self.contacts = WarpContactTracker(
                self.model,
                self.data,
                reference.contacts,
                num_worlds,
                self.num_robots,
            )
        self.qpos = wp.to_torch(self.data.qpos)
        self.qvel = wp.to_torch(self.data.qvel)
        self.ctrl = wp.to_torch(self.data.ctrl)
        self.xpos = wp.to_torch(self.data.xpos)
        self.xmat = wp.to_torch(self.data.xmat)
        self.sensor = wp.to_torch(self.data.sensordata)
        self.overflow = wp.to_torch(self.data.overflow)
        self.overflow_seen = torch.zeros_like(self.overflow)
        self.invalid_seen = torch.zeros(
            num_worlds, dtype=torch.bool, device=self.device
        )
        self.joint_q = torch.as_tensor(reference.joint_q, device=self.device)
        self.joint_v = torch.as_tensor(reference.joint_v, device=self.device)
        self.actuators = torch.as_tensor(reference.actuators, device=self.device)
        self.base_ids = torch.as_tensor(reference.base_ids, device=self.device)
        self.gyro_indices = torch.as_tensor(reference.gyro_indices, device=self.device)
        self.commands = torch.zeros(num_worlds, self.num_robots, 3, device=self.device)
        self.previous_action = torch.zeros(
            num_worlds,
            self.num_robots,
            18,
            device=self.device,
        )
        self.targets = torch.zeros_like(self.previous_action)
        self.graph = None

    @torch.no_grad()
    def _advance(self):
        self.contacts.reset()
        for _ in range(self.control_steps):
            mjw.forward(self.model, self.data)
            inputs = (
                self.qpos[:, self.joint_q],
                self.qvel[:, self.joint_v],
                self.previous_action,
                self.commands,
                self.sensor[:, self.gyro_indices],
                -self.xmat[:, self.base_ids, 2, :],
            )
            target, previous = self.controller(
                *(x.reshape(-1, x.shape[-1]) for x in inputs)
            )
            self.targets.copy_(target.reshape_as(self.targets))
            self.previous_action.copy_(previous.reshape_as(self.previous_action))
            for _ in range(PHYSICS_STEPS):
                voltage = (
                    VOLTAGE_GAIN * (self.targets - self.qpos[:, self.joint_q])
                ).clamp(
                    -SUPPLY_VOLTAGE,
                    SUPPLY_VOLTAGE,
                )
                self.ctrl[:, self.actuators] = voltage
                mjw.step(self.model, self.data)
                self.overflow_seen.bitwise_or_(self.overflow)
                self.contacts.record()
        # As in native MuJoCo, expose transforms corresponding to the final qpos.
        mjw.forward(self.model, self.data)

    def compile(self):
        """Warm all kernels before capture; the caller restores episode state."""
        started = time.perf_counter()
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        with (
            torch.cuda.stream(self.stream),
            wp.ScopedStream(self.warp_stream, sync_enter=False),
        ):
            self._advance()
            self._advance()
        self.stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with (
            torch.cuda.graph(self.graph, stream=self.stream),
            wp.ScopedStream(self.warp_stream, sync_enter=False),
            wp.ScopedCapture(
                stream=self.warp_stream,
                external=True,
                capture_mode=wp.CaptureMode.GLOBAL,
            ) as capture,
        ):
            self._advance()
        self.warp_graph = capture.graph
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        self.compile_seconds = time.perf_counter() - started

    @torch.no_grad()
    def step(self, commands):
        self.commands.copy_(commands)
        self.graph.replay()
        self.invalid_seen.logical_or_(
            ~(torch.isfinite(self.qpos).all(-1) & torch.isfinite(self.qvel).all(-1))
        )

    def reset(self, mask):
        """Reset only selected physical worlds, retaining running robot histories."""
        with wp.ScopedStream(
            wp.stream_from_torch(torch.cuda.current_stream(self.device))
        ):
            mjw.reset_data(self.model, self.data, wp.from_torch(mask))
        self.previous_action.masked_fill_(mask[:, None, None], 0)

    def forward(self):
        with wp.ScopedStream(
            wp.stream_from_torch(torch.cuda.current_stream(self.device))
        ):
            mjw.forward(self.model, self.data)

    def check(self):
        """Synchronize at rollout boundaries, rather than on every physics tick."""
        if bool((self.overflow_seen | self.overflow).any()):
            raise RuntimeError(
                "Warp contact/constraint capacity exceeded; check physics buffers"
            )
        if bool(self.invalid_seen.any()) or not bool(
            torch.isfinite(self.qpos).all() & torch.isfinite(self.qvel).all()
        ):
            raise FloatingPointError("Non-finite MuJoCo Warp state")
