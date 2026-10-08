"""GPU tensor version of the pose task; native MuJoCo remains the Gym renderer.

Each lane has independent dynamics, walker history, goal and success tolerances.
Only reset lanes receive pending curriculum settings. No teacher actions or
external forces are used. CPU transfers are reserved for optional rendering.
"""

import mujoco
import numpy as np
import torch

from .config import COMMAND_HIGH, COMMAND_LIMIT, COMMAND_LOW, STAND, STAND_HEIGHT
from .env import (
    MAX_SETTLED_SPEED,
    MAX_SETTLED_YAW_SPEED,
    MIN_CARGO_UPRIGHT,
    MIN_ROBOT_HEIGHT,
    MIN_ROBOT_UPRIGHT,
)
from .pose_push import PosePushEnv
from .pose_rewards import pose_reward_terms
from .warp_physics import WarpPhysics


def wrap_angle(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def cargo_coordinates(vector, yaw):
    """Rotate world XY vectors into the cargo frame, preserving leading axes."""
    c, s = yaw.cos(), yaw.sin()
    while c.ndim < vector.ndim - 1:
        c, s = c.unsqueeze(-1), s.unsqueeze(-1)
    return torch.stack(
        (
            vector[..., 0] * c + vector[..., 1] * s,
            -vector[..., 0] * s + vector[..., 1] * c,
        ),
        -1,
    )


class WarpPoseWorlds:
    """Batched pose physics for TorchRL, with tensor reset/step and no autoreset."""

    def __init__(self, config, num_envs, asset_root=None, device="cuda:0"):
        self.config = self.pending_config = config
        self.num_envs, self.num_robots = num_envs, config.num_robots
        self.device = torch.device(device)
        self.reference = PosePushEnv(config, asset_root, render_mode="rgb_array")
        self.single_observation_space = self.reference.observation_space
        self.provenance = self.reference.provenance
        core = self.reference.core
        self.physics = WarpPhysics(core, num_envs, self.device)
        self.cargo_q, self.cargo_v, self.cargo_id = (
            core.cargo_q,
            core.cargo_v,
            core.cargo_id,
        )
        self.root_q = torch.as_tensor(core.root_q, device=self.device)
        self.root_v = torch.as_tensor(core.root_v, device=self.device)
        self.limits = torch.as_tensor(COMMAND_LIMIT, device=self.device)
        self.high = torch.as_tensor(COMMAND_HIGH, device=self.device)
        self.low = torch.as_tensor(COMMAND_LOW, device=self.device)
        self.slots = torch.as_tensor(
            config.slots, dtype=torch.float32, device=self.device
        )
        side = torch.where(self.slots >= 0, 1.0, -1.0)
        self.signs = torch.stack((torch.ones_like(side), side, side), -1)
        self.goal = torch.zeros(num_envs, 2, device=self.device)
        self.goal_yaw = torch.zeros(num_envs, device=self.device)
        self.last_commands = torch.zeros(
            num_envs, self.num_robots, 3, device=self.device
        )
        self.steps = torch.zeros(num_envs, dtype=torch.int64, device=self.device)
        self.success_time = torch.zeros(num_envs, device=self.device)
        self.position_tolerance = torch.zeros(num_envs, device=self.device)
        self.yaw_tolerance = torch.zeros(num_envs, device=self.device)
        self.hold_seconds = torch.zeros(num_envs, device=self.device)
        self.generator = torch.Generator(device=self.device).manual_seed(0)
        self.reset(seed=0)
        self.physics.compile()
        # Warm-up/capture executes physics. None of it enters the first episode.
        self.reset(seed=0)
        self.physics.check()

    def call(self, method, *args):
        if method == "provenance":
            return (self.provenance,)
        if method == "set_stage":
            self.pending_config = args[0].apply(self.config)
            return ()
        raise AttributeError(method)

    @torch.no_grad()
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.generator.manual_seed(seed)
        options = options or {}
        mask = options.get(
            "reset_mask",
            torch.ones(self.num_envs, device=self.device, dtype=torch.bool),
        )
        mask = torch.as_tensor(mask, device=self.device, dtype=torch.bool)
        self.physics.reset(mask)
        cfg = self.pending_config
        # Fixed shapes avoid a device synchronization to count selected lanes.
        random = torch.rand(
            self.num_envs,
            4 + self.num_robots * 3,
            device=self.device,
            generator=self.generator,
        )
        randomized = options.get("randomize", True)
        heading = (
            (random[:, 0] * 2 - 1) * torch.pi
            if randomized
            else torch.zeros_like(random[:, 0])
        )
        angle = (
            cfg.min_yaw_degrees
            + random[:, 1] * (cfg.max_yaw_degrees - cfg.min_yaw_degrees)
        ) * (torch.pi / 180)
        angle = (
            angle * torch.where(random[:, 2] < 0.5, -1.0, 1.0)
            if randomized
            else torch.full_like(angle, cfg.max_yaw_degrees * torch.pi / 180)
        )
        lateral = (
            (random[:, 3] * 2 - 1) * cfg.goal_lateral_range
            if randomized
            else torch.zeros_like(heading)
        )
        goal = cargo_coordinates(
            torch.stack((torch.full_like(heading, cfg.goal_distance), lateral), -1),
            -heading,
        )
        self.goal.copy_(torch.where(mask[:, None], goal, self.goal))
        self.goal_yaw.copy_(torch.where(mask, heading + angle, self.goal_yaw))
        qpos = self.physics.qpos.clone()
        qpos[:, self.cargo_q : self.cargo_q + 7] = torch.stack(
            (
                torch.zeros_like(heading),
                torch.zeros_like(heading),
                torch.full_like(heading, cfg.cargo_height / 2 + 0.002),
                (heading / 2).cos(),
                torch.zeros_like(heading),
                torch.zeros_like(heading),
                (heading / 2).sin(),
            ),
            -1,
        )
        jitter = random[:, 4:].reshape(self.num_envs, self.num_robots, 3) * 2 - 1
        for robot in range(self.num_robots):
            q = int(self.reference.core.root_q[robot])
            local = torch.stack(
                (
                    torch.full_like(heading, cfg.rear_face - cfg.push_gap - 0.10),
                    self.slots[robot].expand_as(heading),
                ),
                -1,
            )
            xy = cargo_coordinates(local, -heading)
            yaw = heading
            if randomized:
                xy = xy + jitter[:, robot, :2] * cfg.position_jitter
                yaw = yaw + jitter[:, robot, 2] * cfg.yaw_jitter
            qpos[:, q : q + 7] = torch.stack(
                (
                    xy[:, 0],
                    xy[:, 1],
                    torch.full_like(yaw, STAND_HEIGHT),
                    (yaw / 2).cos(),
                    torch.zeros_like(yaw),
                    torch.zeros_like(yaw),
                    (yaw / 2).sin(),
                ),
                -1,
            )
        qpos[:, self.physics.joint_q] = torch.as_tensor(
            STAND, dtype=torch.float32, device=self.device
        )
        self.physics.qpos.copy_(torch.where(mask[:, None], qpos, self.physics.qpos))
        self.last_commands.masked_fill_(mask[:, None, None], 0)
        self.steps.masked_fill_(mask, 0)
        self.success_time.masked_fill_(mask, 0)
        for name, value in (
            ("position_tolerance", cfg.position_tolerance),
            ("yaw_tolerance", cfg.yaw_tolerance),
            ("hold_seconds", cfg.success_hold_seconds),
        ):
            field = getattr(self, name)
            field.copy_(torch.where(mask, value, field))
        self.physics.forward()
        return self.observe(), {"is_success": torch.zeros_like(mask)}

    def coordinates(self):
        p = self.physics
        cargo_xy = p.xpos[:, self.cargo_id, :2]
        mat = p.xmat[:, self.cargo_id]
        yaw = torch.atan2(mat[:, 1, 0], mat[:, 0, 0])
        positions = cargo_coordinates(
            p.xpos[:, p.base_ids, :2] - cargo_xy[:, None], yaw
        )
        robot_mat = p.xmat[:, p.base_ids]
        headings = wrap_angle(
            torch.atan2(robot_mat[..., 1, 0], robot_mat[..., 0, 0]) - yaw[:, None]
        )
        return cargo_xy, yaw, positions, headings

    def observe(self):
        cargo_xy, yaw, positions, headings = self.coordinates()
        p = self.physics
        goal_xy = cargo_coordinates(self.goal - cargo_xy, yaw)
        goal_yaw = wrap_angle(self.goal_yaw - yaw)
        robot_velocity = cargo_coordinates(
            p.qvel[:, self.root_v[:, None] + torch.arange(2, device=self.device)], yaw
        )
        cargo_velocity = cargo_coordinates(
            p.qvel[:, self.cargo_v : self.cargo_v + 2], yaw
        )
        rows = []
        for i in range(self.num_robots):
            side = self.signs[i, 1]
            own = torch.cat(
                (
                    positions[:, i] * self.signs[i, :2],
                    torch.stack(
                        (headings[:, i].sin() * side, headings[:, i].cos()), -1
                    ),
                    torch.cat(
                        (
                            robot_velocity[:, i],
                            p.qvel[:, int(self.reference.core.root_v[i]) + 5, None],
                        ),
                        -1,
                    )
                    * self.signs[i]
                    / self.limits,
                    self.last_commands[:, i] * self.signs[i] / self.limits,
                    goal_xy * self.signs[i, :2],
                    torch.stack((goal_yaw.sin() * side, goal_yaw.cos()), -1),
                    torch.cat((cargo_velocity, p.qvel[:, self.cargo_v + 5, None]), -1)
                    * self.signs[i]
                    / self.limits,
                    (self.slots[i].abs() / self.config.width).expand(self.num_envs, 1),
                ),
                -1,
            )
            peers = [
                torch.cat(
                    (
                        (positions[:, j] - positions[:, i]) * self.signs[i, :2],
                        torch.stack(
                            (
                                (headings[:, j] - headings[:, i]).sin() * side,
                                (headings[:, j] - headings[:, i]).cos(),
                            ),
                            -1,
                        ),
                    ),
                    -1,
                )
                for j in range(self.num_robots)
                if j != i
            ]
            rows.append(torch.cat((own, *peers), -1))
        return torch.stack(rows, 1)

    def measurements(self):
        cargo_xy, yaw, positions, _ = self.coordinates()
        desired = torch.stack(
            (
                torch.full_like(
                    self.slots, self.config.rear_face - self.config.push_gap
                ),
                self.slots,
            ),
            -1,
        )
        return dict(
            distance=torch.linalg.vector_norm(self.goal - cargo_xy, dim=-1),
            yaw_error=wrap_angle(self.goal_yaw - yaw).abs(),
            approach=torch.linalg.vector_norm(positions - desired, dim=-1).mean(-1),
            cargo_speed=torch.linalg.vector_norm(
                self.physics.qvel[:, self.cargo_v : self.cargo_v + 2], dim=-1
            ),
            cargo_yaw_speed=self.physics.qvel[:, self.cargo_v + 5].abs(),
        )

    def status(self, info):
        p, cfg = self.physics, self.config
        robot_fall = (
            (p.xpos[:, p.base_ids, 2] < MIN_ROBOT_HEIGHT)
            | (p.xmat[:, p.base_ids, 2, 2] < MIN_ROBOT_UPRIGHT)
        ).any(-1)
        cargo_fall = p.xmat[:, self.cargo_id, 2, 2] < MIN_CARGO_UPRIGHT
        outside = (
            torch.linalg.vector_norm(p.xpos[:, p.base_ids, :2], dim=-1)
            > cfg.arena_radius
        ).any(-1) | (
            torch.linalg.vector_norm(p.xpos[:, self.cargo_id, :2], dim=-1)
            > cfg.arena_radius
        )
        failed = robot_fall | cargo_fall | outside
        settled = (
            (info["distance"] < self.position_tolerance)
            & (info["yaw_error"] < self.yaw_tolerance)
            & (info["cargo_speed"] < MAX_SETTLED_SPEED)
            & (info["cargo_yaw_speed"] < MAX_SETTLED_YAW_SPEED)
            & ~failed
        )
        self.success_time.copy_(torch.where(settled, self.success_time + cfg.dt, 0.0))
        success = self.success_time + 1e-7 >= self.hold_seconds
        terminated = success | failed
        truncated = (self.steps * cfg.dt + 1e-7 >= cfg.episode_seconds) & ~terminated
        info.update(
            is_success=success,
            success=success,
            failed=failed,
            robot_fall=robot_fall,
            cargo_fall=cargo_fall,
            out_of_bounds=outside,
        )
        return terminated, truncated

    def reward_terms(self, before, after, commands):
        return pose_reward_terms(
            self.config,
            before,
            after,
            commands,
            self.last_commands,
            self.limits,
            self.physics.contacts.part_contact_counts[:, 0].float(),
            self.physics.contacts.robot_collision_steps,
            self.position_tolerance,
            self.yaw_tolerance,
            math=torch,
        )

    @torch.no_grad()
    def step(self, action):
        action = action.clamp(-1, 1) * self.signs
        commands = action * torch.where(action >= 0, self.high, -self.low)
        before = self.measurements()
        self.physics.step(commands)
        self.steps.add_(1)
        info = self.measurements()
        terminated, truncated = self.status(info)
        info["reward_terms"] = self.reward_terms(before, info, commands)
        reward = sum(info["reward_terms"].values())
        self.last_commands.copy_(commands)
        return self.observe(), reward, terminated, truncated, info

    def render(self, world=0):
        """Render the actual GPU state; copying happens only on this explicit call."""
        env = self.reference
        env.core.data.qpos[:] = self.physics.qpos[world].cpu().numpy()
        env.core.data.qvel[:] = self.physics.qvel[world].cpu().numpy()
        env.core.goal = self.goal[world].cpu().numpy()
        env.core.goal_yaw = float(self.goal_yaw[world])
        marker = env.core.model.body("goal")
        marker.pos[:] = (*env.core.goal, self.config.cargo_height / 2)
        marker.quat[:] = (
            np.cos(env.core.goal_yaw / 2),
            0,
            0,
            np.sin(env.core.goal_yaw / 2),
        )
        mujoco.mj_forward(env.core.model, env.core.data)
        env._has_reset = True
        return env.render()

    def close(self):
        self.reference.close()
        torch.cuda.synchronize(self.device)
        self.physics.graph = None
        self.physics.warp_graph = None
