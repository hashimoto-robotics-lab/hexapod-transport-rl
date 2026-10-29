"""Batched cylinder pushing with the original articulated hexapod and walking policy."""

import mujoco
import numpy as np
import torch

from .config import COMMAND_HIGH, COMMAND_LIMIT, COMMAND_LOW, STAND, STAND_HEIGHT
from .env import (
    MAX_SETTLED_SPEED,
    MIN_CARGO_UPRIGHT,
    MIN_ROBOT_HEIGHT,
    MIN_ROBOT_UPRIGHT,
)
from .single_push import (
    SinglePushGymEnv,
    around_start_pose,
    circle_coverage,
    cylinder_approach,
    direction_commands,
    position_reward,
    random_goal_delta,
)
from .warp_physics import WarpPhysics


def wrap_angle(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def cargo_coordinates(vector, yaw):
    """Rotate world XY vectors into the cargo frame, preserving leading axes."""
    c, s = (yaw.cos(), yaw.sin())
    while c.ndim < vector.ndim - 1:
        c, s = (c.unsqueeze(-1), s.unsqueeze(-1))
    return torch.stack(
        (
            vector[..., 0] * c + vector[..., 1] * s,
            -vector[..., 0] * s + vector[..., 1] * c,
        ),
        -1,
    )


class WarpSinglePushWorlds:
    """Independent cylinder worlds with tensor reset/step and no autoreset."""

    reference_env_type = SinglePushGymEnv
    physics_type = WarpPhysics

    def __init__(self, config, num_envs, asset_root=None, device="cuda:0"):
        self.config = self.pending_config = config
        self.num_envs, self.num_robots = (num_envs, config.num_robots)
        self.device = torch.device(device)
        self.reference = self.reference_env_type(
            config, asset_root, render_mode="rgb_array"
        )
        self.single_observation_space = self.reference.observation_space
        self.single_action_space = self.reference.action_space
        self.provenance = self.reference.provenance
        core = self.reference.core
        self.physics = self.physics_type(core, num_envs, self.device)
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
        self._randomize_spawn = (options or {}).get("randomize", True)
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
        spawn_heading = heading
        rear = torch.full_like(
            heading,
            cfg.robot_start == "cargo_rear" or cfg.rear_start_fraction == 1,
            dtype=torch.bool,
        )
        if cfg.robot_start == "goal_side" and 0 < cfg.rear_start_fraction < 1:
            rear = (
                torch.rand(self.num_envs, device=self.device, generator=self.generator)
                < cfg.rear_start_fraction
                if randomized
                else torch.full_like(rear, cfg.rear_start_fraction >= 0.5)
            )
        if cfg.robot_start == "goal_side":
            direction = heading + torch.atan2(
                lateral, torch.full_like(lateral, cfg.goal_distance)
            ) * (cfg.start_angle_degrees / 180)
            spawn_heading = torch.where(rear, heading, direction)
        for robot in range(self.num_robots):
            q = int(self.reference.core.root_q[robot])
            front_x, front_y, front_yaw = cfg.start_pose(cfg.slots[robot])
            rear_x, rear_y, rear_yaw = cfg.start_pose(cfg.slots[robot], rear=True)
            local = torch.stack(
                (
                    torch.where(rear, rear_x, front_x),
                    torch.where(rear, rear_y, front_y),
                ),
                -1,
            )
            xy = cargo_coordinates(local, -spawn_heading)
            yaw = spawn_heading + torch.where(rear, rear_yaw, front_yaw)
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
        if self.physics.joint_q.numel():
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
        self._finish_reset(mask)
        self.physics.forward()
        info = self.measurements()
        info["is_success"] = torch.zeros_like(mask)
        return (self.observe(), info)

    def _finish_reset(self, mask):
        cfg = self.pending_config
        delta = cfg.goal_pose[2] - self.goal_yaw
        fixed = self.goal.new_tensor(cfg.goal_pose[:2])
        p = self.physics
        for q in [self.cargo_q, *self.reference.core.root_q]:
            xy = cargo_coordinates(p.qpos[:, q : q + 2] - self.goal, -delta) + fixed
            yaw = 2 * torch.atan2(p.qpos[:, q + 6], p.qpos[:, q + 3]) + delta
            p.qpos[:, q : q + 2].copy_(
                torch.where(mask[:, None], xy, p.qpos[:, q : q + 2])
            )
            quat = torch.stack(
                (
                    (yaw / 2).cos(),
                    torch.zeros_like(yaw),
                    torch.zeros_like(yaw),
                    (yaw / 2).sin(),
                ),
                -1,
            )
            p.qpos[:, q + 3 : q + 7].copy_(
                torch.where(mask[:, None], quat, p.qpos[:, q + 3 : q + 7])
            )
        self.goal.copy_(torch.where(mask[:, None], fixed, self.goal))
        self.goal_yaw.masked_fill_(mask, cfg.goal_pose[2])
        cfg = self.pending_config
        if cfg.spawn_mode != "around":
            return
        p = self.physics
        cargo = p.qpos[:, self.cargo_q : self.cargo_q + 2]
        if cfg.goal_position_mode == "uniform":
            goal_uniform = torch.rand(
                self.num_envs, 2, device=self.device, generator=self.generator
            )
            if not self._randomize_spawn:
                goal_uniform[:, 0] = 0
                goal_uniform[:, 1] = 0.5
            goal = cargo + random_goal_delta(cfg, goal_uniform, torch)
            self.goal.copy_(torch.where(mask[:, None], goal, self.goal))
        uniform = torch.rand(
            self.num_envs, 3, device=self.device, generator=self.generator
        )
        if not self._randomize_spawn:
            uniform[:, 0] = 0
            uniform[:, 1:] = 0.5
        relative, yaw = around_start_pose(cfg, self.goal - cargo, uniform, torch)
        q = int(self.reference.core.root_q[0])
        pose = torch.stack(
            (
                *(cargo + relative).unbind(-1),
                torch.full_like(yaw, STAND_HEIGHT),
                (yaw / 2).cos(),
                torch.zeros_like(yaw),
                torch.zeros_like(yaw),
                (yaw / 2).sin(),
            ),
            -1,
        )
        p.qpos[:, q : q + 7].copy_(
            torch.where(mask[:, None], pose, p.qpos[:, q : q + 7])
        )

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
        return (cargo_xy, yaw, positions, headings)

    def observe(self):
        p = self.physics
        xy, yaw, _, _ = self.coordinates()
        heading = torch.atan2(
            p.xmat[:, p.base_ids, 1, 0][:, 0], p.xmat[:, p.base_ids, 0, 0][:, 0]
        )
        return torch.cat(
            (p.xpos[:, p.base_ids, :2][:, 0], xy, yaw[:, None], heading[:, None]), -1
        ).unsqueeze(1)

    def measurements(self):
        cargo_xy, yaw, _, _ = self.coordinates()
        xy = self.physics.xpos[:, self.physics.base_ids, :2][:, 0]
        distance = torch.linalg.vector_norm(self.goal - cargo_xy, dim=-1)
        return dict(
            distance=distance,
            yaw_error=wrap_angle(self.goal_yaw - yaw).abs(),
            approach=cylinder_approach(
                self.config, xy - cargo_xy, self.goal - cargo_xy, torch
            ),
            coverage=circle_coverage(distance, self.config.cargo_radius, torch),
            cargo_speed=torch.linalg.vector_norm(
                self.physics.qvel[:, self.cargo_v : self.cargo_v + 2], dim=-1
            ),
            cargo_yaw_speed=self.physics.qvel[:, self.cargo_v + 5].abs(),
        )

    def status(self, info):
        p, cfg = (self.physics, self.config)
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
            & (info["cargo_speed"] < MAX_SETTLED_SPEED)
            & ~failed
        )
        self.success_time.copy_(torch.where(settled, self.success_time + cfg.dt, 0.0))
        success = self.success_time + 1e-07 >= self.hold_seconds
        terminated = success | failed
        truncated = (self.steps * cfg.dt + 1e-07 >= cfg.episode_seconds) & ~terminated
        info.update(
            is_success=success,
            success=success,
            failed=failed,
            robot_fall=robot_fall,
            cargo_fall=cargo_fall,
            out_of_bounds=outside,
        )
        return (terminated, truncated)

    def reward_terms(self, before, after, commands):
        return {"position_task": position_reward(self.config, before, after, torch)}

    @torch.no_grad()
    def step(self, action):
        p = self.physics
        headings = torch.atan2(p.xmat[:, p.base_ids, 1, 0], p.xmat[:, p.base_ids, 0, 0])
        commands = direction_commands(
            self.config, action, headings, torch, low=self.low, high=self.high
        )
        before = self.measurements()
        p.step(commands)
        self.steps.add_(1)
        info = self.measurements()
        terminated, truncated = self.status(info)
        info["reward_terms"] = self.reward_terms(before, info, commands)
        self.last_commands.copy_(commands)
        return (
            self.observe(),
            sum(info["reward_terms"].values()),
            terminated,
            truncated,
            info,
        )

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
