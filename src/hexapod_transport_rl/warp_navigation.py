"""The crossing task on batched real hexapod dynamics, without cargo."""

import mujoco
import torch

from .config import COMMAND_HIGH, COMMAND_LOW, STAND, STAND_HEIGHT
from .navigation import NavigationEnv, navigation_observation, navigation_reward
from .warp_physics import WarpPhysics


class WarpNavigationWorlds:
    def __init__(self, config, num_envs, asset_root=None, device="cuda:0"):
        self.config, self.num_envs = config, num_envs
        self.device = torch.device(device)
        self.reference = NavigationEnv(config, asset_root, render_mode="rgb_array")
        self.provenance = self.reference.provenance
        self.single_observation_space = self.reference.observation_space
        self.single_action_space = self.reference.action_space
        self.physics = WarpPhysics(self.reference, num_envs, self.device)
        self.root_q = torch.as_tensor(self.reference.root_q, device=self.device)
        self.root_v = torch.as_tensor(self.reference.root_v, device=self.device)
        self.high = torch.as_tensor(COMMAND_HIGH, device=self.device)
        self.low = torch.as_tensor(COMMAND_LOW, device=self.device)
        self.stand = torch.as_tensor(STAND, dtype=torch.float32, device=self.device)
        self.pair_a, self.pair_b = torch.triu_indices(
            config.num_robots, config.num_robots, 1, device=self.device
        )
        self.goals = torch.zeros(num_envs, config.num_robots, 2, device=self.device)
        self.commands = torch.zeros(num_envs, config.num_robots, 3, device=self.device)
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.success_time = torch.zeros(num_envs, device=self.device)
        self.is_success = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.failed = torch.zeros_like(self.is_success)
        self.generator = torch.Generator(device=self.device).manual_seed(0)
        self.reset(seed=0)
        self.physics.compile()
        self.reset(seed=0)
        self.physics.check()

    def call(self, method, *args):
        if method == "provenance":
            return (self.provenance,)
        raise AttributeError(method)

    @torch.no_grad()
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.generator.manual_seed(seed)
        mask = torch.as_tensor(
            (options or {}).get(
                "reset_mask",
                torch.ones(self.num_envs, device=self.device, dtype=torch.bool),
            ),
            device=self.device,
            dtype=torch.bool,
        )
        self.physics.reset(mask)
        cfg = self.config
        random = (
            torch.rand(
                self.num_envs,
                1 + 3 * cfg.num_robots,
                device=self.device,
                generator=self.generator,
            )
            * 2
            - 1
        )
        angles = random[:, :1] * torch.pi + torch.arange(
            cfg.num_robots, device=self.device
        )[None] * (2 * torch.pi / cfg.num_robots)
        start = cfg.start_radius * torch.stack((angles.cos(), angles.sin()), -1)
        self.goals.copy_(torch.where(mask[:, None, None], -start, self.goals))
        jitter = random[:, 1:].reshape(self.num_envs, cfg.num_robots, 3)
        xy = start + jitter[..., :2] * cfg.position_jitter
        yaw = angles + torch.pi + jitter[..., 2] * cfg.yaw_jitter
        for robot, root in enumerate(self.reference.root_q):
            pose = torch.stack(
                (
                    xy[:, robot, 0],
                    xy[:, robot, 1],
                    torch.full_like(yaw[:, robot], STAND_HEIGHT),
                    (yaw[:, robot] / 2).cos(),
                    torch.zeros_like(yaw[:, robot]),
                    torch.zeros_like(yaw[:, robot]),
                    (yaw[:, robot] / 2).sin(),
                ),
                -1,
            )
            current = self.physics.qpos[:, root : root + 7]
            current.copy_(torch.where(mask[:, None], pose, current))
        joints = self.physics.qpos[:, self.physics.joint_q]
        self.physics.qpos[:, self.physics.joint_q] = torch.where(
            mask[:, None, None], self.stand, joints
        )
        self.commands.masked_fill_(mask[:, None, None], 0)
        self.steps.masked_fill_(mask, 0)
        self.success_time.masked_fill_(mask, 0)
        self.is_success.masked_fill_(mask, False)
        self.failed.masked_fill_(mask, False)
        self.physics.forward()
        return self.observation(), self.info()

    def measurements(self):
        xy = self.physics.xpos[:, self.physics.base_ids, :2]
        matrices = self.physics.xmat[:, self.physics.base_ids]
        yaw = torch.atan2(matrices[..., 1, 0], matrices[..., 0, 0])
        velocity = self.physics.qvel[
            :, self.root_v[:, None] + torch.arange(2, device=self.device)
        ]
        distances = (self.goals - xy).norm(dim=-1)
        pairs = (xy[:, self.pair_a] - xy[:, self.pair_b]).norm(dim=-1)
        return xy, yaw, velocity, distances, pairs

    def observation(self):
        xy, yaw, velocity, _, _ = self.measurements()
        return navigation_observation(
            xy, yaw, velocity, self.goals, self.commands, self.config, torch
        )

    def info(self):
        _, _, _, distances, pairs = self.measurements()
        return dict(
            is_success=self.is_success,
            distance=distances.mean(-1),
            min_separation=pairs.min(-1).values,
            failed=self.failed,
            robot_collision_fraction=self.physics.contacts.robot_collision_steps
            / (8 * self.config.high_level_decimation),
        )

    @torch.no_grad()
    def step(self, action):
        before = self.measurements()[3]
        self.commands.copy_(action.clamp(-1, 1))
        physical = self.commands * torch.where(self.commands >= 0, self.high, -self.low)
        self.physics.step(physical)
        self.steps += 1
        xy, _, _, distances, pairs = self.measurements()
        self.failed.copy_(
            (self.physics.xpos[:, self.physics.base_ids, 2] < 0.08).any(-1)
            | (self.physics.xmat[:, self.physics.base_ids, 2, 2] < 0.5).any(-1)
            | (xy.abs() > 3).any(-1).any(-1)
        )
        inside = (distances < self.config.goal_tolerance).all(-1)
        self.success_time.copy_(
            torch.where(inside, self.success_time + self.config.dt, 0)
        )
        self.is_success.copy_(
            inside
            & (self.success_time + 1e-8 >= self.config.hold_seconds)
            & ~self.failed
        )
        info = self.info()
        reward = navigation_reward(
            before,
            distances,
            pairs,
            info["robot_collision_fraction"],
            self.is_success,
            self.failed,
            self.config,
            torch,
        )
        return (
            self.observation(),
            reward,
            self.is_success | self.failed,
            self.steps * self.config.dt >= self.config.episode_seconds,
            info,
        )

    def render(self, world=0):
        ref = self.reference
        ref.data.qpos[:] = self.physics.qpos[world].cpu().numpy()
        ref.data.qvel[:] = self.physics.qvel[world].cpu().numpy()
        ref.goals = self.goals[world].cpu().numpy().copy()
        mujoco.mj_forward(ref.model, ref.data)
        return ref.render()

    def close(self):
        self.reference.close()
