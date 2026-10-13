"""GPU Push-T task; same target tracker, footprint and success semantics as Gym."""

import torch

from .pusht import PushTEnv, pusht_reward_terms, target_position_commands
from .pusht_geometry import coverage_torch, local_rectangles, transform_rectangles_torch
from .warp_pose import WarpPoseWorlds, cargo_coordinates


class WarpPushTWorlds(WarpPoseWorlds):
    reference_env_type = PushTEnv

    def __init__(self, config, num_envs, asset_root=None, device="cuda:0"):
        self.rectangles = torch.as_tensor(
            local_rectangles(config), device=device, dtype=torch.float32
        )
        super().__init__(config, num_envs, asset_root, device)

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

    def observe(self):
        p = self.physics
        xy = p.xpos[:, p.base_ids, :2]
        headings = torch.atan2(
            p.xmat[:, p.base_ids, 1, 0], p.xmat[:, p.base_ids, 0, 0]
        ).remainder(2 * torch.pi)
        cargo_xy, yaw, _, _ = self.coordinates()
        block = torch.cat((cargo_xy, yaw.remainder(2 * torch.pi)[:, None]), -1)
        keypoints = transform_rectangles_torch(self.rectangles, cargo_xy, yaw).flatten(
            1
        )
        rows = []
        for i in range(self.num_robots):
            standard = (
                torch.cat((xy[:, i], block), -1)
                if self.config.obs_type == "state"
                else torch.cat((keypoints, xy[:, i]), -1)
            )
            peers = [
                torch.cat((xy[:, j], headings[:, j, None]), -1)
                for j in range(self.num_robots)
                if j != i
            ]
            rows.append(torch.cat((standard, headings[:, i, None], *peers), -1))
        return torch.stack(rows, 1)

    def measurements(self):
        info = super().measurements()
        xy, yaw, _, _ = self.coordinates()
        info["coverage"] = coverage_torch(
            self.rectangles, xy, yaw, self.goal, self.goal_yaw
        )
        return info

    def status(self, info):
        super().status(info)
        success = (info["coverage"] > self.config.coverage_threshold) & ~info["failed"]
        info.update(success=success, is_success=success)
        terminated = success | info["failed"]
        truncated = (
            self.steps * self.config.dt + 1e-7 >= self.config.episode_seconds
        ) & ~terminated
        return terminated, truncated

    def reward_terms(self, before, after, commands):
        contacts = self.physics.contacts
        return pusht_reward_terms(
            self.config,
            after,
            commands,
            self.last_commands,
            contacts.part_contact_counts[:, 0].float(),
            contacts.robot_collision_steps,
            math=torch,
            command_limits=self.limits,
        )

    @torch.no_grad()
    def step(self, targets):
        extent = self.config.workspace_half_extent
        targets = targets.clamp(-extent, extent)
        p = self.physics
        headings = torch.atan2(p.xmat[:, p.base_ids, 1, 0], p.xmat[:, p.base_ids, 0, 0])
        velocity = p.qvel[:, self.root_v[:, None] + torch.arange(2, device=self.device)]
        commands = target_position_commands(
            self.config,
            targets,
            p.xpos[:, p.base_ids, :2],
            headings,
            velocity,
            math=torch,
            low=self.low,
            high=self.high,
        )
        normalized = commands / torch.where(commands >= 0, self.high, -self.low)
        # Parent's velocity API mirrors roles. Undo that input convention here;
        # position targets themselves are world coordinates, with no reflection.
        return super().step(normalized * self.signs)
