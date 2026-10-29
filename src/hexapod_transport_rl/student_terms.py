"""Named, batched physical state and pure tensor terms for student experiments."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PushState:
    """World XY in metres, velocities in m/s, angles in radians.

    Scalars have shape (B,), vectors (B, 2), and heading is (sin, cos).
    Each state owns its tensors; subsequent physics steps cannot overwrite it.
    These fields are available to rewards/diagnostics. The actor receives only
    the observations explicitly selected in CylinderPushTask.
    """

    robot_position: torch.Tensor
    cargo_position: torch.Tensor
    goal_position: torch.Tensor
    robot_yaw: torch.Tensor
    cargo_yaw: torch.Tensor
    robot_velocity: torch.Tensor
    cargo_velocity: torch.Tensor
    robot_yaw_rate: torch.Tensor
    cargo_yaw_rate: torch.Tensor
    goal_distance: torch.Tensor
    approach_distance: torch.Tensor
    success: torch.Tensor
    failed: torch.Tensor
    elapsed_seconds: torch.Tensor
    command_velocity: torch.Tensor
    baseline_observation: torch.Tensor
    cargo_radius: float
    dt: float

    @property
    def robot_to_cargo(self):
        return self.cargo_position - self.robot_position

    @property
    def cargo_to_robot(self):
        return self.robot_position - self.cargo_position

    @property
    def cargo_to_goal(self):
        return self.goal_position - self.cargo_position

    @property
    def robot_heading(self):
        return torch.stack((self.robot_yaw.sin(), self.robot_yaw.cos()), -1)

    @property
    def cargo_speed(self):
        return self.cargo_velocity.norm(dim=-1)

    @property
    def surface_gap(self):
        return (self.robot_to_cargo.norm(dim=-1) - self.cargo_radius).clamp_min(0)


OBSERVATIONS = {
    "baseline": lambda s: s.baseline_observation,
    "robot_to_cargo": lambda s: s.robot_to_cargo,
    "cargo_to_robot": lambda s: s.cargo_to_robot,
    "cargo_to_goal": lambda s: s.cargo_to_goal,
    "robot_heading": lambda s: s.robot_heading,
    "robot_velocity": lambda s: s.robot_velocity,
    "cargo_velocity": lambda s: s.cargo_velocity,
    "robot_yaw_rate": lambda s: s.robot_yaw_rate,
    "cargo_yaw_rate": lambda s: s.cargo_yaw_rate,
    "goal_distance": lambda s: s.goal_distance,
    "surface_gap": lambda s: s.surface_gap,
    "command_velocity": lambda s: s.command_velocity,
}


# (function, integrate over seconds). Progress and terminal bonuses are per
# transition. State costs are rates; the environment multiplies them by dt.
REWARDS = {
    "goal_progress": (
        lambda before, after: before.goal_distance - after.goal_distance,
        False,
    ),
    "approach_progress": (
        lambda before, after: before.approach_distance - after.approach_distance,
        False,
    ),
    "goal_distance": (lambda before, after: after.goal_distance, True),
    "approach_distance": (lambda before, after: after.approach_distance, True),
    "time": (lambda before, after: torch.ones_like(after.goal_distance), True),
    "success": (lambda before, after: after.success.float(), False),
    "failure": (lambda before, after: after.failed.float(), False),
    "cargo_speed": (lambda before, after: after.cargo_speed, True),
    "command_change": (
        lambda before, after: (
            (after.command_velocity - before.command_velocity).square().sum(-1)
        ),
        False,
    ),
}

DEFAULT_REWARDS = {
    "goal_progress": 20.0,
    "approach_progress": 2.0,
    "goal_distance": -4.0,
    "approach_distance": -1.0,
    "time": -0.02,
    "success": 10.0,
    "failure": -10.0,
}
