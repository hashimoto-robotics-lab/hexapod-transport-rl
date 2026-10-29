"""Change only the reward: penalize cylinder motion near the goal."""

from dataclasses import replace

from hexapod_transport_rl.student import CylinderPushTask, PushState, RewardTerm


def near_goal_speed(before: PushState, after: PushState):
    """Nonnegative (m/s)^2 cost; one value for each parallel world."""
    return after.cargo_speed.square() * (after.goal_distance < 0.10)


baseline = CylinderPushTask.baseline()
TASK = replace(
    baseline,
    rewards={
        **baseline.rewards,
        "near_goal_speed": RewardTerm(near_goal_speed, weight=-2.0, per_second=True),
    },
)
