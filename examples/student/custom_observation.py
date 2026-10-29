"""Add a scalar observation using physical quantities with named units."""

from hexapod_transport_rl.student import CylinderPushTask, ObservationTerm, PushState


def gap(state: PushState):
    """Gap between the robot centre and cylinder surface, in metres."""
    return state.surface_gap


TASK = CylinderPushTask(
    observations=(
        "cargo_to_robot",
        "cargo_to_goal",
        "robot_heading",
        ObservationTerm(gap, scale=0.5),
    ),
)
