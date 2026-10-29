"""Construct six named inputs; remove velocity information from the policy."""

from hexapod_transport_rl.student import CylinderPushTask, ObservationTerm

baseline = CylinderPushTask()
TASK = CylinderPushTask(
    observations=(
        ObservationTerm("cargo_to_robot", scale=baseline.config.width),
        ObservationTerm("cargo_to_goal", scale=baseline.config.width),
        ObservationTerm("robot_heading"),
    ),
)
