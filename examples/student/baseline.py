"""Task definition matching the stable transport policy used by the lesson."""

from hexapod_transport_rl.student import CylinderPushTask

TASK = (
    CylinderPushTask.baseline()
)  # Use the lesson settings and initial checkpoint for reward experiments.
