"""Fresh PPO baseline for fair reward/observation/action comparisons."""

from hexapod_transport_rl.student import CylinderPushTask

TASK = CylinderPushTask()  # Rotation disabled in all four teaching examples.
