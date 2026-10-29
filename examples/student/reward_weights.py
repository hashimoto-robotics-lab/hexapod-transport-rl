"""Increase the reward for reducing the cylinder-to-goal distance."""

from dataclasses import replace

from hexapod_transport_rl import CylinderPushTask

baseline = CylinderPushTask()
TASK = replace(baseline, rewards={**baseline.rewards, "goal_progress": 40.0})
