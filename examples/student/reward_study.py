"""The fixed protocol used by the completed reward comparison in Colab."""

from dataclasses import replace

from hexapod_transport_rl import CylinderPushTask

BASELINE = CylinderPushTask()
PROGRESS40 = replace(BASELINE, rewards={**BASELINE.rewards, "goal_progress": 40.0})
TASKS = {"baseline": BASELINE, "progress40": PROGRESS40}

SEEDS = (42, 43, 44)
TRAINING = dict(
    steps=2_621_440,
    num_envs=256,
    horizon=64,
    validation_envs=256,
    validate_every=10,
)
EVALUATION = dict(episodes=256, seed=240000)
VIDEO_SEED = 250000
