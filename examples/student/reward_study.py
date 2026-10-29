"""The fixed protocol used by the completed reward comparison in Colab."""

from dataclasses import replace

from hexapod_transport_rl import CylinderPushTask, PPOSettings

BASELINE = CylinderPushTask.baseline()
PROGRESS40 = replace(BASELINE, rewards={**BASELINE.rewards, "goal_progress": 40.0})
TASKS = {"baseline": BASELINE, "progress40": PROGRESS40}

SEEDS = (42, 43, 44)
TRAINING = dict(
    steps=655_360,
    num_envs=512,
    horizon=64,
    validation_envs=256,
    validate_every=5,
    initial_checkpoint="checkpoints/cylinder_push.pt",
    checkpoint_selection="latest",
    settings=PPOSettings(learning_rate=3e-5, entropy_coeff=0.001),
)
EVALUATION = dict(episodes=1024, seed=540000)
VIDEO_SEED = 550000
