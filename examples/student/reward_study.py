"""The fixed protocol used by the completed reward comparison in Colab."""

from dataclasses import replace

from hexapod_transport_rl import CylinderPushTask, PPOSettings, PushCurriculum

BASELINE = CylinderPushTask.baseline()
PROGRESS40 = replace(BASELINE, rewards={**BASELINE.rewards, "goal_progress": 40.0})
TASKS = {"baseline": BASELINE, "progress40": PROGRESS40}

SEEDS = (42, 43, 44)
TRAINING = dict(
    steps=9_830_400,
    num_envs=512,
    horizon=64,
    validation_envs=256,
    validate_every=10,
    initial_checkpoint=None,
    checkpoint_selection="best",
    curriculum=PushCurriculum(),
    settings=PPOSettings(),
)
EVALUATION = dict(episodes=1024, seed=840000)
VIDEO_SEED = 850000
ACCEPTANCE = dict(
    baseline_mean_success_rate_at_least=0.98,
    every_baseline_seed_success_rate_at_least=0.95,
    every_baseline_goal_sector_success_rate_at_least=0.90,
)
