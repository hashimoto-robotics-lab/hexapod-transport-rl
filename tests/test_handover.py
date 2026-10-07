"""The pusher's reset distribution must describe the actual physical states."""

import numpy as np
import torch

from hexapod_transport_rl.config import PushConfig, rotation
from hexapod_transport_rl.handover_training import HandoverPushEnv


def test_handover_resets_retain_original_cases_and_match_observations():
    torch.set_num_threads(1)
    env = HandoverPushEnv(PushConfig(shape="T"), flatten=False)
    original = adapted = 0
    for seed in range(20):
        obs, _ = env.reset(seed=seed)
        core = env.core
        local = (core.robot_xy - core.cargo_xy) @ rotation(core.cargo_yaw)
        if local[:, 0].mean() < -0.9:
            adapted += 1
            assert np.all(local[:, 0] >= -1.15 - 1e-8)
            assert np.all(local[:, 0] <= -0.95 + 1e-8)
        else:
            original += 1
        np.testing.assert_array_equal(obs, core.observe())
        state = core.data.qpos.copy()
        repeated, _ = env.reset(seed=seed)
        np.testing.assert_array_equal(obs, repeated)
        np.testing.assert_array_equal(state, core.data.qpos)
    assert original > 0 and adapted > 0
    assert env.provenance["reset_distribution"]["name"] == "approach_handover_v1"
    env.close()
