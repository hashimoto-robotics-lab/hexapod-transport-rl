from contextlib import closing

import numpy as np
import pytest
import torch

from hexapod_transport_rl import make_vector_env


@pytest.mark.parametrize("asynchronous", [False, True])
def test_vector_partial_reset_preserves_other_world_and_final_state(asynchronous):
    with closing(
        make_vector_env(
            2, config={"episode_seconds": 0.4}, flatten=False, asynchronous=asynchronous
        )
    ) as envs:
        obs, _ = envs.reset(seed=15)
        assert obs.shape == (2, 2, 20)
        assert envs.action_space.shape == (2, 2, 3)
        actions = np.zeros((2, 2, 3), dtype=np.float32)
        first, _, _, _, _ = envs.step(actions)
        reset_obs, _ = envs.reset(options={"reset_mask": np.array([True, False])})
        np.testing.assert_array_equal(reset_obs[1], first[1])
        assert not np.array_equal(reset_obs[0], first[0])
        final, reward, term, trunc, infos = envs.step(actions)
        assert reward.shape == term.shape == trunc.shape == (2,)
        np.testing.assert_array_equal(trunc, [False, True])
        assert not term.any()
        np.testing.assert_array_equal(infos["elapsed_steps"], [1, 2])
        np.testing.assert_array_equal(final[1].ravel(), infos["state"][1])
        saved_final = final.copy()
        new, _ = envs.reset(options={"reset_mask": term | trunc})
        np.testing.assert_array_equal(new[0], final[0])
        np.testing.assert_array_equal(saved_final, final)
        assert not np.array_equal(new[1], final[1])
        _, _, _, trunc, infos = envs.step(actions)
        np.testing.assert_array_equal(trunc, [True, False])
        np.testing.assert_array_equal(infos["elapsed_steps"], [2, 1])
    if asynchronous:
        assert all(not process.is_alive() for process in envs.processes)


def test_parallel_training_matches_serial_and_can_resume(tmp_path):
    """Parallel physics must preserve seeded PPO updates and checkpoint resume."""
    from hexapod_transport_rl import PushConfig
    from hexapod_transport_rl.mappo import load_checkpoint, train

    cfg = PushConfig(shape="T", episode_seconds=0.4)
    checkpoints = []
    for backend in ("sync", "async"):
        path = train(
            cfg,
            None,
            tmp_path / backend,
            iterations=2,
            num_envs=2,
            horizon=4,
            epochs=1,
            minibatch=4,
            seed=9,
            backend=backend,
        )
        checkpoints.append(load_checkpoint(path)[1])
    serial, parallel = checkpoints
    assert serial["transitions"] == parallel["transitions"] == 16
    assert serial["run"]["backend"] == "SyncVectorEnv"
    assert parallel["run"]["backend"] == "AsyncVectorEnv"
    for name, tensor in serial["model"].items():
        torch.testing.assert_close(tensor, parallel["model"][name], rtol=0, atol=0)
    for parameter, state in serial["optimizer"]["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(
                value, parallel["optimizer"]["state"][parameter][key], rtol=0, atol=0
            )
    resumed = train(
        cfg,
        None,
        tmp_path / "resumed",
        iterations=1,
        num_envs=2,
        horizon=2,
        epochs=1,
        minibatch=4,
        seed=10,
        resume=tmp_path / "sync/checkpoint.pt",
        backend="async",
    )
    _, saved = load_checkpoint(resumed)
    assert saved["iteration"] == 3 and saved["transitions"] == 20
