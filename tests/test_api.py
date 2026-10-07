"""Exercise public RL contracts and real physics, including final transitions."""

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium.utils.env_checker import check_env

from hexapod_transport_rl import GYM_ENV_ID, HexapodPushEnv, PushConfig


@pytest.fixture(autouse=True, scope="module")
def cpu_threads():
    torch.set_num_threads(1)


def test_registered_gymnasium_contract():
    with gym.make(GYM_ENV_ID) as env:
        check_env(env.unwrapped, skip_render_check=True)
        obs, info = env.reset(seed=41)
        assert obs.shape == (40,) and info["state"].shape == (40,)
        assert env.action_space.shape == (6,)
        assert env.observation_space.contains(obs)
        assert env.unwrapped.provenance["walking_checkpoint_iteration"] == 1800


@pytest.mark.parametrize("n,shape", [(3, "box"), (4, "T")])
def test_team_dimensions_and_config_dict(n, shape):
    with HexapodPushEnv(config={"num_robots": n, "shape": shape}, flatten=False) as env:
        obs, _ = env.reset(seed=2)
        assert obs.shape == (n, 16 + 4 * (n - 1))
        assert env.action_space.shape == (n, 3)
        obs, reward, _, _, _ = env.step(np.zeros((n, 3)))
        assert env.observation_space.contains(obs) and np.isfinite(reward)
        assert env.state_space.contains(env.state())


def test_seed_continuation_and_readiness():
    env = HexapodPushEnv()
    with pytest.raises(gym.error.ResetNeeded):
        env.step(np.zeros(6))
    first, _ = env.reset(seed=19)
    second, _ = env.reset()
    assert not np.array_equal(first, second)
    repeated, _ = env.reset(seed=19)
    np.testing.assert_array_equal(first, repeated)
    repeated_second, _ = env.reset()
    np.testing.assert_array_equal(second, repeated_second)
    env.reset(seed=20, options={"randomize": False})
    unrandomized, _ = env.reset(seed=21, options={"randomize": False})
    expected, _ = env.reset(seed=20, options={"randomize": False})
    np.testing.assert_array_equal(expected, unrandomized)
    env.close()
    env.close()
    with pytest.raises(RuntimeError, match="closed"):
        env.reset()


def test_bad_actions_do_not_advance_physics():
    with HexapodPushEnv() as env:
        env.reset(seed=7)
        initial = env.state()
        for action in (
            np.zeros((2, 3)),
            np.full(6, np.nan),
            np.full(6, np.inf),
            np.full(6, 1.1),
        ):
            with pytest.raises(ValueError, match="Action"):
                env.step(action)
        np.testing.assert_array_equal(initial, env.state())
        assert env.core.data.time == 0


def test_timeout_final_observation_and_success_reason():
    with HexapodPushEnv(PushConfig(episode_seconds=0.2), randomize=False) as env:
        env.reset(seed=0)
        final, reward, terminated, truncated, info = env.step(np.zeros(6))
        assert truncated and not terminated
        assert info["termination_reason"] == "time_limit"
        assert info["episode_return"] == reward == info["team_reward"]
        np.testing.assert_array_equal(final, info["state"])
        np.testing.assert_array_equal(final, env.state())
        env.reset(seed=0)
        np.testing.assert_array_equal(final, info["state"])
        assert not np.array_equal(final, env.state())
    with HexapodPushEnv(PushConfig(episode_seconds=1), randomize=False) as env:
        env.reset(seed=0)
        env.core.goal = env.core.cargo_xy.copy()
        for _ in range(3):
            _, _, terminated, truncated, info = env.step(np.zeros(6))
        assert terminated and not truncated
        assert info["is_success"] and info["termination_reason"] == "success"
        with pytest.raises(gym.error.ResetNeeded):
            env.step(np.zeros(6))


def test_rgb_array_render_and_cleanup():
    with HexapodPushEnv(render_mode="rgb_array", width=320, height=240) as env:
        env.reset(seed=0)
        frame = env.render()
        assert frame.shape == (240, 320, 3) and frame.dtype == np.uint8
        assert frame.std() > 1
        env.step(np.zeros(6))
        assert not np.array_equal(frame, env.render())
    assert env._renderer is None
