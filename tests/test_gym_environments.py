"""Registered command and navigation environments obey the Gymnasium contract."""

import gymnasium as gym
import mujoco
import numpy as np
import pytest
import torch
from gymnasium.utils.env_checker import check_env

from hexapod_transport_rl import APPROACH_ENV_ID, WALKING_ENV_ID, ApproachConfig


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


@pytest.mark.parametrize("env_id", [WALKING_ENV_ID, APPROACH_ENV_ID])
def test_registered_environment_passes_gymnasium_checker(env_id):
    with gym.make(env_id) as env:
        check_env(env.unwrapped)
        obs, info = env.reset(seed=41)
        assert env.observation_space.contains(obs)
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool) and isinstance(truncated, bool)
        assert isinstance(info, dict)


@pytest.mark.parametrize("count", [1, 4])
def test_gym_walking_matches_pretrained_simulation_and_preserves_snapshots(count):
    with gym.make(WALKING_ENV_ID, num_robots=count, episode_seconds=0.4) as env:
        observation, info = env.reset(seed=7)
        initial = observation.copy()
        initial_position = info["positions"].copy()
        assert observation.shape == (count, 6)
        assert env.action_space.shape == (count, 3)
        action = np.tile([0.12, 0, 0], (count, 1)).astype(np.float32)
        _, reward, terminated, truncated, first = env.step(action)
        assert reward == 0 and not terminated and not truncated
        assert first["elapsed_seconds"] == pytest.approx(0.2)
        np.testing.assert_array_equal(first["commands"], action)
        assert env.unwrapped.sim._renderer is None
        obs, _, terminated, truncated, final = env.step(action)
        assert truncated and not terminated
        assert final["termination_reason"] == "time_limit"
        with pytest.raises(gym.error.ResetNeeded):
            env.step(action)
        np.testing.assert_array_equal(observation, initial)
        np.testing.assert_array_equal(info["positions"], initial_position)
        env.reset(seed=7)
        np.testing.assert_array_equal(obs[:, :2], final["positions"].astype(np.float32))


def test_walking_pose_options_invalid_actions_and_render_cleanup():
    env = gym.make(WALKING_ENV_ID, render_mode="rgb_array")
    with pytest.raises(gym.error.ResetNeeded):
        env.step(np.zeros((1, 3)))
    obs, _ = env.reset(seed=2, options={"poses": [[1.0, 0.0, np.pi / 2]]})
    np.testing.assert_allclose(obs[0, :3], [1, 0, np.pi / 2])
    for action in (np.zeros(3), np.full((1, 3), np.nan), [[0.3, 0, 0]]):
        with pytest.raises(ValueError):
            env.step(action)
    assert env.unwrapped.sim.time == 0
    image = env.render()
    assert image.shape == (480, 640, 3) and image.dtype == np.uint8
    assert image.std() > 1
    env.close()
    env.close()
    with pytest.raises(RuntimeError, match="closed"):
        env.reset()


def test_walking_fall_returns_termination_instead_of_a_runtime_exception():
    with gym.make(WALKING_ENV_ID) as env:
        env.reset(seed=2)
        sim = env.unwrapped.sim
        address = sim._walking.root_q[0]
        sim.data.qpos[address + 3 : address + 7] = [0, 1, 0, 0]
        mujoco.mj_forward(sim.model, sim.data)
        observation, reward, terminated, truncated, info = env.step(np.zeros((1, 3)))
        assert terminated and not truncated
        assert reward == 0
        assert info["robot_fall"] and info["termination_reason"] == "robot_fall"
        assert env.observation_space.contains(observation)
        env.reset(seed=2)
        _, _, terminated, _, _ = env.step(np.zeros((1, 3)))
        assert not terminated


def test_approach_flat_and_team_interfaces_have_identical_physics_and_rewards():
    config = {"layout": "front", "seconds": 0.2, "reward_weights": {"time": 0.4}}
    with (
        gym.make(APPROACH_ENV_ID, config=config) as team,
        gym.make(APPROACH_ENV_ID, config=config, flatten=True) as flat,
    ):
        obs_a, info_a = team.reset(seed=13)
        obs_b, info_b = flat.reset(seed=13)
        np.testing.assert_array_equal(obs_a.reshape(-1), obs_b)
        assert info_a["state"].shape == info_b["state"].shape == (20,)
        obs_a, reward_a, terminated, truncated, final = team.step(np.zeros((2, 3)))
        obs_b, reward_b, _, _, _ = flat.step(np.zeros(6))
        np.testing.assert_array_equal(obs_a.reshape(-1), obs_b)
        assert reward_a == reward_b == sum(final["reward_terms"].values())
        assert not terminated and truncated
        assert final["termination_reason"] == "time_limit"
        np.testing.assert_array_equal(obs_a.reshape(-1), final["state"])
        assert final["commands"].shape == (2, 3)
        state = final["state"].copy()
        team.reset(seed=14)
        np.testing.assert_array_equal(final["state"], state)


def test_approach_reset_layout_options_apply_to_one_episode_only():
    with gym.make(APPROACH_ENV_ID, config=ApproachConfig(layout="rear")) as env:
        first, info = env.reset(seed=4, options={"layout": "front"})
        assert info["layout"] == "front"
        again, info = env.reset(seed=4, options={"layout": "front"})
        np.testing.assert_array_equal(first, again)
        _, info = env.reset(seed=4)
        assert info["layout"] == "rear"
        assert env.unwrapped.config.layout == "rear"


def test_registered_walking_supports_standard_episode_statistics_wrapper():
    with gym.wrappers.RecordEpisodeStatistics(
        gym.make(WALKING_ENV_ID, episode_seconds=0.2)
    ) as env:
        env.reset(seed=2)
        _, _, _, truncated, info = env.step(np.zeros((1, 3)))
        assert truncated and info["episode"]["l"] == 1
        assert info["episode"]["r"] == 0
