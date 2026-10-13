"""Real hexapod crossing, shared reward and fresh TorchRL learning."""

from dataclasses import replace

import mediapy as media
import numpy as np
import pytest
import torch
from gymnasium.utils.env_checker import check_env
from torchrl.envs.utils import check_env_specs

from hexapod_transport_rl import (
    MAPPOSettings,
    NavigationConfig,
    NavigationEnv,
    NavigationRewardWeights,
    TorchRLTransportEnv,
    evaluate_mappo,
    load_mappo,
    train_mappo,
)
from hexapod_transport_rl.navigation import navigation_observation, navigation_reward


@pytest.mark.parametrize("num_robots", [2, 4])
def test_gym_contract_body_observation_and_real_video(num_robots):
    cfg = NavigationConfig(num_robots=num_robots, episode_seconds=0.8)
    env = NavigationEnv(cfg)
    try:
        check_env(env, skip_render_check=True)
        env.render_mode = "rgb_array_list"
        obs, info = env.reset(seed=42)
        assert obs.shape == (num_robots, cfg.observation_dim)
        assert np.all(obs[:, 0] > 0.8)  # Own opposite goal starts ahead in body axes.
        initial = info["robot_positions"].copy()
        obs, reward, _, _, info = env.step(np.tile([0.8, 0, 0], (num_robots, 1)))
        assert np.linalg.norm(info["robot_positions"] - initial) > 0.01
        assert np.isfinite(reward)
        assert len(env.render()) == 11  # Reset frame plus 0.4s at actual 25 Hz.
        assert env.model.nq == 25 * num_robots  # No object or proxy dynamics.
    finally:
        env.close()


def test_observation_and_reward_numpy_torch_agree():
    cfg = NavigationConfig(num_robots=4)
    rng = np.random.default_rng(5)
    xy, velocity, goals = [rng.normal(size=(3, 4, 2)) for _ in range(3)]
    yaw, commands = rng.normal(size=(3, 4)), rng.normal(size=(3, 4, 3))
    args = (xy, yaw, velocity, goals, commands)
    expected = navigation_observation(*args, cfg, np)
    actual = navigation_observation(*(torch.tensor(x) for x in args), cfg, torch)
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-12)
    args = (
        rng.random((3, 4)),
        rng.random((3, 4)),
        rng.random((3, 6)),
        rng.random(3),
        np.array([True, False, False]),
        np.array([False, False, True]),
    )
    expected = navigation_reward(*args, cfg, np)
    actual = navigation_reward(*(torch.tensor(x) for x in args), cfg, torch)
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-12)
    # Removing the safety terms changes reward only when robots get close/contact.
    unsafe = replace(
        cfg, reward_weights=NavigationRewardWeights(separation=0, collision=0)
    )
    no_penalty = navigation_reward(*args, unsafe, np)
    assert np.all(no_penalty >= expected)


@pytest.mark.parametrize("backend", ["cpu", "warp"])
def test_safety_metric_detects_actual_robot_contact(backend):
    if backend == "warp" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    cfg = NavigationConfig(high_level_decimation=1)
    if backend == "cpu":
        env = NavigationEnv(cfg)
        try:
            # Touching feet: this must count even though there is no cargo.
            env.sim.reset(poses=[[0, 0, 0], [0.45, 0, 0]])
            info = env.step(np.zeros((2, 3), np.float32))[-1]
            assert info["robot_collision_fraction"] > 0
        finally:
            env.close()
    else:
        from hexapod_transport_rl.warp_navigation import WarpNavigationWorlds

        worlds = WarpNavigationWorlds(cfg, 2)
        try:
            for robot, root in enumerate(worlds.reference.root_q):
                worlds.physics.qpos[:, root : root + 7] = torch.tensor(
                    [0.45 * robot, 0, 0.16, 1, 0, 0, 0], device=worlds.device
                )
            worlds.physics.forward()
            info = worlds.step(torch.zeros(2, 2, 3, device=worlds.device))[-1]
            assert bool((info["robot_collision_fraction"] > 0).all())
            worlds.physics.check()
        finally:
            worlds.close()


@pytest.mark.parametrize("backend", ["cpu", "warp"])
@pytest.mark.parametrize("num_robots", [2, 4])
def test_torchrl_update_reload_and_evaluate(tmp_path, backend, num_robots):
    if backend == "warp" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    cfg = NavigationConfig(
        num_robots=num_robots, episode_seconds=0.8, high_level_decimation=5
    )
    env = TorchRLTransportEnv(cfg, num_envs=2, backend=backend, asynchronous=False)
    try:
        check_env_specs(env)
        if backend == "warp":
            worlds = env.worlds
            previous = worlds.physics.previous_action[1].clone()
            qpos, goal = worlds.physics.qpos[1].clone(), worlds.goals[1].clone()
            worlds.reset(seed=17, options={"reset_mask": [True, False]})
            torch.testing.assert_close(qpos, worlds.physics.qpos[1], atol=0, rtol=0)
            torch.testing.assert_close(goal, worlds.goals[1], atol=0, rtol=0)
            torch.testing.assert_close(
                previous, worlds.physics.previous_action[1], atol=0, rtol=0
            )
    finally:
        env.close()
    path = train_mappo(
        cfg,
        tmp_path,
        seed=17,
        steps=8,
        num_envs=2,
        horizon=4,
        max_seconds=None,
        backend=backend,
        settings=MAPPOSettings(epochs=1, minibatch_size=8, value_normalization=True),
    )
    _, _, saved = load_mappo(path)
    initial = torch.load(tmp_path / "initial.pt", weights_only=True, map_location="cpu")
    assert not initial["optimizer"]["state"]
    assert float(initial["value_normalizer"]["debiasing_term"]) == 0
    assert not saved["training"]["demonstration_actions_used"]
    assert any(
        not torch.equal(v, saved["actor"][k]) for k, v in initial["actor"].items()
    )
    report = evaluate_mappo(path, episodes=2, backend=backend, video=num_robots == 2)
    assert len(report["episodes"]) == 2
    if num_robots == 2:
        assert len(media.read_video(report["video_path"])) == 21
