"""Official Push-T semantics, exact area oracle, XY tracking and MAPPO reload."""

from dataclasses import asdict
from math import pi

import gymnasium as gym
import mujoco
import numpy as np
import pytest
import torch
from shapely.geometry import Polygon
from shapely.ops import unary_union
from torchrl.collectors import Collector
from torchrl.envs.utils import check_env_specs

from hexapod_transport_rl import (
    MAPPOSettings,
    PoseCurriculum,
    PushTConfig,
    PushTEnv,
    TorchRLTransportEnv,
    evaluate_pose,
    load_mappo,
    make_mappo_loss,
    make_mappo_networks,
    policy_action,
    save_mappo,
)
from hexapod_transport_rl.pusht_geometry import (
    coverage_numpy,
    coverage_torch,
    local_rectangles,
    transform_rectangles,
)
from hexapod_transport_rl.torchrl_mappo import PUSHT_FORMAT


@pytest.mark.parametrize("num_robots", [2, 4])
def test_coverage_matches_official_shapely_area(num_robots):
    """Includes overlap, no overlap, exact coincident edges and rotated pieces."""
    cfg = PushTConfig(num_robots=num_robots)
    local = local_rectangles(cfg)
    goal = unary_union([Polygon(p) for p in local])
    rng = np.random.default_rng(20261013)
    states = [(rng.uniform(-2, 2, 2), rng.uniform(-pi, pi)) for _ in range(200)]
    states += [
        (np.array([x, y]), angle)
        for x in [0, 0.01, 0.1, 0.2, 1.3]
        for y in [0, 0.1, 0.2]
        for angle in [0, pi / 2, pi / 4, pi]
    ]
    xy = np.array([s[0] for s in states])
    yaw = np.array([s[1] for s in states])
    expected = np.array(
        [
            unary_union([Polygon(p) for p in transform_rectangles(local, pos, angle)])
            .intersection(goal)
            .area
            / goal.area
            for pos, angle in states
        ]
    )
    native = np.array(
        [coverage_numpy(local, pos, angle, [0, 0], 0) for pos, angle in states]
    )
    tensor = coverage_torch(
        torch.tensor(local),
        torch.tensor(xy),
        torch.tensor(yaw),
        torch.zeros_like(torch.tensor(xy)),
        torch.zeros_like(torch.tensor(yaw)),
    )
    np.testing.assert_allclose(native, expected, atol=1e-12)
    np.testing.assert_allclose(tensor.numpy(), expected, atol=1e-12)
    if torch.cuda.is_available():
        gpu = coverage_torch(
            torch.tensor(local, device="cuda", dtype=torch.float32),
            torch.tensor(xy, device="cuda", dtype=torch.float32),
            torch.tensor(yaw, device="cuda", dtype=torch.float32),
            torch.zeros(len(states), 2, device="cuda"),
            torch.zeros(len(states), device="cuda"),
        )
        np.testing.assert_allclose(gpu.cpu().numpy(), expected, atol=2e-5)
    # Coverage is intersection / goal area, not intersection / union.
    actual = unary_union([Polygon(p) for p in transform_rectangles(local, [0.1, 0], 0)])
    assert (
        coverage_numpy(local, [0.1, 0], 0, [0, 0], 0)
        > actual.intersection(goal).area / actual.union(goal).area
    )


@pytest.mark.parametrize(
    "obs_type,num_robots,dimension",
    [("state", 2, 9), ("state", 4, 15), ("keypoints", 2, 22), ("keypoints", 4, 28)],
)
def test_official_observation_prefix_and_world_targets(obs_type, num_robots, dimension):
    cfg = PushTConfig(obs_type=obs_type, num_robots=num_robots)
    env = gym.make("HexapodPushT-v0", config=cfg)
    observation, info = env.reset(seed=21)
    core = env.unwrapped.core
    assert observation.shape == (num_robots, dimension)
    assert env.action_space.shape == (num_robots, 2)
    np.testing.assert_allclose(core.goal, cfg.goal_pose[:2])
    assert core.goal_yaw == cfg.goal_pose[2]
    headings = core.yaw(core.base_ids) % (2 * pi)
    block = np.r_[core.cargo_xy, core.cargo_yaw % (2 * pi)]
    for i in range(num_robots):
        if obs_type == "state":
            np.testing.assert_allclose(
                observation[i, :5], np.r_[core.robot_xy[i], block], atol=1e-6
            )
        else:
            np.testing.assert_allclose(
                observation[i, :18],
                np.r_[
                    transform_rectangles(
                        core.rectangles, core.cargo_xy, core.cargo_yaw
                    ).ravel(),
                    core.robot_xy[i],
                ],
                atol=1e-6,
            )
        assert observation[i, 5 if obs_type == "state" else 18] == pytest.approx(
            headings[i]
        )
    direction = (core.goal - core.cargo_xy) / np.linalg.norm(core.goal - core.cargo_xy)
    assert ((core.robot_xy - core.goal) @ direction > 0).all()
    action = core.robot_xy.astype(np.float32)
    _, reward, _, _, info = env.step(action)
    assert reward == pytest.approx(np.clip(info["coverage"] / 0.95, 0, 1))
    assert info["reward_terms"]["time"] == info["reward_terms"]["body_contact"] == 0
    env.close()


def test_success_is_coverage_immediate_not_old_pose_tolerances():
    env = PushTEnv(PushTConfig())
    env.reset(seed=42)
    core = env.core
    q = core.cargo_q
    core.data.qpos[q : q + 2] = core.goal
    core.data.qpos[q + 3 : q + 7] = [
        np.cos(core.goal_yaw / 2),
        0,
        0,
        np.sin(core.goal_yaw / 2),
    ]
    mujoco.mj_forward(core.model, core.data)
    _, reward, terminated, truncated, info = env.step(core.robot_xy.astype(np.float32))
    assert terminated and not truncated and info["is_success"]
    assert info["coverage"] > 0.95 and reward == pytest.approx(1)
    assert info["elapsed_seconds"] == 0.2  # no one-second hold condition
    env.close()


@pytest.mark.parametrize("backend", ["cpu", "warp"])
@pytest.mark.parametrize("num_robots", [2, 4])
def test_xy_mappo_real_update_and_checkpoint_reload(tmp_path, backend, num_robots):
    if backend == "warp" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.set_num_threads(1)
    cfg = PushTConfig(num_robots=num_robots, episode_seconds=0.4)
    env = TorchRLTransportEnv(cfg, num_envs=2, asynchronous=False, backend=backend)
    check_env_specs(env)
    actor, critic = make_mappo_networks(
        num_robots, env.obs_dim, action_dim=2, action_low=-4, action_high=4
    )
    actor.to(env.device)
    critic.to(env.device)
    settings = MAPPOSettings(gamma=0.995, value_normalization=True)
    loss = make_mappo_loss(actor, critic, settings)
    optimizer = torch.optim.Adam(loss.parameters(), lr=settings.learning_rate)
    # Coverage has no potential shaping term, so it does not constrain PPO gamma.
    log = PoseCurriculum(
        cfg,
        env,
        actor,
        tmp_path / "training",
        seed=42,
        settings=settings,
        horizon=4,
        validate_every=None,
    )
    assert log.state["stage"] == "fixed_task"
    assert log.training["success_criterion"] == {
        "type": "coverage",
        "threshold": 0.95,
        "hold_required": False,
    }
    collector = Collector(
        env,
        actor,
        frames_per_batch=8,
        total_frames=8,
        auto_register_policy_transforms=True,
        env_device=env.device,
        policy_device=env.device,
        storing_device=env.device,
    )
    try:
        before = [p.detach().clone() for p in actor.parameters()]
        batch = next(iter(collector))
        assert batch["agents", "action"].shape == (2, 4, num_robots, 2)
        loss.value_estimator(batch)
        metrics = loss(batch.reshape(-1))
        optimizer.zero_grad()
        (
            metrics["loss_objective"] + metrics["loss_critic"] + metrics["loss_entropy"]
        ).backward()
        optimizer.step()
        assert any(
            not torch.equal(a, b)
            for a, b in zip(before, actor.parameters(), strict=True)
        )
        obs = batch["agents", "observation"][0, 0].cpu().numpy()
        expected = policy_action(actor, obs)
        path = save_mappo(
            tmp_path / "pusht.pt",
            actor,
            critic,
            optimizer,
            config=cfg,
            provenance=env.provenance,
            training={"settings": asdict(settings)},
            curriculum={},
            loss=loss,
        )
        loaded, _, saved = load_mappo(path)
        np.testing.assert_allclose(policy_action(loaded, obs), expected, atol=1e-5)
        assert saved["format"] == PUSHT_FORMAT and saved["network"]["action_dim"] == 2
        assert type(cfg).from_checkpoint(saved["pose_config"]) == cfg
        report = evaluate_pose(path, episodes=2, workers=1)
        assert "mean_coverage" in report and "mean_max_coverage" in report
        if backend == "warp":
            gpu = evaluate_pose(path, episodes=2, backend="warp")
            assert "mean_coverage" in gpu
    finally:
        log.close()
        collector.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_partial_reset_preserves_fixed_goal_and_native_observations():
    cfg = PushTConfig(obs_type="keypoints")
    env = TorchRLTransportEnv(cfg, num_envs=2, backend="warp")
    worlds = env.worlds
    native = PushTEnv(cfg)
    try:
        for lane in range(2):
            obs, _ = native.reset(seed=42 + lane)
            core = native.core
            worlds.physics.qpos[lane].copy_(
                torch.as_tensor(core.data.qpos, device=env.device)
            )
            worlds.physics.qvel[lane].copy_(
                torch.as_tensor(core.data.qvel, device=env.device)
            )
            worlds.goal[lane].copy_(torch.as_tensor(core.goal, device=env.device))
            worlds.goal_yaw[lane] = core.goal_yaw
            worlds.physics.forward()
            np.testing.assert_allclose(
                worlds.observe()[lane].cpu().numpy(), obs, atol=2e-6
            )
            assert float(worlds.measurements()["coverage"][lane]) == pytest.approx(
                core.info()["coverage"], abs=2e-6
            )
        q = worlds.physics.qpos[1].clone()
        memory = worlds.physics.previous_action[1].clone()
        worlds.reset(
            seed=99,
            options={"reset_mask": torch.tensor([True, False], device=env.device)},
        )
        torch.testing.assert_close(worlds.physics.qpos[1], q, atol=0, rtol=0)
        torch.testing.assert_close(
            worlds.physics.previous_action[1], memory, atol=0, rtol=0
        )
        np.testing.assert_allclose(
            worlds.goal.cpu().numpy(), np.tile(cfg.goal_pose[:2], (2, 1))
        )
    finally:
        native.close()
        env.close()
