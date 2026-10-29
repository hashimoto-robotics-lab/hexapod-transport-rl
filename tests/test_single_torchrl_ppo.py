"""Single-agent shapes, physical task boundaries and actual TorchRL PPO updates."""

import json
from dataclasses import replace

import mujoco
import numpy as np
import pytest
import torch
from torchrl.envs.utils import check_env_specs

from hexapod_transport_rl.single_push import (
    DIRECTIONS,
    SinglePushAzimuthStage,
    SinglePushConfig,
    SinglePushEnv,
    SinglePushGoalStage,
    SinglePushGymEnv,
    SinglePushPoseStage,
    SinglePushResetStage,
    around_start_pose,
    circle_coverage,
    cylinder_approach,
    direction_commands,
    random_goal_delta,
)
from hexapod_transport_rl.torchrl_ppo import (
    GoalOctantMLP,
    PPOSettings,
    evaluate_ppo,
    load_ppo,
    make_ppo_loss,
    make_ppo_networks,
    train_ppo,
)


def test_nine_world_directions_and_original_walker_stop_commands():
    directions = np.asarray(DIRECTIONS)
    np.testing.assert_array_equal(directions[0], [0, 0])
    np.testing.assert_allclose(np.linalg.norm(directions[1:], axis=-1), 1)
    assert len(np.unique(directions, axis=0)) == 9
    env = SinglePushGymEnv(SinglePushConfig())
    try:
        env.reset(seed=42)
        core = env.core
        assert core.controller is not None and core.model.nq == 32
        core.step(np.zeros((1, 2)))
        np.testing.assert_array_equal(core.last_commands, np.zeros((1, 3)))
        assert not np.any(core.data.xfrc_applied)
    finally:
        env.close()


@pytest.mark.parametrize("shape", ["cylinder"])
def test_position_success_requires_a_hold_but_not_the_cargo_orientation(shape):
    cfg = SinglePushConfig(shape=shape)
    env = SinglePushGymEnv(cfg)
    try:
        env.reset(seed=42)
        core = env.core
        q = core.cargo_q
        core.data.qpos[q : q + 2] = cfg.goal_pose[:2]
        # Zero yaw intentionally differs from the target's 45 degrees.
        core.data.qpos[q + 3 : q + 7] = [1, 0, 0, 0]
        core.data.qvel[:] = 0
        mujoco.mj_forward(core.model, core.data)
        info = core.info()
        core._update_episode_status(info)
        assert not info["success"]
        if shape == "T":
            assert info["coverage"] < 0.95
        else:
            assert info["coverage"] == pytest.approx(1)
        core._update_episode_status(info)
        assert info["success"]
        core.data.qpos[q] += cfg.position_tolerance
        mujoco.mj_forward(core.model, core.data)
        info = core.info()
        info["distance"] = cfg.position_tolerance
        core._update_episode_status(info)
        assert not info["success"]
        core.data.qpos[q : q + 2] = cfg.goal_pose[:2]
        core.data.qpos[q + 3 : q + 7] = [0, 1, 0, 0]
        mujoco.mj_forward(core.model, core.data)
        info = core.info()
        core._update_episode_status(info)
        assert info["failed"] and not info["success"]
    finally:
        env.close()


@pytest.mark.parametrize("backend", ["cpu", "warp"])
@pytest.mark.parametrize("mode", ["discrete", "continuous"])
@pytest.mark.parametrize("shape", ["cylinder"])
def test_flat_specs_partial_reset_and_actual_ppo_gradients(backend, mode, shape):
    if backend == "warp" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    env = SinglePushEnv(
        SinglePushConfig(episode_seconds=0.4, shape=shape),
        action_mode=mode,
        num_envs=2,
        backend=backend,
    )
    try:
        check_env_specs(env)
        env.set_seed(42)
        before = env.reset()
        assert before["observation"].shape == (2, 14)
        reset = before.clone()
        reset["_reset"] = torch.tensor([[True], [False]], device=env.device)
        after = env.reset(reset)
        torch.testing.assert_close(before["observation"][1], after["observation"][1])
        actor, critic = make_ppo_networks(mode, device=env.device)
        if backend == "warp":
            from hexapod_transport_rl.warp_step_graph import capture_world_step

            capture_world_step(env.worlds)
        with torch.no_grad():
            batch = env.rollout(4, actor, break_when_any_done=False)
            preserved = batch.clone()
            env.rollout(4, actor, break_when_any_done=False)
            torch.testing.assert_close(batch["observation"], preserved["observation"])
        assert "agents" not in batch.keys()
        assert batch["next", "truncated"].any()
        assert not batch["next", "terminated"].any()
        loss = make_ppo_loss(actor, critic, PPOSettings())
        with torch.no_grad():
            loss.value_estimator(batch)
        metrics = loss(batch.reshape(-1))
        objective = sum(
            metrics[k] for k in ("loss_objective", "loss_critic", "loss_entropy")
        )
        objective.backward()
        assert torch.isfinite(objective)
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in actor.parameters()
        )
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in critic.parameters()
        )
    finally:
        env.close()


@pytest.mark.parametrize("mode", ["discrete", "continuous"])
def test_fresh_training_checkpoint_reload_and_independent_evaluation(tmp_path, mode):
    cfg = SinglePushConfig(episode_seconds=0.4)
    path = train_ppo(
        cfg,
        tmp_path / mode,
        action_mode=mode,
        steps=8,
        num_envs=2,
        horizon=4,
        backend="cpu",
        max_seconds=None,
        validate_every=1,
        settings=PPOSettings(epochs=1, minibatch_size=8),
    )
    saved, config, actor, critic = load_ppo(path)
    initial, _, _, _ = load_ppo(path.parent / "initial.pt")
    assert not initial["optimizer"]["state"]
    assert saved["optimizer"]["state"]
    assert saved["training"]["steps"] == 8
    best, _, _, _ = load_ppo(path.parent / "best.pt")
    assert best["training"]["steps"] == 8 and best["training"]["rollouts"] == 1
    assert config == cfg
    assert any(
        not torch.equal(v, initial["actor"][k]) for k, v in actor.state_dict().items()
    )
    assert any(
        not torch.equal(v, initial["critic"][k]) for k, v in critic.state_dict().items()
    )
    result = evaluate_ppo(path, episodes=2, seed=140000, backend="cpu")
    assert result["episode_count"] == 2 and len(result["episodes"]) == 2
    assert result["goal_mode"] == "position"
    if mode == "discrete":
        tuned = train_ppo(
            replace(
                cfg,
                goal_distance_cost_weight=4.0,
                success_bonus=30.0,
                start_angle_degrees=180.0,
                spawn_mode="around",
                spawn_heading_mode="uniform",
                goal_position_mode="uniform",
                policy_rotation_mode="goal_octant",
            ),
            tmp_path / "fine_tuned",
            initial_checkpoint=path,
            steps=8,
            num_envs=2,
            horizon=4,
            backend="cpu",
            max_seconds=None,
            validate_every=1,
            settings=PPOSettings(epochs=1, minibatch_size=8),
        )
        tuned_initial, _, _, _ = load_ppo(tuned.parent / "initial.pt")
        assert not tuned_initial["optimizer"]["state"]
        for key, value in saved["actor"].items():
            torch.testing.assert_close(value, tuned_initial["actor"][key])
        assert tuned_initial["training"]["initial_checkpoint"]["training"]["steps"] == 8
        assert tuned_initial["config"]["start_angle_degrees"] == 180.0
        assert tuned_initial["config"]["spawn_mode"] == "around"
        assert tuned_initial["config"]["spawn_heading_mode"] == "uniform"
        assert tuned_initial["config"]["goal_position_mode"] == "uniform"
        assert tuned_initial["config"]["policy_rotation_mode"] == "goal_octant"
        around_validations = json.loads((tuned.parent / "validations.json").read_text())
        assert (
            sum(
                s["episode_count"]
                for s in around_validations[0]["report"]["angle_sectors"]
            )
            == 2
        )
        with pytest.raises(ValueError, match="same physics"):
            train_ppo(
                replace(cfg, cargo_radius=0.3),
                tmp_path / "incompatible",
                initial_checkpoint=path,
                steps=8,
                num_envs=2,
                horizon=4,
                backend="cpu",
                max_seconds=None,
            )
    with pytest.raises(FileExistsError):
        train_ppo(cfg, path.parent, steps=8, num_envs=2, horizon=4, backend="cpu")
    with pytest.raises(ValueError):
        replace(cfg, num_robots=2)


def test_direction_conversion_matches_on_cpu_gpu_and_uses_no_cargo_state():
    cfg = SinglePushConfig()
    direction = np.asarray(DIRECTIONS, dtype=np.float32)
    headings = np.linspace(-np.pi, np.pi, 9).astype(np.float32)
    commands = direction_commands(cfg, direction, headings, np)
    converted = direction_commands(
        cfg, torch.tensor(direction), torch.tensor(headings), torch
    )
    np.testing.assert_allclose(commands, converted.numpy(), atol=1e-6)
    np.testing.assert_array_equal(commands[0], [0, 0, 0])
    for i in range(1, 9):
        aligned = np.arctan2(direction[i, 1], direction[i, 0])
        result = direction_commands(cfg, direction[i : i + 1], np.array([aligned]), np)
        assert result[0, 0] == pytest.approx(cfg.direction_speed)
        assert result[0, 1] == pytest.approx(0, abs=1e-7)
        assert result[0, 2] == pytest.approx(0, abs=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("shape", ["cylinder"])
def test_gpu_position_hold_ignores_yaw_but_rejects_falls(shape):
    config = SinglePushConfig(shape=shape)
    env = SinglePushEnv(config, num_envs=2, backend="warp")
    try:
        worlds = env.worlds
        p = worlds.physics
        assert p.controller is not None and p.qpos.shape[-1] == 32
        assert worlds.reference.core.model.opt.timestep == 0.005
        q = worlds.cargo_q
        p.qpos[:, q : q + 2] = worlds.goal
        p.qpos[:, q + 3 : q + 7] = p.qpos.new_tensor([1.0, 0.0, 0.0, 0.0])
        p.qvel.zero_()
        p.forward()
        worlds.success_time.zero_()
        info = worlds.measurements()
        worlds.status(info)
        assert not info["success"].any()
        if shape == "T":
            assert (info["coverage"] < 0.95).all()
        else:
            torch.testing.assert_close(
                info["coverage"], torch.ones_like(info["coverage"])
            )
        worlds.status(info)
        assert info["success"].all()
        p.qpos[0, q + 3 : q + 7] = p.qpos.new_tensor([0.0, 1.0, 0.0, 0.0])
        p.forward()
        info = worlds.measurements()
        worlds.status(info)
        assert info["failed"][0] and not info["success"][0]
    finally:
        env.close()


def test_real_cylinder_geometry_and_rotationally_invariant_measurements():
    cfg = SinglePushConfig()
    assert cfg.shape == "cylinder" and cfg.width == cfg.depth == 0.5
    env = SinglePushGymEnv(cfg)
    try:
        env.reset(seed=42)
        core = env.core
        geom = core.model.geom("cargo_cylinder")
        assert geom.type[0] == mujoco.mjtGeom.mjGEOM_CYLINDER
        np.testing.assert_allclose(
            geom.size[:2], [cfg.cargo_radius, cfg.cargo_height / 2]
        )
        assert geom.friction[0] == pytest.approx(0.3)
        assert core.model.body("cargo").mass[0] == pytest.approx(2.0)
        marker = core.model.geom("goal_cargo_cylinder")
        assert marker.type[0] == mujoco.mjtGeom.mjGEOM_CYLINDER
        assert marker.contype[0] == marker.conaffinity[0] == 0
        assert [
            core.model.geom(i).name
            for i in range(core.model.ngeom)
            if core.model.geom(i).name.startswith("cargo_")
        ] == ["cargo_cylinder"]
        before = core.info()
        core.data.qpos[core.cargo_q + 3 : core.cargo_q + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(core.model, core.data)
        after = core.info()
        assert after["coverage"] == pytest.approx(before["coverage"])
        assert after["approach"] == pytest.approx(before["approach"])
        assert before["approach"] > 0.3  # Legs start clear of the cylinder.
        assert not np.any(core.data.xfrc_applied)
    finally:
        env.close()
    distances = np.array(
        [0.0, cfg.cargo_radius, 2 * cfg.cargo_radius, 3 * cfg.cargo_radius]
    )
    expected = [1.0, 2 / 3 - np.sqrt(3) / (2 * np.pi), 0.0, 0.0]
    np.testing.assert_allclose(
        circle_coverage(distances, cfg.cargo_radius, np), expected, atol=1e-12
    )
    torch.testing.assert_close(
        circle_coverage(torch.tensor(distances), cfg.cargo_radius, torch),
        torch.tensor(expected),
    )
    with pytest.raises(ValueError, match="position"):
        replace(cfg, goal_mode="coverage")
    with pytest.raises(ValueError):
        replace(cfg, cargo_radius=0.0)


def test_rear_approach_couples_side_and_surface_without_changing_success():
    cfg = SinglePushConfig(approach_strategy="rear_contact")
    positions = np.array([[0.8, 0.0], [0.0, 0.8], [-0.8, 0.0], [-0.5, 0.0]])
    goals = np.tile([0.4, 0.0], (4, 1))
    costs = cylinder_approach(cfg, positions, goals, np)
    assert costs[0] > costs[1] > costs[2] > costs[3]
    mirrored = positions * [1, -1]
    np.testing.assert_allclose(costs, cylinder_approach(cfg, mirrored, goals, np))
    np.testing.assert_allclose(
        costs,
        cylinder_approach(
            cfg, torch.tensor(positions), torch.tensor(goals), torch
        ).numpy(),
    )
    # At the goal the direction is undefined; the side cost smoothly vanishes.
    at_goal = cylinder_approach(cfg, positions, np.zeros_like(goals), np)
    np.testing.assert_allclose(
        at_goal, np.linalg.norm(positions, axis=-1) - cfg.cargo_radius
    )


@pytest.mark.parametrize("backend", ["cpu", "warp"])
def test_front_layout_rear_reward_and_graph_keep_physics_and_observations(backend):
    if backend == "warp" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    cfg = SinglePushConfig(
        start_angle_degrees=180.0,
        goal_distance=0.4,
        approach_strategy="rear_contact",
        episode_seconds=0.4,
    )
    env = SinglePushEnv(cfg, backend=backend, num_envs=2)
    try:
        check_env_specs(env)
        td = env.reset()
        relative = td["observation"][:, :2]
        goal = td["observation"][:, 2:4]
        assert ((relative * goal).sum(-1) > (goal * goal).sum(-1)).all()
        assert td["observation"].shape == (2, 14)
        if backend == "warp":
            from hexapod_transport_rl.warp_step_graph import capture_world_step

            capture_world_step(env.worlds)
        actor, critic = make_ppo_networks(device=env.device)
        with torch.no_grad():
            batch = env.rollout(4, actor, break_when_any_done=False)
        loss = make_ppo_loss(actor, critic, PPOSettings())
        with torch.no_grad():
            loss.value_estimator(batch)
        metrics = loss(batch.reshape(-1))
        assert torch.isfinite(metrics["loss_objective"])
        assert not batch["next", "success"].any()
    finally:
        env.close()


@pytest.mark.parametrize("backend", ["cpu", "warp"])
def test_reset_curriculum_changes_only_requested_lanes_and_preserves_success(backend):
    if backend == "warp" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    cfg = SinglePushConfig(
        start_angle_degrees=180.0,
        goal_distance=0.4,
        approach_strategy="rear_contact",
        approach_cost_weight=1.0,
    )
    env = SinglePushEnv(cfg, backend=backend, num_envs=2)
    try:
        env.set_seed(42)
        before = env.reset()
        env.set_reset_stage(SinglePushResetStage("rear", 1.0))
        # Scheduling a stage has no effect on active observations/physics.
        if backend == "warp":
            raw = env.worlds.observe()
            info = env.worlds.measurements()
            info["success"] = torch.zeros(2, dtype=torch.bool, device=env.device)
            measured = env._observation(raw, info)
        else:
            raw = np.stack([e.core.observe() for e in env.worlds.envs])
            infos = [e.core.info() for e in env.worlds.envs]
            measured = env._observation(
                raw,
                {
                    k: np.array([i[k] for i in infos])
                    for k in ("distance", "coverage", "approach", "success")
                },
            )
        torch.testing.assert_close(before["observation"], measured["observation"])
        reset = before.clone()
        reset["_reset"] = torch.tensor([[True], [False]], device=env.device)
        after = env.reset(reset)
        dot = (after["observation"][:, :2] * after["observation"][:, 2:4]).sum(-1)
        assert dot[0] < 0 < dot[1]
        torch.testing.assert_close(before["observation"][1], after["observation"][1])
        assert (
            env.config.position_tolerance == 0.05
            and env.config.success_hold_seconds == 0.4
        )
    finally:
        env.close()


def test_bootstrap_saves_final_front_task_and_training_stage(tmp_path):
    cfg = SinglePushConfig(
        start_angle_degrees=180.0,
        goal_distance=0.4,
        approach_strategy="rear_contact",
        approach_cost_weight=1.0,
        episode_seconds=0.4,
    )
    path = train_ppo(
        cfg,
        tmp_path / "bootstrap",
        num_envs=2,
        horizon=2,
        steps=16,
        max_seconds=None,
        backend="cpu",
        bootstrap_rollouts=1,
        settings=PPOSettings(epochs=1, minibatch_size=4),
        validate_every=2,
    )
    saved, restored, _, _ = load_ppo(path)
    assert restored.rear_start_fraction == 0 and restored.start_angle_degrees == 180
    assert saved["training"]["front_only_started_at_step"] == 4
    assert saved["training"]["training_stage"] == "front_only"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cylinder_collision_workspace_and_overflow_diagnostic():
    from mujoco_warp._src import collision_convex
    from mujoco_warp._src.types import OverflowType

    env = SinglePushEnv(num_envs=2, backend="warp")
    try:
        p = env.worlds.physics
        assert collision_convex.MJ_MAX_EPAHORIZON >= 64
        p.check()
        p.overflow_seen[0] = int(OverflowType.EPA_HORIZON)
        with pytest.raises(RuntimeError, match="EPA_HORIZON"):
            p.check()
        p.overflow_seen.zero_()
    finally:
        env.close()


def test_annulus_sampling_covers_full_circle_and_matches_numpy_torch():
    cfg = SinglePushConfig(spawn_mode="around")
    uniform = np.stack(
        (np.arange(360) / 360, np.linspace(0, 1, 360), np.full(360, 0.5)), -1
    )
    goal = np.broadcast_to([0.3, 0.4], (360, 2))
    xy, yaw = around_start_pose(cfg, goal, uniform, np)
    txy, tyaw = around_start_pose(cfg, torch.tensor(goal), torch.tensor(uniform), torch)
    np.testing.assert_allclose(xy, txy.numpy(), atol=1e-12)
    np.testing.assert_allclose(np.sin(yaw), tyaw.sin().numpy(), atol=1e-12)
    radius = np.linalg.norm(xy, axis=-1)
    assert radius.min() == pytest.approx(0.65) and radius.max() == pytest.approx(1.0)
    # The entire circle is covered, including both sides and the goal side.
    theta = np.arctan2(0.6 * xy[:, 1] - 0.8 * xy[:, 0], -(xy @ np.array([0.6, 0.8])))
    np.testing.assert_allclose(
        np.cos(theta), np.cos(2 * np.pi * uniform[:, 0]), atol=1e-12
    )
    np.testing.assert_allclose(
        np.sin(theta), np.sin(2 * np.pi * uniform[:, 0]), atol=1e-12
    )
    with pytest.raises(ValueError, match="clear"):
        replace(cfg, spawn_radius_min=0.5)
    with pytest.raises(ValueError, match="azimuth"):
        replace(cfg, spawn_azimuth_degrees=360)


@pytest.mark.parametrize("backend", ["cpu", "warp"])
def test_around_spawning_reproducible_partial_reset_and_angular_audit(backend):
    if backend == "warp" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    cfg = SinglePushConfig(
        spawn_mode="around", goal_distance=0.4, approach_strategy="rear_contact"
    )
    env = SinglePushEnv(cfg, num_envs=8, backend=backend)
    try:
        env.set_seed(77)
        before = env.reset()
        r = before["observation"][:, :2] * cfg.width
        assert ((r.norm(dim=-1) >= 0.65 - 1e-6) & (r.norm(dim=-1) <= 1.0 + 1e-6)).all()
        heading = before["observation"][:, 11:13]
        toward = -r / r.norm(dim=-1, keepdim=True)
        assert (
            heading[:, 1] * toward[:, 0] + heading[:, 0] * toward[:, 1]
            > np.cos(cfg.yaw_jitter) - 1e-6
        ).all()
        env.set_seed(77)
        torch.testing.assert_close(env.reset()["observation"], before["observation"])
        env.set_reset_stage(SinglePushAzimuthStage("left_side", 90.0))
        reset = before.clone()
        reset["_reset"] = torch.zeros(8, 1, dtype=torch.bool, device=env.device)
        reset["_reset"][0] = True
        after = env.reset(reset)
        torch.testing.assert_close(before["observation"][1:], after["observation"][1:])
        r, g = after["observation"][0, :2], after["observation"][0, 2:4]
        assert abs(float(r @ g)) < 1e-5
        assert float(g[0] * r[1] - g[1] * r[0]) > 0
        assert (
            env.config.position_tolerance == 0.05
            and env.config.success_hold_seconds == 0.4
        )
        env.check_physics()
    finally:
        env.close()


def test_independent_full_yaw_sampling_and_fixed_heading_offsets():
    cfg = SinglePushConfig(spawn_mode="around", spawn_heading_mode="uniform")
    azimuth, yaw = np.meshgrid(np.arange(8) / 8, np.arange(8) / 8)
    u = np.stack((azimuth.ravel(), np.full(64, 0.5), yaw.ravel()), -1)
    goal = np.broadcast_to([0.3, 0.4], (64, 2))
    xy, heading = around_start_pose(cfg, goal, u, np)
    txy, theading = around_start_pose(cfg, torch.tensor(goal), torch.tensor(u), torch)
    np.testing.assert_allclose(xy, txy.numpy(), atol=1e-12)
    np.testing.assert_allclose(heading, theading.numpy(), atol=1e-12)
    # Every position receives the same eight absolute yaw values.
    np.testing.assert_allclose(
        heading.reshape(8, 8),
        np.broadcast_to((np.arange(8) / 8 * 2 - 1)[:, None] * np.pi, (8, 8)),
    )
    np.testing.assert_allclose(xy.reshape(8, 8, 2), np.broadcast_to(xy[:8], (8, 8, 2)))
    away = replace(cfg, spawn_heading_offset_degrees=180)
    xy, heading = around_start_pose(away, goal, u, np)
    toward = -xy / np.linalg.norm(xy, axis=-1, keepdims=True)
    forward = np.stack((np.cos(heading), np.sin(heading)), -1)
    np.testing.assert_allclose((forward * toward).sum(-1), -1, atol=1e-12)
    with pytest.raises(ValueError, match="heading offset"):
        replace(cfg, spawn_heading_offset_degrees=360)


@pytest.mark.parametrize("backend", ["cpu", "warp"])
def test_random_heading_reset_and_pose_audit_preserve_other_worlds(backend):
    if backend == "warp" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    cfg = SinglePushConfig(
        spawn_mode="around", spawn_heading_mode="uniform", goal_distance=0.4
    )
    env = SinglePushEnv(cfg, num_envs=8, backend=backend)
    try:
        env.set_seed(171)
        before = env.reset()
        env.set_seed(171)
        torch.testing.assert_close(env.reset()["observation"], before["observation"])
        env.set_reset_stage(SinglePushPoseStage("away_from_cargo", 180.0, 180.0))
        reset = before.clone()
        reset["_reset"] = torch.zeros(8, 1, dtype=torch.bool, device=env.device)
        reset["_reset"][0] = True
        after = env.reset(reset)
        torch.testing.assert_close(after["observation"][1:], before["observation"][1:])
        r, g = after["observation"][0, :2], after["observation"][0, 2:4]
        assert float(r @ g) > 0
        heading = after["observation"][0, 11:13]
        forward = torch.stack((heading[1], heading[0]))
        assert float(forward @ (-r / r.norm())) == pytest.approx(-1, abs=1e-5)
        assert cfg.position_tolerance == 0.05 and cfg.success_hold_seconds == 0.4
        env.check_physics()
    finally:
        env.close()


def test_goal_annulus_formula_and_independent_position_heading_sampling():
    cfg = SinglePushConfig(
        spawn_mode="around", spawn_heading_mode="uniform", goal_position_mode="uniform"
    )
    angle, radius = np.meshgrid(np.arange(360) / 360, [0.0, 0.5, 1.0])
    u = np.stack((angle.ravel(), radius.ravel()), -1)
    goal = random_goal_delta(cfg, u, np)
    tgoal = random_goal_delta(cfg, torch.tensor(u), torch)
    np.testing.assert_allclose(goal, tgoal.numpy(), atol=1e-12)
    radii = np.linalg.norm(goal, axis=-1)
    np.testing.assert_allclose(radii**2, 0.3**2 + u[:, 1] * (0.6**2 - 0.3**2))
    # Changing the goal cannot select the robot's radius, relative angle or yaw.
    spawn_u = np.broadcast_to([0.125, 0.4, 0.7], (len(u), 3))
    robot, yaw = around_start_pose(cfg, goal, spawn_u, np)
    np.testing.assert_allclose(yaw, 0.4 * np.pi)
    np.testing.assert_allclose(
        np.linalg.norm(robot, axis=-1) ** 2, 0.65**2 + 0.4 * (1 - 0.65**2)
    )
    unit = goal / radii[:, None]
    np.testing.assert_allclose(
        (robot * unit).sum(-1) / np.linalg.norm(robot, axis=-1), -np.cos(np.pi / 4)
    )
    fixed = replace(cfg, goal_azimuth_degrees=270)
    fixed_goal = random_goal_delta(fixed, u, np)
    np.testing.assert_allclose(fixed_goal[:, 0], 0, atol=1e-12)
    assert (fixed_goal[:, 1] < 0).all()
    for changes in (
        {"goal_radius_min": 0.0},
        {"goal_radius_max": 0.2},
        {"goal_azimuth_degrees": 360},
        {"goal_position_mode": "other"},
    ):
        with pytest.raises(ValueError):
            replace(cfg, **changes)


def test_goal_octant_policy_rotates_world_logits_and_keeps_critic_invariant():
    torch.manual_seed(45)
    actor = GoalOctantMLP(torch.nn.Linear(14, 9), actor=True)
    critic = GoalOctantMLP(torch.nn.Linear(14, 1))
    obs = torch.randn(7, 3, 14, dtype=torch.float64)
    # Stay clear of half-octant boundaries in this group-law check.
    theta = torch.linspace(0.13, 0.17, 21).reshape(7, 3)
    obs[..., 2], obs[..., 3] = theta.cos(), theta.sin()
    obs[..., 11], obs[..., 12] = theta.sin(), theta.cos()
    actor.double()
    critic.double()
    base, value = actor(obs), critic(obs)
    for k in range(8):
        angle = k * torch.pi / 4
        c, s = np.cos(angle), np.sin(angle)
        rotated = obs.clone()
        for start in (0, 2, 6, 8):
            x, y = obs[..., start], obs[..., start + 1]
            rotated[..., start] = c * x - s * y
            rotated[..., start + 1] = s * x + c * y
        sine, cosine = obs[..., 11], obs[..., 12]
        rotated[..., 11] = sine * c + cosine * s
        rotated[..., 12] = cosine * c - sine * s
        expected = torch.cat((base[..., :1], base[..., 1:].roll(k, dims=-1)), -1)
        torch.testing.assert_close(actor(rotated), expected)
        torch.testing.assert_close(critic(rotated), value)
    # Zero logits still permit all nine actions, including a heading-independent stop.
    assert actor(obs).shape == (7, 3, 9)
    with pytest.raises(ValueError, match="discrete"):
        make_ppo_networks("continuous", rotation_mode="goal_octant")


@pytest.mark.parametrize("backend", ["cpu", "warp"])
def test_random_goals_reset_only_requested_world_and_preserve_success(backend):
    if backend == "warp" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    cfg = SinglePushConfig(
        spawn_mode="around", spawn_heading_mode="uniform", goal_position_mode="uniform"
    )
    env = SinglePushEnv(cfg, num_envs=32, backend=backend)
    try:
        env.set_seed(451)
        before = env.reset()
        goal = before["observation"][:, 2:4] * cfg.width
        assert (
            (goal.norm(dim=-1) >= 0.3 - 1e-6) & (goal.norm(dim=-1) <= 0.6 + 1e-6)
        ).all()
        assert (
            (goal[:, 0] > 0).any()
            and (goal[:, 0] < 0).any()
            and (goal[:, 1] > 0).any()
            and (goal[:, 1] < 0).any()
        )
        env.set_seed(451)
        torch.testing.assert_close(env.reset()["observation"], before["observation"])
        env.set_reset_stage(SinglePushGoalStage("goal_south", 270.0))
        reset = before.clone()
        reset["_reset"] = torch.zeros(32, 1, dtype=torch.bool, device=env.device)
        reset["_reset"][0] = True
        after = env.reset(reset)
        torch.testing.assert_close(after["observation"][1:], before["observation"][1:])
        g = after["observation"][0, 2:4] * cfg.width
        assert abs(float(g[0])) < 1e-6 and float(g[1]) < 0
        if backend == "warp":
            w = env.worlds
            w.physics.qpos[0, w.cargo_q : w.cargo_q + 2] = w.goal[0]
            w.physics.qvel[0].zero_()
            w.physics.forward()
            info = w.measurements()
            w.status(info)
            assert not info["success"][0]
            w.status(info)
            assert info["success"][0]
        else:
            core = env.worlds.envs[0].core
            np.testing.assert_allclose(core.model.body("goal").pos[:2], core.goal)
            core.data.qpos[core.cargo_q : core.cargo_q + 2] = core.goal
            core.data.qvel[:] = 0
            mujoco.mj_forward(core.model, core.data)
            info = core.info()
            core._update_episode_status(info)
            assert not info["success"]
            core._update_episode_status(info)
            assert info["success"]
        env.check_physics()
    finally:
        env.close()
