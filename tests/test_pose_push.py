"""Equal-arm geometry, actual pose rewards, learning boundaries and deployment."""

from dataclasses import asdict, replace

import numpy as np
import pytest
import torch
from gymnasium.utils.env_checker import check_env
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, ReplayBuffer

from hexapod_transport_rl import (
    POSE_STAGES,
    MAPPOSettings,
    PoseCurriculum,
    PosePushConfig,
    PosePushEnv,
    PoseRewardWeights,
    PoseStage,
    PushConfig,
    TorchRLTransportEnv,
    anneal_exploration,
    load_mappo,
    make_mappo_loss,
    make_mappo_networks,
    policy_action,
    save_mappo,
)
from hexapod_transport_rl.config import find_asset_root
from hexapod_transport_rl.model import build_model
from hexapod_transport_rl.pose_evaluation import evaluate_pose


@pytest.mark.parametrize("geometry", ["equal_bars", "equal_arms"])
@pytest.mark.parametrize("num_robots", [2, 3, 4])
def test_t_nonoverlap_uniform_density_and_matching_marker(geometry, num_robots):
    cfg = PushConfig(shape="T", t_geometry=geometry, num_robots=num_robots)
    model = build_model(find_asset_root(), cfg)
    bar, stem = model.geom("cargo_bar"), model.geom("cargo_stem")
    junction = bar.pos[:2]
    endpoints = np.array(
        [
            [junction[0], -bar.size[1]],
            [junction[0], bar.size[1]],
            [stem.pos[0] + stem.size[0], 0],
        ]
    )
    np.testing.assert_allclose(
        np.linalg.norm(endpoints - junction, axis=1),
        [cfg.t_arm_length, cfg.t_arm_length, cfg.t_stem_length],
    )
    if geometry == "equal_bars":
        assert 2 * bar.size[1] == pytest.approx(stem.pos[0] + stem.size[0])
    assert stem.pos[0] - stem.size[0] == pytest.approx(bar.pos[0] + bar.size[0])
    assert model.body("cargo").mass[0] == pytest.approx(cfg.cargo_mass)
    bar_area = 4 * bar.size[0] * bar.size[1]
    stem_area = 4 * stem.size[0] * stem.size[1]
    expected_com = (bar_area * bar.pos + stem_area * stem.pos) / (bar_area + stem_area)
    np.testing.assert_allclose(model.body("cargo").ipos, expected_com)
    for name in ("cargo_bar", "cargo_stem"):
        actual, target = model.geom(name), model.geom("goal_" + name)
        np.testing.assert_array_equal(actual.size, target.size)
        np.testing.assert_array_equal(actual.pos, target.pos)
        assert target.contype[0] == target.conaffinity[0] == 0
    assert cfg.rear_face == pytest.approx(-0.1)
    assert PushConfig.from_checkpoint({"shape": "T"}).t_geometry == "legacy"
    assert PushConfig.from_checkpoint(asdict(cfg)) == cfg


@pytest.mark.parametrize("num_robots", [2, 3, 4])
def test_pose_starts_beyond_goal_facing_cargo_without_initial_contact(num_robots):
    env = PosePushEnv(PosePushConfig(num_robots=num_robots))
    try:
        for seed in range(20):
            env.reset(seed=seed)
            core = env.core
            direction = core.goal / np.linalg.norm(core.goal)
            projection = (core.robot_xy - core.cargo_xy) @ direction
            assert np.all(projection > np.linalg.norm(core.goal) + 0.5)
            assert np.all(np.abs(projection - core.cfg.start_distance) < 0.06)
            headings = core.yaw(core.base_ids)
            facing = np.stack((np.cos(headings), np.sin(headings)), -1)
            assert np.all(facing @ direction < -0.99)
            owners = core.contacts.geom_owner[core.data.contact.geom]
            assert not np.any(
                ((owners[:, 0] == -2) & (owners[:, 1] >= 0))
                | ((owners[:, 1] == -2) & (owners[:, 0] >= 0))
            )
    finally:
        env.close()


def test_old_pose_checkpoint_retains_geometry_start_and_horizon():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "checkpoints/pose_transport.pt"
    _, _, saved = load_mappo(path)
    cfg = PosePushConfig.from_checkpoint(saved["pose_config"])
    assert cfg.t_geometry == "equal_arms"
    assert cfg.robot_start == "cargo_rear"
    assert cfg.approach_metric == "euclidean"
    assert cfg.approach_clearance == 0.16
    assert cfg.reward_weights.approach_error == cfg.reward_weights.push_heading == 0
    assert cfg.reward_weights.orientation_error == 0
    assert cfg.action_frame == "body"
    assert cfg.episode_seconds == 12
    env = PosePushEnv(cfg)
    try:
        env.reset(options={"randomize": False})
        np.testing.assert_allclose(
            env.core.robot_xy[:, 0], cfg.rear_face - cfg.push_gap - 0.1
        )
    finally:
        env.close()


def test_pose_gym_api_seed_and_reflected_actions():
    torch.set_num_threads(1)
    env = PosePushEnv(PosePushConfig(action_frame="body"))
    try:
        check_env(env, skip_render_check=True)
        obs, info = env.reset(seed=51)
        assert obs.shape == (2, 22)
        qpos = env.core.data.qpos.copy()
        repeated, _ = env.reset(seed=51)
        np.testing.assert_array_equal(obs, repeated)
        np.testing.assert_array_equal(qpos, env.core.data.qpos)
        angle = np.rad2deg(info["yaw_error"])
        assert 5 <= angle <= 30
        action = np.array([[0.2, 0.3, 0.4], [0.2, 0.3, 0.4]], np.float32)
        _, reward, _, _, after = env.step(action)
        assert reward == pytest.approx(sum(after["reward_terms"].values()))
        np.testing.assert_allclose(
            after["commands"][0], [0.04, -0.03, -0.24], atol=1e-7
        )
        np.testing.assert_allclose(after["commands"][1], [0.04, 0.03, 0.24], atol=1e-7)
        assert after["footprint_error_m"] > after["distance"]
    finally:
        env.close()


def test_pose_success_requires_angle_low_motion_and_one_second_hold():
    env = PosePushEnv()
    try:
        env.reset(seed=3)
        core = env.core
        core.goal = core.cargo_xy.copy()
        core.goal_yaw = core.cargo_yaw + 0.1
        for _ in range(6):
            assert not core._update_episode_status(core.info())[0]
        core.goal_yaw = core.cargo_yaw
        core.data.qvel[core.cargo_v] = 0.07
        assert not core._update_episode_status(core.info())[0]
        core.data.qvel[core.cargo_v] = 0
        for _ in range(4):
            assert not core._update_episode_status(core.info())[0]
        assert core._update_episode_status(core.info())[0]
    finally:
        env.close()


def test_pose_reward_changes_without_changing_physics():
    config = PosePushConfig()
    first = PosePushEnv(config)
    second = PosePushEnv(
        replace(config, reward_weights=replace(config.reward_weights, orientation=0))
    )
    try:
        first.reset(seed=12)
        second.reset(seed=12)
        actions = np.array([[0.5, 0, 0.1], [0.8, 0, -0.1]], np.float32)
        _, reward_a, _, _, a = first.step(actions)
        _, reward_b, _, _, b = second.step(actions)
        np.testing.assert_array_equal(first.core.data.qpos, second.core.data.qpos)
        assert reward_a - reward_b == pytest.approx(a["reward_terms"]["orientation"])
        assert b["reward_terms"]["orientation"] == 0
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("heading", [0, np.pi / 2, np.pi, -np.pi / 2])
def test_cargo_frame_actions_are_rotated_into_walking_commands(heading):
    import mujoco

    from hexapod_transport_rl.config import rotation
    from hexapod_transport_rl.pose_push import POSE_COMMAND_LIMIT

    env = PosePushEnv()
    try:
        env.reset(options={"randomize": False})
        for q in env.core.root_q:
            env.core.data.qpos[q + 3 : q + 7] = [
                np.cos(heading / 2),
                0,
                0,
                np.sin(heading / 2),
            ]
        mujoco.mj_forward(env.core.model, env.core.data)
        action = np.tile([0.2, 0.3, 0.4], (2, 1)).astype(np.float32)
        _, _, _, _, info = env.step(action)
        requested = action * POSE_COMMAND_LIMIT
        requested[0, 1:] *= -1
        commands = info["commands"].copy()
        commands[:, :2] = commands[:, :2] @ rotation(heading).T
        np.testing.assert_allclose(commands, requested, atol=1e-7)
    finally:
        env.close()


def test_approach_cost_and_push_heading_are_state_rewards():
    from hexapod_transport_rl import PoseRewardWeights

    config = PosePushConfig(
        approach_metric="collision_free",
        robot_start="cargo_rear",
        reward_weights=PoseRewardWeights(
            orientation_error=2, approach_error=2, push_heading=3
        ),
    )
    env = PosePushEnv(config)
    try:
        env.reset(options={"randomize": False})
        core = env.core
        assert core.info()["push_heading_error"] == pytest.approx(0)
        for q in core.root_q:
            core.data.qpos[q + 3 : q + 7] = [0, 0, 0, 1]
        import mujoco

        mujoco.mj_forward(core.model, core.data)
        assert core.info()["push_heading_error"] == pytest.approx(1)
        _, _, _, _, info = env.step(np.zeros((2, 3), dtype=np.float32))
        terms = info["reward_terms"]
        assert terms["orientation_error"] == pytest.approx(
            -config.reward_weights.orientation_error
            * config.dt
            * info["yaw_error"]
            / 0.5
        )
        assert terms["approach_error"] == pytest.approx(
            -2 * config.dt * core._mean_approach_distance()
        )
        assert terms["push_heading"] == pytest.approx(
            -3 * config.dt * info["push_heading_error"]
        )
    finally:
        env.close()


def test_forward_baseline_keeps_body_direction_at_a_cargo_frame_limit():
    import mujoco

    from hexapod_transport_rl.pose_evaluation import rule_action

    env = PosePushEnv(
        PosePushConfig(robot_start="cargo_rear", cargo_command_limits=(0.2, 0.1, 0.6))
    )
    try:
        env.reset(options={"randomize": False})
        for q in env.core.root_q:
            env.core.data.qpos[q + 3 : q + 7] = [2**-0.5, 0, 0, 2**-0.5]
        mujoco.mj_forward(env.core.model, env.core.data)
        action = rule_action(env.core, "forward")
        assert env.action_space.contains(action)
        env.step(action)
        np.testing.assert_allclose(env.core.last_commands, [[0.1, 0, 0]] * 2, atol=1e-7)
    finally:
        env.close()


def test_smooth_rendering_keeps_physics_and_records_actual_control_ticks():
    plain = PosePushEnv()
    recorded = PosePushEnv(render_mode="rgb_array_list")
    times = []

    def capture():
        times.append(recorded.core.data.time)
        return np.zeros((2, 2, 3), dtype=np.uint8)

    recorded._render_rgb = capture
    try:
        plain.reset(seed=44)
        recorded.reset(seed=44)
        assert len(recorded.render()) == 1
        action = np.full((2, 3), 0.2, np.float32)
        a, b = plain.step(action), recorded.step(action)
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1:4] == b[1:4]
        np.testing.assert_array_equal(plain.core.data.qpos, recorded.core.data.qpos)
        assert len(recorded.render()) == 5
        assert recorded.render() == []
        assert recorded.metadata["render_fps"] == 25
        np.testing.assert_allclose(times, np.arange(6) * 0.04, atol=1e-12)
    finally:
        plain.close()
        recorded.close()


def test_pose_stage_changes_only_next_reset():
    env = PosePushEnv()
    try:
        env.reset(seed=1)
        oldcfg = env.config
        positions = env.core.data.qpos.copy()
        env.set_stage(POSE_STAGES[0])
        assert env.config == oldcfg
        np.testing.assert_array_equal(positions, env.core.data.qpos)
        env.reset(seed=1)
        assert env.config.position_tolerance == 0.08
        assert env.config.goal_distance == 0.3
    finally:
        env.close()


def test_reset_curriculum_mixes_start_states_without_changing_final_task():
    final = PosePushConfig()
    mixed = PoseStage("mixed", 0.3, 15, 0.08, np.deg2rad(8), 0.6, 0.5)
    env = PosePushEnv(final)
    try:
        env.set_stage(mixed)
        starts = []
        for seed in range(20):
            env.reset(seed=seed)
            core = env.core
            local = (core.robot_xy - core.cargo_xy) @ np.array(
                [
                    [np.cos(core.cargo_yaw), -np.sin(core.cargo_yaw)],
                    [np.sin(core.cargo_yaw), np.cos(core.cargo_yaw)],
                ]
            )
            starts.append(bool((local[:, 0] < 0).all()))
        assert 5 <= sum(starts) <= 15
        env.set_stage(POSE_STAGES[-1])
        env.reset(seed=4)
        assert env.config.rear_start_fraction == 0
        direction = env.core.goal / np.linalg.norm(env.core.goal)
        assert (env.core.robot_xy @ direction > 1.7).all()
    finally:
        env.close()


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is unavailable",
            ),
        ),
    ],
)
def test_real_pose_update_save_relocation_and_four_robot_loading(tmp_path, device):
    torch.set_num_threads(1)
    config = PosePushConfig(num_robots=4, episode_seconds=0.2)
    env = TorchRLTransportEnv(config, num_envs=2, asynchronous=False)
    env.set_seed(21)
    settings = MAPPOSettings(
        epochs=1,
        minibatch_size=4,
        value_normalization=True,
        initial_log_std=-1.5,
        final_log_std=-3.0,
    )
    actor, critic = make_mappo_networks(
        env.num_robots, env.obs_dim, initial_log_std=settings.initial_log_std
    )
    actor.to(device)
    critic.to(device)
    loss = make_mappo_loss(actor, critic, settings)
    optimizer = torch.optim.Adam(loss.parameters(), lr=settings.learning_rate)
    curriculum = PoseCurriculum(
        config,
        env,
        actor,
        tmp_path / "log",
        seed=21,
        settings=settings,
        horizon=2,
        validate_every=10,
    )
    collector = Collector(
        env,
        actor,
        frames_per_batch=4,
        total_frames=4,
        env_device="cpu",
        policy_device="cpu",
        storing_device="cpu",
        auto_register_policy_transforms=True,
    )
    try:
        batch = next(iter(collector))
        assert batch.device == torch.device("cpu")
        assert next(collector.policy.parameters()).device.type == "cpu"
        assert next(actor.parameters()).device.type == device
        batch = batch.to(device)
        loss.value_estimator(batch)
        buffer = ReplayBuffer(storage=LazyTensorStorage(4, device=device), batch_size=4)
        buffer.extend(batch.reshape(-1))
        previous = [p.detach().clone() for p in actor.parameters()]
        metrics = loss(buffer.sample())
        objective = sum(
            metrics[k] for k in ("loss_objective", "loss_critic", "loss_entropy")
        )
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()
        assert next(actor.parameters()).grad.device.type == device
        assert any(
            not torch.equal(a, b)
            for a, b in zip(previous, actor.parameters(), strict=True)
        )
        curriculum.record(batch, metrics)
        obs = batch["agents", "observation"][0, 0].cpu().numpy()
        before_annealing = policy_action(actor, obs)
        anneal_exploration(actor, settings, 0.5)
        np.testing.assert_array_equal(before_annealing, policy_action(actor, obs))
        scales = [m.log_std for m in actor.modules() if hasattr(m, "log_std")]
        assert len(scales) == 1 and scales[0].max() <= -2.25
        anneal_exploration(actor, settings, 0, restart=True)
        assert torch.all(scales[0] == settings.initial_log_std)
        np.testing.assert_array_equal(before_annealing, policy_action(actor, obs))
        collector.update_policy_weights_()
        np.testing.assert_allclose(
            policy_action(collector.policy, obs),
            policy_action(actor, obs),
            rtol=1e-5,
            atol=1e-6,
        )
        rng = torch.get_rng_state().clone()
        path = save_mappo(
            tmp_path / "pose.pt",
            actor,
            critic,
            optimizer,
            config=config,
            provenance=env.provenance,
            training=curriculum.training,
            curriculum=curriculum.state,
            loss=loss,
        )
        torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
        expected = policy_action(actor, obs)
        relocated = tmp_path / "relocated.pt"
        path.rename(relocated)
        loaded, loaded_critic, saved = load_mappo(relocated)
        np.testing.assert_allclose(
            expected, policy_action(loaded, obs), rtol=1e-5, atol=1e-6
        )
        assert saved["training"]["learning_device"].startswith(device)
        assert saved["training"]["physics_device"] == "cpu"
        if device == "cuda":
            torch.testing.assert_close(
                saved["cuda_rng_state"], torch.cuda.get_rng_state()
            )
            reloaded_gpu, _, _ = load_mappo(relocated, device="cuda")
            np.testing.assert_array_equal(expected, policy_action(reloaded_gpu, obs))
        assert saved["optimizer"]["state"] and "pushing_checkpoint" not in saved
        assert saved["num_robots"] == 4 and saved["obs_dim"] == 30
        restored_loss = make_mappo_loss(
            loaded,
            loaded_critic,
            settings,
            value_normalizer_state=saved["value_normalizer"],
        )
        for key, value in loss.value_norm.state_dict().items():
            torch.testing.assert_close(
                restored_loss.value_norm.state_dict()[key], value.cpu(), rtol=0, atol=0
            )
    finally:
        curriculum.close()
        collector.shutdown()


def test_rule_evaluation_uses_same_final_accuracy_and_reset_seeds(tmp_path):
    config = PosePushConfig(episode_seconds=0.2)
    reports = [
        evaluate_pose(
            config=config,
            policy=policy,
            episodes=2,
            seed=81000,
            workers=1,
            output=tmp_path / f"{policy}.json",
        )
        for policy in ("forward", "feedback")
    ]
    assert reports[0]["config"] == reports[1]["config"] == asdict(config)
    for a, b in zip(reports[0]["episodes"], reports[1]["episodes"], strict=True):
        assert a["seed"] == b["seed"]
        assert a["initial_distance"] == b["initial_distance"]
        assert a["initial_yaw_error"] == b["initial_yaw_error"]
    assert all(not report["demonstration_actions_used"] for report in reports)
    for report in reports:
        for row in report["episodes"]:
            assert row["position_error_integral_m_s"] == pytest.approx(
                config.dt * row["distance"]
            )
            assert row["yaw_error_integral_rad_s"] == pytest.approx(
                config.dt * row["yaw_error"]
            )
            assert row["time_to_success_or_limit_s"] == config.episode_seconds


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_parallel_pose_evaluation_matches_native_resets_and_stops_each_lane(tmp_path):
    """Some lanes succeed early; others time out. Neither may accrue extra costs."""
    torch.set_num_threads(1)
    config = PosePushConfig(
        episode_seconds=0.8,
        goal_distance=0.04,
        goal_lateral_range=0.16,
        min_yaw_degrees=0,
        max_yaw_degrees=0,
        success_hold_seconds=0.4,
    )
    env = PosePushEnv(config)
    actor, critic = make_mappo_networks(2, 22, action_grid=(-1, 0, 1))
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.zero_()
        for module in actor.modules():
            if isinstance(module, torch.nn.Linear) and module.out_features == 9:
                module.bias.reshape(3, 3)[:, 1] = 10
    optimizer = torch.optim.Adam([*actor.parameters(), *critic.parameters()])
    path = save_mappo(
        tmp_path / "stop.pt",
        actor,
        critic,
        optimizer,
        config=config,
        provenance=env.provenance,
        training={},
        curriculum={},
    )
    env.close()
    native = evaluate_pose(path, config=config, seed=95000, episodes=16, workers=1)
    gpu = evaluate_pose(path, config=config, seed=95000, episodes=16, backend="warp")
    assert 0 < native["successes"] < 16
    for a, b in zip(native["episodes"], gpu["episodes"], strict=True):
        assert a["seed"] == b["seed"]
        assert a["initial_distance"] == b["initial_distance"]
        assert a["success"] == b["success"]
        assert b["elapsed_seconds"] == pytest.approx(a["elapsed_seconds"], abs=1e-6)
        assert b["position_error_integral_m_s"] == pytest.approx(
            a["position_error_integral_m_s"], abs=1e-4
        )
        assert b["footprint_error_m"] == pytest.approx(a["footprint_error_m"], abs=1e-4)
        assert b["time_to_success_or_limit_s"] == pytest.approx(
            b["elapsed_seconds"] if b["success"] else config.episode_seconds
        )


def test_fixed_task_reward_adaptation_records_common_initializer(tmp_path):
    """A reward change starts from the common weights, with a new optimizer."""
    import hashlib
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "checkpoints/goal_side_pose.pt"
    actor, critic, saved = load_mappo(path)
    config = replace(
        PosePushConfig.from_checkpoint(saved["pose_config"]),
        reward_weights=replace(PoseRewardWeights(), time=0.5),
    )
    settings = MAPPOSettings(gamma=config.shaping_discount, value_normalization=True)
    env = TorchRLTransportEnv(config, num_envs=2, asynchronous=False)
    loss = make_mappo_loss(
        actor,
        critic,
        settings,
        value_normalizer_state=saved["value_normalizer"],
    )
    optimizer = torch.optim.Adam(loss.parameters(), lr=settings.learning_rate)
    curriculum = PoseCurriculum(
        config,
        env,
        actor,
        tmp_path / "adaptation",
        seed=20,
        settings=settings,
        horizon=2,
        stages=(POSE_STAGES[-1],),
        validate_every=None,
        initial_checkpoint=path,
    )
    collector = Collector(
        env,
        actor,
        frames_per_batch=4,
        total_frames=4,
        auto_register_policy_transforms=True,
    )
    try:
        assert not optimizer.state
        for key, value in saved["actor"].items():
            torch.testing.assert_close(actor.state_dict()[key], value)
        batch = next(iter(collector))
        loss.value_estimator(batch)
        metrics = loss(batch.reshape(-1))
        optimizer.zero_grad()
        (
            metrics["loss_objective"] + metrics["loss_critic"] + metrics["loss_entropy"]
        ).backward()
        optimizer.step()
        curriculum.record(batch, metrics)
        assert optimizer.state and curriculum.validation_env is None
        assert curriculum.state["stage"] == "settle"
        assert curriculum.training["validate_every"] is None
        initial = curriculum.training["initialization"]
        assert initial["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert initial["parent_team_steps"] == 3145728
        assert initial["optimizer_inherited"] is False
        assert initial["demonstration_actions_used"] is False
    finally:
        curriculum.close()
        collector.shutdown()
