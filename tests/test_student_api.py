"""Student terms, original physics parity, real PPO, and portable task sources."""

import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace

import mujoco
import pytest
import torch
from torchrl.envs.utils import check_env_specs

from hexapod_transport_rl.single_push import SinglePushConfig, SinglePushEnv
from hexapod_transport_rl.student import (
    ContinuousVelocity,
    CylinderPushTask,
    DiscreteDirections,
    ObservationTerm,
    RewardTerm,
    compare,
    load_task,
    load_task_file,
    train,
)
from hexapod_transport_rl.torchrl_ppo import PPOSettings, load_ppo, train_ppo

SETTINGS = PPOSettings(epochs=1, minibatch_size=4)


def scalar_observation(state):
    return state.robot_to_cargo.norm(dim=-1)


def progress_reward(before, after):
    return before.goal_distance - after.goal_distance


def wrong_observation(state):
    return state.robot_position[:, None, :]


def nan_observation(state):
    return state.goal_distance * float("nan")


def quick_task(**kwargs):
    task = CylinderPushTask(**kwargs)
    return replace(task, config=replace(task.config, episode_seconds=0.4))


@pytest.mark.parametrize("backend", ["cpu", "warp"])
def test_baseline_adapter_matches_original_physics_inputs_and_reward(backend):
    if backend == "warp" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    task = CylinderPushTask.baseline()
    reference = SinglePushEnv(task.config, num_envs=2, backend=backend)
    env = task.make_env(num_envs=2, backend=backend)
    try:
        reference.set_seed(712)
        env.set_seed(712)
        before = reference.reset()
        td = env.reset()
        torch.testing.assert_close(td["observation"], before["observation"])
        if backend == "warp":
            from hexapod_transport_rl.warp_step_graph import capture_world_step

            capture_world_step(env.worlds)
            env.set_seed(712)
            td = env.reset()
        for action in (0, 2, 5, 8):
            td["action"] = torch.full((2,), action, device=env.device)
            before["action"] = td["action"].clone()
            td = env.step(td)["next"]
            before = reference.step(before)["next"]
            torch.testing.assert_close(
                td["evaluation_state"], before["observation"], atol=3e-5, rtol=3e-5
            )
            torch.testing.assert_close(
                td["reward"], before["reward"], atol=3e-5, rtol=3e-5
            )
            torch.testing.assert_close(td["reward"], sum(td["reward_terms"].values()))
            assert not td["done"].any()
        env.check_physics()
    finally:
        env.close()
        assert env.is_closed
        if backend == "cpu":
            assert env.worlds.closed and all(e._closed for e in env.worlds.envs)
        reference.close()


@pytest.mark.parametrize("backend", ["cpu", "warp"])
@pytest.mark.parametrize(
    "actions", [DiscreteDirections(4), ContinuousVelocity(frame="body")]
)
def test_custom_features_action_specs_and_partial_reset(backend, actions):
    if backend == "warp" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    task = CylinderPushTask(
        observations=("cargo_to_goal", ObservationTerm(scalar_observation, scale=0.5)),
        actions=actions,
    )
    env = task.make_env(num_envs=3, backend=backend)
    try:
        check_env_specs(env)
        env.set_seed(777)
        td = env.reset()
        assert td["observation"].shape == (3, 3)
        torch.testing.assert_close(td["observation"][:, :2], env.state.cargo_to_goal)
        torch.testing.assert_close(
            td["observation"][:, 2], env.state.robot_to_cargo.norm(dim=-1) / 0.5
        )
        action = (
            torch.ones(3, dtype=torch.long, device=env.device)
            if task.action_mode == "discrete"
            else torch.tensor([[1.0, 0.0]] * 3, device=env.device)
        )
        previous_yaw = env.state.robot_yaw.clone()
        td["action"] = action
        td = env.step(td)["next"]
        expected_command = (
            torch.tensor([[1.0, 0.0]] * 3, device=env.device) * actions.speed
        )
        if actions.frame == "body":
            expected_command = (
                torch.stack((previous_yaw.cos(), previous_yaw.sin()), -1)
                * actions.speed
            )
        torch.testing.assert_close(env.state.command_velocity, expected_command)
        old_distance = env.state.goal_distance.clone()
        reset = td.clone()
        reset["_reset"] = torch.tensor([[True], [False], [False]], device=env.device)
        after = env.reset(reset)
        torch.testing.assert_close(after["observation"][1:], td["observation"][1:])
        torch.testing.assert_close(env.state.goal_distance[1:], old_distance[1:])
        torch.testing.assert_close(env.state.command_velocity[1:], expected_command[1:])
        assert not env.state.command_velocity[0].any()
        torch.testing.assert_close(
            env.state.goal_distance, env.state.cargo_to_goal.norm(dim=-1)
        )
    finally:
        env.close()


@pytest.mark.parametrize("backend", ["cpu", "warp"])
def test_success_is_independent_of_reward_and_policy_observations(backend):
    if backend == "warp" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    task = CylinderPushTask(observations=("robot_yaw_rate",), rewards={"time": 0.0})
    task = replace(task, config=replace(task.config, spawn_azimuth_degrees=0.0))
    env = task.make_env(num_envs=2, backend=backend)
    try:
        td = env.reset()
        if backend == "warp":
            w = env.worlds
            w.physics.qpos[:, w.cargo_q : w.cargo_q + 2] = w.goal
            w.physics.qvel.zero_()
            w.physics.forward()
        else:
            for native in env.worlds.envs:
                c = native.core
                c.data.qpos[c.cargo_q : c.cargo_q + 2] = c.goal
                c.data.qvel[:] = 0
                mujoco.mj_forward(c.model, c.data)
        for step in range(2):
            td["action"] = torch.zeros(2, dtype=torch.long, device=env.device)
            td = env.step(td)["next"]
            assert bool(td["success"].all()) == (step == 1)
            assert not td["reward"].any()
    finally:
        env.close()


@pytest.mark.parametrize(
    "actions",
    [
        DiscreteDirections(4),
        DiscreteDirections(16, include_stop=False),
        ContinuousVelocity(),
    ],
)
def test_actual_ppo_updates_dynamic_networks_saves_and_evaluates(tmp_path, actions):
    task = quick_task(observations=("robot_yaw_rate",), actions=actions)
    run = train(
        task,
        tmp_path / str(actions.action_count),
        seed=21,
        steps=8,
        num_envs=2,
        horizon=2,
        backend="cpu",
        validation_envs=2,
        validate_every=20,
        settings=SETTINGS,
    )
    saved, _, actor, critic = load_ppo(run.latest)
    initial, _, _, _ = load_ppo(run.directory / "initial.pt")
    assert saved["network"]["observation_dim"] == 1
    assert saved["network"]["action_count"] == (actions.action_count or 9)
    assert saved["optimizer"]["state"] and not initial["optimizer"]["state"]
    assert any(
        not torch.equal(v, initial["actor"][k]) for k, v in actor.state_dict().items()
    )
    assert any(
        not torch.equal(v, initial["critic"][k]) for k, v in critic.state_dict().items()
    )
    progress = json.loads((run.directory / "progress.json").read_text())
    assert saved["training"]["steps"] == 8
    assert (
        json.loads((run.directory / "validations.json").read_text())[-1]["rollout"] == 2
    )
    for row in progress:
        assert sum(row["reward_terms"].values()) == pytest.approx(
            row["reward"], abs=1e-5
        )
    report = run.evaluate(episodes=3, seed=900000, backend="cpu")
    assert report["policy_input_dim"] == 1
    assert report["episode_count"] == 3
    assert len(report["goal_sectors"]) == 8
    assert all(0.3 <= r["initial_goal_radius_m"] <= 0.6 for r in report["episodes"])
    assert all(set(r["reward_terms"]) == set(task.rewards) for r in report["episodes"])
    assert (run.directory / "task.json").exists()


def test_old_checkpoint_transfer_checks_semantics_even_with_same_dimensions(tmp_path):
    cfg = SinglePushConfig(episode_seconds=0.4)
    old = train_ppo(
        cfg,
        tmp_path / "old",
        steps=8,
        num_envs=2,
        horizon=2,
        backend="cpu",
        max_seconds=None,
        validate_every=1,
        settings=SETTINGS,
    )
    task = replace(load_task(old), rewards={"goal_progress": 3.0})
    run = train(
        replace(task, config=replace(task.config, episode_seconds=0.6)),
        tmp_path / "new",
        initial_checkpoint=old,
        steps=8,
        num_envs=2,
        horizon=2,
        backend="cpu",
        validate_every=1,
        settings=SETTINGS,
        checkpoint_selection="latest",
    )
    assert run.checkpoint == run.latest
    assert run.best.is_file()
    metadata = json.loads((run.directory / "run.json").read_text())
    assert metadata["checkpoint_selection"] == "latest"
    assert metadata["steps"] == 8
    original, _, _, _ = load_ppo(old)
    copied, _, _, _ = load_ppo(run.directory / "initial.pt")
    for name, value in original["actor"].items():
        torch.testing.assert_close(value, copied["actor"][name])
    assert not copied["optimizer"]["state"]
    for index, changed in enumerate(
        (
            replace(task, observations=(ObservationTerm("baseline", scale=2),)),
            replace(task, actions=DiscreteDirections(frame="body")),
            replace(task, actions=DiscreteDirections(4)),
        )
    ):
        with pytest.raises(ValueError, match="semantics"):
            train(
                changed,
                tmp_path / f"bad_{index}",
                initial_checkpoint=old,
                steps=8,
                num_envs=2,
                horizon=2,
                backend="cpu",
                settings=SETTINGS,
            )


def test_custom_source_survives_original_removal_and_run_move(tmp_path):
    source = tmp_path / "my_task.py"
    source.write_text("""from hexapod_transport_rl.student import CylinderPushTask, ObservationTerm, RewardTerm
def distance(state):
    return state.goal_distance
def progress(before, after):
    return before.goal_distance-after.goal_distance
TASK = CylinderPushTask(observations=(ObservationTerm(distance),), rewards={"custom_progress": RewardTerm(progress, weight=7.)})
""")
    original_task = load_task_file(source)
    task = replace(
        original_task, config=replace(original_task.config, episode_seconds=0.4)
    )
    run = train(
        task,
        tmp_path / "original",
        steps=8,
        num_envs=2,
        horizon=2,
        backend="cpu",
        validate_every=1,
        settings=SETTINGS,
    )
    signature = task.policy_signature()
    source.unlink()
    moved = tmp_path / "moved"
    shutil.move(run.directory, moved)
    assert load_task(moved / "latest.pt").policy_signature() == signature
    code = f"from hexapod_transport_rl.student import evaluate; r=evaluate({str(moved / 'latest.pt')!r}, episodes=2, backend='cpu'); assert r['policy_input_dim']==1 and r['episode_count']==2; print('restored')"
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "MUJOCO_GL": "egl"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "restored" in result.stdout
    for file in (moved / "task_sources").glob("*.py"):
        file.write_text("changed")
    with pytest.raises(ValueError, match="missing or changed"):
        load_task(moved / "latest.pt")


def test_rotation_and_invalid_terms_fail_before_long_training(tmp_path):
    with pytest.raises(ValueError, match="rotation='none'"):
        replace(CylinderPushTask.baseline(), actions=DiscreteDirections(4))
    with pytest.raises(ValueError, match="Unknown observation"):
        CylinderPushTask(observations=("typo",))
    with pytest.raises(TypeError, match="module-level"):
        ObservationTerm(lambda s: s.goal_distance)
    with pytest.raises(ValueError, match="scale"):
        ObservationTerm("goal_distance", scale=0)
    for func, error in ((wrong_observation, ValueError), (nan_observation, ValueError)):
        with pytest.raises(error):
            CylinderPushTask(observations=(ObservationTerm(func),)).make_env(
                num_envs=2, backend="cpu"
            )
    with pytest.raises(ValueError, match="multiple"):
        train(
            quick_task(),
            tmp_path / "inexact",
            steps=7,
            num_envs=2,
            horizon=2,
            backend="cpu",
        )


def test_mask_preserves_dimension_and_comparison_writes_seed_summary(tmp_path):
    full = quick_task(observations=("cargo_to_goal", "robot_velocity"))
    masked = replace(
        full,
        observations=(
            "cargo_to_goal",
            ObservationTerm("robot_velocity", enabled=False),
        ),
    )
    env = masked.make_env(num_envs=2, backend="cpu")
    try:
        td = env.reset()
        assert td["observation"].shape == (2, 4)
        assert not td["observation"][:, 2:].any()
        assert env.observation_layout[1]["enabled"] is False
    finally:
        env.close()
    result = compare(
        {"full": full, "masked": masked},
        tmp_path / "comparison",
        seeds=(1, 2),
        evaluation_episodes=2,
        steps=8,
        num_envs=2,
        horizon=2,
        backend="cpu",
        validation_envs=2,
        validate_every=1,
        settings=SETTINGS,
    )
    assert len(result["records"]) == 4
    assert result["summary"]["masked"]["seeds_completed"] == 2
    assert (tmp_path / "comparison/comparison.csv").is_file()
    changed = replace(full, config=replace(full.config, position_tolerance=0.1))
    with pytest.raises(ValueError, match="share physics"):
        compare(
            {"full": full, "changed": changed}, tmp_path / "unfair", seeds=(1,), steps=8
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_student_custom_terms_actual_update_and_graph(tmp_path):
    task = quick_task(
        observations=("cargo_to_goal", ObservationTerm(scalar_observation)),
        actions=DiscreteDirections(4, frame="body"),
        rewards={"progress": RewardTerm(progress_reward, weight=20.0), "time": -0.02},
    )
    run = train(
        task,
        tmp_path / "cuda",
        steps=16,
        num_envs=2,
        horizon=4,
        backend="warp",
        validation_envs=2,
        validate_every=1,
        settings=SETTINGS,
    )
    saved, _, _, _ = load_ppo(run.latest)
    assert saved["network"] == dict(observation_dim=3, action_count=5)
    assert saved["training"]["capture_step"] and saved["optimizer"]["state"]
    assert run.evaluate(episodes=2, backend="warp")["episode_count"] == 2
