"""Fresh experiment state, real TorchRL updates, time budgets and 25 fps video."""

import json

import mediapy as media
import numpy as np
import pytest
import torch

from hexapod_transport_rl import (
    PushTConfig,
    PushTEnv,
    PushTRewardWeights,
    evaluate_mappo,
    load_mappo,
    make_pusht_networks,
    policy_action,
    train_mappo,
)


@pytest.mark.parametrize("obs_type", ["state", "keypoints"])
@pytest.mark.parametrize("num_robots", [2, 4])
def test_initial_mean_centers_on_own_position_without_a_cargo_rule(
    obs_type, num_robots
):
    cfg = PushTConfig(obs_type=obs_type, num_robots=num_robots)
    env = PushTEnv(cfg)
    try:
        obs, info = env.reset(seed=42)
        actor, _ = make_pusht_networks(cfg)
        # A zero learned residual must hold own XY, irrespective of the cargo.
        for name, parameter in actor.named_parameters():
            if not name.endswith("log_std"):
                parameter.data.zero_()
        np.testing.assert_allclose(
            policy_action(actor, obs), info["agent_pos"], atol=2e-6
        )
        changed = obs.copy()
        if obs_type == "state":
            changed[:, 2:5] += 1.0
        else:
            changed[:, :16] += 1.0
        np.testing.assert_allclose(
            policy_action(actor, changed), info["agent_pos"], atol=2e-6
        )
    finally:
        env.close()


@pytest.mark.parametrize("backend", ["cpu", "warp"])
@pytest.mark.parametrize("num_robots", [2, 4])
def test_fresh_mappo_state_update_reload_and_video(tmp_path, backend, num_robots):
    if backend == "warp" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    cfg = PushTConfig(
        num_robots=num_robots,
        high_level_decimation=10,
        episode_seconds=0.8,
        reward_weights=PushTRewardWeights(coverage=0, progress=10, approach=1),
    )
    checkpoint = train_mappo(
        cfg,
        tmp_path / "first",
        seed=17,
        steps=8,
        num_envs=2,
        horizon=4,
        max_seconds=None,
        backend=backend,
    )
    actor, _, saved = load_mappo(checkpoint)
    initial = torch.load(
        checkpoint.parent / "initial.pt", weights_only=True, map_location="cpu"
    )
    assert not initial["optimizer"]["state"]
    assert float(initial["value_normalizer"]["debiasing_term"]) == 0
    assert any(
        not torch.equal(v, saved["actor"][k]) for k, v in initial["actor"].items()
    )
    assert saved["network"]["position_index"] == 0
    assert saved["network"]["initial_log_std"] == -1.8
    assert saved["training"]["team_steps"] == 8
    assert saved["training"]["stopped_by"] == "step_budget"
    assert not saved["training"]["demonstration_actions_used"]
    report = evaluate_mappo(
        checkpoint, episodes=2, seed=98000, backend=backend, video=num_robots == 2
    )
    assert len(report["episodes"]) == 2
    if num_robots == 2:
        frames = media.read_video(report["video_path"])
        # The coarser action interval still records actual 25 Hz physics frames.
        assert len(frames) == 21  # 0.8 s / 0.04 s + the reset frame
    env = PushTEnv(cfg)
    try:
        obs, _ = env.reset(seed=42)
        assert env.action_space.contains(policy_action(actor, obs))
    finally:
        env.close()
    # Repeating the seed creates the same initial state, not the previous final one.
    second = train_mappo(
        cfg,
        tmp_path / "second",
        seed=17,
        steps=8,
        num_envs=2,
        horizon=4,
        max_seconds=None,
        backend=backend,
    )
    repeated = torch.load(
        second.parent / "initial.pt", weights_only=True, map_location="cpu"
    )
    for key in ("actor", "critic", "value_normalizer"):
        for name, value in initial[key].items():
            torch.testing.assert_close(value, repeated[key][name], rtol=0, atol=0)
    assert not repeated["optimizer"]["state"]


def test_time_budget_records_zero_updates_instead_of_claiming_learning(tmp_path):
    checkpoint = train_mappo(
        PushTConfig(),
        tmp_path,
        steps=8,
        num_envs=2,
        horizon=4,
        backend="cpu",
        max_seconds=1e-6,
    )
    metadata = json.loads((tmp_path / "run.json").read_text())
    assert metadata["team_steps"] == metadata["rollouts"] == 0
    assert metadata["stopped_by"] == "time_budget"
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    assert not saved["optimizer"]["state"]
