"""Reward experiments must alter learning without changing physics or other runs."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from hexapod_transport_rl import (
    ApproachConfig,
    ApproachEnv,
    ApproachRewardWeights,
    PushConfig,
    PushEnv,
    PushRewardWeights,
    train_approach,
)
from hexapod_transport_rl.approach import make_approach_vector
from hexapod_transport_rl.approach_training import load_approach_checkpoint
from hexapod_transport_rl.experiments import archive_results, create_experiment
from hexapod_transport_rl.handover_training import compose, train_handover
from hexapod_transport_rl.mappo import load_checkpoint
from hexapod_transport_rl.rewards import approach_reward_terms

ROOT = Path(__file__).resolve().parents[1]
PUSHER = ROOT / "checkpoints/pusher.pt"


def test_old_and_new_checkpoint_configs_round_trip():
    for config in (ApproachConfig(), PushConfig()):
        cls = type(config)
        assert cls(**asdict(config)) == config
        legacy = asdict(config)
        legacy.pop("reward_weights")
        assert cls(**legacy) == config
    with pytest.raises(ValueError, match="finite"):
        ApproachRewardWeights(robot_contact=float("nan"))


def test_contact_coefficients_apply_to_actual_contact_duration():
    arguments = dict(
        progress=0.1,
        dt=0.2,
        robot_contact=0.5,
        body_contact=0.0,
        cargo_displacement=0.0,
        success=False,
        failed=False,
    )
    first = approach_reward_terms(ApproachRewardWeights(), **arguments)
    changed = approach_reward_terms(
        ApproachRewardWeights(robot_contact=12), **arguments
    )
    assert first["robot_contact"] == pytest.approx(-0.4)
    assert changed["robot_contact"] == pytest.approx(-1.2)
    assert sum(changed.values()) - sum(first.values()) == pytest.approx(-0.8)


def test_environment_rewards_are_independent_and_match_diagnostics():
    torch.set_num_threads(1)
    cfg = ApproachConfig(layout="front")
    changed = replace(cfg, reward_weights=replace(cfg.reward_weights, time=0.4))
    with (
        ApproachEnv(cfg, render_mode="rgb_array", width=160, height=120) as baseline,
        ApproachEnv(changed) as variant,
    ):
        baseline.reset(seed=123)
        variant.reset(seed=123)
        action = np.zeros((2, 3), dtype=np.float32)
        obs_a, reward_a, _, _, info_a = baseline.step(action)
        obs_b, reward_b, _, _, info_b = variant.step(action)
        np.testing.assert_array_equal(obs_a, obs_b)
        np.testing.assert_array_equal(baseline.core.data.qpos, variant.core.data.qpos)
        assert reward_b - reward_a == pytest.approx(-0.36 * 0.2)
        assert reward_a == sum(info_a["reward_terms"].values())
        assert reward_b == sum(info_b["reward_terms"].values())
        np.testing.assert_array_equal(baseline.state(), obs_a.reshape(-1))
        assert baseline.render().shape == (120, 160, 3)
    with pytest.raises(RuntimeError, match="closed"):
        baseline.render()


def test_push_rewards_change_reward_but_preserve_physics():
    cfg = PushConfig(shape="T")
    first = PushEnv(cfg)
    second = PushEnv(
        replace(cfg, reward_weights=replace(cfg.reward_weights, time=1.01))
    )
    first.reset(seed=50)
    second.reset(seed=50)
    action = np.zeros((2, 3), dtype=np.float32)
    obs_a, reward_a, _, _, _ = first.step(action)
    obs_b, reward_b, _, _, info = second.step(action)
    np.testing.assert_array_equal(obs_a, obs_b)
    assert reward_b - reward_a == pytest.approx(-cfg.dt)
    assert reward_b == sum(info["reward_terms"].values())


def test_custom_rewards_reach_spawned_training_workers(tmp_path):
    cfg = ApproachConfig(reward_weights=ApproachRewardWeights(time=0.4))
    envs = make_approach_vector(cfg, num_envs=2)
    try:
        envs.reset(seed=50)
        _, rewards, _, _, info = envs.step(np.zeros((2, 2, 3), dtype=np.float32))
        assert isinstance(envs, gym.vector.AsyncVectorEnv)
        np.testing.assert_allclose(info["reward_terms"]["time"], -0.08)
        assert np.isfinite(rewards).all()
        assert all(
            item["approach_config"]["reward_weights"]["time"] == 0.4
            for item in envs.call("provenance")
        )
    finally:
        envs.close()
    output = train_approach(
        tmp_path / "nav",
        PUSHER,
        config=cfg,
        iterations=1,
        num_envs=1,
        horizon=8,
        seed=50,
    )
    _, _, saved = load_approach_checkpoint(output)
    assert (
        ApproachConfig(**saved["approach_config"]).reward_weights == cfg.reward_weights
    )
    run = json.loads((output.parent / "run.json").read_text())
    assert run["approach_config"]["reward_weights"]["time"] == 0.4
    with pytest.raises(ValueError, match="settings differ"):
        train_approach(
            tmp_path / "wrong_resume",
            PUSHER,
            resume=output,
            config=ApproachConfig(),
            iterations=1,
            num_envs=1,
            horizon=8,
        )
    resumed = train_approach(
        tmp_path / "resumed",
        PUSHER,
        resume=output,
        iterations=1,
        num_envs=1,
        horizon=8,
        seed=50,
    )
    assert load_approach_checkpoint(resumed)[2]["iteration"] == 2


def test_push_reward_warm_start_and_physical_composition(tmp_path):
    weights = PushRewardWeights(time=0.5)
    output = train_handover(
        PUSHER,
        tmp_path / "push",
        iterations=1,
        num_envs=1,
        horizon=8,
        reward_weights=weights,
        seed=12,
    )
    _, saved = load_checkpoint(output)
    assert PushConfig(**saved["config"]).reward_weights == weights
    navigator = ROOT / "checkpoints/transport.pt"
    composed = compose(navigator, output, tmp_path / "composed.pt")
    _, pusher, nav_saved = load_approach_checkpoint(composed)
    assert pusher is not None
    assert Path(nav_saved["pushing_checkpoint"]) == output.resolve()


def test_results_snapshot_includes_exact_sources_assets_and_native_runtime(tmp_path):
    # A real repository snapshot must describe edits, not merely the Git commit.
    output = create_experiment(ROOT, f"pytest_reward_{tmp_path.name}")
    try:
        record = json.loads((output / "experiment.json").read_text())
        assert "src/hexapod_transport_rl/rewards.py" in record["source_sha256"]
        assert record["packages"]["torch"] == torch.__version__
        snapshot = output / "source_snapshot/src/hexapod_transport_rl"
        assert (snapshot / "rewards.py").read_bytes() == (
            ROOT / "src/hexapod_transport_rl/rewards.py"
        ).read_bytes()
        assert (
            output / "source_snapshot/checkpoints/pusher.pt"
        ).read_bytes() == PUSHER.read_bytes()
        with pytest.raises(FileExistsError):
            create_experiment(ROOT, output.name)
        assert archive_results(output).is_file()
    finally:
        import shutil

        shutil.rmtree(output)
        output.with_suffix(".zip").unlink(missing_ok=True)
