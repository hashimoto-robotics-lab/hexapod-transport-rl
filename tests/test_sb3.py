"""Real SB3 training, decentralized shared actors, and portable transport replay."""

import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv

from hexapod_transport_rl import (
    ApproachConfig,
    ApproachCurriculum,
    ApproachEnv,
    ApproachRewardWeights,
    SharedTeamPolicy,
    bind_transport,
    evaluate_transport,
)
from hexapod_transport_rl.approach_training import load_approach_checkpoint

PUSHER = Path(__file__).resolve().parents[1] / "checkpoints/pusher.pt"


@pytest.fixture(autouse=True)
def cpu_workers(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")


def test_actor_is_local_shared_and_critic_uses_both_robots():
    env = ApproachEnv(flatten=True)
    model = PPO(SharedTeamPolicy, env, n_steps=8, batch_size=8, seed=4, device="cpu")
    obs = torch.linspace(-0.3, 0.3, 20).reshape(1, 20)
    modified = obs.clone()
    modified[:, 10:] += 0.5
    with torch.no_grad():
        first = model.policy.get_distribution(obs).mode().reshape(2, 3)
        second = model.policy.get_distribution(modified).mode().reshape(2, 3)
        swapped = (
            model.policy.get_distribution(obs.reshape(1, 2, 10).flip(1).reshape(1, 20))
            .mode()
            .reshape(2, 3)
        )
    torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
    torch.testing.assert_close(first.flip(0), swapped, rtol=1e-6, atol=1e-8)
    assert not torch.equal(first[1], second[1])
    assert model.policy.log_std.shape == (3,)
    obs.requires_grad_()
    model.policy.predict_values(obs).sum().backward()
    assert obs.grad[:, :10].abs().sum() > 0
    assert obs.grad[:, 10:].abs().sum() > 0
    env.close()


def test_sb3_updates_weights_and_replays_saved_policy_after_moving_bundle(tmp_path):
    config = ApproachConfig(seconds=0.4, reward_weights=ApproachRewardWeights(time=0.4))
    envs = make_vec_env(
        "hexapod_transport_rl:HexapodApproach-v0",
        n_envs=2,
        seed=12,
        env_kwargs={"config": config, "flatten": True},
    )
    model = PPO(
        SharedTeamPolicy,
        envs,
        n_steps=8,
        batch_size=16,
        n_epochs=2,
        seed=12,
        device="cpu",
        policy_kwargs={"log_std_init": -1.0},
    )
    before = {key: value.clone() for key, value in model.policy.state_dict().items()}
    model.learn(total_timesteps=64)
    assert model.num_timesteps == 64
    assert any(
        not torch.equal(value, before[key])
        for key, value in model.policy.state_dict().items()
    )
    initial = envs.reset()[0]
    expected, _ = model.predict(initial, deterministic=True)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    model.save(bundle / "navigator.zip")
    shutil.copy2(PUSHER, bundle / "pusher.pt")
    manifest = bind_transport(
        bundle / "navigator.zip",
        bundle / "pusher.pt",
        bundle / "transport.json",
        config=config,
        provenance=envs.get_attr("provenance")[0],
    )
    with pytest.raises(ValueError, match="reward/reset settings differ"):
        bind_transport(
            bundle / "navigator.zip",
            bundle / "pusher.pt",
            bundle / "wrong_settings.json",
            config=replace(config, reward_weights=ApproachRewardWeights()),
            provenance=envs.get_attr("provenance")[0],
        )
    assert json.loads(manifest.read_text())["num_timesteps"] == 64
    envs.close()
    moved = tmp_path / "relocated"
    bundle.rename(moved)
    navigator, _, saved = load_approach_checkpoint(moved / "transport.json")
    np.testing.assert_array_equal(
        navigator.act(initial.reshape(2, 10)).reshape(6), expected
    )
    assert Path(saved["pushing_checkpoint"]) == moved / "pusher.pt"
    report = evaluate_transport(
        moved / "transport.json",
        tmp_path / "evaluation.json",
        episodes=2,
        workers=2,
        seed=80000,
        episode_seconds=0.4,
    )
    assert len(report["episodes"]) == 2
    assert all(
        row["elapsed_seconds"] == pytest.approx(0.4) for row in report["episodes"]
    )
    (moved / "navigator.zip").write_bytes(
        (moved / "navigator.zip").read_bytes() + b"changed"
    )
    with pytest.raises(ValueError, match="navigation checkpoint has changed"):
        load_approach_checkpoint(moved / "transport.json")


def test_subprocess_rewards_and_real_curriculum_validation_are_recorded(tmp_path):
    cfg = ApproachConfig(
        seconds=0.2, layout="near", reward_weights=ApproachRewardWeights(time=0.4)
    )
    envs = make_vec_env(
        "hexapod_transport_rl:HexapodApproach-v0",
        n_envs=2,
        seed=42,
        env_kwargs={"config": cfg, "flatten": True},
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "spawn"},
    )
    envs.reset()
    _, _, _, infos = envs.step(np.zeros((2, 6), dtype=np.float32))
    assert all(info["reward_terms"]["time"] == pytest.approx(-0.08) for info in infos)
    model = PPO(
        SharedTeamPolicy,
        envs,
        n_steps=4,
        batch_size=8,
        n_epochs=1,
        seed=42,
        device="cpu",
    )
    model.set_logger(configure(str(tmp_path), ["csv"]))
    curriculum = ApproachCurriculum(
        config=cfg, output=tmp_path, validate_every=1, minimum_rollouts=1
    )
    model.learn(total_timesteps=16, callback=curriculum)
    assert curriculum.validation_env.closed
    run = json.loads((tmp_path / "run.json").read_text())
    assert run["approach_config"]["reward_weights"]["time"] == 0.4
    assert run["seed"] == 42
    assert "Stable-Baselines3 PPO" in run["algorithm"]
    state = json.loads((tmp_path / "curriculum.json").read_text())
    assert state["rollouts"] == 2 and state["level"] == 0
    assert state["validation_success_rate"] == 0
    assert (tmp_path / "progress.csv").is_file()
    envs.close()


def test_curriculum_changes_reset_distribution_without_replacing_ppo_rollouts(
    tmp_path, monkeypatch
):
    cfg = ApproachConfig(seconds=0.2, layout="near")
    envs = make_vec_env(
        "hexapod_transport_rl:HexapodApproach-v0",
        n_envs=1,
        env_kwargs={"config": cfg, "flatten": True},
    )
    model = PPO(
        SharedTeamPolicy,
        envs,
        n_steps=4,
        batch_size=4,
        n_epochs=1,
        seed=4,
        device="cpu",
    )
    curriculum = ApproachCurriculum(cfg, tmp_path, validate_every=1, minimum_rollouts=1)
    # Test the scheduling hook; this does not supply actions or targets to PPO.
    monkeypatch.setattr(curriculum, "_validate", lambda: 1.0)
    model.learn(total_timesteps=8, callback=curriculum)
    assert curriculum.level == 2
    assert envs.get_attr("config")[0].layout == "side"
    assert model.num_timesteps == 8
    envs.close()


def test_normal_mlp_policy_is_not_silently_labeled_as_shared_team_policy(tmp_path):
    env = ApproachEnv(flatten=True)
    model = PPO("MlpPolicy", env, n_steps=4, batch_size=4, seed=4, device="cpu")
    model.save(tmp_path / "plain.zip")
    with pytest.raises(ValueError, match="SharedTeamPolicy"):
        bind_transport(
            tmp_path / "plain.zip",
            PUSHER,
            tmp_path / "invalid.json",
            config=ApproachConfig(),
            provenance=env.provenance,
        )
    assert not (tmp_path / "invalid.json").exists()
    env.close()
