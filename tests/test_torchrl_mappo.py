"""Real TorchRL MAPPO updates and tests of the MARL/time-limit semantics."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from tensordict import TensorDict
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, ReplayBuffer, SamplerWithoutReplacement
from torchrl.envs.utils import check_env_specs
from torchrl.objectives import MAPPOLoss

from hexapod_transport_rl import (
    ApproachConfig,
    ApproachCurriculum,
    ApproachRewardWeights,
    MAPPOSettings,
    PushConfig,
    TorchRLTransportEnv,
    evaluate_transport,
    load_mappo,
    make_mappo_loss,
    make_mappo_networks,
    policy_action,
    save_mappo,
)
from hexapod_transport_rl.approach_training import load_approach_checkpoint

PUSHER = Path(__file__).resolve().parents[1] / "checkpoints/pusher.pt"


@pytest.fixture(autouse=True)
def cpu_workers(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")


def test_actor_is_local_and_shared_with_a_central_critic():
    actor, critic = make_mappo_networks()
    observation = torch.linspace(-0.3, 0.3, 20).reshape(2, 10)
    modified = observation.clone()
    modified[1] += 0.5
    first = policy_action(actor, observation)
    second = policy_action(actor, modified)
    np.testing.assert_array_equal(first[0], second[0])
    assert not np.array_equal(first[1], second[1])
    np.testing.assert_allclose(
        policy_action(actor, observation.flip(0)), first[::-1], rtol=1e-6, atol=1e-8
    )
    scales = [
        param for name, param in actor.named_parameters() if name.endswith("log_std")
    ]
    assert len(scales) == 1 and scales[0].shape == (3,)
    observation.requires_grad_()
    td = TensorDict({"agents": {"observation": observation}}, batch_size=[])
    values = critic(td)["agents", "state_value"]
    torch.testing.assert_close(values[0], values[1])
    values.sum().backward()
    assert observation.grad[0].abs().sum() > 0
    assert observation.grad[1].abs().sum() > 0


def test_mappo_clips_each_robot_ratio_instead_of_the_joint_ratio():
    actor, critic = make_mappo_networks()
    loss = make_mappo_loss(actor, critic, MAPPOSettings())
    assert isinstance(loss, MAPPOLoss)
    loss.normalize_advantage = False
    with torch.no_grad():
        batch = actor(
            TensorDict(
                {"agents": {"observation": torch.zeros(4, 2, 10)}}, batch_size=[4]
            )
        )
    assert batch["agents", "sample_log_prob"].shape == (4, 2)
    old_log_prob = batch["agents", "sample_log_prob"].clone()
    old_log_prob[:, 0] -= np.log(2.0)  # First robot ratio 2, second robot ratio 1.
    batch["agents", "sample_log_prob"] = old_log_prob
    batch["agents", "advantage"] = torch.ones(4, 2, 1)
    batch["agents", "value_target"] = torch.zeros(4, 2, 1)
    metrics = loss(batch)
    assert float(metrics["loss_objective"].detach()) == pytest.approx(-1.1, abs=1e-6)
    assert float(metrics["clip_fraction"]) == pytest.approx(0.5)


def test_time_limit_uses_the_final_observation_and_team_reward_broadcast():
    env = TorchRLTransportEnv(
        ApproachConfig(seconds=0.2, layout="near"), num_envs=2, asynchronous=False
    )
    check_env_specs(env)
    actor, critic = make_mappo_networks()
    loss = make_mappo_loss(actor, critic, MAPPOSettings())
    collector = Collector(
        env,
        actor,
        frames_per_batch=4,
        total_frames=4,
        auto_register_policy_transforms=True,
    )
    batch = next(iter(collector))
    assert batch["next", "truncated"].all()
    assert not batch["next", "terminated"].any()
    assert batch["next", "reward"].shape == (2, 2, 1)
    assert not torch.equal(
        batch["next", "agents", "observation"], batch["agents", "observation"]
    )
    loss.value_estimator(batch)
    with torch.no_grad():
        terminal_values = critic(batch["next"].clone())["agents", "state_value"]
    reward = batch["next", "reward"].unsqueeze(-2).expand(2, 2, 2, 1)
    torch.testing.assert_close(
        batch["agents", "value_target"], reward + 0.99 * terminal_values
    )
    torch.testing.assert_close(
        batch["agents", "advantage"][:, :, 0], batch["agents", "advantage"][:, :, 1]
    )
    # A true terminal state must not bootstrap, even with a nonzero critic.
    batch["next", "terminated"] = torch.ones_like(batch["next", "terminated"])
    loss.value_estimator(batch)
    torch.testing.assert_close(batch["agents", "value_target"], reward)
    collector.shutdown()


def test_partial_reset_does_not_reset_active_worlds():
    env = TorchRLTransportEnv(
        ApproachConfig(seconds=2, layout="near"), num_envs=2, asynchronous=False
    )
    env.set_seed(42)
    td = env.reset()
    td["agents", "action"] = torch.zeros(2, 2, 3)
    stepped = env.step(td)["next"]
    before = stepped["agents", "observation"].clone()
    stepped["_reset"] = torch.tensor([[True], [False]])
    reset = env.reset(stepped)
    assert not torch.equal(reset["agents", "observation"][0], before[0])
    torch.testing.assert_close(reset["agents", "observation"][1], before[1])
    env.close()


def test_real_spawn_training_save_resume_and_portable_transport(tmp_path):
    cfg = ApproachConfig(
        seconds=0.4, layout="near", reward_weights=ApproachRewardWeights(time=0.4)
    )
    env = TorchRLTransportEnv(cfg, num_envs=2)
    env.set_seed(42)
    settings = MAPPOSettings(epochs=2, minibatch_size=8)
    actor, critic = make_mappo_networks()
    loss = make_mappo_loss(actor, critic, settings)
    optimizer = torch.optim.Adam(loss.parameters(), lr=settings.learning_rate)
    collector = Collector(
        env,
        actor,
        frames_per_batch=16,
        total_frames=32,
        auto_register_policy_transforms=True,
    )
    buffer = ReplayBuffer(
        storage=LazyTensorStorage(16), sampler=SamplerWithoutReplacement(), batch_size=8
    )
    before = {key: value.clone() for key, value in actor.state_dict().items()}
    bundle = tmp_path / "bundle"
    curriculum = ApproachCurriculum(
        cfg,
        env,
        actor,
        bundle,
        seed=42,
        settings=settings,
        horizon=8,
        validate_every=1,
        minimum_rollouts=1,
    )
    update_count = 0
    for batch in collector:
        loss.value_estimator(batch)
        buffer.empty()
        buffer.extend(batch.reshape(-1))
        for _ in range(settings.epochs):
            for minibatch in buffer:
                metrics = loss(minibatch)
                objective = sum(
                    metrics[key]
                    for key in ("loss_objective", "loss_critic", "loss_entropy")
                )
                optimizer.zero_grad()
                objective.backward()
                torch.nn.utils.clip_grad_norm_(
                    loss.parameters(), settings.max_grad_norm
                )
                optimizer.step()
                update_count += 1
        curriculum.record(batch, metrics)
    assert update_count == 8
    assert any(
        not torch.equal(value, before[key]) for key, value in actor.state_dict().items()
    )
    assert curriculum.transitions == 32 and curriculum.last_validation_success_rate == 0
    assert env.worlds.call("config")[0].reward_weights.time == 0.4
    run = json.loads((bundle / "run.json").read_text())
    assert run["algorithm"].startswith("TorchRL MAPPO")
    assert run["settings"] == curriculum.training["settings"]
    assert (bundle / "progress.csv").is_file()
    observation = batch["agents", "observation"][0, 0].numpy()
    expected = policy_action(actor, observation)
    (bundle / "pusher.pt").write_bytes(PUSHER.read_bytes())
    checkpoint = save_mappo(
        bundle / "transport.pt",
        actor,
        critic,
        optimizer,
        config=cfg,
        pushing_checkpoint=bundle / "pusher.pt",
        provenance=env.provenance,
        training=curriculum.training,
        curriculum=curriculum.state,
    )
    with pytest.raises(ValueError, match="reward/reset settings differ"):
        save_mappo(
            bundle / "wrong.pt",
            actor,
            critic,
            optimizer,
            config=replace(cfg, reward_weights=ApproachRewardWeights()),
            pushing_checkpoint=PUSHER,
            provenance=env.provenance,
            training=curriculum.training,
            curriculum=curriculum.state,
        )
    curriculum.close()
    collector.shutdown()
    assert env.is_closed and curriculum.validation_env is None
    moved = tmp_path / "relocated"
    bundle.rename(moved)
    navigator, _, saved = load_approach_checkpoint(moved / checkpoint.name)
    np.testing.assert_array_equal(navigator.act(observation), expected)
    report = evaluate_transport(
        moved / "transport.pt",
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
    resumed_actor, resumed_critic, saved = load_mappo(moved / "transport.pt")
    resumed_loss = make_mappo_loss(resumed_actor, resumed_critic, settings)
    resumed_optimizer = torch.optim.Adam(
        resumed_loss.parameters(), lr=settings.learning_rate
    )
    resumed_optimizer.load_state_dict(saved["optimizer"])
    assert resumed_optimizer.state_dict()["state"]
    resumed_env = TorchRLTransportEnv(cfg, num_envs=1, asynchronous=False)
    resumed = ApproachCurriculum(
        cfg,
        resumed_env,
        resumed_actor,
        tmp_path / "resume",
        seed=42,
        settings=settings,
        horizon=8,
        resume=saved,
    )
    assert resumed.transitions == 32 and resumed.rollouts == 2
    resumed.close()
    resumed_env.close()
    (moved / "pusher.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Frozen pushing checkpoint has changed"):
        load_mappo(moved / "transport.pt")


def test_curriculum_advances_without_replacing_collector_data(tmp_path, monkeypatch):
    cfg = ApproachConfig(seconds=0.2, layout="near")
    env = TorchRLTransportEnv(cfg, num_envs=1, asynchronous=False)
    actor, critic = make_mappo_networks()
    loss = make_mappo_loss(actor, critic, MAPPOSettings())
    collector = Collector(
        env,
        actor,
        frames_per_batch=4,
        total_frames=8,
        auto_register_policy_transforms=True,
    )
    curriculum = ApproachCurriculum(
        cfg,
        env,
        actor,
        tmp_path,
        seed=4,
        settings=MAPPOSettings(),
        horizon=4,
        validate_every=1,
        minimum_rollouts=1,
    )
    monkeypatch.setattr(curriculum, "_validate", lambda: 1.0)
    for batch in collector:
        loss.value_estimator(batch)
        curriculum.record(batch, loss(batch.reshape(-1)))
    assert curriculum.level == 2 and curriculum.transitions == 8
    assert env.worlds.call("config")[0].layout == "side"
    curriculum.close()
    collector.shutdown()


def test_four_robot_pushing_connects_to_torchrl_mappo():
    cfg = PushConfig(shape="T", num_robots=4, episode_seconds=0.4)
    env = TorchRLTransportEnv(cfg, num_envs=1, asynchronous=False)
    check_env_specs(env)
    actor, critic = make_mappo_networks(num_robots=4, obs_dim=env.obs_dim)
    loss = make_mappo_loss(actor, critic, MAPPOSettings())
    collector = Collector(
        env,
        actor,
        frames_per_batch=4,
        total_frames=4,
        auto_register_policy_transforms=True,
    )
    batch = next(iter(collector))
    assert batch["agents", "action"].shape == (1, 4, 4, 3)
    loss.value_estimator(batch)
    metrics = loss(batch.reshape(-1))
    objective = sum(
        metrics[key] for key in ("loss_objective", "loss_critic", "loss_entropy")
    )
    objective.backward()
    assert torch.isfinite(objective)
    assert any(param.grad is not None for param in actor.parameters())
    collector.shutdown()
