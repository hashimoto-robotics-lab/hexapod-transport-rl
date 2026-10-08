"""Real CUDA physics, formula parity, reset isolation and TorchRL time limits."""

import mujoco
import numpy as np
import pytest
import torch
from torchrl.collectors import Collector
from torchrl.envs.utils import check_env_specs

from hexapod_transport_rl import (
    POSE_STAGES,
    MAPPOSettings,
    PoseCurriculum,
    PosePushConfig,
    PosePushEnv,
    TorchRLTransportEnv,
    load_mappo,
    make_mappo_loss,
    make_mappo_networks,
    save_mappo,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="NVIDIA CUDA is required for MuJoCo Warp"
)


@pytest.fixture
def worlds():
    pytest.importorskip("mujoco_warp")
    torch.set_num_threads(1)
    env = TorchRLTransportEnv(PosePushConfig(), num_envs=4, backend="warp")
    yield env.worlds
    env.close()


def copy_lane(worlds, native, lane):
    core = native.core
    core.data.qpos[:] = worlds.physics.qpos[lane].cpu().numpy()
    core.data.qvel[:] = worlds.physics.qvel[lane].cpu().numpy()
    core.goal = worlds.goal[lane].cpu().numpy()
    core.goal_yaw = float(worlds.goal_yaw[lane])
    core.last_commands = worlds.last_commands[lane].cpu().numpy().copy()
    mujoco.mj_forward(core.model, core.data)


def test_gpu_observation_and_reward_match_native_formulas(worlds):
    native = PosePushEnv()
    before = worlds.measurements()
    before = {key: value.clone() for key, value in before.items()}
    action = torch.tensor([0.7, -0.2, 0.3], device="cuda").expand(4, 2, 3)
    observation, reward, _, _, after = worlds.step(action)
    for lane in range(4):
        copy_lane(worlds, native, lane)
        np.testing.assert_allclose(
            observation[lane].cpu(), native.core.observe(), atol=2e-6, rtol=2e-6
        )
        contacts = worlds.physics.contacts
        native.core.contacts.part_contact_counts = (
            contacts.part_contact_counts[lane].cpu().numpy()
        )
        native.core.contacts.robot_collision_steps = int(
            contacts.robot_collision_steps[lane]
        )
        commands = worlds.last_commands[lane].cpu().numpy()
        native.core.last_commands = np.zeros_like(commands)
        old = {key: float(value[lane]) for key, value in before.items()}
        new = {
            key: value[lane].item()
            for key, value in after.items()
            if isinstance(value, torch.Tensor)
        }
        expected = native.core._reward_terms(old, new, old["approach"], commands)
        for key, value in expected.items():
            assert float(after["reward_terms"][key][lane]) == pytest.approx(
                value, abs=2e-6
            )
        assert float(reward[lane]) == pytest.approx(sum(expected.values()), abs=3e-6)
    worlds.physics.check()
    native.close()


def test_partial_reset_preserves_active_physics_walker_and_tolerances(worlds):
    worlds.step(torch.full((4, 2, 3), 0.2, device="cuda"))
    qpos, qvel = worlds.physics.qpos.clone(), worlds.physics.qvel.clone()
    history = worlds.physics.previous_action.clone()
    previous = worlds.last_commands.clone()
    worlds.call("set_stage", POSE_STAGES[0])
    mask = torch.tensor([True, False, False, True], device="cuda")
    worlds.reset(options={"reset_mask": mask})
    torch.testing.assert_close(worlds.physics.qpos[~mask], qpos[~mask], rtol=0, atol=0)
    torch.testing.assert_close(worlds.physics.qvel[~mask], qvel[~mask], rtol=0, atol=0)
    torch.testing.assert_close(
        worlds.physics.previous_action[~mask], history[~mask], rtol=0, atol=0
    )
    torch.testing.assert_close(
        worlds.last_commands[~mask], previous[~mask], rtol=0, atol=0
    )
    assert bool((worlds.physics.previous_action[mask] == 0).all())
    torch.testing.assert_close(
        worlds.hold_seconds, torch.tensor([0.6, 1.0, 1.0, 0.6], device="cuda")
    )
    torch.testing.assert_close(
        worlds.yaw_tolerance,
        torch.tensor(np.deg2rad([8, 5, 5, 8]), device="cuda", dtype=torch.float32),
    )
    first, _ = worlds.reset(seed=42)
    second, _ = worlds.reset(seed=42)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_gpu_contacts_include_physical_legs_and_detect_body(worlds):
    # Bring the T close enough for the standing front legs to touch it.
    worlds.reset(options={"randomize": False})
    worlds.physics.qpos[:, worlds.root_q + 0] += 0.11
    worlds.physics.forward()
    leg_ticks = torch.zeros(4, device="cuda")
    for _ in range(5):
        worlds.step(torch.tensor([0.7, 0.0, 0.0], device="cuda").expand(4, 2, 3))
        counts = worlds.physics.contacts.part_contact_counts
        leg_ticks += counts[:, 1:].sum((1, 2))
    assert bool((leg_ticks > 0).all())
    assert bool((counts[:, 0] == 0).all())
    worlds.reset(options={"randomize": False})
    worlds.physics.qpos[:, worlds.root_q] += 0.20
    worlds.physics.forward()
    worlds.step(torch.zeros(4, 2, 3, device="cuda"))
    assert bool((worlds.physics.contacts.part_contact_counts[:, 0].sum(-1) > 0).all())
    assert bool((worlds.physics.contacts.normal_impulses[:, 0].sum(-1) > 0).all())
    worlds.physics.check()


def test_four_robot_cuda_learning_timeout_bootstrap_and_portable_checkpoint(tmp_path):
    pytest.importorskip("mujoco_warp")
    torch.set_num_threads(1)
    config = PosePushConfig(num_robots=4, episode_seconds=0.2)
    env = TorchRLTransportEnv(config, num_envs=2, backend="warp")
    env.set_seed(41)
    check_env_specs(env)
    actor, critic = make_mappo_networks(4, 30)
    actor.cuda()
    critic.cuda()
    settings = MAPPOSettings(epochs=1, minibatch_size=4, value_normalization=True)
    loss = make_mappo_loss(actor, critic, settings)
    optimizer = torch.optim.Adam(loss.parameters(), lr=settings.learning_rate)
    curriculum = PoseCurriculum(
        config, env, actor, tmp_path / "run", seed=41, settings=settings, horizon=2
    )
    collector = Collector(
        env,
        actor,
        frames_per_batch=4,
        total_frames=4,
        env_device=env.device,
        policy_device=env.device,
        storing_device=env.device,
        auto_register_policy_transforms=True,
    )
    batch = next(iter(collector))
    assert batch.device.type == "cuda"
    assert bool(batch["next", "truncated"].all())
    assert not bool(batch["next", "terminated"].any())
    final = batch["next", "agents", "observation"][:, 0]
    reset = batch["agents", "observation"][:, 1]
    assert not torch.equal(final, reset)
    loss.value_estimator(batch)
    metrics = loss(batch.reshape(-1))
    objective = sum(
        metrics[key] for key in ("loss_objective", "loss_critic", "loss_entropy")
    )
    optimizer.zero_grad()
    objective.backward()
    optimizer.step()
    assert next(actor.parameters()).grad.is_cuda
    curriculum.record(batch, metrics)
    saved = save_mappo(
        tmp_path / "model.pt",
        actor,
        critic,
        optimizer,
        config=config,
        provenance=env.provenance,
        training=curriculum.training,
        curriculum=curriculum.state,
        loss=loss,
    )
    loaded, _, metadata = load_mappo(saved)
    assert next(loaded.parameters()).device.type == "cpu"
    assert metadata["training"]["physics_backend"] == "warp"
    assert metadata["training"]["physics_device"] == "cuda:0"
    assert np.isfinite(env.render()).all()
    collector.shutdown()
    curriculum.close()


def test_rollout_check_rejects_capacity_overflow(worlds):
    worlds.physics.overflow[0] = 1
    with pytest.raises(RuntimeError, match="capacity"):
        worlds.physics.check()
    worlds.physics.overflow_seen[0] = 1
    worlds.physics.overflow.zero_()
    worlds.reset()
    with pytest.raises(RuntimeError, match="capacity"):
        worlds.physics.check()
    worlds.physics.overflow_seen.zero_()
