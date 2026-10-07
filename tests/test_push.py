import numpy as np
import pytest
import torch

from hexapod_transport_rl import PushConfig, PushEnv
from hexapod_transport_rl.config import find_asset_root
from hexapod_transport_rl.mappo import compute_gae, load_checkpoint, train
from hexapod_transport_rl.model import build_model


@pytest.fixture(scope="module")
def env():
    torch.set_num_threads(1)
    return PushEnv()


def test_articulated_team_and_collision_masks(env):
    assert (env.model.nq, env.model.nv, env.model.nu) == (57, 54, 36)
    assert env.model.body("cargo").mass[0] == pytest.approx(2)
    model = env.model

    def collides(a, b):
        a, b = model.geom(a).id, model.geom(b).id
        return bool(
            (model.geom_contype[a] & model.geom_conaffinity[b])
            or (model.geom_contype[b] & model.geom_conaffinity[a])
        )

    assert not any("bumper" in (model.geom(g).name or "") for g in range(model.ngeom))
    assert collides("r0/LF_tibia_collision1", "cargo_box")
    assert collides("r1/RF_foot_collision", "cargo_box")
    assert collides("r0/base_collision0", "cargo_box")
    assert collides("r0/LF_foot_collision", "r1/LF_foot_collision")
    assert collides("r0/LF_foot_collision", "floor")
    assert not collides("r0/LF_foot_collision", "r0/base_collision0")
    assert not collides("goal_cargo_box", "r0/LF_foot_collision")


def test_reset_seed_and_per_robot_action_memory(env):
    a, _ = env.reset(seed=91, randomize=True)
    q = env.data.qpos.copy()
    env.step(np.array([[0.5, 0, 0], [0, 0, 0]], dtype=np.float32))
    assert not np.array_equal(env.previous_action[0], env.previous_action[1])
    b, _ = env.reset(seed=91, randomize=True)
    np.testing.assert_array_equal(q, env.data.qpos)
    np.testing.assert_array_equal(a, b)
    assert not env.previous_action.any()
    with pytest.raises(ValueError):
        env.step(np.full((2, 3), np.nan))


def test_timeout_and_success_are_distinct():
    env = PushEnv(PushConfig(episode_seconds=0.2))
    _, _, terminated, truncated, _ = env.step(np.zeros((2, 3)))
    assert truncated and not terminated
    with pytest.raises(RuntimeError):
        env.step(np.zeros((2, 3)))
    env = PushEnv(PushConfig(episode_seconds=1))
    env.goal = env.cargo_xy.copy()
    for _ in range(3):
        _, _, terminated, truncated, info = env.step(np.zeros((2, 3)))
    assert terminated and not truncated and info["success"]


@pytest.mark.parametrize("n,shape", [(3, "box"), (4, "T")])
def test_other_team_sizes_and_payloads(n, shape):
    model = build_model(find_asset_root(), PushConfig(num_robots=n, shape=shape))
    assert model.nu == 18 * n
    assert model.nq == 25 * n + 7
    assert model.body("cargo").mass[0] == pytest.approx(2)


def test_timeout_gae_bootstraps_without_crossing_reset():
    rewards = torch.tensor([[1.0], [100.0]])
    values = torch.zeros(2, 1)
    next_values = torch.tensor([[10.0], [20.0]])
    terminated = torch.tensor([[False], [True]])
    truncated = torch.tensor([[True], [False]])
    advantages, returns = compute_gae(
        rewards, values, next_values, terminated, truncated, gamma=0.9
    )
    torch.testing.assert_close(advantages, torch.tensor([[10.0], [100.0]]))
    torch.testing.assert_close(returns, advantages)


def test_training_checkpoint_and_replay(tmp_path):
    torch.set_num_threads(1)
    cfg = PushConfig(episode_seconds=0.2)
    path = train(
        cfg,
        None,
        tmp_path / "train",
        iterations=2,
        num_envs=1,
        horizon=4,
        epochs=1,
        minibatch=4,
        seed=11,
    )
    agent, saved = load_checkpoint(path)
    assert saved["iteration"] == 2
    assert saved["transitions"] == 8
    assert saved["optimizer"]["state"]
    assert agent.actor[-1].bias.abs().sum().item() > 0
    env = PushEnv(cfg)
    obs, _ = env.reset(seed=11)
    action = agent.act(obs)
    assert action.shape == (2, 3) and np.max(np.abs(action)) <= 1
    assert np.isfinite(env.step(action)[1])
    again, _ = load_checkpoint(path)
    np.testing.assert_array_equal(action, again.act(obs))
    resumed = train(
        cfg,
        None,
        tmp_path / "resumed",
        iterations=1,
        num_envs=1,
        horizon=2,
        epochs=1,
        minibatch=2,
        seed=12,
        resume=path,
    )
    _, resumed_state = load_checkpoint(resumed)
    assert resumed_state["iteration"] == 3
    assert resumed_state["transitions"] == 10


def test_bumper_checkpoint_rejected(tmp_path):
    path = tmp_path / "old.pt"
    torch.save({"format": "sol4-push-mappo-v1"}, path)
    with pytest.raises(ValueError, match="hexapod leg-pushing v1"):
        load_checkpoint(path)
