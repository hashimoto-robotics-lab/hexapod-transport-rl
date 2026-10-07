"""Learning, deployment and checkpoint loading must use the same actor."""

import numpy as np
import torch

from hexapod_transport_rl import PushConfig
from hexapod_transport_rl.mappo import MAPPO, load_checkpoint, train


def test_reflection_changes_only_lateral_and_turn_commands():
    torch.manual_seed(11)
    actor = MAPPO(20, 2, mirror_equivariant=True)
    obs = torch.randn(7, 2, 20)
    signs = torch.tensor(
        [1, -1, 1, -1, 1, -1, -1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1]
    )
    commands = torch.tensor([1, -1, -1])
    torch.testing.assert_close(
        actor.actor_mean(obs * signs), actor.actor_mean(obs) * commands
    )
    torch.testing.assert_close(actor.distribution(obs).mean, actor.actor_mean(obs))
    np.testing.assert_allclose(
        actor.act(obs.numpy()), actor.actor_mean(obs).tanh().detach().numpy()
    )


def test_mirrored_training_roundtrip_and_resume(tmp_path):
    torch.set_num_threads(1)
    cfg = PushConfig(shape="T", episode_seconds=0.2)
    checkpoint = train(
        cfg,
        None,
        tmp_path / "first",
        iterations=2,
        num_envs=1,
        horizon=4,
        epochs=1,
        minibatch=4,
        seed=10,
        mirror_equivariant=True,
        initial_log_std=-1.5,
    )
    actor, saved = load_checkpoint(checkpoint)
    assert saved["format"] == "hexapod-transport-mappo-v2"
    assert actor.mirror_equivariant
    assert actor.log_std.max().item() < -1
    assert saved["run"]["initial_log_std"] == -1.5
    obs = np.random.default_rng(42).normal(size=(2, 20)).astype(np.float32)
    again, _ = load_checkpoint(checkpoint)
    np.testing.assert_array_equal(actor.act(obs), again.act(obs))
    continuation = train(
        cfg,
        None,
        tmp_path / "resumed",
        iterations=1,
        num_envs=1,
        horizon=4,
        epochs=1,
        minibatch=4,
        seed=11,
        resume=checkpoint,
    )
    resumed, _ = load_checkpoint(continuation)
    assert resumed.mirror_equivariant


def test_legacy_actor_keeps_original_mean():
    actor = MAPPO(20, 2)
    obs = torch.randn(2, 20)
    torch.testing.assert_close(actor.actor_mean(obs), actor.actor(obs), rtol=0, atol=0)
