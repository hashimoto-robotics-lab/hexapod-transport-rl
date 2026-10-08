"""The reward distance cannot reward a shortcut through the cargo."""

import numpy as np
import pytest
import torch

from hexapod_transport_rl import PosePushConfig
from hexapod_transport_rl.push_approach import PushApproachDistance


@pytest.mark.parametrize("num_robots", [2, 3, 4])
@pytest.mark.parametrize("clearance", [0.16, 0.4])
def test_obstacle_distance_is_finite_and_matches_euclidean_on_rear_side(
    num_robots, clearance
):
    cfg = PosePushConfig(num_robots=num_robots, approach_clearance=clearance)
    metric = PushApproachDistance(cfg)
    np.testing.assert_allclose(metric(metric.targets), 0, atol=1e-12)
    rear = metric.targets - [0.5, 0]
    np.testing.assert_allclose(metric(rear), 0.5)
    rng = np.random.default_rng(20)
    points = rng.uniform(-3, 3, (1000, num_robots, 2))
    distance = metric(points)
    assert np.isfinite(distance).all()
    assert np.all(distance + 1e-4 >= np.linalg.norm(points - metric.targets, axis=-1))


def test_going_around_bar_is_rewarded_at_the_front_wall():
    metric = PushApproachDistance(PosePushConfig())
    front = np.array([[0.4, -0.4], [0.4, 0.4]])
    outside = np.array([[0.4, -0.7], [0.4, 0.7]])
    assert np.all(metric(outside) < metric(front))
    # Direct distance incorrectly prefers staying next to the assigned Y slot.
    assert np.all(
        np.linalg.norm(outside - metric.targets, axis=-1)
        > np.linalg.norm(front - metric.targets, axis=-1)
    )
    edge = np.array([[0.26, -0.4], [0.26, 0.4]])
    np.testing.assert_allclose(
        metric(edge + [1e-4, 0]),
        metric(edge - [1e-4, 0]),
        atol=2.1e-4 * metric.penetration_cost,
    )


def test_entering_clearance_buffer_cannot_shorten_the_reward_distance():
    metric = PushApproachDistance(PosePushConfig())
    outside = np.array([[0, -1.125], [0, 1.125]])
    inside = np.array([[0, -0.7], [0, 0.7]])
    assert np.all(metric(inside) > metric(outside))
    # The two closest exits exchange order here. Reward stays continuous.
    boundary_switch = np.array([[0.12, -0.7], [0.12, 0.7]])
    np.testing.assert_allclose(
        metric(boundary_switch + [1e-5, 0]),
        metric(boundary_switch - [1e-5, 0]),
        atol=3e-4,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("clearance", [0.16, 0.4])
def test_cuda_distance_matches_native_for_all_regions(clearance):
    cfg = PosePushConfig(num_robots=4, approach_clearance=clearance)
    cpu = PushApproachDistance(cfg)
    gpu = PushApproachDistance(cfg, device="cuda")
    rng = np.random.default_rng(32)
    points = rng.uniform(-3, 3, (1000, 4, 2)).astype(np.float32)
    np.testing.assert_allclose(
        gpu(torch.as_tensor(points, device="cuda")).cpu(),
        cpu(points),
        atol=2e-5,
        rtol=1e-5,
    )
