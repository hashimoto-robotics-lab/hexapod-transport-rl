"""Reward topology, reset invariants, symmetry, and navigation handover."""

import mujoco
import numpy as np
import torch

from hexapod_transport_rl.approach import (
    ApproachConfig,
    ApproachEnv,
    approach_ready,
    cargo_frame,
    observe_approach,
    physical_actions,
    route_distance,
    segment_clear,
)
from hexapod_transport_rl.approach_evaluation import LearnedTransport
from hexapod_transport_rl.config import rotation


def test_reward_credits_going_around_and_penalizes_cutting_through():
    assert not segment_clear(np.array([0.65, 0.65]), np.array([-1.05, 0.325]))
    assert route_distance(np.array([0.65, 0.75])) < route_distance(
        np.array([0.65, 0.65])
    )
    assert route_distance(np.array([0.45, 0.65])) > route_distance(
        np.array([0.55, 0.65])
    )
    assert route_distance(np.array([-1.05, 0.325])) == 0
    assert np.isfinite(route_distance(np.array([0.0, 0.0])))


def test_reproducible_resets_and_local_observation_contract():
    torch.set_num_threads(1)
    env = ApproachEnv(ApproachConfig(layout="front"))
    first, _ = env.reset(seed=54)
    state = env.core.data.qpos.copy()
    second, _ = env.reset(seed=54)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(state, env.core.data.qpos)
    assert env.observation_space.contains(first)
    assert not approach_ready(env.core).any()
    commands = physical_actions(np.ones((2, 3), dtype=np.float32))
    np.testing.assert_array_equal(commands, [[1, -1, -1], [1, 1, 1]])
    obs, reward, _, _, info = env.step(np.zeros((2, 3), dtype=np.float32))
    assert env.observation_space.contains(obs)
    assert np.isfinite(reward)
    assert info["elapsed_seconds"] == 0.2


def test_handover_requires_both_position_and_heading():
    env = ApproachEnv(ApproachConfig(layout="near"))
    env.reset(seed=17)
    for i, address in enumerate(env.core.root_q):
        xy = rotation(env.core.cargo_yaw) @ [-1.05, env.core.cfg.slots[i]]
        yaw = env.core.cargo_yaw
        env.core.data.qpos[address : address + 7] = [
            *xy,
            0.16,
            np.cos(yaw / 2),
            0,
            0,
            np.sin(yaw / 2),
        ]
    mujoco.mj_forward(env.core.model, env.core.data)
    assert approach_ready(env.core).all()
    pos, yaw = cargo_frame(env.core)
    np.testing.assert_allclose(pos, [[-1.05, 0.325], [-1.05, 0.325]], atol=1e-12)
    np.testing.assert_allclose(yaw, 0, atol=1e-12)
    obs = observe_approach(env.core)
    np.testing.assert_allclose(obs[0], obs[1], atol=1e-7)
    q = env.core.root_q[0]
    heading = env.core.cargo_yaw + 0.5
    env.core.data.qpos[q + 3 : q + 7] = [np.cos(heading / 2), 0, 0, np.sin(heading / 2)]
    mujoco.mj_forward(env.core.model, env.core.data)
    assert not approach_ready(env.core)[0]
    assert approach_ready(env.core)[1]


def test_deployment_handover_calls_only_the_two_actors():
    class Actor:
        def __init__(self, value):
            self.value, self.calls = value, 0

        def act(self, obs):
            self.calls += 1
            return np.full((2, 3), self.value, dtype=np.float32)

    env = ApproachEnv(ApproachConfig(layout="front"))
    env.reset(seed=65)
    nav, push = Actor(0.1), Actor(0.2)
    control = LearnedTransport(nav, push)
    np.testing.assert_allclose(control(env.core), [[0.1, -0.1, -0.1], [0.1, 0.1, 0.1]])
    assert (nav.calls, push.calls) == (1, 0)
    for i, q in enumerate(env.core.root_q):
        yaw = env.core.cargo_yaw
        env.core.data.qpos[q : q + 2] = rotation(yaw) @ [-1.05, env.core.cfg.slots[i]]
        env.core.data.qpos[q + 3 : q + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    mujoco.mj_forward(env.core.model, env.core.data)
    control(env.core)
    assert not control.pushing
    np.testing.assert_allclose(control(env.core), 0.2)
    assert control.pushing
    assert (nav.calls, push.calls) == (2, 1)
    control.reset()
    assert not control.pushing
