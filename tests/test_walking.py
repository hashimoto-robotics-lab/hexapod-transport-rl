"""Exercise physical command semantics independently of transport rewards."""

import mujoco
import numpy as np
import pytest
import torch

from hexapod_transport_rl import WalkingSimulation


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_scene_contains_only_requested_robots_and_commands_do_not_move_time(count):
    with WalkingSimulation(num_robots=count) as sim:
        assert sim.positions.shape == (count, 2)
        assert sim.model.body_mass.sum() == pytest.approx(3 * count)
        assert mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "cargo") == -1
        assert mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "goal") == -1
        sim.set_velocity(robot_id=count - 1, vx=0.12)
        assert sim.time == 0
        assert sim.velocities[count - 1, 0] == pytest.approx(0.12)
        sim.run_for(seconds=0.8)
        assert sim.time == pytest.approx(0.8)
        sim.reset()
        assert sim.time == 0
        np.testing.assert_array_equal(sim.velocities, np.zeros((count, 3)))


def test_four_robots_retain_independent_commands_and_stop_one_at_a_time():
    with WalkingSimulation(num_robots=4) as sim:
        sim.run_for(seconds=0.8)
        start = sim.positions
        sim.set_velocity(robot_id=0, vx=0.12)
        sim.set_velocity(robot_id=1, vx=-0.08)
        sim.set_velocity(robot_id=2, vx=0.08, vy=0.06)
        sim.set_velocity(robot_id=3, yaw_rate=0.4)
        sim.run_for(seconds=2.4)
        change = sim.positions - start
        assert change[0, 0] > 0.2
        assert change[1, 0] < -0.1
        assert change[2, 1] > 0.1
        assert sim.headings[3] > 0.6
        sim.stop(robot_id=0)
        np.testing.assert_array_equal(sim.velocities[0], [0, 0, 0])
        assert sim.velocities[1, 0] < 0  # Other commands persist.
        sim.stop()
        sim.run_for(seconds=0.8)
        np.testing.assert_array_equal(sim.velocities, np.zeros((4, 3)))


def test_commands_are_body_relative_and_omitted_components_become_zero():
    with WalkingSimulation() as sim:
        sim.reset(poses=[[0, 0, np.pi / 2]])
        sim.run_for(seconds=0.8)
        start = sim.positions
        sim.set_velocity(vy=0.06, yaw_rate=0.2)
        sim.set_velocity(vx=0.12)
        np.testing.assert_allclose(sim.velocities, [[0.12, 0, 0]])
        sim.run_for(seconds=2.4)
        assert sim.positions[0, 1] - start[0, 1] > 0.2
        assert abs(sim.positions[0, 0] - start[0, 0]) < 0.04
        positions = sim.positions
        positions[:] = 999
        assert not np.any(sim.positions == 999)


def test_invalid_inputs_do_not_change_state_and_closed_simulation_rejects_commands():
    with WalkingSimulation() as sim:
        sim.set_velocity(vx=0.1)
        state = sim.data.qpos.copy()
        for velocity in (
            dict(vx=0.21),
            dict(vx=-0.16),
            dict(vy=np.nan),
            dict(yaw_rate=0.7),
        ):
            with pytest.raises(ValueError):
                sim.set_velocity(**velocity)
        for robot_id in (-1, 1, True):
            with pytest.raises(ValueError):
                sim.set_velocity(robot_id=robot_id)
        for seconds in (0, -1, np.inf, 0.03):
            with pytest.raises(ValueError):
                sim.run_for(seconds=seconds)
        with pytest.raises(ValueError):
            sim.reset(poses=[[0, 0]])
        assert sim.time == 0
        np.testing.assert_array_equal(sim.data.qpos, state)
        np.testing.assert_allclose(sim.velocities, [[0.1, 0, 0]])
    sim.close()
    with pytest.raises(RuntimeError, match="closed"):
        sim.set_velocity(vx=0.1)
