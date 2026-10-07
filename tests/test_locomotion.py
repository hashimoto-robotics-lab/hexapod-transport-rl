"""Integration regressions for the hexapod-specific deployment contract."""

import mujoco
import numpy as np
import pytest
import torch

from hexapod_transport_rl import PushEnv
from hexapod_transport_rl.config import (
    COMMAND_HIGH,
    COMMAND_LOW,
    CONTROL_DT,
    POLICY_RELATIVE_PATH,
    STAND,
    action_to_command,
    command_to_action,
)
from hexapod_transport_rl.motor import SUPPLY_VOLTAGE, voltage_command


@pytest.fixture(scope="module")
def env():
    torch.set_num_threads(1)
    return PushEnv()


def test_asymmetric_commands_keep_zero_and_limits():
    np.testing.assert_allclose(action_to_command(np.ones(3)), COMMAND_HIGH)
    np.testing.assert_allclose(action_to_command(-np.ones(3)), COMMAND_LOW)
    np.testing.assert_array_equal(action_to_command(np.zeros(3)), np.zeros(3))
    commands = np.array([[0.12, -0.05, 0.3], [-0.08, 0.02, -0.6]])
    np.testing.assert_allclose(
        action_to_command(command_to_action(commands)), commands, atol=1e-7
    )


def test_imu_axes_and_exported_actor_equivalence(env):
    env.reset()
    # Tilt and rotate each robot independently; gyro must remain in IMU axes.
    for i, root in enumerate(env.root_q):
        quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(quat, np.array([1.0, 0.0, 0.0]), 0.25 * (i + 1))
        env.data.qpos[root + 3 : root + 7] = quat
        env.data.qvel[env.root_v[i] + 3 : env.root_v[i] + 6] = [0.1, 0.2, 0.3]
    mujoco.mj_forward(env.model, env.data)
    commands = np.array([[0.1, 0, 0.2], [-0.1, 0.05, 0]], dtype=np.float32)
    q, qdot, prev, command, gyro, gravity = env.locomotion_inputs(commands)
    for i, body_id in enumerate(env.base_ids):
        np.testing.assert_allclose(gyro[i], [0.2, -0.1, 0.3], atol=1e-6)
        np.testing.assert_allclose(
            gravity[i], env.data.xmat[body_id].reshape(3, 3).T @ [0, 0, -1], atol=1e-7
        )
    obs = torch.cat(
        [
            gyro,
            gravity,
            q - torch.tensor(STAND, dtype=torch.float32),
            qdot,
            prev,
            command,
        ],
        dim=-1,
    )
    actor = torch.jit.load(
        str(env.asset_root / POLICY_RELATIVE_PATH / "actor.ts")
    ).eval()
    with torch.inference_mode():
        expected = actor(obs).clamp(-3, 3)
        targets, actions = env.controller(q, qdot, prev, command, gyro, gravity)
    torch.testing.assert_close(actions, expected)
    torch.testing.assert_close(
        targets, torch.tensor(STAND, dtype=torch.float32) + 0.3 * expected
    )


def test_motor_voltage_and_control_timing(env):
    env.reset()
    assert (
        np.max(np.abs(voltage_command(np.full((2, 18), 10.0), np.zeros((2, 18)))))
        == SUPPLY_VOLTAGE
    )
    ticks = []
    env.step(
        np.array([[0.5, 0, 0], [0, 0, 0]]),
        on_control_step=lambda: ticks.append(env.data.time),
    )
    np.testing.assert_allclose(
        ticks, CONTROL_DT * np.arange(1, env.cfg.high_level_decimation + 1)
    )
    assert np.max(np.abs(env.data.ctrl)) <= SUPPLY_VOLTAGE
    assert np.all(env.model.dof_armature[env.joint_v] == 0.025)


def test_matches_original_walking_model(env):
    # Optional when using a standalone CPU install; the source project's venv
    # includes mjlab and its canonical robot loader, used for this comparison.
    robot = pytest.importorskip("sixtrail_flat_rl.robot")
    original = robot.load_model()
    for i in range(env.cfg.num_robots):
        for attr in (
            "actuator_gainprm",
            "actuator_biasprm",
            "actuator_dynprm",
            "actuator_gaintype",
            "actuator_gear",
        ):
            np.testing.assert_allclose(
                getattr(env.model, attr)[env.actuators[i]],
                getattr(original, attr),
                atol=1e-10,
            )
        for b in range(1, original.nbody):
            source = original.body(b)
            actual = env.model.body(f"r{i}/{source.name}")
            for attr in ("mass", "inertia", "ipos", "iquat"):
                np.testing.assert_allclose(
                    getattr(actual, attr), getattr(source, attr), atol=1e-8
                )
    assert env.model.body_mass.sum() == pytest.approx(
        3 * env.cfg.num_robots + env.cfg.cargo_mass
    )
