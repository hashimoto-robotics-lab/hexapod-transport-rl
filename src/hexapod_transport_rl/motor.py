"""Match sixtrail_flat_rl's XM430 voltage loop and native MuJoCo motor.

Constants and conventions are from src/sixtrail_flat_rl/{xm430,actuator}.py.
Position targets stay fixed for 40 ms; voltage feedback updates every 5 ms.
"""

import mujoco
import numpy as np

SUPPLY_VOLTAGE = 11.5
VOLTAGE_GAIN = (800 / 128) * (4096 / (2 * np.pi)) * SUPPLY_VOLTAGE / 885


def voltage_command(target: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Position feedback: target/q are radians; output is clipped motor volts."""
    return np.clip(VOLTAGE_GAIN * (target - q), -SUPPLY_VOLTAGE, SUPPLY_VOLTAGE)


def add_motor(spec: mujoco.MjSpec, joint_name: str) -> None:
    """Attach the source model's XM430 actuator in native voltage-input mode."""
    actuator = spec.add_actuator(name=joint_name, target=joint_name)
    actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
    actuator.gear[0] = 1.0
    actuator.set_to_dcmotor(
        motorconst=[0.0, 0.0],
        resistance=0.0,
        nominal=[12.0, 4.1, 46 * 2 * np.pi / 60],
        saturation=[0.0] * 3,
        controller=[0.0] * 6,
        cogging=[0.0] * 3,
        inductance=[0.0] * 2,
        thermal=[0.0] * 6,
        lugre=[0.0] * 5,
        input_mode=0,
    )
