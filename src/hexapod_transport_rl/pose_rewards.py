"""One reward formula shared by native MuJoCo and the GPU pose environment.

``math`` is NumPy for one Gym world, or Torch for batched GPU tensors. Inputs
stay on their original device. Only the terminal success/failure removes the
next potential; time limits retain it for value bootstrapping.
"""


def pose_reward_terms(
    config,
    before,
    after,
    commands,
    previous_commands,
    command_limits,
    body_contact_ticks,
    robot_contact_ticks,
    position_tolerance,
    yaw_tolerance,
    *,
    math,
):
    weights = config.reward_weights
    terminal = after["success"] | after["failed"]
    delta = {
        key: before[key] / 0.5
        - config.shaping_discount * math.where(terminal, 0.0, after[key] / 0.5)
        for key in ("distance", "yaw_error", "approach")
    }
    near = math.exp(
        -((after["distance"] / (0.75 * position_tolerance)) ** 2)
        - (after["yaw_error"] / (0.75 * yaw_tolerance)) ** 2
    )
    motion = (
        after["cargo_speed"] / command_limits[0]
        + after["cargo_yaw_speed"] / command_limits[2]
    )
    ticks = config.high_level_decimation * 8
    return dict(
        position=weights.position * delta["distance"],
        position_error=-weights.position_error * config.dt * after["distance"] / 0.5,
        orientation=weights.orientation * delta["yaw_error"],
        approach=weights.approach * delta["approach"],
        settling=-weights.settling * config.dt * near * motion,
        command_change=-weights.command_change
        * (((commands - previous_commands) / command_limits) ** 2).mean(axis=(-2, -1)),
        time=after["distance"] * 0 - weights.time * config.dt,
        robot_contact=-weights.robot_contact * config.dt * robot_contact_ticks / ticks,
        body_contact=-weights.body_contact
        * config.dt
        * body_contact_ticks.mean(axis=-1)
        / ticks,
        terminal=math.where(
            after["success"],
            weights.success,
            math.where(after["failed"], weights.failure, 0.0),
        ),
    )
