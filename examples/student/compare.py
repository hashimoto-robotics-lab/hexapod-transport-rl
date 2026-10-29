"""Compare 14 vs 6 inputs and 9 vs 5 actions at equal sample counts."""

import argparse
from dataclasses import replace
from pathlib import Path

from hexapod_transport_rl.student import (
    CylinderPushTask,
    DiscreteDirections,
    ObservationTerm,
    compare,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--backend", choices=["auto", "cpu", "warp"], default="auto")
    args = parser.parse_args()
    baseline = CylinderPushTask()
    options = dict(steps=655_360, backend=args.backend)
    if args.smoke:
        baseline = replace(
            baseline, config=replace(baseline.config, episode_seconds=0.4)
        )
        options.update(
            steps=16,
            num_envs=2,
            horizon=4,
            validation_envs=2,
            validate_every=1,
            backend="cpu" if args.backend == "auto" else args.backend,
        )
    tasks = {
        "baseline": baseline,
        "without_velocity": replace(
            baseline,
            observations=(
                ObservationTerm("cargo_to_robot", scale=baseline.config.width),
                ObservationTerm("cargo_to_goal", scale=baseline.config.width),
                "robot_heading",
            ),
        ),
        "four_directions": replace(baseline, actions=DiscreteDirections(4)),
    }
    result = compare(
        tasks,
        args.output,
        seeds=(42, 43) if args.smoke else (42, 43, 44),
        evaluation_episodes=2 if args.smoke else 256,
        **options,
    )
    print(result["summary"])


if __name__ == "__main__":
    main()
