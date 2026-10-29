"""Train a student task file, optionally comparing multiple random seeds."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from colab_setup import quiet_dependency_notices


def main():
    quiet_dependency_notices()
    from hexapod_transport_rl import (
        PPOSettings,
        PushCurriculum,
        compare,
        load_task_file,
        train,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["auto", "cpu", "warp"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--steps", type=int, default=655_360)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--validation-envs", type=int)
    parser.add_argument("--validate-every", type=int, default=20)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument(
        "--checkpoint-selection", choices=["best", "latest"], default="best"
    )
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--evaluation-seed", type=int, default=240000)
    parser.add_argument("--video", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Short CPU connection check, not a learning benchmark",
    )
    args = parser.parse_args()
    task = load_task_file(args.task_file)
    if args.smoke:
        if args.initial_checkpoint is not None:
            parser.error("--smoke starts a fresh short task; omit --initial-checkpoint")
        task = replace(task, config=replace(task.config, episode_seconds=0.4))
        if args.backend == "auto":
            args.backend = "cpu"
        args.steps, args.num_envs, args.horizon = 16, 2, 4
        args.validation_envs, args.validate_every, args.episodes = 2, 1, 2
    options = dict(
        steps=args.steps,
        num_envs=args.num_envs,
        horizon=args.horizon,
        max_seconds=args.seconds,
        backend=args.backend,
        validation_envs=args.validation_envs,
        validate_every=args.validate_every,
        initial_checkpoint=args.initial_checkpoint,
        curriculum=PushCurriculum() if args.curriculum else None,
        checkpoint_selection=args.checkpoint_selection,
        settings=PPOSettings(
            learning_rate=args.learning_rate, entropy_coeff=args.entropy_coeff
        ),
    )
    if args.seeds is not None:
        if args.video:
            parser.error("Record a selected checkpoint after the multi-seed comparison")
        result = compare(
            {args.task_file.stem: task},
            args.output,
            seeds=tuple(args.seeds),
            evaluation_episodes=args.episodes,
            evaluation_seed=args.evaluation_seed,
            **options,
        )
        print(json.dumps(dict(smoke=args.smoke, summary=result["summary"]), indent=2))
    else:
        run = train(task, args.output, seed=args.seed, **options)
        report = run.evaluate(
            episodes=args.episodes, seed=args.evaluation_seed, backend=args.backend
        )
        result = dict(
            smoke=args.smoke,
            checkpoint=str(run.checkpoint),
            successes=report["successes"],
            episodes=report["episode_count"],
            mean_distance_m=report["mean_distance_m"],
            policy_input_dim=report["policy_input_dim"],
        )
        if args.video:
            result["video"] = run.video()
        (run.directory / "example_run.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
