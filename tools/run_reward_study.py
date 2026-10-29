"""Reproduce the two conditions with the protocol used in the student lesson."""

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

from colab_setup import quiet_dependency_notices


def model_hash(checkpoint, key="actor"):
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)[key]
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    quiet_dependency_notices()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "examples/student"))
    import torch
    from reward_study import EVALUATION, SEEDS, TASKS, TRAINING

    from hexapod_transport_rl import train

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=TASKS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()
    for seed in args.seeds:
        output = args.output / args.condition / f"seed_{seed}"
        started = time.perf_counter()
        run = train(
            TASKS[args.condition], output, seed=seed, backend="warp", **TRAINING
        )
        wall_seconds = time.perf_counter() - started
        report = run.evaluate(backend="warp", **EVALUATION)
        metadata = json.loads((run.directory / "run.json").read_text())
        selected = torch.load(run.checkpoint, map_location="cpu", weights_only=True)
        result = dict(
            condition=args.condition,
            seed=seed,
            trained_steps=metadata["steps"],
            selected_steps=selected["training"]["steps"],
            wall_training_seconds=wall_seconds,
            successes=report["successes"],
            episodes=report["episode_count"],
            success_rate=report["successes"] / report["episode_count"],
            mean_distance_m=report["mean_distance_m"],
            gpu=torch.cuda.get_device_name(0),
            python=platform.python_version(),
            torch=str(torch.__version__),
            initial_actor_sha256=model_hash(run.directory / "initial.pt"),
            initial_critic_sha256=model_hash(run.directory / "initial.pt", "critic"),
            selected_checkpoint_sha256=hashlib.sha256(
                run.checkpoint.read_bytes()
            ).hexdigest(),
        )
        (run.directory / "study_result.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
