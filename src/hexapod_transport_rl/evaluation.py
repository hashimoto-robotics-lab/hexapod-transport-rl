"""Small experiment-facing evaluation API; simulation stays in the Gym evaluator."""

import json
import time
from pathlib import Path

import torch

from .pose_evaluation import evaluate_pose


def evaluate_mappo(checkpoint, *, episodes=8, seed=98000, video=True, backend="auto"):
    """Evaluate the saved task and optionally record one real-time 25 fps movie.

    Numeric evaluation uses Warp with CUDA. Recording always uses native
    MuJoCo for the first of the same reset seeds. The two backends can differ.
    """
    started = time.monotonic()
    checkpoint = Path(checkpoint)
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    if saved.get("format") == "hexapod-navigation-torchrl-mappo-v1":
        from .navigation_evaluation import evaluate_navigation

        return evaluate_navigation(
            checkpoint, episodes=episodes, seed=seed, video=video, backend=backend
        )
    output = checkpoint.parent
    if backend == "auto":
        backend = "warp" if torch.cuda.is_available() else "cpu"
    report = evaluate_pose(
        checkpoint,
        episodes=episodes,
        seed=seed,
        workers=1,
        backend=backend,
        output=output / "evaluation.json",
    )
    if video:
        native = evaluate_pose(
            checkpoint,
            episodes=1,
            seed=seed,
            workers=1,
            video_dir=output / "video",
            video_width=384,
            video_height=288,
            output=output / "video_evaluation.json",
        )
        report["video_path"] = str(output / "video" / f"seed_{seed}.mp4")
        report["video_backend"] = "cpu"
        report["video_success"] = bool(native["successes"])
        report["video_seconds"] = native["evaluation_seconds"]
    report["evaluation_and_video_seconds"] = time.monotonic() - started
    (output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
