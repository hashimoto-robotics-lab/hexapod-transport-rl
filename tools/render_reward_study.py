"""Render the two preselected lesson videos using native CPU MuJoCo."""

import argparse
import json
from pathlib import Path

from colab_setup import quiet_dependency_notices


def draw_frames(study, conditions):
    import imageio.v2 as imageio
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(conditions), 4, figsize=(12, 3 * len(conditions)), squeeze=False
    )
    for row, condition in enumerate(conditions):
        with imageio.get_reader(study / "videos" / f"{condition}.mp4") as reader:
            fps = reader.get_meta_data()["fps"]
            last = reader.count_frames() - 1
            for column, fraction in enumerate((0, 1 / 3, 2 / 3, 1)):
                index = round(fraction * last)
                ax = axes[row, column]
                ax.imshow(reader.get_data(index))
                ax.set_title(f"{condition}, {index / fps:.1f} s", fontsize=10)
                ax.axis("off")
    fig.tight_layout()
    fig.savefig(study / "video_frames.png", dpi=130)
    plt.close(fig)


def main():
    quiet_dependency_notices()
    from hexapod_transport_rl import record_video

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    args = parser.parse_args()
    protocol = json.loads((args.study / "protocol.json").read_text())
    video = protocol["video"]
    for condition in protocol["conditions"]:
        name = f"{condition}_s{video['training_seed']}.pt"
        checkpoint = args.study / "checkpoints" / name
        path = args.study / "videos" / f"{condition}.mp4"
        result = record_video(checkpoint, path, seed=video["episode_seed"])
        # Portable paths keep the published metadata useful after cloning.
        result.update(
            checkpoint=f"../checkpoints/{name}",
            video=path.name,
            backend="cpu",
            training_seed=video["training_seed"],
        )
        path.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    draw_frames(args.study, protocol["conditions"])


if __name__ == "__main__":
    main()
