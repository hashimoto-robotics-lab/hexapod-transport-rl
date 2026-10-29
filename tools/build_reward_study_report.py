"""Publish compact, auditable data and figures from the completed reward study."""

import argparse
import hashlib
import json
import shutil
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"baseline": "#2767b0", "progress40": "#d66a23"}
LABELS = {"baseline": "Baseline (20)", "progress40": "Progress weight (40)"}


def draw_task(path):
    from matplotlib.patches import Circle, Ellipse

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    cargo = np.array([0.0, 0.0])
    goal = np.array([0.45, 0.12])
    robot = np.array([0.48, -0.65])
    ax.add_patch(Circle(cargo, 0.25, color="#e7b13a", alpha=0.85))
    ax.add_patch(
        Circle(goal, 0.25, edgecolor="#39985a", facecolor="#39985a", alpha=0.2)
    )
    ax.add_patch(Circle(goal, 0.05, fill=False, color="#24824b", lw=1))
    ax.plot(*goal, marker="+", color="#24824b", ms=7)
    ax.add_patch(
        Circle(cargo, 0.65, fill=False, linestyle="--", color="#777777", alpha=0.5)
    )
    ax.add_patch(
        Circle(cargo, 1.0, fill=False, linestyle="--", color="#777777", alpha=0.5)
    )
    ax.add_patch(Ellipse(robot, 0.22, 0.34, angle=-35, color="#465367"))
    for side in (-1, 1):
        for offset in (-0.11, 0, 0.11):
            ax.plot(
                [robot[0] + side * 0.07, robot[0] + side * 0.24],
                [robot[1] + offset, robot[1] + offset - 0.05],
                color="#465367",
                lw=2,
            )
    ax.annotate(
        "", goal, cargo, arrowprops=dict(arrowstyle="->", color="#24824b", lw=2)
    )
    ax.annotate(
        "", robot, cargo, arrowprops=dict(arrowstyle="->", color="#2767b0", lw=2)
    )
    ax.annotate("cargo_to_goal", (-0.12, 0.42), fontsize=10, color="#24824b")
    ax.annotate("cargo_to_robot", (0.28, -0.43), fontsize=10, color="#2767b0")
    ax.text(-0.2, 0.02, "Cylinder", fontsize=11)
    ax.text(goal[0] + 0.08, goal[1] + 0.2, "Goal", color="#24824b", fontsize=11)
    ax.text(robot[0] + 0.08, robot[1] - 0.25, "Hexapod", fontsize=10)
    fig.text(
        0.5,
        0.02,
        "Spawn radius: 0.65-1.00 m. Robot yaw and goal direction: random.",
        ha="center",
        fontsize=9,
    )
    ax.set(
        xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), xlabel="World X [m]", ylabel="World Y [m]"
    )
    ax.set_aspect("equal")
    ax.grid(alpha=0.15)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def draw_curves(curves, records, output):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for condition, group in curves.groupby("condition", sort=False):
        for seed, seed_rows in group.groupby("seed"):
            selected_steps = records.loc[
                (records.condition == condition) & (records.seed == seed),
                "selected_steps",
            ].iloc[0]
            selected = seed_rows[seed_rows.steps == selected_steps]
            for ax, metric in zip(
                axes, ("validation_success_rate", "validation_distance_m"), strict=True
            ):
                ax.plot(
                    seed_rows.steps / 1e6,
                    seed_rows[metric],
                    color=COLORS[condition],
                    alpha=0.22,
                    lw=0.8,
                )
                ax.scatter(
                    selected.steps / 1e6,
                    selected[metric],
                    color=COLORS[condition],
                    s=26,
                    edgecolors="white",
                    linewidths=0.6,
                    zorder=4,
                )
        grouped = group.groupby("steps")
        for ax, metric in zip(
            axes, ("validation_success_rate", "validation_distance_m"), strict=True
        ):
            mean, spread = grouped[metric].mean(), grouped[metric].std()
            x = mean.index.to_numpy() / 1e6
            low = (mean - spread).clip(lower=0)
            high = mean + spread
            if metric == "validation_success_rate":
                high = high.clip(upper=1)
            ax.plot(x, mean, label=LABELS[condition], color=COLORS[condition], lw=2)
            ax.fill_between(x, low, high, alpha=0.13, color=COLORS[condition])
    axes[0].set(
        ylabel="Validation success rate",
        ylim=(0, 1.02),
    )
    axes[1].set(ylabel="Validation final goal distance [m]")
    for ax in axes:
        ax.set_xlabel("Training samples [million]")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)
    fig.suptitle(
        "Validation: mean and SD over 3 seeds; dots = reported checkpoints",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7))
    for ax, metric, label in zip(
        axes,
        ("success_rate", "mean_distance_m"),
        ("Held-out success rate", "Held-out final goal distance [m]"),
        strict=True,
    ):
        for seed, rows in records.groupby("seed"):
            pair = rows.set_index("condition").loc[["baseline", "progress40"]]
            ax.plot([0, 1], pair[metric], marker="o", alpha=0.8, label=f"Seed {seed}")
        ax.set(
            xticks=[0, 1], xticklabels=["Baseline (20)", "Progress (40)"], ylabel=label
        )
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=9)
    axes[0].set_ylim(
        max(0, records.success_rate.min() - 0.03),
        min(1, max(0.1, records.success_rate.max() * 1.4)),
    )
    fig.tight_layout()
    fig.savefig(output / "paired_results.png", dpi=170)
    plt.close(fig)


def build(source, output):
    output.mkdir(parents=True, exist_ok=True)
    for name in ("checkpoints", "evaluations", "metadata", "videos"):
        (output / name).mkdir(exist_ok=True)
    protocol = json.loads((source / "protocol.json").read_text())
    rows, curves, updates = [], [], []
    source_hashes = None
    for condition in protocol["conditions"]:
        for seed in protocol["training_seeds"]:
            run = source / "main" / condition / f"seed_{seed}"
            snapshot = run / "source_snapshot/hexapod_transport_rl"
            hashes = {
                file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                for file in sorted(snapshot.glob("*.py"))
            }
            if source_hashes is not None:
                assert hashes == source_hashes
            source_hashes = hashes
            result = json.loads((run / "study_result.json").read_text())
            row = {**result, "checkpoint": f"checkpoints/{condition}_s{seed}.pt"}
            evaluation_name = f"evaluation_warp_s{protocol['evaluation']['seed']}.json"
            evaluation = json.loads((run / evaluation_name).read_text())
            row["mean_episode_seconds"] = statistics.mean(
                episode["seconds"] for episode in evaluation["episodes"]
            )
            rows.append(row)
            shutil.copy2(
                run / protocol["report_checkpoint"], output / row["checkpoint"]
            )
            for name in (
                "run.json",
                "task.json",
                "environment.json",
                "study_result.json",
            ):
                shutil.copy2(
                    run / name, output / "metadata" / f"{condition}_s{seed}_{name}"
                )
            shutil.copy2(
                run / evaluation_name,
                output / "evaluations" / f"{condition}_s{seed}.json",
            )
            initial = json.loads((run / "initial_evaluation.json").read_text())
            training_rows = json.loads((run / "progress.json").read_text())
            stages = {
                entry["steps"]: entry.get("curriculum_stage", "fixed_task")
                for entry in training_rows
            }
            row["initial_validation_success_rate"] = (
                initial["successes"] / initial["episode_count"]
            )
            validations = [(0, initial)]
            validations += [
                (
                    r["rollout"]
                    * protocol["training"]["num_envs"]
                    * protocol["training"]["horizon"],
                    r["report"],
                )
                for r in json.loads((run / "validations.json").read_text())
            ]
            for steps, report in validations:
                if steps == row["selected_steps"]:
                    row["selected_validation_success_rate"] = (
                        report["successes"] / report["episode_count"]
                    )
                curves.append(
                    dict(
                        condition=condition,
                        seed=seed,
                        steps=steps,
                        curriculum_stage=stages.get(steps, "initial"),
                        validation_success_rate=report["successes"]
                        / report["episode_count"],
                        validation_distance_m=report["mean_distance_m"],
                        validation_successes=report["successes"],
                        validation_episodes=report["episode_count"],
                        worst_goal_rate=min(
                            s["success_rate"] or 0 for s in report["goal_sectors"]
                        ),
                        worst_pose_rate=min(
                            s["success_rate"] or 0 for s in report["pose_cells"]
                        ),
                        worst_heading_rate=min(
                            s["success_rate"] or 0 for s in report["heading_sectors"]
                        ),
                        worst_position_rate=min(
                            s["success_rate"] or 0 for s in report["angle_sectors"]
                        ),
                        selected=steps == row["selected_steps"],
                    )
                )
            for update in training_rows:
                flat = {
                    key: value for key, value in update.items() if key != "reward_terms"
                }
                flat.update(
                    {
                        f"reward_{name}": value
                        for name, value in update["reward_terms"].items()
                    }
                )
                updates.append(dict(condition=condition, seed=seed, **flat))

    for seed in protocol["training_seeds"]:
        pair = [row for row in rows if row["seed"] == seed]
        assert len({row["initial_actor_sha256"] for row in pair}) == 1
        assert len({row["initial_critic_sha256"] for row in pair}) == 1
    frame, curve_frame = pd.DataFrame(rows), pd.DataFrame(curves)
    frame.to_csv(output / "runs.csv", index=False)
    curve_frame.to_csv(output / "learning_curve.csv", index=False)
    pd.DataFrame(updates).to_csv(output / "training.csv", index=False)
    summary = {"conditions": {}, "paired_differences": [], "protocol": "protocol.json"}
    for condition, group in frame.groupby("condition", sort=False):
        summary["conditions"][condition] = dict(
            seeds=len(group),
            successes=int(group.successes.sum()),
            episodes=int(group.episodes.sum()),
            mean_success_rate=float(group.success_rate.mean()),
            std_success_rate=float(group.success_rate.std()),
            mean_distance_m=float(group.mean_distance_m.mean()),
            std_distance_m=float(group.mean_distance_m.std()),
            mean_training_seconds=float(group.wall_training_seconds.mean()),
            mean_episode_seconds=float(group.mean_episode_seconds.mean()),
        )
    for seed, group in frame.groupby("seed"):
        pair = group.set_index("condition")
        summary["paired_differences"].append(
            dict(
                seed=int(seed),
                success_rate=float(
                    pair.loc["progress40", "success_rate"]
                    - pair.loc["baseline", "success_rate"]
                ),
                distance_m=float(
                    pair.loc["progress40", "mean_distance_m"]
                    - pair.loc["baseline", "mean_distance_m"]
                ),
            )
        )
    differences = [row["success_rate"] for row in summary["paired_differences"]]
    summary["mean_paired_success_difference"] = statistics.mean(differences)
    summary["std_paired_success_difference"] = statistics.stdev(differences)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "source.json").write_text(
        json.dumps(
            dict(source_commit=protocol["source_commit"], package_files=source_hashes),
            indent=2,
        )
        + "\n"
    )
    shutil.copy2(source / "protocol.json", output / "protocol.json")
    draw_task(output / "task.png")
    draw_curves(curve_frame, frame, output)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output), indent=2))
