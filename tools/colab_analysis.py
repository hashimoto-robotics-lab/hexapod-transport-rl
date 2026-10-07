"""Generate learning curves and evaluation tables in the simulation environment."""

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

run = Path(sys.argv[1])
fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
for column, phase in enumerate(("navigation", "handover")):
    by_iteration = {}
    for log in sorted((run / phase).glob("attempt_*/metrics.jsonl")):
        for line in log.read_text().splitlines():
            row = json.loads(line)
            by_iteration[row["iteration"]] = row
    records = [by_iteration[key] for key in sorted(by_iteration)]
    settings = json.loads((run / "experiment.json").read_text())
    first = records[0]["transitions"] - settings["num_envs"] * settings["horizon"]
    steps = [row["transitions"] - first for row in records]
    axes[0, column].plot(
        steps,
        [row["mean_reward"] for row in records],
        marker=".",
        label="Training reward",
    )
    key = "training_success_rate" if phase == "navigation" else "success_rate"
    measured_success = [
        row[key] if row["completed_episodes"] else float("nan") for row in records
    ]
    axes[1, column].plot(steps, measured_success, marker=".", label="Training success")
    validated = [row for row in records if "validation_success_rate" in row]
    if validated:
        axes[1, column].plot(
            [row["transitions"] - first for row in validated],
            [row["validation_success_rate"] for row in validated],
            marker="o",
            label="Validation success",
        )
    for axis in axes[:, column]:
        axis.set_title(phase)
        axis.set_xlabel("Team transitions since phase start")
        axis.grid(alpha=0.3)
        axis.legend()
    axes[0, column].set_ylabel("Mean reward per step")
    axes[1, column].set_ylabel("Episode success rate")
    axes[1, column].set_ylim(-0.05, 1.05)
    for row in records:
        if "advanced_to" in row:
            for axis in axes[:, column]:
                axis.axvline(row["transitions"] - first, color="gray", linestyle=":")
fig.savefig(run / "learning_curves.png", dpi=180)
fig.savefig(run / "learning_curves.pdf")
plt.close(fig)

rows = []
for layout in ("front", "side"):
    report = json.loads((run / f"test_{layout}.json").read_text())
    episodes = report["episodes"]
    count = len(episodes)
    proportion = report["successes"] / count
    z = 1.96
    denominator = 1 + z * z / count
    center = (proportion + z * z / (2 * count)) / denominator
    half = (
        z
        * math.sqrt(proportion * (1 - proportion) / count + z * z / (4 * count * count))
        / denominator
    )
    rows.append(
        dict(
            layout=layout,
            mode=settings["mode"],
            training_seed=settings["training_seed"],
            episodes=count,
            success_rate=proportion,
            success_ci95_low=center - half,
            success_ci95_high=center + half,
            handover_rate=report["handovers"] / count,
            body_contact_episodes=report["body_contact_episodes"],
            robot_contact_episodes=report["robot_contact_episodes"],
            navigation_robot_contact_episodes=report[
                "navigation_robot_contact_episodes"
            ],
            falls=report["falls"],
            mean_distance_m=report["mean_final_distance_m"],
            mean_yaw_error_rad=sum(row["yaw_error"] for row in episodes) / count,
            mean_elapsed_seconds=sum(row["elapsed_seconds"] for row in episodes)
            / count,
        )
    )
with (run / "evaluation.csv").open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(run / "evaluation_table.json").write_text(json.dumps(rows, indent=2) + "\n")
