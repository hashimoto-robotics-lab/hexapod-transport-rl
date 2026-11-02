"""Check the bundled lesson tables; detailed records remain in Git history."""

import csv
import json
import statistics
from pathlib import Path

import pytest

STUDY = Path(__file__).resolve().parents[1] / "results"


def read_json(path):
    return json.loads(path.read_text())


def read_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def test_recorded_metrics_and_paired_initialization():
    protocol = read_json(STUDY / "protocol.json")
    summary = read_json(STUDY / "summary.json")
    rows = read_rows(STUDY / "runs.csv")
    assert len(rows) == 6
    assert {int(row["seed"]) for row in rows} == set(protocol["training_seeds"])
    assert protocol["conditions"] == {
        "baseline": {"goal_progress": 20.0},
        "progress40": {"goal_progress": 40.0},
    }
    assert protocol["training"]["initial_checkpoint"] is None
    for seed in protocol["training_seeds"]:
        pair = [row for row in rows if int(row["seed"]) == seed]
        assert len(pair) == 2
        for key in ("initial_actor_sha256", "initial_critic_sha256"):
            assert len({row[key] for row in pair}) == 1
    for model in ("actor", "critic"):
        assert len({row[f"initial_{model}_sha256"] for row in rows}) == 3
        assert len({row[f"final_{model}_sha256"] for row in rows}) == 6
        assert all(
            row[f"final_{model}_sha256"] != row[f"initial_{model}_sha256"]
            for row in rows
        )
    for row in rows:
        assert row["checkpoint_url"].startswith(
            "https://raw.githubusercontent.com/hashimoto-robotics-lab/"
            "hexapod-transport-rl/f85f3d754f18fe1efe57d724ffc23697e626c09f/"
            "results/reward_study/checkpoints/"
        )
        assert row["checkpoint_url"].endswith(f"{row['condition']}_s{row['seed']}.pt")
        assert int(row["trained_steps"]) == protocol["training"]["steps"]
        assert int(row["initial_optimizer_entries"]) == 0
        assert float(row["success_rate"]) == int(row["successes"]) / int(
            row["episodes"]
        )
        assert int(row["episodes"]) == protocol["evaluation"]["episodes"]
        full_started = int(
            protocol["curriculum"]["phases"][-2]["until"] * int(row["trained_steps"])
        )
        assert full_started < int(row["selected_steps"]) <= int(row["trained_steps"])
        assert int(row["full_task_steps"]) == int(row["trained_steps"]) - full_started
        expected_updates = (
            int(row["selected_steps"])
            // protocol["ppo"]["minibatch_size"]
            * protocol["ppo"]["epochs"]
        )
        assert int(row["optimizer_steps"]) == expected_updates
    for condition in protocol["conditions"]:
        group = [row for row in rows if row["condition"] == condition]
        rates = [float(row["success_rate"]) for row in group]
        distances = [float(row["mean_distance_m"]) for row in group]
        actual = summary["conditions"][condition]
        assert actual["mean_success_rate"] == pytest.approx(statistics.mean(rates))
        assert actual["std_success_rate"] == pytest.approx(statistics.stdev(rates))
        assert actual["mean_distance_m"] == pytest.approx(statistics.mean(distances))
        assert actual["std_distance_m"] == pytest.approx(statistics.stdev(distances))
    gates = protocol["acceptance"]
    assert (
        summary["conditions"]["baseline"]["mean_success_rate"]
        >= gates["baseline_mean_success_rate_at_least"]
    )
    assert all(
        float(row["success_rate"]) >= gates["every_baseline_seed_success_rate_at_least"]
        for row in rows
        if row["condition"] == "baseline"
    )


def test_all_learning_curves_and_paired_differences_are_reported():
    protocol = read_json(STUDY / "protocol.json")
    curves = read_rows(STUDY / "learning_curve.csv")
    runs = read_rows(STUDY / "runs.csv")
    for run in runs:
        group = [
            row
            for row in curves
            if (row["condition"], row["seed"]) == (run["condition"], run["seed"])
        ]
        rollouts = protocol["training"]["steps"] // (
            protocol["training"]["num_envs"] * protocol["training"]["horizon"]
        )
        assert len(group) == 1 + rollouts // protocol["training"]["validate_every"]
        assert int(group[0]["steps"]) == 0
        assert float(group[0]["validation_success_rate"]) < 0.1
        assert int(group[-1]["steps"]) == int(run["trained_steps"])
        selected = [row for row in group if row["selected"] == "True"]
        assert len(selected) == 1
        assert int(selected[0]["steps"]) == int(run["selected_steps"])
        assert selected[0]["curriculum_stage"] == "full_task"
    summary = read_json(STUDY / "summary.json")
    differences = []
    for seed in protocol["training_seeds"]:
        pair = {
            row["condition"]: float(row["success_rate"])
            for row in runs
            if int(row["seed"]) == seed
        }
        difference = pair["progress40"] - pair["baseline"]
        actual = next(
            row for row in summary["paired_differences"] if row["seed"] == seed
        )
        assert actual["success_rate"] == pytest.approx(difference)
        differences.append(difference)
    assert summary["mean_paired_success_difference"] == pytest.approx(
        statistics.mean(differences)
    )
