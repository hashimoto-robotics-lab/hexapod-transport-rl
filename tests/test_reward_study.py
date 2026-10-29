"""Audit the completed study using the original run and evaluation records."""

import csv
import hashlib
import json
import statistics
from pathlib import Path

import pytest

STUDY = Path(__file__).resolve().parents[1] / "results/reward_study"


def read_json(path):
    return json.loads(path.read_text())


def test_completed_study_metrics_and_paired_initialization():
    protocol = read_json(STUDY / "protocol.json")
    summary = read_json(STUDY / "summary.json")
    with (STUDY / "runs.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6
    assert {int(row["seed"]) for row in rows} == set(protocol["training_seeds"])
    for seed in protocol["training_seeds"]:
        pair = [row for row in rows if int(row["seed"]) == seed]
        assert len(pair) == 2
        for key in ("initial_actor_sha256", "initial_critic_sha256"):
            assert len({row[key] for row in pair}) == 1
    assert len({row["initial_actor_sha256"] for row in rows}) == 3
    for condition in protocol["conditions"]:
        group = [row for row in rows if row["condition"] == condition]
        rates, distances = [], []
        for row in group:
            seed = int(row["seed"])
            report = read_json(STUDY / "evaluations" / f"{condition}_s{seed}.json")
            assert report["seed"] == protocol["evaluation"]["seed"]
            assert report["backend"] == "warp"
            assert report["episode_count"] == len(report["episodes"]) == 256
            assert report["successes"] == sum(e["success"] for e in report["episodes"])
            rate = report["successes"] / 256
            assert float(row["success_rate"]) == rate
            distance = statistics.mean(e["distance_m"] for e in report["episodes"])
            assert float(row["mean_distance_m"]) == pytest.approx(distance)
            assert int(row["trained_steps"]) == protocol["training"]["steps"]
            assert 0 <= int(row["selected_steps"]) <= int(row["trained_steps"])
            checkpoint = STUDY / row["checkpoint"]
            assert (
                hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                == row["selected_checkpoint_sha256"]
            )
            rates.append(rate)
            distances.append(distance)
        actual = summary["conditions"][condition]
        assert actual["mean_success_rate"] == pytest.approx(statistics.mean(rates))
        assert actual["std_success_rate"] == pytest.approx(statistics.stdev(rates))
        assert actual["mean_distance_m"] == pytest.approx(statistics.mean(distances))


def test_completed_study_changes_one_reward_and_uses_common_layouts():
    protocol = read_json(STUDY / "protocol.json")
    reference_layouts = None
    original = None
    for condition in ("baseline", "progress40"):
        for seed in (42, 43, 44):
            task = read_json(STUDY / "metadata" / f"{condition}_s{seed}_task.json")
            if original is None:
                original = task
            assert task["config"] == original["config"]
            assert task["actions"] == original["actions"]
            assert task["observations"] == original["observations"]
            assert task["rotation"] == "none"
            differences = {
                name
                for name in task["rewards"]
                if task["rewards"][name] != original["rewards"][name]
            }
            assert differences == (
                {"goal_progress"} if condition == "progress40" else set()
            )
            assert task["rewards"]["goal_progress"]["weight"] == (
                40 if condition == "progress40" else 20
            )
            run = read_json(STUDY / "metadata" / f"{condition}_s{seed}_run.json")
            assert run["settings"] == protocol["ppo"]
            assert run["seed"] == seed
            for key in ("num_envs", "horizon", "validation_envs"):
                assert run[key] == protocol["training"][key]
            assert run["backend"] == protocol["evaluation"]["backend"]
            assert run["initial_checkpoint"] is None
            assert run["stopped_by"] == "step_budget"
            episodes = read_json(STUDY / "evaluations" / f"{condition}_s{seed}.json")[
                "episodes"
            ]
            layouts = [
                [
                    e[name]
                    for name in (
                        "initial_azimuth_degrees",
                        "initial_radius_m",
                        "initial_goal_azimuth_degrees",
                        "initial_goal_radius_m",
                        "initial_yaw_degrees",
                    )
                ]
                for e in episodes
            ]
            if reference_layouts is None:
                reference_layouts = layouts
            for expected, actual in zip(reference_layouts, layouts, strict=True):
                assert actual == pytest.approx(expected, abs=2e-5)


def test_selected_checkpoints_use_validation_and_all_seeds_are_reported():
    with (STUDY / "learning_curve.csv").open() as stream:
        curves = list(csv.DictReader(stream))
    with (STUDY / "runs.csv").open() as stream:
        runs = list(csv.DictReader(stream))

    def score(row):
        return tuple(
            float(row[key])
            for key in (
                "worst_goal_rate",
                "worst_pose_rate",
                "worst_heading_rate",
                "worst_position_rate",
                "validation_successes",
            )
        ) + (-float(row["validation_distance_m"]),)

    for run in runs:
        group = [
            row
            for row in curves
            if (row["condition"], row["seed"]) == (run["condition"], run["seed"])
        ]
        assert len(group) == 17
        assert int(group[0]["steps"]) == 0
        assert int(group[-1]["steps"]) == int(run["trained_steps"])
        selected = [row for row in group if row["selected"] == "True"]
        assert len(selected) == 1
        assert int(selected[0]["steps"]) == int(run["selected_steps"])
        assert score(selected[0]) == max(score(row) for row in group)
    summary = read_json(STUDY / "summary.json")
    differences = []
    for seed in (42, 43, 44):
        pair = {
            r["condition"]: float(r["success_rate"])
            for r in runs
            if int(r["seed"]) == seed
        }
        difference = pair["progress40"] - pair["baseline"]
        actual = next(r for r in summary["paired_differences"] if r["seed"] == seed)
        assert actual["success_rate"] == pytest.approx(difference)
        differences.append(difference)
    assert summary["mean_paired_success_difference"] == pytest.approx(
        statistics.mean(differences)
    )
