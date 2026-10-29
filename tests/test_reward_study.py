"""Audit the completed study using the original run and evaluation records."""

import csv
import hashlib
import json
import statistics
from pathlib import Path

import pytest
import torch

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
    assert len({row["initial_actor_sha256"] for row in rows}) == len(
        protocol["training_seeds"]
    )
    assert len({row["initial_critic_sha256"] for row in rows}) == len(
        protocol["training_seeds"]
    )
    for row in rows:
        assert row["final_actor_sha256"] != row["initial_actor_sha256"]
        assert row["final_critic_sha256"] != row["initial_critic_sha256"]
    assert len({row["final_actor_sha256"] for row in rows}) == 6
    for condition in protocol["conditions"]:
        group = [row for row in rows if row["condition"] == condition]
        rates, distances = [], []
        for row in group:
            seed = int(row["seed"])
            report = read_json(STUDY / "evaluations" / f"{condition}_s{seed}.json")
            assert report["seed"] == protocol["evaluation"]["seed"]
            assert report["backend"] == "warp"
            assert (
                report["episode_count"]
                == len(report["episodes"])
                == protocol["evaluation"]["episodes"]
            )
            assert report["successes"] == sum(e["success"] for e in report["episodes"])
            rate = report["successes"] / protocol["evaluation"]["episodes"]
            assert float(row["success_rate"]) == rate
            distance = statistics.mean(e["distance_m"] for e in report["episodes"])
            assert float(row["mean_distance_m"]) == pytest.approx(distance)
            assert int(row["trained_steps"]) == protocol["training"]["steps"]
            assert int(row["selected_steps"]) <= int(row["trained_steps"])
            full_started = int(
                protocol["curriculum"]["phases"][-2]["until"]
                * int(row["trained_steps"])
            )
            assert int(row["selected_steps"]) > full_started
            assert (
                int(row["full_task_steps"]) == int(row["trained_steps"]) - full_started
            )
            assert (
                int(row["selected_full_task_steps"])
                == int(row["selected_steps"]) - full_started
            )
            assert int(row["initial_optimizer_entries"]) == 0
            expected_updates = (
                int(row["selected_steps"])
                // protocol["ppo"]["minibatch_size"]
                * protocol["ppo"]["epochs"]
            )
            assert int(row["optimizer_steps"]) == expected_updates
            checkpoint = STUDY / row["checkpoint"]
            assert (
                hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                == row["selected_checkpoint_sha256"]
            )
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            assert saved["training"]["initial_checkpoint"] is None
            assert saved["training"]["training_stage"] == "full_task"
            assert {int(s["step"]) for s in saved["optimizer"]["state"].values()} == {
                expected_updates
            }
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
            assert task["rotation"] == "goal_octant"
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
            assert run["initialization"].startswith("random actor and critic")
            assert run["curriculum"] == protocol["curriculum"]
            assert run["training_stage"] == "full_task"
            assert run["checkpoint_selection"] == "best"
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


def test_final_updated_policies_and_all_seeds_are_reported():
    protocol = read_json(STUDY / "protocol.json")
    with (STUDY / "learning_curve.csv").open() as stream:
        curves = list(csv.DictReader(stream))
    with (STUDY / "runs.csv").open() as stream:
        runs = list(csv.DictReader(stream))

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


def test_standard_condition_meets_the_fixed_stability_thresholds():
    protocol = read_json(STUDY / "protocol.json")
    summary = read_json(STUDY / "summary.json")
    gates = protocol["acceptance"]
    assert (
        summary["conditions"]["baseline"]["mean_success_rate"]
        >= gates["baseline_mean_success_rate_at_least"]
    )
    for seed in protocol["training_seeds"]:
        report = read_json(STUDY / "evaluations" / f"baseline_s{seed}.json")
        assert (
            report["successes"] / report["episode_count"]
            >= gates["every_baseline_seed_success_rate_at_least"]
        )
        assert (
            min(s["success_rate"] for s in report["goal_sectors"])
            >= gates["every_baseline_goal_sector_success_rate_at_least"]
        )
