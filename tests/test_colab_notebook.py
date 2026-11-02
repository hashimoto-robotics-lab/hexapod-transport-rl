"""Clone a published snapshot and execute the full lesson on CPU or GPU."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import nbformat
import pytest
import torch
from jupyter_client import KernelManager
from nbclient import NotebookClient


@pytest.mark.parametrize(
    ("backend", "optional_experiments"), [("cpu", False), ("cpu", True), ("warp", True)]
)
def test_cloned_notebook_runs_all_steps(
    tmp_path, monkeypatch, backend, optional_experiments
):
    if backend == "warp" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    root = Path(__file__).resolve().parents[1]
    # Publish the working files to a local Git remote. GitHub need not contain
    # uncommitted changes for this test to exercise the actual clone path.
    remote = tmp_path / "remote"
    remote.mkdir()
    for name in ("src", "results"):
        shutil.copytree(
            root / name, remote / name, ignore=shutil.ignore_patterns("__pycache__")
        )
    for name in ("pyproject.toml", "colab_setup.py", "cylinder_pushing.ipynb"):
        shutil.copy2(root / name, remote / name)
    for args in (
        ["init", "--initial-branch=main"],
        ["add", "."],
        [
            "-c",
            "user.name=Notebook Test",
            "-c",
            "user.email=notebook@example.invalid",
            "-c",
            "commit.gpgSign=false",
            "commit",
            "-m",
            "Publish test materials",
        ],
    ):
        subprocess.run(
            ["git", "-C", str(remote), *args], check=True, capture_output=True
        )
    project = tmp_path / "hexapod_transport_rl"
    if backend == "cpu":
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("MUJOCO_GL", os.environ.get("MUJOCO_GL", "egl"))
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    notebook = nbformat.read(root / "cylinder_pushing.ipynb", 4)
    settings = next(c for c in notebook.cells if 'PROFILE = "research"' in c.source)
    assert "initial_checkpoint=None" in settings.source
    # Only test budgets and optional exercises change. Production starts a full
    # fresh run; this test executes every API path without waiting for training.
    settings.source = settings.source.replace(
        'PROFILE = "research"', 'PROFILE = "demo"'
    )
    for flag in (
        "RUN_REWARD_EXPERIMENT",
        "RUN_MULTISEED_STUDY",
        "RUN_CUSTOM_REWARD",
        "RUN_INPUT_EXPERIMENT",
    ):
        flag_cell = next(c for c in notebook.cells if f"{flag} = False" in c.source)
        if optional_experiments:
            flag_cell.source = flag_cell.source.replace(
                f"{flag} = False", f"{flag} = True"
            )
    setup = next(c for c in notebook.cells if "setup" in c.metadata.get("tags", []))
    setup.source = (
        setup.source.replace(
            '"https://github.com/hashimoto-robotics-lab/hexapod-transport-rl.git"',
            repr(remote.as_uri()),
        )
        .replace('Path("/content/hexapod_transport_rl")', f"Path({str(project)!r})")
        .replace(
            "prepare_colab(PROJECT_DIR)", "prepare_colab(PROJECT_DIR, install=False)"
        )
    )
    # Keep machine paths, images and results out of the distributed source.
    assert all(
        c.execution_count is None and not c.outputs
        for c in notebook.cells
        if c.cell_type == "code"
    )
    # Check the values produced by the introductory API exercises, before the
    # later experiments reuse any variables. These checks stay out of the lesson.
    first_experiment = next(
        i
        for i, cell in enumerate(notebook.cells)
        if cell.source.startswith("## Step 2 ")
    )
    notebook.cells.insert(
        first_experiment,
        nbformat.v4.new_code_cell(
            """
assert task.actions.action_count == 9
assert td["observation"].shape == (1, 14)
assert torch.allclose(td["observation"][:, :2], before.cargo_to_robot / task.config.width)
assert torch.allclose(td["observation"][:, 2:4], before.cargo_to_goal / task.config.width)
assert torch.allclose(td["observation"][:, 11:13], before.robot_heading)
assert action_table.shape == (9, 2)
assert transition["action"].tolist() == [action_id]
assert next_td["observation"].shape == (1, 14)
expected_command = torch.tensor([task.actions.directions[action_id]]) * task.actions.speed
assert torch.allclose(after.command_velocity, expected_command)
assert torch.allclose(after.elapsed_seconds - before.elapsed_seconds, torch.tensor([task.config.dt]))
assert torch.allclose(next_td["reward"], sum(next_td["reward_terms"].values()))
assert next_td["done"].shape == (1, 1)
assert small_td["observation"].shape == (1, 6)
assert torch.allclose(small_td["observation"], td["observation"][:, [0, 1, 2, 3, 11, 12]])
assert [(row["start"], row["stop"]) for row in small_env.observation_layout] == [(0, 2), (2, 4), (4, 6)]
assert env.is_closed and small_env.is_closed
""".strip()
        ),
    )
    # Exercise the new reward function before spending time training it. Both
    # sides of the distance threshold must retain the world's batch dimension.
    custom_preview = next(
        i for i, cell in enumerate(notebook.cells) if cell.id == "b3ae2e0f"
    )
    notebook.cells.insert(
        custom_preview + 1,
        nbformat.v4.new_code_cell(
            """
assert custom_contribution.shape == (1,)
assert torch.allclose(custom_contribution, torch.tensor([-0.004]))
far_state = replace(example_state, goal_distance=example_state.goal_distance.new_tensor([0.15]))
assert torch.equal(custom_task.rewards["near_goal_speed"].compute(before, far_state), torch.zeros(1))
assert example_state is not after
""".strip()
        ),
    )
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    try:
        NotebookClient(
            notebook,
            km=manager,
            timeout=300,
            resources={"metadata": {"path": str(tmp_path)}},
        ).execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    assert (project / ".git").is_dir()
    outputs = list((project / "runs").glob("tutorial_*/comparison_*/comparison.csv"))
    assert len(outputs) == int(optional_experiments)
    if optional_experiments:
        # Every condition is actually trained with two seeds.
        import csv

        with outputs[0].open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 4
        assert {row["condition"] for row in rows} == {"baseline", "changed"}
        assert {int(row["seed"]) for row in rows} == {42, 43}
        assert all(int(row["trained_steps"]) == 80 for row in rows)
        assert all(int(row["episodes"]) == 4 for row in rows)
    result = next(p for p in (project / "runs").glob("tutorial_*") if p.is_dir())
    assert (result / "baseline.mp4").stat().st_size > 1000
    if optional_experiments:
        assert (result / "reward_weights.mp4").stat().st_size > 1000
    assert (result / "student_learning_curves.png").stat().st_size > 1000
    assert (result / "student_paired_results.png").stat().st_size > 1000
    assert (result / "student_results.csv").is_file()
    assert (result / "student_summary.csv").is_file()
    assert (result / "my_reward.py").is_file()
    assert (result / "report.md").is_file()
    report_text = (result / "report.md").read_text()
    if optional_experiments:
        assert "標準と変更条件を実行済み" in report_text
        input_study = next(result.glob("input_comparison_*/comparison.csv"))
        import csv

        with input_study.open() as stream:
            input_rows = list(csv.DictReader(stream))
        assert len(input_rows) == 4
        assert {row["condition"] for row in input_rows} == {"reference", "modified"}
        for condition, dimensions in (("reference", 14), ("modified", 6)):
            task_json = input_study.parent / condition / "seed_42" / "task.json"
            input_config = json.loads(task_json.read_text())
            assert input_config["rotation"] == "none"
            assert (
                sum(row["size"] for row in input_config["observation_layout"])
                == dimensions
            )
    else:
        assert "比較未実施（標準条件のみ）" in report_text
        assert "未計算（学習1回）" in (result / "student_summary.csv").read_text()
    plan = json.loads((result / "experiment_plan.json").read_text())
    assert plan["changed_parameter"] == "goal_progress"
    assert plan["evaluation_seed"] == 940000
    assert plan["initial_checkpoint"] is None
    assert plan["training_seeds"] == [42, 43]
    baseline = next(result.glob("baseline_*/task.json"))
    task_files = [baseline]
    if optional_experiments:
        weighted = next(result.glob("reward_weights_*/task.json"))
        custom = next(result.glob("custom_reward_*/task.json"))
        original = json.loads(baseline.read_text())
        changed = json.loads(weighted.read_text())
        custom_data = json.loads(custom.read_text())
        assert changed["rewards"]["goal_progress"]["weight"] == 40.0
        assert "near_goal_speed" in custom_data["rewards"]
        for task in (changed, custom_data):
            assert task["observations"] == original["observations"]
            assert task["actions"] == original["actions"]
        assert list((custom.parent / "task_sources").glob("*.py"))
        task_files += [weighted, custom]
    else:
        assert len(list(result.glob("*/run.json"))) == 1
    for task_json in task_files:
        progress = json.loads((task_json.parent / "progress.json").read_text())
        assert progress[-1]["steps"] == 80
        assert progress[-1]["curriculum_stage"] == "full_task"
        assert (task_json.parent / f"evaluation_{backend}_s940000.json").is_file()
        metadata = json.loads((task_json.parent / "run.json").read_text())
        assert metadata["backend"] == backend
        assert metadata["initial_checkpoint"] is None
        assert metadata["initialization"].startswith("random actor and critic")
        initial = torch.load(
            task_json.parent / "initial.pt", map_location="cpu", weights_only=True
        )
        final = torch.load(
            task_json.parent / "latest.pt", map_location="cpu", weights_only=True
        )
        assert not initial["optimizer"]["state"]
        assert final["optimizer"]["state"]
        for model in ("actor", "critic"):
            assert any(
                not torch.equal(initial[model][key], final[model][key])
                for key in initial[model]
            )
        assert metadata["checkpoint_selection"] == "best"
        assert metadata["config"]["policy_rotation_mode"] == "goal_octant"
    completed = next(
        c for c in notebook.cells if "completed_results" in c.metadata.get("tags", [])
    )
    assert any(o.output_type == "display_data" for o in completed.outputs)
    reset_cell = next(
        c for c in notebook.cells if "api_reset" in c.metadata.get("tags", [])
    )
    assert any("image/png" in output.get("data", {}) for output in reset_cell.outputs)
    for condition in ("baseline", "progress40"):
        assert (project / "results/videos" / f"{condition}.mp4").is_file()
    errors = "\n".join(
        output.get("text", "")
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.outputs
        if output.output_type == "stream" and output.name == "stderr"
    )
    assert "Warning" not in errors, errors
