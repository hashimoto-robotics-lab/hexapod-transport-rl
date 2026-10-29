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
    for name in ("src", "tools", "notebooks", "results"):
        shutil.copytree(
            root / name, remote / name, ignore=shutil.ignore_patterns("__pycache__")
        )
    shutil.copy2(root / "pyproject.toml", remote / "pyproject.toml")
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
    notebook = nbformat.read(root / "notebooks/student_api_colab.ipynb", 4)
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
    assert (result / "my_reward.py").is_file()
    assert (result / "report.md").is_file()
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
    for condition in ("baseline", "progress40"):
        assert (project / "results/reward_study/videos" / f"{condition}.mp4").is_file()
    errors = "\n".join(
        output.get("text", "")
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.outputs
        if output.output_type == "stream" and output.name == "stderr"
    )
    assert "Warning" not in errors, errors
