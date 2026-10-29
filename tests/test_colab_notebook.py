"""Clone a published snapshot and execute the notebook with installed CPU deps."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient


def test_cloned_notebook_runs_all_steps(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    # Publish the working files to a local Git remote. GitHub need not contain
    # uncommitted changes for this test to exercise the actual clone path.
    remote = tmp_path / "remote"
    remote.mkdir()
    for name in ("src", "tools", "notebooks", "checkpoints"):
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
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("MUJOCO_GL", os.environ.get("MUJOCO_GL", "egl"))
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    notebook = nbformat.read(root / "notebooks/student_api_colab.ipynb", 4)
    notebook.cells[2].source = (
        notebook.cells[2]
        .source.replace(
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
    assert len(outputs) == 1
    # Each independent change is actually trained with two seeds.
    import csv

    with outputs[0].open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 10
    assert {row["condition"] for row in rows} == {
        "baseline",
        "reward_weights",
        "custom_reward",
        "observation",
        "action",
    }
    assert all(int(row["episodes"]) == 4 for row in rows)
    result = outputs[0].parents[1]
    assert (result / "baseline.mp4").stat().st_size > 1000
    assert (result / "reward_weights.mp4").stat().st_size > 1000
    assert (result / "custom_reward.mp4").stat().st_size > 1000
    assert (result / "reference.mp4").stat().st_size > 1000
    weighted = next(result.glob("reward_weights_*/task.json"))
    custom = next(result.glob("custom_reward_*/task.json"))
    baseline = next(result.glob("baseline_*/task.json"))
    original = json.loads(baseline.read_text())
    changed = json.loads(weighted.read_text())
    custom_data = json.loads(custom.read_text())
    assert changed["rewards"]["goal_progress"]["weight"] == 40.0
    assert "near_goal_speed" in custom_data["rewards"]
    for task in (changed, custom_data):
        assert task["observations"] == original["observations"]
        assert task["actions"] == original["actions"]
    assert list((custom.parent / "task_sources").glob("*.py"))
    for task_json in (baseline, weighted, custom):
        progress = json.loads((task_json.parent / "progress.json").read_text())
        assert progress[-1]["steps"] == 32
    errors = "\n".join(
        output.get("text", "")
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.outputs
        if output.output_type == "stream" and output.name == "stderr"
    )
    assert "Warning" not in errors, errors
    manifest = json.loads((project / "checkpoints/manifest.json").read_text())
    assert (
        hashlib.sha256(
            (project / "checkpoints/cylinder_push.pt").read_bytes()
        ).hexdigest()
        == manifest["sha256"]
    )
