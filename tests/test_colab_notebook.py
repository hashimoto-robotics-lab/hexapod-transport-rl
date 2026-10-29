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
    monkeypatch.setenv("HEXAPOD_PROJECT_DIR", str(project))
    monkeypatch.setenv("HEXAPOD_REPOSITORY", remote.as_uri())
    monkeypatch.setenv("HEXAPOD_REVISION", "main")
    monkeypatch.setenv("HEXAPOD_SKIP_INSTALL", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("MUJOCO_GL", os.environ.get("MUJOCO_GL", "egl"))
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    notebook = nbformat.read(root / "notebooks/student_api_colab.ipynb", 4)
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
    outputs = list((project / "runs").glob("tutorial_*/comparison/comparison.csv"))
    assert len(outputs) == 1
    # Four independent changes, each with two seeds in the teaching profile.
    import csv

    with outputs[0].open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 8
    assert {row["condition"] for row in rows} == {
        "baseline",
        "reward",
        "observation",
        "action",
    }
    assert all(int(row["episodes"]) == 4 for row in rows)
    result = outputs[0].parents[1]
    assert (result / "baseline.mp4").stat().st_size > 1000
    assert (result / "reference.mp4").stat().st_size > 1000
    assert (result / "custom/task.json").is_file()
    assert list((result / "custom/task_sources").glob("*.py"))
    manifest = json.loads((project / "checkpoints/manifest.json").read_text())
    assert (
        hashlib.sha256(
            (project / "checkpoints/cylinder_push.pt").read_bytes()
        ).hexdigest()
        == manifest["sha256"]
    )
