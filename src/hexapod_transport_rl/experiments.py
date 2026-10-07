"""Small storage helpers; no simulation, training or model selection is hidden here."""

import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def create_experiment(project_dir: str | Path, name: str) -> Path:
    """Create a new results directory and snapshot the exact research sources/assets."""
    project_dir = Path(project_dir).resolve()
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("Use a single directory name for the experiment")
    output = project_dir / "runs" / name
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    sources = {}
    for directory in ("src", "tools", "notebooks", "checkpoints"):
        for source in sorted((project_dir / directory).rglob("*")):
            if not source.is_file() or "__pycache__" in source.parts:
                continue
            relative = source.relative_to(project_dir)
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            sources[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    for name in ("pyproject.toml", "uv.lock", "README.md"):
        shutil.copy2(project_dir / name, snapshot / name)
    commit = subprocess.check_output(
        ["git", "-C", str(project_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    record = dict(
        repository_commit=commit,
        source_sha256=sources,
        python=platform.python_version(),
        executable=sys.executable,
        packages={
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "torch",
                "mujoco",
                "gymnasium",
                "mediapy",
                "pandas",
                "torchrl",
                "tensordict",
            )
        },
    )
    (output / "experiment.json").write_text(json.dumps(record, indent=2) + "\n")
    return output


def archive_results(output: str | Path) -> Path:
    """ZIP the completed experiment, including weights and its source snapshot."""
    output = Path(output).resolve()
    return Path(shutil.make_archive(str(output), "zip", root_dir=output))
