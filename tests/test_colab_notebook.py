"""The student notebook exposes experiments as ordinary Python API calls."""

import ast
import subprocess
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/hexapod_transport_rl_colab.ipynb"


def test_notebook_uses_direct_simulation_imports():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    walking_environments = 0
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source, filename=f"cell_{index}")
        compile(tree, f"cell_{index}", "exec")
        # Student cells contain the experiment, not error handling or helper definitions.
        assert not any(
            isinstance(
                node,
                (
                    ast.Raise,
                    ast.Assert,
                    ast.Try,
                    ast.FunctionDef,
                    ast.With,
                    ast.AsyncWith,
                ),
            )
            for node in ast.walk(tree)
        )
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "make"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "HexapodWalking-v0"
            ):
                walking_environments += 1
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in ("run_example", "run_python")
    assert walking_environments == 2
    assert all(
        cell.execution_count is None and cell.outputs == []
        for cell in notebook.cells
        if cell.cell_type == "code"
    )


def test_student_notebook_excludes_infrastructure_settings_and_payloads():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)
    assert "hashimoto-robotics-lab/hexapod-transport-rl" in source
    for unnecessary in (
        "GIT_REF",
        "ColabLesson",
        "lesson.",
        "ValueError",
        "RuntimeError",
        "RESOURCE_ARCHIVE",
        "base64",
        "getpass",
        "HEXAPOD_GIT_TOKEN",
        "checkpoint_iteration",
        "Popen",
        "IPython.display import Video",
    ):
        assert unnecessary not in source
    assert NOTEBOOK.stat().st_size < 35_000


def test_public_checkout_preserves_student_edits(tmp_path, monkeypatch):
    source = tmp_path / "source"
    subprocess.run(
        ["git", "init", "--quiet", "--initial-branch=main", str(source)], check=True
    )
    (source / "student.py").write_text("original\n")
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "Fixture",
        ],
        check=True,
    )
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", f"url.{source}.insteadOf")
    monkeypatch.setenv(
        "GIT_CONFIG_VALUE_0",
        "https://github.com/hashimoto-robotics-lab/hexapod-transport-rl.git",
    )
    destination = tmp_path / "student"
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    cell = next(cell.source for cell in notebook.cells if cell.cell_type == "code")
    cell = cell.replace("/content/hexapod_transport_rl", str(destination))
    namespace = {}
    exec(cell, namespace)
    expected = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    assert (
        subprocess.check_output(
            ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True
        ).strip()
        == expected
    )
    (destination / "student.py").write_text("student edits\n")
    exec(cell, namespace)
    assert (destination / "student.py").read_text() == "student edits\n"


def test_reward_parameters_flow_directly_to_training_and_paired_evaluation():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )
    assert "config=config" in source
    assert "from torchrl.collectors import Collector" in source
    assert "total_frames=TRAINING_STEPS" in source
    assert "make_mappo_loss(actor, critic, settings)" in source
    assert "buffer.empty()" in source
    assert "save_mappo(" in source
    assert "train_approach(" not in source
    assert "reward_weights=changed_reward" in source
    assert "seed=TRAINING_SEED" in source
    assert "seed=TEST_SEED" in source
    assert "PoseCurriculum(" in source
    assert 'policy in ("forward", "feedback")' in source
    assert "pushing_checkpoint=" not in source
    assert 'gym.make("HexapodPosePush-v0"' in source
    assert "comparison.to_csv" in source
    assert "media.show_videos" in source
    assert 'torch.device("cuda" if torch.cuda.is_available() else "cpu")' in source
    assert "actor.to(DEVICE)" in source and "critic.to(DEVICE)" in source
    assert "batch = batch.to(DEVICE)" in source
    assert "LazyTensorStorage(FRAMES_PER_BATCH, device=DEVICE)" in source
    assert "policy_device=envs.device" in source
    assert 'backend="auto"' in source
    assert "NUM_ENVS = 64" in source
    assert "collector.update_policy_weights_()" in source
