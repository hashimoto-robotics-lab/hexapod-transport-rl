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
    assert NOTEBOOK.stat().st_size < 20_000


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


def test_reward_parameters_flow_directly_to_one_fresh_training_run():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )
    assert "checkpoint = train_mappo(" in source
    assert "config, RUN_DIR" in source
    assert "max_seconds=MAX_TRAINING_SECONDS" in source
    assert "report = evaluate_mappo(checkpoint" in source
    assert "train_approach(" not in source
    assert "reward_weights=reward" in source
    assert "seed=TRAINING_SEED" in source
    assert "seed=98000" in source
    assert "PoseCurriculum(" not in source
    assert "pushing_checkpoint=" not in source
    assert 'gym.make("HexapodPushT-v0"' in source
    assert "media.show_video(media.read_video(report" in source
    assert "COMMON_CHECKPOINT" not in source
    assert "unique=True" in source
    assert "MAX_TRAINING_SECONDS = 360" in source
    assert len(notebook.cells) <= 20
    # The compact notebook calls a normal API; TorchRL owns the actual updates.
    training = (ROOT / "src/hexapod_transport_rl/training.py").read_text()
    assert "from torchrl.collectors import Collector" in training
    assert "loss.value_estimator(batch)" in training
    assert "buffer.empty()" in training
    assert "optimizer.step()" in training
    assert "load_mappo(" not in training


def test_student_reward_experiment_distinguishes_study_terms_from_official_baseline():
    from dataclasses import asdict

    from hexapod_transport_rl import PushTConfig, PushTRewardWeights

    notebook = nbformat.read(NOTEBOOK, as_version=4)
    cell = next(
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code" and "reward = PushTRewardWeights(" in cell.source
    )
    namespace = {"PushTConfig": PushTConfig, "PushTRewardWeights": PushTRewardWeights}
    exec(cell, namespace)
    baseline = asdict(PushTRewardWeights())
    assert baseline["coverage"] == 1
    assert all(value == 0 for key, value in baseline.items() if key != "coverage")
    config = namespace["config"]
    assert config.reward_weights.coverage == 0
    assert config.reward_weights.progress == 10
    assert config.reward_weights.approach == 1
    assert config.coverage_threshold == 0.95
    assert config.goal_pose == PushTConfig().goal_pose
    assert config.robot_start == "goal_side"
    assert config.observation_dim == 9
