"""Verify student cells, experiment provenance and resumable lesson training."""

import ast
import importlib.util
import json
import subprocess
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/hexapod_transport_rl_colab.ipynb"
spec = importlib.util.spec_from_file_location(
    "colab_runtime", ROOT / "tools/colab_runtime.py"
)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def test_notebook_uses_direct_simulation_imports():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    walking_simulations = 0
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source, filename=f"cell_{index}")
        compile(tree, f"cell_{index}", "exec")
        # Student cells contain the experiment, not error handling or helper definitions.
        assert not any(
            isinstance(node, (ast.Raise, ast.Assert, ast.Try, ast.FunctionDef))
            for node in ast.walk(tree)
        )
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "WalkingSimulation"
            ):
                walking_simulations += 1
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in ("run_example", "run_python")
    assert walking_simulations == 2
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
    assert NOTEBOOK.stat().st_size < 25_000


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


@pytest.fixture
def lesson(tmp_path, monkeypatch):
    # No installations or simulation libraries are needed to test storage and resume.
    course = runtime.ColabLesson.__new__(runtime.ColabLesson)
    course.project_dir = tmp_path
    course.repository = "hashimoto-robotics-lab/hexapod-transport-rl"
    course.repository_commit = "test-commit"
    course.in_colab = False
    for directory in ("src", "tools"):
        (tmp_path / directory).mkdir()
        (tmp_path / directory / "example.py").write_text("# teaching source\n")
    for name in ("pyproject.toml", "uv.lock", "README.md", "run.sh"):
        (tmp_path / name).write_text("fixture\n")
    monkeypatch.setattr(
        course,
        "_record_runtime",
        lambda: (course.run_dir / "runtime.json").write_text("{}\n"),
    )
    return course


def test_experiment_records_sources_and_prevents_mixing_conditions(lesson):
    lesson.configure()
    config = lesson.run_dir / "experiment.json"
    original = config.read_text()
    settings = json.loads(original)
    assert settings["repository_commit"] == "test-commit"
    assert "repository_ref" not in settings
    assert set(settings["source_sha256"]) == {"src/example.py", "tools/example.py"}
    assert (lesson.run_dir / "source_snapshot/tools/example.py").is_file()
    lesson.configure()
    assert config.read_text() == original
    with pytest.raises(RuntimeError, match="新しい実験名"):
        lesson.configure(seed=17)
    assert config.read_text() == original
    (lesson.project_dir / "tools/example.py").write_text("# edited\n")
    with pytest.raises(RuntimeError, match="新しい実験名"):
        lesson.configure()
    assert config.read_text() == original
    lesson.configure(name="trial_02")
    assert (
        lesson.run_dir / "source_snapshot/tools/example.py"
    ).read_text() == "# edited\n"


@pytest.mark.parametrize(
    "settings",
    [{"num_envs": 0}, {"num_envs": True}, {"name": "../other"}, {"mode": "invalid"}],
)
def test_invalid_experiment_settings_do_not_create_results(lesson, settings):
    with pytest.raises(ValueError):
        lesson.configure(**settings)
    assert not (lesson.project_dir / "runs").exists()


def test_training_resumes_remaining_updates_and_skips_completed_phase(
    lesson, monkeypatch
):
    lesson.configure()
    initial = lesson.run_dir / "base.pt"
    initial.write_text("100")
    saved = lesson.run_dir / "handover/attempt_001/checkpoint.pt"
    saved.parent.mkdir(parents=True)
    saved.write_text("101")
    monkeypatch.setattr(
        lesson, "_checkpoint_iteration", lambda path: int(path.read_text())
    )
    commands = []

    def train(arguments, label):
        commands.append(arguments)
        output = Path(arguments[arguments.index("--output") + 1])
        output.mkdir()
        (output / "checkpoint.pt").write_text("102")
        (output / "metrics.jsonl").write_text('{"transitions_per_second": 50}\n')

    monkeypatch.setattr(lesson, "_run_cli", train)
    result = lesson._train_phase(
        "handover",
        "train-handover",
        initial_checkpoint=initial,
        extra_args=("--initial-log-std", "-1.5"),
    )
    args = commands[0]
    assert args[args.index("--iterations") + 1] == "1"
    assert Path(args[args.index("--checkpoint") + 1]) == saved
    assert "--initial-log-std" not in args
    assert (
        lesson._train_phase("handover", "train-handover", initial_checkpoint=initial)
        == result
    )
    assert len(commands) == 1
