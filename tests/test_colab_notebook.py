"""Validate readable notebook cells and the public-repository bootstrap."""

import ast
import subprocess
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/hexapod_transport_rl_colab.ipynb"


def test_notebook_and_nested_worker_scripts_compile():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    workers = 0
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source, filename=f"cell_{index}")
        compile(tree, f"cell_{index}", "exec")
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in ("run_python", "run_example")
            ):
                compile(ast.literal_eval(node.args[0]), f"worker_{index}", "exec")
                workers += 1
    assert workers == 4


def bootstrap_namespace():
    namespace = {}
    exec((ROOT / "tools/colab_checkout.py").read_text(), namespace)
    return namespace


def test_notebook_fetches_configured_repository_without_binary_payload():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)
    assert "hashimoto-robotics-lab/hexapod-transport-rl" in source
    assert "REPOSITORY_COMMIT" in source
    assert "RESOURCE_ARCHIVE" not in source
    assert "BUNDLE_SHA256" not in source
    assert "base64" not in source
    assert "getpass" not in source
    assert "HEXAPOD_GIT_TOKEN" not in source
    assert "PROJECT_SUBDIR" not in source
    assert NOTEBOOK.stat().st_size < 100_000
    assert (ROOT / "tools/colab_checkout.py").read_text().strip() in source


@pytest.fixture
def local_repository(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(
        ["git", "init", "--quiet", "--initial-branch=main", str(source)], check=True
    )
    (source / "pyproject.toml").write_text(
        '[project]\nname = "test-teaching-project"\n'
    )
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
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    # Exercise real git fetch/checkout while redirecting GitHub to a local fixture.
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", f"url.{source}.insteadOf")
    monkeypatch.setenv(
        "GIT_CONFIG_VALUE_0", "https://github.com/classroom/transport.git"
    )
    return source, commit


@pytest.mark.parametrize("use_commit", [False, True])
def test_public_checkout_records_commit_and_preserves_student_edits(
    tmp_path, monkeypatch, local_repository, use_commit
):
    namespace = bootstrap_namespace()
    # Even with an inherited prompt, public checkout must run unattended.
    askpass = tmp_path / "unexpected-prompt.sh"
    askpass.write_text("#!/bin/sh\nexit 1\n")
    askpass.chmod(0o700)
    monkeypatch.setenv("GIT_ASKPASS", str(askpass))
    _, expected_commit = local_repository
    ref = expected_commit if use_commit else "main"
    destination = tmp_path / "student"
    project, commit = namespace["checkout_repository"](
        "classroom/transport", ref, destination
    )
    assert project == destination
    assert commit == expected_commit
    (project / "pyproject.toml").write_text("student edits\n")
    reused, second_commit = namespace["checkout_repository"](
        "classroom/transport", ref, destination
    )
    assert reused == project and second_commit == commit
    assert (project / "pyproject.toml").read_text() == "student edits\n"
    with pytest.raises(RuntimeError, match="別の教材"):
        namespace["checkout_repository"](
            "classroom/transport", "another-ref", destination
        )


def test_failed_fetch_leaves_no_partial_checkout(tmp_path, monkeypatch):
    namespace = bootstrap_namespace()
    original_run = subprocess.run

    def reject_fetch(arguments, **kwargs):
        if "fetch" in arguments:
            raise subprocess.CalledProcessError(128, arguments, stderr="unknown ref")
        return original_run(arguments, **kwargs)

    monkeypatch.setattr(subprocess, "run", reject_fetch)
    with pytest.raises(RuntimeError, match="教材を取得できません"):
        namespace["checkout_repository"](
            "classroom/transport", "main", tmp_path / "student"
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "repository,ref",
    [
        ("https://example.org/classroom/transport", "main"),
        ("classroom/transport", "--upload-pack=anything"),
    ],
)
def test_invalid_checkout_input_is_rejected(tmp_path, repository, ref):
    namespace = bootstrap_namespace()
    with pytest.raises(ValueError):
        namespace["checkout_repository"](repository, ref, tmp_path / "student")
