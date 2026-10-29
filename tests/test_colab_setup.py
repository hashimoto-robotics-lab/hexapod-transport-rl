"""Changed Colab Torch builds must choose the tested wheels and request restart."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("gpu", [False, True])
def test_changed_runtime_pins_torch_and_requests_kernel_restart(
    tmp_path, monkeypatch, gpu
):
    path = Path(__file__).resolve().parents[1] / "tools/colab_setup.py"
    spec = importlib.util.spec_from_file_location("colab_setup_test", path)
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)
    monkeypatch.setattr(
        setup.shutil, "which", lambda _: "/usr/bin/nvidia-smi" if gpu else None
    )
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if command[0].endswith("nvidia-smi"):
            return subprocess.CompletedProcess(command, 0, "Tesla T4\n", "")
        if "-c" in command:
            # A future Colab default differs from the teaching environment.
            return subprocess.CompletedProcess(
                command, 0, '["2.14.0+cu130", "13.0", true]', ""
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(setup.subprocess, "run", run)
    assert sys.modules["torch"] is torch
    with pytest.raises(RuntimeError, match="セッションを再起動"):
        setup.prepare_colab(tmp_path)
    torch_install = next(c for c in commands if "torch==2.11.0" in c)
    assert (
        torch_install[-1]
        == f"https://download.pytorch.org/whl/{'cu128' if gpu else 'cpu'}"
    )
    project_install = next(c for c in commands if "-e" in c)
    assert project_install[project_install.index("-e") + 1].endswith(
        "[warp,notebook]" if gpu else "[notebook]"
    )
    assert (tmp_path / ".colab-constraints.txt").read_text() == "torch==2.11.0\n"


def test_colab_rendering_setup_does_not_require_upload_import(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "tools/colab_setup.py"
    spec = importlib.util.spec_from_file_location("colab_setup_render_test", path)
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    monkeypatch.setattr(setup.shutil, "which", lambda _: None)
    monkeypatch.setattr(
        setup.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace() if name == "google.colab" else None,
    )
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        output = '["2.11.0+cpu", null, false]' if "-c" in command else ""
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(setup.subprocess, "run", run)
    setup.prepare_colab(tmp_path)
    apt = [command for command in commands if command[0] == "apt-get"]
    assert len(apt) == 2
    assert {"libegl1", "libgl1", "libosmesa6"}.issubset(apt[1])
