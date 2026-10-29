"""Colab setup preserves loaded PyTorch and keeps teaching output focused."""

import importlib.util
import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def setup_module():
    path = Path(__file__).resolve().parents[1] / "tools/colab_setup.py"
    spec = importlib.util.spec_from_file_location("colab_setup_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("gpu", [False, True])
def test_setup_preserves_loaded_pytorch(tmp_path, monkeypatch, setup_module, gpu):
    setup = setup_module
    monkeypatch.setattr(torch.cuda, "is_available", lambda: gpu)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "T4")
    monkeypatch.setattr(setup.importlib.util, "find_spec", lambda _: None)
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(setup.subprocess, "run", run)
    original = sys.modules["torch"]
    setup.prepare_colab(tmp_path)
    assert sys.modules["torch"] is original
    assert (
        tmp_path / ".colab-constraints.txt"
    ).read_text() == f"torch=={torch.__version__}\n"
    assert len(commands) == 1
    install = commands[0]
    assert "--index-url" not in install
    assert install[install.index("-e") + 1].endswith(
        "[warp,notebook]" if gpu else "[notebook]"
    )


def test_colab_rendering_setup(tmp_path, monkeypatch, setup_module):
    setup = setup_module
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(setup.importlib.util, "find_spec", lambda _: SimpleNamespace())
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(setup.subprocess, "run", run)
    setup.prepare_colab(tmp_path)
    apt = [command for command in commands if command[0] == "apt-get"]
    assert len(apt) == 2
    assert {"libegl1", "libgl1", "libosmesa6"}.issubset(apt[1])


def test_setup_keeps_student_warnings_visible(tmp_path, setup_module):
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        setup_module.prepare_colab(tmp_path, install=False)
        warnings.warn(
            "`torch.jit.load` is deprecated. Please switch to `torch.export`.",
            FutureWarning,
            stacklevel=2,
        )
        warnings.warn(
            "Student reward returned an unexpected value", RuntimeWarning, stacklevel=2
        )
    assert [str(item.message) for item in observed] == [
        "Student reward returned an unexpected value"
    ]
