"""Install the teaching environment before importing simulation libraries."""

import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def prepare_colab(project_dir, *, install=True):
    """Pin the tested Torch/CUDA combination; keep CPU notebooks usable too."""
    project_dir = Path(project_dir).resolve()
    if not (3, 12) <= sys.version_info[:2] < (3, 14):
        raise RuntimeError(
            "Python 3.12/3.13が必要です。Colabのランタイムのバージョンを確認してください。"
        )
    gpu = False
    if shutil.which("nvidia-smi"):
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        gpu = probe.returncode == 0 and bool(probe.stdout.strip())
    if install:
        # Do not import torch here: installation must precede kernel imports.
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                "import json,torch; print(json.dumps([torch.__version__,torch.version.cuda,torch.cuda.is_available()]))",
            ],
            capture_output=True,
            text=True,
        )
        version, cuda, available = (
            json.loads(probe.stdout) if probe.returncode == 0 else ("", None, False)
        )
        replace_torch = version.split("+")[0] != "2.11.0" or (
            gpu and (cuda != "12.8" or not available)
        )
        if replace_torch:
            index = "cu128" if gpu else "cpu"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-q",
                    "torch==2.11.0",
                    "torchvision==0.26.0",
                    "torchaudio==2.11.0",
                    "--index-url",
                    f"https://download.pytorch.org/whl/{index}",
                ],
                check=True,
            )
        constraints = project_dir / ".colab-constraints.txt"
        constraints.write_text("torch==2.11.0\n")
        extras = "warp,notebook" if gpu else "notebook"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "-e",
                f"{project_dir}[{extras}]",
                "-c",
                str(constraints),
            ],
            check=True,
        )
        if replace_torch and "torch" in sys.modules:
            raise RuntimeError(
                "インストールが完了しました。ランタイム→セッションを再起動し、このセルから再実行してください。"
            )
    # Set this before importing MuJoCo/PyOpenGL, including CPU video rendering.
    backend = os.environ.setdefault("MUJOCO_GL", "egl" if gpu else "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", backend)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        in_colab = (
            importlib.util.find_spec("google.colab") is not None if install else False
        )
    except ModuleNotFoundError:
        in_colab = False
    if in_colab:
        subprocess.run(
            ["apt-get", "update", "-qq"], check=True, stdout=subprocess.DEVNULL
        )
        subprocess.run(
            [
                "apt-get",
                "install",
                "-y",
                "-qq",
                "libegl1",
                "libgl1",
                "libosmesa6",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    sys.path.insert(0, str(project_dir / "src"))
    importlib.invalidate_caches()
    import torch

    torch.set_num_threads(1)
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Python {sys.version.split()[0]} / PyTorch {torch.__version__} / {device}")
    print("準備完了。以降は通常のPython APIを使います。")
