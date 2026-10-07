"""Installation only; research runs through the public package in the kernel."""

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path


def prepare_colab(project_dir: str | Path) -> None:
    """Install into the active kernel and configure headless rendering before import."""
    if (
        importlib.util.find_spec("google") is not None
        and importlib.util.find_spec("google.colab") is not None
    ):
        subprocess.run(["apt-get", "update", "-qq"], check=True)
        subprocess.run(
            ["apt-get", "install", "-y", "-qq", "libosmesa6", "libgl1"], check=True
        )
    if shutil.which("uv") is None:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "uv==0.11.7"], check=True
        )
    subprocess.run(
        [
            shutil.which("uv"),
            "pip",
            "install",
            "--python",
            sys.executable,
            "--editable",
            str(project_dir),
            "--torch-backend",
            "cpu",
        ],
        check=True,
    )
    backend = os.environ.get("HEXAPOD_RENDER_BACKEND", "osmesa")
    os.environ.update(
        MUJOCO_GL=backend,
        PYOPENGL_PLATFORM=backend,
        MPLBACKEND="Agg",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    sys.path.insert(0, str(Path(project_dir) / "src"))
    importlib.invalidate_caches()
    import imageio_ffmpeg
    import mediapy as media
    import torch

    torch.set_num_threads(1)
    media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    print("準備完了。以降は通常のimportで環境・学習・評価APIを使います。")
