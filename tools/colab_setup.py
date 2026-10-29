"""Install the teaching environment before importing simulation libraries."""

import importlib
import importlib.util
import os
import subprocess
import sys
import warnings
from pathlib import Path


def quiet_dependency_notices():
    """Keep known compatibility notices out of teaching output."""
    for message in (
        r"Creating .* which inherits from WeightUpdaterBase is deprecated\.",
        r"`torch\.jit\.load` is deprecated\.",
        r"MULTICCD is enabled, but the scene contains CCD pairs without multicontact support:",
    ):
        warnings.filterwarnings("ignore", message=message)


def prepare_colab(project_dir, *, install=True):
    """Use the runtime's PyTorch and prepare the simulator before student imports."""
    project_dir = Path(project_dir).resolve()
    import torch

    gpu = torch.cuda.is_available()
    if install:
        # Keep the CUDA build already provided by Colab, including after imports.
        constraints = project_dir / ".colab-constraints.txt"
        constraints.write_text(f"torch=={torch.__version__}\n")
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
    # Set this before importing MuJoCo/PyOpenGL, including CPU video rendering.
    backend = os.environ.setdefault("MUJOCO_GL", "egl" if gpu else "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", backend)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    if (
        install
        and importlib.util.find_spec("google")
        and importlib.util.find_spec("google.colab")
    ):
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

    quiet_dependency_notices()

    torch.set_num_threads(1)
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Python {sys.version.split()[0]} / PyTorch {torch.__version__} / {device}")
    print("準備完了。以降は通常のPython APIを使います。")
