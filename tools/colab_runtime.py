"""Colab infrastructure, kept separate from student simulation and research cells.

Walking and environment inspection run directly in the notebook kernel.
Parallel training and evaluation use the locked Python 3.12 environment.
"""

import ctypes.util
import hashlib
import html
import importlib
import io
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from IPython.display import HTML, Image, display


class ColabLesson:
    """Run the course workflow without exposing process or storage boilerplate."""

    def __init__(self, project_dir):
        """Prepare direct notebook imports and the isolated training environment."""
        self.project_dir = Path(project_dir).resolve()
        self.repository = "hashimoto-robotics-lab/hexapod-transport-rl"
        self.repository_commit = subprocess.check_output(
            ["git", "-C", str(self.project_dir), "rev-parse", "HEAD"], text=True
        ).strip()
        self.run_dir = None
        self.drive_dir = None
        self.colab_files = None
        try:
            from google.colab import files as colab_files

            self.in_colab = True
            self.colab_files = colab_files
        except ImportError:
            self.in_colab = False
        if self.in_colab:
            subprocess.run(["apt-get", "update", "-qq"], check=True)
            subprocess.run(
                ["apt-get", "install", "-y", "-qq", "libosmesa6", "libgl1"], check=True
            )
        if shutil.which("uv") is None:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "uv==0.11.7"], check=True
            )
        self.uv = shutil.which("uv")
        subprocess.run(
            [self.uv, "sync", "--locked", "--no-dev", "--python", "3.12"],
            cwd=self.project_dir,
            check=True,
        )
        self.python = self.project_dir / ".venv/bin/python"
        self.process_env = {
            **os.environ,
            "MUJOCO_GL": os.environ.get("HEXAPOD_RENDER_BACKEND", "osmesa"),
            "PYOPENGL_PLATFORM": os.environ.get("HEXAPOD_RENDER_BACKEND", "osmesa"),
            "MPLBACKEND": "Agg",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(self.project_dir / "src"),
        }
        if self.process_env["MUJOCO_GL"] == "osmesa" and (
            not ctypes.util.find_library("OSMesa")
        ):
            raise RuntimeError(
                "OSMesaが見つかりません。Step 2のインストール結果を確認してください。"
            )
        self._prepare_notebook_kernel()
        print("セルから直接importできる環境と、並列学習用の環境を準備しました。")

    def _prepare_notebook_kernel(self):
        """Install into the actual kernel, retaining compatible preinstalled packages."""
        # Training keeps its lockfile; the notebook retains Colab's compatible
        # NumPy/PyTorch so imported scientific libraries do not need replacing.
        subprocess.run(
            [
                self.uv,
                "pip",
                "install",
                "--python",
                sys.executable,
                "--editable",
                str(self.project_dir),
                "--torch-backend",
                "cpu",
            ],
            check=True,
        )
        for key in (
            "MUJOCO_GL",
            "PYOPENGL_PLATFORM",
            "MPLBACKEND",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
        ):
            os.environ[key] = self.process_env[key]
        sys.path.insert(0, str(self.project_dir / "src"))
        importlib.invalidate_caches()
        import torch

        torch.set_num_threads(1)
        import imageio_ffmpeg
        import mediapy as media

        media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())

    def configure(
        self,
        *,
        name="trial_01",
        mode="quick",
        num_envs=2,
        seed=20261006,
        start_from_scratch=False,
        save_to_drive=False,
        restore_results=False,
    ):
        """Record settings and source snapshots; resume only matching experiments."""
        self.mode = mode
        self.start_from_scratch = start_from_scratch
        self.num_envs = num_envs
        self.training_seed = seed
        self.experiment_id = name
        self.save_to_drive = save_to_drive
        self.restore_results = restore_results
        if (
            self.mode not in ("quick", "research")
            or not isinstance(self.num_envs, int)
            or isinstance(self.num_envs, bool)
            or (self.num_envs < 1)
        ):
            raise ValueError(
                "modeはquick/research、num_envsは1以上の整数を指定してください。"
            )
        if not re.fullmatch("[A-Za-z0-9_-]+", self.experiment_id):
            raise ValueError("実験名は英数字・_・-で指定してください。")
        self.run_dir = self.project_dir / "runs" / self.experiment_id
        self.drive_dir = None
        if self.save_to_drive:
            if not self.in_colab:
                raise RuntimeError("Drive接続はColabで実行してください。")
            from google.colab import drive

            drive.mount("/content/drive")
            self.drive_dir = (
                Path("/content/drive/MyDrive/HexapodTransportRL") / self.experiment_id
            )
            if self.drive_dir.exists() and (not self.run_dir.exists()):
                shutil.copytree(self.drive_dir, self.run_dir)
                print("Driveから実験を復元しました。")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.restore_results:
            if not self.in_colab:
                raise RuntimeError("結果ZIPのアップロードはColabで実行してください。")
            uploaded = self.colab_files.upload()
            if len(uploaded) != 1:
                raise ValueError("結果ZIPを1ファイル指定してください。")
            with zipfile.ZipFile(io.BytesIO(next(iter(uploaded.values())))) as archive:
                for item in archive.infolist():
                    if (
                        not (self.run_dir / item.filename)
                        .resolve()
                        .is_relative_to(self.run_dir.resolve())
                    ):
                        raise ValueError("Invalid results archive path")
                archive.extractall(self.run_dir)
        self.horizon = 8 if self.mode == "quick" else 64
        targets = {
            "push_initial": 32,
            "push_refined": 32,
            "navigation": 64,
            "handover": 32,
        }
        self.research_targets = {
            "push_initial": 512000,
            "push_refined": 512000,
            "navigation": 409600,
            "handover": 153600,
        }
        if self.mode == "research":
            targets = self.research_targets.copy()
        self.iterations = {
            phase: math.ceil(steps / (self.num_envs * self.horizon))
            for phase, steps in targets.items()
        }
        self.eval_episodes = 2 if self.mode == "quick" else 50
        self.validation_episodes = 2 if self.mode == "quick" else 24
        self.eval_seconds = 10 if self.mode == "quick" else 100
        settings = dict(
            mode=self.mode,
            start_from_scratch=self.start_from_scratch,
            num_envs=self.num_envs,
            training_seed=self.training_seed,
            experiment_id=self.experiment_id,
            horizon=self.horizon,
            planned_iterations=self.iterations,
            evaluation_episodes=self.eval_episodes,
            evaluation_seconds=self.eval_seconds,
            repository=self.repository,
            repository_commit=self.repository_commit,
            source_sha256={
                str(path.relative_to(self.project_dir)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for directory in ("src", "tools")
                for path in sorted((self.project_dir / directory).rglob("*"))
                if path.is_file()
                and "__pycache__" not in path.parts
                and (path.suffix != ".pyc")
            },
        )
        config_path = self.run_dir / "experiment.json"
        if config_path.exists() and json.loads(config_path.read_text()) != settings:
            raise RuntimeError(
                "この実験名は異なる設定で使用済みです。configureのnameに新しい実験名を指定してください。"
            )
        self.settings = settings
        config_path.write_text(json.dumps(settings, indent=2) + "\n")
        snapshot = self.run_dir / "source_snapshot"
        if not snapshot.exists():
            snapshot.mkdir()
            shutil.copytree(
                self.project_dir / "src",
                snapshot / "src",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
            shutil.copytree(
                self.project_dir / "tools",
                snapshot / "tools",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
            for name in ("pyproject.toml", "uv.lock", "README.md", "run.sh"):
                shutil.copy2(self.project_dir / name, snapshot / name)
        print(f"実験: {self.experiment_id} / {self.mode} / 並列世界数: {self.num_envs}")
        print("CPU数:", os.cpu_count(), "独立世界数:", self.num_envs)
        if self.mode == "quick":
            print("quickでは新しい方策の運搬成功は期待せず、接続を確認します。")
        self._record_runtime()

    def _backup_to_drive(self):
        """Copy checkpoints and logs to Drive without running simulations there."""
        if self.drive_dir is not None:
            shutil.copytree(
                self.run_dir,
                self.drive_dir,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("*.tmp", "*.partial", "source_snapshot"),
            )
            if not (self.drive_dir / "source_snapshot").exists():
                shutil.copytree(
                    self.run_dir / "source_snapshot", self.drive_dir / "source_snapshot"
                )

    def _run_command(self, arguments, label):
        """Launch a logged subprocess and preserve backups on interruption."""
        logs = self.run_dir / "logs"
        logs.mkdir(exist_ok=True)
        command = [str(self.python), *map(str, arguments)]
        print("実行:", label)
        with (logs / f"{label}.txt").open("w") as log:
            process = subprocess.Popen(
                command,
                cwd=self.project_dir,
                env=self.process_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(record, dict):
                        continue
                    if "iteration" in record and (
                        record["iteration"] % 50 == 0 or self.mode == "quick"
                    ):
                        print(
                            {
                                key: record[key]
                                for key in (
                                    "iteration",
                                    "transitions",
                                    "layout",
                                    "mean_reward",
                                    "validation_success_rate",
                                )
                                if key in record
                            }
                        )
                    if "iteration" in record and record["iteration"] % 25 == 0:
                        self._backup_to_drive()
                returncode = process.wait()
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                process.wait()
                self._backup_to_drive()
                raise
        if returncode:
            print((logs / f"{label}.txt").read_text()[-4000:])
            raise RuntimeError(
                f"{label} が失敗しました。ログ: {logs / (label + '.txt')}"
            )
        self._backup_to_drive()

    def _run_cli(self, arguments, label):
        """Invoke the same CLI used for local research experiments."""
        self._run_command(["-m", "hexapod_transport_rl.cli", *arguments], label)

    def _checkpoint_iteration(self, path):
        """Read progress with PyTorch in the simulation process."""
        source = "import sys, torch; print(torch.load(sys.argv[1], map_location='cpu', weights_only=True)['iteration'])"
        result = subprocess.run(
            [str(self.python), "-c", source, str(path)],
            cwd=self.project_dir,
            env=self.process_env,
            check=True,
            capture_output=True,
            text=True,
        )
        return int(result.stdout.strip())

    def _train_phase(
        self, phase, command, *, initial_checkpoint=None, extra_args=(), seed_offset=0
    ):
        """Add only missing updates, keeping attempts and checkpoint ancestry."""
        phase_dir = self.run_dir / phase
        phase_dir.mkdir(exist_ok=True)
        saved = sorted(phase_dir.glob("attempt_*/checkpoint.pt"))
        latest = saved[-1] if saved else None
        initial_iteration = (
            self._checkpoint_iteration(initial_checkpoint) if initial_checkpoint else 0
        )
        target_iteration = initial_iteration + self.iterations[phase]
        current_iteration = (
            self._checkpoint_iteration(latest) if latest else initial_iteration
        )
        if current_iteration >= target_iteration:
            print(phase, "は完了済みです。", latest)
            return latest
        attempt = (
            max(
                [int(path.name.split("_")[-1]) for path in phase_dir.glob("attempt_*")]
                + [0]
            )
            + 1
        )
        output = phase_dir / f"attempt_{attempt:03d}"
        extra_args = list(extra_args)
        if latest and "--initial-log-std" in extra_args:
            option = extra_args.index("--initial-log-std")
            del extra_args[option : option + 2]
        arguments = [
            command,
            "--output",
            str(output),
            "--iterations",
            str(target_iteration - current_iteration),
            "--num-envs",
            str(self.num_envs),
            "--horizon",
            str(self.horizon),
            "--seed",
            str(self.training_seed + seed_offset),
            *extra_args,
        ]
        resume = latest or initial_checkpoint
        if command == "train-handover":
            arguments.extend(["--checkpoint", str(resume)])
        elif resume:
            arguments.extend(["--resume", str(resume)])
        started = time.perf_counter()
        self._run_cli(arguments, f"{phase}_attempt_{attempt:03d}")
        checkpoint = output / "checkpoint.pt"
        if not checkpoint.exists():
            raise RuntimeError(
                "学習済みモデルが保存されませんでした。ログを確認してください。"
            )
        elapsed = time.perf_counter() - started
        metrics = [
            json.loads(line)
            for line in (output / "metrics.jsonl").read_text().splitlines()
        ]
        rate = sum((item["transitions_per_second"] for item in metrics)) / len(metrics)
        print(f"{phase}: {elapsed:.1f}秒、収集・更新 {rate:.1f}チームステップ/秒")
        if self.mode == "quick":
            print(
                f"同じ速度で{self.research_targets[phase]:,}ステップ: 約{self.research_targets[phase] / max(rate, 1) / 60:.1f}分（準備・検証を含まない概算）"
            )
        return checkpoint

    def show_results(self):
        """Display training curves and held-out evaluation; save PNG, PDF and CSV."""
        self._run_command(
            [self.project_dir / "tools/colab_analysis.py", self.run_dir], "make_figures"
        )
        display(Image(filename=str(self.run_dir / "learning_curves.png")))
        rows = json.loads((self.run_dir / "evaluation_table.json").read_text())
        table = "<table><tr><th>配置</th><th>運搬成功率</th><th>95%区間</th><th>回り込み成功率</th><th>胴体接触</th><th>転倒</th><th>位置誤差 [m]</th></tr>"
        for row in rows:
            table += f"<tr><td>{html.escape(row['layout'])}</td><td>{row['success_rate']:.1%}</td><td>{row['success_ci95_low']:.1%}–{row['success_ci95_high']:.1%}</td><td>{row['handover_rate']:.1%}</td><td>{row['body_contact_episodes']}</td><td>{row['falls']}</td><td>{row['mean_distance_m']:.3f}</td></tr>"
        display(HTML(table + "</table>"))
        print(
            "保存:",
            self.run_dir / "evaluation.csv",
            self.run_dir / "learning_curves.pdf",
        )

    def show_reference(self):
        """Simulate the bundled reward-trained reference policy and show its video."""
        reference_dir = self.run_dir / "reference"
        reference_result = reference_dir / "result.json"
        if not reference_result.exists():
            self._run_cli(
                [
                    "eval",
                    "--checkpoint",
                    self.project_dir / "checkpoints/transport.pt",
                    "--episodes",
                    "1",
                    "--seed",
                    "70000",
                    "--output",
                    reference_result,
                    "--video-dir",
                    reference_dir / "video",
                    "--video-width",
                    "640",
                    "--video-height",
                    "480",
                    "--video-fps",
                    "5",
                    "--fast-video",
                ],
                "reference",
            )
        reference = json.loads(reference_result.read_text())
        print("参考方策の成功:", reference["successes"], "/ 1")
        print("最終位置誤差 [m]:", reference["mean_final_distance_m"])
        self._show_video(reference_dir / "video/seed_70000.mp4")

    def prepare_pusher(self):
        """Use a reward-trained pusher or learn it from random weights."""
        if self.start_from_scratch:
            initial_push = self._train_phase(
                "push_initial",
                "train-push",
                extra_args=("--epochs", "4", "--minibatch", "128"),
            )
            refined_push = self._train_phase(
                "push_refined",
                "train-push",
                initial_checkpoint=initial_push,
                extra_args=(
                    "--epochs",
                    "8",
                    "--minibatch",
                    "256",
                    "--mirror-equivariant",
                    "--initial-log-std",
                    "-1.5",
                ),
                seed_offset=1,
            )
            base_source = refined_push
        else:
            base_source = self.project_dir / "checkpoints/base_pusher.pt"
        self.base_pusher = self.run_dir / "base_pusher.pt"
        if self.base_pusher.exists():
            if self.base_pusher.read_bytes() != base_source.read_bytes():
                raise RuntimeError(
                    "初期pusherが変わりました。新しい実験名を使ってください。"
                )
        else:
            shutil.copy2(base_source, self.base_pusher)
        print("初期pusher:", self.base_pusher)
        self._backup_to_drive()
        return self.base_pusher

    def train_navigation(self, pusher):
        """Learn to navigate around the cargo using MAPPO rewards."""
        self.base_pusher = Path(pusher)
        self.navigator = self._train_phase(
            "navigation",
            "train",
            extra_args=(
                "--pushing-checkpoint",
                str(self.base_pusher),
                "--epochs",
                "4",
                "--minibatch",
                "128",
                "--validate-every",
                "50",
            ),
            seed_offset=2,
        )
        return self.navigator

    def train_handover(self, pusher):
        """Adapt pushing to the distribution of navigation completion poses."""
        self.base_pusher = Path(pusher)
        self.pusher = self._train_phase(
            "handover",
            "train-handover",
            initial_checkpoint=self.base_pusher,
            seed_offset=3,
        )
        return self.pusher

    def select_model(self, navigator, pusher):
        """Compose candidate models and select using validation seeds only."""
        self.navigator = Path(navigator)
        self.pusher = Path(pusher)
        candidates_dir = self.run_dir / "selection"
        candidates_dir.mkdir(exist_ok=True)
        candidate_models = [self.navigator]
        best_navigation = self.navigator.parent / "best_front_checkpoint.pt"
        if best_navigation.exists():
            candidate_models.append(best_navigation)
        candidates = []
        for index, navigation in enumerate(candidate_models):
            model = candidates_dir / f"candidate_{index:02d}.pt"
            if not model.exists():
                self._run_cli(
                    [
                        "compose",
                        "--navigation-checkpoint",
                        navigation,
                        "--pushing-checkpoint",
                        self.pusher,
                        "--output",
                        model,
                    ],
                    f"compose_{index:02d}",
                )
            result = candidates_dir / f"candidate_{index:02d}.json"
            if not result.exists():
                self._run_cli(
                    [
                        "eval",
                        "--checkpoint",
                        model,
                        "--episodes",
                        str(self.validation_episodes),
                        "--workers",
                        str(self.num_envs),
                        "--seed",
                        "66000",
                        "--seconds",
                        str(self.eval_seconds),
                        "--output",
                        result,
                    ],
                    f"validation_{index:02d}",
                )
            report = json.loads(result.read_text())
            candidates.append((model, report))
        selected, validation = max(
            candidates,
            key=lambda item: (
                item[1]["successes"],
                -item[1]["body_contact_episodes"],
                -item[1]["navigation_robot_contact_episodes"],
                -item[1]["mean_final_distance_m"],
            ),
        )
        self.selected_checkpoint = selected
        selection = {
            "checkpoint": str(selected.relative_to(self.run_dir)),
            "validation_seed": 66000,
            "validation_episodes": self.validation_episodes,
            "successes": validation["successes"],
            "candidates": [
                dict(
                    checkpoint=str(model.relative_to(self.run_dir)),
                    successes=report["successes"],
                    mean_final_distance_m=report["mean_final_distance_m"],
                )
                for model, report in candidates
            ],
        }
        (self.run_dir / "selection.json").write_text(
            json.dumps(selection, indent=2) + "\n"
        )
        print(json.dumps(selection, indent=2))
        return self.selected_checkpoint

    def evaluate(self, checkpoint):
        """Evaluate front and side layouts using independent test seeds."""
        self.selected_checkpoint = Path(checkpoint)
        test_reports = {}
        for layout, seed in (("front", 80000), ("side", 81000)):
            output = self.run_dir / f"test_{layout}.json"
            if not output.exists():
                self._run_cli(
                    [
                        "eval",
                        "--checkpoint",
                        self.selected_checkpoint,
                        "--layout",
                        layout,
                        "--seed",
                        str(seed),
                        "--episodes",
                        str(self.eval_episodes),
                        "--workers",
                        str(self.num_envs),
                        "--seconds",
                        str(self.eval_seconds),
                        "--output",
                        output,
                    ],
                    f"test_{layout}",
                )
            test_reports[layout] = json.loads(output.read_text())
        print(
            {
                layout: dict(
                    successes=report["successes"],
                    episodes=len(report["episodes"]),
                    handovers=report["handovers"],
                    body_contact_episodes=report["body_contact_episodes"],
                    falls=report["falls"],
                )
                for layout, report in test_reports.items()
            }
        )
        return test_reports

    def show_learned_video(self, checkpoint):
        """Record the selected policy at a fixed seed, including failed trials."""
        self.selected_checkpoint = Path(checkpoint)
        learned_dir = self.run_dir / "learned_video"
        learned_result = learned_dir / "result.json"
        if not learned_result.exists():
            self._run_cli(
                [
                    "eval",
                    "--checkpoint",
                    self.selected_checkpoint,
                    "--episodes",
                    "1",
                    "--seed",
                    "82000",
                    "--layout",
                    "near" if self.mode == "quick" else "front",
                    "--seconds",
                    str(self.eval_seconds),
                    "--output",
                    learned_result,
                    "--video-dir",
                    learned_dir / "video",
                    "--video-width",
                    "640",
                    "--video-height",
                    "480",
                    "--video-fps",
                    "5",
                    "--fast-video",
                ],
                "learned_video",
            )
        learned = json.loads(learned_result.read_text())
        print("自分の方策の運搬成功:", learned["successes"], "/ 1")
        self._show_video(learned_dir / "video/seed_82000.mp4")

    def save_results(self):
        """Archive the complete experiment and offer a Colab download."""
        self._backup_to_drive()
        archive_path = self.project_dir / "runs" / f"{self.experiment_id}_results.zip"
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(self.run_dir.rglob("*")):
                if path.is_file() and path.suffix not in (".tmp", ".partial"):
                    archive.write(path, path.relative_to(self.run_dir))
        print(
            "結果ZIP:",
            archive_path,
            f"({archive_path.stat().st_size / 1024**2:.1f} MiB)",
        )
        if self.in_colab:
            self.colab_files.download(str(archive_path))
        return archive_path

    def _record_runtime(self):
        """Record notebook and training versions separately for reproducibility."""
        import mujoco
        import numpy
        import torch

        runtime = dict(
            python=sys.version,
            executable=sys.executable,
            torch=torch.__version__,
            mujoco=mujoco.__version__,
            numpy=numpy.__version__,
            cpu_count=os.cpu_count(),
            torch_threads=torch.get_num_threads(),
            rendering_backend=os.environ.get("MUJOCO_GL"),
        )
        (self.run_dir / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
        source = r"""import json, os, sys
from pathlib import Path
import mujoco, numpy, torch
torch.set_num_threads(1)
runtime = dict(python=sys.version, torch=torch.__version__, mujoco=mujoco.__version__,
               numpy=numpy.__version__, cpu_count=os.cpu_count(), torch_threads=torch.get_num_threads(),
               rendering_backend=os.environ.get("MUJOCO_GL"))
Path(sys.argv[1]).write_text(json.dumps(runtime, indent=2) + "\n")
"""
        self._run_command(
            ["-c", source, self.run_dir / "training_runtime.json"],
            "record_training_runtime",
        )

    def _show_video(self, path):
        """Display a recorded transport rollout using the same mediapy UI."""
        import mediapy as media

        frames = media.read_video(str(path))
        media.show_video(frames, fps=frames.metadata.fps)
