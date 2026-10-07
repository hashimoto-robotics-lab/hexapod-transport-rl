"""Build the Colab lesson that fetches its teaching code from public GitHub."""

import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/hexapod_transport_rl_colab.ipynb"
REPOSITORY = "hashimoto-robotics-lab/hexapod-transport-rl"


def main():
    cells = []

    def markdown(source):
        cells.append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": textwrap.dedent(source).strip() + "\n",
            }
        )

    def code(source, *, form=False):
        index = len(cells)
        cells.append(
            {
                "cell_type": "code",
                "metadata": {
                    "id": f"step_{index:02d}",
                    **({"cellView": "form"} if form else {}),
                },
                "source": textwrap.dedent(source).strip() + "\n",
                "outputs": [],
                "execution_count": None,
            }
        )

    markdown("""
    # 六足ロボットの協調物資運搬

    ## なぜこの研究をするのか

    複数のロボットで荷物を押すには、荷物の周囲で移動する位置、押す向き、
    相手に合わせるタイミングを決める必要があります。
    この研究では、**2台の六足ロボットがT字物体を回り込み、脚で目標位置まで押す協調動作**を学習します。
    接触や初期位置によって必要な指令が変わるため、試行した結果を報酬として返し、
    状況に合った指令を学ぶ強化学習を使います。
    まず平坦な床のシミュレーションで、どの初期配置から成功し、どんな条件で失敗するかを調べます。

    研究で確かめたい問いは、次の3つです。

    - 前方や側方から始めても、荷物を避けて押す位置へ回り込めるか。
    - 2台が荷物の位置と向きを合わせながら、目標まで運べるか。
    - 報酬や学習量、初期配置を変えると、成功率や接触・転倒はどう変わるか。

    ## 出発点：六足ロボットは、すでに歩行を学習している

    この教材には、**速度コマンドを受け取り、18関節を動かす学習済み歩行モデル**が入っています。
    前進・横移動・旋回の目標速度を送ると、ロボットが脚を動かします。
    最初はこのモデルを読み込んで、指令を変えるだけで動かせることを動画で確かめましょう。

    ```text
    速度コマンド [前後速度, 左右速度, 旋回速度]
                       ↓
             学習済み歩行モデル（固定）
                       ↓
               18関節の目標角度
                       ↓
              モーター・脚・床の接触
                       ↓
                 ロボットが移動
    ```

    その後、**荷物と相手の状態から、各機へ送る速度コマンドを決める運搬方策**を強化学習します。
    歩行モデルは固定したままです。運搬方策にはチームの報酬を使い、デモを教師には使いません。

    **上から順に実行**してください。
    研究の動機 → 準備 → コマンドで歩行を体験 → 協調運搬の学習 → 評価・動画 → 保存、の順に進みます。
    学習は最初に `quick` で一巡し、その後 `research` で本学習に進みます。
    """)
    markdown("""
    ## Step 1 — GitHubから教材を取得する

    公開リポジトリからコード・ロボット形状・学習済みモデルを取得します。
    **GitHubへのログインやアクセストークンは不要**です。

    `GIT_REF` は通常 `main` のままで構いません。卒論の再現実験ではタグ名・コミットSHAを指定できます。
    """)
    bootstrap = (ROOT / "tools/colab_checkout.py").read_text()
    code(
        "#@title 教材の取得\n"
        f'REPOSITORY = "{REPOSITORY}"\n'
        'GIT_REF = "main" #@param {type:"string"}\n'
        "\n"
        + bootstrap
        + '\nCHECKOUT_DIR = Path(os.environ.get("HEXAPOD_WORKSPACE_ROOT", "/content/hexapod_transport_rl"))\n'
        + "PROJECT_DIR, REPOSITORY_COMMIT = checkout_repository(REPOSITORY, GIT_REF, CHECKOUT_DIR)\n"
        + 'print("教材:", PROJECT_DIR)\nprint("使用するコミット:", REPOSITORY_COMMIT)\n',
        form=True,
    )
    markdown("""
    ## Step 2 — 実行環境を準備する
    ColabのPythonとは別に **Python 3.12・固定した依存ライブラリ** を用意します。以降、学習と描画は専用環境の別プロセスで実行します。
    CPUで使えるOSMesaを入れて、ウィンドウを開かずに録画します。インストールにはインターネット接続が必要です。
    """)
    code(
        r"""
    #@title ライブラリの準備（変更不要）
    import ctypes.util
    import hashlib
    import io
    import zipfile
    import shutil
    import subprocess
    import sys
    from IPython.display import HTML, Image, Video, display

    try:
        from google.colab import files as colab_files
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False

    if IN_COLAB:
        subprocess.run(["apt-get", "update", "-qq"], check=True)
        subprocess.run(["apt-get", "install", "-y", "-qq", "libosmesa6", "libgl1"], check=True)
    if shutil.which("uv") is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv==0.11.7"], check=True)
    UV = shutil.which("uv")
    subprocess.run([UV, "sync", "--locked", "--no-dev", "--python", "3.12"], cwd=PROJECT_DIR, check=True)
    PYTHON = PROJECT_DIR / ".venv/bin/python"
    PROCESS_ENV = {
        **os.environ,
        "MUJOCO_GL": os.environ.get("HEXAPOD_RENDER_BACKEND", "osmesa"),
        "PYOPENGL_PLATFORM": os.environ.get("HEXAPOD_RENDER_BACKEND", "osmesa"),
        "MPLBACKEND": "Agg",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(PROJECT_DIR / "src"),
    }
    if PROCESS_ENV["MUJOCO_GL"] == "osmesa" and not ctypes.util.find_library("OSMesa"):
        raise RuntimeError("OSMesaが見つかりません。Step 2のインストール結果を確認してください。")
    subprocess.run([str(PYTHON), "-c", "import sys, mujoco, torch; print(sys.version); print('MuJoCo', mujoco.__version__, 'PyTorch', torch.__version__)"], env=PROCESS_ENV, check=True)
    print("CPU実行環境と描画環境を準備しました。")

    def run_example(source, *, output_dir):
        # Run student API calls using the isolated simulation Python.
        output_dir.mkdir(parents=True, exist_ok=True)
        script = output_dir / "example.py"
        script.write_text(source)
        subprocess.run([str(PYTHON), str(script)], cwd=output_dir, env=PROCESS_ENV, check=True)
    """,
        form=True,
    )
    markdown("""
    ## Step 3 — 学習済みロボットにコマンドを送る

    `WalkingSimulation` は、**床と指定した台数のロボットだけ**を用意する歩行用APIです。
    新しい学習は行いません。まず1台にコマンドを送り、歩行を動画で確かめます。

    - `set_velocity(robot_id=0, vx=0.12)`：0番のロボットの前進指令を0.12 m/sにする。
    - `run_for(seconds=3.0)`：全機を、現在の指令で同時に3秒間動かす。
    - `stop()`：全機にゼロ速度を指令する。実際の減速を観察するには、その後も時間を進める。

    指令は、次に変更するまで続きます。`set_velocity()` で省略した速度成分はゼロになります。
    `robot_id` は0から始まります。`vx`・`vy` は機体座標の速度、`yaw_rate` は旋回速度です。
    指令は目標速度なので、実際の動きには姿勢や接触による差が出ます。

    | 引数 | 単位 | 指定できる範囲 | 正の値の意味 |
    |---|---|---|---|
    | `vx` | m/s | −0.15〜0.20 | 前進 |
    | `vy` | m/s | −0.10〜0.10 | 左移動 |
    | `yaw_rate` | rad/s | −0.60〜0.60 | 左旋回 |

    下のAPI呼び出しを1つ変えて再実行し、動きの違いを確かめてください。
    `with` は最後に動画の保存と後片付けを行います。
    `run_example()` はセル内のPythonコードをStep 2の専用環境で実行するための補助関数です。
    `run_for()` の時間は、歩行制御周期の0.04秒刻みで指定します。
    """)
    code(
        r"""
WALKING_DIR = PROJECT_DIR / "runs/walking_commands"
run_example(r"""
        + "'''"
        + r"""
from hexapod_transport_rl import WalkingSimulation

with WalkingSimulation(num_robots=1, video_path="walking_commands.mp4") as sim:
    sim.stop()
    sim.run_for(seconds=0.8)

    sim.set_velocity(robot_id=0, vx=0.12)
    sim.run_for(seconds=3.0)
    print("前進後の位置 [m]:", sim.positions)

    sim.set_velocity(robot_id=0, vy=0.06)
    sim.run_for(seconds=2.4)
    print("左移動後の位置 [m]:", sim.positions)

    sim.set_velocity(robot_id=0, yaw_rate=0.40)
    sim.run_for(seconds=2.4)
    print("旋回後の向き [rad]:", sim.headings)

    sim.stop()
    sim.run_for(seconds=0.8)
''', output_dir=WALKING_DIR)
display(Video(str(WALKING_DIR / "walking_commands.mp4"), embed=True))
    """
    )
    markdown("""
    ### 同じAPIで4台を動かす

    `num_robots=4` とし、0〜3番へ別々の指令を設定します。
    **指令の設定だけでは時間は進みません。** `run_for()` で4台が同時に動きます。
    2番には前進と左移動を同時に指令します。固定歩行モデルは、停止から横移動だけを始めると速度の追従が弱い場合があります。
    ここでも学習は行わず、荷物はありません。4台の協調運搬を学習する環境は、今後拡張する研究課題です。
    """)
    code(
        r"""
FOUR_ROBOTS_DIR = PROJECT_DIR / "runs/four_robot_commands"
run_example(r"""
        + "'''"
        + r"""
from hexapod_transport_rl import WalkingSimulation

with WalkingSimulation(num_robots=4, video_path="four_robots.mp4") as sim:
    sim.run_for(seconds=0.8)
    sim.set_velocity(robot_id=0, vx=0.10)
    sim.set_velocity(robot_id=1, vx=0.05)
    sim.set_velocity(robot_id=2, vx=0.08, vy=0.04)
    sim.set_velocity(robot_id=3, yaw_rate=0.30)
    sim.run_for(seconds=3.0)
    print("4台の位置 [m]:", sim.positions)
    print("4台の向き [rad]:", sim.headings)
    sim.stop()
    sim.run_for(seconds=0.8)
''', output_dir=FOUR_ROBOTS_DIR)
display(Video(str(FOUR_ROBOTS_DIR / "four_robots.mp4"), embed=True))
    """
    )
    markdown("""
    ## 歩行コマンドから協調運搬へ

    今は私たちが速度コマンドを指定しました。ここからは、**荷物・目標・相手の状態を観測して、
    2台それぞれのコマンドを運搬方策が決める**ようにします。

    ```text
    荷物・目標・相手の状態 → 運搬方策（今回MAPPOで学習）
                                     ↓ 各機の速度コマンド
                            歩行モデル（学習済み・固定）
                                     ↓ 関節の動作
                              2台とT字物体の接触・移動
                                     ↓ チーム報酬
                               運搬方策を更新
    ```

    歩行体験で指定したコマンドは、運搬学習の教師には使いません。
    次の工程では、環境APIの観測・行動・報酬を確認し、参考の運搬方策を再生してから、自分の運搬方策を学習します。
    """)
    markdown("""
    ## Step 4 — 実験を設定する
    `quick` は数回の更新だけで全工程を確認するモードです。**短い確認の成功率を卒論の性能として扱わないでください。**
    `research` では回り込み409,600、押す方策の適応153,600チームステップを目安に学習します。
    `START_FROM_SCRATCH=True` の場合は上位の押す方策も新規学習します。固定する歩行モデルはどちらでも同じです。

    `NUM_ENVS` は独立した世界数です。1世界には2台のロボットがいます。Colabではまず2世界から始めてください。
    本学習では `SAVE_TO_DRIVE=True` を選ぶと途中のモデルを保存・復元できます。
    条件を変える際は `EXPERIMENT_ID` を変更してください。
    """)
    code(
        r"""
    #@title 実験設定
    MODE = "quick" #@param ["quick", "research"]
    START_FROM_SCRATCH = False #@param {type:"boolean"}
    NUM_ENVS = 2 #@param {type:"integer"}
    TRAINING_SEED = 20261006 #@param {type:"integer"}
    EXPERIMENT_ID = "trial_01" #@param {type:"string"}
    SAVE_TO_DRIVE = False #@param {type:"boolean"}
    RESTORE_RESULTS = False #@param {type:"boolean"}

    import json
    import math
    import re
    import time

    assert MODE in ("quick", "research") and NUM_ENVS >= 1
    assert re.fullmatch(r"[A-Za-z0-9_-]+", EXPERIMENT_ID), "実験名は英数字・_・-で指定してください。"
    RUN_DIR = PROJECT_DIR / "runs" / EXPERIMENT_ID
    DRIVE_DIR = None
    if SAVE_TO_DRIVE:
        if not IN_COLAB:
            raise RuntimeError("Drive接続はColabで実行してください。")
        from google.colab import drive
        drive.mount("/content/drive")
        DRIVE_DIR = Path("/content/drive/MyDrive/HexapodTransportRL") / EXPERIMENT_ID
        if DRIVE_DIR.exists() and not RUN_DIR.exists():
            shutil.copytree(DRIVE_DIR, RUN_DIR)
            print("Driveから実験を復元しました。")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if RESTORE_RESULTS:
        if not IN_COLAB:
            raise RuntimeError("結果ZIPのアップロードはColabで実行してください。")
        uploaded = colab_files.upload()
        if len(uploaded) != 1:
            raise ValueError("結果ZIPを1ファイル指定してください。")
        with zipfile.ZipFile(io.BytesIO(next(iter(uploaded.values())))) as archive:
            for item in archive.infolist():
                if not (RUN_DIR / item.filename).resolve().is_relative_to(RUN_DIR.resolve()):
                    raise ValueError("Invalid results archive path")
            archive.extractall(RUN_DIR)

    HORIZON = 8 if MODE == "quick" else 64
    targets = {"push_initial": 32, "push_refined": 32, "navigation": 64, "handover": 32}
    research_targets = {"push_initial": 512000, "push_refined": 512000, "navigation": 409600, "handover": 153600}
    if MODE == "research":
        targets = research_targets.copy()
    iterations = {phase: math.ceil(steps / (NUM_ENVS * HORIZON)) for phase, steps in targets.items()}
    EVAL_EPISODES = 2 if MODE == "quick" else 50
    VALIDATION_EPISODES = 2 if MODE == "quick" else 24
    EVAL_SECONDS = 10 if MODE == "quick" else 100
    settings = dict(mode=MODE, start_from_scratch=START_FROM_SCRATCH, num_envs=NUM_ENVS,
                    training_seed=TRAINING_SEED, experiment_id=EXPERIMENT_ID,
                    horizon=HORIZON, planned_iterations=iterations,
                    evaluation_episodes=EVAL_EPISODES, evaluation_seconds=EVAL_SECONDS,
                    repository=REPOSITORY, repository_commit=REPOSITORY_COMMIT,
                    repository_ref=GIT_REF,
                    source_sha256={str(path.relative_to(PROJECT_DIR)): hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in sorted((PROJECT_DIR / "src").rglob("*"))
                                   if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"})
    config_path = RUN_DIR / "experiment.json"
    if config_path.exists() and json.loads(config_path.read_text()) != settings:
        raise RuntimeError("この実験名は異なる設定で使用済みです。新しいEXPERIMENT_IDを指定してください。")
    config_path.write_text(json.dumps(settings, indent=2) + "\n")
    snapshot = RUN_DIR / "source_snapshot"
    if not snapshot.exists():
        snapshot.mkdir()
        shutil.copytree(PROJECT_DIR / "src", snapshot / "src",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for name in ("pyproject.toml", "uv.lock", "README.md", "run.sh"):
            shutil.copy2(PROJECT_DIR / name, snapshot / name)
    print(json.dumps({key: value for key, value in settings.items() if key != "source_sha256"}, indent=2))
    print("CPU数:", os.cpu_count(), "独立世界数:", NUM_ENVS)
    print("quickでは新しい方策の運搬成功は期待せず、接続を確認します。")
    """,
        form=True,
    )
    code(
        r"""
    #@title 学習と保存の準備（変更不要）
    # 学習の別プロセス起動、進捗表示、途中保存、再開を共通化する。
    import signal

    def backup_to_drive():
        if DRIVE_DIR is not None:
            shutil.copytree(RUN_DIR, DRIVE_DIR, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("*.tmp", "*.partial", "source_snapshot"))
            if not (DRIVE_DIR / "source_snapshot").exists():
                shutil.copytree(RUN_DIR / "source_snapshot", DRIVE_DIR / "source_snapshot")

    def run_command(arguments, label):
        logs = RUN_DIR / "logs"
        logs.mkdir(exist_ok=True)
        command = [str(PYTHON), *map(str, arguments)]
        print("実行:", label)
        with (logs / f"{label}.txt").open("w") as log:
            process = subprocess.Popen(command, cwd=PROJECT_DIR, env=PROCESS_ENV,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1, start_new_session=True)
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
                    if "iteration" in record and (record["iteration"] % 50 == 0 or MODE == "quick"):
                        print({key: record[key] for key in ("iteration", "transitions", "layout", "mean_reward", "validation_success_rate") if key in record})
                    if "iteration" in record and record["iteration"] % 25 == 0:
                        backup_to_drive()
                returncode = process.wait()
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                process.wait()
                backup_to_drive()
                raise
        if returncode:
            print((logs / f"{label}.txt").read_text()[-4000:])
            raise RuntimeError(f"{label} が失敗しました。ログ: {logs / (label + '.txt')}")
        backup_to_drive()

    def run_cli(arguments, label):
        run_command(["-m", "hexapod_transport_rl.cli", *arguments], label)

    def run_python(source, *arguments, name="lesson_step.py"):
        scripts = RUN_DIR / "scripts"
        scripts.mkdir(exist_ok=True)
        script = scripts / name
        script.write_text(source)
        run_command([script, *arguments], Path(name).stem)

    def checkpoint_iteration(path):
        source = "import sys, torch; print(torch.load(sys.argv[1], map_location='cpu', weights_only=True)['iteration'])"
        result = subprocess.run([str(PYTHON), "-c", source, str(path)], cwd=PROJECT_DIR,
                                env=PROCESS_ENV, check=True, capture_output=True, text=True)
        return int(result.stdout.strip())

    def train_phase(phase, command, *, initial_checkpoint=None, extra_args=(), seed_offset=0):
        phase_dir = RUN_DIR / phase
        phase_dir.mkdir(exist_ok=True)
        saved = sorted(phase_dir.glob("attempt_*/checkpoint.pt"))
        latest = saved[-1] if saved else None
        initial_iteration = checkpoint_iteration(initial_checkpoint) if initial_checkpoint else 0
        target_iteration = initial_iteration + iterations[phase]
        current_iteration = checkpoint_iteration(latest) if latest else initial_iteration
        if current_iteration >= target_iteration:
            print(phase, "は完了済みです。", latest)
            return latest
        attempt = max([int(path.name.split('_')[-1]) for path in phase_dir.glob("attempt_*")] + [0]) + 1
        output = phase_dir / f"attempt_{attempt:03d}"
        extra_args = list(extra_args)
        if latest and "--initial-log-std" in extra_args:
            option = extra_args.index("--initial-log-std")
            del extra_args[option:option + 2]
        arguments = [command, "--output", str(output), "--iterations", str(target_iteration - current_iteration),
                     "--num-envs", str(NUM_ENVS), "--horizon", str(HORIZON),
                     "--seed", str(TRAINING_SEED + seed_offset), *extra_args]
        resume = latest or initial_checkpoint
        if command == "train-handover":
            arguments.extend(["--checkpoint", str(resume)])
        elif resume:
            arguments.extend(["--resume", str(resume)])
        started = time.perf_counter()
        run_cli(arguments, f"{phase}_attempt_{attempt:03d}")
        checkpoint = output / "checkpoint.pt"
        assert checkpoint.exists()
        elapsed = time.perf_counter() - started
        metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
        rate = sum(item["transitions_per_second"] for item in metrics) / len(metrics)
        print(f"{phase}: {elapsed:.1f}秒、収集・更新 {rate:.1f}チームステップ/秒")
        if MODE == "quick":
            print(f"同じ速度で{research_targets[phase]:,}ステップ: 約{research_targets[phase] / max(rate, 1) / 60:.1f}分（準備・検証を含まない概算）")
        return checkpoint
    """,
        form=True,
    )
    markdown("""
    ## Step 5 — 環境APIを確認する
    2台の局所観測は `(2,20)`、各機の行動は `[前後, 左右, 旋回]` の `(2,3)` です。
    1 stepは0.2秒。1つのMuJoCo世界に2台とTがあり、報酬はチームで共有します。
    """)
    code(
        r"""
    run_python(r"""
        + "'''"
        + r"""
import json
import os
import sys
from pathlib import Path
import mujoco
import numpy as np
import torch
from hexapod_transport_rl import HexapodPushEnv, PushConfig

torch.set_num_threads(1)
runtime = dict(python=sys.version, torch=torch.__version__, mujoco=mujoco.__version__,
               numpy=np.__version__, cpu_count=os.cpu_count(), torch_threads=torch.get_num_threads(),
               rendering_backend=os.environ.get("MUJOCO_GL"))
Path(sys.argv[1]).write_text(json.dumps(runtime, indent=2) + "\n")
print(json.dumps(runtime, indent=2))
with HexapodPushEnv(PushConfig(shape="T"), flatten=False) as env:
    observation, info = env.reset(seed=42)
    action = np.zeros((2, 3), dtype=np.float32)
    next_observation, reward, terminated, truncated, info = env.step(action)
    print(json.dumps({"observation_shape": list(observation.shape), "action_shape": list(action.shape),
                      "team_reward": reward, "terminated": terminated, "truncated": truncated,
                      "reward_terms": info["reward_terms"]}, indent=2))
"""
        + "'''"
        + r""", RUN_DIR / "runtime.json", name="inspect_api.py")
print((RUN_DIR / "logs/inspect_api.txt").read_text())
    """
    )
    markdown("""
    ## Step 6 — 参考の学習済みモデルを再生する
    これは事前に**報酬だけで学習した参考方策**です。次の学習の教師データには使用しません。
    同じseedで実際にMuJoCoを動かし、動画を生成します。モデルを読み込むだけの静止画ではありません。
    640×480、5 fpsで影・反射を省いて描画し、物理更新は200 Hzを保ちます。
    """)
    code(r"""
    reference_dir = RUN_DIR / "reference"
    reference_result = reference_dir / "result.json"
    if not reference_result.exists():
        run_cli(["eval", "--checkpoint", PROJECT_DIR / "checkpoints/transport.pt", "--episodes", "1",
                 "--seed", "70000", "--output", reference_result, "--video-dir", reference_dir / "video",
                 "--video-width", "640", "--video-height", "480", "--video-fps", "5", "--fast-video"], "reference")
    reference = json.loads(reference_result.read_text())
    print("参考方策の成功:", reference["successes"], "/ 1")
    print("最終位置誤差 [m]:", reference["mean_final_distance_m"])
    display(Video(str(reference_dir / "video/seed_70000.mp4"), embed=True))
    """)
    markdown("""
    ## Step 7 — 初期の押す方策を準備する
    `START_FROM_SCRATCH=False` では、報酬で学習済みの押す方策を利用します。
    `True` では、押す方策をランダムな重みから学習した後、左右対称化と探索幅の調整を適用して追加学習します。
    歩行方策は固定です。両方の方法でデモ行動や模倣損失は使いません。
    """)
    code(r"""
    if START_FROM_SCRATCH:
        initial_push = train_phase("push_initial", "train-push", extra_args=("--epochs", "4", "--minibatch", "128"))
        refined_push = train_phase("push_refined", "train-push", initial_checkpoint=initial_push,
                                  extra_args=("--epochs", "8", "--minibatch", "256", "--mirror-equivariant", "--initial-log-std", "-1.5"), seed_offset=1)
        base_source = refined_push
    else:
        base_source = PROJECT_DIR / "checkpoints/base_pusher.pt"
    BASE_PUSHER = RUN_DIR / "base_pusher.pt"
    if BASE_PUSHER.exists():
        assert hashlib.sha256(BASE_PUSHER.read_bytes()).hexdigest() == hashlib.sha256(base_source.read_bytes()).hexdigest(), "初期pusherが変わりました。新しい実験名を使ってください。"
    else:
        shutil.copy2(base_source, BASE_PUSHER)
    print("初期pusher:", BASE_PUSHER)
    backup_to_drive()
    """)
    markdown("""
    ## Step 8 — 回り込みを学習する
    actorをランダムな重みからMAPPOで学習します。初期配置は `near → rear → side → front`。
    検証成功率が75%以上かつ一定の反復数を経過すると、次の難度へ進みます。
    報酬はTを避けて後方へ向かう距離の減少、接触、時間などから計算します。距離計算は行動の教師には使いません。
    両機が後方の担当位置と向きに整列した時点で、回り込みのエピソードを終了します。
    """)
    code(r"""
    NAVIGATOR = train_phase("navigation", "train", extra_args=("--pushing-checkpoint", str(BASE_PUSHER),
                            "--epochs", "4", "--minibatch", "128", "--validate-every", "50"), seed_offset=2)
    """)
    markdown("""
    ## Step 9 — 回り込み後の位置から押す動作を学習する
    回り込み完了位置のばらつきに適応する追加のMAPPOです。
    毎回の学習で回り込みを再生せず、その完了位置付近から始めて経験を集めます。
    初期配置の25%には、元の押す学習の配置も残します。
    """)
    code(r"""
    PUSHER = train_phase("handover", "train-handover", initial_checkpoint=BASE_PUSHER, seed_offset=3)
    """)
    markdown("""
    ## Step 10 — 検証用の初期配置でモデルを選ぶ
    回り込みの最終モデルと、保存されていれば前方検証の最良モデルを候補にします。
    押す方策と組み合わせ、**運搬全体**をseed 66000以降で評価します。
    成功数、胴体接触、回り込み中の機体同士の接触、最終位置誤差の順に選びます。
    このseed集合を、次のテスト集合と混ぜません。
    """)
    code(r"""
    candidates_dir = RUN_DIR / "selection"
    candidates_dir.mkdir(exist_ok=True)
    candidate_models = [NAVIGATOR]
    best_navigation = NAVIGATOR.parent / "best_front_checkpoint.pt"
    if best_navigation.exists():
        candidate_models.append(best_navigation)
    candidates = []
    for index, navigation in enumerate(candidate_models):
        model = candidates_dir / f"candidate_{index:02d}.pt"
        if not model.exists():
            run_cli(["compose", "--navigation-checkpoint", navigation, "--pushing-checkpoint", PUSHER, "--output", model], f"compose_{index:02d}")
        result = candidates_dir / f"candidate_{index:02d}.json"
        if not result.exists():
            run_cli(["eval", "--checkpoint", model, "--episodes", str(VALIDATION_EPISODES), "--workers", str(NUM_ENVS),
                     "--seed", "66000", "--seconds", str(EVAL_SECONDS), "--output", result], f"validation_{index:02d}")
        report = json.loads(result.read_text())
        candidates.append((model, report))
    selected, validation = max(candidates, key=lambda item: (item[1]["successes"], -item[1]["body_contact_episodes"],
                              -item[1]["navigation_robot_contact_episodes"], -item[1]["mean_final_distance_m"]))
    SELECTED_CHECKPOINT = selected
    selection = {"checkpoint": str(selected.relative_to(RUN_DIR)), "validation_seed": 66000,
                 "validation_episodes": VALIDATION_EPISODES, "successes": validation["successes"],
                 "candidates": [dict(checkpoint=str(model.relative_to(RUN_DIR)), successes=report["successes"],
                                     mean_final_distance_m=report["mean_final_distance_m"]) for model, report in candidates]}
    (RUN_DIR / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(json.dumps(selection, indent=2))
    """)
    markdown("""
    ## Step 11 — 未使用の初期配置で評価する
    前方はseed 80000以降、側方は81000以降を使います。Tの初期yaw角は目標方向に対して±30度です。
    `quick` の評価は10秒で打ち切ります。`research` の100秒評価と混ぜて比較しないでください。
    成功しなかった試行も、接触・転倒・誤差を含めて記録します。
    """)
    code(r"""
    test_reports = {}
    for layout, seed in (("front", 80000), ("side", 81000)):
        output = RUN_DIR / f"test_{layout}.json"
        if not output.exists():
            run_cli(["eval", "--checkpoint", SELECTED_CHECKPOINT, "--layout", layout,
                     "--seed", str(seed), "--episodes", str(EVAL_EPISODES), "--workers", str(NUM_ENVS),
                     "--seconds", str(EVAL_SECONDS), "--output", output], f"test_{layout}")
        test_reports[layout] = json.loads(output.read_text())
    print({layout: dict(successes=report["successes"], episodes=len(report["episodes"]),
                        handovers=report["handovers"], body_contact_episodes=report["body_contact_episodes"],
                        falls=report["falls"]) for layout, report in test_reports.items()})
    """)
    markdown("""
    ## Step 12 — 学習曲線と論文用の評価表を作る
    学習中のsuccess rateはその時点のカリキュラムの初期配置での値です。運搬全体のテスト成功率とは異なります。
    完了エピソードがない時点は、成功率を描画しません。
    成功率、接触、転倒、位置・yaw誤差、時間をCSVにまとめ、成功率のWilson 95%区間も表示します。
    区間は初期配置の試行に対するもので、異なる学習seedのばらつきではありません。卒論では学習seedを変えて複数回実行してください。
    """)
    code(
        r"""
    run_python(r"""
        + "'''"
        + r"""
from pathlib import Path
import csv
import json
import math
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

run = Path(sys.argv[1])
fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
for column, phase in enumerate(("navigation", "handover")):
    by_iteration = {}
    for log in sorted((run / phase).glob("attempt_*/metrics.jsonl")):
        for line in log.read_text().splitlines():
            row = json.loads(line)
            by_iteration[row["iteration"]] = row
    records = [by_iteration[key] for key in sorted(by_iteration)]
    settings = json.loads((run / "experiment.json").read_text())
    first = records[0]["transitions"] - settings["num_envs"] * settings["horizon"]
    steps = [row["transitions"] - first for row in records]
    axes[0, column].plot(steps, [row["mean_reward"] for row in records], marker=".", label="Training reward")
    key = "training_success_rate" if phase == "navigation" else "success_rate"
    measured_success = [row[key] if row["completed_episodes"] else float("nan") for row in records]
    axes[1, column].plot(steps, measured_success, marker=".", label="Training success")
    validated = [row for row in records if "validation_success_rate" in row]
    if validated:
        axes[1, column].plot([row["transitions"] - first for row in validated], [row["validation_success_rate"] for row in validated], marker="o", label="Validation success")
    for axis in axes[:, column]:
        axis.set_title(phase)
        axis.set_xlabel("Team transitions since phase start")
        axis.grid(alpha=0.3)
        axis.legend()
    axes[0, column].set_ylabel("Mean reward per step")
    axes[1, column].set_ylabel("Episode success rate")
    axes[1, column].set_ylim(-0.05, 1.05)
    for row in records:
        if "advanced_to" in row:
            for axis in axes[:, column]:
                axis.axvline(row["transitions"] - first, color="gray", linestyle=":")
fig.savefig(run / "learning_curves.png", dpi=180)
fig.savefig(run / "learning_curves.pdf")
plt.close(fig)

rows = []
for layout in ("front", "side"):
    report = json.loads((run / f"test_{layout}.json").read_text())
    episodes = report["episodes"]
    count = len(episodes)
    proportion = report["successes"] / count
    z = 1.96
    denominator = 1 + z * z / count
    center = (proportion + z * z / (2 * count)) / denominator
    half = z * math.sqrt(proportion * (1 - proportion) / count + z * z / (4 * count * count)) / denominator
    rows.append(dict(layout=layout, mode=settings["mode"], training_seed=settings["training_seed"],
                     episodes=count, success_rate=proportion, success_ci95_low=center - half, success_ci95_high=center + half,
                     handover_rate=report["handovers"] / count, body_contact_episodes=report["body_contact_episodes"],
                     robot_contact_episodes=report["robot_contact_episodes"], navigation_robot_contact_episodes=report["navigation_robot_contact_episodes"],
                     falls=report["falls"], mean_distance_m=report["mean_final_distance_m"],
                     mean_yaw_error_rad=sum(row["yaw_error"] for row in episodes) / count,
                     mean_elapsed_seconds=sum(row["elapsed_seconds"] for row in episodes) / count))
with (run / "evaluation.csv").open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(run / "evaluation_table.json").write_text(json.dumps(rows, indent=2) + "\n")
"""
        + "'''"
        + r""", RUN_DIR, name="make_figures.py")
import html
display(Image(filename=str(RUN_DIR / "learning_curves.png")))
rows = json.loads((RUN_DIR / "evaluation_table.json").read_text())
table = "<table><tr><th>配置</th><th>運搬成功率</th><th>95%区間</th><th>回り込み成功率</th><th>胴体接触</th><th>転倒</th><th>位置誤差 [m]</th></tr>"
for row in rows:
    table += (f"<tr><td>{html.escape(row['layout'])}</td><td>{row['success_rate']:.1%}</td>"
              f"<td>{row['success_ci95_low']:.1%}–{row['success_ci95_high']:.1%}</td>"
              f"<td>{row['handover_rate']:.1%}</td><td>{row['body_contact_episodes']}</td>"
              f"<td>{row['falls']}</td><td>{row['mean_distance_m']:.3f}</td></tr>")
display(HTML(table + "</table>"))
print("保存:", RUN_DIR / "evaluation.csv", RUN_DIR / "learning_curves.pdf")
    """
    )
    markdown("""
    ## Step 13 — 自分で学習した方策を録画する
    参考モデルを使い回さず、Step 10で選んだ**自分のモデル**を実行します。
    seedは82000で固定し、成功した試行だけを探す選び方はしません。
    `quick` は後方近くから10秒、`research` は前方から100秒を上限にします。失敗した動画も結果として確認します。
    """)
    code(r"""
    learned_dir = RUN_DIR / "learned_video"
    learned_result = learned_dir / "result.json"
    if not learned_result.exists():
        run_cli(["eval", "--checkpoint", SELECTED_CHECKPOINT, "--episodes", "1", "--seed", "82000",
                 "--layout", "near" if MODE == "quick" else "front", "--seconds", str(EVAL_SECONDS),
                 "--output", learned_result, "--video-dir", learned_dir / "video",
                 "--video-width", "640", "--video-height", "480", "--video-fps", "5", "--fast-video"], "learned_video")
    learned = json.loads(learned_result.read_text())
    print("自分の方策の運搬成功:", learned["successes"], "/ 1")
    display(Video(str(learned_dir / "video/seed_82000.mp4"), embed=True))
    """)
    markdown("""
    ## Step 14 — 実験結果を保存する
    設定、実行時のソース・資産、ライブラリの版、重み、optimizer、学習ログ、検証とテストの結果、図、CSV、動画をZIPへまとめます。
    Colabではダウンロードします。Driveを選んだ場合はバックアップも更新します。
    再開時は同じノートブック・実験名・設定で `RESTORE_RESULTS=True` を選び、結果ZIPを指定してください。
    """)
    code(r"""
    backup_to_drive()
    archive_path = PROJECT_DIR / "runs" / f"{EXPERIMENT_ID}_results.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(RUN_DIR.rglob("*")):
            if path.is_file() and path.suffix not in (".tmp", ".partial"):
                archive.write(path, path.relative_to(RUN_DIR))
    print("結果ZIP:", archive_path, f"({archive_path.stat().st_size / 1024**2:.1f} MiB)")
    if IN_COLAB:
        colab_files.download(str(archive_path))
    """)
    markdown("""
    ## 卒論で次に行う比較実験

    1. `research` で学習seedを変え、別の実験名で少なくとも3回実行する。
    2. 同じテストseed集合で、報酬係数・カリキュラム・初期配置など1条件ずつ変えて比較する。
    3. 運搬成功率だけでなく、胴体接触・機体同士の接触・転倒・位置/yaw誤差・所要時間も報告する。
    4. 回り込み環境は2台専用で、障害物・観測誤差・実機条件は含まないことを論文で明示する。

    編集場所は展開した `docs/code-guide.md` を参照してください。回り込みの報酬は `ApproachRewardWeights`、押す報酬は `PushRewardWeights`、学習設定は `PPOSettings` にあります。
    ソースを編集した後は新しい実験名で別プロセスの学習を開始してください。変更内容も論文・記録へ残してください。

    Colabの実行環境には時間・資源の制限があり、ランタイム内の未保存ファイルは失われることがあります。
    [Colab公式FAQ](https://research.google.com/colaboratory/faq.html) / [uvのPython管理](https://docs.astral.sh/uv/guides/install-python/) / [MuJoCoのPython API](https://mujoco.readthedocs.io/en/stable/python.html)
    """)
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {"name": OUTPUT.name, "provenance": []},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python", "version": "3.12"},
            "hexapod_transport_rl": {
                "repository": REPOSITORY,
                "mode_default": "quick",
            },
        },
        "cells": cells,
    }
    for index, cell in enumerate(cells):
        cell["id"] = f"lesson_{index:02d}"
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
    print(f"Built {OUTPUT}: {len(cells)} cells, {OUTPUT.stat().st_size / 1024:.1f} KiB")


if __name__ == "__main__":
    main()
