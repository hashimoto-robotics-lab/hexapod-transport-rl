"""Build the compact, one-condition student Colab notebook."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CELLS = [
    (
        "markdown",
        r"""# 六足ロボットの協調運搬を強化学習で調べる

複数台でTを運ぶには、押す場所・向き・タイミングを協調させる必要があります。
まず学習済みの歩行モデルに指令を送り、次に運搬方策を**ランダムな重みから**学習します。
教師デモ・模倣学習は使いません。ロボットは「T → ゴール → ロボット」の配置から始めます。

1回の実験は **報酬を決める → 新規学習 → 数値評価 → 動画1本** です。
初回の環境準備を除き10分以内を目標に、学習には360秒の予算を指定します。
時間内に95%の成功へ到達することは保証しません。成功・停滞・失敗を同じ指標で比較します。
[仕様と公式Push-Tとの差分](https://github.com/hashimoto-robotics-lab/hexapod-transport-rl/blob/main/docs/pusht-protocol.md)も参照できます。""",
    ),
    (
        "markdown",
        r"""## 1. 準備
Colabの「ランタイム → ランタイムのタイプを変更」で **T4 GPU** を選び、次の2セルを実行します。
公開GitHubから取得するので認証は不要です。取得済みの教材と学生の編集は上書きしません。
初回のインストール・カーネルのコンパイルは、以後の実験と分けて記録します。""",
    ),
    (
        "code",
        r"""from pathlib import Path
import subprocess
import sys

PROJECT_DIR = Path("/content/hexapod_transport_rl")
if not PROJECT_DIR.exists():
    subprocess.run([
        "git", "clone", "--depth", "1", "--branch", "main",
        "https://github.com/hashimoto-robotics-lab/hexapod-transport-rl.git", str(PROJECT_DIR)
    ], check=True)
sys.path.insert(0, str(PROJECT_DIR / "tools"))
""",
    ),
    (
        "code",
        r"""from colab_runtime import prepare_colab

prepare_colab(PROJECT_DIR)
""",
    ),
    (
        "markdown",
        r"""## 2. 学習済みロボットにコマンドを送る
`reset()`で開始、`step(action)`で0.2秒進め、`render()`で画像を得ます。
歩行の行動は `(台数,3)` の `[vx, vy, yaw_rate]`。機体座標の速度 [m/s, m/s, rad/s] です。
範囲は前進−0.15〜0.20、横移動±0.10、旋回±0.60。ゼロは停止指令です。
ここでは新しい学習を行いません。コマンドを1つ変えて、位置・向きと動画の違いを確認します。""",
    ),
    (
        "code",
        r"""import gymnasium as gym
import numpy as np
import mediapy as media
import hexapod_transport_rl  # Gymnasiumに教材の環境を登録する

env = gym.make("HexapodWalking-v0", num_robots=1, episode_seconds=9.4, render_mode="rgb_array")
observation, info = env.reset(seed=42)
frames = [env.render()]
for steps, command in [
    (4, [0.0, 0.0, 0.0]),
    (15, [0.12, 0.0, 0.0]),
    (12, [0.0, 0.06, 0.0]),
    (12, [0.0, 0.0, 0.40]),
    (4, [0.0, 0.0, 0.0]),
]:
    action = np.array([command], dtype=np.float32)
    for _ in range(steps):
        observation, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render())
        if terminated or truncated:
            break
    print("位置 [m]:", info["positions"], "向き [rad]:", info["headings"])
    if terminated or truncated:
        break
env.close()
media.show_video(frames, fps=5)
""",
    ),
    (
        "markdown",
        r"""## 3. 台数を増やす
行動の0行目が0番、1行目が1番のロボットです。
全機を1回の`step()`で同時に動かします。同じ歩行モデルでも、別々の指令を送れます。""",
    ),
    (
        "code",
        r"""env = gym.make("HexapodWalking-v0", num_robots=4, episode_seconds=4.6, render_mode="rgb_array")
observation, info = env.reset(seed=42)
frames = [env.render()]
commands = np.array([
    [0.10, 0.00, 0.00],
    [0.05, 0.00, 0.00],
    [0.08, 0.04, 0.00],
    [0.00, 0.00, 0.30],
], dtype=np.float32)
for step in range(23):
    action = commands if 4 <= step < 19 else np.zeros((4, 3), dtype=np.float32)
    observation, reward, terminated, truncated, info = env.step(action)
    frames.append(env.render())
    if terminated or truncated:
        break
print("4台の位置 [m]:", info["positions"])
print("4台の向き [rad]:", info["headings"])
env.close()
media.show_video(frames, fps=5)
""",
    ),
    (
        "markdown",
        r"""## 4. Tを押すGym API
運搬の行動は `(台数,2)` の **世界座標のXY目標 [m]**。`[0,0]`は原点への移動です。
その場に留めるには現在位置を指定します。位置追従は自分の状態とXY目標だけを見ます。

状態観測は各機 `[自分x,y, Tのx,y, Tの角度, 自分の角度, 他機のx,y,角度,…]`。
先頭5成分が公式Push-Tに対応し、2台で各機9成分、4台で15成分です。
目標位置・向きは固定です。キーポイント版を使う場合は`obs_type="keypoints"`にします。
成功は**目標との面積重なり率95%超**。転倒・領域外は失敗、制限時間は時間切れです。""",
    ),
    (
        "code",
        r"""from hexapod_transport_rl import PushTConfig

config = PushTConfig(num_robots=2, obs_type="state")
env = gym.make("HexapodPushT-v0", config=config, render_mode="rgb_array_list")
observation, info = env.reset(seed=42)
print("観測:", observation, "行動:", env.action_space)
print("固定ゴール [x, y, yaw]:", info["goal_pose"])
print("初期の重なり率:", info["coverage"])
frames = env.render()
targets = np.asarray(info["agent_pos"], dtype=np.float32)
for step in range(20):
    action = targets if step < 5 else targets + np.array([-0.15, 0.15], dtype=np.float32)
    observation, reward, terminated, truncated, info = env.step(action)
    frames.extend(env.render())
    if terminated or truncated:
        break
print("重なり率:", info["coverage"], "報酬:", reward, "内訳:", info["reward_terms"])
env.close()
media.show_video(frames, fps=25)
""",
    ),
    (
        "markdown",
        r"""## 5. 報酬を決める
公式の基準は `clip(coverage / 0.95, 0, 1)` です。
`coverage`は **Tと目標の交差面積÷目標の面積**。IoUとは異なります。

高速実験では、次の研究用の追加項を試します。公式報酬だけの条件と区別して記録します。

| 係数 | 何を評価するか |
|---|---|
| `coverage` | 毎ステップの重なり率 |
| `progress` | 重なり率の増加量 |
| `approach` | Tを避けて押す側へ近づいた距離 |
| `success` | 95%を超えたときの報酬 |
| `time` / `body_contact` | 時間・胴体接触のコスト |

回り込みの距離は状態の評価値です。経路・教師行動を方策へ渡しません。
例の係数は成功を保証しません。「何が変わると予想するか」を書き、1項ずつ変更してください。""",
    ),
    (
        "code",
        r"""from hexapod_transport_rl import PushTConfig, PushTRewardWeights

reward = PushTRewardWeights(
    coverage=0.0, progress=10.0, approach=1.0, success=10.0,
    time=0.02, body_contact=0.5, failure=-10.0,
)
# 公式報酬だけを試す場合: reward = PushTRewardWeights()
config = PushTConfig(num_robots=2, reward_weights=reward, high_level_decimation=20)""",
    ),
    (
        "markdown",
        r"""## 6. 1条件をゼロから学習する
`train_mappo()`は毎回、運搬actor・critic・Adam・価値統計を新しく作ります。
GPUでは512世界を並列に使います。歩行モデルは固定です。
XY目標は0.8秒保ち、歩行は25 Hzで動きます。短い指令の切り替えで歩行が乱れるのを抑える設定です。
方策は現在位置を基準にXY目標を探索します。観測はメートル・ラジアンのままです。

TorchRLの **収集 → GAE → ミニバッチ更新 → 記録** は
[`training.py`](https://github.com/hashimoto-robotics-lab/hexapod-transport-rl/blob/main/src/hexapod_transport_rl/training.py) にあります。
学習は通常のPython関数として呼び出します。
予算は環境構築・学習を含み、ロールアウトの境界で終了します。10分の厳密なタイマーではありません。""",
    ),
    (
        "code",
        r"""import time
from hexapod_transport_rl import train_mappo
from hexapod_transport_rl.experiments import create_experiment

EXPERIMENT_NAME = "progress_reward"
TRAINING_SEED = 42
MAX_TRAINING_SECONDS = 360
started = time.perf_counter()
RUN_DIR = create_experiment(PROJECT_DIR, EXPERIMENT_NAME, unique=True)
checkpoint = train_mappo(
    config, RUN_DIR, seed=TRAINING_SEED, max_seconds=MAX_TRAINING_SECONDS,
)""",
    ),
    (
        "markdown",
        r"""## 7. 数値と動画で評価する
学習に使わない開発用8配置で、成功率・最大重なり率・接触・転倒を確認します。
数値評価はGPU、動画1本は同じseedのCPU MuJoCoです。数値精度により軌跡が異なることがあります。
映像は25 fps・実時間です。総報酬の大小だけで、異なる報酬条件の性能を比べません。""",
    ),
    (
        "code",
        r"""import pandas as pd
from hexapod_transport_rl import evaluate_mappo

report = evaluate_mappo(checkpoint, episodes=8, seed=98000)
display(pd.DataFrame([{
    "成功率": report["success_rate"],
    "最大重なり率の平均": report["mean_max_coverage"],
    "胴体接触の件数": report["body_contact_episodes"],
    "転倒の件数": report["falls"],
}]))
media.show_video(media.read_video(report["video_path"]), fps=25)
print("学習・評価・動画確認:", round((time.perf_counter() - started) / 60, 2), "分")""",
    ),
    (
        "code",
        r"""import matplotlib.pyplot as plt

curve = pd.read_csv(RUN_DIR / "progress.csv")
curve = curve.astype({"elapsed_seconds": float, "mean_coverage": float})
curve.plot(x="elapsed_seconds", y="mean_coverage", ylim=(0, 1), legend=False)
plt.xlabel("Training wall time [s]")
plt.ylabel("Mean coverage in training")
plt.tight_layout()
plt.savefig(RUN_DIR / "learning_curve.png", dpi=120)
plt.show()""",
    ),
    (
        "markdown",
        r"""## 8. 保存して次の実験へ
コード・資産・設定・初期重み・学習ログ・評価・動画をZIPで保存します。
報酬のセルを変更し、6〜8を再実行すると、別のフォルダーに**ゼロから**学習します。

卒論の比較では、同じ台数・配置・seed・学習ステップ数を揃えます。
時間制限で止まるとステップ数は変わるため、`run.json`も確認します。
固定ステップ数で比べる場合は、試運転で予算を決め、`train_mappo(..., steps=262144, max_seconds=None)`等を使います。
開発後に条件を固定し、3種類以上の学習seed・未使用50配置以上で評価します。
4台は`num_robots=4`で新しく学習します。4台の性能とT4の時間は別途測定します。""",
    ),
    (
        "code",
        r"""from hexapod_transport_rl.experiments import archive_results

archive = archive_results(RUN_DIR)
print("保存したZIP:", archive)""",
    ),
]


def main():
    cells = []
    for index, (kind, source) in enumerate(CELLS):
        cell = dict(cell_type=kind, id=f"cell-{index:02d}", metadata={}, source=source)
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = dict(
        cells=cells,
        nbformat=4,
        nbformat_minor=5,
        metadata=dict(
            kernelspec=dict(display_name="Python 3", language="python", name="python3"),
            language_info=dict(name="python", version="3.12"),
            colab=dict(provenance=[]),
        ),
    )
    output = ROOT / "notebooks/hexapod_transport_rl_colab.ipynb"
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
