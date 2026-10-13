"""Build a short introductory MAPPO crossing lesson with ordinary imports."""

import json
from pathlib import Path

from build_colab import CELLS as PUSH_CELLS

ROOT = Path(__file__).resolve().parents[1]
CELLS = [
    (
        "markdown",
        """# 複数の六足ロボットに、すれ違いを学習させる

向かい合うロボットが、相手を避けながら反対側の目的地へ進む課題です。
接触で物を運ぶ前に、**複数台の観測・行動・共有報酬とMAPPO**を理解します。
歩行モデルは固定し、速度コマンドを選ぶ上位方策を毎回ゼロから学習します。
教師デモ・模倣学習・決められた回避経路は使いません。

**歩行コマンド → Gym API → 報酬 → 新規学習 → 評価・動画**の順で進めます。
最初は2台。次に台数を4台にして、新しい方策を学習します。""",
    ),
    *PUSH_CELLS[1:6],
    (
        "markdown",
        """## 3. 複数台の環境APIを使う

`reset()`で初期配置を作り、`step(action)`で全ロボットを0.4秒進めます。
観測は各機について**自分の目標への相対位置、自分の速度、直前の行動、相手の相対位置・速度**です。
すべて機体座標なので、世界の向きが変わっても同じ意味で読めます。
2台で `(2,11)`、4台で `(4,19)` です。

行動は `(台数,3)` の `[前進, 横移動, 旋回]`、各成分−1〜1。
歩行コマンドの物理的な範囲へ変換します。ゼロは停止です。
下の短い例では乱数を送ります。後で学習したコマンドに置き換えます。
緑の円が目的地で、中心から30 cm以内へ全機が到着し、0.8秒留まると成功です。""",
    ),
    (
        "code",
        """from hexapod_transport_rl import NavigationConfig, NavigationEnv

config = NavigationConfig(num_robots=2)
env = NavigationEnv(config, render_mode="rgb_array_list")
observation, info = env.reset(seed=42)
print("観測:", observation.shape, "行動:", env.action_space.shape)
rng = np.random.default_rng(42)
for _ in range(10):
    action = rng.uniform(-0.3, 0.3, (config.num_robots, 3)).astype(np.float32)
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
frames = env.render()
env.close()
media.show_video(frames, fps=25)
""",
    ),
    (
        "markdown",
        """## 4. 報酬を決める

全機に同じチーム報酬を渡します。目標距離の改善、近づきすぎ、実際の接触、到着を組み合わせます。
`progress`を大きくすると接近を、`separation`と`collision`を大きくすると安全な間隔を重視します。
まず標準設定で学習し、次にこの2項を0にした条件を別の名前・同じseedで試して比較してください。
ロボットが目的地へ向かう経路や行動を計算して教師に与える処理はありません。""",
    ),
    (
        "code",
        """from hexapod_transport_rl import NavigationRewardWeights

reward = NavigationRewardWeights(progress=20, separation=2, collision=1)
config = NavigationConfig(num_robots=2, reward_weights=reward)
""",
    ),
    (
        "markdown",
        """## 5. ゼロからMAPPOで学習する

TorchRLのMAPPOを使い、各機で共有するActorと、チーム全体を見るCriticを学習します。
毎回新しい重み・Adam・価値統計から始めます。以前の運搬モデルは読みません。
1条件の学習に240秒を指定します。ロールアウト単位の目安で、厳密な締切ではありません。
T4の速度と成功率は実測が必要です。時間内の成功は保証しません。""",
    ),
    (
        "code",
        """import time
from hexapod_transport_rl import train_mappo
from hexapod_transport_rl.experiments import create_experiment

NAME = "crossing_safe"
SEED = 42
experiment_started = time.monotonic()
RUN_DIR = create_experiment(PROJECT_DIR, NAME, unique=True)
checkpoint = train_mappo(config, RUN_DIR, seed=SEED, max_seconds=240)
""",
    ),
    (
        "markdown",
        """## 6. 到着・安全性と動画で評価する

学習で使っていない初期配置を8例評価し、同じ最初の例を25 fpsで記録します。
到達率だけでなく、接触なしで成功した件数と平均接触時間も比較します。
動画は補間した軌跡ではなく、MuJoCoの実際の歩行フレームです。
数値評価はGPU、記録はCPU版MuJoCoなので結果が少し異なる場合があります。
学習中は探索のため行動をサンプリングし、評価は方策の平均行動を使います。成功数は一致するとは限りません。""",
    ),
    (
        "code",
        """import pandas as pd
from hexapod_transport_rl import evaluate_mappo

report = evaluate_mappo(checkpoint, episodes=8, seed=98000)
display(pd.DataFrame([{
    "成功率": report["success_rate"],
    "接触なしで成功した件数": report["collision_free_successes"],
    "平均最終目標距離 [m]": report["mean_final_distance_m"],
    "平均接触時間 [s]": report["mean_contact_seconds"],
}]))
media.show_video(media.read_video(report["video_path"]), fps=25)
print("学習・評価・動画の合計 [s]:", time.monotonic() - experiment_started)
""",
    ),
    (
        "markdown",
        """## 7. 条件を変えて比較する

1. 安全項の有無で、成功率・接触時間・すれ違い方がどう変わるか。
2. 台数を2台から4台に変えるとどうなるか。観測とCriticの大きさも変わるので、新規学習が必要です。
3. 成功しやすくなったら`goal_tolerance`を小さくして、精密な停止を課題に追加する。

一度に変更する要因は1つにし、同じ評価seedを使います。方策の学習seedも複数試して再現性を確認します。
この課題で複数台の制御を理解してから、T押しを発展課題として扱えます。""",
    ),
    (
        "code",
        """import matplotlib.pyplot as plt
from hexapod_transport_rl.experiments import archive_results

progress = pd.read_csv(RUN_DIR / "progress.csv")
progress.plot(x="elapsed_seconds", y="mean_distance_m", legend=False)
plt.xlabel("Training time [s]")
plt.ylabel("Mean goal distance [m]")
plt.show()
print("結果・設定・コードのZIP:", archive_results(RUN_DIR))
""",
    ),
]


def main():
    cells = []
    for index, (kind, source) in enumerate(CELLS):
        cell = dict(
            cell_type=kind, id=f"navigation-{index:02d}", metadata={}, source=source
        )
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
    (ROOT / "notebooks/cooperative_navigation_colab.ipynb").write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
    )


if __name__ == "__main__":
    main()
