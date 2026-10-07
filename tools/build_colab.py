"""Build the research notebook from its readable cell definitions."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CELLS = [
    (
        "markdown",
        r"""# 六足ロボットの協調運搬：報酬設計と比較実験

## 研究のモチベーション
複数のロボットで物資を運ぶには、荷物の周囲での移動、押す向き、相手との接触を考える必要があります。
この研究では、**六足ロボットがT字物体に回り込み、脚を当てて目標へ押す協調動作**を強化学習します。
機体の色は元のロボットと同じ、床にはグリッドを表示します。胴体に押すための突っ張りは追加しません。

このノートブックの目的は、研究の仕組みとAPIを理解し、**報酬を変えると挙動がどう変わるかを検証すること**です。
研究の問いの例：「機体同士の接触ペナルティを強くすると、安全に回り込めるか。それとも動かなくなるか？」

## 学習済みロボットへコマンドを送るところから始める
歩行モデルはすでに学習済みです。前後・左右・旋回の速度指令を受け取り、18関節の目標角度を出します。
この歩行モデルは固定し、上位方策が各ロボットへ送る速度指令を学習します。

```text
Tとロボットの状態 → 上位方策 → 各機の速度指令 → 固定した歩行モデル → 脚・床・Tの接触
        ↑                                                        ↓
        └──────────────── 次の状態とチーム報酬 ────────────────────┘
```

上位方策は **Stable-Baselines3のPPO** で学習します。
各機の観測だけを見るactorの重みを共有し、criticはチーム全体の観測を見ます。
このネットワーク構造を `SharedTeamPolicy` で指定し、経験収集・GAE・PPO更新は外部ライブラリに任せます。
デモを教師にせず、報酬から学びます。

同梱の参考モデルは以前のMAPPO実装で学習したものです。
SB3はチームの結合行動の確率比を使い、旧実装は各機の比を個別にclipします。
また、SB3はGaussian行動を範囲内へclipし、旧実装はtanh変換します。
**ネットワーク構造は共通でも更新方法が違うため、同じ学習結果になる保証はありません。**
まず2台で報酬の比較実験を行います。歩行APIは1〜4台、押す環境は2〜4台に対応しますが、
**この回り込みカリキュラムと同梱の運搬モデルは2台用**です。4台の運搬性能は、別途環境拡張と学習で検証します。

上から順に、準備 → 歩行API → 強化学習環境API → 報酬 → 比較学習 → 定量評価・動画、を実行してください。
""",
    ),
    (
        "markdown",
        r"""## 1. 教材と実行環境を準備する
公開GitHubから取得します。認証は不要です。次の2セルはそのまま実行します。
既に取得したコードがある場合は、学生の編集を残すため再取得しません。新しい教材を使うときは新規ランタイムで始めます。
""",
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
        r"""## 2. 学習済みの1台に速度コマンドを送る
歩行・回り込み・押す環境を、同じGymnasiumのAPIで操作します。
`gym.make()` で環境を作り、`reset()` で開始、`step(action)` で0.2秒進め、`render()` で画像を取得します。
歩行環境の行動は、各機の機体座標の `[vx, vy, yaw_rate]` を並べた `(台数, 3)` の配列です。
このセルでは学習済み歩行モデルに指令を送るだけで、新しい学習は行いません。

| 引数 | 単位 | 範囲 | 正の方向 |
|---|---|---|---|
| `vx` | m/s | −0.15〜0.20 | 前進 |
| `vy` | m/s | −0.10〜0.10 | 左移動 |
| `yaw_rate` | rad/s | −0.60〜0.60 | 左旋回 |

まず速度を1つ変え、実際の移動を観察してください。目標速度と実速度には接触や姿勢による差があります。
1ステップは0.2秒なので、15ステップで3秒間の前進です。
行動は毎ステップ渡します。ゼロ行動は停止指令ですが、実際の減速には時間がかかります。
`frames.append(env.render())` でRGB画像を集め、mediapyで5 fpsの動画として表示します。
歩行の観測は `(台数, 6)` の `[世界x, 世界y, 向き, 前回vx指令, 前回vy指令, 前回旋回指令]` です。
`info` には位置・向き・指令・経過時間が入ります。歩行体験用の報酬は常に0、転倒で終了、制限時間で時間切れになります。
[MuJoCo公式チュートリアル](https://github.com/google-deepmind/mujoco/blob/main/python/tutorial.ipynb) と同じように、mediapyで表示します。
""",
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
        r"""## 3. 複数台に独立したコマンドを送る
行動配列の0行目が0番、1行目が1番のロボットです。
`step(action)` は全機の物理シミュレーションを同時に進めます。
同じ歩行モデルを各機が使っていても、送る速度を変えると別々に動きます。
ここでの指令は手で指定しています。後の学習では上位方策が指令を決めます。
""",
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
        r"""## 4. 強化学習環境のAPIを理解する
運搬を「押す開始位置までの回り込み」と「目標まで押す」の2段階に分けています。
この比較実験では**回り込みを学習し、押す方策は全条件で同じ学習済みモデルに固定**します。
これにより、回り込み報酬を変えた影響を調べます。

| 環境 | 観測の形 | 行動の形 | 成功条件 |
|---|---|---|---|
| `HexapodApproach-v0` | `(2, 10)` | `(2, 3)` | 両機が担当する押す位置・向きに整列 |
| `HexapodPush-v0` (`flatten=False`) | `(台数, 16 + 4×(台数−1))` | `(台数, 3)` | Tの位置・向きが目標の許容範囲内 |

回り込みの各機の観測10成分は、T基準の位置2、向きのsin/cos 2、機体速度3、前回指令3です。
担当する左右の位置を同じ役割として扱うため、左右方向を反転して共有actorへ入力します。
回り込みの行動もこの反転座標の `[前後, 左右, 旋回]` で、各成分は−1〜1です。
環境内部で各機の実際の方向に戻し、歩行APIと同じ物理速度の範囲へ変換します。

**2台でも1つの相互作用する環境**として扱います。
`step()` には2台分の行動をまとめて渡し、報酬はチーム全体の1つの値です。
歩行環境は物理速度、学習用の回り込み・押す環境は−1〜1の正規化行動を使います。
`action_space` でそれぞれの範囲を確認できます。

`reset(seed=...)` が初期状態を作り、`step(action)` が0.2秒進めます。
戻り値は `observation, reward, terminated, truncated, info` です。
`terminated` は成功や転倒などの終了、`truncated` は時間切れです。どちらかがTrueなら、そのエピソードを終えます。
このセルでは1エピソードを2秒間進め、報酬の内訳を見てから動画を表示します。
終了後に続けるには `reset()`、使い終わったら `env.close()` で描画資源を解放します。
`info["reward_terms"]` は**このステップで実際に返した報酬の内訳**です。
""",
    ),
    (
        "code",
        r"""from hexapod_transport_rl import ApproachConfig

config = ApproachConfig(layout="front", seconds=2.0)
env = gym.make("HexapodApproach-v0", config=config, render_mode="rgb_array")
observation, info = env.reset(seed=80000)
print("観測:", env.observation_space, "行動:", env.action_space)
frames = [env.render()]
terminated = truncated = False
while not (terminated or truncated):
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    observation, reward, terminated, truncated, info = env.step(action)
    frames.append(env.render())
    print("チーム報酬:", reward, "内訳:", info["reward_terms"])
print("終了理由:", info["termination_reason"])
env.close()
media.show_video(frames, fps=5)
""",
    ),
    (
        "markdown",
        r"""## 5. 報酬の意味と、変更する場所を理解する
回り込みの報酬は次の項の合計です。ペナルティ係数は正の値で指定し、式の中で減算します。

| 項 | 観測する量 | 既定の係数 |
|---|---|---|
| `progress` | Tを避ける経路距離の減少（m） | 6 |
| `time` | 1ステップの時間（秒） | 0.04 |
| `robot_contact` | 機体同士が接触した時間（秒） | 4 |
| `body_contact` | 胴体がTに接触した時間（秒、2台平均） | 6 |
| `cargo_displacement` | 回り込み開始時からTが動いた距離（m、毎ステップ） | 0.1 |
| `success` / `failure` | 整列成功 / 転倒等の失敗時の一度の報酬 | +15 / −10 |

経路距離はTの周囲を通る距離で、直線距離だけを縮めて胴体を押し当てることを避けます。
これは状態から計算する報酬であり、教師の行動ではありません。

次のセルは**ライブラリが実際に学習で使う関数**を表示します。
最初は係数を変更しましょう。報酬の項や式そのものを研究する場合は
`src/hexapod_transport_rl/rewards.py` のこの関数を編集して、新規ランタイム・新規実験で学習します。
""",
    ),
    (
        "code",
        r"""import inspect
from hexapod_transport_rl.rewards import approach_reward_terms

print(inspect.getsource(approach_reward_terms))
""",
    ),
    (
        "markdown",
        r"""## 6. 仮説を立て、報酬を1項目だけ変える
例：「接触ペナルティを3倍にすると、機体同士の接触が減る」。
接触を避けるために遠回りしたり、動かなくなる可能性もあります。
成功率・時間・動画も見て、仮説を検証しましょう。

以下の `robot_contact=12.0` が学生の編集箇所です。
`replace()` は既定の設定の1項目だけを変更し、元の設定を残します。
報酬設定は環境ごとに持つため、他の条件や並列ワーカーへ混ざりません。
""",
    ),
    (
        "code",
        r"""from dataclasses import replace
from hexapod_transport_rl import ApproachRewardWeights

baseline_reward = ApproachRewardWeights()
changed_reward = replace(baseline_reward, robot_contact=12.0)

baseline_config = ApproachConfig(easier_reset_fraction=0.25, reward_weights=baseline_reward)
changed_config = replace(baseline_config, reward_weights=changed_reward)
conditions = {"baseline": baseline_config, "strong_contact": changed_config}
""",
    ),
    (
        "markdown",
        r"""まず同じ初期状態・同じ行動で報酬を確認します。
この短い1ステップで機体同士が接触しなければ、変更した項は0で、報酬も同じです。
「係数を変更しただけで常に報酬が変わる」わけではありません。
`time` の係数を変えると、停止指令のこのステップでも違いを確認できます。
""",
    ),
    (
        "code",
        r"""for name, config in conditions.items():
    env = gym.make("HexapodApproach-v0", config=replace(config, layout="front"))
    env.reset(seed=80000)
    _, reward, _, _, info = env.step(np.zeros((2, 3), dtype=np.float32))
    print(name, "報酬:", reward, "内訳:", info["reward_terms"])
    env.close()
""",
    ),
    (
        "markdown",
        r"""## 7. 同じ学習条件で、2種類の報酬を学習する
両条件とも回り込みactorは同じseedでランダム初期化します。
歩行モデル・押すモデル・物理条件・学習量・PPOの設定・カリキュラムの規則を共通にします。
近い配置 → 後方 → 側方 → 前方と進み、検証成功率75%以上・各段階50ロールアウト以上で難度が上がります。
`ApproachCurriculum` はSB3のcallbackで、50ロールアウトごとに学習済み方策を別の検証seedで評価します。
検証はそのロールアウトのPPO更新前に行います。難度変更は次のエピソードのresetから適用します。
このcallbackは環境の初期配置だけを変更し、学習の更新処理は `model.learn()` が担当します。
報酬によって進む段階が変わる可能性があるため、到達段階もログで確認してください。

初期値の **256チームステップ／条件は動作確認だけ**です。運搬成功を期待する学習量ではありません。
一巡できたら、新しい `EXPERIMENT_NAME` と `TRAINING_STEPS = 409600` を設定して本学習します。
1チームステップは「2台のいる世界を1回進める」ことです。並列数×horizonが1ロールアウトの収集量です。
`TRAINING_STEPS` は `NUM_ENVS * HORIZON` の倍数で指定します。SB3はロールアウト単位で収集するため、
端数を指定すると実際の収集量は切り上がります。
`NUM_ENVS` は独立した世界の数で、ロボット台数ではありません。
学習中は描画せず、CPUの並列シミュレーションを使います。

結果を混ぜないため、新規実験名で実行します。保存補助はソース・資産・実行環境の記録だけを行い、
学習や評価は下のセルから直接呼び出します。
""",
    ),
    (
        "code",
        r"""from hexapod_transport_rl.experiments import create_experiment

EXPERIMENT_NAME = "reward_trial_01"
TRAINING_SEED = 20261008
NUM_ENVS = 2
HORIZON = 64
TRAINING_STEPS = 256

RUN_DIR = create_experiment(PROJECT_DIR, EXPERIMENT_NAME)
PUSHER = RUN_DIR / "source_snapshot/checkpoints/pusher.pt"
print("1条件あたりの収集量:", TRAINING_STEPS, "チームステップ")
""",
    ),
    (
        "markdown",
        r"""### 参考：以前に学習した運搬モデルの動きを見る
自分の比較学習の前に、Tを回り込んで押す目標の挙動を確認します。
これは以前の報酬学習モデルの再生で、今回の学習の結果ではありません。
この動画や行動を教師として学習に使いません。
""",
    ),
    (
        "code",
        r"""from hexapod_transport_rl import evaluate_transport
import imageio.v3 as iio

REFERENCE = PROJECT_DIR / "checkpoints/transport.pt"
evaluate_transport(
    checkpoint=REFERENCE, output=RUN_DIR / "reference.json",
    episodes=1, seed=70000, workers=1, layout="front", episode_seconds=100.0,
    video_dir=RUN_DIR / "reference_video", video_width=640, video_height=480,
    video_fps=5, fast_video=True,
)
media.show_video(iio.imread(next((RUN_DIR / "reference_video").glob("*.mp4"))), fps=5)
""",
    ),
    (
        "code",
        r"""from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from hexapod_transport_rl import SharedTeamPolicy, ApproachCurriculum, bind_transport

models = {}
for name, config in conditions.items():
    output = RUN_DIR / name
    envs = make_vec_env(
        "hexapod_transport_rl:HexapodApproach-v0",
        n_envs=NUM_ENVS, seed=TRAINING_SEED,
        env_kwargs={"config": replace(config, layout="near"), "flatten": True},
        vec_env_cls=SubprocVecEnv, vec_env_kwargs={"start_method": "spawn"},
    )
    model = PPO(
        SharedTeamPolicy, envs, n_steps=HORIZON, batch_size=64,
        n_epochs=4, ent_coef=0.005, policy_kwargs={"log_std_init": -1.0},
        seed=TRAINING_SEED, device="cpu", verbose=1,
    )
    model.set_logger(configure(str(output), ["stdout", "csv"]))
    curriculum = ApproachCurriculum(config=config, output=output)
    checkpoints = CheckpointCallback(
        save_freq=25 * NUM_ENVS * HORIZON, save_path=str(output / "checkpoints"),
        name_prefix="navigation",
    )
    model.learn(total_timesteps=TRAINING_STEPS, callback=[curriculum, checkpoints])
    model.save(output / "navigator")
    models[name] = bind_transport(
        output / "navigator.zip", PUSHER, output / "transport.json",
        config=config, provenance=envs.get_attr("provenance")[0],
    )
    envs.close()
""",
    ),
    (
        "markdown",
        r"""### 保存した方策を読み込み、観測から行動を決める
`PPO.load()` で自分のモデルを読み込み、`model.predict(observation)` で行動を決めます。
このセルでは行動を手で指定しません。`deterministic=True` は評価用に平均の行動を使います。
SB3へ渡す `flatten=True` の回り込み環境は、観測が `(20,)`、行動が `(6,)` です。
行動を `(2,3)` に並べると、各機に送る前後・左右・旋回の正規化値を確認できます。
""",
    ),
    (
        "code",
        r"""model = PPO.load(RUN_DIR / "baseline/navigator.zip", device="cpu")
env = gym.make("HexapodApproach-v0", config=replace(baseline_config, layout="front"), flatten=True)
observation, info = env.reset(seed=83000)
action, _ = model.predict(observation, deterministic=True)
print("方策が決めた2台分の行動:", action.reshape(2, 3))
observation, reward, terminated, truncated, info = env.step(action)
print("報酬:", reward, "内訳:", info["reward_terms"])
env.close()
""",
    ),
    (
        "markdown",
        r"""SB3の `progress.csv` に収集量・学習速度・損失と、callbackが記録したカリキュラムが残ります。
以下で実測速度と本学習の概算を確認します。速度には検証時間も含まれますが、CPUの性能や学習が進んだ後の配置で変わるため概算です。
異なる報酬設計の総報酬の大小は、性能の良し悪しとして比較しません。
""",
    ),
    (
        "code",
        r"""import pandas as pd

training_logs = {}
for name in conditions:
    log = pd.read_csv(RUN_DIR / name / "progress.csv")
    training_logs[name] = log
    display(log.tail(3))
    speed = log["time/fps"].iloc[-1]
    print(name, "学習速度:", speed, "チームステップ/秒")
    print("409600ステップの概算:", round(409600 / speed / 3600, 2), "時間/条件")
""",
    ),
    (
        "markdown",
        r"""## 8. 学習と別の初期配置で定量評価する
同じ学習量の**最後のモデル**を比較します。テスト結果を見てモデルを選び直しません。
両条件に同じテストseedを使い、前方・側方から始めます。
回り込みが成功すると、共通の押す方策へ切り替えて運搬を続けます。

初期値は2試行・10秒で、評価APIの動作確認です。本評価では
`EVALUATION_EPISODES = 50`、`EPISODE_SECONDS = 100.0` に変更します。
50試行の結果にも初期配置によるばらつきがあります。
学習seedも少なくとも3種類で実験を繰り返し、条件ごとの平均・ばらつきを報告しましょう。

報酬とは独立した、成功率・接触・転倒・最終位置誤差を比較します。
所要時間は成功例のみで集計し、成功数も併記します。
""",
    ),
    (
        "code",
        r"""from hexapod_transport_rl import evaluate_transport

EVALUATION_EPISODES = 2
EPISODE_SECONDS = 10.0
TEST_SEEDS = {"front": 80000, "side": 81000}

reports = {}
for name, checkpoint in models.items():
    for layout, seed in TEST_SEEDS.items():
        reports[name, layout] = evaluate_transport(
            checkpoint=checkpoint,
            output=RUN_DIR / "evaluation" / name / f"{layout}.json",
            episodes=EVALUATION_EPISODES,
            seed=seed,
            workers=NUM_ENVS,
            layout=layout,
            episode_seconds=EPISODE_SECONDS,
        )
""",
    ),
    (
        "code",
        r"""rows = []
for (name, layout), report in reports.items():
    successes = [episode["elapsed_seconds"] for episode in report["episodes"] if episode["success"]]
    rows.append({
        "condition": name, "layout": layout,
        "success_rate": report["success_rate"],
        "successes": report["successes"], "episodes": len(report["episodes"]),
        "handovers": report["handovers"],
        "robot_contact_episodes": report["robot_contact_episodes"],
        "navigation_robot_contact_episodes": report["navigation_robot_contact_episodes"],
        "body_contact_episodes": report["body_contact_episodes"],
        "falls": report["falls"],
        "final_distance_m": report["mean_final_distance_m"],
        "final_yaw_error_rad": np.mean([episode["yaw_error"] for episode in report["episodes"]]),
        "successful_time_s": np.mean(successes) if successes else np.nan,
    })
comparison = pd.DataFrame(rows)
comparison.to_csv(RUN_DIR / "comparison.csv", index=False)
display(comparison)
""",
    ),
    (
        "markdown",
        r"""## 9. 図と動画で違いを説明する
まず成功率の図を見て、続いて同じ初期配置の動画を比較します。
成功率だけでなく、回り込む経路・機体同士の接触・停止・脚と胴体の接触を観察してください。
接触したエピソードの数だけでは接触時間や強さは分からないため、必要なら評価JSONの接触力積も調べます。
""",
    ),
    (
        "code",
        r"""import matplotlib.pyplot as plt

success_table = comparison.pivot(index="layout", columns="condition", values="success_rate")
ax = success_table.plot.bar(ylim=(0, 1), ylabel="Transport success rate", rot=0)
ax.figure.tight_layout()
ax.figure.savefig(RUN_DIR / "success_rate.png", dpi=160)
plt.show()
""",
    ),
    (
        "code",
        r"""import imageio.v3 as iio

videos = []
for name, checkpoint in models.items():
    video_dir = RUN_DIR / "videos" / name
    evaluate_transport(
        checkpoint=checkpoint,
        output=RUN_DIR / "video_evaluation" / f"{name}.json",
        episodes=1,
        seed=82000,
        workers=1,
        layout="front",
        episode_seconds=EPISODE_SECONDS,
        video_dir=video_dir,
        video_width=640,
        video_height=480,
        video_fps=5,
        fast_video=True,
    )
    videos.append(iio.imread(next(video_dir.glob("*.mp4"))))
media.show_videos(videos, fps=5, titles=list(models))
""",
    ),
    (
        "markdown",
        r"""## 10. 結果を保存し、研究として考察する
保存するものは実験条件・使用したコードとモデル・学習ログ・評価JSON・比較CSV・図・動画です。
次のセルでZIPを作ります。Colabではファイル一覧からダウンロードしてください。
ランタイムを終了すると未保存の結果は失われます。本学習では途中checkpointもDrive等へ保存してください。

レポートには、次の内容を書きましょう。

1. 変更した報酬と、変更前に立てた仮説。
2. 共通にした条件、学習量、学習seed、評価seed、到達カリキュラム。
3. 成功率・接触・転倒・位置誤差・成功時の時間と、動画で分かった違い。
4. 仮説が支持されたか。失敗した場合、報酬設計・学習不足・初期配置のどれが原因か。
5. 次に1つだけ変更して検証する条件。

**拡張課題：押す段階の報酬を調べる場合**は、`PushRewardWeights` を `PushConfig(reward_weights=...)`
へ渡せます。押す方策を新しく学習する場合も、`gym.make("HexapodPush-v0", config=..., flatten=True)` と
`PPO(SharedTeamPolicy, env, ...)` を使えます。旧MAPPOの `.pt` はSB3の初期重みとして直接読み込めません。
比較する両条件とも同じ乱数seedから開始し、同じ学習量で比較します。
回り込みと押す報酬を同時に変えると原因を切り分けにくいため、最初は片方ずつ調べてください。
APIの詳細・再開方法は [教材ガイド](https://github.com/hashimoto-robotics-lab/hexapod-transport-rl/blob/main/docs/colab-guide.md) にあります。
""",
    ),
    (
        "code",
        r"""from hexapod_transport_rl.experiments import archive_results

archive = archive_results(RUN_DIR)
print("保存先:", archive)
""",
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
