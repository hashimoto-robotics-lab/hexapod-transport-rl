"""Build the research notebook from its readable cell definitions."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CELLS = [
    (
        "markdown",
        r"""# 六足ロボットの協調運搬：回り込み・回転・停止と報酬設計

## 研究のモチベーション
複数のロボットで物資を運ぶには、押す場所と速度を協調させる必要があります。
この研究では、**六足ロボットがTの背後へ回り込み、脚・足で押して目標の位置と向きへ静止させる動作**を学習します。
まず2台で、APIを理解し、報酬を変えると挙動がどう変わるかを評価します。

Tの横棒と縦棒はどちらも1.3 m、太さは0.20 mです。
最終課題の初期配置は **T → ゴール → ロボット**。ゴール側から約1.85 mの位置でTを向きます。
制限時間40秒、目標は **位置8 cm・角度5度以内で低速状態を1秒維持** です。
研究の問い：「角度のズレが残る時間のペナルティを強くすると、回り込み・運搬の精度は改善するか？」
同じ未使用の配置で比較し、報酬の変更とルールベースとの挙動の違いを調べます。

## 学習済みロボットへコマンドを送るところから始める
固定した歩行モデルは、前後・左右・旋回の指令を18関節の目標角度へ変換します。
このモデルを使って、上位の運搬方策をランダムな重みから新しく学びます。旧運搬モデルや教師デモは使いません。

```text
T・目標・相手の状態 → MAPPO → 速度指令 → 固定歩行モデル → 脚・床・Tの接触
          ↑                                                    ↓
          └──────────────── 次の状態とチーム報酬 ───────────────┘
```

TorchRLのMAPPOを使い、各機のactorの重みを共有し、criticは全機の観測を見ます。
経験収集は `Collector`、更新は `MAPPOLoss`。収集・更新は普通のPythonセルから実行します。
APIは2〜4台を扱えます。4台の運搬性能は別実験で確認します。
準備 → 歩行API → 運搬API → 報酬 → 比較学習 → 評価・動画、の順に進めてください。
""",
    ),
    (
        "markdown",
        r"""## 1. 教材と実行環境を準備する
公開GitHubから取得します。認証は不要です。次の2セルはそのまま実行します。
Colabの「ランタイム → ランタイムのタイプを変更」で **T4 GPU** を選びます。
GPUが使えると、MuJoCo Warpによる物理計算・歩行モデル・MAPPOをGPUで実行します。
GPUが使えない場合は従来のCPU環境で実行します。最初はGPU用カーネルのコンパイルに時間がかかります。
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
歩行・運搬環境を、同じGymnasiumのAPIで操作します。
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
        r"""### 歩行指令を協調させた運搬の例
左は目標との角度差約17度、右は約28度から、学習したMAPPOが回り込んで押す例です。
ローカルGPUでランダムな重みから学習した参考動画で、教師データや学生の初期重みには使いません。
未使用50配置の評価では44件成功しました。これは1つの学習seedの結果で、4台の性能は未検証です。
再生は25 fps・実時間です。左の最終誤差は位置4.4 cm・角度1.0度、右は0.8 cm・4.9度です。
""",
    ),
    (
        "code",
        r"""media.show_videos([
    media.read_video(PROJECT_DIR / "docs/figures/goal_side_learned_25fps.mp4"),
    media.read_video(PROJECT_DIR / "docs/figures/goal_side_turning_25fps.mp4"),
], fps=25, columns=2)
""",
    ),
    (
        "markdown",
        r"""## 4. 運搬環境のAPIを理解する
`HexapodPosePush-v0` は、ゴール側から開始する2台が回り込んでTを押し、目標の位置・向きに静止させる環境です。
`reset()`・`step()`・`render()` は歩行環境と共通です。1回の `step()` で0.2秒進みます。

| 項目 | 2台の場合 |
|---|---|
| 観測 | `(2, 22)`：自分の状態・前回指令・目標・Tの速度・相手の相対状態 |
| 行動 | `(2, 3)`：T基準のX移動・Y移動・旋回、各成分−1〜1 |
| 報酬 | チーム全体で1つ |
| 成功 | 位置8 cm・角度5度以内、低速状態を1秒維持 |

観測の位置・向きはT基準です。左右の役割を同じactorで扱うため、Yが負の側では、Y・旋回の符号を反転します。
行動もこの反転座標で渡します。標準の倍率は `[0.2, 0.2, 0.6]`（m/s・m/s・rad/s）です。
環境がT基準の移動を機体基準へ回転し、歩行モデルの速度範囲に収めます。この変換は行動を作る制御器ではありません。
歩行APIは機体基準、運搬APIはT基準です。観測と行動の座標を揃え、座標変換を学習する負担を減らします。
観測の詳細は [APIガイド](https://github.com/hashimoto-robotics-lab/hexapod-transport-rl/blob/main/docs/api.md) にあります。

`terminated` は成功・転倒等、`truncated` は時間切れです。どちらかがTrueなら次は `reset()` します。
`info["reward_terms"]` は実際に返した報酬の内訳、`distance`・`yaw_error` は位置・角度の誤差です。
`footprint_error_m` はTの3つの端と目標の対応する端の誤差で、見た目のズレを評価します。
`PosePushConfig(num_robots=2)` が標準です。`robot_start="goal_side"` はT→ゴール→ロボット、
`start_clearance=0.55` はTの縦棒先端よりさらに離す距離の基準です。
下のセルではゼロ指令を2秒間送り、APIを確認します。新しい学習は行いません。
""",
    ),
    (
        "code",
        r"""from hexapod_transport_rl import PosePushConfig

config = PosePushConfig(episode_seconds=2.0)
env = gym.make("HexapodPosePush-v0", config=config, render_mode="rgb_array_list")
observation, info = env.reset(seed=84000)
print("観測:", env.observation_space, "行動:", env.action_space)
frames = env.render()
terminated = truncated = False
while not (terminated or truncated):
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    observation, reward, terminated, truncated, info = env.step(action)
    frames.extend(env.render())
    print("報酬:", reward, "内訳:", info["reward_terms"])
print("位置誤差 [m]:", info["distance"], "角度誤差 [rad]:", info["yaw_error"])
env.close()
media.show_video(frames, fps=25)
""",
    ),
    (
        "markdown",
        r"""## 5. 報酬を理解し、1つだけ変更する
Tの位置・向きと、ロボットが押す位置に近づく程度を評価します。
距離は0.5 m、角度は0.5 radで割って尺度を揃えます。

| 項 | 意味 |
|---|---|
| `position` | Tと目標の位置誤差から作る補助報酬 |
| `position_error` | 位置誤差が残る時間へのペナルティ |
| `orientation` | Tと目標の角度誤差から作る補助報酬 |
| `orientation_error` | 角度のズレが残る時間へのペナルティ |
| `approach` | Tの外側を通って押す位置へ近づく補助報酬 |
| `approach_error` | 押す位置から離れたままでいる時間へのペナルティ |
| `push_heading` | Tの背後で、押す向きへ向いていないことへのペナルティ |
| `settling` | 目標付近でTが動き続けることへのペナルティ |
| `command_change` | 指令の急な変化へのペナルティ |
| `time`・`robot_contact`・`body_contact` | 時間・機体同士の接触・胴体とTの接触へのペナルティ |
| `terminal`（係数 `success`・`failure`） | 成功時・転倒等で終了したときの報酬 |

`position`・`orientation`・`approach` には `gamma × Phi(次の状態) − Phi(今の状態)` を使います。
例えば位置の `Phi` は「−位置誤差／0.5」です。進捗のないステップでも割引による項が残るため、
その瞬間の報酬が正でも、運搬が上手いとは限りません。**性能は別の評価で比較**します。
成功・失敗の終端では `Phi=0` とし、時間切れでは最終観測の値を保ってGAEへ渡します。
割引率は報酬とMAPPOで同じ0.995に揃えます。行動の教師値やルールの経路を与えるものではありません。

直線距離だけを評価すると、Tを通り抜ける方向へ近づこうとして停滞します。
標準ではTの外側を通る距離を使い、脚の広がりを考慮して前方・側方に0.4 mの余裕を設けます。
余裕領域へ入り込むほどコストを増やし、Tを通り抜ける近道を評価しません。後方は脚で押せる距離を保ちます。この距離計算は報酬用で、経路や速度指令をactorへ渡しません。
`push_heading` はTの背後で強く働き、前方をTの縦棒の方向へ向ける姿勢を評価します。
式を研究する場合は `pose_rewards.py` の `pose_reward_terms()` を編集します。

以下では**残る角度のズレの係数 `orientation_error` だけを0から4へ増やす**比較を行います。
角度の改善量の報酬と成功条件は両条件に残し、「残るズレを強く評価すると、より正確に静止できるか」を調べます。強いペナルティが学習を難しくする可能性も検証します。
実験前に、どちらが速く正確に学べるか仮説を記録してください。
""",
    ),
    (
        "code",
        r"""from dataclasses import replace
from hexapod_transport_rl import PoseRewardWeights

baseline_reward = PoseRewardWeights()
changed_reward = replace(baseline_reward, orientation_error=4.0)

baseline_config = PosePushConfig(num_robots=2, reward_weights=baseline_reward, shaping_discount=0.995)
changed_config = replace(baseline_config, reward_weights=changed_reward)
conditions = {"baseline": baseline_config, "more_orientation_cost": changed_config}
for name, config in conditions.items():
    print(name, config.reward_weights)
""",
    ),
    (
        "markdown",
        r"""## 6. 学習と比較実験の条件を揃える
両条件で同じ初期重み・seed・歩行モデル・物理設定・学習量・MAPPO設定を使います。
最終課題は **T→ゴール→ロボット**、横棒・縦棒は同じ長さ、制限時間40秒です。
いきなり難しい配置から始める代わりに、開始位置を45度ずつTの周りで移します。
**各段階ともMAPPOが行動を決めます。** 教師デモ・旧運搬モデル・Tへの外力は使いません。

| 段階 | 開始位置 | 目標距離・角度差 | 成功精度（位置・角度・維持時間） |
|---|---|---|---|
| approach_0 | Tの背後・近く | 0.3 m・±5〜15度 | 8 cm・8度・0.6秒 |
| approach_45 / 90 / 135 | 側方から徐々にゴール側へ | 0.3 m・±5〜15度 | 8 cm・8度・0.6秒 |
| approach_180 | T→ゴール→ロボット | 0.3 m・±5〜15度 | 8 cm・8度・0.6秒 |
| transport | T→ゴール→ロボット | 0.4 m・±5〜30度 | 8 cm・5度・0.6秒 |
| settle | T→ゴール→ロボット | 0.4 m・±5〜30度 | 8 cm・5度・1秒 |

![開始位置の段階](https://raw.githubusercontent.com/hashimoto-robotics-lab/hexapod-transport-rl/main/docs/figures/goal_side_curriculum.png)

別seedの6配置を8ロールアウトごとに検証します。3/6成功・現在の段階で8ロールアウト以上で進みます。
変更は次のresetから適用し、検証経験はPPOへ渡しません。最終評価は常に最終課題の配置と精度です。
両条件で到達段階が違う可能性も、報酬を変えた結果として記録します。

GPUでは256世界・horizon 64・3,145,728チームステップ／条件を使います。
1回の収集は16,384ステップ、速度指令は5 Hzなので、各世界で12.8秒間を観測してから更新します。
CPUでは2世界・horizon 128・1,024ステップのAPI確認だけを行います。本課題の実学習にはGPUを使ってください。
学習中の描画は省き、物理・固定歩行モデル・観測・報酬・MAPPOをGPUで計算します。
段階の検証も独立したGPU環境で行い、最後に通常のCPU MuJoCoで別途評価します。
T4の所要時間は後のCSVで実測します。ローカルGPUの時間をColabの実測として扱いません。

1チームステップは全機のいる世界を1回進めることです。`NUM_ENVS` はロボット台数ではなく並列世界数です。
`TRAINING_STEPS` は `NUM_ENVS * HORIZON` の倍数で指定します。小さい予算はAPI接続確認用です。
まずbaselineだけを試す場合は `conditions = {"baseline": baseline_config}` とします。
結果を混ぜないよう、新しい実験名で始めます。保存補助はコード・資産・実行環境の記録だけを担当します。
""",
    ),
    (
        "code",
        r"""from hexapod_transport_rl.experiments import create_experiment

EXPERIMENT_NAME = "curriculum_reward_trial_01"
TRAINING_SEED = 20261010
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_ENVS = 256 if DEVICE.type == "cuda" else 2
HORIZON = 64 if DEVICE.type == "cuda" else 128
TRAINING_STEPS = 3145728 if DEVICE.type == "cuda" else 1024

RUN_DIR = create_experiment(PROJECT_DIR, EXPERIMENT_NAME)
print("1条件あたりの収集量:", TRAINING_STEPS, "チームステップ")
""",
    ),
    (
        "markdown",
        r"""## 7. TorchRLで回り込みと運搬を学ぶ
### 経験を収集し、MAPPOで更新する
`TorchRLTransportEnv` は先ほどのGymnasium環境と同じ課題をTorchRLへ接続します。
`backend="auto"` はCUDAがあればWarp、なければCPUを選びます。
機体・T・DCモーター・歩行モデル・報酬式は共通ですが、GPUの物理演算はfloat32で、CPUの軌跡と完全には一致しません。
名前付き配列を `TensorDict` にまとめ、観測は `(並列世界, 時間, ロボット, 観測成分)` となります。
`make_mappo_networks()` は共有actor・中央critic、`make_mappo_loss()` はTorchRLの `MAPPOLoss` とGAEを設定します。
観測と行動のロボット軸を保って更新します。価値正規化でcriticの更新尺度を揃えます。

下のループは **収集 → GAE → ミニバッチ更新 → 記録** の順です。
`ReplayBuffer` は今回の経験を混ぜてミニバッチにするために使い、毎ロールアウト空にします。
時間切れでは最終観測の価値を使い、成功・転倒では使いません。終了した世界だけCollectorがresetします。
各軸の指令候補を `[-1, -0.5, -0.25, 0, 0.25, 0.5, 1]` とし、actorが各候補の確率を学びます。
例えばT基準のX指令は `[-0.20, -0.10, -0.05, 0, 0.05, 0.10, 0.20]` m/sです。
学習中はその確率で選び、評価時は各軸で最も確率が高い指令を選びます。
停止も明示的な選択肢です。反対方向の指令を平均して、小さすぎる指令で止まることを避けます。
確率の偏りはMAPPOとentropy項で学習し、探索ノイズの段階別調整は不要です。
この候補は状態によらない行動空間で、回り込みや押す順番のルールは含みません。
`PoseCurriculum` は検証・難度の変更・ログだけを行い、行動を作ったりPPO更新を代行したりしません。
`DEVICE` は学習先です。actor・critic・GAE・損失計算をGPUへ置きます。
Collectorも `envs.device` で方策を実行・収集し、WarpではGPU内で経験を受け渡します。
更新後は `update_policy_weights_()` で、登録済みactorの最新の重みをCollectorへ戻します。
CPU環境ではCPUで収集してから学習先へ転送します。物理計算の各ステップでCPUとGPUを往復しません。
接触バッファの不足や非有限値はロールアウトごとにライブラリが確認します。
保存する `.pt` には重み・optimizer・価値正規化の統計・報酬・到達段階・乱数状態が入ります。別の押すモデルは必要ありません。
""",
    ),
    (
        "code",
        r"""import torch
from torchrl.collectors import Collector
from torchrl.data import ReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
from hexapod_transport_rl import (
    TorchRLTransportEnv, MAPPOSettings, make_mappo_networks, make_mappo_loss,
    PoseCurriculum, save_mappo,
)

settings = MAPPOSettings(
    learning_rate=3e-4, minibatch_size=512 if DEVICE.type == "cuda" else 128, value_normalization=True,
    entropy_coeff=0.005, gamma=0.995, gae_lambda=0.99,
)
ACTION_GRID = (-1, -0.5, -0.25, 0, 0.25, 0.5, 1)
print("学習先:", DEVICE)
FRAMES_PER_BATCH = NUM_ENVS * HORIZON
models = {}
for name, config in conditions.items():
    torch.manual_seed(TRAINING_SEED)
    output = RUN_DIR / name
    envs = TorchRLTransportEnv(config=config, num_envs=NUM_ENVS, backend="auto")
    print(name, "物理計算:", envs.backend, envs.device)
    envs.set_seed(TRAINING_SEED)
    actor, critic = make_mappo_networks(num_robots=envs.num_robots, obs_dim=envs.obs_dim,
        action_grid=ACTION_GRID,
    )
    actor.to(DEVICE)
    critic.to(DEVICE)
    loss = make_mappo_loss(actor, critic, settings)
    optimizer = torch.optim.Adam(loss.parameters(), lr=settings.learning_rate)
    curriculum = PoseCurriculum(
        config=config, env=envs, actor=actor, output=output,
        seed=TRAINING_SEED, settings=settings, horizon=HORIZON,
    )
    save_mappo(
        output / "initial.pt", actor, critic, optimizer,
        config=config, provenance=envs.provenance,
        training=curriculum.training, curriculum=curriculum.state, loss=loss,
    )
    collector = Collector(
        envs, actor, frames_per_batch=FRAMES_PER_BATCH, total_frames=TRAINING_STEPS,
        env_device=envs.device, policy_device=envs.device, storing_device=envs.device,
        auto_register_policy_transforms=True,
    )
    buffer = ReplayBuffer(
        storage=LazyTensorStorage(FRAMES_PER_BATCH, device=DEVICE), sampler=SamplerWithoutReplacement(),
        batch_size=settings.minibatch_size,
    )
    for batch in collector:
        batch = batch.to(DEVICE)
        loss.value_estimator(batch)
        buffer.empty()
        buffer.extend(batch.reshape(-1))
        for _ in range(settings.epochs):
            for minibatch in buffer:
                metrics = loss(minibatch)
                objective = metrics["loss_objective"] + metrics["loss_critic"] + metrics["loss_entropy"]
                optimizer.zero_grad()
                objective.backward()
                torch.nn.utils.clip_grad_norm_(loss.parameters(), settings.max_grad_norm)
                optimizer.step()
        curriculum.record(batch, metrics)
        collector.update_policy_weights_()
        if curriculum.rollouts % 50 == 0:
            save_mappo(
                output / "checkpoints" / f"step_{curriculum.transitions}.pt", actor, critic, optimizer,
                config=config, provenance=envs.provenance,
                training=curriculum.training, curriculum=curriculum.state, loss=loss,
            )
    models[name] = save_mappo(
        output / "pose.pt", actor, critic, optimizer,
        config=config, provenance=envs.provenance,
        training=curriculum.training, curriculum=curriculum.state, loss=loss,
    )
    curriculum.close()
    collector.shutdown()
""",
    ),
    (
        "markdown",
        r"""### 保存した方策から行動を決める
`load_mappo()` で自分のactorを読み、`policy_action(actor, observation)` で行動を決めます。
評価は探索の乱数を加えず、各軸で確率が最も高い指令を使います。ルールの移動指令を混ぜません。
""",
    ),
    (
        "code",
        r"""from hexapod_transport_rl import load_mappo, policy_action

actor, critic, saved = load_mappo(models["baseline"])
env = gym.make("HexapodPosePush-v0", config=baseline_config)
observation, info = env.reset(seed=86000)
action = policy_action(actor, observation)
print("方策が決めた行動:", action)
observation, reward, terminated, truncated, info = env.step(action)
print("報酬:", reward, "内訳:", info["reward_terms"])
env.close()
""",
    ),
    (
        "markdown",
        r"""### 学習時間と到達段階を確認する
`progress.csv` に収集量・学習速度・到達段階・損失が残ります。損失は最後のミニバッチの値です。
速度には段階移行の検証も含まれます。総報酬は報酬式が違う条件間の性能比較には使いません。
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
    speed = log["team_steps_per_second"].iloc[-1]
    print(name, "実際の学習時間:", round(log["elapsed_seconds"].iloc[-1] / 60, 2), "分/条件")
    print("設定した学習量の概算:", round(TRAINING_STEPS / speed / 60, 2), "分/条件")
""",
    ),
    (
        "markdown",
        r"""## 8. 未使用の初期配置でルールベースと比較する
同じ学習量の**最後のモデル**を評価し、テスト結果を見てモデルを選び直しません。
テストは全条件に共通のseed・最終精度・制限40秒を使います。学習で最終段階へ進めなかったモデルも同じ条件です。

比較対象は、一定速度で前進して位置が近くなったら止まる `forward` と、
位置・角度の誤差から左右の速度を調整する比例制御 `feedback` です。
ルールの状態取得条件は学習方策と揃えます。**ルールは評価専用で、学習の教師には使いません。**
制御器の詳細は `pose_evaluation.py` にあります。ゲインを変更する場合は検証seedで選び、テストseedを残します。

標準は12試行です。卒論では50試行以上、学習seedも少なくとも3種類へ増やし、平均・ばらつきを報告します。
成功率・最終位置・角度・Tの端の誤差・接触・転倒を比較します。時間は成功例のみで集計します。
""",
    ),
    (
        "code",
        r"""from hexapod_transport_rl import evaluate_pose

EVALUATION_EPISODES = 12
TEST_SEED = 88000

reports = {}
for name, checkpoint in models.items():
    reports[name] = evaluate_pose(
        checkpoint, config=baseline_config, episodes=EVALUATION_EPISODES,
        seed=TEST_SEED, workers=2, output=RUN_DIR / "evaluation" / f"{name}.json",
    )
for policy in ("forward", "feedback"):
    reports[policy] = evaluate_pose(
        config=baseline_config, policy=policy, episodes=EVALUATION_EPISODES,
        seed=TEST_SEED, workers=2, output=RUN_DIR / "evaluation" / f"{policy}.json",
    )
""",
    ),
    (
        "code",
        r"""rows = []
for name, report in reports.items():
    successful_times = [episode["elapsed_seconds"] for episode in report["episodes"] if episode["success"]]
    rows.append({
        "condition": name, "success_rate": report["success_rate"],
        "successes": report["successes"], "episodes": len(report["episodes"]),
        "final_distance_m": report["mean_distance_m"],
        "final_yaw_error_degrees": report["mean_yaw_error_degrees"],
        "footprint_error_m": report["mean_footprint_error_m"],
        "body_contact_episodes": report["body_contact_episodes"],
        "robot_contact_episodes": report["robot_contact_episodes"], "falls": report["falls"],
        "successful_time_s": np.mean(successful_times) if successful_times else np.nan,
    })
comparison = pd.DataFrame(rows)
comparison.to_csv(RUN_DIR / "comparison.csv", index=False)
display(comparison)
""",
    ),
    (
        "markdown",
        r"""## 9. 図と動画で違いを説明する
成功率に加え、Tの最終的なズレを比較します。動画は全条件を同じseedから再生します。
左右の押し方・回転・目標付近の減速・停止を観察してください。失敗例も結果に含めます。
""",
    ),
    (
        "code",
        r"""import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
comparison.plot.bar(x="condition", y="success_rate", ax=axes[0], ylim=(0, 1), legend=False, rot=20)
comparison.plot.bar(x="condition", y="footprint_error_m", ax=axes[1], legend=False, rot=20)
axes[0].set_ylabel("Pose success rate")
axes[1].set_ylabel("T endpoint error [m]")
fig.tight_layout()
fig.savefig(RUN_DIR / "pose_comparison.png", dpi=160)
plt.show()
""",
    ),
    (
        "code",
        r"""import imageio.v3 as iio

videos = []
video_names = []
video_models = {"before_training": RUN_DIR / "baseline/initial.pt", **models}
for name, checkpoint in video_models.items():
    video_dir = RUN_DIR / "videos" / name
    evaluate_pose(
        checkpoint, config=baseline_config, episodes=1, seed=86000, workers=1,
        output=RUN_DIR / "video_evaluation" / f"{name}.json", video_dir=video_dir,
    )
    videos.append(iio.imread(next(video_dir.glob("*.mp4"))))
    video_names.append(name)
for policy in ("forward", "feedback"):
    video_dir = RUN_DIR / "videos" / policy
    evaluate_pose(
        config=baseline_config, policy=policy, episodes=1, seed=86000, workers=1,
        output=RUN_DIR / "video_evaluation" / f"{policy}.json", video_dir=video_dir,
    )
    videos.append(iio.imread(next(video_dir.glob("*.mp4"))))
    video_names.append(policy)
media.show_videos(videos, fps=25, titles=video_names, columns=2)
""",
    ),
    (
        "markdown",
        r"""## 10. 結果を保存し、研究として考察する
結果フォルダに、実行時のコード・モデル・設定・学習ログ・評価JSON・CSV・図・動画が残ります。
最後にZIPを作ります。Colabのファイル一覧からダウンロードしてください。
未保存の結果はランタイムの削除で失われます。長い実験では途中checkpointもDrive等へ保存します。

レポートには、変更した報酬と事前の仮説、共通条件、学習量・seed、到達段階を書きます。
成功率だけでなく、最終精度・接触・転倒・成功時の時間と、動画で見えた動きを説明します。
ルールが良い結果になった場合も、学習方策の利点が出なかった条件として報告してください。
次の実験では、報酬・初期角度・質量・摩擦などを1つずつ変更し、未使用条件での性能を調べます。
今回の開始配置で回り込みが学べたかを軌跡と動画で確認してください。
4台での運搬は `baseline_config` の `num_robots=4` で新しく学習し、2台とは別に評価します。

[教材ガイド](https://github.com/hashimoto-robotics-lab/hexapod-transport-rl/blob/main/docs/colab-guide.md)に設定・再開方法があります。
""",
    ),
    (
        "code",
        r"""from hexapod_transport_rl.experiments import archive_results

archive = archive_results(RUN_DIR)
print("保存したZIP:", archive)
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
