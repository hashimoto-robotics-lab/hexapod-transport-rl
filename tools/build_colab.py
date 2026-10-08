"""Build the student reward-adaptation notebook from readable cell definitions."""

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
研究の問い：「速さと胴体接触へのペナルティを組み合わせると、協調運搬はどう変わるか？」
同じ未使用の配置で比較し、報酬の変更とルールベースとの挙動の違いを調べます。

## 学習済みロボットへコマンドを送るところから始める
固定した歩行モデルは、前後・左右・旋回の指令を18関節の目標角度へ変換します。
上位の運搬方策も、教師デモを使わずMAPPOで学習したものを配布します。
この教材では**共通のRL学習済みモデルから報酬を変えて追加学習**し、比較にかかる時間を減らします。
研究対象は「報酬変更後の適応」です。最初からの学習速度・到達性能を示す実験とは区別します。
ランダムな重みからの再現は別の [新規学習ノートブック](https://github.com/hashimoto-robotics-lab/hexapod-transport-rl/blob/main/notebooks/hexapod_transport_rl_from_scratch.ipynb) にあります。

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
ローカルGPUでランダムな重みから報酬だけで学んだ、共通モデルの動作です。
このモデルの重みを各条件の初期値へ使います。動画・行動を教師には使いません。
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
        r"""## 5. 報酬の項を組み合わせる
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

まず時間コスト `time` を0.02から0.5、胴体接触コスト `body_contact` を6から60へ変える2×2比較です。
標準・時間のみ・接触のみ・両方の4条件で、各項の効果と組み合わせの効果を調べます。
その他の報酬と成功条件は揃えます。報酬を強くしても改善するとは限りません。
実験前に「時間は短くなるか」「接触は減るか」「成功率を落とさないか」の仮説を記録してください。
次はこのセルの係数・組み合わせだけを変えて、別の実験名で繰り返します。
""",
    ),
    (
        "code",
        r"""from dataclasses import replace
from hexapod_transport_rl import PoseRewardWeights, load_mappo

COMMON_CHECKPOINT = PROJECT_DIR / "checkpoints/goal_side_pose.pt"
_, _, common = load_mappo(COMMON_CHECKPOINT)
baseline_reward = PoseRewardWeights()
baseline_config = replace(
    PosePushConfig.from_checkpoint(common["pose_config"]), reward_weights=baseline_reward,
)
conditions = {
    "baseline": baseline_config,
    "time_cost": replace(baseline_config, reward_weights=replace(baseline_reward, time=0.5)),
    "body_cost": replace(baseline_config, reward_weights=replace(baseline_reward, body_contact=60.0)),
    "time_body_cost": replace(baseline_config, reward_weights=replace(baseline_reward, time=0.5, body_contact=60.0)),
}
for name, config in conditions.items():
    print(name, config.reward_weights)
""",
    ),
    (
        "markdown",
        r"""## 6. 短時間の比較実験を揃える
4条件は毎回**同じ共通モデル**から始め、同じseed・歩行モデル・物理・追加学習量・MAPPO設定を使います。
前の条件で学んだ重みを次へ渡しません。actor・critic・価値正規化の統計を読み、新しいAdamで更新します。
教師デモ、ルールの行動、模倣損失は使いません。追加学習中もMAPPOが指令を選びます。
すべての条件を最初から最終課題 **T→ゴール→ロボット・40秒・8 cm・5度・低速1秒** で学びます。
難度の途中変更・段階検証は行わず、収集と更新に時間を使います。

GPUでは1024世界・horizon 64・262,144チームステップ／条件です。
1回に65,536ステップ、各世界12.8秒分を収集し、4回更新します。`NUM_ENVS` はロボット台数ではありません。
CPUの設定は接続確認用です。本実験はGPUで行ってください。
学習中は描画しません。物理・歩行モデル・経験収集・MAPPOをGPU内で計算します。
GPUカーネルの最初のコンパイル、評価、動画、インストールの時間は学習時間と分けて記録します。
Colab/T4の所要時間は実際のログで確認します。ローカルGPUの実測をT4の時間とは扱いません。

標準では1つの追加学習seedです。卒論では `TRAINING_SEED` を3種類以上に変え、実験名も変えて繰り返します。
共通モデルの事前学習は1 seedなので、この繰り返しは**同じ初期モデルからの適応のばらつき**です。
最初からの学習を研究する場合は、別ノートブックで初期モデルから独立に3回以上学びます。
まず2条件で試す場合は `conditions` の辞書を標準と時間コストだけにします。
""",
    ),
    (
        "code",
        r"""import torch
from hexapod_transport_rl.experiments import create_experiment

EXPERIMENT_NAME = "reward_adaptation_01"
TRAINING_SEED = 20261020
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_ENVS = 1024 if DEVICE.type == "cuda" else 2
HORIZON = 64 if DEVICE.type == "cuda" else 128
TRAINING_STEPS = 262144 if DEVICE.type == "cuda" else 1024
RUN_DIR = create_experiment(PROJECT_DIR, EXPERIMENT_NAME)
print("追加学習:", TRAINING_STEPS, "チームステップ/条件", "学習先:", DEVICE)
""",
    ),
    (
        "markdown",
        r"""## 7. TorchRLで報酬変更後の方策を学ぶ
`TorchRLTransportEnv` はGymと同じ課題をTorchRLへ接続します。CUDAがあればWarp、なければCPUを使います。
共有actorは各機の観測から指令を決め、中央criticは全機の観測からチームの価値を推定します。
`load_mappo()` で共通モデルを読み、`make_mappo_loss()` でTorchRLの `MAPPOLoss` とGAEを設定します。
価値正規化の統計も読みます。Adamは各条件で新しく作ります。

ループは **収集 → GAE → ミニバッチ更新 → 記録** の順です。
`Collector` が経験を集め、`ReplayBuffer` が今回の経験を混ぜてミニバッチにします。毎回bufferを空にします。
観測は `(世界, 時間, ロボット, 観測成分)` です。ロボット軸を保ったままMAPPOで更新します。
時間切れでは最後の観測の価値を使い、成功・転倒では使いません。終了した世界だけresetします。

各軸は停止を含む `[-1, -0.5, -0.25, 0, 0.25, 0.5, 1]` の候補から選びます。
学習中はactorの確率で探索し、評価時は確率最大の指令を使います。回り込みのルールは含みません。
`PoseCurriculum` は最終課題を固定し、記録に使います。PPO更新や指令の生成は行いません。
`update_policy_weights_()` で更新した重みをCollectorへ戻します。
物理・歩行・観測・報酬・方策・更新はGPU内で実行します。接触容量と非有限値は毎回確認します。
保存する `.pt` は重み・Adam・価値統計・設定・乱数状態を含み、CPUでも読み込めます。
""",
    ),
    (
        "code",
        r"""import torch
from torchrl.collectors import Collector
from torchrl.data import ReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
from hexapod_transport_rl import (
    TorchRLTransportEnv, MAPPOSettings, load_mappo, make_mappo_loss,
    POSE_STAGES, PoseCurriculum, save_mappo,
)

settings = MAPPOSettings(
    learning_rate=3e-4, minibatch_size=1024 if DEVICE.type == "cuda" else 128, value_normalization=True,
    entropy_coeff=0.005, gamma=0.995, gae_lambda=0.99,
)
print("学習先:", DEVICE)
FRAMES_PER_BATCH = NUM_ENVS * HORIZON
models = {}
for name, config in conditions.items():
    torch.manual_seed(TRAINING_SEED)
    output = RUN_DIR / name
    envs = TorchRLTransportEnv(config=config, num_envs=NUM_ENVS, backend="auto")
    print(name, "物理計算:", envs.backend, envs.device)
    envs.set_seed(TRAINING_SEED)
    actor, critic, common = load_mappo(COMMON_CHECKPOINT, device=DEVICE)
    torch.manual_seed(TRAINING_SEED)
    loss = make_mappo_loss(
        actor, critic, settings, value_normalizer_state=common["value_normalizer"],
    )
    optimizer = torch.optim.Adam(loss.parameters(), lr=settings.learning_rate)
    curriculum = PoseCurriculum(
        config=config, env=envs, actor=actor, output=output,
        seed=TRAINING_SEED, settings=settings, horizon=HORIZON,
        stages=(POSE_STAGES[-1],), validate_every=None, initial_checkpoint=COMMON_CHECKPOINT,
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
        r"""### 実際の学習時間を確認する
`progress.csv` に収集量・学習速度・損失を記録します。損失は最後のミニバッチの値です。
時間は環境の作成・コンパイル後、Collectorの準備から収集・更新・物理確認・記録までです。
共通モデルの事前学習時間は追加学習時間に含めず、卒論では別に報告します。
総報酬は、報酬式が違う条件間の性能比較には使いません。
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
        r"""## 8. 共通モデルと追加学習後を同じ配置で評価する
同じ学習量の**最後のモデル**を評価します。初期の共通モデルも同じ配置で評価し、変化を比べます。
標準は12配置です。これは動作を早く確認するための評価で、卒論では50配置以上・3追加学習seed以上へ増やします。
モデル選択・係数調整には `91000` からの開発seedを使い、最終テストのseedは先に決めて残してください。
設定を凍結してから最終テストを行い、その結果でモデルを選び直しません。
下では開発用seedを使います。成功条件・制限時間・初期配置は共通で、難度を緩めません。

学習方策は `backend="warp"` で全配置を並列評価します。初期状態はCPU MuJoCoと同じseedで作ります。
GPUではfloat32を使うためCPUと軌跡が異なることがあります。卒論の最終評価は `backend="cpu"` でも確認してください。
ルールベースの `forward`・`feedback` はCPUで評価します。ルールは評価専用で、教師には使いません。
成功率・位置・角度・Tの端の誤差・接触・転倒に加え、**位置誤差の時間積分**を比較します。
例えば0.4 mのズレが10秒続くと約4 m·sです。早く目標へ近づくほど小さくなります。
失敗を含む所要時間では、失敗を40秒として集計します。成功例だけの平均時間と区別します。
""",
    ),
    (
        "code",
        r"""from hexapod_transport_rl import evaluate_pose

EVALUATION_EPISODES = 12
TEST_SEED = 91000  # 開発用。卒論の最終テストには別の未使用seedを事前に決める。
EVALUATION_BACKEND = "warp" if DEVICE.type == "cuda" else "cpu"
reports = {}
for name, checkpoint in {"common_model": COMMON_CHECKPOINT, **models}.items():
    reports[name] = evaluate_pose(
        checkpoint, config=baseline_config, episodes=EVALUATION_EPISODES,
        seed=TEST_SEED, backend=EVALUATION_BACKEND,
        output=RUN_DIR / "evaluation" / f"{name}.json",
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
        "time_including_failures_s": report["mean_time_to_success_or_limit_s"],
        "position_error_integral_m_s": report["mean_position_error_integral_m_s"],
        "yaw_error_integral_rad_s": report["mean_yaw_error_integral_rad_s"],
        "evaluation_seconds": report["evaluation_seconds"],
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
標準では共通モデルと両方のコストを変えたモデルを同じ配置で再生します。
左右の押し方・回転・減速・停止を見ます。他の条件は `video_models` の辞書へ追加できます。失敗も数値に含めます。
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
video_models = {"common_model": COMMON_CHECKPOINT, "time_body_cost": models["time_body_cost"]}
for name, checkpoint in video_models.items():
    video_dir = RUN_DIR / "videos" / name
    evaluate_pose(
        checkpoint, config=baseline_config, episodes=1, seed=86000, workers=1,
        output=RUN_DIR / "video_evaluation" / f"{name}.json", video_dir=video_dir,
    )
    videos.append(iio.imread(next(video_dir.glob("*.mp4"))))
    video_names.append(name)
media.show_videos(videos, fps=25, titles=video_names, columns=2)
""",
    ),
    (
        "markdown",
        r"""## 10. 卒論の実験へ広げる
コード・共通モデルのSHA256・報酬・seed・追加学習量・GPU・時間・評価JSON・CSV・図・動画を保存します。
最後のセルでZIPを作り、Colabのファイル一覧から保存してください。結果はランタイム削除で消えます。

1. 研究の問いと仮説を決める。まず2項の2×2比較で単独効果と組み合わせの効果を調べる。
2. 開発配置で学習量を決める。262,144→524,288→1,048,576など、同じ予算で全条件を比較する。
3. 設定を固定し、3種類以上の追加学習seed、50以上の未使用配置で平均・ばらつきを報告する。
4. 成功率・位置誤差の時間積分・失敗込み時間・接触・転倒を数値と動画で説明する。

短い追加学習では共通モデルの習慣が残ります。「報酬の効果がない」とは直ちに結論しません。
成功率が高い初期モデルでは精度の差が小さくなるため、速さ・接触・途中のズレも見ます。
この実験で言えるのは、**この共通モデル・2台・この物理条件での報酬変更後の適応**です。
最初からの学習、別質量・摩擦、4台への一般化は別の実験で検証します。2台のモデルを4台へそのまま使いません。

[教材ガイド](https://github.com/hashimoto-robotics-lab/hexapod-transport-rl/blob/main/docs/colab-guide.md)に保存・新規学習の手順があります。
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
