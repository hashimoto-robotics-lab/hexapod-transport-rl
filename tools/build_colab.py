"""Build the student notebook using official Push-T task conventions."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CELLS = [
    (
        "markdown",
        r"""# 六足ロボットの協調運搬：Push-Tの設定に沿った強化学習

複数台で物を運ぶには、押す場所と動かすタイミングを協調させる必要があります。
この教材は、まず学習済み六足ロボットへ歩行コマンドを送り、Gym APIを確認し、
次にTorchRLのMAPPOでTの運搬を学び、報酬の組み合わせによる挙動の違いを評価します。
教師デモ・模倣損失は使いません。固定するのは歩行モデルです。

[公式gym-pusht](https://github.com/huggingface/gym-pusht)に合わせ、**世界座標のXY目標位置を行動、目標との面積重なり率を報酬**にします。
重なり率95%超で成功です。以前の8 cm・5度・1秒静止の条件は使いません。
目標は固定位置・固定向きです。観測は公式の状態5成分に、六足の向きと相手の姿勢を追加します。
六足の3D物理、T→ゴール→ロボットの開始配置、等長2棒のT、5 Hzの上位制御は研究用の拡張です。
公式2Dベンチマークと同じ環境ではないため、成功率や時間を直接比較しません。

```text
観測 → MAPPO → 各機のXY目標 → 位置追従 → 固定歩行モデル → 脚・床・Tの接触
  ↑                                                                  ↓
  └──────────────────── 次の観測とチーム報酬 ─────────────────────────┘
```

**旧課題の学習済みモデルは新しい観測・行動へ流用しません。** 新しい運搬actor・criticを同じ乱数seedから作って比較します。
以前の約42秒の追加学習・成功率は旧課題の結果です。新仕様の性能・時間はここで改めて計測します。
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
        r"""## 4. Push-Tの観測とXY目標位置を試す
`reset()`・`step()`・`render()` は歩行環境と同じGym APIです。
行動は `(台数, 2)` の **世界座標のXY目標位置 [m]** です。`[0, 0]` は原点へ移動する指令です。
現在位置を指定するとその場に留まろうとします。歩行の速度指令と区別してください。

位置追従は自分の位置・向き・速度とXY目標だけから歩行指令を作ります。
Tやゴールを見て経路を決めるルールはありません。Tは脚・足の接触で動きます。
上位は0.2秒ごと、歩行は25 Hz、物理・モーターは200 Hzです。

`obs_type="state"` の2台では各機9成分です。

| 番号 | 内容 |
|---|---|
| 0〜1 | 自分の世界XY [m] |
| 2〜3 | Tの世界XY [m] |
| 4 | Tの向き [rad、0〜2π] |
| 5 | 自分の向き [rad、0〜2π] |
| 6〜8 | 相手の世界XY・向き |

先頭5成分は公式の状態版に対応します。4台なら各機15成分です。
`obs_type="keypoints"` はTの8頂点のXYと自分のXYが先頭18成分、向きと相手を含め2台で22成分です。
キーポイントは2本の物理矩形の4頂点ずつです。ゴールは固定なので観測に追加しません。
下では現在位置の指定と、XY目標を少し変えたときの動きを確認します。教師データには使いません。
""",
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
        r"""## 5. 公式の報酬を基準に、研究用の項を追加する
`coverage = Tと目標の交差面積 / 目標の面積` です。IoU（交差面積÷合併面積）とは異なります。
標準報酬は公式実装と同じ `clip(coverage / 0.95, 0, 1)`、成功は `coverage > 0.95` です。
成功の静止時間や、位置・角度の個別閾値は追加しません。六足の転倒・領域外は失敗として終了します。

基準条件は面積重なり報酬だけです。研究条件だけに成功時の報酬を追加します。時間・胴体接触のコストも任意に設定できます。
これらの追加項は公式仕様ではないため、基準条件と区別して記録します。
報酬式の編集先は `pusht.py` の `pusht_reward_terms()` で、CPU・GPUは同じ式を使います。

公式の正の面積報酬をそのままRLで合計すると、成功直前に留まって報酬を得続ける方が、早く成功して終了するより有利になる場合があります。
これは報酬式からの推論で、位置表現の悪さとは別の問題です。
仮説：「成功時の追加報酬によって、成功直前の停滞を減らせるか？」
係数を強めても改善するとは限りません。実験前に予想を書いてください。
""",
    ),
    (
        "code",
        r"""from dataclasses import replace
from hexapod_transport_rl import PushTRewardWeights

baseline_reward = PushTRewardWeights()
changed_reward = replace(baseline_reward, success=200.0)
baseline_config = PushTConfig(num_robots=2, obs_type="state", reward_weights=baseline_reward)
conditions = {
    "coverage": baseline_config,
    "coverage_success": replace(baseline_config, reward_weights=changed_reward),
}
for name, config in conditions.items():
    print(name, config.reward_weights)
""",
    ),
    (
        "markdown",
        r"""## 6. 比較条件を揃える
各条件で同じ初期重み・seed・物理・配置・学習量・MAPPO設定を使います。
前の条件の重みを次へ渡しません。共有actor・全機の観測を見る中央criticを使います。
新しい課題なので運搬モデルをランダムな重みから学びます。標準予算は最初の比較用で、成功を保証する予算ではありません。

GPUでは1024世界・horizon 64・1,048,576チームステップ／条件です。
1回に65,536ステップ、各世界12.8秒分を収集して更新します。`NUM_ENVS` はロボット台数ではありません。
本実験ではGPUを使い、学習中は描画しません。CPU設定はAPIの接続確認用です。
まず1条件に絞って学習曲線・重なり率・時間を確認し、その後、全条件で学習量を揃えて増やします。
T4の時間は実際のログで測ります。環境準備・初回コンパイル・評価・動画は学習時間と分けます。

同じ最終課題へ固定し、途中で成功閾値や開始配置を変えません。
新規学習は旧課題の追加学習より時間がかかる可能性があります。旧モデルの実測時間を新仕様の時間として使いません。
""",
    ),
    (
        "code",
        r"""import torch
from hexapod_transport_rl.experiments import create_experiment

EXPERIMENT_NAME = "pusht_reward_trial_01"
TRAINING_SEED = 20261013
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_ENVS = 1024 if DEVICE.type == "cuda" else 2
HORIZON = 64 if DEVICE.type == "cuda" else 128
TRAINING_STEPS = 1048576 if DEVICE.type == "cuda" else 1024
RUN_DIR = create_experiment(PROJECT_DIR, EXPERIMENT_NAME)
print("学習先:", DEVICE, "収集量:", TRAINING_STEPS, "チームステップ/条件")
""",
    ),
    (
        "markdown",
        r"""## 7. TorchRLのMAPPOで学習する
ループは **Collectorで収集 → GAE → ミニバッチ更新 → 記録** の順です。
`ReplayBuffer` は今回の経験を混ぜるために使い、毎回空にします。
時間切れでは最後の観測の価値を使い、成功・転倒では使いません。
actorは連続したXY目標の分布を学び、学習時はサンプリング、評価時は分布の代表値を使います。
行動の境界はGymと同じメートル単位で、追加の座標変換を学生が書く必要はありません。

`PoseCurriculum` は最終課題を固定してCSV記録に使います。行動の生成やPPO更新は行いません。
保存する `.pt` に観測方式・XY行動の範囲・報酬・重み・Adam・価値統計を記録します。
旧速度指令モデルとは保存形式を分け、同じ入力次元でも別の観測として扱います。
""",
    ),
    (
        "code",
        r"""from torchrl.collectors import Collector
from torchrl.data import ReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
from hexapod_transport_rl import (
    TorchRLTransportEnv, MAPPOSettings, make_mappo_networks, make_mappo_loss,
    PoseCurriculum, save_mappo,
)

settings = MAPPOSettings(
    learning_rate=3e-4, minibatch_size=1024 if DEVICE.type == "cuda" else 128,
    epochs=4, value_normalization=True, gamma=0.995, gae_lambda=0.99,
)
FRAMES_PER_BATCH = NUM_ENVS * HORIZON
models = {}
for name, config in conditions.items():
    torch.manual_seed(TRAINING_SEED)
    output = RUN_DIR / name
    envs = TorchRLTransportEnv(config=config, num_envs=NUM_ENVS, backend="auto")
    envs.set_seed(TRAINING_SEED)
    actor, critic = make_mappo_networks(
        envs.num_robots, envs.obs_dim, action_dim=2,
        action_low=-config.workspace_half_extent, action_high=config.workspace_half_extent,
    )
    actor.to(DEVICE)
    critic.to(DEVICE)
    loss = make_mappo_loss(actor, critic, settings)
    optimizer = torch.optim.Adam(loss.parameters(), lr=settings.learning_rate)
    log = PoseCurriculum(config, envs, actor, output, seed=TRAINING_SEED,
        settings=settings, horizon=HORIZON, validate_every=None)
    save_mappo(output / "initial.pt", actor, critic, optimizer, config=config,
        provenance=envs.provenance, training=log.training, curriculum=log.state, loss=loss)
    collector = Collector(
        envs, actor, frames_per_batch=FRAMES_PER_BATCH, total_frames=TRAINING_STEPS,
        env_device=envs.device, policy_device=envs.device, storing_device=envs.device,
        auto_register_policy_transforms=True,
    )
    buffer = ReplayBuffer(storage=LazyTensorStorage(FRAMES_PER_BATCH, device=DEVICE),
        sampler=SamplerWithoutReplacement(), batch_size=settings.minibatch_size)
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
        log.record(batch, metrics)
        collector.update_policy_weights_()
    models[name] = save_mappo(output / "pusht.pt", actor, critic, optimizer, config=config,
        provenance=envs.provenance, training=log.training, curriculum=log.state, loss=loss)
    log.close()
    collector.shutdown()
""",
    ),
    (
        "markdown",
        r"""## 8. 保存したモデルとAPIで行動を決める
`policy_action()` はXY目標をそのまま返します。Gymの `step()` へ渡せます。
評価では探索のサンプリングを止め、固定した同じ方策の挙動を比較します。
""",
    ),
    (
        "code",
        r"""from hexapod_transport_rl import load_mappo, policy_action

actor, critic, saved = load_mappo(models["coverage"])
env = gym.make("HexapodPushT-v0", config=baseline_config)
observation, info = env.reset(seed=96000)
action = policy_action(actor, observation)
print("方策が決めたXY目標 [m]:", action)
observation, reward, terminated, truncated, info = env.step(action)
print("重なり率:", info["coverage"], "報酬:", reward)
env.close()
""",
    ),
    (
        "markdown",
        r"""## 9. 時間・成功率・重なり率を比較する
同じ学習量の最後のモデルを、全条件で同じ配置・95%の成功閾値・40秒の制限で評価します。
まず開発用12配置です。GPUで並列評価し、卒論の最終結果はCPU MuJoCoでも確認します。
総報酬の大小は追加項の異なる条件間の性能比較には使いません。
成功率、最大重なり率、失敗を40秒として集計した時間、接触、転倒を比較します。
""",
    ),
    (
        "code",
        r"""import pandas as pd
from hexapod_transport_rl import evaluate_pose

EVALUATION_EPISODES = 12
TEST_SEED = 96000  # 開発用。最終テストのseedは別に事前に決める。
EVALUATION_BACKEND = "warp" if DEVICE.type == "cuda" else "cpu"
reports = {}
rows = []
for name, checkpoint in models.items():
    training_log = pd.read_csv(RUN_DIR / name / "progress.csv")
    report = evaluate_pose(checkpoint, config=baseline_config,
        episodes=EVALUATION_EPISODES, seed=TEST_SEED, backend=EVALUATION_BACKEND,
        output=RUN_DIR / "evaluation" / f"{name}.json")
    reports[name] = report
    rows.append({
        "condition": name, "learning_seconds": training_log["elapsed_seconds"].iloc[-1],
        "success_rate": report["success_rate"], "max_coverage": report["mean_max_coverage"],
        "time_including_failures_s": report["mean_time_to_success_or_limit_s"],
        "body_contact_episodes": report["body_contact_episodes"], "falls": report["falls"],
        "evaluation_seconds": report["evaluation_seconds"],
    })
comparison = pd.DataFrame(rows)
comparison.to_csv(RUN_DIR / "comparison.csv", index=False)
display(comparison)
""",
    ),
    (
        "code",
        r"""import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
comparison.plot.bar(x="condition", y="success_rate", ax=axes[0], ylim=(0, 1), legend=False, rot=15)
comparison.plot.bar(x="condition", y="max_coverage", ax=axes[1], ylim=(0, 1), legend=False, rot=15)
axes[0].set_ylabel("Success rate")
axes[1].set_ylabel("Mean maximum coverage")
fig.tight_layout()
fig.savefig(RUN_DIR / "comparison.png", dpi=160)
plt.show()
""",
    ),
    (
        "markdown",
        r"""## 10. 同じ配置の動画で挙動を比べる
成功数だけでなく、回り込み、左右の押し方、Tの回転を観察します。
学習前と学習後を同じ配置で再生し、失敗も数値の評価へ含めます。映像は25 fps・実時間です。
初期モデルは歩行モデルだけが学習済みで、運搬の動作は未学習です。
""",
    ),
    (
        "code",
        r"""videos = []
video_names = []
video_models = {"before_training": RUN_DIR / "coverage/initial.pt", **models}
for name, checkpoint in video_models.items():
    video_dir = RUN_DIR / "videos" / name
    evaluate_pose(checkpoint, config=baseline_config, episodes=1, seed=96000,
        workers=1, video_dir=video_dir,
        output=RUN_DIR / "video_evaluation" / f"{name}.json")
    videos.append(media.read_video(next(video_dir.glob("*.mp4"))))
    video_names.append(name)
media.show_videos(videos, fps=25, titles=video_names, columns=2)
""",
    ),
    (
        "markdown",
        r"""## 11. 保存して卒論の比較へ広げる
コード・資産・設定・初期重み・学習ログ・評価JSON・CSV・図・動画を同じフォルダへ保存します。
最後のZIPをColabのファイル一覧から保存してください。ランタイム削除で結果は消えます。

1. 公式の重なり報酬を基準に、変更する項と仮説を先に決める。
2. 開発配置で予算を決め、全条件の学習量を揃える。成功がない場合も最大重なり率と動画で学習の進展を確認する。
3. 設定を固定して、学習seedを3種類以上・最終評価を50以上の未使用配置で繰り返す。
4. 平均・ばらつき、接触・転倒、CPUとGPUの違いも報告する。

`obs_type="keypoints"`との比較は、全条件で観測方式・初期重み・予算を揃えて別実験にします。
4台は `num_robots=4` で新しく学習します。2台のモデルをそのまま読みません。
[公式との対応表](https://github.com/hashimoto-robotics-lab/hexapod-transport-rl/blob/main/docs/pusht-protocol.md)と
[教材ガイド](https://github.com/hashimoto-robotics-lab/hexapod-transport-rl/blob/main/docs/colab-guide.md)に詳細があります。
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
