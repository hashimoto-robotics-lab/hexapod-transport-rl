# 歩行コマンドと学習環境API

実行手順は [README](../README.md)、内部構造は [コードガイド](code-guide.md) を参照してください。
歩行・運搬環境を、[Gymnasium標準のAPI](https://gymnasium.farama.org/api/env/) で操作します。
複数のロボットが同じ世界で接触するため、全機分の行動をまとめて1つの `step()` へ渡します。
学習用の報酬はチーム全体の1つの値です。

## 共通の操作

```python
import gymnasium as gym
import numpy as np
import mediapy as media
import hexapod_transport_rl  # 環境を登録する

env = gym.make("HexapodWalking-v0", num_robots=1, render_mode="rgb_array")
observation, info = env.reset(seed=42)
frames = [env.render()]
terminated = truncated = False
while not (terminated or truncated):
    action = np.array([[0.12, 0.0, 0.0]], dtype=np.float32)
    observation, reward, terminated, truncated, info = env.step(action)
    frames.append(env.render())
env.close()
media.show_video(frames, fps=5)
```

| 操作 | 意味 |
|---|---|
| `gym.make(id, ...)` | 環境を作り、固定した歩行モデルを読み込む |
| `reset(seed=..., options=...)` | 初期状態を作り、観測と診断情報を返す |
| `step(action)` | 全機を0.2秒進め、観測・報酬・終了・時間切れ・診断情報を返す |
| `render()` | `render_mode="rgb_array"` のときRGB画像を返す |
| `close()` | 使い終わった環境の描画資源を解放する |
| `action_space` / `observation_space` | 有効な行動・観測の形と範囲 |

`terminated` は成功や転倒、`truncated` は時間切れです。どちらかがTrueならエピソードを終えます。
自動resetはしないので、次のエピソードを始めるときは `reset()` を呼びます。
`gym.wrappers.RecordEpisodeStatistics` 等の標準ラッパーも使えます。

| 環境ID | 対応台数 | 観測 | 行動 |
|---|---|---|---|
| `HexapodWalking-v0` | 1〜4 | `(N, 6)` | `(N, 3)`、物理速度 |
| `HexapodPosePush-v0` | 2〜4 | `(N,18+4×(N−1))` | `(N,3)`、−1〜1 |
| `HexapodPush-v0` (`flatten=False`) | 2〜4 | `(N, 16 + 4×(N−1))` | `(N, 3)`、−1〜1 |

姿勢運搬は機体軸を保ち、`flatten`引数を使いません。
`HexapodPush-v0`は旧モデル向けの基本環境で、既定は`flatten=True`です。
Gymnasiumでは全機分の行動をまとめて1つの環境へ渡します。

## 学習済み歩行モデルへの指令

歩行環境の各行は `[vx, vy, yaw_rate]` です。指令は機体座標で、
前進 `vx` は−0.15〜0.20 m/s、左移動 `vy` は±0.10 m/s、左旋回 `yaw_rate` は±0.60 rad/sです。
歩行モデルの物理速度指令と、上位の学習方策の正規化行動を混同しないでください。
ゼロ行動は停止指令ですが、実際の減速には時間が必要です。

観測の6成分は `[世界x, 世界y, 向き, 前回vx指令, 前回vy指令, 前回旋回指令]` です。
最後の3成分は実測速度ではなく、送った目標速度です。
`info` に `positions`、`headings`、`commands`、`elapsed_seconds`、`robot_fall` を返します。
歩行体験用なので報酬は常に0です。転倒で `terminated=True`、`episode_seconds` に達すると `truncated=True` です。
制限時間は0.2秒刻みで指定します。

`reset(options={"poses": [[x, y, yaw], ...]})` で世界座標の配置を指定できます。
歩行APIは指定した配置から始め、ランダム配置は行いません。
録画しない場合は描画処理を実行しません。フレームは `render()` を呼んだときだけ取得します。
動画は `media.write_video("walking.mp4", frames, fps=5)` で保存できます。
内部では従来の `WalkingSimulation` と同じ歩行・モーター・接触計算を使います。
直接の `set_velocity()` / `run_for()` も既存スクリプトとの互換性のため利用できます。

## Tの姿勢運搬

```python
from hexapod_transport_rl import PosePushConfig, PoseRewardWeights

config = PosePushConfig(reward_weights=PoseRewardWeights(orientation=3.0))
env = gym.make("HexapodPosePush-v0", config=config, render_mode="rgb_array")
observation, info = env.reset(seed=42)
action = np.zeros(env.action_space.shape, dtype=np.float32)
observation, reward, terminated, truncated, info = env.step(action)
env.close()
```

Tの原点は2本の中心線の交点です。交点から横棒の左右端・縦棒の先端までの長さは等しく、
2台用では0.65 mです。太さ0.20 mの箱2つを重複しないよう接続し、面積に比例して質量を分配します。
目標マーカーは物理物体の形状をコピーした、接触しない表示です。
`PushConfig(shape="T")`もこの新しい形状を使います。

標準の目標はT基準で前方0.4 m・横方向±0.08 m、向きは初期Tから±5〜30度です。
課題全体の向きもランダム化します。各機はTの後方から、位置±0.04 m・向き±0.12 radの揺らぎを加えて開始します。
`PosePushConfig`に角度・距離・精度・報酬を指定します。学習段階の変更は次のresetでだけ適用します。

成功は0.2秒ごとの判定で、交点の位置誤差8 cm未満、向きの誤差5度未満、Tの速度0.06 m/s未満・角速度0.1 rad/s未満を1秒維持することです。
制限時間は12秒です。成功閾値は`position_tolerance`・`yaw_tolerance`・`success_hold_seconds`で変更します。

### 観測と行動

各機の観測はTの座標系で作ります。右側の機体はY方向と旋回の符号を反転し、左右の役割でactorを共有します。

| 番号 | 内容 | 尺度 |
|---|---|---|
| 0〜1 | Tに対する自分の位置 | 1 m |
| 2〜3 | Tに対する自分の向きのsin/cos | sinのみ左右反転 |
| 4〜6 | 自分の平面速度・角速度 | 歩行指令上限で割る |
| 7〜9 | 前回の速度指令 | 歩行指令上限で割る |
| 10〜11 | Tから目標への位置 | 1 m |
| 12〜13 | Tから目標への角度差のsin/cos | sinのみ左右反転 |
| 14〜16 | Tの平面速度・角速度 | 歩行指令上限で割る |
| 17 | 担当位置の横方向距離の絶対値 | Tの横幅で割る |
| 18以降 | 相手の相対位置2・相対向きのsin/cos | 他機ごとに4成分 |

行動は各機の`[前後, 左右, 旋回]`、範囲−1〜1です。右側は左右・旋回の符号を反転して実際の機体座標の指令へ戻します。
`policy_action(actor, observation)`と`rule_action(core, policy)`は、このAPIに渡す行動を返します。
共有・座標変換は構造上の工夫であり、学習行動をルールで生成するものではありません。

`info`には`distance`（m）、`yaw_error`（rad）、`cargo_speed`、`cargo_yaw_speed`、`reward_terms`、
接触力積・転倒・終了理由が入ります。`footprint_error_m`はTの3つの端と目標の対応する端のRMS位置誤差です。
位置・角度をまとめた見た目のズレの指標ですが、成功判定は位置と角度を個別に使います。

## TorchRLとモデル保存

```python
from hexapod_transport_rl import (
    TorchRLTransportEnv, MAPPOSettings, make_mappo_networks, make_mappo_loss,
)

envs = TorchRLTransportEnv(config=config, num_envs=64, backend="auto")
actor, critic = make_mappo_networks(envs.num_robots, envs.obs_dim)
loss = make_mappo_loss(actor, critic, MAPPOSettings())
envs.close()
```

学習の収集・GAE・更新ループはノートブックにあります。`num_envs`は独立した世界数です。
`backend="auto"` はNVIDIA CUDAでMuJoCo Warp、GPUがないときはCPUです。
明示指定は `backend="warp"` または `backend="cpu"`。省略時は既存コードとの互換性のためCPUです。
Warpは現在 `PosePushConfig` に対応し、観測・指令・固定歩行・接触・報酬をGPUに保持します。
Collectorの `env_device`・`policy_device`・`storing_device` は `envs.device` を指定します。
終了時の観測を保持し、リセットしたレーンだけ新しい難度と歩行履歴を初期化します。
`envs.check_physics()` は収集後のバッファ不足・非有限値検査、`envs.render(world=0)` は実際のGPU状態のRGB描画です。
actorは局所観測、criticは全機の観測を使います。報酬はチームで共有し、PPOの確率比は機体別です。
`PoseCurriculum`は検証・reset段階・ログを管理し、学習経験や行動を与えません。
`save_mappo()`はpose方策の場合、別の押すcheckpointを必要としません。
価値正規化を使うときは`loss=loss`も渡し、再開用の統計を保存します。
`anneal_exploration(actor, settings, progress)`はPPO更新後に呼び、次の収集で使う
探索ノイズの上限を下げます。`progress`は総学習予算に対する収集済みの割合です。
`load_mappo()`は保存した機体数と観測の形を復元し、標準ではCPUへ読み込みます。
GPUで学習を再開するときは`load_mappo(path, device="cuda")`を使えます。
`policy_action()`はactorの置かれたCPU/GPUへ観測を送り、Gymnasium用のNumPy行動を返します。
旧モデルとの重みの相互変換は行いません。

## 未使用seedで評価・録画

```python
from hexapod_transport_rl import evaluate_pose

report = evaluate_pose(
    "runs/pose_reward_trial_01/baseline/pose.pt",
    episodes=20, seed=84000, workers=2, output="runs/evaluation.json",
)
rule_report = evaluate_pose(
    config=config, policy="feedback", episodes=20, seed=84000, workers=2,
    output="runs/feedback.json",
)
```

`policy="forward"`は前進だけ、`"feedback"`は比例制御です。ルールは評価専用で学習の教師には使いません。
モデルの標準評価は、学習中の到達段階によらず保存された最終設定を使います。
比較するときは`config=baseline_config`を全条件へ指定して物理・成功判定を揃えます。
録画は`video_dir`を指定し、mediapyで5 fpsのMP4を保存します。録画時は逐次評価します。
位置・角度・T端の誤差、成功数・接触・転倒を返します。全試行の結果を含め、成功例だけを選んで集計しません。

## 旧モデルの再生

`HexapodApproach-v0`と`evaluate_transport()`は以前の形状で学習した参考モデル用です。
古いcheckpointに形状設定がない場合、読み込み側が`legacy`形状を明示して再生します。
新しい等長Tの性能と旧実験の成功率を混同しないでください。
