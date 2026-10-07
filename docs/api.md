# 歩行コマンドと学習環境API

実行手順は [README](../README.md)、内部構造は [コードガイド](code-guide.md) を参照してください。
歩行・回り込み・押す環境を、[Gymnasium標準のAPI](https://gymnasium.farama.org/api/env/) で操作します。
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
| `HexapodApproach-v0` | 2 | `(2, 10)` | `(2, 3)`、−1〜1 |
| `HexapodPush-v0` (`flatten=False`) | 2〜4 | `(N, 16 + 4×(N−1))` | `(N, 3)`、−1〜1 |

回り込みで `flatten=True` にすると観測 `(20,)`・行動 `(6,)` です。
押す環境は既定が `flatten=True` で、2台なら観測 `(40,)`・行動 `(6,)` です。
Gymnasiumではチームを1つの意思決定主体として扱います。これは機体別の辞書を返すPettingZooのAPIとは異なります。

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

## 押す環境

```python
from hexapod_transport_rl import PushConfig

env = gym.make("HexapodPush-v0", config=PushConfig(shape="T"), flatten=False)
obs, info = env.reset(seed=42)
action = env.action_space.sample()  # 実際の学習ではactorの出力
next_obs, reward, terminated, truncated, info = env.step(action)
env.close()
```

| 項目 | 2台の場合 |
|---|---|
| 観測 | `float32 (2, 20)` |
| 行動 | `float32 (2, 3)`、各値 `[-1, 1]` |
| 報酬 | チーム共通のfloat |
| 中央critic入力 | `env.unwrapped.state()` の `(40,)` |
| 制御周期 | 1 step = 0.2秒、歩行25 Hz、MuJoCo物理200 Hz |

各機の行動は機体座標の `[前後, 左右, 旋回]` です。
前進上限0.20 m/s、後退上限0.15 m/s、左右0.10 m/s、旋回0.60 rad/sへ変換します。
ゼロ指令は停止です。関節は固定した学習済み歩行モデルが制御します。

コンストラクタの既定は `flatten=True` で、観測 `(40,)`、行動 `(6,)` になります。
`PushConfig` の既定形状は箱のため、APIを直接使う場合は **`shape="T"` を指定**してください。
`run.sh` の学習コマンドは2台のT字運搬を選びます。

局所観測の添字は以下です。位置ベクトルは観測する機体の座標へ変換します。

| 添字 | 内容 |
|---|---|
| `0:2` | 自機→荷物のXY / 3 m |
| `2:4` | 荷物→目標のXY / 3 m |
| `4:7` | 自機のXY速度とyaw角速度 |
| `7:9` | 荷物と自機のyaw差のsin・cos |
| `9:11` | 目標と自機のyaw差のsin・cos |
| `11:14` | 直前の速度指令 / `[0.20, 0.10, 0.60]` |
| `14` | 自機の上向き軸と世界Z軸の内積 |
| `15` | 担当する押し位置の横座標 / 荷物幅 |
| `16:20` | 相手の相対XY / 3 mとyaw差のsin・cos |

## 回り込み環境

```python
from hexapod_transport_rl import ApproachConfig

env = gym.make("HexapodApproach-v0", config=ApproachConfig(layout="front"))
obs, info = env.reset(seed=42)
next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

`reset(options={"layout": "side"})` でそのエピソードだけ初期配置を指定できます。
`config` は `ApproachConfig` または同じフィールドの辞書を受け取ります。
診断情報には `state`、実際の機体速度指令 `commands`、`layout`、`is_success`、
`episode_return`、`termination_reason`、`reward_terms` が入ります。

観測は `(2, 10)`、行動は `(2, 3)` です。
`observe_approach()` がT基準の位置・向き・自機速度・前回指令を作り、左右の機体を共通の座標へ反転します。
`physical_actions()` がactorの出力を実際の左右・旋回方向へ戻します。この変換は `ApproachEnv.step()` 内で行います。

回り込みの成功は、2台がT後方の担当位置 `x=-1.05 m, y=±0.325 m` へ近づいて向きを合わせることです。
回り込み中の成功判定と、Tを目標まで運ぶ成功判定は別です。
観測の正確な並びと正規化は `approach.py` の `observe_approach()` を参照してください。

## 終了、診断、並列化

`reset(seed=...)` は `(obs, info)`、`step(action)` は `(obs, reward, terminated, truncated, info)` を返します。
`terminated` は成功・失敗、`truncated` は時間切れです。最終観測を返した後、自動resetはしません。
押す環境での成功は位置誤差0.18 m未満、yaw誤差0.25 rad未満、速度0.06 m/s未満、yaw速度0.10 rad/s未満を0.5秒以上保つことです。

`info` に成功、目標までの距離、yaw誤差、接触、報酬内訳を返します。
押す環境の `reward_terms` は距離・向き・押す位置への接近、指令変化、時間、衝突、胴体接触、終了報酬を含みます。
`body_normal_impulse_ns`、`leg_link_normal_impulse_ns`、`foot_normal_impulse_ns` は各機の部位別の法線力積です。

押す学習の `make_vector_env(B, flatten=False, asynchronous=True)` はB個の独立世界を別プロセスで動かします。
観測 `(B, 2, 20)`、行動 `(B, 2, 3)`、報酬 `(B,)` になります。
回り込み学習は `make_approach_vector()` を使い、観測が `(B, 2, 10)` になります。
終了した世界だけ `reset(options={"reset_mask": terminated | truncated})` でリセットします。
reset前に最終観測から価値を計算する処理は `mappo.py` に共通化しています。

別プロセスを使う自作スクリプトでは `if __name__ == "__main__":` で起動処理を囲み、最後に `close()` を呼んでください。
`HexapodPushEnv(render_mode="rgb_array")` はRGB画像、`render_mode="human"` はGUIを提供します。
運搬全体の25 fps録画は `./run.sh eval --video-dir ... --episodes 1` を使ってください。

## 報酬の比較実験API

```python
from dataclasses import replace
from hexapod_transport_rl import ApproachConfig, ApproachRewardWeights

reward = replace(ApproachRewardWeights(), robot_contact=12.0)
config = ApproachConfig(layout="front", reward_weights=reward)
env = gym.make("HexapodApproach-v0", config=config, render_mode="rgb_array")
observation, info = env.reset(seed=80000)
observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
terms = info["reward_terms"]  # 合計がこのstepのチーム報酬
frame = env.render()
env.close()
```

回り込みの観測は `(2, 10)`、行動は `(2, 3)` の−1〜1です。
担当する左右の役割を反転した座標で行動を出し、内部で実際の機体速度へ戻します。
中央critic用の `env.unwrapped.state()` は両機の観測を並べた `(20,)` です。
`train_approach(config=config, ...)` へ同じ設定を渡すと、並列環境もその報酬で学習し、
係数をcheckpointへ保存します。既存モデルの読み込みは従来の既定値を補います。
式は `rewards.approach_reward_terms()`、押す報酬は `PushConfig(reward_weights=PushRewardWeights(...))` で設定します。

`evaluate_transport(checkpoint=..., output=..., episodes=..., seed=..., layout=...)` は
回り込みから押すまでを評価し、物理的な成功・接触・転倒・誤差をJSONに保存して辞書を返します。
異なる報酬の実験同士を比較するときは、総報酬ではなくこれらの指標を使います。
