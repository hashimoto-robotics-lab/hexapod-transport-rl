# 学生用API

## 歩行

```python
import gymnasium as gym
import numpy as np
import hexapod_transport_rl

env = gym.make("HexapodWalking-v0", num_robots=1)
observation, info = env.reset(seed=42)
action = np.array([[0.12, 0.0, 0.0]], dtype=np.float32)
observation, reward, terminated, truncated, info = env.step(action)
env.close()
```

歩行の行動は `(N,3)` の `[vx,vy,yaw_rate]`。機体座標の物理速度です。
範囲は−0.15〜0.20 m/s、±0.10 m/s、±0.60 rad/s。ゼロは停止指令です。
歩行モデルは固定で、このAPIは新しい学習を行いません。

## 協調Push-T

```python
from hexapod_transport_rl import PushTConfig

config = PushTConfig()
env = gym.make("HexapodPushT-v0", config=config, render_mode="rgb_array_list")
observation, info = env.reset(seed=42)
targets = np.asarray(info["agent_pos"], dtype=np.float32)
observation, reward, terminated, truncated, info = env.step(targets)
frames = env.render()
env.close()
```

運搬の行動は `(N,2)` の世界XY目標 [m]、各軸±4 mです。`[0,0]`は原点への移動です。
現在位置を指定すると、その場に留まろうとします。
`state`観測は `(N,3+3N)`、`keypoints`は `(N,16+3N)`。
`info`には位置・姿勢・`coverage`・`reward_terms`・接触・転倒が入ります。
`render()`は前回取得後の実際の25 Hzフレームを返します。mediapyの`fps=25`で再生します。
終了したエピソードは明示的に`reset()`します。`terminated`は成功・失敗、`truncated`は時間切れです。

## 学習と評価

```python
from hexapod_transport_rl import train_mappo, evaluate_mappo

checkpoint = train_mappo(config, "runs/trial_01", seed=42, max_seconds=360)
report = evaluate_mappo(checkpoint, episodes=8, seed=98000)
```

`train_mappo()`は運搬actor・critic・Adam・価値統計を新しく作ります。
既存の運搬モデルを読む引数や教師行動はありません。歩行だけは既存モデルを使います。
時間予算は環境構築・収集・GAE・更新を含み、次のロールアウトが予算に収まるかを見て終了します。
GPUでは512世界、horizon 32、2 epochsを使います。CPU設定はAPI確認用です。
試運転で予算を決め、`steps=262144, max_seconds=None`等で同じステップ数の比較にできます。

数値評価はCUDAがあればWarp、動画1本はCPU MuJoCoです。`video=False`で録画を省けます。
最大重なり率・成功率・失敗込み時間・接触・転倒を比較します。
保存・直接のTorchRL接続の詳細は[コードガイド](code-guide.md)、課題の全仕様は[公式との対応](pusht-protocol.md)です。
過去のAPIは[旧資料](archive/api.md)に保存しています。

学生用の運搬ではXY目標を0.8秒保ちます（`high_level_decimation=20`）。標準Gym設定は0.2秒です。
歩行25 Hz・物理200 Hzは同じで、記録する動画は実際の25 Hzフレームです。
