# 公式Push-Tとの対応

基準は [gym-pusht](https://github.com/huggingface/gym-pusht/blob/main/gym_pusht/envs/pusht.py) と
[Diffusion PolicyのPushTEnv](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/env/pusht/pusht_env.py) です。
六足の物理と以前指定した開始配置を保ち、課題の観測・行動・報酬・成功判定を合わせました。
名称は「六足ロボットによる協調Push-T」で、公式2Dベンチマークそのものとは区別します。

| 項目 | 公式2D版 | 六足版 |
|---|---|---|
| 状態版の先頭 | 押し手XY、TのXY、Tの角度 | 各機で同じ5成分、メートル単位 |
| キーポイントの内容 | gym-pushtのTの8頂点XY＋押し手XY | 同じ内容を各機の先頭18成分に並べる |
| 六足用の追加観測 | 円形押し手なので姿勢なし | 自機の向き1、他機のXY・向き3/機 |
| 座標 | 世界座標 | 世界座標、左右反転なし |
| 行動 | 押し手のXY目標位置 | 各機のXY目標位置、形 `(N,2)` |
| 報酬 | `clip(coverage / .95, 0, 1)` | 標準は同じ式のみ |
| 成功 | `coverage > .95` | 同じ。静止・維持時間は要求しない |
| 目標 | 固定位置・固定向き | 固定位置・固定向き |
| 初期配置 | 押し手・Tの位置と角度を独立にランダム化 | T→ゴール→ロボット、位置・向きに揺らぎ |
| Tの形状 | 既定の2矩形 | 以前指定した等長2棒（2台で各1.3 m） |
| 物理 | 2D Pymunk、円形押し手1つ | 3D MuJoCo/Warp、脚・床・Tの接触、2〜4台 |
| 制御周期 | 10 Hz | 標準5 Hz／学生用1.25 Hz、歩行25 Hz、物理・モーター200 Hz |
| 安全終了 | 2D版に六足の転倒なし | 転倒・領域外も失敗として終了 |

物理・形状・初期化・制御周期は同一ではないので、公式論文の時間・成功率と直接比較しません。
固定目標は `[0.4 m, 0 m, π/4 rad]` です。公式の画面座標 `[256,256,π/4]` をそのままメートルとしては使いません。
初期場面全体をreset時に剛体変換して目標を固定し、T→ゴール→ロボットの関係を保ちます。
途中の状態・Tの運動を修正する処理はありません。

## 観測

標準の `PushTConfig(obs_type="state")` は各機の先頭を次にします。

```text
[自分x, 自分y, Tのx, Tのy, Tの角度, 自分の角度, 他機1のx,y,角度, ...]
```

角度は0〜2πです。相対座標・sin/cos・担当する押し位置・前回指令は標準へ追加しません。
自分の角度は円形ではない六足の位置追従に必要な情報、他機の姿勢は協調作業の拡張です。
2台は各機9成分、4台は15成分です。actorは各機の1行、criticは全機の行を見ます。
目標が固定なので目標の座標を観測へ重複して渡しません。目標を変える場合は別の課題として学習します。

`obs_type="keypoints"` の先頭18成分は、Tの2矩形の4頂点ずつのXYと、自分のXYです。
向きと他機を追加して、2台は22成分、4台は28成分です。旧課題の22成分とは意味が異なります。
gym-pushtの`environment_state_agent_pos`はキー点16成分と押し手XYを辞書で返します。本版は全機の配列へ並べます。
Diffusion Policyのキー点環境には可視性マスクもありますが、本版は全頂点を観測し、マスクは追加しません。

## XY目標と歩行の接続

```python
import gymnasium as gym
import numpy as np
import hexapod_transport_rl
from hexapod_transport_rl import PushTConfig

env = gym.make("HexapodPushT-v0", config=PushTConfig())
observation, info = env.reset(seed=42)
targets = np.asarray(info["agent_pos"], dtype=np.float32)
observation, reward, terminated, truncated, info = env.step(targets)
print(info["coverage"], info["reward_terms"])
env.close()
```

`[0,0]`は原点への移動で、停止指令ではありません。現在位置を指定するとその場に留まろうとします。
行動範囲は各軸±4 mです。MAPPOはこの物理単位の範囲へ収まる2次元の連続分布を学びます。
位置追従は自分の位置・速度・向きとXY目標だけから、元の歩行モデルの速度指令を作ります。
Tの位置・目標姿勢から行動を生成するルール、教師行動は含みません。
研究用の接近報酬では、Tを避けて後方の押す位置へ近づいた距離を評価します。この状態評価は方策へ経路・行動を渡しません。
歩行の速度・旋回上限は元のモデルの範囲です。位置制御のゲインは物理単位と機体に合わせており、公式の円形押し手のPDゲインをそのままコピーしていません。

## 面積計算と成功

`coverage = intersection(T, goal).area / goal.area` です。IoUの分母である合併面積は使いません。
物理モデルと同じ、重ならない2矩形の組み合わせを使います。4通りの凸多角形の交差面積を合計します。
CPUはNumPy、GPUはTorchの多角形クリッピングで計算し、GPUの状態を毎ステップCPUへ転送しません。
平面XY・yawによる投影面積で、転倒は別の物理条件として失敗にします。

以前の位置8 cm・角度5度・低速1秒は `HexapodPosePush-v0` の条件です。
新しい `HexapodPushT-v0` の成功には使いません。終了は成功・転倒・領域外、時間切れは`truncated`です。
CPUとGPUは数値精度が異なるので、最終性能は両方で評価します。

## 強化学習の報酬比較

`PushTRewardWeights()`は`coverage=1`、追加項はすべて0です。これが公式に対応する基準条件です。
`progress`・`approach`・`success`・`time`・`body_contact`・`robot_contact`・`command_change`・`failure`は研究用に明示して追加する項です。
成功閾値は両条件とも95%のままです。追加項のある結果を「公式報酬だけの結果」とは扱いません。

公式の正の面積報酬をRLで合計すると、成功直前に留まって長く報酬を得る方が、早く成功して終了するより有利になる場合があります。
これは報酬と終了の式からの推論です。学生用Colabは1条件ずつ実行します。公式報酬だけを基準に、重なり率の増加・接近・成功時報酬などを明示して追加します。
成功時報酬を加えれば必ず学べるわけではありません。成功がまだ探索されていなければ、その項から学習する機会もありません。

## 保存・過去の結果

新しい保存形式は `hexapod-pusht-torchrl-mappo-v1` です。観測方式とXY行動の範囲も保存します。
旧モデルは旧観測・速度指令・姿勢誤差の条件で再生し、新しい課題へそのままロードしません。
旧課題の約42秒の追加学習、95〜97%の平均成功率は新仕様の実績ではありません。
[旧課題のColab](https://colab.research.google.com/github/hashimoto-robotics-lab/hexapod-transport-rl/blob/main/notebooks/archive/hexapod_transport_rl_pose_legacy.ipynb)で再現できます。

新仕様の初期試験は [実行記録](results/pusht_protocol_20261013.json) に保存します。
時間を測って実行できたことと、95%の運搬を習得したことを区別します。

## 短時間の新規実験

現在のColabは`train_mappo()`・`evaluate_mappo()`で1条件を実行します。運搬actor・critic・Adam・価値統計は毎回新規です。
方策の分布は自分のXYを基準に置き、学習する残差から世界XY目標を出します。T・ゴールによる行動の補正はありません。
標準観測のメートル・ラジアン、XY目標の範囲、95%の成功は同じです。

追加項は `progress = weight * (coverage_after - coverage_before)`、
`approach = weight * (outside_path_before - outside_path_after)` です。公式報酬や方策不変性を保証するpotential shapingとは区別します。
単純な増加量なので、割引・終了を含めて最適方策が保存されるという主張はしません。

学習に360秒の予算を指定し、数値評価8配置・動画1本を後で実行します。終了はロールアウトの境界です。
時間内の95%成功、4台の学習性能、Colab/T4の所要時間は保証しません。
[全試行](results/compact_experiments_20261013.json)に学習時間・全体時間・失敗を含めて記録します。

学生用実験では`high_level_decimation=20`、上位1.25 Hzです。公式10 Hzとは異なります。
γ・GAE λは基準0.2秒の値を制御間隔に応じてべき乗し、物理時間あたりの割引を揃えます。
