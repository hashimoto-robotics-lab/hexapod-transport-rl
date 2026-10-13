# 学生が読むコード

現在の標準は公式Push-Tに近づけたXY目標・面積報酬の課題です。
歩行指令から始め、観測・行動・報酬を理解してからTorchRLの収集・更新を読みます。

| 順番 | ファイル | 内容 |
|---|---|---|
| 1 | `walking_env.py` | 固定歩行モデルへ速度を送るGym API |
| 2 | `pusht.py` | 状態・キーポイント観測、XY追従、報酬、成功判定、Gym API |
| 3 | `pusht_geometry.py` | 物理と同じ2矩形の交差面積、CPU/GPUのcoverage |
| 4 | `torchrl_env.py` | 世界・ロボット軸を保ったTorchRLの接続 |
| 5 | `torchrl_mappo.py` | 共有actor・中央critic・MAPPO損失・保存・読込 |
| 6 | `pose_training.py` | 固定課題のCSV記録、旧課題ではカリキュラム管理 |
| 7 | `pose_evaluation.py` | 保存形式を判別する評価、25 fps録画 |

学生が主に変えるのはノートブックの`PushTRewardWeights`と`replace()`です。
式を変える場合は `pusht_reward_terms()` を編集します。CPUとGPUが同じ式を使います。
収集・GAE・ミニバッチ更新はノートブックに直接書き、独自クラスへ実行を隠しません。
`tools/build_colab.py`が教材のセル定義、`tools/colab_runtime.py`が準備、`experiments.py`がコード記録とZIP保存です。

GPUの詳細は `warp_pusht.py`（固定目標・観測・XY追従・成功）、`warp_physics.py`（歩行・モーター・物理）、
`warp_contacts.py`（接触・力積）、`pose_gpu_evaluation.py`（終了済み試行を除外する評価）の順です。
位置追従はロボットとXY目標だけを見ます。Tやゴールから経路・教師行動を作りません。

`pose_push.py`・`pose_rewards.py`は旧課題の相対姿勢・速度指令・静止判定です。
旧チェックポイントの再現性を保ち、新課題の学習済みモデルとして扱いません。
過去の`approach`・`handover`・`mappo`も歴史的なモデル用で、学生の新規Push-T更新ループでは使いません。
