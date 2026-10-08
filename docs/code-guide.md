# コードを読む順番

学生の学習は普通のPythonセルからTorchRLを呼び出します。
まず歩行APIを試し、運搬の観測・行動・報酬を理解してから収集・更新ループを読んでください。

| 順番 | ファイル | 内容 |
|---|---|---|
| 1 | `walking_env.py` | 学習済み歩行モデルへの速度指令、Gymnasium API |
| 2 | `pose_push.py` | Tの目標姿勢・観測・T座標の行動・運搬Gym API |
| 3 | `rewards.py` | `PoseRewardWeights`の変更可能な係数 |
| 4 | `torchrl_env.py` | Gymの物理環境をTorchRLのTensorDictへ接続 |
| 5 | `torchrl_mappo.py` | TorchRLのactor・critic・MAPPOLoss・保存・読み込み |
| 6 | `pose_training.py` | 学習とは別の検証、resetの難度変更、CSV記録 |
| 7 | `pose_evaluation.py` | 最終精度での評価、比較用ルール、MP4録画 |
| 8 | `pose_gpu_evaluation.py` | 同じCPU初期配置でのGPU並列評価、終了済み試行の除外 |

収集・GAE・ミニバッチ更新・集計・図・動画表示はノートブックにあります。
`PoseCurriculum`は行動や教師データを作らず、PPO更新も行いません。
`tools/colab_runtime.py`はインストールと描画準備、`experiments.py`は実験コードのコピーとZIP保存だけを担当します。
教材を変更するときは`tools/build_colab.py`を編集し、`uv run python tools/build_colab.py`で再生成します。

## 研究で変更する場所

| 変更 | 定義 |
|---|---|
| 報酬係数 | `PoseRewardWeights`とノートブックの`replace()` |
| 報酬の式 | `pose_rewards.py`の`pose_reward_terms()`（CPU・GPU共通） |
| 接近コストの形 | `push_approach.py`：Tの外側の距離と食い込みのコスト。行動は作らない |
| 初期角度・距離・精度 | `PosePushConfig`・`POSE_STAGES` |
| 観測と左右の座標変換 | `observe_pose()`・`PosePushEnv.step()` |
| 学習率・更新回数・PPO clip | `MAPPOSettings` |
| 速度の候補と確率 | notebookの`ACTION_GRID`・`velocity_distribution.py` |
| 検証と段階移行 | `PoseCurriculum` |
| 比較する比例制御 | `rule_action()`、評価専用 |
| Tの等長形状と質量配分 | `model.py`の`_add_cargo()` |

学生用の標準は2台、共通のRL学習済みモデルから262,144チームステップ／条件の追加学習です。
時間と胴体接触のコストの2×2比較を行います。ランダムな重みからの新規学習は別ノートブックへ分けています。
物理の外力やデモの行動は与えません。追加学習時間・成功率・途中の誤差・接触を評価します。事前学習時間は別に報告します。

## 共通の物理基盤

`PosePhysics`は既存の`PushEnv`と同じ関節・接触・歩行モデルを使い、reset・観測・報酬だけを変更します。
`env.py`は物理の更新・接触計測・終了判定、`model.py`は機体の複製・床・T・目標を作ります。
`locomotion.py`は固定歩行モデルとモーター、`contacts.py`は胴体・脚リンク・足の接触力積を扱います。
ロボット・歩行の資産は`assets`に同梱しています。

GPU用の詳細を読む場合は `warp_pose.py`（並列reset・観測・終了判定）、
`warp_physics.py`（歩行・元のDCモーター・CUDA graph）、
`warp_contacts.py`（各物理ステップの接触・力積）の順です。
`pose_rewards.py`の式と`PoseRewardWeights`はCPU・GPU双方が使います。
学生が報酬係数を変える場合、Warpカーネルの編集は必要ありません。

`approach.py`・`approach_training.py`・`handover_training.py`・`mappo.py`は旧形状で学習した歴史的なモデルの再生・再学習用です。
新しい姿勢運搬の学習ループはこれらを使いません。
旧モデルの検証を残し、新しい形状へ置き換えた結果として誤って再生しないようにします。
