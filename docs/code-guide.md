# コードを読む順番

学生の学習は普通のPythonセルからTorchRLを呼び出します。
まず歩行APIを試し、運搬の観測・行動・報酬を理解してから収集・更新ループを読んでください。

| 順番 | ファイル | 内容 |
|---|---|---|
| 1 | `walking_env.py` | 学習済み歩行モデルへの速度指令、Gymnasium API |
| 2 | `pose_push.py` | Tの目標姿勢・観測・行動の反転・報酬・運搬Gym API |
| 3 | `rewards.py` | `PoseRewardWeights`の変更可能な係数 |
| 4 | `torchrl_env.py` | Gymの物理環境をTorchRLのTensorDictへ接続 |
| 5 | `torchrl_mappo.py` | TorchRLのactor・critic・MAPPOLoss・探索量・保存・読み込み |
| 6 | `pose_training.py` | 学習とは別の検証、resetの難度変更、CSV記録 |
| 7 | `pose_evaluation.py` | 最終精度での評価、比較用ルール、MP4録画 |

収集・GAE・ミニバッチ更新・集計・図・動画表示はノートブックにあります。
`PoseCurriculum`は行動や教師データを作らず、PPO更新も行いません。
`tools/colab_runtime.py`はインストールと描画準備、`experiments.py`は実験コードのコピーとZIP保存だけを担当します。
教材を変更するときは`tools/build_colab.py`を編集し、`uv run python tools/build_colab.py`で再生成します。

## 研究で変更する場所

| 変更 | 定義 |
|---|---|
| 報酬係数 | `PoseRewardWeights`とノートブックの`replace()` |
| 報酬の式 | `PosePhysics._reward_terms()` |
| 初期角度・距離・精度 | `PosePushConfig`・`POSE_STAGES` |
| 観測と左右の座標変換 | `observe_pose()`・`PosePushEnv.step()` |
| 学習率・更新回数・PPO clip | `MAPPOSettings` |
| 検証と段階移行 | `PoseCurriculum` |
| 比較する比例制御 | `rule_action()`、評価専用 |
| Tの等長形状と質量配分 | `model.py`の`_add_cargo()` |

ノートブックの標準は2台・65,536チームステップ／条件です。
同じ初期重みと学習量で角度の補助報酬の有無を比較します。
物理の外力やデモの行動は与えません。精度に達するまでの学習時間も評価項目です。

## 共通の物理基盤

`PosePhysics`は既存の`PushEnv`と同じ関節・接触・歩行モデルを使い、reset・観測・報酬だけを変更します。
`env.py`は物理の更新・接触計測・終了判定、`model.py`は機体の複製・床・T・目標を作ります。
`locomotion.py`は固定歩行モデルとモーター、`contacts.py`は胴体・脚リンク・足の接触力積を扱います。
ロボット・歩行の資産は`assets`に同梱しています。

`approach.py`・`approach_training.py`・`handover_training.py`・`mappo.py`は旧形状で学習した歴史的なモデルの再生・再学習用です。
新しい姿勢運搬の学習ループはこれらを使いません。
旧モデルの検証を残し、新しい形状へ置き換えた結果として誤って再生しないようにします。
