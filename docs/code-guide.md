# 学生が読む6ファイル

| ファイル | 読む目的 |
|---|---|
| `walking_env.py` | 学習済み歩行モデルへ速度を送るGym API |
| `pusht.py` | XY目標、観測、報酬、95%の成功判定 |
| `training.py` | TorchRLの収集・GAE・ミニバッチ更新・記録 |
| `torchrl_mappo.py` | 共有actor、中央critic、損失、保存・再読込 |
| `evaluation.py` | 数値評価と動画1本を呼び出すAPI |
| `experiments.py` | 編集したコードの記録とZIP保存 |

学生が主に変えるのはノートブックの`PushTRewardWeights`です。
式を変える場合は`pusht.py`の`pusht_reward_terms()`を編集します。CPU・GPUは同じ式を使います。

面積は`pusht_geometry.py`、回り込みの状態評価は`push_approach.py`、GPU接続は`torchrl_env.py`・`warp_pusht.py`です。
`warp_physics.py`・`warp_contacts.py`は固定歩行・モーター・物理・接触の実装です。
元のsolver・機体・床・Tの物理設定を使い、cargoへの外力は加えません。

`pose_push.py`は共有する物理・resetと旧課題の互換性、`pose_evaluation.py`は既存の評価・録画処理です。
過去の`approach`・`handover`・独自`mappo`は旧モデル用で、現在の学生用更新ループは使いません。
歴史的な教材は[archive](archive/)に分け、以前の成果を新規学習の成果として表示しません。
