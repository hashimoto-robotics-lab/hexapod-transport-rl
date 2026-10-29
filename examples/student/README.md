# 六足1台の実験サンプル

まず[学生用APIの説明](../../docs/student-api.md)を読み、リポジトリ直下から実行します。

```bash
PYTHONPATH=src MUJOCO_GL=egl .venv/bin/python tools/train_student.py \
  examples/student/reward_only.py --output runs/student_reward_smoke --smoke
```

本学習ではCUDA対応Pythonを使い、`--smoke`を外します。
`baseline.py`、`reward_weights.py`、`reward_only.py`、`observations.py`、`actions.py`は回転対称化なしの共通条件です。
`custom_observation.py`は関数を追加する例、`pretrained.py`は既存モデルの構成を再現する例です。
各ファイルは`TASK`を定義し、学習と評価は共通CLIに任せます。

報酬の例は2つあります。

| 課題ファイル | 編集する内容 |
|---|---|
| [reward_weights.py](reward_weights.py) | `goal_progress`の重みを20から40に変更 |
| [reward_only.py](reward_only.py) | 目標の近くで円柱の速度を抑える関数を追加 |

```bash
MUJOCO_GL=egl .venv/bin/python tools/train_student.py \
  examples/student/reward_weights.py --output runs/reward_weights_s42 --smoke
```

[ColabのStep 5と7](../../notebooks/student_api_colab.ipynb)では、標準課題と同じseed・サンプル数で学習し、成功率・距離・動画を比較できます。

```bash
PYTHONPATH=src MUJOCO_GL=egl .venv/bin/python examples/student/compare.py \
  --output runs/student_comparison_smoke --smoke
```

比較結果は`comparison.csv`と`comparison.json`です。
短い接続試験の成功率は、運搬を学習した成績として使いません。
