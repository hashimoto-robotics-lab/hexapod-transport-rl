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

[Colab](../../notebooks/student_api_colab.ipynb)の前半は、標準条件と報酬1項目の変更を実際に学習した完成例です。
Step 8〜10で自分の仮説から新規学習・複数seed比較・作図を進め、Step 11で独自報酬も試せます。
[reward_study.py](reward_study.py)は完成例の固定条件です。実測値は[results/reward_study](../../results/reward_study)にあります。

```bash
PYTHONPATH=src MUJOCO_GL=egl .venv/bin/python examples/student/compare.py \
  --output runs/student_comparison_smoke --smoke
```

比較結果は`comparison.csv`と`comparison.json`です。
短い接続試験の成功率は、運搬を学習した成績として使いません。
