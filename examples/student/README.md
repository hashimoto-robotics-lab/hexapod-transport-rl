# 六足1台の実験サンプル

まず[学生用APIの説明](../../docs/student-api.md)を読み、リポジトリ直下から実行します。

```bash
PYTHONPATH=src MUJOCO_GL=egl .venv/bin/python tools/train_student.py \
  examples/student/reward_only.py --output runs/student_reward_smoke --smoke
```

本学習ではCUDA対応Pythonを使い、`--smoke`を外します。
`baseline.py`、`reward_only.py`、`observations.py`、`actions.py`は回転対称化なしの共通条件です。
`custom_observation.py`は関数を追加する例、`pretrained.py`は既存モデルの構成を再現する例です。
各ファイルは`TASK`を定義し、学習と評価は共通CLIに任せます。

```bash
PYTHONPATH=src MUJOCO_GL=egl .venv/bin/python examples/student/compare.py \
  --output runs/student_comparison_smoke --smoke
```

比較結果は`comparison.csv`と`comparison.json`です。
短い接続試験の成功率は、運搬を学習した成績として使いません。
