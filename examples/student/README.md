# 学生が編集する課題ファイル

最初は [Colab](../../notebooks/student_api_colab.ipynb) で、標準条件をゼロから学習します。

| ファイル | 内容 |
|---|---|
| [reward_study.py](reward_study.py) | 完成例の2条件・3 seed・共通の学習と評価の設定 |
| [baseline.py](baseline.py) | 14入力・停止＋8方向・標準報酬・共通の回転対称化 |
| [reward_weights.py](reward_weights.py) | 進捗報酬の重みだけを20から40へ変更 |
| [reward_only.py](reward_only.py) | 目標付近の円柱速度を減点する独自関数を追加 |
| [observations.py](observations.py) / [actions.py](actions.py) | 観測・行動を変更する定義 |
| [custom_observation.py](custom_observation.py) | 自作の観測関数 |
| [compare.py](compare.py) | 観測・行動の比較とCSV保存のAPI例 |

## 報酬を変更してゼロから学習する

[READMEのGPU環境](../../README.md#ローカルで実行する) で、リポジトリ直下から実行します。

```bash
MUJOCO_GL=egl python tools/train_student.py examples/student/reward_weights.py \
  --output runs/my_reward_s42 --backend warp --seed 42 \
  --steps 9830400 --num-envs 512 --horizon 64 \
  --validation-envs 256 --validate-every 10 --curriculum \
  --checkpoint-selection best \
  --episodes 1024 --evaluation-seed 940000 --video
```

initial-checkpointは指定しません。ランダムなActor・Critic、空のAdamから開始します。
curriculumで開始配置を広げ、最後は位置・向き・目標が全方向ランダムな課題で学習します。
同じ設定・seedでbaseline.pyも学習し、変更した報酬の影響を比較します。
reward_only.pyに置き換えると、独自報酬の例を実行できます。
完成例の再実行は [こちら](../../results/reward_study/README.md) です。

CPUで接続だけを確認する場合：

```bash
MUJOCO_GL=egl .venv/bin/python tools/train_student.py \
  examples/student/reward_only.py --output runs/reward_smoke --smoke
```

この短い実行で押す動作を獲得することや、成功率の比較はできません。

## 観測・行動を研究する

観測・行動を変更する例では、標準側もrotation="none"に揃えます。
新しい入力・行動で学習が進むか、予算も含めて検証します。完成例の成功率をそのまま保証するものではありません。
compare.pyは比較APIの使用例です。共通の前処理を使い、観測か行動の1つだけを変えます。

独自関数はPythonファイルのトップレベルへ書き、TASKを定義します。
load_task_fileで読み込むと、保存時に定義ファイルとハッシュもコピーされます。
詳しくは [APIリファレンス](../../docs/student-api.md) を参照してください。
