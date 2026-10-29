# 学生が編集する課題ファイル

最初は[Colab](../../notebooks/student_api_colab.ipynb)と[学習方法の説明](../../docs/training-method.md)を読みます。

| ファイル | 内容 |
|---|---|
| [reward_study.py](reward_study.py) | 完成例の標準条件・報酬変更条件・共通の追加学習設定 |
| [baseline.py](baseline.py) | 学習済み運搬モデルに対応する標準課題 |
| [reward_weights.py](reward_weights.py) | 進捗報酬の重みだけを20から40へ変更 |
| [reward_only.py](reward_only.py) | 目標付近の円柱速度を減点する独自関数を追加 |
| [pretrained.py](pretrained.py) | 学習済みモデルに対応する課題の別表記 |
| [observations.py](observations.py) / [actions.py](actions.py) | 観測・行動を変更する新規学習の定義 |
| [custom_observation.py](custom_observation.py) | 自作の観測関数 |
| [compare.py](compare.py) | 観測・行動を変える新規学習の比較とCSV保存 |

## 安定したモデルから報酬実験を行う

[READMEのGPU環境](../../README.md#ローカルで実行する)で、リポジトリ直下から実行します。

```bash
MUJOCO_GL=egl python tools/train_student.py examples/student/reward_weights.py \
  --output runs/my_reward_s42 --backend warp --seed 42 \
  --steps 655360 --num-envs 512 --horizon 64 \
  --validation-envs 256 --validate-every 5 \
  --initial-checkpoint checkpoints/cylinder_push.pt \
  --learning-rate 3e-5 --entropy-coeff 0.001 --checkpoint-selection latest \
  --episodes 1024 --evaluation-seed 640000 --video
```

`reward_only.py`へ置き換えると独自関数を使えます。
同じ初期モデル・設定で標準条件も学習して比較します。Colabではこの流れをまとめて実行できます。
完成例と同じ両条件・3 seedの再実行は[こちら](../../results/reward_study/README.md)です。

CPUで読み込みと新規学習の接続だけを確認する場合：

```bash
MUJOCO_GL=egl .venv/bin/python tools/train_student.py \
  examples/student/reward_only.py --output runs/reward_smoke --smoke
```

この短い試験の成功率を運搬性能として使いません。

## 観測・行動の研究

観測・行動を変更する例では`rotation="none"`を使います。
今回の運搬モデルは入力・行動が異なるモデルへそのまま継承できません。
新規学習では`initial_checkpoint`を省略し、標準側も同じ前処理にして、学習方法と予算を改めて検証します。
今回の追加学習で得た高成功率は、この新しい条件の成績ではありません。

独自の関数は`.py`のトップレベルへ書きます。課題ファイルは`TASK`を定義し、`load_task_file()`で読み込みます。
保存時に定義ファイルとハッシュもコピーされます。詳しくは[APIリファレンス](../../docs/student-api.md)を参照してください。
