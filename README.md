# 六足ロボットで円柱を押す強化学習教材

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hashimoto-robotics-lab/hexapod-transport-rl/blob/main/notebooks/student_api_colab.ipynb)

六足ロボット1台で円柱を目標まで押す課題を、TorchRL PPOで学びます。
**押す方向を選ぶActor・Criticはランダムな重みから学習します。**
18関節を制御する歩行モデルは固定して使います。
報酬1項目を変えた完成例を読み、仮説・実験・評価・考察を自分で進める教材です。

標準条件は3 seedで平均98.80%成功しました（各seed 98.54〜98.93%、各1,024配置）。
報酬の進捗項を20から40にした条件は平均91.31%で、今回の設定では改善しませんでした。
本学習はRTX A6000で標準1回あたり平均約38分です。Colab T4の所要時間は未測定です。

## Colabで実行する

1. [Colab教材を開く](https://colab.research.google.com/github/hashimoto-robotics-lab/hexapod-transport-rl/blob/main/notebooks/student_api_colab.ipynb)。
2. ランタイムを **T4 GPU / ランタイムバージョン2026.07** に変更する。
3. 完成例を読み、標準条件をゼロから学習する。

Step 0〜7は実際に学習した2条件×3 seedの完成例です。読むための再学習は不要です。
Step 8〜9の既定は `PROFILE="research"` で、標準条件の本学習・全方向での評価・動画作成を1回実行します。
報酬変更は `RUN_REWARD_EXPERIMENT=True`、複数seedの比較は `RUN_MULTISEED_STUDY=True` で追加実行します。
標準条件を学習した後は、報酬変更のセルだけを実行できます。
Step 11の独自報酬は `RUN_CUSTOM_REWARD=True` で学習できます。各実験に追加の学習時間が必要です。

2026.07は、本学習と同じPython 3.12.13・PyTorch 2.11.0の版を使う設定です。
ランタイムの選択方法と提供期間は[Colab公式の一覧](https://research.google.com/colaboratory/runtime-version-faq.html)にあります。

`PROFILE="demo"` は短いAPIの操作確認です。この実行量では押す動作の獲得や成功率の比較はできません。
最初のセルでGitHubのmainをcloneし、Colabの既存PyTorchを保持して依存を準備します。
更新版はリンクから開き直し、新しいランタイムで実行してください。

## ゼロから押す動作を学ぶ

既存の運搬モデルは読み込みません。1回のtrainの中で、押しやすい配置から開始位置と向きの範囲を広げます。
段階が変わっても重みとAdamを保持します。最後は位置・向き・目標が全方向ランダムな課題です。
両条件で同じカリキュラム・回転対称化・PPO・総学習量を使い、報酬の重みだけを変えます。

- 同じseedの条件間では同じランダム初期重み。異なるseed間では異なる初期重み。
- 検証と最終評価は、最初から全方向ランダムの条件。
- 最後のランダム条件の検証で選んだbest.ptを、別の配置で最終評価。
- 簡単な段階の学習時間も総学習時間に含む。

完成例の設定は [reward_study.py](examples/student/reward_study.py)、実測値と考察は [results/reward_study](results/reward_study/README.md) にあります。
学習量・学習時間・全seedの成績・方向別成功率を保存しています。
GPUと負荷により時間は変わります。RTX A6000の実測時間をColab T4での所要時間として扱いません。

[API](docs/student-api.md)、[実験の進め方](docs/experiment-guide.md)、[学習方法と「ゼロから」の範囲](docs/training-method.md)も参照してください。

## ローカルで実行する

Python 3.12/3.13を使います。CPUの接続確認：

```bash
uv sync --locked
MUJOCO_GL=egl .venv/bin/python tools/train_student.py \
  examples/student/baseline.py --output runs/smoke --smoke
```

GPUでは次の環境を用意できます。uvの既定はCPU版PyTorchなので、GPU環境へはpipで導入します。

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[warp,notebook]'
MUJOCO_GL=egl python tools/run_reward_study.py \
  --condition baseline --seeds 42 --output runs/scratch_study/main
```

## 課題

- 円柱：半径0.25 m、高さ0.25 m、質量2 kg。
- ロボット開始位置：円柱から0.65〜1.00 mの円環、向きは全周でランダム。
- 目標：円柱から0.30〜0.60 m、全方向でランダム。
- 成功：位置誤差5 cm未満、円柱速度0.06 m/s未満を0.4秒保持。制限時間40秒。
- 物理200 Hz、歩行25 Hz、上位行動5 Hz。行動は停止＋世界座標8方向。

## 構成

```text
notebooks/student_api_colab.ipynb  # 最初に開く教材
examples/student/                 # 編集する小さな課題と完成例の設定
src/hexapod_transport_rl/         # API、PPO、物理、機体・固定歩行モデル
results/reward_study/             # 実測値、図、動画、学習後の方策
docs/                            # API、学習方法、実験と考察の手引き
tools/                           # Colab準備、学習・集計・動画作成
tests/                           # API、物理、PPO、教材、結果の検証
```

runsは実行時の出力でGit管理しません。

```bash
uv run ruff check src tests tools examples/student
MUJOCO_GL=egl uv run --extra notebook pytest
```
