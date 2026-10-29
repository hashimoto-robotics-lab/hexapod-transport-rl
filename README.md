# 六足ロボットで円柱を押す強化学習教材

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hashimoto-robotics-lab/hexapod-transport-rl/blob/main/notebooks/student_api_colab.ipynb)

六足ロボット1台が円柱を目標まで押す課題を、TorchRL PPOで学びます。
18関節の機体、固定した歩行モデル、モーターモデルを同梱しています。
教材では、安定した運搬モデルを出発点に、報酬1項目を変更した**追加学習**を比較します。

## Colabで学ぶ

1. [Colab教材を開く](https://colab.research.google.com/github/hashimoto-robotics-lab/hexapod-transport-rl/blob/main/notebooks/student_api_colab.ipynb)。
2. ランタイムを **T4 GPU** に変更する。
3. 上から実行して完成例を読み、自分の仮説で比較実験を行う。

Step 0〜7は、仮説・固定条件・学習曲線・最終評価・動画・考察の完成例です。読むための再学習は不要です。
Step 8〜10で、自分の仮説から報酬を変更し、学習・複数seed比較・グラフ作成を行います。
Step 11は独自報酬関数、Step 12は観測・行動を扱う発展課題、Step 13で考察と実験結果を保存します。

`PROFILE="demo"`は2秒・32サンプルの操作練習です。この成績で運搬性能を判断しません。
`"research"`は40秒の課題、各条件655,360サンプルの追加学習、3つの追加学習seed、1,024配置の最終評価です。
完成例の[結果と考察](results/reward_study/discussion.md)、[学習方法の見直し](docs/training-method.md)、[実験の進め方](docs/experiment-guide.md)も参照できます。

最初のセルはGitHubの`main`をcloneし、Colabの既存PyTorchを保持して依存を準備します。
再実行時は取得済みの教材を使って学生の編集を保持します。更新版はリンクから開き直し、新しいランタイムで実行してください。


実測の平均成功率は標準99.41%、報酬変更後99.41%でした。標準条件の各seedは99.22〜99.51%です。
今回の1項目変更による一貫した改善は確認できませんでしたが、両条件とも高成功率を維持しました。

## 安定した運搬を出発点にする

以前の教材は、全方向の初期状態をランダムな重みから学習し、成功率が約5%に留まりました。
現在の報酬実験は、段階的な学習で得た共通の運搬モデルから開始し、学習率と探索を抑えて追加学習します。

- 両条件・全seedでActorとCriticの初期重みが同じ。Adamは毎回新規。
- 共通の回転対称化で目標方向を45度刻みで揃え、行動を世界座標へ戻す。
- 学習率`3e-5`、エントロピー係数`0.001`。
- 比較には規定量を学習した**最後の方策**を使用。初期方策や途中の最良方策へ差し替えない。
- ロボット位置・向き・目標位置の分布、物理、成功条件は従来と共通。

これは共通モデルからの追加学習の比較です。独立したランダム初期重みからの学習3回とは区別します。
過去の事前学習コストも、今回の追加学習時間には含めません。

```python
from hexapod_transport_rl import CylinderPushTask, PPOSettings, train

task = CylinderPushTask.baseline()
run = train(
    task, "runs/my_reward_experiment", seed=42,
    steps=655_360, num_envs=512, horizon=64,
    validation_envs=256, validate_every=5,
    initial_checkpoint="checkpoints/cylinder_push.pt",
    checkpoint_selection="latest",
    settings=PPOSettings(learning_rate=3e-5, entropy_coeff=0.001),
)
report = run.evaluate(episodes=1024, seed=640000)
run.video()
```

`CylinderPushTask()`と`train()`の既定は新規学習です。上の設定を明示した場合に追加学習になります。
入力・行動を変える研究では、そのモデルへ重みをそのまま継承できません。両条件の前処理を揃え、新規学習の方法も別途検証します。
[APIリファレンス](docs/student-api.md)と[小さな課題ファイル](examples/student/README.md)に説明があります。

## ローカルで実行する

Python 3.12/3.13を使います。CPUの接続確認：

```bash
uv sync --locked
MUJOCO_GL=egl .venv/bin/python tools/train_student.py \
  examples/student/baseline.py --output runs/smoke --smoke
```

GPUでは、検証に使った次の環境を用意できます。`uv`の既定はCPU版PyTorchなので、GPU環境へは`pip`で導入します。

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[warp,notebook]'
MUJOCO_GL=egl python tools/run_reward_study.py \
  --condition baseline --seeds 42 --output runs/stable_study/main
```

[完成例の再実行方法](results/reward_study/README.md)には、両条件・全seedの学習、集計、動画の作成方法があります。
所要時間はGPUと負荷に依存します。RTX A6000の実測時間をColab T4の所要時間として扱いません。

## 課題

- 円柱：半径0.25 m、高さ0.25 m、質量2 kg。
- ロボット開始位置：円柱から0.65〜1.00 mの円環、向きは全周でランダム。
- 目標：円柱から0.30〜0.60 m、全方向でランダム。
- 成功：位置誤差5 cm未満、円柱速度0.06 m/s未満を0.4秒保持。制限時間40秒。
- 物理200 Hz、歩行25 Hz、上位行動5 Hz。行動は停止＋世界座標8方向。

## 構成

```text
notebooks/student_api_colab.ipynb  # 学生が最初に開く教材
examples/student/                 # 編集する小さな課題と完成例の設定
src/hexapod_transport_rl/         # API、PPO、物理、機体・歩行資産
checkpoints/                     # 共通の事前学習済み運搬モデルと来歴
results/reward_study/             # 実測値、図、動画、更新後の方策
docs/                            # API、学習方法、実験と考察の手引き
tools/                           # Colab準備、学習・集計・動画作成
tests/                           # API、物理、PPO、教材、結果の検証
```

`runs/`は実行時の出力でGit管理しません。

```bash
uv run ruff check src tests tools examples/student
MUJOCO_GL=egl uv run --extra notebook pytest
```
