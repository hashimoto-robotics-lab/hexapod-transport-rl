# 六足ロボットで円柱を押す強化学習教材

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hashimoto-robotics-lab/hexapod-transport-rl/blob/main/notebooks/student_api_colab.ipynb)

六足ロボット1台が、円柱を目標位置まで押す動作をTorchRL PPOで学習します。
学生は **報酬・観測・行動** を変更し、共通の物理・成功条件で比較実験できます。
18関節の機体・学習済み歩行モデル・モーターモデルを同梱しています。

## 最初はColabで学ぶ

[ステップごとのデモ](notebooks/student_api_colab.ipynb)をColabで開き、上から実行してください。
標準条件と、報酬1項目だけを変更した実測の完成例を読み、仮説・学習曲線・評価・動画・考察を順に確認します。
その後、自分の仮説で新規学習し、複数seedの比較・グラフ・結果保存まで進めます。

1. [Colabでデモを開く](https://colab.research.google.com/github/hashimoto-robotics-lab/hexapod-transport-rl/blob/main/notebooks/student_api_colab.ipynb)をクリックします。
2. ランタイムを **T4 GPU** に変更します。
3. 最初のセルを実行すると、GitHubから教材をcloneし、実行環境を準備します。

Colabに入っているPyTorchをそのまま使います。

取得先は `https://github.com/hashimoto-robotics-lab/hexapod-transport-rl.git`、既定のブランチは `main` です。
別の公開済みブランチを使う場合は、最初のセルの `REVISION` を変更してください。
再実行時は取得済みの教材を使い、学生の編集を保持します。最新版を取り直す場合は新しいセッションを使います。

完成例は、各条件をランダムな重みから2,621,440サンプルずつ、3つの学習seedで実際に学習したものです。
`goal_progress`の重みを20から40へ変更し、各方策を共通の未使用256配置で評価しています。
結果を読むために再学習する必要はありません。[実測結果と考察](results/reward_study/discussion.md)も参照できます。
今回の平均成功率は標準5.47%、変更5.34%でした。標準条件も学習が不安定なため、その限界と次の実験計画まで教材に記載しています。

自分で動かす部分の初期設定 `PROFILE="demo"` は2秒・32サンプルの短い操作練習です。
`"research"` にすると完成例と同じ40秒・2,621,440サンプル・3つの学習seedに切り替わります。
学習時間・成功率は環境と学習予算に依存します。短いデモの成績を学習性能として使わないでください。

## ローカルで実行する

Python 3.12/3.13を使います。CPUの接続確認は次のコマンドです。

```bash
uv sync --locked
MUJOCO_GL=egl .venv/bin/python tools/train_student.py \
  examples/student/baseline.py --output runs/smoke --smoke
```

ローカルGPUでは、検証済みの次の構成を使えます。
CUDA対応PythonへPyTorch 2.11.0のCUDA 12.8版を導入してから、追加依存を入れます。
`uv`の既定設定はCPU版PyTorchを使うため、GPU環境には下記の`pip`を使ってください。

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[warp,notebook]'
MUJOCO_GL=egl python tools/train_student.py \
  examples/student/baseline.py --output runs/baseline_s42 \
  --backend warp --num-envs 256 --steps 2621440 \
  --validation-envs 256 --validate-every 10
```

## 学生が編集するAPI

```python
from hexapod_transport_rl import CylinderPushTask, DiscreteDirections, train

task = CylinderPushTask(
    observations=("robot_to_cargo", "cargo_to_goal", "robot_heading"),
    actions=DiscreteDirections(count=4),  # 停止＋4方向
    rewards={"goal_progress": 20, "time": -0.02, "success": 10, "failure": -10},
)
run = train(task, "runs/my_experiment", steps=655_360)
report = run.evaluate(episodes=128)
run.video()
```

[APIの説明・項目一覧](docs/student-api.md)と[Pythonサンプル](examples/student/README.md)を参照してください。
独自関数は `.py` ファイルに置き、`load_task_file()`で読み込みます。
保存時にソース・設定・依存バージョンも記録されます。

## 報酬を変えて学習する例

ColabのStep 0〜7は、完成した比較実験を読む教材です。
Step 8〜10では、自分の仮説を書き、重み1つを変えて新規学習・共通配置での評価・複数seedの比較・図の保存を行います。
Step 11では、目標の近くで円柱が速く動くと減点する独自関数を書き、報酬だけを変更して学習します。
各学習の保存先に時刻を付けるので、値を変えてセルを再実行できます。
[実験の進め方と考察の書式](docs/experiment-guide.md)に、結果が予想と異なったときの確認事項もまとめています。

同じ課題をPythonファイルから学習する例もあります。

```bash
# 報酬の重みを変える
.venv/bin/python tools/train_student.py examples/student/reward_weights.py \
  --output runs/reward_weights_demo --smoke

# 独自の報酬関数を追加する
.venv/bin/python tools/train_student.py examples/student/reward_only.py \
  --output runs/custom_reward_demo --smoke
```

本学習では`--smoke`を外し、CUDA対応Pythonで実行します。

## 課題と参照モデル

- 円柱：半径0.25 m、高さ0.25 m、質量2 kg。
- 初期位置：円柱から0.65〜1.00 mの円環。位置・機体の向き・目標方向がランダム。
- 目標：円柱から0.30〜0.60 m。位置誤差5 cm未満・速度0.06 m/s未満を0.4秒保持すると成功。
- 物理200 Hz、固定歩行25 Hz、上位行動5 Hz。標準行動は停止＋世界座標8方向。

`checkpoints/cylinder_push.pt`は以前の実験で選んだ学習済み方策です。
`load_task()`で当時の設定を復元できます。このモデルは45度刻みの回転対称化を使います。
新規の学生実験は対称化なしが既定なので、比較時は前処理を揃えてください。
旧円柱チェックポイントと学生APIのチェックポイントを読み込めます。

## 構成

```text
notebooks/student_api_colab.ipynb  # 最初に開く教材（1冊）
examples/student/                 # 学生が変更する小さな課題ファイル
src/hexapod_transport_rl/         # 学生API、PPO、CPU/GPU物理、機体・歩行資産
checkpoints/                     # 参照方策1つと来歴・ハッシュ
results/reward_study/             # 完成例の実測値・図・動画・評価用方策
docs/                            # APIリファレンス、実験と考察の手引き
tools/                           # 学習CLI、Colab準備、完成例の再実行・集計
tests/                          # 物理・PPO・学生API・教材の検証
pyproject.toml / uv.lock          # 依存関係
```

`runs/`は実行時に作られる出力でGit管理しません。
旧タスク、旧ノートブック、過去のレポート・実験用スクリプトは整理済みです。

開発用チェック：

```bash
uv run ruff check src tests tools
MUJOCO_GL=egl uv run --extra notebook pytest
```
