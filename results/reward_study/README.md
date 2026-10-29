# 安定した運搬モデルからの報酬比較

同じ学習済み運搬モデルを出発点に、標準条件（`goal_progress=20`）と、重みだけを40にした条件を追加学習した完成例です。
両条件・全seedで同じActor・Criticを使い、Adamを毎回新しく作ります。各条件3 seed、各seed 655,360サンプルです。
ランダム初期重みからの独立した学習3回とは区別します。

[考察の完成例](discussion.md)、[学習方法の見直し](../../docs/training-method.md)、[Colab](https://colab.research.google.com/github/hashimoto-robotics-lab/hexapod-transport-rl/blob/main/notebooks/student_api_colab.ipynb)を順に参照してください。

| 条件 | 成功率の平均±標準偏差 | 最終距離の平均±標準偏差 |
|---|---|---|
| 標準20 | 99.41% ± 0.17ポイント | 0.0339 ± 0.0013 m |
| 変更40 | 99.41% ± 0.10ポイント | 0.0339 ± 0.0010 m |

## 固定した条件

- 回転対称化：`goal_octant`。14入力、停止＋世界座標8方向。
- 親モデル：`checkpoints/cylinder_push.pt`。ハッシュは`protocol.json`に記録。
- PPO：学習率`3e-5`、エントロピー係数`0.001`。他の設定も両条件で共通。
- 開始位置・向き・目標の分布、物理、40秒の制限時間、5 cmの精度と停止保持の成功条件は共通。
- 報告方策：規定量の追加学習を終えた`latest.pt`。途中の重みや親モデルへの差し替えは行わない。
- 検証：seed131000の256配置、5更新ごと。
- 最終評価：未使用のseed540000の1,024配置、貪欲な離散行動。全条件・全seedで初期配置を共通にする。
- 動画：事前指定した追加学習seed42・動画seed550000。CPUのMuJoCoで描画する。

数値はGPUのMuJoCo Warp、動画はCPUのMuJoCoです。動画の成否をGPU評価の成功数に加算しません。

## 保存したもの

| ファイル | 内容 |
|---|---|
| `protocol.json` | 仮説、共通の初期モデル、固定条件、学習量、seed、事前に決めた安定性の基準 |
| `runs.csv` / `summary.json` | 全6回の最終評価、追加学習seed間の平均・標本標準偏差、同じseed同士の差 |
| `learning_curve.csv` | 追加学習前と検証時点の成績、報告する最後の時点 |
| `training.csv` | 全更新の収集中の指標、重みを掛けた報酬項の寄与 |
| `evaluations/*.json` | 全評価エピソードの初期配置、成否、最終距離、時間、報酬内訳 |
| `checkpoints/*.pt` | 最後まで更新した6つの方策。ハッシュは`runs.csv` |
| `metadata/` | 元の学習記録、課題、依存バージョン、初期重みと更新後の重みのハッシュ |
| `source.json` | 全6回の学習で一致を確認したライブラリソースのハッシュとGit commit |
| `*.png` / `videos/` | 図、事前指定した2本の動画、動画の結果JSON |

成功率は0〜1、距離はm、時間は秒です。平均と標準偏差は3つの追加学習seedから計算します。
1,024エピソード×3 seedを、独立した3,072回の学習として扱いません。
横軸0は学習済みの親モデルです。`trained_steps`と`selected_steps`は今回の追加学習量で、共通の事前学習コストを含みません。
開発用seed17の確認結果は比較本体に含めていません。

RTX A6000の共用ワークステーションで計測しました。GPUの片方には別の処理の負荷もあります。
時間は今回の追加学習と検証を含み、過去の事前学習・最終評価・動画作成を含みません。
条件間の速度差やColab T4の所要時間を示すものではありません。

## 再実行する

[READMEのCUDA環境](../../README.md#ローカルで実行する)でリポジトリ直下から実行します。
設定は[reward_study.py](../../examples/student/reward_study.py)です。出力先は新しい名前にします。

```bash
mkdir -p runs/reproduce
cp results/reward_study/protocol.json runs/reproduce/protocol.json
MUJOCO_GL=egl python tools/run_reward_study.py \
  --condition baseline --output runs/reproduce/main
MUJOCO_GL=egl python tools/run_reward_study.py \
  --condition progress40 --output runs/reproduce/main
python tools/build_reward_study_report.py runs/reproduce runs/reproduce/report
MUJOCO_GL=egl python tools/render_reward_study.py runs/reproduce/report
```

同じ型のGPUでも、接触計算の数値差で軌跡が完全一致するとは限りません。
全seedの結果を残し、平均だけでなくばらつきを比較します。考察文は数値を確認して自分で書きます。
自分の仮説では、新しい最終評価seedを事前に決めます。Colabの実習は640000を使います。

保存方策を再評価する例：

```python
from hexapod_transport_rl import evaluate, load_task

checkpoint = "results/reward_study/checkpoints/baseline_s42.pt"
task = load_task(checkpoint)
report = evaluate(checkpoint, episodes=1024, seed=540000, backend="warp")
```

以前の約5%の新規学習結果は、Git履歴のcommit `8853477`に残しています。
