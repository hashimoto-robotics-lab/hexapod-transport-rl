# ゼロから学習するPPOの報酬比較

標準条件（goal_progress=20）と、その重みだけを40にした条件を比較します。
押す方向を選ぶActor・Criticは毎回ランダムな重み、Adamは空の状態から開始します。
同じseedの条件間では同じ初期重み、異なるseed間では異なる初期重みです。
18関節の歩行モデルは固定して使います。

[考察の完成例](discussion.md)、[学習方法](../../docs/training-method.md)、[Colab](https://colab.research.google.com/github/hashimoto-robotics-lab/hexapod-transport-rl/blob/main/notebooks/student_api_colab.ipynb)を参照してください。
全seedの数値は [runs.csv](runs.csv)、平均と標本標準偏差は [summary.json](summary.json) です。

| 条件 | 成功率の平均±標本標準偏差 | 終了時距離の平均±標本標準偏差 |
|---|---|---|
| 標準20 | 98.80% ± 0.23ポイント | 0.0327 ± 0.0019 m |
| 変更40 | 91.31% ± 6.41ポイント | 0.0403 ± 0.0014 m |

全3 seedで標準条件を下回り、成功率の平均差は−7.49ポイントでした。
標準条件は事前に決めた安定性の基準をすべて満たしました。変更後の失敗例も含め、[考察](discussion.md)で解釈と限界を説明しています。

## 固定した条件

- 各条件3 seed（42・43・44）、各seed 9,830,400サンプル。
- 14入力、停止＋世界座標8方向、共通のgoal_octant回転対称化。
- 同じカリキュラムで開始配置の範囲を広げ、最後は位置・向き・目標が全方向ランダム。
- 段階が変わっても重みとAdamを保持。既存の運搬モデルは読み込まない。
- 基本学習率3e-4、最終段階は3e-5。エントロピー係数0.01。
- 物理、40秒の制限時間、誤差5 cm未満と速度0.06 m/s未満を0.4秒保持する成功条件は共通。
- 検証は最初から全方向ランダム。seed131000の256配置で10更新ごと。
- 規定量を学習し、最後の全方向ランダムの段階の検証からbest.ptを選ぶ。最終評価は選択に使わない。
- 最終評価はseed840000の1,024配置、貪欲な離散行動。全条件・全seedで初期配置を共通にする。
- 動画は事前に指定した学習seed42・動画seed850000。CPU MuJoCoで描画する。

モデル選択では目標方向別の最低成功率、位置と向き別の最低成功率などを順に比べます。
正確な基準はprotocol.jsonのcheckpoint_selectionに記録しています。
数値はGPU MuJoCo Warp、動画はCPU MuJoCoです。動画の成否をGPU評価の成功数に加算しません。

## 保存したもの

| ファイル | 内容 |
|---|---|
| protocol.json | 仮説、初期化、カリキュラム、PPO、総学習量、seed、事前に決めた判定基準 |
| runs.csv / summary.json | 全6回の最終評価、学習seed間の平均・標本標準偏差、同じseed同士の差 |
| learning_curve.csv | ランダムな初期方策と各検証時点の成績、選択した時点 |
| training.csv | 全更新の収集中の指標、段階と学習率、重みを掛けた報酬項の寄与 |
| evaluations/*.json | 全評価エピソードの初期配置、成否、最終距離、時間、報酬内訳 |
| checkpoints/*.pt | 検証で選んだ6つの方策。ハッシュはruns.csv |
| metadata/ | 元の学習記録、課題、依存バージョン、初期重みと学習後の重みのハッシュ |
| source.json | 全6回で一致を確認したライブラリソースのハッシュとGit commit |
| *.png / videos/ | 図、事前に指定した2本の動画、動画の結果JSON |

成功率は0〜1、距離はm、時間は秒です。平均と標準偏差は3つの独立した学習seedから計算します。
1,024エピソード×3 seedを、独立した3,072回の学習として扱いません。
横軸0はランダムな初期方策です。trained_stepsが総学習量、selected_stepsが検証で選んだ方策の学習時点です。
選択した方策の時点と、最後の更新は一致するとは限りません。
初期Adamの状態数と、選択した方策のAdamの更新回数も記録し、段階ごとに初期化していないことを確認します。
開発用seed21の確認結果は比較本体に含めません。

## 所要時間と再実行

学習時間は [runs.csv](runs.csv) のwall_training_secondsです。
簡単な配置での練習、最終条件での学習、途中の検証をすべて含みます。
インストール、学習後の独立評価、動画作成は含みません。
RTX A6000の共用ワークステーションで測定しており、GPUの片方には別の処理の負荷もあります。
Colab T4の所要時間や、条件間の処理速度の優劣を示す値ではありません。

ノートブックの接続試験は、Gitからの取得、ランダム初期化、カリキュラム、学習・保存・評価・動画、報酬変更、複数seed比較、独自報酬までを短縮設定でCPU・GPU実行しています。
ColabのIPython 7.34.0・ipykernel 6.17.1・NumPy 2.0.2・pandas 2.2.3でも確認しました。
この短い接続試験の成績を、本学習の成功率として使っていません。検証コードは [test_colab_notebook.py](../../tests/test_colab_notebook.py) です。

[READMEのCUDA環境](../../README.md#ローカルで実行する) で、リポジトリ直下から実行します。
設定は [reward_study.py](../../examples/student/reward_study.py) です。出力先は新しい名前にします。

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
自分の仮説では、新しい最終評価seedを事前に決めます。Colabの実習は940000を使います。

保存方策を再評価する例：

```python
from hexapod_transport_rl import evaluate

checkpoint = "results/reward_study/checkpoints/baseline_s42.pt"
report = evaluate(checkpoint, episodes=1024, seed=840000, backend="warp")
```

以前の約5%の新規学習結果はGit履歴の8853477、学習済み運搬モデルからの追加学習の比較は7fcb8c6に残しています。
