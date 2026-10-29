# 実測した報酬1項目の比較

標準条件（`goal_progress=20`）と、その重みだけを40にした条件を、ランダムな初期重みから実際に学習しました。
各条件3つの学習seed、各seed 2,621,440サンプルです。観測・行動・初期状態分布・成功条件・PPOは共通です。
回転対称化や既存の運搬方策からの重み継承は使っていません。歩行モデルは両条件で同じ固定モデルです。

まず[完成例の考察](discussion.md)を読み、[Colab](https://colab.research.google.com/github/hashimoto-robotics-lab/hexapod-transport-rl/blob/main/notebooks/student_api_colab.ipynb)から自分の実験へ進めます。

| 条件 | 成功率の平均±標準偏差 | 最終距離の平均±標準偏差 |
|---|---|---|
| 標準20 | 5.47% ± 2.44ポイント | 0.853 ± 0.007 m |
| 変更40 | 5.34% ± 1.48ポイント | 1.248 ± 0.928 m |

この予算では、両条件とも運搬の成功が不安定でした。seedによって優劣が逆転し、変更による改善は確認できていません。

## 保存したもの

| ファイル | 内容 |
|---|---|
| `protocol.json` | 比較前に固定した仮説・条件・予算・seed・評価方法 |
| `runs.csv` / `summary.json` | 全6学習の最終評価、学習seed間の平均・標本標準偏差、同じseed同士の差 |
| `learning_curve.csv` | 学習前と検証時点の成績・選択基準の各値・選ばれた時点 |
| `training.csv` | 全更新の収集中の指標と、重みを掛けた報酬項ごとの寄与 |
| `evaluations/*.json` | 最終評価の全エピソード。初期配置・成否・最終距離・時間・報酬内訳 |
| `checkpoints/*.pt` | 検証だけで選んだ6つの方策。ハッシュは`runs.csv`に記載 |
| `metadata/` | 元の学習記録、課題設定、依存バージョン、初期重みのハッシュ |
| `source.json` | 全6学習で一致を確認したライブラリソースのハッシュと元のGit commit |
| `*.png` | 課題の模式図、検証曲線、seedごとの比較、動画の抜粋 |
| `videos/*.mp4` / `*.json` | 事前指定した学習seed42・動画seed250000の両条件の動画と結果 |

`success_rate`は0〜1、距離はm、時間は秒です。平均と標準偏差は3つの学習seedから計算します。
同じ256配置で評価した各seedのエピソードを、独立した768個の学習実験として扱いません。
`selected_steps`は選ばれた方策の学習時点、`trained_steps`は実際に使った学習予算です。
初期方策も選択候補なので、`selected_steps=0`の場合もあります。

## 評価とモデル選択

- 検証：seed131000の256配置、10更新ごと。初期方策も含めて`best.pt`を選択。
- 選択順：目標方向の最低成功率、位置と向きの組の最低成功率、向き・位置の最低成功率、全体成功数、最終距離。両条件で共通。
- 最終評価：seed240000の256配置、貪欲な離散行動。結果を使ってモデルを選び直していません。
- 位置・向き・目標の初期配置は、両条件・全学習seedで共通です。
- 数値はGPUのMuJoCo Warp、動画はCPUのMuJoCoです。動画の成否をGPUの成功数に加算しません。

以前の65万サンプルの予算をseed7で予備確認し、両条件の本比較を始める前に4倍の予算を決めました。
この予備学習は3 seedの集計に含めません。

計測はRTX A6000の共用ワークステーションです。GPUの片方には別の処理の負荷もあります。
学習時間は環境準備・検証を含み、最終評価・動画作成を含みません。
条件間の速度差を示す実験でも、Colab T4の所要時間を測ったものでもありません。
依存バージョンは`metadata/*_environment.json`に記録しています。

## 再実行する

[READMEのCUDA環境](../../README.md#ローカルで実行する)で、リポジトリ直下から実行します。
設定は[reward_study.py](../../examples/student/reward_study.py)です。出力先は新しい名前を使います。

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

両条件で6回の新規学習を実行します。異なるGPU・ライブラリ・接触計算では、同じseedでも結果が完全一致するとは限りません。
公開値と異なった場合も、全seedの結果を残して比較します。考察文は数値を確認して自分で書きます。
独自の実験では、未使用の最終評価seedを事前に決めてください。Colabの実習では340000を使います。

保存方策を再評価する場合も、同じ学生APIを使えます。

```python
from hexapod_transport_rl import evaluate, load_task

checkpoint = "results/reward_study/checkpoints/baseline_s42.pt"
task = load_task(checkpoint)
report = evaluate(checkpoint, episodes=256, seed=240000, backend="warp")
```
