# 学生向けColab：報酬設計の比較実験

[ノートブックを開く](https://colab.research.google.com/github/hashimoto-robotics-lab/hexapod-transport-rl/blob/main/notebooks/hexapod_transport_rl_colab.ipynb)。公開リポジトリなのでGitHub認証は不要です。
CPUランタイム、Python 3.12・3.13に対応します。最初の準備以外は通常のライブラリ呼び出しです。
学習もアクティブなカーネルから開始し、独立した物理世界だけを子プロセスで並列実行します。

## 教材で理解すること

1. 研究の動機と、固定した歩行モデル・学習する速度指令方策の関係。
2. `gym.make("HexapodWalking-v0")` での物理速度コマンドとmediapyでの歩行観察。
3. `gym.make("HexapodApproach-v0")` の観測・正規化行動・チーム報酬・終了条件。
4. `info["reward_terms"]` と実際に使う報酬関数の対応。
5. 1つの係数を変更した仮説と、同じ学習条件での比較。
6. 共通のテストseedによる成功・接触・転倒・位置誤差・所要時間と動画の評価。

回り込み報酬は `ApproachRewardWeights`、押す報酬は `PushRewardWeights` を環境の設定へ渡します。
報酬関数の項そのものを変える場合は、`src/hexapod_transport_rl/rewards.py` の
`approach_reward_terms()` を編集します。ソースを編集したら新規ランタイムで新しい実験を始めます。
`PushEnv._reward_terms()` が押す動作の式です。実際に学習へ渡していない別の式をノートブックだけに作らないでください。

## 動作確認から研究へ

最初は256チームステップ／条件・評価2試行／配置・制限10秒です。
これはAPIを接続して一巡する確認で、運搬成功を期待する学習量ではありません。
研究では新しい実験名にして、409,600ステップ／条件・評価50試行／配置・制限100秒を目安にします。
並列世界数はColab CPUの能力に合わせて2から調整します。描画は評価時だけ、物理更新は200 Hzです。
GPUによるシミュレーションの高速化は組み込んでいません。

両条件とも同じ初期化seed・学習量・物理設定・固定歩行モデル・固定押す方策・カリキュラム規則を使います。
カリキュラムは成功率によって進むため、報酬による到達段階の違いも記録します。
同じ学習量の最後のモデルを共通の未使用テストseedで比較し、テスト結果からモデルを選び直しません。
ハイパーパラメータやモデルの選定が必要なら、別の検証seedを使い、最終テストを残してください。
総報酬は式を変えると尺度が変わるため、条件間の性能比較には使いません。
少なくとも3つの学習seedで繰り返し、学習によるばらつきも報告します。

## 保存と再開

`create_experiment(PROJECT_DIR, name)` は新規出力先を作り、実行時のソース・資産・モデル・コミットSHA・版を保存します。
同じ名前を再利用せず、新しい条件には新しい名前を付けます。各学習の `run.json` に報酬係数とseedを記録し、
`metrics.jsonl` に学習速度・到達段階、`checkpoint.pt` に重み・optimizerを保存します。
最後の `archive_results(RUN_DIR)` で結果ZIPを作り、Colabのファイル一覧からダウンロードします。
未保存の結果はランタイムの削除で失われるため、本学習では途中結果もDrive等へコピーしてください。

保存済み実験をColabへ復元した後、通常のAPIで追加学習できます。

```python
continued = train_approach(
    output=RUN_DIR / "baseline_continued",
    pushing_checkpoint=PUSHER,
    resume=RUN_DIR / "baseline/checkpoint.pt",
    iterations=100,
    num_envs=NUM_ENVS,
    horizon=HORIZON,
    seed=TRAINING_SEED,
)
```

`iterations` は追加する更新数です。再開では保存済み報酬設定を引き継ぎます。
設定も指定する場合は、保存済み設定と一致する必要があります。
中断時のエピソード・乱数列の完全復元ではありません。条件比較では両条件の再開方針も揃えてください。

## 押す報酬を研究する場合

回り込みを固定し、両条件とも共通の押すモデルから同じ量を追加学習します。
`train_handover` は引き継ぎ付近のランダム初期配置から、報酬のみで学習します。

```python
from dataclasses import replace
from hexapod_transport_rl import PushRewardWeights
from hexapod_transport_rl.handover_training import train_handover, compose

reward = replace(PushRewardWeights(), body_contact=3.0)
pusher = train_handover(
    checkpoint=PUSHER, output=RUN_DIR / "push_changed",
    reward_weights=reward, iterations=150, num_envs=2, horizon=64,
)
model = compose(models["baseline"], pusher, RUN_DIR / "transport_changed.pt")
```

変更なしの条件も同じ元モデルから同じ更新数で追加学習してください。
回り込みと押す段階の変更を同時に行う前に、それぞれの影響を切り分けます。
元のモデルは上書きしません。報酬のみを変えた押すモデルの結合は可能ですが、物理設定の異なるモデルは結合できません。

## 実験の範囲

このノートブックの回り込み学習は2台、平坦な床、T字物体1つ、他の障害物なし、
位置・姿勢をシミュレータから取得する条件です。脚・足による押し動作を使い、胴体への押し具は追加しません。
歩行APIは1〜4台、基本の押す環境は2〜4台ですが、同梱運搬モデル・回り込みは2台専用です。
4台での役割割当・観測・報酬・中央criticの拡張は別に実装・学習して評価します。
