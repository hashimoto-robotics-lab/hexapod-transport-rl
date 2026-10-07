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

学生の経験収集・GAE・PPO更新はStable-Baselines3が担当します。
両条件とも同じ初期化seed・学習量・物理設定・固定歩行モデル・固定押す方策・カリキュラム規則を使います。
カリキュラムは成功率によって進むため、報酬による到達段階の違いも記録します。
同じ学習量の最後のモデルを共通の未使用テストseedで比較し、テスト結果からモデルを選び直しません。
ハイパーパラメータやモデルの選定が必要なら、別の検証seedを使い、最終テストを残してください。
総報酬は式を変えると尺度が変わるため、条件間の性能比較には使いません。
少なくとも3つの学習seedで繰り返し、学習によるばらつきも報告します。

## 保存と再開

`create_experiment(PROJECT_DIR, name)` は新規出力先を作り、実行時のソース・資産・モデル・コミットSHA・版を保存します。
同じ名前を再利用せず、新しい条件には新しい名前を付けます。各学習の `run.json` に報酬係数・seed・PPO設定・ライブラリの版を記録し、
SB3の `progress.csv` に学習速度・損失、callbackが到達段階を記録します。
`navigator.zip` がSB3の重み・optimizer、`transport.json` が共通の押す方策と結合する設定です。
`curriculum.json` に到達段階と検証結果を保存します。SB3の標準 `CheckpointCallback` で
25ロールアウトごとに中間モデルを `checkpoints/` へ保存します。
最後の `archive_results(RUN_DIR)` で結果ZIPを作り、Colabのファイル一覧からダウンロードします。
未保存の結果はランタイムの削除で失われるため、本学習では途中結果もDrive等へコピーしてください。

保存したSB3モデルの再開は標準APIを使います。

```python
from stable_baselines3 import PPO

model = PPO.load(RUN_DIR / "baseline/navigator.zip", env=envs, device="cpu")
model.learn(total_timesteps=12800, reset_num_timesteps=False)
model.save("navigation_continued")
envs.close()
```

`envs` は元と同じ報酬・物理設定で作ってください。追加学習はSB3のoptimizerも引き継ぎますが、
中断時の世界・乱数列の完全復元ではありません。この短い例は環境の初期配置を固定した追加学習です。
新しい `ApproachCurriculum` を指定した場合は近い配置からカリキュラムを始め直します。
条件間で再開方針を揃え、保存モデルと使用した初期配置・報酬を別実験として記録してください。

## 押す報酬を研究する場合

回り込みを固定して押す報酬だけを調べる場合、同じseedから新しいSB3の押すactorを学習します。

```python
import gymnasium as gym
from dataclasses import replace
from stable_baselines3 import PPO
from hexapod_transport_rl import PushConfig, PushRewardWeights, SharedTeamPolicy

reward = replace(PushRewardWeights(), body_contact=3.0)
env = gym.make("HexapodPush-v0", config=PushConfig(shape="T", reward_weights=reward), flatten=True)
model = PPO(SharedTeamPolicy, env, n_steps=64, batch_size=64, device="cpu", seed=42)
model.learn(total_timesteps=256)
model.save("push_changed")
env.close()
```

この短い例は押す環境とSB3の接続確認です。旧MAPPOの `pusher.pt` をSB3へ直接読み込む追加学習ではありません。
上記で作る新しい押す `.zip` と同梱の押す `.pt` は別形式です。
現在の `bind_transport()` はSB3の回り込み `.zip` と固定した同梱の押す `.pt` の評価を扱います。
新しい押す `.zip` の運搬全体への組み込みは別途評価アダプターを拡張する課題として扱います。
回り込みと押す段階の変更を同時に行う前に、それぞれの影響を切り分けます。

## 実験の範囲

このノートブックの回り込み学習は2台、平坦な床、T字物体1つ、他の障害物なし、
位置・姿勢をシミュレータから取得する条件です。脚・足による押し動作を使い、胴体への押し具は追加しません。
歩行APIは1〜4台、基本の押す環境は2〜4台ですが、同梱運搬モデル・回り込みは2台専用です。
4台での役割割当・観測・報酬・中央criticの拡張は別に実装・学習して評価します。
