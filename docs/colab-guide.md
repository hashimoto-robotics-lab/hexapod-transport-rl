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

## TorchRLのMAPPO

[TorchRLの公式MARLチュートリアル](https://docs.pytorch.org/rl/stable/tutorials/multiagent_ppo.html)と同じく、
局所観測の共有actorと、チーム全体の観測を使う中央criticを組み合わせます。
更新にはTorchRL 0.14.0の `MAPPOLoss` と `MultiAgentGAE`、収集には `Collector` を使います。
行動分布は `TanhNormal`、機体別の確率比をclipします。

ノートブックには収集・更新ループを直接書いています。独自のPPO損失やGAEの実装は使いません。
`make_mappo_networks()` と `make_mappo_loss()` はTorchRLのネットワークと損失の設定をまとめます。
設定は `MAPPOSettings`、勾配更新はPyTorchのAdamです。
`ReplayBuffer` は現在のロールアウトをミニバッチに分けるためのものです。
毎回 `buffer.empty()` を呼び、古い方策の経験を次のロールアウトへ持ち越しません。
機体数を時間や並列世界と一緒にflattenせず、各機の行動・確率比を維持します。

`TorchRLTransportEnv` は既存のMuJoCo環境を接続するだけです。
報酬はチームで1つ、観測・行動は機体別の配列です。最終観測と終了フラグを保持し、
時間切れでは最終観測の価値を使い、成功・転倒では使いません。
Collectorが終了した世界だけresetし、GAEがエピソード境界を越えないようにします。

## 動作確認から研究へ

最初は256チームステップ／条件・評価2試行／配置・制限10秒です。
これはAPIを接続して一巡する確認で、運搬成功を期待する学習量ではありません。
研究では新しい実験名にして、409,600ステップ／条件・評価50試行／配置・制限100秒を目安にします。
学習量は `NUM_ENVS * HORIZON` の倍数で指定します。端数は収集単位へ切り上がります。
並列世界数はColab CPUの能力に合わせて2から調整します。描画は評価時だけ、物理更新は200 Hzです。
GPUによるMuJoCoシミュレーションの高速化は組み込んでいません。

両条件とも同じ初期化seed・学習量・物理設定・固定歩行モデル・固定押す方策・カリキュラム規則を使います。
配置は `near → rear → side → front` の順です。50ロールアウトごとに更新後の方策を別seedで検証し、
検証成功率75%以上かつ各段階50ロールアウト以上で難度を上げます。
初期配置の変更は後続のresetに適用し、進行中のエピソードは維持します。検証経験は学習へ渡しません。
カリキュラムは成功率によって進むため、報酬による到達段階の違いも記録します。
同じ学習量の最後のモデルを共通の未使用テストseedで比較し、テスト結果からモデルを選び直しません。
ハイパーパラメータやモデルの選定が必要なら、別の検証seedを使い、最終テストを残してください。
総報酬は式を変えると尺度が変わるため、条件間の性能比較には使いません。
少なくとも3つの学習seedで繰り返し、学習によるばらつきも報告します。

## 保存と再開

`create_experiment(PROJECT_DIR, name)` は新規出力先を作り、実行時のソース・資産・モデル・コミットSHA・版を保存します。
同じ名前を再利用せず、新しい条件には新しい名前を付けます。
各学習の `run.json` に報酬係数・seed・MAPPO設定・ライブラリの版・カリキュラムの規則を記録します。
`progress.csv` に収集量・学習速度・TorchRLの損失・到達段階、`curriculum.json` に進捗を保存します。
ノートブックは25ロールアウトごとに `checkpoints/step_*.pt`、最後に `transport.pt` を保存します。

`.pt` にはactor・critic・optimizer・設定・カリキュラム・PyTorchの乱数状態と、固定押すモデルへの相対パス・ハッシュが入ります。
結果フォルダ全体を移動しても再生できます。`evaluate_transport()` は新形式と同梱の旧形式を自動判別します。
新しいネットワークへ旧形式の重みを直接読み込むことはしません。
最後の `archive_results(RUN_DIR)` で結果ZIPを作り、Colabのファイル一覧からダウンロードします。
未保存の結果はランタイムの削除で失われるため、本学習では途中結果もDrive等へコピーしてください。

再開時は同じ報酬・MAPPO設定を復元し、新しい出力先を使います。

```python
import torch
from hexapod_transport_rl import (
    load_mappo, ApproachConfig, TorchRLTransportEnv, MAPPOSettings,
    make_mappo_loss, ApproachCurriculum,
)

actor, critic, saved = load_mappo("runs/reward_trial_01/baseline/transport.pt")
config = ApproachConfig(**saved["approach_config"])
settings = MAPPOSettings(**saved["training"]["settings"])
loss = make_mappo_loss(actor, critic, settings)
optimizer = torch.optim.Adam(loss.parameters(), lr=settings.learning_rate)
optimizer.load_state_dict(saved["optimizer"])
envs = TorchRLTransportEnv(config, num_envs=saved["training"]["num_envs"])
envs.set_seed(saved["training"]["seed"])
curriculum = ApproachCurriculum(
    config, envs, actor, "runs/reward_continued",
    seed=saved["training"]["seed"], settings=settings,
    horizon=saved["training"]["horizon"], resume=saved,
)
torch.set_rng_state(saved["torch_rng_state"])
```

この準備の後にノートブックと同じCollector・buffer・更新ループを使います。
Collectorの `total_frames` は追加で学習するチームステップ数です。
カリキュラムの段階・重み・optimizerは引き継ぎますが、中断時の物理状態やエピソードは復元しません。
同じseedから現在の段階で新しくresetするため、連続実行と完全に同じ軌跡にはなりません。
比較条件ごとに再開の方針を揃え、追加学習量を記録してください。
最後に `curriculum.close()` と `collector.shutdown()` を呼びます。

## 押す報酬を研究する場合

`TorchRLTransportEnv(PushConfig(...))` も同じCollector・MAPPO更新に接続できます。

```python
from dataclasses import replace
from hexapod_transport_rl import PushConfig, PushRewardWeights

reward = replace(PushRewardWeights(), body_contact=3.0)
envs = TorchRLTransportEnv(PushConfig(shape="T", reward_weights=reward), num_envs=2)
actor, critic = make_mappo_networks(num_robots=envs.num_robots, obs_dim=envs.obs_dim)
loss = make_mappo_loss(actor, critic, settings)
```

これは押す学習の準備部分です。以降の収集・更新ループは回り込みと同じです。
比較する両条件とも同じseedでランダム初期化し、同じ学習量で比較します。
`save_mappo()` / `load_mappo()` は現在、2台の回り込みモデルと同梱の固定押すモデルの結合を扱います。
新しい押すモデルを保存する場合は `torch.save()` でactor・critic・optimizer・`PushConfig` を記録し、
運搬全体へ接続する際には押すモデルの評価アダプターを拡張します。
回り込みと押す段階の変更を同時に行う前に、それぞれの影響を切り分けます。

## 実験の範囲

このノートブックの回り込み学習は2台、平坦な床、T字物体1つ、他の障害物なし、
位置・姿勢をシミュレータから取得する条件です。脚・足による押し動作を使い、胴体への押し具は追加しません。
歩行APIは1〜4台、基本の押す環境は2〜4台で、TorchRLへの接続も2〜4台に対応します。
同梱運搬モデル・回り込みの配置と観測は2台専用です。
4台での役割割当・回り込み環境は別に実装し、学習・運搬成功率を評価します。
