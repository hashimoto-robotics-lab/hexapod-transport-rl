# コードを読む順番

歩行コマンドの使い方は [APIガイド](api.md) を参照してください。`walking.py` が学生向けの操作、
`locomotion.py` が歩行と運搬で共通の固定モデル・モーター制御です。

運搬の学習では、まず環境と報酬の2ファイルを読み、次に学習・評価を追います。

| 順番 | ファイル | 読む内容 |
|---|---|---|
| 1 | `src/hexapod_transport_rl/approach.py` | 回り込みの観測・行動、初期配置、整列条件 |
| 2 | `src/hexapod_transport_rl/rewards.py` | 報酬係数と、実際に学習で使う回り込み報酬の式 |
| 3 | `src/hexapod_transport_rl/env.py` | 押す動作の観測・報酬・成功判定と物理シミュレーション |
| 4 | `src/hexapod_transport_rl/mappo.py` | 共有actor・中央critic、経験収集、GAE、PPO更新 |
| 5 | `src/hexapod_transport_rl/approach_training.py` | 回り込みのカリキュラム、検証、保存・再開 |
| 6 | `src/hexapod_transport_rl/approach_evaluation.py` | 方策の切り替え、運搬全体の評価と録画 |

## 研究で変更する場所

| 変更したい内容 | 最初に見る定義 |
|---|---|
| 回り込みの報酬 | `rewards.py` の `ApproachRewardWeights` と `approach_reward_terms()` |
| 押す動作の報酬 | `rewards.py` の `PushRewardWeights` と `env.py` の `_reward_terms()` |
| 初期配置 | `approach.py` の `ApproachConfig` と `_sample_start_pose()` |
| 回り込みから押す動作へ移る条件 | `approach.py` の `HANDOVER_*` と `approach_ready()` |
| 学習率・PPOのclip幅・各損失の係数 | `mappo.py` の `PPOSettings` |
| 初期配置の難度を上げる条件 | `approach_training.py` の `ADVANCE_SUCCESS_RATE` と `MIN_PHASE_ITERATIONS` |

既定値は従来の成功した学習と同じです。係数は各環境の設定に渡し、保存モデルにも記録します。
関数の式を編集した場合は新しいランタイム・新しい実験名で学習します。
`PPOSettings` は回り込みと押す学習で共通です。保存モデルの層数・重み名・観測順は従来の形式を保っています。

## 学生の比較実験

学生用ノートブックは、環境操作・報酬設定・`train_approach()`・`evaluate_transport()` を
通常のPythonコードで直接呼び出します。報酬は `ApproachConfig(reward_weights=...)` へ渡します。
`ApproachEnv.step()` は `rewards.py` の関数を使い、返した報酬と同じ内訳を `info["reward_terms"]` に格納します。

`tools/colab_runtime.py` はアクティブなカーネルへのインストールと描画設定だけを担当します。
`experiments.py` の2関数は実験ソース・資産・版の保存とZIP作成だけを担当します。
学習・集計・図・mediapy動画表示はノートブックのセルに残しています。
手順や説明を変えるときは `tools/build_colab.py` を編集し、
`uv run python tools/build_colab.py` で再生成してください。

## 実行の流れ

```text
cli.py                    コマンドの入口（./run.sh）
  ├─ train                approach_training.py → approach.py
  ├─ train-handover       handover_training.py → api.py → env.py
  ├─ train-push           mappo.py → vector.py → api.py → env.py
  ├─ compose             handover_training.py：2つの保存モデルを結合
  └─ eval                approach_evaluation.py → env.py

各学習 → mappo.py：経験収集、GAE、PPO更新
各環境 → env.py：固定歩行モデル → MuJoCo → 接触計測 → 観測・報酬
```

`handover_training.py` は押す方策を追加学習する際の初期配置を定義します。
75%は回り込み完了位置付近、25%は元の押す学習と同じ配置です。
回り込み動作を毎回シミュレーションせず、引き継ぎ位置からの経験を集めます。

残りは共通の基盤です。

| ファイル | 役割 |
|---|---|
| `config.py` | 物理設定、座標変換、歩行指令の範囲 |
| `walking.py` | 1〜4台への速度指令、同時実行、配置、録画 |
| `locomotion.py` | 固定歩行モデルの検証と推論、25 Hz／200 Hzの共通制御 |
| `model.py` | 元のロボットXMLを複製し、歩行用の床、または運搬用のT・目標を配置 |
| `motor.py` | 固定歩行モデルに合わせた電圧・トルク計算 |
| `contacts.py` | 胴体・脚リンク・足の接触と力積を集計 |
| `api.py` | Gymnasiumの `reset` / `step` / `render` インターフェース |
| `vector.py` | 押す学習用の同期・別プロセスによる複数世界の実行 |
| `types.py`、`__init__.py` | 共通型と公開API |

## 学習アルゴリズム

回り込みと押す動作は、それぞれ独立したMAPPOモデルです。各モデル内では2台がactorを共有し、
中央criticが同じ世界の2台の観測をまとめて入力します。チーム報酬とadvantageも共有します。
actorとcriticは128ユニット×2層のMLP、行動は正規分布からサンプリングしてtanhで範囲を制限します。

`collect_rollout()` が経験を集め、`compute_gae()` がadvantageを計算し、`update_policy()` がPPO更新をします。
時間切れ時はreset前の最終観測から価値を計算します。GAEの伝播は終了・時間切れの両方で切ります。

`train_approach()` 内のコメント1〜5が、初期化、環境準備、収集・更新、検証・難度変更、保存に対応します。
重みとoptimizerの準備は `_initialize_navigator()`、検証用の1エピソードは `validate()` が担当します。
評価側は `run_batch()` が録画リソースを扱い、`_run_episode()` が物理更新と結果集計を担当します。
録画時は物理更新後にコールバックで画像を取得します。

回り込みは `near → rear → side → front` の順で初期配置を難しくします。
報酬にはTを避けて後方へ向かう距離の減少を使います。この距離は**報酬の計算用**です。
動作の教師値や決定的な経路指令には使いません。

回り込みの観測は左右の役割に応じて座標を反転します。選定済みの押すactorは左右対称化したMLPです。
いずれもネットワークの入力・構造の工夫であり、行動そのものは報酬から学習しています。

評価時は `LearnedTransport` が学習済み回り込みactorを実行し、両機の位置と向きが整列条件を
2ステップ連続で満たすと、押すactorへ一度だけ切り替えます。速度指令を作るルール制御はありません。

基礎の押す環境は2〜4台を扱えますが、今回の回り込み環境・保存モデルは2台専用です。

## 初期の押す方策から研究したい場合

初期の押す方策から研究する場合は `checkpoints/base_pusher.pt` を使います。
学生の回り込み報酬の比較では、適応済み `pusher.pt` を両条件で固定します。
初期の押す方策も新たに学習するときだけ `train-push` を使います。

```bash
# ランダムな上位方策から、後方にいる2台でTを押す
./run.sh train-push \
  --num-envs 8 --horizon 64 --iterations 1000 --output runs/push_initial

# 左右対称化と探索幅の調整を適用して追加学習
./run.sh train-push \
  --resume runs/push_initial/checkpoint.pt \
  --num-envs 8 --horizon 128 --epochs 8 --minibatch 256 \
  --mirror-equivariant --initial-log-std -1.5 --seed 20261007 \
  --iterations 500 --output runs/push_refined
```

新しい押すモデルを使う場合は、READMEの回り込み学習・引き継ぎ適応の初期checkpointを置き換えます。
このコマンド例の反復数で同じ成功率が得られる保証はありません。
保存済みの初期モデルは、以前の段階的な追加学習と検証による選定の結果です。
出典は `checkpoints/manifest.json`、元の設定と学習ログは `runs/` に保存しています。

`PushConfig` のフィールドや観測順、モデル形式を変更すると保存モデルとの互換性に影響します。
学生が報酬や初期配置を変える実験では、元のcheckpointを上書きせず新しい `runs/` の出力先を使ってください。
