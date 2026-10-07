# Hexapod Transport RL

**六足ロボットの強化学習による協調物資運搬**を学ぶ卒業研究向けプロジェクトです。
最初は2台がT字物体の後方で整列し、脚・足の接触で目標へ運搬する短時間課題を扱います。
前方から回り込む課題と4台への拡張は、その後の研究課題です。
学生の学習にはTorchRLのMAPPOを使い、actorを各機で共有・criticはチーム全体の観測を入力します。
歩行モデルは固定し、報酬だけから学習します。

## Google Colabで始める

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hashimoto-robotics-lab/hexapod-transport-rl/blob/main/notebooks/hexapod_transport_rl_colab.ipynb)

上の「Open in Colab」を開き、Googleアカウントでログインして、上から順に実行してください。
**GitHubアカウント・招待・アクセストークンは不要**です。
コード・ロボット形状・学習済みモデルは、Step 1で公開リポジトリから自動取得します。

まず研究の動機を読み、学習済み歩行モデルへ速度コマンドを送って1台と4台の動きを確かめます。
歩行・回り込み・押す環境は `gym.make()` で作り、共通の `reset()`・`step()`・`render()` で操作します。
観測・行動の形はspaces、報酬の内訳は `info["reward_terms"]` で確認します。
**報酬係数を1つ変え、同じ条件で学習し、同じ評価seedで図・動画を比較する**教材です。

学習では次の部品を組み合わせます。

```python
from hexapod_transport_rl import (
    ApproachConfig, TorchRLTransportEnv, MAPPOSettings,
    make_mappo_networks, make_mappo_loss,
)

envs = TorchRLTransportEnv(ApproachConfig(layout="near"), num_envs=2)
actor, critic = make_mappo_networks()
loss = make_mappo_loss(actor, critic, MAPPOSettings())  # TorchRLのMAPPOLoss
```

これは学習の準備部分です。ノートブックではTorchRLの `Collector` で経験を集め、
`loss.value_estimator()` でGAE、`loss(minibatch)` でMAPPO損失を計算し、PyTorchのoptimizerで更新します。
機体別の確率比をclipし、`TanhNormal` で行動を−1〜1に収めます。初期配置だけを近い配置→後方のカリキュラムで調整します。
収集した経験はロールアウトごとに置き換え、過去の経験を混ぜません。
回り込みの報酬を比較する際は、歩行モデルと押す方策を共通に固定します。デモは教師に使いません。
動画はmediapyで表示します。準備補助はインストールだけ、保存補助は記録とZIP作成だけを担当します。
標準は32,768チームステップ／条件です。2コアのCPU実測では約3分30秒〜4分／条件でした。
進捗報酬あり／なしを同じ初期重みと学習量で比較します。標準評価は6試行／配置・60秒、動画は30秒です。
本評価は50試行／配置・学習seedを少なくとも3種類で繰り返します。256ステップは接続確認だけに使います。
実測の設定と評価結果は [短時間課題の実験記録](docs/training-time.md) にあります。
4台の協調運搬学習は環境拡張・再学習を伴う課題です。
詳しくは [Colabガイド](docs/colab-guide.md)、[APIガイド](docs/api.md)、[コードガイド](docs/code-guide.md) を参照してください。

## 以前のMAPPOモデルをローカルで再生・再学習する

Python 3.12とuvを使用します。ロボットと歩行モデルはパッケージ内に同梱しており、他プロジェクトを参照しません。

```bash
git clone https://github.com/hashimoto-robotics-lab/hexapod-transport-rl.git
cd hexapod-transport-rl
uv sync --locked
./run.sh --help

# 保存済み方策で運搬全体を評価
./run.sh eval --checkpoint checkpoints/transport.pt \
  --episodes 1 --seed 70000 --output runs/replay.json

# 回り込みの学習
./run.sh train --pushing-checkpoint checkpoints/base_pusher.pt \
  --num-envs 4 --horizon 64 --iterations 1600 --output runs/navigation

# 押す方策を、回り込み完了位置へ適応させる
./run.sh train-handover --checkpoint checkpoints/base_pusher.pt \
  --num-envs 4 --horizon 64 --iterations 600 --output runs/handover

# 2つの方策を結合する
./run.sh compose --navigation-checkpoint runs/navigation/checkpoint.pt \
  --pushing-checkpoint runs/handover/checkpoint.pt --output runs/transport.pt
```

インストール後のコマンド名は `hexapod-transport`、Pythonパッケージ名は `hexapod_transport_rl` です。
旧形式の保存モデルも読み込めます。録画用の描画環境は [Colabガイド](docs/colab-guide.md) を参照してください。

## 保存モデルと評価済みの結果

| ファイル | 用途 |
|---|---|
| `checkpoints/lesson_transport.pt` | TorchRLで32,768ステップ学習した短時間課題の参考モデル |
| `checkpoints/transport.pt` | 選定済みの回り込み方策。隣の `pusher.pt` と併用 |
| `checkpoints/pusher.pt` | 回り込み後の位置へ適応させた押す方策 |
| `checkpoints/base_pusher.pt` | 回り込み・引き継ぎ学習の初期値として使う押す方策 |

`lesson_transport.pt` は今回のTorchRL学習の最終モデルです。後方配置20試行で運搬成功16/20、近い配置20試行で18/20でした。
この2台の短時間課題に限る結果です。押す区間の方策は共通の固定モデルを使います。

`transport.pt`、`pusher.pt`、`base_pusher.pt` は以前の独自MAPPO実装で学習したものです。新しいTorchRLの回り込み学習はランダム初期化し、
旧モデルは長い回り込み課題の参考再生と共通の押す方策に使います。新しい保存形式とは互換ではありません。
機体別の確率比・tanh変換という構成は共通ですが、初期化や細部の設定が違うため同じ結果になる保証はありません。

以前の `transport.pt` を未使用の前方100条件・側方50条件で評価した結果は、運搬成功85/100・40/50でした。
回り込み・整列は150/150、胴体とTの接触・転倒・回り込み中の機体同士の接触はゼロでした。
押す区間では機体同士の接触が前方86/100・側方47/50で発生しました。
これらは以前の実験結果であり、Colabで新しく学習する方策の成功率を保証するものではありません。

基本の押す環境は2〜4台を扱えます。今回の回り込み環境と保存モデルは2台専用です。
機体には具体的な18関節の六足モデルを使っています。別の機体に交換するときは関節・歩行モデルの接続も変更する必要があります。
資産の出典とSHA-256は `src/hexapod_transport_rl/assets/manifest.json`、運搬モデルの出典は `checkpoints/manifest.json` に記録しています。

## 構成と検証

```text
notebooks/hexapod_transport_rl_colab.ipynb  学生向けの入口
src/hexapod_transport_rl/                 学習・環境・評価
  torchrl_env.py                        MuJoCoのGymnasium環境をTorchRLへ接続
  torchrl_mappo.py                      TorchRLのネットワーク・損失設定・モデル保存
  torchrl_training.py                   初期配置のカリキュラム・学習ログ
  walking_env.py                        Gymnasium：1〜4台への速度指令
  walking.py                            共通の歩行シミュレーション
  locomotion.py                         歩行・運搬で共通の固定歩行モデル
  assets/robot/                          XMLとメッシュ
  assets/locomotion/                     固定歩行モデル
checkpoints/                             学習済みの運搬モデル
docs/                                    教材とAPIの説明
tests/                                   物理・学習・モデル再生の確認
tools/build_colab.py                      教員向け：ノートブックの再生成
tools/colab_runtime.py                    Colabのインストールと描画設定のみ
runs/                                    実験ログ・動画（配布には不要）
```

```bash
uv run ruff check .
uv run ruff format --check src tests tools
MUJOCO_GL=egl uv run pytest -q
uv run python tools/build_colab.py
```

実行手順を変更した場合は最後のコマンドでノートブックを再生成してください。
ソース・資産・モデルの変更はリポジトリにcommitして共有します。取得時のコミットSHAは実験結果へ自動で記録されます。
`runs/` と仮想環境はGitで共有しません。
