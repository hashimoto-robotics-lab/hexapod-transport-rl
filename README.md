# Hexapod Transport RL

**六足ロボットの強化学習による協調物資運搬**を学ぶ卒業研究向けプロジェクトです。
2台がT字物体の周囲を回り込み、脚・足の接触で目標へ運搬します。
上位方策は共有actor・中央criticのMAPPO、歩行モデルは固定します。学習は報酬だけを使います。

## Google Colabで始める

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hashimoto-robotics-lab/hexapod-transport-rl/blob/main/notebooks/hexapod_transport_rl_colab.ipynb)

上の「Open in Colab」を開き、Googleアカウントでログインして、上から順に実行してください。
**GitHubアカウント・招待・アクセストークンは不要**です。
コード・ロボット形状・学習済みモデルは、Step 1で公開リポジトリから自動取得します。

まず研究の動機を読み、学習済み歩行モデルへ速度コマンドを送って1台と4台の動きを確かめます。
続いて環境の `reset()`・`step()` と、観測・行動・報酬の内訳を確認します。
**報酬係数を1つ変え、同じ条件で学習し、同じ評価seedで図・動画を比較する**教材です。

```python
from dataclasses import replace
from hexapod_transport_rl import ApproachConfig, ApproachRewardWeights, train_approach

reward = replace(ApproachRewardWeights(), robot_contact=12.0)
config = ApproachConfig(reward_weights=reward, easier_reset_fraction=0.25)
model = train_approach(
    output="runs/contact_trial",
    pushing_checkpoint="checkpoints/pusher.pt",
    config=config, iterations=3200, num_envs=2, horizon=64,
)
```

学習・評価は通常のimportと関数呼び出しで、ColabのPythonから直接実行します。
回り込みの報酬を比較する際は、歩行モデルと押す方策を共通に固定します。デモは教師に使いません。
動画はmediapyで表示します。準備補助はインストールだけ、保存補助は記録とZIP作成だけを担当します。
最初の256チームステップ／条件は動作確認です。本学習の目安は409,600ステップ／条件で、
本評価は50試行・100秒、学習seedを少なくとも3種類で繰り返します。
4台の協調運搬学習は環境拡張・再学習を伴う課題です。
詳しくは [Colabガイド](docs/colab-guide.md)、[APIガイド](docs/api.md)、[コードガイド](docs/code-guide.md) を参照してください。

## ローカルで実行する

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
| `checkpoints/transport.pt` | 選定済みの回り込み方策。隣の `pusher.pt` と併用 |
| `checkpoints/pusher.pt` | 回り込み後の位置へ適応させた押す方策 |
| `checkpoints/base_pusher.pt` | 回り込み・引き継ぎ学習の初期値として使う押す方策 |

保存モデルを未使用の前方100条件・側方50条件で評価した結果は、運搬成功85/100・40/50でした。
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
  walking.py                            学生向け：1〜4台への速度指令
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
