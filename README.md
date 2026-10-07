# Hexapod Transport RL

**六足ロボットの強化学習による協調物資運搬**を学ぶ卒業研究向けプロジェクトです。
2台がT字物体の周囲を回り込み、脚・足の接触で目標へ運搬します。
上位方策は共有actor・中央criticのMAPPO、歩行モデルは固定します。学習は報酬だけを使います。

## Google Colabで始める

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hashimoto-robotics-lab/hexapod-transport-rl/blob/main/notebooks/hexapod_transport_rl_colab.ipynb)

上の「Open in Colab」を開き、Googleアカウントでログインして、上から順に実行してください。
**GitHubアカウント・招待・アクセストークンは不要**です。
コード・ロボット形状・学習済みモデルは、Step 1で公開リポジトリから自動取得します。

まず研究の動機を読み、学習済み歩行モデルへ前進・横移動・旋回のコマンドを送って、動画で動きを確かめます。
歩行の操作は `WalkingSimulation` の `set_velocity()`・`run_for()`・`stop()` に統一し、1台と4台の例を用意しています。
続いて、2台へのコマンドを決める運搬方策を学習します。4台の協調運搬学習は、今後の拡張課題です。
上から順に、準備、歩行体験、API確認、参考の運搬モデルの再生、学習、検証用モデル選定、独立したテスト、
学習曲線・評価表、動画、結果ZIPの保存を実行します。CPUランタイムを使用できます。

- `MODE="quick"`：短い学習で全工程の接続を確認。卒論の性能評価には使いません。
- `MODE="research"`：チームステップ数を指定して本学習。
- `START_FROM_SCRATCH=True`：上位の押す方策もランダムな重みから学習。
- `START_FROM_SCRATCH=False`：報酬で学習済みの押す方策を使い、回り込みと引き継ぎ動作を学習。

どちらでも歩行方策は固定します。デモ行動や模倣損失は使いません。
詳しい進め方と保存・再開は [Colabガイド](docs/colab-guide.md)、実装は [コードガイド](docs/code-guide.md) を参照してください。

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
runs/                                    実験ログ・動画（配布には不要）
```

```bash
uv run ruff check .
uv run ruff format --check src tests tools
MUJOCO_GL=egl uv run pytest -q
uv run python tools/build_colab.py
```

実行手順を変更した場合は最後のコマンドでノートブックを再生成してください。
ソース・資産・モデルの変更はリポジトリにcommitして共有します。学生は取得時のコミットSHAを実験結果に記録できます。
`runs/` と仮想環境はGitで共有しません。
