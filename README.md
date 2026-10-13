# Hexapod Transport RL

六足ロボットによる協調運搬を、TorchRLのMAPPOで調べる卒業研究向け教材です。
固定した歩行モデルへ指令を送り、複数台が脚・足でTを押す上位方策を学びます。教師デモ・模倣損失は使いません。

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hashimoto-robotics-lab/hexapod-transport-rl/blob/main/notebooks/hexapod_transport_rl_colab.ipynb)

Google ColabでT4 GPUを選び、上から実行してください。公開GitHubから取得するので認証は不要です。
歩行コマンド → Gym API → 報酬 → TorchRLの収集・GAE・更新 → 評価・mediapy動画 → ZIP保存の順です。
コードは普通のimportとPythonセルです。GPUではMuJoCo Warpで物理・固定歩行・学習・面積計算を行います。

## 現在のPush-T仕様

[公式gym-pusht](https://github.com/huggingface/gym-pusht)の観測・行動・報酬・成功判定に近づけています。
六足の3D物理、T→ゴール→ロボットの開始配置、等長2棒のTは研究用の拡張です。
公式の2D環境と同じベンチマークではありません。[対応表](docs/pusht-protocol.md)に差分を記録しています。

| 項目 | 2台の場合 |
|---|---|
| 環境 | `HexapodPushT-v0` |
| 状態観測 | 各機9成分：公式の5成分＋自分の向き＋相手の姿勢 |
| キーポイント観測 | 各機22成分：Tの8頂点XY＋自分XY＋向き＋相手の姿勢 |
| 行動 | `(2,2)`、世界座標のXY目標 [m] |
| 目標 | 固定位置・固定向き |
| 標準報酬 | `clip(coverage / 0.95, 0, 1)` |
| 成功 | 目標との面積重なり率95%超で直ちに成功 |
| 物理 | 元の六足・モーター・歩行モデル、押し具なし、接触でTが動く |

横棒・縦棒は2台用でともに1.3 m、棒の太さは0.20 mです。
ロボットの色は元の機体と同じで、床にはグリッドがあります。
XY目標は位置追従で歩行指令へ変換します。Tを見て経路を決める制御器や教師行動はありません。
`[0,0]`は原点への移動です。その場に留めるには現在位置を指定します。

```python
import gymnasium as gym
import numpy as np
import hexapod_transport_rl
from hexapod_transport_rl import PushTConfig

env = gym.make("HexapodPushT-v0", config=PushTConfig())
observation, info = env.reset(seed=42)
targets = np.asarray(info["agent_pos"], dtype=np.float32)
observation, reward, terminated, truncated, info = env.step(targets)
print(info["coverage"], info["reward_terms"])
env.close()
```

## 報酬比較と学習の実績

公式の面積報酬だけを基準条件にし、成功時報酬・時間・接触などを追加する研究条件と区別します。
正の報酬と成功時の早期終了をRLで組み合わせると、成功直前の停滞が有利になる場合があります。
学生用Colabでは基準条件と成功時報酬を追加する条件を、同じ初期重み・物理・予算で比べます。

新仕様の初期試験は、各条件1,048,576チームステップ、ローカルRTX A6000で約2.5分でした。
CPU・Warpとも開発24配置の成功は各0/24で、運搬獲得はまだ確認できていません。
これは1学習seedの短い試験です。Colab/T4の時間や、成功する学習予算ではありません。
[全設定・試行・検証](docs/results/pusht_protocol_20261013.json)を保存します。

新しいモデルの保存形式は `hexapod-pusht-torchrl-mappo-v1` です。
旧課題の観測・速度指令モデルは新しいXY行動へ流用しません。旧課題の成功率と約42秒の追加学習も別の実績です。
[旧課題の追加学習Colab](https://colab.research.google.com/github/hashimoto-robotics-lab/hexapod-transport-rl/blob/main/notebooks/hexapod_transport_rl_pose_legacy.ipynb)と
[旧課題の新規学習Colab](https://colab.research.google.com/github/hashimoto-robotics-lab/hexapod-transport-rl/blob/main/notebooks/hexapod_transport_rl_from_scratch.ipynb)に保存しています。

## 読む場所

- [公式との対応](docs/pusht-protocol.md)：観測・XY行動・面積・成功・研究用の追加項。
- [学生用Colabの手順](docs/colab-guide.md)：比較条件・実験計画・保存。
- [APIガイド](docs/api.md)：Gym API、歩行、TorchRL、評価。
- [コードガイド](docs/code-guide.md)：環境・報酬・学習接続を読む順番。
- [GPUの構成](docs/warp-training.md)：MuJoCo Warp、接触容量、実測の区別。

歩行は1〜4台、押す環境は2〜4台です。4台のAPI更新・保存・読み込みは検証していますが、運搬の学習性能は未確認です。
平坦床・障害物なしで、シミュレータの状態を観測します。実機の認識・通信誤差は含みません。

## ローカル

Python 3.12〜3.13とuvを使います。機体と歩行モデルはパッケージに同梱しています。

```bash
git clone https://github.com/hashimoto-robotics-lab/hexapod-transport-rl.git
cd hexapod-transport-rl
uv sync --locked
MUJOCO_GL=egl uv run pytest -q
```

`uv sync --locked`はCPU版PyTorchです。Colabの準備セルはインストール済みのCUDA版PyTorchを維持します。
ローカルGPUでは `uv pip install --python <CUDA版Python> --no-sources -e ".[warp]"` を実行します。
元の機体・歩行資産の出典は `src/hexapod_transport_rl/assets/manifest.json`、旧運搬モデルは `checkpoints/manifest.json` にあります。
