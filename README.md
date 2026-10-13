# Hexapod Transport RL

複数の六足ロボットの協調動作を、TorchRLのMAPPOで調べる卒業研究向け教材です。
導入課題は **衝突を避けてすれ違う**、発展課題は **Tを協調して押す** です。
固定した歩行モデルへ指令を送り、上位方策を毎回ランダムな重みから学習します。教師デモ・模倣学習は使いません。

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hashimoto-robotics-lab/hexapod-transport-rl/blob/main/notebooks/cooperative_navigation_colab.ipynb)

ColabでT4 GPUを選び、上から実行します。公開GitHubなので認証は不要です。
教材は **歩行コマンド → Gym API → 報酬 → 新規学習 → 評価・動画** の順です。
1つの報酬条件ごとに、初回の準備を除いて10分以内の実験ループを目指します。

```python
from hexapod_transport_rl import NavigationConfig, train_mappo, evaluate_mappo

config = NavigationConfig(num_robots=2)
checkpoint = train_mappo(config, "runs/crossing_01", max_seconds=240)
report = evaluate_mappo(checkpoint, episodes=8, seed=98000)
```

学習は環境構築を含め240秒を目安に、ロールアウトの境界で終了します。
評価と動画1本を別に計測します。初回コンパイルやGPUの混雑によって時間は変わり、厳密な10分の保証ではありません。
保存する初期重み・Adam・価値統計は新規です。前の条件の重みを次へ渡しません。

## 導入課題：協調すれ違い

2台が向かい合って出発し、相手を避けながら反対側の目的地へ進みます。
観測は機体座標の目標・速度・他機の状態、行動は前進・横移動・旋回の正規化コマンドです。
目的地30 cm以内で全機が0.8秒留まると成功です。報酬で接近と安全な間隔の重視を変えられます。

手元のA6000では1学習seedで **学習・16例の評価・動画まで約242秒、接触なしの成功16/16** でした。
初期配置の向きと小さい揺らぎを変えた範囲の確認です。T4・4台・複数学習seedの性能は未確認です。
[課題・観測・報酬の説明](docs/cooperative-navigation.md)と[両試行の実測](docs/results/navigation_experiments_20261013.json)を参照してください。

## 発展課題：T押し

[運搬用Colab](https://colab.research.google.com/github/hashimoto-robotics-lab/hexapod-transport-rl/blob/main/notebooks/hexapod_transport_rl_colab.ipynb)は別に残しています。

| 項目 | 2台の場合 |
|---|---|
| 環境 | `HexapodPushT-v0` |
| 観測 | 各機9成分：自分XY、TのXY・角度、自分の向き、相手の姿勢 |
| 行動 | `(2,2)`、世界座標のXY目標 [m] |
| 標準報酬 | `clip(coverage / 0.95, 0, 1)` |
| 成功 | 目標との面積重なり率95%超 |
| 初期配置 | T → ゴール → ロボット |

[公式Push-T](https://github.com/huggingface/gym-pusht)の観測・行動・面積報酬・成功判定に対応させています。
六足の3D物理・複数台・指定した初期配置と等長2棒は研究用の拡張です。[差分](docs/pusht-protocol.md)を記録しています。
元の機体・歩行・モーターを使い、脚・足の接触で押します。色は元の機体と同じ、床はグリッドです。

## 読む場所

- [導入用ColabとAPI](docs/cooperative-navigation.md)：すれ違い、報酬の比較、2〜4台。
- [運搬の実験手順](docs/colab-guide.md)：T押しの予算、繰り返し、卒論での比較。
- [API](docs/api.md)：歩行・XY目標・新規学習・評価。
- [コード](docs/code-guide.md)：学生が読む6ファイル。
- [実測と全試行](docs/results/compact_experiments_20261013.json)：時間と学習結果を分けて記録。

短い試験で実行できることと、95%の運搬を習得することは別です。成功する予算はまだ確立していません。
ローカルA6000の値をColab/T4の実測とは扱いません。4台のAPIは検証していますが、学習性能は未確認です。
以前の速度指令・姿勢誤差の課題は[資料](docs/archive/)と[ノートブック](notebooks/archive/)へ移しています。
旧課題の成功率と約42秒の追加学習は、新しい課題の実績ではありません。

## ローカル

Python 3.12〜3.13、uvを使います。`uv sync --locked`はCPU版PyTorchです。
CUDA版Pythonでは `uv pip install --python <Python> --no-sources -e ".[warp]"` を実行します。
GPUの構成は[こちら](docs/warp-training.md)、元の機体・歩行資産の出典は`src/hexapod_transport_rl/assets/manifest.json`です。

学生用の運搬ではXY目標を0.8秒保ちます（`high_level_decimation=20`）。標準Gym設定は0.2秒です。
歩行25 Hz・物理200 Hzは同じで、記録する動画は実際の25 Hzフレームです。
