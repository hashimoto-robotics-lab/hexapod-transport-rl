# 六足ロボットで円柱を押す強化学習

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hashimoto-robotics-lab/hexapod-transport-rl/blob/main/cylinder_pushing.ipynb)

TorchRLのPPOで、六足ロボット1台が円柱を目標まで押す動作を学習します。
観測・行動・報酬を変更し、学習結果を比較できます。
**方向を選ぶPPOはランダムな重みから学習し、18関節を動かす歩行モデルは固定して使います。**

## 進め方

1. [cylinder_pushing.ipynbをColabで開く](https://colab.research.google.com/github/hashimoto-robotics-lab/hexapod-transport-rl/blob/main/cylinder_pushing.ipynb)。
2. ランタイムを **T4 GPU / ランタイムバージョン2026.07** に変更する。
3. Step 0〜7でAPIを動かし、比較実験の結果を確認する。ここまではPPOの学習を実行しません。
4. Step 8で計画を決め、Step 9で標準条件を学習・評価する。
5. 報酬変更、複数seed、独自報酬、観測・行動の比較へ進み、Step 13で考察と結果を保存する。

使い方・用語・API一覧はノートブックにまとめています。
最初のセルがGitHubからコードと実験データを取得し、実行環境を準備します。更新版は新しいランタイムでStep 0から実行してください。

`PROFILE="research"`が本学習、`"demo"`は短縮した動作確認です。短縮実行は性能比較には使いません。
報酬変更などの追加学習は、各セルの`RUN_...`を`True`にすると実行します。
標準条件しか実行していない場合は、比較は未実施として報告します。

## ファイル構成

```text
cylinder_pushing.ipynb     # 説明・実験・API一覧
colab_setup.py             # インストールと描画の準備
pyproject.toml             # 必要なライブラリとパッケージ設定
src/hexapod_transport_rl/  # API・PPO・物理・機体・固定歩行モデル
results/                  # 比較実験の表・図・動画・考察
tests/                   # APIと教材の動作確認（ノートブックの実行には不要）
```

実験を実行すると`runs/`に結果を保存します。Colab終了前にノートブック末尾の保存セルで持ち帰ってください。
