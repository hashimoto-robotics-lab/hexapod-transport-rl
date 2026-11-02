# 六足ロボットで円柱を押す強化学習教材

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hashimoto-robotics-lab/hexapod-transport-rl/blob/main/cylinder_pushing.ipynb)

TorchRLのPPOで、六足ロボット1台が円柱を目標まで押す動作を学習します。
観測・行動・報酬の設定から、学習・比較・考察まで進める教材です。
**方向を選ぶPPOはランダムな重みから学習し、18関節を動かす歩行モデルは固定して使います。**

## 進め方

1. [cylinder_pushing.ipynbをColabで開く](https://colab.research.google.com/github/hashimoto-robotics-lab/hexapod-transport-rl/blob/main/cylinder_pushing.ipynb)。
2. ランタイムを **T4 GPU / ランタイムバージョン2026.07** に変更する。
3. Step 0〜7でAPIを動かし、収録した比較実験を読む。ここまではPPOの学習を実行しません。
4. Step 8で計画を決め、Step 9で標準条件を学習・評価する。
5. 報酬変更、複数seed、独自報酬、観測・行動の比較へ進み、Step 13で考察と結果を保存する。

使い方・用語・API一覧はノートブックにまとめています。
最初のセルがGitHubから教材を取得して準備します。更新版は新しいランタイムでStep 0から実行してください。

`PROFILE="research"`が本学習、`"demo"`は短縮した動作確認です。短縮実行は性能比較には使いません。
報酬変更などの追加学習は、各セルの`RUN_...`を`True`にすると実行します。
標準条件しか実行していない場合は、比較は未実施として報告します。

## ファイル構成

```text
cylinder_pushing.ipynb     # 最初に開く教材。説明・実験・API一覧
colab_setup.py             # インストールと描画の準備
pyproject.toml             # 必要なライブラリとパッケージ設定
src/hexapod_transport_rl/  # API・PPO・物理・機体・固定歩行モデル
results/                  # 教材が表示する表・図・動画・考察
tests/                   # APIと教材の動作確認（ノートブックの実行には不要）
```

実験を実行すると`runs/`に結果を保存します。Colab終了前にノートブック末尾の保存セルで持ち帰ってください。

## 収録した比較実験

乱数の初期値（学習seed）を変えて各条件を3回学習し、各モデルを共通の1,024配置で評価しました。
標準報酬の成功率は平均98.80%、目標への進捗報酬の重みだけを20から40にした条件は平均91.31%でした。
標準1回の本学習はRTX A6000で平均約38分です。Colab T4の所要時間は未測定です。

教材が使う集計と表示用ファイルを`results/`に同梱しています。
詳細な評価記録・学習済みPPO・実験の再実行用スクリプトは、[整理前の実験記録](https://github.com/hashimoto-robotics-lab/hexapod-transport-rl/tree/f85f3d754f18fe1efe57d724ffc23697e626c09f/results/reward_study)と[当時のリポジトリ](https://github.com/hashimoto-robotics-lab/hexapod-transport-rl/tree/f85f3d754f18fe1efe57d724ffc23697e626c09f)から参照できます。
