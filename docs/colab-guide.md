# Colabで協調Push-Tを調べる

[学生用Colab](https://colab.research.google.com/github/hashimoto-robotics-lab/hexapod-transport-rl/blob/main/notebooks/hexapod_transport_rl_colab.ipynb)でT4 GPUを選び、上から実行します。
公開GitHubから取得するので認証は不要です。学生の編集を上書きしないため、更新後は新規ランタイムを使います。
研究の動機 → 歩行コマンド → Push-TのGym API → 報酬 → TorchRL MAPPO → 評価・動画・保存の順です。
普通のimportとセルを使い、動画はmediapyで表示します。

## 課題の基準

公式Push-Tの状態版・XY目標位置・面積重なり報酬・95%超の成功判定へ合わせています。
六足の3D物理・T→ゴール→ロボットの配置・等長2棒・5 Hzは研究用の拡張です。
[対応表と差分](pusht-protocol.md)を確認してください。教師デモ・模倣損失は使いません。

標準の`PushTConfig(obs_type="state")`は2台で各機9成分です。先頭5成分が公式の状態に対応します。
行動は`(2,2)`の世界XY目標 [m] です。`[0,0]`を停止と解釈せず、現在位置を指定して保持します。
目標は固定です。成功に1秒静止を要求しません。六足の転倒・領域外は失敗です。

`PushTRewardWeights()`は面積重なり報酬だけです。研究条件で`success=200`を追加します。
正の報酬と早期終了の組み合わせが、成功直前の停滞を生む可能性を調べる比較です。
時間・接触のコストも追加できます。変更する項と仮説を先に決めます。
総報酬の大小で性能を比較せず、成功率・最大重なり率・失敗込み時間・接触・転倒を使います。

## 学習と評価

旧課題の運搬モデルは新しい観測・行動へ流用しません。固定歩行モデル以外はランダムな重みから学びます。
全条件で同じ初期seed・物理・配置・学習量・MAPPO設定を使い、前の条件の重みを次へ渡しません。
GPUの初期予算は1,048,576チームステップ、1024世界・horizon 64・ミニバッチ1024・4 epochsです。
学習中は描画せず、物理・歩行・収集・更新・面積計算はGPUで行います。
CPU設定はAPI確認用です。新規学習の予算は成功を保証する値ではありません。

最初は1条件の学習曲線・重なり率・時間を見て、予算を決めてから全条件を揃えます。
Colab/T4の時間は実際の`progress.csv`で測ります。初回コンパイル・評価・動画・インストールは別です。
GPUの並列評価は`evaluate_pose(..., backend="warp")`、最終確認は`backend="cpu"`でも行います。
評価関数は保存形式から新仕様を判別し、同じ初期配置をCPU・GPUへ渡します。

標準の12配置は開発用です。卒論では設定を固定してから、学習seed3種類以上・未使用50配置以上で評価します。
最大重なり率が上がっても95%を超えなければ成功とは数えません。失敗や停滞も保存します。
観測の`state`と`keypoints`の比較は別実験にし、初期重み・予算・物理を揃えます。
4台は`num_robots=4`で新しく学習します。2台モデルをそのまま使いません。

## 保存

`create_experiment()`はコード・資産・版・SHA256を記録します。新しい実験名を使って上書きを避けます。
条件ごとに`run.json`・`initial.pt`・`progress.csv`・最後の`pusht.pt`を保存します。
新しい保存形式は`hexapod-pusht-torchrl-mappo-v1`で、XY行動範囲・観測・重み・Adam・価値統計も入ります。
評価JSON・CSV・図・動画と一緒にZIPを作り、ランタイムを削除する前に保存します。

旧課題の約42秒の追加学習と成功率は、[旧Colab](https://colab.research.google.com/github/hashimoto-robotics-lab/hexapod-transport-rl/blob/main/notebooks/hexapod_transport_rl_pose_legacy.ipynb)へ分けています。
新仕様の学習時間・精度としては扱いません。旧課題を最初から学ぶノートブックも保存しています。
